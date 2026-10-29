"""XRT-1 / INV-2 (Task BR): llama.cpp llama-server memory census and OOM boundary on one host.

Pinned build and model: configs/xrt/llamacpp_pin.yaml (llama.cpp b11559, SYCL and Vulkan Windows
zips; Qwen/Qwen3-4B-GGUF Q4_K_M). Flag syntax below was read from common/arg.cpp at that tag.

Modes (each launches llama-server.exe as a fresh child process per combination / trial):

census   E1 load-only buffer census. For every ctx x ubatch x kv x fa combination: start
         llama-server with --fit off --no-warmup, wait until /health returns 200 (or the log says
         "model loaded"), the process exits, or --timeout-s; then stop it and parse the log for
         model / KV / compute / output buffer sizes per device. A quantized V cache with -fa off
         is refused by llama.cpp at load ("quantized V cache requires flash_attn"); that combo is
         recorded as status "refused", not as a crash.
boundary E2 real OOM boundary. Bisection over n_ctx on [--low, --high] at --resolution. Each trial
         starts a fresh server with -c n, waits ready, then POSTs /completion with a token-id
         prompt of exactly n - 64 tokens (tiled from one /tokenize call on a fixed ASCII
         paragraph; ids cached across trials), n_predict 1, cache_prompt false, temperature 0.
         hold = HTTP 200, no "error" key, prompt_n (or tokens_evaluated) >= n - 64. The bisection
         mirrors tools/boot4_session.py bisect_holds: low must hold, otherwise the cell reports
         "below low rung"; then high; then grid midpoints.

Child env = os.environ + ZES_ENABLE_SYSMAN=1. Every row is appended to --out-jsonl; the run ends
with one line "XRT_DONE {json}". Logs are never overwritten (a run refuses if a log path for its
label already exists). No silent retries: a failed trial is a row, a harness error aborts (exit 2).
Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

KV_TYPES = ("f16", "q8_0", "q4_0")
FA_VALUES = ("on", "off")
PROMPT_MARGIN = 64
LOG_TAIL_LINES = 40

# Fixed ASCII paragraph; tokenized once per boundary run, ids tiled to the prompt length.
PARAGRAPH = (
    "The survey team walked the river valley at dawn, counting stones along the old wall and "
    "writing each number in a small notebook. By noon they had mapped the mill, the bridge and "
    "the two farms on the north bank, and they rested under the oak trees while the cart was "
    "loaded. In the afternoon the wind turned cold, so they measured the last field quickly, "
    "checked every page twice, and carried the instruments back to the station before dark. "
)

# llama.cpp b11559 buffer lines, e.g.
#   load_tensors:        SYCL0 model buffer size =  2000.00 MiB
#   llama_kv_cache:      SYCL0 KV buffer size =   576.00 MiB
#   llama_context:       SYCL0 compute buffer size =   300.75 MiB
#   llama_context:   SYCL_Host  output buffer size =     0.58 MiB
_BUFFER_RE = re.compile(
    r"(?P<dev>\S+)\s+(?P<kind>model|KV|compute|output|RS)\s+buffer size\s*=\s*"
    r"(?P<mib>[0-9]+(?:\.[0-9]+)?)\s*MiB"
)
_KIND = {"model": "model", "KV": "kv", "compute": "compute", "output": "output", "RS": "rs"}
_REFUSAL_RE = re.compile(
    r"quantized V cache requires flash_attn|V cache quantization requires flash_attn"
    r"|quantized V cache was requested, but this requires Flash Attention",
    re.IGNORECASE,
)
_ERROR_RE = re.compile(
    r"error|failed|exception|out of memory|abort|UR_RESULT_ERROR|ErrorOutOfDeviceMemory",
    re.IGNORECASE,
)


class ServerHandle(Protocol):
    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...


@dataclass
class HttpResult:
    status: int  # HTTP status; 0 = connection error; -1 = client timeout
    body: Any  # parsed JSON or None
    text: str


Launcher = Callable[[list[str], dict[str, str], Path], ServerHandle]
HttpFn = Callable[[str, str, Any, float], HttpResult]


def popen_launcher(cmd: list[str], env: dict[str, str], log_path: Path) -> ServerHandle:
    """Start the server with stdout+stderr to log_path. Refuses to overwrite an existing log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("xb") as fh:
        return subprocess.Popen(
            cmd, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env
        )


def urllib_http(method: str, url: str, body: Any, timeout_s: float) -> HttpResult:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout_s) as resp:
            status = int(resp.status)
            text = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        text = exc.read().decode("utf-8", "replace")
    except TimeoutError as exc:
        return HttpResult(-1, None, f"timeout: {exc}")
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            return HttpResult(-1, None, f"timeout: {exc.reason}")
        return HttpResult(0, None, f"URLError: {exc.reason}")
    except OSError as exc:
        return HttpResult(0, None, f"{type(exc).__name__}: {exc}")
    try:
        parsed: Any = json.loads(text)
    except ValueError:
        parsed = None
    return HttpResult(status, parsed, text)


def parse_tasklist_pids(text: str, image: str = "llama-server.exe") -> list[int]:
    """PIDs from `tasklist /FO CSV /NH` output for one image name."""
    pids: list[int] = []
    for line in text.splitlines():
        cols = [c.strip().strip('"') for c in line.split('","')]
        if len(cols) >= 2 and cols[0].lower() == image.lower() and cols[1].isdigit():
            pids.append(int(cols[1]))
    return pids


def llama_server_pids() -> list[int]:
    """Running llama-server.exe PIDs (Windows only; [] elsewhere or if tasklist fails)."""
    if sys.platform != "win32":
        return []
    try:
        out = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq llama-server.exe", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=15, check=False,
        )  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return []
    return parse_tasklist_pids(out.stdout)


@dataclass
class Runtime:
    launcher: Launcher = popen_launcher
    http: HttpFn = urllib_http
    server_pids: Callable[[], list[int]] = llama_server_pids
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    poll_s: float = 0.5
    settle_s: float = 2.0
    env_base: Mapping[str, str] = field(default_factory=lambda: dict(os.environ))


# ---------------------------------------------------------------- pure helpers


def parse_buffers(log_text: str) -> dict[str, dict[str, float]]:
    """{kind: {device: MiB}} for model / kv / compute / output / rs buffer lines.

    A device that appears more than once for one kind (e.g. two KV caches) is summed.
    """
    out: dict[str, dict[str, float]] = {}
    for m in _BUFFER_RE.finditer(log_text):
        kind = _KIND[m.group("kind")]
        dev = m.group("dev")
        per = out.setdefault(kind, {})
        per[dev] = round(per.get(dev, 0.0) + float(m.group("mib")), 4)
    return out


def find_refusal(log_text: str) -> str | None:
    for line in log_text.splitlines():
        if _REFUSAL_RE.search(line):
            return line.strip()
    return None


def error_lines(log_text: str, limit: int = 20) -> list[str]:
    found = [ln.strip() for ln in log_text.splitlines() if _ERROR_RE.search(ln)]
    return found[:limit]


def log_tail(log_text: str, n: int = LOG_TAIL_LINES) -> str:
    return "\n".join(log_text.splitlines()[-n:])


def batch_for(ubatch: int) -> int:
    return max(ubatch, 2048)


def build_server_cmd(
    *,
    server_exe: str,
    model: str,
    ctx: int,
    ubatch: int,
    kv: str,
    fa: str,
    ngl: int,
    port: int,
) -> list[str]:
    """llama-server argv (flag syntax verified in common/arg.cpp at b11559)."""
    if kv not in KV_TYPES:
        raise ValueError(f"kv must be one of {KV_TYPES}, got {kv!r}")
    if fa not in FA_VALUES:
        raise ValueError(f"fa must be one of {FA_VALUES}, got {fa!r}")
    if ctx <= 0 or ubatch <= 0:
        raise ValueError("ctx and ubatch must be > 0")
    return [
        server_exe,
        "-m", model,
        "-c", str(ctx),
        "-b", str(batch_for(ubatch)),
        "-ub", str(ubatch),
        "-ctk", kv,
        "-ctv", kv,
        "-fa", fa,
        "-ngl", str(ngl),
        "--fit", "off",
        # Task BR registration: one slot (no auto 4-slot split) and no host prompt cache
        # (default 8192 MiB of RAM on a 16 GB machine); both verified in arg.cpp at b11559.
        "-np", "1",
        "--cache-ram", "0",
        "--no-warmup",
        "--host", "127.0.0.1",
        "--port", str(port),
    ]  # fmt: skip


def build_child_env(base: Mapping[str, str]) -> dict[str, str]:
    env = dict(base)
    env["ZES_ENABLE_SYSMAN"] = "1"
    return env


def tile_tokens(ids: list[int], n: int) -> list[int]:
    """Repeat ids until exactly n tokens."""
    if n < 0:
        raise ValueError("n must be >= 0")
    if not ids:
        raise ValueError("ids must not be empty")
    reps = n // len(ids) + 1
    return (list(ids) * reps)[:n]


def grid_mid(lo: int, hi: int, resolution: int) -> int | None:
    """Same rule as tools/boot4_session.py grid_mid."""
    if resolution <= 0:
        raise ValueError("resolution must be > 0")
    if hi - lo <= resolution:
        return None
    snapped = ((lo + hi) // 2 // resolution) * resolution
    if snapped <= lo:
        snapped = lo + resolution
    if snapped >= hi:
        return None
    return snapped


def bisect_plan(
    *, low: int, high: int, resolution: int, attempt: Callable[[int], bool]
) -> dict[str, Any]:
    """Largest n on the grid that holds; mirrors boot4_session.bisect_holds.

    Each rung is attempted once. low is attempted first: if it fails the cell reports
    "below low rung". Then high: if it holds, the limit is above the search range.
    """
    if high <= low:
        raise ValueError("high must be > low")
    if resolution <= 0 or (high - low) % resolution != 0:
        raise ValueError("high - low must be a positive multiple of resolution")
    seen: set[int] = set()
    points: list[dict[str, Any]] = []

    def once(n: int) -> bool:
        if n in seen:
            raise RuntimeError(f"rung {n} already attempted")
        if (n - low) % resolution != 0:
            raise RuntimeError(f"rung {n} is off the {resolution} grid")
        seen.add(n)
        holds = bool(attempt(n))
        points.append({"n": n, "holds": holds})
        return holds

    if not once(low):
        return {
            "status": "below_low_rung",
            "note": "below low rung",
            "largest_hold": None,
            "first_fail": low,
            "failure_in_range": True,
            "points": points,
        }
    if once(high):
        return {
            "status": "above_range",
            "note": "high held; the limit is above the search range",
            "largest_hold": high,
            "first_fail": None,
            "failure_in_range": False,
            "points": points,
        }
    lo, hi = low, high
    while True:
        mid = grid_mid(lo, hi, resolution)
        if mid is None:
            break
        if once(mid):
            lo = mid
        else:
            hi = mid
    return {
        "status": "complete",
        "largest_hold": lo,
        "first_fail": hi,
        "failure_in_range": True,
        "points": points,
    }


def parse_int_list(text: str) -> list[int]:
    vals = [int(x) for x in text.split(",") if x.strip()]
    if not vals or any(v <= 0 for v in vals):
        raise ValueError(f"expected a comma list of positive ints, got {text!r}")
    return vals


def parse_choice_list(text: str, allowed: tuple[str, ...]) -> list[str]:
    vals = [x.strip() for x in text.split(",") if x.strip()]
    bad = [v for v in vals if v not in allowed]
    if not vals or bad:
        raise ValueError(f"expected a comma list from {allowed}, got {text!r}")
    return vals


# ---------------------------------------------------------------- server lifecycle


def _read_log(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_bytes().decode("utf-8", "replace")


def _base_url(port: int) -> str:
    return f"http://127.0.0.1:{port}"


def ensure_port_free(port: int, rt: Runtime) -> None:
    """Refuse only on a real HTTP response. Status 0 (refused / connection error) and -1 (client
    timeout; Windows retries a refused localhost connect for ~2 s) mean nothing answers: port free."""
    res = rt.http("GET", _base_url(port) + "/health", None, 2.0)
    pids = rt.server_pids()
    pid_note = (
        f"llama-server.exe running, PID(s) {', '.join(map(str, pids))}"
        if pids
        else "no llama-server.exe process found"
    )
    if res.status > 0:
        raise RuntimeError(
            f"port {port} already answers /health (HTTP {res.status}); {pid_note}. A stale server "
            "is running. Stop it before this run; refusing to measure against an unknown process."
        )
    if pids:
        print(
            f"note: port {port} gave no HTTP answer (status {res.status}: {res.text[:120]}), "
            f"treated as free; {pid_note}",
            file=sys.stderr, flush=True,
        )  # fmt: skip


def wait_ready(
    handle: ServerHandle, log_path: Path, port: int, timeout_s: float, t0: float, rt: Runtime
) -> str:
    """'ready' | 'exit' | 'timeout'."""
    while True:
        if handle.poll() is not None:
            return "exit"
        res = rt.http("GET", _base_url(port) + "/health", None, 5.0)
        if res.status == 200:
            return "ready"
        text = _read_log(log_path)
        if "model loaded" in text and "listening on" in text:
            return "ready"
        if rt.clock() - t0 >= timeout_s:
            return "timeout"
        rt.sleep(rt.poll_s)


def stop_server(handle: ServerHandle) -> int | None:
    """terminate, wait 10 s, kill, wait 30 s. Raises if the process survives the kill."""
    if handle.poll() is not None:
        return handle.poll()
    handle.terminate()
    try:
        return handle.wait(timeout=10)
    except subprocess.TimeoutExpired:
        handle.kill()
    try:
        return handle.wait(timeout=30)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("llama-server survived terminate + kill; aborting run") from exc


def _append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
        fh.flush()


# ---------------------------------------------------------------- census


def census_combos(
    ctx_list: list[int], ub_list: list[int], kv_list: list[str], fa_list: list[str]
) -> list[tuple[int, int, str, str]]:
    return [(c, u, k, f) for c in ctx_list for u in ub_list for k in kv_list for f in fa_list]


def census_log_path(log_dir: Path, label: str, idx: int, combo: tuple[int, int, str, str]) -> Path:
    c, u, k, f = combo
    return log_dir / f"{label}_census_{idx:03d}_c{c}_ub{u}_{k}_fa{f}.log"


def run_census(
    *,
    server_exe: str,
    model: str,
    ctx_list: list[int],
    ub_list: list[int],
    kv_list: list[str],
    fa_list: list[str],
    ngl: int,
    port: int,
    timeout_s: float,
    log_dir: Path,
    out_jsonl: Path,
    label: str,
    rt: Runtime,
) -> dict[str, Any]:
    combos = census_combos(ctx_list, ub_list, kv_list, fa_list)
    paths = [census_log_path(log_dir, label, i, c) for i, c in enumerate(combos)]
    existing = [str(p) for p in paths if p.exists()]
    if existing:
        raise RuntimeError(f"log path(s) already exist for label {label!r}: {existing[:3]}")
    env = build_child_env(rt.env_base)
    counts: dict[str, int] = {}
    for idx, (combo, log_path) in enumerate(zip(combos, paths, strict=True)):
        ctx, ub, kv, fa = combo
        cmd = build_server_cmd(
            server_exe=server_exe, model=model, ctx=ctx, ubatch=ub, kv=kv, fa=fa,
            ngl=ngl, port=port,
        )  # fmt: skip
        if idx > 0:
            rt.sleep(rt.settle_s)
        ensure_port_free(port, rt)
        t0 = rt.clock()
        handle = rt.launcher(cmd, env, log_path)
        try:
            status = wait_ready(handle, log_path, port, timeout_s, t0, rt)
        finally:
            returncode = stop_server(handle)
        wall_s = round(rt.clock() - t0, 3)
        text = _read_log(log_path)
        refusal = find_refusal(text)
        errs = error_lines(text)
        if status == "exit" and refusal is not None:
            status = "refused"
        error: str | None = None
        if status == "refused":
            error = refusal
        elif status != "ready":
            error = "\n".join(errs) if errs else log_tail(text)
        row = {
            "mode": "census",
            "label": label,
            "ctx": ctx,
            "ubatch": ub,
            "batch": batch_for(ub),
            "kv": kv,
            "fa": fa,
            "ngl": ngl,
            "status": status,
            "returncode": returncode,
            "buffers": parse_buffers(text),
            "error": error,
            "error_lines": errs,
            "wall_s": wall_s,
            "log_path": str(log_path),
            "cmd": cmd,
        }
        _append_row(out_jsonl, row)
        counts[status] = counts.get(status, 0) + 1
        print(
            f"XRT_ROW census {idx + 1}/{len(combos)} c={ctx} ub={ub} kv={kv} fa={fa} "
            f"status={status} wall_s={wall_s}",
            flush=True,
        )
    return {"mode": "census", "label": label, "n": len(combos), "counts": counts}


# ---------------------------------------------------------------- boundary


@dataclass
class BoundaryCell:
    server_exe: str
    model: str
    kv: str
    fa: str
    ubatch: int
    ngl: int
    port: int
    timeout_s: float
    slider_pct: int | None
    log_dir: Path
    out_jsonl: Path
    label: str
    token_ids: list[int] = field(default_factory=list)
    trial_idx: int = 0


def _tokenize(cell: BoundaryCell, rt: Runtime) -> list[int]:
    res = rt.http(
        "POST",
        _base_url(cell.port) + "/tokenize",
        {"content": PARAGRAPH, "add_special": False},
        60.0,
    )
    toks = res.body.get("tokens") if isinstance(res.body, dict) else None
    if res.status != 200 or not isinstance(toks, list) or not toks:
        raise RuntimeError(f"/tokenize failed: status {res.status}: {res.text[:400]}")
    if not all(isinstance(t, int) and not isinstance(t, bool) for t in toks):
        raise RuntimeError(f"/tokenize returned non-integer tokens: {toks[:8]}")
    return [int(t) for t in toks]


def run_trial(cell: BoundaryCell, n: int, phase: str, rt: Runtime) -> dict[str, Any]:
    """One fresh server at -c n, one prefill of n - 64 tokens. Returns the row (also appended)."""
    n_prompt = n - PROMPT_MARGIN
    if n_prompt <= 0:
        raise ValueError(f"n_ctx {n} leaves no prompt after the {PROMPT_MARGIN}-token margin")
    cmd = build_server_cmd(
        server_exe=cell.server_exe, model=cell.model, ctx=n, ubatch=cell.ubatch, kv=cell.kv,
        fa=cell.fa, ngl=cell.ngl, port=cell.port,
    )  # fmt: skip
    log_path = cell.log_dir / f"{cell.label}_bnd_{cell.trial_idx:03d}_n{n}.log"
    if log_path.exists():
        raise RuntimeError(f"log path already exists: {log_path}")
    if cell.trial_idx > 0:
        rt.sleep(rt.settle_s)
    cell.trial_idx += 1
    ensure_port_free(cell.port, rt)
    env = build_child_env(rt.env_base)
    t0 = rt.clock()
    handle = rt.launcher(cmd, env, log_path)
    status = "fail_load"
    http_status: int | None = None
    prompt_ms: float | None = None
    prompt_n: int | None = None
    req_error: str | None = None
    try:
        ready = wait_ready(handle, log_path, cell.port, cell.timeout_s, t0, rt)
        if ready == "timeout":
            status = "timeout"
        elif ready == "ready":
            if not cell.token_ids:
                cell.token_ids = _tokenize(cell, rt)
            prompt = tile_tokens(cell.token_ids, n_prompt)
            remaining = max(1.0, cell.timeout_s - (rt.clock() - t0))
            res = rt.http(
                "POST",
                _base_url(cell.port) + "/completion",
                {"prompt": prompt, "n_predict": 1, "cache_prompt": False, "temperature": 0},
                remaining,
            )
            http_status = res.status
            body: dict[str, Any] = res.body if isinstance(res.body, dict) else {}
            raw_timings = body.get("timings")
            timings: dict[str, Any] = raw_timings if isinstance(raw_timings, dict) else {}
            pn = timings.get("prompt_n", body.get("tokens_evaluated"))
            prompt_n = int(pn) if isinstance(pn, int | float) else None
            pms = timings.get("prompt_ms")
            prompt_ms = float(pms) if isinstance(pms, int | float) else None
            if res.status == -1:
                status = "timeout"
                req_error = res.text
            elif (
                res.status == 200
                and "error" not in body
                and prompt_n is not None
                and prompt_n >= n_prompt
                and handle.poll() is None
            ):
                status = "hold"
            else:
                status = "fail_prefill"
                req_error = res.text[:2000]
    finally:
        returncode = stop_server(handle)
    wall_s = round(rt.clock() - t0, 3)
    text = _read_log(log_path)
    error: str | None = None
    if status != "hold":
        parts = [p for p in (req_error, log_tail(text)) if p]
        error = "\n".join(parts) if parts else None
    row = {
        "mode": "boundary",
        "label": cell.label,
        "phase": phase,
        "n_ctx": n,
        "n_prompt": n_prompt,
        "prompt_n": prompt_n,
        "kv": cell.kv,
        "fa": cell.fa,
        "ubatch": cell.ubatch,
        "batch": batch_for(cell.ubatch),
        "ngl": cell.ngl,
        "slider_pct": cell.slider_pct,
        "status": status,
        "http_status": http_status,
        "returncode": returncode,
        "error": error,
        "prompt_ms": prompt_ms,
        "buffers": parse_buffers(text),
        "wall_s": wall_s,
        "log_path": str(log_path),
        "cmd": cmd,
    }
    _append_row(cell.out_jsonl, row)
    print(f"XRT_ROW boundary {phase} n={n} status={status} wall_s={wall_s}", flush=True)
    return row


def run_boundary(
    cell: BoundaryCell, *, low: int, high: int, resolution: int, repeats: int, rt: Runtime
) -> dict[str, Any]:
    prefix = f"{cell.label}_bnd_"
    if cell.log_dir.exists() and any(p.name.startswith(prefix) for p in cell.log_dir.iterdir()):
        raise RuntimeError(
            f"boundary logs for label {cell.label!r} already exist in {cell.log_dir}"
        )
    result = bisect_plan(
        low=low,
        high=high,
        resolution=resolution,
        attempt=lambda n: run_trial(cell, n, "bisect", rt)["status"] == "hold",
    )
    repeat_rows: list[dict[str, Any]] = []
    targets: list[int] = []
    if result["largest_hold"] is not None:
        targets = [v for v in (result["largest_hold"], result["first_fail"]) if v is not None]
    for _ in range(repeats):
        for n in targets:
            row = run_trial(cell, n, "repeat", rt)
            repeat_rows.append({"n": n, "status": row["status"]})
    return {
        "mode": "boundary",
        "label": cell.label,
        "kv": cell.kv,
        "fa": cell.fa,
        "ubatch": cell.ubatch,
        "slider_pct": cell.slider_pct,
        "status": result["status"],
        "largest_hold": result["largest_hold"],
        "first_fail": result["first_fail"],
        "points": result["points"],
        "repeats": repeat_rows,
        "trials": cell.trial_idx,
    }


# ---------------------------------------------------------------- CLI


def main(argv: list[str] | None = None, rt: Runtime | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    sub = ap.add_subparsers(dest="mode", required=True)

    def common(p: argparse.ArgumentParser, timeout_default: float) -> None:
        p.add_argument("--server-exe", required=True)
        p.add_argument("--model", required=True)
        p.add_argument("--ngl", type=int, default=99)
        p.add_argument("--port", type=int, default=8091)
        p.add_argument("--timeout-s", type=float, default=timeout_default)
        p.add_argument("--log-dir", type=Path, required=True)
        p.add_argument("--out-jsonl", type=Path, required=True)
        p.add_argument("--label", required=True)

    pc = sub.add_parser("census", help="E1 load-only buffer census")
    common(pc, 300.0)
    pc.add_argument("--ctx-list", default="4096,16384,32768")
    pc.add_argument("--ubatch-list", default="128,256,512,1024,2048,4096")
    pc.add_argument("--kv-list", default="f16,q8_0,q4_0")
    pc.add_argument("--fa-list", default="on,off")

    pb = sub.add_parser("boundary", help="E2 OOM boundary with a prefill to depth")
    common(pb, 1800.0)
    pb.add_argument("--kv", required=True, choices=KV_TYPES)
    pb.add_argument("--fa", default="on", choices=FA_VALUES)
    pb.add_argument("--ubatch", type=int, default=512)
    pb.add_argument("--low", type=int, required=True)
    pb.add_argument("--high", type=int, required=True)
    pb.add_argument("--resolution", type=int, required=True)
    pb.add_argument("--slider-pct", type=int, default=None, help="recorded only")
    pb.add_argument("--repeats-at-boundary", type=int, default=0)

    args = ap.parse_args(argv)
    if args.mode == "boundary" and args.repeats_at_boundary < 0:
        ap.error("--repeats-at-boundary must be >= 0")
    runtime = rt if rt is not None else Runtime()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.label):
        print("XRT_ERROR label must match [A-Za-z0-9_.-]+", flush=True)
        return 2
    try:
        if args.mode == "census":
            summary = run_census(
                server_exe=args.server_exe,
                model=args.model,
                ctx_list=parse_int_list(args.ctx_list),
                ub_list=parse_int_list(args.ubatch_list),
                kv_list=parse_choice_list(args.kv_list, KV_TYPES),
                fa_list=parse_choice_list(args.fa_list, FA_VALUES),
                ngl=args.ngl,
                port=args.port,
                timeout_s=args.timeout_s,
                log_dir=args.log_dir,
                out_jsonl=args.out_jsonl,
                label=args.label,
                rt=runtime,
            )
        else:
            cell = BoundaryCell(
                server_exe=args.server_exe,
                model=args.model,
                kv=args.kv,
                fa=args.fa,
                ubatch=args.ubatch,
                ngl=args.ngl,
                port=args.port,
                timeout_s=args.timeout_s,
                slider_pct=args.slider_pct,
                log_dir=args.log_dir,
                out_jsonl=args.out_jsonl,
                label=args.label,
            )
            summary = run_boundary(
                cell,
                low=args.low,
                high=args.high,
                resolution=args.resolution,
                repeats=args.repeats_at_boundary,
                rt=runtime,
            )
    except (RuntimeError, ValueError) as exc:
        print(f"XRT_ERROR {type(exc).__name__}: {exc}", flush=True)
        return 2
    print("XRT_DONE " + json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
