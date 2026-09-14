"""Quality-degradation sweep runner.

Sweeps every probe in evaluation/probes/prompts.jsonl across a range of filler
context depths, with N repetitions per (probe, depth) cell. Uses
evaluation/probes/scorers.py for all scoring so there is one canonical scorer
shared between validation and the sweep.

Usage:
    py -3.11 -m harness.runner [options]
    py -3.11 -m harness.runner --probe-ids rea_01 str_01 --reps 2
    py -3.11 -m harness.runner --dry-run
    py -3.11 -m harness.runner --preflight-only
    py -3.11 -m harness.runner --deadline 6h30m
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from harness import cache, context, telemetry
from harness.context import DEFAULT_FILLER_MODE, FILLER_MODES

REPO_ROOT = Path(__file__).parent.parent
PROBES_DIR = REPO_ROOT / "evaluation" / "probes"
RESULTS_DIR = REPO_ROOT / "results"

CONTEXT_DEPTHS = [0, 2_000, 8_000, 16_000, 32_000, 64_000]
DEFAULT_MODEL = "qwen3:4b-instruct"
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_REPS = 5

# Fields required in every result row written to results/run_*.jsonl.
# memory_architecture distinguishes unified AI PC configs from discrete
# measurement hosts so they are never silently pooled in analysis.
REQUIRED_ROW_FIELDS = frozenset({
    "probe_id", "category", "depth", "rep", "filler_mode",
    "score", "config_hash", "hardware_config", "memory_architecture",
    "model", "model_variant", "thinking_enabled",
    "ctx_suspect", "position_in_cell",
    # Span instrumentation (Zachary category names, per-call)
    "orch_setup_ns", "http_client_ns", "tool_compute_ns",
})


def validate_result_row(row: dict) -> None:
    """Raise ValueError if a result row is missing required fields."""
    missing = REQUIRED_ROW_FIELDS - row.keys()
    if missing:
        raise ValueError(f"Result row missing required fields: {sorted(missing)}")


def _load_scorers():
    spec = importlib.util.spec_from_file_location("probes_scorers", PROBES_DIR / "scorers.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_outcome():
    """Load evaluation.outcome without triggering evaluation.__init__ (which imports openai)."""
    spec = importlib.util.spec_from_file_location(
        "evaluation_outcome",
        REPO_ROOT / "evaluation" / "outcome.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _config_hash(cfg: dict) -> str:
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True).encode()
    ).hexdigest()[:12]


def _make_count_fn(host: str, model: str):
    """Return a callable that counts filler tokens via a single-token generation.

    Sends the filler as a user message, requests num_predict=1, and reads
    prompt_eval_count from the done chunk. Subtracts an estimate of the chat
    template overhead (~8 tokens for qwen3 instruct) so the count reflects
    the filler text alone. Callers should treat the result as ±10 tokens
    accurate; the 2% filler tolerance absorbs this.
    """
    TEMPLATE_OVERHEAD = 8

    def count_fn(text: str) -> int:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": text}],
            "stream": True,
            "options": {"num_predict": 1, "temperature": 0},
        }
        with httpx.stream("POST", f"{host}/api/chat", json=payload, timeout=300) as r:
            r.raise_for_status()
            for raw in r.iter_lines():
                raw = raw.strip()
                if not raw:
                    continue
                c = json.loads(raw)
                if c.get("done"):
                    return max(0, c.get("prompt_eval_count", 0) - TEMPLATE_OVERHEAD)
        return 0

    return count_fn


def _call_ollama_streaming(
    model: str, prompt: str, max_tokens: int, host: str,
) -> tuple[str, float, float, int, int]:
    """Return (text, latency_ms, ttft_ms, tokens_in, tokens_out).

    Uses /api/chat so the model's chat template is applied. The composed
    filler+probe string is wrapped as a single user message. For the instruct
    model (qwen3:4b-instruct) no think flag is needed — it answers directly.
    """
    url = f"{host}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0,
        },
    }
    start = time.perf_counter()
    ttft_ms: float | None = None
    full_text = ""
    tokens_in = 0
    tokens_out = 0

    with httpx.stream("POST", url, json=payload, timeout=300) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines():
            raw = raw.strip()
            if not raw:
                continue
            chunk = json.loads(raw)
            token = chunk.get("message", {}).get("content", "")
            if token and ttft_ms is None:
                ttft_ms = (time.perf_counter() - start) * 1000
            full_text += token
            if chunk.get("done"):
                tokens_in = chunk.get("prompt_eval_count", 0)
                tokens_out = chunk.get("eval_count", 0)
                break

    latency_ms = (time.perf_counter() - start) * 1000
    return full_text, latency_ms, ttft_ms or latency_ms, tokens_in, tokens_out


def run_cell(
    probe: dict[str, Any],
    filler: str,
    depth: int,
    rep: int,
    position_in_cell: int,
    cell_probe_seed: int,
    model: str,
    host: str,
    cfg_hash: str,
    filler_mode: str,
    hardware_config: str,
    memory_architecture: str,
    model_variant: str,
    thinking_enabled: bool,
    scorers,
    outcome_mod,
) -> dict[str, Any]:
    # ORCH_SETUP: harness cost to assemble the full prompt (wrap_prompt).
    # CPU-bound; wall elapsed is a valid CPU proxy (no I/O).
    _t0 = time.perf_counter_ns()
    prompt = context.wrap_prompt(filler, probe["prompt"], filler_mode=filler_mode)
    orch_setup_ns = time.perf_counter_ns() - _t0

    max_tokens: int = probe["max_tokens"]
    params = {
        "max_tokens": max_tokens,
        "temperature": 0,
        "filler_mode": filler_mode,
        "model_variant": model_variant,
    }

    cached = cache.get(model, prompt, params)
    if cached:
        output = cached["output"]
        tel = telemetry.Telemetry.from_dict(cached["telemetry"])
    else:
        output, latency_ms, ttft_ms, tokens_in, tokens_out = _call_ollama_streaming(
            model, prompt, max_tokens, host,
        )
        gpu_val, gpu_source = telemetry.gpu_mem_mb(memory_architecture)
        tel = telemetry.Telemetry(
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            mem_rss_mb=telemetry.rss_mb(),
            gpu_mem_mb=gpu_val,
            gpu_mem_source=gpu_source,
        )
        cache.put(model, prompt, params, {"output": output, "telemetry": tel.to_dict()})

    # HTTP_CLIENT: Ollama inference wall time.  Recorded from _call_ollama_streaming
    # (or from cached telemetry on a cache hit — the cached latency is the original
    # measured value, so the span is historically accurate).
    # cpu_ns = 0: the thread blocks on network/GPU I/O, not user-space CPU.
    http_client_ns = int(tel.latency_ms * 1e6)

    # TOOL_COMPUTE: 0 — the quality sweep does not dispatch tool calls.
    tool_compute_ns = 0

    score_val, score_detail = scorers.score(probe, output)

    outcome_result = outcome_mod.classify(
        output=output,
        expected=probe.get("expected", ""),
        scorer_type=probe.get("scorer_type", ""),
        score=score_val,
        done_reason=None,  # runner path does not receive done_reason from Ollama
    )

    # Flag rows where context delivered is materially less than requested.
    # tokens_in includes filler + probe + chat template; at depth>0 the filler
    # dominates. A ratio < 0.9 almost certainly indicates a filler undershoot.
    ctx_suspect = depth > 0 and tel.tokens_in < depth * 0.9

    return {
        "probe_id": probe["id"],
        "category": probe["category"],
        "difficulty": probe["difficulty"],
        "depth": depth,
        "rep": rep,
        "position_in_cell": position_in_cell,
        "cell_probe_seed": cell_probe_seed,
        "filler_mode": filler_mode,
        "score": score_val,
        "score_detail": score_detail,
        "outcome_class": outcome_result["outcome_class"],
        "classification_method": outcome_result["classification_method"],
        "format_compliant": outcome_result["format_compliant"],
        "latency_ms": round(tel.latency_ms, 1),
        "ttft_ms": round(tel.ttft_ms, 1),
        "tokens_in": tel.tokens_in,
        "tokens_out": tel.tokens_out,
        "max_tokens": max_tokens,
        "ctx_suspect": ctx_suspect,
        "mem_rss_mb": round(tel.mem_rss_mb, 1),
        "gpu_mem_mb": None if tel.gpu_mem_mb is None else round(tel.gpu_mem_mb, 1),
        "gpu_mem_source": tel.gpu_mem_source,
        "config_hash": cfg_hash,
        "hardware_config": hardware_config,
        "memory_architecture": memory_architecture,
        "model": model,
        "model_variant": model_variant,
        "thinking_enabled": thinking_enabled,
        # Span instrumentation — Zachary's category names, per-call
        "orch_setup_ns":   orch_setup_ns,
        "http_client_ns":  http_client_ns,
        "tool_compute_ns": tool_compute_ns,
    }


# ── resume helper ─────────────────────────────────────────────────────────────

def _load_completed(path: Path, cfg_hash: str) -> set[tuple[int, int, str]]:
    """Return (depth, rep, probe_id) triples already written to *path*.

    File-size note: at the default sweep (44 probes × 6 depths × 5 reps) the
    resume file is ≈0.8 MB; at the largest plausible sweep ≈11 MB.  Loading
    the whole file with read_text() is fine at these sizes and lets us detect
    the last non-empty line without two-pass streaming.

    A malformed FINAL line is treated as a truncated write (the power-loss
    pattern: write + flush reached the OS buffer, fsync did not complete before
    power cut) and is discarded with a warning that names the file and line
    number.  A malformed line at any other position raises — that is real
    corruption, not a truncated write.
    """
    completed: set[tuple[int, int, str]] = set()
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    nonempty = [(i + 1, ln) for i, ln in enumerate(raw_lines) if ln.strip()]
    for idx, (lineno, raw) in enumerate(nonempty):
        is_last = idx == len(nonempty) - 1
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            if is_last:
                print(
                    f"WARNING: discarding truncated final line in {path} "
                    f"(line {lineno}): {exc}",
                    flush=True,
                )
                continue
            raise ValueError(
                f"Malformed JSON at {path}:{lineno} "
                f"(not the final line — this is corruption, not a truncated write): {exc}"
            ) from exc
        existing_hash = row.get("config_hash")
        if existing_hash and existing_hash != cfg_hash:
            raise ValueError(
                f"Config hash mismatch: resume file has {existing_hash!r}, "
                f"current run has {cfg_hash!r}. Check --depths/--reps/--model."
            )
        if "depth" in row and "rep" in row and "probe_id" in row:
            completed.add((row["depth"], row["rep"], row["probe_id"]))
    return completed


def _fsync_dir(d: Path) -> None:
    """Fsync the directory so a newly-created file's directory entry is durable.

    Required after creating a new result file: os.fsync on the file descriptor
    makes the file contents durable but does not guarantee the directory entry
    (the file's existence) survives a power loss before the directory page is
    written.  Silently no-ops on platforms that do not support opening a
    directory with O_RDONLY (e.g. Windows).
    """
    try:
        fd = os.open(str(d), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


# ── duration parsing ──────────────────────────────────────────────────────────

_DURATION_RE = re.compile(
    r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$"
)


def _parse_duration(s: str) -> float:
    """Parse a duration string such as '6h30m', '2h', '90s', '45m' into seconds.

    Raises ValueError on unrecognised format.
    """
    m = _DURATION_RE.fullmatch(s.strip())
    if m is None or not any(m.groups()):
        raise ValueError(f"Unrecognised duration format: {s!r}  (expected e.g. '6h30m', '45m', '90s')")
    h = int(m.group(1) or 0)
    mn = int(m.group(2) or 0)
    sc = int(m.group(3) or 0)
    total = h * 3600 + mn * 60 + sc
    if total <= 0:
        raise ValueError(f"Duration must be positive, got: {s!r}")
    return float(total)


def _deadline_exceeded(deadline_s: float | None, run_start: float) -> bool:
    """Return True if the wall-clock deadline has been reached."""
    if deadline_s is None:
        return False
    return time.monotonic() - run_start >= deadline_s


# ── git / system helpers ──────────────────────────────────────────────────────

def _git_state() -> dict[str, Any]:
    """Return {'sha': str|None, 'dirty': bool|None} for the repo at REPO_ROOT."""
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, timeout=5,
        ).decode().strip()
        dirty_out = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            stderr=subprocess.DEVNULL, timeout=5,
        ).decode().strip()
        return {"sha": sha, "dirty": bool(dirty_out)}
    except Exception:
        return {"sha": None, "dirty": None}


def _free_bytes_output() -> str | None:
    """Run 'free -b' and return its stdout; None if the command is unavailable."""
    try:
        return subprocess.check_output(
            ["free", "-b"], stderr=subprocess.DEVNULL, timeout=5,
        ).decode()
    except Exception:
        return None


# ── preflight ─────────────────────────────────────────────────────────────────

def _run_preflight(
    host: str,
    model: str,
    evalset_path: Path,
    probes: list[dict],
    min_disk_gb: float,
    min_mem_gb: float,
) -> list[str]:
    """Run preflight checks; return a list of failure messages (empty = OK).

    Checks (in order):
    1. Ollama server reachable and model loadable.
    2. Free disk space above min_disk_gb on RESULTS_DIR filesystem.
    3. Free physical memory above min_mem_gb.
    4. Evalset parsed without error (probes list already populated by caller).
    5. Git state recorded (informational only — never a failure).
    """
    failures: list[str] = []

    # 1 — model loadable
    try:
        resp = httpx.get(f"{host}/api/tags", timeout=10)
        resp.raise_for_status()
        tags = resp.json()
        names = [m.get("name", "") for m in tags.get("models", [])]
        if not any(model in n for n in names):
            failures.append(
                f"Model {model!r} not found in Ollama tags at {host}. "
                f"Available: {names[:5]}{'…' if len(names) > 5 else ''}"
            )
    except Exception as exc:
        failures.append(f"Ollama not reachable at {host}: {exc}")

    # 2 — free disk
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        free_gb = shutil.disk_usage(RESULTS_DIR).free / 1024 ** 3
        if free_gb < min_disk_gb:
            failures.append(
                f"Free disk on {RESULTS_DIR}: {free_gb:.2f} GB < required {min_disk_gb:.2f} GB"
            )
    except Exception as exc:
        failures.append(f"Could not check disk space: {exc}")

    # 3 — free memory
    try:
        import psutil
        free_mb = psutil.virtual_memory().available / 1024 / 1024
        min_mb = min_mem_gb * 1024
        if free_mb < min_mb:
            failures.append(
                f"Free memory: {free_mb:.0f} MB < required {min_mb:.0f} MB"
            )
    except ImportError:
        pass  # psutil absent — skip silently
    except Exception as exc:
        failures.append(f"Could not check free memory: {exc}")

    # 4 — evalset validated (probes already loaded by caller; empty = suspicious)
    if not probes:
        failures.append(
            f"Evalset at {evalset_path} produced zero probes after filtering. "
            "Check --evalset path and --probe-ids filter."
        )

    return failures


# ── background sampler ────────────────────────────────────────────────────────

def _start_sampler(path: Path, run_start: float) -> threading.Event:
    """Start a daemon thread writing CPU temp / clocks / free memory every 10 s.

    Returns a threading.Event; set it to stop the thread.
    Writes a CSV with columns:
        timestamp_utc, elapsed_s, cpu_pkg_temp_c, cpu_freq_mhz_mean, mem_free_mb
    """
    stop_event = threading.Event()

    def _loop() -> None:
        header_written = path.exists()
        while not stop_event.wait(10):
            now = datetime.now(timezone.utc).isoformat()
            elapsed = round(time.monotonic() - run_start, 1)
            pkg_temp: float | None = None
            freq_mean: float | None = None
            mem_free: float | None = None
            try:
                import psutil
                # CPU package temperature (Linux; empty dict on Windows)
                temps = psutil.sensors_temperatures()
                for key in ("coretemp", "k10temp", "acpitz", "cpu_thermal"):
                    if key in temps:
                        entries = temps[key]
                        pkg_candidates = [e.current for e in entries if "package" in e.label.lower()]
                        if pkg_candidates:
                            pkg_temp = round(pkg_candidates[0], 1)
                            break
                        elif entries:
                            pkg_temp = round(entries[0].current, 1)
                            break
                # Per-core clocks
                freqs = psutil.cpu_freq(percpu=True)
                if freqs:
                    freq_mean = round(sum(f.current for f in freqs) / len(freqs), 1)
                # Free memory
                mem_free = round(psutil.virtual_memory().available / 1024 / 1024, 1)
            except Exception:
                pass

            row = [now, elapsed, pkg_temp, freq_mean, mem_free]
            try:
                with open(path, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    if not header_written:
                        w.writerow(["timestamp_utc", "elapsed_s",
                                    "cpu_pkg_temp_c", "cpu_freq_mhz_mean", "mem_free_mb"])
                        header_written = True
                    w.writerow(row)
            except Exception:
                pass

    t = threading.Thread(target=_loop, daemon=True, name="telemetry-sampler")
    t.start()
    return stop_event


# ── manifest / summary helpers ────────────────────────────────────────────────

def _write_json_atomic(path: Path, data: dict, no_fsync: bool) -> None:
    """Write *data* as JSON to *path*, flushing+fsyncing unless --no-fsync."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.flush()
        if not no_fsync:
            try:
                os.fsync(f.fileno())
            except OSError:
                pass


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Quality-degradation context sweep")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument(
        "--depths", nargs="+", type=int, default=CONTEXT_DEPTHS,
        metavar="N", help="Token depths for filler (default: 0 2000 8000 16000 32000 64000)",
    )
    parser.add_argument(
        "--probe-ids", nargs="*", metavar="ID",
        help="Run only these probe IDs; default runs all",
    )
    parser.add_argument(
        "--evalset", default=str(PROBES_DIR / "prompts.jsonl"),
        help="Path to prompts.jsonl",
    )
    parser.add_argument(
        "--skip-judge", action="store_true", default=True,
        help="Skip judge-scored probes (default: True)",
    )
    parser.add_argument(
        "--filler-mode", default=DEFAULT_FILLER_MODE, choices=FILLER_MODES,
        help=(
            "unlabelled (default): filler prepended with no framing, measures "
            "context degradation. labelled: filler in <background_context> tags "
            "with ignore instruction, measures instruction-following instead."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print sweep plan and exit without calling the model",
    )
    parser.add_argument(
        "--hardware-config", default="unknown",
        metavar="NAME",
        help=(
            "Name of the hardware config running this sweep "
            "(e.g. blade14_rtx4070, strix_halo_64gb). Written to every result row "
            "so runs from different hosts cannot be silently pooled."
        ),
    )
    parser.add_argument(
        "--memory-architecture", default="unknown",
        choices=["unified", "discrete", "unknown"],
        help=(
            "Memory architecture of the host: 'unified' (AI PC, weights and KV "
            "share one pool) or 'discrete' (separate VRAM). Written to every "
            "result row. Quality-vs-depth results transfer across architectures; "
            "TTFT and throughput do not."
        ),
    )
    parser.add_argument(
        "--model-variant", default=None, metavar="VARIANT",
        choices=["instruct", "reasoning"],
        help=(
            "Model variant: 'instruct' (answers directly, no chain-of-thought) or "
            "'reasoning' (thinking model). Auto-detected from model name if omitted."
        ),
    )
    parser.add_argument(
        "--resume", default=None, metavar="FILE",
        help=(
            "Resume an interrupted sweep. Reads FILE to find already-completed "
            "(depth, rep, probe_id) triples and skips them; appends new results "
            "to the same FILE. Config hash must match."
        ),
    )
    parser.add_argument(
        "--no-fsync", action="store_true",
        help=(
            "Skip os.fsync after each row write. "
            "Faster for local runs; unsafe across power loss."
        ),
    )
    parser.add_argument(
        "--deadline", default=None, metavar="DURATION",
        help=(
            "Stop after this wall-clock duration (e.g. '6h30m', '2h', '90m', '3600s'). "
            "The current probe always finishes; the deadline is checked between probes. "
            "SESSION_SUMMARY.json records deadline_hit=true."
        ),
    )
    parser.add_argument(
        "--preflight-only", action="store_true",
        help=(
            "Run preflight checks (server reachable, disk/memory space, evalset valid, "
            "git state) then exit 0 on success or 1 on failure. No probes are run."
        ),
    )
    parser.add_argument(
        "--quant", default=None, metavar="QUANT",
        help="Quantisation used (e.g. 'q4_k_m', 'q8_0'). Recorded in MANIFEST.json only.",
    )
    parser.add_argument(
        "--kv-precision", default=None, metavar="PREC",
        help="KV-cache precision (e.g. 'fp16', 'int8'). Recorded in MANIFEST.json only.",
    )
    parser.add_argument(
        "--preflight-disk-gb", type=float, default=1.0,
        help="Minimum free disk space (GB) required to pass preflight (default: 1.0).",
    )
    parser.add_argument(
        "--preflight-mem-gb", type=float, default=2.0,
        help="Minimum free memory (GB) required to pass preflight (default: 2.0).",
    )
    args = parser.parse_args()

    # Validate --deadline early so a bad value fails before any work.
    deadline_s: float | None = None
    if args.deadline:
        deadline_s = _parse_duration(args.deadline)

    scorers = _load_scorers()
    outcome_mod = _load_outcome()

    model_variant = args.model_variant or (
        "instruct" if "instruct" in args.model.lower() else "reasoning"
    )
    thinking_enabled = model_variant == "reasoning"

    cfg = {
        "model": args.model,
        "model_variant": model_variant,
        "host": args.host,
        "reps": args.reps,
        "depths": sorted(args.depths),
        "filler_mode": args.filler_mode,
        "hardware_config": args.hardware_config,
        "memory_architecture": args.memory_architecture,
    }
    cfg_hash = _config_hash(cfg)

    probes: list[dict] = []
    with open(args.evalset, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                probes.append(json.loads(line))

    if args.probe_ids:
        probes = [p for p in probes if p["id"] in args.probe_ids]
    if args.skip_judge:
        probes = [p for p in probes if p["scorer_type"] != "judge"]

    total = len(probes) * len(args.depths) * args.reps

    # ── preflight-only path ───────────────────────────────────────────────────
    if args.preflight_only:
        git = _git_state()
        print(f"Git SHA : {git['sha'] or 'unknown'}  dirty={git['dirty']}")
        print(f"Evalset : {args.evalset}  ({len(probes)} probes after filter)")
        failures = _run_preflight(
            host=args.host,
            model=args.model,
            evalset_path=Path(args.evalset),
            probes=probes,
            min_disk_gb=args.preflight_disk_gb,
            min_mem_gb=args.preflight_mem_gb,
        )
        if failures:
            for msg in failures:
                print(f"PREFLIGHT FAIL: {msg}", file=sys.stderr)
            sys.exit(1)
        print("Preflight OK")
        sys.exit(0)

    # Load already-completed rows when resuming.
    completed: set[tuple[int, int, str]] = set()
    resume_path: Path | None = None
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume file not found: {resume_path}")
        completed = _load_completed(resume_path, cfg_hash)
        print(f"Resume: {len(completed)} rows already done in {resume_path.name}")

    count_fn = _make_count_fn(args.host, args.model)

    print(f"Sweep : {len(probes)} probes x {len(args.depths)} depths x {args.reps} reps = {total} calls")
    print(f"Model : {args.model}  variant={model_variant}  thinking={thinking_enabled}")
    print(f"Host  : {args.host}")
    print(f"Filler: {args.filler_mode}  (count_fn calibration enabled)")
    print(f"HW    : {args.hardware_config}  arch={args.memory_architecture}")
    print(f"Config: {cfg_hash}")
    if deadline_s is not None:
        h, rem = divmod(int(deadline_s), 3600)
        m, s = divmod(rem, 60)
        print(f"Deadline: {h}h{m:02d}m{s:02d}s from run start")

    # Pre-build all unique (depth, rep) fillers with the calibrated count_fn.
    # count_fn fires once per unique pair (not once per probe × depth × rep).
    # These calls are instrumentation: they do not appear in result rows or in
    # latency statistics, which capture only _call_ollama_streaming.
    unique_depth_reps = sorted({(d, r) for d in args.depths for r in range(args.reps)})
    filler_cache: dict[tuple[int, int], str] = {}
    non_zero = [(d, r) for d, r in unique_depth_reps if d > 0]
    if non_zero:
        print(f"\nCalibrating filler for {len(non_zero)} depth×rep pairs "
              f"({len(unique_depth_reps) - len(non_zero)} at d=0 need no calibration)...")
    for d, r in unique_depth_reps:
        fn = count_fn if (d > 0 and not args.dry_run) else None
        filler_cache[(d, r)] = context.build_filler(d, seed=r, count_fn=fn)
    if non_zero:
        print("Filler calibration complete.\n")

    if args.dry_run:
        print("\n[dry-run] exiting before any model calls (filler calibration skipped).")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if resume_path is not None:
        out_path = resume_path
        file_mode = "a"
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = RESULTS_DIR / f"run_{ts}.jsonl"
        file_mode = "w"
    remaining = total - len(completed)
    print(f"Output: {out_path}  ({'append' if file_mode == 'a' else 'new'})\n")

    # Derive sibling paths from the result file stem.
    stem = out_path.stem
    manifest_path = out_path.parent / f"{stem}_MANIFEST.json"
    summary_path = out_path.parent / f"{stem}_SESSION_SUMMARY.json"
    sampler_path = out_path.parent / f"{stem}_telemetry.csv"

    # Record run start and git state.
    run_start = time.monotonic()
    utc_start = datetime.now(timezone.utc).isoformat()
    git = _git_state()

    # Write initial MANIFEST (exit_code and utc_end filled in on exit).
    manifest: dict[str, Any] = {
        "utc_start": utc_start,
        "utc_end": None,
        "exit_code": None,
        "argv": sys.argv,
        "git_sha": git["sha"],
        "git_dirty": git["dirty"],
        "model": args.model,
        "quant": args.quant,
        "kv_precision": args.kv_precision,
        "hardware_config": args.hardware_config,
        "memory_architecture": args.memory_architecture,
        "config_hash": cfg_hash,
        "free_b_at_start": _free_bytes_output(),
        "result_file": str(out_path),
    }
    _write_json_atomic(manifest_path, manifest, args.no_fsync)

    # Start background telemetry sampler.
    sampler_stop = _start_sampler(sampler_path, run_start)

    done = 0
    rows_this_session = 0
    deadline_hit = False
    exit_code = 0

    try:
        w = len(str(total))
        with open(out_path, file_mode, encoding="utf-8") as fout:
            if file_mode == "w" and not args.no_fsync:
                # Fsync the directory so this new file's existence is durable,
                # not just its contents.  Only needed on the create path; the
                # resume (append) path implies the entry is already committed.
                _fsync_dir(out_path.parent)
            for depth in sorted(args.depths):
                if deadline_hit:
                    break
                for rep in range(args.reps):
                    if deadline_hit:
                        break
                    cell_probe_seed = depth * 100 + rep
                    cell_probes = list(probes)
                    random.Random(cell_probe_seed).shuffle(cell_probes)
                    for pos, probe in enumerate(cell_probes):
                        if (depth, rep, probe["id"]) in completed:
                            done += 1
                            continue
                        try:
                            row = run_cell(
                                probe,
                                filler_cache[(depth, rep)],
                                depth, rep, pos, cell_probe_seed,
                                args.model, args.host, cfg_hash,
                                args.filler_mode,
                                args.hardware_config,
                                args.memory_architecture,
                                model_variant,
                                thinking_enabled,
                                scorers,
                                outcome_mod,
                            )
                        except Exception as exc:
                            row = {
                                "probe_id": probe["id"],
                                "category": probe["category"],
                                "depth": depth,
                                "rep": rep,
                                "position_in_cell": pos,
                                "cell_probe_seed": cell_probe_seed,
                                "filler_mode": args.filler_mode,
                                "score": None,
                                "score_detail": None,
                                "outcome_class": None,
                                "classification_method": None,
                                "format_compliant": None,
                                "error": str(exc),
                                "config_hash": cfg_hash,
                                "hardware_config": args.hardware_config,
                                "memory_architecture": args.memory_architecture,
                                "model": args.model,
                                "model_variant": model_variant,
                                "thinking_enabled": thinking_enabled,
                                "ctx_suspect": False,
                                "orch_setup_ns":   0,
                                "http_client_ns":  0,
                                "tool_compute_ns": 0,
                            }

                        fout.write(json.dumps(row) + "\n")
                        fout.flush()
                        if not args.no_fsync:
                            os.fsync(fout.fileno())
                        done += 1
                        rows_this_session += 1

                        score_str = (
                            f"{row['score']:.3f}" if row.get("score") is not None else "ERR"
                        )
                        lat = row.get("latency_ms", 0)
                        print(
                            f"[{done:{w}}/{total}] "
                            f"{probe['id']} d={depth:>5} r={rep} pos={pos} "
                            f"score={score_str} lat={lat:.0f}ms",
                            flush=True,
                        )

                        # Check deadline after every completed probe.
                        if _deadline_exceeded(deadline_s, run_start):
                            deadline_hit = True
                            elapsed = time.monotonic() - run_start
                            print(
                                f"\nDeadline reached ({elapsed/3600:.2f}h elapsed). "
                                f"Finished current probe; stopping after {rows_this_session} "
                                f"rows this session.",
                                flush=True,
                            )
                            break

    except Exception:
        exit_code = 1
        raise
    finally:
        sampler_stop.set()

        elapsed_s = round(time.monotonic() - run_start, 1)
        mean_s = round(elapsed_s / rows_this_session, 2) if rows_this_session > 0 else None
        rows_remaining = total - done
        projected = (
            round(rows_remaining / rows_this_session, 2)
            if (rows_this_session > 0 and rows_remaining > 0)
            else 0
        )
        summary: dict[str, Any] = {
            "rows_completed_this_session": rows_this_session,
            "rows_remaining": rows_remaining,
            "elapsed_wall_s": elapsed_s,
            "mean_s_per_row": mean_s,
            "projected_sessions_remaining": projected,
            "deadline_hit": deadline_hit,
        }
        _write_json_atomic(summary_path, summary, args.no_fsync)

        utc_end = datetime.now(timezone.utc).isoformat()
        manifest["utc_end"] = utc_end
        manifest["exit_code"] = exit_code
        _write_json_atomic(manifest_path, manifest, args.no_fsync)

    print(f"\nDone. Results -> {out_path}")
    print(f"Manifest     -> {manifest_path}")
    print(f"Summary      -> {summary_path}")
    print(f"Telemetry    -> {sampler_path}")


if __name__ == "__main__":
    main()
