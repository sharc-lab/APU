"""
kv_quality_sweep.py -- does KV quantization cost task quality? (evo-t2s)

Independent variables: KV precision (f16, q4_0, q8_0) x context length
(32768, 8192) x artifact position (ADJACENT, START). 12 configs x 10 probes
= 120 calls.

ADJACENT: {filler}\n\n{artifact}\n\n{question}   -- artifact immediately before
          the question. Control: should pass at every precision.
START:    {artifact}\n\n{filler}\n\n{question}   -- artifact at the very
          beginning, filler after it. Maximum-distance retrieval test.

Order (fixed, not alphabetical): ctx=32768 f16 and q4_0 first (both positions),
before anything else, per explicit instruction -- these are the cells most
likely to show a quantization effect and should not wait behind the rest of
the grid. Remainder: 32768/q8_0, then the 8192 tier (f16, q4_0, q8_0).

Server is restarted once per (ctx, precision) pair (6 restarts); ADJACENT and
START share a server instance since neither changes -c/-ctk/-ctv. cache_prompt
is always false so KV is not reused across calls within a config.

KV precision confirmation: b10970 Vulkan does not emit a KV buffer-size log
line at any verbosity (confirmed empirically against kv_val_f16.txt and
bw_sweep_run.log -- see docs/FINDINGS.md and docs/RESULT_PROVENANCE.md). The
only per-config confirmation available on this build is the sys_free memory
delta before/after server start, which is recorded in every row as
sys_free_before_server_mib / sys_free_after_server_mib. Do not read these as
"verified K and V MiB from the server log" -- no such log line exists on this
build; they are a memory-delta proxy, documented as such.

Do NOT claim a null result at ctx=32768 extends to ctx=131072. Quantization
error in the softmax accumulates as key count grows; long context is the
condition where degradation would appear, and it is out of scope for this run.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(r"C:\apu\APU")
PROBES_DIR = REPO / "evaluation" / "probes"
RESULTS_DIR = REPO / "results"

sys.path.insert(0, str(REPO / "harness"))
import context as ctx_mod  # noqa: E402

SERVER_BIN = r"C:\apu\bin\llama-b10970\llama-server.exe"
MODEL_PATH = r"C:\apu\models\qwen3-4b-instruct-85e4a5b7.gguf"
PORT = 8384
SERVER_URL = f"http://127.0.0.1:{PORT}"

MAX_TOKENS = 32          # all 10 artifact probes are short exact-match answers
TEMPERATURE = 0
FILL_RATIO = 0.90        # target: filler+artifact+question ~= 90% of ctx
FILLER_SEED = 42

ARTIFACT_PROBE_IDS = [f"art_{i:02d}" for i in range(1, 11)]
POSITIONS = ["ADJACENT", "START"]

# (ctx, precision) run order -- 32768/f16 and 32768/q4_0 first, per instruction.
CONFIG_ORDER = [
    (32768, "f16"),
    (32768, "q4_0"),
    (32768, "q8_0"),
    (8192, "f16"),
    (8192, "q4_0"),
    (8192, "q8_0"),
]


# ------------------------------------------------------------------ server mgmt

def sys_free_mib() -> float:
    try:
        import ctypes

        class MEMSTATUS(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        ms = MEMSTATUS()
        ms.dwLength = ctypes.sizeof(ms)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        return ms.ullAvailPhys / (1024 * 1024)
    except Exception:
        return -1.0


def kill_server():
    subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-Process -Name llama-server -ErrorAction SilentlyContinue | Stop-Process -Force"],
        capture_output=True, timeout=15,
    )
    time.sleep(3)


def start_server(ctx_size: int, kv_prec: str, log_path: str) -> subprocess.Popen:
    kill_server()
    cmd = [
        SERVER_BIN,
        "-m", MODEL_PATH,
        "--port", str(PORT),
        "-c", str(ctx_size),
        "-ctk", kv_prec,
        "-ctv", kv_prec,
        "-ngl", "99",
        "-np", "1",
        "--no-context-shift",
        "--reasoning-format", "deepseek",
        "--reasoning-budget", "0",
        "--log-file", log_path,
        "--log-verbosity", "3",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def wait_healthy(timeout: int = 180) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{SERVER_URL}/health", timeout=5) as r:
                if json.loads(r.read()).get("status") == "ok":
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def parse_kv_log(log_path: str) -> dict:
    """Best-effort KV buffer-size extraction. Empirically returns {} on b10970
    Vulkan (no such log line exists at any verbosity) -- kept only so a future
    build that does log it is picked up automatically, not because this build
    is expected to populate it."""
    result = {}
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError as e:
        return {"log_error": str(e)}
    for line in text.splitlines():
        m = re.search(r"KV buffer size\s*[=:]\s*([\d.]+)\s*MiB", line, re.IGNORECASE)
        if m:
            result.setdefault("kv_buffer_mib_list", []).append(float(m.group(1)))
        m = re.search(r"llama_kv_cache:\s+K\s+\(\s*\w+\s*\):\s+([\d.]+)\s+MiB", line)
        if m:
            result["K_MiB"] = float(m.group(1))
        m = re.search(r"llama_kv_cache:\s+V\s+\(\s*\w+\s*\):\s+([\d.]+)\s+MiB", line)
        if m:
            result["V_MiB"] = float(m.group(1))
    return result


# ------------------------------------------------------------------ tokenize / truncate

def _tokenize(text: str) -> int:
    body = json.dumps({"content": text, "add_special": False}).encode()
    req = urllib.request.Request(
        f"{SERVER_URL}/tokenize", data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return len(json.loads(r.read())["tokens"])


def left_truncate_tokens(text: str, target_tokens: int) -> str:
    """Token-accurate left truncation via /tokenize binary search (filler only)."""
    if _tokenize(text) <= target_tokens:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi) // 2
        if _tokenize(text[mid:]) <= target_tokens:
            hi = mid
        else:
            lo = mid + 1
    return text[lo:]


def build_prompt(position: str, filler: str, artifact: str, question: str) -> str:
    if position == "ADJACENT":
        return f"{filler}\n\n{artifact}\n\n{question}"
    else:  # START
        return f"{artifact}\n\n{filler}\n\n{question}"


# ------------------------------------------------------------------ inference

def chat_streaming(prompt: str) -> tuple[str, float, float, int, str | None]:
    """Returns (text, latency_ms, ttft_ms, n_prompt_tokens_api, finish_reason)."""
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "stream": True,
        "cache_prompt": False,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        f"{SERVER_URL}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    ttft_ms = None
    chunks: list[str] = []
    tokens_in_api = 0
    finish_reason = None

    with urllib.request.urlopen(req, timeout=180) as r:
        for raw_line in r:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            choice0 = choices[0] if choices else {}
            delta = choice0.get("delta", {})
            content = delta.get("content", "")
            if content and ttft_ms is None:
                ttft_ms = (time.perf_counter() - t0) * 1000.0
            if content:
                chunks.append(content)
            fr = choice0.get("finish_reason")
            if fr:
                finish_reason = fr
            usage = chunk.get("usage")
            if usage:
                tokens_in_api = usage.get("prompt_tokens", tokens_in_api)

    latency_ms = (time.perf_counter() - t0) * 1000.0
    text = "".join(chunks).strip()
    return text, latency_ms, ttft_ms or latency_ms, tokens_in_api, finish_reason


def _load_scorers():
    spec = importlib.util.spec_from_file_location("probes_scorers", PROBES_DIR / "scorers.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ main

def main(smoke: bool = False):
    scorers = _load_scorers()

    segs = [
        json.loads(l)
        for l in (PROBES_DIR / "segments.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    seg_by_id = {s["id"]: s for s in segs}
    missing = [pid for pid in ARTIFACT_PROBE_IDS if pid not in seg_by_id]
    if missing:
        raise RuntimeError(f"Probes missing from segments.jsonl: {missing}")
    probes = [seg_by_id[pid] for pid in ARTIFACT_PROBE_IDS]

    global CONFIG_ORDER
    if smoke:
        probes = probes[:2]
        CONFIG_ORDER = [(8192, "f16")]
        print("SMOKE MODE: 1 config x 2 probes x 2 positions = 4 calls\n")

    total_calls = len(CONFIG_ORDER) * len(POSITIONS) * len(probes)
    print(f"Grid: {len(CONFIG_ORDER)} (ctx,precision) configs x {len(POSITIONS)} positions "
          f"x {len(probes)} probes = {total_calls} calls")
    for ctx_size, kv_prec in CONFIG_ORDER:
        print(f"  ctx={ctx_size:>6} prec={kv_prec}")
    print()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    t_wall_start = time.monotonic()
    seen_ctx: set[int] = set()

    for ctx_size, kv_prec in CONFIG_ORDER:
        print(f"\n{'=' * 70}\nCONFIG: ctx={ctx_size} prec={kv_prec}\n{'=' * 70}")
        log_path = rf"C:\apu\kv_qual_{kv_prec}_{ctx_size}.txt"

        mem_before = sys_free_mib()
        print(f"  sys_free_before_server = {mem_before:.0f} MiB")
        print("  Starting server...")
        proc = start_server(ctx_size, kv_prec, log_path)
        if not wait_healthy():
            print("  FAIL: server did not become healthy")
            proc.kill()
            continue
        mem_after = sys_free_mib()
        print(f"  Server up. sys_free_after_server = {mem_after:.0f} MiB "
              f"(delta = {mem_before - mem_after:.0f} MiB)")

        kv_log = parse_kv_log(log_path)
        print(f"  KV from log: {kv_log}")

        # Build filler once per ctx (independent of position/probe): target the
        # widest artifact+question pair to guarantee headroom, then trim per-probe.
        max_art_q_tokens = max(
            _tokenize(p["artifact"].strip()) + _tokenize(p["question"].strip())
            for p in probes
        )
        target_total = round(ctx_size * FILL_RATIO)
        filler_target = max(target_total - max_art_q_tokens, 256)
        print(f"  Building filler (~{filler_target} tokens, seed={FILLER_SEED})...")
        filler_full = ctx_mod.build_filler(filler_target, seed=FILLER_SEED, count_fn=_tokenize)
        filler_full_tokens = _tokenize(filler_full)
        print(f"  filler: {filler_full_tokens} tokens ({len(filler_full)} chars)")

        for probe in probes:
            pid = probe["id"]
            artifact = probe["artifact"].strip()
            question = probe["question"].strip()
            expected = probe["expected"]
            scorer_type = probe["scorer_type"]

            art_tokens = _tokenize(artifact)
            q_tokens = _tokenize(question)
            # Per-probe filler budget so total (filler+artifact+question) hits target_total.
            per_probe_filler_target = max(target_total - art_tokens - q_tokens, 64)
            filler = left_truncate_tokens(filler_full, per_probe_filler_target) \
                if per_probe_filler_target < filler_full_tokens else filler_full
            filler_tokens = _tokenize(filler)

            for position in POSITIONS:
                prompt = build_prompt(position, filler, artifact, question)
                n_prompt_tokens = _tokenize(prompt)

                try:
                    output, latency_ms, ttft_ms, tokens_in_api, finish_reason = \
                        chat_streaming(prompt)
                    error = None
                except Exception as e:
                    output, latency_ms, ttft_ms, tokens_in_api, finish_reason = "", 0.0, None, 0, None
                    error = str(e)

                probe_dict = {"id": pid, "scorer_type": scorer_type, "expected": expected}
                try:
                    score, score_detail = scorers.score(probe_dict, output)
                except Exception as e:
                    score, score_detail = None, f"scorer_error:{e}"

                row = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "probe_id": pid,
                    "category": probe.get("category"),
                    "difficulty": probe.get("difficulty"),
                    "position": position,
                    "ctx_size": ctx_size,
                    "kv_precision": kv_prec,
                    "score": score,
                    "score_detail": score_detail,
                    "output": output,
                    "expected": expected,
                    "scorer_type": scorer_type,
                    "artifact_tokens": art_tokens,
                    "question_tokens": q_tokens,
                    "filler_tokens": filler_tokens,
                    "n_prompt_tokens_actual": n_prompt_tokens,
                    "target_total_tokens": target_total,
                    "fill_ratio": FILL_RATIO,
                    "ttft_ms": round(ttft_ms, 1) if ttft_ms else None,
                    "latency_ms": round(latency_ms, 1),
                    "tokens_in_api": tokens_in_api,
                    "finish_reason": finish_reason,
                    "error": error,
                    "sys_free_before_server_mib": round(mem_before),
                    "sys_free_after_server_mib": round(mem_after),
                    "sys_free_delta_mib": round(mem_before - mem_after),
                    "kv_log": kv_log,
                    "kv_log_note": "b10970 Vulkan emits no KV buffer-size log line at "
                                   "any verbosity; kv_log is empty by construction on "
                                   "this build. sys_free_delta_mib is the only per-config "
                                   "KV confirmation available; it is a memory-delta proxy, "
                                   "not a log-verified K/V byte count.",
                    "server_props_build": "b10970-bfdc32183",
                    "max_tokens": MAX_TOKENS,
                    "temperature": TEMPERATURE,
                    "cache_prompt": False,
                }
                all_rows.append(row)
                print(f"  [{pid} {position}] score={score} tok={n_prompt_tokens} "
                      f"ttft={ttft_ms}ms out={output[:40]!r}")

        seen_ctx.add(ctx_size)
        # Write raw results once all configs for this ctx length are done, i.e.
        # this was the last (ctx_size, prec) entry for this ctx in CONFIG_ORDER.
        # NOT committed here: evo-t2s has no local git identity configured (its
        # existing history was pulled from GitHub, not committed on-box), so
        # committing is left to the controlling workstation, which pulls this
        # file over SSH/SCP once it appears.
        remaining_configs_this_ctx = [
            c for c in CONFIG_ORDER[CONFIG_ORDER.index((ctx_size, kv_prec)) + 1:]
            if c[0] == ctx_size
        ]
        if not remaining_configs_this_ctx:
            out_path = RESULTS_DIR / f"kv_quality_{ts}_ctx{ctx_size}.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                for r in all_rows:
                    if r["ctx_size"] == ctx_size:
                        f.write(json.dumps(r) + "\n")
            print(f"\n  Wrote {out_path.name} ({sum(1 for r in all_rows if r['ctx_size'] == ctx_size)} rows)")
            # Marker file: controlling workstation polls for this to know when
            # to pull this ctx tier's results back.
            (RESULTS_DIR / f"kv_quality_{ts}_ctx{ctx_size}.DONE").write_text("done\n")

    kill_server()
    wall_s = time.monotonic() - t_wall_start
    print(f"\n{'=' * 70}\nDONE. {len(all_rows)} rows. Wall clock: {wall_s / 60:.1f} min "
          f"({wall_s:.0f} s)\n{'=' * 70}")

    out_path_all = RESULTS_DIR / f"kv_quality_{ts}_full.jsonl"
    with open(out_path_all, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    (RESULTS_DIR / f"kv_quality_{ts}_full.DONE").write_text("done\n")
    print(f"Wrote combined file -> {out_path_all.name}")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
