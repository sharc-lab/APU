"""
kv_provisioning_sweep.py -- provisioning comparison, evo-t2s.

Question: q4_0 at ctx=131072 vs f16 at ctx=32768 cost approximately the same
resident memory (~8.4 GiB vs ~7.5 GiB: KV + 2991 MiB model weight, from
docs/KV_MEASUREMENT.md and results/bw_saturation_20260923T065604Z.jsonl
header table). Same budget, 4x the context -- does q4_0 at long range still
answer correctly, or does the storage win come at an accuracy cost that the
ctx=32768/8192 quality sweep (results/kv_quality_20260923T181840Z_*) did not
test because it never went past ctx=32768?

This is explicitly the follow-up that sweep's own caveat called for: "Do not
claim a null result at ctx=32768 extends to ctx=131072. Quantization error
accumulates over the softmax as key count grows." This run is where that
would show up, if it exists.

Design:
  - 2 configs only: (ctx=131072, q4_0) and (ctx=32768, f16). No cross grid --
    this is a paired provisioning comparison, not a full precision x ctx sweep
    (that already exists in the quality sweep above).
  - START position only, per instruction: artifact at the very beginning,
    filler after it, question last. This is the maximum-distance retrieval
    test and the condition most likely to reveal degradation, since ADJACENT
    (artifact immediately before the question) is a recency-dominated control
    that passed at every precision in the ctx=32768/8192 sweep.
  - 4 probes (art_01 structured/config, art_05 narrative/coordinates,
    art_06 narrative/motion-vs-second -- the one probe with a known
    pre-existing field-confusion failure mode independent of KV precision,
    included deliberately as a diagnostic control -- and art_09
    mixed/keepalive-timeout), 1 rep each (temp=0, deterministic).
  - Order: f16/32768 FIRST (fast, ~5 min/probe from bw_saturation TTFT data
    -> ~20 min total) as a live sanity check before committing to the much
    longer q4_0/131072 leg (~79 min/probe from bw_saturation TTFT=4747s ->
    ~5.3 hours for 4 probes). Commit raw results after each ctx completes,
    per instruction.

Filler sizing FIX vs kv_quality_sweep.py: that script sized filler against
the probe with the LARGEST artifact+question token count, then only ever
trimmed shorter per probe -- which means every probe below that maximum was
under-filled (trimming can't lengthen a too-short filler). Fixed here: size
filler against the SMALLEST artifact+question pairing among the probe subset,
so the built filler is always long enough to be trimmed down for every probe
in the set, hitting the ~90% target precisely for all of them.

KV precision confirmation: as established for this build (b10970 Vulkan,
see docs/FINDINGS.md and docs/RESULT_PROVENANCE.md), no KV buffer-size log
line exists at any verbosity. sys_free memory delta per config is the only
confirmation available and is recorded as such, not represented as a
log-verified measurement.
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

MAX_TOKENS = 32
TEMPERATURE = 0
FILL_RATIO = 0.90
FILLER_SEED = 42

PROBE_IDS = ["art_01", "art_05", "art_06", "art_09"]
POSITIONS = ["START"]

# f16/32768 first (fast sanity check), then the long q4_0/131072 leg.
CONFIG_ORDER = [
    (32768, "f16"),
    (131072, "q4_0"),
]

# Approximate resident memory (KV + 2991 MiB model weight), from
# docs/KV_MEASUREMENT.md and the bw_saturation header table. Recorded per
# config in the output rows for reference; not re-measured by this script
# (that measurement already exists and is cited, not repeated).
APPROX_TOTAL_MIB = {
    (32768, "f16"): 4516.56 + 2991,     # ~7508 MiB
    (131072, "q4_0"): 5578.2 + 2991,    # ~8569 MiB
}


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


def wait_healthy(timeout: int = 240) -> bool:
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
    """Empirically returns {} on b10970 Vulkan -- see module docstring."""
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


def build_prompt_start(filler: str, artifact: str, question: str) -> str:
    return f"{artifact}\n\n{filler}\n\n{question}"


# ------------------------------------------------------------------ inference

def chat_streaming(prompt: str) -> tuple[str, float, float, int, str | None]:
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

    with urllib.request.urlopen(req, timeout=600) as r:
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
    missing = [pid for pid in PROBE_IDS if pid not in seg_by_id]
    if missing:
        raise RuntimeError(f"Probes missing from segments.jsonl: {missing}")
    probes = [seg_by_id[pid] for pid in PROBE_IDS]

    global CONFIG_ORDER
    if smoke:
        probes = probes[:1]
        CONFIG_ORDER = [(32768, "f16")]
        print("SMOKE MODE: 1 config x 1 probe x 1 position = 1 call\n")

    total_calls = len(CONFIG_ORDER) * len(POSITIONS) * len(probes)
    print(f"Grid: {len(CONFIG_ORDER)} configs x {len(POSITIONS)} position(s) "
          f"x {len(probes)} probes = {total_calls} calls")
    for ctx_size, kv_prec in CONFIG_ORDER:
        approx = APPROX_TOTAL_MIB.get((ctx_size, kv_prec))
        print(f"  ctx={ctx_size:>6} prec={kv_prec}  (~{approx:.0f} MiB resident, if known)"
              if approx else f"  ctx={ctx_size:>6} prec={kv_prec}")
    print()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    t_wall_start = time.monotonic()

    for ctx_size, kv_prec in CONFIG_ORDER:
        print(f"\n{'=' * 70}\nCONFIG: ctx={ctx_size} prec={kv_prec}\n{'=' * 70}")
        log_path = rf"C:\apu\kv_prov_{kv_prec}_{ctx_size}.txt"

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

        # FIX vs kv_quality_sweep.py: size filler against the SMALLEST
        # artifact+question pairing in this probe subset, so trimming down
        # is always valid for every probe (never needs to lengthen).
        min_art_q_tokens = min(
            _tokenize(p["artifact"].strip()) + _tokenize(p["question"].strip())
            for p in probes
        )
        target_total = round(ctx_size * FILL_RATIO)
        # +64 token buffer: ctx_mod.build_filler converges to within ~2% of its
        # target, not exactly -- without slack the invariant below can be
        # violated by a handful of tokens on convergence noise alone.
        filler_target = max(target_total - min_art_q_tokens, 256) + 64
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
            per_probe_filler_target = max(target_total - art_tokens - q_tokens, 64)
            if per_probe_filler_target > filler_full_tokens:
                # Convergence noise in ctx_mod.build_filler (~2% tolerance) can
                # occasionally undershoot even with the +64 buffer above. Not
                # fatal: fall back to the full filler built for this ctx,
                # which under-fills by at most a few tokens -- visible in
                # n_prompt_tokens_actual, not silently wrong.
                print(f"  WARNING: {pid} wanted {per_probe_filler_target} filler tokens, "
                      f"only have {filler_full_tokens}; using full filler as-is")
            filler = left_truncate_tokens(filler_full, per_probe_filler_target)
            filler_tokens = _tokenize(filler)

            prompt = build_prompt_start(filler, artifact, question)
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
                "position": "START",
                "ctx_size": ctx_size,
                "kv_precision": kv_prec,
                "approx_resident_mib": APPROX_TOTAL_MIB.get((ctx_size, kv_prec)),
                "approx_resident_mib_basis": "KV (bw_saturation header table) + 2991 MiB "
                                              "model weight (kv_val_vulkan_20260923.json); "
                                              "cited, not re-measured by this script",
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
                "kv_log_note": "b10970 Vulkan emits no KV buffer-size log line at any "
                               "verbosity; kv_log is empty by construction on this build. "
                               "sys_free_delta_mib is the only per-config KV confirmation "
                               "available; it is a memory-delta proxy, not a log-verified "
                               "K/V byte count.",
                "server_props_build": "b10970-bfdc32183",
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
                "cache_prompt": False,
            }
            all_rows.append(row)
            print(f"  [{pid} START] score={score} tok={n_prompt_tokens} "
                  f"ttft={ttft_ms}ms out={output[:40]!r}")

        out_path = RESULTS_DIR / f"kv_provisioning_{ts}_ctx{ctx_size}_{kv_prec}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in all_rows:
                if r["ctx_size"] == ctx_size and r["kv_precision"] == kv_prec:
                    f.write(json.dumps(r) + "\n")
        print(f"\n  Wrote {out_path.name} "
              f"({sum(1 for r in all_rows if r['ctx_size'] == ctx_size and r['kv_precision'] == kv_prec)} rows)")
        (RESULTS_DIR / f"{out_path.stem}.DONE").write_text("done\n")

    kill_server()
    wall_s = time.monotonic() - t_wall_start
    print(f"\n{'=' * 70}\nDONE. {len(all_rows)} rows. Wall clock: {wall_s / 60:.1f} min "
          f"({wall_s:.0f} s)\n{'=' * 70}")

    out_path_all = RESULTS_DIR / f"kv_provisioning_{ts}_full.jsonl"
    with open(out_path_all, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    (RESULTS_DIR / f"kv_provisioning_{ts}_full.DONE").write_text("done\n")
    print(f"Wrote combined file -> {out_path_all.name}")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
