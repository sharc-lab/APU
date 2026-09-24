"""
kv_neardup_sweep.py -- near-duplicate discrimination under KV quantization, evo-t2s.

DESIGNED, NOT YET RUN. Queued to run immediately after the provisioning
comparison (kv_provisioning_sweep.py) finishes.

HYPOTHESIS: quantized keys lose the fine differences that separate
near-identical records. q4_0 fails to pick the correct record out of a set of
same-schema distractors where f16 succeeds, and the failure rate should
increase with distractor density -- this is what agentic tool output (many
JSON/log records with the same field names, differing only in values) looks
like, so it's a more realistic failure mode than the single-fact retrieval
tested by kv_quality_sweep.py, which found no measurable cost (see
docs/FINDINGS.md "KV Precision Quality Sweep") but was not a
discrimination-under-interference test.

REUSED MATERIAL (found before building anything, per instruction):
  - harness/schema_collision.py already contains F-SCHEMA distractor
    generators for art_01, art_06, art_07 -- each is a closure that produces
    ONE same-schema, different-identifying-value record per call (e.g.
    _build_art01_schema's make_entry() emits a CONFIG JSON block with the
    exact same 6 fields as the real art_01 artifact -- listen_port,
    max_payload_kb, auth_scheme, flush_interval_ms, retry_backoff_base_ms,
    circuit_breaker_threshold -- but a different service name/port/etc each
    time). This is exactly the per-record generator this experiment needs
    for precise distractor COUNT control, so art_01's entry format is
    reused directly below (ported, not imported, to avoid pulling in that
    script's own watchdog-thread and Ollama-specific plumbing).
  - evaluation/probes/segments.jsonl has art_01's artifact+question in the
    separated-field format needed to build prompts (same source
    kv_quality_sweep.py and kv_provisioning_sweep.py already used).
  - docs/FINDINGS.md "Stage 1.1 -- Schema Collision" (135 calls, 2026-08-24)
    already ran F-SCHEMA at r=1.20 (artifact present, full context) and found
    ZERO genuine filler lifts at f16 across all three models -- "art_07/
    F-SCHEMA ... Score=1.00 across all three models. No version confusion
    induced by same-schema competing records." That was an uncontrolled
    density (whatever filler-fill produced) at f16 only. This design adds
    controlled density steps and precision as a second variable, testing
    whether an effect that isn't visible at f16 appears under quantization.
  - art_06 is deliberately NOT the primary probe here: docs/FINDINGS.md
    documents a pre-existing, precision-independent field-confusion failure
    mode specific to art_06 (Motion field vs. Second field), which recurred
    in the kv_quality_sweep run. Using art_06 would conflate that known bug
    with a density/quantization effect. art_01 (clean single numeric field,
    no known confound) is used instead.

DESIGN:
  - Target position FIXED: the real art_01 artifact is always the LAST
    record before the question (ADJACENT-style, already validated as a
    retrieval control in kv_quality_sweep.py). N distractor records
    (same schema, different values, target's own port value excluded from
    the distractor pool) are placed before it. Density is the only
    manipulated placement variable.
  - Distractor density: N in {0, 4, 16, 64}.
  - KV precision: f16, q8_0, q4_0 -- ALL with -fa on passed EXPLICITLY.
    (Per docs/FINDINGS.md: q8_0/q4_0 get flash-attn forced on regardless by
    llama-context.cpp:3704-3707 since quantized V requires it; f16's AUTO
    resolution was only inferred from source + empirical corroboration, not
    logged. Passing -fa on for every precision removes that ambiguity
    entirely for this run -- all three configs provably run the identical
    attention codepath, so any precision-density interaction found cannot be
    attributed to a flash-attn on/off difference between arms.)
  - ctx = 8192 only for this pass (fast: prompts here are NOT filled to 90%
    of ctx like the quality/provisioning sweeps -- they're just N distractor
    records + the target + the question, a few hundred to ~3000 tokens even
    at N=64 -- so this should run faster than the quality sweep's 90%-fill
    ctx=8192 calls, not the same ~25s/call. See wall-clock estimate below).
  - 1 probe (art_01), 1 rep per cell (temp=0, deterministic -- matches
    established project convention that reps add no information at temp=0).
  - Grid: 3 precisions x 4 densities x 1 probe x 1 rep = 12 calls, 3 server
    restarts (one per precision; density doesn't require a restart).

BASELINE GATE: f16 at density=0 must score 1.0 (this is exactly the
ADJACENT/ctx=8192/f16/art_01 cell from kv_quality_sweep.py, which already
scored 1.0 there -- so this is a sanity check on this script's plumbing, not
a new empirical question). If f16 ALSO degrades as density increases (4/16/
64), that is reported as its own finding -- a general context-interference
effect independent of quantization -- not treated as a gate failure that
aborts the run. Nothing here aborts on a bad result; every cell always runs.

CALL COUNT: 12. WALL CLOCK ESTIMATE: user's own ceiling estimate is
~25s/call (matching the quality sweep's ctx=8192/90%-fill calls) => 12 x 25s
= 300s = 5 min, plus ~3 server restarts x 30-60s each (model load, same
regardless of prompt size) = 1.5-3 min => ~6.5-8 min total, as an upper
bound. Actual is very likely faster: these prompts are far shorter than a
90%-fill ctx=8192 prompt (max ~3000 tokens at N=64 vs ~7373 for a 90%-fill
prompt), and TTFT scales with prompt length in this build's memory-
bandwidth-bound regime (established in the bandwidth-saturation sweep), so
per-call time should scale down roughly proportionally. Treat 6.5-8 min as
the number to plan around; it may finish in closer to 2-3 min.

PRECISION-CHECK FIX (per docs/FINDINGS.md diagnosis of the quality sweep's
negative sys_free deltas): this script does NOT use system-wide free memory
before/after server start. It waits for the previous llama-server PID to
fully exit, then records the NEW server's own process command line (proves
which -ctk/-ctv/-fa flags actually took effect, closing the log gap this
build has at every verbosity) and process-private memory
(WorkingSet64/PrivateMemorySize64) via a single-process CIM/Get-Process
query -- immune to the previous process's teardown timing, unlike a
system-wide free-memory snapshot.
"""

from __future__ import annotations

import importlib.util
import json
import random
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

SERVER_BIN = r"C:\apu\bin\llama-b10970\llama-server.exe"
MODEL_PATH = r"C:\apu\models\qwen3-4b-instruct-85e4a5b7.gguf"
PORT = 8384
SERVER_URL = f"http://127.0.0.1:{PORT}"

MAX_TOKENS = 32
TEMPERATURE = 0
FILLER_SEED = 42

DENSITIES = [0, 4, 16, 64]
PRECISIONS = ["f16", "q8_0", "q4_0"]
PROBE_ID = "art_01"


# ------------------------------------------------------------------ art_01 F-SCHEMA distractor generator
# Ported from harness/schema_collision.py:_build_art01_schema (same entry
# format: exact same CONFIG JSON schema as the real art_01 artifact, invoice/
# relay-adjacent service names, non-overlapping port values). Not imported
# directly to avoid that script's watchdog thread and Ollama-specific
# module-level setup.

_ART01_SERVICES = [
    "invoice-gateway", "invoice-processor", "invoice-validator",
    "invoice-archiver", "invoice-normalizer", "invoice-router",
    "billing-relay", "payment-relay", "settlement-relay", "audit-relay",
    "sync-relay", "ledger-api", "statement-api", "receipt-processor",
    "charge-router", "refund-handler",
]
_ART01_SCHEMES = ["HMAC-SHA3", "Bearer", "mTLS", "API-Key", "OIDC", "HMAC-SHA256"]


def make_art01_distractor(rng: random.Random, exclude_ports: set[int]) -> str:
    port = rng.randint(10240, 65535)
    while port in exclude_ports:
        port = rng.randint(10240, 65535)
    exclude_ports.add(port)
    svc = rng.choice(_ART01_SERVICES)
    ver = f"{rng.randint(1, 5)}.{rng.randint(0, 12)}"
    kb = rng.choice([64, 128, 192, 256, 512])
    scheme = rng.choice(_ART01_SCHEMES)
    return (
        f"[CONFIG: service={svc} v{ver}]\n"
        f"{{\n"
        f'  "listen_port": {port},\n'
        f'  "max_payload_kb": {kb},\n'
        f'  "auth_scheme": "{scheme}",\n'
        f'  "flush_interval_ms": {rng.randint(500, 5000)},\n'
        f'  "retry_backoff_base_ms": {rng.randint(100, 2000)},\n'
        f'  "circuit_breaker_threshold": {rng.randint(4, 30)}\n'
        f"}}"
    )


def build_prompt(n_distractors: int, target_artifact: str, question: str, seed: int) -> str:
    """Target is always the LAST record before the question (fixed position)."""
    rng = random.Random(seed)
    exclude_ports = {51847}  # the real art_01 artifact's listen_port
    distractors = [make_art01_distractor(rng, exclude_ports) for _ in range(n_distractors)]
    blocks = distractors + [target_artifact]
    return "\n\n".join(blocks) + "\n\n" + question


# ------------------------------------------------------------------ server mgmt (fixed precision check)

def wait_process_exit(name: str = "llama-server", timeout: int = 30) -> bool:
    """Wait for the previous server to fully exit before the next config's
    'before' snapshot -- fixes the negative-delta bug from kv_quality_sweep.py
    (see docs/FINDINGS.md), where a fixed 3s sleep raced the OS/driver's
    memory release for a large Vulkan-backed process."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"(Get-Process -Name {name} -ErrorAction SilentlyContinue | Measure-Object).Count"],
            capture_output=True, text=True, timeout=10,
        )
        if r.stdout.strip() in ("", "0"):
            return True
        time.sleep(1)
    return False


def kill_server():
    subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-Process -Name llama-server -ErrorAction SilentlyContinue | Stop-Process -Force"],
        capture_output=True, timeout=15,
    )
    wait_process_exit()


def start_server(ctx_size: int, kv_prec: str, log_path: str) -> subprocess.Popen:
    kill_server()
    cmd = [
        SERVER_BIN,
        "-m", MODEL_PATH,
        "--port", str(PORT),
        "-c", str(ctx_size),
        "-ctk", kv_prec,
        "-ctv", kv_prec,
        "-fa", "on",              # explicit for every precision -- see module docstring
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


def get_server_process_info(pid: int) -> dict:
    """Single-process command line + private memory. Replaces the system-wide
    sys_free before/after snapshot that produced negative deltas in
    kv_quality_sweep.py -- immune to the previous process's teardown timing
    since it queries only this PID."""
    ps_cmd = (
        f"$p = Get-Process -Id {pid} -ErrorAction SilentlyContinue; "
        f"$cim = Get-CimInstance Win32_Process -Filter \"ProcessId={pid}\" -ErrorAction SilentlyContinue; "
        f"[PSCustomObject]@{{CommandLine=$cim.CommandLine; "
        f"WorkingSet64=$p.WorkingSet64; PrivateMemorySize64=$p.PrivateMemorySize64}} | ConvertTo-Json"
    )
    r = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps_cmd],
        capture_output=True, text=True, timeout=15,
    )
    try:
        return json.loads(r.stdout)
    except Exception:
        return {"error": (r.stdout + r.stderr).strip()}


# ------------------------------------------------------------------ tokenize

def _tokenize(text: str) -> int:
    body = json.dumps({"content": text, "add_special": False}).encode()
    req = urllib.request.Request(
        f"{SERVER_URL}/tokenize", data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return len(json.loads(r.read())["tokens"])


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
    if PROBE_ID not in seg_by_id:
        raise RuntimeError(f"{PROBE_ID} missing from segments.jsonl")
    probe = seg_by_id[PROBE_ID]
    artifact = probe["artifact"].strip()
    question = probe["question"].strip()
    expected = probe["expected"]
    scorer_type = probe["scorer_type"]

    global DENSITIES, PRECISIONS
    if smoke:
        DENSITIES = [0, 4]
        PRECISIONS = ["f16"]
        print("SMOKE MODE: 1 precision x 2 densities = 2 calls\n")

    total_calls = len(PRECISIONS) * len(DENSITIES)
    print(f"Grid: {len(PRECISIONS)} precisions x {len(DENSITIES)} densities "
          f"x 1 probe ({PROBE_ID}) x 1 rep = {total_calls} calls")
    print(f"Densities: {DENSITIES}  Precisions: {PRECISIONS}  ctx=8192 (fixed)")
    print()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    t_wall_start = time.monotonic()
    gate_result = None

    for kv_prec in PRECISIONS:
        print(f"\n{'=' * 70}\nCONFIG: prec={kv_prec} ctx=8192\n{'=' * 70}")
        log_path = rf"C:\apu\kv_neardup_{kv_prec}_8192.txt"

        print("  Waiting for previous server to fully exit, then starting...")
        proc = start_server(8192, kv_prec, log_path)
        if not wait_healthy():
            print("  FAIL: server did not become healthy")
            proc.kill()
            continue

        proc_info = get_server_process_info(proc.pid)
        print(f"  Server up (pid={proc.pid}). Process info: {proc_info}")

        for n_distractors in DENSITIES:
            prompt = build_prompt(n_distractors, artifact, question, seed=FILLER_SEED)
            n_prompt_tokens = _tokenize(prompt)

            try:
                output, latency_ms, ttft_ms, tokens_in_api, finish_reason = \
                    chat_streaming(prompt)
                error = None
            except Exception as e:
                output, latency_ms, ttft_ms, tokens_in_api, finish_reason = "", 0.0, None, 0, None
                error = str(e)

            probe_dict = {"id": PROBE_ID, "scorer_type": scorer_type, "expected": expected}
            try:
                score, score_detail = scorers.score(probe_dict, output)
            except Exception as e:
                score, score_detail = None, f"scorer_error:{e}"

            if kv_prec == "f16" and n_distractors == 0:
                gate_result = score
                print(f"  [GATE] f16 density=0: score={score} "
                      f"({'PASS' if score == 1.0 else 'FAIL -- see note below'})")

            row = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "probe_id": PROBE_ID,
                "ctx_size": 8192,
                "kv_precision": kv_prec,
                "flash_attn": "on (explicit -fa on for every precision)",
                "distractor_density": n_distractors,
                "target_position": "last_before_question (fixed)",
                "score": score,
                "score_detail": score_detail,
                "output": output,
                "expected": expected,
                "scorer_type": scorer_type,
                "n_prompt_tokens_actual": n_prompt_tokens,
                "ttft_ms": round(ttft_ms, 1) if ttft_ms else None,
                "latency_ms": round(latency_ms, 1),
                "tokens_in_api": tokens_in_api,
                "finish_reason": finish_reason,
                "error": error,
                "server_pid": proc.pid,
                "server_process_info": proc_info,
                "server_props_build": "b10970-bfdc32183",
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
                "cache_prompt": False,
            }
            all_rows.append(row)
            print(f"  [density={n_distractors:>2}] score={score} tok={n_prompt_tokens} "
                  f"ttft={ttft_ms}ms out={output[:40]!r}")

    kill_server()
    wall_s = time.monotonic() - t_wall_start

    if gate_result is not None and gate_result != 1.0:
        print(f"\n*** GATE NOTE: f16 density=0 scored {gate_result}, not 1.0. "
              f"Per design, this does NOT abort the run -- all cells already ran. "
              f"Report this as its own result (a baseline retrieval failure "
              f"independent of density/quantization), not as evidence the "
              f"density sweep is invalid. ***\n")

    print(f"\n{'=' * 70}\nDONE. {len(all_rows)} rows. Wall clock: {wall_s / 60:.1f} min "
          f"({wall_s:.0f} s)\n{'=' * 70}")

    out_path = RESULTS_DIR / f"kv_neardup_{ts}_full.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    (RESULTS_DIR / f"{out_path.stem}.DONE").write_text("done\n")
    print(f"Wrote -> {out_path.name}")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
