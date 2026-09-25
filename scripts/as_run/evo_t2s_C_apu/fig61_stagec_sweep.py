"""Fig 6.1 Stage C replication — evo-t2s, qwen3:4b-instruct checkpoint.

11 probes, 6 ratios, 2 arms, 3 reps = 396 calls.
Matches stage_c_position_pressure.py exactly: max_tokens=128, temperature=0,
char-based left truncation, same filler seed/target, same probe set.

Differences from stage C (Blade):
  - llama-server OpenAI-compat endpoint instead of Ollama
  - streaming=True to capture ttft_ms
  - build_id live-queried from /props at startup
  - orch_setup_ns, http_client_ns, tool_compute_ns per row
  - platform = evo-t2s, hardware_config = evox2_evo-t2s
"""

from __future__ import annotations

import importlib.util
import json
import time
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PROBES_DIR = REPO / "evaluation" / "probes"
RESULTS_DIR = REPO / "results"

sys.path.insert(0, str(REPO / "harness"))

SERVER_URL = "http://127.0.0.1:8383"
N_CTX_SLOT = 8192
MAX_TOKENS = 128        # matches stage C
TEMPERATURE = 0         # matches stage C

BUDGET_RATIOS = [1.20, 1.00, 0.85, 0.70, 0.55, 0.40]
ARMS = ["LATE", "EARLY"]
N_REPS = 3

# Stage C probe set (11 probes from segments.jsonl)
PROBE_IDS = ["rag_01", "rag_02", "rag_03", "rag_04", "rag_05", "rag_06",
             "sea_01", "sea_03", "sea_04", "sea_05", "sea_06"]

FILLER_TARGET = 4000
FILLER_SEED = 42
_CHARS_PER_TOK = 5.03

_TEMPLATE = (
    "Administrative log entry {n}: The oversight committee reviewed all "
    "submitted documentation for compliance period {n} and confirmed that "
    "operational metrics remained within established baseline parameters. "
    "No anomalies were recorded in district {n} during the reference interval. "
    "Budget allocations for cycle {n} were processed according to standing "
    "procedure without escalation. Routine maintenance of infrastructure "
    "segment {n} was completed on schedule and filed under reference {n}. "
)

HW_CONFIG = "evox2_evo-t2s"
MEM_ARCH = "unified"
PLATFORM = "evo-t2s"


# ── helpers ──────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> int:
    body = json.dumps({"content": text, "add_special": False}).encode()
    req = urllib.request.Request(
        f"{SERVER_URL}/tokenize",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return len(json.loads(r.read())["tokens"])


def _query_props() -> dict:
    req = urllib.request.Request(f"{SERVER_URL}/props", method="GET")
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def _build_filler(target_tokens: int, seed: int = FILLER_SEED) -> tuple[str, int]:
    """Build F-NUM filler via iterative /tokenize calibration (matches fig61_sweep.py)."""
    import random
    rng = random.Random(seed)
    base_entries = max(1, round(target_tokens / _CHARS_PER_TOK / len(_TEMPLATE) * len(_TEMPLATE)))
    n_entries = max(1, round(target_tokens / _CHARS_PER_TOK / len(_TEMPLATE.format(n=1))))

    filler = "".join(_TEMPLATE.format(n=rng.randint(10000, 99999)) for _ in range(n_entries))
    current_tokens = _tokenize(filler)

    for _ in range(8):
        if abs(current_tokens - target_tokens) / max(target_tokens, 1) < 0.02:
            break
        ratio = target_tokens / max(current_tokens, 1)
        n_entries = max(1, round(n_entries * ratio))
        rng2 = random.Random(seed)
        filler = "".join(_TEMPLATE.format(n=rng2.randint(10000, 99999)) for _ in range(n_entries))
        current_tokens = _tokenize(filler)

    return filler, current_tokens


def build_arm_prompt(arm: str, filler: str, artifact: str, question: str) -> str:
    if arm == "LATE":
        return f"{filler}\n\n{artifact}\n\n{question}"
    else:
        return f"{artifact}\n\n{filler}\n\n{question}"


def left_truncate(prompt: str, full_tokens: int, target_tokens: int) -> str:
    """Char-based left truncation matching stage_c_position_pressure.py exactly."""
    if target_tokens >= full_tokens:
        return prompt
    chars_to_keep = round(len(prompt) * target_tokens / full_tokens)
    chars_to_keep = max(1, min(chars_to_keep, len(prompt)))
    return prompt[len(prompt) - chars_to_keep:]


def _chat_streaming(prompt: str) -> tuple[str, float, float, int, int, str | None]:
    """Streaming POST to /v1/chat/completions.

    Returns (text, latency_ms, ttft_ms, tokens_in, tokens_out, finish_reason).
    ttft_ms = wall time from request start to first non-empty content chunk.
    """
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"{SERVER_URL}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    ttft_ms = None
    chunks = []
    tokens_in = 0
    tokens_out = 0
    finish_reason = None

    with urllib.request.urlopen(req, timeout=120) as r:
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

            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content", "")
            if content and ttft_ms is None:
                ttft_ms = (time.perf_counter() - t0) * 1000.0

            if content:
                chunks.append(content)

            fr = chunk.get("choices", [{}])[0].get("finish_reason")
            if fr:
                finish_reason = fr

            usage = chunk.get("usage")
            if usage:
                tokens_in = usage.get("prompt_tokens", tokens_in)
                tokens_out = usage.get("completion_tokens", tokens_out)

    latency_ms = (time.perf_counter() - t0) * 1000.0
    text = "".join(chunks).strip()

    # fallback: estimate tokens_out from chunks
    if tokens_out == 0:
        tokens_out = len(chunks)

    return text, latency_ms, ttft_ms or latency_ms, tokens_in, tokens_out, finish_reason


def _load_scorers():
    spec = importlib.util.spec_from_file_location("probes_scorers", PROBES_DIR / "scorers.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── main ─────────────────────────────────────────────────────────────────────

def main(smoke: bool = False, gate: bool = False, gate_only: bool = False):
    import socket
    hostname = socket.gethostname().upper()
    assert "EVO-T2S" in hostname or "EVO" in hostname, \
        f"WRONG HOST: {hostname}. This script must run on EVO-T2S."

    # Live build_id from /props
    props = _query_props()
    build_id = props.get("build", "UNKNOWN")
    model_field = props.get("default_generation_settings", {}).get("model", "UNKNOWN")
    chat_template = props.get("chat_template", "UNKNOWN")[:80]
    print(f"HOSTNAME : {hostname}")
    print(f"BUILD    : {build_id}")
    print(f"MODEL    : {model_field}")
    print(f"TEMPLATE : {chat_template}")
    print()

    scorers = _load_scorers()

    # Load segments (all 21) then filter to stage C set
    all_segs = [
        json.loads(l)
        for l in (PROBES_DIR / "segments.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    seg_by_id = {s["id"]: s for s in all_segs}
    probes = [seg_by_id[pid] for pid in PROBE_IDS if pid in seg_by_id]
    missing = [pid for pid in PROBE_IDS if pid not in seg_by_id]
    if missing:
        raise RuntimeError(f"Probes not in segments.jsonl: {missing}")

    print(f"Probes   : {len(probes)} ({[p['id'] for p in probes]})")
    print(f"Building filler ({FILLER_TARGET} tokens)…")
    t_filler_start = time.perf_counter()
    filler, filler_tokens = _build_filler(FILLER_TARGET)
    print(f"  filler_tokens={filler_tokens} ({time.perf_counter() - t_filler_start:.1f}s)")
    print()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if smoke:
        out_path = RESULTS_DIR / f"fig61_stagec_smoke_{ts}.jsonl"
        run_name = f"stagec_smoke_{ts}"
    elif gate or gate_only:
        out_path = RESULTS_DIR / f"fig61_stagec_gate_{ts}.jsonl"
        run_name = f"stagec_gate_{ts}"
    else:
        out_path = RESULTS_DIR / f"fig61_stagec_full_{ts}.jsonl"
        run_name = f"stagec_full_{ts}"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Which ratios/reps to run
    if smoke:
        probe_subset = [probes[0]]      # first probe only
        ratio_list = BUDGET_RATIOS
        n_reps = 1
        print(f"SMOKE TEST: 1 probe × {len(ratio_list)} ratios × 2 arms × 1 rep = "
              f"{len(ratio_list)*2} calls")
    elif gate or gate_only:
        probe_subset = probes
        ratio_list = [1.20]             # baseline only
        n_reps = 1
        print(f"BASELINE GATE: {len(probe_subset)} probes × 1 ratio × 2 arms × 1 rep = "
              f"{len(probe_subset)*2} calls")
    else:
        probe_subset = probes
        ratio_list = BUDGET_RATIOS
        n_reps = N_REPS
        total = len(probe_subset) * len(ratio_list) * len(ARMS) * n_reps
        print(f"FULL RUN: {len(probe_subset)} probes × {len(ratio_list)} ratios × "
              f"2 arms × {n_reps} reps = {total} calls")

    rows = []
    call_n = 0

    # Precompute full token counts per probe
    full_tokens_cache: dict[str, int] = {}

    with open(out_path, "w", encoding="utf-8") as fout:
        for ratio in ratio_list:
            for probe in probe_subset:
                pid = probe["id"]
                artifact = probe["artifact"].strip()
                question = probe["question"].strip()
                expected = probe["expected"]
                scorer_type = probe["scorer_type"]

                # Build full prompt per arm once (for token count)
                if pid not in full_tokens_cache:
                    full_late = build_arm_prompt("LATE", filler, artifact, question)
                    full_tokens = _tokenize(full_late)
                    full_tokens_cache[pid] = full_tokens
                else:
                    full_tokens = full_tokens_cache[pid]

                target_tokens = round(full_tokens * ratio)
                truncating = ratio < 1.0
                intended = min(target_tokens, full_tokens)

                for arm in ARMS:
                    for rep in range(n_reps):
                        call_n += 1
                        t_setup = time.perf_counter_ns()

                        full_prompt = build_arm_prompt(arm, filler, artifact, question)

                        # Left-char truncation (char-based, matches stage C)
                        if truncating:
                            prompt = left_truncate(full_prompt, full_tokens, target_tokens)
                            chars_dropped = len(full_prompt) - len(prompt)
                        else:
                            prompt = full_prompt
                            chars_dropped = 0

                        # artifact fraction retained for EARLY arm
                        art_len = len(artifact)
                        if arm == "EARLY" and truncating and art_len > 0:
                            art_frac = max(0.0, 1.0 - min(chars_dropped, art_len) / art_len)
                        else:
                            art_frac = 1.0

                        orch_setup_ns = time.perf_counter_ns() - t_setup
                        t_http = time.perf_counter_ns()

                        try:
                            output, latency_ms, ttft_ms, tokens_in, tokens_out, finish_reason = \
                                _chat_streaming(prompt)
                            error = None
                        except Exception as e:
                            output = ""
                            latency_ms = 0.0
                            ttft_ms = None
                            tokens_in = 0
                            tokens_out = 0
                            finish_reason = None
                            error = str(e)

                        http_client_ns = time.perf_counter_ns() - t_http

                        # Positive control
                        effective = intended
                        dev = abs(tokens_in - effective) / max(effective, 1)
                        pc_ok = dev <= 0.05

                        # Score
                        try:
                            score, score_detail = scorers.score(
                                scorer_type, output, expected
                            )
                        except Exception as e:
                            score = None
                            score_detail = f"scorer_error:{e}"

                        row = {
                            "probe_id": pid,
                            "category": probe["category"],
                            "difficulty": probe.get("difficulty", ""),
                            "arm": arm,
                            "rep": rep,
                            "run": run_name,
                            "budget_ratio": ratio,
                            "target_tokens": target_tokens,
                            "full_tokens": full_tokens,
                            "filler_tokens": filler_tokens,
                            "truncating": truncating,
                            "truncation_method": "left_char" if truncating else "none",
                            "chars_dropped": chars_dropped,
                            "artifact_fraction_retained": round(art_frac, 4),
                            "score": score,
                            "score_detail": score_detail,
                            "output": output,
                            "latency_ms": round(latency_ms, 1),
                            "ttft_ms": round(ttft_ms, 1) if ttft_ms else None,
                            "orch_setup_ns": orch_setup_ns,
                            "http_client_ns": http_client_ns,
                            "tool_compute_ns": 0,
                            "n_prompt_tokens_actual": tokens_in,
                            "n_ctx_slot": N_CTX_SLOT,
                            "intended_budget_tokens": intended,
                            "positive_control_ok": pc_ok,
                            "tokens_out": tokens_out,
                            "finish_reason": finish_reason,
                            "count_method": "llamaserver_tokenize",
                            "model": "qwen3:4b-instruct",
                            "build_id": build_id,
                            "platform": PLATFORM,
                            "hardware_config": HW_CONFIG,
                            "memory_architecture": MEM_ARCH,
                            "error": error,
                        }
                        rows.append(row)
                        fout.write(json.dumps(row) + "\n")
                        fout.flush()

                        pc_flag = "PC+" if pc_ok else "PC-"
                        print(f"  [{call_n:3d}] {pid} {arm} r={ratio:.2f} rep={rep} "
                              f"score={score} tokens_in={tokens_in} "
                              f"ttft={ttft_ms:.0f}ms lat={latency_ms:.0f}ms "
                              f"{pc_flag} finish={finish_reason}")

    print(f"\nWrote {len(rows)} rows -> {out_path.name}")
    return rows, out_path


def run_gate_check(gate_rows, stage_c_path: Path):
    """Compare baseline gate (ratio=1.20) to committed stage C scores.
    Returns (n_disagree, disagreement_list).
    """
    sc_rows = [
        json.loads(l)
        for l in stage_c_path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

    # stage C baseline at ratio=1.20, mean per (probe_id, arm)
    from collections import defaultdict
    sc_cells: dict[tuple, list] = defaultdict(list)
    for r in sc_rows:
        if abs(r["budget_ratio"] - 1.20) < 0.01:
            sc_cells[(r["probe_id"], r["arm"])].append(r["score"])
    sc_mean = {k: sum(v) / len(v) for k, v in sc_cells.items() if v}

    # gate rows (1 rep at ratio=1.20)
    disagreements = []
    for r in gate_rows:
        key = (r["probe_id"], r["arm"])
        sc_val = sc_mean.get(key)
        if sc_val is None:
            continue
        new_val = r["score"]
        disagree = abs((new_val or 0.0) - sc_val) > 0.5
        if disagree:
            disagreements.append({
                "probe_id": r["probe_id"],
                "arm": r["arm"],
                "stage_c_mean": round(sc_val, 3),
                "evo_t2s": new_val,
                "output": r.get("output", "")[:200],
            })

    return len(disagreements), disagreements


def run_smoke_check(smoke_rows):
    """Verify: positive control passes, EARLY loses artifact at truncating ratios."""
    pc_fails = [r for r in smoke_rows if not r["positive_control_ok"]]
    # EARLY artifact fraction should be ~0 at ratio <= 0.55 (for ~4300-token prompts)
    early_art_ok = all(
        r["artifact_fraction_retained"] < 0.1
        for r in smoke_rows
        if r["arm"] == "EARLY" and r["budget_ratio"] <= 0.55
    )
    late_art_ok = all(
        r["artifact_fraction_retained"] == 1.0
        for r in smoke_rows
        if r["arm"] == "LATE"
    )
    return pc_fails, early_art_ok, late_art_ok


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()

    STAGE_C_PATH = RESULTS_DIR / "stage_c_20260818T040408Z.jsonl"

    if args.smoke:
        smoke_rows, smoke_path = main(smoke=True)
        pc_fails, early_ok, late_ok = run_smoke_check(smoke_rows)
        print(f"\nSMOKE RESULTS:")
        print(f"  PC failures: {len(pc_fails)}")
        print(f"  EARLY loses artifact at low ratios: {early_ok}")
        print(f"  LATE retains artifact: {late_ok}")
        if pc_fails:
            for r in pc_fails:
                print(f"    PC FAIL: {r['probe_id']} {r['arm']} r={r['budget_ratio']} "
                      f"intended={r['intended_budget_tokens']} actual={r['n_prompt_tokens_actual']}")
        smoke_ok = len(pc_fails) == 0 and early_ok and late_ok
        print(f"\nSMOKE {'PASS' if smoke_ok else 'FAIL'}")

    elif args.gate:
        gate_rows, gate_path = main(gate=True)
        n_disagree, disagreements = run_gate_check(gate_rows, STAGE_C_PATH)
        print(f"\nGATE RESULTS: {n_disagree}/22 cells disagree with stage C at ratio=1.20")
        for d in disagreements:
            print(f"  DISAGREE: {d['probe_id']} {d['arm']} stage_C={d['stage_c_mean']} "
                  f"evo_t2s={d['evo_t2s']}")
            print(f"    output: {repr(d['output'][:120])}")
        if n_disagree > 4:
            print(f"\nGATE FAIL: {n_disagree} > 4 disagreements. STOP — do not run full sweep.")
            sys.exit(1)
        else:
            print(f"\nGATE PASS ({n_disagree} disagreements ≤ 4). OK to run full sweep.")

    elif args.full:
        rows, full_path = main(smoke=False, gate=False, gate_only=False)
        print(f"\nFull sweep complete: {len(rows)} rows written to {full_path.name}")
