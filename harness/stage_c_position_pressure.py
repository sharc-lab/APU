"""Stage C — Position under memory pressure.

Two arms, fixed 4000-token filler, 6 budget ratios × 11 probes × 2 arms × 3 reps.
LATE (control):    [filler][artifact][question]
EARLY (treatment): [artifact][filler][question]

Budget-major loop order: ratio → probe → arm → rep.

Gate 2 discovery: attempt the first truncating call via num_ctx lever. If Ollama
returns HTTP 400 (num_ctx < prompt length), switch to harness-side left-char
truncation for all truncating calls. Gate 2 result is recorded in gate2 JSON.

Physical left-truncation: chars dropped proportionally from the start of the
full prompt string, so LATE arm loses filler first and EARLY arm loses artifact
first — the same logical effect as model-side context truncation.

Writes results/stage_c_<timestamp>.jsonl and results/stage_c_<timestamp>_gate2.json.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).parent.parent
PROBES_DIR = REPO / "evaluation" / "probes"
RESULTS = REPO / "results"

sys.path.insert(0, str(REPO / "harness"))
import context as ctx_mod

MODEL = "qwen3:4b-instruct"
HOST = "http://localhost:11434"
FILLER_TARGET = 4000
BUDGET_RATIOS = [1.20, 1.00, 0.85, 0.70, 0.55, 0.40]
ARMS = ["LATE", "EARLY"]
N_REPS = 3
MAX_TOKENS = 128

HW_CONFIG = "blade14_rtx4070"
MEM_ARCH = "discrete"


def _load_scorers():
    spec = importlib.util.spec_from_file_location(
        "probes_scorers", PROBES_DIR / "scorers.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _count_fn(text: str) -> int:
    payload = json.dumps({
        "model": MODEL,
        "prompt": text,
        "options": {"num_ctx": 8192, "num_predict": 1, "temperature": 0},
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{HOST}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["prompt_eval_count"]


def _call_model_ctx(prompt: str, num_ctx: int) -> tuple[str, float]:
    """Call model with explicit num_ctx (may error if num_ctx < prompt length)."""
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "options": {
            "num_ctx": num_ctx,
            "num_predict": MAX_TOKENS,
            "temperature": 0,
        },
        "stream": False,
        "think": False,
    }).encode()
    req = urllib.request.Request(
        f"{HOST}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read())
    latency = time.monotonic() - t0
    return data["message"]["content"].strip(), latency


def _call_model_safe(prompt: str) -> tuple[str, float]:
    """Call model with num_ctx=8192, safe for any prompt up to ~8k tokens."""
    return _call_model_ctx(prompt, 8192)


def build_arm_prompt(arm: str, filler: str, artifact: str, question: str) -> str:
    if arm == "LATE":
        return f"{filler}\n\n{artifact}\n\n{question}"
    else:  # EARLY
        return f"{artifact}\n\n{filler}\n\n{question}"


def left_truncate(prompt: str, full_tokens: int, target_tokens: int) -> str:
    """Drop chars from the left proportionally to the token reduction."""
    if target_tokens >= full_tokens:
        return prompt
    chars_to_keep = round(len(prompt) * target_tokens / full_tokens)
    chars_to_keep = max(1, min(chars_to_keep, len(prompt)))
    return prompt[len(prompt) - chars_to_keep:]


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    scorers = _load_scorers()

    segments = [
        json.loads(l)
        for l in (PROBES_DIR / "segments.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    print("Stage C — Position under Memory Pressure")
    print(f"Model  : {MODEL}")
    print(f"Probes : {len(segments)}  Arms: {len(ARMS)}  Reps: {N_REPS}")
    print(f"Ratios : {BUDGET_RATIOS}")
    print(f"Filler target: {FILLER_TARGET} tokens")
    print()

    # Build filler once
    print("Building filler (4000 tokens, iterative calibration)...")
    filler = ctx_mod.build_filler(FILLER_TARGET, seed=42, count_fn=_count_fn)
    filler_tokens = _count_fn(filler)
    print(f"Filler: {filler_tokens} tokens ({len(filler)} chars)")
    print()

    # Build full prompts and measure token counts per probe×arm
    print("Measuring full prompt token counts per probe...")
    for seg in segments:
        late_p = build_arm_prompt("LATE", filler, seg["artifact"], seg["question"])
        early_p = build_arm_prompt("EARLY", filler, seg["artifact"], seg["question"])
        seg["late_prompt"] = late_p
        seg["early_prompt"] = early_p
        seg["late_tokens"] = _count_fn(late_p)
        # EARLY has same content, just reordered — should be same count ±2
        seg["full_tokens"] = seg["late_tokens"]
        seg["full_chars"] = len(late_p)
        print(f"  {seg['id']}: {seg['full_tokens']} tokens ({seg['full_chars']} chars)")
    print()

    total_calls = len(BUDGET_RATIOS) * len(segments) * len(ARMS) * N_REPS
    print(f"Total calls: {total_calls}")
    print()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS / f"stage_c_{ts}.jsonl"
    gate2_path = RESULTS / f"stage_c_{ts}_gate2.json"
    print(f"Output: {out_path}")
    print()

    gate2 = {
        "filler_tokens": filler_tokens,
        "probe_full_tokens": {s["id"]: s["full_tokens"] for s in segments},
        "gate2_result": None,
        "truncation_method": None,
        "gate2_probe": None,
        "gate2_ratio": None,
        "gate2_num_ctx": None,
        "gate2_full_tokens": None,
        "gate2_detail": None,
    }
    truncation_method = None  # set after Gate 2

    rows = []
    call_n = 0
    gate2_checked = False

    for ratio in BUDGET_RATIOS:
        print(f"\n{'='*60}")
        print(f"Budget ratio: {ratio:.2f}")
        print(f"{'='*60}")

        for seg in segments:
            full_tokens = seg["full_tokens"]
            target_tokens = round(full_tokens * ratio)
            truncating = target_tokens < full_tokens

            # Gate 2: run on first truncating ratio, first probe, LATE arm
            if truncating and not gate2_checked:
                gate2_checked = True
                gate2["gate2_probe"] = seg["id"]
                gate2["gate2_ratio"] = ratio
                gate2["gate2_full_tokens"] = full_tokens
                gate2["gate2_num_ctx"] = target_tokens

                print(f"  [GATE 2] Testing num_ctx lever: num_ctx={target_tokens} "
                      f"vs full_tokens={full_tokens}...")
                try:
                    _, _ = _call_model_ctx(seg["late_prompt"], target_tokens)
                    gate2["gate2_result"] = "TRUNCATED_SILENTLY"
                    gate2["truncation_method"] = "model_context"
                    truncation_method = "model_context"
                    gate2["gate2_detail"] = "Ollama accepted num_ctx < prompt length; model-side truncation active"
                    print(f"  [GATE 2] PASS — model-side truncation accepted")
                except urllib.error.HTTPError as e:
                    gate2["gate2_result"] = "ERRORS"
                    gate2["truncation_method"] = "harness_left_char"
                    truncation_method = "harness_left_char"
                    gate2["gate2_detail"] = (
                        f"HTTP {e.code}: Ollama rejects num_ctx < prompt length. "
                        f"Falling back to harness-side left-char truncation."
                    )
                    print(f"  [GATE 2] HTTP {e.code} — switching to harness-side left-char truncation")

            for arm in ARMS:
                full_prompt = seg["late_prompt"] if arm == "LATE" else seg["early_prompt"]

                for rep in range(N_REPS):
                    call_n += 1

                    # Build the prompt to actually send
                    if not truncating or truncation_method is None:
                        prompt_sent = full_prompt
                        num_ctx_used = 8192
                        chars_dropped = 0
                    elif truncation_method == "model_context":
                        prompt_sent = full_prompt
                        num_ctx_used = target_tokens
                        chars_dropped = 0
                    else:  # harness_left_char
                        prompt_sent = left_truncate(full_prompt, full_tokens, target_tokens)
                        num_ctx_used = 8192
                        chars_dropped = len(full_prompt) - len(prompt_sent)

                    try:
                        if truncation_method == "model_context" and truncating:
                            output, latency = _call_model_ctx(prompt_sent, num_ctx_used)
                        else:
                            output, latency = _call_model_safe(prompt_sent)

                        probe_dict = {
                            "id": seg["id"],
                            "scorer_type": seg["scorer_type"],
                            "expected": seg["expected"],
                        }
                        score_val, score_detail = scorers.score(probe_dict, output)

                        row = {
                            "probe_id": seg["id"],
                            "category": seg["category"],
                            "difficulty": seg["difficulty"],
                            "arm": arm,
                            "budget_ratio": ratio,
                            "target_tokens": target_tokens,
                            "full_tokens": full_tokens,
                            "filler_tokens": filler_tokens,
                            "artifact_tokens": seg["artifact_tokens"],
                            "question_tokens": seg["question_tokens"],
                            "truncating": truncating,
                            "truncation_method": truncation_method if truncating else "none",
                            "chars_dropped": chars_dropped,
                            "rep": rep,
                            "output": output,
                            "score": score_val,
                            "score_detail": score_detail,
                            "latency_s": round(latency, 3),
                            "model": MODEL,
                            "hardware_config": HW_CONFIG,
                            "memory_architecture": MEM_ARCH,
                            "run": ts,
                        }
                        rows.append(row)

                        status = "✓" if score_val == 1.0 else ("✗" if score_val == 0.0 else f"{score_val:.2f}")
                        trunc_flag = "T" if truncating else " "
                        print(f"[{call_n:4d}/{total_calls}] {seg['id']} {arm} r={ratio:.2f} "
                              f"rep={rep} [{trunc_flag}] {status} {latency*1000:.0f}ms")

                    except Exception as e:
                        err_msg = str(e)
                        print(f"[{call_n:4d}/{total_calls}] {seg['id']} {arm} r={ratio:.2f} "
                              f"rep={rep} ERROR: {err_msg[:80]}")
                        rows.append({
                            "probe_id": seg["id"],
                            "category": seg["category"],
                            "difficulty": seg["difficulty"],
                            "arm": arm,
                            "budget_ratio": ratio,
                            "target_tokens": target_tokens,
                            "full_tokens": full_tokens,
                            "filler_tokens": filler_tokens,
                            "artifact_tokens": seg["artifact_tokens"],
                            "question_tokens": seg["question_tokens"],
                            "truncating": truncating,
                            "truncation_method": truncation_method if truncating else "none",
                            "chars_dropped": 0,
                            "rep": rep,
                            "output": None,
                            "score": None,
                            "score_detail": f"ERROR: {err_msg}",
                            "latency_s": None,
                            "model": MODEL,
                            "hardware_config": HW_CONFIG,
                            "memory_architecture": MEM_ARCH,
                            "run": ts,
                        })

    # Write outputs
    out_path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    gate2_path.write_text(json.dumps(gate2, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"Stage C complete: {len(rows)} rows → {out_path.name}")
    print(f"Gate 2: {gate2['gate2_result']} — {gate2['truncation_method']}")

    # Summary table
    print("\nMean score by arm × ratio (all probes):")
    print(f"{'Ratio':>8}  {'LATE':>8}  {'EARLY':>8}  {'delta':>8}")
    cell = defaultdict(list)
    for r in rows:
        if r["score"] is not None:
            cell[(r["budget_ratio"], r["arm"])].append(r["score"])
    for ratio in BUDGET_RATIOS:
        late = cell[(ratio, "LATE")]
        early = cell[(ratio, "EARLY")]
        lm = sum(late) / len(late) if late else float("nan")
        em = sum(early) / len(early) if early else float("nan")
        print(f"{ratio:>8.2f}  {lm:>8.3f}  {em:>8.3f}  {em-lm:>+8.3f}")

    # Per-family breakdown
    print("\nMean score by family × arm × ratio:")
    for family in ("rag", "sea"):
        print(f"\n  {family.upper()}:")
        print(f"  {'Ratio':>8}  {'LATE':>8}  {'EARLY':>8}")
        fcell = defaultdict(list)
        for r in rows:
            if r["score"] is not None and r["probe_id"].startswith(family):
                fcell[(r["budget_ratio"], r["arm"])].append(r["score"])
        for ratio in BUDGET_RATIOS:
            late = fcell[(ratio, "LATE")]
            early = fcell[(ratio, "EARLY")]
            lm = sum(late) / len(late) if late else float("nan")
            em = sum(early) / len(early) if early else float("nan")
            print(f"  {ratio:>8.2f}  {lm:>8.3f}  {em:>8.3f}")

    return str(out_path)


if __name__ == "__main__":
    main()
