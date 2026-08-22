"""Stage 1 — second-model replication of truncation failure mode.

llama3.1:8b on the same EARLY-arm truncation design as Stage C.
5 probes x 4 ratios x 2 sub-arms x 3 reps = 120 calls.

Token counts are measured with llama3.1:8b's own tokenizer (different from
Qwen3-4B). All other experimental parameters are identical to Stage C so the
comparison is clean.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).parent.parent
PROBES_DIR = REPO / "evaluation" / "probes"
RESULTS = REPO / "results"
sys.path.insert(0, str(REPO / "harness"))

import context as ctx_mod

MODEL = "llama3.1:8b"
HOST = "http://localhost:11434"
FILLER_TARGET = 4000
TARGET_PROBES = ["rag_01", "rag_02", "rag_05", "sea_01", "sea_04"]
BUDGET_RATIOS = [1.20, 0.85, 0.70, 0.40]
N_REPS = 3
MAX_TOKENS = 256  # llama3.1:8b tends toward longer outputs

ARM3_SUFFIX = (
    "\nFirst state whether the information needed to answer is present above,"
    " then answer. Format: AVAILABLE: yes|no, then the answer."
)


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


def _call(prompt: str) -> tuple[str | None, float]:
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"num_ctx": 8192, "num_predict": MAX_TOKENS, "temperature": 0},
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{HOST}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            data = json.loads(r.read())
        return data["message"]["content"].strip(), time.monotonic() - t0
    except Exception as e:
        return None, time.monotonic() - t0


def left_truncate(prompt: str, full_tokens: int, target_tokens: int) -> str:
    if target_tokens >= full_tokens:
        return prompt
    chars_to_keep = round(len(prompt) * target_tokens / full_tokens)
    chars_to_keep = max(1, min(chars_to_keep, len(prompt)))
    return prompt[len(prompt) - chars_to_keep:]


def classify_available(output: str) -> str:
    t = output.lower()
    if "available: no" in t or "available:no" in t:
        return "no"
    if "available: yes" in t or "available:yes" in t:
        return "yes"
    return "absent"


def fabrication_source(output: str | None, truncated_prompt: str) -> str:
    if output is None:
        return "error"
    normed = output.strip()
    if not normed:
        return "error"
    tp = truncated_prompt.lower()
    if normed.lower() in tp:
        return "verbatim_in_context"
    tokens = [t for t in normed.split() if len(t) >= 4]
    if tokens and any(t.lower() in tp for t in tokens):
        return "partial_in_context"
    return "prior_only"


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    spec = importlib.util.spec_from_file_location(
        "scorers", PROBES_DIR / "scorers.py"
    )
    scorers_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorers_mod)

    segments = {
        s["id"]: s
        for s in [
            json.loads(l)
            for l in (PROBES_DIR / "segments.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        if s["id"] in TARGET_PROBES
    }

    print(f"Stage 1 — Second-model replication")
    print(f"Model : {MODEL}")
    print(f"Probes: {TARGET_PROBES}")
    print(f"Ratios: {BUDGET_RATIOS}")
    print()

    print("Building F-NUM filler with llama3.1:8b tokenizer...")
    filler = ctx_mod.build_filler(FILLER_TARGET, seed=42, count_fn=_count_fn, variant="F-NUM")
    filler_tokens = _count_fn(filler)
    print(f"Filler: {filler_tokens} tokens ({len(filler)} chars)")
    print()

    print("Measuring full EARLY prompt token counts (llama3.1:8b tokenizer)...")
    full_tokens_by_probe: dict[str, int] = {}
    for pid in TARGET_PROBES:
        seg = segments[pid]
        early = f"{seg['artifact']}\n\n{filler}\n\n{seg['question']}"
        ft = _count_fn(early)
        full_tokens_by_probe[pid] = ft
        print(f"  {pid}: {ft} tokens")
    print()

    total_calls = len(TARGET_PROBES) * len(BUDGET_RATIOS) * 2 * N_REPS
    print(f"Total calls: {total_calls}")
    print()

    rows: list[dict] = []
    call_n = 0

    for ratio in BUDGET_RATIOS:
        print(f"--- ratio {ratio:.2f} ---")
        for pid in TARGET_PROBES:
            seg = segments[pid]
            full_tokens = full_tokens_by_probe[pid]
            target_tokens = round(full_tokens * ratio)
            truncating = target_tokens < full_tokens

            for arm_name, question in [
                ("arm1_baseline", seg["question"]),
                ("arm3_self_report", seg["question"] + ARM3_SUFFIX),
            ]:
                full_prompt = f"{seg['artifact']}\n\n{filler}\n\n{question}"
                truncated = left_truncate(full_prompt, full_tokens, target_tokens)
                chars_dropped = len(full_prompt) - len(truncated)

                for rep in range(N_REPS):
                    call_n += 1
                    output, latency = _call(truncated)

                    probe_dict = {
                        "id": pid,
                        "scorer_type": seg["scorer_type"],
                        "expected": seg["expected"],
                    }
                    score, score_detail = scorers_mod.score(probe_dict, output)
                    outcome = scorers_mod.outcome_class(output, score, truncated=truncating)

                    fab_src = fabrication_source(output, truncated) if outcome == "incorrect" else None

                    row: dict = {
                        "probe_id": pid,
                        "model": MODEL,
                        "model_variant": "instruct",
                        "arm": arm_name,
                        "budget_ratio": ratio,
                        "target_tokens": target_tokens,
                        "full_tokens": full_tokens,
                        "truncating": truncating,
                        "chars_dropped": chars_dropped,
                        "rep": rep,
                        "output": output,
                        "score": score,
                        "score_detail": score_detail,
                        "outcome": outcome,
                        "fabrication_source": fab_src,
                        "latency_s": round(latency, 3),
                        "hardware": "blade_rtx4070",
                    }
                    if arm_name == "arm3_self_report":
                        row["available_field"] = classify_available(output or "")

                    rows.append(row)

                    status = "✓" if score == 1.0 else ("A" if outcome == "abstained" else "✗")
                    avail = row.get("available_field", "")
                    print(
                        f"[{call_n:3d}/{total_calls}] {pid} {arm_name[:5]}"
                        f" r={ratio:.2f} rep={rep} {status}"
                        f" av={avail:3s} {latency*1000:.0f}ms"
                        f"  {repr((output or '')[:60])}"
                    )

    out = RESULTS / "model2_truncation.json"
    out.write_text(json.dumps({"model": MODEL, "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nWritten {len(rows)} rows to {out}")

    # Quick summary
    from collections import defaultdict
    print("\n=== Score by probe × ratio (arm1 baseline) ===")
    cell: dict = defaultdict(list)
    for r in rows:
        if r["arm"] == "arm1_baseline" and r.get("score") is not None:
            cell[(r["probe_id"], r["budget_ratio"])].append(r["score"])
    for pid in TARGET_PROBES:
        vals = [f"r={rt:.2f}:{sum(cell[(pid,rt)])/len(cell[(pid,rt)]):.2f}" for rt in BUDGET_RATIOS if cell[(pid,rt)]]
        print(f"  {pid}: {' | '.join(vals)}")

    print("\n=== Outcome breakdown (truncated rows, arm1 baseline) ===")
    trunc = [r for r in rows if r["truncating"] and r["arm"] == "arm1_baseline" and r.get("score") is not None]
    n = len(trunc)
    fab = sum(1 for r in trunc if r["outcome"] == "incorrect")
    ab = sum(1 for r in trunc if r["outcome"] == "abstained")
    cor = sum(1 for r in trunc if r["outcome"] == "correct")
    print(f"  n={n}  fabrication={fab} ({fab/n:.1%})  abstained={ab} ({ab/n:.1%})  correct={cor}")

    print("\n=== Arm3 AVAILABLE field (truncated rows) ===")
    a3_trunc = [r for r in rows if r["truncating"] and r["arm"] == "arm3_self_report"]
    yes = sum(1 for r in a3_trunc if r.get("available_field") == "yes")
    no = sum(1 for r in a3_trunc if r.get("available_field") == "no")
    absent = sum(1 for r in a3_trunc if r.get("available_field") == "absent")
    print(f"  yes={yes}  no={no}  absent={absent}")
    yes_fab = sum(1 for r in a3_trunc if r.get("available_field") == "yes" and r["outcome"] == "incorrect")
    no_ans = sum(1 for r in a3_trunc if r.get("available_field") == "no" and r["outcome"] == "incorrect")
    print(f"  AVAILABLE:yes+fabricate: {yes_fab}/{yes}")
    print(f"  AVAILABLE:no+answers: {no_ans}/{no}")


if __name__ == "__main__":
    main()
