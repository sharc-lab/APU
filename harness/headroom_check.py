"""Deletion + headroom checks for art_* probes.

For each probe in evaluation/probes/artifact.jsonl:

  Deletion check (NEGATIVE): run with artifact region removed.
    Prompt: <filler>\\n\\n<question>
    Both models, 3 reps. Score must be 0.0 in all reps.
    If any rep scores > 0.0, the probe is artifact-independent and must be
    redesigned — reported explicitly.

  Headroom check (POSITIVE): run full EARLY arm at r=1.20 (no truncation).
    Prompt: <artifact>\\n\\n<filler>\\n\\n<question>
    Both models, 3 reps. Probe passes headroom iff all non-None reps score 1.0.

Results written to results/art_headroom.json.
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

MODELS = ["qwen3:4b-instruct", "llama3.1:8b"]
HOST = "http://localhost:11434"
FILLER_TARGET = 4000
N_REPS = 3
MAX_TOKENS = 128


def _count_fn(text: str, model: str = "qwen3:4b-instruct") -> int:
    payload = json.dumps({
        "model": model,
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


def _call(prompt, model):
    extra = {}
    if "qwen" in model:
        extra["think"] = False
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"num_ctx": 8192, "num_predict": MAX_TOKENS, "temperature": 0},
        "stream": False,
        **extra,
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
        content = data["message"]["content"].strip()
        return content, time.monotonic() - t0, data.get("eval_count"), data.get("prompt_eval_count"), data.get("done_reason")
    except Exception as e:
        print(f"    [CALL ERROR: {e}]")
        return None, time.monotonic() - t0, None, None, None


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    spec = importlib.util.spec_from_file_location("scorers", PROBES_DIR / "scorers.py")
    scorers_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorers_mod)

    art_probes = [
        json.loads(l)
        for l in (PROBES_DIR / "artifact.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    segments = {
        s["id"]: s
        for s in [
            json.loads(l)
            for l in (PROBES_DIR / "segments.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        if s["id"].startswith("art_")
    }

    print("Building F-NUM filler...")
    filler = ctx_mod.build_filler(FILLER_TARGET, seed=42, count_fn=_count_fn, variant="F-NUM")
    filler_tokens = _count_fn(filler)
    print(f"Filler: {filler_tokens} tokens ({len(filler)} chars)")
    print()

    probe_ids = [p["id"] for p in art_probes]
    total_calls = len(probe_ids) * 2 * len(MODELS) * N_REPS
    print(f"Probes  : {probe_ids}")
    print(f"Models  : {MODELS}")
    print(f"N_reps  : {N_REPS}")
    print(f"Calls   : {total_calls} (2 check types × 2 models × 3 reps × 10 probes)")
    print()

    rows: list[dict] = []
    call_n = 0

    for pid in probe_ids:
        seg = segments[pid]
        probe_dict = {
            "id": pid,
            "scorer_type": seg["scorer_type"],
            "expected": seg["expected"],
        }
        headroom_prompt = f"{seg['artifact']}\n\n{filler}\n\n{seg['question']}"
        deletion_prompt = f"{filler}\n\n{seg['question']}"

        for check_type, prompt in [("deletion", deletion_prompt), ("headroom", headroom_prompt)]:
            for model in MODELS:
                for rep in range(N_REPS):
                    call_n += 1
                    output, latency, eval_count, prompt_eval_count, done_reason = _call(prompt, model)
                    score, score_detail = (
                        scorers_mod.score(probe_dict, output) if output is not None
                        else (None, "no output")
                    )
                    row = {
                        "probe_id": pid,
                        "check_type": check_type,
                        "model": model,
                        "rep": rep,
                        "output": output,
                        "score": score,
                        "score_detail": score_detail,
                        "eval_count": eval_count,
                        "prompt_eval_count": prompt_eval_count,
                        "done_reason": done_reason,
                        "latency_s": round(latency, 3),
                        "hardware": "blade14_rtx4070",
                    }
                    rows.append(row)
                    status = "✓" if score == 1.0 else ("✗" if score is not None else "E")
                    print(
                        f"[{call_n:3d}/{total_calls}] {pid} {check_type} {model[:14]} r{rep}"
                        f" {status} {latency*1000:.0f}ms  {repr((output or '')[:50])}"
                    )

    # Analysis
    deletion_failures: list[dict] = []
    headroom_results: dict[str, dict[str, bool]] = {}

    for pid in probe_ids:
        headroom_results[pid] = {}
        for model in MODELS:
            del_rows = [r for r in rows if r["probe_id"] == pid
                        and r["check_type"] == "deletion" and r["model"] == model]
            head_rows = [r for r in rows if r["probe_id"] == pid
                         and r["check_type"] == "headroom" and r["model"] == model]

            del_scores = [r["score"] for r in del_rows if r["score"] is not None]
            head_scores = [r["score"] for r in head_rows if r["score"] is not None]

            del_any_correct = any(s == 1.0 for s in del_scores)
            head_all_correct = bool(head_scores) and all(s == 1.0 for s in head_scores)

            headroom_results[pid][model] = head_all_correct
            if del_any_correct:
                deletion_failures.append({
                    "probe_id": pid,
                    "model": model,
                    "deletion_scores": del_scores,
                    "deletion_outputs": [r["output"] for r in del_rows],
                })

    # Valid cross-model set: both models pass headroom
    valid_set = [pid for pid in probe_ids
                 if all(headroom_results[pid].get(m) for m in MODELS)]

    analysis = {
        "deletion_failures": deletion_failures,
        "deletion_failure_count": len(deletion_failures),
        "headroom_by_probe": headroom_results,
        "valid_crossmodel_set": valid_set,
        "valid_crossmodel_count": len(valid_set),
    }

    out = RESULTS / "art_headroom.json"
    out.write_text(json.dumps({"rows": rows, "analysis": analysis}, indent=2), encoding="utf-8")
    print(f"\nWritten {len(rows)} rows to {out}")

    print("\n=== DELETION CHECK (must all be ✗) ===")
    for pid in probe_ids:
        for model in MODELS:
            del_scores = [r["score"] for r in rows if r["probe_id"] == pid
                          and r["check_type"] == "deletion" and r["model"] == model
                          and r["score"] is not None]
            any_pass = any(s == 1.0 for s in del_scores)
            tag = "FAIL — redesign needed" if any_pass else "ok"
            print(f"  {pid} {model[:20]}: {del_scores}  {tag}")

    print("\n=== HEADROOM CHECK (must all be ✓) ===")
    for pid in probe_ids:
        row_parts = []
        for model in MODELS:
            passes = headroom_results[pid].get(model, False)
            row_parts.append(f"{model[:14]}: {'PASS' if passes else 'FAIL'}")
        print(f"  {pid}: {' | '.join(row_parts)}")

    print(f"\n=== VALID CROSS-MODEL SET ({len(valid_set)}/{len(probe_ids)}) ===")
    print(f"  {valid_set}")
    if deletion_failures:
        print(f"\n!!! DELETION FAILURES ({len(deletion_failures)}) — probes need redesign:")
        for f in deletion_failures:
            print(f"  {f['probe_id']} {f['model']}: {f['deletion_outputs']}")


if __name__ == "__main__":
    main()
