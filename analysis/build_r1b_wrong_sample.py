"""Builds the stratified 150-row wrong-answer sample for fabrication-label validation (item 6 of the 2026-09-30
instruction): a labeled CSV for spot-checking and a blinded copy with no labels, for three independent raters
(the existing heuristic, hand labels, and an LLM judge queued as x2_llm_judge) to later fill in.

Reconstructs the exact visible (post-truncation) prompt excerpt near the answer from each row's own `chars_dropped`
field and the probe's `artifact` text in evaluation/probes/segments.jsonl -- this is an exact reconstruction, not an
approximation: R1b's truncation (harness/stage_c_position_pressure.py's left_truncate) drops exactly `chars_dropped`
characters from the START of `artifact + filler + question`, so `artifact[chars_dropped:]` is byte-identical to
whatever of the artifact the model actually saw, with no need to rebuild the filler or use a live tokenizer.

Usage: py -3.12 analysis/build_r1b_wrong_sample.py <t2s_jsonl> <x2_jsonl> [--n 150] [--seed 20260930]
Writes: results/labeling/r1b_wrong_sample.csv (with expected/output/labels columns)
        results/labeling/r1b_wrong_sample_blinded.csv (same rows, no expected/output/label columns)
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_segments():
    segs = {}
    path = ROOT / "evaluation" / "probes" / "segments.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        segs[d["id"]] = d
    return segs


def load_wrong_rows(path, machine):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("kind") != "call" or not str(r.get("probe_id", "")).startswith("art_"):
                continue
            if r.get("arm") not in ("arm1_baseline", "arm3_self_report"):
                continue
            if r.get("score") == 1.0:
                continue  # only wrong answers are sampled
            r["_machine"] = machine
            rows.append(r)
    return rows


def visible_excerpt(row, segs):
    probe = segs.get(row["probe_id"])
    if probe is None:
        return ""
    artifact = probe["artifact"]
    chars_dropped = row.get("chars_dropped") or 0
    if chars_dropped >= len(artifact):
        return "[artifact fully truncated -- not visible to the model]"
    return artifact[chars_dropped:]


def stratify_key(row):
    return (row["model_id"], row["budget_ratio"], row["arm"], row["_machine"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("t2s_jsonl")
    ap.add_argument("x2_jsonl")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20260930)
    args = ap.parse_args()

    segs = load_segments()
    all_rows = load_wrong_rows(args.t2s_jsonl, "evo-t2s") + load_wrong_rows(args.x2_jsonl, "evo-x2")
    if not all_rows:
        raise SystemExit("no wrong rows found in either input file")

    by_stratum = {}
    for r in all_rows:
        by_stratum.setdefault(stratify_key(r), []).append(r)

    rng = random.Random(args.seed)
    strata = list(by_stratum.keys())
    rng.shuffle(strata)
    sample = []
    # round-robin across strata so every (model, ratio, arm, machine) combination present in the wrong-answer
    # population gets at least one row before any stratum gets a second, up to args.n total
    pools = {k: list(v) for k, v in by_stratum.items()}
    for pool in pools.values():
        rng.shuffle(pool)
    i = 0
    while len(sample) < args.n and any(pools.values()):
        k = strata[i % len(strata)]
        if pools[k]:
            sample.append(pools[k].pop())
        i += 1
        if i > args.n * 20:  # safety valve if every pool is empty but the while condition raced
            break
    sample = sample[: args.n]

    out_dir = ROOT / "results" / "labeling"
    out_dir.mkdir(parents=True, exist_ok=True)
    full_path = out_dir / "r1b_wrong_sample.csv"
    blind_path = out_dir / "r1b_wrong_sample_blinded.csv"

    fields = ["id", "model", "ratio", "arm", "machine", "question", "expected", "output",
              "visible_prompt_excerpt_near_answer", "heuristic_label"]
    blind_fields = ["id", "model", "ratio", "arm", "machine", "question", "visible_prompt_excerpt_near_answer"]

    sys.path.insert(0, str(ROOT / "evaluation" / "probes"))
    import scorers as sc  # noqa: E402

    with open(full_path, "w", encoding="utf-8", newline="") as f_full, \
         open(blind_path, "w", encoding="utf-8", newline="") as f_blind:
        w_full = csv.DictWriter(f_full, fieldnames=fields)
        w_blind = csv.DictWriter(f_blind, fieldnames=blind_fields)
        w_full.writeheader()
        w_blind.writeheader()
        for idx, r in enumerate(sample):
            probe = segs.get(r["probe_id"], {})
            heuristic = classify_heuristic(r.get("output"))
            row_id = f"r1bw_{idx:03d}_{r['_machine']}_{r['model_id']}_{r['probe_id']}"
            full_row = {
                "id": row_id, "model": r["model_id"], "ratio": r["budget_ratio"], "arm": r["arm"],
                "machine": r["_machine"], "question": probe.get("question", ""), "expected": probe.get("expected", ""),
                "output": r.get("output", ""), "visible_prompt_excerpt_near_answer": visible_excerpt(r, segs),
                "heuristic_label": heuristic,
            }
            w_full.writerow(full_row)
            w_blind.writerow({k: full_row[k] for k in blind_fields})

    print(f"wrote {len(sample)} rows to {full_path} and {blind_path}")
    print(f"strata covered: {len(by_stratum)}, wrong-answer population: {len(all_rows)}")


def classify_heuristic(output):
    import re
    if output is None or not str(output).strip():
        return "REFUSAL"
    refusal_re = re.compile(
        r"\b(i (can'?t|cannot|don'?t have|do not have|am unable)|no [a-z ]{0,40} (was|were|is|are) (not )?"
        r"(provided|found|available|recorded|given|specified|mentioned)|"
        r"not (provided|available|found|recorded|specified) in the (context|text|document|log|passage)|"
        r"(does not|doesn'?t) (include|contain|mention|specify|provide)|"
        r"unable to determine|insufficient (information|context))\b", re.I)
    if refusal_re.search(str(output)):
        return "REFUSAL"
    return "FABRICATION"


if __name__ == "__main__":
    main()
