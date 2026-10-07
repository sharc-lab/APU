"""Builds a HELD-OUT 150-row blinded labeling sample for the scorer-vs-annotator kappa study.

Why this exists: the original kappa sample (results/labeling/kappa_sample.csv, 150 rows) scored kappa
0.55 against the then-current scorer, and 1.0 after two fixes to evaluation/outcome.py. Those fixes were
derived from that same sample's disagreement rows, so the 1.0 is in-sample. This script draws a fresh
150-row sample, the same way, from the same population, excluding every row of the original sample, so
the current scorer can be checked out-of-sample.

Source population (same as the original, verified, not assumed): the original sample
(results/labeling/r1b_wrong_sample.csv) is reproduced byte-for-byte by
analysis/build_r1b_wrong_sample.py's sampling procedure (seed 20260930) run over these two committed
files, so they are its real source (KAPPA_STUDY_NOTE.md's earlier statement that the raw files were never
committed is superseded; they were committed later):
  results/t2s_night2_20260929T202603Z.jsonl  (evo-t2s)
  results/t2s_night2_20260929T205109Z.jsonl  (evo-x2)
Filter (identical to build_r1b_wrong_sample.load_wrong_rows): kind == "call", probe_id art_*, arm in
{arm1_baseline, arm3_self_report}, score != 1.0 (wrong-answer population only).

Exclusions (by exact raw-row identity, (machine, item_id), which is unique in this population):
  1. every row of the original 150-row sample (reproduced in-process and asserted equal to the committed
     r1b_wrong_sample.csv before anything is written; the script refuses to run if it does not match);
  2. every row of results/labeling/ritz_spotcheck_30.csv (reproduced the same way via
     build_ritz_spotcheck_30's seed), so the new human subset stays disjoint from the earlier one.

Stratification (identical to the original): round-robin across every (model, ratio, arm, machine)
stratum present in the remaining wrong-answer population, stratum order and within-stratum order both
shuffled by the seed, capped at 150 rows. Seed: 20261006 (the original used 20260930).

Blinding: the labeling files carry no scorer verdict, no original score, and no score_detail. The scorer
verdict (current evaluation/outcome.py classify(), called with the row's real score and the probe's real
scorer_type) goes only to the separate key file, which the annotator must not open while labeling.

Usage: py -3.12 analysis/build_kappa_heldout_sample.py
Writes:
  results/labeling/kappa_heldout_blinded.csv     -- 150 rows to label; annotator columns empty
  results/labeling/kappa_heldout_key.csv         -- id -> source row identity + scorer verdict (the key)
  results/labeling/ritz_spotcheck_heldout_30.csv -- blinded 30-row subset for a human rater (label empty)
Prints only population/stratum counts, never scorer verdicts.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

import build_r1b_wrong_sample as orig_builder  # noqa: E402

SEED = 20261006
ORIG_SEED = 20260930
RITZ_SEED = 20261001
N = 150
N_HUMAN = 30
HUMAN_SUBSET_SEED = 20261007

SOURCE_FILES = {
    "evo-t2s": "results/t2s_night2_20260929T202603Z.jsonl",
    "evo-x2": "results/t2s_night2_20260929T205109Z.jsonl",
}
LAB = ROOT / "results" / "labeling"
BLIND_PATH = LAB / "kappa_heldout_blinded.csv"
KEY_PATH = LAB / "kappa_heldout_key.csv"
HUMAN_PATH = LAB / "ritz_spotcheck_heldout_30.csv"

BLIND_FIELDS = ["id", "model", "ratio", "arm", "machine", "question", "expected", "output",
                "visible_prompt_excerpt_near_answer", "annotator_claude", "annotator_claude_reason"]
KEY_FIELDS = ["id", "machine", "source_file", "source_item_id", "model", "probe_id", "scorer_type",
              "ratio", "arm", "raw_score", "score_detail", "scorer_classification",
              "classification_method", "format_compliant"]
HUMAN_FIELDS = ["id", "question", "expected", "output", "label"]


def _load_outcome():
    spec = importlib.util.spec_from_file_location("outcome", ROOT / "evaluation" / "outcome.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _scorer_types():
    types = {}
    for line in (ROOT / "evaluation" / "probes" / "artifact.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            types[d["id"]] = d["scorer_type"]
    return types


def row_key(r) -> tuple[str, str]:
    return (r["_machine"], r["item_id"])


def load_population():
    rows = []
    for machine, rel in SOURCE_FILES.items():
        for r in orig_builder.load_wrong_rows(str(ROOT / rel), machine):
            r["_source_file"] = rel
            rows.append(r)
    keys = Counter(row_key(r) for r in rows)
    dups = [k for k, v in keys.items() if v > 1]
    if dups:
        raise SystemExit(f"(machine, item_id) is not unique in the population: {dups[:5]}")
    return rows


def round_robin_sample(rows, n, seed):
    """Exactly build_r1b_wrong_sample.main()'s sampling procedure, factored out so it can be re-run."""
    by_stratum = {}
    for r in rows:
        by_stratum.setdefault(orig_builder.stratify_key(r), []).append(r)
    rng = random.Random(seed)
    strata = list(by_stratum.keys())
    rng.shuffle(strata)
    pools = {k: list(v) for k, v in by_stratum.items()}
    for pool in pools.values():
        rng.shuffle(pool)
    sample, i = [], 0
    while len(sample) < n and any(pools.values()):
        k = strata[i % len(strata)]
        if pools[k]:
            sample.append(pools[k].pop())
        i += 1
        if i > n * 20:
            break
    return sample[:n], len(by_stratum)


def original_sample_keys(population):
    """Reproduce the original 150-row sample and assert it equals the committed CSV, row for row."""
    sample, _ = round_robin_sample(population, 150, ORIG_SEED)
    with open(LAB / "r1b_wrong_sample.csv", encoding="utf-8") as f:
        committed = list(csv.DictReader(f))
    if len(committed) != len(sample):
        raise SystemExit("original sample reproduction failed: row count differs")
    for idx, (r, c) in enumerate(zip(sample, committed)):
        rid = f"r1bw_{idx:03d}_{r['_machine']}_{r['model_id']}_{r['probe_id']}"
        if rid != c["id"] or (r.get("output") or "") != c["output"] or r["arm"] != c["arm"] \
                or str(r["budget_ratio"]) != c["ratio"]:
            raise SystemExit(f"original sample reproduction failed at row {idx} ({c['id']})")
    return {row_key(r) for r in sample}


def ritz30_keys():
    """Reproduce build_ritz_spotcheck_30's 30 rows (all rows, not only wrong ones) and check the ids."""
    import build_ritz_spotcheck_30 as ritz
    pool = []
    for machine, path in ritz.SOURCE_FILES.items():
        pool.extend(ritz.load_rows(path, machine))
    rng = random.Random(RITZ_SEED)
    rng.shuffle(pool)
    sample = pool[:30]
    with open(LAB / "ritz_spotcheck_30.csv", encoding="utf-8") as f:
        committed_ids = [r["id"] for r in csv.DictReader(f)]
    ids = [f"ritz30_{i:02d}_{r['_machine']}_{r['model_id']}_{r['probe_id']}_r{r['budget_ratio']}_{r['arm']}"
           for i, r in enumerate(sample)]
    if ids != committed_ids:
        raise SystemExit("ritz_spotcheck_30 reproduction failed: ids differ from the committed CSV")
    return {row_key(r) for r in sample}


def main():
    population = load_population()
    orig_keys = original_sample_keys(population)
    ritz_keys = ritz30_keys()
    excluded = orig_keys | ritz_keys
    remaining = [r for r in population if row_key(r) not in excluded]
    sample, n_strata = round_robin_sample(remaining, N, SEED)
    assert not ({row_key(r) for r in sample} & excluded)

    segs = orig_builder.load_segments()
    outcome = _load_outcome()
    stypes = _scorer_types()

    blind_rows, key_rows = [], []
    for idx, r in enumerate(sample):
        seg = segs.get(r["probe_id"], {})
        rid = f"kh_{idx:03d}_{r['_machine']}_{r['model_id']}_{r['probe_id']}"
        blind_rows.append({
            "id": rid, "model": r["model_id"], "ratio": r["budget_ratio"], "arm": r["arm"],
            "machine": r["_machine"], "question": seg.get("question", ""), "expected": seg.get("expected", ""),
            "output": r.get("output") or "", "visible_prompt_excerpt_near_answer": orig_builder.visible_excerpt(r, segs),
            "annotator_claude": "", "annotator_claude_reason": "",
        })
        stype = stypes.get(r["probe_id"], "exact")
        res = outcome.classify(output=r.get("output"), expected=seg.get("expected", ""), scorer_type=stype,
                               score=r.get("score"), done_reason=r.get("done_reason"))
        key_rows.append({
            "id": rid, "machine": r["_machine"], "source_file": r["_source_file"], "source_item_id": r["item_id"],
            "model": r["model_id"], "probe_id": r["probe_id"], "scorer_type": stype, "ratio": r["budget_ratio"],
            "arm": r["arm"], "raw_score": r.get("score"), "score_detail": r.get("score_detail"),
            "scorer_classification": res["outcome_class"], "classification_method": res["classification_method"],
            "format_compliant": res["format_compliant"],
        })

    LAB.mkdir(parents=True, exist_ok=True)
    with open(BLIND_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=BLIND_FIELDS)
        w.writeheader()
        w.writerows(blind_rows)
    with open(KEY_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=KEY_FIELDS)
        w.writeheader()
        w.writerows(key_rows)

    human = random.Random(HUMAN_SUBSET_SEED).sample(blind_rows, N_HUMAN)
    human.sort(key=lambda r: r["id"])
    with open(HUMAN_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HUMAN_FIELDS)
        w.writeheader()
        for r in human:
            w.writerow({"id": r["id"], "question": r["question"], "expected": r["expected"],
                        "output": r["output"], "label": ""})

    print(f"population (wrong-answer rows, both files): {len(population)}")
    print(f"excluded: {len(orig_keys)} original-sample rows + {len(ritz_keys & set(map(row_key, population)))} "
          f"ritz30 rows in the wrong-answer population (union {len(excluded & set(map(row_key, population)))})")
    print(f"remaining: {len(remaining)}, strata: {n_strata}, sampled: {len(sample)} (seed {SEED})")
    for field in ("model", "arm", "machine", "ratio"):
        print(f"  by {field}: {dict(Counter(str(r[field]) for r in blind_rows).most_common())}")
    print(f"wrote {BLIND_PATH.name}, {KEY_PATH.name} (KEY: do not open while labeling), {HUMAN_PATH.name}")


if __name__ == "__main__":
    main()
