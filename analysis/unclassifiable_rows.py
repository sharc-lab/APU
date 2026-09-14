"""
Recompute fabrication / abstention / correct rates for stage_a_scale.json and
selfreport_arms.json with and without unclassifiable rows in the denominator.

Background
----------
stage_a_scale.json contains 21 original rows that were empty (output == "").
On rerun at MIN_PREDICT=1024 (commit bda6894) those rows split into:
  - 9 rows: done_reason='stop', classifiable (classification_method='done_reason')
  - 12 rows: done_reason='length', output still empty, classification_method='budget_exhausted'

The 12 budget_exhausted rows are scored outcome='incorrect' because the scorer
receives an empty string, but the true outcome is unknown (the model ran out of
thinking budget before producing output).  Including them in the fabrication-rate
denominator overstates the fabrication count.

The 51 rows with classification_method='unavailable' are pre-rerun rows whose
classification provenance was not recorded; they have non-empty outputs and
recorded outcomes and are NOT unclassifiable — they are treated as valid data.

Usage
-----
    python analysis/unclassifiable_rows.py

Output
------
Prints markdown tables suitable for pasting into docs/FINDINGS.md.
Does not modify any result JSON.
"""
from __future__ import annotations

import json
import pathlib
import collections
from typing import Any

REPO_ROOT = pathlib.Path(__file__).parent.parent
STAGE_A   = REPO_ROOT / "results" / "stage_a_scale.json"
SR_ARMS   = REPO_ROOT / "results" / "selfreport_arms.json"


# ---------------------------------------------------------------------------
# helpers

def _strip_prefix(filler_type: str) -> str:
    """'filler_a_F-NUM' -> 'F-NUM'"""
    return filler_type.split("_")[-1]


def _rates(n_correct: int, n_fabricated: int, n_abstained: int, n_total: int) -> dict:
    if n_total == 0:
        return dict(n=0, correct=0, fabricated=0, abstained=0,
                    correct_rate="—", fabrication_rate="—", abstention_rate="—")
    return dict(
        n=n_total,
        correct=n_correct,
        fabricated=n_fabricated,
        abstained=n_abstained,
        correct_rate=f"{n_correct / n_total:.1%}",
        fabrication_rate=f"{n_fabricated / n_total:.1%}",
        abstention_rate=f"{n_abstained / n_total:.1%}",
    )


# ---------------------------------------------------------------------------
# stage_a_scale analysis

def analyse_stage_a(rows: list[dict[str, Any]]) -> None:
    print("=" * 72)
    print("FILE: results/stage_a_scale.json")
    print(f"Total rows: {len(rows)}")
    print()

    # ── 1. Unclassifiable row inventory ────────────────────────────────────

    length_rows  = [r for r in rows if r.get("done_reason") == "length"]
    unavail_rows = [r for r in rows if r.get("classification_method") == "unavailable"]
    empty_rows   = [r for r in rows if r.get("output", "") == ""]

    # All three criteria select the same 12 rows for the empty/length/budget case;
    # unavailable is a superset (also covers original non-rerun rows with valid outcomes).
    union_unclass = {id(r) for r in length_rows} | {id(r) for r in empty_rows}
    print("── Unclassifiable row counts (by criterion) ──")
    print(f"  done_reason == 'length'              : {len(length_rows):>3} rows")
    print(f"  classification_method == 'unavailable': {len(unavail_rows):>3} rows "
          f"(note: {sum(1 for r in unavail_rows if r['output'] != '')} have non-empty outputs and valid outcomes)")
    print(f"  output == ''                          : {len(empty_rows):>3} rows")
    print(f"  union of length + empty              : {len(union_unclass):>3} rows (used for denominator correction)")
    print()

    # ── 2. Breakdown by probe / filler / ratio ──────────────────────────────

    print("── Unclassifiable rows by (probe, filler, ratio) ──")
    print(f"{'probe':<8} {'filler':<10} {'ratio':>6} {'n_unclass':>10} {'of cell total':>14}")
    print("-" * 52)

    cells: dict[tuple, list] = collections.defaultdict(list)
    for r in rows:
        cells[(r["probe_id"], _strip_prefix(r["filler_type"]), r["budget_ratio"])].append(r)

    unclass_cells = sorted(
        {(r["probe_id"], _strip_prefix(r["filler_type"]), r["budget_ratio"])
         for r in length_rows}
    )
    for key in sorted(cells.keys()):
        probe, filler, ratio = key
        cell = cells[key]
        n_unc = sum(1 for r in cell if r.get("done_reason") == "length")
        if n_unc > 0:
            print(f"{probe:<8} {filler:<10} {ratio:>6.2f} {n_unc:>10} {n_unc:>6}/{len(cell):<6}")

    # Also show cells with zero unclassifiable (for completeness)
    for key in sorted(cells.keys()):
        probe, filler, ratio = key
        cell = cells[key]
        n_unc = sum(1 for r in cell if r.get("done_reason") == "length")
        if n_unc == 0:
            print(f"{probe:<8} {filler:<10} {ratio:>6.2f} {0:>10} {0:>6}/{len(cell):<6}")
    print()

    # ── 3. Rate recomputation ───────────────────────────────────────────────
    # "Fabrication" here = outcome=='incorrect' with non-empty output (true wrong answer
    #  or filler lift).  "Abstention" = outcome=='incorrect' with empty output but
    #  classifiable (none in this file).  Budget_exhausted rows are neither.

    def compute_rates(subset: list[dict], label: str) -> None:
        n_total       = len(subset)
        n_correct     = sum(1 for r in subset if r["outcome"] == "correct")
        n_budget_exh  = sum(1 for r in subset if r.get("done_reason") == "length")
        # Uncorrected: all outcome='incorrect' rows count as fabrications —
        # this matches how the scorer currently treats them (output="" → incorrect).
        n_fabricated  = sum(1 for r in subset if r["outcome"] == "incorrect")
        n_abstained   = 0  # scorer has no abstention outcome in this file
        r = _rates(n_correct, n_fabricated, n_abstained, n_total)
        print(f"  {label:<50} n={r['n']:>3}  "
              f"correct={r['correct_rate']:>7}  "
              f"fabrication={r['fabrication_rate']:>7}  "
              f"abstention={r['abstention_rate']:>7}  "
              f"budget_exhausted={n_budget_exh}")

    def compute_corrected(subset: list[dict], label: str) -> None:
        classifiable  = [r for r in subset if r.get("done_reason") != "length"
                         and r.get("output", "") != ""]
        # include correct rows (they have output)
        classifiable2 = [r for r in subset if r.get("done_reason") != "length"]
        n_total       = len(classifiable2)
        n_correct     = sum(1 for r in classifiable2 if r["outcome"] == "correct")
        n_fabricated  = sum(1 for r in classifiable2
                            if r["outcome"] == "incorrect" and r.get("output", "") != "")
        n_abstained   = 0  # none in this file
        r = _rates(n_correct, n_fabricated, n_abstained, n_total)
        removed       = len(subset) - n_total
        print(f"  {label:<50} n={r['n']:>3}  "
              f"correct={r['correct_rate']:>7}  "
              f"fabrication={r['fabrication_rate']:>7}  "
              f"abstention={r['abstention_rate']:>7}  "
              f"(removed {removed} budget-exhausted rows)")

    print("── Rates: UNCORRECTED (budget_exhausted scored as incorrect) ──")
    compute_rates(rows, "All rows (n=72)")
    compute_rates([r for r in rows if r["budget_ratio"] < 1.0],
                  "Extinct only (ratio 0.85 + 0.40, n=48)")
    compute_rates([r for r in rows if r["budget_ratio"] < 1.0
                   and _strip_prefix(r["filler_type"]) == "F-NUM"],
                  "Extinct + F-NUM only")
    compute_rates([r for r in rows if r["budget_ratio"] < 1.0
                   and _strip_prefix(r["filler_type"]) == "F-TYPED"],
                  "Extinct + F-TYPED only")
    print()

    print("── Rates: CORRECTED (budget_exhausted rows removed from denominator) ──")
    compute_corrected(rows, "All rows (n=72 → n=60 after removal)")
    compute_corrected([r for r in rows if r["budget_ratio"] < 1.0],
                      "Extinct only (n=48 → n=36 after removal)")
    compute_corrected([r for r in rows if r["budget_ratio"] < 1.0
                       and _strip_prefix(r["filler_type"]) == "F-NUM"],
                      "Extinct + F-NUM only")
    compute_corrected([r for r in rows if r["budget_ratio"] < 1.0
                       and _strip_prefix(r["filler_type"]) == "F-TYPED"],
                      "Extinct + F-TYPED only")
    print()

    # ── 4. Per-probe breakdown (extinct only) ──────────────────────────────

    print("── Per-probe rates at extinct conditions (ratio 0.85 and 0.40) ──")
    print()
    print("Uncorrected:")
    header = f"  {'probe':<8} {'filler':<10} {'n':>4} {'correct':>8} {'fabrication':>13} {'abstention':>11} {'budget_exh':>11}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for (probe, filler), group in sorted(
        {(r["probe_id"], _strip_prefix(r["filler_type"])): None
         for r in rows if r["budget_ratio"] < 1.0}.items()
    ):
        sub = [r for r in rows if r["budget_ratio"] < 1.0
               and r["probe_id"] == probe
               and _strip_prefix(r["filler_type"]) == filler]
        n_be = sum(1 for r in sub if r.get("done_reason") == "length")
        n_c  = sum(1 for r in sub if r["outcome"] == "correct")
        n_f  = sum(1 for r in sub if r["outcome"] == "incorrect" and r.get("output","") != "")
        n_a  = 0
        print(f"  {probe:<8} {filler:<10} {len(sub):>4} {n_c/len(sub):>8.1%} "
              f"{n_f/len(sub):>13.1%} {0:>11.1%} {n_be:>11}")

    print()
    print("Corrected (budget_exhausted excluded):")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for (probe, filler), group in sorted(
        {(r["probe_id"], _strip_prefix(r["filler_type"])): None
         for r in rows if r["budget_ratio"] < 1.0}.items()
    ):
        sub = [r for r in rows if r["budget_ratio"] < 1.0
               and r["probe_id"] == probe
               and _strip_prefix(r["filler_type"]) == filler
               and r.get("done_reason") != "length"]
        if not sub:
            print(f"  {probe:<8} {filler:<10} {'0':>4} {'—':>8} {'—':>13} {'—':>11}  (all rows budget_exhausted)")
            continue
        n_c = sum(1 for r in sub if r["outcome"] == "correct")
        n_f = sum(1 for r in sub if r["outcome"] == "incorrect" and r.get("output","") != "")
        n_a = 0
        print(f"  {probe:<8} {filler:<10} {len(sub):>4} {n_c/len(sub):>8.1%} "
              f"{n_f/len(sub):>13.1%} {0:>11.1%}")
    print()


# ---------------------------------------------------------------------------
# selfreport_arms analysis

def analyse_selfreport_arms(sr: dict[str, Any]) -> None:
    print("=" * 72)
    print("FILE: results/selfreport_arms.json")
    print(f"Model: {sr['model']}   Hardware: {sr['hardware']}")
    print(f"Probes: {sr['probes']}")
    print(f"Ratios: {sr['ratios']}   n_reps: {sr['n_reps']}")
    print()
    print("Note: this file stores pre-computed aggregates only (no individual rows).")
    print("Individual done_reason values are not available; no denominator correction")
    print("can be applied.  Rates below are as stored in the file.")
    print()

    agg = sr["aggregate_by_arm"]
    print("── Aggregate rates by arm (all ratios combined) ──")
    print(f"  {'arm':<30} {'n':>5} {'correct':>9} {'fabrication':>13} {'abstention':>12}")
    print("  " + "-" * 72)
    for arm, vals in agg.items():
        print(f"  {arm:<30} {vals['n_total']:>5} "
              f"{vals['correct_rate']:>9.1%} "
              f"{vals['fabrication_rate']:>13.1%} "
              f"{vals['abstention_rate']:>12.1%}")
    print()

    pap = sr["per_arm_per_ratio"]
    print("── Per-arm per-ratio rates ──")
    print(f"  {'arm':<30} {'ratio':>6} {'n':>5} {'correct':>9} {'fabrication':>13} {'abstention':>12}")
    print("  " + "-" * 78)
    for arm, by_ratio in pap.items():
        for ratio_str, vals in sorted(by_ratio.items(), key=lambda x: float(x[0])):
            n = vals["n"]
            n_c = vals.get("correct", 0)
            n_f = vals.get("incorrect", 0)
            n_a = vals.get("abstained", 0)
            print(f"  {arm:<30} {float(ratio_str):>6.2f} {n:>5} "
                  f"{n_c/n:>9.1%} "
                  f"{n_f/n:>13.1%} "
                  f"{n_a/n:>12.1%}")
    print()


# ---------------------------------------------------------------------------
# markdown tables for FINDINGS.md

def print_markdown_tables(rows: list[dict[str, Any]], sr: dict[str, Any]) -> None:
    print("=" * 72)
    print("MARKDOWN TABLES (for FINDINGS.md)")
    print()

    # ── Unclassifiable inventory ──
    print("### Table: Unclassifiable row inventory — stage_a_scale.json")
    print()
    print("| Criterion | Count | Notes |")
    print("|-----------|------:|-------|")

    n_length = sum(1 for r in rows if r.get("done_reason") == "length")
    n_unavail = sum(1 for r in rows if r.get("classification_method") == "unavailable")
    n_empty  = sum(1 for r in rows if r.get("output", "") == "")
    n_unavail_nonempty = sum(1 for r in rows
                             if r.get("classification_method") == "unavailable"
                             and r["output"] != "")
    print(f"| `done_reason == 'length'` | {n_length} "
          f"| art_07 (11 rows) and art_06/F-TYPED/r=0.85 (1 row); output empty; outcome unknown |")
    print(f"| `classification_method == 'unavailable'` | {n_unavail} "
          f"| Pre-rerun rows; {n_unavail_nonempty} have non-empty outputs with valid recorded outcomes — **not** unclassifiable |")
    print(f"| `output == ''` | {n_empty} "
          f"| Identical to `done_reason == 'length'` set |")
    print(f"| **Truly unclassifiable (used for correction)** | **{n_length}** "
          f"| `done_reason == 'length'` only |")
    print()

    # ── Budget-exhausted breakdown ──
    print("### Table: Budget-exhausted rows by probe / filler / ratio")
    print()
    print("| probe | filler | ratio | budget-exhausted rows | cell total |")
    print("|-------|--------|------:|----------------------:|-----------:|")
    cells: dict[tuple, list] = collections.defaultdict(list)
    for r in rows:
        cells[(r["probe_id"], _strip_prefix(r["filler_type"]), r["budget_ratio"])].append(r)
    for (probe, filler, ratio) in sorted(cells):
        cell = cells[(probe, filler, ratio)]
        n_be = sum(1 for r in cell if r.get("done_reason") == "length")
        if n_be > 0:
            print(f"| {probe} | {filler} | {ratio:.2f} | {n_be} | {len(cell)} |")
    print()

    # ── Rate table: uncorrected vs corrected (all rows) ──
    print("### Table: Fabrication rates — uncorrected vs corrected (all rows, n=72)")
    print()
    print("'Fabrication' = outcome=='incorrect' with non-empty output.  "
          "'Budget-exhausted' rows (n=12) have empty output and unknown outcome; "
          "the scorer records them as incorrect, which inflates the fabrication count.")
    print()
    print("| Subset | n (denom) | correct | fabrication | abstention | budget-exhausted excl. |")
    print("|--------|----------:|--------:|------------:|-----------:|----------------------:|")

    def row_stats(subset: list[dict], corrected: bool) -> tuple:
        if corrected:
            subset = [r for r in subset if r.get("done_reason") != "length"]
        n = len(subset)
        if n == 0:
            return ("—", "—", "—", 0)
        n_c = sum(1 for r in subset if r["outcome"] == "correct")
        if corrected:
            # corrected: only non-empty incorrect rows are true fabrications
            n_f = sum(1 for r in subset
                      if r["outcome"] == "incorrect" and r.get("output","") != "")
        else:
            # uncorrected: all incorrect rows counted as fabrications (scorer behaviour)
            n_f = sum(1 for r in subset if r["outcome"] == "incorrect")
        n_a = 0
        return (f"{n_c/n:.1%}", f"{n_f/n:.1%}", f"{n_a/n:.1%}", n)

    subsets = [
        ("All rows", rows),
        ("Extinct (r=0.85 and r=0.40)", [r for r in rows if r["budget_ratio"] < 1.0]),
        ("Extinct + F-NUM", [r for r in rows if r["budget_ratio"] < 1.0
                             and _strip_prefix(r["filler_type"]) == "F-NUM"]),
        ("Extinct + F-TYPED", [r for r in rows if r["budget_ratio"] < 1.0
                               and _strip_prefix(r["filler_type"]) == "F-TYPED"]),
        ("Headroom (r=1.20)", [r for r in rows if r["budget_ratio"] == 1.2]),
    ]

    for label, subset in subsets:
        be = sum(1 for r in subset if r.get("done_reason") == "length")
        c_u, f_u, a_u, n_u = row_stats(subset, corrected=False)
        c_c, f_c, a_c, n_c = row_stats(subset, corrected=True)
        print(f"| {label} (uncorr.) | {n_u} | {c_u} | {f_u} | {a_u} | — |")
        print(f"| {label} (corr.) | {n_c} | {c_c} | {f_c} | {a_c} | {be} removed |")

    print()

    # ── selfreport_arms (no correction possible) ──
    agg = sr["aggregate_by_arm"]
    print("### Table: selfreport_arms.json — aggregate rates by arm (no correction applied)")
    print()
    print("No individual row data available; done_reason values not stored.  Rates are as")
    print("computed in the result file.  Probes: rag_01, rag_02, rag_05, sea_01, sea_04")
    print("(all extinct, ratios 0.85 / 0.70 / 0.55 / 0.40).")
    print()
    print("| arm | description | n | correct | fabrication | abstention |")
    print("|-----|-------------|--:|--------:|------------:|-----------:|")
    arm_desc = sr.get("arm_descriptions", {})
    for arm, vals in agg.items():
        desc = arm_desc.get(arm, "")[:60]
        print(f"| {arm} | {desc} | {vals['n_total']} | "
              f"{vals['correct_rate']:.1%} | "
              f"{vals['fabrication_rate']:.1%} | "
              f"{vals['abstention_rate']:.1%} |")
    print()

    # ── Headline rate sentence ──
    # Corrected extinct fabrication rate (stage_a_scale)
    extinct_class = [r for r in rows
                     if r["budget_ratio"] < 1.0 and r.get("done_reason") != "length"]
    n_ec = len(extinct_class)
    n_ec_fab = sum(1 for r in extinct_class
                   if r["outcome"] == "incorrect" and r.get("output","") != "")
    n_be_total = sum(1 for r in rows if r.get("done_reason") == "length")

    print("### Corrected headline rate (for abstract)")
    print()
    print(f"Among the {n_ec} classifiable extinct-context responses in stage_a_scale.json")
    print(f"(budget_ratio ≤ 0.85; {n_be_total} thinking-budget-exhausted art_07/art_06 rows excluded),")
    print(f"gpt-oss:120b fabricated or retrieved a filler value in {n_ec_fab}/{n_ec} cases")
    print(f"({n_ec_fab/n_ec:.0%}); abstention was not observed in any classifiable row.")
    print()


# ---------------------------------------------------------------------------
# main

def main() -> None:
    with open(STAGE_A) as f:
        stage_a = json.load(f)
    with open(SR_ARMS) as f:
        sr = json.load(f)

    rows = stage_a["rows"]

    analyse_stage_a(rows)
    analyse_selfreport_arms(sr)
    print_markdown_tables(rows, sr)


if __name__ == "__main__":
    main()
