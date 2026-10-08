"""Cohen's kappa and confusion matrix between two raters' categorical labels.

Implemented directly (no scikit-learn import) even though scikit-learn is already a pyproject.toml
dependency of this project -- the instruction for this study was to implement kappa directly rather
than lean on a library call, so the computation here is auditable from the formula alone.

Cohen's kappa:
    kappa = (p_o - p_e) / (1 - p_e)
where
    p_o = observed agreement = (number of rows where both raters agree) / n
    p_e = expected agreement by chance = sum_k( P(rater1 == k) * P(rater2 == k) )
over all categories k present in the label set (the standard marginal-product formula).

Usage:
    py -3.12 analysis/kappa_agreement.py <csv_path> <col_a> <col_b>
    py -3.12 analysis/kappa_agreement.py --heldout
        held-out sample (2026-10-06): joins results/labeling/kappa_heldout_key.csv (current scorer) with
        results/labeling/kappa_heldout_annotator_claude.csv (blinded labels) on id, prints kappa with an
        analytic and a paired-bootstrap 95% CI, the confusion matrix, and every disagreement row.

Writes nothing; prints kappa and the confusion matrix (rater A rows x rater B columns) to stdout.
"""
from __future__ import annotations

import csv
import sys
from collections import Counter


def cohens_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """Cohen's kappa for two equal-length lists of categorical labels over the same n items."""
    if len(labels_a) != len(labels_b):
        raise ValueError("labels_a and labels_b must be the same length")
    n = len(labels_a)
    if n == 0:
        raise ValueError("cannot compute kappa over zero rows")

    categories = sorted(set(labels_a) | set(labels_b))

    observed_agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    p_o = observed_agree / n

    count_a = Counter(labels_a)
    count_b = Counter(labels_b)
    p_e = sum((count_a.get(k, 0) / n) * (count_b.get(k, 0) / n) for k in categories)

    if p_e == 1.0:
        # Every row falls in a single shared category for both raters: agreement is forced, not
        # informative. By convention kappa is undefined here; we return 1.0 only when observed
        # agreement is also total (the degenerate, literally-all-one-label case), matching the
        # usual software convention, and raise otherwise (should not occur with real multi-class data).
        if p_o == 1.0:
            return 1.0
        raise ZeroDivisionError("expected agreement is 1.0 but observed agreement is not -- degenerate input")

    return (p_o - p_e) / (1 - p_e)


def confusion_matrix(labels_a: list[str], labels_b: list[str]) -> tuple[list[str], dict]:
    """Returns (sorted category list, {(a_label, b_label): count}) over the union of both raters'
    categories, so a category used by only one rater still gets a row/column of zeros."""
    categories = sorted(set(labels_a) | set(labels_b))
    matrix = {(a, b): 0 for a in categories for b in categories}
    for a, b in zip(labels_a, labels_b):
        matrix[(a, b)] += 1
    return categories, matrix


def print_confusion_matrix(categories: list[str], matrix: dict, label_a="rater A", label_b="rater B"):
    width = max(len(c) for c in categories + [label_a]) + 2
    header = " " * width + "".join(f"{c:>14}" for c in categories)
    print(f"rows = {label_a}, columns = {label_b}")
    print(header)
    for a in categories:
        row = "".join(f"{matrix[(a, b)]:>14}" for b in categories)
        print(f"{a:<{width}}{row}")


def kappa_analytic_ci(labels_a: list[str], labels_b: list[str], z: float = 1.959964) -> tuple[float, float, float]:
    """(se, lo, hi): Cohen's (1960) large-sample approximation SE = sqrt(p_o (1 - p_o) / (n (1 - p_e)^2)),
    CI = kappa +/- z * SE, clipped to [-1, 1]. Simple and conservative-ish; reported next to the bootstrap."""
    n = len(labels_a)
    k = cohens_kappa(labels_a, labels_b)
    p_o = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    ca, cb = Counter(labels_a), Counter(labels_b)
    p_e = sum((ca.get(c, 0) / n) * (cb.get(c, 0) / n) for c in set(labels_a) | set(labels_b))
    se = (p_o * (1 - p_o) / (n * (1 - p_e) ** 2)) ** 0.5
    return se, max(-1.0, k - z * se), min(1.0, k + z * se)


def kappa_bootstrap_ci(labels_a: list[str], labels_b: list[str], n_boot: int = 10000, seed: int = 20261006,
                       alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for kappa: resample item indices with replacement (paired), recompute kappa.
    A resample where both raters put every item in one shared category (p_e == 1) is degenerate; those
    resamples are skipped (vanishingly rare at n=150 with three categories)."""
    import random
    rng = random.Random(seed)
    n = len(labels_a)
    ks = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        a = [labels_a[i] for i in idx]
        b = [labels_b[i] for i in idx]
        try:
            ks.append(cohens_kappa(a, b))
        except ZeroDivisionError:
            continue
    ks.sort()
    lo = ks[int((alpha / 2) * len(ks))]
    hi = ks[min(len(ks) - 1, int((1 - alpha / 2) * len(ks)))]
    return lo, hi


HELDOUT_KEY = "results/labeling/kappa_heldout_key.csv"
HELDOUT_LABELS = "results/labeling/kappa_heldout_annotator_claude.csv"


def heldout_stats(repo_root) -> dict:
    """Held-out kappa: current scorer (key file, built by analysis/build_kappa_heldout_sample.py) vs
    annotator_claude's blinded labels, joined on id. Returns kappa, both CIs, confusion matrix, and the
    disagreement rows."""
    from pathlib import Path
    root = Path(repo_root)
    with open(root / HELDOUT_KEY, encoding="utf-8") as f:
        key = {r["id"]: r for r in csv.DictReader(f)}
    with open(root / HELDOUT_LABELS, encoding="utf-8") as f:
        lab = list(csv.DictReader(f))
    if set(key) != {r["id"] for r in lab}:
        raise ValueError("held-out key and label files do not cover the same ids")
    a = [key[r["id"]]["scorer_classification"] for r in lab]
    b = [r["annotator_claude"] for r in lab]
    n = len(lab)
    agree = sum(1 for x, y in zip(a, b) if x == y)
    k = cohens_kappa(a, b)
    se, alo, ahi = kappa_analytic_ci(a, b)
    blo, bhi = kappa_bootstrap_ci(a, b)
    cats, m = confusion_matrix(a, b)
    dis = [{"id": r["id"], "scorer": key[r["id"]]["scorer_classification"],
            "method": key[r["id"]]["classification_method"], "annotator": r["annotator_claude"],
            "output": r["output"], "expected": r["expected"], "reason": r["annotator_claude_reason"]}
           for r in lab if key[r["id"]]["scorer_classification"] != r["annotator_claude"]]
    # Secondary, method-controlled comparison: the ORIGINAL study's annotator was a function
    # (analysis/build_kappa_sample.py::classify_human_label), not a manual read. Applying that same
    # function to the held-out rows isolates "new rows" from "new annotator method".
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_kappa_sample", root / "analysis" / "build_kappa_sample.py")
    bks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bks)
    f = [bks.classify_human_label(r)[0] for r in lab]
    k_fn = cohens_kappa(a, f)
    fn_lo, fn_hi = kappa_bootstrap_ci(a, f)
    k_manual_vs_fn = cohens_kappa(b, f)
    return {"n": n, "agree": agree, "kappa": k, "analytic_se": se, "analytic_ci": (alo, ahi),
            "bootstrap_ci": (blo, bhi), "categories": cats, "matrix": m, "disagreements": dis,
            "scorer_counts": dict(Counter(a)), "annotator_counts": dict(Counter(b)),
            "fn_annotator_kappa": k_fn, "fn_annotator_bootstrap_ci": (fn_lo, fn_hi),
            "fn_annotator_agree": sum(1 for x, y in zip(a, f) if x == y),
            "manual_vs_fn_kappa": k_manual_vs_fn,
            "manual_vs_fn_agree": sum(1 for x, y in zip(b, f) if x == y)}


def summarize_heldout(repo_root) -> None:
    s = heldout_stats(repo_root)
    print(f"n = {s['n']}")
    print(f"raw agreement = {s['agree']}/{s['n']} = {s['agree'] / s['n']:.4f}")
    print(f"Cohen's kappa = {s['kappa']:.4f}")
    print(f"  analytic 95% CI = [{s['analytic_ci'][0]:.4f}, {s['analytic_ci'][1]:.4f}] (SE {s['analytic_se']:.4f})")
    print(f"  bootstrap 95% CI = [{s['bootstrap_ci'][0]:.4f}, {s['bootstrap_ci'][1]:.4f}] (10000 paired resamples, seed 20261006)")
    print(f"scorer counts: {s['scorer_counts']}; annotator counts: {s['annotator_counts']}")
    print(f"secondary: scorer vs original annotator FUNCTION (classify_human_label) on the same rows: "
          f"agree {s['fn_annotator_agree']}/{s['n']}, kappa {s['fn_annotator_kappa']:.4f}, bootstrap 95% CI "
          f"[{s['fn_annotator_bootstrap_ci'][0]:.4f}, {s['fn_annotator_bootstrap_ci'][1]:.4f}]")
    print(f"secondary: manual blinded labels vs that function: agree {s['manual_vs_fn_agree']}/{s['n']}, "
          f"kappa {s['manual_vs_fn_kappa']:.4f}")
    print()
    print_confusion_matrix(s["categories"], s["matrix"], label_a="scorer_classification", label_b="annotator_claude")
    print()
    print(f"disagreements ({len(s['disagreements'])}):")
    for d in s["disagreements"]:
        print(f"  {d['id']}: scorer={d['scorer']} ({d['method']}) annotator={d['annotator']} "
              f"expected={d['expected']!r} output={d['output']!r}")


def summarize(path: str, col_a: str, col_b: str) -> None:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    labels_a = [r[col_a] for r in rows]
    labels_b = [r[col_b] for r in rows]

    n = len(rows)
    agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    kappa = cohens_kappa(labels_a, labels_b)
    categories, matrix = confusion_matrix(labels_a, labels_b)

    print(f"n = {n}")
    print(f"raw agreement = {agree}/{n} = {agree / n:.4f}")
    print(f"Cohen's kappa = {kappa:.4f}")
    print()
    print_confusion_matrix(categories, matrix, label_a=col_a, label_b=col_b)


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "--heldout":
        from pathlib import Path
        summarize_heldout(Path(__file__).resolve().parents[1])
        return
    if len(sys.argv) != 4:
        raise SystemExit(f"usage: {sys.argv[0]} <csv_path> <col_a> <col_b>  |  {sys.argv[0]} --heldout")
    summarize(sys.argv[1], sys.argv[2], sys.argv[3])


if __name__ == "__main__":
    main()
