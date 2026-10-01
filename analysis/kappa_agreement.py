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
    if len(sys.argv) != 4:
        raise SystemExit(f"usage: {sys.argv[0]} <csv_path> <col_a> <col_b>")
    summarize(sys.argv[1], sys.argv[2], sys.argv[3])


if __name__ == "__main__":
    main()
