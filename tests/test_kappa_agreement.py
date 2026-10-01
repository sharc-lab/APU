"""Unit tests for analysis/kappa_agreement.py: Cohen's kappa and confusion matrix.

Fixture cases are hand-computed (worked in the comments) rather than taken from a library, so a
regression in the formula implementation is caught against an independently-checked number.
"""
from __future__ import annotations

import pytest

from kappa_agreement import cohens_kappa, confusion_matrix


def test_perfect_agreement_is_one():
    a = ["CORRECT", "REFUSED", "FABRICATED", "CORRECT", "REFUSED", "FABRICATED"]
    b = list(a)
    assert cohens_kappa(a, b) == pytest.approx(1.0)


def test_chance_level_agreement_is_zero():
    # A = [x,x,y,y], B = [x,y,x,y]: 2/4 raw agreement, but marginals are 50/50 for both raters on
    # both categories, so p_e = .5*.5 + .5*.5 = .5 = p_o -> kappa = 0 exactly.
    a = ["x", "x", "y", "y"]
    b = ["x", "y", "x", "y"]
    assert cohens_kappa(a, b) == pytest.approx(0.0, abs=1e-9)


def test_hand_computed_binary_case():
    # Classic 2x2 worked example, 100 items:
    #            B=yes  B=no
    #   A=yes      20     5      (25)
    #   A=no       10    65      (75)
    #             (30)  (70)
    # p_o = (20+65)/100 = 0.85
    # p(A=yes)=.25, p(B=yes)=.30 -> p_e = .25*.30 + .75*.70 = .075 + .525 = 0.60
    # kappa = (0.85 - 0.60) / (1 - 0.60) = 0.25 / 0.40 = 0.625
    a = ["yes"] * 25 + ["no"] * 75
    b = ["yes"] * 20 + ["no"] * 5 + ["yes"] * 10 + ["no"] * 65
    assert cohens_kappa(a, b) == pytest.approx(0.625)


def test_hand_computed_three_category_case():
    # A = [C, C, R, R, F, F]; B = [C, F, R, R, F, C]
    # Raw agreement: positions 0,2,3,4 match -> 4/6 = 0.66667
    # Marginals are identical for both raters (2 C, 2 R, 2 F each) -> p(k) = 1/3 for every category
    # p_e = 3 * (1/3 * 1/3) = 1/3 = 0.33333
    # kappa = (0.66667 - 0.33333) / (1 - 0.33333) = 0.33333 / 0.66667 = 0.5
    a = ["C", "C", "R", "R", "F", "F"]
    b = ["C", "F", "R", "R", "F", "C"]
    assert cohens_kappa(a, b) == pytest.approx(0.5)


def test_kappa_can_be_negative_for_systematic_disagreement():
    # Raters that systematically disagree do worse than chance -> kappa < 0.
    a = ["x", "x", "y", "y"]
    b = ["y", "y", "x", "x"]
    assert cohens_kappa(a, b) < 0


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        cohens_kappa(["x"], ["x", "y"])


def test_empty_input_raises():
    with pytest.raises(ValueError):
        cohens_kappa([], [])


def test_confusion_matrix_counts():
    a = ["C", "C", "R", "F"]
    b = ["C", "F", "R", "R"]
    categories, matrix = confusion_matrix(a, b)
    assert categories == ["C", "F", "R"]
    assert matrix[("C", "C")] == 1
    assert matrix[("C", "F")] == 1
    assert matrix[("R", "R")] == 1
    assert matrix[("F", "R")] == 1
    # every other cell is zero
    assert sum(matrix.values()) == 4
    assert matrix[("F", "C")] == 0


def test_confusion_matrix_includes_category_seen_by_only_one_rater():
    a = ["C", "C"]
    b = ["C", "X"]
    categories, matrix = confusion_matrix(a, b)
    assert "X" in categories
    assert matrix[("C", "X")] == 1
    assert matrix[("X", "C")] == 0
