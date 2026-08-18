"""Tests for the abstention classifier and outcome_class function in scorers.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "scorers", Path(__file__).parent.parent / "evaluation" / "probes" / "scorers.py"
)
scorers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scorers)

classify_abstention = scorers.classify_abstention
outcome_class = scorers.outcome_class


# ───────────────────────────────────────────────────────────── abstentions

ABSTAIN_CASES = [
    # rag_02 observed outputs (verbatim from Stage C EARLY truncated rows)
    (
        "The provided text does not mention any array or drift measurements, "
        "so it is impossible to determine which array shows drift near 0.9% "
        "per month or why.",
        True,
    ),
    (
        "The provided text does not mention any array or drift, nor does it "
        "contain information about performance metrics or deviations from "
        "baseline. Therefore, it is impossible to determine which array shows "
        "drift near 0.9% per month or why.",
        True,
    ),
    (
        "The provided text does not mention any array or drift, so it is "
        "impossible to determine which array shows drift near 0.9% per month "
        "or why.",
        True,
    ),
    # Minimal abstentions
    ("No information is provided to answer this question.", True),
    ("The documents do not contain this value.", True),
    ("Cannot determine from the given text.", True),
    ("Unable to determine — not stated in the source material.", True),
    ("Insufficient information to answer.", True),
    ("Not mentioned in the provided context.", True),
]


@pytest.mark.parametrize("output,expected", ABSTAIN_CASES)
def test_classify_abstention_true(output, expected):
    assert classify_abstention(output) is expected


# ────────────────────────────────────────────── confident wrong answers

FABRICATION_CASES = [
    # Observed Stage C EARLY truncated outputs — no abstention language
    ("0.0001", False),
    ("0.00", False),
    ("1", False),
    ("93850", False),
    ("A9", False),
    ("SE", False),
    ("134.00", False),
    ("B2", False),
    # Bare number
    ("3", False),
    # Short categorical answer
    ("Meridian-3", False),
]


@pytest.mark.parametrize("output,expected", FABRICATION_CASES)
def test_classify_abstention_false_fabrication(output, expected):
    assert classify_abstention(output) is expected


# ────────────────────────────────────────── hedge-then-answer (fabrication)

HEDGE_THEN_ANSWER_CASES = [
    (
        "I cannot determine from the provided text, but it is approximately 0.15.",
        False,
    ),
    (
        "No information is given, but the answer would be around 134.00.",
        False,
    ),
    (
        "The documents do not contain this value, however it is likely A9.",
        False,
    ),
    (
        "Unable to determine from context. However, based on typical values "
        "it is probably 0.8.",
        False,
    ),
    (
        "Not stated here, so the answer would be Meridian-3.",
        False,
    ),
]


@pytest.mark.parametrize("output,expected", HEDGE_THEN_ANSWER_CASES)
def test_classify_abstention_false_hedge_then_answer(output, expected):
    assert classify_abstention(output) is expected


# ─────────────────────────────────────────────────────────── outcome_class

class TestOutcomeClass:
    def test_correct(self):
        assert outcome_class("0.15", 1.0, truncated=False) == "correct"

    def test_correct_truncated(self):
        # score=1.0 is correct regardless of truncated flag
        assert outcome_class("0.15", 1.0, truncated=True) == "correct"

    def test_error(self):
        assert outcome_class(None, None, truncated=False) == "error"

    def test_error_truncated(self):
        assert outcome_class(None, None, truncated=True) == "error"

    def test_abstained(self):
        out = "The provided text does not mention any array or drift."
        assert outcome_class(out, 0.0, truncated=True) == "abstained"

    def test_incorrect_confident(self):
        assert outcome_class("0.0001", 0.0, truncated=True) == "incorrect"

    def test_incorrect_hedge_then_answer(self):
        out = "Cannot determine from context, but it is approximately 0.15."
        assert outcome_class(out, 0.0, truncated=True) == "incorrect"

    def test_score_takes_precedence_over_abstention_when_correct(self):
        # If somehow the scorer returns 1.0 on an abstaining output, correct wins.
        out = "The provided text does not mention this."
        assert outcome_class(out, 1.0, truncated=False) == "correct"
