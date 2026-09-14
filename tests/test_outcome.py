"""Tests for the four-way outcome classifier in evaluation/outcome.py."""

from __future__ import annotations

import pytest

from evaluation.outcome import (
    CORRECT,
    FABRICATED,
    REFUSED,
    UNCLASSIFIABLE,
    _extract_last_numeric_token,
    classify,
)


# ─────────────────────────────────────── _extract_last_numeric_token

class TestExtractLastNumericToken:
    def test_simple_number(self):
        assert _extract_last_numeric_token("26") == "26"

    def test_expression_with_equals(self):
        # "130/5=26" → last token "26"
        assert _extract_last_numeric_token("130/5=26") == "26"

    def test_sum_expression(self):
        # "1130+612+765=2507" → last token "2507"
        assert _extract_last_numeric_token("1130+612+765=2507") == "2507"

    def test_embedded_in_text(self):
        assert _extract_last_numeric_token("The answer is 42.") == "42"

    def test_decimal(self):
        assert _extract_last_numeric_token("0.0147") == "0.0147"

    def test_negative(self):
        assert _extract_last_numeric_token("-3.5") == "-3.5"

    def test_no_number(self):
        assert _extract_last_numeric_token("no numbers here") is None

    def test_word_boundary_prevents_substring(self):
        # "1200" should not match as "120" — word boundary must be respected
        tok = _extract_last_numeric_token("1200")
        assert tok == "1200"
        assert tok != "120"

    def test_last_wins_when_multiple(self):
        assert _extract_last_numeric_token("first 10 then 20") == "20"


# ─────────────────────────────────────── classify — UNCLASSIFIABLE

class TestClassifyUnclassifiable:
    def test_none_output(self):
        result = classify(
            output=None, expected="26", scorer_type="exact", score=0.0
        )
        assert result["outcome_class"] == UNCLASSIFIABLE
        assert result["format_compliant"] is None
        assert result["classification_method"] == "unclassifiable"

    def test_empty_output(self):
        result = classify(
            output="", expected="26", scorer_type="exact", score=0.0
        )
        assert result["outcome_class"] == UNCLASSIFIABLE

    def test_done_reason_length(self):
        result = classify(
            output="partial",
            expected="26",
            scorer_type="exact",
            score=0.0,
            done_reason="length",
        )
        assert result["outcome_class"] == UNCLASSIFIABLE
        assert result["classification_method"] == "unclassifiable"

    def test_done_reason_budget_exhausted(self):
        result = classify(
            output="partial",
            expected="26",
            scorer_type="exact",
            score=0.0,
            done_reason="budget_exhausted",
        )
        assert result["outcome_class"] == UNCLASSIFIABLE


# ─────────────────────────────────────── classify — REFUSED

class TestClassifyRefused:
    def test_insufficient_context_sentinel(self):
        result = classify(
            output="INSUFFICIENT_CONTEXT",
            expected="26",
            scorer_type="exact",
            score=0.0,
        )
        assert result["outcome_class"] == REFUSED
        assert result["format_compliant"] is None
        assert result["classification_method"] == "refused_sentinel"

    def test_insufficient_context_embedded(self):
        # Sentinel string may appear in longer output
        result = classify(
            output="Based on the context: INSUFFICIENT_CONTEXT.",
            expected="26",
            scorer_type="exact",
            score=0.0,
        )
        assert result["outcome_class"] == REFUSED
        assert result["classification_method"] == "refused_sentinel"

    def test_abstention_phrase(self):
        result = classify(
            output="The provided text does not mention any array or drift.",
            expected="A9",
            scorer_type="exact",
            score=0.0,
        )
        assert result["outcome_class"] == REFUSED
        assert result["classification_method"] == "refused_abstention"

    def test_abstention_cannot_determine(self):
        result = classify(
            output="Cannot determine from the given text.",
            expected="120",
            scorer_type="exact",
            score=0.0,
        )
        assert result["outcome_class"] == REFUSED

    def test_hedge_then_answer_is_fabricated_not_refused(self):
        # Hedge + answer pivot → FABRICATED, not REFUSED
        result = classify(
            output="Cannot determine, but it is approximately 0.15.",
            expected="0.15",
            scorer_type="exact",
            score=0.0,
        )
        assert result["outcome_class"] == FABRICATED


# ─────────────────────────────────────── classify — CORRECT

class TestClassifyCorrect:
    def test_score_1_is_correct_format_compliant(self):
        result = classify(
            output="26",
            expected="26",
            scorer_type="exact",
            score=1.0,
        )
        assert result["outcome_class"] == CORRECT
        assert result["format_compliant"] is True
        assert result["classification_method"] == "score"

    def test_format_non_compliant_last_token_matches(self):
        # "130/5=26" — last token "26" matches expected "26"
        result = classify(
            output="130/5=26",
            expected="26",
            scorer_type="exact",
            score=0.0,
        )
        assert result["outcome_class"] == CORRECT
        assert result["format_compliant"] is False
        assert result["classification_method"] == "last_token"

    def test_sum_expression_last_token(self):
        # "1130+612+765=2507" — last token "2507" matches "2507"
        result = classify(
            output="1130+612+765=2507",
            expected="2507",
            scorer_type="exact",
            score=0.0,
        )
        assert result["outcome_class"] == CORRECT
        assert result["format_compliant"] is False

    def test_non_exact_scorer_no_last_token_check(self):
        # For span_match, last-token check does not apply
        result = classify(
            output="some answer text",
            expected="some answer text",
            scorer_type="span_match",
            score=1.0,
        )
        assert result["outcome_class"] == CORRECT
        assert result["format_compliant"] is True


# ─────────────────────────────────────── classify — FABRICATED

class TestClassifyFabricated:
    def test_confident_wrong_answer(self):
        result = classify(
            output="0.0001",
            expected="0.15",
            scorer_type="exact",
            score=0.0,
        )
        assert result["outcome_class"] == FABRICATED
        assert result["format_compliant"] is False

    def test_rea_05_unit_error(self):
        # "1200" when "120" expected: last token "1200" ≠ "120" → FABRICATED
        # "120" is a substring of "1200" but word-boundary matching prevents match
        result = classify(
            output="1200",
            expected="120",
            scorer_type="exact",
            score=0.0,
        )
        assert result["outcome_class"] == FABRICATED
        assert result["classification_method"] == "score"

    def test_short_categorical_wrong(self):
        result = classify(
            output="Meridian-3",
            expected="Meridian-7",
            scorer_type="exact",
            score=0.0,
        )
        assert result["outcome_class"] == FABRICATED

    def test_non_exact_scorer_fabricated(self):
        # For non-exact scorers, format_compliant is None on FABRICATED
        result = classify(
            output="wrong answer",
            expected="correct answer",
            scorer_type="span_match",
            score=0.0,
        )
        assert result["outcome_class"] == FABRICATED
        assert result["format_compliant"] is None


# ─────────────────────────────────────── classify — priority ordering

class TestClassifyPriorityOrder:
    def test_unclassifiable_beats_refused(self):
        # done_reason=length on an output that would otherwise be abstention
        result = classify(
            output="Cannot determine from text.",
            expected="26",
            scorer_type="exact",
            score=0.0,
            done_reason="length",
        )
        assert result["outcome_class"] == UNCLASSIFIABLE

    def test_refused_beats_correct_via_last_token(self):
        # Abstention output that happens to end in "26": REFUSED wins
        result = classify(
            output="The document does not mention the value 26.",
            expected="26",
            scorer_type="exact",
            score=0.0,
        )
        assert result["outcome_class"] == REFUSED

    def test_score_1_beats_abstention_language(self):
        # If scorer returns 1.0 on an output containing abstention language,
        # CORRECT wins (score is the authoritative signal once we pass UNCLASSIFIABLE).
        result = classify(
            output="Not provided in context. Score: 1.0",
            expected="something",
            scorer_type="exact",
            score=1.0,
        )
        assert result["outcome_class"] == CORRECT
