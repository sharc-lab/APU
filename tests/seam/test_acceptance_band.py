"""The acceptance band is derived from the runs, not asserted.

s is the larger of the two runs' within-run relative spread; the tolerance is band_multiple times
it; and a spread too wide to support a claim fails regardless of how close the ratio came out.
"""

from __future__ import annotations

from typing import Any

import pytest

from seam.tools.fixed_throughput import _stats, criterion_rule, derive_verdict

CRITERION: dict[str, Any] = {
    "basis": "r_decode_tok_s",
    "spread_estimator": "cv",
    "band_multiple": 2.0,
    "max_within_run_spread": 0.10,
    "prior_unexplained_ratio": 1.98,
    "gate_mode": "derived_from_within_run_spread",
    "gap_s": 1200,
    "rule": (
        "s = max over the two runs of the within-run relative spread on `basis`; "
        "band = band_multiple * s; PASS iff |ratio - 1| <= band AND "
        "s <= max_within_run_spread"
    ),
}


def summary(median: float, cv: float | None) -> dict[str, Any]:
    return {"r_decode_tok_s": {"median": median, "cv": cv}}


def test_stats_reports_cv_as_the_band_input() -> None:
    stats = _stats([10.0, 11.0, 12.0])
    assert stats["median"] == 11.0
    assert stats["cv"] == pytest.approx(1.0 / 11.0)
    assert stats["n"] == 3
    assert stats["n_missing"] == 0


def test_stats_reports_absent_spread_as_none_not_nan() -> None:
    """nan does not survive a JSON round-trip, and the band is reconstructed from sealed JSON."""
    assert _stats([10.0])["cv"] is None
    assert _stats([None, None])["cv"] is None


def test_band_is_twice_the_larger_within_run_spread() -> None:
    gate = derive_verdict(
        summary_a=summary(100.0, 0.02),
        summary_b=summary(103.0, 0.05),
        criterion=CRITERION,
    )
    assert gate["s"] == pytest.approx(0.05)
    assert gate["s_source"] == "run_b"
    assert gate["band"] == pytest.approx(0.10)
    assert gate["ratio"] == pytest.approx(1.03)
    assert gate["verdict"] == "PASS"


def test_ratio_outside_the_derived_band_fails() -> None:
    gate = derive_verdict(
        summary_a=summary(100.0, 0.01),
        summary_b=summary(110.0, 0.01),
        criterion=CRITERION,
    )
    assert gate["band"] == pytest.approx(0.02)
    assert gate["verdict"] == "FAIL"
    assert any("exceeds the derived band" in r for r in gate["reasons"])


def test_a_noisy_pair_cannot_pass_on_the_strength_of_its_own_noise() -> None:
    """|ratio - 1| <= 2s is satisfiable by inflating s; the spread ceiling is what stops that."""
    gate = derive_verdict(
        summary_a=summary(100.0, 0.25),
        summary_b=summary(140.0, 0.25),
        criterion=CRITERION,
    )
    assert gate["within_band"] is True, "the ratio is inside the band the noise itself created"
    assert gate["spread_acceptable"] is False
    assert gate["verdict"] == "FAIL"
    assert any("cannot support a claim" in r for r in gate["reasons"])


def test_spread_exactly_at_the_ceiling_is_accepted() -> None:
    gate = derive_verdict(
        summary_a=summary(100.0, 0.10),
        summary_b=summary(101.0, 0.10),
        criterion=CRITERION,
    )
    assert gate["spread_acceptable"] is True
    assert gate["verdict"] == "PASS"


def test_ratio_is_orientation_free() -> None:
    """Run order must not change the verdict."""
    forward = derive_verdict(
        summary_a=summary(100.0, 0.03),
        summary_b=summary(104.0, 0.03),
        criterion=CRITERION,
    )
    reverse = derive_verdict(
        summary_a=summary(104.0, 0.03),
        summary_b=summary(100.0, 0.03),
        criterion=CRITERION,
    )
    assert forward["ratio"] == pytest.approx(reverse["ratio"])
    assert forward["verdict"] == reverse["verdict"]
    assert forward["signed_ratio_b_over_a"] != pytest.approx(
        reverse["signed_ratio_b_over_a"]
    ), "direction of drift is still recorded"


def test_underivable_band_is_indeterminate_never_a_pass() -> None:
    gate = derive_verdict(
        summary_a=summary(100.0, None),
        summary_b=summary(100.0, 0.02),
        criterion=CRITERION,
    )
    assert gate["s"] is None
    assert gate["band"] is None
    assert gate["verdict"] == "INDETERMINATE"


def test_the_prior_ratio_would_fail_at_any_admissible_spread() -> None:
    """1.98x is the number to beat; at the widest spread the gate tolerates it still fails."""
    gate = derive_verdict(
        summary_a=summary(100.0, 0.10),
        summary_b=summary(198.0, 0.10),
        criterion=CRITERION,
    )
    assert gate["ratio"] == pytest.approx(1.98)
    assert gate["band"] == pytest.approx(0.20)
    assert gate["verdict"] == "FAIL"


def test_every_input_to_the_bound_is_recorded() -> None:
    """The bound has to be reconstructible from the sealed record alone."""
    gate = derive_verdict(
        summary_a=summary(100.0, 0.02),
        summary_b=summary(103.0, 0.05),
        criterion=CRITERION,
    )
    for key in (
        "s",
        "s_source",
        "band",
        "band_multiple",
        "max_within_run_spread",
        "within_run_spread",
        "medians",
        "ratio",
        "abs_ratio_minus_one",
        "within_band",
        "spread_acceptable",
        "verdict",
        "reasons",
        "derivation",
    ):
        assert key in gate, key
    assert gate["band"] == pytest.approx(gate["band_multiple"] * gate["s"])


def test_criterion_equality_ignores_this_run_within_run_spread() -> None:
    """Per-run CV is an observation; nesting it under acceptance_criterion must not block compare.

    The orchestrated pair crashed because whole-dict equality treated differing measured spreads
    as different criteria. Rule-field equality lets those summaries reach derive_verdict.
    """
    criterion_a = {
        **CRITERION,
        "this_run_within_run_spread": {"basis": "r_decode_tok_s", "cv": 0.14469900338790184},
    }
    criterion_b = {
        **CRITERION,
        "this_run_within_run_spread": {"basis": "r_decode_tok_s", "cv": 0.018422416614057052},
    }
    assert criterion_a != criterion_b
    assert criterion_rule(criterion_a) == criterion_rule(criterion_b)

    gate = derive_verdict(
        summary_a=summary(16.641574962173326, 0.14469900338790184),
        summary_b=summary(17.375979740969534, 0.018422416614057052),
        criterion=criterion_a,
    )
    assert gate["ratio"] == pytest.approx(17.375979740969534 / 16.641574962173326)
    assert gate["s"] == pytest.approx(0.14469900338790184)
    assert gate["spread_acceptable"] is False
    assert gate["verdict"] == "FAIL"
    assert any("cannot support a claim" in r for r in gate["reasons"])
