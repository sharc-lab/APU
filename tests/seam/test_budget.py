"""The budget ceiling must be structural, not advisory.

These tests exist because "we'll watch the spend" is not a control. Each one pins a property that,
if it silently broke, would let a run exceed the authorized ceiling without anything failing.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from seam.budget import BudgetGuard, BudgetLedger, CallUsage, PricingTable, usd_cost
from seam.errors import BudgetExceededError, ConfigError

REPO_ROOT = Path(__file__).resolve().parents[1]
PRICING_PATH = REPO_ROOT / "configs" / "pricing" / "anthropic.yaml"


@pytest.fixture
def pricing() -> PricingTable:
    return PricingTable.load(PRICING_PATH)


@pytest.fixture
def guard(tmp_path: Path, pricing: PricingTable) -> BudgetGuard:
    return BudgetGuard(
        ledger=BudgetLedger(tmp_path / "budget"),
        pricing=pricing,
        slice_ceiling_usd=10.0,
        project_ceiling_usd=50.0,
        phase="test",
    )


# --------------------------------------------------------------------------------------------
# Dated pricing
# --------------------------------------------------------------------------------------------


def test_pricing_is_date_keyed_not_constant(pricing: PricingTable) -> None:
    """The 2026-09-01 increase must actually change the rate.

    A constant would silently under-report every call made from September onward.
    """
    introductory = pricing.tier_for("claude-sonnet-5", date(2026, 8, 2))
    standard = pricing.tier_for("claude-sonnet-5", date(2026, 9, 1))

    assert introductory.label == "introductory"
    assert (introductory.input_per_mtok, introductory.output_per_mtok) == (2.0, 10.0)
    assert standard.label == "standard"
    assert (standard.input_per_mtok, standard.output_per_mtok) == (3.0, 15.0)


def test_pricing_boundary_is_inclusive_on_the_last_introductory_day(pricing: PricingTable) -> None:
    assert pricing.tier_for("claude-sonnet-5", date(2026, 8, 31)).label == "introductory"


def test_unknown_model_refuses_rather_than_guessing(pricing: PricingTable) -> None:
    with pytest.raises(ConfigError, match="Refusing to guess"):
        pricing.tier_for("claude-nonexistent", date(2026, 8, 2))


def test_cache_read_is_priced_separately_from_input(pricing: PricingTable) -> None:
    """Conflating cache reads with input tokens misprices a cached run by ~10x."""
    tier = pricing.tier_for("claude-sonnet-5", date(2026, 8, 2))
    cached = usd_cost(
        CallUsage(input_tokens=0, output_tokens=0, cache_read_input_tokens=1_000_000), tier
    )
    uncached = usd_cost(CallUsage(input_tokens=1_000_000, output_tokens=0), tier)
    assert cached == pytest.approx(0.20)
    assert uncached == pytest.approx(2.00)


# --------------------------------------------------------------------------------------------
# The ceiling
# --------------------------------------------------------------------------------------------


def test_call_is_refused_when_worst_case_exceeds_remaining(
    tmp_path: Path, pricing: PricingTable
) -> None:
    """The bound, not the expected cost, is what gets checked."""
    guard = BudgetGuard(
        ledger=BudgetLedger(tmp_path / "budget"),
        pricing=pricing,
        slice_ceiling_usd=0.001,
        project_ceiling_usd=50.0,
        phase="test",
    )
    with pytest.raises(BudgetExceededError, match="worst-case"):
        guard.authorize_call(model="claude-sonnet-5", prompt_tokens=100_000, max_tokens=4096)


def test_guard_stays_tripped_so_a_caller_cannot_retry_past_the_ceiling(
    tmp_path: Path, pricing: PricingTable
) -> None:
    """A loop that retries a refused call must not eventually succeed."""
    guard = BudgetGuard(
        ledger=BudgetLedger(tmp_path / "budget"),
        pricing=pricing,
        slice_ceiling_usd=0.0,
        project_ceiling_usd=50.0,
        phase="test",
    )
    for _ in range(3):
        with pytest.raises(BudgetExceededError):
            guard.authorize_call(model="claude-sonnet-5", prompt_tokens=1, max_tokens=1)
    assert guard.tripped is True


def test_run_is_refused_before_it_starts(guard: BudgetGuard) -> None:
    with pytest.raises(BudgetExceededError, match="refusing to start"):
        guard.authorize_run(projected_usd=999.0, description="main run")


def test_project_ceiling_binds_even_when_slice_ceiling_would_allow(
    tmp_path: Path, pricing: PricingTable
) -> None:
    """The tighter of the two ceilings must win."""
    guard = BudgetGuard(
        ledger=BudgetLedger(tmp_path / "budget"),
        pricing=pricing,
        slice_ceiling_usd=10.0,
        project_ceiling_usd=0.5,
        phase="test",
    )
    assert guard.remaining_usd() == pytest.approx(0.5)


def test_the_tighter_ceiling_wins_after_spend(guard: BudgetGuard) -> None:
    guard.record(
        model="claude-sonnet-5",
        usage=CallUsage(input_tokens=1_000_000, output_tokens=0),
        projected_usd=2.0,
        run_id="r1",
        when=date(2026, 8, 2),
    )
    assert guard.spent_usd() == pytest.approx(2.0)
    assert guard.remaining_usd() == pytest.approx(8.0)


# --------------------------------------------------------------------------------------------
# Durability and calibration
# --------------------------------------------------------------------------------------------


def test_ledger_survives_a_restart(tmp_path: Path, pricing: PricingTable) -> None:
    """A per-process ceiling is not a ceiling.

    If spend is not reloaded from disk, a crashed-and-restarted run resumes with the full budget
    available and can spend it a second time.
    """
    budget_dir = tmp_path / "budget"
    first = BudgetGuard(
        ledger=BudgetLedger(budget_dir),
        pricing=pricing,
        slice_ceiling_usd=10.0,
        project_ceiling_usd=50.0,
        phase="p1",
    )
    first.record(
        model="claude-sonnet-5",
        usage=CallUsage(input_tokens=1_000_000, output_tokens=100_000),
        projected_usd=3.0,
        run_id="r1",
        when=date(2026, 8, 2),
    )
    spent = first.spent_usd()
    assert spent == pytest.approx(3.0)

    reopened = BudgetGuard(
        ledger=BudgetLedger(budget_dir),
        pricing=pricing,
        slice_ceiling_usd=10.0,
        project_ceiling_usd=50.0,
        phase="p2",
    )
    assert reopened.spent_usd() == pytest.approx(spent)
    assert reopened.remaining_usd() == pytest.approx(7.0)


def test_calibration_reports_projection_error(guard: BudgetGuard) -> None:
    """A systematically low projection is how a guarded budget still gets overrun."""
    guard.record(
        model="claude-sonnet-5",
        usage=CallUsage(input_tokens=1_000_000, output_tokens=0),
        projected_usd=1.0,
        run_id="r1",
        when=date(2026, 8, 2),
    )
    calibration = guard.ledger.calibration()
    assert calibration["n_calls"] == 1
    # Actual $2.00 against a $1.00 projection: the projection was low by 100%.
    assert calibration["mean_rel_error"] == pytest.approx(1.0)


def test_checkpoint_reports_per_phase_breakdown(guard: BudgetGuard) -> None:
    guard.phase = "aa"
    guard.record(
        model="claude-sonnet-5",
        usage=CallUsage(input_tokens=500_000, output_tokens=0),
        projected_usd=1.0,
        run_id="r1",
        when=date(2026, 8, 2),
    )
    guard.phase = "noise_floor"
    guard.record(
        model="claude-sonnet-5",
        usage=CallUsage(input_tokens=500_000, output_tokens=0),
        projected_usd=1.0,
        run_id="r2",
        when=date(2026, 8, 2),
    )
    checkpoint = guard.checkpoint()
    assert checkpoint["by_phase"] == {
        "aa": pytest.approx(1.0),
        "noise_floor": pytest.approx(1.0),
    }
    assert checkpoint["spent_usd"] == pytest.approx(2.0)
