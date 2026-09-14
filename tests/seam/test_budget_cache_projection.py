"""Cache-state-aware cost estimates (Phase E 0.3). Authorization stays worst-case."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from seam.budget import BudgetGuard, BudgetLedger, PricingTable

REPO_ROOT = Path(__file__).resolve().parents[1]
PRICING_PATH = REPO_ROOT / "configs" / "pricing" / "anthropic.yaml"


@pytest.fixture
def guard(tmp_path: Path) -> BudgetGuard:
    return BudgetGuard(
        ledger=BudgetLedger(tmp_path / "budget"),
        pricing=PricingTable.load(PRICING_PATH),
        slice_ceiling_usd=10.0,
        project_ceiling_usd=50.0,
        phase="test",
    )


def test_warm_estimate_is_cheaper_than_cold(guard: BudgetGuard) -> None:
    cold = guard.projected_call_usd(
        model="claude-sonnet-5",
        prompt_tokens=30_374,
        max_tokens=32,
        cache_state="cold",
        when=date(2026, 8, 2),
    )
    warm = guard.projected_call_usd(
        model="claude-sonnet-5",
        prompt_tokens=30_374,
        max_tokens=32,
        cache_state="warm",
        uncached_suffix_tokens=110,
        when=date(2026, 8, 2),
    )
    assert warm < cold
    # Matches the Phase E pattern: warm ~$0.0066 vs cold write ~$0.076.
    assert warm == pytest.approx(0.0065928, rel=1e-4)
    assert cold == pytest.approx(0.076255, rel=1e-4)


def test_authorize_still_uses_worst_case_even_when_estimate_is_warm(
    guard: BudgetGuard,
) -> None:
    """A warm estimate must not authorize a call the cold bound cannot afford."""
    # Tiny ceiling: warm estimate would fit, cold would not.
    guard.slice_ceiling_usd = 0.01
    warm_est = guard.projected_call_usd(
        model="claude-sonnet-5",
        prompt_tokens=30_374,
        max_tokens=32,
        cache_state="warm",
        when=date(2026, 8, 2),
    )
    assert warm_est < 0.01
    from seam.errors import BudgetExceededError

    with pytest.raises(BudgetExceededError):
        guard.authorize_call(
            model="claude-sonnet-5", prompt_tokens=30_374, max_tokens=32, when=date(2026, 8, 2)
        )


def test_unknown_cache_state_matches_cold(guard: BudgetGuard) -> None:
    a = guard.projected_call_usd(
        model="claude-sonnet-5", prompt_tokens=1000, max_tokens=10, cache_state="unknown"
    )
    b = guard.projected_call_usd(
        model="claude-sonnet-5", prompt_tokens=1000, max_tokens=10, cache_state="cold"
    )
    assert a == b
