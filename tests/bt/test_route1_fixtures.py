"""ROUTE-1 hand-computed fixtures (synthetic sessions only; $ per 1,000 sessions, abs tol 0.01)."""

from __future__ import annotations

import pytest

from seam.route1 import policies as pol
from seam.route1.device import I8_RESIDENT, Constants
from seam.route1.prices import FLASH_LITE, HAIKU, SONNET, PriceSheet
from seam.route1.session import synthetic
from seam.route1.sim import default_policies, usd_per_1000

C = Constants(B=3.27e9)
S = synthetic(4000, 1500, 30, 300)


def _run(name: str, sheet: PriceSheet) -> pol.Result:
    return default_policies()[name](S, sheet, C)


@pytest.mark.parametrize(
    ("name", "usd", "turn"),
    [
        ("all-cloud", 281.25, 1),
        ("D1-stateless", 262.75, 6),
        ("D1-hardfail", 235.60, 12),
        ("A0", 235.60, 12),
        ("D1-resident", 165.10, 24),
        ("ROUTE-1", 151.25, 26),
        ("Oracle", 151.25, 26),
        ("B-4K", 277.85, 2),
        ("B-8K", 270.60, 4),
    ],
)
def test_sonnet_fixtures(name: str, usd: float, turn: int) -> None:
    r = _run(name, SONNET)
    assert usd_per_1000([r]) == pytest.approx(usd, abs=0.01)
    assert r.first_cloud_turn == turn


def test_a0_has_no_failed_attempts() -> None:
    assert _run("A0", SONNET).failed_attempts == 0


def test_route1_i8_resident_margin() -> None:
    r = pol.route1(S, SONNET, C, configs=(I8_RESIDENT,))
    assert r.first_cloud_turn == 22
    assert usd_per_1000([r]) == pytest.approx(178.35, abs=0.01)


def test_route1_picks_u4_and_never_fails() -> None:
    r = _run("ROUTE-1", SONNET)
    assert r.config == "u4-c256+resident"
    assert r.failed_attempts == 0 and r.silent_overruns == 0


@pytest.mark.parametrize(
    ("sheet", "name", "usd"),
    [
        (HAIKU, "all-cloud", 17.69),
        (HAIKU, "D1-hardfail", 14.77),
        (HAIKU, "ROUTE-1", 8.44),
        (FLASH_LITE, "all-cloud", 43.50),
        (FLASH_LITE, "D1-hardfail", 35.39),
        (FLASH_LITE, "ROUTE-1", 18.50),
    ],
)
def test_cheap_tier_fixtures(sheet: PriceSheet, name: str, usd: float) -> None:
    assert usd_per_1000([_run(name, sheet)]) == pytest.approx(usd, abs=0.01)


def test_a2_kv_only_fails_13_times() -> None:
    assert _run("A2", SONNET).failed_attempts == 13


def test_margin_one_turn_early_costs() -> None:
    def cost_from(turn: int) -> float:
        run = pol._Run("x", S, SONNET, pol.TTFT_PARAMS)
        for t in range(len(S)):
            if t + 1 >= turn:
                run.cloud(t)
        return run.r.usd

    assert cost_from(11) - cost_from(12) == pytest.approx(0.0049, abs=1e-6)


def test_a1_overruns_window_silently() -> None:
    r = _run("A1", SONNET)
    assert r.silent_overruns >= 1 and r.failed_attempts == 0
