"""DUET closed-form rules, TTFT model, kappa unit and planner sanity (synthetic only)."""

from __future__ import annotations

import pytest

from seam.duet import keepalive as ka
from seam.duet import params as prm
from seam.duet import planner as pl
from seam.duet import ttft
from seam.duet.device import cold_mem_ok, session_cap
from seam.duet.ladder import table
from seam.duet.shapes import big_share


def test_keepalive_and_compaction_rules() -> None:
    assert ka.ping_threshold(prm.SONNET) == pytest.approx(0.0417, abs=1e-4)
    assert ka.ping_threshold(prm.HAIKU) == pytest.approx(0.087, abs=1e-3)
    assert ka.compaction_r_star(0.25, prm.SONNET) == pytest.approx(8.3, abs=0.05)
    assert ka.max_pings(prm.SONNET) == pytest.approx(24)
    assert ka.one_hour_breakeven_min(prm.SONNET) == pytest.approx(67, abs=1)
    assert ka.one_hour_breakeven_min(prm.HAIKU) == pytest.approx(34, abs=1)


def test_kappa_unit_is_microseconds() -> None:
    # 2K-token turn at 16,473 cached took 11.2 s (d7e8b43e): 0.34 us per pair, not ns.
    kappa_s = 11.2 / (2000 * 16473)
    assert kappa_s == pytest.approx(0.34e-6, rel=0.01)
    assert abs(prm.BETA - 0.339e-6) < 1e-12
    assert all(0.2 < v < 1.0 for cell in prm.KAPPA_US.values() for v in cell.values())


def test_ttft_model_anchors_conservative() -> None:
    assert ttft.ext(1000, 16561) == pytest.approx(6.9, abs=0.05)  # measured 5.97 s
    assert ttft.ext(2000, 61771) == pytest.approx(57.0, abs=0.3)  # measured 53.1 s
    assert ttft.ext(4700, 10100) == pytest.approx(15.7, abs=0.1)
    assert ttft.cold(10100) == pytest.approx(9.9, abs=0.05)
    assert ttft.reprefill_wins(4700, 10100)
    # Model crossover n_new/P ~0.25-0.4: REPREFILL wins at 4,700/10,100 (0.47), not at 1,500/15,000 (0.1).
    # At the measured 4K@16.4K anchor (0.24) the model favours REPREFILL narrowly, while the measured
    # extend (20.4 s) still won; the addendum records that the true crossover is likely 0.3-0.4.
    assert not ttft.reprefill_wins(1500, 15000)


def test_device_caps() -> None:
    assert session_cap("u4", 1500, prm.MARGIN_M) == prm.W - prm.WINDOW_HEADROOM
    assert round(session_cap("i8", 1500, 0.0)) == 37_288
    assert cold_mem_ok(10_100, "i8") and not cold_mem_ok(20_000, "i8")


def test_planner_background_picks_u4_interactive_migrates_earlier() -> None:
    bg = pl.plan(4000, 1500, 30, 300, prm.SONNET, slo=False)
    it = pl.plan(4000, 1500, 30, 300, prm.SONNET, slo=True)
    assert bg.prec == "u4" and it.migrate_turn < bg.migrate_turn


def test_loop_b_step() -> None:
    assert pl.step(700, 4000, 9400, "i8", prm.SONNET, slo=True) == "EXCURSION_STUB"
    assert pl.step(700, 0, 9400, "i8", prm.SONNET, slo=True) == "EXTEND"


def test_ladder_table_and_generator() -> None:
    t = table()
    assert "SYNTHETIC" in t and t.count("\n| ") == 9 + 1
    s = big_share(30, 0.2, seed=7)
    assert len(s) == 30 and s == big_share(30, 0.2, seed=7)
