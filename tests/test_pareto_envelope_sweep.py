"""Unit test for analysis/pareto.py's generate_envelope_pareto_sweep (B6c)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pareto import generate_envelope_pareto_sweep  # noqa: E402


def test_sweep_uses_exactly_the_real_r1b_ratio_points():
    out = generate_envelope_pareto_sweep(quality_floors=[0.0], spend_caps_usd=[0.0])
    assert out["n_real_points"] == 4  # R1B_BASELINE_SCORE_BY_RATIO has exactly 4 real ratio buckets
    ratios = sorted(p["real_score_at_ratio"] for p in out["points"])
    assert ratios == [0.0, 0.0, 0.3, 1.0]  # the real measured scores at 0.4, 0.85, 0.98, 1.2


def test_sweep_fractions_sum_to_one_per_cell():
    out = generate_envelope_pareto_sweep(quality_floors=[0.0, 0.5, 1.0], spend_caps_usd=[0.0, 50.0])
    for row in out["sweep"]:
        total = row["frac_local"] + row["frac_cloud"] + row["frac_compact"] + row["frac_refuse"]
        assert abs(total - 1.0) < 1e-9


def test_raising_quality_floor_never_increases_local_fraction():
    out = generate_envelope_pareto_sweep(quality_floors=[0.0, 0.3, 0.5, 0.8, 1.0], spend_caps_usd=[50.0])
    by_floor = {row["quality_floor"]: row["frac_local"] for row in out["sweep"]}
    floors = sorted(by_floor)
    for a, b in zip(floors, floors[1:]):
        assert by_floor[b] <= by_floor[a] + 1e-9


def test_note_states_this_is_a_partial_real_sweep_not_a_full_frontier():
    out = generate_envelope_pareto_sweep()
    assert "NOT a full Pareto frontier" in out["note"]
    assert "B4 not built" in out["note"]
