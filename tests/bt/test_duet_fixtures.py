"""DUET addendum fixtures (synthetic shapes only; $ per 1,000 sessions, abs tol 0.01; turns 1-based).

On a miss, the failure message carries the per-turn trace. Constants are never tuned to match.
"""

from __future__ import annotations

import pytest

from seam.duet import params as prm
from seam.duet.policies import simulate
from seam.duet.shapes import DTurn, shape_h, shape_t, shape_u

SHAPES = {"U": shape_u, "H": shape_h, "T": shape_t}


@pytest.mark.parametrize(
    ("shape", "name", "usd", "turn"),
    [
        ("U", "D0", 281.25, None),
        ("U", "D1-stateless", 262.75, 6),
        ("U", "D1-resident", 245.25, 10),
        ("U", "DUET-HW", 245.25, 10),
        ("U", "D3", 194.85, None),
        ("U", "config-only-bg", 151.25, 26),
        ("U", "D1-bg", 165.10, 24),
        ("U", "D3-bg", 87.40, None),
        ("U", "DUET-bg", 73.25, None),
        ("H", "D0", 283.17, None),
        ("H", "D1-resident", 272.76, 4),
        ("H", "DUET-HW", 145.54, 9),
        ("H", "DUET-HW-excursion-only", 149.55, None),
        ("H", "DUET-HW-reprefill-only", 252.01, None),
        ("H", "D3", 121.67, None),
        ("H", "DUET-full", 65.18, None),
        ("T", "D0", 1273.80, None),
        ("T", "D1-resident", 1264.10, None),
        ("T", "DUET-HW", 1199.30, None),
        ("T", "D3", 724.77, None),
    ],
)
def test_sonnet(shape: str, name: str, usd: float, turn: int | None) -> None:
    r = simulate(name, SHAPES[shape]())
    trace = "\n".join(r.trace)
    assert r.usd * 1000 == pytest.approx(usd, abs=0.01), trace
    if turn is not None:
        assert r.first_cloud_turn == turn, trace


def test_duet_hw_h_shape_counts() -> None:
    r = simulate("DUET-HW", shape_h())
    assert (r.local_turns, r.excursions, r.first_cloud_turn) == (25, 5, 9)
    assert simulate("DUET-HW", shape_u()).waste_s == 0.0


@pytest.mark.parametrize(
    ("shape", "name", "usd"),
    [("U", "D0", 17.69), ("H", "D1-resident", 17.19), ("H", "DUET-HW", 7.85)],
)
def test_haiku(shape: str, name: str, usd: float) -> None:
    assert simulate(name, SHAPES[shape](), prm.HAIKU).usd * 1000 == pytest.approx(usd, abs=0.01)


def test_kappa_blind_admits_14_slo_violations() -> None:
    assert simulate("kappa-blind", shape_u()).slo_violations == 14


def test_nnew_blind_leaves_at_first_big_tool_output() -> None:
    assert (
        simulate("nnew-blind", shape_h()).first_cloud_turn
        == simulate("D1-resident", shape_h()).first_cloud_turn
        == 4
    )


def test_shapes() -> None:
    h = shape_h()
    assert [t for t, x in enumerate(h) if x.tool] == [3, 8, 13, 18, 23, 28]
    assert h[0] == DTurn(4000, 0, 300)
    assert [t for t, x in enumerate(shape_t()) if x.tool][:3] == [2, 6, 10]
