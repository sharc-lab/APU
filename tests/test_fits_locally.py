"""src/dse/fits_locally.py: the "fits locally?" predictor interface (operator brief 2026-10-26, item 3).

Proves (1) the baseline reproduces the old inline rules exactly, (2) route-at-start and the DSE go through the
interface, and (3) the register rows hybrid-route-at-start-measured and own-dse-v1 compute identically through it
(value text equal to the committed docs/NUMBERS_REGISTER.md line)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "analysis", REPO / "harness"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.dse import fits_locally as F  # noqa: E402
from src.dse import route_at_start as R  # noqa: E402


class Counting:
    """Wraps the baseline and counts calls, to prove a caller goes through the interface."""

    source = "counting-baseline"

    def __init__(self):
        self.n = 0
        self.inner = F.ProjectionBaseline()

    def predict(self, task_context, device_profile):
        self.n += 1
        return self.inner.predict(task_context, device_profile)


class AlwaysCloud:
    source = "always-cloud"

    def predict(self, task_context, device_profile):
        return {"fits": False, "projected_tokens": int(task_context["projected_tokens"]),
                "ceiling_tokens": 0, "source": self.source}


def _register_value(rid):
    rows = R.register_rows((REPO / "docs" / "NUMBERS_REGISTER.md").read_text(encoding="utf-8"))
    return rows[rid]["value"]


def test_protocol_and_result_shape():
    b = F.ProjectionBaseline()
    assert isinstance(b, F.FitsLocallyPredictor)
    assert isinstance(F.ZachCeilingPredictor(), F.FitsLocallyPredictor)
    r = b.predict({"projected_tokens": 8192}, {"window": 8192})
    assert r == {"fits": True, "projected_tokens": 8192, "ceiling_tokens": 8192, "source": "projection-baseline"}
    assert b.predict({"projected_tokens": 8193}, {"window": 8192})["fits"] is False
    # chars/4 estimate times calibration
    assert b.predict({"est_tokens": 1000, "calib": 1.25}, {"window": 2000})["projected_tokens"] == 1250
    with pytest.raises(ValueError):
        b.predict({}, {"window": 1})
    with pytest.raises(ValueError):
        b.predict({"projected_tokens": 1}, {})


def test_zach_stub_raises():
    with pytest.raises(NotImplementedError, match="awaiting Zach's files"):
        F.ZachCeilingPredictor().predict({"projected_tokens": 1}, {"window": 2})


@pytest.mark.parametrize("window", [2048, 4096, 8192, 40960])
@pytest.mark.parametrize("margin", [0.0, R.MARGIN])
def test_baseline_equals_old_inline_rules(window, margin):
    for need in range(window - 600, window + 50, 7):
        old = need <= math.floor(window * (1 - margin))          # old route_at_start.fits
        assert R.fits(need, window, margin) is old
        assert F.fits_locally(need, window, margin)["fits"] is old
        if margin == 0.0:
            assert F.fits_locally(need, window)["fits"] is (not need > window)  # old impact_model / dse_v1 rule


def test_route_goes_through_predictor():
    prof = R.load_profiles()
    c = Counting()
    a = R.route(prof, "evo-x2-8192", 1000, 1.0, margin=0, need_source="test", predictor=c)
    assert c.n >= 1 and a["route"] == "local"
    assert a == R.route(prof, "evo-x2-8192", 1000, 1.0, margin=0, need_source="test")
    b = R.route(prof, "evo-x2-8192", 1000, 1.0, margin=0, need_source="test", predictor=AlwaysCloud())
    assert b["route"] == "cloud"


def test_policy_goes_through_predictor():
    import impact_model as im
    k = [(1, "a"), (1, "b")]
    local = {x: {"flag": False, "ok": True, "wall_s": 1.0} for x in k}
    cloud = {x: {"ok": True, "usd": 0.01, "seconds": 2.0} for x in k}
    proj = {k[0]: 100, k[1]: 900}
    c = Counting()
    r = im.route_measured_policy(local, cloud, proj, 500, seeds=[1], predictor=c)
    assert c.n == 2 and r["routed"] == 1
    assert im.route_measured_policy(local, cloud, proj, 500, seeds=[1], predictor=AlwaysCloud())["routed"] == 2


def test_long_candidates_go_through_predictor():
    from src.dse import dse_v1 as D
    k = [(1, "a"), (1, "b")]
    local = {"ollama_default": {x: {"ok": True, "wall_s": 1.0, "flag": False} for x in k}}
    cloud = {"m": {x: {"ok": True, "usd": 0.01} for x in k}}
    proj = {k[0]: 100, k[1]: 900}
    c = Counting()
    _c, notes = D.build_long_candidates_pure(local, cloud, proj, {"ollama_default": 500}, 1000, 50, predictor=c)
    assert c.n == 2 and any("routed 1/2" in n for n in notes)
    _c, notes = D.build_long_candidates_pure(local, cloud, proj, {"ollama_default": 500}, 1000, 50,
                                             predictor=AlwaysCloud())
    assert any("routed 2/2" in n for n in notes)


def test_register_rows_compute_identically_through_interface():
    """hybrid-route-at-start-measured and own-dse-v1, recomputed with an explicit baseline predictor (uncached, every
    fit decision counted), equal the committed register values."""
    import impact_model as im
    from src.dse import dse_v1 as D
    c = Counting()
    d = im.route_at_start_measured(REPO, predictor=c)
    assert d["status"] == "VERIFIED" and c.n > 0
    assert im.route_measured_text(d) == _register_value("hybrid-route-at-start-measured")
    c2 = Counting()
    dv = D.own_dse_v1(REPO, predictor=c2)
    assert c2.n > c.n          # the B3 hybrids plus the long-task route at start
    assert D.own_dse_v1_text(dv) == _register_value("own-dse-v1")
