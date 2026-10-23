"""Tests for src/dse/own_profiler.py (row own-profiler-loo): pure probe/predictor logic on synthetic data, plus a
smoke run on the committed files when they are present."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from src.dse import own_profiler as op

REPO = Path(__file__).resolve().parents[1]


def _t(idx, wall, ok=True, prompt=1000, loaded=4096):
    return {"idx": idx, "wall": wall, "ok": ok, "prompt": prompt, "loaded": loaded}


def test_probe_prefix_stops_at_budget():
    turns = [_t(0, 300), _t(1, 250), _t(2, 100)]
    assert [t["idx"] for t in op.probe_prefix(turns, 600)] == [0, 1]
    assert op.probe_prefix([_t(0, 700)], 600) == []


def test_projected_final_prompt_linear_and_single():
    probe = [_t(0, 1, prompt=1000), _t(1, 1, prompt=2000)]
    assert op.projected_final_prompt(probe, 4) == pytest.approx(4000)
    assert op.projected_final_prompt([_t(0, 1, prompt=1500)], 40) == 1500


def test_session_features_uses_held_out_seeds_only():
    cfg = {"sessions": {1: [_t(0, 100, ok=False), _t(1, 100)], 2: [_t(0, 10), _t(1, 30)]}}
    f = op.session_features(cfg, 600)
    assert f["probe_seed"] == 1 and f["probe_s"] == 100 and f["probe_acc"] == 0.5
    assert f["actual_s"] == 20 and f["actual_acc"] == 1.0


def test_no_probe_is_a_miss():
    cfg = {"sessions": {1: [_t(0, 900)], 2: [_t(0, 900)]}}
    f = {"device": "a", "family": "x", "model": "m", "arm": "r", **op.session_features(cfg, 600)}
    res = op.evaluate_loo([f, {**f, "device": "b"}])
    assert all(r["err_s"] == math.inf for r in res)


def test_loo_ratios_come_from_other_device():
    a = {"device": "a", "family": "f", "model": "m", "arm": "1", "stratum": "fits", "probe_s": 10, "probe_acc": 1.0,
         "actual_s": 20, "actual_acc": 0.9}
    b = {**a, "device": "b", "arm": "2", "probe_s": 5, "actual_s": 15}
    res = {r["device"]: r for r in op.evaluate_loo([a, b])}
    assert res["a"]["pred_s"] == pytest.approx(10 * 3)  # b's ratio 15/5
    assert res["b"]["pred_s"] == pytest.approx(5 * 2)   # a's ratio 20/10
    assert res["a"]["pred_acc"] == pytest.approx(0.9)


def test_fit_ratios_falls_back_to_pooled():
    train = [{"stratum": "overflow", "probe_s": 1, "actual_s": 2, "probe_acc": 1, "actual_acc": 1}]
    r = op.fit_ratios(train)
    ps, _ = op.predict({"probe_s": 3, "probe_acc": 1, "stratum": "fits"}, r)
    assert ps == pytest.approx(6)


def test_ape():
    assert op.ape(11, 10) == pytest.approx(10)
    assert op.ape(None, 10) == math.inf
    assert op.ape(1, None) is None


@pytest.mark.skipif(not all((REPO / f).exists() for f in op.SESSION_DATA_FILES), reason="session files not synced")
def test_committed_loo_runs():
    d = op.own_profiler_loo(REPO)
    assert d["n_by_device"].get("evo-t2s", 0) >= 1 and d["n_by_device"].get("evo-x2", 0) >= 1
    assert "pre-registered target" in op.own_profiler_text(d)
    assert d["blade_turns"] == 0
