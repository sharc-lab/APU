"""Tests for src/dse/dse_v1.py (row own-dse-v1): cost model, ranking, Pareto and validation on synthetic candidates,
plus a smoke run on the committed files."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.dse import dse_v1 as D

REPO = Path(__file__).resolve().parents[1]
KWH = 0.1453


def test_cost_month_parts():
    est = D.Estimate(quality=0.9, local_share=1.0, local_s=86.4, cloud_usd_task=0.0)
    c = D.cost_month(est, 1000, hw_usd=3600, watts=100, usd_kwh=0.1)
    assert c["machines"] == 1                      # 86,400 s/day fits one machine
    assert c["hardware"] == pytest.approx(100.0)   # 3600 / 36
    assert c["electricity"] == pytest.approx(86400 * 30 / 3600 * 0.1 * 0.1)
    assert c["cloud"] == 0
    c2 = D.cost_month(est, 1001, hw_usd=3600, watts=100, usd_kwh=0.1)
    assert c2["machines"] == 2


def test_cloud_only_needs_no_hardware():
    est = D.Estimate(1.0, 0.0, 0.0, 0.01)
    c = D.cost_month(est, 100, hw_usd=None, watts=None, usd_kwh=KWH)
    assert c["machines"] == 0 and c["total"] == pytest.approx(30.0)


def test_local_without_price_raises():
    with pytest.raises(ValueError):
        D.cost_month(D.Estimate(1.0, 0.5, 10, 0), 10, hw_usd=None, watts=None, usd_kwh=KWH)


def _cand(name, pred, meas, hw=None):
    return D.Candidate(hw or "none", name, "w", "kv", "b", "p", pred=pred, meas=meas,
                       hw_usd=2000 if hw else None, watts=90 if hw else None)


def test_validation_scores_pick_on_measured_values():
    # predicted: local looks cheap and accurate; measured: local misses the accuracy target
    local = _cand("local", D.Estimate(0.95, 1, 10, 0), D.Estimate(0.80, 1, 10, 0), hw="dev")
    cloud = _cand("cloud", D.Estimate(0.97, 0, 0, 0.02), D.Estimate(0.97, 0, 0, 0.02))
    v = D.validate_cell([local, cloud], 10000, 0.9, 1e9, KWH)
    assert v["pick"] == local.label and v["best"] == cloud.label
    assert v["pick_meas_feasible"] is False and v["met"] is False


def test_validation_met_when_pick_is_best():
    a = _cand("a", D.Estimate(0.95, 0, 0, 0.001), D.Estimate(0.95, 0, 0, 0.001))
    b = _cand("b", D.Estimate(0.99, 0, 0, 0.01), D.Estimate(0.99, 0, 0, 0.01))
    v = D.validate_cell([a, b], 100, 0.9, 1e9, KWH)
    assert v["met"] and v["gap"] == 0


def test_budget_excludes_expensive_cloud():
    a = _cand("a", D.Estimate(0.99, 0, 0, 0.01), D.Estimate(0.99, 0, 0, 0.01))
    assert D.rank([a], 10000, 0.9, budget=500, usd_kwh=KWH) == []


def test_pareto_drops_dominated():
    a = _cand("a", D.Estimate(0.95, 0, 0, 0.001), None)
    b = _cand("b", D.Estimate(0.90, 0, 0, 0.002), None)
    labels = [c.label for c, _e in D.pareto([a, b], 100, KWH)]
    assert labels == [a.label]


@pytest.mark.skipif(not (REPO / "results" / "x2_evalset_tasks_real_v2.jsonl").exists(), reason="B3 v2 not synced")
def test_committed_dse_runs():
    d = D.own_dse_v1(REPO)
    assert ("short", 100) in d["cells"] and d["n_candidates"] >= 3
    assert "pre-registered target" in D.own_dse_v1_text(d)
