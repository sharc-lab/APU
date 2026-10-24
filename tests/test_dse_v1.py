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
    t = D.own_dse_v1_text(d)
    assert "pre-registered target" in t or "target not tested" in t
    assert all("trivial" in c for c in d["cells"].values() if c.get("status") == "computed")


# ── trivial cells (pre-registered 2026-10-23, docs/EXPERIMENTS.md "own-dse-v1 trivial cells") ─────────────

def test_one_eligible_is_trivial():
    a = _cand("a", D.Estimate(0.95, 0, 0, 0.001), D.Estimate(0.95, 0, 0, 0.001))
    b = _cand("b", D.Estimate(0.80, 0, 0, 0.0001), D.Estimate(0.80, 0, 0, 0.0001))
    t = D.trivial_cell([a, b], 100, 0.9, 1e9, KWH)
    assert t["trivial"] and t["reason"] == "1 eligible"
    v = D.validate_cell([a, b], 100, 0.9, 1e9, KWH)
    assert (v["n_candidates"], v["n_eligible"], v["n_eligible_local"]) == (2, 1, 0) and v["trivial"]


def test_cloud_best_far_below_every_local_floor_is_trivial():
    cloud = _cand("cloud", D.Estimate(0.97, 0, 0, 0.0001), D.Estimate(0.97, 0, 0, 0.0001))   # USD 0.30/month
    cloud2 = _cand("cloud2", D.Estimate(0.99, 0, 0, 0.001), D.Estimate(0.99, 0, 0, 0.001))
    local = _cand("local", D.Estimate(0.95, 1, 1, 0), D.Estimate(0.95, 1, 1, 0), hw="dev")    # floor 55.56
    assert D.cost_floor(local, 100) == pytest.approx(2000 / 36)
    assert D.trivial_cell([cloud, cloud2, local], 100, 0.9, 1e9, KWH)["trivial"]


def test_local_floor_within_tolerance_is_non_trivial():
    cloud = _cand("cloud", D.Estimate(0.97, 0, 0, 0.002), D.Estimate(0.97, 0, 0, 0.002))      # USD 60/month at 1000
    local = _cand("local", D.Estimate(0.95, 1, 1, 0), D.Estimate(0.95, 1, 1, 0), hw="dev")    # floor 55.56 < 66
    assert not D.trivial_cell([cloud, local], 1000, 0.9, 1e9, KWH)["trivial"]
    v = D.validate_cell([cloud, local], 1000, 0.9, 1e9, KWH)
    assert v["trivial"] is False and v["n_eligible_local"] == 1


def test_best_with_probe_input_is_non_trivial():
    local = _cand("local", D.Estimate(0.95, 1, 1, 0), D.Estimate(0.95, 1, 1, 0), hw="dev")
    cloud = _cand("cloud", D.Estimate(0.97, 0, 0, 0.01), D.Estimate(0.97, 0, 0, 0.01))       # USD 3,000 at 10,000
    assert not D.trivial_cell([local, cloud], 10000, 0.9, 1e9, KWH)["trivial"]


def test_hybrid_with_zero_local_share_does_not_count_as_local():
    cloud = _cand("cloud", D.Estimate(0.97, 0, 0, 0.0007), D.Estimate(0.97, 0, 0, 0.0007))
    cloud2 = _cand("cloud2", D.Estimate(0.99, 0, 0, 0.01), D.Estimate(0.99, 0, 0, 0.01))
    hyb = _cand("hyb", D.Estimate(0.97, 0, 70, 0.0007), D.Estimate(0.97, 0, 0, 0.0007), hw="dev")  # routes all
    assert D.trivial_cell([cloud, cloud2, hyb], 100, 0.9, 1e9, KWH)["trivial"]
    assert not D.trivial_cell([cloud, cloud2, hyb], 100, 0.9, 1e9, KWH, local_by="label")["trivial"]


def test_text_with_no_non_trivial_cell():
    a = _cand("a", D.Estimate(0.95, 0, 0, 0.001), D.Estimate(0.95, 0, 0, 0.001))
    cells = {("short", 100): D.validate_cell([a], 100, 0.9, 1e9, KWH),
             ("long", 100): {"status": "PENDING: x", "met": False}}
    d = {"cells": cells, "inputs": {"seeds": [1], "hw_usd": 2000, "hw_label": "x", "watts": 90, "usd_kwh": KWH},
         "n_candidates": 1, "accuracy_target": 0.9, "budget": 500}
    t = D.own_dse_v1_text(d)
    assert "validation: 0 non-trivial cells, target not tested" in t and "TRIVIAL (1 eligible" in t
    assert chr(0x2014) not in t
