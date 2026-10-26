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


# ── long-task and mixed candidates (fake rows) ────────────────────────────────────────────────────────────────

def _long_fixture():
    # 4 task runs on one arm (window 1000): t1 correct, t2 wrong, t3 projected over the window (routed), t4 flagged
    local = {"ollama_default": {(1, "t1"): {"ok": True, "wall_s": 10.0, "flag": False},
                                (1, "t2"): {"ok": False, "wall_s": 20.0, "flag": False},
                                (1, "t3"): {"ok": False, "wall_s": 30.0, "flag": False},
                                (1, "t4"): {"ok": False, "wall_s": 40.0, "flag": True}}}
    cloud = {"haiku": {(1, "t3"): {"ok": True, "usd": 0.04}, (1, "t4"): {"ok": True, "usd": 0.02}}}
    proj = {(1, "t1"): 500, (1, "t2"): 900, (1, "t3"): 1500, (1, "t4"): 800}
    return local, cloud, proj, {"ollama_default": 1000}


def test_long_builder_local_cloud_and_hybrid():
    local, cloud, proj, win = _long_fixture()
    cands, notes = D.build_long_candidates_pure(local, cloud, proj, win, hw_usd=2000, watts=90)
    by = {c.policy: c for c in cands}
    cl, lo, hy = by["all-cloud"], by["local only"], by["per-task route at start + window flag"]
    assert cl.meas.quality == 1.0 and cl.meas.cloud_usd_task == pytest.approx(0.03) and cl.pred == cl.meas
    assert lo.meas.quality == 0.25 and lo.meas.local_s == pytest.approx(25.0) and lo.pred == lo.meas  # no probe
    # hybrid: t1 local correct, t2 local wrong, t3 routed (cloud correct), t4 flagged then escalated (cloud correct)
    assert hy.meas.quality == 0.75
    assert hy.meas.local_share == 0.75 and hy.meas.local_s == pytest.approx((10 + 20 + 40) / 3)
    assert hy.meas.cloud_usd_task == pytest.approx(0.06 / 4)
    assert hy.pred.local_share == 0.75                                    # 1 - routed/n from the projection
    assert hy.pred.quality == pytest.approx(0.75 * 0.25 + 0.25 * 1.0)
    assert hy.pred.cloud_usd_task == pytest.approx(0.25 * 0.03)
    assert any("routed 1/4, flagged 1/4" in n for n in notes)


def test_long_hybrid_not_built_without_cloud_row():
    local, cloud, proj, win = _long_fixture()
    del cloud["haiku"][(1, "t4")]
    cands, notes = D.build_long_candidates_pure(local, cloud, proj, win, hw_usd=2000, watts=90)
    assert not any(c.policy.startswith("per-task") for c in cands)
    assert any("hybrid not built" in n for n in notes)


def test_long_hybrid_that_routes_nothing_equals_local():
    local, cloud, proj, win = _long_fixture()
    proj = {k: 10 for k in proj}
    for v in local["ollama_default"].values():
        v["flag"] = False
    cands, _n = D.build_long_candidates_pure(local, cloud, proj, win, hw_usd=2000, watts=90)
    by = {c.policy: c for c in cands}
    assert by["per-task route at start + window flag"].meas == by["local only"].meas


def test_mix_is_half_short_half_long_by_count():
    s = _cand("s", D.Estimate(0.9, 1.0, 10.0, 0.0), D.Estimate(0.8, 1.0, 12.0, 0.0), hw="dev")
    lg = _cand("l", D.Estimate(1.0, 0.0, 0.0, 0.02), D.Estimate(1.0, 0.0, 0.0, 0.02))
    (m,) = D.mix_candidates([s], [lg])
    assert m.pred.quality == pytest.approx(0.95) and m.pred.local_share == 0.5 and m.pred.local_s == 10.0
    assert m.pred.cloud_usd_task == pytest.approx(0.01)
    assert m.meas.quality == pytest.approx(0.9) and m.meas.local_s == 12.0
    assert m.hardware == "dev" and m.hw_usd == 2000 and m.label == f"short [{s.label}] + long [{lg.label}]"
    both = D.mix_candidates([s], [s])[0]
    assert both.meas.local_share == 1.0 and both.meas.local_s == 12.0
    assert D.mix_candidates([lg], [lg])[0].hardware == "none"


def test_zero_eligible_cell_is_computed_and_trivial():
    a = _cand("a", D.Estimate(0.5, 0, 0, 0.001), D.Estimate(0.5, 0, 0, 0.001))
    v = D.validate_cell([a], 100, 0.9, 1e9, KWH)
    assert v["status"] == "computed" and v["trivial"] and v["trivial_reason"] == "0 eligible"
    assert (v["n_candidates"], v["n_eligible"], v["n_eligible_local"]) == (1, 0, 0) and v["met"] is False
    d = {"cells": {("long", 100): v}, "inputs": {"seeds": [1], "hw_usd": 2000, "hw_label": "x", "watts": 90,
                                                 "usd_kwh": KWH},
         "n_candidates": 1, "accuracy_target": 0.9, "budget": 500}
    t = D.own_dse_v1_text(d)
    assert "long 100/day: TRIVIAL (0 eligible; not counted as validation)" in t and "no pick, no gap" in t
    assert "0 PENDING" in t and "target not tested" in t


def test_non_trivial_text_names_lowest_floor_candidate():
    cloud = _cand("cloud", D.Estimate(0.97, 0, 0, 0.002), D.Estimate(0.97, 0, 0, 0.002))
    cloud2 = _cand("cloud2", D.Estimate(0.99, 0, 0, 0.003), D.Estimate(0.99, 0, 0, 0.003))
    local = _cand("local", D.Estimate(0.5, 1, 1, 0), D.Estimate(0.4, 1, 1, 0), hw="dev")
    v = D.validate_cell([cloud, cloud2, local], 1000, 0.9, 1e9, KWH)
    assert v["trivial"] is False and v["lowest_floor"]["label"] == local.label
    t = D._cell_text("mixed", 1000, v)
    assert "lowest-floor local candidate" in t and "predicted accuracy 0.500, measured 0.400" in t


@pytest.mark.skipif(not (REPO / "results" / "x2_longtask_local_v1.jsonl").exists(), reason="long task not synced")
def test_committed_long_and_mixed_cells_compute():
    d = D.own_dse_v1(REPO)
    if d["long_inputs"] is None:
        pytest.skip("long-task rows not VERIFIED in this checkout")
    assert all(d["cells"][(w, v)]["status"] == "computed" for w in ("long", "mixed") for v in D.VOLUMES)
    assert len(d["mixed_candidates"]) == len(d["candidates"]) * len(d["long_candidates"])
    assert chr(0x2014) not in D.own_dse_v1_text(d)
