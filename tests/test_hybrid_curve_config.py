"""Operator decisions 2026-10-22 on hybrid-savings-v1: savings as a formula over task size (hybrid-savings-curve),
the ceiling label, the evo-t2s configuration table first (t2s-config-comparison), the flash-attention attribution in
P2 and cost-cut-config-time, and the cited list price label. No network."""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
import impact_model as im  # noqa: E402
import numbers_register as nr  # noqa: E402

# local {(seed, task): (flag, wrong)}, cloud {(seed, task): (ok, usd)}
LOCAL = {(1, "a"): (False, False), (1, "b"): (True, True), (1, "c"): (False, True), (1, "d"): (True, False),
         (9, "a"): (True, True)}
CLOUD = {(1, "a"): (True, 0.01), (1, "b"): (True, 0.02), (1, "c"): (False, 0.03), (1, "d"): (True, 0.04)}


def test_savings_formula_is_bill_times_share_minus_escalation():
    s = im.savings_formula(0.01, 0.8, 0.05, 1000)
    assert s["bill"] == pytest.approx(0.01 * 1000 * 30)
    assert s["kept"] == pytest.approx(300 * 0.8) and s["escalation"] == pytest.approx(300 * 0.05)
    assert s["savings"] == pytest.approx(300 * 0.8 - 300 * 0.05)


def test_curve_inputs_interim_and_measured_redo():
    ci = im.curve_inputs(LOCAL, CLOUD, seeds=[1])
    # kept at matched quality: a (local ok, cloud ok) and d (local ok, cloud ok); b and c escalated
    assert (ci["n"], ci["kept"]) == (4, 2) and ci["k"] == pytest.approx(0.5)
    c = (0.01 + 0.02 + 0.03 + 0.04) / 4
    assert ci["bill_task"] == pytest.approx(c)
    assert ci["esc_task"] == pytest.approx((0.02 + 0.03) / 4)
    assert ci["x_ratio"] == pytest.approx(((0.02 + 0.03) / 4 - 0.5 * c) / c)
    # with a measured redo mean the escalated tasks cost that mean instead
    assert im.curve_inputs(LOCAL, CLOUD, [1], esc_usd=0.1)["esc_task"] == pytest.approx(2 * 0.1 / 4)
    # the formula at the measured point equals the task-level saving of keeping exactly the kept set local
    s = im.savings_formula(ci["bill_task"], ci["k"], ci["x_ratio"], 1, days=1)
    assert s["savings"] == pytest.approx(c - (0.02 + 0.03) / 4)
    with pytest.raises(ValueError):
        im.curve_inputs(LOCAL, CLOUD, seeds=[7])


def test_pending_without_rows(tmp_path):
    c = im.hybrid_curve(tmp_path)
    assert c["status"] == "PENDING" and "cloud-cost-per-real-task" in c["pending"]
    assert im.hybrid_curve_text(c).startswith("PENDING") and "PENDING" in im.render_curve_block(c)
    t = im.t2s_config_comparison(tmp_path)
    assert t["status"] == "PENDING" and set(t["pending"]) == set(im.T2S_CONFIG_REQUIRED)
    assert "PENDING" in im.render_t2s_config_block(t)


def test_t2s_config_share_is_sessions_without_silent_failure(monkeypatch):
    monkeypatch.setattr(im, "register_has", lambda repo, cid: True)
    r2 = {"cpu": {"m": {"loaded": [4096], "silent": 5, "n": 5}},
          "igpu": {"m": {"loaded": [32768], "silent": 1, "n": 5}}}
    curve = {"status": "VERIFIED", "tiers": [{"model": "cloud-x", "bill_task": 0.01, "x_ratio": 0.3,
                                              "redo_premium": 0.0}]}
    t = im.t2s_config_comparison(REPO, curve, r2)
    cpu, igpu = t["rows"]
    assert cpu["config"].startswith("naive") and cpu["k"] == 0 and igpu["k"] == pytest.approx(0.8)
    # interim escalation cost: X = 0, so savings = B x k (the curve's B3 selection premium is not transferred)
    assert cpu["savings"]["cloud-x"][1000]["savings"] == pytest.approx(0.0)
    assert igpu["savings"]["cloud-x"][1000]["savings"] == pytest.approx(0.01 * 1000 * 30 * 0.8)
    # a measured redo premium scales with the escalated share
    curve["tiers"][0]["redo_premium"] = 0.5
    t2 = im.t2s_config_comparison(REPO, curve, r2)
    assert t2["rows"][1]["savings"]["cloud-x"][1000]["escalation"] == pytest.approx(300 * 0.2 * 0.5)
    txt = im.t2s_config_text(t)
    assert "to be replaced by the evo-t2s B5 results" in txt and "no gated silent failure" in txt
    blk = im.render_t2s_config_block(t)
    assert "naive: Intel CPU" in blk and "recommended: Intel iGPU" in blk


def test_fa_attribution_text():
    assert nr.fa_attribution({"a": nr.FA_KV_READING, "b": nr.FA_KV_READING}) == (
        "the flash-attention control attributes the speedup to KV precision, not flash attention (b5-fa-control)")
    assert nr.fa_attribution({"a": nr.FA_KV_READING, "b": "both contribute"}) is None
    assert nr.fa_attribution({}) is None


def test_p2_caveat_uses_attribution_and_list_price_label():
    prof = {"assumptions": {"model": "m", "team_size": 1, "agent_sessions_per_dev_per_day": 10},
            "hour_value": {"usd_per_hour": 100.0}}
    cc = {"models": {"m": {"default_min": 200.0, "best_min": 100.0, "saved_min": 100.0, "best": "g",
                           "best_name": "kv_q8_0"}}, "fa_included": True,
          "fa_attribution": "the flash-attention control attributes the speedup to KV precision, not flash attention "
                            "(b5-fa-control)"}
    r = im.p2_v2(prof, cc, {"usd": 2000.0, "label": "list price, Src, 2025-04-15"}, fa_verified=True)
    assert r["caveat"] == cc["fa_attribution"]
    txt = im.p2_v2_value(r)
    assert "list price, Src, 2025-04-15" in txt and "ASSUMPTION, must be verified" not in txt
    assert "half the hardware" in txt   # 200 vs 100 minutes: exactly half the machines
    assert im.price_label({"usd": 1.0}) == "ASSUMPTION, must be verified"


def test_price_file_carries_label_and_source():
    m = json.loads((REPO / im.REF_MACHINE_FILE).read_text(encoding="utf-8"))
    assert m["label"].startswith("list price, Liliputing, ") and m["usd"] == 2000.0
    assert m["status"].startswith("cited public list price") and m["vendor_check_20261022"]["used"] is False


def test_impact_doc_order_and_labels():
    text = (REPO / "docs" / "IMPACT.md").read_text(encoding="utf-8")
    sec = text[text.index("## Hybrid savings vs all-cloud"):text.index(im.HYB_END)]
    assert sec.index(im.TB_BEGIN) < sec.index(im.CFG_BEGIN) < sec.index(im.CURVE_BEGIN) < sec.index(im.HYB_BEGIN)
    assert "ceiling, no such detector" in sec and "general verifier" in sec and "perfect detector" not in sec
    assert "UPPER BOUND, not" not in sec
    assert "ASSUMPTION, must be verified" not in text
    assert "KV precision, not flash attention (b5-fa-control)" in text and "half the hardware" in text
    assert "list price, Liliputing, 2025-04-15" in text
    assert "—" not in sec


def test_real_curve_if_available():
    if not all(im.register_has(REPO, c) for c in im.CURVE_REQUIRED):
        pytest.skip("register rows not VERIFIED in this checkout")
    c = im.hybrid_curve(REPO)
    assert c["status"] == "VERIFIED"
    for t in c["tiers"]:
        labels = [p["label"] for p in t["points"]]
        assert labels.count("measured") == 1 and labels.count(im.CURVE_PROJ_LABEL) == 3
        assert [p["size"] for p in t["points"]] == sorted(p["size"] for p in t["points"])
        assert all(set(p["volumes"]) == set(im.HYBRID_VOLUMES) for p in t["points"])
        if c and t["esc_label"] == im.INTERIM_ESC:
            assert t["redo_premium"] == 0.0
    txt = im.hybrid_curve_text(c)
    assert "ceiling, no such detector" in txt and "—" not in txt


# Operator decision 2026-10-22, savings curve lead: the measured trade-off (row hybrid-savings-tradeoff) first.
def _pol(correct, n, savings_task, escalated=0, silent=0, delta=0.0):
    return {"correct": correct, "n": n, "quality": correct / n, "delta_points": delta, "escalated": escalated,
            "silent_wrong": silent, "savings_task": savings_task}


def _fake_h_curve():
    pols = {"all_cloud": _pol(9, 10, 0.0, escalated=10), "naive_local": _pol(8, 10, 0.01, silent=2),
            "hybrid_flag": _pol(9, 10, 0.004, escalated=6), "hybrid_perfect": _pol(10, 10, 0.008, escalated=2)}
    cells = [{"device": "evo-x2", "setting": "d", "tier": im.DEFAULT_TIER, "label": "measured", "model": "cloud-x",
              "ref_usd": 0.01, "policies": pols, "esc_label": im.INTERIM_ESC},
             {"device": "evo-x2", "setting": "b", "tier": "other", "label": "measured", "model": "cloud-x",
              "ref_usd": 0.01, "policies": pols, "esc_label": im.INTERIM_ESC},
             {"device": "evo-t2s Intel CPU", "setting": "4096", "tier": "ollama_ctx_4096", "label": "projected",
              "model": "cloud-x", "ref_usd": 0.01, "policies": pols, "esc_label": im.INTERIM_ESC}]
    h = {"status": "VERIFIED", "cells": cells, "seeds": [1, 2, 3]}
    curve = {"status": "VERIFIED", "case": "realistic", "hit_rate": 0.7, "tiers": [{"model": "cloud-x", "points": [
        {"name": "B3", "label": "measured", "size": 10, "bill_task": 0.01},
        {"name": "big median", "label": im.CURVE_PROJ_LABEL, "size": 1000, "bill_task": 1.0}]}]}
    return h, curve


def test_tradeoff_options_savings_and_accuracy():
    h, curve = _fake_h_curve()
    t = im.hybrid_tradeoff(h, curve)
    assert t["status"] == "VERIFIED"
    # evo-x2 non-default tier dropped; 2 devices x 2 points
    assert len(t["rows"]) == 4 and {r["setting"] for r in t["rows"]} == {"d", "4096"}
    m = next(r for r in t["rows"] if r["measured"])
    o = m["options"]
    assert list(o) == ["all_cloud", "naive_local", "hybrid_flag"] and "hybrid_perfect" not in o
    assert o["all_cloud"]["savings_month"][1000] == 0
    # naive local saves the full bill at the measured point
    assert o["naive_local"]["savings_month"][1000] == pytest.approx(0.01 * 1000 * 30)
    assert o["naive_local"]["silent_wrong"] == 2 and o["naive_local"]["correct"] == 8
    assert o["hybrid_flag"]["savings_month"][100] == pytest.approx(0.004 * 100 * 30)
    assert m["ceiling"]["savings_month"][1000] == pytest.approx(0.008 * 1000 * 30)
    # a projected point scales by its bill, keeps the B3 accuracy, labelled
    p = next(r for r in t["rows"] if r["device"] == "evo-x2" and r["point"] == "big median")
    assert p["options"]["naive_local"]["savings_month"][1000] == pytest.approx(1.0 * 1000 * 30)
    assert p["options"]["hybrid_flag"]["savings_month"][1000] == pytest.approx(0.4 * 1000 * 30)
    assert p["options"]["naive_local"]["correct"] == 8 and p["acc_label"] == im.TRADEOFF_ACC_CARRIED
    q = next(r for r in t["rows"] if r["device"] != "evo-x2" and r["point"] == "big median")
    assert im.TRADEOFF_DEVICE_PROJ in q["acc_label"]
    txt = im.hybrid_tradeoff_text(t)
    assert "ceiling, no such detector" in txt and im.TRADEOFF_SILENT in txt and im.CURVE_PROJ_LABEL in txt
    assert "—" not in txt
    blk = im.render_tradeoff_block(t)
    assert blk.startswith(im.TB_BEGIN) and blk.endswith(im.TB_END) and "Ceiling, no such detector" in blk
    o29 = im.render_oct29_block(t)
    assert "Oct 29" in o29 and "Demo story step 4" in o29 and "ceiling, no such detector" in o29
    assert "big median" in o29 and "—" not in o29


def test_tradeoff_pending():
    t = im.hybrid_tradeoff({"status": "PENDING", "pending": ["x"]}, {"status": "VERIFIED"})
    assert t["status"] == "PENDING" and im.hybrid_tradeoff_text(t).startswith("PENDING")
    assert "PENDING" in im.render_tradeoff_block(t) and "PENDING" in im.render_oct29_block(t)


def test_t2s_share_relabelled():
    assert im.T2S_KEPT_LABEL == "measured by R2 session pass rate (not B3 task accuracy)"
    text = (REPO / "docs" / "IMPACT.md").read_text(encoding="utf-8")
    cfg = text[text.index(im.CFG_BEGIN):text.index(im.CFG_END)]
    assert im.T2S_KEPT_LABEL in cfg and "Kept-local share" not in cfg


def test_impact_doc_tradeoff_first():
    text = (REPO / "docs" / "IMPACT.md").read_text(encoding="utf-8")
    sec = text[text.index("## Hybrid savings vs all-cloud"):text.index(im.HYB_END)]
    assert (sec.index(im.TB_BEGIN) < sec.index(im.OCT29_BEGIN) < sec.index(im.CFG_BEGIN) < sec.index(im.CURVE_BEGIN)
            < sec.index(im.HYB_BEGIN))
