"""Operator decisions 2026-10-22: late escalation vs a fresh all-cloud run (row late-escalation-cost-vs-fresh), the
route-at-session-start policy (row hybrid-route-at-start) and the Oct 29 / demo step 4 lead. No network."""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
import impact_model as im  # noqa: E402


def _redo(seed, task, usd, ok, tier="t"):
    return {"seed": seed, "task_id": task, "usd": usd, "cloud_end_state_ok": ok, "tier": tier}


def test_late_vs_fresh_matched_on_seed_and_task():
    redo = [_redo(1, "a", 0.04, True), _redo(1, "a", 0.02, False, "u"), _redo(9, "b", 0.06, True)]
    fresh = {(1, "a"): (True, 0.01), (1, "b"): (True, 0.03)}
    s = im.late_vs_fresh_stats(redo, fresh, n_wrong=4)
    assert s["redo"]["mean"] == pytest.approx(0.04) and s["fresh"]["mean"] == pytest.approx(0.02)
    assert s["ratio_mean"] == pytest.approx(2.0)
    # seed 9 has no fresh run: only the two (1, "a") rows match, one distinct task
    assert s["m_redo"]["n"] == 2 and s["n_matched_keys"] == 1 and s["matched_seeds"] == [1]
    assert s["m_redo"]["mean"] == pytest.approx(0.03) and s["m_fresh"]["mean"] == pytest.approx(0.01)
    assert s["m_ratio_mean"] == pytest.approx(3.0)
    # tasks not run count as not fixed
    assert s["fixed"] == 2 and s["fix_rate"] == pytest.approx(0.5)
    assert s["ci"][0] < 0.5 < s["ci"][1]


def test_late_text_names_inherited_actions_and_t2s_cost():
    d = {"status": "VERIFIED", "model": "cloud-x", "redo": {"mean": 0.02, "median": 0.015, "n": 10},
         "fresh": {"mean": 0.01, "median": 0.01, "n": 9}, "ratio_mean": 2.0, "ratio_median": 1.5,
         "m_redo": {"mean": 0.02, "median": 0.02, "n": 6}, "m_fresh": {"mean": 0.01, "median": 0.01, "n": 6},
         "m_ratio_mean": 2.0, "m_ratio_median": 2.0, "n_matched_keys": 4, "matched_seeds": [1], "fixed": 7,
         "n_wrong": 10, "fix_rate": 0.7, "ci": (0.4, 0.9), "unfixed": 3,
         "classes": {"earlier_state_repair": 2, "cloud_model_error": 1, "grader_issue": 0},
         "t2s": {"device": "evo-t2s Intel CPU", "setting": "4096", "label": "projected",
                 "flag": {"correct": 9, "n": 9, "quality": 1.0, "escalated": 8,
                          "savings_month": {100: -1.0, 1000: -10.0, 10000: -100.0}},
                 "all_cloud": {"correct": 9, "n": 9, "quality": 1.0}},
         "volumes": (100, 1000, 10000)}
    txt = im.late_text(d)
    assert "2.00x on the mean" in txt and "fixed 7/10 (70.0%)" in txt and "40.0% to 90.0%" in txt
    assert "2 inherited the local model's earlier executed actions (earlier_state_repair)" in txt
    assert "cloud_model_error 1" in txt and "grader_issue" not in txt
    assert "costs more than all-cloud" in txt and "—" not in txt
    lead = im.step4_lead(d, None, "cloud-x", 1000)
    assert lead[0].startswith('"Fix after the fact: about 2 cents, 70.0% fixed, but executed actions cannot be undone; '
                              'decide up front instead."')
    assert "95% CI 40.0% to 90.0%" in lead[0] and "2 inherited actions" in lead[0]
    assert im.late_text({"status": "PENDING", "pending": ["x"]}).startswith("PENDING")
    assert im.step4_lead({"status": "PENDING"}, None, "m", 1000) == []


def test_route_decision_rule():
    assert im.route_decision(0.9, 0.9, 0) == "local"
    assert im.route_decision(0.889, 0.9, 0) == "cloud"
    assert im.route_decision(0.85, 0.9, 5) == "local"
    assert im.route_decision(0.84, 0.9, 5) == "cloud"


def test_ceiling_measured_uses_redo_outcome():
    local = {(1, "a"): (False, False), (1, "b"): (True, True), (1, "c"): (False, True), (2, "d"): (False, True)}
    cloud = {(1, "a"): (True, 0.1), (1, "b"): (True, 0.1), (1, "c"): (True, 0.1)}
    c = im.ceiling_measured(local, cloud, [1], {(1, "b"): True, (1, "c"): False})
    assert (c["correct"], c["n"], c["escalated"], c["no_row"]) == (2, 3, 2, 0)
    c = im.ceiling_measured(local, cloud, [1], {(1, "b"): True})
    assert c["no_row"] == 1 and c["correct"] == 2


def _opt(correct, n, sav, escalated=0, silent=0):
    return {"correct": correct, "n": n, "quality": correct / n, "escalated": escalated, "silent_wrong": silent,
            "savings_month": {v: sav * v for v in (100, 1000, 10000)}}


def _fake():
    def row(device, setting, point_label="measured", point="B3", bill=0.01, nv=(8, 10)):
        return {"device": device, "setting": setting, "device_label": "measured", "model": "cloud-x",
                "point": point, "point_label": point_label, "bill_task": bill, "esc_label": "e",
                "options": {"all_cloud": _opt(9, 10, 0.0, escalated=10), "naive_local": _opt(*nv, 0.3, silent=2),
                            "hybrid_flag": _opt(9, 10, -0.1, escalated=9)},
                "ceiling": _opt(10, 10, 0.2, escalated=2)}
    tb = {"status": "VERIFIED", "volumes": (100, 1000, 10000), "esc_label": "e",
          "rows": [row("evo-x2", "ollama_default, default = best"),
                   row("evo-x2", "ollama_default, default = best", im.CURVE_PROJ_LABEL, "big median", 1.0),
                   row("evo-t2s Intel CPU", "default, loaded window 4096", nv=(5, 10))]}
    cfg = {"status": "VERIFIED", "rows": [
        {"backend": "cpu", "model": im.B3_V2_MODEL, "loaded": [4096], "silent": 5, "n": 5, "k": 0.0},
        {"backend": "igpu", "model": im.B3_V2_MODEL, "loaded": [32768], "silent": 1, "n": 5, "k": 0.8},
        {"backend": "igpu", "model": "other", "loaded": [32768], "silent": 0, "n": 5, "k": 1.0}]}
    return tb, cfg


def test_route_at_start_per_device_and_tolerance():
    tb, cfg = _fake()
    ce = {("evo-x2", "cloud-x"): {"correct": 9, "n": 10, "quality": 0.9, "escalated": 2, "no_row": 0}}
    t = im.route_at_start(tb, cfg, ce)
    assert t["status"] == "VERIFIED" and len(t["rows"]) == 2
    x2 = next(r for r in t["rows"] if r["device"] == "evo-x2")
    t2 = next(r for r in t["rows"] if r["device"] == "evo-t2s")
    # evo-x2 local 8/10 vs all-cloud 9/10: gap 10 points, cloud at 0 and 5, local at 10
    assert x2["gap"] == pytest.approx(10.0)
    assert [x2["routes"][k]["route"] for k in (0, 5, 10)] == ["cloud", "cloud", "local"]
    assert x2["routes"][0]["savings_month"][1000] == 0.0
    assert x2["routes"][10]["savings_month"][1000] == pytest.approx(0.01 * 1000 * 30)
    assert x2["ceiling"]["correct"] == 9 and "measured redo" in x2["ceiling"]["label"]
    assert x2["projected"][0]["label"] == im.CURVE_PROJ_LABEL
    # evo-t2s: recommended Intel iGPU qwen3:14b share, labelled; ceiling falls back to the trade-off field
    assert t2["local"]["quality"] == pytest.approx(0.8) and t2["local"]["label"] == im.T2S_KEPT_LABEL
    assert "32768" in t2["recommended"] and t2["ceiling"]["correct"] == 10
    assert im.TRADEOFF_DEVICE_PROJ in t2["ceiling"]["label"] and im.TRADEOFF_DEVICE_PROJ in t2["naive"]
    txt = im.route_text(t)
    assert "ceiling, no such detector" in txt and im.T2S_KEPT_LABEL in txt and "—" not in txt
    blk = im.render_route_block(t)
    assert blk.startswith(im.ROUTE_BEGIN) and blk.endswith(im.ROUTE_END) and "tolerance 10" in blk
    assert im.route_at_start({"status": "PENDING", "pending": ["x"]}, cfg)["status"] == "PENDING"


def test_impact_doc_route_first_and_step4_lead():
    text = (REPO / "docs" / "IMPACT.md").read_text(encoding="utf-8")
    sec = text[text.index("## Hybrid savings vs all-cloud"):text.index(im.HYB_END)]
    assert sec.index("### Why the product decides at session start") < sec.index(im.ROUTE_BEGIN) < sec.index(
        im.TB_BEGIN) < sec.index(im.OCT29_BEGIN)
    o29 = text[text.index(im.OCT29_BEGIN):text.index(im.OCT29_END)]
    step4 = o29[o29.index("Demo story step 4"):]
    if "late-escalation-cost-vs-fresh" in o29:
        assert step4.split("\n")[2].startswith('"Fix after the fact: about ')
        assert "executed actions cannot be undone; decide up front instead" in step4 and "95% CI" in step4
    assert "—" not in sec


def test_detection_row_uses_full_history_redo_only():
    import numbers_register as nr
    c = next(x for x in nr.NUMBER_ENTRIES if x["claim_id"] == "impact-v3-detection-value-measured")
    assert "results/cloud_redo_v1.jsonl" not in c["data_files"]
    assert nr.CLOUD_REDO_FULL_CONTEXT in c["data_files"]
    d = {"tiers": {"t": {"tp": 1, "fp": 1, "fn": 0, "tn": 2, "n": 4, "precision": 0.5, "recall": 1.0, "fn_share": 0.0,
                         "redo_usd": 0.02, "flagged": 2, "every_usd": 0.08, "flagged_usd": 0.04, "saved_usd": 0.04}},
         "redo_tag": "measured", "redo_source": "cloud redo USD per task: tier mean of the full-history redo"}
    txt = im.measured_detection_text(d)
    assert "full-history redo" in txt and "cloud-redo-cost-per-wrong-task (" not in txt
