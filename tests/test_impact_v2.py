"""analysis/impact_model.py impact v2 (operator override 2026-10-21): P1 = excess wrong-task rate x (cloud redo cost,
a named PENDING input, + wrong-turn wall time); PENDING inputs never become numbers; ranking puts them last."""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
import impact_model as im  # noqa: E402


def _profile(**over):
    a = {"team_size": 2, "agent_sessions_per_dev_per_day": 5, "turns_per_session": 10,
         "single_shot_calls_per_dev_per_day": 3, "workdays_per_month": 20, "co_runner_share": 0.5,
         "cloud_session_share": 0.5, "device": "evo-x2", "model": "qwen3:14b", "runtime": "ollama_ctx_4096",
         "scenario_runtime": "ollama_ctx_4096", "cloud_model": "claude-sonnet-5-5", "cloud_tier": "standard"}
    a.update(over)
    return {"hour_value": {"usd_per_hour": 100.0}, "assumptions": a}


B3 = {"ollama_ctx_4096": {"excess": 0.3, "ci95": (0.2, 0.4), "rate": 0.4, "default_rate": 0.1, "n_sessions": 5,
                          "tasks_per_session": 30.0, "wrong_turn_wall_s": 18.0, "n_wrong": 60, "silent_wrong": 60,
                          "valid": True}}


def test_cost_p1_v2_pending_cloud_part_stays_none():
    cloud, tm = im.cost_p1_v2(10, 30, 0.2, None, 36.0, 100.0)
    assert cloud is None
    assert tm == pytest.approx(10 * 30 * 0.2 * 36.0 / 3600.0 * 100.0)
    cloud, _ = im.cost_p1_v2(10, 30, 0.2, 0.5, 36.0, 100.0)
    assert cloud == pytest.approx(10 * 30 * 0.2 * 0.5)
    assert im.cost_p1_v2(10, 30, -0.1, 0.5, 36.0, 100.0) == (0.0, 0.0)


def test_p1_v2_lower_bound_while_pending():
    prof = _profile()
    r = im.p1_v2(REPO, prof, {"b3v2": B3})
    assert r["pending"] == ["cloud-redo-cost-per-wrong-task"] and r["lower_bound_only"]
    assert r["usd_day"] == pytest.approx(10 * 30 * 0.3 * 18.0 / 3600.0 * 100.0)
    r2 = im.p1_v2(REPO, prof, {"b3v2": B3, "redo_usd": 1.0})
    assert not r2["pending"] and r2["usd_day"] == pytest.approx(r["usd_day"] + 10 * 30 * 0.3)


def test_v2_ranking_puts_unknown_last_and_split():
    rows = {"P1": {"usd_day": 5.0, "status": "mixed"}, "P2": {"usd_day": None, "status": "PENDING"},
            "P3": {"usd_day": 9.0, "status": "measured"}}
    assert im.v2_ranking(rows) == ["P3", "P1", "P2"]
    assert im.v2_split(rows) == {"mixed": ["P1"], "PENDING": ["P2"], "measured": ["P3"]}


def test_v2_profile_is_stated_and_register_has(tmp_path):
    assert im.V2_PROFILE["model"] == "qwen3:14b" and im.V2_PROFILE["runtime"] == "ollama_ctx_4096"
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "NUMBERS_REGISTER.md").write_text("| x-row | VERIFIED | v |\n| y-row | PENDING | v |\n",
                                                           encoding="utf-8")
    assert im.register_has(tmp_path, "x-row") and not im.register_has(tmp_path, "y-row")
    assert not im.register_has(tmp_path, "z-row")


def test_p1_v2_per_tier_cloud_part_is_tagged_and_not_lower_bound():
    prof = _profile()
    r = im.p1_v2(REPO, prof, {"b3v2": B3, "redo_usd": {"ollama_ctx_4096": 0.03}, "redo_tag": im.CLOUD_TAG_STUB})
    cloud = 10 * 30 * 0.3 * 0.03
    assert not r["lower_bound_only"] and not r["pending"]
    assert r["cloud_tag"] == "priced, not executed" and r["status"] == "mixed"
    assert r["low"] == pytest.approx(cloud)
    assert r["usd_day"] == pytest.approx(cloud + 10 * 30 * 0.3 * 18.0 / 3600.0 * 100.0)
    assert r["cell"]["cloud_usd_day_ci"] == pytest.approx((10 * 30 * 0.2 * 0.03, 10 * 30 * 0.4 * 0.03))
    # a tier missing from the redo data keeps its cloud part PENDING
    r2 = im.p1_v2(REPO, prof, {"b3v2": B3, "redo_usd": {"ollama_ctx_8192": 0.03}})
    assert r2["lower_bound_only"] and r2["cloud_tag"] is None


CC = {"models": {"qwen3:14b": {"default_min": 200.0, "best": "g", "best_name": "kv_q8_0", "best_min": 80.0,
                               "saved_min": 120.0}}, "fa_included": False}


MACHINE = {"usd": 2000.0}


def test_p2_capacity_sessions_machines_and_amortized_usd():
    c = im.p2_capacity(40, 214.8, 75.4, 2000.0)
    assert c["sessions_per_machine_day_default"] == pytest.approx(1440 / 214.8)
    assert c["sessions_per_machine_day_best"] == pytest.approx(1440 / 75.4)
    assert (c["machines_default"], c["machines_best"], c["machines_saved"]) == (6, 3, 3)
    assert c["usd_day"] == pytest.approx(3 * 2000.0 / (3 * 365.0))
    # exact fit does not round up; a measured concurrency multiplies capacity
    assert im.p2_capacity(10, 144.0, 144.0, 1.0)["machines_default"] == 1
    assert im.p2_capacity(40, 214.8, 75.4, 2000.0, concurrency=2)["machines_default"] == 3


def test_p2_v2_capacity_primary_wait_upper_bound_and_pending_until_fa_control():
    prof = _profile()   # 10 sessions/day, hour value 100
    r = im.p2_v2(prof, CC, MACHINE)
    cap = im.p2_capacity(10, 200.0, 80.0, 2000.0)
    assert (cap["machines_default"], cap["machines_best"]) == (2, 1)
    assert r["usd_day"] == pytest.approx(cap["usd_day"]) and r["low"] == pytest.approx(cap["usd_day"])
    wait = 10 * 120.0 / 60.0 * 100.0 * im.INTERACTIVE_SHARE
    assert r["wait_upper_bound"] == pytest.approx(wait) and r["high"] == pytest.approx(cap["usd_day"] + wait)
    assert r["status"] == "PENDING" and r["pending"] == ["b5-fa-control"] and "b5-fa-control" in r["caveat"]
    txt = im.p2_v2_value(r)
    assert "without flash attention" in txt and "ASSUMPTION, must be verified" in txt and "upper bound" in txt
    v = im.p2_v2(prof, dict(CC, fa_included=True), MACHINE, fa_verified=True)
    assert v["status"] == "mixed" and not v["pending"] and v["caveat"] == im.P2_CAVEAT_FA_INCLUDED
    # PENDING with no values without the row, the profile model, or the machine price
    assert im.p2_v2(prof, None, MACHINE)["usd_day"] is None
    assert im.p2_v2(_profile(model="qwen3:8b"), CC, MACHINE)["pending"] == ["cost-cut-config-time"]
    assert im.p2_v2(prof, CC, None)["pending"] == ["impact-ref-machine-price"]


def test_v2_ranking_does_not_place_a_pending_p2_with_values():
    rows = {"P1": {"usd_day": 5.0, "status": "mixed"}, "P2": {"usd_day": 900.0, "status": "PENDING"},
            "P3": {"usd_day": 1.0, "status": "measured"}}
    assert im.v2_ranking(rows) == ["P1", "P3", "P2"]
    rows["P2"]["status"] = "mixed"
    assert im.v2_ranking(rows) == ["P2", "P1", "P3"]


def test_v2_inputs_read_only_verified_rows(tmp_path, monkeypatch):
    import numbers_register as nr
    calls = []
    monkeypatch.setattr(nr, "cloud_redo_data", lambda repo: calls.append("cr") or {
        "stub": True, "per_tier": {"ollama_ctx_4096": {"mean_usd": 0.03}, "ollama_default": {"mean_usd": None}}})
    monkeypatch.setattr(nr, "cost_cut_config_time_data", lambda repo: calls.append("cc") or CC)
    (tmp_path / "docs").mkdir()
    reg = tmp_path / "docs" / "NUMBERS_REGISTER.md"
    reg.write_text("| cloud-redo-cost-per-wrong-task | PENDING | v |\n| cost-cut-config-time | PENDING | v |\n",
                   encoding="utf-8")
    assert im.v2_inputs(tmp_path) == {} and calls == []
    reg.write_text("| cloud-redo-cost-per-wrong-task | VERIFIED | v |\n| cost-cut-config-time | VERIFIED | v |\n",
                   encoding="utf-8")
    got = im.v2_inputs(tmp_path)
    assert got["P1"] == {"redo_usd": {"ollama_ctx_4096": 0.03}, "redo_tag": "priced, not executed"}
    assert got["P2"] is CC
    assert "P2_machine" not in got and "P2_fa_verified" not in got
    monkeypatch.setattr(nr, "cloud_redo_data", lambda repo: {"stub": False, "per_tier": {}})
    assert im.v2_inputs(tmp_path)["P1"]["redo_tag"] == "measured"
    monkeypatch.setattr(im, "ref_machine", lambda repo: MACHINE)
    reg.write_text(reg.read_text(encoding="utf-8") + "| impact-ref-machine-price | VERIFIED | v |\n"
                   "| b5-fa-control | VERIFIED | v |\n", encoding="utf-8")
    got = im.v2_inputs(tmp_path)
    assert got["P2_machine"] is MACHINE and got["P2_fa_verified"] is True


def test_detection_value_every_step_vs_flagged_steps():
    d = im.detection_value(10, 30, 0.4, 0.03, 1.0, 1.0)
    assert d["steps_day"] == 300 and d["wrong_day"] == pytest.approx(120)
    assert d["every_usd_day"] == pytest.approx(300 * 0.03) and d["flagged_usd_day"] == pytest.approx(120 * 0.03)
    assert d["saved_usd_day"] == pytest.approx(180 * 0.03) and d["missed_day"] == 0
    # precision 0.5 doubles the flagged steps; recall 0.75 leaves a quarter of wrong steps unfixed
    d = im.detection_value(10, 30, 0.4, 0.03, 0.5, 0.75)
    assert d["flagged_day"] == pytest.approx(120 * 0.75 / 0.5) and d["missed_day"] == pytest.approx(30)
    assert im.detection_value(10, 30, 0.4, 0.03, None, None)["flagged_usd_day"] is None


def test_detection_projection_is_labelled_and_needs_a_redo_cost():
    prof = _profile()
    conf = {"precision": 1.0, "recall": 1.0, "tp": 4, "fp": 0, "fn": 0, "tn": 6, "scope": "measured on X"}
    p1 = im.p1_v2(REPO, prof, {"b3v2": B3, "redo_usd": {"ollama_ctx_4096": 0.03}, "redo_tag": im.CLOUD_TAG_STUB})
    d = im.detection_projection(p1, prof, conf)
    assert d["label"] == "projection" and d["cloud_tag"] == "priced, not executed"
    assert d["every_usd_day"] == pytest.approx(10 * 30 * 0.03) and d["wrong_day"] == pytest.approx(10 * 30 * 0.4)
    txt = im.detection_value_text(d)
    assert txt.startswith("PROJECTION") and "measured on X" in txt and "priced, not executed" in txt
    p1_live = im.p1_v2(REPO, prof, {"b3v2": B3, "redo_usd": 0.03, "redo_tag": im.CLOUD_TAG_LIVE})
    assert "(measured," in im.detection_value_text(im.detection_projection(p1_live, prof, conf))
    assert im.detection_projection(im.p1_v2(REPO, prof, {"b3v2": B3}), prof, conf) is None
    assert im.detection_value_text(None).startswith("PENDING")
