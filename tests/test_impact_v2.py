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


def test_p2_v2_minutes_saved_times_volume_and_caveat():
    prof = _profile()
    r = im.p2_v2(prof, CC)
    assert r["usd_day"] == pytest.approx(10 * 120.0 / 60.0 * 100.0) and r["low"] == 0.0
    assert r["status"] == "mixed" and not r["pending"] and "b5-fa-control" in r["caveat"]
    assert "without flash attention" in im.p2_v2_value(r)
    assert im.p2_v2(prof, dict(CC, fa_included=True))["caveat"] == im.P2_CAVEAT_FA_INCLUDED
    # PENDING without the row, or without the profile model in it
    assert im.p2_v2(prof, None)["status"] == "PENDING"
    assert im.p2_v2(_profile(model="qwen3:8b"), CC)["pending"] == ["cost-cut-config-time"]


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
    monkeypatch.setattr(nr, "cloud_redo_data", lambda repo: {"stub": False, "per_tier": {}})
    assert im.v2_inputs(tmp_path)["P1"]["redo_tag"] == "measured"
