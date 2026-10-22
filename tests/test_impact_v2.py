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
