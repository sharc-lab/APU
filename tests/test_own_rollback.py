"""Tests for analysis/own_rollback.py (own-rollback, 2026-10-23). Pure or offline over committed result files; no
network."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
import own_rollback as orb  # noqa: E402

LOCAL = {(1, "a"): {"ok": True, "flag": False, "turn_idx": 1},
         (1, "b"): {"ok": False, "flag": True, "turn_idx": 2},
         (1, "c"): {"ok": False, "flag": False, "turn_idx": 3},
         (1, "d"): {"ok": True, "flag": True, "turn_idx": 4}}
CLOUD = {k: {"ok": True, "usd": 0.01} for k in LOCAL}
REDO = {(1, "b"): {"ok": False, "usd": 0.03}, (1, "c"): {"ok": True, "usd": 0.02}}


def test_oracle_signal_escalates_exactly_the_wrong_tasks():
    p = orb.local_first_policy(LOCAL, CLOUD, REDO, {}, "oracle", 0.025, {(1, 2): "earlier_state_repair"})
    assert p["kept_local"] == 2 and p["escalated"] == 2 and p["full_history_redo"] == 2 and p["imputed"] == 0
    assert p["correct"] == 3 and p["usd_task"] == pytest.approx(0.05 / 4)
    assert p["escalated_unfixed"] == 1 and p["unfixed_repair"] == 1


def test_window_signal_imputes_right_local_tasks_and_misses_unflagged():
    p = orb.local_first_policy(LOCAL, CLOUD, REDO, {}, "window", 0.025)
    assert p["escalated"] == 2 and p["full_history_redo"] == 1 and p["imputed"] == 1
    assert p["correct"] == 2   # a local, d imputed from all-cloud; b redo wrong; c missed (not flagged)
    assert p["usd_task"] == pytest.approx((0.03 + 0.025) / 4)


def test_rollback_redo_row_takes_precedence():
    p = orb.local_first_policy(LOCAL, CLOUD, REDO, {(1, "b"): {"ok": True, "usd": 0.04}}, "oracle", 0.025)
    assert p["rollback_redo"] == 1 and p["full_history_redo"] == 1 and p["correct"] == 4


def test_wrong_task_without_redo_row_is_an_error():
    with pytest.raises(ValueError, match="no redo row"):
        orb.local_first_policy(LOCAL, CLOUD, {}, {}, "oracle", 0.025)


def test_rollback_analysis_on_committed_files():
    a = orb.rollback_analysis(REPO)
    assert a["n"] == 28
    assert a["rollback_equals_redo_env"] == 28 and a["with_earlier_wrong"] == 28 and a["diff_within_dirty"] == 28
    assert a["guardrail_state_changes"] == 0
    assert "28/28" in orb.rollback_analysis_text(a)
