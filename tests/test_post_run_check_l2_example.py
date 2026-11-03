"""Amendment 2026-11-03 (L2 tool-call example from the most recent completed turn): analysis/post_run_check_l2_example.py.
Pure pieces on hand-built rows, then the offline recomputation on the committed live 8192 local phase file. No network."""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "harness", REPO / "analysis"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import post_run_check_l2_example as X  # noqa: E402

CALL = {"render_ok": True, "system_kept": True, "current_user_message_kept": True, "dropped_turns": [1, 2],
        "example_turn": 1, "example_kept": False, "l2_fail": True}


def test_example_kept_and_recomputed_call():
    assert X.example_kept(1, CALL) is False
    assert X.example_kept(3, CALL) is True
    assert X.example_kept(None, CALL) is None
    assert X.example_kept(3, {**CALL, "render_ok": False}) is None
    assert X.recomputed_call(CALL, 1)["l2_fail"]
    assert not X.recomputed_call(CALL, 3)["l2_fail"]
    assert X.recomputed_call({**CALL, "system_kept": False}, 3)["l2_fail"]
    assert X.recomputed_call({**CALL, "render_ok": False}, 3)["l2_fail"]
    assert not X.recomputed_call(CALL, None)["l2_fail"]


def test_summary_and_false_esc_layers():
    tasks = [{"local_ok": False, "b": "escalate", "fired_b": ["L3"]},
             {"local_ok": True, "b": "escalate", "fired_b": ["L1", "L2"]},
             {"local_ok": True, "b": "accept", "fired_b": []}]
    s = X.summary(tasks, "b")
    assert (s["caught"], s["wrong"], s["false_esc"], s["right"], s["kept_local"]) == (1, 1, 1, 2, 1)
    assert X.false_esc_layers(tasks, "b") == {"L1": {"fired": 1, "alone": 0}, "L2": {"fired": 1, "alone": 0}}


LIVE = REPO / X.LIVE_LOCAL_FILE


@pytest.mark.skipif(not LIVE.exists(), reason="live local phase file not present")
def test_live_recompute_reproduces_recorded_rows_and_never_adds_l2_failures():
    d = X.live_recompute(REPO)
    assert d["status"] == "VERIFIED", d.get("mismatches")
    assert d["l2_fail_calls_after"] <= d["l2_fail_calls_before"]
    for t in d["tasks"]:   # the new rule can only remove the example condition's failures, never add a layer
        assert set(t["fired_after"]) <= set(t["fired_before"])
        assert t["example_after"] is None or t["example_after"] < t["turn_idx"]
        if t["example_before"] is not None:
            assert t["example_after"] >= t["example_before"]
