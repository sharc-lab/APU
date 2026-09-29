"""Pure-logic tests for harness/q0_token_calibration.py's summarize(). No server, no real tokenize call."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
import q0_token_calibration as qc  # noqa: E402


def test_summarize_all_ok():
    rows = [
        {"task_type": "niah_multikey", "target_tokens": 2000, "actual_tokens": 1990, "ratio": 0.995, "ok": True},
        {"task_type": "niah_multikey", "target_tokens": 8000, "actual_tokens": 8010, "ratio": 1.001, "ok": True},
    ]
    s = qc.summarize(rows)
    assert s["niah_multikey"]["all_ok"] is True
    assert len(s["niah_multikey"]["cells"]) == 2


def test_summarize_flags_out_of_range_task():
    rows = [
        {"task_type": "common_words_extraction", "target_tokens": 2000, "actual_tokens": 12657, "ratio": 6.33, "ok": False},
        {"task_type": "common_words_extraction", "target_tokens": 8000, "actual_tokens": 8010, "ratio": 1.001, "ok": True},
    ]
    s = qc.summarize(rows)
    assert s["common_words_extraction"]["all_ok"] is False
    assert s["common_words_extraction"]["worst_ratio"] == 6.33


def test_summarize_independent_across_task_types():
    rows = [
        {"task_type": "a", "target_tokens": 2000, "actual_tokens": 1990, "ratio": 0.995, "ok": True},
        {"task_type": "b", "target_tokens": 2000, "actual_tokens": 4000, "ratio": 2.0, "ok": False},
    ]
    s = qc.summarize(rows)
    assert s["a"]["all_ok"] is True
    assert s["b"]["all_ok"] is False


def test_ok_range_constants_match_the_users_bar():
    assert qc.OK_LOW == 0.95
    assert qc.OK_HIGH == 1.05
