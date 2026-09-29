"""Pure-logic tests for harness/q0_positive_control.py: truncate_front and summarize. No server, no real Q0 call."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
import q0_positive_control as qc  # noqa: E402


def test_truncate_front_removes_half_the_characters():
    prompt = "0123456789"
    assert qc.truncate_front(prompt, 0.5) == "56789"
    assert qc.truncate_front(prompt, 0.0) == prompt
    assert qc.truncate_front(prompt, 1.0) == ""


def test_summarize_pass_case():
    rows = [
        {"task_type": "niah_multikey", "condition": "full", "score": 1.0},
        {"task_type": "niah_multikey", "condition": "truncated_50pct_front", "score": 0.0},
    ]
    s = qc.summarize(rows)
    assert s["niah_multikey"]["full_pass"] is True
    assert s["niah_multikey"]["truncated_clearly_lower"] is True
    assert s["niah_multikey"]["control_ok"] is True


def test_summarize_fail_case_low_full_score():
    rows = [
        {"task_type": "niah_multikey", "condition": "full", "score": 0.5},
        {"task_type": "niah_multikey", "condition": "truncated_50pct_front", "score": 0.0},
    ]
    s = qc.summarize(rows)
    assert s["niah_multikey"]["full_pass"] is False
    assert s["niah_multikey"]["control_ok"] is False


def test_summarize_fail_case_truncation_not_lower():
    rows = [
        {"task_type": "niah_multikey", "condition": "full", "score": 1.0},
        {"task_type": "niah_multikey", "condition": "truncated_50pct_front", "score": 1.0},
    ]
    s = qc.summarize(rows)
    assert s["niah_multikey"]["full_pass"] is True
    assert s["niah_multikey"]["truncated_clearly_lower"] is False
    assert s["niah_multikey"]["control_ok"] is False


def test_summarize_multiple_task_types_independent():
    rows = [
        {"task_type": "a", "condition": "full", "score": 1.0},
        {"task_type": "a", "condition": "truncated_50pct_front", "score": 0.0},
        {"task_type": "b", "condition": "full", "score": 0.2},
        {"task_type": "b", "condition": "truncated_50pct_front", "score": 0.0},
    ]
    s = qc.summarize(rows)
    assert s["a"]["control_ok"] is True
    assert s["b"]["control_ok"] is False
