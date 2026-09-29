"""Tests for analysis/offline_token_calibration.py's calibrate()/summarize() logic, with a fake tokenizer (no real
Hugging Face download, no network)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
import offline_token_calibration as otc  # noqa: E402
import quality_suite as qs  # noqa: E402


class _FakeTokenizer:
    """encode() returns one token per 4 characters, matching build_task's own char-based length estimate closely
    enough that most cells land inside the 0.95-1.05 band in tests, while still being a deterministic, fast stand-in
    for a real tokenizer."""
    def __init__(self, chars_per_token=4.0):
        self.chars_per_token = chars_per_token

    def encode(self, text):
        return list(range(max(1, int(len(text) / self.chars_per_token))))


def test_calibrate_returns_one_cell_per_target_length_per_task():
    tokenizers = {"fake": _FakeTokenizer()}
    results = otc.calibrate(tokenizers, task_types=[qs.TASK_TYPES[0]], target_lengths=(2000, 8000))
    cells = results["fake"][qs.TASK_TYPES[0]]
    assert len(cells) == 2
    assert {c["target"] for c in cells} == {2000, 8000}
    assert all("actual" in c and "ratio" in c and "ok" in c for c in cells)


def test_calibrate_flags_out_of_range_ratio():
    tokenizers = {"bad": _FakeTokenizer(chars_per_token=1.0)}  # wildly over-counts -> ratio way above 1.05
    results = otc.calibrate(tokenizers, task_types=[qs.TASK_TYPES[0]], target_lengths=(2000,))
    cell = results["bad"][qs.TASK_TYPES[0]][0]
    assert cell["ok"] is False
    assert cell["ratio"] > otc.OK_HIGH


def test_summarize_all_ok_when_every_cell_ok():
    results = {"fam": {"t1": [{"task_type": "t1", "target": 2000, "actual": 1990, "ratio": 0.995, "ok": True}]}}
    s = otc.summarize(results)
    assert s["fam"]["all_ok"] is True
    assert s["fam"]["fail_cells"] == []


def test_summarize_collects_fail_cells_and_worst():
    results = {"fam": {"t1": [{"task_type": "t1", "target": 2000, "actual": 1990, "ratio": 0.995, "ok": True}],
                        "t2": [{"task_type": "t2", "target": 2000, "actual": 12657, "ratio": 6.33, "ok": False}]}}
    s = otc.summarize(results)
    assert s["fam"]["all_ok"] is False
    assert len(s["fam"]["fail_cells"]) == 1
    assert s["fam"]["fail_cells"][0]["task_type"] == "t2"
    assert s["fam"]["worst"]["task_type"] == "t2"


def test_ok_range_matches_the_labs_own_bar():
    assert otc.OK_LOW == 0.95
    assert otc.OK_HIGH == 1.05


def test_calibrate_covers_every_task_type_by_default():
    tokenizers = {"fake": _FakeTokenizer()}
    results = otc.calibrate(tokenizers, target_lengths=(2000,))
    assert set(results["fake"]) == set(qs.TASK_TYPES)
