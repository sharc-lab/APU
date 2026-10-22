import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
import x2_queue_fill as qf  # noqa: E402


def test_every_entry_releases_itself_and_sets_no_pause():
    for j in qf.ENTRIES:
        e = qf.entry(j)
        assert e["status"] == "pending"
        assert "tq.advance()" in e["note"] and "sets no pause" in e["note"]
        assert e["cmd"][0].lower().endswith("python.exe")


def test_append_keeps_existing_and_refuses_duplicates():
    items = [{"id": "old", "status": "done"}]
    new = qf.append_entries(items, ["x2_num_keep_qwen3_14b"])
    assert [i["id"] for i in new] == ["old", "x2_num_keep_qwen3_14b"]
    assert items == [{"id": "old", "status": "done"}]
    with pytest.raises(ValueError):
        qf.append_entries(new, ["x2_num_keep_qwen3_14b"])


def test_gated_entry_needs_its_dependency_ahead():
    for j, e in qf.ENTRIES.items():
        req = (e.get("gate") or {}).get("requires_done")
        if req:
            with pytest.raises(ValueError):
                qf.append_entries([], [j])
            new = qf.append_entries([], [req, j])
            assert [i["id"] for i in new] == [req, j]


def test_num_keep_qwen_entry_matches_llama_test_command():
    cmd = qf.entry("x2_num_keep_qwen3_14b")["cmd"]
    assert cmd[1] == "numkeep\\x2_r2_sidecar.py"
    joined = " ".join(cmd)
    for part in ("--mode real", "--call2-tools off", "--models qwen3:14b", "--tiers 4096",
                 "--seeds 20260901,20260902,20260903", "--require-validation-gates", "--num-keep system"):
        assert part in joined
