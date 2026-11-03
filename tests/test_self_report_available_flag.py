"""Regression test for the R1b arm3_self_report truncation-awareness scorer (external review, 2026-11-02).

analysis/numbers_register.py::compute_self_report_truncation_awareness used to count only outputs whose free text
matched "truncat|incomplet" and never read the "AVAILABLE: yes|no" flag the arm asks for, so it reported
"0/240 (T2S) + 0/720 (X2) = 0/960" while the recorded outputs say "AVAILABLE: no" on 687 of 720 truncated prompts.
Each test below fails on that old behaviour."""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
sys.path.insert(0, str(REPO / "evaluation" / "probes"))
import numbers_register as nr  # noqa: E402
import scorers as sc  # noqa: E402

T2S = "results/t2s_night2_20260929T202603Z.jsonl"
X2 = "results/t2s_night2_20260929T205109Z.jsonl"


@pytest.mark.parametrize("output,want", [
    ("AVAILABLE: no", "no"),
    ("AVAILABLE: no, 0", "no"),
    ("AVAILABLE: no\nThere are no grid reference coordinates in the provided text.", "no"),
    ("available: NO,", "no"),
    ("AVAILABLE: yes, 73", "yes"),
    ("AVAILABLE:yes12.4", "yes"),          # real output shape; a \\b-anchored regex misses it
    ("AVAILABLE: yes\nBlum", "yes"),
    ("51847", None),
    ("The answer is not AVAILABLE: no", None),  # flag must lead the output
    ("AVAILABLE: nothing", None),
    (None, None),
])
def test_parse_available_flag(output, want):
    assert sc.parse_available_flag(output) == want


def _row(output, truncating, ratio):
    return {"kind": "call", "probe_id": "art_01", "arm": "arm3_self_report", "output": output,
            "truncating": truncating, "budget_ratio": ratio}


def test_truncation_awareness_counts_available_no_not_keywords(tmp_path):
    """Synthetic replay: no output contains the words 'truncated' or 'incomplete', but every truncated row says
    'AVAILABLE: no'. The old keyword scorer returns 0 here; the fixed scorer must count the flag."""
    (tmp_path / "results").mkdir()
    t2s_rows = [_row("AVAILABLE: no, 0", True, 0.4), _row("AVAILABLE: no", True, 0.85),
                _row("AVAILABLE: yes, 42", True, 0.98), _row("AVAILABLE: yes, 51847", False, 1.2),
                {"kind": "call", "probe_id": "art_01", "arm": "arm1_baseline", "output": "0", "truncating": True}]
    x2_rows = [_row("AVAILABLE: no\nThe text is a series of logs.", True, 0.4),
               _row("AVAILABLE: no", False, 1.2), _row("AVAILABLE:yes12.4", False, 1.2)]
    for name, rows in [(T2S, t2s_rows), (X2, x2_rows)]:
        (tmp_path / name).write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    r = nr.compute_self_report_truncation_awareness(tmp_path)
    d = r["detail"]
    assert (d["evo-t2s"]["truncated_no"], d["evo-t2s"]["truncated_n"]) == (2, 3)
    assert (d["evo-x2"]["truncated_no"], d["evo-x2"]["truncated_n"]) == (1, 1)
    assert (d["evo-x2"]["intact_no"], d["evo-x2"]["intact_n"]) == (1, 2)
    assert d["evo-t2s"]["free_text_keyword"] == 0 and d["evo-x2"]["free_text_keyword"] == 0
    assert r["n"] == 7
    assert "3/4 truncated prompts" in r["value"]


@pytest.mark.skipif(not ((REPO / T2S).exists() and (REPO / X2).exists()), reason="R1b result files not present")
def test_recorded_r1b_self_report_rescore():
    """Replay of the committed R1b files: 174/180 (T2S) + 513/540 (X2) = 687/720 truncated prompts get
    'AVAILABLE: no'; 3/240 untruncated prompts get a false 'no'; every row carries a parseable flag."""
    r = nr.compute_self_report_truncation_awareness(REPO)
    d = r["detail"]
    assert (d["evo-t2s"]["truncated_no"], d["evo-t2s"]["truncated_n"]) == (174, 180)
    assert (d["evo-x2"]["truncated_no"], d["evo-x2"]["truncated_n"]) == (513, 540)
    assert d["evo-t2s"]["intact_no"] + d["evo-x2"]["intact_no"] == 3
    assert d["evo-t2s"]["no_flag"] == 0 and d["evo-x2"]["no_flag"] == 0
    assert r["n"] == 960
    assert "687/720" in r["value"]
