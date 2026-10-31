"""The q4_0 28928 operating point (operator 2026-10-30): the frontier figure's values come from the register rows
operating-point-q4-28928 and hybrid-full-pipeline-v1, and those rows match a fresh computation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "analysis", REPO / "harness"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import operating_point_frontier as F  # noqa: E402


def _rows_ok():
    try:
        F.row_value(REPO, F.OP_ROW)
        F.row_value(REPO, F.HFP_ROW)
        return True
    except (KeyError, ValueError, FileNotFoundError):
        return False


pytestmark = pytest.mark.skipif(not _rows_ok(), reason="operating-point rows not VERIFIED in the register")


def test_committed_json_equals_values_parsed_from_rows():
    committed = json.loads((REPO / F.OUT_JSON).read_text(encoding="utf-8"))
    assert committed == F.frontier_data(REPO)
    assert (REPO / F.OUT_PNG).exists()


def test_headline_points_are_the_three_named_points():
    d = F.frontier_data(REPO)
    head = {p["label"]: p for p in d["points"] if p["role"] == "headline"}
    assert set(head) == {"hybrid q4_0 28928", "hybrid f16 8192", "claude-sonnet-5-5 all-cloud"}
    op = F.row_value(REPO, F.OP_ROW)
    q4 = head["hybrid q4_0 28928"]
    assert f"kept local {q4['kept_local']}/{q4['n']}" in op and f"({q4['accuracy']:.1f}%)" in op
    assert f"cloud USD {q4['usd_task']:.5f}/task" in op and "replay of recorded runs" in op and "IMPUTED" in op


def test_rows_match_a_fresh_computation():
    import numbers_register as nr
    o = nr.operating_point_data(nr.REPO)
    assert o["status"] == "VERIFIED"
    head = {p["label"]: p for p in F.frontier_data(REPO)["points"]}
    for name, c in [("hybrid q4_0 28928", o["op"])] + [(f"hybrid {f['config']}", f) for f in o["frontier"]]:
        assert head[name]["usd_task"] == pytest.approx(round(c["usd_task"], 5))
        assert head[name]["accuracy"] == pytest.approx(round(100 * c["accuracy"], 1))
        assert head[name]["kept_local"] == c["kept_local"]
    a = head["claude-sonnet-5-5 all-cloud"]
    assert a["usd_task"] == pytest.approx(round(o["all_cloud"]["usd_task"], 5))
    assert a["accuracy"] == pytest.approx(round(100 * o["all_cloud"]["correct"] / o["all_cloud"]["n"], 1))
