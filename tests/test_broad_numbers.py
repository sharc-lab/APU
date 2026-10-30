"""F1-F5 register rows (docs/EXPERIMENTS.md "PRE-REGISTRATION (2026-10-29)"): the result rows stay PENDING until
their files exist, the estimate rows carry their targets, and the cross-model rows carry the operator's label
("R2 rule-following evidence ... not B3 task accuracy"; "3+ models on B3: NOT MET")."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
import numbers_register as nr  # noqa: E402

ENTRIES = {e["claim_id"]: e for e in nr.NUMBER_ENTRIES}
NEW = ("hybrid-long-tasks", "hybrid-long-tasks-estimate", "cuad-licence", "cuad-pipeline-estimate", "cuad-pipeline",
       "xmodel-wrong-rate", "bls-oews-hourly-wage", "time-dollar-value", "pitch-savings-table", "device-db-coverage")


def _value(cid):
    return ENTRIES[cid]["compute"](REPO)["value"]


def test_rows_registered():
    for c in NEW:
        assert c in ENTRIES


@pytest.mark.parametrize("cid,path", [("hybrid-long-tasks", nr.HLT_FILE), ("cuad-pipeline", nr.CUAD_FILE)])
def test_result_rows_pending_until_file(cid, path):
    if (REPO / path).exists():
        pytest.skip("result file present")
    assert nr._pending(ENTRIES[cid]) == [path]


def test_hybrid_long_tasks_result_target(tmp_path):
    rows = [{"record": "run_start", "attempt_id": "a"}]
    rows += [{"record": "hybrid_long_task", "arm": "all_cloud", "correct": True, "usd_billed": 0.1, "attempt_id": "a"}
             for _ in range(10)]
    rows += [{"record": "hybrid_long_task", "arm": "hybrid", "correct": True, "usd_billed": 0.05, "attempt_id": "a"}
             for _ in range(10)]
    rows.append({"record": "run_end", "attempt_id": "a"})
    (tmp_path / "results").mkdir()
    (tmp_path / nr.HLT_FILE).write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    v = nr.compute_hybrid_long_tasks(tmp_path)["value"]
    assert "savings 50.0%" in v and "target MET" in v


def test_xmodel_label_and_b3_not_met():
    v = _value("xmodel-wrong-rate")
    assert "R2 rule-following evidence" in v and "not B3 task accuracy" in v
    assert "3+ models on B3: NOT MET" in v
    for m in ("mistral-small3.2:24b", "llama3.3:70b", "qwen3:8b"):
        assert m in v.split("no valid B3 v2 run")[1]
    assert "target MISSED" in v or "target MET" in v


def test_xmodel_matches_source_rows():
    d = nr.xmodel_data(REPO)
    assert d["r2"]["llama3.1:8b"]["excess"] == pytest.approx(1.0)
    assert d["r2"]["qwen3:14b"]["excess"] == pytest.approx(0.8)
    assert d["r2"]["qwen3:8b"]["excess"] == pytest.approx(0.2)
    assert d["b3_valid"] == ["qwen3:14b"]


def test_pitch_table_marks_and_label():
    v = _value("pitch-savings-table")
    assert "[measured" in v and "[projected]" in v
    assert "R2 rule-following evidence" in v and "3+ models on B3: NOT MET" in v


def test_hybrid_estimate_has_target_per_case():
    v = _value("hybrid-long-tasks-estimate")
    assert v.startswith("ESTIMATE and PROJECTION")
    assert v.count("target M") == 3


def test_time_dollar_uses_bls_file():
    w = json.loads((REPO / nr.BLS_OEWS_FILE).read_text(encoding="utf-8"))["value_usd_per_hour"]
    assert f"USD {w:.2f}/h" in _value("time-dollar-value")


def test_no_em_dash_in_new_values():
    for c in ("xmodel-wrong-rate", "hybrid-long-tasks-estimate", "cuad-pipeline-estimate", "time-dollar-value"):
        assert chr(0x2014) not in _value(c)
