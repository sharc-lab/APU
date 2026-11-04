"""long-task-scale-v1 (operator decision 2026-11-03): the queue entries, the hours estimate copied into them, and the
result rows (PENDING until both runs end; the savings re-run only for a model with at least 1/30)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "analysis", REPO / "harness"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numbers_register as NR  # noqa: E402
import x2_queue_fill as qf  # noqa: E402

JOBS = {"qwen3:32b": "x2_longtask_scale_v1_qwen3_32b", "llama3.3:70b": "x2_longtask_scale_v1_llama33_70b"}


def test_queue_entries():
    for model, job in JOBS.items():
        e = qf.entry(job)
        c = e["cmd"]
        assert c[1] == "ltscale\\x2_longtask.py" and c[c.index("--model") + 1] == model
        assert c[c.index("--arms") + 1] == "ctx_40960" and c[c.index("--job-id") + 1] == "x2_longtask_scale_v1"
        assert c[c.index("--tasks") + 1].split(",") == [f"lt_openhands_{i:02d}" for i in range(1, 11)]
        assert c[c.index("--out") + 1].endswith(Path(NR.LTS_MODELS[model]).name)
        assert e["projected_duration_h"] == qf.LTS_HOURS[model][0]
        assert "tq.advance()" in e["note"] and "sets no pause" in e["note"] and "no cloud call" in e["note"]


@pytest.mark.skipif(not (REPO / NR.LONGTASK_LOCAL_FILE).exists(), reason="data not synced")
def test_queue_hours_match_register_estimate():
    h = NR.long_task_scale_hours(REPO)
    for model, (pt, hi) in qf.LTS_HOURS.items():
        assert round(h["models"][model]["point_h"], 2) == pt and round(h["models"][model]["high_h"], 2) == hi
    assert h["n_tasks"] == 30


def test_rows_listed_and_pending_on_both_files():
    by = {e["claim_id"]: e for e in NR.NUMBER_ENTRIES}
    for cid in ("long-task-scale-local", "spillway-long-task-savings-scale"):
        e = by[cid]
        assert e["pending_ok"] and sorted(e["require_run_end"]) == sorted(NR.LTS_MODELS.values())
    assert "long-task-scale-hours" in by


def _task(tid, ok, seed=20260901):
    return {"record": "longtask_task", "arm_id": "longtask_ctx_40960", "stratum": "openhands", "task_id": tid,
            "seed": seed, "correct": ok, "attempt_id": "a"}


def test_savings_rerun_only_with_a_correct_task(tmp_path, monkeypatch):
    for f in NR.LTS_MODELS.values():
        p = tmp_path / f
        p.parent.mkdir(parents=True, exist_ok=True)
        rows = [_task(f"lt_openhands_{i:02d}", False) for i in range(1, 11)]
        rows.append({"record": "run_end", "note": "completed", "attempt_id": "a"})
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    called = []
    monkeypatch.setattr(NR, "spillway_long_task_data", lambda *a, **k: called.append(a))
    v = NR.compute_spillway_long_task_savings_scale(tmp_path)["value"]
    assert not called and v.count("not re-run (pre-registered rule: at least 1/30)") == 2
