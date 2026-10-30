"""hybrid-full-pipeline-v1 (analysis/hybrid_full_pipeline.py): the replay pipeline with stand-ins for the post-run
check (A) and the per-task budget (C). The stand-ins live only here; the register row never uses them and stays
PENDING until A and C provide their factories."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "analysis", REPO / "harness"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import hybrid_full_pipeline as H  # noqa: E402


class FlagCheck:
    """Stand-in for A: escalate when the recorded window flag is set. Tests only."""

    source = "test stand-in (window flag)"

    def __init__(self):
        self.seen = []

    def check(self, task_record):
        self.seen.append(task_record)
        return {"escalate": bool(task_record.get("over_loaded_window")), "reason": "flag"}


class CapBudget:
    """Stand-in for C: a per-task cap and a total cap. Tests only."""

    source = "test stand-in (caps)"

    def __init__(self, per_task=1.0, total=10.0):
        self.per_task, self.total, self.spent = per_task, total, 0.0

    def allow(self, task_key, usd_estimate):
        ok = usd_estimate <= self.per_task and self.spent + usd_estimate <= self.total
        if ok:
            self.spent += usd_estimate
        return {"allow": ok, "cap_usd": self.per_task}


def _inputs():
    first = {1: [(0, 100), (1, 200), (2, 300)]}            # growth 100/turn, ceiling 1000 -> turn >= 10 overflows
    tasks, local, cloud, esc = [], {}, {}, {}
    spec = {  # task: (turn_idx, local ok, window flag)
        "t0": (0, True, False), "t3": (3, False, True), "t5": (5, False, False), "t8": (8, True, False),
        "t12": (12, True, False)}
    for tid, (idx, ok, flag) in spec.items():
        k = (1, tid)
        tasks.append(k)
        local[k] = {"turn_idx": idx, "end_state_ok": ok, "over_loaded_window": flag, "rule1_json_keys": True,
                    "end_state_diff": [], "task_id": tid}
        cloud[k] = {"ok": True, "usd": 0.01}
        esc[k] = {"ok": True, "usd": 0.02, "src": "imputed"}
    return tasks, local, first, cloud, esc


def test_pipeline_route_check_rollback_budget():
    tasks, local, first, cloud, esc = _inputs()
    chk = FlagCheck()
    r = H.run_pipeline(tasks, local, first, 1000, cloud, esc, chk, CapBudget())
    assert r["per_task"] == {(1, "t0"): "local", (1, "t3"): "escalated", (1, "t5"): "local", (1, "t8"): "local",
                             (1, "t12"): "routed"}
    assert (r["kept_local"], r["routed"], r["escalated"], r["correct"], r["wrong_kept_local"]) == (3, 1, 1, 4, 1)
    assert r["usd"] == pytest.approx(0.03) and r["imputed_escalations"] == 1
    # A never sees the grader's fields
    assert all("end_state_ok" not in s and "end_state_diff" not in s and "rule1_json_keys" not in s for s in chk.seen)
    assert len(chk.seen) == 4            # the routed task is not checked


def test_budget_refusals_fall_back_to_local():
    tasks, local, first, cloud, esc = _inputs()
    r = H.run_pipeline(tasks, local, first, 1000, cloud, esc, FlagCheck(), CapBudget(per_task=0.0))
    assert r["routed"] == 0 and r["escalated"] == 0
    assert r["budget_refused_route"] == 1 and r["budget_refused_escalation"] == 1
    assert r["kept_local"] == 5 and r["correct"] == 3


def test_interfaces_and_pending():
    assert isinstance(FlagCheck(), H.PostRunCheck) and isinstance(CapBudget(), H.TaskBudget)
    if not (REPO / H.A_MODULE).exists():
        assert H.pending_reason(REPO) == H.PENDING_A
        assert H.hybrid_full_pipeline(REPO) == {"status": "PENDING", "reason": H.PENDING_A}


def test_escalation_order():
    cloud = {(1, "a"): {"ok": True, "usd": 0.01}, (1, "b"): {"ok": False, "usd": 0.01}, (1, "c"): {"ok": True,
                                                                                                    "usd": 0.01}}
    rb = {("T", 1, "a"): {"ok": False, "usd": 0.5}}
    redo = {("T", 1, "a"): {"ok": True, "usd": 0.4}, ("T", 1, "b"): {"ok": True, "usd": 0.3}}
    e = H.escalation_outcomes(REPO, "T", cloud, redo, rb, 0.9)
    assert e[(1, "a")]["src"] == "rollback_redo" and e[(1, "b")]["src"] == "full_history_redo"
    assert e[(1, "c")] == {"ok": True, "usd": 0.9, "src": "imputed"}
    assert all(v["src"] == "imputed" for v in H.escalation_outcomes(REPO, None, cloud, redo, rb, 0.9).values())


@pytest.mark.skipif(not (REPO / "results/x2_own_config_probe_q4_28928.jsonl").exists(), reason="files not synced")
def test_replay_runs_with_stand_ins():
    d = H.hybrid_full_pipeline(REPO, check=FlagCheck(), budget=CapBudget(per_task=1.0, total=100.0))
    assert d["status"] == "VERIFIED" and [c["config"] for c in d["cells"]][0] == H.PRIMARY
    assert all(c["n"] == 90 for c in d["cells"])
    assert "MET" in H.hybrid_full_pipeline_text(d) or "MISSED" in H.hybrid_full_pipeline_text(d)
