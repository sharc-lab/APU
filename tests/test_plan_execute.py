"""harness/plan_execute.py with fake runtimes only (no network, no model, no server, no queue)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))

import longtask_env as L  # noqa: E402
import plan_execute as PE  # noqa: E402

SEED = 20260901


def _task(tid):
    return L.build_tasks(SEED, [tid])[0]


def test_target_text_is_the_operators():
    assert PE.TARGET_TEXT == "within 3 points of all-cloud accuracy at under 30% of all-cloud cost"
    assert PE.TARGET_POINTS == 3.0 and PE.TARGET_COST_RATIO == 0.30


def test_work_order_items_and_parse_plan():
    task, env = _task("lt_swegym_06")
    items = PE.work_order_items(env["files"][task.path]["content"])
    assert items == list(task.items)
    good = json.dumps({"steps": [{"n": k, "instruction": t} for k, t in enumerate(items, 1)]})
    steps, prob = PE.parse_plan("Here: " + good, len(items))
    assert prob is None and [s["instruction"] for s in steps] == items
    assert PE.parse_plan(good, len(items) + 1)[0] is None
    assert PE.parse_plan("no json", len(items))[1].startswith("no JSON plan")
    assert PE.parse_plan(json.dumps({"steps": [{"n": 1}]}), 1)[1] == "step 1 has no instruction"
    assert PE.parse_plan(good, None)[0] is None


def test_parse_verdict():
    assert PE.parse_verdict('{"redo": [3]}', [1, 2, 3]) == ({3}, True)
    assert PE.parse_verdict('{"redo": []}', [1, 2]) == (set(), True)
    assert PE.parse_verdict("garbage", [1, 2]) == ({1, 2}, False)
    assert PE.parse_verdict('{"redo": [9]}', [1, 2]) == ({1, 2}, False)


def test_item_reference_rebuilds_the_reference():
    for s in (20260901, 20260902, 20260903):
        for task, _env in L.build_tasks(s):
            ref = PE.item_reference(task)
            assert len(ref) == len(task.items)
            assert [c for seq in ref for c in seq if c["name"] != "get_record"] == list(task.reference_writes)


def test_all_oracle_is_correct_and_escalates_nothing():
    task, env = _task("lt_openhands_03")
    row = PE.run_task(PE.OracleCloud(task), PE.OracleLocal(task), task, env, log=lambda *a: None)
    assert row["correct"] and row["n_escalated"] == 0 and row["n_steps"] == len(task.items)
    assert len(row["verdicts"]) == -(-len(task.items) // PE.CHECKPOINT_EVERY)


def test_shape_flagged_steps_are_escalated_and_fixed():
    task, env = _task("lt_swegym_10")
    row = PE.run_task(PE.OracleCloud(task), PE.OracleLocal(task, wrong={1, 4}), task, env, log=lambda *a: None)
    assert row["n_flagged_shape"] == 2 and row["n_escalated"] == 2 and row["correct"]
    assert {e["n"] for e in row["escalations"]} == {1, 4}
    assert all(e["reason"].startswith("shape:no_tool_call") for e in row["escalations"])


class WrongWriteLocal(PE.OracleLocal):
    """A silent wrong step: a write with a wrong value that passes checkpoint (a)."""

    def __init__(self, task, k):
        super().__init__(task)
        self.k = k
        seq = self.ref[k - 1]
        bad = []
        for c in seq:
            if c["name"] == "transfer_funds":
                c = {"name": c["name"], "arguments": {**c["arguments"], "amount": c["arguments"]["amount"] + 1}}
            elif c["name"] == "update_record" and c["arguments"]["field"] == "unit_price":
                c = {"name": c["name"], "arguments": {**c["arguments"], "value": "999.99"}}
            bad.append(c)
        self.ref[k - 1] = bad


def _wrong_write_k(task):
    return next(k for k, kind in enumerate(task.item_kinds, 1) if kind in ("pay", "price"))


def test_silent_wrong_write_needs_the_verifier():
    task, env = _task("lt_openhands_03")
    k = _wrong_write_k(task)
    missed = PE.run_task(PE.OracleCloud(task), WrongWriteLocal(task, k), task, env, log=lambda *a: None)
    assert missed["n_flagged_shape"] == 0 and missed["n_escalated"] == 0 and not missed["correct"]
    caught = PE.run_task(PE.OracleCloud(task, redo_all=True), WrongWriteLocal(task, k), task, env,
                         log=lambda *a: None)
    assert caught["n_redo_verifier"] == len(task.items) and caught["correct"]


class BadPlanCloud(PE.OracleCloud):
    def chat(self, model, messages, num_ctx, tools, think, extra=None):
        if messages[-1]["role"] == "user" and messages[-1]["content"] == PE.PLAN_INSTRUCTION:
            return PE._text("I cannot plan this.")
        if messages[1]["content"] == L.user_message(self.task) and len(messages) > 2:
            return PE._text("Done.")      # the whole-task fallback answers without tools
        return super().chat(model, messages, num_ctx, tools, think, extra)


def test_plan_failure_falls_back_to_the_whole_task_on_the_cloud():
    task, env = _task("lt_swegym_06")
    row = PE.run_task(BadPlanCloud(task), PE.OracleLocal(task), task, env, log=lambda *a: None)
    assert row["plan_failed"] and row["fallback"] == "whole task on the cloud" and row["n_steps"] == 0


class WritingPlanner(PE.OracleCloud):
    def chat(self, model, messages, num_ctx, tools, think, extra=None):
        if len(messages) == 2 and messages[1]["content"] == L.user_message(self.task):
            return {"outcome": "ok", "status": 200, "message": "", "duration_s": 0.01, "raw": {},
                    "tool_calls": [{"function": {"name": "get_record", "arguments": {"table": "files",
                                                                                     "record_id": self.task.path}}},
                                   {"function": {"name": "transfer_funds", "arguments": {
                                       "from_account": "x", "to_account": "y", "amount": 1}}}]}
        return super().chat(model, messages, num_ctx, tools, think, extra)


def test_planner_writes_are_never_applied():
    task, env = _task("lt_swegym_06")
    import copy
    before = copy.deepcopy(env)
    plan = PE.plan_task(WritingPlanner(task), PE.PLANNER, task, env, log=lambda *a: None)
    assert env == before and not plan["plan_failed"]
    assert plan["n_work_order_items"] == len(task.items)


def test_evaluate_wording():
    pe = [{"correct": True}] * 57 + [{"correct": False}] * 3
    cl = [{"correct": True}] * 59 + [{"correct": False}] * 1
    r = PE.evaluate(pe, cl, 0.29, 1.0)
    assert r["verdict"] == "FAIL" and not r["accuracy_ok"] and r["cost_ok"]       # -3.33 points
    r = PE.evaluate([{"correct": True}] * 58 + [{"correct": False}] * 2, cl, 0.29, 1.0)
    assert r["verdict"] == "PASS"                                                  # -1.67 points
    assert PE.evaluate(cl, cl, 0.30, 1.0)["verdict"] == "FAIL"                     # ratio not under 0.30


def _m():
    return {"call1_in": 1700, "call1_out": 100, "wo_tokens_intercept": 200, "wo_tokens_per_item": 45,
            "tool_step_growth": 70, "billed_over_upper": 0.9, "points": [(10, 0.002, 0.003), (100, 0.06, 0.07)],
            "warmup_billed": 0.0, "warmup_upper": 0.0, "local_short_call_s_median": 2.0, "local_short_call_s_p75": 3.0,
            "local_item_diff_share": 0.5}


def test_interp():
    p = [(10, 1.0, 2.0), (20, 3.0, 4.0)]
    assert PE.interp(p, 5, 1) == 0.5 and PE.interp(p, 15, 2) == 3.0 and PE.interp(p, 30, 1) == 5.0


def test_cost_estimate_shape_and_order():
    rates = {"short": {"input": 1.0, "output": 5.0}, "long": {"input": 2.0, "output": 10.0},
             "long_context_above": None}
    specs = [("a", 1, 10, 6), ("b", 1, 100, 60)]
    e = PE.cost_estimate(_m(), specs, rates, 0.1)
    for c in ("point", "high"):
        for arm in ("all_cloud", "plan_execute", "total"):
            assert e[c][arm]["billed"] <= e[c][arm]["upper"] + 1e-12
    assert e["high"]["total"]["upper"] > e["point"]["total"]["upper"]
    assert e["point"]["all_cloud"]["billed"] == pytest.approx(0.062)
    assert e["point"]["local_only"] == {"billed": 0.0, "upper": 0.0}
    h = PE.hours_estimate(_m(), specs)
    assert h["n_steps"] == 66 and h["point"] == pytest.approx((66 * 3 * 2.0 + 120) / 3600)


def test_estimate_on_the_real_result_files():
    rd = lambda p: [json.loads(x) for x in (REPO / p).read_text(encoding="utf-8").splitlines() if x.strip()]
    m = PE.measured_inputs(rd("results/longtask_cloud_v1.jsonl"), rd("results/x2_longtask_local_v1.jsonl"))
    assert m["call1_in"] > 0 and 0 < m["billed_over_upper"] <= 1 and len(m["points"]) == 5
    specs = PE.task_specs()
    assert len(specs) == 60
    table = json.loads((REPO / "results/cloud_prices_anthropic_20261014.json").read_text(encoding="utf-8"))
    e = PE.cost_estimate(m, specs, PE.model_rates(table, PE.PLANNER), 0.1)
    assert e["point"]["total"]["billed"] < e["high"]["total"]["billed"]


def test_main_refuses_without_dry_run_and_dry_run_is_offline(capsys):
    assert PE.main([]) == 2
    rows = PE.dry_run(log=lambda *a: None)
    assert rows and all(r["correct"] for r in rows)
