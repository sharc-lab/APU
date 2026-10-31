"""Per-task cloud budget (src/cloud/task_budget.py, operator item C): a fake client in a forced retry loop. No network.

test_forced_retry_loop_stops_at_cap also writes results/per_task_budget_check_v1.jsonl (deterministic content, no
timestamps), the file row per-task-budget-check reads.
"""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from src.cloud import client as cc  # noqa: E402
from src.cloud import task_budget as TB  # noqa: E402

MODEL = "claude-sonnet-5-5"
RESULT = REPO / "results" / "per_task_budget_check_v1.jsonl"
MESSAGES = [{"role": "user", "content": "word " * 2000}]


def _resp(p=2600, o=200):
    return {"choices": [{"message": {"content": "not done yet", "tool_calls": None}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": p, "completion_tokens": o, "cache_read_input_tokens": 0,
                      "cache_creation_input_tokens": 0}}


def _client(tmp_path, fn, cap=cc.HARD_SPEND_CAP_USD):
    return cc.CloudClient(mode="real", api_key="test-not-a-key", role_override="controller", completion_fn=fn,
                          ledger_path=tmp_path / "ledger.jsonl", spend_cap_usd=cap)


def _ledger(tmp_path):
    p = tmp_path / "ledger.jsonl"
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def test_forced_retry_loop_stops_at_cap(tmp_path):
    sent = []

    def fake(*, model_id, messages, api_key, **kw):
        sent.append(model_id)
        return _resp()

    inner = _client(tmp_path, fake)
    kw = {"max_tokens": 1024}
    reservation = inner.projected_cost_usd(model_id=MODEL, messages=MESSAGES, **kw)
    cap = round(3.5 * reservation, 6)          # room for 3 calls at the reservation, not a 4th
    lane = TB.BudgetedClient(inner, cap)
    lane.start_task("seed:task")
    calls, outcome, refusal = [], None, None
    for attempt in range(50):                  # forced retry loop: the answer never satisfies the caller
        try:
            res = lane.call(model_id=MODEL, messages=MESSAGES, **kw)
        except TB.TaskBudgetExhausted as e:
            outcome, refusal = TB.OUTCOME_EXHAUSTED, e.record
            break
        calls.append({"call_idx": attempt, "reservation_usd": round(reservation, 8),
                      "billed_usd": round(res.cost_usd, 8), "upper_bound_usd": round(res.cost_usd_upper_bound, 8),
                      "task_counted_usd": round(lane.task_spent_usd, 8)})
    assert outcome == TB.OUTCOME_EXHAUSTED
    assert len(sent) == 3 and len(calls) == 3                 # stopped at the 4th call, before sending it
    assert lane.task_spent_usd <= cap + 1e-12                 # nothing past the cap
    assert sum(c["billed_usd"] for c in calls) <= cap
    with pytest.raises(TB.TaskBudgetExhausted):               # a later retry is refused again, nothing sent
        lane.call(model_id=MODEL, messages=MESSAGES, **kw)
    assert len(sent) == 3
    led = _ledger(tmp_path)
    assert sum(1 for r in led if r.get("record") == TB.LEDGER_RECORD) == 1
    assert sum(1 for r in led if r.get("record") != TB.LEDGER_RECORD) == 3   # the client's own call rows
    assert inner._reserved_usd == 0.0
    lane.start_task("seed:task2")                             # the next task gets its own budget
    lane.call(model_id=MODEL, messages=MESSAGES, **kw)
    assert len(sent) == 4
    rows = [{"record": "run_start", "harness": "tests/test_task_budget.py::test_forced_retry_loop_stops_at_cap",
             "client": "src/cloud/client.py CloudClient(mode=real) with a fake completion_fn (no network)",
             "wrapper": "src/cloud/task_budget.py BudgetedClient", "model_id": MODEL, "max_tokens": 1024,
             "task_cap_usd": cap, "reservation_usd": round(reservation, 8)},
            *({"record": "budget_call", **c} for c in calls),
            {"record": "budget_refusal", "outcome": outcome, "refused_call_idx": len(calls),
             "task_counted_usd_at_refusal": round(refusal["task_spent_usd"], 8),
             "next_call_reservation_usd": round(refusal["next_call_reservation_usd"], 8),
             "ledger_records_written": 1, "provider_requests_sent": 3},
            {"record": "run_end", "stopped_at_expected_call": len(calls) == 3,
             "task1_counted_usd": round(refusal["task_spent_usd"], 8),
             "task1_counted_within_cap": refusal["task_spent_usd"] <= cap + 1e-12,
             "billed_within_cap": sum(c["billed_usd"] for c in calls) <= cap,
             "retry_after_refusal_sent": False, "outcome_recorded": True, "next_task_fresh_budget": True}]
    out = tmp_path / RESULT.name               # never the committed file
    out.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8", newline="\n")
    assert out.read_bytes() == RESULT.read_bytes().replace(b"\r\n", b"\n"), "result differs from the committed file"


PINNED_SHA256 = "3d9c8495d27527480b86c6c0dc8707901d58daa3b86a451fed1436add3f159c6"   # committed content, LF ends


def test_committed_result_file_unchanged():
    """The committed results/per_task_budget_check_v1.jsonl is never written by tests (checked again at session end
    in tests/conftest.py)."""
    import hashlib
    got = hashlib.sha256(RESULT.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    assert got == PINNED_SHA256, "results/per_task_budget_check_v1.jsonl was modified"


def test_per_task_check_runs_before_job_cap_and_never_loosens_it(tmp_path):
    def fake(*, model_id, messages, api_key, **kw):
        return _resp()

    inner = _client(tmp_path, fake, cap=0.0001)               # job cap below one call
    lane = TB.BudgetedClient(inner, 10.0)                      # a generous per-task cap cannot lift it
    lane.start_task("t")
    with pytest.raises(cc.SpendCapExceeded):
        lane.call(model_id=MODEL, messages=MESSAGES, max_tokens=1024)
    inner2 = _client(tmp_path, fake)
    lane2 = TB.BudgetedClient(inner2, 1e-6)                    # per-task cap below one call: refused first
    lane2.start_task("t")
    with pytest.raises(TB.TaskBudgetExhausted):
        lane2.call(model_id=MODEL, messages=MESSAGES, max_tokens=1024)
    assert inner2._reserved_usd == 0.0 and inner2.running_total_usd == 0.0


def test_needs_task_and_positive_cap(tmp_path):
    inner = _client(tmp_path, lambda **k: _resp())
    with pytest.raises(ValueError):
        TB.BudgetedClient(inner, 0)
    with pytest.raises(RuntimeError):
        TB.BudgetedClient(inner, 1.0).prepare(model_id=MODEL, messages=MESSAGES)


def test_cloud_redo_live_turn_tool_loop_hits_cap(tmp_path):
    """The cloud_redo lane: a model that keeps calling tools is stopped by the per-task cap mid-task."""
    sys.path.insert(0, str(REPO / "scripts"))
    import cloud_redo as CR
    src = CR.load_rows(CR.SOURCE)
    r = CR.wrong_tasks(src)[0]
    es = CR.build_session_for(r, {})
    prior = [x for x in CR.session_rows(src, r["attempt_id"]) if x["turn_idx"] < r["turn_idx"]]
    built = CR.rebuild(es, prior, r["turn_idx"])
    n = []

    def fake(*, model_id, messages, api_key, **kw):
        n.append(1)
        tc = [{"id": f"t{len(n)}", "type": "function",
               "function": {"name": "log_event", "arguments": json.dumps({"event": "again"})}}]
        return {"choices": [{"message": {"content": "", "tool_calls": tc}, "finish_reason": "tool_use"}],
                "usage": {"prompt_tokens": 9000, "completion_tokens": 20, "cache_read_input_tokens": 0,
                          "cache_creation_input_tokens": 0}}

    inner = _client(tmp_path, fake)
    model = CR.escalation_model(CR.load_price_table())
    lane = TB.BudgetedClient(inner, 0.05)
    lane.start_task("k")
    rates = CR.model_rates(CR.load_price_table(), model)
    with pytest.raises(TB.TaskBudgetExhausted):
        CR.live_turn(lane, model, es, built, rates, 2048, CR.THINKING_OFF.get(model))
    assert 1 <= len(n) < CR.MAX_TOOL_ROUNDS + 1 and lane.task_spent_usd <= 0.05
