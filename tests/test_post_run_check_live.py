"""Live post-run check pieces (harness/post_run_check_live.py, scripts/post_run_check_cloud.py). No server, no network:
fake runtimes and a fake completion function only."""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "harness", REPO / "src", REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import post_run_check_live as PRCL  # noqa: E402

MSGS = [{"role": "system", "content": "RULES"}, {"role": "user", "content": "TURN1"},
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "log_event"}},
                                                            {"function": {"name": "get_record"}}]},
        {"role": "tool", "content": "{}", "tool_name": "log_event"}, {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "TURN2"}]


def test_example_and_l2_row():
    assert PRCL.example_user_content(MSGS) == (1, "TURN1")
    assert not PRCL.l2_row("RULES TURN1 ok TURN2", MSGS)["l2_fail"]
    assert PRCL.l2_row("TURN1 ok TURN2", MSGS)["l2_fail"]            # rules gone
    assert PRCL.l2_row("RULES ok TURN2", MSGS)["l2_fail"]            # example gone
    assert PRCL.l2_row(None, MSGS)["l2_fail"]


def test_turn_l2_citability():
    calls = [{"seed": 1, "turn_idx": 3, "l2_fail": False}, {"seed": 1, "turn_idx": 4, "l2_fail": False}]
    assert not PRCL.turn_l2(calls, [], 1, 3)["fail"]
    checks = [{"seed": 1, "turn_idx": 4, "match": False}]
    assert not PRCL.turn_l2(calls, checks, 1, 3)["fail"]
    assert PRCL.turn_l2(calls, checks, 1, 4)["fail"]
    assert PRCL.turn_l2(calls, [], 1, 9)["fail"]                       # no render row


def test_render_runtime_emits_and_forwards(tmp_path):
    log = tmp_path / "serve.log"
    log.write_text("", encoding="utf-8")
    rows = []

    class Base:
        def render_only(self, model, messages, num_ctx, tools, think, extra=None):
            with open(log, "a", encoding="utf-8") as f:
                f.write('[GIN] 2026/10/31 | 200 | 1ms | 127.0.0.1 | POST "/api/chat"\n')
            return "TURN1 ok TURN2", None

    class Inner:
        def chat(self, model, messages, num_ctx, tools, think):
            return {"outcome": "ok", "prompt_eval_count": 10}

        def drain_turn(self):
            return ["x"]

    rt = PRCL.L2RenderRuntime(Inner(), Base(), str(log), rows.append, checker=None, sleep=lambda s: None)
    rt.begin_session("m", "a", 1, None)
    assert rt.chat("m", MSGS, 8192, None, False)["prompt_eval_count"] == 10
    assert rows[0]["record"] == "prc_l2_call" and rows[0]["l2_fail"] and rows[0]["turn_idx"] == 2
    assert rt.drain_turn() == ["x"]


def _source(tmp_path, flag_turns=(), l2_fail_turns=()):
    rows = [json.loads(x) for x in (REPO / "results" / "x2_evalset_tasks_real_v2.jsonl").read_text(
        encoding="utf-8").splitlines() if x.strip()]
    keep = [r for r in rows if r.get("arm_id") == "ollama_ctx_8192_call2_notools" and r.get("seed") == 20260905
            and r.get("record") in ("r2a_turn", "r2a_session")]
    out = [{"record": "run_start"}]
    for r in keep:
        if r["record"] == "r2a_turn":
            r = dict(r, cut_v2_flagged=r["turn_idx"] in flag_turns, cut_v2_no_log_calls=0, cut_v2_evidence=[])
            out.append(r)
            out.append({"record": "prc_l2_call", "seed": r["seed"], "turn_idx": r["turn_idx"],
                        "l2_fail": r["turn_idx"] in l2_fail_turns})
        else:
            out.append(r)
    out.append({"record": "run_end"})
    p = tmp_path / "local.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in out), encoding="utf-8")
    return p


def test_cloud_phase_dry_run_decides_without_a_client(tmp_path):
    import post_run_check_cloud as PC
    src = _source(tmp_path, flag_turns=(30,), l2_fail_turns=(29,))
    rows = PC.run(src, tmp_path / "out.jsonl", live=False, emit_print=lambda *a: None)
    tasks = [r for r in rows if r["record"] == "prc_live_task"]
    assert len(tasks) == 30
    by = {t["turn_idx"]: t for t in tasks}
    assert "L1" in by[30]["fired"] and "L2" in by[29]["fired"]
    assert all(t.get("cloud", {}).get("outcome") == "not_run_dry" for t in tasks if t["decision"] == "escalate")
    assert rows[-1]["record"] == "run_end"


def test_cloud_phase_refuses_job_cap_over_2(tmp_path):
    import post_run_check_cloud as PC
    with pytest.raises(SystemExit):
        PC.run(_source(tmp_path), tmp_path / "o.jsonl", live=False, job_cap=2.5)


def test_cloud_phase_live_job_cap_stops_cleanly(tmp_path):
    """A fake real client with a job cap below one call: the first escalation is refused by the client's own cap,
    nothing is sent, and every later escalated task is recorded not_run_job_cap."""
    import post_run_check_cloud as PC
    from cloud import client as cc
    sent = []

    def fake(**kw):
        sent.append(1)
        return {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1}}

    client = cc.CloudClient(mode="real", api_key="test-not-a-key", role_override="controller", completion_fn=fake,
                            ledger_path=tmp_path / "ledger.jsonl", spend_cap_usd=0.0001)
    rows = PC.run(_source(tmp_path, flag_turns=tuple(range(1, 31))), tmp_path / "o.jsonl", live=True, client=client,
                  ledger=tmp_path / "ledger.jsonl", emit_print=lambda *a: None, gate_fn=lambda *a, **k: {"gated": True})
    tasks = [r for r in rows if r["record"] == "prc_live_task"]
    assert not sent and rows[-1]["stopped"].startswith("job cap")
    assert all(t["cloud"]["outcome"] == "not_run_job_cap" for t in tasks)
