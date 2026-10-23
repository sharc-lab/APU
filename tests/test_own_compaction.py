"""own-compaction (harness/own_compaction.py) and its x2_r2_evalset integration (--plan own_compaction_v1,
--plan own_config_probe_v1): pure helpers, detector v2 on log lines in the formats x2_r2_mechanism parses, the context
manager on a fake runtime, a full B3 v2 session through PolicySessionRuntime, outcome / verdict, the hours estimate
on the committed files. Fake runtimes only; no network, no process launches, no queue."""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO / "tests"))

import own_compaction as oc  # noqa: E402
import x2_r2_agent as ag  # noqa: E402
import x2_r2_evalset as ev  # noqa: E402

M = "qwen3:14b"
SFX = "_call2_notools"


def _msgs(n_turns, example_at=1, filler=400):
    """system + n_turns turns (user, assistant tool call, tool results, final answer); the turn example_at calls
    log_event and get_record, the others get_record only."""
    out = [{"role": "system", "content": "S" * 800}]
    for t in range(n_turns):
        out.append({"role": "user", "content": f"task {t} " + "f" * filler})
        names = ["log_event", "get_record"] if t == example_at else ["get_record"]
        out.append({"role": "assistant", "content": "", "tool_calls": [{"function": {"name": n, "arguments": {}}}
                                                                        for n in names]})
        out += [{"role": "tool", "content": "{}", "tool_name": n} for n in names]
        out.append({"role": "assistant", "content": f"done {t}"})
    return out


# ── pure helpers ─────────────────────────────────────────────────────────────────────────────────────

def test_example_is_the_first_turn_with_log_event_and_a_task_tool():
    m = _msgs(5, example_at=2)
    head, turns = oc.ct.split_turns(m)
    assert oc.choose_example(m, turns) == 2
    assert oc.choose_example(_msgs(3, example_at=9), oc.ct.split_turns(_msgs(3, example_at=9))[1]) is None
    assert not oc.is_example_turn(["log_event"]) and oc.is_example_turn(["get_record", "log_event"])


def test_assemble_keeps_chronology_and_places_the_summary_after_the_older_example():
    m = _msgs(6, example_at=0)
    head, turns = oc.ct.split_turns(m)
    sent, info = oc.assemble(m, head, turns, 5, 0, [1, 2, 3], "SUMMARY", "STANDING")
    users = [x["content"].split(" ")[0] + x["content"].split(" ")[1] if x["content"].startswith("task") else "PAIR"
             for x in sent if x["role"] == "user"]
    assert users == ["task0", "PAIR", "task4", "task5"]
    assert info["example_sent"] and info["summary_sent"] and not info["standing_pinned"]   # turn 0 still verbatim
    pair = [x for x in sent if x["role"] == "user" and x["content"].startswith("[")][0]
    assert "SUMMARY" in pair["content"] and "STANDING" not in pair["content"]
    # example later than the folded turns: chronological, standing pinned since turn 0 is folded
    m = _msgs(6, example_at=3)
    head, turns = oc.ct.split_turns(m)
    sent, info = oc.assemble(m, head, turns, 5, 3, [0, 1], "SUMMARY", "STANDING")
    users = [x["content"][:6] for x in sent if x["role"] == "user"]
    assert users[0].startswith("[Stand") and users[1:] == ["task 2", "task 3", "task 4", "task 5"]
    assert info["standing_pinned"] and sent[0]["role"] == "system"


def test_assemble_without_compaction_is_the_full_transcript():
    m = _msgs(4, example_at=1)
    head, turns = oc.ct.split_turns(m)
    sent, info = oc.assemble(m, head, turns, 3, 1, [], None, "STANDING")
    assert sent == m and info["example_sent"]


# ── detector v2 ──────────────────────────────────────────────────────────────────────────────────────

DROP = 'time=x level=DEBUG msg="truncating input messages which exceed context length" truncated=3'
KEEP_ALL = 'time=x level=DEBUG msg="truncating input messages which exceed context length" truncated=5'
SHIFT = "slot update_slots: id 0 | task 9 | slot context shift, n_keep = 5, n_left = 4090, n_discard = 2045"
CUT = 'level=WARN msg="truncating input prompt" limit=4096 prompt=5000 keep=4 new=3000'
NEWP = "slot launch_slot_: new prompt, n_ctx_slot = 4096, n_keep = 0, task.n_tokens = 3760"


def test_detector_v2_flags_only_the_server_cut_lines():
    assert oc.cut_v2_verdict(KEEP_ALL + "\n" + NEWP, 5)["flagged"] is False
    v = oc.cut_v2_verdict(KEEP_ALL + "\n" + NEWP, 5)
    assert v["used_tokens"] == 3760 and v["n_ctx_slot"] == 4096
    assert oc.cut_v2_verdict(DROP, 5)["flagged"] is True
    assert oc.cut_v2_verdict(SHIFT, 5)["evidence"] == ["context shift x1"]
    assert oc.cut_v2_verdict(CUT, 5)["flagged"] is True
    assert oc.cut_v2_verdict("", 5)["flagged"] is False
    assert oc.cut_v2_verdict(None, 5)["flagged"] is None


class LogFake:
    """Fake base runtime that appends lines to a log file on each chat (simulating the server)."""

    def __init__(self, log, lines_for=None, used=None):
        self.log, self.lines_for, self.used, self.calls = log, lines_for or (lambda m: []), used, []

    def chat(self, model, messages, num_ctx, tools, think, extra=None):
        self.calls.append({"messages": [dict(m) for m in messages], "num_ctx": num_ctx, "tools": tools})
        lines = list(self.lines_for(messages))
        if self.used:
            lines.append(f"new prompt, n_ctx_slot = {num_ctx}, n_keep = 0, task.n_tokens = {self.used(messages)}")
        lines.append('[GIN] 2026/10/23 | 200 | 1s | 127.0.0.1 | POST "/api/chat"')
        with open(self.log, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        sysm = messages[0]["content"]
        text = "record: things happened" if sysm == oc.SUMMARIZER_SYSTEM else "ok"
        return {"outcome": "ok", "status": 200, "message": text, "tool_calls": None, "prompt_eval_count": 100,
                "eval_count": 10, "duration_s": 0.1, "raw": {}}


def test_cut_log_runtime_reads_the_call_slice(tmp_path):
    log = tmp_path / "serve.log"
    log.write_text("startup OLLAMA_DEBUG:DEBUG\n" + SHIFT + "\n", encoding="utf-8")   # before the call: not counted
    rt = oc.CutLogRuntime(LogFake(log), log, sleep=lambda s: None)
    r = rt.chat(M, [{"role": "user", "content": "x"}], 4096, None, False)
    assert r["cut_v2"]["flagged"] is False and r["cut_v2"]["log_slice_complete"]
    rt2 = oc.CutLogRuntime(LogFake(log, lines_for=lambda m: [DROP]), log, sleep=lambda s: None)
    assert rt2.chat(M, [{"role": "user", "content": "x"}] * 5, 4096, None, False)["cut_v2"]["flagged"] is True


# ── the context manager ──────────────────────────────────────────────────────────────────────────────

def test_compaction_triggers_summarizes_and_keeps_the_example(tmp_path):
    log = tmp_path / "serve.log"
    log.write_text("", encoding="utf-8")
    base = LogFake(log)
    rt = oc.OwnCompactionRuntime(oc.CutLogRuntime(base, log, sleep=lambda s: None), standing="STANDING")
    m = _msgs(12, example_at=1)   # about 12 x 130 + 200 estimated tokens x 1.2
    r = rt.chat(M, m, 1600, None, False)
    rec = r["own_compaction"]
    assert rec["triggered"] and rec["summary_calls"] and rec["example_sent"] and rec["summary_sent"]
    assert rec["proj_sent"] <= rec["trigger_tokens"] and rec["proj_full"] > rec["trigger_tokens"]
    assert rec["standing_pinned"] and 2 not in rec["folded_turns"] and 1 in rec["folded_turns"]
    sent = base.calls[-1]["messages"]
    assert sent[0] == m[0] and sent[-1] == m[-1]
    assert any(x.get("tool_name") == "log_event" for x in sent)   # the example's tool results are sent
    s_calls = [c for c in base.calls if c["messages"][0]["content"] == oc.SUMMARIZER_SYSTEM]
    assert s_calls and all(c["tools"] is None for c in s_calls)
    # the next call reuses the summary without a new summarizer call while under the trigger
    n = len(base.calls)
    r2 = rt.chat(M, m + [{"role": "user", "content": "next " + "f" * 50}], 1600, None, False)
    assert not r2["own_compaction"]["summary_calls"] and len(base.calls) == n + 1


def test_floor_over_trigger_sends_system_and_current_turn_only(tmp_path):
    log = tmp_path / "serve.log"
    log.write_text("", encoding="utf-8")
    base = LogFake(log)
    rt = oc.OwnCompactionRuntime(oc.CutLogRuntime(base, log, sleep=lambda s: None))
    m = _msgs(4, example_at=1, filler=2000)
    r = rt.chat(M, m, 800, None, False)
    rec = r["own_compaction"]
    assert rec["floor_over_trigger"] and not rec["summary_calls"] and rec["over_trigger_sent"]
    sent = base.calls[-1]["messages"]
    assert sent[0] == m[0] and [x for x in sent if x["role"] == "user"] == [[x for x in m if x["role"] == "user"][-1]]


def test_ratio_calibrates_from_the_server_total(tmp_path):
    log = tmp_path / "serve.log"
    log.write_text("", encoding="utf-8")
    base = LogFake(log, used=lambda msgs: round(ag.transcript_tokens(msgs) * 1.5))
    rt = oc.OwnCompactionRuntime(oc.CutLogRuntime(base, log, sleep=lambda s: None))
    rt.chat(M, _msgs(1), 100000, None, False)
    assert rt.ratio == pytest.approx(1.5, rel=0.02)


# ── full B3 v2 sessions through PolicySessionRuntime ─────────────────────────────────────────────────

class SessionFake:
    """A perfect B3 v2 model that finds its turn from the last real user message (the sent list may be compacted)."""

    def __init__(self, es, log, loaded):
        self.es, self.log, self.loaded, self.calls = es, log, loaded, []
        self.by_text = {ev.user_message(t, es): t.idx for t in es.turns}
        from test_x2_r2_evalset import perfect
        self.policy = perfect(es)

    def chat(self, model, messages, num_ctx, tools, think, extra=None):
        self.calls.append(len(messages))
        with open(self.log, "a", encoding="utf-8") as f:
            f.write('[GIN] | 200 | POST "/api/chat"\n')
        if messages[0]["content"] == oc.SUMMARIZER_SYSTEM:
            return {"outcome": "ok", "status": 200, "message": "record", "tool_calls": None, "prompt_eval_count": 50,
                    "eval_count": 5, "duration_s": 0.1, "raw": {}}
        users = [m for m in messages if m["role"] == "user" and m["content"] in self.by_text]
        turn_idx = self.by_text[users[-1]["content"]]
        call_no = 1 if messages[-1]["role"] == "user" else 2
        content, native = self.policy(call_no, messages, turn_idx)
        return {"outcome": "ok", "status": 200, "message": content, "tool_calls": native, "prompt_eval_count": None,
                "eval_count": 20, "duration_s": 0.1, "raw": {"done_reason": "stop", "message": {"content": content}}}

    def loaded_context(self, model):
        return self.loaded

    def unload(self, model):
        return True

    def recover(self):
        return True

    def render_only(self, *a, **k):
        raise RuntimeError("no render in the fake")   # trim falls back to the estimate, visibly


@pytest.mark.parametrize("arm,policy", [("own_compaction_8192", "own_compaction"), ("client_trim_4096", "client_trim")])
def test_full_session_rows_carry_policy_and_detector_fields(tmp_path, arm, policy):
    log = tmp_path / "serve.log"
    log.write_text("OLLAMA_DEBUG:DEBUG\n", encoding="utf-8")
    es = ev.build_session(20260901, 30, "evalset", False, M, design="v2")
    rows = []
    n = ag.ARMS[arm + SFX]["num_ctx"]
    rt = oc.PolicySessionRuntime(SessionFake(es, log, n), log, rows.append, ev.ctx_policy_of,
                                 counter_factory=lambda b: (lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))),
                                 sleep=lambda s: None)
    rt.cut.gin_wait_s = 0.0
    ev.run_session(rt, M, arm + SFX, 20260901, 30, rows.append, lambda m: None, "real", "evalset", False, None, "v2")
    turns = [r for r in rows if r.get("record") == "r2a_turn"]
    assert len(turns) == 30 and all(r["ctx_policy"] == policy for r in turns)
    assert all(r["cut_v2_flagged"] is False for r in turns)
    assert any(r["record"] == "ctx_policy_session" and r["policy"] == policy for r in rows)
    if policy == "own_compaction":
        assert sum(r["n_summary_calls"] for r in turns) > 0
        assert all(r["sent_final_prompt_proj"] <= int(0.85 * n) for r in turns[1:])
    out = oc.outcome(rows)
    assert out[arm]["n"] == 30 and out[arm]["silent_wrong"] == out[arm]["wrong"]


def test_outcome_and_verdict_split_flagged_and_silent():
    rows = []
    for arm in ("own_compaction_4096", "client_trim_4096"):
        for i in range(10):
            rows.append({"record": "r2a_turn", "item_kind": "task", "arm_id": arm + SFX, "seed": 1, "attempt_id": "a",
                         "end_state_ok": i >= (1 if arm.startswith("own") else 3), "cut_v2_flagged": i == 0,
                         "loaded_context": 4096})
        rows.append({"record": "r2a_session", "arm_id": arm + SFX, "seed": 1, "attempt_id": "a"})
    rows.append({"record": "r2a_turn", "item_kind": "task", "arm_id": "own_compaction_4096" + SFX, "seed": 2,
                 "attempt_id": "b", "end_state_ok": False})   # unfinished session: not counted
    out = oc.outcome(rows)
    c = out["own_compaction_4096"]
    assert (c["n"], c["wrong"], c["flagged_wrong"], c["silent_wrong"]) == (10, 1, 1, 0)
    assert out["client_trim_4096"]["silent_wrong"] == 2
    v = oc.verdict(out)
    assert v[4096]["meets_target"] and v[4096]["beats_trim"] and v[4096]["beats_numkeep"]
    assert v[8192] == {"status": "missing"}


# ── plans and CLI ────────────────────────────────────────────────────────────────────────────────────

def test_plans_and_subsets():
    p = ev.ctx_policy_plan(ev.OWN_COMPACTION_PLAN)
    assert [a for a, _, _ in p] == [a + SFX for a in ev.OWN_COMPACTION_ARMS]
    assert all(s == ev.OWN_COMPACTION_SEEDS and t == 30 for _, s, t in p)
    p = ev.ctx_policy_plan(ev.OWN_COMPACTION_PLAN, "own_compaction_8192", "20260901,20260902")
    assert p == [("own_compaction_8192" + SFX, (20260901, 20260902), 30)]
    with pytest.raises(ValueError):
        ev.ctx_policy_plan(ev.OWN_COMPACTION_PLAN, "ollama_ctx_4096")
    with pytest.raises(ValueError):
        ev.ctx_policy_plan(ev.OWN_CONFIG_PLAN, "ollama_ctx_15360,ollama_ctx_28928")
    assert ev.ctx_policy_plan(ev.OWN_CONFIG_PLAN, "ollama_ctx_15360")[0][1] == ev.OWN_CONFIG_SEEDS
    assert ag.ARMS["ollama_ctx_28928" + SFX]["num_ctx"] == 28928
    assert ev.ctx_policy_of("client_trim_8192" + SFX) == "client_trim" and ev.ctx_policy_of("ollama_ctx_15360") is None


def _base_cmd(plan, extra=()):
    return ["--design", "v2", "--mode", "real", "--workload", "evalset", "--guardrails", "off", "--models", M,
            "--plan", plan, "--preflight", "v2_te_deviation", "--rules-from", "v.jsonl", "--out", "o.jsonl",
            *extra]


def test_cli_checks(capsys):
    with pytest.raises(SystemExit):
        ev.main(_base_cmd(ev.OWN_CONFIG_PLAN, ["--tiers", "ollama_ctx_15360", "--check-plan"]), advance=False)
    assert ev.main(_base_cmd(ev.OWN_CONFIG_PLAN, ["--tiers", "ollama_ctx_15360", "--server-env",
                                                  "OLLAMA_KV_CACHE_TYPE=q8_0", "--server-env",
                                                  "OLLAMA_FLASH_ATTENTION=1", "--check-plan"]), advance=False) == "checked"
    assert ev.main(_base_cmd(ev.OWN_COMPACTION_PLAN, ["--check-plan"]), advance=False) == "checked"
    assert '"own_compaction_8192_call2_notools"' in capsys.readouterr().out
    with pytest.raises(SystemExit):   # plan without its preflight
        ev.main([a for a in _base_cmd(ev.OWN_COMPACTION_PLAN, ["--check-plan"]) if a != "v2_te_deviation"
                 and a != "--preflight"], advance=False)


# ── hours ESTIMATE ───────────────────────────────────────────────────────────────────────────────────

def test_hours_estimate_on_the_committed_files():
    b3 = ag.read_rows(REPO / "results/x2_evalset_tasks_real_v2.jsonl")
    mit = ag.read_rows(REPO / "results/x2_r2_mitigation_v1.jsonl")
    e = oc.hours_estimate(b3, mit)
    for a in ev.OWN_COMPACTION_ARMS:
        assert 0 < e[a]["point_h"] <= e[a]["high_h"] < 5
    assert set(e["_trim_overhead"]) >= {4096, 8192}
