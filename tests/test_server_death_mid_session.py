"""Server death mid-session (docs/METHODOLOGY.md, "Server death mid-session", defined 2026-10-27): a call that fails at
the connection level (no HTTP status) raises x2_r2_agent.SessionIncomplete with no retry and no recover(); the session
gets one incomplete row with the error and no completion row, so no analysis counts it; the run stops without run_end
and the register reads the file as partial. Fake servers only; no network, no process launch, no queue."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO / "analysis"))

import x2_r2_agent as ag  # noqa: E402
import x2_r2_evalset as ev  # noqa: E402

DEAD = {"outcome": "error", "status": None, "error": "connection refused (fake server died)", "duration_s": 0.01}
OK = {"outcome": "ok", "status": 200, "message": "Done.", "tool_calls": None, "prompt_eval_count": 900,
      "eval_count": 3, "duration_s": 0.01, "raw": {"done_reason": "stop", "message": {"content": "Done."}}}


class DyingRuntime:
    """Answers `alive_calls` chat calls, then every call fails at the connection level. Counts calls and recover()."""

    def __init__(self, alive_calls):
        self.alive_calls, self.n_chat, self.n_recover = alive_calls, 0, 0

    def chat(self, model, messages, num_ctx, tools, think, extra=None):
        self.n_chat += 1
        return dict(OK) if self.n_chat <= self.alive_calls else dict(DEAD)

    def loaded_context(self, model):
        return 40960

    def unload(self, model, wait_s=30):
        return True

    def recover(self):
        self.n_recover += 1
        return True


def _check_incomplete(rows, rt, alive_calls):
    assert rt.n_chat == alive_calls + 1, "exactly one failed call: no retry"
    assert rt.n_recover == 0, "recover() is never called"
    inc = [r for r in rows if r.get("record") == "r2a_session_incomplete"]
    assert len(inc) == 1 and inc[0]["incomplete"] is True and inc[0]["retried"] is False
    assert "connection refused" in inc[0]["error"]
    assert not any(r.get("record") == "r2a_session" for r in rows)
    assert ag.completed_sessions(rows) == []
    assert all(c["retried_after_connection_error"] is False
               for r in rows if r.get("record") == "r2a_turn" for c in r["calls"])
    return inc[0]


def test_r2_session_marked_incomplete_without_retry():
    rt, rows = DyingRuntime(alive_calls=3), []
    with pytest.raises(ag.SessionIncomplete):
        ag.run_session(rt, "qwen3:14b", "ollama_ctx_131072_call2_notools", ag.SEEDS[0], 10, rows.append,
                       lambda m: None, "real")
    inc = _check_incomplete(rows, rt, 3)
    turns = [r for r in rows if r.get("record") == "r2a_turn"]
    assert inc["n_turns_completed"] == len(turns) and inc["n_turns_planned"] == 10
    assert inc["attempt_id"] == turns[0]["attempt_id"]


def test_b3_v2_evalset_session_marked_incomplete_without_retry():
    rt, rows = DyingRuntime(alive_calls=2), []
    with pytest.raises(ag.SessionIncomplete):
        ev.run_session(rt, "qwen3:14b", "ollama_ctx_131072_call2_notools", 20260901, 6, rows.append, lambda m: None,
                       "real", "evalset", False, None, "v2")
    inc = _check_incomplete(rows, rt, 2)
    assert inc["harness"] == ev.HARNESS and inc["n_turns_planned"] == 6


def test_http_error_with_status_is_not_a_server_death():
    """A 4xx/5xx with a status is recorded on the call row and scored as before (no exception, session completes)."""
    class Http500(DyingRuntime):
        def chat(self, *a, **k):
            self.n_chat += 1
            return {"outcome": "http_error", "status": 500, "error": "boom", "duration_s": 0.01}
    rt, rows = Http500(0), []
    ag.run_session(rt, "qwen3:14b", "ollama_ctx_131072_call2_notools", ag.SEEDS[0], 2, rows.append, lambda m: None,
                   "real")
    assert [r["record"] for r in rows][-1] == "r2a_session" and rt.n_recover == 0
    assert all(r["any_http_error"] for r in rows if r["record"] == "r2a_turn")


def test_xruntime_llamaserver_death_mid_session(tmp_path):
    import xruntime_llamacpp as X

    class Srv:
        def __init__(self, k, log_path):
            self.k, self.log_path, self.pid, self.url, self.n_ctx = k, str(log_path), 111, "http://fake", 40960
            self.guard = type("G", (), {"check": lambda s: (True, [111])})()
            self.restarts = 0

        def start(self):
            Path(self.log_path).write_text("", encoding="utf-8")
            return {"arm_key": self.k, "started": True, "props_n_ctx": self.n_ctx, "server_pid": self.pid}

        def alive(self):
            return False

        def restart(self):
            self.restarts += 1
            return {"started": True}

        def stop(self):
            return {"after_stop_listeners": []}
    n = {"post": 0}
    holder = {}

    def factory(k, lp):
        holder["s"] = Srv(k, lp)
        return holder["s"]

    def post(url, body):
        n["post"] += 1
        if n["post"] <= 4:
            return 200, {"choices": [{"finish_reason": "stop", "message": {"content": "Done."}}],
                         "usage": {"prompt_tokens": 900, "completion_tokens": 3}}, None, 0.01
        return None, None, "connection refused (fake server died)", 0.01
    rows = []
    with pytest.raises(ag.SessionIncomplete):
        X.run_arm_session("fixed", 20260901, tmp_path / "o.jsonl", rows.append, lambda m: None, server_factory=factory,
                          post=post, settle_s=0)
    assert n["post"] == 5 and holder["s"].restarts == 0
    kinds = [r["record"] for r in rows]
    assert "r2a_session_incomplete" in kinds and "r2a_session" not in kinds and "server_restart" not in kinds
    assert kinds[-1] == "server_stop"   # run_arm_session's finally still stops the server


def test_own_compaction_summarizer_death_is_incomplete():
    import own_compaction as OC
    rt = DyingRuntime(alive_calls=0)
    oc = OC.OwnCompactionRuntime(rt)
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "do a"},
            {"role": "assistant", "content": "done a"}]
    with pytest.raises(ag.SessionIncomplete):
        oc._summarize("qwen3:14b", 40960, False, msgs, [[1, 2]], [1], [0])
    assert rt.n_chat == 1 and rt.n_recover == 0


def test_longtask_main_task_incomplete_and_run_stops(tmp_path):
    import x2_longtask as XL
    sys.path.insert(0, str(REPO / "tests"))
    from test_longtask import FakeHC, FakeServer, FakeTQ, OracleRT

    class DieRT(OracleRT):
        def chat(self, model, messages, num_ctx, tools, think, extra=None):
            self.k = getattr(self, "k", 0) + 1
            if self.k > 3:   # warm-up + 2 task calls, then the server is gone
                return dict(DEAD)
            return super().chat(model, messages, num_ctx, tools, think, extra)
    out, srv, tq = tmp_path / "lt.jsonl", FakeServer(), FakeTQ()
    rt = DieRT(20260901, ["lt_swegym_01"])
    note = XL.main(["--out", str(out), "--seeds", "20260901", "--tasks", "lt_swegym_01"],
                   deps={"hc": FakeHC(), "hostname": "EVO-X2", "server": srv, "runtime": rt, "tq": tq,
                         "log": lambda *_: None})
    rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    kinds = [r["record"] for r in rows]
    assert note.startswith("stopped:") and "run_end" not in kinds and "longtask_task" not in kinds
    inc = [r for r in rows if r["record"] == "longtask_task_incomplete"]
    assert len(inc) == 1 and inc[0]["retried"] is False and inc[0]["task_id"] == "lt_swegym_01"
    assert rt.k == 4 and tq.notes == [note] and srv.stopped == 1


def test_register_reads_the_stopped_run_as_partial_until_resumed(tmp_path):
    import numbers_register as NR
    p = tmp_path / "results" / "r.jsonl"
    p.parent.mkdir()
    rows = [{"record": "run_start", "mode": "real"},
            {"record": "r2a_turn", "attempt_id": "a1"},
            {"record": "r2a_session_incomplete", "attempt_id": "a1", "incomplete": True, "retried": False}]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert NR.run_file_incomplete(tmp_path, "results/r.jsonl") == "partial: no run_end in results/r.jsonl"
    with open(p, "a", encoding="utf-8") as f:   # a resume: new run_start, the session rerun, run_end
        f.write(json.dumps({"record": "run_start", "mode": "real"}) + "\n")
        f.write(json.dumps({"record": "run_end"}) + "\n")
    assert NR.run_file_incomplete(tmp_path, "results/r.jsonl") is None
