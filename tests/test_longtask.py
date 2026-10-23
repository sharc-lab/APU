"""harness/longtask_env.py and harness/x2_longtask.py with fake runtimes (no server, no network, no queue)."""
from __future__ import annotations

import copy
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))

import agent_trace_replay as R  # noqa: E402
import evalset_env as E  # noqa: E402
import longtask_env as L  # noqa: E402
import x2_longtask as X  # noqa: E402

SEEDS = (20260901, 20260902, 20260903)


def oracle_calls(task, env):
    msgs, _ = L.oracle_transcript(task, env)
    return [m["tool_calls"][0]["function"] for m in msgs if m.get("role") == "assistant" and m.get("tool_calls")]


class FakeRuntime:
    """Replays a list of calls (one per response), then answers in text. drop: indices of calls to skip."""

    def __init__(self, calls, drop=(), never_stop=False, http_error_at=None):
        self.calls, self.drop, self.never_stop, self.http_error_at = list(calls), set(drop), never_stop, http_error_at
        self.i = self.n = 0
        self.seen = []

    def chat(self, model, messages, num_ctx, tools, think, extra=None):
        self.seen.append({"num_ctx": num_ctx, "tools": tools is not None, "n_msgs": len(messages)})
        self.n += 1
        if self.http_error_at is not None and self.n == self.http_error_at:
            return {"outcome": "http_error", "status": 500, "error": "boom", "duration_s": 0.1}
        while self.i in self.drop:
            self.i += 1
        if tools is not None and (self.never_stop or self.i < len(self.calls)):
            c = self.calls[self.i % len(self.calls)] if self.calls else {"name": "get_record", "arguments": {
                "table": "files", "record_id": "/x"}}
            self.i += 1
            return {"outcome": "ok", "status": 200, "message": "", "tool_calls": [{"function": dict(c)}],
                    "prompt_eval_count": 1000, "eval_count": 30, "duration_s": 0.5, "raw": {"done_reason": "stop"}}
        return {"outcome": "ok", "status": 200, "message": "All done.", "tool_calls": None, "prompt_eval_count": 1200,
                "eval_count": 10, "duration_s": 0.4, "raw": {"done_reason": "stop"}}

    def loaded_context(self, model):
        return 40960

    def recover(self):
        return False


# ── task set ────────────────────────────────────────────────────────────────────────────────────────

def test_long_tasks_frozen_from_trace_schedule():
    assert L.LONG_TASKS == L.task_targets(R.load_schedule())
    assert len(L.LONG_TASKS) == 20 and len(set(L.TASK_IDS)) == 20


def test_strata_step_profile():
    sw = [t[2] for t in L.LONG_TASKS if t[1] == "swegym"]
    oh = [t[2] for t in L.LONG_TASKS if t[1] == "openhands"]
    prof = L.trace_profile(R.load_schedule())
    assert prof["openhands"]["median_steps"] == 62 and prof["swegym"]["median_steps"] == 6
    assert abs(statistics.median(oh) - 62) <= 5
    assert abs(statistics.median(sw) - 6) <= 3


@pytest.mark.parametrize("seed", SEEDS)
def test_oracle_passes_grader_and_matches_target(seed):
    for task, env in L.build_tasks(seed):
        msgs, after = L.oracle_transcript(task, env)
        assert L.check_long_task(task, env, after)["end_state_ok"], task.task_id
        assert task.target_steps <= task.ref_steps <= task.target_steps + 1
        assert len(oracle_calls(task, env)) == task.ref_steps
        assert task.step_cap == int(1.5 * task.ref_steps + 0.999999) + 5


def test_tasks_deterministic_and_seed_dependent():
    a = [(t.items, t.reference_writes) for t, _ in L.build_tasks(20260901)]
    b = [(t.items, t.reference_writes) for t, _ in L.build_tasks(20260901)]
    c = [(t.items, t.reference_writes) for t, _ in L.build_tasks(20260902)]
    assert a == b and a != c


def test_offered_tools_identical_to_b3_v2_definitions():
    v2 = {t["name"]: t for t in E.TOOLS_SCHEMA_V2}
    for t in L.tools_schema():
        assert t == v2[t["name"]]
    assert {t["function"]["name"] for t in L.tools_payload()} == set(L.OFFERED_TOOLS)


def test_execute_refuses_tool_not_offered():
    task, env = L.build_tasks(20260901, ["lt_swegym_06"])[0]
    e = copy.deepcopy(env)
    r = L.execute(e, {"name": "log_event", "arguments": {"event": "x"}})
    assert r["status"] == "error" and e == env
    acc = sorted(e["accounts"])[0]
    r = L.execute(e, {"name": "close_account", "arguments": {"account_id": acc}})
    assert r["status"] == "error" and e["accounts"][acc]["status"] == "open"


def test_grader_wrong_write_and_outbox_order():
    task, env = L.build_tasks(20260901, ["lt_openhands_01"])[0]
    _, after = L.oracle_transcript(task, env)
    bad = copy.deepcopy(after)
    t = next(iter(bad["tickets"]))
    bad["tickets"][t]["assignee"] = "nobody"
    assert not L.check_long_task(task, env, bad)["end_state_ok"]
    if len(after["outbox"]) >= 2:
        swapped = copy.deepcopy(after)
        swapped["outbox"].reverse()
        assert L.check_long_task(task, env, swapped)["end_state_ok"]


def test_workspace_has_no_secret_field_and_customer_read_drops_marker():
    env = L.build_long_env(20260901)
    assert not E.secrets_of(env)
    c = sorted(env["customers"])[0]
    r = L.execute(copy.deepcopy(env), {"name": "get_record", "arguments": {"table": "customers", "record_id": c}})
    assert r["status"] == "ok" and "secret_fields" not in r


# ── run_long_task ───────────────────────────────────────────────────────────────────────────────────

def _task(tid="lt_swegym_08", seed=20260901):
    return L.build_tasks(seed, [tid])[0]


def test_run_long_task_oracle_correct():
    task, env = _task()
    rt = FakeRuntime(oracle_calls(task, env))
    row = X.run_long_task(rt, "qwen3:14b", None, False, task, env, lambda *_: None)
    assert row["correct"] and row["outcome"] == "correct" and not row["step_cap_hit"]
    assert row["n_calls"] == task.ref_steps + 1 and row["n_tool_calls"] == task.ref_steps
    assert len(row["call_seconds"]) == row["n_calls"] and row["task_label"] == L.LABEL
    assert all(s["tools"] for s in rt.seen)


def test_run_long_task_skipped_write_is_silent_wrong():
    task, env = _task()
    calls = oracle_calls(task, env)
    w = next(i for i, c in enumerate(calls) if c["name"] not in ("get_record", "search_records"))
    row = X.run_long_task(FakeRuntime(calls, drop={w}), "m", None, False, task, env, lambda *_: None)
    assert not row["correct"] and row["outcome"] == "wrong_silent" and row["silent_wrong"]
    assert row["end_state_diff"]


def test_run_long_task_step_cap_is_surfaced():
    task, env = _task("lt_swegym_01")
    rd = {"name": "get_record", "arguments": {"table": "files", "record_id": task.path}}
    rt = FakeRuntime([rd], never_stop=True)
    row = X.run_long_task(rt, "m", None, False, task, env, lambda *_: None)
    assert row["step_cap_hit"] and row["outcome"] == "wrong_surfaced"
    assert row["n_calls"] == task.step_cap + 1 and rt.seen[-1]["tools"] is False


def test_run_long_task_http_error_is_surfaced():
    task, env = _task()
    row = X.run_long_task(FakeRuntime(oracle_calls(task, env), http_error_at=1), "m", None, False, task, env,
                          lambda *_: None)
    assert row["any_http_error"] and row["outcome"] == "wrong_surfaced"


def test_text_written_calls_are_parsed_and_coerced():
    task, env = _task("lt_swegym_01")
    calls = oracle_calls(task, env)

    class TextRT(FakeRuntime):
        def chat(self, model, messages, num_ctx, tools, think, extra=None):
            r = super().chat(model, messages, num_ctx, tools, think, extra)
            if r.get("tool_calls"):
                f = r["tool_calls"][0]["function"]
                args = {k: (str(v) if isinstance(v, (int, float)) else v) for k, v in f["arguments"].items()}
                r = {**r, "message": json.dumps({"name": f["name"], "arguments": args}), "tool_calls": None}
            return r
    row = X.run_long_task(TextRT(calls), "m", None, False, task, env, lambda *_: None)
    assert row["correct"]


# ── summaries and estimate ──────────────────────────────────────────────────────────────────────────

def _row(arm, seed, tid, correct, outcome, wall=60.0, stratum="swegym"):
    return {"record": "longtask_task", "arm_id": X.ARMS[arm]["arm_id"], "seed": seed, "task_id": tid,
            "stratum": stratum, "correct": correct, "outcome": outcome, "task_wall_s": wall, "n_calls": 5,
            "step_cap_hit": outcome == "wrong_surfaced", "any_http_error": False, "thinking_present": False,
            "max_prompt_tokens_est": 3000, "loaded_context": 40960}


def test_arm_summary_counts():
    rows = [_row("ollama_default", 1, "a", True, "correct", 60), _row("ollama_default", 1, "b", False, "wrong_silent", 120),
            _row("ollama_default", 2, "a", False, "wrong_surfaced", 180, "openhands"),
            _row("kv_q8_0", 1, "a", True, "correct", 30)]
    s = X.summarize(rows)
    d = s["ollama_default"]
    assert (d["n_tasks"], d["correct"], d["wrong"], d["silent_wrong"], d["surfaced_wrong"]) == (3, 1, 2, 1, 1)
    assert d["silent_share_of_wrong"] == 0.5 and abs(d["minutes_per_task_mean"] - 2.0) < 1e-9
    assert d["by_stratum"]["openhands"]["n"] == 1 and s["kv_q8_0"]["accuracy"] == 1.0
    assert len(X.per_task_wall(rows)) == 4


def test_hours_estimate_uses_b5_bands():
    pc = []
    for arm, b5, d in (("ollama_default", "b5_a_default_call2_notools", 10.0), ("kv_q8_0", "b5_g_kv_q8_0_call2_notools", 5.0)):
        for pe in (3000, 12000, 25000):
            pc.append({"record": "r2a_turn", "model_id": "qwen3:14b", "arm_id": b5, "over_loaded_window": False,
                       "calls": [{"prompt_eval_count_info_only": pe, "duration_s": d}]})
    e = X.hours_estimate(pc, seeds=(20260901,), overhead_s_per_arm=0.0)
    calls = sum(t.ref_steps + 1 for t, _ in L.build_tasks(20260901))
    assert abs(e["ollama_default"]["point_h"] - calls * 10.0 / 3600) < 1e-6
    assert abs(e["kv_q8_0"]["high_h"] - 1.5 * calls * 5.0 / 3600) < 1e-6
    assert abs(e["total"]["point_h"] - calls * 15.0 / 3600) < 1e-6


def test_kv_check():
    good = "server config env=map[OLLAMA_FLASH_ATTENTION:true OLLAMA_KV_CACHE_TYPE:q8_0]\nllama_kv_cache: type_k = 'q8_0'"
    assert X.kv_check(good, "q8_0")["valid"]
    assert not X.kv_check(good.replace("q8_0]", "f16]"), "q8_0")["valid"]
    assert not X.kv_check(good.replace("type_k = 'q8_0'", "type_k = 'f16'"), "q8_0")["valid"]
    assert X.kv_check("", None)["valid"]


# ── main with fakes ─────────────────────────────────────────────────────────────────────────────────

class FakeHC:
    def require_host(self, h):
        return {"host": h}

    def enforce_or_record_interactive_session(self, cfg):
        return {"interactive": False}


class FakeServer:
    def __init__(self):
        self.env, self.starts, self.stopped = None, [], 0

    def start(self, env):
        self.env = dict(env)
        self.starts.append(dict(env))

    def stop(self):
        self.stopped += 1

    def log_text(self):
        if self.env and self.env.get("OLLAMA_KV_CACHE_TYPE"):
            return ("server config env=map[OLLAMA_FLASH_ATTENTION:1 OLLAMA_KV_CACHE_TYPE:q8_0]\n"
                    "llama_kv_cache: type_k = 'q8_0'")
        return "server config env=map[]"


class OracleRT(FakeRuntime):
    """Answers each task with its oracle calls (task found from the user message's file path)."""

    def __init__(self, seed, tids):
        super().__init__([])
        self.by_path = {t.path: oracle_calls(t, e) for t, e in L.build_tasks(seed, tids)}
        self.cur = None

    def chat(self, model, messages, num_ctx, tools, think, extra=None):
        if len(messages) == 1:   # warm-up
            return {"outcome": "ok", "status": 200, "message": "OK", "duration_s": 0.1, "raw": {}}
        path = next(p for p in self.by_path if p in messages[1]["content"])
        if path != self.cur:
            self.cur, self.calls, self.i, self.n = path, self.by_path[path], 0, 0
        return super().chat(model, messages, num_ctx, tools, think, extra)


class FakeTQ:
    def __init__(self):
        self.notes = []

    def advance(self, note):
        self.notes.append(note)


def test_main_runs_both_arms_and_advances_once(tmp_path):
    out = tmp_path / "lt.jsonl"
    tids = ["lt_swegym_01", "lt_swegym_06"]
    srv, tq = FakeServer(), FakeTQ()
    note = X.main(["--out", str(out), "--seeds", "20260901", "--tasks", ",".join(tids)],
                  deps={"hc": FakeHC(), "hostname": "EVO-X2", "server": srv, "runtime": OracleRT(20260901, tids),
                        "tq": tq, "log": lambda *_: None})
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    kinds = [r["record"] for r in rows]
    assert kinds[0] == "run_start" and kinds[-1] == "run_end" and note == "completed"
    tr = [r for r in rows if r["record"] == "longtask_task"]
    assert len(tr) == 4 and all(r["correct"] for r in tr)
    assert [r["arm"] for r in tr] == ["ollama_default"] * 2 + ["kv_q8_0"] * 2
    assert tr[0]["num_ctx_requested"] is None and tr[-1]["num_ctx_requested"] == 40960
    assert srv.starts == [{}, X.ARMS["kv_q8_0"]["env"]] and srv.stopped == 1
    assert tq.notes == ["completed"] and len({r["attempt_id"] for r in rows}) == 1
    w = [r for r in rows if r["record"] == "warmup"]
    assert len(w) == 2 and all(x["kv_check"]["valid"] for x in w)
    assert Path(str(out) + ".report.json").exists()
    # a finished file refuses a second run (and still advances the queue once)
    tq2 = FakeTQ()
    note2 = X.main(["--out", str(out), "--seeds", "20260901", "--tasks", tids[0]],
                   deps={"hc": FakeHC(), "hostname": "EVO-X2", "server": FakeServer(),
                         "runtime": OracleRT(20260901, tids), "tq": tq2, "log": lambda *_: None})
    assert note2.startswith("stopped") and len(tq2.notes) == 1


def test_main_skips_arm_when_kv_check_fails(tmp_path):
    class BadKV(FakeServer):
        def log_text(self):
            return "server config env=map[]"
    out = tmp_path / "lt.jsonl"
    tids = ["lt_swegym_01"]
    X.main(["--out", str(out), "--seeds", "20260901", "--tasks", tids[0]],
           deps={"hc": FakeHC(), "hostname": "EVO-X2", "server": BadKV(), "runtime": OracleRT(20260901, tids),
                 "tq": FakeTQ(), "log": lambda *_: None})
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert [r["arm_id"] for r in rows if r["record"] == "arm_skipped"] == ["longtask_kv_q8_0"]
    assert [r["arm"] for r in rows if r["record"] == "longtask_task"] == ["ollama_default"]


def test_main_stop_after_deadline(tmp_path):
    out = tmp_path / "lt.jsonl"
    tq = FakeTQ()
    note = X.main(["--out", str(out), "--seeds", "20260901", "--tasks", "lt_swegym_01",
                   "--stop-after-utc", "2026-10-01T00:00:00+00:00"],
                  deps={"hc": FakeHC(), "hostname": "EVO-X2", "server": FakeServer(),
                        "runtime": OracleRT(20260901, ["lt_swegym_01"]), "tq": tq, "log": lambda *_: None,
                        "now": lambda: datetime(2026, 10, 2, tzinfo=timezone.utc)})
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert not [r for r in rows if r["record"] == "longtask_task"]
    assert "stop-after-utc" in note and rows[-1]["record"] == "run_end" and tq.notes == [note]


def test_main_refuses_wrong_host(tmp_path):
    tq = FakeTQ()
    note = X.main(["--out", str(tmp_path / "x.jsonl")], deps={"hc": FakeHC(), "hostname": "EVO-T2S",
                                                               "server": FakeServer(), "tq": tq,
                                                               "log": lambda *_: None})
    assert note.startswith("stopped") and "EVO-X2" in note and len(tq.notes) == 1


def test_check_plan_no_server():
    p = X.check_plan(seeds=(20260901,))
    assert p["oracle_all_ok"] and p["n_tasks"] == 20 and p["think"] is False
    assert p["arms"]["kv_q8_0"]["num_ctx"] == 40960 and p["arms"]["ollama_default"]["num_ctx"] is None
