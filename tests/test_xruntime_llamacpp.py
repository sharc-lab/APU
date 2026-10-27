"""Cross-runtime replication on llama.cpp server (harness/xruntime_llamacpp.py): the plan and arms, the server command
lines and guard expectations, the log detector and call/turn classes, the response mapping, a session on a fake server,
the guard stop, the pre-registered outcome on fake records, the hours estimate, the queue entry and the register rows.
No network, no server, no queue access."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO / "analysis"))

import xruntime_llamacpp as X  # noqa: E402
import x2_queue_fill as qf  # noqa: E402

ag = X.ag


# ── plan, arms, commands ─────────────────────────────────────────────────────────────────────────────

def test_plan_arms_and_seeds_match_the_b3_v2_core():
    assert X.SEEDS == tuple(ag.SEEDS_STRONG)
    assert [a for a, _s, _t in X.plan()] == [X.ARM[k] for k in ("default", "4096", "fixed")]
    assert all(s == X.SEEDS and t == 30 for _a, s, t in X.plan())
    assert ag.ARMS[X.ARM["default"]]["num_ctx"] is None
    assert ag.ARMS[X.ARM["4096"]]["num_ctx"] == 4096
    assert ag.ARMS[X.ARM["fixed"]]["num_ctx"] == X.N_CTX_TRAIN == 40960
    assert all(ag.ARMS[a]["call2_mode"] == "off" for a in X.ARM.values())
    assert X.MODEL_ID == "qwen3:14b" and ag.MODELS[X.MODEL_ID] is False


def test_server_commands_differ_only_in_the_window():
    d, b, c = (X.server_cmd(k, "L") for k in ("default", "4096", "fixed"))
    assert "-c" not in d and "--ctx-size" not in d
    assert b == d + ["-c", "4096"] and c == d + ["-c", "40960"]
    for cmd in (d, b, c):
        assert not any("context-shift" in t for t in cmd)   # the server default, recorded from --help
        assert cmd[cmd.index("--port") + 1] == str(X.PORT) and cmd[cmd.index("-np") + 1] == "1"
        assert cmd[cmd.index("-m") + 1] == X.GGUF and "--jinja" in cmd
    assert X.server_cmd(None, "L", n_ctx=X.PROBE_CTX)[-2:] == ["-c", str(X.PROBE_CTX)]
    assert "n_ctx" not in X.guard_expected("default")
    assert X.guard_expected("4096")["n_ctx"] == 4096 and X.guard_expected("fixed")["n_ctx"] == 40960
    assert X.guard_expected("default") == {"model_path": X.GGUF, "np": 1, "ngl": 99}


def test_help_lines_take_option_names_only():
    text = ("-c,    --ctx-size N                     size of the prompt context (default: 0, 0 = loaded from model)\n"
            "                                        (env: LLAMA_ARG_CTX_SIZE)\n"
            "-fit,  --fit [on|off]                   whether to adjust unset arguments\n"
            "-fitt, --fit-target MiB                 target margin per device for --fit, comma-separated\n"
            "--context-shift, --no-context-shift     whether to use context shift (default: disabled)\n")
    h = X.help_lines(text)
    assert h["--ctx-size"].endswith("(env: LLAMA_ARG_CTX_SIZE)") and "default: 0" in h["--ctx-size"]
    assert h["--fit"].startswith("-fit,  --fit [on|off]") and "target margin" not in h["--fit"]
    assert "default: disabled" in h["--context-shift"]


# ── detector ─────────────────────────────────────────────────────────────────────────────────────────

def test_classify_log_and_call_class():
    log = X.classify_log("srv  send_error: task id = 3, error: the request exceeds the available context size\n"
                         "slot update_slots: id  0 | task 9 | slot context shift, n_keep = 0, n_left = 4095\n"
                         "srv  log_server_r: request: POST /v1/chat/completions 200\n"
                         "context shift is disabled\n")
    assert len(log["reject"]) == 1 and len(log["shift"]) == 1 and len(log["ctx_stop"]) == 1 and not log["trunc"]
    assert X.classify_log("slot update_slots: input truncated, n_ctx = 4096")["trunc"]
    none = X.classify_log("")
    assert X.call_class(400, None, None, None, 4096, none,
                        '{"error":{"type":"exceed_context_size_error"}}') == "flagged_reject"
    assert X.call_class(400, None, None, None, 4096, none, "bad tools") == "error_other"
    assert X.call_class(None, None, None, None, 4096, none, "connection reset") == "error_other"
    assert X.call_class(200, "stop", 3000, 50, 4096, {"shift": ["slot context shift"]}) == "silent_cut"
    assert X.call_class(200, "stop", 3000, 50, 4096, {"trunc": ["input truncated"]}) == "silent_cut"
    assert X.call_class(200, "length", 3900, 196, 4096, none) == "ctx_stop"
    assert X.call_class(200, "length", 1000, 384, 4096, none) == "none"     # max_tokens, not the window
    assert X.call_class(200, "stop", 1000, 30, 4096, none) == "none"
    assert X.turn_class([{"class": "none"}, {"class": "silent_cut"}, {"class": "flagged_reject"}]) == "flagged_reject"
    assert X.turn_class([{"class": "none"}, {"class": "ctx_stop"}]) == "ctx_stop"
    assert X.turn_class([]) == "none"


def test_detector_validated_needs_both_probes():
    ok = [{"probe": "prompt_over_window", "class": "flagged_reject"},
          {"probe": "generation_over_window", "class": "ctx_stop"}]
    assert X.detector_validated(ok)
    assert not X.detector_validated(ok[:1])
    assert not X.detector_validated([ok[0], {"probe": "generation_over_window", "class": "none"}])
    assert not X.detector_validated([{"probe": "prompt_over_window", "class": "none"}, ok[1]])


def test_response_mapping_matches_the_ollama_shape():
    data = {"choices": [{"finish_reason": "tool_calls", "message": {
        "content": "", "reasoning_content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": '{"key": "a"}'}},
            {"id": "c2", "type": "function", "function": {"name": "x", "arguments": "not json"}}]}}],
        "usage": {"prompt_tokens": 1200, "completion_tokens": 20}}
    r = X.map_response(200, data, None, 1.5)
    assert r["outcome"] == "ok" and r["prompt_eval_count"] == 1200 and r["eval_count"] == 20
    assert r["tool_calls"][0]["function"] == {"name": "lookup", "arguments": {"key": "a"}}
    assert r["tool_calls"][1]["function"]["arguments"] == "not json"
    assert r["raw"]["done_reason"] == "tool_calls" and r["raw"]["message"]["thinking"] == ""
    calls, method = ag.extract_tool_calls(r["message"], r["tool_calls"])
    assert method == "native" and calls[0] == {"name": "lookup", "arguments": {"key": "a"}}
    rec = ag._call_record(r, [{"role": "user", "content": "hi"}], None)
    assert rec["thinking_present"] is False and rec["done_reason"] == "tool_calls"
    err = X.map_response(400, None, "exceeds the available context size", 0.1)
    assert err["outcome"] == "http_error" and err["status"] == 400
    assert X.map_response(None, None, "refused", 0.1)["outcome"] == "error"
    body = X.chat_body([{"role": "user", "content": "q"}], None, False)
    assert body["chat_template_kwargs"] == {"enable_thinking": False} and body["temperature"] == 0
    assert body["max_tokens"] == ag.MAX_TOKENS_PER_CALL and body["seed"] == 42 and "tools" not in body


# ── a session on a fake server ───────────────────────────────────────────────────────────────────────

class FakeGuard:
    def __init__(self, ok=True):
        self.ok = ok

    def check(self):
        return (True, [111]) if self.ok else (False, [222])


class FakeServer:
    guard_ok = True

    def __init__(self, k, log_path):
        self.k, self.log_path, self.pid, self.url = k, str(log_path), 111, "http://fake"
        self.n_ctx = X.BASES[k][1] or 40960
        self.guard = FakeGuard(self.guard_ok)

    def start(self):
        Path(self.log_path).write_text("", encoding="utf-8")
        return {"arm_key": self.k, "started": True, "props_n_ctx": self.n_ctx, "server_pid": self.pid}

    def alive(self):
        return True

    def stop(self):
        return {"after_stop_listeners": []}


def _fake_post(log_path_of, kind):
    def post(url, body):
        with open(log_path_of(), "a", encoding="utf-8") as f:
            if kind == "reject":
                f.write("srv  send_error: error: the request exceeds the available context size\n")
                return 400, None, '{"error":{"type":"exceed_context_size_error"}}', 0.01
            if kind == "shift":
                f.write("slot update_slots: id 0 | slot context shift, n_keep = 0\n")
            return 200, {"choices": [{"finish_reason": "stop", "message": {"content": "Done."}}],
                         "usage": {"prompt_tokens": 900, "completion_tokens": 3}}, None, 0.01
    return post


@pytest.mark.parametrize("arm,kind,want_class,want_http", [
    ("4096", "reject", "flagged_reject", True), ("fixed", "shift", "silent_cut", False),
    ("default", "plain", "none", False)])
def test_session_rows_carry_the_server_log_class(tmp_path, arm, kind, want_class, want_http):
    rows = []
    srv = {}

    def factory(k, log_path):
        srv["s"] = FakeServer(k, log_path)
        return srv["s"]
    res = X.run_arm_session(arm, 20260901, tmp_path / "out.jsonl", rows.append, lambda m: None, server_factory=factory,
                            post=_fake_post(lambda: srv["s"].log_path, kind), settle_s=0)
    assert res == "ok"
    turns = [r for r in rows if r.get("record") == "r2a_turn"]
    assert len(turns) == 30
    assert all(t["context_event"] == want_class and t["any_http_error"] is want_http for t in turns)
    assert all(t["arm_id"] == X.ARM[arm] and t["runtime"] == "llama-server" and t["llamacpp_calls"] for t in turns)
    assert all(t["loaded_context"] == srv["s"].n_ctx for t in turns)
    assert [r["record"] for r in rows if r["record"] in ("server_start", "r2a_session", "server_stop")] == [
        "server_start", "r2a_session", "server_stop"]


def test_guard_mismatch_stops_the_condition(tmp_path):
    rows = []

    class Bad(FakeServer):
        guard_ok = False
    res = X.run_arm_session("4096", 20260901, tmp_path / "o.jsonl", rows.append, lambda m: None, server_factory=Bad,
                            post=_fake_post(lambda: str(tmp_path / "x.log"), "plain"), settle_s=0)
    assert res == "guard_stop"
    assert any(r["record"] == "guard_stop" for r in rows)
    assert not any(r["record"] == "r2a_session" for r in rows) and rows[-1]["record"] == "server_stop"


# ── outcome on fake records ──────────────────────────────────────────────────────────────────────────

def fake_rows(wrong, default_ctx=40960, run_end=True, probes=("flagged_reject", "ctx_stop"), tool_rounds=1):
    rows = [{"record": "run_start", "harness": X.HARNESS}]
    rows += [{"record": "detector_probe", "probe": p, "class": c}
             for p, c in zip(("prompt_over_window", "generation_over_window"), probes)]
    ctx = {"default": default_ctx, "4096": 4096, "fixed": 40960}
    for k in X.ARM_ORDER:
        for i, seed in enumerate(X.SEEDS):
            att = f"{k}{seed}"
            w = wrong[k][i]
            rows.append({"record": "server_start", "arm_key": k, "seed": seed, "started": True,
                         "props_n_ctx": ctx[k]})
            for j in range(30):
                bad = j < w
                ev = "flagged_reject" if (bad and k == "4096") else "none"
                rows.append({"record": "r2a_turn", "attempt_id": att, "item_kind": "task", "model_id": X.MODEL_ID,
                             "arm_id": X.ARM[k], "seed": seed, "turn_idx": j + 1, "end_state_ok": not bad,
                             "any_http_error": ev == "flagged_reject", "context_event": ev,
                             "tool_rounds": tool_rounds})
            rows.append({"record": "r2a_session", "mode": "real", "model_id": X.MODEL_ID, "arm_id": X.ARM[k],
                         "seed": seed, "attempt_id": att})
    if run_end:
        rows.append({"record": "run_end"})
    return rows


def ollama_rows(wrong):
    rows = []
    for i, seed in enumerate(X.SEEDS):
        att = f"o{seed}"
        rows += [{"record": "r2a_turn", "attempt_id": att, "item_kind": "task", "model_id": "qwen3:14b",
                  "arm_id": X.OLLAMA_DEFAULT_ARM, "seed": seed, "end_state_ok": j >= wrong[i]} for j in range(30)]
        rows.append({"record": "r2a_session", "mode": "real", "model_id": "qwen3:14b", "arm_id": X.OLLAMA_DEFAULT_ARM,
                     "seed": seed, "attempt_id": att})
    return rows


W_REP = {"default": [3, 4, 3, 4, 3], "4096": [25, 27, 26, 28, 25], "fixed": [3, 4, 3, 4, 3]}
W_NULL = {"default": [3, 4, 3, 4, 3], "4096": [4, 3, 4, 3, 4], "fixed": [3, 4, 3, 4, 3]}


def test_outcome_replicated_with_flagged_failures():
    o = X.outcome(fake_rows(W_REP), ollama_rows([3, 4, 3, 4, 3]), n_boot=500)
    assert o["valid"] and o["detector_validated"] and o["default_loaded_n_ctx"] == 40960
    assert o["primary"]["comparator"] == "default" and o["primary"]["reading"] == "replicated"
    b = o["arms"]["4096"]
    assert b["wrong"] == 131 and b["n_task_turns"] == 150
    assert b["wrong_by_context_event"]["flagged_reject"] == 131 and b["silent_share"] == 0.0
    assert b["harness_silent_wrong"] == 0 and b["loaded_n_ctx"] == [4096]
    f = o["default_fixed"]
    assert f["persists"]["reading"] == "matches Ollama's default-window rate" and f["persists"]["ollama_sessions"] == 5
    assert f["arm"]["harness_silent_wrong"] == f["arm"]["wrong"] == 17 and f["arm"]["context_event_turns"] == 0


def test_outcome_not_replicated_and_fixed_higher_than_ollama():
    o = X.outcome(fake_rows(W_NULL), ollama_rows([0, 0, 0, 0, 0]), n_boot=500)
    assert o["valid"] and o["primary"]["reading"] == "not replicated"
    assert o["default_fixed"]["persists"]["reading"] == "higher on llama.cpp at the fixed window"


def test_outcome_falls_back_to_c_when_the_default_loads_4096():
    o = X.outcome(fake_rows(W_REP, default_ctx=4096), n_boot=200)
    assert o["primary"]["comparator"] == "fixed" and "fallback" in o["primary"]["comparator_reason"]
    assert o["primary"]["reading"] == "replicated" and o["default_fixed"]["persists"] is None


def test_outcome_invalid_runs_are_not_read():
    o = X.outcome(fake_rows(W_REP, run_end=False), n_boot=100)
    assert not o["valid"] and o["primary"]["reading"].startswith("not read (invalid: no run_end")
    o = X.outcome(fake_rows(W_REP, tool_rounds=0), n_boot=100)
    assert not o["valid"] and any("tool path gate" in p for p in o["problems"])
    o = X.outcome(fake_rows(W_REP, probes=("none", "none")), n_boot=100)
    assert o["valid"] and not o["detector_validated"] and o["arms"]["4096"]["silent_share"] is None
    rows = fake_rows(W_REP)
    rows.insert(-1, {"record": "guard_stop", "arm_key": "4096"})
    assert not X.outcome(rows, n_boot=100)["valid"]
    rows = [r for r in fake_rows(W_REP) if not (r.get("record") == "r2a_session" and r["seed"] == 20260905
                                                and r["arm_id"] == X.ARM["fixed"])]
    o = X.outcome(rows, n_boot=100)
    assert not o["valid"] and any(p.startswith("fixed: sessions completed") for p in o["problems"])


# ── hours, queue, register ───────────────────────────────────────────────────────────────────────────

def test_hours_estimate_from_fake_rows():
    b3 = ([{"record": "r2a_turn", "model_id": "qwen3:14b", "arm_id": X.OLLAMA_DEFAULT_ARM, "turn_wall_s": 10.0}] * 4
          + [{"record": "r2a_turn", "model_id": "qwen3:14b", "arm_id": "ollama_ctx_4096" + X.SFX, "turn_wall_s": 20.0}])
    ot = [{"record": "outcome_row", "model_id": "qwen3-14b", "item_id": "a", "config": "ollama_default",
           "http_status": 200, "latency_s": 10.0},
          {"record": "outcome_row", "model_id": "qwen3-14b", "item_id": "a", "config": "llama_server",
           "http_status": 200, "latency_s": 12.0}]
    sp = [{"record": "sweep_point", "started": True, "requested_n_ctx": 1, "wall_s": 50.0},
          {"record": "sweep_call", "ok": True, "requested_n_ctx": 1, "latency_ms": 20000.0}]
    h = X.hours_estimate(b3, ot, sp)
    assert h["latency_ratio"] == pytest.approx(1.2) and h["start_overhead_s"] == pytest.approx(30.0)
    assert h["session_h"] == pytest.approx(1.2 * 150 * 40 / 3600)
    assert h["point_h"] == pytest.approx(h["session_h"] + 16 * 30 / 3600 + X.PROBE_S / 3600)
    assert h["high_h"] == pytest.approx(h["point_h"] * X.HIGH_FACTOR)


def test_queue_entry():
    e = qf.entry("x2_xruntime_llamacpp")
    assert e["cmd"][1] == "xrt\\xruntime_llamacpp.py" and e["cmd"][-1].endswith("xruntime_llamacpp_qwen3_14b.jsonl")
    assert e["projected_duration_h"] == qf.XRT_HOURS[0]
    assert "tq.advance()" in e["note"] and "sets no pause" in e["note"]


def test_register_rows_pending_and_hours_match_the_queue(tmp_path):
    import numbers_register as nr
    ent = {e["claim_id"]: e for e in nr.NUMBER_ENTRIES}
    for c in ("xruntime-llamacpp-wrong-rate", "xruntime-default-fixed", "xruntime-openvino-ceiling"):
        assert ent[c]["pending_ok"]
        assert nr._pending(ent[c], tmp_path)
    assert ent["xruntime-llamacpp-wrong-rate"]["require_run_end"] == [nr.XRT_FILE]
    v = nr.compute_xruntime_llamacpp_hours(REPO)["value"]
    assert v.startswith(f"ESTIMATE {qf.XRT_HOURS[0]:.2f} h (high {qf.XRT_HOURS[1]:.2f} h)")


def test_register_result_rows_compute_from_a_fake_file(tmp_path):
    import numbers_register as nr
    (tmp_path / "results").mkdir()
    with open(tmp_path / nr.XRT_FILE, "w", encoding="utf-8") as f:
        for r in fake_rows(W_REP):
            f.write(json.dumps(r) + "\n")
    with open(tmp_path / nr.X2_EVALSET_TASKS_REAL_V2, "w", encoding="utf-8") as f:
        for r in ollama_rows([3, 4, 3, 4, 3]):
            f.write(json.dumps(r) + "\n")
    v = nr.compute_xruntime_llamacpp_wrong_rate(tmp_path)["value"]
    assert "valid True" in v and "reading: replicated" in v and "flagged_reject 131" in v
    v = nr.compute_xruntime_default_fixed(tmp_path)["value"]
    assert "reading: matches Ollama's default-window rate" in v and "context-event turns at (c) 0/150" in v
