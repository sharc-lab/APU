"""harness/x2_r2_mechanism.py: log parsing, render analysis, verdicts, and the per-call wrapper driven through a real
x2_r2_agent session against a fake Ollama that writes Ollama 0.34.4-shaped log lines to a temp file. No network, no
process launches."""

import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
import x2_r2_agent as ag  # noqa: E402
import x2_r2_mechanism as mech  # noqa: E402

# Lines in the shapes Ollama 0.34.4 (slog text handler, Gin) and llama.cpp b11081's server (--no-log-prefix) write.
L_TRUNC = ('time=2026-10-08T10:00:00.000Z level=DEBUG source=prompt.go:73 msg="truncating input messages which '
           'exceed context length" truncated={}')
L_TRUNC_NATIVE = ('time=2026-10-08T10:00:00.000Z level=DEBUG source=routes.go:3104 msg="truncating native chat '
                  'messages which exceed context length" truncated={}')
L_COMPLETION = ('time=2026-10-08T10:00:00.000Z level=DEBUG source=llama_server.go:1624 msg="llama-server completion '
                'request" media=0 prompt_len={}')
L_CHATREQ = ('time=2026-10-08T10:00:00.000Z level=DEBUG source=llama_server.go:1992 msg="llama-server chat request" '
             'messages={} tools={}')
L_TOKCUT = ('time=2026-10-08T10:00:00.000Z level=WARN source=llama_server.go:320 msg="truncating input prompt" '
            'limit={} prompt={} keep={} new={}')
L_NEWPROMPT = "slot update_slots: id  0 | task 12 | new prompt, n_ctx_slot = {}, n_keep = {}, task.n_tokens = {}"
L_SHIFT = "slot   pre_decode: id  0 | task 12 | slot context shift, n_keep = {}, n_left = {}, n_discard = {}"
L_GIN = '[GIN] 2026/10/08 - 10:00:01 | 200 |  1.234567s |       127.0.0.1 | POST     "/api/chat"'
L_EXCEED = ("srv    send_error: task id = 3, error: request (5000 tokens) exceeds the available context size "
            "(4096 tokens), try increasing it")


def test_parse_call_log_rendered_path():
    text = "\n".join([L_TRUNC.format(7), L_COMPLETION.format(15000), L_TOKCUT.format(2050, 5000, 4, 2050),
                      L_NEWPROMPT.format(4096, 4, 2050), L_SHIFT.format(5, 4000, 2000), L_GIN,
                      "unrelated line"])
    ev = mech.parse_call_log(text)
    assert ev["chat_path"] == "rendered" and ev["msg_truncation_value"] == 7
    assert ev["completion_prompt_len"] == 15000
    assert ev["token_cut"] == {"limit": 2050, "prompt": 5000, "keep": 4, "new": 2050}
    assert ev["new_prompt"] == [{"n_ctx_slot": 4096, "n_keep": 4, "n_tokens": 2050}]
    assert ev["context_shifts"] == [{"n_keep": 5, "n_left": 4000, "n_discard": 2000}]
    assert ev["gin_status"] == [[200, "/api/chat"]]
    assert "unrelated line" not in ev["excerpt"] and len(ev["excerpt"]) == 6


def test_parse_call_log_native_path_and_exceed():
    ev = mech.parse_call_log("\n".join([L_TRUNC_NATIVE.format(5), L_CHATREQ.format(9, 2), L_EXCEED]))
    assert ev["chat_path"] == "native" and ev["msg_truncation_value"] == 5
    assert ev["chat_request_messages"] == 9 and ev["chat_request_tools"] == 2
    assert ev["exceed_context_error"] and ev["token_cut"] is None


def _msgs(n_turns):
    msgs = [{"role": "system", "content": "SYSTEM PROMPT rules"}]
    for t in range(1, n_turns + 1):
        msgs.append({"role": "user", "content": f"user task for turn {t} with filler {t * 7919}"})
        msgs.append({"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "log_event"}}]})
        msgs.append({"role": "tool", "content": '{"status": "ok"}'})
        msgs.append({"role": "assistant", "content": f'{{"answer": "turn {t} done"}}'})
    return msgs


def _render(msgs, first_kept):
    kept = [m for m in msgs if m["role"] == "system"] + msgs[first_kept:]
    return "\n".join((m.get("content") or json.dumps(m.get("tool_calls"))) for m in kept)


def test_first_kept_index_both_paths():
    assert mech.first_kept_index({"chat_path": "rendered", "msg_truncation_value": 7}, 20) == 13
    assert mech.first_kept_index({"chat_path": "native", "msg_truncation_value": 5}, 20) == 5
    assert mech.first_kept_index({"chat_path": "rendered", "msg_truncation_value": None}, 20) is None


def test_analyse_render_drop_old_turns_keep_system():
    msgs = _msgs(4) + [{"role": "user", "content": "user task for turn 5 with filler 39595"}]
    idx = 1 + 4 * 2  # turns 1 and 2 dropped whole
    ra = mech.analyse_render(_render(msgs, idx), msgs)
    assert ra["system_kept"] and ra["dropped_turns"] == [1, 2] and ra["first_kept_turn"] == 3
    assert ra["kept_is_contiguous_tail"] and ra["current_user_message_kept"] and ra["user_messages_unique"]
    ev = {"chat_path": "rendered", "msg_truncation_value": len(msgs) - idx, "token_cut": None, "context_shifts": []}
    v = mech.call_verdict(ra, ev, msgs)
    assert v["message_level_truncation"] and v["cut_at_turn_boundary_log"] is True
    assert v["first_kept_message_turn_log"] == 3 and v["n_messages_dropped_log"] == 8
    assert v["render_and_log_agree"] is True and v["token_level_cut"] is False


def test_cut_mid_turn_and_system_lost():
    msgs = _msgs(3) + [{"role": "user", "content": "user task for turn 4 with filler 31676"}]
    idx = 1 + 4 + 2  # inside turn 2: its user message and tool call dropped, its tool result kept
    rendered = "\n".join(m.get("content") or "" for m in msgs[idx:])  # system prompt NOT in the text
    ra = mech.analyse_render(rendered, msgs)
    ev = {"chat_path": "rendered", "msg_truncation_value": len(msgs) - idx}
    v = mech.call_verdict(ra, ev, msgs)
    assert v["cut_at_turn_boundary_log"] is False and v["first_kept_message_role_log"] == "tool"
    assert ra["dropped_turns"] == [1, 2] and v["render_and_log_agree"] is True
    assert ra["system_kept"] is False


def test_verdict_cases():
    base = {"n_calls": 10, "n_calls_message_truncated": 3, "system_kept_on_every_truncated_call": True,
            "kept_contiguous_tail_on_every_truncated_call": True, "token_level_cut_calls": 0,
            "context_shift_calls": 1, "exceed_context_error_calls": 0, "render_log_disagreements": 0,
            "log_slices_incomplete": 0}
    assert mech.verdict([base])["drop_old_turns_keep_system"] == "confirmed"
    assert mech.verdict([base])["context_shift_seen"] is True
    assert mech.verdict([{**base, "token_level_cut_calls": 1}])["drop_old_turns_keep_system"] == "refuted"
    assert mech.verdict([{**base, "system_kept_on_every_truncated_call": False}])[
        "drop_old_turns_keep_system"] == "refuted"
    assert mech.verdict([{**base, "n_calls_message_truncated": 0}])["drop_old_turns_keep_system"] == "not_observed"


def test_debug_enabled():
    assert mech.debug_enabled('level=INFO source=routes.go msg="server config" env="map[OLLAMA_DEBUG:DEBUG]"')
    assert not mech.debug_enabled('level=INFO msg="server config" env="map[OLLAMA_DEBUG:INFO]"')


# ── wrapper driven through a real session ─────────────────────────────────────────────────────────────

class FakeLoggingOllama:
    """Keeps at most `keep_msgs` non-system messages (whole messages, system always kept), as Ollama's chat
    truncation does, writes the corresponding log lines, and answers like a compliant model."""

    def __init__(self, log_path: Path, keep_msgs=9, token_cut=False, shift_every=0):
        self.log_path, self.keep_msgs, self.token_cut, self.shift_every = log_path, keep_msgs, token_cut, shift_every
        self.n_chat = 0
        self.renders = 0
        log_path.write_text('level=INFO msg="server config" env="map[OLLAMA_DEBUG:DEBUG]"\n', encoding="utf-8")

    def _w(self, *lines):
        with open(self.log_path, "a", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")

    def _cut(self, messages):
        non_sys = [i for i, m in enumerate(messages) if m["role"] != "system"]
        if len(non_sys) <= self.keep_msgs:
            return 0
        return non_sys[-self.keep_msgs]

    def render_only(self, model, messages, num_ctx, tools, think, extra=None):
        self.renders += 1
        idx = self._cut(messages)
        if idx:
            self._w(L_TRUNC.format(len(messages) - idx))
        self._w(L_GIN)
        return _render(messages, idx if idx else 1), None

    def chat(self, model, messages, num_ctx, tools, think, extra=None):
        self.n_chat += 1
        idx = self._cut(messages)
        lines = []
        if idx:
            lines.append(L_TRUNC.format(len(messages) - idx))
        lines.append(L_COMPLETION.format(12345))
        if self.token_cut and idx:
            lines.append(L_TOKCUT.format(2050, 5000, 4, 2050))
        lines.append(L_NEWPROMPT.format(4096, 4, 3000))
        if self.shift_every and self.n_chat % self.shift_every == 0:
            lines.append(L_SHIFT.format(5, 4000, 2000))
        lines.append(L_GIN)
        self._w(*lines)
        turn_idx = sum(1 for m in messages if m["role"] == "user")
        if messages[-1]["role"] == "user":
            return {"outcome": "ok", "status": 200, "message": "",
                    "tool_calls": [{"function": {"name": "log_event", "arguments": {"event": f"t{turn_idx}"}}}],
                    "prompt_eval_count": 100, "eval_count": 5, "duration_s": 0.1, "raw": {"done_reason": "stop"}}
        return {"outcome": "ok", "status": 200, "message": json.dumps({"answer": "ok", "source": "tool"}),
                "tool_calls": None, "prompt_eval_count": 100, "eval_count": 5, "duration_s": 0.1,
                "raw": {"done_reason": "stop"}}

    def loaded_context(self, model):
        return 4096

    def unload(self, model):
        return True

    def recover(self):
        return True


def _run_mech(tmp_path, **kw):
    log = tmp_path / "serve.log"
    inner = FakeLoggingOllama(log, **kw)
    rows = []
    mrt = mech.MechanismRuntime(inner, log, rows.append, tmp_path / "logs", sleep=lambda s: None)
    mrt.begin_session("llama3.1:8b", "ollama_ctx_4096_call2_notools", ag.SEEDS[0], call2_mode="off")
    ag.run_session(mrt, "llama3.1:8b", "ollama_ctx_4096_call2_notools", ag.SEEDS[0], 6, rows.append,
                   lambda m: None, "mechanism")
    sess = mrt.end_session()
    return rows, sess, inner


def test_wrapper_session_confirms_drop_old_turns(tmp_path):
    rows, sess, inner = _run_mech(tmp_path, keep_msgs=9, shift_every=4)
    calls = [r for r in rows if r.get("record") == "r2m_call"]
    assert len(calls) == inner.n_chat == inner.renders == 12  # 6 turns x 2 calls, one render each
    assert all(r["log_slice_complete"] for r in calls)
    assert all(r["call2_mode"] == "off" for r in calls)
    first = next(r for r in calls if r["message_level_truncation"])
    assert first["turn_idx"] == 3 and first["system_kept"] and first["kept_is_contiguous_tail"]
    assert first["render_and_log_agree"] is True and first["token_level_cut"] is False
    assert first["log_excerpt"] and first["rendered_prompt_file"]
    assert (tmp_path / "logs" / first["rendered_prompt_file"]).exists()
    # each call's slice holds its own lines only: exactly one completion request per real call
    assert all(r["completion_prompt_len"] == 12345 for r in calls)
    assert sess["record"] == "r2m_session" and sess["first_truncated_turn"] == 3
    assert sess["context_shift_calls"] == 3 and sess["token_level_cut_calls"] == 0
    raw = (tmp_path / "logs" / sess["raw_log_file"]).read_text(encoding="utf-8")
    assert raw.count('msg="llama-server completion request"') == 12
    rep = mech.report(rows)
    assert rep["verdict"]["drop_old_turns_keep_system"] == "confirmed" and rep["verdict"]["context_shift_seen"]
    turns = [r for r in rows if r.get("record") == "r2a_turn"]
    assert len(turns) == 6 and all(r["mode"] == "mechanism" and r["call2_mode"] == "off" for r in turns)


def test_wrapper_session_token_cut_refutes(tmp_path):
    rows, sess, _ = _run_mech(tmp_path, keep_msgs=9, token_cut=True)
    assert sess["token_level_cut_calls"] > 0
    assert mech.report(rows)["verdict"]["drop_old_turns_keep_system"] == "refuted"


def test_wrapper_no_truncation_not_observed(tmp_path):
    rows, sess, _ = _run_mech(tmp_path, keep_msgs=1000)
    assert sess["n_calls_message_truncated"] == 0 and sess["first_truncated_turn"] is None
    assert mech.report(rows)["verdict"]["drop_old_turns_keep_system"] == "not_observed"


def test_mechanism_plan_and_model():
    plan = ag.mechanism_plan("off")
    assert [a for a, _, _ in plan] == [t + "_call2_notools" for t in ag.STRONG_TIERS]
    assert all(s == (ag.SEEDS[0],) and n == 40 for _, s, n in plan)
    assert ag.MECH_MODEL == "llama3.1:8b"
    assert ag.ARMS["ollama_ctx_16384_call2_notools"]["num_ctx"] == 16384
    assert ag.ARMS["ollama_ctx_8192_call2_notools"]["num_ctx"] == 8192


# ── main --mode mechanism with fakes for host_config, the queue and the runtime ──────────────────────

def _patch_main(monkeypatch, tmp_path, debug=True):
    import socket
    log = tmp_path / "serve_mech.log"
    calls = {"start": [], "stop": 0, "notes": []}

    def start(env=None, log_path=None):
        calls["start"].append({"env": env, "log_path": log_path})
        Path(log_path).write_text(
            f'level=INFO msg="server config" env="map[OLLAMA_DEBUG:{"DEBUG" if debug else "INFO"}]"\n',
            encoding="utf-8")

    def stop():
        calls["stop"] += 1

    monkeypatch.setattr(socket, "gethostname", lambda: "EVO-X2")
    monkeypatch.setitem(sys.modules, "host_config", types.SimpleNamespace(
        start_ollama_server=start, stop_ollama_server=stop, wait_for_ollama_ready=lambda timeout_s=0: True,
        require_host=lambda h: {"hw_id": "evo-x2", "interactive_guard": False},
        enforce_or_record_interactive_session=lambda cfg: {}))
    monkeypatch.setitem(sys.modules, "t2s_queue", types.SimpleNamespace(advance=calls["notes"].append))
    monkeypatch.setattr(mech, "MECH_SERVE_LOG", str(log))

    class RT(FakeLoggingOllama):
        def __init__(self):
            self.log_path, self.keep_msgs, self.token_cut, self.shift_every = log, 9, False, 0
            self.n_chat = self.renders = 0
            self.server_env = self.server_log = None

        def available_models(self):
            return ["llama3.1:8b"]

        def version(self):
            return "0.34.4"

    monkeypatch.setattr(ag, "OllamaRuntime", RT)
    monkeypatch.setattr(ag, "mechanism_plan", lambda c2: [("ollama_ctx_4096_call2_notools", (ag.SEEDS[0],), 4),
                                                          ("ollama_default_call2_notools", (ag.SEEDS[0],), 2)])
    return calls


def test_main_mechanism_starts_debug_server_and_reports(tmp_path, monkeypatch):
    calls = _patch_main(monkeypatch, tmp_path)
    out = tmp_path / "mech.jsonl"
    ag.main(["--mode", "mechanism", "--call2-tools", "off", "--out", str(out)])
    assert calls["notes"] == ["completed"]
    assert calls["start"][0]["env"] == {"OLLAMA_DEBUG": "1"} and calls["start"][0]["log_path"].endswith("serve_mech.log")
    assert calls["stop"] >= 2  # before the debug start (env must take effect) and in finally
    rows = ag.read_rows(out)
    recs = [r["record"] for r in rows]
    assert recs[0] == "run_start" and recs[-1] == "run_end" and recs.count("r2m_session") == 2
    assert all("call2_mode" in r for r in rows if r["record"] in ("run_start", "runtime", "heartbeat", "r2a_turn",
                                                                    "r2a_session", "run_end", "r2m_call",
                                                                    "r2m_session"))
    assert {r["call2_mode"] for r in rows if r["record"] in ("r2m_call", "r2a_turn")} == {"off"}
    rep = json.loads(Path(str(out) + ".report.json").read_text(encoding="utf-8"))
    assert rep["verdict"]["drop_old_turns_keep_system"] == "confirmed"
    assert (tmp_path / "mech_logs").is_dir()


def test_main_mechanism_refuses_without_debug(tmp_path, monkeypatch):
    calls = _patch_main(monkeypatch, tmp_path, debug=False)
    out = tmp_path / "mech.jsonl"
    ag.main(["--mode", "mechanism", "--call2-tools", "off", "--out", str(out)])
    assert len(calls["notes"]) == 1 and calls["notes"][0].startswith("stopped:") and "OLLAMA_DEBUG" in calls["notes"][0]
    assert not any(r["record"] == "r2m_call" for r in ag.read_rows(out))
