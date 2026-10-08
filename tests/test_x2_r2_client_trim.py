"""harness/x2_r2_client_trim.py: the R2 mitigation's client-side trimming (budget, turn grouping, the exact-count
search, the render-trimmed check, the estimate fallback), driven through a real x2_r2_agent session behind the
mechanism wrapper against a fake Ollama that writes Ollama 0.34.4-shaped log lines, and main --mode mitigation with
fakes for host_config, the queue, the runtime and the tokenizer. No network, no process launches."""

import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
import x2_r2_agent as ag  # noqa: E402
import x2_r2_client_trim as ct  # noqa: E402
import x2_r2_mechanism as mech  # noqa: E402

L_TRUNC = ('time=2026-10-08T10:00:00.000Z level=DEBUG source=prompt.go:73 msg="truncating input messages which '
           'exceed context length" truncated={}')
L_COMPLETION = ('time=2026-10-08T10:00:00.000Z level=DEBUG source=llama_server.go:1624 msg="llama-server completion '
                'request" media=0 prompt_len={}')
L_NEWPROMPT = "slot update_slots: id  0 | task 12 | new prompt, n_ctx_slot = {}, n_keep = {}, task.n_tokens = {}"
L_SHIFT = "slot   pre_decode: id  0 | task 12 | slot context shift, n_keep = {}, n_left = {}, n_discard = {}"
L_GIN = '[GIN] 2026/10/08 - 10:00:01 | 200 |  1.234567s |       127.0.0.1 | POST     "/api/chat"'

ARM4K = "ollama_ctx_4096_call2_notools"


# ── pure functions ────────────────────────────────────────────────────────────────────────────────────

def test_parse_client_trim():
    assert ct.parse_client_trim("margin=0.05") == {"margin": 0.05}
    assert ct.parse_client_trim(None) is None and ct.parse_client_trim("") is None
    for bad in ("margin=", "keep=3", "margin=0.7", "margin=-0.1"):
        with pytest.raises(ValueError):
            ct.parse_client_trim(bad)


def test_prompt_budget_matches_preregistration():
    assert ct.prompt_budget(4096, 0.05, 384) == 3507
    assert ct.prompt_budget(8192, 0.05, 384) == 7398
    assert ag.MAX_TOKENS_PER_CALL == 384


def _transcript(n_turns, with_tool=True):
    msgs = [{"role": "system", "content": "SYS", "tok": 100}]
    for t in range(1, n_turns + 1):
        msgs.append({"role": "user", "content": f"U{t}", "tok": 50})
        if with_tool:
            msgs.append({"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "log_event"}}],
                         "tok": 5})
            msgs.append({"role": "tool", "content": f"T{t}", "tok": 5})
        msgs.append({"role": "assistant", "content": f"A{t}", "tok": 10})
    return msgs


def _tok(msgs):
    return sum(m["tok"] for m in msgs)


def test_split_turns_groups_tool_round_trip_with_its_user_message():
    msgs = _transcript(2)
    head, turns = ct.split_turns(msgs)
    assert head == [0] and turns == [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert ct.session_turn_numbers(msgs, turns) == [1, 2]
    # call 1 of turn 3: the current turn is just the user message
    msgs.append({"role": "user", "content": "U3", "tok": 50})
    head, turns = ct.split_turns(msgs)
    assert turns[-1] == [9]


@pytest.mark.parametrize("budget,dropped", [(1000, 0), (360, 0), (359, 1), (289, 2), (219, 3)])
def test_choose_trim_smallest_drop_that_fits(budget, dropped):
    msgs = _transcript(3) + [{"role": "user", "content": "U4", "tok": 50}]
    calls = []

    def count(kept):
        calls.append(len(kept))
        return _tok(kept)

    rec = ct.choose_trim(msgs, budget, count, _tok)
    assert rec["n_turns_dropped"] == dropped == rec["_chosen"]
    assert rec["dropped_turns"] == list(range(1, dropped + 1)) and rec["kept_turns"][-1] == 4
    assert rec["count_method"] == ct.COUNT_RENDER and rec["over_budget"] is False
    assert rec["kept_prompt_tokens_exact"] <= budget
    kept = ct.kept_messages(msgs, *ct.split_turns(msgs), rec["_chosen"])
    assert kept[0]["role"] == "system" and kept[1]["role"] == "user"  # never an orphaned tool/assistant message
    assert kept[-1]["content"] == "U4"


def test_choose_trim_steps_back_when_estimate_overshoots():
    msgs = _transcript(3) + [{"role": "user", "content": "U4", "tok": 50}]
    # estimate says everything is twice as large: first candidate drops too much, exact counts step back
    rec = ct.choose_trim(msgs, 359, _tok, lambda m: 2 * _tok(m))
    assert rec["n_turns_dropped"] == 1 and rec["n_counts"] == 4


def test_choose_trim_render_trimmed_counts_as_over_and_current_turn_always_kept():
    msgs = _transcript(2) + [{"role": "user", "content": "U3", "tok": 50}]
    rec = ct.choose_trim(msgs, 120, lambda kept: None if len(kept) > 2 else _tok(kept), _tok)
    assert rec["n_turns_dropped"] == 2 and rec["kept_turns"] == [3]
    assert rec["over_budget"] is True  # system + current turn = 150 > 120: still sent, flagged
    rec = ct.choose_trim(msgs, 1000, lambda kept: None, _tok)
    assert rec["kept_turns"] == [3] and rec["over_budget"] is True


def test_choose_trim_falls_back_to_estimate_with_safety_when_counting_fails():
    msgs = _transcript(3) + [{"role": "user", "content": "U4", "tok": 50}]

    def boom(kept):
        raise RuntimeError("tokenizer missing")
    rec = ct.choose_trim(msgs, 360, boom, _tok)
    assert rec["count_method"] == ct.COUNT_FALLBACK and "tokenizer missing" in rec["count_error"]
    # 1.25 x estimate must fit: 1 dropped (est 290 -> 362.5 > 360) is not enough, 2 dropped (220 -> 275) is
    assert rec["n_turns_dropped"] == 2 and rec["kept_prompt_tokens_exact"] is None


def test_render_was_trimmed():
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "a"}, {"role": "user", "content": "second question"}]
    assert not ct.render_was_trimmed("S first question a second question", msgs)
    assert ct.render_was_trimmed("S a second question", msgs)
    with pytest.raises(RuntimeError):
        ct.render_was_trimmed("S escaped", msgs)


def test_render_token_counter_uses_render_and_tokenizer(tmp_path):
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "q1"}, {"role": "user", "content": "q2"}]

    class Base:
        def __init__(self, text):
            self.text = text

        def render_only(self, model, messages, num_ctx, tools, think, extra=None):
            return self.text, None

    def run(cmd, stdin=None, capture_output=True, timeout=None):
        n = len(stdin.read().split())
        return types.SimpleNamespace(returncode=0, stdout=f"[{', '.join(['1'] * n)}]\nTotal number of tokens: {n}\n",
                                     stderr="")

    blob = tmp_path / "blob"
    blob.write_bytes(b"gguf")
    c = ct.RenderTokenCounter(Base("S q1 q2 x"), run=run, tmp_dir=tmp_path)
    c._blobs["m"] = blob
    assert c("m", msgs, 4096, None, None) == 4
    c.base = Base("S q2")          # Ollama dropped q1 from the front: over num_ctx
    assert c("m", msgs, 4096, None, None) is None
    assert c.self_check(["m"])["m"]["ok"] is True


# ── through a real session, behind the mechanism wrapper ───────────────────────────────────────────

def _tokens(messages, tools):
    return ag.transcript_tokens(messages) + (ag.est_tokens(json.dumps(tools)) if tools else 0)


class FakeWindowOllama:
    """Drops whole messages from the front while the estimate exceeds num_ctx (system kept), and logs a llama.cpp
    context shift whenever prompt + num_predict would pass num_ctx, as Ollama 0.34.4 + llama-server do."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.n_chat = 0
        self.sent = []
        log_path.write_text('level=INFO msg="server config" env="map[OLLAMA_DEBUG:DEBUG]"\n', encoding="utf-8")

    def _w(self, *lines):
        with open(self.log_path, "a", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")

    def _cut(self, messages, tools, num_ctx):
        idx = 1
        while idx < len(messages) - 1 and _tokens([messages[0]] + messages[idx:], tools) > num_ctx:
            idx += 1
        return idx

    def render_only(self, model, messages, num_ctx, tools, think, extra=None):
        idx = self._cut(messages, tools, num_ctx)
        if idx > 1:
            self._w(L_TRUNC.format(len(messages) - idx))
        self._w(L_GIN)
        return "\n".join(m["content"] for m in [messages[0]] + messages[idx:]), None

    def chat(self, model, messages, num_ctx, tools, think, extra=None):
        self.n_chat += 1
        self.sent.append([dict(m) for m in messages])
        idx = self._cut(messages, tools, num_ctx)
        n = _tokens([messages[0]] + messages[idx:], tools)
        lines = [L_TRUNC.format(len(messages) - idx)] if idx > 1 else []
        lines += [L_COMPLETION.format(999), L_NEWPROMPT.format(num_ctx, 5, n)]
        if n + ag.MAX_TOKENS_PER_CALL > num_ctx:
            lines.append(L_SHIFT.format(5, num_ctx - 5, (num_ctx - 5) // 2))
        lines.append(L_GIN)
        self._w(*lines)
        turn_idx = sum(1 for m in messages if m["role"] == "user")
        if messages[-1]["role"] == "user":
            return {"outcome": "ok", "status": 200, "message": "",
                    "tool_calls": [{"function": {"name": "log_event", "arguments": {"event": f"t{turn_idx}"}}}],
                    "prompt_eval_count": n, "eval_count": 5, "duration_s": 0.1, "raw": {"done_reason": "stop"}}
        return {"outcome": "ok", "status": 200, "message": json.dumps({"answer": "ok", "source": "tool"}),
                "tool_calls": None, "prompt_eval_count": n, "eval_count": 5, "duration_s": 0.1,
                "raw": {"done_reason": "stop"}}

    def loaded_context(self, model):
        return 4096

    def unload(self, model):
        return True

    def recover(self):
        return True


def _run(tmp_path, trim: bool, turns=6):
    log = tmp_path / "serve.log"
    inner = FakeWindowOllama(log)
    rows = []
    mrt = mech.MechanismRuntime(inner, log, rows.append, tmp_path / "logs", sleep=lambda s: None)
    rt = mrt
    mode = "mechanism"
    if trim:
        rt = ct.ClientTrimRuntime(mrt, lambda model, msgs, num_ctx, tools, think, extra=None: _tokens(msgs, tools),
                                  margin=0.05)
        rt.begin_session()
        mode = "mitigation"
    mrt.begin_session("llama3.1:8b", ARM4K, ag.SEEDS[0], mode=mode, call2_mode="off")
    ag.run_session(rt, "llama3.1:8b", ARM4K, ag.SEEDS[0], turns, rows.append, lambda m: None, mode)
    mrt.end_session()
    return rows, inner


def test_without_trim_the_fake_shows_the_4096_mechanism(tmp_path):
    rows, _ = _run(tmp_path, trim=False)
    calls = [r for r in rows if r.get("record") == "r2m_call"]
    assert any(r["context_shift_fired"] for r in calls) and any(r["message_level_truncation"] for r in calls)


def test_client_trim_session_prevents_shifts_and_drops_and_records_everything(tmp_path):
    rows, inner = _run(tmp_path, trim=True)
    turns = [r for r in rows if r.get("record") == "r2a_turn"]
    mcalls = [r for r in rows if r.get("record") == "r2m_call"]
    assert len(turns) == 6 and len(mcalls) == 12 == inner.n_chat
    assert all(r["mode"] == "mitigation" for r in turns)
    # observed: nothing left for Ollama or llama.cpp to cut
    assert not any(r["context_shift_fired"] or r["message_level_truncation"] or r["token_level_cut"] for r in mcalls)
    # r2m_call rows carry the SESSION turn index, not the index within the trimmed list
    assert [r["turn_idx"] for r in mcalls] == [t for t in range(1, 7) for _ in (1, 2)]
    budget = ct.prompt_budget(4096, 0.05, ag.MAX_TOKENS_PER_CALL)
    recs = []
    for r in turns:
        for i, c in enumerate(r["calls"], 1):
            t = c["client_trim"]
            recs.append(t)
            assert c["num_predict_sent"] == ag.MAX_TOKENS_PER_CALL == t["num_predict"]
            assert t["applied"] and t["budget_prompt_tokens"] == budget and t["margin"] == 0.05
            assert t["kept_turns"][-1] == r["turn_idx"]                     # current turn always kept
            assert t["dropped_turns"] == list(range(1, len(t["dropped_turns"]) + 1))  # oldest first, contiguous
            assert t["kept_prompt_tokens_exact"] <= budget and t["over_budget"] is False
            o = t["observed"]
            assert o["context_shifts"] == 0 and not o["message_level_truncation"] and o["log_slice_complete"]
            assert o["client_exact_vs_server_diff"] == 0
    assert any(t["dropped_turns"] for t in recs)                            # trimming did happen
    # what was sent: system first, then a user message, never an orphaned tool or assistant message
    for sent in inner.sent:
        assert sent[0]["role"] == "system" and sent[1]["role"] == "user"
    s = ct.summarize(rows)["llama3.1:8b|" + ARM4K]
    assert s["n_calls"] == 12 and s["context_shift_events"] == 0 and s["ollama_message_drop_calls"] == 0
    assert s["count_method"] == {ct.COUNT_RENDER: 12, ct.COUNT_FALLBACK: 0} and s["over_budget_calls"] == 0
    assert s["client_vs_server_tokens_equal"] == 12


def test_mitigation_plan():
    plan = ag.mitigation_plan("off")
    assert [a for a, _, _ in plan] == ["ollama_ctx_4096_call2_notools", "ollama_ctx_8192_call2_notools"]
    assert all(s == tuple(ag.SEEDS) and n == 40 for _, s, n in plan)
    assert tuple(ag.SEEDS) == (20260901, 20260902, 20260903)


# ── main --mode mitigation ──────────────────────────────────────────────────────────────────────────

def _patch_main(monkeypatch, tmp_path):
    import socket
    log = tmp_path / "serve_mech.log"
    calls = {"start": [], "stop": 0, "notes": []}

    def start(env=None, log_path=None):
        calls["start"].append({"env": env, "log_path": log_path})
        Path(log_path).write_text('level=INFO msg="server config" env="map[OLLAMA_DEBUG:DEBUG]"\n', encoding="utf-8")

    def stop():
        calls["stop"] += 1

    monkeypatch.setattr(socket, "gethostname", lambda: "EVO-X2")
    monkeypatch.setitem(sys.modules, "host_config", types.SimpleNamespace(
        start_ollama_server=start, stop_ollama_server=stop, wait_for_ollama_ready=lambda timeout_s=0: True))
    monkeypatch.setitem(sys.modules, "t2s_queue", types.SimpleNamespace(advance=calls["notes"].append))
    monkeypatch.setattr(mech, "MECH_SERVE_LOG", str(log))

    class RT(FakeWindowOllama):
        def __init__(self):
            self.log_path, self.n_chat, self.sent = log, 0, []
            self.server_env = self.server_log = None

        def available_models(self):
            return ["llama3.1:8b", "qwen3:14b"]

        def version(self):
            return "0.34.4"

    class Counter:
        exe = "fake-llama-tokenize"

        def __init__(self, base):
            self.base = base

        def __call__(self, model, messages, num_ctx, tools, think, extra=None):
            return _tokens(messages, tools)

        def self_check(self, models):
            return {m: {"ok": True} for m in models}

    monkeypatch.setattr(ag, "OllamaRuntime", RT)
    monkeypatch.setattr(ct, "RenderTokenCounter", Counter)
    monkeypatch.setattr(ag, "mitigation_plan", lambda c2: [(ARM4K, (ag.SEEDS[0],), 3)])
    return calls


def test_main_mitigation_runs_debug_server_trims_and_reports(tmp_path, monkeypatch):
    calls = _patch_main(monkeypatch, tmp_path)
    out = tmp_path / "mit.jsonl"
    ag.main(["--mode", "mitigation", "--client-trim", "margin=0.05", "--call2-tools", "off", "--order", "seed_major",
             "--out", str(out)])
    assert calls["notes"] == ["completed"]
    assert calls["start"][0]["env"] == {"OLLAMA_DEBUG": "1"}
    rows = ag.read_rows(out)
    recs = [r["record"] for r in rows]
    assert recs[0] == "run_start" and recs[-1] == "run_end" and "client_trim_selfcheck" in recs
    start = rows[0]
    assert start["mode"] == "mitigation" and start["client_trim"] == {"margin": 0.05}
    assert start["models"] == ["llama3.1:8b", "qwen3:14b"]
    sess = [r for r in rows if r["record"] == "r2a_session"]
    assert {(s["model_id"], s["mode"]) for s in sess} == {("llama3.1:8b", "mitigation"), ("qwen3:14b", "mitigation")}
    assert {r["think"] for r in rows if r["record"] == "r2a_turn" and r["model_id"] == "qwen3:14b"} == {False}
    assert all(c["client_trim"]["applied"] for r in rows if r["record"] == "r2a_turn" for c in r["calls"])
    assert recs.count("r2m_session") == 2
    rep = json.loads(Path(str(out) + ".report.json").read_text(encoding="utf-8"))
    assert set(rep["trim_summary"]) == {"llama3.1:8b|" + ARM4K, "qwen3:14b|" + ARM4K}
    assert all(v["context_shift_events"] == 0 for v in rep["trim_summary"].values())


def test_main_mitigation_needs_client_trim_and_client_trim_needs_mitigation(tmp_path, monkeypatch):
    _patch_main(monkeypatch, tmp_path)
    with pytest.raises(SystemExit):
        ag.main(["--mode", "mitigation", "--out", str(tmp_path / "a.jsonl")])
    with pytest.raises(SystemExit):
        ag.main(["--mode", "real", "--client-trim", "margin=0.05", "--out", str(tmp_path / "b.jsonl")])
