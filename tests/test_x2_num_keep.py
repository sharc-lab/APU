"""harness/x2_num_keep.py and x2_r2_agent --num-keep system (x2_num_keep_guardrail_test, pre-registered in
docs/FINDINGS.md 2026-10-13): the system-prompt token measurement (render and count), num_keep on every request body,
the observed n_keep per call from Ollama 0.34.4-shaped log lines, the pre-registered verdict, and main() with fakes for
host_config, the queue, the runtime and the tokenizer. No network, no process launches."""

import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
import x2_num_keep as nk  # noqa: E402
import x2_r2_agent as ag  # noqa: E402
import x2_r2_mechanism as mech  # noqa: E402

L_TRUNC = ('time=2026-10-13T10:00:00.000Z level=DEBUG source=prompt.go:73 msg="truncating input messages which '
           'exceed context length" truncated={}')
L_COMPLETION = ('time=2026-10-13T10:00:00.000Z level=DEBUG source=llama_server.go:1624 msg="llama-server completion '
                'request" media=0 prompt_len={}')
L_TOKCUT = ('time=2026-10-13T10:00:00.000Z level=WARN source=llama_server.go:900 msg="truncating input prompt" '
            'limit={} prompt={} keep={} new={}')
L_NEWPROMPT = "slot update_slots: id  0 | task 12 | new prompt, n_ctx_slot = {}, n_keep = {}, task.n_tokens = {}"
L_SHIFT = "slot   pre_decode: id  0 | task 12 | slot context shift, n_keep = {}, n_left = {}, n_discard = {}"
L_START = 'time=2026-10-13T10:00:00.000Z level=INFO msg="starting llama-server" cmd="llama-server.exe --keep {} -c 4096"'
L_GIN = '[GIN] 2026/10/13 - 10:00:01 | 200 |  1.234567s |       127.0.0.1 | POST     "/api/chat"'
ARM4K = "ollama_ctx_4096_call2_notools"


def fake_render(model, messages, num_ctx, tools, think):
    """A llama3.1-like template: system block (with a tool preamble when tools are given), then user turns, then an
    assistant header."""
    sys_c = next((m["content"] for m in messages if m["role"] == "system"), "")
    out = "<|sys|>" + sys_c + ("\n[tools preamble]" if tools else "") + "<|eot|>"
    for m in messages:
        if m["role"] == "user":
            out += "<|user|>" + m["content"] + "<|eot|>"
    return out + "<|assistant|>", None


def count_chars4(text):
    return len(text) // 4 + 1   # +1: BOS


# ── measuring S ─────────────────────────────────────────────────────────────────────────────────────

def test_measure_system_tokens_takes_the_common_prefix_and_the_larger_variant():
    m = nk.measure_system_tokens(fake_render, count_chars4, "llama3.1:8b", "RULES " * 40, "hello turn one", 4096,
                                 None, [{"type": "function"}])
    with_tools = count_chars4("<|sys|>" + "RULES " * 40 + "\n[tools preamble]<|eot|><|")
    without = count_chars4("<|sys|>" + "RULES " * 40 + "<|eot|><|")
    assert m["variants"]["with_tools"]["tokens"] == with_tools
    assert m["variants"]["without_tools"]["tokens"] == without
    assert m["system_prompt_tokens"] == max(with_tools, without) == with_tools


def test_system_block_refuses_a_prefix_without_the_system_prompt():
    with pytest.raises(ValueError):
        nk.system_block("<a>SYSTEM PROMPT", "<b>SYSTEM PROMPT", "SYSTEM PROMPT")
    with pytest.raises(ValueError):
        nk.system_block(None, "x", "x")


def test_measure_refuses_a_failed_render():
    with pytest.raises(ValueError):
        nk.measure_system_tokens(lambda *a: (None, "500 boom"), count_chars4, "m", "S", "U", 4096, None, None)


# ── request bodies ──────────────────────────────────────────────────────────────────────────────────

def test_native_chat_body_merges_options_extra_and_is_unchanged_without_it():
    base = ag.native_chat_body("m", [], 4096, None, None)
    assert base["options"] == {"num_predict": ag.MAX_TOKENS_PER_CALL, "temperature": 0, "seed": 42, "num_ctx": 4096}
    b = ag.native_chat_body("m", [], 4096, None, None, options_extra={"num_keep": 1234})
    assert b["options"] == {**base["options"], "num_keep": 1234}


def test_ollama_runtime_sends_num_keep_on_chat_and_render_only(monkeypatch):
    rt = ag.OllamaRuntime.__new__(ag.OllamaRuntime)
    rt.base = "http://x"
    sent = []

    class Client:
        def chat(self, *a, **k):
            sent.append(("client", k))
            return {"outcome": "ok"}
    rt.client = Client()
    monkeypatch.setattr(rt, "_post", lambda path, body: (sent.append(("post", body)) or (200, {}, None, 0.0)))
    rt.chat("m", [{"role": "user", "content": "x"}], 4096, None, None)
    assert sent[-1][0] == "client"                     # no options_extra: the old client path, byte for byte
    rt.options_extra = {"num_keep": 777}
    rt.chat("m", [{"role": "user", "content": "x"}], 4096, None, None)
    assert sent[-1][0] == "post" and sent[-1][1]["options"]["num_keep"] == 777
    rt.render_only("m", [{"role": "user", "content": "x"}], 4096, None, None)
    assert sent[-1][1]["_debug_render_only"] is True and sent[-1][1]["options"]["num_keep"] == 777


# ── through a real session behind the mechanism wrapper ─────────────────────────────────────────────

class FakeNumKeepOllama:
    """Writes Ollama 0.34.4-shaped log lines. A context shift fires on every other call, with n_keep = num_keep + 1
    (BOS) when num_keep was sent, else 5; the token cut keeps num_keep (else 4). apply=False ignores num_keep (as a
    runtime that drops the option would)."""

    def __init__(self, log_path: Path, apply=True, shifts=True):
        self.log_path, self.apply, self.shifts = log_path, apply, shifts
        self.options_extra = None
        self.bodies = []
        self.n_chat = 0
        log_path.write_text('level=INFO msg="server config" env="map[OLLAMA_DEBUG:DEBUG]"\n', encoding="utf-8")

    def _w(self, *lines):
        with open(self.log_path, "a", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")

    def _keep(self):
        nkp = (self.options_extra or {}).get("num_keep")
        return nkp if (nkp and self.apply) else 4

    def render_only(self, model, messages, num_ctx, tools, think, extra=None):
        self.bodies.append(("render", dict(self.options_extra or {})))
        self._w(L_GIN)
        text, _ = fake_render(model, messages, num_ctx, tools, think)
        return text, None

    def chat(self, model, messages, num_ctx, tools, think, extra=None):
        self.n_chat += 1
        self.bodies.append(("chat", dict(self.options_extra or {})))
        keep = self._keep()
        lines = []
        if self.n_chat == 1:
            lines.append(L_START.format(keep))
        lines += [L_COMPLETION.format(999), L_NEWPROMPT.format(num_ctx, keep + 1, 3000)]
        if self.shifts and self.n_chat % 2 == 0:
            lines.append(L_SHIFT.format(keep + 1, 4000 - keep, (4000 - keep) // 2))
        if self.n_chat == 3:
            lines.append(L_TOKCUT.format(2050, 4097, keep, 2050))
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


def _session_rows(tmp_path, seeds=(20260901,), turns=4, apply=True, shifts=True, riu=("rule1_json_keys",)):
    log = tmp_path / "serve.log"
    base = FakeNumKeepOllama(log, apply=apply, shifts=shifts)
    rows = [{"record": "run_start", "rules_in_use": {"llama3.1:8b": list(riu)}}]
    mrt = mech.MechanismRuntime(base, log, rows.append, tmp_path / "logs", sleep=lambda s: None, checker=None)

    def measure(model, arm, seed):
        ags = ag.build_agent_session(seed, turns)
        return nk.measure_system_tokens(fake_render, count_chars4, model, ags.system_prompt,
                                        ag.user_message(ags.spec.turns[0], ags), 4096, None, [{"t": 1}])
    nkr = nk.NumKeepRuntime(mrt, base, measure, rows.append)
    for seed in seeds:
        mrt.begin_session("llama3.1:8b", ARM4K, seed, mode="real", call2_mode="off")
        nkr.begin_session("llama3.1:8b", ARM4K, seed)
        ag.run_session(nkr, "llama3.1:8b", ARM4K, seed, turns, rows.append, lambda m: None, "real")
        mrt.end_session()
        nkr.end_session()
    rows.append({"record": "run_end"})
    return rows, base


def test_every_request_of_the_session_carries_num_keep_s(tmp_path):
    rows, base = _session_rows(tmp_path)
    s = next(r for r in rows if r["record"] == "num_keep_session")["num_keep"]
    assert s > 0
    # measured with the test's own render; every body the runtime then sends (render-only and chat) carries S
    assert base.bodies and all(o == {"num_keep": s} for _, o in base.bodies)
    assert {k for k, _ in base.bodies} == {"render", "chat"}
    turns = [r for r in rows if r["record"] == "r2a_turn"]
    for t in turns:
        for c in t["calls"]:
            assert c["num_keep"]["sent"] == s == c["num_keep"]["system_prompt_tokens"]
    obs = [c["num_keep"]["observed"] for t in turns for c in t["calls"]]
    assert sum(len(o["shift_n_keep"]) for o in obs) == len(obs) // 2
    assert all(v == s + 1 for o in obs for v in o["shift_n_keep"])
    assert obs[0]["keep_args"] == [s] and obs[2]["token_cut_keep"] == s
    mcalls = [r for r in rows if r["record"] == "r2m_call"]
    assert all(r["num_keep_sent"] == s for r in mcalls)
    assert base.options_extra is None                    # cleared at the session end


def test_verdict_one_line_fix_when_shifts_keep_s_and_rules_hold(tmp_path):
    rows, _ = _session_rows(tmp_path, seeds=(20260901, 20260902), turns=4)
    v = nk.verdict(rows, seeds=(20260901, 20260902), n_turns=4)
    assert v["verdict"] == "one-line fix" and v["valid"]
    assert v["context_shift_events"] == 8 and v["token_cuts"] == 1
    assert v["rule_rates"]["rule1_json_keys"]["rate"] == 1.0 and v["rules_in_use"] == ["rule1_json_keys"]
    assert set(v["system_prompt_tokens_per_seed"]) == {20260901, 20260902}


def test_verdict_invalid_when_the_option_is_not_applied(tmp_path):
    rows, _ = _session_rows(tmp_path, apply=False)
    v = nk.verdict(rows, seeds=(20260901,), n_turns=4)
    assert v["verdict"] == "invalid (num_keep not applied)" and not v["valid"]
    assert v["shift_n_keep_values"] == [5] and v["token_cut_keep_values"] == [4]


def test_verdict_inconclusive_without_an_observed_shift(tmp_path):
    rows, _ = _session_rows(tmp_path, shifts=False)
    v = nk.verdict(rows, seeds=(20260901,), n_turns=4)
    assert v["verdict"] == "inconclusive (flag effect not observed)"


def test_verdict_not_a_fix_when_a_rule_in_use_is_below_90(tmp_path):
    rows, _ = _session_rows(tmp_path, turns=4)
    turns = [r for r in rows if r["record"] == "r2a_turn"]
    turns[0]["rule1_json_keys"] = False                 # 3/4 = 75% < 90%
    v = nk.verdict(rows, seeds=(20260901,), n_turns=4)
    assert v["verdict"] == "not a one-line fix"
    assert v["rule_rates"]["rule1_json_keys"]["first_failure_per_seed"] == {20260901: 1}


def test_verdict_inconclusive_when_incomplete(tmp_path):
    rows, _ = _session_rows(tmp_path, turns=4)
    rows = [r for r in rows if r["record"] != "run_end"]
    v = nk.verdict(rows, seeds=(20260901, 20260902), n_turns=4)
    assert v["verdict"].startswith("inconclusive (incomplete: ") and "no run_end" in v["verdict"]
    assert "expected [20260901, 20260902]" in v["verdict"]


def test_verdict_defaults_are_the_preregistered_cell():
    assert nk.EXPECTED_SEEDS == tuple(ag.SEEDS) == (20260901, 20260902, 20260903)
    assert nk.EXPECTED_TURNS == 40 and nk.EXPECTED_ARM == ARM4K and nk.RULE_THRESHOLD == 0.90


def test_tray_app_running_reads_the_process_count():
    assert nk.tray_app_running(run=lambda *a, **k: types.SimpleNamespace(stdout="1\n")) is True
    assert nk.tray_app_running(run=lambda *a, **k: types.SimpleNamespace(stdout="0\n")) is False


# ── main --mode real --num-keep system ──────────────────────────────────────────────────────────────

JOB_ARGV = ["--mode", "real", "--call2-tools", "off", "--models", "llama3.1:8b", "--tiers", "4096", "--seeds",
            "20260901,20260902,20260903", "--rules-from", "results/x2_r2_validation_v2.jsonl",
            "--require-validation-gates", "--num-keep", "system",
            "--out", "C:\\apu\\ovn\\results\\x2_num_keep_guardrail_test.jsonl"]


def test_job_argv_parses_to_the_preregistered_plan():
    a = ag.build_arg_parser().parse_args(JOB_ARGV)
    assert a.num_keep == "system" and a.mode == "real" and a.call2_tools == "off" and a.require_validation_gates
    plan = ag.restrict_plan(ag.real_plan(a.call2_tools), ag.parse_tiers(a.tiers), ag.parse_seeds(a.seeds))
    assert plan == [(ARM4K, (20260901, 20260902, 20260903), 40)]
    assert ag.ARMS[ARM4K]["num_ctx"] == 4096


def _patch_main(monkeypatch, tmp_path, tray=False):
    import socket
    log = tmp_path / "serve_nk.log"
    calls = {"start": [], "stop": 0, "notes": []}

    def start(env=None, log_path=None):
        calls["start"].append({"env": env, "log_path": log_path})
        Path(log_path).write_text('level=INFO msg="server config" env="map[OLLAMA_DEBUG:DEBUG]"\n', encoding="utf-8")

    def stop():
        calls["stop"] += 1

    monkeypatch.setattr(socket, "gethostname", lambda: "EVO-X2")
    monkeypatch.setitem(sys.modules, "host_config", types.SimpleNamespace(
        start_ollama_server=start, stop_ollama_server=stop, wait_for_ollama_ready=lambda timeout_s=0: True,
        require_host=lambda h: {"hw_id": "evo-x2", "interactive_guard": False},
        enforce_or_record_interactive_session=lambda cfg: {}))
    monkeypatch.setitem(sys.modules, "t2s_queue", types.SimpleNamespace(advance=calls["notes"].append))
    monkeypatch.setattr(ag, "NUM_KEEP_SERVE_LOG", str(log))
    monkeypatch.setattr(nk, "tray_app_running", lambda run=None: tray)
    monkeypatch.setattr(nk, "tokenizer_count_fn", lambda model, **k: count_chars4)

    class RT(FakeNumKeepOllama):
        def __init__(self):
            self.log_path, self.apply, self.shifts = log, True, True
            self.options_extra, self.bodies, self.n_chat = None, [], 0
            self.server_env = self.server_log = None

        def render_only(self, model, messages, num_ctx, tools, think, extra=None):
            return super().render_only(model, messages, num_ctx, tools, think, extra)

        def available_models(self):
            return ["llama3.1:8b"]

        def version(self):
            return "0.34.4"

    monkeypatch.setattr(ag, "OllamaRuntime", RT)
    monkeypatch.setattr(ag, "real_plan", lambda c2: [(ARM4K, (ag.SEEDS[0],), 3)])
    return calls


def test_main_num_keep_runs_debug_server_and_records_everything(tmp_path, monkeypatch):
    calls = _patch_main(monkeypatch, tmp_path)
    out = tmp_path / "nk.jsonl"
    note = ag.main(["--mode", "real", "--call2-tools", "off", "--models", "llama3.1:8b", "--num-keep", "system",
                    "--out", str(out)])
    assert note == "completed" and calls["notes"] == ["completed"]
    assert calls["start"][0]["env"] == {"OLLAMA_DEBUG": "1"} and calls["start"][0]["log_path"] == ag.NUM_KEEP_SERVE_LOG
    rows = ag.read_rows(out)
    recs = [r["record"] for r in rows]
    assert recs[0] == "run_start" and recs[-1] == "run_end" and rows[0]["num_keep"] == "system"
    assert recs.count("num_keep_session") == 1 and recs.count("r2m_session") == 1
    assert "r2m_prompt_check" not in recs                 # no fresh-reload checks in this job
    s = next(r for r in rows if r["record"] == "num_keep_session")["num_keep"]
    assert all(c["num_keep"]["sent"] == s for r in rows if r["record"] == "r2a_turn" for c in r["calls"])
    assert Path(str(out) + ".report.json").exists()


def test_main_num_keep_refuses_while_the_tray_app_runs(tmp_path, monkeypatch):
    calls = _patch_main(monkeypatch, tmp_path, tray=True)
    note = ag.main(["--mode", "real", "--call2-tools", "off", "--models", "llama3.1:8b", "--num-keep", "system",
                    "--out", str(tmp_path / "nk.jsonl")])
    assert note.startswith("stopped:") and "tray app" in note
    assert calls["start"] == [] and calls["stop"] == 0 and calls["notes"] == [note]


def test_num_keep_only_in_real_mode(tmp_path, monkeypatch):
    _patch_main(monkeypatch, tmp_path)
    with pytest.raises(SystemExit):
        ag.main(["--mode", "validation", "--num-keep", "system", "--out", str(tmp_path / "a.jsonl")])
    with pytest.raises(SystemExit):
        ag.main(["--mode", "real", "--num-keep", "system", "--server-env", "OLLAMA_DEBUG=0",
                 "--out", str(tmp_path / "b.jsonl")])


# ── register row ────────────────────────────────────────────────────────────────────────────────────

def test_register_row_computes_from_the_result_file(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
    import numbers_register as nr
    entry = next(e for e in nr.NUMBER_ENTRIES if e["claim_id"] == "num-keep-guardrail-test")
    assert entry["pending_ok"] and entry["data_files"] == [nr.NUM_KEEP_FILE]
    rows, _ = _session_rows(tmp_path, seeds=(20260901, 20260902, 20260903), turns=40)
    f = tmp_path / nr.NUM_KEEP_FILE
    f.parent.mkdir(parents=True)
    f.write_text("\n".join(json.dumps(r, default=str) for r in rows) + "\n", encoding="utf-8")
    r = nr.compute_num_keep_guardrail_test(tmp_path)
    assert r["value"].endswith("verdict: one-line fix") and r["n"] == "120 turns"
    assert "rule1 100.0% (n=120, first failure -/-/-)" in r["value"]
    with pytest.raises(FileNotFoundError):
        nr.compute_num_keep_guardrail_test(tmp_path / "nothing")
