"""Dry-run tests for harness/t2s_k1_ollama.py. No real Ollama server, no real llama-server process, no SSH: every
external dependency (Ollama HTTP client, `ollama ps`, the Ollama log, the occupying llama-server) is a fake object
passed in through the functions' dependency-injection parameters."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
import t2s_k1_ollama as K  # noqa: E402


# ---------------------------------------------------------------------------------------------------- fakes
class FakeOllama:
    def __init__(self, responses):
        """responses: list of dicts popped in call order, or a single dict reused for every call."""
        self.responses = responses
        self.calls = []

    def chat(self, model, prompt, num_ctx=None, max_tokens=64):
        self.calls.append({"model": model, "num_ctx": num_ctx, "max_tokens": max_tokens, "prompt_len": len(prompt)})
        if isinstance(self.responses, list):
            return self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        return self.responses


class FakeServer:
    def __init__(self, ok=True, load_s=5.0, tokens=1000, chat_outcome="ok", output="answer"):
        self.ok, self.load_s, self._tokens, self.chat_outcome, self.output = ok, load_s, tokens, chat_outcome, output
        self.stopped = False
        self.start_info = {}

    def start(self):
        self.start_info = {"ok": self.ok, "load_s": self.load_s, "error": None if self.ok else "boom"}
        return self.start_info

    def tokenize(self, text):
        return self._tokens

    def chat(self, prompt, max_tokens, ignore_eos):
        if self.chat_outcome != "ok":
            return {"outcome": self.chat_outcome, "error": "context size exceeded", "output": None}
        return {"outcome": "ok", "output": self.output}

    def stop(self):
        self.stopped = True


def make_lab(tmp_path, host="evo-t2s", resume=None):
    host_cfg = K.require_host(host)
    args = argparse.Namespace(resume=resume, out_dir=str(tmp_path), ollama_port=11434)
    prov = {"git_head": "deadbeef", "committed": True, "problems": [], "files": []}
    return K.K1Lab(args, host_cfg, prov)


# ---------------------------------------------------------------------------------------------------- require_host
def test_require_host_known_and_unknown():
    cfg = K.require_host("evo-t2s")
    assert cfg["gpu_vendor"] == "intel"
    cfg2 = K.require_host("evo-x2")
    assert cfg2["gpu_vendor"] == "amd"
    try:
        K.require_host("not-a-real-host")
        assert False, "expected SystemExit"
    except SystemExit:
        pass


# ---------------------------------------------------------------------------------------------------- ollama ps
def test_parse_ollama_ps_with_context_column():
    text = (
        "NAME          ID              SIZE      PROCESSOR    CONTEXT    UNTIL\n"
        "qwen3:8b      abc123def456    5.2 GB    100% GPU     65536      5 minutes from now\n"
    )
    rows = K.parse_ollama_ps(text)
    assert len(rows) == 1
    assert rows[0]["name"] == "qwen3:8b"
    assert rows[0]["context_len"] == 65536


def test_parse_ollama_ps_without_context_column_returns_none_context():
    text = (
        "NAME          ID              SIZE      PROCESSOR    UNTIL\n"
        "qwen3:8b      abc123def456    5.2 GB    100% GPU     5 minutes from now\n"
    )
    rows = K.parse_ollama_ps(text)
    assert rows[0]["context_len"] is None
    assert rows[0]["name"] == "qwen3:8b"


def test_parse_ollama_ps_empty():
    assert K.parse_ollama_ps("") == []


# ---------------------------------------------------------------------------------------------------- log parsing
def test_parse_ollama_log_context_finds_num_ctx_and_mem(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("some banner\nloading model with num_ctx=65536 on device\ntotal 12000 MiB free 4000 MiB\n", encoding="utf-8")
    info = K.parse_ollama_log_context(str(log), since_pos=0)
    assert info["available"] is True
    assert 65536 in info["num_ctx_seen"]
    assert any(m["kind"] == "total" for m in info["mem_seen"])


def test_parse_ollama_log_context_missing_path():
    info = K.parse_ollama_log_context(None)
    assert info["available"] is False


def test_parse_ollama_log_context_respects_since_pos(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("num_ctx=4096\n", encoding="utf-8")
    pos = log.stat().st_size
    with open(log, "a", encoding="utf-8") as f:
        f.write("num_ctx=65536\n")
    info = K.parse_ollama_log_context(str(log), since_pos=pos)
    assert info["num_ctx_seen"] == [65536]


# ---------------------------------------------------------------------------------------------------- phase_tier
def test_phase_tier_prefers_ps_context_over_log(tmp_path):
    lab = make_lab(tmp_path)
    ollama = FakeOllama({"outcome": "ok", "status": 200, "prompt_eval_count": 20, "duration_s": 1.2})
    row = K.phase_tier(lab, "qwen3:8b", rep=0, ollama=ollama,
                       ps_fn=lambda: [{"name": "qwen3:8b", "context_len": 65536, "raw": {}}],
                       log_finder=lambda: None)
    assert row["ollama_default_ctx"] == 65536
    assert row["record"] == "tier"
    assert ollama.calls[0]["num_ctx"] is None  # no override, per spec


def test_phase_tier_falls_back_to_log_when_ps_has_no_context(tmp_path):
    lab = make_lab(tmp_path)
    ollama = FakeOllama({"outcome": "ok", "status": 200, "prompt_eval_count": 20, "duration_s": 1.0})
    row = K.phase_tier(lab, "qwen3:8b", rep=0, ollama=ollama,
                       ps_fn=lambda: [{"name": "qwen3:8b", "context_len": None, "raw": {}}],
                       log_finder=lambda: "/fake/does/not/exist.log")
    assert row["ollama_default_ctx"] is None  # log path doesn't exist -> available False -> no fallback value


# ---------------------------------------------------------------------------------------------------- phase_memory
def test_phase_memory_pressure_confirms_and_cleans_up(tmp_path):
    lab = make_lab(tmp_path)
    ollama = FakeOllama({"outcome": "ok", "status": 200, "prompt_eval_count": 20, "duration_s": 1.0})
    fake_srv = FakeServer(ok=True)
    avails = iter([100000.0, 100000.0 - 30 * 1024])  # 30 GB drop
    row = K.phase_memory_pressure(
        lab, "qwen3:8b", occupier_mi=object(), target_gb=30.0, occupier_n_ctx=32768, rep=0, ollama=ollama,
        ps_fn=lambda: [{"name": "qwen3:8b", "context_len": 60000, "raw": {}}], log_finder=lambda: None,
        server_factory=lambda mi, n_ctx, tag: fake_srv, avail_mb_fn=lambda: next(avails))
    assert fake_srv.stopped is True
    assert row["held_confirmed"] is True
    assert abs(row["held_gb"] - 30.0) < 0.01
    assert row["ollama_default_ctx"] == 60000


def test_phase_memory_pressure_flags_unconfirmed_when_drop_too_small(tmp_path):
    lab = make_lab(tmp_path)
    ollama = FakeOllama({"outcome": "ok", "status": 200, "prompt_eval_count": 20, "duration_s": 1.0})
    fake_srv = FakeServer(ok=True)
    avails = iter([100000.0, 100000.0 - 5 * 1024])  # only 5 GB drop against a 30 GB target
    row = K.phase_memory_pressure(
        lab, "qwen3:8b", occupier_mi=object(), target_gb=30.0, occupier_n_ctx=32768, rep=0, ollama=ollama,
        ps_fn=lambda: [], log_finder=lambda: None, server_factory=lambda mi, n_ctx, tag: fake_srv,
        avail_mb_fn=lambda: next(avails))
    assert fake_srv.stopped is True
    assert row["held_confirmed"] is False


def test_phase_memory_pressure_cleans_up_on_start_failure(tmp_path):
    lab = make_lab(tmp_path)
    ollama = FakeOllama({"outcome": "ok", "status": 200})
    fake_srv = FakeServer(ok=False)
    row = K.phase_memory_pressure(
        lab, "qwen3:8b", occupier_mi=object(), target_gb=30.0, occupier_n_ctx=32768, rep=0, ollama=ollama,
        server_factory=lambda mi, n_ctx, tag: fake_srv, avail_mb_fn=lambda: 100000.0)
    assert row["occupier_ok"] is False
    assert row["held_gb"] is None
    # no ollama chat should have been attempted once the occupier failed to start
    assert ollama.calls == []


# ---------------------------------------------------------------------------------------------------- phase_curves
def test_phase_quality_curves_writes_three_arms_per_condition(tmp_path):
    lab = make_lab(tmp_path)
    ollama = FakeOllama({"outcome": "ok", "status": 200, "prompt_eval_count": 20, "message": "the code is 123456", "duration_s": 1.0})
    fake_srv = FakeServer(ok=True, tokens=3000, chat_outcome="ok", output="the code is 123456")
    rows = K.phase_quality_curves(
        lab, "qwen3:8b", gguf_mi=object(), default_ctx=32768, rep_count=1, lengths=(3000,), task_types=("niah_multikey",),
        ollama=ollama, server_factory=lambda mi, n_ctx, tag: fake_srv)
    arms = sorted(r["arm"] for r in rows)
    assert arms == ["a", "b", "c"]
    assert fake_srv.stopped is True
    b_row = next(r for r in rows if r["arm"] == "b")
    assert b_row["num_ctx_requested"] > 3000  # fit_ctx has headroom over the prompt length


def test_phase_quality_curves_arm_c_error_recorded_as_errored(tmp_path):
    lab = make_lab(tmp_path)
    ollama = FakeOllama({"outcome": "ok", "status": 200, "prompt_eval_count": 20, "message": "123456", "duration_s": 1.0})
    fake_srv = FakeServer(ok=True, tokens=96000, chat_outcome="error", output=None)
    rows = K.phase_quality_curves(
        lab, "qwen3:8b", gguf_mi=object(), default_ctx=8192, rep_count=1, lengths=(96000,), task_types=("niah_multikey",),
        ollama=ollama, server_factory=lambda mi, n_ctx, tag: fake_srv)
    c_row = next(r for r in rows if r["arm"] == "c")
    assert c_row["errored"] is True
    assert c_row["score"] is None


# ---------------------------------------------------------------------------------------------------- build_default_ctx_from_rows
def test_build_default_ctx_from_rows_filters_by_host():
    rows = [
        {"record": "tier", "host": "evo-t2s", "ollama_default_ctx": 60000},
        {"record": "tier", "host": "evo-x2", "ollama_default_ctx": 70000},
    ]
    assert K.build_default_ctx_from_rows(rows, host="evo-x2") == 70000
    assert K.build_default_ctx_from_rows(rows, host="evo-t2s") == 60000
    assert K.build_default_ctx_from_rows([]) is None


# ---------------------------------------------------------------------------------------------------- K1Lab resume
def test_k1lab_resume_reads_done_phases(tmp_path):
    lab1 = make_lab(tmp_path)
    lab1.phase_done("phase_tier")
    stem = lab1.stem
    args2 = argparse.Namespace(resume=stem, out_dir=str(tmp_path), ollama_port=11434)
    lab2 = K.K1Lab(args2, K.require_host("evo-t2s"), {"git_head": "x", "committed": True, "problems": [], "files": []})
    assert "phase_tier" in lab2.done_phases


# ---------------------------------------------------------------------------------------------------- kill_criteria
def test_kill_criteria_defaults_pass_when_both_hosts_at_or_above_threshold():
    rows = [
        {"record": "tier", "host": "evo-t2s", "ollama_default_ctx": 65536},
        {"record": "tier", "host": "evo-x2", "ollama_default_ctx": 131072},
    ]
    result = K.kill_criteria(rows)
    assert result["defaults_cover_agent_prompts"]["ok"] is True


def test_kill_criteria_defaults_fail_when_one_host_below_threshold():
    rows = [
        {"record": "tier", "host": "evo-t2s", "ollama_default_ctx": 65536},
        {"record": "tier", "host": "evo-x2", "ollama_default_ctx": 4096},
    ]
    result = K.kill_criteria(rows)
    assert result["defaults_cover_agent_prompts"]["ok"] is False
    assert "evo-x2" in result["defaults_cover_agent_prompts"]["reason"]


def test_kill_criteria_truncation_passes_within_tolerance():
    rows = [
        {"record": "curve", "arm": "b", "prompt_len_target": 48000, "task_type": "needle_recall", "rep": 0, "score": 1.0},
        {"record": "curve", "arm": "c", "prompt_len_target": 48000, "task_type": "needle_recall", "rep": 0, "score": 0.95,
         "errored": True, "prompt_eval_count": None},
    ]
    result = K.kill_criteria(rows)
    assert result["truncated_cases_keep_quality"]["ok"] is True


def test_kill_criteria_truncation_fails_beyond_tolerance():
    rows = [
        {"record": "curve", "arm": "b", "prompt_len_target": 48000, "task_type": "needle_recall", "rep": 0, "score": 1.0},
        {"record": "curve", "arm": "c", "prompt_len_target": 48000, "task_type": "needle_recall", "rep": 0, "score": 0.0,
         "errored": True, "prompt_eval_count": None},
    ]
    result = K.kill_criteria(rows)
    assert result["truncated_cases_keep_quality"]["ok"] is False


def test_kill_criteria_truncation_no_truncated_cases_is_not_a_pass():
    rows = [
        {"record": "curve", "arm": "b", "prompt_len_target": 48000, "task_type": "needle_recall", "rep": 0, "score": 1.0},
        {"record": "curve", "arm": "c", "prompt_len_target": 48000, "task_type": "needle_recall", "rep": 0, "score": 1.0,
         "errored": False, "prompt_eval_count": 48000},
    ]
    result = K.kill_criteria(rows)
    assert result["truncated_cases_keep_quality"]["ok"] is False
    assert "no truncated" in result["truncated_cases_keep_quality"]["reason"]


def test_kill_criteria_truncation_uses_prompt_eval_count_shortfall_not_just_error_flag():
    rows = [
        {"record": "curve", "arm": "b", "prompt_len_target": 96000, "task_type": "needle_recall", "rep": 1, "score": 1.0},
        {"record": "curve", "arm": "c", "prompt_len_target": 96000, "task_type": "needle_recall", "rep": 1, "score": 1.0,
         "errored": False, "prompt_eval_count": 8000},  # silently truncated, not errored, but still short
    ]
    result = K.kill_criteria(rows)
    assert result["truncated_cases_keep_quality"]["ok"] is True  # score still matches, so this is a pass, but it must have been *checked*


# ---------------------------------------------------------------------------------------------------- ollama lifecycle wiring
def test_main_starts_and_stops_its_own_ollama_server():
    """K1 is one of only two scripts allowed to run Ollama (2026-09-29 contamination check); it must start its own
    server and always stop it, even on failure, rather than assume one is already running in the background."""
    import inspect
    src = inspect.getsource(K.main)
    assert "_hc.start_ollama_server()" in src
    assert "_hc.stop_ollama_server()" in src
    # the stop call must be inside a finally block so it always runs
    start_idx = src.index("_hc.start_ollama_server()")
    finally_idx = src.index("finally:")
    stop_idx = src.index("_hc.stop_ollama_server()")
    assert start_idx < finally_idx < stop_idx


# ---------------------------------------------------------------------------------------------------- queue advance wiring
def test_main_calls_tq_advance_in_a_finally_block():
    """K1 never called t2s_queue.advance() at all (found 2026-09-29 on evo-x2: the queue sat 'running' after a
    successful run finished, until the watchdog's heartbeat-staleness threshold expired). main() must always
    advance the queue on every exit path."""
    import inspect
    src = inspect.getsource(K.main)
    assert "tq.advance(note)" in src
    finally_idx = src.rindex("finally:")
    advance_idx = src.index("tq.advance(note)")
    assert finally_idx < advance_idx
