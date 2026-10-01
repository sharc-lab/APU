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
    def __init__(self, responses, ps_models=None, ps_after_unload=None):
        """responses: list of dicts popped in call order, or a single dict reused for every call. ps_models: list
        of model dicts get_ps() returns before unload() is called; ps_after_unload defaults to [] (confirmed
        empty) unless overridden to test the "unload did not actually clear it" case."""
        self.responses = responses
        self.calls = []
        self.ps_models = ps_models if ps_models is not None else []
        self.ps_after_unload = ps_after_unload if ps_after_unload is not None else []
        self.unload_called = False

    def chat(self, model, prompt, num_ctx=None, max_tokens=64, keep_alive=None):
        self.calls.append({"model": model, "num_ctx": num_ctx, "max_tokens": max_tokens, "prompt_len": len(prompt),
                           "keep_alive": keep_alive})
        if isinstance(self.responses, list):
            return self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        return self.responses

    def get_ps(self):
        models = self.ps_after_unload if self.unload_called else self.ps_models
        return {"outcome": "ok", "models": models}

    def unload(self, model):
        self.unload_called = True
        return self.chat(model, "", num_ctx=None, max_tokens=1, keep_alive=0)


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
    ollama = FakeOllama({"outcome": "ok", "status": 200, "prompt_eval_count": 20, "duration_s": 1.2},
                        ps_models=[{"name": "qwen3:8b", "context_length": 65536}])
    row = K.phase_tier(lab, "qwen3:8b", rep=0, ollama=ollama, log_finder=lambda: None)
    assert row["ollama_default_ctx"] == 65536
    assert row["ollama_default_ctx_source"] == "api_ps"
    assert row["record"] == "tier"
    assert ollama.calls[0]["num_ctx"] is None  # no override, per spec
    assert ollama.calls[0]["keep_alive"] == "10m"  # (a): kept alive so ps has something to read
    assert ollama.unload_called is True  # (a): explicitly unloaded afterward
    assert row["unload_confirmed_empty"] is True


def test_phase_tier_falls_back_to_log_when_ps_has_no_context(tmp_path):
    lab = make_lab(tmp_path)
    log = tmp_path / "server.log"
    log.write_text("", encoding="utf-8")  # log_pos must start at 0 for the new-content-only read below to see it
    ollama = FakeOllama({"outcome": "ok", "status": 200, "prompt_eval_count": 20, "duration_s": 1.0},
                        ps_models=[{"name": "qwen3:8b"}])  # no context_length field

    def fake_chat(model, prompt, num_ctx=None, max_tokens=64, keep_alive=None):
        ollama.calls.append({"model": model, "keep_alive": keep_alive})
        if keep_alive == "10m":
            # simulates the real Ollama server appending its runner start line to server.log during this call
            with open(log, "a", encoding="utf-8") as f:
                f.write("...starting runner... --ctx-size 32768 --other-flag\n")
        return {"outcome": "ok", "status": 200, "prompt_eval_count": 20, "duration_s": 1.0, "message": "Paris."}

    ollama.chat = fake_chat
    row = K.phase_tier(lab, "qwen3:8b", rep=0, ollama=ollama, log_finder=lambda: str(log))
    assert row["ollama_default_ctx"] == 32768
    assert row["ollama_default_ctx_source"] == "server_log"


def test_phase_tier_falls_back_to_empirical_probe_when_ps_and_log_both_empty(tmp_path):
    lab = make_lab(tmp_path)
    # tier chat + unload + 7 empirical probes = 9 calls; every call reports prompt_eval_count == target (no
    # truncation at any length in this test), so the probe should report the largest length as the estimate.
    ollama = FakeOllama({"outcome": "ok", "status": 200, "prompt_eval_count": 1000, "duration_s": 1.0},
                        ps_models=[])  # nothing loaded (confirms the OLLAMA_KEEP_ALIVE=0 real-world finding)

    probe_targets = iter(K.EMPIRICAL_CTX_PROBE_LENGTHS)
    call_kinds = []  # "tier", "unload", then one "probe" per empirical call, in order

    def fake_chat(model, prompt, num_ctx=None, max_tokens=64, keep_alive=None):
        ollama.calls.append({"model": model, "keep_alive": keep_alive})
        if not call_kinds:
            call_kinds.append("tier")
            return {"outcome": "ok", "status": 200, "prompt_eval_count": 20, "duration_s": 1.0, "message": "Paris."}
        if keep_alive == 0 and "unload" not in call_kinds:
            call_kinds.append("unload")
            return {"outcome": "ok", "status": 200, "prompt_eval_count": 0, "duration_s": 0.1, "message": ""}
        call_kinds.append("probe")
        target = next(probe_targets)
        return {"outcome": "ok", "status": 200, "prompt_eval_count": target, "duration_s": 1.0, "message": ""}

    ollama.chat = fake_chat
    row = K.phase_tier(lab, "qwen3:8b", rep=0, ollama=ollama, log_finder=lambda: None)
    assert row["ollama_default_ctx_source"] == "empirical_probe"
    assert row["ollama_default_ctx_from_empirical"] == K.EMPIRICAL_CTX_PROBE_LENGTHS[-1]


def test_phase_tier_raises_when_no_signal_captures_context(tmp_path):
    """(d): must never write a row a kill-criteria check could read as a verdict on missing data."""
    lab = make_lab(tmp_path)
    ollama = FakeOllama({"outcome": "ok", "status": 200, "prompt_eval_count": None, "duration_s": 1.0},
                        ps_models=[])
    try:
        K.phase_tier(lab, "qwen3:8b", rep=0, ollama=ollama, log_finder=lambda: None)
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "INVALID" in str(e)
    # the row was still emitted (for forensics) but must not carry a usable ollama_default_ctx
    rows = lab.all_rows()
    tier_rows = [r for r in rows if r.get("record") == "tier"]
    assert len(tier_rows) == 1
    assert tier_rows[0]["ollama_default_ctx"] is None


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


# ---------------------------------------------------------------------------------------------------- phase_curves: calibration gate (2026-09-29 fix)
def test_phase_quality_curves_unfiltered_when_calibration_pass_set_is_none(tmp_path):
    """None means 'unknown, run everything' -- the pre-fix behavior, and the default when no --calibration-file is
    given at all, so an operator who never ran calibration is not silently blocked."""
    lab = make_lab(tmp_path)
    ollama = FakeOllama({"outcome": "ok", "status": 200, "prompt_eval_count": 20, "message": "123456", "duration_s": 1.0})
    fake_srv = FakeServer(ok=True, tokens=3000, output="123456")
    rows = K.phase_quality_curves(
        lab, "qwen3:8b", gguf_mi=object(), default_ctx=32768, rep_count=1, lengths=(3000,),
        task_types=("niah_multikey", "common_words_extraction"), ollama=ollama,
        server_factory=lambda mi, n_ctx, tag: fake_srv, calibration_pass_set=None)
    assert {r["task_type"] for r in rows} == {"niah_multikey", "common_words_extraction"}


def test_phase_quality_curves_excludes_task_types_not_in_pass_set(tmp_path):
    """The actual gate fix: a task_type that failed q0_token_calibration for this model must not run here, even
    though the calibration job's queue status was 'done' (found 2026-09-29: common_words_extraction failed
    calibration while the job as a whole still reached done, on both machines)."""
    lab = make_lab(tmp_path)
    ollama = FakeOllama({"outcome": "ok", "status": 200, "prompt_eval_count": 20, "message": "123456", "duration_s": 1.0})
    fake_srv = FakeServer(ok=True, tokens=3000, output="123456")
    rows = K.phase_quality_curves(
        lab, "qwen3:8b", gguf_mi=object(), default_ctx=32768, rep_count=1, lengths=(3000,),
        task_types=("niah_multikey", "common_words_extraction"), ollama=ollama,
        server_factory=lambda mi, n_ctx, tag: fake_srv, calibration_pass_set={"niah_multikey"})
    assert {r["task_type"] for r in rows} == {"niah_multikey"}


def test_phase_quality_curves_empty_pass_set_excludes_everything():
    """An empty (but non-None) pass set means every requested task_type failed calibration; this must not fall
    back to running unfiltered, and must not raise -- it just produces no rows."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        lab = make_lab(Path(d))
        ollama = FakeOllama({"outcome": "ok", "status": 200, "prompt_eval_count": 20, "message": "123456", "duration_s": 1.0})
        fake_srv = FakeServer(ok=True, tokens=3000, output="123456")
        rows = K.phase_quality_curves(
            lab, "qwen3:8b", gguf_mi=object(), default_ctx=32768, rep_count=1, lengths=(3000,),
            task_types=("niah_multikey",), ollama=ollama,
            server_factory=lambda mi, n_ctx, tag: fake_srv, calibration_pass_set=set())
        assert rows == []


def test_phase_quality_curves_logs_exclusion_reason(tmp_path, capsys):
    lab = make_lab(tmp_path)
    ollama = FakeOllama({"outcome": "ok", "status": 200, "prompt_eval_count": 20, "message": "123456", "duration_s": 1.0})
    fake_srv = FakeServer(ok=True, tokens=3000, output="123456")
    K.phase_quality_curves(
        lab, "qwen3:8b", gguf_mi=object(), default_ctx=32768, rep_count=1, lengths=(3000,),
        task_types=("niah_multikey", "common_words_extraction"), ollama=ollama,
        server_factory=lambda mi, n_ctx, tag: fake_srv, calibration_pass_set={"niah_multikey"})
    out = capsys.readouterr()
    combined = out.out + out.err
    assert "common_words_extraction" in combined
    assert "did not pass q0_token_calibration" in combined


# ---------------------------------------------------------------------------------------------------- main(): --calibration-file wiring
_REQUIRED_ARGS = ["--host", "evo-t2s", "--ollama-model", "qwen3:8b"]


def test_build_arg_parser_has_calibration_file_flag():
    ap = K.build_arg_parser()
    args = ap.parse_args(_REQUIRED_ARGS + ["--calibration-file", "some/path.jsonl"])
    assert args.calibration_file == "some/path.jsonl"


def test_build_arg_parser_calibration_file_defaults_to_none():
    ap = K.build_arg_parser()
    args = ap.parse_args(_REQUIRED_ARGS)
    assert args.calibration_file is None


def test_main_passes_calibration_pass_set_to_phase_quality_curves():
    """main() must load --calibration-file via quality_suite.load_calibration_pass_set and thread the result into
    the phase_quality_curves(...) call site, not just accept the flag and drop it."""
    import inspect
    src = inspect.getsource(K.main)
    assert "load_calibration_pass_set" in src
    assert "calibration_pass_set=" in src


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
# ---------------------------------------------------------------------------------------------------- K1 v3: OllamaClient.pull
class _FakeHTTPResponse:
    """Minimal fake for the object urllib.request.urlopen(...) returns when used as a context manager: iterating
    over it yields raw response lines, exactly like a real streamed HTTP body."""

    def __init__(self, lines):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        return iter(self._lines)


def test_ollama_client_pull_blocks_until_success_line(monkeypatch):
    lines = [
        b'{"status": "pulling manifest"}\n',
        b'{"status": "pulling abc123", "total": 100, "completed": 50}\n',
        b'{"status": "verifying sha256 digest"}\n',
        b'{"status": "success"}\n',
    ]
    monkeypatch.setattr(K.urllib.request, "urlopen", lambda req, timeout=None: _FakeHTTPResponse(lines))
    client = K.OllamaClient()
    res = client.pull("qwen3:4b-instruct-2507")
    assert res["outcome"] == "ok"
    assert res["final_status"] == "success"
    assert len(res["status_lines"]) == 4


def test_ollama_client_pull_reports_error_line(monkeypatch):
    lines = [b'{"status": "pulling manifest"}\n', b'{"error": "model not found"}\n']
    monkeypatch.setattr(K.urllib.request, "urlopen", lambda req, timeout=None: _FakeHTTPResponse(lines))
    client = K.OllamaClient()
    res = client.pull("nonexistent:tag")
    assert res["outcome"] == "error"
    assert res["final_status"] == "model not found"


def test_create_model_from_gguf_writes_a_modelfile_with_just_the_from_line(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setenv("OLLAMA_BIN", "ollama")  # deterministic exe resolution, independent of this machine's PATH

    class FakeResult:
        returncode = 0
        stdout = "writing manifest\nsuccess"
        stderr = ""

    def fake_run_fn(argv):
        assert argv[0] == "ollama"
        assert argv[1] == "create"
        assert argv[2] == "qwen3-4b-2507"
        assert argv[3] == "-f"
        modelfile_path = argv[4]
        captured["modelfile_content"] = open(modelfile_path, encoding="utf-8").read()
        return FakeResult()

    res = K.create_model_from_gguf("qwen3-4b-2507", r"C:\apu\models\qwen3-4b-instruct-85e4a5b7.gguf",
                                   run_fn=fake_run_fn)
    assert res["outcome"] == "ok"
    assert captured["modelfile_content"] == "FROM C:\\apu\\models\\qwen3-4b-instruct-85e4a5b7.gguf\n"


def test_create_model_from_gguf_resolves_exe_via_shutil_which_when_no_env_var(tmp_path, monkeypatch):
    """2026-09-30 bug: plain 'ollama' failed with FileNotFoundError from subprocess.run (no shell=True) on
    evo-t2s even though the Ollama server itself was already running. Resolve via OLLAMA_BIN or shutil.which,
    matching stage_a_kv_precision.py's existing convention for this exact binary. create_model_from_gguf does
    "import shutil" locally, which reuses the same cached sys.modules['shutil'] object -- patching .which on the
    module imported here at the top of this test file affects it too."""
    import shutil
    monkeypatch.delenv("OLLAMA_BIN", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: r"C:\real\path\ollama.exe" if name == "ollama" else None)
    captured = {}

    class FakeResult:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run_fn(argv):
        captured["exe"] = argv[0]
        return FakeResult()

    K.create_model_from_gguf("x", "/nope.gguf", run_fn=fake_run_fn)
    assert captured["exe"] == r"C:\real\path\ollama.exe"


def test_resolve_ollama_exe_falls_back_to_localappdata_when_which_and_env_fail(tmp_path, monkeypatch):
    """Found live 2026-10-01 (evo-t2s): both OLLAMA_BIN and shutil.which failed in a WMI-launched process even
    though the exe is confirmed present at the standard per-user install path -- the most likely explanation
    is that WMI-spawned processes do not inherit the interactive user's PATH/profile. This fallback does not
    depend on PATH at all."""
    import shutil
    monkeypatch.delenv("OLLAMA_BIN", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    fake_local = tmp_path / "Local"
    ollama_dir = fake_local / "Programs" / "Ollama"
    ollama_dir.mkdir(parents=True)
    exe_path = ollama_dir / "ollama.exe"
    exe_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("LOCALAPPDATA", str(fake_local))
    assert K._resolve_ollama_exe() == str(exe_path)


def test_resolve_ollama_exe_prefers_env_var_over_everything(monkeypatch):
    import shutil
    monkeypatch.setenv("OLLAMA_BIN", r"C:\explicit\ollama.exe")
    monkeypatch.setattr(shutil, "which", lambda name: r"C:\which\ollama.exe")
    assert K._resolve_ollama_exe() == r"C:\explicit\ollama.exe"


def test_resolve_ollama_exe_bare_command_as_last_resort(tmp_path, monkeypatch):
    import shutil
    monkeypatch.delenv("OLLAMA_BIN", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))  # exists but no Programs/Ollama/ollama.exe inside it
    assert K._resolve_ollama_exe() == "ollama"


def test_create_model_from_gguf_reports_error_without_raising():
    class FakeResult:
        returncode = 1
        stdout = ""
        stderr = "Error: gguf file not found"

    res = K.create_model_from_gguf("x", "/nope.gguf", run_fn=lambda argv: FakeResult())
    assert res["outcome"] == "error"
    assert "not found" in res["stderr"]


def test_create_model_from_gguf_never_raises_on_a_run_fn_exception():
    def failing_run_fn(argv):
        raise FileNotFoundError("ollama executable not found")

    res = K.create_model_from_gguf("x", "/nope.gguf", run_fn=failing_run_fn)
    assert res["outcome"] == "error"
    assert "FileNotFoundError" in res["stderr"]


def test_pull_model_wrapper_calls_ollama_pull():
    calls = []

    class FakePullOllama:
        def pull(self, tag):
            calls.append(tag)
            return {"outcome": "ok", "final_status": "success", "status_lines": []}

    res = K.pull_model(FakePullOllama(), "llama3.1:8b")
    assert calls == ["llama3.1:8b"]
    assert res["outcome"] == "ok"


# ---------------------------------------------------------------------------------------------------- K1 v3: phase_tier_v3
class FakeV3Ollama:
    """Records every call (pull/chat/get_ps/unload) in call order, in one shared call_log, so tests can assert on
    ordering (pull-first) as well as content. chat/get_ps/unload always succeed with fixed, deterministic content;
    real quality_suite marker text is irrelevant to what these tests check."""

    def __init__(self):
        self.call_log = []
        self.pull_calls = []

    def pull(self, tag):
        self.call_log.append(("pull", tag))
        self.pull_calls.append(tag)
        return {"outcome": "ok", "final_status": "success", "status_lines": [{"status": "success"}]}

    def chat(self, model, prompt, num_ctx=None, max_tokens=64, keep_alive=None, messages=None, tools=None,
             timeout=None):
        self.call_log.append(("chat", model))
        return {"outcome": "ok", "status": 200, "prompt_eval_count": 20, "duration_s": 0.1, "message": "Paris."}

    def get_ps(self):
        self.call_log.append(("get_ps", None))
        return {"outcome": "ok", "models": []}

    def unload(self, model):
        self.call_log.append(("unload", model))
        return self.chat(model, "", num_ctx=None, max_tokens=1, keep_alive=0)


def _fake_create_fn_ok(calls):
    def create_fn(name, gguf_path):
        calls.append((name, gguf_path))
        return {"outcome": "ok", "returncode": 0, "stdout": "success", "stderr": ""}
    return create_fn


def test_phase_tier_v3_makes_all_models_available_before_any_measurement(tmp_path):
    """Requirement (2): making every model available (pull or create) is the literal first step, before any
    measurement, nothing else concurrent. qwen3-4b-2507 is created from its local GGUF (its registry tag does not
    exist, found 2026-09-30); the other two are pulled as before."""
    lab = make_lab(tmp_path, host="evo-x2")
    fake = FakeV3Ollama()
    create_calls = []
    K.phase_tier_v3(lab, ollama=fake, log_finder=lambda: None, create_fn=_fake_create_fn_ok(create_calls))
    assert create_calls == [("qwen3-4b-2507", K.QWEN3_4B_GGUF_PATH)]
    pull_tags = [m["tag"] for m in K.K1_V3_MODELS if m.get("source", "pull") == "pull"]
    first_pulls = fake.call_log[:2]
    assert [c[0] for c in first_pulls] == ["pull", "pull"]
    assert [c[1] for c in first_pulls] == pull_tags
    first_non_pull_idx = next(i for i, c in enumerate(fake.call_log) if c[0] != "pull")
    assert first_non_pull_idx == 2  # exactly the 2 registry pulls, then measurement starts


def test_phase_tier_v3_one_model_failing_does_not_abort_the_others(tmp_path):
    """2026-09-30 fix: qwen3-4b-2507's create failing (or any model's pull failing) must not prevent the other
    models in K1_V3_MODELS from being measured -- this is exactly what happened live, twice, before the fix."""
    lab = make_lab(tmp_path, host="evo-x2")
    fake = FakeV3Ollama()

    def failing_create_fn(name, gguf_path):
        return {"outcome": "error", "returncode": 1, "stdout": "", "stderr": "gguf not found"}

    result = K.phase_tier_v3(lab, ollama=fake, log_finder=lambda: None, create_fn=failing_create_fn)
    assert len(result["failures"]) == 1
    assert result["failures"][0]["model_tag"] == "qwen3-4b-2507"
    # the other two models (both plain pulls) still got fully measured
    remaining_tags = {m["tag"] for m in K.K1_V3_MODELS if m["tag"] != "qwen3-4b-2507"}
    assert {r["model_tag"] for r in result["tier_rows"]} == remaining_tags
    assert {r["model_tag"] for r in result["meta_rows"]} == remaining_tags
    assert len(result["probe_rows"]) == 2 * len(K.K1_V3_PROBE_LENGTHS)


def test_phase_tier_v3_captures_real_pull_error_text_not_just_final_status(tmp_path):
    """Found live 2026-10-01 (evo-t2s): a pull that fails with a real exception (OllamaClient.pull's except
    Exception branch, e.g. a connection error) sets "error", not "final_status" -- final_status is only ever
    populated on an actual pull response. Reading only final_status silently discarded the real cause, logging
    'pull failed for x: None' with nothing to diagnose. The tier_v3_pull row's pull_error must carry the real
    text."""
    lab = make_lab(tmp_path, host="evo-t2s")

    def failing_pull_fn(ollama, tag):
        return {"outcome": "error", "final_status": None, "status_lines": [],
                "error": "ConnectionRefusedError: server not ready"}

    result = K.phase_tier_v3(lab, ollama=FakeV3Ollama(), log_finder=lambda: None,
                             create_fn=_fake_create_fn_ok([]), pull_fn=failing_pull_fn)
    pull_rows = [r for r in lab.all_rows() if r["record"] == "tier_v3_pull"]
    assert pull_rows
    assert all(r["pull_error"] == "ConnectionRefusedError: server not ready" for r in pull_rows)
    assert all("ConnectionRefusedError: server not ready" in f["error"] for f in result["failures"])


def test_phase_tier_v3_probe_sweep_five_rows_per_model_right_lengths(tmp_path):
    lab = make_lab(tmp_path, host="evo-x2")
    fake = FakeV3Ollama()
    result = K.phase_tier_v3(lab, ollama=fake, log_finder=lambda: None, create_fn=_fake_create_fn_ok([]))
    probe_rows = result["probe_rows"]
    assert len(probe_rows) == 3 * len(K.K1_V3_PROBE_LENGTHS)
    for m in K.K1_V3_MODELS:
        rows_for_model = [r for r in probe_rows if r["model_tag"] == m["tag"]]
        assert len(rows_for_model) == 5
        assert sorted(r["probe_target_tokens"] for r in rows_for_model) == sorted(K.K1_V3_PROBE_LENGTHS)
        for r in rows_for_model:
            assert r["record"] == "tier_v3_probe"
            assert r["http_status"] == 200
            assert r["sent_tokens_target"] == r["probe_target_tokens"]


def test_phase_tier_v3_labels_qwen3_8b_as_capped(tmp_path):
    """Requirement (1): qwen3:8b is kept as a labeled 'capped' control, contrasted against the two uncapped
    models -- capped must be False for the two models whose native ceiling exceeds any plausible tier."""
    lab = make_lab(tmp_path, host="evo-x2")
    fake = FakeV3Ollama()
    result = K.phase_tier_v3(lab, ollama=fake, log_finder=lambda: None, create_fn=_fake_create_fn_ok([]))
    meta_by_tag = {r["model_tag"]: r for r in result["meta_rows"]}
    assert meta_by_tag["qwen3:8b"]["capped"] is True
    assert meta_by_tag["qwen3:8b"]["native_ctx"] == 40960
    assert meta_by_tag["qwen3-4b-2507"]["capped"] is False
    assert meta_by_tag["qwen3-4b-2507"]["native_ctx"] == 262144
    assert meta_by_tag["llama3.1:8b"]["capped"] is False
    assert meta_by_tag["llama3.1:8b"]["native_ctx"] == 131072
    capped_probe_rows = [r for r in result["probe_rows"] if r["model_tag"] == "qwen3:8b"]
    assert capped_probe_rows and all(r["capped"] is True for r in capped_probe_rows)
    uncapped_probe_rows = [r for r in result["probe_rows"] if r["model_tag"] != "qwen3:8b"]
    assert uncapped_probe_rows and all(r["capped"] is False for r in uncapped_probe_rows)


def test_phase_tier_v3_dry_run_call_counts(tmp_path):
    """Stub dry run: exact call counts for the 3-model x 5-length K1 v3 design (1 create + 2 pulls + 3 tier
    calls + 15 probe rows)."""
    lab = make_lab(tmp_path, host="evo-x2")
    fake = FakeV3Ollama()
    create_calls = []
    result = K.phase_tier_v3(lab, ollama=fake, log_finder=lambda: None, create_fn=_fake_create_fn_ok(create_calls))
    assert len(create_calls) == 1
    assert len(fake.pull_calls) == 2
    assert len(result["tier_rows"]) == 3
    assert len(result["probe_rows"]) == 15


# ---------------------------------------------------------------------------------------------------- K1 v3: real
# labeling fix (A1, 2026-10-01): tokenizer-ratio calibration, long-probe timeout, marker-lost-vs-token-truncated
def _patch_qs_build_task_identity(monkeypatch):
    """probe_v3_effective_context builds prompts via qs_build_task(_EMPIRICAL_CTX_TASK_TYPE, target, seed) --
    stub it to something deterministic and fast, independent of the real quality_suite task generator, and
    expose which target each call corresponds to via a simple length-tagged string so a fake ollama can derive
    its response from the target without guessing from call order."""
    def fake_build(_task_type, target, _seed):
        return f"PROMPT_FOR_TARGET_{target}", "EXPECTED_MARKER", None
    monkeypatch.setattr(K, "qs_build_task", fake_build)


class _TargetAwareFakeOllama:
    """chat() reads the target back out of the prompt text (set by _patch_qs_build_task_identity) so its
    response can depend on which length is being probed, without the test needing to track call order."""

    def __init__(self, ratio=1.0, cap_processed_at=None, truncate_from=None, marker_fails_for=()):
        self.ratio = ratio
        self.cap_processed_at = cap_processed_at
        self.truncate_from = truncate_from or set()
        self.marker_fails_for = set(marker_fails_for)
        self.calls = []

    def _target_from_prompt(self, prompt):
        return int(prompt.rsplit("_", 1)[-1])

    def chat(self, model, prompt, num_ctx=None, max_tokens=64, keep_alive=None, messages=None, tools=None,
             timeout=None):
        target = self._target_from_prompt(prompt)
        self.calls.append({"target": target, "num_ctx": num_ctx, "max_tokens": max_tokens, "timeout": timeout})
        real_sent = round(target * self.ratio)
        if max_tokens == 1:  # the calibration call
            return {"outcome": "ok", "status": 200, "prompt_eval_count": real_sent, "message": ""}
        if target in self.truncate_from:
            pec = self.cap_processed_at if self.cap_processed_at is not None else round(real_sent * 0.5)
        else:
            pec = real_sent
        message = "" if target in self.marker_fails_for else "EXPECTED_MARKER"
        return {"outcome": "ok", "status": 200, "prompt_eval_count": pec, "message": message}

    def get_ps(self):
        return {"outcome": "ok", "models": []}


def test_probe_v3_uses_real_tokenizer_ratio_not_raw_target(tmp_path, monkeypatch):
    """The exact bug found live: a model whose own tokenizer counts a target-16000 prompt as ~0.818x that many
    tokens must NOT be labeled truncated just because prompt_eval_count < 16000 -- only if it falls short of
    its OWN real count (sent_tokens_real)."""
    _patch_qs_build_task_identity(monkeypatch)
    lab = make_lab(tmp_path, host="evo-t2s")
    fake = _TargetAwareFakeOllama(ratio=0.818)  # no truncate_from -- every length fully processed relative to real
    rows = K.probe_v3_effective_context(lab, fake, "llama3.1:8b", capped=False, lengths=(16000, 32000),
                                        native_ctx=131072)
    for r in rows:
        assert r["sent_tokens_real"] == round(r["probe_target_tokens"] * 0.818)
        assert r["token_truncated"] is False
        assert r["silently_truncated"] is False


def test_probe_v3_flags_real_token_truncation_against_its_own_tokenizer_count(tmp_path, monkeypatch):
    _patch_qs_build_task_identity(monkeypatch)
    lab = make_lab(tmp_path, host="evo-x2")
    fake = _TargetAwareFakeOllama(ratio=1.0, truncate_from={48000, 96000, 128000}, cap_processed_at=20482)
    rows = K.probe_v3_effective_context(lab, fake, "qwen3:8b", capped=True,
                                        lengths=(16000, 32000, 48000, 96000, 128000), native_ctx=40960)
    by_target = {r["probe_target_tokens"]: r for r in rows}
    assert by_target[16000]["token_truncated"] is False
    assert by_target[32000]["token_truncated"] is False
    assert by_target[48000]["token_truncated"] is True
    assert by_target[48000]["prompt_eval_count"] == 20482
    assert by_target[96000]["token_truncated"] is True
    assert by_target[128000]["token_truncated"] is True


def test_probe_v3_marker_lost_without_token_truncation_is_flagged_separately(tmp_path, monkeypatch):
    """A model can lose the marker for a reason unrelated to context truncation (the real qwen3:8b case at
    16K/32K, where token math shows no truncation at all) -- this must be reported as
    marker_lost_unexplained, and silently_truncated still fires (the marker loss itself is a real failure
    signal worth surfacing), but token_truncated must stay False so the two mechanisms are never conflated."""
    _patch_qs_build_task_identity(monkeypatch)
    lab = make_lab(tmp_path, host="evo-x2")
    fake = _TargetAwareFakeOllama(ratio=1.0, marker_fails_for={16000})
    rows = K.probe_v3_effective_context(lab, fake, "qwen3:8b", capped=True, lengths=(16000,), native_ctx=40960)
    r = rows[0]
    assert r["token_truncated"] is False
    assert r["marker_lost_unexplained"] is True
    assert r["silently_truncated"] is True


def test_probe_v3_passes_long_timeout_only_for_long_lengths(tmp_path, monkeypatch):
    _patch_qs_build_task_identity(monkeypatch)
    lab = make_lab(tmp_path, host="evo-x2")
    fake = _TargetAwareFakeOllama(ratio=1.0)
    K.probe_v3_effective_context(lab, fake, "qwen3-4b-2507", capped=False, lengths=(32000, 96000),
                                 native_ctx=262144, long_probe_threshold=64000, long_probe_timeout_s=3600)
    probe_calls = [c for c in fake.calls if c["max_tokens"] == 32]
    timeouts_by_target = {c["target"]: c["timeout"] for c in probe_calls}
    assert timeouts_by_target[32000] is None
    assert timeouts_by_target[96000] == 3600


def test_probe_v3_calibration_call_uses_native_ctx_and_shortest_length(tmp_path, monkeypatch):
    _patch_qs_build_task_identity(monkeypatch)
    lab = make_lab(tmp_path, host="evo-x2")
    fake = _TargetAwareFakeOllama(ratio=0.818)
    K.probe_v3_effective_context(lab, fake, "llama3.1:8b", capped=False, lengths=(16000, 32000),
                                 native_ctx=131072)
    cal_call = next(c for c in fake.calls if c["max_tokens"] == 1)
    assert cal_call["target"] == 16000
    assert cal_call["num_ctx"] == 131072


def test_probe_v3_emits_a_calibration_row_not_counted_in_returned_probe_rows(tmp_path, monkeypatch):
    _patch_qs_build_task_identity(monkeypatch)
    lab = make_lab(tmp_path, host="evo-x2")
    fake = _TargetAwareFakeOllama(ratio=1.0)
    rows = K.probe_v3_effective_context(lab, fake, "qwen3-4b-2507", capped=False, lengths=(16000, 32000),
                                        native_ctx=262144)
    assert len(rows) == 2  # calibration row not included
    assert all(r["record"] == "tier_v3_probe" for r in rows)
    cal_rows = [r for r in lab.all_rows() if r["record"] == "tier_v3_probe_calibration"]
    assert len(cal_rows) == 1
    assert cal_rows[0]["tokenizer_ratio"] == 1.0


# ---------------------------------------------------------------------------------------------------- K1 v3: GPU/
# compute-device detection (A2, 2026-10-01)
def test_parse_ollama_log_device_detects_gpu_from_library_line(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("some noise\nmsg=\"inference compute\" library=vulkan name=\"Intel Arc\"\nmore noise\n",
                   encoding="utf-8")
    info = K.parse_ollama_log_device(str(log))
    assert info["available"] is True
    assert info["compute_device_guess"] == "gpu"


def test_parse_ollama_log_device_detects_cpu_fallback(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("looking for compatible GPUs\nno compatible GPUs were discovered, falling back to CPU\n",
                   encoding="utf-8")
    info = K.parse_ollama_log_device(str(log))
    assert info["compute_device_guess"] == "cpu"


def test_parse_ollama_log_device_unknown_when_nothing_matches(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("totally unrelated log content\n", encoding="utf-8")
    info = K.parse_ollama_log_device(str(log))
    assert info["compute_device_guess"] == "unknown"


def test_parse_ollama_log_device_missing_file_is_unknown_not_a_crash():
    info = K.parse_ollama_log_device(None)
    assert info["available"] is False
    assert info["compute_device_guess"] == "unknown"


def test_phase_tier_v3_emits_one_device_detect_row_per_run(tmp_path, monkeypatch):
    lab = make_lab(tmp_path, host="evo-t2s")
    fake = FakeV3Ollama()
    K.phase_tier_v3(lab, ollama=fake, log_finder=lambda: None, create_fn=_fake_create_fn_ok([]))
    device_rows = [r for r in lab.all_rows() if r["record"] == "tier_v3_device_detect"]
    assert len(device_rows) == 1
    assert device_rows[0]["compute_device_guess"] == "unknown"  # log_finder returns None in this test


# ---------------------------------------------------------------------------------------------------- K1 v3: CLI wiring
def test_build_arg_parser_accepts_tier_v3_phase():
    ap = K.build_arg_parser()
    args = ap.parse_args(_REQUIRED_ARGS + ["--phase", "tier_v3"])
    assert args.phase == "tier_v3"


def test_main_wires_tier_v3_phase_to_phase_tier_v3():
    import inspect
    src = inspect.getsource(K.main)
    assert '"tier_v3" in phases' in src
    assert "phase_tier_v3(lab)" in src


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
