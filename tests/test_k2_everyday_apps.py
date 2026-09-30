"""Dry-run tests for K2 pressure arm (c), everyday_apps: harness/browser_pressure.py (page generation, the
browser start/hold/clean-kill lifecycle) and harness/t2s_k2_pressure.py's wiring of that arm into run_k2_everyday_
apps_run/phase_k2_everyday_apps/everyday_apps_kill_criterion. No real browser, no real subprocess, no network: every
launch/kill is dependency-injected and faked."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import browser_pressure as bap  # noqa: E402
import t2s_k2_pressure as k2  # noqa: E402
import t2s_overnight as ov  # noqa: E402

from test_k2_pressure import StubLab, StubProc, StubServer, _mi  # noqa: E402  (reuse existing stubs, no duplicate path)


# ---------------------------------------------------------------- page generator (independently testable)

def test_generate_pressure_pages_writes_correct_page_count(tmp_path):
    paths = bap.generate_pressure_pages(tmp_path, n_pages=20, target_mb=150)
    assert len(paths) == 20
    assert all(p.exists() for p in paths)


def test_generate_pressure_pages_default_count_and_size_match_the_spec(tmp_path):
    paths = bap.generate_pressure_pages(tmp_path)
    assert len(paths) == bap.N_PAGES == 20
    assert bap.TARGET_MB_PER_PAGE == 150


def test_generate_pressure_pages_allocation_target_is_roughly_correct_per_page(tmp_path):
    target_mb = 150
    paths = bap.generate_pressure_pages(tmp_path, n_pages=3, target_mb=target_mb)
    for p in paths:
        text = p.read_text(encoding="utf-8")
        # The exact byte count the inline script computes must appear verbatim (target_mb * 1024 * 1024).
        assert str(target_mb * 1024 * 1024) in text
        assert "Float64Array" in text


def test_generate_pressure_pages_no_network_references():
    text = bap._PAGE_TEMPLATE.format(idx=0, target_mb=150, target_bytes=150 * 1024 * 1024)
    assert "http://" not in text and "https://" not in text
    assert "src=" not in text  # no external script/image/stylesheet references at all


def test_generate_pressure_pages_is_deterministic(tmp_path):
    out1, out2 = tmp_path / "a", tmp_path / "b"
    p1 = bap.generate_pressure_pages(out1, n_pages=5, target_mb=10)
    p2 = bap.generate_pressure_pages(out2, n_pages=5, target_mb=10)
    for a, b in zip(p1, p2):
        assert a.read_text(encoding="utf-8") == b.read_text(encoding="utf-8")


def test_generate_pressure_pages_pages_differ_only_by_index():
    t0 = bap._PAGE_TEMPLATE.format(idx=0, target_mb=150, target_bytes=150 * 1024 * 1024)
    t7 = bap._PAGE_TEMPLATE.format(idx=7, target_mb=150, target_bytes=150 * 1024 * 1024)
    assert t0 != t7
    assert "pressure page 0" in t0 and "__pressure_arr_0" in t0
    assert "pressure page 7" in t7 and "__pressure_arr_7" in t7


# ---------------------------------------------------------------- browser executable lookup (no real filesystem probing beyond candidates)

def test_find_browser_executable_returns_first_existing_candidate(tmp_path):
    fake_edge = tmp_path / "msedge.exe"
    fake_edge.write_text("stub")
    found = bap.find_browser_executable(candidates=[str(fake_edge)])
    assert found == str(fake_edge)


def test_find_browser_executable_returns_none_when_nothing_found():
    with patch.object(bap.shutil, "which", return_value=None), \
         patch.object(bap.Path, "exists", return_value=False):
        found = bap.find_browser_executable(candidates=[r"C:\definitely\not\a\real\path.exe"])
    assert found is None


def test_build_launch_cmd_uses_headless_flags_and_file_urls(tmp_path):
    p = tmp_path / "page.html"
    p.write_text("<html></html>")
    cmd = bap.build_launch_cmd("browser.exe", [p], tmp_path / "profile")
    assert "--headless" in cmd
    assert "--disable-gpu" in cmd
    assert any(part.startswith("file://") for part in cmd)
    assert not any(part.startswith("http://") or part.startswith("https://") for part in cmd)


# ---------------------------------------------------------------- browser lifecycle (dependency-injected, no real process)

class FakeProc:
    """Stands in for subprocess.Popen's return value: records nothing launched for real, just a pid and a
    poll() that starts 'running' until killed."""
    def __init__(self, pid=9999):
        self.pid = pid
        self._alive = True

    def poll(self):
        return None if self._alive else 0

    def kill_for_test(self):
        self._alive = False


def test_everyday_apps_pressure_start_launches_with_generated_pages(tmp_path):
    lab = StubLab(tmp_path)
    launched = {}

    def fake_launch(cmd):
        launched["cmd"] = cmd
        return FakeProc(pid=4321)

    pressure = bap.EverydayAppsPressure(lab, "tag1", executable="fake_browser.exe", launch_fn=fake_launch,
                                        kill_tree_fn=lambda pid: None)
    info = pressure.start(None)
    assert info["ok"] is True
    assert info["pid"] == 4321
    assert info["n_pages"] == bap.N_PAGES
    assert len(pressure.pages) == bap.N_PAGES
    assert "cmd" in launched


def test_everyday_apps_pressure_start_fails_cleanly_with_no_browser_found(tmp_path):
    lab = StubLab(tmp_path)
    pressure = bap.EverydayAppsPressure(lab, "tag1", executable_finder=lambda: None)
    info = pressure.start(None)
    assert info["ok"] is False
    assert info["pid"] is None


def test_everyday_apps_pressure_alive_reflects_process_state(tmp_path):
    lab = StubLab(tmp_path)
    proc = FakeProc()
    pressure = bap.EverydayAppsPressure(lab, "tag1", executable="fake.exe", launch_fn=lambda cmd: proc,
                                        kill_tree_fn=lambda pid: None)
    pressure.start(None)
    assert pressure.alive() is True
    proc.kill_for_test()
    assert pressure.alive() is False


def test_everyday_apps_pressure_stop_calls_injected_kill_tree_on_every_child(tmp_path):
    """Matches this repo's existing kill-tree convention (t2s_lab.Balloon.stop, tests/test_server_cleanup.py):
    stop() must call the injected kill_tree_fn with the browser process's own pid, which on a real host kills the
    whole process tree (taskkill /F /T), i.e. every child the browser spawned."""
    killed = []
    lab = StubLab(tmp_path)
    proc = FakeProc(pid=7777)
    pressure = bap.EverydayAppsPressure(lab, "tag1", executable="fake.exe", launch_fn=lambda cmd: proc,
                                        kill_tree_fn=lambda pid: killed.append(pid))
    pressure.start(None)
    result = pressure.stop()
    assert killed == [7777]
    assert result["stopped"] is True


def test_everyday_apps_pressure_stop_is_a_noop_if_never_started(tmp_path):
    lab = StubLab(tmp_path)
    killed = []
    pressure = bap.EverydayAppsPressure(lab, "tag1", kill_tree_fn=lambda pid: killed.append(pid))
    result = pressure.stop()
    assert killed == []
    assert result["stopped"] is False


# ---------------------------------------------------------------- turn scheduling (start at fixed turn, hold exactly N turns, clean-kill after)

class FakeEverydayAppsPressure:
    """Records every start()/stop()/alive() call with the turn it happened on (injected via a closure over a
    shared 'current_turn' box in the test), standing in for bap.EverydayAppsPressure -- no real browser."""

    def __init__(self, lab, tag):
        self.lab, self.tag = lab, tag
        self.calls = []
        self._alive = False

    def start(self, target_available_mb=None):
        self._alive = True
        self.calls.append(("start", None))
        return {"ok": True, "pid": 1234, "n_pages": bap.N_PAGES}

    def alive(self):
        return self._alive

    def stop(self):
        self._alive = False
        self.calls.append(("stop", None))
        return {"stopped": True}


def _fake_llama_server(lab_, mi_, n_ctx, tag="s"):
    return StubServer(lab_, mi_, n_ctx, tag=tag, script=[{"outcome": "ok", "output": "no fact here"}])


def test_run_k2_everyday_apps_run_starts_at_fixed_turn_and_holds_for_exactly_hold_turns(tmp_path, monkeypatch):
    lab = StubLab(tmp_path)
    mi = _mi(tmp_path)
    pressure_holder = {}

    def make_pressure(lab_, tag):
        p = FakeEverydayAppsPressure(lab_, tag)
        pressure_holder["p"] = p
        return p

    monkeypatch.setattr(k2.L, "Server", _fake_llama_server)
    monkeypatch.setattr(k2, "build_quality_task", lambda *a, **kw: ("prompt", "999999", "exact"))

    turn_rows = []
    orig = k2.run_everyday_apps_item

    def spy(lab_, srv, mi_, tag, runtime, turn_idx, browser_active, rep):
        row = orig(lab_, srv, mi_, tag, runtime, turn_idx, browser_active, rep)
        turn_rows.append((turn_idx, browser_active))
        return row

    monkeypatch.setattr(k2, "run_everyday_apps_item", spy)

    rows = k2.run_k2_everyday_apps_run(lab, mi, "llama31-8b", "llama_server", n_ctx=16384,
                                       start_turn=3, hold_turns=10, tail_turns=2,
                                       everyday_apps_cls=make_pressure)

    n_turns = 3 + 10 + 2
    assert len(rows) == n_turns
    p = pressure_holder["p"]
    assert p.calls[0] == ("start", None)
    assert p.calls[1] == ("stop", None)
    assert len(p.calls) == 2  # exactly one start, one stop -- never started/stopped a second time

    # browser_active must be False for turns [0, start_turn), True for [start_turn, start_turn+hold_turns),
    # and False again afterwards -- exactly hold_turns=10 active turns.
    active_turns = [t for t, active in turn_rows if active]
    assert active_turns == list(range(3, 3 + 10))
    assert len(active_turns) == 10


def test_run_k2_everyday_apps_run_kills_cleanly_even_if_server_dies_mid_hold(tmp_path, monkeypatch):
    """If the server dies while the browser is holding pressure, the browser must still be stopped (the 'safety
    net' branch in run_k2_everyday_apps_run's finally block)."""
    lab = StubLab(tmp_path)
    mi = _mi(tmp_path)
    pressure_holder = {}

    def make_pressure(lab_, tag):
        p = FakeEverydayAppsPressure(lab_, tag)
        pressure_holder["p"] = p
        return p

    # ok until turn 3 (browser start), then the server "dies" (alive_and_ours returns False) right after.
    def fake_server(lab_, mi_, n_ctx, tag="s"):
        srv = StubServer(lab_, mi_, n_ctx, tag=tag, script=[{"outcome": "ok", "output": "x"}] * 4)
        return srv

    monkeypatch.setattr(k2.L, "Server", fake_server)
    monkeypatch.setattr(k2, "build_quality_task", lambda *a, **kw: ("prompt", "999999", "exact"))

    real_alive_and_ours = StubServer.alive_and_ours
    call_count = {"n": 0}

    def flaky_alive_and_ours(self):
        call_count["n"] += 1
        if call_count["n"] > 4:
            return False, []
        return real_alive_and_ours(self)

    monkeypatch.setattr(StubServer, "alive_and_ours", flaky_alive_and_ours)

    rows = k2.run_k2_everyday_apps_run(lab, mi, "llama31-8b", "llama_server", n_ctx=16384,
                                       start_turn=1, hold_turns=10, tail_turns=2,
                                       everyday_apps_cls=make_pressure)
    p = pressure_holder["p"]
    assert not p.alive()  # stopped one way or another
    assert len(rows) < 13  # broke out early, not all 13 turns ran


# ---------------------------------------------------------------- Ollama adapter + effective-context signal

class FakeOllamaClient:
    def __init__(self, chat_response, ps_models=None):
        self._chat_response = chat_response
        self.ps_models = ps_models or []

    def chat(self, model, prompt, num_ctx=None, max_tokens=64, keep_alive=None):
        return self._chat_response

    def get_ps(self):
        return {"outcome": "ok", "models": self.ps_models}


def test_ollama_server_adapter_chat_ok_maps_fields():
    ollama = FakeOllamaClient({"outcome": "ok", "status": 200, "message": "hello", "eval_count": 40,
                              "duration_s": 2.0, "prompt_eval_count": 123})
    srv = k2.OllamaServerAdapter(ollama, "llama3.1:8b")
    res = srv.chat("prompt", 64, True)
    assert res["outcome"] == "ok"
    assert res["decode_tok_s"] == 20.0
    assert res["prompt_eval_count"] == 123
    assert res["ttft_s"] is None  # not observable through Ollama's non-streaming /api/chat


def test_ollama_server_adapter_chat_error_surfaces_http_status_and_error_text():
    ollama = FakeOllamaClient({"outcome": "http_error", "status": 500, "error": "internal server error"})
    srv = k2.OllamaServerAdapter(ollama, "llama3.1:8b")
    res = srv.chat("prompt", 64, True)
    assert res["outcome"] == "error"
    assert res["http_status"] == 500
    assert res["error"] == "internal server error"


def test_effective_context_signal_ollama_reads_api_ps():
    ollama = FakeOllamaClient({"outcome": "ok"}, ps_models=[{"name": "llama3.1:8b", "context_length": 8192}])
    srv = k2.OllamaServerAdapter(ollama, "llama3.1:8b")
    sig = k2.effective_context_signal("ollama", srv)
    assert sig["effective_ctx"] == 8192
    assert sig["source"] == "api_ps"


def test_effective_context_signal_llama_server_uses_configured_n_ctx(tmp_path):
    lab = StubLab(tmp_path)
    mi = _mi(tmp_path)
    srv = StubServer(lab, mi, 16384)
    sig = k2.effective_context_signal("llama_server", srv)
    assert sig["effective_ctx"] == 16384
    assert sig["source"] == "configured"


# ---------------------------------------------------------------- calibration-exclusion filtering

def test_phase_k2_everyday_apps_excludes_cwe_style_task_when_not_calibrated(tmp_path, monkeypatch):
    """The task requires excluding common_words_extraction/CWE-style tasks explicitly; this arm reuses the same
    per-task calibration_pass_set gate as phase_k2 -- confirm it actually blocks the whole arm when TASK_TYPE
    (K2's one task_type) is not in the pass set, the same way a CWE-only pass set would."""
    lab = StubLab(tmp_path)
    lab.models["llama31-8b"] = _mi(tmp_path, "llama31-8b")
    ran = []
    monkeypatch.setattr(k2, "run_k2_everyday_apps_run", lambda *a, **kw: ran.append(1) or [])

    # A pass set containing only the miscalibrated CWE-style task, never k2.TASK_TYPE itself.
    pass_set = {"common_words_extraction"}
    out = k2.phase_k2_everyday_apps(lab, calibration_pass_set=pass_set)
    assert out == []
    assert ran == []
    assert lab.rows[-1]["record"] == "k2_disabled"
    assert lab.rows[-1]["pressure_arm"] == k2.EVERYDAY_APPS_ARM


def test_phase_k2_everyday_apps_runs_when_task_type_is_calibration_passing(tmp_path, monkeypatch):
    lab = StubLab(tmp_path)
    for m in k2.EVERYDAY_APPS_MODELS:
        lab.models[m["model_id"]] = _mi(tmp_path, m["model_id"])
    monkeypatch.setattr(k2, "run_k2_everyday_apps_run", lambda *a, **kw: [
        {"score": 0.9, "browser_active": False, "outcome": "ok", "error": None, "turn_idx": 0}])
    out = k2.phase_k2_everyday_apps(lab, calibration_pass_set={k2.TASK_TYPE})
    assert len(out) == len(k2.EVERYDAY_APPS_MODELS) * len(k2.EVERYDAY_APPS_RUNTIMES)


def test_phase_k2_everyday_apps_unfiltered_when_calibration_pass_set_is_none(tmp_path, monkeypatch):
    lab = StubLab(tmp_path)
    for m in k2.EVERYDAY_APPS_MODELS:
        lab.models[m["model_id"]] = _mi(tmp_path, m["model_id"])
    monkeypatch.setattr(k2, "run_k2_everyday_apps_run", lambda *a, **kw: [
        {"score": 0.9, "browser_active": False, "outcome": "ok", "error": None, "turn_idx": 0}])
    out = k2.phase_k2_everyday_apps(lab, calibration_pass_set=None)
    assert len(out) == len(k2.EVERYDAY_APPS_MODELS) * len(k2.EVERYDAY_APPS_RUNTIMES)


def test_phase_k2_everyday_apps_skips_a_missing_model(tmp_path, monkeypatch):
    lab = StubLab(tmp_path)  # no models loaded at all
    calls = []
    monkeypatch.setattr(k2, "run_k2_everyday_apps_run", lambda *a, **kw: calls.append(1) or [])
    out = k2.phase_k2_everyday_apps(lab, calibration_pass_set=None)
    assert out == []
    assert calls == []
    assert all(r["record"] == "k2_disabled" for r in lab.rows)


# ---------------------------------------------------------------- everyday_apps_kill_criterion (pre-registered check)

def _turn(turn_idx, score, browser_active, outcome="ok", error=None):
    return {"turn_idx": turn_idx, "score": score, "browser_active": browser_active, "outcome": outcome,
            "error": error}


def test_kill_criterion_claim_supported_on_a_silent_drop():
    rows = [_turn(0, 0.9, False), _turn(1, 0.9, False),
            _turn(2, 0.9, True), _turn(3, 0.70, True)]  # 0.20 drop, outcome ok, no error: silent
    kc = k2.everyday_apps_kill_criterion(rows, score_drop_abs=0.10)
    assert kc["claim_supported"] is True
    assert kc["silent_drop_turns"] == [3]


def test_kill_criterion_claim_not_supported_when_every_drop_has_a_surfaced_error():
    rows = [_turn(0, 0.9, False), _turn(1, 0.9, False),
            _turn(2, 0.9, True), _turn(3, 0.2, True, outcome="error", error="connection reset")]
    kc = k2.everyday_apps_kill_criterion(rows, score_drop_abs=0.10)
    assert kc["claim_supported"] is False
    assert kc["silent_drop_turns"] == []


def test_kill_criterion_claim_not_supported_when_no_drop_occurs():
    rows = [_turn(0, 0.9, False), _turn(1, 0.9, True), _turn(2, 0.92, True)]
    kc = k2.everyday_apps_kill_criterion(rows, score_drop_abs=0.10)
    assert kc["claim_supported"] is False
    assert kc["silent_drop_turns"] == []


def test_kill_criterion_no_baseline_is_not_supported():
    rows = [_turn(0, 0.9, True), _turn(1, 0.2, True)]  # every row already under pressure, no baseline
    kc = k2.everyday_apps_kill_criterion(rows)
    assert kc["claim_supported"] is False
    assert "baseline" in kc["reason"]


# ---------------------------------------------------------------- dry-run call counts

def test_everyday_apps_dry_run_call_counts_matches_spec_default_constants():
    counts = k2.everyday_apps_dry_run_call_counts()
    expected_n_turns = k2.EVERYDAY_APPS_START_TURN + k2.EVERYDAY_APPS_HOLD_TURNS + k2.EVERYDAY_APPS_TAIL_TURNS
    assert counts["n_turns_per_combo"] == expected_n_turns
    assert counts["n_combinations"] == 4  # 2 models x 2 runtimes
    assert counts["total_calls"] == expected_n_turns * 4
    assert set(counts["per_combo"].values()) == {expected_n_turns}
    assert len(counts["per_combo"]) == 4


# ---------------------------------------------------------------- duplicate-keyword-argument guard (same class as test_k2_pressure.py)

def test_run_everyday_apps_item_extra_dict_has_no_reserved_do_call_keys():
    extra = {"k2_phase": "turn_5", "task_type": k2.TASK_TYPE, "everyday_apps_turn": 5,
             "everyday_apps_browser_active": True}
    reserved = {"n_ctx", "prompt_tokens", "mmap", "load_mode", "co_runner", "rep", "mem_headroom_gb", "load_s",
                "item_id", "kind", "warmup", "server_pid", "llama_build"}
    assert not (set(extra) & reserved)


def test_no_duplicate_kwargs_end_to_end_via_run_everyday_apps_item(tmp_path, monkeypatch):
    lab = StubLab(tmp_path)
    mi = _mi(tmp_path)
    srv = StubServer(lab, mi, 16384, script=[{"outcome": "ok", "output": "x"}])
    monkeypatch.setattr(k2, "build_quality_task", lambda *a, **kw: ("prompt", "999999", "exact"))
    try:
        row = k2.run_everyday_apps_item(lab, srv, mi, "tag", "llama_server", 0, False, 0)
    except TypeError as e:
        pytest.fail(f"duplicate-keyword-argument bug in run_everyday_apps_item: {e}")
    assert row["kind"] == "everyday_apps_turn"
