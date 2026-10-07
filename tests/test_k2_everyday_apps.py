"""Dry-run tests for K2 pressure arm (c), everyday_apps: harness/browser_pressure.py (page generation, the
browser start/hold/clean-kill lifecycle) and harness/t2s_k2_pressure.py's wiring of that arm into run_k2_everyday_
apps_run/phase_k2_everyday_apps/everyday_apps_kill_criterion. No real browser, no real subprocess, no network: every
launch/kill is dependency-injected and faked."""

from __future__ import annotations

import math
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


# ---------------------------------------------------------------- pages_for_total_mb (K2 arm (d) page scaling)

def test_pages_for_total_mb_zero_is_zero_pages():
    assert bap.pages_for_total_mb(0) == 0
    assert bap.pages_for_total_mb(-5) == 0


def test_pages_for_total_mb_exact_multiple():
    assert bap.pages_for_total_mb(8 * 1024, page_mb=150) == math.ceil(8 * 1024 / 150)


def test_pages_for_total_mb_rounds_up_and_is_at_least_one_page_when_positive():
    assert bap.pages_for_total_mb(1, page_mb=150) == 1


def test_pages_for_total_mb_scales_across_the_5_step_sweep():
    # 0/8/16/24/32 GB -> monotonically non-decreasing page counts, 0 GB -> 0 pages.
    counts = [bap.pages_for_total_mb(gb * 1024, page_mb=bap.TARGET_MB_PER_PAGE) for gb in (0, 8, 16, 24, 32)]
    assert counts[0] == 0
    assert counts == sorted(counts)
    assert all(c > 0 for c in counts[1:])


def test_pages_for_total_mb_default_page_size_matches_arm_c():
    assert bap.pages_for_total_mb(300) == math.ceil(300 / bap.TARGET_MB_PER_PAGE)


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


# ======================================================================================================
# K2 pressure arm (d), pause_resume (idle-unload-reload). No real Ollama, no real server.log, no real
# browser, no real wall-clock sleep: every dependency is dependency-injected or a plain tmp_path file.
# ======================================================================================================

class FakeOllamaPsClient:
    """FakeOllamaClient plus a get_ps() a caller can reconfigure between calls (run_pause_resume_run's own
    OllamaClient is constructed fresh per run, but capture_ollama_load_placement is called twice against the SAME
    client instance -- once at the initial load, once at the reload -- so tests need to change what get_ps()
    returns in between)."""

    def __init__(self, ps_models=None):
        self.ps_models = ps_models or []
        self.chat_calls = 0

    def chat(self, model, prompt, num_ctx=None, max_tokens=64, keep_alive=None, messages=None):
        self.chat_calls += 1
        return {"outcome": "ok", "status": 200, "message": "ok", "eval_count": 20, "duration_s": 1.0,
                "prompt_eval_count": 100}

    def get_ps(self):
        return {"outcome": "ok", "models": self.ps_models}


# ---------------------------------------------------------------- capture_ollama_load_placement (log + /api/ps)

def test_capture_ollama_load_placement_reads_layers_context_and_ps_fields(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("starting llama runner\n"
                    "llama_model_load: offloaded 32/32 layers to GPU\n"
                    "--ctx-size 8192 loaded\n", encoding="utf-8")
    ollama = FakeOllamaPsClient(ps_models=[{"name": "llama3.1:8b", "size": 5_000_000_000,
                                            "size_vram": 4_800_000_000, "context_length": 8192}])
    info = k2.capture_ollama_load_placement(ollama, "llama3.1:8b", since_pos=0, log_path_fn=lambda: str(log))
    assert info["layers_gpu"] == 32 and info["layers_total"] == 32
    assert info["size"] == 5_000_000_000 and info["size_vram"] == 4_800_000_000
    assert info["context_length_ps"] == 8192
    assert any("offloaded 32/32 layers to GPU" in l for l in info["server_log_placement_lines"])
    assert info["log_new_pos"] == log.stat().st_size


def test_capture_ollama_load_placement_respects_since_pos_for_the_reload_read(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("offloaded 32/32 layers to GPU\n", encoding="utf-8")
    pos_after_initial = log.stat().st_size
    with open(log, "a", encoding="utf-8") as f:
        f.write("offloaded 20/32 layers to GPU\n")  # the reload: fewer layers fit now
    ollama = FakeOllamaPsClient(ps_models=[])
    info = k2.capture_ollama_load_placement(ollama, "llama3.1:8b", since_pos=pos_after_initial,
                                             log_path_fn=lambda: str(log))
    assert info["layers_gpu"] == 20 and info["layers_total"] == 32  # only sees the NEW line, not the initial one


def test_capture_ollama_load_placement_no_log_path_is_graceful():
    ollama = FakeOllamaPsClient(ps_models=[])
    info = k2.capture_ollama_load_placement(ollama, "llama3.1:8b", log_path_fn=lambda: None)
    assert info["layers_gpu"] is None and info["layers_total"] is None
    assert info["server_log_placement_lines"] == []


def test_capture_ollama_load_placement_includes_telemetry_snapshot_when_lab_given(tmp_path):
    lab = StubLab(tmp_path)
    ollama = FakeOllamaPsClient(ps_models=[])
    info = k2.capture_ollama_load_placement(ollama, "llama3.1:8b", log_path_fn=lambda: None, lab=lab, t0=0, t1=1)
    assert info["telemetry"] is not None
    assert "shared_usage_mib" in info["telemetry"]
    # dedicated GPU usage is not exposed by t2s_lab.Telemetry.metrics() today -- flagged, not assumed.
    assert info["telemetry"]["dedicated_usage_mib"] is None
    assert "dedicated_usage_note" in info["telemetry"]


# ---------------------------------------------------------------- run_pause_resume_run lifecycle

class FakePressureForPauseResume:
    """Same shape as FakeEverydayAppsPressure above, but fails the test outright if constructed at all -- used to
    prove the 0 GB step never launches a browser (see test_run_pause_resume_run_0gb_never_starts_a_browser)."""

    def __init__(self, lab, tag, n_pages=0, target_mb=150):
        raise AssertionError("EverydayAppsPressure must not be constructed at all for the 0 GB (control) step")


class _SafeFakePressure:
    """Default everyday_apps_cls for the fixture below: never launches a real subprocess (no real browser, no real
    process tree), used whenever a test does not care about the pressure object itself."""

    def __init__(self, lab, tag, n_pages=0, target_mb=150):
        self._alive = False

    def start(self, x):
        self._alive = True
        return {"ok": True, "pid": 1}

    def alive(self):
        return self._alive

    def stop(self):
        self._alive = False


class ResidencyFakeOllama(FakeOllamaPsClient):
    """FakeOllamaPsClient with Ollama's real residency behavior modeled: a chat() loads the model (it shows in
    /api/ps), and expire() (called by the fixture's sleep_fn when the idle wait elapses, i.e. keep_alive ran out)
    unloads it. never_expire=True models a keep_alive that did NOT run out (the unload-not-confirmed case)."""

    def __init__(self, ps_models=None, never_expire=False, on_chat=None):
        super().__init__(ps_models=ps_models)
        self.resident = False
        self.never_expire = never_expire
        self.on_chat = on_chat

    def chat(self, model, prompt, num_ctx=None, max_tokens=64, keep_alive=None, messages=None, **kw):
        was_resident = self.resident
        self.resident = True
        self.last_kwargs = dict(keep_alive=keep_alive, **kw)
        if self.on_chat is not None and not was_resident:
            self.on_chat(self)  # called only on a real (re)load, the way Ollama writes its load lines
        return super().chat(model, prompt, num_ctx=num_ctx, max_tokens=max_tokens, keep_alive=keep_alive)

    def expire(self):
        if not self.never_expire:
            self.resident = False

    def get_ps(self):
        return {"outcome": "ok", "models": self.ps_models if self.resident else []}


def _make_pause_resume_fixture(tmp_path, app_load_gb, everyday_apps_cls=_SafeFakePressure, log_path=None,
                               ollama=None, **extra):
    lab = StubLab(tmp_path)
    mi = _mi(tmp_path, "llama31-8b")
    ollama = ollama or ResidencyFakeOllama(
        ps_models=[{"name": "llama3.1:8b", "size": 1, "size_vram": 1, "context_length": 4096}])
    sleeps = []

    def sleep_fn(s):
        sleeps.append(s)
        if s == k2.PAUSE_RESUME_IDLE_S:
            ollama.expire()

    kwargs = dict(turns_before=3, turns_after=4, idle_s=k2.PAUSE_RESUME_IDLE_S, sleep_fn=sleep_fn,
                  ollama_client_cls=lambda: ollama, log_path_fn=lambda: log_path, everyday_apps_cls=everyday_apps_cls,
                  avail_mb_fn=lambda: 100000.0)
    kwargs.update(extra)
    # do_call's own 1.5 s post-call telemetry settle is real wall-clock; skipped here (no telemetry to settle).
    with patch.object(ov.time, "sleep", lambda s: None):
        res = k2.run_pause_resume_run(lab, mi, "llama31-8b", "llama3.1:8b", app_load_gb, **kwargs)
    return lab, res, sleeps, ollama


def test_run_pause_resume_run_0gb_never_starts_a_browser(tmp_path):
    lab, res, sleeps, ollama = _make_pause_resume_fixture(tmp_path, 0, everyday_apps_cls=FakePressureForPauseResume)
    assert res["app_load_gb"] == 0
    assert not any(r.get("record") == "k2_pause_resume_browser_start" for r in lab.rows)
    assert sleeps == [k2.PAUSE_RESUME_IDLE_S]  # the idle wait still happens even with no app load


def test_run_pause_resume_run_positive_gb_starts_and_stops_the_browser(tmp_path):
    started = {}

    class RecordingPressure:
        def __init__(self, lab, tag, n_pages=0, target_mb=150):
            started["n_pages"] = n_pages
            self._alive = False

        def start(self, x):
            self._alive = True
            return {"ok": True, "pid": 1}

        def alive(self):
            return self._alive

        def stop(self):
            self._alive = False
            started["stopped"] = True

    lab, res, sleeps, ollama = _make_pause_resume_fixture(tmp_path, 8, everyday_apps_cls=RecordingPressure)
    assert started["n_pages"] == bap.pages_for_total_mb(8 * 1024, k2.PAUSE_RESUME_PAGE_MB)
    assert started.get("stopped") is True
    assert any(r.get("record") == "k2_pause_resume_browser_start" for r in lab.rows)


def test_run_pause_resume_run_uses_ollamas_default_keep_alive_not_zero(tmp_path):
    lab, res, sleeps, ollama = _make_pause_resume_fixture(tmp_path, 0)
    assert res["keep_alive"] == k2.PAUSE_RESUME_KEEP_ALIVE
    assert res["keep_alive"] != 0 and res["keep_alive"] != "0"
    assert all(ev["keep_alive"] == k2.PAUSE_RESUME_KEEP_ALIVE for ev in res["load_events"])


def _appending_loader(log, lines_per_load):
    """on_chat hook for ResidencyFakeOllama: each real load appends the next entry of lines_per_load to the fake
    server.log, the way Ollama writes one block of load lines per model load."""
    loads = iter(lines_per_load)

    def on_chat(_ollama):
        with open(log, "a", encoding="utf-8") as f:
            f.write(next(loads))
    return on_chat


def test_run_pause_resume_run_captures_placement_at_both_the_initial_load_and_the_reload(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("stale line from an earlier session: offloaded 99/99 layers to GPU\n", encoding="utf-8")
    ollama = ResidencyFakeOllama(ps_models=[{"name": "llama3.1:8b", "size": 1, "size_vram": 1, "context_length": 4096}],
                                 on_chat=_appending_loader(log, ["offloaded 32/32 layers to GPU\n--ctx-size 8192\n",
                                                                 "offloaded 32/32 layers to GPU\n--ctx-size 8192\n"]))
    lab, res, sleeps, ollama = _make_pause_resume_fixture(tmp_path, 8, log_path=str(log), ollama=ollama)
    assert len(res["load_events"]) == 2
    initial = next(e for e in res["load_events"] if e["load_event"] == "initial")
    reload = next(e for e in res["load_events"] if e["load_event"] == "reload")
    assert initial["turn_idx"] == 0 and reload["turn_idx"] == 3  # turns_before=3 in the fixture
    # each read sees only its own load's lines: never the stale 99/99 line written before the step started
    assert initial["layers_gpu"] == 32 and reload["layers_gpu"] == 32
    assert initial["size_vram"] == 1 and reload["size_vram"] == 1
    assert initial["context_length_ps"] == 4096 and reload["context_length_ps"] == 4096


def test_run_pause_resume_run_idles_for_exactly_idle_s_between_app_load_and_resume(tmp_path):
    lab, res, sleeps, ollama = _make_pause_resume_fixture(tmp_path, 16)
    assert sleeps == [k2.PAUSE_RESUME_IDLE_S]


def test_run_pause_resume_run_turn_count_matches_before_plus_after(tmp_path):
    lab, res, sleeps, ollama = _make_pause_resume_fixture(tmp_path, 0)
    assert len(res["rows"]) == 3 + 4  # turns_before=3, turns_after=4 in the fixture


def test_run_pause_resume_run_default_keep_alive_constant_is_5_minutes_not_0():
    """OLLAMA_KEEP_ALIVE=0 is K1/K2's usual convention everywhere else (contamination avoidance); this arm is a
    deliberate, documented exception (see docs/FINDINGS.md's pre-registration)."""
    assert k2.PAUSE_RESUME_KEEP_ALIVE == "5m"


# ---------------------------------------------------------------- pause_resume_report (per-step table, one call)

def _load_event(tag, load_event, turn_idx, layers_gpu, layers_total, ctx):
    return {"item_tag": tag, "load_event": load_event, "turn_idx": turn_idx, "layers_gpu": layers_gpu,
            "layers_total": layers_total, "context_length_ps": ctx, "context_length_log": ctx}


def _pr_row(turn_idx, score, ttft, decode, outcome="ok", error=None):
    return {"turn_idx": turn_idx, "score": score, "ttft_s": ttft, "decode_tok_s": decode, "outcome": outcome,
            "error": error}


def test_pause_resume_report_produces_one_row_per_step_with_expected_shape():
    step_results = [
        {"tag": "t_0gb", "model_id": "llama31-8b", "app_load_gb": 0, "keep_alive": "5m", "turns_before": 3,
         "rows": [_pr_row(0, 0.9, 0.1, 30), _pr_row(1, 0.9, 0.1, 30), _pr_row(2, 0.9, 0.1, 30),
                  _pr_row(3, 0.9, 0.1, 30), _pr_row(4, 0.9, 0.1, 30)],
         "load_events": [_load_event("t_0gb", "initial", 0, 32, 32, 4096), _load_event("t_0gb", "reload", 3, 32, 32, 4096)]},
        {"tag": "t_32gb", "model_id": "llama31-8b", "app_load_gb": 32, "keep_alive": "5m", "turns_before": 3,
         "rows": [_pr_row(0, 0.9, 0.1, 30), _pr_row(1, 0.9, 0.1, 30), _pr_row(2, 0.9, 0.1, 30),
                  _pr_row(3, 0.6, 0.4, 10), _pr_row(4, 0.6, 0.4, 10)],
         "load_events": [_load_event("t_32gb", "initial", 0, 32, 32, 4096),
                         _load_event("t_32gb", "reload", 3, 20, 32, 2048)]},
    ]
    report = k2.pause_resume_report(step_results)
    assert len(report) == 2
    r0, r32 = report[0], report[1]
    assert r0["app_load_gb"] == 0 and r32["app_load_gb"] == 32
    assert r0["placement_changed"] is False and r0["context_changed"] is False
    assert r32["placement_changed"] is True and r32["context_changed"] is True
    assert r32["quality_score_before_median"] == 0.9 and r32["quality_score_after_median"] == 0.6
    assert r32["ttft_s_before_median"] == 0.1 and r32["ttft_s_after_median"] == 0.4
    assert r32["decode_tok_s_before_median"] == 30 and r32["decode_tok_s_after_median"] == 10
    for key in ("placement_before", "placement_after", "context_before", "context_after", "errors"):
        assert key in r0


def test_pause_resume_report_records_surfaced_errors():
    step_results = [{"tag": "t", "model_id": "m", "app_load_gb": 8, "keep_alive": "5m", "turns_before": 1,
                     "rows": [_pr_row(0, 0.9, 0.1, 30), _pr_row(1, None, None, None, outcome="error", error="boom")],
                     "load_events": []}]
    report = k2.pause_resume_report(step_results)
    assert len(report[0]["errors"]) == 1
    assert report[0]["errors"][0]["error"] == "boom"


def test_pause_resume_report_handles_missing_load_events_gracefully():
    step_results = [{"tag": "t", "model_id": "m", "app_load_gb": 0, "keep_alive": "5m", "turns_before": 1,
                     "rows": [_pr_row(0, 0.9, 0.1, 30)], "load_events": []}]
    report = k2.pause_resume_report(step_results)
    assert report[0]["placement_before"]["layers_gpu"] is None
    assert report[0]["placement_changed"] is False


# ---------------------------------------------------------------- pause_resume_prediction_verdict

def test_prediction_verdict_p1_when_nothing_changes():
    step = {"placement_before": {"layers_gpu": 32}, "placement_after": {"layers_gpu": 32},
            "placement_changed": False, "context_changed": False, "errors": []}
    assert k2.pause_resume_prediction_verdict(step) == "P1"


def test_prediction_verdict_p2_when_placement_changes_with_no_error():
    step = {"placement_before": {"layers_gpu": 32}, "placement_after": {"layers_gpu": 20},
            "placement_changed": True, "context_changed": False, "errors": []}
    assert k2.pause_resume_prediction_verdict(step) == "P2"


def test_prediction_verdict_inconclusive_when_an_error_surfaced():
    step = {"placement_before": {"layers_gpu": 32}, "placement_after": {"layers_gpu": 20},
            "placement_changed": True, "context_changed": False,
            "errors": [{"turn_idx": 4, "error": "connection reset"}]}
    assert k2.pause_resume_prediction_verdict(step) == "inconclusive"


def test_prediction_verdict_inconclusive_when_placement_data_missing():
    step = {"placement_before": {"layers_gpu": None}, "placement_after": {"layers_gpu": None},
            "placement_changed": False, "context_changed": False, "errors": []}
    assert k2.pause_resume_prediction_verdict(step) == "inconclusive"


# ---------------------------------------------------------------- 0GB-step-is-the-control (explicit conclusion)

def test_0gb_step_is_the_control_not_a_separate_arm_variant(tmp_path):
    """Direct test of this file's own stated conclusion (docs/FINDINGS.md's 'Control' paragraph): the 0 GB step
    takes the IDENTICAL run_pause_resume_run code path as every other step (same turn counts, same keep_alive, same
    idle sleep_fn call) and is distinguished only by never constructing/starting EverydayAppsPressure at all -- so
    no separate 'no app load' arm variant exists or is needed in this codebase."""
    lab, res, sleeps, ollama = _make_pause_resume_fixture(tmp_path, 0, everyday_apps_cls=FakePressureForPauseResume)
    # Proves identical lifecycle: turns run, placement captured at both loads, idle wait happens -- exactly like
    # any other step -- yet FakePressureForPauseResume (which raises if constructed) was never triggered.
    assert len(res["rows"]) == 3 + 4
    assert len(res["load_events"]) == 2
    assert sleeps == [k2.PAUSE_RESUME_IDLE_S]
    assert res["app_load_gb"] == 0


# ---------------------------------------------------------------- phase_k2_pause_resume

def test_phase_k2_pause_resume_runs_every_model_x_step_combination(tmp_path, monkeypatch):
    lab = StubLab(tmp_path)
    for m in k2.EVERYDAY_APPS_MODELS:
        lab.models[m["model_id"]] = _mi(tmp_path, m["model_id"])
    calls = []

    def fake_run(lab_, mi_, model_id, ollama_model, app_load_gb, **kw):
        calls.append((model_id, app_load_gb))
        return {"tag": f"{model_id}_{app_load_gb}", "model_id": model_id, "app_load_gb": app_load_gb,
                "keep_alive": "5m", "turns_before": 10, "rows": [], "load_events": []}

    monkeypatch.setattr(k2, "run_pause_resume_run", fake_run)
    out = k2.phase_k2_pause_resume(lab, calibration_pass_set=None)
    assert len(calls) == len(k2.EVERYDAY_APPS_MODELS) * len(k2.PAUSE_RESUME_APP_LOAD_STEPS_GB)
    assert set(gb for _, gb in calls) == set(k2.PAUSE_RESUME_APP_LOAD_STEPS_GB)
    assert len(out) == len(calls)
    assert any(r.get("record") == "k2_pause_resume_step_report" for r in lab.rows)


def test_phase_k2_pause_resume_skips_when_task_type_not_calibrated(tmp_path, monkeypatch):
    lab = StubLab(tmp_path)
    ran = []
    monkeypatch.setattr(k2, "run_pause_resume_run", lambda *a, **kw: ran.append(1) or {})
    out = k2.phase_k2_pause_resume(lab, calibration_pass_set=set())
    assert out == []
    assert ran == []
    assert lab.rows[-1]["record"] == "k2_disabled"


def test_phase_k2_pause_resume_skips_a_missing_model(tmp_path, monkeypatch):
    lab = StubLab(tmp_path)  # no models loaded
    calls = []
    monkeypatch.setattr(k2, "run_pause_resume_run", lambda *a, **kw: calls.append(1) or {})
    out = k2.phase_k2_pause_resume(lab, calibration_pass_set=None)
    assert out == []
    assert calls == []
    assert all(r["record"] == "k2_disabled" for r in lab.rows)


# ---------------------------------------------------------------- pause_resume_dry_run_call_shape

def test_pause_resume_dry_run_call_shape_matches_5_step_x_2_machine_design():
    shape = k2.pause_resume_dry_run_call_shape()
    assert shape["n_app_load_steps"] == 5
    assert shape["n_machines"] == 2
    n_turns = k2.PAUSE_RESUME_TURNS_BEFORE + k2.PAUSE_RESUME_TURNS_AFTER
    assert shape["n_turns_per_run"] == n_turns
    assert shape["total_runs"] == 5 * len(k2.EVERYDAY_APPS_MODELS) * 2
    assert shape["pause_resume_calls"] == shape["total_runs"] * n_turns
    assert len(shape["per_run"]) == shape["total_runs"]


def test_pause_resume_dry_run_call_shape_with_vs_without_existing_arms():
    shape = k2.pause_resume_dry_run_call_shape()
    assert shape["total_calls_with_existing_arms"] > shape["total_calls_without_existing_arms"]
    assert shape["total_calls_without_existing_arms"] == shape["pause_resume_calls"]
    expected_everyday_apps_both_machines = k2.everyday_apps_dry_run_call_counts()["total_calls"] * 2
    assert (shape["total_calls_with_existing_arms"] - shape["total_calls_without_existing_arms"]
            == expected_everyday_apps_both_machines)


# ======================================================================================================
# Standalone arm (d) job (2026-10-06): --pause-resume-only, explicit no-app control label, idle-unload
# verification, fresh initial load per step, and the mechanical P1/P2/inconclusive design check end to end.
# ======================================================================================================

class FailingTurnOllama(ResidencyFakeOllama):
    """Returns an error on the chat call numbers listed in fail_calls (1-based), the way a real Ollama 500 would."""

    def __init__(self, fail_calls, **kw):
        super().__init__(**kw)
        self.fail_calls = set(fail_calls)
        self.n = 0

    def chat(self, model, prompt, **kw):
        self.n += 1
        if self.n in self.fail_calls:
            self.resident = True
            return {"outcome": "http_error", "status": 500, "error": "model runner crashed", "duration_s": 0.1}
        return super().chat(model, prompt, **kw)


_PS = [{"name": "llama3.1:8b", "size": 1, "size_vram": 1, "context_length": 4096}]


def test_standalone_0gb_step_is_labeled_as_the_no_app_reload_control(tmp_path):
    lab, res, sleeps, ollama = _make_pause_resume_fixture(tmp_path, 0, everyday_apps_cls=FakePressureForPauseResume)
    assert res["condition"] == k2.PAUSE_RESUME_CONTROL_LABEL == "no_app_reload_control"
    labeled = [r for r in lab.rows if str(r.get("record", "")).startswith("k2_pause_resume")]
    assert labeled and all(r.get("condition") == "no_app_reload_control" for r in labeled)
    step = k2.pause_resume_report([res])[0]
    assert step["condition"] == "no_app_reload_control"


def test_standalone_positive_gb_step_is_labeled_app_load(tmp_path):
    lab, res, sleeps, ollama = _make_pause_resume_fixture(tmp_path, 24)
    assert res["condition"] == "app_load"
    assert k2.pause_resume_report([res])[0]["condition"] == "app_load"


def test_default_steps_include_the_0gb_control():
    assert k2.PAUSE_RESUME_APP_LOAD_STEPS_GB == (0, 8, 16, 24, 32)
    assert k2.pause_resume_condition(0) == k2.PAUSE_RESUME_CONTROL_LABEL


def test_unload_is_confirmed_after_the_idle_wait_when_keep_alive_expires(tmp_path):
    lab, res, sleeps, ollama = _make_pause_resume_fixture(tmp_path, 8)
    assert res["unload_check"]["unload_confirmed"] is True
    assert res["unload_check"]["extra_wait_s"] == 0.0
    assert sleeps == [k2.PAUSE_RESUME_IDLE_S]  # no extra poll sleeps were needed
    assert any(r.get("record") == "k2_pause_resume_unload_check" and r.get("unload_confirmed") is True
               for r in lab.rows)


def test_unload_not_confirmed_polls_for_the_grace_period_and_forces_inconclusive(tmp_path):
    ollama = ResidencyFakeOllama(ps_models=_PS, never_expire=True)
    lab, res, sleeps, _ = _make_pause_resume_fixture(tmp_path, 8, ollama=ollama, unload_grace_s=60.0,
                                                     unload_poll_s=15.0)
    assert res["unload_check"]["unload_confirmed"] is False
    assert sleeps == [k2.PAUSE_RESUME_IDLE_S, 15.0, 15.0, 15.0, 15.0]
    step = k2.pause_resume_report([res])[0]
    step["placement_before"] = {"layers_gpu": 32}
    step["placement_after"] = {"layers_gpu": 32}
    assert k2.pause_resume_prediction_verdict(step) == "inconclusive"


def test_wait_for_unload_unknown_when_ps_never_answers():
    class NoPs:
        def get_ps(self):
            raise ConnectionError("refused")
    slept = []
    out = k2.wait_for_unload(NoPs(), "llama3.1:8b", grace_s=30.0, poll_s=10.0, sleep_fn=slept.append)
    assert out["unload_confirmed"] is None
    assert slept == [10.0, 10.0, 10.0]


def test_each_step_requests_a_fresh_unload_before_turn_1(tmp_path):
    calls = []
    lab, res, sleeps, ollama = _make_pause_resume_fixture(
        tmp_path, 0, unload_fn=lambda o, tag: calls.append(tag) or {"ok": True})
    assert calls == ["llama3.1:8b"]
    start = next(r for r in lab.rows if r.get("record") == "k2_pause_resume_step_start")
    assert start["pre_unload"] == {"ok": True} and start["pre_unload_confirmed"] is True


def test_ollama_unload_request_is_a_noop_for_a_client_without_a_base_url():
    assert k2.ollama_unload_request(object(), "llama3.1:8b")["ok"] is None


def test_think_is_omitted_for_non_thinking_models(tmp_path):
    lab, res, sleeps, ollama = _make_pause_resume_fixture(tmp_path, 0)
    assert "think" not in ollama.last_kwargs
    assert res["think_requested"] is None
    turn_rows = [r for r in lab.rows if r.get("kind") == "everyday_apps_turn"]
    assert turn_rows and all(r["think_requested"] is None and r["think_tag"] is False for r in turn_rows)


def test_think_false_is_sent_and_recorded_when_set(tmp_path):
    lab, res, sleeps, ollama = _make_pause_resume_fixture(tmp_path, 0, think=False)
    assert ollama.last_kwargs["think"] is False
    assert res["think_requested"] is False
    assert k2.pause_resume_report([res])[0]["think_requested"] is False


def test_pause_resume_models_match_the_preregistration():
    tags = {m["model_id"]: m["ollama_model"] for m in k2.PAUSE_RESUME_MODELS}
    assert tags == {"llama31-8b": "llama3.1:8b", "qwen3-4b-2507": "qwen3:4b-instruct-2507"}
    assert all("think" in m for m in k2.PAUSE_RESUME_MODELS)


def test_adapter_reports_ollama_load_duration_and_thinking_fields():
    class RawClient:
        def chat(self, model, prompt, **kw):
            return {"outcome": "ok", "status": 200, "message": "<think>x</think> 123", "eval_count": 10,
                    "duration_s": 2.0, "prompt_eval_count": 50,
                    "raw": {"message": {"content": "123", "thinking": "hmm"}, "load_duration": 3_000_000_000,
                            "prompt_eval_duration": 1_500_000_000, "eval_duration": 500_000_000}}
    srv = k2.OllamaServerAdapter(RawClient(), "llama3.1:8b", keep_alive="5m")
    out = srv.chat("p", 16, True)
    assert out["ollama_load_duration_s"] == 3.0 and out["ollama_prompt_eval_duration_s"] == 1.5
    assert out["think_tag"] is True and out["thinking_field_present"] is True


# ---------------------------------------------------------------- mechanical verdict design check, end to end

def _run_step(tmp_path, gb, initial_line, reload_line, ollama_cls=ResidencyFakeOllama, **ollama_kw):
    log = tmp_path / f"server_{gb}.log"
    log.write_text("", encoding="utf-8")
    ollama = ollama_cls(ps_models=_PS, on_chat=_appending_loader(log, [initial_line, reload_line]), **ollama_kw)
    sub = tmp_path / f"step_{gb}"
    sub.mkdir()
    lab, res, sleeps, _ = _make_pause_resume_fixture(sub, gb, log_path=str(log), ollama=ollama)
    return res


def test_verdict_design_check_produces_p1_p2_and_inconclusive_mechanically(tmp_path):
    """The pre-registration requires the per-step P1/P2/inconclusive determination to be mechanical. Drive three
    real run_pause_resume_run steps (fakes only) through pause_resume_report and pause_resume_prediction_verdict:
    an unchanged reload (P1), a reload with fewer GPU layers and no error (P2), and the same placement change with
    an error surfaced after the reload (inconclusive)."""
    same = _run_step(tmp_path, 0, "offloaded 33/33 layers to GPU\n", "offloaded 33/33 layers to GPU\n")
    fewer = _run_step(tmp_path, 32, "offloaded 33/33 layers to GPU\n", "offloaded 20/33 layers to GPU\n")
    # 3 turns before + the reload turn succeed, then chat call 6 (turn idx 5) errors
    errored = _run_step(tmp_path, 16, "offloaded 33/33 layers to GPU\n", "offloaded 20/33 layers to GPU\n",
                        ollama_cls=FailingTurnOllama, fail_calls=[6])
    report = k2.pause_resume_report([same, fewer, errored])
    verdicts = {r["app_load_gb"]: k2.pause_resume_prediction_verdict(r) for r in report}
    assert verdicts == {0: "P1", 32: "P2", 16: "inconclusive"}
    by_gb = {r["app_load_gb"]: r for r in report}
    assert by_gb[0]["condition"] == "no_app_reload_control"
    assert by_gb[32]["placement_before"]["layers_gpu"] == 33 and by_gb[32]["placement_after"]["layers_gpu"] == 20
    assert by_gb[16]["errors"] and by_gb[16]["errors"][0]["turn_idx"] == 5


# ---------------------------------------------------------------- phase_k2_pause_resume standalone path

def test_phase_standalone_runs_all_steps_with_injected_fakes_and_marks_each_step_done(tmp_path):
    lab = StubLab(tmp_path)
    for m in k2.PAUSE_RESUME_MODELS:
        lab.models[m["model_id"]] = _mi(tmp_path, m["model_id"])
    clients = []

    def factory():
        c = ResidencyFakeOllama(ps_models=[{"name": "llama3.1:8b"}, {"name": "qwen3:4b-instruct-2507"}])
        clients.append(c)
        return c

    def sleep_fn(s):
        if s == k2.PAUSE_RESUME_IDLE_S:
            clients[-1].expire()

    run_kwargs = dict(turns_before=2, turns_after=2, sleep_fn=sleep_fn, ollama_client_cls=factory,
                      log_path_fn=lambda: None, everyday_apps_cls=_SafeFakePressure, avail_mb_fn=lambda: 1.0)
    with patch.object(ov.time, "sleep", lambda s: None):
        out = k2.phase_k2_pause_resume(lab, calibration_pass_set=None, run_kwargs=run_kwargs)
    assert len(out) == 2 * 5
    reports = [r for r in lab.rows if r.get("record") == "k2_pause_resume_step_report"]
    assert len(reports) == 10
    assert sum(r["condition"] == "no_app_reload_control" for r in reports) == 2  # one control per model
    assert all(r["prediction"] in ("P1", "P2", "inconclusive") for r in reports)
    assert all(r["unload_confirmed"] is True for r in reports)
    assert "k2_pause_resume_llama31-8b_0gb_done" in lab.done
    # resume: a second call with every step already done runs nothing
    assert k2.phase_k2_pause_resume(lab, calibration_pass_set=None, run_kwargs=run_kwargs) == []


def test_phase_standalone_respects_model_and_step_subsets(tmp_path, monkeypatch):
    lab = StubLab(tmp_path)
    for m in k2.PAUSE_RESUME_MODELS:
        lab.models[m["model_id"]] = _mi(tmp_path, m["model_id"])
    calls = []

    def fake_run(lab_, mi_, model_id, ollama_model, app_load_gb, **kw):
        calls.append((model_id, app_load_gb, kw.get("think"), kw.get("marker")))
        return {"tag": "t", "model_id": model_id, "app_load_gb": app_load_gb, "keep_alive": "5m",
                "turns_before": 10, "rows": [], "load_events": []}

    monkeypatch.setattr(k2, "run_pause_resume_run", fake_run)
    models = (k2.PAUSE_RESUME_MODELS[1],)
    k2.phase_k2_pause_resume(lab, models=models, app_load_steps_gb=(0, 32), run_kwargs={"marker": 1})
    assert calls == [("qwen3-4b-2507", 0, None, 1), ("qwen3-4b-2507", 32, None, 1)]


def test_main_pause_resume_only_skips_arms_a_b_and_everyday_apps():
    import inspect
    src = inspect.getsource(k2.main)
    assert "--pause-resume-only" in src
    assert "model_ids = [] if args.pause_resume_only" in src
    assert 'stem_prefix="t2s_k2_pause_resume" if args.pause_resume_only' in src
    assert "args.everyday_apps = False" in src
    assert "wait_for_ollama_ready" in src
    assert 'run_kwargs={"log_path_fn": owned_ollama_log_path}' in src


def test_owned_ollama_log_path_prefers_the_owned_serve_log(tmp_path, monkeypatch):
    owned = tmp_path / "ollama_serve.log"
    owned.write_text("x", encoding="utf-8")
    monkeypatch.setattr(k2, "OWNED_OLLAMA_SERVE_LOG", str(owned))
    assert k2.owned_ollama_log_path() == str(owned)
    monkeypatch.setattr(k2, "OWNED_OLLAMA_SERVE_LOG", str(tmp_path / "missing.log"))
    monkeypatch.setattr(k2.k1, "find_ollama_log", lambda: "fallback")
    assert k2.owned_ollama_log_path() == "fallback"


def test_ollama_tag_overrides_replace_only_named_models_and_keep_the_default():
    out = k2.apply_ollama_tag_overrides(k2.PAUSE_RESUME_MODELS, "qwen3-4b-2507=qwen3-4b-2507")
    by = {m["model_id"]: m for m in out}
    assert by["qwen3-4b-2507"]["ollama_model"] == "qwen3-4b-2507"
    assert by["qwen3-4b-2507"]["ollama_model_default"] == "qwen3:4b-instruct-2507"
    assert by["llama31-8b"]["ollama_model"] == "llama3.1:8b" and "ollama_model_default" not in by["llama31-8b"]
    assert k2.PAUSE_RESUME_MODELS[1]["ollama_model"] == "qwen3:4b-instruct-2507"  # module constant untouched
    assert k2.apply_ollama_tag_overrides(k2.PAUSE_RESUME_MODELS, None) == k2.PAUSE_RESUME_MODELS


def test_pause_resume_duration_estimate_arithmetic():
    est = k2.pause_resume_duration_estimate_s({"a": 10.0}, app_load_steps_gb=(0, 8), turns_before=1, turns_after=1,
                                              idle_s=100.0, unload_grace_s=50.0, per_step_overhead_s=20.0,
                                              per_turn_overhead_s=0.0)
    assert est["per_model_s"]["a"] == 2 * (2 * 10.0 + 100.0 + 20.0)
    assert est["worst_case_total_s"] == est["total_s"] + 2 * 50.0
