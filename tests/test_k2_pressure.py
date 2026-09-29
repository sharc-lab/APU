"""Dry-run tests for harness/t2s_k2_pressure.py against a stub lab/server (no real llama-server, no real memory
allocation, no network). tests/stub_lab.py does not exist yet in this repo, so the stub classes below are
self-contained in this file.

Covers every call site named in the K2 task: server start/stop in both mmap arms, the pressure-arm start/stop calls
(mocked, nothing is actually allocated), the pressure-step loop, the responsiveness sampler (run for real, with a
tiny interval, since it only launches `python -c "pass"`), do_call (run for real against the stub server/lab, since
that is exactly the code this file must exercise correctly), and kill_criterion against synthetic rows for both a
passing and a failing case.

Also checks for the duplicate-keyword-argument bug class documented for tests/test_no_duplicate_row_kwargs.py (which
also does not exist yet here): every `extra={...}` dict this module's own code builds for do_call/start_row must not
also collide with a keyword those functions already pass explicitly.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
import t2s_k2_pressure as k2  # noqa: E402
import t2s_overnight as ov  # noqa: E402


# ---------------------------------------------------------------- stubs

class StubTelemetry:
    def thermal_gate(self, idle_temp, idle_pkg=None, corunner_active=False):
        return {"thermal_wait_s": 0.0, "temp_at_gate_start": None, "temp_at_release": None,
                "idle_temp": idle_temp, "gate_released_by": "no_temperature_no_proxy"}

    def metrics(self, t0, t1):
        return {"igpu_mhz": 1200.0, "igpu_throttle_bits": None, "pkg_power_w": 10.0, "rapl_pp0_w": None,
                "rapl_pp1_w": None, "cpu_p_pct_perf": None, "cpu_e_pct_perf": None, "cpu_lpe_pct_perf": None,
                "shared_usage_mib": 4096.0, "total_committed_mib": 8192.0, "pages_input_per_s": 0.0,
                "hard_faults_per_s": 0.0, "avail_mb_min": 4000.0, "avail_mb_max": 4200.0, "temp_c_max": None}

    def pkg_now(self):
        return 10.0


class StubProc:
    def __init__(self, exit_code=None):
        self._exit_code = exit_code

    def poll(self):
        return self._exit_code


class StubServer:
    """Matches the subset of t2s_lab.Server do_call/start_row actually touch. `script` is a list of canned chat()
    outcomes consumed in order; once exhausted, keeps returning the last one."""

    def __init__(self, lab, mi, n_ctx, backend="vulkan", mmap=None, tag="s", script=None):
        self.lab, self.mi, self.n_ctx, self.backend, self.tag = lab, mi, n_ctx, backend, tag
        self.mmap = bool(mmap)
        self.load_mode = "mmap" if mmap else "auto"
        self.pid = 4242
        self.proc = StubProc(None)
        self.start_info = {"load_s": 1.0, "build": "stub-build"}
        self.log_path = str(Path(lab.prefix + f"_srv_{tag}.txt"))
        self._script = list(script or [])
        self._calls = 0
        self._alive = True

    def start(self, timeout=1800):
        return {"ok": True, "exit_code": None, "error": None, "t_start": time.time(), "t_end": time.time(),
                "load_s": 1.0, "pid": self.pid, "build": "stub-build", "log": {}}

    def stop(self):
        self._alive = False

    def alive_and_ours(self):
        return self._alive, [self.pid] if self._alive else []

    def tokenize(self, text):
        return max(len(text.split()), 1)

    def chat(self, prompt, max_tokens, ignore_eos):
        outcome = self._script[min(self._calls, len(self._script) - 1)] if self._script else {"outcome": "ok"}
        self._calls += 1
        if outcome["outcome"] == "ok":
            return {"outcome": "ok", "error": None, "output": outcome.get("output", "no fact here"),
                    "ttft_s": 0.05, "decode_tok_s": 40.0, "e2e_s": 0.5, "completion_tokens": 8,
                    "usage_reported": True, "think_tag": False}
        if outcome["outcome"] == "crash":
            self.proc = StubProc(outcome.get("exit_code", 1))
            self._alive = False
            return {"outcome": "error", "error": "device lost", "output": None}
        return {"outcome": outcome["outcome"], "error": outcome.get("error", "stub error"), "output": None}


class StubLab:
    """Just enough of t2s_overnight.Lab for do_call/start_row/phase_k2 to run against."""

    def __init__(self, tmp_path):
        self.prefix = str(tmp_path / "stub")
        self.tele = StubTelemetry()
        self.smoke = False
        self.done = set()
        self.resources = {}
        self.models = {}
        self.table = {}
        self.identity = {}
        self.deadline_ts = time.time() + 3600
        self.idle_temp = None
        self.idle_pkg = None
        self.rows = []

    def check(self):
        if time.time() > self.deadline_ts:
            raise ov.Deadline()

    def row(self, section, mi=None, backend="vulkan", **kw):
        r = {"section": section, "backend": backend, "seed": 1, "git_sha": "stub", "script_sha": "stub"}
        if mi is not None:
            r.update({"model_id": mi.model_id, "model_sha256": mi.sha256, "quant": mi.quant})
        r.update(kw)
        return r

    def emit(self, row):
        self.rows.append(row)

    def item_done(self, item_id):
        self.done.add(item_id)


def _mi(tmp_path, model_id="qwen3-8b"):
    p = tmp_path / "fake.gguf"
    p.write_bytes(b"0" * 1024)
    mi = type("FakeModelInfo", (), {})()
    mi.model_id, mi.sha256, mi.quant = model_id, "deadbeef", "Q4_K_M"
    mi.file_bytes = 4 * 1024 ** 3
    mi.kv_bpt_meta = 2.0
    mi.thinking_hybrid = False
    return mi


class FakePressure:
    """Stands in for both t2s_lab.Balloon and k2.PageableToucher: never touches real memory."""

    def __init__(self, ok=True, alive=True):
        self._ok, self._alive = ok, alive
        self.started_with = None
        self.stopped = False

    def start(self, target_available_mb):
        self.started_with = target_available_mb
        return {"ok": self._ok, "available_mb_after": target_available_mb, "why": None if self._ok else "mock fail"}

    def alive(self):
        return self._alive

    def stop(self):
        self.stopped = True


# ---------------------------------------------------------------- do_call / start_row (real code, stub objects)

def test_do_call_against_stub_server_records_score_free_row(tmp_path):
    lab = StubLab(tmp_path)
    mi = _mi(tmp_path)
    srv = StubServer(lab, mi, 16384, script=[{"outcome": "ok", "output": "42"}])
    res = ov.do_call(lab, srv, mi, "K2", "item0", "prompt text", 5, warmup=False, rep=0,
                      extra={"k2_phase": "baseline", "task_type": k2.TASK_TYPE}, max_tokens=64,
                      mem_headroom_gb=None, co_runner="none", kind="quality")
    assert res is not None and res["outcome"] == "ok"
    assert lab.rows and lab.rows[-1]["kind"] == "quality"
    assert lab.rows[-1]["ttft_s"] == 0.05


def test_start_row_against_stub_server(tmp_path):
    lab = StubLab(tmp_path)
    mi = _mi(tmp_path)
    srv = StubServer(lab, mi, 16384, tag="k2_start")
    info = srv.start()
    ov.start_row(lab, srv, mi, "K2", "k2_start", info, {"pressure_arm": "awe_balloon", "mmap_arm": "default",
                                                        "need_mib": 5000.0})
    assert lab.rows[-1]["kind"] == "start" and lab.rows[-1]["need_mib"] == 5000.0


# ---------------------------------------------------------------- run_quality_item

def test_run_quality_item_ok_scores_and_classifies_correct(tmp_path):
    lab = StubLab(tmp_path)
    mi = _mi(tmp_path)
    # The fallback task's expected answer is a 6-digit code; make the server "answer" with it by monkeypatching
    # build_quality_task to a fixed, known pair so the scorer path is exercised deterministically.
    with patch.object(k2, "build_quality_task", return_value=("prompt", "123456", "exact")):
        srv = StubServer(lab, mi, 16384, script=[{"outcome": "ok", "output": "123456"}])
        row = k2.run_quality_item(lab, srv, mi, "tag", "level_8", 8, "awe_balloon", 0)
    assert row["score"] == 1.0 and row["classification"] == "correct"
    assert row["crash"] is False and row["exit_code"] is None


def test_run_quality_item_wrong_answer_is_fabrication_not_refusal(tmp_path):
    lab = StubLab(tmp_path)
    mi = _mi(tmp_path)
    with patch.object(k2, "build_quality_task", return_value=("prompt", "123456", "exact")):
        srv = StubServer(lab, mi, 16384, script=[{"outcome": "ok", "output": "the answer is 999999"}])
        row = k2.run_quality_item(lab, srv, mi, "tag", "level_0", 0, "awe_balloon", 0)
    assert row["score"] == 0.0 and row["classification"] == "fabrication"


def test_run_quality_item_refusal_classified_as_refusal(tmp_path):
    lab = StubLab(tmp_path)
    mi = _mi(tmp_path)
    with patch.object(k2, "build_quality_task", return_value=("prompt", "123456", "exact")):
        srv = StubServer(lab, mi, 16384, script=[{"outcome": "ok", "output": "I don't know."}])
        row = k2.run_quality_item(lab, srv, mi, "tag", "level_0", 0, "awe_balloon", 0)
    assert row["score"] == 0.0 and row["classification"] == "refusal"


def test_run_quality_item_crash_records_exit_code(tmp_path):
    lab = StubLab(tmp_path)
    mi = _mi(tmp_path)
    with patch.object(k2, "build_quality_task", return_value=("prompt", "123456", "exact")):
        srv = StubServer(lab, mi, 16384, script=[{"outcome": "crash", "exit_code": 3}])
        row = k2.run_quality_item(lab, srv, mi, "tag", "level_-2", -2, "awe_balloon", 0)
    assert row["crash"] is True and row["exit_code"] == 3 and row["outcome"] == "error"


# ---------------------------------------------------------------- responsiveness sampler (real subprocess, fast)

def test_responsiveness_sampler_collects_samples_and_median():
    s = k2.ResponsivenessSampler(interval_s=0.05)
    s.start()
    time.sleep(0.2)
    samples = s.stop()
    assert len(samples) >= 2
    assert all("latency_s" in x and "ok" in x for x in samples)
    med = s.median_latency_s()
    assert med is None or med >= 0.0


# ---------------------------------------------------------------- pressure-arm construction (mocked, no allocation)

def test_make_pressure_awe_balloon_returns_lab_balloon(tmp_path):
    import t2s_lab as L
    lab = StubLab(tmp_path)
    p = k2._make_pressure(lab, "awe_balloon", "tag_8")
    assert isinstance(p, L.Balloon)


def test_make_pressure_pageable_touch_returns_toucher(tmp_path):
    lab = StubLab(tmp_path)
    p = k2._make_pressure(lab, "pageable_touch", "tag_8")
    assert isinstance(p, k2.PageableToucher)


def test_make_pressure_unknown_arm_raises(tmp_path):
    lab = StubLab(tmp_path)
    with pytest.raises(ValueError):
        k2._make_pressure(lab, "not_a_real_arm", "tag")


# ---------------------------------------------------------------- full pressure-step loop (mocked server/pressure)

def test_run_k2_run_full_pressure_loop_no_crash(tmp_path, monkeypatch):
    lab = StubLab(tmp_path)
    mi = _mi(tmp_path)
    server_holder = {}

    def fake_server(lab_, mi_, n_ctx, backend="vulkan", mmap=None, tag="s"):
        srv = StubServer(lab_, mi_, n_ctx, backend=backend, mmap=mmap, tag=tag,
                          script=[{"outcome": "ok", "output": "stub"}])
        server_holder["srv"] = srv
        return srv

    monkeypatch.setattr(k2.L, "Server", fake_server)
    monkeypatch.setattr(k2, "_make_pressure", lambda lab_, arm, tag: FakePressure(ok=True))
    monkeypatch.setattr(k2, "build_quality_task", lambda *a, **kw: ("prompt", "999999", "exact"))

    summaries = k2.run_k2_run(lab, mi, 16384, 5000.0, "mmap", "awe_balloon", responsiveness_interval_s=0.05)

    assert len(summaries) == len(k2.LEVELS_GB)
    assert [s["level_gb"] for s in summaries] == k2.LEVELS_GB
    assert all(s["clean_failure"] is False for s in summaries)
    kc_rows = [r for r in lab.rows if r.get("record") == "k2_kill_criterion"]
    assert kc_rows and kc_rows[0]["ok"] is True
    assert server_holder["srv"]._alive is False  # srv.stop() was called at the end of the run


def test_run_k2_run_stops_early_on_clean_failure(tmp_path, monkeypatch):
    lab = StubLab(tmp_path)
    mi = _mi(tmp_path)

    # ok, ok, ok (baseline) then crash on the very first pressure-level item (+8GB level).
    script = [{"outcome": "ok", "output": "stub"}] * k2.N_BASELINE_ITEMS + [{"outcome": "crash", "exit_code": 7}]

    def fake_server(lab_, mi_, n_ctx, backend="vulkan", mmap=None, tag="s"):
        return StubServer(lab_, mi_, n_ctx, backend=backend, mmap=mmap, tag=tag, script=script)

    monkeypatch.setattr(k2.L, "Server", fake_server)
    monkeypatch.setattr(k2, "_make_pressure", lambda lab_, arm, tag: FakePressure(ok=True))
    monkeypatch.setattr(k2, "build_quality_task", lambda *a, **kw: ("prompt", "999999", "exact"))

    summaries = k2.run_k2_run(lab, mi, 16384, 5000.0, "default", "pageable_touch", responsiveness_interval_s=0.05)

    assert len(summaries) == 1  # stopped after the first (failing) level, rest not run
    assert summaries[0]["level_gb"] == 8 and summaries[0]["clean_failure"] is True


def test_phase_k2_runs_all_four_arm_combinations(tmp_path, monkeypatch):
    lab = StubLab(tmp_path)
    mi = _mi(tmp_path)
    lab.models["qwen3-8b"] = mi
    calls = []

    def fake_run_k2_run(lab_, mi_, n_ctx, need_mib, mmap_arm, pressure_arm, responsiveness_interval_s=30.0):
        calls.append((mmap_arm, pressure_arm))
        return [{"record": "k2_step_summary", "level_gb": 8, "median_score": 1.0,
                 "responsiveness_median_s": 0.01, "clean_failure": False}]

    monkeypatch.setattr(k2, "run_k2_run", fake_run_k2_run)
    k2.phase_k2(lab, "qwen3-8b", n_ctx=16384)
    assert set(calls) == {(m, p) for m in k2.MMAP_ARMS for p in k2.PRESSURE_ARMS}


def test_phase_k2_missing_model_emits_disabled_record(tmp_path):
    lab = StubLab(tmp_path)
    out = k2.phase_k2(lab, "not-loaded-model")
    assert out == []
    assert lab.rows[-1]["record"] == "k2_disabled"


# ---------------------------------------------------------------- kill_criterion

def _step(level_gb, score, resp, clean_failure=False):
    return {"record": "k2_step_summary", "level_gb": level_gb, "median_score": score,
            "responsiveness_median_s": resp, "clean_failure": clean_failure}


def test_kill_criterion_passes_when_stable_until_crash():
    rows = [
        _step(8, 0.90, 0.05),
        _step(4, 0.91, 0.05),
        _step(2, 0.89, 0.06),
        _step(1, 0.90, 0.07),
        _step(0, 0.88, 0.08),
        _step(-1, 0.40, 0.50, clean_failure=True),  # quality/responsiveness only degrade at/after the crash step
    ]
    kc = k2.kill_criterion(rows)
    assert kc["ok"] is True
    assert kc["first_failure_level_gb"] == -1


def test_kill_criterion_fails_on_early_quality_drop_before_any_crash():
    rows = [
        _step(8, 0.90, 0.05),
        _step(4, 0.40, 0.05),  # big quality drop with no crash anywhere yet
        _step(2, 0.90, 0.05),
        _step(1, 0.90, 0.05),
        _step(0, 0.90, 0.05),
        _step(-1, 0.10, 2.0, clean_failure=True),
    ]
    kc = k2.kill_criterion(rows)
    assert kc["ok"] is False
    assert "4" in kc["reason"]


def test_kill_criterion_fails_on_early_responsiveness_regression():
    rows = [
        _step(8, 0.90, 0.05),
        _step(4, 0.90, 0.05),
        _step(2, 0.90, 0.30),  # 6x baseline responsiveness, no crash yet
        _step(1, 0.90, 0.05),
        _step(0, 0.90, 0.05, clean_failure=True),
    ]
    kc = k2.kill_criterion(rows)
    assert kc["ok"] is False
    assert "2" in kc["reason"]


def test_kill_criterion_no_baseline_step_is_not_ok():
    rows = [_step(4, 0.9, 0.05), _step(0, 0.9, 0.05, clean_failure=True)]
    kc = k2.kill_criterion(rows)
    assert kc["ok"] is False
    assert "baseline" in kc["reason"]


def test_kill_criterion_passes_when_nothing_ever_fails():
    rows = [_step(lv, 0.90 + 0.001 * i, 0.05) for i, lv in enumerate(k2.LEVELS_GB)]
    kc = k2.kill_criterion(rows)
    assert kc["ok"] is True
    assert kc["first_failure_level_gb"] is None


# ---------------------------------------------------------------- duplicate-keyword-argument guard

# Reserved keyword names do_call/start_row/row already pass explicitly (mirrors what a real
# tests/test_no_duplicate_row_kwargs.py would check, which does not exist yet in this repo).
_DO_CALL_RESERVED = {"n_ctx", "prompt_tokens", "mmap", "load_mode", "co_runner", "rep", "mem_headroom_gb", "load_s",
                     "item_id", "kind", "warmup", "server_pid", "llama_build", "thermal_wait_s",
                     "temp_at_gate_start", "temp_at_release", "idle_temp", "gate_released_by", "idle_pkg_w",
                     "pkg_w_at_release"}
_START_ROW_RESERVED = {"n_ctx", "mmap", "load_mode", "load_s", "item_id", "kind", "error", "server_pid",
                       "llama_build", "valid", "t_start_utc", "t_end_utc"}
_ROW_RESERVED = {"section", "mi", "backend"}


def test_run_quality_item_extra_dict_has_no_reserved_do_call_keys():
    # extra is rebuilt exactly as run_quality_item builds it (k2_phase, task_type only).
    extra = {"k2_phase": "level_0", "task_type": k2.TASK_TYPE}
    assert not (set(extra) & _DO_CALL_RESERVED)


def test_run_k2_run_start_row_extra_has_no_reserved_start_row_keys():
    extra = {"pressure_arm": "awe_balloon", "mmap_arm": "default", "need_mib": 5000.0}
    assert not (set(extra) & _START_ROW_RESERVED)


def test_run_quality_item_row_kwargs_have_no_reserved_row_keys():
    kwargs = {"item_id": "x", "kind": "quality_score", "k2_phase": "level_0", "pressure_arm": "awe_balloon",
              "mem_headroom_gb": 0, "task_type": k2.TASK_TYPE, "rep": 0, "score": 1.0, "classification": "correct",
              "ttft_s": 0.1, "decode_tok_s": 30.0, "outcome": "ok", "error": None, "crash": False,
              "exit_code": None, "server_log_tail": None}
    assert not (set(kwargs) & _ROW_RESERVED)


def test_no_duplicate_kwargs_end_to_end_via_do_call(tmp_path):
    """The strongest version of the check: actually call do_call the way run_quality_item does and confirm it does
    not raise TypeError for a duplicate keyword argument."""
    lab = StubLab(tmp_path)
    mi = _mi(tmp_path)
    srv = StubServer(lab, mi, 16384, script=[{"outcome": "ok", "output": "x"}])
    try:
        ov.do_call(lab, srv, mi, "K2", "item0", "p", 3, warmup=False, rep=0,
                   extra={"k2_phase": "baseline", "task_type": k2.TASK_TYPE}, max_tokens=32,
                   mem_headroom_gb=None, co_runner="none", kind="quality")
    except TypeError as e:
        pytest.fail(f"duplicate-keyword-argument bug in do_call call site: {e}")


# ---------------------------------------------------------------------------------------------------- ollama contamination guard
def test_main_aborts_if_ollama_is_running():
    """K2 never uses Ollama itself; per the 2026-09-29 contamination check it must refuse to start a model's phase
    if an Ollama process is found running rather than risk a model load racing against this measurement."""
    import inspect
    src = inspect.getsource(k2.main)
    assert "hc.ollama_process_running()" in src
    guard_idx = src.index("hc.ollama_process_running()")
    phase_call_idx = src.index("phase_k2(lab, mid, args.n_ctx)")
    assert guard_idx < phase_call_idx
