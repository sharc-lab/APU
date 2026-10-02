"""2026-10-02: MX2's own bisection mostly hit a guard mismatch (server negotiated n_ctx=131072 regardless of a
much higher requested value) -- see docs/FINDINGS.md's pre-registration. These tests cover the real logic this
validation pass adds: actual-vs-requested n_ctx classification, resumability, the crash-reproduction started/
not-started branches, and the overall run() orchestration -- with the real subprocess/GPU/network calls
stubbed out."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))

import mx2_validation as mv  # noqa: E402


# --------------------------------------------------------------------------------------------------- classify_point
def test_classify_point_fits_when_non_device_local_usage_is_low():
    result = mv.classify_point(requested_n_ctx=20480, actual_n_ctx=20480, non_device_local_usage_mib=10.0)
    assert result["regime"] == "FITS"
    assert result["clamped"] is False
    assert result["spill_evidence"] == []


def test_classify_point_silent_spill_when_non_device_local_usage_is_high():
    """The primary, direct spill signal: real live usage on a non-device-local heap, well past the noise
    threshold -- this is the pre-registered SILENT_SPILL criterion (docs/FINDINGS.md), not an inference from
    llama-server's own startup-log buffer sizes (which require --log-verbosity 4 to even appear)."""
    result = mv.classify_point(requested_n_ctx=115200, actual_n_ctx=115200, non_device_local_usage_mib=5000.0)
    assert result["regime"] == "SILENT_SPILL"
    assert result["clamped"] is False
    assert "non_device_local_heap_usage_over_threshold" in result["spill_evidence"]


def test_classify_point_silent_spill_from_logged_buffers_even_without_live_usage_reading(monkeypatch):
    """The secondary evidence path: when the live heap-usage reading is unavailable (None) but the log-parsed
    buffer sizes are, and they cross the device-local line, that alone is still enough evidence."""
    result = mv.classify_point(requested_n_ctx=115200, actual_n_ctx=115200, non_device_local_usage_mib=None,
                               logged_mib=80500.0, device_local_mib=76000.0)
    assert result["regime"] == "SILENT_SPILL"
    assert "logged_buffers_over_device_local_line" in result["spill_evidence"]


def test_classify_point_flags_clamped_when_actual_differs_from_requested():
    """The exact 2026-10-02 bug this whole validation pass exists to resolve: a requested n_ctx beyond the
    model's effective ceiling gets silently served at a smaller actual n_ctx."""
    result = mv.classify_point(requested_n_ctx=290000, actual_n_ctx=131072, non_device_local_usage_mib=10.0)
    assert result["clamped"] is True


def test_classify_point_not_clamped_when_actual_matches_requested():
    result = mv.classify_point(requested_n_ctx=20480, actual_n_ctx=20480, non_device_local_usage_mib=10.0)
    assert result["clamped"] is False


def test_classify_point_fits_when_no_evidence_is_available_at_all():
    result = mv.classify_point(requested_n_ctx=1, actual_n_ctx=1, non_device_local_usage_mib=None,
                               logged_mib=None, device_local_mib=76000.0)
    assert result["logged_mib"] is None
    assert result["regime"] == "FITS"  # no evidence of a spill without any reading to compare


# --------------------------------------------------------------------------------------------------- live heap usage
def test_read_non_device_local_usage_mib_sums_non_local_heaps(monkeypatch):
    monkeypatch.setattr(mv.night2, "mx2_read_vulkaninfo", lambda: "fake vulkaninfo text")
    monkeypatch.setattr(mv.night2, "mx2_parse_heaps", lambda txt: [
        {"index": 0, "usage_mib": 100.0, "device_local": False},
        {"index": 1, "usage_mib": 5000.0, "device_local": True},
        {"index": 2, "usage_mib": 50.0, "device_local": False},
    ])
    assert mv.read_non_device_local_usage_mib() == 150.0


def test_read_non_device_local_usage_mib_is_zero_not_none_when_no_non_local_heap_exists(monkeypatch):
    monkeypatch.setattr(mv.night2, "mx2_read_vulkaninfo", lambda: "fake vulkaninfo text")
    monkeypatch.setattr(mv.night2, "mx2_parse_heaps", lambda txt: [{"index": 0, "usage_mib": 5000.0, "device_local": True}])
    assert mv.read_non_device_local_usage_mib() == 0.0  # a real zero reading, distinct from "could not read"


def test_read_non_device_local_usage_mib_is_none_when_vulkaninfo_unavailable(monkeypatch):
    monkeypatch.setattr(mv.night2, "mx2_read_vulkaninfo", lambda: None)
    assert mv.read_non_device_local_usage_mib() is None


# --------------------------------------------------------------------------------------------------- resumability
def test_already_done_keys_tracks_regime_points_and_crash_repros(tmp_path):
    out_path = tmp_path / "out.jsonl"
    rows = [
        {"record": "mx2v_regime_point", "model_id": "llama-3.3-70b", "point_type": "deep_fits", "rep": 0},
        {"record": "mx2v_regime_point", "model_id": "llama-3.3-70b", "point_type": "deep_fits", "rep": 1},
        {"record": "mx2v_crash_repro", "model_id": "qwen3-32b", "rep": 0},
        {"record": "heartbeat", "model_id": "qwen3-32b"},  # must not be treated as a done key
    ]
    out_path.write_text("\n".join(__import__("json").dumps(r) for r in rows), encoding="utf-8")
    done = mv.already_done_keys(out_path)
    assert ("regime_point", "llama-3.3-70b", "deep_fits", 0) in done
    assert ("regime_point", "llama-3.3-70b", "deep_fits", 1) in done
    assert ("crash_repro", "qwen3-32b", 0) in done
    assert ("regime_point", "llama-3.3-70b", "deep_fits", 2) not in done
    assert len(done) == 3


def test_already_done_keys_empty_when_file_absent(tmp_path):
    assert mv.already_done_keys(tmp_path / "missing.jsonl") == set()


# --------------------------------------------------------------------------------------------------- fixed prompt
def test_build_fixed_prompt_is_deterministic_and_roughly_sized():
    p1 = mv.build_fixed_prompt(2048)
    p2 = mv.build_fixed_prompt(2048)
    assert p1 == p2
    assert len(p1) == 2048 * 4


# --------------------------------------------------------------------------------------------------- GPU memory query
def test_query_gpu_process_memory_parses_dedicated_and_shared(monkeypatch):
    import json as _json
    payload = _json.dumps([
        {"Path": r"\gpu process memory(pid_123_luid_0x1)\dedicated usage", "CookedValue": 2 ** 20 * 5000},
        {"Path": r"\gpu process memory(pid_123_luid_0x1)\shared usage", "CookedValue": 2 ** 20 * 1000},
    ])
    result = mv.query_gpu_process_memory(123, ps_fn=lambda cmd, timeout: payload)
    assert result == {"dedicated_mib": 5000.0, "shared_mib": 1000.0}


def test_query_gpu_process_memory_returns_none_on_any_failure(monkeypatch):
    def raise_it(cmd, timeout):
        raise RuntimeError("no counters")
    result = mv.query_gpu_process_memory(123, ps_fn=raise_it)
    assert result == {"dedicated_mib": None, "shared_mib": None}


# --------------------------------------------------------------------------------------------------- responsiveness
def test_responsiveness_probe_reports_ok_and_a_real_duration(monkeypatch):
    monkeypatch.setattr(mv.subprocess, "run", lambda *a, **kw: None)
    result = mv.responsiveness_probe()
    assert result["ok"] is True
    assert result["round_trip_s"] >= 0.0


def test_responsiveness_probe_reports_not_ok_on_failure(monkeypatch):
    def raise_it(*a, **kw):
        raise RuntimeError("wedged")
    monkeypatch.setattr(mv.subprocess, "run", raise_it)
    result = mv.responsiveness_probe()
    assert result["ok"] is False


# --------------------------------------------------------------------------------------------------- run() orchestration
def test_run_skips_already_done_points_and_calls_the_rest(tmp_path, monkeypatch):
    out_path = tmp_path / "out.jsonl"
    monkeypatch.setattr(mv, "responsiveness_probe", lambda: {"round_trip_s": 0.01, "ok": True})
    calls = []
    monkeypatch.setattr(mv, "run_regime_point",
                        lambda model_id, point_type, n_ctx, rep, out_path, log: calls.append(("point", model_id, point_type, rep)))
    monkeypatch.setattr(mv, "run_crash_repro",
                        lambda model_id, n_ctx, rep, out_path, log: calls.append(("crash", model_id, rep)))
    # pre-seed one regime point and one crash rep as already done
    import json
    done_rows = [
        {"record": "mx2v_regime_point", "model_id": "llama-3.3-70b", "point_type": "deep_fits", "rep": 0},
        {"record": "mx2v_crash_repro", "model_id": "llama-3.3-70b", "rep": 0},
    ]
    out_path.write_text("\n".join(json.dumps(r) for r in done_rows) + "\n", encoding="utf-8")

    mv.run(out_path, smoke=False, deadline_h=10.0)

    assert ("point", "llama-3.3-70b", "deep_fits", 0) not in calls  # already done, skipped
    assert ("point", "llama-3.3-70b", "deep_fits", 1) in calls  # not done, ran
    assert ("point", "llama-3.3-70b", "deep_fits", 2) in calls
    assert ("crash", "llama-3.3-70b", 0) not in calls  # already done, skipped
    assert ("crash", "llama-3.3-70b", 1) in calls


def test_run_in_smoke_mode_only_touches_one_model_one_point_one_rep(tmp_path, monkeypatch):
    out_path = tmp_path / "out.jsonl"
    monkeypatch.setattr(mv, "responsiveness_probe", lambda: {"round_trip_s": 0.01, "ok": True})
    calls = []
    monkeypatch.setattr(mv, "run_regime_point",
                        lambda model_id, point_type, n_ctx, rep, out_path, log: calls.append(("point", model_id, point_type, rep)))
    monkeypatch.setattr(mv, "run_crash_repro",
                        lambda model_id, n_ctx, rep, out_path, log: calls.append(("crash", model_id, rep)))

    mv.run(out_path, smoke=True, deadline_h=10.0)

    point_calls = [c for c in calls if c[0] == "point"]
    crash_calls = [c for c in calls if c[0] == "crash"]
    # smoke mode restricts to models[:1] and 1 rep each, but still exercises BOTH code paths end to end --
    # a smoke test that skipped crash_repro entirely would never catch a crash_repro-specific bug before a
    # full live run did.
    assert point_calls == [("point", "llama-3.3-70b", "deep_fits", 0)]
    assert crash_calls == [("crash", "llama-3.3-70b", 0)]


def test_run_stops_at_deadline(tmp_path, monkeypatch):
    """A real run_regime_point/run_crash_repro takes real wall-clock time, so the deadline check between
    iterations has something to measure; stubbed calls here are near-instant, so the clock itself is faked
    to jump forward past the deadline on the very first check rather than relying on real elapsed time."""
    out_path = tmp_path / "out.jsonl"
    monkeypatch.setattr(mv, "responsiveness_probe", lambda: {"round_trip_s": 0.01, "ok": True})
    calls = []
    monkeypatch.setattr(mv, "run_regime_point",
                        lambda model_id, point_type, n_ctx, rep, out_path, log: calls.append(1))
    monkeypatch.setattr(mv, "run_crash_repro", lambda *a, **kw: calls.append(1))
    times = iter([0.0, 100.0])  # t_start=0.0, first deadline check already reads 100.0
    monkeypatch.setattr(mv.time, "monotonic", lambda: next(times, 100.0))

    mv.run(out_path, smoke=False, deadline_h=0.001)  # 3.6s -- the faked clock jump dwarfs this regardless

    assert calls == []


# --------------------------------------------------------------------------------------------------- crash repro
def test_run_crash_repro_detects_a_process_that_exits_before_health_check(tmp_path, monkeypatch):
    """The real HARD_FAIL case: llama-server crashes before /health ever responds."""
    class FakeProc:
        returncode = 1
        def poll(self):
            return 1  # already exited
        def wait(self, timeout=None):
            return None

    monkeypatch.setattr(mv.subprocess, "Popen", lambda *a, **kw: FakeProc())
    (tmp_path / "dummy.log").write_text("some log content\n", encoding="utf-8")
    monkeypatch.setattr(mv, "REPO", tmp_path)
    out_path = tmp_path / "out.jsonl"
    row = mv.run_crash_repro("llama-3.3-70b", 221696, 0, out_path, log=lambda s: None)
    assert row["started"] is False
    assert row["exit_code"] == 1


def test_run_crash_repro_kills_a_process_that_does_start(tmp_path, monkeypatch):
    """If a crash point unexpectedly DOES start (e.g. the boundary moved since the original bisection), the
    repro must not leave the server running -- it kills it and reports started=True, not treat that as a
    script error."""
    class FakeProc:
        returncode = None
        killed = False
        def poll(self):
            return None
        def kill(self):
            FakeProc.killed = True
            self.returncode = -9
        def wait(self, timeout=None):
            return None

    monkeypatch.setattr(mv.subprocess, "Popen", lambda *a, **kw: FakeProc())
    monkeypatch.setattr(mv, "REPO", tmp_path)

    class FakeHttpxResp:
        status_code = 200

    fake_httpx = type("M", (), {"get": staticmethod(lambda url, timeout=2.0: FakeHttpxResp())})
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    out_path = tmp_path / "out.jsonl"
    row = mv.run_crash_repro("qwen3-32b", 367360, 0, out_path, log=lambda s: None)
    assert row["started"] is True
    assert FakeProc.killed is True
    # 2026-10-02 bug found live: Windows' own TerminateProcess (what proc.kill() calls) sets exit code 1 by
    # default -- that must never be reported as if it were a real crash exit code when the server actually
    # started fine and was killed only for cleanup.
    assert row["exit_code"] is None
    assert row["killed_by_probe"] is True
