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
def test_classify_point_on_the_two_real_rows_found_live_2026_10_02():
    """llama-3.3-70b, requested_n_ctx==actual_n_ctx==20480 (deep_fits) and ==115200 (mid_spill), from the real
    live run. deep_fits's own gpu_shared_mib (633.67578125) is its own baseline (shared_delta_mib=0, FITS).
    mid_spill's gpu_shared_mib (14030.6875) against that same baseline is a real, large spill that
    vulkaninfo's non_device_local_usage_mib=0.0 (also a real recorded value, same row) completely missed --
    this is the exact case the classifier was rewritten for."""
    deep_fits = mv.classify_point(requested_n_ctx=20480, actual_n_ctx=20480, gpu_shared_mib=633.67578125,
                                  baseline_shared_mib=633.67578125, non_device_local_usage_mib=0.0)
    assert deep_fits["regime"] == "FITS"
    assert deep_fits["shared_delta_mib"] == 0.0

    mid_spill = mv.classify_point(requested_n_ctx=115200, actual_n_ctx=115200, gpu_shared_mib=14030.6875,
                                  baseline_shared_mib=633.67578125, non_device_local_usage_mib=0.0)
    assert mid_spill["regime"] == "SILENT_SPILL"
    assert mid_spill["clamped"] is False
    assert abs(mid_spill["shared_delta_mib"] - 13397.01171875) < 0.01


def test_classify_point_fits_when_shared_delta_is_within_threshold():
    result = mv.classify_point(requested_n_ctx=20480, actual_n_ctx=20480, gpu_shared_mib=700.0,
                               baseline_shared_mib=633.67578125)
    assert result["regime"] == "FITS"


def test_classify_point_fits_when_clamped_even_with_a_large_shared_delta():
    """The third pre-registered AND-condition: a CLAMPED point (actual != requested) is never confirmed
    SILENT_SPILL regardless of its shared-memory reading -- it tested a different, smaller actual
    configuration than the one it is labeled with."""
    result = mv.classify_point(requested_n_ctx=290000, actual_n_ctx=131072, gpu_shared_mib=20000.0,
                               baseline_shared_mib=633.67578125)
    assert result["clamped"] is True
    assert result["regime"] == "FITS"


def test_classify_point_fits_when_no_baseline_or_reading_available():
    result = mv.classify_point(requested_n_ctx=1, actual_n_ctx=1, gpu_shared_mib=None, baseline_shared_mib=None)
    assert result["shared_delta_mib"] is None
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
                        lambda model_id, point_type, n_ctx, rep, out_path, log, baseline_shared_mib=None: (calls.append(("point", model_id, point_type, rep)), {})[1])
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


def test_run_passes_the_deep_fits_baseline_to_later_points_in_the_same_session(tmp_path, monkeypatch):
    out_path = tmp_path / "out.jsonl"
    monkeypatch.setattr(mv, "responsiveness_probe", lambda: {"round_trip_s": 0.01, "ok": True})
    monkeypatch.setattr(mv, "run_crash_repro", lambda *a, **kw: None)
    baselines_seen = []

    def fake_point(model_id, point_type, n_ctx, rep, out_path, log, baseline_shared_mib=None):
        baselines_seen.append((point_type, rep, baseline_shared_mib))
        if point_type == "deep_fits":
            return {"started": True, "gpu_shared_mib": 633.67578125}
        return {"started": True, "gpu_shared_mib": 14030.6875}

    monkeypatch.setattr(mv, "run_regime_point", fake_point)
    mv.run(out_path, smoke=False, deadline_h=10.0)

    # deep_fits itself never had a baseline yet (it IS the baseline)
    assert ("deep_fits", 0, None) in baselines_seen
    # every later point in this model, this session, gets deep_fits's own rep-0 gpu_shared_mib as baseline
    assert ("mid_spill", 0, 633.67578125) in baselines_seen
    assert ("near_crash", 0, 633.67578125) in baselines_seen


def test_run_seeds_the_deep_fits_baseline_from_an_already_written_resume_file(tmp_path, monkeypatch):
    """--resume: deep_fits rows from an earlier run of this file are skipped (already done), so the baseline
    a later point needs must be read back from the file, not just tracked in this process's own memory."""
    out_path = tmp_path / "out.jsonl"
    monkeypatch.setattr(mv, "responsiveness_probe", lambda: {"round_trip_s": 0.01, "ok": True})
    monkeypatch.setattr(mv, "run_crash_repro", lambda *a, **kw: None)
    import json
    done_rows = [
        {"record": "mx2v_regime_point", "model_id": "llama-3.3-70b", "point_type": "deep_fits", "rep": 0,
         "started": True, "gpu_shared_mib": 633.67578125},
        {"record": "mx2v_regime_point", "model_id": "llama-3.3-70b", "point_type": "deep_fits", "rep": 1,
         "started": True, "gpu_shared_mib": 633.67578125},
        {"record": "mx2v_regime_point", "model_id": "llama-3.3-70b", "point_type": "deep_fits", "rep": 2,
         "started": True, "gpu_shared_mib": 633.67578125},
    ]
    out_path.write_text("\n".join(json.dumps(r) for r in done_rows) + "\n", encoding="utf-8")
    baselines_seen = []
    monkeypatch.setattr(mv, "run_regime_point",
                        lambda model_id, point_type, n_ctx, rep, out_path, log, baseline_shared_mib=None:
                            (baselines_seen.append((model_id, baseline_shared_mib)), {"started": True, "gpu_shared_mib": 1.0})[1])

    mv.run(out_path, smoke=False, deadline_h=10.0)

    # llama-3.3-70b's deep_fits reps were all already done (seeded from the resume file), so every call this
    # run actually makes for that model is mid_spill/near_crash, and each one must get the seeded baseline --
    # qwen3-32b has no seeded data at all, so its own calls legitimately see baseline_shared_mib=None instead.
    seventy_b_baselines = [b for m, b in baselines_seen if m == "llama-3.3-70b"]
    assert seventy_b_baselines  # at least one call happened for this model (mid_spill/near_crash)
    assert all(b == 633.67578125 for b in seventy_b_baselines)


def test_run_in_smoke_mode_only_touches_one_model_one_point_one_rep(tmp_path, monkeypatch):
    out_path = tmp_path / "out.jsonl"
    monkeypatch.setattr(mv, "responsiveness_probe", lambda: {"round_trip_s": 0.01, "ok": True})
    calls = []
    monkeypatch.setattr(mv, "run_regime_point",
                        lambda model_id, point_type, n_ctx, rep, out_path, log, baseline_shared_mib=None: (calls.append(("point", model_id, point_type, rep)), {})[1])
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
                        lambda model_id, point_type, n_ctx, rep, out_path, log, baseline_shared_mib=None: (calls.append(1), {})[1])
    monkeypatch.setattr(mv, "run_crash_repro", lambda *a, **kw: calls.append(1))
    times = iter([0.0, 100.0])  # t_start=0.0, first deadline check already reads 100.0
    monkeypatch.setattr(mv.time, "monotonic", lambda: next(times, 100.0))

    mv.run(out_path, smoke=False, deadline_h=0.001)  # 3.6s -- the faked clock jump dwarfs this regardless

    assert calls == []


# --------------------------------------------------------------------------------------------------- regime point
def test_run_regime_point_suppresses_thinking_mode(tmp_path, monkeypatch):
    """2026-10-02 bug found live: qwen3-32b defaults to thinking mode, and without reasoning_budget=0 the
    whole n_predict budget went into hidden reasoning content -- ttft_ms/decode_tok_s stayed None for every
    point on this model despite a real, complete generation (tokens_out=128, done_reason='length')."""
    captured = {}

    class FakeSession:
        n_ctx_slot = 20480
        _log_path = None
        _proc = None
        def __init__(self, cfg):
            captured["cfg"] = cfg
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def call(self, prompt, max_tokens):
            return ("", 1000.0, 500.0, 10, 20, "stop", 0)

    monkeypatch.setattr(mv, "LlamaServerSession", FakeSession)
    monkeypatch.setattr(mv, "read_heap_lines", lambda: {"device_local_mib": 76000.0})
    monkeypatch.setattr(mv, "read_non_device_local_usage_mib", lambda: 0.0)
    monkeypatch.setattr(mv.L, "parse_server_log", lambda path: {})
    monkeypatch.setattr(mv, "query_gpu_process_memory", lambda pid: {"dedicated_mib": 1.0, "shared_mib": 1.0})

    mv.run_regime_point("qwen3-32b", "deep_fits", 20480, 0, tmp_path / "out.jsonl", log=lambda s: None)

    assert captured["cfg"].reasoning_budget == 0


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


def test_run_crash_repro_survives_a_slow_to_die_process_after_kill(tmp_path, monkeypatch):
    """2026-10-02 bug found live: a real 32B model process, killed mid-load at a near-crash n_ctx, took
    longer than a bare 10s proc.wait(timeout=10) to actually exit -- that raised an UNCAUGHT TimeoutExpired
    (no try/except around that specific wait call) and crashed the whole job. A slow-to-die process after
    kill() must never crash the script -- it has already been sent SIGKILL/TerminateProcess and will die on
    its own eventually."""
    class FakeProc:
        pid = 4242
        returncode = None
        wait_calls = 0
        def poll(self):
            return None
        def kill(self):
            pass
        def wait(self, timeout=None):
            FakeProc.wait_calls += 1
            raise mv.subprocess.TimeoutExpired(cmd="llama-server", timeout=timeout)

    monkeypatch.setattr(mv.subprocess, "Popen", lambda *a, **kw: FakeProc())
    monkeypatch.setattr(mv, "REPO", tmp_path)

    class FakeHttpxResp:
        status_code = 200

    fake_httpx = type("M", (), {"get": staticmethod(lambda url, timeout=2.0: FakeHttpxResp())})
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    out_path = tmp_path / "out.jsonl"
    warnings = []
    row = mv.run_crash_repro("qwen3-32b", 367360, 0, out_path, log=warnings.append)
    assert row["started"] is True
    assert row["exit_code"] is None
    assert FakeProc.wait_calls >= 1
    assert any("did not exit" in w for w in warnings)


# --------------------------------------------------------------------------------------------------- reclassify_file
def test_reclassify_file_updates_regime_from_saved_gpu_shared_mib_without_rerunning(tmp_path):
    """The real post-processing use case: rows already written with the OLD (wrong, vulkaninfo-based)
    classifier get their regime/clamped/shared_delta_mib corrected from their own already-recorded
    gpu_shared_mib -- no server is started, nothing is re-measured."""
    import json
    out_path = tmp_path / "out.jsonl"
    rows = [
        {"record": "mx2v_baseline_responsiveness", "ok": True},
        {"record": "mx2v_regime_point", "model_id": "llama-3.3-70b", "point_type": "deep_fits", "rep": 0,
         "started": True, "requested_n_ctx": 20480, "actual_n_ctx": 20480, "gpu_shared_mib": 633.67578125,
         "regime": "FITS", "non_device_local_usage_mib": 0.0,
         "decode_tok_s": 5.295, "ttft_ms": 21477.9},  # untouched fields that must survive reclassification
        {"record": "mx2v_regime_point", "model_id": "llama-3.3-70b", "point_type": "mid_spill", "rep": 0,
         "started": True, "requested_n_ctx": 115200, "actual_n_ctx": 115200, "gpu_shared_mib": 14030.6875,
         "regime": "FITS", "non_device_local_usage_mib": 0.0},  # the real wrong label this fix exists for
        {"record": "mx2v_crash_repro", "model_id": "llama-3.3-70b", "n_ctx": 221696, "started": False},
    ]
    out_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    n = mv.reclassify_file(out_path)

    updated = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    deep_fits = next(r for r in updated if r.get("point_type") == "deep_fits")
    mid_spill = next(r for r in updated if r.get("point_type") == "mid_spill")
    assert n == 2
    assert deep_fits["regime"] == "FITS"
    assert mid_spill["regime"] == "SILENT_SPILL"  # corrected from the wrong FITS label
    assert mid_spill["baseline_shared_mib"] == 633.67578125
    # untouched raw measurement fields survive verbatim
    assert deep_fits["decode_tok_s"] == 5.295
    assert deep_fits["ttft_ms"] == 21477.9
    # non-regime_point records pass through completely unchanged
    assert updated[0] == rows[0]
    assert updated[3] == rows[3]


def test_reclassify_file_is_idempotent(tmp_path):
    import json
    out_path = tmp_path / "out.jsonl"
    rows = [
        {"record": "mx2v_regime_point", "model_id": "llama-3.3-70b", "point_type": "deep_fits", "rep": 0,
         "started": True, "requested_n_ctx": 20480, "actual_n_ctx": 20480, "gpu_shared_mib": 633.67578125,
         "regime": "FITS"},
        {"record": "mx2v_regime_point", "model_id": "llama-3.3-70b", "point_type": "mid_spill", "rep": 0,
         "started": True, "requested_n_ctx": 115200, "actual_n_ctx": 115200, "gpu_shared_mib": 14030.6875,
         "regime": "FITS"},
    ]
    out_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    mv.reclassify_file(out_path)
    once = out_path.read_text(encoding="utf-8")
    mv.reclassify_file(out_path)
    twice = out_path.read_text(encoding="utf-8")
    assert once == twice


def test_reclassify_file_skips_rows_that_never_started(tmp_path):
    import json
    out_path = tmp_path / "out.jsonl"
    rows = [{"record": "mx2v_regime_point", "model_id": "llama-3.3-70b", "point_type": "deep_fits", "rep": 0,
            "started": False, "error": "something broke"}]
    out_path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    n = mv.reclassify_file(out_path)
    assert n == 0
    updated = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
    assert updated == rows[0]  # left exactly as written
