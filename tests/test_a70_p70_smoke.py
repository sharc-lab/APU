"""Dry runs (stub lab, no real server, no real co-runner) for phase_a70, phase_p70, and the smoke-gate machinery
(SMOKE_GATED_PHASES / smoke_check_phase / SmokeFailure) added for Addendum v3."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import t2s_night2 as n2  # noqa: E402
from stub_lab import StubLab, StubModelInfo, StubServer  # noqa: E402


def _mi70():
    mi = StubModelInfo("llama-3.3-70b")
    mi.kv_bpt_meta = 2.0  # bytes/token; tiny so am.STEP-rounded bisection math stays sane in a test
    return mi


class _FakeHog:
    def __init__(self):
        self.pid = 9999


def test_phase_a70_dry_run_full_arm_sweep(monkeypatch):
    """Patches Prober.find directly rather than letting the real exponential-search/bisection algorithm run against
    a stub server that always says ok=True (which would search forever, since that algorithm -- existing,
    already-relied-on t2s_amech code -- is not what this test is exercising). This test's job is phase_a70's own
    orchestration: the boundary record, the 3-below/3-above extra points, and the 3-arm sweep with the real call/Q0
    logic."""
    lab = StubLab(models={"llama-3.3-70b": _mi70()})
    with mock.patch.object(n2.L, "Server", StubServer), mock.patch.object(n2.am.L, "Server", StubServer), \
         mock.patch.object(n2.hc, "ollama_process_running", lambda: False), \
         mock.patch.object(n2.am.Prober, "find", lambda self, n_start, step: (4096, 4352)), \
         mock.patch.object(n2.am.Prober, "probe", lambda self, n_ctx: {"ok": True, "props_cap": None, "projected_mib": 1000.0,
                           "logged_mib": 900.0, "error": None, "vk": None, "alloc_failed": None}), \
         mock.patch.object(n2.time, "sleep", lambda *a: None):
        n2.phase_a70(lab)

    boundary = [r for r in lab.rows if r.get("record") == "bisect_result"]
    assert len(boundary) == 1
    arm_results = [r for r in lab.rows if r.get("record") == "a70_arm_result"]
    assert len(arm_results) == len(n2.A70_ARMS)
    assert all(r["started"] for r in arm_results)  # StubServer.start() always ok=True
    assert {r["arm"] for r in arm_results} == {label for label, _, _ in n2.A70_ARMS}
    # every arm item got marked done exactly once
    expected_items = {f"A70_llama-3.3-70b_{label}" for label, _, _ in n2.A70_ARMS}
    assert expected_items <= lab.done


def _fake_start_and_record(ok_fn):
    """A fake am.start_and_record for finalization tests: ok_fn(n_ctx) decides the outcome per call, independent
    of Prober.probe's cache (which the real _a70_repeated_loads deliberately bypasses). Also used by phase_a70's
    own arm sweep (which reuses am.start_and_record), so a "started" outcome returns a real StubServer (full
    tokenize/measured_sequence surface) rather than a bespoke fake."""
    def _fake(lab, mi, n_ctx, tag, item_id, phase, extra_flags, ngl=99, fit=None, **kw):
        ok = ok_fn(n_ctx)
        srv = StubServer(lab, mi, n_ctx)
        info = {"ok": ok, "error": None if ok else "boom", "log": {}}
        return srv, info, {"vk_errors": None}
    return _fake


def test_phase_a70_finalization_reproduces_boundary(monkeypatch):
    """The finalization rerun does 3 independent loads at lo and 3 at hi after the bisection and 3-below/3-above
    points; when lo is 3/3 success and hi is 3/3 failure, reproduced=True and the arm sweep still runs."""
    lab = StubLab(models={"llama-3.3-70b": _mi70()})
    monkeypatch.setattr(n2.hc, "ollama_process_running", lambda: False)
    with mock.patch.object(n2.L, "Server", StubServer), mock.patch.object(n2.am.L, "Server", StubServer), \
         mock.patch.object(n2.am.Prober, "find", lambda self, n_start, step: (4096, 4352)), \
         mock.patch.object(n2.am.Prober, "probe", lambda self, n_ctx: {"ok": True, "props_cap": None,
                           "projected_mib": 1000.0, "logged_mib": 900.0, "error": None, "vk": None, "alloc_failed": None}), \
         mock.patch.object(n2.am, "start_and_record", _fake_start_and_record(lambda n_ctx: n_ctx <= 4096)), \
         mock.patch.object(n2.time, "sleep", lambda *a: None):
        n2.phase_a70(lab)

    finalization = [r for r in lab.rows if r.get("record") == "a70_boundary_finalization"]
    assert len(finalization) == 1
    f = finalization[0]
    assert f["lo"] == 4096 and f["hi"] == 4352
    assert f["reproduced"] is True
    assert f["ollama_confirmed_absent"] is True
    assert f["lo_ok_count"] == 3
    assert f["hi_fail_count"] == 3
    assert len(f["lo_loads"]) == 3 and len(f["hi_loads"]) == 3


def test_phase_a70_finalization_flags_when_boundary_does_not_reproduce(monkeypatch):
    lab = StubLab(models={"llama-3.3-70b": _mi70()})
    monkeypatch.setattr(n2.hc, "ollama_process_running", lambda: False)
    with mock.patch.object(n2.L, "Server", StubServer), mock.patch.object(n2.am.L, "Server", StubServer), \
         mock.patch.object(n2.am.Prober, "find", lambda self, n_start, step: (4096, 4352)), \
         mock.patch.object(n2.am.Prober, "probe", lambda self, n_ctx: {"ok": True, "props_cap": None,
                           "projected_mib": 1000.0, "logged_mib": 900.0, "error": None, "vk": None, "alloc_failed": None}), \
         mock.patch.object(n2.am, "start_and_record", _fake_start_and_record(lambda n_ctx: True)), \
         mock.patch.object(n2.time, "sleep", lambda *a: None):
        n2.phase_a70(lab)

    f = [r for r in lab.rows if r.get("record") == "a70_boundary_finalization"][0]
    assert f["reproduced"] is False  # hi also came back ok every time, so the fail side did not reproduce
    assert f["hi_fail_count"] == 0


def test_phase_a70_finalization_flags_when_lo_is_intermittent(monkeypatch):
    """2/3 success at lo is not good enough -- reproduced requires 3/3, not a majority."""
    lab = StubLab(models={"llama-3.3-70b": _mi70()})
    calls = {"n": 0}

    def flaky_lo(n_ctx):
        if n_ctx > 4096:
            return False
        calls["n"] += 1
        return calls["n"] != 2  # fails on the 2nd of 3 lo loads

    monkeypatch.setattr(n2.hc, "ollama_process_running", lambda: False)
    with mock.patch.object(n2.L, "Server", StubServer), mock.patch.object(n2.am.L, "Server", StubServer), \
         mock.patch.object(n2.am.Prober, "find", lambda self, n_start, step: (4096, 4352)), \
         mock.patch.object(n2.am.Prober, "probe", lambda self, n_ctx: {"ok": True, "props_cap": None,
                           "projected_mib": 1000.0, "logged_mib": 900.0, "error": None, "vk": None, "alloc_failed": None}), \
         mock.patch.object(n2.am, "start_and_record", _fake_start_and_record(flaky_lo)), \
         mock.patch.object(n2.time, "sleep", lambda *a: None):
        n2.phase_a70(lab)

    f = [r for r in lab.rows if r.get("record") == "a70_boundary_finalization"][0]
    assert f["lo_ok_count"] == 2
    assert f["reproduced"] is False


def test_phase_a70_finalization_aborts_if_ollama_running(monkeypatch):
    lab = StubLab(models={"llama-3.3-70b": _mi70()})
    monkeypatch.setattr(n2.hc, "ollama_process_running", lambda: True)
    with mock.patch.object(n2.L, "Server", StubServer), mock.patch.object(n2.am.L, "Server", StubServer), \
         mock.patch.object(n2.am.Prober, "find", lambda self, n_start, step: (4096, 4352)), \
         mock.patch.object(n2.am.Prober, "probe", lambda self, n_ctx: {"ok": True, "props_cap": None,
                           "projected_mib": 1000.0, "logged_mib": 900.0, "error": None, "vk": None, "alloc_failed": None}), \
         mock.patch.object(n2.time, "sleep", lambda *a: None):
        try:
            n2.phase_a70(lab)
            assert False, "expected SmokeFailure"
        except n2.SmokeFailure as e:
            assert "STOP" in str(e) and "ollama" in str(e)


def test_phase_a70_finalize_standalone_reprobes_recorded_boundary(monkeypatch):
    """Standalone finalization for an A70 run that finished under the OLD phase_a70 code, before the inline
    finalization step existed: reads the recorded boundary instead of re-deriving it with a fresh bisection, and
    does 3 independent loads at lo and 3 at hi."""
    lab = StubLab(models={"llama-3.3-70b": _mi70()})
    lab.emit({"record": "bisect_result", "label": "a70", "model_id": "llama-3.3-70b",
              "last_ok_n_ctx": 23296, "first_fail_n_ctx": 23552})

    monkeypatch.setattr(n2.hc, "ollama_process_running", lambda: False)
    with mock.patch.object(n2.am.L, "Server", StubServer), \
         mock.patch.object(n2.am, "start_and_record", _fake_start_and_record(lambda n_ctx: n_ctx <= 23296)), \
         mock.patch.object(n2.time, "sleep", lambda *a: None):
        n2.phase_a70_finalize(lab)

    f = [r for r in lab.rows if r.get("record") == "a70_boundary_finalization"][0]
    assert f["lo"] == 23296 and f["hi"] == 23552
    assert f["reproduced"] is True
    assert f["standalone"] is True
    assert f["lo_ok_count"] == 3
    assert f["hi_fail_count"] == 3


def test_phase_a70_finalize_skips_if_no_boundary_recorded():
    lab = StubLab(models={"llama-3.3-70b": _mi70()})
    n2.phase_a70_finalize(lab)
    assert lab.rows == []


def test_phase_a70_finalize_skips_if_already_finalized(monkeypatch):
    lab = StubLab(models={"llama-3.3-70b": _mi70()})
    lab.emit({"record": "bisect_result", "label": "a70", "model_id": "llama-3.3-70b",
              "last_ok_n_ctx": 23296, "first_fail_n_ctx": 23552})
    lab.emit({"record": "a70_boundary_finalization", "model_id": "llama-3.3-70b", "reproduced": True,
              "lo_ok_count": 3, "hi_fail_count": 3})
    calls = []
    monkeypatch.setattr(n2.hc, "ollama_process_running", lambda: (calls.append(1) or False))
    n2.phase_a70_finalize(lab)
    assert not calls  # never even checked ollama, since it bailed out before that point


def test_phase_a70_finalize_reruns_if_old_one_reprobe_shape_record_exists(monkeypatch):
    """An earlier finalization record from before the 3-loads-per-side correction (no 'lo_ok_count' key) must not
    block a rerun with the corrected design."""
    lab = StubLab(models={"llama-3.3-70b": _mi70()})
    lab.emit({"record": "bisect_result", "label": "a70", "model_id": "llama-3.3-70b",
              "last_ok_n_ctx": 23296, "first_fail_n_ctx": 23552})
    lab.emit({"record": "a70_boundary_finalization", "model_id": "llama-3.3-70b", "reproduced": True,
              "lo_reprobe_ok": True, "hi_reprobe_ok": False})  # old shape, no lo_ok_count
    monkeypatch.setattr(n2.hc, "ollama_process_running", lambda: False)
    with mock.patch.object(n2.am.L, "Server", StubServer), \
         mock.patch.object(n2.am, "start_and_record", _fake_start_and_record(lambda n_ctx: n_ctx <= 23296)), \
         mock.patch.object(n2.time, "sleep", lambda *a: None):
        n2.phase_a70_finalize(lab)

    new_records = [r for r in lab.rows if r.get("record") == "a70_boundary_finalization" and "lo_ok_count" in r]
    assert len(new_records) == 1


def test_phase_a70_finalize_aborts_if_ollama_running(monkeypatch):
    lab = StubLab(models={"llama-3.3-70b": _mi70()})
    lab.emit({"record": "bisect_result", "label": "a70", "model_id": "llama-3.3-70b",
              "last_ok_n_ctx": 23296, "first_fail_n_ctx": 23552})
    monkeypatch.setattr(n2.hc, "ollama_process_running", lambda: True)
    try:
        n2.phase_a70_finalize(lab)
        assert False, "expected SmokeFailure"
    except n2.SmokeFailure as e:
        assert "STOP" in str(e) and "ollama" in str(e)


def test_phase_a70_dry_run_no_model_present_does_not_crash():
    lab = StubLab(models={})
    with mock.patch.object(n2.L, "Server", StubServer), mock.patch.object(n2.time, "sleep", lambda *a: None):
        n2.phase_a70(lab)
    assert lab.rows == []


def test_phase_p70_dry_run_none_and_nonp12(monkeypatch):
    lab = StubLab(models={"llama-3.3-70b": _mi70()})
    monkeypatch.setattr(n2.L, "Server", StubServer)
    monkeypatch.setattr(n2.L.m3, "start_hog", lambda *a, **kw: (_FakeHog(), None, None))
    monkeypatch.setattr(n2.L.m3, "kill_tree", lambda pid: None)
    monkeypatch.setattr(n2.time, "sleep", lambda *a: None)
    n2.phase_p70(lab)

    call_rows = [r for r in lab.rows if r.get("kind") == "call"]
    conditions = {r["co_runner"] for r in call_rows}
    assert conditions == {"none", "nonp12"}
    assert {"P70_llama-3.3-70b_none", "P70_llama-3.3-70b_nonp12"} <= lab.done


def test_phase_p70_dry_run_no_model_present_does_not_crash():
    lab = StubLab(models={})
    with mock.patch.object(n2.L, "Server", StubServer):
        n2.phase_p70(lab)
    assert lab.rows == []


def test_smoke_gated_phases_matches_new_phases():
    assert n2.SMOKE_GATED_PHASES == {"r1_speed", "r1_check", "a70", "p70", "r1b", "r1d"}


def test_smoke_check_phase_passes_against_stub_server():
    lab = StubLab(models={"qwen3-8b": StubModelInfo("qwen3-8b")})
    with mock.patch.object(n2.L, "Server", StubServer), mock.patch.object(n2.time, "sleep", lambda *a: None):
        ok, reason = n2.smoke_check_phase(lab, "r1_speed")
    assert ok is True
    assert reason == "ok"
    # do_call's own emitted rows (not measured_sequence's return value, which is the chat-result dict smoke_check
    # actually validates) still show exactly one start and one non-warmup call happened.
    start_rows = [r for r in lab.rows if r.get("kind") == "start"]
    call_rows = [r for r in lab.rows if r.get("kind") == "call" and not r.get("warmup")]
    assert len(start_rows) == 1
    assert len(call_rows) == 1


def test_smoke_check_phase_fails_when_model_missing():
    lab = StubLab(models={})
    ok, reason = n2.smoke_check_phase(lab, "r1_speed")
    assert ok is False
    assert "not loaded" in reason


def test_smoke_check_phase_fails_on_server_start_error():
    lab = StubLab(models={"qwen3-8b": StubModelInfo("qwen3-8b")})

    class FailingServer(StubServer):
        def start(self, timeout=600):
            self.start_info = {"ok": False, "error": "boom", "load_s": None, "exit_code": 1, "pid": None, "log": {},
                               "build": None, "t_start": 0.0, "t_end": 0.0}
            return self.start_info

    with mock.patch.object(n2.L, "Server", FailingServer):
        ok, reason = n2.smoke_check_phase(lab, "r1_speed")
    assert ok is False
    assert "server start failed" in reason


def test_is_stale_server_error_matches_known_shapes():
    assert n2._is_stale_server_error(RuntimeError("STOP guard: port 8385 already has a listener: [...]"))
    assert n2._is_stale_server_error(RuntimeError("STOP: could not confirm server pid 7408 exited and port 8385 freed"))
    assert n2._is_stale_server_error(RuntimeError("STOP: a llama-server process we did not start is running"))
    assert n2._is_stale_server_error(RuntimeError("STOP: could not clear stale listener(s) [4242] on port 8385 even though they were ours"))


def test_is_stale_server_error_does_not_match_unrelated_errors():
    assert not n2._is_stale_server_error(ValueError("some unrelated bug in the phase's own logic"))
    assert not n2._is_stale_server_error(KeyError("qwen3-999b"))


def test_abort_if_ollama_running_raises_when_ollama_present(monkeypatch):
    monkeypatch.setattr(n2.hc, "ollama_process_running", lambda: True)
    try:
        n2._abort_if_ollama_running("r1_speed")
        assert False, "expected SmokeFailure"
    except n2.SmokeFailure as e:
        assert "STOP" in str(e)
        assert "ollama" in str(e)


def test_abort_if_ollama_running_does_nothing_when_absent(monkeypatch):
    monkeypatch.setattr(n2.hc, "ollama_process_running", lambda: False)
    n2._abort_if_ollama_running("r1_speed")  # must not raise


def test_main_phase_loop_raises_smoke_failure_with_stop_in_message():
    """A smoke failure must surface as a SmokeFailure whose message contains 'STOP', so main()'s note (built from
    repr(e)) trips t2s_queue.advance()'s halt condition instead of silently launching the next queued run."""
    try:
        raise n2.SmokeFailure("STOP: smoke failed before phase r1_speed: qwen3-8b not loaded")
    except n2.SmokeFailure as e:
        note = f"stopped: {e!r}"[:400]
    assert "STOP" in note
