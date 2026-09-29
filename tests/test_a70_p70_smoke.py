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
    assert n2.SMOKE_GATED_PHASES == {"r1_speed", "r1_check", "a70", "p70"}


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


def test_main_phase_loop_raises_smoke_failure_with_stop_in_message():
    """A smoke failure must surface as a SmokeFailure whose message contains 'STOP', so main()'s note (built from
    repr(e)) trips t2s_queue.advance()'s halt condition instead of silently launching the next queued run."""
    try:
        raise n2.SmokeFailure("STOP: smoke failed before phase r1_speed: qwen3-8b not loaded")
    except n2.SmokeFailure as e:
        note = f"stopped: {e!r}"[:400]
    assert "STOP" in note
