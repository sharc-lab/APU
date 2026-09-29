"""ensure_port_free_or_cleanup() and Server.stop()'s escalation path, added after the PID 7408 incident (2026-09-29):
a leftover llama-server survived Server.stop()'s existing 3x45s taskkill retry and then blocked every subsequent
server start on port 8385 until the controller manually killed it. Never touches a real process or port; sg/m3/ps/
time are all mocked."""
import itertools
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
import t2s_lab as L  # noqa: E402


def _fast_clock():
    """An itertools.count-based time.monotonic() stand-in that jumps forward by 16s per call, so any deadline-loop
    test (30s or 45s windows) resolves in a couple of iterations instead of a real wait."""
    return mock.patch.object(L.time, "monotonic", side_effect=itertools.count(0, 16))


def test_ensure_port_free_returns_not_cleaned_when_port_is_free():
    with mock.patch.object(L.sg, "port_listeners", return_value=[]):
        result = L.ensure_port_free_or_cleanup(8385)
    assert result == {"cleaned": False, "pids": []}


def test_ensure_port_free_kills_our_own_leftover_and_reports_cleaned():
    calls = {"port_listeners": 0}

    def fake_listeners(port):
        calls["port_listeners"] += 1
        return [4242] if calls["port_listeners"] == 1 else []

    with mock.patch.object(L.sg, "port_listeners", side_effect=fake_listeners), \
         mock.patch.object(L.sg, "process_cmdline", return_value=r"C:\apu\bin\llama-b10970\llama-server.exe -m x"), \
         mock.patch.object(L.m3, "kill_tree") as kill_tree, \
         mock.patch.object(L.time, "sleep", lambda s: None):
        result = L.ensure_port_free_or_cleanup(8385)
    assert result == {"cleaned": True, "pids": [4242]}
    kill_tree.assert_called_once_with(4242)


def test_ensure_port_free_refuses_to_kill_a_process_that_is_not_ours():
    with mock.patch.object(L.sg, "port_listeners", return_value=[9999]), \
         mock.patch.object(L.sg, "process_cmdline", return_value=r"C:\Windows\System32\svchost.exe"), \
         mock.patch.object(L.m3, "kill_tree") as kill_tree:
        try:
            L.ensure_port_free_or_cleanup(8385)
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "NOT ours" in str(e)
    kill_tree.assert_not_called()


def test_ensure_port_free_raises_if_our_own_leftover_will_not_clear():
    with mock.patch.object(L.sg, "port_listeners", return_value=[4242]), \
         mock.patch.object(L.sg, "process_cmdline", return_value=r"C:\apu\bin\llama-b10970\llama-server.exe"), \
         mock.patch.object(L.m3, "kill_tree"), \
         mock.patch.object(L.time, "sleep", lambda s: None), _fast_clock():
        try:
            L.ensure_port_free_or_cleanup(8385)
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "could not clear" in str(e)


class _StubServerForStop:
    """Just enough of Server for .stop()'s own logic (not .start()): proc/pid/lab/_out, no real subprocess."""
    def __init__(self):
        self.proc = mock.Mock()
        self.pid = 4242
        self.lab = mock.Mock()
        self._out = mock.Mock()

    stop = L.Server.stop


def test_server_stop_last_resort_kills_a_different_pid_squatting_the_port_if_ours():
    """The PID 7408 scenario generalized: self.pid never clears in 3x45s, but by the last-resort check a DIFFERENT
    pid is squatting the port (e.g. self.pid finally died and something else raced onto the port) -- killed only
    because its command line is ours."""
    srv = _StubServerForStop()
    alive_then_dead = itertools.chain([True] * 20, [False])  # "alive" checks: stays alive through all 3 escalations

    def fake_ps(cmd, timeout=20):
        return "1" if next(alive_then_dead, False) else "0"

    with mock.patch.object(L, "ps", side_effect=fake_ps), \
         mock.patch.object(L.m3, "kill_tree") as kill_tree, \
         mock.patch.object(L.sg, "port_listeners", side_effect=[[4242]] * 100), \
         mock.patch.object(L.sg, "process_cmdline", return_value=r"C:\apu\bin\llama-b10970\llama-server.exe"), \
         mock.patch.object(L.time, "sleep", lambda s: None), _fast_clock():
        try:
            srv.stop()
            assert False, "expected RuntimeError (port never actually clears in this test)"
        except RuntimeError as e:
            assert "could not confirm" in str(e)
    assert kill_tree.call_count >= 3  # 3 escalation attempts + at least the last-resort attempt on pid 4242


def test_server_stop_refuses_to_kill_a_non_ours_process_at_last_resort():
    srv = _StubServerForStop()
    with mock.patch.object(L, "ps", return_value="0"), \
         mock.patch.object(L.m3, "kill_tree") as kill_tree, \
         mock.patch.object(L.sg, "port_listeners", return_value=[]), \
         mock.patch.object(L.time, "sleep", lambda s: None):
        srv.stop()  # port already free on the very first check: returns cleanly, no escalation needed
    kill_tree.assert_called_once()  # only the initial kill_tree(self.pid) attempt, no last-resort logic reached
    srv.lab.tele.set_pid.assert_called_once_with(None)
