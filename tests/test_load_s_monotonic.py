"""Server.start()'s load_s must be a monotonic-clock duration, immune to a wall-clock step during the load (NTP
correction, DST, or the clock simply being wrong, as found live on evo-x2). t_start/t_end stay time.time() (UTC
anchors for t_start_utc and Telemetry.metrics() windowing), only load_s must not be their difference."""

import sys
import time
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
import t2s_lab as L  # noqa: E402


class _FakeProc:
    pid = 4242

    def poll(self):
        return None  # never "exits" during the test


def test_load_s_survives_a_wall_clock_jump_between_t0_and_t1(tmp_path):
    srv = L.Server.__new__(L.Server)
    srv.log_path = str(tmp_path / "srv.log")
    srv.n_ctx = 8192
    srv.backend = "vulkan"
    srv.ngl, srv.fit, srv.load_mode, srv.extra = 99, None, "auto", []
    srv.mi = mock.Mock(path="model.gguf")
    srv.lab = mock.Mock()

    real_time = time.time
    real_monotonic = time.monotonic
    # time.time() jumps forward by 10000 s between the two calls Server.start() makes for t0/t1; time.monotonic()
    # advances normally. A correct load_s reflects the tiny real elapsed time, not the injected jump.
    time_calls = {"n": 0}

    def fake_time():
        time_calls["n"] += 1
        return real_time() + (10000.0 if time_calls["n"] > 1 else 0.0)

    with mock.patch.object(L.sg, "assert_port_free", lambda *a, **k: None), \
         mock.patch.object(L, "ps", return_value="0"), \
         mock.patch.object(L.subprocess, "Popen", return_value=_FakeProc()), \
         mock.patch.object(L, "parse_server_log", return_value={"error_lines": []}), \
         mock.patch.object(L.time, "time", side_effect=fake_time):
        info = srv.start(timeout=0)  # loop body never runs: healthy stays False immediately

    assert info["ok"] is False
    assert info["load_s"] < 5.0, f"load_s should reflect real (monotonic) elapsed time, got {info['load_s']}"
    assert info["t_end"] - info["t_start"] >= 9999.0, "the injected wall-clock jump should still show up in t_start/t_end"
