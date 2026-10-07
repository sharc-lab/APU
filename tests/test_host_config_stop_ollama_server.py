"""2026-10-06 bug found live, real data: x2_outcome_table_v2's per-call start/stop ollama cycle hit
"connection refused" on 409 of ~650 real ollama_default calls over the weekend. Root cause: Stop-Process
-Force returning does not guarantee the target process has actually finished terminating on Windows; the
NEXT call's start_ollama_server() own idempotency check then saw the dying zombie and skipped launching a
fresh server. stop_ollama_server() now polls ollama_process_running() until it reports False before
returning."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))

import host_config as hc  # noqa: E402


def test_stop_ollama_server_waits_until_process_actually_gone(monkeypatch):
    calls = {"n": 0}

    def fake_ps(cmd, timeout):
        return "stopped"

    def fake_running(ps_fn=None):
        calls["n"] += 1
        return calls["n"] < 3  # reports alive twice, then gone

    sleeps = []
    monkeypatch.setattr(hc, "ollama_process_running", fake_running)
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

    hc.stop_ollama_server(ps_fn=fake_ps, wait_s=5, poll_s=0.1)

    assert calls["n"] == 3  # polled until it actually reported gone
    assert len(sleeps) == 2  # slept between the two "still alive" checks


def test_stop_ollama_server_gives_up_after_wait_s_if_never_confirmed_gone(monkeypatch):
    """Never hangs forever -- a process that genuinely won't die within wait_s must not block the caller
    indefinitely; the caller (run_one_ollama) has its own retry/infra-failure path for this case."""
    monkeypatch.setattr(hc, "ollama_process_running", lambda ps_fn=None: True)  # never reports gone
    times = iter([0, 0, 1, 2, 3, 4, 5, 6])
    monkeypatch.setattr(time, "monotonic", lambda: next(times, 100))
    monkeypatch.setattr(time, "sleep", lambda s: None)

    out = hc.stop_ollama_server(ps_fn=lambda cmd, timeout: "stopped", wait_s=5, poll_s=0.1)
    assert out == "stopped"  # returns (does not hang/raise) once the deadline passes


def test_stop_ollama_server_returns_immediately_when_already_gone(monkeypatch):
    monkeypatch.setattr(hc, "ollama_process_running", lambda ps_fn=None: False)
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    hc.stop_ollama_server(ps_fn=lambda cmd, timeout: "stopped", wait_s=5, poll_s=0.1)
    assert sleeps == []  # never had to wait at all
