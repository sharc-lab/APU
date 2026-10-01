"""Unit tests for host_config.wait_for_ollama_ready (added 2026-10-01, A1/A2 session): a real race found live
on evo-t2s where phase_tier_v3's first pull, issued 9s after start_ollama_server()'s WMI-reported pid, failed
with a connection error because the HTTP server was not actually listening yet."""
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))

import host_config as hc  # noqa: E402


class _FakeResponse:
    def __init__(self, status=200):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_wait_for_ollama_ready_returns_true_immediately_when_already_up(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=3: _FakeResponse(200))
    assert hc.wait_for_ollama_ready(timeout_s=5, poll_interval_s=0.01) is True


def test_wait_for_ollama_ready_retries_until_success(monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(req, timeout=3):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionRefusedError("not up yet")
        return _FakeResponse(200)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert hc.wait_for_ollama_ready(timeout_s=5, poll_interval_s=0.01) is True
    assert calls["n"] == 3


def test_wait_for_ollama_ready_times_out_and_returns_false_without_raising(monkeypatch):
    def always_fails(req, timeout=3):
        raise ConnectionRefusedError("never comes up")

    monkeypatch.setattr(urllib.request, "urlopen", always_fails)
    assert hc.wait_for_ollama_ready(timeout_s=0.1, poll_interval_s=0.02) is False
