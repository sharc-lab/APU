"""Unit tests for the maintenance lock (A5, 2026-10-01): t2s_queue.set_pause/clear_pause/is_paused.

Built after the third self-inflicted collision: an operator's ad-hoc debug run (run_positive_control's own
start/stop_ollama_server() calls) killed x2_k2's own Ollama server mid-run, because nothing told the queue
watchdog a human was actively working on the machine. See queue_watchdog.py's tick() docstring for how this
is wired in (paused -> tick() does nothing at all, not even crash detection)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))

import t2s_queue as q  # noqa: E402


def _isolate(tmp_path, monkeypatch):
    pause_file = tmp_path / "queue_pause.flag"
    monkeypatch.setattr(q, "PAUSE_FLAG", pause_file)
    return pause_file


def test_is_paused_false_by_default(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    assert q.is_paused() is None


def test_set_pause_then_is_paused_returns_reason(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    q.set_pause("operator debugging x2_k2")
    rec = q.is_paused()
    assert rec is not None
    assert rec["reason"] == "operator debugging x2_k2"
    assert "ts_utc_epoch" in rec


def test_clear_pause_removes_the_flag(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    q.set_pause("x")
    q.clear_pause()
    assert q.is_paused() is None


def test_clear_pause_is_a_noop_when_not_set(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    q.clear_pause()  # must not raise
    assert q.is_paused() is None


def test_is_paused_fails_safe_on_unparseable_flag_content(tmp_path, monkeypatch):
    """A flag file that exists but cannot be parsed must still count as paused (fail safe: block launches),
    never silently treated as 'not paused'."""
    pause_file = _isolate(tmp_path, monkeypatch)
    pause_file.write_text("not valid json {{{", encoding="utf-8")
    rec = q.is_paused()
    assert rec is not None
    assert "reason" in rec


def test_set_pause_overwrites_a_previous_pause(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    q.set_pause("first reason")
    q.set_pause("second reason")
    assert q.is_paused()["reason"] == "second reason"
