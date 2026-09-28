"""Standing rule: 'If the queue is empty, notify me immediately rather than leaving the machine idle.' advance()
must write queue_empty.flag whenever it finishes and finds nothing pending to launch (empty queue, halted queue, or
no pending entry left), and must clear that flag the moment a next run actually launches."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))

import t2s_queue as q  # noqa: E402


def _isolate(tmp_path, monkeypatch):
    queue_file = tmp_path / "queue_state.json"
    flag_file = tmp_path / "queue_empty.flag"
    monkeypatch.setattr(q, "QUEUE_FILE", queue_file)
    monkeypatch.setattr(q, "EMPTY_FLAG", flag_file)
    return queue_file, flag_file


def test_advance_flags_when_no_pending_entry_left(tmp_path, monkeypatch):
    queue_file, flag_file = _isolate(tmp_path, monkeypatch)
    q.write_queue([{"id": "night3", "cmd": ["echo", "hi"], "status": "running"}])
    q.advance("completed")
    assert flag_file.exists()
    flag = json.loads(flag_file.read_text(encoding="utf-8"))
    assert "no pending entry" in flag["reason"]
    items = q.read_queue()
    assert items[0]["status"] == "done"


def test_advance_flags_on_empty_queue(tmp_path, monkeypatch):
    queue_file, flag_file = _isolate(tmp_path, monkeypatch)
    q.advance("completed")
    assert flag_file.exists()


def test_advance_flags_on_halt(tmp_path, monkeypatch):
    queue_file, flag_file = _isolate(tmp_path, monkeypatch)
    q.write_queue([{"id": "night3", "cmd": ["echo", "hi"], "status": "running"},
                   {"id": "backlog", "cmd": ["echo", "bye"], "status": "pending"}])
    q.advance("stopped: another interactive session is logged in")
    assert flag_file.exists()
    items = q.read_queue()
    assert items[1]["status"] == "pending"  # never launched: halted before reaching the pending entry


def test_advance_clears_stale_flag_when_next_run_launches(tmp_path, monkeypatch):
    queue_file, flag_file = _isolate(tmp_path, monkeypatch)
    flag_file.write_text("{}", encoding="utf-8")  # a stale flag from an earlier empty moment
    q.write_queue([{"id": "night3", "cmd": ["echo", "hi"], "status": "running"},
                   {"id": "backlog", "cmd": ["echo", "bye"], "status": "pending"}])
    monkeypatch.setattr(q, "_launch", lambda cmd, log_path: "rc=0 pid=1234")
    q.advance("completed")
    assert not flag_file.exists()
    items = q.read_queue()
    assert items[1]["status"] == "running"
