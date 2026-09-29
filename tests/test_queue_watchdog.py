"""Dry-run tests for harness/queue_watchdog.py against a stub queue file, one per required case (running-alive,
running-dead, pending-only, empty), plus t2s_queue's new launch_next/_parse_pid pid-tracking. No real process, no
real scheduled task."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))

import queue_watchdog as wd  # noqa: E402
import t2s_queue as q  # noqa: E402


def _isolate(tmp_path, monkeypatch):
    queue_file = tmp_path / "queue_state.json"
    flag_file = tmp_path / "queue_empty.flag"
    monkeypatch.setattr(q, "QUEUE_FILE", queue_file)
    monkeypatch.setattr(q, "EMPTY_FLAG", flag_file)
    monkeypatch.setattr(q, "_ps", lambda script: ("rc=0 pid=4242", ""))  # launch_next's own WMI call, fully stubbed
    return queue_file, flag_file


# ---------------------------------------------------------------- t2s_queue.launch_next / _parse_pid
def test_parse_pid_extracts_int_or_none():
    assert q._parse_pid("rc=0 pid=1234") == 1234
    assert q._parse_pid("rc=0 pid=1234 | stderr: ") == 1234
    assert q._parse_pid("") is None
    assert q._parse_pid(None) is None
    assert q._parse_pid("garbage, no rc or pid here") is None


def test_launch_next_records_pid_on_the_launched_entry(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    q.write_queue([{"id": "a", "cmd": ["echo", "hi"], "status": "pending"}])
    launched = q.launch_next(q.read_queue())
    assert launched["status"] == "running"
    assert launched["pid"] == 4242


# ---------------------------------------------------------------- queue_watchdog.tick
def test_tick_running_alive_does_nothing(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    q.write_queue([{"id": "a", "cmd": ["echo"], "status": "running", "pid": 111}])
    result = wd.tick(pid_alive_fn=lambda pid: pid == 111, run_end_fn=lambda e: False)
    assert result["action"] == "none"
    assert q.read_queue()[0]["status"] == "running"


def test_tick_running_dead_no_run_end_marks_crashed_and_launches_next(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    q.write_queue([{"id": "a", "cmd": ["echo"], "status": "running", "pid": 111},
                   {"id": "b", "cmd": ["echo", "next"], "status": "pending"}])
    result = wd.tick(pid_alive_fn=lambda pid: False, run_end_fn=lambda e: False)
    assert result["action"] == "crashed_launched_next"
    items = q.read_queue()
    assert items[0]["status"] == "crashed"
    assert items[1]["status"] == "running"
    assert items[1]["pid"] == 4242


def test_tick_running_dead_but_run_end_written_does_nothing(tmp_path, monkeypatch):
    """The run actually finished cleanly and its own finally block already called advance() -- the watchdog must not
    double-advance on top of that (e.g. if it ticks in the narrow window right as the process is exiting)."""
    _isolate(tmp_path, monkeypatch)
    q.write_queue([{"id": "a", "cmd": ["echo"], "status": "running", "pid": 111},
                   {"id": "b", "cmd": ["echo", "next"], "status": "pending"}])
    result = wd.tick(pid_alive_fn=lambda pid: False, run_end_fn=lambda e: True)
    assert result["action"] == "none"
    items = q.read_queue()
    assert items[0]["status"] == "running"  # untouched; advance() (called by the finishing run itself) owns this
    assert items[1]["status"] == "pending"


def test_tick_running_dead_crash_with_no_pending_writes_empty_flag(tmp_path, monkeypatch):
    queue_file, flag_file = _isolate(tmp_path, monkeypatch)
    q.write_queue([{"id": "a", "cmd": ["echo"], "status": "running", "pid": 111}])
    result = wd.tick(pid_alive_fn=lambda pid: False, run_end_fn=lambda e: False)
    assert result["action"] == "crashed_no_next"
    assert flag_file.exists()
    assert q.read_queue()[0]["status"] == "crashed"


def test_tick_pending_only_launches_next(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    q.write_queue([{"id": "a", "cmd": ["echo"], "status": "done"},
                   {"id": "b", "cmd": ["echo", "next"], "status": "pending"}])
    result = wd.tick(pid_alive_fn=lambda pid: False, run_end_fn=lambda e: False)
    assert result["action"] == "launched"
    assert result["launched"] == "b"
    assert q.read_queue()[1]["status"] == "running"


def test_tick_empty_writes_flag(tmp_path, monkeypatch):
    queue_file, flag_file = _isolate(tmp_path, monkeypatch)
    q.write_queue([{"id": "a", "cmd": ["echo"], "status": "done"}])
    result = wd.tick(pid_alive_fn=lambda pid: False, run_end_fn=lambda e: False)
    assert result["action"] == "empty_flag"
    assert flag_file.exists()
    flag = json.loads(flag_file.read_text(encoding="utf-8"))
    assert "reason" in flag


# ---------------------------------------------------------------- run_end_written
def test_run_end_written_finds_record_in_resumed_stem_file(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "myrun.jsonl").write_text('{"kind": "call"}\n{"record": "run_end", "note": "completed"}\n', encoding="utf-8")
    entry = {"cmd": ["python", "x.py", "--resume", "myrun"]}
    assert wd.run_end_written(entry, results_dir=results) is True


def test_run_end_written_false_when_no_run_end_record(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "myrun.jsonl").write_text('{"kind": "call"}\n', encoding="utf-8")
    entry = {"cmd": ["python", "x.py", "--resume", "myrun"]}
    assert wd.run_end_written(entry, results_dir=results) is False


def test_run_end_written_false_when_stem_file_missing():
    entry = {"cmd": ["python", "x.py", "--resume", "does_not_exist"]}
    assert wd.run_end_written(entry, results_dir=Path("C:/definitely/not/a/real/dir")) is False
