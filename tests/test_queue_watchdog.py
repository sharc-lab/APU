"""Dry-run tests for harness/queue_watchdog.py against a stub queue file, one per required case (running-alive,
pid-check-disagrees-with-heartbeat, really-dead, pending-only, empty), plus t2s_queue's new launch_next/_parse_pid
pid-tracking and failed-launch-reverts-to-pending behavior. No real process, no real scheduled task.

2026-09-29 incident (see queue_watchdog.py's module docstring): the old single-signal PID check produced a false
crash on evo-t2s (pid died, real work kept running) and the old run_end-guessing heuristic produced a false negative
on evo-x2 (a truly dead job was reported as "already finished" for 10+ hours). These tests cover the two cases the
fix (pid check AND heartbeat staleness must agree) is built for."""
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


def test_launch_next_reverts_to_pending_on_launch_failure_and_tries_next(tmp_path, monkeypatch):
    """A launch that produces no usable pid (WMI Create failed) must never be left as 'running, pid None' -- it
    reverts straight back to pending, and the next eligible pending entry is tried instead."""
    _isolate(tmp_path, monkeypatch)
    monkeypatch.setattr(q, "_ps", lambda script: ("rc=-2147024891 pid=", "Access is denied"))
    items = [{"id": "a", "cmd": ["echo", "hi"], "status": "pending"},
             {"id": "b", "cmd": ["echo", "next"], "status": "pending"}]
    monkeypatch.setattr(q, "_launch", lambda cmd, log_path: "rc=-2147024891 pid=")
    launched = q.launch_next(items)
    assert launched is None  # both attempts failed to produce a pid
    assert items[0]["status"] == "pending"
    assert "last_launch_failure" in items[0]
    assert items[1]["status"] == "pending"
    assert "last_launch_failure" in items[1]


def test_launch_next_skips_failed_launch_and_succeeds_on_next(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    calls = {"n": 0}

    def fake_launch(cmd, log_path):
        calls["n"] += 1
        return "rc=-1 pid=" if calls["n"] == 1 else "rc=0 pid=4242"

    monkeypatch.setattr(q, "_launch", fake_launch)
    items = [{"id": "a", "cmd": ["echo", "hi"], "status": "pending"},
             {"id": "b", "cmd": ["echo", "next"], "status": "pending"}]
    launched = q.launch_next(items)
    assert launched is not None
    assert launched["id"] == "b"
    assert launched["pid"] == 4242
    assert items[0]["status"] == "pending"  # "a" reverted, never marked running or crashed


# ---------------------------------------------------------------- queue_watchdog heartbeat helpers
def test_heartbeat_age_s_none_when_log_missing(tmp_path):
    entry = {"id": "nope"}
    assert wd.heartbeat_age_s(entry, log_dir=tmp_path) is None


def test_heartbeat_age_s_computed_from_log_mtime(tmp_path):
    log = tmp_path / "queue_a.log"
    log.write_text("hi", encoding="utf-8")
    import os, time as _time
    old = _time.time() - 120
    os.utime(log, (old, old))
    age = wd.heartbeat_age_s({"id": "a"}, log_dir=tmp_path, now=_time.time())
    assert 115 < age < 125


def test_is_70b_phase_detects_model_name_in_cmd():
    assert wd.is_70b_phase({"cmd": ["python", "t2s_night2.py", "--r1-check-models", "llama-3.3-70b"]}) is True
    assert wd.is_70b_phase({"cmd": ["python", "t2s_night2.py", "--r1-check-models", "qwen3-8b"]}) is False


def test_is_stale_uses_longer_threshold_for_70b():
    non70b = {"cmd": ["python", "x.py", "--models", "qwen3-8b"]}
    is70b = {"cmd": ["python", "x.py", "--models", "llama-3.3-70b"]}
    # 40 minutes: stale for a non-70b phase (30 min threshold), fresh for a 70B phase (60 min threshold)
    age = 40 * 60
    assert wd.is_stale(non70b, age) is True
    assert wd.is_stale(is70b, age) is False


def test_is_stale_none_age_is_always_stale():
    assert wd.is_stale({"cmd": []}, None) is True


# ---------------------------------------------------------------- queue_watchdog.tick
def test_tick_running_alive_does_nothing(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    q.write_queue([{"id": "a", "cmd": ["echo"], "status": "running", "pid": 111}])
    result = wd.tick(pid_alive_fn=lambda pid: pid == 111, heartbeat_age_fn=lambda e: 0)
    assert result["action"] == "none"
    assert q.read_queue()[0]["status"] == "running"


def test_tick_running_no_pid_recorded_leaves_alone(tmp_path, monkeypatch):
    """A 'running' entry with no pid at all should not happen post-fix (launch_next reverts failed launches to
    pending), but if the watchdog ever sees one, it must not guess -- never mark a never-launched job crashed."""
    _isolate(tmp_path, monkeypatch)
    q.write_queue([{"id": "a", "cmd": ["echo"], "status": "running"}])
    result = wd.tick(pid_alive_fn=lambda pid: False, heartbeat_age_fn=lambda e: 0)
    assert result["action"] == "none"
    assert q.read_queue()[0]["status"] == "running"


def test_tick_pid_dead_but_heartbeat_fresh_disagrees_and_does_nothing(tmp_path, monkeypatch):
    """The 2026-09-29 evo-t2s false-crash case: pid check says dead, but the job's own log is still being written
    to (heartbeat fresh) -- trust the heartbeat, do not mark crashed, do not launch anything else."""
    _isolate(tmp_path, monkeypatch)
    q.write_queue([{"id": "a", "cmd": ["echo"], "status": "running", "pid": 111},
                   {"id": "b", "cmd": ["echo", "next"], "status": "pending"}])
    result = wd.tick(pid_alive_fn=lambda pid: False, heartbeat_age_fn=lambda e: 60.0)
    assert result["action"] == "pid_check_disagrees"
    items = q.read_queue()
    assert items[0]["status"] == "running"  # untouched
    assert items[1]["status"] == "pending"  # nothing launched on top of it


def test_tick_pid_dead_and_heartbeat_stale_marks_crashed_and_launches_next(tmp_path, monkeypatch):
    """The 2026-09-29 evo-x2 false-negative case, corrected: pid check says dead AND the heartbeat has been stale
    well past the threshold -- now it is actually safe to declare this job dead."""
    _isolate(tmp_path, monkeypatch)
    q.write_queue([{"id": "a", "cmd": ["echo"], "status": "running", "pid": 111},
                   {"id": "b", "cmd": ["echo", "next"], "status": "pending"}])
    result = wd.tick(pid_alive_fn=lambda pid: False, heartbeat_age_fn=lambda e: 3 * 3600)
    assert result["action"] == "crashed_launched_next"
    items = q.read_queue()
    assert items[0]["status"] == "crashed"
    assert "heartbeat stale" in items[0]["note"]
    assert items[1]["status"] == "running"
    assert items[1]["pid"] == 4242


def test_tick_pid_dead_and_no_heartbeat_log_at_all_marks_crashed(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    q.write_queue([{"id": "a", "cmd": ["echo"], "status": "running", "pid": 111}])
    result = wd.tick(pid_alive_fn=lambda pid: False, heartbeat_age_fn=lambda e: None)
    assert result["action"] == "crashed_no_next"
    assert q.read_queue()[0]["status"] == "crashed"


def test_tick_running_dead_crash_with_no_pending_writes_empty_flag(tmp_path, monkeypatch):
    queue_file, flag_file = _isolate(tmp_path, monkeypatch)
    q.write_queue([{"id": "a", "cmd": ["echo"], "status": "running", "pid": 111}])
    result = wd.tick(pid_alive_fn=lambda pid: False, heartbeat_age_fn=lambda e: 3 * 3600)
    assert result["action"] == "crashed_no_next"
    assert flag_file.exists()
    assert q.read_queue()[0]["status"] == "crashed"


def test_tick_pending_only_launches_next(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    q.write_queue([{"id": "a", "cmd": ["echo"], "status": "done"},
                   {"id": "b", "cmd": ["echo", "next"], "status": "pending"}])
    result = wd.tick(pid_alive_fn=lambda pid: False, heartbeat_age_fn=lambda e: None)
    assert result["action"] == "launched"
    assert result["launched"] == "b"
    assert q.read_queue()[1]["status"] == "running"


def test_tick_pending_but_gate_blocked_writes_flag_not_launched(tmp_path, monkeypatch):
    """A gated entry (e.g. r1b_r1d waiting on t2s_q0_token_calibration) that is not yet satisfied must not be
    reported as 'launched': nothing actually started this tick."""
    queue_file, flag_file = _isolate(tmp_path, monkeypatch)
    q.write_queue([{"id": "calib", "status": "error"},
                   {"id": "r1b_r1d", "status": "pending", "gate": {"requires_done": "calib"}}])
    result = wd.tick(pid_alive_fn=lambda pid: False, heartbeat_age_fn=lambda e: None)
    assert result["action"] == "gate_blocked"
    assert flag_file.exists()
    assert q.read_queue()[1]["status"] == "pending"


def test_tick_empty_writes_flag(tmp_path, monkeypatch):
    queue_file, flag_file = _isolate(tmp_path, monkeypatch)
    q.write_queue([{"id": "a", "cmd": ["echo"], "status": "done"}])
    result = wd.tick(pid_alive_fn=lambda pid: False, heartbeat_age_fn=lambda e: None)
    assert result["action"] == "empty_flag"
    assert flag_file.exists()
    flag = json.loads(flag_file.read_text(encoding="utf-8"))
    assert "reason" in flag


# ---------------------------------------------------------------------------------------------------- run_digest
def test_run_digest_skips_when_script_not_deployed(monkeypatch, tmp_path):
    monkeypatch.setattr(wd, "DEPLOY", tmp_path)  # no results_digest.py here
    result = wd.run_digest()
    assert result["action"] == "skipped"


def test_run_digest_runs_the_script_when_present(monkeypatch, tmp_path):
    script = tmp_path / "results_digest.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    monkeypatch.setattr(wd, "DEPLOY", tmp_path)
    result = wd.run_digest()
    assert result["action"] == "ran"
    assert result["rc"] == 0
    assert "ok" in result["stdout"]


def test_run_digest_never_raises_on_a_script_error(monkeypatch, tmp_path):
    script = tmp_path / "results_digest.py"
    script.write_text("raise SystemExit(1)\n", encoding="utf-8")
    monkeypatch.setattr(wd, "DEPLOY", tmp_path)
    result = wd.run_digest()
    assert result["action"] == "ran"
    assert result["rc"] == 1


def test_main_calls_digest_after_tick(monkeypatch, tmp_path):
    """The digest must run after every watchdog tick, per the standing instruction that results must not depend
    on a controller session being awake."""
    _isolate(tmp_path, monkeypatch)
    q.write_queue([{"id": "a", "cmd": ["echo"], "status": "done"}])
    calls = []
    monkeypatch.setattr(wd, "run_digest", lambda: (calls.append(1) or {"action": "ran", "rc": 0, "stdout": ""}))
    wd.main()
    assert calls == [1]
