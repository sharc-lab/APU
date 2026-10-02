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
    pause_file = tmp_path / "queue_pause.flag"
    monkeypatch.setattr(q, "QUEUE_FILE", queue_file)
    monkeypatch.setattr(q, "EMPTY_FLAG", flag_file)
    monkeypatch.setattr(q, "PAUSE_FLAG", pause_file)
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
    monkeypatch.setattr(q, "_launch", lambda cmd, log_path, job_id: "rc=-2147024891 pid=")
    launched = q.launch_next(items)
    assert launched is None  # both attempts failed to produce a pid
    assert items[0]["status"] == "pending"
    assert "last_launch_failure" in items[0]
    assert items[1]["status"] == "pending"
    assert "last_launch_failure" in items[1]


def test_launch_next_skips_failed_launch_and_succeeds_on_next(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    calls = {"n": 0}

    def fake_launch(cmd, log_path, job_id):
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


# ---------------------------------------------------------------- maintenance lock (2026-10-01)
def test_tick_does_nothing_while_paused_even_with_a_dead_running_job(tmp_path, monkeypatch):
    """The whole point of the maintenance lock: while paused, tick() must not even run its own crash-detection
    logic, since an operator may have deliberately killed the running job's process tree as part of the
    debug procedure (set_pause -> stop job -> debug -> clear_pause) -- if tick() still crash-detected and
    launched the next pending entry mid-debug, the lock would not have prevented the third-collision scenario
    it exists to fix."""
    _isolate(tmp_path, monkeypatch)
    q.write_queue([{"id": "a", "cmd": ["echo"], "status": "running", "pid": 111},
                   {"id": "b", "cmd": ["echo"], "status": "pending"}])
    q.set_pause("operator debugging x2_k2")
    result = wd.tick(pid_alive_fn=lambda pid: False, heartbeat_age_fn=lambda e: 99999)
    assert result["action"] == "paused"
    items = q.read_queue()
    assert items[0]["status"] == "running"  # untouched -- not marked crashed
    assert items[1]["status"] == "pending"  # untouched -- not launched


def test_tick_resumes_normal_behavior_after_clear_pause(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    q.write_queue([{"id": "a", "cmd": ["echo"], "status": "running", "pid": 111}])
    q.set_pause("x")
    q.clear_pause()
    result = wd.tick(pid_alive_fn=lambda pid: pid == 111, heartbeat_age_fn=lambda e: 0)
    assert result["action"] == "none"  # normal running-alive path, not "paused"


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


def _no_circuit_kwargs(**overrides):
    """Default circuit-breaker injections for a test that does not care about the circuit breaker
    itself: no cleanup ever actually runs, no real sleep, no real file I/O for the small state file,
    and the crash does not look like a fast-fail or a repeat (so a plain 'requeued, not an immediate
    retry' path is exercised) unless a test overrides tail_signature_fn/now_fn/started_ts itself."""
    base = dict(
        cleanup_fn=lambda ownership: {"killed": [], "ports_freed": {}},
        sleep_fn=lambda s: None,
        now_fn=lambda: 10_000_000.0,  # far from any started_ts used below -> never a fast_fail by default
        ownership_fn=lambda p: False,
        load_circuit_state_fn=lambda: {"last_crashed_id": None, "last_crash_signature": None},
        save_circuit_state_fn=lambda state: None,
        tail_signature_fn=lambda entry: None,
    )
    base.update(overrides)
    return base


def test_tick_pid_dead_and_heartbeat_stale_marks_crashed_and_launches_next(tmp_path, monkeypatch):
    """The 2026-09-29 evo-x2 false-negative case, corrected: pid check says dead AND the heartbeat has been stale
    well past the threshold -- now it is actually safe to declare this job dead. 2026-10-02: the job is requeued
    (not left 'crashed') with its own attempt counter; since this is neither a fast-fail nor a repeat-of-last-error,
    no cleanup/wait runs and it goes to the back of the line, letting the next pending entry launch instead."""
    _isolate(tmp_path, monkeypatch)
    q.write_queue([{"id": "a", "cmd": ["echo"], "status": "running", "pid": 111, "started_ts": 0.0},
                   {"id": "b", "cmd": ["echo", "next"], "status": "pending"}])
    result = wd.tick(pid_alive_fn=lambda pid: False, heartbeat_age_fn=lambda e: 3 * 3600,
                     **_no_circuit_kwargs())
    assert result["action"] == "crashed_requeued"
    assert result["circuit_triggered"] is False
    items = q.read_queue()
    a = next(it for it in items if it["id"] == "a")
    assert a["status"] == "pending"  # requeued, never left "crashed"
    assert a["attempt"] == 2
    assert "heartbeat stale" in a["note"]
    b = next(it for it in items if it["id"] == "b")
    assert b["status"] == "running"
    assert b["pid"] == 4242
    assert items[-1]["id"] == "a"  # ordinary (non-circuit) requeue goes to the back of the line


def test_tick_pid_dead_and_no_heartbeat_log_at_all_marks_crashed(tmp_path, monkeypatch):
    """Requeuing always produces at least one pending entry (the job itself), so launch_next()
    immediately retries it within the same tick -- it ends up 'running' again (with a fresh pid),
    never stuck 'crashed'. attempt is incremented to prove it is a real retry, not the original run."""
    _isolate(tmp_path, monkeypatch)
    q.write_queue([{"id": "a", "cmd": ["echo"], "status": "running", "pid": 111, "started_ts": 0.0}])
    result = wd.tick(pid_alive_fn=lambda pid: False, heartbeat_age_fn=lambda e: None,
                     **_no_circuit_kwargs())
    assert result["action"] == "crashed_requeued"
    assert result["launched"] == "a"
    item = q.read_queue()[0]
    assert item["status"] == "running"
    assert item["pid"] == 4242  # the stubbed launch_next pid -- a genuinely new launch, not the old pid 111
    assert item["attempt"] == 2


def test_tick_launch_next_failing_after_requeue_writes_empty_flag(tmp_path, monkeypatch):
    """The genuine empty-flag case: requeuing succeeds, but the retry launch itself fails (e.g. WMI
    Create errors) -- launch_next reverts it to pending and returns None, so this tick must still
    report the queue as having nothing successfully running, not silently claim a launch happened."""
    queue_file, flag_file = _isolate(tmp_path, monkeypatch)
    monkeypatch.setattr(q, "_ps", lambda script: ("rc=1 pid=", "WMI Create failed"))
    q.write_queue([{"id": "a", "cmd": ["echo"], "status": "running", "pid": 111, "started_ts": 0.0}])
    result = wd.tick(pid_alive_fn=lambda pid: False, heartbeat_age_fn=lambda e: 3 * 3600,
                     **_no_circuit_kwargs())
    assert result["action"] == "crashed_requeued"
    assert result.get("launched") is None
    assert flag_file.exists()
    assert q.read_queue()[0]["status"] == "pending"


def test_tick_circuit_breaker_fast_fail_runs_cleanup_and_retries_immediately(tmp_path, monkeypatch):
    """A job that dies within CIRCUIT_FAST_FAIL_S of its own launch looks like a stale-state trip:
    cleanup runs, the wait happens, and the retried entry is placed FIRST so it launches next, ahead
    of other pending work."""
    _isolate(tmp_path, monkeypatch)
    q.write_queue([{"id": "a", "cmd": ["echo"], "status": "running", "pid": 111, "started_ts": 1000.0},
                   {"id": "b", "cmd": ["echo"], "status": "pending"}])
    cleanup_calls = []
    slept = []
    result = wd.tick(pid_alive_fn=lambda pid: False, heartbeat_age_fn=lambda e: 3 * 3600,
                     **_no_circuit_kwargs(
                         now_fn=lambda: 1010.0,  # 10s after started_ts -- well under the 120s fast-fail window
                         cleanup_fn=lambda ownership: cleanup_calls.append(ownership) or {"killed": [1, 2], "ports_freed": {}},
                         sleep_fn=lambda s: slept.append(s),
                     ))
    assert result["circuit_triggered"] is True
    assert len(cleanup_calls) == 1
    assert slept == [wd.CIRCUIT_CLEANUP_WAIT_S]
    # "a" was placed first in line, so launch_next() retries it immediately, ahead of "b"
    assert result["launched"] == "a"
    items = q.read_queue()
    a = next(it for it in items if it["id"] == "a")
    assert a["status"] == "running"  # successfully retried within the same tick
    assert a["attempt"] == 2
    b = next(it for it in items if it["id"] == "b")
    assert b["status"] == "pending"  # "b" did not jump ahead of the retry


def test_tick_circuit_breaker_repeat_error_also_triggers_cleanup(tmp_path, monkeypatch):
    """2 consecutive crashes with the same tail-of-log error signature also trigger the circuit
    breaker, even if the second one ran for a while (not a fast-fail by elapsed time alone)."""
    _isolate(tmp_path, monkeypatch)
    q.write_queue([{"id": "a", "cmd": ["echo"], "status": "running", "pid": 111, "started_ts": 0.0}])
    cleanup_calls = []
    same_error = "RuntimeError: STOP: a llama-server process we did not start is running"
    result = wd.tick(pid_alive_fn=lambda pid: False, heartbeat_age_fn=lambda e: 3 * 3600,
                     **_no_circuit_kwargs(
                         now_fn=lambda: 10_000.0,  # far past the fast-fail window
                         tail_signature_fn=lambda entry: same_error,
                         load_circuit_state_fn=lambda: {"last_crashed_id": "z", "last_crash_signature": same_error},
                         cleanup_fn=lambda ownership: cleanup_calls.append(ownership) or {"killed": [], "ports_freed": {}},
                     ))
    assert result["circuit_triggered"] is True
    assert len(cleanup_calls) == 1


def test_tick_gives_up_after_max_attempts_without_retrying_again(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    q.write_queue([{"id": "a", "cmd": ["echo"], "status": "running", "pid": 111, "started_ts": 0.0,
                   "attempt": wd.CIRCUIT_MAX_ATTEMPTS}])
    cleanup_calls = []
    result = wd.tick(pid_alive_fn=lambda pid: False, heartbeat_age_fn=lambda e: 3 * 3600,
                     **_no_circuit_kwargs(cleanup_fn=lambda ownership: cleanup_calls.append(ownership)))
    assert result["action"] == "failed_max_attempts"
    assert cleanup_calls == []  # exhausted -- no further cleanup/retry attempted
    a = q.read_queue()[0]
    assert a["status"] == "failed_max_attempts"
    assert a["id"] == "a"


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


# ---------------------------------------------------------------------------------------------------- progress-staleness (2026-09-30)
# Found 2026-09-30: two separate jobs, two separate machines, both reported "running" for hours past when they had
# actually finished or errored, because every liveness check only re-confirmed pid-alive/heartbeat-fresh without
# ever checking whether a NEW RESULT ROW had been written recently. These tests cover the new, distinct check.

def _phases(last_ts_map):
    """Builds a minimal phases dict shaped like results_digest.collect_phase_summaries' output, for the one field
    progress_stale_age_s/check_progress_stale actually read."""
    return {k: {"n": 1, "last_ts_utc": v, "metrics": {}, "contamination_tags": {}, "outcomes": {}, "files": []}
            for k, v in last_ts_map.items()}


def test_is_long_prompt_phase_detects_96k_plus_literal_in_cmd():
    entry = {"cmd": ["python", "t2s_k1_ollama.py", "--phase", "tier_v3", "--lengths", "16000,32000,96000"]}
    assert wd.is_long_prompt_phase(entry) is True


def test_is_long_prompt_phase_false_for_short_prompts_only():
    entry = {"cmd": ["python", "t2s_night2.py", "--phases", "r1b", "--r1b-models", "qwen3-8b,qwen3-14b"]}
    assert wd.is_long_prompt_phase(entry) is False


def test_needs_long_progress_threshold_true_for_70b_or_long_prompt():
    assert wd.needs_long_progress_threshold({"cmd": ["...", "llama-3.3-70b"]}) is True
    assert wd.needs_long_progress_threshold({"cmd": ["...", "128000"]}) is True
    assert wd.needs_long_progress_threshold({"cmd": ["...", "qwen3-8b"]}) is False


def test_progress_stale_age_s_uses_the_most_recent_timestamp_across_phases():
    now = 1_800_000_000.0
    phases = _phases({
        "R1b": "2026-09-30T00:00:00+00:00",
        "R1d": "2026-09-30T00:30:00+00:00",  # more recent -- this one should win
    })
    age = wd.progress_stale_age_s(phases, now=now)
    from datetime import datetime, timezone
    expected = now - datetime(2026, 9, 30, 0, 30, 0, tzinfo=timezone.utc).timestamp()
    assert abs(age - expected) < 1e-6


def test_progress_stale_age_s_none_when_no_phase_has_a_timestamp():
    assert wd.progress_stale_age_s(_phases({}), now=1_800_000_000.0) is None


def test_check_progress_stale_none_when_nothing_running():
    result = wd.check_progress_stale(_phases({"R1b": "2026-09-30T00:00:00+00:00"}),
                                       [{"id": "a", "status": "pending", "cmd": []}])
    assert result is None


def test_check_progress_stale_none_when_recent():
    now = 1_800_000_000.0
    from datetime import datetime, timezone
    recent_ts = datetime.fromtimestamp(now - 10 * 60, tz=timezone.utc).isoformat()
    phases = _phases({"R1b": recent_ts})
    queue_items = [{"id": "r1b_controls", "status": "running", "cmd": ["...", "qwen3-8b"]}]
    result = wd.check_progress_stale(phases, queue_items, now=now)
    assert result is None


def test_check_progress_stale_alerts_past_60min_for_ordinary_phase():
    now = 1_800_000_000.0
    from datetime import datetime, timezone
    stale_ts = datetime.fromtimestamp(now - 90 * 60, tz=timezone.utc).isoformat()
    phases = _phases({"R1b": stale_ts})
    queue_items = [{"id": "r1b_controls", "status": "running", "cmd": ["...", "qwen3-8b"]}]
    result = wd.check_progress_stale(phases, queue_items, now=now)
    assert result is not None
    assert "r1b_controls" in result
    assert "90 min" in result or "90" in result


def test_check_progress_stale_does_not_alert_at_90min_for_70b_phase():
    """90 min is stale for an ordinary phase (60 min threshold) but not for a 70B phase (120 min threshold) -- the
    same age must produce different verdicts depending on which phase is running."""
    now = 1_800_000_000.0
    from datetime import datetime, timezone
    stale_ts = datetime.fromtimestamp(now - 90 * 60, tz=timezone.utc).isoformat()
    phases = _phases({"P70": stale_ts})
    queue_items = [{"id": "x2_r1b_scale", "status": "running", "cmd": ["...", "llama-3.3-70b"]}]
    result = wd.check_progress_stale(phases, queue_items, now=now)
    assert result is None


def test_check_progress_stale_alerts_past_120min_for_70b_phase():
    now = 1_800_000_000.0
    from datetime import datetime, timezone
    stale_ts = datetime.fromtimestamp(now - 150 * 60, tz=timezone.utc).isoformat()
    phases = _phases({"P70": stale_ts})
    queue_items = [{"id": "x2_r1b_scale", "status": "running", "cmd": ["...", "llama-3.3-70b"]}]
    result = wd.check_progress_stale(phases, queue_items, now=now)
    assert result is not None


def test_check_progress_stale_alerts_when_no_phase_has_ever_written_a_row():
    queue_items = [{"id": "fresh_job", "status": "running", "cmd": ["...", "qwen3-8b"]}]
    result = wd.check_progress_stale(_phases({}), queue_items, now=1_800_000_000.0)
    assert result is not None
    assert "no phase has ever written" in result
