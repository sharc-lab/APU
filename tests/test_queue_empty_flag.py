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
    # These tests exercise the empty-flag logic, not the 2026-09-30 ownership guard (see
    # test_queue_ownership_guard below for that) -- bypass it so advance() behaves as it did before.
    monkeypatch.setattr(q, "_caller_owns_running_job", lambda running: (True, "test bypass"))
    return queue_file, flag_file


def test_advance_flags_when_no_pending_entry_left(tmp_path, monkeypatch):
    queue_file, flag_file = _isolate(tmp_path, monkeypatch)
    q.write_queue([{"id": "night3", "cmd": ["echo", "hi"], "status": "running"}])
    q.advance("completed", cleanup_fn=lambda: None)
    assert flag_file.exists()
    flag = json.loads(flag_file.read_text(encoding="utf-8"))
    assert "no pending entry" in flag["reason"]
    items = q.read_queue()
    assert items[0]["status"] == "done"


def test_advance_flags_on_empty_queue(tmp_path, monkeypatch):
    queue_file, flag_file = _isolate(tmp_path, monkeypatch)
    q.advance("completed", cleanup_fn=lambda: None)
    assert flag_file.exists()


def test_advance_flags_on_halt(tmp_path, monkeypatch):
    queue_file, flag_file = _isolate(tmp_path, monkeypatch)
    q.write_queue([{"id": "night3", "cmd": ["echo", "hi"], "status": "running"},
                   {"id": "backlog", "cmd": ["echo", "bye"], "status": "pending"}])
    q.advance("stopped: another interactive session is logged in", cleanup_fn=lambda: None)
    assert flag_file.exists()
    items = q.read_queue()
    assert items[1]["status"] == "pending"  # never launched: halted before reaching the pending entry


def test_read_queue_tolerates_a_utf8_bom(tmp_path, monkeypatch):
    """A manual PowerShell edit (ConvertTo-Json | Set-Content -Encoding utf8) writes UTF-8 WITH a BOM despite the
    encoding name; that BOM crashed read_queue() with JSONDecodeError on 2026-09-29, which crashed advance() before
    it reached the queue_empty.flag write -- exactly the notification this mechanism exists to guarantee."""
    queue_file, flag_file = _isolate(tmp_path, monkeypatch)
    payload = json.dumps([{"id": "x", "cmd": ["echo"], "status": "done"}])
    queue_file.write_bytes(b"\xef\xbb\xbf" + payload.encode("utf-8"))
    items = q.read_queue()
    assert items == [{"id": "x", "cmd": ["echo"], "status": "done"}]


def test_advance_clears_stale_flag_when_next_run_launches(tmp_path, monkeypatch):
    queue_file, flag_file = _isolate(tmp_path, monkeypatch)
    flag_file.write_text("{}", encoding="utf-8")  # a stale flag from an earlier empty moment
    q.write_queue([{"id": "night3", "cmd": ["echo", "hi"], "status": "running"},
                   {"id": "backlog", "cmd": ["echo", "bye"], "status": "pending"}])
    monkeypatch.setattr(q, "_launch", lambda cmd, log_path, job_id: "rc=0 pid=1234")
    q.advance("completed", cleanup_fn=lambda: None)
    assert not flag_file.exists()
    items = q.read_queue()
    assert items[1]["status"] == "running"


# ---------------------------------------------------------------------------------------------------- gated entries
def test_gate_satisfied_with_no_gate():
    assert q._gate_satisfied({"id": "x"}, [{"id": "x"}]) is True


def test_gate_satisfied_when_dependency_done():
    items = [{"id": "calib", "status": "done"}, {"id": "r1b_r1d", "gate": {"requires_done": "calib"}}]
    assert q._gate_satisfied(items[1], items) is True


def test_gate_not_satisfied_when_dependency_pending():
    items = [{"id": "calib", "status": "pending"}, {"id": "r1b_r1d", "gate": {"requires_done": "calib"}}]
    assert q._gate_satisfied(items[1], items) is False


def test_gate_not_satisfied_when_dependency_failed():
    """A gate checks the dependency actually succeeded (status 'done'), not merely that it ran -- error/stopped/
    crashed must not wave the gated entry through."""
    for bad_status in ("error", "stopped", "crashed"):
        items = [{"id": "calib", "status": bad_status}, {"id": "r1b_r1d", "gate": {"requires_done": "calib"}}]
        assert q._gate_satisfied(items[1], items) is False


def test_gate_not_satisfied_when_dependency_missing():
    items = [{"id": "r1b_r1d", "gate": {"requires_done": "calib"}}]
    assert q._gate_satisfied(items[0], items) is False


def test_launch_next_skips_gated_entry_for_next_eligible_one(tmp_path, monkeypatch):
    queue_file, flag_file = _isolate(tmp_path, monkeypatch)
    items = [
        {"id": "calib", "cmd": ["echo", "calib"], "status": "pending"},
        {"id": "r1b_r1d", "cmd": ["echo", "r1b"], "status": "pending", "gate": {"requires_done": "calib"}},
        {"id": "backlog", "cmd": ["echo", "backlog"], "status": "pending"},
    ]
    monkeypatch.setattr(q, "_launch", lambda cmd, log_path, job_id: "rc=0 pid=1234")
    # calib itself is pending (not done), so it is the first eligible entry and launches normally.
    launched = q.launch_next(items, cleanup_fn=lambda: None)
    assert launched["id"] == "calib"
    assert launched["status"] == "running"


def test_launch_next_skips_gated_entry_when_its_dependency_already_ran(tmp_path, monkeypatch):
    queue_file, flag_file = _isolate(tmp_path, monkeypatch)
    items = [
        {"id": "calib", "cmd": ["echo", "calib"], "status": "error"},  # ran, but failed -- gate must not be satisfied
        {"id": "r1b_r1d", "cmd": ["echo", "r1b"], "status": "pending", "gate": {"requires_done": "calib"}},
        {"id": "backlog", "cmd": ["echo", "backlog"], "status": "pending"},
    ]
    monkeypatch.setattr(q, "_launch", lambda cmd, log_path, job_id: "rc=0 pid=1234")
    launched = q.launch_next(items, cleanup_fn=lambda: None)
    assert launched["id"] == "backlog"  # r1b_r1d skipped (left pending), backlog launched instead
    assert items[1]["status"] == "pending"


def test_launch_next_returns_none_when_every_pending_entry_is_gate_blocked(tmp_path, monkeypatch):
    queue_file, flag_file = _isolate(tmp_path, monkeypatch)
    items = [
        {"id": "calib", "status": "error"},
        {"id": "r1b_r1d", "status": "pending", "gate": {"requires_done": "calib"}},
    ]
    assert q.launch_next(items, cleanup_fn=lambda: None) is None
    assert items[1]["status"] == "pending"


def test_advance_reports_gate_blocked_reason_not_no_pending_entry(tmp_path, monkeypatch):
    queue_file, flag_file = _isolate(tmp_path, monkeypatch)
    q.write_queue([
        {"id": "night3", "cmd": ["echo", "hi"], "status": "running"},
        {"id": "calib", "status": "error"},
        {"id": "r1b_r1d", "status": "pending", "gate": {"requires_done": "calib"}},
    ])
    q.advance("completed", cleanup_fn=lambda: None)
    flag = json.loads(flag_file.read_text(encoding="utf-8"))
    assert "gate-blocked" in flag["reason"]
    assert "r1b_r1d" in flag["reason"]


# ---------------------------------------------------------------- 2026-09-30 ownership guard
# See the 2026-09-30 incident (docs/X2_CHANGELOG.md): a script run as a bare ad-hoc subprocess (not launched
# through t2s_queue.launch_next()) called advance() in its own finally block, mismarked an unrelated running job,
# and cascaded launch_next() through several more unintended entries. advance() must now refuse to touch queue
# state unless the calling process is actually the one launch_next() started for the currently-running entry.

def test_advance_is_a_noop_when_no_job_id_env_var_set(tmp_path, monkeypatch):
    """The manual-run case: a bare 'python some_script.py' invocation has no APU_QUEUE_JOB_ID in its environment
    at all, so advance() must refuse and leave queue state completely untouched."""
    queue_file = tmp_path / "queue_state.json"
    flag_file = tmp_path / "queue_empty.flag"
    monkeypatch.setattr(q, "QUEUE_FILE", queue_file)
    monkeypatch.setattr(q, "EMPTY_FLAG", flag_file)
    monkeypatch.delenv(q.JOB_ID_ENV_VAR, raising=False)
    q.write_queue([{"id": "real_job", "cmd": ["echo", "hi"], "status": "running", "pid": 99999}])
    q.advance("completed", cleanup_fn=lambda: None)
    items = q.read_queue()
    assert items[0]["status"] == "running"  # untouched
    assert "pid" in items[0] and items[0]["pid"] == 99999  # untouched
    assert not flag_file.exists()  # advance() returned before reaching any of its normal side effects


def test_advance_is_a_noop_when_job_id_does_not_match_running_entry(tmp_path, monkeypatch):
    """The env var is set, but to a different job id than the one actually running -- still a no-op, not a
    silent redirect onto the wrong entry."""
    queue_file = tmp_path / "queue_state.json"
    flag_file = tmp_path / "queue_empty.flag"
    monkeypatch.setattr(q, "QUEUE_FILE", queue_file)
    monkeypatch.setattr(q, "EMPTY_FLAG", flag_file)
    monkeypatch.setenv(q.JOB_ID_ENV_VAR, "some_other_job")
    q.write_queue([{"id": "real_job", "cmd": ["echo", "hi"], "status": "running", "pid": 99999}])
    q.advance("completed", cleanup_fn=lambda: None)
    items = q.read_queue()
    assert items[0]["status"] == "running"


def test_advance_is_a_noop_when_job_id_matches_but_pid_chain_does_not(tmp_path, monkeypatch):
    """The env var matches the running entry's id, but this process is not a descendant of the pid
    launch_next() actually started -- a copy-pasted or forged env var must not be enough on its own."""
    queue_file = tmp_path / "queue_state.json"
    flag_file = tmp_path / "queue_empty.flag"
    monkeypatch.setattr(q, "QUEUE_FILE", queue_file)
    monkeypatch.setattr(q, "EMPTY_FLAG", flag_file)
    monkeypatch.setenv(q.JOB_ID_ENV_VAR, "real_job")
    monkeypatch.setattr(q, "_pid_in_ancestor_chain", lambda target_pid, start_pid=None, max_depth=16: False)
    q.write_queue([{"id": "real_job", "cmd": ["echo", "hi"], "status": "running", "pid": 99999}])
    q.advance("completed", cleanup_fn=lambda: None)
    items = q.read_queue()
    assert items[0]["status"] == "running"


def test_advance_proceeds_normally_when_job_id_and_pid_chain_both_match(tmp_path, monkeypatch):
    """The correct-job case: env var matches the running entry's id, and the pid chain check passes (this
    process really is a descendant of the recorded pid) -- advance() must behave exactly as it always has."""
    queue_file = tmp_path / "queue_state.json"
    flag_file = tmp_path / "queue_empty.flag"
    monkeypatch.setattr(q, "QUEUE_FILE", queue_file)
    monkeypatch.setattr(q, "EMPTY_FLAG", flag_file)
    monkeypatch.setenv(q.JOB_ID_ENV_VAR, "real_job")
    monkeypatch.setattr(q, "_pid_in_ancestor_chain", lambda target_pid, start_pid=None, max_depth=16: True)
    q.write_queue([{"id": "real_job", "cmd": ["echo", "hi"], "status": "running", "pid": 99999}])
    q.advance("completed", cleanup_fn=lambda: None)
    items = q.read_queue()
    assert items[0]["status"] == "done"  # advance() proceeded and marked it finished
    assert flag_file.exists()  # nothing pending -> normal empty-flag behavior, unaffected by the guard


def test_pid_in_ancestor_chain_true_for_self():
    import os
    assert q._pid_in_ancestor_chain(os.getpid()) is True


def test_pid_in_ancestor_chain_false_for_unrelated_pid():
    """A pid that is neither this process nor any of its real ancestors (0 is never a valid Windows pid to be an
    ancestor of a normal user process) must return False, not raise."""
    assert q._pid_in_ancestor_chain(0) is False


def test_caller_owns_running_job_reports_missing_pid_without_raising(monkeypatch):
    monkeypatch.setenv(q.JOB_ID_ENV_VAR, "x")
    result = q._caller_owns_running_job({"id": "x", "status": "running"})  # no "pid" key at all
    assert result[0] is False
    assert "no recorded pid" in result[1]


def test_launch_next_still_works_with_the_new_job_id_argument(tmp_path, monkeypatch):
    """_launch() gained a required job_id parameter for the JOB_ID_ENV_VAR injection -- confirm launch_next()
    itself (not just advance()) passes it correctly end to end."""
    queue_file = tmp_path / "queue_state.json"
    flag_file = tmp_path / "queue_empty.flag"
    monkeypatch.setattr(q, "QUEUE_FILE", queue_file)
    monkeypatch.setattr(q, "EMPTY_FLAG", flag_file)
    captured = {}

    def fake_launch(cmd, log_path, job_id):
        captured["job_id"] = job_id
        return "rc=0 pid=4242"

    monkeypatch.setattr(q, "_launch", fake_launch)
    items = [{"id": "the_real_job_id", "cmd": ["echo", "hi"], "status": "pending"}]
    launched = q.launch_next(items, cleanup_fn=lambda: None)
    assert launched["id"] == "the_real_job_id"
    assert captured["job_id"] == "the_real_job_id"


def test_launch_cmdline_quotes_the_set_assignment_no_trailing_space(monkeypatch):
    """Real 2026-09-30 bug: cmd.exe's "set VAR=value && nextcmd" (unquoted) includes the space BEFORE "&&" as
    part of the value, so the launched process's own APU_QUEUE_JOB_ID env var came out as "the_id " (trailing
    space) and its own, correct advance() call was rejected by the ownership check as a result -- a completed job
    sat stuck until the watchdog's 30-minute staleness fallback wrongly declared it "crashed". The fix is
    cmd.exe's own documented one: quote the whole assignment, `set "VAR=value"`. This test checks the exact
    string _launch() builds, not just that it runs, since the bug was invisible to every test that mocks _ps and
    never actually parses the string through real cmd.exe."""
    captured = {}
    monkeypatch.setattr(q, "_ps", lambda script: (captured.setdefault("ps_script", script), ("rc=0 pid=1", ""))[1])
    q._launch(["echo", "hi"], r"C:\apu\ovn\queue_x.log", "the_id")
    ps_script = captured["ps_script"]
    assert 'set "APU_QUEUE_JOB_ID=the_id"' in ps_script
    assert "set APU_QUEUE_JOB_ID=the_id " not in ps_script  # the exact unquoted-with-trailing-space bug pattern
