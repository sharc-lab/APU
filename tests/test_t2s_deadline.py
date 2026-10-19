"""The evo-t2s handover deadline (harness/t2s_deadline.py, t2s_queue.deadline_reached, queue_watchdog.tick's deadline
branch, x2_r2_agent.run_plan's session-boundary stop), all with fakes: no real process, file outside tmp_path, or
scheduled task."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))

import queue_watchdog as wd  # noqa: E402
import t2s_deadline as tdl  # noqa: E402
import t2s_queue as q  # noqa: E402

NO_LAUNCH = "2026-10-19T03:00:00Z"
HARD = "2026-10-19T03:30:00Z"
T_BEFORE = q._epoch("2026-10-19T02:59:00Z")
T_DRAIN = q._epoch("2026-10-19T03:05:00Z")
T_KILL = q._epoch("2026-10-19T03:21:00Z")   # first tick at or after hard - 600 s
REASON = "handover to Zach 2026-10-18; resume only in the next confirmed window"


@pytest.fixture
def iso(tmp_path, monkeypatch):
    monkeypatch.setattr(q, "QUEUE_FILE", tmp_path / "queue_state.json")
    monkeypatch.setattr(q, "EMPTY_FLAG", tmp_path / "queue_empty.flag")
    monkeypatch.setattr(q, "PAUSE_FLAG", tmp_path / "queue_pause.flag")
    monkeypatch.setattr(q, "DEADLINE_FILE", tmp_path / "queue_deadline.json")
    monkeypatch.setattr(q, "_ps", lambda script: ("rc=0 pid=4242", ""))
    monkeypatch.setattr(tdl, "HANDOVER_DONE_PATH", tmp_path / "HANDOVER_DONE.json")
    return tmp_path


def _arm(iso):
    return tdl.arm(NO_LAUNCH, HARD, REASON)


def _clock(monkeypatch, t):
    monkeypatch.setattr(q.time, "time", lambda: t)


# ------------------------------------------------------------------ arm / deadline_reached
def test_no_file_means_no_deadline(iso):
    assert q.read_deadline() is None and q.deadline_reached(now=T_KILL) is None


def test_arm_writes_the_record_and_the_kill_threshold(iso):
    rec = _arm(iso)
    on_disk = json.loads(q.DEADLINE_FILE.read_text(encoding="utf-8"))
    assert on_disk == rec
    assert rec["no_launch_after_utc"].startswith("2026-10-19T03:00:00")
    assert rec["kill_running_at_or_after_utc"].startswith("2026-10-19T03:20:00")
    assert tdl.kill_threshold(rec) == q._epoch("2026-10-19T03:20:00Z")
    assert "resume" in rec["pause_reason"] and "Zach" in rec["pause_reason"]


def test_arm_refuses_a_reason_without_its_release_and_a_backwards_window(iso):
    with pytest.raises(ValueError):
        tdl.arm(NO_LAUNCH, HARD, "handover")
    with pytest.raises(ValueError):
        tdl.arm(HARD, NO_LAUNCH, REASON)


def test_deadline_reached_only_after_no_launch_time(iso):
    _arm(iso)
    assert q.deadline_reached(now=T_BEFORE) is None
    assert q.deadline_reached(now=T_DRAIN) is not None


def test_unparseable_deadline_file_fails_safe(iso):
    q.DEADLINE_FILE.write_text("{not json", encoding="utf-8")
    assert q.deadline_reached(now=T_BEFORE) is not None


# ------------------------------------------------------------------ launch_next / advance
def test_launch_next_launches_before_the_deadline(iso, monkeypatch):
    _arm(iso)
    _clock(monkeypatch, T_BEFORE)
    q.write_queue([{"id": "a", "cmd": ["x"], "status": "pending"}])
    assert q.launch_next(q.read_queue(), cleanup_fn=lambda: None)["id"] == "a"


def test_launch_next_launches_nothing_after_the_deadline(iso, monkeypatch):
    _arm(iso)
    _clock(monkeypatch, T_DRAIN)
    q.write_queue([{"id": "a", "cmd": ["x"], "status": "pending"}])
    called = []
    assert q.launch_next(q.read_queue(), cleanup_fn=lambda: called.append(1)) is None
    assert q.read_queue()[0]["status"] == "pending" and not called


def test_advance_after_deadline_marks_done_launches_nothing_and_says_why(iso, monkeypatch):
    _arm(iso)
    _clock(monkeypatch, T_DRAIN)
    q.write_queue([{"id": "a", "cmd": ["x"], "status": "running", "pid": 1},
                   {"id": "b", "cmd": ["x"], "status": "pending"}])
    monkeypatch.setattr(q, "_caller_owns_running_job", lambda running: (True, "ok"))
    q.advance("stopped: HandoverDeadline('...')", cleanup_fn=lambda: None)
    st = {it["id"]: it["status"] for it in q.read_queue()}
    assert st == {"a": "error", "b": "pending"}
    assert "handover deadline" in json.loads(q.EMPTY_FLAG.read_text(encoding="utf-8"))["reason"]


# ------------------------------------------------------------------ watchdog tick
def _tick(monkeypatch, t, alive, handover_calls):
    monkeypatch.setattr(q.time, "time", lambda: t)
    return wd.tick(pid_alive_fn=lambda pid: alive, heartbeat_age_fn=lambda e: 10.0, cleanup_fn=lambda o: None,
                   now_fn=lambda: t, ownership_fn=lambda p: False,
                   handover_fn=lambda: handover_calls.append(t) or {"stopped_queue_entries": [],
                                                                    "processes_stopped": [],
                                                                    "pause_reason": REASON})


def test_tick_before_deadline_is_unchanged(iso, monkeypatch):
    _arm(iso)
    q.write_queue([{"id": "a", "cmd": ["x"], "status": "pending"}])
    calls = []
    assert _tick(monkeypatch, T_BEFORE, True, calls)["action"] == "launched" and not calls


def test_tick_drains_a_live_job_until_the_kill_threshold(iso, monkeypatch):
    _arm(iso)
    q.write_queue([{"id": "a", "cmd": ["x"], "status": "running", "pid": 7},
                   {"id": "b", "cmd": ["x"], "status": "pending"}])
    calls = []
    r = _tick(monkeypatch, T_DRAIN, True, calls)
    assert r["action"] == "deadline_draining" and not calls
    assert [it["status"] for it in q.read_queue()] == ["running", "pending"]


def test_tick_hands_over_at_the_kill_threshold(iso, monkeypatch):
    _arm(iso)
    q.write_queue([{"id": "a", "cmd": ["x"], "status": "running", "pid": 7}])
    calls = []
    assert _tick(monkeypatch, T_KILL, True, calls)["action"] == "deadline_handover" and calls == [T_KILL]


def test_tick_hands_over_at_once_when_nothing_is_running(iso, monkeypatch):
    _arm(iso)
    q.write_queue([{"id": "a", "cmd": ["x"], "status": "error"}, {"id": "b", "cmd": ["x"], "status": "pending"}])
    calls = []
    assert _tick(monkeypatch, T_DRAIN, True, calls)["action"] == "deadline_handover" and calls
    assert q.read_queue()[1]["status"] == "pending"   # nothing launched


def test_tick_never_requeues_a_dead_job_after_the_deadline(iso, monkeypatch):
    _arm(iso)
    q.write_queue([{"id": "a", "cmd": ["x"], "status": "running", "pid": 7}])
    calls = []
    assert _tick(monkeypatch, T_DRAIN, False, calls)["action"] == "deadline_handover"


def test_tick_after_handover_is_a_noop_because_paused(iso, monkeypatch):
    _arm(iso)
    q.set_pause(REASON)
    calls = []
    assert _tick(monkeypatch, T_KILL, True, calls)["action"] == "paused" and not calls


# ------------------------------------------------------------------ handover itself (fake processes)
def _procs():
    return [
        {"pid": 100, "ppid": 1, "name": "cmd.exe", "cmdline": "cmd.exe /c cd /d C:\\apu\\ovn && ..."},
        {"pid": 101, "ppid": 100, "name": "python.exe", "cmdline": "python.exe t2s_r2_agent.py --mode real",
         "cwd": "C:\\apu\\ovn", "job_env": "t2s_wk_r2_real_cpu"},
        {"pid": 102, "ppid": 101, "name": "ollama.exe",
         "cmdline": "C:\\apu\\bin\\ollama-0.34.4\\ollama.exe serve"},
        {"pid": 103, "ppid": 102, "name": "llama-server.exe",
         "cmdline": "C:\\apu\\bin\\ollama-0.34.4\\lib\\ollama\\llama-server.exe --model x"},
        {"pid": 200, "ppid": 1, "name": "ollama app.exe", "cmdline": "D:\\Zach\\Ollama\\ollama app.exe"},
        {"pid": 201, "ppid": 1, "name": "python.exe", "cmdline": "python.exe D:\\Zach\\notebook.py",
         "cwd": "D:\\Zach", "job_env": None},
        {"pid": 300, "ppid": 1, "name": "python.exe", "cmdline": "python.exe C:\\apu\\ovn\\queue_watchdog.py",
         "cwd": "C:\\apu\\ovn", "job_env": None},
        {"pid": 301, "ppid": 1, "name": "python.exe", "cmdline": "python.exe some_tool.py",
         "cwd": "C:\\apu\\ovn\\tmp", "job_env": None},
    ]


def test_handover_kills_ours_spares_others_and_self_pauses_and_writes_record(iso):
    _arm(iso)
    q.write_queue([{"id": "t2s_wk_r2_real_cpu", "cmd": ["x"], "status": "running", "pid": 100},
                   {"id": "t2s_wk_px2i", "cmd": ["x"], "status": "pending"}])
    old = tdl.HANDOVER_DONE_PATH
    old.write_text('{"old": true}', encoding="utf-8")
    alive = {p["pid"] for p in _procs()}

    def kill(pid):
        alive.discard(pid)
        return True
    rec = tdl.handover(list_processes=lambda: [p for p in _procs() if p["pid"] in alive], kill_pid=kill,
                       self_chain=lambda: {300}, check_igpu=lambda: {"hkcu": None, "hklm": None},
                       now=T_KILL, log=lambda m: None)
    assert alive == {1, 200, 201, 300} - {1}   # Zach's processes and the watchdog itself survive
    assert q.is_paused()["reason"] == REASON
    st = {it["id"]: it for it in q.read_queue()}
    assert st["t2s_wk_r2_real_cpu"]["status"] == "stopped_for_handover"
    assert "rows kept" in st["t2s_wk_r2_real_cpu"]["note"]
    assert st["t2s_wk_px2i"]["status"] == "pending"
    assert sorted(rec["stopped_queue_entries"][0]["tree"]) == [100, 101, 102, 103]
    assert {p["pid"] for p in rec["processes_stopped"]} == {301}
    assert rec["our_processes_remaining"] == []
    assert json.loads(old.read_text(encoding="utf-8"))["record"] == "t2s_deadline_handover"
    prev = list(iso.glob("HANDOVER_DONE_prev_*.json"))
    assert len(prev) == 1 and json.loads(prev[0].read_text(encoding="utf-8")) == {"old": True}
    assert "next confirmed window" in rec["release"]


def test_handover_with_nothing_running_still_pauses_and_records(iso):
    _arm(iso)
    q.write_queue([{"id": "a", "cmd": ["x"], "status": "done"}])
    rec = tdl.handover(list_processes=lambda: [], kill_pid=lambda pid: True, self_chain=lambda: set(),
                       check_igpu=lambda: {"hkcu": None, "hklm": None}, now=T_DRAIN, log=lambda m: None)
    assert rec["stopped_queue_entries"] == [] and q.is_paused() is not None
    assert tdl.HANDOVER_DONE_PATH.exists()


def test_full_sequence_with_fakes_drain_then_kill_then_noop(iso, monkeypatch):
    """End to end through the real tick and the real handover (fake processes): the job keeps running through the
    drain window, is killed at the threshold, and the next tick does nothing."""
    _arm(iso)
    q.write_queue([{"id": "t2s_wk_r2_real_cpu", "cmd": ["x"], "status": "running", "pid": 100},
                   {"id": "t2s_wk_px2i", "cmd": ["x"], "status": "pending"}])
    alive = {p["pid"] for p in _procs()}
    real_handover = tdl.handover

    def ho():
        return real_handover(list_processes=lambda: [p for p in _procs() if p["pid"] in alive],
                             kill_pid=lambda pid: alive.discard(pid) or True, self_chain=lambda: {300},
                             check_igpu=lambda: {}, log=lambda m: None)

    def tick(t):
        monkeypatch.setattr(q.time, "time", lambda: t)
        return wd.tick(pid_alive_fn=lambda pid: pid in alive, heartbeat_age_fn=lambda e: 10.0,
                       cleanup_fn=lambda o: None, now_fn=lambda: t, ownership_fn=lambda p: False, handover_fn=ho)
    assert tick(T_DRAIN)["action"] == "deadline_draining"
    assert tick(q._epoch("2026-10-19T03:15:00Z"))["action"] == "deadline_draining"
    assert tick(T_KILL)["action"] == "deadline_handover"
    assert 101 not in alive and 200 in alive
    assert tick(T_KILL + 600)["action"] == "paused"
    assert [it["status"] for it in q.read_queue()] == ["stopped_for_handover", "pending"]


# ------------------------------------------------------------------ R2 session boundary
def test_r2_run_plan_stops_at_the_next_session_boundary(iso, monkeypatch, tmp_path):
    import x2_r2_agent as x2
    _arm(iso)
    t = {"now": T_BEFORE}
    monkeypatch.setattr(q.time, "time", lambda: t["now"])
    ran = []

    def fake_session(runtime, model, arm, seed, turns, emit, log, mode):
        ran.append((model, seed))
        t["now"] = T_DRAIN   # the deadline passes during the first session
        return {"first_failure_turn": None, "first_failure_reasons": [], "truncation_detected_turn": None,
                "canary_sys_misses": 0, "canary_hist_misses": 0}
    monkeypatch.setattr(x2, "run_session", fake_session)
    rows = []
    plan = [("ollama_default", (1, 2, 3), 40)]
    with pytest.raises(x2.HandoverDeadline):
        x2.run_plan(None, plan, ["m"], "real", tmp_path / "o.jsonl", log=lambda m: None, emit=rows.append)
    assert ran == [("m", 1)]
    assert rows[-1]["record"] == "deadline_stop" and rows[-1]["next_session"] == ["m", "ollama_default", 2]


def test_r2_run_plan_unaffected_without_a_deadline(iso, monkeypatch, tmp_path):
    import x2_r2_agent as x2
    ran = []
    monkeypatch.setattr(x2, "run_session", lambda *a: ran.append(a[3]) or {
        "first_failure_turn": None, "first_failure_reasons": [], "truncation_detected_turn": None,
        "canary_sys_misses": 0, "canary_hist_misses": 0})
    x2.run_plan(None, [("ollama_default", (1, 2), 40)], ["m"], "real", tmp_path / "o.jsonl", log=lambda m: None,
                emit=lambda r: None)
    assert ran == [1, 2]
