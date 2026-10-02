"""Unit tests for scripts/t2s_handover.py, entirely via dependency injection -- no real machine,
no real registry, no real scheduled task, no real processes."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import t2s_handover as h  # noqa: E402


def make_procs():
    # root 100 (our queue job) with children 101, 102; unrelated process 200; ollama 300; llama-server 301
    return [
        {"pid": 100, "ppid": 1, "name": "cmd.exe"},
        {"pid": 101, "ppid": 100, "name": "python.exe"},
        {"pid": 102, "ppid": 101, "name": "python.exe"},
        {"pid": 200, "ppid": 1, "name": "notepad.exe"},
        {"pid": 300, "ppid": 1, "name": "ollama.exe"},
        {"pid": 301, "ppid": 1, "name": "llama-server.exe"},
    ]


def test_find_descendants_walks_the_whole_tree():
    procs = make_procs()
    assert set(h.find_descendants(100, procs)) == {100, 101, 102}


def test_find_descendants_does_not_touch_unrelated_pids():
    procs = make_procs()
    descendants = h.find_descendants(100, procs)
    assert 200 not in descendants


def test_dry_run_kills_nothing_and_touches_no_real_io(tmp_path):
    killed = []
    queue_written = []
    items = [{"id": "r2_outcome_block_t2s", "status": "running", "pid": 100, "note": ""}]

    record = h.run_handover(
        dry_run=True, wait_s=1, poll_s=1,
        list_processes=make_procs,
        kill_pid=lambda pid: killed.append(pid),
        pid_alive=lambda pid: True,
        check_igpu=lambda: {"hkcu": None, "hklm": None},
        clear_igpu=lambda scope: (_ for _ in ()).throw(AssertionError("must not clear in dry run")),
        disable_task=lambda name: (_ for _ in ()).throw(AssertionError("must not disable in dry run")),
        query_task=lambda name: "Ready",
        read_queue=lambda: items,
        write_queue=lambda items: queue_written.append(items),
        set_pause=lambda reason: (_ for _ in ()).throw(AssertionError("must not set pause in dry run")),
        sleep=lambda s: None,
        done_path=tmp_path / "done.json",
    )
    assert killed == []
    assert queue_written == []
    assert record["dry_run"] is True
    assert set(a for a in record["actions"] if "pid 100" in a)  # still reports what it WOULD do


def test_real_run_stops_only_the_recorded_running_jobs_tree_and_ollama(tmp_path):
    done_path = tmp_path / "HANDOVER_DONE.json"
    killed = []
    paused = []
    queue_written = []
    items = [{"id": "r2_outcome_block_t2s", "status": "running", "pid": 100, "note": ""},
            {"id": "unrelated_done_job", "status": "done", "pid": 999, "note": ""}]

    record = h.run_handover(
        dry_run=False, wait_s=1, poll_s=1,
        list_processes=make_procs,
        kill_pid=lambda pid: killed.append(pid) or True,
        pid_alive=lambda pid: True,  # never finishes on its own -> must be stopped
        check_igpu=lambda: {"hkcu": None, "hklm": None},
        clear_igpu=lambda scope: None,
        disable_task=lambda name: (True, "ok"),
        query_task=lambda name: "Ready" if name == h.WATCHDOG_TASK_NAME else None,
        read_queue=lambda: items,
        write_queue=lambda written: queue_written.append(written),
        set_pause=lambda reason: paused.append(reason),
        sleep=lambda s: None,
        done_path=done_path,
    )
    # the running job's whole tree was stopped
    assert set(killed[:3]) == {100, 101, 102}
    # ollama and llama-server were stopped too
    assert 300 in killed and 301 in killed
    # the unrelated "done" job's pid (999, not even in the process list) was never touched
    assert 999 not in killed
    # notepad (200) was never touched
    assert 200 not in killed
    assert paused == ["T2S handover to Zach, 2026-10-02"]
    assert queue_written[-1][0]["status"] == "stopped_for_handover"
    assert queue_written[-1][1]["status"] == "done"  # untouched, already done
    assert record["dry_run"] is False
    assert done_path.exists()


def test_job_that_finishes_on_its_own_during_the_wait_is_not_killed(tmp_path):
    killed = []
    queue_written = []
    items = [{"id": "r2_outcome_block_t2s", "status": "running", "pid": 100, "note": ""}]
    call_count = {"n": 0}

    def pid_alive(pid):
        call_count["n"] += 1
        return call_count["n"] < 2  # alive on first poll, gone on second

    h.run_handover(
        dry_run=False, wait_s=100, poll_s=1,
        list_processes=make_procs,
        kill_pid=lambda pid: killed.append(pid) or True,
        pid_alive=pid_alive,
        check_igpu=lambda: {"hkcu": None, "hklm": None},
        clear_igpu=lambda scope: None,
        disable_task=lambda name: (True, "ok"),
        query_task=lambda name: "Ready",
        read_queue=lambda: items,
        write_queue=lambda written: queue_written.append(written),
        set_pause=lambda reason: None,
        sleep=lambda s: None,
        done_path=tmp_path / "done.json",
    )
    assert 100 not in killed and 101 not in killed and 102 not in killed  # the job's own tree, untouched
    assert queue_written[-1][0]["status"] == "done"


def test_persistent_igpu_enable_found_is_cleared_and_reported(tmp_path):
    cleared = []
    h_record = h.run_handover(
        dry_run=False, wait_s=1, poll_s=1,
        list_processes=lambda: [],
        kill_pid=lambda pid: True,
        pid_alive=lambda pid: False,
        check_igpu=lambda: {"hkcu": "1", "hklm": None},
        clear_igpu=lambda scope: cleared.append(scope),
        disable_task=lambda name: (True, "ok"),
        query_task=lambda name: "Ready",
        read_queue=lambda: [],
        write_queue=lambda written: None,
        set_pause=lambda reason: None,
        sleep=lambda s: None,
        done_path=tmp_path / "done.json",
    )
    assert cleared == ["hkcu"]
    assert h_record["persistent_igpu_enable_found"] == {"hkcu": "1"}
    assert h_record["persistent_igpu_enable_cleared"] == ["hkcu"]


def test_watchdog_is_disabled_not_deleted(tmp_path):
    disable_calls = []
    h.run_handover(
        dry_run=False, wait_s=1, poll_s=1,
        list_processes=lambda: [],
        kill_pid=lambda pid: True,
        pid_alive=lambda pid: False,
        check_igpu=lambda: {"hkcu": None, "hklm": None},
        clear_igpu=lambda scope: None,
        disable_task=lambda name: disable_calls.append(name) or (True, "ok"),
        query_task=lambda name: "Ready",
        read_queue=lambda: [],
        write_queue=lambda written: None,
        set_pause=lambda reason: None,
        sleep=lambda s: None,
        done_path=tmp_path / "done.json",
    )
    assert disable_calls == [h.WATCHDOG_TASK_NAME]  # disable, never a delete call anywhere in this module


def test_never_calls_restart_or_reboot():
    """Static guard: no code path in this module may actually invoke a restart/reboot primitive
    (checks for the real call shapes, not the word "reboot" -- which legitimately appears in this
    module's own docstrings explaining that it never does this)."""
    import inspect
    src = inspect.getsource(h)
    for forbidden in ("Restart-Computer", "subprocess.run([\"shutdown", "os.system(\"shutdown",
                      "win32api.InitiateSystemShutdown"):
        assert forbidden not in src, f"found forbidden reboot-related call: {forbidden}"
