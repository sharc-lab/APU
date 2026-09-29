"""Independent watchdog for the evo-t2s/evo-x2 run queue. Runs as a Windows scheduled task (every 10 min, as SYSTEM)
so a crashed or hung orchestrator process cannot itself silently leave the queue stuck -- unlike t2s_queue.advance(),
which only runs from inside a finishing orchestrator's own process and therefore cannot recover if that process
never gets to its finally block at all (killed externally, powered off mid-run, or crashed hard enough to skip
cleanup).

2026-09-29 incident (see docs/T2S_CHANGELOG.md / docs/X2_CHANGELOG.md): the original version of this module trusted
a single PID-liveness check as authoritative. On evo-t2s that produced a false crash: the WMI-launched wrapper PID
died (root cause under separate investigation) while the actual measurement process kept running and legitimately
finished 2+ hours later, but the watchdog had already declared it "crashed" and cascaded through launching (or
failing to launch) every subsequent queued job. On evo-x2, the opposite failure hit the OLD run_end_written()
heuristic (guessing "the newest .jsonl file mtime after started_ts" when a job's cmd has no --resume flag): it
picked a DIFFERENT run's file that happened to have a run_end record, concluded the truly-dead x2_section0_retry
had already finished and advanced itself, and did nothing for 10+ hours while the job sat dead.

Fix: a job is declared dead only when BOTH signals agree it is dead.
  - pid check: is the recorded pid a live process.
  - heartbeat: how long since C:\\apu\\ovn\\queue_<id>.log (the job's own dedicated log file, created once by
    _launch and appended to for as long as the process runs) was last modified. This replaces the old "guess the
    newest jsonl across all stems" heuristic, which cannot tell one run's file from another's when several stems'
    files are touched in the same window. Stale = no update for HEARTBEAT_STALE_S (30 min), or
    HEARTBEAT_STALE_S_70B (60 min) for a 70B-model phase, which runs single calls that can legitimately take
    several minutes each.

Decision per tick, in order:
  1. If an entry is "running": pid alive -> do nothing. pid dead, heartbeat fresh -> log "pid_check_disagrees" and
     do nothing else (trust the heartbeat; a later tick will re-check). pid dead AND heartbeat stale -> mark
     "crashed", launch the next pending entry.
  2. Else if nothing is "running" and something is "pending" (and its gate, if any, is satisfied), launch it.
  3. Else if nothing is "running" and nothing eligible is "pending", write queue_empty.flag with a timestamp.
  4. A "running" entry with no pid ever recorded (its own launch failed) is never marked crashed by this watchdog --
     see t2s_queue.launch_next, which now reverts a failed launch straight back to "pending" instead of leaving a
     phantom "running, pid None" entry for the watchdog to find.

Every decision is logged to C:\\apu\\ovn\\watchdog.log, one line per tick, so a human reviewing the machine after the
fact can see what the watchdog saw and did, not just its final effect on queue_state.json.

Standing rule: every writer of queue_state.json uses t2s_queue.write_queue (this module does too, via
t2s_queue.launch_next / t2s_queue.write_queue) -- never a manual PowerShell edit, which is what produced the UTF-8
BOM that crashed advance() on 2026-09-29 in the first place.

Usage: python queue_watchdog.py   (no arguments; reads/writes C:\\apu\\ovn\\queue_state.json via t2s_queue)
Installed as a scheduled task by scripts/install_queue_watchdog.ps1.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

DEPLOY = Path(__file__).resolve().parent
sys.path.insert(0, str(DEPLOY))
import t2s_queue as q  # noqa: E402

WATCHDOG_LOG = Path(r"C:\apu\ovn\watchdog.log")
QUEUE_LOG_DIR = Path(r"C:\apu\ovn")
HEARTBEAT_STALE_S = 30 * 60
HEARTBEAT_STALE_S_70B = 60 * 60


def _log(msg):
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}"
    try:
        with open(WATCHDOG_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
    print(line)


def pid_alive(pid, ps_fn=None):
    """ps_fn is injectable for tests (real usage calls t2s_lab.ps via a lazy import, so this module has no hard
    dependency on t2s_lab -- the watchdog must keep working even if that module fails to import for some reason)."""
    if pid is None:
        return False
    if ps_fn is None:
        import t2s_lab as L  # noqa: WPS433  (lazy: keep the watchdog's own import surface minimal)
        ps_fn = L.ps
    out = ps_fn(f"(Get-Process -Id {pid} -ErrorAction SilentlyContinue | Measure-Object).Count", 20)
    return out.strip() not in ("", "0")


def is_70b_phase(entry):
    """True if this entry's own cmd mentions a 70B model -- those phases run single calls that can legitimately
    take several minutes each (see P70's ~176s e2e_s per call), so they get the longer stale threshold."""
    cmd = entry.get("cmd") or []
    return any("70b" in str(c).lower() for c in cmd)


def heartbeat_age_s(entry, log_dir=QUEUE_LOG_DIR, now=None):
    """Seconds since C:\\apu\\ovn\\queue_<id>.log (this entry's own dedicated log file) was last modified, or None
    if that file does not exist. This is the one signal that reflects the actual measurement process's real
    progress regardless of whether the entry's cmd used --resume or which results stem it ended up writing to --
    see the module docstring for the 2026-09-29 incident this replaces run_end_written()'s guessing for."""
    log_path = Path(log_dir) / f"queue_{entry['id']}.log"
    try:
        mtime = log_path.stat().st_mtime
    except OSError:
        return None
    now = now if now is not None else time.time()
    return now - mtime


def is_stale(entry, age_s):
    if age_s is None:
        return True
    threshold = HEARTBEAT_STALE_S_70B if is_70b_phase(entry) else HEARTBEAT_STALE_S
    return age_s > threshold


def tick(pid_alive_fn=pid_alive, heartbeat_age_fn=heartbeat_age_s):
    """One watchdog decision cycle. Injectable pid_alive_fn/heartbeat_age_fn for tests; production defaults hit the
    real machine. Returns a dict describing what happened, for logging and for tests to assert on."""
    items = q.read_queue()
    running = next((it for it in items if it["status"] == "running"), None)

    if running is not None:
        if running.get("pid") is None:
            # A launch that never produced a pid should have been reverted to "pending" by launch_next() itself
            # (see t2s_queue.py); if one is still seen here, leave it alone rather than guess -- do not mark a
            # never-launched job crashed.
            return {"action": "none", "reason": f"entry {running['id']} has status running but no pid; leaving alone"}
        if pid_alive_fn(running.get("pid")):
            return {"action": "none", "reason": f"entry {running['id']} running, pid {running.get('pid')} alive"}
        age = heartbeat_age_fn(running)
        if not is_stale(running, age):
            age_str = f"{age:.0f}s" if age is not None else "unknown"
            reason = (f"entry {running['id']}: pid {running.get('pid')} check says dead but heartbeat age "
                      f"{age_str} is still fresh; trusting the heartbeat, leaving running")
            return {"action": "pid_check_disagrees", "reason": reason}
        running["status"] = "crashed"
        running["finished_ts"] = time.time()
        age_str = f"{age:.0f}s" if age is not None else "no heartbeat log found"
        running["note"] = f"watchdog: pid {running.get('pid')} dead and heartbeat stale ({age_str})"
        q.write_queue(items)
        launched = q.launch_next(items)
        if launched is None:
            q.write_empty_flag(items, f"entry {running['id']} crashed and no pending entry to launch")
            return {"action": "crashed_no_next", "reason": running["note"]}
        return {"action": "crashed_launched_next", "reason": running["note"], "launched": launched["id"]}

    pending = next((it for it in items if it["status"] == "pending"), None)
    if pending is not None:
        launched = q.launch_next(items)
        if launched is None:
            # every remaining pending entry has an unsatisfied gate (see t2s_queue._gate_satisfied) -- nothing to
            # launch this tick, but this is not the same as an empty queue, so say so rather than claim "launched".
            still_pending = [it["id"] for it in items if it["status"] == "pending"]
            q.write_empty_flag(items, f"all {len(still_pending)} remaining pending entries are gate-blocked: {still_pending}")
            return {"action": "gate_blocked", "reason": still_pending}
        return {"action": "launched", "launched": launched["id"]}

    q.write_empty_flag(items, "watchdog: nothing running and nothing pending")
    return {"action": "empty_flag"}


def main():
    result = tick()
    _log(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
