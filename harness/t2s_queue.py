"""A run queue for evo-t2s so the machine is never idle between scripts.

State lives in C:\\apu\\ovn\\queue_state.json: a list of {id, cmd, status, note}. status is one of
pending / running / done / error / stopped. Each orchestrator script (t2s_amech.py, t2s_night2.py) calls
`advance(run_end_note)` once, in its own `finally` block, after writing its own run_end and .DONE marker. advance()
marks the currently running entry finished (done, or error/stopped if the note says so) and launches the next pending
entry via WMI (Invoke-CimMethod Win32_Process Create), so the chain continues even if the session that queued it has
ended. A note containing "STOP" or "another interactive session" halts the whole queue instead of advancing, because
that is the safety condition, not a routine failure.

Usage from a script's finally block:
    import t2s_queue
    t2s_queue.advance(note)   # note is the same string written to the run_end record

To seed or inspect the queue from the controller (not on evo-t2s): read/write the JSON directly, or call
`t2s_queue.write_queue(items)` / `t2s_queue.read_queue()` locally against a copy, then scp it up.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

QUEUE_FILE = Path(r"C:\apu\ovn\queue_state.json")
EMPTY_FLAG = Path(r"C:\apu\ovn\queue_empty.flag")
PAUSE_FLAG = Path(r"C:\apu\ovn\queue_pause.flag")


def set_pause(reason: str):
    """Maintenance lock (2026-10-01, added after the third self-inflicted collision: an operator's ad-hoc
    debug run calling start/stop_ollama_server() killed x2_k2's own Ollama server mid-run because nothing
    told the watchdog a human was working on the machine). While this flag file exists, queue_watchdog.py's
    tick() does nothing at all -- no crash detection, no launching the next pending entry -- so an operator
    can safely stop the running job (at a cell boundary, with --resume preserved), do ad-hoc debug work, and
    clear the flag when done, without the watchdog racing them by launching something mid-debug. Required
    procedure, enforced by convention not code (the watchdog cannot know what debug work is about to happen):
    1. set_pause(reason) before touching any runtime process by hand.
    2. stop the currently running job's process tree yourself (never a job you did not start).
    3. do the debug work.
    4. clear_pause() when done, so the watchdog resumes normal operation.
    Writes PAUSE_FLAG as JSON ({ts_utc_epoch, reason}) so a later reader can see who paused it and why."""
    PAUSE_FLAG.write_text(json.dumps({"ts_utc_epoch": time.time(), "reason": reason}, indent=1), encoding="utf-8")


def clear_pause():
    """Removes the maintenance lock. A no-op (does not raise) if the flag was not set, so a caller can call this
    unconditionally in a finally block without checking is_paused() first."""
    try:
        PAUSE_FLAG.unlink()
    except FileNotFoundError:
        pass


def is_paused() -> dict | None:
    """Returns the pause record ({ts_utc_epoch, reason}) if the maintenance lock is set, else None. Tolerates a
    corrupt/unreadable flag file the same permissive way read_queue() tolerates a BOM -- a flag that exists but
    cannot be parsed must still count as "paused" (fail safe: block launches) rather than silently falling
    through to "not paused" and launching something during what was meant to be a locked maintenance window."""
    if not PAUSE_FLAG.exists():
        return None
    try:
        return json.loads(PAUSE_FLAG.read_text(encoding="utf-8-sig"))
    except Exception:
        return {"ts_utc_epoch": None, "reason": "unparseable pause flag content; treating as paused"}


def read_queue():
    if not QUEUE_FILE.exists():
        return []
    # utf-8-sig (not utf-8) tolerates a leading BOM. A manual PowerShell edit of this file
    # (ConvertTo-Json | Set-Content -Encoding utf8) writes UTF-8 WITH a BOM despite the encoding name -- that BOM
    # made a plain utf-8 read raise JSONDecodeError here on 2026-09-29, which crashed advance() before it ever
    # reached the code that writes queue_empty.flag, so a real idle gap (night3 finished, nothing launched next)
    # went unnoticed instead of being caught by the very mechanism built to catch it. Reading permissively is the
    # fix that matters; write_queue() below still writes plain UTF-8 (Python's json.dumps + Path.write_text never
    # add a BOM), so this is a read-side defense against however the file was last written, not a format change.
    return json.loads(QUEUE_FILE.read_text(encoding="utf-8-sig"))


def write_queue(items):
    QUEUE_FILE.write_text(json.dumps(items, indent=1), encoding="utf-8")


def _ps(script):
    """Runs the script and returns (stdout, stderr). A PowerShell parse error (e.g. bad quoting) lands on stderr, so
    a caller that only looks at stdout sees an empty string and nothing else -- that silent failure is exactly what
    the first version of _launch below hit (a backslash-quote is not a PowerShell escape, it just ends the string
    early); both streams are returned now so that mistake shows up instead of vanishing."""
    p = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True, timeout=60)
    return p.stdout, p.stderr


JOB_ID_ENV_VAR = "APU_QUEUE_JOB_ID"


def _launch(cmd, log_path, job_id):
    """PowerShell single-quoted strings are literal (no escape character except a doubled '), so building the whole
    cmd.exe command line in Python first and dropping it into one single-quoted PS string avoids the double-escaping
    that broke the first version of this function. None of cmd, log_path here ever contains a single quote.

    2026-09-30 fix: `set APU_QUEUE_JOB_ID=<job_id>` is injected ahead of the real command, in the same cmd.exe
    invocation, so the environment variable is inherited by the launched Python process (and everything it spawns)
    the normal way -- this is what advance() checks before it is willing to touch queue_state.json at all, see
    that function's own docstring."""
    quoted = " ".join(f'"{c}"' if " " in c else c for c in cmd)
    # 2026-09-30 bug found live: cmd.exe's "set VAR=value && nextcmd" includes the space BEFORE "&&" as part of
    # value -- quoting the whole assignment ("set VAR=value") is cmd.exe's own documented fix for exactly this,
    # and is required here (found the hard way: t2s_k1_tier_v3 finished cleanly, called advance(), and the
    # ownership check failed because its own env var was "t2s_k1_tier_v3 " with a trailing space, so the real,
    # completed job's own advance() was rejected as a no-op and the queue sat on it until the watchdog's 30-minute
    # staleness fallback wrongly declared it "crashed").
    full_cmdline = (f'cmd.exe /c cd /d C:\\apu\\ovn && set "{JOB_ID_ENV_VAR}={job_id}" && {quoted} > "{log_path}" 2>&1')
    ps_cmd = (f"$cmd = '{full_cmdline}'; "
              f"$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{{CommandLine=$cmd; CurrentDirectory='C:\\apu\\ovn'}}; "
              f"'rc=' + $r.ReturnValue + ' pid=' + $r.ProcessId")
    out, err = _ps(ps_cmd)
    return (out + (" | stderr: " + err if err.strip() else "")).strip()


def _parse_pid(launch_result):
    """The cmd.exe wrapper PID from _launch()'s own 'rc=0 pid=1234' output, as an int, or None if the launch failed
    or the output could not be parsed. This is the PID watchdog scripts check liveness against, since it is the one
    process every launched run's whole tree hangs off of (killing it, or it dying, ends the run)."""
    m = re.search(r"pid=(\d+)", launch_result or "")
    return int(m.group(1)) if m else None


def _gate_satisfied(item, items):
    """An entry may carry {"gate": {"requires_done": "<other id>"}} -- e.g. r1b_r1d must not start until
    t2s_q0_token_calibration has passed (STEP 3/10, 2026-09-29). Satisfied when no gate is set, or the named
    dependency's own status is "done" (not pending/running/error/stopped/crashed -- a gate that only checks
    "ran" would let a *failed* calibration wave r1b_r1d through, which is exactly the case this exists to block)."""
    gate = item.get("gate")
    if not gate:
        return True
    req = gate.get("requires_done")
    if not req:
        return True
    dep = next((it for it in items if it["id"] == req), None)
    return dep is not None and dep.get("status") == "done"


def _default_pre_launch_cleanup():
    """Real cleanup-before-launch (X1, 2026-10-02): a stale llama-server/ollama process left over from the
    previous job trips the NEXT job's own stale-process guard -- this was the real root cause of the X2 crash
    cascade (one job's orphan killed the next, repeatedly). launch_next() is only ever called when no entry has
    status "running" in queue_state.json (see queue_watchdog.tick()'s control flow: the pending branch only fires
    when `running is None`, and the crash branch clears the running entry before calling this), so it is always
    safe to sweep stray servers immediately before a launch -- there is no currently-running job's server to
    mistake for a stray one. Lazily imports stale_server_cleanup to match this module's minimal-import-surface
    convention; uses queue_watchdog.host_ownership_predicate() so evo-x2 (dedicated: name match only) and
    evo-t2s (shared: name match AND path/port match) each get the right policy without duplicating it here."""
    import stale_server_cleanup as scc
    import queue_watchdog as qw
    result = scc.cleanup_stale_servers(qw.host_ownership_predicate())
    if result.get("killed"):
        print(f"queue: pre-launch cleanup killed {result['killed']}")
    return result


def launch_next(items, cleanup_fn=_default_pre_launch_cleanup):
    """Finds the next pending entry whose gate (if any) is satisfied, marks it running with its launched pid, writes
    the queue, and clears any stale queue_empty.flag. A gated entry that is not yet satisfied is skipped (left
    pending, logged) in favor of the next eligible pending entry, so one unmet gate does not stall the whole queue.

    Before launching the found entry, runs cleanup_fn() (default: _default_pre_launch_cleanup, real stale-server
    sweep) so the new job never inherits a previous job's orphaned llama-server/ollama process -- X1's fix for
    the crash-cascade root cause. Tests pass a stub (e.g. a no-op lambda) here instead of patching the real
    cleanup through two layers of lazy import.

    If the launch itself fails to produce a usable pid (WMI Create returned a non-zero rc, or its output could not
    be parsed -- see the 2026-09-29 evo-t2s incident, where this left a phantom "running, pid None" entry that a
    later watchdog tick then wrongly declared crashed), the entry is reverted straight back to "pending" instead:
    a job that never actually launched must never be marked crashed (only a job that WAS running can die). The
    failure is recorded on the entry as "last_launch_failure" and the search continues with the next eligible
    pending entry, bounded so a machine with every launch failing (e.g. WMI itself down) cannot loop forever.

    Shared by advance() (called from inside a finished run's own process) and queue_watchdog.py (called from an
    independent, always-on scheduled task) so the launch bookkeeping is identical either way. Returns the launched
    entry, or None if nothing eligible could be launched."""
    failed_this_call = set()
    max_attempts = sum(1 for it in items if it["status"] == "pending") or 1
    for _attempt in range(max_attempts):
        skipped = []
        nxt = None
        for it in items:
            if it["status"] != "pending" or it["id"] in failed_this_call:
                continue
            if _gate_satisfied(it, items):
                nxt = it
                break
            skipped.append(it["id"])
        for sid in skipped:
            print(f"queue: skipping {sid} (gate not satisfied yet)")
        if nxt is None:
            return None
        cleanup_fn()
        nxt["status"] = "running"
        nxt["started_ts"] = time.time()
        write_queue(items)
        log_path = f"C:\\apu\\ovn\\queue_{nxt['id']}.log"
        out = _launch(nxt["cmd"], log_path, nxt["id"])
        nxt["launch_result"] = out.strip()
        pid = _parse_pid(out)
        if pid is None:
            nxt["status"] = "pending"
            nxt["last_launch_failure"] = {"ts_utc_epoch": time.time(), "launch_result": nxt["launch_result"]}
            for k in ("started_ts",):
                nxt.pop(k, None)
            write_queue(items)
            failed_this_call.add(nxt["id"])
            print(f"queue: launch of {nxt['id']} produced no pid ({nxt['launch_result']!r}); left pending, trying next")
            continue
        nxt["pid"] = pid
        write_queue(items)
        if EMPTY_FLAG.exists():
            EMPTY_FLAG.unlink()  # a run is now launched; clear any stale flag from a prior empty-queue moment
        return nxt
    return None


def write_empty_flag(items, reason):
    """The machine is about to sit idle with no queued work. Written so a controller (this session or a later one)
    polling this file, or a person checking the machine, sees it immediately instead of the gap only being noticed
    hours later the way night2b's queue-empty gap was (idle 09:06 UTC to discovery ~14:30+ UTC, 2026-09-28)."""
    EMPTY_FLAG.write_text(json.dumps({"ts_utc_epoch": time.time(), "reason": reason,
                                      "queue_tail": items[-3:] if items else []}, indent=1, default=str),
                          encoding="utf-8")


def _pid_in_ancestor_chain(target_pid, start_pid=None, max_depth=16):
    """True if target_pid is start_pid itself or one of its ancestors (walking parent -> grandparent -> ... up to
    max_depth hops, to bound the walk if the process tree is ever malformed/cyclic). Uses psutil, lazily imported
    to keep this module's own import surface minimal (matching queue_watchdog.py's lazy-import convention for the
    same reason). start_pid defaults to os.getpid() -- the calling process itself."""
    import os
    import psutil  # noqa: WPS433  (lazy: this module's own import surface stays minimal otherwise)
    start_pid = start_pid if start_pid is not None else os.getpid()
    try:
        proc = psutil.Process(start_pid)
    except psutil.NoSuchProcess:
        return False
    for _ in range(max_depth):
        if proc.pid == target_pid:
            return True
        try:
            proc = proc.parent()
        except psutil.NoSuchProcess:
            return False
        if proc is None:
            return False
    return False


def _caller_owns_running_job(running):
    """2026-09-30 structural fix: verifies the calling process is actually the job queue_state.json currently
    records as "running" before advance() is allowed to touch queue state at all -- see the 2026-09-30 incident
    (docs/X2_CHANGELOG.md) this replaces, where a script run as a bare ad-hoc subprocess (a "quick live smoke
    test") called tq.advance() in its own finally block and mismarked an unrelated, genuinely-running job, then
    cascaded launch_next() through several more unintended entries.

    Two checks, both must pass:
      1. The APU_QUEUE_JOB_ID environment variable (set by _launch() ahead of the real command, inherited by this
         process the normal way) equals `running["id"]`.
      2. `running["pid"]` (the cmd.exe wrapper PID _launch()'s WMI Create call returned) appears in this process's
         own ancestor chain -- the calling Python process is a descendant of the exact process launch_next()
         started for this entry, not just a same-named job id from a stale env var or a copy-pasted invocation.

    Returns (bool, reason_str). A bare "python some_script.py" invocation from an interactive shell has no
    APU_QUEUE_JOB_ID at all, so check 1 fails immediately and cheaply -- no process-walking needed for the common
    "run outside the queue" case."""
    import os
    job_id = os.environ.get(JOB_ID_ENV_VAR)
    if not job_id:
        return False, (f"no {JOB_ID_ENV_VAR} in this process's environment -- not launched via "
                       f"t2s_queue.launch_next(), refusing to touch queue state")
    if job_id != running.get("id"):
        return False, f"{JOB_ID_ENV_VAR}={job_id!r} does not match the running entry's id {running.get('id')!r}"
    running_pid = running.get("pid")
    if running_pid is None:
        return False, f"running entry {running['id']!r} has no recorded pid to verify against"
    if not _pid_in_ancestor_chain(running_pid):
        return False, (f"this process is not a descendant of pid {running_pid} (the process launch_next() "
                       f"started for {running['id']!r}) -- {JOB_ID_ENV_VAR} matched but the pid chain did not, "
                       f"refusing to touch queue state")
    return True, "ok"


def advance(note, cleanup_fn=_default_pre_launch_cleanup):
    items = read_queue()
    if not items:
        write_empty_flag([], "advance() called with an empty queue_state.json")
        return
    running = next((it for it in items if it["status"] == "running"), None)
    if running is not None:
        owns, reason = _caller_owns_running_job(running)
        if not owns:
            print(f"queue: advance() refusing to act -- caller does not own the running entry "
                 f"{running['id']!r} ({reason}); no-op")
            return
    halt = note and ("STOP" in str(note) or "another interactive session" in str(note))
    if running is not None:
        running["status"] = "stopped" if halt else ("error" if note and "stopped:" in str(note) else "done")
        running["note"] = note
        running["finished_ts"] = time.time()
    write_queue(items)
    if halt:
        write_empty_flag(items, f"queue halted: {note}")
        return
    if launch_next(items, cleanup_fn=cleanup_fn) is None:
        still_pending = [it["id"] for it in items if it["status"] == "pending"]
        reason = (f"all {len(still_pending)} remaining pending entries are gate-blocked: {still_pending}"
                  if still_pending else "no pending entry left after the current run finished")
        write_empty_flag(items, reason)


if __name__ == "__main__":
    print(json.dumps(read_queue(), indent=1))
