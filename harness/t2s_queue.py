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


def _launch(cmd, log_path):
    """PowerShell single-quoted strings are literal (no escape character except a doubled '), so building the whole
    cmd.exe command line in Python first and dropping it into one single-quoted PS string avoids the double-escaping
    that broke the first version of this function. None of cmd, log_path here ever contains a single quote."""
    quoted = " ".join(f'"{c}"' if " " in c else c for c in cmd)
    full_cmdline = f'cmd.exe /c cd /d C:\\apu\\ovn && {quoted} > "{log_path}" 2>&1'
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


def launch_next(items):
    """Finds the next pending entry whose gate (if any) is satisfied, marks it running with its launched pid, writes
    the queue, and clears any stale queue_empty.flag. A gated entry that is not yet satisfied is skipped (left
    pending, logged) in favor of the next eligible pending entry, so one unmet gate does not stall the whole queue.
    Shared by advance() (called from inside a finished run's own process) and queue_watchdog.py (called from an
    independent, always-on scheduled task) so the launch bookkeeping is identical either way. Returns the launched
    entry, or None if there was nothing eligible to launch."""
    skipped = []
    nxt = None
    for it in items:
        if it["status"] != "pending":
            continue
        if _gate_satisfied(it, items):
            nxt = it
            break
        skipped.append(it["id"])
    for sid in skipped:
        print(f"queue: skipping {sid} (gate not satisfied yet)")
    if nxt is None:
        return None
    nxt["status"] = "running"
    nxt["started_ts"] = time.time()
    write_queue(items)
    log_path = f"C:\\apu\\ovn\\queue_{nxt['id']}.log"
    out = _launch(nxt["cmd"], log_path)
    nxt["launch_result"] = out.strip()
    nxt["pid"] = _parse_pid(out)
    write_queue(items)
    if EMPTY_FLAG.exists():
        EMPTY_FLAG.unlink()  # a run is now launched; clear any stale flag from a prior empty-queue moment
    return nxt


def write_empty_flag(items, reason):
    """The machine is about to sit idle with no queued work. Written so a controller (this session or a later one)
    polling this file, or a person checking the machine, sees it immediately instead of the gap only being noticed
    hours later the way night2b's queue-empty gap was (idle 09:06 UTC to discovery ~14:30+ UTC, 2026-09-28)."""
    EMPTY_FLAG.write_text(json.dumps({"ts_utc_epoch": time.time(), "reason": reason,
                                      "queue_tail": items[-3:] if items else []}, indent=1, default=str),
                          encoding="utf-8")


def advance(note):
    items = read_queue()
    if not items:
        write_empty_flag([], "advance() called with an empty queue_state.json")
        return
    halt = note and ("STOP" in str(note) or "another interactive session" in str(note))
    running = next((it for it in items if it["status"] == "running"), None)
    if running is not None:
        running["status"] = "stopped" if halt else ("error" if note and "stopped:" in str(note) else "done")
        running["note"] = note
        running["finished_ts"] = time.time()
    write_queue(items)
    if halt:
        write_empty_flag(items, f"queue halted: {note}")
        return
    if launch_next(items) is None:
        still_pending = [it["id"] for it in items if it["status"] == "pending"]
        reason = (f"all {len(still_pending)} remaining pending entries are gate-blocked: {still_pending}"
                  if still_pending else "no pending entry left after the current run finished")
        write_empty_flag(items, reason)


if __name__ == "__main__":
    print(json.dumps(read_queue(), indent=1))
