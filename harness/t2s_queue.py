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
import subprocess
import sys
import time
from pathlib import Path

QUEUE_FILE = Path(r"C:\apu\ovn\queue_state.json")


def read_queue():
    if not QUEUE_FILE.exists():
        return []
    return json.loads(QUEUE_FILE.read_text(encoding="utf-8"))


def write_queue(items):
    QUEUE_FILE.write_text(json.dumps(items, indent=1), encoding="utf-8")


def _ps(script):
    return subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True, timeout=60).stdout


def _launch(cmd, log_path):
    quoted = " ".join(f'"{c}"' if " " in c else c for c in cmd)
    ps_cmd = (f'$cmd = "cmd.exe /c cd /d C:\\apu\\ovn && {quoted} > \\"{log_path}\\" 2>&1"; '
              f'$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{{CommandLine=$cmd; CurrentDirectory="C:\\apu\\ovn"}}; '
              f'"rc=" + $r.ReturnValue + " pid=" + $r.ProcessId')
    return _ps(ps_cmd)


def advance(note):
    items = read_queue()
    if not items:
        return
    halt = note and ("STOP" in str(note) or "another interactive session" in str(note))
    running = next((it for it in items if it["status"] == "running"), None)
    if running is not None:
        running["status"] = "stopped" if halt else ("error" if note and "stopped:" in str(note) else "done")
        running["note"] = note
        running["finished_ts"] = time.time()
    write_queue(items)
    if halt:
        return
    nxt = next((it for it in items if it["status"] == "pending"), None)
    if nxt is None:
        return
    nxt["status"] = "running"
    nxt["started_ts"] = time.time()
    write_queue(items)
    log_path = f"C:\\apu\\ovn\\queue_{nxt['id']}.log"
    out = _launch(nxt["cmd"], log_path)
    nxt["launch_result"] = out.strip()
    write_queue(items)


if __name__ == "__main__":
    print(json.dumps(read_queue(), indent=1))
