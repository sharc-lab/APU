# evo-t2s change log

evo-t2s is Zachary's machine. This file tracks every durable Windows setting or persistent artifact (scheduled
tasks, registry/power-plan changes, installed software) made on it during this project, mirroring the format of
`docs/X2_CHANGELOG.md`. Every entry states what changed, why, and the exact revert command.

## 2026-09-29 -- APU-QueueWatchdog scheduled task installed

**What:** a Windows scheduled task, `APU-QueueWatchdog`, runs `C:\apu\ovn\queue_watchdog.py` via
`C:\Users\SHARC\AppData\Local\Programs\Python\Python312\python.exe` every 10 minutes, indefinitely, as SYSTEM.

**Why:** `harness/t2s_queue.py`'s `advance()` only runs from inside a finishing orchestrator's own process, so it
cannot recover if that process never reaches its `finally` block (crashed hard, externally killed, powered off
mid-run). This happened for real on 2026-09-29: night3 crashed on a stale port-8385 listener (see
`docs/RESULT_PROVENANCE.md`/commit history for the PID 7408 incident), and separately `advance()` itself crashed on
a UTF-8-BOM-corrupted `queue_state.json` before it could write `queue_empty.flag` -- the machine sat idle for about
56 minutes with no notification. The watchdog is an independent backstop: every 10 minutes it checks whether the
"running" queue entry's process is actually alive and, if not, whether that run's own `run_end` record was written;
if neither, it marks the entry "crashed" and launches the next pending entry itself (or writes `queue_empty.flag` if
nothing is pending). See `harness/queue_watchdog.py`'s module docstring for the full decision logic and
`tests/test_queue_watchdog.py` for its dry-run coverage.

**Installed via:** `scripts/install_queue_watchdog.ps1 -PythonExe 'C:\Users\SHARC\AppData\Local\Programs\Python\Python312\python.exe'`
(idempotent; safe to re-run to update the task definition).

**Revert:** `Unregister-ScheduledTask -TaskName "APU-QueueWatchdog" -Confirm:$false`

**Confirmed installed:** `Get-ScheduledTask -TaskName 'APU-QueueWatchdog'` returned `State: Ready` immediately after
registration.

**Footprint:** reads/writes only `C:\apu\ovn\queue_state.json`, `C:\apu\ovn\queue_empty.flag`, and appends to
`C:\apu\ovn\watchdog.log`; it can launch a new orchestrator process (the same way `t2s_queue.advance()` already
does) but does not otherwise touch machine settings. No admin/system state outside the Task Scheduler entry itself.
