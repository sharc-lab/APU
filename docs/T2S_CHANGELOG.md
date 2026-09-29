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

## 2026-09-29 -- Ollama installed (K1 tier detection + memory-in-use only)

**What:** Ollama v0.33.2 installed via `winget install -e --id Ollama.Ollama` (per-user install, no reboot, no
service-restart side effects observed). Server started headless via WMI `Win32_Process Create` (same pattern as
`t2s_queue._launch`) with `OLLAMA_KEEP_ALIVE=0` so a model never lingers in GPU memory, matching the same setting
used on evo-x2. Pulled `qwen3:8b`.

**Why:** K1 (quality vs available memory through runtime policy) needs Ollama's own tier-detection and memory-
pressure behavior on evo-t2s for the tier + memory-in-use phases only -- quality curves stay evo-x2-only (that part
of K1 needs the model-independence check to be meaningful, per the addendum).

**Revert:** uninstall via `C:\Users\SHARC\AppData\Local\Programs\Ollama\unins000.exe /SILENT`, or leave installed but
stop the server: `Stop-Process` the `ollama.exe serve` process.

**Caution noted:** a `ConnectionRefusedError` interrupted the running r1_a70_p70 experiment's r1_check phase at
almost exactly the same time as this install (`winget install` for Ollama, ~90s duration). The error's shape (URLError/
ConnectionRefusedError from a `/tokenize` or `/chat` call, not a stale-server-guard error) is consistent with an
ordinary race at a model-transition boundary in the harness's own code, not with anything the installer specifically
does to running processes -- the install requires no reboot, no service restart, and does not touch port 8385 -- but
the timing coincidence could not be fully ruled out. The run recovered on its own (moved to the next phase); the
abandoned r1_check phase was re-queued as `r1_check_backfill` (`--resume` the same stem) rather than re-running
research judgement calls, since only the harness's own retry classification (not this note) decides what was a
stale-server error.

## 2026-09-29 -- Ollama contamination check: stopped the idle server, no autostart entry found

Checked every Ollama process for GPU memory residency and any autostart entry before trusting in-progress A70/P70
data (full detail in `docs/RESULT_PROVENANCE.md`). Found one process, `ollama.exe` PID 2016 (this session's own
install), zero GPU memory on every sample, no model ever loaded (`ollama ps` empty throughout), no Startup/HKCU/HKLM
Run entry. Stopped it (`Stop-Process -Id 2016 -Force`, confirmed gone) since Ollama should only run inside K1/K2 jobs
going forward, not idle in the background between them.

**Revert:** none needed (no persistent setting was changed; the server was only ever a plain foreground-launched
process). To restart it for a K1/K2 run: the same WMI `Win32_Process Create` launch documented in the entry above.
