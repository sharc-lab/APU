# evo-t2s change log

evo-t2s is Zachary's machine. This file tracks every durable Windows setting or persistent artifact (scheduled
tasks, registry/power-plan changes, installed software) made on it during this project, mirroring the format of
`docs/X2_CHANGELOG.md`. Every entry states what changed, why, and the exact revert command.

## 2026-09-29 -- APU-QueueWatchdog scheduled task installed

**What:** a Windows scheduled task, `APU-QueueWatchdog`, runs `C:\apu\ovn\queue_watchdog.py` via
`%USERPROFILE%\AppData\Local\Programs\Python\Python312\python.exe` every 10 minutes, indefinitely, as SYSTEM.

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

**Installed via:** `scripts/install_queue_watchdog.ps1 -PythonExe '%USERPROFILE%\AppData\Local\Programs\Python\Python312\python.exe'`
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

**Revert:** uninstall via `%USERPROFILE%\AppData\Local\Programs\Ollama\unins000.exe /SILENT`, or leave installed but
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

## 2026-10-15 -- window opened: preflight facts, Windows Update paused to 2026-11-10

Window 2026-10-15 00:00 ET to 2026-10-18 23:59 ET (handover to Zach). Facts, all in
`results/t2s_window_20261015_preflight.json`: OS build 26200.9711 (25H2), Intel Arc B390 GPU driver 32.0.101.8509,
Intel NPU driver 32.0.100.4512, Tailscale 1.102.3 with ForceDaemon (run unattended) true and the service set to Auto
(read only, nothing changed), nobody logged in, no pending reboot, the 2026-10-02 handover record present.

**What:** Windows Update paused until 2026-11-10T23:59:59Z. Under
`HKLM\SOFTWARE\Microsoft\WindowsUpdate\UX\Settings`: PauseUpdatesStartTime, PauseQualityUpdatesStartTime,
PauseFeatureUpdatesStartTime = 2026-10-15T04:39:56Z and PauseUpdatesExpiryTime, PauseQualityUpdatesEndTime,
PauseFeatureUpdatesEndTime = 2026-11-10T23:59:59Z (all six were absent before). Policy
`HKLM\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate\ExcludeWUDriversInQualityUpdate` was already 1 and was
rewritten as 1. No restart.
**Why:** operator instruction 2026-10-14, so an update reboot cannot kill a running job during this window or the
requested 2026-10-26 to 2026-11-08 window, and no driver changes under the measurements.
**Revert:** `Remove-ItemProperty -Path 'HKLM:\SOFTWARE\Microsoft\WindowsUpdate\UX\Settings' -Name
PauseUpdatesStartTime,PauseUpdatesExpiryTime,PauseQualityUpdatesStartTime,PauseQualityUpdatesEndTime,PauseFeatureUpdatesStartTime,PauseFeatureUpdatesEndTime`
(leave ExcludeWUDriversInQualityUpdate, it predates this window).

## 2026-10-15 -- Ollama 0.34.4 installed side by side (C:\apu\bin\ollama-0.34.4)

**What:** the v0.34.4 Windows zip (`ollama-windows-amd64.zip`, 1461155106 bytes, SHA-256
535193F38F3344E5B08F5D1C171C31CE11AA17F0124FF69AE26D8EC7FE06FA62) downloaded to `C:\apu\dl` and extracted to
`C:\apu\bin\ollama-0.34.4` (ollama.exe SHA-256 D571E4CF0AD4456FCCD33D5B985B6D63BA5669C6E5A812391704083952AAE3D5, the
same exe as the Blade's pinned 0.34.4), and `C:\apu\ovn\ollama_pin.json` pointing the queued jobs at it. The per-user
install (`<profile>\AppData\Local\Programs\Ollama`, 0.33.2) is untouched.
**Why:** every evo-x2 R2 and outcome-table row was measured with Ollama 0.34.4; the T2S week (docs/T2S_WEEK_PLAN.md)
must run the same version.
**Revert:** `Remove-Item -Recurse -Force C:\apu\bin\ollama-0.34.4, C:\apu\ovn\ollama_pin.json, C:\apu\dl`

## 2026-10-15 -- T2S week queue loaded, watchdog re-enabled, handover deadline armed

**What:** deployed the week's files (scripts/deploy_evo.py; the per-file blob and source commit are in
`C:\apu\ovn\expected_blobs.json`), later also the fixed `t2s_week_preflight.py` (the first preflight run parsed the
b10970 version line as build 0; requeued with `t2s_week_queue.py requeue`, passed 2026-10-15T05:03Z), including the new `t2s_deadline.py` and
`t2s_model_pull.py`; loaded `t2s_week_pulls.json` (qwen3:14b pull, missing from the store) and
`t2s_week_queue.json` as one read-modify-write (`t2s_week_queue.py load-window`); that command's own pause replaced the
2026-10-02 handover pause flag and its finally block cleared it. `C:\apu\ovn\queue_deadline.json` armed: no launch
after 2026-10-19T03:00Z, running job killed at the first watchdog tick after 03:20Z (before 03:30Z), then pause
"handover to Zach 2026-10-18; resume only in the next confirmed window", our Ollama/llama-server/python stopped and
`HANDOVER_DONE.json` written (harness/t2s_deadline.py). `Enable-ScheduledTask APU-QueueWatchdog` (the task the
2026-10-02 handover disabled; not re-registered); first launch t2s_wk_model_pull at 2026-10-15T04:48:33Z. Two helper
scripts used for the facts and the update pause are in `C:\apu\ovn` (t2s_facts.ps1, t2s_wu_pause.ps1).
**Why:** operator instructions 2026-10-14, docs/T2S_WEEK_PLAN.md.
**Revert:** at handover the deadline job pauses the queue by itself. To stop sooner: `Disable-ScheduledTask
APU-QueueWatchdog`; `Remove-Item C:\apu\ovn\queue_deadline.json` removes the deadline (only with the watchdog stopped).

## 2026-10-23 -- incident: SSH to evo-t2s outside the access window (stray `--help` directory)

**What:** at about 2026-10-23T20:20Z, from the controller, `py -3.12 scripts/deploy_evo.py --help` was run to read the
script's usage. The script had no help flag and defaulted to `--host evo-t2s`, so it treated `--help` as the remote
directory and ran against evo-t2s: `hostname` (answered EVO-T2S), `New-Item -ItemType Directory -Force --help` (a
directory named `--help`, expected under `%USERPROFILE%` of the SSH account, the remote shell's start directory), a read of
`--help\expected_blobs.json` (absent), and an scp of a fresh `expected_blobs.json` with an empty blob list into that
directory. Nothing else: no job, queue file, scheduled task, Ollama or setting was touched, and no reboot. This was
outside the access window (the next one opens 2026-10-26), so it breaks the no-SSH rule for evo-t2s.
**Why it happened:** deploy_evo.py had a default host and no argument check.
**Fixed:** deploy_evo.py now requires `--host` and refuses any other `-` argument or a call with no repo paths before
any SSH (`parse_args`, test in tests/test_deploy_evo_merge.py).
**Not cleaned up yet:** operator decision 2026-10-24: no SSH to evo-t2s before the window; the removal is step 0
of the 2026-10-26 handover check (`harness/t2s_deadline.py open-window` runs `remove_stray_help` before
`check-handover`; it removes the directory only if it holds nothing but `expected_blobs.json`, otherwise it leaves it
and reports the file list). Record the outcome here when the window opens.
**Revert (in the window):** `Remove-Item -LiteralPath "$env:USERPROFILE\--help" -Recurse -Force` (first check that
it holds only `expected_blobs.json`).
**Cleaned up 2026-10-28 00:37Z:** open-window step 0 found the directory holding only `expected_blobs.json` and removed
it (`{'present': True, 'removed': True, 'files': ['expected_blobs.json']}`).

## 2026-10-28 -- window opened (operator go 2026-10-27), deadline armed, window queue loaded

**Read-only look first** (operator: stop if any of Zach's files or jobs are found): no ollama, llama, python, vllm,
lms or openvino process running; under C:\apu only `RESULTS_DIGEST.md` and `watchdog.log` were newer than
`HANDOVER_DONE.json` (2026-10-19), both written by our own APU-QueueWatchdog (paused no-op ticks and its read-only
digest). Nothing of Zach's found, so the staged order went ahead.
**What, in order (UTC 2026-10-28):**
1. 00:37 open-window: step 0 removed the stray `--help` directory (entry above). The handover check then refused,
   twice, with an empty output: the deployed `t2s_deadline.py` was the 2026-10-14 copy, without `check-handover`, and
   the controller sent the quoted python path to PowerShell without the call operator `&`. Fixed in 981ad4d
   (`remote_command`, test) after deploying the current `t2s_deadline.py`, `t2s_queue.py`, `stale_server_cleanup.py`,
   `proc_util.py`, `host_config.py`. The check then passed (`ok: true`, no process of ours).
2. 00:39 open-window: sync widened to both hosts (controller `.sync_host_window.json`, until 2026-11-09T04:30Z), and
   `queue_deadline.json` armed: no launch after 2026-11-09T04:00Z, running job killed from 04:20Z, hard deadline
   04:30Z, then the handover pause. The open-window process waits on the controller until the hard deadline and
   removes the window file in its finally block; if it dies, sync_results ignores and deletes the expired file.
3. Deploy into C:\apu\ovn of the 35 `t2s_week_preflight.DEPLOYED_FILES` plus `context.py`, `gguf_meta.py`,
   `llama_server.py`, `quality_suite.py`, `agent_trace_steps_uncensored.json`, `x2_num_keep.py`, `t2s_week_queue.py`
   and `t2s_oct26_queue.json`; `analysis\` (impact_model, envelope_model, make_failure_map, trace_weighted_pack);
   the router stack `src\`, `src\cloud\` (client, prices, console_gate), `src\dse\` (router, envelope_data.json,
   fits_locally); `results\cloud_prices_anthropic_20261014.json`. An import check found `own_compaction.py` missing
   (x2_r2_evalset imports it since 2026-10-23); added to DEPLOYED_FILES (eeec0f9) and deployed. Import check then
   passed (router builds, 14 price entries, all job modules import).
4. `python t2s_week_queue.py load-window t2s_oct26_queue.json`: "loaded 12 pending entries under the queue pause
   (cleared)", in the staged order (preflight, window_override_methods, B3 v2 CPU and iGPU, the three B5 cells, the
   B6 pull and validations). The pause it cleared is the release of the 2026-10-18 handover pause.
**Why:** operator go 2026-10-27, docs/T2S_WEEK_PLAN.md staged order.
**Revert:** at the deadline the queue pauses itself (handover). To stop sooner: `Disable-ScheduledTask
APU-QueueWatchdog`, and on the controller stop the open-window process and delete `.sync_host_window.json`.
