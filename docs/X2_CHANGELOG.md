# EVO-X2 Changelog

Every Windows setting, firewall rule, service, driver, installed tool or file changed on evo-x2 for this project is
recorded here: the exact command, the value before, the value after, the reason, whether it is a standing change or
restored in a finally block by the experiment that needed it, and the revert command. Read back after applying to
confirm it took effect, every time.

Machine identity: hostname EVO-X2, local admin account `evo-x2\ritz` (not a Microsoft account), Tailscale name
`evox2-strix-halo`, IP `100.118.33.76`, tag `tag:apu-x2`. SSH is key-only (`PasswordAuthentication no`,
`KbdInteractiveAuthentication no`, `ChallengeResponseAuthentication no`); that configuration predates this changelog
and was set up and verified by the user at the keyboard, not by this harness, so it is not itself a logged change.

---

## 2026-09-28 -- Windows Update paused for 35 days

**Command:**
```powershell
$path = "HKLM:\SOFTWARE\Microsoft\WindowsUpdate\UX\Settings"
Set-ItemProperty -Path $path -Name PauseUpdatesStartTime -Value "2026-09-28T09:08:29Z" -Type String
Set-ItemProperty -Path $path -Name PauseUpdatesExpiryTime -Value "2026-11-02T09:08:29Z" -Type String
Restart-Service -Name wuauserv -Force
```
**Before:** `PauseUpdatesStartTime` absent; `PauseUpdatesExpiryTime` already `2026-11-02T02:08:09Z` (this machine had
updates paused before this session touched it, by whoever prepared it; not caused by this harness).
**After:** `PauseUpdatesStartTime` = `2026-09-28T09:08:29Z`, `PauseUpdatesExpiryTime` = `2026-11-02T09:08:29Z` (35 days
from the moment this ran, the maximum single-shot pause). Read back with `Get-ItemProperty` immediately after: both
values match what was set.
**Reason:** pre-approved by the user, so no experiment on this machine is interrupted by an update-driven reboot.
**Standing or per-experiment:** standing. **Revert:** clear both values (`Remove-ItemProperty -Path $path -Name
PauseUpdatesStartTime, PauseUpdatesExpiryTime`) or set `PauseUpdatesExpiryTime` to a past date, then
`Restart-Service wuauserv`.
**Note:** a reboot was already pending on this machine before this change (`HKLM:\SOFTWARE\Microsoft\Windows\
CurrentVersion\Component Based Servicing\RebootPending` exists), for a reason not caused by this session. No reboot
has been performed. Flagged to the user; not the download window's fault, but a real risk if anything else on the
machine triggers a reboot before `C:\apu\models\sha256.txt` says `ALL DONE`.

---

## 2026-09-28 -- SeLockMemoryPrivilege granted to Ritz

See `docs/RAM_CAP_PROTOCOL.md`, section "GRANTED on evo-x2, 2026-09-28" for the exact `secedit` commands, before/after
values, verification and revert command. Done at the keyboard by the user before this harness connected.

---

## Deploy directory created

**Command:** `New-Item -ItemType Directory -Force C:\apu\ovn`
**Before:** absent. **After:** present (`Test-Path` true).
**Reason:** deploy target for `scripts/deploy_evo.py`, matching evo-t2s's `C:\apu\ovn`.
**Standing.** **Revert:** `Remove-Item -Recurse -Force C:\apu\ovn` (not planned; nothing here needs it removed).

---

## 2026-09-28 -- PawnIO 2.2.0 driver installed

**Command:**
```powershell
curl.exe -L -o C:\apu\bin\installers\PawnIO_setup.exe https://github.com/namazso/PawnIO.Setup/releases/download/2.2.0/PawnIO_setup.exe
# verified sha256 1F519A22E47187F70A1379A48CA604981C4FCF694F4E65B734AAA74A9FBA3032, size 3,410,960 bytes (matches the
# GitHub release asset size exactly), Authenticode signature Status Valid, signer CN=namazso.eu / O=namazso
# (E=admin@namazso.eu, L=Debrecen, C=HU), issued by GLOBALTRUST 2015 CODESIGNING 1
Start-Process -FilePath C:\apu\bin\installers\PawnIO_setup.exe -ArgumentList "-install","-silent" -Wait -PassThru
```
**First attempt failed:** `/VERYSILENT /SUPPRESSMSGBOXES /NORESTART` (NSIS-style flags, wrong for this installer) hung
with no driver service created; the process stayed responsive but produced nothing, so it was not a UAC prompt on the
active console session, just the wrong argument syntax. Killed (PID 19272, started by this session). The correct
flags for this installer are `-install -silent`.
**Before:** no `PawnIO` service. **After:** `Get-Service PawnIO` reports Status Running (4), StartType Manual (3).
Read back and confirmed.
**Reason:** kernel driver LibreHardwareMonitor needs for CPU/GPU sensor access, approved for the X2 telemetry
inventory (no other source found for iGPU clock or GPU power).
**Standing.** **Revert:** run `PawnIO_setup.exe -uninstall -silent` (uninstaller lives at the same path,
`C:\apu\bin\installers\PawnIO_setup.exe`; the installer also self-registers an uninstall entry), then confirm
`Get-Service PawnIO` returns nothing.
**Note:** `RebootPending` was already `True` before this install (see the Windows Update entry above); this install
did not newly set it and no reboot was performed or is needed for the driver to be running now.

---

## 2026-09-28 -- LibreHardwareMonitor v0.9.6 downloaded and extracted

**Command:**
```powershell
curl.exe -L -o C:\apu\bin\installers\LibreHardwareMonitor.zip https://github.com/LibreHardwareMonitor/LibreHardwareMonitor/releases/download/v0.9.6/LibreHardwareMonitor.zip
Expand-Archive -Path C:\apu\bin\installers\LibreHardwareMonitor.zip -DestinationPath C:\apu\bin\lhm-0.9.6 -Force
```
**Note:** the release also publishes a `LibreHardwareMonitor.NET.10.zip` asset (a .NET 10 build); the plain
`LibreHardwareMonitor.zip` (net472, the classic build our `lhm_sensors.ps1` targets from Windows PowerShell 5.1) is
the one used, matching evo-t2s.
**Before:** `C:\apu\bin\lhm-0.9.6` absent. **After:** present, `LibreHardwareMonitorLib.dll` etc. extracted.
**sha256 of the zip:** `086d9f1b5a99e643edc2cfaaac16051685b551e4c5ac0b32a57c58c0e529c001` (matches the value given, verified
before extracting).
**Reason:** headless CPU/GPU sensor reading (clock, power, load, temperature) for the X2 telemetry inventory.
**Standing.** **Revert:** `Remove-Item -Recurse -Force C:\apu\bin\lhm-0.9.6` (no install step, files only; the PawnIO
driver it depends on is removed separately, see the PawnIO entry above).

---

## 2026-09-28 -- reboot resilience: boot task, single-instance downloader, preflight

**Root cause of the earlier SSH outage (found by the user at the console):** a pending Windows Update service-pack
restart fired via `MoUsoCoreWorker.exe` at 09:36 UTC. The earlier "pause updates 35 days" registry change blocks new
updates, not an already-pending mandatory restart -- it did not and could not have prevented this one. After the
reboot, `sshd` came back `Stopped`/`Manual` (its `StartupType` had reverted); the user set it to `Automatic` and
started it, and confirmed password/keyboard-interactive logins still refused. The `sshd-any-profile-22` and a second
`sshd-tailscale-22` firewall rule (both TCP 22, Profile Any, Allow) were already present alongside the original
Private-only OpenSSH rules; no firewall change was needed for connectivity.

**Command (boot task):**
```powershell
# scripts/x2_register_boot_task.ps1, run once over SSH:
Register-ScheduledTask -TaskName 'APU-BootResilience' -Action (New-ScheduledTaskAction -Execute 'powershell.exe' `
    -Argument '-NoProfile -ExecutionPolicy Bypass -File C:\apu\ovn\x2_boot_task.ps1') `
  -Trigger (New-ScheduledTaskTrigger -AtStartup) `
  -Principal (New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest) -Force
```
**Before:** no `APU-BootResilience` task. **After:** present, `State` `Ready` (read back with `Get-ScheduledTask`).
**Reason:** so a future reboot (planned or forced) cannot silently revert `sshd` to Manual/Stopped again; the task
(`scripts/x2_boot_task.ps1`) sets `sshd` back to Automatic/Running and starts `Tailscale` if either is down, every boot.
**Standing.** **Revert:** `Unregister-ScheduledTask -TaskName "APU-BootResilience" -Confirm:$false`.

**Command (download script):** `C:\apu\dl.ps1` replaced with `scripts/x2_download_models.ps1` (committed), which
takes a named mutex (`Global\apu_dl`) so a second launch while one copy is already running exits immediately instead
of racing the first -- two copies racing on 2026-09-28 produced a corrupted `sha256.txt` line and a `Get-FileHash`
failure on a locked file. The already-downloaded 4B, Qwen3-8B and Llama-3.1-8B files and their correct hash lines were
not touched; the file swap was done while a Qwen3-14B download was in progress and did not interrupt it (PowerShell
had already loaded the running script into memory).
**Before/after:** functionally the same download list and resumable `curl -C -` logic; added: the mutex, an
idempotent skip-if-already-correct check per file, and a de-duplication pass over `sha256.txt` (keeps the last line
per filename) before the final `ALL DONE` marker, so a relaunch's repeated lines for an already-finished file cannot
make the file ambiguous.
**Standing.** **Revert:** not applicable (this is the working downloader now); the previous version is in git history
if ever needed.

**Command (preflight):** `harness/x2_preflight.py`, read-only by default (`--fix` restarts sshd/Tailscale if stopped,
relaunches `dl.ps1` if it died before `ALL DONE`, relaunches a queue entry marked "running" whose process is gone).
Run once after the fixes above: `sshd_ok` true, `tailscale_ok` true, `port_22_listening` true,
`firewall_rule_present` true, `windows_update_paused` true (until 2026-11-02T09:08:29Z), `reboot_pending` false,
`dl_ps1_running` true, `downloads_all_done` false (Qwen3-14B in progress at check time).
**No revert needed** (read-only tool; `--fix` actions are the same idempotent restarts already covered above).

---

## 2026-09-28 -- LibreHardwareMonitor positive controls, evo-x2

**First run found a real bug, not just a bad control.** `harness/lhm_control.py`'s (evo-t2s, never yet run) and
`lhm_x2_control.py`'s timestamp parser used `time.mktime(...) - time.timezone`, which assumes the local zone's
STANDARD offset. The X2 reports its zone ID as "Pacific Standard Time" while PDT (UTC-7) is actually in effect in
September, so every windowed lookup landed an hour outside the ~4-minute sample period and returned nothing. Fixed in
both files (`calendar.timegm`, commit c892ae3) and confirmed correct data on the second run.

| sensor | idle | loaded | rise required | pass |
|---|---|---|---|---|
| CPU package temperature (60 s idle vs 60 s all-core spin) | 50.81 C | 59.50 C | >= 5 C | yes (+8.69 C) |
| CPU package power (same spin) | 21.52 W | 84.99 W | any rise | yes |
| iGPU core clock (idle vs during a real 4B call, ~7000-token prompt) | 600 MHz | 2602 MHz | any rise | yes |
| iGPU power (same call) | 0 W | 32 W | any rise | yes |
| iGPU temperature ("GPU VR SoC", the only GPU temperature LHM exposes here -- not a core die sensor; 60 s repeated calls) | 41.0 C | 54.5 C | any rise | yes (not required, passed anyway) |
| MSAcpi_ThermalZoneTemperature (side comparison, not a control) | 47.05 C | 47.05 C | >= 5 C to trust | no (flat, same as evo-t2s) |
| GPU memory cross-check | LHM 32,251 MiB used | Windows counter 0 MiB | agree within 5% | no, see below |

Spin positive control: 212,600,000-226,095,519 integer iterations/s across the two runs (co-runner genuinely loaded).

**GPU memory cross-check failure is a bug in the control script, not the sensors.** It queried
`\GPU Process Memory(pid_<launcher-pid>_*)`, where `<launcher-pid>` was the WMI-created `cmd.exe` wrapper's PID, not
`llama-server.exe`'s own child PID (the same PID-indirection found manually earlier this session). LHM's own
`SmallData | AMD Radeon(TM) 8060S Graphics | GPU Memory Used` sensor (32,251 MiB) is itself a real, working,
system-wide reading; only the per-process cross-check needs the correct child PID, not resolved yet.

**Sensors available and control-verified on evo-x2:** CPU package temperature and power (`Temperature | ... | Core
(Tctl/Tdie)`, `Power | ... | Package`), iGPU core clock and power (`Clock`/`Power | AMD Radeon(TM) 8060S Graphics |
GPU Core`), iGPU VR temperature, system-wide GPU memory (`SmallData | ... | GPU Memory Used/Free/Total`). Per-core CPU
clock/load/power (SMU) also present but not yet used. `MSAcpi_ThermalZoneTemperature` confirmed not trustworthy
(flat under load), same conclusion as evo-t2s.

**Not yet done:** wiring these sensors into the evo-x2 per-call telemetry (harness/t2s_lab.py's Telemetry class is
Level-Zero-Sysman-shaped and Intel-only; evo-x2 needs its own LHM-based sampler feeding the same igpu_mhz/pkg_power_w/
temp_c fields) and fixing the per-process PID in the memory cross-check. Queued, not started.

---

## 2026-09-28 -- X2 system clock was wrong by about 3h53m; NTP sync fixed it

**Found by the user's check, not caught earlier.** The timezone (`Pacific Standard Time`, correctly applying PDT
-07:00 via .NET's DST rules) was never the problem; the underlying UTC clock itself was wrong.
Measured (same-second comparison, controller `date -u` vs X2 `[DateTime]::UtcNow`): X2 read
`2026-09-28T10:16:00.07Z` while the controller read `2026-09-28T06:22:42.82Z` -- **X2 was about 3h 53m 17s ahead of
real UTC.** `w32time` was `Stopped`/`Manual` (never running), so nothing had ever corrected it.

**Command:**
```powershell
Set-Service w32time -StartupType Automatic
Start-Service w32time
w32tm /config /manualpeerlist:"time.windows.com,0x8" /syncfromflags:manual /reliable:yes /update
Restart-Service w32time
w32tm /resync /force
```
**Before:** offset about +3h53m17s, `w32time` Stopped/Manual.
**After:** offset under 2 s (X2 `06:23:38.75Z` vs controller `06:23:40.71Z`, the remaining ~2 s is SSH round-trip, not
clock error); `w32tm /query /status` reports `Stratum: 5 (secondary reference - syncd by (S)NTP)`, source
`time.windows.com`, last successful sync logged. Read back and confirmed.
**Reason:** the harness matches call windows on UTC; a multi-hour clock error makes every timestamp in every row
wrong by that amount, and was the reason the LHM positive-control reader appeared broken before the parser fix.
**Standing.** **Revert:** not applicable (a correct clock is not something to revert); if ever needed,
`Set-Service w32time -StartupType Manual; Stop-Service w32time`.

**Correction to the reboot root-cause entry above:** it reported the restart at "09:36 UTC" from `LastBootUpTime`
converted via the (at-the-time wrong) X2 clock. Subtracting the measured offset, the reboot actually happened at
approximately **05:43 UTC**, matching the ~05:35 UTC the user observed directly. The cause (a pending Windows Update
service-pack restart via `MoUsoCoreWorker.exe`) is unchanged; only the clock time was wrong.

**Rows recorded on evo-x2 before this fix carry a timestamp offset of about +3h53m and must not be read as literal
UTC:** the four `lhm_x2_control*.log` JSON results (`lhmctl` through `lhmctl4`), the `x2_preflight.py` run whose
output is quoted above (its own printed JSON has no timestamp field, but the queue and boot-task log lines it read do),
`x2_boot_task.log`, and every earlier X2_CHANGELOG entry's "before/after" timestamps taken directly from the machine.
Durations measured entirely on-machine (e.g. the LHM control script's own idle/load windows, all computed from
`time.time()` calls on the same clock) are internally consistent and unaffected; only comparisons against another
machine's clock or against the true wall-clock time are off by the offset above.

---

## 2026-09-28 -- GPU memory cross-check re-run after the PID fix: both sources now read real numbers, but disagree

**srv_pid 21680 (the real llama-server.exe PID this time).** LHM `GPU Memory Used`: 32,258.7 MiB. Windows
`\GPU Process Memory(pid_21680_*)`: Dedicated 39,788.8 MiB + Shared 574.4 MiB = 40,363.3 MiB total. Difference about
25%, outside the 5% agreement bar. The PID bug is fixed (both readings are now real and process-specific, not the
earlier 0/0); the remaining gap looks like a metrology difference rather than a bug: the Windows Dedicated Usage
counter for this process is close to the same order of magnitude seen earlier when a fresh server load alone pushed
Dedicated Usage from 939 MB idle to 42.6 GB (see the 2c inventory notes above), suggesting that counter reflects a
reserved/committed VRAM segment size for the process rather than bytes actually holding model data, while LHM's
sensor is plausibly closer to genuine usage. Not resolved further; both sources are usable individually, just not
cross-validated against each other at 5%.

---

## 2026-09-28 -- Qwen3-14B download stall: process-level restart did not fix it, replaced the downloader

The Qwen3-14B download stuck at exactly 52,879,360 bytes for several hours, `curl.exe` (PID 9908) alive the whole
time, 0 bytes/s. Killed only that PID (not the `dl.ps1` parent, PID 3052) as an initial fix attempt: the retry loop
did relaunch a fresh `curl.exe` (PID 25340) within seconds, but the file still did not grow after several more
minutes. Diagnosed further (read-only): the new curl's only TCP connections were two `127.0.0.1` loopback sockets,
not anything to huggingface.co; `netsh winhttp show proxy` confirmed no system proxy; a fresh, isolated
`curl -r 0-1048576` range request to the same URL completed in 0.19s (302 redirect), so basic outbound HTTPS to HF
was not broken in general. Root cause of the stall itself was not identified -- decided to stop debugging and replace
the mechanism instead, per the user's instruction.

**Fix:** added `--speed-limit 100000 --speed-time 120 --retry 10 --retry-delay 10 --retry-all-errors` to every curl
call in `scripts/x2_download_models.ps1` (commit e2eb6ee), so curl itself aborts and retries any transfer that drops
below 100 KB/s for 2 minutes, instead of relying on an external process-level restart (which this incident showed
does not reliably get a stuck transfer moving again).

**Replacement sequence, in order:**
1. Killed `dl.ps1` (PID 3052) and its `curl.exe` child (PID 25340) -- confirmed via `Get-CimInstance Win32_Process`
   afterwards that no `dl.ps1` or `curl.exe` process remained.
2. Deleted the partial `C:\apu\models\Qwen3-14B-Q4_K_M.gguf` (52,879,360 bytes, confirmed size read immediately
   before deletion).
3. Deployed the fixed script: extracted the committed blob (`git show HEAD:scripts/x2_download_models.ps1` at commit
   e2eb6ee), scp'd to `C:\apu\dl_new.ps1`, verified byte-exact via SHA-256
   (`a49e6992270a1f9adedb88b11a1b63e7c5e266932a786382e691e1eccfaac97a` matched on both sides), archived the old script
   to `C:\apu\dl.ps1.old_20260928`, moved the new one into place as `C:\apu\dl.ps1`.
4. Launched once via WMI (`powershell.exe -WindowStyle Minimized -NoProfile -ExecutionPolicy Bypass -File
   C:\apu\dl.ps1`), confirmed exactly one `dl.ps1` process running (PID 2744) via a process listing filtered on
   `CommandLine -like '*dl.ps1*'`.
5. Confirmed real progress: `Qwen3-14B-Q4_K_M.gguf` size at t=0 (a few seconds after launch) 42,319,872 bytes; t=60s
   680,869,888 bytes; t=120s 1,383,284,736 bytes -- roughly 11 MB/s sustained, well above the mutex-protected
   downloader's own new 100 KB/s stall floor. No need for the Hugging Face CLI fallback path.

Files at this point per `sha256.txt` (mid-run, before the final dedupe pass): qwen3-4b, Qwen3-8B and
Meta-Llama-3.1-8B-Instruct already complete and hash-verified; Qwen3-14B in progress under the new downloader;
Qwen3-30B-A3B, Qwen3-32B and Llama-3.3-70B still queued.

## 2026-09-29 -- all 7 downloads finished; sha256.txt corruption found and fixed

`sha256.txt` said `ALL DONE`, but its per-file lines had been replaced by the literal string
`System.Collections.Specialized.OrderedDictionary+OrderedDictionaryKeyValueCollection` -- the dedupe pass's
`Set-Content C:\apu\models\sha256.txt $lastByName.Values` did not enumerate the OrderedDictionary's `.Values`
collection into lines, it wrote the collection object's own `.ToString()` as one line, destroying every hash line
while leaving `ALL DONE` intact. The actual model files were unaffected (this only corrupted the summary file).
Recomputed all 7 hashes directly with `Get-FileHash`; every one matched its known published hash exactly:

| file | sha256 | bytes |
|---|---|---|
| Llama-3.3-70B-Instruct-Q4_K_M.gguf | `32df3bacc...3a664` | 42,520,398,816 |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf | `7b064f584...33557c` | 4,920,739,232 |
| Qwen3-14B-Q4_K_M.gguf | `500a8806e...ffeb6b81f0` | 9,001,752,960 |
| Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf | `6c997b8af...774d0` | 18,556,686,752 |
| Qwen3-32B-Q4_K_M.gguf | `efd971561...d1e689` | 19,762,149,024 |
| qwen3-4b-instruct-85e4a5b7.gguf | `85e4a5b7b...4b18b9` | 2,497,280,480 |
| Qwen3-8B-Q4_K_M.gguf | `d98cdcbd0...745785` | 5,027,783,488 |

`sha256.txt` rewritten by hand with the correct lines plus `ALL DONE`. Fixed `scripts/x2_download_models.ps1`'s
dedupe pass (cast to `[string[]]` first, forcing real enumeration) so this cannot recur on a future re-run.
Generated `C:\apu\ovn\downloads.jsonl` (the format `t2s_overnight.read_downloads`/`load_models` actually reads) from
the corrected `sha256.txt`, one `"event": "done"` line per model.

## 2026-09-29 -- APU-QueueWatchdog scheduled task installed

Same task and purpose as the evo-t2s entry in `docs/T2S_CHANGELOG.md` (see there for the full rationale): runs
`C:\apu\ovn\queue_watchdog.py` via `C:\Users\Ritz\AppData\Local\Programs\Python\Python312\python.exe` every 10
minutes, as SYSTEM. Installed via `scripts/install_queue_watchdog.ps1 -PythonExe
'C:\Users\Ritz\AppData\Local\Programs\Python\Python312\python.exe'`. Confirmed `State: Ready` immediately after
registration. Revert: `Unregister-ScheduledTask -TaskName "APU-QueueWatchdog" -Confirm:$false`.

## 2026-09-29 -- Section 0 / Q0 / R1 launch blocked by an active console session

Queued `x2_section0` (Section 0 smoke, all 7 models) -> `x2_q0_control` (Q0 positive control, qwen3-8b) ->
`x2_r1_check_full_ladder` (R1 quality side, all 7 models) -> `x2_r1_check_repeat_backlog`, and launched the first
entry. It refused to start: `query.exe user` showed user `ritz` on the console session, `STATE: Active`,
`IDLE TIME: none` -- a live, non-idle interactive session, not a stale leftover. This is the same "another
interactive session is logged in" guard every orchestrator script uses; not overridden, since overriding it risks
interrupting real interactive use of the machine. Queue is seeded and will run as soon as that session is not
active (or the operator confirms it is safe to proceed anyway).

## 2026-09-29 -- Ollama installed, OLLAMA_KEEP_ALIVE=0

Installed via `winget install -e --id Ollama.Ollama` (v0.34.4). The desktop tray app ("ollama app.exe") fails to
start over SSH (`Failed to start: Unable to init instance: Unspecified error` -- its UI-server component needs an
interactive desktop session SSH does not provide); the headless `ollama.exe serve` works fine launched via WMI
`Win32_Process Create` (independent of the SSH session, same pattern as `t2s_queue._launch`). ROCm backend detected
(AMD Radeon 8060S Graphics, gfx1151, 87.9 GiB VRAM); ROCm is preferred over Vulkan by Ollama's own GPU discovery
("dropping integrated GPU; to enable, set OLLAMA_IGPU_ENABLE=1" refers to a Vulkan fallback path, not iGPU-vs-dGPU).
`vram-based default context`: default_num_ctx 262144 at this VRAM size.

Pulled `qwen3:8b` (5.2 GB, digest `500a1f067a9f`).

Restarted the server with `OLLAMA_KEEP_ALIVE=0` (per the user's instruction: a model must never linger in GPU memory
while R1 or any other experiment runs) -- confirmed in the server's own startup log
(`OLLAMA_KEEP_ALIVE:0s`) and via `ollama ps` (empty, both before and after the restart). Every evo-x2 result row now
also records `ollama_model_loaded` (queries `/api/ps` live, `harness/host_config.get_ollama_loaded_model`), so a
lingering model would show up in the data even if the env var setting were ever undone.

**Revert:** `Stop-Process` the `ollama.exe serve` process and relaunch without the `OLLAMA_KEEP_ALIVE=0` env var, or
simply do not set it on the next launch (default is 5m).
