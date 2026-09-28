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
