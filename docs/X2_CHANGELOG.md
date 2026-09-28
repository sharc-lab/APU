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
