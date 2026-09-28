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
