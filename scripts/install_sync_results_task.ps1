# Registers the "APU-SyncResults" scheduled task ON THE CONTROLLER (not evo-t2s/evo-x2): runs
# scripts/sync_results.py every 2 hours, as the current interactive user (not SYSTEM -- it needs this user's
# own SSH keys to reach the remote machines and this user's own git identity to commit). Idempotent:
# re-running replaces the existing task definition with the same one.
#
# Usage: powershell -File scripts/install_sync_results_task.ps1 -PythonExe <path to python.exe> -RepoRoot <path> [-SyncHost evo-x2|evo-t2s|both]
#   this controller: -PythonExe 'C:\Users\<redacted>\AppData\Local\Programs\Python\Python312\python.exe'
#                    -RepoRoot 'C:\Users\<redacted>\OneDrive\Documents\GitHub\APU'
#
# -SyncHost (default evo-x2; allowed evo-x2, evo-t2s, both) becomes the task's --host argument.
# evo-x2 ONLY until the operator says evo-t2s is back (2026-10-08); then re-run with -SyncHost both.
# The live task's argument was switched to --host evo-x2 by hand on 2026-10-08; re-running this installer with
# the default reproduces that.
#
# Revert: Unregister-ScheduledTask -TaskName "APU-SyncResults" -Confirm:$false

param(
    [Parameter(Mandatory = $true)]
    [string]$PythonExe,
    [Parameter(Mandatory = $true)]
    [string]$RepoRoot,
    [ValidateSet('evo-x2', 'evo-t2s', 'both')]
    [string]$SyncHost = 'evo-x2'
)

$scriptPath = Join-Path $RepoRoot 'scripts\sync_results.py'
$action = New-ScheduledTaskAction -Execute $PythonExe -Argument "`"$scriptPath`" --host $SyncHost" `
    -WorkingDirectory $RepoRoot
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Hours 2) `
    -RepetitionDuration (New-TimeSpan -Days 3650)
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 1)

Register-ScheduledTask -TaskName 'APU-SyncResults' -Action $action -Trigger $trigger -Principal $principal `
    -Settings $settings -Description "Every 2h: pulls results files from $SyncHost (--host $SyncHost), checksummed, incremental, and commits them as 'results sync', skipping the commit while a merge, rebase or git lock is in progress (scripts/sync_results.py)." `
    -Force | Out-Null

Get-ScheduledTask -TaskName 'APU-SyncResults' | Select-Object TaskName, State
