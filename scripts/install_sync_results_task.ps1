# Registers the "APU-SyncResults" scheduled task ON THE CONTROLLER (not evo-t2s/evo-x2): runs
# scripts/sync_results.py every 2 hours, as the current interactive user (not SYSTEM -- it needs this user's
# own SSH keys to reach both remote machines and this user's own git identity to commit). Idempotent:
# re-running replaces the existing task definition with the same one.
#
# Usage: powershell -File scripts/install_sync_results_task.ps1 -PythonExe <path to python.exe> -RepoRoot <path>
#   this controller: -PythonExe 'C:\Users\<redacted>\AppData\Local\Programs\Python\Python312\python.exe'
#                    -RepoRoot 'C:\Users\<redacted>\OneDrive\Documents\GitHub\APU'
#
# Revert: Unregister-ScheduledTask -TaskName "APU-SyncResults" -Confirm:$false

param(
    [Parameter(Mandatory = $true)]
    [string]$PythonExe,
    [Parameter(Mandatory = $true)]
    [string]$RepoRoot
)

$scriptPath = Join-Path $RepoRoot 'scripts\sync_results.py'
$action = New-ScheduledTaskAction -Execute $PythonExe -Argument "`"$scriptPath`" --host both" `
    -WorkingDirectory $RepoRoot
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Hours 2) `
    -RepetitionDuration (New-TimeSpan -Days 3650)
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 1)

Register-ScheduledTask -TaskName 'APU-SyncResults' -Action $action -Trigger $trigger -Principal $principal `
    -Settings $settings -Description 'Every 2h: pulls every results file from evo-t2s and evo-x2, checksummed, incremental, and commits whatever changed (scripts/sync_results.py).' `
    -Force | Out-Null

Get-ScheduledTask -TaskName 'APU-SyncResults' | Select-Object TaskName, State
