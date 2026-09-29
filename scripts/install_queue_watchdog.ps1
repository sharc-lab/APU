# Registers the "APU-QueueWatchdog" scheduled task: runs harness/queue_watchdog.py (deployed to
# C:\apu\ovn\queue_watchdog.py) as SYSTEM every 10 minutes, indefinitely. Run this once per machine, interactively or
# over SSH with an administrator session (registering a task that runs as SYSTEM needs admin rights). Idempotent:
# re-running replaces the existing task definition with the same one.
#
# Usage: powershell -File install_queue_watchdog.ps1 -PythonExe <path to python.exe>
#   evo-t2s: -PythonExe 'C:\Users\SHARC\AppData\Local\Programs\Python\Python312\python.exe'
#   evo-x2:  -PythonExe 'C:\Users\Ritz\AppData\Local\Programs\Python\Python312\python.exe'
#
# Revert: Unregister-ScheduledTask -TaskName "APU-QueueWatchdog" -Confirm:$false

param(
    [Parameter(Mandatory = $true)]
    [string]$PythonExe
)

$action = New-ScheduledTaskAction -Execute $PythonExe -Argument 'C:\apu\ovn\queue_watchdog.py' `
    -WorkingDirectory 'C:\apu\ovn'
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 10) `
    -RepetitionDuration (New-TimeSpan -Days 3650)  # ~10 years; [TimeSpan]::MaxValue overflows the task XML duration field
$principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 5)

Register-ScheduledTask -TaskName 'APU-QueueWatchdog' -Action $action -Trigger $trigger -Principal $principal `
    -Settings $settings -Description 'Every 10 min: detects a crashed/stuck queue_state.json entry and launches the next pending run, or writes queue_empty.flag (harness/queue_watchdog.py).' `
    -Force | Out-Null

Get-ScheduledTask -TaskName 'APU-QueueWatchdog' | Select-Object TaskName, State
