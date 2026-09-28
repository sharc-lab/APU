# Registers the "APU-BootResilience" scheduled task on evo-x2: runs scripts/x2_boot_task.ps1 (deployed to
# C:\apu\ovn\x2_boot_task.ps1) as SYSTEM at every boot. Run this once, interactively or over SSH with an
# administrator session (registering a task that runs as SYSTEM needs admin rights). Idempotent: re-running replaces
# the existing task definition with the same one.
#
# Revert: Unregister-ScheduledTask -TaskName "APU-BootResilience" -Confirm:$false

$action = New-ScheduledTaskAction -Execute 'powershell.exe' `
    -Argument '-NoProfile -ExecutionPolicy Bypass -File C:\apu\ovn\x2_boot_task.ps1'
$trigger = New-ScheduledTaskTrigger -AtStartup
$principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

Register-ScheduledTask -TaskName 'APU-BootResilience' -Action $action -Trigger $trigger -Principal $principal `
    -Settings $settings -Description 'Ensures sshd and Tailscale are running after every boot (harness/scripts/x2_boot_task.ps1).' `
    -Force | Out-Null

Get-ScheduledTask -TaskName 'APU-BootResilience' | Select TaskName, State
