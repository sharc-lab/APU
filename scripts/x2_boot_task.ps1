# Run at boot as SYSTEM (registered by scripts/x2_register_boot_task.ps1) so a reboot can never lock us out of
# evo-x2 again: makes sure sshd is set to auto-start and running, and that Tailscale is running. Idempotent -- safe
# to run when everything is already correct. Logs one line per boot to C:\apu\ovn\x2_boot_task.log.

$log = 'C:\apu\ovn\x2_boot_task.log'
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null
$actions = @()

$svc = Get-Service sshd -ErrorAction SilentlyContinue
if ($svc) {
    if ($svc.StartType -ne 'Automatic') {
        Set-Service sshd -StartupType Automatic
        $actions += "set sshd StartupType Automatic (was $($svc.StartType))"
    }
    if ($svc.Status -ne 'Running') {
        Start-Service sshd
        $actions += "started sshd (was $($svc.Status))"
    }
} else {
    $actions += "WARNING: no sshd service found"
}

$ts = Get-Service Tailscale -ErrorAction SilentlyContinue
if ($ts) {
    if ($ts.Status -ne 'Running') {
        Start-Service Tailscale
        $actions += "started Tailscale (was $($ts.Status))"
    }
} else {
    $actions += "WARNING: no Tailscale service found"
}

$line = ([DateTime]::UtcNow.ToString('o')) + ' | ' + ($(if ($actions) { $actions -join '; ' } else { 'nothing to do' }))
Add-Content -Path $log -Value $line
