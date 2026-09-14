<#
.SYNOPSIS
  Prove that a detached launch survives the SSH session that started it.

.DESCRIPTION
  Two invocations, from the Mac:

    ssh xps "cd C:/Users/zjohn/Projects/gnn-hls-accel; powershell -File tools/verify_detach.ps1 -Start"
    <the SSH session ends here -- this is the event under test>
    ssh xps "cd C:/Users/zjohn/Projects/gnn-hls-accel; powershell -File tools/verify_detach.ps1 -Check"

  -Start spawns a heartbeat writer through tools/spawn_detached.ps1 and returns immediately.
  -Check reads the heartbeat twice, eight seconds apart, and reports whether the tick advanced.

  A tick that advances across a disconnect is proof. A heartbeat frozen at the disconnect
  timestamp is the job-object kill, and means no unattended run can be trusted.

  -Start also returns immediately as its own assertion: if the SSH command hangs instead of
  returning, the spawned process is holding the session's output handle, which is a second way
  the workflow fails even when the process itself survives.
#>
[CmdletBinding(DefaultParameterSetName = "Start")]
param(
    [Parameter(ParameterSetName = "Start")][switch]$Start,
    [Parameter(ParameterSetName = "Check")][switch]$Check,
    [int]$DurationS = 1800
)

$ErrorActionPreference = "Stop"
$root = "C:\Users\zjohn\Projects\gnn-hls-accel"
$outDir = Join-Path $root "derived\detach_check"
$statePath = Join-Path $outDir "current.json"

if ($Check) {
    if (-not (Test-Path $statePath)) {
        Write-Output "NO ACTIVE CHECK -- run with -Start first."
        exit 1
    }
    $state = Get-Content $statePath -Raw | ConvertFrom-Json
    $hb = $state.heartbeat_path

    if (-not (Test-Path $hb)) {
        Write-Output "VERDICT: FAIL -- heartbeat file was never written. See $($state.log_path)"
        exit 1
    }

    $first = Get-Content $hb -Raw | ConvertFrom-Json
    Start-Sleep -Seconds 8
    $second = Get-Content $hb -Raw | ConvertFrom-Json
    $alive = $null -ne (Get-Process -Id $state.pid -ErrorAction SilentlyContinue)
    $advanced = $second.tick -gt $first.tick

    Write-Output "spawned_pid      : $($state.pid)"
    Write-Output "spawn_parent     : $($state.parent_name) (pid $($state.ppid))"
    Write-Output "spawning_shell   : pid $($state.spawner_pid) -- gone with the SSH session"
    Write-Output "started_utc      : $($first.started_utc)"
    Write-Output "process_alive    : $alive"
    Write-Output "tick             : $($first.tick) -> $($second.tick) over 8s"
    Write-Output "last_seen_utc    : $($second.utc)"

    if ($advanced -and $alive) {
        Write-Output "VERDICT: PASS -- the process outlived the session that spawned it."
        exit 0
    }
    if ($second.finished) {
        Write-Output "VERDICT: PASS (already finished) -- ran to completion at $($second.finished_utc)."
        exit 0
    }
    Write-Output "VERDICT: FAIL -- heartbeat is frozen. Detachment does not survive disconnect."
    exit 1
}

New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$tag = Get-Date -Format "yyyyMMdd_HHmmss"
$hb = Join-Path $outDir "heartbeat_$tag.json"
$log = Join-Path $outDir "detach_$tag.log"
$py = Join-Path $root ".venv-seam\Scripts\python.exe"
$script = Join-Path $root "tools\_detach_heartbeat.py"

$cmd = '"' + $py + '" -u "' + $script + '" "' + $hb + '" ' + $DurationS
$json = & (Join-Path $root "tools\spawn_detached.ps1") -CommandLine $cmd -LogPath $log -WorkingDirectory $root
$info = $json | ConvertFrom-Json

$state = [pscustomobject]@{
    pid            = $info.pid
    ppid           = $info.ppid
    parent_name    = $info.parent_name
    spawner_pid    = $info.spawner_pid
    heartbeat_path = $hb
    log_path       = $log
    tag            = $tag
}
$state | ConvertTo-Json | Set-Content -Path $statePath -Encoding UTF8

Write-Output "started detached heartbeat"
Write-Output "  pid            : $($info.pid)"
Write-Output "  parent         : $($info.parent_name) (pid $($info.ppid))"
Write-Output "  spawning shell : pid $($info.spawner_pid)"
Write-Output "  heartbeat      : $hb"
Write-Output "  runs for       : $DurationS s"
Write-Output ""
Write-Output "Close this SSH session, then reconnect and run:"
Write-Output "  powershell -File tools/verify_detach.ps1 -Check"
