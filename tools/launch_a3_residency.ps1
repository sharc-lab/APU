# A3 v2 residency sweep - detached bare-machine launch (Cursor closed).
# Reboot, log in, wait for services to settle, open NOTHING except this PowerShell.
# Do NOT run from inside Cursor.
#
# No Phase 1. No 8192 MB headroom gate. Paging gate is reporting-only for this run.

$ErrorActionPreference = "Stop"
Set-Location "C:\Users\zjohn\Projects\gnn-hls-accel"

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = "derived\prompt_a"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stdout = Join-Path $logDir "a3_residency_$stamp.out.log"
$stderr = Join-Path $logDir "a3_residency_$stamp.err.log"

$py = Join-Path (Get-Location) ".venv-seam\Scripts\python.exe"
if (-not (Test-Path $py)) {
    throw "missing $py - create the SEAM venv before launch"
}

# Detached: parent PowerShell can exit; Cursor must already be closed.
$pyArgs = @(
    "-u",
    "-m", "seam.tools.a3_residency",
    "--allow-dirty",
    "--startup-grace-s", "120"
)

$proc = Start-Process -FilePath $py `
    -ArgumentList $pyArgs `
    -WorkingDirectory (Get-Location).Path `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -WindowStyle Hidden `
    -PassThru

Write-Host "detached_pid=$($proc.Id)"
Write-Host "stdout=$stdout"
Write-Host "stderr=$stderr"
Write-Host "tail: Get-Content -Wait $stdout"
Write-Host "Expect early lines: run_id=... and launch_free_memory_mb=... (no headroom refusal)."
