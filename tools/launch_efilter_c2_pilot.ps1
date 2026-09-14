# E-FILTER C2e pilot - detached bare-machine smoke (Cursor closed preferred).
# Prefer tools/launch_efilter_c2e_pilot.ps1 (same command).
# Pass: median per-task C_max/C_min >= 3.0 AND the run seals (verify_sealed).
# Paging invalidations recorded per block, NOT run-fatal (C2e).
# Clearance is written only on full pass+seal. Do NOT start the full run from this script.
# No absolute wall-clock claims (AM-032). Dirty tree allowed with --allow-dirty.

$ErrorActionPreference = "Stop"
Set-Location "C:\Users\zjohn\Projects\gnn-hls-accel"

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = "derived\efilter"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stdout = Join-Path $logDir "c2e_pilot_$stamp.out.log"
$stderr = Join-Path $logDir "c2e_pilot_$stamp.err.log"

$py = Join-Path (Get-Location) ".venv-seam\Scripts\python.exe"
if (-not (Test-Path $py)) {
    throw "missing $py - create the SEAM venv before launch"
}

$env:PYTHONIOENCODING = "utf-8"

$pyArgs = @(
    "-u",
    "-m", "seam.tools.efilter_run",
    "pilot",
    "--allow-dirty"
    # n_tasks_pilot=5 is in configs/efilter.yaml; override with --n-tasks if needed
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
Write-Host "PYTHONIOENCODING=$env:PYTHONIOENCODING"
Write-Host "tail: Get-Content -Wait $stdout"
Write-Host "Expect: sealed run_id=...; median C_max/C_min >= 3.0; verify_sealed; paging recorded not fatal; clearance written. Else exit 2."
