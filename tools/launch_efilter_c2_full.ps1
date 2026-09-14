# E-FILTER C2e full Stage-1 collection - detached bare-machine launch.
# Prefer tools/launch_efilter_c2e_full.ps1 (same command).
# Run ONLY after the C2e pilot clears median C_max/C_min >= 3.0, seals, and writes
# derived/efilter/pilot_context_clearance.json. Paging is per-endpoint in analysis (C2e).
# Freeze policy.n_out_pred_tokens from the pilot's measured median (both keys identical) first.
# Analysis: python -m seam.analysis.efilter <run_id>  (OP + C_max/C_min ceiling; P1-P6).
# No absolute wall-clock claims (AM-032). Dirty tree allowed with --allow-dirty.

$ErrorActionPreference = "Stop"
Set-Location "C:\Users\zjohn\Projects\gnn-hls-accel"

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = "derived\efilter"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stdout = Join-Path $logDir "c2e_full_$stamp.out.log"
$stderr = Join-Path $logDir "c2e_full_$stamp.err.log"

$py = Join-Path (Get-Location) ".venv-seam\Scripts\python.exe"
if (-not (Test-Path $py)) {
    throw "missing $py - create the SEAM venv before launch"
}

$clearance = Join-Path $logDir "pilot_context_clearance.json"
if (-not (Test-Path $clearance)) {
    throw "missing $clearance - run the C2e pilot to clearance before full collection"
}

$env:PYTHONIOENCODING = "utf-8"

$pyArgs = @(
    "-u",
    "-m", "seam.tools.efilter_run",
    "full",
    "--allow-dirty"
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
Write-Host "After seal: .\.venv-seam\Scripts\python.exe -m seam.analysis.efilter <run_id>"
