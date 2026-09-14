# E-FILTER C2f full Stage-1 collection - detached bare-machine launch.
# Run ONLY after the C2f pilot clears median C_max/C_min >= 3.0, seals (verify_sealed),
# and writes derived/efilter/pilot_context_clearance.json.
# Requires canary baseline applied to measurement.yaml.
# Discard-first warmup + full timed task set. Timing = canary∩paging admissible.
# Freeze policy.n_out_pred_tokens from the pilot's measured median (both keys identical) first.
# Analysis: python -m seam.analysis.efilter <run_id>
# No absolute wall-clock claims (AM-032). Dirty tree allowed with --allow-dirty.

$ErrorActionPreference = "Stop"
Set-Location "C:\Users\zjohn\Projects\gnn-hls-accel"

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = "derived\efilter"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stdout = Join-Path $logDir "c2f_full_$stamp.out.log"
$stderr = Join-Path $logDir "c2f_full_$stamp.err.log"

$py = Join-Path (Get-Location) ".venv-seam\Scripts\python.exe"
if (-not (Test-Path $py)) {
    throw "missing $py - create the SEAM venv before launch"
}

$clearance = Join-Path $logDir "pilot_context_clearance.json"
if (-not (Test-Path $clearance)) {
    throw "missing $clearance - run the C2f pilot to clearance before full collection"
}

$gate = Join-Path $logDir "canary_gate.json"
if (-not (Test-Path $gate)) {
    throw "missing $gate - seal canary baseline before full collection"
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
