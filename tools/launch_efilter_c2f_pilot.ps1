# E-FILTER C2f pilot - detached bare-machine smoke (Cursor closed preferred).
# Prerequisites: canary baseline sealed and measurement.yaml threshold updated
#   (tools/launch_canary_baseline.ps1 -UpdateMeasurementYaml).
# Pass: median per-task C_max/C_min >= 3.0 AND run seals (verify_sealed).
# Discard-first task warmup (execution_position=0) + at least 5 timed tasks.
# Paging and canary invalidations are per-endpoint (NOT clearance-fatal for envelope primary;
# wall-clock timeout audit NONE). Memory-floor invalidations still refuse clearance.
# Clearance written only on full pass+seal. Do NOT start the full run from this script.
# No absolute wall-clock claims (AM-032). Dirty tree allowed with --allow-dirty.

$ErrorActionPreference = "Stop"
Set-Location "C:\Users\zjohn\Projects\gnn-hls-accel"

$audit = "derived\efilter\c2f_wallclock_timeout_audit.json"
if (-not (Test-Path $audit)) {
    throw "missing $audit - C2f wall-clock timeout audit required before pilot"
}

$gate = "derived\efilter\canary_gate.json"
if (-not (Test-Path $gate)) {
    Write-Host "WARNING: missing $gate - run tools/launch_canary_baseline.ps1 first (READY_FOR_BASELINE)."
    Write-Host "Pilot may still run against the asserted placeholder threshold in measurement.yaml."
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = "derived\efilter"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stdout = Join-Path $logDir "c2f_pilot_$stamp.out.log"
$stderr = Join-Path $logDir "c2f_pilot_$stamp.err.log"

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
Write-Host "status=READY_FOR_PILOT_LAUNCHED"
Write-Host "PYTHONIOENCODING=$env:PYTHONIOENCODING"
Write-Host "tail: Get-Content -Wait $stdout"
Write-Host "Expect: discard warmup pos=0; >=5 timed; seal; median ratio>=3.0; canary/paging recorded not clearance-fatal; clearance written."
