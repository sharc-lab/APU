# C2f canary noise-floor baseline — detached bare-machine launch.
# Machine lock + quiescence, NO inference workload. ≥31 canary samples (≥30 consecutive pairs)
# with spacing = efilter canary.settle_s (default 10s). Derives threshold = idle_p95 + margin
# (margin default 0.05). Does NOT choose the threshold to pass a pilot.
#
# After seal: derived/efilter/canary_gate.json is written. Pass -UpdateMeasurementYaml to copy
# threshold fields into configs/measurement.yaml before the C2f pilot.
# READY_FOR_BASELINE until this completes on a quiet machine.

param(
    [switch]$UpdateMeasurementYaml
)

$ErrorActionPreference = "Stop"
Set-Location "C:\Users\zjohn\Projects\gnn-hls-accel"

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = "derived\efilter"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stdout = Join-Path $logDir "canary_baseline_$stamp.out.log"
$stderr = Join-Path $logDir "canary_baseline_$stamp.err.log"

$py = Join-Path (Get-Location) ".venv-seam\Scripts\python.exe"
if (-not (Test-Path $py)) {
    throw "missing $py - create the SEAM venv before launch"
}

$env:PYTHONIOENCODING = "utf-8"

$pyArgs = @(
    "-u",
    "-m", "seam.tools.canary_baseline",
    "--allow-dirty",
    "--n-samples", "31",
    "--margin", "0.05"
)
if ($UpdateMeasurementYaml) {
    $pyArgs += "--update-measurement-yaml"
}

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
Write-Host "status=READY_FOR_BASELINE_LAUNCHED"
Write-Host "Expect: sealed run_id; canary_gate threshold=idle_p95+0.05; derived/efilter/canary_gate.json"
Write-Host "Then: update measurement.yaml (or re-run with -UpdateMeasurementYaml) before C2f pilot."
Write-Host "tail: Get-Content -Wait $stdout"
