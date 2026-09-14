<#
.SYNOPSIS
  DISPATCH P interleaved f16/u8/u4 delta-prefill matrix with drift canary.

.DESCRIPTION
  Thin wrapper around tools/run_delta_prefill_matrix.ps1. Does NOT use
  chain_kv_precision.ps1.

  288 matrix cells + canary every N=12 (gpu_only_f16 nc=4000 d=400 RESIDENT).
  f16 control is the pinned arm gpu_only_f16 (not bare gpu_only).
  See derived/kv_precision/DISPATCH_P_TIME_ESTIMATE.md and OPERATOR_CMD_DISPATCH_P.md.
#>
[CmdletBinding()]
param(
    [switch]$Orchestrate,
    [switch]$Status,
    [switch]$DryRunGate,
    [string]$PythonExe = "C:\Users\zjohn\Projects\gnn-hls-accel\.venv-seam\Scripts\python.exe",
    [string]$Tag = "dispatch_p_interleaved",
    [int]$CellTimeoutS = 1500,
    [int]$Repeats = 3,
    # Derived: floor(657s onset / 51.34s mean cell wall) = 12 — see DISPATCH_P_TIME_ESTIMATE.md
    [int]$CanaryEveryN = 12,
    [int]$CanaryCalibrationCount = 3,
    [double]$CanaryRelDriftFloor = 0.05
)

$ErrorActionPreference = "Stop"
$root = "C:\Users\zjohn\Projects\gnn-hls-accel"
$self = Join-Path $root "tools\run_delta_prefill_matrix.ps1"

$modeCount = 0
if ($Orchestrate) { $modeCount++ }
if ($Status) { $modeCount++ }
if ($DryRunGate) { $modeCount++ }
if ($modeCount -ne 1) {
    Write-Output "REFUSED -- specify exactly one of -Orchestrate, -DryRunGate, or -Status"
    exit 2
}

if ($Status) {
    & $self -Status -PythonExe $PythonExe
    exit $LASTEXITCODE
}

if ($DryRunGate) {
    & $self `
        -DryRunGate `
        -PythonExe $PythonExe `
        -Tag $Tag `
        -Arms "gpu_only_f16,gpu_only_u8,gpu_only_u4" `
        -NCached "2000,4000,8000,12000" `
        -Deltas "50,150,400,1000" `
        -CellTimeoutS $CellTimeoutS `
        -Repeats $Repeats `
        -CanaryEveryN $CanaryEveryN `
        -CanaryArm "gpu_only_f16" `
        -CanaryNCached 4000 `
        -CanaryDelta 400 `
        -CanaryMode RESIDENT `
        -CanaryCalibrationCount $CanaryCalibrationCount `
        -CanaryRelDriftFloor $CanaryRelDriftFloor
    exit $LASTEXITCODE
}

if ($Orchestrate) {
    Write-Output "DISPATCH P interleaved precision matrix"
    Write-Output "  cells           : 3 arms x 4 n_cached x 4 deltas x 2 modes x 3 repeats = 288"
    Write-Output "  arms            : gpu_only_f16, gpu_only_u8, gpu_only_u4"
    Write-Output "  canary_every_n  : $CanaryEveryN (gpu_only_f16 nc=4000 d=400 RESIDENT)"
    Write-Output "  do_not_use      : chain_kv_precision.ps1"
    & $self `
        -Orchestrate `
        -PythonExe $PythonExe `
        -Tag $Tag `
        -Arms "gpu_only_f16,gpu_only_u8,gpu_only_u4" `
        -NCached "2000,4000,8000,12000" `
        -Deltas "50,150,400,1000" `
        -CellTimeoutS $CellTimeoutS `
        -Repeats $Repeats `
        -CanaryEveryN $CanaryEveryN `
        -CanaryArm "gpu_only_f16" `
        -CanaryNCached 4000 `
        -CanaryDelta 400 `
        -CanaryMode RESIDENT `
        -CanaryCalibrationCount $CanaryCalibrationCount `
        -CanaryRelDriftFloor $CanaryRelDriftFloor
    exit $LASTEXITCODE
}

Write-Output "REFUSED -- specify -Orchestrate, -DryRunGate, or -Status"
exit 2
