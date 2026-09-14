# Regression: a tripped canary must abort the caller, not complete as a clean run.
#
# 89f77871: canaries 4/5 had drift_tripped=true, plan.status=complete, and no
# FAIL_CANARY_DRIFT in the log. Root cause: Invoke-DriftCanary did
#   Write-Output "REFUSED -- FAIL_CANARY_DRIFT: ..."
#   return $false
# Callers used `if (-not (Invoke-DriftCanary))`. The subexpression captured
# Object[]{string, false}; a non-empty array is truthy, so -not never fired and
# the REFUSED line was swallowed into the capture (never reached the log).
#
# A guard that detects without enforcing is worse than no guard.
#
# Run:
#   powershell -NoProfile -File tools/test_canary_drift_abort_enforcement.ps1

$ErrorActionPreference = "Stop"
$failed = 0
$root = "C:\Users\zjohn\Projects\gnn-hls-accel"
. (Join-Path $root "tools\SeamPsCommon.ps1")

function Assert-True {
    param([bool]$Condition, [string]$Message)
    if ($Condition) {
        Write-Output "PASS  $Message"
    } else {
        Write-Output "FAIL  $Message"
        $script:failed++
    }
}

# --- 0) Document the exact swallow hazard ---
function Invoke-PollutedTripReturn {
    Write-Output "REFUSED -- FAIL_CANARY_DRIFT: synthetic"
    return $false
}

$enterAbortPolluted = $false
if (-not (Invoke-PollutedTripReturn)) {
    $enterAbortPolluted = $true
}
Assert-True -Condition (-not $enterAbortPolluted) `
    -Message "polluted Write-Output+return `$false does NOT enter abort (hazard still real)"

$captured = @(Invoke-PollutedTripReturn)
Assert-True -Condition ($captured.Count -eq 2) `
    -Message "polluted return is Object[2] (message + bool)"

# --- 1) Fixed return shape: Host message + NoEnumerate bool ---
function Invoke-FixedTripReturn {
    param([bool]$Tripped)
    if ($Tripped) {
        Write-Host "REFUSED -- FAIL_CANARY_DRIFT: synthetic fixed"
        Write-Output -NoEnumerate $false
        return
    }
    Write-Output -NoEnumerate $true
}

$enterAbortFixed = $false
$okFixed = Invoke-FixedTripReturn -Tripped $true
Assert-True -Condition ($okFixed -is [bool]) -Message "fixed trip return is System.Boolean"
Assert-True -Condition ($okFixed -eq $false) -Message "fixed trip return value is `$false"
if (-not $okFixed) { $enterAbortFixed = $true }
Assert-True -Condition $enterAbortFixed `
    -Message "fixed pattern enters abort when assigned then tested"

$enterAbortSubexpr = $false
if (-not (Invoke-FixedTripReturn -Tripped $true)) {
    $enterAbortSubexpr = $true
}
Assert-True -Condition $enterAbortSubexpr `
    -Message "fixed pattern enters abort under if (-not (Invoke-...)) subexpression"

$okPass = Invoke-FixedTripReturn -Tripped $false
Assert-True -Condition ($okPass -eq $true) -Message "fixed pass return is `$true"

# --- 2) End-to-end: Update-CanaryDriftBookkeeping trip -> caller abort ---
function Invoke-SyntheticDriftCanaryAbortPath {
    <#
      Mirrors post-fix Invoke-DriftCanary enforcement without launching a cell:
      bookkeeping on a synthetic OK canary that exceeds an armed gate, then the
      same bool return contract the matrix callers use.
      (No Write-Output except the final bool — host messages only.)
    #>
    $gate = [ordered]@{
        armed                = $true
        calibration_complete = $true
        ref_turn1_prefill_s  = 2.657964355
        ref_turn2_prefill_s  = 1.056574707
        early_max_rel_t1     = 0.018399
        early_max_rel_t2     = 0.215588
        threshold_t1         = 0.05
        threshold_t2         = 0.431175
        derivation_applied   = "synthetic"
    }
    # Same breach class as 89f77871 canary 4: t1 ~2.517 vs ref 2.658 -> rel~0.053 > 0.05
    $rec = [ordered]@{
        cell_index       = 99
        is_canary        = $true
        canary_index     = 4
        classification   = "OK"
        turn1_prefill_s  = 2.51738208
        turn2_prefill_s  = 0.802464172
        pipeline_type    = "llm"
    }
    $bkRaw = Update-CanaryDriftBookkeeping `
        -Rec $rec `
        -CanaryGate $gate `
        -PriorCanaryRecords @() `
        -CanaryCalibrationCount 3 `
        -CanaryRelDriftFloor 0.05
    $bk = Get-SinglePipelineRecord -InputObject $bkRaw
    $tripped = [bool]$bk["tripped"]
    $recOut = Get-SinglePipelineRecord -InputObject $bk["rec"]
    if ([bool]$recOut["drift_tripped"]) { $tripped = $true }
    $script:synth_tripped = $tripped
    $script:synth_drift_tripped = [bool]$recOut["drift_tripped"]
    $script:synth_trip_detail = $bk["trip_detail"]

    if ($tripped) {
        Write-Host ("REFUSED -- FAIL_CANARY_DRIFT: {0}" -f $bk["trip_detail"])
        Write-Output -NoEnumerate $false
        return
    }
    Write-Output -NoEnumerate $true
}

$sessionWouldAbort = $false
$status = "running"
$canaryOk = Invoke-SyntheticDriftCanaryAbortPath
Assert-True -Condition ($script:synth_tripped -eq $true) `
    -Message "synthetic canary 4-class breach sets tripped"
Assert-True -Condition ($script:synth_drift_tripped -eq $true) `
    -Message "synthetic record drift_tripped=true"
if (-not $canaryOk) {
    $sessionWouldAbort = $true
    $status = "FAIL_CANARY_DRIFT"
}
Assert-True -Condition ($canaryOk -is [bool] -and $canaryOk -eq $false) `
    -Message "synthetic abort path returns scalar `$false"
Assert-True -Condition $sessionWouldAbort -Message "caller sets abort when canary returns `$false"
Assert-True -Condition ($status -eq "FAIL_CANARY_DRIFT") `
    -Message "summary.status would be FAIL_CANARY_DRIFT (not complete)"

if ($failed -gt 0) {
    Write-Output ""
    Write-Output "FAILED: $failed assertion(s)"
    exit 1
}
Write-Output ""
Write-Output "OK: all assertions passed"
exit 0
