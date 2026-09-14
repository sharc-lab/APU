# Regression: canary drift bookkeeping must survive Write-Output-polluted returns
# and the n<3 calibration branch (DISPATCH P CANARY0, 2026-08-12):
#   Cannot convert value "rel_drift_t1" to type "System.Int32"
#   InvalidCastFromStringToInteger
#
# Exercises Update-CanaryDriftBookkeeping end-to-end with 1, 2, 3, and 4 canaries
# recorded (synthetic timings; no model / no matrix launch).
#
# Run:
#   powershell -NoProfile -File tools/test_canary_drift_bookkeeping.ps1

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

function New-CanaryGate {
    return [ordered]@{
        armed                 = $false
        calibration_complete  = $false
        ref_turn1_prefill_s   = $null
        ref_turn2_prefill_s   = $null
        early_max_rel_t1      = $null
        early_max_rel_t2      = $null
        threshold_t1          = $null
        threshold_t2          = $null
        derivation_applied    = $null
    }
}

function New-SyntheticCanaryRec {
    param(
        [int]$Index,
        [double]$T1,
        [double]$T2,
        [string]$Classification = "OK"
    )
    return [ordered]@{
        cell_index       = $Index
        is_canary        = $true
        canary_index     = $Index
        classification   = $Classification
        turn1_prefill_s  = $(if ($Classification -eq "OK") { $T1 } else { $null })
        turn2_prefill_s  = $(if ($Classification -eq "OK") { $T2 } else { $null })
    }
}

function Invoke-PollutedCanaryReturn {
    # Mirror pre-fix Invoke-OneDeltaCell: Write-Output progress + return OrderedDictionary
    # → System.Object[] of length 3. Member enumeration still exposes .classification.
    param([System.Collections.IDictionary]$Rec)
    Write-Output ("-- CANARY arm=gpu_only n_cached=4000 mode=RESIDENT delta=400 r=0 @ synthetic")
    Write-Output ("   classification={0} t1_prefill={1} t2_prefill={2}" -f `
        $Rec.classification, $Rec.turn1_prefill_s, $Rec.turn2_prefill_s)
    return $Rec
}

# --- 0) Document the exact CANARY0 hazard shape ---
$polluted = Invoke-PollutedCanaryReturn -Rec (New-SyntheticCanaryRec -Index 0 -T1 5.374 -T2 1.250)
Assert-True -Condition ($polluted -is [object[]]) -Message "polluted return is System.Object[]"
Assert-True -Condition (@($polluted).Count -eq 3) -Message "polluted return count=3"
Assert-True -Condition ($polluted.classification -eq "OK") `
    -Message "member enumeration still exposes .classification=OK (the trap)"
$indexThrew = $false
$indexFqid = $null
$indexMsg = $null
try {
    $polluted["rel_drift_t1"] = 0.0
} catch {
    $indexThrew = $true
    $indexFqid = [string]$_.FullyQualifiedErrorId
    $indexMsg = [string]$_.Exception.Message
}
Assert-True -Condition $indexThrew -Message "string-key index on Object[] throws"
Assert-True -Condition ($indexFqid -match "InvalidCastFromStringToInteger" `
        -or $indexMsg -match 'rel_drift_t1') `
    -Message "error is InvalidCastFromStringToInteger / rel_drift_t1 (fqid='$indexFqid')"

# --- 1) Get-SinglePipelineRecord unwraps Object[] via ConvertTo-ObjectArray ---
$unwrapped = Get-SinglePipelineRecord -InputObject $polluted
Assert-True -Condition ($unwrapped -is [System.Collections.IDictionary]) `
    -Message "Get-SinglePipelineRecord -> IDictionary"
Assert-True -Condition ([double]$unwrapped.turn1_prefill_s -eq 5.374) `
    -Message "unwrapped turn1_prefill_s=5.374"
$unwrapped["rel_drift_t1"] = 0.01
Assert-True -Condition ($unwrapped["rel_drift_t1"] -eq 0.01) `
    -Message "string-key write succeeds on unwrapped record"

# --- 2) End-to-end canary path: 1, 2, 3, 4 canaries (calib_n=3) ---
$calibN = 3
$floor = 0.05
# Stable early canaries (within floor after arm); canary 3 drifts hard on turn1.
$synth = @(
    @{ t1 = 5.374; t2 = 1.250 },
    @{ t1 = 5.400; t2 = 1.260 },
    @{ t1 = 5.350; t2 = 1.240 },
    @{ t1 = 8.000; t2 = 1.250 }  # post-arm trip
)

$gate = New-CanaryGate
$records = New-Object System.Collections.Generic.List[object]
$armedAt = $null
$trippedAt = $null

for ($i = 0; $i -lt $synth.Count; $i++) {
    $clean = New-SyntheticCanaryRec -Index $i -T1 $synth[$i].t1 -T2 $synth[$i].t2
    $raw = Invoke-PollutedCanaryReturn -Rec $clean
    Assert-True -Condition ($raw -is [object[]]) `
        -Message ("canary{0} raw return is Object[] (pollution still simulated)" -f $i)

    $bk = $null
    $threw = $false
    try {
        $bk = Update-CanaryDriftBookkeeping `
            -Rec $raw `
            -CanaryGate $gate `
            -PriorCanaryRecords (ConvertTo-ObjectArray $records) `
            -CanaryCalibrationCount $calibN `
            -CanaryRelDriftFloor $floor
    } catch {
        $threw = $true
        Write-Output ("FAIL  canary{0} Update-CanaryDriftBookkeeping threw: {1}" -f $i, $_.Exception.Message)
        $script:failed++
    }
    Assert-True -Condition (-not $threw) `
        -Message ("canary{0} bookkeeping did not throw" -f $i)
    if ($threw) { break }

    Assert-True -Condition ($bk -is [hashtable]) `
        -Message ("canary{0} bookkeeping return is hashtable (NoEnumerate)" -f $i)
    Assert-True -Condition ($bk.rec -is [System.Collections.IDictionary]) `
        -Message ("canary{0} bk.rec is IDictionary" -f $i)
    Assert-True -Condition ($null -ne $bk.rec.PSObject.Properties["rel_drift_t1"] `
            -or $bk.rec.Contains("rel_drift_t1")) `
        -Message ("canary{0} record has rel_drift_t1 key after bookkeeping" -f $i)

    $nAfter = $i + 1
    if ($nAfter -lt $calibN) {
        Assert-True -Condition (-not $gate.calibration_complete) `
            -Message ("canary{0} (n={1}<{2}): still calibrating" -f $i, $nAfter, $calibN)
        Assert-True -Condition (-not $gate.armed) `
            -Message ("canary{0}: gate not armed during calibration" -f $i)
        Assert-True -Condition (-not $bk.tripped) `
            -Message ("canary{0}: not tripped during calibration" -f $i)
        Assert-True -Condition (-not $bk.just_armed) `
            -Message ("canary{0}: just_armed=false during calibration" -f $i)
    } elseif ($nAfter -eq $calibN) {
        Assert-True -Condition $gate.calibration_complete `
            -Message ("canary{0} (n={1}): calibration_complete" -f $i, $nAfter)
        Assert-True -Condition $gate.armed `
            -Message ("canary{0}: gate armed at calib boundary" -f $i)
        Assert-True -Condition $bk.just_armed `
            -Message ("canary{0}: just_armed=true at calib boundary" -f $i)
        Assert-True -Condition (-not $bk.tripped) `
            -Message ("canary{0}: arming canary itself does not trip" -f $i)
        Assert-True -Condition ($null -ne $gate.threshold_t1 -and $gate.threshold_t1 -ge $floor) `
            -Message ("canary{0}: threshold_t1 >= floor ({1})" -f $i, $floor)
        $armedAt = $i
    } else {
        # n=4: synthetic turn1=8.0 vs ~5.37 ref → large rel drift → trip
        Assert-True -Condition $gate.calibration_complete `
            -Message ("canary{0}: still armed post-calibration" -f $i)
        Assert-True -Condition $bk.tripped `
            -Message ("canary{0}: drifted canary trips gate" -f $i)
        Assert-True -Condition ($bk.trip_detail -match "turn1 rel_drift") `
            -Message ("canary{0}: trip_detail names turn1" -f $i)
        $trippedAt = $i
    }

    $records.Add($bk.rec) | Out-Null
    # Plan serialization path used by Invoke-DriftCanary
    $planSlice = [ordered]@{
        canaries    = ConvertTo-ObjectArray $records
        canary_gate = $gate
    }
    $json = $null
    try {
        $json = $planSlice | ConvertTo-Json -Depth 10
    } catch {
        Write-Output ("FAIL  canary{0} plan serialize threw: {1}" -f $i, $_.Exception.Message)
        $script:failed++
    }
    Assert-True -Condition ($null -ne $json -and $json.Length -gt 0) `
        -Message ("canary{0}: ConvertTo-ObjectArray plan serialize ok" -f $i)
}

Assert-True -Condition ($armedAt -eq 2) -Message "gate armed on canary index 2 (3rd OK canary)"
Assert-True -Condition ($trippedAt -eq 3) -Message "gate tripped on canary index 3 (4th)"
Assert-True -Condition ($records.Count -eq 4) -Message "four canary records retained"

# --- 3) Clean (non-polluted) OrderedDictionary path also works for n=1 ---
$gate2 = New-CanaryGate
$cleanOnly = New-SyntheticCanaryRec -Index 0 -T1 1.0 -T2 0.5
$bk2 = Update-CanaryDriftBookkeeping `
    -Rec $cleanOnly `
    -CanaryGate $gate2 `
    -PriorCanaryRecords @() `
    -CanaryCalibrationCount 3 `
    -CanaryRelDriftFloor 0.05
Assert-True -Condition (-not $gate2.calibration_complete) `
    -Message "clean OrderedDictionary n=1 stays in calibration"
Assert-True -Condition ($bk2.rec["rel_drift_t1"] -eq $null -or $null -eq $bk2.rec["rel_drift_t1"]) `
    -Message "clean path writes rel_drift_t1 key (null during calib)"

# --- 4) Settle-shaped PSCustomObject (N-1 failure: no classification/tag whitelist) ---
$settlePso = [pscustomobject]@{
    ok           = $true
    available_mb = 9500.0
    min_mb       = 7000.0
    max_wait_s   = 120.0
    probe_error  = $null
}
$settleUnwrapped = $null
$settleThrew = $false
try {
    $settleUnwrapped = Get-SinglePipelineRecord -InputObject $settlePso
} catch {
    $settleThrew = $true
    Write-Output ("FAIL  settle PSCustomObject unwrap threw: {0}" -f $_.Exception.Message)
    $script:failed++
}
Assert-True -Condition (-not $settleThrew) `
    -Message "settle-shaped PSCustomObject unwrap does not throw"
Assert-True -Condition ($settleUnwrapped -is [System.Collections.IDictionary]) `
    -Message "settle PSCustomObject -> IDictionary (normalised)"
Assert-True -Condition ($true -eq $settleUnwrapped["ok"]) `
    -Message "settle unwrap ok=true via string indexer"
$settleUnwrapped["post_settle"] = "wrote"
Assert-True -Condition ($settleUnwrapped["post_settle"] -eq "wrote") `
    -Message "string-key write on normalised settle record"

# ConvertFrom-Json shape (WinPS 5.1: PSCustomObject, no -AsHashtable)
$fromJson = '{"ok":true,"available_mb":8800.0}' | ConvertFrom-Json
$jsonUnwrapped = Get-SinglePipelineRecord -InputObject $fromJson
Assert-True -Condition ($jsonUnwrapped -is [System.Collections.IDictionary]) `
    -Message "ConvertFrom-Json PSCustomObject -> IDictionary"
Assert-True -Condition ([double]$jsonUnwrapped["available_mb"] -eq 8800.0) `
    -Message "ConvertFrom-Json unwrap preserves available_mb"

# Multi-record refuse
$multiThrew = $false
try {
    Get-SinglePipelineRecord -InputObject @(
        [ordered]@{ a = 1 }
        [ordered]@{ b = 2 }
    ) | Out-Null
} catch {
    $multiThrew = ($_.Exception.Message -match "multi-record")
}
Assert-True -Condition $multiThrew -Message "two IDictionary peers -> multi-record refuse"

if ($failed -gt 0) {
    Write-Output ""
    Write-Output "FAILED: $failed assertion(s)"
    exit 1
}
Write-Output ""
Write-Output "OK: all assertions passed"
exit 0
