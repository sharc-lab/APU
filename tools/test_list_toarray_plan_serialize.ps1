# Regression: System.Collections.Generic.List[object] must not be wrapped with @()
# before plan/summary serialization. @($list) throws:
#   System.ArgumentException: Argument types do not match
# Seen in run_gpu_only_matrix.ps1 (2026-08-07) and run_delta_prefill_matrix.ps1
# session 1f519c47-c554-4939-a16a-63e120cfb254 (2026-08-09, died in Get-CellSpecs).
#
# Also: bare .ToArray() fails when PowerShell unwraps a single-item List return to
# PSCustomObject (Invoke-DetachedSpawn / Get-Launches path). Use ConvertTo-ObjectArray
# from tools/SeamPsCommon.ps1 for either shape.
#
# Run:
#   powershell -NoProfile -File tools/test_list_toarray_plan_serialize.ps1

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

# --- 1) Document the hazard: @($List[object]) throws ---
$threw = $false
try {
    $hazard = New-Object System.Collections.Generic.List[object]
    $hazard.Add([ordered]@{ cell_index = 0; arm = "A" }) | Out-Null
    $null = @($hazard)
} catch [System.ArgumentException] {
    $threw = $true
}
Assert-True -Condition $threw -Message "@(List[object] of OrderedDictionary) throws ArgumentException"

# --- 2) Plan-serialisation path (fixed pattern) ---
$cellRecords = New-Object System.Collections.Generic.List[object]
$cellRecords.Add([ordered]@{
    cell_index     = 0
    arm            = "A"
    mode           = "RESIDENT"
    n_cached       = 12000
    delta          = 500
    repeat         = 0
    classification = "OK"
    turn2_prefill_s = 1.23
    error          = $null
}) | Out-Null
$cellRecords.Add([ordered]@{
    cell_index     = 1
    arm            = "gpu_only"
    mode           = "NON_RESIDENT"
    n_cached       = 12000
    delta          = 2000
    repeat         = 0
    classification = "OK"
    turn2_prefill_s = 2.34
    error          = $null
}) | Out-Null

$plan = $null
$json = $null
try {
    $plan = [ordered]@{
        tag        = "delta_prefill"
        status     = "complete"
        cells      = @()
    }
    # Intentional List.ToArray() on a known Generic.List (plan.cells aggregation).
    $plan.cells = $cellRecords.ToArray()
    $json = $plan | ConvertTo-Json -Depth 10
} catch {
    Write-Output "FAIL  plan serialize threw: $($_.Exception.GetType().FullName): $($_.Exception.Message)"
    $failed++
}

Assert-True -Condition ($null -ne $plan) -Message "plan object built without exception"
Assert-True -Condition ($plan.cells -is [object[]]) -Message "plan.cells is object[] after ToArray()"
Assert-True -Condition ($plan.cells.Count -eq 2) -Message "plan.cells count is 2"
Assert-True -Condition ($null -ne $json -and $json.Length -gt 0) -Message "ConvertTo-Json produced output"
Assert-True -Condition ($json -match '"cell_index"') -Message "JSON contains cell_index"

# --- 3) Get-CellSpecs-shaped List via ConvertTo-ObjectArray (early crash site) ---
$specs = New-Object System.Collections.Generic.List[object]
$specs.Add([pscustomobject]@{ arm = "A"; mode = "RESIDENT"; delta = 500 }) | Out-Null
$specs.Add([pscustomobject]@{ arm = "gpu_only"; mode = "NON_RESIDENT"; delta = 2000 }) | Out-Null
$specArr = $null
try {
    $specArr = ConvertTo-ObjectArray $specs
    $order = @($specArr)  # safe once already object[]
    $null = $order.Count
} catch {
    Write-Output "FAIL  specs ConvertTo-ObjectArray path threw: $($_.Exception.Message)"
    $failed++
}
Assert-True -Condition ($null -ne $specArr -and $specArr.Count -eq 2) -Message "Get-CellSpecs-shaped ConvertTo-ObjectArray yields 2 specs"

# --- 4) ConvertTo-ObjectArray handles all four input shapes ---
# 4a) Generic.List[object]
$listIn = New-Object System.Collections.Generic.List[object]
$listIn.Add([pscustomobject]@{ tag = "a"; pid = 1; log_path = "x" }) | Out-Null
$listIn.Add([pscustomobject]@{ tag = "b"; pid = 2; log_path = "y" }) | Out-Null
$fromList = $null
$threwList = $false
try {
    $fromList = ConvertTo-ObjectArray $listIn
} catch {
    $threwList = $true
    Write-Output "FAIL  List case threw: $($_.Exception.Message)"
    $failed++
}
Assert-True -Condition (-not $threwList) -Message "ConvertTo-ObjectArray(List[object]) does not throw"
Assert-True -Condition ($fromList -is [object[]] -and $fromList.Count -eq 2) -Message "ConvertTo-ObjectArray(List[object]) -> object[2]"

# 4b) PSCustomObject (single-item unwrap shape that broke bare .ToArray())
$singleObj = [pscustomobject]@{ tag = "solo"; pid = 99; log_path = "z" }
$fromObj = $null
$threwObj = $false
try {
    $fromObj = ConvertTo-ObjectArray $singleObj
} catch {
    $threwObj = $true
    Write-Output "FAIL  PSCustomObject case threw: $($_.Exception.Message)"
    $failed++
}
Assert-True -Condition (-not $threwObj) -Message "ConvertTo-ObjectArray(PSCustomObject) does not throw"
Assert-True -Condition (@($fromObj).Count -eq 1 -and @($fromObj)[0].tag -eq "solo") -Message "ConvertTo-ObjectArray(PSCustomObject) -> 1-element array"

# Bare .ToArray() on PSCustomObject must still throw (documents the hazard)
$bareThrew = $false
try {
    $null = $singleObj.ToArray()
} catch {
    $bareThrew = $true
}
Assert-True -Condition $bareThrew -Message "bare PSCustomObject.ToArray() throws (hazard still real)"

# 4c) single scalar
$fromScalar = $null
$threwScalar = $false
try {
    $fromScalar = ConvertTo-ObjectArray 42
} catch {
    $threwScalar = $true
    Write-Output "FAIL  scalar case threw: $($_.Exception.Message)"
    $failed++
}
Assert-True -Condition (-not $threwScalar) -Message "ConvertTo-ObjectArray(scalar) does not throw"
Assert-True -Condition (@($fromScalar).Count -eq 1 -and @($fromScalar)[0] -eq 42) -Message "ConvertTo-ObjectArray(scalar) -> @(42)"

# 4d) $null
$fromNull = $null
$threwNull = $false
try {
    $fromNull = ConvertTo-ObjectArray $null
} catch {
    $threwNull = $true
    Write-Output "FAIL  null case threw: $($_.Exception.Message)"
    $failed++
}
Assert-True -Condition (-not $threwNull) -Message "ConvertTo-ObjectArray(`$null) does not throw"
# return @() may assign as $null; callers re-wrap with @() (same as Get-Launches empty path).
Assert-True -Condition (@($fromNull).Count -eq 0) -Message "ConvertTo-ObjectArray(`$null) -> empty when wrapped with @()"

if ($failed -gt 0) {
    Write-Output ""
    Write-Output "FAILED: $failed assertion(s)"
    exit 1
}
Write-Output ""
Write-Output "OK: all assertions passed"
exit 0
