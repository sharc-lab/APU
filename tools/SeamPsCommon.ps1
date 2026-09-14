# Shared PowerShell helpers for SEAM tools (matrix scripts, ceiling_a, etc.).
# Dot-source: . (Join-Path $root "tools\SeamPsCommon.ps1")

function ConvertTo-ObjectArray {
    <#
    .SYNOPSIS
      Normalize List[object] / scalar / array to object[] without enumerating IDictionary.

    .DESCRIPTION
      `@($orderedDict)` / `@($hashtable)` enumerates DictionaryEntry items — never use bare
      `@()` on a record. Empty List.ToArray() must be -NoEnumerate or callers see $null.
    #>
    param($InputObject)
    if ($null -eq $InputObject) {
        Write-Output -NoEnumerate ([object[]]@())
        return
    }
    if ($InputObject -is [System.Collections.Generic.List[object]]) {
        Write-Output -NoEnumerate ([object[]]$InputObject.ToArray())
        return
    }
    # IDictionary (OrderedDictionary / Hashtable): keep as one element, do not enumerate keys.
    if ($InputObject -is [System.Collections.IDictionary]) {
        Write-Output -NoEnumerate ([object[]](, $InputObject))
        return
    }
    if ($InputObject -is [System.Array]) {
        Write-Output -NoEnumerate ([object[]]$InputObject)
        return
    }
    Write-Output -NoEnumerate ([object[]](, $InputObject))
}

function ConvertTo-SeamOrderedRecord {
    <#
    .SYNOPSIS
      Normalize one record to OrderedDictionary (string-key indexer safe).

    .DESCRIPTION
      Windows PowerShell 5.1 ConvertFrom-Json returns PSCustomObject; -AsHashtable does
      not exist there. Cell/settle helpers may also return [pscustomobject]@{...}.
      Callers that write $rec["key"] need IDictionary. Already-IDictionary inputs are
      returned as-is (no copy). Scalars / strings / arrays are refused.
    #>
    param(
        [Parameter(Mandatory = $true)]
        [AllowNull()]
        $InputObject
    )
    if ($null -eq $InputObject) {
        throw "REFUSED -- ConvertTo-SeamOrderedRecord: input is null"
    }
    if ($InputObject -is [System.Collections.IDictionary]) {
        Write-Output -NoEnumerate $InputObject
        return
    }
    if ($InputObject -is [string] -or $InputObject -is [System.Array] `
            -or $InputObject -is [System.ValueType]) {
        throw ("REFUSED -- ConvertTo-SeamOrderedRecord: not a record (type={0})" -f `
            $InputObject.GetType().FullName)
    }
    if ($InputObject -is [System.Management.Automation.PSObject]) {
        $names = @($InputObject.PSObject.Properties.Name)
        if ($names.Count -eq 0) {
            throw "REFUSED -- ConvertTo-SeamOrderedRecord: PSObject has no properties"
        }
        $od = [ordered]@{}
        foreach ($p in $InputObject.PSObject.Properties) {
            $od[$p.Name] = $p.Value
        }
        Write-Output -NoEnumerate $od
        return
    }
    throw ("REFUSED -- ConvertTo-SeamOrderedRecord: unsupported type {0}" -f `
        $InputObject.GetType().FullName)
}

function Test-SeamPipelineRecordShape {
    <#
    .SYNOPSIS
      True if $Item is an IDictionary or a non-scalar PSObject with properties.
    #>
    param($Item)
    if ($null -eq $Item) { return $false }
    if ($Item -is [System.Collections.IDictionary]) { return $true }
    if ($Item -is [string] -or $Item -is [System.Array] -or $Item -is [System.ValueType]) {
        return $false
    }
    if ($Item -is [System.Management.Automation.PSObject]) {
        return (@($Item.PSObject.Properties.Name).Count -gt 0)
    }
    return $false
}

function Get-SinglePipelineRecord {
    <#
    .SYNOPSIS
      Recover one record from a function return that may be Object[] polluted by Write-Output.

    .DESCRIPTION
      PowerShell merges Write-Output and `return` into one success stream. Callers that
      capture `$rec = Invoke-Foo` then index `$rec["key"]` hit InvalidCastFromStringToInteger
      when `$rec` is Object[] (array indexer requires Int32). Member enumeration still makes
      `$rec.classification` appear to work — that is the trap (DISPATCH P CANARY0, 2026-08-12).

      Flattens via ConvertTo-ObjectArray, then returns the last record in the stream as an
      OrderedDictionary (PSCustomObject from ConvertFrom-Json / [pscustomobject]@{} is
      normalised — WinPS 5.1 has no ConvertFrom-Json -AsHashtable).

      Refuses an empty stream. Refuses when two or more record-shaped items appear (ambiguous
      multi-record). Non-record companions (progress strings) are ignored so Write-Output
      pollution still unwraps.
    #>
    param(
        [Parameter(Mandatory = $true)]
        [AllowNull()]
        $InputObject
    )
    $arr = ConvertTo-ObjectArray $InputObject
    if ($arr.Count -eq 0) {
        throw "REFUSED -- pipeline record stream empty"
    }
    $records = New-Object System.Collections.Generic.List[object]
    foreach ($item in $arr) {
        if (Test-SeamPipelineRecordShape -Item $item) {
            $records.Add($item) | Out-Null
        }
    }
    if ($records.Count -eq 0) {
        $types = @(
            $arr | ForEach-Object {
                if ($null -eq $_) { "null" } else { $_.GetType().FullName }
            }
        ) -join ", "
        throw "REFUSED -- no dictionary/record in pipeline return (count=$($arr.Count); types=$types)"
    }
    if ($records.Count -gt 1) {
        $types = @(
            $records | ForEach-Object { $_.GetType().FullName }
        ) -join ", "
        throw ("REFUSED -- multi-record pipeline return (n_records={0}; types={1})" -f `
            $records.Count, $types)
    }
    # -NoEnumerate: bare `return $orderedDict` enumerates as DictionaryEntry[].
    $normalized = ConvertTo-SeamOrderedRecord -InputObject $records[0]
    Write-Output -NoEnumerate $normalized
}

function Get-RelativeDrift {
    param([double]$Value, [double]$Ref)
    if ($Ref -eq 0) { return $null }
    return [math]::Abs($Value - $Ref) / $Ref
}

function Get-MedianDouble {
    param([double[]]$vals)
    if (-not $vals -or $vals.Count -eq 0) { return $null }
    $s = @($vals | Sort-Object)
    $n = $s.Count
    if ($n % 2 -eq 1) { return [double]$s[[int]($n / 2)] }
    return ([double]$s[$n / 2 - 1] + [double]$s[$n / 2]) / 2.0
}

function Update-CanaryDriftBookkeeping {
    <#
    .SYNOPSIS
      Post-canary calibration / relative-drift gate. Mutates Rec and CanaryGate in place.

    .DESCRIPTION
      Calibration (n < CanaryCalibrationCount): record only; gate stays unarmed.
      At n == CanaryCalibrationCount: derive ref/threshold from early canaries and arm.
      After arm: compare relative drift; set tripped when either turn exceeds threshold.
      Always writes rel_drift_* / gate_* / drift_tripped onto the (unwrapped) record.

    .OUTPUTS
      Hashtable (NoEnumerate): rec, tripped, trip_detail, just_armed, derivation_applied
    #>
    param(
        [Parameter(Mandatory = $true)]$Rec,
        [Parameter(Mandatory = $true)]$CanaryGate,
        [AllowNull()]
        $PriorCanaryRecords,
        [Parameter(Mandatory = $true)][int]$CanaryCalibrationCount,
        [Parameter(Mandatory = $true)][double]$CanaryRelDriftFloor
    )
    $rec = Get-SinglePipelineRecord -InputObject $Rec
    $prior = ConvertTo-ObjectArray $PriorCanaryRecords
    $wasComplete = [bool]$CanaryGate.calibration_complete

    $relT1 = $null
    $relT2 = $null
    $tripped = $false
    $tripDetail = $null

    if ($rec.classification -eq "OK" -and $null -ne $rec.turn1_prefill_s -and $null -ne $rec.turn2_prefill_s) {
        if (-not $CanaryGate.calibration_complete) {
            $okCal = @(
                $prior | Where-Object {
                    $_.classification -eq "OK" -and $null -ne $_.turn1_prefill_s -and $null -ne $_.turn2_prefill_s
                }
            )
            $okCal = @($okCal + @([pscustomobject]$rec))
            if ($okCal.Count -ge $CanaryCalibrationCount) {
                $t1s = @($okCal | ForEach-Object { [double]$_.turn1_prefill_s })
                $t2s = @($okCal | ForEach-Object { [double]$_.turn2_prefill_s })
                $ref1 = Get-MedianDouble $t1s
                $ref2 = Get-MedianDouble $t2s
                $early1 = @($t1s | ForEach-Object { Get-RelativeDrift -Value $_ -Ref $ref1 })
                $early2 = @($t2s | ForEach-Object { Get-RelativeDrift -Value $_ -Ref $ref2 })
                $em1 = [double](($early1 | Measure-Object -Maximum).Maximum)
                $em2 = [double](($early2 | Measure-Object -Maximum).Maximum)
                $th1 = [math]::Max(2.0 * $em1, [double]$CanaryRelDriftFloor)
                $th2 = [math]::Max(2.0 * $em2, [double]$CanaryRelDriftFloor)
                $CanaryGate.armed = $true
                $CanaryGate.calibration_complete = $true
                $CanaryGate.ref_turn1_prefill_s = $ref1
                $CanaryGate.ref_turn2_prefill_s = $ref2
                $CanaryGate.early_max_rel_t1 = $em1
                $CanaryGate.early_max_rel_t2 = $em2
                $CanaryGate.threshold_t1 = $th1
                $CanaryGate.threshold_t2 = $th2
                $CanaryGate.derivation_applied = (
                    ("calib_n={0}; ref_t1={1:N6} ref_t2={2:N6}; early_max_rel_t1={3:N6} early_max_rel_t2={4:N6}; " +
                     "threshold_t1=max(2*early_max_t1,floor={5})={6:N6}; threshold_t2={7:N6}") -f `
                        $CanaryCalibrationCount, $ref1, $ref2, $em1, $em2, $CanaryRelDriftFloor, $th1, $th2
                )
            }
        } else {
            $relT1 = Get-RelativeDrift -Value ([double]$rec.turn1_prefill_s) `
                -Ref ([double]$CanaryGate.ref_turn1_prefill_s)
            $relT2 = Get-RelativeDrift -Value ([double]$rec.turn2_prefill_s) `
                -Ref ([double]$CanaryGate.ref_turn2_prefill_s)
            if ($null -ne $relT1 -and $relT1 -gt [double]$CanaryGate.threshold_t1) {
                $tripped = $true
                $tripDetail = ("turn1 rel_drift={0:N6} > threshold_t1={1:N6} (t1={2} ref={3})" -f `
                    $relT1, $CanaryGate.threshold_t1, $rec.turn1_prefill_s, $CanaryGate.ref_turn1_prefill_s)
            } elseif ($null -ne $relT2 -and $relT2 -gt [double]$CanaryGate.threshold_t2) {
                $tripped = $true
                $tripDetail = ("turn2 rel_drift={0:N6} > threshold_t2={1:N6} (t2={2} ref={3})" -f `
                    $relT2, $CanaryGate.threshold_t2, $rec.turn2_prefill_s, $CanaryGate.ref_turn2_prefill_s)
            }
        }
    } elseif ($CanaryGate.calibration_complete) {
        $tripped = $true
        $tripDetail = ("canary classification={0} (no usable turn1/turn2 after gate armed)" -f $rec.classification)
    }

    # IDictionary indexer is safe only after Get-SinglePipelineRecord unwrap.
    if ($rec -is [System.Collections.IDictionary]) {
        $rec["rel_drift_t1"] = $relT1
        $rec["rel_drift_t2"] = $relT2
        $rec["gate_armed"] = [bool]$CanaryGate.armed
        $rec["threshold_t1"] = $CanaryGate.threshold_t1
        $rec["threshold_t2"] = $CanaryGate.threshold_t2
        $rec["drift_tripped"] = $tripped
        $rec["trip_detail"] = $tripDetail
    } else {
        $rec | Add-Member -NotePropertyName rel_drift_t1 -NotePropertyValue $relT1 -Force
        $rec | Add-Member -NotePropertyName rel_drift_t2 -NotePropertyValue $relT2 -Force
        $rec | Add-Member -NotePropertyName gate_armed -NotePropertyValue ([bool]$CanaryGate.armed) -Force
        $rec | Add-Member -NotePropertyName threshold_t1 -NotePropertyValue $CanaryGate.threshold_t1 -Force
        $rec | Add-Member -NotePropertyName threshold_t2 -NotePropertyValue $CanaryGate.threshold_t2 -Force
        $rec | Add-Member -NotePropertyName drift_tripped -NotePropertyValue $tripped -Force
        $rec | Add-Member -NotePropertyName trip_detail -NotePropertyValue $tripDetail -Force
    }

    $justArmed = ((-not $wasComplete) -and [bool]$CanaryGate.calibration_complete)
    # -NoEnumerate: bare `return @{...}` enumerates as DictionaryEntry[].
    Write-Output -NoEnumerate ([hashtable]@{
            rec                 = $rec
            tripped             = $tripped
            trip_detail         = $tripDetail
            just_armed          = $justArmed
            derivation_applied  = $CanaryGate.derivation_applied
        })
}

function Get-LaunchEntriesFromNode {
    <#
    .SYNOPSIS
      Flatten launches.json trees that historically nested {value,Count} wrappers.
    #>
    param([Parameter(Mandatory = $true)]$Node)
    $out = New-Object System.Collections.Generic.List[object]
    foreach ($n in @($Node)) {
        if ($null -eq $n) { continue }
        $names = @($n.PSObject.Properties.Name)
        if ($names -contains "tag" -and $names -contains "pid" -and $names -contains "log_path") {
            $out.Add($n) | Out-Null
        } elseif ($names -contains "value") {
            foreach ($child in (Get-LaunchEntriesFromNode -Node $n.value)) {
                $out.Add($child) | Out-Null
            }
        }
    }
    return $out
}

function ConvertTo-IntList {
    <#
    .SYNOPSIS
      Normalize -NCached / -Deltas style multi-value params to int[].

    .DESCRIPTION
      PowerShell binds unquoted comma lists (4000,12000) as Object[] before -File
      parameter binding. A typed [int] param then throws
      "Cannot convert System.Object[] to System.Int32". A typed [string] param
      usually survives -File, but in-process splatting / nested calls can still
      deliver Object[] or int[]. Accept [object], flatten, split CSV strings.
    #>
    param(
        [Parameter(Mandatory = $true)]$Value,
        [Parameter(Mandatory = $true)][string]$Name
    )
    $out = New-Object System.Collections.Generic.List[int]
    foreach ($item in @($Value)) {
        if ($null -eq $item) { continue }
        if ($item -is [string]) {
            foreach ($part in $item.Split(",")) {
                $t = $part.Trim()
                if ($t -eq "") { continue }
                try {
                    $out.Add([int]$t) | Out-Null
                } catch {
                    throw "REFUSED -- $Name entry '$t' is not an integer (raw=$Value)"
                }
            }
            continue
        }
        # Nested arrays (Object[] / int[]): flatten one level; avoid treating a scalar
        # as IEnumerable (strings already handled above).
        if (($item -is [System.Array]) -or (
                $item -is [System.Collections.IList] -and -not ($item -is [string])
            )) {
            foreach ($sub in @($item)) {
                if ($null -eq $sub) { continue }
                if ($sub -is [string]) {
                    foreach ($part in $sub.Split(",")) {
                        $t = $part.Trim()
                        if ($t -eq "") { continue }
                        $out.Add([int]$t) | Out-Null
                    }
                } else {
                    $out.Add([int]$sub) | Out-Null
                }
            }
            continue
        }
        try {
            $out.Add([int]$item) | Out-Null
        } catch {
            throw "REFUSED -- $Name entry '$item' is not an integer (raw=$Value)"
        }
    }
    if ($out.Count -lt 1) {
        throw "REFUSED -- $Name must list at least one positive integer (got '$Value')"
    }
    # -NoEnumerate: callers must receive a flat int[], not Object[]{ int[] }.
    Write-Output -NoEnumerate ([int[]]$out.ToArray())
}

function ConvertTo-StringList {
    <#
    .SYNOPSIS
      Normalize comma-separated string params (e.g. -Arms) that may arrive as Object[].
    #>
    param(
        [Parameter(Mandatory = $true)]$Value,
        [Parameter(Mandatory = $true)][string]$Name
    )
    $out = New-Object System.Collections.Generic.List[string]
    foreach ($item in @($Value)) {
        if ($null -eq $item) { continue }
        if ($item -is [string]) {
            foreach ($part in $item.Split(",")) {
                $t = $part.Trim()
                if ($t -ne "") { $out.Add($t) | Out-Null }
            }
            continue
        }
        $t = ([string]$item).Trim()
        if ($t -ne "") { $out.Add($t) | Out-Null }
    }
    if ($out.Count -lt 1) {
        throw "REFUSED -- $Name must list at least one entry (got '$Value')"
    }
    Write-Output -NoEnumerate ([string[]]$out.ToArray())
}

function Get-LaunchRecordScalarPid {
    <#
    .SYNOPSIS
      Extract a single Int32 pid from a launch record (never member-enumerate an array).
    #>
    param([Parameter(Mandatory = $true)]$Record)
    if ($null -eq $Record) {
        throw "REFUSED -- launch record is null; cannot read pid"
    }
    if ($Record -is [System.Array] -or $Record -is [System.Collections.IList]) {
        throw "REFUSED -- launch record is a collection (member-enumeration trap); pick one entry first"
    }
    $raw = $Record.pid
    if ($null -eq $raw) {
        throw "REFUSED -- launch record missing pid"
    }
    if ($raw -is [System.Array] -or ($raw -is [System.Collections.IEnumerable] -and -not ($raw -is [string]))) {
        $arr = @($raw)
        if ($arr.Count -ne 1) {
            throw "REFUSED -- launch record pid is Object[] (count=$($arr.Count)); not a scalar"
        }
        $raw = $arr[0]
    }
    return [int]$raw
}

function Invoke-SeamChildProcess {
    <#
    .SYNOPSIS
      Start a child, wait up to TimeoutSeconds, kill the process tree on hang.

    .DESCRIPTION
      generation.timeout_s only bounds generate() inside the child. Session
      d5c98342-a0b2-41a9-b6e2-93ac7a39c3ba showed a child that wrote its cell
      record after CL_OUT_OF_RESOURCES then spun one core ~47 min (exit=-1 after
      operator kill). This wall-clock bound is the parent-side guard.

    .OUTPUTS
      Hashtable: exit_code, timed_out, elapsed_s, pid
    #>
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList,
        [Parameter(Mandatory = $true)][int]$TimeoutSeconds,
        [string]$WorkingDirectory = $null
    )
    if ($TimeoutSeconds -lt 1) {
        throw "Invoke-SeamChildProcess: TimeoutSeconds must be >= 1 (got $TimeoutSeconds)"
    }
    $start = Get-Date
    $psi = @{
        FilePath         = $FilePath
        ArgumentList     = $ArgumentList
        NoNewWindow      = $true
        PassThru         = $true
    }
    if ($WorkingDirectory) { $psi.WorkingDirectory = $WorkingDirectory }
    $proc = Start-Process @psi
    $pidVal = [int]$proc.Id
    $timedOut = $false
    $finished = $proc.WaitForExit([int]($TimeoutSeconds * 1000))
    if (-not $finished) {
        $timedOut = $true
        # Kill the whole tree — OpenVINO/oneDNN workers are children of python/powershell.
        & taskkill.exe /PID $pidVal /T /F 2>$null | Out-Null
        try { $proc.WaitForExit(15000) | Out-Null } catch { }
    }
    $elapsed = ((Get-Date) - $start).TotalSeconds
    $code = $null
    try {
        if (-not $proc.HasExited) {
            $code = -1
        } else {
            $code = [int]$proc.ExitCode
        }
    } catch {
        $code = -1
    }
    if ($timedOut) { $code = -1 }
    return @{
        exit_code = $code
        timed_out = $timedOut
        elapsed_s = [math]::Round($elapsed, 3)
        pid       = $pidVal
    }
}
