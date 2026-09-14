<#
.SYNOPSIS
  Unattended chain: ceiling ladder + delta-prefill matrix for gpu_only_u8 then gpu_only_u4.

.DESCRIPTION
  Thin orchestrator. One detached process runs four stages in order; each stage seals
  (or records failure) before the next begins. Failed stages are recorded; the chain
  continues.

  Stages:
    1. ceiling_a -Orchestrate -Arms gpu_only_u8   (native seal inside ceiling_a)
    2. delta_prefill -Orchestrate -Arms gpu_only_u8 -NCached 4000,12000 -Deltas 100,400,1000,2000
       then tools/seal_delta_prefill_session.py (derived_diagnostic)
    3. ceiling_a -Orchestrate -Arms gpu_only_u4
    4. delta_prefill for gpu_only_u4 + seal

  Prefer -Orchestrate (WMI-detached via spawn_detached.ps1). Preconditions: AC power,
  tier-1 closed, Available >= pre_run floor. Estimate must be <= 6 h (see
  derived/kv_precision/TIME_ESTIMATE.md).

  Mac:
    ssh xps "cd C:/Users/zjohn/Projects/gnn-hls-accel; powershell -NoProfile -File tools/chain_kv_precision.ps1 -Orchestrate"
    ssh xps "cd C:/Users/zjohn/Projects/gnn-hls-accel; powershell -NoProfile -File tools/chain_kv_precision.ps1 -Status"
#>
[CmdletBinding(DefaultParameterSetName = "Run")]
param(
    [Parameter(ParameterSetName = "Orchestrate")][switch]$Orchestrate,
    [Parameter(ParameterSetName = "Status")][switch]$Status,
    [Parameter(ParameterSetName = "DryRunGate")][switch]$DryRunGate,
    [Parameter(ParameterSetName = "Run")][switch]$DetachedWorker,
    [Parameter(ParameterSetName = "Run")][string]$ChainId,
    [Parameter(ParameterSetName = "Run")]
    [Parameter(ParameterSetName = "Orchestrate")]
    [Parameter(ParameterSetName = "Status")]
    [Parameter(ParameterSetName = "DryRunGate")]
    [string]$PythonExe = "C:\Users\zjohn\Projects\gnn-hls-accel\.venv-seam\Scripts\python.exe",
    [Parameter(ParameterSetName = "Run")]
    [Parameter(ParameterSetName = "Orchestrate")]
    [int]$CellTimeoutS = 1500,
    [Parameter(ParameterSetName = "Run")]
    [Parameter(ParameterSetName = "Orchestrate")]
    [int]$PollS = 30,
    [Parameter(ParameterSetName = "Run")]
    [Parameter(ParameterSetName = "Orchestrate")]
    [int]$StageTimeoutS = 28800
)

$ErrorActionPreference = "Stop"
$root = "C:\Users\zjohn\Projects\gnn-hls-accel"
. (Join-Path $root "tools\SeamPsCommon.ps1")

$outRoot = Join-Path $root "derived\kv_precision"
$launchDir = Join-Path $outRoot "_launches"
$statePath = Join-Path $launchDir "chain_launches.json"
$cfgPath = Join-Path $root "configs\delta_n.yaml"
$ceilingPs1 = Join-Path $root "tools\ceiling_a.ps1"
$deltaPs1 = Join-Path $root "tools\run_delta_prefill_matrix.ps1"
$sealPy = Join-Path $root "tools\seal_delta_prefill_session.py"
$estimatePath = Join-Path $outRoot "TIME_ESTIMATE.md"
$predictionPath = Join-Path $outRoot "PREDICTION_BEFORE_RUN.md"

$NCached = "4000,12000"
$Deltas = "100,400,1000,2000"
$ArmsOrder = @("gpu_only_u8", "gpu_only_u4")

try {
    $py = [System.IO.Path]::GetFullPath($PythonExe)
} catch {
    $py = $PythonExe
}

function Save-Json {
    param([string]$Path, [object]$Obj)
    $dir = Split-Path -Parent $Path
    if ($dir) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    $tmp = "$Path.partial"
    $utf8 = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($tmp, ($Obj | ConvertTo-Json -Depth 8), $utf8)
    Move-Item -LiteralPath $tmp -Destination $Path -Force
}

function Get-AcPowerOk {
    try {
        $batt = Get-CimInstance -ClassName Win32_Battery -ErrorAction SilentlyContinue
        if (-not $batt) {
            return [pscustomobject]@{ ok = $true; note = "no_battery_device_assume_desktop_ac"; BatteryStatus = $null }
        }
        # BatteryStatus 2 = Connected to AC (charging / on AC).
        $status = @($batt | ForEach-Object { [int]$_.BatteryStatus })
        $onAc = ($status | Where-Object { $_ -eq 2 }).Count -gt 0 -or ($status | Where-Object { $_ -eq 2 }).Count -eq $status.Count
        # Any battery not reporting 2 → refuse AC.
        $allAc = ($status | Where-Object { $_ -ne 2 }).Count -eq 0
        return [pscustomobject]@{
            ok            = [bool]$allAc
            note          = if ($allAc) { "ac_online" } else { "battery_or_unknown" }
            BatteryStatus = ($status -join ",")
        }
    } catch {
        return [pscustomobject]@{ ok = $false; note = "probe_failed:$($_.Exception.Message)"; BatteryStatus = $null }
    }
}

function Assert-Interpreter {
    if (-not (Test-Path -LiteralPath $py)) {
        Write-Host "REFUSED -- INTERPRETER_MISSING: $py"
        exit 2
    }
}

function Assert-EstimateOk {
    if (-not (Test-Path -LiteralPath $estimatePath)) {
        Write-Host "REFUSED -- missing $estimatePath (write TIME_ESTIMATE before launch)"
        exit 2
    }
    $text = Get-Content -LiteralPath $estimatePath -Raw
    if ($text -notmatch 'central_estimate_h:\s*([0-9.]+)') {
        Write-Host "REFUSED -- TIME_ESTIMATE.md missing central_estimate_h field"
        exit 2
    }
    $central = [double]$Matches[1]
    $upper = $central
    if ($text -match 'upper_estimate_h:\s*([0-9.]+)') {
        $upper = [double]$Matches[1]
    }
    if ($upper -gt 6.0) {
        Write-Host ("REFUSED -- upper_estimate_h={0:N2} > 6.0; operator must acknowledge before launch" -f $upper)
        Write-Host "  See derived/kv_precision/TIME_ESTIMATE.md and OPERATOR_CMD.md"
        exit 2
    }
    Write-Host ("estimate gate PASS -- central={0:N2}h upper={1:N2}h (<=6h)" -f $central, $upper)
}

function Assert-PredictionPresent {
    if (-not (Test-Path -LiteralPath $predictionPath)) {
        Write-Host "REFUSED -- missing $predictionPath (record prediction BEFORE launch)"
        exit 2
    }
    Write-Host "prediction present: $predictionPath"
}

# Reuse delta_prefill's DryRunGate for tier-1 + Available (same floor).
# Human lines go to Write-Host so they do not pollute the boolean return value.
function Invoke-CleanlinessGate {
    param([switch]$ReportOnly)
    $ac = Get-AcPowerOk
    Write-Host ("AC power         : ok={0} ({1}; BatteryStatus={2})" -f $ac.ok, $ac.note, $ac.BatteryStatus)
    if (-not $ac.ok) {
        Write-Host "REFUSED -- XPS must be on AC power"
        if (-not $ReportOnly) { exit 1 }
        return $false
    }
    Assert-Interpreter
    Assert-PredictionPresent
    Assert-EstimateOk
    # Nested stdout must not become this function's return value (PowerShell pipelines
    # all Write-Output into the caller's assignment).
    $armsCsv = $ArmsOrder -join ","
    & powershell -NoProfile -File $deltaPs1 -DryRunGate -PythonExe $py `
        -Arms "$armsCsv" -NCached "$NCached" -Deltas "$Deltas" -CellTimeoutS $CellTimeoutS `
        | ForEach-Object { Write-Host $_ }
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        Write-Host ("tier1/Available DryRunGate FAIL (exit={0})" -f $code)
        if (-not $ReportOnly) { exit $code }
        return $false
    }
    return $true
}

function Write-ChainHeartbeat {
    param(
        [string]$Dir,
        [string]$Cid,
        [string]$Phase,
        [hashtable]$Extra = $null
    )
    $hb = [ordered]@{
        chain_id      = $Cid
        phase         = $Phase
        heartbeat_utc = (Get-Date).ToUniversalTime().ToString("o")
        pid           = $PID
    }
    if ($Extra) {
        foreach ($k in $Extra.Keys) { $hb[$k] = $Extra[$k] }
    }
    Save-Json -Path (Join-Path $Dir "heartbeat.json") -Obj $hb
}

function Wait-ProcessExit {
    param([int]$ProcId, [int]$TimeoutS, [string]$Label)
    $deadline = (Get-Date).AddSeconds($TimeoutS)
    while ((Get-Date) -lt $deadline) {
        $p = Get-Process -Id $ProcId -ErrorAction SilentlyContinue
        if (-not $p) { return $true }
        Start-Sleep -Seconds $PollS
        Write-Host ("   waiting {0} pid={1} ..." -f $Label, $ProcId)
    }
    Write-Host ("WARNING: {0} pid={1} still alive after {2}s" -f $Label, $ProcId, $TimeoutS)
    return $false
}

function Wait-CeilingSealed {
    param(
        [string]$ResultPath,
        [string]$LogPath,
        [datetime]$NotBeforeUtc,
        [int]$TimeoutS
    )
    # Only accept THIS stage's --result-json (or a session result whose session_id
    # appears in this stage's launch log). Never pick an older sealed session.
    $deadline = (Get-Date).AddSeconds($TimeoutS)
    $lastPing = [datetime]::MinValue
    while ((Get-Date) -lt $deadline) {
        if ($ResultPath -and (Test-Path -LiteralPath $ResultPath)) {
            try {
                $fi = Get-Item -LiteralPath $ResultPath
                if ($fi.LastWriteTimeUtc -ge $NotBeforeUtc.AddMinutes(-1)) {
                    $j = Get-Content -LiteralPath $ResultPath -Raw | ConvertFrom-Json
                    if ($j.sealed -eq $true) {
                        return [pscustomobject]@{
                            ok         = $true
                            session_id = [string]$j.session_id
                            result     = $j
                        }
                    }
                }
            } catch { }
        }
        if (((Get-Date) - $lastPing).TotalSeconds -ge 60) {
            Write-Host ("   waiting ceiling seal result={0}" -f $ResultPath)
            $lastPing = Get-Date
        }
        Start-Sleep -Seconds $PollS
    }
    return [pscustomobject]@{ ok = $false; session_id = $null; result = $null }
}

function Wait-DeltaComplete {
    param([string]$SessionId, [int]$TimeoutS)
    $plan = Join-Path $root ("derived\delta_prefill\{0}\plan.json" -f $SessionId)
    $deadline = (Get-Date).AddSeconds($TimeoutS)
    while ((Get-Date) -lt $deadline) {
        if (Test-Path -LiteralPath $plan) {
            try {
                $j = Get-Content -LiteralPath $plan -Raw | ConvertFrom-Json
                if ($j.status -eq "complete" -or $j.status -eq "FAIL") {
                    return [pscustomobject]@{ ok = ($j.status -eq "complete"); status = [string]$j.status; plan = $j }
                }
            } catch { }
        }
        Start-Sleep -Seconds $PollS
    }
    return [pscustomobject]@{ ok = $false; status = "timeout"; plan = $null }
}

function Invoke-CeilingStage {
    param([string]$Arm, [string]$ChainDir, [string]$StageName)
    Write-Host ("=== STAGE {0}: ceiling_a -Arms {1} ===" -f $StageName, $Arm)
    Write-ChainHeartbeat -Dir $ChainDir -Cid $ChainId -Phase "ceiling_launch" -Extra @{ arm = $Arm; stage = $StageName }

    $notBefore = [datetime]::UtcNow
    # ceiling_a.ps1 -Orchestrate already gates contending + spawns detached.
    $out = & powershell -NoProfile -File $ceilingPs1 -Orchestrate -Arms $Arm -PythonExe $py 2>&1
    $out | ForEach-Object { Write-Host $_ }
    if ($LASTEXITCODE -ne 0) {
        return [ordered]@{
            stage = $StageName; kind = "ceiling_a"; arm = $Arm; ok = $false
            error = "ceiling_a Orchestrate exit=$LASTEXITCODE"; session_id = $null
        }
    }
    # Parse log path from launches.json (last entry for this arm).
    # Flatten nested {value,Count} wrappers; never assign an Object[] to $last
    # (member-enumeration makes $last.pid an Object[] → Int32 cast death).
    $launchesPath = Join-Path $root "derived\ceiling_a\_launches\launches.json"
    $parsed = Get-Content -LiteralPath $launchesPath -Raw | ConvertFrom-Json
    $launches = ConvertTo-ObjectArray (Get-LaunchEntriesFromNode -Node $parsed)
    $last = $null
    for ($i = $launches.Count - 1; $i -ge 0; $i--) {
        $e = $launches[$i]
        if ($null -eq $e) { continue }
        if ($e -is [System.Array]) { continue }
        if ([string]$e.arms -eq $Arm) { $last = $e; break }
    }
    if ($null -eq $last) {
        for ($i = $launches.Count - 1; $i -ge 0; $i--) {
            $e = $launches[$i]
            if ($null -ne $e -and -not ($e -is [System.Array])) { $last = $e; break }
        }
    }
    if ($null -eq $last) {
        return [ordered]@{
            stage = $StageName; kind = "ceiling_a"; arm = $Arm; ok = $false
            error = "no scalar launch record in $launchesPath"; session_id = $null
        }
    }
    $resultPath = [string]$last.result_path
    $logPath = [string]$last.log_path
    $pidLaunch = Get-LaunchRecordScalarPid -Record $last

    Write-ChainHeartbeat -Dir $ChainDir -Cid $ChainId -Phase "ceiling_wait" -Extra @{
        arm = $Arm; stage = $StageName; pid = $pidLaunch; result_path = $resultPath
    }

    # Poll result_json for sealed=true (cmd.exe pid from spawn_detached is transient;
    # do not treat process exit alone as stage completion).
    $sealed = Wait-CeilingSealed -ResultPath $resultPath -LogPath $logPath `
        -NotBeforeUtc $notBefore -TimeoutS $StageTimeoutS
    if (-not $sealed.ok) {
        # Stage timed out / died without seal — record and continue.
        return [ordered]@{
            stage = $StageName; kind = "ceiling_a"; arm = $Arm; ok = $false
            error = "ceiling did not seal within StageTimeoutS (result missing or sealed!=true)"
            session_id = $null
            result_path = $resultPath
            log_path = $logPath
        }
    }
    return [ordered]@{
        stage = $StageName; kind = "ceiling_a"; arm = $Arm; ok = $true
        session_id = $sealed.session_id
        ceiling_tokens = $sealed.result.ceiling_tokens
        verdict = $sealed.result.verdict
        result_path = $resultPath
        log_path = $logPath
        sealed = $true
    }
}

function Invoke-DeltaStage {
    param([string]$Arm, [string]$ChainDir, [string]$StageName)
    Write-Host ("=== STAGE {0}: delta_prefill -Arms {1} ===" -f $StageName, $Arm)
    Write-ChainHeartbeat -Dir $ChainDir -Cid $ChainId -Phase "delta_launch" -Extra @{ arm = $Arm; stage = $StageName }

    $tag = "kvprec_$Arm"
    $notBefore = [datetime]::UtcNow
    # Quote CSV strings so nested powershell never sees Object[] for multi-value params.
    $out = & powershell -NoProfile -File $deltaPs1 -Orchestrate `
        -Arms "$Arm" -NCached "$NCached" -Deltas "$Deltas" `
        -CellTimeoutS $CellTimeoutS -Tag $tag -PythonExe $py 2>&1
    $out | ForEach-Object { Write-Host $_ }
    if ($LASTEXITCODE -ne 0) {
        return [ordered]@{
            stage = $StageName; kind = "delta_prefill"; arm = $Arm; ok = $false
            error = "delta_prefill Orchestrate exit=$LASTEXITCODE"; session_id = $null
        }
    }

    $launchesPath = Join-Path $root "derived\delta_prefill\_launches\launches.json"
    $parsed = Get-Content -LiteralPath $launchesPath -Raw | ConvertFrom-Json
    $launches = ConvertTo-ObjectArray (Get-LaunchEntriesFromNode -Node $parsed)
    $last = $null
    for ($i = $launches.Count - 1; $i -ge 0; $i--) {
        $e = $launches[$i]
        if ($null -eq $e -or ($e -is [System.Array])) { continue }
        $launched = $null
        try { $launched = [datetime]$e.launched_utc } catch { }
        if ($e.arms -and [string]$e.arms -eq $Arm -and $launched -and $launched -ge $notBefore.AddMinutes(-1)) {
            $last = $e; break
        }
        if ([string]$e.matrix_tag -eq $tag) { $last = $e; break }
    }
    if ($null -eq $last) {
        for ($i = $launches.Count - 1; $i -ge 0; $i--) {
            $e = $launches[$i]
            if ($null -ne $e -and -not ($e -is [System.Array])) { $last = $e; break }
        }
    }
    if ($null -eq $last) {
        return [ordered]@{
            stage = $StageName; kind = "delta_prefill"; arm = $Arm; ok = $false
            error = "no scalar launch record in $launchesPath"; session_id = $null
        }
    }
    $sid = [string]$last.session_id
    $pidLaunch = Get-LaunchRecordScalarPid -Record $last

    Write-ChainHeartbeat -Dir $ChainDir -Cid $ChainId -Phase "delta_wait" -Extra @{
        arm = $Arm; stage = $StageName; pid = $pidLaunch; session_id = $sid
    }

    # Poll plan.json status (spawn pid may be transient cmd.exe).
    $done = Wait-DeltaComplete -SessionId $sid -TimeoutS $StageTimeoutS
    $sealOk = $false
    $sealError = $null
    if ($sid -and $done.status -ne "timeout") {
        Write-Host ("sealing delta_prefill session {0} (derived_diagnostic)..." -f $sid)
        & $py -u $sealPy --session-id $sid --allow-dirty 2>&1 | ForEach-Object { Write-Host $_ }
        if ($LASTEXITCODE -eq 0) { $sealOk = $true }
        else { $sealError = "seal_delta_prefill exit=$LASTEXITCODE" }
    } elseif (-not $sid) {
        $sealError = "no session_id from launches.json"
    } else {
        $sealError = "matrix did not reach complete/FAIL within StageTimeoutS"
    }

    return [ordered]@{
        stage = $StageName; kind = "delta_prefill"; arm = $Arm
        ok = [bool]($done.ok -and $sealOk)
        matrix_status = $done.status
        session_id = $sid
        sealed_derived = $sealOk
        seal_error = $sealError
        log_path = [string]$last.log_path
    }
}

# ----------------------------------------------------------------------------------------------
if ($Status) {
    if (-not (Test-Path -LiteralPath $statePath)) {
        Write-Output "no kv_precision chain has been launched"
        exit 0
    }
    $launches = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    $last = @($launches)[-1]
    Write-Output "last chain launch:"
    Write-Output ("  chain_id   : {0}" -f $last.chain_id)
    Write-Output ("  pid        : {0}" -f $last.pid)
    Write-Output ("  log        : {0}" -f $last.log_path)
    $alive = $null -ne (Get-Process -Id ([int]$last.pid) -ErrorAction SilentlyContinue)
    Write-Output ("  alive      : {0}" -f $alive)
    $hb = Join-Path $outRoot ("{0}\heartbeat.json" -f $last.chain_id)
    if (Test-Path -LiteralPath $hb) {
        Write-Output ""
        Write-Output "--- heartbeat ---"
        Get-Content -LiteralPath $hb -Raw
    }
    $summary = Join-Path $outRoot ("{0}\chain_result.json" -f $last.chain_id)
    if (Test-Path -LiteralPath $summary) {
        Write-Output ""
        Write-Output "--- chain_result ---"
        Get-Content -LiteralPath $summary -Raw
    }
    exit 0
}

if ($DryRunGate) {
    $ok = [bool](Invoke-CleanlinessGate -ReportOnly | Select-Object -Last 1)
    if (-not $ok) {
        Write-Output "dry-run FAIL -- BLOCKED_ON_OPERATOR"
        Write-Output "See derived/kv_precision/OPERATOR_CMD.md"
        exit 1
    }
    Write-Output "dry-run PASS -- AC + estimate + prediction + tier-1/Available"
    exit 0
}

if ($Orchestrate) {
    Assert-Interpreter
    $gateOk = [bool](Invoke-CleanlinessGate | Select-Object -Last 1)
    if (-not $gateOk) {
        Write-Output "BLOCKED_ON_OPERATOR -- cleanliness gate failed; not launching"
        Write-Output "See derived/kv_precision/OPERATOR_CMD.md"
        exit 1
    }

    New-Item -ItemType Directory -Force -Path $launchDir | Out-Null
    $cid = [guid]::NewGuid().ToString()
    $tag = "kv_precision_chain_" + (Get-Date -Format "yyyyMMdd_HHmmss")
    $log = Join-Path $launchDir "$tag.log"
    $self = Join-Path $root "tools\chain_kv_precision.ps1"
    $inner = 'powershell -NoProfile -File "' + $self + '"' +
             ' -DetachedWorker -ChainId ' + $cid +
             ' -CellTimeoutS ' + $CellTimeoutS +
             ' -PollS ' + $PollS +
             ' -StageTimeoutS ' + $StageTimeoutS +
             ' -PythonExe "' + $py + '"'
    $json = & (Join-Path $root "tools\spawn_detached.ps1") `
        -CommandLine $inner -LogPath $log -WorkingDirectory $root
    $info = $json | ConvertFrom-Json
    $entry = [pscustomobject]@{
        tag          = $tag
        chain_id     = $cid
        pid          = $info.pid
        parent_name  = $info.parent_name
        launched_utc = (Get-Date).ToUniversalTime().ToString("o")
        log_path     = $log
        arms         = ($ArmsOrder -join ",")
        n_cached     = $NCached
        deltas       = $Deltas
    }
    $prev = @()
    if (Test-Path -LiteralPath $statePath) {
        $prev = @(Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json)
    }
    Save-Json -Path $statePath -Obj (@($prev) + $entry)

    Write-Output "launched detached kv_precision chain"
    Write-Output "  chain_id : $cid"
    Write-Output "  pid      : $($info.pid)"
    Write-Output "  log      : $log"
    Write-Output "  stages   : ladder u8 -> delta u8 -> ladder u4 -> delta u4"
    Write-Output ""
    Write-Output "Poll: powershell -NoProfile -File tools/chain_kv_precision.ps1 -Status"
    Write-Output ("Heartbeat: derived/kv_precision/{0}/heartbeat.json" -f $cid)
    exit 0
}

if (-not $DetachedWorker) {
    Write-Output "REFUSED -- pass -Orchestrate (detached chain) or -DryRunGate / -Status"
    exit 1
}
if (-not $ChainId) {
    Write-Output "REFUSED -- -DetachedWorker requires -ChainId"
    exit 1
}

Assert-Interpreter
$chainDir = Join-Path $outRoot $ChainId
New-Item -ItemType Directory -Force -Path $chainDir | Out-Null
Write-ChainHeartbeat -Dir $chainDir -Cid $ChainId -Phase "start"

$stages = New-Object System.Collections.Generic.List[object]
$stageIdx = 0
foreach ($arm in $ArmsOrder) {
    $stageIdx++
    $ceil = Invoke-CeilingStage -Arm $arm -ChainDir $chainDir -StageName ("{0}_ceiling" -f $arm)
    $stages.Add($ceil) | Out-Null
    Save-Json -Path (Join-Path $chainDir "stages.json") -Obj @($stages)
    Write-ChainHeartbeat -Dir $chainDir -Cid $ChainId -Phase "after_ceiling" -Extra @{
        arm = $arm; ok = $ceil.ok; session_id = $ceil.session_id
    }

    $stageIdx++
    $delta = Invoke-DeltaStage -Arm $arm -ChainDir $chainDir -StageName ("{0}_delta" -f $arm)
    $stages.Add($delta) | Out-Null
    Save-Json -Path (Join-Path $chainDir "stages.json") -Obj @($stages)
    Write-ChainHeartbeat -Dir $chainDir -Cid $ChainId -Phase "after_delta" -Extra @{
        arm = $arm; ok = $delta.ok; session_id = $delta.session_id
    }
}

$result = [ordered]@{
    chain_id     = $ChainId
    status       = "complete"
    ended_utc    = (Get-Date).ToUniversalTime().ToString("o")
    stages       = @($stages)
    all_ok       = -not (@($stages | Where-Object { -not $_.ok }).Count -gt 0)
    n_cached     = $NCached
    deltas       = $Deltas
    arms         = $ArmsOrder
}
Save-Json -Path (Join-Path $chainDir "chain_result.json") -Obj $result
Write-ChainHeartbeat -Dir $chainDir -Cid $ChainId -Phase "complete" -Extra @{ all_ok = $result.all_ok }
Write-Output ("chain complete all_ok={0}" -f $result.all_ok)
exit 0
