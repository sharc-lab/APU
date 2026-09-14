<#
.SYNOPSIS
  Session-level BFCL residency A/B: RESIDENT vs NON_RESIDENT on real multi_turn_base.

.DESCRIPTION
  Four cells (chain via spawn_detached), all n=20 paired:
    1. gpu_only  RESIDENT      n=20
    2. gpu_only  NON_RESIDENT  n=20
    3. arm A     RESIDENT      n=20
    4. arm A     NON_RESIDENT  n=20  (cold prefill ~55 s turn-1; budgeted)

  ChatHistory fix (DISPATCH K): raw messages + tools + enable_thinking=False;
  no pre-rendered string into start_chat. Same entry prefix / seed for pairing.

  DISPATCH L cold fix: NON_RESIDENT loads with SchedulerConfig(enable_prefix_caching=False).
  -Orchestrate runs session_residency_cold_control (gpu_only, 1 entry, turn-2) BEFORE
    launching the four-cell matrix; matrix is refused if the control fails.

  Prefer -Orchestrate (ssh_detached). Preconditions: AC, tier-1 closed, Available>=7000.

  Mac:
    ssh xps "cd C:/Users/zjohn/Projects/gnn-hls-accel; powershell -NoProfile -File tools/run_bfcl_session_residency.ps1 -DryRunGate"
    ssh xps "cd C:/Users/zjohn/Projects/gnn-hls-accel; powershell -NoProfile -File tools/run_bfcl_session_residency.ps1 -ColdControl"
    ssh xps "cd C:/Users/zjohn/Projects/gnn-hls-accel; powershell -NoProfile -File tools/run_bfcl_session_residency.ps1 -Orchestrate"
    ssh xps "cd C:/Users/zjohn/Projects/gnn-hls-accel; powershell -NoProfile -File tools/run_bfcl_session_residency.ps1 -Status"
#>
[CmdletBinding(DefaultParameterSetName = "Run")]
param(
    [Parameter(ParameterSetName = "Orchestrate")][switch]$Orchestrate,
    [Parameter(ParameterSetName = "Status")][switch]$Status,
    [Parameter(ParameterSetName = "DryRunGate")][switch]$DryRunGate,
    [Parameter(ParameterSetName = "ColdControl")][switch]$ColdControl,
    [Parameter(ParameterSetName = "Run")][switch]$DetachedWorker,
    [Parameter(ParameterSetName = "Run")][string]$SessionId,
    [Parameter(ParameterSetName = "Run")]
    [Parameter(ParameterSetName = "Orchestrate")]
    [Parameter(ParameterSetName = "Status")]
    [Parameter(ParameterSetName = "DryRunGate")]
    [Parameter(ParameterSetName = "ColdControl")]
    [string]$PythonExe = "C:\Users\zjohn\Projects\gnn-hls-accel\.venv-seam\Scripts\python.exe",
    [Parameter(ParameterSetName = "Run")]
    [Parameter(ParameterSetName = "Orchestrate")]
    [Parameter(ParameterSetName = "ColdControl")]
    [int]$MaxNewTokens = 512,
    [Parameter(ParameterSetName = "Run")]
    [Parameter(ParameterSetName = "Orchestrate")]
    [Parameter(ParameterSetName = "ColdControl")]
    [int]$Seed = 20260810,
    [Parameter(ParameterSetName = "Run")]
    [Parameter(ParameterSetName = "Orchestrate")]
    [int]$PollS = 30,
    [Parameter(ParameterSetName = "ColdControl")]
    [Parameter(ParameterSetName = "Orchestrate")]
    [string]$ColdControlArm = "gpu_only"
)

$ErrorActionPreference = "Stop"
$root = "C:\Users\zjohn\Projects\gnn-hls-accel"
. (Join-Path $root "tools\SeamPsCommon.ps1")

$outRoot = Join-Path $root "derived\bfcl_feasibility\session_residency"
$launchDir = Join-Path $outRoot "_launches"
$statePath = Join-Path $launchDir "launches.json"
$deltaPs1 = Join-Path $root "tools\run_delta_prefill_matrix.ps1"
$probePy = Join-Path $root "tools\bfcl_feasibility_probe.py"
$estimatePath = Join-Path $root "derived\bfcl_feasibility\TIME_ESTIMATE_SESSION_RESIDENCY.md"
$operatorCmd = Join-Path $root "derived\bfcl_feasibility\OPERATOR_CMD_SESSION_RESIDENCY.md"

try {
    $py = [System.IO.Path]::GetFullPath($PythonExe)
} catch {
    $py = $PythonExe
}

$Cells = @(
    [pscustomobject]@{ arm = "gpu_only"; mode = "RESIDENT";     n = 20; reason = $null }
    [pscustomobject]@{ arm = "gpu_only"; mode = "NON_RESIDENT"; n = 20; reason = $null }
    [pscustomobject]@{ arm = "A";        mode = "RESIDENT";     n = 20; reason = $null }
    [pscustomobject]@{ arm = "A";        mode = "NON_RESIDENT"; n = 20
        reason = "arm A cold prefill ~55s turn-1 fails TTFT<=10s SLO by construction; n=20 paired" }
)

function Save-Json {
    param([string]$Path, [object]$Obj)
    $dir = Split-Path -Parent $Path
    if ($dir) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    $tmp = "$Path.partial"
    $utf8 = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($tmp, ($Obj | ConvertTo-Json -Depth 10), $utf8)
    Move-Item -LiteralPath $tmp -Destination $Path -Force
}

function Get-AcPowerOk {
    try {
        $batt = Get-CimInstance -ClassName Win32_Battery -ErrorAction SilentlyContinue
        if (-not $batt) {
            return [pscustomobject]@{ ok = $true; note = "no_battery_assume_ac"; BatteryStatus = $null }
        }
        $status = @($batt | ForEach-Object { [int]$_.BatteryStatus })
        $allAc = ($status | Where-Object { $_ -ne 2 }).Count -eq 0
        return [pscustomobject]@{
            ok = [bool]$allAc
            note = if ($allAc) { "ac_online" } else { "battery_or_unknown" }
            BatteryStatus = ($status -join ",")
        }
    } catch {
        return [pscustomobject]@{ ok = $false; note = "probe_failed:$($_.Exception.Message)"; BatteryStatus = $null }
    }
}

function Get-LaunchEntriesFromNode {
    param($Node)
    $out = New-Object System.Collections.Generic.List[object]
    foreach ($n in @($Node)) {
        if ($null -eq $n) { continue }
        $names = @($n.PSObject.Properties.Name)
        if ($names -contains "tag" -and $names -contains "pid" -and $names -contains "log_path") {
            $out.Add($n) | Out-Null
        } elseif ($names -contains "value") {
            foreach ($child in (Get-LaunchEntriesFromNode -Node $n.value)) { $out.Add($child) | Out-Null }
        }
    }
    return $out
}

function Get-Launches {
    if (-not (Test-Path -LiteralPath $statePath)) { return @() }
    $parsed = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    return ConvertTo-ObjectArray (Get-LaunchEntriesFromNode -Node $parsed)
}

function Save-Launches {
    param([object[]]$Entries)
    New-Item -ItemType Directory -Force -Path $launchDir | Out-Null
    Save-Json -Path $statePath -Obj @($Entries)
}

function Write-Heartbeat {
    param([string]$Dir, [string]$Sid, [string]$Phase, [hashtable]$Extra = $null)
    $hb = [ordered]@{
        session_id    = $Sid
        phase         = $Phase
        heartbeat_utc = (Get-Date).ToUniversalTime().ToString("o")
        pid           = $PID
    }
    if ($Extra) { foreach ($k in $Extra.Keys) { $hb[$k] = $Extra[$k] } }
    Save-Json -Path (Join-Path $Dir "heartbeat.json") -Obj $hb
}

function Invoke-CleanlinessGate {
    param([switch]$ReportOnly)
    $ac = Get-AcPowerOk
    Write-Host ("AC power         : ok={0} ({1}; BatteryStatus={2})" -f $ac.ok, $ac.note, $ac.BatteryStatus)
    if (-not $ac.ok) {
        Write-Host "REFUSED -- XPS must be on AC power"
        Write-Host "BLOCKED_ON_OPERATOR -- see $operatorCmd"
        if (-not $ReportOnly) { exit 1 }
        return $false
    }
    if (-not (Test-Path -LiteralPath $py)) {
        Write-Host "REFUSED -- INTERPRETER_MISSING: $py"
        if (-not $ReportOnly) { exit 2 }
        return $false
    }
    & powershell -NoProfile -File $deltaPs1 -DryRunGate -PythonExe $py `
        -Arms "A,gpu_only" -NCached "4000" -Deltas "100" |
        ForEach-Object { Write-Host $_ }
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        Write-Host ("tier1/Available DryRunGate FAIL (exit={0})" -f $code)
        Write-Host "BLOCKED_ON_OPERATOR -- see $operatorCmd"
        if (-not $ReportOnly) { exit $code }
        return $false
    }
    return $true
}

function Write-EstimateReport {
    Write-Host "=== TIME ESTIMATE (before launch) ==="
    if (Test-Path -LiteralPath $estimatePath) {
        Get-Content -LiteralPath $estimatePath | ForEach-Object { Write-Host $_ }
    } else {
        Write-Host "missing $estimatePath"
    }
    Write-Host ("cells: {0}" -f (($Cells | ForEach-Object { "{0}/{1}/n={2}" -f $_.arm, $_.mode, $_.n }) -join "; "))
}

function Invoke-ColdControl {
    param(
        [string]$Arm = "gpu_only",
        [string]$ControlOut
    )
    if (-not $ControlOut) {
        $ControlOut = Join-Path $outRoot "_cold_control"
    }
    New-Item -ItemType Directory -Force -Path $ControlOut | Out-Null
    Write-Host ("=== DISPATCH L cold control (arm={0} NON_RESIDENT, 1 entry, turn-2) ===" -f $Arm)
    $args = @(
        "-u", $probePy,
        "--mode", "session_residency_cold_control",
        "--out", $ControlOut,
        "--arm", $Arm,
        "--seed", "$Seed",
        "--max-new-tokens", "$MaxNewTokens"
    )
    $proc = Start-Process -FilePath $py -ArgumentList $args -WorkingDirectory $root `
        -NoNewWindow -PassThru -Wait
    $exit = $proc.ExitCode
    $reportPath = Join-Path $ControlOut ("session_residency_cold_control_{0}.json" -f $Arm)
    if ($exit -ne 0) {
        Write-Host ("cold control FAIL (exit={0}) -- refusing four-cell matrix" -f $exit)
        if (Test-Path -LiteralPath $reportPath) {
            Write-Host ("artifact: {0}" -f $reportPath)
        }
        return $false
    }
    Write-Host ("cold control PASS -- {0}" -f $reportPath)
    return $true
}

if ($DryRunGate) {
    Write-EstimateReport
    $ok = Invoke-CleanlinessGate -ReportOnly
    if (-not $ok) {
        Write-Output "dry-run FAIL -- BLOCKED_ON_OPERATOR"
        exit 1
    }
    Write-Output "dry-run PASS -- AC + tier-1 + Available floor"
    exit 0
}

if ($ColdControl) {
    if (-not (Invoke-CleanlinessGate)) { exit 1 }
    $ctrlOut = Join-Path $outRoot "_cold_control"
    if (-not (Invoke-ColdControl -Arm $ColdControlArm -ControlOut $ctrlOut)) {
        Write-Output "BLOCKED -- cold control failed; see DISPATCH_L note"
        exit 3
    }
    Write-Output "cold control PASS -- safe to -Orchestrate four-cell matrix"
    exit 0
}

if ($Status) {
    if (-not (Test-Path -LiteralPath $statePath)) {
        Write-Output "no session_residency launch recorded"
        exit 1
    }
    $launches = Get-Launches
    $last = $launches[-1]
    $alive = $null -ne (Get-Process -Id ([int]$last.pid) -ErrorAction SilentlyContinue)
    Write-Output ("tag        : {0}" -f $last.tag)
    Write-Output ("pid        : {0} alive={1}" -f $last.pid, $alive)
    Write-Output ("session_id : {0}" -f $last.session_id)
    Write-Output ("log        : {0}" -f $last.log_path)
    if ($last.session_id) {
        $hb = Join-Path $outRoot ("{0}\heartbeat.json" -f $last.session_id)
        if (Test-Path -LiteralPath $hb) {
            Write-Output ""
            Write-Output "--- heartbeat ---"
            Get-Content -LiteralPath $hb -Raw
        }
        $plan = Join-Path $outRoot ("{0}\plan.json" -f $last.session_id)
        if (Test-Path -LiteralPath $plan) {
            $p = Get-Content -LiteralPath $plan -Raw | ConvertFrom-Json
            Write-Output ("plan.status : {0}" -f $p.status)
        }
    }
    exit 0
}

if ($Orchestrate) {
    Write-EstimateReport
    if (-not (Invoke-CleanlinessGate)) { exit 1 }
    # DISPATCH L: do not launch the matrix until NON_RESIDENT is provably cold.
    $ctrlOut = Join-Path $outRoot "_cold_control"
    if (-not (Invoke-ColdControl -Arm $ColdControlArm -ControlOut $ctrlOut)) {
        Write-Output "BLOCKED -- cold control failed; four-cell matrix not launched"
        Write-Output "BLOCKED_ON_OPERATOR -- fix NON_RESIDENT cold path, then retry"
        exit 3
    }
    New-Item -ItemType Directory -Force -Path $launchDir | Out-Null
    $sid = [guid]::NewGuid().ToString()
    $tagLaunch = "bfcl_session_residency_" + (Get-Date -Format "yyyyMMdd_HHmmss")
    $log = Join-Path $launchDir "$tagLaunch.log"
    $self = Join-Path $root "tools\run_bfcl_session_residency.ps1"
    $inner = 'set SEAM_LAUNCH_CONTEXT=ssh_detached' +
             '&& powershell -NoProfile -File "' + $self + '"' +
             ' -DetachedWorker -SessionId ' + $sid +
             ' -PythonExe "' + $py + '"' +
             ' -MaxNewTokens ' + $MaxNewTokens +
             ' -Seed ' + $Seed
    $json = & (Join-Path $root "tools\spawn_detached.ps1") `
        -CommandLine $inner -LogPath $log -WorkingDirectory $root
    $info = $json | ConvertFrom-Json
    $entry = [pscustomobject]@{
        tag            = $tagLaunch
        pid            = $info.pid
        parent_name    = $info.parent_name
        ppid           = $info.ppid
        launch_context = "ssh_detached"
        session_id     = $sid
        launched_utc   = (Get-Date).ToUniversalTime().ToString("o")
        log_path       = $log
        cold_control   = (Join-Path $ctrlOut ("session_residency_cold_control_{0}.json" -f $ColdControlArm))
    }
    Save-Launches -Entries (@(Get-Launches) + $entry)
    Write-Output "launched detached bfcl session_residency chain"
    Write-Output ("  tag        : {0}" -f $tagLaunch)
    Write-Output ("  pid        : {0}" -f $info.pid)
    Write-Output ("  session_id : {0}" -f $sid)
    Write-Output ("  log        : {0}" -f $log)
    Write-Output ("  cold_ctrl  : {0}" -f $entry.cold_control)
    Write-Output "Close this session. Poll with: tools/run_bfcl_session_residency.ps1 -Status"
    exit 0
}

if (-not $DetachedWorker) {
    Write-Output "REFUSED -- use -Orchestrate (detached) or -DryRunGate / -ColdControl / -Status"
    exit 2
}
if (-not $SessionId) {
    Write-Output "REFUSED -- -DetachedWorker requires -SessionId"
    exit 1
}

$sessionDir = Join-Path $outRoot $SessionId
New-Item -ItemType Directory -Force -Path $sessionDir | Out-Null
$plan = [ordered]@{
    session_id     = $SessionId
    kind           = "bfcl_session_residency"
    seed           = $Seed
    max_new_tokens = $MaxNewTokens
    cells          = @($Cells | ForEach-Object {
            [ordered]@{ arm = $_.arm; residency_mode = $_.mode; n_entries = $_.n; reduced_reason = $_.reason }
        })
    started_utc    = (Get-Date).ToUniversalTime().ToString("o")
    status         = "running"
    cell_results   = @()
}
Save-Json -Path (Join-Path $sessionDir "plan.json") -Obj $plan
Write-Heartbeat -Dir $sessionDir -Sid $SessionId -Phase "starting"

$cellResults = New-Object System.Collections.Generic.List[object]
$cellIndex = 0
foreach ($cell in $Cells) {
    $label = "{0}_{1}" -f $cell.arm, $cell.mode
    Write-Output ("=== cell {0}: arm={1} mode={2} n={3} ===" -f $cellIndex, $cell.arm, $cell.mode, $cell.n)
    Write-Heartbeat -Dir $sessionDir -Sid $SessionId -Phase "cell" -Extra @{
        cell_index = $cellIndex; arm = $cell.arm; residency_mode = $cell.mode; n_entries = $cell.n
    }
    $args = @(
        "-u", $probePy,
        "--mode", "run_session_residency",
        "--out", $sessionDir,
        "--arm", $cell.arm,
        "--residency-mode", $cell.mode,
        "--n-entries", "$($cell.n)",
        "--seed", "$Seed",
        "--max-new-tokens", "$MaxNewTokens"
    )
    $started = (Get-Date).ToUniversalTime().ToString("o")
    $proc = Start-Process -FilePath $py -ArgumentList $args -WorkingDirectory $root `
        -NoNewWindow -PassThru -Wait
    $exit = $proc.ExitCode
    $reportName = "session_residency_{0}_{1}_report.json" -f $cell.arm, $cell.mode
    $reportPath = Join-Path $sessionDir $reportName
    $rec = [ordered]@{
        cell_index     = $cellIndex
        arm            = $cell.arm
        residency_mode = $cell.mode
        n_entries      = $cell.n
        reduced_reason = $cell.reason
        started_utc    = $started
        ended_utc      = (Get-Date).ToUniversalTime().ToString("o")
        exit_code      = $exit
        report         = $reportPath
        report_exists  = (Test-Path -LiteralPath $reportPath)
    }
    $cellResults.Add($rec) | Out-Null
    $cellIndex++
}

# Paired comparisons per arm when both reports exist.
$compareArtifacts = New-Object System.Collections.Generic.List[object]
foreach ($arm in @("gpu_only", "A")) {
    $rPath = Join-Path $sessionDir ("session_residency_{0}_RESIDENT_report.json" -f $arm)
    $nPath = Join-Path $sessionDir ("session_residency_{0}_NON_RESIDENT_report.json" -f $arm)
    if ((Test-Path -LiteralPath $rPath) -and (Test-Path -LiteralPath $nPath)) {
        $cmpArgs = @(
            "-u", $probePy,
            "--mode", "compare_session_residency",
            "--out", $sessionDir,
            "--resident-report", $rPath,
            "--non-resident-report", $nPath
        )
        $cproc = Start-Process -FilePath $py -ArgumentList $cmpArgs -WorkingDirectory $root `
            -NoNewWindow -PassThru -Wait
        $cmpPath = Join-Path $sessionDir ("session_residency_compare_{0}.json" -f $arm)
        $compareArtifacts.Add([ordered]@{
                arm = $arm; path = $cmpPath; exit_code = $cproc.ExitCode
                exists = (Test-Path -LiteralPath $cmpPath)
            }) | Out-Null
    }
}

$plan.cell_results = $cellResults.ToArray()
$plan.compare_artifacts = $compareArtifacts.ToArray()
$plan.ended_utc = (Get-Date).ToUniversalTime().ToString("o")
$plan.status = "complete"
Save-Json -Path (Join-Path $sessionDir "plan.json") -Obj $plan
Save-Json -Path (Join-Path $sessionDir "summary.json") -Obj $plan
Write-Heartbeat -Dir $sessionDir -Sid $SessionId -Phase "complete" -Extra @{
    cells_completed = $cellResults.Count
}
Save-Json -Path (Join-Path $launchDir "last_result.json") -Obj ([ordered]@{
        status     = "complete"
        session_id = $SessionId
        cells      = $cellResults.Count
        ended_utc  = $plan.ended_utc
    })
Write-Output "=== session_residency complete ==="
Write-Output ("session_dir : {0}" -f $sessionDir)
exit 0
