<#
.SYNOPSIS
  Delta-prefill under session residency: RESIDENT vs NON_RESIDENT.

.DESCRIPTION
  Diagnostic only. No seal, no amendment, no commit.
  Artifacts under derived/delta_prefill/.

  Matrix (one process per cell; turn 2 always in the same warm process as turn 1):
    arms        -Arms comma list (default A,gpu_only; also gpu_only_f16 / gpu_only_u8 / gpu_only_u4)
    n_cached    -NCached comma list (default 12000; e.g. 4000,12000)
    deltas      -Deltas comma list (default 500,2000)
    modes       RESIDENT and NON_RESIDENT at EVERY provided delta.
                NON_RESIDENT: finish_chat then full re-prefill of n_cached+delta.
                DEFECT (fixed): earlier revisions ran NON_RESIDENT only at max(-Deltas),
                so residency ratios at small deltas had no matched cold cell — see
                derived/delta_prefill/DEFECT_non_resident_max_delta_only.md.
    repeats     3, interleaved, randomized order of (arm, n_cached, mode, delta) per round
    max_new     64
    cell wall   -CellTimeoutS (default 1500). Parent kills hung children after the
                cell writes its record; records classification=TIMEOUT and continues.
                Derivation (session d5c98342): OK max elapsed_s=616.8; 2× → 1234;
                rounded up to 1500. Hung cell was 2856.5s (operator kill). generation.timeout_s
                (1800) bounds only generate() inside the child.

  Prefer -Orchestrate (WMI-detached via tools/spawn_detached.ps1). Declares
  launch_context=ssh_detached. Refuses if Cursor/Chrome/etc are resident.
  Pre-run: Available MBytes >= isolation.pre_run_available_mb_min (7000).

  From the Mac, with Cursor and browsers closed on the XPS:

    ssh xps "cd C:/Users/zjohn/Projects/gnn-hls-accel; powershell -NoProfile -File tools/run_delta_prefill_matrix.ps1 -Orchestrate"
    ssh xps "cd C:/Users/zjohn/Projects/gnn-hls-accel; powershell -NoProfile -File tools/run_delta_prefill_matrix.ps1 -Orchestrate -Arms gpu_only_u8 -NCached 4000,12000 -Deltas 100,400,1000,2000"
    ssh xps "cd C:/Users/zjohn/Projects/gnn-hls-accel; powershell -NoProfile -File tools/run_delta_prefill_matrix.ps1 -Orchestrate -Arms gpu_only_u4 -NCached 2000,4000,8000,12000 -Deltas 50,150,400,1000 -Tag dispatch_o_u4"
    ssh xps "cd C:/Users/zjohn/Projects/gnn-hls-accel; powershell -NoProfile -File tools/run_dispatch_p_precision_matrix.ps1 -Orchestrate"
    ssh xps "cd C:/Users/zjohn/Projects/gnn-hls-accel; powershell -NoProfile -File tools/run_delta_prefill_matrix.ps1 -Status"

  Drift canary (optional; required for DISPATCH P interleaved precision):
    -CanaryEveryN N>0 inserts a fixed canary cell (default gpu_only / nc=4000 /
    d=400 / RESIDENT) at the start and after every N matrix cells. Threshold is
    derived in-run from the canary's own early-run relative variance (see plan
    canary_gate); abort status=FAIL_CANARY_DRIFT rather than silently degrade.

  Three numbers (after complete):
    1. turn2_prefill_s RESIDENT vs NON_RESIDENT per (arm, n_cached, delta)
    2. gpu_only:A on turn2_prefill_s RESIDENT (max delta) when both arms present
    3. Whether turn2 RESIDENT scales across provided deltas (when >=2)

  Falsification: RESIDENT turn-2 not materially cheaper than NON_RESIDENT → KV not retained.
#>
[CmdletBinding(DefaultParameterSetName = "Run")]
param(
    [Parameter(ParameterSetName = "Run")]
    [Parameter(ParameterSetName = "Orchestrate")]
    [ValidateSet("local_console", "ssh_foreground", "ssh_detached")]
    [string]$LaunchContext = "ssh_foreground",

    [Parameter(ParameterSetName = "Run")]
    [Parameter(ParameterSetName = "Orchestrate")]
    [Parameter(ParameterSetName = "VerifyDetach")]
    [Parameter(ParameterSetName = "DryRunGate")]
    [Parameter(ParameterSetName = "EmitSchedule")]
    [string]$Tag = "delta_prefill",

    [Parameter(ParameterSetName = "Run")]
    [Parameter(ParameterSetName = "Orchestrate")]
    [Parameter(ParameterSetName = "Status")]
    [Parameter(ParameterSetName = "VerifyDetach")]
    [Parameter(ParameterSetName = "DryRunGate")]
    [Parameter(ParameterSetName = "EmitSchedule")]
    [string]$PythonExe = "C:\Users\zjohn\Projects\gnn-hls-accel\.venv-seam\Scripts\python.exe",

    # Comma-separated n_cached values. Default preserves single 12000.
    # Typed as [object] (not [int]/[string]): unquoted 4000,12000 binds as Object[] and
    # must not throw ConvertToFinalInvalidCastException at parameter binding.
    [Parameter(ParameterSetName = "Run")]
    [Parameter(ParameterSetName = "Orchestrate")]
    [Parameter(ParameterSetName = "DryRunGate")]
    [Parameter(ParameterSetName = "EmitSchedule")]
    [object]$NCached = "12000",

    # Comma-separated turn-2 deltas. Default preserves {500,2000}.
    [Parameter(ParameterSetName = "Run")]
    [Parameter(ParameterSetName = "Orchestrate")]
    [Parameter(ParameterSetName = "DryRunGate")]
    [Parameter(ParameterSetName = "EmitSchedule")]
    [object]$Deltas = "500,2000",

    # Comma-separated arm ids from configs/delta_n.yaml. Default preserves {A,gpu_only}.
    [Parameter(ParameterSetName = "Run")]
    [Parameter(ParameterSetName = "Orchestrate")]
    [Parameter(ParameterSetName = "DryRunGate")]
    [Parameter(ParameterSetName = "EmitSchedule")]
    [object]$Arms = "A,gpu_only",

    # Comma-separated FetchedModelSpec YAML paths. Default: single element from
    # configs/delta_n.yaml openvino.model_spec (today's behaviour). Participates in
    # the Fisher-Yates cell tuple. Nesting is innermost before modes so a single
    # default model yields the same cell_specs order and shuffle as before.
    [Parameter(ParameterSetName = "Run")]
    [Parameter(ParameterSetName = "Orchestrate")]
    [Parameter(ParameterSetName = "DryRunGate")]
    [Parameter(ParameterSetName = "EmitSchedule")]
    [object]$ModelSpecs = $null,

    # Canary weight IR — fixed reference; MUST NOT inherit the per-cell model.
    # Default: same delta_n.yaml openvino.model_spec as the matrix default.
    [Parameter(ParameterSetName = "Run")]
    [Parameter(ParameterSetName = "Orchestrate")]
    [Parameter(ParameterSetName = "DryRunGate")]
    [Parameter(ParameterSetName = "EmitSchedule")]
    [string]$CanaryModelSpec = $null,

    # Comma-separated pipeline types (llm | cb_no_eviction). Default: single
    # element "llm" so existing behaviour is byte-identical. Participates in the
    # Fisher-Yates cell tuple. Nesting is innermost before modes (inside
    # ModelSpecs) so a single default pipeline yields the same cell_specs order
    # and shuffle as the pre-PipelineTypes runner.
    [Parameter(ParameterSetName = "Run")]
    [Parameter(ParameterSetName = "Orchestrate")]
    [Parameter(ParameterSetName = "DryRunGate")]
    [Parameter(ParameterSetName = "EmitSchedule")]
    [object]$PipelineTypes = $null,

    # Canary pipeline — fixed reference; MUST NOT inherit the per-cell pipeline.
    # Default: llm (plain LLMPipeline). Same reasoning as -CanaryModelSpec.
    [Parameter(ParameterSetName = "Run")]
    [Parameter(ParameterSetName = "Orchestrate")]
    [Parameter(ParameterSetName = "DryRunGate")]
    [Parameter(ParameterSetName = "EmitSchedule")]
    [string]$CanaryPipelineType = "llm",

    # Optional pre-run amendment text file. Contents are copied into plan.json
    # BEFORE the first cell (prediction registration). Does not affect interleave.
    [Parameter(ParameterSetName = "Run")]
    [Parameter(ParameterSetName = "Orchestrate")]
    [Parameter(ParameterSetName = "EmitSchedule")]
    [string]$PreRunAmendmentFile = "",

    # WorkloadsSessionHost sibling-process re-kill interval. 0 = off (default;
    # preserves pre-N-1 matrix behaviour). N-1 sets 300.
    [Parameter(ParameterSetName = "Run")]
    [Parameter(ParameterSetName = "Orchestrate")]
    [Parameter(ParameterSetName = "DryRunGate")]
    [Parameter(ParameterSetName = "EmitSchedule")]
    [int]$WatchdogIntervalS = 0,

    # Abort if this orchestrator process WorkingSet exceeds N MB. 0 = off.
    # N-1 sets 512 (orchestrator must not eat the Available floor reserved for cells).
    [Parameter(ParameterSetName = "Run")]
    [Parameter(ParameterSetName = "Orchestrate")]
    [Parameter(ParameterSetName = "DryRunGate")]
    [Parameter(ParameterSetName = "EmitSchedule")]
    [int]$OrchestratorWsCeilingMb = 0,

    [Parameter(ParameterSetName = "Run")]
    [Parameter(ParameterSetName = "Orchestrate")]
    [Parameter(ParameterSetName = "DryRunGate")]
    [Parameter(ParameterSetName = "EmitSchedule")]
    [int]$Repeats = 3,

    # Per-cell wall-clock timeout (parent kills child tree). See synopsis derivation.
    # Citing session d5c98342-a0b2-41a9-b6e2-93ac7a39c3ba: OK max=616.8s; 2×→1234; use 1500.
    [Parameter(ParameterSetName = "Run")]
    [Parameter(ParameterSetName = "Orchestrate")]
    [Parameter(ParameterSetName = "DryRunGate")]
    [Parameter(ParameterSetName = "EmitSchedule")]
    [int]$CellTimeoutS = 1500,

    # Drift canary: 0 disables. N>0 → canary at start and after every N matrix cells.
    # N derivation for DISPATCH P is recorded in derived/kv_precision/DISPATCH_P_TIME_ESTIMATE.md
    # (last-good→first-bad onset in session 7f569929 / mean cell wall).
    [Parameter(ParameterSetName = "Run")]
    [Parameter(ParameterSetName = "Orchestrate")]
    [Parameter(ParameterSetName = "DryRunGate")]
    [Parameter(ParameterSetName = "EmitSchedule")]
    [int]$CanaryEveryN = 0,

    [Parameter(ParameterSetName = "Run")]
    [Parameter(ParameterSetName = "Orchestrate")]
    [Parameter(ParameterSetName = "DryRunGate")]
    [Parameter(ParameterSetName = "EmitSchedule")]
    [string]$CanaryArm = "gpu_only",

    [Parameter(ParameterSetName = "Run")]
    [Parameter(ParameterSetName = "Orchestrate")]
    [Parameter(ParameterSetName = "DryRunGate")]
    [Parameter(ParameterSetName = "EmitSchedule")]
    [int]$CanaryNCached = 4000,

    [Parameter(ParameterSetName = "Run")]
    [Parameter(ParameterSetName = "Orchestrate")]
    [Parameter(ParameterSetName = "DryRunGate")]
    [Parameter(ParameterSetName = "EmitSchedule")]
    [int]$CanaryDelta = 400,

    [Parameter(ParameterSetName = "Run")]
    [Parameter(ParameterSetName = "Orchestrate")]
    [Parameter(ParameterSetName = "DryRunGate")]
    [Parameter(ParameterSetName = "EmitSchedule")]
    [ValidateSet("RESIDENT", "NON_RESIDENT")]
    [string]$CanaryMode = "RESIDENT",

    # First K successful canaries establish ref + early relative variance; gate arms after.
    [Parameter(ParameterSetName = "Run")]
    [Parameter(ParameterSetName = "Orchestrate")]
    [Parameter(ParameterSetName = "DryRunGate")]
    [Parameter(ParameterSetName = "EmitSchedule")]
    [int]$CanaryCalibrationCount = 3,

    # Absolute relative-drift floor when early_max≈0 (same 0.05 spirit as C2f margin).
    [Parameter(ParameterSetName = "Run")]
    [Parameter(ParameterSetName = "Orchestrate")]
    [Parameter(ParameterSetName = "DryRunGate")]
    [Parameter(ParameterSetName = "EmitSchedule")]
    [double]$CanaryRelDriftFloor = 0.05,

    [Parameter(ParameterSetName = "Run")][switch]$SkipSettle,

    [Parameter(ParameterSetName = "Orchestrate")][switch]$Orchestrate,
    [Parameter(ParameterSetName = "Status")][switch]$Status,
    [Parameter(ParameterSetName = "VerifyDetach")][switch]$VerifyDetach,
    [Parameter(ParameterSetName = "DryRunGate")][switch]$DryRunGate,
    # Emit cell_specs + per-repeat Fisher-Yates order; no measurement, no isolation gate.
    [Parameter(ParameterSetName = "EmitSchedule")][switch]$EmitSchedule,
    [Parameter(ParameterSetName = "EmitSchedule")][string]$ScheduleOut = "",

    [Parameter(ParameterSetName = "Run")][switch]$DetachedWorker,
    [Parameter(ParameterSetName = "Run")][string]$SessionId,
    [Parameter(ParameterSetName = "Run")][switch]$HeartbeatOnly,
    [Parameter(ParameterSetName = "Run")][int]$HeartbeatSeconds = 30
)

$ErrorActionPreference = "Stop"
$root = "C:\Users\zjohn\Projects\gnn-hls-accel"
. (Join-Path $root "tools\SeamPsCommon.ps1")
$cfgPath = Join-Path $root "configs\delta_n.yaml"
$smokePy = Join-Path $root "tools\smoke_delta_prefill.py"
$sessionRoot = Join-Path $root "derived\delta_prefill"
$launchDir = Join-Path $sessionRoot "_launches"
$statePath = Join-Path $launchDir "launches.json"
try {
    $arms = [string[]](ConvertTo-StringList -Value $Arms -Name "Arms")
    $NCachedList = [int[]](ConvertTo-IntList -Value $NCached -Name "NCached")
    $DeltaList = [int[]](ConvertTo-IntList -Value $Deltas -Name "Deltas")
} catch {
    Write-Output $_.Exception.Message
    exit 2
}
# Resolve default model spec from delta_n.yaml (same source smoke_delta_prefill uses).
function Get-DefaultModelSpecPath {
    # Nested under openvino:; match the first model_spec assignment in the file.
    $line = Select-String -Path $cfgPath -Pattern '^\s*model_spec:\s*(.+)$' |
        Select-Object -First 1
    if ($null -eq $line) { return $null }
    $raw = $line.Matches[0].Groups[1].Value.Trim().Trim('"').Trim("'")
    if (-not $raw) { return $null }
    if ([System.IO.Path]::IsPathRooted($raw)) { return [string]$raw }
    return [string](Join-Path $root ($raw -replace '/', '\'))
}
$defaultModelSpec = Get-DefaultModelSpecPath
if (-not $defaultModelSpec) {
    Write-Output "REFUSED -- openvino.model_spec missing from $cfgPath"
    exit 2
}
if ($null -eq $ModelSpecs -or ([string]$ModelSpecs).Trim() -eq "") {
    $ModelSpecList = [string[]]@($defaultModelSpec)
} else {
    try {
        $ModelSpecList = [string[]](ConvertTo-StringList -Value $ModelSpecs -Name "ModelSpecs")
    } catch {
        Write-Output $_.Exception.Message
        exit 2
    }
}
# Normalize to absolute paths under $root when relative.
$ModelSpecList = @(
    foreach ($ms in $ModelSpecList) {
        if ([System.IO.Path]::IsPathRooted($ms)) { $ms }
        else { Join-Path $root ($ms -replace '/', '\') }
    }
)
if ([string]::IsNullOrWhiteSpace($CanaryModelSpec)) {
    $CanaryModelSpecResolved = $defaultModelSpec
} elseif ([System.IO.Path]::IsPathRooted($CanaryModelSpec)) {
    $CanaryModelSpecResolved = $CanaryModelSpec
} else {
    $CanaryModelSpecResolved = Join-Path $root ($CanaryModelSpec -replace '/', '\')
}
foreach ($ms in ($ModelSpecList + @($CanaryModelSpecResolved))) {
    if (-not (Test-Path -LiteralPath $ms)) {
        Write-Output "REFUSED -- model spec not found: $ms"
        exit 2
    }
}
$ModelSpecsCsv = ($ModelSpecList -join ",")
function Get-IrSha256FromSpec {
    param([Parameter(Mandatory = $true)][string]$SpecPath)
    $line = Select-String -Path $SpecPath -Pattern '^\s*ir_sha256:\s*([0-9a-fA-F]+)\s*$' |
        Select-Object -First 1
    if ($null -eq $line) { return $null }
    return [string]$line.Matches[0].Groups[1].Value.ToLowerInvariant()
}
$ModelSpecIrSha = @{}
foreach ($ms in ($ModelSpecList + @($CanaryModelSpecResolved))) {
    $sha = Get-IrSha256FromSpec -SpecPath $ms
    if (-not $sha) {
        Write-Output "REFUSED -- ir_sha256 missing in model spec: $ms"
        exit 2
    }
    $ModelSpecIrSha[$ms] = $sha
}
# Pipeline types: default single "llm" → byte-identical to pre-PipelineTypes runner.
$script:AllowedPipelineTypes = @("llm", "cb_no_eviction")
if ($null -eq $PipelineTypes -or ([string]$PipelineTypes).Trim() -eq "") {
    $PipelineTypeList = [string[]]@("llm")
} else {
    try {
        $PipelineTypeList = [string[]](ConvertTo-StringList -Value $PipelineTypes -Name "PipelineTypes")
    } catch {
        Write-Output $_.Exception.Message
        exit 2
    }
}
$PipelineTypeList = @(
    foreach ($pt in $PipelineTypeList) {
        $p = ([string]$pt).Trim().ToLowerInvariant()
        if ($script:AllowedPipelineTypes -notcontains $p) {
            Write-Output ("REFUSED -- -PipelineTypes entry must be one of {0} (got {1})" -f `
                ($script:AllowedPipelineTypes -join "|"), $pt)
            exit 2
        }
        $p
    }
)
if ($PipelineTypeList.Count -lt 1) {
    Write-Output "REFUSED -- PipelineTypes normalized to empty"
    exit 2
}
$CanaryPipelineTypeResolved = ([string]$CanaryPipelineType).Trim().ToLowerInvariant()
if ($script:AllowedPipelineTypes -notcontains $CanaryPipelineTypeResolved) {
    Write-Output ("REFUSED -- -CanaryPipelineType must be one of {0} (got {1})" -f `
        ($script:AllowedPipelineTypes -join "|"), $CanaryPipelineType)
    exit 2
}
$PipelineTypesCsv = ($PipelineTypeList -join ",")
if ($arms.Count -lt 1 -or $NCachedList.Count -lt 1 -or $DeltaList.Count -lt 1) {
    Write-Output "REFUSED -- Arms/NCached/Deltas normalized to empty"
    exit 2
}
foreach ($nc in $NCachedList) {
    if ($nc -lt 1) {
        Write-Output "REFUSED -- each -NCached entry must be >= 1 (got $nc)"
        exit 2
    }
}
foreach ($d in $DeltaList) {
    if ($d -lt 1) {
        Write-Output "REFUSED -- each -Deltas entry must be >= 1 (got $d)"
        exit 2
    }
}
# Max delta retained for three-number summary item 2 (gpu_only:A at largest delta).
# NON_RESIDENT cells are scheduled for every delta (DEFECT_non_resident_max_delta_only fix).
$MaxDelta = [int](($DeltaList | Measure-Object -Maximum).Maximum)
# Canonical CSV strings for Orchestrate inner cmdline (never pass Object[] through).
$ArmsCsv = ($arms -join ",")
$NCachedCsv = ($NCachedList -join ",")
$DeltasCsv = ($DeltaList -join ",")
$PipelineTypesCsv = ($PipelineTypeList -join ",")
if ($CellTimeoutS -lt 1) {
    Write-Output "REFUSED -- -CellTimeoutS must be >= 1 (got $CellTimeoutS)"
    exit 2
}
if ($CanaryEveryN -lt 0) {
    Write-Output "REFUSED -- -CanaryEveryN must be >= 0 (0=off; got $CanaryEveryN)"
    exit 2
}
if ($CanaryEveryN -gt 0) {
    if ($CanaryNCached -lt 1 -or $CanaryDelta -lt 1) {
        Write-Output "REFUSED -- canary n_cached/delta must be >= 1"
        exit 2
    }
    if ($CanaryCalibrationCount -lt 2) {
        Write-Output "REFUSED -- -CanaryCalibrationCount must be >= 2 to derive early variance (got $CanaryCalibrationCount)"
        exit 2
    }
    if ($CanaryRelDriftFloor -lt 0) {
        Write-Output "REFUSED -- -CanaryRelDriftFloor must be >= 0 (got $CanaryRelDriftFloor)"
        exit 2
    }
}

try {
    $py = [System.IO.Path]::GetFullPath($PythonExe)
} catch {
    $py = $PythonExe
}

function Get-LaunchEntriesFromNode {
    param([Parameter(Mandatory = $true)]$Node)
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
    try {
        $parsed = Get-Content -LiteralPath $statePath -Raw -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop
    } catch {
        Write-Output "REFUSED -- launches.json unreadable: $statePath"
        Write-Output "  $($_.Exception.Message)"
        exit 1
    }
    # ConvertTo-ObjectArray: @($genericList) throws ArgumentException; bare .ToArray()
    # fails when PowerShell unwraps a single-item List to PSCustomObject.
    return ConvertTo-ObjectArray (Get-LaunchEntriesFromNode -Node $parsed)
}

function Save-Launches {
    param([Parameter(Mandatory = $true)][object[]]$Entries)
    New-Item -ItemType Directory -Force -Path $launchDir | Out-Null
    $json = ConvertTo-Json -InputObject @($Entries) -Depth 6
    Set-Content -LiteralPath $statePath -Value $json -Encoding UTF8
}

function Assert-Interpreter {
    if (-not (Test-Path -LiteralPath $py)) {
        Write-Output "REFUSED -- INTERPRETER_MISSING"
        Write-Output "  pinned PythonExe does not exist: $py"
        exit 2
    }
}

# Keep in sync with seam/isolation.py TIER1 / TIER2 (case-insensitive; no .exe).
# Tier 1 REFUSE -- operator-controlled. Tier 2 RECORD only -- auto-respawning shell/vendor
# (CBS + WebExperience msedgewebview2 hosts; 5 s respawn verified 2026-08-09). Available
# floor stays 7000; tier 2 does not refuse.
$script:Tier1ContendingProcessNames = @(
    "Cursor", "Code", "chrome", "msedge",
    "firefox", "brave", "slack", "Discord", "Teams", "ms-teams", "Spotify", "OUTLOOK",
    "obsidian", "docker desktop", "vmmem", "claude"
)
$script:Tier2ContendingProcessNames = @(
    "msedgewebview2", "SearchHost", "Widgets", "WorkloadsSessionHost",
    "DellOptimizer.Systray", "SupportAssistAgent", "ICPS"
)
$script:ContendingProcessNames = $script:Tier1ContendingProcessNames

function Get-ProcessesByNameList {
    param([string[]]$Names)
    $wanted = @{}
    foreach ($n in $Names) { $wanted[$n.ToLowerInvariant()] = $true }
    return @(Get-Process -ErrorAction SilentlyContinue | Where-Object {
        $wanted.ContainsKey($_.ProcessName.ToLowerInvariant())
    })
}

function Get-ContendingProcesses {
    return @(Get-ProcessesByNameList -Names $script:Tier1ContendingProcessNames)
}

function Get-Tier2RecordedProcesses {
    return @(Get-ProcessesByNameList -Names $script:Tier2ContendingProcessNames)
}

function Write-ProcessGroupReport {
    param(
        [Parameter(Mandatory = $true)][object[]]$Processes,
        [string]$Prefix = "  -"
    )
    $Processes | Group-Object ProcessName | ForEach-Object {
        $privateMb = ($_.Group | Measure-Object -Property PrivateMemorySize64 -Sum).Sum / 1MB
        $pids = ($_.Group | Select-Object -ExpandProperty Id | Select-Object -First 5) -join ", "
        $more = if ($_.Count -gt 5) { ", ..." } else { "" }
        Write-Host ("{0} {1} x{2} private={3:N0} MiB (pids {4}{5})" -f `
            $Prefix, $_.Name, $_.Count, $privateMb, $pids, $more)
    }
}

function Write-ContendingReport {
    param([Parameter(Mandatory = $true)][object[]]$Contending)
    Write-Host "REFUSED -- tier-1 operator-controlled software is resident (private working set):"
    Write-ProcessGroupReport -Processes $Contending
    Write-Host ""
    Write-Host "Close them on the XPS, then relaunch over SSH (script waits pre_run_settle_s)."
    Write-Host "This script does not terminate those processes."
    Write-Host "Tier-2 shell/vendor agents (msedgewebview2, etc.) are recorded only and do not refuse."
}

function Write-Tier2RecordReport {
    param([Parameter(Mandatory = $true)][object[]]$Tier2)
    Write-Host "tier-2 (record only, do not refuse) -- auto-respawning shell/vendor:"
    Write-ProcessGroupReport -Processes $Tier2
}

function Assert-No-Contending {
    $contending = Get-ContendingProcesses
    if ($contending.Count -gt 0) {
        Write-ContendingReport -Contending $contending
        exit 1
    }
    $tier2 = Get-Tier2RecordedProcesses
    if ($tier2.Count -gt 0) {
        Write-Tier2RecordReport -Tier2 $tier2
    }
}

function Assert-CbNoEvictionTokenGate {
    param(
        [string]$LaunchCtx,
        [string]$ArmId,
        [string]$ModelSpecPath,
        [string]$OutPath = ""
    )
    # Hard gate: no matrix that includes cb_no_eviction may start unless this
    # 1-cell CB generate returns non-empty text (c4ddfd55: every CB cell OTHER).
    if ($PipelineTypeList -notcontains "cb_no_eviction") {
        return
    }
    Write-Output ""
    Write-Output "=== CB token gate (cb_no_eviction must generate non-empty text) ==="
    if (-not $OutPath) {
        $gateDir = if ($sessionDir) { $sessionDir } else { $sessionRoot }
        New-Item -ItemType Directory -Force -Path $gateDir | Out-Null
        $OutPath = Join-Path $gateDir "cb_token_gate.json"
    }
    $gateArgs = @(
        "-u", $smokePy,
        "--launch-context", $LaunchCtx,
        "--arm", $ArmId,
        "--model-spec", $ModelSpecPath,
        "--cb-token-gate",
        "--out", $OutPath
    )
    Write-Host ("  cmd: {0} {1}" -f $py, ($gateArgs -join " "))
    $child = Invoke-SeamChildProcess -FilePath $py -ArgumentList $gateArgs `
        -TimeoutSeconds $CellTimeoutS -WorkingDirectory $root
    $exit = [int]$child.exit_code
    $gateOk = $false
    $failReason = "exit_code=$exit"
    if (Test-Path -LiteralPath $OutPath) {
        try {
            $gateObj = Get-Content -LiteralPath $OutPath -Raw | ConvertFrom-Json
            $gateOk = [bool]$gateObj.ok
            if ($gateObj.fail_reason) { $failReason = [string]$gateObj.fail_reason }
        } catch {
            $failReason = "gate JSON parse failed: $_"
        }
    } else {
        $failReason = "gate artifact missing: $OutPath (exit=$exit)"
    }
    if (-not $gateOk -or $exit -ne 0) {
        Write-Output ("REFUSED -- CB token gate FAILED: {0}" -f $failReason)
        Write-Output "  No cb_no_eviction matrix cell will run until this gate passes."
        if ($null -ne $plan -and $sessionDir -and (Test-Path -LiteralPath (Join-Path $sessionDir "plan.json"))) {
            $plan.status = "refused_cb_token_gate"
            $plan.cb_token_gate = [ordered]@{
                ok          = $false
                fail_reason = $failReason
                artifact    = $OutPath
                exit_code   = $exit
            }
            $plan | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $sessionDir "plan.json") -Encoding utf8
        }
        exit 2
    }
    Write-Output ("CB token gate PASS (artifact={0})" -f $OutPath)
    if ($null -ne $plan -and $sessionDir -and (Test-Path -LiteralPath (Join-Path $sessionDir "plan.json"))) {
        $plan.cb_token_gate = [ordered]@{
            ok        = $true
            artifact  = $OutPath
            exit_code = $exit
        }
        $plan | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $sessionDir "plan.json") -Encoding utf8
    }
}

function Assert-PipelineRecordUnwrapGate {
    <#
    .SYNOPSIS
      Pre-matrix regression: settle-shaped PSCustomObject (WinPS 5.1 ConvertFrom-Json
      / [pscustomobject]@{} shape) must unwrap to a string-key IDictionary via the same
      Get-SinglePipelineRecord path used after Wait-InterCellAvailable and canaries.

    .DESCRIPTION
      N-1 died after canary 0 + settle OK: Get-SinglePipelineRecord refused a single
      PSCustomObject that lacked classification/turn1_prefill_s/tag (SeamPsCommon:77).
      W-2's 40 cells never hit that shape — cell records are [ordered], and the settle
      abort path that returns a settle record into Get-SinglePipelineRecord landed with
      N-1. -PipelineTypes did not change unwrap. Fail here before any cell runs.
    #>
    Write-Output ""
    Write-Output "=== pipeline-record unwrap gate (PSCustomObject -> OrderedDictionary) ==="
    # Exact failure shape: one PSCustomObject, settle keys only (no classification/tag).
    $settleShape = [pscustomobject]@{
        ok           = $true
        available_mb = 9500.0
        min_mb       = 7000.0
        max_wait_s   = 120.0
        probe_error  = $null
    }
    $jsonShape = ('{"ok":true,"available_mb":9500.0,"min_mb":7000.0,"max_wait_s":120.0,' +
        '"probe_error":null}') | ConvertFrom-Json
    $polluted = @(
        "-- CANARY arm=gpu_only n_cached=4000 mode=RESIDENT delta=400 r=0 @ unwrap-gate"
        [pscustomobject]@{
            classification  = "OK"
            turn1_prefill_s = 2.82
            turn2_prefill_s = 1.10
        }
    )
    try {
        $uSettle = Get-SinglePipelineRecord -InputObject $settleShape
        $uJson = Get-SinglePipelineRecord -InputObject $jsonShape
        $uPol = Get-SinglePipelineRecord -InputObject $polluted
    } catch {
        Write-Output ("REFUSED -- pipeline-record unwrap gate FAILED: {0}" -f $_.Exception.Message)
        Write-Output "  No matrix cell will run until Get-SinglePipelineRecord accepts PSCustomObject."
        exit 2
    }
    $ok = ($uSettle -is [System.Collections.IDictionary]) `
        -and ($uJson -is [System.Collections.IDictionary]) `
        -and ($uPol -is [System.Collections.IDictionary]) `
        -and ($true -eq $uSettle["ok"]) `
        -and ([double]$uSettle["available_mb"] -eq 9500.0) `
        -and ($true -eq $uJson["ok"]) `
        -and ($uPol["classification"] -eq "OK")
    # String-key write must succeed (the CANARY0 hazard).
    try {
        $uSettle["unwrap_gate"] = "ok"
        $uPol["rel_drift_t1"] = 0.0
    } catch {
        $ok = $false
        Write-Output ("REFUSED -- pipeline-record unwrap gate FAILED on indexer write: {0}" -f `
            $_.Exception.Message)
        exit 2
    }
    if (-not $ok) {
        Write-Output "REFUSED -- pipeline-record unwrap gate FAILED: normalised record shape mismatch"
        exit 2
    }
    Write-Output "pipeline-record unwrap gate PASS"
    if ($null -ne $plan -and $sessionDir -and (Test-Path -LiteralPath (Join-Path $sessionDir "plan.json"))) {
        $plan.pipeline_record_unwrap_gate = [ordered]@{
            ok     = $true
            note   = "PSCustomObject settle + ConvertFrom-Json + polluted canary -> IDictionary"
        }
        $plan | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $sessionDir "plan.json") -Encoding utf8
    }
}

function Get-AvailableMBytes {
    # Bounded probe: unbounded Get-Counter hang made inter_cell max_wait ineffective
    # (c4ddfd55: heartbeat frozen in settle for hours). Timeout => treat as unreadable.
    param([int]$TimeoutMs = 10000)
    $errorActionPreferencePrev = $ErrorActionPreference
    try {
        $ps = [System.Management.Automation.PowerShell]::Create()
        [void]$ps.AddScript({
            $ErrorActionPreference = "Stop"
            $sample = (Get-Counter '\Memory\Available MBytes' -ErrorAction Stop).CounterSamples[0]
            return [double]$sample.CookedValue
        })
        $async = $ps.BeginInvoke()
        if (-not $async.AsyncWaitHandle.WaitOne($TimeoutMs)) {
            try { $ps.Stop() } catch {}
            $ps.Dispose()
            return [pscustomobject]@{
                available_mb = $null
                method       = "Get-Counter_timeout_${TimeoutMs}ms"
                probe_error  = "Get-Counter did not return within ${TimeoutMs}ms"
            }
        }
        $vals = $ps.EndInvoke($async)
        $ps.Dispose()
        $mb = $null
        if ($vals -and $vals.Count -ge 1) { $mb = [double]$vals[0] }
        return [pscustomobject]@{
            available_mb = $mb
            method       = "Get-Counter:\\Memory\\Available MBytes"
            probe_error  = $null
        }
    } catch {
        return [pscustomobject]@{
            available_mb = $null
            method       = "Get-Counter_failed:$($_.Exception.Message)"
            probe_error  = $_.Exception.Message
        }
    } finally {
        $ErrorActionPreference = $errorActionPreferencePrev
    }
}

function Wait-InterCellAvailable {
    param(
        [double]$MinMb,
        [double]$SettleS,
        [double]$MaxWaitS,
        [double]$PollS = 5,
        [string]$SessionDir = "",
        [string]$Sid = "",
        [object]$CellIndex = $null,
        [string]$Arm = "",
        [string]$Mode = "",
        [object]$Delta = $null,
        [object]$Repeat = $null
    )
    # Fail-fast: if Available stays below MinMb through max_wait, ABORT.
    # Do not continue. Heartbeat ticks every poll so a stall is not a silent crash.
    Write-Host ("   inter-cell settle {0}s then Available >= {1} MB (max_wait={2}s; miss => settle_floor_not_met)..." -f `
        $SettleS, $MinMb, $MaxWaitS)

    function Write-SettleHb {
        param([string]$PhaseNote, [object]$AvailableMb, [object]$ElapsedS)
        if (-not $SessionDir) { return }
        Write-MatrixHeartbeat -SessionDir $SessionDir -Sid $Sid -Phase "inter_cell_settle" `
            -CellIndex $CellIndex -Arm $Arm -Mode $Mode -Delta $Delta -Repeat $Repeat `
            -Extra @{
                settle_note     = $PhaseNote
                available_mb    = $AvailableMb
                settle_elapsed_s = $ElapsedS
                min_mb          = $MinMb
                max_wait_s      = $MaxWaitS
            }
    }

    $tSettle0 = Get-Date
    $settled = 0.0
    while ($settled -lt $SettleS) {
        $chunk = [math]::Min($PollS, $SettleS - $settled)
        Start-Sleep -Seconds $chunk
        $settled += $chunk
        Write-SettleHb -PhaseNote "initial_settle" -AvailableMb $null `
            -ElapsedS ([math]::Round(((Get-Date) - $tSettle0).TotalSeconds, 1))
    }

    $deadline = (Get-Date).AddSeconds($MaxWaitS)
    $lastAvail = $null
    $lastProbeError = $null
    while ((Get-Date) -lt $deadline) {
        $avail = Get-AvailableMBytes
        $lastAvail = $avail.available_mb
        $lastProbeError = $avail.probe_error
        Write-SettleHb -PhaseNote "poll" -AvailableMb $lastAvail `
            -ElapsedS ([math]::Round(((Get-Date) - $tSettle0).TotalSeconds, 1))
        if ($null -ne $avail.available_mb -and [double]$avail.available_mb -ge $MinMb) {
            Write-Host ("   Available recovered: {0:N1} MB" -f $avail.available_mb)
            # OrderedDictionary (not [pscustomobject]): WinPS 5.1 has no
            # ConvertFrom-Json -AsHashtable; Get-SinglePipelineRecord normalises
            # PSCustomObject too, but settle should match cell-record shape.
            return [ordered]@{
                ok            = $true
                available_mb  = [double]$avail.available_mb
                min_mb        = $MinMb
                max_wait_s    = $MaxWaitS
                probe_error   = $null
            }
        }
        Start-Sleep -Seconds $PollS
    }
    $final = Get-AvailableMBytes
    if ($null -ne $final.available_mb) { $lastAvail = $final.available_mb }
    if ($final.probe_error) { $lastProbeError = $final.probe_error }
    Write-SettleHb -PhaseNote "floor_not_met" -AvailableMb $lastAvail `
        -ElapsedS ([math]::Round(((Get-Date) - $tSettle0).TotalSeconds, 1))
    Write-Host ("ABORT settle_floor_not_met: Available={0} MB (floor={1}; max_wait={2}s)" -f `
        $lastAvail, $MinMb, $MaxWaitS)
    return [ordered]@{
        ok           = $false
        available_mb = $lastAvail
        min_mb       = $MinMb
        max_wait_s   = $MaxWaitS
        probe_error  = $lastProbeError
        status       = "settle_floor_not_met"
    }
}

function Get-UptimeSeconds {
    # Seconds since LastBootUpTime. Additive cell field; does not feed the shuffle.
    try {
        $boot = (Get-CimInstance -ClassName Win32_OperatingSystem -ErrorAction Stop).LastBootUpTime
        return [double][math]::Round(((Get-Date) - [datetime]$boot).TotalSeconds, 3)
    } catch {
        return $null
    }
}

function Get-MachineDriftSnapshot {
    # Best-effort; null fields mean unreachable (do not invent).
    $avail = Get-AvailableMBytes
    $freq = $null
    $perf = $null
    $pkgC = $null
    try {
        $freq = [double](Get-Counter '\Processor Information(_Total)\Processor Frequency' -ErrorAction Stop).CounterSamples[0].CookedValue
    } catch { $freq = $null }
    try {
        $perf = [double](Get-Counter '\Processor Information(_Total)\% Processor Performance' -ErrorAction Stop).CounterSamples[0].CookedValue
    } catch { $perf = $null }
    try {
        $tz = @(Get-CimInstance -Namespace root\wmi -ClassName MSAcpi_ThermalZoneTemperature -ErrorAction Stop)
        if ($tz.Count -gt 0) {
            $pkgC = [double](($tz[0].CurrentTemperature / 10.0) - 273.15)
        }
    } catch { $pkgC = $null }
    return [ordered]@{
        available_mb              = $avail.available_mb
        available_method          = $avail.method
        processor_frequency_mhz   = $freq
        processor_performance_pct = $perf
        package_temp_c            = $pkgC
        package_temp_source       = $(if ($null -ne $pkgC) { "MSAcpi_ThermalZoneTemperature" } else { "unreachable" })
        snapshot_utc              = (Get-Date).ToUniversalTime().ToString("o")
    }
}

# Get-RelativeDrift / Get-MedianDouble / Get-SinglePipelineRecord /
# Update-CanaryDriftBookkeeping live in tools/SeamPsCommon.ps1.

function Assert-PreRunAvailable {
    param(
        [double]$MinMb,
        [string]$Citation,
        [switch]$ReportOnly
    )
    $avail = Get-AvailableMBytes
    $availText = if ($null -eq $avail.available_mb) { "null" } else { "{0:N1}" -f $avail.available_mb }
    Write-Host ("Available MBytes : {0} (method={1})" -f $availText, $avail.method)
    Write-Host ("pre_run floor    : {0} MB" -f $MinMb)
    if ($Citation) { Write-Host ("citation         : {0}" -f $Citation) }
    if ($null -eq $avail.available_mb) {
        Write-Host "REFUSED -- could not read \Memory\Available MBytes; cannot enforce pre-run gate."
        if (-not $ReportOnly) { exit 2 }
        return $false
    }
    if ([double]$avail.available_mb -lt $MinMb) {
        Write-Host ("REFUSED -- Available MBytes {0:N1} < pre_run_available_mb_min {1}." -f `
            $avail.available_mb, $MinMb)
        $contending = Get-ContendingProcesses
        if ($contending.Count -gt 0) {
            Write-ContendingReport -Contending $contending
        } else {
            Write-Host "No tier-1 operator-controlled process names are resident; other residents still hold memory."
        }
        $tier2 = Get-Tier2RecordedProcesses
        if ($tier2.Count -gt 0) {
            Write-Tier2RecordReport -Tier2 $tier2
        }
        if (-not $ReportOnly) { exit 1 }
        return $false
    }
    return $true
}

function Assert-LaunchContext-Honest {
    param(
        [string]$Context,
        [switch]$OrchestrateMode,
        [switch]$WorkerMode
    )
    if ($OrchestrateMode) {
        if ($Context -eq "ssh_foreground") {
            Write-Output "REFUSED -- -Orchestrate requires launch_context=ssh_detached; ssh_foreground rejected."
            exit 1
        }
        if ($Context -ne "ssh_detached") {
            Write-Output "REFUSED -- -Orchestrate requires launch_context=ssh_detached, got $Context"
            exit 1
        }
        return
    }
    if ($WorkerMode) {
        if ($Context -ne "ssh_detached") {
            Write-Output "REFUSED -- DetachedWorker must run with launch_context=ssh_detached, got $Context"
            exit 1
        }
        return
    }
    if ($Context -eq "ssh_foreground" -or $Context -eq "ssh_detached") {
        if (-not $env:SSH_CLIENT -and -not $env:SSH_CONNECTION) {
            Write-Output "REFUSED -- LaunchContext=$Context claimed but SSH_CLIENT/SSH_CONNECTION unset."
            Write-Output "Do not mislabel a local Cursor/console session as ssh_foreground/ssh_detached."
            Write-Output "For the real matrix: -Orchestrate (ssh_detached via spawn_detached)."
            exit 1
        }
    }
}

function Get-IsolationScalar {
    param(
        [string]$Path,
        [string]$Key
    )
    $inIsolation = $false
    foreach ($line in Get-Content -LiteralPath $Path) {
        if ($line -match '^\s*#') { continue }
        if ($line -match '^\s*isolation:\s*$') { $inIsolation = $true; continue }
        if ($inIsolation -and $line -match '^\S') { $inIsolation = $false }
        if ($inIsolation -and $line -match ("^\s*{0}:\s*(.+)$" -f [regex]::Escape($Key))) {
            $raw = $Matches[1].Trim()
            if ($raw -match '^"(.*)"\s*(#.*)?$') { return $Matches[1] }
            if ($raw -match "^'(.*)'\s*(#.*)?$") { return $Matches[1] }
            $token = ($raw -split '\s+#', 2)[0].Trim()
            if ($token -match '^(\S+)') { return $Matches[1] }
            return $token
        }
    }
    return $null
}

function Get-RecoveryScalar {
    param(
        [string]$Path,
        [string]$Key
    )
    $inRecovery = $false
    foreach ($line in Get-Content -LiteralPath $Path) {
        if ($line -match '^\s*#') { continue }
        if ($line -match '^\s*recovery:\s*$') { $inRecovery = $true; continue }
        if ($inRecovery -and $line -match '^\S') { $inRecovery = $false }
        if ($inRecovery -and $line -match ("^\s*{0}:\s*(.+)$" -f [regex]::Escape($Key))) {
            $raw = $Matches[1].Trim()
            $token = ($raw -split '\s+#', 2)[0].Trim()
            if ($token -match '^(\S+)') { return $Matches[1] }
            return $token
        }
    }
    return $null
}

function Get-YamlScalar {
    param(
        [string]$Path,
        [string]$Key
    )
    foreach ($line in Get-Content -LiteralPath $Path) {
        if ($line -match ("^\s*{0}:\s*(\S+)" -f [regex]::Escape($Key))) {
            return $Matches[1]
        }
    }
    return $null
}

function Write-MatrixHeartbeat {
    param(
        [Parameter(Mandatory = $true)][string]$SessionDir,
        [Parameter(Mandatory = $true)][string]$Sid,
        [string]$Phase,
        [object]$CellIndex = $null,
        [string]$Arm = $null,
        [string]$Mode = $null,
        [object]$Delta = $null,
        [object]$Repeat = $null,
        [hashtable]$Extra = $null
    )
    New-Item -ItemType Directory -Force -Path $SessionDir | Out-Null
    $hb = [ordered]@{
        session_id    = $Sid
        phase         = $Phase
        cell_index    = $CellIndex
        arm           = $Arm
        mode          = $Mode
        delta         = $Delta
        n_cached      = $null
        repeat        = $Repeat
        heartbeat_utc = (Get-Date).ToUniversalTime().ToString("o")
        pid           = $PID
    }
    if ($Extra) {
        foreach ($k in $Extra.Keys) { $hb[$k] = $Extra[$k] }
    }
    $path = Join-Path $SessionDir "heartbeat.json"
    $tmp = Join-Path $SessionDir "heartbeat.json.partial"
    $utf8 = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($tmp, ($hb | ConvertTo-Json -Depth 4), $utf8)
    Move-Item -LiteralPath $tmp -Destination $path -Force
}

function Get-CellSpecs {
    # RESIDENT and NON_RESIDENT at every -Deltas entry (one cell per arm/n_cached/mode/delta[/model][/pipeline]).
    # Prior design ran NON_RESIDENT only at max(-Deltas); that was a defect — see
    # derived/delta_prefill/DEFECT_non_resident_max_delta_only.md. Do not restore that rule.
    # ModelSpecs nested before modes; PipelineTypes nested innermost before modes so a
    # single default model+pipeline yields the same cell_specs order (and Fisher-Yates
    # sequence) as the pre-ModelSpecs / pre-PipelineTypes runner.
    $specs = New-Object System.Collections.Generic.List[object]
    foreach ($nc in $NCachedList) {
        foreach ($arm in $arms) {
            foreach ($d in $DeltaList) {
                foreach ($ms in $ModelSpecList) {
                    foreach ($pt in $PipelineTypeList) {
                        $specs.Add([pscustomobject]@{
                                arm = $arm
                                n_cached = [int]$nc
                                mode = "RESIDENT"
                                delta = [int]$d
                                model_spec = [string]$ms
                                ir_sha256 = [string]$ModelSpecIrSha[$ms]
                                pipeline_type = [string]$pt
                            }) | Out-Null
                        $specs.Add([pscustomobject]@{
                                arm = $arm
                                n_cached = [int]$nc
                                mode = "NON_RESIDENT"
                                delta = [int]$d
                                model_spec = [string]$ms
                                ir_sha256 = [string]$ModelSpecIrSha[$ms]
                                pipeline_type = [string]$pt
                            }) | Out-Null
                    }
                }
            }
        }
    }
    # ConvertTo-ObjectArray: @($specs) on List[object] throws ArgumentException before
    # plan write (session 1f519c47-c554-4939-a16a-63e120cfb254).
    return ConvertTo-ObjectArray $specs
}

function Get-OrchestratorWorkingSetBytes {
    return [int64](Get-Process -Id $PID).WorkingSet64
}

function Note-OrchestratorWorkingSetPeak {
    # Track peak across the session; written onto plan at end (ceiling unverifiable
    # from ws_at_start alone — 89f77871).
    param([int64]$WsBytes)
    if ($null -eq $script:orchestratorWsPeakBytes) {
        $script:orchestratorWsPeakBytes = [int64]0
    }
    if ($WsBytes -gt [int64]$script:orchestratorWsPeakBytes) {
        $script:orchestratorWsPeakBytes = $WsBytes
    }
}

function Assert-OrchestratorWorkingSet {
    param(
        [Parameter(Mandatory = $true)][int64]$CeilingBytes,
        [Parameter(Mandatory = $true)][string]$AtStep,
        [string]$SessionDir = ""
    )
    if ($CeilingBytes -le 0) {
        Write-Output -NoEnumerate $true
        return
    }
    $ws = Get-OrchestratorWorkingSetBytes
    Note-OrchestratorWorkingSetPeak -WsBytes $ws
    if ($ws -le $CeilingBytes) {
        Write-Output -NoEnumerate $true
        return
    }
    $status = "aborted_orchestrator_ws_ceiling"
    $rec = [ordered]@{
        status            = $status
        at_step           = $AtStep
        working_set_bytes = $ws
        ceiling_bytes     = $CeilingBytes
        working_set_mb    = [math]::Round($ws / 1MB, 1)
        ceiling_mb        = [math]::Round($CeilingBytes / 1MB, 1)
        utc               = (Get-Date).ToUniversalTime().ToString("o")
        note              = (
            "Orchestrator working set exceeded recorded ceiling. An orchestrator " +
            "that consumes the memory its cells are gated on defeats the Available " +
            "floor. Aborting before further cells."
        )
    }
    Write-Host ("ABORT $status at={0} ws_mb={1:N1} ceiling_mb={2:N1}" -f `
        $AtStep, ($ws / 1MB), ($CeilingBytes / 1MB))
    if ($SessionDir) {
        $abortPath = Join-Path $SessionDir "orchestrator_ws_abort.json"
        $json = ConvertTo-Json -InputObject $rec -Depth 6
        $utf8 = New-Object System.Text.UTF8Encoding $false
        [System.IO.File]::WriteAllText($abortPath, ($json + "`n"), $utf8)
    }
    Write-Output -NoEnumerate $false
}

function Start-WshWatchdog {
    param(
        [string]$KillLogPath,
        [int]$IntervalS,
        [string]$SessionDir
    )
    if ($IntervalS -le 0) { return $null }
    $t0 = (Get-Date).ToUniversalTime().ToString("o")
    $initial = @(Get-Process -Name "WorkloadsSessionHost" -ErrorAction SilentlyContinue)
    $initRecord = [ordered]@{
        utc        = $t0
        event      = "watchdog_start_kill"
        interval_s = $IntervalS
        pids_found = @($initial | ForEach-Object { [int]$_.Id })
        n_killed   = 0
    }
    if ($initial.Count -gt 0) {
        $initial | Stop-Process -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 1
        $left = @(Get-Process -Name "WorkloadsSessionHost" -ErrorAction SilentlyContinue)
        $initRecord.n_killed = $initial.Count - $left.Count
        $initRecord.pids_remaining = @($left | ForEach-Object { [int]$_.Id })
    }
    Add-Content -LiteralPath $KillLogPath -Value (
        (ConvertTo-Json -InputObject $initRecord -Compress -Depth 4))

    # Sibling powershell.exe — not Start-Job (job buffers grow; null IntervalS was a
    # tight Get-Process loop). Owns its own memory; only appends small JSONL lines.
    $watchScript = Join-Path $SessionDir "_wsh_watchdog.ps1"
    $watchBody = @'
param([string]$LogPath, [int]$IntervalS)
$ErrorActionPreference = "Continue"
if ($IntervalS -lt 30) { $IntervalS = 30 }
while ($true) {
    Start-Sleep -Seconds $IntervalS
    $utc = (Get-Date).ToUniversalTime().ToString("o")
    $procs = @(Get-Process -Name "WorkloadsSessionHost" -ErrorAction SilentlyContinue)
    if ($procs.Count -eq 0) {
        $rec = [ordered]@{ utc = $utc; event = "watchdog_poll"; n_found = 0; n_killed = 0 }
        Add-Content -LiteralPath $LogPath -Value (ConvertTo-Json -InputObject $rec -Compress)
        continue
    }
    $pids = @($procs | ForEach-Object { [int]$_.Id })
    $procs | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
    $left = @(Get-Process -Name "WorkloadsSessionHost" -ErrorAction SilentlyContinue)
    $rec = [ordered]@{
        utc            = $utc
        event          = "watchdog_kill"
        n_found        = $pids.Count
        pids_found     = $pids
        n_killed       = $pids.Count - $left.Count
        pids_remaining = @($left | ForEach-Object { [int]$_.Id })
    }
    Add-Content -LiteralPath $LogPath -Value (ConvertTo-Json -InputObject $rec -Compress -Depth 4)
}
'@
    Set-Content -LiteralPath $watchScript -Value $watchBody -Encoding UTF8
    $watchProc = Start-Process -FilePath "powershell.exe" -PassThru -WindowStyle Hidden `
        -ArgumentList @(
            "-NoProfile", "-NoLogo", "-ExecutionPolicy", "Bypass",
            "-File", $watchScript,
            "-LogPath", $KillLogPath,
            "-IntervalS", "$IntervalS"
        )
    return [pscustomobject]@{
        kind       = "process"
        pid        = $watchProc.Id
        script     = $watchScript
        interval_s = $IntervalS
    }
}

function Stop-WshWatchdog {
    param($Handle)
    if ($null -eq $Handle) { return }
    if ($Handle.pid) {
        try {
            $p = Get-Process -Id ([int]$Handle.pid) -ErrorAction SilentlyContinue
            if ($p) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
        } catch {}
    }
}

function Invoke-DetachedSpawn {
    param(
        [Parameter(Mandatory = $true)][string]$InnerCommand,
        [Parameter(Mandatory = $true)][string]$LogPath,
        [Parameter(Mandatory = $true)][string]$TagName,
        [Parameter(Mandatory = $true)][string]$LaunchCtx,
        [string]$Sid = $null,
        [string]$ResultPath = $null,
        [bool]$VerifyOnly = $false
    )
    $json = & (Join-Path $root "tools\spawn_detached.ps1") `
        -CommandLine $InnerCommand -LogPath $LogPath -WorkingDirectory $root
    $info = $json | ConvertFrom-Json
    $entry = [pscustomobject]@{
        tag             = $TagName
        pid             = $info.pid
        parent_name     = $info.parent_name
        ppid            = $info.ppid
        isolation_mode  = "remote"
        launch_context  = $LaunchCtx
        session_id      = $Sid
        matrix_tag      = $Tag
        n_cached        = $NCachedCsv
        deltas          = $DeltasCsv
        arms            = $ArmsCsv
        cell_timeout_s  = $CellTimeoutS
        verify_detach   = $VerifyOnly
        launched_utc    = (Get-Date).ToUniversalTime().ToString("o")
        log_path        = $LogPath
        result_path     = $ResultPath
    }
    Save-Launches -Entries (@(Get-Launches) + $entry)
    return $info
}

function Write-ThreeNumbers {
    param([System.Collections.IEnumerable]$Cells)
    $rows = @($Cells | Where-Object { $_.classification -eq "OK" -and $null -ne $_.turn2_prefill_s })
    Write-Output ""
    Write-Output "=== THREE NUMBERS ==="

    function Median([double[]]$vals) {
        if (-not $vals -or $vals.Count -eq 0) { return $null }
        $s = @($vals | Sort-Object)
        $n = $s.Count
        if ($n % 2 -eq 1) { return $s[[int]($n / 2)] }
        return ($s[$n / 2 - 1] + $s[$n / 2]) / 2.0
    }

    foreach ($nc in $NCachedList) {
    foreach ($arm in $arms) {
        foreach ($cmpDelta in $DeltaList) {
            $dCmp = [int]$cmpDelta
            $res = @($rows | Where-Object {
                    $_.arm -eq $arm -and $_.mode -eq "RESIDENT" -and [int]$_.delta -eq $dCmp -and
                    ([int]$_.n_cached -eq [int]$nc)
                } | ForEach-Object { [double]$_.turn2_prefill_s })
            $non = @($rows | Where-Object {
                    $_.arm -eq $arm -and $_.mode -eq "NON_RESIDENT" -and [int]$_.delta -eq $dCmp -and
                    ([int]$_.n_cached -eq [int]$nc)
                } | ForEach-Object { [double]$_.turn2_prefill_s })
            $medR = Median $res
            $medN = Median $non
            Write-Output ("1. arm={0} n_cached={1} delta={2} turn2_prefill_s RESIDENT median={3} (n={4}) vs NON_RESIDENT median={5} (n={6})" -f `
                $arm, $nc, $dCmp,
                $(if ($null -eq $medR) { "null" } else { "{0:N3}" -f $medR }),
                $res.Count,
                $(if ($null -eq $medN) { "null" } else { "{0:N3}" -f $medN }),
                $non.Count)
            if ($null -ne $medR -and $null -ne $medN -and $medN -ne 0) {
                $ratio = $medR / $medN
                $falsified = $ratio -ge 0.5
                Write-Output ("   ratio RESIDENT/NON_RESIDENT={0:N3} falsified_kv_not_retained={1}" -f $ratio, $falsified)
            }
        }
    }
    }

    $cmpDelta = [int]$MaxDelta
    $aRes = @($rows | Where-Object { $_.arm -eq "A" -and $_.mode -eq "RESIDENT" -and [int]$_.delta -eq $cmpDelta } |
        ForEach-Object { [double]$_.turn2_prefill_s })
    $gRes = @($rows | Where-Object { $_.arm -eq "gpu_only" -and $_.mode -eq "RESIDENT" -and [int]$_.delta -eq $cmpDelta } |
        ForEach-Object { [double]$_.turn2_prefill_s })
    $medA = Median $aRes
    $medG = Median $gRes
    Write-Output ("2. turn2_prefill_s RESIDENT delta={0} gpu_only:A = {1}" -f `
        $cmpDelta,
        $(if ($null -eq $medA -or $null -eq $medG -or $medA -eq 0) { "null" } else { "{0:N3}" -f ($medG / $medA) }))
    Write-Output ("   (gpu_only median={0} A median={1})" -f `
        $(if ($null -eq $medG) { "null" } else { "{0:N3}" -f $medG }),
        $(if ($null -eq $medA) { "null" } else { "{0:N3}" -f $medA }))

    foreach ($nc in $NCachedList) {
    foreach ($arm in $arms) {
        $parts = New-Object System.Collections.Generic.List[string]
        $medByDelta = @{}
        foreach ($d in $DeltaList) {
            $vals = @($rows | Where-Object {
                    $_.arm -eq $arm -and $_.mode -eq "RESIDENT" -and [int]$_.delta -eq [int]$d -and
                    ([int]$_.n_cached -eq [int]$nc)
                } | ForEach-Object { [double]$_.turn2_prefill_s })
            $med = Median $vals
            $medByDelta[[int]$d] = $med
            $parts.Add(("delta{0} median={1}" -f $d, $(if ($null -eq $med) { "null" } else { "{0:N3}" -f $med }))) | Out-Null
        }
        Write-Output ("3. arm={0} n_cached={1} RESIDENT turn2_prefill_s {2}" -f $arm, $nc, ($parts -join " "))
        if ($DeltaList.Count -ge 2) {
            $dLo = [int]($DeltaList | Measure-Object -Minimum).Minimum
            $dHi = [int]($DeltaList | Measure-Object -Maximum).Maximum
            $mLo = $medByDelta[$dLo]
            $mHi = $medByDelta[$dHi]
            if ($null -ne $mLo -and $null -ne $mHi -and $mLo -gt 0) {
                $scale = $mHi / $mLo
                $ideal = $dHi / [double]$dLo
                Write-Output ("   scale {0}/{1}={2:N3} (delta-like~{3:N1}; n_cached-dominated~1)" -f `
                    $dHi, $dLo, $scale, $ideal)
            }
        }
    }
    }
}

# ----------------------------------------------------------------------------------------------
if ($Status) {
    # Fixed pattern: job timeout, heartbeat-dir discovery, bounded FileShare.ReadWrite reads,
    # no Select-String against the live spawn log.
    $statusJob = Start-Job -ScriptBlock {
        param($root, $statePath, $sessionRoot)
        $ErrorActionPreference = "Stop"
        . (Join-Path $root "tools\SeamPsCommon.ps1")

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

        function Read-SharedTextFile {
            param([string]$Path, [int]$MaxBytes = 65536, [switch]$Tail)
            if (-not (Test-Path -LiteralPath $Path)) { return $null }
            $fs = $null
            try {
                $fs = [System.IO.File]::Open(
                    $Path,
                    [System.IO.FileMode]::Open,
                    [System.IO.FileAccess]::Read,
                    [System.IO.FileShare]::ReadWrite
                )
                if ($Tail -and $fs.Length -gt $MaxBytes) {
                    $fs.Seek(-[int64]$MaxBytes, [System.IO.SeekOrigin]::End) | Out-Null
                }
                $toRead = [int][Math]::Min([int64]$MaxBytes, $fs.Length - $fs.Position)
                if (-not $Tail) {
                    $toRead = [int][Math]::Min([int64]$MaxBytes, $fs.Length)
                    $fs.Seek(0, [System.IO.SeekOrigin]::Begin) | Out-Null
                }
                $buf = New-Object byte[] $toRead
                $read = $fs.Read($buf, 0, $toRead)
                return [System.Text.Encoding]::UTF8.GetString($buf, 0, $read)
            } catch {
                return $null
            } finally {
                if ($fs) { $fs.Dispose() }
            }
        }

        $lines = New-Object System.Collections.Generic.List[string]
        if (-not (Test-Path -LiteralPath $statePath)) {
            $lines.Add("no delta_prefill run has been launched") | Out-Null
            return @{ exit = 1; lines = $lines }
        }
        $parsed = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
        $launches = ConvertTo-ObjectArray (Get-LaunchEntriesFromNode -Node $parsed)
        if ($launches.Count -eq 0) {
            $lines.Add("no delta_prefill run has been launched") | Out-Null
            return @{ exit = 1; lines = $lines }
        }
        $last = $launches[-1]
        $alive = $null -ne (Get-Process -Id ([int]$last.pid) -ErrorAction SilentlyContinue)
        $lines.Add("tag            : $($last.tag)") | Out-Null
        $lines.Add("launched       : $($last.launched_utc)") | Out-Null
        $lines.Add("declared       : isolation_mode=$($last.isolation_mode) launch_context=$($last.launch_context)") | Out-Null
        if ($last.matrix_tag) { $lines.Add("matrix_tag     : $($last.matrix_tag)") | Out-Null }
        if ($last.n_cached) { $lines.Add("n_cached       : $($last.n_cached)") | Out-Null }
        if ($last.verify_detach) { $lines.Add("verify_detach  : True") | Out-Null }
        $lines.Add("process        : pid $($last.pid), alive=$alive") | Out-Null
        if ($last.parent_name) {
            $lines.Add("parent         : $($last.parent_name) (ppid $($last.ppid))") | Out-Null
        }

        $sid = $last.session_id
        if (-not $sid) {
            $newest = Get-ChildItem $sessionRoot -Directory -ErrorAction SilentlyContinue |
                      Where-Object { $_.Name -match '^[0-9a-f-]{36}$' } |
                      Sort-Object LastWriteTime -Descending |
                      Select-Object -First 1
            if ($newest) { $sid = $newest.Name }
        }
        if (-not $sid -and $last.log_path) {
            $head = Read-SharedTextFile -Path ([string]$last.log_path) -MaxBytes 65536
            if ($head -match '"session_id"\s*:\s*"([0-9a-f-]{36})"') {
                $sid = $Matches[1]
            } elseif ($head -match 'session_id\s*:\s*([0-9a-f-]{36})') {
                $sid = $Matches[1]
            }
        }
        if ($sid) { $lines.Add("session_id     : $sid") | Out-Null }
        else { $lines.Add("session_id     : (not yet known)") | Out-Null }

        if ($sid) {
            $hb = Join-Path $sessionRoot ("{0}\heartbeat.json" -f $sid)
            $plan = Join-Path $sessionRoot ("{0}\plan.json" -f $sid)
            $hbText = Read-SharedTextFile -Path $hb -MaxBytes 16384
            if ($hbText) {
                $lines.Add("") | Out-Null
                $lines.Add("--- heartbeat ---") | Out-Null
                $lines.Add($hbText.TrimEnd()) | Out-Null
            } elseif (Test-Path -LiteralPath $plan) {
                $lines.Add("plan exists; heartbeat not yet readable") | Out-Null
            }
            $planText = Read-SharedTextFile -Path $plan -MaxBytes 65536
            if ($planText -and $planText -match '"status"\s*:\s*"(complete|verify_detach_complete|FAIL|FAIL_CANARY_DRIFT)"') {
                $lines.Add("") | Out-Null
                $lines.Add("state          : COMPLETE ($($Matches[1]))") | Out-Null
                return @{ exit = 0; lines = $lines }
            }
        }

        if ($last.result_path) {
            $launchResult = Read-SharedTextFile -Path ([string]$last.result_path) -MaxBytes 65536
            if ($launchResult) {
                $lines.Add("state          : COMPLETE") | Out-Null
                $lines.Add("") | Out-Null
                $lines.Add($launchResult.TrimEnd()) | Out-Null
                return @{ exit = 0; lines = $lines }
            }
        }
        if ($alive) {
            $lines.Add("state          : RUNNING -- do not open a session on this machine") | Out-Null
        } else {
            $lines.Add("state          : ENDED WITHOUT A RESULT -- the log below is the evidence") | Out-Null
        }
        $lines.Add("") | Out-Null
        $lines.Add("--- last ~8 KiB of $($last.log_path) ---") | Out-Null
        if ($last.log_path) {
            $tail = Read-SharedTextFile -Path ([string]$last.log_path) -MaxBytes 8192 -Tail
            if ($null -eq $tail) {
                $lines.Add("(log unreadable with FileShare.ReadWrite; not waiting on runner lock)") | Out-Null
            } else {
                $tailLines = $tail -split "`r?`n"
                if ($tailLines.Count -gt 1) { $tailLines = $tailLines[1..($tailLines.Count - 1)] }
                foreach ($tl in $tailLines) { $lines.Add($tl) | Out-Null }
            }
        }
        return @{ exit = $(if ($alive) { 0 } else { 1 }); lines = $lines }
    } -ArgumentList $root, $statePath, $sessionRoot

    $finished = Wait-Job -Job $statusJob -Timeout 15
    if (-not $finished) {
        Stop-Job $statusJob -ErrorAction SilentlyContinue
        Remove-Job $statusJob -Force -ErrorAction SilentlyContinue
        Write-Output "REFUSED -- -Status timed out after 15s (non-blocking by design)"
        Write-Output "  Prefer: Get-Content derived/delta_prefill/<session_id>/heartbeat.json"
        Write-Output "  Do not Select-String / Get-Content -Wait the live launch log while the runner holds it."
        exit 2
    }
    $payload = Receive-Job $statusJob
    Remove-Job $statusJob -Force -ErrorAction SilentlyContinue
    foreach ($line in @($payload.lines)) { Write-Output $line }
    exit ([int]$payload.exit)
}

# ----------------------------------------------------------------------------------------------
Assert-Interpreter

if ($VerifyDetach) {
    $launchContext = "ssh_detached"
    New-Item -ItemType Directory -Force -Path $launchDir | Out-Null
    $sid = [guid]::NewGuid().ToString()
    $tagLaunch = "delta_prefill_verify_" + (Get-Date -Format "yyyyMMdd_HHmmss")
    $log = Join-Path $launchDir "$tagLaunch.log"
    $resultPath = Join-Path $launchDir "$tagLaunch.result.json"
    $self = Join-Path $root "tools\run_delta_prefill_matrix.ps1"
    $inner = 'set SEAM_LAUNCH_CONTEXT=ssh_detached' +
             '&& powershell -NoProfile -File "' + $self + '"' +
             ' -DetachedWorker -HeartbeatOnly -HeartbeatSeconds 30' +
             ' -LaunchContext ssh_detached -SessionId ' + $sid +
             ' -Tag "' + $Tag + '"' +
             ' -PythonExe "' + $py + '"'
    $info = Invoke-DetachedSpawn -InnerCommand $inner -LogPath $log -TagName $tagLaunch `
        -LaunchCtx $launchContext -Sid $sid -ResultPath $resultPath -VerifyOnly $true
    Write-Output "launched detached delta_prefill VERIFY (heartbeat-only)"
    Write-Output "  tag            : $tagLaunch"
    Write-Output "  pid            : $($info.pid) (parent $($info.parent_name) -- not this session)"
    Write-Output "  launch_context : $launchContext"
    Write-Output "  session_id     : $sid"
    Write-Output "  log            : $log"
    Write-Output ""
    Write-Output "Poll with:"
    Write-Output "  powershell -NoProfile -File tools/run_delta_prefill_matrix.ps1 -Status"
    exit 0
}

if ($DryRunGate) {
    $preRunMin = Get-IsolationScalar -Path $cfgPath -Key "pre_run_available_mb_min"
    $preRunCite = Get-IsolationScalar -Path $cfgPath -Key "pre_run_available_mb_min_citation"
    if (-not $preRunMin) {
        Write-Output "REFUSED -- isolation.pre_run_available_mb_min missing from $cfgPath"
        exit 2
    }
    Write-Output "dry-run cleanliness gate (no matrix)"
    Write-Output ("  arms             : {0}" -f ($arms -join ","))
    Write-Output ("  model_specs      : {0}" -f ($ModelSpecList -join ","))
    Write-Output ("  canary_model_spec: {0}" -f $CanaryModelSpecResolved)
    Write-Output ("  pipeline_types   : {0}" -f ($PipelineTypeList -join ","))
    Write-Output ("  canary_pipeline  : {0}" -f $CanaryPipelineTypeResolved)
    Write-Output ("  n_cached         : {0}" -f ($NCachedList -join ","))
    Write-Output ("  deltas           : {0}" -f ($DeltaList -join ","))
    Write-Output ("  max_delta        : {0} (summary item 2; NON_RESIDENT runs every -Deltas entry)" -f $MaxDelta)
    Write-Output ("  cell_timeout_s   : {0}" -f $CellTimeoutS)
    Write-Output ("  cell_timeout_derivation: session d5c98342 OK max elapsed_s=616.8; 2x=1234; rounded to 1500")
    Write-Output ("  cells/round      : {0}" -f (Get-CellSpecs).Count)
    Write-Output ("  canary_every_n   : {0}" -f $CanaryEveryN)
    Write-Output ("  watchdog_interval_s: {0}" -f $WatchdogIntervalS)
    Write-Output ("  orchestrator_ws_ceiling_mb: {0}" -f $OrchestratorWsCeilingMb)
    if ($CanaryEveryN -gt 0) {
        Write-Output ("  canary_config    : arm={0} nc={1} d={2} mode={3} model={4} pipeline={5} calib={6} floor={7}" -f `
            $CanaryArm, $CanaryNCached, $CanaryDelta, $CanaryMode, $CanaryModelSpecResolved, `
            $CanaryPipelineTypeResolved, $CanaryCalibrationCount, $CanaryRelDriftFloor)
    }
    Write-Output ("  tier1 refuse names: {0}" -f ($script:Tier1ContendingProcessNames -join ", "))
    Write-Output ("  tier2 record names: {0}" -f ($script:Tier2ContendingProcessNames -join ", "))
    $contending = Get-ContendingProcesses
    if ($contending.Count -gt 0) {
        Write-ContendingReport -Contending $contending
    } else {
        Write-Output "tier1 refuse list: none resident"
    }
    $tier2 = Get-Tier2RecordedProcesses
    if ($tier2.Count -gt 0) {
        Write-Tier2RecordReport -Tier2 $tier2
    } else {
        Write-Output "tier2 record list: none resident"
    }
    $ok = Assert-PreRunAvailable -MinMb ([double]$preRunMin) -Citation $preRunCite -ReportOnly
    if ($contending.Count -gt 0 -or -not $ok) {
        Write-Output "dry-run FAIL -- close tier-1 software / reclaim memory; no matrix started"
        exit 1
    }
    Write-Output "dry-run PASS -- Available above floor and no tier-1 residents (tier-2 recorded only)"
    exit 0
}

if ($EmitSchedule) {
    $baseSeedEmit = Get-YamlScalar -Path $cfgPath -Key "randomization_seed"
    if (-not $baseSeedEmit) { $baseSeedEmit = "20260805" }
    $cellSpecsEmit = Get-CellSpecs
    $rounds = New-Object System.Collections.Generic.List[object]
    for ($r = 0; $r -lt $Repeats; $r++) {
        $seed = [int]$baseSeedEmit + (10007 * $r) + ($NCachedList.Count * 17) + ($DeltaList.Count * 3)
        $rng = [System.Random]::new($seed)
        $order = @($cellSpecsEmit)
        for ($i = $order.Count - 1; $i -gt 0; $i--) {
            $j = $rng.Next(0, $i + 1)
            $tmp = $order[$i]
            $order[$i] = $order[$j]
            $order[$j] = $tmp
        }
        $rounds.Add([ordered]@{
                repeat = $r
                seed   = $seed
                order  = @(
                    $order | ForEach-Object {
                        [ordered]@{
                            arm           = $_.arm
                            n_cached      = [int]$_.n_cached
                            mode          = $_.mode
                            delta         = [int]$_.delta
                            model_spec    = [string]$_.model_spec
                            ir_sha256     = [string]$_.ir_sha256
                            pipeline_type = [string]$_.pipeline_type
                        }
                    }
                )
            }) | Out-Null
    }
    $sched = [ordered]@{
        kind                 = "delta_prefill_matrix_schedule"
        arms                 = $arms
        n_cached             = @($NCachedList | ForEach-Object { [int]$_ })
        deltas               = @($DeltaList | ForEach-Object { [int]$_ })
        repeats              = $Repeats
        randomization_seed   = [int]$baseSeedEmit
        model_specs          = @($ModelSpecList)
        canary_model_spec    = $CanaryModelSpecResolved
        pipeline_types       = @($PipelineTypeList)
        canary_pipeline_type = $CanaryPipelineTypeResolved
        cell_specs_per_round = @(
            $cellSpecsEmit | ForEach-Object {
                [ordered]@{
                    arm = $_.arm; n_cached = [int]$_.n_cached; mode = $_.mode; delta = [int]$_.delta
                    model_spec = [string]$_.model_spec; ir_sha256 = [string]$_.ir_sha256
                    pipeline_type = [string]$_.pipeline_type
                }
            }
        )
        rounds               = @(
            foreach ($rd in $rounds) { $rd }
        )
    }
    if (-not [string]::IsNullOrWhiteSpace($PreRunAmendmentFile)) {
        $amendPathEmit = $PreRunAmendmentFile
        if (-not [System.IO.Path]::IsPathRooted($amendPathEmit)) {
            $amendPathEmit = Join-Path $root ($amendPathEmit -replace '/', '\')
        }
        if (-not (Test-Path -LiteralPath $amendPathEmit)) {
            Write-Output "REFUSED -- -PreRunAmendmentFile not found: $amendPathEmit"
            exit 1
        }
        $amendTextEmit = [System.IO.File]::ReadAllText($amendPathEmit)
        if ([string]::IsNullOrWhiteSpace($amendTextEmit)) {
            Write-Output "REFUSED -- -PreRunAmendmentFile is empty: $amendPathEmit"
            exit 1
        }
        $sched["pre_run_amendment"] = [ordered]@{
            source_path  = $amendPathEmit
            text         = $amendTextEmit
            note         = "EmitSchedule preview of plan.pre_run_amendment (same copy path as live run)."
        }
    }
    $outPath = if ($ScheduleOut) { $ScheduleOut } else {
        Join-Path $sessionRoot ("{0}_schedule_emit.json" -f $Tag)
    }
    if (-not [System.IO.Path]::IsPathRooted($outPath)) {
        $outPath = Join-Path $root $outPath
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $outPath) | Out-Null
    $sched | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $outPath -Encoding utf8
    Write-Output ("EMIT_SCHEDULE wrote {0}" -f $outPath)
    Write-Output ("  cells_per_round={0} repeats={1} model_specs={2} pipeline_types={3}" -f `
        $cellSpecsEmit.Count, $Repeats, ($ModelSpecList -join ","), ($PipelineTypeList -join ","))
    exit 0
}

if ($Orchestrate) {
    if ($PSBoundParameters.ContainsKey("LaunchContext") -and $LaunchContext -ne "ssh_detached") {
        Assert-LaunchContext-Honest -Context $LaunchContext -OrchestrateMode
    }
    $launchContext = "ssh_detached"
    Assert-LaunchContext-Honest -Context $launchContext -OrchestrateMode
    Assert-No-Contending
    $preRunMinOrch = Get-IsolationScalar -Path $cfgPath -Key "pre_run_available_mb_min"
    $preRunCiteOrch = Get-IsolationScalar -Path $cfgPath -Key "pre_run_available_mb_min_citation"
    if (-not $preRunMinOrch) {
        Write-Output "REFUSED -- isolation.pre_run_available_mb_min missing from $cfgPath"
        exit 2
    }
    Assert-PreRunAvailable -MinMb ([double]$preRunMinOrch) -Citation $preRunCiteOrch | Out-Null

    New-Item -ItemType Directory -Force -Path $launchDir | Out-Null

    # CB token gate before detach: fail here rather than in the orphaned worker.
    Assert-CbNoEvictionTokenGate -LaunchCtx "ssh_detached" -ArmId $arms[0] `
        -ModelSpecPath $CanaryModelSpecResolved `
        -OutPath (Join-Path $launchDir ("cb_token_gate_{0}.json" -f (Get-Date -Format "yyyyMMdd_HHmmss")))
    Assert-PipelineRecordUnwrapGate

    $sid = [guid]::NewGuid().ToString()
    $tagLaunch = "delta_prefill_" + (Get-Date -Format "yyyyMMdd_HHmmss")
    $log = Join-Path $launchDir "$tagLaunch.log"
    $resultPath = Join-Path $launchDir "$tagLaunch.result.json"
    $self = Join-Path $root "tools\run_delta_prefill_matrix.ps1"
    # Always quote CSV scalars — never splice Object[] into the cmd line.
    $inner = 'set SEAM_LAUNCH_CONTEXT=ssh_detached' +
             '&& powershell -NoProfile -File "' + $self + '"' +
             ' -DetachedWorker -LaunchContext ssh_detached' +
             ' -SessionId ' + $sid +
             ' -Tag "' + $Tag + '"' +
             ' -Arms "' + $ArmsCsv + '"' +
             ' -NCached "' + $NCachedCsv + '"' +
             ' -Deltas "' + $DeltasCsv + '"' +
             ' -ModelSpecs "' + $ModelSpecsCsv + '"' +
             ' -CanaryModelSpec "' + $CanaryModelSpecResolved + '"' +
             ' -PipelineTypes "' + $PipelineTypesCsv + '"' +
             ' -CanaryPipelineType "' + $CanaryPipelineTypeResolved + '"' +
             ' -CellTimeoutS ' + $CellTimeoutS +
             ' -Repeats ' + $Repeats +
             ' -CanaryEveryN ' + $CanaryEveryN +
             ' -CanaryArm "' + $CanaryArm + '"' +
             ' -CanaryNCached ' + $CanaryNCached +
             ' -CanaryDelta ' + $CanaryDelta +
             ' -CanaryMode ' + $CanaryMode +
             ' -CanaryCalibrationCount ' + $CanaryCalibrationCount +
             ' -CanaryRelDriftFloor ' + $CanaryRelDriftFloor +
             ' -WatchdogIntervalS ' + $WatchdogIntervalS +
             ' -OrchestratorWsCeilingMb ' + $OrchestratorWsCeilingMb +
             ' -PythonExe "' + $py + '"'
    if (-not [string]::IsNullOrWhiteSpace($PreRunAmendmentFile)) {
        $amendFwd = $PreRunAmendmentFile
        if (-not [System.IO.Path]::IsPathRooted($amendFwd)) {
            $amendFwd = Join-Path $root ($amendFwd -replace '/', '\')
        }
        $inner = $inner + ' -PreRunAmendmentFile "' + $amendFwd + '"'
    }
    $info = Invoke-DetachedSpawn -InnerCommand $inner -LogPath $log -TagName $tagLaunch `
        -LaunchCtx $launchContext -Sid $sid -ResultPath $resultPath -VerifyOnly $false

    Write-Output "launched detached delta_prefill matrix"
    Write-Output "  tag            : $tagLaunch"
    Write-Output "  pid            : $($info.pid) (parent $($info.parent_name) -- not this session)"
    Write-Output "  isolation_mode : remote"
    Write-Output "  launch_context : $launchContext"
    Write-Output "  session_id     : $sid"
    Write-Output "  matrix_tag     : $Tag"
    Write-Output "  arms           : $($arms -join ',')"
    Write-Output "  n_cached       : $($NCachedList -join ',')"
    Write-Output "  deltas         : $($DeltaList -join ',')"
    Write-Output "  max_delta      : $MaxDelta"
    Write-Output "  cell_timeout_s : $CellTimeoutS"
    Write-Output "  repeats        : $Repeats"
    Write-Output "  canary_every_n : $CanaryEveryN"
    if ($CanaryEveryN -gt 0) {
        Write-Output ("  canary         : {0} nc={1} d={2} {3} calib={4} floor={5}" -f `
            $CanaryArm, $CanaryNCached, $CanaryDelta, $CanaryMode, $CanaryCalibrationCount, $CanaryRelDriftFloor)
    }
    Write-Output "  log            : $log"
    Write-Output ""
    Write-Output "Close this session. Do not touch the XPS. Poll with:"
    Write-Output "  powershell -NoProfile -File tools/run_delta_prefill_matrix.ps1 -Status"
    Write-Output ""
    Write-Output "Heartbeat:"
    Write-Output ("  derived/delta_prefill/{0}/heartbeat.json" -f $sid)
    exit 0
}

# ----------------------------------------------------------------------------------------------
# Worker / foreground run path
if ($DetachedWorker) {
    Assert-LaunchContext-Honest -Context $LaunchContext -WorkerMode
    if (-not $SessionId) {
        Write-Output "REFUSED -- -DetachedWorker requires -SessionId"
        exit 1
    }
} else {
    Assert-No-Contending
    Assert-LaunchContext-Honest -Context $LaunchContext
}

$sessionDir = $null
if ($SessionId) {
    $sessionDir = Join-Path $sessionRoot $SessionId
    New-Item -ItemType Directory -Force -Path $sessionDir | Out-Null
    Write-Output ("session_id     : {0}" -f $SessionId)
}

if ($HeartbeatOnly) {
    if (-not $sessionDir) {
        Write-Output "REFUSED -- -HeartbeatOnly requires -SessionId"
        exit 1
    }
    $deadline = (Get-Date).AddSeconds([Math]::Max(5, $HeartbeatSeconds))
    $tick = 0
    Write-MatrixHeartbeat -SessionDir $sessionDir -Sid $SessionId -Phase "verify_detach_start" `
        -Extra @{ tick = $tick; heartbeat_only = $true }
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 2
        $tick++
        Write-MatrixHeartbeat -SessionDir $sessionDir -Sid $SessionId -Phase "verify_detach" `
            -Extra @{ tick = $tick; heartbeat_only = $true }
    }
    Write-MatrixHeartbeat -SessionDir $sessionDir -Sid $SessionId -Phase "verify_detach_complete" `
        -Extra @{ tick = $tick; heartbeat_only = $true }
    $plan = [ordered]@{
        session_id     = $SessionId
        tag            = $Tag
        launch_context = $LaunchContext
        status         = "verify_detach_complete"
        ticks          = $tick
        ended_utc      = (Get-Date).ToUniversalTime().ToString("o")
    }
    $plan | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $sessionDir "plan.json") -Encoding utf8
    $plan | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $sessionDir "result.json") -Encoding utf8
    Write-Output "verify_detach complete ticks=$tick"
    exit 0
}

$preRunSettle = Get-IsolationScalar -Path $cfgPath -Key "pre_run_settle_s"
if (-not $preRunSettle) { $preRunSettle = "300" }
$preRunAvailableMin = Get-IsolationScalar -Path $cfgPath -Key "pre_run_available_mb_min"
$preRunAvailableCite = Get-IsolationScalar -Path $cfgPath -Key "pre_run_available_mb_min_citation"
if (-not $preRunAvailableMin) {
    Write-Output "REFUSED -- isolation.pre_run_available_mb_min missing from $cfgPath"
    exit 2
}
$recoverySettle = Get-RecoveryScalar -Path $cfgPath -Key "settle_s"
if (-not $recoverySettle) { $recoverySettle = "20" }
$recoveryMaxWait = Get-RecoveryScalar -Path $cfgPath -Key "max_wait_s"
if (-not $recoveryMaxWait) { $recoveryMaxWait = "420" }

$baseSeed = Get-YamlScalar -Path $cfgPath -Key "randomization_seed"
if (-not $baseSeed) { $baseSeed = "20260805" }

New-Item -ItemType Directory -Force -Path $sessionRoot | Out-Null
$planPath = if ($sessionDir) {
    Join-Path $sessionDir "plan.json"
} else {
    Join-Path $sessionRoot "${Tag}_plan.json"
}

$cellSpecs = Get-CellSpecs
$canaryEnabled = ($CanaryEveryN -gt 0)
$canaryNDerivation = $null
if ($canaryEnabled) {
    $canaryNDerivation = (
        "N=CanaryEveryN from measured thermal onset in session 7f569929-4484-4af7-8231-5b535526f653 " +
        "(dispatch_o_u8): last-good nc=8000 RESIDENT turn1_prefill_s=6.993 at 2026-08-12T03:15:35Z " +
        "-> first-bad turn1=33.958 at 03:26:32Z (delta_t=657s). Mean cell wall from DISPATCH_O_TIME_ESTIMATE " +
        "4929s/96cells=51.34s -> cells_in_onset=657/51.34~=12.8. N=floor(cells_in_onset)=12 so >=1 canary " +
        "falls inside the observed onset window. Not a round number chosen for convenience."
    )
}
$plan = [ordered]@{
    tag                               = $Tag
    session_id                        = $SessionId
    launch_context                    = $LaunchContext
    kind                              = "delta_prefill_matrix"
    arms                              = $arms
    n_cached                          = @($NCachedList | ForEach-Object { [int]$_ })
    deltas                            = @($DeltaList | ForEach-Object { [int]$_ })
    max_delta                         = [int]$MaxDelta
    non_resident_delta_rule           = "every -Deltas entry (paired with RESIDENT); see DEFECT_non_resident_max_delta_only.md"
    cell_timeout_s                    = [int]$CellTimeoutS
    cell_timeout_derivation           = (
        "session d5c98342-a0b2-41a9-b6e2-93ac7a39c3ba: OK max elapsed_s=616.8; " +
        "2x=1234; rounded up to 1500. Hung cell gpu_only RESIDENT d500 r1 elapsed_s=2856.5 " +
        "(CL_OUT_OF_RESOURCES / oneDNN errcode -5 / post-record spin / operator kill exit=-1). " +
        "generation.timeout_s=1800 bounds only generate() inside the child."
    )
    repeats                           = $Repeats
    model_specs                       = @($ModelSpecList)
    canary_model_spec                 = $CanaryModelSpecResolved
    pipeline_types                    = @($PipelineTypeList)
    canary_pipeline_type              = $CanaryPipelineTypeResolved
    cell_specs_per_round              = @(
        $cellSpecs | ForEach-Object {
            # Single-model + single-pipeline default: keep the historical 4-field
            # shape so sealed plan.json cell_specs_per_round diffs byte-for-byte
            # on those keys (41e419bd acceptance).
            if ($ModelSpecList.Count -eq 1 -and $PipelineTypeList.Count -eq 1) {
                [ordered]@{
                    arm = $_.arm; n_cached = [int]$_.n_cached; mode = $_.mode; delta = [int]$_.delta
                }
            } elseif ($ModelSpecList.Count -eq 1) {
                [ordered]@{
                    arm = $_.arm
                    n_cached = [int]$_.n_cached
                    mode = $_.mode
                    delta = [int]$_.delta
                    pipeline_type = [string]$_.pipeline_type
                }
            } elseif ($PipelineTypeList.Count -eq 1) {
                [ordered]@{
                    arm = $_.arm
                    n_cached = [int]$_.n_cached
                    mode = $_.mode
                    delta = [int]$_.delta
                    model_spec = [string]$_.model_spec
                    ir_sha256 = [string]$_.ir_sha256
                }
            } else {
                [ordered]@{
                    arm = $_.arm
                    n_cached = [int]$_.n_cached
                    mode = $_.mode
                    delta = [int]$_.delta
                    model_spec = [string]$_.model_spec
                    ir_sha256 = [string]$_.ir_sha256
                    pipeline_type = [string]$_.pipeline_type
                }
            }
        }
    )
    cells_per_round                   = $cellSpecs.Count
    canary                            = [ordered]@{
        enabled              = $canaryEnabled
        every_n_matrix_cells = [int]$CanaryEveryN
        arm                  = $CanaryArm
        n_cached             = [int]$CanaryNCached
        delta                = [int]$CanaryDelta
        mode                 = $CanaryMode
        model_spec           = $CanaryModelSpecResolved
        ir_sha256            = [string]$ModelSpecIrSha[$CanaryModelSpecResolved]
        pipeline_type        = $CanaryPipelineTypeResolved
        calibration_count    = [int]$CanaryCalibrationCount
        rel_drift_floor      = [double]$CanaryRelDriftFloor
        n_derivation         = $canaryNDerivation
        threshold_derivation = $(if ($canaryEnabled) {
            "After CanaryCalibrationCount successful canaries: ref_t1=median(turn1_prefill_s), " +
            "ref_t2=median(turn2_prefill_s). early_max_t1=max_i |t1_i-ref_t1|/ref_t1 (same for t2). " +
            "threshold_t1=max(2*early_max_t1, CanaryRelDriftFloor); same for t2. " +
            "2x early envelope = allow as much additional deviation as already observed in calibration; " +
            "floor=0.05 matches C2f canary_gate margin_above_idle_p95 spirit when early_max~=0. " +
            "Gate arms only after calibration. Abort status=FAIL_CANARY_DRIFT if either turn exceeds threshold. " +
            "Threshold is NOT a pre-chosen round number (e.g. not 2.0x or 5x)."
        } else { $null })
    }
    pre_run_settle_s                  = [double]$preRunSettle
    inter_cell_settle_s               = [double]$recoverySettle
    inter_cell_settle_source          = "recovery.settle_s + poll Available to pre_run floor"
    inter_cell_max_wait_s             = [double]$recoveryMaxWait
    pre_run_available_mb_min          = [double]$preRunAvailableMin
    pre_run_available_mb_min_citation = $preRunAvailableCite
    randomization_seed                = [int]$baseSeed
    started_utc                       = (Get-Date).ToUniversalTime().ToString("o")
    environment_launch                = Get-MachineDriftSnapshot
    ssh_client                        = $env:SSH_CLIENT
    ssh_connection                    = $env:SSH_CONNECTION
    seam_launch_context_env           = $env:SEAM_LAUNCH_CONTEXT
    detached_worker                   = [bool]$DetachedWorker
    watchdog                          = [ordered]@{
        enabled     = ($WatchdogIntervalS -gt 0)
        process     = "WorkloadsSessionHost"
        interval_s  = [int]$WatchdogIntervalS
        mechanism   = "sibling_powershell_process"
        note        = $(if ($WatchdogIntervalS -gt 0) {
            "Re-kill every WatchdogIntervalS for life of run; measured respawn ~10-15 min. Not Start-Job."
        } else { "off (WatchdogIntervalS=0)" })
    }
    orchestrator_ws_guard             = [ordered]@{
        enabled           = ($OrchestratorWsCeilingMb -gt 0)
        ceiling_mb        = [int]$OrchestratorWsCeilingMb
        ceiling_bytes     = [int64]($OrchestratorWsCeilingMb) * 1MB
        abort_status      = "aborted_orchestrator_ws_ceiling"
        ws_peak_bytes     = $null
        ws_peak_mb        = $null
        note              = $(if ($OrchestratorWsCeilingMb -gt 0) {
            "Checked each iteration. Peak working set persisted at end so the ceiling is verifiable. An orchestrator that consumes the memory its cells are gated on defeats the Available floor."
        } else { "off (OrchestratorWsCeilingMb=0)" })
    }
    cells                             = @()
    canaries                          = @()
    status                            = "running"
}

# Pre-run amendment MUST be in the plan before cell 1 (prediction registration).
if (-not [string]::IsNullOrWhiteSpace($PreRunAmendmentFile)) {
    $amendPath = $PreRunAmendmentFile
    if (-not [System.IO.Path]::IsPathRooted($amendPath)) {
        $amendPath = Join-Path $root ($amendPath -replace '/', '\')
    }
    if (-not (Test-Path -LiteralPath $amendPath)) {
        Write-Output "REFUSED -- -PreRunAmendmentFile not found: $amendPath"
        exit 1
    }
    $amendText = [System.IO.File]::ReadAllText($amendPath)
    if ([string]::IsNullOrWhiteSpace($amendText)) {
        Write-Output "REFUSED -- -PreRunAmendmentFile is empty: $amendPath"
        exit 1
    }
    $plan["pre_run_amendment"] = [ordered]@{
        source_path   = $amendPath
        recorded_utc  = (Get-Date).ToUniversalTime().ToString("o")
        text          = $amendText
        note          = "Copied into plan before first cell. A prediction not present here does not exist."
    }
}

$plan | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $planPath -Encoding utf8

$OrchestratorWsCeilingBytes = [int64]$OrchestratorWsCeilingMb * 1MB
$script:orchestratorWsPeakBytes = [int64]0
if ($OrchestratorWsCeilingMb -gt 0) {
    $wsAtStart = Get-OrchestratorWorkingSetBytes
    Note-OrchestratorWorkingSetPeak -WsBytes $wsAtStart
    $plan.orchestrator_ws_guard.ws_at_start_bytes = $wsAtStart
    $plan.orchestrator_ws_guard.ws_at_start_mb = [math]::Round($wsAtStart / 1MB, 1)
    $plan | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $planPath -Encoding utf8
    if (-not (Assert-OrchestratorWorkingSet -CeilingBytes $OrchestratorWsCeilingBytes `
            -AtStep "after_plan_write" -SessionDir $(if ($sessionDir) { $sessionDir } else { "" }))) {
        $plan.status = "aborted_orchestrator_ws_ceiling"
        $plan.abort_reason = "aborted_orchestrator_ws_ceiling"
        $plan.orchestrator_ws_guard.ws_peak_bytes = [int64]$script:orchestratorWsPeakBytes
        $plan.orchestrator_ws_guard.ws_peak_mb = [math]::Round($script:orchestratorWsPeakBytes / 1MB, 1)
        $plan | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $planPath -Encoding utf8
        Write-Host "REFUSED -- aborted_orchestrator_ws_ceiling after_plan_write"
        exit 3
    }
}

$watchHandle = $null
$killLog = $null
if ($sessionDir -and $WatchdogIntervalS -gt 0) {
    $killLog = Join-Path $sessionDir "watchdog_kills.jsonl"
    $plan.watchdog.kill_log = $killLog
    $plan | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $planPath -Encoding utf8
    $watchHandle = Start-WshWatchdog -KillLogPath $killLog -IntervalS $WatchdogIntervalS -SessionDir $sessionDir
    Write-Host ("watchdog pid={0} interval_s={1} log={2}" -f $watchHandle.pid, $WatchdogIntervalS, $killLog)
}

if ($sessionDir) {
    Write-MatrixHeartbeat -SessionDir $sessionDir -Sid $SessionId -Phase "starting" `
        -Extra @{ pre_run_settle_s = [double]$preRunSettle }
}

Write-Output "delta_prefill matrix"
Write-Output "  launch_context : $LaunchContext"
Write-Output "  tag            : $Tag"
Write-Output "  arms           : $($arms -join ', ')"
Write-Output "  model_specs    : $($ModelSpecList -join ',')"
Write-Output "  pipeline_types : $($PipelineTypeList -join ',')"
Write-Output "  n_cached       : $($NCachedList -join ',')"
Write-Output "  deltas         : $($DeltaList -join ',')"
Write-Output "  max_delta      : $MaxDelta"
Write-Output "  cell_timeout_s : $CellTimeoutS"
Write-Output "  cells/round    : $($cellSpecs.Count) (RESIDENT + NON_RESIDENT x every delta x arms x n_cached)"
Write-Output "  repeats        : $Repeats"
Write-Output "  canary_every_n : $CanaryEveryN"
Write-Output ("  watchdog_interval_s : {0}" -f $WatchdogIntervalS)
Write-Output ("  orchestrator_ws_ceiling_mb : {0}" -f $OrchestratorWsCeilingMb)
if ($canaryEnabled) {
    Write-Output ("  canary         : {0} nc={1} d={2} {3} model={4} pipeline={5} calib={6} floor={7}" -f `
        $CanaryArm, $CanaryNCached, $CanaryDelta, $CanaryMode, `
        $CanaryModelSpecResolved, $CanaryPipelineTypeResolved, `
        $CanaryCalibrationCount, $CanaryRelDriftFloor)
}
Write-Output "  pre_run_settle : $preRunSettle s"
Write-Output "  inter_cell     : recovery.settle_s=$recoverySettle + Available>=$preRunAvailableMin"
Write-Output "  plan           : $planPath"

Assert-No-Contending

if (-not $SkipSettle) {
    Write-Output ""
    Write-Output "Waiting pre_run_settle_s=$preRunSettle (isolation after interactive software closed)..."
    if ($sessionDir) {
        Write-MatrixHeartbeat -SessionDir $sessionDir -Sid $SessionId -Phase "pre_run_settle" `
            -Extra @{ pre_run_settle_s = [double]$preRunSettle }
    }
    Start-Sleep -Seconds ([double]$preRunSettle)
} else {
    Write-Output "WARNING: -SkipSettle set; pre_run_settle skipped (diagnostic only)."
}

Write-Output ""
Write-Output "Pre-run Available gate (before first cell):"
Assert-PreRunAvailable -MinMb ([double]$preRunAvailableMin) -Citation $preRunAvailableCite | Out-Null
Assert-No-Contending

# Hard gate before canaries/matrix when cb_no_eviction is in the design.
Assert-CbNoEvictionTokenGate -LaunchCtx $LaunchContext -ArmId $arms[0] `
    -ModelSpecPath $CanaryModelSpecResolved

# Same unwrap path as post-settle / post-canary; refuse before any cell if broken.
Assert-PipelineRecordUnwrapGate

$cellRecords = New-Object System.Collections.Generic.List[object]
$canaryRecords = New-Object System.Collections.Generic.List[object]
$cellIndex = 0
$matrixCellsSinceCanary = 0
$abortReason = $null
$script:settleFloorAbort = $null
$canaryGate = [ordered]@{
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

function Invoke-OneDeltaCell {
    param(
        [Parameter(Mandatory = $true)][string]$Arm,
        [Parameter(Mandatory = $true)][string]$Mode,
        [Parameter(Mandatory = $true)][int]$NCached,
        [Parameter(Mandatory = $true)][int]$Delta,
        [Parameter(Mandatory = $true)][int]$Repeat,
        [Parameter(Mandatory = $true)][string]$ModelSpec,
        [Parameter(Mandatory = $true)][string]$PipelineType,
        [object]$Seed = $null,
        [Parameter(Mandatory = $true)][bool]$IsCanary,
        [object]$CanaryIndex = $null,
        [object]$AfterMatrixCellIndex = $null
    )
    $started = (Get-Date).ToUniversalTime().ToString("o")
    $uptimeS = Get-UptimeSeconds
    $label = if ($IsCanary) { "CANARY" } else { "cell" }
    $irSha = [string]$ModelSpecIrSha[$ModelSpec]
    if (-not $irSha) {
        throw "REFUSED -- no ir_sha256 for model spec $ModelSpec (cell is not a measurement)"
    }
    # Write-Host: callers capture the OrderedDictionary return. Write-Output here made
    # `$rec = Invoke-OneDeltaCell` an Object[3]; `$rec["rel_drift_t1"]` then threw
    # InvalidCastFromStringToInteger (DISPATCH P CANARY0, 2026-08-12).
    Write-Host ("-- {0} arm={1} n_cached={2} mode={3} delta={4} r={5} model={6} pipeline={7} @ {8}" -f `
        $label, $Arm, $NCached, $Mode, $Delta, $Repeat, $ModelSpec, $PipelineType, $started)
    if ($sessionDir) {
        $phase = if ($IsCanary) { "canary" } else { "cell" }
        Write-MatrixHeartbeat -SessionDir $sessionDir -Sid $SessionId -Phase $phase `
            -CellIndex $cellIndex -Arm $Arm -Mode $Mode -Delta $Delta -Repeat $Repeat `
            -Extra @{
                seed                   = $Seed
                n_cached               = $NCached
                is_canary              = $IsCanary
                canary_index           = $CanaryIndex
                after_matrix_cell_index = $AfterMatrixCellIndex
                model_spec             = $ModelSpec
                ir_sha256              = $irSha
                pipeline_type          = $PipelineType
            }
    }

    if ($IsCanary) {
        $artifactName = "{0}_{1}_CANARY{2}_arm{3}_{4}_nc{5}_d{6}.json" -f `
            $Tag, $LaunchContext, $CanaryIndex, $Arm, $Mode, $NCached, $Delta
    } elseif ($ModelSpecList.Count -gt 1 -and $PipelineTypeList.Count -gt 1) {
        $modelStem = [System.IO.Path]::GetFileNameWithoutExtension($ModelSpec)
        $artifactName = "{0}_{1}_arm{2}_{3}_nc{4}_d{5}_r{6}_{7}_{8}.json" -f `
            $Tag, $LaunchContext, $Arm, $Mode, $NCached, $Delta, $Repeat, $modelStem, $PipelineType
    } elseif ($ModelSpecList.Count -gt 1) {
        $modelStem = [System.IO.Path]::GetFileNameWithoutExtension($ModelSpec)
        $artifactName = "{0}_{1}_arm{2}_{3}_nc{4}_d{5}_r{6}_{7}.json" -f `
            $Tag, $LaunchContext, $Arm, $Mode, $NCached, $Delta, $Repeat, $modelStem
    } elseif ($PipelineTypeList.Count -gt 1) {
        $artifactName = "{0}_{1}_arm{2}_{3}_nc{4}_d{5}_r{6}_{7}.json" -f `
            $Tag, $LaunchContext, $Arm, $Mode, $NCached, $Delta, $Repeat, $PipelineType
    } else {
        # Single-model + single-pipeline default: preserve historical artifact naming.
        $artifactName = "{0}_{1}_arm{2}_{3}_nc{4}_d{5}_r{6}.json" -f `
            $Tag, $LaunchContext, $Arm, $Mode, $NCached, $Delta, $Repeat
    }
    $artifactPath = Join-Path $sessionRoot $artifactName
    if ($sessionDir) {
        $artifactPath = Join-Path $sessionDir $artifactName
    }

    $childArgs = @(
        "-u", $smokePy,
        "--launch-context", $LaunchContext,
        "--arm", $Arm,
        "--mode", $Mode,
        "--n-cached", "$NCached",
        "--delta", "$Delta",
        "--max-new-tokens", "64",
        "--repeat", "$Repeat",
        "--tag", $Tag,
        "--model-spec", $ModelSpec,
        "--pipeline-type", $PipelineType,
        "--out", $artifactPath
    )
    $child = Invoke-SeamChildProcess -FilePath $py -ArgumentList $childArgs `
        -TimeoutSeconds $CellTimeoutS -WorkingDirectory $root
    $exit = [int]$child.exit_code
    $timedOut = [bool]$child.timed_out
    $elapsedS = [double]$child.elapsed_s

    $cls = $null
    $t1 = $null
    $t2 = $null
    $d1 = $null
    $d2 = $null
    $peak1 = $null
    $peak2 = $null
    $retained = $null
    $envStart = $null
    $envPeak = $null
    $err = $null
    $obj = $null
    $kvReadback = $null
    if (Test-Path -LiteralPath $artifactPath) {
        try {
            $obj = Get-Content -LiteralPath $artifactPath -Raw | ConvertFrom-Json
            $cls = $obj.classification
            $err = $obj.execute_error
            if (-not $err) { $err = $obj.compile_error }
            if ($obj.turn1) {
                $t1 = $obj.turn1.prefill_s
                $d1 = $obj.turn1.decode_tok_s
                $peak1 = $obj.turn1.peak_ws_bytes
            }
            if ($obj.turn2) {
                $t2 = $obj.turn2.prefill_s
                $d2 = $obj.turn2.decode_tok_s
                $peak2 = $obj.turn2.peak_ws_bytes
            }
            if ($obj.cache_retention) {
                $retained = $obj.cache_retention.cache_retained
            }
            $envStart = $obj.environment_start
            $envPeak = $obj.environment_peak
            if ($obj.kv_cache_precision_readback) {
                $kvReadback = $obj.kv_cache_precision_readback
            }
        } catch {
            $cls = "PARSE_ERROR"
            $err = "$_"
        }
    } else {
        $cls = "MISSING_ARTIFACT"
    }
    if ($timedOut) {
        $cls = "TIMEOUT"
        $timeoutNote = ("parent cell wall timeout after {0}s (CellTimeoutS={1}; child pid={2})" -f `
            $elapsedS, $CellTimeoutS, $child.pid)
        if ($err) { $err = "$timeoutNote | prior: $err" } else { $err = $timeoutNote }
    }

    $rec = [ordered]@{
        cell_index       = $cellIndex
        is_canary        = $IsCanary
        canary_index     = $CanaryIndex
        after_matrix_cell_index = $AfterMatrixCellIndex
        arm              = $Arm
        mode             = $Mode
        n_cached         = $NCached
        delta            = $Delta
        repeat           = $Repeat
        seed             = $Seed
        model_spec       = $ModelSpec
        ir_sha256        = $irSha
        pipeline_type    = $PipelineType
        uptime_s         = $uptimeS
        started_utc      = $started
        ended_utc        = (Get-Date).ToUniversalTime().ToString("o")
        exit_code        = $exit
        elapsed_s        = $elapsedS
        timed_out        = $timedOut
        cell_timeout_s   = [int]$CellTimeoutS
        kv_cache_precision_readback = $kvReadback
        classification   = $cls
        turn1_prefill_s  = $t1
        turn2_prefill_s  = $t2
        turn1_decode_tok_s = $d1
        turn2_decode_tok_s = $d2
        peak_ws_turn1    = $peak1
        peak_ws_turn2    = $peak2
        cache_retained   = $retained
        environment_start = $envStart
        environment_peak  = $envPeak
        error            = $err
        artifact         = $artifactPath
    }
    Write-Host ("   classification={0} t1_prefill={1} t2_prefill={2} retained={3} peak_ws=({4}->{5}) exit={6} elapsed_s={7}{8}" -f `
        $cls, $t1, $t2, $retained, $peak1, $peak2, $exit, $elapsedS, `
        $(if ($timedOut) { " TIMEOUT" } else { "" }))

    if ($sessionDir) {
        Write-MatrixHeartbeat -SessionDir $sessionDir -Sid $SessionId -Phase "inter_cell_settle" `
            -CellIndex $cellIndex -Arm $Arm -Mode $Mode -Delta $Delta -Repeat $Repeat `
            -Extra @{ classification = $cls; is_canary = $IsCanary; settle_note = "enter" }
    }
    $settle = Wait-InterCellAvailable -MinMb ([double]$preRunAvailableMin) `
        -SettleS ([double]$recoverySettle) -MaxWaitS ([double]$recoveryMaxWait) `
        -SessionDir $(if ($sessionDir) { $sessionDir } else { "" }) `
        -Sid $(if ($SessionId) { $SessionId } else { "" }) `
        -CellIndex $cellIndex -Arm $Arm -Mode $Mode -Delta $Delta -Repeat $Repeat
    # Unwrap: Wait returns a single object; defend against pipeline pollution.
    $settle = Get-SinglePipelineRecord -InputObject $settle
    if (-not $settle.ok) {
        $script:settleFloorAbort = [ordered]@{
            status       = "settle_floor_not_met"
            available_mb = $settle.available_mb
            min_mb       = $settle.min_mb
            max_wait_s   = $settle.max_wait_s
            probe_error  = $settle.probe_error
            after_cell   = [ordered]@{
                cell_index    = $cellIndex
                is_canary     = $IsCanary
                arm           = $Arm
                mode          = $Mode
                n_cached      = $NCached
                delta         = $Delta
                repeat        = $Repeat
                pipeline_type = $PipelineType
                classification = $cls
            }
            utc = (Get-Date).ToUniversalTime().ToString("o")
        }
        Write-Host ("ABORT settle_floor_not_met available_mb={0} floor={1}" -f `
            $settle.available_mb, $settle.min_mb)
        Write-Output -NoEnumerate $rec
        return
    }
    Assert-No-Contending
    # -NoEnumerate: defend against any residual success-stream companions.
    Write-Output -NoEnumerate $rec
}

function Invoke-DriftCanary {
    param([object]$AfterMatrixCellIndex)
    $cIdx = $canaryRecords.Count
    $raw = Invoke-OneDeltaCell -Arm $CanaryArm -Mode $CanaryMode `
        -NCached $CanaryNCached -Delta $CanaryDelta -Repeat 0 -Seed $null `
        -ModelSpec $CanaryModelSpecResolved -PipelineType $CanaryPipelineTypeResolved `
        -IsCanary $true -CanaryIndex $cIdx -AfterMatrixCellIndex $AfterMatrixCellIndex
    # Unwrap Write-Output pollution / member-enumeration trap before string-key writes.
    $rec = Get-SinglePipelineRecord -InputObject $raw
    $script:cellIndex++

    if ($script:settleFloorAbort) {
        $canaryRecords.Add($rec) | Out-Null
        $plan.canaries = ConvertTo-ObjectArray $canaryRecords
        $plan.canary_gate = $canaryGate
        $plan | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $planPath -Encoding utf8
        # Write-Host (not Write-Output): callers use `if (-not (Invoke-DriftCanary))`.
        # A Write-Output string + return $false becomes Object[2]; non-empty arrays are
        # truthy, so -not never fires and FAIL_CANARY_DRIFT is swallowed
        # (89f77871 canaries 4/5: drift_tripped=true, status=complete, no REFUSED in log).
        Write-Host ("REFUSED -- settle_floor_not_met during canary: available_mb={0}" -f `
            $script:settleFloorAbort.available_mb)
        Write-Output -NoEnumerate $false
        return
    }

    $bkRaw = Update-CanaryDriftBookkeeping `
        -Rec $rec `
        -CanaryGate $canaryGate `
        -PriorCanaryRecords (ConvertTo-ObjectArray $canaryRecords) `
        -CanaryCalibrationCount $CanaryCalibrationCount `
        -CanaryRelDriftFloor $CanaryRelDriftFloor
    # Bookkeeping returns a hashtable via Write-Output -NoEnumerate; unwrap so
    # $bk["tripped"] is a scalar bool (same pollution class as cell records).
    $bk = Get-SinglePipelineRecord -InputObject $bkRaw
    $rec = $bk["rec"]
    if ($null -eq $rec) { $rec = $bk.rec }
    $rec = Get-SinglePipelineRecord -InputObject $rec
    $justArmed = $false
    if ($bk -is [System.Collections.IDictionary]) {
        $justArmed = [bool]$bk["just_armed"]
        $tripped = [bool]$bk["tripped"]
        $tripDetail = $bk["trip_detail"]
        $derivation = $bk["derivation_applied"]
    } else {
        $justArmed = [bool]$bk.just_armed
        $tripped = [bool]$bk.tripped
        $tripDetail = $bk.trip_detail
        $derivation = $bk.derivation_applied
    }
    # Defense in depth: record flag is the source of truth written onto the canary.
    if ($rec -is [System.Collections.IDictionary]) {
        if ([bool]$rec["drift_tripped"]) { $tripped = $true }
        if (-not $tripDetail) { $tripDetail = $rec["trip_detail"] }
    } elseif ([bool]$rec.drift_tripped) {
        $tripped = $true
        if (-not $tripDetail) { $tripDetail = $rec.trip_detail }
    }
    if ($justArmed) {
        Write-Host ("   canary_gate ARMED: {0}" -f $derivation)
    }
    $canaryRecords.Add($rec) | Out-Null

    # Persist canary progress into plan for -Status visibility.
    $plan.canaries = ConvertTo-ObjectArray $canaryRecords
    $plan.canary_gate = $canaryGate
    $plan | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $planPath -Encoding utf8

    if ($tripped) {
        Write-Host ("REFUSED -- FAIL_CANARY_DRIFT: {0}" -f $tripDetail)
        Write-Output -NoEnumerate $false
        return
    }
    Write-Output -NoEnumerate $true
}

# Opening canary (matrix cell index -1 = before first matrix cell).
if ($canaryEnabled) {
    Write-Output ""
    Write-Output "=== canary (start) ==="
    $canaryOk = Invoke-DriftCanary -AfterMatrixCellIndex -1
    if (-not $canaryOk) {
        if ($script:settleFloorAbort) {
            $abortReason = "settle_floor_not_met"
        } else {
            $abortReason = "FAIL_CANARY_DRIFT"
        }
    }
}

for ($r = 0; $r -lt $Repeats -and -not $abortReason; $r++) {
    $seed = [int]$baseSeed + (10007 * $r) + ($NCachedList.Count * 17) + ($DeltaList.Count * 3)
    $rng = [System.Random]::new($seed)
    $order = @($cellSpecs)
    for ($i = $order.Count - 1; $i -gt 0; $i--) {
        $j = $rng.Next(0, $i + 1)
        $tmp = $order[$i]
        $order[$i] = $order[$j]
        $order[$j] = $tmp
    }
    Write-Output ""
    Write-Output ("=== repeat={0} seed={1} order=[{2}] ===" -f $r, $seed, `
        (($order | ForEach-Object {
            "{0}:nc{1}:{2}:d{3}:m{4}:p{5}" -f $_.arm, $_.n_cached, $_.mode, $_.delta,
                ([System.IO.Path]::GetFileNameWithoutExtension([string]$_.model_spec)),
                [string]$_.pipeline_type
        }) -join ", "))

    foreach ($spec in $order) {
        if ($abortReason) { break }
        if ($OrchestratorWsCeilingMb -gt 0) {
            if (-not (Assert-OrchestratorWorkingSet -CeilingBytes $OrchestratorWsCeilingBytes `
                    -AtStep ("before_cell_r{0}_i{1}" -f $r, $cellRecords.Count) `
                    -SessionDir $(if ($sessionDir) { $sessionDir } else { "" }))) {
                $abortReason = "aborted_orchestrator_ws_ceiling"
                break
            }
        }
        $arm = [string]$spec.arm
        $mode = [string]$spec.mode
        $delta = [int]$spec.delta
        $nc = [int]$spec.n_cached
        $ms = [string]$spec.model_spec
        $pt = [string]$spec.pipeline_type
        $rawCell = Invoke-OneDeltaCell -Arm $arm -Mode $mode -NCached $nc -Delta $delta `
            -Repeat $r -Seed $seed -ModelSpec $ms -PipelineType $pt -IsCanary $false
        $rec = Get-SinglePipelineRecord -InputObject $rawCell
        $cellRecords.Add($rec) | Out-Null
        $cellIndex++
        $matrixCellsSinceCanary++

        if ($script:settleFloorAbort) {
            $abortReason = "settle_floor_not_met"
            break
        }

        # Checkpoint plan periodically so Status sees progress.
        if (($cellRecords.Count % 4) -eq 0) {
            $plan.cells = ConvertTo-ObjectArray $cellRecords
            $plan | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $planPath -Encoding utf8
        }

        if ($canaryEnabled -and $matrixCellsSinceCanary -ge $CanaryEveryN) {
            Write-Output ""
            Write-Output ("=== canary (after matrix cell {0}) ===" -f ($cellRecords.Count - 1))
            if ($OrchestratorWsCeilingMb -gt 0) {
                if (-not (Assert-OrchestratorWorkingSet -CeilingBytes $OrchestratorWsCeilingBytes `
                        -AtStep ("before_canary_after_cell_{0}" -f ($cellRecords.Count - 1)) `
                        -SessionDir $(if ($sessionDir) { $sessionDir } else { "" }))) {
                    $abortReason = "aborted_orchestrator_ws_ceiling"
                    break
                }
            }
            $canaryOk = Invoke-DriftCanary -AfterMatrixCellIndex ($cellRecords.Count - 1)
            if (-not $canaryOk) {
                if ($script:settleFloorAbort) {
                    $abortReason = "settle_floor_not_met"
                } else {
                    $abortReason = "FAIL_CANARY_DRIFT"
                }
                break
            }
            $matrixCellsSinceCanary = 0
        }
    }
}

$plan.cells = ConvertTo-ObjectArray $cellRecords
$plan.canaries = ConvertTo-ObjectArray $canaryRecords
$plan.canary_gate = $canaryGate
$plan.environment_end = Get-MachineDriftSnapshot
$plan.ended_utc = (Get-Date).ToUniversalTime().ToString("o")
# Final WS sample so peak covers the whole session, not only Assert call sites.
if ($OrchestratorWsCeilingMb -gt 0) {
    Note-OrchestratorWorkingSetPeak -WsBytes (Get-OrchestratorWorkingSetBytes)
    $plan.orchestrator_ws_guard.ws_peak_bytes = [int64]$script:orchestratorWsPeakBytes
    $plan.orchestrator_ws_guard.ws_peak_mb = [math]::Round($script:orchestratorWsPeakBytes / 1MB, 1)
}
Stop-WshWatchdog -Handle $watchHandle
if ($script:settleFloorAbort) {
    $abortReason = "settle_floor_not_met"
    $plan.settle_abort = $script:settleFloorAbort
}
if ($abortReason) {
    $plan.status = $abortReason
    $plan.abort_reason = $abortReason
} else {
    $plan.status = "complete"
}
$summaryPath = if ($sessionDir) {
    Join-Path $sessionDir "summary.json"
} else {
    Join-Path $sessionRoot "${Tag}_summary.json"
}
$plan | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $planPath -Encoding utf8
$plan | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $summaryPath -Encoding utf8

if ($sessionDir) {
    $hbPhase = if ($abortReason) { "aborted" } else { "complete" }
    Write-MatrixHeartbeat -SessionDir $sessionDir -Sid $SessionId -Phase $hbPhase `
        -CellIndex $cellIndex -Extra @{
            cells_completed    = $cellRecords.Count
            canaries_completed = $canaryRecords.Count
            status             = $plan.status
        }
    if ($DetachedWorker) {
        $resultPath = Join-Path $launchDir "last_result.json"
        @{
            status     = $plan.status
            session_id = $SessionId
            tag        = $Tag
            cells      = $cellRecords.Count
            canaries   = $canaryRecords.Count
            plan_path  = $planPath
            ended_utc  = $plan.ended_utc
            abort_reason = $abortReason
        } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $resultPath -Encoding utf8
    }
}

Write-Output ""
if ($abortReason) {
    Write-Output ("=== matrix ABORTED ({0}) ===" -f $abortReason)
} else {
    Write-Output "=== matrix complete ==="
}
Write-Output "plan    : $planPath"
Write-Output "summary : $summaryPath"
Write-Output ("cells   : {0}" -f $cellRecords.Count)
Write-Output ("canaries: {0}" -f $canaryRecords.Count)
if ($canaryGate.calibration_complete) {
    Write-Output ("canary_gate: {0}" -f $canaryGate.derivation_applied)
}
Write-ThreeNumbers -Cells $cellRecords
if ($abortReason) { exit 3 }
