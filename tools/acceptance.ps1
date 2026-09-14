<#
.SYNOPSIS
  Fixed-config throughput acceptance gate. Prefer -Orchestrate for the ΔN-mode measurement.

.DESCRIPTION
  From the Mac, with Cursor and browsers closed on the XPS:

    ssh xps "cd C:/Users/zjohn/Projects/gnn-hls-accel; powershell -File tools/acceptance.ps1 -Orchestrate"
    ssh xps "cd C:/Users/zjohn/Projects/gnn-hls-accel; powershell -File tools/acceptance.ps1 -Status"

  -Orchestrate is one WMI-detached job: after the interactive-software check passes, idle
  isolation.pre_run_settle_s (300 from configs/delta_n.yaml) *inside the detached process*,
  seal arm 1, idle gap_s (1200), seal arm 2, seal the derived verdict. Declares
  isolation_mode=remote and launch_context=ssh_detached. Refuses if Cursor or other tier-1
  CONTENDING processes are resident -- it does not kill them (tier-2 shell/vendor agents are
  recorded only). The settle wait is authoritative in Python (recorded with actual_wait_s);
  closing SSH after launch does not cut it short.

  Legacy two-launch path (still available):

    ssh xps "... acceptance.ps1"
    # poll -Status, wait out the gap, launch again, then -Compare

  The launch is detached through tools/spawn_detached.ps1, so closing the SSH session does not
  touch it. Pass -Local only for non-gated development; local and remote results are never pooled.

  The interpreter is pinned to .venv-seam\Scripts\python.exe. Bare `python` is never invoked:
  over SSH, PATH often resolves to the Microsoft Store stub.
#>
[CmdletBinding(DefaultParameterSetName = "Run")]
param(
    [Parameter(ParameterSetName = "Run")][switch]$Local,
    [Parameter(ParameterSetName = "Run")][switch]$Orchestrate,
    [Parameter(ParameterSetName = "Status")][switch]$Status,
    [Parameter(ParameterSetName = "Compare")][switch]$Compare,
    [Parameter(ParameterSetName = "Compare")][string]$RunA,
    [Parameter(ParameterSetName = "Compare")][string]$RunB,
    [Parameter(ParameterSetName = "Run")]
    [Parameter(ParameterSetName = "Status")]
    [Parameter(ParameterSetName = "Compare")]
    [string]$PythonExe = "C:\Users\zjohn\Projects\gnn-hls-accel\.venv-seam\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$root = "C:\Users\zjohn\Projects\gnn-hls-accel"
$outDir = Join-Path $root "derived\fixed_throughput\_launches"
$statePath = Join-Path $outDir "launches.json"

try {
    $py = [System.IO.Path]::GetFullPath($PythonExe)
} catch {
    $py = $PythonExe
}

function Get-Launches {
    if (Test-Path $statePath) { return @(Get-Content $statePath -Raw | ConvertFrom-Json) }
    return @()
}

function Assert-Interpreter {
    if (-not (Test-Path -LiteralPath $py)) {
        Write-Output "REFUSED -- INTERPRETER_MISSING"
        Write-Output "  pinned PythonExe does not exist: $py"
        Write-Output "  refusing to invoke bare python or a Store stub"
        exit 1
    }
}

# ----------------------------------------------------------------------------------------------
if ($Status) {
    $launches = Get-Launches
    if ($launches.Count -eq 0) { Write-Output "no acceptance run has been launched"; exit 1 }
    $last = $launches[-1]

    $alive = $null -ne (Get-Process -Id $last.pid -ErrorAction SilentlyContinue)
    Write-Output "tag            : $($last.tag)"
    Write-Output "launched       : $($last.launched_utc)"
    Write-Output "declared       : isolation_mode=$($last.isolation_mode) launch_context=$($last.launch_context)"
    Write-Output "orchestrated   : $($last.orchestrated)"
    Write-Output "process        : pid $($last.pid), alive=$alive"

    if (Test-Path $last.result_path) {
        Write-Output "state          : COMPLETE"
        Write-Output ""
        Get-Content $last.result_path -Raw
        exit 0
    }
    if ($alive) {
        Write-Output "state          : RUNNING -- do not open a session on this machine"
    } else {
        Write-Output "state          : ENDED WITHOUT A RESULT -- the log below is the evidence"
    }
    Write-Output ""
    Write-Output "--- last 40 lines of $($last.log_path) ---"
    if (Test-Path $last.log_path) { Get-Content $last.log_path -Tail 40 }
    if (-not $alive) { exit 1 }
    exit 0
}

# ----------------------------------------------------------------------------------------------
if ($Compare) {
    Assert-Interpreter
    if (-not $RunA -or -not $RunB) {
        $done = @(Get-Launches | Where-Object { Test-Path $_.result_path })
        if ($done.Count -lt 2) {
            Write-Output "need two completed acceptance runs; have $($done.Count)"
            exit 1
        }
        $RunA = (Get-Content $done[-2].result_path -Raw | ConvertFrom-Json).run_id
        $RunB = (Get-Content $done[-1].result_path -Raw | ConvertFrom-Json).run_id
    }
    Write-Output "comparing $RunA vs $RunB"
    # The comparison measures nothing, so its own isolation_mode carries no claim -- `local` is
    # declared so that reading a result after reopening the editor is not refused. The mode that
    # bears on validity is the one both source runs share, which the harness checks separately
    # and records as compared_isolation_mode.
    $env:SEAM_ISOLATION_MODE = "local"
    Remove-Item Env:SEAM_LAUNCH_CONTEXT -ErrorAction SilentlyContinue
    & $py -u -m seam.tools.fixed_throughput --compare $RunA $RunB --allow-dirty
    exit $LASTEXITCODE
}

# ----------------------------------------------------------------------------------------------
Assert-Interpreter
$mode = if ($Local) { "local" } else { "remote" }
$launchContext = if ($Local) { "local_console" } else { "ssh_detached" }

# Fail before the launch, not ten minutes into it. The harness enforces this too, but discovering
# it here costs a second instead of a measurement window. This check REFUSES; it does not kill.
# Operator kills outside the harness are unobservable to the measurement process.
if ($mode -eq "remote") {
    # Keep in sync with seam/isolation.py TIER1 (refuse) / TIER2 (record only).
    $tier1Names = @(
        "Cursor", "Code", "chrome", "msedge",
        "firefox", "brave", "slack", "Discord", "Teams", "ms-teams", "Spotify", "OUTLOOK",
        "obsidian", "docker desktop", "vmmem", "claude"
    )
    $tier2Names = @(
        "msedgewebview2", "SearchHost", "Widgets", "WorkloadsSessionHost",
        "DellOptimizer.Systray", "SupportAssistAgent", "ICPS"
    )
    $tier1Wanted = @{}
    foreach ($n in $tier1Names) { $tier1Wanted[$n.ToLowerInvariant()] = $true }
    $tier2Wanted = @{}
    foreach ($n in $tier2Names) { $tier2Wanted[$n.ToLowerInvariant()] = $true }
    $contending = @(Get-Process -ErrorAction SilentlyContinue | Where-Object {
        $tier1Wanted.ContainsKey($_.ProcessName.ToLowerInvariant())
    })
    $tier2 = @(Get-Process -ErrorAction SilentlyContinue | Where-Object {
        $tier2Wanted.ContainsKey($_.ProcessName.ToLowerInvariant())
    })
    if ($contending.Count -gt 0) {
        Write-Output "REFUSED -- isolation_mode=remote but tier-1 operator-controlled software is resident (private WS):"
        $contending | ForEach-Object {
            Write-Output ("  - {0} (pid {1}, private {2:N0} MiB)" -f `
                $_.ProcessName, $_.Id, ($_.PrivateMemorySize64 / 1MB))
        }
        Write-Output ""
        Write-Output "Close them on the XPS and relaunch. Measuring with these running and"
        Write-Output "labelling it remote is the failure behind the unexplained 1.98x."
        Write-Output "This script does not terminate those processes."
        Write-Output "Tier-2 shell/vendor agents are recorded only and do not refuse."
        exit 1
    }
    if ($tier2.Count -gt 0) {
        Write-Output "tier-2 (record only, do not refuse) -- auto-respawning shell/vendor:"
        $tier2 | Group-Object ProcessName | ForEach-Object {
            $privateMb = ($_.Group | Measure-Object -Property PrivateMemorySize64 -Sum).Sum / 1MB
            Write-Output ("  - {0} x{1} private={2:N0} MiB" -f $_.Name, $_.Count, $privateMb)
        }
    }
}

if ($Local -and $Orchestrate) {
    Write-Output "REFUSED -- -Orchestrate is the remote/ssh_detached gate path; do not combine with -Local"
    exit 1
}

New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$tag = if ($Orchestrate) { "acc_orch_" } else { "acc_" }
$tag = $tag + (Get-Date -Format "yyyyMMdd_HHmmss")
$log = Join-Path $outDir "$tag.log"
$resultPath = Join-Path $outDir "$tag.result.json"

# --allow-dirty is not optional in practice: the tree has been uncommitted since 2026-07-30, so
# without it the run would refuse -- and it would refuse only after the machine had been closed
# down for it. The manifest records git_dirty and the full list of uncommitted files, so nothing
# is hidden by allowing it.
#
# SEAM_ISOLATION_MODE / SEAM_LAUNCH_CONTEXT reach a process the launcher does not otherwise talk
# to. The harness has no default for isolation_mode and refuses to emit without it; launch_context
# is required for acceptance arms.
$moduleArgs = if ($Orchestrate) { " --orchestrate" } else { "" }
$inner = 'set SEAM_ISOLATION_MODE=' + $mode +
         '&& set SEAM_LAUNCH_CONTEXT=' + $launchContext +
         '&& "' + $py + '" -u -m seam.tools.fixed_throughput' + $moduleArgs +
         ' --allow-dirty --result-json "' + $resultPath + '"'

$json = & (Join-Path $root "tools\spawn_detached.ps1") -CommandLine $inner -LogPath $log -WorkingDirectory $root
$info = $json | ConvertFrom-Json

$entry = [pscustomobject]@{
    tag             = $tag
    pid             = $info.pid
    parent_name     = $info.parent_name
    isolation_mode  = $mode
    launch_context  = $launchContext
    orchestrated    = [bool]$Orchestrate
    launched_utc    = (Get-Date).ToUniversalTime().ToString("o")
    log_path        = $log
    result_path     = $resultPath
}
$launches = @(Get-Launches) + $entry
$launches | ConvertTo-Json -Depth 5 | Set-Content -Path $statePath -Encoding UTF8

Write-Output "launched detached acceptance run"
Write-Output "  tag            : $tag"
Write-Output "  pid            : $($info.pid) (parent $($info.parent_name) -- not this session)"
Write-Output "  isolation_mode : $mode"
Write-Output "  launch_context : $launchContext"
Write-Output "  orchestrated   : $Orchestrate"
if ($Orchestrate) {
    Write-Output "  pre_run_settle : isolation.pre_run_settle_s inside detached job before arm 1"
    Write-Output "                   (authoritative wait + actual_wait_s recorded in manifests)"
}
Write-Output "  log            : $log"
Write-Output ""
Write-Output "Close this session. Do not touch the XPS. Poll with:"
Write-Output "  powershell -File tools/acceptance.ps1 -Status"
