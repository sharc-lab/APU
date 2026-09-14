<#
.SYNOPSIS
  A vs gpu_only memory+speed matrix (default N∈{2000,12000}; override with -NsList), 3 repeats, interleaved by arm.

.DESCRIPTION
  Diagnostic only. No seal, no amendment, no commit.

  Prefer -Orchestrate (WMI-detached via tools/spawn_detached.ps1). Declares launch_context=ssh_detached.
  Refuses if Cursor/Chrome/etc are resident. Refuses -Orchestrate with LaunchContext=ssh_foreground.
  Child cells call tools/smoke_gpu_exec.ps1 with the same ssh_detached label.

  From the Mac, with Cursor and browsers closed on the XPS:

    ssh xps "cd C:/Users/zjohn/Projects/gnn-hls-accel; powershell -NoProfile -File tools/run_gpu_only_matrix.ps1 -Orchestrate -Tag reconcile_n12000 -NsList 12000"
    ssh xps "cd C:/Users/zjohn/Projects/gnn-hls-accel; powershell -NoProfile -File tools/run_gpu_only_matrix.ps1 -Status"

  Checkpoint / heartbeat:
    derived/gpu_only_matrix/<session_id>/heartbeat.json
    derived/gpu_only_matrix/<session_id>/plan.json
    derived/gpu_only_matrix/_launches/<tag>.log

  -VerifyDetach proves spawn+Status without a full matrix (heartbeat-only ~30s; does not require
  Cursor closed). Real -Orchestrate still refuses interactive software.

  Legacy foreground (dies under memory pressure / SSH keepalive loss — not for n=12000):

    ssh xps "... run_gpu_only_matrix.ps1 -LaunchContext ssh_foreground -Tag ... -NsList ..."

  Waits isolation.pre_run_settle_s after the refuse check, then requires
  \Memory\Available MBytes >= isolation.pre_run_available_mb_min (7000; cite 2026-08-09
  clean session / 7477 MB) before the first cell. For each N runs repeats of interleaved
  arms [A, gpu_only] with a recorded per-round shuffle seed. Inter-cell settle uses
  isolation.pre_run_settle_s (NOT recovery.settle_s). Settle-adequacy gate fails the run if
  max−min available_mb (environment_start) across arms at that n exceeds
  isolation.settle_adequacy_max_spread_mb (provisional; contaminated 39a8a0f5).

  Per-cell wall timeout (-CellTimeoutS, default 1500): parent kills hung children and
  records classification=TIMEOUT. Derivation from delta_prefill session d5c98342
  (OK max elapsed_s=616.8; 2×→1234; rounded to 1500). generation.timeout_s only bounds
  generate() inside the child.

  Dry-run the cleanliness / contending gates without measuring:

    powershell -NoProfile -File tools/run_gpu_only_matrix.ps1 -DryRunGate

  Report-only (no measurement; aggregate existing artifacts):

    powershell -NoProfile -File tools/run_gpu_only_matrix.ps1 -Tag reconcile_n12000 -NsList 12000 -ReportOnly
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
    [string]$Tag = "gpu_only_matrix",

    [Parameter(ParameterSetName = "Run")]
    [Parameter(ParameterSetName = "Orchestrate")]
    [Parameter(ParameterSetName = "Status")]
    [Parameter(ParameterSetName = "VerifyDetach")]
    [Parameter(ParameterSetName = "DryRunGate")]
    [string]$PythonExe = "C:\Users\zjohn\Projects\gnn-hls-accel\.venv-seam\Scripts\python.exe",

    # Comma-separated prompt lengths. Default preserves the original {2000,12000} matrix.
    [Parameter(ParameterSetName = "Run")]
    [Parameter(ParameterSetName = "Orchestrate")]
    [string]$NsList = "2000,12000",

    # Per-cell wall-clock timeout (parent kills child tree). See synopsis derivation.
    [Parameter(ParameterSetName = "Run")]
    [Parameter(ParameterSetName = "Orchestrate")]
    [Parameter(ParameterSetName = "DryRunGate")]
    [int]$CellTimeoutS = 1500,

    [Parameter(ParameterSetName = "Run")][switch]$ReportOnly,
    [Parameter(ParameterSetName = "Run")][switch]$SkipSettle,

    [Parameter(ParameterSetName = "Orchestrate")][switch]$Orchestrate,
    [Parameter(ParameterSetName = "Status")][switch]$Status,
    [Parameter(ParameterSetName = "VerifyDetach")][switch]$VerifyDetach,
    [Parameter(ParameterSetName = "DryRunGate")][switch]$DryRunGate,

    # Internal: WMI-spawned worker (matrix or verify heartbeat). Not a user-facing path.
    [Parameter(ParameterSetName = "Run")][switch]$DetachedWorker,
    [Parameter(ParameterSetName = "Run")][string]$SessionId,
    [Parameter(ParameterSetName = "Run")][switch]$HeartbeatOnly,
    [Parameter(ParameterSetName = "Run")][int]$HeartbeatSeconds = 30
)

$ErrorActionPreference = "Stop"
$root = "C:\Users\zjohn\Projects\gnn-hls-accel"
. (Join-Path $root "tools\SeamPsCommon.ps1")
$cfgPath = Join-Path $root "configs\delta_n.yaml"
$smokePs1 = Join-Path $root "tools\smoke_gpu_exec.ps1"
$reportPy = Join-Path $root "tools\report_gpu_only_matrix.py"
$smokeOutDir = Join-Path $root "derived\gpu_smoke"
$sessionRoot = Join-Path $root "derived\gpu_only_matrix"
$launchDir = Join-Path $sessionRoot "_launches"
$statePath = Join-Path $launchDir "launches.json"
$arms = @("A", "gpu_only")
$Ns = @($NsList.Split(",") | ForEach-Object { [int]($_.Trim()) })
$repeats = 3
if ($CellTimeoutS -lt 1) {
    Write-Output "REFUSED -- -CellTimeoutS must be >= 1 (got $CellTimeoutS)"
    exit 2
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
        Write-Output "  refusing to invoke bare python or a Store stub"
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
# Legacy alias: refuse gates use tier 1 only.
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
    # Tier 1 only -- these refuse remote / DryRunGate.
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
    # Write-Host: safe when a caller captures a boolean return from Assert-PreRunAvailable.
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

function Get-AvailableMBytes {
    # Prefer the same counter Windows reports (\Memory\Available MBytes), not FreePhysicalMemory.
    try {
        $sample = (Get-Counter '\Memory\Available MBytes' -ErrorAction Stop).CounterSamples[0]
        return [pscustomobject]@{
            available_mb = [double]$sample.CookedValue
            method       = "Get-Counter:\\Memory\\Available MBytes"
        }
    } catch {
        return [pscustomobject]@{
            available_mb = $null
            method       = "Get-Counter_failed:$($_.Exception.Message)"
        }
    }
}

function Assert-PreRunAvailable {
    param(
        [double]$MinMb,
        [string]$Citation,
        [switch]$ReportOnly
    )
    # Messages via Write-Host so callers can capture the boolean return without swallowing
    # the operator-facing Available MBytes / refusal lines (PowerShell merges Write-Output
    # into the function success stream).
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
            Write-Host "Close residual workloads; this script does not terminate processes."
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
            Write-Output "  Foreground SSH dies under memory pressure (sshd keepalive / Broken pipe)."
            Write-Output "  Omit -LaunchContext or pass -LaunchContext ssh_detached."
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
        # WMI child has no SSH_* env; honesty is the Orchestrate parent + forced label.
        return
    }
    if ($Context -eq "ssh_foreground" -or $Context -eq "ssh_detached") {
        if (-not $env:SSH_CLIENT -and -not $env:SSH_CONNECTION) {
            Write-Output "REFUSED -- LaunchContext=$Context claimed but SSH_CLIENT/SSH_CONNECTION unset."
            Write-Output "Do not mislabel a local Cursor/console session as ssh_foreground/ssh_detached."
            Write-Output "For the real matrix: -Orchestrate (ssh_detached via spawn_detached)."
            Write-Output "For a true foreground SSH session: run over real SSH from the Mac."
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
            # Strip inline comments; first token for scalars.
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

function Get-FreePhysicalMbStart {
    param($Obj)
    if ($null -eq $Obj) { return $null }
    if ($null -ne $Obj.free_physical_mb_start) {
        return [double]$Obj.free_physical_mb_start
    }
    if ($null -ne $Obj.diagnostics -and $null -ne $Obj.diagnostics.rss_window) {
        $rw = $Obj.diagnostics.rss_window
        if ($null -ne $rw.free_physical_mb_start) {
            return [double]$rw.free_physical_mb_start
        }
    }
    return $null
}

function Get-AvailableMbStart {
    param($Obj)
    if ($null -eq $Obj) { return $null }
    if ($null -ne $Obj.available_mb_start) {
        return [double]$Obj.available_mb_start
    }
    if ($null -ne $Obj.environment_start -and $null -ne $Obj.environment_start.available_mb) {
        return [double]$Obj.environment_start.available_mb
    }
    if ($null -ne $Obj.diagnostics -and $null -ne $Obj.diagnostics.rss_window) {
        $rw = $Obj.diagnostics.rss_window
        if ($null -ne $rw.available_mb_start) {
            return [double]$rw.available_mb_start
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
        [object]$N = $null,
        [object]$Repeat = $null,
        [hashtable]$Extra = $null
    )
    New-Item -ItemType Directory -Force -Path $SessionDir | Out-Null
    $hb = [ordered]@{
        session_id    = $Sid
        phase         = $Phase
        cell_index    = $CellIndex
        arm           = $Arm
        n_tokens      = $N
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

function Invoke-Report {
    param([string]$Context)
    $summaryPath = Join-Path $smokeOutDir "${Tag}_summary.json"
    & $py -u $reportPy `
        --launch-context $Context `
        --tag $Tag `
        --smoke-dir $smokeOutDir `
        --ns-list $NsList `
        --json-out $summaryPath
    if ($LASTEXITCODE -ne 0) {
        Write-Output "report aggregator failed with exit $LASTEXITCODE"
        exit $LASTEXITCODE
    }
    Write-Output ""
    Write-Output "summary json: $summaryPath"
}

function Test-SettleAdequacy {
    param(
        [System.Collections.IEnumerable]$Cells,
        [int]$N,
        [double]$MaxSpreadMb,
        [string]$Citation = "contaminated 39a8a0f5 / pending clean re-derive",
        [bool]$Provisional = $true
    )
    $starts = New-Object System.Collections.Generic.List[double]
    $byArm = @{}
    foreach ($c in $Cells) {
        if ([int]$c.n_tokens -ne $N) { continue }
        if ($null -eq $c.available_mb_start) { continue }
        $v = [double]$c.available_mb_start
        $starts.Add($v) | Out-Null
        $arm = [string]$c.arm
        if (-not $byArm.ContainsKey($arm)) {
            $byArm[$arm] = New-Object System.Collections.Generic.List[double]
        }
        $byArm[$arm].Add($v) | Out-Null
    }
    if ($starts.Count -lt 2 -or $byArm.Count -lt 2) {
        return [ordered]@{
            evaluated = $false
            reason    = "need available_mb_start from both arms at n=$N"
            spread_mb = $null
            max_mb    = $null
            min_mb    = $null
            by_arm    = $byArm
            pass      = $null
            metric    = "available_mb"
            provisional = $Provisional
        }
    }
    $max = ($starts | Measure-Object -Maximum).Maximum
    $min = ($starts | Measure-Object -Minimum).Minimum
    $spread = $max - $min
    $pass = $spread -le $MaxSpreadMb
    return [ordered]@{
        evaluated = $true
        n_tokens  = $N
        spread_mb = $spread
        max_mb    = $max
        min_mb    = $min
        threshold_mb = $MaxSpreadMb
        by_arm    = $byArm
        pass      = $pass
        metric    = "available_mb"
        provisional = $Provisional
        rule      = "max(available_mb)-min(...) across arms at same n <= settle_adequacy_max_spread_mb"
        citation  = $Citation
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
        ns_list         = $NsList
        verify_detach   = $VerifyOnly
        launched_utc    = (Get-Date).ToUniversalTime().ToString("o")
        log_path        = $LogPath
        result_path     = $ResultPath
    }
    Save-Launches -Entries (@(Get-Launches) + $entry)
    return $info
}

# ----------------------------------------------------------------------------------------------
if ($Status) {
    # Fixed pattern from ceiling_a status repair: job timeout, heartbeat-dir discovery,
    # bounded FileShare.ReadWrite reads, no Select-String against the live spawn log.
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
            $lines.Add("no gpu_only_matrix run has been launched") | Out-Null
            return @{ exit = 1; lines = $lines }
        }
        $parsed = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
        $launches = ConvertTo-ObjectArray (Get-LaunchEntriesFromNode -Node $parsed)
        if ($launches.Count -eq 0) {
            $lines.Add("no gpu_only_matrix run has been launched") | Out-Null
            return @{ exit = 1; lines = $lines }
        }
        $last = $launches[-1]
        $alive = $null -ne (Get-Process -Id ([int]$last.pid) -ErrorAction SilentlyContinue)
        $lines.Add("tag            : $($last.tag)") | Out-Null
        $lines.Add("launched       : $($last.launched_utc)") | Out-Null
        $lines.Add("declared       : isolation_mode=$($last.isolation_mode) launch_context=$($last.launch_context)") | Out-Null
        if ($last.matrix_tag) { $lines.Add("matrix_tag     : $($last.matrix_tag)") | Out-Null }
        if ($last.ns_list) { $lines.Add("ns_list        : $($last.ns_list)") | Out-Null }
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
            if ($planText -and $planText -match '"status"\s*:\s*"(complete|FAIL_SETTLE_ADEQUACY|verify_detach_complete)"') {
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
        Write-Output "  Prefer: Get-Content derived/gpu_only_matrix/<session_id>/heartbeat.json"
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

if ($ReportOnly) {
    Invoke-Report -Context $LaunchContext
    exit 0
}

# ----------------------------------------------------------------------------------------------
# -VerifyDetach: spawn heartbeat-only child via spawn_detached; does NOT require Cursor closed.
if ($VerifyDetach) {
    $launchContext = "ssh_detached"
    New-Item -ItemType Directory -Force -Path $launchDir | Out-Null
    $sid = [guid]::NewGuid().ToString()
    $tagLaunch = "gpu_matrix_verify_" + (Get-Date -Format "yyyyMMdd_HHmmss")
    $log = Join-Path $launchDir "$tagLaunch.log"
    $resultPath = Join-Path $launchDir "$tagLaunch.result.json"
    $self = Join-Path $root "tools\run_gpu_only_matrix.ps1"
    $inner = 'set SEAM_LAUNCH_CONTEXT=ssh_detached' +
             '&& powershell -NoProfile -File "' + $self + '"' +
             ' -DetachedWorker -HeartbeatOnly -HeartbeatSeconds 30' +
             ' -LaunchContext ssh_detached -SessionId ' + $sid +
             ' -Tag "' + $Tag + '" -NsList "' + $NsList + '"' +
             ' -PythonExe "' + $py + '"'
    $info = Invoke-DetachedSpawn -InnerCommand $inner -LogPath $log -TagName $tagLaunch `
        -LaunchCtx $launchContext -Sid $sid -ResultPath $resultPath -VerifyOnly $true
    Write-Output "launched detached gpu_only_matrix VERIFY (heartbeat-only)"
    Write-Output "  tag            : $tagLaunch"
    Write-Output "  pid            : $($info.pid) (parent $($info.parent_name) -- not this session)"
    Write-Output "  isolation_mode : remote"
    Write-Output "  launch_context : $launchContext"
    Write-Output "  session_id     : $sid"
    Write-Output "  log            : $log"
    Write-Output ""
    Write-Output "Poll with:"
    Write-Output "  powershell -NoProfile -File tools/run_gpu_only_matrix.ps1 -Status"
    Write-Output "Heartbeat:"
    Write-Output ("  derived/gpu_only_matrix/{0}/heartbeat.json" -f $sid)
    exit 0
}

# ----------------------------------------------------------------------------------------------
# -DryRunGate: report Available MBytes + contending list + pre_run floor; no matrix.
if ($DryRunGate) {
    $preRunMin = Get-IsolationScalar -Path $cfgPath -Key "pre_run_available_mb_min"
    $preRunCite = Get-IsolationScalar -Path $cfgPath -Key "pre_run_available_mb_min_citation"
    if (-not $preRunMin) {
        Write-Output "REFUSED -- isolation.pre_run_available_mb_min missing from $cfgPath"
        exit 2
    }
    Write-Output "dry-run cleanliness gate (no matrix)"
    Write-Output ("  ns_list          : {0}" -f ($Ns -join ","))
    Write-Output ("  cell_timeout_s   : {0}" -f $CellTimeoutS)
    Write-Output ("  cell_timeout_derivation: session d5c98342 OK max elapsed_s=616.8; 2x=1234; rounded to 1500")
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

# ----------------------------------------------------------------------------------------------
# -Orchestrate: refuse Contending + Available floor, force ssh_detached, spawn via spawn_detached.ps1.
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
    $sid = [guid]::NewGuid().ToString()
    $tagLaunch = "gpu_matrix_" + (Get-Date -Format "yyyyMMdd_HHmmss")
    $log = Join-Path $launchDir "$tagLaunch.log"
    $resultPath = Join-Path $launchDir "$tagLaunch.result.json"
    $self = Join-Path $root "tools\run_gpu_only_matrix.ps1"
    $inner = 'set SEAM_LAUNCH_CONTEXT=ssh_detached' +
             '&& powershell -NoProfile -File "' + $self + '"' +
             ' -DetachedWorker -LaunchContext ssh_detached' +
             ' -SessionId ' + $sid +
             ' -Tag "' + $Tag + '" -NsList "' + $NsList + '"' +
             ' -CellTimeoutS ' + $CellTimeoutS +
             ' -PythonExe "' + $py + '"'
    $info = Invoke-DetachedSpawn -InnerCommand $inner -LogPath $log -TagName $tagLaunch `
        -LaunchCtx $launchContext -Sid $sid -ResultPath $resultPath -VerifyOnly $false

    Write-Output "launched detached gpu_only_matrix run"
    Write-Output "  tag            : $tagLaunch"
    Write-Output "  pid            : $($info.pid) (parent $($info.parent_name) -- not this session)"
    Write-Output "  isolation_mode : remote"
    Write-Output "  launch_context : $launchContext"
    Write-Output "  session_id     : $sid"
    Write-Output "  matrix_tag     : $Tag"
    Write-Output "  ns_list        : $NsList"
    Write-Output "  cell_timeout_s : $CellTimeoutS"
    Write-Output "  pre_run_settle : isolation.pre_run_settle_s inside detached job before first cell"
    Write-Output "  log            : $log"
    Write-Output ""
    Write-Output "Close this session. Do not touch the XPS. Poll with:"
    Write-Output "  powershell -NoProfile -File tools/run_gpu_only_matrix.ps1 -Status"
    Write-Output ""
    Write-Output "Heartbeat:"
    Write-Output ("  derived/gpu_only_matrix/{0}/heartbeat.json" -f $sid)
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

# Heartbeat-only verify path (no matrix cells).
if ($HeartbeatOnly) {
    if (-not $sessionDir) {
        Write-Output "REFUSED -- -HeartbeatOnly requires -SessionId (via -DetachedWorker)"
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
    if ($DetachedWorker) {
        $resultPath = Join-Path $launchDir ("last_verify_result.json")
        # Prefer the launch record's result_path if present in env; else write session result.
        $plan | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $sessionDir "result.json") -Encoding utf8
    }
    Write-Output "verify_detach complete ticks=$tick"
    exit 0
}

$preRunSettle = Get-IsolationScalar -Path $cfgPath -Key "pre_run_settle_s"
if (-not $preRunSettle) { $preRunSettle = "300" }
# Inter-cell settle = ladder/isolation full settle. Do NOT use recovery.settle_s.
$interCellSettle = $preRunSettle
$preRunAvailableMin = Get-IsolationScalar -Path $cfgPath -Key "pre_run_available_mb_min"
$preRunAvailableCite = Get-IsolationScalar -Path $cfgPath -Key "pre_run_available_mb_min_citation"
if (-not $preRunAvailableMin) {
    Write-Output "REFUSED -- isolation.pre_run_available_mb_min missing from $cfgPath"
    Write-Output "Pre-register the Available MBytes floor (2026-08-09 clean session / 7477 MB)."
    exit 2
}
$adequacySpread = Get-IsolationScalar -Path $cfgPath -Key "settle_adequacy_max_spread_mb"
if (-not $adequacySpread) {
    Write-Output "REFUSED -- isolation.settle_adequacy_max_spread_mb missing from $cfgPath"
    Write-Output "Pre-register the gate threshold before measuring (provisional; see 39a8a0f5)."
    exit 2
}
$adequacyMetric = Get-IsolationScalar -Path $cfgPath -Key "settle_adequacy_metric"
if (-not $adequacyMetric) { $adequacyMetric = "available_mb" }
$adequacyProvisionalRaw = Get-IsolationScalar -Path $cfgPath -Key "settle_adequacy_provisional"
$adequacyProvisional = $true
if ($adequacyProvisionalRaw) {
    $adequacyProvisional = ($adequacyProvisionalRaw.ToString().ToLowerInvariant() -eq "true")
}
$adequacyCitation = Get-IsolationScalar -Path $cfgPath -Key "settle_adequacy_citation"
if (-not $adequacyCitation) { $adequacyCitation = "contaminated 39a8a0f5 / pending clean re-derive" }

$baseSeed = Get-YamlScalar -Path $cfgPath -Key "randomization_seed"
if (-not $baseSeed) { $baseSeed = "20260805" }

New-Item -ItemType Directory -Force -Path $smokeOutDir | Out-Null
$planPath = if ($sessionDir) {
    Join-Path $sessionDir "plan.json"
} else {
    Join-Path $smokeOutDir "${Tag}_plan.json"
}
$plan = [ordered]@{
    tag                                    = $Tag
    session_id                             = $SessionId
    launch_context                         = $LaunchContext
    arms                                   = $arms
    n_tokens                               = $Ns
    repeats                                = $repeats
    cell_timeout_s                         = [int]$CellTimeoutS
    cell_timeout_derivation                = (
        "session d5c98342-a0b2-41a9-b6e2-93ac7a39c3ba: OK max elapsed_s=616.8; " +
        "2x=1234; rounded up to 1500. Hung cell class: post-record spin after " +
        "CL_OUT_OF_RESOURCES / oneDNN errcode -5; generation.timeout_s=1800 bounds only generate()."
    )
    pre_run_settle_s                       = [double]$preRunSettle
    inter_cell_settle_s                    = [double]$interCellSettle
    inter_cell_settle_source               = "isolation.pre_run_settle_s"
    pre_run_available_mb_min               = [double]$preRunAvailableMin
    pre_run_available_mb_min_citation      = $preRunAvailableCite
    settle_adequacy_metric                 = $adequacyMetric
    settle_adequacy_max_spread_mb          = [double]$adequacySpread
    settle_adequacy_provisional            = [bool]$adequacyProvisional
    settle_adequacy_rule                   = "max-min available_mb across arms at same n"
    settle_adequacy_citation               = $adequacyCitation
    randomization_seed                     = [int]$baseSeed
    started_utc                            = (Get-Date).ToUniversalTime().ToString("o")
    ssh_client                             = $env:SSH_CLIENT
    ssh_connection                         = $env:SSH_CONNECTION
    seam_launch_context_env                = $env:SEAM_LAUNCH_CONTEXT
    detached_worker                        = [bool]$DetachedWorker
    settle_adequacy_checks                 = @()
    cells                                  = @()
    status                                 = "running"
}
$plan | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $planPath -Encoding utf8
# Keep tag-prefixed plan under gpu_smoke for report-only discoverability.
if ($sessionDir) {
    Copy-Item -LiteralPath $planPath -Destination (Join-Path $smokeOutDir "${Tag}_plan.json") -Force
}

if ($sessionDir) {
    Write-MatrixHeartbeat -SessionDir $sessionDir -Sid $SessionId -Phase "starting" `
        -Extra @{ pre_run_settle_s = [double]$preRunSettle }
}

Write-Output "gpu_only matrix"
Write-Output "  launch_context : $LaunchContext"
Write-Output "  tag            : $Tag"
Write-Output "  arms           : $($arms -join ', ')"
Write-Output "  N              : $($Ns -join ', ')"
Write-Output "  repeats        : $repeats"
Write-Output "  cell_timeout_s : $CellTimeoutS"
Write-Output "  pre_run_settle : $preRunSettle s"
Write-Output "  inter_cell     : $interCellSettle s (isolation.pre_run_settle_s; NOT recovery.settle_s)"
Write-Output "  pre_run floor  : Available MBytes >= $preRunAvailableMin MB"
Write-Output ("  settle gate    : max-min available_mb across arms <= {0} MB (provisional={1})" -f `
    $adequacySpread, $adequacyProvisional)
Write-Output "  plan           : $planPath"

# Detached worker skipped the early refuse; foreground already checked. Re-check is cheap.
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

$cellRecords = New-Object System.Collections.Generic.List[object]
$adequacyChecks = New-Object System.Collections.Generic.List[object]
$failedAdequacy = $false
$cellIndex = 0

foreach ($n in $Ns) {
    for ($r = 0; $r -lt $repeats; $r++) {
        $seed = [int]$baseSeed + ([int]$n * 100) + $r
        $rng = [System.Random]::new($seed)
        $order = @($arms)
        # Fisher-Yates shuffle; recorded seed makes the interleave auditable.
        for ($i = $order.Count - 1; $i -gt 0; $i--) {
            $j = $rng.Next(0, $i + 1)
            $tmp = $order[$i]
            $order[$i] = $order[$j]
            $order[$j] = $tmp
        }
        Write-Output ""
        Write-Output ("=== N={0} repeat={1} seed={2} order=[{3}] ===" -f $n, $r, $seed, ($order -join ", "))

        foreach ($arm in $order) {
            $started = (Get-Date).ToUniversalTime().ToString("o")
            Write-Output ("-- arm={0} N={1} r={2} @ {3}" -f $arm, $n, $r, $started)
            if ($sessionDir) {
                Write-MatrixHeartbeat -SessionDir $sessionDir -Sid $SessionId -Phase "cell" `
                    -CellIndex $cellIndex -Arm $arm -N $n -Repeat $r `
                    -Extra @{ seed = $seed; order = ($order -join ",") }
            }

            $childArgs = @(
                "-NoProfile", "-File", $smokePs1,
                "-LaunchContext", $LaunchContext,
                "-Arm", $arm,
                "-N", "$n",
                "-Repeat", "$r",
                "-PythonExe", $py
            )
            $child = Invoke-SeamChildProcess -FilePath "powershell" -ArgumentList $childArgs `
                -TimeoutSeconds $CellTimeoutS -WorkingDirectory $root
            $exit = [int]$child.exit_code
            $timedOut = [bool]$child.timed_out
            $elapsedS = [double]$child.elapsed_s

            $defaultName = "{0}_arm{1}_n{2}_r{3}.json" -f $LaunchContext, $arm, $n, $r
            $defaultPath = Join-Path $smokeOutDir $defaultName
            $taggedName = "{0}_{1}" -f $Tag, $defaultName
            $taggedPath = Join-Path $smokeOutDir $taggedName
            if (Test-Path -LiteralPath $defaultPath) {
                Copy-Item -LiteralPath $defaultPath -Destination $taggedPath -Force
            }

            $cls = $null
            $peak = $null
            $err = $null
            $freeStart = $null
            $freePeak = $null
            $availableStart = $null
            $availablePeak = $null
            $envStart = $null
            $envPeak = $null
            $prefill = $null
            $decode = $null
            if (Test-Path -LiteralPath $taggedPath) {
                try {
                    $obj = Get-Content -LiteralPath $taggedPath -Raw | ConvertFrom-Json
                    $cls = $obj.classification
                    $peak = $obj.peak_ws_bytes
                    $err = $obj.execute_error
                    if (-not $err) { $err = $obj.compile_error }
                    $freeStart = Get-FreePhysicalMbStart -Obj $obj
                    if ($null -ne $obj.free_physical_at_peak) {
                        $freePeak = [double]$obj.free_physical_at_peak / (1024.0 * 1024.0)
                    }
                    $availableStart = Get-AvailableMbStart -Obj $obj
                    if ($null -ne $obj.environment_peak -and $null -ne $obj.environment_peak.available_mb) {
                        $availablePeak = [double]$obj.environment_peak.available_mb
                    }
                    $envStart = $obj.environment_start
                    $envPeak = $obj.environment_peak
                    $prefill = $obj.prefill_s
                    $decode = $obj.decode_tok_s
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

            $cell = [ordered]@{
                cell_index              = $cellIndex
                arm                     = $arm
                n_tokens                = $n
                repeat                  = $r
                seed                    = $seed
                order                   = $order
                started_utc             = $started
                ended_utc               = (Get-Date).ToUniversalTime().ToString("o")
                exit_code               = $exit
                elapsed_s               = $elapsedS
                timed_out               = $timedOut
                cell_timeout_s          = [int]$CellTimeoutS
                classification          = $cls
                peak_ws_bytes           = $peak
                free_physical_mb_start  = $freeStart
                free_physical_mb_at_peak = $freePeak
                available_mb_start      = $availableStart
                available_mb_at_peak    = $availablePeak
                environment_start       = $envStart
                environment_peak        = $envPeak
                prefill_s               = $prefill
                decode_tok_s            = $decode
                error                   = $err
                artifact                = $taggedPath
            }
            $cellRecords.Add($cell) | Out-Null
            Write-Output ("   classification={0} peak_ws_bytes={1} available_start_mb={2} free_start_mb={3} prefill_s={4} decode_tok_s={5} exit={6} elapsed_s={7}{8}" -f `
                $cls, $peak, $availableStart, $freeStart, $prefill, $decode, $exit, $elapsedS, `
                $(if ($timedOut) { " TIMEOUT" } else { "" }))

            if ($cls -eq "UNSUPPORTED") {
                Write-Output "   UNSUPPORTED recorded (no retry). HDR4 ended by platform fact if this is gpu_only."
            }

            if ($sessionDir) {
                Write-MatrixHeartbeat -SessionDir $sessionDir -Sid $SessionId -Phase "inter_cell_settle" `
                    -CellIndex $cellIndex -Arm $arm -N $n -Repeat $r `
                    -Extra @{ inter_cell_settle_s = [double]$interCellSettle; classification = $cls }
            }
            Write-Output ("   inter-cell settle {0}s (isolation.pre_run_settle_s)..." -f $interCellSettle)
            Start-Sleep -Seconds ([double]$interCellSettle)
            $cellIndex++
        }

        $check = Test-SettleAdequacy -Cells $cellRecords -N $n -MaxSpreadMb ([double]$adequacySpread) `
            -Citation $adequacyCitation -Provisional $adequacyProvisional
        $check.round = $r
        $adequacyChecks.Add($check) | Out-Null
        if ($check.evaluated) {
            Write-Output ("   settle-adequacy n={0} r={1}: available_mb spread={2:N1} MB (max={3:N1} min={4:N1} threshold={5} MB provisional={6}) pass={7}" -f `
                $n, $r, $check.spread_mb, $check.max_mb, $check.min_mb, $check.threshold_mb, $check.provisional, $check.pass)
            if (-not $check.pass) {
                Write-Output "FAIL -- settle adequacy: available_mb spread across arms exceeds pre-registered threshold."
                Write-Output "  Contaminated interleave would silently bias the comparison; aborting."
                $failedAdequacy = $true
                break
            }
        } else {
            Write-Output ("   settle-adequacy n={0} r={1}: not yet evaluable ({2})" -f $n, $r, $check.reason)
        }
    }
    if ($failedAdequacy) { break }
}

$plan.cells = $cellRecords.ToArray()
$plan.settle_adequacy_checks = $adequacyChecks.ToArray()
$plan.ended_utc = (Get-Date).ToUniversalTime().ToString("o")
if ($failedAdequacy) {
    $plan.status = "FAIL_SETTLE_ADEQUACY"
} else {
    $plan.status = "complete"
}
$plan | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $planPath -Encoding utf8
if ($sessionDir) {
    Copy-Item -LiteralPath $planPath -Destination (Join-Path $smokeOutDir "${Tag}_plan.json") -Force
    Write-MatrixHeartbeat -SessionDir $sessionDir -Sid $SessionId -Phase $plan.status `
        -CellIndex $cellIndex -Extra @{ cells_completed = $cellRecords.Count }
}

Write-Output ""
if ($failedAdequacy) {
    Write-Output "=== matrix aborted (settle adequacy); writing partial report ==="
    Invoke-Report -Context $LaunchContext
    exit 3
}

Write-Output "=== matrix complete; aggregating ==="
Invoke-Report -Context $LaunchContext
