<#
.SYNOPSIS
  ceiling_a — interleaved context-ceiling ladder under remote isolation.

.DESCRIPTION
  Phase 3 of ΔN. Arms are selected via -Arms (comma-separated ids from
  configs/delta_n.yaml). Default is A,A_prime (unchanged). Pass -Arms B,B_prime
  for the iGPU-resident pair, or -Arms gpu_only for the single GPU pipeline.
  stop_if_ceiling_at_position_limit fires when any selected arm PASSes the top rung.

  From the Mac, with Cursor and browsers closed on the XPS:

    ssh xps "cd C:/Users/zjohn/Projects/gnn-hls-accel; powershell -File tools/ceiling_a.ps1 -Orchestrate"
    ssh xps "cd C:/Users/zjohn/Projects/gnn-hls-accel; powershell -File tools/ceiling_a.ps1 -Orchestrate -Arms B,B_prime"
    ssh xps "cd C:/Users/zjohn/Projects/gnn-hls-accel; powershell -File tools/ceiling_a.ps1 -Orchestrate -Arms gpu_only"
    ssh xps "cd C:/Users/zjohn/Projects/gnn-hls-accel; powershell -File tools/ceiling_a.ps1 -Orchestrate -Arms gpu_only_u8"
    ssh xps "cd C:/Users/zjohn/Projects/gnn-hls-accel; powershell -File tools/ceiling_a.ps1 -Orchestrate -Arms gpu_only_u4"
    ssh xps "cd C:/Users/zjohn/Projects/gnn-hls-accel; powershell -File tools/ceiling_a.ps1 -Status"
    ssh xps "cd C:/Users/zjohn/Projects/gnn-hls-accel; powershell -File tools/ceiling_a.ps1 -Resume -SessionId <session_id>"

  -Orchestrate is one WMI-detached job: after the interactive-software check passes,
  isolation.pre_run_settle_s (300) runs inside the detached process, then the interleaved
  ladder for the selected arms. Declares isolation_mode=remote and launch_context=ssh_detached.
  Holds PowerRequestSystemRequired for the whole ladder; refuses to start if the assertion fails.

  Checkpoint / heartbeat:
    derived/ceiling_a/<session_id>/checkpoint.json
    derived/ceiling_a/<session_id>/heartbeat.json
    derived/ceiling_a/<session_id>/result.json   (when sealed)

  Pass -Local only for non-gated development; local and remote results are never pooled.
  Prefer --smoke on the Python module for harness validation under an interactive session.
#>
[CmdletBinding(DefaultParameterSetName = "Run")]
param(
    [Parameter(ParameterSetName = "Run")][switch]$Local,
    [Parameter(ParameterSetName = "Run")][switch]$Orchestrate,
    [Parameter(ParameterSetName = "Status")][switch]$Status,
    [Parameter(ParameterSetName = "Resume")][switch]$Resume,
    [Parameter(ParameterSetName = "Resume")][string]$SessionId,
    # [object] so unquoted A,A_prime / gpu_only_u4 never fails Object[]→string/int binding.
    [Parameter(ParameterSetName = "Run")]
    [Parameter(ParameterSetName = "Resume")]
    [object]$Arms = "A,A_prime",
    [Parameter(ParameterSetName = "Run")]
    [Parameter(ParameterSetName = "Status")]
    [Parameter(ParameterSetName = "Resume")]
    [string]$PythonExe = "C:\Users\zjohn\Projects\gnn-hls-accel\.venv-seam\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$root = "C:\Users\zjohn\Projects\gnn-hls-accel"
. (Join-Path $root "tools\SeamPsCommon.ps1")
$outDir = Join-Path $root "derived\ceiling_a\_launches"
$statePath = Join-Path $outDir "launches.json"

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
    # Piping arrays into ConvertTo-Json historically nested {value,Count} wrappers into
    # launches.json. Flatten on read so $launches[-1] is always a real launch record.
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
    # -InputObject keeps the array as a JSON array. Piping an array unwraps it and produces
    # the nested {value,Count} shape that previously broke -Status.
    $json = ConvertTo-Json -InputObject @($Entries) -Depth 6
    Set-Content -LiteralPath $statePath -Value $json -Encoding UTF8
}

function Assert-Interpreter {
    if (-not (Test-Path -LiteralPath $py)) {
        Write-Output "REFUSED -- INTERPRETER_MISSING"
        Write-Output "  pinned PythonExe does not exist: $py"
        Write-Output "  refusing to invoke bare python or a Store stub"
        exit 1
    }
}

function Assert-No-Contending {
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
        Write-Output "Close them on the XPS and relaunch. This script does not terminate those processes."
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

# ----------------------------------------------------------------------------------------------
if ($Status) {
    # -Status must never block on the runner's locks, never read checkpoint.json (grows to
    # hundreds of KB and is rewritten every cell), and never Select-String / Get-Content -Wait
    # the live cmd-redirected log. Root cause of the ~80 min wedge: lines formerly used
    # Select-String -Path $last.log_path against the growing cmd.exe `>` log while the
    # orchestrator held it open under memory pressure (see closeout artifact).
    $statusJob = Start-Job -ScriptBlock {
        param($root, $statePath)
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
            $lines.Add("no ceiling_a run has been launched") | Out-Null
            return @{ exit = 1; lines = $lines }
        }
        $parsed = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
        $launches = ConvertTo-ObjectArray (Get-LaunchEntriesFromNode -Node $parsed)
        if ($launches.Count -eq 0) {
            $lines.Add("no ceiling_a run has been launched") | Out-Null
            return @{ exit = 1; lines = $lines }
        }
        $last = $launches[-1]
        $alive = $null -ne (Get-Process -Id ([int]$last.pid) -ErrorAction SilentlyContinue)
        $lines.Add("tag            : $($last.tag)") | Out-Null
        $lines.Add("launched       : $($last.launched_utc)") | Out-Null
        $lines.Add("declared       : isolation_mode=$($last.isolation_mode) launch_context=$($last.launch_context)") | Out-Null
        if ($last.arms) { $lines.Add("arms           : $($last.arms)") | Out-Null }
        $lines.Add("process        : pid $($last.pid), alive=$alive") | Out-Null

        # Prefer heartbeat dir mtime over scanning the live log (cmd `>` holds the log open).
        $sid = $last.session_id
        if (-not $sid) {
            $newest = Get-ChildItem (Join-Path $root "derived\ceiling_a") -Directory -ErrorAction SilentlyContinue |
                      Where-Object { $_.Name -match '^[0-9a-f-]{36}$' } |
                      Sort-Object LastWriteTime -Descending |
                      Select-Object -First 1
            if ($newest) { $sid = $newest.Name }
        }
        if (-not $sid -and $last.log_path) {
            # Bounded head read only — never Select-String the whole growing log.
            $head = Read-SharedTextFile -Path ([string]$last.log_path) -MaxBytes 65536
            if ($head -match '"session_id"\s*:\s*"([0-9a-f-]{36})"') {
                $sid = $Matches[1]
            }
        }
        if ($sid) { $lines.Add("session_id     : $sid") | Out-Null }
        else { $lines.Add("session_id     : (not yet known)") | Out-Null }

        if ($sid) {
            $hb = Join-Path $root ("derived\ceiling_a\{0}\heartbeat.json" -f $sid)
            $rs = Join-Path $root ("derived\ceiling_a\{0}\result.json" -f $sid)
            # Intentionally never open checkpoint.json here.
            $hbText = Read-SharedTextFile -Path $hb -MaxBytes 16384
            if ($hbText) {
                $lines.Add("") | Out-Null
                $lines.Add("--- heartbeat ---") | Out-Null
                $lines.Add($hbText.TrimEnd()) | Out-Null
            } elseif (Test-Path -LiteralPath (Join-Path $root ("derived\ceiling_a\{0}\checkpoint.json" -f $sid))) {
                $lines.Add("checkpoint exists; heartbeat not yet readable (not loading checkpoint)") | Out-Null
            }
            $rsText = Read-SharedTextFile -Path $rs -MaxBytes 65536
            if ($rsText) {
                $lines.Add("") | Out-Null
                $lines.Add("state          : COMPLETE") | Out-Null
                $lines.Add("") | Out-Null
                $lines.Add($rsText.TrimEnd()) | Out-Null
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
                # Drop a possible partial first line from mid-file seek.
                $tailLines = $tail -split "`r?`n"
                if ($tailLines.Count -gt 1) { $tailLines = $tailLines[1..($tailLines.Count - 1)] }
                foreach ($tl in $tailLines) { $lines.Add($tl) | Out-Null }
            }
        }
        return @{ exit = $(if ($alive) { 0 } else { 1 }); lines = $lines }
    } -ArgumentList $root, $statePath

    $finished = Wait-Job -Job $statusJob -Timeout 15
    if (-not $finished) {
        Stop-Job $statusJob -ErrorAction SilentlyContinue
        Remove-Job $statusJob -Force -ErrorAction SilentlyContinue
        Write-Output "REFUSED -- -Status timed out after 15s (non-blocking by design)"
        Write-Output "  Prefer: Get-Content derived/ceiling_a/<session_id>/heartbeat.json"
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

if ($Resume) {
    if (-not $SessionId) {
        Write-Output "REFUSED -- -Resume requires -SessionId <uuid>"
        exit 1
    }
    $ck = Join-Path $root ("derived\ceiling_a\{0}\checkpoint.json" -f $SessionId)
    if (-not (Test-Path $ck)) {
        Write-Output "REFUSED -- no checkpoint at $ck"
        exit 1
    }
    # Checkpoint arms_selected is authoritative; -Arms must match if explicitly passed.
    $ckptObj = Get-Content $ck -Raw | ConvertFrom-Json
    if ($PSBoundParameters.ContainsKey("Arms")) {
        try {
            $armsForPy = (ConvertTo-StringList -Value $Arms -Name "Arms") -join ","
        } catch {
            Write-Output $_.Exception.Message
            exit 2
        }
    } elseif ($ckptObj.arms_selected) {
        $armsForPy = @($ckptObj.arms_selected) -join ","
    } else {
        $armsForPy = "A,A_prime"
    }
    Assert-No-Contending
    $mode = "remote"
    $launchContext = "ssh_detached"
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    $tag = "ceiling_a_resume_" + (Get-Date -Format "yyyyMMdd_HHmmss")
    $log = Join-Path $outDir "$tag.log"
    $resultPath = Join-Path $outDir "$tag.result.json"
    $inner = 'set SEAM_ISOLATION_MODE=' + $mode +
             '&& set SEAM_LAUNCH_CONTEXT=' + $launchContext +
             '&& "' + $py + '" -u -m seam.tools.ceiling_a --resume ' + $SessionId +
             ' --arms "' + $armsForPy + '"' +
             ' --allow-dirty --result-json "' + $resultPath + '"'
    $json = & (Join-Path $root "tools\spawn_detached.ps1") -CommandLine $inner -LogPath $log -WorkingDirectory $root
    $info = $json | ConvertFrom-Json
    $entry = [pscustomobject]@{
        tag             = $tag
        pid             = $info.pid
        parent_name     = $info.parent_name
        isolation_mode  = $mode
        launch_context  = $launchContext
        session_id      = $SessionId
        arms            = $armsForPy
        resumed         = $true
        launched_utc    = (Get-Date).ToUniversalTime().ToString("o")
        log_path        = $log
        result_path     = $resultPath
    }
    Save-Launches -Entries (@(Get-Launches) + $entry)
    Write-Output "launched detached ceiling_a RESUME"
    Write-Output "  session_id     : $SessionId"
    Write-Output "  arms           : $armsForPy"
    Write-Output "  pid            : $($info.pid)"
    Write-Output "  log            : $log"
    Write-Output "Poll: powershell -File tools/ceiling_a.ps1 -Status"
    exit 0
}

# ----------------------------------------------------------------------------------------------
if ($Local) {
    Write-Output "REFUSED -- the hours-long ceiling_a ladder is remote/ssh_detached only."
    Write-Output "  For harness validation under an interactive session (not a measurement):"
    Write-Output "    .venv-seam\Scripts\python.exe -u -m seam.tools.ceiling_a --smoke --allow-dirty"
    Write-Output "  For the real run, from the Mac with Cursor/browsers closed:"
    Write-Output "    ssh xps `"cd C:/Users/zjohn/Projects/gnn-hls-accel; powershell -File tools/ceiling_a.ps1 -Orchestrate`""
    Write-Output "    ssh xps `"cd C:/Users/zjohn/Projects/gnn-hls-accel; powershell -File tools/ceiling_a.ps1 -Orchestrate -Arms B,B_prime`""
    Write-Output "    ssh xps `"cd C:/Users/zjohn/Projects/gnn-hls-accel; powershell -File tools/ceiling_a.ps1 -Orchestrate -Arms gpu_only`""
    exit 1
}

if (-not $Orchestrate) {
    Write-Output "REFUSED -- pass -Orchestrate (remote gate path)"
    Write-Output "  Mac: ssh xps `"cd C:/Users/zjohn/Projects/gnn-hls-accel; powershell -File tools/ceiling_a.ps1 -Orchestrate`""
    Write-Output "  Mac: ssh xps `"cd C:/Users/zjohn/Projects/gnn-hls-accel; powershell -File tools/ceiling_a.ps1 -Orchestrate -Arms B,B_prime`""
    Write-Output "  Mac: ssh xps `"cd C:/Users/zjohn/Projects/gnn-hls-accel; powershell -File tools/ceiling_a.ps1 -Orchestrate -Arms gpu_only`""
    exit 1
}

$mode = "remote"
$launchContext = "ssh_detached"
Assert-No-Contending
try {
    $armsCsv = ([string[]](ConvertTo-StringList -Value $Arms -Name "Arms")) -join ","
} catch {
    Write-Output $_.Exception.Message
    exit 2
}

New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$tag = "ceiling_a_" + (Get-Date -Format "yyyyMMdd_HHmmss")
$log = Join-Path $outDir "$tag.log"
$resultPath = Join-Path $outDir "$tag.result.json"

# --allow-dirty: tree has been uncommitted since 2026-07-30; manifest records git_dirty.
# Quote arms CSV — never splice Object[] into the cmd line.
$inner = 'set SEAM_ISOLATION_MODE=' + $mode +
         '&& set SEAM_LAUNCH_CONTEXT=' + $launchContext +
         '&& "' + $py + '" -u -m seam.tools.ceiling_a --orchestrate' +
         ' --arms "' + $armsCsv + '"' +
         ' --allow-dirty --result-json "' + $resultPath + '"'

$json = & (Join-Path $root "tools\spawn_detached.ps1") -CommandLine $inner -LogPath $log -WorkingDirectory $root
$info = $json | ConvertFrom-Json

# session_id is assigned inside Python after launch; Status reads it from the log/heartbeat once present.
$entry = [pscustomobject]@{
    tag             = $tag
    pid             = $info.pid
    parent_name     = $info.parent_name
    isolation_mode  = $mode
    launch_context  = $launchContext
    session_id      = $null
    arms            = $armsCsv
    resumed         = $false
    launched_utc    = (Get-Date).ToUniversalTime().ToString("o")
    log_path        = $log
    result_path     = $resultPath
}
Save-Launches -Entries (@(Get-Launches) + $entry)

Write-Output "launched detached ceiling_a run"
Write-Output "  tag            : $tag"
Write-Output "  pid            : $($info.pid) (parent $($info.parent_name) -- not this session)"
Write-Output "  isolation_mode : $mode"
Write-Output "  launch_context : $launchContext"
Write-Output "  pre_run_settle : isolation.pre_run_settle_s inside detached job before first rung"
Write-Output "  arms           : $armsCsv"
Write-Output "  log            : $log"
Write-Output ""
Write-Output "Close this session. Do not touch the XPS. Poll with:"
Write-Output "  powershell -File tools/ceiling_a.ps1 -Status"
Write-Output ""
Write-Output "After the first log line with session_id, heartbeat is at:"
Write-Output "  derived/ceiling_a/<session_id>/heartbeat.json"
Write-Output "Resume (if the process died after some rungs):"
Write-Output "  powershell -File tools/ceiling_a.ps1 -Resume -SessionId <session_id>"
