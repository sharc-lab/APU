<#
.SYNOPSIS
  Equivalence probe: bare gpu_only vs pinned gpu_only_f16 (12 cells).

.DESCRIPTION
  Sealed ceiling 2804d7fa and the 171.7 KB/token slope used bare gpu_only with
  no KV precision readback. This probe asks whether bare behaves like pinned f16.

  Grid (12 cells):
    arms      gpu_only, gpu_only_f16
    n_cached  2000, 8000, 16000
    delta     200
    mode      RESIDENT
    repeats   2
    order     interleaved bare/pinned per n_cached (A/B/A/B within each repeat)

  Emits one smoke_delta_prefill.json per cell under derived/delta_prefill/.
  Refuses if Available MBytes < isolation.pre_run_available_mb_min (7000).

  Modes:
    -DryRunGate   Available + tier-1 check only
    -Run          gate then 12 cells
    -ReportOnly   aggregate existing tag artifacts; no measurement
#>
[CmdletBinding()]
param(
    [switch]$DryRunGate,
    [switch]$Run,
    [switch]$ReportOnly,
    [string]$Tag = "f16_equiv",
    [string]$PythonExe = "C:\Users\zjohn\Projects\gnn-hls-accel\.venv-seam\Scripts\python.exe",
    [ValidateSet("local_console", "ssh_foreground", "ssh_detached")]
    [string]$LaunchContext = "ssh_foreground",
    [int]$CellTimeoutS = 1500,
    [switch]$SkipSettle
)

$ErrorActionPreference = "Stop"
$root = "C:\Users\zjohn\Projects\gnn-hls-accel"
. (Join-Path $root "tools\SeamPsCommon.ps1")

$cfgPath = Join-Path $root "configs\delta_n.yaml"
$smokePy = Join-Path $root "tools\smoke_delta_prefill.py"
$outRoot = Join-Path $root "derived\delta_prefill"

$arms = @("gpu_only", "gpu_only_f16")
$nCachedList = @(2000, 8000, 16000)
$delta = 200
$mode = "RESIDENT"
$repeats = 2

$modeCount = 0
if ($DryRunGate) { $modeCount++ }
if ($ReportOnly) { $modeCount++ }
if ($Run) { $modeCount++ }
if ($modeCount -eq 0) { $Run = $true; $modeCount = 1 }
if ($modeCount -ne 1) {
    Write-Host "REFUSED -- specify exactly one of -DryRunGate, -Run, or -ReportOnly"
    exit 2
}

function Get-IsolationScalar {
    param([string]$Path, [string]$Key)
    $lines = Get-Content -LiteralPath $Path -ErrorAction Stop
    $inIsolation = $false
    foreach ($line in $lines) {
        if ($line -match '^\s*isolation\s*:') { $inIsolation = $true; continue }
        if ($inIsolation -and $line -match '^\S') { break }
        if ($inIsolation -and $line -match ("^\s*{0}\s*:\s*(.+)\s*$" -f [regex]::Escape($Key))) {
            $raw = $Matches[1].Trim().Trim("'").Trim('"')
            if ($raw -match '^([^#]+)') { $raw = $Matches[1].Trim() }
            if ($raw -match '^\d+(\.\d+)?$') { return [double]$raw }
            return $raw
        }
    }
    return $null
}

function Get-AvailableMBytes {
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

$script:Tier1ContendingProcessNames = @(
    "Cursor", "Code", "chrome", "msedge", "firefox", "brave",
    "slack", "Discord", "Teams", "ms-teams", "Spotify", "OUTLOOK",
    "claude", "vmmem", "obsidian", "Docker Desktop"
)

function Get-ContendingProcesses {
    $hits = New-Object System.Collections.Generic.List[object]
    foreach ($name in $script:Tier1ContendingProcessNames) {
        Get-Process -Name $name -ErrorAction SilentlyContinue | ForEach-Object {
            $hits.Add([pscustomobject]@{
                    name  = $_.ProcessName
                    id    = $_.Id
                    ws_mb = [math]::Round($_.WorkingSet64 / 1MB, 1)
                }) | Out-Null
        }
    }
    return , @($hits.ToArray())
}

function Assert-PreRunAvailable {
    param([double]$MinMb, [string]$Citation, [switch]$ReportOnly)
    $avail = Get-AvailableMBytes
    $availText = if ($null -eq $avail.available_mb) { "null" } else { "{0:N1}" -f $avail.available_mb }
    # Write-Host: safe when caller assigns $ok = Assert-PreRunAvailable ...
    Write-Host ("Available MBytes : {0} (method={1})" -f $availText, $avail.method)
    Write-Host ("pre_run floor    : {0} MB" -f $MinMb)
    if ($Citation) { Write-Host ("citation         : {0}" -f $Citation) }
    if ($null -eq $avail.available_mb) {
        Write-Host "BLOCKED_ON_OPERATOR -- could not read \Memory\Available MBytes"
        if (-not $ReportOnly) { exit 2 }
        return $false
    }
    if ([double]$avail.available_mb -lt $MinMb) {
        Write-Host ("BLOCKED_ON_OPERATOR -- Available MBytes {0:N1} < {1}" -f $avail.available_mb, $MinMb)
        $contending = Get-ContendingProcesses
        if ($contending.Count -gt 0) {
            Write-Host "tier-1 residents:"
            foreach ($c in $contending) {
                Write-Host ("  {0} pid={1} ws_mb={2}" -f $c.name, $c.id, $c.ws_mb)
            }
        } else {
            Write-Host "No tier-1 process names resident; other residents still hold memory."
        }
        if (-not $ReportOnly) { exit 1 }
        return $false
    }
    return $true
}

function Get-MedianDouble {
    param([double[]]$Values)
    $s = @($Values | Sort-Object)
    $n = $s.Count
    if ($n -eq 0) { return $null }
    if ($n % 2 -eq 1) { return [double]$s[[int]([math]::Floor($n / 2))] }
    return ([double]$s[$n / 2 - 1] + [double]$s[$n / 2]) / 2.0
}

function Get-RelSpreadMax {
    param([double[]]$Values)
    if ($Values.Count -lt 2) { return $null }
    $med = Get-MedianDouble -Values $Values
    if ($null -eq $med -or $med -eq 0) { return $null }
    $devs = @($Values | ForEach-Object { [math]::Abs($_ - $med) / [math]::Abs($med) })
    return [double](($devs | Measure-Object -Maximum).Maximum)
}

function Get-CellPlan {
    $specs = New-Object System.Collections.Generic.List[object]
    $idx = 0
    for ($r = 0; $r -lt $repeats; $r++) {
        foreach ($nc in $nCachedList) {
            foreach ($arm in $arms) {
                $specs.Add([pscustomobject]@{
                        cell_index = $idx
                        repeat     = $r
                        arm        = $arm
                        n_cached   = [int]$nc
                        delta      = [int]$delta
                        mode       = $mode
                    }) | Out-Null
                $idx++
            }
        }
    }
    return , @($specs.ToArray())
}

function Invoke-Report {
    param([string]$TagName)
    $files = @(Get-ChildItem -LiteralPath $outRoot -Filter ("{0}_*.json" -f $TagName) -ErrorAction SilentlyContinue)
    if ($files.Count -eq 0) {
        $files = @(Get-ChildItem -Path $outRoot -Recurse -Filter ("{0}_*.json" -f $TagName) -ErrorAction SilentlyContinue)
    }
    Write-Host ("report tag={0} files={1}" -f $TagName, $files.Count)
    if ($files.Count -eq 0) {
        Write-Host "REFUSED -- no cell manifests found for tag"
        exit 2
    }

    $rows = New-Object System.Collections.Generic.List[object]
    foreach ($f in $files) {
        $j = Get-Content -LiteralPath $f.FullName -Raw -Encoding utf8 | ConvertFrom-Json
        if ($j.classification -ne "OK") {
            Write-Host ("  SKIP non-OK {0} classification={1}" -f $f.Name, $j.classification)
            continue
        }
        $rb = $null
        if ($j.kv_cache_precision_readback -and $j.kv_cache_precision_readback.Count -gt 0) {
            $rb = $j.kv_cache_precision_readback[0].readback.normalized
        }
        $rows.Add([pscustomobject]@{
                arm             = [string]$j.arm_id
                n_cached        = [int]$j.n_cached
                turn1_prefill_s = [double]$j.turn1.prefill_s
                turn2_prefill_s = [double]$j.turn2.prefill_s
                peak_ws_turn1   = [double]$j.turn1.peak_ws_bytes
                path            = $f.Name
                kv_bpt          = $j.kv_bytes_per_token
                readback        = $rb
            }) | Out-Null
    }

    Write-Host ""
    Write-Host "=== per-cell (OK only) ==="
    foreach ($row in ($rows | Sort-Object n_cached, arm, turn1_prefill_s)) {
        Write-Host ("  nc={0} arm={1} t1={2:N4}s t2={3:N4}s peak_ws1={4:N0} kv_bpt={5} readback={6}" -f `
            $row.n_cached, $row.arm, $row.turn1_prefill_s, $row.turn2_prefill_s, `
            $row.peak_ws_turn1, $row.kv_bpt, $row.readback)
    }

    Write-Host ""
    Write-Host "=== ratios bare/pinned by n_cached ==="
    $verdictParts = New-Object System.Collections.Generic.List[string]
    $anyDiverge = $false
    foreach ($nc in $nCachedList) {
        $bare = @($rows | Where-Object { $_.arm -eq "gpu_only" -and $_.n_cached -eq $nc })
        $pin = @($rows | Where-Object { $_.arm -eq "gpu_only_f16" -and $_.n_cached -eq $nc })
        if ($bare.Count -lt 1 -or $pin.Count -lt 1) {
            Write-Host ("  nc={0}: INCOMPLETE bare_n={1} pinned_n={2}" -f $nc, $bare.Count, $pin.Count)
            $anyDiverge = $true
            $verdictParts.Add(("nc{0}=incomplete" -f $nc)) | Out-Null
            continue
        }

        $metrics = @(
            @{ name = "turn1_prefill_s"; bare = [double[]]@($bare | ForEach-Object { $_.turn1_prefill_s }); pin = [double[]]@($pin | ForEach-Object { $_.turn1_prefill_s }) },
            @{ name = "turn2_prefill_s"; bare = [double[]]@($bare | ForEach-Object { $_.turn2_prefill_s }); pin = [double[]]@($pin | ForEach-Object { $_.turn2_prefill_s }) },
            @{ name = "peak_ws_turn1"; bare = [double[]]@($bare | ForEach-Object { $_.peak_ws_turn1 }); pin = [double[]]@($pin | ForEach-Object { $_.peak_ws_turn1 }) }
        )

        Write-Host ("  nc={0}:" -f $nc)
        $ncOk = $true
        foreach ($m in $metrics) {
            $mb = Get-MedianDouble -Values $m.bare
            $mp = Get-MedianDouble -Values $m.pin
            $sb = Get-RelSpreadMax -Values $m.bare
            $sp = Get-RelSpreadMax -Values $m.pin
            $ratio = if ($null -ne $mp -and $mp -ne 0) { $mb / $mp } else { $null }
            $spreadVals = @()
            if ($null -ne $sb) { $spreadVals += $sb }
            if ($null -ne $sp) { $spreadVals += $sp }
            $spreadMax = if ($spreadVals.Count -gt 0) { ($spreadVals | Measure-Object -Maximum).Maximum } else { $null }
            $tol = if ($null -ne $spreadMax) { [math]::Max([double]$spreadMax, 0.05) } else { 0.05 }
            $absDev = if ($null -ne $ratio) { [math]::Abs($ratio - 1.0) } else { $null }
            $okMetric = ($null -ne $absDev) -and ($absDev -le $tol)
            if (-not $okMetric) { $ncOk = $false; $anyDiverge = $true }
            $bareTxt = (($m.bare | ForEach-Object { "{0:G5}" -f $_ }) -join ",")
            $pinTxt = (($m.pin | ForEach-Object { "{0:G5}" -f $_ }) -join ",")
            Write-Host ("    {0}: bare_med={1:G6} pinned_med={2:G6} ratio={3:G4} |r-1|={4:G4} tol={5:G4} within_spread={6} bare=[{7}] pinned=[{8}]" -f `
                $m.name, $mb, $mp, $ratio, $absDev, $tol, $okMetric, $bareTxt, $pinTxt)
        }
        $verdictParts.Add(("nc{0}={1}" -f $nc, $(if ($ncOk) { "consistent" } else { "DIVERGE" }))) | Out-Null
    }

    Write-Host ""
    if ($anyDiverge) {
        Write-Host "VERDICT: DIVERGE -- bare gpu_only is NOT consistent with pinned gpu_only_f16 within run-to-run spread."
        Write-Host "STOP -- sealed 2804d7fa and 171.7 KB/token slope on bare gpu_only are SUSPECT pending re-measure under pinned f16."
        Write-Host ("detail: {0}" -f ($verdictParts -join "; "))
        exit 3
    }

    Write-Host "VERDICT: CONSISTENT -- bare/pinned ratios within the observed within-arm repeat spread (tol >= 5%)."
    Write-Host ("detail: {0}" -f ($verdictParts -join "; "))
    exit 0
}

$preRunMin = Get-IsolationScalar -Path $cfgPath -Key "pre_run_available_mb_min"
$preRunCite = Get-IsolationScalar -Path $cfgPath -Key "pre_run_available_mb_min_citation"
if (-not $preRunMin) { $preRunMin = 7000 }

if ($ReportOnly) {
    Invoke-Report -TagName $Tag
    exit $LASTEXITCODE
}

Write-Host "f16 equivalence probe"
Write-Host ("  arms      : {0}" -f ($arms -join ", "))
Write-Host ("  n_cached  : {0}" -f ($nCachedList -join ", "))
Write-Host ("  delta     : {0}" -f $delta)
Write-Host ("  mode      : {0}" -f $mode)
Write-Host ("  repeats   : {0}" -f $repeats)
Write-Host ("  cells     : 12 (interleaved bare/pinned per n)")
Write-Host ("  tag       : {0}" -f $Tag)

$contending = Get-ContendingProcesses
if ($contending.Count -gt 0) {
    Write-Host "tier-1 residents:"
    foreach ($c in $contending) {
        Write-Host ("  {0} pid={1} ws_mb={2}" -f $c.name, $c.id, $c.ws_mb)
    }
} else {
    Write-Host "tier1 refuse list: none resident"
}

$memOk = Assert-PreRunAvailable -MinMb ([double]$preRunMin) -Citation ([string]$preRunCite) -ReportOnly
$contOk = ($contending.Count -eq 0)

if ($DryRunGate) {
    if (-not $contOk -or -not $memOk) {
        Write-Host "dry-run FAIL -- BLOCKED_ON_OPERATOR (no cells started)"
        exit 1
    }
    Write-Host "dry-run PASS -- Available above floor and no tier-1 residents"
    exit 0
}

if (-not $memOk) {
    Write-Host "BLOCKED_ON_OPERATOR -- Available below floor; no cells started; no equivalence numbers invented"
    exit 1
}
if (-not $contOk) {
    Write-Host "BLOCKED_ON_OPERATOR -- tier-1 residents present; no cells started"
    exit 1
}

$settleS = Get-IsolationScalar -Path $cfgPath -Key "pre_run_settle_s"
if (-not $settleS) { $settleS = 0 }
if (-not $SkipSettle -and [double]$settleS -gt 0) {
    Write-Host ("Waiting pre_run_settle_s={0} ..." -f $settleS)
    Start-Sleep -Seconds ([double]$settleS)
    $memOk2 = Assert-PreRunAvailable -MinMb ([double]$preRunMin) -Citation ([string]$preRunCite) -ReportOnly
    if (-not $memOk2) {
        Write-Host "BLOCKED_ON_OPERATOR -- Available below floor after settle"
        exit 1
    }
}

$plan = Get-CellPlan
Write-Host ("plan cells={0}" -f $plan.Count)
$i = 0
foreach ($spec in $plan) {
    $i++
    Write-Host ("-- cell {0}/{1} arm={2} nc={3} d={4} r={5}" -f `
        $i, $plan.Count, $spec.arm, $spec.n_cached, $spec.delta, $spec.repeat)
    $argList = @(
        $smokePy,
        "--launch-context", $LaunchContext,
        "--arm", $spec.arm,
        "--mode", $spec.mode,
        "--n-cached", ([string]$spec.n_cached),
        "--delta", ([string]$spec.delta),
        "--repeat", ([string]$spec.repeat),
        "--tag", $Tag
    )
    $p = Start-Process -FilePath $PythonExe -ArgumentList $argList -NoNewWindow -PassThru -Wait
    if ($p.ExitCode -ne 0) {
        Write-Host ("REFUSED -- smoke exit {0} on cell {1}" -f $p.ExitCode, $i)
        exit 2
    }
}

Write-Host ""
Write-Host "=== post-run report ==="
Invoke-Report -TagName $Tag
exit $LASTEXITCODE
