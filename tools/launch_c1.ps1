# C-1 launcher — KV-precision context ceiling (binary search), int4 / gpu_only_*.
#
# Production (Zach, bare SSH after cold boot):
#   powershell -NoProfile -File tools/launch_c1.ps1
#
# Dry-run (spawn suppressed; gates report-only; no measurement):
#   powershell -NoProfile -File tools/launch_c1.ps1 -DryRun
#
# Steps: (1) non-persistent host clean  (2) five gates  (3) spawn_detached
#        (4) print run_id + artifact dir and exit without waiting.
#
# Payload: tools/run_c1_ceiling.py
#   Arms: gpu_only_f16 (pinned f16 readback), gpu_only_u8, gpu_only_u4
#   Search: low=12000 (known good), high=45000; bisect to +/- 250; 2 repeats
#   Pre-registers n_max from measured Available M before first probe.
#   Classifies every failure: memory_wall vs position_limit (verbatim).
#   If high passes (no failure in range): status=aborted,
#   abort_reason=no_ceiling_found_in_range — never report high as a ceiling.
#
# WSH watchdog interval 60 s (ENV_CHANGELOG).
# Corrections (AM-037): max_position_embeddings not enforced at inference;
#   16 GB host has no hard memory ceiling (OS pages; Available can hit 0).

[CmdletBinding()]
param(
    [switch]$DryRun,
    [string]$ModelSpec = "",
    [int]$Low = 12000,
    [int]$High = 45000,
    [int]$Resolution = 250,
    [int]$Repeats = 2
)

$ErrorActionPreference = "Stop"
$root = "C:\Users\zjohn\Projects\gnn-hls-accel"
Set-Location $root

$Tag = "c1_kv_ceiling"
$WatchdogIntervalS = 60
$ArmsCsv = "gpu_only_f16,gpu_only_u8,gpu_only_u4"
if ([string]::IsNullOrWhiteSpace($ModelSpec)) {
    $ModelSpec = Join-Path $root "configs\models\Qwen3-4B-int4-ov.yaml"
} elseif (-not [System.IO.Path]::IsPathRooted($ModelSpec)) {
    $ModelSpec = Join-Path $root $ModelSpec
}
$PythonExe = Join-Path $root ".venv-seam\Scripts\python.exe"
$WorkerPy = Join-Path $root "tools\run_c1_ceiling.py"
$SpawnPs1 = Join-Path $root "tools\spawn_detached.ps1"
$SessionRoot = Join-Path $root "derived\c1_ceiling"
$LaunchDir = Join-Path $SessionRoot "_launches"

$WorkloadAppxNames = @(
    "WindowsWorkload.EP.Intel.OpenVINO.1.8",
    "WindowsWorkload.EP.Intel.OpenVINO.Framework.1.8"
)

function Get-AvailableMBytes {
    try {
        $sample = (Get-Counter '\Memory\Available MBytes' -ErrorAction Stop).CounterSamples[0]
        return [double]$sample.CookedValue
    } catch {
        Write-Host "REFUSED -- Available MBytes unreadable: $($_.Exception.Message)"
        exit 2
    }
}

function Refuse {
    param([Parameter(Mandatory = $true)][string]$Reason)
    Write-Host ""
    Write-Host "REFUSED -- $Reason"
    if ($DryRun) { return }
    exit 1
}

Write-Host "=== launch_c1.ps1 ==="
Write-Host ("mode           : {0}" -f $(if ($DryRun) { "DRY-RUN (spawn suppressed)" } else { "LIVE" }))
Write-Host ("cwd            : {0}" -f (Get-Location).Path)
Write-Host ("started_utc    : {0}" -f (Get-Date).ToUniversalTime().ToString("o"))
Write-Host ("Arms           : {0}" -f $ArmsCsv)
Write-Host ("Search         : low={0} high={1} resolution={2} repeats={3}" -f $Low, $High, $Resolution, $Repeats)
Write-Host ("ModelSpec      : {0}" -f $ModelSpec)
Write-Host ("WatchdogIntervalS: {0}" -f $WatchdogIntervalS)
Write-Host ""
Write-Host "PRE-FLIGHT (recorded): prompt builder realizes n=45000 exactly;"
Write-Host "  gpu_only_f16 requests KV_CACHE_PRECISION=f16 with matched readback=f16"
Write-Host "  (not bare gpu_only / dynamic)."
Write-Host ""

# ---------------------------------------------------------------------------
# 1. Non-persistent host cleaning
# ---------------------------------------------------------------------------
Write-Host "=== 1. host cleaning (non-persistent) ==="
$availBefore = Get-AvailableMBytes
Write-Host ("Available MBytes BEFORE clean: {0:N1}" -f $availBefore)

$wsh = @(Get-Process -Name "WorkloadsSessionHost" -ErrorAction SilentlyContinue)
if ($wsh.Count -eq 0) {
    Write-Host "WorkloadsSessionHost: none resident"
} else {
    Write-Host ("WorkloadsSessionHost: killing {0} process(es) pids={1}" -f `
        $wsh.Count, (($wsh | ForEach-Object { $_.Id }) -join ","))
    if (-not $DryRun) {
        $wsh | Stop-Process -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
        $left = @(Get-Process -Name "WorkloadsSessionHost" -ErrorAction SilentlyContinue)
        if ($left.Count -gt 0) {
            $leftPids = (($left | ForEach-Object { $_.Id }) -join ",")
            Refuse "WorkloadsSessionHost still resident after Stop-Process (pids $leftPids)"
            if (-not $DryRun) { exit 1 }
        } else {
            Write-Host "WorkloadsSessionHost: cleared"
        }
    } else {
        Write-Host "DRY-RUN: would Stop-Process WorkloadsSessionHost (kill suppressed)"
    }
}

foreach ($pkgName in $WorkloadAppxNames) {
    $pkgs = @(Get-AppxPackage -Name $pkgName -ErrorAction SilentlyContinue)
    if ($pkgs.Count -eq 0) {
        Write-Host ("Appx {0}: not present" -f $pkgName)
        continue
    }
    foreach ($pkg in $pkgs) {
        Write-Host ("Appx present: {0}" -f $pkg.PackageFullName)
        if ($DryRun) {
            Write-Host ("DRY-RUN: would Remove-AppxPackage {0} (remove suppressed)" -f $pkg.PackageFullName)
            continue
        }
        try {
            Remove-AppxPackage -Package $pkg.PackageFullName -ErrorAction Stop
            Write-Host ("Appx removed: {0}" -f $pkg.PackageFullName)
        } catch {
            Write-Host ("Remove-AppxPackage failed for {0}: {1}" -f `
                $pkg.PackageFullName, $_.Exception.Message)
            Write-Host "  Continuing: kill is the load-bearing non-persistent step."
        }
    }
}

$availAfter = Get-AvailableMBytes
Write-Host ("Available MBytes AFTER  clean: {0:N1}  (delta={1:N1})" -f `
    $availAfter, ($availAfter - $availBefore))
Write-Host ""

# ---------------------------------------------------------------------------
# 2. Five gates
# ---------------------------------------------------------------------------
Write-Host "=== 2. five gates ==="
Write-Host "GATE NOTE -- uptime < 2 h: CHOSEN / PROVISIONAL (X-2 onset: no detectable"
Write-Host "  TTFT degradation in 0-2.25 h window; gate not yet re-derived)."
Write-Host ""
$gateFails = New-Object System.Collections.Generic.List[string]

$os = Get-CimInstance Win32_OperatingSystem
$boot = [datetime]$os.LastBootUpTime
$uptime = (Get-Date) - $boot
$uptimeS = [math]::Round($uptime.TotalSeconds, 3)
$uptimeOk = $uptime.TotalHours -lt 2.0
Write-Host ("gate uptime:     uptime_s={0}  hours={1:N2}  ok={2}  [CHOSEN/PROVISIONAL]" -f `
    $uptimeS, $uptime.TotalHours, $uptimeOk)
if (-not $uptimeOk) { $gateFails.Add("uptime_not_cold (>= 2 h since boot; gate is CHOSEN/PROVISIONAL)") | Out-Null }

$batt = @(Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue)
if ($batt.Count -eq 0) {
    $acOk = $true
    Write-Host "gate AC:         ok=True (no battery; assume AC)"
} else {
    $statuses = @($batt | ForEach-Object { [int]$_.BatteryStatus })
    $acOk = ($statuses | Where-Object { $_ -ne 2 }).Count -eq 0
    Write-Host ("gate AC:         ok={0}  BatteryStatus={1}" -f $acOk, ($statuses -join ","))
}
if (-not $acOk) { $gateFails.Add("AC_offline") | Out-Null }

$schemeLines = @(powercfg /getactivescheme)
$schemeText = ($schemeLines -join " ")
$planOk = ($schemeText -match "Best Performance")
Write-Host ("gate power_plan: {0}  ok={1}" -f $schemeText.Trim(), $planOk)
if (-not $planOk) { $gateFails.Add("power_plan_not_Best_Performance") | Out-Null }

$availGate = Get-AvailableMBytes
$availOk = $availGate -ge 7000.0
Write-Host ("gate Available:  {0:N1} MB  floor=7000  ok={1}" -f $availGate, $availOk)
if (-not $availOk) { $gateFails.Add("Available_MBytes_below_7000") | Out-Null }

$tier1Names = @("Cursor", "chrome", "msedge", "claude", "vmmem")
$wanted = @{}
foreach ($n in $tier1Names) { $wanted[$n.ToLowerInvariant()] = $true }
$tier1 = @(Get-Process -ErrorAction SilentlyContinue | Where-Object {
    $wanted.ContainsKey($_.ProcessName.ToLowerInvariant())
})
$tier1Ok = $tier1.Count -eq 0
if ($tier1Ok) {
    Write-Host "gate tier1:      ok=True (Cursor/chrome/msedge/claude/vmmem absent)"
} else {
    Write-Host "gate tier1:      ok=False -- resident:"
    $tier1 | Group-Object ProcessName | ForEach-Object {
        $privateMb = ($_.Group | Measure-Object -Property PrivateMemorySize64 -Sum).Sum / 1MB
        Write-Host ("  - {0} x{1} private={2:N0} MiB" -f $_.Name, $_.Count, $privateMb)
    }
    $gateFails.Add("tier1_resident") | Out-Null
}

Write-Host ""
Write-Host ("NOTE -- WorkloadsSessionHost watchdog: worker re-kills every {0}s (respawn ~4 min)" -f $WatchdogIntervalS)
Write-Host ""
if ($gateFails.Count -gt 0) {
    Write-Host ("GATES FAILED ({0}):" -f $gateFails.Count)
    foreach ($r in $gateFails) { Write-Host ("  - {0}" -f $r) }
    if (-not $DryRun) {
        Write-Host ""
        Write-Host "REFUSED -- gates failed; no spawn"
        exit 1
    }
    Write-Host "DRY-RUN: continuing past gate failures (report-only)"
} else {
    Write-Host "GATES: all five PASS"
}
Write-Host ""

# ---------------------------------------------------------------------------
# 3. Resolved command
# ---------------------------------------------------------------------------
Write-Host "=== 3. resolved C-1 command ==="

foreach ($p in @($WorkerPy, $SpawnPs1, $PythonExe, $ModelSpec)) {
    if (-not (Test-Path -LiteralPath $p)) {
        Refuse "missing required path: $p"
        exit 2
    }
}

$sid = [guid]::NewGuid().ToString()
$tagLaunch = "c1_" + (Get-Date -Format "yyyyMMdd_HHmmss")
New-Item -ItemType Directory -Force -Path $LaunchDir | Out-Null
$log = Join-Path $LaunchDir "$tagLaunch.log"
$artifactDir = Join-Path $SessionRoot $sid

$resolvedCmd = 'set SEAM_LAUNCH_CONTEXT=ssh_detached' +
    '&& "' + $PythonExe + '" -u "' + $WorkerPy + '"' +
    ' --session-id ' + $sid +
    ' --out "' + $artifactDir + '"' +
    ' --model-spec "' + $ModelSpec + '"' +
    ' --arms ' + $ArmsCsv +
    ' --low ' + $Low +
    ' --high ' + $High +
    ' --resolution ' + $Resolution +
    ' --repeats ' + $Repeats +
    ' --watchdog-interval-s ' + $WatchdogIntervalS

Write-Host "RESOLVED_CMD:"
Write-Host $resolvedCmd
Write-Host ""
Write-Host ("session_id (run_id) : {0}" -f $sid)
Write-Host ("artifact_dir        : {0}" -f $artifactDir)
Write-Host ("launch_log          : {0}" -f $log)
Write-Host ""

# Pre-register prediction sketch at current Available (dry-run / live both print).
$M = $availGate
$W = 2290768181.0
$headroom = ($M * 1024.0 * 1024.0) - $W
Write-Host "=== pre-registered prediction sketch (this host Available now) ==="
Write-Host ("M_available_mb={0:N1}  W_ir_bytes={1}  headroom_B={2:N0}" -f $M, [int64]$W, $headroom)
foreach ($pair in @(
    @{ arm = "gpu_only_f16"; kw = 234827 },
    @{ arm = "gpu_only_u8";  kw = 171532 },
    @{ arm = "gpu_only_u4";  kw = 135000 }
)) {
    $nMem = $headroom / $pair.kw
    $nMax = [math]::Min($nMem, 40960.0)
    $bind = if ($nMem -lt 40960.0) { "MEMORY" } else { "POSITION" }
    $pred = if ($bind -eq "MEMORY") { [int][math]::Round($nMem) } else { 40960 }
    Write-Host ("  {0}: n_max_memory={1:N0}  binding={2}  predicted_ceiling={3}" -f `
        $pair.arm, $nMem, $bind, $pred)
}
Write-Host "PRIMARY: u8 and u4 same ceiling 40960 (position). FALSIFIED if u4 > u8 beyond +/-250."
Write-Host "SECONDARY: f16 memory-walls below 40960. FALSIFIED if f16 reaches position limit."
Write-Host "DECLARED RISK: fit on n<=12000 extrapolated ~3.4x; workspace term not constant (12% spread)."
Write-Host ""

if ($DryRun) {
    Write-Host "=== DRY-RUN: prompt 12000+45000 + arm pin check (no generation, no spawn) ==="
    $dryOut = Join-Path $SessionRoot ("{0}_dryrun" -f $Tag)
    New-Item -ItemType Directory -Force -Path $dryOut | Out-Null
    $dryPy = Join-Path $dryOut "_dryrun_preflight.py"
    @"
import json, sys
from pathlib import Path
sys.path.insert(0, r"$root")
from transformers import AutoTokenizer
from seam.config import resolve_config
from seam.tools.delta_n import build_exact_prompt, _PLATFORM_PATH, _MEASUREMENT_PATH, _DELTA_N_PATH
from tools.run_c1_ceiling import _predictions, ARMS

root = Path(r"$root")
resolved = resolve_config(
    [root / _PLATFORM_PATH, root / _MEASUREMENT_PATH, root / _DELTA_N_PATH],
    repo_root=root,
)
cfg = resolved.data
arms_by_id = {a["id"]: a for a in cfg["arms"]}
for aid in ARMS:
    assert aid in arms_by_id, aid
props = arms_by_id["gpu_only_f16"].get("properties") or {}
assert str(props.get("KV_CACHE_PRECISION")).lower() == "f16", props
print("ARM_PIN_OK gpu_only_f16 KV_CACHE_PRECISION=f16")
tok = AutoTokenizer.from_pretrained(str(root / "models" / "Qwen3-4B-int4-ov"))
unit = cfg["ladder"]["filler_unit"]
for n in (12000, 45000):
    text, realized = build_exact_prompt(tok, target_tokens=n, unit=unit)
    assert realized == n, (n, realized)
    print("PROMPT_OK", n, "chars", len(text))
pred = _predictions(float($M))
print("PREDICTIONS", json.dumps(pred["arms"], indent=2, sort_keys=True))
print("PRIMARY", pred["primary_prediction"]["claim"][:80], "...")
print("DRYRUN_OK")
"@ | Set-Content -LiteralPath $dryPy -Encoding utf8

    & $PythonExe -u $dryPy
    if ($LASTEXITCODE -ne 0) {
        Write-Host "REFUSED -- dry-run preflight failed"
        exit $LASTEXITCODE
    }
    Write-Host ""
    Write-Host "=== DRY-RUN gate refusal summary ==="
    if ($gateFails.Count -eq 0) {
        Write-Host "Would refuse: (none -- all five gates PASS on this host right now)"
    } else {
        Write-Host "Would refuse:"
        foreach ($r in $gateFails) { Write-Host ("  - {0}" -f $r) }
    }
    Write-Host ""
    Write-Host "DRY-RUN complete. Spawn NOT executed. No measurement."
    exit 0
}

Write-Host "=== 4. spawn_detached (live) ==="
New-Item -ItemType Directory -Force -Path $artifactDir | Out-Null
$json = & $SpawnPs1 -CommandLine $resolvedCmd -LogPath $log -WorkingDirectory $root
Write-Host $json
$info = $json | ConvertFrom-Json
Write-Host ""
Write-Host "launched C-1 detached"
Write-Host ("  run_id           : {0}" -f $sid)
Write-Host ("  artifact_dir     : {0}" -f $artifactDir)
Write-Host ("  pid              : {0} (parent {1})" -f $info.pid, $info.parent_name)
Write-Host ("  log              : {0}" -f $log)
Write-Host ""
Write-Host "Exiting launcher now. Close SSH. Poll:"
Write-Host ("  Get-Content {0}\plan.json" -f $artifactDir)
Write-Host ("  Get-Content {0}\summary.json" -f $artifactDir)
exit 0
