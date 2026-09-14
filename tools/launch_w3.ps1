# W-3 launcher — BFCL quality (axis 4) vs a621ff7d ordered entry-prefix pin.
#
# Production (Zach, one line, no args, bare SSH after cold boot — byte-identical
# defaults: NEntries=20, ModelSpec=Qwen3-4B-int8-ov.yaml):
#   powershell -NoProfile -File tools/launch_w3.ps1
#
# Scale / int4 arm (same launcher):
#   powershell -NoProfile -File tools/launch_w3.ps1 -NEntries 200
#   powershell -NoProfile -File tools/launch_w3.ps1 -NEntries 200 -ModelSpec configs\models\Qwen3-4B-int4-ov.yaml
#
# Dry-run (spawn suppressed; gates report-only; no measurement):
#   powershell -NoProfile -File tools/launch_w3.ps1 -DryRun -NEntries 200
#
# Steps: (1) non-persistent host clean  (2) five gates  (3) spawn_detached
#        (4) print run_id + artifact dir and exit without waiting.
#
# Payload: tools/run_w3_bfcl_quality.py → bfcl_feasibility_probe session_residency
# / gpu_only / RESIDENT. Entry ids: ordered PREFIX of a621ff7d (plan.entry_assert
# mode=prefix). Gold selftest must be n/n or refuse.
#
# Dual weight: the probe binds one --model-spec per process. int4 cannot share
# an int8 session; drive int4 via -ModelSpec on a separate launch.

[CmdletBinding()]
param(
    [switch]$DryRun,
    [int]$NEntries = 20,
    [string]$ModelSpec = ""
)

$ErrorActionPreference = "Stop"
$root = "C:\Users\zjohn\Projects\gnn-hls-accel"
Set-Location $root

# --- fixed W-3 coordinates (NEntries / ModelSpec overridable; defaults preserve no-arg path) ---
$Tag = "w3_bfcl_quality"
$Arm = "gpu_only"
$ResidencyMode = "RESIDENT"
$Seed = 20260810
$MaxNewTokens = 512
if ([string]::IsNullOrWhiteSpace($ModelSpec)) {
    $ModelSpec = Join-Path $root "configs\models\Qwen3-4B-int8-ov.yaml"
} elseif (-not [System.IO.Path]::IsPathRooted($ModelSpec)) {
    $ModelSpec = Join-Path $root $ModelSpec
}
$PythonExe = Join-Path $root ".venv-seam\Scripts\python.exe"
$WorkerPy = Join-Path $root "tools\run_w3_bfcl_quality.py"
$ProbePy = Join-Path $root "tools\bfcl_feasibility_probe.py"
$SpawnPs1 = Join-Path $root "tools\spawn_detached.ps1"
$SessionRoot = Join-Path $root "derived\bfcl_feasibility\w3_weight_quality"
$LaunchDir = Join-Path $SessionRoot "_launches"
$A621Entries = Join-Path $root "derived\bfcl_feasibility\session_residency\a621ff7d-2919-463d-aaf6-673f9e6bafbc\session_residency_entries_gpu_only_RESIDENT.json"

# Commissioning: Copilot+ OpenVINO Workload packages whose hosts respawn
# WorkloadsSessionHost after reboot (same non-persistent step as launch_w2).
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
    if ($DryRun) {
        return
    }
    exit 1
}

Write-Host "=== launch_w3.ps1 ==="
Write-Host ("mode           : {0}" -f $(if ($DryRun) { "DRY-RUN (spawn suppressed)" } else { "LIVE" }))
Write-Host ("cwd            : {0}" -f (Get-Location).Path)
Write-Host ("started_utc    : {0}" -f (Get-Date).ToUniversalTime().ToString("o"))
Write-Host ("NEntries       : {0}" -f $NEntries)
Write-Host ("ModelSpec      : {0}" -f $ModelSpec)
Write-Host ""
Write-Host "NOTE -- dual_weight: bfcl_feasibility_probe binds one --model-spec per process."
Write-Host "  int4 cannot run in the SAME session as int8. Drive int4 via -ModelSpec"
Write-Host "  on a separate launch; dual-spec is reported in plan.json, not worked around."
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
            $hr = $null
            if ($_.Exception -and $_.Exception.HResult) {
                $hr = ("0x{0:X8}" -f ([uint32]$_.Exception.HResult))
            }
            Write-Host ("REFUSED -- Remove-AppxPackage failed for {0}: {1} (hr={2})" -f `
                $pkg.PackageFullName, $_.Exception.Message, $hr)
            Write-Host "  Continuing: kill is the load-bearing non-persistent step; Appx remove may be package-in-use."
        }
    }
}

$availAfter = Get-AvailableMBytes
Write-Host ("Available MBytes AFTER  clean: {0:N1}  (delta={1:N1})" -f `
    $availAfter, ($availAfter - $availBefore))
Write-Host ""

# ---------------------------------------------------------------------------
# 2. Five gates (print all; refuse with named reason; no partial proceed live)
#
# Uptime < 2 h is CHOSEN, not derived. Measured basis: ~1 h clean vs ~22 h at
# 2.56x degradation (README_characterizations / late discovery), with nothing
# measured in between. Flag: PROVISIONAL pending the onset experiment that
# places the knee. Do not treat 2 h as an empirically fitted threshold.
# ---------------------------------------------------------------------------
Write-Host "=== 2. five gates ==="
Write-Host "GATE NOTE -- uptime < 2 h: CHOSEN / PROVISIONAL (not derived)."
Write-Host "  Basis: 1 h clean vs 22 h at 2.56x; no intervening onset curve yet."
Write-Host ""
$gateFails = New-Object System.Collections.Generic.List[string]

# 2a. uptime < 2 h (CHOSEN / PROVISIONAL — see GATE NOTE above)
$os = Get-CimInstance Win32_OperatingSystem
$boot = [datetime]$os.LastBootUpTime
$uptime = (Get-Date) - $boot
$uptimeS = [math]::Round($uptime.TotalSeconds, 3)
$uptimeOk = $uptime.TotalHours -lt 2.0
Write-Host ("gate uptime:     uptime_s={0}  hours={1:N2}  last_boot={2:o}  ok={3}  [CHOSEN/PROVISIONAL]" -f `
    $uptimeS, $uptime.TotalHours, $boot.ToUniversalTime(), $uptimeOk)
if (-not $uptimeOk) { $gateFails.Add("uptime_not_cold (>= 2 h since boot; gate is CHOSEN/PROVISIONAL)") | Out-Null }

# 2b. AC online
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

# 2c. Best Performance
$schemeLines = @(powercfg /getactivescheme)
$schemeText = ($schemeLines -join " ")
$planOk = ($schemeText -match "Best Performance")
Write-Host ("gate power_plan: {0}  ok={1}" -f $schemeText.Trim(), $planOk)
if (-not $planOk) { $gateFails.Add("power_plan_not_Best_Performance") | Out-Null }

# 2d. Available >= 7000
$availGate = Get-AvailableMBytes
$availOk = $availGate -ge 7000.0
Write-Host ("gate Available:  {0:N1} MB  floor=7000  ok={1}" -f $availGate, $availOk)
if (-not $availOk) { $gateFails.Add("Available_MBytes_below_7000") | Out-Null }

# 2e. tier-1 absent
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
        $pids = ($_.Group | Select-Object -ExpandProperty Id | Select-Object -First 8) -join ", "
        Write-Host ("  - {0} x{1} private={2:N0} MiB (pids {3})" -f $_.Name, $_.Count, $privateMb, $pids)
    }
    $gateFails.Add("tier1_resident") | Out-Null
}

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
# 3. Build resolved worker command; print; spawn (or dry-run preflight)
# ---------------------------------------------------------------------------
Write-Host "=== 3. resolved W-3 command ==="

if (-not (Test-Path -LiteralPath $WorkerPy)) {
    Refuse "W-3 worker missing: $WorkerPy"
    exit 2
}
if (-not (Test-Path -LiteralPath $ProbePy)) {
    Refuse "bfcl probe missing: $ProbePy"
    exit 2
}
if (-not (Test-Path -LiteralPath $SpawnPs1)) {
    Refuse "spawn_detached missing: $SpawnPs1"
    exit 2
}
if (-not (Test-Path -LiteralPath $PythonExe)) {
    Refuse "PythonExe missing: $PythonExe"
    exit 2
}
if (-not (Test-Path -LiteralPath $ModelSpec)) {
    Refuse "model spec missing: $ModelSpec"
    exit 2
}
if (-not (Test-Path -LiteralPath $A621Entries)) {
    Refuse "a621 entry pin missing: $A621Entries"
    exit 2
}

$sid = [guid]::NewGuid().ToString()
$tagLaunch = "w3_bfcl_" + (Get-Date -Format "yyyyMMdd_HHmmss")
New-Item -ItemType Directory -Force -Path $LaunchDir | Out-Null
$log = Join-Path $LaunchDir "$tagLaunch.log"
$artifactDir = Join-Path $SessionRoot $sid

# Detached worker: run_w3_bfcl_quality.py (asserts ids, gold 20/20, then arm).
$resolvedCmd = 'set SEAM_LAUNCH_CONTEXT=ssh_detached' +
    '&& "' + $PythonExe + '" -u "' + $WorkerPy + '"' +
    ' --session-id ' + $sid +
    ' --out "' + $artifactDir + '"' +
    ' --model-spec "' + $ModelSpec + '"' +
    ' --arm ' + $Arm +
    ' --residency-mode ' + $ResidencyMode +
    ' --n-entries ' + $NEntries +
    ' --seed ' + $Seed +
    ' --max-new-tokens ' + $MaxNewTokens +
    ' --a621-entries "' + $A621Entries + '"'

Write-Host "RESOLVED_CMD:"
Write-Host $resolvedCmd
Write-Host ""
Write-Host ("session_id (run_id) : {0}" -f $sid)
Write-Host ("artifact_dir        : {0}" -f $artifactDir)
Write-Host ("launch_log          : {0}" -f $log)
Write-Host ""

if ($DryRun) {
    Write-Host "=== DRY-RUN: entry-id prefix assert + gold selftest (no generation, no spawn) ==="
    $dryOut = Join-Path $SessionRoot ("{0}_dryrun_n{1}" -f $Tag, $NEntries)
    New-Item -ItemType Directory -Force -Path $dryOut | Out-Null
    $dryPy = Join-Path $dryOut "_dryrun_preflight.py"
    @"
import json
import sys
from pathlib import Path

sys.path.insert(0, r"$root")
import tools.bfcl_feasibility_probe as probe

a621 = Path(r"$A621Entries")
dry = Path(r"$dryOut")
n_entries = int($NEntries)
model_spec = r"$ModelSpec"
entries = probe.select_multi_turn_entries()[:n_entries]
pin = dry / probe.PINNED_MULTI_TURN_ENTRIES
pin.write_text(json.dumps(entries, indent=2, default=str) + "\n", encoding="utf-8")
got = [str(e["id"]) for e in entries]
ref = [str(e["id"]) for e in json.loads(a621.read_text(encoding="utf-8-sig"))]
entry_assert = {"mode": "prefix", "n_reference": len(ref), "n_run": len(got)}
print("PIN_PATH", pin)
print("MODEL_SPEC", model_spec)
print("ENTRY_ASSERT", json.dumps(entry_assert, sort_keys=True))
print("N_GOT", len(got), "N_REF", len(ref))
prefix_ok = got[: len(ref)] == ref
print("PREFIX_OK", prefix_ok)
print("FIRST20_MATCH_A621", got[:20] == ref[:20] if len(got) >= 20 and len(ref) >= 20 else False)
if not prefix_ok:
    for i, (a, b) in enumerate(zip(got, ref)):
        if a != b:
            print("FIRST_MISMATCH", i, a, b)
            break
    if len(got) < len(ref):
        print("LENGTH_SHORTFALL", len(got), len(ref))
    raise SystemExit(3)
print("IDS_PREFIX", ",".join(got[: min(20, len(got))]))
gold = probe.run_multi_turn_gold_selftest(entries)
(dry / "multi_turn_gold_selftest.json").write_text(
    json.dumps(gold, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
)
print("GOLD_N", gold.get("n"), "GOLD_N_VALID", gold.get("n_valid"))
print("GOLD_OK", gold.get("n_valid") == gold.get("n") == n_entries)
if gold.get("n_valid") != n_entries or gold.get("n") != n_entries:
    raise SystemExit(3)
print("DUAL_WEIGHT_SAME_SESSION", False)
print(
    "DUAL_WEIGHT_NOTE",
    "probe binds one --model-spec per process; int4 via separate -ModelSpec launch",
)
print("ENTRY_POPULATION_NOTE",
      "select_multi_turn_entries applies no difficulty or API filter; "
      "plain file-order prefix — never biased relative to the benchmark.")
"@ | Set-Content -LiteralPath $dryPy -Encoding utf8

    & $PythonExe -u $dryPy
    if ($LASTEXITCODE -ne 0) {
        Write-Host "REFUSED -- dry-run entry assert or gold selftest failed"
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

# LIVE: spawn detached, then exit without waiting.
Write-Host "=== 4. spawn_detached (live) ==="
New-Item -ItemType Directory -Force -Path $artifactDir | Out-Null
$json = & $SpawnPs1 -CommandLine $resolvedCmd -LogPath $log -WorkingDirectory $root
Write-Host $json
$info = $json | ConvertFrom-Json
Write-Host ""
Write-Host "launched W-3 detached"
Write-Host ("  run_id           : {0}" -f $sid)
Write-Host ("  artifact_dir     : {0}" -f $artifactDir)
Write-Host ("  pid              : {0} (parent {1})" -f $info.pid, $info.parent_name)
Write-Host ("  log              : {0}" -f $log)
Write-Host ""
Write-Host "Exiting launcher now. Close SSH. Poll:"
Write-Host ("  Get-Content {0}\plan.json" -f $artifactDir)
Write-Host ("  Get-Content {0}\w3_entry_ledger.json" -f $artifactDir)
exit 0
