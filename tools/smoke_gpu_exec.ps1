<#
.SYNOPSIS
  Run an arm GPU/CPU compile+execute smoke in one named launch context.

.DESCRIPTION
  Diagnostic, not a measurement. Writes only under derived/gpu_smoke/, seals nothing, gates
  nothing, and leaves the run ledger untouched. Completing compile+execute for an arm device
  configuration from configs/delta_n.yaml is the question; capacity is not. With -N, also
  records prefill_s / decode_tok_s / wall_s (max_new_tokens raised to 64).

  Enumeration (probe_igpu) is not execution. openvino#34390 (CL_INVALID_WORK_GROUP_SIZE)
  manifests at kernel compilation; this smoke is the first exercise of that path under each
  launch context that matters.

  Contexts (ssh_detached skipped: session-equivalent to ssh_foreground per probe_igpu):

    A  local_console    physically at the XPS, interactive desktop session
       powershell -File tools/smoke_gpu_exec.ps1 -LaunchContext local_console

    B  ssh_foreground   over ssh, blocking in the SSH session
       ssh xps "cd C:/Users/zjohn/Projects/gnn-hls-accel; powershell -File tools/smoke_gpu_exec.ps1 -LaunchContext ssh_foreground"

  Optional:
    -Arm <id>    arm id from configs/delta_n.yaml (default B); resolved against yaml
    -N <tokens>  exact-N ladder prompt (filler_unit from delta_n.yaml); omit for short fixed prompt
    -Repeat <i>  append _r<i> to the artifact name so matrix cells do not overwrite

  Artifact names include arm and N so runs do not overwrite:
    derived/gpu_smoke/local_console_armB.json
    derived/gpu_smoke/local_console_armB_n2048.json
    derived/gpu_smoke/ssh_foreground_armA_n2000_r0.json

  The context is passed in, never inferred. The interpreter is pinned via -PythonExe (default:
  the SEAM venv). Bare `python` is never invoked: over SSH, PATH often resolves to the
  Microsoft Store stub and exit 9009 would be indistinguishable from a GPU failure.
#>
[CmdletBinding(DefaultParameterSetName = "Run")]
param(
    [Parameter(ParameterSetName = "Run", Mandatory = $true)]
    [ValidateSet("local_console", "ssh_foreground", "ssh_detached")]
    [string]$LaunchContext,

    [Parameter(ParameterSetName = "Show", Mandatory = $true)]
    [ValidateSet("local_console", "ssh_foreground", "ssh_detached")]
    [string]$Show,

    [Parameter(ParameterSetName = "Run")]
    [Parameter(ParameterSetName = "Show")]
    [string]$Arm = "B",

    [Parameter(ParameterSetName = "Run")]
    [Parameter(ParameterSetName = "Show")]
    [ValidateRange(1, [int]::MaxValue)]
    [int]$N = 0,

    [Parameter(ParameterSetName = "Run")]
    [Parameter(ParameterSetName = "Show")]
    [ValidateRange(0, [int]::MaxValue)]
    [int]$Repeat = -1,

    [Parameter(ParameterSetName = "Run")]
    [switch]$PromptOnly,

    [Parameter(ParameterSetName = "Run")]
    [Parameter(ParameterSetName = "Show")]
    [string]$ModelSpec = "",

    [Parameter(ParameterSetName = "Run")]
    [Parameter(ParameterSetName = "Show")]
    [string]$PythonExe = "C:\Users\zjohn\Projects\gnn-hls-accel\.venv-seam\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$root = "C:\Users\zjohn\Projects\gnn-hls-accel"
$script = Join-Path $root "tools\smoke_gpu_exec.py"
$outDir = Join-Path $root "derived\gpu_smoke"
$cfgPath = Join-Path $root "configs\delta_n.yaml"

function Get-SmokeArtifactName {
    param(
        [string]$Context,
        [string]$ArmId,
        [int]$Tokens,
        [int]$RepeatIndex
    )
    $name = if ($Tokens -gt 0) {
        "${Context}_arm${ArmId}_n${Tokens}"
    } else {
        "${Context}_arm${ArmId}"
    }
    if ($RepeatIndex -ge 0) {
        $name = "${name}_r${RepeatIndex}"
    }
    return "${name}.json"
}

function Assert-ArmId {
    param([string]$ArmId)
    if (-not (Test-Path -LiteralPath $cfgPath)) {
        Write-Output "REFUSED -- configs/delta_n.yaml missing at $cfgPath"
        exit 1
    }
    $raw = Get-Content -LiteralPath $cfgPath -Raw
    # Resolve arm ids from yaml without requiring a Python round-trip for -Show.
    $ids = [System.Collections.Generic.List[string]]::new()
    foreach ($line in ($raw -split "`n")) {
        if ($line -match '^\s*-\s*id:\s*(\S+)\s*$') {
            $ids.Add($Matches[1].Trim("'`""))
        }
    }
    if ($ids.Count -eq 0) {
        Write-Output "REFUSED -- no arm ids parsed from $cfgPath"
        exit 1
    }
    if ($ids -notcontains $ArmId) {
        Write-Output ("REFUSED -- -Arm {0} not in configs/delta_n.yaml arms: {1}" -f $ArmId, ($ids -join ", "))
        exit 1
    }
}

Assert-ArmId -ArmId $Arm

if ($Show) {
    $path = Join-Path $outDir (Get-SmokeArtifactName -Context $Show -ArmId $Arm -Tokens $N -RepeatIndex $Repeat)
    if (-not (Test-Path $path)) {
        Write-Output "no artifact yet at $path"
        $log = Join-Path $outDir "$Show.log"
        if (Test-Path $log) {
            Write-Output "--- $log ---"
            Get-Content $log -Tail 40
        }
        exit 1
    }
    Get-Content $path -Raw
    exit 0
}

New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$outName = Get-SmokeArtifactName -Context $LaunchContext -ArmId $Arm -Tokens $N -RepeatIndex $Repeat
$outPath = Join-Path $outDir $outName

# Resolve to absolute path so the artifact agrees on what ran.
try {
    $py = [System.IO.Path]::GetFullPath($PythonExe)
} catch {
    $py = $PythonExe
}

if (-not (Test-Path -LiteralPath $py)) {
    # INTERPRETER_MISSING must never be recordable as compile_ok/execute_ok=false.
    $missing = [ordered]@{
        hostname           = $env:COMPUTERNAME
        timestamp_utc      = (Get-Date).ToUniversalTime().ToString("o")
        launch_context     = $LaunchContext
        arm_id             = $Arm
        n_tokens           = $(if ($N -gt 0) { $N } else { $null })
        session_id         = $null
        window_station     = $null
        is_interactive     = $null
        device_config      = $null
        compile_ok         = $null
        compile_error      = $null
        execute_ok         = $null
        execute_error      = $null
        classification     = "INTERPRETER_MISSING"
        tokens_generated   = $null
        prefill_s          = $null
        decode_tok_s       = $null
        wall_s             = $null
        ov_version         = $null
        python_exe         = $py
        model_id           = $null
        peak_ws_bytes      = $null
        free_physical_mb_start = $null
        free_physical_at_peak = $null
        kv_bytes_expected  = $null
        wslock_granted     = $null
        power_request      = $null
        error              = "INTERPRETER_MISSING"
        path_head          = @(
            ($env:PATH -split ';' | Where-Object { $_ } | Select-Object -First 6)
        )
        diagnostics        = [ordered]@{
            message = "Pinned PythonExe does not exist; refusing to invoke bare python or a Store stub."
            python_exe_requested = $PythonExe
            python_exe_resolved  = $py
        }
    }
    $rendered = $missing | ConvertTo-Json -Depth 6
    Set-Content -LiteralPath $outPath -Value $rendered -Encoding utf8
    Write-Output $rendered
    exit 2
}

$pyArgs = @(
    "-u", $script,
    "--launch-context", $LaunchContext,
    "--arm", $Arm,
    "--out", $outPath
)
if ($N -gt 0) {
    $pyArgs += @("-N", "$N")
}
if ($PromptOnly) {
    $pyArgs += "--prompt-only"
}
if ($ModelSpec) {
    $pyArgs += @("--model-spec", $ModelSpec)
}

& $py @pyArgs
exit $LASTEXITCODE
