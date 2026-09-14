<#
.SYNOPSIS
  Run the iGPU enumeration probe in one named launch context.

.DESCRIPTION
  Diagnostic, not a measurement. Writes only under derived/igpu_probe/, seals nothing, gates
  nothing, and leaves the run ledger untouched. A negative result is a plumbing fact, not a
  finding about context capacity.

  The three contexts, same binary and same model-free code path in each:

    A  local_console    physically at the XPS, interactive desktop session
       powershell -File tools/probe_igpu.ps1 -LaunchContext local_console

    B  ssh_foreground   over ssh, blocking in the SSH session
       ssh xps "cd C:/Users/zjohn/Projects/gnn-hls-accel; powershell -File tools/probe_igpu.ps1 -LaunchContext ssh_foreground"

    C  ssh_detached     spawned through the WMI path verify_detach.ps1 proved survives disconnect
       ssh xps "cd C:/Users/zjohn/Projects/gnn-hls-accel; powershell -File tools/probe_igpu.ps1 -LaunchContext ssh_detached -Detached"
       ssh xps "cd C:/Users/zjohn/Projects/gnn-hls-accel; powershell -File tools/probe_igpu.ps1 -Show ssh_detached"

  C is the context that matters most and is the one most likely to fail: Win32_Process.Create
  parents the process to the WMI service, which can place it in session 0 with no window station.
  A display device that does not enumerate there would make an unattended arm B impossible via
  the only launch path that survives an SSH disconnect.

  The context is passed in, never inferred, so the three artifacts cannot be confused. The probe
  records the session and window station it actually landed in, so a mislabelled context is
  visible in the artifact rather than silently wrong.

  The interpreter is pinned via -PythonExe (default: the SEAM venv). Bare `python` is never
  invoked: over SSH, PATH often resolves to the Microsoft Store stub and exit 9009 would be
  indistinguishable from a GPU failure.
#>
[CmdletBinding(DefaultParameterSetName = "Run")]
param(
    [Parameter(ParameterSetName = "Run", Mandatory = $true)]
    [ValidateSet("local_console", "ssh_foreground", "ssh_detached")]
    [string]$LaunchContext,

    [Parameter(ParameterSetName = "Run")][switch]$Detached,

    [Parameter(ParameterSetName = "Show", Mandatory = $true)]
    [ValidateSet("local_console", "ssh_foreground", "ssh_detached")]
    [string]$Show,

    [Parameter(ParameterSetName = "Run")]
    [Parameter(ParameterSetName = "Show")]
    [string]$PythonExe = "C:\Users\zjohn\Projects\gnn-hls-accel\.venv-seam\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$root = "C:\Users\zjohn\Projects\gnn-hls-accel"
$script = Join-Path $root "tools\probe_igpu.py"
$outDir = Join-Path $root "derived\igpu_probe"

if ($Show) {
    $path = Join-Path $outDir "$Show.json"
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
$outPath = Join-Path $outDir "$LaunchContext.json"

# Resolve to absolute path so the WMI/cmd wrapper and the artifact agree on what ran.
try {
    $py = [System.IO.Path]::GetFullPath($PythonExe)
} catch {
    $py = $PythonExe
}

if (-not (Test-Path -LiteralPath $py)) {
    # INTERPRETER_MISSING must never be recordable as gpu_present=false.
    $missing = [ordered]@{
        hostname           = $env:COMPUTERNAME
        timestamp_utc      = (Get-Date).ToUniversalTime().ToString("o")
        launch_context     = $LaunchContext
        gpu_present        = $null
        gpu_probe_error    = $null
        error              = "INTERPRETER_MISSING"
        python_exe         = $py
        python_version     = $null
        python_in_venv     = $null
        ov_version         = $null
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

if ($Detached) {
    $log = Join-Path $outDir "$LaunchContext.log"
    # Absolute python path inside the cmd wrapper — never bare `python`.
    $inner = '"' + $py + '" -u "' + $script + '" --launch-context ' + $LaunchContext +
             ' --out "' + $outPath + '"'
    $json = & (Join-Path $root "tools\spawn_detached.ps1") `
        -CommandLine $inner -LogPath $log -WorkingDirectory $root
    $info = $json | ConvertFrom-Json
    Write-Output "spawned detached probe"
    Write-Output "  pid    : $($info.pid) (parent $($info.parent_name))"
    Write-Output "  artifact: $outPath"
    Write-Output "  log     : $log"
    Write-Output "  python  : $py"
    Write-Output ""
    Write-Output "Read it with:"
    Write-Output "  powershell -File tools/probe_igpu.ps1 -Show $LaunchContext"
    exit 0
}

& $py -u $script --launch-context $LaunchContext --out $outPath
exit $LASTEXITCODE
