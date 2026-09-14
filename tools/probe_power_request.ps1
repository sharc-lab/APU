<#
.SYNOPSIS
  Run the PowerRequestSystemRequired Modern Standby probe.

.DESCRIPTION
  Diagnostic, not a measurement. Writes only under derived/power_request_probe/,
  seals nothing. Discovers STANDBYIDLE, holds PowerCreateRequest/PowerSetRequest
  (PowerRequestSystemRequired), idles past the timeout, and checks for Kernel-Power
  506/507. Does not modify powercfg.

  Examples:
    powershell -File tools/probe_power_request.ps1
    powershell -File tools/probe_power_request.ps1 -Mode full
    powershell -File tools/probe_power_request.ps1 -Mode smoke
#>
[CmdletBinding()]
param(
    [ValidateSet("auto", "smoke", "full")]
    [string]$Mode = "auto",

    [double]$MarginS = 120.0,

    [double]$SmokeHoldS = 15.0,

    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not $PythonExe) {
    foreach ($rel in @(".venv-seam\Scripts\python.exe", ".venv\Scripts\python.exe")) {
        $candidate = Join-Path $Root $rel
        if (Test-Path $candidate) {
            $PythonExe = $candidate
            break
        }
    }
    if (-not $PythonExe) {
        $PythonExe = "python"
    }
}

$probe = Join-Path $Root "tools\probe_power_request.py"
Write-Host "probe_power_request: mode=$Mode python=$PythonExe"
& $PythonExe $probe --mode $Mode --margin-s $MarginS --smoke-hold-s $SmokeHoldS
exit $LASTEXITCODE
