<#
.SYNOPSIS
  CC cells (Task BJ): sample GPU and system memory counters about once a second until a stop
  file appears, and write them to a CSV outside the repo tree.
.DESCRIPTION
  Counters (PDH; on an iGPU the Dedicated counters read about 0, KB 4490156):
    \GPU Adapter Memory(*)\Shared Usage and Dedicated Usage
    \GPU Process Memory(*)\Shared Usage (instances pid_<pid>_luid_...; zero rows dropped)
    \Memory\Available MBytes, \Memory\Page Reads/sec (hard-fault reads), \Memory\Pages Input/sec
    \Paging File(_Total)\% Usage
  One Get-Counter call per loop (it takes about 1 s for the rate counters), then 0.5 s sleep.
  Rows: t_unix,path,value. The caller starts it with tools\spawn_detached.ps1 before a cell and
  creates the stop file after the cell. It also stops after -MaxHours. Read-only: it changes no
  setting and touches no process.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$OutCsv,
    [Parameter(Mandatory = $true)][string]$StopFile,
    [double]$MaxHours = 12
)
$ErrorActionPreference = "Continue"
$counters = @(
    '\GPU Adapter Memory(*)\Shared Usage',
    '\GPU Adapter Memory(*)\Dedicated Usage',
    '\GPU Process Memory(*)\Shared Usage',
    '\Memory\Available MBytes',
    '\Memory\Page Reads/sec',
    '\Memory\Pages Input/sec',
    '\Paging File(_Total)\% Usage'
)
$dir = Split-Path -Parent $OutCsv
if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Force $dir | Out-Null }
Set-Content -Path $OutCsv -Value "t_unix,path,value" -Encoding ascii
$deadline = (Get-Date).AddHours($MaxHours)
$n = 0
while (-not (Test-Path $StopFile) -and (Get-Date) -lt $deadline) {
    $t = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() / 1000.0
    $s = Get-Counter -Counter $counters -ErrorAction SilentlyContinue
    if ($s) {
        $rows = New-Object System.Collections.Generic.List[string]
        foreach ($c in $s.CounterSamples) {
            if ($c.Path -like '*gpu process memory*' -and $c.CookedValue -eq 0) { continue }
            $rows.Add(('{0:F3},{1},{2}' -f $t, $c.Path, $c.CookedValue))
        }
        if ($rows.Count -gt 0) { Add-Content -Path $OutCsv -Value $rows -Encoding ascii }
    } else {
        Add-Content -Path $OutCsv -Value ('{0:F3},ERROR,Get-Counter returned nothing' -f $t) -Encoding ascii
    }
    $n += 1
    Start-Sleep -Milliseconds 500
}
"CC_SIDECAR_DONE samples=$n stopfile=$(Test-Path $StopFile)"
