<#
.SYNOPSIS
  A26 (Task BM): sample power and battery counters about once a second until a stop file
  appears; CSV outside the repo. Read-only.
.DESCRIPTION
  Counters, each only if the system exposes it (missing ones are recorded once as ERROR rows):
    \Power Meter(*)\Power            (milliwatts, where Windows exposes a meter)
    \Energy Meter(*)\Energy          (EMI energy counters, where exposed)
  WMI root\wmi BatteryStatus: DischargeRate, ChargeRate, RemainingCapacity, PowerOnline.
  Rows: t_unix,path,value.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$OutCsv,
    [Parameter(Mandatory = $true)][string]$StopFile,
    [double]$MaxHours = 6
)
$ErrorActionPreference = "Continue"
$dir = Split-Path -Parent $OutCsv
if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Force $dir | Out-Null }
Set-Content -Path $OutCsv -Value "t_unix,path,value" -Encoding ascii
$counters = @('\Power Meter(*)\Power', '\Energy Meter(*)\Energy')
$deadline = (Get-Date).AddHours($MaxHours)
$noted = @{}
while (-not (Test-Path $StopFile) -and (Get-Date) -lt $deadline) {
    $t = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() / 1000.0
    $rows = New-Object System.Collections.Generic.List[string]
    foreach ($c in $counters) {
        $s = Get-Counter -Counter $c -ErrorAction SilentlyContinue
        if ($s) {
            foreach ($x in $s.CounterSamples) { $rows.Add(('{0:F3},{1},{2}' -f $t, $x.Path, $x.CookedValue)) }
        } elseif (-not $noted.ContainsKey($c)) {
            $rows.Add(('{0:F3},ERROR,counter not exposed: {1}' -f $t, $c)); $noted[$c] = 1
        }
    }
    $b = Get-CimInstance -Namespace root\wmi -ClassName BatteryStatus -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($b) {
        foreach ($f in 'DischargeRate', 'ChargeRate', 'RemainingCapacity', 'PowerOnline') { $rows.Add(('{0:F3},battery\{1},{2}' -f $t, $f, $b.$f)) }
    }
    if ($rows.Count -gt 0) { Add-Content -Path $OutCsv -Value $rows -Encoding ascii }
    Start-Sleep -Milliseconds 500
}
"CC_POWER_SIDECAR_DONE"
