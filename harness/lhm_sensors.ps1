param(
    [Parameter(Mandatory = $true)][string]$Dll,
    [string]$OutFile = "",
    [int]$Seconds = 0
)
# Headless LibreHardwareMonitor reader (Windows PowerShell 5.1, .NET Framework build of the library).
# One JSON line per second: { ts, sensors: { "<type> | <hardware> | <sensor>": value }, hw: [ "<type>:<name>" ] }.
# Reads Temperature, Clock, Power and Load sensors (the four types the evo-x2 telemetry inventory positive controls
# need: CPU/iGPU clock, CPU/iGPU power, CPU/iGPU temperature, CPU/iGPU load). CPU and GPU hardware groups are enabled;
# motherboard, memory, storage, network, controller and PSU are not.
# Needs administrator rights and the PawnIO driver for the CPU sensors.
$ErrorActionPreference = 'Stop'
Add-Type -Path $Dll
$c = New-Object LibreHardwareMonitor.Hardware.Computer
$c.IsCpuEnabled = $true
$c.IsGpuEnabled = $true
$c.IsMotherboardEnabled = $false
$c.IsMemoryEnabled = $false
$c.IsStorageEnabled = $false
$c.IsNetworkEnabled = $false
$c.IsControllerEnabled = $false
$c.IsPsuEnabled = $false
$c.Open()
$wantedTypes = @('Temperature', 'Clock', 'Power', 'Load')
$t0 = [DateTime]::UtcNow
try {
    while ($true) {
        $tick = [DateTime]::UtcNow
        $sensors = [ordered]@{}
        $hw = @()
        foreach ($h in $c.Hardware) {
            $h.Update()
            $hw += ("{0}:{1}" -f $h.HardwareType, $h.Name)
            $all = @($h.Sensors)
            foreach ($sh in $h.SubHardware) { $sh.Update(); $all += @($sh.Sensors) }
            foreach ($s in $all) {
                if ($wantedTypes -contains $s.SensorType.ToString() -and $null -ne $s.Value) {
                    $sensors[("{0} | {1} | {2}" -f $s.SensorType, $h.Name, $s.Name)] = [math]::Round([double]$s.Value, 3)
                }
            }
        }
        $line = ([ordered]@{ ts = $tick.ToString('o'); sensors = $sensors; hw = $hw } | ConvertTo-Json -Compress -Depth 4)
        if ($OutFile) { Add-Content -Path $OutFile -Value $line } else { [Console]::Out.WriteLine($line); [Console]::Out.Flush() }
        if ($Seconds -gt 0 -and ($tick - $t0).TotalSeconds -ge $Seconds) { break }
        $left = 1000 - ([DateTime]::UtcNow - $tick).TotalMilliseconds
        if ($left -gt 0) { Start-Sleep -Milliseconds $left }
    }
}
finally {
    $c.Close()
}
