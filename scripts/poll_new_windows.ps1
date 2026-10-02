# Enumerates visible top-level windows (MainWindowHandle != 0) once per second for the duration of a command,
# plus one snapshot immediately before starting and one immediately after it exits, and reports every window
# that was NOT present in the "before" snapshot. Built 2026-10-02 to prove the terminal-window fix: run the
# APU-SyncResults scheduled task's exact command line through this and the "new windows" count must be 0.
#
# Usage: powershell -File scripts\poll_new_windows.ps1 -Exe <exe> -Arguments <args as one string> -WorkingDirectory <dir>
param(
    [Parameter(Mandatory = $true)] [string]$Exe,
    [Parameter(Mandatory = $true)] [string]$Arguments,
    [Parameter(Mandatory = $true)] [string]$WorkingDirectory,
    [int]$PollIntervalMs = 500
)

function Snapshot {
    Get-Process | Where-Object { $_.MainWindowHandle -ne 0 } |
        ForEach-Object { "$($_.Id):$($_.ProcessName):$($_.MainWindowTitle)" }
}

$before = @(Snapshot)
$seenNew = New-Object System.Collections.Generic.HashSet[string]

$p = Start-Process -FilePath $Exe -ArgumentList $Arguments -WorkingDirectory $WorkingDirectory `
    -WindowStyle Hidden -PassThru

while (-not $p.HasExited) {
    foreach ($w in (Snapshot)) {
        if (-not ($before -contains $w) -and -not $seenNew.Contains($w)) {
            [void]$seenNew.Add($w)
        }
    }
    Start-Sleep -Milliseconds $PollIntervalMs
}

# one last check immediately after exit, in case a window appeared and closed between the last poll and exit
foreach ($w in (Snapshot)) {
    if (-not ($before -contains $w) -and -not $seenNew.Contains($w)) {
        [void]$seenNew.Add($w)
    }
}

$result = [ordered]@{
    exit_code         = $p.ExitCode
    new_window_count  = $seenNew.Count
    new_windows       = @($seenNew)
}
$result | ConvertTo-Json -Depth 3
