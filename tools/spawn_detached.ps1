<#
.SYNOPSIS
  Spawn a process that is structurally incapable of being killed when an SSH session closes.

.DESCRIPTION
  Win32-OpenSSH places the processes of a session into a job object. Depending on version and
  configuration that job can be kill-on-close, which terminates every descendant of the remote
  shell the instant the connection drops -- including anything started with Start-Process, which
  remains a child of the shell and therefore a member of the same job.

  This launcher does not create a child at all. It asks the WMI service to create the process, so
  the new process is parented to WmiPrvSE.exe and was never a member of the SSH session's job
  object. There is nothing for a job teardown to reach.

  The cost is that Win32_Process.Create cannot redirect standard output, so the command is wrapped
  in cmd.exe to do the redirection itself. That is also why the caller must supply -LogPath: a
  detached process with nowhere to write its output is a process whose failure is invisible.

.OUTPUTS
  A single JSON object on stdout: pid, ppid, parent_name, mechanism, log_path.
  Exits non-zero if the spawn failed or the process was not observable afterwards.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$CommandLine,
    [Parameter(Mandatory = $true)][string]$LogPath,
    [string]$WorkingDirectory = "C:\Users\zjohn\Projects\gnn-hls-accel"
)

$ErrorActionPreference = "Stop"

$logDir = Split-Path -Parent $LogPath
if ($logDir -and -not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
}

# cmd.exe performs the redirection the WMI Create path cannot. The outer quote pair is consumed
# by cmd's /c argument parsing, so the inner command keeps its own quoting intact.
$wrapped = 'cmd.exe /c "' + $CommandLine + ' > "' + $LogPath + '" 2>&1"'

$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine      = $wrapped
    CurrentDirectory = $WorkingDirectory
}

if ($result.ReturnValue -ne 0) {
    Write-Error "Win32_Process.Create failed with ReturnValue=$($result.ReturnValue)"
    exit 1
}

$spawnedPid = [int]$result.ProcessId

# cmd.exe is transient -- it may already have handed off to the real process. Report on whichever
# of the two is still alive, and resolve the parent so the caller can see it is not this shell.
Start-Sleep -Milliseconds 400
$proc = Get-CimInstance Win32_Process -Filter "ProcessId = $spawnedPid" -ErrorAction SilentlyContinue

$ppid = $null
$parentName = $null
if ($proc) {
    $ppid = [int]$proc.ParentProcessId
    $parent = Get-CimInstance Win32_Process -Filter "ProcessId = $ppid" -ErrorAction SilentlyContinue
    if ($parent) { $parentName = $parent.Name }
}

[pscustomobject]@{
    pid           = $spawnedPid
    ppid          = $ppid
    parent_name   = $parentName
    spawner_pid   = $PID
    mechanism     = "Win32_Process.Create"
    log_path      = $LogPath
    command_line  = $wrapped
} | ConvertTo-Json -Compress
