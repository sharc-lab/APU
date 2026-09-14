"""Reproduce the sshd session teardown locally and see which launch mechanism survives it.

Win32-OpenSSH puts a session's processes into a job object. When the connection drops the job
handle closes, and if the job carries JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE every member dies at
once. That is the mechanism that would silently kill an unattended run, and it cannot be observed
by simply letting a parent shell exit normally -- a normal exit kills nothing.

This probe builds that exact situation without needing an SSH session:

  1. create a kill-on-close job and assign this process to it -- this stands in for the remote shell
  2. spawn one child with subprocess, the way Start-Process would -- it inherits the job
  3. spawn one child through the WMI service -- it is parented to WmiPrvSE and joins no job
  4. exit, closing the last job handle and triggering the teardown

Whichever heartbeat is still advancing afterwards belongs to a launch path that survives an SSH
disconnect. Run tools/verify_detach.ps1 semantics against the two PIDs written to the report file.
"""

from __future__ import annotations

import ctypes
import json
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path

JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOBOBJECTINFOCLASS_EXTENDED_LIMIT = 9

ROOT = Path(__file__).resolve().parents[1]
HEARTBEAT = ROOT / "tools" / "_detach_heartbeat.py"


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
        ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.POINTER(wintypes.ULONG)),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def spawn_via_subprocess(python: str, out_path: Path, duration_s: int) -> int:
    proc = subprocess.Popen(
        [python, "-u", str(HEARTBEAT), str(out_path), str(duration_s)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_CONSOLE,
    )
    return proc.pid


def spawn_via_wmi(python: str, out_path: Path, duration_s: int, log_path: Path) -> int:
    cmd = f'"{python}" -u "{HEARTBEAT}" "{out_path}" {duration_s}'
    ps = (
        "$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments "
        f"@{{CommandLine = 'cmd.exe /c \"' + '{cmd}' + ' > \"{log_path}\" 2>&1\"'; "
        f"CurrentDirectory = '{ROOT}'}}; "
        "Write-Output $r.ProcessId"
    )
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(out.stdout.strip().splitlines()[-1])


def main() -> int:
    duration_s = int(sys.argv[1]) if len(sys.argv) > 1 else 180
    out_dir = ROOT / "derived" / "detach_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "job_teardown_probe.json"
    python = sys.executable

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.IsProcessInJob.argtypes = [
        wintypes.HANDLE,
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.BOOL),
    ]

    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())

    info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel32.SetInformationJobObject(
        job, JOBOBJECTINFOCLASS_EXTENDED_LIMIT, ctypes.byref(info), ctypes.sizeof(info)
    ):
        raise ctypes.WinError(ctypes.get_last_error())

    if not kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess()):
        raise ctypes.WinError(ctypes.get_last_error())

    in_job = wintypes.BOOL()
    kernel32.IsProcessInJob(kernel32.GetCurrentProcess(), None, ctypes.byref(in_job))

    subprocess_hb = out_dir / "probe_subprocess.json"
    wmi_hb = out_dir / "probe_wmi.json"
    wmi_log = out_dir / "probe_wmi.log"

    subprocess_pid = spawn_via_subprocess(python, subprocess_hb, duration_s)
    wmi_pid = spawn_via_wmi(python, wmi_hb, duration_s, wmi_log)

    time.sleep(6.0)

    report = {
        "simulated_session_pid": kernel32.GetCurrentProcessId(),
        "simulated_session_in_job": bool(in_job.value),
        "job_limit_flags": "JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE",
        "duration_s": duration_s,
        "subprocess": {"pid": subprocess_pid, "heartbeat": str(subprocess_hb)},
        "wmi": {"pid": wmi_pid, "heartbeat": str(wmi_hb), "log": str(wmi_log)},
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))

    # Exiting drops the last handle to the job, which triggers the teardown under test.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
