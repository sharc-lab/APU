"""Telemetry and environment helpers for the Blade (discrete GPU) experiments.

Three background collectors, all flushed line by line:
  <prefix>_smi.csv     nvidia-smi --query-gpu ... -lms 1000  (has its own local-time timestamp column)
  <prefix>_dmon.txt    nvidia-smi dmon -s pucvmt -d 1 -o DT   (date + time columns, PCIe rx/tx)
  <prefix>_winctr.csv  one PowerShell Get-Counter -Continuous process per server start:
                       server-PID GPU Process Memory Dedicated/Shared Usage, \\Processor(_Total),
                       adapter-wide Shared Usage. Rows carry a UTC ISO timestamp and the server PID.

Only processes started here are ever terminated, by PID.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import threading
import time
from datetime import datetime, timezone

SMI_QUERY = ("timestamp,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu,"
             "memory.used,clocks_throttle_reasons.active")
WINCTR_HEADER = "ts_iso_utc,server_pid,dedicated_bytes,shared_bytes,cpu_total_pct,adapter_shared_bytes"

_WINCTR_PS = r"""
$ErrorActionPreference = 'SilentlyContinue'
$p = __PID__
$paths = @("\GPU Process Memory(pid_${p}_*)\Dedicated Usage","\GPU Process Memory(pid_${p}_*)\Shared Usage","\Processor(_Total)\% Processor Time","\GPU Adapter Memory(*)\Shared Usage")
Get-Counter -Counter $paths -SampleInterval 1 -Continuous | ForEach-Object {
  $s = $_.CounterSamples
  $ded = ($s | Where-Object { $_.Path -like '*gpu process memory*dedicated usage' } | Measure-Object -Property CookedValue -Sum).Sum
  $sh  = ($s | Where-Object { $_.Path -like '*gpu process memory*shared usage' } | Measure-Object -Property CookedValue -Sum).Sum
  $cpu = ($s | Where-Object { $_.Path -like '*processor(_total)*' } | Select-Object -First 1).CookedValue
  $ad  = ($s | Where-Object { $_.Path -like '*gpu adapter memory*shared usage' } | Measure-Object -Property CookedValue -Sum).Sum
  [Console]::Out.WriteLine(([DateTime]::UtcNow.ToString('o')) + ",$p,$ded,$sh,$cpu,$ad")
  [Console]::Out.Flush()
}
"""


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ps(cmd: str, timeout: int = 30) -> str:
    """Run a PowerShell command. Never raises; returns '' on any failure."""
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True,
                           timeout=timeout, stdin=subprocess.DEVNULL)
        return r.stdout
    except Exception:
        return ""


def listener_pid(port: int):
    out = ps(f"(Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue | "
             "Select-Object -First 1).OwningProcess", timeout=20).strip()
    return int(out) if out.isdigit() else None


def pid_alive(pid: int) -> bool:
    out = ps(f"(Get-Process -Id {pid} -ErrorAction SilentlyContinue | Measure-Object).Count", timeout=20).strip()
    return out not in ("", "0")


def gpu_temp():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10, stdin=subprocess.DEVNULL)
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return None


def process_info(pid: int) -> dict:
    """Command line, private bytes and working set of one process."""
    out = ps(
        f"$w = Get-CimInstance Win32_Process -Filter 'ProcessId={pid}'; "
        f"$g = Get-Process -Id {pid} -ErrorAction SilentlyContinue; "
        "[pscustomobject]@{cmd=$w.CommandLine; private=$g.PrivateMemorySize64; ws=$g.WorkingSet64} | ConvertTo-Json",
        timeout=30)
    try:
        d = json.loads(out)
        return {"command_line": d.get("cmd"), "private_bytes": d.get("private"), "working_set_bytes": d.get("ws")}
    except Exception:
        return {"command_line": None, "private_bytes": None, "working_set_bytes": None}


class Telemetry:
    def __init__(self, prefix: str):
        self.prefix = prefix
        self.smi_path = prefix + "_smi.csv"
        self.dmon_path = prefix + "_dmon.txt"
        self.winctr_path = prefix + "_winctr.csv"
        self._procs = {}
        self._threads = []
        self._winctr_header_written = False

    def _pump(self, proc, path):
        def run():
            with open(path, "a", encoding="utf-8") as f:
                for line in proc.stdout:
                    f.write(line if line.endswith("\n") else line + "\n")
                    f.flush()
        t = threading.Thread(target=run, daemon=True)
        t.start()
        self._threads.append(t)

    def _popen(self, argv):
        return subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                                text=True, bufsize=1)

    def start_gpu(self):
        smi = self._popen(["nvidia-smi", f"--query-gpu={SMI_QUERY}", "--format=csv", "-lms", "1000"])
        dmon = self._popen(["nvidia-smi", "dmon", "-s", "pucvmt", "-d", "1", "-o", "DT"])
        self._procs["smi"], self._procs["dmon"] = smi, dmon
        self._pump(smi, self.smi_path)
        self._pump(dmon, self.dmon_path)

    def start_winctr(self, server_pid: int):
        self.stop_winctr()
        if not self._winctr_header_written:
            with open(self.winctr_path, "a", encoding="utf-8") as f:
                f.write(WINCTR_HEADER + "\n")
            self._winctr_header_written = True
        script = _WINCTR_PS.replace("__PID__", str(server_pid))
        enc = base64.b64encode(script.encode("utf-16-le")).decode()
        p = self._popen(["powershell", "-NoProfile", "-EncodedCommand", enc])
        self._procs["winctr"] = p
        self._pump(p, self.winctr_path)

    def stop_winctr(self):
        p = self._procs.pop("winctr", None)
        if p is not None:
            _kill(p.pid)

    def stop(self):
        self.stop_winctr()
        for k in ("smi", "dmon"):
            p = self._procs.pop(k, None)
            if p is not None:
                _kill(p.pid)


def _kill(pid: int):
    try:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, timeout=20,
                       stdin=subprocess.DEVNULL)
    except Exception:
        pass


def require_ac():
    ac = ps("(Get-CimInstance -Namespace root\\wmi -ClassName BatteryStatus -ErrorAction SilentlyContinue | "
            "Select-Object -First 1).PowerOnline").strip()
    if ac.lower() != "true":
        raise RuntimeError(f"NOT ON AC POWER (PowerOnline={ac!r}). STOP.")


def environment_manifest() -> dict:
    """Records the run environment. Raises RuntimeError if the machine is not on AC power."""
    out = {"captured_utc": utcnow_iso()}
    out["hostname"] = os.environ.get("COMPUTERNAME")
    out["power_plan"] = ps("powercfg /getactivescheme").strip()
    ac = ps("(Get-CimInstance -Namespace root\\wmi -ClassName BatteryStatus -ErrorAction SilentlyContinue | "
            "Select-Object -First 1).PowerOnline").strip()
    bat = ps("(Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue | Select-Object -First 1).BatteryStatus").strip()
    out["ac_power_online"] = ac
    out["win32_battery_status_code"] = bat
    if ac.lower() != "true":
        raise RuntimeError(f"NOT ON AC POWER (PowerOnline={ac!r}, BatteryStatus={bat!r}). STOP.")
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=15, stdin=subprocess.DEVNULL)
        out["gpu"] = r.stdout.strip()
    except Exception as e:
        out["gpu"] = f"ERROR:{e}"
    out["cuda_sysmem_fallback_policy"] = ("UNKNOWN: not exposed by nvidia-smi and not read from the driver profile; "
                                          "never changed by this harness")
    out["os"] = ps("(Get-CimInstance Win32_OperatingSystem).Caption + ' ' + (Get-CimInstance Win32_OperatingSystem).Version").strip()
    out["logical_processors"] = os.cpu_count()
    out["utc_offset_hours_local"] = -time.timezone / 3600 + (1 if time.localtime().tm_isdst > 0 else 0)
    cpu = ps(
        "$n=[Environment]::ProcessorCount; $c=(Get-Counter '\\Process(*)\\% Processor Time' -SampleInterval 1 -MaxSamples 3).CounterSamples | "
        "Group-Object InstanceName | ForEach-Object { [pscustomobject]@{name=$_.Name; pct=[math]::Round((($_.Group | Measure-Object CookedValue -Average).Average)/$n,1)} } | "
        "Where-Object { $_.pct -gt 5 -and $_.name -ne '_total' -and $_.name -ne 'idle' } | ConvertTo-Json -Compress",
        timeout=40).strip()
    try:
        parsed = json.loads(cpu) if cpu else []
        out["processes_over_5pct_cpu_before_start"] = parsed if isinstance(parsed, list) else [parsed]
    except Exception:
        out["processes_over_5pct_cpu_before_start"] = f"UNPARSED:{cpu[:200]}"
    return out
