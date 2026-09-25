"""evo-t2s preflight for the overnight run. Read-only: changes no setting. Prints one JSON document and exits 0 if
every hard check passes, 2 otherwise. Hard checks: no other interactive session, AC power, free disk >= 70 GB,
the 4B GGUF sha256, and the llama-server binary present. Everything else is recorded, not judged.

Logical-processor classes come from GetSystemCpuSetInformation: EfficiencyClass separates P from the rest, and
LastLevelCacheIndex separates LP-E cores (no shared L3 with the compute tile) from E cores when it can.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

MODEL_4B = r"C:\apu\models\qwen3-4b-instruct-85e4a5b7.gguf"
MODEL_4B_SHA = "85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25fce54b18b9"
SERVER_BIN = r"C:\apu\bin\llama-b10970\llama-server.exe"
MIN_FREE_DISK_GB = 70


def ps(cmd, timeout=60):
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True,
                           timeout=timeout, stdin=subprocess.DEVNULL)
        return r.stdout.strip()
    except Exception as e:
        return f"ERROR {e}"


def cpu_sets():
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.GetSystemCpuSetInformation.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32),
                                             ctypes.c_void_p, ctypes.c_uint32]
    k.GetSystemCpuSetInformation.restype = ctypes.c_bool
    need = ctypes.c_uint32(0)
    k.GetSystemCpuSetInformation(None, 0, ctypes.byref(need), None, 0)
    buf = ctypes.create_string_buffer(need.value)
    k.GetSystemCpuSetInformation(buf, need.value, ctypes.byref(need), None, 0)
    out, off = [], 0
    raw = buf.raw
    while off < need.value:
        size = int.from_bytes(raw[off:off + 4], "little")
        typ = int.from_bytes(raw[off + 4:off + 8], "little")
        if typ == 0:
            out.append({"logical": raw[off + 14], "core": raw[off + 15], "llc": raw[off + 16], "numa": raw[off + 17],
                        "eff": raw[off + 18]})
        off += size
    return out


def classify(sets):
    top = max(s["eff"] for s in sets)
    p = [s for s in sets if s["eff"] == top]
    rest = [s for s in sets if s["eff"] != top]
    p_llc = {s["llc"] for s in p}
    e = [s for s in rest if s["llc"] in p_llc]
    lpe = [s for s in rest if s["llc"] not in p_llc]
    f = lambda xs: sorted(s["logical"] for s in xs)
    return {"P": f(p), "E": f(e), "LP_E": f(lpe), "raw": sets,
            "note": "LP-E separated from E by LastLevelCacheIndex; if E and LP-E share a class and cache index they cannot be told apart here"}


def main():
    out = {"captured_utc": datetime.now(timezone.utc).isoformat(), "hostname": socket.gethostname()}
    fails = []
    q = ps("try { (& query.exe user 2>&1) -join \"`n\" } catch { $_.Exception.Message }")
    out["query_user"] = q
    # "No User exists for *" means there is no interactive session at all. Otherwise line 1 is the header.
    sessions = [] if "No User exists" in q else [l for l in q.splitlines()[1:] if l.strip()]
    others = [l for l in sessions if not l.strip().lstrip(">").lower().startswith("sharc")]
    out["other_interactive_sessions"] = others
    if others:
        fails.append("another interactive session is logged in")
    out["cpu_name"] = ps("(Get-CimInstance Win32_Processor | Select-Object -First 1).Name")
    out["cpu_cores_logical"] = ps("(Get-CimInstance Win32_Processor | Select-Object -First 1).NumberOfLogicalProcessors")
    gpus = ps("Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion | ConvertTo-Json -Compress")
    out["gpus"] = gpus
    os_ = ps("$o=Get-CimInstance Win32_OperatingSystem; \"$($o.Caption) $($o.Version) build $($o.BuildNumber)\"")
    out["windows"] = os_
    out["power_scheme"] = ps("powercfg /getactivescheme")
    m = re.search(r"([0-9a-fA-F-]{36})", out["power_scheme"])
    out["power_scheme_guid"] = m.group(1) if m else None
    pq = ps("powercfg /query SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX")
    out["procthrottlemax_query"] = pq
    ac = re.search(r"Current AC Power Setting Index:\s*(0x[0-9a-fA-F]+)", pq)
    out["procthrottlemax_ac_percent"] = int(ac.group(1), 16) if ac else None
    acs = ps("(Get-CimInstance -Namespace root\\wmi -ClassName BatteryStatus -ErrorAction SilentlyContinue | Select-Object -First 1).PowerOnline")
    bat = ps("(Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue | Measure-Object).Count")
    out["ac_power_online_wmi"] = acs
    out["battery_devices"] = bat
    if bat not in ("0", "") and acs.lower() != "true":
        fails.append("on battery")
    du = shutil.disk_usage("C:\\")
    out["disk_free_gb"] = round(du.free / 2 ** 30, 1)
    if out["disk_free_gb"] < MIN_FREE_DISK_GB:
        fails.append(f"free disk {out['disk_free_gb']} GB < {MIN_FREE_DISK_GB}")
    out["ram_total_gb"] = ps("[math]::Round((Get-CimInstance Win32_OperatingSystem).TotalVisibleMemorySize/1MB,2)")
    out["ram_free_gb"] = ps("[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,2)")
    out["physical_memory_gb_smbios"] = ps("[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,2)")

    # iGPU memory budget, three ways (the third, the llama-server device line, is recorded by the orchestrator)
    out["gpu_adapter_memory_counters"] = ps(
        "(Get-Counter -ListSet 'GPU Adapter Memory' -ErrorAction SilentlyContinue).Counter -join '; '")
    out["gpu_adapter_memory_shared_limit"] = ps(
        "try { (Get-Counter '\\GPU Adapter Memory(*)\\Shared Limit' -ErrorAction Stop).CounterSamples | "
        "ForEach-Object { $_.Path + ' = ' + $_.CookedValue } } catch { 'counter not present' }")
    dx = os.path.join(tempfile.gettempdir(), "ovn_dxdiag.txt")
    try:
        subprocess.run(["dxdiag", "/t", dx], timeout=120, stdin=subprocess.DEVNULL)
        txt = open(dx, encoding="utf-8", errors="replace").read()
        out["dxdiag_memory_lines"] = [l.strip() for l in txt.splitlines()
                                      if re.search(r"(Dedicated Memory|Shared Memory|Display Memory|Card name)", l)]
    except Exception as e:
        out["dxdiag_memory_lines"] = f"ERROR {e}"
    out["intel_gmm_registry"] = ps("try { Get-ItemProperty 'HKLM:\\SOFTWARE\\Intel\\GMM' -ErrorAction Stop | "
                                   "ConvertTo-Json -Compress } catch { 'key not present' }")

    out["cpu_classes"] = classify(cpu_sets())

    out["server_bin_exists"] = os.path.exists(SERVER_BIN)
    if not out["server_bin_exists"]:
        fails.append("llama-server binary missing")
    if os.path.exists(MODEL_4B):
        h = hashlib.sha256()
        with open(MODEL_4B, "rb") as f:
            for c in iter(lambda: f.read(1 << 20), b""):
                h.update(c)
        out["model_4b_sha256"] = h.hexdigest()
        if h.hexdigest() != MODEL_4B_SHA:
            fails.append("4B sha256 mismatch")
    else:
        fails.append("4B GGUF missing")
    out["listeners_8385"] = ps("(Get-NetTCPConnection -LocalPort 8385 -State Listen -ErrorAction SilentlyContinue | Measure-Object).Count")
    out["llama_server_processes"] = ps("(Get-Process llama-server -ErrorAction SilentlyContinue | Measure-Object).Count")
    out["hard_check_failures"] = fails
    print(json.dumps(out, indent=1))
    sys.exit(2 if fails else 0)


if __name__ == "__main__":
    main()
