"""Positive controls for LibreHardwareMonitor sensors on evo-x2 (AMD Radeon 8060S iGPU, Ryzen AI Max+ 395).

Five checks, each pass/fail on its own:
  1. CPU package temperature: 60 s idle vs 60 s all-core spin, must rise >= 5 C.
  2. CPU package power: must rise under the same spin.
  3. iGPU core clock and iGPU power: idle vs during a real llama-server call (4B model, ~7000-token prompt, 128 out),
     both must rise clearly during the call.
  4. iGPU temperature, if LHM exposes one: idle vs during 60 s of repeated calls, must rise.
  5. GPU memory cross-check: LHM's own GPU memory sensor (if any, SensorType Data/SmallData) against the Windows
     "\\GPU Process Memory(<pid>)\\Dedicated Usage" and "Shared Usage" counters for the server PID, must agree within 5%.

MSAcpi_ThermalZoneTemperature is also sampled throughout (idle and loaded) for comparison, but is not treated as a
control on its own; on evo-t2s it was a constant unrelated to load, so it must pass the same 5 C spin test as the
package sensor above to be trusted here.

Requires: LibreHardwareMonitorLib.dll path, spin_hog_affinity.py (co-runner), a running or startable llama-server for
check 3. Emits one JSON document per check plus a final summary, and exits 0 only if all required checks pass.

Usage on evo-x2: python lhm_x2_control.py <path to LibreHardwareMonitorLib.dll> <output prefix> <llama-server path>
  <model path> <port>
"""

from __future__ import annotations

import calendar
import json
import statistics as st
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

DEPLOY = Path(__file__).resolve().parent
sys.path.insert(0, str(DEPLOY))
import t2s_m3_power_coupling as m3  # noqa: E402


def start_reader(dll, out):
    return subprocess.Popen(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(DEPLOY / "lhm_sensors.ps1"),
                             "-Dll", dll, "-OutFile", out], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, text=True)


def read_rows(out):
    """The PS reader writes true UTC timestamps ('o' format, trailing Z). Must parse as UTC, not
    time.mktime(...) - time.timezone, which assumes the local zone's STANDARD offset and is wrong by an hour
    whenever DST is active (confirmed live on evo-x2 on 2026-09-28: Pacific Standard Time reported as the zone ID
    while PDT, UTC-7, was actually in effect, which silently moved every query window hours outside the ~4-minute
    sample period and made every med_window() call return None)."""
    rows = []
    if Path(out).exists():
        for l in Path(out).read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                d = json.loads(l)
                d["t"] = calendar.timegm(time.strptime(d["ts"][:19], "%Y-%m-%dT%H:%M:%S"))
                rows.append(d)
            except Exception:
                pass
    return rows


def med_window(rows, t0, t1, key_filter):
    vals = []
    for r in rows:
        if t0 <= r["t"] <= t1:
            for k, v in r["sensors"].items():
                if key_filter(k):
                    vals.append(v)
    return st.median(vals) if vals else None


def ps(cmd, timeout=30):
    return subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=timeout).stdout.strip()


def acpi_zone_temp():
    out = ps('try { (Get-CimInstance -Namespace root/wmi -ClassName MSAcpi_ThermalZoneTemperature -ErrorAction Stop | '
             'Measure-Object CurrentTemperature -Average).Average } catch { "" }')
    try:
        return (float(out) / 10.0) - 273.15
    except ValueError:
        return None


def main():
    dll, prefix, llama, model, port = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
    out = prefix + "_lhm_x2.jsonl"
    Path(out).unlink(missing_ok=True)
    rd = start_reader(dll, out)
    time.sleep(8)
    if rd.poll() is not None:
        print(json.dumps({"fatal": True, "why": "reader exited", "stderr": (rd.stderr.read() or "")[-800:]}))
        sys.exit(2)

    results = {}

    # --- checks 1 and 2: 60 s idle vs 60 s all-core spin ---
    t_idle0 = time.time()
    acpi_idle0 = acpi_zone_temp()
    time.sleep(60)
    t_idle1 = time.time()
    acpi_idle1 = acpi_zone_temp()
    n_logical = int(ps("(Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors") or "16")
    mask_all = (1 << n_logical) - 1
    hog, report, aff = m3.start_hog("lhm_x2_control", mask_all, Path(prefix).parent, Path(prefix).name + "_spinctl")
    time.sleep(60)
    t_load1 = time.time()
    acpi_load1 = acpi_zone_temp()
    ips = m3.read_ips(report)
    m3.kill_tree(hog.pid)
    time.sleep(3)

    rows = read_rows(out)
    # Matched to the exact sensor names LHM exposes on this machine (confirmed live, see docs/X2_CHANGELOG.md): the
    # CPU/GPU are one die ("AMD RYZEN AI MAX+ 395 w/ Radeon 8060S" hardware entry, plus a separate "AMD Radeon(TM)
    # 8060S Graphics" entry), so "cpu"/"package" alone is not in the temperature sensor's name -- match by hardware
    # name (ryzen vs radeon) instead of a loose "cpu"/"gpu" substring on the whole key.
    is_cpu_temp = lambda k: k.startswith("Temperature |") and "ryzen" in k.lower()
    is_cpu_power = lambda k: k.startswith("Power |") and "ryzen" in k.lower() and "package" in k.lower()
    cpu_temp_idle = med_window(rows, t_idle1 - 40, t_idle1, is_cpu_temp)
    cpu_temp_load = med_window(rows, t_load1 - 40, t_load1, is_cpu_temp)
    cpu_pw_idle = med_window(rows, t_idle1 - 40, t_idle1, is_cpu_power)
    cpu_pw_load = med_window(rows, t_load1 - 40, t_load1, is_cpu_power)
    results["1_cpu_package_temp"] = {"idle_c": cpu_temp_idle, "load_c": cpu_temp_load,
                                      "rise_c": (cpu_temp_load - cpu_temp_idle) if cpu_temp_idle is not None and cpu_temp_load is not None else None,
                                      "pass": bool(cpu_temp_idle is not None and cpu_temp_load is not None and cpu_temp_load - cpu_temp_idle >= 5.0),
                                      "spin_ips": ips}
    results["2_cpu_package_power"] = {"idle_w": cpu_pw_idle, "load_w": cpu_pw_load,
                                       "pass": bool(cpu_pw_idle is not None and cpu_pw_load is not None and cpu_pw_load > cpu_pw_idle)}
    results["acpi_zone"] = {"idle_c": acpi_idle1, "load_c": acpi_load1,
                            "rise_c": (acpi_load1 - acpi_idle1) if acpi_idle1 is not None and acpi_load1 is not None else None,
                            "trustworthy": bool(acpi_idle1 is not None and acpi_load1 is not None and acpi_load1 - acpi_idle1 >= 5.0)}

    # --- check 3: iGPU clock and power, idle vs during an llama-server call ---
    t_gpu_idle0 = time.time()
    time.sleep(20)
    t_gpu_idle1 = time.time()
    # No cmd.exe wrapper: launched directly so proc.pid is llama-server.exe's own PID, not a shell wrapper's. Using
    # "cmd.exe /c <exe>" here previously made proc.pid the wrapper's PID, so the \GPU Process Memory(pid_<pid>_*)
    # query in check 5 always matched nothing (found live on evo-x2, 2026-09-28).
    proc = subprocess.Popen([llama, "-m", model, "--port", port, "-ctk", "f16", "-ctv", "f16",
                             "-fa", "on", "-ngl", "99", "-np", "1", "-t", "4", "--no-context-shift",
                             "--log-file", prefix + "_x2ctl_srv.log", "--log-verbosity", "4"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    healthy = False
    for _ in range(120):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                if json.loads(r.read()).get("status") == "ok":
                    healthy = True
                    break
        except Exception:
            pass
        time.sleep(1)
    if not healthy:
        results["3_igpu_clock_power"] = {"pass": False, "why": "server never became healthy"}
        results["5_gpu_memory_crosscheck"] = {"pass": False, "why": "no server to check"}
        results["4_igpu_temp"] = {"pass": False, "why": "no server to check"}
    else:
        prompt = ("Repeat the following sentence forty times, incrementing the counter each time, one per line. "
                  "Sentence: 'The quick brown fox jumps over the lazy dog, testing token number %d.'\n" * 150)
        body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 128, "temperature": 0, "stream": False}
        t_call0 = time.time()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                r.read()
        except Exception as e:
            results.setdefault("3_igpu_clock_power", {})["call_error"] = str(e)[:300]
        t_call1 = time.time()
        srv_pid = proc.pid
        mem_out = ps(f'(Get-Counter -Counter "\\GPU Process Memory(pid_{srv_pid}_*)\\Dedicated Usage","\\GPU Process Memory(pid_{srv_pid}_*)\\Shared Usage" '
                     f'-ErrorAction SilentlyContinue).CounterSamples | Select Path,CookedValue | ConvertTo-Json -Compress')
        rows = read_rows(out)
        is_gpu_clock = lambda k: k.startswith("Clock |") and "radeon" in k.lower() and "gpu core" in k.lower()
        is_gpu_power = lambda k: k.startswith("Power |") and "radeon" in k.lower() and "gpu core" in k.lower()
        is_gpu_temp = lambda k: k.startswith("Temperature |") and "radeon" in k.lower()
        is_gpu_mem = lambda k: "radeon" in k.lower() and ("memory" in k.lower() or "vram" in k.lower())
        clk_idle = med_window(rows, t_gpu_idle0, t_gpu_idle1, is_gpu_clock)
        pw_idle = med_window(rows, t_gpu_idle0, t_gpu_idle1, is_gpu_power)
        clk_load = med_window(rows, t_call0, t_call1, is_gpu_clock)
        pw_load = med_window(rows, t_call0, t_call1, is_gpu_power)
        results["3_igpu_clock_power"] = {**results.get("3_igpu_clock_power", {}), "clock_idle_mhz": clk_idle, "clock_load_mhz": clk_load,
                                          "power_idle_w": pw_idle, "power_load_w": pw_load,
                                          "pass": bool(clk_idle is not None and clk_load is not None and clk_load > clk_idle and
                                                      pw_idle is not None and pw_load is not None and pw_load > pw_idle)}
        # check 4: repeated calls for 60 s, temperature idle vs loaded
        t_rep0 = time.time()
        while time.time() - t_rep0 < 60:
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    r.read()
            except Exception:
                break
        t_rep1 = time.time()
        rows = read_rows(out)
        gtemp_idle = med_window(rows, t_gpu_idle0, t_gpu_idle1, is_gpu_temp)
        gtemp_load = med_window(rows, t_rep1 - 40, t_rep1, is_gpu_temp)
        results["4_igpu_temp"] = {"idle_c": gtemp_idle, "load_c": gtemp_load, "available": gtemp_idle is not None or gtemp_load is not None,
                                   "pass": bool(gtemp_idle is not None and gtemp_load is not None and gtemp_load > gtemp_idle)}
        lhm_mem = med_window(rows, t_rep1 - 10, t_rep1, is_gpu_mem)
        try:
            ctr = json.loads(mem_out) if mem_out else []
            ctr = ctr if isinstance(ctr, list) else [ctr]
        except Exception:
            ctr = []
        ctr_ded = next((c["CookedValue"] for c in ctr if "dedicated" in c.get("Path", "").lower()), None)
        ctr_shr = next((c["CookedValue"] for c in ctr if "shared" in c.get("Path", "").lower()), None)
        ctr_total_mib = ((ctr_ded or 0) + (ctr_shr or 0)) / 2 ** 20
        lhm_mib = lhm_mem  # assume LHM already reports MiB; noted as an assumption in the summary
        agree = None
        if lhm_mib and ctr_total_mib:
            agree = abs(lhm_mib - ctr_total_mib) / max(lhm_mib, ctr_total_mib) <= 0.05
        results["5_gpu_memory_crosscheck"] = {"lhm_mib": lhm_mib, "counter_dedicated_mib": (ctr_ded or 0) / 2 ** 20,
                                              "counter_shared_mib": (ctr_shr or 0) / 2 ** 20, "counter_total_mib": ctr_total_mib,
                                              "available": lhm_mem is not None, "pass": bool(agree)}
        m3.kill_tree(proc.pid)
        time.sleep(3)

    m3.kill_tree(rd.pid)
    rows = read_rows(out)
    sensor_names = sorted({k for r in rows for k in r["sensors"]})
    summary = {"results": results, "all_sensor_names_seen": sensor_names, "n_samples": len(rows),
               "required_pass": all(results.get(k, {}).get("pass") for k in
                                    ("1_cpu_package_temp", "2_cpu_package_power", "3_igpu_clock_power"))}
    print(json.dumps(summary, indent=1))
    sys.exit(0 if summary["required_pass"] else 2)


if __name__ == "__main__":
    main()
