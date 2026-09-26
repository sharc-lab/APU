"""Positive control for the LibreHardwareMonitor temperature sensor on evo-t2s.

60 s idle, then 60 s of the 12-core compute spin (co-runner mask 0xFFF0, the same load as the M3 nonp12 condition),
reading CPU sensors every second through harness/lhm_sensors.ps1. The package temperature must rise by at least 5 C
between the two windows (medians of the last 40 s of each), otherwise the sensor is not trustworthy and the power-proxy
gate stays. Prints one JSON document and exits 0 on pass, 2 on fail.

Usage on evo-t2s: python lhm_control.py <path to LibreHardwareMonitorLib.dll> <output prefix>
"""

from __future__ import annotations

import json
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

DEPLOY = Path(__file__).resolve().parent
sys.path.insert(0, str(DEPLOY))
import t2s_m3_power_coupling as m3  # noqa: E402

PKG_KEYS = ("package",)


def start_reader(dll, out):
    return subprocess.Popen(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(DEPLOY / "lhm_sensors.ps1"),
                             "-Dll", dll, "-OutFile", out], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, text=True)


def read(out):
    rows = []
    if Path(out).exists():
        for l in Path(out).read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                d = json.loads(l)
                d["t"] = time.mktime(time.strptime(d["ts"][:19], "%Y-%m-%dT%H:%M:%S")) - time.timezone
                rows.append(d)
            except Exception:
                pass
    return rows


def med_window(rows, t0, t1, key_filter):
    vals = []
    for r in rows:
        if t0 <= r["t"] <= t1:
            for k, v in r["temps"].items():
                if key_filter(k):
                    vals.append(v)
    return st.median(vals) if vals else None


def main():
    dll, prefix = sys.argv[1], sys.argv[2]
    out = prefix + "_lhm_control.jsonl"
    Path(out).unlink(missing_ok=True)
    rd = start_reader(dll, out)
    time.sleep(8)
    if rd.poll() is not None:
        print(json.dumps({"pass": False, "why": "reader exited", "stderr": (rd.stderr.read() or "")[-800:]}))
        sys.exit(2)
    t_idle0 = time.time()
    time.sleep(60)
    t_idle1 = time.time()
    hog, report, aff = m3.start_hog("lhm_control", 0xFFF0, Path(prefix).parent, Path(prefix).name + "_lhmctl")
    time.sleep(60)
    t_load1 = time.time()
    ips = m3.read_ips(report)
    m3.kill_tree(hog.pid)
    time.sleep(3)
    m3.kill_tree(rd.pid)
    rows = read(out)
    names = sorted({k for r in rows for k in r["temps"]})
    is_pkg = lambda k: any(p in k.lower() for p in PKG_KEYS)
    idle = med_window(rows, t_idle1 - 40, t_idle1, is_pkg)
    load = med_window(rows, t_load1 - 40, t_load1, is_pkg)
    per_sensor = {}
    for k in names:
        a = med_window(rows, t_idle1 - 40, t_idle1, lambda x, k=k: x == k)
        b = med_window(rows, t_load1 - 40, t_load1, lambda x, k=k: x == k)
        per_sensor[k] = {"idle": a, "load": b, "delta": (b - a) if a is not None and b is not None else None}
    ok = bool(idle is not None and load is not None and load - idle >= 5.0 and ips and ips > 0)
    print(json.dumps({"pass": ok, "package_idle_c": idle, "package_load_c": load,
                      "package_rise_c": (load - idle) if idle is not None and load is not None else None,
                      "spin_ips": ips, "sensors": per_sensor, "hardware": rows[-1]["hw"] if rows else None,
                      "n_samples": len(rows)}, indent=1))
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
