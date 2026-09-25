"""Analysis for harness/t2s_m3_power_coupling.py output.

Usage: py -3.12 analysis/t2s_m3_analysis.py <path prefix without extension>, for example
       results/t2s_m3_power_coupling_20260925T075353Z

Per condition (aggregated over its measured calls): TTFT and decode medians, slowdown against the mean of the two
no-co-runner conditions (none and none_end), hog iterations per second, and telemetry medians over the call windows:
iGPU actual frequency (Sysman domain 0), the power-limited frequency (tdp), throttle-reason bits, iGPU power from the
Sysman energy counter, Windows RAPL package power (Energy Meter), CPU frequency and iGPU 3D engine utilization.
n = 3 calls per condition, so ranges are shown instead of confidence intervals.
"""

from __future__ import annotations

import csv
import json
import statistics as st
import sys
from datetime import datetime
from pathlib import Path


def pt(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def main():
    prefix = sys.argv[1]
    rows = [json.loads(l) for l in Path(prefix + ".jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    calls = [r for r in rows if r.get("record") == "call" and r.get("outcome") == "ok" and not r.get("invalid")]
    bad = [r for r in rows if r.get("record") == "call" and (r.get("outcome") != "ok" or r.get("invalid"))]
    sys_rows = list(csv.DictReader(open(prefix + "_sysman.csv", encoding="utf-8")))
    for r in sys_rows:
        r["t"] = pt(r["ts_iso_utc"])
    win = []
    for l in Path(prefix + "_wincounters.jsonl").read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            d = json.loads(l)
            d["t"] = pt(d["ts"])
            win.append(d)
        except Exception:
            pass

    conds = []
    for r in calls:
        if r["cond"] not in conds:
            conds.append(r["cond"])
    base_ttft = med([r["ttft_s"] for r in calls if r["cond"] in ("none", "none_end")])
    base_dec = med([r["decode_tps"] for r in calls if r["cond"] in ("none", "none_end")])
    print(f"invalid or failed calls: {len(bad)}")
    print(f"no-co-runner baseline: TTFT {base_ttft:.2f} s, decode {base_dec:.2f} tok/s")
    print("cond | n | TTFT med [min,max] | TTFT x | decode med | decode x | hog Miter/s | hog Miter/s per core | "
          "iGPU MHz med [min] | tdp MHz med | throttle bits | iGPU W med [max] | pkg W (RAPL) med | CPU MHz med | 3D util % med")
    for c in conds:
        cs = [r for r in calls if r["cond"] == c]
        t0s = [(pt(r["t_start_utc"]), pt(r["t_end_utc"])) for r in cs]

        def inw(t):
            return any(a <= t <= b for a, b in t0s)
        f = [r for r in sys_rows if r["kind"] == "freq" and r["domain"] == "0" and inw(r["t"])]
        p = [r for r in sys_rows if r["kind"] == "power" and r["domain"] == "0" and inw(r["t"])]
        w = [r for r in win if inw(r["t"])]
        ttft = [r["ttft_s"] for r in cs]
        dec = [r["decode_tps"] for r in cs]
        ips = [r.get("hog_ips") for r in cs if r.get("hog_ips")]
        ncores = bin(cs[0].get("affinity_mask") or 0).count("1")
        pw = [float(r["power_w"]) for r in p if r["power_w"]]
        pkg = [x["energy_meter_mw"].get("rapl_package0_pkg") for x in w if x.get("energy_meter_mw")]
        pkg = [v / 1000 for v in pkg if v is not None]
        fa = [float(r["actual_mhz"]) for r in f]
        thr = sorted({r["throttle_reasons"] for r in f})
        print(f"{c} | {len(cs)} | {med(ttft):.2f} [{min(ttft):.2f},{max(ttft):.2f}] | {med(ttft) / base_ttft:.2f} | "
              f"{med(dec):.2f} | {base_dec / med(dec):.2f} | "
              f"{(med(ips) / 1e6):.0f}" if ips else f"{c} | {len(cs)} | {med(ttft):.2f} [{min(ttft):.2f},{max(ttft):.2f}] | {med(ttft) / base_ttft:.2f} | {med(dec):.2f} | {base_dec / med(dec):.2f} | NA",
              end="")
        print(f" | {(med(ips) / 1e6 / ncores):.1f}" if ips and ncores else " | NA", end="")
        print(f" | {med(fa):.0f} [{min(fa):.0f}] | {med([float(r['tdp_mhz']) for r in f]):.0f} | {thr} | "
              f"{med(pw):.1f} [{max(pw):.1f}] | {(med(pkg) if pkg else float('nan')):.1f} | "
              f"{med([x.get('cpu_freq_mhz') for x in w]):.0f} | {med([x.get('gpu_3d_pct') for x in w]):.0f}")


if __name__ == "__main__":
    main()
