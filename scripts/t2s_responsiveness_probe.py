"""External, read-only SSH responsiveness probe for evo-t2s, run from the controller (this machine), never on evo-t2s
itself. Does not modify night2 or night2b in any way -- it only opens its own short-lived SSH connections.

Measures how usable evo-t2s is while a C1 memory-lock cell is running (found necessary live on 2026-09-28: a bare
`echo hi` over a fresh SSH session took nearly 2 minutes during a C1 cell). Every poll opens a fresh
`ssh ... "cmd /c echo hi"` and times the round trip; polls every 60 s while the night2b log's last phase line says
"phase c1", every 300 s otherwise. The probe itself is a second thing running against evo-t2s, so it must not become
part of the load it is measuring: the current item_id/mem_headroom_gb are refreshed by a separate, infrequent,
lightweight tail read (at most once every 5 polls) and cached between refreshes rather than being added to every
timed echo call.

For the next run this probe will move inside the harness as a per-C1-cell interactive-latency measurement; this
script is the standalone, external version for tonight.

Usage: py -3.12 scripts/t2s_responsiveness_probe.py [--minutes N]
Writes results/t2s_responsiveness_<UTC stamp>.jsonl, one line per poll, fsynced.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SSH = r"C:\Windows\System32\OpenSSH\ssh.exe"
HOST = "sharc@100.72.40.24"
REPO = Path(__file__).resolve().parents[1]
LOG_PATH = r"C:\apu\ovn\queue_night2b.log"
RESULTS_PATH = r"C:\apu\ovn\results\t2s_night2_20260928T004924Z.jsonl"
REFRESH_EVERY_N_POLLS = 5


def utc_iso():
    return datetime.now(timezone.utc).isoformat()


def echo_probe():
    t0 = time.monotonic()
    try:
        p = subprocess.run([SSH, "-o", "BatchMode=yes", "-o", "ConnectTimeout=120", HOST, "cmd /c echo hi"],
                           capture_output=True, text=True, timeout=130)
        rtt = time.monotonic() - t0
        ok = p.returncode == 0 and "hi" in p.stdout
        return rtt, ok, p.stdout.strip()[:100]
    except subprocess.TimeoutExpired:
        return time.monotonic() - t0, False, "timeout"


def current_phase():
    try:
        p = subprocess.run([SSH, "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", HOST,
                           f"Get-Content {LOG_PATH} -ErrorAction SilentlyContinue | Select-String 'phase ' | Select -Last 1"],
                           capture_output=True, text=True, timeout=40)
        m = re.search(r"phase (\w+)", p.stdout)
        return m.group(1) if m else None
    except Exception:
        return None


def tail_item(prev):
    """Lightweight tail of the live results file for the current item_id and mem_headroom_gb. Called at most once
    every REFRESH_EVERY_N_POLLS; falls back to the previous poll's cached value on any failure or timeout."""
    try:
        p = subprocess.run([SSH, "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", HOST,
                           f"Get-Content {RESULTS_PATH} -ErrorAction SilentlyContinue -Tail 1"],
                           capture_output=True, text=True, timeout=40)
        d = json.loads(p.stdout.strip().splitlines()[-1]) if p.stdout.strip() else {}
        return d.get("item_id", prev[0]), d.get("mem_headroom_gb", prev[1])
    except Exception:
        return prev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=600.0, help="stop after this many minutes (default 10 h)")
    args = ap.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = REPO / "results" / f"t2s_responsiveness_{stamp}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"logging to {out_path}")
    item = (None, None)
    n = 0
    t_end = time.monotonic() + args.minutes * 60
    while time.monotonic() < t_end:
        phase = current_phase()
        in_c1 = phase == "c1"
        if n % REFRESH_EVERY_N_POLLS == 0:
            item = tail_item(item)
        rtt, ok, note = echo_probe()
        row = {"ts_utc": utc_iso(), "round_trip_s": round(rtt, 3), "success": ok, "note": note,
              "phase": phase, "in_c1": in_c1, "item_id": item[0], "mem_headroom_gb": item[1]}
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
            os.fsync(f.fileno())
        print(json.dumps(row))
        n += 1
        time.sleep(max(0.0, (60.0 if in_c1 else 300.0) - rtt))


if __name__ == "__main__":
    main()
