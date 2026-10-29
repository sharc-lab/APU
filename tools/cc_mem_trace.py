"""CC cells (Task BJ): read a tools/cc_mem_sidecar.ps1 CSV and summarise a time window.

summarise(rows, t0, t1, pids) gives, inside [t0, t1]:
- peak and last GPU adapter Shared Usage (MB, summed over adapter instances);
- peak GPU process Shared Usage (MB) for the given pids (summed over their instances);
- minimum Available MBytes; peak and total Page Reads/sec (hard-fault reads).
Rows that failed to sample ("ERROR") are counted, never dropped silently.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

MB = 1024.0 * 1024.0
_PID = re.compile(r"pid_(\d+)_", re.IGNORECASE)


def read_rows(path: Path) -> list[tuple[float, str, str]]:
    with path.open(encoding="ascii", errors="replace", newline="") as fh:
        reader = csv.reader(fh)
        next(reader, None)
        return [(float(r[0]), r[1], ",".join(r[2:])) for r in reader if len(r) >= 3]


def summarise(
    rows: Iterable[tuple[float, str, str]], t0: float, t1: float, pids: Iterable[int] = ()
) -> dict[str, Any]:
    want = {int(p) for p in pids}
    adapter: dict[float, float] = defaultdict(float)
    proc: dict[float, float] = defaultdict(float)
    avail: list[float] = []
    reads: list[float] = []
    errors = 0
    for t, path, raw in rows:
        if not (t0 <= t <= t1):
            continue
        if path == "ERROR":
            errors += 1
            continue
        low = path.lower()
        try:
            v = float(raw)
        except ValueError:
            errors += 1
            continue
        if "gpu adapter memory" in low and low.endswith("shared usage"):
            adapter[t] += v
        elif "gpu process memory" in low and low.endswith("shared usage"):
            m = _PID.search(path)
            if m and (not want or int(m.group(1)) in want):
                proc[t] += v
        elif low.endswith("available mbytes"):
            avail.append(v)
        elif low.endswith("page reads/sec"):
            reads.append(v)
    times = sorted(adapter)
    return {
        "samples": len(times),
        "errors": errors,
        "adapter_shared_peak_mb": round(max(adapter.values()) / MB, 1) if adapter else None,
        "adapter_shared_last_mb": round(adapter[times[-1]] / MB, 1) if times else None,
        "process_shared_peak_mb": round(max(proc.values()) / MB, 1) if proc else None,
        "available_min_mb": min(avail) if avail else None,
        "page_reads_peak_per_s": max(reads) if reads else None,
        "page_reads_sum": round(sum(reads), 1) if reads else None,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float, default=float("inf"))
    ap.add_argument("--pid", type=int, action="append", default=[])
    a = ap.parse_args(argv)
    print(json.dumps(summarise(read_rows(a.csv), a.t0, a.t1, a.pid), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
