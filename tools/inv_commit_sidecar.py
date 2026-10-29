"""INV-1 system commit sampler (default 20 Hz) for the allocate-until-failure cells.

On Windows each period calls psapi GetPerformanceInfo (PERFORMANCE_INFORMATION; page counts are
multiplied by PageSize) and appends one CSV row:
t_unix,commit_total_bytes,commit_limit_bytes,commit_peak_bytes,physical_total_bytes,
physical_available_bytes,system_cache_bytes,kernel_nonpaged_bytes
The file is flushed about once a second. A failed sample is written as a row with empty fields
and the error in the last column (never skipped silently). Stops when --stop-file exists or after
--max-hours. Read-only; it changes no setting.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

FIELDS = (
    "commit_total_bytes",
    "commit_limit_bytes",
    "commit_peak_bytes",
    "physical_total_bytes",
    "physical_available_bytes",
    "system_cache_bytes",
    "kernel_nonpaged_bytes",
)
HEADER = ("t_unix", *FIELDS, "error")


class _PerfInfo(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("CommitTotal", ctypes.c_size_t),
        ("CommitLimit", ctypes.c_size_t),
        ("CommitPeak", ctypes.c_size_t),
        ("PhysicalTotal", ctypes.c_size_t),
        ("PhysicalAvailable", ctypes.c_size_t),
        ("SystemCache", ctypes.c_size_t),
        ("KernelTotal", ctypes.c_size_t),
        ("KernelPaged", ctypes.c_size_t),
        ("KernelNonpaged", ctypes.c_size_t),
        ("PageSize", ctypes.c_size_t),
        ("HandleCount", ctypes.c_uint32),
        ("ProcessCount", ctypes.c_uint32),
        ("ThreadCount", ctypes.c_uint32),
    ]


Sampler = Callable[[], dict[str, int]]


def make_windows_sampler() -> Sampler:
    """Returns a function that reads GetPerformanceInfo; raises off Windows."""
    if sys.platform != "win32":
        raise OSError("GetPerformanceInfo needs Windows")
    psapi = ctypes.WinDLL("psapi", use_last_error=True)  # type: ignore[attr-defined]
    fn = psapi.GetPerformanceInfo
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.POINTER(_PerfInfo), ctypes.c_uint32]

    def sample() -> dict[str, int]:
        pi = _PerfInfo()
        pi.cb = ctypes.sizeof(_PerfInfo)
        if not fn(ctypes.byref(pi), pi.cb):
            raise OSError(f"GetPerformanceInfo failed GetLastError={ctypes.get_last_error()}")  # type: ignore[attr-defined]
        ps = int(pi.PageSize)
        return {
            "commit_total_bytes": int(pi.CommitTotal) * ps,
            "commit_limit_bytes": int(pi.CommitLimit) * ps,
            "commit_peak_bytes": int(pi.CommitPeak) * ps,
            "physical_total_bytes": int(pi.PhysicalTotal) * ps,
            "physical_available_bytes": int(pi.PhysicalAvailable) * ps,
            "system_cache_bytes": int(pi.SystemCache) * ps,
            "kernel_nonpaged_bytes": int(pi.KernelNonpaged) * ps,
        }

    return sample


def run(
    out_csv: Path,
    stop_file: Path,
    *,
    hz: float,
    max_hours: float,
    sampler: Sampler,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Sample until the stop file or max_hours. Returns the number of rows written."""
    if hz <= 0 or max_hours <= 0:
        raise ValueError("hz and max_hours must be > 0")
    period = 1.0 / hz
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    t_end = clock() + max_hours * 3600
    with out_csv.open("w", newline="", encoding="ascii") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        fh.flush()
        last_flush = clock()
        while not stop_file.exists() and clock() < t_end:
            t = clock()
            row: list[Any]
            try:
                s = sampler()
                row = [f"{t:.3f}", *(int(s[k]) for k in FIELDS), ""]
            except Exception as exc:  # written as an error row, never skipped
                row = [f"{t:.3f}", *([""] * len(FIELDS)), f"{type(exc).__name__}: {exc}"[:200]]
            w.writerow(row)
            rows += 1
            now = clock()
            if now - last_flush >= 1.0:
                fh.flush()
                last_flush = now
            sleep(max(0.0, period - (now - t)))
        fh.flush()
    return rows


def main(argv: list[str] | None = None, sampler: Sampler | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--stop-file", type=Path, required=True)
    ap.add_argument("--hz", type=float, default=20.0)
    ap.add_argument("--max-hours", type=float, default=8.0)
    a = ap.parse_args(argv)
    if not (a.hz > 0 and a.max_hours > 0):
        ap.error("--hz and --max-hours must be > 0")
    try:
        smp = sampler or make_windows_sampler()
    except Exception as exc:
        print(f"SIDECAR_FAILED {type(exc).__name__}: {exc}", flush=True)
        return 2
    n = run(a.out_csv, a.stop_file, hz=a.hz, max_hours=a.max_hours, sampler=smp)
    print(f"SIDECAR_DONE rows={n} out={a.out_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
