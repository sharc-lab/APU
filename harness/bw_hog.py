"""bw_hog.py: memory-bandwidth co-runner pinned to a set of logical CPUs, the DRAM-traffic counterpart to
harness/spin_hog_affinity.py (which spins on integers and touches no memory).

Same shape as spin_hog_affinity.py on purpose, so harness/t2s_m3_power_coupling.py start_hog/kill_tree/read_ips drives
either one unchanged: the parent sets its affinity mask first (children inherit it on Windows), every worker reports the
mask it actually runs under, the process count defaults to the number of CPUs in the mask, --duration-s bounds the run,
and the report file is a single float rewritten every 2 s so read_ips() can read a positive control mid-condition. The
only differences are the inner loop (a STREAM triad instead of an integer loop) and the units in the report file (GB/s,
not iterations/s).

Each worker allocates its own three float64 arrays of --array-mib each (default 320 MiB, so one array alone exceeds the
32 MiB L3 slice of either Strix Halo CCD by an order of magnitude and the working set cannot be cached) and loops
a[:] = b + scalar * c over them. numpy is used for the triad because a pure-Python loop over 40M elements is bound by
the interpreter, not by DRAM, and would measure the wrong thing entirely; numpy is already a declared dependency in
pyproject.toml. The ufuncs used here are single-threaded, so one worker process per pinned logical CPU saturates the
mask the same way spin_hog_affinity.py does.

Reported bandwidth uses the STREAM triad convention: 3 arrays touched per iteration (read b, read c, write a), so
bytes_moved = 3 * array_bytes * iterations. The two-ufunc form below (multiply into a, then add into a) actually moves
more than that, because the second pass reads a back, so the reported figure is a conservative lower bound on the real
DRAM traffic, not an upper bound. That matters for reading the numbers: a condition whose reported GB/s is close to the
platform's peak is definitely saturating the controller.

Usage:
  py bw_hog.py --affinity-mask 0xF0 --duration-s 900 --report-file r.txt --affinity-file a.json
  py bw_hog.py --self-test          # tiny arrays, well under a second, for CI
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wt
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

BYTES_PER_ELEM = 8  # float64
ARRAYS_PER_ITER = 3  # STREAM triad: read b, read c, write a


def _k32():
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.GetCurrentProcess.restype = wt.HANDLE
    k.GetCurrentProcess.argtypes = []
    k.SetProcessAffinityMask.restype = wt.BOOL
    k.SetProcessAffinityMask.argtypes = [wt.HANDLE, ctypes.c_size_t]
    k.GetProcessAffinityMask.restype = wt.BOOL
    k.GetProcessAffinityMask.argtypes = [wt.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    return k


def current_affinity() -> int:
    k = _k32()
    proc, sysm = ctypes.c_size_t(), ctypes.c_size_t()
    k.GetProcessAffinityMask(k.GetCurrentProcess(), ctypes.byref(proc), ctypes.byref(sysm))
    return proc.value


def gbps(array_bytes, iterations, elapsed_s, arrays_per_iter=ARRAYS_PER_ITER):
    """Achieved bandwidth in GB/s for `iterations` triad passes over arrays of `array_bytes` each. GB here is 1e9
    bytes, the STREAM convention, not 2**30 -- mixing the two is a 7% error and the two are easy to confuse. Returns
    None rather than dividing by zero when no time has elapsed, so a report written before the first pass is honest
    about having no measurement yet."""
    if elapsed_s is None or elapsed_s <= 0:
        return None
    return arrays_per_iter * float(array_bytes) * float(iterations) / elapsed_s / 1e9


def elems_for_mib(array_mib):
    """Elements per array for a target size in MiB (2**20 bytes), at least 1."""
    return max(1, int(array_mib * 2 ** 20) // BYTES_PER_ELEM)


def triad_worker(duration_s, array_mib, counter, lock, q):
    """One triad process. Reports its affinity mask, then loops until duration_s, adding completed passes to the shared
    counter. Allocation happens inside the worker so each process touches (and first-faults) its own pages on whatever
    node its pinned CPU belongs to."""
    import numpy as np

    q.put(current_affinity())
    n = elems_for_mib(array_mib)
    a = np.zeros(n, dtype=np.float64)
    b = np.ones(n, dtype=np.float64)
    c = np.full(n, 2.0, dtype=np.float64)
    scalar = 3.0
    t_end = time.monotonic() + duration_s
    local = 0
    while time.monotonic() < t_end:
        np.multiply(c, scalar, out=a)
        np.add(a, b, out=a)
        local += 1
        if local % 8 == 0:
            with lock:
                counter.value += 8
    rem = local % 8
    if rem:
        with lock:
            counter.value += rem


def self_test():
    """Tiny in-process run: no multiprocessing, no affinity change, well under a second. Checks that the triad runs and
    that the byte arithmetic produces a positive rate, so CI covers the measurement path without needing real cores or
    a 320 MiB allocation."""
    import numpy as np

    n = elems_for_mib(1)
    a = np.zeros(n, dtype=np.float64)
    b = np.ones(n, dtype=np.float64)
    c = np.full(n, 2.0, dtype=np.float64)
    t0 = time.monotonic()
    iters = 0
    while time.monotonic() - t0 < 0.2:
        np.multiply(c, 3.0, out=a)
        np.add(a, b, out=a)
        iters += 1
    el = time.monotonic() - t0
    rate = gbps(n * BYTES_PER_ELEM, iters, el)
    # numpy comparisons return numpy.bool_, which json cannot encode: coerce to a plain bool before reporting.
    value_ok = bool(abs(float(a[0]) - 7.0) < 1e-9)
    ok = bool(iters > 0 and rate and rate > 0 and value_ok)
    out = {"self_test": True, "pass": ok, "array_bytes": n * BYTES_PER_ELEM, "iterations": iters,
           "elapsed_s": round(el, 4), "gbps": round(rate, 3) if rate else None, "triad_value_ok": value_ok}
    print(json.dumps(out))
    return 0 if ok else 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--affinity-mask", type=lambda s: int(s, 0), default=None)
    ap.add_argument("--n-procs", type=int, default=0)
    ap.add_argument("--duration-s", type=float, default=None)
    ap.add_argument("--report-file", default=None)
    ap.add_argument("--affinity-file", default=None)
    ap.add_argument("--array-mib", type=float, default=320.0,
                    help="per-worker array size in MiB; three of these are allocated per worker")
    ap.add_argument("--self-test", action="store_true", help="tiny arrays, sub-second, for CI")
    args = ap.parse_args()
    if args.self_test:
        sys.exit(self_test())
    for name in ("affinity_mask", "duration_s", "report_file", "affinity_file"):
        if getattr(args, name) is None:
            ap.error(f"--{name.replace('_', '-')} is required unless --self-test is given")

    k = _k32()
    ok = k.SetProcessAffinityMask(k.GetCurrentProcess(), args.affinity_mask)
    parent = current_affinity()
    n = args.n_procs or bin(args.affinity_mask).count("1")
    array_bytes = elems_for_mib(args.array_mib) * BYTES_PER_ELEM

    counter = mp.Value("q", 0)
    lock = mp.Lock()
    q = mp.Queue()
    procs = [mp.Process(target=triad_worker, args=(args.duration_s, args.array_mib, counter, lock, q), daemon=True)
             for _ in range(n)]
    t0 = time.monotonic()
    for p in procs:
        p.start()
    masks = []
    for _ in procs:
        try:
            masks.append(q.get(timeout=120))  # allocating 3 x array_mib per worker can take a while under memory pressure
        except Exception:
            masks.append(None)
    Path(args.affinity_file).write_text(json.dumps(
        {"requested_mask": args.affinity_mask, "set_ok": bool(ok), "parent_mask": parent, "worker_masks": masks,
         "n_procs": n, "array_mib": args.array_mib, "array_bytes": array_bytes,
         "bytes_per_iteration": ARRAYS_PER_ITER * array_bytes, "hog_kind": "bandwidth_triad"}), encoding="utf-8")

    t_end = t0 + args.duration_s
    last_t, last_c = t0, 0
    while time.monotonic() < t_end and any(p.is_alive() for p in procs):
        time.sleep(min(2.0, max(0.0, t_end - time.monotonic())))
        now = time.monotonic()
        with lock:
            cur = counter.value
        rate = gbps(array_bytes, cur - last_c, now - last_t)
        # A single float, the shape t2s_m3_power_coupling.read_ips() parses -- blank when there is no rate yet, which
        # read_ips turns into None rather than a fabricated zero.
        Path(args.report_file).write_text(f"{rate:.3f}\n" if rate is not None else "\n", encoding="utf-8")
        last_t, last_c = now, cur
    for p in procs:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()


if __name__ == "__main__":
    mp.freeze_support()
    main()
