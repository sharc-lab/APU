"""spin_hog_affinity.py: CPU-only co-runner pinned to a set of logical CPUs, for the evo-t2s M3 power-coupling test.

The loop is the same pure integer loop as scripts/as_run/blade_C_apu/spin_hog2.py (no memory traffic). Differences:
the parent process sets its affinity mask first (children inherit it on Windows), every worker reports the mask it
actually runs under, and the process count defaults to the number of CPUs in the mask so the pinned cores are saturated.

Usage: py spin_hog_affinity.py --affinity-mask 0xF --duration-s 900 --report-file r.txt --affinity-file a.json
The report file holds iterations per second (positive control), rewritten every 2 s.
"""

import argparse
import ctypes
import ctypes.wintypes as wt
import json
import multiprocessing as mp
import time


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


def spin_worker(duration_s, counter, lock, q):
    q.put(current_affinity())
    t_end = time.monotonic() + duration_s
    x = 0
    while time.monotonic() < t_end:
        for _ in range(1_000_000):
            x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        with lock:
            counter.value += 1_000_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--affinity-mask", type=lambda s: int(s, 0), required=True)
    ap.add_argument("--n-procs", type=int, default=0)
    ap.add_argument("--duration-s", type=float, required=True)
    ap.add_argument("--report-file", required=True)
    ap.add_argument("--affinity-file", required=True)
    args = ap.parse_args()

    k = _k32()
    ok = k.SetProcessAffinityMask(k.GetCurrentProcess(), args.affinity_mask)
    parent = current_affinity()
    n = args.n_procs or bin(args.affinity_mask).count("1")

    counter = mp.Value("q", 0)
    lock = mp.Lock()
    q = mp.Queue()
    procs = [mp.Process(target=spin_worker, args=(args.duration_s, counter, lock, q), daemon=True) for _ in range(n)]
    t0 = time.monotonic()
    for p in procs:
        p.start()
    masks = []
    for _ in procs:
        try:
            masks.append(q.get(timeout=30))
        except Exception:
            masks.append(None)
    with open(args.affinity_file, "w") as f:
        json.dump({"requested_mask": args.affinity_mask, "set_ok": bool(ok), "parent_mask": parent,
                   "worker_masks": masks, "n_procs": n}, f)

    t_end = t0 + args.duration_s
    last_t, last_c = t0, 0
    while time.monotonic() < t_end and any(p.is_alive() for p in procs):
        time.sleep(min(2.0, max(0.0, t_end - time.monotonic())))
        now = time.monotonic()
        with lock:
            cur = counter.value
        ips = (cur - last_c) / (now - last_t) if now > last_t else 0.0
        with open(args.report_file, "w") as f:
            f.write(f"{ips:.1f}\n")
        last_t, last_c = now, cur
    for p in procs:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()


if __name__ == "__main__":
    mp.freeze_support()
    main()
