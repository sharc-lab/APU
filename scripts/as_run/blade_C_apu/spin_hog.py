"""spin_hog.py -- CPU-only control for bandwidth_hog.py: N threads doing pure
compute on an array that stays hot in L1 cache, so this occupies CPU cores at
the same thread counts as the memcpy hog WITHOUT generating memory-bandwidth
traffic. The comparison this exists for: bandwidth_hog(N) slowdown includes
BOTH core contention (N threads competing for the same cores llama-server's
own threads want) and bandwidth contention (N threads reading/writing DRAM
llama-server also needs). spin_hog(N) isolates the core-contention component
alone. memcpy_slowdown - spin_slowdown at the same N = the bandwidth-specific
component.

IMPLEMENTATION NOTE: pure-Python arithmetic in a threading.Thread does NOT
release the GIL between bytecode ops, so N such threads would NOT achieve
real N-core parallelism -- they'd just time-slice on effectively one core,
which is not a valid CPU-only control (it wouldn't occupy N cores the way
bandwidth_hog's N memmove threads genuinely do, since ctypes.memmove is a C
call that releases the GIL for its duration). Fixed here the same way:
numpy's ufuncs release the GIL during their C-level computation, so N
threads each running numpy ops DO run in true parallel across N cores. The
array is small (8 KiB, 1024 float64s) -- comfortably inside L1 (typically
32-48 KiB per core) -- so this is compute-bound, not memory-bound."""

import argparse
import threading
import time

import numpy as np

ARRAY_SIZE = 1024  # 1024 float64 = 8 KiB, well inside L1 cache


def spin_thread(duration_s: float, stop_event: threading.Event):
    arr = np.random.randn(ARRAY_SIZE)
    t_end = time.monotonic() + duration_s
    while time.monotonic() < t_end and not stop_event.is_set():
        # np.sin/np.cos on a small, cache-resident array: releases the GIL
        # during the C loop (same mechanism as ctypes.memmove), pure compute,
        # no allocation, no memory traffic beyond what already fits in L1.
        arr = np.sin(arr) + np.cos(arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-threads", type=int, required=True)
    ap.add_argument("--duration-s", type=float, required=True)
    args = ap.parse_args()

    if args.n_threads <= 0:
        print("n-threads=0, idling (this is the 0% co-runner condition).")
        time.sleep(args.duration_s)
        return

    stop_event = threading.Event()
    threads = [
        threading.Thread(target=spin_thread, args=(args.duration_s, stop_event), daemon=True)
        for _ in range(args.n_threads)
    ]
    t0 = time.monotonic()
    for t in threads:
        t.start()
    print(f"Started {args.n_threads} spin threads for {args.duration_s}s.")
    for t in threads:
        t.join(timeout=args.duration_s + 10)
    print(f"Done after {time.monotonic() - t0:.1f}s.")


if __name__ == "__main__":
    main()
