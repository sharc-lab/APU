"""
bandwidth_hog.py -- a multithreaded memory-bandwidth co-runner.

Each of --n-threads OS threads continuously copies between two 128 MiB
buffers via ctypes.memmove in a tight loop until --duration-s elapses (or
until told to stop). ctypes.memmove is a C call and releases the GIL for the
duration of the copy, so N threads genuinely run in parallel on N cores --
this is not defeated by Python's GIL the way a pure-Python copy loop would be.

This is a bandwidth stressor, not a compute stressor: the buffers (256 MiB
total per thread) are far larger than any cache level, so each memmove call
is DRAM-bandwidth-bound, not cache-bound. Running N threads is meant to
consume roughly N/total_cores of the machine's aggregate memory bandwidth,
contending directly with whatever the co-resident llama-server is trying to
read from the same DRAM.

Usage: python bandwidth_hog.py --n-threads N --duration-s S
Exits after duration-s regardless of anything else -- no external stop
signal needed, so the orchestrator just launches it and waits.
"""

import argparse
import ctypes
import threading
import time

BUF_SIZE = 128 * 1024 * 1024  # 128 MiB per buffer, far larger than any cache


def hog_thread(duration_s: float, stop_event: threading.Event):
    src = ctypes.create_string_buffer(BUF_SIZE)
    dst = ctypes.create_string_buffer(BUF_SIZE)
    ctypes.memset(src, 0x5A, BUF_SIZE)
    t_end = time.monotonic() + duration_s
    while time.monotonic() < t_end and not stop_event.is_set():
        ctypes.memmove(dst, src, BUF_SIZE)
        ctypes.memmove(src, dst, BUF_SIZE)


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
        threading.Thread(target=hog_thread, args=(args.duration_s, stop_event), daemon=True)
        for _ in range(args.n_threads)
    ]
    t0 = time.monotonic()
    for t in threads:
        t.start()
    print(f"Started {args.n_threads} bandwidth-hog threads for {args.duration_s}s.")
    for t in threads:
        t.join(timeout=args.duration_s + 10)
    print(f"Done after {time.monotonic() - t0:.1f}s.")


if __name__ == "__main__":
    main()
