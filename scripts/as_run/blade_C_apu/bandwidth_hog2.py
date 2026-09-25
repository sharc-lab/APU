"""bandwidth_hog2.py -- memcpy co-runner with a logged GB/s positive control.

N threads, each doing ctypes.memmove between two 128 MiB buffers in a tight
loop (releases the GIL during the C call -> genuine N-core parallelism).
Every --report-interval-s, writes aggregate achieved GB/s to --report-file
as a single line (overwritten each time, so a reader always sees the latest
figure) -- this is the positive control the orchestrator checks before
trusting any measurement taken while this hog is running. If N=0, idles.
"""
import argparse
import ctypes
import sys
import threading
import time

BUF_SIZE = 128 * 1024 * 1024
_lock = threading.Lock()
_bytes_moved = [0]


def hog_thread(stop_event):
    src = ctypes.create_string_buffer(BUF_SIZE)
    dst = ctypes.create_string_buffer(BUF_SIZE)
    ctypes.memset(src, 0x5A, BUF_SIZE)
    while not stop_event.is_set():
        ctypes.memmove(dst, src, BUF_SIZE)
        ctypes.memmove(src, dst, BUF_SIZE)
        with _lock:
            _bytes_moved[0] += 2 * BUF_SIZE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-threads", type=int, required=True)
    ap.add_argument("--duration-s", type=float, required=True)
    ap.add_argument("--report-file", type=str, required=True)
    ap.add_argument("--report-interval-s", type=float, default=2.0)
    args = ap.parse_args()

    if args.n_threads <= 0:
        with open(args.report_file, "w") as f:
            f.write("0.0\n")
        time.sleep(args.duration_s)
        return

    stop_event = threading.Event()
    threads = [threading.Thread(target=hog_thread, args=(stop_event,), daemon=True)
               for _ in range(args.n_threads)]
    t0 = time.monotonic()
    for t in threads:
        t.start()

    t_end = t0 + args.duration_s
    last_report_t = t0
    last_report_bytes = 0
    while time.monotonic() < t_end and not stop_event.is_set():
        time.sleep(min(args.report_interval_s, max(0.0, t_end - time.monotonic())))
        now = time.monotonic()
        with _lock:
            cur_bytes = _bytes_moved[0]
        interval_s = now - last_report_t
        gbps = ((cur_bytes - last_report_bytes) / (1024 ** 3)) / interval_s if interval_s > 0 else 0.0
        with open(args.report_file, "w") as f:
            f.write(f"{gbps:.3f}\n")
        last_report_t = now
        last_report_bytes = cur_bytes

    stop_event.set()
    for t in threads:
        t.join(timeout=5)


if __name__ == "__main__":
    main()
