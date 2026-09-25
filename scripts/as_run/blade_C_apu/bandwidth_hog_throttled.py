"""bandwidth_hog_throttled.py -- memcpy co-runner with a TARGET achieved GB/s
(dose-response control), not just full-speed. N threads each do ctypes.memmove
between two 128 MiB buffers, but sleep between copies to pace toward
--target-gbps in aggregate. Logs actual achieved GB/s to --report-file every
--report-interval-s, same positive-control convention as bandwidth_hog2.py --
the target is a request, not a guarantee; what matters is the LOGGED achieved
rate, which is what dose-response analysis is against, not the requested one.
"""
import argparse
import ctypes
import threading
import time

BUF_SIZE = 128 * 1024 * 1024
_lock = threading.Lock()
_bytes_moved = [0]


def hog_thread(stop_event, per_thread_target_gbps: float):
    src = ctypes.create_string_buffer(BUF_SIZE)
    dst = ctypes.create_string_buffer(BUF_SIZE)
    ctypes.memset(src, 0x5A, BUF_SIZE)
    bytes_per_copy = 2 * BUF_SIZE  # one round trip: dst<-src, src<-dst
    target_bps = per_thread_target_gbps * (1024 ** 3)
    while not stop_event.is_set():
        t0 = time.perf_counter()
        ctypes.memmove(dst, src, BUF_SIZE)
        ctypes.memmove(src, dst, BUF_SIZE)
        with _lock:
            _bytes_moved[0] += bytes_per_copy
        if target_bps > 0:
            elapsed = time.perf_counter() - t0
            ideal_time = bytes_per_copy / target_bps
            sleep_time = ideal_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-threads", type=int, required=True)
    ap.add_argument("--target-gbps", type=float, default=0.0,
                     help="0 = unthrottled (full speed, same as bandwidth_hog2.py). "
                          ">0 = pace toward this AGGREGATE GB/s across all threads.")
    ap.add_argument("--duration-s", type=float, required=True)
    ap.add_argument("--report-file", type=str, required=True)
    ap.add_argument("--report-interval-s", type=float, default=2.0)
    args = ap.parse_args()

    if args.n_threads <= 0:
        with open(args.report_file, "w") as f:
            f.write("0.0\n")
        time.sleep(args.duration_s)
        return

    per_thread_target = args.target_gbps / args.n_threads if args.target_gbps > 0 else 0.0

    stop_event = threading.Event()
    threads = [threading.Thread(target=hog_thread, args=(stop_event, per_thread_target), daemon=True)
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
