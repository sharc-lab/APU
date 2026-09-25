"""spin_hog2.py -- CPU-only co-runner control, pure integer loop, no memory
traffic, with a logged iterations/s positive control.

N SEPARATE PROCESSES (multiprocessing, not threading): a pure-Python integer
loop inside a threading.Thread would NOT achieve real N-core occupation --
the GIL serializes bytecode execution across threads in one process, so N
such threads just time-slice on effectively one core. bandwidth_hog2.py
sidesteps this because ctypes.memmove is a C call that releases the GIL;
a pure integer loop has no such release point. Separate OS processes have
independent interpreters/GILs, so this is the correct way to get N cores
genuinely busy with nothing but integer arithmetic, no memory-bandwidth
component, matching the literal "pure integer loop, no memory traffic" spec.
"""
import argparse
import multiprocessing as mp
import time


def spin_worker(duration_s: float, counter, lock):
    t_end = time.monotonic() + duration_s
    x = 0
    local_iters = 0
    while time.monotonic() < t_end:
        for _ in range(1_000_000):
            x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        local_iters += 1_000_000
        with lock:
            counter.value += 1_000_000
    return local_iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-threads", type=int, required=True)  # kept name for CLI parity with memcpy hog
    ap.add_argument("--duration-s", type=float, required=True)
    ap.add_argument("--report-file", type=str, required=True)
    ap.add_argument("--report-interval-s", type=float, default=2.0)
    args = ap.parse_args()

    if args.n_threads <= 0:
        with open(args.report_file, "w") as f:
            f.write("0.0\n")
        time.sleep(args.duration_s)
        return

    counter = mp.Value("q", 0)
    lock = mp.Lock()
    procs = [mp.Process(target=spin_worker, args=(args.duration_s, counter, lock), daemon=True)
             for _ in range(args.n_threads)]
    t0 = time.monotonic()
    for p in procs:
        p.start()

    t_end = t0 + args.duration_s
    last_report_t = t0
    last_report_count = 0
    while time.monotonic() < t_end and any(p.is_alive() for p in procs):
        time.sleep(min(args.report_interval_s, max(0.0, t_end - time.monotonic())))
        now = time.monotonic()
        with lock:
            cur_count = counter.value
        interval_s = now - last_report_t
        ips = (cur_count - last_report_count) / interval_s if interval_s > 0 else 0.0
        with open(args.report_file, "w") as f:
            f.write(f"{ips:.1f}\n")
        last_report_t = now
        last_report_count = cur_count

    for p in procs:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()


if __name__ == "__main__":
    mp.freeze_support()
    main()
