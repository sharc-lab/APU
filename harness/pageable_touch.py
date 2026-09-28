"""Companion process for K2 (harness/t2s_k2_pressure.py), pressure arm (b): allocate and touch ordinary pageable
memory, no AWE lock. This is the counterpart to memory_balloon_awe.py's locked balloon: the memory taken here stays in
this process's normal working set, so Windows can trim or page it out under pressure exactly like any other
interactive app's memory, unlike the AWE-locked balloon which the OS can never reclaim.

A repo search for "touch"/"pageable" across harness/ (done before writing this) found nothing that allocates and
touches ordinary pageable memory outside memory_balloon_awe.py's own AWE mechanism, so this is a new, small
companion script rather than a reuse of something existing.

TARGET: allocate --target-mb of memory and touch every page (write one byte per 4 KiB page, not just reserve the
allocation) so it is actually committed and resident, matching how a real foreground application would occupy
memory. Every --sample-interval-s it re-touches every page (a write, not a read, so a copy-on-write zero page cannot
satisfy it) and appends one log line; the resulting Pages Input/sec is a genuine memory-pressure signal, not a no-op.

SAFETY VALVE: same pattern as memory_balloon_awe.py's heartbeat -- if --heartbeat-file goes stale for more than
--heartbeat-timeout-s, or is deleted, this frees the allocation and exits. Killing this PID by any means also frees
the memory (normal process exit), no special handling needed since these are ordinary Python objects, not
AWE-reserved physical pages.

Usage: python pageable_touch.py --target-mb 8192 --heartbeat-file <path> [--heartbeat-timeout-s 300]
                                 [--sample-interval-s 5] [--log-path <path>]

Prints "TARGET REACHED target_mb=<n> pages=<n>" once the allocation and first touch pass are done, the same
marker-line convention memory_balloon_awe.py uses so a caller can poll its stdout/log file for readiness.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

PAGE_BYTES = 4096


def utc_iso():
    return datetime.now(timezone.utc).isoformat()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-mb", type=float, required=True)
    ap.add_argument("--heartbeat-file", required=True)
    ap.add_argument("--heartbeat-timeout-s", type=float, default=300.0)
    ap.add_argument("--sample-interval-s", type=float, default=5.0)
    ap.add_argument("--log-path", default=None)
    args = ap.parse_args()

    target_mb = max(args.target_mb, 0.0)
    n_pages = max(int(target_mb * 1024 * 1024 / PAGE_BYTES), 1) if target_mb > 0 else 0
    block = bytearray(n_pages * PAGE_BYTES)
    if n_pages:
        block[0::PAGE_BYTES] = bytes([1]) * n_pages  # one write per page: touches every page, not just the allocation
    print(f"TARGET REACHED target_mb={target_mb} pages={n_pages}", flush=True)

    log_f = open(args.log_path, "w", encoding="utf-8") if args.log_path else None
    flip = 0
    try:
        while True:
            try:
                hb_age = time.time() - os.path.getmtime(args.heartbeat_file)
            except OSError:
                break
            if hb_age > args.heartbeat_timeout_s:
                break
            if n_pages:
                flip ^= 1
                block[0::PAGE_BYTES] = bytes([flip]) * n_pages
            if log_f is not None:
                log_f.write(json.dumps({"ts": utc_iso(), "target_mb": target_mb, "n_pages": n_pages}) + "\n")
                log_f.flush()
            time.sleep(args.sample_interval_s)
    finally:
        del block
        if log_f is not None:
            log_f.close()


if __name__ == "__main__":
    main()
