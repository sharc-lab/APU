"""Demo hold on evo-x2 (operational entry, pre-registered in docs/EXPERIMENTS.md "2026-10-23: Demo hold job"): keeps
the queue from launching the next measurement while the live demo uses the machine, and releases itself.

It touches nothing: no Ollama start or stop, no model, no GPU, no file other than its own output rows and its stdout
(which the queue writes to queue_<id>.log, the watchdog's heartbeat). It waits until a fixed release time, printing a
heartbeat line every HEARTBEAT_S, then calls t2s_queue.advance() once. If it starts at or after the release time it
calls advance() at once. It sets no pause flag, so the release is automatic and does not depend on anyone.

    py -3.12 demo\\x2_demo_hold.py --start-utc 2026-10-29T15:00:00+00:00 --release-utc 2026-10-30T01:00:00+00:00
        --out C:\\apu\\ovn\\results\\demo_hold.jsonl --queued
    py -3.12 demo\\x2_demo_hold.py --start-utc ... --release-utc ... --dry-run   (prints the plan; no wait, no advance)
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HEARTBEAT_S = 300            # well under the watchdog's 30 min heartbeat threshold
MAX_HOLD_H = 168.0           # refuse a release time more than 7 days ahead (a typo guard); the hold may start days early


def parse_release(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        raise ValueError("--release-utc needs an explicit UTC offset, e.g. 2026-10-29T23:00:00+00:00")
    return dt.astimezone(timezone.utc)


def plan(release: datetime, now: datetime, start: datetime | None = None) -> dict:
    if start is not None and start >= release:
        raise ValueError(f"window start {start.isoformat()} is not before the release {release.isoformat()}")
    wait = (release - now).total_seconds()
    if wait > MAX_HOLD_H * 3600:
        raise ValueError(f"release {release.isoformat()} is {wait / 3600:.1f} h ahead, above {MAX_HOLD_H} h")
    return {"window_start_utc": start.isoformat() if start else None, "release_utc": release.isoformat(),
            "now_utc": now.isoformat(), "wait_s": max(0.0, wait), "release_at_once": wait <= 0,
            "launched_after_window_start": (now >= start) if start else None}


def hold(release: datetime, emit, clock=lambda: datetime.now(timezone.utc), sleep=time.sleep,
         heartbeat_s: float = HEARTBEAT_S, start: datetime | None = None) -> dict:
    """Holds from launch until release. The queue has no time gate, so the machine is free for the demo window only
    if the entry is placed to launch at or before the window start; launched_after_window_start records whether it
    did (a late launch means a measurement job may have been running inside the window)."""
    p = plan(release, clock(), start)
    emit({"record": "run_start", "harness": "x2_demo_hold", "host": socket.gethostname(), **p})
    while True:
        left = (release - clock()).total_seconds()
        if left <= 0:
            break
        print(f"[{clock().isoformat()}] demo hold: {left / 60:.1f} min until release at {release.isoformat()}",
              flush=True)
        sleep(min(heartbeat_s, left))
    end = {"record": "run_end", "released_utc": clock().isoformat(), "release_utc": release.isoformat(),
           "started_after_release": p["release_at_once"]}
    emit(end)
    return end


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release-utc", required=True)
    ap.add_argument("--start-utc", default=None, help="the demo window start (recorded and checked; ISO with offset)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--queued", action="store_true", help="call t2s_queue.advance() on exit")
    a = ap.parse_args(argv)
    release = parse_release(a.release_utc)
    start = parse_release(a.start_utc) if a.start_utc else None
    if a.dry_run:
        print(json.dumps({"dry_run": True, "host": socket.gethostname(),
                          **plan(release, datetime.now(timezone.utc), start)}))
        return 0

    def emit(row):
        line = json.dumps({"ts_utc": datetime.now(timezone.utc).isoformat(), **row})
        print(line, flush=True)
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            with open(a.out, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    note = f"completed: demo hold released at {release.isoformat()}"
    try:
        hold(release, emit, start=start)
    except Exception as e:
        note = f"stopped: demo hold error {e!r}"[:300]
    finally:
        if a.queued:
            here = Path(__file__).resolve().parent
            if not (here / "t2s_queue.py").exists():
                sys.path.insert(0, str(here.parent))
            try:
                import t2s_queue as tq
                tq.advance(note)
            except Exception as e:
                print(f"queue advance failed: {e!r}")
    print(note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
