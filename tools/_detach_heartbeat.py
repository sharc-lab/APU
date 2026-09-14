"""Heartbeat writer used to prove a detached launch outlives the shell that started it.

Writes an advancing tick to a JSON file. If the tick is still advancing after the SSH session
that spawned it has been closed, the detachment mechanism works. If it stops at the moment of
disconnect, the process was inside the session's job object and the launcher is unsafe.

Records its own parent PID, which is the second half of the evidence: a process whose parent is
the WMI provider host was never a child of the SSH session in the first place.
"""

from __future__ import annotations

import datetime
import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    out_path = Path(sys.argv[1])
    duration_s = float(sys.argv[2])
    started = time.time()
    started_utc = datetime.datetime.now(datetime.UTC).isoformat()
    tick = 0
    while time.time() - started < duration_s:
        tick += 1
        payload = {
            "tick": tick,
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "started_utc": started_utc,
            "utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "elapsed_s": round(time.time() - started, 1),
            "duration_s": duration_s,
        }
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(out_path)
        time.sleep(2.0)

    final = json.loads(out_path.read_text(encoding="utf-8"))
    final["finished"] = True
    final["finished_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
    out_path.write_text(json.dumps(final, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
