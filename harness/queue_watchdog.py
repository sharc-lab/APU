"""Independent watchdog for the evo-t2s/evo-x2 run queue. Runs as a Windows scheduled task (every 10 min, as SYSTEM)
so a crashed or hung orchestrator process cannot itself silently leave the queue stuck -- unlike t2s_queue.advance(),
which only runs from inside a finishing orchestrator's own process and therefore cannot recover if that process
never gets to its finally block at all (killed externally, powered off mid-run, or crashed hard enough to skip
cleanup).

Decision per tick, in order:
  1. If an entry is "running" but its pid is dead and no run_end record exists in its own results file, mark it
     "crashed" and launch the next pending entry.
  2. Else if nothing is "running" and something is "pending", launch the next pending entry.
  3. Else if nothing is "running" and nothing is "pending", write queue_empty.flag with a timestamp (t2s_queue's own
     mechanism already does this from inside advance(), but the watchdog is the backstop for the case advance()
     itself never got the chance to run).
  4. Else (something legitimately running with a live pid): do nothing.

Every decision is logged to C:\\apu\\ovn\\watchdog.log, one line per tick, so a human reviewing the machine after the
fact can see what the watchdog saw and did, not just its final effect on queue_state.json.

Standing rule: every writer of queue_state.json uses t2s_queue.write_queue (this module does too, via
t2s_queue.launch_next / t2s_queue.write_queue) -- never a manual PowerShell edit, which is what produced the UTF-8
BOM that crashed advance() on 2026-09-29 in the first place.

Usage: python queue_watchdog.py   (no arguments; reads/writes C:\\apu\\ovn\\queue_state.json via t2s_queue)
Installed as a scheduled task by scripts/install_queue_watchdog.ps1.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

DEPLOY = Path(__file__).resolve().parent
sys.path.insert(0, str(DEPLOY))
import t2s_queue as q  # noqa: E402

WATCHDOG_LOG = Path(r"C:\apu\ovn\watchdog.log")
RESULTS_DIR = Path(r"C:\apu\ovn\results")


def _log(msg):
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}"
    try:
        with open(WATCHDOG_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
    print(line)


def pid_alive(pid, ps_fn=None):
    """ps_fn is injectable for tests (real usage calls t2s_lab.ps via a lazy import, so this module has no hard
    dependency on t2s_lab -- the watchdog must keep working even if that module fails to import for some reason)."""
    if pid is None:
        return False
    if ps_fn is None:
        import t2s_lab as L  # noqa: WPS433  (lazy: keep the watchdog's own import surface minimal)
        ps_fn = L.ps
    out = ps_fn(f"(Get-Process -Id {pid} -ErrorAction SilentlyContinue | Measure-Object).Count", 20)
    return out.strip() not in ("", "0")


def run_end_written(entry, results_dir=RESULTS_DIR):
    """True if a run_end record exists for this entry's stem in its results .jsonl. The stem is only known if the
    entry's own cmd included --resume <stem>; a fresh (non-resumed) launch generates its own timestamped stem that
    this function cannot predict, so it falls back to the newest .jsonl file modified after the entry's started_ts,
    which is a reasonable proxy: only one orchestrator runs at a time per machine (t2s_queue's own single-running-
    entry invariant), so the newest results file touched since this entry started is that entry's own file."""
    cmd = entry.get("cmd") or []
    stem = None
    if "--resume" in cmd:
        i = cmd.index("--resume")
        if i + 1 < len(cmd):
            stem = cmd[i + 1]
    candidates = []
    if stem:
        p = results_dir / f"{stem}.jsonl"
        if p.exists():
            candidates = [p]
    else:
        started = entry.get("started_ts") or 0
        try:
            candidates = sorted(
                (p for p in results_dir.glob("*.jsonl") if p.stat().st_mtime >= started - 5),
                key=lambda p: p.stat().st_mtime, reverse=True)[:1]
        except OSError:
            candidates = []
    for p in candidates:
        try:
            with open(p, encoding="utf-8", errors="replace") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if r.get("record") == "run_end":
                        return True
        except OSError:
            continue
    return False


def tick(pid_alive_fn=pid_alive, run_end_fn=run_end_written):
    """One watchdog decision cycle. Injectable pid_alive_fn/run_end_fn for tests; production defaults hit the real
    machine. Returns a dict describing what happened, for logging and for tests to assert on."""
    items = q.read_queue()
    running = next((it for it in items if it["status"] == "running"), None)

    if running is not None:
        if pid_alive_fn(running.get("pid")):
            return {"action": "none", "reason": f"entry {running['id']} running, pid {running.get('pid')} alive"}
        if run_end_fn(running):
            # The pid is gone but the run itself finished and wrote its own run_end (and presumably already called
            # advance() before exiting) -- nothing for the watchdog to do; a subsequent tick will see the launched
            # next entry (or an empty queue) instead. Do not double-advance.
            return {"action": "none", "reason": f"entry {running['id']} pid dead but run_end already written"}
        running["status"] = "crashed"
        running["finished_ts"] = time.time()
        running["note"] = f"watchdog: pid {running.get('pid')} dead, no run_end found"
        q.write_queue(items)
        launched = q.launch_next(items)
        if launched is None:
            q.write_empty_flag(items, f"entry {running['id']} crashed and no pending entry to launch")
            return {"action": "crashed_no_next", "reason": running["note"]}
        return {"action": "crashed_launched_next", "reason": running["note"], "launched": launched["id"]}

    pending = next((it for it in items if it["status"] == "pending"), None)
    if pending is not None:
        launched = q.launch_next(items)
        return {"action": "launched", "launched": launched["id"] if launched else None}

    q.write_empty_flag(items, "watchdog: nothing running and nothing pending")
    return {"action": "empty_flag"}


def main():
    result = tick()
    _log(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
