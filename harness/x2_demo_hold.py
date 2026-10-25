"""Demo hold on evo-x2 (operational entry, pre-registered in docs/EXPERIMENTS.md "2026-10-23: Demo hold job"): keeps
the queue from launching the next measurement while the live demo uses the machine, and releases itself.

It touches nothing: no Ollama start or stop, no model, no GPU, no file other than its own output rows and its stdout
(which the queue writes to queue_<id>.log, the watchdog's heartbeat). It waits until a fixed release time, printing a
heartbeat line every HEARTBEAT_S, then calls t2s_queue.advance() once. If it starts at or after the release time it
calls advance() at once. It sets no pause flag, so the release is automatic and does not depend on anyone.

    py -3.12 demo\\x2_demo_hold.py --start-utc 2026-10-29T15:00:00+00:00 --release-utc 2026-10-30T01:00:00+00:00
        --out C:\\apu\\ovn\\results\\demo_hold.jsonl --queued
    py -3.12 demo\\x2_demo_hold.py --start-utc ... --release-utc ... --dry-run   (prints the plan; no wait, no advance)

Rolling hold (--rolling, 2026-10-24, for the Tue 2026-10-27 showing; --warmup-min 30 frees the machine 30 minutes
before --start-utc for a warm-up run). The queue has no time gate, so the hold is inserted directly after the running
job and decides each time it is launched (decide()): if the next pending job fits entirely before the window start by
its HIGH estimate (the "high X h" in its note, else 1.5 x projected_duration_h), the hold re-inserts itself right after
that job and releases, so the job runs and the hold decides again when it ends; otherwise the hold keeps the queue held
until --release-utc and then releases with its own tq.advance(). A running job is never interrupted. If the job that is
already running would overrun the window by its own high estimate, nothing here can change that; the dry run and
check-plan say so.
"""
from __future__ import annotations

import argparse
import json
import re
import socket
import sys
import time
from datetime import datetime, timedelta, timezone
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


# ── rolling hold (2026-10-24, Tue 2026-10-27 showing): let a job ahead run only if it fits before the window ────────
HIGH_FACTOR_DEFAULT = 1.5    # a queue entry whose note states no "high X h" estimate: high = 1.5 x projected_duration_h
_HIGH_RE = re.compile(r"high ([\d.]+) h")


def high_estimate_h(entry: dict) -> float | None:
    """The entry's high duration estimate in hours: the "high X h" its note states (every entry x2_queue_fill writes
    states one), else HIGH_FACTOR_DEFAULT x projected_duration_h; None if neither is known (treated as not fitting)."""
    m = _HIGH_RE.search(entry.get("note") or "")
    if m:
        return float(m.group(1))
    p = entry.get("projected_duration_h")
    return None if p is None else HIGH_FACTOR_DEFAULT * float(p)


def next_pending(items: list[dict], after_id: str) -> dict | None:
    ids = [it["id"] for it in items]
    for it in items[ids.index(after_id) + 1:]:
        if it.get("status") == "pending":
            return it
    return None


def decide(now: datetime, hold_start: datetime, end: datetime, nxt: dict | None) -> dict:
    """What the hold does when it is launched (pure).
      release : now >= end (the window is over, or the hold started after it).
      defer   : the next pending job fits entirely before hold_start by its HIGH estimate (now + high <= hold_start);
                the hold re-inserts itself right after that job and releases, so the job runs now and the hold
                re-evaluates when it ends. A running job is never interrupted: the hold only ever runs between jobs.
      hold    : otherwise (no next job, its estimate unknown, or it would overrun hold_start by its high estimate);
                the queue stays held from now until end, then releases."""
    if now >= end:
        return {"action": "release", "why": f"now {now.isoformat()} is at or after the release {end.isoformat()}"}
    if nxt is None:
        return {"action": "hold", "why": "no pending job after the hold"}
    hi = high_estimate_h(nxt)
    if hi is None:
        return {"action": "hold", "why": f"{nxt['id']} has no duration estimate"}
    finish = now + timedelta(hours=hi)
    if finish <= hold_start:
        return {"action": "defer", "next": nxt["id"], "high_h": hi, "finish_high_utc": finish.isoformat(),
                "why": f"{nxt['id']} fits before {hold_start.isoformat()} by its high estimate {hi:.2f} h"}
    return {"action": "hold", "next": nxt["id"], "high_h": hi, "finish_high_utc": finish.isoformat(),
            "why": f"{nxt['id']} would end at {finish.isoformat()} by its high estimate {hi:.2f} h, after the window "
                   f"start {hold_start.isoformat()}"}


def deferred_items(items: list[dict], hold_id: str, next_id: str) -> list[dict]:
    """Pure: the queue with a fresh pending copy of the running hold entry inserted right after next_id (id gets a
    _r<n> suffix). The running entry itself is not changed; t2s_queue.advance() then marks it done and launches next_id."""
    items = [dict(it) for it in items]
    ids = [it["id"] for it in items]
    me = items[ids.index(hold_id)]
    base = hold_id.split("_r")[0] if "_r" in hold_id and hold_id.rsplit("_r", 1)[1].isdigit() else hold_id
    n = 1 + max([int(i.rsplit("_r", 1)[1]) for i in ids if i.startswith(base + "_r") and
                 i.rsplit("_r", 1)[1].isdigit()] or [0])
    copy = {k: v for k, v in me.items() if k in ("cmd", "note", "projected_duration_h", "gate")}
    copy.update({"id": f"{base}_r{n}", "status": "pending"})
    items.insert(ids.index(next_id) + 1, copy)
    return items


def simulate(items: list[dict], now: datetime, hold_start: datetime, end: datetime) -> dict:
    """Read-only projection for --check-plan (pure): the hold inserted right after the running job; the running job
    ends by its HIGH estimate from its started_ts; then decide() is applied job by job, each deferred job also taking
    its high estimate. Says which jobs run before the window, when the hold engages at the latest, and whether the
    running job itself could overrun the window (which the hold cannot prevent: it never interrupts a job)."""
    running = next((it for it in items if it.get("status") == "running"), None)
    t, warn = now, None
    if running is not None:
        hi = high_estimate_h(running)
        st = running.get("started_ts")
        if hi is not None and st is not None:
            t = max(now, datetime.fromtimestamp(float(st), timezone.utc) + timedelta(hours=hi))
        if hi is None or t > hold_start:
            warn = (f"the running job {running['id']} could still be running at the window start "
                    f"{hold_start.isoformat()} (high end {t.isoformat()}); the hold cannot interrupt it")
    pend = [it for it in items if it.get("status") == "pending"]
    ran = []
    for it in pend:
        d = decide(t, hold_start, end, it)
        if d["action"] != "defer":
            break
        t += timedelta(hours=d["high_h"])
        ran.append({"id": it["id"], "high_h": d["high_h"], "ends_by_high_utc": t.isoformat()})
    return {"running": running["id"] if running else None, "hold_engages_by_utc": t.isoformat(),
            "jobs_run_before_window": ran, "warning": warn, "hold_start_utc": hold_start.isoformat(),
            "release_utc": end.isoformat()}


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


def _tq():
    here = Path(__file__).resolve().parent
    if not (here / "t2s_queue.py").exists():
        sys.path.insert(0, str(here.parent))
    import t2s_queue as tq
    return tq


def rolling_step(tq, hold_start: datetime, end: datetime, emit, clock=lambda: datetime.now(timezone.utc)) -> dict:
    """--rolling, at launch: read the queue, decide (decide()), and for "defer" re-insert this hold right after the
    next job in one read-modify-write under a pause cleared in this function's finally block (released within
    seconds; the reason names that release). Returns the decision."""
    items = tq.read_queue()
    me = next((it for it in items if it.get("status") == "running"), None)
    if me is None or "x2_demo_hold.py" not in " ".join(map(str, me.get("cmd") or [])):
        raise RuntimeError("the running queue entry is not this hold")
    d = decide(clock(), hold_start, end, next_pending(items, me["id"]))
    emit({"record": "decision", "hold_id": me["id"], **d})
    if d["action"] == "defer":
        tq.set_pause(f"x2_demo_hold {me['id']}: one read-modify-write re-inserting the hold after {d['next']}; "
                     "released by clear_pause() in the same function's finally block within seconds")
        try:
            tq.write_queue(deferred_items(tq.read_queue(), me["id"], d["next"]))
        finally:
            tq.clear_pause()
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release-utc", required=True)
    ap.add_argument("--start-utc", default=None, help="the demo window start (the showing time; ISO with offset)")
    ap.add_argument("--warmup-min", type=float, default=0.0,
                    help="minutes before --start-utc the machine must already be free (the warm-up run)")
    ap.add_argument("--rolling", action="store_true",
                    help="let the next job run first if it fits before the window by its high estimate (decide())")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check-plan", action="store_true",
                    help="read the queue (read-only) and project the rolling hold with high estimates (simulate())")
    ap.add_argument("--queued", action="store_true", help="call t2s_queue.advance() on exit")
    a = ap.parse_args(argv)
    release = parse_release(a.release_utc)
    start = parse_release(a.start_utc) if a.start_utc else None
    hold_start = (start - timedelta(minutes=a.warmup_min)) if start else None
    if a.check_plan:
        if hold_start is None:
            ap.error("--check-plan needs --start-utc")
        print(json.dumps({"check_plan": True, "host": socket.gethostname(), "warmup_min": a.warmup_min,
                          **simulate(_tq().read_queue(), datetime.now(timezone.utc), hold_start, release)}, indent=1))
        return 0
    if a.dry_run:
        print(json.dumps({"dry_run": True, "host": socket.gethostname(), "warmup_min": a.warmup_min,
                          "hold_start_utc": hold_start.isoformat() if hold_start else None,
                          **plan(release, datetime.now(timezone.utc), hold_start)}))
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
        if a.rolling and a.queued and hold_start is not None:
            d = rolling_step(_tq(), hold_start, release, emit)
            if d["action"] == "defer":
                note = f"completed: demo hold deferred behind {d['next']} (re-inserted after it; releases at " \
                       f"{release.isoformat()})"
            elif d["action"] == "hold":
                hold(release, emit, start=hold_start)
        else:
            hold(release, emit, start=hold_start)
    except Exception as e:
        note = f"stopped: demo hold error {e!r}"[:300]
    finally:
        if a.queued:
            try:
                _tq().advance(note)
            except Exception as e:
                print(f"queue advance failed: {e!r}")
    print(note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
