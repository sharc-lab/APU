"""Machine-independent results digest: reads every results JSONL under a run folder plus that machine's own
queue_state.json, and writes a markdown digest with one table per phase (per-cell counts, key metrics where the row
shape exposes them, contamination tags, and the UTC time of the last cell), plus an idle alert line when the queue
has pending work but nothing has measured anything recently.

Why this exists (2026-09-29): results must not depend on a controller session being awake to notice them, or to
notice a machine sitting idle. This script is read-only end to end -- it must never start, stop, or kill any
process, and never touches queue_state.json or any results file; it only reads them and writes RESULTS_DIGEST.md.

2026-09-30: added a second, distinct alert alongside the idle one -- check_idle above only proves the running job's
process/log is alive; it says nothing about whether that process is actually producing result rows. A job can look
"running, heartbeat fresh" for hours after it has effectively stalled (stuck retrying, looping on a guard failure,
or otherwise not completing calls) while its log file keeps getting touched. harness/queue_watchdog.check_progress_
stale reads this script's own collect_phase_summaries() output (the real last_ts_utc across every phase) and alerts
if nothing has been written anywhere in over an hour (two hours for a 70B or 96K+-prompt phase, where a single call
can legitimately take minutes) -- see that function's own docstring for why it checks globally across phases rather
than trying to map a queue entry id to one exact phase/section key.

Usage (run locally on evo-t2s or evo-x2, where C:\\apu\\ovn\\results and C:\\apu\\ovn\\queue_state.json live):
  python results_digest.py --out C:\\apu\\ovn\\RESULTS_DIGEST.md

Usage against a pulled-back copy from the controller (testing, or building the repo's docs/RESULTS_DIGEST.md from
copies of both machines' results dirs):
  python results_digest.py --results-dir <dir> --queue-file <queue_state.json> --out <path> --machine evo-t2s

The queue job "digest" (see harness/t2s_queue.py-style entries) runs this after every job on both machines. A
separate scheduled task runs it every 15 minutes regardless of queue activity, specifically so the idle alert does
not depend on a job finishing to get noticed.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

# queue_watchdog.py is deployed flat alongside this file on both machines (scripts/deploy_evo.py copies everything
# into C:\apu\ovn regardless of its source subdirectory in the repo), so this import resolves there. Locally, add
# harness/ to the path the same way every other analysis/*.py script that reuses harness code does.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
import queue_watchdog as _wd  # noqa: E402

# Fields that, when present and non-null on a row, are treated as a numeric metric worth summarizing per phase.
# Deliberately generic (not hardcoded per phase) so this does not need updating every time a new phase is added;
# see the module docstring for the tradeoff (less rich than a bespoke per-phase table, but truthful about what it
# actually knows and does not silently go stale when a new phase's field names differ slightly).
METRIC_FIELDS = ("ttft_s", "decode_tok_s", "e2e_s", "resp_median_s", "resp_max_s", "score", "ratio")
CONTAMINATION_PREFIX = "contaminated_"


def _iter_jsonl(path: Path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except OSError:
        return


def _phase_key(row):
    """Groups rows into a phase bucket. Prefers 'section' (the field every t2s_overnight-style call/start row
    carries), then 'record' (for summary-only records like bisect_result or a70_arm_result that have no section),
    else 'unclassified'."""
    return row.get("section") or row.get("record") or "unclassified"


def collect_phase_summaries(jsonl_paths):
    """Reads every row from every given jsonl path and groups it by phase. Returns {phase: {"n": int,
    "last_ts_utc": str|None, "metrics": {field: {"n", "median", "min", "max"}}, "contamination_tags": {tag: count},
    "outcomes": {value: count}, "files": [str, ...]}}."""
    phases = {}
    for path in jsonl_paths:
        for row in _iter_jsonl(path):
            if not isinstance(row, dict):
                continue
            key = _phase_key(row)
            p = phases.setdefault(key, {"n": 0, "last_ts_utc": None, "metrics": {}, "contamination_tags": {},
                                        "outcomes": {}, "files": set()})
            p["n"] += 1
            p["files"].add(str(path))
            ts = row.get("ts_utc")
            if isinstance(ts, str) and (p["last_ts_utc"] is None or ts > p["last_ts_utc"]):
                p["last_ts_utc"] = ts
            for field in METRIC_FIELDS:
                val = row.get(field)
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    p["metrics"].setdefault(field, []).append(val)
            for k, v in row.items():
                if k.startswith(CONTAMINATION_PREFIX) and v:
                    p["contamination_tags"][k] = p["contamination_tags"].get(k, 0) + 1
                if k == "tag" and isinstance(v, str) and v.startswith(CONTAMINATION_PREFIX):
                    p["contamination_tags"][v] = p["contamination_tags"].get(v, 0) + 1
            for field in ("outcome", "ok", "reproduced", "started"):
                if field in row:
                    val = row.get(field)
                    label = f"{field}={val}"
                    p["outcomes"][label] = p["outcomes"].get(label, 0) + 1
    for p in phases.values():
        summarized = {}
        for field, vals in p["metrics"].items():
            summarized[field] = {"n": len(vals), "median": round(st.median(vals), 4),
                                 "min": round(min(vals), 4), "max": round(max(vals), 4)}
        p["metrics"] = summarized
        p["files"] = sorted(p["files"])
    return phases


def check_idle(queue_items, log_dir, now=None):
    """Read-only idle check: if the queue has a "running" entry, uses that entry's own queue_<id>.log mtime as the
    liveness signal (same heartbeat harness/queue_watchdog.py uses, imported directly rather than duplicated, so
    the two never drift apart) -- this function never touches pid_alive, so it cannot ever be mistaken for
    something that starts, stops, or kills a process. Stale threshold is the same as the watchdog's own: 30 min, or
    60 min for a 70B-model phase (not a flat 15 min -- a single 70B call can legitimately take minutes). Returns an
    ALERT line (str) or None. Also alerts if there is pending work and nothing is running at all (a stalled queue)."""
    now = now if now is not None else time.time()
    running = next((it for it in queue_items if it.get("status") == "running"), None)
    pending = [it for it in queue_items if it.get("status") == "pending"]
    if running is None:
        if pending:
            return f"queue has {len(pending)} pending job(s) and nothing running"
        return None
    age = _wd.heartbeat_age_s(running, log_dir=log_dir, now=now)
    if _wd.is_stale(running, age):
        age_str = f"{age:.0f}s" if age is not None else "no log file found"
        threshold = _wd.HEARTBEAT_STALE_S_70B if _wd.is_70b_phase(running) else _wd.HEARTBEAT_STALE_S
        return f"job {running['id']} has not written to its log in {age_str} (threshold {threshold}s)"
    return None


def render_markdown(machine, phases, idle_alert=None, progress_alert=None, generated_ts_utc=None):
    lines = []
    if idle_alert:
        lines.append(f"**ALERT: {machine} idle since check, reason: {idle_alert}**")
        lines.append("")
    if progress_alert:
        lines.append(f"**ALERT: {machine} progress stale, reason: {progress_alert}**")
        lines.append("")
    lines.append(f"# Results digest -- {machine}")
    lines.append("")
    lines.append(f"Generated {generated_ts_utc or time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}. "
                 "Read-only: this digest never starts, stops, or kills anything.")
    lines.append("")
    for phase in sorted(phases):
        p = phases[phase]
        lines.append(f"## {phase}")
        lines.append("")
        lines.append(f"- cells: {p['n']}")
        lines.append(f"- last cell UTC: {p['last_ts_utc'] or 'unknown'}")
        if p["contamination_tags"]:
            tags = ", ".join(f"{k} ({v})" for k, v in sorted(p["contamination_tags"].items()))
            lines.append(f"- contamination tags: {tags}")
        if p["outcomes"]:
            outs = ", ".join(f"{k}: {v}" for k, v in sorted(p["outcomes"].items()))
            lines.append(f"- outcomes: {outs}")
        if p["metrics"]:
            lines.append("")
            lines.append("| metric | n | median | min | max |")
            lines.append("|---|---|---|---|---|")
            for field, m in sorted(p["metrics"].items()):
                lines.append(f"| {field} | {m['n']} | {m['median']} | {m['min']} | {m['max']} |")
        lines.append("")
    return "\n".join(lines)


def build_arg_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=r"C:\apu\ovn\results")
    ap.add_argument("--queue-file", default=r"C:\apu\ovn\queue_state.json")
    ap.add_argument("--out", default=r"C:\apu\ovn\RESULTS_DIGEST.md")
    ap.add_argument("--machine", default=None, help="label for the digest header; defaults to the queue file's directory name")
    return ap


def main():
    args = build_arg_parser().parse_args()
    results_dir = Path(args.results_dir)
    jsonl_paths = sorted(results_dir.glob("*.jsonl")) if results_dir.is_dir() else []
    phases = collect_phase_summaries(jsonl_paths)

    queue_items = []
    qpath = Path(args.queue_file)
    if qpath.exists():
        try:
            queue_items = json.loads(qpath.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            queue_items = []
    idle_alert = check_idle(queue_items, qpath.parent) if queue_items else None
    progress_alert = _wd.check_progress_stale(phases, queue_items) if queue_items else None

    machine = args.machine or qpath.parent.name or "unknown"
    md = render_markdown(machine, phases, idle_alert=idle_alert, progress_alert=progress_alert)
    Path(args.out).write_text(md, encoding="utf-8")
    alerts = ", ".join(a for a in (idle_alert, progress_alert) if a)
    print(f"wrote {args.out}: {len(phases)} phases, {sum(p['n'] for p in phases.values())} rows"
          f"{', ' + alerts if alerts else ''}")


if __name__ == "__main__":
    main()
