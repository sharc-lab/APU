"""Plan and checks for the evo-t2s window 2026-10-26 to 2026-11-08 (pending Zach's confirmation). NOT loaded anywhere.

Canonical staged file: queues/t2s_oct26_queue.json, built by scripts/t2s_week_queue.py build-oct26 (one builder, gates
on its own t2s_o26_preflight, loaded at the window with `python t2s_week_queue.py load-window t2s_oct26_queue.json`).
The earlier second file queues/t2s_window_20261026_staged.json (B5 for llama3.1:8b on the iGPU only, 3 seeds, and a
3-seed iGPU-only B6, gated on the previous window's t2s_wk_* ids) was retired on 2026-10-22: two files for one window
with different B6 designs and gates pointing at a finished queue invited loading the wrong one. B5 is now in the
canonical file as four entries (t2s_week_queue.OCT26_B5_CELLS), and the B6 entries follow B5.

  py -3.12 scripts/t2s_window_stage.py build      # rebuilds queues/t2s_oct26_queue.json (same as build-oct26)
  py -3.12 scripts/t2s_window_stage.py dry-run    # every entry through the week queue's checks; no machine, no SSH
  py -3.12 scripts/t2s_window_stage.py plan       # hours per entry (ESTIMATES) and what fits in the window

Order of the window (docs/T2S_WEEK_PLAN.md "Staged for 2026-10-26"; process rule 2026-10-20, docs/STATE.md):
  1. On the CONTROLLER, first: `py -3.12 harness/t2s_deadline.py open-window --no-launch-after <NO_LAUNCH_AFTER_UTC>
     --hard-deadline <HARD_DEADLINE_UTC> --reason "<handover to Zach ...; resume only in the next confirmed window>"`.
     It runs the handover check on evo-t2s (HANDOVER_DONE.json, the pause flag naming its releaser, no process of
     ours; refuses and changes nothing on failure), then widens the results sync to both hosts (an expiring window
     file; the APU-SyncResults task keeps --host evo-x2), arms the handover deadline, waits to the hard deadline, and
     removes the sync window file in its finally block. The same job that arms the deadline is the one that switches
     the sync and switches it back.
  2. Deploy the window's files, then `python t2s_week_queue.py load-window t2s_oct26_queue.json` (one read-modify-write
     under the queue pause, cleared in its finally block: this is the release of the handover pause).
  3. The queue runs: preflight, A5, B5 (four cells), then B6. The B5 job stops at a session boundary at the deadline.

HOURS ARE ESTIMATES. B5: harness/policy_compare.estimate_t2s, from MEASURED evo-t2s session minutes
(register row t2s-r2-session-minutes); B6: no timing of those models on evo-t2s, no estimate.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO / "scripts"))

import t2s_week_queue as wq  # noqa: E402

CANONICAL = wq.OCT26_PATH
RETIRED = REPO / "queues" / "t2s_window_20261026_staged.json"
WINDOW = ("2026-10-26", "2026-11-08")
WINDOW_START_UTC = "2026-10-26T04:00:00+00:00"       # 2026-10-26 00:00 EDT
NO_LAUNCH_AFTER_UTC = "2026-11-09T04:00:00+00:00"    # 2026-11-08 23:00 EST (DST ends 2026-11-01)
HARD_DEADLINE_UTC = "2026-11-09T04:30:00+00:00"      # 2026-11-08 23:30 EST
START_ALLOWANCE_H = 1.0   # ASSUMED: handover check, deploy, load and preflight before the first launch (not measured)
NOMINAL_H = {wq.OCT26_PREFIX + "window_override_methods": 0.3}   # the entry's own nominal (no evo-t2s timing)


def _ts(s):
    return datetime.fromisoformat(s).timestamp()


def available_h() -> float:
    return (_ts(NO_LAUNCH_AFTER_UTC) - _ts(WINDOW_START_UTC)) / 3600.0 - START_ALLOWANCE_H


def plan(items=None) -> dict:
    """Entries in queue order with estimated hours, cumulative start/end, and whether each fits before the no-launch
    time. A B5 entry that starts but does not finish gets the number of whole seeds that fit (the job stops at a
    session boundary at the deadline; a cell counts for the verdict at 5 seeds, or at the first 3)."""
    import policy_compare as pc
    items = items if items is not None else json.loads(CANONICAL.read_text(encoding="utf-8"))
    cells = {wq.OCT26_PREFIX + "b5_" + wq._b5_slug(p, m): (p, m) for p, m in wq.OCT26_B5_CELLS}
    est = {p: pc.estimate_t2s(p) for p in {p for p, _m in wq.OCT26_B5_CELLS}}
    avail, cum, rows = available_h(), 0.0, []
    for it in items:
        h = it.get("est_hours")
        basis = it.get("est_basis")
        if h is None and it["id"] in NOMINAL_H:
            h, basis = NOMINAL_H[it["id"]], "nominal"
        r = {"id": it["id"], "est_hours": h, "basis": basis, "start_h": round(cum, 1)}
        if h is None:
            r.update(fits="no estimate" if cum < avail else "not reached", end_h=None)
        else:
            end = cum + h
            r["end_h"] = round(end, 1)
            if end <= avail:
                r["fits"] = "yes"
            elif cum >= avail:
                r["fits"] = "not reached"
            else:
                r["fits"] = "partial"
            if it["id"] in cells:
                p, m = cells[it["id"]]
                e = est[p]["models"][m]
                fixed = e["probe_h"]
                k = 0 if cum >= avail else min(5, max(0, math.floor((avail - cum - fixed) / e["per_seed_h"])))
                r.update(per_seed_h=round(e["per_seed_h"], 1), seeds_that_fit=k,
                         cell_counts=("yes (5 seeds)" if k >= 5 else "yes (first 3 seeds)" if k >= 3 else "no"))
            cum = end
        rows.append(r)
    b5 = [r for r in rows if r["id"] in cells]
    return {"label": "ESTIMATES (hours), not measurements", "window": WINDOW, "window_start_utc": WINDOW_START_UTC,
            "no_launch_after_utc": NO_LAUNCH_AFTER_UTC, "hard_deadline_utc": HARD_DEADLINE_UTC,
            "start_allowance_h": START_ALLOWANCE_H, "available_h": round(avail, 1), "entries": rows,
            "b5_total_h": round(sum(r["est_hours"] for r in b5), 1),
            "b5_cells_counting": [r["id"] for r in b5 if r.get("cell_counts", "no") != "no"],
            "estimate_basis": pc.T2S_ESTIMATE_ASSUMPTIONS}


def plan_value(p=None) -> str:
    p = p or plan()
    parts = []
    for r in p["entries"]:
        s = f"{r['id']} {('n/a' if r['est_hours'] is None else r['est_hours'])} h ({r['fits']}"
        if "seeds_that_fit" in r:
            s += f"; {r['per_seed_h']} h/seed, {r['seeds_that_fit']} of 5 seeds fit, cell counts {r['cell_counts']}"
        parts.append(s + ")")
    return (f"evo-t2s window {p['window'][0]} to {p['window'][1]} ESTIMATE: available {p['available_h']} h "
            f"({p['window_start_utc']} to no-launch {p['no_launch_after_utc']} minus an assumed "
            f"{p['start_allowance_h']} h start allowance); B5 all four cells {p['b5_total_h']} h; in queue order: "
            + "; ".join(parts) + f"; B5 cells that count: {', '.join(p['b5_cells_counting']) or 'none'}; basis: "
            + p["estimate_basis"])


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "plan"
    if cmd == "build":
        CANONICAL.write_text(json.dumps(wq.build_oct26(), indent=1) + "\n", encoding="utf-8")
        print(f"wrote {CANONICAL} (staged, NOT loaded)")
        return 0
    if cmd == "dry-run":
        items = json.loads(CANONICAL.read_text(encoding="utf-8"))
        rep = wq.dry_run(items, prefix=wq.OCT26_PREFIX)
        rep["fresh_build_matches_file"] = items == wq.build_oct26()
        rep["retired_file_absent"] = not RETIRED.exists()
        print(json.dumps(rep, indent=1, default=str))
        return 0 if rep["ok"] and rep["fresh_build_matches_file"] and rep["retired_file_absent"] else 1
    if cmd == "plan":
        print(json.dumps(plan(), indent=1))
        return 0
    raise SystemExit(f"unknown command {cmd!r}")


if __name__ == "__main__":
    sys.exit(main())
