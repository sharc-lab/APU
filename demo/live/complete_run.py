"""Complete the 5-task acceptance run on the controller: the evo-x2 job (harness/demo_live_x2.py --mode acceptance)
recorded the local parts with the cloud parts pending; this fills each pending cloud part (Ours routed to the cloud or
escalated after a flag, and the All cloud lane) through demo/live/cloud_lane.CloudLane and writes the final run file
with the tally and run_end.

    $env:DEMO_CLOUD_LIVE = "1"   # plus the key in this process only (never printed)
    py -3.12 -m demo.live.complete_run --local results/demo_live_run_local_<date>.jsonl \
        --out results/demo_live_run_<date>.jsonl

Without live calls the cloud parts are REPLAYED (labelled) or "not available"; the run file says which.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from demo.live import cloud_lane as CL

L = CL.L
import demo_run as DR  # noqa: E402


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def complete(local_rows: list[dict], lane: CL.CloudLane, b3_rows: list[dict], emit) -> list[dict]:
    start = next(r for r in local_rows if r.get("record") == "run_start")
    if not any(r.get("record") == "run_end" for r in local_rows):
        raise ValueError("the evo-x2 part has no run_end; refusing to complete an unfinished run")
    emit({"record": "run_start", "harness": "demo_live_run", "part": "complete (evo-x2 local + controller cloud)",
          "local_run_start": start, "ledger_total_before_usd": round(CL.ES.ledger_total(lane.ledger), 6),
          "live_possible": lane.live_possible()[0]})
    out = []
    for r in local_rows:
        if r.get("record") != "task_result":
            continue
        task = L.build_task(b3_rows, r["seed"], r["task_id"], with_cloud=True)
        lanes = dict(r["lanes"])
        o = lanes["ours"]
        if o.get("route") == "cloud" and o.get("pending"):
            lanes["ours"] = {**lane("ours", task, o["model"]), "route": "cloud", "decision": o.get("decision")}
        elif o.get("escalated") and o.get("pending"):
            c = lane("ours", task, o["cloud"]["model"])
            lanes["ours"] = {**o, "cloud": c, "end_state_ok": c.get("end_state_ok"), "pending": False}
        if lanes["cloud"].get("pending"):
            lanes["cloud"] = {**lane("cloud", task, lanes["cloud"]["model"]), "route": "cloud"}
        row = {**r, "lanes": lanes}
        emit(row)
        out.append(row)
    by_sc = {}
    for row in out:
        by_sc.setdefault(row["scenario"], []).append(row)
    for sc, rows in by_sc.items():
        emit({"record": "tally", "scenario": sc, "tally": DR.tally(rows)})
    emit({"record": "run_end", "ledger_total_after_usd": round(CL.ES.ledger_total(lane.ledger), 6),
          "session_spend_usd": round(lane.session_spend, 6)})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--session-limit", type=float, default=CL.SESSION_LIMIT_USD,
                    help="live cloud spend limit for this session in USD (default 3; above 3 needs "
                         "--operator-approval)")
    ap.add_argument("--operator-approval", default=None,
                    help="operator only: approve this session above USD 3, e.g. "
                         "'approve demo_live_session up to USD <n>: <reason>'; written to the demo ledger before the "
                         "first live call; never raises the USD 10 cap or skips the Console check")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite a run file")
    local_rows = L.read_jsonl(a.local)
    b3 = L.read_jsonl(CL.REPO / "results" / L.B3_FILE)

    def emit(row):
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts_utc": utc_iso(), **row}, default=str) + "\n")
    lane = CL.CloudLane(session_limit=a.session_limit, operator_approval=a.operator_approval)
    complete(local_rows, lane, b3, emit)
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
