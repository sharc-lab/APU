"""Quantify WSH contamination windows for X-2 cpu-p cells."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "derived" / "bfcl_feasibility" / "x2_feasibility_table"


def parse(u: str) -> datetime:
    return datetime.fromisoformat(u.replace("Z", "+00:00"))


def main() -> None:
    out: dict = {"cells": []}
    for sid in (
        "cb781dbf-3486-4fbc-a69a-34026f801abe",
        "9fdedb46-3318-4abc-a56f-50b7d23d25ca",
        "afd1aa21-d4b2-4491-81b4-b1b6f4fa681a",
        "0963168f-9144-4d21-99cf-5a77232dd477",
    ):
        d = BASE / sid
        wd = [
            json.loads(l)
            for l in (d / "watchdog_kills.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        led = json.loads((d / "x2_entry_ledger.json").read_text(encoding="utf-8-sig"))
        plan = json.loads((d / "plan.json").read_text(encoding="utf-8-sig"))
        events = sorted(wd, key=lambda e: e.get("utc") or "")
        kills = [e for e in events if e.get("event") == "watchdog_kill"]
        gaps = []
        for i, e in enumerate(events):
            if e.get("event") != "watchdog_kill":
                continue
            prev = next(
                (x for x in reversed(events[:i]) if x.get("event") == "watchdog_poll"),
                None,
            )
            gap = (parse(e["utc"]) - parse(prev["utc"])).total_seconds() if prev else None
            gaps.append(
                {
                    "utc": e["utc"],
                    "n_found": e.get("n_found"),
                    "pids_found": e.get("pids_found"),
                    "gap_from_prev_poll_s": gap,
                }
            )
        # Upper-bound contamination: each kill implies WSH present for < poll interval
        # since previous clean poll (or since start). Use measured gap_from_prev_poll.
        bound_s = sum(g["gap_from_prev_poll_s"] or 60.0 for g in gaps)
        started = plan.get("started_utc")
        ended = plan.get("ended_utc") or led.get("ended_utc")
        wall_s = None
        if started and ended:
            wall_s = (parse(ended) - parse(started)).total_seconds()
        out["cells"].append(
            {
                "session_id": sid,
                "arm_cli": plan.get("arm_cli"),
                "residency_mode": plan.get("residency_mode"),
                "n_kills": len(kills),
                "kill_detail": gaps,
                "upper_bound_contaminated_s": bound_s,
                "session_wall_s": wall_s,
                "upper_bound_contaminated_fraction": (
                    bound_s / wall_s if wall_s and wall_s > 0 else None
                ),
                "watchdog_interval_s": (plan.get("watchdog") or {}).get("interval_s"),
            }
        )
    dest = BASE / "x2_wsh_contamination_bound.json"
    dest.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))
    print("WROTE", dest)


if __name__ == "__main__":
    main()
