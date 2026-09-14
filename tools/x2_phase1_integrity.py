"""X-2 Phase 1 integrity audit — report only, no seal."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "derived" / "bfcl_feasibility" / "x2_feasibility_table"
A621_ENTRIES = (
    ROOT
    / "derived"
    / "bfcl_feasibility"
    / "session_residency"
    / "a621ff7d-2919-463d-aaf6-673f9e6bafbc"
    / "session_residency_entries_gpu_only_RESIDENT.json"
)
IR_PIN = "c1821f29332faa48871c7f16426a21443fbc701fa4e4c8a581a8c51ab7bf2cb2"

CELLS = [
    ("cb781dbf-3486-4fbc-a69a-34026f801abe", "cpu-p", "NON_RESIDENT"),
    ("9fdedb46-3318-4abc-a56f-50b7d23d25ca", "cpu-p", "RESIDENT"),
    ("afd1aa21-d4b2-4491-81b4-b1b6f4fa681a", "gpu_only", "NON_RESIDENT"),
    ("0963168f-9144-4d21-99cf-5a77232dd477", "gpu_only", "RESIDENT"),
]


def load(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8-sig"))


def ids(path: Path) -> list[str]:
    return [str(e["id"]) for e in load(path)]


def audit_cell(sid: str, arm: str, mode: str) -> dict[str, Any]:
    d = BASE / sid
    plan = load(d / "plan.json")
    summary = load(d / "summary.json")
    ledger = load(d / "x2_entry_ledger.json")
    gold = load(d / "multi_turn_gold_selftest.json")
    entries = ids(d / "multi_turn_probe_entries.json")
    ref = ids(A621_ENTRIES)
    wd_path = d / "watchdog_kills.jsonl"
    wd_lines = [
        json.loads(ln) for ln in wd_path.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    kills = [e for e in wd_lines if e.get("event") == "watchdog_kill"]
    polls = [e for e in wd_lines if e.get("event") == "watchdog_poll"]
    start = [e for e in wd_lines if e.get("event") == "watchdog_start_kill"]

    # Timed work window: first entry uptime to last entry uptime (+ session end).
    per = ledger.get("per_entry") or []
    onset_missing = [
        r.get("id") for r in per if r.get("uptime_s") is None or r.get("available_mb") is None
    ]
    avail = [float(r["available_mb"]) for r in per if r.get("available_mb") is not None]
    uptimes = [float(r["uptime_s"]) for r in per if r.get("uptime_s") is not None]

    # WSH resident during timed work if any kill during/after first entry,
    # or any poll with n_found>0 overlapping the entry window.
    # Kill events mean WSH was found and killed — contamination window until kill.
    # Strict: any watchdog_kill after start, or poll n_found>0, fails the clean claim.
    wsh_seen = []
    for e in wd_lines:
        if e.get("event") == "watchdog_kill" and int(e.get("n_found") or 0) > 0:
            wsh_seen.append(e)
        if e.get("event") == "watchdog_poll" and int(e.get("n_found") or 0) > 0:
            wsh_seen.append(e)
        if e.get("event") == "watchdog_start_kill" and int(e.get("n_killed") or 0) > 0:
            wsh_seen.append(e)

    ir = (summary.get("ir_sha256") or plan.get("ir_sha256") or "").lower()
    traj = summary.get("accuracy_trajectory") or {}
    slo = summary.get("fraction_turns_slo_ok")
    lat = summary.get("session_total_latency_s") or {}

    return {
        "session_id": sid,
        "arm_cli": arm,
        "residency_mode": mode,
        "status": summary.get("status"),
        "entry_ids_exact_a621": entries == ref,
        "n_entries": len(entries),
        "n_ref": len(ref),
        "gold_n": gold.get("n"),
        "gold_n_valid": gold.get("n_valid"),
        "gold_20_20": gold.get("n") == 20 and gold.get("n_valid") == 20,
        "gold_before_generation": plan.get("gold_selftest") is not None,
        "ir_sha256": ir,
        "ir_ok": ir == IR_PIN,
        "model_spec": summary.get("model_spec") or plan.get("model_spec"),
        "watchdog_line_count": len(wd_lines),
        "watchdog_start_events": len(start),
        "watchdog_poll_events": len(polls),
        "watchdog_kill_events": len(kills),
        "watchdog_kills": [
            {
                "utc": e.get("utc"),
                "n_found": e.get("n_found"),
                "pids_found": e.get("pids_found"),
                "n_killed": e.get("n_killed"),
            }
            for e in kills
        ],
        "wsh_seen_in_watchdog_log": len(wsh_seen) > 0,
        "wsh_clean_during_run": len(wsh_seen) == 0,
        "wsh_seen_events": wsh_seen[:5],
        "onset_fields_present_per_entry": len(onset_missing) == 0 and len(per) == 20,
        "onset_missing_entry_ids": onset_missing,
        "onset_ok_flag": ledger.get("onset_fields_present_per_entry"),
        "onset_ok_asserts": (
            "summary/ledger onset_fields_present_per_entry == True means every "
            "ledger row has non-null uptime_s and available_mb (host state at "
            "entry start). Does not assert WSH-clean or gate-derived knee."
        ),
        "available_mb_min": min(avail) if avail else None,
        "available_mb_median": statistics.median(avail) if avail else None,
        "available_mb_max": max(avail) if avail else None,
        "uptime_s_min": min(uptimes) if uptimes else None,
        "uptime_s_max": max(uptimes) if uptimes else None,
        "uptime_hours_span": (
            (max(uptimes) - min(uptimes)) / 3600.0 if len(uptimes) >= 2 else None
        ),
        "fraction_turns_slo_ok": slo,
        "session_latency_sum_s": lat.get("sum"),
        "trajectory": traj,
        "integrity_pass": None,  # filled below
    }


def main() -> None:
    rows = [audit_cell(*c) for c in CELLS]
    for r in rows:
        r["integrity_pass"] = all(
            [
                r["entry_ids_exact_a621"],
                r["n_entries"] == 20,
                r["gold_20_20"],
                r["ir_ok"],
                r["wsh_clean_during_run"],
                r["onset_fields_present_per_entry"],
                r["watchdog_line_count"] > 1,  # start + at least one poll
                r["status"] == "complete",
            ]
        )
    out = {
        "kind": "x2_phase1_integrity",
        "ir_pin": IR_PIN,
        "cells": rows,
        "all_pass": all(r["integrity_pass"] for r in rows),
    }
    dest = BASE / "x2_phase1_integrity.json"
    dest.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))
    print("WROTE", dest)
    print("ALL_PASS", out["all_pass"])


if __name__ == "__main__":
    main()
