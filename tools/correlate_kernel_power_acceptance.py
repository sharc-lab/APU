#!/usr/bin/env python3
"""Re-analyse sealed fixed-throughput acceptance arms against System Kernel-Power.

Reads sealed ``raw/<run_id>/summary.json`` only (never edits raw/). Queries the live
System event log for Microsoft-Windows-Kernel-Power in the reconstructed windows,
correlates each repeat, and writes a derived JSON under ``derived/fixed_throughput/``.

Usage (from repo root)::

    python tools/correlate_kernel_power_acceptance.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from seam.tools.acceptance_instrumentation import (  # noqa: E402
    correlate_kernel_power_with_repeats,
    parse_utc,
)

# Two acceptance pairs sealed 2026-08-06. Full UUIDs.
PAIRS = [
    {
        "label": "acceptance_pair_1_pre_settle",
        "verdict_run_id": "9e3ca312-d6fa-4a8f-8215-3a8b270a049d",
        "arm_1_run_id": "b11553d7-3bbb-4162-ad3e-ba9a8b18987b",
        "arm_2_run_id": "21c1a2ab-af60-4196-84df-3a99f7ef0419",
        "orchestrator_run_id": None,
        "orchestrator_note": (
            "acc_orch_20260806_153848 launched; no sealed orchestrator summary on disk "
            "(result.json absent). Window reconstructed from arm summaries."
        ),
        "outlier": {"arm": "arm_1", "repeat": 0, "note": "sole arm_1 decode outlier; CV 0.145"},
    },
    {
        "label": "acceptance_pair_2_with_pre_run_settle",
        "verdict_run_id": "09dfe95d-a4b2-4b93-9757-10c4fead16b5",
        "arm_1_run_id": "483751fd-79fc-4be4-9bfc-5aec448b6249",
        "arm_2_run_id": "c3319520-3787-4904-8816-507bc2c599a4",
        "orchestrator_run_id": "68e1c0d9-4813-4058-8b65-a34f39728ff0",
        "orchestrator_note": (
            "Sealed orchestrator holds kernel_power.events (count=5) but not per-repeat flags."
        ),
        "outlier": {"arm": "arm_2", "repeat": 2, "note": "sole arm_2 decode outlier; CV 0.159"},
    },
]


def _load_summary(run_id: str) -> dict[str, Any]:
    path = ROOT / "raw" / run_id / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _repeat_rows(run_id: str, arm_label: str, summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pr in summary.get("per_repeat") or []:
        env = pr.get("envelope") or {}
        window = pr.get("window_utc") or {}
        start = window.get("start") or env.get("lock_acquired_utc") or pr.get("started_utc")
        end = window.get("end") or env.get("lock_released_utc") or pr.get("ended_utc")
        source = window.get("source")
        if not source:
            if env.get("lock_acquired_utc") and env.get("lock_released_utc"):
                source = "envelope.lock_acquired/released_utc"
            else:
                source = "missing"
        rows.append(
            {
                "arm": arm_label,
                "run_id": run_id,
                "repeat": pr.get("repeat"),
                "start_utc": start,
                "end_utc": end,
                "window_source": source,
                "r_decode_tok_s": pr.get("r_decode_tok_s"),
                "r_prefill_tok_s": pr.get("r_prefill_tok_s"),
                "wall_s": pr.get("wall_s"),
                "power_state_start": pr.get("power_state_start"),
                "power_state_end": pr.get("power_state_end"),
                "arm_power_start": summary.get("power_start"),
                "paging_gate_verdict": env.get("paging_gate_verdict"),
                "hard_page_reads_max_per_s": env.get("hard_page_reads_max_per_s"),
            }
        )
    return rows


def _query_kernel_power(start_utc: str, end_utc: str) -> dict[str, Any]:
    """Query System log; prefer live Get-WinEvent over sealed lists so pair 1 is covered."""
    ps = (
        f"$start = [datetimeoffset]::Parse('{start_utc}').UtcDateTime; "
        f"$end = [datetimeoffset]::Parse('{end_utc}').UtcDateTime; "
        "$events = Get-WinEvent -FilterHashtable @{"
        "LogName='System'; ProviderName='Microsoft-Windows-Kernel-Power'"
        "} -ErrorAction SilentlyContinue | "
        "Where-Object { $_.TimeCreated.ToUniversalTime() -ge $start "
        "-and $_.TimeCreated.ToUniversalTime() -le $end }; "
        "if (-not $events) { '[]'; exit 0 }; "
        "$events | Select-Object "
        "@{n='time_utc';e={$_.TimeCreated.ToUniversalTime().ToString('o')}}, "
        "Id, LevelDisplayName, Message, "
        "@{n='properties';e={@($_.Properties | ForEach-Object { $_.Value })}} "
        "| ConvertTo-Json -Depth 6 -Compress"
    )
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    raw = (completed.stdout or "").strip()
    if completed.returncode != 0:
        return {
            "start_utc": start_utc,
            "end_utc": end_utc,
            "events": [],
            "error": f"returncode={completed.returncode}; stderr={(completed.stderr or '')[-600:]}",
        }
    parsed = json.loads(raw) if raw else []
    if isinstance(parsed, dict):
        parsed = [parsed]
    events = []
    for row in parsed:
        message = row.get("Message") or ""
        props = row.get("properties")
        if props is not None and not isinstance(props, list):
            props = [props]
        events.append(
            {
                "time_utc": row.get("time_utc"),
                "id": row.get("Id"),
                "level": row.get("LevelDisplayName"),
                "message_head": message[:240],
                "payload": {"message": message, "properties": props},
            }
        )
    return {
        "start_utc": start_utc,
        "end_utc": end_utc,
        "events": events,
        "error": None,
        "source": "Get-WinEvent System/Microsoft-Windows-Kernel-Power",
    }


def _window_bounds(rows: list[dict[str, Any]]) -> tuple[str, str]:
    starts = [parse_utc(r["start_utc"]) for r in rows if r.get("start_utc")]
    ends = [parse_utc(r["end_utc"]) for r in rows if r.get("end_utc")]
    starts_n = [t for t in starts if t is not None]
    ends_n = [t for t in ends if t is not None]
    if not starts_n or not ends_n:
        raise SystemExit("could not reconstruct acceptance window from sealed repeats")
    return min(starts_n).isoformat(), max(ends_n).isoformat()


def _mark_outlier(rows: list[dict[str, Any]], outlier: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        marked = dict(row)
        marked["is_outlier"] = (
            row.get("arm") == outlier["arm"] and row.get("repeat") == outlier["repeat"]
        )
        out.append(marked)
    return out


def analyse_pair(pair: dict[str, Any]) -> dict[str, Any]:
    arm1 = _load_summary(pair["arm_1_run_id"])
    arm2 = _load_summary(pair["arm_2_run_id"])
    verdict = _load_summary(pair["verdict_run_id"])
    rows = _repeat_rows(pair["arm_1_run_id"], "arm_1", arm1) + _repeat_rows(
        pair["arm_2_run_id"], "arm_2", arm2
    )
    # Pad the query a few seconds beyond lock windows so gap-edge events are visible.
    start, end = _window_bounds(rows)
    # Also include arm-level uptime observation so pre-arm events near start are caught.
    arm_starts = [
        parse_utc((arm1.get("uptime_wake") or {}).get("observed_utc")),
        parse_utc((arm2.get("uptime_wake") or {}).get("observed_utc")),
    ]
    arm_starts_n = [t for t in arm_starts if t is not None]
    start_dt = parse_utc(start)
    end_dt = parse_utc(end)
    assert start_dt is not None and end_dt is not None
    if arm_starts_n:
        start_dt = min([start_dt, *arm_starts_n])
    query = _query_kernel_power(start_dt.isoformat(), end_dt.isoformat())
    correlated = correlate_kernel_power_with_repeats(
        events=query["events"], repeats=_mark_outlier(rows, pair["outlier"])
    )

    outlier_rows = [r for r in correlated if r.get("is_outlier")]
    clean_rows = [r for r in correlated if not r.get("is_outlier")]
    outlier_has = any(r.get("kernel_power_event_in_window") for r in outlier_rows)
    clean_has = any(r.get("kernel_power_event_in_window") for r in clean_rows)
    if outlier_has and not clean_has:
        coincidence = "outlier_only"
        coincidence_note = (
            "Outlier repeat window contains Kernel-Power event(s); clean repeats do not. "
            "Cause named for this pair."
        )
    elif outlier_has and clean_has:
        coincidence = "outlier_and_clean"
        coincidence_note = "Kernel-Power present in outlier and at least one clean repeat."
    elif not outlier_has and clean_has:
        coincidence = "clean_only"
        coincidence_note = "Kernel-Power in clean repeat(s) only — not explanatory for the outlier."
    else:
        coincidence = "none"
        coincidence_note = (
            "No Kernel-Power event inside any reconstructed repeat window for this pair."
        )

    mem = (verdict.get("memory_dispersion") or {}) if "memory_dispersion" in verdict else {}
    # Verdict summaries nest memory under instrument path differently; pull from orch if needed.
    if not mem and pair.get("orchestrator_run_id"):
        orch = _load_summary(pair["orchestrator_run_id"])
        mem = orch.get("memory_dispersion") or {}

    return {
        "label": pair["label"],
        "verdict_run_id": pair["verdict_run_id"],
        "arm_1_run_id": pair["arm_1_run_id"],
        "arm_2_run_id": pair["arm_2_run_id"],
        "orchestrator_run_id": pair["orchestrator_run_id"],
        "orchestrator_note": pair["orchestrator_note"],
        "outlier": pair["outlier"],
        "window_reconstruction": {
            "method": (
                "per_repeat envelope.lock_acquired_utc / lock_released_utc from sealed "
                "summary.json; query start also min'd with arm uptime_wake.observed_utc"
            ),
            "uncertainty": (
                "Lock window includes quiescence/canary/child lifetime, not generation-only. "
                "Generation wall_s is nested inside the lock window. Exact harness "
                "started_utc/ended_utc per repeat was not sealed on these runs."
            ),
            "query_start_utc": query["start_utc"],
            "query_end_utc": query["end_utc"],
        },
        "kernel_power": query,
        "repeat_kernel_power": correlated,
        "coincidence_verdict": coincidence,
        "coincidence_note": coincidence_note,
        "arm_power_state_recoverable": {
            "arm_1_power_start": arm1.get("power_start"),
            "arm_2_power_start": arm2.get("power_start"),
            "per_repeat_power_state": "unrecoverable_from_sealed_artifacts",
            "note": (
                "Sealed arms record power_start once per arm (end of arm capture). "
                "Per-repeat AC/charging was not instrumented at seal time; future runs "
                "record power_state_start/end per repeat."
            ),
        },
        "instrument_gate": verdict.get("instrument_gate"),
        "memory_dispersion": mem or verdict.get("memory_dispersion"),
        "within_run_spread": {
            "arm_1_cv_decode": (arm1.get("r_decode_tok_s") or {}).get("cv"),
            "arm_2_cv_decode": (arm2.get("r_decode_tok_s") or {}).get("cv"),
        },
    }


def main() -> int:
    analysed_at = datetime.now(UTC).isoformat()
    pairs = [analyse_pair(p) for p in PAIRS]

    # Cross-pair notes required by the dispatch.
    pair1 = pairs[0]
    pair2 = pairs[1]
    free_peak = None
    if pair2.get("memory_dispersion"):
        free_peak = (pair2["memory_dispersion"] or {}).get("free_physical_bytes_at_peak")
    if free_peak is None and pair2.get("orchestrator_run_id"):
        orch = _load_summary(pair2["orchestrator_run_id"])
        free_peak = (orch.get("memory_dispersion") or {}).get("free_physical_bytes_at_peak")
        ws_peak = (orch.get("memory_dispersion") or {}).get("peak_working_set_bytes")
    else:
        ws_peak = (pair2.get("memory_dispersion") or {}).get("peak_working_set_bytes")

    report = {
        "analysis_id": "kernel_power_repeat_correlation_20260806",
        "analysed_utc": analysed_at,
        "purpose": (
            "Decisive test: do Kernel-Power events coincide with outlier acceptance repeats "
            "and spare clean repeats? Report only — no re-measurement, no policy/band change."
        ),
        "pairs": pairs,
        "decisive_table": [
            {
                "verdict_run_id": p["verdict_run_id"],
                "outlier": p["outlier"],
                "coincidence_verdict": p["coincidence_verdict"],
                "coincidence_note": p["coincidence_note"],
                "outlier_events": [
                    e
                    for r in p["repeat_kernel_power"]
                    if r.get("is_outlier")
                    for e in (r.get("kernel_power_events_in_window") or [])
                ],
            }
            for p in pairs
        ],
        "pre_run_settle_status": {
            "status": "disconfirmed_as_complete_explanation",
            "kept": True,
            "evidence": (
                "With pre_run_settle (pair 2 / 09dfe95d), arm_1 CV improved 0.145→0.037, but "
                "arm_2 repeat 2 became the sole outlier (CV 0.159) during Modern Standby. "
                "Prior pair (9e3ca312) outlier was arm_1 repeat 0 with no Kernel-Power in-window. "
                "Dip is sporadic, not positional — settle is not a complete explanation."
            ),
            "arm1_cv_before": pair1["within_run_spread"]["arm_1_cv_decode"],
            "arm1_cv_after": pair2["within_run_spread"]["arm_1_cv_decode"],
            "citing_verdicts": [
                pair1["verdict_run_id"],
                pair2["verdict_run_id"],
            ],
        },
        "free_physical_headroom_risk": {
            "citing_orchestrator_run_id": pair2.get("orchestrator_run_id"),
            "citing_verdict_run_id": pair2["verdict_run_id"],
            "free_physical_bytes_at_peak": free_peak,
            "peak_working_set_bytes": ws_peak,
            "note": (
                "free_physical_bytes_at_peak differed ~1.47× between arms "
                "(2.688 GB vs 1.824 GB) while peak_working_set agreed to ~1.0002×. "
                "Headroom at peak is far less reproducible than consumption — direct risk "
                "to ceiling determination; understand before ceiling(A), not after."
            ),
        },
        "charging_state_recovered": {
            "pair_1": {
                "arm_1": pair1["arm_power_state_recoverable"]["arm_1_power_start"],
                "arm_2": pair1["arm_power_state_recoverable"]["arm_2_power_start"],
            },
            "pair_2": {
                "arm_1": pair2["arm_power_state_recoverable"]["arm_1_power_start"],
                "arm_2": pair2["arm_power_state_recoverable"]["arm_2_power_start"],
            },
            "cross_pair_note": (
                "Pair 1 arm_2: 99% charging=True; pair 2 arm_1: 100% charging=False. "
                "A full pack on AC cycles its charge controller. Per-repeat charging within "
                "an arm is unrecoverable from these sealed artifacts."
            ),
            "kernel_power_105_in_broader_day": (
                "Id=105 Power source change events at 20:12:55 (False) and 20:14:26 (True) "
                "UTC sit after pair 1 arm_2 ended (~20:05:44) and before pair 2 started "
                "(~20:45) — outside both measurement windows; recorded for context only."
            ),
        },
        "raw_untouched": True,
    }

    out_dir = ROOT / "derived" / "fixed_throughput"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "kernel_power_repeat_correlation_20260806.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md_path = out_dir / "kernel_power_repeat_correlation_20260806.md"
    lines = [
        "# Kernel-Power × acceptance repeat correlation (2026-08-06)",
        "",
        f"Analysed UTC: `{analysed_at}`",
        "",
        "Report only. No re-measurement. Power policy / band / basis / repeats unchanged.",
        "",
        "## Decisive table",
        "",
        "| Verdict | Outlier | Coincidence | Note |",
        "|:--|:--|:--|:--|",
    ]
    for row in report["decisive_table"]:
        o = row["outlier"]
        lines.append(
            f"| `{row['verdict_run_id'][:8]}…` | {o['arm']} r{o['repeat']} | "
            f"**{row['coincidence_verdict']}** | {row['coincidence_note']} |"
        )
    lines.extend(
        [
            "",
            "## pre_run_settle",
            "",
            report["pre_run_settle_status"]["evidence"],
            "",
            "## free_physical headroom risk",
            "",
            report["free_physical_headroom_risk"]["note"],
            "",
            f"Artifact JSON: `{out_path.relative_to(ROOT).as_posix()}`",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {"json": str(out_path), "md": str(md_path), "decisive_table": report["decisive_table"]},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
