"""X-2 feasibility-table worker — clean protocol, one arm per invocation.

Detached payload for tools/launch_x2.ps1. No measurement when imported.

Re-measures the README_characterizations Axis 1×2 feasibility table
(cpu-p | gpu_only) × (RESIDENT | NON_RESIDENT) under the five-gate clean
protocol, with WorkloadsSessionHost watchdog and per-entry host onset fields.

CRITICAL: each entry records uptime_s + available_mb at generation start so
within-arm latency vs uptime (onset) is measurable. cpu-p NON_RESIDENT alone
spans ~9630 s (~2.7 h); if TTFT trends up with uptime_s inside that arm, the
chosen 2 h gate is replaced by a measured knee.

Steps:
  1. Map CLI arm (cpu-p → A, gpu_only → gpu_only); start WSH watchdog.
  2. Select fixed multi_turn_base prefix; ASSERT ids == a621ff7d's 20.
  3. Gold selftest; refuse unless n/n BEFORE generation.
  4. run_session_residency (greedy GenerationConfig, same as a621ff7d).
  5. Write ledger with per-entry uptime_s / available_mb / TTFT.
"""

from __future__ import annotations

import argparse
import atexit
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

A621_SESSION = (
    ROOT
    / "derived"
    / "bfcl_feasibility"
    / "session_residency"
    / "a621ff7d-2919-463d-aaf6-673f9e6bafbc"
)
A621_ENTRIES = A621_SESSION / "session_residency_entries_gpu_only_RESIDENT.json"
DEFAULT_MODEL_SPEC = ROOT / "configs" / "models" / "Qwen3-4B-int4-ov.yaml"

# Launcher CLI arm → delta_n.yaml arm id (cpu-p is label; probe id is A).
ARM_CLI_TO_PROBE: dict[str, str] = {
    "cpu-p": "A",
    "gpu_only": "gpu_only",
}

WATCHDOG_SCRIPT_BODY = r"""param([string]$LogPath, [int]$IntervalS)
$ErrorActionPreference = "Continue"
if ($IntervalS -lt 30) { $IntervalS = 30 }
while ($true) {
    Start-Sleep -Seconds $IntervalS
    $utc = (Get-Date).ToUniversalTime().ToString("o")
    $procs = @(Get-Process -Name "WorkloadsSessionHost" -ErrorAction SilentlyContinue)
    if ($procs.Count -eq 0) {
        $rec = [ordered]@{ utc = $utc; event = "watchdog_poll"; n_found = 0; n_killed = 0 }
        Add-Content -LiteralPath $LogPath -Value (ConvertTo-Json -InputObject $rec -Compress)
        continue
    }
    $pids = @($procs | ForEach-Object { [int]$_.Id })
    $procs | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
    $left = @(Get-Process -Name "WorkloadsSessionHost" -ErrorAction SilentlyContinue)
    $rec = [ordered]@{
        utc            = $utc
        event          = "watchdog_kill"
        n_found        = $pids.Count
        pids_found     = $pids
        n_killed       = $pids.Count - $left.Count
        pids_remaining = @($left | ForEach-Object { [int]$_.Id })
    }
    Add-Content -LiteralPath $LogPath -Value (ConvertTo-Json -InputObject $rec -Compress -Depth 4)
}
"""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _load_ids(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, list):
        raise SystemExit(f"REFUSED -- expected entry list in {path}")
    ids: list[str] = []
    for row in raw:
        if not isinstance(row, dict) or "id" not in row:
            raise SystemExit(f"REFUSED -- entry missing id in {path}")
        ids.append(str(row["id"]))
    return ids


def _ir_sha256(spec_path: Path) -> str:
    for line in spec_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("ir_sha256:"):
            return s.split(":", 1)[1].strip().strip('"').strip("'").lower()
    raise SystemExit(f"REFUSED -- ir_sha256 missing from {spec_path}")


def _start_wsh_watchdog(session_dir: Path, interval_s: int) -> dict[str, Any] | None:
    """Sibling powershell.exe re-kill loop for the life of this worker.

    Diff vs broken 91905556 spawn:
      - Used subprocess.Popen(..., creationflags=DETACHED_PROCESS|CREATE_NEW_PROCESS_GROUP).
      - DETACHED_PROCESS makes powershell.exe exit within ~1 s → only the Python-side
        watchdog_start_kill line, zero poll/kill events. Confirmed in spawn A/B:
        flags 0x8|0x200 dead; flags 0 or CREATE_NEW_PROCESS_GROUP alone alive + polls.
      - Matrix Start-WshWatchdog uses Start-Process -WindowStyle Hidden *inside the
        long-lived orchestrator*. Spawning Start-Process from a short-lived helper
        that then exits can still lose the child under a kill-on-job-close job.
      - Fix: Popen without DETACHED_PROCESS; keep the handle alive until atexit.
    """
    if interval_s <= 0:
        return None
    session_dir.mkdir(parents=True, exist_ok=True)
    kill_log = session_dir / "watchdog_kills.jsonl"
    watch_script = session_dir / "_wsh_watchdog.ps1"
    watch_script.write_text(WATCHDOG_SCRIPT_BODY, encoding="utf-8")

    t0 = _utc_now()
    init: dict[str, Any] = {
        "utc": t0,
        "event": "watchdog_start_kill",
        "interval_s": interval_s,
        "pids_found": [],
        "n_killed": 0,
    }
    try:
        listed = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                (
                    "$p=@(Get-Process -Name WorkloadsSessionHost -EA SilentlyContinue); "
                    "$ids=@($p | ForEach-Object { [int]$_.Id }); "
                    "if($p.Count -gt 0){$p|Stop-Process -Force -EA SilentlyContinue; "
                    "Start-Sleep -Seconds 1}; "
                    "$left=@(Get-Process -Name WorkloadsSessionHost -EA SilentlyContinue); "
                    "Write-Output (('ids=' + ($ids -join ',')) + '; left=' + $left.Count)"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        init["start_kill_stdout"] = (listed.stdout or "").strip()
    except Exception as exc:
        init["start_kill_error"] = f"{type(exc).__name__}: {exc}"

    with kill_log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(init, sort_keys=True) + "\n")

    # CREATE_NEW_PROCESS_GROUP only (0x200). Never DETACHED_PROCESS (0x8).
    creationflags = 0x00000200 if sys.platform == "win32" else 0
    proc = subprocess.Popen(
        [
            "powershell.exe",
            "-NoProfile",
            "-NoLogo",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(watch_script),
            "-LogPath",
            str(kill_log),
            "-IntervalS",
            str(interval_s),
        ],
        cwd=str(ROOT),
        creationflags=creationflags,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1.0)
    if proc.poll() is not None:
        raise SystemExit(
            f"REFUSED -- WSH watchdog exited immediately (pid={proc.pid} "
            f"code={proc.returncode}). DETACHED_PROCESS must not be used; "
            "see docs/ENV_CHANGELOG.md 2026-09-01."
        )
    handle: dict[str, Any] = {
        "kind": "process",
        "pid": proc.pid,
        "script": str(watch_script),
        "interval_s": interval_s,
        "kill_log": str(kill_log),
        "mechanism": "Popen_CREATE_NEW_PROCESS_GROUP",
        "spawn_fix": (
            "subprocess.Popen without DETACHED_PROCESS; handle kept alive for "
            "worker lifetime (matrix Start-Process is equivalent only when the "
            "parent stays alive). Broken 91905556 used DETACHED_PROCESS."
        ),
        "_proc": proc,  # keep referenced so GC cannot collect mid-run
    }

    def _stop() -> None:
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except Exception:
                    proc.kill()
        except Exception:
            pass

    atexit.register(_stop)
    # Drop non-JSON-serializable handle before plan.json write sites copy it.
    public = {k: v for k, v in handle.items() if k != "_proc"}
    # Retain proc on the function attribute so atexit + GC stay correct.
    _start_wsh_watchdog._alive_proc = proc  # type: ignore[attr-defined]
    return public


def _entry_ledger_row(
    *,
    entry_result: dict[str, Any],
    model_spec: Path,
    ir_sha256: str,
) -> dict[str, Any]:
    score = entry_result.get("score") or {}
    turns: list[dict[str, Any]] = []
    for tm in entry_result.get("turn_metrics") or []:
        acc = tm.get("per_turn_accuracy") or {}
        turns.append(
            {
                "turn": tm.get("turn"),
                "pass": bool(acc.get("correct")) if acc else None,
                "per_turn_accuracy": acc,
                "wall_s_last_generation": tm.get("last_wall_s"),
                "n_generations": tm.get("n_generations"),
                "ttft_s": tm.get("ttft_s"),
                "decode_tok_s": tm.get("decode_tok_s"),
                "slo_ok": tm.get("slo_ok"),
            }
        )
    force_quit = bool(entry_result.get("force_quit"))
    error_type = score.get("error_type")
    return {
        "id": entry_result.get("id"),
        "model_spec": str(model_spec),
        "ir_sha256": ir_sha256,
        # Per-entry host onset (X-2 CRITICAL) — not run-level aggregates.
        "uptime_s": entry_result.get("uptime_s"),
        "available_mb": entry_result.get("available_mb"),
        "available_method": entry_result.get("available_method"),
        "trajectory_pass": score.get("valid") is True,
        "failure_bucket": error_type,
        "force_quit": force_quit,
        "force_terminated": error_type == "multi_turn:force_terminated" or force_quit,
        "generation_timeout_hit": False,
        "wall_s": entry_result.get("wall_s"),
        "n_completed_turns": entry_result.get("n_completed_turns"),
        "n_user_turns": entry_result.get("n_user_turns"),
        "per_turn": turns,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--model-spec",
        type=Path,
        default=DEFAULT_MODEL_SPEC,
        help="FetchedModelSpec YAML (feasibility table: Qwen3-4B-int4-ov.yaml)",
    )
    parser.add_argument(
        "--arm",
        required=True,
        choices=sorted(ARM_CLI_TO_PROBE.keys()),
        help="Launcher arm label: cpu-p (probe A) or gpu_only",
    )
    parser.add_argument(
        "--residency-mode",
        required=True,
        choices=("RESIDENT", "NON_RESIDENT"),
    )
    parser.add_argument("--n-entries", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--a621-entries",
        type=Path,
        default=A621_ENTRIES,
        help="Reference entry list whose ids must match exactly (a621's 20)",
    )
    parser.add_argument(
        "--watchdog-interval-s",
        type=int,
        default=60,
        help=(
            "WorkloadsSessionHost re-kill interval (0=off). Default 60. "
            "Measured respawn ~4 min; 300 s loses the race even when the "
            "sibling stays alive (see docs/ENV_CHANGELOG.md)."
        ),
    )
    args = parser.parse_args(argv)

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    model_spec = args.model_spec if args.model_spec.is_absolute() else ROOT / args.model_spec
    if not model_spec.is_file():
        print(f"REFUSED -- model spec not found: {model_spec}", flush=True)
        return 2
    if not args.a621_entries.is_file():
        print(f"REFUSED -- a621 reference entries missing: {args.a621_entries}", flush=True)
        return 2
    if int(args.n_entries) != 20:
        # Comparability with a621's 20 and the feasibility table; refuse silent scale-up.
        print(
            f"REFUSED -- X-2 n_entries must be 20 (got {args.n_entries}); "
            "comparability with a621ff7d / feasibility table outweighs power; "
            "cpu-p at 200 would run ~27 h",
            flush=True,
        )
        return 2

    arm_cli = str(args.arm)
    arm_id = ARM_CLI_TO_PROBE[arm_cli]
    residency_mode = str(args.residency_mode).upper()

    import tools.bfcl_feasibility_probe as probe

    probe.apply_model_spec(model_spec)
    ir_sha = _ir_sha256(model_spec)

    watch = _start_wsh_watchdog(out_dir, int(args.watchdog_interval_s))
    if watch:
        print(
            f"WSH watchdog started pid={watch['pid']} interval_s={watch['interval_s']} "
            f"log={watch['kill_log']}",
            flush=True,
        )

    generation_config = {
        "source": "tools/bfcl_feasibility_probe.py generate_turn GenerationConfig",
        "mode": "greedy",
        "do_sample": False,
        "max_new_tokens": int(args.max_new_tokens),
        "apply_chat_template": False,
        "sampler_active": False,
        "pairing_seed": int(args.seed),
        "identical_to_a621ff7d": True,
        "identical_note": (
            "Same generate_turn GenerationConfig as a621ff7d run_session_residency "
            "(do_sample=False, max_new_tokens=512, apply_chat_template=False)."
        ),
    }

    plan: dict[str, Any] = {
        "kind": "x2_feasibility_table",
        "session_id": args.session_id,
        "started_utc": _utc_now(),
        "arm_cli": arm_cli,
        "arm_id": arm_id,
        "arm_map_note": "cpu-p → delta_n.yaml id A; gpu_only → gpu_only",
        "residency_mode": residency_mode,
        "n_entries": args.n_entries,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "model_spec": str(model_spec),
        "ir_sha256": ir_sha,
        "generation_config": generation_config,
        "a621_reference_session": "a621ff7d-2919-463d-aaf6-673f9e6bafbc",
        "a621_entries_path": str(args.a621_entries),
        "feasibility_table": {
            "source": "docs/README_characterizations.md Axis 1×2",
            "cells": [
                "cpu-p × NON_RESIDENT",
                "cpu-p × RESIDENT",
                "gpu_only × NON_RESIDENT",
                "gpu_only × RESIDENT",
            ],
            "this_cell": f"{arm_cli} × {residency_mode}",
            "one_arm_per_invocation": True,
            "n_entries_rationale": (
                "20 not 200: original table is n=20; comparability > power; " "cpu-p at 200 ≈ 27 h."
            ),
        },
        "onset": {
            "per_entry_fields": ["uptime_s", "available_mb", "available_method"],
            "purpose": (
                "Within-arm TTFT vs uptime_s. If upward trend inside one long arm "
                "(esp. cpu-p NON_RESIDENT ~9630 s), replace chosen 2 h gate with "
                "measured knee."
            ),
            "chosen_2h_gate": "PROVISIONAL — this arm is the derivation source",
        },
        "watchdog": {
            "enabled": watch is not None,
            "process": "WorkloadsSessionHost",
            "interval_s": int(args.watchdog_interval_s),
            "handle": watch,
            "note": (
                "Sibling powershell via Start-Process -WindowStyle Hidden "
                "(matrix Start-WshWatchdog). Not Start-Job. Not DETACHED_PROCESS. "
                "Measured WSH respawn ~4 min (X-2 cell 1); interval must be << that."
            ),
        },
        "status": "running",
    }
    (out_dir / "plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # --- entry pin: exact equality to a621's 20 ---
    entries = probe.select_multi_turn_entries()[: args.n_entries]
    pin_path = out_dir / probe.PINNED_MULTI_TURN_ENTRIES
    pin_path.write_text(json.dumps(entries, indent=2, default=str) + "\n", encoding="utf-8")
    got_ids = [str(e["id"]) for e in entries]
    ref_ids = _load_ids(args.a621_entries)
    plan["entry_assert"] = {
        "mode": "exact",
        "n_reference": len(ref_ids),
        "n_run": len(got_ids),
    }
    if got_ids != ref_ids:
        plan["status"] = "refused_entry_id_mismatch"
        plan["entry_ids_got"] = got_ids
        plan["entry_ids_a621"] = ref_ids
        (out_dir / "plan.json").write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            "REFUSED -- multi_turn_probe_entries.json ids != a621ff7d's 20 "
            f"(mode=exact n_got={len(got_ids)} n_ref={len(ref_ids)})",
            flush=True,
        )
        for i, (a, b) in enumerate(zip(got_ids, ref_ids, strict=False)):
            if a != b:
                print(f"  first mismatch at [{i}]: got={a} a621={b}", flush=True)
                break
        if len(got_ids) != len(ref_ids):
            print(f"  length differs got={len(got_ids)} a621={len(ref_ids)}", flush=True)
        return 3
    (out_dir / "plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"entry_ids ASSERT ok: exact match a621ff7d n={len(got_ids)}", flush=True)

    gold = probe.run_multi_turn_gold_selftest(entries)
    gold_path = out_dir / "multi_turn_gold_selftest.json"
    gold_path.write_text(
        json.dumps(gold, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    n_gold = int(gold.get("n") or 0)
    n_valid = int(gold.get("n_valid") or 0)
    plan["gold_selftest"] = {"n": n_gold, "n_valid": n_valid, "path": str(gold_path)}
    if n_gold != len(entries) or n_valid != n_gold or n_gold < 1:
        plan["status"] = "refused_gold_selftest"
        plan["ended_utc"] = _utc_now()
        (out_dir / "plan.json").write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            f"REFUSED -- gold selftest {n_valid}/{n_gold} (need {len(entries)}/{len(entries)})",
            flush=True,
        )
        return 3
    print(f"gold selftest PASS: {n_valid}/{n_gold}", flush=True)

    print(
        f"running session_residency arm_cli={arm_cli} arm_id={arm_id} "
        f"mode={residency_mode} model_spec={model_spec}",
        flush=True,
    )
    report = probe.run_session_residency(
        out_dir,
        arm_id=arm_id,
        residency_mode=residency_mode,
        n_entries=args.n_entries,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    per_entry_raw = (report.get("gpu_probe") or {}).get("per_entry") or []
    ledger = [
        _entry_ledger_row(entry_result=row, model_spec=model_spec, ir_sha256=ir_sha)
        for row in per_entry_raw
    ]
    # Sanity: onset fields must be present per entry (not only run-level).
    missing_onset = [
        row.get("id")
        for row in ledger
        if row.get("uptime_s") is None or row.get("available_mb") is None
    ]
    traj = (report.get("gpu_probe") or {}).get("accuracy_trajectory") or {}
    per_turn = (report.get("gpu_probe") or {}).get("accuracy_per_turn_f2") or {}
    slo_frac = (report.get("gpu_probe") or {}).get("fraction_turns_slo_ok")
    session_lat = (report.get("gpu_probe") or {}).get("session_total_latency_s") or {}

    summary = {
        "kind": "x2_feasibility_table",
        "session_id": args.session_id,
        "model_spec": str(model_spec),
        "ir_sha256": ir_sha,
        "arm_cli": arm_cli,
        "arm_id": arm_id,
        "residency_mode": residency_mode,
        "entry_ids_match_a621": True,
        "entry_assert": plan.get("entry_assert"),
        "gold_selftest_n_valid": n_valid,
        "gold_selftest_n": n_gold,
        "accuracy_trajectory": traj,
        "accuracy_per_turn_f2": per_turn,
        "fraction_turns_slo_ok": slo_frac,
        "session_total_latency_s": session_lat,
        "onset_fields_present_per_entry": len(missing_onset) == 0,
        "onset_missing_entry_ids": missing_onset,
        "per_entry": ledger,
        "report_artifact": report.get("artifact"),
        "watchdog": plan.get("watchdog"),
        "ended_utc": _utc_now(),
        "status": "complete",
    }
    (out_dir / "x2_entry_ledger.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    plan["status"] = "complete"
    plan["ended_utc"] = summary["ended_utc"]
    plan["accuracy_trajectory"] = traj
    plan["fraction_turns_slo_ok"] = slo_frac
    plan["session_total_latency_s"] = session_lat
    plan["onset_fields_present_per_entry"] = summary["onset_fields_present_per_entry"]
    plan["ledger"] = str(out_dir / "x2_entry_ledger.json")
    (out_dir / "plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "ok": True,
                "session_id": args.session_id,
                "arm_cli": arm_cli,
                "arm_id": arm_id,
                "residency_mode": residency_mode,
                "trajectory": traj,
                "fraction_turns_slo_ok": slo_frac,
                "session_latency_sum_s": session_lat.get("sum"),
                "onset_ok": summary["onset_fields_present_per_entry"],
                "out": str(out_dir),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
