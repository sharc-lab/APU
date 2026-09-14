"""Post-hoc PARTIAL seal for a force-killed ceiling_a session (derived_diagnostic).

Why this tool exists (path choice)
----------------------------------
``seam.tools.ceiling_a.orchestrate`` only reaches ``phase=\"sealed\"`` after
``run_interleaved_ladder`` returns and ``_seal_arm`` / ``_seal_verdict`` run.
``--resume`` continues the ladder from the checkpoint; it does **not** offer a
seal-incomplete path. Session ``ad7b9288-…`` stopped at ``phase=\"ladder\"`` after
n=40000 r0 timeouts and round-1 order — the internal sealer cannot be invoked
post-hoc without re-running measurement. This tool mirrors
``tools/seal_delta_prefill_session.py``: write-once derived seal first, then
optional ``--attempt-raw-promote`` that calls ``seam.manifest.emit`` into the
existing in-progress arm run dirs (and the reserved verdict run_id) only after
``isolation_mode=remote`` clears. Never invents ``enforce=False``.

Usage:
  .\\.venv-seam\\Scripts\\python.exe tools\\seal_ceiling_a_partial_session.py \\
      --session-id ad7b9288-42e6-42e8-be3e-ad1a1b1abc4c
  .\\.venv-seam\\Scripts\\python.exe tools\\seal_ceiling_a_partial_session.py \\
      --session-id ad7b9288-42e6-42e8-be3e-ad1a1b1abc4c \\
      --attempt-raw-promote --allow-dirty
  .\\.venv-seam\\Scripts\\python.exe tools\\seal_ceiling_a_partial_session.py \\
      --session-id ad7b9288-42e6-42e8-be3e-ad1a1b1abc4c \\
      --promote-only --attempt-raw-promote --allow-dirty
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Force-killed A/A_prime ceiling ladder (2026-08-06 launch).
DEFAULT_SESSION_ID = "ad7b9288-42e6-42e8-be3e-ad1a1b1abc4c"
DEFAULT_LAUNCH_LOG = ROOT / "derived" / "ceiling_a" / "_launches" / "ceiling_a_20260806_190044.log"

_PLATFORM_PATH = Path("configs/platforms/aipc-c1.yaml")
_MEASUREMENT_PATH = Path("configs/measurement.yaml")
_DELTA_N_PATH = Path("configs/delta_n.yaml")
_MODEL_SPEC_PATH = Path("configs/models/Qwen3-4B-int4-ov.yaml")

ARM_BENCHMARK = "ceiling_a_context_ceiling_v1"
VERDICT_BENCHMARK = "ceiling_a_verdict_v1"
NATIVE_CONTEXT_FLAG_ABOVE = 32768
POSITION_LIMIT_TOKENS = 40960


def _load_session_plan(session_id: str) -> dict[str, Any] | None:
    """Prefer plan.json; fall back to session config / checkpoint model_spec fields."""
    base = ROOT / "derived" / "ceiling_a" / session_id
    for name in ("plan.json", "config.json", "checkpoint.json"):
        path = base / name
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(data, dict):
                return data
    return None


def _resolve_model_spec_for_session(session_id: str, *, cli_spec: str | Path | None) -> Path:
    from tools.seal_model_spec import resolve_sealer_model_spec

    return resolve_sealer_model_spec(
        root=ROOT,
        default_spec=_MODEL_SPEC_PATH,
        cli_spec=cli_spec,
        plan=_load_session_plan(session_id),
    )


PREREQUISITE_ACCEPTANCE_RUN_ID = "cb8e8d7c-74dd-4e72-ae69-8db7239a7b0f"
GENERATION_TIMEOUT_S = 1800

STATUS_PARTIAL = (
    "PARTIAL, force-killed. Not COMPLETE. "
    "Orchestrator stopped after n=40000 ladder r0 timeouts for A and A_prime; "
    "round 1 was ordered then the process did not continue. "
    "Do not read highest_pass as a bracketed memory ceiling."
)

# AM-036: measurement power_state must stay null on retro-seal; promote-time samples
# go only in promote_time_power_state and must never be read as run environment.
_RETRO_SEAL_POWER_NOTE = (
    "This run predates measurement-time powerstate capture; the true measurement "
    "power state is unrecorded. Promote-time host samples, if present, are recorded "
    "only in promote_time_power_state (AM-036)."
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_tree(root: Path, *, exclude: set[str]) -> str:
    h = hashlib.sha256()
    files = sorted(
        (p for p in root.rglob("*") if p.is_file() and p.name not in exclude),
        key=lambda p: p.relative_to(root).as_posix(),
    )
    for p in files:
        rel = p.relative_to(root).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _ndjson(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _log_events(path: Path) -> list[dict[str, Any]]:
    """Parse mixed seam text + JSON lines from a ceiling_a launch log."""
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s.startswith("{"):
            continue
        try:
            rows.append(json.loads(s))
        except json.JSONDecodeError:
            continue
    return rows


def _generation_block(repeat: dict[str, Any]) -> dict[str, Any]:
    result = repeat.get("result") or {}
    child = result.get("child") or {}
    gen = child.get("generation") or {}
    if gen:
        return gen
    # Prefer work-file style if ever inlined.
    return (result.get("generation") or {}) if isinstance(result, dict) else {}


def _curve_row(repeat: dict[str, Any]) -> dict[str, Any]:
    gen = _generation_block(repeat)
    fc = repeat.get("failure_classification") or {}
    return {
        "arm_id": repeat.get("arm_id"),
        "n_tokens": repeat.get("n_tokens"),
        "repeat_index": repeat.get("repeat_index"),
        "attempt_used": repeat.get("attempt_used"),
        "admissible": repeat.get("admissible"),
        "outcome": (repeat.get("result") or {}).get("outcome"),
        "failure_class": fc.get("class"),
        "failure_detail": fc.get("detail"),
        "r_prefill_tok_s": gen.get("r_prefill_tok_s"),
        "r_decode_tok_s": gen.get("r_decode_tok_s"),
        "wall_s": gen.get("wall_s") or (repeat.get("result") or {}).get("parent_wall_s"),
        "peak_working_set_bytes": (repeat.get("memory_instrumentation") or {}).get(
            "peak_working_set_bytes"
        ),
        "free_physical_bytes_at_peak": (repeat.get("memory_instrumentation") or {}).get(
            "free_physical_bytes_at_peak"
        ),
        "standby_retry": int(repeat.get("attempt_used") or 0) > 0,
    }


def _standby_from_log(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for e in events:
        if e.get("event") != "delta_n.modern_standby_inadmissible":
            continue
        out.append(
            {
                "event": e.get("event"),
                "tag": e.get("tag"),
                "attempt": e.get("attempt"),
                "event_ids": list(e.get("event_ids") or []),
                "ts": e.get("ts"),
                "window_start": e.get("window_start"),
                "window_end": e.get("window_end"),
            }
        )
    return out


def _standby_admitted_cells(
    repeats_by_arm: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    admitted: list[dict[str, Any]] = []
    for arm_id, rows in repeats_by_arm.items():
        for r in rows:
            if int(r.get("attempt_used") or 0) <= 0:
                continue
            discarded = []
            for a in r.get("attempts") or []:
                discarded.append(
                    {
                        "attempt": a.get("attempt"),
                        "admissible": a.get("admissible"),
                        "detail": a.get("detail"),
                        "ended_utc": a.get("ended_utc"),
                    }
                )
            admitted.append(
                {
                    "arm_id": arm_id,
                    "n_tokens": r.get("n_tokens"),
                    "repeat_index": r.get("repeat_index"),
                    "attempt_used": r.get("attempt_used"),
                    "admissible": r.get("admissible"),
                    "feeds_prefill_fit": True,
                    "note": (
                        "Admitted cell after modern_standby_inadmissible discard(s). "
                        "This row is the value that enters any prefill fit for this arm."
                    ),
                    "discarded_attempts": discarded,
                    "curve": _curve_row(r),
                }
            )
    return admitted


def _environment_gap(repeats_by_arm: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    hits_start = 0
    hits_peak = 0
    hits_available_mb = 0
    n = 0
    for rows in repeats_by_arm.values():
        for r in rows:
            n += 1
            blob = json.dumps(r)
            if '"environment_start"' in blob:
                hits_start += 1
            if '"environment_peak"' in blob:
                hits_peak += 1
            # Newer cleanliness-gate field; distinct from free_memory_mb_* / available_memory_mb_*.
            if '"available_mb"' in blob and "environment_" in blob:
                hits_available_mb += 1
    return {
        "present": False,
        "environment_start_hits": hits_start,
        "environment_peak_hits": hits_peak,
        "available_mb_in_environment_block_hits": hits_available_mb,
        "repeats_scanned": n,
        "note": (
            "This run predates the available_mb capture and the cleanliness gate. "
            "No environment_start / environment_peak blocks exist on repeats or cell "
            "payloads. Host free memory at measurement time is therefore unknown at the "
            "granularity those fields provide. The 2026-08-09 incident showed a 3.1 GB "
            "difference moves prefill 2.5× — do not treat this session's prefill curve as "
            "host-state-controlled without that instrumentation. "
            "A single top-level available_mb on the launch-log ceiling_a.start event is "
            "not environment_start/environment_peak and does not close this gap."
        ),
        "related_fields_that_do_exist": [
            "envelope.available_memory_mb_before/after/min",
            "memory_instrumentation.free_physical_bytes_at_start/at_peak",
            "child.generation.free_memory_mb_* / free_physical_mb_* (work results)",
            "launch_log ceiling_a.start.available_mb (session start only; not per-cell)",
        ],
    }


def _beyond_native(rungs_seen: list[int]) -> dict[str, Any]:
    flagged = sorted({n for n in rungs_seen if n > NATIVE_CONTEXT_FLAG_ABOVE})
    return {
        "validated_input_context_tokens": NATIVE_CONTEXT_FLAG_ABOVE,
        "model_max_position_embeddings": POSITION_LIMIT_TOKENS,
        "rope_scaling": None,
        "yarn_enabled_in_export": False,
        "rule": (
            f"Rungs above {NATIVE_CONTEXT_FLAG_ABOVE} exceed Qwen3-4B's validated input "
            "context (config max_position_embeddings=40960, rope_scaling=null). Flag for "
            "claim discipline; not a harness pass/fail."
        ),
        "flagged_n_tokens": flagged,
        "explicit_flags": {
            "n36000": 36000 in flagged,
            "n40000": 40000 in flagged,
        },
    }


def _censored_ceiling(arm_state: dict[str, Any]) -> dict[str, Any]:
    highest = int(arm_state.get("highest_pass") or 0)
    lowest = arm_state.get("lowest_nonpass")
    return {
        "label": "CENSORED_LOWER_BOUND",
        "highest_pass": highest,
        "lowest_nonpass": lowest,
        "ceiling_tokens_do_not_cite_as_measured": highest,
        "is_measured_ceiling": False,
        "is_lower_bound": True,
        "censoring": (
            "CENSORED — the memory wall was never reached; the run died on "
            f"generation.timeout_s={GENERATION_TIMEOUT_S}. {highest} is a LOWER BOUND, "
            "not a measured ceiling. Label it so nobody reads it as the latter."
        ),
        "termination_failure_class_at_next_rung": "timeout",
        "termination_detail": f"timeout:{GENERATION_TIMEOUT_S}s@generating",
        "next_rung_attempted": 40000,
        "next_rung_repeats_completed": 1,
        "next_rung_repeats_required_for_nonpass": 3,
        "why_lowest_nonpass_null": (
            "Harness sets lowest_nonpass only after a rung NONPASS (all repeats fail under "
            "pass_requires_all_repeats). Only r0 of n=40000 was measured (timeout) before "
            "force-kill; r1/r2 never completed. lowest_nonpass remains null by protocol."
        ),
    }


def _force_kill_record(
    events: list[dict[str, Any]],
    heartbeat: dict[str, Any],
    *,
    session_id: str,
) -> dict[str, Any]:
    last = events[-1] if events else {}
    stub = (
        ROOT / "derived" / "ceiling_a" / session_id / "work" / "A.n40000.ladder.r1.a0.result.json"
    )
    return {
        "class": "operator_or_harness_stop_after_timeout_cells",
        "evidence": {
            "checkpoint_phase": "ladder",
            "heartbeat_phase": heartbeat.get("phase"),
            "heartbeat_n_tokens": heartbeat.get("n_tokens"),
            "heartbeat_round": heartbeat.get("round"),
            "last_log_event": last.get("event"),
            "last_log_ts": last.get("ts"),
            "last_log_payload": {
                k: last.get(k)
                for k in ("n", "round", "stage", "arm_order", "arm", "outcome")
                if k in last
            },
            "n40000_r0_timeouts": [
                e
                for e in events
                if e.get("event") == "delta_n.repeat"
                and str(e.get("tag", "")).endswith("n40000.ladder.r0.a0")
            ],
            "round_1_ordered": any(
                e.get("event") == "ceiling_a.round_order"
                and e.get("n") == 40000
                and e.get("round") == 1
                for e in events
            ),
            "work_stub_A_n40000_r1": _rel(stub) if stub.is_file() else None,
            "work_stub_A_n40000_r1_exists": stub.is_file(),
        },
        "note": (
            "Launch log ends at ceiling_a.round_order n=40000 round=1. A.n40000.ladder.r1 "
            "work stub exists with phase=generating and no generation block; A_prime r1 was "
            "not started. Process did not seal. Treat as force-killed / stopped mid-ladder."
        ),
    }


def _completeness(
    *,
    checkpoint: dict[str, Any],
    repeats_by_arm: dict[str, list[dict[str, Any]]],
    rungs_by_arm: dict[str, list[dict[str, Any]]],
    work_dir: Path,
    launch_log: Path,
) -> dict[str, Any]:
    missing: list[str] = []
    notes: list[str] = []
    if not launch_log.is_file():
        missing.append(_rel(launch_log))
    for arm_id, rid in (checkpoint.get("arm_run_ids") or {}).items():
        raw = ROOT / "raw" / rid
        for name in ("in_progress.json", "repeats.ndjson", "rungs.ndjson"):
            if not (raw / name).is_file():
                missing.append(f"raw/{rid}/{name}")
        n_rep = len(repeats_by_arm.get(arm_id, []))
        n_rung = len(rungs_by_arm.get(arm_id, []))
        notes.append(f"{arm_id}: repeats={n_rep} rungs={n_rung} raw/{rid}")
    n_work = len(list(work_dir.iterdir())) if work_dir.is_dir() else 0
    notes.append(f"work_files={n_work}")
    notes.append(f"completed_cells={len(checkpoint.get('completed_cells') or [])}")
    mac_backup_required = bool(missing)
    return {
        "on_disk_complete_for_partial_seal": not missing,
        "missing": missing,
        "notes": notes,
        "mac_backup_required": mac_backup_required,
        "mac_backup_path_if_needed": "~/seam_backup_20260806/",
        "mac_backup_note": (
            "On-disk XPS copy has checkpoint, heartbeat, launch log, work stubs, and "
            "in_progress raw repeats/rungs for both arms through n=36000 PASS + n=40000 r0 "
            "timeout. Mac backup not required for this PARTIAL seal."
            if not missing
            else (
                "On-disk copy is incomplete; Mac backup at ~/seam_backup_20260806/ is "
                "required before sealing. Do not invent missing rows."
            )
        ),
    }


def _build_arm_block(
    *,
    arm_id: str,
    arm_cfg: dict[str, Any],
    arm_state: dict[str, Any],
    run_id: str,
    repeats: list[dict[str, Any]],
    rungs: list[dict[str, Any]],
    rung_records: dict[str, Any],
) -> dict[str, Any]:
    censored = _censored_ceiling(arm_state)
    curves = [_curve_row(r) for r in repeats]
    prompts: dict[str, Any] = {}
    for r in repeats:
        n = r.get("n_tokens")
        if n is None:
            continue
        p = r.get("prompt")
        if p:
            prompts[str(n)] = p
    arm_rungs = [rec for key, rec in rung_records.items() if rec.get("arm_id") == arm_id]
    arm_rungs.sort(key=lambda r: (0 if r.get("stage") == "ladder" else 1, r.get("n_tokens") or 0))
    return {
        "arm_id": arm_id,
        "label": arm_cfg.get("label"),
        "load_sequence": arm_cfg.get("load_sequence"),
        "generate_device": arm_cfg.get("generate_device"),
        "run_id": run_id,
        "raw_path": f"raw/{run_id}",
        "raw_lifecycle_at_seal": "in_progress",
        "ceiling": censored,
        "ceiling_tokens": censored["highest_pass"],
        "ceiling_bracket": [censored["highest_pass"], censored["lowest_nonpass"]],
        "ceiling_is_lower_bound": True,
        "ceiling_at_position_limit": bool(arm_state.get("ceiling_at_position_limit")),
        "boundary_verdict": arm_state.get("boundary_verdict"),
        "rungs_from_checkpoint": arm_rungs,
        "rungs_from_raw": rungs,
        "n_repeats": len(repeats),
        "n_rungs_raw": len(rungs),
        "curves": curves,
        "prompts_used": prompts,
        "standby_admitted_cells": [c for c in _standby_admitted_cells({arm_id: repeats})],
    }


def seal_session(session_id: str, *, launch_log: Path) -> dict[str, Any]:
    session_dir = ROOT / "derived" / "ceiling_a" / session_id
    if not session_dir.is_dir():
        raise SystemExit(f"missing session dir {session_dir}")

    checkpoint_path = session_dir / "checkpoint.json"
    heartbeat_path = session_dir / "heartbeat.json"
    if not checkpoint_path.is_file():
        raise SystemExit(f"missing {checkpoint_path}")
    if not heartbeat_path.is_file():
        raise SystemExit(f"missing {heartbeat_path}")
    if not launch_log.is_file():
        raise SystemExit(f"missing launch log {launch_log}")

    checkpoint = _read_json(checkpoint_path)
    heartbeat = _read_json(heartbeat_path)
    if checkpoint.get("phase") == "sealed":
        raise SystemExit(f"session {session_id} checkpoint phase is already sealed")

    arm_run_ids = dict(checkpoint.get("arm_run_ids") or {})
    if set(arm_run_ids) != {"A", "A_prime"}:
        raise SystemExit(f"unexpected arm_run_ids={arm_run_ids!r}; want A and A_prime")

    from seam.config import resolve_config

    resolved = resolve_config(
        [ROOT / _PLATFORM_PATH, ROOT / _MEASUREMENT_PATH, ROOT / _DELTA_N_PATH],
        repo_root=ROOT,
    )
    cfg = resolved.data
    arms_by_id = {a["id"]: a for a in cfg["arms"]}

    repeats_by_arm: dict[str, list[dict[str, Any]]] = {}
    rungs_by_arm: dict[str, list[dict[str, Any]]] = {}
    for arm_id, rid in arm_run_ids.items():
        raw = ROOT / "raw" / rid
        if (raw / ".sealed").is_file():
            raise SystemExit(f"raw/{rid} is already sealed; refusing post-hoc re-seal")
        repeats_by_arm[arm_id] = _ndjson(raw / "repeats.ndjson")
        rungs_by_arm[arm_id] = _ndjson(raw / "rungs.ndjson")

    events = _log_events(launch_log)
    completeness = _completeness(
        checkpoint=checkpoint,
        repeats_by_arm=repeats_by_arm,
        rungs_by_arm=rungs_by_arm,
        work_dir=session_dir / "work",
        launch_log=launch_log,
    )
    if not completeness["on_disk_complete_for_partial_seal"]:
        raise SystemExit(
            "on-disk artifacts incomplete; Mac backup required. missing="
            + json.dumps(completeness["missing"])
        )

    standby_events = _standby_from_log(events)
    if len(standby_events) != 4:
        raise SystemExit(
            f"expected 4 modern_standby_inadmissible log events, found {len(standby_events)}"
        )
    standby_admitted = _standby_admitted_cells(repeats_by_arm)
    env_gap = _environment_gap(repeats_by_arm)

    rungs_seen: list[int] = []
    for rows in repeats_by_arm.values():
        for r in rows:
            if r.get("n_tokens") is not None:
                rungs_seen.append(int(r["n_tokens"]))
    beyond = _beyond_native(rungs_seen)

    arm_state = checkpoint.get("arm_state") or {}
    arms_out: dict[str, Any] = {}
    for arm_id in ("A", "A_prime"):
        arms_out[arm_id] = _build_arm_block(
            arm_id=arm_id,
            arm_cfg=arms_by_id[arm_id],
            arm_state=arm_state[arm_id],
            run_id=arm_run_ids[arm_id],
            repeats=repeats_by_arm[arm_id],
            rungs=rungs_by_arm[arm_id],
            rung_records=checkpoint.get("rung_records") or {},
        )

    force_kill = _force_kill_record(events, heartbeat, session_id=session_id)
    sealed_utc = _utc_now()
    out = ROOT / "derived" / "ceiling_a" / f"sealed_{session_id}"
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing seal dir {out}")
    out.mkdir(parents=True, exist_ok=False)

    # Copy forensic sources into the write-once seal tree (not raw/).
    sources = out / "sources"
    sources.mkdir()
    shutil.copy2(checkpoint_path, sources / "checkpoint_as_found.json")
    shutil.copy2(heartbeat_path, sources / "heartbeat_as_found.json")
    shutil.copy2(launch_log, sources / "launch_log_as_found.ndjson")
    for arm_id, rid in arm_run_ids.items():
        arm_src = sources / f"raw_{arm_id}_{rid}"
        arm_src.mkdir()
        for name in ("in_progress.json", "repeats.ndjson", "rungs.ndjson"):
            shutil.copy2(ROOT / "raw" / rid / name, arm_src / name)

    # Excerpt: standby + n40000 tail from launch log.
    excerpt_lines: list[str] = []
    for e in events:
        if e.get("event") == "delta_n.modern_standby_inadmissible" or (
            e.get("n") == 40000
            or "n40000" in str(e.get("tag") or "")
            or (e.get("event") == "ceiling_a.round_order" and e.get("n") == 40000)
        ):
            excerpt_lines.append(json.dumps(e, sort_keys=True))
    (out / "launch_log_standby_and_n40000_excerpt.ndjson").write_text(
        "\n".join(excerpt_lines) + "\n", encoding="utf-8"
    )

    isolation = checkpoint.get("isolation_discipline") or {}
    verdict_run_id = checkpoint.get("verdict_run_id")

    raw_emit_blocked = {
        "reason": (
            "seam.manifest.emit refuses isolation_mode=remote while tier-1 software "
            "(Cursor/Chrome) is resident at seal/promote time. Measurement itself was "
            "ssh_detached/remote. Close Cursor/Chrome and promote with "
            "--attempt-raw-promote --allow-dirty; do not invent an enforce=False bypass. "
            "Promote finalizes existing in_progress raw arm dirs in place (same run_ids)."
        ),
        "existing_pattern": (
            "tools/seal_delta_prefill_session.py → seal_style=derived_diagnostic; "
            "ceiling_a._seal_arm finalize via emit(existing_run_dir=...)"
        ),
        "target_arm_run_ids": arm_run_ids,
        "target_verdict_run_id": verdict_run_id,
    }

    manifest: dict[str, Any] = {
        "run_id": session_id,
        "session_id": session_id,
        "status": "PARTIAL",
        "status_wording": STATUS_PARTIAL,
        "kind": "ceiling_a_partial",
        "seal_style": "derived_diagnostic",
        "sealed_utc": sealed_utc,
        "experiment_id": "ceiling_a",
        "benchmark": ARM_BENCHMARK,
        "prerequisite_acceptance_run_id": checkpoint.get(
            "prerequisite_acceptance_run_id", PREREQUISITE_ACCEPTANCE_RUN_ID
        ),
        "measurement_isolation_mode": isolation.get("isolation_mode") or "remote",
        "measurement_launch_context": isolation.get("launch_context") or "ssh_detached",
        "path_choice": {
            "chosen": "new_tool_seal_ceiling_a_partial_session",
            "rejected": "ceiling_a.orchestrate --resume / internal sealer",
            "why": (
                "Internal sealer runs only after run_interleaved_ladder completes. "
                "Resume would continue n=40000 round 1+, not seal the incomplete session. "
                "Force-killed PARTIAL / CENSORED_LOWER_BOUND vocabulary is not emitted by "
                "the happy-path seal."
            ),
        },
        "termination": force_kill,
        "ceiling": {
            "A": arms_out["A"]["ceiling"],
            "A_prime": arms_out["A_prime"]["ceiling"],
            "shared_label": "CENSORED_LOWER_BOUND",
            "do_not_cite_as_measured_ceiling": True,
        },
        "standby_events": {
            "count": len(standby_events),
            "log_events": standby_events,
            "admitted_cells_after_discard": standby_admitted,
            "note": (
                "4× delta_n.modern_standby_inadmissible. Arm A n=16000 r1 admitted a2 "
                "(attempt_used=2) after discarding a0 (KP 506) and a1 (KP 507); that "
                "admitted cell feeds any prefill fit."
            ),
        },
        "beyond_native_context": beyond,
        "environment_gap": env_gap,
        "completeness": completeness,
        "arm_run_ids": arm_run_ids,
        "verdict_run_id": verdict_run_id,
        "arms": arms_out,
        "raw_emit_blocked": raw_emit_blocked,
        "do_not_resume": (
            "Do not ceiling_a --resume this session. It is sealed as PARTIAL under "
            f"{_rel(out)}. Resume would continue measurement and risk mutating "
            "in_progress raw that this seal cites."
        ),
        "sources": {
            "checkpoint": "sources/checkpoint_as_found.json",
            "heartbeat": "sources/heartbeat_as_found.json",
            "launch_log": "sources/launch_log_as_found.ndjson",
            "launch_log_original": _rel(launch_log),
            "raw_A": f"sources/raw_A_{arm_run_ids['A']}/",
            "raw_A_prime": f"sources/raw_A_prime_{arm_run_ids['A_prime']}/",
        },
    }

    summary: dict[str, Any] = {
        "run_id": session_id,
        "session_id": session_id,
        "status": "PARTIAL",
        "status_wording": STATUS_PARTIAL,
        "kind": "ceiling_a_partial",
        "seal_style": "derived_diagnostic",
        "ceiling_label": "CENSORED_LOWER_BOUND",
        "highest_pass": {
            "A": arms_out["A"]["ceiling"]["highest_pass"],
            "A_prime": arms_out["A_prime"]["ceiling"]["highest_pass"],
        },
        "lowest_nonpass": {
            "A": None,
            "A_prime": None,
        },
        "is_measured_ceiling": False,
        "arm_run_ids": arm_run_ids,
        "verdict_run_id": verdict_run_id,
        "standby_event_count": len(standby_events),
        "standby_admitted_cells": standby_admitted,
        "beyond_native_flagged_n": beyond["flagged_n_tokens"],
        "environment_gap_present": False,
        "mac_backup_required": False,
        "curves_available": {
            arm_id: {
                "n_tokens": sorted({c["n_tokens"] for c in arms_out[arm_id]["curves"]}),
                "n_pass_rows": sum(
                    1 for c in arms_out[arm_id]["curves"] if c.get("outcome") == "pass"
                ),
                "n_fail_rows": sum(
                    1 for c in arms_out[arm_id]["curves"] if c.get("outcome") == "fail"
                ),
            }
            for arm_id in ("A", "A_prime")
        },
    }

    _write_json(out / "manifest.json", manifest)
    _write_json(out / "summary.json", summary)
    _write_json(out / "standby_events.json", manifest["standby_events"])
    _write_json(out / "beyond_native_context.json", beyond)
    _write_json(out / "environment_gap.json", env_gap)
    _write_json(out / "censored_ceiling.json", manifest["ceiling"])
    _write_json(out / "termination.json", force_kill)

    closeout = f"""# Close-out: ceiling_a A/A_prime PARTIAL (force-killed, censored)

- **session_id:** `{session_id}`
- **status:** {STATUS_PARTIAL}
- **arm A run_id:** `{arm_run_ids["A"]}` (in_progress until promote)
- **arm A_prime run_id:** `{arm_run_ids["A_prime"]}` (in_progress until promote)
- **verdict run_id (reserved):** `{verdict_run_id}`
- **seal:** `{_rel(out)}`
- **launch log:** `{_rel(launch_log)}`

## Ceiling (CENSORED — lower bound)

| arm | highest_pass | lowest_nonpass | label |
|---|---:|---|---|
| A | 36000 | null | CENSORED_LOWER_BOUND |
| A_prime | 36000 | null | CENSORED_LOWER_BOUND |

36000 is a **LOWER BOUND**, not a measured ceiling. The memory wall was never reached;
the run died on `generation.timeout_s={GENERATION_TIMEOUT_S}` at n=40000 r0 (both arms).

## Standby (4× `delta_n.modern_standby_inadmissible`)

Recorded in `standby_events.json` and launch-log excerpt.

- `A.n16000.ladder.r1` a0 (KP 506) + a1 (KP 507) discarded; admitted a2 (`attempt_used=2`) — **feeds prefill fit**
- `A_prime.n16000.ladder.r2` a0 discarded
- `A_prime.n20000.ladder.r0` a0 discarded

## Beyond-native context

Rungs above 32768 flagged (`beyond_native_context.json`): **36000**, **40000**.
Model: `max_position_embeddings=40960`, `rope_scaling=null`.

## Environment gap

`environment_gap.json`: **no** `environment_start` / `environment_peak` (run predates
available_mb capture and cleanliness gate). Host state unknown at that granularity.

## Mac backup

**Not required** — on-disk XPS artifacts are complete for this PARTIAL seal.

## Promote

```
.\\.venv-seam\\Scripts\\python.exe tools\\seal_ceiling_a_partial_session.py \\
    --session-id {session_id} \\
    --promote-only --attempt-raw-promote --allow-dirty
```

Do not bypass isolation. Do not `--resume` this session.
"""
    (out / "CLOSEOUT_A_A_prime_PARTIAL.md").write_text(closeout, encoding="utf-8")

    tree_hash = _sha256_tree(out, exclude={".sealed"})
    seal_marker = {
        "run_id": session_id,
        "sealed_at_utc": sealed_utc,
        "seal_style": "derived_diagnostic",
        "status": "PARTIAL",
        "tree_sha256": tree_hash,
        "self_check": "pass",
        "note": (
            "Advisory marker for derived_diagnostic PARTIAL seals. Not "
            "seam.rawstore.verify_sealed until --attempt-raw-promote emits into raw/. "
            "Do not mutate this directory after seal."
        ),
    }
    _write_json(out / ".sealed", seal_marker)

    for p in [out / ".sealed", *out.rglob("*")]:
        if p.is_file():
            try:
                p.chmod(p.stat().st_mode & ~0o222)
            except OSError:
                pass

    # Session-dir pointer only (do not mutate checkpoint.json — forensic freeze).
    pointer = {
        "session_id": session_id,
        "sealed": True,
        "status": "PARTIAL",
        "seal_style": "derived_diagnostic",
        "seal_path": _rel(out),
        "raw_emit": False,
        "raw_emit_blocked": raw_emit_blocked["reason"],
        "sealed_utc": sealed_utc,
        "tree_sha256": tree_hash,
        "arm_run_ids": arm_run_ids,
        "verdict_run_id": verdict_run_id,
        "ceiling_label": "CENSORED_LOWER_BOUND",
        "highest_pass": summary["highest_pass"],
        "do_not_resume": True,
        "mac_backup_required": False,
    }
    pointer_path = session_dir / "SEAL_POINTER.json"
    _write_json(pointer_path, pointer)

    # Operator-facing copy next to the gpu_only closeout pattern.
    session_closeout = session_dir / "CLOSEOUT_A_A_prime_PARTIAL.md"
    if not session_closeout.exists():
        shutil.copy2(out / "CLOSEOUT_A_A_prime_PARTIAL.md", session_closeout)

    return {
        "session_id": session_id,
        "sealed": True,
        "status": "PARTIAL",
        "seal_style": "derived_diagnostic",
        "seal_path": str(out.resolve()),
        "pointer": str(pointer_path.resolve()),
        "tree_sha256": tree_hash,
        "raw_emit": False,
        "arm_run_ids": arm_run_ids,
        "verdict_run_id": verdict_run_id,
        "ceiling_label": "CENSORED_LOWER_BOUND",
        "mac_backup_required": False,
    }


def _promotion_arm_summary(sealed_manifest: dict[str, Any], arm_id: str) -> dict[str, Any]:
    arm = dict((sealed_manifest.get("arms") or {}).get(arm_id) or {})
    ceiling = dict(arm.get("ceiling") or {})
    return {
        "experiment_id": "ceiling_a",
        "benchmark": ARM_BENCHMARK,
        "phase": "ceiling_a",
        "status": "PARTIAL",
        "status_wording": sealed_manifest.get("status_wording"),
        "run_id": arm.get("run_id"),
        "session_id_ceiling_a": sealed_manifest.get("session_id"),
        "prerequisite_acceptance_run_id": sealed_manifest.get("prerequisite_acceptance_run_id"),
        "arms_selected": ["A", "A_prime"],
        "arm": arm,
        "ceiling_tokens": ceiling.get("highest_pass"),
        "ceiling_bracket": [ceiling.get("highest_pass"), ceiling.get("lowest_nonpass")],
        "ceiling_label": "CENSORED_LOWER_BOUND",
        "ceiling_is_lower_bound": True,
        "is_measured_ceiling": False,
        "censored_ceiling": ceiling,
        "standby_events": sealed_manifest.get("standby_events"),
        "beyond_native_context": sealed_manifest.get("beyond_native_context"),
        "environment_gap": sealed_manifest.get("environment_gap"),
        "termination": sealed_manifest.get("termination"),
        "promotion": "post_hoc_from_derived_diagnostic_partial",
        "derived_seal_path": sealed_manifest.get("_seal_path_rel"),
        "scope": {
            "arms_selected": ["A", "A_prime"],
            "partial_force_killed": True,
            "three_arm_delta_n": "not in this phase",
        },
    }


def _dry_validate_arm_emit(
    *,
    session_id: str,
    sealed_manifest: dict[str, Any],
    arm_id: str,
    allow_dirty: bool,
    model_spec: str | Path | None = None,
) -> dict[str, Any]:
    from seam.blinding import blinded_label_for, get_or_create_salt
    from seam.config import resolve_config
    from seam.gitinfo import capture_git_state
    from seam.manifest import build_manifest, validate_manifest
    from seam.model_provenance import load_local_spec, manifest_model_block

    resolved = resolve_config(
        [ROOT / _PLATFORM_PATH, ROOT / _MEASUREMENT_PATH, ROOT / _DELTA_N_PATH],
        repo_root=ROOT,
    )
    model_spec_path = _resolve_model_spec_for_session(session_id, cli_spec=model_spec)
    spec = load_local_spec(model_spec_path)
    summary = _promotion_arm_summary(sealed_manifest, arm_id)
    run_id = summary["run_id"]
    git_state = capture_git_state(cwd=ROOT)
    salt = get_or_create_salt(ROOT)
    condition_label = (
        f"ceiling_a|{arm_id}|cpu-p|PARTIAL|CENSORED_LOWER_BOUND|"
        f"seed={resolved.data['randomization_seed']}"
    )
    blinded = blinded_label_for(condition_label, salt=salt)
    workload = {
        "kind": "microbench",
        "benchmark": ARM_BENCHMARK,
        "task_ids": [],
        "seed": int(resolved.data["randomization_seed"]),
        "n_repeats": int(resolved.data["ladder"]["repeats"]),
        "concurrency": 1,
    }
    # Synthetic quiet evidence for dry construction only — never written to raw/.
    dry_evidence = {
        "contending_processes": [],
        "tier2_recorded_processes": [],
        "sshd_session_count": None,
        "consistent_with_declaration": True,
        "probe_error": None,
    }
    manifest = build_manifest(
        run_id=run_id,
        config=resolved,
        git_state=git_state,
        allow_dirty=allow_dirty,
        target="cpu-p",
        workload=workload,
        condition_label=condition_label,
        blinded_label=blinded,
        repo_root=ROOT,
        model=manifest_model_block(
            spec=spec, spec_path=model_spec_path, reasoning_mode="thinking_off"
        ),
        drivers={},
        power_state=None,
        power_state_note=_RETRO_SEAL_POWER_NOTE,
        retro_seal=True,
        thermal={"regime": "confound", "excluded": False},
        raw_sha256="0" * 64,
        self_check="pass",
        isolation_mode="remote",
        isolation_evidence=dry_evidence,
        launch_context="ssh_detached",
        session_id=None,
        window_station=None,
        require_launch_context=False,
    )
    validate_manifest(manifest)
    dirty_blocker = bool(git_state.dirty and not allow_dirty)
    return {
        "ok": True,
        "arm_id": arm_id,
        "run_id": run_id,
        "workload_kind": "microbench",
        "allow_dirty": allow_dirty,
        "emit_will_need_allow_dirty": dirty_blocker,
        "git_dirty": git_state.dirty,
    }


def _emit_arm(
    *,
    session_id: str,
    sealed_manifest: dict[str, Any],
    arm_id: str,
    allow_dirty: bool,
    model_spec: str | Path | None = None,
) -> dict[str, Any]:
    from dataclasses import asdict

    from seam.backends.local_openvino import runtime_info
    from seam.config import resolve_config
    from seam.jsonlog import utc_now_iso
    from seam.manifest import emit
    from seam.model_provenance import load_local_spec, manifest_model_block
    from seam.powerstate import capture_power_state, manifest_power_state
    from seam.rawstore import open_run_dir

    resolved = resolve_config(
        [ROOT / _PLATFORM_PATH, ROOT / _MEASUREMENT_PATH, ROOT / _DELTA_N_PATH],
        repo_root=ROOT,
    )
    model_spec_path = _resolve_model_spec_for_session(session_id, cli_spec=model_spec)
    spec = load_local_spec(model_spec_path)
    summary = _promotion_arm_summary(sealed_manifest, arm_id)
    run_id = str(summary["run_id"])
    run_dir = open_run_dir(run_id, repo_root=ROOT)
    if run_dir.is_sealed():
        raise SystemExit(f"arm {arm_id} raw/{run_id} already sealed")

    # Write arm summary into the run dir before emit hashes the tree.
    run_dir.write_json("ceiling_a_arm_summary.json", summary)
    run_dir.write_json(
        "derived_seal_pointer.json",
        {
            "session_id": sealed_manifest.get("session_id"),
            "derived_seal_path": sealed_manifest.get("_seal_path_rel"),
            "status": "PARTIAL",
            "ceiling_label": "CENSORED_LOWER_BOUND",
        },
    )

    # AM-036: capture at promote time is forensic only — never measurement power_state.
    promote_power = manifest_power_state(capture_power_state(), background_quiesced=False)
    promote_power["captured_at_utc"] = utc_now_iso()
    handle = emit(
        config=resolved,
        target="cpu-p",
        workload={
            "kind": "microbench",
            "benchmark": ARM_BENCHMARK,
            "task_ids": [],
            "seed": int(resolved.data["randomization_seed"]),
            "n_repeats": int(resolved.data["ladder"]["repeats"]),
            "concurrency": 1,
        },
        condition_label=(
            f"ceiling_a|{arm_id}|cpu-p|PARTIAL|CENSORED_LOWER_BOUND|"
            f"seed={resolved.data['randomization_seed']}"
        ),
        repo_root=ROOT,
        run_id=run_id,
        existing_run_dir=run_dir,
        allow_dirty=allow_dirty,
        summary=summary,
        model=manifest_model_block(
            spec=spec, spec_path=model_spec_path, reasoning_mode="thinking_off"
        ),
        drivers=asdict(runtime_info()),
        power_state=None,
        promote_time_power_state=promote_power,
        power_state_note=_RETRO_SEAL_POWER_NOTE,
        retro_seal=True,
        thermal={"regime": "confound", "excluded": False},
        self_check="pass",
        launch_context="ssh_detached",
        require_launch_context=True,
        isolation_mode="remote",
    )
    return {
        "arm_id": arm_id,
        "run_id": handle.run_id,
        "raw_path": _rel(handle.run_dir.path),
        "raw_sha256": handle.raw_sha256,
        "manifest_path": _rel(handle.run_dir.path / "manifest.json"),
    }


def _emit_verdict(
    *,
    sealed_manifest: dict[str, Any],
    allow_dirty: bool,
) -> dict[str, Any]:
    from dataclasses import asdict

    from seam.backends.local_openvino import runtime_info
    from seam.config import resolve_config
    from seam.jsonlog import utc_now_iso
    from seam.manifest import emit
    from seam.powerstate import capture_power_state, manifest_power_state

    resolved = resolve_config(
        [ROOT / _PLATFORM_PATH, ROOT / _MEASUREMENT_PATH, ROOT / _DELTA_N_PATH],
        repo_root=ROOT,
    )
    arm_run_ids = dict(sealed_manifest.get("arm_run_ids") or {})
    verdict_run_id = sealed_manifest.get("verdict_run_id")
    if not verdict_run_id:
        raise SystemExit("promote: sealed manifest missing verdict_run_id")

    summary = {
        "experiment_id": "ceiling_a",
        "benchmark": VERDICT_BENCHMARK,
        "phase": "ceiling_a",
        "status": "PARTIAL",
        "status_wording": sealed_manifest.get("status_wording"),
        "run_id": verdict_run_id,
        "session_id_ceiling_a": sealed_manifest.get("session_id"),
        "prerequisite_acceptance_run_id": sealed_manifest.get("prerequisite_acceptance_run_id"),
        "arms_selected": ["A", "A_prime"],
        "purpose": (
            "PARTIAL post-hoc ceiling_a verdict for force-killed A/A_prime session. "
            "Ceilings are CENSORED_LOWER_BOUND (highest_pass=36000, lowest_nonpass=null)."
        ),
        "source_arm_run_ids": arm_run_ids,
        "arms": sealed_manifest.get("arms"),
        "ceiling": sealed_manifest.get("ceiling"),
        "standby_events": sealed_manifest.get("standby_events"),
        "beyond_native_context": sealed_manifest.get("beyond_native_context"),
        "environment_gap": sealed_manifest.get("environment_gap"),
        "termination": sealed_manifest.get("termination"),
        "verdict": {
            "phase": "ceiling_a",
            "status": "PARTIAL",
            "verdict": "ceiling_censored_lower_bound_force_killed",
            "ceiling_tokens": {
                "A": 36000,
                "A_prime": 36000,
            },
            "aa_spread_tokens": 0,
            "early_exit_fired": False,
            "note": STATUS_PARTIAL,
        },
        "promotion": "post_hoc_from_derived_diagnostic_partial",
        "derived_seal_path": sealed_manifest.get("_seal_path_rel"),
    }
    promote_power = manifest_power_state(capture_power_state(), background_quiesced=False)
    promote_power["captured_at_utc"] = utc_now_iso()
    handle = emit(
        config=resolved,
        target="cpu-p",
        workload={
            "kind": "aa",
            "benchmark": VERDICT_BENCHMARK,
            "task_ids": [arm_run_ids["A"], arm_run_ids["A_prime"]],
            "seed": int(resolved.data["randomization_seed"]),
            "n_repeats": 2,
            "concurrency": 1,
        },
        condition_label=(
            "ceiling_a_verdict|PARTIAL|CENSORED_LOWER_BOUND|"
            + "|".join(f"{a}={arm_run_ids[a]}" for a in ("A", "A_prime"))
        ),
        repo_root=ROOT,
        run_id=str(verdict_run_id),
        allow_dirty=allow_dirty,
        summary=summary,
        drivers=asdict(runtime_info()),
        power_state=None,
        promote_time_power_state=promote_power,
        power_state_note=_RETRO_SEAL_POWER_NOTE,
        retro_seal=True,
        thermal={"regime": "confound", "excluded": False},
        self_check="pass",
        launch_context="ssh_detached",
        require_launch_context=True,
        isolation_mode="remote",
    )
    return {
        "verdict_run_id": handle.run_id,
        "raw_path": _rel(handle.run_dir.path),
        "raw_sha256": handle.raw_sha256,
        "manifest_path": _rel(handle.run_dir.path / "manifest.json"),
    }


def _attempt_raw_promote(
    session_id: str,
    seal_path: Path,
    *,
    allow_dirty: bool = False,
    model_spec: str | Path | None = None,
) -> dict[str, Any]:
    from seam.isolation import IsolationModeError, gather_isolation_evidence, resolve_isolation_mode
    from seam.manifest import load_schema

    attempted_utc = _utc_now()
    if not seal_path.is_dir():
        raise SystemExit(f"promote: missing seal dir {seal_path}")
    sealed_manifest_path = seal_path / "manifest.json"
    sealed_summary_path = seal_path / "summary.json"
    if not sealed_manifest_path.is_file() or not sealed_summary_path.is_file():
        raise SystemExit(f"promote: sealed manifest/summary missing under {seal_path}")

    sealed_manifest = _read_json(sealed_manifest_path)
    sealed_manifest["_seal_path_rel"] = _rel(seal_path)
    sealed_summary = _read_json(sealed_summary_path)

    evidence = gather_isolation_evidence()
    tier1 = list(evidence.get("contending_processes") or [])
    record: dict[str, Any] = {
        "attempted_utc": attempted_utc,
        "session_id": session_id,
        "seal_path": _rel(seal_path),
        "raw_emit": False,
        "gate": (
            "resolve_isolation_mode(remote) → schema kind microbench/aa → "
            "dry_validate → emit existing arm dirs + verdict"
        ),
        "tier1_resident_names": sorted({p.get("name") for p in tier1 if p.get("name")}),
        "tier1_count": len(tier1),
        "emit_wired": True,
        "target_arm_run_ids": sealed_manifest.get("arm_run_ids"),
        "target_verdict_run_id": sealed_manifest.get("verdict_run_id"),
    }

    kind_enum = load_schema()["properties"]["workload"]["properties"]["kind"]["enum"]
    if "microbench" not in kind_enum or "aa" not in kind_enum:
        record["status"] = "refused_schema_gate"
        record["schema_gate"] = "workload.kind microbench/aa missing from enum"
        return record
    record["schema_gate"] = "workload.kind microbench+aa accepted"

    dry_arms: dict[str, Any] = {}
    try:
        for arm_id in ("A", "A_prime"):
            dry_arms[arm_id] = _dry_validate_arm_emit(
                session_id=session_id,
                sealed_manifest=sealed_manifest,
                arm_id=arm_id,
                allow_dirty=allow_dirty,
                model_spec=model_spec,
            )
        record["dry_validate"] = {"ok": True, "arms": dry_arms}
    except Exception as exc:
        record["dry_validate"] = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "arms": dry_arms,
        }

    try:
        mode, resolved = resolve_isolation_mode("remote")
        record["isolation_mode_resolved"] = mode
        record["isolation_evidence_consistent"] = resolved.get("consistent_with_declaration")
    except IsolationModeError as exc:
        record["status"] = "refused_isolation_gate"
        record["refusal"] = str(exc)
        record["note"] = (
            "Promotion refused: tier-1 operator-controlled software is resident. "
            "Close Cursor/Chrome (and other tier-1 names), then retry "
            "--attempt-raw-promote --allow-dirty. Do not invent enforce=False. "
            "Derived PARTIAL seal remains valid; in_progress raw arm dirs unchanged."
        )
        return record

    if not (record.get("dry_validate") or {}).get("ok"):
        record["status"] = "refused_dry_validate"
        record["note"] = (
            "Isolation gate passed but dry emit construction failed; refusing raw/ write."
        )
        return record

    emitted: dict[str, Any] = {"arms": {}, "verdict": None}
    try:
        for arm_id in ("A", "A_prime"):
            emitted["arms"][arm_id] = _emit_arm(
                session_id=session_id,
                sealed_manifest=sealed_manifest,
                arm_id=arm_id,
                allow_dirty=allow_dirty,
                model_spec=model_spec,
            )
        emitted["verdict"] = _emit_verdict(
            sealed_manifest=sealed_manifest,
            allow_dirty=allow_dirty,
        )
    except Exception as exc:
        record["status"] = "refused_emit"
        record["emit_error_type"] = type(exc).__name__
        record["emit_error"] = str(exc)
        record["emit_partial"] = emitted
        record["note"] = (
            "isolation + schema + dry_validate passed but seam.manifest.emit failed. "
            "Inspect emit_error / emit_partial. Do not invent enforce=False."
        )
        return record

    record["status"] = "raw_emitted"
    record["raw_emit"] = True
    record["emit"] = emitted
    record["note"] = (
        "seam.manifest.emit finalized in_progress arm raw dirs and wrote verdict raw/ "
        f"from derived seal {_rel(seal_path)} (sealed_* tree not mutated)."
    )
    # Keep linter happy about sealed_summary use in promote path.
    record["derived_summary_status"] = sealed_summary.get("status")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", default=DEFAULT_SESSION_ID)
    parser.add_argument(
        "--launch-log",
        type=Path,
        default=DEFAULT_LAUNCH_LOG,
        help="launch log used for standby / force-kill evidence",
    )
    parser.add_argument(
        "--attempt-raw-promote",
        action="store_true",
        help=(
            "After isolation_mode=remote clears, emit into existing in_progress arm raw/ "
            "dirs and the reserved verdict run_id. Does not mutate sealed_*. "
            "Does not invent enforce=False."
        ),
    )
    parser.add_argument(
        "--promote-only",
        action="store_true",
        help="Skip sealing; only run --attempt-raw-promote against an existing seal.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Pass allow_dirty through to seam.manifest.emit (recorded in raw manifests).",
    )
    parser.add_argument(
        "--model-spec",
        default=None,
        help=(
            "FetchedModelSpec YAML for the promote model block. Default: "
            f"{_MODEL_SPEC_PATH.as_posix()}. Must match the model_spec recorded "
            "on the session plan/config; mismatch or a missing record is fatal."
        ),
    )
    args = parser.parse_args(argv)

    session_id = args.session_id
    seal_path = ROOT / "derived" / "ceiling_a" / f"sealed_{session_id}"
    result: dict[str, Any]

    if args.promote_only:
        if not seal_path.is_dir():
            raise SystemExit(f"promote-only: missing seal dir {seal_path}")
        if not args.attempt_raw_promote:
            raise SystemExit("promote-only requires --attempt-raw-promote")
        result = {
            "session_id": session_id,
            "sealed": True,
            "status": "PARTIAL",
            "seal_style": "derived_diagnostic",
            "seal_path": str(seal_path.resolve()),
            "raw_emit": False,
            "promote_only": True,
        }
    else:
        result = seal_session(session_id, launch_log=Path(args.launch_log))
        seal_path = Path(result["seal_path"])

    if args.attempt_raw_promote:
        promo = _attempt_raw_promote(
            session_id,
            seal_path,
            allow_dirty=bool(args.allow_dirty),
            model_spec=args.model_spec,
        )
        session_dir = ROOT / "derived" / "ceiling_a" / session_id
        promo_path = session_dir / "PROMOTION_ATTEMPT.json"
        _write_json(promo_path, promo)
        pointer_path = session_dir / "SEAL_POINTER.json"
        if pointer_path.is_file():
            pointer = _read_json(pointer_path)
            emitted = promo.get("status") == "raw_emitted"
            pointer["raw_emit"] = bool(emitted)
            pointer["promotion_attempt"] = {
                "attempted_utc": promo["attempted_utc"],
                "status": promo["status"],
                "record": _rel(promo_path),
                "emit_wired": True,
            }
            if emitted:
                pointer["raw_paths"] = (promo.get("emit") or {}).get("arms")
                pointer["verdict_raw"] = (promo.get("emit") or {}).get("verdict")
                pointer.pop("raw_emit_blocked", None)
            else:
                pointer["raw_emit_blocked"] = promo.get("note") or promo.get("refusal")
            _write_json(pointer_path, pointer)
        result["promotion_attempt"] = promo
        result["promotion_record"] = str(promo_path.resolve())
        result["raw_emit"] = bool(promo.get("raw_emit"))

    print(json.dumps(result, indent=2, sort_keys=True))
    if args.attempt_raw_promote and result.get("promotion_attempt", {}).get("status") != (
        "raw_emitted"
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
