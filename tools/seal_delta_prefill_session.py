"""Seal a completed delta_prefill matrix session (derived_diagnostic) and promote to raw/.

Derived seal matches ``derived/gpu_smoke/_seal_partial_matrix.py`` (write-once under
``derived/``). Promotion into ``raw/`` uses ``seam.manifest.emit`` after the isolation gate
and schema gate both pass — same pattern as ``seam.tools.affinity_matrix._emit_matrix_manifest``.

Why promote is gated: ``emit()`` re-resolves ``isolation_mode`` against the *promote-time*
machine. Declaring ``remote`` (measurement was ssh_detached) is refused while Cursor /
Chrome are resident. Close them and re-run with ``--attempt-raw-promote``; do not invent
an ``enforce=False`` bypass. Sealed ``derived/.../sealed_*`` trees are never mutated.

Usage:
  .\\.venv-seam\\Scripts\\python.exe tools\\seal_delta_prefill_session.py \\
      --session-id 9f38eb15-6fe6-40b4-871b-a02ec5629bb1
  .\\.venv-seam\\Scripts\\python.exe tools\\seal_delta_prefill_session.py \\
      --session-id d5c98342-a0b2-41a9-b6e2-93ac7a39c3ba \\
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

# Session with the recorded CL_OUT_OF_RESOURCES / hung-child / operator-kill cell.
ROBUSTNESS_SESSION_ID = "d5c98342-a0b2-41a9-b6e2-93ac7a39c3ba"
ROBUSTNESS_LAUNCH_LOG = (
    ROOT / "derived" / "delta_prefill" / "_launches" / "delta_prefill_20260809_183202.log"
)

# Verbatim oneDNN stderr line from the launch log (not retained in the cell JSON).
ONEDNN_CL_OUT_OF_RESOURCES_LINE = (
    "onednn_verbose,v1,primitive,error,ocl,errcode -5,CL_OUT_OF_RESOURCES,"
    "src\\gpu\\intel\\ocl\\kernel.cpp:232,src\\gpu\\intel\\ocl\\kernel.cpp:232"
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_tree(root: Path, *, exclude: set[str]) -> str:
    """Canonical tree hash over relative posix paths (sorted), matching rawstore spirit."""
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


def _parse_utc(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _elapsed_s(started: str, ended: str) -> float:
    return (_parse_utc(ended) - _parse_utc(started)).total_seconds()


def _cell_durations(cells: list[dict[str, Any]], *, citing_session_id: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for c in cells:
        el = _elapsed_s(c["started_utc"], c["ended_utc"])
        rows.append(
            {
                "cell_index": c.get("cell_index"),
                "arm": c.get("arm"),
                "mode": c.get("mode"),
                "delta": c.get("delta"),
                "repeat": c.get("repeat"),
                "classification": c.get("classification"),
                "exit_code": c.get("exit_code"),
                "elapsed_s": round(el, 3),
            }
        )
    ok = [r["elapsed_s"] for r in rows if r["classification"] == "OK"]
    out: dict[str, Any] = {
        "per_cell": rows,
        "ok_n": len(ok),
        "ok_max_elapsed_s": max(ok) if ok else None,
        "ok_median_elapsed_s": sorted(ok)[len(ok) // 2] if ok else None,
        "non_ok_max_elapsed_s": max(
            (r["elapsed_s"] for r in rows if r["classification"] != "OK"),
            default=None,
        ),
    }
    if citing_session_id == ROBUSTNESS_SESSION_ID and ok:
        out["cell_timeout_derivation"] = {
            "citing_session_id": citing_session_id,
            "ok_max_elapsed_s": max(ok),
            "two_times_ok_max_s": round(2 * max(ok), 3),
            "chosen_cell_timeout_s": 1500,
            "rule": (
                "ceil(2 * ok_max_elapsed_s) from this session → 1234s, rounded up to 1500s. "
                "Hung cell was 2856.5s; generation.timeout_s=1800 bounds only generate() inside "
                "the child, not post-record teardown."
            ),
        }
    return out


def _robustness_finding(summary: dict[str, Any], durations: dict[str, Any]) -> dict[str, Any]:
    failed = None
    for c in summary.get("cells") or []:
        if (
            c.get("arm") == "gpu_only"
            and c.get("mode") == "RESIDENT"
            and int(c.get("delta") or 0) == 500
            and int(c.get("repeat") or -1) == 1
            and int(c.get("n_cached") or 0) == 12000
        ):
            failed = c
            break
    if failed is None:
        raise SystemExit(
            "expected failed cell gpu_only RESIDENT nc12000 d500 r1 missing from summary"
        )

    elapsed = _elapsed_s(failed["started_utc"], failed["ended_utc"])
    return {
        "class": "post_record_hung_child_operator_kill",
        "admissibility": "robustness_finding",
        "note": (
            "Not merely a bad cell: after CL_OUT_OF_RESOURCES / oneDNN failure the child "
            "wrote its cell record, then spun one core ~47 min until the operator killed it "
            "(exit=-1). Matrix continued; twelve cells remained after this hang. "
            "generation.timeout_s bounds generate() inside the child only — nothing bounded "
            "the post-record hang. Recorded verbatim as a harness robustness finding."
        ),
        "cell": {
            "arm": "gpu_only",
            "mode": "RESIDENT",
            "n_cached": 12000,
            "delta": 500,
            "repeat": 1,
            "cell_index": failed.get("cell_index"),
            "classification_recorded": failed.get("classification"),
            "exit_code": failed.get("exit_code"),
            "started_utc": failed.get("started_utc"),
            "ended_utc": failed.get("ended_utc"),
            "elapsed_s": round(elapsed, 3),
            "artifact": failed.get("artifact"),
            "execute_error_from_cell_json": failed.get("error"),
        },
        "onednn_stderr_verbatim": ONEDNN_CL_OUT_OF_RESOURCES_LINE,
        "onednn_stderr_source": str(ROBUSTNESS_LAUNCH_LOG.relative_to(ROOT)).replace("\\", "/"),
        "operator_kill": {
            "exit_code": -1,
            "narrative": (
                "child spun one core ~47 min after writing its record and was killed by "
                "the operator, exit=-1"
            ),
            "elapsed_s_observed": round(elapsed, 3),
        },
        "cell_duration_context": {
            "ok_max_elapsed_s": durations.get("ok_max_elapsed_s"),
            "hung_elapsed_s": round(elapsed, 3),
            "citing_session_id": ROBUSTNESS_SESSION_ID,
        },
    }


_PLATFORM_PATH = Path("configs/platforms/aipc-c1.yaml")
_MEASUREMENT_PATH = Path("configs/measurement.yaml")
_DELTA_N_PATH = Path("configs/delta_n.yaml")
_MODEL_SPEC_PATH = Path("configs/models/Qwen3-4B-int4-ov.yaml")


def _load_session_plan(session_id: str) -> dict[str, Any] | None:
    path = ROOT / "derived" / "delta_prefill" / session_id / "plan.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _resolve_model_spec_for_session(session_id: str, *, cli_spec: str | Path | None) -> Path:
    """Single-spec resolve for promote paths that still emit one model block."""
    from tools.seal_model_spec import resolve_sealer_model_spec

    return resolve_sealer_model_spec(
        root=ROOT,
        default_spec=_MODEL_SPEC_PATH,
        cli_spec=cli_spec,
        plan=_load_session_plan(session_id),
    )


def _resolve_model_spec_set_for_session(
    session_id: str,
    *,
    cells: list[dict[str, Any]],
    cli_spec: str | Path | None = None,
    cli_specs: list[str] | None = None,
) -> list[dict[str, str]]:
    """Multi-spec resolve: record every (path, ir_sha256); fail if a cell is outside the set."""
    from tools.seal_model_spec import resolve_sealer_model_spec_set

    return resolve_sealer_model_spec_set(
        root=ROOT,
        plan=_load_session_plan(session_id),
        cells=cells,
        cli_spec=cli_spec,
        cli_specs=cli_specs,
    )


def _unwrap_ps_list(obj: Any) -> list[Any]:
    from tools.seal_model_spec import _unwrap_ps_list as _unwrap

    return _unwrap(obj)


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _measurement_placement(seal_path: Path) -> dict[str, Any]:
    """Placement recorded on measurement cells (not promote-time process placement)."""
    cells = sorted((seal_path / "cells").glob("delta_prefill_*.json"))
    if not cells:
        raise SystemExit(f"promote: no cell JSON under {seal_path / 'cells'}")
    sample = _read_json(cells[0])
    return {
        "launch_context": sample.get("launch_context") or "ssh_detached",
        "session_id": sample.get("session_id"),
        "window_station": sample.get("window_station"),
        "source_cell": f"cells/{cells[0].name}",
        "openvino": sample.get("ov_version"),
        "genai": ((sample.get("diagnostics") or {}).get("chat_api") or {}).get("genai_version"),
        "model_id": sample.get("model_id"),
    }


def _task_ids_from_sealed(sealed_manifest: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for cell in sealed_manifest.get("cells") or []:
        ids.append(
            f"{cell.get('arm_id')}:{cell.get('mode')}:nc{cell.get('n_cached')}:"
            f"d{cell.get('delta')}:r{cell.get('repeat_index')}"
        )
    return ids


def _build_promotion_summary(
    *,
    session_id: str,
    seal_path: Path,
    sealed_manifest: dict[str, Any],
    sealed_summary: dict[str, Any],
    placement: dict[str, Any],
) -> dict[str, Any]:
    matrix = dict(sealed_manifest.get("matrix") or {})
    summary: dict[str, Any] = {
        "kind": "delta_prefill_matrix",
        "promotion": "post_hoc_from_derived_diagnostic",
        "session_id": session_id,
        "run_id": session_id,
        "derived_seal_path": _rel(seal_path),
        "derived_seal_style": sealed_manifest.get("seal_style"),
        "derived_tree_sha256": None,
        "measurement_isolation_mode": sealed_manifest.get("measurement_isolation_mode"),
        "measurement_launch_context": sealed_manifest.get("measurement_launch_context"),
        "measurement_placement": {
            "launch_context": placement.get("launch_context"),
            "session_id": placement.get("session_id"),
            "window_station": placement.get("window_station"),
            "source_cell": placement.get("source_cell"),
        },
        "matrix": matrix,
        "admissibility": sealed_manifest.get("admissibility"),
        "cell_durations": sealed_manifest.get("cell_durations"),
        "cells_completed": sealed_summary.get("cells_completed"),
        "cells_ok": sealed_summary.get("cells_ok"),
        "cells_failed": sealed_summary.get("cells_failed"),
        "notes": {
            "raw_artifacts_copied_from": _rel(seal_path),
            "sealed_derived_not_mutated": True,
            "power_state": (
                "null at promote time: post-hoc emit does not re-capture quiescence; "
                "per-cell environment_* blocks remain in cell JSON under cells/"
            ),
        },
    }
    sealed_marker = seal_path / ".sealed"
    if sealed_marker.is_file():
        summary["derived_tree_sha256"] = _read_json(sealed_marker).get("tree_sha256")

    robustness_path = seal_path / "robustness_finding.json"
    if robustness_path.is_file():
        summary["robustness_finding"] = _read_json(robustness_path)
        summary["notes"]["robustness_finding_path"] = "robustness_finding.json"
        summary["notes"]["robustness_finding_class"] = summary["robustness_finding"].get("class")
    elif session_id == ROBUSTNESS_SESSION_ID:
        raise SystemExit(
            f"promote: robustness session {session_id} missing "
            f"{robustness_path.name} under sealed tree"
        )
    return summary


def _stage_raw_artifacts(
    run_dir: Any,
    *,
    seal_path: Path,
    sealed_manifest: dict[str, Any],
    sealed_summary: dict[str, Any],
) -> dict[str, Any]:
    """Copy sealed diagnostic artifacts into raw/<run_id>/ before integrity hash.

    Reads from the sealed derived tree only; never writes back into sealed_*.
    """
    cells_src = seal_path / "cells"
    cells_dst = run_dir.path / "cells"
    cells_dst.mkdir(exist_ok=False)
    copied_cells: list[str] = []
    for src in sorted(cells_src.glob("delta_prefill_*.json")):
        dest = cells_dst / src.name
        shutil.copy2(src, dest)
        copied_cells.append(f"cells/{src.name}")

    for name in (
        "session_summary_as_found.json",
        "session_plan_as_found.json",
        "launch_log_failed_cell_excerpt.txt",
        "robustness_finding.json",
    ):
        src = seal_path / name
        if src.is_file():
            shutil.copy2(src, run_dir.path / name)

    # Pointers / digests of the derived seal (not a mutation of it).
    run_dir.write_json(
        "derived_seal_manifest.json",
        sealed_manifest,
    )
    run_dir.write_json(
        "derived_seal_summary.json",
        sealed_summary,
    )
    marker = seal_path / ".sealed"
    if marker.is_file():
        shutil.copy2(marker, run_dir.path / "derived_seal_marker.json")

    extra: dict[str, Any] = {
        "n_cells_copied": len(copied_cells),
        "derived_seal_path": _rel(seal_path),
        "artifact_index": {
            "cells": copied_cells,
            "derived_seal_manifest": "derived_seal_manifest.json",
            "derived_seal_summary": "derived_seal_summary.json",
            "robustness_finding": (
                "robustness_finding.json"
                if (run_dir.path / "robustness_finding.json").is_file()
                else None
            ),
        },
    }
    return extra


def _dry_validate_emit_construction(
    *,
    session_id: str,
    seal_path: Path,
    sealed_manifest: dict[str, Any],
    sealed_summary: dict[str, Any],
    placement: dict[str, Any],
    allow_dirty: bool,
    model_spec: str | Path | None = None,
) -> dict[str, Any]:
    """Construct and schema-validate the promote manifest without writing raw/.

    Uses synthetic empty tier-1 evidence so Cursor-open promote can still prove the
    emit payload would validate. Live ``emit()`` always re-resolves isolation and will
    refuse while tier-1 is resident.
    """
    from seam.blinding import blinded_label_for, get_or_create_salt
    from seam.config import resolve_config
    from seam.gitinfo import capture_git_state
    from seam.manifest import build_manifest, validate_manifest
    from seam.model_provenance import load_local_spec, manifest_model_block

    matrix = sealed_manifest.get("matrix") or {}
    resolved = resolve_config(
        [ROOT / _PLATFORM_PATH, ROOT / _MEASUREMENT_PATH, ROOT / _DELTA_N_PATH],
        repo_root=ROOT,
    )
    model_spec_path = _resolve_model_spec_for_session(session_id, cli_spec=model_spec)
    spec = load_local_spec(model_spec_path)
    model_block = manifest_model_block(
        spec=spec,
        spec_path=model_spec_path,
        reasoning_mode="thinking_off",
    )
    verification = spec.get("verification") or {}
    file_methods = {
        name: str(entry.get("method"))
        for name, entry in verification.items()
        if isinstance(entry, dict)
        and entry.get("method") in {"sha256", "git-blob-sha1", "size-only"}
    }
    if file_methods:
        model_block["provenance"]["file_verification"] = file_methods

    workload = {
        "kind": "delta_prefill_matrix",
        "benchmark": "delta_prefill_resident_vs_nonresident_matrix",
        "task_ids": _task_ids_from_sealed(sealed_manifest),
        "seed": matrix.get("randomization_seed"),
        "n_repeats": matrix.get("repeats"),
        "concurrency": 1,
    }
    condition_label = (
        f"delta_prefill_matrix|nc={matrix.get('n_cached')}|"
        f"arms={','.join(matrix.get('arms') or [])}|promote_from_derived"
    )
    git_state = capture_git_state(cwd=ROOT)
    salt = get_or_create_salt(ROOT)
    blinded = blinded_label_for(condition_label, salt=salt)
    summary = _build_promotion_summary(
        session_id=session_id,
        seal_path=seal_path,
        sealed_manifest=sealed_manifest,
        sealed_summary=sealed_summary,
        placement=placement,
    )
    # Synthetic quiet evidence for dry construction only — never written to raw/.
    dry_evidence = {
        "contending_processes": [],
        "tier2_recorded_processes": [],
        "sshd_session_count": None,
        "consistent_with_declaration": True,
        "probe_error": None,
    }
    drivers = {
        "npu": None,
        "igpu": None,
        "openvino": placement.get("openvino"),
        "genai": placement.get("genai"),
        "lhm_bridge": None,
    }
    # Schema-validate with the same allow_dirty bit emit will record. Dirty-tree
    # refusal is enforced inside emit(), not build_manifest().
    manifest = build_manifest(
        run_id=session_id,
        config=resolved,
        git_state=git_state,
        allow_dirty=allow_dirty,
        target="cpu-placement",
        workload=workload,
        condition_label=condition_label,
        blinded_label=blinded,
        repo_root=ROOT,
        model=model_block,
        drivers=drivers,
        power_state=None,
        power_state_note=(
            "Post-hoc promote: measurement power_state left null (AM-036). "
            "Per-cell environment_* blocks remain in cell JSON under cells/; "
            "do not treat promote-time host state as run environment."
        ),
        retro_seal=True,
        thermal={"regime": "confound", "excluded": False},
        raw_sha256="0" * 64,
        self_check="pass",
        isolation_mode="remote",
        isolation_evidence=dry_evidence,
        launch_context=str(placement.get("launch_context") or "ssh_detached"),
        session_id=placement.get("session_id"),
        window_station=placement.get("window_station"),
        require_launch_context=True,
    )
    validate_manifest(manifest)

    staging = {
        "cells_glob": "cells/delta_prefill_*.json",
        "cells_n": len(list((seal_path / "cells").glob("delta_prefill_*.json"))),
        "copy_if_present": [
            "session_summary_as_found.json",
            "session_plan_as_found.json",
            "launch_log_failed_cell_excerpt.txt",
            "robustness_finding.json",
        ],
        "write_json": [
            "derived_seal_manifest.json",
            "derived_seal_summary.json",
            "summary.json",
        ],
        "robustness_finding_included": bool(summary.get("robustness_finding")),
    }
    dirty_blocker = bool(git_state.dirty and not allow_dirty)
    return {
        "ok": True,
        "workload": workload,
        "condition_label": condition_label,
        "target": "cpu-placement",
        "model_name": model_block.get("name"),
        "model_ir_sha256": model_block.get("ir_sha256"),
        "config_hash": resolved.config_hash,
        "git_dirty": git_state.dirty,
        "allow_dirty": allow_dirty,
        "emit_will_need_allow_dirty": dirty_blocker,
        "launch_context": placement.get("launch_context"),
        "measurement_session_id": placement.get("session_id"),
        "measurement_window_station": placement.get("window_station"),
        "staging_plan": staging,
        "summary_keys": sorted(summary.keys()),
        "note": (
            "Dry construction validated against run_manifest.schema.json with synthetic "
            "empty tier-1 isolation_evidence. Live emit() re-resolves isolation_mode=remote "
            "and will refuse while Cursor/Chrome are resident."
            + (
                " Git tree is dirty: pass --allow-dirty when isolation clears or emit "
                "will refuse DirtyTreeError."
                if dirty_blocker
                else ""
            )
        ),
    }


def _emit_raw_from_seal(
    *,
    session_id: str,
    seal_path: Path,
    sealed_manifest: dict[str, Any],
    sealed_summary: dict[str, Any],
    placement: dict[str, Any],
    allow_dirty: bool,
    model_spec: str | Path | None = None,
) -> dict[str, Any]:
    """Call seam.manifest.emit; isolation_mode=remote is enforced inside emit()."""
    from seam.config import resolve_config
    from seam.manifest import emit
    from seam.model_provenance import load_local_spec, manifest_model_block

    raw_dir = ROOT / "raw" / session_id
    if raw_dir.exists():
        raise SystemExit(f"refusing emit: raw/{session_id} already exists (write-once)")

    matrix = sealed_manifest.get("matrix") or {}
    resolved = resolve_config(
        [ROOT / _PLATFORM_PATH, ROOT / _MEASUREMENT_PATH, ROOT / _DELTA_N_PATH],
        repo_root=ROOT,
    )
    model_spec_path = _resolve_model_spec_for_session(session_id, cli_spec=model_spec)
    spec = load_local_spec(model_spec_path)
    model_block = manifest_model_block(
        spec=spec,
        spec_path=model_spec_path,
        reasoning_mode="thinking_off",
    )
    verification = spec.get("verification") or {}
    file_methods = {
        name: str(entry.get("method"))
        for name, entry in verification.items()
        if isinstance(entry, dict)
        and entry.get("method") in {"sha256", "git-blob-sha1", "size-only"}
    }
    if file_methods:
        model_block["provenance"]["file_verification"] = file_methods

    workload = {
        "kind": "delta_prefill_matrix",
        "benchmark": "delta_prefill_resident_vs_nonresident_matrix",
        "task_ids": _task_ids_from_sealed(sealed_manifest),
        "seed": matrix.get("randomization_seed"),
        "n_repeats": matrix.get("repeats"),
        "concurrency": 1,
    }
    condition_label = (
        f"delta_prefill_matrix|nc={matrix.get('n_cached')}|"
        f"arms={','.join(matrix.get('arms') or [])}|promote_from_derived"
    )
    summary = _build_promotion_summary(
        session_id=session_id,
        seal_path=seal_path,
        sealed_manifest=sealed_manifest,
        sealed_summary=sealed_summary,
        placement=placement,
    )
    drivers = {
        "npu": None,
        "igpu": None,
        "openvino": placement.get("openvino"),
        "genai": placement.get("genai"),
        "lhm_bridge": None,
    }

    def _before(run_dir: Any) -> dict[str, Any]:
        return _stage_raw_artifacts(
            run_dir,
            seal_path=seal_path,
            sealed_manifest=sealed_manifest,
            sealed_summary=sealed_summary,
        )

    # AM-036: retro-seal must not populate measurement power_state from promote-time
    # sampling. Per-cell environment_* blocks remain under cells/; measurement power
    # stays explicitly null with a provenance note.
    handle = emit(
        config=resolved,
        target="cpu-placement",
        workload=workload,
        condition_label=condition_label,
        repo_root=ROOT,
        run_id=session_id,
        allow_dirty=allow_dirty,
        summary=summary,
        model=model_block,
        drivers=drivers,
        power_state=None,
        power_state_note=(
            "Post-hoc promote: measurement power_state left null (AM-036). "
            "Per-cell environment_* blocks remain in cell JSON under cells/; "
            "do not treat promote-time host state as run environment."
        ),
        retro_seal=True,
        thermal={"regime": "confound", "excluded": False},
        self_check="pass",
        isolation_mode="remote",
        launch_context=str(placement.get("launch_context") or "ssh_detached"),
        session_id=placement.get("session_id"),
        window_station=placement.get("window_station"),
        require_launch_context=True,
        before_integrity_hash=_before,
    )
    return {
        "run_id": handle.run_id,
        "raw_path": _rel(handle.run_dir.path),
        "raw_sha256": handle.raw_sha256,
        "manifest_path": _rel(handle.run_dir.path / "manifest.json"),
        "robustness_finding_in_summary": bool(summary.get("robustness_finding")),
    }


def _attempt_raw_promote(
    session_id: str,
    seal_path: Path,
    *,
    allow_dirty: bool = False,
    model_spec: str | Path | None = None,
) -> dict[str, Any]:
    """Promote a derived_diagnostic seal into raw/ via seam.manifest.emit.

    Order: isolation_mode=remote → schema kind gate → dry-validate construction →
    emit (only if isolation cleared). Never bypasses enforce. Never mutates sealed_*.
    """
    from seam.isolation import IsolationModeError, gather_isolation_evidence, resolve_isolation_mode
    from seam.manifest import load_schema

    attempted_utc = datetime.now(UTC).isoformat()
    if not seal_path.is_dir():
        raise SystemExit(f"promote: missing seal dir {seal_path}")
    sealed_manifest_path = seal_path / "manifest.json"
    sealed_summary_path = seal_path / "summary.json"
    if not sealed_manifest_path.is_file() or not sealed_summary_path.is_file():
        raise SystemExit(f"promote: sealed manifest/summary missing under {seal_path}")

    sealed_manifest = _read_json(sealed_manifest_path)
    sealed_summary = _read_json(sealed_summary_path)
    placement = _measurement_placement(seal_path)

    evidence = gather_isolation_evidence()
    tier1 = list(evidence.get("contending_processes") or [])
    record: dict[str, Any] = {
        "attempted_utc": attempted_utc,
        "session_id": session_id,
        "seal_path": _rel(seal_path),
        "raw_emit": False,
        "gate": (
            "resolve_isolation_mode(remote) → schema kind → dry_validate → " "seam.manifest.emit"
        ),
        "tier1_resident_names": sorted({p.get("name") for p in tier1 if p.get("name")}),
        "tier1_count": len(tier1),
        "emit_wired": True,
    }

    # Schema gate is independent of isolation; check even when isolation refuses so
    # PROMOTION_ATTEMPT records a complete residual picture.
    kind_enum = load_schema()["properties"]["workload"]["properties"]["kind"]["enum"]
    if "delta_prefill_matrix" not in kind_enum:
        record["status"] = "refused_schema_gate"
        record["schema_gate"] = "workload.kind delta_prefill_matrix missing from enum"
        record["note"] = (
            "workload.kind delta_prefill_matrix is not in "
            "seam/schemas/run_manifest.schema.json; add it before raw/ promotion."
        )
        return record
    record["schema_gate"] = "workload.kind delta_prefill_matrix accepted"

    try:
        dry = _dry_validate_emit_construction(
            session_id=session_id,
            seal_path=seal_path,
            sealed_manifest=sealed_manifest,
            sealed_summary=sealed_summary,
            placement=placement,
            allow_dirty=allow_dirty,
            model_spec=model_spec,
        )
        record["dry_validate"] = dry
    except Exception as exc:
        record["dry_validate"] = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
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
            "--attempt-raw-promote. Do not invent enforce=False. "
            "Emit wiring is present; dry_validate records whether the promote payload "
            "would schema-validate without writing raw/."
        )
        return record

    if not (record.get("dry_validate") or {}).get("ok"):
        record["status"] = "refused_dry_validate"
        record["note"] = (
            "Isolation gate passed but dry emit construction failed; refusing raw/ write. "
            "See dry_validate error."
        )
        return record

    try:
        emitted = _emit_raw_from_seal(
            session_id=session_id,
            seal_path=seal_path,
            sealed_manifest=sealed_manifest,
            sealed_summary=sealed_summary,
            placement=placement,
            allow_dirty=allow_dirty,
            model_spec=model_spec,
        )
    except Exception as exc:
        record["status"] = "refused_emit"
        record["emit_error_type"] = type(exc).__name__
        record["emit_error"] = str(exc)
        record["note"] = (
            "isolation + schema + dry_validate passed but seam.manifest.emit failed. "
            "raw/ was not sealed; inspect emit_error. Do not invent enforce=False."
        )
        return record

    record["status"] = "raw_emitted"
    record["raw_emit"] = True
    record["emit"] = emitted
    record["note"] = (
        f"seam.manifest.emit wrote sealed raw/{session_id}/ from derived seal "
        f"{_rel(seal_path)} (sealed_* tree not mutated)."
    )
    return record


def seal_session(
    session_id: str,
    *,
    run_id: str | None = None,
    model_spec: str | Path | None = None,
    model_specs: list[str] | None = None,
) -> dict[str, Any]:
    run_id = run_id or session_id
    session_dir = ROOT / "derived" / "delta_prefill" / session_id
    if not session_dir.is_dir():
        raise SystemExit(f"missing session dir {session_dir}")

    summary_path = session_dir / "summary.json"
    plan_path = session_dir / "plan.json"
    if not summary_path.is_file():
        raise SystemExit(f"missing {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    plan = json.loads(plan_path.read_text(encoding="utf-8-sig")) if plan_path.is_file() else {}

    if summary.get("status") != "complete":
        raise SystemExit(f"refusing to seal: status={summary.get('status')!r} (want complete)")

    cells = _unwrap_ps_list(summary.get("cells"))
    expected_cells = len(cells)
    if expected_cells < 1:
        raise SystemExit("refusing to seal: summary has no cells")

    # Resolve cell JSON from per-cell artifact paths (tag-based names), not a
    # hardcoded delta_prefill_*.json glob — W-2 artifacts are w2_weight_precision_*.
    cell_pairs: list[tuple[Path, dict[str, Any]]] = []
    for cell_meta in cells:
        art = cell_meta.get("artifact")
        if not art:
            raise SystemExit(
                f"refusing to seal: cell_index={cell_meta.get('cell_index')} missing artifact path"
            )
        src = Path(str(art))
        if not src.is_file():
            raise SystemExit(f"refusing to seal: missing cell artifact {src}")
        cell_pairs.append((src, cell_meta))
    if len(cell_pairs) != expected_cells:
        raise SystemExit(f"expected {expected_cells} cell JSON files, found {len(cell_pairs)}")

    model_specs_block = _resolve_model_spec_set_for_session(
        session_id,
        cells=cells,
        cli_spec=model_spec,
        cli_specs=model_specs,
    )

    durations = _cell_durations(cells, citing_session_id=session_id)
    cells_ok = sum(1 for c in cells if c.get("classification") == "OK")
    cells_failed = expected_cells - cells_ok

    robustness: dict[str, Any] | None = None
    if session_id == ROBUSTNESS_SESSION_ID:
        if not ROBUSTNESS_LAUNCH_LOG.is_file():
            raise SystemExit(f"missing launch log {ROBUSTNESS_LAUNCH_LOG}")
        log_text = ROBUSTNESS_LAUNCH_LOG.read_text(encoding="utf-8", errors="replace")
        if "errcode -5,CL_OUT_OF_RESOURCES" not in log_text:
            raise SystemExit("launch log missing CL_OUT_OF_RESOURCES / errcode -5 line")
        robustness = _robustness_finding(summary, durations)

    out = ROOT / "derived" / "delta_prefill" / f"sealed_{run_id}"
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing seal dir {out}")
    out.mkdir(parents=True, exist_ok=False)
    cells_dir = out / "cells"
    cells_dir.mkdir()

    manifest_cells: list[dict[str, Any]] = []
    summary_cells: list[dict[str, Any]] = []
    for src, cell_meta in cell_pairs:
        dest = cells_dir / src.name
        shutil.copy2(src, dest)
        sha = _sha256_file(dest)
        rec = json.loads(src.read_text(encoding="utf-8-sig"))
        entry = {
            "arm_id": rec.get("arm_id") or cell_meta.get("arm"),
            "mode": rec.get("mode"),
            "n_cached": rec.get("n_cached"),
            "delta": rec.get("delta"),
            "repeat_index": cell_meta.get("repeat"),
            "model_spec": cell_meta.get("model_spec")
            or (rec.get("diagnostics") or {}).get("model_spec"),
            "ir_sha256": cell_meta.get("ir_sha256")
            or (rec.get("diagnostics") or {}).get("ir_sha256"),
            "classification": rec.get("classification"),
            "artifact": f"cells/{src.name}",
            "artifact_sha256": sha,
            "process_id": rec.get("process_id"),
            "timestamp_utc": rec.get("timestamp_utc"),
            "execute_ok": rec.get("execute_ok"),
            "compile_ok": rec.get("compile_ok"),
            "turn1_prefill_s": (rec.get("turn1") or {}).get("prefill_s"),
            "turn2_prefill_s": (rec.get("turn2") or {}).get("prefill_s"),
            "exit_code": cell_meta.get("exit_code"),
            "elapsed_s": (
                round(
                    _elapsed_s(cell_meta["started_utc"], cell_meta["ended_utc"]),
                    3,
                )
                if cell_meta.get("started_utc") and cell_meta.get("ended_utc")
                else None
            ),
        }
        manifest_cells.append(entry)
        summary_cells.append(
            {
                "arm_id": entry["arm_id"],
                "mode": entry["mode"],
                "n_cached": entry["n_cached"],
                "delta": entry["delta"],
                "repeat_index": entry["repeat_index"],
                "model_spec": entry["model_spec"],
                "ir_sha256": entry["ir_sha256"],
                "classification": entry["classification"],
                "turn1_prefill_s": entry["turn1_prefill_s"],
                "turn2_prefill_s": entry["turn2_prefill_s"],
                "exit_code": entry["exit_code"],
                "elapsed_s": entry["elapsed_s"],
            }
        )

    shutil.copy2(summary_path, out / "session_summary_as_found.json")
    if plan_path.is_file():
        shutil.copy2(plan_path, out / "session_plan_as_found.json")

    if robustness is not None:
        log_text = ROBUSTNESS_LAUNCH_LOG.read_text(encoding="utf-8", errors="replace")
        lines = log_text.splitlines()
        start = end = None
        for i, line in enumerate(lines):
            if "arm=gpu_only mode=RESIDENT delta=500 r=1" in line:
                start = i
            if start is not None and "arm=gpu_only mode=NON_RESIDENT delta=2000 r=1" in line:
                end = i
                break
        if start is None:
            raise SystemExit("could not locate failed-cell block in launch log")
        excerpt = "\n".join(lines[start : (end if end is not None else start + 20)]) + "\n"
        (out / "launch_log_failed_cell_excerpt.txt").write_text(excerpt, encoding="utf-8")
        (out / "robustness_finding.json").write_text(
            json.dumps(robustness, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    sealed_utc = datetime.now(UTC).isoformat()
    raw_emit_blocked = {
        "reason": (
            "seam.manifest.emit refuses isolation_mode=remote while tier-1 software "
            "(Cursor/Chrome) is resident at seal time. Measurement itself was "
            "ssh_detached/remote. Close Cursor/Chrome and promote with "
            "--attempt-raw-promote; do not invent an enforce=False bypass."
        ),
        "existing_pattern": "derived/gpu_smoke/_seal_partial_matrix.py → seal_style=derived_diagnostic",
        "schema_note": (
            "workload.kind delta_prefill_matrix is registered in "
            "seam/schemas/run_manifest.schema.json; --attempt-raw-promote calls "
            "seam.manifest.emit after isolation clears (sealed_* never mutated)."
        ),
    }

    admissibility: dict[str, Any]
    if robustness is not None:
        admissibility = {
            "matrix_status": "COMPLETE",
            "failed_cells_are_data": True,
            "robustness_finding_recorded": True,
            "note": (
                "One cell failed (gpu_only RESIDENT nc12000 d500 r1). Failure is retained "
                "as a robustness finding (CL_OUT_OF_RESOURCES + post-record hung child + "
                "operator kill), not dropped. OK-cell numbers cite this run_id / seal dir."
            ),
        }
    else:
        admissibility = {
            "matrix_status": "COMPLETE",
            "failed_cells_are_data": False,
            "robustness_finding_recorded": False,
            "note": (
                f"All {cells_ok}/{expected_cells} cells OK. Numbers cite this run_id / seal dir."
            ),
        }

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "status": "COMPLETE",
        "kind": "delta_prefill_matrix",
        "seal_style": "derived_diagnostic",
        "sealed_utc": sealed_utc,
        "session_id": session_id,
        "measurement_isolation_mode": "remote",
        "measurement_launch_context": summary.get("launch_context") or plan.get("launch_context"),
        "raw_emit_blocked": raw_emit_blocked,
        "matrix": {
            "tag": summary.get("tag"),
            "launch_context": summary.get("launch_context"),
            "arms": summary.get("arms"),
            "n_cached": summary.get("n_cached"),
            "deltas": summary.get("deltas"),
            "repeats": summary.get("repeats"),
            "model_specs": model_specs_block,
            "canary_model_spec": (summary.get("canary") or plan.get("canary") or {}).get(
                "model_spec"
            )
            or summary.get("canary_model_spec")
            or plan.get("canary_model_spec"),
            "cell_specs_per_round": summary.get("cell_specs_per_round"),
            "cells_completed": len(manifest_cells),
            "cells_expected": expected_cells,
            "cells_ok": cells_ok,
            "cells_failed": cells_failed,
            "plan_started_utc": summary.get("started_utc"),
            "plan_ended_utc": summary.get("ended_utc"),
            "pre_run_settle_s": summary.get("pre_run_settle_s"),
            "inter_cell_settle_s": summary.get("inter_cell_settle_s"),
            "randomization_seed": summary.get("randomization_seed"),
            "detached_worker": summary.get("detached_worker"),
            "cell_timeout_s": summary.get("cell_timeout_s"),
            "cell_timeout_derivation": summary.get("cell_timeout_derivation"),
        },
        "model_specs": model_specs_block,
        "admissibility": admissibility,
        "cell_durations": durations,
        "cells": manifest_cells,
    }
    if robustness is not None:
        manifest["robustness_finding"] = robustness

    sealed_summary: dict[str, Any] = {
        "run_id": run_id,
        "status": "COMPLETE",
        "kind": "delta_prefill_matrix",
        "seal_style": "derived_diagnostic",
        "session_id": session_id,
        "model_specs": model_specs_block,
        "cells_completed": len(summary_cells),
        "cells_ok": cells_ok,
        "cells_failed": cells_failed,
        "cells": summary_cells,
    }
    if robustness is not None:
        sealed_summary["robustness_finding_class"] = robustness["class"]
        sealed_summary["onednn_stderr_verbatim"] = ONEDNN_CL_OUT_OF_RESOURCES_LINE
        sealed_summary["operator_kill_narrative"] = robustness["operator_kill"]["narrative"]
        sealed_summary["cell_timeout_derivation"] = durations.get("cell_timeout_derivation")

    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out / "summary.json").write_text(
        json.dumps(sealed_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    tree_hash = _sha256_tree(out, exclude={".sealed"})
    seal_marker = {
        "run_id": run_id,
        "sealed_at_utc": sealed_utc,
        "seal_style": "derived_diagnostic",
        "tree_sha256": tree_hash,
        "self_check": "pass",
        "note": (
            "Advisory marker for derived_diagnostic seals. Not seam.rawstore.verify_sealed; "
            "raw/ was not written. Do not mutate this directory after seal."
        ),
    }
    (out / ".sealed").write_text(
        json.dumps(seal_marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    for p in [out / ".sealed", *out.rglob("*")]:
        if p.is_file():
            try:
                p.chmod(p.stat().st_mode & ~0o222)
            except OSError:
                pass

    pointer = {
        "session_id": session_id,
        "run_id": run_id,
        "sealed": True,
        "seal_style": "derived_diagnostic",
        "seal_path": str(out.relative_to(ROOT)).replace("\\", "/"),
        "raw_emit": False,
        "raw_emit_blocked": raw_emit_blocked["reason"],
        "sealed_utc": sealed_utc,
        "tree_sha256": tree_hash,
        "cells_ok": cells_ok,
        "cells_failed": cells_failed,
    }
    if robustness is not None:
        pointer["robustness_finding"] = (
            "gpu_only RESIDENT nc12000 d500 r1 CL_OUT_OF_RESOURCES hung-child operator-kill"
        )
    pointer_path = session_dir / "SEAL_POINTER.json"
    pointer_path.write_text(json.dumps(pointer, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "run_id": run_id,
        "sealed": True,
        "seal_style": "derived_diagnostic",
        "seal_path": str(out.resolve()),
        "pointer": str(pointer_path.resolve()),
        "tree_sha256": tree_hash,
        "raw_emit": False,
        "cells_ok": cells_ok,
        "cells_failed": cells_failed,
        "robustness_finding_class": (robustness or {}).get("class"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    parser.add_argument(
        "--run-id",
        default=None,
        help="defaults to --session-id (session UUID is a valid uuid4)",
    )
    parser.add_argument(
        "--attempt-raw-promote",
        action="store_true",
        help=(
            "After isolation_mode=remote clears, call seam.manifest.emit into raw/ "
            "from the sealed derived tree. Does not mutate sealed_*. "
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
        help=(
            "Pass allow_dirty through to seam.manifest.emit (recorded in the raw "
            "manifest). Required when the git tree is dirty at promote time."
        ),
    )
    parser.add_argument(
        "--model-spec",
        action="append",
        dest="model_specs",
        default=None,
        help=(
            "FetchedModelSpec YAML for the seal. Repeatable / comma-separated for a "
            "SET (W-2). Default: every distinct model_spec recorded on the session "
            "plan. Fail only if a cell's model is outside the declared set."
        ),
    )
    args = parser.parse_args(argv)

    session_id = args.session_id
    run_id = args.run_id or session_id
    seal_path = ROOT / "derived" / "delta_prefill" / f"sealed_{run_id}"
    result: dict[str, Any]

    # Flatten comma-separated --model-spec values into a list of paths.
    cli_specs: list[str] | None = None
    if args.model_specs:
        cli_specs = []
        for item in args.model_specs:
            cli_specs.extend(p.strip() for p in str(item).split(",") if p.strip())

    if args.promote_only:
        if not seal_path.is_dir():
            raise SystemExit(f"promote-only: missing seal dir {seal_path}")
        if not args.attempt_raw_promote:
            raise SystemExit("promote-only requires --attempt-raw-promote")
        result = {
            "run_id": run_id,
            "sealed": True,
            "seal_style": "derived_diagnostic",
            "seal_path": str(seal_path.resolve()),
            "raw_emit": False,
            "promote_only": True,
        }
    else:
        result = seal_session(
            session_id,
            run_id=run_id,
            model_specs=cli_specs,
        )
        seal_path = Path(result["seal_path"])

    if args.attempt_raw_promote:
        # Promote still uses a single model block today; pass the first CLI
        # path if given, else None (single-spec sessions only until promote
        # schema grows a models[] array).
        promo_spec = cli_specs[0] if cli_specs and len(cli_specs) == 1 else None
        if cli_specs and len(cli_specs) > 1:
            raise SystemExit(
                "FATAL: --attempt-raw-promote cannot yet emit a multi-weight "
                f"model set {cli_specs}; derived seal records the set. Promote "
                "schema needs a models[] block before raw/ emit for W-2."
            )
        promo = _attempt_raw_promote(
            session_id,
            seal_path,
            allow_dirty=bool(args.allow_dirty),
            model_spec=promo_spec,
        )
        session_dir = ROOT / "derived" / "delta_prefill" / session_id
        promo_path = session_dir / "PROMOTION_ATTEMPT.json"
        promo_path.write_text(json.dumps(promo, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        # Update pointer fields without touching sealed_* (write-once).
        pointer_path = session_dir / "SEAL_POINTER.json"
        if pointer_path.is_file():
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            emitted = promo.get("status") == "raw_emitted"
            pointer["raw_emit"] = bool(emitted)
            pointer["promotion_attempt"] = {
                "attempted_utc": promo["attempted_utc"],
                "status": promo["status"],
                "record": _rel(promo_path),
                "emit_wired": True,
            }
            if emitted:
                pointer["raw_path"] = (promo.get("emit") or {}).get("raw_path")
                pointer.pop("raw_emit_blocked", None)
            else:
                pointer["raw_emit_blocked"] = promo.get("note") or promo.get("refusal")
            pointer_path.write_text(
                json.dumps(pointer, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
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
