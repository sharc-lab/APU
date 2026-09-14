"""Prompt A2 run-lifecycle helpers: dry-run, early raw dir, per-block JSONL, cycle guard."""

from __future__ import annotations

import json
import shutil
import statistics
import subprocess
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from seam.analysis.prompt_a import BLOCKS_FILENAME, IN_PROGRESS_FILENAME
from seam.config import resolve_config
from seam.errors import SeamError
from seam.gitinfo import capture_git_state
from seam.locks import exclusive
from seam.manifest import emit
from seam.powerstate import capture_power_state, manifest_power_state
from seam.rawstore import RunDir, create_run_dir, verify_sealed

__all__ = [
    "append_block_jsonl",
    "assert_acyclic",
    "block_jsonl_record",
    "build_flat_block_record",
    "fail_dry_run_on_cyclic_summary",
    "find_cycles",
    "open_in_progress_run",
    "progress_status",
    "reproduce_alias_cycle_paths",
    "startup_output_path_dry_run",
    "synthetic_block_record",
    "synthetic_measurement_record",
    "synthetic_placement_summary",
]


def find_cycles(obj: Any, path: str = "$", ancestors: frozenset[int] = frozenset()) -> list[str]:
    """Report paths where a container references one of its own ancestors."""
    if not isinstance(obj, (dict, list, tuple)):
        return []
    oid = id(obj)
    if oid in ancestors:
        return [path]
    next_ancestors = ancestors | {oid}
    found: list[str] = []
    items = obj.items() if isinstance(obj, dict) else enumerate(obj)
    for key, value in items:
        subpath = f"{path}.{key}" if isinstance(obj, dict) else f"{path}[{key}]"
        found.extend(find_cycles(value, subpath, next_ancestors))
    return found


def assert_acyclic(obj: Any, *, label: str) -> None:
    """Fail fast on circular references; never fall back to default=str/repr."""
    cycles = find_cycles(obj)
    if cycles:
        raise SeamError(
            f"{label} has circular reference(s) at: {cycles}. "
            "Serialization refuses default=str/repr; fix the alias/reference."
        )


def progress_status(
    *,
    cell_id: str,
    cell_ids: list[str],
    repeat_index: int,
    repeats_per_cell: int,
) -> dict[str, Any]:
    """Status fields: cell N of 4, repeat M of 7 (1-based display indices)."""
    n_cells = len(cell_ids)
    cell_index = cell_ids.index(cell_id) + 1
    repeat_m = repeat_index + 1
    return {
        "cell_id": cell_id,
        "cell_progress": f"cell {cell_index} of {n_cells}",
        "repeat_progress": f"repeat {repeat_m} of {repeats_per_cell}",
        "cell_index_1based": cell_index,
        "repeat_index_1based": repeat_m,
        "n_cells": n_cells,
        "repeats_per_cell": repeats_per_cell,
    }


def _cited(run_id: str, value: Any) -> dict[str, Any]:
    """AM-027(b): every number carries run_id at the point of use."""
    return {"run_id": run_id, "value": value}


def _canary_median_ns(canary: dict[str, Any] | None) -> int | None:
    if not canary or not canary.get("workers"):
        return None
    return round(statistics.median(float(worker["runtime_ns"]) for worker in canary["workers"]))


def _matches_request(
    affinity_requested: list[int] | None, readback: list[int] | None
) -> bool | None:
    if affinity_requested is None:
        return None
    if readback is None:
        return False
    return list(readback) == list(affinity_requested)


def build_flat_block_record(
    *,
    run_id: str,
    block_id: str,
    cell_id: str,
    repeat_index: int,
    measurement: dict[str, Any],
    placement: dict[str, Any],
    canary: dict[str, Any],
    telemetry: dict[str, Any],
    admissible: bool,
    reasons: list[str],
    timestamp_utc: str,
    discarded_warmup: bool = False,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One block record: siblings, never measurement⊂validity or validity⊂measurement."""
    record = {
        "block_id": block_id,
        "cell_id": cell_id,
        "repeat_index": repeat_index,
        "measurement": measurement,
        "placement": placement,
        "canary": canary,
        "telemetry": telemetry,
        "validity": {
            "admissible": bool(admissible),
            "reasons": list(reasons),
            "block_id": block_id,
        },
        "run_id": run_id,
        "timestamp_utc": timestamp_utc,
        "discarded_warmup": bool(discarded_warmup),
    }
    if extras:
        for key, value in extras.items():
            if key in record:
                raise SeamError(f"flat block extras cannot overwrite sibling {key!r}")
            record[key] = value
    # Guard the class of bug: validity must reference block_id and never embed measurement.
    validity_obj = record["validity"]
    if not isinstance(validity_obj, dict):
        raise SeamError("validity must be a mapping")
    if "timed_measurement" in validity_obj:
        raise SeamError("validity must not embed timed_measurement")
    if "measurement" in validity_obj:
        raise SeamError("validity must not embed measurement")
    assert_acyclic(record, label=f"flat block record block_id={block_id}")
    return record


def block_jsonl_record(*, run_id: str, measurement: dict[str, Any]) -> dict[str, Any]:
    """Normalize a flat block (or legacy nested measurement) into the JSONL sibling schema.

    Production callers pass an already-flat block from ``build_flat_block_record``. Legacy nested
    shapes (``measurement['validity']`` holding the machine envelope) are accepted only so sealed
    historical fixtures and cycle-reproduction helpers keep working.
    """
    if (
        "validity" in measurement
        and isinstance(measurement["validity"], dict)
        and "admissible" in measurement["validity"]
        and "placement" in measurement
        and "canary" in measurement
        and "telemetry" in measurement
        and "measurement" in measurement
    ):
        # Already flat. Ensure run_id citation and return a shallow copy.
        record = dict(measurement)
        record["run_id"] = run_id
        if record.get("block_id") is None:
            record["block_id"] = str(uuid.uuid4())
        validity = dict(record["validity"])
        validity["block_id"] = record["block_id"]
        # Never re-introduce mutual containment.
        validity.pop("timed_measurement", None)
        validity.pop("measurement", None)
        record["validity"] = validity
        assert_acyclic(record, label=f"blocks.jsonl flat record run_id={run_id}")
        return record

    # Legacy nested shape → flatten (used by older synthetic helpers / regression fixtures).
    validity_src = measurement["validity"]
    placement_src = measurement["placement"]
    block_id = str(measurement.get("block_id") or uuid.uuid4())
    affinity_requested = measurement.get("requested_affinity")
    after_spinup = placement_src["after_worker_threads_spin_up"]
    mid = placement_src["mid_generation"]
    spinup_mask = after_spinup.get("process_affinity_readback")
    before_ns = _canary_median_ns(validity_src.get("canary_pre"))
    after_ns = _canary_median_ns(validity_src.get("canary_post"))
    drift = validity_src.get("canary_relative_drift")
    drift_pct = None if drift is None else float(drift) * 100.0
    threshold = float(validity_src.get("canary_drift_threshold", 0.15))
    canary_admissible = (
        drift is not None
        and float(drift) <= threshold
        and bool(validity_src.get("canary_pre", {}).get("affinity_passed", True))
        and bool(validity_src.get("canary_post", {}).get("affinity_passed", True))
    )
    page_rates = validity_src.get("hard_page_reads_per_s") or []
    page_reads_per_sec = None
    if page_rates:
        numeric = [float(rate) for rate in page_rates if rate is not None]
        page_reads_per_sec = statistics.fmean(numeric) if numeric else None
    admissible = bool(validity_src.get("valid", validity_src.get("admissible", False)))
    reasons = list(validity_src.get("invalid_reasons") or validity_src.get("reasons") or [])
    return build_flat_block_record(
        run_id=run_id,
        block_id=block_id,
        cell_id=str(measurement["cell_id"]),
        repeat_index=int(measurement["repeat_index"]),
        measurement={
            "R_prefill": _cited(run_id, measurement.get("r_prefill_tok_s")),
            "R_decode": _cited(run_id, measurement.get("r_decode_tok_s")),
            "wall_ns": _cited(run_id, round(float(measurement["wall_s"]) * 1e9)),
            "ttft_ns": _cited(run_id, round(float(measurement["ttft_s"]) * 1e9)),
            "r_prefill_tok_s": measurement.get("r_prefill_tok_s"),
            "r_decode_tok_s": measurement.get("r_decode_tok_s"),
            "prompt_tokens": measurement.get("prompt_tokens"),
            "completion_tokens": measurement.get("completion_tokens"),
            "ttft_source": measurement.get("ttft_source"),
            "frequency_sample_count": measurement.get("frequency_sample_count"),
        },
        placement={
            "affinity_requested": affinity_requested,
            "readback_after_spinup": {
                "process_affinity_mask": spinup_mask,
                "per_thread": after_spinup.get("per_thread_cpu_placement")
                or {
                    "available": False,
                    "reason": "unavailable",
                },
                "captured_utc": after_spinup.get("captured_utc"),
                "generation_active": after_spinup.get("generation_active"),
            },
            "readback_mid_generation": {
                "process_affinity_mask": mid.get("process_affinity_readback"),
                "per_thread": mid.get("per_thread_cpu_placement")
                or {
                    "available": False,
                    "reason": "unavailable",
                },
                "captured_utc": mid.get("captured_utc"),
                "generation_active": mid.get("generation_active"),
            },
            "matches_request": _matches_request(affinity_requested, spinup_mask),
        },
        canary={
            "before_ns": _cited(run_id, before_ns),
            "after_ns": _cited(run_id, after_ns),
            "drift_pct": _cited(run_id, drift_pct),
            "admissible": canary_admissible,
            "threshold": threshold,
        },
        telemetry={
            "memory": {
                "available_memory_mb_before": _cited(
                    run_id, validity_src.get("available_memory_mb_before")
                ),
                "available_memory_mb_after": _cited(
                    run_id, validity_src.get("available_memory_mb_after")
                ),
            },
            "paging": {
                "page_reads_per_sec": _cited(run_id, page_reads_per_sec),
                "hard_page_reads_samples": _cited(run_id, page_rates),
            },
            "cpu": {
                "cpu_pct_total": _cited(run_id, validity_src.get("cpu_pct_total")),
                "cpu_pct_per_core": _cited(run_id, validity_src.get("cpu_pct_per_core")),
            },
            "package_temp": None,
            "lock_acquired_utc": validity_src.get("lock_acquired_utc"),
            "lock_released_utc": validity_src.get("lock_released_utc"),
        },
        admissible=admissible,
        reasons=reasons,
        timestamp_utc=str(
            validity_src.get("lock_released_utc")
            or validity_src.get("lock_acquired_utc")
            or measurement.get("timestamp_utc")
            or "1970-01-01T00:00:00+00:00"
        ),
        discarded_warmup=bool(measurement.get("discarded_warmup", False)),
    )


def append_block_jsonl(run_dir: RunDir, record: dict[str, Any]) -> Path:
    """Append one strict-JSON block line with flush+fsync (via rawstore.append_ndjson)."""
    assert_acyclic(record, label=f"blocks.jsonl record run_id={record.get('run_id')}")
    if "timed_measurement" in (record.get("validity") or {}):
        raise SeamError("refusing to append block whose validity embeds timed_measurement")
    return run_dir.append_ndjson(BLOCKS_FILENAME, record)


def open_in_progress_run(
    *,
    root: Path,
    run_id: str,
    marker: dict[str, Any],
) -> RunDir:
    """Create raw/<run_id>/ and write the in-progress marker before model load / first cell."""
    with exclusive(root / "raw"):
        run_dir = create_run_dir(run_id, repo_root=root)
        run_dir.write_json(IN_PROGRESS_FILENAME, marker)
    return run_dir


def _placement_stub(stage: str, *, affinity: list[int] | None) -> dict[str, Any]:
    return {
        "stage": stage,
        "captured_utc": "1970-01-01T00:00:00+00:00",
        "process_affinity_readback": affinity,
        "process_thread_count": 1,
        "generation_active": stage not in {"before_generation", "after_generation"},
        "per_thread_cpu_placement": {
            "available": False,
            "reason": "synthetic dry-run record",
            "sampled_thread_ids": [1],
        },
    }


def synthetic_block_record(
    *,
    cell_id: str,
    repeat_index: int,
    requested_affinity: list[int] | None,
    burn: bool,
    run_id: str = "00000000-0000-4000-8000-000000000001",
    block_id: str | None = None,
    admissible: bool = True,
    reasons: list[str] | None = None,
) -> dict[str, Any]:
    """Production-shaped flat block for dry-run / partial-reader fixtures."""
    readback = requested_affinity if requested_affinity is not None else [0, 1, 2, 3, 4, 5, 6, 7]
    bid = block_id or str(uuid.uuid4())
    after_spinup = _placement_stub("after_worker_threads_spin_up", affinity=readback)
    mid = _placement_stub("mid_generation", affinity=readback)
    return build_flat_block_record(
        run_id=run_id,
        block_id=bid,
        cell_id=cell_id,
        repeat_index=repeat_index,
        measurement={
            "R_prefill": _cited(run_id, 512.0),
            "R_decode": _cited(run_id, 51.2),
            "wall_ns": _cited(run_id, 1_500_000_000),
            "ttft_ns": _cited(run_id, 250_000_000),
            "r_prefill_tok_s": 512.0,
            "r_decode_tok_s": 51.2,
            "prompt_tokens": 128,
            "completion_tokens": 64,
            "ttft_source": "synthetic",
            "frequency_sample_count": 0,
            "cell_label": f"synthetic_{cell_id}",
            "burn": burn,
            "frequency": {"interpretation": "synthetic"},
        },
        placement={
            "affinity_requested": requested_affinity,
            "readback_after_spinup": {
                "process_affinity_mask": after_spinup["process_affinity_readback"],
                "per_thread": after_spinup["per_thread_cpu_placement"],
                "captured_utc": after_spinup["captured_utc"],
                "generation_active": after_spinup["generation_active"],
            },
            "readback_mid_generation": {
                "process_affinity_mask": mid["process_affinity_readback"],
                "per_thread": mid["per_thread_cpu_placement"],
                "captured_utc": mid["captured_utc"],
                "generation_active": mid["generation_active"],
            },
            "matches_request": _matches_request(requested_affinity, readback),
            "affinity_application": {
                "requested_affinity": requested_affinity,
                "application": "synthetic",
                "launch_baseline_affinity": [0, 1, 2, 3, 4, 5, 6, 7],
                "applied_mask": readback,
                "readback": readback,
                "contradictory_readback": False,
            },
            "before_generation": _placement_stub("before_generation", affinity=readback),
            "after_generation": _placement_stub("after_generation", affinity=readback),
        },
        canary={
            "before_ns": _cited(run_id, 1000),
            "after_ns": _cited(run_id, 1000),
            "drift_pct": _cited(run_id, 0.0),
            "admissible": True,
            "threshold": 0.15,
        },
        telemetry={
            "memory": {
                "available_memory_mb_before": _cited(run_id, 4096.0),
                "available_memory_mb_after": _cited(run_id, 4096.0),
            },
            "paging": {
                "page_reads_per_sec": _cited(run_id, 0.0),
                "hard_page_reads_samples": _cited(run_id, [0.0, 0.0]),
            },
            "cpu": {
                "cpu_pct_total": _cited(run_id, 5.0),
                "cpu_pct_per_core": _cited(run_id, [5.0] * 8),
            },
            "package_temp": None,
            "lock_acquired_utc": "1970-01-01T00:00:00+00:00",
            "lock_released_utc": "1970-01-01T00:00:01+00:00",
        },
        admissible=admissible,
        reasons=list(reasons or []),
        timestamp_utc="1970-01-01T00:00:01+00:00",
    )


def synthetic_measurement_record(
    *,
    cell_id: str,
    repeat_index: int,
    requested_affinity: list[int] | None,
    burn: bool,
    alias_timed_measurement: bool = False,
    run_id: str = "00000000-0000-4000-8000-000000000001",
) -> dict[str, Any]:
    """Synthetic record.

    Default: flat production shape. With ``alias_timed_measurement=True``, deliberately rebuilds
    the historical ``measurement ↔ validity.timed_measurement`` cycle for regression tests only.
    """
    if not alias_timed_measurement:
        return synthetic_block_record(
            cell_id=cell_id,
            repeat_index=repeat_index,
            requested_affinity=requested_affinity,
            burn=burn,
            run_id=run_id,
        )
    readback = requested_affinity if requested_affinity is not None else [0, 1, 2, 3, 4, 5, 6, 7]
    measurement: dict[str, Any] = {
        "cell_id": cell_id,
        "cell_label": f"synthetic_{cell_id}",
        "burn": burn,
        "repeat_index": repeat_index,
        "requested_affinity": requested_affinity,
        "placement": {
            "affinity_application": {
                "requested_affinity": requested_affinity,
                "application": "synthetic",
                "launch_baseline_affinity": [0, 1, 2, 3, 4, 5, 6, 7],
                "applied_mask": readback,
                "readback": readback,
                "contradictory_readback": False,
            },
            "before_generation": _placement_stub("before_generation", affinity=readback),
            "after_worker_threads_spin_up": _placement_stub(
                "after_worker_threads_spin_up", affinity=readback
            ),
            "mid_generation": _placement_stub("mid_generation", affinity=readback),
            "after_generation": _placement_stub("after_generation", affinity=readback),
        },
        "prompt_tokens": 128,
        "completion_tokens": 64,
        "wall_s": 1.5,
        "ttft_s": 0.25,
        "ttft_source": "synthetic",
        "r_prefill_tok_s": 512.0,
        "r_decode_tok_s": 51.2,
        "frequency": {"interpretation": "synthetic"},
        "frequency_sample_count": 0,
    }
    validity: dict[str, Any] = {
        "label": f"prompt-a/{cell_id}/{repeat_index}",
        "valid": True,
        "invalid_reasons": [],
        "lock_acquired_utc": "1970-01-01T00:00:00+00:00",
        "lock_released_utc": "1970-01-01T00:00:01+00:00",
        "canary_pre": {
            "workers": [{"runtime_ns": 1000}],
            "affinity_passed": True,
        },
        "canary_post": {
            "workers": [{"runtime_ns": 1000}],
            "affinity_passed": True,
        },
        "canary_relative_drift": 0.0,
        "available_memory_mb_before": 4096.0,
        "available_memory_mb_after": 4096.0,
        "hard_page_reads_per_s": [0.0, 0.0],
        "cpu_pct_total": 5.0,
        "cpu_pct_per_core": [5.0] * 8,
        "package_temp_c": None,
        "memory_pressure": {
            "valid": True,
            "invalid_reasons": [],
            "sustained_definition": ">= 2 consecutive samples with hard_page_reads_per_s > 0",
        },
    }
    # Historical defect behind cb0ed2e3: measurement <-> validity.timed_measurement alias.
    validity["timed_measurement"] = measurement
    measurement["validity"] = validity
    return measurement


def synthetic_placement_summary(cfg: dict[str, Any]) -> dict[str, Any]:
    """Build a production-shaped summary from synthetic flat records (no model load)."""
    records = [
        synthetic_block_record(
            cell_id=str(cell["id"]),
            repeat_index=0,
            requested_affinity=(
                [int(cpu) for cpu in cell["requested_affinity"]]
                if cell.get("requested_affinity") is not None
                else None
            ),
            burn=bool(cell["burn"]),
        )
        for cell in cfg["design"]["cells"]
    ]
    return {
        "experiment_id": cfg["experiment_id"],
        "measurement_session_id": "00000000-0000-4000-8000-000000000099",
        "verdict": "UNEXPLAINED",
        "verdict_rule": {
            "allowed": [
                "PLACEMENT",
                "CONTENTION_DRIVEN_MIGRATION",
                "DIRECT_CONTENTION",
                "UNEXPLAINED",
            ],
            "evidence": {"synthetic_dry_run": True},
        },
        "design": {
            "experiment": "four-cell placement sweep",
            "cells": cfg["design"]["cells"],
            "initial_randomized_schedule": [str(cell["id"]) for cell in cfg["design"]["cells"]],
            "executed_schedule": [record["cell_id"] for record in records],
            "valid_counts": {str(cell["id"]): 1 for cell in cfg["design"]["cells"]},
            "attempted_counts": {str(cell["id"]): 1 for cell in cfg["design"]["cells"]},
            "cooldown_s": cfg["design"]["cooldown_s"],
            "randomization_seed": cfg["design"]["randomization_seed"],
        },
        "records": records,
        "warmup_discarded": synthetic_block_record(
            cell_id="WARMUP",
            repeat_index=-1,
            requested_affinity=None,
            burn=False,
        ),
        "cells": {
            str(cell["id"]): {
                "definition": dict(cell),
                "r_prefill_tok_s": {"median": 512.0, "ci_low": 512.0, "ci_high": 512.0},
                "r_decode_tok_s": {"median": 51.2, "ci_low": 51.2, "ci_high": 51.2},
            }
            for cell in cfg["design"]["cells"]
        },
        "ratios": {
            "C1_over_C2": {
                "target_ratio": float(cfg["design"]["target_ratio"]),
                "target_ratio_source_run_ids": cfg["design"]["target_ratio_source_runs"],
                "endpoints": {
                    "r_prefill_tok_s": {"ratio": 1.0, "ci_low": 1.0, "ci_high": 1.0},
                    "r_decode_tok_s": {"ratio": 1.0, "ci_low": 1.0, "ci_high": 1.0},
                },
            },
            "C3_over_C4": {
                "endpoints": {
                    "r_prefill_tok_s": {"ratio": 1.0, "ci_low": 1.0, "ci_high": 1.0},
                    "r_decode_tok_s": {"ratio": 1.0, "ci_low": 1.0, "ci_high": 1.0},
                },
                "interpretation": "unrequested clean over unrequested P-core burn",
            },
        },
        "placement_mechanism": {
            "thread_placement_available": False,
            "classification": None,
        },
        "frequency_interpretation": cfg["frequency"]["interpretation"],
        "dry_run": True,
    }


def reproduce_alias_cycle_paths(
    cfg: dict[str, Any],
    *,
    build_report: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None,
) -> list[str]:
    """Reproduce the cb0ed2e3 cycle with production report builder + aliased synthetic records.

    Limitation: the original in-memory summary from
    ``raw/cb0ed2e3-ed70-4627-b231-51016d2b255b`` is gone (directory holds only
    ``events.ndjson``; no ``summary.json``). Reproduction uses the production report
    builder when provided, otherwise the synthetic summary shape, plus same-shape
    synthetic records with the historical ``measurement ↔ validity.timed_measurement`` alias.
    """
    broken = synthetic_measurement_record(
        cell_id="C1",
        repeat_index=0,
        requested_affinity=[0, 1, 2, 3],
        burn=False,
        alias_timed_measurement=True,
    )
    if build_report is not None:
        summary = build_report([broken])
    else:
        summary = synthetic_placement_summary(cfg)
        summary["records"] = [broken]
    return find_cycles(summary)


def _prepare_isolated_emit_root(tmp_root: Path, real_root: Path) -> Any:
    """Temp repo outside project raw/: platform config, provenance stubs, git HEAD."""
    platform_src = real_root / "configs" / "platforms" / "aipc-c1.yaml"
    platform_dst = tmp_root / "configs" / "platforms" / "aipc-c1.yaml"
    platform_dst.parent.mkdir(parents=True)
    shutil.copy(platform_src, platform_dst)
    provisional = resolve_config([platform_dst], repo_root=tmp_root)
    for relative in provisional.get("provenance_artifacts") or []:
        artifact = tmp_root / str(relative)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        if not artifact.is_file():
            artifact.write_text(f"dry-run provenance stub: {relative}\n", encoding="utf-8")
    dry_config = resolve_config(
        [platform_dst],
        overrides={
            "topology": {
                "verified": True,
                "p_cpus": [0, 1, 2, 3],
                "lpe_cpus": [4, 5, 6, 7],
                "measured": {"run_id": "00000000-0000-4000-8000-0000000000aa"},
            }
        },
        repo_root=tmp_root,
    )
    # emit() takes exclusive(raw) and writes under raw/; keep the dry-run git tree clean.
    (tmp_root / ".gitignore").write_text("raw/\nraw.lock\n.locks/\n", encoding="utf-8")
    subprocess.run(
        ["git", "init"],
        cwd=tmp_root,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "add", "-A"],
        cwd=tmp_root,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=dryrun@seam.local",
            "-c",
            "user.name=seam-dryrun",
            "commit",
            "-m",
            "prompt-a2 output-path dry-run",
        ],
        cwd=tmp_root,
        check=True,
        capture_output=True,
        text=True,
    )
    # Prove the temp tree has a resolvable clean HEAD before emit.
    git_state = capture_git_state(cwd=tmp_root)
    if git_state.dirty:
        raise SeamError(f"dry-run temp repo is dirty after commit: {git_state.dirty_files}")
    return dry_config


def startup_output_path_dry_run(
    *,
    root: Path,
    cfg: dict[str, Any],
    allow_dirty: bool,
) -> dict[str, Any]:
    """Unconditional output-path dry-run before model load; temp dir outside project raw/."""
    del allow_dirty  # isolated temp repo always records its own clean git commit
    summary = synthetic_placement_summary(cfg)
    assert_acyclic(summary, label="startup output-path dry-run synthetic summary")
    # Deliberate regression surface: callers may inject cycles into a copy and expect fail-fast.
    with tempfile.TemporaryDirectory(prefix="seam-prompt-a2-output-dryrun-") as tmp:
        tmp_root = Path(tmp)
        if (
            root.resolve() in tmp_root.resolve().parents
            or tmp_root.resolve() == (root / "raw").resolve()
        ):
            raise SeamError("dry-run temp dir must remain outside project raw/")
        dry_config = _prepare_isolated_emit_root(tmp_root, root)
        power = capture_power_state()
        handle = emit(
            config=dry_config,
            target="cpu-placement",
            workload={
                "kind": "prompt_a2_preflight",
                "benchmark": "synthetic_summary_serialization_dry_run",
                "task_ids": [],
                "seed": int(cfg["design"]["randomization_seed"]),
                "n_repeats": 1,
                "concurrency": 1,
            },
            condition_label="prompt_a2_output_path_dry_run",
            repo_root=tmp_root,
            allow_dirty=False,
            summary=summary,
            power_state=manifest_power_state(power, background_quiesced=True),
            thermal={"regime": "confound", "excluded": False},
            self_check="pass",
        )
        if not verify_sealed(handle.run_dir):
            raise SeamError(
                f"startup output-path dry-run seal verification failed for {handle.run_id}"
            )
        summary_on_disk = json.loads(
            (handle.run_dir.path / "summary.json").read_text(encoding="utf-8")
        )
        assert_acyclic(summary_on_disk, label="dry-run summary.json on disk")
        return {
            "passed": True,
            "dry_run_run_id": handle.run_id,
            "temp_repo_root": str(tmp_root),
            "outside_project_raw": True,
            "seal_verified": True,
            "summary_keys": sorted(summary_on_disk),
        }


def fail_dry_run_on_cyclic_summary(cfg: dict[str, Any]) -> None:
    """Demonstrate fail-fast: cyclic synthetic summary is refused with find_cycles paths."""
    summary = synthetic_placement_summary(cfg)
    cyclic_record = synthetic_measurement_record(
        cell_id="C1",
        repeat_index=0,
        requested_affinity=[0, 1, 2, 3],
        burn=False,
        alias_timed_measurement=True,
    )
    summary["records"] = [cyclic_record]
    cycles = find_cycles(summary)
    if not cycles:
        raise SeamError("expected deliberate cyclic summary to expose find_cycles paths")
    assert_acyclic(summary, label="deliberate cyclic dry-run summary")
