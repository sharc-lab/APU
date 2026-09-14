"""Prompt A throughput-integrity audit and clean-versus-P-core-burn reproduction."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import socket
import statistics
import threading
import time
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from seam.agent.tools import SYSTEM_PROMPT, TOOL_SPECS
from seam.backends.base import GenerationRequest
from seam.backends.local_openvino import LocalOpenVinoBackend, runtime_info
from seam.config import resolve_config
from seam.errors import SeamError
from seam.gitinfo import repo_root
from seam.locks import exclusive, read_lock_record
from seam.manifest import emit
from seam.measurement import (
    MeasurementRefusalError,
    capture_process_affinity,
    machine_measurement,
    measure_quiescence,
)
from seam.model_provenance import load_local_spec, manifest_model_block, quantization_summary
from seam.powerstate import (
    capture_battery_status_wmi,
    capture_power_state,
    is_charging_complete,
    manifest_power_state,
)
from seam.rawstore import open_run_dir, verify_sealed
from seam.telemetry.frequency import FrequencySampler
from seam.tools.affinity_matrix import check_forbidden_processes
from seam.tools.prompt_a_lifecycle import (
    append_block_jsonl,
    assert_acyclic,
    block_jsonl_record,
    build_flat_block_record,
    find_cycles,
    open_in_progress_run,
    progress_status,
    reproduce_alias_cycle_paths,
    startup_output_path_dry_run,
)

__all__ = ["find_cycles", "main", "reproduce_alias_cycle_paths"]

_CONFIG_PATH = Path("configs/prompt_a.yaml")
_PLATFORM_PATH = Path("configs/platforms/aipc-c1.yaml")
_RUNTIME_ENV_PREFIXES = ("OV_", "OPENVINO_", "OMP_", "KMP_", "TBB_")


def _load_config(root: Path) -> tuple[dict[str, Any], Any]:
    path = root / _CONFIG_PATH
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SeamError(f"{path} is not a mapping")
    measurement_path = root / str(data["measurement_config"])
    measurement = yaml.safe_load(measurement_path.read_text(encoding="utf-8"))
    if not isinstance(measurement, dict):
        raise SeamError(f"{measurement_path} is not a mapping")
    quiescence = {**measurement["quiescence"], **data["quiescence"]}
    data = {**data, **measurement, "quiescence": quiescence}
    resolved = resolve_config([root / _PLATFORM_PATH, measurement_path, path], repo_root=root)
    return data, resolved


def _json_pointer(parts: list[str | int]) -> str:
    if not parts:
        return ""
    return "/" + "/".join(str(part).replace("~", "~0").replace("/", "~1") for part in parts)


def _recursive_diff(
    left: Any, right: Any, parts: list[str | int] | None = None
) -> list[dict[str, Any]]:
    path = parts or []
    if type(left) is not type(right):
        return [{"path": _json_pointer(path), "left": left, "right": right}]
    if isinstance(left, dict):
        rows: list[dict[str, Any]] = []
        for key in sorted(set(left) | set(right)):
            if key not in left:
                rows.append(
                    {
                        "path": _json_pointer([*path, key]),
                        "left": {"__missing__": True},
                        "right": right[key],
                    }
                )
            elif key not in right:
                rows.append(
                    {
                        "path": _json_pointer([*path, key]),
                        "left": left[key],
                        "right": {"__missing__": True},
                    }
                )
            else:
                rows.extend(_recursive_diff(left[key], right[key], [*path, key]))
        return rows
    if isinstance(left, list):
        rows = []
        for index in range(max(len(left), len(right))):
            if index >= len(left):
                rows.append(
                    {
                        "path": _json_pointer([*path, index]),
                        "left": {"__missing__": True},
                        "right": right[index],
                    }
                )
            elif index >= len(right):
                rows.append(
                    {
                        "path": _json_pointer([*path, index]),
                        "left": left[index],
                        "right": {"__missing__": True},
                    }
                )
            else:
                rows.extend(_recursive_diff(left[index], right[index], [*path, index]))
        return rows
    return [] if left == right else [{"path": _json_pointer(path), "left": left, "right": right}]


def _categorize_diff(rows: list[dict[str, Any]], *, scope: str) -> list[dict[str, Any]]:
    """Attach an exhaustive first-component category without dropping any diff row."""
    categorized: list[dict[str, Any]] = []
    for row in rows:
        first = str(row["path"]).lstrip("/").split("/", maxsplit=1)[0] or "root"
        categorized.append({**row, "scope": scope, "category": first})
    return categorized


def _flatten_json(
    value: Any,
    *,
    parts: list[str | int] | None = None,
    leaves: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Flatten JSON leaves independently of :func:`_recursive_diff` for completeness checks."""
    path = parts or []
    result = leaves if leaves is not None else {}
    if isinstance(value, dict) and value:
        for key in sorted(value):
            _flatten_json(value[key], parts=[*path, key], leaves=result)
    elif isinstance(value, list) and value:
        for index, item in enumerate(value):
            _flatten_json(item, parts=[*path, index], leaves=result)
    else:
        result[_json_pointer(path)] = value
    return result


def _diff_completeness(left: Any, right: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Prove the recursive artifact has every differing path and both source values."""
    left_leaves = _flatten_json(left)
    right_leaves = _flatten_json(right)
    missing = {"__missing__": True}
    expected: dict[str, dict[str, Any]] = {}
    pending: list[tuple[Any, Any, list[str | int]]] = [(left, right, [])]
    while pending:
        left_value, right_value, path = pending.pop()
        pointer = _json_pointer(path)
        if type(left_value) is not type(right_value):
            expected[pointer] = {"left": left_value, "right": right_value}
        elif isinstance(left_value, dict):
            for key in sorted(set(left_value) | set(right_value), reverse=True):
                if key not in left_value:
                    expected[_json_pointer([*path, key])] = {
                        "left": missing,
                        "right": right_value[key],
                    }
                elif key not in right_value:
                    expected[_json_pointer([*path, key])] = {
                        "left": left_value[key],
                        "right": missing,
                    }
                else:
                    pending.append((left_value[key], right_value[key], [*path, key]))
        elif isinstance(left_value, list):
            for index in reversed(range(max(len(left_value), len(right_value)))):
                if index >= len(left_value):
                    expected[_json_pointer([*path, index])] = {
                        "left": missing,
                        "right": right_value[index],
                    }
                elif index >= len(right_value):
                    expected[_json_pointer([*path, index])] = {
                        "left": left_value[index],
                        "right": missing,
                    }
                else:
                    pending.append((left_value[index], right_value[index], [*path, index]))
        elif left_value != right_value:
            expected[pointer] = {"left": left_value, "right": right_value}
    observed = {str(row["path"]): {"left": row["left"], "right": row["right"]} for row in rows}
    duplicate_paths = len(observed) != len(rows)
    return {
        "left_leaf_count": len(left_leaves),
        "right_leaf_count": len(right_leaves),
        "expected_differing_leaf_count": len(expected),
        "observed_diff_row_count": len(rows),
        "duplicate_paths": duplicate_paths,
        "missing_paths": sorted(set(expected) - set(observed)),
        "unexpected_paths": sorted(set(observed) - set(expected)),
        "value_mismatch_paths": sorted(
            path for path in set(expected) & set(observed) if expected[path] != observed[path]
        ),
        "complete": (
            not duplicate_paths
            and set(expected) == set(observed)
            and all(expected[path] == observed[path] for path in expected)
        ),
        "method": (
            "independent iterative walk of both sealed JSON documents; path-set and both-side "
            "value equality checked against recursive diff rows"
        ),
    }


def _at(value: Any, *path: str) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _comparison(
    name: str,
    pilot: Any,
    full: Any,
    *,
    interpretation: str | None = None,
    verified: bool = True,
) -> dict[str, Any]:
    return {
        "field": name,
        "pilot": pilot,
        "full": full,
        "equal": pilot == full,
        "verified": verified,
        "interpretation": interpretation,
    }


def _manifest_audit(root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    pilot_id = str(cfg["source_runs"]["pilot"])
    full_id = str(cfg["source_runs"]["full"])
    pilot_dir = root / "raw" / pilot_id
    full_dir = root / "raw" / full_id
    pilot_run = open_run_dir(pilot_id, repo_root=root)
    full_run = open_run_dir(full_id, repo_root=root)
    pilot_integrity_ok = verify_sealed(pilot_run)
    full_integrity_ok = verify_sealed(full_run)
    if not pilot_integrity_ok or not full_integrity_ok:
        raise SeamError(
            "source-run integrity verification failed; no manifest comparison or measurement "
            "is permitted"
        )
    pilot_manifest = json.loads((pilot_dir / "manifest.json").read_text(encoding="utf-8"))
    full_manifest = json.loads((full_dir / "manifest.json").read_text(encoding="utf-8"))
    pilot_summary = json.loads((pilot_dir / "summary.json").read_text(encoding="utf-8"))
    full_summary = json.loads((full_dir / "summary.json").read_text(encoding="utf-8"))
    pilot_seal = json.loads((pilot_dir / ".sealed").read_text(encoding="utf-8"))
    full_seal = json.loads((full_dir / ".sealed").read_text(encoding="utf-8"))

    pilot_props = _at(pilot_summary, "backend_config", "properties") or {}
    full_props = _at(full_summary, "backend_config", "properties") or {}
    manifest_rows = _recursive_diff(pilot_manifest, full_manifest)
    summary_rows = _recursive_diff(pilot_summary, full_summary)
    confirmations = [
        _comparison(
            "INFERENCE_NUM_THREADS",
            pilot_props.get("INFERENCE_NUM_THREADS"),
            full_props.get("INFERENCE_NUM_THREADS"),
        ),
        _comparison(
            "thread_count_inference",
            {
                "declared": pilot_props.get("INFERENCE_NUM_THREADS"),
                "basis": "backend property, not cpu_count()",
            },
            {
                "declared": full_props.get("INFERENCE_NUM_THREADS"),
                "basis": "backend property, not cpu_count()",
            },
        ),
        _comparison(
            "SCHEDULING_CORE_TYPE",
            pilot_props.get("SCHEDULING_CORE_TYPE"),
            full_props.get("SCHEDULING_CORE_TYPE"),
            interpretation="null means the property was not set",
        ),
        _comparison(
            "ENABLE_CPU_PINNING",
            pilot_props.get("ENABLE_CPU_PINNING"),
            full_props.get("ENABLE_CPU_PINNING"),
            interpretation="null means the property was not set",
        ),
        _comparison(
            "requested_process_affinity",
            _at(pilot_summary, "backend_config", "affinity_cpus"),
            _at(full_summary, "backend_config", "affinity_cpus"),
            interpretation="null means no process-affinity mask was requested",
        ),
        _comparison(
            "process_affinity_readback",
            _at(pilot_summary, "backend_config", "process_affinity_readback"),
            _at(full_summary, "backend_config", "process_affinity_readback"),
            interpretation=(
                "not recorded in either sealed run; absence is equal but actual affinity equality "
                "cannot be established"
            ),
            verified=False,
        ),
        _comparison(
            "model_ir_sha256",
            _at(pilot_manifest, "model", "ir_sha256"),
            _at(full_manifest, "model", "ir_sha256"),
        ),
        _comparison(
            "openvino_version",
            _at(pilot_manifest, "drivers", "openvino"),
            _at(full_manifest, "drivers", "openvino"),
        ),
        _comparison(
            "openvino_genai_version",
            _at(pilot_manifest, "drivers", "genai"),
            _at(full_manifest, "drivers", "genai"),
        ),
        _comparison(
            "power_plan",
            _at(pilot_manifest, "power_state", "power_plan"),
            _at(full_manifest, "power_state", "power_plan"),
        ),
        _comparison(
            "ANTHROPIC_API_KEY_presence",
            _at(pilot_summary, "quiesce", "cloud_credential_present"),
            _at(full_summary, "quiesce", "cloud_credential_present"),
            interpretation="presence only; key material is never read into the artifact",
        ),
    ]
    environment_trace = {
        "seam.backends.local_openvino": {
            "environment_reads": [],
            "evidence": "source trace: no os.environ/os.getenv access",
        },
        "installed_openvino_python_inference_import_path": {
            "environment_reads": ["OPENVINO_LIB_PATHS"],
            "evidence": (
                ".venv-seam/Lib/site-packages/openvino/package_utils.py reads it with os.getenv "
                "while resolving runtime libraries"
            ),
            "pilot_capture": "absent",
            "full_capture": "absent",
            "historical_equality_verified": False,
        },
        "installed_openvino_genai_python_shim": {
            "environment_reads": [],
            "evidence": (
                "source trace of the installed openvino_genai Python package found no "
                "os.environ/os.getenv access"
            ),
        },
        "seam.tools.efilter_run": {
            "environment_reads": ["ANTHROPIC_API_KEY"],
            "use": "presence-only refusal check",
            "pilot_presence": _at(pilot_summary, "quiesce", "cloud_credential_present"),
            "full_presence": _at(full_summary, "quiesce", "cloud_credential_present"),
        },
        "openvino_external_runtime": {
            "environment_reads": "not introspectable from repository or Python-package source",
            "sealed_run_capture": "absent",
            "consequence": (
                "runtime-environment equality for the historical runs cannot be established; "
                "the reproduction captures runtime-affecting prefix families"
            ),
        },
    }
    configuration_explanation = any(
        not row["equal"]
        for row in confirmations
        if row["field"]
        in {
            "INFERENCE_NUM_THREADS",
            "thread_count_inference",
            "SCHEDULING_CORE_TYPE",
            "ENABLE_CPU_PINNING",
            "requested_process_affinity",
            "model_ir_sha256",
            "openvino_version",
            "openvino_genai_version",
            "power_plan",
        }
    )
    manifest_completeness = _diff_completeness(pilot_manifest, full_manifest, manifest_rows)
    summary_completeness = _diff_completeness(pilot_summary, full_summary, summary_rows)
    if not manifest_completeness["complete"] or not summary_completeness["complete"]:
        raise SeamError("recursive diff completeness proof failed; refusing to publish artifact")
    return {
        "schema_version": 1,
        "pilot_run_id": pilot_id,
        "full_run_id": full_id,
        "source_integrity_verified": {
            "pilot": pilot_integrity_ok,
            "full": full_integrity_ok,
        },
        "seals": {"pilot": pilot_seal, "full": full_seal},
        "source_sha256": {
            "pilot_manifest": _sha256_path(pilot_dir / "manifest.json"),
            "full_manifest": _sha256_path(full_dir / "manifest.json"),
            "pilot_summary": _sha256_path(pilot_dir / "summary.json"),
            "full_summary": _sha256_path(full_dir / "summary.json"),
        },
        "completeness_proof": {
            "manifest.json": manifest_completeness,
            "summary.json": summary_completeness,
        },
        "manifest_diff": _categorize_diff(manifest_rows, scope="manifest.json"),
        "summary_diff": _categorize_diff(summary_rows, scope="summary.json"),
        "explicit_confirmations": confirmations,
        "environment_read_trace": environment_trace,
        "configuration_fully_explains_2x": configuration_explanation,
        "step1_verdict": "CONFIGURATION" if configuration_explanation else "UNRESOLVED",
    }


def _write_json_locked(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with exclusive(path):
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _wait_for_machine_lock(root: Path, cfg: dict[str, Any]) -> float:
    lock_path = root / ".locks" / "machine.lock"
    started = time.monotonic()
    while lock_path.exists():
        if time.monotonic() - started >= float(cfg["locking"]["wait_timeout_s"]):
            raise SeamError(f"timed out waiting for machine lock {lock_path}")
        time.sleep(float(cfg["locking"]["poll_interval_s"]))
    return time.monotonic() - started


def _burn_worker(cpu: int, ready: Any, stop: Any) -> None:
    import psutil

    process = psutil.Process()
    process.cpu_affinity([cpu])
    counter = 0
    state = hashlib.sha256(f"prompt-a-burn-{cpu}".encode("ascii")).digest()
    ready.put(
        {
            "pid": os.getpid(),
            "cpu": cpu,
            "requested_affinity": [cpu],
            "affinity_readback": [int(value) for value in process.cpu_affinity()],
        }
    )
    while not stop.is_set():
        state = hashlib.sha256(state + counter.to_bytes(8, "little")).digest()
        counter = (counter + 1) & ((1 << 64) - 1)
    if not state:  # pragma: no cover - keeps the consumed state semantically live
        raise RuntimeError("unreachable burn digest")


def _start_burn(cfg: dict[str, Any]) -> tuple[Any, list[Any], list[dict[str, Any]]]:
    burn = cfg["burn"]
    cpus = [int(cpu) for cpu in burn["cpus"]]
    if int(burn["processes"]) != len(cpus):
        raise SeamError("burn.processes must equal the number of explicitly pinned CPUs")
    context = mp.get_context("spawn")
    stop = context.Event()
    ready = context.Queue()
    processes = [
        context.Process(target=_burn_worker, args=(cpu, ready, stop), name=f"prompt-a-burn-{cpu}")
        for cpu in cpus
    ]
    for process in processes:
        process.start()
    readbacks: list[dict[str, Any]] = []
    deadline = time.monotonic() + float(burn["startup_timeout_s"])
    for _ in processes:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            stop.set()
            raise SeamError("timed out waiting for P-core burn workers")
        readbacks.append(ready.get(timeout=remaining))
    readbacks.sort(key=lambda row: int(row["cpu"]))
    if any(row["affinity_readback"] != [row["cpu"]] for row in readbacks):
        stop.set()
        for process in processes:
            process.join(timeout=5)
        raise SeamError(f"burn affinity readback failed: {readbacks}")
    return stop, processes, readbacks


def _stop_burn(stop: Any, processes: list[Any]) -> list[dict[str, Any]]:
    stop.set()
    states: list[dict[str, Any]] = []
    for process in processes:
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        states.append(
            {
                "pid": process.pid,
                "exitcode": process.exitcode,
                "terminated_after_timeout": process.exitcode not in (0, None),
            }
        )
    return states


@contextmanager
def _burn_context(cfg: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Keep the declared burn active across both canaries and the timed generation."""
    stop, processes, readbacks = _start_burn(cfg)
    record = {
        "algorithm": cfg["burn"]["algorithm"],
        "worker_start_readbacks": readbacks,
        "worker_stop_states": None,
    }
    try:
        yield record
    finally:
        record["worker_stop_states"] = _stop_burn(stop, processes)


def _bootstrap_median(values: list[float], *, cfg: dict[str, Any]) -> dict[str, Any]:
    design = cfg["design"]
    n_resamples = int(design["bootstrap_resamples"])
    confidence = float(design["confidence"])
    rng = random.Random(int(design["bootstrap_seed"]))
    n = len(values)
    draws = sorted(
        statistics.median(values[rng.randrange(n)] for _ in range(n)) for _ in range(n_resamples)
    )
    alpha = (1.0 - confidence) / 2.0
    lo = draws[max(0, math.floor(alpha * n_resamples))]
    hi = draws[min(n_resamples - 1, math.ceil((1.0 - alpha) * n_resamples) - 1)]
    return {
        "median": statistics.median(values),
        "ci_low": lo,
        "ci_high": hi,
        "confidence": confidence,
        "bootstrap_resamples": n_resamples,
        "n": n,
        "values": values,
    }


def _bootstrap_ratio(
    numerator: list[float], denominator: list[float], *, cfg: dict[str, Any]
) -> dict[str, Any]:
    """Independent percentile bootstrap for a ratio of condition medians."""
    design = cfg["design"]
    n_resamples = int(design["bootstrap_resamples"])
    confidence = float(design["confidence"])
    rng = random.Random(int(design["bootstrap_seed"]))
    draws: list[float] = []
    for _ in range(n_resamples):
        top = statistics.median(
            numerator[rng.randrange(len(numerator))] for _ in range(len(numerator))
        )
        bottom = statistics.median(
            denominator[rng.randrange(len(denominator))] for _ in range(len(denominator))
        )
        draws.append(top / bottom)
    draws.sort()
    alpha = (1.0 - confidence) / 2.0
    lo = draws[max(0, math.floor(alpha * n_resamples))]
    hi = draws[min(n_resamples - 1, math.ceil((1.0 - alpha) * n_resamples) - 1)]
    return {
        "ratio": statistics.median(numerator) / statistics.median(denominator),
        "ci_low": lo,
        "ci_high": hi,
        "confidence": confidence,
        "bootstrap_resamples": n_resamples,
        "statistic": "ratio of independent condition medians",
        "n_numerator": len(numerator),
        "n_denominator": len(denominator),
    }


def _runtime_environment() -> dict[str, str]:
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if key.upper().startswith(_RUNTIME_ENV_PREFIXES)
    }


def _runtime_environment_metadata() -> dict[str, Any]:
    """Capture runtime configuration without reading credential families."""
    return {
        "captured_prefix_values": _runtime_environment(),
        "OPENVINO_LIB_PATHS": {
            "present": "OPENVINO_LIB_PATHS" in os.environ,
            "value": os.environ.get("OPENVINO_LIB_PATHS"),
            "credential_material": False,
        },
        "scope": (
            "values for OpenVINO/OpenMP/TBB configuration families only; credential variables "
            "are not read"
        ),
    }


def _placement_sample(stage: str, *, generation_active: bool | None = None) -> dict[str, Any]:
    """Capture the process mask and explicitly delimit unavailable Windows thread placement."""
    import psutil

    process = psutil.Process()
    threads = process.threads()
    return {
        "stage": stage,
        "captured_utc": _utc_now(),
        "process_affinity_readback": capture_process_affinity(),
        "process_thread_count": len(threads),
        "generation_active": generation_active,
        "per_thread_cpu_placement": {
            "available": False,
            "reason": (
                "psutil.Process.threads() on Windows exposes thread IDs and CPU times but not "
                "the processor currently executing each thread; process affinity is not used "
                "to infer per-thread placement"
            ),
            "sampled_thread_ids": [int(thread.id) for thread in threads],
        },
    }


def _apply_cell_affinity(
    requested_affinity: list[int] | None, *, baseline_affinity: list[int]
) -> dict[str, Any]:
    """Apply a requested mask, or restore the launch mask for an unrequested cell."""
    import psutil

    process = psutil.Process()
    applied = requested_affinity if requested_affinity is not None else baseline_affinity
    process.cpu_affinity(applied)
    readback = [int(cpu) for cpu in process.cpu_affinity()]
    return {
        "requested_affinity": requested_affinity,
        "application": (
            "explicit_cell_request"
            if requested_affinity is not None
            else "restored_launch_mask_for_unrequested_cell"
        ),
        "launch_baseline_affinity": baseline_affinity,
        "applied_mask": applied,
        "readback": readback,
        "contradictory_readback": (
            requested_affinity is not None and readback != requested_affinity
        ),
    }


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _emit_preflight(
    root: Path, cfg: dict[str, Any], resolved: Any, *, allow_dirty: bool
) -> tuple[str, dict[str, Any]]:
    """Seal Step 0 process/lock evidence before any OpenVINO model construction."""
    import psutil

    process_rows: list[dict[str, Any]] = []
    for process in psutil.process_iter(["pid", "name", "memory_info", "cmdline"]):
        try:
            name = str(process.info.get("name") or "<unknown>")
            command = " ".join(str(part) for part in (process.info.get("cmdline") or []))
            if not any(
                token in f"{name} {command}".lower()
                for token in ("python", "openvino", "ovms", "benchmark_app", "optimum")
            ):
                continue
            process_rows.append(
                {
                    "name": name,
                    "pid": int(process.info["pid"]),
                    "rss_bytes": int(process.info["memory_info"].rss),
                    "role": (
                        "prompt_a2_preflight_recorder"
                        if int(process.info["pid"]) in {os.getpid(), os.getppid()}
                        else (
                            "cursor_project_state_service"
                            if "tools\\seam_mcp\\project_state.py" in command
                            else "unclassified_model_related_candidate"
                        )
                    ),
                }
            )
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
    process_rows.sort(key=lambda row: (-int(row["rss_bytes"]), int(row["pid"])))
    unclassified = [
        row for row in process_rows if row["role"] == "unclassified_model_related_candidate"
    ]
    refused = bool(unclassified)
    lock_path = root / ".locks" / "machine.lock"
    summary = {
        "experiment_id": cfg["experiment_id"],
        "classification": ("PROMPT_A2_PREFLIGHT_REFUSAL" if refused else "PROMPT_A2_PREFLIGHT"),
        "process_check": {
            "captured_utc": _utc_now(),
            "boot_time": datetime.fromtimestamp(psutil.boot_time(), UTC).isoformat(),
            "hostname": socket.gethostname(),
            "filter": "python/openvino/ovms/benchmark_app/optimum in process name or command",
            "processes": process_rows,
            "unclassified_candidates": unclassified,
            "orphan_openvino_verdict": "NONE" if not unclassified else "CANDIDATE_PRESENT",
            "verdict_basis": (
                "all matched Python processes were identified as this recorder or Cursor's "
                "project-state service; no OpenVINO workload process remained"
                if not unclassified
                else "one or more matched processes could not be classified; refusal required"
            ),
        },
        "machine_lock_at_inspection": {
            "path": str(lock_path),
            "exists": lock_path.is_file(),
            "record": read_lock_record(lock_path) if lock_path.is_file() else None,
        },
        "lock_reclamation_policy": {
            "record_fields": ["pid", "boot_time", "hostname"],
            "cross_boot": "automatic only on the same hostname; both boot times logged",
            "same_boot_dead_pid": "default refusal; explicit recovery flag and log required",
            "live_owner": "refusal",
            "legacy_record": "refusal because boot identity is absent",
        },
        "governing_amendment_source": {
            "followed": (
                "pinned docs/SEAM_research_blueprint.md AM-027 dated 2026-08-04 and human "
                "Prompt A2 dispatch"
            ),
            "stale_source": (
                "MCP seam_amendments/AMENDMENTS.md exposes the superseded v1 AM-027 while the "
                "pinned blueprint says only v1 AM-001..AM-024 remain in force"
            ),
            "issue": "stale amendment ledger/parser source; not used for this run",
        },
    }
    power = capture_power_state()
    handle = emit(
        config=resolved,
        target="cpu-p",
        workload={
            "kind": "prompt_a2_refusal" if refused else "prompt_a2_preflight",
            "benchmark": "process_and_lock_integrity_check",
            "task_ids": [],
            "seed": None,
            "n_repeats": 1,
            "concurrency": 1,
        },
        condition_label=(
            "prompt_a2_preflight_candidate_refusal"
            if refused
            else "prompt_a2_preflight_no_measurement"
        ),
        repo_root=root,
        allow_dirty=allow_dirty,
        summary=summary,
        power_state=manifest_power_state(power, background_quiesced=False),
        thermal={"regime": "confound", "excluded": False},
        self_check="fail" if refused else "pass",
    )
    return handle.run_id, summary


def _build_backend(root: Path, cfg: dict[str, Any]) -> tuple[LocalOpenVinoBackend, dict[str, Any]]:
    spec_path = root / cfg["openvino"]["model_spec"]
    spec = load_local_spec(spec_path)
    ov = cfg["openvino"]
    backend = LocalOpenVinoBackend(
        model_dir=Path(spec["ir_dir"]),
        target="cpu-p",
        scheduling_core_type=ov["scheduling_core_type"],
        inference_num_threads=int(ov["inference_num_threads"]),
        enable_cpu_pinning=ov["enable_cpu_pinning"],
        model_ref=f"{spec['name']}@{str(spec['revision'])[:12]}+{quantization_summary(spec)}",
        enable_thinking=bool(ov["enable_thinking"]),
        affinity_cpus=None,
    )
    return backend, spec


def _probe_request(cfg: dict[str, Any]) -> GenerationRequest:
    return GenerationRequest(
        messages=[{"role": "user", "content": str(cfg["probe"]["prompt"])}],
        system=SYSTEM_PROMPT,
        tools=TOOL_SPECS,
        max_tokens=int(cfg["probe"]["max_tokens"]),
        temperature=0.0,
    )


def _measure_generate(
    *,
    root: Path,
    cfg: dict[str, Any],
    backend: LocalOpenVinoBackend,
    request: GenerationRequest,
    cell: dict[str, Any],
    repeat_index: int,
    p_cpus: list[int],
    baseline_affinity: list[int],
    startup_quiescence: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    sampler = FrequencySampler(interval_s=float(cfg["frequency"]["sample_interval_s"]), n_cpus=8)
    secondary: dict[str, Any] = {}
    requested = cell.get("requested_affinity")
    requested_affinity = [int(cpu) for cpu in requested] if requested is not None else None
    affinity = _apply_cell_affinity(requested_affinity, baseline_affinity=baseline_affinity)
    # Affinity contradiction is a FINDING, not a failure - record and continue.
    if affinity.get("contradictory_readback"):
        print(
            json.dumps(
                {
                    "event": "prompt_a2.affinity_readback_finding",
                    "run_id": run_id,
                    "cell_id": str(cell["id"]),
                    "repeat_index": repeat_index,
                    "requested_affinity": requested_affinity,
                    "readback": affinity.get("readback"),
                    "severity": "finding_not_failure",
                },
                sort_keys=True,
            ),
            flush=True,
        )
    workload_cpus = [int(cpu) for cpu in affinity["readback"]]
    if not workload_cpus:
        raise SeamError(f"{cell['id']} process-affinity readback was empty")
    active_load = (lambda: _burn_context(cfg)) if bool(cell["burn"]) else None
    block_id = str(uuid.uuid4())
    try:
        with machine_measurement(
            repo_root=root,
            label=f"prompt-a/{cell['id']}/{repeat_index}",
            config=cfg,
            p_cpus=p_cpus,
            workload_cpus=workload_cpus,
            secondary_name_check=secondary,
            active_load=active_load,
            prevalidated_quiescence={
                "passed": True,
                "source": "single_detached_startup_gate",
                "gate_lock_acquired_utc": startup_quiescence["lock_acquired_utc"],
                "gate_lock_released_utc": startup_quiescence["lock_released_utc"],
                "thresholds": startup_quiescence["quiescence"]["thresholds"],
            },
        ) as block:
            block.record["cell"] = {
                "id": str(cell["id"]),
                "label": str(cell["label"]),
                "burn": bool(cell["burn"]),
                "affinity": affinity,
                "canary_cpus_basis": "process affinity readback applied for this block",
            }
            hits = check_forbidden_processes(cfg["quiescence"]["forbidden_processes"])
            block.record["secondary_process_name_check"] = {
                "patterns": list(cfg["quiescence"]["forbidden_processes"]),
                "hits": hits,
                "role": "secondary to measured-load quiescence",
            }
            if hits:
                raise SeamError(f"secondary process-name quiescence refusal: {hits}")
            placement_before = _placement_sample("before_generation")
            sampler.start()
            try:
                with ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="prompt-a-generation"
                ) as pool:
                    future = pool.submit(
                        backend.generate,
                        request,
                        ignore_eos=bool(cfg["probe"]["ignore_eos"]),
                    )
                    spin_deadline = time.monotonic() + float(
                        cfg["design"]["worker_spinup_timeout_s"]
                    )
                    while not future.running() and time.monotonic() < spin_deadline:
                        time.sleep(0.01)
                    placement_after_spinup = _placement_sample(
                        "after_worker_threads_spin_up",
                        generation_active=not future.done(),
                    )
                    mid_deadline = time.monotonic() + float(
                        cfg["design"]["mid_generation_sample_delay_s"]
                    )
                    while not future.done() and time.monotonic() < mid_deadline:
                        time.sleep(min(0.05, max(0.0, mid_deadline - time.monotonic())))
                    placement_mid = _placement_sample(
                        "mid_generation",
                        generation_active=not future.done(),
                    )
                    result = future.result()
            finally:
                frequency_samples = sampler.stop()
            ttft_s = float(result.ttft_ns or 0) / 1e9
            decode_s = max(float(result.wall_ns) / 1e9 - ttft_s, 1e-12)
            r_prefill = result.prompt_tokens / ttft_s if ttft_s > 0 else None
            r_decode = result.completion_tokens / decode_s
            placement_after = _placement_sample("after_generation", generation_active=False)
            # Capture envelope fields before context exit finalizes lock_released_utc.
            envelope = block.record
        # Flat siblings: validity references block_id and never embeds measurement.
        spinup_mask = placement_after_spinup.get("process_affinity_readback")
        matches_request = (
            None
            if requested_affinity is None
            else list(spinup_mask or []) == list(requested_affinity)
        )
        before_ns = None
        after_ns = None
        if envelope.get("canary_pre", {}).get("workers"):
            before_ns = round(
                statistics.median(float(w["runtime_ns"]) for w in envelope["canary_pre"]["workers"])
            )
        if envelope.get("canary_post", {}).get("workers"):
            after_ns = round(
                statistics.median(
                    float(w["runtime_ns"]) for w in envelope["canary_post"]["workers"]
                )
            )
        drift = envelope.get("canary_relative_drift")
        drift_pct = None if drift is None else float(drift) * 100.0
        canary_threshold = float(
            envelope.get(
                "canary_drift_threshold",
                cfg["canary"]["max_relative_drift"],
            )
        )
        canary_admissible = (
            drift is not None
            and float(drift) <= canary_threshold
            and bool(envelope.get("canary_pre", {}).get("affinity_passed", True))
            and bool(envelope.get("canary_post", {}).get("affinity_passed", True))
        )
        page_rates = envelope.get("hard_page_reads_per_s") or []
        page_mean = None
        if page_rates:
            numeric = [float(rate) for rate in page_rates if rate is not None]
            page_mean = statistics.fmean(numeric) if numeric else None
        return build_flat_block_record(
            run_id=run_id,
            block_id=block_id,
            cell_id=str(cell["id"]),
            repeat_index=repeat_index,
            measurement={
                "R_prefill": {"run_id": run_id, "value": r_prefill},
                "R_decode": {"run_id": run_id, "value": r_decode},
                "wall_ns": {"run_id": run_id, "value": round(float(result.wall_ns))},
                "ttft_ns": {"run_id": run_id, "value": round(float(result.ttft_ns or 0))},
                "r_prefill_tok_s": r_prefill,
                "r_decode_tok_s": r_decode,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "ttft_source": result.extra.get("ttft_source"),
                "frequency": sampler.summary(),
                "frequency_sample_count": len(frequency_samples),
                "cell_label": str(cell["label"]),
                "burn": bool(cell["burn"]),
            },
            placement={
                "affinity_requested": requested_affinity,
                "readback_after_spinup": {
                    "process_affinity_mask": spinup_mask,
                    "per_thread": placement_after_spinup.get("per_thread_cpu_placement")
                    or {
                        "available": False,
                        "reason": "unavailable",
                    },
                    "captured_utc": placement_after_spinup.get("captured_utc"),
                    "generation_active": placement_after_spinup.get("generation_active"),
                    "process_thread_count": placement_after_spinup.get("process_thread_count"),
                },
                "readback_mid_generation": {
                    "process_affinity_mask": placement_mid.get("process_affinity_readback"),
                    "per_thread": placement_mid.get("per_thread_cpu_placement")
                    or {
                        "available": False,
                        "reason": "unavailable",
                    },
                    "captured_utc": placement_mid.get("captured_utc"),
                    "generation_active": placement_mid.get("generation_active"),
                    "process_thread_count": placement_mid.get("process_thread_count"),
                },
                "matches_request": matches_request,
                "affinity_application": affinity,
                "before_generation": placement_before,
                "after_generation": placement_after,
            },
            canary={
                "before_ns": {"run_id": run_id, "value": before_ns},
                "after_ns": {"run_id": run_id, "value": after_ns},
                "drift_pct": {"run_id": run_id, "value": drift_pct},
                "admissible": canary_admissible,
                "threshold": canary_threshold,
            },
            telemetry={
                "memory": {
                    "available_memory_mb_before": {
                        "run_id": run_id,
                        "value": envelope.get("available_memory_mb_before"),
                    },
                    "available_memory_mb_after": {
                        "run_id": run_id,
                        "value": envelope.get("available_memory_mb_after"),
                    },
                },
                "paging": {
                    "page_reads_per_sec": {"run_id": run_id, "value": page_mean},
                    "hard_page_reads_samples": {"run_id": run_id, "value": page_rates},
                    "paging_gate": envelope.get("paging_gate"),
                    "memory_pressure": envelope.get("memory_pressure"),
                },
                "cpu": {
                    "cpu_pct_total": {
                        "run_id": run_id,
                        "value": envelope.get("cpu_pct_total"),
                    },
                    "cpu_pct_per_core": {
                        "run_id": run_id,
                        "value": envelope.get("cpu_pct_per_core"),
                    },
                },
                "package_temp": None,
                "lock_acquired_utc": envelope.get("lock_acquired_utc"),
                "lock_released_utc": envelope.get("lock_released_utc"),
                "quiescence": envelope.get("quiescence"),
            },
            admissible=bool(envelope.get("valid", False)),
            reasons=list(envelope.get("invalid_reasons") or []),
            timestamp_utc=str(
                envelope.get("lock_released_utc") or envelope.get("lock_acquired_utc") or _utc_now()
            ),
            extras={
                "affinity_contradiction_finding": bool(affinity.get("contradictory_readback")),
            },
        )
    finally:
        _apply_cell_affinity(None, baseline_affinity=baseline_affinity)


def _single_startup_quiescence(
    root: Path, cfg: dict[str, Any], *, p_cpus: list[int]
) -> dict[str, Any]:
    """Run the one authorized detached quiescence gate, under a short machine lock."""
    record: dict[str, Any] = {
        "lock_resource": str(root / ".locks" / "machine"),
        "lock_wait_s": _wait_for_machine_lock(root, cfg),
    }
    with exclusive(root / ".locks" / "machine"):
        record["lock_acquired_utc"] = _utc_now()
        load = measure_quiescence(
            window_s=float(cfg["quiescence"]["window_s"]),
            sample_interval_s=float(cfg["quiescence"]["sample_interval_s"]),
            p_cpus=p_cpus,
            total_cpu_max_pct=float(cfg["quiescence"]["total_cpu_max_pct"]),
            p_core_cpu_max_pct=float(cfg["quiescence"]["p_core_cpu_max_pct"]),
            available_memory_min_mb=float(cfg["quiescence"]["available_memory_min_mb"]),
        )
        record["quiescence"] = load
    record["lock_released_utc"] = _utc_now()
    if not load["passed"]:
        raise MeasurementRefusalError(
            "single detached startup quiescence refusal: " + "; ".join(load["failures"]),
            record=record,
        )
    return record


def _collect_placement_sweep(
    root: Path,
    cfg: dict[str, Any],
    resolved: Any,
    *,
    startup_quiescence: dict[str, Any],
    run_id: str,
    run_dir: Any,
    status_path: Path | None = None,
) -> dict[str, Any]:
    """Collect randomized placement cells with an independent lock for every timed block."""
    baseline_affinity = capture_process_affinity()
    if not baseline_affinity:
        raise SeamError("Prompt A placement sweep requires process-affinity readback")
    backend, spec = _build_backend(root, cfg)
    preflight = backend.preflight()
    if preflight.status != "OK":
        raise SeamError(f"Prompt A backend preflight {preflight.status}: {preflight.reason}")
    power = capture_power_state()
    if power.on_battery is not False:
        raise SeamError("Prompt A requires AC; refusing battery measurement")
    battery = capture_battery_status_wmi()
    ac_profile = (resolved.get("power.profiles") or {}).get("ac-pinned") or {}
    charging_complete, charging_reason = is_charging_complete(
        power,
        battery,
        charge_rate_max_mw=ac_profile.get("charge_rate_max_mw"),
        charging_complete_soc_pct=ac_profile.get("charging_complete_soc_pct"),
    )
    charging_complete = bool(charging_complete and power.on_battery is False)
    if not charging_complete:
        raise SeamError(f"Prompt A requires settled AC charging complete: {charging_reason}")
    expected_plan = "ec87a53a-19a6-4f4a-980f-ab27cc929b25"
    if str(power.power_plan_guid).lower() != expected_plan:
        raise SeamError(
            f"Prompt A power plan mismatch: {power.power_plan_guid!r} != {expected_plan}"
        )
    p_cpus = [int(cpu) for cpu in resolved.require("topology.p_cpus")]
    lp_e_cpus = [int(cpu) for cpu in resolved.require("topology.lpe_cpus")]
    if p_cpus != [0, 1, 2, 3] or lp_e_cpus != [4, 5, 6, 7]:
        raise SeamError(
            f"Prompt A expected P=[0,1,2,3], LP-E=[4,5,6,7], read P={p_cpus}, LP-E={lp_e_cpus}"
        )

    request = _probe_request(cfg)
    session_id = str(uuid.uuid4())
    cell_ids = [str(cell["id"]) for cell in cfg["design"]["cells"]]
    repeats_per_cell = int(cfg["design"]["repeats_per_cell"])
    time.sleep(float(cfg["design"]["cooldown_s"]))
    warmup = _measure_generate(
        root=root,
        cfg=cfg,
        backend=backend,
        request=request,
        cell={
            "id": "WARMUP",
            "label": "unrequested_affinity_discarded_warmup",
            "requested_affinity": None,
            "burn": False,
        },
        repeat_index=-1,
        p_cpus=p_cpus,
        baseline_affinity=baseline_affinity,
        startup_quiescence=startup_quiescence,
        run_id=run_id,
    )
    warmup["discarded_warmup"] = True
    append_block_jsonl(run_dir, block_jsonl_record(run_id=run_id, measurement=warmup))
    print(
        json.dumps(
            {
                "event": "prompt_a2.block_completed",
                "run_id": run_id,
                "cell_id": "WARMUP",
                "repeat_index": -1,
                "discarded_warmup": True,
                "valid": warmup["validity"]["admissible"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    time.sleep(float(cfg["design"]["cooldown_s"]))
    cells = {str(cell["id"]): dict(cell) for cell in cfg["design"]["cells"]}
    schedule = [cell_id for cell_id in cells for _ in range(repeats_per_cell)]
    rng = random.Random(int(cfg["design"]["randomization_seed"]))
    rng.shuffle(schedule)
    initial_schedule = list(schedule)
    records: list[dict[str, Any]] = []
    valid_counts = dict.fromkeys(cells, 0)
    attempted_counts = dict.fromkeys(cells, 0)
    position = 0
    while position < len(schedule):
        cell_id = str(schedule[position])
        if records:
            time.sleep(float(cfg["design"]["cooldown_s"]))
        repeat_index = attempted_counts[cell_id]
        attempted_counts[cell_id] += 1
        progress = progress_status(
            cell_id=cell_id,
            cell_ids=cell_ids,
            repeat_index=repeat_index,
            repeats_per_cell=repeats_per_cell,
        )
        if status_path is not None:
            _write_json_locked(
                status_path,
                {
                    "state": "sweep_running",
                    "run_id": run_id,
                    "worker_pid": os.getpid(),
                    **progress,
                    "blocks_completed": len(records),
                    "schedule_position": position,
                },
            )
        print(
            json.dumps(
                {
                    "event": "prompt_a2.block_starting",
                    "run_id": run_id,
                    **progress,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        try:
            record = _measure_generate(
                root=root,
                cfg=cfg,
                backend=backend,
                request=request,
                cell=cells[cell_id],
                repeat_index=repeat_index,
                p_cpus=p_cpus,
                baseline_affinity=baseline_affinity,
                startup_quiescence=startup_quiescence,
                run_id=run_id,
            )
        except MeasurementRefusalError as exc:
            exc.record["sweep_context"] = {
                "cell_id": cell_id,
                "repeat_index": repeat_index,
                "schedule_position": position,
                "initial_randomized_schedule": initial_schedule,
                "executed_records_before_refusal": records,
                "valid_counts_before_refusal": valid_counts,
                "attempted_counts_at_refusal": attempted_counts,
                "progress": progress,
            }
            raise
        append_block_jsonl(run_dir, block_jsonl_record(run_id=run_id, measurement=record))
        print(
            json.dumps(
                {
                    "event": "prompt_a2.block_completed",
                    "run_id": run_id,
                    **progress,
                    "valid": record["validity"]["admissible"],
                    "invalid_reasons": list(record["validity"].get("reasons") or []),
                    "R_prefill": record["measurement"].get("r_prefill_tok_s"),
                    "R_decode": record["measurement"].get("r_decode_tok_s"),
                    "placement_matches_request": record["placement"].get("matches_request"),
                    "canary_admissible": record["canary"].get("admissible"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        records.append(record)
        if record["validity"]["admissible"]:
            valid_counts[cell_id] += 1
        else:
            insertion = rng.randrange(position + 1, len(schedule) + 1)
            schedule.insert(insertion, cell_id)
        position += 1
    return {
        "backend": backend,
        "spec": spec,
        "power": power,
        "charging_complete": {
            "passed": charging_complete,
            "reason": charging_reason,
            "charge_rate_mw": battery.charge_rate_mw,
            "power_online_wmi": battery.power_online,
        },
        "measurement_session_id": session_id,
        "warmup": warmup,
        "initial_schedule": initial_schedule,
        "records": records,
        "valid_counts": valid_counts,
        "attempted_counts": attempted_counts,
        "launch_baseline_affinity": baseline_affinity,
        "p_cpus": p_cpus,
        "lp_e_cpus": lp_e_cpus,
    }


def _classify_placement_verdict(
    *,
    c1_c2_ratios: dict[str, dict[str, Any]],
    c3_c4_ratios: dict[str, dict[str, Any]],
    target_ratio: float,
    placement_mechanism: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Apply predeclared evidence rules without inferring unavailable thread placement."""
    placement_matches_target = all(
        float(row["ci_low"]) <= target_ratio <= float(row["ci_high"])
        for row in c1_c2_ratios.values()
    )
    contention_effect = all(float(row["ci_low"]) > 1.0 for row in c3_c4_ratios.values())
    thread_placement_available = bool(placement_mechanism["thread_placement_available"])
    mechanism = placement_mechanism.get("classification")
    if placement_matches_target:
        verdict = "PLACEMENT"
    elif contention_effect and thread_placement_available and mechanism == "migration":
        verdict = "CONTENTION_DRIVEN_MIGRATION"
    elif contention_effect and thread_placement_available and mechanism == "stable_p_core":
        verdict = "DIRECT_CONTENTION"
    else:
        verdict = "UNEXPLAINED"
    return verdict, {
        "placement_matches_target": placement_matches_target,
        "contention_effect": contention_effect,
        "thread_placement_available": thread_placement_available,
        "placement_mechanism_classification": mechanism,
        "precedence": [
            "PLACEMENT when both C1/C2 endpoint CIs contain the sealed target ratio",
            "CONTENTION_DRIVEN_MIGRATION when both C3/C4 endpoint CIs exclude one and sampled "
            "thread placement directly shows migration",
            "DIRECT_CONTENTION when both C3/C4 endpoint CIs exclude one and sampled thread "
            "placement directly shows stable P-core execution",
            "otherwise UNEXPLAINED",
        ],
    }


def _placement_readback_sample(record: dict[str, Any], key: str) -> dict[str, Any]:
    """Normalize flat Fix-C placement readbacks and legacy stage-keyed samples."""
    placement = record["placement"]
    if key in placement and isinstance(placement[key], dict):
        sample = placement[key]
        if "per_thread" in sample:
            return {
                "process_affinity_readback": sample.get("process_affinity_mask"),
                "per_thread_cpu_placement": sample["per_thread"],
            }
        return sample
    legacy_key = {
        "readback_after_spinup": "after_worker_threads_spin_up",
        "readback_mid_generation": "mid_generation",
    }.get(key, key)
    return placement[legacy_key]


def _placement_mechanism_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize what the primary placement instrument can and cannot establish."""
    c3_c4 = [record for record in records if record["cell_id"] in {"C3", "C4"}]
    samples = [
        _placement_readback_sample(record, stage)
        for record in c3_c4
        for stage in ("readback_after_spinup", "readback_mid_generation")
    ]
    available = bool(samples) and all(
        sample["per_thread_cpu_placement"]["available"] for sample in samples
    )
    return {
        "thread_placement_available": available,
        "classification": None,
        "process_affinity_masks": {
            cell_id: [
                _placement_readback_sample(record, "readback_mid_generation")[
                    "process_affinity_readback"
                ]
                for record in c3_c4
                if record["cell_id"] == cell_id
            ]
            for cell_id in ("C3", "C4")
        },
        "per_thread_unavailable_reasons": sorted(
            {
                str(sample["per_thread_cpu_placement"]["reason"])
                for sample in samples
                if not sample["per_thread_cpu_placement"]["available"]
            }
        ),
        "interpretation": (
            "process masks are observed, not thread execution placement; no migration or direct "
            "contention mechanism is inferred when per-thread processor samples are unavailable"
        ),
    }


def _endpoint_value(record: dict[str, Any], endpoint: str) -> float | None:
    """Read a throughput endpoint from a flat block or legacy nested measurement."""
    measurement = record.get("measurement")
    if isinstance(measurement, dict) and endpoint in measurement:
        value = measurement[endpoint]
        if isinstance(value, dict) and "value" in value:
            return None if value["value"] is None else float(value["value"])
        return None if value is None else float(value)
    value = record.get(endpoint)
    return None if value is None else float(value)


def _record_admissible(record: dict[str, Any]) -> bool:
    validity = record.get("validity") or {}
    if "admissible" in validity:
        return bool(validity["admissible"])
    return bool(validity.get("valid", False))


def _record_invalid_reasons(record: dict[str, Any]) -> list[str]:
    validity = record.get("validity") or {}
    if "reasons" in validity:
        return list(validity.get("reasons") or [])
    return list(validity.get("invalid_reasons") or [])


def _build_placement_report(
    *,
    cfg: dict[str, Any],
    records: list[dict[str, Any]],
    valid_counts: dict[str, int],
    attempted_counts: dict[str, int],
    initial_schedule: list[str],
    session_id: str,
    warmup: dict[str, Any] | None,
    startup_quiescence: dict[str, Any] | None,
    prelaunch_run_id: str | None,
    startup_grace: dict[str, Any],
    audit: dict[str, Any] | None,
    backend: Any | None,
    spec: Any | None,
    power: Any | None,
    charging_complete: dict[str, Any] | None,
    root: Path | None,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Production placement-summary builder (also used for cycle reproduction)."""
    required = int(cfg["design"]["repeats_per_cell"])
    valid_records = [record for record in records if _record_admissible(record)]
    cells: dict[str, Any] = {}
    for cell in cfg["design"]["cells"]:
        cell_id = str(cell["id"])
        rows = [record for record in valid_records if record["cell_id"] == cell_id]
        endpoint_stats: dict[str, Any] = {}
        for endpoint in cfg["design"]["endpoints"]:
            values = [
                value for row in rows if (value := _endpoint_value(row, endpoint)) is not None
            ]
            endpoint_stats[endpoint] = (
                _bootstrap_median(values, cfg=cfg)
                if values
                else {"median": None, "ci_low": None, "ci_high": None, "n": 0}
            )
        cells[cell_id] = {"definition": dict(cell), **endpoint_stats}
    c1_c2_ratios: dict[str, dict[str, Any]] = {}
    c3_c4_ratios: dict[str, dict[str, Any]] = {}
    for endpoint in cfg["design"]["endpoints"]:
        c1 = [
            value
            for row in valid_records
            if row["cell_id"] == "C1" and (value := _endpoint_value(row, endpoint)) is not None
        ]
        c2 = [
            value
            for row in valid_records
            if row["cell_id"] == "C2" and (value := _endpoint_value(row, endpoint)) is not None
        ]
        c3 = [
            value
            for row in valid_records
            if row["cell_id"] == "C3" and (value := _endpoint_value(row, endpoint)) is not None
        ]
        c4 = [
            value
            for row in valid_records
            if row["cell_id"] == "C4" and (value := _endpoint_value(row, endpoint)) is not None
        ]
        empty_ratio = {
            "ratio": None,
            "ci_low": None,
            "ci_high": None,
            "n_num": 0,
            "n_den": 0,
        }
        c1_c2_ratios[endpoint] = (
            _bootstrap_ratio(c1, c2, cfg=cfg)
            if c1 and c2
            else {**empty_ratio, "n_num": len(c1), "n_den": len(c2)}
        )
        c3_c4_ratios[endpoint] = (
            _bootstrap_ratio(c3, c4, cfg=cfg)
            if c3 and c4
            else {**empty_ratio, "n_num": len(c3), "n_den": len(c4)}
        )
    target_ratio = float(cfg["design"]["target_ratio"])
    placement_mechanism = _placement_mechanism_summary(valid_records)
    if all(
        row.get("ci_low") is not None and row.get("ci_high") is not None
        for row in c1_c2_ratios.values()
    ) and all(
        row.get("ci_low") is not None and row.get("ci_high") is not None
        for row in c3_c4_ratios.values()
    ):
        verdict, verdict_evidence = _classify_placement_verdict(
            c1_c2_ratios=c1_c2_ratios,
            c3_c4_ratios=c3_c4_ratios,
            target_ratio=target_ratio,
            placement_mechanism=placement_mechanism,
        )
    else:
        verdict, verdict_evidence = "UNEXPLAINED", {"incomplete_endpoints": True}
    report: dict[str, Any] = {
        "experiment_id": cfg["experiment_id"],
        "measurement_session_id": session_id,
        "prelaunch_quiescence_run_id": prelaunch_run_id,
        "startup_grace": startup_grace,
        "single_startup_quiescence": startup_quiescence,
        "verdict": verdict,
        "verdict_rule": {
            "allowed": [
                "PLACEMENT",
                "CONTENTION_DRIVEN_MIGRATION",
                "DIRECT_CONTENTION",
                "UNEXPLAINED",
            ],
            "evidence": verdict_evidence,
        },
        "design": {
            "experiment": "four-cell placement sweep",
            "cells": cfg["design"]["cells"],
            "initial_randomized_schedule": initial_schedule,
            "executed_schedule": [record["cell_id"] for record in records],
            "valid_counts": valid_counts,
            "attempted_counts": attempted_counts,
            "cooldown_s": cfg["design"]["cooldown_s"],
            "randomization_seed": cfg["design"]["randomization_seed"],
            "required_valid_repeats_per_cell": required,
        },
        "source_manifest_audit": audit,
        "prior_explicit_manifest_equalities": (
            None if audit is None else audit.get("explicit_confirmations")
        ),
        "runtime_environment": _runtime_environment_metadata(),
        "warmup_discarded": warmup,
        "records": records,
        "cells": cells,
        "ratios": {
            "C1_over_C2": {
                "target_ratio": target_ratio,
                "target_ratio_source_run_ids": cfg["design"]["target_ratio_source_runs"],
                "endpoints": c1_c2_ratios,
            },
            "C3_over_C4": {
                "endpoints": c3_c4_ratios,
                "interpretation": "unrequested clean over unrequested P-core burn",
            },
        },
        "placement_mechanism": placement_mechanism,
        "frequency_interpretation": cfg["frequency"]["interpretation"],
        "paging_gate": {
            "gate_mode": cfg.get("paging", {}).get("gate_mode"),
            "hard_page_reads_threshold_per_s": cfg.get("paging", {}).get(
                "hard_page_reads_threshold_per_s"
            ),
            "sustained_nonzero_consecutive_samples": cfg.get("paging", {}).get(
                "sustained_nonzero_consecutive_samples"
            ),
            "available_memory_min_mb": cfg.get("paging", {}).get("available_memory_min_mb"),
            "baseline_run_id": cfg.get("paging", {}).get("baseline_run_id"),
            "idle_p95": cfg.get("paging", {}).get("idle_p95"),
            "margin_above_idle_p95": cfg.get("paging", {}).get("margin_above_idle_p95"),
            "threshold_justification": cfg.get("paging", {}).get("threshold_justification"),
            "manifest_field": (
                "paging_gate - reconstructible absolute threshold for this sealed run; "
                "copied from configs/measurement.yaml at emit time"
            ),
        },
        "excluded_invalid_blocks": [
            {
                "run_id_citation_note": (
                    "invalid blocks remain in records[]; never silently dropped"
                ),
                "block_id": record.get("block_id"),
                "cell_id": record["cell_id"],
                "repeat_index": record["repeat_index"],
                "invalid_reasons": _record_invalid_reasons(record),
            }
            for record in records
            if not _record_admissible(record)
        ],
    }
    if root is not None:
        report["measurement_code_sha256"] = {
            "seam/tools/prompt_a.py": _sha256_path(root / "seam" / "tools" / "prompt_a.py"),
            "seam/measurement.py": _sha256_path(root / "seam" / "measurement.py"),
            "seam/telemetry/frequency.py": _sha256_path(
                root / "seam" / "telemetry" / "frequency.py"
            ),
        }
    if backend is not None:
        report["openvino"] = asdict(runtime_info())
        report["backend_config"] = backend.config_record()
    if power is not None:
        report["power_start"] = asdict(power)
    if charging_complete is not None:
        report["charging_complete"] = charging_complete
    if spec is not None and root is not None:
        report["model_ir_sha256"] = manifest_model_block(
            spec=spec,
            spec_path=root / cfg["openvino"]["model_spec"],
            reasoning_mode="thinking_off",
        )["ir_sha256"]
    if extras:
        report.update(extras)
    return report


def _run_reproduction(
    root: Path,
    cfg: dict[str, Any],
    resolved: Any,
    audit: dict[str, Any],
    *,
    allow_dirty: bool,
    run_id: str | None = None,
    prelaunch_run_id: str | None = None,
    startup_grace_s: float = 0.0,
    status_path: Path | None = None,
) -> tuple[str, dict[str, Any]]:
    if run_id is None:
        run_id = str(uuid.uuid4())
    # Fix 2: exercise the entire output path before model load / first cell.
    dry_run = startup_output_path_dry_run(root=root, cfg=cfg, allow_dirty=allow_dirty)
    print(
        json.dumps(
            {"event": "prompt_a2.startup_output_path_dry_run_passed", "run_id": run_id, **dry_run},
            sort_keys=True,
        ),
        flush=True,
    )
    grace_started_utc = _utc_now()
    grace_deadline_utc = (datetime.now(UTC) + timedelta(seconds=startup_grace_s)).isoformat()
    # Fix 1: create run directory + in-progress marker before model load / first cell.
    run_dir = open_in_progress_run(
        root=root,
        run_id=run_id,
        marker={
            "run_id": run_id,
            "state": "IN_PROGRESS",
            "lifecycle": "PARTIAL/INCOMPLETE until sealed",
            "created_utc": _utc_now(),
            "startup_output_path_dry_run": dry_run,
            "prelaunch_quiescence_run_id": prelaunch_run_id,
            "n_cells": len(cfg["design"]["cells"]),
            "repeats_per_cell": int(cfg["design"]["repeats_per_cell"]),
        },
    )
    time.sleep(startup_grace_s)
    if status_path is not None:
        _write_json_locked(
            status_path,
            {
                "state": "startup_quiescence",
                "run_id": run_id,
                "worker_pid": os.getpid(),
                "grace_started_utc": grace_started_utc,
                "grace_deadline_utc": grace_deadline_utc,
                "raw_run_dir": str(run_dir.path),
            },
        )
    p_cpus = [int(cpu) for cpu in resolved.require("topology.p_cpus")]
    startup_quiescence = _single_startup_quiescence(root, cfg, p_cpus=p_cpus)
    if status_path is not None:
        _write_json_locked(
            status_path,
            {
                "state": "sweep_running",
                "run_id": run_id,
                "worker_pid": os.getpid(),
                "startup_quiescence_passed": True,
                "startup_quiescence_lock_released_utc": startup_quiescence["lock_released_utc"],
                "raw_run_dir": str(run_dir.path),
            },
        )
    collected = _collect_placement_sweep(
        root,
        cfg,
        resolved,
        startup_quiescence=startup_quiescence,
        run_id=run_id,
        run_dir=run_dir,
        status_path=status_path,
    )
    backend = collected["backend"]
    spec = collected["spec"]
    power = collected["power"]
    charging_complete = collected["charging_complete"]
    session_id = collected["measurement_session_id"]
    warmup = collected["warmup"]
    initial_schedule = collected["initial_schedule"]
    records = collected["records"]
    valid_counts = collected["valid_counts"]
    attempted_counts = collected["attempted_counts"]
    required = int(cfg["design"]["repeats_per_cell"])
    if any(count < required for count in valid_counts.values()):
        raise SeamError(f"Prompt A ended below required valid repeats: {valid_counts}")
    report = _build_placement_report(
        cfg=cfg,
        records=records,
        valid_counts=valid_counts,
        attempted_counts=attempted_counts,
        initial_schedule=initial_schedule,
        session_id=session_id,
        warmup=warmup,
        startup_quiescence=startup_quiescence,
        prelaunch_run_id=prelaunch_run_id,
        startup_grace={
            "duration_s": startup_grace_s,
            "started_utc": grace_started_utc,
            "deadline_utc": grace_deadline_utc,
            "machine_lock_held": False,
        },
        audit=audit,
        backend=backend,
        spec=spec,
        power=power,
        charging_complete=charging_complete,
        root=root,
        extras={"startup_output_path_dry_run": dry_run},
    )
    # Fix 3: locate cycles structurally; never default=str/repr.
    assert_acyclic(report, label=f"placement summary run_id={run_id}")
    model_block = manifest_model_block(
        spec=spec,
        spec_path=root / cfg["openvino"]["model_spec"],
        reasoning_mode="thinking_off",
    )
    handle = emit(
        config=resolved,
        target="cpu-placement",
        workload={
            "kind": "prompt_a2_placement_reproduction",
            "benchmark": "fixed_openvino_generation_probe",
            "task_ids": [],
            "seed": int(cfg["design"]["randomization_seed"]),
            "n_repeats": required,
            "concurrency": 1,
        },
        condition_label="prompt_a2_four_cell_placement",
        repo_root=root,
        run_id=run_id,
        existing_run_dir=run_dir,
        allow_dirty=allow_dirty,
        summary=report,
        model=model_block,
        drivers=asdict(runtime_info()),
        power_state=manifest_power_state(power, background_quiesced=True),
        thermal={"regime": "confound", "excluded": False},
        self_check="pass",
    )
    report["run_id"] = handle.run_id
    report["sealed"] = True
    report["seal_raw_sha256"] = handle.raw_sha256
    return handle.run_id, report


def _emit_launch_quiescence(
    root: Path,
    cfg: dict[str, Any],
    resolved: Any,
    audit: dict[str, Any],
    *,
    run_id: str,
    allow_dirty: bool,
) -> tuple[bool, dict[str, Any]]:
    """Seal the unchanged launch gate before any detached measurement process starts."""
    p_cpus = [int(cpu) for cpu in resolved.require("topology.p_cpus")]
    load = measure_quiescence(
        window_s=float(cfg["quiescence"]["window_s"]),
        sample_interval_s=float(cfg["quiescence"]["sample_interval_s"]),
        p_cpus=p_cpus,
        total_cpu_max_pct=float(cfg["quiescence"]["total_cpu_max_pct"]),
        p_core_cpu_max_pct=float(cfg["quiescence"]["p_core_cpu_max_pct"]),
        available_memory_min_mb=float(cfg["quiescence"]["available_memory_min_mb"]),
    )
    power = capture_power_state()
    battery = capture_battery_status_wmi()
    ac_profile = (resolved.get("power.profiles") or {}).get("ac-pinned") or {}
    charging_complete, charging_reason = is_charging_complete(
        power,
        battery,
        charge_rate_max_mw=ac_profile.get("charge_rate_max_mw"),
        charging_complete_soc_pct=ac_profile.get("charging_complete_soc_pct"),
    )
    charging_complete = bool(charging_complete and power.on_battery is False)
    expected_plan = "ec87a53a-19a6-4f4a-980f-ab27cc929b25"
    launch_failures = list(load["failures"])
    if not charging_complete:
        launch_failures.append(f"settled AC charging incomplete: {charging_reason}")
    if str(power.power_plan_guid).lower() != expected_plan:
        launch_failures.append(f"power plan mismatch: {power.power_plan_guid!r} != {expected_plan}")
    passed = not launch_failures
    summary = {
        "experiment_id": cfg["experiment_id"],
        "classification": ("PROMPT_A2_LAUNCH_QUIESCENCE" if passed else "PROMPT_A2_LAUNCH_REFUSAL"),
        "run_id": run_id,
        "passed": passed,
        "failures": launch_failures,
        "quiescence": load,
        "charging_complete": {
            "passed": charging_complete,
            "reason": charging_reason,
            "charge_rate_mw": battery.charge_rate_mw,
            "power_online_wmi": battery.power_online,
        },
        "source_manifest_audit": audit,
        "prior_explicit_manifest_equalities": audit["explicit_confirmations"],
        "runtime_environment": _runtime_environment_metadata(),
    }
    handle = emit(
        config=resolved,
        target="cpu-placement",
        workload={
            "kind": "prompt_a2_preflight" if passed else "prompt_a2_refusal",
            "benchmark": "placement_sweep_launch_quiescence",
            "task_ids": [],
            "seed": int(cfg["design"]["randomization_seed"]),
            "n_repeats": int(cfg["design"]["repeats_per_cell"]),
            "concurrency": 1,
        },
        condition_label=(
            "prompt_a2_placement_launch_pass" if passed else "prompt_a2_placement_launch_refusal"
        ),
        repo_root=root,
        run_id=run_id,
        allow_dirty=allow_dirty,
        summary=summary,
        power_state=manifest_power_state(power, background_quiesced=passed),
        thermal={"regime": "confound", "excluded": not passed},
        self_check="pass" if passed else "fail",
    )
    summary["sealed"] = True
    summary["seal_raw_sha256"] = handle.raw_sha256
    return passed, summary


def _start_heartbeat(
    *,
    path: Path,
    run_id: str,
    status_path: Path | None,
    interval_s: float,
) -> tuple[threading.Event, threading.Thread]:
    stop = threading.Event()

    def write(sequence: int) -> None:
        _write_json_locked(
            path,
            {
                "state": "running",
                "run_id": run_id,
                "worker_pid": os.getpid(),
                "updated_utc": _utc_now(),
                "sequence": sequence,
                "status_path": str(status_path) if status_path is not None else None,
            },
        )

    write(0)

    def loop() -> None:
        sequence = 1
        while not stop.wait(interval_s):
            write(sequence)
            sequence += 1

    thread = threading.Thread(target=loop, name="prompt-a-heartbeat", daemon=True)
    thread.start()
    return stop, thread


def _stop_heartbeat(
    heartbeat: tuple[threading.Event, threading.Thread] | None,
) -> None:
    if heartbeat is None:
        return
    stop, thread = heartbeat
    stop.set()
    thread.join(timeout=5)


def _emit_refusal(
    root: Path,
    cfg: dict[str, Any],
    resolved: Any,
    audit: dict[str, Any],
    *,
    run_id: str,
    error: Exception,
    allow_dirty: bool,
    prelaunch_run_id: str | None = None,
) -> dict[str, Any]:
    measurement_record = error.record if isinstance(error, MeasurementRefusalError) else None
    summary: dict[str, Any] = {
        "experiment_id": cfg["experiment_id"],
        "classification": "PROMPT_A2_REFUSAL",
        "run_id": run_id,
        "prelaunch_quiescence_run_id": prelaunch_run_id,
        "refusal": {
            "exception_type": type(error).__name__,
            "reason": str(error),
            "measurement_taken": False,
        },
        "measurement_preflight_record": measurement_record,
        "source_manifest_audit": audit,
        "runtime_environment": _runtime_environment_metadata(),
    }
    spec = load_local_spec(root / cfg["openvino"]["model_spec"])
    power = capture_power_state()
    handle = emit(
        config=resolved,
        target="cpu-placement",
        workload={
            "kind": "prompt_a2_refusal",
            "benchmark": "fixed_openvino_generation_probe",
            "task_ids": [],
            "seed": int(cfg["design"]["randomization_seed"]),
            "n_repeats": int(cfg["design"]["repeats_per_cell"]),
            "concurrency": 1,
        },
        condition_label="prompt_a2_preflight_refusal",
        repo_root=root,
        run_id=run_id,
        allow_dirty=allow_dirty,
        summary=summary,
        model=manifest_model_block(
            spec=spec,
            spec_path=root / cfg["openvino"]["model_spec"],
            reasoning_mode="thinking_off",
        ),
        drivers=asdict(runtime_info()),
        power_state=manifest_power_state(power, background_quiesced=False),
        thermal={"regime": "confound", "excluded": True},
        self_check="fail",
    )
    summary["sealed"] = True
    summary["seal_raw_sha256"] = handle.raw_sha256
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help=(
            "Deprecated for placement sweep / launch-quiescence: those paths always refuse a "
            "dirty tree. Retained only for --preflight-only diagnostics."
        ),
    )
    parser.add_argument("--diff-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--launch-quiescence-only", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--prelaunch-run-id")
    parser.add_argument("--status-path")
    parser.add_argument("--heartbeat-path")
    parser.add_argument("--startup-grace-s", type=float, default=0.0)
    args = parser.parse_args(argv)
    root = repo_root(Path(__file__).parent)
    cfg, resolved = _load_config(root)
    if args.preflight_only:
        run_id, summary = _emit_preflight(root, cfg, resolved, allow_dirty=bool(args.allow_dirty))
        print(
            json.dumps(
                {
                    "classification": summary["classification"],
                    "run_id": run_id,
                    "sealed": True,
                    "orphan_openvino_verdict": summary["process_check"]["orphan_openvino_verdict"],
                },
                indent=2,
            )
        )
        return 0
    audit = _manifest_audit(root, cfg)
    diff_path = root / cfg["outputs"]["manifest_diff"]
    _write_json_locked(diff_path, audit)
    if audit["configuration_fully_explains_2x"]:
        print(json.dumps({"verdict": "CONFIGURATION", "diff": str(diff_path)}, indent=2))
        return 0
    if args.diff_only:
        print(json.dumps({"verdict": "UNRESOLVED", "diff": str(diff_path)}, indent=2))
        return 0
    if args.launch_quiescence_only:
        if not args.run_id:
            raise SeamError("--launch-quiescence-only requires --run-id")
        if args.allow_dirty:
            raise SeamError(
                "placement sweep launch refuses --allow-dirty; commit a clean tree first"
            )
        launch_run_id = str(uuid.UUID(args.run_id))
        passed, launch = _emit_launch_quiescence(
            root,
            cfg,
            resolved,
            audit,
            run_id=launch_run_id,
            allow_dirty=False,
        )
        launch_path = root / "derived" / "prompt_a" / f"launch_{launch_run_id}.json"
        _write_json_locked(launch_path, launch)
        print(
            json.dumps(
                {
                    "classification": launch["classification"],
                    "passed": passed,
                    "run_id": launch_run_id,
                    "sealed": True,
                    "artifact": str(launch_path),
                },
                indent=2,
            )
        )
        return 0 if passed else 2
    if args.allow_dirty:
        raise SeamError("placement sweep launch refuses --allow-dirty; commit a clean tree first")
    run_id = str(uuid.UUID(args.run_id)) if args.run_id else str(uuid.uuid4())
    status_path = root / args.status_path if args.status_path else None
    heartbeat_path = root / args.heartbeat_path if args.heartbeat_path else None
    started_utc = _utc_now()
    grace_deadline_utc = (
        datetime.now(UTC) + timedelta(seconds=float(args.startup_grace_s))
    ).isoformat()
    if status_path is not None:
        _write_json_locked(
            status_path,
            {
                "state": "startup_grace",
                "run_id": run_id,
                "started_utc": started_utc,
                "startup_grace_s": float(args.startup_grace_s),
                "grace_deadline_utc": grace_deadline_utc,
                "worker_pid": os.getpid(),
                "prelaunch_quiescence_run_id": args.prelaunch_run_id,
                "heartbeat_path": str(heartbeat_path) if heartbeat_path is not None else None,
            },
        )
    heartbeat = (
        _start_heartbeat(
            path=heartbeat_path,
            run_id=run_id,
            status_path=status_path,
            interval_s=float(cfg["design"]["heartbeat_interval_s"]),
        )
        if heartbeat_path is not None
        else None
    )
    try:
        run_id, report = _run_reproduction(
            root,
            cfg,
            resolved,
            audit,
            allow_dirty=False,
            run_id=run_id,
            prelaunch_run_id=args.prelaunch_run_id,
            startup_grace_s=float(args.startup_grace_s),
            status_path=status_path,
        )
    except Exception as exc:
        _stop_heartbeat(heartbeat)
        if (root / "raw" / run_id).exists():
            # Early lifecycle: unsealed dir is PARTIAL/INCOMPLETE evidence, not corrupt.
            if status_path is not None:
                _write_json_locked(
                    status_path,
                    {
                        "state": "PARTIAL/INCOMPLETE",
                        "run_id": run_id,
                        "ended_utc": _utc_now(),
                        "reason": str(exc),
                        "exception_type": type(exc).__name__,
                        "usable_for_final_analysis": False,
                    },
                )
            raise
        refusal = _emit_refusal(
            root,
            cfg,
            resolved,
            audit,
            run_id=run_id,
            error=exc,
            allow_dirty=False,
            prelaunch_run_id=args.prelaunch_run_id,
        )
        refusal_path = root / "derived" / "prompt_a" / f"refusal_{run_id}.json"
        _write_json_locked(refusal_path, refusal)
        if status_path is not None:
            _write_json_locked(
                status_path,
                {
                    "state": "refused",
                    "run_id": run_id,
                    "ended_utc": _utc_now(),
                    "refusal": str(refusal_path),
                    "reason": str(exc),
                },
            )
        if heartbeat_path is not None:
            _write_json_locked(
                heartbeat_path,
                {
                    "state": "refused",
                    "run_id": run_id,
                    "worker_pid": os.getpid(),
                    "updated_utc": _utc_now(),
                    "refusal": str(refusal_path),
                },
            )
        print(
            json.dumps(
                {
                    "classification": "PROMPT_A2_REFUSAL",
                    "run_id": run_id,
                    "sealed": True,
                    "refusal": str(refusal_path),
                },
                indent=2,
            )
        )
        return 2
    _stop_heartbeat(heartbeat)
    report_path = root / cfg["outputs"]["report"]
    _write_json_locked(report_path, report)
    if status_path is not None:
        _write_json_locked(
            status_path,
            {
                "state": "completed",
                "run_id": run_id,
                "ended_utc": _utc_now(),
                "verdict": report["verdict"],
                "report": str(report_path),
            },
        )
    if heartbeat_path is not None:
        _write_json_locked(
            heartbeat_path,
            {
                "state": "completed",
                "run_id": run_id,
                "worker_pid": os.getpid(),
                "updated_utc": _utc_now(),
                "verdict": report["verdict"],
                "report": str(report_path),
            },
        )
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "run_id": run_id,
                "sealed": True,
                "diff": str(diff_path),
                "report": str(report_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    mp.freeze_support()
    raise SystemExit(main())
