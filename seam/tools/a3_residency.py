"""A3 v2 - continuous free-memory residency sweep (no Phase 1, no headroom gate).

One interleaved ladder from launch free memory down to ~1.2 GB. Paging gate is
reporting-only for this run (``paging.exclude_on_failure: false``); default remains
exclusionary elsewhere. Do not launch the long sweep from an agent session - use the
detached PowerShell recipe.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import yaml

from seam.analysis.a3_residency import (
    block_position_slope,
    classify_mechanism_line,
    page_read_distribution,
    phase3_fork_from_top_blocks,
    regression_with_bootstrap_extremes,
)
from seam.backends.local_openvino import LocalOpenVinoBackend, runtime_info
from seam.config import resolve_config
from seam.errors import SeamError
from seam.gitinfo import repo_root
from seam.locks import exclusive
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
from seam.telemetry.frequency import FrequencySampler
from seam.telemetry.memory import sample_memory_pressure_once
from seam.tools.affinity_matrix import check_forbidden_processes
from seam.tools.memory_balloon import ResidentBalloon
from seam.tools.prompt_a import _placement_sample
from seam.tools.prompt_a_lifecycle import (
    append_block_jsonl,
    assert_acyclic,
    build_flat_block_record,
    find_cycles,
    open_in_progress_run,
    startup_output_path_dry_run,
)

__all__ = [
    "build_free_memory_ladder",
    "build_interleaved_schedule",
    "count_ladder_crossings",
    "main",
    "pretouch_mapped_weights",
    "set_process_working_set_size",
]

_CONFIG_PATH = Path("configs/a3_residency.yaml")
_PLATFORM_PATH = Path("configs/platforms/aipc-c1.yaml")
_FAILURE_EVIDENCE_RUN_IDS = (
    "9b25332b-cbb7-453d-99ad-1c9ac67e4b89",
    "cb0ed2e3-ed70-4627-b231-51016d2b255b",
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert_acyclic(payload, label=f"json write {path.name}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_free_memory_ladder(
    launch_free_mb: float,
    *,
    n_steps: int = 6,
    min_free_mb: float = 1200.0,
) -> list[float]:
    """Evenly spaced ladder from launch free memory down to ``min_free_mb`` (inclusive)."""
    if n_steps < 2:
        raise SeamError(f"ladder requires n_steps >= 2; got {n_steps}")
    top = float(launch_free_mb)
    bottom = float(min_free_mb)
    if top < bottom:
        # Machine already below the nominal floor - still emit a ladder (spread may be zero).
        return [top for _ in range(n_steps)]
    span = top - bottom
    return [top - i * span / (n_steps - 1) for i in range(n_steps)]


def count_ladder_crossings(schedule: list[int]) -> int:
    """Count transitions that cross the ladder midpoint (high half ↔ low half)."""
    if len(schedule) < 2:
        return 0
    lo = min(schedule)
    hi = max(schedule)
    if lo == hi:
        return 0
    mid = (lo + hi) / 2.0
    crossings = 0
    for a, b in pairwise(schedule):
        if (a - mid) * (b - mid) < 0:
            crossings += 1
    return crossings


def build_interleaved_schedule(
    n_steps: int,
    repeats: int,
    *,
    seed: int,
    min_crossings: int = 3,
    max_attempts: int = 10_000,
) -> list[int]:
    """Randomized interleaved schedule with each step appearing ``repeats`` times."""
    if n_steps < 2 or repeats < 1:
        raise SeamError("schedule requires n_steps >= 2 and repeats >= 1")
    base = [step for step in range(n_steps) for _ in range(repeats)]
    rng = random.Random(seed)
    for attempt in range(max_attempts):
        schedule = list(base)
        rng.shuffle(schedule)
        if count_ladder_crossings(schedule) >= min_crossings:
            return schedule
        rng = random.Random(seed + attempt + 1)
    raise SeamError(
        f"could not build schedule with >= {min_crossings} midpoint crossings "
        f"in {max_attempts} shuffles"
    )


def pretouch_mapped_weights(model_dir: Path) -> dict[str, Any]:
    """Sequential read of every page under the IR directory (Phase 3 hook)."""
    started = time.perf_counter()
    files_touched = 0
    bytes_touched = 0
    page = 4096
    for path in sorted(model_dir.rglob("*")):
        if not path.is_file():
            continue
        data = path.read_bytes()
        for offset in range(0, len(data), page):
            _ = data[offset]
        files_touched += 1
        bytes_touched += len(data)
    elapsed_s = time.perf_counter() - started
    return {
        "implemented": True,
        "model_dir": str(model_dir),
        "files_touched": files_touched,
        "bytes_touched": bytes_touched,
        "elapsed_s": elapsed_s,
    }


def set_process_working_set_size(min_bytes: int, max_bytes: int) -> dict[str, Any]:
    """Raise the process minimum working set (Phase 3 hook; may require privilege)."""
    import ctypes
    import sys

    if sys.platform != "win32":
        return {
            "implemented": True,
            "granted": False,
            "reason": "SetProcessWorkingSetSize is Windows-only",
            "min_bytes": min_bytes,
            "max_bytes": max_bytes,
        }
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.SetProcessWorkingSetSize.argtypes = [
        wintypes.HANDLE,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    kernel32.SetProcessWorkingSetSize.restype = wintypes.BOOL
    handle = kernel32.GetCurrentProcess()
    ok = bool(
        kernel32.SetProcessWorkingSetSize(
            handle, ctypes.c_size_t(min_bytes), ctypes.c_size_t(max_bytes)
        )
    )
    err = ctypes.get_last_error() if not ok else 0
    return {
        "implemented": True,
        "granted": ok,
        "win32_error": err,
        "min_bytes": min_bytes,
        "max_bytes": max_bytes,
    }


def _load_config(root: Path) -> tuple[dict[str, Any], Any]:
    path = root / _CONFIG_PATH
    a3 = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(a3, dict):
        raise SeamError(f"{path} is not a mapping")
    prompt_path = root / str(a3["prompt_a_config"])
    prompt = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
    if not isinstance(prompt, dict):
        raise SeamError(f"{prompt_path} is not a mapping")
    measurement_path = root / str(a3["measurement_config"])
    measurement = yaml.safe_load(measurement_path.read_text(encoding="utf-8"))
    if not isinstance(measurement, dict):
        raise SeamError(f"{measurement_path} is not a mapping")
    # prompt_a constants + measurement envelope, then A3 overrides (paging exclude, etc.).
    quiescence = {
        **measurement.get("quiescence", {}),
        **prompt.get("quiescence", {}),
        **a3.get("quiescence", {}),
    }
    paging = {**measurement.get("paging", {}), **a3.get("paging", {})}
    cfg = {
        **measurement,
        **prompt,
        **a3,
        "quiescence": quiescence,
        "paging": paging,
        "canary": measurement["canary"],
        "locking": measurement["locking"],
        "frequency": measurement["frequency"],
    }
    resolved = resolve_config(
        [root / _PLATFORM_PATH, measurement_path, prompt_path, path],
        repo_root=root,
    )
    return cfg, resolved


def _paging_override_manifest_note(cfg: dict[str, Any]) -> dict[str, Any]:
    exclude = bool(cfg["paging"].get("exclude_on_failure", True))
    return {
        "PAGING_GATE_OVERRIDE": (
            "REPORTING_ONLY - paging.exclude_on_failure=false for this run; "
            "page-read gate verdict is recorded on every block and NEVER used to "
            "exclude/drop blocks. Default exclude_on_failure=true is restored by "
            "configs/measurement.yaml for all other experiments."
            if not exclude
            else "exclusionary default (exclude_on_failure=true)"
        ),
        "paging_exclude_on_failure": exclude,
        "prominent": True,
    }


def _build_backend(root: Path, cfg: dict[str, Any]) -> tuple[LocalOpenVinoBackend, dict[str, Any]]:
    ov = cfg["openvino"]
    spec = load_local_spec(root / ov["model_spec"])
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


def _probe_request(cfg: dict[str, Any]) -> Any:
    from seam.agent.tools import SYSTEM_PROMPT, TOOL_SPECS
    from seam.backends.base import GenerationRequest

    return GenerationRequest(
        messages=[{"role": "user", "content": str(cfg["probe"]["prompt"])}],
        system=SYSTEM_PROMPT,
        tools=TOOL_SPECS,
        max_tokens=int(cfg["probe"]["max_tokens"]),
        temperature=0.0,
    )


def _single_startup_quiescence(
    root: Path, cfg: dict[str, Any], *, p_cpus: list[int]
) -> dict[str, Any]:
    """Measured quiescence at launch. CPU gates refuse; free-memory headroom does not."""
    record: dict[str, Any] = {"lock_resource": str(root / ".locks" / "machine")}
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
        # Keep thresholds; refuse with process table (actionable).
        raise MeasurementRefusalError(
            "A3 residency startup quiescence refusal: " + "; ".join(load["failures"]),
            record=record,
        )
    return record


def _measure_block(
    *,
    root: Path,
    cfg: dict[str, Any],
    backend: LocalOpenVinoBackend,
    request: Any,
    p_cpus: list[int],
    startup_quiescence: dict[str, Any],
    run_id: str,
    block_position: int,
    ladder_step_index: int,
    ladder_target_mb: float,
    balloon: BalloonFields,
) -> dict[str, Any]:
    sampler = FrequencySampler(interval_s=float(cfg["frequency"]["sample_interval_s"]), n_cpus=8)
    workload_cpus = capture_process_affinity() or list(p_cpus)
    block_id = str(uuid.uuid4())
    with machine_measurement(
        repo_root=root,
        label=f"a3-residency/{ladder_step_index}/{block_position}",
        config=cfg,
        p_cpus=p_cpus,
        workload_cpus=workload_cpus,
        prevalidated_quiescence={
            "passed": True,
            "source": "single_detached_startup_gate",
            "gate_lock_acquired_utc": startup_quiescence["lock_acquired_utc"],
            "gate_lock_released_utc": startup_quiescence["lock_released_utc"],
            "thresholds": startup_quiescence["quiescence"]["thresholds"],
        },
    ) as block:
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
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="a3-residency-gen") as pool:
                future = pool.submit(
                    backend.generate,
                    request,
                    ignore_eos=bool(cfg["probe"]["ignore_eos"]),
                )
                spin_deadline = time.monotonic() + float(cfg["design"]["worker_spinup_timeout_s"])
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
        envelope = block.record

    before_ns = None
    after_ns = None
    if envelope.get("canary_pre", {}).get("workers"):
        before_ns = round(
            statistics.median(float(w["runtime_ns"]) for w in envelope["canary_pre"]["workers"])
        )
    if envelope.get("canary_post", {}).get("workers"):
        after_ns = round(
            statistics.median(float(w["runtime_ns"]) for w in envelope["canary_post"]["workers"])
        )
    drift = envelope.get("canary_relative_drift")
    drift_pct = None if drift is None else float(drift) * 100.0
    canary_threshold = float(
        envelope.get("canary_drift_threshold", cfg["canary"]["max_relative_drift"])
    )
    canary_admissible = (
        drift is not None
        and float(drift) <= canary_threshold
        and bool(envelope.get("canary_pre", {}).get("affinity_passed", True))
        and bool(envelope.get("canary_post", {}).get("affinity_passed", True))
    )
    page_rates = [float(r) for r in (envelope.get("hard_page_reads_per_s") or []) if r is not None]
    threshold = float(cfg["paging"]["hard_page_reads_threshold_per_s"])
    page_dist = page_read_distribution(page_rates, threshold=threshold)
    paging_gate = dict(envelope.get("paging_gate") or {})
    spinup_mask = placement_after_spinup.get("process_affinity_readback")

    return build_flat_block_record(
        run_id=run_id,
        block_id=block_id,
        cell_id=f"L{ladder_step_index}",
        repeat_index=block_position,
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
        },
        placement={
            "affinity_requested": None,
            "readback_after_spinup": {
                "process_affinity_mask": spinup_mask,
                "per_thread": placement_after_spinup.get("per_thread_cpu_placement")
                or {"available": False, "reason": "unavailable"},
                "captured_utc": placement_after_spinup.get("captured_utc"),
                "generation_active": placement_after_spinup.get("generation_active"),
                "process_thread_count": placement_after_spinup.get("process_thread_count"),
            },
            "readback_mid_generation": {
                "process_affinity_mask": placement_mid.get("process_affinity_readback"),
                "per_thread": placement_mid.get("per_thread_cpu_placement")
                or {"available": False, "reason": "unavailable"},
                "captured_utc": placement_mid.get("captured_utc"),
                "generation_active": placement_mid.get("generation_active"),
                "process_thread_count": placement_mid.get("process_thread_count"),
            },
            "matches_request": None,
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
                "hard_page_reads_per_s": page_dist,
                "paging_gate": paging_gate,
                "memory_pressure": envelope.get("memory_pressure"),
            },
            "cpu": {
                "cpu_pct_total": {"run_id": run_id, "value": envelope.get("cpu_pct_total")},
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
        # Validity fields may mark canary/memory issues; analysis still uses ALL blocks.
        admissible=bool(envelope.get("valid", False)),
        reasons=list(envelope.get("invalid_reasons") or []),
        timestamp_utc=str(
            envelope.get("lock_released_utc") or envelope.get("lock_acquired_utc") or _utc_now()
        ),
        extras={
            "block_position": block_position,
            "ladder_step_index": ladder_step_index,
            "free_memory_mb_target": {
                "run_id": run_id,
                "value": ladder_target_mb,
            },
            "free_memory_mb_achieved_before": {
                "run_id": run_id,
                "value": balloon.free_memory_mb_achieved_before,
            },
            "free_memory_mb_achieved_after": {
                "run_id": run_id,
                "value": balloon.free_memory_mb_achieved_after,
            },
            "balloon_bytes_requested": {
                "run_id": run_id,
                "value": balloon.balloon_bytes_requested,
            },
            "balloon_bytes_touched": {
                "run_id": run_id,
                "value": balloon.balloon_bytes_touched,
            },
            "paging_gate": paging_gate,
            "hard_page_reads_per_s": page_dist,
            "cpu_pct_total": envelope.get("cpu_pct_total"),
            "cpu_pct_per_core": envelope.get("cpu_pct_per_core"),
            "analyze_despite_validity": True,
            "analyze_despite_validity_note": (
                "A3 residency analyses every block; canary/memory validity fields are "
                "recorded but never silently drop a block from the regression."
            ),
        },
    )


class BalloonFields:
    __slots__ = (
        "balloon_bytes_requested",
        "balloon_bytes_touched",
        "free_memory_mb_achieved_after",
        "free_memory_mb_achieved_before",
    )

    def __init__(self, result: Any) -> None:
        self.free_memory_mb_achieved_before = result.free_memory_mb_achieved_before
        self.free_memory_mb_achieved_after = result.free_memory_mb_achieved_after
        self.balloon_bytes_requested = result.balloon_bytes_requested
        self.balloon_bytes_touched = result.balloon_bytes_touched


def _analyze_blocks(
    blocks: list[dict[str, Any]], *, cfg: dict[str, Any], run_id: str
) -> dict[str, Any]:
    ladder = cfg["ladder"]
    xs: list[float] = []
    r_decode: list[float] = []
    r_prefill: list[float] = []
    page_medians: list[float] = []
    page_p95s: list[float] = []
    positions: list[int] = []
    decode_by_pos: list[float] = []

    for block in blocks:
        achieved = block["free_memory_mb_achieved_after"]["value"]
        rd = block["measurement"]["R_decode"]["value"]
        rp = block["measurement"]["R_prefill"]["value"]
        page = block["hard_page_reads_per_s"]
        if achieved is None or rd is None:
            continue
        xs.append(float(achieved))
        r_decode.append(float(rd))
        if rp is not None:
            r_prefill.append(float(rp))
        else:
            r_prefill.append(float("nan"))
        page_medians.append(float(page["median"] if page["median"] is not None else 0.0))
        page_p95s.append(float(page["p95"] if page["p95"] is not None else 0.0))
        positions.append(int(block["block_position"]))
        decode_by_pos.append(float(rd))

    # Prefill may contain nan when ttft is zero - drop those pairs for that regression only.
    prefill_pairs = [(x, y) for x, y in zip(xs, r_prefill, strict=True) if y == y]
    prefill_x = [p[0] for p in prefill_pairs]
    prefill_y = [p[1] for p in prefill_pairs]

    resamples = int(ladder["bootstrap_resamples"])
    seed = int(ladder["bootstrap_seed"])
    confidence = float(ladder["confidence"])
    target_ratio = float(ladder["target_ratio"])

    decode_reg = regression_with_bootstrap_extremes(
        xs, r_decode, resamples=resamples, seed=seed, confidence=confidence, label="R_decode"
    )
    prefill_reg = regression_with_bootstrap_extremes(
        prefill_x,
        prefill_y,
        resamples=resamples,
        seed=seed + 1,
        confidence=confidence,
        label="R_prefill",
    )
    page_med_reg = regression_with_bootstrap_extremes(
        xs,
        page_medians,
        resamples=resamples,
        seed=seed + 2,
        confidence=confidence,
        label="hard_page_reads_median",
    )
    page_p95_reg = regression_with_bootstrap_extremes(
        xs,
        page_p95s,
        resamples=resamples,
        seed=seed + 3,
        confidence=confidence,
        label="hard_page_reads_p95",
    )
    drift = block_position_slope(positions, decode_by_pos)
    extremes_ratio = decode_reg["bootstrap_extremes"]["extremes_ratio"]["point"]
    mechanism = classify_mechanism_line(
        r_decode_slope=float(decode_reg["ols"]["slope"]),
        page_read_slope=float(page_med_reg["ols"]["slope"]),
        extremes_ratio=float(extremes_ratio) if extremes_ratio == extremes_ratio else float("nan"),
        target_ratio=target_ratio,
    )
    return {
        "run_id": run_id,
        "n_blocks_analyzed": len(blocks),
        "n_blocks_in_regression": len(xs),
        "all_blocks_analyzed": True,
        "no_silent_drops": True,
        "regressions": {
            "R_decode": decode_reg,
            "R_prefill": prefill_reg,
            "hard_page_reads_median": page_med_reg,
            "hard_page_reads_p95": page_p95_reg,
        },
        "extremes_ratio_vs_target": {
            "run_id": run_id,
            "fitted_extremes_ratio": extremes_ratio,
            "target_ratio": target_ratio,
            "target_ratio_source_runs": list(ladder["target_ratio_source_runs"]),
            "bootstrap_ci": decode_reg["bootstrap_extremes"]["extremes_ratio"],
        },
        "block_position_drift": drift,
        "mechanism_line": mechanism,
        "mechanism_line_note": "paging, placement, both, or unexplained",
    }


def _run_phase3_if_indicated(
    *,
    cfg: dict[str, Any],
    backend: LocalOpenVinoBackend,
    spec: dict[str, Any],
    root: Path,
    request: Any,
    p_cpus: list[int],
    startup_quiescence: dict[str, Any],
    run_id: str,
    run_dir: Any,
    blocks: list[dict[str, Any]],
    ladder: list[float],
    balloon: ResidentBalloon,
    block_position_start: int,
) -> dict[str, Any]:
    phase3_cfg = cfg["phase3"]
    top_idx = 0
    top_blocks = [b for b in blocks if int(b["ladder_step_index"]) == top_idx]
    fork = phase3_fork_from_top_blocks(
        top_blocks,
        threshold=float(phase3_cfg["page_read_threshold_per_s"]),
        sustained_consecutive_samples=int(phase3_cfg["sustained_consecutive_samples"]),
    )
    result: dict[str, Any] = {
        "auto_enabled": bool(phase3_cfg["enabled_auto"]),
        "fork": fork,
        "ran": False,
    }
    if not phase3_cfg["enabled_auto"] or not fork["phase3_indicated"]:
        result["skipped_reason"] = (
            "fork criterion not met" if not fork["phase3_indicated"] else "auto disabled"
        )
        return result

    pretouch = (
        pretouch_mapped_weights(Path(spec["ir_dir"]))
        if phase3_cfg["pretouch_weights"]
        else {"implemented": True, "skipped": True}
    )
    working_set = (
        set_process_working_set_size(
            min_bytes=int(2.5 * 1024**3),
            max_bytes=(8 * 1024**3),
        )
        if phase3_cfg["set_working_set_size"]
        else {"implemented": True, "skipped": True}
    )
    bottom_target = ladder[-1]
    balloon_result = balloon.set_free_memory_mb(bottom_target)
    forced = _measure_block(
        root=root,
        cfg=cfg,
        backend=backend,
        request=request,
        p_cpus=p_cpus,
        startup_quiescence=startup_quiescence,
        run_id=run_id,
        block_position=block_position_start,
        ladder_step_index=len(ladder) - 1,
        ladder_target_mb=bottom_target,
        balloon=BalloonFields(balloon_result),
    )
    forced["phase3_forced_residency"] = True
    append_block_jsonl(run_dir, forced)
    blocks.append(forced)
    result.update(
        {
            "ran": True,
            "pretouch": pretouch,
            "working_set": working_set,
            "forced_block_id": forced["block_id"],
            "forced_R_decode": forced["measurement"]["R_decode"],
        }
    )
    return result


def _emit_refusal(
    *,
    root: Path,
    resolved: Any,
    run_id: str,
    allow_dirty: bool,
    summary: dict[str, Any],
    run_dir: Any | None = None,
) -> str:
    power = capture_power_state()
    handle = emit(
        config=resolved,
        target="cpu-p",
        workload={
            "kind": "a3_residency_refusal",
            "benchmark": "a3_residency_sweep_v2",
            "task_ids": [],
            "seed": None,
            "n_repeats": 1,
            "concurrency": 1,
        },
        condition_label="a3_residency_refusal",
        repo_root=root,
        run_id=run_id,
        existing_run_dir=run_dir,
        allow_dirty=allow_dirty,
        summary=summary,
        power_state=manifest_power_state(power, background_quiesced=False),
        thermal={"regime": "confound", "excluded": False},
        self_check="fail",
    )
    return handle.run_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--startup-grace-s", type=float, default=120.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Exercise output-path dry-run, ladder, schedule, balloon touch; no model/sweep",
    )
    parser.add_argument(
        "--skip-phase3",
        action="store_true",
        help="Never auto-run Phase 3 even if fork criterion is met",
    )
    args = parser.parse_args(argv)

    root = repo_root(Path(__file__).parent)
    cfg, resolved = _load_config(root)
    if not args.allow_dirty and not args.dry_run:
        raise SeamError(
            "A3 residency currently requires --allow-dirty until AWAITING_COMMIT lands; "
            "dirty-tree allowance is recorded prominently in the report"
        )
    if args.skip_phase3:
        cfg = {**cfg, "phase3": {**cfg["phase3"], "enabled_auto": False}}

    run_id = str(uuid.UUID(args.run_id)) if args.run_id else str(uuid.uuid4())
    derived = root / "derived" / "prompt_a"
    status_path = derived / f"status_a3_residency_{run_id}.json"
    report_path = root / str(cfg["outputs"]["report"]).replace(
        "a3_residency_report.json", f"a3_residency_report_{run_id}.json"
    )

    # --- dry-run path (no long sweep) ---
    if args.dry_run:
        dry = startup_output_path_dry_run(root=root, cfg=cfg, allow_dirty=True)
        launch_free = float(sample_memory_pressure_once().available_memory_mb)
        ladder = build_free_memory_ladder(
            launch_free,
            n_steps=int(cfg["ladder"]["n_steps"]),
            min_free_mb=float(cfg["ladder"]["min_free_mb"]),
        )
        schedule = build_interleaved_schedule(
            int(cfg["ladder"]["n_steps"]),
            int(cfg["ladder"]["repeats"]),
            seed=int(cfg["ladder"]["randomization_seed"]),
            min_crossings=int(cfg["ladder"]["min_crossings"]),
        )
        from seam.tools.memory_balloon import touch_allocate

        _buf, touched = touch_allocate(16 * 4096)
        payload = {
            "dry_run": True,
            "run_id": run_id,
            "startup_output_path_dry_run": dry,
            "launch_free_memory_mb": launch_free,
            "ladder_targets_mb": ladder,
            "schedule": schedule,
            "schedule_crossings": count_ladder_crossings(schedule),
            "balloon_touch_smoke_bytes": touched,
            "paging_override": _paging_override_manifest_note(cfg),
            "failure_evidence_policy": cfg["failure_evidence_policy"],
            "phase1_deleted": True,
            "headroom_gate_deleted": True,
            "find_cycles": find_cycles({"ladder": ladder, "schedule": schedule}),
        }
        assert_acyclic(payload, label="a3 residency dry-run")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    # Print run_id and free memory BEFORE sweep - visible on mis-launch.
    launch_free = float(sample_memory_pressure_once().available_memory_mb)
    print(f"run_id={run_id}", flush=True)
    print(f"launch_free_memory_mb={launch_free:.3f}", flush=True)
    print(
        "NOTE: no headroom gate; Phase 1 deleted; paging gate reporting-only "
        f"(exclude_on_failure={cfg['paging'].get('exclude_on_failure')})",
        flush=True,
    )

    dry = startup_output_path_dry_run(root=root, cfg=cfg, allow_dirty=True)
    print(
        json.dumps(
            {"event": "a3_residency.startup_output_path_dry_run_passed", "run_id": run_id, **dry},
            sort_keys=True,
        ),
        flush=True,
    )

    run_dir = open_in_progress_run(
        root=root,
        run_id=run_id,
        marker={
            "run_id": run_id,
            "state": "IN_PROGRESS",
            "lifecycle": "PARTIAL/INCOMPLETE until sealed",
            "created_utc": _utc_now(),
            "experiment": "a3_residency_sweep_v2",
            "phase1_deleted": True,
            "headroom_gate_deleted": True,
            "paging_override": _paging_override_manifest_note(cfg),
            "dirty_tree_allowed": True,
            "dirty_tree_note": cfg["dirty_tree"]["note"],
            "failure_evidence_policy": cfg["failure_evidence_policy"],
            "startup_output_path_dry_run": dry,
            "launch_free_memory_mb": launch_free,
        },
    )
    _write_json(
        status_path,
        {
            "run_id": run_id,
            "state": "startup_grace",
            "launch_free_memory_mb": launch_free,
            "worker_pid": os.getpid(),
            "raw_run_dir": str(run_dir.path),
        },
    )

    if float(args.startup_grace_s) > 0:
        time.sleep(float(args.startup_grace_s))

    # Re-sample after grace for the ladder top actually used.
    launch_free = float(sample_memory_pressure_once().available_memory_mb)
    print(f"run_id={run_id}", flush=True)
    print(f"launch_free_memory_mb={launch_free:.3f}", flush=True)

    ladder = build_free_memory_ladder(
        launch_free,
        n_steps=int(cfg["ladder"]["n_steps"]),
        min_free_mb=float(cfg["ladder"]["min_free_mb"]),
    )
    schedule = build_interleaved_schedule(
        int(cfg["ladder"]["n_steps"]),
        int(cfg["ladder"]["repeats"]),
        seed=int(cfg["ladder"]["randomization_seed"]),
        min_crossings=int(cfg["ladder"]["min_crossings"]),
    )
    print(
        json.dumps(
            {
                "event": "a3_residency.ladder_ready",
                "run_id": run_id,
                "ladder_targets_mb": ladder,
                "schedule": schedule,
                "crossings": count_ladder_crossings(schedule),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    p_cpus = [int(cpu) for cpu in resolved.require("topology.p_cpus")]
    try:
        startup_quiescence = _single_startup_quiescence(root, cfg, p_cpus=p_cpus)
    except MeasurementRefusalError as exc:
        summary = {
            "experiment_id": cfg["experiment_id"],
            "classification": "A3_RESIDENCY_REFUSAL",
            "run_id": run_id,
            "reason": str(exc),
            "refusal_record": exc.record,
            "phase1_deleted": True,
            "headroom_gate_deleted": True,
            "paging_override": _paging_override_manifest_note(cfg),
            "dirty_tree": cfg["dirty_tree"],
            "failure_evidence_policy": cfg["failure_evidence_policy"],
            "launch_free_memory_mb": launch_free,
        }
        _emit_refusal(
            root=root,
            resolved=resolved,
            run_id=run_id,
            allow_dirty=True,
            summary=summary,
            run_dir=run_dir,
        )
        _write_json(report_path, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 2

    power = capture_power_state()
    if power.on_battery is not False:
        raise SeamError("A3 residency requires AC; refusing battery measurement")
    battery = capture_battery_status_wmi()
    ac_profile = (resolved.get("power.profiles") or {}).get("ac-pinned") or {}
    charging_complete, charging_reason = is_charging_complete(
        power,
        battery,
        charge_rate_max_mw=ac_profile.get("charge_rate_max_mw"),
        charging_complete_soc_pct=ac_profile.get("charging_complete_soc_pct"),
    )
    if not (charging_complete and power.on_battery is False):
        raise SeamError(f"A3 residency requires settled AC charging complete: {charging_reason}")
    expected_plan = "ec87a53a-19a6-4f4a-980f-ab27cc929b25"
    if str(power.power_plan_guid).lower() != expected_plan:
        raise SeamError(
            f"A3 residency power plan mismatch: {power.power_plan_guid!r} != {expected_plan}"
        )

    backend, spec = _build_backend(root, cfg)
    preflight = backend.preflight()
    if preflight.status != "OK":
        raise SeamError(f"A3 residency preflight {preflight.status}: {preflight.reason}")
    request = _probe_request(cfg)

    # Warmup discard (same as prompt_a), not part of the ladder regression.
    if bool(cfg["probe"].get("discard_first_generation", True)):
        with ResidentBalloon() as warm_balloon:
            warm_balloon.set_free_memory_mb(ladder[0])
            _ = backend.generate(request, ignore_eos=bool(cfg["probe"]["ignore_eos"]))

    blocks: list[dict[str, Any]] = []
    cooldown = float(cfg["ladder"]["cooldown_s"])
    with ResidentBalloon() as balloon:
        for block_position, step_index in enumerate(schedule):
            target_mb = ladder[step_index]
            balloon_result = balloon.set_free_memory_mb(target_mb)
            print(
                json.dumps(
                    {
                        "event": "a3_residency.block_start",
                        "run_id": run_id,
                        "block_position": block_position,
                        "ladder_step_index": step_index,
                        "free_memory_mb_target": target_mb,
                        "free_memory_mb_achieved_after": (
                            balloon_result.free_memory_mb_achieved_after
                        ),
                        "balloon_bytes_touched": balloon_result.balloon_bytes_touched,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            record = _measure_block(
                root=root,
                cfg=cfg,
                backend=backend,
                request=request,
                p_cpus=p_cpus,
                startup_quiescence=startup_quiescence,
                run_id=run_id,
                block_position=block_position,
                ladder_step_index=step_index,
                ladder_target_mb=target_mb,
                balloon=BalloonFields(balloon_result),
            )
            append_block_jsonl(run_dir, record)
            blocks.append(record)
            _write_json(
                status_path,
                {
                    "run_id": run_id,
                    "state": "sweep_running",
                    "blocks_completed": len(blocks),
                    "blocks_total": len(schedule),
                    "worker_pid": os.getpid(),
                },
            )
            if cooldown > 0 and block_position + 1 < len(schedule):
                time.sleep(cooldown)

        phase3 = _run_phase3_if_indicated(
            cfg=cfg,
            backend=backend,
            spec=spec,
            root=root,
            request=request,
            p_cpus=p_cpus,
            startup_quiescence=startup_quiescence,
            run_id=run_id,
            run_dir=run_dir,
            blocks=blocks,
            ladder=ladder,
            balloon=balloon,
            block_position_start=len(schedule),
        )

    analysis = _analyze_blocks(blocks, cfg=cfg, run_id=run_id)
    summary: dict[str, Any] = {
        "experiment_id": cfg["experiment_id"],
        "classification": "A3_RESIDENCY_SWEEP_V2",
        "run_id": run_id,
        "phase1_deleted": True,
        "headroom_gate_deleted": True,
        "no_8192_mb_refusal": True,
        "paging_override": _paging_override_manifest_note(cfg),
        "dirty_tree": {
            **cfg["dirty_tree"],
            "prominent": True,
            "AWAITING_COMMIT": True,
        },
        "failure_evidence_policy": {
            **cfg["failure_evidence_policy"],
            "retained_run_ids": list(_FAILURE_EVIDENCE_RUN_IDS),
            "resolved_inconsistency": (
                "Both raw/9b25332b and raw/cb0ed2e3 retained as unsealed failure "
                "evidence; neither deleted, staged, or committed."
            ),
        },
        "launch_free_memory_mb": {"run_id": run_id, "value": launch_free},
        "ladder_targets_mb": {"run_id": run_id, "value": ladder},
        "schedule": {"run_id": run_id, "value": schedule},
        "schedule_crossings": count_ladder_crossings(schedule),
        "startup_quiescence": startup_quiescence,
        "startup_output_path_dry_run": dry,
        "blocks": blocks,
        "n_blocks": len(blocks),
        "phase3": phase3,
        "analysis": analysis,
        "mechanism_line": analysis["mechanism_line"],
        "openvino": asdict(runtime_info()),
        "backend_config": backend.config_record(),
        "power_start": asdict(power),
    }
    assert_acyclic(summary, label=f"a3 residency summary run_id={run_id}")
    model_block = manifest_model_block(
        spec=spec,
        spec_path=root / cfg["openvino"]["model_spec"],
        reasoning_mode="thinking_off",
    )
    handle = emit(
        config=resolved,
        target="cpu-p",
        workload={
            "kind": "a3_residency_sweep",
            "benchmark": "free_memory_ladder_residency_v2",
            "task_ids": [],
            "seed": int(cfg["ladder"]["randomization_seed"]),
            "n_repeats": int(cfg["ladder"]["repeats"]),
            "concurrency": 1,
        },
        condition_label=(
            "a3_residency_sweep_v2|PAGING_GATE_REPORTING_ONLY|exclude_on_failure=false"
        ),
        repo_root=root,
        run_id=run_id,
        existing_run_dir=run_dir,
        allow_dirty=True,
        summary=summary,
        model=model_block,
        drivers=asdict(runtime_info()),
        power_state=manifest_power_state(power, background_quiesced=True),
        thermal={"regime": "confound", "excluded": False},
        self_check="pass",
    )
    summary["sealed"] = True
    summary["seal_raw_sha256"] = handle.raw_sha256
    _write_json(report_path, summary)
    _write_json(
        status_path,
        {
            "run_id": handle.run_id,
            "state": "complete",
            "mechanism_line": analysis["mechanism_line"],
            "report_path": str(report_path),
            "sealed": True,
        },
    )
    print(
        json.dumps(
            {
                "run_id": handle.run_id,
                "sealed": True,
                "mechanism_line": analysis["mechanism_line"],
                "extremes_ratio": analysis["extremes_ratio_vs_target"],
                "paging_override": _paging_override_manifest_note(cfg),
                "report_path": str(report_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
