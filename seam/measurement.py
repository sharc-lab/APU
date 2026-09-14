"""Common validity envelope for every machine-timed measurement block.

The machine lock protects the measurement, not analysis.  A block therefore performs, in order:
measured-load quiescence, a fixed-work canary, the caller's timed work, a second canary, and lock
release.  The returned mutable record is complete only after the context exits.
"""

from __future__ import annotations

import hashlib
import os
import statistics
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from seam.errors import SeamError
from seam.locks import exclusive, read_lock_record

__all__ = [
    "MeasurementBlock",
    "MeasurementRefusalError",
    "capture_process_affinity",
    "machine_measurement",
    "measure_quiescence",
    "run_compute_canary",
]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def capture_process_affinity() -> list[int] | None:
    """Read the current process affinity; return null only when the OS cannot expose it."""
    try:
        import psutil

        return [int(cpu) for cpu in psutil.Process().cpu_affinity()]
    except (ImportError, AttributeError, OSError):
        return None


class MeasurementRefusalError(SeamError):
    """Measured-load refusal carrying the complete preflight record for sealing."""

    def __init__(self, message: str, *, record: dict[str, Any]) -> None:
        super().__init__(message)
        self.record = record


def _rank_process_samples(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    normalized: list[dict[str, Any]] = [
        {
            "name": str(row["name"]),
            "pid": int(row["pid"]),
            "sampled_cpu_pct": float(row["sampled_cpu_pct"]),
            "rss_bytes": int(row["rss_bytes"]),
        }
        for row in rows
        if int(row["pid"]) != 0
    ]
    return {
        "top_cpu": sorted(
            normalized,
            key=lambda row: (-row["sampled_cpu_pct"], -row["rss_bytes"], row["pid"]),
        )[:10],
        "top_rss": sorted(
            normalized,
            key=lambda row: (-row["rss_bytes"], -row["sampled_cpu_pct"], row["pid"]),
        )[:10],
    }


def measure_quiescence(
    *,
    window_s: float,
    sample_interval_s: float,
    p_cpus: Sequence[int],
    total_cpu_max_pct: float,
    p_core_cpu_max_pct: float,
    available_memory_min_mb: float,
) -> dict[str, Any]:
    """Measure idle load over a declared window and apply frozen thresholds."""
    import psutil

    if window_s <= 0 or sample_interval_s <= 0:
        raise SeamError("quiescence window and sample interval must be positive")
    tracked_processes: dict[int, Any] = {}
    for process in psutil.process_iter(["pid"]):
        try:
            process.cpu_percent(interval=None)
            tracked_processes[int(process.info["pid"])] = process
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
    psutil.cpu_percent(interval=None, percpu=True)
    started = time.perf_counter()
    samples: list[dict[str, Any]] = []
    while time.perf_counter() - started < window_s:
        remaining = window_s - (time.perf_counter() - started)
        time.sleep(min(sample_interval_s, max(remaining, 0.0)))
        per_cpu = [float(value) for value in psutil.cpu_percent(interval=None, percpu=True)]
        memory = psutil.virtual_memory()
        samples.append(
            {
                "captured_utc": _utc_now(),
                "total_cpu_pct": sum(per_cpu) / len(per_cpu) if per_cpu else None,
                "per_cpu_pct": per_cpu,
                "p_core_cpu_pct": {
                    str(cpu): per_cpu[cpu] if cpu < len(per_cpu) else None for cpu in p_cpus
                },
                "available_memory_mb": float(memory.available) / (1024.0 * 1024.0),
            }
        )
    if not samples:
        raise SeamError("quiescence sampler produced no samples")

    total_values = [float(row["total_cpu_pct"]) for row in samples]
    per_p_core_mean = {
        str(cpu): sum(
            float(row["per_cpu_pct"][cpu]) for row in samples if cpu < len(row["per_cpu_pct"])
        )
        / sum(1 for row in samples if cpu < len(row["per_cpu_pct"]))
        for cpu in p_cpus
    }
    minimum_available = min(float(row["available_memory_mb"]) for row in samples)
    mean_total_cpu_pct = sum(total_values) / len(total_values)
    max_mean_p_core_cpu_pct = max(per_p_core_mean.values()) if per_p_core_mean else None
    aggregate: dict[str, Any] = {
        "mean_total_cpu_pct": mean_total_cpu_pct,
        "max_sample_total_cpu_pct": max(total_values),
        "mean_p_core_cpu_pct": per_p_core_mean,
        "max_mean_p_core_cpu_pct": max_mean_p_core_cpu_pct,
        "min_available_memory_mb": minimum_available,
    }
    failures: list[str] = []
    if mean_total_cpu_pct > total_cpu_max_pct:
        failures.append(f"mean total CPU {mean_total_cpu_pct:.3f}% > {total_cpu_max_pct:.3f}%")
    if max_mean_p_core_cpu_pct is not None and max_mean_p_core_cpu_pct > p_core_cpu_max_pct:
        failures.append(
            f"maximum mean P-core CPU {max_mean_p_core_cpu_pct:.3f}% > {p_core_cpu_max_pct:.3f}%"
        )
    if minimum_available < available_memory_min_mb:
        failures.append(
            f"minimum available memory {minimum_available:.3f} MiB "
            f"< {available_memory_min_mb:.3f} MiB"
        )
    process_tables: dict[str, Any] | None = None
    if failures:
        process_rows: list[dict[str, Any]] = []
        for process in psutil.process_iter(["pid", "name", "memory_info"]):
            try:
                pid = int(process.info["pid"])
                if pid == 0:
                    # Windows' synthetic System Idle Process reports unused CPU as its own CPU
                    # consumption. It is not load and cannot be closed, so including it would
                    # displace an actionable process from the refusal table.
                    continue
                tracked = tracked_processes.get(pid)
                sampled_cpu = (
                    float(tracked.cpu_percent(interval=None)) if tracked is not None else 0.0
                )
                memory_info = process.info["memory_info"]
                process_rows.append(
                    {
                        "name": str(process.info.get("name") or "<unknown>"),
                        "pid": pid,
                        "sampled_cpu_pct": sampled_cpu,
                        "rss_bytes": int(memory_info.rss),
                    }
                )
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
        process_tables = {
            **_rank_process_samples(process_rows),
            "sample_window_s": time.perf_counter() - started,
            "sampled_cpu_pct_basis": (
                "psutil Process.cpu_percent over the measured quiescence window; values are "
                "percent of one logical CPU and may exceed 100 for multithreaded processes"
            ),
        }
    return {
        "window_s": window_s,
        "sample_interval_s": sample_interval_s,
        "thresholds": {
            "total_cpu_max_pct": total_cpu_max_pct,
            "p_core_cpu_max_pct": p_core_cpu_max_pct,
            "available_memory_min_mb": available_memory_min_mb,
        },
        "threshold_basis": (
            "pre-declared before Prompt A reproduction: 20% package mean and 30% per-P-core "
            "mean exclude material background compute while tolerating short Windows service "
            "bursts; 2048 MiB preserves the Phase -1 unified-memory headroom floor"
        ),
        "samples": samples,
        "aggregate": aggregate,
        "passed": not failures,
        "failures": failures,
        "refusal_process_tables": process_tables,
    }


def _canary_worker(cpu: int, iterations: int, seed: int) -> dict[str, Any]:
    import psutil

    process = psutil.Process()
    process.cpu_affinity([cpu])
    affinity = [int(value) for value in process.cpu_affinity()]
    mask = (1 << 64) - 1
    state = (seed ^ ((cpu + 1) * 0x9E3779B97F4A7C15)) & mask
    accumulator = 0
    started = time.perf_counter_ns()
    for index in range(iterations):
        state ^= (state << 13) & mask
        state ^= state >> 7
        state ^= (state << 17) & mask
        state = (state + index * 0xD6E8FEB86659FD93) & mask
        accumulator = (accumulator + (state ^ index)) & mask
    runtime_ns = time.perf_counter_ns() - started
    digest = hashlib.sha256(
        f"{cpu}:{iterations}:{seed}:{state}:{accumulator}".encode("ascii")
    ).hexdigest()
    return {
        "cpu": cpu,
        "affinity_readback": affinity,
        "runtime_ns": runtime_ns,
        "digest": digest,
        "final_state": state,
        "accumulator": accumulator,
    }


def run_compute_canary(
    *,
    cpus: Sequence[int],
    iterations_per_cpu: int,
    seed: int,
) -> dict[str, Any]:
    """Run deterministic fixed integer work once on each declared CPU."""
    from concurrent.futures import ProcessPoolExecutor

    cpu_list = [int(cpu) for cpu in cpus]
    if not cpu_list:
        raise SeamError("compute canary requires at least one CPU")
    if iterations_per_cpu <= 0:
        raise SeamError("compute canary iterations_per_cpu must be positive")
    started_utc = _utc_now()
    started_ns = time.perf_counter_ns()
    with ProcessPoolExecutor(max_workers=len(cpu_list)) as pool:
        futures = [pool.submit(_canary_worker, cpu, iterations_per_cpu, seed) for cpu in cpu_list]
        workers = [future.result() for future in futures]
    wall_ns = time.perf_counter_ns() - started_ns
    workers.sort(key=lambda row: int(row["cpu"]))
    affinity_passed = all(row["affinity_readback"] == [row["cpu"]] for row in workers)
    combined_digest = hashlib.sha256(
        "".join(str(row["digest"]) for row in workers).encode("ascii")
    ).hexdigest()
    return {
        "started_utc": started_utc,
        "finished_utc": _utc_now(),
        "work": {
            "algorithm": "xorshift64-plus-index-mix",
            "iterations_per_cpu": iterations_per_cpu,
            "cpus": cpu_list,
            "seed": seed,
            "result_consumption": "worker states and combined SHA-256 retained in sealed summary",
        },
        "wall_ns": wall_ns,
        "workers": workers,
        "combined_digest": combined_digest,
        "affinity_passed": affinity_passed,
    }


@dataclass(slots=True)
class MeasurementBlock:
    """Mutable record filled as a machine-timed block progresses."""

    record: dict[str, Any]

    def invalidate(self, reason: str) -> None:
        self.record.setdefault("invalid_reasons", []).append(reason)
        self.record["valid"] = False


def _canary_median_runtime_ns(canary_result: Mapping[str, Any]) -> float:
    return statistics.median(float(worker["runtime_ns"]) for worker in canary_result["workers"])


def _relative_drift(pre_runtime: float, post_runtime: float) -> float:
    return abs(post_runtime - pre_runtime) / pre_runtime if pre_runtime else float("inf")


def _apply_post_canaries(
    block: MeasurementBlock,
    *,
    record: dict[str, Any],
    pre: Mapping[str, Any],
    canary: Mapping[str, Any],
    canary_cpus: Sequence[int],
) -> None:
    """Run the after-canary; optionally a second settled sample (C2c).

    When ``canary.settle_s`` > 0 (or ``canary.double_after`` is true with a positive settle),
    sample immediately and again after settle. Settled drift is authoritative for admissibility;
    immediate drift is retained as evidence of transient reclaim. Threshold is never loosened.
    """
    threshold = float(canary["max_relative_drift"])
    settle_s = float(canary.get("settle_s") or 0.0)
    double_after = bool(canary.get("double_after", settle_s > 0.0)) and settle_s > 0.0
    pre_runtime = _canary_median_runtime_ns(pre)
    record["canary_drift_runtime_basis"] = "median per-core fixed-work runtime_ns"
    record["canary_drift_threshold"] = threshold
    baseline_run_id = canary.get("baseline_run_id")
    if baseline_run_id:
        record["canary_drift_threshold_basis"] = (
            f"baseline-relative (C2f): idle canary-pair p95 + margin from "
            f"baseline_run_id={baseline_run_id}; threshold={threshold}"
        )
    else:
        record["canary_drift_threshold_basis"] = (
            "asserted placeholder pending C2f canary baseline "
            f"(max_relative_drift={threshold}); wider than scheduler jitter but not yet "
            "derived from idle pair-drift p95 + margin"
        )
    record["canary_settle_s"] = settle_s if double_after else 0.0
    record["canary_double_after"] = double_after
    record["canary_gate"] = {
        "gate_mode": canary.get("gate_mode", "asserted_pending_baseline"),
        "max_relative_drift": threshold,
        "idle_p95": canary.get("idle_p95"),
        "margin_above_idle_p95": canary.get("margin_above_idle_p95"),
        "baseline_run_id": baseline_run_id,
        "justification": canary.get("threshold_justification"),
        "verdict": "pending",
        "reasons": [],
    }

    canary_reasons: list[str] = []
    immediate = run_compute_canary(
        cpus=canary_cpus,
        iterations_per_cpu=int(canary["iterations_per_cpu"]),
        seed=int(canary["seed"]),
    )
    if not immediate["affinity_passed"]:
        reason = "post-measurement compute canary affinity readback failed"
        canary_reasons.append(reason)
        block.invalidate(reason)
    immediate_drift = _relative_drift(pre_runtime, _canary_median_runtime_ns(immediate))

    if not double_after:
        record["canary_post"] = immediate
        record["canary_relative_drift"] = immediate_drift
        record["canary_drift_authoritative"] = "immediate"
        if immediate_drift > threshold:
            reason = f"compute canary drift {immediate_drift:.6f} > {threshold:.6f}"
            canary_reasons.append(reason)
            block.invalidate(reason)
        record["canary_gate"]["verdict"] = "fail" if canary_reasons else "pass"
        record["canary_gate"]["reasons"] = list(canary_reasons)
        record["canary_gate"]["authoritative_drift"] = immediate_drift
        record["canary_verdict"] = "pass" if not canary_reasons else "invalidate"
        return

    record["canary_post_immediate"] = immediate
    record["canary_relative_drift_immediate"] = immediate_drift
    time.sleep(settle_s)
    settled = run_compute_canary(
        cpus=canary_cpus,
        iterations_per_cpu=int(canary["iterations_per_cpu"]),
        seed=int(canary["seed"]),
    )
    if not settled["affinity_passed"]:
        reason = "post-measurement settled compute canary affinity readback failed"
        canary_reasons.append(reason)
        block.invalidate(reason)
    settled_drift = _relative_drift(pre_runtime, _canary_median_runtime_ns(settled))
    # Settled sample is the authoritative post-canary for admissibility (C2c).
    record["canary_post"] = settled
    record["canary_post_settled"] = settled
    record["canary_relative_drift"] = settled_drift
    record["canary_relative_drift_settled"] = settled_drift
    record["canary_drift_authoritative"] = "settled"
    immediate_drifted = immediate_drift > threshold
    settled_drifted = settled_drift > threshold
    record["canary_immediate_drifted"] = immediate_drifted
    record["canary_settled_drifted"] = settled_drifted
    if immediate_drifted and settled_drifted:
        reason = (
            f"compute canary drift persistent: immediate={immediate_drift:.6f} "
            f"settled={settled_drift:.6f} > {threshold:.6f}"
        )
        canary_reasons.append(reason)
        block.invalidate(reason)
    elif settled_drifted:
        reason = f"compute canary settled drift {settled_drift:.6f} > {threshold:.6f}"
        canary_reasons.append(reason)
        block.invalidate(reason)
    elif immediate_drifted:
        # Transient reclaim: settled recovered. Do not invalidate on immediate alone.
        record["canary_transient_drift_evidence"] = {
            "immediate_drift": immediate_drift,
            "settled_drift": settled_drift,
            "threshold": threshold,
            "note": (
                "immediate after-canary exceeded threshold but settled sample recovered; "
                "settled is authoritative for admissibility"
            ),
        }
    record["canary_gate"]["verdict"] = "fail" if canary_reasons else "pass"
    record["canary_gate"]["reasons"] = list(canary_reasons)
    record["canary_gate"]["authoritative_drift"] = settled_drift
    record["canary_gate"]["immediate_drift"] = immediate_drift
    record["canary_verdict"] = "pass" if not canary_reasons else "invalidate"


@contextmanager
def machine_measurement(
    *,
    repo_root: Path,
    label: str,
    config: Mapping[str, Any],
    p_cpus: Sequence[int],
    workload_cpus: Sequence[int] | None = None,
    secondary_name_check: Mapping[str, Any] | None = None,
    active_load: Callable[[], AbstractContextManager[Any]] | None = None,
    lock_already_held: bool = False,
    prevalidated_quiescence: Mapping[str, Any] | None = None,
    sample_paging: bool = True,
) -> Iterator[MeasurementBlock]:
    """Hold the global machine lock around validity checks and the caller's timed work.

    When ``sample_paging`` is true (default), :class:`MemoryPressureSampler` runs only for the
    yielded timed block - after the pre-canary and before post-canaries. Setup, teardown, model
    load, warmup generation, and cooldown must pass ``sample_paging=False`` (or run outside this
    context) so page-reads from those phases are never sampled (C2d). Thresholds are unchanged.
    """
    locking = config["locking"]
    quiesce = config["quiescence"]
    canary = config["canary"]
    canary_cpus = list(workload_cpus if workload_cpus is not None else p_cpus)
    lock_resource = repo_root / ".locks" / "machine"
    record: dict[str, Any] = {
        "label": label,
        "resource": str(lock_resource),
        "valid": True,
        "invalid_reasons": [],
        "secondary_process_name_check": dict(secondary_name_check or {}),
        "paging_sampling_enabled": bool(sample_paging),
        "paging_window_start_utc": None,
        "paging_window_end_utc": None,
    }
    block = MeasurementBlock(record)
    lock_path = Path(str(lock_resource) + ".lock")
    wait_started = time.monotonic()
    if lock_already_held:
        owner = read_lock_record(lock_path) if lock_path.is_file() else {}
        if owner.get("pid") != os.getpid():
            raise SeamError(
                "lock_already_held requires the current process to own the machine lock; "
                f"observed {owner!r} at {lock_path}"
            )
    else:
        # Wait only for a live same-boot holder. Cross-boot stale locks are
        # reclaimable by exclusive(); spinning on exists() here hung C-2 for
        # up to wait_timeout_s (7200) without ever reaching reclaim
        # (py-spy: machine_measurement sleep at this poll).
        import socket

        import psutil

        current_boot = datetime.fromtimestamp(int(psutil.boot_time()), UTC).isoformat()
        current_hostname = socket.gethostname()
        while lock_path.exists():
            if time.monotonic() - wait_started >= float(locking["wait_timeout_s"]):
                raise SeamError(
                    f"timed out waiting for machine lock {lock_path}; no measurement was taken"
                )
            owner = read_lock_record(lock_path)
            recorded_hostname = owner.get("hostname")
            recorded_boot = owner.get("boot_time")
            same_host = recorded_hostname == current_hostname
            cross_boot = (
                same_host and isinstance(recorded_boot, str) and recorded_boot != current_boot
            )
            if cross_boot:
                break
            time.sleep(float(locking["poll_interval_s"]))
    record["lock_wait_s"] = time.monotonic() - wait_started
    from contextlib import nullcontext

    lock_context = nullcontext(None) if lock_already_held else exclusive(lock_resource)
    record["lock_mode"] = (
        "preheld_experiment_reservation" if lock_already_held else "block_acquisition"
    )
    try:
        with lock_context:
            record["lock_acquired_utc"] = _utc_now()
            load = (
                dict(prevalidated_quiescence)
                if prevalidated_quiescence is not None
                else measure_quiescence(
                    window_s=float(quiesce["window_s"]),
                    sample_interval_s=float(quiesce["sample_interval_s"]),
                    p_cpus=p_cpus,
                    total_cpu_max_pct=float(quiesce["total_cpu_max_pct"]),
                    p_core_cpu_max_pct=float(quiesce["p_core_cpu_max_pct"]),
                    available_memory_min_mb=float(quiesce["available_memory_min_mb"]),
                )
            )
            record["quiescence"] = load
            if not load["passed"]:
                raise MeasurementRefusalError(
                    "measured-load quiescence refusal: " + "; ".join(load["failures"]),
                    record=record,
                )
            paging = config.get("paging")
            if not isinstance(paging, Mapping):
                raise SeamError(
                    "measurement config missing required paging block "
                    "(available_memory_min_mb + sustained hard page-read rule)"
                )
            from seam.telemetry.memory import MemoryPressureSampler

            gate_mode = str(paging.get("gate_mode", "absolute_zero"))
            page_read_threshold = float(paging.get("hard_page_reads_threshold_per_s", 0.0))
            # Default exclusionary; only an experiment config may set false (A3 residency).
            exclude_on_failure = bool(paging.get("exclude_on_failure", True))
            record["paging_gate"] = {
                "gate_mode": gate_mode,
                "hard_page_reads_threshold_per_s": page_read_threshold,
                "sustained_nonzero_consecutive_samples": int(
                    paging["sustained_nonzero_consecutive_samples"]
                ),
                "available_memory_min_mb": float(paging["available_memory_min_mb"]),
                "baseline_run_id": paging.get("baseline_run_id"),
                "idle_p95": paging.get("idle_p95"),
                "margin_above_idle_p95": paging.get("margin_above_idle_p95"),
                "justification": paging.get("threshold_justification"),
                "exclude_on_failure": exclude_on_failure,
                # Filled after the sampler stops; present so readers always see the keys.
                "verdict": "pending",
                "reasons": [],
            }
            load_context = active_load() if active_load is not None else nullcontext(None)
            with load_context as load_record:
                if load_record is not None:
                    record["active_load"] = load_record
                pre = run_compute_canary(
                    cpus=canary_cpus,
                    iterations_per_cpu=int(canary["iterations_per_cpu"]),
                    seed=int(canary["seed"]),
                )
                record["canary_pre"] = pre
                if not pre["affinity_passed"]:
                    raise SeamError("pre-measurement compute canary affinity readback failed")
                if sample_paging:
                    pressure = MemoryPressureSampler(
                        interval_s=float(paging["sample_interval_s"]),
                        available_memory_min_mb=float(paging["available_memory_min_mb"]),
                        sustained_nonzero_samples=int(
                            paging["sustained_nonzero_consecutive_samples"]
                        ),
                        page_read_threshold=page_read_threshold,
                        gate_mode=gate_mode,
                    )
                    pressure.start()
                    record["paging_window_start_utc"] = _utc_now()
                    try:
                        yield block
                    finally:
                        memory_pressure = pressure.stop()
                        record["paging_window_end_utc"] = _utc_now()
                        record["memory_pressure"] = memory_pressure
                        record["available_memory_mb_before"] = memory_pressure[
                            "available_memory_mb_before"
                        ]
                        record["available_memory_mb_after"] = memory_pressure[
                            "available_memory_mb_after"
                        ]
                        record["hard_page_reads_per_s"] = memory_pressure["hard_page_reads_per_s"]
                        record["package_temp_c"] = None
                        record["package_temp_unavailable_reason"] = (
                            "Platform A package temperature is not exposed on this path; "
                            "recorded null"
                        )
                        if memory_pressure["cpu_pct_total"]:
                            record["cpu_pct_total"] = memory_pressure["cpu_pct_total"][-1]
                            record["cpu_pct_per_core"] = memory_pressure["cpu_pct_per_core"][-1]
                        else:
                            record["cpu_pct_total"] = None
                            record["cpu_pct_per_core"] = None
                        paging_reasons = [
                            reason
                            for reason in memory_pressure["invalid_reasons"]
                            if "page read" in reason.lower()
                        ]
                        other_reasons = [
                            reason
                            for reason in memory_pressure["invalid_reasons"]
                            if "page read" not in reason.lower()
                        ]
                        record["paging_gate"]["verdict"] = "fail" if paging_reasons else "pass"
                        record["paging_gate"]["reasons"] = list(paging_reasons)
                        for reason in other_reasons:
                            block.invalidate(reason)
                        if exclude_on_failure:
                            for reason in paging_reasons:
                                block.invalidate(reason)
                        _apply_post_canaries(
                            block,
                            record=record,
                            pre=pre,
                            canary=canary,
                            canary_cpus=canary_cpus,
                        )
                else:
                    # Warmup / setup / non-measurement: lock+canary only; never sample page reads.
                    record["paging_gate"]["verdict"] = "not_sampled"
                    record["paging_gate"]["reasons"] = []
                    record["memory_pressure"] = {
                        "sampled": False,
                        "reason": (
                            "paging sampler disabled for this block "
                            "(outside timed measurement window; C2d)"
                        ),
                        "invalid_reasons": [],
                        "valid": True,
                    }
                    record["available_memory_mb_before"] = None
                    record["available_memory_mb_after"] = None
                    record["hard_page_reads_per_s"] = None
                    record["package_temp_c"] = None
                    record["package_temp_unavailable_reason"] = (
                        "Platform A package temperature is not exposed on this path; recorded null"
                    )
                    record["cpu_pct_total"] = None
                    record["cpu_pct_per_core"] = None
                    try:
                        yield block
                    finally:
                        _apply_post_canaries(
                            block,
                            record=record,
                            pre=pre,
                            canary=canary,
                            canary_cpus=canary_cpus,
                        )
    finally:
        if "lock_acquired_utc" in record:
            record["lock_released_utc"] = _utc_now()
