"""E-FILTER Stage 1 collection: Arm L, escalation disabled, fully instrumented.

Pre-registration: ``docs/EXPERIMENT_escalation_filter.md`` §2 (Stage 1). This module only
**collects**; the counterfactual replay that produces the result lives in
:mod:`seam.analysis.efilter` and reads the sealed run.

Arm L needs no new code path. Escalation is ``t_pred > deadline_s``, so setting the deadline to
:data:`seam.agent.policy.DEADLINE_DISABLED_S` keeps every step local through exactly the same
harness a hybrid run uses. The cloud backend passed in **raises** rather than no-ops, so a step that
somehow escalated would kill the run instead of producing a hybrid trajectory labelled local-only.

Two modes:

``pilot``
    Discard-first task warmup + ≥5 timed tasks (C2f/C2g). Clears when median ``C_max/C_min ≥ 3.0``
    and the run seals with **zero memory-floor trips**. Paging and canary invalidations are
    recorded per block (``paging_admissible`` / ``canary_admissible``) and are **not** run-fatal
    for clearance when they only contaminate timing (C2e/C2f; wall-clock timeout audit NONE).
    Memory-floor invalidations inside timed blocks still refuse clearance (safety gate; C2g).
    Absolute peak context is reported, never gated. Also probes cache reuse and freezes a
    **constant** ``n_out_pred`` from the measured median completion tokens. C2g adds per-step
    RSS / free-memory series and a peak-memory-vs-context fit on timed tasks.

``full``
    The declared N tasks, randomized order, cooled between tasks, sealed with a manifest.

Everything that could drift silently is read back rather than assumed: the power state, the
charging-complete condition, the escalation-disabled sentinel (per step), the KV cache precision,
and the absence of a cloud credential.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import statistics
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Final

import yaml

from seam.agent.harness import StepRecord, TaskResult, run_task
from seam.agent.policy import (
    DEADLINE_DISABLED_S,
    ThroughputModel,
    assert_escalation_disabled,
)
from seam.agent.steplog import STEPS_FILENAME, StepLogWriter, step_to_record
from seam.agent.tools import SYSTEM_PROMPT, TOOL_SPECS, Workload, load_workload
from seam.backends.base import GenerationRequest
from seam.backends.local_openvino import (
    CONSTRAINED_DECODING_MECHANISM,
    LocalOpenVinoBackend,
    runtime_info,
)
from seam.backends.refusing_cloud import RefusingCloudBackend
from seam.config import load_platform_config
from seam.errors import ConfigError, SeamError
from seam.gitinfo import repo_root
from seam.hashing import sha256_file
from seam.jsonlog import log_event, utc_now_iso
from seam.kvmath import KvGeometry, load_kv_geometry
from seam.measurement import machine_measurement
from seam.model_provenance import load_local_spec, manifest_model_block, quantization_summary
from seam.powerstate import (
    PowerState,
    capture_battery_status_wmi,
    capture_power_state,
    is_charging_complete,
    manifest_power_state,
)
from seam.rawstore import RunDir
from seam.stdio_utf8 import NON_ASCII_PROBE, configure_utf8_stdio
from seam.telemetry.frequency import FrequencySampler
from seam.telemetry.rss import rss_bytes_now
from seam.tools.affinity_matrix import check_forbidden_processes
from seam.tools.prompt_a_lifecycle import (
    _prepare_isolated_emit_root,
    assert_acyclic,
    open_in_progress_run,
)

__all__ = [
    "PILOT_MIN_MEDIAN_CONTEXT_RATIO_DEFAULT",
    "build_efilter_summary",
    "canary_admissibility",
    "classify_invalidation_reason",
    "efilter_output_paths",
    "evaluate_pilot_context_gate",
    "free_memory_mb",
    "main",
    "median_context_ratio",
    "memory_series_summary",
    "nested_key_set",
    "nested_key_set_structure",
    "paging_admissibility",
    "per_task_context_ratio",
    "split_invalidation_reasons",
    "stub_efilter_summary_kwargs",
    "task_memory_instrumentation",
    "timed_block_invalidation_count",
    "timed_block_invalidation_split",
    "wait_free_memory_recovery",
]

#: C2b pilot gate: median per-task ``C_max/C_min`` (min/max of ``context_tokens_by_step``).
PILOT_MIN_MEDIAN_CONTEXT_RATIO_DEFAULT: Final = 3.0
#: Retained name only so older imports fail loudly if still referenced with the 20k semantics.
PILOT_MIN_PEAK_CONTEXT_DEFAULT: Final = PILOT_MIN_MEDIAN_CONTEXT_RATIO_DEFAULT
_TASKS_FILENAME: Final = "tasks.jsonl"
_CLEARANCE_FILENAME: Final = "pilot_context_clearance.json"

#: Cloud credential env var. Read from ``os.environ`` DIRECTLY and never through
#: ``seam.credentials.credential_status``, because that helper calls ``load_dotenv``: checking
#: for a credential with it would load the credential the check exists to rule out.
_CLOUD_CREDENTIAL_ENV: Final = "ANTHROPIC_API_KEY"

#: Fixed probe for the inline throughput baseline. Workload-shaped: the real system prompt and the
#: real tool definitions, so the measured prefill rate is taken on the prefix the run actually pays.
_PROBE_PROMPT: Final = (
    "List the files in /docs, then report how many there are. Use the tools rather than guessing."
)


# ==================================================================================================
# Config
# ==================================================================================================


def _load_cfg(root: Path) -> tuple[dict[str, Any], str]:
    path = root / "configs" / "efilter.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"{path} is not a mapping")
    measurement_path = root / str(data["measurement_config"])
    measurement = yaml.safe_load(measurement_path.read_text(encoding="utf-8"))
    if not isinstance(measurement, dict):
        raise ConfigError(f"{measurement_path} is not a mapping")
    # E-FILTER may tighten canary (settle / double-after) and declare teardown recovery without
    # changing the shared measurement.yaml defaults used by other experiments.
    data = {
        **data,
        **measurement,
        "quiesce": data["quiesce"],
        "quiescence": {**measurement["quiescence"], **data["quiesce"]},
        "canary": {**measurement.get("canary", {}), **(data.get("canary") or {})},
        "paging": {**measurement.get("paging", {}), **(data.get("paging") or {})},
        "teardown": dict(data.get("teardown") or {}),
    }
    return data, sha256_file(path)


def free_memory_mb() -> float:
    """Host available memory in MiB (psutil ``virtual_memory().available``)."""
    import psutil

    return float(psutil.virtual_memory().available) / (1024.0 * 1024.0)


def wait_free_memory_recovery(
    *,
    threshold_mb: float,
    timeout_s: float,
    poll_interval_s: float = 2.0,
    clock: Any = time,
    free_memory_fn: Any = None,
) -> dict[str, Any]:
    """Poll until free memory recovers above ``threshold_mb``, or time out (C2c).

    ``clock`` / ``free_memory_fn`` are injectable for unit tests.
    """
    read_free = free_memory_fn or free_memory_mb
    t0 = clock.monotonic()
    samples: list[dict[str, Any]] = []
    while True:
        free_mb = float(read_free())
        elapsed = float(clock.monotonic() - t0)
        samples.append({"elapsed_s": elapsed, "free_memory_mb": free_mb})
        if free_mb >= threshold_mb:
            return {
                "recovered": True,
                "threshold_mb": float(threshold_mb),
                "free_memory_mb": free_mb,
                "elapsed_s": elapsed,
                "timeout_s": float(timeout_s),
                "samples": samples,
            }
        if elapsed >= float(timeout_s):
            return {
                "recovered": False,
                "threshold_mb": float(threshold_mb),
                "free_memory_mb": free_mb,
                "elapsed_s": elapsed,
                "timeout_s": float(timeout_s),
                "samples": samples,
            }
        clock.sleep(float(poll_interval_s))


def memory_series_summary(per_task: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize per-task free-memory / RSS series and monotonic-decline flags (C2c/C2g)."""

    def _series(key: str) -> list[float | None]:
        out: list[float | None] = []
        for row in per_task:
            value = row.get(key)
            out.append(float(value) if value is not None else None)
        return out

    def _monotonic_non_increasing(values: list[float | None]) -> bool | None:
        present = [v for v in values if v is not None]
        if len(present) < 2:
            return None
        return all(present[i] >= present[i + 1] for i in range(len(present) - 1))

    def _monotonic_non_decreasing(values: list[float | None]) -> bool | None:
        present = [v for v in values if v is not None]
        if len(present) < 2:
            return None
        return all(present[i] <= present[i + 1] for i in range(len(present) - 1))

    free_before = _series("free_memory_mb_before_task")
    free_after = _series("free_memory_mb_after_task")
    free_td = _series("free_memory_mb_after_teardown")
    rss_before = _series("process_rss_before")
    rss_after = _series("process_rss_after")
    free_min = _series("free_memory_mb_min")
    rss_peak = _series("process_rss_peak")
    return {
        "free_memory_mb_before_task": free_before,
        "free_memory_mb_after_task": free_after,
        "free_memory_mb_after_teardown": free_td,
        "free_memory_mb_at_task_start": _series("free_memory_mb_at_task_start"),
        "free_memory_mb_at_task_end": _series("free_memory_mb_at_task_end"),
        "process_rss_before": rss_before,
        "process_rss_after": rss_after,
        "free_memory_mb_min": free_min,
        "process_rss_peak": rss_peak,
        "monotonic_decline_free_memory_mb_before_task": _monotonic_non_increasing(free_before),
        "monotonic_decline_free_memory_mb_after_task": _monotonic_non_increasing(free_after),
        "monotonic_increase_process_rss_after": _monotonic_non_decreasing(rss_after),
        "note": (
            "Monotonic free-memory decline (or RSS rise) across tasks is the accumulation "
            "signature that inter-task teardown is meant to break. C2g adds per-task "
            "free_memory_mb_min / process_rss_peak with step attribution."
        ),
    }


def task_memory_instrumentation(
    result: TaskResult,
    *,
    free_before: float,
    free_after: float,
    rss_before: int,
    rss_after: int,
) -> dict[str, Any]:
    """Per-task C2g memory fields from step RSS / free-memory windows (and task endpoints)."""
    step_memory: list[dict[str, Any]] = []
    free_min: float | None = None
    free_min_step: int | None = None
    free_min_context: int | None = None
    rss_peak: int | None = None
    rss_peak_step: int | None = None

    # Task-boundary free memory participates in the minimum (safety floor can trip outside generate).
    for free_mb, step_idx, ctx in (
        (float(free_before), None, None),
        (float(free_after), None, None),
    ):
        if free_min is None or free_mb < free_min:
            free_min = free_mb
            free_min_step = step_idx
            free_min_context = ctx

    for step in result.steps:
        row = {
            "step_idx": int(step.step_idx),
            "context_tokens_total": int(step.context_tokens_total),
            "rss_before_generate": step.rss_before_generate,
            "rss_peak_during_generate": step.rss_peak_during_generate,
            "rss_after_generate": step.rss_after_generate,
            "free_memory_mb_before_generate": step.free_memory_mb_before_generate,
            "free_memory_mb_min_during_generate": step.free_memory_mb_min_during_generate,
            "free_memory_mb_after_generate": step.free_memory_mb_after_generate,
        }
        step_memory.append(row)
        step_free = step.free_memory_mb_min_during_generate
        if step_free is not None and (free_min is None or float(step_free) < free_min):
            free_min = float(step_free)
            free_min_step = int(step.step_idx)
            free_min_context = int(step.context_tokens_total)
        step_rss = step.rss_peak_during_generate
        if step_rss is None:
            step_rss = step.peak_rss_bytes
        if step_rss is not None and (rss_peak is None or int(step_rss) > rss_peak):
            rss_peak = int(step_rss)
            rss_peak_step = int(step.step_idx)

    if rss_peak is None:
        # No steps (e.g. context_cap before generate): fall back to task-boundary RSS.
        rss_peak = max(int(rss_before), int(rss_after))
        rss_peak_step = None

    return {
        "step_memory": step_memory,
        "free_memory_mb_min": free_min,
        "free_memory_mb_min_step_idx": free_min_step,
        "free_memory_mb_min_context_tokens": free_min_context,
        "process_rss_peak": rss_peak,
        "process_rss_peak_step_idx": rss_peak_step,
        "free_memory_mb_at_task_start": float(free_before),
        "free_memory_mb_at_task_end": float(free_after),
        # Existing C2c names retained as aliases.
        "free_memory_mb_before_task": float(free_before),
        "free_memory_mb_after_task": float(free_after),
        "process_rss_before": int(rss_before),
        "process_rss_after": int(rss_after),
        "peak_rss_bytes": rss_peak,
    }


def _assert_task_list(cfg: dict[str, Any], workload: Workload) -> None:
    pinned = cfg["workload"].get("task_list_sha256")
    if pinned and pinned != workload.sha256:
        raise SystemExit(
            f"task list hash mismatch: config pins {pinned}, file is {workload.sha256}. A task "
            f"list that changed mid-experiment invalidates comparability with everything "
            f"already collected."
        )


def _local_backend(root: Path, cfg: dict[str, Any]) -> LocalOpenVinoBackend:
    spec_path = root / cfg["models"]["local"]["spec"]
    spec = load_local_spec(spec_path)
    ov = cfg["openvino"]
    return LocalOpenVinoBackend(
        model_dir=Path(spec["ir_dir"]),
        target="cpu-p",
        scheduling_core_type=ov["scheduling_core_type"],
        inference_num_threads=int(ov["inference_num_threads"]),
        enable_cpu_pinning=ov["enable_cpu_pinning"],
        model_ref=f"{spec['name']}@{str(spec['revision'])[:12]}+{quantization_summary(spec)}",
        enable_thinking=bool(cfg["enable_thinking"]),
        affinity_cpus=None,
    )


# ==================================================================================================
# Quiescence - every field recorded BY VALUE, and read back
# ==================================================================================================


def _quiesce(cfg: dict[str, Any], platform_cfg: Any) -> dict[str, Any]:
    """Capture and check the pinned conditions. Deviations are returned, not silently tolerated."""
    quiesce_cfg = cfg["quiesce"]
    power = capture_power_state()
    battery = capture_battery_status_wmi()
    ac_profile = (platform_cfg.get("power.profiles") or {}).get("ac-pinned") or {}
    flag_complete, reason = is_charging_complete(
        power,
        battery,
        charge_rate_max_mw=ac_profile.get("charge_rate_max_mw"),
        charging_complete_soc_pct=ac_profile.get("charging_complete_soc_pct"),
    )
    # is_charging_complete answers "is the battery still taking charge", and returns True for
    # `charging is False` - which is also true of a machine running on battery. Its callers gate AC
    # separately, so that is correct there. Recorded unqualified in an E-FILTER manifest it would
    # read as "AC, settled", which is the opposite of a discharging machine. Qualify it here.
    complete = bool(flag_complete and power.on_battery is False)
    if flag_complete and power.on_battery is not False:
        reason = f"{reason} but on_battery={power.on_battery}: discharging, not settled on AC"
    forbidden = check_forbidden_processes(quiesce_cfg["forbidden_processes"])
    credential_present = bool(os.environ.get(_CLOUD_CREDENTIAL_ENV))

    deviations: list[str] = []
    if quiesce_cfg["require_ac_connected"] and power.on_battery is not False:
        deviations.append(f"AC not connected (on_battery={power.on_battery})")
    if quiesce_cfg["require_charging_complete"] and not complete:
        deviations.append(f"charging not complete: {reason}")
    if forbidden:
        deviations.append(f"forbidden processes running: {[h['pattern'] for h in forbidden]}")
    if quiesce_cfg["require_no_cloud_credential"] and credential_present:
        deviations.append(
            f"{_CLOUD_CREDENTIAL_ENV} is present in the environment; this experiment makes no "
            f"cloud calls and should not have a credential loaded at all"
        )
    if power.battery_saver:
        deviations.append("Windows battery saver is engaged, which clamps turbo")

    extras = _quiesce_extras()
    record = {
        "captured_utc": utc_now_iso(),
        "on_battery": power.on_battery,
        "battery_pct": power.battery_pct,
        "charging": power.charging,
        "charging_complete": complete,
        "charging_complete_reason": reason,
        "charging_complete_definition": (
            "powerstate.is_charging_complete() AND on_battery is False; the helper alone returns "
            "True for a discharging machine because its callers gate AC separately"
        ),
        "charge_rate_mw": battery.charge_rate_mw,
        "power_online_wmi": battery.power_online,
        "battery_saver": power.battery_saver,
        "power_plan_name": power.power_plan_name,
        "power_plan_guid": power.power_plan_guid,
        "overlay_guid": power.overlay_guid,
        "forbidden_process_hits": forbidden,
        # Presence only. Never the value, never a prefix, never a hash.
        "cloud_credential_present": credential_present,
        "cloud_credential_check_method": (
            "os.environ direct read; load_dotenv deliberately NOT called"
        ),
        "deviations": deviations,
        **extras,
    }
    log_event(
        "efilter.quiesce",
        severity="error" if deviations else "info",
        message=("quiesce deviations: " + "; ".join(deviations)) if deviations else "quiesce ok",
        **{k: v for k, v in record.items() if k != "forbidden_process_hits"},
    )
    return record


def _quiesce_extras() -> dict[str, Any]:
    """Brightness, Defender, and the temperature source that does not exist on this platform."""
    import subprocess as sp

    out: dict[str, Any] = {
        "display_brightness": None,
        "defender_realtime": None,
        "package_temp_c": None,
        "package_temp_source": "unavailable",
        "package_temp_evidence": None,
    }
    try:
        bright = sp.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance -Namespace root/WMI -ClassName WmiMonitorBrightness "
                "| Select-Object -First 1 -ExpandProperty CurrentBrightness)",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
        if bright.returncode == 0 and bright.stdout.strip():
            out["display_brightness"] = float(bright.stdout.strip())
    except (OSError, sp.SubprocessError, ValueError) as exc:
        out["display_brightness_error"] = type(exc).__name__

    try:
        defender = sp.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-MpPreference).DisableRealtimeMonitoring",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if defender.returncode == 0 and defender.stdout.strip():
            disabled = defender.stdout.strip().lower() in {"true", "1"}
            out["defender_realtime"] = "disabled" if disabled else "enabled"
    except (OSError, sp.SubprocessError) as exc:
        out["defender_realtime_error"] = type(exc).__name__

    # Probed rather than assumed absent, so the null is evidence-backed.
    try:
        probe = sp.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance -Namespace root/WMI -ClassName MSAcpi_ThermalZoneTemperature "
                "| Select-Object -First 1 -ExpandProperty CurrentTemperature",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
        if probe.returncode == 0 and probe.stdout.strip():
            deci_kelvin = float(probe.stdout.strip())
            out["package_temp_c"] = deci_kelvin / 10.0 - 273.15
            out["package_temp_source"] = "MSAcpi_ThermalZoneTemperature"
        else:
            out["package_temp_evidence"] = (probe.stderr or probe.stdout).strip()[:200]
            out["package_temp_source"] = "unavailable_pending_M2.2_M2.3"
    except (OSError, sp.SubprocessError, ValueError) as exc:
        out["package_temp_evidence"] = type(exc).__name__
        out["package_temp_source"] = "unavailable_pending_M2.2_M2.3"
    return out


# ==================================================================================================
# Warmup, throughput baseline, cache probe
# ==================================================================================================


def _probe_request(cfg: dict[str, Any], *, prompt: str = _PROBE_PROMPT) -> GenerationRequest:
    return GenerationRequest(
        messages=[{"role": "user", "content": prompt}],
        system=SYSTEM_PROMPT,
        tools=TOOL_SPECS,
        max_tokens=int(cfg["workload"]["max_tokens"]),
        temperature=0.0,
    )


def _warm_and_measure(
    backend: LocalOpenVinoBackend, cfg: dict[str, Any]
) -> tuple[ThroughputModel, dict[str, Any]]:
    """Warm to steady state and measure prefill/decode throughput on the way.

    The warmup generations are the throughput baseline: measuring separately would either add a
    cold-cache sample or a second warm-up, and the router's ``t_pred`` has to be priced with the
    rate the run actually sustains.
    """
    warmup_s = float(cfg["thermal"]["warmup_s"])
    request = _probe_request(cfg)
    samples: list[dict[str, Any]] = []
    t_start = time.perf_counter()
    while True:
        result = backend.generate(request)
        ttft_s = (result.ttft_ns or 0) / 1e9
        decode_s = max(result.wall_ns / 1e9 - ttft_s, 1e-9)
        samples.append(
            {
                "i": len(samples),
                "elapsed_s": round(time.perf_counter() - t_start, 3),
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "wall_s": result.wall_ns / 1e9,
                "ttft_s": ttft_s,
                "r_prefill_tok_s": (result.prompt_tokens / ttft_s) if ttft_s > 0 else None,
                "r_decode_tok_s": result.completion_tokens / decode_s,
                "ttft_source": result.extra.get("ttft_source"),
            }
        )
        print(
            f"  warmup {len(samples)}: {samples[-1]['elapsed_s']}s elapsed, "
            f"prefill {samples[-1]['r_prefill_tok_s']}, decode {samples[-1]['r_decode_tok_s']}"
        )
        if time.perf_counter() - t_start >= warmup_s:
            break

    # The first generation carries any residual first-call cost; it warms, it does not measure.
    scored = samples[1:] or samples
    prefill = [s["r_prefill_tok_s"] for s in scored if s["r_prefill_tok_s"]]
    decode = [s["r_decode_tok_s"] for s in scored if s["r_decode_tok_s"]]
    if not prefill or not decode:
        raise SeamError(
            "warmup produced no usable throughput samples; t_pred cannot be priced and the replay "
            "grid would be built on a fabricated rate"
        )
    throughput = ThroughputModel(
        target="cpu-p",
        r_prefill_tok_s=statistics.median(prefill),
        r_decode_tok_s=statistics.median(decode),
        measured_by_run_id="inline_warmup",
    )
    record = {
        "warmup_s_requested": warmup_s,
        "warmup_s_actual": samples[-1]["elapsed_s"],
        "n_generations": len(samples),
        "n_scored": len(scored),
        "discarded_first": len(samples) > 1,
        "samples": samples,
        "r_prefill_tok_s": throughput.r_prefill_tok_s,
        "r_decode_tok_s": throughput.r_decode_tok_s,
        "r_prefill_cv": _cv(prefill),
        "r_decode_cv": _cv(decode),
        "measured_by": "inline warmup of this run; recorded in this run's sealed summary",
    }
    return throughput, record


def _cv(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = statistics.fmean(values)
    return statistics.stdev(values) / mean if mean else None


def _cache_probe(backend: LocalOpenVinoBackend, cfg: dict[str, Any]) -> dict[str, Any]:
    """Determine whether the local path reports OR exhibits prefix-cache reuse.

    Two independent questions, and the study needs both answered:

    *Instrumented?* Does the runtime return cache counters at all. A zero ``cached_prompt_tokens``
    from a runtime that has no such counter is not a measurement of "no reuse".

    *Reuse observed?* Three calls, because two cannot answer it. Repeating a prompt and finding the
    second call faster is equally explained by first-call warmup, so the probe adds a
    **length-matched DISTINCT prompt** as a control. Reuse requires the repeat to be much faster
    than the first call *and* much faster than the distinct control; if the repeat and the control
    are alike, the machine simply warmed up and nothing was cached.

    Must run **after** warmup. Run cold, every ratio here measures lazy initialization instead.
    """
    repeated = _probe_request(cfg)
    control = _probe_request(cfg, prompt=_control_prompt())
    first = backend.generate(repeated)
    second = backend.generate(repeated)
    third = backend.generate(control)
    ttft1 = (first.ttft_ns or 0) / 1e9
    ttft2 = (second.ttft_ns or 0) / 1e9
    ttft3 = (third.ttft_ns or 0) / 1e9

    ratio_repeat = (ttft2 / ttft1) if ttft1 > 0 else None
    ratio_control = (ttft2 / ttft3) if ttft3 > 0 else None
    # A real prefix-cache hit removes almost all of prefill, on BOTH comparisons.
    reuse_observed = bool(
        ratio_repeat is not None
        and ratio_control is not None
        and ratio_repeat < 0.5
        and ratio_control < 0.5
    )
    warmup_artifact = bool(
        ratio_repeat is not None
        and ratio_control is not None
        and ratio_repeat < 0.5
        and ratio_control >= 0.5
    )
    record = {
        "cache_instrumented": bool(first.extra.get("cache_instrumented", False)),
        "cache_instrumentation_note": first.extra.get("cache_instrumentation_note"),
        "cached_prompt_tokens_reported": [
            first.cache_read_input_tokens,
            second.cache_read_input_tokens,
            third.cache_read_input_tokens,
        ],
        "call_labels": ["first(P1)", "repeat(P1)", "control(P2, length-matched, distinct)"],
        "ttft_s": [ttft1, ttft2, ttft3],
        "prompt_tokens": [first.prompt_tokens, second.prompt_tokens, third.prompt_tokens],
        "ttft_ratio_second_over_first": ratio_repeat,
        "ttft_ratio_second_over_control": ratio_control,
        "reuse_observed": reuse_observed,
        "reuse_criterion": (
            "ttft(repeat)/ttft(first) < 0.5 AND ttft(repeat)/ttft(distinct control) < 0.5; the "
            "control is what separates prefix reuse from first-call warmup"
        ),
        "probe_position": "after warmup",
        "verdict": (
            "cache reuse observed"
            if reuse_observed
            else (
                "no reuse: the repeat is faster than the first call but no faster than a distinct "
                "length-matched prompt, which is warmup, not caching"
                if warmup_artifact
                else "no cross-call prefix reuse; full re-prefill"
            )
        ),
    }
    log_event("efilter.cache_probe", message=str(record["verdict"]), cache_probe=record)
    return record


def _control_prompt() -> str:
    """A distinct prompt of the same character length as :data:`_PROBE_PROMPT`.

    Length-matched so that a TTFT difference between it and the repeated prompt cannot be a prompt
    -length difference in disguise.
    """
    control = (
        "Count the entries under /data, then state the total. Prefer calling a tool over recalling."
    )
    if len(control) < len(_PROBE_PROMPT):
        control += " " * (len(_PROBE_PROMPT) - len(control))
    return control[: len(_PROBE_PROMPT)]


# ==================================================================================================
# The run
# ==================================================================================================


def _throwaway_warmup_generation(
    backend: LocalOpenVinoBackend, cfg: dict[str, Any]
) -> dict[str, Any]:
    """One untimed generation after pipeline reconstruct so the next task is not cold-load."""
    td = cfg.get("teardown") or {}
    max_tokens = int(td.get("warmup_max_tokens") or 8)
    request = GenerationRequest(
        messages=[{"role": "user", "content": _PROBE_PROMPT}],
        system=SYSTEM_PROMPT,
        tools=TOOL_SPECS,
        max_tokens=max_tokens,
        temperature=0.0,
        expect_tool_call=False,
    )
    t0 = time.perf_counter()
    result = backend.generate(request)
    return {
        "wall_s": time.perf_counter() - t0,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "max_tokens": max_tokens,
    }


def _teardown_between_tasks(
    *,
    backend: LocalOpenVinoBackend,
    cfg: dict[str, Any],
    launch_free_memory_mb: float,
    task_id: str,
) -> dict[str, Any]:
    """Destroy pipeline, GC, wait for free-memory recovery, reload + throwaway warmup (C2c).

    Runs **between** timed tasks with the machine lock released. Failure to recover free memory
    is a C9 leak finding - refuse rather than continue.
    """
    td = cfg.get("teardown") or {}
    fraction = float(td.get("free_memory_recovery_fraction", 0.90))
    timeout_s = float(td.get("recovery_timeout_s", 180.0))
    poll_s = float(td.get("recovery_poll_interval_s", 2.0))
    threshold_mb = float(launch_free_memory_mb) * fraction
    free_before_close = free_memory_mb()
    backend.close()
    gc.collect()
    recovery = wait_free_memory_recovery(
        threshold_mb=threshold_mb,
        timeout_s=timeout_s,
        poll_interval_s=poll_s,
    )
    record: dict[str, Any] = {
        "after_task_id": task_id,
        "launch_free_memory_mb": float(launch_free_memory_mb),
        "free_memory_recovery_fraction": fraction,
        "recovery_threshold_mb": threshold_mb,
        "free_memory_mb_before_close": free_before_close,
        "recovery": recovery,
        "warmup": None,
    }
    if not recovery["recovered"]:
        record["c9_leak_finding"] = {
            "kind": "inter_task_memory_recovery_failure",
            "message": (
                f"free memory {recovery['free_memory_mb']:.1f} MiB did not recover above "
                f"{threshold_mb:.1f} MiB "
                f"(launch_free_memory_mb={launch_free_memory_mb:.1f} x {fraction}) "
                f"within {timeout_s:.0f}s after pipeline teardown"
            ),
            "citing_protocol": "C2c §3 / C9 accumulating-leak finding",
        }
        return record
    backend.reload()
    record["warmup"] = _throwaway_warmup_generation(backend, cfg)
    record["free_memory_mb_after_teardown"] = free_memory_mb()
    return record


def _run_tasks(
    *,
    root: Path,
    cfg: dict[str, Any],
    workload: Workload,
    backend: LocalOpenVinoBackend,
    throughput: ThroughputModel,
    n_out_pred: dict[str, int],
    kv: KvGeometry,
    n_tasks: int,
    label: str,
    writer: StepLogWriter,
    platform_cfg: Any,
    p_cpus: list[int],
    launch_free_memory_mb: float,
    run_dir: RunDir | None = None,
    run_id: str | None = None,
) -> tuple[
    list[TaskResult],
    list[dict[str, Any]],
    list[dict[str, Any]],
    int,
    dict[str, Any],
]:
    """Run timed tasks in randomized order; optional discard-first warmup (C2f).

    When ``workload.discard_first_task_warmup`` is true, one extra full task runs first at
    ``execution_position=0`` and is discarded from ratio/analysis (not written to steps/tasks
    analysis streams). Timed tasks begin at position 1.

    Returns
    ``(results, per_task, machine_blocks, timed_invalidation_events, discard_meta)``.
    """
    discard_first = bool(cfg["workload"].get("discard_first_task_warmup", False))
    n_execute = int(n_tasks) + (1 if discard_first else 0)
    rng = random.Random(int(cfg["workload"]["seed"]))
    tasks = list(workload.tasks)
    rng.shuffle(tasks)
    if len(tasks) < n_execute:
        raise SeamError(
            f"workload has {len(tasks)} tasks but need {n_execute} "
            f"(n_tasks={n_tasks} + discard_first={discard_first})"
        )
    tasks = tasks[:n_execute]
    cooldown_s = float(cfg["thermal"]["cooldown_between_tasks_s"])
    freq_interval = float(cfg["thermal"]["frequency_sample_interval_s"])
    rss_interval = float(cfg["sampling"]["rss_interval_s"])
    scaffold = int(cfg["policy"].get("prompt_token_scaffold_tokens") or 0)
    context_cap = cfg["workload"].get("context_cap_tokens")
    context_cap_tokens = int(context_cap) if context_cap is not None else None
    constrain = bool(cfg.get("constrained_decoding", {}).get("enabled", False))
    cloud = RefusingCloudBackend(run_id=label)
    startup_quiesce = {
        "passed": True,
        "source": "efilter_run_startup_quiesce",
        "thresholds": cfg["quiescence"],
    }

    results: list[TaskResult] = []
    per_task: list[dict[str, Any]] = []
    machine_blocks: list[dict[str, Any]] = []
    invalidations = 0
    discarded_warmup_task: dict[str, Any] | None = None
    canary_by_position: list[dict[str, Any]] = []
    timed_index = 0
    for exec_index, task in enumerate(tasks):
        is_discard = bool(discard_first and exec_index == 0)
        execution_position = 0 if is_discard else (timed_index + 1 if discard_first else exec_index)
        if exec_index > 0:
            print(f"  cooldown {cooldown_s}s")
            time.sleep(cooldown_s)

        free_before = free_memory_mb()
        rss_before = rss_bytes_now()
        power_before = capture_power_state()
        sampler = FrequencySampler(interval_s=freq_interval, n_cpus=8)
        block_record: dict[str, Any]
        # Discarded warmup: full task under lock+canary, but do not stream steps into analysis.
        step_sink = None if is_discard else writer.write
        with machine_measurement(
            repo_root=root,
            label=f"{label}/{task.task_id}" + ("/discard_warmup" if is_discard else ""),
            config=cfg,
            p_cpus=p_cpus,
            prevalidated_quiescence=startup_quiesce,
        ) as machine_block:
            sampler.start()
            t0 = time.perf_counter()
            result = run_task(
                task=task,
                world=workload.world,
                local_backend=backend,
                cloud_backend=cloud,
                throughput=throughput,
                deadline_s=DEADLINE_DISABLED_S,
                n_out_pred_tokens=n_out_pred,
                max_steps=int(cfg["workload"]["max_steps"]),
                max_tokens=int(cfg["workload"]["max_tokens"]),
                run_id=run_id or label,
                program_id=f"{label}/{task.task_id}",
                kv_bytes_per_token=kv.bytes_per_token,
                rss_sample_interval_s=rss_interval,
                step_sink=step_sink,
                prompt_token_scaffold_tokens=scaffold,
                context_cap_tokens=context_cap_tokens,
                constrain_tool_calls=constrain,
            )
            wall_s = time.perf_counter() - t0
            sampler.stop()
            free_after = free_memory_mb()
            rss_after = rss_bytes_now()
            power_after = capture_power_state()
            block_record = machine_block.record

        paging_adm = paging_admissibility(block_record)
        canary_adm = canary_admissibility(block_record)
        block_record = {**dict(block_record), **paging_adm, **canary_adm}
        block_valid = bool(block_record.get("valid", True))

        for step in result.steps:
            assert_escalation_disabled(
                deadline_s=float(step.routing["deadline_s"]),
                t_pred_s=float(step.routing["t_pred_s"]),
            )
            if step.assigned_target != "local":
                raise SeamError(
                    f"{task.task_id} step {step.step_idx} executed on "
                    f"{step.assigned_target!r} in an escalation-disabled arm"
                )

        ctx_by_step = [s.context_tokens_total for s in result.steps]
        ratio = per_task_context_ratio(ctx_by_step)
        mem_inst = task_memory_instrumentation(
            result,
            free_before=free_before,
            free_after=free_after,
            rss_before=rss_before,
            rss_after=rss_after,
        )
        task_record: dict[str, Any] = {
            "execution_position": execution_position,
            "discarded_warmup": is_discard,
            "order_index": None if is_discard else timed_index,
            "task_id": task.task_id,
            "success": result.success,
            "realized_steps": result.realized_steps,
            "terminated_reason": result.terminated_reason,
            "projected_next_context_tokens": result.projected_next_context_tokens,
            "wall_s": wall_s,
            "jct_s": result.jct_s,
            "context_tokens_by_step": ctx_by_step,
            "context_ratio_cmax_over_cmin": ratio,
            "context_ratio_definition": (
                "C_max/C_min = max(context_tokens_by_step)/min(context_tokens_by_step)"
            ),
            "prompt_tokens_new_by_step": [s.prompt_tokens_new for s in result.steps],
            "completion_tokens_by_step": [s.completion_tokens for s in result.steps],
            "kv_bytes_resident_by_step": [s.kv_bytes_resident for s in result.steps],
            "tool_well_formed_by_step": [s.tool is not None for s in result.steps],
            **mem_inst,
            "free_memory_mb_after_teardown": None,
            "free_memory_mb_at_task_after_teardown": None,
            "machine_measurement_valid": block_valid,
            "machine_measurement_invalid_reasons": list(block_record.get("invalid_reasons") or []),
            "paging_sampling_enabled": block_record.get("paging_sampling_enabled"),
            "paging_window_start_utc": block_record.get("paging_window_start_utc"),
            "paging_window_end_utc": block_record.get("paging_window_end_utc"),
            "paging_gate": block_record.get("paging_gate"),
            "paging_admissible": paging_adm["paging_admissible"],
            "paging_admissible_reasons": list(paging_adm["paging_admissible_reasons"]),
            "paging_gate_verdict": paging_adm["paging_gate_verdict"],
            "canary_gate": block_record.get("canary_gate"),
            "canary_admissible": canary_adm["canary_admissible"],
            "canary_admissible_reasons": list(canary_adm["canary_admissible_reasons"]),
            "canary_relative_drift": block_record.get("canary_relative_drift"),
            "canary_relative_drift_immediate": block_record.get("canary_relative_drift_immediate"),
            "canary_relative_drift_settled": block_record.get("canary_relative_drift_settled"),
            "canary_drift_authoritative": block_record.get("canary_drift_authoritative"),
            "timing_admissible": bool(
                paging_adm["paging_admissible"] and canary_adm["canary_admissible"]
            ),
            "frequency": sampler.summary(),
            "power_before": _power_brief(power_before),
            "power_after": _power_brief(power_after),
        }
        canary_by_position.append(
            {
                "execution_position": execution_position,
                "discarded_warmup": is_discard,
                "task_id": task.task_id,
                "canary_relative_drift": task_record["canary_relative_drift"],
                "canary_relative_drift_immediate": task_record["canary_relative_drift_immediate"],
                "canary_relative_drift_settled": task_record.get("canary_relative_drift_settled"),
                "canary_admissible": task_record["canary_admissible"],
                "canary_admissible_reasons": list(task_record["canary_admissible_reasons"]),
            }
        )

        teardown_record: dict[str, Any] | None = None
        if exec_index < len(tasks) - 1:
            print(
                f"  teardown after {task.task_id}"
                f"{' (discard warmup)' if is_discard else ''} (lock released)"
            )
            teardown_record = _teardown_between_tasks(
                backend=backend,
                cfg=cfg,
                launch_free_memory_mb=launch_free_memory_mb,
                task_id=task.task_id,
            )
            task_record["teardown"] = teardown_record
            free_td = teardown_record.get(
                "free_memory_mb_after_teardown",
                teardown_record.get("recovery", {}).get("free_memory_mb"),
            )
            task_record["free_memory_mb_after_teardown"] = free_td
            task_record["free_memory_mb_at_task_after_teardown"] = free_td
            if teardown_record.get("c9_leak_finding"):
                finding_path = root / "derived" / "efilter" / "c9_inter_task_memory_leak.json"
                finding_path.parent.mkdir(parents=True, exist_ok=True)
                finding_path.write_text(
                    json.dumps(
                        {
                            "run_id": run_id or label,
                            "task_id": task.task_id,
                            "written_utc": utc_now_iso(),
                            **teardown_record["c9_leak_finding"],
                            "teardown": teardown_record,
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                raise SeamError(
                    teardown_record["c9_leak_finding"]["message"] + f" (wrote {finding_path})"
                )

        if is_discard:
            discarded_warmup_task = task_record
            print(
                f"  DISCARD warmup pos=0 {task.task_id}: steps={result.realized_steps} "
                f"success={result.success} wall={wall_s:.1f}s "
                f"canary_drift={task_record['canary_relative_drift']} "
                f"(not in ratio/analysis)"
            )
        else:
            machine_blocks.append(block_record)
            if not block_valid:
                invalidations += len(block_record.get("invalid_reasons") or []) or 1
            per_task.append(task_record)
            results.append(result)
            if run_dir is not None:
                run_dir.append_ndjson(
                    _TASKS_FILENAME,
                    {
                        "run_id": run_id or label,
                        "block_id": f"{task.task_id}:{timed_index}",
                        "validity": {
                            "admissible": block_valid,
                            "block_id": f"{task.task_id}:{timed_index}",
                            "invalid_reasons": list(block_record.get("invalid_reasons") or []),
                        },
                        **task_record,
                    },
                )
            proj = result.projected_next_context_tokens
            proj_note = f" projected_next={proj}" if proj is not None else ""
            print(
                f"  pos={execution_position} {task.task_id}: steps={result.realized_steps} "
                f"success={result.success} wall={wall_s:.1f}s context {ctx_by_step} "
                f"term={result.terminated_reason}{proj_note} block_valid={block_valid} "
                f"canary_adm={task_record['canary_admissible']} "
                f"paging_adm={task_record['paging_admissible']}"
            )
            timed_index += 1

        if power_after.on_battery is not False:
            raise SeamError(
                f"AC was lost during {task.task_id}: every task after this point would be measured "
                f"in a different power regime. Stopping rather than pooling two regimes."
            )
    _ = platform_cfg
    discard_meta = {
        "discard_first_task_warmup": discard_first,
        "discarded_warmup_task": discarded_warmup_task,
        "canary_drift_by_execution_position": canary_by_position,
        "n_timed_tasks": len(per_task),
        "n_executed_including_discard": len(tasks),
    }
    return results, per_task, machine_blocks, invalidations, discard_meta


def efilter_output_paths() -> dict[str, str]:
    """Manifest ``outputs`` catalog for E-FILTER - shared by dry-run and real ``_emit`` (C2e).

    ``tasks`` indexes the per-task JSONL the harness actually writes. The schema must allow it;
    omitting the pointer would hide a sealed artifact from the catalog (see C9 / e66701aa).
    """
    return {"steps": STEPS_FILENAME, "tasks": _TASKS_FILENAME}


def nested_key_set(obj: Any, *, prefix: str = "") -> set[str]:
    """Collect key paths at every nesting level (dict keys only; list elements are walked)."""
    keys: set[str] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            keys.add(path)
            keys |= nested_key_set(value, prefix=path)
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            keys |= nested_key_set(value, prefix=f"{prefix}[{index}]")
    return keys


def classify_invalidation_reason(reason: str) -> str:
    """Classify a machine-measurement reason for per-endpoint gating (C2e/C2f)."""
    text = str(reason).lower()
    if "page read" in text:
        return "paging"
    if "canary" in text:
        return "canary"
    if "available memory" in text or "memory" in text:
        return "memory_floor"
    return "other"


def split_invalidation_reasons(reasons: list[str] | None) -> dict[str, list[str]]:
    """Partition reasons into paging / canary / memory_floor / other (and legacy non_paging)."""
    paging: list[str] = []
    canary: list[str] = []
    memory_floor: list[str] = []
    other: list[str] = []
    for reason in reasons or []:
        text = str(reason)
        kind = classify_invalidation_reason(text)
        if kind == "paging":
            paging.append(text)
        elif kind == "canary":
            canary.append(text)
        elif kind == "memory_floor":
            memory_floor.append(text)
        else:
            other.append(text)
    non_paging = canary + memory_floor + other
    return {
        "paging": paging,
        "canary": canary,
        "memory_floor": memory_floor,
        "other": other,
        "non_paging": non_paging,
    }


def paging_admissibility(block_record: dict[str, Any]) -> dict[str, Any]:
    """Tag a timed block for per-endpoint analysis (C2e).

    Keeps the measurement gate verdict/reasons. Paging inadmissibility contaminates timing
    endpoints only; envelope endpoints (KV, context, AI) are arithmetic from token counts.
    """
    gate = dict(block_record.get("paging_gate") or {})
    gate_reasons = [str(r) for r in (gate.get("reasons") or [])]
    split = split_invalidation_reasons(list(block_record.get("invalid_reasons") or []))
    # Prefer the dedicated gate reasons; fall back to invalid_reasons classified as paging.
    reasons = gate_reasons or list(split["paging"])
    verdict = gate.get("verdict")
    if verdict == "not_sampled":
        admissible = True
        reasons = []
    elif verdict == "fail":
        admissible = False
    else:
        admissible = not reasons
    return {
        "paging_admissible": bool(admissible),
        "paging_admissible_reasons": list(reasons),
        "paging_gate_verdict": verdict,
        "paging_gate_reasons": list(gate_reasons),
    }


def canary_admissibility(block_record: dict[str, Any]) -> dict[str, Any]:
    """Tag a timed block for per-endpoint canary admissibility (C2f).

    Wall-clock timeout audit verdict NONE: canary drift cannot change token production, so it
    contaminates timing endpoints only (same causal set as paging). Envelope stays over all
    blocks. Affinity failures are treated as canary-inadmissible.
    """
    gate = dict(block_record.get("canary_gate") or {})
    gate_reasons = [str(r) for r in (gate.get("reasons") or [])]
    split = split_invalidation_reasons(list(block_record.get("invalid_reasons") or []))
    reasons = gate_reasons or list(split["canary"])
    verdict = gate.get("verdict")
    if verdict == "fail":
        admissible = False
    elif verdict == "pass":
        admissible = not reasons
    else:
        # Pre-C2f records or pending: infer from classified canary reasons / drift fields.
        admissible = not reasons
    drift = block_record.get("canary_relative_drift")
    drift_immediate = block_record.get("canary_relative_drift_immediate")
    return {
        "canary_admissible": bool(admissible),
        "canary_admissible_reasons": list(reasons),
        "canary_gate_verdict": verdict,
        "canary_relative_drift": drift,
        "canary_relative_drift_immediate": drift_immediate,
        "canary_drift_authoritative": block_record.get("canary_drift_authoritative"),
    }


def timed_block_invalidation_split(
    per_task: list[dict[str, Any]],
) -> dict[str, int]:
    """Count timed-task blocks by invalidation class (C2e/C2f)."""
    paging_blocks = 0
    canary_blocks = 0
    memory_floor_blocks = 0
    other_blocks = 0
    for row in per_task:
        if row.get("paging_admissible") is False:
            paging_blocks += 1
        if row.get("canary_admissible") is False:
            canary_blocks += 1
        split = split_invalidation_reasons(
            list(row.get("machine_measurement_invalid_reasons") or [])
        )
        if split["memory_floor"]:
            memory_floor_blocks += 1
        if split["other"]:
            other_blocks += 1
        # Fallback when canary_admissible tag absent but reasons present (pre-tag rows).
        if row.get("canary_admissible") is None and split["canary"]:
            canary_blocks += 1
    clearance_fatal = memory_floor_blocks + other_blocks
    return {
        "paging_invalidations_inside_timed_blocks": paging_blocks,
        "canary_invalidations_inside_timed_blocks": canary_blocks,
        "memory_floor_invalidations_inside_timed_blocks": memory_floor_blocks,
        "other_invalidations_inside_timed_blocks": other_blocks,
        "non_paging_invalidations_inside_timed_blocks": (
            canary_blocks + memory_floor_blocks + other_blocks
        ),
        "clearance_fatal_invalidations_inside_timed_blocks": clearance_fatal,
        "inside_timed_blocks": paging_blocks + canary_blocks + memory_floor_blocks + other_blocks,
    }


def nested_key_set_structure(obj: Any, *, prefix: str = "") -> set[str]:
    """Key-set comparison that ignores list indices (compares element-0 shape only).

    Used by the dry-run parity test so two builder outputs with different list lengths still
    share one structural key set.
    """
    keys: set[str] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            keys.add(path)
            keys |= nested_key_set_structure(value, prefix=path)
    elif isinstance(obj, list) and obj:
        keys |= nested_key_set_structure(obj[0], prefix=f"{prefix}[]")
    return keys


def build_efilter_summary(
    *,
    cfg: dict[str, Any],
    mode: str,
    cfg_sha: str,
    workload_sha: str,
    model_spec_sha: str,
    dry_run_meta: dict[str, Any],
    launch_free_memory_mb: float,
    mem_series: dict[str, Any],
    quiesce: dict[str, Any],
    openvino_info: dict[str, Any],
    backend_config: dict[str, Any],
    kv_geometry: dict[str, Any],
    cache_probe: dict[str, Any],
    cache_instrumented: bool,
    throughput_block: dict[str, Any],
    warmup: dict[str, Any],
    n_out_pred: dict[str, int],
    n_out_source: str,
    n_out_freeze: dict[str, Any],
    warmup_machine: dict[str, Any],
    task_machine_blocks: list[dict[str, Any]],
    inv_split: dict[str, int],
    paging_split: dict[str, int],
    growth: dict[str, Any],
    gate: dict[str, Any],
    tool_rate: dict[str, Any],
    t_pred_values: list[float],
    per_task: list[dict[str, Any]],
    n_tasks: int,
    n_steps: int,
    run_wall_s: float,
    success_rate: float | None,
    max_t_pred: float,
    escalated_steps: int,
    discard_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble ``summary.json`` for E-FILTER. Dry-run and real emit MUST call this (C2e/C2f).

    Shape parity is structural: every field the real run seals is produced here from stub or
    measured inputs. Do not hand-construct a parallel fixture for the dry-run.
    """
    from seam.analysis.c2g_memory import fit_memory_vs_context

    cd_cfg = cfg.get("constrained_decoding") or {}
    memory_vs_context = fit_memory_vs_context(per_task)
    return {
        "experiment_id": cfg["experiment_id"],
        "mode": mode,
        "pre_registration": cfg["pre_registration"],
        "c2_amendment": cfg.get("c2_amendment"),
        "c2b_amendment": cfg.get("c2b_amendment"),
        "c2c_amendment": cfg.get("c2c_amendment"),
        "c2d_amendment": cfg.get("c2d_amendment"),
        "c2e_amendment": cfg.get("c2e_amendment"),
        "c2f_amendment": cfg.get("c2f_amendment"),
        "c2g_amendment": cfg.get("c2g_amendment"),
        "arm": cfg["arm"],
        "launch_free_memory_mb": launch_free_memory_mb,
        "teardown": cfg.get("teardown"),
        "memory_series": mem_series,
        "memory_vs_context": memory_vs_context,
        "chunked_prefill": cfg.get("chunked_prefill"),
        "c2g_chunked_prefill_cpu_audit": {
            "path": "derived/efilter/c2g_chunked_prefill_cpu_audit.json",
        },
        "c2g_memory_vs_context_note": {
            "path": "derived/efilter/c2g_memory_vs_context_note.json",
        },
        "config_sha256": {
            "efilter_yaml": cfg_sha,
            "task_list": workload_sha,
            "model_spec": model_spec_sha,
        },
        "startup_output_path_dry_run": dry_run_meta,
        "lifecycle": {
            "early_run_dir": True,
            "steps_append_fsync": True,
            "per_task_jsonl": _TASKS_FILENAME,
            "paging_exclude_on_failure": True,
            "summary_builder": "build_efilter_summary",
        },
        "escalation": {
            "deadline_s": DEADLINE_DISABLED_S,
            "sentinel": "DEADLINE_DISABLED_S",
            "sentinel_is_finite": True,
            "max_observed_t_pred_s": max_t_pred,
            "headroom_factor": DEADLINE_DISABLED_S / max(max_t_pred, 1e-12),
            "escalated_steps": escalated_steps,
            "cloud_call_attempts": 0,
            "verified_by": "policy.assert_escalation_disabled per step, on the logged deadline",
        },
        "openvino": openvino_info,
        "backend_config": backend_config,
        "confinement": {
            "mechanism": None,
            "reason": (
                "mslice-a1a6 found zero mechanisms CONFINED in both prefill and decode "
                "(run 5eb09eba-b321-4b7e-b7df-e9b01194d388); none adopted, so none applied"
            ),
            "default_placement_citing_run_id": cfg["openvino"]["default_placement_citing_run_id"],
            "target_label_basis": (
                "OpenVINO default placement measured to load p_cpus=[0,1,2,3] regardless of thread "
                "count; the manifest target names where work ran, not a confinement claim"
            ),
        },
        "kv_geometry": kv_geometry,
        "cache_probe": cache_probe,
        "cache_instrumented": cache_instrumented,
        "throughput": throughput_block,
        "policy": {
            **{k: v for k, v in cfg["policy"].items() if k != "n_out_pred_tokens_pilot"},
            "n_out_pred_tokens_used": n_out_pred,
            "n_out_pred_source": n_out_source,
            "n_out_pred_constant_across_step_types": True,
            "n_out_observed_from_pilot": n_out_freeze,
            "prompt_token_scaffold_tokens": int(
                cfg["policy"].get("prompt_token_scaffold_tokens") or 0
            ),
            "scaffold_tokens_source": cfg["policy"].get("scaffold_tokens_source"),
            "prompt_token_source": cfg["policy"].get("prompt_token_source"),
        },
        "workload_memory_controls": {
            "max_tokens": int(cfg["workload"]["max_tokens"]),
            "context_cap_tokens": cfg["workload"].get("context_cap_tokens"),
            "constrained_decoding": {
                "enabled": bool(cd_cfg.get("enabled", False)),
                "mechanism": cd_cfg.get("mechanism", CONSTRAINED_DECODING_MECHANISM),
                "mechanism_detail": cd_cfg.get("mechanism_detail"),
            },
        },
        "thermal": {**cfg["thermal"], "regime": "confound"},
        "quiesce": quiesce,
        "machine_measurement_warmup": warmup_machine,
        "machine_measurement_tasks": task_machine_blocks,
        "memory_or_canary_invalidations": int(
            paging_split.get("non_paging_invalidations_inside_timed_blocks", 0)
        ),
        "memory_or_canary_invalidations_inside_timed_blocks": int(
            paging_split.get("non_paging_invalidations_inside_timed_blocks", 0)
        ),
        "paging_invalidations_inside_timed_blocks": int(
            paging_split.get("paging_invalidations_inside_timed_blocks", 0)
        ),
        "canary_invalidations_inside_timed_blocks": int(
            paging_split.get("canary_invalidations_inside_timed_blocks", 0)
        ),
        "memory_floor_invalidations_inside_timed_blocks": int(
            paging_split.get("memory_floor_invalidations_inside_timed_blocks", 0)
        ),
        "clearance_fatal_invalidations_inside_timed_blocks": int(
            paging_split.get("clearance_fatal_invalidations_inside_timed_blocks", 0)
        ),
        "warmup_invalidations_not_gated": int(inv_split.get("warmup_not_gated", 0)),
        "paging_window_scope": "timed_task_blocks_only",
        "paging_gate_scope": "per_endpoint_admissibility_c2e",
        "canary_gate_scope": "per_endpoint_admissibility_c2f",
        "timing_admissibility_rule": "canary_admissible AND paging_admissible",
        "wallclock_timeout_audit": {
            "path": cfg.get(
                "wallclock_timeout_audit",
                "derived/efilter/c2f_wallclock_timeout_audit.json",
            ),
            "verdict": "NONE",
            "section_3_stands": True,
        },
        "canary": {
            "max_relative_drift": cfg.get("canary", {}).get("max_relative_drift"),
            "settle_s": cfg.get("canary", {}).get("settle_s"),
            "double_after": cfg.get("canary", {}).get("double_after"),
            "gate_mode": cfg.get("canary", {}).get("gate_mode"),
            "idle_p95": cfg.get("canary", {}).get("idle_p95"),
            "margin_above_idle_p95": cfg.get("canary", {}).get("margin_above_idle_p95"),
            "baseline_run_id": cfg.get("canary", {}).get("baseline_run_id"),
            "threshold_justification": cfg.get("canary", {}).get("threshold_justification"),
        },
        "canary_gate": {
            "baseline_run_id": cfg.get("canary", {}).get("baseline_run_id"),
            "idle_p95": cfg.get("canary", {}).get("idle_p95"),
            "margin": cfg.get("canary", {}).get("margin_above_idle_p95"),
            "threshold": cfg.get("canary", {}).get("max_relative_drift"),
            "justification": cfg.get("canary", {}).get("threshold_justification"),
            "gate_mode": cfg.get("canary", {}).get("gate_mode"),
        },
        "discard_first_task_warmup": (discard_meta or {}).get(
            "discard_first_task_warmup",
            bool(cfg.get("workload", {}).get("discard_first_task_warmup", False)),
        ),
        "discarded_warmup_task": (discard_meta or {}).get("discarded_warmup_task"),
        "canary_drift_by_execution_position": (discard_meta or {}).get(
            "canary_drift_by_execution_position", []
        ),
        "c9_output_path_defects_note": {
            "path": "derived/efilter/c9_output_path_defects_note.json",
            "citing_run_ids": [
                "cb0ed2e3-ed70-4627-b231-51016d2b255b",
                "a161f89e-fcc2-4cb8-9b04-c215358a2f88",
                "e66701aa-37b9-4b00-b86b-7ffb477f4e66",
            ],
        },
        "n_tasks": n_tasks,
        "n_steps": n_steps,
        "run_wall_s": run_wall_s,
        "success_rate": success_rate,
        "success_rate_not_gated": True,
        "context_growth": growth,
        "context_ceiling": {
            "definition": (
                "per-task C_max/C_min = max/min of context_tokens_by_step; aggregate = median"
            ),
            "per_task_ratios": growth["per_task_ratios"],
            "median_ratio": growth["median_context_ratio"],
            "mean_ratio": (
                statistics.fmean([r for r in growth["per_task_ratios"] if r is not None])
                if any(r is not None for r in growth["per_task_ratios"])
                else None
            ),
            "max_context_tokens_observed": growth["max_context_tokens_observed"],
            "max_context_tokens_is_gate": False,
        },
        "pilot_context_gate": gate,
        "tool_call_rate": tool_rate,
        "t_pred_distribution": {
            "n": len(t_pred_values),
            "min_s": min(t_pred_values) if t_pred_values else None,
            "median_s": (float(statistics.median(t_pred_values)) if t_pred_values else None),
            "max_s": max(t_pred_values) if t_pred_values else None,
            "seconds_caveat": "R unverified pending A4 - report deadlines in distribution units",
        },
        "per_task": per_task,
        "c9_practical_ceiling_note": {
            "path": "derived/efilter/c9_practical_ceiling_note.json",
            "citing_run_id": "0fe5e4c7-bb38-4666-826b-2c512b17a969",
        },
        "step_type_limitation": (
            "step_type is hardcoded to tool_call_synthesis except on the terminal step; C2b keeps "
            "n_out_pred constant across types so ranking by t_pred equals ranking by context. "
            "Stratify by step_idx."
        ),
        "optional_memory_headroom_arm": cfg.get("optional_memory_headroom_arm"),
        "throughput_warmup_detail": warmup,
    }


def stub_efilter_summary_kwargs(cfg: dict[str, Any], *, mode: str = "pilot") -> dict[str, Any]:
    """Stub inputs for :func:`build_efilter_summary` - production key shape, synthetic values."""
    paging_gate_pass = {
        "gate_mode": "baseline_relative",
        "verdict": "pass",
        "reasons": [],
        "exclude_on_failure": True,
    }
    paging_gate_fail = {
        **paging_gate_pass,
        "verdict": "fail",
        "reasons": ["sustained hard page reads above baseline-relative threshold"],
    }
    canary_gate_pass = {
        "gate_mode": "asserted_pending_baseline",
        "verdict": "pass",
        "reasons": [],
        "max_relative_drift": 0.15,
        "authoritative_drift": 0.01,
    }
    machine_ok = {
        "valid": True,
        "invalid_reasons": [],
        "paging_sampling_enabled": True,
        "paging_window_start_utc": "2026-08-04T00:00:00+00:00",
        "paging_window_end_utc": "2026-08-04T00:01:00+00:00",
        "paging_gate": paging_gate_pass,
        "canary_gate": canary_gate_pass,
        "canary_relative_drift": 0.01,
        "canary_relative_drift_immediate": 0.02,
        "canary_drift_authoritative": "settled",
        **paging_admissibility(
            {
                "paging_gate": paging_gate_pass,
                "invalid_reasons": [],
            }
        ),
        **canary_admissibility(
            {
                "canary_gate": canary_gate_pass,
                "canary_relative_drift": 0.01,
                "canary_relative_drift_immediate": 0.02,
                "canary_drift_authoritative": "settled",
                "invalid_reasons": [],
            }
        ),
    }
    machine_paging_fail = {
        **machine_ok,
        "valid": False,
        "invalid_reasons": list(paging_gate_fail["reasons"]),
        "paging_gate": paging_gate_fail,
        **paging_admissibility(
            {
                "paging_gate": paging_gate_fail,
                "invalid_reasons": list(paging_gate_fail["reasons"]),
            }
        ),
    }
    per_task = [
        {
            "execution_position": 1,
            "discarded_warmup": False,
            "order_index": 0,
            "task_id": "C2T_STUB_0",
            "success": True,
            "realized_steps": 2,
            "terminated_reason": "answer",
            "projected_next_context_tokens": None,
            "wall_s": 1.0,
            "jct_s": 1.0,
            "context_tokens_by_step": [100, 400],
            "context_ratio_cmax_over_cmin": 4.0,
            "context_ratio_definition": (
                "C_max/C_min = max(context_tokens_by_step)/min(context_tokens_by_step)"
            ),
            "prompt_tokens_new_by_step": [100, 50],
            "completion_tokens_by_step": [32, 32],
            "kv_bytes_resident_by_step": [100 * 73728, 400 * 73728],
            "tool_well_formed_by_step": [True, False],
            "peak_rss_bytes": 1_000_000,
            "step_memory": [
                {
                    "step_idx": 0,
                    "context_tokens_total": 100,
                    "rss_before_generate": 400_000,
                    "rss_peak_during_generate": 900_000,
                    "rss_after_generate": 500_000,
                    "free_memory_mb_before_generate": 8100.0,
                    "free_memory_mb_min_during_generate": 7800.0,
                    "free_memory_mb_after_generate": 8000.0,
                },
                {
                    "step_idx": 1,
                    "context_tokens_total": 400,
                    "rss_before_generate": 500_000,
                    "rss_peak_during_generate": 1_000_000,
                    "rss_after_generate": 600_000,
                    "free_memory_mb_before_generate": 7900.0,
                    "free_memory_mb_min_during_generate": 7600.0,
                    "free_memory_mb_after_generate": 7800.0,
                },
            ],
            "free_memory_mb_min": 7600.0,
            "free_memory_mb_min_step_idx": 1,
            "free_memory_mb_min_context_tokens": 400,
            "process_rss_peak": 1_000_000,
            "process_rss_peak_step_idx": 1,
            "free_memory_mb_at_task_start": 8000.0,
            "free_memory_mb_at_task_end": 7500.0,
            "free_memory_mb_before_task": 8000.0,
            "free_memory_mb_after_task": 7500.0,
            "free_memory_mb_after_teardown": None,
            "free_memory_mb_at_task_after_teardown": None,
            "process_rss_before": 500_000,
            "process_rss_after": 600_000,
            "machine_measurement_valid": True,
            "machine_measurement_invalid_reasons": [],
            "paging_sampling_enabled": True,
            "paging_window_start_utc": machine_ok["paging_window_start_utc"],
            "paging_window_end_utc": machine_ok["paging_window_end_utc"],
            "paging_gate": paging_gate_pass,
            "paging_admissible": True,
            "paging_admissible_reasons": [],
            "paging_gate_verdict": "pass",
            "canary_gate": canary_gate_pass,
            "canary_admissible": True,
            "canary_admissible_reasons": [],
            "canary_relative_drift": 0.01,
            "canary_relative_drift_immediate": 0.02,
            "canary_drift_authoritative": "settled",
            "timing_admissible": True,
            "frequency": {"n_samples": 1, "mean_mhz": None},
            "power_before": {
                "on_battery": False,
                "battery_pct": 100.0,
                "charging": False,
                "battery_saver": False,
            },
            "power_after": {
                "on_battery": False,
                "battery_pct": 100.0,
                "charging": False,
                "battery_saver": False,
            },
        },
        {
            "execution_position": 2,
            "discarded_warmup": False,
            "order_index": 1,
            "task_id": "C2T_STUB_1",
            "success": True,
            "realized_steps": 2,
            "terminated_reason": "answer",
            "projected_next_context_tokens": None,
            "wall_s": 2.0,
            "jct_s": 2.0,
            "context_tokens_by_step": [120, 480],
            "context_ratio_cmax_over_cmin": 4.0,
            "context_ratio_definition": (
                "C_max/C_min = max(context_tokens_by_step)/min(context_tokens_by_step)"
            ),
            "prompt_tokens_new_by_step": [120, 60],
            "completion_tokens_by_step": [32, 32],
            "kv_bytes_resident_by_step": [120 * 73728, 480 * 73728],
            "tool_well_formed_by_step": [True, False],
            "peak_rss_bytes": 1_100_000,
            "step_memory": [
                {
                    "step_idx": 0,
                    "context_tokens_total": 120,
                    "rss_before_generate": 500_000,
                    "rss_peak_during_generate": 1_000_000,
                    "rss_after_generate": 600_000,
                    "free_memory_mb_before_generate": 7600.0,
                    "free_memory_mb_min_during_generate": 7300.0,
                    "free_memory_mb_after_generate": 7500.0,
                },
                {
                    "step_idx": 1,
                    "context_tokens_total": 480,
                    "rss_before_generate": 600_000,
                    "rss_peak_during_generate": 1_100_000,
                    "rss_after_generate": 700_000,
                    "free_memory_mb_before_generate": 7400.0,
                    "free_memory_mb_min_during_generate": 7000.0,
                    "free_memory_mb_after_generate": 7200.0,
                },
            ],
            "free_memory_mb_min": 7000.0,
            "free_memory_mb_min_step_idx": 1,
            "free_memory_mb_min_context_tokens": 480,
            "process_rss_peak": 1_100_000,
            "process_rss_peak_step_idx": 1,
            "free_memory_mb_at_task_start": 7500.0,
            "free_memory_mb_at_task_end": 7000.0,
            "free_memory_mb_before_task": 7500.0,
            "free_memory_mb_after_task": 7000.0,
            "free_memory_mb_after_teardown": None,
            "free_memory_mb_at_task_after_teardown": None,
            "process_rss_before": 600_000,
            "process_rss_after": 700_000,
            "machine_measurement_valid": False,
            "machine_measurement_invalid_reasons": list(paging_gate_fail["reasons"]),
            "paging_sampling_enabled": True,
            "paging_window_start_utc": machine_paging_fail["paging_window_start_utc"],
            "paging_window_end_utc": machine_paging_fail["paging_window_end_utc"],
            "paging_gate": paging_gate_fail,
            "paging_admissible": False,
            "paging_admissible_reasons": list(paging_gate_fail["reasons"]),
            "paging_gate_verdict": "fail",
            "canary_gate": canary_gate_pass,
            "canary_admissible": True,
            "canary_admissible_reasons": [],
            "canary_relative_drift": 0.01,
            "canary_relative_drift_immediate": 0.02,
            "canary_drift_authoritative": "settled",
            "timing_admissible": False,
            "frequency": {"n_samples": 1, "mean_mhz": None},
            "power_before": {
                "on_battery": False,
                "battery_pct": 100.0,
                "charging": False,
                "battery_saver": False,
            },
            "power_after": {
                "on_battery": False,
                "battery_pct": 100.0,
                "charging": False,
                "battery_saver": False,
            },
        },
    ]
    growth = {
        "curve": [
            {"step_idx": 0, "mean_context_tokens": 110.0, "n": 2},
            {"step_idx": 1, "mean_context_tokens": 440.0, "n": 2},
        ],
        "growth_ratio_last_over_first": 4.0,
        "max_context_tokens_observed": 480,
        "per_task_ratios": [4.0, 4.0],
        "median_context_ratio": 4.0,
    }
    paging_split = timed_block_invalidation_split(per_task)
    gate = evaluate_pilot_context_gate(
        growth,
        per_task=per_task,
        min_median_context_ratio=float(
            cfg["workload"].get(
                "pilot_min_median_context_ratio", PILOT_MIN_MEDIAN_CONTEXT_RATIO_DEFAULT
            )
        ),
        memory_or_canary_invalidations=paging_split["non_paging_invalidations_inside_timed_blocks"],
        paging_invalidations=paging_split["paging_invalidations_inside_timed_blocks"],
        canary_invalidations=paging_split["canary_invalidations_inside_timed_blocks"],
        memory_floor_invalidations=paging_split[
            "clearance_fatal_invalidations_inside_timed_blocks"
        ],
    )
    n_out = {"tool_call_synthesis": 64, "answer_synthesis": 64}
    return {
        "cfg": cfg,
        "mode": mode,
        "cfg_sha": "0" * 64,
        "workload_sha": "1" * 64,
        "model_spec_sha": "2" * 64,
        "dry_run_meta": {
            "passed": True,
            "mode": "startup_dry_run",
            "synthetic_via_builder": True,
            "encoding_probe": NON_ASCII_PROBE,
        },
        "launch_free_memory_mb": 9000.0,
        "mem_series": memory_series_summary(per_task),
        "quiesce": {
            "passed": True,
            "deviations": [],
            "battery_pct": 100.0,
            "display_brightness": None,
            "defender_realtime": None,
            "forbidden_process_hits": [],
            "ambient_c": None,
        },
        "openvino_info": {"openvino": "stub", "genai": "stub"},
        "backend_config": {"device": "CPU", "stub": True},
        "kv_geometry": {
            "kv_bytes_per_token": 73728,
            "n_layers": 1,
            "n_kv_heads": 1,
            "head_dim": 1,
            "dtype": "f16",
        },
        "cache_probe": {"verdict": "stub", "ttft_ratio_second_over_first": None},
        "cache_instrumented": False,
        "throughput_block": {
            "target": "cpu-p",
            "r_prefill_tok_s": 100.0,
            "r_decode_tok_s": 50.0,
            "measured_by_run_id": "stub",
            "warmup": {"stub": True},
        },
        "warmup": {"stub": True},
        "n_out_pred": n_out,
        "n_out_source": "stub",
        "n_out_freeze": {
            "median_completion_tokens": 64,
            "n_steps": 4,
            "n_out_pred_tokens": n_out,
            "constant_across_step_types": True,
            "rationale": "stub",
            "per_type_medians_diagnostic_only": {},
        },
        "warmup_machine": {
            "valid": True,
            "invalid_reasons": [],
            "paging_sampling_enabled": False,
            "paging_gate": {"verdict": "not_sampled", "reasons": []},
            **paging_admissibility(
                {"paging_gate": {"verdict": "not_sampled", "reasons": []}, "invalid_reasons": []}
            ),
            **canary_admissibility(
                {
                    "canary_gate": canary_gate_pass,
                    "canary_relative_drift": 0.01,
                    "invalid_reasons": [],
                }
            ),
        },
        "task_machine_blocks": [machine_ok, machine_paging_fail],
        "inv_split": {"inside_timed_blocks": 1, "warmup_not_gated": 0},
        "paging_split": paging_split,
        "growth": growth,
        "gate": gate,
        "tool_rate": {
            "n_steps": 4,
            "n_well_formed_tool_calls": 2,
            "n_tool_none": 2,
            "well_formed_rate": 0.5,
            "tool_none_rate": 0.5,
        },
        "t_pred_values": [0.5, 1.0, 0.6, 1.2],
        "per_task": per_task,
        "n_tasks": 2,
        "n_steps": 4,
        "run_wall_s": 3.0,
        "success_rate": 1.0,
        "max_t_pred": 1.2,
        "escalated_steps": 0,
        "discard_meta": {
            "discard_first_task_warmup": True,
            "discarded_warmup_task": {
                "execution_position": 0,
                "discarded_warmup": True,
                "task_id": "C2T_STUB_DISCARD",
                "canary_relative_drift": 0.05,
                "canary_admissible": True,
            },
            "canary_drift_by_execution_position": [
                {
                    "execution_position": 0,
                    "discarded_warmup": True,
                    "task_id": "C2T_STUB_DISCARD",
                    "canary_relative_drift": 0.05,
                    "canary_admissible": True,
                },
                {
                    "execution_position": 1,
                    "discarded_warmup": False,
                    "task_id": "C2T_STUB_0",
                    "canary_relative_drift": 0.01,
                    "canary_admissible": True,
                },
                {
                    "execution_position": 2,
                    "discarded_warmup": False,
                    "task_id": "C2T_STUB_1",
                    "canary_relative_drift": 0.01,
                    "canary_admissible": True,
                },
            ],
            "n_timed_tasks": 2,
            "n_executed_including_discard": 3,
        },
    }


def _efilter_startup_dry_run(
    *, root: Path, cfg: dict[str, Any], allow_dirty: bool
) -> dict[str, Any]:
    """Exercise the emit/seal path in a temp repo before any measurement (C2.4 / C2e).

    Constructs the summary via :func:`build_efilter_summary` (same builder as the real run) with
    stub records, then emit→validate→verify_sealed using :func:`efilter_output_paths` so
    ``outputs.tasks`` and every future summary field are covered structurally.

    Also emits non-ASCII through the real logger and the real print path so a cp1252
    stdout/stderr surfaces in seconds rather than after a completed sweep (C2d §2.2).
    """
    import tempfile

    from seam.manifest import emit
    from seam.rawstore import verify_sealed

    del allow_dirty  # isolated temp repo always records its own clean git commit
    # Fail fast on the same write path that destroyed a161f89e (print) and the logger path.
    log_event(
        "efilter.startup_dry_run_encoding_probe",
        message=NON_ASCII_PROBE,
        probe=NON_ASCII_PROBE,
    )
    print(NON_ASCII_PROBE)
    kwargs = stub_efilter_summary_kwargs(cfg, mode="startup_dry_run")
    summary = build_efilter_summary(**kwargs)
    assert_acyclic(summary, label="efilter startup dry-run summary")
    output_paths = efilter_output_paths()

    def _write_stub_outputs(run_dir: Any) -> dict[str, Any]:
        steps_path = run_dir.path / STEPS_FILENAME
        if not steps_path.exists():
            steps_path.write_text("", encoding="utf-8")
        tasks_path = run_dir.path / _TASKS_FILENAME
        if not tasks_path.exists():
            tasks_path.write_text(
                json.dumps({"stub": True, "task_id": "C2T_STUB_0"}, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        (run_dir.path / "efilter_run.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
        )
        return {"n_steps": 0, "steps_file": STEPS_FILENAME, "tasks_file": _TASKS_FILENAME}

    with tempfile.TemporaryDirectory(prefix="seam-efilter-output-dryrun-") as tmp:
        tmp_root = Path(tmp)
        dry_config = _prepare_isolated_emit_root(tmp_root, root)
        handle = emit(
            config=dry_config,
            target="cpu-p",
            workload={
                "kind": "efilter_pilot",
                "benchmark": "synthetic_summary_serialization_dry_run",
                "task_ids": [t["task_id"] for t in summary["per_task"]],
                "seed": int(cfg["workload"]["seed"]),
                "n_repeats": 1,
                "concurrency": 1,
            },
            condition_label="efilter_output_path_dry_run",
            repo_root=tmp_root,
            allow_dirty=False,
            summary=summary,
            # Same outputs catalog as real _emit - this is what caught e66701aa (C2e).
            outputs=output_paths,
            before_integrity_hash=_write_stub_outputs,
            power_state=manifest_power_state(capture_power_state(), background_quiesced=True),
            thermal={"regime": "confound", "excluded": False},
            self_check="pass",
        )
        if not verify_sealed(handle.run_dir):
            raise SeamError(
                f"efilter startup output-path dry-run seal verification failed for {handle.run_id}"
            )
        # RunHandle.run_dir is a RunDir (.path); tests may stub a bare Path.
        sealed_dir = getattr(handle.run_dir, "path", handle.run_dir)
        sealed_summary = json.loads((sealed_dir / "summary.json").read_text(encoding="utf-8"))
        sealed_manifest = json.loads((sealed_dir / "manifest.json").read_text(encoding="utf-8"))
        if "tasks" not in (sealed_manifest.get("outputs") or {}):
            raise SeamError(
                "efilter dry-run sealed manifest missing outputs.tasks - builder/emit path drift"
            )
        return {
            "passed": True,
            "dry_run_run_id": handle.run_id,
            "outside_project_raw": True,
            "seal_verified": True,
            "experiment_id": cfg["experiment_id"],
            "paging_exclude_on_failure": bool(
                cfg.get("paging", {}).get("exclude_on_failure", True)
            ),
            "encoding_probe_emitted": True,
            "encoding_probe": NON_ASCII_PROBE,
            "summary_builder": "build_efilter_summary",
            "outputs_catalog": output_paths,
            "summary_key_count": len(nested_key_set_structure(sealed_summary)),
        }


def _power_brief(state: PowerState) -> dict[str, Any]:
    return {
        "on_battery": state.on_battery,
        "battery_pct": state.battery_pct,
        "charging": state.charging,
        "battery_saver": state.battery_saver,
    }


def _context_growth(results: list[TaskResult]) -> dict[str, Any]:
    """The pilot's decisive number: does context keep growing, or plateau?"""
    by_step: dict[int, list[int]] = {}
    for result in results:
        for step in result.steps:
            by_step.setdefault(step.step_idx, []).append(step.context_tokens_total)
    curve = [
        {
            "step_idx": idx,
            "n": len(values),
            "mean_context_tokens": statistics.fmean(values),
            "min_context_tokens": min(values),
            "max_context_tokens": max(values),
        }
        for idx, values in sorted(by_step.items())
    ]
    first = curve[0]["mean_context_tokens"] if curve else 0.0
    last = curve[-1]["mean_context_tokens"] if curve else 0.0
    per_step_growth = [
        curve[i]["mean_context_tokens"] - curve[i - 1]["mean_context_tokens"]
        for i in range(1, len(curve))
    ]
    return {
        "curve": curve,
        "growth_ratio_last_over_first": (last / first) if first else None,
        "mean_growth_tokens_per_step": (
            statistics.fmean(per_step_growth) if per_step_growth else None
        ),
        "max_context_tokens_observed": max((c["max_context_tokens"] for c in curve), default=0),
        "max_realized_steps": max((r.realized_steps for r in results), default=0),
    }


def per_task_context_ratio(context_tokens_by_step: list[int]) -> float | None:
    """``C_max/C_min`` from a task's ``context_tokens_by_step`` (min/max, not first/last)."""
    if not context_tokens_by_step:
        return None
    c_min = min(context_tokens_by_step)
    c_max = max(context_tokens_by_step)
    if c_min <= 0:
        return None
    return float(c_max) / float(c_min)


def median_context_ratio(per_task_ratios: list[float]) -> float | None:
    """Median of finite per-task ``C_max/C_min`` values."""
    values = [float(r) for r in per_task_ratios if r is not None and r > 0]
    if not values:
        return None
    return float(statistics.median(values))


def timed_block_invalidation_count(
    *,
    task_invalidations: int,
    warmup_invalidations: int = 0,
) -> dict[str, int]:
    """Split invalidation counts: only ``inside_timed_blocks`` feeds the pilot gate (C2d)."""
    return {
        "inside_timed_blocks": int(task_invalidations),
        "warmup_not_gated": int(warmup_invalidations),
    }


def evaluate_pilot_context_gate(
    growth: dict[str, Any],
    *,
    per_task: list[dict[str, Any]] | None = None,
    min_median_context_ratio: float = PILOT_MIN_MEDIAN_CONTEXT_RATIO_DEFAULT,
    memory_or_canary_invalidations: int = 0,
    paging_invalidations: int = 0,
    canary_invalidations: int = 0,
    memory_floor_invalidations: int | None = None,
    # Legacy kwarg accepted only to fail closed if a caller still passes the 20k gate.
    min_peak_context_tokens: int | None = None,
) -> dict[str, Any]:
    """Pilot gate: median ``C_max/C_min`` ≥ 3.0; paging/canary recorded, not clearance-fatal.

    Absolute peak context is recorded for reporting only - it is never a clearance criterion.
    Ratio definition: ``max(context_tokens_by_step) / min(context_tokens_by_step)`` per task.

    Given C2f wall-clock timeout audit NONE, canary drift cannot change the primary envelope
    endpoint (peak KV / context). ``paging_invalidations`` and ``canary_invalidations`` are
    recorded but do **not** refuse clearance - timing endpoints filter them in analysis.
    ``memory_floor_invalidations`` (and legacy residual non-canary non-paging counts) still
    refuse clearance. Warmup / discarded warmup are outside these counts (C2d/C2f).
    """
    if min_peak_context_tokens is not None:
        raise ConfigError(
            "evaluate_pilot_context_gate no longer accepts min_peak_context_tokens (the 20k peak "
            "gate was withdrawn under C2b). Use min_median_context_ratio."
        )
    tasks = list(per_task or [])
    ratios = [
        float(t["context_ratio_cmax_over_cmin"])
        for t in tasks
        if t.get("context_ratio_cmax_over_cmin") is not None
    ]
    if not ratios and growth.get("per_task_ratios"):
        ratios = [float(r) for r in growth["per_task_ratios"] if r is not None]
    median_ratio = median_context_ratio(ratios)
    peak = int(growth.get("max_context_tokens_observed") or 0)
    ratio_ok = median_ratio is not None and median_ratio >= float(min_median_context_ratio)
    # C2f: canary is timing-only (audit NONE). Legacy callers that only pass
    # memory_or_canary_invalidations without a canary split still treat the combined count as
    # clearance-fatal unless canary_invalidations is supplied explicitly.
    if memory_floor_invalidations is None:
        # Prefer explicit split when provided via canary_invalidations kwarg path.
        memory_floor_invalidations = max(
            0, int(memory_or_canary_invalidations) - int(canary_invalidations)
        )
    clearance_fatal = int(memory_floor_invalidations)
    memory_ok = clearance_fatal == 0
    cleared = bool(ratio_ok and memory_ok)
    stop_parts: list[str] = []
    if not ratio_ok:
        stop_parts.append(
            f"STOP: pilot median context ratio {median_ratio} < {min_median_context_ratio} "
            f"(C_max/C_min via min/max of context_tokens_by_step). Absolute peak {peak} is "
            f"reported only - not a gate."
        )
    if not memory_ok:
        stop_parts.append(
            f"STOP: {clearance_fatal} clearance-fatal invalidation(s) inside timed blocks "
            f"(memory floor / other); pilot requires zero on endpoints the quantity can "
            f"causally affect. Paging and canary invalidations are recorded per block and are "
            f"not run-fatal for clearance (C2e/C2f; wall-clock timeout audit NONE)."
        )
    return {
        "gate": "median_context_ratio_cmax_over_cmin",
        "ratio_definition": "max(context_tokens_by_step)/min(context_tokens_by_step) per task",
        "min_median_context_ratio": float(min_median_context_ratio),
        "median_context_ratio": median_ratio,
        "per_task_context_ratios": ratios,
        "max_context_tokens_observed": peak,
        "max_context_tokens_is_gate": False,
        "memory_or_canary_invalidations": int(memory_or_canary_invalidations),
        "memory_or_canary_invalidations_scope": (
            "legacy_combined_non_paging; canary no longer clearance-fatal under C2f"
        ),
        "canary_invalidations": int(canary_invalidations),
        "canary_invalidations_run_fatal": False,
        "canary_invalidations_scope": "inside_timed_blocks_recorded_not_gated_c2f",
        "memory_floor_invalidations": int(memory_floor_invalidations),
        "memory_floor_invalidations_run_fatal": True,
        "paging_invalidations": int(paging_invalidations),
        "paging_invalidations_run_fatal": False,
        "paging_invalidations_scope": "inside_timed_blocks_recorded_not_gated",
        "primary_endpoint": "peak_kv_envelope",
        "wallclock_timeout_audit_verdict": "NONE",
        "cleared": cleared,
        "stop_message": " ".join(stop_parts) if stop_parts else None,
    }


def _n_out_constant_from_pilot(results: list[TaskResult]) -> dict[str, Any]:
    """Single median completion length, applied to every step type (C2b)."""
    values = [step.completion_tokens for result in results for step in result.steps]
    by_type: dict[str, list[int]] = {}
    for result in results:
        for step in result.steps:
            by_type.setdefault(step.step_type, []).append(step.completion_tokens)
    if not values:
        return {
            "median_completion_tokens": None,
            "n_steps": 0,
            "n_out_pred_tokens": None,
            "constant_across_step_types": True,
            "per_type_medians_diagnostic_only": {},
        }
    median = int(statistics.median(values))
    return {
        "median_completion_tokens": median,
        "n_steps": len(values),
        "n_out_pred_tokens": {
            "tool_call_synthesis": median,
            "answer_synthesis": median,
        },
        "constant_across_step_types": True,
        "rationale": (
            "C2b keeps n_out_pred constant so t_pred ranking equals context ranking; freeze BOTH "
            "keys from this measured median after the pilot."
        ),
        "per_type_medians_diagnostic_only": {
            step_type: int(statistics.median(vals)) for step_type, vals in by_type.items() if vals
        },
    }


def _tool_call_rate(results: list[TaskResult]) -> dict[str, Any]:
    n_steps = 0
    n_well_formed = 0
    n_none = 0
    for result in results:
        for step in result.steps:
            n_steps += 1
            if step.tool is None:
                n_none += 1
            else:
                n_well_formed += 1
    return {
        "n_steps": n_steps,
        "n_well_formed_tool_calls": n_well_formed,
        "n_tool_none": n_none,
        "well_formed_rate": (n_well_formed / n_steps) if n_steps else None,
        "tool_none_rate": (n_none / n_steps) if n_steps else None,
    }


def _assert_n_out_constant(n_out_pred: dict[str, int]) -> None:
    values = {int(v) for v in n_out_pred.values()}
    if len(values) != 1:
        raise ConfigError(
            "C2b requires n_out_pred constant across step types; got "
            f"{n_out_pred}. Set tool_call_synthesis and answer_synthesis to the same value."
        )


# ==================================================================================================
# Emission
# ==================================================================================================


def _emit(
    *,
    root: Path,
    platform_cfg: Any,
    cfg: dict[str, Any],
    report: dict[str, Any],
    steps: list[StepRecord],
    kind: str,
    allow_dirty: bool,
    self_check: str,
    existing_run_dir: RunDir | None = None,
    run_id: str | None = None,
) -> str:
    from seam.manifest import emit

    spec_path = root / cfg["models"]["local"]["spec"]
    spec = load_local_spec(spec_path)
    model_block = manifest_model_block(
        spec=spec, spec_path=spec_path, reasoning_mode=cfg["reasoning_mode"]
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

    quiesce = report["quiesce"]

    def _write_outputs(run_dir: Any) -> dict[str, Any]:
        steps_path = run_dir.path / STEPS_FILENAME
        if not steps_path.exists():
            # Fallback when the early-run-dir path was not used.
            with steps_path.open("w", encoding="utf-8") as handle:
                for step in steps:
                    line = json.dumps(step_to_record(step), sort_keys=True, allow_nan=False)
                    handle.write(line + "\n")
        (run_dir.path / "efilter_run.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
        )
        in_progress = run_dir.path / "in_progress.json"
        if in_progress.exists():
            in_progress.unlink()
        return {"n_steps": len(steps), "steps_file": STEPS_FILENAME}

    handle = emit(
        config=platform_cfg,
        # Placement, not confinement: OpenVINO's default was MEASURED to load the P-cores on this
        # part (run 5eb09eba...), and confinement_mechanism stays null because none was applied.
        target="cpu-p",
        workload={
            "kind": kind,
            "benchmark": cfg["workload"]["benchmark"],
            "task_ids": [t["task_id"] for t in report["per_task"]],
            "seed": int(cfg["workload"]["seed"]),
            "n_repeats": 1,
            "concurrency": 1,
        },
        condition_label=f"efilter_{report['arm']['id']}_escalation_disabled",
        repo_root=root,
        run_id=run_id,
        existing_run_dir=existing_run_dir,
        allow_dirty=allow_dirty,
        summary=report,
        model=model_block,
        drivers={
            "openvino": report["openvino"]["openvino"],
            "genai": report["openvino"]["genai"],
        },
        power_state=manifest_power_state(
            capture_power_state(),
            battery_pct_end=quiesce.get("battery_pct"),
            display_brightness=quiesce.get("display_brightness"),
            defender_realtime=quiesce.get("defender_realtime"),
            background_quiesced=not quiesce.get("forbidden_process_hits"),
        ),
        thermal={
            "regime": "confound",
            "warmup_s": float(cfg["thermal"]["warmup_s"]),
            "excluded": False,
            "ambient_c": quiesce.get("ambient_c"),
            "pkg_temp_series_path": None,
        },
        # C2e: outputs.tasks is a legitimate artifact path (tasks.jsonl). Schema must allow it -
        # the emitter was correct; rejecting the property (e66701aa) was a schema omission under
        # additionalProperties:false. Dry-run uses efilter_output_paths() for the same catalog.
        outputs=efilter_output_paths(),
        before_integrity_hash=_write_outputs,
        self_check=self_check,
    )
    return handle.run_id


# ==================================================================================================
# CLI
# ==================================================================================================


def main(argv: list[str] | None = None) -> int:
    # Before any logging / print of non-ASCII (C2d). Belt-and-braces with PYTHONIOENCODING.
    configure_utf8_stdio()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["pilot", "full"])
    parser.add_argument("--platform", default="aipc-c1")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument(
        "--n-tasks",
        type=int,
        default=None,
        help="override the declared N; the config value is the pre-registered one",
    )
    args = parser.parse_args(argv)

    root = repo_root(Path(__file__).parent)
    cfg, cfg_sha = _load_cfg(root)
    platform_cfg = load_platform_config(args.platform, repo_root=root)
    workload = load_workload(root / cfg["workload"]["task_list"])
    _assert_task_list(cfg, workload)

    label = f"efilter/{args.mode}"
    kind = "efilter_pilot" if args.mode == "pilot" else "efilter_stage1"
    n_tasks = args.n_tasks or int(
        cfg["workload"]["n_tasks_pilot" if args.mode == "pilot" else "n_tasks_full"]
    )
    min_median_ratio = float(
        cfg["workload"].get(
            "pilot_min_median_context_ratio", PILOT_MIN_MEDIAN_CONTEXT_RATIO_DEFAULT
        )
    )
    out_dir = root / "derived" / "efilter"
    out_dir.mkdir(parents=True, exist_ok=True)
    clearance_path = out_dir / _CLEARANCE_FILENAME

    if args.mode == "full" and bool(cfg["workload"].get("require_pilot_context_clearance")):
        if not clearance_path.is_file():
            raise SystemExit(
                "STOP: workload.require_pilot_context_clearance is set but "
                f"{clearance_path} is missing. Run the pilot and clear the median "
                f"C_max/C_min >= {min_median_ratio} ratio gate first."
            )
        clearance = json.loads(clearance_path.read_text(encoding="utf-8"))
        if not clearance.get("cleared"):
            raise SystemExit(
                "STOP: prior pilot did not clear the C2b context-ratio gate; full run refused. "
                f"See {clearance_path}."
            )
        if clearance.get("gate") == "min_peak_context_tokens" or (
            "min_peak_context_tokens" in clearance and "median_context_ratio" not in clearance
        ):
            raise SystemExit(
                "STOP: clearance artifact still records the withdrawn 20k peak gate. "
                "Re-run the C2b pilot so clearance uses median_context_ratio_cmax_over_cmin."
            )

    print(f"E-FILTER {cfg['experiment_id']} - {args.mode}: N={n_tasks} tasks, escalation DISABLED")
    # C2.4: verify exclusionary paging is the active measurement default.
    if cfg.get("paging", {}).get("exclude_on_failure") is not True:
        raise SystemExit(
            "STOP: measurement paging.exclude_on_failure must be true for E-FILTER "
            "(exclusionary paging gate). configs/measurement.yaml default is true."
        )

    quiesce = _quiesce(cfg, platform_cfg)
    if quiesce["deviations"]:
        print("QUIESCE REFUSAL:")
        for deviation in quiesce["deviations"]:
            print(f"  - {deviation}")
        refusal = out_dir / f"refused_quiesce_{args.mode}.json"
        refusal.write_text(json.dumps(quiesce, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {refusal}")
        return 3

    dry = _efilter_startup_dry_run(root=root, cfg=cfg, allow_dirty=args.allow_dirty)
    print(
        json.dumps({"event": "efilter.startup_output_path_dry_run_passed", **dry}, sort_keys=True)
    )

    run_id = str(uuid.uuid4())
    run_dir = open_in_progress_run(
        root=root,
        run_id=run_id,
        marker={
            "run_id": run_id,
            "state": "IN_PROGRESS",
            "lifecycle": "PARTIAL/INCOMPLETE until sealed",
            "created_utc": utc_now_iso(),
            "experiment_id": cfg["experiment_id"],
            "mode": args.mode,
            "startup_output_path_dry_run": dry,
            "paging_exclude_on_failure": True,
        },
    )
    writer = StepLogWriter(run_dir.path / STEPS_FILENAME)

    backend = _local_backend(root, cfg)
    verdict = backend.preflight()
    if verdict.status != "OK":
        raise SystemExit(f"local preflight {verdict.status}: {verdict.reason}")
    kv = load_kv_geometry(
        Path(load_local_spec(root / cfg["models"]["local"]["spec"])["ir_dir"]),
        device=cfg["openvino"]["device"],
        assumed_kv_dtype=cfg["openvino"]["assumed_kv_dtype"],
    )
    print(f"KV: {kv.bytes_per_token} bytes/token ({kv.kv_dtype} via {kv.kv_dtype_source})")

    if args.mode == "pilot":
        n_out_pred = dict(cfg["policy"]["n_out_pred_tokens_pilot"])
        n_out_source = (
            "config seed (pilot); freeze a SINGLE constant from the measured median "
            "completion tokens for the full run (both step types identical)"
        )
    else:
        frozen = cfg["policy"].get("n_out_pred_tokens")
        if not frozen:
            raise SystemExit(
                "policy.n_out_pred_tokens is null. Run the pilot first and freeze it from the "
                "observed median - the full run must not seed the router from a guess."
            )
        n_out_pred = dict(frozen)
        n_out_source = (
            "frozen from the pilot's measured median completion tokens "
            "(configs/efilter.yaml; constant across step types)"
        )
    _assert_n_out_constant(n_out_pred)

    p_cpus = [int(cpu) for cpu in platform_cfg.require("topology.p_cpus")]
    launch_free_memory_mb = free_memory_mb()
    print(f"launch_free_memory_mb={launch_free_memory_mb:.3f}")
    hits = check_forbidden_processes(cfg["quiesce"]["forbidden_processes"])
    if hits:
        raise SeamError(f"secondary process-name quiescence refusal: {hits}")

    # Warmup + cache probe under lock+canary, but WITHOUT paging sampling (C2d).
    # Model load / warmup generation fault pages by construction; that must not gate the pilot.
    with machine_measurement(
        repo_root=root,
        label=f"efilter/{args.mode}/warmup",
        config=cfg,
        p_cpus=p_cpus,
        sample_paging=False,
    ) as warmup_block:
        warmup_block.record["secondary_process_name_check"] = {
            "patterns": list(cfg["quiesce"]["forbidden_processes"]),
            "hits": hits,
            "role": "secondary to measured-load quiescence",
        }
        print(f"warming to steady state ({cfg['thermal']['warmup_s']}s)")
        throughput, warmup = _warm_and_measure(backend, cfg)
        print(
            f"throughput: prefill {throughput.r_prefill_tok_s:.2f} tok/s, "
            f"decode {throughput.r_decode_tok_s:.2f} tok/s"
        )
        # After warmup, never before: on a cold pipeline the first call carries lazy initialization
        # and every TTFT ratio would measure that instead of caching.
        cache = _cache_probe(backend, cfg)
        print(f"cache: {cache['verdict']} (instrumented={cache['cache_instrumented']})")
    warmup_machine = {
        **dict(warmup_block.record),
        **paging_admissibility(dict(warmup_block.record)),
        **canary_admissibility(dict(warmup_block.record)),
    }
    warmup_invalidations = 0
    if not warmup_machine.get("valid", True):
        warmup_invalidations = len(warmup_machine.get("invalid_reasons") or []) or 1
        print(
            "WARNING: warmup block invalidated (not gated; outside paging window): "
            + "; ".join(warmup_machine.get("invalid_reasons") or [])
        )

    t_run0 = time.perf_counter()
    results, per_task, task_machine_blocks, task_invalidations, discard_meta = _run_tasks(
        root=root,
        cfg=cfg,
        workload=workload,
        backend=backend,
        throughput=throughput,
        n_out_pred=n_out_pred,
        kv=kv,
        n_tasks=n_tasks,
        label=label,
        writer=writer,
        platform_cfg=platform_cfg,
        p_cpus=p_cpus,
        launch_free_memory_mb=launch_free_memory_mb,
        run_dir=run_dir,
        run_id=run_id,
    )
    run_wall_s = time.perf_counter() - t_run0
    inv_split = timed_block_invalidation_count(
        task_invalidations=task_invalidations,
        warmup_invalidations=warmup_invalidations,
    )
    # C2e/C2f: paging+canary recorded per endpoint; only memory-floor/other refuse clearance.
    paging_split = timed_block_invalidation_split(per_task)
    non_paging_invalidation_count = int(
        paging_split["non_paging_invalidations_inside_timed_blocks"]
    )
    paging_invalidation_count = int(paging_split["paging_invalidations_inside_timed_blocks"])
    canary_invalidation_count = int(paging_split["canary_invalidations_inside_timed_blocks"])
    clearance_fatal_count = int(paging_split["clearance_fatal_invalidations_inside_timed_blocks"])
    mem_series = memory_series_summary(per_task)

    steps = [step for result in results for step in result.steps]
    growth = _context_growth(results)
    growth["per_task_ratios"] = [t.get("context_ratio_cmax_over_cmin") for t in per_task]
    growth["median_context_ratio"] = median_context_ratio(
        [r for r in growth["per_task_ratios"] if r is not None]
    )
    gate = evaluate_pilot_context_gate(
        growth,
        per_task=per_task,
        min_median_context_ratio=min_median_ratio,
        memory_or_canary_invalidations=non_paging_invalidation_count,
        paging_invalidations=paging_invalidation_count,
        canary_invalidations=canary_invalidation_count,
        memory_floor_invalidations=clearance_fatal_count,
    )
    n_out_freeze = _n_out_constant_from_pilot(results)
    tool_rate = _tool_call_rate(results)
    t_pred_values = [float(s.t_pred_total_s or 0.0) for s in steps if s.t_pred_total_s is not None]
    max_t_pred = max(t_pred_values, default=0.0)
    report = build_efilter_summary(
        cfg=cfg,
        mode=args.mode,
        cfg_sha=cfg_sha,
        workload_sha=workload.sha256,
        model_spec_sha=sha256_file(root / cfg["models"]["local"]["spec"]),
        dry_run_meta=dry,
        launch_free_memory_mb=launch_free_memory_mb,
        mem_series=mem_series,
        quiesce=quiesce,
        openvino_info=asdict(runtime_info()),
        backend_config=backend.config_record(),
        kv_geometry=kv.to_record(),
        cache_probe=cache,
        cache_instrumented=bool(cache["cache_instrumented"]),
        throughput_block={
            "target": throughput.target,
            "r_prefill_tok_s": throughput.r_prefill_tok_s,
            "r_decode_tok_s": throughput.r_decode_tok_s,
            "measured_by_run_id": throughput.measured_by_run_id,
            "warmup": warmup,
        },
        warmup=warmup,
        n_out_pred=n_out_pred,
        n_out_source=n_out_source,
        n_out_freeze=n_out_freeze,
        warmup_machine=warmup_machine,
        task_machine_blocks=task_machine_blocks,
        inv_split=inv_split,
        paging_split=paging_split,
        growth=growth,
        gate=gate,
        tool_rate=tool_rate,
        t_pred_values=t_pred_values,
        per_task=per_task,
        n_tasks=n_tasks,
        n_steps=len(steps),
        run_wall_s=run_wall_s,
        success_rate=(sum(1 for r in results if r.success) / len(results) if results else None),
        max_t_pred=max_t_pred,
        escalated_steps=sum(1 for s in steps if s.assigned_target != "local"),
        discard_meta=discard_meta,
    )
    memory_vs_context = report["memory_vs_context"]

    # Persist report JSON, then seal, THEN fancy console report. a161f89e lost the seal because
    # a UnicodeEncodeError on a decorative print ran before _emit (C2d / C9).
    (out_dir / f"{args.mode}_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )

    pilot_failed = args.mode == "pilot" and not gate["cleared"]
    from seam.rawstore import open_run_dir, verify_sealed

    sealed_id = _emit(
        root=root,
        platform_cfg=platform_cfg,
        cfg=cfg,
        report={
            **report,
            "run_id": run_id,
            **({"pilot_gate_failed": True} if pilot_failed else {}),
        },
        steps=steps,
        kind=kind,
        allow_dirty=args.allow_dirty,
        self_check="pass",
        existing_run_dir=run_dir,
        run_id=run_id,
    )
    if not verify_sealed(open_run_dir(sealed_id, repo_root=root)):
        raise SeamError(
            f"verify_sealed failed for {sealed_id} immediately after emit; "
            f"do not treat this run as sealed"
        )
    report["run_id"] = sealed_id
    report["verify_sealed"] = True
    # C2g: fill peak-memory-vs-context note from timed tasks (warmup discarded).
    try:
        from seam.analysis.c2g_memory import write_memory_vs_context_note

        write_memory_vs_context_note(
            out_dir / "c2g_memory_vs_context_note.json",
            analysis=memory_vs_context,
            run_id=sealed_id,
        )
    except Exception as exc:
        print(f"WARNING: c2g_memory_vs_context_note write failed ({type(exc).__name__}: {exc})")
    (out_dir / f"{args.mode}_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )

    def _post_seal_report() -> None:
        if pilot_failed:
            print(gate["stop_message"])
        print("\ncontext growth (mean tokens by step_idx):")
        for row in growth["curve"]:
            print(f"  step {row['step_idx']}: {row['mean_context_tokens']:.0f} (n={row['n']})")
        print(
            f"growth ratio last/first: {growth['growth_ratio_last_over_first']}, "
            f"max context {growth['max_context_tokens_observed']} tokens (reported, not a gate)"
        )
        print(
            f"per-task C_max/C_min: {growth['per_task_ratios']}; "
            f"median={growth['median_context_ratio']}"
        )
        print(
            f"pilot context gate: median_ratio={gate['median_context_ratio']} "
            f"min={gate['min_median_context_ratio']} "
            f"canary_invalidations={gate['canary_invalidations']} "
            f"(not clearance-fatal) "
            f"paging_invalidations={gate['paging_invalidations']} "
            f"(not clearance-fatal) "
            f"memory_floor_invalidations={gate['memory_floor_invalidations']} "
            f"warmup_not_gated={inv_split['warmup_not_gated']} cleared={gate['cleared']}"
        )
        print(
            "canary drift by execution_position: "
            + ", ".join(
                f"pos={row.get('execution_position')} "
                f"drift={row.get('canary_relative_drift')} "
                f"adm={row.get('canary_admissible')}"
                for row in (discard_meta.get("canary_drift_by_execution_position") or [])
            )
        )
        print(
            f"memory series monotonic free-before decline="
            f"{mem_series['monotonic_decline_free_memory_mb_before_task']} "
            f"rss-after increase={mem_series['monotonic_increase_process_rss_after']}"
        )
        rss_fit = memory_vs_context.get("rss_peak_vs_context") or {}
        print(
            f"C2g memory vs context: verdict={memory_vs_context.get('overall_verdict')} "
            f"n_points={memory_vs_context.get('n_points')} "
            f"rss_linear_R2={(rss_fit.get('linear') or {}).get('r_squared')} "
            f"rss_quad_R2={(rss_fit.get('quadratic') or {}).get('r_squared')}"
        )
        for row in per_task:
            print(
                f"  mem {row.get('task_id')}: free_min={row.get('free_memory_mb_min')} "
                f"@step={row.get('free_memory_mb_min_step_idx')} "
                f"ctx={row.get('free_memory_mb_min_context_tokens')} "
                f"rss_peak={row.get('process_rss_peak')} "
                f"@step={row.get('process_rss_peak_step_idx')}"
            )
        print(
            f"output-length median={n_out_freeze['median_completion_tokens']} "
            f"-> n_out_pred freeze {n_out_freeze['n_out_pred_tokens']}; "
            f"tool well-formed rate={tool_rate['well_formed_rate']}"
        )
        print(f"\nsealed run_id={sealed_id}")
        print(f"steps={len(steps)} wall={run_wall_s:.0f}s")

    try:
        _post_seal_report()
    except Exception as exc:
        print(
            f"WARNING: post-seal report print failed ({type(exc).__name__}: {exc}); "
            f"run is sealed as {sealed_id}"
        )

    # C2c: clearance only after full pass AND seal. Never write clearance on a failed gate.
    if args.mode == "pilot":
        if gate["cleared"] and sealed_id:
            clearance_path.write_text(
                json.dumps(
                    {
                        **gate,
                        "experiment_id": cfg["experiment_id"],
                        "written_utc": utc_now_iso(),
                        "run_id": sealed_id,
                        "sealed": True,
                        "context_growth": growth,
                        "memory_series": mem_series,
                        "c2c": True,
                        "c2d": True,
                        "c2e": True,
                        "c2f": True,
                        "c2g": True,
                        "context_cap_tokens": cfg["workload"].get("context_cap_tokens"),
                        "paging_window_scope": "timed_task_blocks_only",
                        "paging_gate_scope": "per_endpoint_admissibility_c2e",
                        "canary_gate_scope": "per_endpoint_admissibility_c2f",
                        "memory_floor_invalidations_run_fatal": True,
                        "paging_invalidations_run_fatal": False,
                        "canary_invalidations_run_fatal": False,
                        "wallclock_timeout_audit_verdict": "NONE",
                        "discard_first_task_warmup": bool(
                            discard_meta.get("discard_first_task_warmup")
                        ),
                        "warmup_invalidations_not_gated": int(inv_split["warmup_not_gated"]),
                        "memory_vs_context": memory_vs_context,
                    },
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
            print(f"wrote clearance {clearance_path}")
        else:
            print(
                f"no clearance written (cleared={gate['cleared']} sealed={bool(sealed_id)}); "
                f"see {out_dir / f'{args.mode}_report.json'}"
            )
            return 2

    if clearance_fatal_count > 0 and args.mode != "pilot":
        # Full mode: seal evidence but refuse a clean exit on memory-floor/other failures.
        # Paging/canary inadmissibility is recorded per block and analyzed per-endpoint (C2e/C2f).
        print(
            f"STOP: {clearance_fatal_count} clearance-fatal invalidation(s) inside timed "
            f"blocks on full run; exit non-zero (sealed run_id={sealed_id}). "
            f"paging_invalidations={paging_invalidation_count} "
            f"canary_invalidations={canary_invalidation_count} recorded, not exit-fatal."
        )
        return 2
    if paging_invalidation_count > 0 or canary_invalidation_count > 0:
        print(
            f"NOTE: paging_invalidations={paging_invalidation_count} "
            f"canary_invalidations={canary_invalidation_count} inside timed blocks "
            f"recorded with per-endpoint admissibility; not clearance-fatal (C2e/C2f). "
            f"sealed run_id={sealed_id}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
