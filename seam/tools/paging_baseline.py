"""Fix A - seal idle and generation Page-Reads/sec baselines before the A2 sweep.

Samples Windows ``\\Memory\\Page Reads/sec`` at 1 Hz under the machine lock, reports the
distribution (median, p95, max, fraction > 0), and seals both phases under one run_id so the
paging gate decision is reconstructible from raw/.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import uuid
from pathlib import Path
from typing import Any

from seam.config import resolve_config
from seam.errors import SeamError
from seam.gitinfo import repo_root
from seam.locks import exclusive
from seam.manifest import emit
from seam.powerstate import capture_power_state, manifest_power_state
from seam.telemetry.memory import _hard_page_reads_per_s

__all__ = ["distribution_stats", "main", "sample_page_reads"]


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


def distribution_stats(rates: list[float]) -> dict[str, Any]:
    """Median / p95 / max / fraction>0 for a page-read rate series (AM-027(b) numbers)."""
    if not rates:
        raise SeamError("page-read baseline produced no samples")
    ordered = sorted(rates)
    n = len(ordered)
    # nearest-rank p95
    p95_index = min(n - 1, max(0, round(0.95 * (n - 1))))
    above_zero = sum(1 for rate in rates if rate > 0.0)
    return {
        "n_samples": n,
        "median": float(statistics.median(ordered)),
        "p95": float(ordered[p95_index]),
        "max": float(ordered[-1]),
        "fraction_above_zero": above_zero / n,
        "n_above_zero": above_zero,
        "min": float(ordered[0]),
        "mean": float(statistics.fmean(ordered)),
    }


def sample_page_reads(
    *,
    duration_s: float,
    interval_s: float = 1.0,
) -> dict[str, Any]:
    """Sample ``\\Memory\\Page Reads/sec`` for ``duration_s`` at ``interval_s`` (default 1 Hz)."""
    import math

    if duration_s < 60.0:
        raise SeamError(f"Fix A idle baseline requires duration_s >= 60; got {duration_s}")
    if interval_s <= 0:
        raise SeamError("sample interval must be positive")
    n_target = max(1, math.ceil(duration_s / interval_s))
    samples: list[dict[str, Any]] = []
    methods: set[str] = set()
    started = time.monotonic()
    while len(samples) < n_target:
        rate, method = _hard_page_reads_per_s()
        methods.add(method)
        if rate is None:
            raise SeamError(f"Page Reads/sec unavailable during baseline: {method}")
        samples.append({"timestamp_utc": _utc_now(), "hard_page_reads_per_s": float(rate)})
        if len(samples) >= n_target:
            break
        # Keep wall spacing near interval_s without dropping below n_target.
        target_elapsed = len(samples) * interval_s
        sleep_s = target_elapsed - (time.monotonic() - started)
        if sleep_s > 0:
            time.sleep(sleep_s)
    rates = [float(row["hard_page_reads_per_s"]) for row in samples]
    return {
        "duration_s_requested": duration_s,
        "interval_s": interval_s,
        "n_target": n_target,
        "elapsed_s": time.monotonic() - started,
        "method": sorted(methods),
        "samples": samples,
        "rates": rates,
        "distribution": distribution_stats(rates),
    }


def _decide_threshold(idle: dict[str, Any], generation: dict[str, Any]) -> dict[str, Any]:
    """Decide absolute-zero vs baseline-relative from the two sealed distributions only."""
    idle_dist = idle["distribution"]
    gen_dist = generation["distribution"]
    idle_clean = idle_dist["fraction_above_zero"] == 0.0 and idle_dist["max"] == 0.0
    gen_clean = gen_dist["fraction_above_zero"] == 0.0 and gen_dist["max"] == 0.0
    if idle_clean and gen_clean:
        return {
            "gate_mode": "absolute_zero",
            "hard_page_reads_threshold_per_s": 0.0,
            "sustained_consecutive_samples": 2,
            "margin_above_idle_p95": None,
            "justification": (
                "Both idle and generation Page Reads/sec distributions are cleanly zero "
                "(fraction_above_zero=0, max=0). Keep absolute >0 sustained gate; baselines "
                "are sealed evidence that the absolute threshold is not false-positive-prone "
                "on this platform at the time of decision."
            ),
        }
    # Idle shows periodic non-zero (or generation alone is noisy with idle clean but non-zero
    # idle is the Fix A trigger). Baseline-relative = idle p95 + stated margin.
    margin = 1.0  # page reads/sec; stated margin above idle p95
    idle_p95 = float(idle_dist["p95"])
    threshold = idle_p95 + margin
    return {
        "gate_mode": "baseline_relative",
        "hard_page_reads_threshold_per_s": threshold,
        "sustained_consecutive_samples": 2,
        "margin_above_idle_p95": margin,
        "idle_p95": idle_p95,
        "justification": (
            f"Idle Page Reads/sec is not cleanly zero "
            f"(fraction_above_zero={idle_dist['fraction_above_zero']}, "
            f"p95={idle_p95}, max={idle_dist['max']}). Gate becomes baseline-relative: "
            f"threshold = idle_p95 + margin = {idle_p95} + {margin} = {threshold} "
            f"page reads/sec, with min sustained duration of 2 consecutive samples at the "
            f"measurement sampler interval. Generation distribution "
            f"(fraction_above_zero={gen_dist['fraction_above_zero']}, "
            f"p95={gen_dist['p95']}, max={gen_dist['max']}) informed the decision but the "
            f"threshold is anchored on idle p95 only."
        ),
    }


def _run_generation_baseline(
    root: Path,
    *,
    sample_interval_s: float,
) -> dict[str, Any]:
    """One known-good generation under machine lock with the same 1 Hz page-read sampler."""
    import yaml

    from seam.agent.tools import SYSTEM_PROMPT, TOOL_SPECS
    from seam.backends.base import GenerationRequest
    from seam.backends.local_openvino import LocalOpenVinoBackend
    from seam.model_provenance import load_local_spec, quantization_summary
    from seam.telemetry.memory import MemoryPressureSampler

    prompt_cfg = yaml.safe_load((root / "configs" / "prompt_a.yaml").read_text(encoding="utf-8"))
    measurement_cfg = yaml.safe_load(
        (root / "configs" / "measurement.yaml").read_text(encoding="utf-8")
    )
    ov = prompt_cfg["openvino"]
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
    preflight = backend.preflight()
    if preflight.status != "OK":
        raise SeamError(f"generation baseline preflight {preflight.status}: {preflight.reason}")
    request = GenerationRequest(
        messages=[{"role": "user", "content": str(prompt_cfg["probe"]["prompt"])}],
        system=SYSTEM_PROMPT,
        tools=TOOL_SPECS,
        max_tokens=int(prompt_cfg["probe"]["max_tokens"]),
        temperature=0.0,
    )
    # Keep available-memory invalidation; canary/quiesce thresholds unchanged - this path only
    # characterizes page reads during a known-good generation, not the A2 validity envelope.
    paging = measurement_cfg["paging"]
    pressure = MemoryPressureSampler(
        interval_s=float(sample_interval_s),
        available_memory_min_mb=float(paging["available_memory_min_mb"]),
        sustained_nonzero_samples=int(paging["sustained_nonzero_consecutive_samples"]),
        page_read_threshold=0.0,
    )
    started = _utc_now()
    pressure.start()
    try:
        result = backend.generate(request, ignore_eos=bool(prompt_cfg["probe"]["ignore_eos"]))
    finally:
        memory_pressure = pressure.stop()
    ended = _utc_now()
    rates = [float(rate) for rate in memory_pressure["hard_page_reads_per_s"] if rate is not None]
    return {
        "phase": "generation",
        "started_utc": started,
        "ended_utc": ended,
        "interval_s": sample_interval_s,
        "method": memory_pressure.get("hard_page_reads_method"),
        "generation": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "wall_s": result.wall_ns / 1e9,
            "ttft_s": (result.ttft_ns or 0) / 1e9,
            "ttft_source": result.extra.get("ttft_source"),
        },
        "memory_pressure": memory_pressure,
        "rates": rates,
        "distribution": distribution_stats(rates),
        "available_memory_mb_min": memory_pressure.get("available_memory_mb_min"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idle-duration-s", type=float, default=60.0)
    parser.add_argument("--interval-s", type=float, default=1.0)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Permit dirty tree for this pre-commit baseline seal only (not the A2 sweep).",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--skip-generation", action="store_true")
    args = parser.parse_args(argv)
    root = repo_root(Path(__file__).parent)
    run_id = str(uuid.UUID(args.run_id)) if args.run_id else str(uuid.uuid4())
    platform = root / "configs" / "platforms" / "aipc-c1.yaml"
    measurement_path = root / "configs" / "measurement.yaml"
    resolved = resolve_config([platform, measurement_path], repo_root=root)

    lock_resource = root / ".locks" / "machine"
    with exclusive(lock_resource):
        idle_started = _utc_now()
        idle = sample_page_reads(
            duration_s=float(args.idle_duration_s),
            interval_s=float(args.interval_s),
        )
        idle["phase"] = "idle"
        idle["started_utc"] = idle_started
        idle["ended_utc"] = _utc_now()
        idle["machine_lock"] = {"resource": str(lock_resource), "held": True}

        if args.skip_generation:
            generation = {
                "phase": "generation",
                "skipped": True,
                "reason": "--skip-generation",
                "distribution": None,
            }
            decision = {
                "gate_mode": "undecided",
                "justification": "generation phase skipped; threshold not decided",
            }
        else:
            generation = _run_generation_baseline(root, sample_interval_s=float(args.interval_s))
            generation["machine_lock"] = {"resource": str(lock_resource), "held": True}
            decision = _decide_threshold(idle, generation)

    idle_summary: dict[str, Any] = {
        "run_id": run_id,
        **{k: v for k, v in idle.items() if k != "samples"},
        "samples": idle.get("samples"),
        "distribution": {
            "run_id": run_id,
            **idle["distribution"],
        },
    }
    if isinstance(idle_summary.get("samples"), list):
        idle_summary["sample_count"] = len(idle_summary["samples"])
    if generation.get("skipped"):
        generation_summary: dict[str, Any] = {"run_id": run_id, **generation}
    else:
        gen_dist = generation.get("distribution")
        generation_summary = {
            "run_id": run_id,
            **{k: v for k, v in generation.items() if k not in {"memory_pressure"}},
            "distribution": (
                None if not isinstance(gen_dist, dict) else {"run_id": run_id, **gen_dist}
            ),
            "memory_pressure": generation.get("memory_pressure"),
        }
    summary: dict[str, Any] = {
        "experiment_id": "prompt-a2-paging-baseline",
        "classification": "PROMPT_A2_PAGING_BASELINE",
        "run_id": run_id,
        "counter": r"\Memory\Page Reads/sec",
        "idle": idle_summary,
        "generation": generation_summary,
        "threshold_decision": {"run_id": run_id, **decision},
        "linked_phases": "idle+generation sealed under the same run_id",
    }
    power = capture_power_state()
    handle = emit(
        config=resolved,
        target="cpu-p",
        workload={
            "kind": "prompt_a2_paging_baseline",
            "benchmark": "page_reads_idle_and_generation",
            "task_ids": [],
            "seed": None,
            "n_repeats": 1,
            "concurrency": 1,
        },
        condition_label="prompt_a2_paging_baseline",
        repo_root=root,
        run_id=run_id,
        allow_dirty=bool(args.allow_dirty),
        summary=summary,
        power_state=manifest_power_state(power, background_quiesced=True),
        thermal={"regime": "confound", "excluded": False},
        self_check="pass",
        before_integrity_hash=lambda run_dir: {
            "idle_samples_path": str(
                run_dir.write_json(
                    "idle_page_reads.json",
                    {"run_id": run_id, "idle": idle},
                )
            ),
            "generation_samples_path": str(
                run_dir.write_json(
                    "generation_page_reads.json",
                    {"run_id": run_id, "generation": generation},
                )
            ),
        },
    )
    print(
        json.dumps(
            {
                "run_id": handle.run_id,
                "sealed": True,
                "idle_distribution": summary["idle"]["distribution"],
                "generation_distribution": summary["generation"]["distribution"],
                "threshold_decision": summary["threshold_decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
