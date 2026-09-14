"""C2f - seal an idle compute-canary noise-floor baseline and derive the drift threshold.

Machine lock held, quiesced, **no inference workload**. Run the compute canary ≥30 times
back-to-back (configurable), form consecutive-pair relative drifts, report median / p95 / max /
fraction above the asserted 0.15 placeholder, then set:

    threshold = idle_p95 + margin_above_idle_p95

Exactly as ``paging_gate`` records ``baseline_run_id`` / idle p95 / margin / justification.

Do **not** choose the threshold to make a pilot pass. Derive it, seal it, apply it.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import uuid
from pathlib import Path
from typing import Any

import yaml

from seam.config import resolve_config
from seam.errors import SeamError
from seam.gitinfo import repo_root
from seam.locks import exclusive
from seam.manifest import emit
from seam.measurement import measure_quiescence, run_compute_canary
from seam.powerstate import capture_power_state, manifest_power_state

__all__ = [
    "DEFAULT_MARGIN",
    "DEFAULT_N_SAMPLES",
    "canary_pair_drift_stats",
    "derive_canary_gate",
    "main",
    "run_idle_canary_series",
]


def _median_runtime_ns(canary_result: dict[str, Any]) -> float:
    return float(statistics.median(float(w["runtime_ns"]) for w in canary_result["workers"]))


def _relative_drift(pre_runtime: float, post_runtime: float) -> float:
    return abs(post_runtime - pre_runtime) / pre_runtime if pre_runtime else float("inf")


# Stated margin above idle canary-drift p95 (same spirit as paging's margin_above_idle_p95=1.0
# page reads/sec). Units here are relative drift (dimensionless). Declared, not tuned to pass.
DEFAULT_MARGIN: float = 0.05
DEFAULT_N_SAMPLES: int = 31  # ≥30 consecutive pairs => 31 canary invocations
ASSERTED_PLACEHOLDER: float = 0.15


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


def canary_pair_drift_stats(
    drifts: list[float], *, placeholder_threshold: float = ASSERTED_PLACEHOLDER
) -> dict[str, Any]:
    """Median / p95 / max / fraction above placeholder for consecutive-pair drifts."""
    if not drifts:
        raise SeamError("canary baseline produced no consecutive-pair drifts")
    ordered = sorted(float(d) for d in drifts)
    n = len(ordered)
    p95_index = min(n - 1, max(0, round(0.95 * (n - 1))))
    above = sum(1 for d in ordered if d > float(placeholder_threshold))
    return {
        "n_pairs": n,
        "median": float(statistics.median(ordered)),
        "p95": float(ordered[p95_index]),
        "max": float(ordered[-1]),
        "min": float(ordered[0]),
        "mean": float(statistics.fmean(ordered)),
        "fraction_above_placeholder": above / n,
        "n_above_placeholder": above,
        "placeholder_threshold": float(placeholder_threshold),
    }


def derive_canary_gate(
    distribution: dict[str, Any],
    *,
    margin: float,
    baseline_run_id: str,
    spacing_s: float,
    n_samples: int,
) -> dict[str, Any]:
    """threshold = idle_p95 + margin. Never choose margin/threshold to pass a pilot."""
    if margin < 0:
        raise SeamError(f"canary baseline margin must be >= 0; got {margin}")
    idle_p95 = float(distribution["p95"])
    threshold = idle_p95 + float(margin)
    return {
        "baseline_run_id": baseline_run_id,
        "idle_p95": idle_p95,
        "margin": float(margin),
        "margin_above_idle_p95": float(margin),
        "threshold": threshold,
        "max_relative_drift": threshold,
        "gate_mode": "baseline_relative",
        "spacing_s": float(spacing_s),
        "n_samples": int(n_samples),
        "n_pairs": int(distribution["n_pairs"]),
        "distribution": {
            "median": distribution["median"],
            "p95": distribution["p95"],
            "max": distribution["max"],
            "fraction_above_placeholder": distribution["fraction_above_placeholder"],
            "placeholder_threshold": distribution["placeholder_threshold"],
        },
        "justification": (
            f"Idle compute-canary consecutive-pair relative drift "
            f"(n_pairs={distribution['n_pairs']}, spacing_s={spacing_s}: "
            f"median={distribution['median']}, p95={idle_p95}, max={distribution['max']}, "
            f"fraction_above_{distribution['placeholder_threshold']}="
            f"{distribution['fraction_above_placeholder']}). "
            f"Gate is baseline-relative: threshold = idle_p95 + margin = "
            f"{idle_p95} + {margin} = {threshold}. Threshold derived from sealed baseline "
            f"run_id={baseline_run_id}; not chosen to make a pilot pass (C2f)."
        ),
        "threshold_justification": (
            f"Idle compute-canary consecutive-pair relative drift "
            f"(n_pairs={distribution['n_pairs']}, spacing_s={spacing_s}: "
            f"median={distribution['median']}, p95={idle_p95}, max={distribution['max']}, "
            f"fraction_above_{distribution['placeholder_threshold']}="
            f"{distribution['fraction_above_placeholder']}). "
            f"Gate is baseline-relative: threshold = idle_p95 + margin = "
            f"{idle_p95} + {margin} = {threshold}. Threshold derived from sealed baseline "
            f"run_id={baseline_run_id}; not chosen to make a pilot pass (C2f)."
        ),
    }


def run_idle_canary_series(
    *,
    cpus: list[int],
    iterations_per_cpu: int,
    seed: int,
    n_samples: int,
    spacing_s: float,
) -> dict[str, Any]:
    """Run ``n_samples`` canaries under the caller's lock; return pair drifts + raw samples."""
    if n_samples < 2:
        raise SeamError(f"need n_samples >= 2 for consecutive pairs; got {n_samples}")
    if spacing_s < 0:
        raise SeamError(f"spacing_s must be >= 0; got {spacing_s}")
    samples: list[dict[str, Any]] = []
    started = time.monotonic()
    for index in range(int(n_samples)):
        if index > 0 and spacing_s > 0:
            time.sleep(float(spacing_s))
        result = run_compute_canary(
            cpus=cpus,
            iterations_per_cpu=iterations_per_cpu,
            seed=seed,
        )
        if not result["affinity_passed"]:
            raise SeamError(f"canary sample {index}: affinity readback failed")
        median_ns = _median_runtime_ns(result)
        samples.append(
            {
                "index": index,
                "started_utc": result["started_utc"],
                "finished_utc": result["finished_utc"],
                "wall_ns": result["wall_ns"],
                "median_runtime_ns": median_ns,
                "combined_digest": result["combined_digest"],
            }
        )
    drifts: list[float] = []
    for i in range(1, len(samples)):
        drifts.append(
            _relative_drift(
                float(samples[i - 1]["median_runtime_ns"]),
                float(samples[i]["median_runtime_ns"]),
            )
        )
    distribution = canary_pair_drift_stats(drifts)
    return {
        "n_samples": len(samples),
        "spacing_s": float(spacing_s),
        "spacing_note": (
            "Inter-canary sleep matching E-FILTER settle_s when settle_s>0 (double-after "
            "spacing); 0 means true back-to-back. No inference workload between samples."
        ),
        "elapsed_s": time.monotonic() - started,
        "samples": samples,
        "pair_drifts": drifts,
        "distribution": distribution,
    }


def _write_derived_gate(root: Path, gate: dict[str, Any], *, run_id: str) -> Path:
    out = root / "derived" / "efilter" / "canary_gate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "canary_gate": gate,
        "run_id": run_id,
        "written_utc": _utc_now(),
        "consumers": [
            "configs/measurement.yaml canary.* (apply after seal)",
            "seam.measurement._apply_post_canaries via max_relative_drift",
            "seam.tools.efilter_run summary.canary_gate",
        ],
        "status": "READY_FOR_MEASUREMENT_YAML_UPDATE",
        "note": (
            "Copy threshold fields into configs/measurement.yaml canary block before the "
            "next E-FILTER pilot. Do not edit sealed raw/."
        ),
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out


def _patch_measurement_yaml(root: Path, gate: dict[str, Any]) -> None:
    """Update canary threshold fields in measurement.yaml from the sealed baseline."""
    path = root / "configs" / "measurement.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    canary = dict(data.get("canary") or {})
    canary["max_relative_drift"] = float(gate["threshold"])
    canary["gate_mode"] = "baseline_relative"
    canary["idle_p95"] = float(gate["idle_p95"])
    canary["margin_above_idle_p95"] = float(gate["margin"])
    canary["baseline_run_id"] = gate["baseline_run_id"]
    canary["threshold_justification"] = gate["justification"]
    data["canary"] = canary
    path.write_text(
        yaml.safe_dump(data, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument(
        "--spacing-s",
        type=float,
        default=None,
        help="Sleep between canary invocations (default: efilter canary.settle_s, else 0).",
    )
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--update-measurement-yaml",
        action="store_true",
        help="Rewrite configs/measurement.yaml canary threshold fields from this baseline.",
    )
    parser.add_argument(
        "--skip-quiescence",
        action="store_true",
        help="Emergency only; default requires measured-load quiescence under the lock.",
    )
    args = parser.parse_args(argv)
    root = repo_root(Path(__file__).parent)
    run_id = str(uuid.UUID(args.run_id)) if args.run_id else str(uuid.uuid4())

    measurement_path = root / "configs" / "measurement.yaml"
    efilter_path = root / "configs" / "efilter.yaml"
    platform = root / "configs" / "platforms" / "aipc-c1.yaml"
    measurement = yaml.safe_load(measurement_path.read_text(encoding="utf-8"))
    efilter = yaml.safe_load(efilter_path.read_text(encoding="utf-8"))
    canary_cfg = {
        **dict(measurement.get("canary") or {}),
        **dict(efilter.get("canary") or {}),
    }
    spacing = (
        float(args.spacing_s)
        if args.spacing_s is not None
        else float(canary_cfg.get("settle_s") or 0.0)
    )
    cpus = [int(c) for c in canary_cfg["cpus"]]
    iterations = int(canary_cfg["iterations_per_cpu"])
    seed = int(canary_cfg["seed"])
    n_samples = int(args.n_samples)
    if n_samples < 31:
        raise SeamError(
            f"C2f requires ≥30 consecutive pairs (≥31 samples); got n_samples={n_samples}"
        )

    resolved = resolve_config([platform, measurement_path], repo_root=root)
    p_cpus = [int(c) for c in resolved.require("topology.p_cpus")]
    lock_resource = root / ".locks" / "machine"

    with exclusive(lock_resource):
        quiesce_record: dict[str, Any]
        if args.skip_quiescence:
            quiesce_record = {
                "passed": True,
                "skipped": True,
                "reason": "--skip-quiescence",
            }
        else:
            q = measurement["quiescence"]
            quiesce_record = measure_quiescence(
                window_s=float(q["window_s"]),
                sample_interval_s=float(q["sample_interval_s"]),
                p_cpus=p_cpus,
                total_cpu_max_pct=float(q["total_cpu_max_pct"]),
                p_core_cpu_max_pct=float(q["p_core_cpu_max_pct"]),
                available_memory_min_mb=float(q["available_memory_min_mb"]),
            )
            if not quiesce_record["passed"]:
                raise SeamError(
                    "measured-load quiescence refusal: "
                    + "; ".join(quiesce_record.get("failures") or [])
                )
        series = run_idle_canary_series(
            cpus=cpus,
            iterations_per_cpu=iterations,
            seed=seed,
            n_samples=n_samples,
            spacing_s=spacing,
        )
        series["machine_lock"] = {"resource": str(lock_resource), "held": True}
        series["quiescence"] = quiesce_record
        series["inference_workload"] = False

    gate = derive_canary_gate(
        series["distribution"],
        margin=float(args.margin),
        baseline_run_id=run_id,
        spacing_s=spacing,
        n_samples=n_samples,
    )
    summary: dict[str, Any] = {
        "experiment_id": "efilter-c2f-canary-baseline",
        "classification": "EFILTER_CANARY_NOISE_BASELINE",
        "run_id": run_id,
        "canary_work": {
            "algorithm": canary_cfg.get("algorithm"),
            "cpus": cpus,
            "iterations_per_cpu": iterations,
            "seed": seed,
        },
        "series": {
            **{k: v for k, v in series.items() if k != "samples"},
            "sample_count": len(series["samples"]),
        },
        "distribution": {"run_id": run_id, **series["distribution"]},
        "canary_gate": gate,
        "asserted_placeholder_was": ASSERTED_PLACEHOLDER,
        "wallclock_timeout_audit": "derived/efilter/c2f_wallclock_timeout_audit.json",
        "audit_verdict": "NONE",
    }
    power = capture_power_state()
    handle = emit(
        config=resolved,
        target="cpu-p",
        workload={
            "kind": "efilter_canary_baseline",
            "benchmark": "compute_canary_idle_pair_drift",
            "task_ids": [],
            "seed": seed,
            "n_repeats": 1,
            "concurrency": 1,
        },
        condition_label="efilter_canary_baseline",
        repo_root=root,
        run_id=run_id,
        allow_dirty=bool(args.allow_dirty),
        summary=summary,
        power_state=manifest_power_state(power, background_quiesced=True),
        thermal={"regime": "confound", "excluded": False},
        self_check="pass",
        before_integrity_hash=lambda run_dir: {
            "canary_samples_path": str(
                run_dir.write_json(
                    "canary_idle_samples.json",
                    {"run_id": run_id, "series": series},
                )
            ),
        },
    )
    derived_path = _write_derived_gate(root, gate, run_id=handle.run_id)
    if args.update_measurement_yaml:
        _patch_measurement_yaml(root, gate)
    print(
        json.dumps(
            {
                "run_id": handle.run_id,
                "sealed": True,
                "distribution": summary["distribution"],
                "canary_gate": gate,
                "derived_canary_gate_path": str(derived_path),
                "measurement_yaml_updated": bool(args.update_measurement_yaml),
                "status": "BASELINE_SEALED",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
