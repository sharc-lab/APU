"""Pure classification logic for the A0-A6 confinement matrix.

Given per-core utilization deltas (phase mean minus idle baseline) and a measured noise band,
classify each core as LOADED, QUIET, or AMBIGUOUS, then derive a cell confinement verdict.

LOADED thresholds are per-core and two-pass: collect every cell first, then derive
``threshold(c) = 0.5 * delta(c)`` from the cell that deliberately targets ``c``
(A5 decode for P-cores, A6 decode for LP-E). The same per-core map is applied to both
phases. Do not classify during collection.
"""

from __future__ import annotations

import statistics
from typing import Any, Literal

__all__ = [
    "ConfinementVerdict",
    "CoreState",
    "classify_cell",
    "classify_core",
    "compute_noise_band",
    "loaded_threshold",
    "per_core_loaded_thresholds",
]

CoreState = Literal["LOADED", "QUIET", "AMBIGUOUS"]
ConfinementVerdict = Literal["CONFINED", "LEAKED", "UNCLEAR", "INVALID", "N/A"]


def loaded_threshold(deltas: dict[int, float], requested: set[int]) -> float:
    """Half the median delta over cores in the requested set ``S``.

    Legacy helper for single-threshold diagnostics. Production classification uses
    :func:`per_core_loaded_thresholds`.
    """
    in_set = [deltas[c] for c in requested if c in deltas]
    if not in_set:
        return 0.0
    return 0.5 * statistics.median(in_set)


def per_core_loaded_thresholds(
    a5_decode_deltas: dict[int, float],
    a6_decode_deltas: dict[int, float],
    *,
    p_cpus: set[int],
    lpe_cpus: set[int],
) -> dict[int, float]:
    """``threshold(c) = 0.5 * delta(c)`` from the cell that deliberately targets ``c``.

    - P-cores (typically 0-3): A5 decode deltas
    - LP-E (typically 4-7): A6 decode deltas

    Applied identically to prefill and decode. Excluded cores of a cell are judged against
    a *different* cell's reference (A5's LP-E vs A6; A6's P vs A5). Included cores judged
    against their own reference cell are a sanity check.
    """
    out: dict[int, float] = {}
    for cpu in sorted(p_cpus):
        out[int(cpu)] = 0.5 * float(a5_decode_deltas.get(int(cpu), 0.0))
    for cpu in sorted(lpe_cpus):
        out[int(cpu)] = 0.5 * float(a6_decode_deltas.get(int(cpu), 0.0))
    return out


def classify_core(delta: float, *, threshold: float, noise_band: float) -> CoreState:
    """Classify one core's delta against the LOADED / QUIET / AMBIGUOUS rules."""
    if delta > threshold:
        return "LOADED"
    if delta <= noise_band:
        return "QUIET"
    return "AMBIGUOUS"


def classify_cell(
    deltas: dict[int, float],
    requested: set[int],
    *,
    noise_band: float,
    all_cpus: set[int] | None = None,
    loaded_thresholds: dict[int, float] | None = None,
    loaded_threshold_value: float | None = None,
) -> dict[str, Any]:
    """Classify every core and derive the cell confinement verdict.

    For unconfined references (A0a/A0b), pass ``requested`` equal to all logical CPUs;
    the verdict is ``N/A``.

    Prefer ``loaded_thresholds`` (per-core map from A5/A6). When only
    ``loaded_threshold_value`` is provided it is used for every core (legacy/tests).
    Otherwise the threshold falls back to ``0.5 * median(deltas in requested)``.
    """
    cpus = all_cpus if all_cpus is not None else set(deltas)

    if loaded_thresholds is not None:
        thr_map = {int(k): float(v) for k, v in loaded_thresholds.items()}
        fallback = (
            float(loaded_threshold_value)
            if loaded_threshold_value is not None
            else loaded_threshold(deltas, requested)
        )
        states: dict[int, CoreState] = {
            cpu: classify_core(
                deltas.get(cpu, 0.0),
                threshold=thr_map.get(cpu, fallback),
                noise_band=noise_band,
            )
            for cpu in cpus
        }
        # Report the median of applied per-core thresholds for summary tables.
        applied = [thr_map.get(cpu, fallback) for cpu in sorted(cpus)]
        threshold_report: float | dict[str, float] = {
            str(cpu): round(thr_map.get(cpu, fallback), 4) for cpu in sorted(cpus)
        }
        threshold_scalar = statistics.fmean(applied) if applied else 0.0
    else:
        threshold_scalar = (
            float(loaded_threshold_value)
            if loaded_threshold_value is not None
            else loaded_threshold(deltas, requested)
        )
        states = {
            cpu: classify_core(
                deltas.get(cpu, 0.0), threshold=threshold_scalar, noise_band=noise_band
            )
            for cpu in cpus
        }
        threshold_report = threshold_scalar

    outside = cpus - requested
    inside = requested & cpus

    invalid = any(states[c] != "LOADED" for c in inside)
    leaked = any(states[c] == "LOADED" for c in outside)
    unclear = any(states[c] == "AMBIGUOUS" for c in outside)

    if len(requested) == len(cpus):
        verdict: ConfinementVerdict = "N/A"
    elif invalid:
        verdict = "INVALID"
    elif leaked:
        verdict = "LEAKED"
    elif unclear:
        verdict = "UNCLEAR"
    else:
        verdict = "CONFINED"

    return {
        "requested_cpus": sorted(requested),
        "loaded_threshold": threshold_report,
        "loaded_threshold_mean": round(threshold_scalar, 4)
        if loaded_thresholds is not None
        else threshold_scalar,
        "noise_band": noise_band,
        "core_states": {str(k): v for k, v in sorted(states.items())},
        "verdict": verdict,
    }


def compute_noise_band(baseline_means_per_core: dict[int, list[float]]) -> float:
    """``NOISE_BAND = 2 * mean(CV across cores)`` on pooled idle-baseline means.

    Each core's CV is computed across its pooled baseline mean values (one per scored generation).
    Cores with fewer than two samples contribute CV=0.
    """
    from seam.analysis.slice_stats import coefficient_of_variation

    if not baseline_means_per_core:
        return 0.0
    cvs: list[float] = []
    for values in baseline_means_per_core.values():
        if len(values) < 2:
            cvs.append(0.0)
            continue
        cv = coefficient_of_variation(values)
        cvs.append(0.0 if cv != cv else cv)  # nan -> 0
    return 2.0 * statistics.fmean(cvs)
