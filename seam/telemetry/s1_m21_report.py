"""Offline M2.1 report from a sealed S1 ``samples.ndjson`` (spec §7 M2.1 items a-g)."""

from __future__ import annotations

import argparse
import itertools
import json
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from seam.errors import SeamError
from seam.telemetry.s1_battery_char import (
    BatterySample,
    analyze_capacity_series,
    load_samples_ndjson,
)

__all__ = ["M21Report", "build_m21_report"]


@dataclass(frozen=True, slots=True)
class M21Report:
    """Full M2.1 deliverable - distributions with bootstrap CIs, not point estimates alone."""

    run_id: str
    n_samples: int
    duration_s: float
    # (a) update period
    update_period_median_s: float | None
    update_period_iqr_s: tuple[float, float] | None
    update_period_median_ci95_s: tuple[float, float] | None
    update_period_ecdf: list[list[float]]  # [[x, F(x)], ...]
    quantized: bool
    # (b) quantization step
    quantization_step_mwh_median: float | None
    quantization_step_mwh_iqr: tuple[float, float] | None
    quantization_step_median_ci95_mwh: tuple[float, float] | None
    capacity_delta_value_counts: dict[str, int]
    # (c)
    min_resolvable_energy_mwh: float | None
    # (d) load-bearing
    min_viable_energy_run_duration_s: float | None
    min_viable_energy_run_duration_ci95_s: tuple[float, float] | None
    # (e) settle
    settle_s: float | None
    settle_band_frac: float
    settle_window_s: float
    discharge_rate_stable_band_frac: float
    # (f) SoC dependence
    soc_dependence: dict[str, Any]
    proposed_soc_window_pct: list[float]
    # (g) DischargeRate vs ΔRemainingCapacity
    discharge_vs_delta: dict[str, Any]
    prefer_delta_remaining_capacity: bool


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        raise SeamError("percentile of empty series")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def _bootstrap_median_ci(
    values: list[float], *, n_boot: int = 2000, seed: int = 42
) -> tuple[float, float] | None:
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    meds: list[float] = []
    n = len(values)
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        meds.append(statistics.median(sample))
    meds.sort()
    return _percentile(meds, 0.025), _percentile(meds, 0.975)


def _ecdf(values: list[float]) -> list[list[float]]:
    if not values:
        return []
    ordered = sorted(values)
    n = len(ordered)
    out: list[list[float]] = []
    for i, x in enumerate(ordered):
        out.append([x, (i + 1) / n])
    return out


def _estimate_settle_s(
    samples: list[BatterySample],
    *,
    window_s: float = 60.0,
    band_frac: float = 0.05,
    consecutive_windows: int = 3,
) -> float | None:
    """Time from first sample until rolling discharge-rate means stay within ``band_frac``.

    Band: each window mean must lie within ``band_frac`` of the mean of those
    ``consecutive_windows`` window means (relative).
    """
    rates: list[tuple[float, float]] = []
    t0 = samples[0].t_ns
    for s in samples:
        if s.discharge_rate_mw is None:
            continue
        t = (s.t_ns - t0) / 1e9
        rates.append((t, float(s.discharge_rate_mw)))
    if len(rates) < 10:
        return None

    t_max = rates[-1][0]
    window_means: list[tuple[float, float]] = []  # (window_end_s, mean_mw)
    t = window_s
    while t <= t_max + 1e-9:
        lo = t - window_s
        vals = [r for ts, r in rates if lo <= ts < t]
        if vals:
            window_means.append((t, statistics.mean(vals)))
        t += window_s

    if len(window_means) < consecutive_windows:
        return None

    for i in range(consecutive_windows - 1, len(window_means)):
        chunk = window_means[i - consecutive_windows + 1 : i + 1]
        means = [m for _, m in chunk]
        center = statistics.mean(means)
        if center <= 0:
            continue
        if all(abs(m - center) / center <= band_frac for m in means):
            return chunk[-1][0]
    return None


def _soc_dependence(samples: list[BatterySample]) -> dict[str, Any]:
    """Compare DischargeRate in first vs second half of the run (high vs lower SoC)."""
    with_rate = [
        s
        for s in samples
        if s.discharge_rate_mw is not None and s.remaining_capacity_mwh is not None
    ]
    if len(with_rate) < 20:
        return {"sufficient_data": False}

    mid = len(with_rate) // 2
    high = with_rate[:mid]
    low = with_rate[mid:]

    def pack(group: list[BatterySample]) -> dict[str, Any]:
        rates = [float(s.discharge_rate_mw) for s in group if s.discharge_rate_mw is not None]
        caps = [
            float(s.remaining_capacity_mwh) for s in group if s.remaining_capacity_mwh is not None
        ]
        rates_sorted = sorted(rates)
        return {
            "n": len(rates),
            "discharge_mw_median": statistics.median(rates_sorted),
            "discharge_mw_iqr": (
                _percentile(rates_sorted, 0.25),
                _percentile(rates_sorted, 0.75),
            ),
            "discharge_mw_mean": statistics.mean(rates),
            "remaining_mwh_median": statistics.median(caps),
            "remaining_mwh_range": [min(caps), max(caps)],
        }

    high_p = pack(high)
    low_p = pack(low)
    rel = None
    if high_p["discharge_mw_median"] and high_p["discharge_mw_median"] > 0:
        rel = (low_p["discharge_mw_median"] - high_p["discharge_mw_median"]) / high_p[
            "discharge_mw_median"
        ]
    return {
        "sufficient_data": True,
        "high_soc_segment": high_p,
        "lower_soc_segment": low_p,
        "relative_median_discharge_change": rel,
        "independent_at_5pct": rel is not None and abs(rel) < 0.05,
    }


def _discharge_vs_delta(samples: list[BatterySample]) -> dict[str, Any]:
    """Compare integrated DischargeRate to Σ|ΔRemainingCapacity| over the run."""
    if len(samples) < 2:
        return {"sufficient_data": False}

    # Integrate rate: trapezoid in hours → mWh
    energy_from_rate = 0.0
    for a, b in itertools.pairwise(samples):
        if a.discharge_rate_mw is None or b.discharge_rate_mw is None:
            continue
        dt_h = (b.t_ns - a.t_ns) / 1e9 / 3600.0
        energy_from_rate += 0.5 * (a.discharge_rate_mw + b.discharge_rate_mw) * dt_h

    base = analyze_capacity_series(samples)
    energy_from_delta = sum(base.capacity_deltas_mwh)
    ratio = None
    if energy_from_delta > 0:
        ratio = energy_from_rate / energy_from_delta

    # EC-smoothed if rate energy diverges materially from stepped capacity, or if
    # DischargeRate changes more smoothly than capacity (lower relative step frequency).
    prefer_delta = True
    note = (
        "Prefer ΔRemainingCapacity: capacity steps are the EC's quantized ground truth; "
        "DischargeRate is typically EC-smoothed and can disagree with integrated capacity loss."
    )
    if ratio is not None and abs(ratio - 1.0) > 0.15:
        note += f" Observed rate/delta energy ratio={ratio:.3f} (>15% disagreement)."
    elif ratio is not None:
        note += f" Observed rate/delta energy ratio={ratio:.3f}."

    return {
        "sufficient_data": True,
        "energy_from_discharge_rate_mwh": energy_from_rate,
        "energy_from_capacity_deltas_mwh": energy_from_delta,
        "rate_over_delta_ratio": ratio,
        "prefer_delta_remaining_capacity": prefer_delta,
        "note": note,
    }


def build_m21_report(
    samples: list[BatterySample],
    *,
    run_id: str,
    settle_band_frac: float = 0.05,
    settle_window_s: float = 60.0,
    proposed_soc_window_pct: list[float] | None = None,
) -> M21Report:
    if len(samples) < 2:
        raise SeamError("need samples to build M2.1 report")

    base = analyze_capacity_series(samples)
    intervals = base.inter_change_intervals_s
    deltas = base.capacity_deltas_mwh

    period_ci = _bootstrap_median_ci(intervals) if intervals else None
    step_ci = _bootstrap_median_ci(deltas) if deltas else None

    # Quantized if a single delta value dominates (≥70% of changes).
    counts: dict[str, int] = {}
    for d in deltas:
        key = f"{d:g}"
        counts[key] = counts.get(key, 0) + 1
    quantized = False
    if deltas:
        top = max(counts.values())
        quantized = (top / len(deltas)) >= 0.70

    # Smallest observed per-update energy increment (power-dependent; not a device quantum).
    min_resolvable = min(deltas) if deltas else base.min_resolvable_energy_mwh

    # Numeric unchanged: 20 * period_median. Justification is edge-effect vs update period
    # (AUDIT_LOG M2.1 closeout Item 1), not ">=20 quanta".
    min_viable = base.min_viable_energy_run_duration_s
    min_viable_ci = None
    if period_ci is not None:
        min_viable_ci = (20.0 * period_ci[0], 20.0 * period_ci[1])

    settle = _estimate_settle_s(samples, window_s=settle_window_s, band_frac=settle_band_frac)
    soc = _soc_dependence(samples)
    dvd = _discharge_vs_delta(samples)

    # SoC window: if dependence is weak, permissive [40, 95]; else tighten around observed.
    if proposed_soc_window_pct is None:
        proposed_soc_window_pct = [40.0, 95.0] if soc.get("independent_at_5pct") else [40.0, 85.0]

    return M21Report(
        run_id=run_id,
        n_samples=base.n_samples,
        duration_s=base.duration_s,
        update_period_median_s=base.update_period_median_s,
        update_period_iqr_s=base.update_period_iqr_s,
        update_period_median_ci95_s=period_ci,
        update_period_ecdf=_ecdf(intervals),
        quantized=quantized,
        quantization_step_mwh_median=base.quantization_step_mwh_median,
        quantization_step_mwh_iqr=base.quantization_step_mwh_iqr,
        quantization_step_median_ci95_mwh=step_ci,
        capacity_delta_value_counts=counts,
        min_resolvable_energy_mwh=min_resolvable,
        min_viable_energy_run_duration_s=min_viable,
        min_viable_energy_run_duration_ci95_s=min_viable_ci,
        settle_s=settle,
        settle_band_frac=settle_band_frac,
        settle_window_s=settle_window_s,
        discharge_rate_stable_band_frac=settle_band_frac,
        soc_dependence=soc,
        proposed_soc_window_pct=proposed_soc_window_pct,
        discharge_vs_delta=dvd,
        prefer_delta_remaining_capacity=bool(dvd.get("prefer_delta_remaining_capacity", True)),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    run_dir = args.repo_root / "raw" / args.run_id
    samples_path = run_dir / "samples.ndjson"
    if not samples_path.is_file():
        raise SystemExit(f"missing {samples_path}")

    samples = load_samples_ndjson(samples_path)
    report = build_m21_report(samples, run_id=args.run_id)
    payload = asdict(report)
    # JSON-friendly tuples
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    out = args.out or (args.repo_root / "derived" / "m2_1" / f"{args.run_id}_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {out}")
    # Prominent load-bearing line
    print(
        f"MINIMUM_VIABLE_ENERGY_RUN_DURATION_S={report.min_viable_energy_run_duration_s} "
        f"CI95={report.min_viable_energy_run_duration_ci95_s}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
