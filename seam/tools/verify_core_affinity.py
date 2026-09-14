"""Confirm that OpenVINO's core selection actually took effect.

WHY THIS EXISTS
---------------
Panther Lake's low-power die has 4 P-cores and 4 **LP-E** cores and **no standard E-cores**. When
OpenVINO is asked for ``ECORE_ONLY`` it may map that onto LP-E, or it may match nothing and
quietly run on every core. Setting the property is a *request*, not a guarantee.

If affinity leaks, both arms of M-SLICE run on the same silicon. The throughput ratio collapses to
1, escalation rates coincide, and the experiment returns a null - a null that is indistinguishable
from a real one by inspection of the results alone. That is the single most expensive failure
available here, so it is checked empirically rather than trusted.

The check samples **per-CPU utilization** during live inference and compares the loaded set against
the M1-committed mapping. A run that disagrees is reported as a finding, and the caller falls back
to process affinity via :func:`seam.topology.affinity_for`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from seam.config import load_platform_config
from seam.gitinfo import repo_root
from seam.jsonlog import log_event

__all__ = ["AffinityEvidence", "sample_per_cpu", "verify_target"]

#: A logical CPU is "loaded" when inference raises its utilization by more than this many points
#: above the idle baseline. Generous on purpose: the question is whether a cluster is being used at
#: all, not how efficiently.
#:
#: The threshold is applied to the BASELINE-SUBTRACTED value. Judging absolute utilization would
#: charge unrelated background work to the pipeline under test and manufacture a leak verdict -
#: which is exactly what happened when this check was first run beside a large file download.
_LOADED_PCT = 25.0

#: Seconds of idle sampling taken immediately before inference. Short enough not to dominate the
#: run, long enough to average over Windows' scheduler quantum.
_BASELINE_S = 3.0


@dataclass(slots=True)
class AffinityEvidence:
    """Per-core utilization measured while the model was generating."""

    target: str
    requested_scheduling_core_type: str
    expected_cpus: list[int]
    observed_loaded_cpus: list[int]
    mean_pct_per_cpu: list[float]
    n_samples: int
    matched: bool
    leaked_cpus: list[int] = field(default_factory=list)
    missing_cpus: list[int] = field(default_factory=list)
    verdict: str = "refused"
    note: str = ""
    #: Per-CPU utilization measured with the machine idle, immediately before inference.
    baseline_pct_per_cpu: list[float] = field(default_factory=list)
    #: ``mean_pct_per_cpu`` minus ``baseline_pct_per_cpu``. The discriminant.
    delta_pct_per_cpu: list[float] = field(default_factory=list)
    n_baseline_samples: int = 0
    #: Process affinity mask read back from the OS. Proves the mask was set; does not prove where
    #: the threads actually ran, which is what the utilization delta is for.
    process_affinity_mask: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sample_per_cpu(stop: threading.Event, interval_s: float = 0.25) -> list[list[float]]:
    """Sample per-CPU utilization until ``stop`` is set."""
    import psutil

    series: list[list[float]] = []
    psutil.cpu_percent(percpu=True)  # prime the counter; the first read is meaningless
    while not stop.is_set():
        time.sleep(interval_s)
        series.append(list(psutil.cpu_percent(percpu=True)))
    return series


def verify_target(
    *,
    target: str,
    scheduling_core_type: str,
    expected_cpus: list[int],
    generate: Any,
    interval_s: float = 0.25,
) -> AffinityEvidence:
    """Run ``generate()`` while sampling per-CPU load, and compare against ``expected_cpus``.

    ``generate`` is a zero-argument callable that performs a representative inference. It is
    supplied rather than constructed here so the verification exercises the *same* pipeline object
    the measurement will use - verifying a differently-configured pipeline would prove nothing.

    An idle baseline is sampled first and subtracted, so background activity on the machine is not
    charged to the pipeline under test.
    """
    import psutil

    baseline_stop = threading.Event()
    baseline: list[list[float]] = []
    baseline_thread = threading.Thread(
        target=lambda: baseline.extend(sample_per_cpu(baseline_stop, interval_s)),
        name="cpu-baseline",
        daemon=True,
    )
    baseline_thread.start()
    time.sleep(_BASELINE_S)
    baseline_stop.set()
    baseline_thread.join(timeout=5.0)

    stop = threading.Event()
    series: list[list[float]] = []

    def _sampler() -> None:
        series.extend(sample_per_cpu(stop, interval_s))

    thread = threading.Thread(target=_sampler, name="cpu-sampler", daemon=True)
    thread.start()
    try:
        generate()
    finally:
        stop.set()
        thread.join(timeout=5.0)

    try:
        mask = sorted(psutil.Process().cpu_affinity())
    except Exception:  # affinity readback is diagnostic, not load-bearing
        mask = []

    if not series:
        return AffinityEvidence(
            target=target,
            requested_scheduling_core_type=scheduling_core_type,
            expected_cpus=expected_cpus,
            observed_loaded_cpus=[],
            mean_pct_per_cpu=[],
            n_samples=0,
            matched=False,
            verdict="refused",
            note="no utilization samples captured; inference finished faster than one sample "
            "interval. Re-run with a longer generation.",
            process_affinity_mask=mask,
        )

    n_cpus = len(series[0])
    means = [statistics.fmean(sample[cpu] for sample in series) for cpu in range(n_cpus)]
    if baseline:
        base = [statistics.fmean(sample[cpu] for sample in baseline) for cpu in range(n_cpus)]
    else:
        base = [0.0] * n_cpus
    deltas = [means[cpu] - base[cpu] for cpu in range(n_cpus)]
    loaded = [cpu for cpu, pct in enumerate(deltas) if pct >= _LOADED_PCT]

    expected = set(expected_cpus)
    leaked = sorted(set(loaded) - expected)
    missing = sorted(expected - set(loaded))
    matched = not leaked and not missing

    evidence = AffinityEvidence(
        target=target,
        requested_scheduling_core_type=scheduling_core_type,
        expected_cpus=sorted(expected),
        observed_loaded_cpus=loaded,
        mean_pct_per_cpu=[round(m, 2) for m in means],
        n_samples=len(series),
        matched=matched,
        leaked_cpus=leaked,
        missing_cpus=missing,
        verdict="pass" if matched else "refused",
        note=(
            ""
            if matched
            else (
                f"{scheduling_core_type} did not confine work to the M1-committed "
                f"{target} cpus. Leaked onto {leaked}; expected-but-idle {missing}. This is a "
                f"FINDING: fall back to process affinity via topology.affinity_for() and record "
                f"the disagreement."
            )
        ),
        baseline_pct_per_cpu=[round(b, 2) for b in base],
        delta_pct_per_cpu=[round(d, 2) for d in deltas],
        n_baseline_samples=len(baseline),
        process_affinity_mask=mask,
    )

    log_event(
        "affinity.verdict",
        severity="info" if matched else "warning",
        message=(
            f"{target}: requested {scheduling_core_type}, loaded cpus {loaded}, "
            f"expected {sorted(expected)} -> {evidence.verdict}"
        ),
        **evidence.to_dict(),
    )
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", default="aipc-c1")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    config = load_platform_config(args.platform, repo_root=repo_root(Path(__file__).parent))
    print("M1-committed mapping:")
    print(f"  p_cpus   : {config.get('topology.p_cpus')}")
    print(f"  lpe_cpus : {config.get('topology.lpe_cpus')}")
    print(f"  verified : {config.get('topology.verified')}")
    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "p_cpus": config.get("topology.p_cpus"),
                    "lpe_cpus": config.get("topology.lpe_cpus"),
                    "verified": config.get("topology.verified"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
