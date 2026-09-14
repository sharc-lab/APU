"""Baseline subtraction in the affinity verifier.

Regression cover for a real false verdict: the check was first run beside a multi-gigabyte
download, whose load appeared on every CPU and was charged to the pipeline under test, producing
a "leaked" verdict for a target that had not leaked.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from seam.tools import verify_core_affinity as vca


@pytest.fixture
def fake_sampler(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Feed scripted per-CPU readings: first call is the baseline, second the live window."""
    state: dict[str, Any] = {"calls": 0, "baseline": [], "during": []}

    def _sample(stop: threading.Event, interval_s: float = 0.25) -> list[list[float]]:
        state["calls"] += 1
        series: list[list[float]] = state["baseline"] if state["calls"] == 1 else state["during"]
        return series

    monkeypatch.setattr(vca, "sample_per_cpu", _sample)
    monkeypatch.setattr(vca.time, "sleep", lambda _s: None)
    return state


def test_busy_background_does_not_manufacture_a_leak(fake_sampler: dict[str, Any]) -> None:
    """Cores loaded equally before and during inference are background, not leakage."""
    # CPUs 4-7 are pinned at 80% by an unrelated process, both before and during.
    fake_sampler["baseline"] = [[2.0, 2.0, 2.0, 2.0, 80.0, 80.0, 80.0, 80.0]] * 8
    fake_sampler["during"] = [[95.0, 95.0, 95.0, 95.0, 82.0, 82.0, 82.0, 82.0]] * 8

    evidence = vca.verify_target(
        target="cpu-p",
        scheduling_core_type="PCORE_ONLY",
        expected_cpus=[0, 1, 2, 3],
        generate=lambda: None,
    )

    assert evidence.observed_loaded_cpus == [0, 1, 2, 3]
    assert evidence.leaked_cpus == []
    assert evidence.matched is True
    assert evidence.verdict == "pass"


def test_genuine_leak_is_still_caught(fake_sampler: dict[str, Any]) -> None:
    """A cluster that goes from idle to busy during inference is real leakage."""
    fake_sampler["baseline"] = [[2.0] * 8] * 8
    fake_sampler["during"] = [[95.0, 95.0, 95.0, 95.0, 90.0, 90.0, 90.0, 90.0]] * 8

    evidence = vca.verify_target(
        target="cpu-p",
        scheduling_core_type="PCORE_ONLY",
        expected_cpus=[0, 1, 2, 3],
        generate=lambda: None,
    )

    assert evidence.leaked_cpus == [4, 5, 6, 7]
    assert evidence.matched is False
    assert evidence.verdict == "refused"


def test_expected_cluster_idle_is_reported_as_missing(fake_sampler: dict[str, Any]) -> None:
    fake_sampler["baseline"] = [[2.0] * 8] * 8
    fake_sampler["during"] = [[3.0, 3.0, 3.0, 3.0, 90.0, 90.0, 90.0, 90.0]] * 8

    evidence = vca.verify_target(
        target="cpu-p",
        scheduling_core_type="PCORE_ONLY",
        expected_cpus=[0, 1, 2, 3],
        generate=lambda: None,
    )

    assert evidence.missing_cpus == [0, 1, 2, 3]
    assert evidence.leaked_cpus == [4, 5, 6, 7]
    assert evidence.matched is False


def test_baseline_and_delta_are_recorded(fake_sampler: dict[str, Any]) -> None:
    """The evidence must show its work, not just the verdict."""
    fake_sampler["baseline"] = [[10.0] * 8] * 4
    fake_sampler["during"] = [[60.0, 60.0, 60.0, 60.0, 11.0, 11.0, 11.0, 11.0]] * 4

    evidence = vca.verify_target(
        target="cpu-p",
        scheduling_core_type="PCORE_ONLY",
        expected_cpus=[0, 1, 2, 3],
        generate=lambda: None,
    )

    assert evidence.baseline_pct_per_cpu == [10.0] * 8
    assert evidence.delta_pct_per_cpu[:4] == [50.0] * 4
    assert evidence.delta_pct_per_cpu[4:] == [1.0] * 4
    assert evidence.n_baseline_samples == 4
