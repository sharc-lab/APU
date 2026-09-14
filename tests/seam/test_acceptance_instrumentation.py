"""Unit tests for acceptance additive instruments (no OpenVINO, no sealing)."""

from __future__ import annotations

import pytest

from seam.tools.acceptance_instrumentation import (
    classify_arm_confinement,
    memory_dispersion,
)
from seam.tools.fixed_throughput import derive_verdict


def test_memory_dispersion_ratio() -> None:
    a = {
        "peak_working_set_bytes": 1000,
        "peak_commit_bytes": 2000,
        "free_physical_bytes_at_start": 8000,
        "free_physical_bytes_at_peak": 4000,
    }
    b = {
        "peak_working_set_bytes": 1100,
        "peak_commit_bytes": 2000,
        "free_physical_bytes_at_start": 8000,
        "free_physical_bytes_at_peak": 3900,
    }
    disp = memory_dispersion(a, b)
    assert disp["peak_working_set_bytes"]["ratio_max_over_min"] == pytest.approx(1.1)
    assert disp["peak_commit_bytes"]["ratio_max_over_min"] == 1.0


def test_classify_arm_confinement_records_verdict() -> None:
    # Loaded on P-cores, quiet on LP-E → a real verdict under the single-threshold path.
    baseline = [[1.0] * 8 for _ in range(5)]
    util = [[40.0, 40.0, 40.0, 40.0, 1.0, 1.0, 1.0, 1.0] for _ in range(10)]
    result = classify_arm_confinement(
        p_cpus=[0, 1, 2, 3],
        lpe_cpus=[4, 5, 6, 7],
        util_series=util,
        baseline_series=baseline,
    )
    assert result["verdict"] in {"CONFINED", "UNCLEAR", "LEAKED", "INVALID", "N/A"}
    assert "core_states" in result


def test_derived_band_not_asserted_1_10() -> None:
    criterion = {
        "basis": "r_decode_tok_s",
        "spread_estimator": "cv",
        "band_multiple": 2.0,
        "max_within_run_spread": 0.10,
        "prior_unexplained_ratio": 1.98,
    }
    summary_a = {
        "r_decode_tok_s": {"median": 10.0, "cv": 0.02},
        "r_prefill_tok_s": {"median": 100.0, "cv": 0.02},
    }
    summary_b = {
        "r_decode_tok_s": {"median": 10.5, "cv": 0.03},
        "r_prefill_tok_s": {"median": 105.0, "cv": 0.03},
    }
    gate = derive_verdict(summary_a=summary_a, summary_b=summary_b, criterion=criterion)
    assert gate["gate_mode"] == "derived_from_within_run_spread"
    assert gate["band"] == pytest.approx(0.06)
    assert gate["verdict"] == "PASS"
