from __future__ import annotations

import random

import pytest

from seam.analysis.attrib_fit import RankDeficientDesignError, fit_interaction


def _synthetic_records() -> tuple[list[dict[str, float | int | str | bool]], dict[str, float]]:
    truth = {
        "a": 0.075,
        "R_prefill": 520.0,
        "d0": 0.0065,
        "d1": 1.8e-6,
    }
    rng = random.Random(20260803)
    records: list[dict[str, float | int | str | bool]] = []
    sequence = 0
    for repeat in range(10):
        for prompt in (1024, 4096, 16384, 32768):
            for n_out in (8, 64, 256):
                wall = (
                    truth["a"]
                    + prompt / truth["R_prefill"]
                    + n_out * truth["d0"]
                    + n_out * prompt * truth["d1"]
                    + rng.gauss(0.0, 0.002)
                )
                records.append(
                    {
                        "total_prompt_tokens": prompt,
                        "n_out": n_out,
                        "wall_s": wall,
                        "repeat_idx": repeat,
                        "block_idx": repeat,
                        "sequence_idx": sequence,
                        "execution_target": "cpu-p",
                        "quantization": "int4",
                        "held_out": False,
                    }
                )
                sequence += 1
    return records, truth


def test_interaction_fitter_recovers_known_coefficients_within_ci() -> None:
    records, truth = _synthetic_records()
    fit = fit_interaction(
        records,
        bootstrap_resamples=2_000,
        bootstrap_seed=20260803,
    )

    assert fit.rank == 4
    assert fit.condition_number < 30
    assert fit.vif["centered_interaction"] < 5
    for name, expected in truth.items():
        interval = fit.parameters[name]
        assert interval.lo <= expected <= interval.hi, (name, interval, expected)


def test_interaction_fitter_fails_loudly_on_rank_deficient_design() -> None:
    records = [
        {
            "total_prompt_tokens": prompt,
            "n_out": prompt // 16,
            "wall_s": 0.1 + prompt / 500.0,
            "execution_target": "cpu-p",
            "quantization": "int4",
            "held_out": False,
        }
        for prompt in (1024, 2048, 4096, 8192)
        for _ in range(3)
    ]

    with pytest.raises(RankDeficientDesignError, match="Refusing a pseudo-inverse"):
        fit_interaction(records, bootstrap_resamples=200)


def test_centering_keeps_balanced_interaction_vif_below_gate() -> None:
    records, _ = _synthetic_records()
    fit = fit_interaction(records, bootstrap_resamples=500)

    assert fit.vif["total_prompt_centered"] == pytest.approx(1.0, abs=1e-12)
    assert fit.vif["n_out_centered"] == pytest.approx(1.0, abs=1e-12)
    assert fit.vif["centered_interaction"] == pytest.approx(1.0, abs=1e-12)
