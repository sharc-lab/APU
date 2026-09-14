"""Empty / zero-duration measurement windows must hard-fail a cell (never ok=True)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from seam.tools.affinity_matrix import (
    CONFIGS,
    _generation_record,
    _run_is_ok,
    _TimedSample,
    assert_measurement_windows,
)


def test_assert_measurement_windows_rejects_empty_prefill() -> None:
    with pytest.raises(RuntimeError, match="prefill util window"):
        assert_measurement_windows(
            n_prefill=0, n_decode=20, ttft_s=1.0, decode_s=2.0, min_samples=1
        )


def test_assert_measurement_windows_rejects_zero_duration() -> None:
    with pytest.raises(RuntimeError, match="decode duration"):
        assert_measurement_windows(
            n_prefill=10, n_decode=10, ttft_s=1.0, decode_s=0.0, min_samples=1
        )


def test_generation_record_rejects_synthesized_empty_prefill_window() -> None:
    """Synthesize a run whose util samples all land after TTFT → empty prefill → hard fail."""
    ttft_ns = 500_000_000  # 0.5 s
    # All samples AFTER the split → n_prefill=0
    samples = [_TimedSample(ttft_ns + 100_000_000 * (i + 1), [80.0] * 8) for i in range(20)]
    result = SimpleNamespace(
        ttft_ns=ttft_ns,
        wall_ns=ttft_ns + 2_000_000_000,
        prompt_tokens=2048,
        completion_tokens=128,
        extra={"ttft_source": "streamer_first_token"},
    )
    baseline = [[10.0] * 8] * 3
    with pytest.raises(RuntimeError, match="prefill util window"):
        _generation_record(
            gen_index=1,
            baseline_samples=baseline,
            result=result,  # type: ignore[arg-type]
            util_samples=samples,
            sampler_overhead={},
            ttft_split_ns=ttft_ns,
            min_phase_samples=1,
        )


def test_run_is_ok_false_when_generations_missing_windows() -> None:
    run = {
        "ok": True,
        "generations": [
            {
                "ok": True,
                "n_prefill_util_samples": 0,
                "n_decode_util_samples": 20,
                "prefill_duration_s": 1.0,
                "decode_duration_s": 2.0,
                "prefill_delta_pct_per_cpu": [],
                "decode_delta_pct_per_cpu": [1.0] * 8,
            }
        ],
    }
    assert _run_is_ok(run) is False


def test_configs_are_a0a_a0b_plus_a1_a6() -> None:
    assert set(CONFIGS) == {"A0a", "A0b", "A1", "A2", "A3", "A4", "A5", "A6"}
    assert CONFIGS["A0a"].inference_num_threads == 8
    assert CONFIGS["A0b"].inference_num_threads == 4
    assert CONFIGS["A0a"].reference_exempt is True
    assert CONFIGS["A0b"].reference_exempt is True
    for cid in ("A1", "A2", "A3", "A4", "A5", "A6"):
        assert CONFIGS[cid].inference_num_threads == 4
        assert CONFIGS[cid].reference_exempt is False
