"""TTFT resolution must never invent zero when GenAI omits perf_metrics."""

from __future__ import annotations

import pytest

from seam.backends.local_openvino import resolve_ttft_ns
from seam.tools.affinity_matrix import _completed_cells, _phase_means


def test_resolve_prefers_perf_metrics() -> None:
    ttft, source = resolve_ttft_ns(12_000_000, 99_000_000)
    assert ttft == 12_000_000
    assert source == "perf_metrics"


def test_resolve_falls_back_to_streamer() -> None:
    ttft, source = resolve_ttft_ns(None, 45_000_000)
    assert ttft == 45_000_000
    assert source == "streamer_first_token"


def test_resolve_rejects_zero_metrics_and_uses_streamer() -> None:
    ttft, source = resolve_ttft_ns(0, 45_000_000)
    assert ttft == 45_000_000
    assert source == "streamer_first_token"


def test_resolve_unavailable_when_both_missing() -> None:
    ttft, source = resolve_ttft_ns(None, None)
    assert ttft is None
    assert source == "unavailable"


def test_phase_means_empty_prefill_when_ttft_missing() -> None:
    from seam.tools.affinity_matrix import _TimedSample

    samples = [
        _TimedSample(t_ns=1_000_000, pct=[10.0] * 8),
        _TimedSample(t_ns=2_000_000, pct=[90.0] * 8),
    ]
    overall, prefill, decode, n_prefill, n_decode = _phase_means(samples, None)
    assert len(overall) == 8
    assert prefill == []
    assert decode == []
    assert n_prefill == 0
    assert n_decode == 0


def test_completed_cells_requires_ok_windows_and_block() -> None:
    ok_gen = {
        "ok": True,
        "n_prefill_util_samples": 12,
        "n_decode_util_samples": 20,
        "prefill_duration_s": 1.0,
        "decode_duration_s": 2.0,
        "prefill_delta_pct_per_cpu": [1.0] * 8,
        "decode_delta_pct_per_cpu": [1.0] * 8,
    }
    runs = {
        "A5": [{"ok": True, "block": 0, "generations": [ok_gen]}],
        "A2": [
            {"ok": True, "block": 0, "generations": []},
            {"ok": True, "block": 1, "generations": [ok_gen]},
        ],
        "A0a": [{"ok": True, "generations": [ok_gen]}],  # missing block → ignored
        "A1": [
            {
                "ok": True,
                "block": 0,
                "generations": [{"ok": True, "n_prefill_util_samples": 0}],  # empty → not done
            }
        ],
    }
    assert _completed_cells(runs) == {(0, "A5"), (1, "A2")}


def test_phase_means_splits_on_positive_ttft() -> None:
    from seam.tools.affinity_matrix import _TimedSample

    samples = [
        _TimedSample(t_ns=1_000_000, pct=[10.0] * 8),
        _TimedSample(t_ns=5_000_000, pct=[90.0] * 8),
    ]
    _overall, prefill, decode, n_prefill, n_decode = _phase_means(samples, 3_000_000)
    assert prefill[0] == pytest.approx(10.0)
    assert decode[0] == pytest.approx(90.0)
    assert n_prefill == 1
    assert n_decode == 1
