"""Prefill/decode util phase split must not collapse when TTFT is mid-generation."""

from __future__ import annotations

from types import SimpleNamespace

from seam.backends.local_openvino import resolve_ttft_ns
from seam.tools.affinity_matrix import (
    _deltas,
    _generation_record,
    _phase_means,
    _TimedSample,
)


def _samples_across(ttft_ns: int, *, n_prefill: int = 5, n_decode: int = 5) -> list[_TimedSample]:
    """Synthetic 8-CPU util series with distinct prefill vs decode load."""
    out: list[_TimedSample] = []
    step = ttft_ns // (n_prefill + 1)
    for i in range(n_prefill):
        t = step * (i + 1)
        assert t < ttft_ns
        # Prefill: P-cores hot
        out.append(_TimedSample(t, [90.0, 88.0, 91.0, 87.0, 5.0, 4.0, 6.0, 3.0]))
    decode_span = ttft_ns  # same magnitude after TTFT
    for i in range(n_decode):
        t = ttft_ns + (decode_span // (n_decode + 1)) * (i + 1)
        # Decode: same pattern, slightly lower
        out.append(_TimedSample(t, [80.0, 79.0, 81.0, 78.0, 4.0, 3.0, 5.0, 2.0]))
    return out


class TestPhaseMeans:
    def test_mid_generation_ttft_yields_nonempty_prefill_and_decode(self) -> None:
        ttft_ns = 2_000_000_000  # 2 s mid-generation
        samples = _samples_across(ttft_ns, n_prefill=5, n_decode=7)
        overall, prefill, decode, n_prefill, n_decode = _phase_means(samples, ttft_ns)

        assert n_prefill == 5
        assert n_decode == 7
        assert len(prefill) == 8
        assert len(decode) == 8
        assert len(overall) == 8
        # Prefill means reflect P-core load, not the empty/zero collapse.
        assert prefill[0] > 85.0
        assert prefill[4] < 10.0
        assert decode[0] > 75.0

    def test_missing_ttft_does_not_pool_into_decode(self) -> None:
        samples = _samples_across(2_000_000_000)
        _overall, prefill, decode, n_prefill, n_decode = _phase_means(samples, None)
        assert prefill == []
        assert decode == []
        assert n_prefill == 0
        assert n_decode == 0

    def test_zero_ttft_does_not_pool_into_decode(self) -> None:
        samples = _samples_across(2_000_000_000)
        _overall, prefill, decode, n_prefill, n_decode = _phase_means(samples, 0)
        assert prefill == []
        assert decode == []
        assert n_prefill == 0
        assert n_decode == 0


class TestDeltas:
    def test_empty_phase_mean_yields_empty_deltas(self) -> None:
        assert _deltas([], [10.0] * 8) == []


class TestGenerationRecord:
    def test_record_keeps_separate_phase_means_with_split_ns(self) -> None:
        ttft_ns = 1_500_000_000
        samples = _samples_across(ttft_ns, n_prefill=4, n_decode=4)
        # Sampler clock: generate started 50 ms after sampler t0.
        offset_ns = 50_000_000
        split_ns = offset_ns + ttft_ns
        # Shift sample timestamps onto sampler clock (already absolute from t0).
        result = SimpleNamespace(
            ttft_ns=ttft_ns,
            wall_ns=ttft_ns + 2_000_000_000,
            prompt_tokens=2048,
            completion_tokens=128,
            extra={"ttft_source": "perf_metrics"},
        )
        baseline = [[10.0] * 8] * 3
        record = _generation_record(
            gen_index=1,
            baseline_samples=baseline,
            result=result,  # type: ignore[arg-type]
            util_samples=samples,
            sampler_overhead={},
            ttft_split_ns=split_ns,
        )
        assert record["n_prefill_util_samples"] >= 4
        assert record["n_decode_util_samples"] >= 1
        assert len(record["prefill_mean_pct_per_cpu"]) == 8
        assert len(record["decode_mean_pct_per_cpu"]) == 8
        assert record["r_prefill_tok_s"] is not None
        assert record["ttft_source"] == "perf_metrics"
        # Phases must not be identical pooled vectors.
        assert record["prefill_mean_pct_per_cpu"] != record["decode_mean_pct_per_cpu"]


class TestResolveTtft:
    def test_prefers_perf_metrics(self) -> None:
        ttft, src = resolve_ttft_ns(1_000_000, 2_000_000)
        assert ttft == 1_000_000
        assert src == "perf_metrics"

    def test_falls_back_to_streamer(self) -> None:
        ttft, src = resolve_ttft_ns(None, 2_000_000)
        assert ttft == 2_000_000
        assert src == "streamer_first_token"

    def test_unavailable_when_both_missing(self) -> None:
        ttft, src = resolve_ttft_ns(0, None)
        assert ttft is None
        assert src == "unavailable"


class TestPostHocPhaseSplitDensity:
    """Continuous sampling + post-hoc TTFT split must yield dense phase windows."""

    def test_short_generation_both_phases_ge_10_samples(self) -> None:
        # 3 s prefill + 3 s decode at 10 Hz → >=10 samples each side of the split.
        ttft_ns = 3_000_000_000
        samples: list[_TimedSample] = []
        # 0.1 s steps from 0.05 s through 5.95 s (60 samples)
        for i in range(60):
            t = int((0.05 + i * 0.1) * 1e9)
            load = 90.0 if t < ttft_ns else 70.0
            samples.append(_TimedSample(t, [load] * 8))
        _overall, prefill, decode, n_prefill, n_decode = _phase_means(samples, ttft_ns)
        assert n_prefill >= 10
        assert n_decode >= 10
        assert prefill and decode
        record = _generation_record(
            gen_index=1,
            baseline_samples=[[5.0] * 8] * 3,
            result=SimpleNamespace(  # type: ignore[arg-type]
                ttft_ns=ttft_ns,
                wall_ns=ttft_ns + 3_000_000_000,
                prompt_tokens=2048,
                completion_tokens=128,
                extra={"ttft_source": "streamer_first_token"},
            ),
            util_samples=samples,
            sampler_overhead={},
            ttft_split_ns=ttft_ns,
            min_phase_samples=10,
        )
        assert record["ok"] is True
        assert record["n_prefill_util_samples"] >= 10
        assert record["n_decode_util_samples"] >= 10
        assert record["prefill_duration_s"] > 0
        assert record["decode_duration_s"] > 0
