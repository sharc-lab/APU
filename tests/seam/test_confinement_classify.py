"""Unit tests for confinement matrix classification (no OpenVINO)."""

from __future__ import annotations

from seam.tools.confinement_classify import (
    classify_cell,
    classify_core,
    compute_noise_band,
    loaded_threshold,
    per_core_loaded_thresholds,
)


class TestClassifyCore:
    def test_loaded_above_half_median(self) -> None:
        assert classify_core(60.0, threshold=25.0, noise_band=5.0) == "LOADED"

    def test_quiet_within_noise_band(self) -> None:
        assert classify_core(3.0, threshold=25.0, noise_band=5.0) == "QUIET"

    def test_ambiguous_between_bands(self) -> None:
        assert classify_core(15.0, threshold=25.0, noise_band=5.0) == "AMBIGUOUS"

    def test_loaded_takes_precedence_over_quiet_when_both_apply(self) -> None:
        # delta above threshold is LOADED even if it were also <= noise_band (edge case)
        assert classify_core(30.0, threshold=10.0, noise_band=50.0) == "LOADED"


class TestLoadedThreshold:
    def test_half_median_over_requested_set(self) -> None:
        deltas = {0: 40.0, 1: 60.0, 2: 20.0, 3: 80.0}
        assert loaded_threshold(deltas, {0, 1, 2, 3}) == 25.0  # 0.5 * median(40,60,20,80)=50


class TestPerCoreLoadedThresholds:
    """threshold(c) = 0.5 * delta(c) from the cell that deliberately targets c."""

    def test_p_from_a5_lpe_from_a6(self) -> None:
        a5 = {0: 80.0, 1: 70.0, 2: 60.0, 3: 50.0, 4: 2.0, 5: 1.0, 6: 1.5, 7: 0.5}
        a6 = {0: 1.0, 1: 2.0, 2: 1.5, 3: 0.5, 4: 90.0, 5: 88.0, 6: 86.0, 7: 84.0}
        thr = per_core_loaded_thresholds(a5, a6, p_cpus={0, 1, 2, 3}, lpe_cpus={4, 5, 6, 7})
        assert thr[0] == 40.0  # 0.5 * A5[0]
        assert thr[3] == 25.0  # 0.5 * A5[3]
        assert thr[4] == 45.0  # 0.5 * A6[4]
        assert thr[7] == 42.0  # 0.5 * A6[7]
        # A5's LP-E deltas must NOT set LP-E thresholds (cross-cell property).
        assert thr[4] != 0.5 * a5[4]
        # A6's P deltas must NOT set P thresholds.
        assert thr[0] != 0.5 * a6[0]

    def test_missing_delta_defaults_zero(self) -> None:
        thr = per_core_loaded_thresholds({}, {}, p_cpus={0}, lpe_cpus={4})
        assert thr == {0: 0.0, 4: 0.0}

    def test_classify_uses_per_core_map(self) -> None:
        # Core 4 has high delta but its A6-derived threshold is even higher → QUIET/AMBIGUOUS
        # depending on noise; core 0 uses A5-derived threshold.
        deltas = {0: 50.0, 1: 55.0, 2: 52.0, 3: 48.0, 4: 20.0, 5: 1.0, 6: 3.0, 7: 2.0}
        thr = {
            0: 30.0,
            1: 30.0,
            2: 30.0,
            3: 30.0,
            4: 40.0,  # 20 < 40 → not LOADED on excluded core
            5: 40.0,
            6: 40.0,
            7: 40.0,
        }
        result = classify_cell(
            deltas,
            {0, 1, 2, 3},
            noise_band=5.0,
            all_cpus=set(range(8)),
            loaded_thresholds=thr,
        )
        # 20 > noise 5 and <= threshold 40 → AMBIGUOUS outside → UNCLEAR (not CONFINED)
        assert result["core_states"]["4"] == "AMBIGUOUS"
        assert result["verdict"] == "UNCLEAR"

    def test_excluded_core_judged_against_other_cell_reference(self) -> None:
        """A5 cell: LP-E excluded cores use A6-derived thresholds (not A5's own LP-E)."""
        # Simulate A5 deltas: P loaded, LP-E near-idle small bump.
        a5_deltas = {0: 80.0, 1: 78.0, 2: 76.0, 3: 74.0, 4: 8.0, 5: 6.0, 6: 4.0, 7: 2.0}
        a6_deltas = {0: 2.0, 1: 2.0, 2: 2.0, 3: 2.0, 4: 90.0, 5: 88.0, 6: 86.0, 7: 84.0}
        thr = per_core_loaded_thresholds(
            a5_deltas, a6_deltas, p_cpus={0, 1, 2, 3}, lpe_cpus={4, 5, 6, 7}
        )
        # LP-E thresholds are high (~42-45); A5's LP-E deltas (2-8) are QUIET under noise=5
        # for cores with delta<=5, AMBIGUOUS for 6-8.
        result = classify_cell(
            a5_deltas,
            {0, 1, 2, 3},
            noise_band=5.0,
            all_cpus=set(range(8)),
            loaded_thresholds=thr,
        )
        assert result["core_states"]["0"] == "LOADED"  # 80 > 40
        assert result["core_states"]["4"] != "LOADED"  # 8 << 45
        assert thr[4] == 45.0  # from A6, not 0.5*8=4 from A5

    def test_included_core_own_reference_is_sanity(self) -> None:
        """Included P-cores in A5 are judged vs A5-derived thresholds (sanity)."""
        a6 = {4: 90.0, 5: 88.0, 6: 86.0, 7: 84.0}
        a5_bad = {0: 80.0, 1: 70.0, 2: 60.0, 3: 2.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0}
        thr_bad = per_core_loaded_thresholds(
            {0: 80.0, 1: 70.0, 2: 60.0, 3: 50.0},  # reference says core 3 should be ~50
            a6,
            p_cpus={0, 1, 2, 3},
            lpe_cpus={4, 5, 6, 7},
        )
        # When classifying a weak A5-like cell against healthy A5/A6 references:
        result = classify_cell(
            a5_bad,
            {0, 1, 2, 3},
            noise_band=1.0,
            all_cpus=set(range(8)),
            loaded_thresholds=thr_bad,
        )
        assert thr_bad[3] == 25.0
        assert result["core_states"]["3"] != "LOADED"
        assert result["verdict"] == "INVALID"


class TestClassifyCell:
    def test_confined_when_outside_quiet_and_inside_loaded(self) -> None:
        deltas = {0: 50.0, 1: 55.0, 2: 52.0, 3: 48.0, 4: 2.0, 5: 1.0, 6: 3.0, 7: 2.0}
        result = classify_cell(deltas, {0, 1, 2, 3}, noise_band=5.0, all_cpus=set(range(8)))
        assert result["verdict"] == "CONFINED"
        assert result["core_states"]["4"] == "QUIET"

    def test_leaked_when_outside_core_loaded(self) -> None:
        deltas = {0: 50.0, 1: 55.0, 2: 52.0, 3: 48.0, 4: 40.0, 5: 1.0, 6: 3.0, 7: 2.0}
        result = classify_cell(deltas, {0, 1, 2, 3}, noise_band=5.0, all_cpus=set(range(8)))
        assert result["verdict"] == "LEAKED"

    def test_unclear_when_outside_ambiguous(self) -> None:
        deltas = {0: 50.0, 1: 55.0, 2: 52.0, 3: 48.0, 4: 15.0, 5: 1.0, 6: 3.0, 7: 2.0}
        result = classify_cell(deltas, {0, 1, 2, 3}, noise_band=5.0, all_cpus=set(range(8)))
        assert result["verdict"] == "UNCLEAR"

    def test_invalid_when_requested_not_all_loaded(self) -> None:
        deltas = {0: 50.0, 1: 55.0, 2: 10.0, 3: 48.0, 4: 2.0, 5: 1.0, 6: 3.0, 7: 2.0}
        result = classify_cell(deltas, {0, 1, 2, 3}, noise_band=5.0, all_cpus=set(range(8)))
        assert result["verdict"] == "INVALID"

    def test_a0_verdict_is_na(self) -> None:
        deltas = {cpu: 40.0 + cpu for cpu in range(8)}
        result = classify_cell(deltas, set(range(8)), noise_band=5.0, all_cpus=set(range(8)))
        assert result["verdict"] == "N/A"

    def test_legacy_global_override_still_works(self) -> None:
        deltas = {0: 50.0, 1: 55.0, 2: 52.0, 3: 48.0, 4: 25.0, 5: 1.0, 6: 3.0, 7: 2.0}
        result = classify_cell(
            deltas,
            {0, 1, 2, 3},
            noise_band=5.0,
            all_cpus=set(range(8)),
            loaded_threshold_value=20.0,
        )
        assert result["loaded_threshold"] == 20.0
        assert result["verdict"] == "LEAKED"


class TestNoiseBand:
    def test_two_times_mean_cv(self) -> None:
        pooled = {
            0: [10.0, 12.0, 11.0, 13.0],
            1: [5.0, 5.5, 4.5, 5.0],
        }
        band = compute_noise_band(pooled)
        assert band > 0.0

    def test_empty_pooled_returns_zero(self) -> None:
        assert compute_noise_band({}) == 0.0

    def test_single_sample_per_core_yields_zero_cv(self) -> None:
        pooled = {0: [10.0], 1: [5.0]}
        assert compute_noise_band(pooled) == 0.0


class TestScheduleBlocks:
    def test_no_consecutive_duplicates_within_block(self) -> None:
        from seam.tools.affinity_matrix import CONFIGS, schedule_blocks

        blocks, _seed = schedule_blocks(list(CONFIGS), n_blocks=3, seed=42)
        assert len(blocks) == 3
        for block in blocks:
            assert len(block) == 8
            assert len(set(block)) == 8
            for i in range(len(block) - 1):
                assert block[i] != block[i + 1]

    def test_no_duplicate_across_block_boundary(self) -> None:
        from seam.tools.affinity_matrix import schedule_blocks

        blocks, _seed = schedule_blocks(["A0a", "A0b", "A1"], n_blocks=5, seed=99)
        for i in range(len(blocks) - 1):
            assert blocks[i][-1] != blocks[i + 1][0]


class TestA0ReferenceExempt:
    """A0a/A0b are exempt from loaded-cores sanity; they report diagnostics only."""

    def test_a0b_partial_load_is_reference_not_gate(self) -> None:
        from seam.tools.affinity_matrix import CONFIGS, _summarize_cell

        class _Cfg:
            def get(self, key: str, default: object = None) -> object:
                if key == "topology.p_cpus":
                    return (0, 1, 2, 3)
                if key == "topology.lpe_cpus":
                    return (4, 5, 6, 7)
                return default

        gen = {
            "ok": True,
            "n_prefill_util_samples": 20,
            "n_decode_util_samples": 40,
            "prefill_duration_s": 2.0,
            "decode_duration_s": 4.0,
            "prefill_delta_pct_per_cpu": [70.0, 72.0, 71.0, 69.0, 1.0, 1.0, 0.5, 0.5],
            "decode_delta_pct_per_cpu": [78.0, 80.0, 82.0, 67.0, 4.0, 4.5, 2.0, 0.8],
            "baseline_mean_pct_per_cpu": [10.0] * 8,
            "r_prefill_tok_s": 400.0,
            "r_decode_tok_s": 15.0,
        }
        cell_runs = [{"ok": True, "generations": [gen]}]
        thr = dict.fromkeys(range(8), 30.0)
        summary = _summarize_cell(
            "A0b",
            cell_runs,
            noise_band=1.5,
            all_cpus=set(range(8)),
            platform_cfg=_Cfg(),
            loaded_thresholds=thr,
        )
        diag = summary["reference_diagnostics"]
        assert diag["exempt_from_loaded_cores_sanity"] is True
        assert diag["threads"] == 4
        assert diag["loaded_cpus"] == [0, 1, 2, 3]
        assert summary["classification"]["verdict"] == "N/A"
        assert CONFIGS["A0b"].cluster == "all"

    def test_a0a_four_core_load_does_not_fail_verification_gate(self) -> None:
        """Demoted: A0a loading only 4 cores is diagnostic, not a verification failure."""
        from seam.tools.affinity_matrix import _verification_gates

        report = {
            "runs": {},
            "summary": {
                "A0a": {
                    "reference_diagnostics": {
                        "loaded_core_count": 4,
                        "loaded_cpus": [0, 1, 2, 3],
                    }
                }
            },
            "quiesce": {
                "charging": False,
                "charging_end": False,
                "charging_complete": True,
            },
            "loaded_threshold": {
                "per_core": {str(i): 20.0 + i for i in range(8)},
                "value": 23.5,
            },
            "noise_band": {"value": 1.4},
        }
        gates = _verification_gates(report, min_phase_samples=10)
        assert gates["a0a_saturation_gate"] == "demoted_diagnostic_only"
        assert gates["a0a_loaded_core_count"] == 4
        assert not any("A0a loaded_core_count" in f for f in gates["failures"])
