"""A3 v2 residency sweep: ladder, schedule crossings, balloon touch, reporting-only gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from seam.analysis.a3_residency import (
    ols_regression,
    phase3_fork_from_top_blocks,
    regression_with_bootstrap_extremes,
)
from seam.errors import SeamError
from seam.telemetry.memory import MemoryPressureSample, summarize_memory_pressure
from seam.tools.a3_residency import (
    build_free_memory_ladder,
    build_interleaved_schedule,
    count_ladder_crossings,
    main,
    pretouch_mapped_weights,
    set_process_working_set_size,
)
from seam.tools.memory_balloon import touch_allocate


def test_ladder_even_spacing_from_launch_to_floor() -> None:
    ladder = build_free_memory_ladder(7200.0, n_steps=6, min_free_mb=1200.0)
    assert len(ladder) == 6
    assert ladder[0] == pytest.approx(7200.0)
    assert ladder[-1] == pytest.approx(1200.0)
    gaps = [ladder[i] - ladder[i + 1] for i in range(len(ladder) - 1)]
    assert all(g == pytest.approx(gaps[0]) for g in gaps)


def test_ladder_accepts_low_launch_without_refusal() -> None:
    # No headroom gate: even a 6 GB launch builds a ladder rather than refusing.
    ladder = build_free_memory_ladder(6000.0, n_steps=6, min_free_mb=1200.0)
    assert ladder[0] == pytest.approx(6000.0)
    assert ladder[-1] == pytest.approx(1200.0)


def test_schedule_has_min_crossings() -> None:
    schedule = build_interleaved_schedule(6, 5, seed=20260804, min_crossings=3)
    assert len(schedule) == 30
    assert sorted(schedule.count(i) for i in range(6)) == [5, 5, 5, 5, 5, 5]
    assert count_ladder_crossings(schedule) >= 3


def test_balloon_touch_marks_pages_resident() -> None:
    buf, touched = touch_allocate(3 * 4096)
    assert len(buf) == 3 * 4096
    assert touched == 3 * 4096
    assert buf[0] == 1
    assert buf[4096] == 1
    assert buf[-1] == 1


def test_measurement_yaml_default_exclude_on_failure_true() -> None:
    root = Path(__file__).resolve().parents[1]
    measurement = yaml.safe_load(
        (root / "configs" / "measurement.yaml").read_text(encoding="utf-8")
    )
    assert measurement["paging"]["exclude_on_failure"] is True
    a3 = yaml.safe_load((root / "configs" / "a3_residency.yaml").read_text(encoding="utf-8"))
    assert a3["paging"]["exclude_on_failure"] is False


def test_reporting_only_paging_reasons_do_not_require_exclusion() -> None:
    """Page-read failures are classified separately from available-memory failures."""
    samples = [
        MemoryPressureSample(
            timestamp_utc="t0",
            available_memory_mb=2000.0,
            hard_page_reads_per_s=5.0,
            hard_page_reads_method="fixture",
            cpu_pct_total=1.0,
            cpu_pct_per_core=[1.0],
        ),
        MemoryPressureSample(
            timestamp_utc="t1",
            available_memory_mb=2000.0,
            hard_page_reads_per_s=5.0,
            hard_page_reads_method="fixture",
            cpu_pct_total=1.0,
            cpu_pct_per_core=[1.0],
        ),
    ]
    summary = summarize_memory_pressure(
        samples,
        available_memory_min_mb=500.0,
        sustained_nonzero_samples=2,
        page_read_threshold=1.0,
        gate_mode="baseline_relative",
    )
    assert summary["valid"] is False
    paging_reasons = [r for r in summary["invalid_reasons"] if "page read" in r.lower()]
    other_reasons = [r for r in summary["invalid_reasons"] if "page read" not in r.lower()]
    assert paging_reasons
    assert other_reasons == []
    # Reporting-only path: exclude_on_failure=false → do not treat paging as exclusion.
    exclude_on_failure = False
    invalidate = list(other_reasons)
    if exclude_on_failure:
        invalidate.extend(paging_reasons)
    assert invalidate == []


def test_ols_and_extremes_ratio_helpers() -> None:
    xs = [1200.0, 2400.0, 3600.0, 4800.0, 6000.0, 7200.0]
    # R_decode rises with free memory; ~2x from low to high.
    ys = [8.0, 10.0, 12.0, 14.0, 15.5, 16.0]
    fit = ols_regression(xs, ys)
    assert fit["slope"] > 0
    assert fit["r_squared"] > 0.9
    reg = regression_with_bootstrap_extremes(
        xs, ys, resamples=200, seed=1, confidence=0.95, label="R_decode"
    )
    ratio = reg["bootstrap_extremes"]["extremes_ratio"]["point"]
    assert ratio == pytest.approx(fit["fitted_at_max_x"] / fit["fitted_at_min_x"])


def test_phase3_fork_criterion() -> None:
    top = [
        {
            "block_position": 0,
            "hard_page_reads_per_s": {"samples": [0.0, 5.0, 5.0, 0.0]},
        }
    ]
    fork = phase3_fork_from_top_blocks(top, threshold=1.0, sustained_consecutive_samples=2)
    assert fork["phase3_indicated"] is True
    clean = [
        {
            "block_position": 0,
            "hard_page_reads_per_s": {"samples": [0.0, 0.0, 0.5]},
        }
    ]
    assert (
        phase3_fork_from_top_blocks(clean, threshold=1.0, sustained_consecutive_samples=2)[
            "phase3_indicated"
        ]
        is False
    )


def test_phase3_hooks_are_callable(tmp_path: Path) -> None:
    weight = tmp_path / "openvino_model.bin"
    weight.write_bytes(b"x" * 8192)
    pretouch = pretouch_mapped_weights(tmp_path)
    assert pretouch["implemented"] is True
    assert pretouch["bytes_touched"] == 8192
    ws = set_process_working_set_size(1024 * 1024, 64 * 1024 * 1024)
    assert ws["implemented"] is True


def test_a3_phase1_refuses_as_deleted() -> None:
    from seam.tools.a3_phase1 import main as phase1_main

    with pytest.raises(SeamError, match="Phase 1 is deleted"):
        phase1_main([])


def test_failure_evidence_both_retained() -> None:
    root = Path(__file__).resolve().parents[1]
    for run_id in (
        "9b25332b-cbb7-453d-99ad-1c9ac67e4b89",
        "cb0ed2e3-ed70-4627-b231-51016d2b255b",
    ):
        evidence = root / "raw" / run_id
        assert evidence.is_dir(), f"missing failure evidence {run_id}"
        assert not (evidence / ".sealed").exists()
    a3 = yaml.safe_load((root / "configs" / "a3_residency.yaml").read_text(encoding="utf-8"))
    assert a3["failure_evidence_policy"]["policy"] == "retain_both_unsealed"


def test_schema_includes_a3_residency_kinds() -> None:
    root = Path(__file__).resolve().parents[1]
    schema = json.loads(
        (root / "seam" / "schemas" / "run_manifest.schema.json").read_text(encoding="utf-8")
    )
    kinds = schema["properties"]["workload"]["properties"]["kind"]["enum"]
    assert "a3_residency_sweep" in kinds
    assert "a3_residency_refusal" in kinds
    assert "a3_residency_partial" in kinds


def test_dry_run_exercises_ladder_and_output_path(monkeypatch: pytest.MonkeyPatch) -> None:
    power_block = {
        "on_battery": False,
        "battery_pct_start": 100.0,
        "battery_pct_end": None,
        "soc_at_start": 100.0,
        "soc_at_end": None,
        "charging": False,
        "power_plan": "Best Performance (ec87a53a-19a6-4f4a-980f-ab27cc929b25)",
        "display_brightness": None,
        "defender_realtime": None,
        "windows_update_paused": None,
        "pinned_profile": None,
        "ac_disconnected_for_s": None,
        "discharge_rate_stable": None,
        "background_quiesced": True,
        "wifi_state": None,
        "design_capacity_mwh": None,
        "full_charge_capacity_mwh": None,
    }
    monkeypatch.setattr(
        "seam.tools.prompt_a_lifecycle.capture_power_state",
        lambda: object(),
    )
    monkeypatch.setattr(
        "seam.tools.prompt_a_lifecycle.manifest_power_state",
        lambda _power, background_quiesced=True: {
            **power_block,
            "background_quiesced": background_quiesced,
        },
    )

    class _Sample:
        available_memory_mb = 6400.0

    monkeypatch.setattr(
        "seam.tools.a3_residency.sample_memory_pressure_once",
        lambda: _Sample(),
    )

    rc = main(["--dry-run", "--run-id", "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"])
    assert rc == 0
