"""Prompt A2 run-lifecycle: dry-run, cycles, partial reads, paging invalidation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from seam.analysis.prompt_a import read_prompt_a_run
from seam.errors import SeamError
from seam.rawstore import create_run_dir
from seam.telemetry.memory import MemoryPressureSample, summarize_memory_pressure
from seam.tools.prompt_a import (
    _build_placement_report,
    _load_config,
    find_cycles,
    main,
    reproduce_alias_cycle_paths,
)
from seam.tools.prompt_a_lifecycle import (
    append_block_jsonl,
    assert_acyclic,
    block_jsonl_record,
    fail_dry_run_on_cyclic_summary,
    open_in_progress_run,
    progress_status,
    startup_output_path_dry_run,
    synthetic_block_record,
    synthetic_placement_summary,
)


def test_find_cycles_reports_alias_path_for_cb0ed2e3_defect() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg, _resolved = _load_config(root)

    def build_report(records: list[dict[str, Any]]) -> dict[str, Any]:
        return _build_placement_report(
            cfg=cfg,
            records=records,
            valid_counts={"C1": 1, "C2": 0, "C3": 0, "C4": 0},
            attempted_counts={"C1": 1, "C2": 0, "C3": 0, "C4": 0},
            initial_schedule=["C1"],
            session_id="synthetic-session",
            warmup=None,
            startup_quiescence=None,
            prelaunch_run_id=None,
            startup_grace={"duration_s": 0, "machine_lock_held": False},
            audit=None,
            backend=None,
            spec=None,
            power=None,
            charging_complete=None,
            root=None,
        )

    cycles = reproduce_alias_cycle_paths(cfg, build_report=build_report)
    assert cycles
    assert any(
        path.endswith(".validity.timed_measurement") or ".validity.timed_measurement" in path
        for path in cycles
    ), cycles
    assert "$.records[0].validity.timed_measurement" in cycles


def test_production_synthetic_summary_has_no_cycles() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg, _resolved = _load_config(root)
    summary = synthetic_placement_summary(cfg)
    assert find_cycles(summary) == []
    assert_acyclic(summary, label="production synthetic summary")
    for record in summary["records"]:
        assert "admissible" in record["validity"]
        assert record["validity"]["block_id"] == record["block_id"]
        assert "timed_measurement" not in record["validity"]
        assert "measurement" not in record["validity"]
        assert "affinity_requested" in record["placement"]
        assert "readback_after_spinup" in record["placement"]
        assert "readback_mid_generation" in record["placement"]
        assert "matches_request" in record["placement"]
        assert "before_ns" in record["canary"]
        assert "after_ns" in record["canary"]
        assert "drift_pct" in record["canary"]
        assert "admissible" in record["canary"]


def test_flat_block_has_no_mutual_containment() -> None:
    record = synthetic_block_record(
        cell_id="C1",
        repeat_index=0,
        requested_affinity=[0, 1, 2, 3],
        burn=False,
    )
    assert find_cycles(record) == []
    assert record["validity"]["block_id"] == record["block_id"]
    assert "timed_measurement" not in record["validity"]


def test_deliberate_cyclic_dry_run_fails_fast_with_find_cycles() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg, _resolved = _load_config(root)
    with pytest.raises(SeamError, match="circular reference") as caught:
        fail_dry_run_on_cyclic_summary(cfg)
    assert "$.records[0].validity.timed_measurement" in str(caught.value)


def test_startup_output_path_dry_run_passes_and_deletes_temp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[1]
    cfg, _resolved = _load_config(root)

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

    result = startup_output_path_dry_run(root=root, cfg=cfg, allow_dirty=True)
    assert result["passed"] is True
    assert result["seal_verified"] is True
    assert result["outside_project_raw"] is True
    assert not Path(result["temp_repo_root"]).exists()


def test_partial_run_reader_requires_allow_partial(tmp_path: Path) -> None:
    run_id = "11111111-1111-4111-8111-111111111111"
    run_dir = open_in_progress_run(
        root=tmp_path,
        run_id=run_id,
        marker={"run_id": run_id, "state": "IN_PROGRESS"},
    )
    measurement = synthetic_block_record(
        cell_id="C1",
        repeat_index=0,
        requested_affinity=[0, 1, 2, 3],
        burn=False,
        run_id=run_id,
    )
    append_block_jsonl(run_dir, block_jsonl_record(run_id=run_id, measurement=measurement))

    with pytest.raises(SeamError, match="PARTIAL/INCOMPLETE"):
        read_prompt_a_run(tmp_path, run_id)

    partial = read_prompt_a_run(tmp_path, run_id, allow_partial=True)
    assert partial["lifecycle"] == "PARTIAL/INCOMPLETE"
    assert partial["usable_for_final_analysis"] is False
    assert partial["sealed"] is False
    assert len(partial["blocks"]) == 1
    assert partial["blocks"][0]["run_id"] == run_id
    assert partial["blocks"][0]["measurement"]["R_prefill"]["run_id"] == run_id
    assert partial["blocks"][0]["placement"]["readback_mid_generation"]
    assert partial["blocks"][0]["canary"]["admissible"] is True


def test_progress_status_cell_and_repeat_wording() -> None:
    status = progress_status(
        cell_id="C3",
        cell_ids=["C1", "C2", "C3", "C4"],
        repeat_index=2,
        repeats_per_cell=7,
    )
    assert status["cell_progress"] == "cell 3 of 4"
    assert status["repeat_progress"] == "repeat 3 of 7"


def test_sustained_hard_page_reads_invalidation_rule() -> None:
    samples = [
        MemoryPressureSample(
            timestamp_utc="t0",
            available_memory_mb=2000.0,
            hard_page_reads_per_s=0.0,
            hard_page_reads_method="fixture",
            cpu_pct_total=1.0,
            cpu_pct_per_core=[1.0],
        ),
        MemoryPressureSample(
            timestamp_utc="t1",
            available_memory_mb=2000.0,
            hard_page_reads_per_s=1.0,
            hard_page_reads_method="fixture",
            cpu_pct_total=1.0,
            cpu_pct_per_core=[1.0],
        ),
        MemoryPressureSample(
            timestamp_utc="t2",
            available_memory_mb=2000.0,
            hard_page_reads_per_s=2.0,
            hard_page_reads_method="fixture",
            cpu_pct_total=1.0,
            cpu_pct_per_core=[1.0],
        ),
    ]
    summary = summarize_memory_pressure(
        samples, available_memory_min_mb=500.0, sustained_nonzero_samples=2
    )
    assert summary["valid"] is False
    assert any("sustained hard page reads" in reason for reason in summary["invalid_reasons"])
    assert summary["longest_consecutive_nonzero_page_read_samples"] == 2

    single = summarize_memory_pressure(
        [
            samples[0],
            samples[1],
            MemoryPressureSample(
                timestamp_utc="t3",
                available_memory_mb=2000.0,
                hard_page_reads_per_s=0.0,
                hard_page_reads_method="fixture",
                cpu_pct_total=1.0,
                cpu_pct_per_core=[1.0],
            ),
        ],
        available_memory_min_mb=500.0,
        sustained_nonzero_samples=2,
    )
    assert single["valid"] is True


def test_baseline_relative_page_read_threshold() -> None:
    samples = [
        MemoryPressureSample(
            timestamp_utc="t0",
            available_memory_mb=2000.0,
            hard_page_reads_per_s=1.5,
            hard_page_reads_method="fixture",
            cpu_pct_total=1.0,
            cpu_pct_per_core=[1.0],
        ),
        MemoryPressureSample(
            timestamp_utc="t1",
            available_memory_mb=2000.0,
            hard_page_reads_per_s=1.5,
            hard_page_reads_method="fixture",
            cpu_pct_total=1.0,
            cpu_pct_per_core=[1.0],
        ),
    ]
    ok = summarize_memory_pressure(
        samples,
        available_memory_min_mb=500.0,
        sustained_nonzero_samples=2,
        page_read_threshold=2.0,
        gate_mode="baseline_relative",
    )
    assert ok["valid"] is True
    bad = summarize_memory_pressure(
        samples,
        available_memory_min_mb=500.0,
        sustained_nonzero_samples=2,
        page_read_threshold=1.0,
        gate_mode="baseline_relative",
    )
    assert bad["valid"] is False


def test_low_available_memory_invalidation_rule() -> None:
    samples = [
        MemoryPressureSample(
            timestamp_utc="t0",
            available_memory_mb=499.0,
            hard_page_reads_per_s=0.0,
            hard_page_reads_method="fixture",
            cpu_pct_total=1.0,
            cpu_pct_per_core=[1.0],
        ),
        MemoryPressureSample(
            timestamp_utc="t1",
            available_memory_mb=600.0,
            hard_page_reads_per_s=0.0,
            hard_page_reads_method="fixture",
            cpu_pct_total=1.0,
            cpu_pct_per_core=[1.0],
        ),
    ]
    summary = summarize_memory_pressure(
        samples, available_memory_min_mb=500.0, sustained_nonzero_samples=2
    )
    assert summary["valid"] is False
    assert any("available memory" in reason for reason in summary["invalid_reasons"])


def test_block_jsonl_cites_run_id_on_stored_numbers() -> None:
    run_id = "22222222-2222-4222-8222-222222222222"
    measurement = synthetic_block_record(
        cell_id="C2",
        repeat_index=1,
        requested_affinity=[4, 5, 6, 7],
        burn=False,
        run_id=run_id,
    )
    row = block_jsonl_record(run_id=run_id, measurement=measurement)
    assert row["measurement"]["R_prefill"]["run_id"] == run_id
    assert row["measurement"]["R_decode"]["run_id"] == run_id
    assert row["measurement"]["wall_ns"]["run_id"] == run_id
    assert row["measurement"]["ttft_ns"]["run_id"] == run_id
    assert row["canary"]["before_ns"]["run_id"] == run_id
    assert row["canary"]["after_ns"]["run_id"] == run_id
    assert row["canary"]["drift_pct"]["run_id"] == run_id
    assert row["telemetry"]["memory"]["available_memory_mb_before"]["run_id"] == run_id
    assert row["telemetry"]["paging"]["page_reads_per_sec"]["run_id"] == run_id
    assert row["telemetry"]["package_temp"] is None
    assert row["placement"]["affinity_requested"] == [4, 5, 6, 7]
    assert row["placement"]["matches_request"] is True
    assert row["placement"]["readback_mid_generation"]["process_affinity_mask"]


def test_append_block_jsonl_fsync_line(tmp_path: Path) -> None:
    run_dir = create_run_dir("33333333-3333-4333-8333-333333333333", repo_root=tmp_path)
    measurement = synthetic_block_record(
        cell_id="C1",
        repeat_index=0,
        requested_affinity=[0, 1, 2, 3],
        burn=False,
        run_id=run_dir.run_id,
    )
    path = append_block_jsonl(
        run_dir,
        block_jsonl_record(run_id=run_dir.run_id, measurement=measurement),
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["cell_id"] == "C1"
    assert payload["block_id"]
    assert payload["validity"]["block_id"] == payload["block_id"]
    assert "timed_measurement" not in payload["validity"]
    assert set(payload["placement"]) >= {
        "affinity_requested",
        "readback_after_spinup",
        "readback_mid_generation",
        "matches_request",
    }
    assert set(payload["canary"]) >= {
        "before_ns",
        "after_ns",
        "drift_pct",
        "admissible",
    }


def test_c3_c4_affinity_requested_is_null() -> None:
    for cell_id in ("C3", "C4"):
        row = synthetic_block_record(
            cell_id=cell_id,
            repeat_index=0,
            requested_affinity=None,
            burn=cell_id == "C4",
        )
        assert row["placement"]["affinity_requested"] is None
        assert row["placement"]["matches_request"] is None


def test_cb0ed2e3_failure_evidence_is_unsealed_events_only() -> None:
    root = Path(__file__).resolve().parents[1]
    evidence = root / "raw" / "cb0ed2e3-ed70-4627-b231-51016d2b255b"
    assert evidence.is_dir()
    names = sorted(path.name for path in evidence.iterdir())
    assert names == ["events.ndjson"]
    assert not (evidence / ".sealed").exists()
    assert not (evidence / "summary.json").exists()
    assert not (evidence / "blocks.jsonl").exists()


def test_placement_sweep_refuses_allow_dirty_flag() -> None:
    with pytest.raises(SeamError, match="refuses --allow-dirty"):
        main(["--allow-dirty"])
