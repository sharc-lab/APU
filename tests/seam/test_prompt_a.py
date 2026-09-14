from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import seam.measurement as measurement
from seam.measurement import (
    MeasurementBlock,
    MeasurementRefusalError,
    _rank_process_samples,
    machine_measurement,
)
from seam.telemetry.frequency import FrequencySample, FrequencySampler
from seam.tools.prompt_a import (
    _classify_placement_verdict,
    _load_config,
    _recursive_diff,
    _runtime_environment_metadata,
)


def test_recursive_diff_lists_every_nested_difference() -> None:
    left: dict[str, Any] = {"a": 1, "nested": {"same": True, "x": [1, 2]}, "left_only": 7}
    right: dict[str, Any] = {"a": 2, "nested": {"same": True, "x": [1, 3, 4]}, "right_only": 8}

    rows = _recursive_diff(left, right)

    assert [row["path"] for row in rows] == [
        "/a",
        "/left_only",
        "/nested/x/1",
        "/nested/x/2",
        "/right_only",
    ]
    assert rows[1]["right"] == {"__missing__": True}
    assert rows[-1]["left"] == {"__missing__": True}


def test_frequency_summary_retains_mhz_and_percentage() -> None:
    sampler = FrequencySampler(interval_s=1.0, n_cpus=2)
    sampler._samples = [  # type: ignore[attr-defined]
        FrequencySample(
            t_ns=1,
            mhz_per_cpu=[1000.0, 2000.0],
            pct_of_max_per_cpu=[50.0, 100.0],
            method="fixture",
        ),
        FrequencySample(
            t_ns=2,
            mhz_per_cpu=[1200.0, 1800.0],
            pct_of_max_per_cpu=[60.0, 90.0],
            method="fixture",
        ),
    ]

    summary = sampler.summary()

    assert summary["mean_mhz_per_cpu"] == [1100.0, 1900.0]
    assert summary["min_mhz_per_cpu"] == [1000.0, 1800.0]
    assert summary["mean_pct_of_max_per_cpu"] == [55.0, 95.0]
    assert "inert/nominal" in summary["interpretation"]


def test_measurement_block_invalidation_is_recorded() -> None:
    record: dict[str, Any] = {"valid": True, "invalid_reasons": []}
    block = MeasurementBlock(record)

    block.invalidate("canary drift")

    assert record == {"valid": False, "invalid_reasons": ["canary drift"]}


def test_prompt_a_config_declares_fixed_placement_design() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg, _resolved = _load_config(root)

    assert cfg["design"]["cells"] == [
        {
            "id": "C1",
            "label": "requested_p_cores_clean",
            "requested_affinity": [0, 1, 2, 3],
            "burn": False,
        },
        {
            "id": "C2",
            "label": "requested_lp_e_cores_clean",
            "requested_affinity": [4, 5, 6, 7],
            "burn": False,
        },
        {
            "id": "C3",
            "label": "unrequested_affinity_clean",
            "requested_affinity": None,
            "burn": False,
        },
        {
            "id": "C4",
            "label": "unrequested_affinity_p_core_burn",
            "requested_affinity": None,
            "burn": True,
        },
    ]
    assert cfg["design"]["endpoints"] == ["r_prefill_tok_s", "r_decode_tok_s"]
    assert cfg["design"]["repeats_per_cell"] >= 7
    assert cfg["design"]["target_ratio"] == 1.983
    assert cfg["burn"]["cpus"] == [0, 1, 2, 3]
    assert cfg["quiescence"]["window_s"] > 0
    assert cfg["canary"]["max_relative_drift"] == 0.15


def test_placement_verdict_requires_direct_evidence() -> None:
    target_ci = {
        endpoint: {"ci_low": 1.9, "ci_high": 2.1}
        for endpoint in ("r_prefill_tok_s", "r_decode_tok_s")
    }
    contention_ci = {
        endpoint: {"ci_low": 1.1, "ci_high": 1.3}
        for endpoint in ("r_prefill_tok_s", "r_decode_tok_s")
    }

    verdict, evidence = _classify_placement_verdict(
        c1_c2_ratios=target_ci,
        c3_c4_ratios=contention_ci,
        target_ratio=1.983,
        placement_mechanism={
            "thread_placement_available": False,
            "classification": None,
        },
    )

    assert verdict == "PLACEMENT"
    assert evidence["placement_matches_target"] is True

    off_target = {
        endpoint: {"ci_low": 1.1, "ci_high": 1.3}
        for endpoint in ("r_prefill_tok_s", "r_decode_tok_s")
    }
    verdict, _evidence = _classify_placement_verdict(
        c1_c2_ratios=off_target,
        c3_c4_ratios=contention_ci,
        target_ratio=1.983,
        placement_mechanism={
            "thread_placement_available": False,
            "classification": None,
        },
    )
    assert verdict == "UNEXPLAINED"


def test_openvino_lib_paths_is_captured_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENVINO_LIB_PATHS", "C:\\safe\\openvino")

    metadata = _runtime_environment_metadata()

    assert metadata["OPENVINO_LIB_PATHS"] == {
        "present": True,
        "value": "C:\\safe\\openvino",
        "credential_material": False,
    }


def test_refusal_process_tables_retain_top_ten_names_and_pids() -> None:
    rows = [
        {
            "name": f"proc-{index}",
            "pid": index,
            "sampled_cpu_pct": float(index),
            "rss_bytes": index * 1000,
        }
        for index in range(15)
    ]
    rows.append(
        {
            "name": "System Idle Process",
            "pid": 0,
            "sampled_cpu_pct": 700.0,
            "rss_bytes": 8192,
        }
    )

    tables = _rank_process_samples(rows)

    assert len(tables["top_cpu"]) == 10
    assert len(tables["top_rss"]) == 10
    assert tables["top_cpu"][0]["name"] == "proc-14"
    assert tables["top_cpu"][0]["pid"] == 14
    assert tables["top_rss"][0]["name"] == "proc-14"
    assert tables["top_rss"][0]["pid"] == 14
    assert all(row["pid"] != 0 for row in tables["top_cpu"])


def test_measured_load_refusal_keeps_evidence_and_symmetric_lock_timestamps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refusal_tables = {
        "top_cpu": [{"name": "busy", "pid": 42, "sampled_cpu_pct": 80.0, "rss_bytes": 1}],
        "top_rss": [{"name": "large", "pid": 43, "sampled_cpu_pct": 0.0, "rss_bytes": 999}],
    }
    monkeypatch.setattr(
        measurement,
        "measure_quiescence",
        lambda **_kwargs: {
            "passed": False,
            "failures": ["mean total CPU exceeded"],
            "refusal_process_tables": refusal_tables,
        },
    )
    config = {
        "locking": {"wait_timeout_s": 1, "poll_interval_s": 0.01},
        "quiescence": {
            "window_s": 1,
            "sample_interval_s": 1,
            "total_cpu_max_pct": 20,
            "p_core_cpu_max_pct": 30,
            "available_memory_min_mb": 2048,
        },
        "canary": {"iterations_per_cpu": 1, "seed": 1, "max_relative_drift": 0.15},
        "paging": {
            "sample_interval_s": 0.5,
            "available_memory_min_mb": 500,
            "sustained_nonzero_consecutive_samples": 2,
        },
    }

    with (
        pytest.raises(MeasurementRefusalError) as caught,
        machine_measurement(
            repo_root=tmp_path,
            label="refusal-test",
            config=config,
            p_cpus=[0, 1, 2, 3],
        ),
    ):
        pytest.fail("refused block must not yield")

    record = caught.value.record
    assert record["quiescence"]["refusal_process_tables"] == refusal_tables
    assert record["lock_acquired_utc"]
    assert record["lock_released_utc"]
    assert not (tmp_path / ".locks" / "machine.lock").exists()


def test_prevalidated_startup_gate_is_not_repeated_per_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        measurement,
        "measure_quiescence",
        lambda **_kwargs: pytest.fail("per-block quiescence must not repeat"),
    )
    canary = {
        "affinity_passed": True,
        "workers": [{"runtime_ns": 10}],
    }
    monkeypatch.setattr(measurement, "run_compute_canary", lambda **_kwargs: canary)
    config = {
        "locking": {"wait_timeout_s": 1, "poll_interval_s": 0.01},
        "quiescence": {
            "window_s": 1,
            "sample_interval_s": 1,
            "total_cpu_max_pct": 20,
            "p_core_cpu_max_pct": 30,
            "available_memory_min_mb": 2048,
        },
        "canary": {"iterations_per_cpu": 1, "seed": 1, "max_relative_drift": 0.15},
        "paging": {
            "sample_interval_s": 0.5,
            "available_memory_min_mb": 500,
            "sustained_nonzero_consecutive_samples": 2,
        },
    }
    startup_gate = {
        "passed": True,
        "source": "single_detached_startup_gate",
        "thresholds": config["quiescence"],
    }

    with machine_measurement(
        repo_root=tmp_path,
        label="prevalidated",
        config=config,
        p_cpus=[0],
        prevalidated_quiescence=startup_gate,
    ) as block:
        assert block.record["quiescence"] == startup_gate

    assert block.record["lock_released_utc"]
