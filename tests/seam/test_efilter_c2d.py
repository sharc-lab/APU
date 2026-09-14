"""C2d: paging window scoped to timed blocks; UTF-8 stdio; warmup excluded from gate."""

from __future__ import annotations

import io
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pytest

from seam import measurement as measurement_mod
from seam.gitinfo import repo_root
from seam.jsonlog import get_logger, log_event
from seam.measurement import machine_measurement
from seam.stdio_utf8 import NON_ASCII_PROBE, configure_utf8_stdio
from seam.tools.efilter_run import (
    evaluate_pilot_context_gate,
    timed_block_invalidation_count,
)

_ROOT = repo_root(Path(__file__).parent)


def _minimal_measurement_config() -> dict[str, Any]:
    return {
        "locking": {"wait_timeout_s": 1, "poll_interval_s": 0.01},
        "quiescence": {
            "window_s": 1,
            "sample_interval_s": 1,
            "total_cpu_max_pct": 20,
            "p_core_cpu_max_pct": 30,
            "available_memory_min_mb": 2048,
        },
        "canary": {
            "iterations_per_cpu": 1,
            "seed": 1,
            "max_relative_drift": 0.15,
        },
        "paging": {
            "sample_interval_s": 0.5,
            "available_memory_min_mb": 500,
            "sustained_nonzero_consecutive_samples": 2,
            "exclude_on_failure": True,
            "gate_mode": "baseline_relative",
            "hard_page_reads_threshold_per_s": 1.0,
        },
    }


class _FakePressure:
    instances: list[_FakePressure] = []

    def __init__(self, **_kwargs: Any) -> None:
        self.started = False
        self.stopped = False
        _FakePressure.instances.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> dict[str, Any]:
        self.stopped = True
        return {
            "available_memory_mb_before": 4000.0,
            "available_memory_mb_after": 4000.0,
            "hard_page_reads_per_s": [0.0],
            "invalid_reasons": [],
            "cpu_pct_total": [1.0],
            "cpu_pct_per_core": [[1.0]],
            "valid": True,
        }


def _patch_measurement_basics(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakePressure.instances.clear()
    monkeypatch.setattr(
        measurement_mod,
        "run_compute_canary",
        lambda **_k: {"affinity_passed": True, "workers": [{"runtime_ns": 1000.0}]},
    )
    monkeypatch.setattr(
        measurement_mod,
        "measure_quiescence",
        lambda **_k: {"passed": True, "failures": []},
    )
    monkeypatch.setattr(
        "seam.telemetry.memory.MemoryPressureSampler",
        _FakePressure,
    )


def test_paging_window_timestamps_present_on_timed_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_measurement_basics(monkeypatch)
    with machine_measurement(
        repo_root=tmp_path,
        label="timed",
        config=_minimal_measurement_config(),
        p_cpus=[0],
        prevalidated_quiescence={"passed": True, "source": "test"},
        sample_paging=True,
    ) as block:
        assert block.record["paging_window_start_utc"] is not None
        # end is filled on exit
        pass
    assert block.record["paging_window_start_utc"] is not None
    assert block.record["paging_window_end_utc"] is not None
    assert block.record["paging_sampling_enabled"] is True
    assert len(_FakePressure.instances) == 1
    assert _FakePressure.instances[0].started is True
    assert _FakePressure.instances[0].stopped is True


def test_sampler_not_running_when_sample_paging_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Warmup / setup path: lock+canary without MemoryPressureSampler (C2d)."""
    _patch_measurement_basics(monkeypatch)
    with machine_measurement(
        repo_root=tmp_path,
        label="warmup",
        config=_minimal_measurement_config(),
        p_cpus=[0],
        prevalidated_quiescence={"passed": True, "source": "test"},
        sample_paging=False,
    ) as block:
        pass
    assert block.record["paging_sampling_enabled"] is False
    assert block.record["paging_window_start_utc"] is None
    assert block.record["paging_window_end_utc"] is None
    assert block.record["paging_gate"]["verdict"] == "not_sampled"
    assert _FakePressure.instances == []


def test_sampler_not_running_during_mock_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Teardown runs outside machine_measurement; sampler must not start."""
    from seam.tools import efilter_run as er

    _FakePressure.instances.clear()
    monkeypatch.setattr(
        "seam.telemetry.memory.MemoryPressureSampler",
        _FakePressure,
    )

    class _Backend:
        name = "stub"

        def close(self) -> None:
            return None

        def reload(self) -> None:
            return None

    monkeypatch.setattr(
        er,
        "wait_free_memory_recovery",
        lambda **_k: {
            "recovered": True,
            "free_memory_mb": 2000.0,
            "threshold_mb": 1800.0,
            "elapsed_s": 0.0,
        },
    )
    monkeypatch.setattr(
        er,
        "_throwaway_warmup_generation",
        lambda *_a, **_k: {"ok": True},
    )
    monkeypatch.setattr(er, "free_memory_mb", lambda: 2000.0)
    monkeypatch.setattr(er.gc, "collect", lambda: None)

    record = er._teardown_between_tasks(
        backend=_Backend(),  # type: ignore[arg-type]
        cfg={
            "teardown": {
                "free_memory_recovery_fraction": 0.9,
                "recovery_timeout_s": 1,
                "recovery_poll_interval_s": 0.01,
                "warmup_max_tokens": 8,
            }
        },
        launch_free_memory_mb=2000.0,
        task_id="T",
    )
    assert record["warmup"] == {"ok": True}
    assert _FakePressure.instances == []


def test_utf8_reconfigure_survives_cp1252_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force a cp1252 TextIO, apply helper, write non-ASCII via logging/print path - no raise."""
    buf = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    monkeypatch.setattr(sys, "stdout", buf)
    monkeypatch.setattr(sys, "stderr", buf)
    # Fresh console handler against the patched stderr.
    import seam.jsonlog as jl

    monkeypatch.setattr(jl, "_console_configured", False)
    logger = logging.getLogger("seam")
    logger.handlers.clear()

    applied = configure_utf8_stdio()
    assert applied["encoding"] == "utf-8"
    # After reconfigure, StreamHandler + print must accept NON_ASCII_PROBE.
    get_logger()
    log_event("efilter.c2d_encoding_stress", message=NON_ASCII_PROBE)
    print(NON_ASCII_PROBE)
    buf.flush()


def test_dry_run_encoding_probe_constant_contains_non_ascii() -> None:
    assert "\u2192" in NON_ASCII_PROBE
    assert "\u03b1" in NON_ASCII_PROBE


def test_startup_dry_run_emits_non_ascii_via_print_and_logger(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Dry-run must exercise the real print + log_event paths with non-ASCII (C2d §2.2)."""
    from seam.tools import efilter_run as er

    printed: list[str] = []
    logged: list[str] = []

    monkeypatch.setattr(
        "builtins.print", lambda *a, **_k: printed.append(" ".join(str(x) for x in a))
    )
    monkeypatch.setattr(
        er,
        "log_event",
        lambda event, **fields: logged.append(str(fields.get("message") or event)),
    )

    monkeypatch.setattr(er, "_prepare_isolated_emit_root", lambda *_a, **_k: {"platform": "test"})
    monkeypatch.setattr("seam.rawstore.verify_sealed", lambda _d: True)
    monkeypatch.setattr(
        er,
        "capture_power_state",
        lambda: type(
            "P",
            (),
            {
                "on_battery": False,
                "battery_pct": 100.0,
                "charging": False,
                "battery_saver": False,
            },
        )(),
    )
    monkeypatch.setattr(er, "manifest_power_state", lambda *_a, **_k: {})
    monkeypatch.setattr(er, "assert_acyclic", lambda *_a, **_k: None)

    from seam.tools.efilter_run import _load_cfg

    cfg, _ = _load_cfg(_ROOT)

    # Dry-run builds via build_efilter_summary; persist stub summary/manifest for post-checks.
    class _Handle:
        run_id = "dry-run-test"
        run_dir = tmp_path / "raw" / "dry-run-test"

        def __init__(self) -> None:
            self.run_dir.mkdir(parents=True, exist_ok=True)

    def _emit_with_files(**kwargs: Any) -> _Handle:
        handle = _Handle()
        (handle.run_dir / "summary.json").write_text(
            json.dumps(kwargs.get("summary") or {}), encoding="utf-8"
        )
        (handle.run_dir / "manifest.json").write_text(
            json.dumps({"outputs": kwargs.get("outputs") or {"tasks": "tasks.jsonl"}}),
            encoding="utf-8",
        )
        return handle

    monkeypatch.setattr("seam.manifest.emit", _emit_with_files)

    result = er._efilter_startup_dry_run(
        root=_ROOT,
        cfg=cfg,
        allow_dirty=True,
    )
    assert result["encoding_probe_emitted"] is True
    assert any(NON_ASCII_PROBE in p for p in printed)
    assert any(NON_ASCII_PROBE in m for m in logged)


def test_pilot_invalidation_counting_excludes_warmup() -> None:
    split = timed_block_invalidation_count(task_invalidations=0, warmup_invalidations=3)
    assert split["inside_timed_blocks"] == 0
    assert split["warmup_not_gated"] == 3
    gate = evaluate_pilot_context_gate(
        {
            "max_context_tokens_observed": 5000,
            "per_task_ratios": [5.0, 6.0, 7.0, 8.0, 4.0],
        },
        per_task=[{"context_ratio_cmax_over_cmin": r} for r in [5.0, 6.0, 7.0, 8.0, 4.0]],
        memory_or_canary_invalidations=split["inside_timed_blocks"],
    )
    assert gate["cleared"] is True
    assert "non_paging" in gate["memory_or_canary_invalidations_scope"]

    # C2e/C2f: paging and canary recorded not clearance-fatal; memory floor still fails.
    split_paging = timed_block_invalidation_count(task_invalidations=1, warmup_invalidations=0)
    gate_paging = evaluate_pilot_context_gate(
        {"max_context_tokens_observed": 5000, "per_task_ratios": [7.0]},
        per_task=[{"context_ratio_cmax_over_cmin": 7.0}],
        memory_or_canary_invalidations=0,
        paging_invalidations=split_paging["inside_timed_blocks"],
    )
    assert gate_paging["cleared"] is True
    assert gate_paging["paging_invalidations_run_fatal"] is False

    gate_canary = evaluate_pilot_context_gate(
        {"max_context_tokens_observed": 5000, "per_task_ratios": [7.0]},
        per_task=[{"context_ratio_cmax_over_cmin": 7.0}],
        memory_or_canary_invalidations=1,
        paging_invalidations=0,
        canary_invalidations=1,
        memory_floor_invalidations=0,
    )
    assert gate_canary["cleared"] is True
    assert gate_canary["canary_invalidations_run_fatal"] is False

    gate_memory = evaluate_pilot_context_gate(
        {"max_context_tokens_observed": 5000, "per_task_ratios": [7.0]},
        per_task=[{"context_ratio_cmax_over_cmin": 7.0}],
        memory_or_canary_invalidations=1,
        paging_invalidations=0,
        canary_invalidations=0,
        memory_floor_invalidations=1,
    )
    assert gate_memory["cleared"] is False


def test_efilter_yaml_pins_c2d() -> None:
    import yaml

    cfg = yaml.safe_load((_ROOT / "configs" / "efilter.yaml").read_text(encoding="utf-8"))
    assert cfg.get("c2d_amendment") == "docs/CURSOR_PROMPT_C2d.md"
    paging = yaml.safe_load((_ROOT / "configs" / "measurement.yaml").read_text(encoding="utf-8"))[
        "paging"
    ]
    assert paging["exclude_on_failure"] is True
    assert float(paging["hard_page_reads_threshold_per_s"]) == 1.0
    assert paging["baseline_run_id"].startswith("e6bae93f")
