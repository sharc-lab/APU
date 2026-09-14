"""Unit tests for M2.1 S1 battery-counter characterization (no hardware required)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from seam.config import ResolvedConfig
from seam.errors import ProfileMismatchError
from seam.gitinfo import GitState
from seam.powerstate import PowerState
from seam.telemetry.s1_battery_char import (
    BatterySample,
    analyze_capacity_series,
    run_characterization,
)


def _sample(
    t_ns: int,
    remaining: float,
    *,
    discharge: float = 15000.0,
    voltage: float = 12000.0,
) -> BatterySample:
    return BatterySample(
        t_ns=t_ns,
        remaining_capacity_mwh=remaining,
        discharge_rate_mw=discharge,
        voltage_mv=voltage,
        sampler_overhead_us=50,
    )


def test_analyze_detects_quantized_updates() -> None:
    # Capacity drops by 10 mWh every 2.0 s - synthetic quantum + period.
    samples = [_sample(0, 50_000.0)]
    for i in range(1, 21):
        samples.append(_sample(int(i * 2.0 * 1e9), 50_000.0 - 10.0 * i))

    result = analyze_capacity_series(samples)
    assert result.n_capacity_changes == 20
    assert result.update_period_median_s is not None
    assert abs(result.update_period_median_s - 2.0) < 1e-9
    assert result.update_period_iqr_s is not None
    assert result.quantization_step_mwh_median == 10.0
    # Smallest observed per-update increment (synthetic series steps by 10 mWh).
    assert result.min_resolvable_energy_mwh == 10.0
    assert result.min_viable_energy_run_duration_s is not None
    # 20 x median update period (edge-effect rule); still 40 s for this synthetic series.
    assert abs(result.min_viable_energy_run_duration_s - 40.0) < 1e-9


def test_analyze_empty_changes_returns_nulls() -> None:
    samples = [_sample(0, 1000.0), _sample(1_000_000_000, 1000.0)]
    result = analyze_capacity_series(samples)
    assert result.n_capacity_changes == 0
    assert result.update_period_median_s is None
    assert result.min_viable_energy_run_duration_s is None


def test_run_characterization_refuses_ac_and_emits_manifest(
    verified_config: ResolvedConfig,
    fake_repo: Path,
    clean_git_state: GitState,
    patch_git: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Profile mismatch must emit a refusal (citable) then raise."""
    from seam.telemetry import s1_battery_char as s1

    patch_git(clean_git_state)
    ac_state = PowerState(
        on_battery=False,
        battery_pct=80.0,
        charging=True,
        battery_saver=False,
        power_plan_name="Best Performance",
        power_plan_guid="ec87a53a-19a6-4f4a-980f-ab27cc929b25",
        overlay_guid="00000000-0000-0000-0000-000000000000",
    )
    monkeypatch.setattr(s1, "capture_power_state", lambda: ac_state)

    with pytest.raises(ProfileMismatchError, match="battery-pinned"):
        run_characterization(
            config=verified_config,
            duration_s=1.0,
            allow_dirty=False,
            repo=fake_repo,
            background_quiesced=True,
        )

    raw = fake_repo / "raw"
    run_dirs = [p for p in raw.iterdir() if p.is_dir() and not p.name.startswith("_")]
    assert len(run_dirs) == 1
    summary = json.loads((run_dirs[0] / "summary.json").read_text(encoding="utf-8"))
    assert summary["verdict"] == "refuse_profile_mismatch"
    manifest = json.loads((run_dirs[0] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["integrity"]["self_check"] == "fail"
    assert manifest["power_state"]["pinned_profile"] == "battery-pinned"
    assert manifest["workload"]["kind"] == "battery_counter_char"
