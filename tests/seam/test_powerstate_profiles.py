"""Profile-scoped pinning tests (spec §3.7 / AM-016)."""

from __future__ import annotations

from typing import Any

import pytest

from seam.errors import ProfileMismatchError
from seam.powerstate import (
    PowerState,
    assert_profile,
    manifest_power_state,
    profile_for_class,
    raise_if_profile_mismatch,
)

POWER_CFG: dict[str, Any] = {
    "profiles": {
        "ac-pinned": {
            "on_battery": False,
            "require_charging": False,
            "power_plan_name": "Best Performance",
            "power_plan_guid": "ec87a53a-19a6-4f4a-980f-ab27cc929b25",
            "display_brightness_pct": 50,
        },
        "battery-pinned": {
            "on_battery": True,
            "require_charging": False,
            "settle_s": None,
            "soc_window_pct": None,
            "discharge_rate_stable_band_frac": None,
            "power_plan_name": "Best Performance",
            "power_plan_guid": "ec87a53a-19a6-4f4a-980f-ab27cc929b25",
            "background_quiesced": True,
        },
    },
    "class_profile_map": {
        "topology_verify": "ac-pinned",
        "battery_counter_char": "battery-pinned",
        "thermal_char": "ac-pinned",
        "energy_calibration": "battery-pinned",
        "affinity_matrix": "ac-pinned",
        "mslice_affinity_matrix": "ac-pinned",
    },
}


def _state(**overrides: Any) -> PowerState:
    fields: dict[str, Any] = {
        "on_battery": False,
        "battery_pct": 80.0,
        "charging": False,
        "battery_saver": False,
        "power_plan_name": "Best Performance",
        "power_plan_guid": "ec87a53a-19a6-4f4a-980f-ab27cc929b25",
        "overlay_guid": "00000000-0000-0000-0000-000000000000",
    }
    fields.update(overrides)
    return PowerState(**fields)


def test_class_map_selects_battery_profile() -> None:
    assert profile_for_class(POWER_CFG, "battery_counter_char") == "battery-pinned"
    assert profile_for_class(POWER_CFG, "topology_verify") == "ac-pinned"


def test_battery_class_refuses_ac_session() -> None:
    """Mismatched class must refuse - AC host cannot satisfy battery-pinned."""
    assertion = assert_profile(
        "battery_counter_char",
        _state(on_battery=False, charging=True),
        power_cfg=POWER_CFG,
        background_quiesced=True,
    )
    assert assertion.deviations
    assert assertion.profile_name == "battery-pinned"
    with pytest.raises(ProfileMismatchError, match="battery-pinned"):
        raise_if_profile_mismatch(assertion)


def test_battery_class_refuses_charging_on_battery() -> None:
    assertion = assert_profile(
        "battery_counter_char",
        _state(on_battery=True, charging=True),
        power_cfg=POWER_CFG,
        background_quiesced=True,
    )
    assert any("charging" in d for d in assertion.deviations)


def test_battery_class_passes_when_discharging_and_quiesced() -> None:
    assertion = assert_profile(
        "battery_counter_char",
        _state(on_battery=True, charging=False),
        power_cfg=POWER_CFG,
        background_quiesced=True,
    )
    assert assertion.deviations == []


def test_settle_s_null_does_not_refuse_missing_timer() -> None:
    """AM-006 discipline: unset settle_s must not invent a refuse."""
    assertion = assert_profile(
        "battery_counter_char",
        _state(on_battery=True, charging=False),
        power_cfg=POWER_CFG,
        ac_disconnected_for_s=None,
        background_quiesced=True,
    )
    assert assertion.deviations == []


def test_settle_s_when_set_is_enforced() -> None:
    cfg = {
        **POWER_CFG,
        "profiles": {
            **POWER_CFG["profiles"],
            "battery-pinned": {
                **POWER_CFG["profiles"]["battery-pinned"],
                "settle_s": 120.0,
            },
        },
    }
    assertion = assert_profile(
        "battery_counter_char",
        _state(on_battery=True, charging=False),
        power_cfg=cfg,
        ac_disconnected_for_s=30.0,
        background_quiesced=True,
    )
    assert any("ac_disconnected_for_s" in d for d in assertion.deviations)


def test_committed_battery_profile_bounds_cite_m21_run(real_platform_config: Any) -> None:
    """After M2.1, settle_s / soc_window must be non-null and cite the characterization run."""
    profiles = (real_platform_config.get("power") or {}).get("profiles") or {}
    batt = profiles["battery-pinned"]
    assert batt["settle_s"] == 360
    assert batt["soc_window_pct"] == [40, 85]
    assert batt["discharge_rate_stable_band_frac"] == 0.05
    assert batt["settle_s_run_id"] == "911965cf-257c-4276-954b-17611a5e75eb"
    assert batt["soc_window_run_id"] == "911965cf-257c-4276-954b-17611a5e75eb"


def test_committed_pack_capacities_and_adopted_duration(real_platform_config: Any) -> None:
    power = real_platform_config.get("power") or {}
    pack = power.get("battery") or {}
    assert pack["design_capacity_mwh"] == 68607
    assert pack["full_charge_capacity_mwh"] == 69043
    s1 = real_platform_config.get("s1_characterization") or {}
    assert s1["min_viable_energy_run_duration_s"] == 395.805232
    assert s1["floor_duration_s"] == 410
    assert s1["adopted_duration_s"] == 480
    assert s1["reserve_p90_duration_s"] == 713


def test_manifest_records_profile_fields() -> None:
    assertion = assert_profile(
        "battery_counter_char",
        _state(on_battery=True, charging=False, battery_pct=55.0),
        power_cfg=POWER_CFG,
        background_quiesced=True,
    )
    block = manifest_power_state(
        _state(on_battery=True, charging=False, battery_pct=55.0),
        battery_pct_end=54.0,
        profile=assertion,
        background_quiesced=True,
    )
    assert block["pinned_profile"] == "battery-pinned"
    assert block["soc_at_start"] == 55.0
    assert block["soc_at_end"] == 54.0
    assert block["background_quiesced"] is True
