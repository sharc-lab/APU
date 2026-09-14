"""ac-pinned charging-complete gate (BatteryStatus WMI + Win32 charging flag)."""

from __future__ import annotations

from seam.powerstate import BatteryStatusWmi, PowerState, is_charging_complete


def _power(**overrides: object) -> PowerState:
    fields = {
        "on_battery": False,
        "battery_pct": 100.0,
        "charging": False,
        "battery_saver": False,
        "power_plan_name": "Best Performance",
        "power_plan_guid": "ec87a53a-19a6-4f4a-980f-ab27cc929b25",
        "overlay_guid": "00000000-0000-0000-0000-000000000000",
    }
    fields.update(overrides)
    return PowerState(**fields)  # type: ignore[arg-type]


def _batt(**overrides: object) -> BatteryStatusWmi:
    fields = {
        "charging": False,
        "discharging": False,
        "charge_rate_mw": 0.0,
        "discharge_rate_mw": 0.0,
        "remaining_capacity_mwh": 69043.0,
        "voltage_mv": 13000.0,
        "power_online": True,
    }
    fields.update(overrides)
    return BatteryStatusWmi(**fields)  # type: ignore[arg-type]


def test_complete_when_charging_false_and_rate_zero() -> None:
    ok, reason = is_charging_complete(
        _power(charging=False),
        _batt(charging=False, charge_rate_mw=0.0),
        charge_rate_max_mw=500.0,
        charging_complete_soc_pct=95.0,
    )
    assert ok is True
    assert reason == "charging_false"


def test_incomplete_while_topping_off() -> None:
    ok, reason = is_charging_complete(
        _power(charging=True, battery_pct=90.0),
        _batt(charging=True, charge_rate_mw=12000.0),
        charge_rate_max_mw=500.0,
        charging_complete_soc_pct=95.0,
    )
    assert ok is False
    assert "charging=true" in reason


def test_alternate_soc_and_low_rate_path() -> None:
    ok, reason = is_charging_complete(
        _power(charging=True, battery_pct=99.0),
        _batt(charging=True, charge_rate_mw=100.0),
        charge_rate_max_mw=500.0,
        charging_complete_soc_pct=95.0,
    )
    assert ok is True
    assert "soc=" in reason
