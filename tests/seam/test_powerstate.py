"""Power-state capture tests (spec §6.1 ``power_state``, AUDIT_LOG.md AF-005).

Windows calls are mocked, so the suite runs anywhere. The behaviour that matters here is that an
unreadable field becomes an explicit ``null`` with a logged reason rather than a plausible default,
and that a session outside MACHINE.md's pinned run conditions is reported rather than tolerated.
"""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

from seam.errors import PinnedConditionError
from seam.powerstate import (
    PowerState,
    _charging_from_battery_flag,
    assert_pinned_for_committed_result,
    capture_power_state,
    check_pinned_conditions,
    manifest_power_state,
)

PINNED = {
    "power_plan_name": "Best Performance",
    "power_plan_guid": "ec87a53a-19a6-4f4a-980f-ab27cc929b25",
}


def _state(**overrides: Any) -> PowerState:
    fields: dict[str, Any] = {
        "on_battery": False,
        "battery_pct": 100.0,
        "charging": False,
        "battery_saver": False,
        "power_plan_name": "Best Performance",
        "power_plan_guid": "ec87a53a-19a6-4f4a-980f-ab27cc929b25",
        "overlay_guid": "00000000-0000-0000-0000-000000000000",
    }
    fields.update(overrides)
    return PowerState(**fields)


# ==================================================================================================
# Manifest rendering
# ==================================================================================================


def test_manifest_block_carries_the_measured_fields() -> None:
    block = manifest_power_state(_state(battery_pct=31.0), battery_pct_end=30.0)

    assert block["on_battery"] is False
    assert block["battery_pct_start"] == 31.0
    assert block["battery_pct_end"] == 30.0
    assert "ec87a53a-19a6-4f4a-980f-ab27cc929b25" in block["power_plan"]


def test_manifest_block_records_the_charging_state() -> None:
    """An AC session under bulk charge is not the same condition as one at full charge.

    Charging consumes adapter headroom and adds chassis heat, both of which depress turbo, so a
    manifest that records only ``on_battery`` cannot distinguish the two.
    """
    charging = manifest_power_state(_state(battery_pct=54.0, charging=True))
    topped_up = manifest_power_state(_state(battery_pct=100.0, charging=False))

    assert charging["charging"] is True
    assert charging["battery_pct_start"] == 54.0
    assert topped_up["charging"] is False
    assert charging["on_battery"] == topped_up["on_battery"] is False


@pytest.mark.parametrize(
    ("battery_flag", "expected"),
    [
        (0x8, True),  # charging
        (0x9, True),  # charging + high
        (0x1, False),  # high, not charging
        (0x0, False),  # neither
        (128, False),  # no system battery
        (255, None),  # unknown - an explicit null, not a guess
    ],
)
def test_charging_is_decoded_from_the_battery_flag(
    battery_flag: int, expected: bool | None
) -> None:
    assert _charging_from_battery_flag(battery_flag) is expected


def test_manifest_block_leaves_unmeasured_quiescence_controls_null() -> None:
    """A plausible value for something nothing measured would be fabricated provenance."""
    block = manifest_power_state(_state())

    assert block["display_brightness"] is None
    assert block["defender_realtime"] is None
    assert block["windows_update_paused"] is None
    assert block["battery_pct_end"] is None
    assert block["pinned_profile"] is None
    assert block["soc_at_start"] == block["battery_pct_start"]
    assert block["soc_at_end"] is None
    assert block["ac_disconnected_for_s"] is None
    assert block["discharge_rate_stable"] is None
    assert block["background_quiesced"] is None
    assert block["wifi_state"] is None
    assert block["design_capacity_mwh"] is None
    assert block["full_charge_capacity_mwh"] is None


# ==================================================================================================
# Pinned run conditions
# ==================================================================================================


def test_pinned_conditions_pass_on_ac_with_the_pinned_plan() -> None:
    assert check_pinned_conditions(_state(), pinned=PINNED) == []


def test_battery_power_is_reported_as_a_deviation() -> None:
    """MACHINE.md: a session off AC power is INVALID, not noisy (ac-pinned profile)."""
    deviations = check_pinned_conditions(_state(on_battery=True), pinned=PINNED)
    assert any("on_battery" in d for d in deviations)


def test_unknown_ac_status_is_a_deviation_rather_than_an_assumption() -> None:
    deviations = check_pinned_conditions(_state(on_battery=None), pinned=PINNED)
    assert any("on_battery" in d for d in deviations)


def test_battery_saver_is_reported_as_a_deviation() -> None:
    """Battery saver clamps turbo, which is exactly the quantity a core-type split measures."""
    deviations = check_pinned_conditions(_state(battery_saver=True), pinned=PINNED)
    assert any("battery saver" in d for d in deviations)


def test_wrong_power_plan_is_reported_as_a_deviation() -> None:
    deviations = check_pinned_conditions(
        _state(
            power_plan_name="Balanced",
            power_plan_guid="381b4222-f694-41f0-9685-ff5bb260df2e",
        ),
        pinned=PINNED,
    )
    assert any("requires plan" in d for d in deviations)


def test_plan_guid_comparison_is_case_insensitive() -> None:
    deviations = check_pinned_conditions(
        _state(power_plan_guid="EC87A53A-19A6-4F4A-980F-AB27CC929B25"), pinned=PINNED
    )
    assert deviations == []


def test_platform_without_pinned_conditions_still_checks_ac() -> None:
    assert check_pinned_conditions(_state(), pinned=None) == []
    assert check_pinned_conditions(_state(on_battery=True), pinned=None) != []


# ==================================================================================================
# Committing a result measured outside the pinned conditions
# ==================================================================================================


def test_a_pinned_session_may_commit_its_result() -> None:
    assert_pinned_for_committed_result([])


def test_an_unpinned_session_may_not_commit_its_result() -> None:
    """MACHINE.md: such a session is INVALID. A passing measurement does not change that."""
    with pytest.raises(PinnedConditionError, match="INVALID"):
        assert_pinned_for_committed_result(["AC power required (ACLineStatus=1)"])


def test_the_refusal_explains_that_the_measurement_is_still_citable() -> None:
    """The run is not discarded - discarding it would recreate the AF-006 traceability hole."""
    with pytest.raises(PinnedConditionError, match="manifest was still emitted"):
        assert_pinned_for_committed_result(["battery saver"])


# ==================================================================================================
# Capture degrades to nulls, never to guesses
# ==================================================================================================


def test_capture_records_null_plan_when_powercfg_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Power state is a recorded property of a run, so it must not be able to fail the run."""

    def _no_powercfg(*_args: Any, **_kwargs: Any) -> None:
        raise FileNotFoundError("powercfg")

    monkeypatch.setattr(subprocess, "run", _no_powercfg)
    monkeypatch.setattr("seam.powerstate._capture_overlay", lambda: None)

    state = capture_power_state()

    assert state.power_plan_name is None
    assert state.power_plan_guid is None


def test_capture_records_null_plan_when_powercfg_output_is_unparseable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_a, **_k: subprocess.CompletedProcess(
            args=[], returncode=0, stdout="unexpected output", stderr=""
        ),
    )
    monkeypatch.setattr("seam.powerstate._capture_overlay", lambda: None)

    state = capture_power_state()

    assert state.power_plan_name is None
    assert state.power_plan_guid is None
