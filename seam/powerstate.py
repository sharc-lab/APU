"""Host power-state capture and profile-scoped pinning (spec §6.1, §3.2 item 1, §3.7 / AM-016).

`analysis/aipc-c1/MACHINE.md` § "Pinned run conditions (MANDATORY)" states that a session run
outside the pinned regime is **INVALID, not noisy**. AF-005 recorded an unlogged violation; this
module makes the state visible in every manifest.

Pinning is **profile-scoped** (spec §3.7): ``ac-pinned`` vs ``battery-pinned``, selected by
measurement class via ``power.class_profile_map`` in platform YAML. A mismatch is a refusal data
point - emit a manifest, then stop - not a silent fallback.
"""

from __future__ import annotations

import ctypes
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass
from typing import Any, Final

from seam.errors import PinnedConditionError, ProfileMismatchError
from seam.jsonlog import log_event

__all__ = [
    "BatteryStatusWmi",
    "PowerState",
    "ProfileAssertion",
    "assert_pinned_for_committed_result",
    "assert_profile",
    "capture_battery_status_wmi",
    "capture_power_state",
    "check_pinned_conditions",
    "is_charging_complete",
    "manifest_power_state",
    "profile_for_class",
    "raise_if_profile_mismatch",
]

#: ``powercfg`` is a local query with no network or lock dependency, but bound it anyway so a
#: hung service cannot stall a run indefinitely.
_TIMEOUT_S: Final = 15

#: ``SYSTEM_POWER_STATUS.ACLineStatus``: 0 offline, 1 online, 255 unknown.
_AC_OFFLINE: Final = 0
_AC_ONLINE: Final = 1
#: ``SYSTEM_POWER_STATUS.BatteryLifePercent`` sentinel for "unknown".
_BATTERY_PCT_UNKNOWN: Final = 255
#: ``SYSTEM_POWER_STATUS.SystemStatusFlag`` bit 0: battery saver is engaged.
_BATTERY_SAVER_ON: Final = 0x1
#: ``SYSTEM_POWER_STATUS.BatteryFlag`` bit 3: the battery is charging.
_BATTERY_FLAG_CHARGING: Final = 0x8
#: ``SYSTEM_POWER_STATUS.BatteryFlag`` sentinel for "unknown".
_BATTERY_FLAG_UNKNOWN: Final = 255

_ACTIVE_SCHEME_RE: Final = re.compile(
    r"GUID:\s*([0-9a-fA-F-]{36})\s*\((?P<name>.+?)\)\s*$", re.MULTILINE
)


class _SystemPowerStatus(ctypes.Structure):
    """``SYSTEM_POWER_STATUS`` (winbase.h)."""

    _fields_ = (
        ("ACLineStatus", ctypes.c_ubyte),
        ("BatteryFlag", ctypes.c_ubyte),
        ("BatteryLifePercent", ctypes.c_ubyte),
        ("SystemStatusFlag", ctypes.c_ubyte),
        ("BatteryLifeTime", ctypes.c_ulong),
        ("BatteryFullLifeTime", ctypes.c_ulong),
    )


@dataclass(frozen=True, slots=True)
class PowerState:
    """A snapshot of the host's power configuration.

    Every field is nullable because each is *recorded*, not required: an unreadable field must show
    up as an explicit ``null`` with a logged reason, never as a plausible default.
    """

    #: True on battery, False on AC, None if Windows reports ``ACLineStatus=255`` (unknown).
    on_battery: bool | None
    battery_pct: float | None
    #: True while the battery is taking bulk charge. Charging is a *load*: it consumes adapter
    #: headroom and adds chassis heat, both of which depress turbo. A measurement on AC at 50% is
    #: therefore taken under different conditions than the same measurement at 100%, and the
    #: difference is not visible from ``on_battery`` alone.
    charging: bool | None
    #: True if Windows battery saver is engaged. Battery saver clamps turbo, so a measurement taken
    #: under it is not comparable with one taken without it.
    battery_saver: bool | None
    power_plan_name: str | None
    power_plan_guid: str | None
    #: The Windows *power-mode overlay* (the Settings performance slider), which is a separate
    #: control from the power scheme and can clamp frequency independently of it. All-zero GUID
    #: means no overlay is in effect.
    overlay_guid: str | None


def _charging_from_battery_flag(battery_flag: int) -> bool | None:
    """Decode ``SYSTEM_POWER_STATUS.BatteryFlag`` into a charging state.

    Split out from the Win32 call so the decoding is unit-testable on any OS.

    Returns:
        True while charging, False when not, None when Windows reports the flag as unknown. Unknown
        becomes an explicit null rather than False, because "not observed" and "observed not
        charging" are different claims about the session.
    """
    if battery_flag == _BATTERY_FLAG_UNKNOWN:
        log_event(
            "powerstate.battery_flag_unknown",
            severity="warning",
            message="Windows reports BatteryFlag as unknown; recording null charging state",
            battery_flag=battery_flag,
        )
        return None
    return bool(battery_flag & _BATTERY_FLAG_CHARGING)


def _capture_system_power_status() -> tuple[bool | None, float | None, bool | None, bool | None]:
    """Read ``GetSystemPowerStatus``.

    Returns:
        ``(on_battery, battery_pct, charging, battery_saver)``, any of which may be None if Windows
        reports the value as unknown or the call fails.
    """
    if sys.platform != "win32":
        log_event(
            "powerstate.unavailable",
            severity="warning",
            message=f"GetSystemPowerStatus is Windows-only; running on {sys.platform!r}",
        )
        return None, None, None, None

    status = _SystemPowerStatus()
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    if not kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        log_event(
            "powerstate.system_power_status_failed",
            severity="warning",
            message="GetSystemPowerStatus failed; recording null AC and battery fields",
            win32_error=ctypes.get_last_error(),
        )
        return None, None, None, None

    if status.ACLineStatus == _AC_ONLINE:
        on_battery: bool | None = False
    elif status.ACLineStatus == _AC_OFFLINE:
        on_battery = True
    else:
        log_event(
            "powerstate.ac_line_status_unknown",
            severity="warning",
            message="Windows reports ACLineStatus as unknown; recording null rather than guessing",
            ac_line_status=int(status.ACLineStatus),
        )
        on_battery = None

    battery_pct = (
        None
        if status.BatteryLifePercent == _BATTERY_PCT_UNKNOWN
        else float(status.BatteryLifePercent)
    )

    charging = _charging_from_battery_flag(int(status.BatteryFlag))
    battery_saver = bool(status.SystemStatusFlag & _BATTERY_SAVER_ON)
    return on_battery, battery_pct, charging, battery_saver


def _capture_active_scheme() -> tuple[str | None, str | None]:
    """Read the active power scheme via ``powercfg /getactivescheme``.

    ``powercfg`` is used rather than ``PowerGetActiveScheme`` because the friendly name is part of
    what MACHINE.md pins, and the Win32 call returns only a GUID.

    Returns:
        ``(name, guid)``, both None if the command or the parse fails.
    """
    try:
        completed = subprocess.run(
            ["powercfg", "/getactivescheme"],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_S,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        log_event(
            "powerstate.powercfg_failed",
            severity="warning",
            message="could not run powercfg /getactivescheme; recording null power plan",
            error=str(exc),
        )
        return None, None

    if completed.returncode != 0:
        log_event(
            "powerstate.powercfg_nonzero",
            severity="warning",
            message="powercfg /getactivescheme exited non-zero; recording null power plan",
            returncode=completed.returncode,
            stderr=completed.stderr.strip(),
        )
        return None, None

    match = _ACTIVE_SCHEME_RE.search(completed.stdout)
    if match is None:
        log_event(
            "powerstate.powercfg_unparsed",
            severity="warning",
            message="could not parse powercfg /getactivescheme output",
            stdout=completed.stdout.strip(),
        )
        return None, None

    return match.group("name").strip(), match.group(1).lower()


def _capture_overlay() -> str | None:
    """Read the effective power-mode overlay via ``PowerGetEffectiveOverlayScheme``.

    Returns:
        The overlay GUID as a lowercase string, or None if unavailable. The all-zero GUID is a real
        value meaning "no overlay in effect", so it is returned rather than nulled.
    """
    if sys.platform != "win32":
        return None

    raw = (ctypes.c_ubyte * 16)()
    try:
        powrprof = ctypes.WinDLL("powrprof", use_last_error=True)  # type: ignore[attr-defined]
        result = powrprof.PowerGetEffectiveOverlayScheme(ctypes.byref(raw))
    except (AttributeError, OSError) as exc:
        log_event(
            "powerstate.overlay_unavailable",
            severity="warning",
            message="PowerGetEffectiveOverlayScheme unavailable; recording null overlay",
            error=str(exc),
        )
        return None

    if result != 0:
        log_event(
            "powerstate.overlay_query_failed",
            severity="warning",
            message="PowerGetEffectiveOverlayScheme returned a non-zero status",
            status=int(result),
        )
        return None

    return str(uuid.UUID(bytes_le=bytes(raw)))


@dataclass(frozen=True, slots=True)
class BatteryStatusWmi:
    """``root\\WMI`` ``BatteryStatus`` snapshot (Platform A EC path).

    Empirically on aipc-c1 (2026-08-03): when charging is complete on AC,
    ``Charging=false``, ``ChargeRate=0``, ``Discharging=false``, ``PowerOnline=true``.
    While topping off, ``Charging=true`` and ``ChargeRate`` is a positive mW value.
    """

    charging: bool | None
    discharging: bool | None
    charge_rate_mw: float | None
    discharge_rate_mw: float | None
    remaining_capacity_mwh: float | None
    voltage_mv: float | None
    power_online: bool | None


def capture_battery_status_wmi() -> BatteryStatusWmi:
    """Read ``root\\WMI:BatteryStatus``. Null fields on failure - never invent zeros."""
    empty = BatteryStatusWmi(None, None, None, None, None, None, None)
    if sys.platform != "win32":
        return empty
    try:
        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                (
                    "Get-CimInstance -Namespace root/WMI -ClassName BatteryStatus | "
                    "Select-Object -First 1 Charging,Discharging,ChargeRate,DischargeRate,"
                    "RemainingCapacity,Voltage,PowerOnline | ConvertTo-Json -Compress"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_S,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        log_event(
            "powerstate.battery_status_wmi_failed",
            severity="warning",
            message="BatteryStatus WMI query failed",
            error=str(exc),
        )
        return empty
    if completed.returncode != 0 or not completed.stdout.strip():
        return empty
    try:
        import json

        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return empty
    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    if not isinstance(payload, dict):
        return empty

    def _bool(key: str) -> bool | None:
        val = payload.get(key)
        if val is None:
            return None
        return bool(val)

    def _float(key: str) -> float | None:
        val = payload.get(key)
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    return BatteryStatusWmi(
        charging=_bool("Charging"),
        discharging=_bool("Discharging"),
        charge_rate_mw=_float("ChargeRate"),
        discharge_rate_mw=_float("DischargeRate"),
        remaining_capacity_mwh=_float("RemainingCapacity"),
        voltage_mv=_float("Voltage"),
        power_online=_bool("PowerOnline"),
    )


def is_charging_complete(
    state: PowerState,
    battery_status: BatteryStatusWmi,
    *,
    charge_rate_max_mw: float | None = None,
    charging_complete_soc_pct: float | None = None,
) -> tuple[bool, str]:
    """Return ``(complete, reason)`` for ac-pinned charging-complete gate.

    Primary: Win32/WMI ``charging is False``.
    Alternate (when configured): SoC >= threshold AND ChargeRate <= max.
    """
    # Prefer explicit false from either Win32 or WMI.
    if state.charging is False or battery_status.charging is False:
        rate = battery_status.charge_rate_mw
        if rate is not None and charge_rate_max_mw is not None and rate > float(charge_rate_max_mw):
            return (
                False,
                f"charging flag false but ChargeRate={rate} mW > max {charge_rate_max_mw} mW",
            )
        return True, "charging_false"

    if state.charging is True or battery_status.charging is True:
        # Alternate settle path: high SoC + low charge current.
        soc_ok = (
            charging_complete_soc_pct is not None
            and state.battery_pct is not None
            and state.battery_pct >= float(charging_complete_soc_pct)
        )
        rate = battery_status.charge_rate_mw
        rate_ok = (
            charge_rate_max_mw is not None
            and rate is not None
            and rate <= float(charge_rate_max_mw)
        )
        if soc_ok and rate_ok:
            return (
                True,
                f"soc={state.battery_pct}%>= {charging_complete_soc_pct} and "
                f"ChargeRate={rate}<={charge_rate_max_mw}",
            )
        return (
            False,
            f"charging=true (soc={state.battery_pct}, ChargeRate={rate})",
        )

    return False, "charging_state_unknown"


def capture_power_state() -> PowerState:
    """Capture the host power state.

    Never raises: power state is a *recorded property* of a run, in the same way
    :func:`seam.manifest.is_elevated` is, so an unreadable field becomes an explicit null plus a
    logged warning rather than a failed run.
    """
    on_battery, battery_pct, charging, battery_saver = _capture_system_power_status()
    plan_name, plan_guid = _capture_active_scheme()
    state = PowerState(
        on_battery=on_battery,
        battery_pct=battery_pct,
        charging=charging,
        battery_saver=battery_saver,
        power_plan_name=plan_name,
        power_plan_guid=plan_guid,
        overlay_guid=_capture_overlay(),
    )

    log_event(
        "powerstate.captured",
        message=(
            f"AC={'battery' if state.on_battery else 'online'} "
            f"battery={state.battery_pct}% charging={state.charging} "
            f"saver={state.battery_saver} "
            f"plan={state.power_plan_name!r} overlay={state.overlay_guid}"
        ),
        on_battery=state.on_battery,
        battery_pct=state.battery_pct,
        charging=state.charging,
        battery_saver=state.battery_saver,
        power_plan_name=state.power_plan_name,
        power_plan_guid=state.power_plan_guid,
        overlay_guid=state.overlay_guid,
    )
    return state


@dataclass(frozen=True, slots=True)
class ProfileAssertion:
    """Result of checking a captured state against a named pinned profile."""

    profile_name: str
    measurement_class: str
    deviations: list[str]
    #: Seconds since AC disconnect; null until a battery-pinned settle timer is implemented.
    ac_disconnected_for_s: float | None
    #: Rolling discharge-rate stability; null until M2.1 settles the band.
    discharge_rate_stable: bool | None
    background_quiesced: bool | None


def manifest_power_state(
    state: PowerState,
    *,
    battery_pct_end: float | None = None,
    profile: ProfileAssertion | None = None,
    ac_disconnected_for_s: float | None = None,
    discharge_rate_stable: bool | None = None,
    background_quiesced: bool | None = None,
    wifi_state: str | None = None,
    display_brightness: float | None = None,
    defender_realtime: str | None = None,
    windows_update_paused: bool | None = None,
    design_capacity_mwh: float | None = None,
    full_charge_capacity_mwh: float | None = None,
) -> dict[str, Any]:
    """Render a :class:`PowerState` into the manifest ``power_state`` block.

    Quiescence fields that nothing has measured stay null - fabricating them would be the
    AM-006 anti-pattern. Profile fields are recorded when a :class:`ProfileAssertion` is supplied.
    """
    pinned_profile = profile.profile_name if profile is not None else None
    return {
        "on_battery": state.on_battery,
        "battery_pct_start": state.battery_pct,
        "battery_pct_end": battery_pct_end,
        "soc_at_start": state.battery_pct,
        "soc_at_end": battery_pct_end,
        "charging": state.charging,
        "power_plan": (
            None
            if state.power_plan_guid is None
            else f"{state.power_plan_name} ({state.power_plan_guid})"
        ),
        "display_brightness": display_brightness,
        "defender_realtime": defender_realtime,
        "windows_update_paused": windows_update_paused,
        "pinned_profile": pinned_profile,
        "ac_disconnected_for_s": (
            ac_disconnected_for_s
            if ac_disconnected_for_s is not None
            else (profile.ac_disconnected_for_s if profile is not None else None)
        ),
        "discharge_rate_stable": (
            discharge_rate_stable
            if discharge_rate_stable is not None
            else (profile.discharge_rate_stable if profile is not None else None)
        ),
        "background_quiesced": (
            background_quiesced
            if background_quiesced is not None
            else (profile.background_quiesced if profile is not None else None)
        ),
        "wifi_state": wifi_state,
        "design_capacity_mwh": design_capacity_mwh,
        "full_charge_capacity_mwh": full_charge_capacity_mwh,
    }


def profile_for_class(power_cfg: dict[str, Any], measurement_class: str) -> str:
    """Resolve the pinned profile name for a measurement class (§3.7 class→profile map)."""
    class_map = power_cfg.get("class_profile_map") or {}
    if measurement_class not in class_map:
        raise ProfileMismatchError(
            f"no pinned profile mapped for measurement class {measurement_class!r}; "
            f"known classes: {sorted(class_map)}"
        )
    return str(class_map[measurement_class])


def assert_profile(
    measurement_class: str,
    state: PowerState,
    *,
    power_cfg: dict[str, Any],
    ac_disconnected_for_s: float | None = None,
    discharge_rate_stable: bool | None = None,
    background_quiesced: bool | None = None,
) -> ProfileAssertion:
    """Check ``state`` against the profile required by ``measurement_class``.

    Returns a :class:`ProfileAssertion` (possibly with deviations). Does **not** raise on
    mismatch - the caller emits a refusal manifest, then calls
    :func:`raise_if_profile_mismatch` (or raises :class:`ProfileMismatchError` itself). That
    ordering preserves the AF-006 / §3.7 pattern: refusal is a citable data point.
    """
    profile_name = profile_for_class(power_cfg, measurement_class)
    profiles = power_cfg.get("profiles") or {}
    if profile_name not in profiles:
        raise ProfileMismatchError(
            f"profile {profile_name!r} is mapped for {measurement_class!r} but missing from "
            f"power.profiles"
        )
    spec = profiles[profile_name]
    deviations: list[str] = []

    expected_on_battery = spec.get("on_battery")
    if expected_on_battery is not None and state.on_battery is not expected_on_battery:
        deviations.append(
            f"profile {profile_name!r} requires on_battery={expected_on_battery}; "
            f"observed {state.on_battery}"
        )

    require_charging = spec.get("require_charging")
    if require_charging is not None and state.charging is not require_charging:
        deviations.append(
            f"profile {profile_name!r} requires charging={require_charging}; "
            f"observed {state.charging}"
        )

    if state.battery_saver:
        deviations.append("Windows battery saver is engaged, which clamps turbo")

    expected_guid = str(spec.get("power_plan_guid") or "").lower()
    expected_name = str(spec.get("power_plan_name") or "")
    actual_guid = (state.power_plan_guid or "").lower()
    actual_name = state.power_plan_name or ""
    if expected_guid and actual_guid != expected_guid and actual_name != expected_name:
        deviations.append(
            f"profile {profile_name!r} requires plan {expected_name!r} ({expected_guid}); "
            f"observed {actual_name!r} ({actual_guid or 'unknown'})"
        )

    # settle_s is null until M2.1 - when set, require ac_disconnected_for_s >= settle_s.
    settle_s = spec.get("settle_s")
    if settle_s is not None:
        if ac_disconnected_for_s is None:
            deviations.append(
                f"profile {profile_name!r} requires settle_s={settle_s} but "
                f"ac_disconnected_for_s was not measured"
            )
        elif float(ac_disconnected_for_s) < float(settle_s):
            deviations.append(
                f"profile {profile_name!r} requires ac_disconnected_for_s>={settle_s}; "
                f"observed {ac_disconnected_for_s}"
            )

    soc_window = spec.get("soc_window_pct")
    if soc_window is not None:
        if not (isinstance(soc_window, list | tuple) and len(soc_window) == 2):
            deviations.append(f"soc_window_pct must be [lo, hi]; got {soc_window!r}")
        elif state.battery_pct is None:
            deviations.append("soc_window_pct set but battery_pct is null")
        else:
            lo, hi = float(soc_window[0]), float(soc_window[1])
            if not (lo <= state.battery_pct <= hi):
                deviations.append(f"SoC {state.battery_pct}% outside profile window [{lo}, {hi}]")

    if spec.get("background_quiesced") is True and background_quiesced is False:
        deviations.append(
            f"profile {profile_name!r} requires background_quiesced=true; observed false"
        )

    # discharge_rate_stable: when the profile band is still null (pre-M2.1 finding), do not refuse
    # on a missing stability flag - the characterization run is what establishes the band.
    if spec.get("discharge_rate_stable_band_frac") is not None and discharge_rate_stable is False:
        deviations.append(
            f"profile {profile_name!r} requires discharge_rate_stable=true; observed false"
        )

    assertion = ProfileAssertion(
        profile_name=profile_name,
        measurement_class=measurement_class,
        deviations=list(deviations),
        ac_disconnected_for_s=ac_disconnected_for_s,
        discharge_rate_stable=discharge_rate_stable,
        background_quiesced=background_quiesced,
    )
    log_event(
        "powerstate.profile_asserted",
        severity="warning" if deviations else "info",
        message=(
            f"profile {profile_name!r} MISMATCH for class {measurement_class!r}: {deviations}"
            if deviations
            else f"profile {profile_name!r} satisfied for class {measurement_class!r}"
        ),
        profile=profile_name,
        measurement_class=measurement_class,
        deviations=deviations,
    )
    return assertion


def raise_if_profile_mismatch(assertion: ProfileAssertion) -> None:
    """Raise :class:`ProfileMismatchError` when ``assertion.deviations`` is non-empty."""
    if not assertion.deviations:
        return
    raise ProfileMismatchError(
        f"refusing measurement class {assertion.measurement_class!r}: host state does not "
        f"match pinned profile {assertion.profile_name!r} "
        f"({'; '.join(assertion.deviations)}). Emit a refusal manifest first; then stop."
    )


def check_pinned_conditions(state: PowerState, *, pinned: dict[str, Any] | None) -> list[str]:
    """Legacy AC-pin check used before profile maps existed.

    Prefer :func:`assert_profile` with ``measurement_class='topology_verify'``. When ``pinned`` is
    the old ``power.pinned`` block only, synthesise an ac-pinned check so existing callers keep
    working.
    """
    power_cfg: dict[str, Any] = {
        "profiles": {
            "ac-pinned": {
                "on_battery": False,
                "require_charging": None,
                "power_plan_name": (pinned or {}).get("power_plan_name"),
                "power_plan_guid": (pinned or {}).get("power_plan_guid"),
            }
        },
        "class_profile_map": {"topology_verify": "ac-pinned"},
    }
    # If the caller passed a full power config (has profiles), use it directly.
    if pinned and ("profiles" in pinned or "class_profile_map" in pinned):
        power_cfg = pinned

    assertion = assert_profile("topology_verify", state, power_cfg=power_cfg)
    return list(assertion.deviations)


def assert_pinned_for_committed_result(deviations: list[str]) -> None:
    """Refuse to persist a platform result measured outside the pinned run conditions.

    A session outside pinned conditions may still run and still emit its manifest - the
    measurement is real. What it may not do is write its result back into ``configs/platforms/``.

    Raises:
        PinnedConditionError: If ``deviations`` is non-empty.
    """
    if not deviations:
        return

    raise PinnedConditionError(
        "refusing to write a measured result to platform config: this session violates "
        "MACHINE.md's pinned run conditions, which classifies it as INVALID, not noisy "
        f"({'; '.join(deviations)}). The run's manifest was still emitted, so the measurement "
        "remains citable as a diagnostic. Restore the pinned conditions and re-measure."
    )
