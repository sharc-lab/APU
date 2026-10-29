"""INF-6: mid-run AC / battery transition refuse at canary cells.

Launch-time AC gate is necessary but not sufficient: on 2026-09-17 the XPS
came off AC mid-run (c3efd48e, 8.3% canary drift). Record Win32_Battery
BatteryStatus + EstimatedChargeRemaining at every canary and refuse on any
AC-status transition. Hosts with no battery pass trivially.
"""

from __future__ import annotations

import subprocess
from typing import Any

_TIMEOUT_S = 30.0

# Win32_Battery.BatteryStatus: 2 = on AC / charging (aipc-c1 / measurement_gates).
_AC_BATTERY_STATUS = 2


class CanaryPowerTransitionAbort(Exception):  # noqa: N818
    """Raised when AC status transitions between canary cells. Hard refuse."""

    def __init__(self, detail: str, *, snapshot: dict[str, Any] | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        self.snapshot = snapshot or {}


def capture_canary_power_snapshot() -> dict[str, Any]:
    """Snapshot Win32_Battery for one canary cell.

    No instances => mains-only host => ``battery_present=False``, ``on_ac=True``.
    Never invents charge values on probe failure — records ``probe_error``.
    """
    ps = (
        "$b = @(Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue); "
        "if ($b.Count -eq 0) { "
        "  Write-Output 'NO_BATTERY'; "
        "} else { "
        "  $s = @($b | ForEach-Object { [int]$_.BatteryStatus }); "
        "  $c = @($b | ForEach-Object { "
        "    if ($null -eq $_.EstimatedChargeRemaining) { 'null' } "
        "    else { [string]([int]$_.EstimatedChargeRemaining) } "
        "  }); "
        "  Write-Output ('BATTERY|' + ($s -join ',') + '|' + ($c -join ',')); "
        "}"
    )
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_S,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        return {
            "battery_present": None,
            "on_ac": None,
            "battery_status_values": None,
            "estimated_charge_remaining_pct": None,
            "note": "probe_failed",
            "probe_error": f"{type(exc).__name__}: {exc}",
        }
    if completed.returncode != 0:
        return {
            "battery_present": None,
            "on_ac": None,
            "battery_status_values": None,
            "estimated_charge_remaining_pct": None,
            "note": "probe_failed",
            "probe_error": (completed.stderr or "").strip() or f"rc={completed.returncode}",
        }
    lines = (completed.stdout or "").strip().splitlines()
    text = lines[-1].strip() if lines else ""
    if text == "NO_BATTERY":
        return {
            "battery_present": False,
            "on_ac": True,
            "battery_status_values": None,
            "estimated_charge_remaining_pct": None,
            "note": "no_battery_mains_only_assume_ac",
            "probe_error": None,
        }
    if text.startswith("BATTERY|"):
        parts = text.split("|")
        statuses_raw = parts[1] if len(parts) > 1 else ""
        charges_raw = parts[2] if len(parts) > 2 else ""
        statuses = [int(x) for x in statuses_raw.split(",") if x != ""]
        charges: list[int | None] = []
        for x in charges_raw.split(","):
            if x == "" or x == "null":
                charges.append(None)
            else:
                charges.append(int(x))
        on_ac = bool(statuses) and all(s == _AC_BATTERY_STATUS for s in statuses)
        return {
            "battery_present": True,
            "on_ac": on_ac,
            "battery_status_values": statuses,
            "estimated_charge_remaining_pct": charges[0] if charges else None,
            "estimated_charge_remaining_pct_all": charges,
            "note": "ac_ok" if on_ac else "AC_offline",
            "probe_error": None,
        }
    return {
        "battery_present": None,
        "on_ac": None,
        "battery_status_values": None,
        "estimated_charge_remaining_pct": None,
        "note": "unparseable",
        "probe_error": f"unparseable AC probe: {text!r}",
    }


def assert_no_ac_transition(
    *,
    previous: dict[str, Any] | None,
    current: dict[str, Any],
) -> dict[str, Any]:
    """Refuse on AC-status transition between canary cells.

    - No battery (current or previous): pass trivially.
    - First snapshot (previous is None): establish baseline; pass.
    - Probe failure with unknown on_ac: refuse (hard refuse, no silent continue).
    - on_ac True→False or False→True: raise CanaryPowerTransitionAbort.
    """
    if current.get("battery_present") is False:
        return {
            "ok": True,
            "action": "pass_no_battery",
            "note": current.get("note") or "no_battery_mains_only_assume_ac",
            "previous_on_ac": None if previous is None else previous.get("on_ac"),
            "current_on_ac": True,
        }
    if previous is not None and previous.get("battery_present") is False:
        return {
            "ok": True,
            "action": "pass_no_battery_baseline",
            "note": "previous canary recorded no battery; continue trivial pass",
            "previous_on_ac": True,
            "current_on_ac": current.get("on_ac"),
        }

    cur_on = current.get("on_ac")
    if cur_on is None:
        detail = (
            "canary power probe failed or unparseable; refuse rather than "
            f"guess AC state ({current.get('probe_error') or current.get('note')})"
        )
        raise CanaryPowerTransitionAbort(detail, snapshot=current)

    if previous is None:
        return {
            "ok": True,
            "action": "baseline",
            "note": "first canary power snapshot; no prior transition to compare",
            "previous_on_ac": None,
            "current_on_ac": cur_on,
        }

    prev_on = previous.get("on_ac")
    if prev_on is None:
        detail = (
            "previous canary power snapshot missing on_ac; refuse rather than "
            "guess whether a transition occurred"
        )
        raise CanaryPowerTransitionAbort(detail, snapshot=current)

    if bool(prev_on) != bool(cur_on):
        detail = (
            f"AC status transition between canaries: on_ac {prev_on} -> {cur_on} "
            f"(BatteryStatus previous={previous.get('battery_status_values')} "
            f"current={current.get('battery_status_values')}; "
            f"charge_pct previous={previous.get('estimated_charge_remaining_pct')} "
            f"current={current.get('estimated_charge_remaining_pct')})"
        )
        raise CanaryPowerTransitionAbort(detail, snapshot=current)

    return {
        "ok": True,
        "action": "pass_stable",
        "note": "on_ac unchanged since previous canary",
        "previous_on_ac": prev_on,
        "current_on_ac": cur_on,
    }
