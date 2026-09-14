"""Verify PowerRequestSystemRequired prevents Kernel-Power 506 across the idle timeout.

Diagnostic only. Writes under derived/power_request_probe/. Seals nothing.

Discovers STANDBYIDLE via powercfg, holds PowerCreateRequest/PowerSetRequest
(PowerRequestSystemRequired), idles past that timeout (+margin), then queries
Kernel-Power for 506/507 in the hold window. A 506 means the assertion is
ineffective on this platform — a platform fact, not a harness workaround.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from seam.tools._winpower import assert_system_required  # noqa: E402
from seam.tools.acceptance_instrumentation import (  # noqa: E402
    collect_kernel_power_events,
)


def _utc() -> str:
    return datetime.now(UTC).isoformat()


def query_standbyidle_s() -> dict[str, Any]:
    """Read AC/DC Sleep after (STANDBYIDLE) from the current power scheme."""
    completed = subprocess.run(
        ["powercfg", "/query", "SCHEME_CURRENT", "SUB_SLEEP", "STANDBYIDLE"],
        capture_output=True,
        text=True,
        check=False,
    )
    text = completed.stdout or ""
    ac = None
    dc = None
    for match in re.finditer(r"Current (AC|DC) Power Setting Index:\s*0x([0-9a-fA-F]+)", text):
        seconds = int(match.group(2), 16)
        if match.group(1) == "AC":
            ac = seconds
        else:
            dc = seconds
    sleep_states = subprocess.run(["powercfg", "/a"], capture_output=True, text=True, check=False)
    return {
        "ac_standbyidle_s": ac,
        "dc_standbyidle_s": dc,
        "powercfg_query_returncode": completed.returncode,
        "powercfg_query_stdout": text,
        "powercfg_a_stdout": sleep_states.stdout or "",
        "modern_standby_available": "Standby (S0 Low Power Idle)" in (sleep_states.stdout or ""),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("auto", "smoke", "full"),
        default="auto",
        help=(
            "auto: full wait if standbyidle <= 45 min, else smoke + document; "
            "smoke: create/set/clear + Kernel-Power query path only; "
            "full: wait standbyidle + margin regardless"
        ),
    )
    parser.add_argument(
        "--margin-s",
        type=float,
        default=120.0,
        help="Extra seconds past STANDBYIDLE for the full hold (default 120)",
    )
    parser.add_argument(
        "--smoke-hold-s",
        type=float,
        default=15.0,
        help="Hold duration for smoke mode (default 15)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_ROOT / "derived" / "power_request_probe",
    )
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    idle = query_standbyidle_s()
    # Prefer AC index; fall back to DC. 0 means never sleep on idle.
    configured_s = idle.get("ac_standbyidle_s")
    if configured_s is None:
        configured_s = idle.get("dc_standbyidle_s")

    mode = args.mode
    if mode == "auto":
        if configured_s is None or configured_s == 0:
            mode = "smoke"
        elif configured_s <= 45 * 60:
            mode = "full"
        else:
            mode = "smoke"

    if mode == "full":
        hold_s = float(configured_s) + float(args.margin_s)
    else:
        hold_s = float(args.smoke_hold_s)

    started_utc = _utc()
    power = assert_system_required(
        reason="SEAM power_request probe; verify no Kernel-Power 506 under hold",
        role="power_request_probe",
    )
    assertion = dict(power.record)
    hold_error = None
    try:
        if not assertion.get("succeeded"):
            hold_error = assertion.get("error") or "power request did not succeed"
        else:
            t0 = time.monotonic()
            # Idle without busy-spin so the only stay-awake claim is the power request.
            while time.monotonic() - t0 < hold_s:
                remaining = hold_s - (time.monotonic() - t0)
                time.sleep(min(30.0, max(0.5, remaining)))
    finally:
        power.release()
        assertion = dict(power.record)
    ended_utc = _utc()

    kp = collect_kernel_power_events(start_utc=started_utc, end_utc=ended_utc)
    events = list(kp.get("events") or [])
    hits_506 = [e for e in events if e.get("id") is not None and int(e["id"]) == 506]
    hits_507 = [e for e in events if e.get("id") is not None and int(e["id"]) == 507]
    modern_standby_fired = bool(hits_506)
    assertion_effective = (
        bool(assertion.get("succeeded"))
        and not modern_standby_fired
        and hold_error is None
        and mode == "full"
    )

    result: dict[str, Any] = {
        "probe": "power_request",
        "diagnostic": True,
        "sealed": False,
        "mode": mode,
        "mode_requested": args.mode,
        "started_utc": started_utc,
        "ended_utc": ended_utc,
        "hold_s": hold_s,
        "margin_s": float(args.margin_s),
        "idle_timeout": idle,
        "configured_standbyidle_s": configured_s,
        "power_request": assertion,
        "hold_error": hold_error,
        "kernel_power": kp,
        "kernel_power_506_count": len(hits_506),
        "kernel_power_507_count": len(hits_507),
        "kernel_power_506_events": hits_506,
        "kernel_power_507_events": hits_507,
        "modern_standby_fired": modern_standby_fired,
        "assertion_effective": assertion_effective,
        "verdict": (
            "EFFECTIVE"
            if assertion_effective
            else (
                "INEFFECTIVE"
                if modern_standby_fired
                else (
                    "SMOKE_OK_FULL_WAIT_NEEDED"
                    if mode == "smoke" and assertion.get("succeeded") and not modern_standby_fired
                    else "ASSERTION_FAILED"
                )
            )
        ),
        "note": (
            "Not a power-policy change. Idle timeout untouched. A 506 during the hold means "
            "PowerRequestSystemRequired does not prevent Modern Standby on this platform; "
            "that is an operator/platform question, not a harness descope."
        ),
    }

    stamp = started_utc.replace(":", "").replace("+", "p")
    out_path = args.out_dir / f"probe_{stamp}.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    latest = args.out_dir / "latest.json"
    latest.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    print(
        json.dumps(
            {
                "event": "power_request_probe.done",
                "verdict": result["verdict"],
                "mode": mode,
                "configured_standbyidle_s": configured_s,
                "hold_s": hold_s,
                "modern_standby_fired": modern_standby_fired,
                "assertion_succeeded": assertion.get("succeeded"),
                "out": str(out_path),
            },
            sort_keys=True,
        )
    )
    return 0 if result["verdict"] in {"EFFECTIVE", "SMOKE_OK_FULL_WAIT_NEEDED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
