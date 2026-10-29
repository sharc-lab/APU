"""Platform-aware pre-measurement gates (PORT-2).

Gates historically hardcoded for XPS / aipc-c1. Values come from
``configs/platforms/<id>.yaml`` ``measurement_gates``; host probes fill observed
fields. No silent fallbacks: unknown onset is recorded; no battery is AC-pass with
an explicit reason; power-plan GUID is recorded but not matched. AC processor
throttle 100/100 is the pass criterion.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import yaml

from seam.isolation import TIER1_CONTENDING_PROCESS_NAMES, _normalize_process_name

__all__ = [
    "UNCOLD_UPTIME_REASON",
    "GateResult",
    "MeasurementGateReport",
    "evaluate_measurement_gates",
    "load_platform_measurement_gates",
    "resolve_platform_id",
    "run_environment_gate_fields",
]

# Recorded when the sequencer skips only the uptime gate (-NoRebootDeviation).
UNCOLD_UPTIME_REASON: Final = (
    "remote host; Tailscale unattended mode not confirmed; "
    "AutoAdminLogon=0; reboot would risk losing access"
)

_GUID_RE: Final = re.compile(
    r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
)
_AC_INDEX_RE: Final = re.compile(
    r"Current AC Power Setting Index:\s*(0x[0-9a-fA-F]+|\d+)",
    re.IGNORECASE,
)
_TIMEOUT_S: Final = 60


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    reason: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MeasurementGateReport:
    platform_id: str
    gates: list[GateResult]
    all_passed: bool
    refusal_reasons: list[str]
    workloads_session_host: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform_id": self.platform_id,
            "all_passed": self.all_passed,
            "refusal_reasons": list(self.refusal_reasons),
            "workloads_session_host": dict(self.workloads_session_host),
            "gates": [
                {
                    "name": g.name,
                    "passed": g.passed,
                    "reason": g.reason,
                    "detail": dict(g.detail),
                }
                for g in self.gates
            ],
        }


def resolve_platform_id(
    repo_root: Path,
    *,
    explicit: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Resolve which platform YAML to load. Never silently invents a host."""
    env = environ if environ is not None else os.environ
    if explicit and str(explicit).strip():
        return str(explicit).strip()
    if env.get("SEAM_PLATFORM_ID", "").strip():
        return env["SEAM_PLATFORM_ID"].strip()
    pin = repo_root / "configs" / "platforms" / "active_platform_id.txt"
    if pin.is_file():
        text = pin.read_text(encoding="utf-8").strip()
        if text:
            return text.splitlines()[0].strip()
    host = socket.gethostname().strip().lower()
    platforms_dir = repo_root / "configs" / "platforms"
    matches: list[str] = []
    if platforms_dir.is_dir():
        for path in sorted(platforms_dir.glob("*.yaml")):
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            identity = data.get("identity") or {}
            cfg_host = str(identity.get("hostname") or "").strip().lower()
            pid = str(data.get("platform_id") or path.stem)
            if cfg_host and cfg_host == host:
                matches.append(pid)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            f"hostname {host!r} matches multiple platforms {matches}; "
            "set SEAM_PLATFORM_ID or configs/platforms/active_platform_id.txt"
        )
    raise RuntimeError(
        "platform id unresolved: set SEAM_PLATFORM_ID, pass --platform-id, "
        "or write configs/platforms/active_platform_id.txt "
        f"(hostname={host!r} matched none)"
    )


def load_platform_measurement_gates(repo_root: Path, platform_id: str) -> dict[str, Any]:
    path = repo_root / "configs" / "platforms" / f"{platform_id}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"platform config absent: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if str(data.get("platform_id")) != platform_id:
        raise RuntimeError(
            f"platform config {path} declares platform_id={data.get('platform_id')!r} "
            f"but was loaded as {platform_id!r}"
        )
    gates = data.get("measurement_gates")
    if gates is None:
        raise RuntimeError(
            f"platform {platform_id}: measurement_gates block missing, "
            "refuse to evaluate gates without declared floors/onset"
        )
    if not isinstance(gates, dict):
        raise TypeError(
            f"platform {platform_id}: measurement_gates must be a dict, got {type(gates).__name__}"
        )
    return gates


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=_TIMEOUT_S,
        check=False,
    )


def platform_declares_no_battery(repo_root: Path, platform_id: str) -> bool:
    """True only when the platform YAML sets power.has_battery to false."""
    path = repo_root / "configs" / "platforms" / f"{platform_id}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    power = data.get("power") or {}
    return power.get("has_battery") is False


def probe_ac_gate() -> GateResult:
    """AC online. No Win32_Battery instances => mains-only host => pass with reason."""
    ps = (
        "$b = @(Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue); "
        "if ($b.Count -eq 0) { "
        "  Write-Output 'NO_BATTERY'; "
        "} else { "
        "  $s = @($b | ForEach-Object { [int]$_.BatteryStatus }); "
        "  Write-Output ('BATTERY|' + ($s -join ',')); "
        "}"
    )
    completed = _run(["powershell", "-NoProfile", "-Command", ps])
    if completed.returncode != 0:
        return GateResult(
            name="ac",
            passed=False,
            reason=f"Win32_Battery query failed: {completed.stderr.strip()}",
            detail={"probe_error": completed.stderr.strip()},
        )
    lines = (completed.stdout or "").strip().splitlines()
    text = lines[-1].strip() if lines else ""
    if text == "NO_BATTERY":
        return GateResult(
            name="ac",
            passed=True,
            reason="no_battery_mains_only_assume_ac",
            detail={
                "battery_present": False,
                "ac_ok": True,
                "recorded_reason": ("Win32_Battery returned no instances; host is mains-only"),
            },
        )
    if text.startswith("BATTERY|"):
        statuses = [int(x) for x in text.split("|", 1)[1].split(",") if x != ""]
        # BatteryStatus 2 = on AC / charging (aipc-c1 launcher convention).
        ac_ok = bool(statuses) and all(s == 2 for s in statuses)
        return GateResult(
            name="ac",
            passed=ac_ok,
            reason="ac_ok" if ac_ok else "AC_offline",
            detail={
                "battery_present": True,
                "battery_status_values": statuses,
                "ac_ok": ac_ok,
            },
        )
    return GateResult(
        name="ac",
        passed=False,
        reason=f"unparseable AC probe: {text!r}",
        detail={"raw": text},
    )


def probe_processor_ac_gate() -> GateResult:
    """Assert AC PROCTHROTTLEMIN/MAX == 100; record scheme GUID (do not match it)."""
    scheme = _run(["powercfg", "/getactivescheme"])
    scheme_text = (scheme.stdout or "").strip()
    guid_m = _GUID_RE.search(scheme_text)
    scheme_guid = guid_m.group(1).lower() if guid_m else None
    name_m = re.search(r"\(([^)]+)\)\s*$", scheme_text)
    scheme_name = name_m.group(1).strip() if name_m else None

    def _ac_percent(alias: str) -> int | None:
        out = _run(["powercfg", "/query", "SCHEME_CURRENT", "SUB_PROCESSOR", alias])
        if out.returncode != 0:
            return None
        matches = _AC_INDEX_RE.findall(out.stdout or "")
        if not matches:
            return None
        raw = matches[0]
        return int(raw, 16) if str(raw).lower().startswith("0x") else int(raw)

    amin = _ac_percent("PROCTHROTTLEMIN")
    amax = _ac_percent("PROCTHROTTLEMAX")
    if amin is None or amax is None:
        ok = False
        reason = "processor_ac_unreadable"
    else:
        ok = amin == 100 and amax == 100
        reason = "processor_ac_100_100" if ok else "processor_ac_not_100_100"
    return GateResult(
        name="processor_ac",
        passed=ok,
        reason=reason,
        detail={
            "scheme_guid": scheme_guid,
            "scheme_name": scheme_name,
            "scheme_text": scheme_text,
            "procthrottlemin_ac": amin,
            "procthrottlemax_ac": amax,
            "required_ac_min": 100,
            "required_ac_max": 100,
        },
    )


def probe_available_mb() -> float | None:
    ps = (
        "try { "
        "(Get-Counter '\\Memory\\Available MBytes' -ErrorAction Stop)"
        ".CounterSamples[0].CookedValue "
        "} catch { Write-Output 'ERR'; exit 1 }"
    )
    completed = _run(["powershell", "-NoProfile", "-Command", ps])
    if completed.returncode != 0:
        return None
    lines = (completed.stdout or "").strip().splitlines()
    if not lines:
        return None
    try:
        return float(lines[-1].strip())
    except ValueError:
        return None


def classify_tier1_processes(process_names: list[str] | tuple[str, ...]) -> GateResult:
    """Refuse when a tier-1 name is resident. Matching is case-insensitive.

    ``chrome`` and ``Chrome.exe`` both normalize to ``chrome.exe``. Tier-2
    names such as ``msedgewebview2.exe`` are not a refusal.
    """
    resident = sorted(
        {
            normalized
            for normalized in (_normalize_process_name(name) for name in process_names)
            if normalized in TIER1_CONTENDING_PROCESS_NAMES
        }
    )
    if resident:
        return GateResult(
            name="tier1_processes",
            passed=False,
            reason="tier1_resident",
            detail={"resident": resident},
        )
    return GateResult(
        name="tier1_processes",
        passed=True,
        reason="tier1_clear",
        detail={"resident": []},
    )


def probe_process_names() -> list[str]:
    """Live process names. An unreadable probe must not pass the gate."""
    import psutil

    found: list[str] = []
    for process in psutil.process_iter(["name"]):
        try:
            name = process.info.get("name")
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
        if name:
            found.append(str(name))
    return found


def probe_uptime_s() -> float | None:
    try:
        import psutil

        return float(time.time() - psutil.boot_time())
    except Exception:
        ps = (
            "$os = Get-CimInstance Win32_OperatingSystem; "
            "$boot = [datetime]$os.LastBootUpTime; "
            "[math]::Round(((Get-Date) - $boot).TotalSeconds, 3)"
        )
        completed = _run(["powershell", "-NoProfile", "-Command", ps])
        if completed.returncode != 0:
            return None
        try:
            return float((completed.stdout or "").strip().splitlines()[-1])
        except (ValueError, IndexError):
            return None


def evaluate_measurement_gates(
    repo_root: Path,
    *,
    platform_id: str | None = None,
    available_mb: float | None = None,
    uptime_s: float | None = None,
    skip_host_probes: bool = False,
    skip_uptime: bool = False,
    ac_override: GateResult | None = None,
    processor_override: GateResult | None = None,
    process_names: list[str] | tuple[str, ...] | None = None,
) -> MeasurementGateReport:
    """Evaluate measurement gates for ``platform_id``.

    Host probes may be skipped in unit tests by supplying overrides / values.
    """
    pid = platform_id or resolve_platform_id(repo_root)
    cfg = load_platform_measurement_gates(repo_root, pid)

    if cfg.get("pre_run_available_mb_min") is None:
        raise RuntimeError(
            f"platform {pid}: measurement_gates.pre_run_available_mb_min is required"
        )
    floor_f = float(cfg["pre_run_available_mb_min"])
    floor_cite = cfg.get("pre_run_available_mb_min_citation")

    if "onset_s" not in cfg:
        raise RuntimeError(
            f"platform {pid}: measurement_gates.onset_s key is required "
            "(use null when unknown; do not omit)"
        )
    onset_raw = cfg.get("onset_s")
    onset_s = None if onset_raw is None else float(onset_raw)
    onset_status = str(
        cfg.get("onset_status") or ("measured" if onset_s is not None else "unknown")
    )
    onset_cite = cfg.get("onset_s_citation")

    if cfg.get("max_uptime_s") is None:
        raise RuntimeError(f"platform {pid}: measurement_gates.max_uptime_s is required")
    max_uptime_f = float(cfg["max_uptime_s"])
    max_uptime_cite = cfg.get("max_uptime_s_citation")

    gates: list[GateResult] = []

    if ac_override is not None:
        gates.append(ac_override)
    elif platform_declares_no_battery(repo_root, pid):
        gates.append(
            GateResult(
                name="ac",
                passed=True,
                reason="no_battery_platform_mains_only",
                detail={
                    "battery_present": False,
                    "ac_ok": True,
                    "recorded_reason": (
                        "platform power.has_battery is false; mains-only; "
                        "Win32_Battery was not consulted"
                    ),
                },
            )
        )
    elif skip_host_probes:
        raise RuntimeError("ac probe required unless ac_override supplied")
    else:
        gates.append(probe_ac_gate())

    if processor_override is not None:
        gates.append(processor_override)
    elif skip_host_probes:
        raise RuntimeError("processor probe required unless processor_override supplied")
    else:
        gates.append(probe_processor_ac_gate())

    obs_mb = available_mb
    if obs_mb is None and not skip_host_probes:
        obs_mb = probe_available_mb()
    if obs_mb is None:
        mem_gate = GateResult(
            name="available_mb",
            passed=False,
            reason="available_mb_unreadable",
            detail={
                "floor_mb": floor_f,
                "observed_mb": None,
                "citation": floor_cite,
            },
        )
    else:
        mem_ok = float(obs_mb) >= floor_f
        mem_gate = GateResult(
            name="available_mb",
            passed=mem_ok,
            reason="available_mb_ok" if mem_ok else "available_mb_below_floor",
            detail={
                "floor_mb": floor_f,
                "observed_mb": float(obs_mb),
                "citation": floor_cite,
            },
        )
    gates.append(mem_gate)

    obs_up = uptime_s
    if obs_up is None and not skip_host_probes:
        obs_up = probe_uptime_s()
    observed = None if obs_up is None else float(obs_up)
    if skip_uptime:
        up_gate = GateResult(
            name="uptime",
            passed=True,
            reason="uptime_skipped_noreboot_deviation",
            detail={
                "max_uptime_s": max_uptime_f,
                "observed_uptime_s": observed,
                "citation": max_uptime_cite,
                "deviation": {
                    "kind": "UNCOLD_UPTIME",
                    "uptime_s": observed,
                    "reason": UNCOLD_UPTIME_REASON,
                },
            },
        )
    elif obs_up is None:
        up_gate = GateResult(
            name="uptime",
            passed=False,
            reason="uptime_unreadable",
            detail={
                "max_uptime_s": max_uptime_f,
                "observed_uptime_s": None,
                "citation": max_uptime_cite,
            },
        )
    else:
        up_ok = float(obs_up) < max_uptime_f
        up_gate = GateResult(
            name="uptime",
            passed=up_ok,
            reason="uptime_ok" if up_ok else "uptime_not_cold",
            detail={
                "max_uptime_s": max_uptime_f,
                "observed_uptime_s": float(obs_up),
                "citation": max_uptime_cite,
            },
        )
    gates.append(up_gate)

    # Onset is informational: unknown must appear in the seal, not refuse the session.
    onset_gate = GateResult(
        name="onset",
        passed=True,
        reason="onset_measured" if onset_status == "measured" else "onset_unknown",
        detail={
            "onset_s": onset_s,
            "onset_status": onset_status,
            "onset_citation": onset_cite,
            "seal_note": (
                None
                if onset_status == "measured"
                else (
                    "platform onset_s is unknown; recorded in run_environment; "
                    "canary N must not assume aipc-c1's 657 s"
                )
            ),
        },
    )
    gates.append(onset_gate)

    if process_names is not None or not skip_host_probes:
        try:
            names = list(process_names) if process_names is not None else probe_process_names()
            gates.append(classify_tier1_processes(names))
        except Exception as exc:
            gates.append(
                GateResult(
                    name="tier1_processes",
                    passed=False,
                    reason="tier1_unreadable",
                    detail={"error": f"{type(exc).__name__}: {exc}"},
                )
            )

    refusals = [f"{g.name}:{g.reason}" for g in gates if (not g.passed) and g.name != "onset"]
    if skip_host_probes:
        wsh: dict[str, Any] = {"skipped": True, "reason": "skip_host_probes"}
    else:
        from seam.run_environment import snapshot_workloads_session_host

        wsh = snapshot_workloads_session_host()
    return MeasurementGateReport(
        platform_id=pid,
        gates=gates,
        all_passed=len(refusals) == 0,
        refusal_reasons=refusals,
        workloads_session_host=wsh,
    )


def run_environment_gate_fields(report: MeasurementGateReport) -> dict[str, Any]:
    """Additive INF-5 fields for the seal's run_environment block."""
    by_name = {g.name: g for g in report.gates}
    ac = by_name.get("ac")
    proc = by_name.get("processor_ac")
    mem = by_name.get("available_mb")
    up = by_name.get("uptime")
    onset = by_name.get("onset")
    tier1 = by_name.get("tier1_processes")
    return {
        "measurement_gates": {
            "platform_id": report.platform_id,
            "all_passed": report.all_passed,
            "ac": {
                "passed": None if ac is None else ac.passed,
                "reason": None if ac is None else ac.reason,
                "detail": None if ac is None else dict(ac.detail),
            },
            "processor_ac": {
                "passed": None if proc is None else proc.passed,
                "reason": None if proc is None else proc.reason,
                "scheme_guid": None if proc is None else (proc.detail or {}).get("scheme_guid"),
                "scheme_name": None if proc is None else (proc.detail or {}).get("scheme_name"),
                "procthrottlemin_ac": None
                if proc is None
                else (proc.detail or {}).get("procthrottlemin_ac"),
                "procthrottlemax_ac": None
                if proc is None
                else (proc.detail or {}).get("procthrottlemax_ac"),
            },
            "available_mb": {
                "passed": None if mem is None else mem.passed,
                "floor_mb": None if mem is None else (mem.detail or {}).get("floor_mb"),
                "observed_mb": None if mem is None else (mem.detail or {}).get("observed_mb"),
                "citation": None if mem is None else (mem.detail or {}).get("citation"),
            },
            "uptime": {
                "passed": None if up is None else up.passed,
                "max_uptime_s": None if up is None else (up.detail or {}).get("max_uptime_s"),
                "observed_uptime_s": None
                if up is None
                else (up.detail or {}).get("observed_uptime_s"),
                "citation": None if up is None else (up.detail or {}).get("citation"),
            },
            "onset_s": None if onset is None else (onset.detail or {}).get("onset_s"),
            "onset_status": None if onset is None else (onset.detail or {}).get("onset_status"),
            "onset_citation": None if onset is None else (onset.detail or {}).get("onset_citation"),
            "onset_seal_note": None if onset is None else (onset.detail or {}).get("seal_note"),
            "workloads_session_host": dict(report.workloads_session_host),
            "tier1_processes": {
                "passed": None if tier1 is None else tier1.passed,
                "reason": None if tier1 is None else tier1.reason,
                "resident": (
                    [] if tier1 is None else list((tier1.detail or {}).get("resident") or [])
                ),
            },
        }
    }


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate SEAM measurement gates")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--platform-id", default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--skip-uptime",
        action="store_true",
        help="Skip only the uptime gate and record an UNCOLD_UPTIME deviation.",
    )
    parser.add_argument(
        "--tier1-only",
        action="store_true",
        help="Refuse if a tier-1 process (chrome, Cursor, Code, msedge, ...) is resident.",
    )
    args = parser.parse_args(argv)
    if args.tier1_only:
        try:
            result = classify_tier1_processes(probe_process_names())
        except Exception as exc:
            print(f"REFUSED -- tier-1 unreadable: {type(exc).__name__}: {exc}")
            return 1
        if result.passed:
            print("tier1_clear")
            return 0
        resident = ", ".join(result.detail.get("resident") or [])
        print(f"REFUSED -- tier-1 resident: {resident}")
        return 1
    root = (args.repo_root or Path(__file__).resolve().parents[1]).resolve()
    report = evaluate_measurement_gates(
        root,
        platform_id=args.platform_id,
        skip_uptime=bool(args.skip_uptime),
    )
    payload = report.to_dict()
    payload["run_environment_fields"] = run_environment_gate_fields(report)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"platform_id={report.platform_id} all_passed={report.all_passed}")
        for g in report.gates:
            print(f"  {g.name}: passed={g.passed} reason={g.reason}")
        if report.refusal_reasons:
            print("REFUSALS:", ", ".join(report.refusal_reasons))
    return 0 if report.all_passed else 1


if __name__ == "__main__":
    sys.exit(_main())
