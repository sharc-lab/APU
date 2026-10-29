"""INF-5 run-environment block: host + session fields required on every new seal.

Closes the W-3 to Q-KV trajectory-drop blind spot. Fields that can be read from the
host at emit time are collected here; session-design / prompt-render /
available-MB bookends / WSH-during-run must be supplied by the caller (or
merged from measurement records). Absent or empty values refuse the seal.
Never silently defaulted.

Callers should use :class:`RunEnvironmentSession` (begin at session start, feed
prompt renders, finalize before seal) or stage the session slice on
``summary["run_environment"]`` / ``emit(run_environment=...)``.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from seam.errors import ManifestValidationError
from seam.hashing import sha256_bytes
from seam.jsonlog import log_event

__all__ = [
    "REQUIRED_RUN_ENVIRONMENT_FIELDS",
    "SESSION_DESIGNS",
    "SESSION_FIELD_NAMES",
    "WSH_PROCESS_NAME",
    "RunEnvironmentSession",
    "capture_host_run_environment",
    "lift_session_fields",
    "merge_run_environment",
    "probe_workloads_session_host",
    "require_run_environment",
    "session_fields",
    "sha256_prompt_render",
    "snapshot_workloads_session_host",
]

#: Canonical process name (normalized) for Copilot+ WorkloadsSessionHost.
WSH_PROCESS_NAME: Final = "workloadssessionhost.exe"

SESSION_DESIGNS: Final = frozenset({"sequential", "interleaved"})

#: Every key must be present and non-empty on a ``spec_version=1.1`` seal.
REQUIRED_RUN_ENVIRONMENT_FIELDS: Final = (
    "gpu_driver_version",
    "windows_build",
    "active_power_scheme_guid",
    "pip_freeze_sha256",
    "tokenizers_version",
    "prompt_render_sha256",
    "available_mb_start",
    "available_mb_end",
    "workloads_session_host_resident",
    "session_design",
    "arm_order",
)

#: Session slice that callers (or :class:`RunEnvironmentSession`) must supply.
SESSION_FIELD_NAMES: Final = (
    "prompt_render_sha256",
    "available_mb_start",
    "session_design",
    "arm_order",
)

_TIMEOUT_S: Final = 60
_GUID_RE: Final = re.compile(
    r"GUID:\s*([0-9a-fA-F-]{36})\s*\(",
    re.MULTILINE,
)
_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")


def sha256_prompt_render(rendered: str | bytes) -> str:
    """SHA-256 of the exact rendered prompt bytes used for the run.

    Callers must pass the same bytes/string the model consumed. Do not re-render
    on a second path. Strings are hashed as UTF-8.
    """
    data = rendered if isinstance(rendered, bytes) else rendered.encode("utf-8")
    return sha256_bytes(data)


def session_fields(
    *,
    prompt_render_sha256: str,
    available_mb_start: float,
    session_design: str,
    arm_order: list[str],
    available_mb_end: float | None = None,
    workloads_session_host_resident: bool | None = None,
) -> dict[str, Any]:
    """Build the caller-supplied INF-5 session slice for staging into emit.

    Does not invent values. Every argument is measured. Optional end/WSH keys are
    included only when supplied (host capture fills them at seal when omitted).
    """
    out: dict[str, Any] = {
        "prompt_render_sha256": str(prompt_render_sha256).lower(),
        "available_mb_start": float(available_mb_start),
        "session_design": str(session_design),
        "arm_order": [str(item) for item in arm_order],
    }
    if available_mb_end is not None:
        out["available_mb_end"] = float(available_mb_end)
    if workloads_session_host_resident is not None:
        out["workloads_session_host_resident"] = bool(workloads_session_host_resident)
    return out


def _as_available_mb(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, Mapping):
        inner = value.get("available_mb")
        if isinstance(inner, (int, float)) and not isinstance(inner, bool):
            return float(inner)
    return None


def lift_session_fields(mapping: Mapping[str, Any] | None) -> dict[str, Any]:
    """Pull INF-5 session keys from a summary/plan-like mapping. No invention.

    Recognizes a nested ``run_environment`` block and common top-level aliases used
    by quality / BFCL runners (``available_mb_start`` as float or probe dict,
    ``session_design``, ``arm_order`` / ``arms``, ``prompt_render_sha256``).
    """
    if not mapping:
        return {}
    out: dict[str, Any] = {}
    nested = mapping.get("run_environment")
    if isinstance(nested, Mapping):
        for key, value in nested.items():
            if not _is_absent(value):
                out[key] = value

    for key in SESSION_FIELD_NAMES:
        if key in out:
            continue
        if key == "available_mb_start":
            mb = _as_available_mb(mapping.get("available_mb_start"))
            if mb is not None:
                out["available_mb_start"] = mb
            continue
        if key == "arm_order":
            order = mapping.get("arm_order")
            if order is None:
                order = mapping.get("arms")
            if (
                isinstance(order, list)
                and order
                and all(isinstance(item, str) and item.strip() for item in order)
            ):
                out["arm_order"] = list(order)
            continue
        value = mapping.get(key)
        if not _is_absent(value):
            out[key] = value

    mb_end = _as_available_mb(mapping.get("available_mb_end"))
    if mb_end is not None and "available_mb_end" not in out:
        out["available_mb_end"] = mb_end
    if "workloads_session_host_resident" not in out and isinstance(
        mapping.get("workloads_session_host_resident"), bool
    ):
        out["workloads_session_host_resident"] = mapping["workloads_session_host_resident"]
    return out


@dataclass
class RunEnvironmentSession:
    """Collect INF-5 session fields during a live measurement session.

    Call :meth:`begin` before work starts, :meth:`add_prompt_render` (or
    :meth:`add_prompt_digest`) for every rendered prompt the model consumed, then
    :meth:`finalize` before staging into ``emit`` / summary.
    """

    available_mb_start: float
    session_design: str
    arm_order: list[str]
    _prompt_hasher: Any = field(repr=False)
    _prompt_updates: int = 0
    _wsh_seen: bool | None = None
    available_mb_start_method: str | None = None

    @classmethod
    def begin(
        cls,
        *,
        session_design: str,
        arm_order: list[str],
        available_mb_start: float | None = None,
        available_mb_start_method: str | None = None,
    ) -> RunEnvironmentSession:
        if session_design not in SESSION_DESIGNS:
            raise ManifestValidationError(
                f"INF-5 session_design must be one of {sorted(SESSION_DESIGNS)}; "
                f"got {session_design!r}"
            )
        if not arm_order or not all(isinstance(a, str) and a.strip() for a in arm_order):
            raise ManifestValidationError(
                "INF-5 arm_order must be a non-empty list[str] at session begin"
            )
        method = available_mb_start_method
        mb = available_mb_start
        if mb is None:
            from seam.telemetry.host_environment import available_mb_now

            mb, method = available_mb_now()
        if mb is None:
            raise ManifestValidationError(
                "INF-5 available_mb_start unreadable at session begin; refusing to proceed"
            )
        wsh = probe_workloads_session_host()
        resident = wsh.get("resident")
        return cls(
            available_mb_start=float(mb),
            session_design=session_design,
            arm_order=[str(a) for a in arm_order],
            _prompt_hasher=hashlib.sha256(),
            _wsh_seen=bool(resident) if isinstance(resident, bool) else None,
            available_mb_start_method=method,
        )

    def prompt_update_count(self) -> int:
        return self._prompt_updates

    def add_prompt_render(self, rendered: str | bytes) -> None:
        """Fold exact rendered prompt bytes into the session digest (null-separated)."""
        data = rendered if isinstance(rendered, bytes) else rendered.encode("utf-8")
        self._prompt_hasher.update(b"\0")
        self._prompt_hasher.update(data)
        self._prompt_updates += 1

    def add_prompt_digest(self, digest_hex: str) -> None:
        """Fold a per-entry / per-cell prompt SHA-256 hex digest into the session digest."""
        text = str(digest_hex).strip().lower()
        if not _SHA256_RE.fullmatch(text):
            raise ManifestValidationError(
                f"INF-5 add_prompt_digest expects sha256 hex; got {digest_hex!r}"
            )
        self._prompt_hasher.update(bytes.fromhex(text))
        self._prompt_updates += 1

    def note_wsh_resident(self, resident: bool) -> None:
        """OR mid-run WorkloadsSessionHost observations into the session flag."""
        if self._wsh_seen is None:
            self._wsh_seen = bool(resident)
        else:
            self._wsh_seen = bool(self._wsh_seen or resident)

    def finalize(self) -> dict[str, Any]:
        """Return the session slice for ``emit(run_environment=...)`` / summary staging.

        Captures ``available_mb_end`` and a final WSH probe. Requires at least one
        prompt-render update. Empty digests are refused (no silent empty-hash).
        """
        if self._prompt_updates < 1:
            raise ManifestValidationError(
                "INF-5 RunEnvironmentSession.finalize: no prompt renders recorded; "
                "refusing empty prompt_render_sha256"
            )
        from seam.telemetry.host_environment import available_mb_now

        mb_end, _method = available_mb_now()
        if mb_end is None:
            raise ManifestValidationError(
                "INF-5 available_mb_end unreadable at session finalize; refusing to seal"
            )
        wsh = probe_workloads_session_host()
        end_resident = wsh.get("resident")
        if isinstance(end_resident, bool):
            self.note_wsh_resident(end_resident)
        if self._wsh_seen is None:
            raise ManifestValidationError(
                "INF-5 workloads_session_host_resident unknown after finalize probes"
            )
        return session_fields(
            prompt_render_sha256=self._prompt_hasher.hexdigest(),
            available_mb_start=self.available_mb_start,
            session_design=self.session_design,
            arm_order=list(self.arm_order),
            available_mb_end=float(mb_end),
            workloads_session_host_resident=bool(self._wsh_seen),
        )


def _capture_gpu_driver_version() -> str | None:
    """Intel / primary display driver version via Win32_VideoController."""
    if sys.platform != "win32":
        return None
    try:
        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                (
                    "Get-CimInstance Win32_VideoController | "
                    "Select-Object -ExpandProperty DriverVersion"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_S,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        log_event(
            "run_environment.gpu_driver_failed",
            severity="warning",
            message="could not query Win32_VideoController.DriverVersion",
            error=str(exc),
        )
        return None
    if completed.returncode != 0:
        log_event(
            "run_environment.gpu_driver_nonzero",
            severity="warning",
            message="Win32_VideoController query exited non-zero",
            returncode=completed.returncode,
            stderr=completed.stderr.strip(),
        )
        return None
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        return None
    # Prefer the first non-empty version; Platform A is unified iGPU (no dGPU).
    return lines[0]


def _capture_windows_build() -> str | None:
    """Measured Windows build string (not the config ``platform.os_build`` pin)."""
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Windows NT\CurrentVersion",
            ) as key:
                current_build, _ = winreg.QueryValueEx(key, "CurrentBuild")
                try:
                    ubr, _ = winreg.QueryValueEx(key, "UBR")
                    return f"{current_build}.{ubr}"
                except OSError:
                    return str(current_build)
        except OSError as exc:
            log_event(
                "run_environment.windows_build_registry_failed",
                severity="warning",
                message="registry CurrentBuild/UBR unavailable; falling back to platform.version",
                error=str(exc),
            )
    version = sys.platform and __import__("platform").version()
    text = str(version).strip() if version else ""
    return text or None


def _capture_active_power_scheme_guid() -> str | None:
    """Active power scheme GUID from ``powercfg /getactivescheme``."""
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
            "run_environment.powercfg_failed",
            severity="warning",
            message="could not run powercfg /getactivescheme",
            error=str(exc),
        )
        return None
    if completed.returncode != 0:
        return None
    match = _GUID_RE.search(completed.stdout)
    if match is None:
        return None
    return match.group(1).lower()


def _capture_pip_freeze_sha256() -> str | None:
    """SHA-256 of full ``pip freeze`` output.

    Canonical form: UTF-8, LF newlines, lines stripped, **sorted** ascending, trailing
    newline. Sorting makes the digest independent of freeze emission order while still
    covering the full environment (not a curated subset).
    """
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_S,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        log_event(
            "run_environment.pip_freeze_failed",
            severity="warning",
            message="pip freeze failed",
            error=str(exc),
        )
        return None
    if completed.returncode != 0:
        log_event(
            "run_environment.pip_freeze_nonzero",
            severity="warning",
            message="pip freeze exited non-zero",
            returncode=completed.returncode,
            stderr=completed.stderr.strip(),
        )
        return None
    lines = sorted(line.strip() for line in completed.stdout.splitlines() if line.strip())
    canonical = ("\n".join(lines) + "\n").encode("utf-8")
    return sha256_bytes(canonical)


def _capture_tokenizers_version() -> str | None:
    try:
        return importlib.metadata.version("tokenizers")
    except importlib.metadata.PackageNotFoundError:
        log_event(
            "run_environment.tokenizers_missing",
            severity="warning",
            message="tokenizers package not installed; cannot record version",
        )
        return None


def snapshot_workloads_session_host() -> dict[str, Any]:
    """Instance count, pids, working set MB, and CPU seconds.

    Lists are aligned with ``pids``. A read failure is recorded and does not
    drop the pid.
    """
    empty: dict[str, Any] = {
        "process": "WorkloadsSessionHost",
        "instance_count": 0,
        "pids": [],
        "WS_MB": [],
        "CPU_s": [],
        "read_errors": [],
    }
    try:
        import psutil
    except ImportError as exc:
        empty["read_errors"] = [f"psutil unavailable: {exc}"]
        return empty

    from seam.isolation import _normalize_process_name

    pids: list[int] = []
    ws_mb: list[float | None] = []
    cpu_s: list[float | None] = []
    errors: list[str] = []
    try:
        processes = list(psutil.process_iter(["pid", "name"]))
    except Exception as exc:  # pragma: no cover - platform-dependent
        empty["read_errors"] = [f"{type(exc).__name__}: {exc}"]
        return empty
    for process in processes:
        try:
            name = str(process.info.get("name") or "")
            if _normalize_process_name(name) != WSH_PROCESS_NAME:
                continue
            pid = int(process.info["pid"])
        except (psutil.Error, TypeError, ValueError) as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            continue
        pids.append(pid)
        try:
            mem = process.memory_info()
            wset = getattr(mem, "wset", None)
            if wset is None:
                wset = getattr(mem, "rss", None)
            ws_mb.append(None if wset is None else float(wset) / (1024.0 * 1024.0))
        except psutil.Error as exc:
            ws_mb.append(None)
            errors.append(f"pid {pid} WS_MB: {exc}")
        try:
            times = process.cpu_times()
            cpu_s.append(float(times.user) + float(times.system))
        except psutil.Error as exc:
            cpu_s.append(None)
            errors.append(f"pid {pid} CPU_s: {exc}")
    order = sorted(range(len(pids)), key=lambda i: pids[i])
    return {
        "process": "WorkloadsSessionHost",
        "instance_count": len(pids),
        "pids": [pids[i] for i in order],
        "WS_MB": [ws_mb[i] for i in order],
        "CPU_s": [cpu_s[i] for i in order],
        "read_errors": errors,
    }


def probe_workloads_session_host() -> dict[str, Any]:
    """One-shot WSH residency probe (psutil)."""
    from seam.isolation import _normalize_process_name

    try:
        import psutil
    except ImportError as exc:
        return {
            "resident": None,
            "pids": [],
            "private_working_set_bytes_total": None,
            "probe_error": f"psutil unavailable: {exc}",
        }

    pids: list[int] = []
    private_total = 0
    try:
        for process in psutil.process_iter(["pid", "name", "memory_info"]):
            try:
                name = str(process.info.get("name") or "")
                if _normalize_process_name(name) != WSH_PROCESS_NAME:
                    continue
                pids.append(int(process.info["pid"]))
                info = process.info.get("memory_info")
                if info is not None:
                    private = getattr(info, "private", None)
                    if private is None:
                        private = getattr(info, "rss", None)
                    if private is not None:
                        private_total += int(private)
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
    except Exception as exc:  # pragma: no cover - platform-dependent
        return {
            "resident": None,
            "pids": [],
            "private_working_set_bytes_total": None,
            "probe_error": f"{type(exc).__name__}:{exc}",
        }

    pids.sort()
    return {
        "resident": bool(pids),
        "pids": pids,
        "private_working_set_bytes_total": private_total if pids else 0,
        "probe_error": None,
    }


def capture_host_run_environment(
    *,
    include_measurement_gates: bool = False,
    platform_id: str | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Collect host-readable INF-5 fields. Session fields stay unset (caller supplies).

    When ``include_measurement_gates`` is true, evaluate PORT-2 gates and attach the
    additive ``measurement_gates`` object (records unknown onset; does not invent values).
    """
    from seam.telemetry.host_environment import available_mb_now

    available, available_method = available_mb_now()
    wsh = probe_workloads_session_host()
    out: dict[str, Any] = {
        "gpu_driver_version": _capture_gpu_driver_version(),
        "windows_build": _capture_windows_build(),
        "active_power_scheme_guid": _capture_active_power_scheme_guid(),
        "pip_freeze_sha256": _capture_pip_freeze_sha256(),
        "tokenizers_version": _capture_tokenizers_version(),
        # End-of-run Available when caller did not bookend; start must still be supplied.
        "available_mb_end": available,
        "available_mb_method": available_method,
        "workloads_session_host_resident": wsh.get("resident"),
        "workloads_session_host_probe": wsh,
    }
    if include_measurement_gates:
        from seam.measurement_gates import (
            evaluate_measurement_gates,
            run_environment_gate_fields,
        )

        root = repo_root or Path(__file__).resolve().parents[1]
        report = evaluate_measurement_gates(root, platform_id=platform_id)
        out.update(run_environment_gate_fields(report))
    return out


def _is_absent(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return isinstance(value, (list, tuple)) and len(value) == 0


def merge_run_environment(
    supplied: dict[str, Any] | None,
    *,
    capture_host: bool = True,
) -> dict[str, Any]:
    """Merge caller-supplied fields over optional host capture.

    Caller values always win. Host capture fills only keys the caller left absent.
    """
    merged: dict[str, Any] = {}
    if capture_host:
        merged.update(capture_host_run_environment())
    if supplied:
        for key, value in supplied.items():
            if not _is_absent(value) or key not in merged:
                merged[key] = value
    return merged


def require_run_environment(env: dict[str, Any] | None) -> dict[str, Any]:
    """Hard-refuse if any INF-5 required field is absent, null, or empty.

    Raises:
        ManifestValidationError: Naming every missing field. No silent defaults.
    """
    if not env:
        raise ManifestValidationError(
            "INF-5 run_environment is required for spec_version 1.1 seals; "
            "refusing to seal with run_environment absent"
        )

    missing: list[str] = []
    for key in REQUIRED_RUN_ENVIRONMENT_FIELDS:
        if key not in env or _is_absent(env.get(key)):
            missing.append(key)

    design = env.get("session_design")
    if design is not None and design not in SESSION_DESIGNS:
        missing.append(f"session_design(not in {sorted(SESSION_DESIGNS)})")

    prompt_hash = env.get("prompt_render_sha256")
    if isinstance(prompt_hash, str) and prompt_hash and not _SHA256_RE.fullmatch(prompt_hash):
        missing.append("prompt_render_sha256(not sha256 hex)")

    pip_hash = env.get("pip_freeze_sha256")
    if isinstance(pip_hash, str) and pip_hash and not _SHA256_RE.fullmatch(pip_hash):
        missing.append("pip_freeze_sha256(not sha256 hex)")

    arm_order = env.get("arm_order")
    if arm_order is not None and (
        not isinstance(arm_order, list)
        or not all(isinstance(item, str) and item.strip() for item in arm_order)
    ):
        missing.append("arm_order(must be non-empty list[str])")

    resident = env.get("workloads_session_host_resident")
    if resident is not None and not isinstance(resident, bool):
        missing.append("workloads_session_host_resident(must be bool)")

    for mb_key in ("available_mb_start", "available_mb_end"):
        mb_val = env.get(mb_key)
        if mb_val is not None and not isinstance(mb_val, (int, float)):
            missing.append(f"{mb_key}(must be number)")

    if missing:
        raise ManifestValidationError(
            "INF-5 refuse seal: run_environment missing or invalid field(s): "
            + ", ".join(missing)
            + ". No silent defaults. Supply measured values or stop."
        )

    out = {
        "gpu_driver_version": str(env["gpu_driver_version"]),
        "windows_build": str(env["windows_build"]),
        "active_power_scheme_guid": str(env["active_power_scheme_guid"]).lower(),
        "pip_freeze_sha256": str(env["pip_freeze_sha256"]).lower(),
        "tokenizers_version": str(env["tokenizers_version"]),
        "prompt_render_sha256": str(env["prompt_render_sha256"]).lower(),
        "available_mb_start": float(env["available_mb_start"]),
        "available_mb_end": float(env["available_mb_end"]),
        "workloads_session_host_resident": bool(env["workloads_session_host_resident"]),
        "session_design": str(env["session_design"]),
        "arm_order": [str(item) for item in env["arm_order"]],
    }
    # PORT-2 additive: preserve measurement_gates when present (never invent).
    if "measurement_gates" in env and isinstance(env["measurement_gates"], dict):
        out["measurement_gates"] = dict(env["measurement_gates"])
    return out
