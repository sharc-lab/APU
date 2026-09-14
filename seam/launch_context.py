"""Launch context: how a measurement process was started, and where it landed.

``isolation_mode`` records whether the machine was closed down. That is necessary but not
sufficient: a remote declaration can still mean an SSH foreground shell (session 0, holding the
session open) or a WMI-detached child (session 0, Service-0x0 window station, parented to
WmiPrvSE). Those are not the same measurement condition, and ΔN uses only the detached path.

Fields recorded on every acceptance / ΔN manifest:

``launch_context``
    Operator-declared label of the launch path. One of ``ssh_detached``, ``ssh_foreground``,
    ``local_console``. Declared via ``SEAM_LAUNCH_CONTEXT`` or an explicit argument - never
    inferred from the process tree, because inference is exactly how a Cursor-session run would
    be mislabelled as detached.

``session_id``
    Windows session of this process (``ProcessIdToSessionId``). Detached WMI children land in 0.

``window_station``
    Name of this process's window station. ``Service-0x0-...`` means session 0 with no desktop;
    ``WinSta0`` is the interactive desktop.

Results whose ``launch_context`` differs are never pooled with each other, for the same reason
``isolation_mode`` is never pooled across modes.
"""

from __future__ import annotations

import ctypes
import os
from ctypes import wintypes
from typing import Any, Final

from seam.errors import SeamError
from seam.jsonlog import log_event

__all__ = [
    "ENV_VAR",
    "LAUNCH_CONTEXTS",
    "LaunchContextError",
    "assert_same_launch_context",
    "capture_process_placement",
    "resolve_launch_context",
]

LAUNCH_CONTEXTS: Final = ("ssh_detached", "ssh_foreground", "local_console")
ENV_VAR: Final = "SEAM_LAUNCH_CONTEXT"

UOI_NAME = 2


class LaunchContextError(SeamError):
    """The launch context was undeclared, invalid, or spanned across compared runs."""


def _session_id(pid: int) -> int | None:
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.ProcessIdToSessionId.argtypes = [
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.ProcessIdToSessionId.restype = wintypes.BOOL
        session = wintypes.DWORD()
        if kernel32.ProcessIdToSessionId(wintypes.DWORD(pid), ctypes.byref(session)):
            return int(session.value)
    except Exception:
        return None
    return None


def _window_station() -> tuple[str | None, str | None]:
    """Return ``(name, error)``. A null name with a null error means the probe was skipped."""
    if os.name != "nt":
        return None, "window_station probe is Windows-only"
    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.GetProcessWindowStation.restype = wintypes.HANDLE
        user32.GetProcessWindowStation.argtypes = []
        user32.GetUserObjectInformationW.restype = wintypes.BOOL
        user32.GetUserObjectInformationW.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        handle = user32.GetProcessWindowStation()
        if not handle:
            return (
                None,
                f"GetProcessWindowStation returned NULL, GetLastError={ctypes.get_last_error()}",
            )
        needed = wintypes.DWORD()
        buffer = ctypes.create_unicode_buffer(256)
        if user32.GetUserObjectInformationW(
            handle,
            UOI_NAME,
            ctypes.cast(buffer, ctypes.c_void_p),
            ctypes.sizeof(buffer),
            ctypes.byref(needed),
        ):
            return str(buffer.value), None
        return None, (
            f"GetUserObjectInformationW(UOI_NAME) failed, GetLastError={ctypes.get_last_error()}"
        )
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def capture_process_placement() -> dict[str, Any]:
    """Measure where this process landed. Never raises: a failed probe is recorded as such."""
    pid = os.getpid()
    station, station_error = _window_station()
    return {
        "process_id": pid,
        "session_id": _session_id(pid),
        "window_station": station,
        "window_station_error": station_error,
    }


def resolve_launch_context(
    declared: str | None = None,
    *,
    required: bool = False,
) -> tuple[str | None, dict[str, Any]]:
    """Resolve the declared launch context and capture process placement.

    Args:
        declared: Explicit declaration. Falls back to ``SEAM_LAUNCH_CONTEXT``.
        required: When True, refuse an undeclared context. Acceptance / ΔN require it; older
            call sites that predate the field may leave it null so their manifests stay valid.
    """
    value = declared if declared is not None else os.environ.get(ENV_VAR)
    placement = capture_process_placement()
    if value is None or not str(value).strip():
        if required:
            raise LaunchContextError(
                "launch_context is not declared. Acceptance and ΔN record whether the process "
                f"was launched via ssh_detached, ssh_foreground, or local_console. Set {ENV_VAR} "
                "or pass the context explicitly. There is no default: a defaulted context would "
                "mislabel a Cursor-session or SSH-foreground run as the detached path ΔN uses."
            )
        return None, placement

    context = str(value).strip().lower()
    if context not in LAUNCH_CONTEXTS:
        raise LaunchContextError(
            f"launch_context must be one of {list(LAUNCH_CONTEXTS)}, got {value!r}"
        )

    log_event(
        "launch_context.resolved",
        message=f"launch_context={context}",
        launch_context=context,
        session_id=placement.get("session_id"),
        window_station=placement.get("window_station"),
    )
    return context, placement


def assert_same_launch_context(manifests: list[dict[str, Any]]) -> str | None:
    """Refuse to compare runs whose launch_context differs (or is only present on some)."""
    seen: dict[str, list[str]] = {}
    missing: list[str] = []
    for manifest in manifests:
        run_id = str(manifest.get("run_id", "<unknown>"))
        context = manifest.get("launch_context")
        if context is None:
            missing.append(run_id)
            continue
        seen.setdefault(str(context), []).append(run_id)

    if missing and seen:
        raise LaunchContextError(
            "refusing to compare runs when some record launch_context and others do not "
            f"(missing on: {', '.join(missing)}). An unknown launch path is exactly what "
            "cannot be pooled with a known one."
        )
    if not seen:
        return None
    if len(seen) > 1:
        rendered = "; ".join(f"{ctx}: {', '.join(runs)}" for ctx, runs in sorted(seen.items()))
        raise LaunchContextError(
            "refusing to compare runs measured under different launch_context values "
            f"({rendered}). ssh_detached, ssh_foreground and local_console are not the same "
            "measurement condition."
        )
    return next(iter(seen))
