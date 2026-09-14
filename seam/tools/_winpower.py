"""Windows power-request assertion (PowerCreateRequest / PowerSetRequest).

Declares that the process is working so the OS should not enter Modern Standby /
DRIPS for the life of the request. This is not a power-policy change: idle
timeout and the active power scheme are untouched.

Prefer PowerCreateRequest + PowerSetRequest(PowerRequestSystemRequired) over
SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED); the latter is the
legacy path and is less reliable against Modern Standby DRIPS.

ctypes argtypes/restype are mandatory. GetCurrentProcess-style HANDLE overflow
already burned a measurement path that lacked them.
"""

from __future__ import annotations

import ctypes
import platform
from ctypes import wintypes
from typing import Any

__all__ = [
    "POWER_REQUEST_SYSTEM_REQUIRED",
    "PowerRequest",
    "assert_system_required",
    "system_required",
]

POWER_REQUEST_CONTEXT_VERSION = 0
POWER_REQUEST_CONTEXT_SIMPLE_STRING = 0x1
POWER_REQUEST_SYSTEM_REQUIRED = 1  # POWER_REQUEST_TYPE.PowerRequestSystemRequired
_INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value


class _REASON_CONTEXT(ctypes.Structure):
    """REASON_CONTEXT with the SIMPLE_STRING form of the Reason union.

    Layout matches winbase.h when Flags == POWER_REQUEST_CONTEXT_SIMPLE_STRING:
    Version, Flags, then LPWSTR SimpleReasonString.
    """

    _fields_ = [
        ("Version", wintypes.ULONG),
        ("Flags", wintypes.DWORD),
        ("SimpleReasonString", wintypes.LPWSTR),
    ]


def _kernel32() -> Any:
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.PowerCreateRequest.argtypes = [ctypes.POINTER(_REASON_CONTEXT)]
    k32.PowerCreateRequest.restype = wintypes.HANDLE
    k32.PowerSetRequest.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k32.PowerSetRequest.restype = wintypes.BOOL
    k32.PowerClearRequest.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k32.PowerClearRequest.restype = wintypes.BOOL
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    k32.CloseHandle.restype = wintypes.BOOL
    return k32


class PowerRequest:
    """One PowerCreateRequest handle with an active PowerRequestSystemRequired."""

    def __init__(self, *, reason: str, role: str) -> None:
        self.reason = reason
        self.role = role
        self._handle: int | None = None
        self._set = False
        self.record: dict[str, Any] = {
            "mechanism": "PowerCreateRequest/PowerSetRequest",
            "request_type": "PowerRequestSystemRequired",
            "request_type_id": POWER_REQUEST_SYSTEM_REQUIRED,
            "reason": reason,
            "role": role,
            "platform": platform.system(),
            "created": False,
            "set": False,
            "cleared": False,
            "closed": False,
            "succeeded": False,
            "handle_repr": None,
            "create_last_error": None,
            "set_last_error": None,
            "clear_last_error": None,
            "close_last_error": None,
            "error": None,
        }

    def acquire(self) -> dict[str, Any]:
        if platform.system() != "Windows":
            self.record["error"] = "not Windows; power request unavailable"
            return self.record
        try:
            k32 = _kernel32()
            ctx = _REASON_CONTEXT()
            ctx.Version = POWER_REQUEST_CONTEXT_VERSION
            ctx.Flags = POWER_REQUEST_CONTEXT_SIMPLE_STRING
            # Keep a Python reference so the buffer outlives the API call and the
            # request object's lifetime (the API may retain the pointer).
            self._reason_buf = ctypes.create_unicode_buffer(self.reason)
            ctx.SimpleReasonString = ctypes.cast(self._reason_buf, wintypes.LPWSTR)
            ctypes.set_last_error(0)
            handle = k32.PowerCreateRequest(ctypes.byref(ctx))
            create_err = ctypes.get_last_error()
            handle_int = int(ctypes.cast(handle, ctypes.c_void_p).value or 0)
            self.record["handle_repr"] = hex(handle_int) if handle_int else None
            if handle is None or handle_int == 0 or handle_int == _INVALID_HANDLE_VALUE:
                self.record["create_last_error"] = create_err
                self.record["error"] = f"PowerCreateRequest failed last_error={create_err}"
                return self.record
            self._handle = handle_int
            self.record["created"] = True
            ctypes.set_last_error(0)
            ok = bool(k32.PowerSetRequest(handle, POWER_REQUEST_SYSTEM_REQUIRED))
            set_err = ctypes.get_last_error()
            self.record["set_last_error"] = 0 if ok else set_err
            if not ok:
                self.record["error"] = f"PowerSetRequest failed last_error={set_err}"
                self._close_handle(k32)
                return self.record
            self._set = True
            self.record["set"] = True
            self.record["succeeded"] = True
            return self.record
        except Exception as exc:
            self.record["error"] = f"{type(exc).__name__}: {exc}"
            return self.record

    def release(self) -> dict[str, Any]:
        if platform.system() != "Windows" or self._handle is None:
            return self.record
        try:
            k32 = _kernel32()
            handle = wintypes.HANDLE(self._handle)
            if self._set:
                ctypes.set_last_error(0)
                ok = bool(k32.PowerClearRequest(handle, POWER_REQUEST_SYSTEM_REQUIRED))
                clear_err = ctypes.get_last_error()
                self.record["clear_last_error"] = 0 if ok else clear_err
                self.record["cleared"] = bool(ok)
                if not ok and self.record.get("error") is None:
                    self.record["error"] = f"PowerClearRequest failed last_error={clear_err}"
                self._set = False
            self._close_handle(k32)
        except Exception as exc:
            if self.record.get("error") is None:
                self.record["error"] = f"release:{type(exc).__name__}: {exc}"
        return self.record

    def _close_handle(self, k32: Any) -> None:
        if self._handle is None:
            return
        handle = wintypes.HANDLE(self._handle)
        ctypes.set_last_error(0)
        ok = bool(k32.CloseHandle(handle))
        close_err = ctypes.get_last_error()
        self.record["close_last_error"] = 0 if ok else close_err
        self.record["closed"] = bool(ok)
        self._handle = None


class system_required:
    """Context manager: hold PowerRequestSystemRequired for a scope."""

    def __init__(self, *, reason: str, role: str) -> None:
        self._req = PowerRequest(reason=reason, role=role)

    def __enter__(self) -> dict[str, Any]:
        return self._req.acquire()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._req.release()


def assert_system_required(*, reason: str, role: str) -> PowerRequest:
    """Create and set a system-required request; caller must ``release()``."""
    req = PowerRequest(reason=reason, role=role)
    req.acquire()
    return req
