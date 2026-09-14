"""Working-set lock and reserved-CPU-set readback (Windows).

Phase 3 attempted ``SetProcessWorkingSetSize`` and got Win32 error 6, ERROR_INVALID_HANDLE.
That is a code defect rather than a permissions wall: the call needs a real process handle from
``GetCurrentProcess()``, the ``SeIncreaseWorkingSetPrivilege`` privilege enabled on the process
token, and ``SetProcessWorkingSetSizeEx`` with ``QUOTA_LIMITS_HARDWS_MIN_ENABLE`` - the plain
``SetProcessWorkingSetSize`` entry point takes no flags and therefore cannot express a hard
minimum at all.

If the hard minimum is granted, the OS cannot trim the resident model mid-run, which removes a
confound that would otherwise appear as context capacity. Granted or denied is recorded either
way; a denial is a platform fact worth the same as a grant.

Reserved CPU sets are **read and recorded, never set.** Machine-level isolation belongs to the
operator, and a harness that quietly reconfigures the machine invalidates the manifest's claim
to describe the conditions the measurement actually ran under.
"""

from __future__ import annotations

import ctypes
import platform
from ctypes import wintypes
from typing import Any

__all__ = [
    "QUOTA_LIMITS_HARDWS_MIN_ENABLE",
    "enable_working_set_privilege",
    "lock_working_set",
    "read_reserved_cpu_sets",
    "read_working_set_limits",
]

QUOTA_LIMITS_HARDWS_MIN_ENABLE = 0x00000001
QUOTA_LIMITS_HARDWS_MAX_DISABLE = 0x00000008

_TOKEN_ADJUST_PRIVILEGES = 0x0020
_TOKEN_QUERY = 0x0008
_SE_PRIVILEGE_ENABLED = 0x00000002
_ERROR_NOT_ALL_ASSIGNED = 1300
_SE_INC_WORKING_SET_NAME = "SeIncreaseWorkingSetPrivilege"


class _LUID(ctypes.Structure):
    _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]


class _LUID_AND_ATTRIBUTES(ctypes.Structure):
    _fields_ = [("Luid", _LUID), ("Attributes", wintypes.DWORD)]


class _TOKEN_PRIVILEGES(ctypes.Structure):
    _fields_ = [("PrivilegeCount", wintypes.DWORD), ("Privileges", _LUID_AND_ATTRIBUTES * 1)]


def _unsupported(reason: str) -> dict[str, Any]:
    return {"attempted": False, "granted": False, "reason": reason}


def enable_working_set_privilege() -> dict[str, Any]:
    """Enable SeIncreaseWorkingSetPrivilege on the current process token."""
    if platform.system() != "Windows":
        return _unsupported(f"not Windows: {platform.system()}")

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetCurrentProcess.argtypes = []
    advapi32.OpenProcessToken.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
    ]
    advapi32.OpenProcessToken.restype = wintypes.BOOL
    advapi32.LookupPrivilegeValueW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        ctypes.POINTER(_LUID),
    ]
    advapi32.LookupPrivilegeValueW.restype = wintypes.BOOL
    advapi32.AdjustTokenPrivileges.argtypes = [
        wintypes.HANDLE,
        wintypes.BOOL,
        ctypes.POINTER(_TOKEN_PRIVILEGES),
        wintypes.DWORD,
        ctypes.POINTER(_TOKEN_PRIVILEGES),
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi32.AdjustTokenPrivileges.restype = wintypes.BOOL

    handle = kernel32.GetCurrentProcess()

    token = wintypes.HANDLE()
    if not advapi32.OpenProcessToken(
        handle, _TOKEN_ADJUST_PRIVILEGES | _TOKEN_QUERY, ctypes.byref(token)
    ):
        return {
            "attempted": True,
            "granted": False,
            "step": "OpenProcessToken",
            "last_error": ctypes.get_last_error(),
        }

    luid = _LUID()
    if not advapi32.LookupPrivilegeValueW(None, _SE_INC_WORKING_SET_NAME, ctypes.byref(luid)):
        return {
            "attempted": True,
            "granted": False,
            "step": "LookupPrivilegeValueW",
            "last_error": ctypes.get_last_error(),
        }

    privileges = _TOKEN_PRIVILEGES()
    privileges.PrivilegeCount = 1
    privileges.Privileges[0].Luid = luid
    privileges.Privileges[0].Attributes = _SE_PRIVILEGE_ENABLED

    ctypes.set_last_error(0)
    ok = advapi32.AdjustTokenPrivileges(token, False, ctypes.byref(privileges), 0, None, None)
    last_error = ctypes.get_last_error()
    # AdjustTokenPrivileges reports success even when it assigned nothing; the distinction is
    # only visible in GetLastError.
    granted = bool(ok) and last_error != _ERROR_NOT_ALL_ASSIGNED
    return {
        "attempted": True,
        "granted": granted,
        "step": "AdjustTokenPrivileges",
        "privilege": _SE_INC_WORKING_SET_NAME,
        "last_error": last_error,
        "last_error_meaning": (
            "ERROR_NOT_ALL_ASSIGNED: the account does not hold this privilege"
            if last_error == _ERROR_NOT_ALL_ASSIGNED
            else "ERROR_SUCCESS"
            if last_error == 0
            else f"win32 error {last_error}"
        ),
    }


def read_working_set_limits() -> dict[str, Any]:
    """Read back the current working-set minimum, maximum, and quota flags."""
    if platform.system() != "Windows":
        return _unsupported(f"not Windows: {platform.system()}")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetProcessWorkingSetSizeEx.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.GetProcessWorkingSetSizeEx.restype = wintypes.BOOL
    minimum = ctypes.c_size_t()
    maximum = ctypes.c_size_t()
    flags = wintypes.DWORD()
    ok = kernel32.GetProcessWorkingSetSizeEx(
        kernel32.GetCurrentProcess(),
        ctypes.byref(minimum),
        ctypes.byref(maximum),
        ctypes.byref(flags),
    )
    if not ok:
        return {"read": False, "last_error": ctypes.get_last_error()}
    return {
        "read": True,
        "minimum_bytes": int(minimum.value),
        "maximum_bytes": int(maximum.value),
        "flags": int(flags.value),
        "hard_min_enabled": bool(int(flags.value) & QUOTA_LIMITS_HARDWS_MIN_ENABLE),
    }


def lock_working_set(*, minimum_bytes: int, maximum_bytes: int) -> dict[str, Any]:
    """Request a hard minimum working set so the OS cannot trim the resident model.

    Returns the full attempt record - privilege result, Win32 error, and the readback - so a
    denial is as auditable as a grant.
    """
    before = read_working_set_limits()
    privilege = enable_working_set_privilege()
    if platform.system() != "Windows":
        return {
            "requested": False,
            "granted": False,
            "privilege": privilege,
            "before": before,
            "reason": f"not Windows: {platform.system()}",
        }

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.SetProcessWorkingSetSizeEx.argtypes = [
        wintypes.HANDLE,
        ctypes.c_size_t,
        ctypes.c_size_t,
        wintypes.DWORD,
    ]
    kernel32.SetProcessWorkingSetSizeEx.restype = wintypes.BOOL

    ctypes.set_last_error(0)
    ok = kernel32.SetProcessWorkingSetSizeEx(
        kernel32.GetCurrentProcess(),
        ctypes.c_size_t(int(minimum_bytes)),
        ctypes.c_size_t(int(maximum_bytes)),
        wintypes.DWORD(QUOTA_LIMITS_HARDWS_MIN_ENABLE),
    )
    last_error = ctypes.get_last_error()
    after = read_working_set_limits()
    return {
        "requested": True,
        "granted": bool(ok) and bool(after.get("hard_min_enabled")),
        "call": "SetProcessWorkingSetSizeEx",
        "flags": QUOTA_LIMITS_HARDWS_MIN_ENABLE,
        "flags_name": "QUOTA_LIMITS_HARDWS_MIN_ENABLE",
        "requested_minimum_bytes": int(minimum_bytes),
        "requested_maximum_bytes": int(maximum_bytes),
        "returned": bool(ok),
        "last_error": last_error,
        "privilege": privilege,
        "before": before,
        "after": after,
    }


def read_reserved_cpu_sets() -> dict[str, Any]:
    """Read the machine's reserved CPU set configuration. Never writes it."""
    result: dict[str, Any] = {
        "method": (
            r"HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Kernel\ReservedCpuSets "
            "(REG_BINARY); absent key means no reserved set is configured"
        ),
        "policy": "read_only_never_set",
    }
    if platform.system() != "Windows":
        result.update(present=None, reason=f"not Windows: {platform.system()}")
        return result
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\Session Manager\Kernel",
        ) as key:
            value, kind = winreg.QueryValueEx(key, "ReservedCpuSets")
    except FileNotFoundError:
        result.update(present=False, value_hex=None)
        return result
    except OSError as exc:
        result.update(present=None, reason=f"{type(exc).__name__}: {exc}")
        return result
    result.update(
        present=True,
        registry_type=kind,
        value_hex=bytes(value).hex() if isinstance(value, (bytes, bytearray)) else None,
    )
    return result
