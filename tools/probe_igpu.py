"""Does the iGPU enumerate, and in which execution contexts?

Arm B of ΔN is cpu-p + iGPU, and the iGPU path has never been exercised on this machine under any
launch path. Enumeration is the first thing that can fail, and it can fail for reasons that have
nothing to do with the hardware: a process in session 0 with no window station may not see a
display device at all. If the detached launch path -- the one an unattended run must use -- cannot
enumerate the GPU, that is a plumbing failure. Discovering it after spending a ceiling run, and
recording it as a fact about capacity, would be a fabricated result.

Hence the same binary and the same model-free code path in every context, with the context passed
in as a label rather than inferred, so three artifacts can be compared without ambiguity.

**A failed probe is a successful probe with gpu_present=false.** Nothing here retries, and nothing
raises: every failure is caught, classified, recorded verbatim, and the process exits 0. A retry
loop would turn a clean negative into a hang, and a non-zero exit would make a recorded answer
look like a broken tool.

Enumeration only. No model load, no inference, no capacity measurement. The per-model iGPU smoke
test for openvino#34390 is a separate dispatch and is deliberately not performed here.

Diagnostic, not a measurement: writes only under ``derived/igpu_probe/``, seals nothing, and
touches no run ledger.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import datetime
import json
import os
import platform
import sys
import traceback
from ctypes import wintypes
from pathlib import Path
from typing import Any

UOI_NAME = 2
UOI_FLAGS = 4
WSF_VISIBLE = 0x0001


class _USEROBJECTFLAGS(ctypes.Structure):
    _fields_ = [
        ("fInherit", wintypes.BOOL),
        ("fReserved", wintypes.BOOL),
        ("dwFlags", wintypes.DWORD),
    ]


def _describe(exc: BaseException) -> str:
    """Verbatim type and message. Never trimmed to something tidier than what happened."""
    return f"{type(exc).__name__}: {exc}"


def _session_id(pid: int) -> int | None:
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        session = wintypes.DWORD()
        if kernel32.ProcessIdToSessionId(wintypes.DWORD(pid), ctypes.byref(session)):
            return int(session.value)
    except Exception:  # a diagnostic must not fail on its own diagnostics
        return None
    return None


def _user32() -> Any:
    """user32 with the two window-station prototypes declared.

    HWINSTA is a pointer. Without an explicit restype ctypes truncates it to a 32-bit int, which
    happens to survive for small handle values and fails for large ones -- the kind of defect that
    works on the machine you tested and returns null on the context you actually care about.
    """
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
    return user32


def _window_station(errors: dict[str, str]) -> str | None:
    """Name of this process's window station. ``Service-0x0-...`` means session 0, no desktop."""
    try:
        user32 = _user32()
        handle = user32.GetProcessWindowStation()
        if not handle:
            errors["window_station"] = (
                f"GetProcessWindowStation returned NULL, GetLastError={ctypes.get_last_error()}"
            )
            return None
        needed = wintypes.DWORD()
        buffer = ctypes.create_unicode_buffer(256)
        if user32.GetUserObjectInformationW(
            handle,
            UOI_NAME,
            ctypes.cast(buffer, ctypes.c_void_p),
            ctypes.sizeof(buffer),
            ctypes.byref(needed),
        ):
            return str(buffer.value)
        errors["window_station"] = (
            "GetUserObjectInformationW(UOI_NAME) failed, " f"GetLastError={ctypes.get_last_error()}"
        )
    except Exception as exc:
        errors["window_station"] = _describe(exc)
    return None


def _is_interactive(errors: dict[str, str]) -> bool | None:
    """The same test ``[Environment]::UserInteractive`` performs: WSF_VISIBLE on the station.

    Reimplemented in-process rather than shelled out to PowerShell, because a separate process
    could land in a different session and would answer a different question than the one asked.

    A null here is not "not interactive" -- it is "could not tell", and the reason is recorded, so
    the session-0 diagnosis is never made from a silently failed call.

    The size is queried first: a fixed ``sizeof(USEROBJECTFLAGS)`` returns ERROR_INSUFFICIENT_BUFFER
    (122) on this build, and a null ``is_interactive`` with a working ``window_station`` would
    leave the session-0 diagnosis half-answered.
    """
    try:
        user32 = _user32()
        handle = user32.GetProcessWindowStation()
        if not handle:
            errors["is_interactive"] = (
                f"GetProcessWindowStation returned NULL, GetLastError={ctypes.get_last_error()}"
            )
            return None
        needed = wintypes.DWORD()
        user32.GetUserObjectInformationW(handle, UOI_FLAGS, None, 0, ctypes.byref(needed))
        if needed.value == 0:
            errors["is_interactive"] = (
                "GetUserObjectInformationW(UOI_FLAGS) size query returned 0, "
                f"GetLastError={ctypes.get_last_error()}"
            )
            return None
        buffer = (ctypes.c_byte * int(needed.value))()
        if not user32.GetUserObjectInformationW(
            handle,
            UOI_FLAGS,
            ctypes.byref(buffer),
            needed.value,
            ctypes.byref(needed),
        ):
            errors["is_interactive"] = (
                "GetUserObjectInformationW(UOI_FLAGS) failed, "
                f"GetLastError={ctypes.get_last_error()}, needed={needed.value}"
            )
            return None
        flags = _USEROBJECTFLAGS.from_buffer_copy(buffer)
        return bool(flags.dwFlags & WSF_VISIBLE)
    except Exception as exc:
        errors["is_interactive"] = _describe(exc)
    return None


def _parent() -> tuple[str | None, int | None, int | None]:
    try:
        import psutil

        parent = psutil.Process().parent()
        if parent is None:
            return None, None, None
        return str(parent.name()), int(parent.pid), _session_id(int(parent.pid))
    except Exception:
        return None, None, None


def _opencl_platforms() -> dict[str, Any] | None:
    """Enumerate OpenCL platforms through ``OpenCL.dll`` directly.

    Done with ctypes rather than pyopencl so the probe has no dependency the three contexts could
    differ on. The iGPU compute stack is reached through OpenCL, so a session that sees zero
    platforms explains a GPU that does not enumerate.
    """
    try:
        opencl = ctypes.CDLL("OpenCL.dll")
    except OSError as exc:
        return {"count": None, "names": None, "error": _describe(exc)}

    try:
        count = ctypes.c_uint(0)
        if opencl.clGetPlatformIDs(0, None, ctypes.byref(count)) != 0:
            return {"count": None, "names": None, "error": "clGetPlatformIDs enumeration failed"}
        n = int(count.value)
        if n == 0:
            return {"count": 0, "names": [], "error": None}

        ids = (ctypes.c_void_p * n)()
        if opencl.clGetPlatformIDs(n, ids, None) != 0:
            return {"count": n, "names": None, "error": "clGetPlatformIDs retrieval failed"}

        names: list[str] = []
        for platform_id in ids:
            buffer = ctypes.create_string_buffer(1024)
            size = ctypes.c_size_t(0)
            # CL_PLATFORM_NAME = 0x0902
            if (
                opencl.clGetPlatformInfo(
                    ctypes.c_void_p(platform_id),
                    0x0902,
                    ctypes.sizeof(buffer),
                    buffer,
                    ctypes.byref(size),
                )
                == 0
            ):
                names.append(buffer.value.decode("utf-8", errors="replace"))
            else:
                names.append("<clGetPlatformInfo failed>")
        return {"count": n, "names": names, "error": None}
    except Exception as exc:
        return {"count": None, "names": None, "error": _describe(exc)}


def _path_head(n: int = 6) -> list[str]:
    """First ``n`` PATH entries for post-hoc diagnosis of interpreter resolution."""
    raw = os.environ.get("PATH", "")
    return [p for p in raw.split(os.pathsep) if p][:n]


def probe(launch_context: str) -> dict[str, Any]:
    parent_name, parent_pid, parent_session = _parent()
    pid = os.getpid()
    probe_errors: dict[str, str] = {}
    python_exe = str(Path(sys.executable).resolve())

    record: dict[str, Any] = {
        "hostname": platform.node(),
        "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "launch_context": launch_context,
        "process_id": pid,
        "session_id": _session_id(pid),
        "parent_process_name": parent_name,
        "parent_session_id": parent_session,
        "is_interactive": _is_interactive(probe_errors),
        "window_station": _window_station(probe_errors),
        "python_exe": python_exe,
        "python_version": sys.version,
        "python_in_venv": sys.prefix != sys.base_prefix,
        "path_head": _path_head(6),
        "ov_version": None,
        "ov_available_devices": None,
        "gpu_present": False,
        "gpu_full_name": None,
        "gpu_probe_error": None,
        "npu_present": False,
        "opencl_platforms": _opencl_platforms(),
        "diagnostics": {
            "parent_process_id": parent_pid,
            "python_executable": python_exe,
            "argv": list(sys.argv),
            "ov_import_error": None,
            "ov_enumerate_error": None,
            "gpu_property_errors": {},
            "npu_full_name": None,
            "context_field_errors": probe_errors,
            "traceback": None,
        },
    }

    try:
        import openvino as ov
    except Exception as exc:
        record["diagnostics"]["ov_import_error"] = _describe(exc)
        record["diagnostics"]["traceback"] = traceback.format_exc()[:4000]
        record["gpu_probe_error"] = _describe(exc)
        return record

    try:
        record["ov_version"] = str(ov.get_version())
    except Exception as exc:
        record["ov_version"] = f"<unavailable: {_describe(exc)}>"

    try:
        core = ov.Core()
        devices = list(core.available_devices)
        record["ov_available_devices"] = devices
    except Exception as exc:
        record["diagnostics"]["ov_enumerate_error"] = _describe(exc)
        record["diagnostics"]["traceback"] = traceback.format_exc()[:4000]
        record["gpu_probe_error"] = _describe(exc)
        return record

    record["gpu_present"] = any(str(d).startswith("GPU") for d in devices)
    record["npu_present"] = any(str(d).startswith("NPU") for d in devices)

    # Recorded, never acted on: the NPU is out of scope for arm B and this dispatch.
    if record["npu_present"]:
        try:
            record["diagnostics"]["npu_full_name"] = str(
                core.get_property("NPU", "FULL_DEVICE_NAME")
            )
        except Exception as exc:
            record["diagnostics"]["gpu_property_errors"]["NPU"] = _describe(exc)

    if record["gpu_present"]:
        try:
            record["gpu_full_name"] = str(core.get_property("GPU", "FULL_DEVICE_NAME"))
        except Exception as exc:
            # Enumerating and then failing to describe it is a distinct state from not
            # enumerating, and the difference is the point of recording both fields.
            record["gpu_probe_error"] = _describe(exc)
            record["diagnostics"]["traceback"] = traceback.format_exc()[:4000]
    else:
        record["gpu_probe_error"] = (
            "no device with an id beginning 'GPU' in "
            f"core.available_devices={devices!r}; this is an enumeration absence, not an error "
            "raised by the runtime"
        )

    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--launch-context",
        required=True,
        choices=["local_console", "ssh_foreground", "ssh_detached"],
        help="how this process was started; passed in rather than inferred",
    )
    parser.add_argument("--out", type=Path, help="also write the object here")
    args = parser.parse_args(argv)

    try:
        record = probe(args.launch_context)
    except BaseException as exc:
        # Belt and braces. The probe already catches everything internally; if something still
        # escapes, it is recorded as the answer rather than allowed to look like a broken tool.
        python_exe = str(Path(sys.executable).resolve())
        record = {
            "launch_context": args.launch_context,
            "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "python_exe": python_exe,
            "python_version": sys.version,
            "python_in_venv": sys.prefix != sys.base_prefix,
            "path_head": _path_head(6),
            "ov_version": None,
            "gpu_present": False,
            "gpu_probe_error": _describe(exc),
            "diagnostics": {"traceback": traceback.format_exc()[:4000]},
        }

    rendered = json.dumps(record, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")

    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(rendered, flush=True)

    # Always 0. A failed probe is a successful probe with gpu_present=false.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
