"""Shared helper so controller-side scripts (anything that can run from a Windows scheduled task or in the
background on this laptop -- sync_results.py is the real example, 2026-10-02) never flash a visible console
window for a child process.

Root cause: the scheduled task's own parent process is pythonw.exe (console-less by design, confirmed live
against the real APU-SyncResults task registration), but a bare subprocess.run/Popen/check_output call for a
console-subsystem child (ssh.exe, scp.exe, git.exe, powershell.exe, cmd.exe) has no console to inherit from a
console-less parent, so Windows allocates a brand-new, visible console for EACH such child. This is a per-call
problem, not a one-time setup -- every call that was not routed through this helper still flashes its own
window regardless of how quiet the parent is.

Fix: subprocess.CREATE_NO_WINDOW (0x08000000) tells Windows not to allocate a console for the child at all.
Also setting STARTUPINFO.wShowWindow = SW_HIDE (0) with STARTF_USESHOWWINDOW is belt-and-suspenders for any
child that creates its own window via other means (CREATE_NO_WINDOW alone covers the console case, which is
the one that actually matters here, but costs nothing extra to also pass).

No-op on non-Windows (the flag and STARTUPINFO class are Windows-only); scripts that run on Linux (e.g. the
Linux-only checks in verify_platform.py) call the same functions safely without special-casing their platform.

Usage -- replace every bare subprocess call in a controller-side script with the matching wrapper here:
    from proc_util import run_hidden, popen_hidden, check_output_hidden
    run_hidden(["git", "status"], capture_output=True)       # instead of subprocess.run(...)
    popen_hidden(["ssh", host, cmd])                          # instead of subprocess.Popen(...)
    check_output_hidden(["free", "-b"])                       # instead of subprocess.check_output(...)
All three accept exactly the same keyword arguments as their subprocess.* counterparts; this module only adds
creationflags/startupinfo on top, it never removes or overrides a caller-supplied value for either.
"""
from __future__ import annotations

import subprocess
import sys


def _hidden_kwargs(kwargs: dict) -> dict:
    """Merges in CREATE_NO_WINDOW + a hidden STARTUPINFO, without clobbering anything the caller already set.
    A caller-supplied creationflags is OR'd with CREATE_NO_WINDOW (never replaced); a caller-supplied startupinfo
    has its own wShowWindow/dwFlags set rather than being overwritten wholesale."""
    if sys.platform != "win32":
        return kwargs
    out = dict(kwargs)
    out["creationflags"] = out.get("creationflags", 0) | subprocess.CREATE_NO_WINDOW
    si = out.get("startupinfo")
    if si is None:
        si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = subprocess.SW_HIDE
    out["startupinfo"] = si
    return out


def run_hidden(args, **kwargs):
    return subprocess.run(args, **_hidden_kwargs(kwargs))


def popen_hidden(args, **kwargs):
    return subprocess.Popen(args, **_hidden_kwargs(kwargs))


def check_output_hidden(args, **kwargs):
    return subprocess.check_output(args, **_hidden_kwargs(kwargs))
