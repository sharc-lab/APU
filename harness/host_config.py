"""Per-host configuration, keyed by hostname (upper case). Every orchestrator script's hostname guard and hw_id come
from here instead of a single hardcoded "EVO-T2S only" check, so a second machine (evo-x2) needs a table entry, not a
fork of the scripts.

Add a machine by adding an entry here. `python_exe` is the interpreter path used by scripts/deploy_evo.py-style WMI
launches (informational here; the launch command itself is built by the caller). `gpu_vendor` gates vendor-specific
telemetry (Level Zero Sysman is Intel-only) without a runtime probe that could hang or print noise on the wrong vendor.
"""

from __future__ import annotations

HOSTS = {
    "EVO-T2S": {
        "hw_id": "evo-t2s",
        "user": "sharc",
        "python_exe": r"C:\Users\SHARC\AppData\Local\Programs\Python\Python312\python.exe",
        "deploy_dir": r"C:\apu\ovn",
        "models_dir": r"C:\apu\models",
        "gpu_vendor": "intel",
    },
    "EVO-X2": {
        "hw_id": "evo-x2",
        "user": "Ritz",
        "python_exe": r"C:\Users\Ritz\AppData\Local\Programs\Python\Python312\python.exe",
        "deploy_dir": r"C:\apu\ovn",
        "models_dir": r"C:\apu\models",
        "gpu_vendor": "amd",
    },
}


def require_host(hostname: str) -> dict:
    """Raises SystemExit (matching the old 'EVO-T2S only' guard's behavior) if hostname is not in the allowlist."""
    h = HOSTS.get(hostname.upper())
    if h is None:
        raise SystemExit(f"unknown host {hostname!r}: not in harness/host_config.py HOSTS, refusing to run")
    return h
