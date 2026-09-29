"""Per-host configuration, keyed by hostname (upper case). Every orchestrator script's hostname guard and hw_id come
from here instead of a single hardcoded "EVO-T2S only" check, so a second machine (evo-x2) needs a table entry, not a
fork of the scripts.

Add a machine by adding an entry here. `python_exe` is the interpreter path used by scripts/deploy_evo.py-style WMI
launches (informational here; the launch command itself is built by the caller). `gpu_vendor` gates vendor-specific
telemetry (Level Zero Sysman is Intel-only) without a runtime probe that could hang or print noise on the wrong vendor.
`interactive_guard` controls what happens when someone is logged in at the console: True (evo-t2s, Zach's machine)
refuses to run at all; False (evo-x2, a dedicated research machine the user themselves may be sitting at) does not
refuse, but every row records console_session_state/console_idle_s/user_active instead (see
enforce_or_record_interactive_session below), so a speed-sensitive analysis can exclude user_active rows rather than
the whole run being blocked by the operator's own presence.
"""

from __future__ import annotations

import re
import subprocess

HOSTS = {
    "EVO-T2S": {
        "hw_id": "evo-t2s",
        "user": "sharc",
        "python_exe": r"C:\Users\SHARC\AppData\Local\Programs\Python\Python312\python.exe",
        "deploy_dir": r"C:\apu\ovn",
        "models_dir": r"C:\apu\models",
        "gpu_vendor": "intel",
        "ssh_host": "sharc@100.72.40.24",
        "interactive_guard": True,
    },
    "EVO-X2": {
        "hw_id": "evo-x2",
        "user": "Ritz",
        "python_exe": r"C:\Users\Ritz\AppData\Local\Programs\Python\Python312\python.exe",
        "deploy_dir": r"C:\apu\ovn",
        "models_dir": r"C:\apu\models",
        "gpu_vendor": "amd",
        "ssh_host": "Ritz@100.118.33.76",
        "interactive_guard": False,
    },
}

# Controller-side alias -> HOSTS key, for scripts/deploy_evo.py's --host flag (a short name is easier to type than the
# hostname the remote machine reports).
ALIASES = {"evo-t2s": "EVO-T2S", "t2s": "EVO-T2S", "evo-x2": "EVO-X2", "x2": "EVO-X2"}


def require_host(hostname: str) -> dict:
    """Raises SystemExit (matching the old 'EVO-T2S only' guard's behavior) if hostname is not in the allowlist."""
    h = HOSTS.get(hostname.upper())
    if h is None:
        raise SystemExit(f"unknown host {hostname!r}: not in harness/host_config.py HOSTS, refusing to run")
    return h


def _query_user_raw():
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command",
                           'try { (& query.exe user 2>&1) -join "`n" } catch { $_.Exception.Message }'],
                          capture_output=True, text=True, timeout=20)
        return r.stdout
    except Exception:
        return ""


def parse_idle_time(raw):
    """query.exe user's IDLE TIME column: 'none' (0s), a bare integer (minutes), 'H:MM' (over an hour), or
    'D+HH:MM' (over a day). Returns seconds as a float, or None if raw is empty/unparseable."""
    if raw is None:
        return None
    s = raw.strip()
    if not s:
        return None
    if s.lower() == "none":
        return 0.0
    m = re.match(r"^(\d+)\+(\d+):(\d+)$", s)
    if m:
        d, h, mi = (int(x) for x in m.groups())
        return float(d * 86400 + h * 3600 + mi * 60)
    m = re.match(r"^(\d+):(\d+)$", s)
    if m:
        h, mi = (int(x) for x in m.groups())
        return float(h * 3600 + mi * 60)
    m = re.match(r"^(\d+)$", s)
    if m:
        return float(int(m.group(1)) * 60)
    return None


def parse_query_user(text):
    """query.exe user's table output -> [{"username", "sessionname", "state", "idle_raw", "idle_s"}, ...]. Empty list
    when nobody is logged in ('No User exists for ...', or unparseable/empty output)."""
    if not text or "no user exists" in text.lower():
        return []
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 2:
        return []
    header = [h.strip().upper() for h in re.split(r"\s{2,}", lines[0].strip())]
    out = []
    for line in lines[1:]:
        fields = re.split(r"\s{2,}", line.lstrip(">").strip())
        if len(fields) < len(header):
            fields += [None] * (len(header) - len(fields))
        d = dict(zip(header, fields))
        idle_raw = d.get("IDLE TIME")
        out.append({"username": d.get("USERNAME"), "sessionname": d.get("SESSIONNAME"), "state": d.get("STATE"),
                    "idle_raw": idle_raw, "idle_s": parse_idle_time(idle_raw)})
    return out


def get_console_session_state(query_user_fn=_query_user_raw):
    """{"console_session_state": str|None, "console_idle_s": float|None, "user_active": bool}. user_active is True
    only when a console session is logged in AND its idle time is under 300s -- the threshold the analysis side uses
    to exclude speed-sensitive rows measured while the operator was actively using the machine."""
    sessions = parse_query_user(query_user_fn())
    console = next((s for s in sessions if (s.get("sessionname") or "").lower() == "console"), None)
    if console is None:
        return {"console_session_state": None, "console_idle_s": None, "user_active": False}
    idle_s = console.get("idle_s")
    return {"console_session_state": console.get("state"), "console_idle_s": idle_s,
            "user_active": idle_s is not None and idle_s < 300.0}


def enforce_or_record_interactive_session(host_cfg, query_user_fn=_query_user_raw):
    """Called once at startup, before the Lab exists. If host_cfg["interactive_guard"] is True (evo-t2s): raises
    SystemExit if anyone is logged in at all, exactly like the old hardcoded check. If False (evo-x2): never raises;
    returns get_console_session_state()'s dict so the caller can stash it on the Lab for every row to read (see
    t2s_overnight.Lab.row). Returns {} when the guard is on and nobody is logged in (nothing to record)."""
    if host_cfg.get("interactive_guard", True):
        text = query_user_fn()
        if "No User exists" not in text:
            raise SystemExit("another interactive session is logged in: " + text)
        return {}
    return get_console_session_state(query_user_fn)
