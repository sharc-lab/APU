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
        "python_exe": r"%USERPROFILE%\AppData\Local\Programs\Python\Python312\python.exe",
        "deploy_dir": r"C:\apu\ovn",
        "models_dir": r"C:\apu\models",
        "gpu_vendor": "intel",
        "ssh_host": "sharc@100.72.40.24",
        "interactive_guard": True,
    },
    "EVO-X2": {
        "hw_id": "evo-x2",
        "user": "Ritz",
        "python_exe": r"%USERPROFILE%\AppData\Local\Programs\Python\Python312\python.exe",
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


def ollama_process_running(ps_fn=None):
    """True if any ollama.exe/"ollama app.exe" process exists right now, regardless of whether it holds a model in
    GPU memory. Added after the 2026-09-29 contamination check (see docs/RESULT_PROVENANCE.md): Ollama should only
    run inside K1/K2 jobs, which start and stop it themselves, not idle in the background during any other phase.
    Every non-Ollama phase checks this before starting and aborts if it finds one, rather than relying on GPU-memory
    residency alone (an idle server with a 0 KEEP_ALIVE can still race a model load against another experiment)."""
    if ps_fn is None:
        p = subprocess.run(["powershell", "-NoProfile", "-Command",
                           "(Get-Process ollama,'ollama app' -ErrorAction SilentlyContinue | Measure-Object).Count"],
                          capture_output=True, text=True, timeout=20)
        out = p.stdout
    else:
        out = ps_fn("(Get-Process ollama,'ollama app' -ErrorAction SilentlyContinue | Measure-Object).Count", 20)
    return out.strip() not in ("", "0")


def start_ollama_server(ps_fn=None):
    """Starts `ollama serve` headless via WMI Win32_Process Create -- a plain Start-Job does not survive past the SSH
    session that launched it (discovered 2026-09-28 the hard way, see docs/T2S_CHANGELOG.md) -- with
    OLLAMA_KEEP_ALIVE=0 so a model never lingers in GPU memory once a call finishes. Idempotent: no-ops (returns None)
    if ollama_process_running() already reports a process. Returns the launched PID as an int, or None if the launch
    output could not be parsed. Called by K1/K2's own job lifecycle, which are the only phases allowed to run Ollama
    at all (see docs/RESULT_PROVENANCE.md, 2026-09-29 contamination check)."""
    if ollama_process_running(ps_fn=ps_fn):
        return None
    cmd = ("$cmd = 'cmd.exe /c set OLLAMA_KEEP_ALIVE=0 && ollama serve > C:\\apu\\ovn\\ollama_serve.log 2>&1'; "
           "$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$cmd}; "
           "'pid=' + $r.ProcessId")
    if ps_fn is None:
        p = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=30)
        out = p.stdout
    else:
        out = ps_fn(cmd, 30)
    m = re.search(r"pid=(\d+)", out or "")
    return int(m.group(1)) if m else None


def stop_ollama_server(ps_fn=None):
    """Stops every ollama.exe/"ollama app.exe" process (Stop-Process -Force). Called from K1/K2's own finally block
    so Ollama never idles in the background once the job that needed it ends -- see start_ollama_server and the
    2026-09-29 contamination check in docs/RESULT_PROVENANCE.md."""
    cmd = "Get-Process ollama,'ollama app' -ErrorAction SilentlyContinue | Stop-Process -Force; 'stopped'"
    if ps_fn is None:
        p = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=30)
        return p.stdout.strip()
    return ps_fn(cmd, 30)


def wait_for_ollama_ready(base_url="http://127.0.0.1:11434", timeout_s=30, poll_interval_s=0.5):
    """Blocks until GET /api/tags responds (Ollama's HTTP server is actually listening), or timeout_s elapses.

    Found live 2026-10-01: start_ollama_server() returns as soon as the WMI Create call reports a pid, with no
    confirmation the HTTP server is actually accepting connections yet. On evo-t2s, phase_tier_v3's immediate
    first pull (9s after the WMI-reported start) failed with a connection error on BOTH non-create models
    (llama3.1:8b, qwen3:8b) -- a real race, not a flaky one-off, since Ollama's own startup time is not
    guaranteed to be under any fixed number of seconds on every boot. Returns True once ready, False if
    timeout_s elapses without a successful response (never raises -- the caller decides what to do with a
    server that never came up, the same way every other best-effort check in this module does)."""
    import time
    import urllib.request
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/api/tags", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(poll_interval_s)
    return False


def get_ollama_loaded_model(base_url="http://127.0.0.1:11434"):
    """The model name Ollama currently holds in GPU memory, or None if nothing is loaded or the Ollama server is not
    reachable (evo-t2s never runs Ollama; on evo-x2 it may not be running yet). Used to stamp ollama_model_loaded on
    every evo-x2 row -- K1 and every other X2 experiment set OLLAMA_KEEP_ALIVE=0 on the Ollama server so a model
    never lingers in GPU memory once a call finishes, but this field records the ground truth per row rather than
    just assuming that setting worked."""
    import json
    import urllib.request
    try:
        with urllib.request.urlopen(f"{base_url}/api/ps", timeout=3) as r:
            data = json.loads(r.read())
    except Exception:
        return None
    models = data.get("models") or []
    return models[0]["name"] if models else None


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
