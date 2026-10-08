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
        # Relative to the host user's profile dir; resolved by _this_host_entry.
        "ollama_exe": r"AppData\Local\Programs\Ollama\ollama.exe",
        "ollama_models": r".ollama\models",
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
        # Relative to the host user's profile dir; resolved by _this_host_entry.
        "ollama_exe": r"AppData\Local\Programs\Ollama\ollama.exe",
        "ollama_models": r".ollama\models",
    },
    # The Razer Blade 14 (RTX 4070 Laptop, 8 GB VRAM, CUDA): the NVIDIA platform. It is also the controller laptop, so
    # its jobs run locally (no ssh_host) under scripts/blade_night.py and harness/blade_queue.py, never from the evo-x2
    # queue. Its Ollama is started by harness/blade_common.LocalOllama (a hidden child process, not the WMI launch
    # start_ollama_server uses: a WMI-created cmd.exe on the local desktop would open a visible console window).
    "RITZLAPTOP": {
        "hw_id": "blade",
        "user": "rithw",
        "python_exe": r"%USERPROFILE%\AppData\Local\Programs\Python\Python312\python.exe",
        "deploy_dir": None,
        "models_dir": r"C:\apu\models",
        "gpu_vendor": "nvidia",
        "ssh_host": None,
        "interactive_guard": False,
        # None: the Blade's jobs use blade_common.PINNED["ollama_exe"] (the side-by-side 0.34.4), never the tray
        # install, so nothing here should resolve to it.
        "ollama_exe": None,
        "ollama_models": r".ollama\models",
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


def _this_host_entry() -> dict:
    """HOSTS entry for the machine this process runs on, or {} if it is not a known measurement host."""
    import os
    import socket
    h = dict(HOSTS.get(socket.gethostname().upper(), {}))
    if h:
        home = os.path.join(os.environ.get("SystemDrive", "C:") + os.sep, "Users", h["user"])
        for k in ("ollama_exe", "ollama_models"):
            if h.get(k):
                h[k] = os.path.join(home, h[k])
    return h


def _resolve_ollama_exe_for_serve():
    """OLLAMA_BIN env var, then shutil.which("ollama"), then the Windows installer's own standard per-user
    install location, then the bare command name as a last resort -- same resolution order as
    t2s_k1_ollama.py's own _resolve_ollama_exe (duplicated here rather than imported, to avoid a circular
    import: t2s_k1_ollama already imports this module).

    Found live 2026-10-01 (evo-t2s): start_ollama_server's own WMI-launched "cmd.exe /c ... && ollama serve"
    used the bare command name, which fails to resolve on evo-t2s specifically (confirmed: `where ollama`
    also fails there even in a plain interactive SSH session, while the exe is confirmed present at the
    standard install path) -- the server then never actually starts, and every subsequent pull/chat call
    fails with a connection-refused error, not a timeout. This is T2S-specific: the identical code path has
    worked on evo-x2 all session, so evo-x2's install must register itself on PATH in a way evo-t2s's does
    not.

    2026-10-06 (evo-x2): the queue watchdog runs as SYSTEM (APU-QueueWatchdog, ServiceAccount), whose
    LOCALAPPDATA is the system profile, so the LOCALAPPDATA fallback misses the per-user install and the bare
    name fails too ('"ollama"' is not recognized; x2_half_context_probe started 0 of its reps). The host's own
    pinned `ollama_exe` from HOSTS is checked before LOCALAPPDATA for that reason."""
    import os
    import shutil
    exe = os.environ.get("OLLAMA_BIN") or shutil.which("ollama")
    if exe:
        return exe
    pinned = _this_host_entry().get("ollama_exe")
    if pinned and os.path.exists(pinned):
        return pinned
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        candidate = os.path.join(local_appdata, "Programs", "Ollama", "ollama.exe")
        if os.path.exists(candidate):
            return candidate
    return "ollama"


DEFAULT_OLLAMA_SERVE_LOG = r"C:\apu\ovn\ollama_serve.log"
_ENV_KEY_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_ENV_VALUE_FORBIDDEN = set("\"'&|<>^%\r\n")  # cmd.exe metacharacters, and ' would end the PowerShell string


def _extra_env_sets(env) -> str:
    """cmd.exe `set "K=V"&& ` prefixes for start_ollama_server's optional env. Quoted assignment, so no trailing
    space ends up in the value (the t2s_queue 2026-09-30 bug). Raises ValueError on a key that is not an
    upper-case identifier, on a value containing a cmd.exe metacharacter or a single quote, and on the two
    variables start_ollama_server sets itself."""
    out = ""
    for k, v in (env or {}).items():
        v = str(v)
        if not _ENV_KEY_RE.match(k):
            raise ValueError(f"start_ollama_server env: bad variable name {k!r}")
        if any(c in _ENV_VALUE_FORBIDDEN for c in v):
            raise ValueError(f"start_ollama_server env: value for {k} contains a forbidden character: {v!r}")
        if k in ("OLLAMA_KEEP_ALIVE", "OLLAMA_MODELS"):
            raise ValueError(f"start_ollama_server env: {k} is set by start_ollama_server itself")
        out += f'set "{k}={v}"&& '
    return out


def start_ollama_server(ps_fn=None, env=None, log_path=None):
    """Starts `ollama serve` headless via WMI Win32_Process Create -- a plain Start-Job does not survive past the SSH
    session that launched it (discovered 2026-09-28 the hard way, see docs/T2S_CHANGELOG.md) -- with
    OLLAMA_KEEP_ALIVE=0 so a model never lingers in GPU memory once a call finishes. Idempotent: no-ops (returns None)
    if ollama_process_running() already reports a process. Returns the launched PID as an int, or None if the launch
    output could not be parsed. Called by K1/K2's own job lifecycle, which are the only phases allowed to run Ollama
    at all (see docs/RESULT_PROVENANCE.md, 2026-09-29 contamination check).

    env (optional, 2026-10-08): extra environment variables for the server process only, e.g. {"OLLAMA_DEBUG": "1"}
    for x2_r2_agent's mechanism job. Because of the idempotency above, a caller that needs env to take effect must
    make sure no Ollama process is running first (stop_ollama_server) and should confirm it from the server log's
    own "server config" line. log_path (optional): where the server's stdout+stderr go, default
    DEFAULT_OLLAMA_SERVE_LOG; cmd.exe `>` truncates it at each start. With env=None and log_path=None the launched
    command line is exactly what it was before these parameters existed."""
    if ollama_process_running(ps_fn=ps_fn):
        return None
    import os
    extra = _extra_env_sets(env)
    log = log_path or DEFAULT_OLLAMA_SERVE_LOG
    if any(c in _ENV_VALUE_FORBIDDEN or c == " " for c in log):
        raise ValueError(f"start_ollama_server log_path contains a forbidden character: {log!r}")
    exe = _resolve_ollama_exe_for_serve()
    # Under SYSTEM, Ollama's default models dir is the system profile's (empty); point it at the host's own store.
    models = os.environ.get("OLLAMA_MODELS") or _this_host_entry().get("ollama_models")
    models_set = f"set OLLAMA_MODELS={models}&& " if models else ""
    cmd = (f"$cmd = 'cmd.exe /c set OLLAMA_KEEP_ALIVE=0 && {models_set}{extra}\"{exe}\" serve > {log} 2>&1'; "
           "$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$cmd}; "
           "'pid=' + $r.ProcessId")
    if ps_fn is None:
        p = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=30)
        out = p.stdout
    else:
        out = ps_fn(cmd, 30)
    m = re.search(r"pid=(\d+)", out or "")
    return int(m.group(1)) if m else None


def stop_ollama_server(ps_fn=None, wait_s=15, poll_s=0.5):
    """Stops every ollama.exe/"ollama app.exe" process (Stop-Process -Force), then polls
    ollama_process_running() until it reports False (or wait_s elapses) before returning. Called from K1/K2's
    own finally block so Ollama never idles in the background once the job that needed it ends -- see
    start_ollama_server and the 2026-09-29 contamination check in docs/RESULT_PROVENANCE.md.

    2026-10-06 bug found live, real data: x2_outcome_table_v2's own per-call start/stop cycle (run_one_ollama)
    hit "connection refused" on 409 of ~650 ollama_default calls across the weekend. Root cause: Stop-Process
    -Force returning does not guarantee the target process has actually finished terminating on Windows (it
    can still hold its process table entry for a brief window). The NEXT call's start_ollama_server() own
    idempotency check (`if ollama_process_running(): return None`) then saw that dying zombie, skipped
    launching a fresh server, and the dying process's port was gone by the time the chat call fired --
    wait_for_ollama_ready's own 60s wait could not help, since nothing was ever going to start. This wait-for-
    exit closes that race at the source; run_one_ollama is also fixed separately to retry once and record a
    distinct infra-failure if the server still isn't ready, as defense in depth."""
    # 2026-10-07, found live on evo-x2: killing ollama.exe does not kill its model runner (a separate
    # llama-server.exe under Ollama's own lib\ollama dir). Three orphaned runners holding a qwen3 model were
    # still alive after stop_ollama_server() and were only removed by the queue's pre-launch cleanup. An
    # orphaned runner keeps its GPU allocation, a candidate cause of the weekend's clustered ollama OOM rows.
    # Kill runners too, matched by executable path so the harnesses' own llama-server (C:\apu\bin) is untouched.
    cmd = ("Get-Process ollama,'ollama app' -ErrorAction SilentlyContinue | Stop-Process -Force; "
           "Get-Process llama-server -ErrorAction SilentlyContinue | Where-Object { $_.Path -like '*\\Ollama\\*' } "
           "| Stop-Process -Force; 'stopped'")
    if ps_fn is None:
        p = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=30)
        out = p.stdout.strip()
    else:
        out = ps_fn(cmd, 30)
    import time
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if not ollama_process_running(ps_fn=ps_fn):
            break
        time.sleep(poll_s)
    return out


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
