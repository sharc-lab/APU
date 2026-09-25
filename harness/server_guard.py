"""Standing stale-server guard for every harness that launches llama-server.

Why: on Windows two processes can both bind the same port (SO_REUSEADDR). A server that was never terminated keeps
answering while a newly started server sits idle and logs "listening". This produced invalid rows in
blade_m2_host_interference Part C. Three checks, all recorded:

  1. assert_port_free(port)        before any server start. If anything listens, record its PID and command line
                                   and raise GuardError. Never reuse it.
  2. assert_server_matches(...)    after each start: GET /props and assert model path, n_ctx and slot count, and
                                   assert KV type, flash-attn and the other launch flags from the process command
                                   line. /props on b10970 does NOT expose KV type or flash-attn, so those two are
                                   asserted from the command line and the record says so.
  3. RequestGuard(port, pid).check()  before every request: the listening PID must equal the PID we started.
                                   On mismatch the caller must mark the row invalid and stop that condition.

Windows only (netstat, PowerShell CIM). No third-party imports.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import urllib.request


class GuardError(Exception):
    def __init__(self, message: str, record: dict | None = None):
        super().__init__(message)
        self.record = record or {}


def _run(argv, timeout=20) -> str:
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
        return r.stdout
    except Exception:
        return ""


def parse_netstat_listeners(text: str, port: int) -> list[int]:
    """PIDs listening on `port` from `netstat -ano -p TCP` output (IPv4 and IPv6 rows)."""
    pids = []
    for line in text.splitlines():
        f = line.split()
        if len(f) >= 5 and f[0].upper() == "TCP" and f[3].upper() == "LISTENING":
            if re.search(rf":{port}$", f[1]):
                pids.append(int(f[4]))
    return pids


def port_listeners(port: int) -> list[int]:
    return parse_netstat_listeners(_run(["netstat", "-ano", "-p", "TCP"]), port)


def process_cmdline(pid: int) -> str | None:
    out = _run(["powershell", "-NoProfile", "-Command",
                f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine"], timeout=30).strip()
    return out or None


def parse_flags(cmdline: str) -> dict:
    """Map llama-server launch flags to values from a Windows command line."""
    toks = [t.strip('"') for t in shlex.split(cmdline, posix=False)]
    alias = {"-m": "model", "--model": "model", "-c": "ctx", "--ctx-size": "ctx", "-ctk": "ctk",
             "--cache-type-k": "ctk", "-ctv": "ctv", "--cache-type-v": "ctv", "-fa": "fa", "--flash-attn": "fa",
             "-ngl": "ngl", "--n-gpu-layers": "ngl", "-np": "np", "--parallel": "np", "--port": "port",
             "-t": "threads", "--threads": "threads"}
    out = {"exe": toks[0] if toks else None}
    for i, t in enumerate(toks):
        if t in alias and i + 1 < len(toks):
            out[alias[t]] = toks[i + 1]
        if t == "--no-context-shift":
            out["no_context_shift"] = True
    return out


def assert_port_free(port: int) -> dict:
    pids = port_listeners(port)
    if pids:
        rec = {"guard": "port_not_free", "port": port,
               "listeners": [{"pid": p, "command_line": process_cmdline(p)} for p in pids]}
        raise GuardError(f"port {port} already has a listener: {rec['listeners']}. STOP, not reusing it.", rec)
    return {"guard": "port_free", "port": port}


def _norm(p: str | None) -> str:
    return os.path.normcase(os.path.normpath(p)) if p else ""


def assert_server_matches(url: str, port: int, server_pid: int, expected: dict) -> dict:
    """expected keys: model_path, n_ctx, ctk, ctv, fa, ngl, np, threads (any subset)."""
    problems = []
    lp = port_listeners(port)
    if lp != [server_pid]:
        problems.append(f"listener pids {lp} != started pid {server_pid}")
    props = {}
    try:
        with urllib.request.urlopen(f"{url}/props", timeout=15) as r:
            props = json.loads(r.read())
    except Exception as e:
        problems.append(f"/props unreadable: {e}")
    n_ctx = (props.get("default_generation_settings") or {}).get("n_ctx")
    if "model_path" in expected and _norm(props.get("model_path")) != _norm(expected["model_path"]):
        problems.append(f"model_path {props.get('model_path')!r} != {expected['model_path']!r}")
    if "n_ctx" in expected and n_ctx != expected["n_ctx"]:
        problems.append(f"n_ctx {n_ctx} != {expected['n_ctx']}")
    if "np" in expected and props.get("total_slots") != int(expected["np"]):
        problems.append(f"total_slots {props.get('total_slots')} != {expected['np']}")
    cmd = process_cmdline(server_pid)
    flags = parse_flags(cmd) if cmd else {}
    if not cmd:
        problems.append("process command line unreadable")
    for key, want in (("ctk", expected.get("ctk")), ("ctv", expected.get("ctv")), ("fa", expected.get("fa")),
                      ("ngl", expected.get("ngl")), ("threads", expected.get("threads"))):
        if want is not None and str(flags.get(key)) != str(want):
            problems.append(f"command-line {key}={flags.get(key)!r} != {want!r}")
    if "n_ctx" in expected and str(flags.get("ctx")) != str(expected["n_ctx"]):
        problems.append(f"command-line ctx={flags.get('ctx')!r} != {expected['n_ctx']!r}")
    rec = {"guard": "server_matches", "server_pid": server_pid, "listener_pids": lp,
           "props_model_path": props.get("model_path"), "props_n_ctx": n_ctx,
           "props_total_slots": props.get("total_slots"), "props_build_info": props.get("build_info"),
           "command_line": cmd, "flags": flags,
           "kv_type_source": "process command line (-ctk/-ctv); /props on b10970 does not expose KV type",
           "flash_attn_source": "process command line (-fa); /props on b10970 does not expose flash-attn",
           "problems": problems}
    if problems:
        raise GuardError("server does not match intended config: " + "; ".join(problems), rec)
    return rec


class RequestGuard:
    """Call check() before every request. ok=False means: mark the row invalid and stop the condition."""

    def __init__(self, port: int, pid: int):
        self.port, self.pid = port, pid

    def check(self) -> tuple[bool, list[int]]:
        lp = port_listeners(self.port)
        return lp == [self.pid], lp
