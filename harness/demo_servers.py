"""One Ollama server per demo configuration (2026-10-23 operator decision 4): its own port, its own log, OLLAMA_DEBUG=1
and the configuration's server env for that process only (never the tray app, never a persistent or system-wide
variable), started and stopped by the job that uses it.

Stale-server guard (harness/server_guard.py): the port must be free before the start (GuardError otherwise, never
reused); after the start the listener on the port must be the process we started; before every request the caller
runs check() and stops that configuration on a mismatch. Stop kills the process tree (taskkill /T), so Ollama's model
runner (a separate llama-server.exe) does not outlive it (host_config.stop_ollama_server's 2026-10-07 note).

Windows child processes are created with proc_util.popen_hidden / run_hidden (no console window; CLAUDE.md).
"""
from __future__ import annotations

import os
import time
import urllib.request
from pathlib import Path

import server_guard as sg
from proc_util import popen_hidden, run_hidden

DEBUG_ENV = {"OLLAMA_DEBUG": "1"}
FORBIDDEN_ENV = ("OLLAMA_HOST", "OLLAMA_MODELS")     # set here, never by a configuration


def port_listeners(port: int) -> list[int]:
    """server_guard.parse_netstat_listeners on a hidden netstat (server_guard.port_listeners uses a bare subprocess)."""
    try:
        out = run_hidden(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True, timeout=20).stdout
    except Exception:
        out = ""
    return sg.parse_netstat_listeners(out or "", port)


def ollama_exe() -> str:
    import host_config as hc
    return hc._resolve_ollama_exe_for_serve()


def models_dir() -> str | None:
    import host_config as hc
    return os.environ.get("OLLAMA_MODELS") or hc._this_host_entry().get("ollama_models")


class LaneServer:
    def __init__(self, port: int, server_env: dict, log_path: str, popen=popen_hidden, listeners=port_listeners,
                 ready_timeout_s: float = 90.0):
        bad = [k for k in server_env if k in FORBIDDEN_ENV]
        if bad:
            raise ValueError(f"server_env may not set {bad}")
        self.port, self.server_env, self.log_path = port, dict(server_env), str(log_path)
        self.popen, self.listeners, self.ready_timeout_s = popen, listeners, ready_timeout_s
        self.proc = None
        self.base = f"http://127.0.0.1:{port}"
        self.record = {}

    def env(self) -> dict:
        e = {k: v for k, v in os.environ.items() if not k.startswith("OLLAMA_")}
        e.update(DEBUG_ENV)
        e.update(self.server_env)
        e["OLLAMA_HOST"] = f"127.0.0.1:{self.port}"
        md = models_dir()
        if md:
            e["OLLAMA_MODELS"] = md
        return e

    def start(self) -> dict:
        held = self.listeners(self.port)
        if held:
            raise sg.GuardError(f"port {self.port} already has a listener {held}. refusing; not reusing it.",
                                {"guard": "port_not_free", "port": self.port, "listeners": held})
        free = {"guard": "port_free", "port": self.port}
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        log = open(self.log_path, "wb")
        self.proc = self.popen([ollama_exe(), "serve"], env=self.env(), stdout=log, stderr=log)
        deadline = time.monotonic() + self.ready_timeout_s
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(self.base + "/api/version", timeout=3) as r:
                    version = __import__("json").loads(r.read()).get("version")
                    break
            except Exception:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"ollama serve on port {self.port} exited early (log {self.log_path})")
                time.sleep(0.5)
        else:
            self.stop()
            raise RuntimeError(f"ollama serve on port {self.port} not ready in {self.ready_timeout_s} s")
        pids = self.listeners(self.port)
        if pids and self.proc.pid not in pids:
            self.stop()
            raise sg.GuardError(f"port {self.port}: listener {pids} is not the server we started ({self.proc.pid})")
        self.record = {"port": self.port, "pid": self.proc.pid, "server_env": self.server_env, "debug": True,
                       "log": self.log_path, "ollama_version": version, **free}
        return self.record

    def check(self) -> bool:
        """Before every request: the listener on the port is still the process we started."""
        pids = self.listeners(self.port)
        return bool(self.proc) and self.proc.poll() is None and (not pids or self.proc.pid in pids)

    def stop(self):
        if self.proc is None:
            return
        if self.proc.poll() is None:
            try:
                run_hidden(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"], capture_output=True, timeout=30)
            except Exception:
                self.proc.kill()
        try:
            self.proc.wait(timeout=15)
        except Exception:
            pass
        self.proc = None
