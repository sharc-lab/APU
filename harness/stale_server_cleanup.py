"""Shared stale-process cleanup for Ollama/llama-server, used by (a) every job's own start/exit path
and (b) the queue watchdog's circuit breaker (2026-10-02, built after a real crash cascade on evo-x2:
a job died with its own server still running, and every subsequent job that needed the llama-server
runtime inherited the same stale-process-guard trip, burning through 5 queued jobs in under an hour).

Ownership rule differs by host, by design:
  - evo-x2 is a machine dedicated to this project -- every ollama.exe/"ollama app.exe"/llama-server.exe
    process on it is ours, so a bare name match is the correct and complete ownership test.
  - evo-t2s is not exclusively ours (an earlier incident on this same project found another
    interactive session on it); a process is only "ours" there if its command line points into
    C:\\apu (our own install/model/working directories) or binds one of our known ports. A plain
    name match is NOT a safe ownership test on evo-t2s.

Every process-listing/killing/port-check call is injected so this is fully unit-testable without a
real machine."""
from __future__ import annotations

import time

OLLAMA_PROCESS_NAMES = ("ollama.exe", "ollama app.exe", "llama-server.exe")


def real_list_processes():
    import psutil
    out = []
    for p in psutil.process_iter(["pid", "ppid", "name", "cmdline"]):
        try:
            info = p.info
            out.append({"pid": info["pid"], "ppid": info.get("ppid"), "name": info.get("name") or "",
                       "cmdline": " ".join(info.get("cmdline") or [])})
        except Exception:
            continue
    return out


def real_kill_pid(pid):
    import psutil
    try:
        psutil.Process(pid).kill()
        return True
    except Exception:
        return False


def real_port_free(port, host="127.0.0.1"):
    import socket
    try:
        with socket.create_connection((host, port), timeout=1):
            return False  # something answered -> not free
    except OSError:
        return True


def is_ours_dedicated_machine(proc):
    """evo-x2's ownership rule: name match is sufficient."""
    return proc["name"] in OLLAMA_PROCESS_NAMES


DEFAULT_SHARED_MACHINE_PATHS = (
    "c:\\apu",  # our own working dir, models, and bundled llama-server builds
    "\\ollama\\lib\\ollama\\llama-server.exe",  # Ollama's own internal engine, wherever Ollama itself
                                                # is installed (its own install dir, not ours to name) --
                                                # found live 2026-10-02: 16 of these had accumulated on
                                                # evo-t2s, invisible to a "c:\\apu"-only path check since
                                                # Ollama's internal engine never references C:\apu at all,
                                                # even though we are the only Ollama user on this machine.
    "\\.ollama\\models\\blobs\\",  # the model blob path Ollama's engine is invoked with, same reasoning
)


def is_ours_shared_machine(proc, our_paths=DEFAULT_SHARED_MACHINE_PATHS, our_ports=()):
    """evo-t2s's ownership rule: name match AND (command line references one of our paths, or binds
    one of our known ports). A bare ollama.exe/llama-server.exe with neither is left alone -- it may
    belong to the machine's other interactive user."""
    if proc["name"] not in OLLAMA_PROCESS_NAMES:
        return False
    cmdline = proc["cmdline"].lower()
    if any(p.lower() in cmdline for p in our_paths):
        return True
    if our_ports and any(f"--port {p}" in cmdline or f"--port={p}" in cmdline for p in our_ports):
        return True
    return False


def cleanup_stale_servers(is_ours_fn, *, list_processes=real_list_processes, kill_pid=real_kill_pid,
                          port_free=real_port_free, ports_to_wait_for=(), wait_s=15, poll_s=1,
                          sleep=time.sleep, log=print):
    """Kills every process matching is_ours_fn(proc), then waits up to wait_s for each of
    ports_to_wait_for to actually free up (a killed process can hold its listening socket in
    TIME_WAIT briefly). Returns {"killed": [...], "ports_freed": {port: bool}}. Logs what it killed
    (log callable, default print) -- the whole point is this is never silent."""
    procs = list_processes()
    targets = [p for p in procs if is_ours_fn(p)]
    killed = []
    for p in targets:
        log(f"stale_server_cleanup: killing pid={p['pid']} name={p['name']} cmdline={p['cmdline'][:200]!r}")
        if kill_pid(p["pid"]):
            killed.append(p["pid"])
    ports_freed = {}
    for port in ports_to_wait_for:
        t0 = 0.0
        freed = port_free(port)
        while not freed and t0 < wait_s:
            sleep(poll_s)
            t0 += poll_s
            freed = port_free(port)
        ports_freed[port] = freed
        if not freed:
            log(f"stale_server_cleanup: port {port} still not free after {wait_s}s")
    return {"killed": killed, "ports_freed": ports_freed}
