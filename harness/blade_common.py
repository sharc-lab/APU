"""Shared pieces for the Blade (Razer Blade 14, RTX 4070 Laptop 8 GB, CUDA) night jobs. docs/BLADE_PLAN.md is the plan.

The Blade is also the controller laptop, so every child process here goes through harness/proc_util (hidden, no
console window; CLAUDE.md "No visible windows on the controller").

  PINNED            the versions every Blade job records in its run_start row and checks before measuring
  versions_record   reads the installed versions (nvidia-smi, ollama --version, llama-server --version, the CUDA
                    runtime DLLs the llama.cpp build ships, the CUDA runner dirs Ollama bundles)
  version_problems  pinned vs installed; a real (non dry-run) job refuses to measure on any problem
  LocalOllama       starts `ollama serve` as a hidden child with its own env and log file, waits for /api/tags,
                    and stops every Ollama process plus Ollama's own llama-server runners (path under an Ollama dir),
                    never the harness's own llama-server
  DryRunClock       the 60-second dry-run budget: emit() raises DryRunTimeUp once the budget is spent and at least
                    one row has been written
"""
from __future__ import annotations

import json
import os
import re
import socket
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
DRYRUN_DIR = RESULTS / "blade_dryrun"
BLADE_DIR = Path(os.environ.get("APU_BLADE_DIR", r"C:\apu\blade"))   # queue state, flags, server logs
LOG_DIR = BLADE_DIR / "logs"
BLADE_HW_ID = "blade"

PINNED = {
    # Ollama: the same release as evo-x2. Installed side by side (docs/BLADE_PLAN.md "Versions"), never over the
    # user's tray install. Release asset ollama-windows-amd64.zip of tag v0.34.4, sha256 from the GitHub release.
    "ollama_version": "0.34.4",
    "ollama_exe": r"C:\apu\bin\ollama-0.34.4\ollama.exe",
    "ollama_asset": "https://github.com/ollama/ollama/releases/download/v0.34.4/ollama-windows-amd64.zip",
    "ollama_asset_bytes": 1461155106,
    "ollama_asset_sha256": "535193f38f3344e5b08f5d1c171c31ce11aa17f0124ff69ae26d8ec7fe06fa62",
    # llama.cpp: build b10970 (commit bfdc32183), the same build as evo-x2 (Vulkan there) and Blade C1/C2. CUDA 12.4
    # asset plus its cudart package, already in C:\apu\bin and matching the release digests (checked 2026-10-08).
    "llama_cpp_build": "b10970",
    "llama_cpp_commit": "bfdc32183",
    "llama_cpp_asset": "llama-b10970-bin-win-cuda-12.4-x64.zip",
    "llama_cpp_asset_sha256": "78c878ae30622a9e4be09e3831066454668ca70398114f23bf74ac814e52dad8",
    "cudart_asset": "cudart-llama-bin-win-cuda-12.4-x64.zip",
    "cudart_asset_sha256": "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6",
    "llama_server_exe": r"C:\apu\bin\llama-b10970-cuda\llama-server.exe",
    "llama_tokenize_exe": r"C:\apu\bin\llama-b10970-cuda\llama-tokenize.exe",
    "llama_cuda_runtime": "12.4",
    # NVIDIA driver as installed for C1/C2 (FINDINGS "C1, Blade") and read again 2026-10-08.
    "nvidia_driver": "610.88",
}

OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_STORE = str(Path.home() / ".ollama" / "models")


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run(argv, timeout=30, run=None):
    if run is None:
        from proc_util import run_hidden as run
    try:
        p = run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
                stdin=__import__("subprocess").DEVNULL)
        return p.returncode, (p.stdout or ""), (p.stderr or "")
    except Exception as e:  # missing exe, timeout
        return None, "", repr(e)[:300]


def ps(cmd: str, timeout=30, run=None) -> str:
    rc, out, _ = _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd], timeout, run)
    return out


# ── host ────────────────────────────────────────────────────────────────────────────────────────────

def require_blade(hostname=None) -> dict:
    import host_config as hc
    name = (hostname or socket.gethostname()).upper()
    h = hc.HOSTS.get(name)
    if not h or h.get("hw_id") != BLADE_HW_ID:
        raise SystemExit(f"Blade jobs run on the Blade only (host_config hw_id 'blade'); this is {name!r}")
    return h


# ── versions ────────────────────────────────────────────────────────────────────────────────────────

_CUDA_UMD = re.compile(r"CUDA (?:UMD )?Version:\s*([\d.]+)")
_OLLAMA_VER = re.compile(r"(?:client )?version is (\d+\.\d+\.\d+\S*)")
_LLAMA_VER = re.compile(r"version:\s*(\S+)\s*\(build (\d+), commit (\w+)\)")


def parse_nvidia_smi_header(text: str) -> dict:
    m = _CUDA_UMD.search(text or "")
    d = re.search(r"Driver Version:\s*([\d.]+)|NVIDIA-SMI\s+([\d.]+)", text or "")
    return {"cuda_driver_api": m.group(1) if m else None,
            "nvidia_driver_from_header": (d.group(1) or d.group(2)) if d else None}


def parse_ollama_version(text: str) -> str | None:
    """`ollama --version` prints 'ollama version is X' (and 'Warning: client version is Y' when no server runs)."""
    vals = _OLLAMA_VER.findall(text or "")
    return vals[0] if vals else None


def parse_llama_version(text: str) -> dict:
    m = _LLAMA_VER.search(text or "")
    return {"version": m.group(1), "build": m.group(2), "commit": m.group(3)} if m else {}


def versions_record(ollama_exe=None, run=None) -> dict:
    """Read-only: nothing here loads a model or starts a server (`ollama --version` only talks to a running one)."""
    ollama_exe = ollama_exe or PINNED["ollama_exe"]
    rec = {"captured_utc": utc_iso(), "pinned": dict(PINNED)}
    rc, out, err = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,vbios_version",
                         "--format=csv,noheader"], 20, run)
    rec["gpu_query"] = out.strip() if rc == 0 else f"ERROR rc={rc} {err[:200]}"
    parts = [p.strip() for p in out.split(",")] if rc == 0 else []
    rec["nvidia_driver"] = parts[1] if len(parts) > 1 else None
    rec["vram_total_mib"] = int(re.sub(r"\D", "", parts[2])) if len(parts) > 2 and re.search(r"\d", parts[2]) else None
    rc, out, _ = _run(["nvidia-smi"], 20, run)
    rec.update(parse_nvidia_smi_header(out))
    rc, out, err = _run([ollama_exe, "--version"], 30, run)
    rec["ollama_exe"] = ollama_exe
    rec["ollama_version_raw"] = (out + err).strip()[:300]
    rec["ollama_version"] = parse_ollama_version(out + err)
    lib = Path(ollama_exe).parent / "lib" / "ollama"
    rec["ollama_bundled_cuda_runners"] = sorted(p.name for p in lib.glob("cuda_v*")) if lib.is_dir() else []
    rc, out, err = _run([PINNED["llama_server_exe"], "--version"], 60, run)
    rec["llama_server_version"] = parse_llama_version(out + err)
    bindir = Path(PINNED["llama_server_exe"]).parent
    rec["llama_cuda_runtime_dlls"] = sorted(p.name for p in bindir.glob("cudart64_*.dll")) if bindir.is_dir() else []
    return rec


def version_problems(rec: dict) -> list[str]:
    p = []
    if rec.get("ollama_version") != PINNED["ollama_version"]:
        p.append(f"ollama {rec.get('ollama_version')!r} at {rec.get('ollama_exe')} != pinned {PINNED['ollama_version']}")
    lv = rec.get("llama_server_version") or {}
    if lv.get("build") != PINNED["llama_cpp_build"].lstrip("b") or lv.get("commit") != PINNED["llama_cpp_commit"]:
        p.append(f"llama-server {lv} != pinned build {PINNED['llama_cpp_build']} commit {PINNED['llama_cpp_commit']}")
    if rec.get("nvidia_driver") != PINNED["nvidia_driver"]:
        p.append(f"NVIDIA driver {rec.get('nvidia_driver')!r} != pinned {PINNED['nvidia_driver']}")
    if "cudart64_12.dll" not in (rec.get("llama_cuda_runtime_dlls") or []):
        p.append("llama.cpp build has no cudart64_12.dll next to llama-server.exe (CUDA 12.4 runtime expected)")
    return p


# ── GPU memory ──────────────────────────────────────────────────────────────────────────────────────

def gpu_memory(run=None) -> dict:
    rc, out, _ = _run(["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free",
                       "--format=csv,noheader,nounits"], 20, run)
    try:
        t, u, f = (int(x.strip()) for x in out.strip().splitlines()[0].split(","))
        return {"vram_total_mib": t, "vram_used_mib": u, "vram_free_mib": f}
    except Exception:
        return {"vram_total_mib": None, "vram_used_mib": None, "vram_free_mib": None}


# ── processes ───────────────────────────────────────────────────────────────────────────────────────

def list_processes(run=None) -> list[dict]:
    """[{pid, ppid, name, cmd, path}] for every process (PowerShell CIM, hidden)."""
    out = ps("Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,CommandLine,ExecutablePath"
             " | ConvertTo-Json -Compress", 60, run)
    try:
        data = json.loads(out) if out.strip() else []
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = [data]
    return [{"pid": d.get("ProcessId"), "ppid": d.get("ParentProcessId"), "name": (d.get("Name") or ""),
             "cmd": d.get("CommandLine") or "", "path": d.get("ExecutablePath") or ""} for d in data]


def is_ollama_process(p: dict) -> bool:
    n = p["name"].lower()
    if n in ("ollama.exe", "ollama app.exe"):
        return True
    return n == "llama-server.exe" and "\\ollama" in (p.get("path") or "").lower()


def ollama_processes(procs) -> list[dict]:
    return [p for p in procs if is_ollama_process(p)]


def harness_llama_servers(procs) -> list[dict]:
    return [p for p in procs if p["name"].lower() == "llama-server.exe" and not is_ollama_process(p)]


def kill_pid(pid: int, run=None):
    _run(["taskkill", "/F", "/T", "/PID", str(pid)], 30, run)


# ── local Ollama ────────────────────────────────────────────────────────────────────────────────────

class LocalOllama:
    """`ollama serve` as a hidden child: OLLAMA_KEEP_ALIVE=0 (the same server default as evo-x2's jobs; R2 sets a
    per-request keep_alive), OLLAMA_MODELS = the user store, optional extra env (OLLAMA_DEBUG for the mechanism and
    mitigation jobs). stop() is idempotent and also removes the tray app, which would otherwise respawn the server."""

    def __init__(self, exe=None, env=None, log_path=None, popen=None, run=None, procs_fn=None, base=OLLAMA_URL):
        self.exe = exe or PINNED["ollama_exe"]
        self.env = dict(env or {})
        self.log_path = Path(log_path) if log_path else LOG_DIR / f"ollama_serve_{int(time.time())}.log"
        self.popen, self.run = popen, run
        self.procs_fn = procs_fn or (lambda: list_processes(run))
        self.base = base
        self.proc = None
        self._log_fh = None

    def child_env(self) -> dict:
        env = dict(os.environ)
        env.update({"OLLAMA_KEEP_ALIVE": "0", "OLLAMA_MODELS": env.get("OLLAMA_MODELS") or OLLAMA_STORE,
                    "OLLAMA_HOST": "127.0.0.1:11434"})
        env.update(self.env)
        return env

    def start(self):
        self.stop()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        popen = self.popen
        if popen is None:
            from proc_util import popen_hidden as popen
        self._log_fh = open(self.log_path, "wb")
        import subprocess
        self.proc = popen([self.exe, "serve"], env=self.child_env(), stdout=self._log_fh,
                          stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        return self.proc.pid

    def wait_ready(self, timeout_s=90, poll_s=0.5) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                return False
            try:
                with urllib.request.urlopen(f"{self.base}/api/tags", timeout=3) as r:
                    if r.status == 200:
                        return True
            except Exception:
                pass
            time.sleep(poll_s)
        return False

    def stop(self, wait_s=20) -> list[dict]:
        """Kills every Ollama process (server, tray app, Ollama's own runners) and waits until none is listed."""
        killed = []
        deadline = time.monotonic() + wait_s
        while True:
            left = ollama_processes(self.procs_fn())
            if not left:
                break
            for p in left:
                kill_pid(p["pid"], self.run)
                killed.append({"pid": p["pid"], "name": p["name"]})
            if time.monotonic() > deadline:
                break
            time.sleep(1)
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except Exception:
                pass
            self._log_fh = None
        self.proc = None
        return killed

    def restart(self) -> bool:
        self.start()
        return self.wait_ready()


# ── dry runs ────────────────────────────────────────────────────────────────────────────────────────

class DryRunTimeUp(Exception):
    pass


class DryRunClock:
    """budget_s None = a real run (never raises). Otherwise check() raises once the budget is spent, and only after
    at least one result row was written, so a dry run always proves the job writes results."""

    def __init__(self, budget_s=None, clock=time.monotonic):
        self.budget_s, self.clock = budget_s, clock
        self.t0 = clock()
        self.rows = 0

    def wrote(self):
        self.rows += 1
        self.check()

    def check(self):
        if self.budget_s is not None and self.rows > 0 and self.clock() - self.t0 >= self.budget_s:
            raise DryRunTimeUp(f"dry run budget {self.budget_s}s spent after {self.rows} rows")


def jsonl_emitter(path: Path, clock: DryRunClock | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def emit(row: dict):
        row = dict(row)
        row.setdefault("ts_utc", utc_iso())
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
        if clock is not None:
            clock.wrote()
        return row
    return emit


def out_path_ok(path, dry_run: bool) -> Path:
    """Result files are named blade_*; dry-run output must live under results/blade_dryrun/."""
    p = Path(path)
    if not p.name.startswith("blade_"):
        raise SystemExit(f"Blade output files are prefixed blade_: {p.name!r}")
    if dry_run and DRYRUN_DIR.resolve() not in p.resolve().parents:
        raise SystemExit(f"a dry run writes under {DRYRUN_DIR}, not {p}")
    return p
