"""Context-window override methods on Ollama 0.34.4 (operator plan v2, 2026-10-13, Part A5). One script for evo-x2
(queue id x2_window_override_methods, --queue), the Blade (a Blade queue job, harness/blade_queue.py NIGHTS[4]) and
evo-t2s (an optional entry in scripts/t2s_week_queue.py OPTIONAL).

For llama3.1:8b, three ways to set the context window, each checked for (a) the context the server actually loaded,
from GET /api/ps context_length and from the server's own log, and (b) whether it persists across a restart of the
job's own server:
  1. request    : per-request options.num_ctx (V_REQUEST). After a restart, a plain request (no num_ctx).
  2. modelfile  : `ollama create <DERIVED_TAG> -f Modelfile` with "FROM llama3.1:8b" and "PARAMETER num_ctx
                  V_MODELFILE", a NEW tag (the job refuses if the tag already exists, and never modifies an existing
                  one); a plain request to the derived tag. After a restart, the derived tag again, plus the base tag as a
                  control (must still load the baseline). The derived tag is then deleted (`ollama rm`, only that tag;
                  a finally block retries if the run stopped in between).
  3. env        : OLLAMA_CONTEXT_LENGTH=V_ENV on the job's own server process; a plain request. Then a restart with
                  the same launch environment, and a restart without it.
A baseline plain request on a plain server comes first (Ollama's own default on this host); each override value
differs from it and from the others, so every loaded value can be attributed to one method.

The server: `ollama serve` started by this job as a hidden child process (harness/proc_util.popen_hidden) on its own
port (PORT, 11500, so it never meets a server on 11434), with OLLAMA_DEBUG=1, OLLAMA_MODELS = the host's store, and
every OLLAMA_* variable inherited from the job's environment removed (their names are recorded) so only this job's
settings apply. It is stopped with `taskkill /F /T /PID <its pid>` (its own process tree, which includes its runner);
no other process is touched, the Ollama tray app included. Refuses to start if the port is taken or if a server on
11434 holds a model (its GPU memory would change this host's default).

Read-only GPU memory record (gpu_memory_readonly): Win32_VideoController, the display-adapter class registry values
HardwareInformation.qwMemorySize / MemorySize (what dxdiag reports as dedicated memory), installed and visible RAM,
nvidia-smi memory.total where present, and Ollama's own "inference compute" and "vram-based default context" log
lines. Nothing is written or changed; the setting that sizes GPU memory per machine is documented in docs/HARDWARE.md.

Statistics convention: this is a configuration check, one request per step (no warm-up and no repeats; each value is
an exact read of the server's state, not a timing). Output rows: run_start, gpu_readonly, wom_server (one per server
start), wom_step (one per request), wom_tag (create / delete), run_end; summarize() gives the per-method verdicts
that analysis/numbers_register.py::compute_window_override_methods reports.

Usage: python window_override_methods.py --out results/x2_window_override_methods_v1.jsonl --queue      (evo-x2)
       py -3.12 harness/window_override_methods.py --out results/blade_window_override_methods_v1.jsonl  (Blade)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))

BASE_MODEL = "llama3.1:8b"
PORT = 11500
V_REQUEST, V_MODELFILE, V_ENV = 12288, 10240, 6144
DERIVED_TAG = "apu-wom-llama3.1:numctx10240"
EXPECTED_OLLAMA = "0.34.4"
KEEP_ALIVE = "10m"
REQUEST_TIMEOUT_S = 600
PROMPT = "Reply with the single word OK."
DEFAULT_PORT_URL = "http://127.0.0.1:11434"
METHODS = ("request", "modelfile", "env")


def utc_iso():
    return datetime.now(timezone.utc).isoformat()


# ── log parsing (pure) ──────────────────────────────────────────────────────────────────────────────

CTX_PATTERNS = {   # name -> regex; first group is the context length
    "n_ctx_slot": re.compile(r"n_ctx_slot\s*=\s*(\d+)"),
    "llama_context_n_ctx": re.compile(r"llama_context:\s+n_ctx\s*=\s*(\d+)"),
    "n_ctx_seq": re.compile(r"n_ctx_seq\s*(?:\([^)]*\))?\s*=\s*(\d+)"),
    "ctx_size_arg": re.compile(r"--ctx-size[= ](\d+)"),
    "kv_size": re.compile(r"KvSize:(\d+)"),
    "num_ctx_field": re.compile(r"\bnum_ctx[=:]\s*(\d+)"),
}
LOG_CTX_PRIORITY = ("n_ctx_slot", "llama_context_n_ctx", "n_ctx_seq", "ctx_size_arg", "kv_size", "num_ctx_field")
_ENV_CTX = re.compile(r"OLLAMA_CONTEXT_LENGTH:(\d*)")
_VRAM_DEFAULT = re.compile(r'vram-based default context"?.*?total_vram="([^"]+)".*?default_num_ctx=(\d+)')
_INFERENCE = re.compile(r'msg="?inference compute"?(.*)')
_KV = re.compile(r'(\w+)=("([^"]*)"|\S+)')
_GIN_CHAT = re.compile(r'\[GIN\].*"/api/chat"')


def parse_ctx(text: str) -> dict:
    """{pattern name: [values in order]} for every context-length line in `text`, and "log_ctx": the last value of
    the highest-priority pattern found (None if none)."""
    found = {}
    for name, pat in CTX_PATTERNS.items():
        vals = [int(m.group(1)) for m in pat.finditer(text or "")]
        if vals:
            found[name] = vals
    log_ctx = next((found[n][-1] for n in LOG_CTX_PRIORITY if n in found), None)
    return {"values": found, "log_ctx": log_ctx,
            "log_ctx_source": next((n for n in LOG_CTX_PRIORITY if n in found), None)}


def parse_server_log(text: str) -> dict:
    """Server-level lines: OLLAMA_CONTEXT_LENGTH in the "server config" line, the vram-based default context line,
    every "inference compute" line (library, name, description, type, total, available), debug level on."""
    env = [m.group(1) for m in _ENV_CTX.finditer(text or "")]
    vd = [{"total_vram": m.group(1), "default_num_ctx": int(m.group(2))} for m in _VRAM_DEFAULT.finditer(text or "")]
    gpus = []
    for m in _INFERENCE.finditer(text or ""):
        kv = {k: (q if q is not None and v.startswith('"') else v) for k, v, q in _KV.findall(m.group(1))}
        gpus.append({k: kv.get(k) for k in ("library", "name", "description", "type", "total", "available")})
    return {"config_ollama_context_length": env[-1] if env else None, "vram_based_default": vd,
            "inference_compute": gpus, "debug_on": "OLLAMA_DEBUG:DEBUG" in (text or "") or "level=DEBUG" in (text or "")}


def norm_tag(tag):
    return tag if not tag or ":" in tag else tag + ":latest"


def ps_entry(ps: dict, model: str) -> dict | None:
    for m in (ps or {}).get("models") or []:
        if norm_tag(m.get("name") or m.get("model")) == norm_tag(model):
            return m
    return None


# ── HTTP ────────────────────────────────────────────────────────────────────────────────────────────

class Api:
    def __init__(self, base):
        self.base = base

    def _req(self, path, body=None, timeout=30, method=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(f"{self.base}{path}", data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
                return r.status, (json.loads(raw) if raw else {}), None, time.monotonic() - t0
        except urllib.error.HTTPError as e:
            return e.code, None, e.read().decode(errors="replace")[:400], time.monotonic() - t0
        except Exception as e:
            return None, None, repr(e)[:400], time.monotonic() - t0

    def get(self, path, timeout=10):
        return self._req(path, timeout=timeout)

    def post(self, path, body, timeout=REQUEST_TIMEOUT_S):
        return self._req(path, body, timeout=timeout)

    def ps(self):
        st, data, err, _ = self.get("/api/ps")
        return data if data is not None else {"error": err, "status": st}

    def tags(self):
        st, data, _, _ = self.get("/api/tags")
        return [m.get("name") for m in (data or {}).get("models") or []] if data is not None else None

    def tag_digest(self, model):
        st, data, _, _ = self.get("/api/tags")
        for m in (data or {}).get("models") or []:
            if norm_tag(m.get("name")) == norm_tag(model):
                return m.get("digest")
        return None

    def version(self):
        st, data, _, _ = self.get("/api/version")
        return (data or {}).get("version")

    def show(self, model):
        st, data, err, _ = self.post("/api/show", {"model": model}, timeout=60)
        return data if data is not None else {"error": err, "status": st}

    def ready(self, timeout_s=90, poll_s=0.5, sleep=time.sleep, clock=time.monotonic):
        deadline = clock() + timeout_s
        while clock() < deadline:
            st, _, _, _ = self.get("/api/tags", timeout=3)
            if st == 200:
                return True
            sleep(poll_s)
        return False


def port_in_use(port, host="127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex((host, port)) == 0


# ── the job's own server ────────────────────────────────────────────────────────────────────────────

def child_env(base_env: dict, port: int, models_dir: str, extra: dict | None) -> tuple[dict, list[str]]:
    """The server's environment: base_env minus every OLLAMA_* variable (names returned), plus OLLAMA_HOST (own
    port), OLLAMA_MODELS, OLLAMA_DEBUG=1 and `extra` (method 3: OLLAMA_CONTEXT_LENGTH)."""
    stripped = sorted(k for k in base_env if k.upper().startswith("OLLAMA_"))
    env = {k: v for k, v in base_env.items() if not k.upper().startswith("OLLAMA_")}
    env.update({"OLLAMA_HOST": f"127.0.0.1:{port}", "OLLAMA_MODELS": models_dir, "OLLAMA_DEBUG": "1"})
    env.update(extra or {})
    return env, stripped


class OwnServer:
    """`ollama serve` as a hidden child on its own port. stop() kills only this process tree (taskkill /T)."""

    def __init__(self, exe, models_dir, port, log_dir: Path, popen=None, run=None, base_env=None):
        self.exe, self.models_dir, self.port, self.log_dir = exe, models_dir, port, Path(log_dir)
        self.popen, self.run = popen, run
        self.base_env = dict(os.environ if base_env is None else base_env)
        self.proc = None
        self.fh = None
        self.log_path = None
        self.n = 0
        self.extra = None

    def start(self, extra=None, label="plain") -> dict:
        import subprocess
        popen = self.popen
        if popen is None:
            from proc_util import popen_hidden as popen
        self.n += 1
        self.extra = dict(extra or {})
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f"serve_{self.n}_{label}.log"
        env, stripped = child_env(self.base_env, self.port, self.models_dir, self.extra)
        self.fh = open(self.log_path, "wb")
        self.proc = popen([self.exe, "serve"], env=env, stdout=self.fh, stderr=subprocess.STDOUT,
                          stdin=subprocess.DEVNULL)
        return {"instance": self.n, "label": label, "pid": self.proc.pid, "env_extra": self.extra,
                "inherited_ollama_env_removed": stripped, "log_file": self.log_path.name}

    def stop(self, wait_s=20, sleep=time.sleep) -> dict:
        out = {"pid": None, "killed": False}
        if self.proc is not None:
            out["pid"] = self.proc.pid
            run = self.run
            if run is None:
                from proc_util import run_hidden as run
            try:
                run(["taskkill", "/F", "/T", "/PID", str(self.proc.pid)], capture_output=True, text=True, timeout=60)
                out["killed"] = True
            except Exception as e:
                out["error"] = repr(e)[:200]
            try:
                self.proc.wait(timeout=wait_s)
            except Exception:
                pass
        if self.fh is not None:
            try:
                self.fh.close()
            except Exception:
                pass
        self.proc = self.fh = None
        deadline = time.monotonic() + wait_s
        while port_in_use(self.port) and time.monotonic() < deadline:
            sleep(0.5)
        out["port_free"] = not port_in_use(self.port)
        return out

    def log_size(self):
        try:
            return self.log_path.stat().st_size
        except (OSError, AttributeError):
            return 0

    def read_log(self, start=0, end=None):
        try:
            with open(self.log_path, "rb") as f:
                f.seek(start)
                data = f.read() if end is None else f.read(max(end - start, 0))
        except (OSError, TypeError):
            return ""
        return data.decode("utf-8", errors="replace")

    def cli_env(self):
        env, _ = child_env(self.base_env, self.port, self.models_dir, None)
        env.pop("OLLAMA_DEBUG", None)
        return env


# ── one measured request ────────────────────────────────────────────────────────────────────────────

def unload_all(api: Api, wait_s=30, sleep=time.sleep):
    """keep_alive 0 for every loaded model, then poll /api/ps until empty. Returns True when nothing is loaded."""
    for m in (api.ps().get("models") or []):
        api.post("/api/generate", {"model": m.get("name") or m.get("model"), "keep_alive": 0}, timeout=60)
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if not (api.ps().get("models") or []):
            return True
        sleep(0.5)
    return False


def chat_body(model, num_ctx=None):
    options = {"num_predict": 4, "temperature": 0, "seed": 42}
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    return {"model": model, "messages": [{"role": "user", "content": PROMPT}], "stream": False, "options": options,
            "keep_alive": KEEP_ALIVE}


def measure(api: Api, server: OwnServer, method, phase, model, num_ctx=None, expected=None, sleep=time.sleep,
            gin_wait_s=5.0) -> dict:
    """One plain chat request on a freshly unloaded server: what /api/ps and the server log say was loaded."""
    unloaded = unload_all(api, sleep=sleep)
    p0 = server.log_size()
    body = chat_body(model, num_ctx)
    st, data, err, dt = api.post("/api/chat", body)
    deadline = time.monotonic() + gin_wait_s
    while time.monotonic() < deadline and not _GIN_CHAT.search(server.read_log(p0)):
        sleep(0.2)
    ps = api.ps()
    ent = ps_entry(ps, model) or {}
    sl = server.read_log(p0)
    ctx = parse_ctx(sl)
    rec = {"record": "wom_step", "method": method, "phase": phase, "model": model, "server_instance": server.n,
           "server_env_extra": server.extra, "options_num_ctx_sent": num_ctx, "expected": expected,
           "http_status": st, "error": err, "duration_s": dt, "unloaded_before": unloaded,
           "prompt_eval_count": (data or {}).get("prompt_eval_count"),
           "ps_context_length": ent.get("context_length"), "ps_size": ent.get("size"),
           "ps_size_vram": ent.get("size_vram"), "ps_found": bool(ent),
           "log_ctx": ctx["log_ctx"], "log_ctx_source": ctx["log_ctx_source"], "log_ctx_values": ctx["values"],
           "log_excerpt": [ln.strip()[:400] for ln in sl.splitlines()
                           if any(p.search(ln) for p in CTX_PATTERNS.values())][:12],
           "ts_utc": utc_iso()}
    rec["applied"] = None if expected is None else rec["ps_context_length"] == expected
    rec["log_agrees_with_ps"] = (None if rec["log_ctx"] is None or rec["ps_context_length"] is None
                                 else rec["log_ctx"] == rec["ps_context_length"])
    unload_all(api, sleep=sleep)
    return rec


# ── derived tag (method 2) ──────────────────────────────────────────────────────────────────────────

def modelfile_text(base=BASE_MODEL, num_ctx=V_MODELFILE) -> str:
    return f"FROM {base}\nPARAMETER num_ctx {num_ctx}\n"


def create_tag(server: OwnServer, tag, base, num_ctx, run=None, tmp_dir=None) -> dict:
    if run is None:
        from proc_util import run_hidden as run
    td = Path(tempfile.mkdtemp(prefix="wom_", dir=tmp_dir))
    mf = td / "Modelfile"
    mf.write_text(modelfile_text(base, num_ctx), encoding="utf-8")
    p = run([server.exe, "create", tag, "-f", str(mf)], env=server.cli_env(), capture_output=True, text=True,
            timeout=600)
    return {"record": "wom_tag", "action": "create", "tag": tag, "from": base, "modelfile": modelfile_text(base, num_ctx),
            "returncode": p.returncode, "stdout_tail": (p.stdout or "")[-400:], "stderr_tail": (p.stderr or "")[-400:],
            "ts_utc": utc_iso()}


def delete_tag(server: OwnServer, tag, run=None) -> dict:
    if run is None:
        from proc_util import run_hidden as run
    p = run([server.exe, "rm", tag], env=server.cli_env(), capture_output=True, text=True, timeout=120)
    return {"record": "wom_tag", "action": "delete", "tag": tag, "returncode": p.returncode,
            "stdout_tail": (p.stdout or "")[-400:], "stderr_tail": (p.stderr or "")[-400:], "ts_utc": utc_iso()}


# ── read-only GPU memory record ─────────────────────────────────────────────────────────────────────

GPU_QUERIES = {
    "video_controller": "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM,DriverVersion,"
                        "VideoProcessor | ConvertTo-Json -Compress",
    "display_class_registry": "Get-ChildItem 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Class\\"
                              "{4d36e968-e325-11ce-bfc1-08002be10318}' -ErrorAction SilentlyContinue | ForEach-Object "
                              "{ $p = Get-ItemProperty $_.PSPath -ErrorAction SilentlyContinue; if ($p.DriverDesc) { "
                              "[pscustomobject]@{key=$_.PSChildName; DriverDesc=$p.DriverDesc; qwMemorySize="
                              "$p.'HardwareInformation.qwMemorySize'; MemorySize=$p.'HardwareInformation.MemorySize'} } }"
                              " | ConvertTo-Json -Compress",
    "visible_ram_bytes": "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
    "installed_ram_bytes": "(Get-CimInstance Win32_PhysicalMemory | Measure-Object -Property Capacity -Sum).Sum",
}


def gpu_memory_readonly(run=None) -> dict:
    """Read-only queries (nothing is set or written). Each value is the raw output, or an error string."""
    if run is None:
        from proc_util import run_hidden as run
    out = {}
    for name, cmd in GPU_QUERIES.items():
        try:
            p = run(["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd], capture_output=True, text=True,
                    timeout=60)
            txt = (p.stdout or "").strip()
            try:
                out[name] = json.loads(txt) if txt[:1] in "[{" else txt
            except json.JSONDecodeError:
                out[name] = txt[:2000]
        except Exception as e:
            out[name] = f"ERROR {e!r}"[:300]
    try:
        p = run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True,
                text=True, timeout=30)
        out["nvidia_smi"] = (p.stdout or "").strip() if p.returncode == 0 else None
    except Exception:
        out["nvidia_smi"] = None
    return out


# ── summary / verdict (pure) ────────────────────────────────────────────────────────────────────────

def _step(rows, method, phase):
    return next((r for r in reversed(rows) if r.get("record") == "wom_step" and r.get("method") == method
                 and r.get("phase") == phase), None)


def summarize(rows) -> dict:
    """Per method: the loaded context (/api/ps and log) when applied and after the restart(s), applied / persists
    booleans; validity (baseline distinct from every override value, every request HTTP 200, derived tag deleted)."""
    base = _step(rows, "baseline", "plain")
    d = (base or {}).get("ps_context_length")

    def ctx(r):
        return None if r is None else {"ps": r.get("ps_context_length"), "log": r.get("log_ctx"),
                                       "log_source": r.get("log_ctx_source"), "http": r.get("http_status")}
    m1a, m1r = _step(rows, "request", "apply"), _step(rows, "request", "after_restart")
    m2a, m2r = _step(rows, "modelfile", "apply"), _step(rows, "modelfile", "after_restart")
    m2c = _step(rows, "modelfile", "base_control_after_restart")
    m3a, m3s = _step(rows, "env", "apply"), _step(rows, "env", "after_restart_same_env")
    m3w = _step(rows, "env", "after_restart_without_env")

    def eq(r, v):
        return None if r is None or r.get("ps_context_length") is None else r["ps_context_length"] == v
    tags = [r for r in rows if r.get("record") == "wom_tag"]
    created = any(t["action"] == "create" and t.get("returncode") == 0 for t in tags)
    deleted = any(t["action"] == "delete" and t.get("returncode") == 0 for t in tags)
    steps = [r for r in rows if r.get("record") == "wom_step"]
    problems = []
    if d is None:
        problems.append("no baseline context")
    elif d in (V_REQUEST, V_MODELFILE, V_ENV):
        problems.append(f"baseline {d} equals an override value")
    bad = [f"{r['method']}/{r['phase']}" for r in steps if r.get("http_status") != 200]
    if bad:
        problems.append(f"non-200 steps {bad}")
    if created and not deleted:
        problems.append(f"derived tag {DERIVED_TAG} created but not deleted")
    if not any(r.get("record") == "run_end" for r in rows):
        problems.append("no run_end")
    out = {
        "baseline": ctx(base),
        "request": {"value": V_REQUEST, "apply": ctx(m1a), "after_restart": ctx(m1r), "applied": eq(m1a, V_REQUEST),
                    "persists_across_restart": eq(m1r, V_REQUEST)},
        "modelfile": {"value": V_MODELFILE, "tag": DERIVED_TAG, "apply": ctx(m2a), "after_restart": ctx(m2r),
                      "base_control_after_restart": ctx(m2c), "applied": eq(m2a, V_MODELFILE),
                      "persists_across_restart": eq(m2r, V_MODELFILE),
                      "base_tag_unchanged": None if d is None else eq(m2c, d),
                      "tag_created": created, "tag_deleted": deleted},
        "env": {"value": V_ENV, "apply": ctx(m3a), "after_restart_same_env": ctx(m3s),
                "after_restart_without_env": ctx(m3w), "applied": eq(m3a, V_ENV),
                "persists_restart_same_env": eq(m3s, V_ENV), "persists_restart_without_env": eq(m3w, V_ENV)},
        "log_agrees_with_ps": {f"{r['method']}/{r['phase']}": r.get("log_agrees_with_ps") for r in steps},
        "valid": not problems, "problems": problems,
    }
    servers = [r for r in rows if r.get("record") == "wom_server"]
    out["vram_based_default"] = [v for s in servers for v in (s.get("server_log") or {}).get("vram_based_default", [])][:1]
    out["inference_compute"] = next((s["server_log"]["inference_compute"] for s in servers
                                     if (s.get("server_log") or {}).get("inference_compute")), [])
    out["env_config_values"] = {s["instance"]: (s.get("server_log") or {}).get("config_ollama_context_length")
                                for s in servers}
    return out


# ── the run ─────────────────────────────────────────────────────────────────────────────────────────

def host_defaults(hostname: str) -> dict:
    """ollama exe and models store for this host (Blade: the pinned side-by-side 0.34.4 and the user store;
    evo-t2s: the operator's ollama_pin.json via t2s_week_preflight, then host_config; evo-x2: host_config)."""
    import host_config as hc
    h = hc.HOSTS.get(hostname.upper()) or {}
    if h.get("hw_id") == "blade":
        import blade_common as bc
        return {"hw_id": "blade", "ollama_exe": bc.PINNED["ollama_exe"], "models_dir": bc.OLLAMA_STORE}
    if h.get("hw_id") == "evo-t2s":
        try:
            import t2s_week_preflight as pf
            pf.apply_ollama_pin()
        except Exception:
            pass
    entry = hc._this_host_entry()
    return {"hw_id": h.get("hw_id"), "ollama_exe": hc._resolve_ollama_exe_for_serve(),
            "models_dir": os.environ.get("OLLAMA_MODELS") or entry.get("ollama_models")}


def run_methods(server: OwnServer, api: Api, emit, log=print, run=None, sleep=time.sleep, tmp_dir=None):
    """The fixed sequence (module docstring). The caller has checked the port and the version policy. Returns
    summarize(rows) of what was emitted (without run_end)."""
    rows = []

    def em(r):
        rows.append(r)
        emit(r)

    def start(extra=None, label="plain"):
        info = server.start(extra, label)
        ok = api.ready(sleep=sleep)
        info.update({"record": "wom_server", "ready": ok, "version": api.version() if ok else None,
                     "ts_utc": utc_iso()})
        info["server_log"] = parse_server_log(server.read_log(0))
        em(info)
        if not ok:
            raise RuntimeError(f"own Ollama server (instance {info['instance']}) did not become ready")
        if info["version"] != EXPECTED_OLLAMA:
            raise RuntimeError(f"own Ollama server reports {info['version']!r}, expected {EXPECTED_OLLAMA}")
        return info

    def restart(extra=None, label="plain"):
        em({"record": "wom_stop", **server.stop(sleep=sleep), "ts_utc": utc_iso()})
        return start(extra, label)

    created = deleted = False
    try:
        start(None, "plain")
        if not any(norm_tag(t) == norm_tag(BASE_MODEL) for t in (api.tags() or [])):
            raise RuntimeError(f"{BASE_MODEL} is not in this host's Ollama store")
        base_digest = api.tag_digest(BASE_MODEL)
        em(measure(api, server, "baseline", "plain", BASE_MODEL, sleep=sleep))
        log(f"baseline: {rows[-1]['ps_context_length']}")
        em(measure(api, server, "request", "apply", BASE_MODEL, num_ctx=V_REQUEST, expected=V_REQUEST, sleep=sleep))
        restart(None, "plain")
        em(measure(api, server, "request", "after_restart", BASE_MODEL, expected=V_REQUEST, sleep=sleep))
        if any(norm_tag(t) == norm_tag(DERIVED_TAG) for t in (api.tags() or [])):
            raise RuntimeError(f"derived tag {DERIVED_TAG} already exists; refusing to modify an existing tag")
        rec = create_tag(server, DERIVED_TAG, BASE_MODEL, V_MODELFILE, run=run, tmp_dir=tmp_dir)
        created = rec["returncode"] == 0
        sh = api.show(DERIVED_TAG)
        rec["show_parameters"] = (sh or {}).get("parameters")
        rec["base_digest_before"] = base_digest
        em(rec)
        if not created:
            raise RuntimeError(f"ollama create {DERIVED_TAG} failed: {rec['stderr_tail']!r}")
        em(measure(api, server, "modelfile", "apply", DERIVED_TAG, expected=V_MODELFILE, sleep=sleep))
        restart(None, "plain")
        em(measure(api, server, "modelfile", "after_restart", DERIVED_TAG, expected=V_MODELFILE, sleep=sleep))
        em(measure(api, server, "modelfile", "base_control_after_restart", BASE_MODEL, sleep=sleep))
        rec = delete_tag(server, DERIVED_TAG, run=run)
        deleted = rec["returncode"] == 0
        rec["tag_listed_after"] = any(norm_tag(t) == norm_tag(DERIVED_TAG) for t in (api.tags() or []))
        rec["base_digest_after"] = api.tag_digest(BASE_MODEL)
        rec["base_digest_unchanged"] = rec["base_digest_after"] == base_digest
        em(rec)
        env3 = {"OLLAMA_CONTEXT_LENGTH": str(V_ENV)}
        restart(env3, "env")
        em(measure(api, server, "env", "apply", BASE_MODEL, expected=V_ENV, sleep=sleep))
        restart(env3, "env")
        em(measure(api, server, "env", "after_restart_same_env", BASE_MODEL, expected=V_ENV, sleep=sleep))
        restart(None, "plain")
        em(measure(api, server, "env", "after_restart_without_env", BASE_MODEL, expected=V_ENV, sleep=sleep))
    finally:
        if created and not deleted:
            try:
                if server.proc is None:
                    start(None, "cleanup")
                rec = delete_tag(server, DERIVED_TAG, run=run)
                rec["cleanup_in_finally"] = True
                em(rec)
            except Exception as e:
                em({"record": "wom_tag", "action": "delete", "tag": DERIVED_TAG, "returncode": None,
                    "error": repr(e)[:300], "cleanup_in_finally": True, "ts_utc": utc_iso()})
        em({"record": "wom_stop", **server.stop(sleep=sleep), "ts_utc": utc_iso()})
    return summarize(rows)


def build_arg_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--queue", action="store_true",
                    help="evo-x2 / evo-t2s queue job: call t2s_queue.advance() exactly once on exit")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--ollama-exe", default=None, help="default: this host's pinned / resolved Ollama 0.34.4")
    return ap


def main(argv=None, deps=None):
    """deps (tests): {"hostname", "host_defaults", "server", "api", "default_api", "gpu", "port_in_use", "run",
    "sleep", "interactive"}."""
    args = build_arg_parser().parse_args(argv)
    deps = deps or {}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def emit(row):
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def log(msg):
        print(f"[{utc_iso()}] {msg}", flush=True)

    note = "completed"
    try:
        hostname = deps.get("hostname") or socket.gethostname()
        import host_config as hc
        try:
            cfg = hc.require_host(hostname)
        except SystemExit as e:
            raise RuntimeError(str(e)) from None
        if cfg.get("hw_id") in ("evo-x2", "evo-t2s"):
            try:
                interactive = (deps["interactive"](cfg) if "interactive" in deps
                               else hc.enforce_or_record_interactive_session(cfg))
            except SystemExit as e:
                note = f"stopped: {e}"[:400]
                return note
        else:
            interactive = None
        hd = (deps.get("host_defaults") or host_defaults)(hostname)
        exe = args.ollama_exe or hd["ollama_exe"]
        in_use = deps.get("port_in_use") or port_in_use
        default_api = deps.get("default_api") or Api(DEFAULT_PORT_URL)
        other = default_api.ps()
        emit({"record": "run_start", "host": hostname, "hw_id": hd.get("hw_id"), "ollama_exe": exe,
              "models_dir": hd.get("models_dir"), "port": args.port, "model": BASE_MODEL,
              "values": {"request": V_REQUEST, "modelfile": V_MODELFILE, "env": V_ENV}, "derived_tag": DERIVED_TAG,
              "expected_ollama": EXPECTED_OLLAMA, "interactive_session": interactive,
              "server_on_11434_ps": other, "ts_utc": utc_iso()})
        emit({"record": "gpu_readonly", **(deps.get("gpu") or gpu_memory_readonly)(), "ts_utc": utc_iso()})
        if in_use(args.port):
            raise RuntimeError(f"port {args.port} is in use; the job runs only on its own free port")
        if (other.get("models") or []):
            raise RuntimeError("a server on 11434 holds a model; its GPU memory would change this host's default")
        server = deps.get("server") or OwnServer(exe, hd["models_dir"], args.port, out.parent / (out.stem + "_logs"),
                                                 run=deps.get("run"))
        api = deps.get("api") or Api(f"http://127.0.0.1:{args.port}")
        summ = run_methods(server, api, emit, log, run=deps.get("run"), sleep=deps.get("sleep") or time.sleep)
        log(json.dumps(summ, default=str)[:3000])
    except Exception as e:
        import traceback
        note = f"stopped: {e!r}"[:400]
        log(note)
        traceback.print_exc()
    finally:
        emit({"record": "run_end", "note": note, "ts_utc": utc_iso()})
        if args.queue:
            try:
                import t2s_queue as tq
                tq.advance(note)
            except Exception as e:
                log(f"queue advance failed: {e!r}")
    return note


if __name__ == "__main__":
    main()
