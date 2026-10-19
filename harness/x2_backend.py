"""--backend vulkan for harness/x2_r2_agent.py (2026-10-19, x2_r2_llama_val_32k_vulkan, pre-registered in
docs/FINDINGS.md "PRE-REGISTRATION: backend numerics check ...").

The job starts its own `ollama serve` process (never the tray app, never a persistent or system-wide variable) with
BACKEND_ENVS[backend] set for that process only, and its log in a file of the job's own (<stem>_serve.log next to
--out, the same naming as x2_num_gpu). Why these variables, from the offload job's own server log on evo-x2 (Ollama
0.34.4): Vulkan is enabled in this build (OLLAMA_VULKAN:true in "server config"), discovery finds the 8060S through
Vulkan but drops it ("dropping integrated GPU; to enable, set OLLAMA_IGPU_ENABLE=1" ... library=Vulkan) and keeps the
ROCm device. So: OLLAMA_IGPU_ENABLE=1 keeps the Vulkan iGPU, OLLAMA_LLM_LIBRARY=vulkan loads only lib\\ollama\\vulkan,
HIP_VISIBLE_DEVICES=-1 hides the device from ROCm as a second guard.

What the backend actually was is read from the log, never assumed:

  run level   check_server_backend(): every "inference compute" line says library=Vulkan and at least one names the
              expected device (8060S). The caller refuses the run (backend_refused row, no model call) otherwise.
  per session BackendRecorder writes a backend_session row from the session's log slice: offload lines, the "gpu
              memory" lines' library= values, Vulkan device lines, "<device> model buffer size" lines, any line naming
              ROCm or CUDA, and the last /api/ps size / size_vram. session_verified() is the pre-registered check.

The verdict that uses these rows is analysis/numbers_register.py::x2_r2_llama_val_32k_vulkan_verdict.
"""
from __future__ import annotations

import re
import time

from x2_num_gpu import parse_offloaded, read_from, server_log_for, tray_app_running  # noqa: F401  (re-exported)

BACKENDS = ("vulkan",)
BACKEND_ENVS = {
    "vulkan": {"OLLAMA_IGPU_ENABLE": "1", "OLLAMA_LLM_LIBRARY": "vulkan", "HIP_VISIBLE_DEVICES": "-1"},
}
BACKEND_LIBRARY = {"vulkan": "Vulkan"}      # the library= value Ollama logs for it
EXPECTED_DEVICE = "8060S"                   # evo-x2's GPU, in the "inference compute" description
N_LAYERS = 33                               # llama3.1:8b: 32 blocks + output layer
FORBIDDEN_RE = re.compile(r"\bROCm\d*\b|\bCUDA\d*\b|library=(?:ROCm|CUDA|cuda|rocm)")

INFERENCE_RE = re.compile(r'msg="inference compute"(.*)')
KV_RE = re.compile(r'(\w+)=("([^"]*)"|\S+)')
GPU_MEMORY_LIB_RE = re.compile(r'msg="gpu memory".*?\blibrary=(\S+)')
VK_DEVICE_RE = re.compile(r"ggml_vulkan: (\d+) = ([^\r\n|]+)")
BUFFER_RE = re.compile(r"\b([A-Za-z_]+\d*) model buffer size")
LOADED_BACKEND_RE = re.compile(r"load_backend: loaded (\S+) backend")
NO_GPU_RE = re.compile(r"no compatible GPUs were discovered|entering low vram mode", re.I)


def _kv(s: str) -> dict:
    return {k: (q if q is not None and v.startswith('"') else v) for k, v, q in
            ((m.group(1), m.group(2), m.group(3)) for m in KV_RE.finditer(s))}


def parse_inference_compute(text: str) -> list[dict]:
    """Every "inference compute" line as {library, name, description, compute, type, total}."""
    out = []
    for m in INFERENCE_RE.finditer(text or ""):
        kv = _kv(m.group(1))
        out.append({k: kv.get(k) for k in ("library", "name", "description", "compute", "type", "total")})
    return out


def check_server_backend(text: str, backend: str, device: str = EXPECTED_DEVICE) -> dict:
    """The pre-registered run-level check on the server log's head: ok when there is at least one "inference compute"
    line, every one has library == BACKEND_LIBRARY[backend], and at least one description contains `device`."""
    want = BACKEND_LIBRARY[backend]
    lines = parse_inference_compute(text)
    reasons = []
    if not lines:
        reasons.append("no inference compute line in the server log"
                       + (" (log says no GPU)" if NO_GPU_RE.search(text or "") else ""))
    else:
        other = sorted({str(x.get("library")) for x in lines if x.get("library") != want})
        if other:
            reasons.append(f"inference compute on library {other}, expected {want}")
        if not any(device in (x.get("description") or "") for x in lines):
            reasons.append(f"no inference compute line names {device}")
    return {"ok": not reasons, "reasons": reasons, "inference_compute": lines, "expected_library": want,
            "expected_device": device}


def wait_for_inference_compute(path, timeout_s: float = 60.0, poll_s: float = 1.0, sleep=time.sleep,
                               clock=time.monotonic) -> str:
    """The server log's text once it holds an "inference compute" line or a no-GPU line (GPU discovery can finish
    after the HTTP listener is up), or whatever it holds when timeout_s elapses."""
    deadline = clock() + timeout_s
    while True:
        text, _ = read_from(path, 0)
        if INFERENCE_RE.search(text) or NO_GPU_RE.search(text) or clock() >= deadline:
            return text
        sleep(poll_s)


def session_evidence(text: str) -> dict:
    return {"offloaded": parse_offloaded(text),
            "gpu_memory_libraries": sorted({x.strip('"') for x in GPU_MEMORY_LIB_RE.findall(text or "")}),
            "vulkan_devices": sorted({f"{i} = {d.strip()}" for i, d in VK_DEVICE_RE.findall(text or "")}),
            "buffer_devices": sorted(set(BUFFER_RE.findall(text or ""))),
            "loaded_backends": sorted(set(LOADED_BACKEND_RE.findall(text or ""))),
            "forbidden_lines": [ln.strip()[:300] for ln in (text or "").splitlines() if FORBIDDEN_RE.search(ln)][:20]}


def session_verified(ev: dict, backend: str = "vulkan", n_layers: int = N_LAYERS) -> bool:
    """Pre-registered per session: at least one offload line, every one n_layers/n_layers, at least one line naming
    the backend (a library=Vulkan gpu-memory line or a Vulkan0 buffer line), no line naming ROCm or CUDA. The
    "load_backend: loaded ... backend" lines are recorded (loaded_backends) but, as pre-registered, not part of it."""
    want = BACKEND_LIBRARY[backend]
    off = ev.get("offloaded") or []
    names_it = want in (ev.get("gpu_memory_libraries") or []) or any(
        b.startswith(want) for b in (ev.get("buffer_devices") or []))
    return (bool(off) and all(n == n_layers and m == n_layers for n, m in off) and names_it
            and not ev.get("forbidden_lines"))


class BackendRuntime:
    """Wraps x2_r2_agent.OllamaRuntime: keeps the last /api/ps size and size_vram of the model on every
    loaded_context call. Request bodies are not changed. Everything else is delegated."""

    def __init__(self, inner, backend: str):
        if backend not in BACKENDS:
            raise ValueError(f"--backend must be one of {BACKENDS}, got {backend!r}")
        self._inner = inner
        self.backend = backend
        self.last_ps = None

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def loaded_context(self, model):
        try:
            for m in self._inner.client.get_ps().get("models", []):
                n = m.get("name") or m.get("model")
                if n and (n if ":" in n else n + ":latest") == (model if ":" in model else model + ":latest"):
                    self.last_ps = {"size": m.get("size"), "size_vram": m.get("size_vram"),
                                    "context_length": m.get("context_length")}
        except Exception:
            pass
        return self._inner.loaded_context(model)


class BackendRecorder:
    """on_session_start / on_session_end hooks for x2_r2_agent.run_plan."""

    def __init__(self, runtime: BackendRuntime, server_log, emit, log=print):
        self.rt, self.server_log, self.emit, self.log = runtime, server_log, emit, log
        self.current = None
        _, self.offset = read_from(server_log, 0)

    def start(self, model, arm, seed):
        _, self.offset = read_from(self.server_log, self.offset)   # skip anything logged between sessions
        self.rt.last_ps = None
        self.current = (model, arm, seed)

    def end(self):
        text, self.offset = read_from(self.server_log, self.offset)
        model, arm, seed = self.current or (None, None, None)
        ev = session_evidence(text)
        ps = self.rt.last_ps or {}
        row = {"record": "backend_session", "model_id": model, "arm_id": arm, "seed": seed,
               "backend": self.rt.backend, **ev, "verified": session_verified(ev, self.rt.backend),
               "ps_size": ps.get("size"), "ps_size_vram": ps.get("size_vram")}
        self.emit(row)
        self.log(f"  backend {self.rt.backend}: verified {row['verified']} offloaded {ev['offloaded']} "
                 f"libraries {ev['gpu_memory_libraries']} buffers {ev['buffer_devices']}")
        self.current = None
        return row
