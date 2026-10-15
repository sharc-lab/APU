"""--num-gpu N for harness/x2_r2_agent.py (2026-10-14, x2_r2_llama_val_32k_offload, pre-registered in docs/FINDINGS.md).

Every /api/chat request of the run carries options.num_gpu = N (Ollama: the number of model layers placed on the GPU),
through OllamaRuntime.options_extra, and the job's own Ollama server writes its log to a file of the job's own
(OFFLOAD_LOG_SUFFIX next to --out). Per session this module records what the server actually did:

  num_gpu_offload row: model, arm, seed, num_gpu sent, every "offloaded N/M layers to GPU" line in the session's log
                       slice, the number of chat calls and how many carried num_gpu, and the last /api/ps size and
                       size_vram seen while the model was loaded (read on every loaded_context call).

The backend library ("inference compute" line, library=...) is read from the log once the server is up. The verdict
that uses these rows is analysis/numbers_register.py::x2_r2_llama_val_32k_offload_verdict.
"""
from __future__ import annotations

import re
from pathlib import Path

OFFLOAD_RE = re.compile(r"offloaded (\d+)/(\d+) layers to GPU")
LIBRARY_RE = re.compile(r'msg="inference compute".*?\blibrary=(\S+)')
OFFLOAD_LOG_SUFFIX = "_serve.log"


def server_log_for(out_path) -> str:
    """The job's own server log, next to --out: <stem>_serve.log."""
    p = Path(out_path)
    return str(p.parent / (p.stem + OFFLOAD_LOG_SUFFIX))


def parse_offloaded(text: str) -> list[list[int]]:
    return [[int(a), int(b)] for a, b in OFFLOAD_RE.findall(text or "")]


def parse_libraries(text: str) -> list[str]:
    return sorted({m.strip('"') for m in LIBRARY_RE.findall(text or "")})


def read_from(path, offset: int) -> tuple[str, int]:
    """Text of `path` from byte `offset` on, and the new end offset ("" and offset when the file is missing). A file
    shorter than `offset` was truncated by a server restart (cmd.exe `>`), so it is read from the start."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            if f.tell() < offset:
                offset = 0
            f.seek(offset)
            data = f.read()
    except OSError:
        return "", offset
    return data.decode("utf-8", errors="replace"), offset + len(data)


class NumGpuRuntime:
    """Wraps x2_r2_agent.OllamaRuntime: sets options_extra {"num_gpu": N} on it, counts chat calls, and keeps the last
    /api/ps size and size_vram of the model on every loaded_context call. Everything else is delegated."""

    def __init__(self, inner, num_gpu: int):
        if int(num_gpu) < 0:
            raise ValueError(f"--num-gpu must be >= 0, got {num_gpu}")
        self._inner = inner
        self.num_gpu = int(num_gpu)
        inner.options_extra = {**(getattr(inner, "options_extra", None) or {}), "num_gpu": self.num_gpu}
        self.reset_counts()

    def reset_counts(self):
        self.n_chat = 0
        self.n_chat_num_gpu = 0
        self.last_ps = None

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def chat(self, model, messages, num_ctx, tools, think, extra=None):
        self.n_chat += 1
        if (getattr(self._inner, "options_extra", None) or {}).get("num_gpu") == self.num_gpu:
            self.n_chat_num_gpu += 1
        return self._inner.chat(model, messages, num_ctx, tools, think, extra)

    def loaded_context(self, model):
        try:
            ps = self._inner.client.get_ps()
            for m in ps.get("models", []):
                if _same_tag(m.get("name") or m.get("model"), model):
                    self.last_ps = {"size": m.get("size"), "size_vram": m.get("size_vram"),
                                    "context_length": m.get("context_length")}
        except Exception:
            pass
        return self._inner.loaded_context(model)


def _same_tag(a, b) -> bool:
    norm = (lambda t: t if (t and ":" in t) else (t or "") + ":latest")
    return a is not None and b is not None and norm(a) == norm(b)


class OffloadRecorder:
    """on_session_start / on_session_end hooks for x2_r2_agent.run_plan."""

    def __init__(self, runtime: NumGpuRuntime, server_log, emit, log=print):
        self.rt, self.server_log, self.emit, self.log = runtime, server_log, emit, log
        self.offset = 0
        self.current = None
        _, self.offset = read_from(server_log, 0)

    def start(self, model, arm, seed):
        _, self.offset = read_from(self.server_log, self.offset)   # skip anything logged between sessions
        self.rt.reset_counts()
        self.current = (model, arm, seed)

    def end(self):
        text, self.offset = read_from(self.server_log, self.offset)
        model, arm, seed = self.current or (None, None, None)
        off = parse_offloaded(text)
        ps = self.rt.last_ps or {}
        row = {"record": "num_gpu_offload", "model_id": model, "arm_id": arm, "seed": seed,
               "num_gpu_sent": self.rt.num_gpu, "offloaded": off, "n_chat_calls": self.rt.n_chat,
               "n_chat_calls_num_gpu": self.rt.n_chat_num_gpu, "ps_size": ps.get("size"),
               "ps_size_vram": ps.get("size_vram"),
               "ps_vram_share": (ps["size_vram"] / ps["size"]) if ps.get("size") and ps.get("size_vram") is not None
               else None, "libraries_in_slice": parse_libraries(text)}
        self.emit(row)
        self.log(f"  num_gpu {self.rt.num_gpu}: offloaded {off} vram share {row['ps_vram_share']}")
        self.current = None
        return row


def split_as_intended(offloaded, num_gpu: int, n_layers: int, tol: int) -> bool:
    """The pre-registered check for one session: at least one offload line, every line N/M with M == n_layers and
    |N - num_gpu| <= tol."""
    return bool(offloaded) and all(m == n_layers and abs(n - num_gpu) <= tol for n, m in offloaded)
