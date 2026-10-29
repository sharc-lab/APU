"""INV-1 raw allocate-until-failure probe (one process, one OpenVINO GPU context).

Question: is the "bigger GPU share -> less context" inversion OpenVINO policy or the platform?
This tool allocates fixed-size chunks through the OpenVINO GPU remote context until the first
exception (or --max-gib) and records memory readouts after every fully touched chunk.

Modes (all through ctx = ov.Core().get_default_context("GPU")):
  device  ctx.create_tensor(u8, [n], {"SHARED_MEM_TYPE": "USM_DEVICE_BUFFER"})  (USM device)
  host    ctx.create_host_tensor(u8, [n])                                       (USM host)
  ocl     ctx.create_tensor(u8, [n], {})                (plugin default buffer for empty params)

Touch: device/ocl copy_from an ov.Tensor wrapping a numpy uint8 buffer filled with 0xA5; the
host buffer is dropped right after (no host copies are retained, unlike tools/gpu_balloon.py).
host mode writes 0xA5 through a numpy view of the host tensor.

The ceiling is the cumulative bytes of fully touched chunks. A failed create or touch is the
measurement: the exception type, message (4000 chars) and step ("create" / "touch") are
recorded and the exit code is 0. Exit 2 only when the GPU context cannot be created.
The JSON is written atomically (tmp file, then replace) and one line
"ALLOC_RESULT {summary}" is printed. It changes no setting.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MIB = 1024 * 1024
GIB = 1024 * MIB
MODES = ("device", "host", "ocl")
FILL = 0xA5
MSG_CHARS = 4000
HEAD_CHARS = 300
USM_DEVICE_PROPS = {"SHARED_MEM_TYPE": "USM_DEVICE_BUFFER"}


class AllocBackend(Protocol):
    """Allocation backend: create returns a held object, touch writes every byte of it."""

    def create(self, n_bytes: int) -> Any: ...

    def touch(self, obj: Any, n_bytes: int) -> None: ...

    def start_info(self) -> dict[str, Any]: ...

    def row_info(self) -> dict[str, Any]: ...


def _err(exc: BaseException, n: int = 300) -> str:
    return f"{type(exc).__name__}: {exc}"[:n]


class OpenVinoBackend:
    """Real backend (Windows XPS). openvino and numpy are imported lazily."""

    def __init__(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        import openvino as ov

        self.mode = mode
        self._ov: Any = ov
        self.core: Any = ov.Core()
        self.ctx: Any = self.core.get_default_context("GPU")
        self.first_params: Any = None

    def create(self, n_bytes: int) -> Any:
        ov = self._ov
        shape = ov.Shape([n_bytes])
        if self.mode == "host":
            t = self.ctx.create_host_tensor(ov.Type.u8, shape)
        elif self.mode == "device":
            t = self.ctx.create_tensor(ov.Type.u8, shape, dict(USM_DEVICE_PROPS))
        else:
            t = self.ctx.create_tensor(ov.Type.u8, shape, {})
        if self.first_params is None and self.mode != "host":
            try:
                self.first_params = {str(k): str(v) for k, v in dict(t.get_params()).items()}
            except Exception as exc:  # recorded, never raised
                self.first_params = {"error": _err(exc)}
        return t

    def touch(self, obj: Any, n_bytes: int) -> None:
        import numpy as np

        if self.mode == "host":
            view = np.asarray(obj.data).reshape(-1)
            if view.size != n_bytes:
                raise RuntimeError(f"host tensor view has {view.size} bytes, expected {n_bytes}")
            view[:] = FILL
            return
        buf = np.full(n_bytes, FILL, dtype=np.uint8)
        try:
            obj.copy_from(self._ov.Tensor(buf))
        finally:
            del buf  # no retained host copy

    def _stats(self) -> dict[str, Any]:
        try:
            raw = self.core.get_property("GPU", "GPU_MEMORY_STATISTICS")
            return {str(k): int(v) for k, v in dict(raw).items()}
        except Exception as exc:  # recorded, never raised
            return {"error": _err(exc)}

    def _prop(self, name: str) -> Any:
        try:
            return int(self.core.get_property("GPU", name))
        except Exception as exc:  # recorded, never raised
            return {"error": _err(exc)}

    def start_info(self) -> dict[str, Any]:
        from seam.dxgi_budget import snapshot

        try:
            dx: Any = snapshot()
        except Exception as exc:  # recorded, never raised
            dx = {"error": _err(exc)}
        return {
            "openvino_version": str(getattr(self._ov, "__version__", "unknown")),
            "GPU_DEVICE_TOTAL_MEM_SIZE": self._prop("GPU_DEVICE_TOTAL_MEM_SIZE"),
            "GPU_DEVICE_MAX_ALLOC_MEM_SIZE": self._prop("GPU_DEVICE_MAX_ALLOC_MEM_SIZE"),
            "gpu_memory_statistics": self._stats(),
            "dxgi_snapshot": dx,
        }

    def row_info(self) -> dict[str, Any]:
        from seam.dxgi_budget import primary_budget

        try:
            dx: Any = primary_budget()
        except Exception as exc:  # recorded, never raised
            dx = {"error": _err(exc)}
        return {"gpu_memory_statistics": self._stats(), "dxgi": dx}


def host_memory() -> dict[str, Any]:
    """psutil available MB and this process's private bytes (when the platform has them)."""
    out: dict[str, Any] = {}
    try:
        import psutil

        out["available_mb"] = round(psutil.virtual_memory().available / MIB, 1)
        mi = psutil.Process(os.getpid()).memory_info()
        priv = getattr(mi, "private", None)
        out["private_bytes"] = int(priv) if priv is not None else None
    except Exception as exc:  # recorded, never raised
        out["error"] = _err(exc)
    return out


@dataclass
class ProbeResult:
    mode: str
    chunk_bytes: int
    max_bytes: int
    label: str
    start: dict[str, Any] = field(default_factory=dict)
    rows: list[dict[str, Any]] = field(default_factory=list)
    ceiling_bytes: int = 0
    stop_reason: str = ""
    failed_step: str | None = None
    failed_chunk: int | None = None
    exception_type: str | None = None
    exception_message: str | None = None
    t_start: float = 0.0
    t_end: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "tool": "inv_alloc_probe",
            "label": self.label,
            "mode": self.mode,
            "chunk_bytes": self.chunk_bytes,
            "chunk_mib": self.chunk_bytes / MIB,
            "max_bytes": self.max_bytes,
            "ceiling_bytes": self.ceiling_bytes,
            "ceiling_gib": round(self.ceiling_bytes / GIB, 4),
            "stop_reason": self.stop_reason,
            "failed_step": self.failed_step,
            "failed_chunk": self.failed_chunk,
            "exception_type": self.exception_type,
            "exception_message": self.exception_message,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "start": self.start,
            "rows": self.rows,
            **self.extra,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "chunk_mib": self.chunk_bytes / MIB,
            "ceiling_bytes": self.ceiling_bytes,
            "ceiling_gib": round(self.ceiling_bytes / GIB, 4),
            "stop_reason": self.stop_reason,
            "exception_type": self.exception_type,
            "exception_head": (self.exception_message or "")[:HEAD_CHARS] or None,
        }


def run_probe(
    backend: AllocBackend,
    *,
    mode: str,
    chunk_bytes: int,
    max_bytes: int,
    label: str = "",
    host_mem: Callable[[], dict[str, Any]] = host_memory,
    clock: Callable[[], float] = time.time,
) -> ProbeResult:
    """Allocate + touch chunks until the first exception or max_bytes. Holds every chunk."""
    if chunk_bytes <= 0 or max_bytes <= 0:
        raise ValueError("chunk_bytes and max_bytes must be > 0")
    res = ProbeResult(mode=mode, chunk_bytes=chunk_bytes, max_bytes=max_bytes, label=label)
    res.t_start = clock()
    res.start = {**backend.start_info(), "host": host_mem()}
    held: list[Any] = []
    i = 0
    while res.ceiling_bytes < max_bytes:
        size = min(chunk_bytes, max_bytes - res.ceiling_bytes)
        step = "create"
        try:
            obj = backend.create(size)
            step = "touch"
            backend.touch(obj, size)
        except Exception as exc:  # the failure is the measurement; recorded in full
            res.stop_reason = "exception"
            res.failed_step = step
            res.failed_chunk = i
            res.exception_type = type(exc).__name__
            res.exception_message = str(exc)[:MSG_CHARS]
            break
        held.append(obj)
        res.ceiling_bytes += size
        res.rows.append(
            {
                "i": i,
                "chunk_bytes": size,
                "cumulative_bytes": res.ceiling_bytes,
                "t_unix": clock(),
                **backend.row_info(),
                **host_mem(),
            }
        )
        i += 1
    else:
        res.stop_reason = "max_gib"
    res.extra["chunks_held"] = len(held)
    first_params = getattr(backend, "first_params", None)
    if first_params is not None:
        res.extra["first_tensor_params"] = first_params
    res.t_end = clock()
    held.clear()
    return res


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def summary_line(res: ProbeResult) -> str:
    return "ALLOC_RESULT " + json.dumps(res.summary(), sort_keys=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--chunk-mib", type=int, default=256)
    ap.add_argument("--max-gib", type=float, default=16.0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", default="")
    return ap


def main(
    argv: list[str] | None = None,
    backend_factory: Callable[[str], AllocBackend] | None = None,
) -> int:
    a = build_parser().parse_args(argv)
    if a.chunk_mib <= 0 or a.max_gib <= 0:
        print("ALLOC_FAILED chunk-mib and max-gib must be > 0", flush=True)
        return 2
    factory: Callable[[str], AllocBackend] = backend_factory or OpenVinoBackend
    try:
        backend = factory(a.mode)
    except Exception as exc:
        print("ALLOC_FAILED " + json.dumps({"error": _err(exc, MSG_CHARS)}), flush=True)
        return 2
    res = run_probe(
        backend,
        mode=a.mode,
        chunk_bytes=a.chunk_mib * MIB,
        max_bytes=int(a.max_gib * GIB),
        label=a.label,
    )
    atomic_write_json(a.out, res.to_json())
    print(summary_line(res), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
