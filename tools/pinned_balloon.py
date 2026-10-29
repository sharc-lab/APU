"""INV-1 pinned (non-trimmable) balloon side process.

--cpu: VirtualAlloc(MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) --gib bytes, raise this process's
  working set with SetProcessWorkingSetSizeEx(min = bytes + 256 MiB, max = bytes + 512 MiB,
  flags 0), write every page, then VirtualLock the region. Every return value and GetLastError
  is recorded. If VirtualLock fails the tool prints BALLOON_FAILED and exits 2 (no fallback to
  an unlocked balloon).
--gpu: allocate --gib in 256 MiB USM device f32 chunks through the OpenVINO GPU context
  (ctx.create_tensor(f32, [n], {"SHARED_MEM_TYPE": "USM_DEVICE_BUFFER"}), touched with copy_from
  of a 0xA5 host buffer that is dropped right after). A tiny model
  (Parameter f32 [n] -> Gather(arange(0, n, 1024), axis 0) -> ReduceSum -> Result) is compiled ON
  THE SAME remote context with core.compile_model(model, ctx, {}), and one infer per chunk runs
  every --touch-period-ms (infer_request.set_tensor(input_name, remote_chunk); the Python
  set_tensor overload that accepts a RemoteTensor takes the tensor NAME, not a port), so every
  chunk is read by a GPU kernel every period (one f32 per 4 KiB page). More than 10 consecutive
  infer failures print BALLOON_FAILED and exit 2.

After setup: "BALLOON_READY {json}". On the stop file (or --max-hours): "BALLOON_RELEASED {json}"
and exit 0. Any setup failure: "BALLOON_FAILED <text>" and exit 2. It changes no setting.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MIB = 1024 * 1024
GIB = 1024 * MIB
GPU_CHUNK = 256 * MIB
PAGE = 4096
WS_MIN_EXTRA = 256 * MIB
WS_MAX_EXTRA = 512 * MIB
MEM_COMMIT = 0x1000
MEM_RESERVE = 0x2000
MEM_RELEASE = 0x8000
PAGE_READWRITE = 0x04
GATHER_STRIDE = 1024  # f32 elements = 4 KiB, one element per page
MAX_CONSEC_ERRORS = 10
FILL = 0xA5
USM_DEVICE_PROPS = {"SHARED_MEM_TYPE": "USM_DEVICE_BUFFER"}


def _err(exc: BaseException, n: int = 500) -> str:
    return f"{type(exc).__name__}: {exc}"[:n]


def _emit(tag: str, payload: Any) -> None:
    text = payload if isinstance(payload, str) else json.dumps(payload, sort_keys=True, default=str)
    print(f"{tag} {text}", flush=True)


# ---------------------------------------------------------------- CPU (Windows ctypes)


class CpuPlatform(Protocol):
    """Win32 calls the CPU balloon needs; each returns (ok_or_value, GetLastError)."""

    def virtual_alloc(self, n_bytes: int) -> tuple[int, int]: ...

    def set_working_set(self, min_bytes: int, max_bytes: int) -> tuple[bool, int]: ...

    def touch_pages(self, addr: int, n_bytes: int) -> None: ...

    def virtual_lock(self, addr: int, n_bytes: int) -> tuple[bool, int]: ...

    def release(self, addr: int, n_bytes: int) -> None: ...


class WinCpuPlatform:
    """Real kernel32 calls. Construct only on Windows."""

    def __init__(self) -> None:
        import ctypes

        if sys.platform != "win32":
            raise OSError("the CPU pinned balloon needs Windows (VirtualLock)")
        self._ct = ctypes
        k32: Any = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        k32.VirtualAlloc.restype = ctypes.c_void_p
        k32.VirtualAlloc.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        k32.VirtualLock.restype = ctypes.c_int
        k32.VirtualLock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        k32.VirtualUnlock.restype = ctypes.c_int
        k32.VirtualUnlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        k32.VirtualFree.restype = ctypes.c_int
        k32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32]
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetProcessWorkingSetSizeEx.restype = ctypes.c_int
        k32.SetProcessWorkingSetSizeEx.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_uint32,
        ]
        self._k32: Any = k32

    def _last(self) -> int:
        return int(self._ct.get_last_error())  # type: ignore[attr-defined]

    def virtual_alloc(self, n_bytes: int) -> tuple[int, int]:
        addr = self._k32.VirtualAlloc(None, n_bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)
        return (int(addr or 0), self._last())

    def set_working_set(self, min_bytes: int, max_bytes: int) -> tuple[bool, int]:
        ok = self._k32.SetProcessWorkingSetSizeEx(
            self._k32.GetCurrentProcess(), min_bytes, max_bytes, 0
        )
        return (bool(ok), self._last())

    def touch_pages(self, addr: int, n_bytes: int) -> None:
        import numpy as np

        arr = (self._ct.c_ubyte * n_bytes).from_address(addr)
        view = np.ctypeslib.as_array(arr)
        view[::PAGE] = FILL

    def virtual_lock(self, addr: int, n_bytes: int) -> tuple[bool, int]:
        ok = self._k32.VirtualLock(addr, n_bytes)
        return (bool(ok), self._last())

    def release(self, addr: int, n_bytes: int) -> None:
        self._k32.VirtualUnlock(addr, n_bytes)
        self._k32.VirtualFree(addr, 0, MEM_RELEASE)


@dataclass
class CpuSetup:
    addr: int
    n_bytes: int
    info: dict[str, Any]


def cpu_setup(plat: CpuPlatform, n_bytes: int) -> CpuSetup:
    """Alloc, raise the working set, touch, lock. Raises RuntimeError on any failed step."""
    addr, err = plat.virtual_alloc(n_bytes)
    if not addr:
        raise RuntimeError(f"VirtualAlloc failed GetLastError={err}")
    ws_min, ws_max = n_bytes + WS_MIN_EXTRA, n_bytes + WS_MAX_EXTRA
    ws_ok, ws_err = plat.set_working_set(ws_min, ws_max)
    info: dict[str, Any] = {
        "working_set_result": {"ok": ws_ok, "last_error": ws_err, "min": ws_min, "max": ws_max}
    }
    plat.touch_pages(addr, n_bytes)
    lk_ok, lk_err = plat.virtual_lock(addr, n_bytes)
    info["locked"] = {"ok": lk_ok, "last_error": lk_err}
    if not lk_ok:
        plat.release(addr, n_bytes)
        raise RuntimeError(
            "VirtualLock failed GetLastError="
            f"{lk_err} (SetProcessWorkingSetSizeEx ok={ws_ok} GetLastError={ws_err})"
        )
    return CpuSetup(addr=addr, n_bytes=n_bytes, info=info)


# ---------------------------------------------------------------- GPU (OpenVINO)


class GpuBalloon:
    """USM device chunks kept resident by a Gather kernel compiled on the same remote context."""

    def __init__(self, n_bytes: int, chunk_bytes: int = GPU_CHUNK) -> None:
        import numpy as np
        import openvino as ov

        self.ov: Any = ov
        self.core: Any = ov.Core()
        self.ctx: Any = self.core.get_default_context("GPU")
        self.chunks: list[Any] = []
        self.sizes: list[int] = []
        left = n_bytes
        while left > 0:
            size = min(chunk_bytes, left)
            size -= size % 4
            if size <= 0:
                break
            n_el = size // 4
            t = self.ctx.create_tensor(ov.Type.f32, ov.Shape([n_el]), dict(USM_DEVICE_PROPS))
            host = np.full(size, FILL, dtype=np.uint8).view(np.float32)
            t.copy_from(ov.Tensor(host))
            del host  # no retained host copy
            self.chunks.append(t)
            self.sizes.append(n_el)
            left -= size
        self.bytes = 4 * sum(self.sizes)
        self.opset_name, self._ops = self._opset()
        self.requests: dict[int, tuple[Any, str]] = {}
        for n_el in sorted(set(self.sizes)):
            compiled = self.core.compile_model(self._model(n_el), self.ctx, {})
            name = compiled.input(0).get_any_name()
            self.requests[n_el] = (compiled.create_infer_request(), name)

    @staticmethod
    def _opset() -> tuple[str, Any]:
        import importlib

        name = "openvino.opset13"
        return name, importlib.import_module(name)

    def _model(self, n_el: int) -> Any:
        import numpy as np

        ops, ov = self._ops, self.ov
        param = ops.parameter([n_el], ov.Type.f32, name="x")
        idx = ops.constant(np.arange(0, n_el, GATHER_STRIDE, dtype=np.int64))
        g = ops.gather(param, idx, ops.constant(np.int64(0)))
        s = ops.reduce_sum(g, ops.constant(np.array([0], dtype=np.int64)), False)
        return ov.Model([ops.result(s)], [param], "inv_pinned_gather")

    def infer_chunk(self, i: int) -> None:
        req, name = self.requests[self.sizes[i]]
        req.set_tensor(name, self.chunks[i])
        req.infer()

    def release(self) -> None:
        self.requests.clear()
        self.chunks.clear()


# ---------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    kind = ap.add_mutually_exclusive_group(required=True)
    kind.add_argument("--gpu", action="store_true")
    kind.add_argument("--cpu", action="store_true")
    ap.add_argument("--gib", type=float, required=True)
    ap.add_argument("--stop-file", type=Path, required=True)
    ap.add_argument("--max-hours", type=float, default=8.0)
    ap.add_argument("--touch-period-ms", type=float, default=10.0)
    return ap


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = build_parser()
    a = ap.parse_args(argv)
    if not a.gib > 0:
        ap.error("--gib must be > 0")
    if not a.max_hours > 0:
        ap.error("--max-hours must be > 0")
    if not a.touch_period_ms > 0:
        ap.error("--touch-period-ms must be > 0")
    return a


def _counters() -> dict[str, Any]:
    from seam.turn_memory import read_gpu_counters

    try:
        return read_gpu_counters(os.getpid())
    except Exception as exc:  # recorded, never raised
        return {"error": _err(exc)}


def run_cpu(
    a: argparse.Namespace,
    plat_factory: Callable[[], CpuPlatform],
    counters: Callable[[], dict[str, Any]],
    sleep: Callable[[float], None],
) -> int:
    n = int(a.gib * GIB)
    try:
        setup = cpu_setup(plat_factory(), n)
    except Exception as exc:
        _emit("BALLOON_FAILED", _err(exc))
        return 2
    _emit(
        "BALLOON_READY",
        {
            "method": "cpu_virtuallock",
            "bytes": n,
            "chunks": 1,
            "period_ms": None,
            "pid": os.getpid(),
            "gpu_counters_after": counters(),
            **setup.info,
        },
    )
    t0 = time.time()
    t_end = t0 + a.max_hours * 3600
    while not a.stop_file.exists() and time.time() < t_end:
        sleep(1.0)
    _emit(
        "BALLOON_RELEASED",
        {
            "n_infers": 0,
            "errors": 0,
            "held_s": round(time.time() - t0, 1),
            "reason": "stop_file" if a.stop_file.exists() else "max_hours",
        },
    )
    return 0


def run_gpu(
    a: argparse.Namespace,
    gpu_factory: Callable[[int], Any],
    counters: Callable[[], dict[str, Any]],
    sleep: Callable[[float], None],
) -> int:
    n = int(a.gib * GIB)
    try:
        bal = gpu_factory(n)
        bal.infer_chunk(0)
    except Exception as exc:
        _emit("BALLOON_FAILED", _err(exc))
        return 2
    _emit(
        "BALLOON_READY",
        {
            "method": "gpu_usm_device_gather_kernel",
            "bytes": int(bal.bytes),
            "chunks": len(bal.chunks),
            "period_ms": a.touch_period_ms,
            "pid": os.getpid(),
            "compile": "core.compile_model(model, ctx, {}) on the default GPU remote context",
            "opset": getattr(bal, "opset_name", None),
            "infer_ok_first": True,
            "gpu_counters_after": counters(),
        },
    )
    n_infers, errors, consec = 1, 0, 0
    last_error: str | None = None
    t0 = time.time()
    t_end = t0 + a.max_hours * 3600
    period = a.touch_period_ms / 1000.0
    while not a.stop_file.exists() and time.time() < t_end:
        for i in range(len(bal.chunks)):
            try:
                bal.infer_chunk(i)
                n_infers += 1
                consec = 0
            except Exception as exc:  # counted; more than MAX_CONSEC_ERRORS in a row fails
                errors += 1
                consec += 1
                last_error = _err(exc)
                if consec > MAX_CONSEC_ERRORS:
                    _emit(
                        "BALLOON_FAILED",
                        f"{consec} consecutive infer failures; last: {last_error}",
                    )
                    return 2
        sleep(period)
    held_s = round(time.time() - t0, 1)
    bal.release()
    _emit(
        "BALLOON_RELEASED",
        {
            "n_infers": n_infers,
            "errors": errors,
            "last_error": last_error,
            "held_s": held_s,
            "reason": "stop_file" if a.stop_file.exists() else "max_hours",
        },
    )
    return 0


def main(
    argv: list[str] | None = None,
    *,
    cpu_platform: Callable[[], CpuPlatform] | None = None,
    gpu_factory: Callable[[int], Any] | None = None,
    counters: Callable[[], dict[str, Any]] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    a = parse_args(argv)
    cnt = counters or _counters
    if a.cpu:
        return run_cpu(a, cpu_platform or WinCpuPlatform, cnt, sleep)
    return run_gpu(a, gpu_factory or GpuBalloon, cnt, sleep)


if __name__ == "__main__":
    raise SystemExit(main())
