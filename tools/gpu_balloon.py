"""A22 / A29 balloon (Task BM): hold X GiB of GPU-shared (or CPU-only) memory in a side process.

--gpu: allocate X GiB in 512 MiB chunks through OpenVINO's GPU context and TOUCH every byte
  (device tensor + copy_from a host buffer; if that API is missing, a GPU host tensor written
  through numpy). Then read this process's PDH GPU Shared Usage (seam.turn_memory) and print
  BALLOON_READY with the measured MB.
--cpu: allocate X GiB of plain process memory (numpy, every page written); no GPU.
The process holds the memory until --stop-file exists (or --max-hours), then exits.
It changes no setting. Exit 2 if allocation fails (the error text is printed).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CHUNK = 512 * 1024 * 1024


def _gpu_alloc(n_bytes: int) -> tuple[list[Any], str]:
    import numpy as np
    import openvino as ov

    core = ov.Core()
    ctx = core.get_default_context("GPU")
    held: list[Any] = []
    method = "device_tensor_copy_from"
    left = n_bytes
    while left > 0:
        size = min(CHUNK, left)
        host = np.full(size, 0xA5, dtype=np.uint8)
        try:
            t = ctx.create_tensor(ov.Type.u8, ov.Shape([size]), {})
            t.copy_from(ov.Tensor(host))
            held.append(t)
        except (AttributeError, TypeError):
            method = "host_tensor_numpy_write"
            ht = ctx.create_host_tensor(ov.Type.u8, ov.Shape([size]))
            np.asarray(ht.data)[:] = 0xA5
            held.append(ht)
        held.append(host)
        left -= size
    return held, method


def _cpu_alloc(n_bytes: int) -> tuple[list[Any], str]:
    import numpy as np

    held = []
    left = n_bytes
    while left > 0:
        size = min(CHUNK, left)
        a = np.empty(size, dtype=np.uint8)
        a[::4096] = 1  # write every page
        held.append(a)
        left -= size
    return held, "numpy_page_touch"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    kind = ap.add_mutually_exclusive_group(required=True)
    kind.add_argument("--gpu", action="store_true")
    kind.add_argument("--cpu", action="store_true")
    ap.add_argument("--gib", type=float, required=True)
    ap.add_argument("--stop-file", type=Path, required=True)
    ap.add_argument("--max-hours", type=float, default=8.0)
    a = ap.parse_args(argv)
    n = int(a.gib * 2**30)
    from seam.turn_memory import read_gpu_counters

    before = read_gpu_counters(os.getpid())
    try:
        held, method = _gpu_alloc(n) if a.gpu else _cpu_alloc(n)
    except Exception as exc:
        print(
            json.dumps({"event": "BALLOON_FAILED", "error": f"{type(exc).__name__}: {exc}"[:500]}),
            flush=True,
        )
        return 2
    after = read_gpu_counters(os.getpid())
    print(
        "BALLOON_READY "
        + json.dumps(
            {
                "kind": "gpu" if a.gpu else "cpu",
                "gib": a.gib,
                "bytes": n,
                "method": method,
                "pid": os.getpid(),
                "gpu_counters_before": before,
                "gpu_counters_after": after,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    t_end = time.time() + a.max_hours * 3600
    while not a.stop_file.exists() and time.time() < t_end:
        time.sleep(1.0)
    print(f"BALLOON_RELEASED chunks={len(held)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
