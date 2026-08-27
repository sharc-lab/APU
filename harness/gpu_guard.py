"""GPU backend verification.

Call verify_gpu_backend() once at harness startup before any model calls.
Aborts with a clear message if the model is running on CPU rather than GPU.

A run that silently executes on CPU produces plausible latency and token counts
from the wrong hardware. That is the worst failure mode: the results look valid
but are incomparable to GPU runs. This guard makes the failure loud.

Works by measuring GPU VRAM before and after a warmup call. If the delta is
below MIN_VRAM_DELTA_MIB, the model is not using the GPU.

Returns a dict with keys: backend_verified (bool), vram_before_mib,
vram_after_mib, vram_delta_mib, warmup_output. Callers should write
backend_verified to every result row.
"""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from typing import Any

MIN_VRAM_DELTA_MIB = 300  # model on GPU uses at least this much VRAM


def _read_vram_mib() -> int | None:
    """Return current GPU VRAM used in MiB, or None if nvidia-smi unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            timeout=10,
        )
        return int(out.decode().strip().split("\n")[0].strip())
    except Exception:
        return None


def _ollama_warmup(host: str, model: str) -> str:
    """Send a minimal prompt to the model and return the output."""
    payload = json.dumps({
        "model": model,
        "prompt": "Say: OK",
        "stream": False,
        "options": {"num_predict": 4, "temperature": 0},
    }).encode()
    req = urllib.request.Request(
        f"{host}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read()).get("response", "").strip()


def verify_gpu_backend(
    host: str = "http://localhost:11434",
    model: str = "qwen3:4b-instruct",
    abort_on_cpu: bool = True,
) -> dict[str, Any]:
    """Verify the model is running on GPU. Abort if not and abort_on_cpu=True."""
    vram_before = _read_vram_mib()

    if vram_before is None:
        # nvidia-smi unavailable — cannot verify. Warn but don't abort.
        print(
            "[gpu_guard] WARNING: nvidia-smi not available. Cannot verify GPU backend.",
            flush=True,
        )
        return {
            "backend_verified": False,
            "vram_before_mib": None,
            "vram_after_mib": None,
            "vram_delta_mib": None,
            "warmup_output": None,
            "note": "nvidia-smi unavailable; verification skipped",
        }

    warmup_out = _ollama_warmup(host, model)

    vram_after = _read_vram_mib()
    delta = (vram_after or 0) - (vram_before or 0)

    verified = delta >= MIN_VRAM_DELTA_MIB or (vram_after or 0) >= MIN_VRAM_DELTA_MIB

    result: dict[str, Any] = {
        "backend_verified": verified,
        "vram_before_mib": vram_before,
        "vram_after_mib": vram_after,
        "vram_delta_mib": delta,
        "warmup_output": warmup_out,
    }

    if not verified:
        msg = (
            f"[gpu_guard] ABORT: model appears to be running on CPU.\n"
            f"  VRAM before warmup: {vram_before} MiB\n"
            f"  VRAM after  warmup: {vram_after} MiB\n"
            f"  Delta: {delta} MiB (threshold: {MIN_VRAM_DELTA_MIB} MiB)\n"
            f"  A CPU run produces plausible numbers from the wrong hardware.\n"
            f"  Ensure ollama has loaded the model to GPU before running."
        )
        print(msg, flush=True)
        if abort_on_cpu:
            sys.exit(2)
    else:
        print(
            f"[gpu_guard] GPU backend confirmed: {vram_after} MiB VRAM in use "
            f"(delta +{delta} MiB after warmup).",
            flush=True,
        )

    return result
