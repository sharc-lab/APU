"""Canary cell in a child process (NPU amendment 2026-10-14b).

The guard spawns one child per canary. The child runs the fixed canary cell
(the same call TtftSloCanaryGuard makes in-process), writes {"cell": raw}
to --out, and exits 0. The parent reads the file; a non-zero exit, a
timeout or a missing file is classified OTHER by the parent.

  python -m tools.canary_child --out <work>/canaries/child.N.json --model-spec <spec>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STDERR_TAIL_CHARS = 2000


def cell_inprocess(model_spec: Path) -> dict[str, Any]:
    """The fixed canary cell, in the calling process."""
    from tools.smoke_delta_prefill import run_cell
    from tools.ttft_slo_canary import CANARY_ARM, CANARY_DELTA, CANARY_MODE, CANARY_N_CACHED

    return run_cell(
        "ttft_slo_canary",
        arm_id=CANARY_ARM,
        mode=CANARY_MODE,
        n_cached=CANARY_N_CACHED,
        delta=CANARY_DELTA,
        model_spec=model_spec,
        pipeline_type="llm",
    )


def child_command(out: Path, model_spec: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "tools.canary_child",
        "--out",
        str(out),
        "--model-spec",
        str(model_spec),
    ]


def _other(error: str) -> dict[str, Any]:
    return {
        "classification": "OTHER",
        "execute_ok": False,
        "execute_error": error,
        "turn1": None,
        "turn2": None,
    }


def run_child(
    cmd: list[str], out: Path, *, timeout_s: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Spawn the child, wait, read its result. (raw cell, child record)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    started = time.perf_counter()
    proc = subprocess.Popen(
        cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    timed_out = False
    try:
        _stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        _stdout, stderr = proc.communicate()
    record = {
        "pid": proc.pid,
        "exit_code": None if timed_out else proc.returncode,
        "wall_s": round(time.perf_counter() - started, 3),
        "timeout_s": float(timeout_s),
        "timed_out": timed_out,
        "stderr_tail": (stderr or "")[-STDERR_TAIL_CHARS:],
        "result_path": out.name,
    }
    if timed_out:
        return _other(f"canary child timeout after {timeout_s} s"), record
    if proc.returncode != 0:
        return _other(f"canary child exit code {proc.returncode}"), record
    try:
        raw = json.loads(out.read_text(encoding="utf-8"))["cell"]
    except Exception as exc:
        return _other(f"canary child wrote no usable result: {type(exc).__name__}: {exc}"), record
    return raw, record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-spec", type=Path, required=True)
    args = parser.parse_args(argv)
    raw = cell_inprocess(args.model_spec)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"cell": raw}, default=str) + "\n", encoding="utf-8")
    t1 = (raw.get("turn1") or {}).get("prefill_s")
    t2 = (raw.get("turn2") or {}).get("prefill_s")
    print(f"CANARY_CHILD classification={raw.get('classification')} t1={t1} t2={t2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
