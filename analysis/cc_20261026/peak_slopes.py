"""Task BK: A12 / section 3.2 peak-minus-settled slopes from the in-child 10 Hz trace.

The turn-1 trace peak (generation.turn1.mem_trace.gpu_stat_total_peak_bytes) is in the child result
files (unsealed work/ copies of the same run_id). Settled is the summed GPU_MEMORY_STATISTICS after
turn 1 (as registered, section 3.2) and, as a second reading, after the last turn 2.
Writes analysis/cc_20261026/peak_slopes.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def tot(m: dict[str, Any] | None) -> int | None:
    g = ((m or {}).get("gpu_memory_statistics") or {}).get("GPU") or {}
    return sum(int(v) for v in g.values()) if g else None


def slope(pts: list[tuple[float, float]]) -> float | None:
    xs = [p[0] for p in pts]
    if len(set(xs)) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(p[1] for p in pts) / len(pts)
    return sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x in xs)


def main() -> int:
    cells = json.loads((OUT / "cells.json").read_text(encoding="utf-8"))
    out: dict[str, Any] = {}
    for c in cells:
        if not c.get("run_id") or c.get("kind") != "resident_limit":
            continue
        base = ROOT / c["dir"]
        work = Path(str(base).replace("__sealed_", "__")) / "work"
        a1: list[tuple[float, float]] = []
        a2: list[tuple[float, float]] = []
        for f in sorted(work.glob("*.result.json")):
            ch = json.loads(f.read_text(encoding="utf-8"))
            if not ch.get("completed") or ch.get("failure_mode"):
                continue
            t1 = ((ch.get("generation") or {}).get("turn1") or {}).get("mem_trace") or {}
            peak = t1.get("gpu_stat_total_peak_bytes")
            cm = ch.get("cc_mem") or []
            s1 = tot(next((m for m in cm if m.get("stage") == "after_turn1"), None))
            t2 = [m for m in cm if str(m.get("stage", "")).startswith("after_turn2")]
            s2 = tot(t2[-1]) if t2 else None
            n = (ch.get("spec") or {}).get("n_cached")
            if n is None:
                n = int(f.name.split(".")[0].lstrip("n").split("_")[0])
            if peak is None or s1 is None:
                continue
            a1.append((float(n), float(peak - s1)))
            if s2 is not None:
                a2.append((float(n), float(peak - s2)))
        if not a1:
            continue
        s_1, s_2 = slope(a1), slope(a2)
        out[f"{c['host']}:{c['label']}"] = {
            "run_id": c["run_id"],
            "arm": c["arm"],
            "cap": (c.get("scheduler") or {}).get("max_num_batched_tokens"),
            "n_points": len(a1),
            "peak_minus_after_turn1_slope": None if s_1 is None else round(s_1),
            "peak_minus_after_turn2_slope": None if s_2 is None else round(s_2),
        }
    (OUT / "peak_slopes.json").write_text(json.dumps(out, indent=1) + "\n", encoding="ascii")
    for k, v in out.items():
        print(k, v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
