"""Task BK: where context bytes go (registration section 3.2) from the sealed point records.

Per resident point with child memory records (cc_mem):
- W: the loaded stage's process Shared Usage (PDH, MB) and usm_host bytes (weights);
- KV pool: cl_mem bytes after the last turn 2 (settled); usm_device after turn 1 (retained prefill
  buffers);
- turn-1 peak: the 1 Hz PDH sidecar's adapter Shared Usage peak inside [loaded, after_turn1]
  (tools/cc_mem_trace.py), minus the settled after-turn-1 Shared Usage = peak-minus-settled.
Per cell: least-squares slopes vs n (B per token) of the settled KV pool and of peak-minus-settled.
Writes analysis/cc_20261026/bytes.json. Sidecar CSVs are read from ~/seam_closeout_1013 (their sha256
is in derived/cc/windows_20261023_25/CSV_MANIFEST.sha256).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.cc_mem_trace import read_rows, summarise  # noqa: E402

OUT = Path(__file__).resolve().parent
QUEUE = Path.home() / "seam_closeout_1013"
MB = 2**20


def slope(xs: list[float], ys: list[float]) -> float | None:
    if len(set(xs)) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sxx


def stage(cm: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    hits = [m for m in cm if str(m.get("stage", "")).startswith(name)]
    return hits[-1] if hits else None


def gpu(m: dict[str, Any] | None, key: str) -> int | None:
    if not m:
        return None
    g = (m.get("gpu_memory_statistics") or {}).get("GPU") or {}
    v = g.get(key)
    return int(v) if isinstance(v, int) else None


def main() -> int:
    cells = json.loads((OUT / "cells.json").read_text(encoding="utf-8"))
    out: dict[str, Any] = {}
    for c in cells:
        if not c.get("run_id") or c.get("kind") != "resident_limit":
            continue
        pdir = ROOT / c["dir"] / "points"
        csv = QUEUE / c["host"] / c["window"] / f"{c['label']}_mem.csv"
        rows = read_rows(csv) if csv.is_file() else []
        pts = []
        for p in sorted(pdir.glob("*.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            cm = d.get("cc_mem") or []
            if not cm:
                continue
            ld, t1 = stage(cm, "loaded"), stage(cm, "after_turn1")
            t2 = stage(cm, "after_turn2")
            rec: dict[str, Any] = {
                "file": p.name,
                "n": d.get("n_cached"),
                "holds": d.get("holds"),
                "W_shared_mb": (ld or {}).get("gpu_process_counters", {}).get("shared_usage_mb"),
                "W_usm_host": gpu(ld, "usm_host"),
                "kv_pool_cl_mem": gpu(t2, "cl_mem") if t2 else gpu(t1, "cl_mem"),
                "usm_device_after_turn1": gpu(t1, "usm_device"),
                "available_mb_loaded": ((ld or {}).get("system") or {}).get("available_mb"),
                "compiled_kv": [
                    (x or {}).get("kv_cache_precision") for x in (d.get("compiled_kv") or [])
                ],
                "backend": [(x or {}).get("backend") for x in (d.get("compiled_kv") or [])],
            }
            if rows and ld and t1:
                s = summarise(rows, float(ld["t_unix"]), float(t1["t_unix"]))
                settled = (t1.get("gpu_process_counters") or {}).get("shared_usage_mb")
                rec["sidecar_peak_mb"] = s["adapter_shared_peak_mb"]
                rec["sidecar_samples"] = s["samples"]
                if s["adapter_shared_peak_mb"] is not None and settled is not None:
                    rec["peak_minus_settled_mb"] = round(s["adapter_shared_peak_mb"] - settled, 1)
            pts.append(rec)
        if not pts:
            continue
        held = [p for p in pts if p["holds"] and p["kv_pool_cl_mem"] is not None]
        xs = [float(p["n"]) for p in held]
        kv = slope(xs, [float(p["kv_pool_cl_mem"]) for p in held])
        pm_pts = [p for p in held if p.get("peak_minus_settled_mb") is not None]
        pm = slope(
            [float(p["n"]) for p in pm_pts], [p["peak_minus_settled_mb"] * MB for p in pm_pts]
        )
        ud = slope(xs, [float(p["usm_device_after_turn1"] or 0) for p in held])
        out[f"{c['host']}:{c['label']}"] = {
            "run_id": c["run_id"],
            "arm": c["arm"],
            "cap": (c.get("scheduler") or {}).get("max_num_batched_tokens"),
            "sidecar_csv": csv.is_file(),
            "points": pts,
            "settled_kv_slope_B_per_token": round(kv) if kv is not None else None,
            "peak_minus_settled_slope_B_per_token": round(pm) if pm is not None else None,
            "usm_device_after_turn1_slope_B_per_token": round(ud) if ud is not None else None,
        }
    (OUT / "bytes.json").write_text(json.dumps(out, indent=1) + "\n", encoding="ascii")
    for k, v in out.items():
        print(
            k,
            v["run_id"],
            v["arm"],
            v["cap"],
            "kv",
            v["settled_kv_slope_B_per_token"],
            "pk-st",
            v["peak_minus_settled_slope_B_per_token"],
            "usmdev",
            v["usm_device_after_turn1_slope_B_per_token"],
            "csv",
            v["sidecar_csv"],
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
