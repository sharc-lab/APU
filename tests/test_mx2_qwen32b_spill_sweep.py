"""qwen3:32b spill sweep (docs/FINDINGS.md 2026-10-22): the pure parts (points, classification, spill point, cost)."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))
import mx2_qwen32b_spill_sweep as sw  # noqa: E402


def test_grid_is_unclamped_range_and_refinement():
    assert sw.GRID[0] == 20480 and sw.GRID[-1] == 40960 and all(b - a == 4096 for a, b in zip(sw.GRID, sw.GRID[1:]))
    assert sw.refine_points(32768) == [29696, 30720, 31744]
    assert sw.refine_points(None) == [] and sw.refine_points(20480) == []
    assert sw.WARMUP_CALLS == 1 and sw.MEASURED_CALLS == 5 and sw.PORT != 58399


def _rows(points):
    rows = []
    for n, (shared, decode, ttft, clamped) in points.items():
        rows.append({"record": "sweep_point", "requested_n_ctx": n, "served_n_ctx": n if not clamped else 40960,
                     "clamped": clamped, "started": True})
        rows.append({"record": "sweep_call", "requested_n_ctx": n, "warmup": True, "ok": True, "decode_tok_s": 1.0,
                     "ttft_ms": 1.0, "gpu_shared_mib": 99999.0})
        for i in range(5):
            rows.append({"record": "sweep_call", "requested_n_ctx": n, "warmup": False, "ok": True,
                         "decode_tok_s": decode + 0.01 * i, "ttft_ms": ttft + i, "gpu_shared_mib": shared,
                         "gpu_dedicated_mib": 30000.0})
    return rows


def test_summarize_finds_spill_point_and_cost():
    pts = {20480: (500.0, 11.45, 10260.0, False), 24576: (520.0, 11.44, 10270.0, False),
           28672: (540.0, 11.43, 10280.0, False), 32768: (9000.0, 11.20, 10600.0, False),
           36864: (15000.0, 11.19, 10620.0, False), 40960: (28000.0, 11.19, 10660.0, False),
           29696: (560.0, 11.43, 10280.0, False), 30720: (4000.0, 11.30, 10500.0, False),
           31744: (6000.0, 11.25, 10550.0, False)}
    s = sw.summarize(_rows(pts), 1024.0)
    assert s["baseline_shared_mib"] == 500.0 and s["spill_point"] == 30720 and s["last_no_spill"] == 29696
    c = s["cost"]
    assert c["decode_ratio"] < 1 and c["ttft_ratio"] > 1
    assert c["decode_reading"] == "decode: spill cost shown" and c["ttft_reading"] == "TTFT: spill cost shown"
    assert s["points"][20480]["n_measured"] == 5            # the warm-up is not counted


def test_clamped_points_never_count_and_no_spill_case():
    pts = {20480: (500.0, 11.45, 10260.0, False), 24576: (600.0, 11.44, 10270.0, False),
           40960: (28000.0, 11.19, 10660.0, True)}
    s = sw.summarize(_rows(pts), 1024.0)
    assert s["points"][40960]["class"] == "clamped" and s["spill_point"] is None and s["cost"] is None
    assert s["last_no_spill"] == 24576


def test_cost_reading_and_welch():
    d = sw.welch_diff_ci([11.4, 11.41, 11.42], [11.2, 11.21, 11.22])
    assert d["hi"] < 0 and sw.cost_reading(d, "decode", lower_is_worse=True) == "decode: spill cost shown"
    same = sw.welch_diff_ci([1.0, 2.0, 3.0], [1.1, 2.1, 2.9])
    assert sw.cost_reading(same, "decode", True) == "decode: no spill cost shown"
    assert sw.cost_reading(None, "TTFT", False).startswith("TTFT: not read")
