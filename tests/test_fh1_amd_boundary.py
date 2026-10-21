"""Unit tests for the FH1 AMD boundary rows in analysis/numbers_register.py (amd_boundary_summary,
format_amd_boundary, compute_fh1_amd_boundary_*), on small fixtures."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analysis.numbers_register as nr  # noqa: E402

M = "fake-70b"


def _pt(req, act, shared, decode, ttft=1000.0, rep=0, dedicated=60000.0, model=M):
    return {"record": "mx2v_regime_point", "model_id": model, "requested_n_ctx": req, "actual_n_ctx": act,
            "rep": rep, "started": True, "gpu_shared_mib": shared, "gpu_dedicated_mib": dedicated,
            "decode_tok_s": decode, "ttft_ms": ttft}


BISECTION = [
    {"record": "mx2_baseline", "lines": {"device_local_mib": 76000.0, "vulkan_total_mib": 114000.0},
     "registry_dedicated": {"qw_memory_size_mib": 65536.0}},
    {"record": "mx2_probe", "model_id": M, "n_ctx": 1000, "regime": "FITS", "exit_code": None, "error": None},
    {"record": "mx2_probe", "model_id": M, "n_ctx": 2000, "regime": "SILENT_SPILL", "exit_code": None,
     "error": "guard: server does not match intended config: n_ctx 131072 != 2000"},
    {"record": "mx2_probe", "model_id": M, "n_ctx": 3000, "regime": "HARD_FAIL", "exit_code": 3221226505,
     "error": "process exited"},
    {"record": "mx2_two_line_result", "model_id": M, "device_local_last_fit_n_ctx": 1000,
     "device_local_first_spill_n_ctx": 2000, "vulkan_total_last_start_n_ctx": 2000,
     "vulkan_total_first_fail_n_ctx": 3000, "max_ctx_native": 131072, "yarn_factor": 1},
]


def test_boundary_uses_unclamped_points_and_reports_cost():
    stats = [_pt(100, 100, 500.0, 10.0, rep=r) for r in range(3)]
    stats += [_pt(200, 200, 600.0, 10.0, rep=r) for r in range(3)]       # +100 MiB: no spill
    stats += [_pt(300, 304, 5600.0, 9.0, ttft=1100.0, rep=r) for r in range(3)]  # rounded up, +5100: spill
    stats += [_pt(900, 400, 50000.0, 8.0, rep=r) for r in range(3)]      # clamped: never the boundary
    s = nr.amd_boundary_summary(M, {"spill_stats": stats}, BISECTION, [])
    assert s["last_fit"]["actual"] == 200
    assert s["first_spill"]["actual"] == 304
    assert abs(s["cost"]["decode_ratio"] - 0.9) < 1e-9
    assert abs(s["cost"]["ttft_ratio"] - 1.1) < 1e-9
    clamped = [p for p in s["points"] if p["clamped"]]
    assert len(clamped) == 1 and clamped[0]["requested"] == 900 and clamped[0]["spill"]
    assert s["bisection"]["first_fail"]["exit_code"] == 3221226505
    assert s["bisection"]["last_start"]["clamped"] is True
    assert s["bisection"]["fail_points"] == [3000]
    text = nr.format_amd_boundary(s)
    assert "last no-spill n_ctx 200" in text and "first spill n_ctx 304" in text
    assert "decode ratio 0.900" in text and "CLAMPED req 900 served 400" in text


def test_no_unclamped_spill_gives_no_cost_and_edge_counts():
    val = [_pt(100, 100, 500.0, 11.0, rep=r) for r in range(3)]
    val += [_pt(5000, 400, 30000.0, 10.0, rep=r) for r in range(3)]
    val.append({"record": "mx2v_regime_point", "model_id": M, "requested_n_ctx": 6000, "rep": 0, "started": False,
                "error": "llama-server port 1 not ready within 120.0s."})
    edge = [{"record": "x2_70b_edge_rep", "model_id": M, "arm": a, "n_ctx": 3000,
             "launch": {"outcome": "CRASH", "exit_code_name": "STATUS_STACK_BUFFER_OVERRUN", "crash_stage": "warmup",
                        "elapsed_s": e}} for a, e in [("warm_read", 120.0), ("cold_purge", 330.0)]]
    s = nr.amd_boundary_summary(M, {"validation": val}, BISECTION, edge)
    assert s["first_spill"] is None and s["cost"] is None
    assert s["last_fit"]["actual"] == 100
    assert s["edge"]["n"] == 2 and s["edge"]["elapsed_s"] == (120.0, 330.0)
    text = nr.format_amd_boundary(s)
    assert "no unclamped spill point measured" in text
    assert "warm_read CRASH STATUS_STACK_BUFFER_OVERRUN at warmup 1" in text
    assert "n_ctx 6000 started=False" in text


def test_compute_reads_the_four_files(tmp_path):
    (tmp_path / "results").mkdir()
    files = dict(zip(nr.AMD_BOUNDARY_FILES, [
        [_pt(100, 100, 500.0, 10.0, model="llama-3.3-70b"), _pt(300, 300, 9000.0, 9.5, model="llama-3.3-70b")],
        [],
        [dict(BISECTION[0])],
        [],
    ]))
    for f, rows in files.items():
        (tmp_path / f).write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    r = nr.compute_fh1_amd_boundary_llama70b(tmp_path)
    assert "first spill n_ctx 300" in r["value"]
    assert r["n"] == "2 measured calls, 0 fresh-process fail reps"
