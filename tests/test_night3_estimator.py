"""night3 pre-launch estimator: real per-call overheads from a prior night2 results file must be used in place of
the overnight-table fit when present, B4/B4b phases must appear in the estimate, and a missing/unreadable
--prior-results file must fall back cleanly (no crash, old table-fit behavior)."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))

import t2s_night2 as n2  # noqa: E402


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            if isinstance(r, str):
                f.write(r if r.endswith("\n") else r + "\n")  # write a deliberately malformed raw line verbatim
            else:
                f.write(json.dumps(r) + "\n")


def test_load_night2_overheads_computes_medians(tmp_path):
    p = tmp_path / "prior.jsonl"
    _write_jsonl(p, [
        {"kind": "start", "model_id": "qwen3-8b", "load_s": 10.0},
        {"kind": "start", "model_id": "qwen3-8b", "load_s": 12.0},
        {"kind": "call", "model_id": "qwen3-8b", "warmup": True, "e2e_s": 999.0},  # warm-up excluded
        {"kind": "call", "model_id": "qwen3-8b", "warmup": False, "e2e_s": 30.0, "thermal_wait_s": 100.0},
        {"kind": "call", "model_id": "qwen3-8b", "warmup": False, "e2e_s": 34.0, "thermal_wait_s": 140.0},
        "{not valid json",  # malformed line must not crash the loader
    ])
    over = n2.load_night2_overheads(str(p))
    assert over["load_s"]["qwen3-8b"] == 11.0
    assert over["call_e2e_s"]["qwen3-8b"] == 32.0
    assert over["gate_wait_s"] == 120.0


def test_load_night2_overheads_missing_file_returns_empty():
    assert n2.load_night2_overheads(None) == {}
    assert n2.load_night2_overheads("C:/definitely/not/a/real/path.jsonl") == {}


def test_estimate_hours_uses_real_overheads_and_includes_b4(tmp_path):
    p = tmp_path / "prior.jsonl"
    _write_jsonl(p, [
        {"kind": "start", "model_id": "qwen3-8b", "load_s": 5.0},
        {"kind": "call", "model_id": "qwen3-8b", "warmup": False, "e2e_s": 20.0, "thermal_wait_s": 60.0},
    ])
    over = n2.load_night2_overheads(str(p))
    lab = SimpleNamespace(table={})
    est_real = n2.estimate_hours(lab, over)
    est_fallback = n2.estimate_hours(lab, {})
    for phase in ("b1", "b2", "b3", "c1", "c1b", "b4", "b4_32b", "perfboost"):
        assert phase in est_real
        assert est_real[phase] >= 0.0
    # real per-call e2e_s (20s) is far below the flat-guess fallback (45s) for an unknown model, so the real-overhead
    # estimate for a phase driven by qwen3-8b must come out lower than the table-fit fallback.
    assert est_real["b4"] < est_fallback["b4"]
