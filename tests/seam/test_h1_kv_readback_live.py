"""Live KV readback assertion tests for tools/run_h1_hybrid.py (D-2c).

Uses a real OpenVINO load_arm_pipeline for gpu_only_u8 — not stubs.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_h1_hybrid import OpenVinoLocalBackend


def _scrub(obj):
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_scrub(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return f"<non_json:{type(obj).__name__}>"


def test_kv_readback_live_gpu_only_u8_match_and_mismatch() -> None:
    import tools.bfcl_feasibility_probe as probe

    print("=== D-2c live load_arm_pipeline('gpu_only_u8') ===", flush=True)
    _pipe, meta, load_s = probe.load_arm_pipeline(
        "gpu_only_u8", enable_prefix_caching=False
    )
    print(f"load_s={load_s:.3f}", flush=True)
    print("VERBATIM_META:", flush=True)
    print(json.dumps(_scrub(meta), indent=2, sort_keys=True), flush=True)

    kv = meta["loads"][0]["kv_cache_precision"]
    assert "readback" in kv
    assert isinstance(kv["readback"], dict)
    assert isinstance(kv["readback"]["normalized"], str)
    assert kv["readback"]["normalized"].lower() == "u8"

    # Matching precision must pass.
    OpenVinoLocalBackend._assert_kv_readback(meta, expected="u8")
    print("ASSERT_MATCH_u8: PASS", flush=True)

    # Mismatched precision must raise SystemExit (REFUSE).
    raised = False
    try:
        OpenVinoLocalBackend._assert_kv_readback(meta, expected="u4")
    except SystemExit as exc:
        raised = True
        msg = str(exc)
        assert "KV_PRECISION_MISMATCH" in msg
        assert "u4" in msg and "u8" in msg
        print(f"ASSERT_MISMATCH_u4: PASS ({msg})", flush=True)
    assert raised, "mismatch must SystemExit"

    # normalized is None => REFUSE (not a pass).
    meta_none = copy.deepcopy(meta)
    meta_none["loads"][0]["kv_cache_precision"]["readback"]["normalized"] = None
    meta_none["loads"][0]["kv_cache_precision"]["readback"]["ok"] = False
    raised_none = False
    try:
        OpenVinoLocalBackend._assert_kv_readback(meta_none, expected="u8")
    except SystemExit as exc:
        raised_none = True
        msg = str(exc)
        assert "normalized is None" in msg
        print(f"ASSERT_NONE_NORMALIZED: PASS ({msg[:200]})", flush=True)
    assert raised_none

    # Missing normalized key must refuse with observed structure (no silent .get).
    meta_bad = copy.deepcopy(meta)
    del meta_bad["loads"][0]["kv_cache_precision"]["readback"]["normalized"]
    raised_shape = False
    try:
        OpenVinoLocalBackend._assert_kv_readback(meta_bad, expected="u8")
    except SystemExit as exc:
        raised_shape = True
        assert "normalized" in str(exc)
        print(f"ASSERT_MISSING_NORMALIZED_KEY: PASS ({exc})", flush=True)
    assert raised_shape


if __name__ == "__main__":
    test_kv_readback_live_gpu_only_u8_match_and_mismatch()
    print("PASS tests/test_h1_kv_readback_live.py")