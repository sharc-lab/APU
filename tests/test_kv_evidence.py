"""analysis/kv_evidence.py and the kv-precision-evidence / kv-output-agreement register rows."""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
import kv_evidence as kv  # noqa: E402

MIB = kv.MIB


def _call(content="", tools=None, ok=True, pec=100, ped=1.0, ct=10, ed=1.0):
    nt = [{"id": "call_random", "function": {"index": i, "name": n, "arguments": a}}
          for i, (n, a) in enumerate(tools or [])]
    return {"content": content, "native_tool_calls": nt or None, "tool_detection_method": "native" if nt else "none",
            "tool_calls_parsed": [{"name": n, "arguments": a} for n, a in (tools or [])], "outcome": "ok" if ok else "err",
            "prompt_eval_count_info_only": pec, "prompt_eval_duration_s": ped, "completion_tokens": ct,
            "eval_duration_s": ed, "done_reason": "stop"}


def _turn(idx, calls, wall=60.0, arm="b5_g_kv_q8_0_call2_notools", seed=1):
    return {"record": "r2a_turn", "model_id": kv.MODEL, "arm_id": arm, "seed": seed, "turn_idx": idx, "calls": calls,
            "turn_wall_s": wall, "loaded_context": 40960, "final_text": calls[-1]["content"]}


def test_kv_size_inference_recovers_bits():
    x = 6400 * MIB
    f16, q8, q4 = 9000 * MIB + x, 9000 * MIB + x * 8.5 / 16, 9000 * MIB + x * 4.5 / 16
    r = kv.kv_size_inference(int(f16), int(q8), int(q4))
    assert r["consistent"] and abs(r["kv_mib"]["q4_0"] - 1800) < 0.01 and abs(r["x_from_f16_q8"] - 6400) < 0.01


def test_kv_size_inference_flags_q4_fallback_to_q8():
    x = 6400 * MIB
    f16, q8 = 9000 * MIB + x, 9000 * MIB + x * 8.5 / 16
    assert not kv.kv_size_inference(int(f16), int(q8), int(q8))["consistent"]


def test_session_evidence_minutes_and_rates():
    ts = [_turn(1, [_call(pec=200, ped=2.0, ct=30, ed=3.0)], wall=90), _turn(2, [_call()], wall=30)]
    e = kv.session_evidence(ts)
    assert e["minutes"] == 2.0 and e["n_calls"] == 2 and e["loaded_context"] == [40960]
    assert e["prefill_median"] == 100.0 and e["decode_median"] == 10.0 and e["length_stops"] == 0


def test_first_divergence_char_and_token():
    assert kv.first_divergence("abc", "abc") is None
    assert kv.first_divergence("abcd", "abxd") == 2
    assert kv.first_divergence("ab", "abc") == 2
    assert kv.first_divergence("a b c", "a b d", tokenize=str.split) == 2


def test_compare_sessions_stops_at_first_divergence():
    a = [_turn(1, [_call("x", [("lookup_fact", {"key": "K"})]), _call("ans")]),
         _turn(2, [_call("same")]), _turn(3, [_call("later")])]
    b = [_turn(1, [_call("x", [("lookup_fact", {"key": "K"})]), _call("ans")]),
         _turn(2, [_call("diff")]), _turn(3, [_call("later")])]
    r = kv.compare_sessions(a, b, 1)
    assert r["comparable"] == 2 and r["exact"] == 1 and r["first_divergent_turn"] == 2
    assert r["first_divergent_call"] == 0 and r["first_divergence_index"] == 0
    # turn 3's prompt differs (it carries turn 2's differing output), so it is not compared
    assert [t["turn_idx"] for t in r["turns"]] == [1, 2]


def test_compare_sessions_ignores_tool_call_ids_and_tracks_tool_agreement():
    a = [_turn(1, [_call("", [("log_event", {"e": 1})]), _call("ok")])]
    b = [_turn(1, [_call("", [("log_event", {"e": 1})]), _call("ok!")])]
    b[0]["calls"][0]["native_tool_calls"][0]["id"] = "call_other"
    r = kv.compare_sessions(a, b, 1)
    assert r["comparable"] == 1 and r["exact"] == 0 and r["tool_agree"] == 1
    assert r["first_divergent_call"] == 1 and r["first_divergence_index"] == 2


def test_equal_means_explanation():
    ev = {("g", 1): {"minutes": 70.0, "length_stops": 0, "n_turns": 40, "completion_tokens": 5},
          ("g", 2): {"minutes": 80.0, "length_stops": 0, "n_turns": 40, "completion_tokens": 6},
          ("h", 1): {"minutes": 79.0, "length_stops": 0, "n_turns": 40, "completion_tokens": 7},
          ("h", 2): {"minutes": 71.0, "length_stops": 0, "n_turns": 40, "completion_tokens": 8}}
    r = kv.equal_means_explanation({1: None, 2: None}, ev)
    assert r["mean_g"] == r["mean_h"] == 75.0 and r["seed_diff_min"] == {1: 9.0, 2: -9.0}


B5 = REPO / kv.B5_FILE
FA = REPO / kv.FA_FILE
needs_files = pytest.mark.skipif(not (B5.exists() and FA.exists()), reason="B5 result files not present")


@needs_files
def test_real_files_g_h_means_and_q4_evidence():
    e = kv.evidence(REPO)
    gh = e["gh"]
    assert round(gh["mean_g"], 1) == round(gh["mean_h"], 1) == 75.4
    assert gh["g"] != gh["h"] and gh["length_stops"] == 0
    assert e["loads"]["q4_0"]["kv_observed"] == ["q4_0"] and e["kv"]["consistent"]
    assert all(e["run_end"].values())


@needs_files
def test_real_files_f16_reruns_identical():
    e = kv.evidence(REPO)
    ag = kv.agreement(e["sessions"], pairs=(("a", "i"),))
    v = ag[("a", "i")]
    assert v["comparable"] == v["exact"] == v["n_turns"] == 120


@needs_files
def test_register_rows_compute():
    import numbers_register as nr
    ids = {e["claim_id"]: e for e in nr.NUMBER_ENTRIES}
    for cid in ("kv-precision-evidence", "kv-output-agreement"):
        assert cid in ids
        r = ids[cid]["compute"](REPO)
        assert r["value"] and "—" not in r["value"]
