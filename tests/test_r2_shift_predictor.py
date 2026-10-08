"""analysis/r2_shift_predictor: shift trigger, step simulation, confusion counts, real-run lineup and orphan counts on
synthetic rows."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import r2_shift_predictor as sp  # noqa: E402

ARM = "ollama_ctx_4096_call2_notools"
M = "llama3.1:8b"


def test_trigger_is_prompt_plus_generated_over_n_ctx():
    assert sp.predict(4082, 15, 4096) is True      # 4097 > 4096
    assert sp.predict(4082, 14, 4096) is False     # 4096: last check sees n_tokens 4094, 4094 + 1 < 4096
    assert sp.predict(4095, 1, 4096) is False      # one token, nothing decoded after the prompt
    assert sp.predict(4095, 2, 4096) is True


def test_simulation_matches_logged_shift_and_trigger():
    # the mechanism log's task 8: prompt 4082, 78 generated, n_keep 4 + BOS
    ev = sp.simulate_shifts(4082, 78, 4096, 5)
    assert ev == [{"n_keep": 5, "n_left": 4090, "n_discard": 2045}]
    for P in range(4000, 4096):
        for C in (1, 2, 50, 96, 97, 384):
            assert bool(sp.simulate_shifts(P, C, 4096, 5)) == sp.predict(P, C, 4096)
    # a tiny context forces repeated shifts within one call
    assert len(sp.simulate_shifts(10, 40, 16, 1)) > 1


def test_token_cut_length():
    assert sp.token_cut_new_len(4096) == 2050


def _call(turn, call, P, C, shifts=(), check=None, stop=None, trunc=False, idx=None, role=None, n_msgs=None):
    r = {"record": "r2m_call", "model_id": M, "arm_id": ARM, "seed": 1, "turn_idx": turn, "call_idx": call,
         "new_prompt": [{"n_ctx_slot": 4096, "n_keep": 4, "n_tokens": P}], "prompt_eval_count": P,
         "context_shifts": list(shifts), "token_cut": None, "message_level_truncation": trunc,
         "first_kept_message_index_log": idx, "first_kept_message_role_log": role, "n_messages_sent": n_msgs,
         "log_excerpt": [] if stop is None else [f"slot release: stop processing: n_tokens = {stop}, truncated = 1"]}
    return r


def _turn(i, c1, c2, tools1=1, tools2=0, **kw):
    calls = [{"completion_tokens": c1, "prompt_eval_count_info_only": None, "outcome": "ok",
              "tool_calls_parsed": [{"name": "x"}] * tools1}]
    if tools1:
        calls.append({"completion_tokens": c2, "outcome": "ok", "tool_calls_parsed": [{"name": "y"}] * tools2})
    row = {"record": "r2a_turn", "model_id": M, "arm_id": ARM, "seed": 1, "turn_idx": i, "calls": calls}
    row.update(kw)
    return row


def test_mechanism_confusion_and_misses():
    shift = {"n_keep": 5, "n_left": 4090, "n_discard": 2045}
    rows = [_turn(1, 78, 20), _turn(2, 10, 10),
            _call(1, 1, 4082, 78, [shift], stop=4082 + 77 - 2045),   # TP
            _call(1, 2, 3000, 20),                                    # TN
            _call(2, 1, 4090, 10),                                    # observed none, predicted shift: FP
            _call(2, 2, 3000, 10, [shift]),                           # observed, not predicted: FN
            {"record": "r2m_prompt_check", "model_id": M, "arm_id": ARM, "seed": 1, "turn_idx": 1, "call_idx": 2,
             "match": True, "check_prompt_eval_count": 3000}]
    ms = sp.mechanism_summary(rows)
    c = ms["per_tier"]["num_ctx_4096"]
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (1, 1, 1, 1)
    assert c["precision"] == 0.5 and c["recall"] == 0.5
    assert {(m["turn"], m["call"]) for m in ms["misses"]} == {(2, 1), (2, 2)}
    assert ms["decoded_tokens_check"] == {"n": 1, "ok": 1}
    assert ms["P_sources"] == {"fresh_check": 1, "new_prompt": 3}
    assert ms["variants"]["budget P+num_predict>n_ctx"]["fp"] >= 1


def _real_turn(seed, i, p1, c1, p2, c2, n_ctx=4096, **kw):
    row = {"record": "r2a_turn", "mode": "real", "model_id": M, "arm_id": ARM, "seed": seed, "turn_idx": i,
           "loaded_context": n_ctx, "rule1_json_keys": True,
           "calls": [{"prompt_eval_count_info_only": p1, "completion_tokens": c1, "outcome": "ok"},
                     {"prompt_eval_count_info_only": p2, "completion_tokens": c2, "outcome": "ok"}]}
    row.update(kw)
    return row


def test_real_summary_counts_and_lineup():
    rows = [_real_turn(1, 1, 2000, 30, 1900, 50),
            _real_turn(1, 2, 3000, 30, 4080, 50),                       # call 2 predicted
            _real_turn(1, 3, 3000, 30, 2050, 50, rule1_json_keys=False),  # token-cut signature, rule failure
            _real_turn(2, 1, 2000, 30, 1900, 50, rule1_json_keys=False),
            _real_turn(2, 2, 4090, 30, 3000, 50)]                       # call 1 predicted
    rs = sp.real_summary(rows, {M: ["rule1_json_keys"]})
    cell = rs["cells"][("num_ctx_4096", M)]
    assert cell["pred_calls"] == 2 and cell["pred_call1"] == 1 and cell["pred_call2"] == 1
    assert cell["cut_sig_calls"] == 1 and cell["sessions_with_pred_shift"] == 2
    s1, s2 = cell["sessions"][1], cell["sessions"][2]
    assert (s1["first_pred_shift"], s1["first_cut_signature"], s1["first_rule_failure"]) == (2, 3, 3)
    assert s1["lineup"] == "failure after first shift" and s1["first_prompt_start_loss"] == 2
    assert s2["lineup"] == "failure before first shift"


def test_rebuild_roles_and_orphans():
    # turn 1: tool call then final; turn 2: no tool call; turn 3: tool call (call 2 of turn 3 is truncated)
    turns = [_turn(1, 10, 10), _turn(2, 10, None, tools1=0), _turn(3, 10, 10)]
    roles = sp.rebuild_roles(turns)
    assert roles[(1, 2)] == ["system", "user", "assistant_toolcall", "tool"]
    assert roles[(3, 2)] == ["system", "user", "assistant_toolcall", "tool", "assistant_final", "user",
                             "assistant_final", "user", "assistant_toolcall", "tool"]
    rows = turns + [
        _call(3, 2, 100, 10, trunc=True, idx=3, role="tool", n_msgs=10),       # orphaned tool result
        _call(3, 1, 100, 10, trunc=True, idx=2, role="assistant", n_msgs=8),   # orphaned assistant tool call
        _call(2, 1, 100, 10, trunc=True, idx=5, role="user", n_msgs=6),        # turn boundary
        _call(1, 1, 100, 10)]
    o = sp.orphan_summary(rows)["num_ctx_4096"]
    assert o["truncated"] == 3 and o["first_tool"] == 1 and o["first_assistant_toolcall"] == 1 and o["first_user"] == 1
    assert o["orphan_first_kept"] == 2 and o["role_check_ok"] == 3 and o["length_check_bad"] == 0
    assert abs(o["orphan_tool_rate"] - 1 / 3) < 1e-9
