"""analysis/r2_info_loss: system-prompt span arithmetic, transcript share, before/after-loss reliability, on synthetic
mechanism and real-run rows."""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))

import r2_info_loss as il  # noqa: E402

M = "llama3.1:8b"
Q = "qwen3:14b"
A4 = "ollama_ctx_4096_call2_notools"
AD = "ollama_default_call2_notools"


def test_system_share_lost_spans():
    # llama.cpp shift keeping BOS + 4 tokens, discarding 2045: the whole llama3.1 system prompt (text at 5..5+S)
    assert il.system_share_lost(M, 400, shifts=[(5, 2045)]) == 1.0
    # qwen3 header is 3 tokens; n_keep 4 keeps one text token
    assert il.system_share_lost(Q, 400, shifts=[(4, 2046)]) == pytest.approx(399 / 400)
    # a discard shorter than the system prompt removes only part of it
    assert il.system_share_lost(M, 400, shifts=[(5, 100)]) == pytest.approx(100 / 400)
    # Ollama token cut: keep 4 (no BOS in its token list), drop 2047
    assert il.system_share_lost(M, 400, cut=(4, 2047)) == 1.0
    assert il.system_share_lost(M, 400) == 0.0
    # repeated shifts are capped at the whole system prompt
    assert il.system_share_lost(M, 400, shifts=[(5, 300), (5, 300)]) == 1.0


def _turn(model, arm, seed, i, ests, rules, calib=0.8, mode="mechanism", P=None, C=None, n_ctx=4096):
    calls = []
    for j, e in enumerate(ests):
        c = {"prompt_tokens_est": e, "completion_tokens": (C or [20] * len(ests))[j], "outcome": "ok"}
        if P is not None:
            c["prompt_eval_count_info_only"] = P[j]
        calls.append(c)
    row = {"record": "r2a_turn", "mode": mode, "model_id": model, "arm_id": arm, "seed": seed, "turn_idx": i,
           "calls": calls, "token_calib_ratio": calib, "loaded_context": n_ctx}
    row.update({r: True for r in il.RULES})
    row.update(rules)
    return row


def _mcall(model, arm, seed, i, j, P, n_ctx=4096, trunc=False, shifts=(), cut=None):
    return {"record": "r2m_call", "model_id": model, "arm_id": arm, "seed": seed, "turn_idx": i, "call_idx": j,
            "new_prompt": [{"n_ctx_slot": n_ctx, "n_keep": 4, "n_tokens": P}], "message_level_truncation": trunc,
            "system_kept": True if trunc else None, "token_cut": cut,
            "context_shifts": [{"n_keep": k, "n_left": 0, "n_discard": d} for k, d in shifts]}


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "results").mkdir()
    mech = [{"record": "run_start"}]
    # untrimmed default-tier session: P / est = 0.5 at every size (the reference ratio)
    for i in range(1, 5):
        mech.append(_turn(M, AD, 1, i, [1000 * i, 1000 * i + 100], {}))
        mech.append(_mcall(M, AD, 1, i, 1, 500 * i, n_ctx=131072))
        mech.append(_mcall(M, AD, 1, i, 2, 500 * i + 50, n_ctx=131072))
    # 4096 session: turn 1 untrimmed, turn 2 message-trimmed (est 6000 -> full 3000, kept 1500), turn 3 shift on
    # call 2 with a format violation on turn 4
    mech += [_turn(M, A4, 1, 1, [1000, 1100], {}), _mcall(M, A4, 1, 1, 1, 500), _mcall(M, A4, 1, 1, 2, 550),
             _turn(M, A4, 1, 2, [6000, 6000], {"rule5_session_code": False}),
             _mcall(M, A4, 1, 2, 1, 1500, trunc=True), _mcall(M, A4, 1, 2, 2, 1500, trunc=True),
             _turn(M, A4, 1, 3, [7000, 7000], {}), _mcall(M, A4, 1, 3, 1, 3000, trunc=True),
             _mcall(M, A4, 1, 3, 2, 4090, trunc=True, shifts=[(5, 2045)]),
             _turn(M, A4, 1, 4, [8000, 8000], {"rule1_json_keys": False}),
             _mcall(M, A4, 1, 4, 1, 4000, trunc=True), _mcall(M, A4, 1, 4, 2, 4000, trunc=True)]
    _write(tmp_path / "results/mech.jsonl", mech)
    real = [{"record": "run_start", "rules_in_use": {M: ["rule1_json_keys", "rule3_no_zebra"]}}]
    # real 4096 session: turn 1 P 1000 (< n_ctx/2, untrimmed), turn 2 P + C > 4096 on call 2 (predicted shift),
    # turn 3 cut signature P == 2050 on call 1; violation (rule3) on turn 3
    real += [_turn(M, A4, 7, 1, [2000, 2100], {}, mode="real", P=[1000, 1050], C=[20, 20]),
             _turn(M, A4, 7, 2, [9000, 9000], {}, mode="real", P=[3000, 4090], C=[20, 30]),
             _turn(M, A4, 7, 3, [9500, 9500], {"rule3_no_zebra": False}, mode="real", P=[2050, 3000], C=[20, 20])]
    _write(tmp_path / "results/real.jsonl", real)
    return tmp_path


def _build(repo):
    return il.build(repo, mech_files=(("evo-x2", "results/mech.jsonl"),), real_files=("results/real.jsonl",),
                    sys_est_fn=lambda seed: 500)


def test_observed_session_shares_and_reliability(repo):
    rep = _build(repo)
    tab = rep["cells"][("evo-x2", "observed", M, "num_ctx_4096")][0]
    by = {t["turn"]: t for t in tab["turns"]}
    assert tab["S"] == 400                                   # 500 chars/4 x calib 0.8
    assert by[1]["transcript_share"] == 0.0 and by[1]["system_share"] == 0.0
    assert by[2]["transcript_share"] == pytest.approx(0.5)   # kept 1500 of est 6000 x 0.5
    assert by[2]["system_share"] == 0.0                      # message trim keeps the system message
    assert by[3]["system_share"] == 1.0
    assert tab["L"] == 3
    # rules in use for llama3.1:8b here: rule1, rule3 (rule5 False on turn 2 is not a violation)
    assert (tab["before_v"], tab["before_n"], tab["after_v"], tab["after_n"]) == (0, 2, 1, 2)
    assert tab["lag"] == 1
    assert tab["by_rule"]["rule5_session_code"] == {"in_use": False, "before": 1, "before_n": 2, "after": 0,
                                                    "after_n": 2}
    assert tab["by_rule"]["rule1_json_keys"]["after"] == 1
    assert rep["system_kept_on_trimmed"] == (6, 6)


def test_predicted_session_uses_shift_predictor_and_cut_signature(repo):
    rep = _build(repo)
    tab = rep["cells"][("evo-x2", "predicted", M, "num_ctx_4096")][0]
    by = {t["turn"]: t for t in tab["turns"]}
    assert by[1]["system_share"] == 0.0 and by[1]["transcript_share"] == 0.0
    assert by[2]["system_share"] == 1.0                      # 4090 + 30 > 4096
    assert by[3]["system_share"] == 1.0                      # P == 4096 - (4096 - 4) // 2 == 2050
    assert tab["L"] == 2 and tab["lag"] == 1
    assert 0.0 < by[2]["transcript_share"] < 1.0


def test_reference_ratio_leaves_own_session_out(repo):
    rep = _build(repo)
    assert rep["est_check"][M]["n"] > 0
    sessions = {}
    for machine, f in (("evo-x2", "results/mech.jsonl"),):
        sessions.update(il.mechanism_sessions(il.read_rows(repo / f), machine, "mech"))
    ref = il.reference_points(sessions)
    own = ("evo-x2", "mech", M, "ollama_default", 1)
    assert il.ratio_at(ref, M, 2000) == pytest.approx(0.5, abs=0.01)
    assert il.ratio_at(ref, M, 2000, exclude_key=own) == pytest.approx(0.5, abs=0.01)  # the 4096 turn 1 calls


def test_register_strings_carry_labels(repo):
    rep = _build(repo)
    s1, s2 = il.info_loss_str(rep), il.reliability_str(rep)
    assert "observed, one session per tier" in s1 and "predicted" in s1
    assert "observed, one session per tier" in s2
    assert "evo-x2 llama3.1:8b num_ctx_4096 [observed, one session per tier, n 1]" in s2
    assert "session code (not in use) 1/2 -> 0/2" in s2
