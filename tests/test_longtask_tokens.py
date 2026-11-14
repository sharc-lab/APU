"""harness/longtask_tokens.py (longtask-tokens-v1): the token-matched set, pure; and its use in run_long_task."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))

import longtask_env as L  # noqa: E402
import longtask_tokens as T  # noqa: E402
import x2_longtask as X  # noqa: E402

OH = [f"lt_openhands_{i:02d}" for i in range(1, 11)]


@pytest.mark.parametrize("seed", [20260907, 20260901])
def test_reference_prompt_follows_the_trace_and_still_passes(seed):
    for task, env in L.build_tasks(seed, OH):
        p = T.oracle_profile(task, env)
        assert p["oracle_ok"]
        assert 0.97 <= p["final_ratio"] <= 1.03, p
        assert p["oracle_max"] > 40960 and p["first_call_over_40960"] is not None


def test_padding_is_neutral_and_deterministic():
    txt = T.log_text(2000, 7)
    assert txt == T.log_text(2000, 7) and txt != T.log_text(2000, 8)
    for pat in (r"SKU-", r"\bT-\d", r"CU-", r"ACC-", r"PAY-", r"EV-", r"/srv/", r"@"):
        assert not re.search(pat, txt), pat
    assert T.log_text(0, 1) == ""


def test_padder_extends_past_the_reference_steps():
    task, env = L.build_tasks(20260907, ["lt_openhands_01"])[0]
    pad = T.Padder(task, env)
    r = pad(len(pad.pads) + 5, {"status": "ok"})
    assert (T.PAD_FIELD in r) == bool(pad.pads[-1])


def test_run_long_task_with_padding_oracle_correct_and_long():
    task, env = L.build_tasks(20260907, ["lt_openhands_01"])[0]
    msgs, _ = L.oracle_transcript(task, env)
    calls = [m["tool_calls"][0]["function"] for m in msgs if m.get("role") == "assistant" and m.get("tool_calls")]

    class RT:
        def __init__(self):
            self.i, self.max_len = 0, 0

        def chat(self, model, messages, num_ctx, tools, think, extra=None):
            self.max_len = max(self.max_len, sum(len(json.dumps(m)) for m in messages))
            if tools is not None and self.i < len(calls):
                c = calls[self.i]
                self.i += 1
                return {"outcome": "ok", "status": 200, "message": "", "tool_calls": [{"function": dict(c)}],
                        "prompt_eval_count": 1000, "eval_count": 30, "duration_s": 0.5, "raw": {"done_reason": "stop"}}
            return {"outcome": "ok", "status": 200, "message": "All done.", "tool_calls": None,
                    "prompt_eval_count": 1200, "eval_count": 10, "duration_s": 0.4, "raw": {"done_reason": "stop"}}

        def loaded_context(self, model):
            return 40960

        def recover(self):
            return False
    rt = RT()
    row = X.run_long_task(rt, "m", None, False, task, env, lambda *_: None, pad=T.Padder(task, env))
    assert row["correct"] and row["task_set"] == "tokens"
    assert row["max_prompt_tokens_est"] * X.PROJ_CALIB > 40960
    base = X.run_long_task(type(rt)(), "m", None, False, task, env, lambda *_: None)
    assert base["task_set"] == "steps" and base["max_prompt_tokens_est"] * X.PROJ_CALIB < 20000


def test_fitting_arm_holds_the_largest_reference_prompt():
    big = max(T.oracle_profile(t, e)["oracle_max"] for t, e in L.build_tasks(20260907, OH))
    assert X.ARMS["ctx_98304"]["num_ctx"] > big


# ── v2 sizing (longtask-tokens-v2): pads sized with the model's tokenizer ────────────────────────────────────────
def _qwen_tokenizer_json():
    import glob
    pytest.importorskip("tokenizers")
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    for repo in ("models--Qwen--Qwen3-30B-A3B-Instruct-2507", "models--Qwen--Qwen3-8B"):
        hits = glob.glob(str(hub / repo / "snapshots" / "*" / "tokenizer.json"))
        if hits:
            return hits[0]
    pytest.skip("no Qwen3 tokenizer.json in the local Hugging Face cache")


def test_log_lines_is_a_fixed_prefix_of_one_sequence():
    a, b = T.log_lines(50, 11), T.log_lines(80, 11)
    assert b.startswith(a) and a.count("\n") == 49 and T.log_lines(0, 11) == ""
    assert T.log_lines(50, 12) != a


def test_v2_reference_prompt_hits_the_trace_target_in_model_tokens():
    sz = T.TokenizerSizing(tokenizer_json=_qwen_tokenizer_json())
    for task, env in L.build_tasks(20260911, OH[:2]):
        p = T.oracle_profile(task, env, sz)
        assert p["oracle_ok"]
        assert 1.0 <= p["final_ratio"] <= 1.005           # within 0.5% above the trace's final step, never below


def test_v2_pad_is_the_smallest_line_count_reaching_each_target():
    import copy
    sz = T.TokenizerSizing(tokenizer_json=_qwen_tokenizer_json())
    task, env = L.build_tasks(20260911, OH[:1])[0]
    lines = T.pad_lines(task, env, sz)
    curve = T.trace_curve(task.trajectory_id)
    calls = T._reference_calls(task, env)
    n, S = len(calls), len(curve)
    target = [curve[round(k * (S - 1) / max(n - 1, 1))] for k in range(n)]
    e = copy.deepcopy(env)
    msgs = [{"role": "system", "content": L.SYSTEM_PROMPT}, {"role": "user", "content": L.user_message(task)}]
    for k, c in enumerate(calls[:3]):
        res = L.execute(e, c)
        msgs.append({"role": "assistant", "content": "", "tool_calls": [{"function": dict(c)}]})
        nxt = target[k + 1] if k + 1 < n else target[-1]
        seed = T._pad_seed(task, k)
        at = lambda m: sz.model_tokens(msgs + [L.tool_message(c, {**res, T.PAD_FIELD: T.log_lines(m, seed)} if m else res)])  # noqa: E731
        if lines[k]:
            assert at(lines[k]) >= nxt > at(lines[k] - 1)
        msgs.append(L.tool_message(c, {**res, T.PAD_FIELD: T.log_lines(lines[k], seed)} if lines[k] else res))


def test_v2_padder_and_v1_padder_differ_only_by_sizing():
    sz = T.TokenizerSizing(encode=lambda s: max(1, len(s) // 3))     # any counter: v2 pads are log_lines
    task, env = L.build_tasks(20260911, OH[:1])[0]
    v1, v2 = T.Padder(task, env), T.Padder(task, env, sz)
    r1, r2 = v1(0, {"ok": True}), v2(0, {"ok": True})
    assert r1["ok"] and r2["ok"]
    assert r2.get(T.PAD_FIELD, "") == T.log_lines(v2.pads[0], T._pad_seed(task, 0))


def test_tokenizer_sizing_refuses_without_a_tokenizer():
    import types
    args = types.SimpleNamespace(pad_sizing="tokenizer", tokenizer_json=None)
    with pytest.raises(SystemExit):
        X._pad_sizing(T, args)
    assert X._pad_sizing(T, types.SimpleNamespace(pad_sizing="estimate", tokenizer_json=None)) is None
