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
