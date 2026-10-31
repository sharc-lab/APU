"""Live post-run check, local phase on evo-x2 (docs/EXPERIMENTS.md "AMENDMENT (2026-10-31)"): the L2 render wrapper used
by harness/x2_r2_evalset.py --plan post_run_check_live_v1 (B3 v2 qwen3:14b, num_ctx 8192, seed 20260905).

The plan already runs under the job's own Ollama server with OLLAMA_DEBUG=1 and own_compaction.PolicySessionRuntime
(no context policy at a plain ollama_ctx_* tier), so every task call carries detector v2 (L1, cut_v2 in the turn row).
L2RenderRuntime sits OUTSIDE it: before each real call it sends the render-only request with identical arguments
(x2_r2_agent.OllamaRuntime.render_only), waits for that request's own log line so the render never lands in the real
call's detector slice, then forwards the real call. Per call it emits one `prc_l2_call` row:

  system_kept, current_user_message_kept, dropped_turns (x2_r2_mechanism.analyse_render); example_turn (the earliest
  completed turn with log_event plus another tool, own_compaction.choose_example) and example_kept; l2_fail.

Citability: x2_r2_mechanism.check_due picks calls for a fresh-load prompt-token check (prompt_token_check, render
tokens vs prompt_eval_count; at most 5 + 8 per session); each is one `prc_l2_prompt_check` row. The controller phase
(scripts/post_run_check_cloud.py) FAILs L2 for a turn whose render shows a loss, and for every turn at or after a
non-matching prompt check (the render is then not citable).

Nothing here talks to the cloud. Pure parts are tested in tests/test_post_run_check_live.py with a fake runtime.
"""
from __future__ import annotations

import time

import own_compaction as OCX
import x2_r2_mechanism as mech


def example_user_content(messages: list[dict]) -> tuple[int | None, str | None]:
    """(turn number, user message content) of the earliest completed turn showing log_event plus a task tool."""
    turns, cur = [], None
    for i, m in enumerate(messages):
        if m.get("role") == "user":
            cur = [i]
            turns.append(cur)
        elif cur is not None:
            cur.append(i)
    done = turns[:-1]   # the last turn is the current one
    k = OCX.choose_example(messages, done)
    if k is None:
        return None, None
    return k + 1, messages[done[k][0]].get("content")


def l2_row(rendered: str | None, messages: list[dict]) -> dict:
    a = mech.analyse_render(rendered, messages)
    ex_turn, ex = example_user_content(messages)
    ex_kept = None if ex is None or rendered is None else ex in rendered
    fail = (rendered is None or not a.get("system_kept") or not a.get("current_user_message_kept")
            or ex_kept is False)
    return {"render_ok": a.get("render_ok"), "system_kept": a.get("system_kept"),
            "current_user_message_kept": a.get("current_user_message_kept"), "dropped_turns": a.get("dropped_turns"),
            "example_turn": ex_turn, "example_kept": ex_kept, "rendered_sha256": a.get("rendered_sha256"),
            "l2_fail": bool(fail)}


class L2RenderRuntime:
    def __init__(self, inner, base, log_path, emit, checker="default", sleep=time.sleep, clock=time.monotonic,
                 gin_wait_s=mech.GIN_WAIT_S):
        self.inner, self.base, self.log_path, self.emit = inner, base, log_path, emit
        self.checker = mech.default_checker(base) if checker == "default" else checker
        self.sleep, self.clock, self.gin_wait_s = sleep, clock, gin_wait_s
        self.ctx = {}
        self._k = {True: 0, False: 0}
        self._n = {True: 0, False: 0}

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def begin_session(self, model, arm, seed, es):
        self.ctx = {"model_id": model, "arm_id": arm, "seed": seed}
        self._k, self._n = {True: 0, False: 0}, {True: 0, False: 0}
        begin = getattr(self.inner, "begin_session", None)
        return begin(model, arm, seed, es) if begin else None

    def _wait(self, start):
        deadline = self.clock() + self.gin_wait_s
        while True:
            pos = mech.log_size(self.log_path)
            if pos < start or mech._PATTERNS["gin"].search(mech.read_log(self.log_path, start, pos)):
                return pos
            if self.clock() >= deadline:
                return pos
            self.sleep(0.1)

    def chat(self, model, messages, num_ctx, tools, think, extra=None):
        snap = [dict(m) for m in messages]
        p0 = mech.log_size(self.log_path)
        rendered, rerr = self.base.render_only(model, snap, num_ctx, tools, think, extra)
        self._wait(p0)
        resp = (self.inner.chat(model, messages, num_ctx, tools, think, extra=extra) if extra
                else self.inner.chat(model, messages, num_ctx, tools, think))
        turn_idx = sum(1 for m in snap if m.get("role") == "user")
        row = {"record": "prc_l2_call", **self.ctx, "num_ctx_requested": num_ctx, "turn_idx": turn_idx,
               "n_messages_sent": len(snap), "render_error": rerr, "prompt_eval_count": resp.get("prompt_eval_count"),
               **l2_row(rendered, snap), "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        self.emit(row)
        post = bool(row["dropped_turns"]) or not row["system_kept"]
        k = self._k[post]
        self._k[post] += 1
        if self.checker is not None and rendered is not None and mech.check_due(post, k, self._n[post]):
            self._n[post] += 1
            try:
                res = self.checker.check(model, snap, num_ctx, tools, think, extra, rendered)
            except Exception as e:
                res = {"error": f"check: {e!r}"[:400], "match": False, "diff": None, "rel_diff": None}
            self.emit({"record": "prc_l2_prompt_check", **self.ctx, "turn_idx": turn_idx, "post_overflow": post,
                       "rendered_sha256": row["rendered_sha256"], "real_call_prompt_eval_count":
                       row["prompt_eval_count"], **res,
                       "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        return resp


def turn_l2(l2_calls: list[dict], checks: list[dict], seed, turn_idx) -> dict:
    """Pure. L2 verdict of one task turn from the session's prc_l2_call and prc_l2_prompt_check rows."""
    calls = [r for r in l2_calls if r["seed"] == seed and r["turn_idx"] == turn_idx]
    bad_check = [c for c in checks if c["seed"] == seed and c["turn_idx"] <= turn_idx and not c.get("match")]
    reasons = []
    if not calls:
        reasons.append("no render row for this turn")
    if any(r.get("l2_fail") for r in calls):
        reasons.append("render lacks the rules, the current request or the tool-call example")
    if bad_check:
        reasons.append("render not citable (a prompt-token check did not match at or before this turn)")
    return {"layer": "L2", "fail": bool(reasons), "reasons": reasons}
