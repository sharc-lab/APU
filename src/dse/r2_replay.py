"""Rebuilds the exact per-call requests of an R2 session (results/x2_r2_real_v1.jsonl format) for Router replay.

The R2 harness (harness/x2_r2_agent.py) is deterministic in everything but the model's replies: the system prompt,
every user message and every simulated tool result are pure functions of the seed, and the model's replies are in
the rows (each call's content, native tool calls and parsed tool calls). So the transcript the harness sent on each
call is rebuilt here with the harness's own functions (build_agent_session, user_message, _assistant_msg,
_tool_msgs), and each step carries the harness's recorded prompt_tokens_est next to the rebuilt one so a
reconstruction error is visible (observed["reconstructed_prompt_tokens_est"] vs ["recorded_prompt_tokens_est"]).
One caveat from the harness: a call's content is stored cut to 4000 characters, so a reply longer than that
rebuilds short; the check above shows it when it happens.

Token counts are the harness's calibrated estimate: chars/4 + 4 per message (transcript_tokens) times the session's
own token_calib_ratio (turn-1 prompt_eval_count / estimate), rounded up per message.

needs_turns: a canary-check turn (every 5th) asks for history tag Hk, which lives only in turn 1's user message, so
those steps need turn 1. The system-prompt canaries and recall facts live in the system prompt, which is never
trimmed.
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "harness"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

R2_HARDWARE = "evo-x2"  # every x2_r2_* run is on evo-x2 (FINDINGS "R2 real run x2_r2_real_v1")


def _agent():
    import x2_r2_agent as agent  # noqa: PLC0415
    return agent


def read_rows(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def r2_sessions(rows: list[dict]) -> dict[tuple, list[dict]]:
    """r2a_turn rows grouped by (model_id, arm_id, seed, attempt_id), in file order, each sorted by turn."""
    out: dict[tuple, list[dict]] = {}
    for r in rows:
        if r.get("record") == "r2a_turn":
            out.setdefault((r["model_id"], r["arm_id"], r["seed"], r.get("attempt_id")), []).append(r)
    for k in out:
        out[k].sort(key=lambda r: r["turn_idx"])
    return out


def session_steps(srows: list[dict]):
    """Yields one Router step per model call of one session (call 1 and, when made, call 2 of every turn).

    Two ways to get an assistant reply's size: from its text when the rows carry each call's content (the result
    file does), or, when they do not (demo/dashboard's cache drops call content), as the residual of the next call's
    recorded prompt_tokens_est minus every other message and the tools, which the harness computed the same way.
    observed["reconstruction"] says which; both give the same counts on the real file (tested)."""
    agent = _agent()
    r2 = agent.r2
    seed = srows[0]["seed"]
    with contextlib.redirect_stdout(io.StringIO()):  # generate_session prints its filler-trim progress
        ags = agent.build_agent_session(seed, max(40, max(r["turn_idx"] for r in srows)))
    turns = {t.idx: t for t in ags.spec.turns}
    tools = r2.ollama_tools_payload()
    tools_raw = agent.est_tokens(json.dumps(tools))
    ratio = srows[0].get("token_calib_ratio") or 1.0
    num_predict = agent.MAX_TOKENS_PER_CALL
    have_content = all("content" in c for r in srows for c in r["calls"])

    def raw(m):
        return (agent.est_tokens(m.get("content")) + agent.PER_MESSAGE_OVERHEAD_TOKENS
                + (agent.est_tokens(json.dumps(m["tool_calls"])) if m.get("tool_calls") else 0))

    messages = [{"role": "system", "content": ags.system_prompt}]
    raws: list = [raw(messages[0])]  # None = an assistant reply whose size is not known yet (residual mode)

    def add(m, known=True):
        messages.append(m)
        raws.append(raw(m) if known else None)

    for row in srows:
        turn = turns[row["turn_idx"]]
        mode = agent.as_mode(row.get("call2_mode", row.get("call2_tools", True)))
        add({"role": "user", "content": agent.user_message(turn, ags)})
        loaded = row.get("loaded_context")
        for ci, rec in enumerate(row["calls"], start=1):
            step_tools = tools if (ci == 1 or mode != "off") else None
            t_raw = tools_raw if step_tools else 0
            unknown = [i for i, v in enumerate(raws) if v is None]
            if unknown:  # at most one: the reply made just before this call
                raws[unknown[0]] = max(0, rec["prompt_tokens_est"] - t_raw
                                       - sum(v for v in raws if v is not None))
            real_cal = round(rec["prompt_tokens_est"] * ratio)
            yield {
                "messages": [dict(m) for m in messages], "num_predict": num_predict, "tools": step_tools,
                "num_ctx": row.get("num_ctx_requested"),
                "token_counts": [math.ceil(v * ratio) for v in raws],
                "tools_tokens": math.ceil(t_raw * ratio),
                "needs_turns": [1] if row.get("canary_check") else [],
                "turn_idx": row["turn_idx"], "call_idx": ci,
                "observed": {
                    "model_id": row["model_id"], "arm_id": row["arm_id"], "seed": seed,
                    "canary_check": bool(row.get("canary_check")), "loaded_context": loaded,
                    "token_calib_ratio": ratio, "reconstruction": "content" if have_content else "residual",
                    "recorded_prompt_tokens_est": rec["prompt_tokens_est"],
                    "reconstructed_prompt_tokens_est": sum(raws) + t_raw,
                    "real_prompt_tokens_calibrated": real_cal,
                    "real_over_loaded_window": loaded is not None and real_cal > loaded,
                    "real_over_with_answer": loaded is not None and real_cal + num_predict > loaded,
                    "canary_hist_ok": row.get("canary_hist_ok"),
                },
            }
            resp = {"message": rec.get("content") or "", "tool_calls": rec.get("native_tool_calls")}
            parsed = rec.get("tool_calls_parsed") or []
            method = "none" if (ci == 1 and not parsed) else rec.get("tool_detection_method")
            add(agent._assistant_msg(resp, method), known=have_content)
            if parsed and not (ci == 1 and not parsed):
                for m in agent._tool_msgs(parsed, ags.store):
                    add(m)


def r2_steps(rows: list[dict]):
    for srows in r2_sessions(rows).values():
        yield from session_steps(srows)


def router_for_session(srows: list[dict], envelope=None, budget_usd: float = 5.0, quality_floor: float = 0.9,
                       latency_target_ms: float | None = None, cloud_client=None, **kwargs):
    """A Router matching one R2 session's arm: evo-x2, runtime = the arm id (Ollama default or num_ctx N), model =
    the session's model, token calibration = the session's own measured ratio."""
    from src.dse.router import Router  # noqa: PLC0415
    r = srows[0]
    kwargs.setdefault("token_calib_ratio", r.get("token_calib_ratio") or 1.0)
    return Router(envelope, budget_usd, quality_floor, latency_target_ms, R2_HARDWARE, r["arm_id"], r["model_id"],
                  cloud_client=cloud_client, **kwargs)
