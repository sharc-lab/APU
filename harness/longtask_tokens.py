"""Token-matched long tasks (longtask-tokens-v1, pre-registered 2026-11-09 in docs/EXPERIMENTS.md).

The step-matched set (harness/longtask_env.py) matches each OpenHands trajectory's STEP count, but its tool results are
small JSON records, so its prompts stay near 11k tokens while the real traces reach 42k to 87k. This module keeps the
same tasks, tools and grader and adds an `activity_log` field to every tool result, sized so the reference solution's
prompt follows the matched trace's per-step token curve (Qwen tokens, harness/agent_trace_steps_uncensored.json). The
log text is neutral (no record id, no value a task depends on, no instruction), so it only takes context: it tests
whether the window, not the task logic, breaks a long task.

Sizing (pure, deterministic): step k of the reference solution (k = 0 .. ref_steps - 1) is matched to trace step
round(k * (S - 1) / (ref_steps - 1)) of the S-step trace; the pad after tool call k makes the reference prompt before
call k + 1 reach that target (model tokens = harness estimate x x2_r2_evalset.PROJ_CALIB). A model call past the
reference steps gets the last reference pad. Pure: no network, no clock, no process.
"""
from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path

import longtask_env as L
import x2_r2_agent as ag
import x2_r2_evalset as EV

LABEL = "long-horizon synthetic, trace-TOKEN matched (activity_log padding)"
SCHEDULE = Path(__file__).resolve().parent / "agent_trace_steps_uncensored.json"
PAD_FIELD = "activity_log"
_VERBS = ("index rebuild finished", "nightly sync completed", "cache warmed", "report archived", "backup verified",
          "quota check passed", "audit export written", "replica caught up", "schema check passed", "batch closed")


@lru_cache(maxsize=1)
def _traces() -> dict:
    d = json.loads(SCHEDULE.read_text(encoding="utf-8"))
    return {t["id"]: t for t in d["trajectories"]}


def trace_curve(trajectory_id: str) -> list[int]:
    return list(_traces()[trajectory_id]["tokens_qwen"])


def log_text(n_tokens: int, seed: int) -> str:
    """Neutral log lines of about n_tokens model tokens (harness estimate x PROJ_CALIB)."""
    if n_tokens <= 0:
        return ""
    rng = random.Random(seed)
    want_est = int(n_tokens / EV.PROJ_CALIB)
    lines, est = [], 0
    while est < want_est:
        ln = (f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d} worker-{rng.randint(1, 64)}"
              f" {rng.choice(_VERBS)}, {rng.randint(100, 99999)} rows, {rng.randint(0, 999)} ms")
        lines.append(ln)
        est += ag.est_tokens(ln + "\n")
    return "\n".join(lines)


def _reference_calls(task: L.LongTask, env_before: dict) -> list[dict]:
    msgs, _ = L.oracle_transcript(task, env_before)
    return [m["tool_calls"][0]["function"] for m in msgs if m.get("role") == "assistant" and m.get("tool_calls")]


def _model_tokens(msgs: list[dict]) -> int:
    return round(ag.prompt_tokens_est(msgs, L.tools_payload()) * EV.PROJ_CALIB)


def pad_tokens(task: L.LongTask, env_before: dict) -> list[int]:
    """Per reference tool call k, the activity_log size in model tokens (see the module docstring)."""
    curve = trace_curve(task.trajectory_id)
    calls = _reference_calls(task, env_before)
    n, S = len(calls), len(curve)
    target = [curve[round(k * (S - 1) / max(n - 1, 1))] for k in range(n)]
    msgs = [{"role": "system", "content": L.SYSTEM_PROMPT}, {"role": "user", "content": L.user_message(task)}]
    import copy
    env = copy.deepcopy(env_before)
    pads = []
    for k, c in enumerate(calls):
        res = L.execute(env, c)
        msgs.append({"role": "assistant", "content": "", "tool_calls": [{"function": dict(c)}]})
        msgs.append(L.tool_message(c, res))
        nxt = target[k + 1] if k + 1 < n else target[-1]
        need = max(0, nxt - _model_tokens(msgs))
        pads.append(need)
        if need:
            msgs[-1] = L.tool_message(c, {**res, PAD_FIELD: log_text(need, _pad_seed(task, k))})
    return pads


def _pad_seed(task: L.LongTask, k: int) -> int:
    return (L.TASK_IDS.index(task.task_id) + 1) * 100003 + k


class Padder:
    """pad(k, result) -> result with the activity_log for the k-th tool call the model makes in this task."""

    def __init__(self, task: L.LongTask, env_before: dict):
        self.task, self.pads = task, pad_tokens(task, env_before)

    def __call__(self, k: int, result: dict) -> dict:
        n = self.pads[k] if k < len(self.pads) else (self.pads[-1] if self.pads else 0)
        return {**result, PAD_FIELD: log_text(n, _pad_seed(self.task, k))} if n else result


def oracle_profile(task: L.LongTask, env_before: dict) -> dict:
    """The padded reference transcript's prompt (model tokens) before each call, against the trace curve."""
    import copy
    pad = Padder(task, env_before)
    calls = _reference_calls(task, env_before)
    env = copy.deepcopy(env_before)
    msgs = [{"role": "system", "content": L.SYSTEM_PROMPT}, {"role": "user", "content": L.user_message(task)}]
    prof = [_model_tokens(msgs)]
    for k, c in enumerate(calls):
        res = pad(k, L.execute(env, c))
        msgs.append({"role": "assistant", "content": "", "tool_calls": [{"function": dict(c)}]})
        msgs.append(L.tool_message(c, res))
        prof.append(_model_tokens(msgs))
    curve = trace_curve(task.trajectory_id)
    first_over = next((i for i, p in enumerate(prof) if p > 40960), None)
    return {"task_id": task.task_id, "ref_steps": task.ref_steps, "trace_steps": len(curve),
            "trace_max": max(curve), "oracle_final": prof[-1], "oracle_max": max(prof),
            "final_ratio": prof[-1] / max(curve), "first_call_over_40960": first_over,
            "oracle_ok": L.check_long_task(task, env_before, env)["end_state_ok"]}
