"""Post-run check (operator, 2026-10-29; Prof. Hao's question "what if the prediction is wrong?"). Pre-registered in
docs/EXPERIMENTS.md "PRE-REGISTRATION (2026-10-29): post-run check and per-task cloud budget".

After every LOCAL task, before its answer is accepted:

  L1 runtime signal: the server's own log slice for the task's calls shows a cut (Ollama: detector v2,
     harness/own_compaction.cut_v2_verdict: message drop, "truncating input prompt", context shift, exceeds-context;
     llama-server: harness/xruntime_llamacpp.classify_log / call_class: rejection, context shift, truncation, context
     stop), or a call returned a context error -> FAIL.
  L2 context integrity: the system message and the tool-call example turn the task depends on are present in the
     prompt the server actually evaluated, judged on the server's own render (harness/x2_r2_mechanism.analyse_render)
     whose llama-tokenize count matches the server's prompt_eval_count (harness/prompt_token_check.compare). A render
     that does not match what the model saw is not citable and FAILs (never a chars/4 estimate).
  L3 task-rule invariants: src/dse/l3_invariants.py (system prompt, task text and tools schema only; never the
     grader, the reference calls, the expected end state or the answer).
  L4 (optional, separate): a cheap self-check by the local model. DESIGN ONLY here (L4_DESIGN); never part of
     decide()'s headline, measured separately (row post-run-check-l4-self-check, PENDING).

decide(): any FAIL -> "escalate" (rollback with harness/evalset_env.SnapshotLog to the pre-task mark, then the cloud
from that clean state under the per-task budget, src/cloud/task_budget.py); all PASS -> "accept". The layers that fired
are logged in order.

PROXIES (recorded runs without server logs or renders, e.g. results/x2_evalset_tasks_real_v2.jsonl): l1_proxy_record
(the recorded over_loaded_window condition) and l2_proxy_record (server prompt_eval_count below the client's chars/4
estimate of the prompt sent). They are labelled PROXY wherever they are reported and are never called L1 or L2.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from src.dse.l3_invariants import l3_check

REPO = Path(__file__).resolve().parents[2]
LAYERS = ("L1", "L2", "L3")
PROXY_LAYERS = ("L1_errors", "L1_proxy", "L2_proxy", "L3")

L4_DESIGN = {
    "what": ("after the task's last local call, one extra call to the same local model on the same session context "
             "(KV cache warm), think off, no tools, num_predict 16: 'Check your last answer against the rules in the "
             "system message and the user's request. Reply PASS or FAIL.' FAIL -> escalate"),
    "cost_basis": ("one prompt-eval of the appended check message plus up to 16 decoded tokens; the session prefix is "
                   "already in the KV cache unless the window was cut, in which case the whole prompt is re-evaluated"),
    "value": "needs live local calls; registered separately, not run (row post-run-check-l4-self-check)",
    "num_predict": 16,
}


def l4_design() -> dict:
    """L4 (local self-check): the design only. Its value needs live local calls; not run here."""
    return dict(L4_DESIGN)


def _harness():
    h = str(REPO / "harness")
    if h not in sys.path:
        sys.path.insert(0, h)


# ── L1 ─────────────────────────────────────────────────────────────────────────────────────────

def l1_runtime_signal(log_slices: list[str | None] | None = None, n_messages: list[int] | None = None,
                      runtime: str = "ollama", call_errors: list[str | None] | None = None,
                      llama_calls: list[dict] | None = None) -> dict:
    """FAIL when any call's server-log slice shows a cut or any call returned a context error. Ollama: log_slices[i]
    with n_messages[i] messages sent. llama-server: llama_calls[i] = {status, finish_reason, prompt_tokens,
    completion_tokens, n_ctx, log_text, error_text}."""
    evidence, no_log = [], 0
    for e in call_errors or []:
        if e and any(k in str(e).lower() for k in ("context", "exceed", "truncat")):
            evidence.append(f"call error: {str(e)[:120]}")
    if runtime == "ollama":
        _harness()
        import own_compaction as oc  # noqa: PLC0415
        for txt, n in zip(log_slices or [], n_messages or []):
            v = oc.cut_v2_verdict(txt, n)
            no_log += v["flagged"] is None
            evidence += v["evidence"]
    elif runtime == "llamacpp":
        _harness()
        import xruntime_llamacpp as xr  # noqa: PLC0415
        for c in llama_calls or []:
            cls = xr.call_class(c.get("status"), c.get("finish_reason"), c.get("prompt_tokens"),
                                c.get("completion_tokens"), c.get("n_ctx"), xr.classify_log(c.get("log_text")),
                                c.get("error_text"))
            if cls in ("flagged_reject", "silent_cut", "ctx_stop"):
                evidence.append(f"llama-server {cls}")
    else:
        raise ValueError(runtime)
    return {"layer": "L1", "fail": bool(evidence), "evidence": evidence, "calls_without_log": no_log,
            "computable": no_log == 0 or bool(evidence)}


# ── L2 ─────────────────────────────────────────────────────────────────────────────────────────

def l2_context_integrity(rendered: str | None, messages: list[dict], example_user_content: str | None = None,
                         render_tokens: int | None = None, prompt_eval_count: int | None = None) -> dict:
    """FAIL if the render is missing, not citable (its token count does not match prompt_eval_count within
    prompt_token_check.MATCH_REL_TOL), or lacks the system message, the current user message, or the tool-call example
    turn's user message (own_compaction.choose_example picks that turn)."""
    _harness()
    import prompt_token_check as ptc  # noqa: PLC0415
    import x2_r2_mechanism as mech  # noqa: PLC0415
    reasons = []
    if rendered is None:
        reasons.append("no render")
        return {"layer": "L2", "fail": True, "reasons": reasons, "citable": False}
    citable = True
    if render_tokens is not None and prompt_eval_count is not None:
        citable = bool(ptc.compare(render_tokens, prompt_eval_count)["match"])
        if not citable:
            reasons.append(ptc.NOT_CITABLE)
    a = mech.analyse_render(rendered, messages)
    if not a.get("system_kept"):
        reasons.append("system message (rules) not in the evaluated prompt")
    if not a.get("current_user_message_kept"):
        reasons.append("current user message not in the evaluated prompt")
    if example_user_content and example_user_content not in rendered:
        reasons.append("tool-call example turn not in the evaluated prompt")
    return {"layer": "L2", "fail": bool(reasons), "reasons": reasons, "citable": citable,
            "dropped_turns": a.get("dropped_turns")}


# ── recorded-run proxies (labelled PROXY) ───────────────────────────────────────────────────────

def l1_errors_record(turn: dict) -> dict:
    """The real part of L1 a recorded turn holds: a call that came back with an error or a non-ok outcome."""
    bad = [c for c in turn.get("calls") or [] if c.get("error") or (c.get("outcome") not in (None, "ok"))
           or (c.get("http_status") not in (None, 200))]
    return {"layer": "L1_errors", "fail": bool(bad), "evidence": [str(c.get("error"))[:120] for c in bad]}


def l1_proxy_record(turn: dict) -> dict:
    """PROXY for L1: the recorded over_loaded_window condition (impact_model.turn_over_window)."""
    v = turn.get("over_loaded_window")
    if v is None:
        p, w = turn.get("final_call_prompt_tokens_calibrated"), turn.get("loaded_context")
        v = p is not None and w is not None and p > w
    return {"layer": "L1_proxy", "fail": bool(v), "label": "PROXY: over_loaded_window, not the server log"}


def l2_proxy_record(turn: dict) -> dict:
    """PROXY for L2: a call whose server prompt_eval_count is below the client's chars/4 estimate of what was sent."""
    short = [i for i, c in enumerate(turn.get("calls") or [])
             if c.get("prompt_eval_count_info_only") and c.get("prompt_tokens_est")
             and c["prompt_eval_count_info_only"] < c["prompt_tokens_est"]]
    return {"layer": "L2_proxy", "fail": bool(short), "calls": short,
            "label": "PROXY: prompt_eval_count below the chars/4 estimate sent, not the render check"}


def l3_record(turn: dict, system_prompt: str, user_text: str, tools_schema: list[dict]) -> dict:
    calls = [c for rec in turn.get("calls") or [] for c in rec.get("tool_calls_parsed") or []]
    r = l3_check(system_prompt, user_text, tools_schema, calls, turn.get("final_text"))
    return {"layer": "L3", "fail": not r["pass"], **r}


def check_record(turn: dict, system_prompt: str, user_text: str, tools_schema: list[dict]) -> dict:
    """Every layer computable from a recorded turn, with the check's own compute time."""
    t0 = time.perf_counter()
    layers = [l1_errors_record(turn), l1_proxy_record(turn), l2_proxy_record(turn),
              l3_record(turn, system_prompt, user_text, tools_schema)]
    d = decide(layers)
    d["check_seconds"] = time.perf_counter() - t0
    d["layers"] = {x["layer"]: x for x in layers}
    return d


# ── decision ───────────────────────────────────────────────────────────────────────────────────

def decide(layer_results: list[dict]) -> dict:
    fired = [x["layer"] for x in layer_results if x.get("fail")]
    return {"decision": "escalate" if fired else "accept", "fired": fired}


def rollback_and_escalate(snapshots, pre_task_mark: str, redo_fn, budgeted_client, task_key: str) -> dict:
    """The FAIL path: restore the environment to the task's pre-task mark (evalset_env.SnapshotLog.restore_mark), then
    redo the task on the cloud from that clean state through the per-task budget. redo_fn(client, env) -> result dict.
    A per-task cap hit is a clean budget_exhausted outcome (hand-off), never a retry."""
    from src.cloud import task_budget as TB  # noqa: PLC0415
    snapshots.restore_mark(pre_task_mark)
    budgeted_client.start_task(task_key)
    try:
        res = redo_fn(budgeted_client, snapshots.env)
        out = {"outcome": "cloud", **res}
    except TB.TaskBudgetExhausted as e:
        out = {"outcome": TB.OUTCOME_EXHAUSTED, "cloud_end_state_ok": False, "task_budget": e.record}
    return {**out, **budgeted_client.task_state()}
