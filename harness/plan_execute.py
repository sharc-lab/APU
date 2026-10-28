"""own-plan-execute pilot (DRAFT pre-registration in docs/EXPERIMENTS.md, 2026-10-25, "own-plan-execute pilot";
pointer in docs/FINDINGS.md). Built with a dry-run mode only: nothing here has run live, and the live job scripts
(cloud phases on the controller with the pre-job console gate, the local phase as an evo-x2 queue job) are NOT built.

Design (the same long-horizon synthetic tasks, tools and grader as harness/x2_longtask.py; never SWE-Gym):
  phase 1, plan (cloud, controller): the planner (claude-haiku-5-5) gets the task's own system prompt, user message
      and tools; its first call reads the work order (its tool calls are executed on a scratch copy, writes are never
      kept); a second call without tools returns a JSON plan, one self-contained step per work-order item. A plan that
      is unparseable or whose step count differs from the work order's numbered items is a plan failure: that task
      runs whole on the cloud (x2_longtask.run_long_task with the cloud runtime), and its cost counts for the arm.
  phase 2, execute (local, evo-x2): the executor (qwen3:14b, think false, ollama_default) runs each step as its own
      fresh short conversation (task system prompt, one user message with the step instruction, the same tools), up to
      STEP_CAP tool-bearing calls then one final call without tools, on the environment with the earlier local steps'
      writes applied. Every step's executed calls and results are recorded; the writes are staged, not final.
  phase 3, checkpoint and escalate (cloud, controller):
      checkpoint (a), deterministic, no grader: a step is flagged on a tool error, the step cap, an HTTP error,
          thinking output, a tool that is not offered, or no tool call at all (shape_flags);
      checkpoint (b), every CHECKPOINT_EVERY steps one verifier call (the planner model, no tools) sees each unflagged
          step's instruction, the records it read and the writes it proposes, and returns the step numbers to redo;
          an unparseable verdict redoes the whole batch;
      escalation: every flagged or redo step is run fresh by the cloud executor (the planner model) on the environment
          rebuilt from env_before plus the kept writes of the earlier steps; escalated steps are not re-verified.
      The final environment = env_before + per step, in order, the kept local writes or the escalated run's own
      executed calls; graded by longtask_env.check_long_task (end_state_ok).

The grader's reference writes are never used by phases 1 to 3. item_reference() rebuilds the per-item reference calls
for the dry-run fakes and the tests only.

    py -3.12 harness/plan_execute.py --dry-run            (fake planner / executor / verifier; no network, no model)
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
import statistics
import sys
import time
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))

import evalset_env as E  # noqa: E402
import longtask_env as L  # noqa: E402
import x2_r2_evalset as EV  # noqa: E402
import x2_r2_agent as ag  # noqa: E402

JOB_ID = "own_plan_execute_v1"
PLANNER = "claude-haiku-5-5"       # also the verifier and the cloud executor for escalated steps
BASELINE = "claude-haiku-5-5"      # the all-cloud arm (row long-task-cloud-priced-vs-billed: 5/5)
EXECUTOR = "qwen3:14b"
SEEDS = (20260901, 20260902, 20260903)
CHECKPOINT_EVERY = 8
STEP_CAP = 8
PLAN_MAX_TOKENS = 4096
VERIFY_MAX_TOKENS = 512
READ_TOOLS = ("get_record", "search_records")
TARGET_POINTS = 3.0
TARGET_COST_RATIO = 0.30
TARGET_TEXT = "within 3 points of all-cloud accuracy at under 30% of all-cloud cost"

PLAN_INSTRUCTION = (
    "Do not call any tool now. Write the plan for the work order you just read: a JSON object "
    '{"steps": [{"n": 1, "instruction": "..."}, ...]} with exactly one step per work-order item, in order. Each '
    "instruction must stand alone for an executor that has not seen the work order: name the table and the record to "
    "read, the condition to check, and the exact change to make in each case (or that nothing changes). Reply with "
    "the JSON only.")
VERIFY_SYSTEM = (
    "You check the work of an executor that carried out steps of a work order with tools. For each step you get its "
    "instruction, the records it read (with their values) and the writes it made. A step is wrong if a write does not "
    "follow from the instruction and the values read, if a needed write is missing, or if a write was not asked for. "
    'Reply with JSON only: {"redo": [step numbers that are wrong]} ({"redo": []} if all are right).')


def step_message(instruction: str) -> str:
    return "Carry out this one step of a work order with the tools, then reply with a short summary. " + instruction


# ── plan ────────────────────────────────────────────────────────────────────────────────────────────

_ITEM = re.compile(r"^\s*(\d+)\.\s+(.+?)\s*$", re.MULTILINE)


def work_order_items(text: str | None) -> list[str]:
    return [m.group(2) for m in _ITEM.finditer(text or "")]


def _json_obj(text: str | None):
    t = text or ""
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(t[i:j + 1])
    except ValueError:
        return None


def parse_plan(text: str | None, n_expected: int | None) -> tuple[list[dict] | None, str | None]:
    """(steps, problem): steps [{"n", "instruction"}] or None with the reason."""
    obj = _json_obj(text)
    steps = obj.get("steps") if isinstance(obj, dict) else None
    if not isinstance(steps, list) or not steps:
        return None, "no JSON plan with a non-empty steps list"
    out = []
    for k, s in enumerate(steps, 1):
        ins = s.get("instruction") if isinstance(s, dict) else None
        if not isinstance(ins, str) or not ins.strip():
            return None, f"step {k} has no instruction"
        out.append({"n": k, "instruction": ins.strip()})
    if n_expected is None:
        return None, "work order not read, step count cannot be checked"
    if len(out) != n_expected:
        return None, f"{len(out)} steps for {n_expected} work-order items"
    return out, None


def _drain(rt) -> list:
    return rt.drain() if hasattr(rt, "drain") else []


def plan_task(rt, model, task: L.LongTask, env_before: dict, log=print) -> dict:
    """Phase 1. Returns the plan row (record "plan"); steps None on a plan failure."""
    _drain(rt)
    tools = L.tools_payload()
    msgs = [{"role": "system", "content": L.SYSTEM_PROMPT}, {"role": "user", "content": L.user_message(task)}]
    resp1, rec1 = ag._do_call(rt, model, msgs, None, tools, None, log)
    raw, method = EV.extract_calls_v2(resp1.get("message"), resp1.get("tool_calls"))
    scratch, wo_text, calls, results = copy.deepcopy(env_before), None, [], []
    for c in raw:
        c2, _ = E.coerce_call(c)
        if c2.get("name") not in READ_TOOLS:        # the planner only reads; anything else is not applied
            res = {"status": "error", "error": "the planner does not write; call not applied"}
        else:
            res = L.execute(scratch, c2)
            if (res.get("status") == "ok" and res.get("table") == "files"
                    and res.get("record_id") == task.path):
                wo_text = (res.get("record") or {}).get("content")
        calls.append(c2)
        results.append(res)
    msgs.append(ag._assistant_msg(resp1, method))
    msgs.extend(L.tool_message(c, r) for c, r in zip(calls, results))
    msgs.append({"role": "user", "content": PLAN_INSTRUCTION})
    resp2, rec2 = ag._do_call(rt, model, msgs, None, None, None, log)
    n_items = len(work_order_items(wo_text)) if wo_text is not None else None
    steps, problem = parse_plan(resp2.get("message"), n_items)
    return {"record": "plan", "task_id": task.task_id, "model_id": model, "steps": steps, "plan_failed": steps is None,
            "problem": problem, "n_work_order_items": n_items, "planner_tool_calls": calls,
            "plan_text": (resp2.get("message") or "")[:8000], "calls": [rec1, rec2], "cloud_calls": _drain(rt)}


# ── one step (local executor, or the cloud executor on escalation) ────────────────────────────────────

def run_step(rt, model, num_ctx, think, instruction: str, env: dict, log=print, step_cap=STEP_CAP) -> dict:
    """Run one step to its end, applying its calls to `env` IN PLACE. Returns the step record."""
    tools = L.tools_payload()
    msgs = [{"role": "system", "content": L.SYSTEM_PROMPT}, {"role": "user", "content": step_message(instruction)}]
    recs, executed, final_text = [], [], ""
    t0 = time.monotonic()
    for i in range(step_cap + 1):
        offer = i < step_cap
        resp, rec = ag._do_call(rt, model, msgs, num_ctx, tools if offer else None, think, log)
        raw, method = (EV.extract_calls_v2(resp.get("message"), resp.get("tool_calls")) if offer else ([], "none"))
        calls = [E.coerce_call(c)[0] for c in raw]
        rec.update({"tool_calls_parsed": calls, "tool_detection_method": method, "tools_offered": offer})
        recs.append(rec)
        if calls:
            results = [L.execute(env, c) for c in calls]
            executed += [{"call": c, "result": r} for c, r in zip(calls, results)]
            msgs.append(ag._assistant_msg(resp, method))
            msgs.extend(L.tool_message(c, r) for c, r in zip(calls, results))
            continue
        final_text = resp.get("message") or ""
        break
    return {
        "instruction": instruction, "model_id": model, "executed": executed,
        "writes": [x["call"] for x in executed if x["call"].get("name") not in READ_TOOLS
                   and x["result"].get("status") == "ok"],
        "n_calls": len(recs), "n_tool_calls": len(executed),
        "n_tool_errors": sum(1 for x in executed if x["result"].get("status") != "ok"),
        "n_not_offered": sum(1 for x in executed if x["call"].get("name") not in L.OFFERED_TOOLS),
        "step_cap_hit": len(recs) > step_cap,
        "any_http_error": any((r["http_status"] not in (200, None)) or r["outcome"] != "ok" or r["error"]
                              for r in recs),
        "thinking_present": any(r.get("thinking_present") or r.get("think_tag_in_content") for r in recs),
        "final_text": final_text[:1000], "wall_s": time.monotonic() - t0,
        "call_seconds": [r.get("duration_s") for r in recs], "prompt_tokens_est": [r.get("prompt_tokens_est")
                                                                                    for r in recs]}


def shape_flags(step: dict) -> list[str]:
    """Checkpoint (a): deterministic flags from the step's own record (no grader, no reference)."""
    f = []
    if step["n_tool_errors"]:
        f.append("tool_error")
    if step["step_cap_hit"]:
        f.append("step_cap")
    if step["any_http_error"]:
        f.append("http_error")
    if step["thinking_present"]:
        f.append("thinking")
    if step["n_not_offered"]:
        f.append("not_offered")
    if not step["n_tool_calls"]:
        f.append("no_tool_call")
    return f


def execute_plan_local(rt, model, task: L.LongTask, env_before: dict, plan: dict, log=print, num_ctx=None,
                       think=False) -> list[dict]:
    """Phase 2: every plan step in order on one live copy of env_before (earlier local writes applied)."""
    live = copy.deepcopy(env_before)
    rows = []
    for st in plan["steps"]:
        r = run_step(rt, model, num_ctx, think, st["instruction"], live, log)
        r.update({"record": "local_step", "task_id": task.task_id, "n": st["n"], "flags": shape_flags(r)})
        rows.append(r)
    return rows


# ── checkpoint (b) and escalation ─────────────────────────────────────────────────────────────────────

def evidence(step: dict) -> dict:
    return {"n": step["n"], "instruction": step["instruction"],
            "reads": [{"call": x["call"], "result": x["result"].get("record", x["result"].get("ids"))}
                      for x in step["executed"] if x["call"].get("name") in READ_TOOLS],
            "writes": step["writes"]}


def verify_messages(batch: list[dict]) -> list[dict]:
    return [{"role": "system", "content": VERIFY_SYSTEM},
            {"role": "user", "content": json.dumps({"steps": [evidence(s) for s in batch]}, sort_keys=True)}]


def parse_verdict(text: str | None, ns: list[int]) -> tuple[set, bool]:
    """(steps to redo, parsed). Unparseable or out-of-batch numbers: the whole batch is redone."""
    obj = _json_obj(text)
    redo = obj.get("redo") if isinstance(obj, dict) else None
    if not isinstance(redo, list) or not all(isinstance(x, int) and x in ns for x in redo):
        return set(ns), False
    return set(redo), True


def finish_task(rt, model, task: L.LongTask, env_before: dict, plan: dict, steps: list[dict], log=print,
                every=CHECKPOINT_EVERY) -> dict:
    """Phase 3: checkpoints, escalations, the final environment and the grade (record "pe_task")."""
    _drain(rt)
    if plan["plan_failed"]:
        import x2_longtask as X
        row = X.run_long_task(rt, model, None, None, task, env_before, log)
        return {"record": "pe_task", "task_id": task.task_id, "plan_failed": True, "correct": row["correct"],
                "end_state_ok": row["end_state_ok"], "end_state_diff": row["end_state_diff"], "n_steps": 0,
                "fallback": "whole task on the cloud", "fallback_n_calls": row["n_calls"],
                "cloud_calls": _drain(rt)}
    shape = {s["n"] for s in steps if s["flags"]}
    verdicts, redo = [], set()
    clean = [s for s in steps if s["n"] not in shape]
    for i in range(0, len(clean), every):
        batch = clean[i:i + every]
        resp, rec = ag._do_call(rt, model, verify_messages(batch), None, None, None, log)
        r, ok = parse_verdict(resp.get("message"), [s["n"] for s in batch])
        redo |= r
        verdicts.append({"ns": [s["n"] for s in batch], "redo": sorted(r), "parsed": ok,
                         "text": (resp.get("message") or "")[:2000]})
    escalate = shape | redo
    env, esc = copy.deepcopy(env_before), []
    for s in steps:
        if s["n"] in escalate:
            e = run_step(rt, model, None, None, s["instruction"], env, log)
            e.update({"n": s["n"], "reason": "shape:" + ",".join(s["flags"]) if s["n"] in shape else "verifier"})
            esc.append(e)
        else:
            for c in s["writes"]:
                L.execute(env, c)
    chk = L.check_long_task(task, env_before, env)
    return {"record": "pe_task", "task_id": task.task_id, "plan_failed": False, "correct": bool(chk["end_state_ok"]),
            "end_state_ok": chk["end_state_ok"], "end_state_diff": chk["end_state_diff"], "n_diff": chk["n_diff"],
            "n_steps": len(steps), "n_flagged_shape": len(shape), "n_redo_verifier": len(redo),
            "n_escalated": len(escalate), "verdicts": verdicts, "escalations": esc,
            "local_seconds": sum(s["wall_s"] for s in steps), "cloud_calls": _drain(rt)}


def run_task(cloud_rt, local_rt, task, env_before, log=print) -> dict:
    """All three phases in one process (dry run and tests; the live job splits them across machines)."""
    plan = plan_task(cloud_rt, PLANNER, task, env_before, log)
    steps = [] if plan["plan_failed"] else execute_plan_local(local_rt, EXECUTOR, task, env_before, plan, log)
    row = finish_task(cloud_rt, PLANNER, task, env_before, plan, steps, log)
    row["plan_cloud_calls"] = plan["cloud_calls"]
    return row


# ── pre-registered reading (pure) ─────────────────────────────────────────────────────────────────────

def evaluate(pe_rows: list[dict], cloud_rows: list[dict], pe_usd_billed: float, cloud_usd_billed: float) -> dict:
    """PASS iff plan-execute accuracy >= all-cloud accuracy - 3 points AND plan-execute cloud billed USD < 0.30 x
    all-cloud billed USD (same tasks and seeds). Otherwise FAIL, with which part failed."""
    a_pe = sum(1 for r in pe_rows if r["correct"]) / len(pe_rows)
    a_cl = sum(1 for r in cloud_rows if r["correct"]) / len(cloud_rows)
    diff = 100 * (a_pe - a_cl)
    ratio = pe_usd_billed / cloud_usd_billed if cloud_usd_billed > 0 else float("inf")
    acc_ok, cost_ok = diff >= -TARGET_POINTS, ratio < TARGET_COST_RATIO
    return {"accuracy_plan_execute": a_pe, "accuracy_all_cloud": a_cl, "diff_points": diff, "cost_ratio": ratio,
            "accuracy_ok": acc_ok, "cost_ok": cost_ok, "verdict": "PASS" if acc_ok and cost_ok else "FAIL"}


# ── cost and hours estimate (pure; inputs are the measured rows) ──────────────────────────────────────

ASSUME = {   # every number here is an ASSUMPTION, labelled as such wherever it is reported
    "plan_instruction_tokens": 150,
    "plan_out_per_item_factor": {"point": 1.5, "high": 3.0},      # x measured work-order tokens per item
    "verify_out_tokens": {"point": 40, "high": 100},
    "esc_calls_per_step": {"point": 3, "high": 6},
    "plan_failure_share": {"point": 0.0, "high": 0.10},
    "all_cloud_high_factor": 1.5,                                 # longtask_cloud MARGIN
    "local_calls_per_step": {"point": 3, "high": 6},
    "overhead_s_per_arm": 120.0,                                  # x2_longtask.hours_estimate default
    "short_prompt_est": 3000,
}


def _haiku_rows(cloud_rows, model=PLANNER):
    return [r for r in cloud_rows if r.get("record") == "longtask_cloud_task" and r["model_id"] == model]


def measured_inputs(cloud_rows, local_rows, model=PLANNER, assume=ASSUME) -> dict:
    tr = _haiku_rows(cloud_rows, model)
    c1_in = statistics.median(r["cloud_calls"][0]["input_tokens"] for r in tr)
    c1_out = statistics.median(r["cloud_calls"][0]["output_tokens"] for r in tr)
    # work-order read: call 2 input - call 1 input - call 1 output, least squares on the item count
    xs = [r["n_items"] for r in tr]
    ys = [r["cloud_calls"][1]["input_tokens"] - r["cloud_calls"][0]["input_tokens"] - r["cloud_calls"][0]["output_tokens"]
          for r in tr]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    a = my - b * mx
    growth = []
    for r in tr:
        cc, cr = r["cloud_calls"], r["calls"]
        if len(cc) != len(cr):
            continue
        for k in range(len(cc) - 1):
            m = len(cr[k].get("tool_calls_parsed") or [])
            if m:
                growth.append((cc[k + 1]["input_tokens"] - cc[k]["input_tokens"] - cc[k]["output_tokens"]) / m)
    billed = sum(r["usd_billed"] for r in tr)
    upper = sum(r["usd_upper_bound"] for r in tr)
    warm = [r for r in cloud_rows if r.get("record") == "warmup" and r.get("model_id", model) == model][:1]
    lt = [r for r in local_rows if r.get("record") == "longtask_task" and r.get("arm") == "ollama_default"]
    cs = sorted(c["duration_s"] for r in lt for c in r["calls"]
                if (c.get("prompt_tokens_est") or 0) < assume["short_prompt_est"] and c.get("duration_s"))
    return {"call1_in": c1_in, "call1_out": c1_out, "wo_tokens_intercept": a, "wo_tokens_per_item": b,
            "tool_step_growth": statistics.median(growth), "n_growth": len(growth),
            "billed_over_upper": billed / upper, "points": sorted((r["ref_steps"], r["usd_billed"],
                                                                   r["usd_upper_bound"]) for r in tr),
            "warmup_billed": warm[0]["usd_ledger"] if warm else 0.0,
            "warmup_upper": (warm[0].get("usd_ledger_upper_bound") or warm[0]["usd_ledger"]) if warm else 0.0,
            "local_short_call_s_median": statistics.median(cs),
            "local_short_call_s_p75": statistics.quantiles(cs, n=4, method="inclusive")[2], "n_local_short_calls": len(cs),
            "local_item_diff_share": sum(r["n_diff"] for r in lt) / sum(r["n_items"] for r in lt),
            "n_local_tasks": len(lt)}


def interp(points, s, idx):
    """Linear interpolation of points[i][idx] over ref steps; below the first point scaled by s / s_min (cost per step
    rises with length, so this over-states short tasks), above the last the last segment extended."""
    if s <= points[0][0]:
        return points[0][idx] * s / points[0][0]
    for (s0, *v0), (s1, *v1) in zip(points, points[1:]):
        if s <= s1:
            return v0[idx - 1] + (v1[idx - 1] - v0[idx - 1]) * (s - s0) / (s1 - s0)
    (s0, *v0), (s1, *v1) = points[-2], points[-1]
    return v1[idx - 1] + (v1[idx - 1] - v0[idx - 1]) * (s - s1) / (s1 - s0)


def call_upper(rates: dict, prompt: float, out: float) -> float:
    """One call's price with every input token uncached (scripts/cloud_redo.py::call_cost, short/long rate)."""
    above = rates.get("long_context_above")
    r = rates["long"] if above is not None and prompt > above else rates["short"]
    return (prompt * r["input"] + out * r["output"]) / 1e6


def model_rates(table: dict, model_id: str) -> dict:
    m = table["models"][model_id]
    return {"short": m["standard"]["short"], "long": m["standard"]["long"],
            "long_context_above": m.get("long_context_above")}


def task_specs(seeds=SEEDS) -> list[tuple]:
    """(task_id, seed, ref_steps, n_items) of the 60 pilot tasks (pure: longtask_env.build_tasks)."""
    return [(t.task_id, s, t.ref_steps, len(t.items)) for s in seeds for t, _ in L.build_tasks(s)]


def cost_estimate(m: dict, specs: list[tuple], rates: dict, wrong_rate_point: float, assume=ASSUME) -> dict:
    """Cloud USD per arm, point and high, billed and upper bound (all input uncached). Billed point = upper point x
    the measured billed/upper ratio; billed high = upper high (no cache saving, ASSUMPTION)."""
    esc_share = {"point": wrong_rate_point, "high": m["local_item_diff_share"]}
    out = {}
    for case in ("point", "high"):
        ac_b = sum(interp(m["points"], s, 1) for _, _, s, _ in specs)
        ac_u = sum(interp(m["points"], s, 2) for _, _, s, _ in specs)
        f = assume["all_cloud_high_factor"] if case == "high" else 1.0
        ac = {"billed": ac_b * f + m["warmup_billed"], "upper": ac_u * f + m["warmup_upper"]}
        planner = verifier = escal = fallback = 0.0
        n_esc_steps = 0.0
        for _, _, s, n in specs:
            c2_in = m["call1_in"] + m["call1_out"] + m["wo_tokens_intercept"] + m["wo_tokens_per_item"] * n
            plan_out = assume["plan_out_per_item_factor"][case] * m["wo_tokens_per_item"] * n
            planner += call_upper(rates, m["call1_in"], m["call1_out"]) + \
                call_upper(rates, c2_in + assume["plan_instruction_tokens"], plan_out)
            ev_item = m["wo_tokens_per_item"] + 2 * m["tool_step_growth"]
            for k in range(math.ceil(n / CHECKPOINT_EVERY)):
                nb = min(CHECKPOINT_EVERY, n - k * CHECKPOINT_EVERY)
                verifier += call_upper(rates, m["call1_in"] + nb * ev_item, assume["verify_out_tokens"][case])
            ne = esc_share[case] * n
            n_esc_steps += ne
            step_in = m["call1_in"] + m["wo_tokens_per_item"]
            per_step = sum(call_upper(rates, step_in + j * (m["call1_out"] + m["tool_step_growth"]), m["call1_out"])
                           for j in range(assume["esc_calls_per_step"][case]))
            escal += ne * per_step
            fallback += assume["plan_failure_share"][case] * interp(m["points"], s, 2) * f
        pe_u = planner + verifier + escal + fallback + m["warmup_upper"]
        pe_b = pe_u * m["billed_over_upper"] if case == "point" else pe_u
        if case == "high":
            ac["billed"] = ac["upper"]
        out[case] = {"all_cloud": ac, "plan_execute": {"billed": pe_b, "upper": pe_u, "planner_upper": planner,
                                                         "verifier_upper": verifier, "escalation_upper": escal,
                                                         "fallback_upper": fallback, "escalated_steps": n_esc_steps},
                     "local_only": {"billed": 0.0, "upper": 0.0},
                     "total": {"billed": ac["billed"] + pe_b, "upper": ac["upper"] + pe_u},
                     "cost_ratio_billed": pe_b / ac["billed"], "escalation_share": esc_share[case]}
    return out


def hours_estimate(m: dict, specs: list[tuple], assume=ASSUME) -> dict:
    """evo-x2 hours of phase 2 (the only local part): steps x calls per step x measured short-call seconds."""
    n_steps = sum(n for *_, n in specs)
    out = {}
    for case, sec in (("point", m["local_short_call_s_median"]), ("high", m["local_short_call_s_p75"])):
        s = n_steps * assume["local_calls_per_step"][case] * sec + assume["overhead_s_per_arm"]
        out[case] = s / 3600
    out["n_steps"] = n_steps
    return out


# ── per-item reference (dry-run fakes and tests only; never used by phases 1 to 3) ─────────────────────

def item_reference(task: L.LongTask) -> list[list[dict]]:
    """Per work-order item, the reference calls (its read, then its writes), grouped as oracle_transcript does."""
    reads, writes = list(task.reference_reads[1:]), list(task.reference_writes)
    out = []
    for kind in task.item_kinds:
        seq = []
        if kind not in ("pay", "price"):
            seq.append(reads.pop(0))
        if kind in ("pay", "price", "notify"):
            seq.append(writes.pop(0))
        elif writes and L._write_matches_last_read(writes[0], seq[-1]):
            seq.append(writes.pop(0))
        out.append(seq)
    return out


class OracleCloud:
    """Dry-run fake for the cloud side: reads the work order, returns the items as the plan, verifies with an empty
    redo list (or `redo_all`), and executes an escalated step with its reference calls."""

    def __init__(self, task, redo_all=False):
        self.task, self.redo_all, self.ref = task, redo_all, item_reference(task)
        self.n = 0
        self._step = None

    def chat(self, model, messages, num_ctx, tools, think, extra=None):
        self.n += 1
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        last = messages[-1]
        if messages[0]["content"] == VERIFY_SYSTEM:
            ns = [s["n"] for s in json.loads(user)["steps"]]
            return _text(json.dumps({"redo": ns if self.redo_all else []}))
        if last["role"] == "user" and last["content"] == PLAN_INSTRUCTION:
            return _text(json.dumps({"steps": [{"n": k, "instruction": t} for k, t in enumerate(self.task.items, 1)]}))
        if user == L.user_message(self.task) and tools is not None:
            return _call({"name": "get_record", "arguments": {"table": "files", "record_id": self.task.path}})
        return _replay(self.task, self.ref, messages, tools)

    def recover(self):
        return False


class OracleLocal:
    """Dry-run fake for the local executor: each step's reference calls, except steps in `wrong` (index from 1),
    where it writes nothing and answers in text (a silent wrong step)."""

    def __init__(self, task, wrong=()):
        self.task, self.ref, self.wrong = task, item_reference(task), set(wrong)

    def chat(self, model, messages, num_ctx, tools, think, extra=None):
        user = messages[1]["content"]
        k = next((i for i, t in enumerate(self.task.items, 1) if user == step_message(t)), None)
        if k in self.wrong:
            return _text("Done.")
        return _replay(self.task, self.ref, messages, tools)

    def recover(self):
        return False


def _text(t):
    return {"outcome": "ok", "status": 200, "message": t, "tool_calls": None, "duration_s": 0.01,
            "raw": {"done_reason": "stop"}}


def _call(c):
    return {"outcome": "ok", "status": 200, "message": "", "tool_calls": [{"function": dict(c)}], "duration_s": 0.01,
            "raw": {"done_reason": "stop"}}


def _replay(task, ref, messages, tools):
    user = messages[1]["content"]
    k = next((i for i, t in enumerate(task.items) if user == step_message(t)), None)
    done = sum(1 for m in messages if m["role"] == "tool")
    seq = ref[k] if k is not None else []
    if tools is not None and done < len(seq):
        return _call(seq[done])
    return _text("Step done.")


def dry_run(seed=SEEDS[0], task_ids=("lt_swegym_06", "lt_openhands_03"), log=print) -> list[dict]:
    rows = []
    for task, env in L.build_tasks(seed, task_ids):
        wrong = {2, 5}
        row = run_task(OracleCloud(task), OracleLocal(task, wrong=wrong), task, env, log=lambda *a: None)
        rows.append(row)
        log(f"{task.task_id}: steps {row['n_steps']}, flagged {row['n_flagged_shape']} (local silent-wrong fakes at "
            f"steps {sorted(wrong)}), verifier redo {row['n_redo_verifier']}, escalated {row['n_escalated']}, "
            f"correct {row['correct']}")
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="fake planner, executor and verifier; no network")
    a = ap.parse_args(argv)
    if not a.dry_run:
        print("refused: only --dry-run is built; the live job scripts are not built", file=sys.stderr)
        return 2
    dry_run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
