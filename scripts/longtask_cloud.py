"""All-cloud on the long-horizon synthetic tasks, small sample (longtask_cloud_v1): priced estimate vs billed cost.

Pre-registered in docs/EXPERIMENTS.md, 2026-10-22, "Long-task cloud priced vs billed" (pointer in docs/FINDINGS.md).
Same task definitions, tool loop and grader as harness/x2_longtask.py (x2_longtask.run_long_task with
longtask_env's tasks; "long-horizon synthetic, trace-length matched", never SWE-Gym): CLOUD_TASKS (5 tasks spanning
the step range) x seed 20260901 on claude-haiku-5-5, thinking disabled as scripts/all_cloud_b3.py does
(all_cloud_b3.THINKING_OFF), temperature 0, max_tokens 384, prompt caching on (breakpoints on the system prompt and
the last user message, all_cloud_b3.CloudRuntime unchanged except that a final call after the step cap keeps this
task set's tools).

Per task: provider tokens (input, uncached, cache read, cache write, output), billed USD (the client's billed cost,
src/cloud/client.py billed_cost_usd, ledger column cost_usd), upper bound USD (every input token uncached, ledger
column cost_usd_upper_bound), seconds (task wall and summed send time), correct (end_state_ok). The priced estimate:
analysis/cloud_task_cost.task_cost (the method of row cloud-cost-per-real-task) on the run's own per-call provider
input and output tokens, cases none / realistic (ASSUMPTION hit rate 0.7) / best.

Budget: job cap USD 6 (client spend_cap_usd = ledger running total at start + 6). Before any measured task: one
warm-up call (excluded) and a one-call pilot (the first task's first call, excluded) giving the provider/estimate
token ratio; projection = every reference step of the 5 tasks as one call at the upper-bound price (all input
uncached, 160 output tokens per call), x 1.5 (step cap) x 1.5 (margin). Over the cap: refuse, nothing measured.
Sonnet: after the haiku tasks, the estimate for the same 5 tasks on claude-sonnet-5-5 = the haiku calls' measured
token breakdown priced at Sonnet rates x 1.5; Sonnet runs only if that estimate is under USD 5 and the job total stays
under the cap.

Operator command (controller laptop, repo root; the key is loaded into this one process only):
    $env:ANTHROPIC_API_KEY = [Environment]::GetEnvironmentVariable('ANTHROPIC_API_KEY','User')
    $env:CLOUD_CLIENT_HOST_ROLE = 'controller'
    py -3.12 scripts/longtask_cloud.py --live --ledger <main checkout>/results/cloud_ledger.jsonl
    Remove-Item Env:\\ANTHROPIC_API_KEY
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "harness", REPO / "src", REPO / "analysis"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import cloud_task_cost as CTC  # noqa: E402
import longtask_env as L  # noqa: E402
import x2_longtask as X  # noqa: E402
import x2_r2_agent as ag  # noqa: E402
from cloud import client as cc  # noqa: E402
from cloud import console_gate  # noqa: E402

_spec = importlib.util.spec_from_file_location("all_cloud_b3", REPO / "scripts" / "all_cloud_b3.py")
ACB = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ACB)
CR = ACB.CR

HARNESS = "longtask_cloud"
OUT = REPO / "results" / "longtask_cloud_v1.jsonl"
PRICE_FILE = REPO / "results" / "cloud_prices_anthropic_20261014.json"
SEED = 20260901
CLOUD_TASKS = ("lt_swegym_06", "lt_swegym_10", "lt_openhands_03", "lt_openhands_06", "lt_openhands_10")
HAIKU, SONNET = "claude-haiku-5-5", "claude-sonnet-5-5"
JOB_CAP_USD = 6.0
SONNET_MAX_USD = 5.0
MARGIN = 1.5
OUT_TOKENS_PER_CALL_ASSUMED = 160
COST_KEYS = ACB.COST_KEYS


def upper_bound_usd(rates: dict, prompt_tokens: int, out_tokens: int) -> float:
    return CR.call_cost(rates, prompt_tokens, 0, 0, out_tokens, prompt_tokens)


def projection(tasks, rates: dict, ratio: float, cap_factor=L.STEP_CAP_FACTOR, margin=MARGIN) -> float:
    """Upper-bound USD of running `tasks` [(task, env)]: each oracle step one call (prompt est x ratio, all uncached,
    OUT_TOKENS_PER_CALL_ASSUMED output), x cap_factor x margin."""
    tools = L.tools_payload()
    tot = 0.0
    for task, env in tasks:
        msgs, _ = L.oracle_transcript(task, env)
        for k in range(task.ref_steps + 1):
            p = round(ag.prompt_tokens_est(msgs[:2 + 2 * k], tools) * ratio)
            tot += upper_bound_usd(rates, p, OUT_TOKENS_PER_CALL_ASSUMED)
    return tot * cap_factor * margin


def reprice(calls: list[dict], rates: dict) -> float:
    """The measured per-call token breakdown priced at another model's rates."""
    return sum(CR.call_cost(rates, c["uncached_input_tokens"], c["cache_write_tokens"], c["cache_read_tokens"],
                            c["output_tokens"], c["input_tokens"]) for c in calls)


def priced_estimate(prices: dict, model: str, calls: list[dict]) -> dict:
    """cloud_task_cost.task_cost on this task's own per-call provider input / output tokens, per caching case."""
    prompts = [c["input_tokens"] for c in calls]
    outputs = [c["output_tokens"] for c in calls]
    return {case: CTC.task_cost(prices, model, prompts, outputs, case) for case in CTC.CASES}


def run_task(runtime, task, env, seed, emit, log, spend) -> dict:
    t0 = time.monotonic()
    runtime.drain()
    row = X.run_long_task(runtime, runtime.model_id, None, None, task, env, log)
    calls = runtime.drain()
    cost = ACB.sum_calls(calls)
    ub = sum(c["usd_ledger_upper_bound"] if c.get("usd_ledger_upper_bound") is not None else c["usd_ledger"]
             for c in calls)
    spend["usd_billed"] += cost["usd_ledger"]
    spend["usd_upper_bound"] += ub
    row.update({"record": "longtask_cloud_task", "harness": HARNESS, "model_id": runtime.model_id, "seed": seed,
                "arm_id": "cloud_all", "ts_utc": ag.utc_iso(), **cost, "usd_billed": cost["usd_ledger"],
                "usd_upper_bound": ub, "task_wall_s_total": time.monotonic() - t0,
                "priced_estimate_usd": priced_estimate(spend["prices"], runtime.model_id, calls),
                "cloud_calls": calls, "job_spend_billed_usd": spend["usd_billed"],
                "ledger_total_usd": runtime.client.running_total_usd})
    emit(row)
    log(f"  {runtime.model_id} {task.task_id}: correct={row['correct']} calls={row['n_calls']} billed "
        f"${row['usd_billed']:.4f} upper ${ub:.4f} realistic-priced ${row['priced_estimate_usd']['realistic']:.4f} "
        f"{row['task_wall_s']:.0f}s | job billed ${spend['usd_billed']:.4f}")
    return row


def summarize(rows) -> dict:
    out = {}
    for m in sorted({r["model_id"] for r in rows if r.get("record") == "longtask_cloud_task"}):
        tr = [r for r in rows if r.get("record") == "longtask_cloud_task" and r["model_id"] == m]
        out[m] = {"n_tasks": len(tr), "correct": sum(1 for r in tr if r["correct"]),
                  "usd_billed": sum(r["usd_billed"] for r in tr), "usd_upper_bound": sum(r["usd_upper_bound"] for r in tr),
                  "priced": {c: sum(r["priced_estimate_usd"][c] for r in tr) for c in CTC.CASES},
                  "input_tokens": sum(r["input_tokens"] for r in tr), "output_tokens": sum(r["output_tokens"] for r in tr),
                  "seconds": sum(r["task_wall_s"] for r in tr)}
    return out


def run(client, out=OUT, price_file=PRICE_FILE, task_ids=CLOUD_TASKS, seed=SEED, job_cap=JOB_CAP_USD,
        ledger_total_at_start=None, log=print, retry_wait_s=10.0) -> dict:
    table = CR.load_price_table(price_file)
    prices = json.loads(Path(price_file).read_text(encoding="utf-8"))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    fh = open(out, "w", encoding="utf-8", newline="\n")

    def emit(r):
        rows.append(r)
        fh.write(json.dumps(r, sort_keys=True, default=str) + "\n")
        fh.flush()
    spend = {"usd_billed": 0.0, "usd_upper_bound": 0.0, "prices": prices}
    decision, stopped, sonnet = "stop", None, {"run": False}
    try:
        tasks = L.build_tasks(seed, task_ids)
        warm, temp = ACB.warmup_probe(client, HAIKU)
        emit({"record": "run_start", "harness": HARNESS, "ts_utc": ag.utc_iso(), "mode": "executed",
              "script": "scripts/longtask_cloud.py", "script_git_blob": CR.git_blob(Path(__file__)),
              "x2_longtask_git_blob": CR.git_blob(REPO / "harness" / "x2_longtask.py"),
              "longtask_env_git_blob": CR.git_blob(REPO / "harness" / "longtask_env.py"),
              "price_file": "results/" + Path(price_file).name, "price_file_git_blob": CR.git_blob(Path(price_file)),
              "seed": seed, "tasks": list(task_ids), "task_label": L.LABEL, "models_planned": [HAIKU],
              "thinking": {HAIKU: ACB.THINKING_OFF.get(HAIKU), SONNET: ACB.THINKING_OFF.get(SONNET)},
              "temperature_sent": temp, "max_tokens": ag.MAX_TOKENS_PER_CALL, "prompt_caching": True,
              "job_cap_usd": job_cap, "ledger_total_at_start_usd": ledger_total_at_start,
              "spend_cap_usd": client.spend_cap_usd, "ledger": client.ledger_path.name})
        emit({**warm, "ts_utc": ag.utc_iso()})
        spend["usd_billed"] += warm["usd_ledger"]
        spend["usd_upper_bound"] += warm.get("usd_ledger_upper_bound") or warm["usd_ledger"]
        rt = ACB.CloudRuntime(client, HAIKU, CR.model_rates(table, HAIKU), temperature=temp, log=log,
                              retry_wait_s=retry_wait_s)
        rt.final_tools = L.tools_payload()
        # pilot: the first task's first call (excluded)
        task0, _env0 = tasks[0]
        msgs = [{"role": "system", "content": L.SYSTEM_PROMPT}, {"role": "user", "content": L.user_message(task0)}]
        rt.chat(HAIKU, msgs, None, L.tools_payload(), None)
        pc = rt.drain()
        ratio = pc[0]["input_tokens"] / ag.prompt_tokens_est(msgs, L.tools_payload()) if pc else float("nan")
        spend["usd_billed"] += sum(c["usd_ledger"] for c in pc)
        spend["usd_upper_bound"] += sum(c.get("usd_ledger_upper_bound") or c["usd_ledger"] for c in pc)
        proj = projection(tasks, CR.model_rates(table, HAIKU), ratio)
        decision = "run" if proj + spend["usd_billed"] <= job_cap else "stop"
        emit({"record": "projection", "ts_utc": ag.utc_iso(), "model_id": HAIKU, "pilot_calls": pc,
              "ratio_provider_over_estimate": ratio, "projected_upper_bound_usd": proj,
              "spend_so_far_billed_usd": spend["usd_billed"], "job_cap_usd": job_cap, "decision": decision})
        log(f"projection {HAIKU}: ${proj:.4f} upper bound (ratio {ratio:.3f}) vs cap ${job_cap:.2f} -> {decision}")
        if decision != "run":
            stopped = "projection over the job cap; no measured task run"
        else:
            hrows = [run_task(rt, t, e, seed, emit, log, spend) for t, e in tasks]
            hcalls = [c for r in hrows for c in r["cloud_calls"]]
            est = reprice(hcalls, CR.model_rates(table, SONNET)) * MARGIN
            fits = est < SONNET_MAX_USD and spend["usd_billed"] + est <= job_cap
            sonnet = {"run": fits, "estimate_usd": est, "basis": "haiku calls' measured token breakdown at Sonnet "
                      f"rates x {MARGIN}", "threshold_usd": SONNET_MAX_USD, "job_billed_so_far": spend["usd_billed"]}
            emit({"record": "sonnet_decision", "ts_utc": ag.utc_iso(), **sonnet})
            log(f"sonnet estimate ${est:.4f} -> {'run' if fits else 'skip'}")
            if fits:
                wsn, tsn = ACB.warmup_probe(client, SONNET)
                emit({**wsn, "ts_utc": ag.utc_iso()})
                spend["usd_billed"] += wsn["usd_ledger"]
                rs = ACB.CloudRuntime(client, SONNET, CR.model_rates(table, SONNET), temperature=tsn, log=log,
                                      retry_wait_s=retry_wait_s)
                rs.final_tools = L.tools_payload()
                for t, e in tasks:
                    run_task(rs, t, e, seed, emit, log, spend)
    except cc.SpendCapExceeded as e:
        stopped = f"SpendCapExceeded: {e}"
        log(stopped)
    finally:
        summary = summarize(rows)
        emit({"record": "run_end", "harness": HARNESS, "ts_utc": ag.utc_iso(), "decision": decision,
              "stopped": stopped, "sonnet": sonnet, "job_spend_billed_usd": spend["usd_billed"],
              "job_spend_upper_bound_usd": spend["usd_upper_bound"], "ledger_total_usd": client.running_total_usd,
              "summary": summary})
        fh.close()
    return {"rows": rows, "decision": decision, "stopped": stopped, "summary": summary}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--live", action="store_true", help="execute on the cloud (needs ANTHROPIC_API_KEY)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--ledger", type=Path, default=cc.DEFAULT_LEDGER_PATH)
    ap.add_argument("--job-cap", type=float, default=JOB_CAP_USD)
    a = ap.parse_args(argv)
    if not a.live:
        print("refused: this job only runs live (--live); nothing was sent", file=sys.stderr)
        return 2
    if not os.environ.get(ACB.KEY_ENV):
        print(f"refused: --live needs {ACB.KEY_ENV} set in the environment; nothing was sent", file=sys.stderr)
        return 2
    if a.job_cap > JOB_CAP_USD:
        print(f"refused: --job-cap over the pre-registered USD {JOB_CAP_USD:.2f}", file=sys.stderr)
        return 2
    total0 = cc.CloudClient(mode="stub", ledger_path=a.ledger)._read_running_total()
    client = cc.CloudClient(mode="real", spend_cap_usd=total0 + a.job_cap, ledger_path=a.ledger)
    print(f"ledger total at start billed ${total0:.4f} (upper bound ${client.running_total_usd_upper_bound:.4f}); "
          f"job cap ${a.job_cap:.2f}; {client.spend_summary()}")
    try:   # console check before the first call (pilot included); the job can spend at most its job cap
        gate = console_gate.require_for_client(client, a.job_cap, None)
    except console_gate.ConsoleCheckRefused as e:
        print(str(e), file=sys.stderr)
        return 2
    print(console_gate.format_report(gate))
    res = run(client, a.out, job_cap=a.job_cap, ledger_total_at_start=total0)
    return 0 if res["decision"] == "run" and res["stopped"] is None else 4


if __name__ == "__main__":
    raise SystemExit(main())
