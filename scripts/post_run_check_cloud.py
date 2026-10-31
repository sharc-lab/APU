"""Live post-run check, controller cloud phase (docs/EXPERIMENTS.md "AMENDMENT (2026-10-31)"). Reads the local phase's
synced file (harness/x2_r2_evalset.py --plan post_run_check_live_v1 on evo-x2: B3 v2 qwen3:14b, num_ctx 8192, seed
20260905), runs the post-run check on every task turn, and escalates the turns it FAILs.

  L1: detector v2 recorded on the turn (cut_v2_flagged; a turn with a call missing its log slice counts as FAIL, since
      L1 could not be read), plus any call error (src/dse/post_run_check.l1_errors_record).
  L2: harness/post_run_check_live.turn_l2 (render lacks the rules, the request or the tool-call example; or the render
      is not citable after a non-matching prompt-token check).
  L3: src/dse/l3_invariants.py with CONFIG_TUNED (never the grader).
  FAIL -> rollback to the pre-task snapshot (scripts/cloud_redo.rollback_env, evalset_env.SnapshotLog) -> the task
  redone on claude-sonnet-5-5 from that state with the full client-side history (cloud_redo.live_turn), through the
  per-task cap (src/cloud/task_budget.py, USD 0.25) and the client's job cap (spend cap = ledger total + job cap,
  USD 2.00 at most). A per-task cap hit is a clean budget_exhausted row; a job-cap refusal stops the job, and every
  escalated task not yet run gets a not_run_job_cap row. The escalated tasks' grades come from cloud_redo.grade.

Default (no --live): no client, no cloud call. Writes the check decisions and the projection only.
--live: needs ANTHROPIC_API_KEY in this process, --ledger (the main checkout's results/cloud_ledger.jsonl), the
Console check (src/cloud/console_gate.pre_job, estimate = the job cap), and never more than USD 2.00 job cap here.

Writes results/x2_post_run_check_live_v1.jsonl (row post-run-check-live-8192).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "harness", REPO / "src", REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import cloud_redo as CR  # noqa: E402
import post_run_check_live as PRCL  # noqa: E402
from cloud import client as cc  # noqa: E402
from cloud import console_gate  # noqa: E402
from cloud import task_budget as TB  # noqa: E402
from src.dse import l3_invariants as L3  # noqa: E402
from src.dse import post_run_check as PRC  # noqa: E402

SOURCE = REPO / "results" / "x2_post_run_check_local_v1.jsonl"
OUT = REPO / "results" / "x2_post_run_check_live_v1.jsonl"
JOB_ID = "x2_post_run_check_live_v1"
JOB_CAP_MAX_USD = 2.00
TASK_CAP_USD = 0.25
MAX_TOKENS = 2048
TIER = "ollama_ctx_8192"


def local_rows(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    att = {s["attempt_id"] for s in CR.ag.completed_sessions(rows) if s.get("mode") == "real"}
    turns = [r for r in rows if r.get("record") == "r2a_turn" and r.get("attempt_id") in att
             and r.get("item_kind") == "task" and r.get("end_state_ok") is not None]
    return (sorted(turns, key=lambda r: (r["seed"], r["turn_idx"])),
            [r for r in rows if r.get("record") == "prc_l2_call"],
            [r for r in rows if r.get("record") == "prc_l2_prompt_check"])


def decide_turn(r: dict, es, l2_calls, checks, config=None) -> dict:
    """The check on one recorded task turn (pure apart from rebuilding nothing: es is the seed's session)."""
    config = L3.CONFIG_TUNED if config is None else config
    l1 = {"layer": "L1", "fail": bool(r.get("cut_v2_flagged")) or bool(r.get("cut_v2_no_log_calls")),
          "evidence": r.get("cut_v2_evidence"), "no_log_calls": r.get("cut_v2_no_log_calls")}
    l1e = PRC.l1_errors_record(r)
    l2 = PRCL.turn_l2(l2_calls, checks, r["seed"], r["turn_idx"])
    user = {t.idx: t.user_text for t in es.turns}[r["turn_idx"]]
    l3 = PRC.l3_record(r, es.system_prompt, user, CR.EV.tools_payload(es.design), config)
    d = PRC.decide([l1, l1e, l2, l3])
    d["l3_failed"], d["l2_reasons"] = l3["failed"], l2["reasons"]
    return d


def run(source: Path = SOURCE, out: Path = OUT, live: bool = False, client=None, job_cap: float = JOB_CAP_MAX_USD,
        ledger: Path | None = None, approval: str | None = None, emit_print=print, gate_fn=None) -> list[dict]:
    if job_cap > JOB_CAP_MAX_USD + 1e-9:
        raise SystemExit(f"refused: job cap ${job_cap:.2f} is above ${JOB_CAP_MAX_USD:.2f}; nothing was sent")
    rows_in = CR.load_rows(source)
    turns, l2_calls, checks = local_rows(rows_in)
    if not any(r.get("record") == "run_end" for r in rows_in):
        raise SystemExit(f"refused: {source.name} has no run_end (local phase unfinished); nothing was sent")
    cache, out_rows = {}, []
    table = CR.load_price_table()
    model = CR.escalation_model(table)
    rates = CR.model_rates(table, model)
    decided = []
    for r in turns:
        es = CR.build_session_for(r, cache)
        d = decide_turn(r, es, l2_calls, checks)
        decided.append((r, es, d))
    esc = [(r, es, d) for r, es, d in decided if d["decision"] == "escalate"]
    proj_upper = sum(CR.price_turn(CR.local_call_tokens(r), rates)["usd_all_uncached"] for r, _, _ in esc)
    start = {"record": "run_start", "harness": JOB_ID, "ts_utc": CR.utc_iso(), "live": live, "source": source.name,
             "source_git_blob": CR.git_blob(source), "script_git_blob": CR.git_blob(Path(__file__)),
             "model_id": model, "task_cap_usd": TASK_CAP_USD, "job_cap_usd": job_cap, "max_tokens": MAX_TOKENS,
             "l3_config": L3.CONFIG_TUNED, "n_tasks": len(turns), "n_escalated": len(esc),
             "projection_local_tokens_upper_usd": proj_upper}
    if live:
        if client is None or getattr(client, "stub_mode", True):
            raise SystemExit("refused: --live needs a real-mode client; nothing was sent")
        gate = (gate_fn or console_gate.pre_job)(client, job_id=JOB_ID, estimate_usd=job_cap, estimate_basis="job cap",
                                    projection_billed_usd=min(proj_upper, job_cap),
                                    projection_upper_usd=proj_upper, approval=approval)
        start["console_check"] = gate
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        def emit(row):
            out_rows.append(row)
            fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            fh.flush()

        emit(start)
        stopped = None
        lane = TB.BudgetedClient(client, TASK_CAP_USD, ledger_path=ledger) if live else None
        src_rows = rows_in
        for r, es, d in decided:
            row = {"record": "prc_live_task", "tier": TIER, "seed": r["seed"], "turn_idx": r["turn_idx"],
                   "task_id": r["task_id"], "decision": d["decision"], "fired": d["fired"],
                   "l3_failed": d["l3_failed"], "l2_reasons": d["l2_reasons"],
                   "local_ok": r["end_state_ok"] is True}   # grader outcome, read after the check, scoring only
            if d["decision"] == "escalate":
                if not live:
                    row["cloud"] = {"outcome": "not_run_dry"}
                elif stopped:
                    row["cloud"] = {"outcome": "not_run_job_cap"}
                else:
                    prior = [x for x in CR.session_rows(src_rows, r["attempt_id"]) if x["turn_idx"] < r["turn_idx"]]
                    built = CR.rebuild(es, prior, r["turn_idx"])
                    built["env_before"] = CR.rollback_env(es, prior, r)["env"]
                    lane.start_task(f"{TIER}:{r['seed']}:{r['turn_idx']}:{r['task_id']}")
                    try:
                        res = CR.live_turn(lane, model, es, built, rates, MAX_TOKENS, CR.THINKING_OFF.get(model))
                        row["cloud"] = {"outcome": "cloud", **res, **lane.task_state()}
                    except TB.TaskBudgetExhausted as e:
                        row["cloud"] = {"outcome": TB.OUTCOME_EXHAUSTED, "cloud_end_state_ok": False,
                                        "task_budget": e.record, **lane.task_state()}
                    except cc.SpendCapExceeded as e:
                        stopped = f"job cap: {e}"
                        row["cloud"] = {"outcome": "not_run_job_cap", "reason": str(e)[:300]}
            emit(row)
        tasks = [x for x in out_rows if x["record"] == "prc_live_task"]
        emit({"record": "run_end", "ts_utc": CR.utc_iso(), "stopped": stopped, "summary": summarize(tasks),
              "ledger_total_usd": getattr(client, "running_total_usd", None) if live else None})
    emit_print(json.dumps(out_rows[-1]["summary"]))
    return out_rows


def summarize(tasks: list[dict]) -> dict:
    wrong = [t for t in tasks if not t["local_ok"]]
    right = [t for t in tasks if t["local_ok"]]
    esc = lambda t: t["decision"] == "escalate"  # noqa: E731

    def ok(t):
        if not esc(t):
            return t["local_ok"]
        return bool((t.get("cloud") or {}).get("cloud_end_state_ok"))
    usd = sum(float((t.get("cloud") or {}).get("usd") or 0.0) for t in tasks)
    return {"n": len(tasks), "wrong": len(wrong), "right": len(right), "caught": sum(map(esc, wrong)),
            "false_esc": sum(map(esc, right)), "kept_local": sum(1 for t in tasks if not esc(t)),
            "final_correct": sum(1 for t in tasks if ok(t)), "cloud_usd_billed": usd,
            "budget_exhausted": sum(1 for t in tasks if (t.get("cloud") or {}).get("outcome") == TB.OUTCOME_EXHAUSTED),
            "not_run": sum(1 for t in tasks if str((t.get("cloud") or {}).get("outcome", "")).startswith("not_run"))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--live", action="store_true", help="escalate on the cloud (needs ANTHROPIC_API_KEY)")
    ap.add_argument("--ledger", type=Path, default=cc.DEFAULT_LEDGER_PATH,
                    help="the shared ledger (pass the main checkout's results/cloud_ledger.jsonl)")
    ap.add_argument("--job-cap", type=float, default=JOB_CAP_MAX_USD, help="USD billed, at most 2.00")
    ap.add_argument(console_gate.APPROVAL_FLAG, dest="operator_approval", default=None)
    a = ap.parse_args(argv)
    client = None
    if a.live:
        if not os.environ.get(CR.KEY_ENV):
            print(f"refused: --live needs {CR.KEY_ENV} set in the environment; nothing was sent", file=sys.stderr)
            return 2
        if Path(a.out).exists():
            print(f"refused: {Path(a.out).name} exists (would be overwritten); nothing was sent", file=sys.stderr)
            return 2
        total0 = cc.CloudClient(mode="stub", ledger_path=a.ledger)._read_running_total()
        client = cc.CloudClient(mode="real", spend_cap_usd=total0 + min(a.job_cap, JOB_CAP_MAX_USD),
                                ledger_path=a.ledger)
    rows = run(a.source, a.out, a.live, client, a.job_cap, a.ledger, a.operator_approval)
    return 0 if rows[-1].get("stopped") is None else 4


if __name__ == "__main__":
    raise SystemExit(main())
