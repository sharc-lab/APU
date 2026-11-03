"""XPS demo kit, cloud step (controller only, operator only): the cloud parts of arms b and c, and arm d, from the
local step's file (harness/xps_demo_kit.py, run on the XPS at USD 0). Then the comparison table. docs/XPS_DEMO_KIT.md
has the commands.

Subcommands:
  estimate   No client, no key, no network. The cost basis from committed result files (estimate_basis(): Sonnet
             all-cloud USD per task at the rolling cache breakpoint, and the highest tier-mean full-history redo USD at
             the rolling breakpoint, both through analysis/rolling_cache_reprice.py) and, given --local files, the
             cloud task count of each arm and how the arm splits into jobs that each stay under the USD 2 cap.
  run        --arm b|c|d [--part N] --live. Refused unless: --live; ANTHROPIC_API_KEY in this process; the local
             file(s) complete (run_start and run_end, run_end not stopped); --operator-approval "<text>" is given
             (always required here, even under USD 2: console_gate.validate_approval checks it, the operator confirms it
             at the console, and it is written to the ledger before the first call); the job estimate is at most the
             job cap and the job cap at most JOB_CAP_MAX_USD = 2.00; the out file does not exist. Then
             src/cloud/console_gate.pre_job (per-job limit, approval, the Console check; fail closed) and a real-mode
             src/cloud/client.CloudClient with spend_cap_usd = ledger total + job cap. Nothing here raises or skips a
             cap.
               b, c: every cloud task of the part, in (seed, turn) order: the session rebuilt from the seed, the
                 transcript before the task rebuilt from the LOCAL run's own prior turns (scripts/cloud_redo.rebuild),
                 the environment rolled back to the pre-task snapshot (cloud_redo.rollback_env; for a deferred task
                 there is nothing to undo), then the task on the escalation model (claude-sonnet-5-5, from the price
                 file) with cloud_redo.live_turn through the per-task cap (src/cloud/task_budget.BudgetedClient, USD
                 0.25), max_tokens 2048, thinking off, rolling cache breakpoint; graded by the same checker. Arm c's
                 escalations are the post-run check's decisions, recomputed here (xps_demo_kit.post_run_check) and
                 compared with the decision recorded on the XPS (mismatches recorded, the recomputed one is used).
               d: the 30-task session per seed on the cloud, scripts/all_cloud_b3.run_cloud_session unchanged
                 (CloudRuntime, tool loop, max_tokens 384, rolling breakpoint). Per-task cap not applied (one session
                 per call sequence); the job cap applies.
  summarize  The comparison per arm from the local and cloud files: correct/90, cloud USD per job (billed, cache
             priced; local calls USD 0) and latency per job (local turn seconds measured on the XPS; for a task sent
             to the cloud, the local seconds spent before it was sent plus the cloud request seconds measured from the
             controller, not from the XPS). A cloud task that was not run counts as not correct and the arm is marked
             incomplete.

This file starts no process of its own (console_gate reads the git head through harness/proc_util.py).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "harness", REPO / "src", REPO / "scripts", REPO / "analysis"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import cloud_redo as CR  # noqa: E402
import xps_demo_kit as K  # noqa: E402
from cloud import client as cc  # noqa: E402
from cloud import console_gate  # noqa: E402
from cloud import task_budget as TB  # noqa: E402

JOB_CAP_MAX_USD = 2.00
TASK_CAP_USD = 0.25
MAX_TOKENS = 2048
CLOUD_ARMS = ("b", "c", "d")
ARM_NAMES = {**K.ARMS, "d": K.CLOUD_ARM_D}
RECORD = "xps_cloud_task"
N_JOBS = 90


def expand(paths) -> list[str]:
    """Command-line paths with wildcards expanded here (PowerShell passes them to a native command unexpanded)."""
    import glob  # noqa: PLC0415
    out = []
    for p in paths or []:
        hits = sorted(glob.glob(str(p))) if any(c in str(p) for c in "*?[") else [str(p)]
        out.extend(h for h in hits if h not in out)
    return out


def load_rows(paths) -> list[dict]:
    rows = []
    for p in paths:
        rows.extend(CR.load_rows(Path(p)))
    return rows


def local_complete(paths) -> list[str]:
    """Problems that make the local files unusable (empty list: complete)."""
    out = []
    for p in paths:
        rows = CR.load_rows(Path(p))
        if not any(r.get("record") == "run_start" for r in rows):
            out.append(f"{Path(p).name}: no run_start")
        ends = [r for r in rows if r.get("record") == "run_end"]
        if not ends:
            out.append(f"{Path(p).name}: no run_end (local step unfinished)")
        elif ends[-1].get("stopped"):
            out.append(f"{Path(p).name}: run_end stopped ({ends[-1]['stopped']})")
    return out


# ── cost basis from files ─────────────────────────────────────────────────────────────────────────────────────

def estimate_basis(repo: Path = REPO) -> dict:
    """USD per task from committed result files, at the rolling cache breakpoint the live client uses: the escalation
    model's all-cloud B3 v2 mean (all_cloud_b3_v1 repriced) and the highest tier-mean full-history redo (the cloud redo
    and own-rollback files repriced; the figure hybrid_full_pipeline imputes for an escalation)."""
    import rolling_cache_reprice as rr  # noqa: PLC0415
    table = CR.load_price_table()
    model = CR.escalation_model(table)
    with contextlib.redirect_stdout(io.StringIO()):
        ac = rr.all_cloud_rolling(repo)
        rd = rr.redo_rolling(repo)
    allc = rr.b3_all_cloud_summary(ac)[model]
    tiers = {}
    for (tier, _s, _t), v in rd["redo"].items():
        tiers.setdefault(tier, []).append(v["usd_new"])
    tier_mean = {t: statistics.fmean(v) for t, v in sorted(tiers.items())}
    worst = max(tier_mean, key=tier_mean.get)
    return {"model": model, "all_cloud_usd_task": allc["usd_new"], "all_cloud_usd_task_recorded": allc["usd_recorded"],
            "all_cloud_n": allc["n"], "all_cloud_correct": allc["correct"], "redo_usd_task": tier_mean[worst],
            "redo_tier": worst, "redo_tier_means": tier_mean, "redo_model": rd["model"],
            "files": [rr.ALL_CLOUD_FILE, "results/cloud_redo_full_context_v1*.jsonl", rr.ROLLBACK_FILE,
                      rr.PRICE_FILE]}


def split_parts(tasks: list, per_task_usd: float, cap: float = JOB_CAP_MAX_USD) -> list[list]:
    """Consecutive parts of at most floor(cap / per_task_usd) tasks, so each part's estimate stays under the cap."""
    if not tasks:
        return []
    size = max(1, math.floor(cap / per_task_usd)) if per_task_usd > 0 else len(tasks)
    return [tasks[i:i + size] for i in range(0, len(tasks), size)]


# ── which tasks each arm sends to the cloud (pure) ────────────────────────────────────────────────────────────

def cloud_tasks(local_rows: list[dict], arm: str, recheck=None) -> list[dict]:
    """[{arm, seed, turn_idx, task_id, reason, attempt_id, row}] in (seed, turn) order. b: every deferred task. c: every
    routed task plus every local task the post-run check escalates (recheck(row) -> decision dict; default: the
    decision recorded on the XPS)."""
    if arm == "d":
        return [{"arm": "d", "seed": s, "reason": "all_cloud"} for s in K.SEEDS]
    out = []
    for r in local_rows:
        if r.get("arm") != arm:
            continue
        if r.get("record") == K.DEFER_RECORD and r.get("item_kind") == "task":
            out.append({"arm": arm, "seed": r["seed"], "turn_idx": r["turn_idx"], "task_id": r["task_id"],
                        "reason": r["reason"], "attempt_id": r["attempt_id"], "row": r})
        elif arm == "c" and r.get("record") == "r2a_turn" and r.get("item_kind") == "task":
            recorded = (r.get("prc_check") or {}).get("decision")
            d = recheck(r) if recheck else r.get("prc_check") or {}
            if d.get("decision") == "escalate":
                out.append({"arm": arm, "seed": r["seed"], "turn_idx": r["turn_idx"], "task_id": r["task_id"],
                            "reason": "post_run_check", "fired": d.get("fired"), "attempt_id": r["attempt_id"],
                            "recorded_decision": recorded, "decision_matches_xps": recorded == d.get("decision"),
                            "row": r})
    return sorted(out, key=lambda t: (t["seed"], t["turn_idx"]))


def recheck_fn(local_rows: list[dict]):
    """Recompute arm c's post-run check on the controller from the recorded rows."""
    cache = {}
    runtime = next((r["runtime_config"]["runtime"] for r in local_rows if r.get("record") == "run_start"), "ollama")

    def f(row):
        es = CR.build_session_for(row, cache)
        return K.post_run_check(row, es, runtime, row.get("l1_events"))
    return f


def unit_usd(arm: str, basis: dict) -> float:
    """Estimate per cloud unit: arm d one 30-task session (30 x the all-cloud USD per task), else one redone task."""
    return basis["all_cloud_usd_task"] * N_JOBS / len(K.SEEDS) if arm == "d" else basis["redo_usd_task"]


def plan(local_rows: list[dict], basis: dict, recheck=None) -> dict:
    out = {}
    for arm in CLOUD_ARMS:
        tasks = cloud_tasks(local_rows, arm, recheck if arm == "c" else None)
        per = unit_usd(arm, basis)
        parts = split_parts(tasks, per)
        out[arm] = {"n_cloud_units": len(tasks), "unit": "session of 30 tasks" if arm == "d" else "task",
                    "usd_per_unit_estimate": per, "estimate_usd": len(tasks) * per,
                    "parts": [{"part": i + 1, "n": len(p), "estimate_usd": len(p) * per} for i, p in enumerate(parts)],
                    "by_reason": _count([t["reason"] for t in tasks])}
    return out


def _count(xs):
    d = {}
    for x in xs:
        d[x] = d.get(x, 0) + 1
    return d


# ── the live job ─────────────────────────────────────────────────────────────────────────────────────────────

def job_id(arm: str, part: int) -> str:
    return f"xps_demo_cloud_{arm}" + (f"_part{part}" if part > 1 else "")


def run(arm: str, part: int, local_paths, out: Path, client, approval: str | None, ledger: Path | None = None,
        job_cap: float = JOB_CAP_MAX_USD, gate_fn=None, basis: dict | None = None, log=print,
        live_turn=None, run_session=None) -> list[dict]:
    if arm not in CLOUD_ARMS:
        raise SystemExit(f"refused: --arm must be one of {CLOUD_ARMS}")
    if job_cap > JOB_CAP_MAX_USD + 1e-9:
        raise SystemExit(f"refused: job cap ${job_cap:.2f} is above ${JOB_CAP_MAX_USD:.2f}; nothing was sent")
    if not (approval and approval.strip()):
        raise SystemExit("refused: the cloud step needs the operator's approval on the command line "
                         f"({console_gate.approval_syntax(job_id(arm, part), job_cap)}); nothing was sent")
    probs = local_complete(local_paths)
    if probs:
        raise SystemExit("refused: " + "; ".join(probs) + "; nothing was sent")
    if client is None or getattr(client, "stub_mode", True):
        raise SystemExit("refused: --live needs a real-mode client; nothing was sent")
    if Path(out).exists():
        raise SystemExit(f"refused: {Path(out).name} exists (never overwritten); nothing was sent")
    local = load_rows(local_paths)
    basis = basis or estimate_basis()
    recheck = recheck_fn(local) if arm == "c" else None
    tasks = cloud_tasks(local, arm, recheck)
    per = unit_usd(arm, basis)
    parts = split_parts(tasks, per, job_cap)
    if part < 1 or part > max(1, len(parts)):
        raise SystemExit(f"refused: arm {arm} has {len(parts)} part(s) at job cap ${job_cap:.2f}; nothing was sent")
    mine = parts[part - 1] if parts else []
    estimate = len(mine) * per
    if estimate > job_cap + 1e-9:
        raise SystemExit(f"refused: estimate ${estimate:.4f} is above the job cap ${job_cap:.2f}; nothing was sent")
    jid = job_id(arm, part)
    gate = (gate_fn or console_gate.pre_job)(
        client, job_id=jid, estimate_usd=estimate,
        estimate_basis=(f"{len(mine)} x USD {per:.5f} ("
                        + ("Sonnet all-cloud B3 v2 USD per task x 30, rolling cache" if arm == "d" else
                           f"highest tier-mean full-history redo, {basis['redo_tier']}, rolling cache") + ")"),
        projection_billed_usd=estimate, approval=approval)
    table = CR.load_price_table()
    model = CR.escalation_model(table)
    rates = CR.model_rates(table, model)
    rows: list[dict] = []
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        def emit(row):
            rows.append(row)
            fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            fh.flush()

        emit({"record": "run_start", "harness": "xps_demo_cloud", "ts_utc": CR.utc_iso(), "mode": "executed",
              "arm": arm, "arm_name": ARM_NAMES[arm], "part": part, "n_parts": len(parts), "job_id": jid,
              "local_files": [Path(p).name for p in local_paths],
              "local_git_blobs": {Path(p).name: CR.git_blob(Path(p)) for p in local_paths},
              "script_git_blob": CR.git_blob(Path(__file__)), "kit_git_blob": CR.git_blob(Path(K.__file__)),
              "model_id": model, "max_tokens": MAX_TOKENS if arm != "d" else None,
              "task_cap_usd": TASK_CAP_USD if arm != "d" else None, "job_cap_usd": job_cap,
              "estimate_usd": estimate, "estimate_basis": basis, "n_units": len(mine),
              "cache_breakpoints": CR.CACHE_BREAKPOINTS, "console_check": gate})
        stopped = None
        if arm == "d":
            import all_cloud_b3 as AC  # noqa: PLC0415
            rt = AC.CloudRuntime(client, model, rates, log=log)
            spend = {"usd": 0.0, "usd_ledger": 0.0}
            for t in mine:
                try:
                    (run_session or AC.run_cloud_session)(rt, t["seed"], emit, log, spend=spend)
                except cc.SpendCapExceeded as e:
                    stopped = f"job cap: {e}"
                    break
        else:
            lane = TB.BudgetedClient(client, TASK_CAP_USD, ledger_path=ledger)
            cache = {}
            for t in mine:
                row = {k: v for k, v in t.items() if k != "row"}
                row.update({"record": RECORD, "ts_utc": CR.utc_iso()})
                if stopped:
                    row["cloud"] = {"outcome": "not_run_job_cap"}
                    emit(row)
                    continue
                es = CR.build_session_for(t["row"], cache)
                prior = [x for x in CR.session_rows(local, t["attempt_id"]) if x["turn_idx"] < t["turn_idx"]]
                failed = t["row"] if t["row"].get("record") == "r2a_turn" else {}
                built = CR.rebuild(es, prior, t["turn_idx"])
                rb = CR.rollback_env(es, prior, failed)
                built["env_before"] = rb["env"]
                row["undone_calls"] = rb["undone_calls"]
                lane.start_task(f"{jid}:{t['seed']}:{t['turn_idx']}:{t['task_id']}")
                try:
                    res = (live_turn or CR.live_turn)(lane, model, es, built, rates, MAX_TOKENS,
                                                      CR.THINKING_OFF.get(model))
                    row["cloud"] = {"outcome": "cloud", **res, **lane.task_state()}
                except TB.TaskBudgetExhausted as e:
                    row["cloud"] = {"outcome": TB.OUTCOME_EXHAUSTED, "cloud_end_state_ok": False,
                                    "task_budget": e.record, **lane.task_state()}
                except cc.SpendCapExceeded as e:
                    stopped = f"job cap: {e}"
                    row["cloud"] = {"outcome": "not_run_job_cap", "reason": str(e)[:300]}
                emit(row)
                log(f"  {jid} seed={t['seed']} turn {t['turn_idx']} {t['task_id']} ({t['reason']}): "
                    f"{(row['cloud'] or {}).get('outcome')} ok={(row['cloud'] or {}).get('cloud_end_state_ok')}")
        emit({"record": "run_end", "ts_utc": CR.utc_iso(), "stopped": stopped,
              "ledger_total_usd": getattr(client, "running_total_usd", None)})
    return rows


# ── the comparison (pure) ─────────────────────────────────────────────────────────────────────────────────────

def _q(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    return s[min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))]


def _stats(vals):
    vals = [v for v in vals if v is not None]
    return {"n": len(vals), "mean": statistics.fmean(vals) if vals else None,
            "median": statistics.median(vals) if vals else None, "p90": _q(vals, 0.9)}


def summarize(local_rows: list[dict], cloud_rows: list[dict], n_jobs: int = N_JOBS) -> dict:
    """Per arm: correct/n_jobs, cloud USD per job, latency per job. Jobs are (seed, task_id) of the B3 v2 sessions."""
    cloud_task = {(r["arm"], r["seed"], r["turn_idx"]): r for r in cloud_rows if r.get("record") == RECORD}
    out = {}
    for arm in ("a", "b", "c"):
        jobs = {}
        for r in local_rows:
            if r.get("arm") != arm or r.get("item_kind") != "task":
                continue
            key = (r["seed"], r["task_id"])
            if r.get("record") == "r2a_turn":
                # escalated: the XPS decision or a cloud row (the cloud step uses the decision recomputed there); a
                # task escalated by one side only and missing its cloud row counts as not run
                if arm == "c" and ((r.get("prc_check") or {}).get("decision") == "escalate"
                                   or (arm, r["seed"], r["turn_idx"]) in cloud_task):
                    c = (cloud_task.get((arm, r["seed"], r["turn_idx"])) or {}).get("cloud")
                    jobs[key] = _cloud_job(c, r.get("local_seconds_with_check") or r["turn_wall_s"], "escalated")
                else:
                    sec = r.get("local_seconds_with_check") or r["turn_wall_s"]
                    jobs[key] = {"ok": r.get("end_state_ok") is True, "usd": 0.0, "seconds": sec, "where": "local",
                                 "ran": True}
            elif r.get("record") == K.DEFER_RECORD:
                c = (cloud_task.get((arm, r["seed"], r["turn_idx"])) or {}).get("cloud")
                jobs[key] = _cloud_job(c, r.get("local_seconds") or 0.0, r["reason"])
        out[ARM_NAMES[arm]] = _arm_summary(jobs, n_jobs)
    d_turns = [r for r in cloud_rows if r.get("record") == "r2a_turn" and r.get("arm_id") == "cloud_all"
               and r.get("item_kind") == "task"]
    djobs = {(r["seed"], r["task_id"]): {"ok": r.get("end_state_ok") is True, "usd": float(r.get("usd") or 0.0),
                                        "seconds": r.get("turn_wall_s"), "where": "cloud", "ran": True}
             for r in d_turns}
    out[ARM_NAMES["d"]] = _arm_summary(djobs, n_jobs)
    return out


def _cloud_job(c: dict | None, local_s: float, where: str) -> dict:
    if not c or str(c.get("outcome", "")).startswith("not_run"):
        return {"ok": False, "usd": 0.0, "seconds": None, "where": where, "ran": False}
    return {"ok": bool(c.get("cloud_end_state_ok")), "usd": float(c.get("usd") or 0.0),
            "seconds": local_s + float(c.get("seconds") or 0.0), "where": where, "ran": True}


def _arm_summary(jobs: dict, n_jobs: int) -> dict:
    ran = [j for j in jobs.values() if j["ran"]]
    return {"jobs_recorded": len(jobs), "jobs_planned": n_jobs, "correct": sum(j["ok"] for j in jobs.values()),
            "correct_of": n_jobs, "complete": len(jobs) == n_jobs and len(ran) == n_jobs,
            "not_run": len(jobs) - len(ran), "missing": n_jobs - len(jobs),
            "kept_local": sum(1 for j in jobs.values() if j["where"] == "local"),
            "to_cloud": _count([j["where"] for j in jobs.values() if j["where"] != "local"]),
            "usd_per_job_mean": sum(j["usd"] for j in jobs.values()) / n_jobs,
            "usd_total": sum(j["usd"] for j in jobs.values()),
            "latency_s": _stats([j["seconds"] for j in ran])}


def format_table(s: dict) -> str:
    lines = ["| arm | correct | cloud USD per job (mean) | latency per job s (median / p90) | kept local | complete |",
             "|---|---|---|---|---|---|"]
    for arm, v in s.items():
        lat = v["latency_s"]
        lines.append(f"| {arm} | {v['correct']}/{v['correct_of']} | {v['usd_per_job_mean']:.5f} | "
                     f"{(lat['median'] or 0):.1f} / {(lat['p90'] or 0):.1f} | {v['kept_local']} | {v['complete']} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("estimate")
    e.add_argument("--local", nargs="*", default=[])
    r = sub.add_parser("run")
    r.add_argument("--arm", choices=CLOUD_ARMS, required=True)
    r.add_argument("--part", type=int, default=1)
    r.add_argument("--local", nargs="+", required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--live", action="store_true")
    r.add_argument("--ledger", type=Path, default=cc.DEFAULT_LEDGER_PATH,
                   help="the shared ledger (pass the main checkout's results/cloud_ledger.jsonl)")
    r.add_argument("--job-cap", type=float, default=JOB_CAP_MAX_USD, help="USD billed, at most 2.00")
    r.add_argument(console_gate.APPROVAL_FLAG, dest="operator_approval", default=None)
    s = sub.add_parser("summarize")
    s.add_argument("--local", nargs="+", required=True)
    s.add_argument("--cloud", nargs="*", default=[])
    s.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    a.local = expand(a.local)
    if hasattr(a, "cloud"):
        a.cloud = expand(a.cloud)
    if a.cmd == "estimate":
        basis = estimate_basis()
        rep = {"basis": basis}
        if a.local:
            rows = load_rows(a.local)
            rep["plan"] = plan(rows, basis, recheck_fn(rows))
        print(json.dumps(rep, indent=1, sort_keys=True, default=str))
        return 0
    if a.cmd == "summarize":
        sm = summarize(load_rows(a.local), load_rows(a.cloud) if a.cloud else [])
        print(format_table(sm))
        if a.out:
            a.out.write_text(json.dumps(sm, indent=1, sort_keys=True), encoding="utf-8")
        return 0
    if not a.live:
        print("refused: the cloud step only runs live (--live); nothing was sent", file=sys.stderr)
        return 2
    if not os.environ.get(CR.KEY_ENV):
        print(f"refused: --live needs {CR.KEY_ENV} set in the environment; nothing was sent", file=sys.stderr)
        return 2
    cap = min(a.job_cap, JOB_CAP_MAX_USD)
    total0 = cc.CloudClient(mode="stub", ledger_path=a.ledger)._read_running_total()
    client = cc.CloudClient(mode="real", spend_cap_usd=total0 + cap, ledger_path=a.ledger)
    try:
        rows = run(a.arm, a.part, a.local, a.out, client, a.operator_approval, a.ledger, cap)
    except console_gate.ConsoleCheckRefused as ex:
        print(str(ex), file=sys.stderr)
        return 2
    return 0 if rows[-1].get("stopped") is None else 4


if __name__ == "__main__":
    raise SystemExit(main())
