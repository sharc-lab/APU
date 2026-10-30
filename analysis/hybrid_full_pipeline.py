"""hybrid-full-pipeline-v1: route at start (B) + local run + post-run check (A) + rollback + cloud under a per-task
budget (C), by replay of recorded runs (operator brief 2026-10-29; pre-registered in docs/EXPERIMENTS.md
"PRE-REGISTRATION (2026-10-29): hybrid-full-pipeline-v1").

A and C are built elsewhere and consumed only through the two interfaces below:
  PostRunCheck.check(task_record) -> {"escalate": bool, "reason": str}
      task_record: the local turn row with the grader's fields removed (strip_grader, GRADER_FIELDS).
      Provided by analysis/post_run_check.py::make_post_run_check(repo) once A lands.
  TaskBudget.allow(task_key, usd_estimate) -> {"allow": bool, "cap_usd": float}
      asked before every cloud use (route or escalation) with that use's USD; may keep state across tasks.
      Provided by analysis/task_budget.py::make_task_budget(repo) once C lands.
Until both modules exist the register row is PENDING ("awaiting post-run-check (A)" first). Stand-ins for A and C
live only in tests/test_hybrid_full_pipeline.py.

live_hours(): the E design (hybrid-live-amd on evo-x2, hybrid-live-intel on evo-t2s, 3 seeds each): ESTIMATED hours
from measured rows and cloud spend (billed and upper bound). Nothing is deployed or queued.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import statistics
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol, runtime_checkable

REPO = Path(__file__).resolve().parents[1]
SEEDS = (20260901, 20260902, 20260903)
CLOUD_MODEL = "claude-sonnet-5-5"
TARGET_KEPT = 50
TARGET_ACC = 0.97
K = 3
A_MODULE = "analysis/post_run_check.py"     # make_post_run_check(repo) -> PostRunCheck
C_MODULE = "analysis/task_budget.py"        # make_task_budget(repo) -> TaskBudget
PENDING_A = "awaiting post-run-check (A)"
PENDING_C = "awaiting per-task budget (C)"
B3_V2_FILE = "results/x2_evalset_tasks_real_v2.jsonl"
ROLLBACK_FILE = "results/own_rollback_redo_v1.jsonl"
# name -> (file, exact arm_id, full-history redo tier or None). Primary first.
CONFIGS = {
    "q4_0 28928": ("results/x2_own_config_probe_q4_28928.jsonl", "ollama_ctx_28928_call2_notools", None),
    "q8_0 15360": ("results/x2_own_config_probe_q8_15360.jsonl", "ollama_ctx_15360_call2_notools", None),
    "f16 8192": (B3_V2_FILE, "ollama_ctx_8192_call2_notools", "ollama_ctx_8192"),
    "f16 4096": (B3_V2_FILE, "ollama_ctx_4096_call2_notools", "ollama_ctx_4096"),
}
PRIMARY = "q4_0 28928"
DATA_FILES = tuple(sorted({v[0] for v in CONFIGS.values()}))

# The grader's per-task scoring fields: A must not read them (it would be reading the answer key).
GRADER_FIELDS = ("end_state_ok", "end_state_diff", "state_ok", "answer_ok", "task_tool_ok", "n_args_correct",
                 "tool_args_all", "recall_ok", "canary_sys_ok", "canary_hist_ok", "other_canaries_in_answer",
                 "lookup_result_used", "recall_key_looked_up", "asked_confirmation", "refusal_wording",
                 "confirmed_targets")
GRADER_PREFIXES = ("rule",)


def strip_grader(row: dict) -> dict:
    return {k: v for k, v in row.items() if k not in GRADER_FIELDS and not k.startswith(GRADER_PREFIXES)}


@runtime_checkable
class PostRunCheck(Protocol):
    """A: after the local run of a task, decide whether to roll back and escalate."""

    def check(self, task_record: dict) -> dict: ...


@runtime_checkable
class TaskBudget(Protocol):
    """C: allow or refuse one cloud use of a task at an estimated USD."""

    def allow(self, task_key, usd_estimate: float) -> dict: ...


def _load_factory(repo, rel: str, factory: str):
    p = Path(repo) / rel
    if not p.exists():
        return None
    spec = importlib.util.spec_from_file_location(Path(rel).stem + "_hfp", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, factory, None)
    return fn(Path(repo)) if fn else None


def load_post_run_check(repo=REPO):
    return _load_factory(repo, A_MODULE, "make_post_run_check")


def load_task_budget(repo=REPO):
    return _load_factory(repo, C_MODULE, "make_task_budget")


def pending_reason(repo=REPO):
    """The register row's PENDING reason, or None once both A and C provide their factories."""
    for rel, fn, why in ((A_MODULE, "make_post_run_check", PENDING_A), (C_MODULE, "make_task_budget", PENDING_C)):
        p = Path(repo) / rel
        if not p.exists() or f"def {fn}(" not in p.read_text(encoding="utf-8"):
            return why
    return None


# ── the pipeline (pure) ────────────────────────────────────────────────────────────────────────────

def run_pipeline(tasks, local_rows, first_turns, ceiling, cloud, esc, check, budget, predictor=None) -> dict:
    """Pure. tasks: [(seed, task_id)]; local_rows {(seed, task): local turn row (end_state_ok used only for
    scoring)}; first_turns {seed: [(turn_idx, prompt)] x K}; ceiling: the config's window; cloud {(seed, task):
    {ok, usd}} all-cloud rows; esc {(seed, task): {ok, usd, src}} rollback-then-cloud outcome; check: PostRunCheck;
    budget: TaskBudget; predictor: FitsLocallyPredictor (default SessionGrowthPredictor, k = 3)."""
    if predictor is None:
        from src.dse.fits_locally import SessionGrowthPredictor
        predictor = SessionGrowthPredictor(k=K)
    res = {"n": 0, "kept_local": 0, "correct": 0, "usd": 0.0, "routed": 0, "escalated": 0, "budget_refused_route": 0,
           "budget_refused_escalation": 0, "wrong_kept_local": 0, "imputed_escalations": 0, "per_task": {}}
    for k in tasks:
        row = local_rows[k]
        ok_local = row["end_state_ok"] is True
        res["n"] += 1
        ft = first_turns[k[0]]
        first_idx = {i for i, _p in ft}
        route = False
        if row["turn_idx"] not in first_idx and row["turn_idx"] > max(first_idx):
            route = not predictor.predict({"first_turns": ft, "target_turn": row["turn_idx"]},
                                          {"window": ceiling})["fits"]
        if route:
            if budget.allow(k, cloud[k]["usd"])["allow"]:
                res["routed"] += 1
                res["correct"] += cloud[k]["ok"]
                res["usd"] += cloud[k]["usd"]
                res["per_task"][k] = "routed"
                continue
            res["budget_refused_route"] += 1
        c = check.check(strip_grader(row))
        if c["escalate"]:
            e = esc[k]
            if budget.allow(k, e["usd"])["allow"]:
                res["escalated"] += 1
                res["imputed_escalations"] += e["src"] == "imputed"
                res["correct"] += e["ok"]
                res["usd"] += e["usd"]
                res["per_task"][k] = "escalated"
                continue
            res["budget_refused_escalation"] += 1
        res["kept_local"] += 1
        res["correct"] += ok_local
        res["wrong_kept_local"] += not ok_local
        res["per_task"][k] = "local"
    n = res["n"]
    res["accuracy"] = res["correct"] / n if n else None
    res["usd_task"] = res["usd"] / n if n else None
    res["meets_target"] = res["kept_local"] >= TARGET_KEPT and (res["accuracy"] or 0) >= TARGET_ACC
    return res


# ── replay inputs from recorded runs ───────────────────────────────────────────────────────────────

def _paths(repo):
    for sub in ("", "analysis", "harness"):
        p = str(Path(repo) / sub) if sub else str(Path(repo))
        if p not in sys.path:
            sys.path.insert(0, p)


def _rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def config_sessions(repo, cfg: str) -> tuple:
    """({(seed, task): task turn row}, {seed: first K (turn_idx, prompt)}, ceiling) of a config's completed
    sessions on SEEDS."""
    from fit_predictor import turn_prompt
    rel, arm, _tier = CONFIGS[cfg]
    rows = _rows(Path(repo) / rel)
    done = {(r["seed"], r.get("attempt_id")) for r in rows if r.get("record") == "r2a_session" and r["arm_id"] == arm}
    sess = {}
    for r in rows:
        if (r.get("record") == "r2a_turn" and r.get("arm_id") == arm and r["seed"] in SEEDS
                and (r["seed"], r.get("attempt_id")) in done):
            sess.setdefault(r["seed"], []).append(r)
    local, first, ceilings = {}, {}, set()
    for s, turns in sess.items():
        turns.sort(key=lambda r: r["turn_idx"])
        first[s] = [(t["turn_idx"], turn_prompt(t)) for t in turns[:K]]
        ceilings.add(turns[0].get("loaded_context"))
        for t in turns:
            if t.get("item_kind") == "task" and t.get("end_state_ok") is not None:
                local[(s, t["task_id"])] = t
    if len(ceilings) != 1:
        raise ValueError(f"{cfg}: loaded windows {ceilings}")
    return local, first, ceilings.pop()


def escalation_outcomes(repo, tier, cloud, redo, rb, imputed_usd) -> dict:
    """{(seed, task): {ok, usd, src}}: own-rollback redo, else full-history redo (configs the redo covers), else
    IMPUTED (all-cloud outcome at imputed_usd)."""
    out = {}
    for k, c in cloud.items():
        if tier and (tier, *k) in rb:
            out[k] = {**rb[(tier, *k)], "src": "rollback_redo"}
        elif tier and (tier, *k) in redo:
            out[k] = {"ok": redo[(tier, *k)]["ok"], "usd": redo[(tier, *k)]["usd"], "src": "full_history_redo"}
        else:
            out[k] = {"ok": c["ok"], "usd": imputed_usd, "src": "imputed"}
    return out


def _rollback_rows(repo) -> dict:
    p = Path(repo) / ROLLBACK_FILE
    out = {}
    if p.exists():
        for r in _rows(p):
            if r.get("record") == "cloud_redo_task" and r.get("mode") == "executed":
                out[(r["tier"], r["seed"], r["task_id"])] = {"ok": bool(r.get("cloud_end_state_ok")),
                                                             "usd": float(r["usd"])}
    return out


def replay_refs(repo=REPO) -> dict:
    _paths(repo)
    import impact_model as im
    with contextlib.redirect_stdout(io.StringIO()):
        ac = im.replay_cloud_tasks(repo)
        redo, _model = im.replay_redo_tasks(repo)
    cloud = {k: v for k, v in ac["models"][CLOUD_MODEL].items() if k[0] in SEEDS}
    tier_mean = {t: statistics.fmean(v["usd"] for (tr, _s, _t), v in redo.items() if tr == t)
                 for t in sorted({tr for tr, _s, _t in redo})}
    return {"cloud": cloud, "redo": redo, "rb": _rollback_rows(repo), "tier_mean": tier_mean,
            "imputed_usd": max(tier_mean.values())}


def hybrid_full_pipeline(repo=REPO, check=None, budget=None) -> dict:
    """The replay. check / budget default to A and C from their modules; without them the result is PENDING."""
    check = check if check is not None else load_post_run_check(repo)
    budget = budget if budget is not None else load_task_budget(repo)
    if check is None or budget is None:
        return {"status": "PENDING", "reason": PENDING_A if check is None else PENDING_C}
    refs = replay_refs(repo)
    cloud = refs["cloud"]
    cells = []
    for cfg, (_rel, _arm, tier) in CONFIGS.items():
        local, first, ceiling = config_sessions(repo, cfg)
        tasks = sorted(k for k in local if k in cloud)
        esc = escalation_outcomes(repo, tier, cloud, refs["redo"], refs["rb"], refs["imputed_usd"])
        r = run_pipeline(tasks, local, first, ceiling, cloud, esc, check, budget)
        naive = {"correct": sum(local[k]["end_state_ok"] is True for k in tasks), "n": len(tasks)}
        cells.append({"config": cfg, "ceiling": ceiling, "primary": cfg == PRIMARY, **r, "naive_local": naive})
    allc = {"correct": sum(c["ok"] for c in cloud.values()), "n": len(cloud),
            "usd_task": statistics.fmean(c["usd"] for c in cloud.values())}
    return {"status": "VERIFIED", "cells": cells, "all_cloud": allc, "imputed_usd": refs["imputed_usd"],
            "check_source": getattr(check, "source", type(check).__name__),
            "budget_source": getattr(budget, "source", type(budget).__name__)}


def hybrid_full_pipeline_text(d: dict) -> str:
    if d["status"] != "VERIFIED":
        return "PENDING: " + d["reason"]
    parts = []
    for c in d["cells"]:
        parts.append(f"{c['config']}{' (primary)' if c['primary'] else ''} ceiling {c['ceiling']}: kept local "
                     f"{c['kept_local']}/{c['n']}, accuracy {c['correct']}/{c['n']} ({100 * c['accuracy']:.1f}%), cloud "
                     f"USD {c['usd_task']:.5f}/task (routed {c['routed']}, escalated {c['escalated']} of which imputed "
                     f"{c['imputed_escalations']}, budget refused {c['budget_refused_route']} routes and "
                     f"{c['budget_refused_escalation']} escalations), wrong and kept local {c['wrong_kept_local']}; "
                     f"naive local {c['naive_local']['correct']}/{c['naive_local']['n']}; target met "
                     f"{c['meets_target']}")
    p = next(c for c in d["cells"] if c["primary"])
    return (f"measured by replay of recorded runs, {CLOUD_MODEL}, seeds 20260901-03; A = {d['check_source']}, C = "
            f"{d['budget_source']}; route at start = fit-predictor-v1's rule (k = {K}); imputed escalation USD "
            f"{d['imputed_usd']:.5f} (highest tier-mean full-history redo USD); all-cloud {d['all_cloud']['correct']}/"
            f"{d['all_cloud']['n']}, USD {d['all_cloud']['usd_task']:.5f}/task / " + " / ".join(parts)
            + f" / pre-registered target >= {TARGET_KEPT}/90 kept local at >= {100 * TARGET_ACC:.0f}% on {PRIMARY}: "
            + ("MET" if p["meets_target"] else "MISSED"))


# ── E design: hybrid-live-amd (evo-x2) and hybrid-live-intel (evo-t2s), 3 seeds each (ESTIMATE) ─────────────────

START_OVERHEAD_S = (60.0, 180.0)   # ASSUMED per session (as own-profiler-probe-hours): point, high
T2S_WINDOW_OPENED_UTC = "2026-10-28T00:39:00+00:00"   # docs/T2S_CHANGELOG.md 2026-10-28, open-window step 2


def live_hours(repo=REPO) -> dict:
    _paths(repo)
    sys.path.insert(0, str(Path(repo) / "scripts"))
    import impact_model as im
    import t2s_window_stage as ws
    refs = replay_refs(repo)
    with contextlib.redirect_stdout(io.StringIO()):
        redo_rows, _m, _n, _p = im.redo_rows_and_model(repo)
    redo_s = statistics.fmean(float(r["seconds"]) for r in redo_rows)
    upper_task = max(float(r["usd_all_uncached"]) for r in redo_rows)
    billed_task = refs["imputed_usd"]
    n_tasks = len(refs["cloud"])
    # AMD: measured q4_0 28928 sessions
    local, _first, _c = config_sessions(repo, PRIMARY)
    rows = _rows(Path(repo) / CONFIGS[PRIMARY][0])
    sess_s = {}
    for r in rows:
        if r.get("record") == "r2a_turn" and r["seed"] in SEEDS:
            sess_s[r["seed"]] = sess_s.get(r["seed"], 0.0) + float(r["turn_wall_s"])
    ns = len(SEEDS)
    cloud_high_h = n_tasks * redo_s / 3600
    amd = {"point_h": (sum(sess_s.values()) + ns * START_OVERHEAD_S[0]) / 3600,
           "high_h": (ns * max(sess_s.values()) + ns * START_OVERHEAD_S[1]) / 3600 + cloud_high_h,
           "sessions_min": sorted(round(v / 60, 1) for v in sess_s.values())}
    b3 = ws.b3_estimates()["b3_real_igpu"]
    intel = {"point_h": b3["hours"] + ns * START_OVERHEAD_S[0] / 3600,
             "high_h": b3["high_h"] + ns * START_OVERHEAD_S[1] / 3600 + cloud_high_h, "basis": b3["basis"]}
    p = ws.plan()
    staged_end = max(e["end_h"] for e in p["entries"] if e.get("end_h") is not None)
    opened = datetime.fromisoformat(T2S_WINDOW_OPENED_UTC)
    nolaunch = datetime.fromisoformat(ws.NO_LAUNCH_AFTER_UTC)
    end = opened + timedelta(hours=ws.START_ALLOWANCE_H + staged_end)
    left_h = (nolaunch - end).total_seconds() / 3600
    spend = {"billed_per_escalated": billed_task, "billed_all_escalated": n_tasks * billed_task,
             "allcloud_billed": n_tasks * statistics.fmean(c["usd"] for c in refs["cloud"].values()),
             "upper_per_task": upper_task, "upper_all": n_tasks * upper_task}
    return {"amd": amd, "intel": intel, "n_tasks": n_tasks, "redo_s": redo_s, "spend": spend,
            "staged_end_h": staged_end, "staged_end_utc": end, "left_h": left_h,
            "fits_after_queue": intel["high_h"] <= left_h - ws.SLACK_H,
            "fits_after_queue_no_slack": intel["high_h"] <= left_h, "slack_h": ws.SLACK_H}


def live_hours_text(d: dict) -> str:
    a, i, s = d["amd"], d["intel"], d["spend"]
    return (f"ESTIMATE, not a measurement (nothing deployed or queued); per run {d['n_tasks']} B3 v2 tasks (3 seeds), "
            f"qwen3:14b local, {CLOUD_MODEL} cloud, route + check + rollback + budget / hybrid-live-amd (evo-x2, KV "
            f"q4_0 num_ctx 28928): {a['point_h']:.2f} h (high {a['high_h']:.2f} h); point = the measured "
            f"x2_own_config_probe_q4_28928 sessions ({', '.join(str(m) for m in a['sessions_min'])} min) plus "
            f"{START_OVERHEAD_S[0]:.0f} s per session start (ASSUMED); high = 3 x the slowest session plus "
            f"{START_OVERHEAD_S[1]:.0f} s starts plus every task escalated at the mean full-history redo time "
            f"{d['redo_s']:.1f} s / hybrid-live-intel (evo-t2s iGPU, OLLAMA_IGPU_ENABLE=1, default window 32768): "
            f"{i['point_h']:.2f} h (high {i['high_h']:.2f} h); local part from row t2s-evalset-tasks-hours "
            f"(t2s_o26_b3_real_igpu: {i['basis']}), replaced by the measured file once t2s_o26_b3_real_igpu syncs; "
            f"same start and escalation terms / cloud spend per run: billed (cache-priced) USD {s['billed_per_escalated']:.5f} "
            f"per escalated task (highest tier-mean full-history redo), USD {s['billed_all_escalated']:.2f} if all "
            f"{d['n_tasks']} escalate (all-cloud billed reference USD {s['allcloud_billed']:.2f}); upper bound (no cache, "
            f"largest recorded full-history redo task USD {s['upper_per_task']:.5f}) USD {s['upper_all']:.2f}; both runs "
            f"together upper bound USD {2 * s['upper_all']:.2f} / evo-t2s queue: staged entries end {d['staged_end_h']:.1f} h "
            f"after the window opened plus the start allowance (row t2s-b5-plan-hours entries; window opened "
            f"{T2S_WINDOW_OPENED_UTC[:16]}Z, docs/T2S_CHANGELOG.md), projected {d['staged_end_utc'].strftime('%m-%d %H:%MZ')}, "
            f"{d['left_h']:.1f} h before no-launch; the Intel run at its high case fits after the staged queue with "
            f"the {d['slack_h']:.0f} h slack: {d['fits_after_queue']}; without the slack: "
            f"{d['fits_after_queue_no_slack']}; it must not displace a staged entry without the operator's say")
