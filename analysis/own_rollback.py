"""Own-rollback (operator item 1, 2026-10-23): does rolling the environment back before escalating fix the 28 tasks the
full-history redo left wrong, and what does a local-first hybrid with rollback give. Offline, no network.

1. rollback_analysis (step 2, before any spend). For each of the 28 earlier_state_repair failures of the full-history
   redo (row cloud-redo-full-context, classes from analysis/cloud_redo_diagnosis.py):
   - rollback_env (scripts/cloud_redo.py: the local session replayed on evalset_env.SnapshotLog, the failed turn's own
     calls applied, then the pre-task mark restored) against rebuild()'s env_before, the state the redo already used;
   - the failed turn's state-changing calls that the rollback undoes;
   - the earlier task turns of the session that ended wrong, and the guardrail turns that changed state;
   - the "clean" pre-task state: the local session replayed with every earlier WRONG task turn replaced by its
     reference calls (what a rollback hybrid with the oracle signal leaves if each earlier escalation succeeded), and
     whether every record of the redo's end-state diff is a record where env_before differs from that clean state.
   The grader (evalset_env.check_task) sets expected = the task's reference applied to env_before, so a cloud model
   that also restores records the local model broke in EARLIER turns is scored wrong by construction.

2. local_first_rollback (step 4, row hybrid-local-first-rollback). By replay of recorded runs: every task runs local
   first (B3 v2 real run, qwen3:14b, the tiers of hybrid-route-at-start-measured, seeds shared with the all-cloud run);
   a task the signal fires on is rolled back (local snapshots, USD 0) and escalated to claude-sonnet-5-5. Signals:
   "oracle" (the task ended wrong; ceiling, oracle signal) and "window" (the recorded over_loaded_window condition,
   impact_model.turn_over_window: a STAND-IN for detector v2, Ollama's own "truncating input prompt" log line, which the
   B3 v2 run did not keep). Escalated task outcome and cache-priced USD: the own-rollback redo row
   (results/own_rollback_redo_v1.jsonl) when present, else the full-history redo row of the same tier, seed and task
   (labelled; its env_before is the pre-task rollback state, checked per task in step 1), else (a task the window flag
   escalates that was right locally, so it has no redo row) IMPUTED as in hybrid-route-at-start-measured: the tier's
   mean redo USD and the Sonnet all-cloud outcome.
"""
from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROLLBACK_FILE = "results/own_rollback_redo_v1.jsonl"
HYBRID_ROW = "hybrid-local-first-rollback"
REDO_ROW = "own-rollback-redo"
TARGET_FIXED = 25
SIGNALS = ("oracle", "window")
SIGNAL_LABEL = {"oracle": "ceiling, oracle signal (escalate iff the task ended wrong)",
                "window": ("measured signal, STAND-IN: the recorded over_loaded_window condition (detector v2's Ollama "
                           "cut-off log line was not kept by the B3 v2 run)")}


def _load(repo: Path, rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(repo) / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, m)
    spec.loader.exec_module(m)
    return m


def _quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def redo_diagnosis(repo: Path = REPO) -> tuple:
    """(cloud_redo module, diagnosis module, diagnosis dict, merged full-history redo rows)."""
    cr = _load(repo, "scripts/cloud_redo.py", "cloud_redo")
    cd = _load(repo, "analysis/cloud_redo_diagnosis.py", "cloud_redo_diagnosis")
    paths = cr.part_files(Path(repo) / "results" / cr.OUT_FULL_CONTEXT.name)
    d = _quiet(cd.compute, Path(repo), paths[0].relative_to(repo).as_posix(),
               tuple(q.relative_to(repo).as_posix() for q in paths[1:]))
    return cr, cd, d, cr.merged_rows(paths)


def clean_pre_task_env(E, es, prior: list[dict]) -> dict:
    """The local session replayed with every earlier wrong task turn replaced by its reference calls."""
    by_idx = {t.idx: it for t, it in zip(es.turns, es.items)}
    env = copy.deepcopy(es.env0)
    for x in prior:
        it = by_idx[x["turn_idx"]]
        if it.task is not None and x.get("end_state_ok") is False:
            calls = list(it.task.reference)
        else:
            calls = [c for rec in x.get("calls") or [] for c in rec.get("tool_calls_parsed") or []]
        for c in calls:
            E.apply_call(env, c)
    return env


def rollback_analysis(repo: Path = REPO) -> dict:
    cr, cd, d, merged = redo_diagnosis(repo)
    E = cr.E
    want = {(p["tier"], p["seed"], p["turn_idx"]) for p in d["per_task"] if p["cls"] == "earlier_state_repair"}
    rows = [r for r in merged if (r["tier"], r["seed"], r["turn_idx"]) in want]
    src = cr.load_rows(cr.SOURCE)
    wrong = {(r["attempt_id"], r["turn_idx"]): r for r in cr.wrong_tasks(src)}
    cache, per = {}, []
    for t in rows:
        r = wrong[(t["attempt_id"], t["turn_idx"])]
        es = _quiet(cr.build_session_for, r, cache)
        prior = [x for x in cr.session_rows(src, r["attempt_id"]) if x["turn_idx"] < r["turn_idx"]]
        built = _quiet(cr.rebuild, es, prior, r["turn_idx"])
        rb = cr.rollback_env(es, prior, r)
        by_idx = {tt.idx: it for tt, it in zip(es.turns, es.items)}
        earlier_wrong = sum(1 for x in prior if by_idx[x["turn_idx"]].task is not None
                            and x.get("end_state_ok") is False)
        guard_changed = 0
        env = copy.deepcopy(es.env0)
        for x in prior:
            before = copy.deepcopy(env)
            for c in (c for rec in x.get("calls") or [] for c in rec.get("tool_calls_parsed") or []):
                E.apply_call(env, c)
            guard_changed += by_idx[x["turn_idx"]].task is None and env != before
        clean = clean_pre_task_env(E, es, prior)
        dirty = {cd._diff_record(k) for k in E.env_diff(built["env_before"], clean, 10_000)}
        diff = {cd._diff_record(k) for k in t.get("cloud_end_state_diff") or []}
        per.append({"tier": t["tier"], "seed": t["seed"], "turn_idx": t["turn_idx"], "task_id": t["task_id"],
                    "rollback_equals_redo_env": rb["env"] == built["env_before"], "undone_calls": rb["undone_calls"],
                    "earlier_wrong_tasks": earlier_wrong, "guardrail_turns_changed_state": guard_changed,
                    "diff_records": sorted(diff), "dirty_records": sorted(dirty), "diff_within_dirty": diff <= dirty,
                    "usd": t["usd"], "usd_all_uncached": t["usd_all_uncached"]})
    n = len(per)
    return {"n": n, "per_task": per,
            "rollback_equals_redo_env": sum(p["rollback_equals_redo_env"] for p in per),
            "failed_turns_with_state_changes": sum(1 for p in per if p["undone_calls"]),
            "undone_calls": sum(p["undone_calls"] for p in per),
            "with_earlier_wrong": sum(1 for p in per if p["earlier_wrong_tasks"]),
            "earlier_wrong_min": min(p["earlier_wrong_tasks"] for p in per),
            "earlier_wrong_max": max(p["earlier_wrong_tasks"] for p in per),
            "guardrail_state_changes": sum(1 for p in per if p["guardrail_turns_changed_state"]),
            "diff_within_dirty": sum(p["diff_within_dirty"] for p in per),
            "billed_usd": sum(p["usd"] for p in per), "upper_usd": sum(p["usd_all_uncached"] for p in per),
            "billed_usd_mean": statistics.fmean(p["usd"] for p in per) if per else None,
            "model_id": rows[0]["model_id"] if rows else None}


def rollback_analysis_text(a: dict) -> str:
    n = a["n"]
    return (f"{n} earlier_state_repair tasks of the full-history redo ({a['model_id']}): pre-task rollback state "
            f"(snapshots, failed turn undone) equals the redo's env_before in {a['rollback_equals_redo_env']}/{n}; "
            f"failed local turns with state-changing calls undone {a['failed_turns_with_state_changes']}/{n} "
            f"({a['undone_calls']} calls); sessions with at least one earlier wrong task {a['with_earlier_wrong']}/{n} "
            f"({a['earlier_wrong_min']} to {a['earlier_wrong_max']} earlier wrong tasks); guardrail turns that changed "
            f"state {a['guardrail_state_changes']}/{n}; redo end-state diff only on records the local model broke in "
            f"earlier wrong tasks (env_before vs the clean pre-task state) {a['diff_within_dirty']}/{n}; measured cost "
            f"of these {n} in the full-history redo: billed (cache-priced) USD {a['billed_usd']:.4f} (mean "
            f"{a['billed_usd_mean']:.5f}), all input uncached USD {a['upper_usd']:.4f}")


# ── own-rollback redo row ───────────────────────────────────────────────────────────────────────────

def own_rollback_redo(repo: Path = REPO, path: str = ROLLBACK_FILE) -> dict:
    """Row own-rollback-redo: fixed / run with a 95% Wilson CI against the pre-registered target (>= 25/28)."""
    import numbers_register as nr
    cr = _load(repo, "scripts/cloud_redo.py", "cloud_redo")
    rows = cr.load_rows(Path(repo) / path)
    start = next(r for r in rows if r.get("record") == "run_start")
    end = next((r for r in rows if r.get("record") == "run_end"), {})
    tasks = [r for r in rows if r.get("record") == "cloud_redo_task"]
    n_all = start.get("n_tasks_this_run") or len(tasks)
    ok = sum(1 for r in tasks if r.get("cloud_end_state_ok"))
    lo, hi = nr.wilson_ci(ok, n_all)
    s = cr.summarize(tasks) if tasks else {"total_usd": 0.0, "mean_usd": None, "total_usd_all_uncached": 0.0}
    return {"start": start, "end": end, "tasks": tasks, "n_all": n_all, "fixed": ok, "ci": (lo, hi), "summary": s,
            "target_met": ok >= TARGET_FIXED}


def own_rollback_redo_text(r: dict) -> str:
    st, s = r["start"], r["summary"]
    eq = sum(1 for t in r["tasks"] if t.get("rollback_env_equals_rebuild"))
    return (f"model {st['model_id']}, rollback {st.get('rollback')}, thinking {st.get('thinking_sent')}, max_tokens "
            f"{st.get('max_tokens')}, full client-side history; fixed {r['fixed']}/{r['n_all']} "
            f"[{r['ci'][0]:.1%}, {r['ci'][1]:.1%}] (target >= {TARGET_FIXED}/{r['n_all']}: "
            f"{'met' if r['target_met'] else 'not met'}); rollback env equals the full-history redo env_before "
            f"{eq}/{len(r['tasks'])}; USD billed (cache-priced) total {s['total_usd']:.4f}, all input uncached "
            f"{s['total_usd_all_uncached']:.4f}; stopped: {r['end'].get('stopped') if r['end'] else 'no run_end'}")


# ── local-first hybrid with rollback (replay) ───────────────────────────────────────────────────────

def local_first_policy(local: dict, cloud: dict, redo: dict, rb: dict, signal: str, tier_usd: float,
                       classes: dict | None = None) -> dict:
    """Pure. local {(seed, task): {flag, ok, turn_idx}}; cloud {(seed, task): {ok, usd}} (Sonnet all-cloud);
    redo / rb {(seed, task): {ok, usd}} (full-history redo / own-rollback redo at this tier); classes
    {(seed, turn_idx): diagnosis class} of still-wrong full-history redo tasks."""
    if signal not in SIGNALS:
        raise ValueError(signal)
    keys = sorted(k for k in local if k in cloud)
    if not keys:
        raise ValueError("no shared (seed, task) pairs")
    tasks = {}
    for k in keys:
        lo = local[k]
        esc = (not lo["ok"]) if signal == "oracle" else bool(lo["flag"])
        if not esc:
            tasks[k] = {"ok": lo["ok"], "usd": 0.0, "src": "local"}
        elif k in rb:
            tasks[k] = {"ok": rb[k]["ok"], "usd": rb[k]["usd"], "src": "rollback_redo"}
        elif k in redo:
            tasks[k] = {"ok": redo[k]["ok"], "usd": redo[k]["usd"], "src": "full_history_redo"}
        else:
            if not lo["ok"]:
                raise ValueError(f"wrong local task {k} has no redo row")
            tasks[k] = {"ok": cloud[k]["ok"], "usd": tier_usd, "src": "imputed"}
    n = len(keys)
    cnt = {s: sum(1 for t in tasks.values() if t["src"] == s)
           for s in ("local", "rollback_redo", "full_history_redo", "imputed")}
    unfixed = [k for k, t in tasks.items() if t["src"] in ("rollback_redo", "full_history_redo") and not t["ok"]]
    classes = classes or {}
    rep = sum(1 for k in unfixed if tasks[k]["src"] == "full_history_redo"
              and classes.get((k[0], local[k]["turn_idx"])) == "earlier_state_repair")
    correct = sum(1 for t in tasks.values() if t["ok"])
    usd = sum(t["usd"] for t in tasks.values())
    return {"n": n, "correct": correct, "usd_task": usd / n, "kept_local": cnt["local"], **cnt,
            "escalated": n - cnt["local"], "escalated_unfixed": len(unfixed), "unfixed_repair": rep, "tasks": tasks}


def local_first_rollback(repo: Path = REPO) -> dict:
    im = _load(repo, "analysis/impact_model.py", "impact_model")
    missing = [c for c in im.ROUTE_M_REQUIRED if not im.register_has(repo, c)]
    if missing:
        return {"status": "PENDING", "pending": missing}
    local_all = im.replay_local_tasks(repo)
    ac = im.replay_cloud_tasks(repo)
    redo_all, model = im.replay_redo_tasks(repo)
    seeds = set(ac["seeds"])
    cloud = ac["models"][model]
    rbp = Path(repo) / ROLLBACK_FILE
    rb_all = {}
    if rbp.exists():
        for r in (json.loads(x) for x in rbp.read_text(encoding="utf-8").splitlines() if x.strip()):
            if r.get("record") == "cloud_redo_task" and r.get("mode") == "executed":
                rb_all[(r["tier"], r["seed"], r["task_id"])] = {"ok": bool(r.get("cloud_end_state_ok")),
                                                                 "usd": float(r["usd"])}
    _cr, _cd, d, _m = redo_diagnosis(repo)
    cells = []
    for tier in im.ROUTE_M_TIERS:
        local = {k: v for k, v in local_all[tier].items() if k[0] in seeds}
        redo = {(s, t): v for (tr, s, t), v in redo_all.items() if tr == tier}
        rb = {(s, t): v for (tr, s, t), v in rb_all.items() if tr == tier}
        t_usd = statistics.fmean(v["usd"] for v in redo.values())
        classes = {(p["seed"], p["turn_idx"]): p["cls"] for p in d["per_task"] if p["tier"] == tier}
        keys = sorted(k for k in local if k in cloud)
        ref = {"all_cloud": {"correct": sum(cloud[k]["ok"] for k in keys),
                             "usd_task": sum(cloud[k]["usd"] for k in keys) / len(keys)},
               "naive_local": {"correct": sum(local[k]["ok"] for k in keys), "usd_task": 0.0}}
        pol = {s: local_first_policy(local, cloud, redo, rb, s, t_usd, classes) for s in SIGNALS}
        cells.append({"tier": tier, "window": im.tier_window(tier), "n": len(keys), "tier_mean_redo_usd": t_usd,
                      **ref, "policies": pol})
    return {"status": "VERIFIED", "cells": cells, "model": model, "seeds": sorted(seeds),
            "rollback_rows": len(rb_all), "label": im.REPLAY_LABEL}


def _pc(c, n):
    return f"{c}/{n} ({100 * c / n:.1f}%)"


def local_first_rollback_text(d: dict) -> str:
    if d["status"] != "VERIFIED":
        return f"PENDING: rows {', '.join(d['pending'])} not VERIFIED"
    parts = []
    for c in d["cells"]:
        n = c["n"]
        sig = []
        for s in SIGNALS:
            p = c["policies"][s]
            sig.append(f"{SIGNAL_LABEL[s]}: accuracy {_pc(p['correct'], n)}, cloud USD {p['usd_task']:.5f}/task, kept "
                       f"local {_pc(p['kept_local'], n)}, escalated {p['escalated']} (graded from the own-rollback "
                       f"redo {p['rollback_redo']}, from the full-history redo {p['full_history_redo']} (its env_before "
                       f"is the pre-task rollback state), imputed {p['imputed']} (right locally, no redo row: tier mean "
                       f"redo USD {c['tier_mean_redo_usd']:.5f}, Sonnet all-cloud outcome)); escalated and still wrong "
                       f"{p['escalated_unfixed']}, of them earlier_state_repair {p['unfixed_repair']}")
        parts.append(f"evo-x2 {c['tier']} (window {c['window']}), {d['model']}: " + "; ".join(sig)
                     + f"; all-cloud {_pc(c['all_cloud']['correct'], n)}, USD {c['all_cloud']['usd_task']:.5f}/task; "
                     f"naive local {_pc(c['naive_local']['correct'], n)}, USD 0")
    return (f"{d['label']}: local first on every task, roll back (local snapshots, USD 0) and escalate on the signal; "
            f"B3 v2 tasks, qwen3:14b, seeds {', '.join(map(str, d['seeds']))}; cloud USD cache-priced (billed basis); "
            f"own-rollback redo rows available: {d['rollback_rows']}; replay assumes a task's local outcome does not "
            f"depend on earlier escalations (in the recorded run every earlier task ran local) / " + " / ".join(parts))


if __name__ == "__main__":
    print(rollback_analysis_text(rollback_analysis()))
