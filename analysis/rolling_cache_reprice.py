"""Reprice the priced and replayed cloud figures under the rolling cache breakpoint (external review, 2026-11-02).

Offline only: no call is made. Every input is a committed result file; the recorded runs keep their billed spend
(the cloud ledger and each row's own `usd` are what was actually paid and do not change). What changes is the PRICED or
REPLAYED figure a deck quotes, under the placement the live client uses from 2026-11-02 on
(src/cloud/prompt_cache.py, a breakpoint on the system prompt and a rolling one on the latest message).

Two inputs, two methods:
  B3 v2 recorded runs (results/all_cloud_b3_v1.jsonl, the full-history redo parts, results/own_rollback_redo_v1.jsonl):
      every call's recorded input and output tokens, repriced with prompt_cache.rolling_split. all_cloud_b3_v1 carries a
      per-call split, so a session's first call keeps its recorded split (its system prompt was already cached by the
      warm-up) and every later call of the session, across task turns, reads the previous call's prompt. The redo rows
      carry per-call input tokens and task totals only; their first call's split is recovered from the totals under
      the earlier placement (prompt_cache.first_split_from_totals, which raises if the totals do not fit it).
  Real agent traces (analysis/cloud_task_cost.py): its "rolling" case beside the ASSUMED 0.7 hit rate of "realistic".
Prices: results/cloud_prices_anthropic_20261014.json, Standard, 5-minute cache writes (cache_write), hits
(cached_input), as every other priced row.
"""
from __future__ import annotations

import contextlib
import io
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "analysis", REPO / "src", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from cloud import prompt_cache as pc  # noqa: E402

ALL_CLOUD_FILE = "results/all_cloud_b3_v1.jsonl"
ROLLBACK_FILE = "results/own_rollback_redo_v1.jsonl"
PRICE_FILE = "results/cloud_prices_anthropic_20261014.json"
SEEDS = (20260901, 20260902, 20260903)
OLD_PLACEMENT = "system prompt and last user message (recorded runs before 2026-11-02)"
ASSUMPTIONS = ("ASSUMPTIONS of the rolling model: consecutive calls of one conversation under the 5-minute cache "
               "lifetime, byte-identical prefix between calls (same system prompt, tools and tool_choice), at most 20 "
               "blocks added per call; a call whose prompt shrank reads nothing")


def _rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def _cr(repo):
    import impact_model as im
    return im._load_module(repo, "scripts/cloud_redo.py", "cloud_redo")


def rates_for(repo, model):
    cr = _cr(repo)
    return cr.model_rates(cr.load_price_table(Path(repo) / PRICE_FILE), model)


def old_cost(rates, calls) -> float:
    """USD of recorded calls at their recorded split (the cost the run itself priced)."""
    return pc.split_cost(rates, [pc.recorded_first_split(c) for c in calls],
                         [int(c["input_tokens"]) for c in calls], [int(c.get("output_tokens") or 0) for c in calls])


# ── B3 v2 all-cloud ──────────────────────────────────────────────────────────────────────────────────

def reprice_all_cloud_sessions(turn_rows: list, rates_by_model: dict) -> dict:
    """Pure. turn_rows: all_cloud_b3 r2a_turn rows (with cloud_calls carrying the per-call split). Per model, per
    (seed, task_id): {ok, usd_recorded (the row's usd), usd_old (recomputed from the recorded split), usd_new (rolling),
    read/write/uncached old and new}. A session = (model, attempt_id), its calls concatenated in turn order."""
    sess = {}
    for r in turn_rows:
        sess.setdefault((r["model_id"], r["attempt_id"]), []).append(r)
    out = {}
    for (model, _att), turns in sess.items():
        turns.sort(key=lambda r: r["turn_idx"])
        rates = rates_by_model[model]
        calls = [c for t in turns for c in t["cloud_calls"]]
        rep = pc.reprice_recorded_calls(calls, rates, rates["min_cacheable"])
        i = 0
        for t in turns:
            n = len(t["cloud_calls"])
            sp = rep["split"][i:i + n]
            tc = calls[i:i + n]
            usd_new = pc.split_cost(rates, sp, [int(c["input_tokens"]) for c in tc],
                                    [int(c.get("output_tokens") or 0) for c in tc])
            out.setdefault(model, {})[(t["seed"], t["task_id"])] = {
                "ok": bool(t["end_state_ok"]), "usd_recorded": float(t["usd"]), "usd_old": old_cost(rates, tc),
                "usd_new": usd_new, "read_old": sum(int(c["cache_read_tokens"]) for c in tc),
                "write_old": sum(int(c["cache_write_tokens"]) for c in tc),
                "uncached_old": sum(int(c["uncached_input_tokens"]) for c in tc),
                "read_new": sum(s[0] for s in sp), "write_new": sum(s[1] for s in sp),
                "uncached_new": sum(s[2] for s in sp), "seconds": float(t["seconds"])}
            i += n
    return out


def all_cloud_rolling(repo=REPO, price_model=None, source_model=None) -> dict:
    """Default: every recorded model at its own prices. With price_model (and source_model, the recorded run whose
    tokens are used): the source run's recorded per-call tokens priced at price_model's list prices, keyed by
    price_model. That is a PRICED figure, not a run on price_model: the outcomes (ok) and token counts are the source
    model's, and usd_recorded is replaced by usd_old (the source run's billed USD was paid at the source's prices)."""
    rows = [r for r in _rows(Path(repo) / ALL_CLOUD_FILE)
            if r.get("record") == "r2a_turn" and r.get("item_kind") == "task" and r.get("end_state_ok") is not None]
    if price_model is None:
        models = sorted({r["model_id"] for r in rows})
        return reprice_all_cloud_sessions(rows, {m: rates_for(repo, m) for m in models})
    rows = [r for r in rows if r["model_id"] == source_model]
    if not rows:
        raise ValueError(f"no recorded all-cloud rows for {source_model}")
    out = reprice_all_cloud_sessions(rows, {source_model: rates_for(repo, price_model)})[source_model]
    return {price_model: {k: {**v, "usd_recorded": v["usd_old"]} for k, v in out.items()}}


# ── redo rows (full-history redo, own-rollback redo) ──────────────────────────────────────────────────

def reprice_redo_row(row: dict, rates: dict) -> dict:
    """Pure. One cloud_redo_task row: per-call input/output tokens, task totals of read / write / uncached."""
    calls = row["cloud_calls"]
    prompts = [int(c["input_tokens"]) for c in calls]
    outputs = [int(c.get("output_tokens") or 0) for c in calls]
    first = pc.first_split_from_totals(prompts, int(row["cache_read_tokens"]), int(row["cache_write_tokens"]))
    split = pc.rolling_split(prompts, rates["min_cacheable"], first)
    old = pc.split_cost(rates, [(int(row["cache_read_tokens"]), int(row["cache_write_tokens"]),
                                 int(row["uncached_input_tokens"]))], [sum(prompts)], [sum(outputs)])
    return {"usd_recorded": float(row["usd"]), "usd_old": old, "usd_new": pc.split_cost(rates, split, prompts, outputs)}


def redo_rolling(repo=REPO, price_model=None) -> dict:
    """{(tier, seed, task_id): {ok, usd_recorded, usd_old, usd_new}} of the merged full-history redo (superseded rows
    excluded, as impact_model.replay_redo_tasks) and {(tier, seed, task_id): ...} of the own-rollback redo.
    With price_model: the same recorded rows (run on the recorded model) priced at price_model's list prices, keyed
    model = price_model; usd_recorded is replaced by usd_old (PRICED from the recorded tokens, not run on it)."""
    import impact_model as im
    with contextlib.redirect_stdout(io.StringIO()):
        tasks, model, _n, _p = im.redo_rows_and_model(repo)
    rates = rates_for(repo, price_model or model)

    def rep(r, rt):
        x = reprice_redo_row(r, rt)
        return {**x, "usd_recorded": x["usd_old"]} if price_model else x
    redo = {(r["tier"], r["seed"], r["task_id"]): {"ok": bool(r.get("cloud_end_state_ok")), **rep(r, rates)}
            for r in tasks}
    rb = {}
    p = Path(repo) / ROLLBACK_FILE
    if p.exists():
        for r in _rows(p):
            if r.get("record") == "cloud_redo_task" and r.get("mode") == "executed":
                rt = rates if price_model else rates_for(repo, r["model_id"])
                rb[(r["tier"], r["seed"], r["task_id"])] = {"ok": bool(r.get("cloud_end_state_ok")), **rep(r, rt)}
    return {"model": price_model or model, "recorded_model": model, "redo": redo, "rb": rb}


# ── the figures the deck quotes ───────────────────────────────────────────────────────────────────────

def _mean(xs):
    return statistics.fmean(xs) if xs else None


def b3_all_cloud_summary(ac: dict, seeds=SEEDS) -> dict:
    out = {}
    for m, d in sorted(ac.items()):
        v = [x for k, x in d.items() if k[0] in set(seeds)]
        out[m] = {"n": len(v), "correct": sum(x["ok"] for x in v), "usd_recorded": _mean([x["usd_recorded"] for x in v]),
                  "usd_old": _mean([x["usd_old"] for x in v]), "usd_new": _mean([x["usd_new"] for x in v]),
                  "read_old": sum(x["read_old"] for x in v), "write_old": sum(x["write_old"] for x in v),
                  "uncached_old": sum(x["uncached_old"] for x in v), "read_new": sum(x["read_new"] for x in v),
                  "write_new": sum(x["write_new"] for x in v), "uncached_new": sum(x["uncached_new"] for x in v)}
    return out


def route_at_start_rolling(repo=REPO, ac=None, rd=None, tier="ollama_ctx_8192") -> dict:
    """hybrid-route-at-start-measured's per-task replay (impact_model.route_measured_policy, unchanged) at one tier vs
    each cloud tier, once with the recorded USD and once with the rolling USD of the same rows."""
    import impact_model as im
    ac = ac or all_cloud_rolling(repo)
    rd = rd or redo_rolling(repo)
    local = im.replay_local_tasks(repo)[tier]
    w = im.tier_window(tier)
    proj = {}
    for s in SEEDS:
        for tid, (_i, p) in im.projected_task_prompts(repo, s).items():
            proj[(s, tid)] = p
    cells = []
    for m in sorted(ac):
        res = {}
        for col in ("usd_recorded", "usd_new"):
            cloud = {k: {"ok": v["ok"], "usd": v[col], "seconds": v["seconds"]} for k, v in ac[m].items()}
            measured = m == rd["model"]
            t_redo = ({(s, t): {"ok": v["ok"], "usd": v[col], "seconds": 0.0} for (tr, s, t), v in rd["redo"].items()
                       if tr == tier} if measured else None)
            t_usd = _mean([v["usd"] for v in t_redo.values()]) if t_redo else None
            esc = ({k: {"ok": v["ok"], "usd": v[col], "seconds": v["seconds"]} for k, v in ac[rd["model"]].items()}
                   if measured else cloud)
            h = im.route_measured_policy(local, cloud, proj, w, SEEDS, redo=t_redo, redo_tier_usd=t_usd,
                                         esc_cloud=esc)
            keys = sorted(h["tasks"])
            allc = _mean([cloud[k]["usd"] for k in keys])
            res[col] = {"hybrid": h["usd_task"], "all_cloud": allc, "saving": 1 - h["usd_task"] / allc,
                        "correct": h["correct"], "n": h["n"], "routed": h["routed"], "escalated": h["escalated"]}
        cells.append({"model": m, **res})
    return {"tier": tier, "window": w, "cells": cells}


def operating_point_rolling(repo=REPO, ac=None, rd=None, configs=("q4_0 28928", "f16 default"), check=None,
                            budget=None, cloud_model=None) -> dict:
    """hybrid-full-pipeline-v1's replay (hybrid_full_pipeline.run_pipeline, unchanged) for the named configs, once at
    the recorded USD and once at the rolling USD: the escalation USD of every escalated task (own-rollback redo, else
    full-history redo, else IMPUTED at the highest tier-mean full-history redo USD) and the all-cloud bill."""
    import hybrid_full_pipeline as hfp
    ac = ac or all_cloud_rolling(repo)
    rd = rd or redo_rolling(repo)
    check = check if check is not None else hfp.load_post_run_check(repo)
    budget_factory = (lambda: budget) if budget is not None else (lambda: hfp.load_task_budget(repo))
    out = {}
    for col in ("usd_recorded", "usd_new"):
        cloud = {k: {"ok": v["ok"], "usd": v[col]} for k, v in ac[cloud_model or hfp.CLOUD_MODEL].items()
                 if k[0] in set(hfp.SEEDS)}
        redo = {k: {"ok": v["ok"], "usd": v[col]} for k, v in rd["redo"].items()}
        rb = {k: {"ok": v["ok"], "usd": v[col]} for k, v in rd["rb"].items()}
        tier_mean = {t: _mean([v["usd"] for (tr, _s, _t), v in redo.items() if tr == t])
                     for t in sorted({tr for tr, _s, _t in redo})}
        imputed = max(tier_mean.values())
        allc = _mean([c["usd"] for c in cloud.values()])
        cells = {}
        for cfg in configs:
            _rel, _arm, tier = hfp.CONFIGS.get(cfg) or hfp.COMPARE_CONFIGS[cfg]
            local, first, ceiling = hfp.config_sessions(repo, cfg)
            tasks = sorted(k for k in local if k in cloud)
            esc = hfp.escalation_outcomes(repo, tier, cloud, redo, rb, imputed)
            r = hfp.run_pipeline(tasks, local, first, ceiling, cloud, esc, check, budget_factory())
            cells[cfg] = {**r, "pct_bill": 100 * r["usd_task"] / allc, "ceiling": ceiling}
        out[col] = {"all_cloud": allc, "imputed_usd": imputed, "cells": cells}
    return out


def real_task_rolling(repo=REPO, res=None) -> dict:
    """cloud_task_cost.compute() with its rolling case beside realistic (0.7 hit ASSUMED) and best."""
    import cloud_task_cost as ct
    return res or ct.compute(Path(repo))
