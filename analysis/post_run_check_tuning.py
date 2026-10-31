"""Post-run check, amendment 2026-10-31 (docs/EXPERIMENTS.md "AMENDMENT (2026-10-31): offline L3 tuning with a held-out
seed, then a small live run at num_ctx 8192"). Offline, no network.

`py analysis/post_run_check_tuning.py` writes results/post_run_check_tuned_v1.jsonl:
  run_start; one prt_task row per task turn (all three windows, all five seeds: the other layers that fired, every L3
  check and variant that failed, the grader outcome read after the check for scoring); prt_breakdown rows (tuning seeds
  only, per window); one prt_tuning row (the procedure's steps and the resulting config); prt_cell rows (per window,
  per seed set {tuning, holdout}, per config {v1, tuned}: the union decision and L3 alone); prt_live_estimate; run_end.

Held-out seed 20260905; tuning seeds 20260901-04. The tuning procedure (tune()) reads tuning-seed rows only.
"""
from __future__ import annotations

import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
import post_run_check_replay as R  # noqa: E402

OUT = "results/post_run_check_tuned_v1.jsonl"
HOLDOUT = 20260905
TUNE_SEEDS = (20260901, 20260902, 20260903, 20260904)
OTHER_LAYERS = ("L1_errors", "L1_proxy", "L2_proxy")
LIVE_TIER = "ollama_ctx_8192"
LIVE_JOB_CAP_USD = 2.00


def _l3():
    R._paths(REPO)
    from src.dse import l3_invariants as L3
    return L3


# ── pure procedure ─────────────────────────────────────────────────────────────────────────────

def fired(task: dict, config: dict, l3_only: bool = False) -> list:
    L3 = _l3()
    out = [] if l3_only else [x for x in task["other_fired"]]
    if L3.select(task["l3_all"], config):
        out.append("L3")
    return out


def counts(tasks: list[dict], name: str, config: dict) -> tuple[int, int]:
    """(F_right, U_wrong) of the check in use called `name` under config."""
    L3 = _l3()
    f_right = sum(1 for t in tasks if t["local_ok"] and name in t["l3_all"])
    u_wrong = sum(1 for t in tasks if not t["local_ok"] and L3.select(t["l3_all"], config) == [name])
    return f_right, u_wrong


def tune(tasks: list[dict]) -> dict:
    """The amendment's procedure on the given (tuning-seed) task rows. Returns {config, changelog}."""
    L3 = _l3()
    if any(t["seed"] == HOLDOUT for t in tasks):
        raise ValueError("tune() must not see the held-out seed")
    cfg = {"variants": {}, "drop": []}
    log = []
    for c in L3.CHECK_ORDER:
        f_c, u_c = counts(tasks, c, cfg)
        step = {"check": c, "F_right": f_c, "U_wrong": u_c, "action": "keep"}
        if f_c == 0:
            log.append(step)
            continue
        in_use = c
        v = L3.VARIANTS.get(c)
        if v:
            trial = {"variants": {**cfg["variants"], c: v}, "drop": list(cfg["drop"])}
            f_v, u_v = counts(tasks, v, trial)
            step.update({"variant": v, "F_right_variant": f_v, "U_wrong_variant": u_v})
            if f_v < f_c and (u_c - u_v) <= (f_c - f_v):
                cfg, in_use = trial, v
                step["action"] = f"tighten to {v}"
        f_u, u_u = counts(tasks, in_use, cfg)
        if f_u > u_u:
            cfg = {"variants": {k: x for k, x in cfg["variants"].items() if k != c}, "drop": [*cfg["drop"], c]}
            step["action"] = "drop" if step["action"] == "keep" else step["action"] + ", then drop"
        step.update({"F_right_in_use": f_u, "U_wrong_in_use": u_u})
        log.append(step)
    return {"config": {"variants": cfg["variants"], "drop": sorted(cfg["drop"], key=L3.CHECK_ORDER.index)},
            "changelog": log}


def breakdown(tasks: list[dict]) -> dict:
    """Per check and variant: firings on wrong and on correct tasks, and unique catches under the v1 checks."""
    L3 = _l3()
    names = list(L3.CHECK_ORDER) + list(L3.VARIANTS.values())
    out = {}
    for n in names:
        fw = sum(1 for t in tasks if not t["local_ok"] and n in t["l3_all"])
        fr = sum(1 for t in tasks if t["local_ok"] and n in t["l3_all"])
        uw = (sum(1 for t in tasks if not t["local_ok"] and L3.select(t["l3_all"], L3.CONFIG_V1) == [n])
              if n in L3.CHECK_ORDER else None)
        if fw or fr:
            out[n] = {"fires_wrong": fw, "fires_right": fr, "unique_wrong": uw}
    return {"n": len(tasks), "wrong": sum(1 for t in tasks if not t["local_ok"]),
            "right": sum(1 for t in tasks if t["local_ok"]), "checks": out}


def cell(tasks: list[dict], config: dict, src: dict, tier_mean: dict) -> dict:
    union = [{**t, "fired": fired(t, config)} for t in tasks]
    alone = [{**t, "fired": fired(t, config, l3_only=True)} for t in tasks]
    pu = R.policy(union, R.LAYER_ORDER, src, tier_mean)
    pa = R.policy(alone, ("L3",), src, tier_mean)
    return {"union": pu, "union_targets": R.targets(pu), "l3_alone": pa, "l3_alone_targets": R.targets(pa)}


# ── run ────────────────────────────────────────────────────────────────────────────────────────

def run(repo: Path = REPO, out: str = OUT) -> list[dict]:
    repo = Path(repo)
    R._paths(repo)
    from src.dse import post_run_check as prc
    L3 = _l3()
    turns = R.local_turns(repo, R.SOURCE)
    seeds = sorted({r["seed"] for v in turns.values() for r in v})
    sess = R.sessions(repo, seeds)
    src = R.cloud_sources(repo)
    rows = [{"record": "run_start", "ts_utc": datetime.now(timezone.utc).isoformat(), "harness": "post_run_check_tuning",
             "amendment": "docs/EXPERIMENTS.md AMENDMENT (2026-10-31)", "source": R.SOURCE,
             "source_git_blob": R._git_blob(repo / R.SOURCE), "holdout_seed": HOLDOUT, "tuning_seeds": list(TUNE_SEEDS),
             "tiers": list(R.TIERS), "task_cap_usd": R.TASK_CAP_USD,
             "targets": {"recall_min": R.TARGET_RECALL, "false_esc_max": R.TARGET_FALSE_ESC,
                         "accuracy_min": R.TARGET_ACC},
             "l3_module_git_blob": R._git_blob(repo / "src/dse/l3_invariants.py"),
             "check_module_git_blob": R._git_blob(repo / "src/dse/post_run_check.py")}]
    tasks = []
    for tier in R.TIERS:
        for r in sorted(turns.get(tier, []), key=lambda x: (x["seed"], x["turn_idx"])):
            sp, users, schema = sess[r["seed"]]
            if any(c.get("system_prompt_sha256") not in (None, R._sha(sp)) for c in r.get("calls") or []):
                raise RuntimeError("rebuilt system prompt differs from the recorded sha256")
            d = prc.check_record(r, sp, users[r["turn_idx"]], schema, L3.CONFIG_V1)
            t = {"record": "prt_task", "tier": tier, "seed": r["seed"], "turn_idx": r["turn_idx"],
                 "task_id": r["task_id"], "other_fired": [x for x in d["fired"] if x in OTHER_LAYERS],
                 "l3_all": d["layers"]["L3"]["all_failed"], "turn_wall_s": r.get("turn_wall_s"),
                 "n_calls": len(r.get("calls") or []),
                 "local_ok": r["end_state_ok"] is True}   # grader outcome, read after the check, scoring only
            tasks.append(t)
            rows.append(t)
    tune_tasks = [t for t in tasks if t["seed"] in TUNE_SEEDS]
    for tier in R.TIERS:
        rows.append({"record": "prt_breakdown", "tier": tier,
                     **breakdown([t for t in tune_tasks if t["tier"] == tier])})
    tn = tune(tune_tasks)
    rows.append({"record": "prt_tuning", **tn})
    configs = {"v1": L3.CONFIG_V1, "tuned": tn["config"]}
    for tier in R.TIERS:
        tr = [v for (t, _s, _k), v in src["redo"].items() if t == tier]
        tier_mean = {"usd": statistics.fmean(v["usd"] for v in tr), "upper": statistics.fmean(v["upper"] for v in tr)}
        for sset, pick in (("tuning", lambda t: t["seed"] in TUNE_SEEDS), ("holdout", lambda t: t["seed"] == HOLDOUT)):
            ts = [t for t in tasks if t["tier"] == tier and pick(t)]
            for cname, cfg in configs.items():
                rows.append({"record": "prt_cell", "tier": tier, "seeds": sset, "config": cname,
                             "tier_mean_redo": tier_mean, **cell(ts, cfg, src, tier_mean)})
    rows.append({"record": "prt_live_estimate", **live_estimate(tasks, tn["config"], src, repo)})
    rows.append({"record": "run_end", "ts_utc": datetime.now(timezone.utc).isoformat(), "n_tasks": len(tasks)})
    with open(repo / out, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    return rows


def live_estimate(tasks, config, src, repo) -> dict:
    """ESTIMATE for the live one-seed run at num_ctx 8192 (seed 20260905): point = the held-out seed's replay decisions
    at 8192 with the tuned config (measured redo USD where the task has a redo row, tier mean otherwise); high = every
    one of the 30 tasks escalated at the largest measured per-task billed redo USD of the tier; upper = every task
    escalated at the largest measured all-input-uncached USD of the tier, bounded by the per-task cap; hours from the
    recorded turn times plus the live check overhead."""
    ts = [t for t in tasks if t["tier"] == LIVE_TIER and t["seed"] == HOLDOUT]
    tr = [v for (t, _s, _k), v in src["redo"].items() if t == LIVE_TIER]
    tier_mean = {"usd": statistics.fmean(v["usd"] for v in tr), "upper": statistics.fmean(v["upper"] for v in tr)}
    esc = [t for t in ts if fired(t, config)]
    pt = [R.grade(True, t["local_ok"], (LIVE_TIER, t["seed"], t["task_id"]), src, tier_mean) for t in esc]
    n = len(ts)
    max_billed = max(v["usd"] for v in tr)
    max_upper = max(v["upper"] for v in tr)
    mech = [r["check_duration_s"] for r in R._rows(Path(repo) / R.MECH) if r.get("record") == "r2m_prompt_check"
            and r.get("check_duration_s") is not None]
    n_calls = sum(t["n_calls"] for t in ts)
    redo_s = statistics.fmean(float(v["seconds"]) for v in tr if v.get("seconds") is not None)
    # prompt checks: x2_r2_mechanism.check_due caps them at CHECK_PRE_MAX + CHECK_TRUNC_MAX = 5 + 8 per session
    n_checks = 13
    local_s = sum(t["turn_wall_s"] for t in ts)
    hours = (local_s + n_calls * (2 * R.GIN_WAIT_S) + n_checks * statistics.median(mech) + len(esc) * redo_s) / 3600
    return {"tier": LIVE_TIER, "seed": HOLDOUT, "n_tasks": n, "escalated_replay": len(esc),
            "point_billed": sum(x["usd"] for x in pt), "point_upper": sum(x["upper"] for x in pt),
            "high_billed": n * max_billed, "upper_bound": min(n * min(max_upper, R.TASK_CAP_USD), n * R.TASK_CAP_USD),
            "cap_ceiling": n * R.TASK_CAP_USD, "job_cap": LIVE_JOB_CAP_USD, "max_billed_task": max_billed,
            "max_upper_task": max_upper, "hours": hours, "local_hours": local_s / 3600, "n_calls": n_calls,
            "prompt_check_s": statistics.median(mech), "n_checks": n_checks, "redo_s": redo_s}


# ── register text ──────────────────────────────────────────────────────────────────────────────

def load(repo: Path = REPO, path: str = OUT) -> dict:
    rows = R._rows(Path(repo) / path)
    return {"start": next(r for r in rows if r["record"] == "run_start"),
            "end": next((r for r in rows if r["record"] == "run_end"), None),
            "breakdown": {r["tier"]: r for r in rows if r["record"] == "prt_breakdown"},
            "tuning": next(r for r in rows if r["record"] == "prt_tuning"),
            "cells": {(r["tier"], r["seeds"], r["config"]): r for r in rows if r["record"] == "prt_cell"},
            "live": next(r for r in rows if r["record"] == "prt_live_estimate")}


def breakdown_text(d: dict) -> str:
    parts = []
    for tier in R.TIERS:
        b = d["breakdown"][tier]
        ch = "; ".join(f"{k}: fires on {v['fires_wrong']} wrong / {v['fires_right']} correct"
                       + (f", only L3 check on {v['unique_wrong']} wrong" if v["unique_wrong"] is not None else
                          " (tightened variant, not in use in v1)")
                       for k, v in b["checks"].items())
        parts.append(f"{tier} ({b['n']} task turns, {b['wrong']} wrong, {b['right']} correct): {ch}")
    return ("tuning seeds 20260901-04 only (held-out 20260905 not read), B3 v2 qwen3:14b, L3 checks of the "
            "pre-registered post-run check: " + " / ".join(parts))


def _mm(m):
    return ", ".join(f"{k} {'MET' if v else 'MISSED'}" for k, v in
                     (("recall >= 80%", m["recall"]), ("false escalations <= 10%", m["false_esc"]),
                      ("final accuracy >= 97%", m["accuracy"])))


def _p(p):
    return (f"recall {R._pc(p['caught'], p['wrong'])}, false escalations {R._pc(p['false_esc'], p['right'])}, final "
            f"accuracy {p['correct']:.2f}/{p['n']} ({100 * p['accuracy']:.1f}%), kept local "
            f"{R._pc(p['kept_local'], p['n'])}, cloud USD {p['usd_task']:.5f}/task billed ({p['upper_task']:.5f} upper "
            f"bound), imputed {p['imputed']}")


def tuned_text(d: dict) -> str:
    cfg = d["tuning"]["config"]
    changes = "; ".join(f"{s['check']}: {s['action']} (F_right {s['F_right']}, U_wrong {s['U_wrong']}"
                        + (f", variant {s['variant']} F_right {s['F_right_variant']} U_wrong {s['U_wrong_variant']}"
                           if s.get("variant") else "") + ")"
                        for s in d["tuning"]["changelog"] if s["action"] != "keep" or s["F_right"])
    head, tun = [], []
    for tier in R.TIERS:
        h = d["cells"][(tier, "holdout", "tuned")]
        head.append(f"{tier}: L1 errors + L1 PROXY + L2 PROXY + tuned L3: {_p(h['union'])}; targets "
                    f"{_mm(h['union_targets'])}; tuned L3 alone: {_p(h['l3_alone'])}; targets {_mm(h['l3_alone_targets'])}")
        t = d["cells"][(tier, "tuning", "tuned")]
        tun.append(f"{tier}: union {_p(t['union'])}; tuned L3 alone {_p(t['l3_alone'])}")
    return (f"HELD-OUT seed {HOLDOUT} only (30 task turns per window; tuned on seeds 20260901-04), replay of the B3 v2 "
            f"real run, qwen3:14b; L3 changes (amendment procedure): {changes or 'none'}; config: variants "
            f"{cfg['variants'] or 'none'}, dropped {', '.join(cfg['drop']) or 'none'}; held-out: " + " / ".join(head)
            + "; TUNING DATA (seeds 20260901-04, not the headline): " + " / ".join(tun)
            + "; held-out tasks right locally that escalate are graded IMPUTED from the task's all-cloud pass fraction "
              "(no all-cloud row for seed 20260905)")


def live8192_text(d: dict) -> str:
    e = d["live"]
    return (f"ESTIMATE, not measured: live post-run check at evo-x2 num_ctx 8192, seed {e['seed']}, {e['n_tasks']} "
            f"tasks, tuned L3, real L1 (detector v2) and L2 (render check): billed point USD {e['point_billed']:.4f} "
            f"({e['escalated_replay']} escalations, the held-out replay's decisions with the L1/L2 PROXIES standing in "
            f"for the real layers), high USD {e['high_billed']:.4f} (all {e['n_tasks']} tasks escalated at the "
            f"tier's largest measured per-task billed redo USD {e['max_billed_task']:.4f}), upper bound USD "
            f"{e['upper_bound']:.4f} (all tasks at the largest measured all-input-uncached USD "
            f"{e['max_upper_task']:.4f}, at most the USD {R.TASK_CAP_USD:.2f} per-task cap: ceiling USD "
            f"{e['cap_ceiling']:.2f}); job cap USD {e['job_cap']:.2f} billed enforced in code (client spend cap = "
            f"ledger total + {e['job_cap']:.2f}; with the per-task cap, no call is sent whose reservation would pass "
            f"either); hours {e['hours']:.2f} (local turns {e['local_hours']:.2f} h from the B3 v2 run, plus "
            f"{e['n_calls']} calls x 2 x {R.GIN_WAIT_S:.0f} s log-slice waits (render and chat), up to {e['n_checks']} "
            f"prompt checks x {e['prompt_check_s']:.2f} s, {e['escalated_replay']} escalations x {e['redo_s']:.2f} s "
            f"mean redo)")


if __name__ == "__main__":
    run()
    d = load()
    print(breakdown_text(d))
    print(tuned_text(d))
    print(live8192_text(d))
