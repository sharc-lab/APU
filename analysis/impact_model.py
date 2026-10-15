"""Impact model (offline): what each measured problem costs a customer team, in USD per day and per month.

Plan v2 Part A3 (2026-10-13 meeting: "quantify every problem in money and time, and rank the problems by cost").
No machine time, no API call. Every input is a committed file, read fresh by the functions below:

  - customer profile: results/impact_profile_20261014.json. hour_value is the BLS median hourly wage (source URL,
    table, reference period and fetch date are inside the file; register row impact-hour-value). Every other field
    is an ASSUMPTION and is labelled so wherever it is used.
  - cloud prices: results/cloud_prices_20261008.json through analysis/cloud_budget_forecast.py (the same table and
    the same call_cost function as the cloud budget and the Pareto frontier).
  - escalation definitions: src/dse/router.py (Router.decide), listed in ESCALATIONS below. Escalated steps are
    found by replaying R2 sessions through that Router and repriced with the price table above (the Router's own
    estimate uses src/cloud/client.py's older table; see docs/IMPACT.md "Price table").
  - R2 agent sessions: every results/x2_r2_real*.jsonl (evo-x2) and results/blade_r2_real*.jsonl (Blade), read
    through analysis/r2_real_report.py (same dedup and session filter as the register's R2-real-v1b-* rows). A new
    R2 file for a new model or device (plan v2 Part B) is picked up by the glob with no code change.
  - decode rates: numbers_register.compute_decode_rate_per_machine (register decode-rate-per-machine) and the Blade
    dry runs (register blade-dryrun-rates); co-runner ratios: src/dse/envelope_data.json (register
    PX2-full-ratio-table, B3-corunner-6model).
  - single-shot failures: results/x2_outcome_table_v3.jsonl (register x2-v3-usability) and
    results/t2s_outcome_table_full.jsonl (register t2s-outcome-allocation-failures).

Problems (one row each; formulas in ROWS[...]["formula"]): P1 silent wrong results from context loss, P2 oversized-window
time cost, P3 cloud overspend from re-sent history, P4 hardware mismatch, P5 co-runner slowdown, P6 failed starts
and retries, P7 guardrail removal. A cost whose per-event input is not measured is returned as None and printed
"not measured", never as a number. A measured rate of 0 gives 0 regardless of the per-event cost.

Usage: py -3.12 analysis/impact_model.py            prints the profile table
       py -3.12 analysis/impact_model.py --write    also rewrites docs/IMPACT.md's generated block
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "analysis", REPO / "harness"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

PROFILE_FILE = "results/impact_profile_20261014.json"
PRICES_FILE = "results/cloud_prices_20261008.json"
R2_GLOBS = {"evo-x2": "results/x2_r2_real*.jsonl", "blade": "results/blade_r2_real*.jsonl"}
R2_VALIDATION_FILES = ["results/x2_r2_validation_v2.jsonl", "results/x2_r2_validation_v2b.jsonl"]
X2_OUTCOME_V3 = "results/x2_outcome_table_v3.jsonl"
T2S_OUTCOME = "results/t2s_outcome_table_full.jsonl"
ENVELOPE = "src/dse/envelope_data.json"
DEFAULT_TIER = "ollama_default"
NOT_MEASURED = "not measured"

# Escalation definitions: the Router's own decision order (src/dse/router.py, Router.decide). Zach's hybrid demo
# must use these ids and this order for the two models to be comparable (docs/IMPACT.md "Escalation definitions").
ESCALATIONS = {
    "E1": "memory: a measured boundary says the local configuration will not start (A-24 last n_ctx that started) "
          "-> cloud if the budget covers it, else refuse",
    "E2": "fit: system prompt + current turn + answer budget exceed the usable window (num_ctx x (1 - CONTEXT_MARGIN)) even after "
          "trimming every older turn -> cloud, else refuse",
    "E3": "latency: the request fits untrimmed but the predicted latency is over the caller's target -> cloud if "
          "affordable",
    "E4": "quality: trimming whole oldest turns would drop a turn the step needs (needs_turns or an identifier in the "
          "current user text) and quality_floor > 0 (or require_full_history) -> cloud, else local_trimmed with "
          "quality_floor_met=False",
}

PROBLEMS = {
    "P1": "silent wrong results from context loss",
    "P2": "oversized-window time cost",
    "P3": "cloud overspend from re-sent history",
    "P4": "hardware mismatch",
    "P5": "co-runner slowdown",
    "P6": "failed starts and retries",
    "P7": "guardrail removal",
}


# ── profile ─────────────────────────────────────────────────────────────────────────────────────────────────────

def load_profile(repo=REPO, path=None) -> dict:
    return json.loads(((Path(path) if path else repo / PROFILE_FILE)).read_text(encoding="utf-8"))


def hour_value(profile) -> float:
    return float(profile["hour_value"]["usd_per_hour"])


def sessions_per_day(profile) -> float:
    a = profile["assumptions"]
    return a["team_size"] * a["agent_sessions_per_dev_per_day"]


def per_month(usd_day, profile):
    return None if usd_day is None else usd_day * profile["assumptions"]["workdays_per_month"]


def tier_of(runtime: str) -> str:
    """Profile runtime / R2 arm id -> the R2 report's tier label ('ollama_default', 'num_ctx_4096')."""
    s = runtime.replace("_call2_notools", "")
    return s.replace("ollama_ctx_", "num_ctx_")


def tier_ctx(tier: str):
    return int(tier.split("_")[-1]) if tier.startswith("num_ctx_") else None


def norm_model(name: str) -> str:
    from src.dse.router import norm_model as nm
    return nm(name)


# ── pure cost formulas (synthetic-testable) ─────────────────────────────────────────────────────────────────────

def cost_p1(sessions_day, p_silent, redo_hours, hour):
    """sessions/day x P(silent context-loss failure per session) x redo hours x USD/hour (lower bound: the time to
    notice a wrong result and any downstream damage are not measured and not included)."""
    if p_silent is None:
        return None
    if p_silent == 0:
        return 0.0
    if redo_hours is None:
        return None
    return sessions_day * p_silent * redo_hours * hour


def cost_p2(sessions_day, delta_s_per_turn, turns, hour):
    """sessions/day x max(0, per-turn wall delta oversized - smaller window, on turns that fit both) x turns x USD/h."""
    if delta_s_per_turn is None:
        return None
    return sessions_day * max(0.0, delta_s_per_turn) * turns / 3600.0 * hour


def cost_p3(sessions_day, cloud_share, overspend_usd_per_session):
    """sessions/day x cloud share x (session cost re-sending full history uncached - same session with caching)."""
    if overspend_usd_per_session is None:
        return None
    return sessions_day * cloud_share * overspend_usd_per_session


def cost_p4(sessions_day, output_tokens_per_session, rate, best_rate, hour):
    """sessions/day x output tokens x (1/rate on this device - 1/best measured rate for the model) x USD/h (decode
    only; prefill not included, so a lower bound)."""
    if None in (output_tokens_per_session, rate, best_rate) or rate <= 0 or best_rate <= 0:
        return None
    return sessions_day * output_tokens_per_session * max(0.0, 1.0 / rate - 1.0 / best_rate) / 3600.0 * hour


def cost_p5(sessions_day, share, prefill_s, decode_s, ttft_ratio, decode_ratio, hour):
    """sessions/day x co-runner share x (prefill s x (TTFT ratio - 1) + decode s x (1/decode ratio - 1)) x USD/h."""
    if None in (prefill_s, decode_s, ttft_ratio):
        return None
    dr = decode_ratio or 1.0
    extra = prefill_s * max(0.0, ttft_ratio - 1.0) + decode_s * max(0.0, 1.0 / dr - 1.0)
    return sessions_day * share * extra / 3600.0 * hour


def cost_p6(calls_day, p_timeout, timeout_s, p_fail, fail_s, hour):
    """calls/day x (P(timeout) x timeout s + P(failed start) x seconds to fail) x USD/h (the retry itself is not
    counted: on evo-t2s no call succeeded after the first allocation failure, so a retry does not recover)."""
    if p_timeout is None and p_fail is None:
        return None
    lost = (p_timeout or 0.0) * (timeout_s or 0.0) + (p_fail or 0.0) * (fail_s or 0.0)
    return calls_day * lost / 3600.0 * hour


def cost_p7(sessions_day, p_rule_loss, usd_per_violation):
    """sessions/day x P(guardrail rule broken after the window was exceeded) x cost per violation (not measured)."""
    if p_rule_loss is None:
        return None
    if p_rule_loss == 0:
        return 0.0
    if usd_per_violation is None:
        return None
    return sessions_day * p_rule_loss * usd_per_violation


def rank(rows: dict) -> list:
    """Problems by profile USD/day, highest first; rows whose USD/day is None go last as unranked."""
    known = sorted((k for k, r in rows.items() if r["usd_day"] is not None), key=lambda k: -rows[k]["usd_day"])
    return known + sorted(k for k, r in rows.items() if r["usd_day"] is None)


# ── R2 sessions ─────────────────────────────────────────────────────────────────────────────────────────────────

def _has_turns(path: Path) -> bool:
    """A file with no r2a_turn rows (e.g. a run refused at its validation gate) carries no session to read."""
    with open(path, encoding="utf-8") as f:
        return any('"record": "r2a_turn"' in line for line in f)


def r2_files(repo=REPO) -> dict:
    return {dev: [p for p in sorted(repo.glob(g)) if _has_turns(p)] for dev, g in R2_GLOBS.items()}


@lru_cache(maxsize=4)
def _r2_cached(repo_str: str):
    import r2_real_report as rr
    repo = Path(repo_str)
    out = {}
    for dev, paths in r2_files(repo).items():
        if not paths:
            continue
        rows, _riu, sessions, _iss = rr._collect(paths)
        if not sessions:
            continue
        keep = {s["attempt_id"] for s in sessions}
        turns = {}
        for r in rows:
            if r.get("record") == "r2a_turn" and r.get("attempt_id") in keep:
                turns.setdefault(r["attempt_id"], []).append(r)
        sess = []
        for s in sessions:
            t = sorted(turns.get(s["attempt_id"], []), key=lambda r: r["turn_idx"])
            if t:
                sess.append({"device": dev, "model": s["model_id"], "tier": rr.tier_of(s["arm_id"]),
                             "seed": s["seed"], "attempt_id": s["attempt_id"], "turns": t})
        val = [repo / f for f in R2_VALIDATION_FILES if (repo / f).exists()]
        rep = rr.build_combined(paths, validation_paths=val) if dev == "evo-x2" else None
        out[dev] = {"sessions": sess, "report": rep, "files": [p.name for p in paths]}
    return out


def r2_data(repo=REPO) -> dict:
    return _r2_cached(str(repo))


def session_stats(turns: list) -> dict:
    """One session's wall, prefill and decode seconds and output tokens, from analysis/r2_session_cost.session_metrics
    (the per-session function behind register row r2-session-wallclock: wall = sum of turn_wall_s, output = sum of
    completion tokens, chars/4 of the reply when a call's completion_tokens is null)."""
    import r2_session_cost as sc
    m = sc.session_metrics(turns)
    return {"wall_s": m["wall_s"], "prefill_s": m["prefill_s"], "decode_s": m["decode_s"],
            "output_tokens": m["output"], "n_turns": m["n_turns"]}


def session_wallclock(repo=REPO) -> dict:
    """{(device, model, tier): {"median_wall_s", "mean_prefill_s", "mean_decode_s", "mean_output_tokens", "n"}}.
    Per-session numbers are register row r2-session-wallclock's (session_stats above); the sessions are this model's
    own R2 set (r2_data: every R2 file of every device, r2_real_report's dedup), and the cell statistic is the median
    wall (the register row reports the mean)."""
    out = {}
    for dev, d in r2_data(repo).items():
        by = {}
        for s in d["sessions"]:
            by.setdefault((dev, s["model"], s["tier"]), []).append(session_stats(s["turns"]))
        for k, v in by.items():
            out[k] = {"median_wall_s": statistics.median(x["wall_s"] for x in v),
                      "mean_prefill_s": statistics.mean(x["prefill_s"] for x in v),
                      "mean_decode_s": statistics.mean(x["decode_s"] for x in v),
                      "mean_output_tokens": statistics.mean(x["output_tokens"] for x in v),
                      "mean_turns": statistics.mean(x["n_turns"] for x in v), "n": len(v)}
    return out


def context_loss_rates(repo=REPO) -> dict:
    """{(model, tier): {"n", "silent_ctx", "rule_after_window"}} on evo-x2 from the R2 report's gated sessions:
    silent_ctx = gated silent failure at or after the first turn over the loaded window (register R2-real-v1b-gated-kill
    "after window exceeded"); rule_after_window = a rule in use first failing at or after that turn."""
    d = r2_data(repo).get("evo-x2")
    out = {}
    if not d or not d["report"]:
        return out
    for key, c in d["report"]["table"].items():
        tier, model = key.split("|")
        rules = c["rules_in_use"]
        rule_after = 0
        for e in c["sessions"]:
            ow = e.get("first_over_window")
            fails = [e.get(f"first_fail_{r}") for r in rules if e.get(f"first_fail_{r}") is not None]
            if ow is not None and fails and min(fails) >= ow:
                rule_after += 1
        out[(model, tier)] = {"n": c["n"], "silent_ctx": sum(1 for e in c["sessions"] if e["silent"]
                                                             and e["after_window_exceeded"]),
                              "rule_after_window": rule_after}
    return out


def oversize_turn_delta(repo=REPO) -> dict:
    """{(model, tier): {"vs", "median_delta_s", "n_turns"}}: per-turn wall time of `tier` minus the next smaller
    measured tier, same model and seed, on turns where neither row was over its loaded window (so both sent the same
    history; only the window differs). The smallest tier has no smaller one (vs None)."""
    d = r2_data(repo).get("evo-x2")
    out = {}
    if not d:
        return out
    by = {}
    for s in d["sessions"]:
        by.setdefault((s["model"], s["tier"], s["seed"]), {r["turn_idx"]: r for r in s["turns"]})
    models = sorted({k[0] for k in by})
    for m in models:
        tiers = sorted({k[1] for k in by if k[0] == m}, key=lambda t: -(tier_ctx(t) or 10**9))
        for i, t in enumerate(tiers):
            if i + 1 >= len(tiers):
                out[(m, t)] = {"vs": None, "median_delta_s": 0.0, "n_turns": 0}
                continue
            smaller = tiers[i + 1]
            deltas = []
            for (mm, tt, seed), turns in by.items():
                if mm != m or tt != t or (m, smaller, seed) not in by:
                    continue
                other = by[(m, smaller, seed)]
                for ti, r in turns.items():
                    o = other.get(ti)
                    if o and not r.get("over_loaded_window") and not o.get("over_loaded_window"):
                        deltas.append((r.get("turn_wall_s") or 0.0) - (o.get("turn_wall_s") or 0.0))
            out[(m, t)] = {"vs": smaller, "median_delta_s": statistics.median(deltas) if deltas else None,
                           "n_turns": len(deltas)}
    return out


# ── cloud (P3, escalation pricing) ──────────────────────────────────────────────────────────────────────────────

def _cbf():
    import cloud_budget_forecast as cbf
    return cbf


def prices(repo=REPO):
    return _cbf().load_prices(repo)


def session_cloud_cost(calls, model, tier, prices_, caching) -> float:
    """USD for one session's calls [(prompt_tokens, output_tokens)] at the price table (cloud_budget_forecast)."""
    return _cbf().session_cost(prices_, calls, model, tier, caching)


def r2_cloud_overspend(repo=REPO, profile=None) -> dict:
    """{model: {"uncached", "cached", "overspend", "n"}} USD per session for the ollama_default R2 sessions of each
    model, prompts converted to o200k_base with cloud_budget_forecast's per-seed ratio, outputs re-tokenized from the
    recorded replies. Input of register row r2-session-cloud-cost (parallel plan item)."""
    profile = profile or load_profile(repo)
    a = profile["assumptions"]
    cbf = _cbf()
    pr = prices(repo)
    by = {}
    for dev, d in r2_data(repo).items():
        for s in d["sessions"]:
            if s["tier"] != DEFAULT_TIER:
                continue
            ratio = cbf.o200k_per_est_ratio(s["seed"])
            calls = [(round(c["prompt_tokens_est"] * ratio), cbf.r2_call_output_tokens(c))
                     for r in s["turns"] for c in r["calls"]]
            by.setdefault(s["model"], []).append(
                (session_cloud_cost(calls, a["cloud_model"], a["cloud_tier"], pr, False),
                 session_cloud_cost(calls, a["cloud_model"], a["cloud_tier"], pr, True)))
    return {m: {"uncached": statistics.mean(u for u, _ in v), "cached": statistics.mean(c for _, c in v),
                "overspend": statistics.mean(u - c for u, c in v), "n": len(v)} for m, v in by.items()}


def router_escalation(repo=REPO, model=None, tier=None, profile=None, quality_floor=0.9, max_sessions=None) -> dict:
    """Replay the evo-x2 R2 sessions of (model, tier) through src/dse/router.py's Router (definitions E1-E4) and
    price each cloud decision with the price table: input = system + full history, output = the step's num_predict
    (the Router's own answer budget, an upper bound), no caching. Token counts are the R2 harness's calibrated
    local-tokenizer estimate. Returns {"cloud_steps_per_session", "usd_per_session", "steps_per_session", "n"}."""
    profile = profile or load_profile(repo)
    a = profile["assumptions"]
    from src.dse import r2_replay as rp
    pr = prices(repo)
    cbf = _cbf()
    d = r2_data(repo).get("evo-x2")
    sess = [s for s in (d["sessions"] if d else []) if s["model"] == model and s["tier"] == tier]
    if max_sessions:
        sess = sess[:max_sessions]
    if not sess:
        return {"cloud_steps_per_session": None, "usd_per_session": None, "steps_per_session": None, "n": 0}
    clouds, usds, steps = [], [], []
    for s in sess:
        router = rp.router_for_session(s["turns"], budget_usd=1e6, quality_floor=quality_floor)
        n_cloud, usd, n = 0, 0.0, 0
        for step in rp.session_steps(s["turns"]):
            dec = router.decide(step)
            n += 1
            if dec.target == "cloud":
                n_cloud += 1
                t = dec.tokens
                usd += cbf.call_cost(pr, a["cloud_model"], a["cloud_tier"], t["full_total"] - t["answer_budget"],
                                     t["answer_budget"])
        clouds.append(n_cloud)
        usds.append(usd)
        steps.append(n)
    return {"cloud_steps_per_session": statistics.mean(clouds), "usd_per_session": statistics.mean(usds),
            "steps_per_session": statistics.mean(steps), "n": len(sess)}


# ── rates and ratios (P4, P5, P6) ───────────────────────────────────────────────────────────────────────────────

def decode_rates(repo=REPO) -> dict:
    """{device: {norm_model: tok/s}}: evo-t2s / evo-x2 from decode-rate-per-machine, the Blade per window from the
    Blade dry runs (device 'blade@<num_ctx>'; median over dry-run files)."""
    import numbers_register as nr
    out = {}
    for hw, ms in nr.compute_decode_rate_per_machine(repo)["detail"].items():
        out[hw] = {norm_model(m): v["median"] for m, v in ms.items()}
    try:
        est = nr._blade_estimator(repo)
        per = {}
        for _f, dd in est.dryrun_rates(nr.BLADE_DRYRUN_FILES).items():
            for key, v in (dd or {}).items():
                if v.get("decode_tps"):
                    m, ctx = key.split("@")
                    per.setdefault((f"blade@{ctx}", norm_model(m)), []).append(v["decode_tps"])
        for (dev, m), v in per.items():
            out.setdefault(dev, {})[m] = statistics.median(v)
    except FileNotFoundError:
        pass
    return out


def co_runner_ratios(repo=REPO) -> dict:
    env = json.loads((repo / ENVELOPE).read_text(encoding="utf-8"))
    return {hw: {"claim_id": (m.get("co_runner") or {}).get("claim_id"),
                 "ttft": (m.get("co_runner") or {}).get("ttft_ratio", {}),
                 "decode": (m.get("co_runner") or {}).get("decode_ratio", {})} for hw, m in env["machines"].items()}


def single_shot_failures(repo=REPO, profile=None) -> dict:
    """{(device, norm_model, config): {"p_timeout", "timeout_s", "p_fail", "fail_s", "n", "escalation_usd_per_call"}}.
    evo-x2: timeout_latency rows of the v3 outcome table (register x2-v3-usability; time lost = the harness per-call
    timeout). evo-t2s: allocation-failure rows (numbers_register._t2s_alloc_kind) of the T2S outcome table (register
    t2s-outcome-allocation-failures; time lost = the failed call's own latency). escalation_usd_per_call = P(event) x
    mean price-table cost of the failing items (input = their prompt tokens, output = 0: not measured for them)."""
    profile = profile or load_profile(repo)
    a = profile["assumptions"]
    import numbers_register as nr
    pr = prices(repo)
    cbf = _cbf()
    out = {}
    x2, cells, items, _td, _tp = nr._x2_v3_timeout_latency(repo)
    for (m, c), (n_done, tl) in cells.items():
        esc = [cbf.call_cost(pr, a["cloud_model"], a["cloud_tier"], items.get(r["item_id"], {}).get("prompt_tokens", 0), 0)
               for r in tl]
        p = len(tl) / n_done if n_done else None
        out[("evo-x2", norm_model(m), c)] = {
            "p_timeout": p, "timeout_s": x2.DEFAULT_CALL_TIMEOUT_S, "p_fail": 0.0, "fail_s": None, "n": n_done,
            "n_events": len(tl), "escalation_usd_per_call": (p * statistics.mean(esc)) if (p and esc) else 0.0}
    path = repo / T2S_OUTCOME
    if path.exists():
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        by = {}
        for r in rows:
            if r.get("record") == "outcome_row":
                by.setdefault((norm_model(r["model_id"]), r["config"]), []).append(r)
        for (m, c), rs in by.items():
            fails = [r for r in rs if nr._t2s_alloc_kind(r.get("error"))]  # the register row's own classifier
            p = len(fails) / len(rs)
            esc = [cbf.call_cost(pr, a["cloud_model"], a["cloud_tier"], r.get("sent_tokens") or 0, 0) for r in fails]
            out[("evo-t2s", m, c)] = {
                "p_timeout": 0.0, "timeout_s": None, "p_fail": p,
                "fail_s": statistics.median(r["latency_s"] for r in fails) if fails else None, "n": len(rs),
                "n_events": len(fails), "escalation_usd_per_call": (p * statistics.mean(esc)) if esc else 0.0}
    return out


# ── the seven rows ──────────────────────────────────────────────────────────────────────────────────────────────

def _cfg(profile, runtime=None):
    a = profile["assumptions"]
    return a["device"], a["model"], tier_of(runtime or a["runtime"])


def p1(repo=REPO, profile=None, runtime=None, inputs=None):
    profile = profile or load_profile(repo)
    dev, model, tier = _cfg(profile, runtime)
    inputs = inputs or {}
    rates = inputs.get("rates") or context_loss_rates(repo)
    wall = inputs.get("wall") or session_wallclock(repo)
    hv, spd = hour_value(profile), sessions_per_day(profile)
    grid = {}
    for (m, t), v in sorted(rates.items()):
        w = wall.get(("evo-x2", m, DEFAULT_TIER))
        p = v["silent_ctx"] / v["n"] if v["n"] else None
        grid[f"{m}@{t}"] = {"p": p, "k": v["silent_ctx"], "n": v["n"],
                            "redo_h": None if not w else w["median_wall_s"] / 3600.0,
                            "usd_day": cost_p1(spd, p, None if not w else w["median_wall_s"] / 3600.0, hv)}
    cell = grid.get(f"{model}@{tier}") if dev == "evo-x2" else None
    return {"id": "P1", "usd_day": None if cell is None else cell["usd_day"], "cell": cell, "grid": grid,
            "config": f"{dev} {model} {tier}"}


def p2(repo=REPO, profile=None, runtime=None, inputs=None):
    profile = profile or load_profile(repo)
    dev, model, tier = _cfg(profile, runtime)
    inputs = inputs or {}
    deltas = inputs.get("deltas") or oversize_turn_delta(repo)
    hv, spd, turns = hour_value(profile), sessions_per_day(profile), profile["assumptions"]["turns_per_session"]
    grid = {f"{m}@{t}": dict(v, usd_day=cost_p2(spd, v["median_delta_s"], turns, hv)) for (m, t), v in sorted(deltas.items())}
    cell = grid.get(f"{model}@{tier}") if dev == "evo-x2" else None
    return {"id": "P2", "usd_day": None if cell is None else cell["usd_day"], "cell": cell, "grid": grid,
            "config": f"{dev} {model} {tier}"}


def p3(repo=REPO, profile=None, runtime=None, inputs=None):
    profile = profile or load_profile(repo)
    a = profile["assumptions"]
    inputs = inputs or {}
    over = inputs.get("overspend") or r2_cloud_overspend(repo, profile)
    spd = sessions_per_day(profile)
    grid = {m: dict(v, usd_day=cost_p3(spd, a["cloud_session_share"], v["overspend"])) for m, v in sorted(over.items())}
    cell = grid.get(a["model"])
    return {"id": "P3", "usd_day": None if cell is None else cell["usd_day"], "cell": cell, "grid": grid,
            "config": f"{a['cloud_model']} {a['cloud_tier']}, sessions shaped like {a['model']}'s"}


def p4(repo=REPO, profile=None, runtime=None, inputs=None):
    profile = profile or load_profile(repo)
    dev, model, _tier = _cfg(profile, runtime)
    inputs = inputs or {}
    rates = inputs.get("rates") or decode_rates(repo)
    wall = inputs.get("wall") or session_wallclock(repo)
    hv, spd = hour_value(profile), sessions_per_day(profile)
    out_tok = {norm_model(m): v["mean_output_tokens"] for (d, m, t), v in wall.items() if t == DEFAULT_TIER}
    grid = {}
    models = sorted({m for ms in rates.values() for m in ms})
    for m in models:
        best = max((ms[m] for ms in rates.values() if m in ms), default=None)
        for d, ms in sorted(rates.items()):
            if m in ms:
                grid[f"{m}@{d}"] = {"rate": ms[m], "best_rate": best, "output_tokens": out_tok.get(m),
                                    "usd_day": cost_p4(spd, out_tok.get(m), ms[m], best, hv)}
    cell = grid.get(f"{norm_model(model)}@{dev}")
    return {"id": "P4", "usd_day": None if cell is None else cell["usd_day"], "cell": cell, "grid": grid,
            "config": f"{dev} {model}"}


def p5(repo=REPO, profile=None, runtime=None, inputs=None):
    profile = profile or load_profile(repo)
    dev, model, tier = _cfg(profile, runtime)
    inputs = inputs or {}
    ratios = inputs.get("ratios") or co_runner_ratios(repo)
    wall = inputs.get("wall") or session_wallclock(repo)
    hv, spd, share = hour_value(profile), sessions_per_day(profile), profile["assumptions"]["co_runner_share"]
    grid = {}
    for (d, m, t), w in sorted(wall.items()):
        r = ratios.get(d, {})
        tr, dr = r.get("ttft", {}).get(norm_model(m)), r.get("decode", {}).get(norm_model(m))
        grid[f"{m}@{d}@{t}"] = {"ttft_ratio": tr, "decode_ratio": dr, "prefill_s": w["mean_prefill_s"],
                                "decode_s": w["mean_decode_s"], "claim_id": r.get("claim_id"),
                                "usd_day": cost_p5(spd, share, w["mean_prefill_s"], w["mean_decode_s"], tr, dr, hv)}
    for hw in ratios:  # devices with ratios but no measured session timings
        if not any(k.split("@")[1] == hw for k in grid):
            grid[f"{model}@{hw}@{tier}"] = {"ttft_ratio": ratios[hw]["ttft"].get(norm_model(model)), "usd_day": None,
                                           "claim_id": ratios[hw]["claim_id"],
                                           "missing": f"no R2 session timings on {hw}"}
    cell = grid.get(f"{model}@{dev}@{tier}")
    return {"id": "P5", "usd_day": None if cell is None else cell["usd_day"], "cell": cell, "grid": grid,
            "config": f"{dev} {model} {tier}"}


def _single_shot_config(tier):
    return "ollama_default" if tier == DEFAULT_TIER else None


def p6(repo=REPO, profile=None, runtime=None, inputs=None):
    profile = profile or load_profile(repo)
    dev, model, tier = _cfg(profile, runtime)
    a = profile["assumptions"]
    inputs = inputs or {}
    ss = inputs.get("single_shot") or single_shot_failures(repo, profile)
    hv = hour_value(profile)
    calls = a["team_size"] * a["single_shot_calls_per_dev_per_day"]
    grid = {}
    for (d, m, c), v in sorted(ss.items()):
        grid[f"{m}@{d}@{c}"] = dict(v, usd_day=cost_p6(calls, v["p_timeout"], v["timeout_s"], v["p_fail"],
                                                      v["fail_s"], hv),
                                    escalation_usd_day=calls * v["escalation_usd_per_call"])
    cfg = _single_shot_config(tier)
    cell = grid.get(f"{norm_model(model)}@{dev}@{cfg}") if cfg else None
    return {"id": "P6", "usd_day": None if cell is None else cell["usd_day"], "cell": cell, "grid": grid,
            "config": f"{dev} {model} {cfg or tier + ' (no single-shot table at this window)'}"}


def p7(repo=REPO, profile=None, runtime=None, inputs=None):
    profile = profile or load_profile(repo)
    dev, model, tier = _cfg(profile, runtime)
    inputs = inputs or {}
    rates = inputs.get("rates") or context_loss_rates(repo)
    spd = sessions_per_day(profile)
    grid = {}
    for (m, t), v in sorted(rates.items()):
        p = v["rule_after_window"] / v["n"] if v["n"] else None
        grid[f"{m}@{t}"] = {"p": p, "k": v["rule_after_window"], "n": v["n"], "usd_day": cost_p7(spd, p, None)}
    cell = grid.get(f"{model}@{tier}") if dev == "evo-x2" else None
    return {"id": "P7", "usd_day": None if cell is None else cell["usd_day"], "cell": cell, "grid": grid,
            "config": f"{dev} {model} {tier}"}


ROW_FUNCS = {"P1": p1, "P2": p2, "P3": p3, "P4": p4, "P5": p5, "P6": p6, "P7": p7}

# Static parts of each row: formula, measured inputs (register ids), missing inputs, fix, escalation.
ROWS = {
    "P1": {"formula": "sessions/day x P(gated silent failure at/after the first over-window turn) x redo hours (median "
                      "session wall time at Ollama's default window) x USD/h; lower bound",
           "measured": ["R2-real-v1b-gated-kill", "R2-real-v1b-survival", "R2-mechanism-verdict",
                        "x2-v3-truncation-all-models", "r2-session-wallclock",
                        "info-loss-by-window"],
           "missing": "developer time to notice a silent wrong result and redo it, and the cost of one that is not "
                      "noticed (not measured; the redo-time lower bound assumes the developer waits for one rerun); "
                      "R2 sessions on evo-t2s and the Blade",
           "fix": "client-side trimming of whole oldest turns to the usable window (never send an overflowing request), "
                  "escalate on E4",
           "escalation": "yes when a needed turn would be trimmed (E4); cost = Router replay, see the escalation table"},
    "P2": {"formula": "sessions/day x max(0, median per-turn wall delta between the configured window and the next smaller "
                      "measured window, on turns that fit both) x turns/session x USD/h",
           "measured": ["r2-session-wallclock", "R2-real-v1-loaded-ctx",
                        "R2-real-v1b-first-events"],
           "missing": "a right-sized window for a full R2 session (every smaller R2 tier is outgrown before the last "
                      "turn, R2-real-v1b-first-events); session timings on evo-t2s and the Blade",
           "fix": "set num_ctx to the session's need, not the runtime default; combined with the P1 fix so a smaller "
                  "window cannot silently cut history",
           "escalation": "no for the window change itself; E4 escalations of the P1 fix apply at the smaller window"},
    "P3": {"formula": "sessions/day x cloud share x mean(session cost uncached - session cost with implicit caching), "
                      "price table cloud_prices_20261008, R2 session token counts",
           "measured": ["cloud-budget-tokens", "cloud-budget-inputs",
                        "r2-session-cloud-cost (chars/4 basis, caching floor; this row: o200k, implicit caching)"],
           "missing": "how much of a real customer's cloud history is a stable cacheable prefix (R2 sessions are "
                      "append-only, the best case for caching)",
           "fix": "keep the history an append-only prefix so implicit caching applies; trim history client-side",
           "escalation": "no (the fix lowers cloud spend)"},
    "P4": {"formula": "sessions/day x output tokens/session x (1/decode rate on this device - 1/best measured rate for "
                      "the model) x USD/h; decode only (prefill excluded), lower bound",
           "measured": ["decode-rate-per-machine", "blade-dryrun-rates", "blade-k1-layer-split",
                        "r2-session-wallclock (output tokens)"],
           "missing": "session prefill time on evo-t2s and the Blade; the XPS (Zach's) rates are not in this repo",
           "fix": "route each model to the device whose envelope fits it (src/dse/pareto.py recommend)",
           "escalation": "no between local devices; E1/E2 only when no local device fits"},
    "P5": {"formula": "sessions/day x co-runner share x (session prefill s x (TTFT ratio - 1) + session decode s x "
                      "(1/decode ratio - 1)) x USD/h",
           "measured": ["PX2-full-ratio-table", "B3-corunner-6model", "PX2-decode-ratio",
                        "r2-session-wallclock"],
           "missing": "R2 session timings on evo-t2s (B3 is TTFT only); how often customers run a heavy co-runner "
                      "(assumed share)",
           "fix": "co-runner-aware latency prediction in the router (co_runner flag)",
           "escalation": "only under a latency target (E3); cost depends on the customer's target, not modelled"},
    "P6": {"formula": "single-shot calls/day x (P(timeout) x per-call timeout s + P(allocation failure) x seconds to "
                      "fail) x USD/h",
           "measured": ["x2-v3-usability", "t2s-outcome-allocation-failures", "t2s-llamaserver-load-failures",
                        "A-24-flip-point-prediction"],
           "missing": "operator time to notice and restart a stuck runtime (not measured); llama-server start "
                      "failures need the gitignored logs (register row only)",
           "fix": "pre-check the memory fit (A-24 shared-heap model) and the latency before sending; route the rest",
           "escalation": "yes for calls that cannot fit or finish locally (E1/E3); cost = P(event) x price of those "
                         "items"},
    "P7": {"formula": "sessions/day x P(a rule in use first broken at/after the first over-window turn) x cost per "
                      "violation",
           "measured": ["R2-real-v1b-first-events", "R2-mechanism-lowlevel", "R2-orphan-rates",
                        "R2-mitigation-v1-verdict", "reliability-after-guardrail-loss"],
           "missing": "cost of one guardrail violation (not measured; depends on what the rule protects)",
           "fix": "never send an overflowing request (the system prompt is always kept by client-side trimming)",
           "escalation": "no for system-prompt rules (the router never trims the system prompt)"},
}

CONFIDENCE = {
    "P1": ("low", "the per-session rate is measured on few sessions per cell (n in the grid), but the cost per event is "
                  "a rerun-time lower bound; detection and rework time are not measured"),
    "P2": ("medium", "paired same-seed turns that fit both windows isolate the window size, but the per-turn delta is "
                     "scaled to a whole session by assumption and the right-sized window was not run"),
    "P3": ("high", "token counts and public prices only; the daily figure scales linearly with the assumed cloud share"),
    "P4": ("medium", "decode rates are measured per machine, but on different workloads (the Blade's from short dry "
                     "runs), and prefill is excluded, so it is a lower bound"),
    "P5": ("low", "co-runner ratios come from single-shot microbenchmarks applied to session timings, and the co-runner "
                  "share is an assumption"),
    "P6": ("medium", "rates are measured on the workload pack; the daily volume of single-shot calls is an assumption"),
    "P7": ("low", "the rate is measured but the cost per violation is not; a nonzero rate prints not measured"),
}

DECISIVE_MEASUREMENT = (
    "Developer time lost per silent wrong result (time to notice plus rework, and how often it is never noticed), "
    "measured on real agent sessions. It is the one input P1 and P7 lack: today P1 is priced at a rerun-time lower "
    "bound and P7 is not priced at all, and at any small window both rates are high (R2-real-v1b-gated-kill), so a "
    "measured per-event cost decides whether context loss ranks first or last.")


def _fmt_usd(v):
    return NOT_MEASURED if v is None else f"{v:,.2f}"


def model_rows(repo=REPO, profile=None, runtime=None, inputs=None) -> dict:
    profile = profile or load_profile(repo)
    out = {}
    for k, f in ROW_FUNCS.items():
        r = f(repo, profile, runtime, (inputs or {}).get(k))
        r["usd_month"] = per_month(r["usd_day"], profile)
        r["confidence"] = CONFIDENCE[k]
        out[k] = r
    return out


def total(rows: dict) -> dict:
    known = {k: r["usd_day"] for k, r in rows.items() if r["usd_day"] is not None}
    return {"usd_day": sum(known.values()), "included": sorted(known),
            "excluded_not_measured": sorted(k for k, r in rows.items() if r["usd_day"] is None)}


def row_value(r: dict, profile) -> str:
    """One register value string for a problem row (profile config)."""
    k = r["id"]
    measured = "measured" if r["usd_day"] is not None else NOT_MEASURED
    c = r["cell"] or {}
    detail = ""
    if k in ("P1", "P7") and c:
        detail = f"; rate {c['k']}/{c['n']} sessions"
    elif k == "P2" and c:
        detail = (f"; median per-turn delta {c['median_delta_s']:.2f} s vs {c['vs']} (n={c['n_turns']} paired turns)"
                  if c.get("vs") else "; smallest measured window")
    elif k == "P3" and c:
        detail = f"; overspend USD {c['overspend']:.4f}/session (uncached {c['uncached']:.4f}, cached {c['cached']:.4f}, n={c['n']})"
    elif k == "P4" and c:
        detail = f"; decode {c['rate']:.2f} tok/s vs best {c['best_rate']:.2f}"
    elif k == "P5" and c and c.get("ttft_ratio"):
        detail = f"; TTFT x{c['ttft_ratio']:.3f}, decode x{(c.get('decode_ratio') or 1):.3f} ({c['claim_id']})"
    elif k == "P6" and c:
        detail = (f"; timeouts {c['p_timeout']:.3f} x {c['timeout_s']} s, allocation failures {c['p_fail']:.3f} "
                  f"(n={c['n']}); escalation USD {c['escalation_usd_day']:.2f}/day")
    return (f"{r['config']}: USD/day {_fmt_usd(r['usd_day'])}, USD/month {_fmt_usd(r['usd_month'])} ({measured}){detail}; "
            f"confidence {r['confidence'][0]}")


def compute_all(repo=REPO, profile=None) -> dict:
    profile = profile or load_profile(repo)
    base = model_rows(repo, profile)
    scen = model_rows(repo, profile, runtime=profile["assumptions"]["scenario_runtime"])
    return {"profile": profile, "base": base, "scenario": scen, "total": total(base), "scenario_total": total(scen),
            "ranking": rank(base), "scenario_ranking": rank(scen)}


# ── docs/IMPACT.md generated block ──────────────────────────────────────────────────────────────────────────────

BEGIN, END = "<!-- impact_model:begin -->", "<!-- impact_model:end -->"


def _grid_lines(k, r):
    lines = []
    for key, v in r["grid"].items():
        u = _fmt_usd(v.get("usd_day"))
        extra = ""
        if k in ("P1", "P7"):
            extra = f"{v['k']}/{v['n']} sessions"
            if k == "P1":
                extra += f", redo {'n/a' if v['redo_h'] is None else f'{v['redo_h']:.2f} h'}"
        elif k == "P2":
            extra = (f"vs {v['vs']}: {v['median_delta_s']:.2f} s/turn (n={v['n_turns']})" if v.get("vs") and
                     v.get("median_delta_s") is not None else "smallest measured window")
        elif k == "P3":
            extra = f"uncached {v['uncached']:.4f}, cached {v['cached']:.4f} USD/session (n={v['n']})"
        elif k == "P4":
            extra = (f"{v['rate']:.2f} tok/s vs best {v['best_rate']:.2f}; output "
                     f"{'n/a (no R2 session)' if v['output_tokens'] is None else f'{v['output_tokens']:.0f}'} tok/session")
        elif k == "P5":
            extra = v.get("missing") or (f"TTFT x{v['ttft_ratio']:.3f}" if v.get("ttft_ratio") else "no ratio")
        elif k == "P6":
            extra = (f"timeouts {v['n_events'] if v['p_timeout'] else 0}, allocation failures "
                     f"{v['n_events'] if v['p_fail'] else 0} of {v['n']}; escalation USD {v['escalation_usd_day']:.2f}/day")
        lines.append(f"| {k} | {key} | {u} | {extra} |")
    return lines


def render_block(res: dict, escalation: dict) -> str:
    p = res["profile"]
    a = p["assumptions"]
    out = [BEGIN, "",
           f"Generated by `analysis/impact_model.py --write`. Profile: hour value USD {hour_value(p):.2f} "
           f"(impact-hour-value); ASSUMPTIONS: team {a['team_size']}, {a['agent_sessions_per_dev_per_day']} agent "
           f"sessions per developer per day, {a['turns_per_session']} turns per session, "
           f"{a['single_shot_calls_per_dev_per_day']} single-shot calls per developer per day, "
           f"{a['workdays_per_month']} workdays per month, co-runner share {a['co_runner_share']}, cloud session share "
           f"{a['cloud_session_share']}, configuration {a['device']} {a['model']} {a['runtime']}, escalation model "
           f"{a['cloud_model']} ({a['cloud_tier']}).", "",
           "### Profile table (status quo: Ollama default window)", "",
           "| row | problem | USD/day | USD/month | measured | confidence | register row |", "|---|---|---:|---:|---|---|---|"]
    for k in ROW_FUNCS:
        r = res["base"][k]
        out.append(f"| {k} | {PROBLEMS[k]} | {_fmt_usd(r['usd_day'])} | {_fmt_usd(r['usd_month'])} | "
                   f"{'measured' if r['usd_day'] is not None else NOT_MEASURED} | {r['confidence'][0]} | impact-{k} |")
    t = res["total"]
    out.append(f"| total | measured rows only ({', '.join(t['included'])}) | {_fmt_usd(t['usd_day'])} | "
               f"{_fmt_usd(per_month(t['usd_day'], p))} | excluded: {', '.join(t['excluded_not_measured']) or 'none'} | "
               f"| impact-total |")
    out += ["", f"### Same team after one flag: `{a['scenario_runtime']}`", "",
            "| row | USD/day | USD/month | measured |", "|---|---:|---:|---|"]
    for k in ROW_FUNCS:
        r = res["scenario"][k]
        out.append(f"| {k} | {_fmt_usd(r['usd_day'])} | {_fmt_usd(r['usd_month'])} | "
                   f"{'measured' if r['usd_day'] is not None else NOT_MEASURED} |")
    st = res["scenario_total"]
    out.append(f"| total (measured rows) | {_fmt_usd(st['usd_day'])} | {_fmt_usd(per_month(st['usd_day'], p))} | "
               f"excluded: {', '.join(st['excluded_not_measured']) or 'none'} |")
    out += ["", "### Rows: formula, measured inputs, missing inputs, fix, escalation", "",
            "| row | formula | measured inputs (register ids) | missing inputs | fix | needs cloud escalation? |",
            "|---|---|---|---|---|---|"]
    for k, d in ROWS.items():
        out.append(f"| {k} | {d['formula']} | {', '.join(d['measured'])} | {d['missing']} | {d['fix']} | "
                   f"{d['escalation']} |")
    out += ["", "### Escalation definitions (src/dse/router.py Router.decide, in decision order)", ""]
    out += [f"- {k}: {v}" for k, v in ESCALATIONS.items()]
    out += ["", "### Ranking (status quo profile)", ""]
    for i, k in enumerate(res["ranking"], 1):
        r = res["base"][k]
        out.append(f"{i}. {k} {PROBLEMS[k]}: USD/day {_fmt_usd(r['usd_day'])}; confidence {r['confidence'][0]} "
                   f"({r['confidence'][1]})")
    out += ["", f"Single most decisive missing measurement: {DECISIVE_MEASUREMENT}"]
    out += ["", "### Escalation cost of the P1 fix (Router replay, E1-E4, priced at the price table)", "",
            "| model | window | cloud steps / session | steps / session | USD / session | sessions |", "|---|---|---:|---:|---:|---:|"]
    for (m, t), e in sorted(escalation.items()):
        out.append(f"| {m} | {t} | {'n/a' if e['cloud_steps_per_session'] is None else f'{e['cloud_steps_per_session']:.1f}'} | "
                   f"{'n/a' if e['steps_per_session'] is None else f'{e['steps_per_session']:.1f}'} | "
                   f"{_fmt_usd(e['usd_per_session']) if e['usd_per_session'] is None else f'{e['usd_per_session']:.4f}'} | {e['n']} |")
    out += ["", "### Per model and device grid (USD/day at the profile volumes)", "",
            "| row | model@device/window | USD/day | inputs |", "|---|---|---:|---|"]
    for k in ROW_FUNCS:
        out += _grid_lines(k, res["base"][k])
    out += ["", END]
    return "\n".join(out)


def escalation_table(repo=REPO, profile=None, max_sessions=None) -> dict:
    profile = profile or load_profile(repo)
    a = profile["assumptions"]
    tiers = [tier_of(a["runtime"]), tier_of(a["scenario_runtime"])]
    models = sorted({s["model"] for s in (r2_data(repo).get("evo-x2") or {"sessions": []})["sessions"]})
    return {(m, t): router_escalation(repo, m, t, profile, max_sessions=max_sessions) for m in models for t in tiers}


def write_doc(repo=REPO, block=None):
    path = repo / "docs" / "IMPACT.md"
    text = path.read_text(encoding="utf-8")
    i, j = text.index(BEGIN), text.index(END) + len(END)
    path.write_text(text[:i] + block + text[j:], encoding="utf-8")


def main(argv=None):
    argv = argv or sys.argv[1:]
    res = compute_all()
    p = res["profile"]
    for k in ROW_FUNCS:
        print(f"{k} {PROBLEMS[k]}: {row_value(res['base'][k], p)}")
    print(f"total: USD/day {_fmt_usd(res['total']['usd_day'])} (excluded {res['total']['excluded_not_measured']})")
    print("ranking:", " > ".join(res["ranking"]))
    if "--write" in argv:
        write_doc(block=render_block(res, escalation_table()))
        print("wrote docs/IMPACT.md generated block")


if __name__ == "__main__":
    main()
