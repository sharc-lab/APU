"""Impact model (offline): what each measured problem costs a customer team, in USD per day and per month.

Plan v2 Part A3 (2026-10-13 meeting: "quantify every problem in money and time, and rank the problems by cost").
No machine time, no API call. Every input is a committed file, read fresh by the functions below:

  - customer profile: results/impact_profile_20261014.json. hour_value is the BLS median hourly wage (source URL,
    table, reference period and fetch date are inside the file; register row impact-hour-value). Every other field
    is an ASSUMPTION and is labelled so wherever it is used.
  - cloud prices: the shared Anthropic table results/cloud_prices_anthropic_20261014.json (transcribed from
    Anthropic's official pricing page; URL and fetch time inside), escalation model claude-sonnet-5-5, read through
    analysis/cloud_budget_forecast.py's call_cost / session_cost. Provider prompt caching is ON in the main case
    (cache-read price for re-sent history); no caching is printed only as the upper bound (operator decision B,
    2026-10-14). The profile's previous_basis (OpenAI table) is used only to show where the first model's one-flag
    figure came from (impact-scenario-one-flag).

Impact audit (operator decision C, 2026-10-14): every row's USD/day is split into a measured part and an assumed
part, with each assumption listed (value, source) and a status: "measured" (USD follows from measured per-session
quantities and cited prices at the profile volumes), "mixed" (a measured quantity whose USD also needs an assumption
beyond volume, e.g. that someone waits for the compute), "assumed" (the per-event cost itself is not measured).
Compute time is valued at the full hour value only as an upper bound; its lower bound is 0 (nobody waits). P1 prints
"not measured" until B3 task runs (end-state checked tool tasks) exist under at least two window settings: a
silent rule or recall failure is not converted into a wrong task result.
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
       py -3.12 analysis/impact_model.py --write-hybrid   rewrites only the hybrid savings block (row hybrid-savings-v1)
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
PRICES_FILE = "results/cloud_prices_anthropic_20261014.json"   # shared Anthropic table (operator decision A)
OPENAI_PRICES_FILE = "results/cloud_prices_20261008.json"       # previous basis only
B3_TASK_GLOB = "results/x2_evalset_tasks*.jsonl"                # B3 tool tasks with end-state checks
B3_MIN_WINDOWS = 2
CACHE_TTL_S = 300   # Anthropic default cache lifetime ("By default, the cache has a 5-minute lifetime")
TIME_ROWS = ("P2", "P4", "P5", "P6")
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


def price_file(profile=None) -> str:
    return ((profile or {}).get("assumptions") or {}).get("cloud_price_file") or PRICES_FILE


def prices(repo=REPO, profile=None):
    """The price table named by the profile (default: the shared Anthropic table)."""
    return json.loads((Path(repo) / price_file(profile)).read_text(encoding="utf-8"))


def session_cloud_cost(calls, model, tier, prices_, caching) -> float:
    """USD for one session's calls [(prompt_tokens, output_tokens)] at the price table (cloud_budget_forecast)."""
    return _cbf().session_cost(prices_, calls, model, tier, caching)


def resent_history_cost(calls, model, tier, prices_) -> dict:
    """The re-sent history of one session (per call, the tokens repeating the previous call's prompt, by
    cloud_budget_forecast.resent_tokens with the model's minimum cacheable prefix), priced at the cache-read rate
    (caching on, main case) and at the uncached input rate (no caching, upper bound)."""
    cbf = _cbf()
    rs = cbf.resent_tokens(calls, cbf.min_cacheable(prices_, model))
    cached = sum(t * cbf._rate(prices_, model, tier, p)["cached_input"] for t, (p, _o) in zip(rs, calls)) / 1e6
    unc = sum(t * cbf._rate(prices_, model, tier, p)["input"] for t, (p, _o) in zip(rs, calls)) / 1e6
    return {"tokens": sum(rs), "cached": cached, "uncached": unc}


def r2_cloud_overspend(repo=REPO, profile=None) -> dict:
    """{model: {...}} USD per session for the ollama_default R2 sessions of each model sent whole to the profile's
    cloud model, prompts converted to o200k_base with cloud_budget_forecast's per-seed ratio, outputs re-tokenized from
    the recorded replies: "cached" / "uncached" = the whole session with provider caching on / off; "resent_cached" /
    "resent_uncached" = the re-sent history alone (P3) at the cache-read / uncached input price; "overspend" = uncached
    minus cached (the first model's P3 definition, kept for the previous-basis figure). Input of register row
    r2-session-cloud-cost (parallel plan item)."""
    profile = profile or load_profile(repo)
    a = profile["assumptions"]
    cbf = _cbf()
    pr = prices(repo, profile)
    by = {}
    for dev, d in r2_data(repo).items():
        for s in d["sessions"]:
            if s["tier"] != DEFAULT_TIER:
                continue
            ratio = cbf.o200k_per_est_ratio(s["seed"])
            calls = [(round(c["prompt_tokens_est"] * ratio), cbf.r2_call_output_tokens(c))
                     for r in s["turns"] for c in r["calls"]]
            rh = resent_history_cost(calls, a["cloud_model"], a["cloud_tier"], pr)
            by.setdefault(s["model"], []).append(
                (session_cloud_cost(calls, a["cloud_model"], a["cloud_tier"], pr, False),
                 session_cloud_cost(calls, a["cloud_model"], a["cloud_tier"], pr, True), rh))
    m_ = statistics.mean
    return {m: {"uncached": m_(u for u, _, _ in v), "cached": m_(c for _, c, _ in v),
                "overspend": m_(u - c for u, c, _ in v), "resent_cached": m_(r["cached"] for _, _, r in v),
                "resent_uncached": m_(r["uncached"] for _, _, r in v), "resent_tokens": m_(r["tokens"] for _, _, r in v),
                "n": len(v)} for m, v in by.items()}


def escalated_cost(esc_calls, model, tier, prices_, ttl_s=CACHE_TTL_S) -> dict:
    """USD for one session's escalated steps [(prompt_tokens, output_tokens, t_start_s)] with caching on and off.
    Caching on: a step reads from cache the previous escalated step's prompt (min(prev, cur) tokens) when that prompt
    reached the model's minimum cacheable prefix and the previous escalated step started at most ttl_s earlier in the
    recorded local session (the cloud's own latency would change this timing; not measured); the rest of a cacheable
    prompt is a 5-minute cache write. Caching off: every prompt token at the input price."""
    cbf = _cbf()
    mc = cbf.min_cacheable(prices_, model)
    on = off = 0.0
    hits = 0
    prev_p, prev_t = 0, None
    for p, o, t in esc_calls:
        off += cbf.call_cost(prices_, model, tier, p, o)
        hit = prev_p >= mc and prev_t is not None and (t - prev_t) <= ttl_s
        cached = min(prev_p, p) if hit else 0
        write = (p - cached) if p >= mc else 0
        hits += bool(hit)
        on += cbf.call_cost(prices_, model, tier, p, o, cached=cached, write=write)
        prev_p, prev_t = p, t
    return {"cached": on, "uncached": off, "hits": hits, "n": len(esc_calls)}


def router_escalation(repo=REPO, model=None, tier=None, profile=None, quality_floor=0.9, max_sessions=None) -> dict:
    """Replay the evo-x2 R2 sessions of (model, tier) through src/dse/router.py's Router (definitions E1-E4) and
    price each cloud decision with the profile's price table: input = system + full history, output = the step's
    num_predict (the Router's own answer budget, an upper bound on output). "usd_per_session" is the main case,
    provider caching on between the session's escalated steps (escalated_cost); "usd_per_session_uncached" the
    no-caching upper bound. Token counts are the R2 harness's calibrated local-tokenizer estimate."""
    profile = profile or load_profile(repo)
    a = profile["assumptions"]
    from src.dse import r2_replay as rp
    pr = prices(repo, profile)
    d = r2_data(repo).get("evo-x2")
    sess = [s for s in (d["sessions"] if d else []) if s["model"] == model and s["tier"] == tier]
    if max_sessions:
        sess = sess[:max_sessions]
    if not sess:
        return {"cloud_steps_per_session": None, "usd_per_session": None, "usd_per_session_uncached": None,
                "steps_per_session": None, "cache_hit_share": None, "n": 0, "cloud_steps_values": [],
                "steps_values": []}
    clouds, on, off, steps, hits, n_esc = [], [], [], [], 0, 0
    for s in sess:
        start, acc = {}, 0.0
        for r in s["turns"]:                      # recorded local start time of each turn
            start[r["turn_idx"]] = acc
            acc += r.get("turn_wall_s") or 0.0
        router = rp.router_for_session(s["turns"], budget_usd=1e6, quality_floor=quality_floor)
        esc, n = [], 0
        for step in rp.session_steps(s["turns"]):
            dec = router.decide(step)
            n += 1
            if dec.target == "cloud":
                t = dec.tokens
                esc.append((t["full_total"] - t["answer_budget"], t["answer_budget"], start.get(step["turn_idx"], 0.0)))
        c = escalated_cost(esc, a["cloud_model"], a["cloud_tier"], pr)
        clouds.append(len(esc))
        on.append(c["cached"])
        off.append(c["uncached"])
        steps.append(n)
        hits += c["hits"]
        n_esc += c["n"]
    return {"cloud_steps_per_session": statistics.mean(clouds), "usd_per_session": statistics.mean(on),
            "usd_per_session_uncached": statistics.mean(off), "steps_per_session": statistics.mean(steps),
            "cache_hit_share": (hits / n_esc) if n_esc else None, "n": len(sess),
            "cloud_steps_values": sorted(set(clouds)), "steps_values": sorted(set(steps))}


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
    pr = prices(repo, profile)
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


def b3_task_windows(repo=REPO) -> dict:
    """B3 task runs (harness/x2_r2_evalset.py, tool tasks with an end-state check) per window. Validation files
    ('_val_' in the name) are the gate check before a real run, not a measurement, and do not count.
    Returns {"files", "windows", "met"}: met = at least B3_MIN_WINDOWS windows with end-state-checked task turns."""
    files, windows = [], set()
    for p in sorted(Path(repo).glob(B3_TASK_GLOB)):
        if "_val_" in p.name or not p.name.endswith(".jsonl"):
            continue
        files.append(p.name)
        with open(p, encoding="utf-8") as f:
            for line in f:
                if '"r2a_turn"' not in line:
                    continue
                r = json.loads(line)
                if r.get("record") == "r2a_turn" and r.get("end_state_ok") is not None:
                    windows.add(tier_of(r.get("arm_id", "")))
    return {"files": files, "windows": sorted(windows), "met": len(windows) >= B3_MIN_WINDOWS}


def p1(repo=REPO, profile=None, runtime=None, inputs=None):
    """P1 USD is NOT MEASURED until B3 task runs exist under >= 2 windows (operator decision C): the R2 rate below is
    a silent rule/recall failure rate, not a wrong-task rate, and is not converted into one. cell["scenario_usd_day"]
    keeps the first model's formula (each silent failure = one wrong result costing one full session rerun at the
    full hour value), labelled ASSUMED, for the scenario section only."""
    profile = profile or load_profile(repo)
    dev, model, tier = _cfg(profile, runtime)
    inputs = inputs or {}
    rates = inputs.get("rates") or context_loss_rates(repo)
    wall = inputs.get("wall") or session_wallclock(repo)
    b3 = inputs.get("b3") or b3_task_windows(repo)
    hv, spd = hour_value(profile), sessions_per_day(profile)
    grid = {}
    for (m, t), v in sorted(rates.items()):
        w = wall.get(("evo-x2", m, DEFAULT_TIER))
        p = v["silent_ctx"] / v["n"] if v["n"] else None
        redo = None if not w else w["median_wall_s"] / 3600.0
        grid[f"{m}@{t}"] = {"p": p, "k": v["silent_ctx"], "n": v["n"], "redo_h": redo, "usd_day": None,
                            "scenario_usd_day": cost_p1(spd, p, redo, hv)}
    cell = grid.get(f"{model}@{tier}") if dev == "evo-x2" else None
    return {"id": "P1", "usd_day": None, "cell": cell, "grid": grid, "b3": b3,
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
    sh = a["cloud_session_share"]
    grid = {m: dict(v, usd_day=cost_p3(spd, sh, v.get("resent_cached")),
                    usd_day_upper=cost_p3(spd, sh, v.get("resent_uncached"))) for m, v in sorted(over.items())}
    cell = grid.get(a["model"])
    return {"id": "P3", "usd_day": None if cell is None else cell["usd_day"],
            "usd_day_upper": None if cell is None else cell["usd_day_upper"], "cell": cell, "grid": grid,
            "config": f"{a['cloud_model']} {a['cloud_tier']} caching on, sessions shaped like {a['model']}'s"}


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
    "P1": {"formula": "NOT MEASURED until B3 task runs (end-state checked tool tasks) exist under >= 2 windows; then "
                      "sessions/day x P(wrong task end state after the window is exceeded) x rework hours x USD/h. "
                      "Scenario only (ASSUMED): the first model's sessions/day x P(gated silent rule/recall failure "
                      "at/after the first over-window turn) x one full session rerun (median session wall at Ollama's "
                      "default window) x USD/h",
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
    "P3": {"formula": "sessions/day x cloud share x mean(re-sent history tokens per session x cache-read price) with "
                      "provider caching on (main case); the same tokens x uncached input price is the no-caching upper "
                      "bound; shared Anthropic table cloud_prices_anthropic_20261014 (claude-sonnet-5-5), R2 session "
                      "token counts (o200k_base)",
           "measured": ["cloud-budget-tokens", "cloud-budget-inputs",
                        "r2-session-cloud-cost (chars/4 basis, caching floor; this row: o200k, implicit caching)"],
           "missing": "how much of a real customer's cloud history is a stable cacheable prefix (R2 sessions are "
                      "append-only, the best case for caching); Anthropic tokenizer counts (o200k_base used)",
           "fix": "keep the history an append-only prefix and turn provider caching on; trim history client-side",
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
    "P1": ("none", "USD not measured: no B3 task run yet, so no wrong-task rate; the R2 silent rule/recall rate (n in "
                   "the grid) is not a wrong-task rate, and rework time is not measured"),
    "P2": ("medium", "paired same-seed turns that fit both windows isolate the window size, but the per-turn delta is "
                     "scaled to a whole session by assumption and the right-sized window was not run"),
    "P3": ("high", "token counts and the official price page only; the daily figure scales linearly with the assumed "
                   "cloud share"),
    "P4": ("medium", "decode rates are measured per machine, but on different workloads (the Blade's from short dry "
                     "runs), and prefill is excluded, so it is a lower bound"),
    "P5": ("low", "co-runner ratios come from single-shot microbenchmarks applied to session timings, and the co-runner "
                  "share is an assumption"),
    "P6": ("medium", "rates are measured on the workload pack; the daily volume of single-shot calls is an assumption"),
    "P7": ("low", "the rate is measured but the cost per violation is not; a nonzero rate prints not measured"),
}

DECISIVE_MEASUREMENT = (
    "Two inputs P1 lacks: (1) a wrong-task rate per window, from B3 task runs with end-state checks under at least two "
    "window settings (not yet run), and (2) developer time lost per wrong result (time to notice plus rework, and how "
    "often it is never noticed). P7 lacks the cost of one guardrail violation. At any small window the silent "
    "rule/recall rates are high (R2-real-v1b-gated-kill), so these inputs decide whether context loss ranks first or "
    "last; until then it is not ranked.")


def _time_note(profile):
    v = profile["assumptions"].get("waiting_time_valuation") or {"lower": 0.0, "upper": 1.0}
    return v


def bounds(k, r, profile) -> tuple:
    """(low, high) USD/day. Time rows: low = lower x full value (0: nobody waits), high = upper x full value (someone
    waits for every second). P3: low = caching on, high = no caching. Others: the row value."""
    u = r["usd_day"]
    if u is None:
        return None, None
    if k in TIME_ROWS:
        v = _time_note(profile)
        return u * v["lower"], u * v["upper"]
    if k == "P3":
        return u, r.get("usd_day_upper")
    return u, u


def audit_row(k, r, profile) -> dict:
    """Operator decision C: split the row's USD/day into measured and assumed parts, list each assumption with its
    value and source, and mark the row measured / mixed / assumed."""
    a = profile["assumptions"]
    vol = [("team size", a["team_size"], "assumption (profile JSON)"),
           ("agent sessions per developer per day", a["agent_sessions_per_dev_per_day"], "assumption (profile JSON)")]
    wait = ("developer waits for this compute", "upper bound values it at the full hour value "
            f"USD {hour_value(profile):.2f}/h (impact-hour-value); lower bound 0 (nobody waits)",
            "assumption: who waits was not measured")
    c = r.get("cell") or {}
    lo, hi = bounds(k, r, profile)
    if k == "P1":
        sc = c.get("scenario_usd_day")
        b3 = r.get("b3") or {}
        return {"measured": None, "assumed": sc, "status": "assumed", "low": None, "high": None, "assumptions": vol + [
            ("a silent rule/recall failure is a wrong task result", f"{c.get('k')}/{c.get('n')} sessions treated as wrong",
             "not measured: B3 task runs under >= 2 windows needed; found "
             f"{', '.join(b3.get('windows') or []) or 'none'} (validation files excluded)"),
            ("rework per wrong result", "one full session rerun = median session wall at the default window "
             f"({'n/a' if c.get('redo_h') is None else f'{c['redo_h']:.2f} h'})", "assumption (not measured)"),
            wait]}
    if k == "P7":
        if r["usd_day"] == 0.0:
            return {"measured": 0.0, "assumed": 0.0, "status": "measured", "low": 0.0, "high": 0.0, "assumptions": vol + [
                ("cost per guardrail violation", "not needed: measured rate "
                 f"{c.get('k')}/{c.get('n')} sessions", "measured zero rate")]}
        return {"measured": None, "assumed": None, "status": "not measured", "low": None, "high": None,
                "assumptions": vol + [("cost per guardrail violation", "none assumed", "not measured")]}
    if k == "P3" and r["usd_day"] is None:
        return {"measured": None, "assumed": None, "status": "not measured", "low": None, "high": None,
                "assumptions": vol}
    if k == "P3":
        return {"measured": r["usd_day"], "assumed": 0.0, "status": "measured", "low": lo, "high": hi,
                "assumptions": vol + [
                    ("cloud session share", a["cloud_session_share"], "assumption (profile JSON)"),
                    ("cloud model", f"{a['cloud_model']} {a['cloud_tier']}, caching on",
                     f"{price_file(profile)} (official Anthropic pricing page)"),
                    ("token basis", "o200k_base counts", "assumption: Anthropic tokenizer counts not measured"),
                    ("re-sent history is a cacheable prefix", "R2 sessions are append-only",
                     "best case for caching; real histories not measured")]}
    extra = {"P2": [("turns per session", a["turns_per_session"], "R2 session length (scales a per-turn delta)")],
             "P4": [("decode only", "prefill excluded", "lower bound on the time itself")],
             "P5": [("co-runner share", a["co_runner_share"], "assumption (profile JSON)")],
             "P6": [("single-shot calls per developer per day", a["single_shot_calls_per_dev_per_day"],
                     "assumption (profile JSON)")]}[k]
    if r["usd_day"] is None:
        return {"measured": None, "assumed": None, "status": "not measured", "low": None, "high": None,
                "assumptions": vol + extra + [wait]}
    return {"measured": lo, "assumed": hi - lo, "status": "mixed", "low": lo, "high": hi,
            "assumptions": vol + extra + [wait], "hours_day": r["usd_day"] / hour_value(profile)}


def audit_total(rows: dict, audits: dict) -> dict:
    """Measured-only total (the measured column) plus a range: low = measured-only total; high = every row that is
    measured or mixed at its upper bound (time at the full hour value, cloud without caching). Assumed rows (P1, a
    nonzero-rate P7) are listed, never added."""
    meas = {k: v["measured"] for k, v in audits.items() if v["measured"] is not None}
    rng_rows = {k: v for k, v in audits.items() if v["status"] in ("measured", "mixed") and v["high"] is not None}
    return {"measured_usd_day": sum(meas.values()), "low": sum(v["low"] for v in rng_rows.values()),
            "high": sum(v["high"] for v in rng_rows.values()), "in_range": sorted(rng_rows),
            "excluded_assumed": sorted(k for k, v in audits.items() if v["status"] in ("assumed", "not measured"))}


def _fmt_usd(v):
    return NOT_MEASURED if v is None else f"{v:,.2f}"


def _rng(lo, hi):
    if lo is None or hi is None:
        return NOT_MEASURED
    return _fmt_usd(lo) if abs(hi - lo) < 0.005 else f"{_fmt_usd(lo)} to {_fmt_usd(hi)}"


def model_rows(repo=REPO, profile=None, runtime=None, inputs=None) -> dict:
    profile = profile or load_profile(repo)
    out = {}
    for k, f in ROW_FUNCS.items():
        r = f(repo, profile, runtime, (inputs or {}).get(k))
        r["usd_month"] = per_month(r["usd_day"], profile)
        r["confidence"] = CONFIDENCE[k]
        r["audit"] = audit_row(k, r, profile)
        out[k] = r
    return out


def total(rows: dict) -> dict:
    """Plain sum of the rows' main-case USD/day (caching on, time at the full hour value); rows without a number are
    listed. The audited figure is audit_total."""
    known = {k: r["usd_day"] for k, r in rows.items() if r["usd_day"] is not None}
    return {"usd_day": sum(known.values()), "included": sorted(known),
            "excluded_not_measured": sorted(k for k, r in rows.items() if r["usd_day"] is None)}


def _assumption_text(au) -> str:
    return "; ".join(f"{n} = {v} ({s})" for n, v, s in au["assumptions"])


def row_value(r: dict, profile) -> str:
    """One register value string for a problem row (profile config): the audit split first, then the measured
    per-session inputs, then the assumptions."""
    k = r["id"]
    au = r.get("audit") or audit_row(k, r, profile)
    c = r["cell"] or {}
    detail = ""
    if k == "P1" and c:
        b3 = r.get("b3") or {}
        detail = (f"; R2 silent rule/recall failures after the window {c['k']}/{c['n']} sessions (measured, not a "
                  f"wrong-task rate); B3 task-run windows {', '.join(b3.get('windows') or []) or 'none'} (needs "
                  f">= {B3_MIN_WINDOWS})")
    elif k == "P7" and c:
        detail = f"; rate {c['k']}/{c['n']} sessions"
    elif k == "P2" and c:
        detail = (f"; median per-turn delta {c['median_delta_s']:.2f} s vs {c['vs']} (n={c['n_turns']} paired turns)"
                  if c.get("vs") else "; smallest measured window")
    elif k == "P3" and c:
        detail = (f"; per session: re-sent history {c['resent_tokens']:,.0f} tok, USD {c['resent_cached']:.4f} caching "
                  f"on, {c['resent_uncached']:.4f} no caching; whole session to the cloud USD {c['cached']:.4f} "
                  f"caching on, {c['uncached']:.4f} no caching (n={c['n']})")
    elif k == "P4" and c:
        detail = f"; decode {c['rate']:.2f} tok/s vs best {c['best_rate']:.2f}"
    elif k == "P5" and c and c.get("ttft_ratio"):
        detail = f"; TTFT x{c['ttft_ratio']:.3f}, decode x{(c.get('decode_ratio') or 1):.3f} ({c['claim_id']})"
    elif k == "P6" and c:
        detail = (f"; timeouts {c['p_timeout']:.3f} x {c['timeout_s']} s, allocation failures {c['p_fail']:.3f} "
                  f"(n={c['n']}); escalation USD {c['escalation_usd_day']:.2f}/day")
    if au.get("hours_day") is not None:
        detail += f"; compute time {au['hours_day']:.2f} h/day"
    return (f"{r['config']}: status {au['status']}; measured USD/day {_fmt_usd(au['measured'])}, assumed USD/day "
            f"{_fmt_usd(au['assumed'])}, range USD/day {_rng(au['low'], au['high'])}{detail}; assumptions: "
            f"{_assumption_text(au)}; confidence {r['confidence'][0]}")


def session_time_by_window(repo=REPO) -> dict:
    """{(model, tier): {"median_wall_s", "n"}} per-session wall time on evo-x2 from the R2 sessions (same helper as
    session_wallclock)."""
    return {(m, t): {"median_wall_s": v["median_wall_s"], "n": v["n"]}
            for (d, m, t), v in session_wallclock(repo).items() if d == "evo-x2"}


def previous_basis(repo, profile, scen) -> dict | None:
    """Where the first model's one-flag figure came from: the scenario rows recomputed on the previous basis (OpenAI
    table, P3 = uncached minus cached session cost, P1 = rerun proxy, time at the full hour value)."""
    import copy
    a = profile["assumptions"]
    pb = a.get("previous_basis")
    if not pb:
        return None
    prof2 = copy.deepcopy(profile)
    prof2["assumptions"].update(cloud_model=pb["cloud_model"], cloud_price_file=pb["cloud_price_file"])
    over = r2_cloud_overspend(repo, prof2).get(a["model"]) or {}
    spd = sessions_per_day(profile)
    c1 = scen["P1"]["cell"] or {}
    terms = {"P1": c1.get("scenario_usd_day"), "P2": scen["P2"]["usd_day"],
             "P3": cost_p3(spd, a["cloud_session_share"], over.get("overspend")),
             "P4": scen["P4"]["usd_day"], "P5": scen["P5"]["usd_day"], "P6": scen["P6"]["usd_day"],
             "P7": scen["P7"]["usd_day"]}
    tot = sum(v for v in terms.values() if v is not None)
    return {"terms": terms, "usd_day": tot, "usd_month": per_month(tot, profile),
            "excluded": sorted(k for k, v in terms.items() if v is None), "sessions_day": spd,
            "p1_k": c1.get("k"), "p1_n": c1.get("n"), "redo_h": c1.get("redo_h"),
            "overspend_per_session": over.get("overspend"), "cloud_model": pb["cloud_model"],
            "price_file": pb["cloud_price_file"]}


def compute_all(repo=REPO, profile=None) -> dict:
    profile = profile or load_profile(repo)
    base = model_rows(repo, profile)
    scen = model_rows(repo, profile, runtime=profile["assumptions"]["scenario_runtime"])
    return {"profile": profile, "base": base, "scenario": scen, "total": total(base), "scenario_total": total(scen),
            "audit_total": audit_total(base, {k: r["audit"] for k, r in base.items()}),
            "scenario_audit_total": audit_total(scen, {k: r["audit"] for k, r in scen.items()}),
            "ranking": rank(base), "scenario_ranking": rank(scen),
            "previous_basis": previous_basis(repo, profile, scen), "session_time": session_time_by_window(repo)}


def scenario_value(res) -> str:
    """Register value for impact-scenario-one-flag: the previous-basis figure term by term, and the same scenario on
    the current basis split into measured and assumed."""
    p = res["profile"]
    a = p["assumptions"]
    pb = res["previous_basis"]
    st = res["scenario_audit_total"]
    sc = res["scenario"]
    f = _fmt_usd
    prev = "no previous basis in the profile"
    if pb:
        t = pb["terms"]
        prev = (f"previous basis (first impact model, before the 2026-10-14 audit): USD/day {f(pb['usd_day'])} (USD/month "
                f"{f(pb['usd_month'])}) = P1 {f(t['P1'])} (ASSUMED: {pb['p1_k']}/{pb['p1_n']} silent rule/recall "
                f"failures treated as wrong results x {pb['sessions_day']:.0f} sessions/day x one rerun "
                f"{pb['redo_h']:.4f} h x USD {hour_value(p):.2f}/h) + P2 {f(t['P2'])} + P3 {f(t['P3'])} ({pb['cloud_model']}, "
                f"uncached minus cached USD {pb['overspend_per_session']:.4f}/session x {pb['sessions_day']:.0f} x "
                f"cloud share {a['cloud_session_share']}) + P4 {f(t['P4'])} + P5 {f(t['P5'])} (time at the full hour "
                f"value); not included: {', '.join(pb['excluded'])}")
    parts = "; ".join(f"{k} {f(r['audit']['measured'])} measured / {f(r['audit']['assumed'])} assumed "
                      f"({r['audit']['status']})" for k, r in sc.items())
    return (f"SCENARIO, depends on assumed rework time per silent wrong result (not measured). One flag "
            f"{a['scenario_runtime']} on {a['device']} {a['model']}: {prev}. Current basis ({a['cloud_model']} caching "
            f"on, P1 not measured): measured-only USD/day {f(st['measured_usd_day'])}, range {_rng(st['low'], st['high'])} "
            f"over {', '.join(st['in_range'])}; assumed-only P1 {f((sc['P1']['cell'] or {}).get('scenario_usd_day'))}; "
            f"per row: {parts}")


def session_time_value(res) -> str:
    st = res["session_time"]
    models = sorted({m for m, _t in st})
    out = []
    for m in models:
        d, s = st.get((m, DEFAULT_TIER)), st.get((m, "num_ctx_4096"))
        if not d or not s:
            continue
        out.append(f"{m}: ollama_default median {d['median_wall_s'] / 60:.1f} min/session (n={d['n']}), num_ctx_4096 "
                   f"median {s['median_wall_s'] / 60:.1f} min/session (n={s['n']}), ratio {s['median_wall_s'] / d['median_wall_s']:.2f}")
    return "evo-x2 R2 sessions (40 turns), median wall time per session: " + "; ".join(out)


# ── impact v2 (operator priority override, 2026-10-21 18:30) ───────────────────────────────────────────────────
# One customer profile, stated explicitly (V2_PROFILE). P1 = measured excess wrong-task rate (B3 v2 task real run,
# qwen3:14b, session-level bootstrap CI: register rows x2-evalset-tasks-real-v2-wrong-rate and
# x2-evalset-tasks-real-v2-ci-units) x (cloud-redo cost per wrong task, per tier [row cloud-redo-cost-per-wrong-task;
# tagged "priced, not executed" while that row is in stub mode, "measured" once a live redo replaces it] + the measured
# wall time of the wrong task turn, valued 0 to the full hour value). P2 (operator decision 2026-10-21) = machine
# capacity: measured session minutes of the default and the fastest measured B5 setting for the profile model [row
# cost-cut-config-time] give sessions per machine per day, machines needed for the profile's sessions/day (ceil) and
# the machines saved x the reference machine price [row impact-ref-machine-price, a cited public list price] /
# an ASSUMED 3-year life; waiting time at the hour value is an UPPER BOUND on the interactive-session share only
# (ASSUMED 0.25). P2 is PENDING (values shown, not ranked) until row b5-fa-control is VERIFIED. P3 as in
# v1. P4 to P7: the v1 row functions at the same profile. Every row carries a status: measured / mixed / assumed /
# PENDING / not measured. An input whose register row is not VERIFIED stays PENDING and is never replaced by a number;
# compute_v2 reads both inputs from the same data functions as their register rows, so regeneration picks up a live
# cloud redo or the flash-attention control (row b5-fa-control) without a code change.

B3_V2_REAL_FILE = "results/x2_evalset_tasks_real_v2.jsonl"
B3_V2_MODEL = "qwen3:14b"
B3_V2_TIERS = ("ollama_ctx_16384", "ollama_ctx_8192", "ollama_ctx_4096", "ollama_ctx_4096_numkeep")
V2_PENDING = {
    "cloud-redo-cost-per-wrong-task": "USD to redo one wrong task in the cloud (P1)",
    "cost-cut-config-time": "minutes saved per session by configuration, flash-attention control on evo-x2 (P2)",
    "b5-fa-control": "flash-attention control on evo-x2: whether the P2 saving is flash attention or KV precision (P2)",
    "impact-ref-machine-price": "reference machine price, an assumption input that must be verified (P2)",
}
CLOUD_TAG_STUB, CLOUD_TAG_LIVE = "priced, not executed", "measured"
P2_CAVEAT_FA_PENDING = ("caveat: the B5 default ran without flash attention, so the saving may be flash attention or KV "
                        "precision; the flash-attention control (row b5-fa-control, PENDING) decides which")
P2_CAVEAT_FA_INCLUDED = ("the flash-attention control (row b5-fa-control) is in the comparison; its reading says whether "
                         "the saving is flash attention or KV precision")
V2_PROFILE = {
    "name": "impact-v2 profile",
    "device": "evo-x2", "model": B3_V2_MODEL, "runtime": "ollama_ctx_4096",
    "why": "qwen3:14b is the model with a measured wrong-task rate (B3 v2); num_ctx 4096 is the window Ollama loads "
           "by default on a CPU-only machine (t2s-r2-real-cpu: loaded window [4096]), so a CPU-only team is in this "
           "cell without choosing it; volumes (team size, sessions per day, cloud share, co-runner share, hour value) "
           "are those of results/impact_profile_20261014.json (ASSUMPTIONS, except the cited hour value)",
}


def v2_profile(repo=REPO, base=None) -> dict:
    import copy
    p = copy.deepcopy(base or load_profile(repo))
    p["assumptions"].update(device=V2_PROFILE["device"], model=V2_PROFILE["model"], runtime=V2_PROFILE["runtime"])
    return p


def register_has(repo, claim_id) -> bool:
    """True when docs/NUMBERS_REGISTER.md has a VERIFIED row for claim_id."""
    path = Path(repo) / "docs" / "NUMBERS_REGISTER.md"
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"| {claim_id} |"):
            return "| VERIFIED |" in line
    return False


def b3_v2_wrong_task(repo=REPO) -> dict:
    """Per tier of the B3 v2 real run (qwen3:14b): excess wrong-task rate over the default with its session-level
    95% CI (harness/x2_r2_evalset.tasks_real_v2_outcome, the pre-registered bootstrap), task turns per session, and
    the mean wall time of a wrong task turn (seconds, turn_wall_s). Raises FileNotFoundError without the file."""
    sys.path.insert(0, str(Path(repo) / "harness"))
    import x2_r2_evalset as ev
    rows = ev.ag.read_rows(Path(repo) / B3_V2_REAL_FILE)
    if not rows:
        raise FileNotFoundError(Path(repo) / B3_V2_REAL_FILE)
    o = ev.tasks_real_v2_outcome(rows, models=(B3_V2_MODEL,))["models"][B3_V2_MODEL]
    att = {s["attempt_id"] for s in ev.ag.completed_sessions(rows)
           if s.get("mode") == "real" and s["model_id"] == B3_V2_MODEL}
    walls = {}
    for r in rows:
        if (r.get("record") == "r2a_turn" and r.get("attempt_id") in att and r.get("item_kind") == "task"
                and r.get("end_state_ok") is False and r.get("turn_wall_s") is not None):
            walls.setdefault(r["arm_id"].replace("_call2_notools", ""), []).append(r["turn_wall_s"])
    out = {}
    for t, e in (o.get("excess_over_default") or {}).get("per_tier", {}).items():
        ti = o["tiers"][t]
        out[t] = {"excess": e["excess"], "ci95": tuple(e["ci95"]), "rate": e["rate"], "default_rate": e["default_rate"],
                  "n_sessions": e["n_sessions"], "tasks_per_session": ti["n_task_turns"] / ti["n_sessions"],
                  "wrong_turn_wall_s": statistics.mean(walls[t]) if walls.get(t) else None,
                  "n_wrong": ti["wrong"], "silent_wrong": ti["silent_wrong"], "valid": o["valid"]}
    return out


def cost_p1_v2(sessions_day, tasks_per_session, excess, redo_usd, wall_s, hour, wait=1.0):
    """sessions/day x task turns/session x excess wrong-task rate x (cloud redo USD per wrong task + wall s of the
    wrong turn / 3600 x USD/h x wait share). Returns (cloud USD/day or None when redo_usd is PENDING, time USD/day)."""
    wrong_day = sessions_day * tasks_per_session * max(0.0, excess)
    cloud = None if redo_usd is None else wrong_day * redo_usd
    time_usd = None if wall_s is None else wrong_day * wall_s / 3600.0 * hour * wait
    return cloud, time_usd


def p1_v2(repo=REPO, profile=None, inputs=None) -> dict:
    profile = profile or v2_profile(repo)
    inputs = inputs or {}
    wt = inputs.get("b3v2") or b3_v2_wrong_task(repo)
    # USD per wrong task: a float (every tier) or {tier: USD}; None while cloud-redo-cost-per-wrong-task is PENDING.
    redo = inputs.get("redo_usd")
    cloud_tag = inputs.get("redo_tag", CLOUD_TAG_STUB) if redo is not None else None
    hv, spd = hour_value(profile), sessions_per_day(profile)
    grid = {}
    for t in B3_V2_TIERS:
        v = wt.get(t)
        if not v:
            continue
        r_t = redo.get(t) if isinstance(redo, dict) else redo
        c, tm = cost_p1_v2(spd, v["tasks_per_session"], v["excess"], r_t, v["wrong_turn_wall_s"], hv)
        c_lo, tm_lo = cost_p1_v2(spd, v["tasks_per_session"], v["ci95"][0], r_t, v["wrong_turn_wall_s"], hv)
        c_hi, tm_hi = cost_p1_v2(spd, v["tasks_per_session"], v["ci95"][1], r_t, v["wrong_turn_wall_s"], hv)
        grid[t] = dict(v, wrong_per_day=spd * v["tasks_per_session"] * max(0.0, v["excess"]), cloud_usd_day=c,
                       cloud_usd_day_ci=(c_lo, c_hi), redo_usd=r_t, time_usd_day=tm, time_usd_day_ci=(tm_lo, tm_hi))
    cell = grid.get(profile["assumptions"]["runtime"])
    cloud = None if cell is None else cell["cloud_usd_day"]
    pending = [] if cloud is not None else ["cloud-redo-cost-per-wrong-task"]
    time_hi = None if cell is None else cell["time_usd_day"]
    usd = None if cell is None else (time_hi if cloud is None else time_hi + cloud)
    return {"id": "P1", "usd_day": usd, "lower_bound_only": bool(pending), "pending": pending, "cell": cell,
            "grid": grid, "status": "mixed", "cloud_tag": cloud_tag if cloud is not None else None,
            "low": 0.0 if cloud is None else cloud, "high": usd}


def v2_rows(repo=REPO, base_profile=None, inputs=None) -> dict:
    """The seven rows at V2_PROFILE: {k: {"usd_day" (main case: time at the full hour value, caching on), "low",
    "high", "status", "pending", "note"}}."""
    inputs = inputs or {}
    prof = v2_profile(repo, base_profile)
    out = {"P1": p1_v2(repo, prof, inputs.get("P1"))}
    out["P2"] = p2_v2(prof, inputs.get("P2"), inputs.get("P2_machine"), bool(inputs.get("P2_fa_verified")))
    conf = inputs.get("shift_conf")
    if conf is not None:
        out["P1"]["detection"] = detection_projection(out["P1"], prof, conf)  # superseded by detection_measured
    p1_in = inputs.get("P1") or {}
    if p1_in.get("redo_usd") is not None:
        redo = p1_in["redo_usd"]
        redo = redo if isinstance(redo, dict) else {t: redo for t in DETECTION_TIERS}
        out["P1"]["detection_measured"] = measured_detection(
            inputs.get("task_turns") or b3_v2_task_turns(repo), redo, p1_in.get("redo_tag", CLOUD_TAG_STUB))
    v1 = model_rows(repo, prof, inputs=inputs.get("v1"))
    for k in ("P3", "P4", "P5", "P6", "P7"):
        r, au = v1[k], v1[k]["audit"]
        out[k] = {"id": k, "usd_day": r["usd_day"], "low": au["low"], "high": au["high"], "status": au["status"],
                  "pending": [], "cell": r.get("cell"), "config": r.get("config")}
    return {"profile": prof, "rows": out}


REF_MACHINE_FILE = "results/hardware_price_evox2_20261021.json"   # cited public list price (row impact-ref-machine-price)
MACHINE_LIFE_YEARS = 3.0    # ASSUMED amortization life of the reference machine
INTERACTIVE_SHARE = 0.25    # ASSUMED share of sessions a person waits on (the waiting-time upper bound applies only here)
MINUTES_PER_DAY = 24 * 60


def ref_machine(repo=REPO) -> dict:
    """The reference machine's price record (results/hardware_price_evox2_20261021.json): a cited public list price
    with its label ("list price, <source>, <date>", operator decision 2026-10-22)."""
    return json.loads((Path(repo) / REF_MACHINE_FILE).read_text(encoding="utf-8"))


def price_label(machine) -> str:
    """The machine price's label as recorded in its file; files without one are an unverified assumption."""
    return (machine or {}).get("label") or "ASSUMPTION, must be verified"


def p2_capacity(sessions_day, default_min, best_min, machine_usd, life_years=MACHINE_LIFE_YEARS, concurrency=1):
    """Machine capacity: sessions per machine per day per setting (24 h / session minutes x concurrency, one session at
    a time unless a measured concurrency is given), machines needed for sessions_day (ceil) under the default and the
    best setting, and the USD of machines saved amortized per day over life_years."""
    per_default = MINUTES_PER_DAY / default_min * concurrency
    per_best = MINUTES_PER_DAY / best_min * concurrency
    n_default, n_best = math.ceil(sessions_day / per_default), math.ceil(sessions_day / per_best)
    saved = max(0, n_default - n_best)
    return {"sessions_per_machine_day_default": per_default, "sessions_per_machine_day_best": per_best,
            "machines_default": n_default, "machines_best": n_best, "machines_saved": saved,
            "usd_saved": saved * machine_usd, "usd_day": saved * machine_usd / (life_years * 365.0),
            "life_years": life_years, "concurrency": concurrency}


def p2_v2(profile, cc=None, machine=None, fa_verified=False, interactive_share=INTERACTIVE_SHARE) -> dict:
    """P2 at the v2 profile (operator decision 2026-10-21). Primary USD = machine capacity from the measured session
    minutes of the runtime default and the fastest measured B5 setting for the profile model (cost_cut_config_time_data
    shape: {"models": {model: {"default_min", "best_min", "saved_min", ...}}, "fa_included"}) at the profile's
    sessions/day (ASSUMED), with the reference machine price (machine: {"usd", "label", ...}, a cited list price) amortized over
    MACHINE_LIFE_YEARS (ASSUMED). Waiting time at the hour value is an UPPER BOUND on top, applied only to the
    interactive-session share (ASSUMED). Status PENDING (values still computed and shown) until row b5-fa-control is
    VERIFIED; the ranking does not place a PENDING row."""
    model = profile["assumptions"]["model"]
    x = ((cc or {}).get("models") or {}).get(model)
    r = {"id": "P2", "usd_day": None, "low": None, "high": None, "status": "PENDING",
         "pending": (["cost-cut-config-time"] if x is None else []) + (["impact-ref-machine-price"] if machine is None
                                                                       else []), "cell": None}
    if x is None or machine is None:
        return r
    spd, hv = sessions_per_day(profile), hour_value(profile)
    cap = p2_capacity(spd, x["default_min"], x["best_min"], float(machine["usd"]))
    wait_full = spd * max(0.0, x["saved_min"]) / 60.0 * hv
    wait_ub = wait_full * interactive_share
    caveat = ((cc.get("fa_attribution") or P2_CAVEAT_FA_INCLUDED) if cc.get("fa_included") else P2_CAVEAT_FA_PENDING)
    r.update(usd_day=cap["usd_day"], low=cap["usd_day"], high=cap["usd_day"] + wait_ub,
             status="mixed" if fa_verified else "PENDING", pending=[] if fa_verified else ["b5-fa-control"],
             cell=dict(x, model=model), caveat=caveat, fa_included=bool(cc.get("fa_included")), capacity=cap,
             machine_usd=float(machine["usd"]), machine_label=price_label(machine), wait_upper_bound=wait_ub, wait_full_share=wait_full,
             interactive_share=interactive_share)
    return r


# Value of detection (operator decision 2026-10-21, P1 framing): fixing one wrong answer in the cloud is cheap (row
# cloud-redo-cost-per-wrong-task); the cost is that the failure is silent, so nobody knows which answers to fix. The
# projection compares escalating every task step to the cloud with escalating only the steps a detector flags,
# using the R2 context-shift predictor's measured precision and recall (row R2-shift-predictor-confusion). That
# predictor was measured on evo-x2, Ollama 0.34.4 (bundled llama.cpp b11081), llama3.1:8b, the R2 mechanism job, and it
# predicts a llama.cpp context shift (P + C > n_ctx), NOT a wrong task: applying it to qwen3:14b wrong tasks at
# num_ctx 4096 is an ASSUMPTION, so every number here is labelled projection.
SHIFT_PREDICTOR_SCOPE = ("measured on evo-x2, Ollama 0.34.4 (bundled llama.cpp b11081), {model}, R2 mechanism job "
                         "({n} calls; tiers {tiers}); it predicts a llama.cpp context shift (P + C > n_ctx) from the "
                         "rendered prompt and completion tokens, not a wrong task; using its precision and recall for "
                         "wrong tasks of another model and runtime cell is an ASSUMPTION")


def shift_predictor_confusion(repo=REPO) -> dict:
    """Overall precision/recall of the R2 context-shift predictor on the mechanism job (row
    R2-shift-predictor-confusion) with the scope it was measured in."""
    sys.path.insert(0, str(Path(repo) / "analysis"))
    import r2_shift_predictor as sp
    rows = sp.read_rows(Path(repo) / sp.MECH_FILE)
    ms = sp.mechanism_summary(rows)
    o = ms["overall"]
    models = sorted({r["model_id"] for r in rows if r.get("record") == "r2m_call" and r.get("model_id")})
    tiers = [t for t, c in ms["per_tier"].items() if c["tp"] + c["fn"] > 0]
    return {"precision": o["precision"], "recall": o["recall"], "tp": o["tp"], "fp": o["fp"], "fn": o["fn"],
            "tn": o["tn"], "n_calls": ms["n_calls"], "models": models, "tiers_with_shifts": tiers,
            "scope": SHIFT_PREDICTOR_SCOPE.format(model=", ".join(models), n=ms["n_calls"],
                                                  tiers=", ".join(ms["per_tier"]))}


def detection_value(sessions_day, tasks_per_session, wrong_rate, redo_usd, precision, recall):
    """Projection: cloud redo USD/day when every task step is escalated vs only the steps a detector flags.
    flagged = recall x wrong / precision (true flags / precision); wrong steps the detector misses stay wrong."""
    steps = sessions_day * tasks_per_session
    wrong = steps * wrong_rate
    every = steps * redo_usd
    caught = None if recall is None else wrong * recall
    flagged = caught / precision if (precision and caught is not None) else None
    pred = None if flagged is None else flagged * redo_usd
    return {"steps_day": steps, "wrong_day": wrong, "every_usd_day": every, "flagged_day": flagged,
            "flagged_usd_day": pred, "caught_day": caught, "missed_day": None if caught is None else wrong - caught,
            "saved_usd_day": None if pred is None else every - pred}


def detection_projection(p1_row, profile, conf) -> dict | None:
    """The detection projection at the v2 profile cell, from the P1 v2 row (measured wrong-task rate at the cell,
    cloud redo USD per wrong task at the cell and its tag) and the shift predictor's confusion."""
    cell = p1_row.get("cell")
    if not cell or cell.get("redo_usd") is None or conf.get("precision") is None:
        return None
    d = detection_value(sessions_per_day(profile), cell["tasks_per_session"], cell["rate"], cell["redo_usd"],
                        conf["precision"], conf["recall"])
    return dict(d, rate=cell["rate"], redo_usd=cell["redo_usd"], cloud_tag=p1_row.get("cloud_tag"),
                runtime=profile["assumptions"]["runtime"], model=profile["assumptions"]["model"], conf=conf,
                label="projection")


def detection_value_text(d) -> str:
    if d is None:
        return "PENDING: needs a cloud redo cost at the profile cell and a predictor with defined precision"
    c, f = d["conf"], _fmt_usd
    return (f"PROJECTION at {d['model']} {d['runtime']}: {d['steps_day']:.0f} task steps/day (ASSUMED volume x "
            f"measured task turns/session), measured wrong-task rate {100 * d['rate']:.1f}% "
            f"(x2-evalset-tasks-real-v2-wrong-rate) = {d['wrong_day']:.1f} wrong/day; cloud redo USD "
            f"{d['redo_usd']:.4f} per step ({d['cloud_tag']}, cloud-redo-cost-per-wrong-task); escalate every step: "
            f"USD {f(d['every_usd_day'])}/day; escalate only predicted-at-risk steps at the shift predictor's measured "
            f"precision {100 * c['precision']:.1f}% and recall {100 * c['recall']:.1f}% (R2-shift-predictor-confusion: "
            f"TP {c['tp']} FP {c['fp']} FN {c['fn']} TN {c['tn']}): {d['flagged_day']:.1f} steps/day escalated, USD "
            f"{f(d['flagged_usd_day'])}/day, {d['caught_day']:.1f} wrong/day fixed and {d['missed_day']:.1f} missed; "
            f"value of detection USD {f(d['saved_usd_day'])}/day; where the predictor applies: {c['scope']}; assumed: "
            f"a redo of any step costs the mean redo of a wrong one, a fixed step is right after the redo (cloud "
            f"correctness not measured in stub mode)")


# Measured detection (operator item 4, 2026-10-22): replaces the borrowed shift-predictor figure above. On the B3 v2 real
# run itself (qwen3:14b, five tiers), the predictor "the task turn's prompt exceeded the loaded window" (the turn's
# over_loaded_window, else final_call_prompt_tokens_calibrated > loaded_context) is scored against "the task ended
# wrong" (end_state_ok false). USD: escalate every task turn vs only the flagged ones, each at the tier's cloud redo USD
# per wrong task (row cloud-redo-cost-per-wrong-task, via cloud_redo_data; labelled by that row's mode).
DETECTION_TIERS = ("ollama_default",) + B3_V2_TIERS


def turn_over_window(r) -> bool:
    if r.get("over_loaded_window") is not None:
        return bool(r["over_loaded_window"])
    p, w = r.get("final_call_prompt_tokens_calibrated"), r.get("loaded_context")
    return p is not None and w is not None and p > w


def confusion_from_turns(turns) -> dict:
    """turns: [(flagged, wrong)] -> TP/FP/FN/TN, precision (None if nothing flagged), recall (None if nothing wrong)."""
    tp = sum(1 for f, w in turns if f and w)
    fp = sum(1 for f, w in turns if f and not w)
    fn = sum(1 for f, w in turns if not f and w)
    tn = sum(1 for f, w in turns if not f and not w)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": tp + fp + fn + tn,
            "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None,
            "fn_share": fn / (tp + fn) if tp + fn else None}


def b3_v2_task_turns(repo=REPO) -> dict:
    """{tier: [(over loaded window, ended wrong)]} over task turns of completed real sessions of the B3 v2 real run."""
    sys.path.insert(0, str(Path(repo) / "harness"))
    import x2_r2_evalset as ev
    rows = ev.ag.read_rows(Path(repo) / B3_V2_REAL_FILE)
    if not rows:
        raise FileNotFoundError(Path(repo) / B3_V2_REAL_FILE)
    att = {s["attempt_id"] for s in ev.ag.completed_sessions(rows)
           if s.get("mode") == "real" and s["model_id"] == B3_V2_MODEL}
    out = {}
    for r in rows:
        if (r.get("record") == "r2a_turn" and r.get("attempt_id") in att and r.get("item_kind") == "task"
                and r.get("end_state_ok") is not None):
            out.setdefault(r["arm_id"].replace("_call2_notools", ""), []).append(
                (turn_over_window(r), r["end_state_ok"] is False))
    return out


def measured_detection(turns_by_tier: dict, redo_usd: dict, redo_tag: str) -> dict:
    """Per tier: confusion of the over-window predictor for wrong tasks, and cloud USD escalating every task turn vs
    only flagged turns (tier redo USD per task; a tier with no redo figure stays None)."""
    out = {}
    for t in DETECTION_TIERS:
        if t not in turns_by_tier:
            continue
        c = confusion_from_turns(turns_by_tier[t])
        u = redo_usd.get(t) if redo_usd else None
        flagged = c["tp"] + c["fp"]
        out[t] = dict(c, redo_usd=u, flagged=flagged,
                      every_usd=None if u is None else c["n"] * u,
                      flagged_usd=None if u is None else flagged * u,
                      saved_usd=None if u is None else (c["n"] - flagged) * u)
    return {"tiers": out, "redo_tag": redo_tag, "label": "measured detection"}


def measured_detection_text(d) -> str:
    def pct(v):
        return "n/a" if v is None else f"{100 * v:.1f}%"

    parts = []
    for t, c in d["tiers"].items():
        usd = ("USD n/a (no redo figure for the tier)" if c["redo_usd"] is None else
               f"escalate all {c['n']} USD {c['every_usd']:.4f} vs flagged {c['flagged']} USD {c['flagged_usd']:.4f} "
               f"(saves USD {c['saved_usd']:.4f} and leaves {c['fn']} wrong unfixed; USD {c['redo_usd']:.4f} per task)")
        parts.append(f"{t}: TP {c['tp']} FP {c['fp']} FN {c['fn']} TN {c['tn']}, precision {pct(c['precision'])}, "
                     f"recall {pct(c['recall'])}, wrong tasks missed {c['fn']}/{c['tp'] + c['fn']} "
                     f"({pct(c['fn_share'])}); {usd}")
    return ("MEASURED on the B3 v2 real run (qwen3:14b, evo-x2, x2_evalset_tasks_real_v2): predictor = the task turn's "
            "prompt exceeded the loaded window (over_loaded_window), outcome = the task ended wrong (end_state_ok "
            "false); per tier: " + "; ".join(parts)
            + (f"; {d['redo_source']}" if d.get("redo_source") else
               f"; cloud redo USD per task: tier mean of cloud-redo-cost-per-wrong-task ({d['redo_tag']}); assumed: a "
               f"redo of any task costs the tier's mean redo of a wrong one, a fixed task is right after the redo "
               f"(cloud correctness not measured in stub mode)")
            + "; USD totals are for the run's own task turns")


# Operator decision 2026-10-22 (row impact-v3-detection-value-measured): no value in the row comes from the first
# (truncated-history) redo any more. The escalate-all vs escalate-flagged USD use the full-history redo's per-tier mean
# USD per task (cloud-redo-full-context, measured); the confusion counts, precision and recall are unchanged (B3 v2
# local run).
def full_redo_detection(repo=REPO) -> dict | None:
    """measured_detection with the full-history redo's per-tier mean USD; None until cloud-redo-full-context is
    VERIFIED."""
    if not register_has(repo, "cloud-redo-full-context"):
        return None
    tasks, model, n_wrong, _p = redo_rows_and_model(repo)
    per = {}
    for r in tasks:
        per.setdefault(r["tier"], []).append(r["usd"])
    usd = {t: statistics.fmean(v) for t, v in per.items()}
    d = measured_detection(b3_v2_task_turns(repo), usd, CLOUD_TAG_LIVE)
    fixed = sum(1 for r in tasks if r.get("cloud_end_state_ok"))
    d["redo_source"] = (f"cloud redo USD per task: tier mean of the full-history redo (cloud-redo-full-context, "
                        f"{model}, {CLOUD_TAG_LIVE}, cache-priced, per-tier n "
                        + ", ".join(f"{t} {len(per[t])}" for t in d["tiers"] if t in per)
                        + f"); no value from the first truncated-history redo (cloud-redo-cost-per-wrong-task); "
                        f"assumed: a redo of any task, flagged or not, costs the tier's mean full-history redo of a "
                        f"wrong one; the redo fixed {fixed}/{n_wrong} wrong tasks (cloud-redo-full-context), so an "
                        f"escalated wrong task is not always fixed")
    return d


def v2_ranking(rows: dict) -> list:
    """Rows ranked by main-case USD/day; a PENDING row (even with its values computed) sorts as unranked."""
    return rank({k: {"usd_day": None if r["status"] == "PENDING" else r["usd_day"]} for k, r in rows.items()})


def v2_split(rows: dict) -> dict:
    s = {}
    for k, r in rows.items():
        s.setdefault(r["status"], []).append(k)
    return s


def p2_v2_value(r) -> str:
    """Text for P2 inside the ranking row: machine capacity first, the waiting-time upper bound second."""
    x, c = r["cell"], r["capacity"]
    return (f"machine capacity: {x['model']} default (a) {x['default_min']:.1f} min per session = "
            f"{c['sessions_per_machine_day_default']:.2f} sessions/machine/day vs ({x['best']}) {x['best_name']} "
            f"{x['best_min']:.1f} min = {c['sessions_per_machine_day_best']:.2f} (MEASURED minutes, cost-cut-config-time; "
            f"24 h / session minutes, one session at a time, no measured concurrency row); machines for the profile's "
            f"sessions/day (ASSUMED volume): {c['machines_default']} default vs {c['machines_best']} "
            f"best (ceil{', half the hardware' if 2 * c['machines_best'] == c['machines_default'] else ''}), "
            f"{c['machines_saved']} saved "
            f"x USD {r['machine_usd']:,.0f} (impact-ref-machine-price, {r.get('machine_label', price_label(None))}) / "
            f"{c['life_years']:.0f}-year life (ASSUMED) = USD {c['usd_day']:.2f}/day; "
            f"upper bound: waiting time {x['saved_min']:.1f} min saved per session x hour value x interactive-session "
            f"share {r['interactive_share']:.2f} (ASSUMED) = USD {r['wait_upper_bound']:,.2f}/day on top; {r['caveat']}")


def p1_v2_value(res) -> str:
    rows, prof = res["rows"], res["profile"]
    r = rows["P1"]
    f = _fmt_usd
    parts = []
    for t, v in r["grid"].items():
        lo, hi = v["ci95"]
        parts.append(f"{t}: excess {100 * v['excess']:+.1f} points (session-level 95% CI {100 * lo:+.1f} to "
                     f"{100 * hi:+.1f}), {v['tasks_per_session']:.0f} task turns/session, {v['wrong_per_day']:.1f} "
                     f"excess wrong tasks/day, wrong-turn wall {v['wrong_turn_wall_s']:.1f} s; time USD/day 0.00 to "
                     f"{f(v['time_usd_day'])} (CI {f(v['time_usd_day_ci'][0])} to {f(v['time_usd_day_ci'][1])}); "
                     f"cloud redo USD/day "
                     + ("PENDING (cloud-redo-cost-per-wrong-task)" if v["cloud_usd_day"] is None else
                        f"{f(v['cloud_usd_day'])} (USD {v['redo_usd']:.4f} per wrong task, {r['cloud_tag']}; CI "
                        f"{f(v['cloud_usd_day_ci'][0])} to {f(v['cloud_usd_day_ci'][1])})"))
    a = prof["assumptions"]
    return (f"profile {a['device']} {a['model']} {a['runtime']}, {sessions_per_day(prof):.0f} sessions/day (ASSUMED "
            f"volume), hour value USD {hour_value(prof):.2f} (impact-hour-value); status {r['status']}"
            + (", lower bound only (cloud redo part PENDING)" if r["lower_bound_only"] else
               f", cloud part {r['cloud_tag']}")
            + f"; profile cell USD/day {_rng(r['low'], r['high'])}; " + "; ".join(parts)
            + "; measured: excess rate and its CI (x2-evalset-tasks-real-v2-wrong-rate, x2-evalset-tasks-real-v2-ci-"
              "units), task turns per session, wrong-turn wall time"
            + ("" if r["lower_bound_only"] else
               f"; {r['cloud_tag']}: cloud redo USD per wrong task, per tier (cloud-redo-cost-per-wrong-task)")
            + "; assumed: volume, who waits (0 to full hour value), a wrong task costs one cloud redo plus the wasted "
              "turn (time to notice not counted)")


def ranking_v2_value(res) -> str:
    rows, prof = res["rows"], res["profile"]
    a = prof["assumptions"]
    rk = v2_ranking(rows)
    lines = []
    for k in rk:
        r = rows[k]
        tag = r["status"] + (f", PENDING inputs {', '.join(r['pending'])}" if r["pending"] else "")
        if r.get("lower_bound_only"):
            tag += ", lower bound"
        if r.get("cloud_tag"):
            tag += f", cloud part {r['cloud_tag']}"
        if k == "P2" and r["usd_day"] is not None:
            tag += "; " + p2_v2_value(r)
        if r["usd_day"] is None:
            head = "PENDING" if r["status"] == "PENDING" else NOT_MEASURED
        elif r["status"] == "PENDING":
            head = f"PENDING (computed {_fmt_usd(r['usd_day'])}, not ranked)"
        else:
            head = _fmt_usd(r["usd_day"])
        lines.append(f"{k} {head} (range {_rng(r['low'], r['high']) if r['usd_day'] is not None else 'n/a'}; {tag})")
    split = v2_split(rows)
    known = [k for k in rk if rows[k]["usd_day"] is not None and rows[k]["status"] != "PENDING"]
    pend = sorted(set(x for r in rows.values() for x in r["pending"]))
    return (f"one profile: {a['device']} {a['model']} {a['runtime']}, team {a['team_size']} x "
            f"{a['agent_sessions_per_dev_per_day']} sessions/day, cloud {a['cloud_model']} caching on (ASSUMED volumes; "
            f"hour value cited); ranking by main-case USD/day (P2: machine capacity; the other rows: time at the full "
            f"hour value; a PENDING row is not ranked): {' > '.join(known)}; "
            f"unranked: {', '.join(k for k in rk if k not in known) or 'none'}; rows: " + "; ".join(lines)
            + "; split: " + ", ".join(f"{s} {'/'.join(ks)}" for s, ks in sorted(split.items()))
            + f"; PENDING inputs: {', '.join(f'{p} ({V2_PENDING[p]})' for p in pend) or 'none'}")


def compute_v2(repo=REPO) -> dict:
    return v2_rows(repo, inputs=v2_inputs(repo))


def v2_inputs(repo=REPO) -> dict:
    """The register-backed inputs, read only when their register row is VERIFIED (otherwise they stay PENDING), from
    the same data functions the rows use: cloud redo USD per wrong task per tier (tagged by the row's mode) for P1;
    minutes saved per session by configuration, the reference machine price (an assumption input) and whether the
    flash-attention control is VERIFIED (P2 stays PENDING until it is) for P2."""
    import numbers_register as nr
    inputs = {}
    if register_has(repo, "cloud-redo-cost-per-wrong-task"):
        d = nr.cloud_redo_data(Path(repo))
        inputs["P1"] = {"redo_usd": {t: v["mean_usd"] for t, v in d["per_tier"].items() if v["mean_usd"] is not None},
                        "redo_tag": CLOUD_TAG_STUB if d["stub"] else CLOUD_TAG_LIVE}
    if register_has(repo, "cost-cut-config-time"):
        inputs["P2"] = nr.cost_cut_config_time_data(Path(repo))
    if register_has(repo, "impact-ref-machine-price"):
        inputs["P2_machine"] = ref_machine(repo)
    if register_has(repo, "b5-fa-control"):
        inputs["P2_fa_verified"] = True
    if register_has(repo, "R2-shift-predictor-confusion"):
        inputs["shift_conf"] = shift_predictor_confusion(repo)
    return inputs


# ── docs/IMPACT.md generated block ──────────────────────────────────────────────────────────────────────────────

BEGIN, END = "<!-- impact_model:begin -->", "<!-- impact_model:end -->"


def _grid_lines(k, r, profile):
    lines = []
    for key, v in r["grid"].items():
        u = _fmt_usd(v.get("usd_day"))
        extra = ""
        if k == "P1":
            u = NOT_MEASURED
            extra = (f"{v['k']}/{v['n']} sessions silent rule/recall failure after the window; scenario (ASSUMED rerun "
                     f"{'n/a' if v['redo_h'] is None else f'{v['redo_h']:.2f} h'}): USD {_fmt_usd(v['scenario_usd_day'])}")
        elif k == "P7":
            extra = f"{v['k']}/{v['n']} sessions"
        elif k == "P2":
            extra = (f"vs {v['vs']}: {v['median_delta_s']:.2f} s/turn (n={v['n_turns']})" if v.get("vs") and
                     v.get("median_delta_s") is not None else "smallest measured window")
        elif k == "P3":
            u = _rng(v.get("usd_day"), v.get("usd_day_upper"))
            extra = (f"re-sent {v['resent_tokens']:,.0f} tok/session: USD {v['resent_cached']:.4f} caching on, "
                     f"{v['resent_uncached']:.4f} no caching; whole session {v['cached']:.4f} / {v['uncached']:.4f} "
                     f"(n={v['n']})")
        elif k == "P4":
            extra = (f"{v['rate']:.2f} tok/s vs best {v['best_rate']:.2f}; output "
                     f"{'n/a (no R2 session)' if v['output_tokens'] is None else f'{v['output_tokens']:.0f}'} tok/session")
        elif k == "P5":
            extra = v.get("missing") or (f"TTFT x{v['ttft_ratio']:.3f}" if v.get("ttft_ratio") else "no ratio")
        elif k == "P6":
            extra = (f"timeouts {v['n_events'] if v['p_timeout'] else 0}, allocation failures "
                     f"{v['n_events'] if v['p_fail'] else 0} of {v['n']}; escalation USD {v['escalation_usd_day']:.2f}/day")
        if k in TIME_ROWS and v.get("usd_day") is not None:
            u = f"0.00 to {u}"
        lines.append(f"| {k} | {key} | {u} | {extra} |")
    return lines


def _audit_table(rows, out, with_assumptions=True):
    hdr = "| row | problem | measured USD/day | assumed USD/day | range USD/day | status |"
    sep = "|---|---|---:|---:|---|---|"
    if with_assumptions:
        hdr += " assumptions (value; source) | register row |"
        sep += "---|---|"
    out += [hdr, sep]
    for k in ROW_FUNCS:
        au = rows[k]["audit"]
        line = (f"| {k} | {PROBLEMS[k]} | {_fmt_usd(au['measured'])} | {_fmt_usd(au['assumed'])} | "
                f"{_rng(au['low'], au['high'])} | {au['status']} |")
        if with_assumptions:
            line += f" {_assumption_text(au)} | impact-{k} |"
        out.append(line)


def render_block(res: dict, escalation: dict) -> str:
    p = res["profile"]
    a = p["assumptions"]
    at, st = res["audit_total"], res["scenario_audit_total"]
    out = [BEGIN, "",
           f"Generated by `analysis/impact_model.py --write`. Profile: hour value USD {hour_value(p):.2f} "
           f"(impact-hour-value); ASSUMPTIONS: team {a['team_size']}, {a['agent_sessions_per_dev_per_day']} agent "
           f"sessions per developer per day, {a['turns_per_session']} turns per session, "
           f"{a['single_shot_calls_per_dev_per_day']} single-shot calls per developer per day, "
           f"{a['workdays_per_month']} workdays per month, co-runner share {a['co_runner_share']}, cloud session share "
           f"{a['cloud_session_share']}, configuration {a['device']} {a['model']} {a['runtime']}. Cloud: "
           f"{a['cloud_model']} ({a['cloud_tier']}) from {price_file(p)}, provider prompt caching ON (main case); "
           "no caching is the upper bound only. Compute time: lower bound 0 (nobody waits), upper bound at the full "
           "hour value (someone waits).", "",
           "### Impact audit (status quo: Ollama default window)", "",
           "Status: measured = USD follows from measured per-session quantities and cited prices at the profile "
           "volumes; mixed = a measured quantity whose USD also rests on an assumption beyond volume (who waits, "
           "co-runner share); assumed = the per-event cost itself is not measured; not measured = no number. "
           "Every USD/day is a projection: it scales with the assumed team size and sessions per day.", ""]
    _audit_table(res["base"], out)
    out += ["", f"impact-total: measured-only total USD/day {_fmt_usd(at['measured_usd_day'])} (USD/month "
            f"{_fmt_usd(per_month(at['measured_usd_day'], p))}); range USD/day {_rng(at['low'], at['high'])} (USD/month "
            f"{_rng(per_month(at['low'], p), per_month(at['high'], p))}) over {', '.join(at['in_range'])}; not added "
            f"(assumed or not measured): {', '.join(at['excluded_assumed']) or 'none'}.", "",
            "### Ranking (status quo profile, caching on, time at its upper bound)", ""]
    for i, k in enumerate(res["ranking"], 1):
        r = res["base"][k]
        au = r["audit"]
        out.append(f"{i}. {k} {PROBLEMS[k]}: USD/day {_fmt_usd(r['usd_day'])} (range {_rng(au['low'], au['high'])}, "
                   f"{au['status']}); confidence {r['confidence'][0]} ({r['confidence'][1]})")
    out += ["", f"Single most decisive missing measurement: {DECISIVE_MEASUREMENT}"]
    out += ["", f"### Scenario: the same team after one flag, `{a['scenario_runtime']}` (register row "
            "impact-scenario-one-flag)", "",
            "SCENARIO, depends on assumed rework time per silent wrong result (not measured). Not a lower bound.", ""]
    _audit_table(res["scenario"], out, with_assumptions=False)
    out += ["", f"Measured-only USD/day {_fmt_usd(st['measured_usd_day'])}; range {_rng(st['low'], st['high'])} over "
            f"{', '.join(st['in_range'])}; not added: {', '.join(st['excluded_assumed']) or 'none'}.", ""]
    pb = res["previous_basis"]
    if pb:
        t = pb["terms"]
        out += [f"Where the earlier USD {_fmt_usd(pb['usd_day'])}/day (USD {_fmt_usd(pb['usd_month'])}/month) came "
                "from (first impact model, before the 2026-10-14 audit; recomputed here from the same files):", "",
                "| term | USD/day | inputs | status under the audit |", "|---|---:|---|---|",
                f"| P1 | {_fmt_usd(t['P1'])} | {pb['p1_k']}/{pb['p1_n']} sessions with a silent rule/recall failure "
                f"after the window, each treated as a wrong result, x {pb['sessions_day']:.0f} sessions/day x one full "
                f"rerun {pb['redo_h']:.4f} h (median session wall at the default window) x USD {hour_value(p):.2f}/h | "
                "assumed (rework per wrong result not measured; rule breaks are not wrong tasks) |",
                f"| P2 | {_fmt_usd(t['P2'])} | smallest measured window | measured |",
                f"| P3 | {_fmt_usd(t['P3'])} | {pb['cloud_model']} ({pb['price_file']}): uncached minus cached USD "
                f"{pb['overspend_per_session']:.4f}/session x {pb['sessions_day']:.0f} sessions/day x cloud share "
                f"{a['cloud_session_share']} | replaced: caching on is now the main case (P3 row above) |",
                f"| P4 | {_fmt_usd(t['P4'])} | decode-rate gap at the full hour value | mixed (time) |",
                f"| P5 | {_fmt_usd(t['P5'])} | co-runner slowdown at the full hour value, co-runner share "
                f"{a['co_runner_share']} | mixed (time) |",
                f"| not included | | {', '.join(pb['excluded'])} not measured at this window | |"]
    out += ["", "### Rows: formula, measured inputs, missing inputs, fix, escalation", "",
            "| row | formula | measured inputs (register ids) | missing inputs | fix | needs cloud escalation? |",
            "|---|---|---|---|---|---|"]
    for k, d in ROWS.items():
        out.append(f"| {k} | {d['formula']} | {', '.join(d['measured'])} | {d['missing']} | {d['fix']} | "
                   f"{d['escalation']} |")
    out += ["", "### Escalation definitions (src/dse/router.py Router.decide, in decision order)", ""]
    out += [f"- {k}: {v}" for k, v in ESCALATIONS.items()]
    out += ["", f"### Escalation cost of the P1 fix (Router replay, E1-E4, {a['cloud_model']})", "",
            "Caching on: an escalated step reads the previous escalated step's prompt from cache when that step started "
            f"at most {CACHE_TTL_S} s earlier in the recorded local session (the default cache lifetime). Output is "
            "priced at the step's full answer budget, an upper bound.", "",
            "| model | window | cloud steps / session | steps / session | USD / session caching on | USD / session no caching | cache-hit share | sessions |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
    for (m, t), e in sorted(escalation.items()):
        f4 = lambda v: NOT_MEASURED if v is None else f"{v:.4f}"  # noqa: E731
        out.append(f"| {m} | {t} | {'n/a' if e['cloud_steps_per_session'] is None else f'{e['cloud_steps_per_session']:.1f}'} | "
                   f"{'n/a' if e['steps_per_session'] is None else f'{e['steps_per_session']:.1f}'} | "
                   f"{f4(e['usd_per_session'])} | {f4(e['usd_per_session_uncached'])} | "
                   f"{'n/a' if e['cache_hit_share'] is None else f'{e['cache_hit_share']:.0%}'} | {e['n']} |")
    out += ["", "### Per model and device grid (USD/day at the profile volumes; time rows 0 to the full hour value)", "",
            "| row | model@device/window | USD/day | inputs |", "|---|---|---:|---|"]
    for k in ROW_FUNCS:
        out += _grid_lines(k, res["base"][k], p)
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


# ── Hybrid savings v1 (operator item 3, 2026-10-21 22:30; framing per later decisions) ─────────────────────────────
# Savings are measured against ALL-CLOUD at matched quality, per device and model, on the 30 B3 v2 tasks. Inputs, each
# read only when its register row is VERIFIED, from the same data functions the rows use:
#   all-cloud quality and USD per (seed, task): all-cloud-b3-quality / all-cloud-b3-cost (results/all_cloud_b3_v1.jsonl)
#   local outcome and the window-overflow flag per (tier, seed, task): x2-evalset-tasks-real-v2-wrong-rate and
#     impact-v3-detection-value-measured (results/x2_evalset_tasks_real_v2.jsonl, qwen3:14b, evo-x2)
#   escalation cost per escalated task: cloud-redo-full-context when VERIFIED and its run finished unstopped; else the
#     all-cloud per-task cost of the same task at the same cloud tier, labelled INTERIM. results/cloud_redo_v1.jsonl
#     (cloud-redo-cost-per-wrong-task) is never used here: that run is a broken-protocol run (cloud redo diagnosis).
# Matching uses only the seeds both runs share (all-cloud has 3 seeds, the local run 5).
HYBRID_ROW = "hybrid-savings-v1"
HYBRID_VOLUMES = (100, 1000, 10000)     # agent tasks/day: a stated parameter range, not a claim
DAYS_PER_MONTH = 30
HYBRID_REQUIRED = ("all-cloud-b3-quality", "all-cloud-b3-cost", "x2-evalset-tasks-real-v2-wrong-rate",
                   "impact-v3-detection-value-measured")
HYBRID_POLICIES = (("all_cloud", "all-cloud (reference)"), ("naive_local", "naive local"),
                   ("hybrid_flag", "hybrid: escalate window-overflow flagged tasks"),
                   ("hybrid_perfect", "escalate every wrong task (ceiling, no such detector)"))
CEILING_LABEL = "ceiling, no such detector"   # operator decision 2026-10-22, item 2: never a headline
INTERIM_ESC = "interim: all-cloud per-task cost, full-history redo pending"
HYBRID_ALL_TIERS = ("ollama_default",) + B3_V2_TIERS
HYB_BEGIN, HYB_END = "<!-- hybrid_savings:begin -->", "<!-- hybrid_savings:end -->"


def all_cloud_task_outcomes(repo=REPO) -> dict:
    """{"seeds": [...], "models": {model: {(seed, task_id): (end_state_ok, usd)}}} from the all-cloud B3 run's graded
    task turns (the turns scripts/all_cloud_b3.py::summarize counts)."""
    import numbers_register as nr
    d = nr.all_cloud_b3_data(Path(repo))
    rows = d["ac"].CR.load_rows(Path(repo) / nr.ALL_CLOUD_B3_V1)
    out = {}
    for r in rows:
        if r.get("record") == "r2a_turn" and r.get("item_kind") == "task" and r.get("end_state_ok") is not None:
            k = (r["seed"], r["task_id"])
            m = out.setdefault(r["model_id"], {})
            if k in m:
                raise ValueError(f"duplicate all-cloud task turn {r['model_id']} {k}")
            m[k] = (bool(r["end_state_ok"]), float(r["usd"]))
    return {"seeds": list(d["start"]["seeds"]), "models": out}


def b3_v2_task_outcomes(repo=REPO) -> dict:
    """{tier: {(seed, task_id): (over loaded window, ended wrong)}} over task turns of completed real qwen3:14b
    sessions of the B3 v2 real run (the turns b3_v2_task_turns scores)."""
    sys.path.insert(0, str(Path(repo) / "harness"))
    import x2_r2_evalset as ev
    rows = ev.ag.read_rows(Path(repo) / B3_V2_REAL_FILE)
    if not rows:
        raise FileNotFoundError(Path(repo) / B3_V2_REAL_FILE)
    att = {s["attempt_id"] for s in ev.ag.completed_sessions(rows)
           if s.get("mode") == "real" and s["model_id"] == B3_V2_MODEL}
    out = {}
    for r in rows:
        if (r.get("record") == "r2a_turn" and r.get("attempt_id") in att and r.get("item_kind") == "task"
                and r.get("end_state_ok") is not None):
            t = out.setdefault(r["arm_id"].replace("_call2_notools", ""), {})
            k = (r["seed"], r["task_id"])
            if k in t:
                raise ValueError(f"duplicate local task turn {r['arm_id']} {k}")
            t[k] = (turn_over_window(r), r["end_state_ok"] is False)
    return out


def full_context_redo(repo=REPO) -> dict | None:
    """The full-history redo's mean USD per task and its model, only when row cloud-redo-full-context is VERIFIED and
    the run is complete across its parts (the first file plus every --resume continuation, superseded rows excluded):
    every wrong task has a row and the last part ended with a run_end that did not stop. A stop or crash in an earlier
    part does not count against completeness, because the next part resumed it. None otherwise (the caller then uses
    the INTERIM stand-in)."""
    import importlib.util
    import numbers_register as nr
    path = Path(repo) / nr.CLOUD_REDO_FULL_CONTEXT
    if not register_has(repo, "cloud-redo-full-context") or not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("cloud_redo", Path(repo) / "scripts" / "cloud_redo.py")
    cr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cr)
    paths = cr.part_files(path)
    tasks = cr.merged_rows(paths)
    start = next(r for r in cr.load_rows(paths[0]) if r.get("record") == "run_start")
    last_end = next((r for r in cr.load_rows(paths[-1]) if r.get("record") == "run_end"), None)
    if last_end is None or last_end.get("stopped") or len(tasks) < start["n_wrong_tasks"]:
        return None
    s = cr.summarize(tasks)
    return {"model": start.get("model_id"), "usd": s["mean_usd"], "n": s["n"]}



def esc_label_all(cells) -> str:
    """The escalation-cost labels of all cells, one per cloud tier, joined (a header must not show only the first)."""
    seen = []
    for c in cells:
        if c["esc_label"] not in seen:
            seen.append(c["esc_label"])
    return "; ".join(seen)

def t2s_loaded_window(repo=REPO, model=B3_V2_MODEL) -> list:
    """The window Ollama loaded for model on evo-t2s Intel CPU at its default (row t2s-r2-real-cpu's file)."""
    import numbers_register as nr
    path = Path(repo) / nr.T2S_REAL["cpu"]
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return sorted({c for r in rows if r.get("record") == "r2a_session" and r.get("model_id") == model
                   for c in r.get("loaded_context_values") or []})


def hybrid_policies(local: dict, cloud: dict, seeds, esc_usd=None) -> dict:
    """Pure: local {(seed, task): (flag, wrong)}, cloud {(seed, task): (ok, usd)}; matched on the shared keys whose seed
    is in seeds. esc_usd None = an escalated task costs that task's all-cloud USD (INTERIM); a float = that mean redo
    USD per escalated task. An escalated task takes the all-cloud outcome of the same task and seed (ASSUMED)."""
    seeds = set(seeds)
    keys = sorted(k for k in local if k in cloud and k[0] in seeds)
    n = len(keys)
    if not n:
        raise ValueError("no shared (seed, task) pairs")
    ref_usd = sum(cloud[k][1] for k in keys) / n
    ref_ok = {k: cloud[k][0] for k in keys}

    def run(escalate, reference=False):
        ok, usd, esc = {}, 0.0, 0
        for k in keys:
            flag, wrong = local[k]
            if escalate(flag, wrong):
                esc += 1
                ok[k] = cloud[k][0]
                usd += cloud[k][1] if (esc_usd is None or reference) else esc_usd
            else:
                ok[k] = not wrong
        c, u = sum(ok.values()), usd / n
        return {"correct": c, "n": n, "quality": c / n, "usd_task": u, "escalated": esc, "kept_local": n - esc,
                "silent_wrong": sum(1 for k in keys if local[k][1] and not escalate(*local[k])),
                "savings_task": ref_usd - u,
                "savings_month": {v: (ref_usd - u) * v * DAYS_PER_MONTH for v in HYBRID_VOLUMES},
                "matched": all(ok[k] >= ref_ok[k] for k in keys),
                "worse_than_cloud": sum(1 for k in keys if ref_ok[k] and not ok[k])}

    out = {"all_cloud": run(lambda f, w: True, reference=True), "naive_local": run(lambda f, w: False),
           "hybrid_flag": run(lambda f, w: f), "hybrid_perfect": run(lambda f, w: w)}
    for p in out.values():
        p["delta_points"] = 100 * (p["quality"] - out["all_cloud"]["quality"])
    wrong = [k for k in keys if local[k][1]]
    return {"policies": out, "n": n, "ref_usd": ref_usd,
            "matched_local": sum(1 for k in keys if not local[k][1] and cloud[k][0]),
            "local_wrong": len(wrong), "flag_tp": sum(1 for k in wrong if local[k][0]),
            "flagged": sum(1 for k in keys if local[k][0]),
            "wrong_inside_window": sum(1 for k in wrong if not local[k][0])}


def hybrid_p2(repo, p2_row) -> dict | None:
    """Hardware capacity beside the hybrid table: the impact-v2-P2 row's machines at its profile, and at the hybrid
    volumes under the ASSUMPTION that one agent task is one turn of a B5 R2 session (harness/policy_compare.TURNS)."""
    if not p2_row or p2_row.get("capacity") is None:
        return None
    sys.path.insert(0, str(Path(repo) / "harness"))
    import policy_compare as pc
    x, usd = p2_row["cell"], p2_row["machine_usd"]
    vol = {v: p2_capacity(v / pc.TURNS, x["default_min"], x["best_min"], usd) for v in HYBRID_VOLUMES}
    return {"row": p2_row, "volumes": vol, "turns": pc.TURNS, "fa_verified": register_has(repo, "b5-fa-control")}


def hybrid_savings(repo=REPO, p2_row=None) -> dict:
    """Per device x cloud tier x policy (see the section comment). PENDING (with the missing rows named) until every
    required register row is VERIFIED. p2_row: the impact v2 P2 row (compute_v2()["rows"]["P2"]) for hardware
    capacity."""
    missing = [c for c in HYBRID_REQUIRED if not register_has(repo, c)]
    if missing:
        return {"status": "PENDING", "pending": missing}
    ac = all_cloud_task_outcomes(repo)
    local = b3_v2_task_outcomes(repo)
    redo = full_context_redo(repo)
    seeds = ac["seeds"]
    local_seeds = sorted({s for t in local.values() for s, _ in t})
    rates = {t: sum(w for _, w in local[t].values()) / len(local[t]) for t in HYBRID_ALL_TIERS if t in local}
    best = min(rates, key=lambda t: (rates[t], t != DEFAULT_TIER))
    devices = [{"device": "evo-x2", "setting": f"{DEFAULT_TIER}, default" + (" = best" if best == DEFAULT_TIER else ""),
                "tier": DEFAULT_TIER, "label": "measured"}]
    if best != DEFAULT_TIER:
        devices.append({"device": "evo-x2", "setting": f"{best}, best", "tier": best, "label": "measured"})
    win = t2s_loaded_window(repo)
    devices.append({"device": "evo-t2s Intel CPU", "setting": f"default, loaded window {'/'.join(map(str, win))}",
                    "tier": "ollama_ctx_4096", "label": "projected"})
    cells = []
    for dv in devices:
        for m in sorted(ac["models"]):
            esc = redo["usd"] if redo and redo["model"] == m else None
            res = hybrid_policies(local[dv["tier"]], ac["models"][m], seeds, esc)
            if esc is not None:
                lab = f"measured for {m}: cloud-redo-full-context mean USD per task"
            elif redo:
                lab = (f"interim for {m}: all-cloud per-task cost (the full-history redo ran on {redo['model']} "
                       f"only)")
            else:
                lab = INTERIM_ESC
            cells.append(dict(dv, model=m, esc_label=lab, **res))
    return {"status": "VERIFIED", "cells": cells, "seeds": seeds, "local_seeds": local_seeds, "rates": rates,
            "best": best, "redo": redo, "volumes": HYBRID_VOLUMES, "t2s_window": win,
            "blade": "not measured (no B3 v2 run on the Blade)", "p2": hybrid_p2(repo, p2_row)}


# Savings as a formula over task size (operator decision 2026-10-22, item 1; row hybrid-savings-curve):
#   savings/month = B x k - X
#   B = all-cloud bill/month = all-cloud USD per task x tasks/day x 30
#   k = share kept local at matched quality (hybrid-savings-v1's "kept local at matched quality": local correct and the
#       all-cloud tier correct on the same task and seed); realizing k needs every other task found and escalated, which
#       needs a verifier for in-window wrong answers that we do not claim
#   X = escalation cost/month beyond the all-cloud bill of the escalated share = (USD of the escalated tasks per task -
#       (1 - k) x all-cloud USD per task) x tasks/day x 30; the escalated tasks' USD uses the same input
#       hybrid-savings-v1 uses (INTERIM: their all-cloud cost; the full-history redo mean once cloud-redo-full-context
#       is VERIFIED and finished unstopped), so X is nonzero today only because the escalated tasks are not average ones
# Measured point: the 30 B3 v2 tasks (3 shared seeds), evo-x2 default. Projected points: real agent task sizes from
# cloud-cost-per-real-task (SWE-Gym median, nebius-rebench-openhands median and p90, realistic caching), with k and
# X / B taken from the measured point: "projected: local quality on long tasks not measured".
CURVE_ROW = "hybrid-savings-curve"
CURVE_CASE = "realistic"
CURVE_POINTS = (("SWE-Gym", "median"), ("nebius-rebench-openhands", "median"), ("nebius-rebench-openhands", "p90"))
CURVE_PROJ_LABEL = "projected: local quality on long tasks not measured"
CURVE_REQUIRED = HYBRID_REQUIRED + ("cloud-cost-per-real-task",)


def savings_formula(bill_task, k, x_ratio, volume, days=DAYS_PER_MONTH) -> dict:
    """Pure: B = bill_task x volume x days; X = x_ratio x B; savings = B x k - X."""
    b = bill_task * volume * days
    return {"bill": b, "kept": b * k, "escalation": x_ratio * b, "savings": b * k - x_ratio * b}


def curve_inputs(local: dict, cloud: dict, seeds, esc_usd=None) -> dict:
    """Pure, same matching as hybrid_policies: k = kept local at matched quality / n; the escalated set is every other
    task; x_ratio = X / B per task = (escalated USD per task - (1 - k) x all-cloud USD per task) / all-cloud USD per
    task. esc_usd None = INTERIM (the escalated task's own all-cloud USD)."""
    seeds = set(seeds)
    keys = sorted(x for x in local if x in cloud and x[0] in seeds)
    if not keys:
        raise ValueError("no shared (seed, task) pairs")
    n = len(keys)
    kept = [x for x in keys if not local[x][1] and cloud[x][0]]
    esc = [x for x in keys if x not in set(kept)]
    c = sum(cloud[x][1] for x in keys) / n
    esc_task = sum(cloud[x][1] if esc_usd is None else esc_usd for x in esc) / n
    k = len(kept) / n
    return {"n": n, "kept": len(kept), "k": k, "bill_task": c, "esc_task": esc_task,
            "x_ratio": (esc_task - (1 - k) * c) / c}


def all_cloud_task_input_tokens(repo=REPO) -> dict:
    """{model: {(seed, task_id): billed input tokens (provider usage, cache reads and writes included)}}."""
    import numbers_register as nr
    d = nr.all_cloud_b3_data(Path(repo))
    out = {}
    for r in d["ac"].CR.load_rows(Path(repo) / nr.ALL_CLOUD_B3_V1):
        if r.get("record") == "r2a_turn" and r.get("item_kind") == "task" and r.get("end_state_ok") is not None:
            out.setdefault(r["model_id"], {})[(r["seed"], r["task_id"])] = int(r["input_tokens"])
    return out


def _cloud_task_cost_module(repo):
    import importlib.util
    spec = importlib.util.spec_from_file_location("cloud_task_cost", Path(repo) / "analysis" / "cloud_task_cost.py")
    ct = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ct)
    return ct


def hybrid_curve(repo=REPO, h=None, real=None) -> dict:
    """Per cloud tier: the measured B3 point and the projected real-task points with B, k x B, X and savings per month
    at HYBRID_VOLUMES. h: hybrid_savings() output; real: cloud_task_cost.compute() output (read when None)."""
    missing = [c for c in CURVE_REQUIRED if not register_has(repo, c)]
    if missing:
        return {"status": "PENDING", "pending": missing}
    h = h or hybrid_savings(repo)
    if h["status"] != "VERIFIED":
        return {"status": "PENDING", "pending": h["pending"]}
    real = real or _cloud_task_cost_module(repo).compute(Path(repo))
    ac, local = all_cloud_task_outcomes(repo), b3_v2_task_outcomes(repo)
    toks = all_cloud_task_input_tokens(repo)
    redo = h["redo"]
    tiers = []
    for m in sorted(ac["models"]):
        esc = redo["usd"] if redo and redo["model"] == m else None
        ci = curve_inputs(local[DEFAULT_TIER], ac["models"][m], h["seeds"], esc)
        cell = next(c for c in h["cells"] if c["model"] == m and c["tier"] == DEFAULT_TIER)
        flag = cell["policies"]["hybrid_flag"]
        keys = [x for x in local[DEFAULT_TIER] if x in ac["models"][m] and x[0] in set(h["seeds"])]
        size = statistics.median(toks[m][x] for x in keys)
        pts = [{"name": "B3 v2 tasks (30 tasks x 3 seeds), median", "size": size, "bill_task": ci["bill_task"],
                "label": "measured", "steps": None}]
        for label, q in CURVE_POINTS:
            e = real["sets"][label]
            pts.append({"name": f"{label} {q}", "size": e[f"input_{q}"], "steps": e["steps_median"] if q == "median"
                        else None, "bill_task": e["usd"][(m, CURVE_CASE)][q], "label": CURVE_PROJ_LABEL})
        pts.sort(key=lambda p: p["size"])   # a curve over task size
        for p in pts:
            p["volumes"] = {v: savings_formula(p["bill_task"], ci["k"], ci["x_ratio"], v) for v in HYBRID_VOLUMES}
        tiers.append({"model": m, **ci, "points": pts, "redo_premium": esc / ci["bill_task"] - 1 if esc is not None
                      else 0.0,
                      "esc_label": (f"measured for {m}: cloud-redo-full-context mean USD per task" if esc is not None
                                    else (f"interim for {m}: all-cloud per-task cost (the full-history redo ran on "
                                          f"{redo['model']} only)" if redo else INTERIM_ESC)),
                      "flag_matched": flag["matched"], "flag_kept": flag["kept_local"],
                      "flag_k": flag["kept_local"] / flag["n"] if flag["matched"] else 0.0})
    return {"status": "VERIFIED", "tiers": tiers, "device": f"evo-x2 {DEFAULT_TIER}", "seeds": h["seeds"],
            "case": CURVE_CASE, "hit_rate": real["hit_rate"], "volumes": HYBRID_VOLUMES}


def hybrid_curve_text(c) -> str:
    if c["status"] != "VERIFIED":
        return f"PENDING: rows {', '.join(c['pending'])} not VERIFIED"
    parts = []
    for t in c["tiers"]:
        pts = []
        for p in t["points"]:
            pts.append(f"{p['name']} [{p['label']}]: billed input {p['size']:,.0f} tokens, all-cloud USD "
                       f"{p['bill_task']:.5f}/task; " + ", ".join(
                           f"{v:,}/day B {x['bill']:,.2f} x k = {x['kept']:,.2f} - X {x['escalation']:,.2f} = saves USD "
                           f"{x['savings']:,.2f}/month" for v, x in p["volumes"].items()))
        parts.append(f"{t['model']}: k = kept local at matched quality {t['kept']}/{t['n']} = {t['k']:.3f}; X / B = "
                     f"{t['x_ratio']:.3f} (escalated tasks USD {t['esc_task']:.5f} per task over all tasks vs (1 - k) x "
                     f"{t['bill_task']:.5f}; escalation cost {t['esc_label']}); window-overflow flag (the only measured "
                     f"detector) at matched quality: "
                     + (f"k {t['flag_k']:.3f}" if t["flag_matched"] else "does not reach all-cloud quality, k 0")
                     + "; " + "; ".join(pts))
    return (f"savings/month = B x k - X (B = all-cloud bill, k = share kept local at matched quality, X = escalation "
            f"cost beyond the all-cloud bill of the escalated share); k and X / B measured on {c['device']} qwen3:14b, "
            f"B3 v2 tasks, {len(c['seeds'])} shared seeds; projected points apply them to real agent task sizes "
            f"(cloud-cost-per-real-task, {c['case']} caching, ASSUMPTION hit rate {c['hit_rate']}); k assumes every "
            f"other task is found and escalated ({CEILING_LABEL}: in-window wrong answers need a general verifier, not "
            f"claimed); the window-overflow flag column is what a measured detector reaches; 30-day month; "
            + " / ".join(parts))


# Per-device configuration (operator decision 2026-10-22, item 3; row t2s-config-comparison): evo-t2s naive Intel CPU
# default (Ollama loaded 4096) vs the recommended setting (Intel iGPU, Ollama loaded 32768), from the measured R2 real
# runs (rows t2s-r2-real-cpu, t2s-r2-real-igpu, gated variant). Kept-local share per device and model = share of R2
# sessions with no gated silent failure. Savings per device: the same formula, B from the measured all-cloud B3
# per-task cost (all-cloud-b3-cost) and X = (1 - k) x B x (escalation USD per escalated task / all-cloud USD per task -
# 1), which is 0 while the escalation cost is INTERIM (an escalated task costs its all-cloud price) and switches to the
# full-history redo mean automatically (ASSUMPTION: an R2 session's silent-failure share stands for the task share). To
# be replaced by the evo-t2s B5 results (window 2026-10-26 to 2026-11-08).
T2S_CONFIG_ROW = "t2s-config-comparison"
T2S_CONFIGS = (("cpu", "naive: Intel CPU, Ollama default"), ("igpu", "recommended: Intel iGPU, Ollama default"))
T2S_CONFIG_REQUIRED = ("t2s-r2-real-cpu", "t2s-r2-real-igpu")
T2S_CONFIG_LABEL = ("measured R2 data (evo-t2s R2 real, 5 seeds x 40 turns); to be replaced by the evo-t2s B5 results "
                    "(2026-10-26 to 2026-11-08) when they land")
# Operator decision 2026-10-22 (trade-off lead): the evo-t2s share is relabelled until the evo-t2s B5 results report task
# accuracy; it is an R2 session pass rate, not B3 task accuracy.
T2S_KEPT_LABEL = "measured by R2 session pass rate (not B3 task accuracy)"


def t2s_config_comparison(repo=REPO, curve=None, r2=None) -> dict:
    """Per evo-t2s configuration and model: loaded window, gated silent sessions, kept-local share (sessions with no
    silent failure) and savings per month per cloud tier at HYBRID_VOLUMES (needs the curve for B and X / B)."""
    missing = [c for c in T2S_CONFIG_REQUIRED if not register_has(repo, c)]
    if missing:
        return {"status": "PENDING", "pending": missing}
    import numbers_register as nr
    r2 = r2 or {b: nr.t2s_r2_real_data(Path(repo), b) for b, _ in T2S_CONFIGS}
    curve = curve if curve is not None else hybrid_curve(repo)
    tiers = ({t["model"]: t for t in curve["tiers"]} if curve.get("status") == "VERIFIED" else {})
    rows = []
    for b, name in T2S_CONFIGS:
        for model, d in sorted(r2[b].items()):
            k = (d["n"] - d["silent"]) / d["n"]
            sav = {m: {v: savings_formula(t["bill_task"], k, (1 - k) * t["redo_premium"], v) for v in HYBRID_VOLUMES}
                   for m, t in tiers.items()}
            rows.append({"backend": b, "config": name, "model": model, "loaded": d["loaded"], "silent": d["silent"],
                         "n": d["n"], "k": k, "savings": sav})
    return {"status": "VERIFIED", "rows": rows, "label": T2S_CONFIG_LABEL,
            "savings_pending": None if tiers else curve.get("pending")}


def t2s_config_text(t) -> str:
    if t["status"] != "VERIFIED":
        return f"PENDING: rows {', '.join(t['pending'])} not VERIFIED"
    parts = []
    for r in t["rows"]:
        s = "; ".join(f"vs {m}: saves USD " + ", ".join(f"{x['savings']:,.2f} at {v:,}/day" for v, x in vols.items())
                      + "/month" for m, vols in r["savings"].items()) or f"savings PENDING ({t['savings_pending']})"
        parts.append(f"{r['config']} {r['model']}: loaded window {'/'.join(map(str, r['loaded']))}, gated silent "
                     f"{r['silent']}/{r['n']} sessions, kept local ({T2S_KEPT_LABEL}) {r['n'] - r['silent']}/{r['n']} = "
                     f"{r['k']:.2f}; "
                     + s)
    return (f"evo-t2s configuration, {t['label']}; kept local, {T2S_KEPT_LABEL} = share of R2 sessions with no gated "
            f"silent failure "
            f"(t2s-r2-real-cpu, t2s-r2-real-igpu); savings = B x k - X with B the measured all-cloud B3 per-task cost "
            f"and X = (1 - k) x B x (escalation USD per escalated task / all-cloud USD per task - 1): 0 for a tier whose escalated task is costed at its all-cloud price (interim), positive for a tier "
            f"costed at the measured full-history redo price (see hybrid-savings-tradeoff for which applies) "
            f"(ASSUMPTION: R2 session share stands for the task share); 30-day month; " + " / ".join(parts))


# Measured trade-off lead (operator decision 2026-10-22, savings curve lead; row hybrid-savings-tradeoff): per device,
# cloud tier, task-size point and volume, each option's savings per month AND accuracy, so a reader sees what each
# dollar of savings costs in correct answers:
#   all-cloud: saves 0, all-cloud accuracy (all-cloud-b3-quality);
#   naive local at the device default: saves the full all-cloud bill, measured local accuracy on the shared seeds
#     (x2-evalset-tasks-real-v2-wrong-rate, as hybrid-savings-v1 uses), every wrong answer silent;
#   hybrid with the window-overflow flag: hybrid-savings-v1's hybrid_flag policy (flagged tasks escalated at the
#     escalation cost input: INTERIM until cloud-redo-full-context is VERIFIED and complete, then automatic);
#   the perfect-detector policy only as a separate field labelled CEILING_LABEL.
# Projected task-size points (cloud-cost-per-real-task, hybrid-savings-curve's points) apply each option's measured
# savings share of the bill (savings per task / all-cloud USD per task) to the point's all-cloud USD per task; their
# accuracy is the B3 accuracy carried over, labelled. Devices as hybrid-savings-v1: evo-x2 default (measured) and evo-t2s
# Intel CPU at its 4096 default (projected from evo-x2's 4096 tier).
TRADEOFF_ROW = "hybrid-savings-tradeoff"
TRADEOFF_OPTIONS = (("all_cloud", "all-cloud"), ("naive_local", "naive local at the device default"),
                    ("hybrid_flag", "hybrid: escalate window-overflow flagged tasks"))
TRADEOFF_CEILING = "hybrid_perfect"
TRADEOFF_SILENT = "wrong answers silent"
TRADEOFF_ACC_CARRIED = "B3 accuracy carried over"
TRADEOFF_DEVICE_PROJ = "projected from evo-x2's ollama_ctx_4096 tier"
TB_BEGIN, TB_END = "<!-- hybrid_tradeoff:begin -->", "<!-- hybrid_tradeoff:end -->"
OCT29_BEGIN, OCT29_END = "<!-- oct29_demo_text:begin -->", "<!-- oct29_demo_text:end -->"
TRADEOFF_HEADLINE_VOLUME = 1000


def tradeoff_option(pol, ref_usd, bill_task, volumes=HYBRID_VOLUMES, days=DAYS_PER_MONTH) -> dict:
    """Pure: one option at one task-size point; savings/month = (policy savings per task / all-cloud USD per task) x
    the point's all-cloud USD per task x volume x days (the measured point has bill_task == ref_usd)."""
    frac = pol["savings_task"] / ref_usd
    return {"correct": pol["correct"], "n": pol["n"], "quality": pol["quality"], "delta_points": pol["delta_points"],
            "escalated": pol["escalated"], "silent_wrong": pol["silent_wrong"], "share": frac,
            "savings_month": {v: frac * bill_task * v * days for v in volumes}}


def hybrid_tradeoff(h, curve) -> dict:
    """Per device x cloud tier x task-size point: the three options and the ceiling field (see the section comment).
    h: hybrid_savings() output; curve: hybrid_curve() output (supplies the task-size points)."""
    for x in (h, curve):
        if x.get("status") != "VERIFIED":
            return {"status": "PENDING", "pending": x["pending"]}
    pts = {t["model"]: t["points"] for t in curve["tiers"]}
    rows = []
    for c in h["cells"]:
        if c["device"] == "evo-x2" and c["tier"] != DEFAULT_TIER:
            continue
        for p in pts[c["model"]]:
            measured = c["label"] == "measured" and p["label"] == "measured"
            acc = ("measured" if measured else TRADEOFF_DEVICE_PROJ if p["label"] == "measured"
                   else TRADEOFF_ACC_CARRIED + ("" if c["label"] == "measured" else f", {TRADEOFF_DEVICE_PROJ}"))
            opts = {k: tradeoff_option(c["policies"][k], c["ref_usd"], p["bill_task"])
                    for k in [o for o, _ in TRADEOFF_OPTIONS] + [TRADEOFF_CEILING]}
            rows.append({"device": c["device"], "setting": c["setting"], "device_label": c["label"], "model": c["model"],
                         "point": p["name"], "point_label": p["label"], "size": p["size"], "bill_task": p["bill_task"],
                         "measured": measured, "acc_label": acc, "options": {k: opts[k] for k, _ in TRADEOFF_OPTIONS},
                         "ceiling": opts[TRADEOFF_CEILING], "esc_label": c["esc_label"]})
    return {"status": "VERIFIED", "rows": rows, "seeds": h["seeds"], "volumes": HYBRID_VOLUMES,
            "esc_label": esc_label_all(h["cells"]), "case": curve["case"], "hit_rate": curve["hit_rate"]}


def _acc(o):
    return f"{o['correct']}/{o['n']} ({100 * o['quality']:.1f}%)"


def _vols(o):
    return " / ".join(f"{o['savings_month'][v]:,.2f}" for v in o["savings_month"])


def hybrid_tradeoff_text(t) -> str:
    """The register value of hybrid-savings-tradeoff."""
    if t["status"] != "VERIFIED":
        return f"PENDING: rows {', '.join(t['pending'])} not VERIFIED"
    vol = " / ".join(f"{v:,}" for v in t["volumes"])
    parts = []
    for r in t["rows"]:
        a, nv, hf, ce = (r["options"]["all_cloud"], r["options"]["naive_local"], r["options"]["hybrid_flag"],
                         r["ceiling"])
        parts.append(f"{r['device']} ({r['setting']}) [{r['device_label']}] vs {r['model']}, {r['point']} "
                     f"[{r['point_label']}; accuracy {r['acc_label']}]: all-cloud saves 0, accuracy {_acc(a)}; naive "
                     f"local saves {_vols(nv)}, accuracy {_acc(nv)}, {nv['silent_wrong']} {TRADEOFF_SILENT}; hybrid "
                     f"flag saves {_vols(hf)}, accuracy {_acc(hf)}, escalated {hf['escalated']}, {hf['silent_wrong']} "
                     f"{TRADEOFF_SILENT}; {CEILING_LABEL}: saves {_vols(ce)}, accuracy {_acc(ce)}")
    return (f"measured trade-off, savings USD/month at {vol} tasks/day (30-day month, stated range) and accuracy per "
            f"option; qwen3:14b B3 v2 tasks vs all-cloud on the {len(t['seeds'])} shared seeds; escalation cost "
            f"{t['esc_label']}; projected task-size points from cloud-cost-per-real-task ({t['case']} caching, "
            f"ASSUMPTION hit rate {t['hit_rate']}) apply each option's measured savings share; evo-t2s "
            f"{TRADEOFF_DEVICE_PROJ}; the ceiling is a separate field, not an option / " + " / ".join(parts))



# Late escalation vs a fresh all-cloud run (operator decision 2026-10-22, item 1; row late-escalation-cost-vs-fresh):
# the full-history redo of a wrong local task (cloud-redo-full-context) costs more per task than running the same task
# fresh on the same cloud tier (all-cloud-b3-cost), because the redo re-sends the whole local session history, and it is
# incomplete because it cannot undo actions the local model already executed (cloud-redo-diagnosis class
# earlier_state_repair). Matched on (seed, task_id) where both runs have the task: the redo rows are wrong local tasks
# over 5 evo-x2 tiers and 5 seeds, the all-cloud run has all 30 tasks on 3 seeds; one task can be wrong at several
# tiers, so matched redo rows and distinct matched tasks are both reported. Also carries the evo-t2s Intel CPU 4096
# window-flag hybrid vs all-cloud from hybrid-savings-tradeoff (the redo model's tier, measured B3 point).
LATE_ROW = "late-escalation-cost-vs-fresh"
LATE_REQUIRED = ("cloud-redo-full-context", "all-cloud-b3-cost", "hybrid-savings-tradeoff")
LATE_INHERITED = "earlier_state_repair"
LATE_INHERITED_TEXT = "inherited the local model's earlier executed actions (earlier_state_repair)"


def _load_module(repo, rel, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, Path(repo) / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def late_vs_fresh_stats(redo_rows, fresh, n_wrong) -> dict:
    """Pure: redo_rows [{seed, task_id, usd, cloud_end_state_ok}], fresh {(seed, task_id): (ok, usd)} on the same cloud
    tier, n_wrong = wrong local tasks the redo set out to fix (tasks not run count as not fixed)."""
    import numbers_register as nr
    if not redo_rows or not fresh:
        raise ValueError("need redo rows and fresh all-cloud tasks")
    r_all = [r["usd"] for r in redo_rows]
    f_all = [u for _, u in fresh.values()]
    matched = [r for r in redo_rows if (r["seed"], r["task_id"]) in fresh]
    r_m = [r["usd"] for r in matched]
    f_m = [fresh[(r["seed"], r["task_id"])][1] for r in matched]
    fixed = sum(1 for r in redo_rows if r.get("cloud_end_state_ok"))
    lo, hi = nr.wilson_ci(fixed, n_wrong)

    def st(xs):
        return {"mean": statistics.fmean(xs), "median": statistics.median(xs), "n": len(xs)} if xs else None

    out = {"redo": st(r_all), "fresh": st(f_all), "m_redo": st(r_m), "m_fresh": st(f_m),
           "n_matched_keys": len({(r["seed"], r["task_id"]) for r in matched}),
           "matched_seeds": sorted({r["seed"] for r in matched}),
           "fixed": fixed, "n_wrong": n_wrong, "fix_rate": fixed / n_wrong, "ci": (lo, hi)}
    for p, a, b in (("", "redo", "fresh"), ("m_", "m_redo", "m_fresh")):
        if out[a] and out[b]:
            out[p + "ratio_mean"] = out[a]["mean"] / out[b]["mean"]
            out[p + "ratio_median"] = out[a]["median"] / out[b]["median"]
    return out


def redo_rows_and_model(repo=REPO):
    """The merged full-history redo rows (superseded rows excluded), its model and the wrong-task count."""
    import numbers_register as nr
    cr = _load_module(repo, "scripts/cloud_redo.py", "cloud_redo")
    paths = cr.part_files(Path(repo) / nr.CLOUD_REDO_FULL_CONTEXT)
    start = next(r for r in cr.load_rows(paths[0]) if r.get("record") == "run_start")
    return cr.merged_rows(paths), start["model_id"], start["n_wrong_tasks"], paths


def redo_classes(repo, paths) -> dict:
    """cloud_redo_diagnosis classes of every still-wrong redo task (the classes row cloud-redo-full-context reports)."""
    import numbers_register as nr
    cd = _load_module(repo, "analysis/cloud_redo_diagnosis.py", "cloud_redo_diagnosis")
    return cd.compute(Path(repo), nr.CLOUD_REDO_FULL_CONTEXT,
                      tuple(str(Path(p).relative_to(Path(repo)).as_posix()) for p in paths[1:]))["counts"]


def late_escalation_vs_fresh(repo=REPO, tradeoff=None, classes=None) -> dict:
    """Row late-escalation-cost-vs-fresh. tradeoff: hybrid_tradeoff() output (for the evo-t2s quote); classes: the
    still-wrong classes (read from cloud_redo_diagnosis when None)."""
    missing = [c for c in LATE_REQUIRED if not register_has(repo, c)]
    if missing:
        return {"status": "PENDING", "pending": missing}
    tasks, model, n_wrong, paths = redo_rows_and_model(repo)
    fresh = all_cloud_task_outcomes(repo)["models"][model]
    s = late_vs_fresh_stats(tasks, fresh, n_wrong)
    cls = classes if classes is not None else redo_classes(repo, paths)
    t2s = None
    if tradeoff and tradeoff.get("status") == "VERIFIED":
        r = next((x for x in tradeoff["rows"] if x["device"] != "evo-x2" and x["model"] == model
                  and x["point_label"] == "measured"), None)
        if r:
            t2s = {"device": r["device"], "setting": r["setting"], "label": r["device_label"],
                   "flag": r["options"]["hybrid_flag"], "all_cloud": r["options"]["all_cloud"]}
    return {"status": "VERIFIED", "model": model, **s, "classes": cls, "unfixed": n_wrong - s["fixed"], "t2s": t2s,
            "volumes": HYBRID_VOLUMES}


def _st(x):
    return f"mean ${x['mean']:.4f}, median ${x['median']:.4f} (n={x['n']})"


def late_text(d) -> str:
    """The register value of late-escalation-cost-vs-fresh."""
    if d["status"] != "VERIFIED":
        return f"PENDING: rows {', '.join(d['pending'])} not VERIFIED"
    cls = d["classes"]
    other = ", ".join(f"{k} {v}" for k, v in cls.items() if v and k != LATE_INHERITED)
    txt = (f"{d['model']}, USD per task (cache-priced): late full-history escalation of a wrong local task "
           f"(cloud-redo-full-context) {_st(d['redo'])} vs a fresh all-cloud run (all-cloud-b3-cost) {_st(d['fresh'])}; "
           f"ratio {d['ratio_mean']:.2f}x on the mean, {d['ratio_median']:.2f}x on the median; matched on (seed, "
           f"task_id) where both exist (seeds {', '.join(map(str, d['matched_seeds']))}): {d['m_redo']['n']} redo rows "
           f"over {d['n_matched_keys']} distinct tasks, redo {_st(d['m_redo'])} vs fresh of the same task "
           f"{_st(d['m_fresh'])}, ratio {d['m_ratio_mean']:.2f}x mean, {d['m_ratio_median']:.2f}x median; incomplete: "
           f"fixed {d['fixed']}/{d['n_wrong']} ({100 * d['fix_rate']:.1f}%), 95% Wilson CI {100 * d['ci'][0]:.1f}% to "
           f"{100 * d['ci'][1]:.1f}%; of the {d['unfixed']} unfixed, {cls.get(LATE_INHERITED, 0)} {LATE_INHERITED_TEXT}"
           + (f", {other}" if other else ""))
    t = d["t2s"]
    if t:
        vs = " / ".join(f"{t['flag']['savings_month'][v]:,.2f}" for v in d["volumes"])
        txt += (f"; {t['device']} ({t['setting']}) [{t['label']}, from evo-x2's ollama_ctx_4096 tier] vs {d['model']} "
                f"(hybrid-savings-tradeoff, B3 v2 tasks): the window-flag hybrid escalates {t['flag']['escalated']} of "
                f"{t['flag']['n']} tasks and saves USD {vs}/month at "
                f"{' / '.join(f'{v:,}' for v in d['volumes'])} tasks/day, accuracy {_acc(t['flag'])} vs all-cloud "
                f"{_acc(t['all_cloud'])}: "
                + ("it costs more than all-cloud" if t["flag"]["savings_month"][d["volumes"][0]] < 0
                   else "it does not cost more than all-cloud"))
    return txt


def ceiling_measured(local, cloud, seeds, redo_ok) -> dict:
    """Pure: escalate every wrong local task (same matching as hybrid_policies) and take the MEASURED redo outcome of
    that task at that tier (redo_ok {(seed, task_id): fixed}); a wrong task with no redo row counts as not fixed."""
    keys = sorted(k for k in local if k in cloud and k[0] in set(seeds))
    wrong = [k for k in keys if local[k][1]]
    fixed = sum(1 for k in wrong if redo_ok.get(k))
    return {"correct": len(keys) - len(wrong) + fixed, "n": len(keys), "quality": (len(keys) - len(wrong) + fixed)
            / len(keys), "escalated": len(wrong), "no_row": sum(1 for k in wrong if k not in redo_ok)}


# Route at session start (operator decision 2026-10-22, item 3; row hybrid-route-at-start): per device and workload the
# whole session runs local at the device's recommended setting, or the whole session goes to the cloud from the start.
# Rule: route local if local accuracy at the recommended setting >= all-cloud accuracy - tolerance (percentage points);
# default tolerance 0 = matched quality, 5 and 10 as a sensitivity. Local: saves the whole all-cloud bill at the local
# accuracy; cloud: saves 0 at the all-cloud accuracy (fresh all-cloud cost, no redo premium). Recommended setting:
# evo-x2 the default window (best measured, B3 task accuracy); evo-t2s the Intel iGPU (Ollama loaded 32768,
# t2s-config-comparison, qwen3:14b, labelled T2S_KEPT_LABEL), with naive = Intel CPU 4096. Compared per device and tier
# with late escalation (the window-flag hybrid and escalating every wrong task, "ceiling, no such detector") and naive
# local, from hybrid-savings-tradeoff at the measured B3 point; projected task sizes only as labelled projections.
ROUTE_ROW = "hybrid-route-at-start"
ROUTE_TOLERANCES = (0, 5, 10)
ROUTE_DEFAULT_TOL = 0
ROUTE_REQUIRED = ("hybrid-savings-tradeoff", "t2s-config-comparison", "cloud-redo-full-context")
ROUTE_BEGIN, ROUTE_END = "<!-- route_at_start:begin -->", "<!-- route_at_start:end -->"
ROUTE_X2_LABEL = "measured, B3 task accuracy"


def route_decision(local_q, cloud_q, tol) -> str:
    """Pure: 'local' if 100 x local_q >= 100 x cloud_q - tol (points), else 'cloud'."""
    return "local" if 100 * local_q >= 100 * cloud_q - tol - 1e-9 else "cloud"


def route_at_start(tradeoff, t2s_cfg, ceilings=None, tolerances=ROUTE_TOLERANCES) -> dict:
    """Per device x cloud tier. tradeoff: hybrid_tradeoff(); t2s_cfg: t2s_config_comparison(); ceilings:
    {(device, model): ceiling_measured() output} for the tier the redo measured (others use the trade-off ceiling)."""
    for x in (tradeoff, t2s_cfg):
        if x.get("status") != "VERIFIED":
            return {"status": "PENDING", "pending": x["pending"]}
    ceilings = ceilings or {}
    rec_t2s = next(r for r in t2s_cfg["rows"] if r["backend"] == "igpu" and r["model"] == B3_V2_MODEL)
    rows = []
    for r in tradeoff["rows"]:
        if r["point_label"] != "measured":
            continue
        o = r["options"]
        ac, nv = o["all_cloud"], o["naive_local"]
        proj = [p for p in tradeoff["rows"] if p["device"] == r["device"] and p["model"] == r["model"]
                and p["point_label"] != "measured"]
        bill = {v: r["bill_task"] * v * DAYS_PER_MONTH for v in tradeoff["volumes"]}
        if r["device"] == "evo-x2":
            dev, rec = "evo-x2", f"{r['setting']}"
            loc = {"correct": nv["correct"], "n": nv["n"], "quality": nv["quality"], "label": ROUTE_X2_LABEL}
            naive_name = r["setting"]
        else:
            dev = "evo-t2s"
            rec = f"Intel iGPU, Ollama default, loaded window {'/'.join(map(str, rec_t2s['loaded']))}"
            loc = {"correct": rec_t2s["n"] - rec_t2s["silent"], "n": rec_t2s["n"], "quality": rec_t2s["k"],
                   "label": T2S_KEPT_LABEL}
            naive_name = f"Intel CPU {r['setting']}, {TRADEOFF_DEVICE_PROJ}"
        ce = ceilings.get((r["device"], r["model"]))
        ceil = {"savings_month": r["ceiling"]["savings_month"],
                **({"correct": ce["correct"], "n": ce["n"], "quality": ce["quality"],
                    "label": "measured redo outcome per task and tier, mean redo USD"} if ce else
                   {"correct": r["ceiling"]["correct"], "n": r["ceiling"]["n"], "quality": r["ceiling"]["quality"],
                    "label": "all-cloud outcome and cost per escalated task (interim)"})}
        if r["device"] != "evo-x2":
            ceil["label"] += f", {TRADEOFF_DEVICE_PROJ}"
        routes = {}
        for tol in tolerances:
            d = route_decision(loc["quality"], ac["quality"], tol)
            routes[tol] = {"route": d, "savings_month": bill if d == "local" else {v: 0.0 for v in bill},
                           "acc": loc if d == "local" else {"correct": ac["correct"], "n": ac["n"],
                                                            "quality": ac["quality"]}}
        rows.append({"device": dev, "recommended": rec, "naive": naive_name, "model": r["model"], "local": loc,
                     "all_cloud": ac, "bill": bill, "gap": 100 * (ac["quality"] - loc["quality"]), "routes": routes,
                     "flag": o["hybrid_flag"], "naive_local": nv, "ceiling": ceil, "esc_label": r["esc_label"],
                     "projected": [{"point": p["point"], "label": p["point_label"],
                                    "local_savings": p["options"]["naive_local"]["savings_month"]} for p in proj]})
    return {"status": "VERIFIED", "rows": rows, "tolerances": tolerances, "volumes": tradeoff["volumes"],
            "esc_label": tradeoff["esc_label"]}


def route_ceilings(repo, h, redo_tasks, redo_model) -> dict:
    """{(device, model): ceiling_measured()} for the redo's model: every wrong local task at the cell's tier gets the
    redo outcome of that task at that tier."""
    local = b3_v2_task_outcomes(repo)
    cloud = all_cloud_task_outcomes(repo)["models"][redo_model]
    out = {}
    for c in h["cells"]:
        if c["model"] != redo_model or (c["device"] == "evo-x2" and c["tier"] != DEFAULT_TIER):
            continue
        ok = {(r["seed"], r["task_id"]): bool(r.get("cloud_end_state_ok")) for r in redo_tasks if r["tier"] == c["tier"]}
        out[(c["device"], c["model"])] = ceiling_measured(local[c["tier"]], cloud, h["seeds"], ok)
    return out


def route_at_start_full(repo, h, tradeoff, t2s_cfg) -> dict:
    """route_at_start with the measured-redo ceilings for the redo's model (PENDING with the missing rows named)."""
    missing = [c for c in ROUTE_REQUIRED if not register_has(repo, c)]
    if missing:
        return {"status": "PENDING", "pending": missing}
    if h.get("status") != "VERIFIED":
        return {"status": "PENDING", "pending": h["pending"]}
    tasks, model, _n, _p = redo_rows_and_model(repo)
    return route_at_start(tradeoff, t2s_cfg, route_ceilings(repo, h, tasks, model))


def _accp(a):
    return f"{a['correct']}/{a['n']} ({100 * a['quality']:.1f}%)"


def route_text(t) -> str:
    """The register value of hybrid-route-at-start."""
    if t["status"] != "VERIFIED":
        return f"PENDING: rows {', '.join(t['pending'])} not VERIFIED"
    vol = " / ".join(f"{v:,}" for v in t["volumes"])
    parts = []
    for r in t["rows"]:
        rt = "; ".join(f"tolerance {tol}: {x['route']}, saves {_vols(x)}, accuracy {_accp(x['acc'])}"
                       for tol, x in r["routes"].items())
        pj = ", ".join(f"{p['point']} {p['local_savings'][TRADEOFF_HEADLINE_VOLUME]:,.2f}" for p in r["projected"])
        parts.append(
            f"{r['device']} vs {r['model']}: recommended {r['recommended']}, local accuracy {_accp(r['local'])} "
            f"[{r['local']['label']}] vs all-cloud {_accp(r['all_cloud'])}, gap {r['gap']:.1f} points (routes local at "
            f"tolerance >= {max(r['gap'], 0):.1f}); {rt}; late escalation, window-flag hybrid: saves {_vols(r['flag'])}, "
            f"accuracy {_accp(r['flag'])}, escalated {r['flag']['escalated']}; late escalation of every wrong task "
            f"({CEILING_LABEL}; {r['ceiling']['label']}): saves {_vols(r['ceiling'])}, accuracy "
            f"{_accp(r['ceiling'])}; naive local ({r['naive']}): saves {_vols(r['naive_local'])}, accuracy "
            f"{_accp(r['naive_local'])}, {r['naive_local']['silent_wrong']} {TRADEOFF_SILENT}; if routed local, at "
            f"projected task sizes [{CURVE_PROJ_LABEL}] saves at {TRADEOFF_HEADLINE_VOLUME:,}/day: {pj}")
    return (f"route at session start: local if local accuracy at the recommended setting >= all-cloud accuracy - "
            f"tolerance (points; default {ROUTE_DEFAULT_TOL} = matched quality, "
            f"{', '.join(str(x) for x in t['tolerances'] if x != ROUTE_DEFAULT_TOL)} as a sensitivity), else the whole "
            f"session to the cloud at the fresh all-cloud cost (all-cloud-b3-cost) and accuracy (all-cloud-b3-quality); "
            f"savings USD/month at {vol} tasks/day (30-day month, stated range), measured B3 v2 point, 3 shared seeds; "
            f"late-escalation and naive columns from hybrid-savings-tradeoff (escalation cost {t['esc_label']}); "
            f"evo-t2s local accuracy {T2S_KEPT_LABEL} (t2s-config-comparison, qwen3:14b), its late-escalation and naive "
            f"columns at the Intel CPU 4096 default, {TRADEOFF_DEVICE_PROJ} / " + " / ".join(parts))


def render_route_block(t) -> str:
    """Markdown for docs/IMPACT.md between ROUTE_BEGIN and ROUTE_END."""
    L = [ROUTE_BEGIN, "", f"Generated by `py -3.12 analysis/impact_model.py --write-hybrid` (register row {ROUTE_ROW}). "
         "Do not edit by hand.", ""]
    if t["status"] != "VERIFIED":
        return "\n".join(L + [f"PENDING: rows {', '.join(t['pending'])} not VERIFIED.", "", ROUTE_END])
    vol = " / ".join(f"{v:,}" for v in t["volumes"])
    tols = t["tolerances"]
    L += [f"Rule: route the whole session local if local accuracy at the device's recommended setting >= all-cloud "
          f"accuracy minus a tolerance (percentage points; default {ROUTE_DEFAULT_TOL} = matched quality), otherwise "
          f"send the whole session to the cloud from the start at the fresh all-cloud cost. Savings in USD per 30-day "
          f"month at {vol} tasks/day (a stated range), measured B3 v2 point, 3 shared seeds. evo-t2s local accuracy is "
          f"{T2S_KEPT_LABEL}. evo-t2s late-escalation and naive columns are at its Intel CPU "
          f"4096 default, {TRADEOFF_DEVICE_PROJ} (no B3 run on evo-t2s). Late-escalation and naive columns from "
          f"hybrid-savings-tradeoff; escalation cost "
          f"{t['esc_label']}. The \"every wrong task\" column is a {CEILING_LABEL}.", "",
          "| Device vs cloud tier | Recommended setting: local accuracy [label] | All-cloud accuracy | Gap, points | "
          + " | ".join(f"Route at start, tolerance {x}: route / saves at {vol} / accuracy" for x in tols)
          + f" | Late escalation, window-flag hybrid: saves at {vol} / accuracy | Late escalation of every wrong task "
          f"({CEILING_LABEL}): saves at {vol} / accuracy | Naive local: saves at {vol} / accuracy |",
          "|---|---|---|---|" + "---|" * len(tols) + "---|---|---|"]
    for r in t["rows"]:
        L.append(f"| {r['device']} vs {r['model']} | {r['recommended']}: {_accp(r['local'])} [{r['local']['label']}] | "
                 f"{_accp(r['all_cloud'])} | {r['gap']:.1f} | "
                 + " | ".join(f"{x['route']} / {_vols(x)} / {_accp(x['acc'])}" for x in r["routes"].values())
                 + f" | {_vols(r['flag'])} / {_accp(r['flag'])} | {_vols(r['ceiling'])} / {_accp(r['ceiling'])} "
                 f"[{r['ceiling']['label']}] | {_vols(r['naive_local'])} / {_accp(r['naive_local'])} "
                 f"[{r['naive']}] |")
    L += ["", f"If routed local, savings at real agent task sizes ({CURVE_PROJ_LABEL}; cloud-cost-per-real-task, "
          f"realistic caching), USD/month at {TRADEOFF_HEADLINE_VOLUME:,} tasks/day:", ""]
    for r in t["rows"]:
        L.append(f"- {r['device']} vs {r['model']}: " + "; ".join(
            f"{p['point']} {p['local_savings'][TRADEOFF_HEADLINE_VOLUME]:,.2f}" for p in r["projected"]) + ".")
    return "\n".join(L + ["", ROUTE_END])


def render_tradeoff_block(t) -> str:
    """Markdown for docs/IMPACT.md between TB_BEGIN and TB_END."""
    L = [TB_BEGIN, "", f"Generated by `py -3.12 analysis/impact_model.py --write-hybrid` (register row {TRADEOFF_ROW}). "
         "Do not edit by hand.", ""]
    if t["status"] != "VERIFIED":
        return "\n".join(L + [f"PENDING: rows {', '.join(t['pending'])} not VERIFIED.", "", TB_END])
    vol = " / ".join(f"{v:,}" for v in t["volumes"])
    L += [f"Savings in USD per 30-day month at {vol} tasks/day (a stated range, not a claim about demand), and accuracy "
          f"(correct tasks) per option, matched on the {len(t['seeds'])} shared seeds. Escalation cost: "
          f"{t['esc_label']}. Projected task-size points apply each option's measured savings share of the all-cloud "
          f"bill to the point's all-cloud USD per task (cloud-cost-per-real-task, {t['case']} caching, ASSUMPTION hit "
          f"rate {t['hit_rate']}); their accuracy is the {TRADEOFF_ACC_CARRIED}. The last column is a {CEILING_LABEL}, "
          "not an option.", "",
          f"| Device [label] | Cloud tier | Task size point [label] | Accuracy label | All-cloud: saves / accuracy | "
          f"Naive local: saves at {vol} / accuracy / {TRADEOFF_SILENT} | Hybrid (window-overflow flag): saves at {vol} / "
          f"accuracy / escalated | {CEILING_LABEL[:1].upper() + CEILING_LABEL[1:]}: saves at {vol} / accuracy |",
          "|---|---|---|---|---|---|---|---|"]
    for r in t["rows"]:
        a, nv, hf, ce = (r["options"]["all_cloud"], r["options"]["naive_local"], r["options"]["hybrid_flag"],
                         r["ceiling"])
        L.append(f"| {r['device']} ({r['setting']}) [{r['device_label']}] | {r['model']} | {r['point']} "
                 f"[{r['point_label']}] | {r['acc_label']} | 0 / {_acc(a)} | {_vols(nv)} / {_acc(nv)} / "
                 f"{nv['silent_wrong']} | {_vols(hf)} / {_acc(hf)} / {hf['escalated']} | {_vols(ce)} / {_acc(ce)} |")
    return "\n".join(L + ["", TB_END])


def _tb_row(t, device, model):
    return next((r for r in t["rows"] if r["device"] == device and r["model"] == model
                 and r["point_label"] == "measured"), None)


def step4_lead(late, route, model, v) -> list:
    """The lead of demo step 4 (operator decision 2026-10-22, item 4): fixing after the fact is costly and incomplete,
    so decide up front, then the per-device decision at session start against the given cloud tier."""
    if not late or late.get("status") != "VERIFIED":
        return []
    cls = late["classes"]
    inh = cls.get(LATE_INHERITED, 0)
    rest = ", ".join(f"{n} {k.replace('_', ' ')}{'s' if n != 1 else ''}" for k, n in cls.items()
                     if n and k != LATE_INHERITED)
    out = [f"\"Fix after the fact: about {round(100 * late['redo']['mean'])} cents, "
           f"{100 * late['fix_rate']:.1f}% fixed, but executed actions cannot be undone; decide up front instead.\" "
           f"(Late full-history escalation on {late['model']}: USD {late['redo']['mean']:.4f} mean per task, "
           f"{late['ratio_mean']:.2f}x a fresh all-cloud run ({late['m_ratio_mean']:.2f}x matched on the same task and "
           f"seed, {late['m_redo']['n']} redo rows over {late['n_matched_keys']} tasks); fixed {late['fixed']}/"
           f"{late['n_wrong']} ({100 * late['fix_rate']:.1f}%), 95% CI {100 * late['ci'][0]:.1f}% to "
           f"{100 * late['ci'][1]:.1f}%; of the {late['unfixed']} not fixed, {inh} inherited actions the local model "
           f"had already executed" + (f"; {rest}" if rest else "") + ".)"]
    if route and route.get("status") == "VERIFIED":
        tol = ROUTE_DEFAULT_TOL
        for r in route["rows"]:
            if r["model"] != model:
                continue
            d = r["routes"][tol]
            out.append(f"- {r['device']}, recommended {r['recommended']}: local {_accp(r['local'])} "
                       f"[{r['local']['label']}] vs all-cloud {_accp(r['all_cloud'])}, gap {r['gap']:.1f} points. At "
                       f"matched quality the whole session goes {'local' if d['route'] == 'local' else 'to the cloud'} "
                       f"from the start; a team that accepts that gap runs it local and saves "
                       f"USD {r['bill'][v]:,.2f} a month at {v:,} tasks/day.")
        t2 = next((r for r in route["rows"] if r["device"] != "evo-x2" and r["model"] == model), None)
        if t2 and t2["flag"]["savings_month"][v] < 0:
            out.append(f"- Late escalation by window flag on evo-t2s's Intel CPU 4096 default ({TRADEOFF_DEVICE_PROJ}) "
                       f"escalates {t2['flag']['escalated']} of {t2['flag']['n']} tasks and costs USD "
                       f"{-t2['flag']['savings_month'][v]:,.2f} a month more than all-cloud at {v:,} tasks/day.")
    return out


def render_oct29_block(t, late=None, route=None) -> str:
    """Prepared text for the Oct 29 summary and the demo's story step 4, from the trade-off row; step 4 leads with the
    late-escalation finding (row late-escalation-cost-vs-fresh) and the per-device decision at session start (row
    hybrid-route-at-start) when those are given and VERIFIED."""
    rows_used = ", ".join([TRADEOFF_ROW] + ([LATE_ROW] if late and late.get("status") == "VERIFIED" else [])
                          + ([ROUTE_ROW] if route and route.get("status") == "VERIFIED" else []))
    L = [OCT29_BEGIN, "", f"Generated by `py -3.12 analysis/impact_model.py --write-hybrid` from register rows "
         f"{rows_used}. Do not edit by hand. Prepared text for the Oct 29 summary and the demo's story step 4.", ""]
    if t["status"] != "VERIFIED":
        return "\n".join(L + [f"PENDING: rows {', '.join(t['pending'])} not VERIFIED.", "", OCT29_END])
    v = TRADEOFF_HEADLINE_VOLUME
    vs = " / ".join(f"{x:,}" for x in t["volumes"])
    models = sorted({r["model"] for r in t["rows"]})
    t2s_dev = next(r["device"] for r in t["rows"] if r["device"] != "evo-x2")
    out = []
    for m in models:
        x, y = _tb_row(t, "evo-x2", m), _tb_row(t, t2s_dev, m)
        a, nv, hf = x["options"]["all_cloud"], x["options"]["naive_local"], x["options"]["hybrid_flag"]
        ya, yn, yh = y["options"]["all_cloud"], y["options"]["naive_local"], y["options"]["hybrid_flag"]
        out.append(
            f"- **vs {m}, measured on the 30 B3 tasks, {v:,} tasks/day:** all-cloud is {_acc(a)} correct. Running "
            f"everything on evo-x2 at its default saves the whole cloud bill, USD {nv['savings_month'][v]:,.2f}/month, "
            f"at {_acc(nv)} correct, and the {nv['silent_wrong']} wrong answers are silent. The window-overflow flag "
            f"escalates {hf['escalated']} tasks there, so the hybrid saves USD {hf['savings_month'][v]:,.2f}/month at "
            f"{_acc(hf)}. On evo-t2s's Intel CPU default ({TRADEOFF_DEVICE_PROJ}) naive local is {_acc(yn)} correct; "
            f"the flag escalates {yh['escalated']} tasks and reaches {_acc(yh)} (all-cloud {_acc(ya)}) but saves only "
            f"USD {yh['savings_month'][v]:,.2f}/month. A detector for in-window wrong answers would save up to USD "
            f"{x['ceiling']['savings_month'][v]:,.2f}/month on evo-x2 at {_acc(x['ceiling'])} ({CEILING_LABEL}).")
        # the largest projected MEDIAN point (not a p90 tail) as the real-task illustration
        big = [r for r in t["rows"] if r["device"] == "evo-x2" and r["model"] == m and r["point_label"] != "measured"
               and r["point"].endswith("median")]
        if big:
            b = max(big, key=lambda r: r["size"])
            bn = b["options"]["naive_local"]
            out.append(f"  At a real agent task size ({b['point']}, {CURVE_PROJ_LABEL}; accuracy {b['acc_label']}) "
                       f"naive local saves USD {bn['savings_month'][v]:,.2f}/month at {_acc(bn)}.")
    L += ["**Oct 29 summary, savings paragraph:**", "",
          f"We lead with the measured trade-off, not a ceiling: every option is shown as savings per month and the "
          f"accuracy it costs (volumes {vs} tasks/day are a stated range; escalation cost {t['esc_label']}).", ""] + out
    m = models[-1]
    x = _tb_row(t, "evo-x2", m)
    nv, hf = x["options"]["naive_local"], x["options"]["hybrid_flag"]
    L += ["", "**Demo story step 4 (one screen):**", ""]
    lead = step4_lead(late, route, m, v)
    if lead:
        L += lead + [""]
    L += [f"\"Local saves money, and the screen shows what it costs. Against {m} at {v:,} tasks/day: all local on "
          f"evo-x2 saves USD {nv['savings_month'][v]:,.2f} a month at {_acc(nv)} correct versus "
          f"{_acc(x['options']['all_cloud'])} all-cloud, and {nv['silent_wrong']} wrong answers are silent. The only "
          f"measured detector, window overflow, escalates {hf['escalated']} tasks at this window. Catching in-window "
          f"wrong answers needs a verifier we do not claim; the dotted line is that {CEILING_LABEL}.\"", "", OCT29_END]
    return "\n".join(L)


def _usd(v, nd=5):
    return f"{v:,.{nd}f}"


def hybrid_savings_text(h) -> str:
    """The register value of hybrid-savings-v1."""
    if h["status"] != "VERIFIED":
        return f"PENDING: rows {', '.join(h['pending'])} not VERIFIED"
    parts = []
    for c in h["cells"]:
        ps = []
        for k, _ in HYBRID_POLICIES:
            p = c["policies"][k]
            ps.append(f"{k}{' (' + CEILING_LABEL + ')' if k == 'hybrid_perfect' else ''} {p['correct']}/{p['n']}, USD {_usd(p['usd_task'])}/task, saves USD {_usd(p['savings_task'])}"
                      f"/task (USD/month " + ", ".join(f"{p['savings_month'][v]:,.2f} at {v:,}/day"
                                                     for v in HYBRID_VOLUMES)
                      + f"), escalated {p['escalated']}, {'matched' if p['matched'] else 'below'} all-cloud quality")
        parts.append(f"{c['device']} {c['setting']} [{c['label']}] vs {c['model']}: kept local at matched quality "
                     f"{c['matched_local']}/{c['n']}; local wrong {c['local_wrong']}, flagged {c['flagged']} "
                     f"(of them wrong {c['flag_tp']}), wrong inside the window {c['wrong_inside_window']}; "
                     + "; ".join(ps))
    p2t = ""
    p2 = h.get("p2")
    if p2:
        r, cap = p2["row"], p2["row"]["capacity"]
        p2t = (f"; P2 hardware capacity (impact-v2-P2; b5-fa-control {'VERIFIED' if p2['fa_verified'] else 'PENDING'}: "
               f"{r.get('caveat')}): machines at the P2 profile {cap['machines_default']} "
               f"default vs {cap['machines_best']} best; at the volumes (ASSUMED 1 task = 1 turn of a {p2['turns']}-turn "
               f"session) " + ", ".join(f"{v:,}/day {c['machines_default']} vs {c['machines_best']} (USD "
                                        f"{c['usd_saved']:,.0f} upfront avoided, USD {c['usd_day']:.2f}/day amortized)"
                                        for v, c in p2["volumes"].items())
               + f"; USD {r['machine_usd']:,.0f} per machine (impact-ref-machine-price, "
               f"{r.get('machine_label', price_label(None))})")
    return (f"qwen3:14b B3 v2 tasks vs all-cloud, matched on the {len(h['seeds'])} shared seeds "
            f"{', '.join(map(str, h['seeds']))} (local run has {len(h['local_seeds'])}); escalated task cost: "
            f"{esc_label_all(h['cells'])}; best evo-x2 tier {h['best']}; evo-t2s = PROJECTION from evo-x2 "
            f"ollama_ctx_4096 (no B3 v2 run on evo-t2s; loaded window from t2s-r2-real-cpu); 30-day month, volumes a "
            f"stated parameter range; " + " / ".join(parts) + f"; Blade: {h['blade']}" + p2t)


def render_hybrid_block(h) -> str:
    """Markdown for docs/IMPACT.md between HYB_BEGIN and HYB_END."""
    L = [HYB_BEGIN, "", f"Generated by `py -3.12 analysis/impact_model.py --write-hybrid` (register row {HYBRID_ROW}). "
         "Do not edit by hand.", ""]
    if h["status"] != "VERIFIED":
        return "\n".join(L + [f"PENDING: rows {', '.join(h['pending'])} not VERIFIED.", "", HYB_END])
    n = h["cells"][0]["n"]
    L += [f"Matched on the {len(h['seeds'])} seeds both runs share ({', '.join(map(str, h['seeds']))}); the local run "
          f"has {len(h['local_seeds'])} seeds and the other {len(h['local_seeds']) - len(h['seeds'])} are not used in "
          f"the table. {n} tasks per cell. Escalated task cost: {esc_label_all(h['cells'])}. Monthly savings: 30-day "
          f"month at " + ", ".join(f"{v:,}" for v in HYBRID_VOLUMES)
          + " agent tasks/day (a stated parameter range, not a claim). Labels: measured = evo-x2 B3 v2 run; projected "
          "= evo-x2's ollama_ctx_4096 tier standing in for evo-t2s.", "",
          "| Device (setting) [label] | Cloud tier | Policy | Correct | vs all-cloud | USD/task | Saves USD/task | "
          "Saves USD/month at " + " / ".join(f"{v:,}" for v in HYBRID_VOLUMES) + " tasks/day | Escalated | "
          "Matched quality |", "|---|---|---|---|---|---|---|---|---|---|"]
    for c in h["cells"]:
        for k, name in HYBRID_POLICIES:
            p = c["policies"][k]
            L.append(f"| {c['device']} ({c['setting']}) [{c['label']}] | {c['model']} | {name} | {p['correct']}/{p['n']}"
                     f" ({100 * p['quality']:.1f}%) | {p['delta_points']:+.1f} pts | {_usd(p['usd_task'])} | "
                     f"{_usd(p['savings_task'])} | " + " / ".join(f"{p['savings_month'][v]:,.2f}" for v in HYBRID_VOLUMES)
                     + f" | {p['escalated']} | {'yes' if p['matched'] else 'no'} |")
    L += [f"| Blade | n/a | all | {h['blade']} | | | | | | |", "",
          "Per cell: tasks kept local at matched quality (local correct and the all-cloud tier correct on the same task "
          "and seed), and what the window-overflow flag sees:", ""]
    for c in h["cells"]:
        L.append(f"- {c['device']} ({c['tier']}) vs {c['model']}: kept local at matched quality {c['matched_local']}/"
                 f"{c['n']}; local wrong {c['local_wrong']}; flagged {c['flagged']} (of them wrong {c['flag_tp']}); "
                 f"wrong tasks inside the window, missed by the flag: {c['wrong_inside_window']}.")
    L += ["", f"Local wrong rate per evo-x2 tier on all {len(h['local_seeds'])} seeds (picks the best tier): "
          + ", ".join(f"{t} {100 * r:.1f}%" for t, r in h["rates"].items()) + f"; best = {h['best']}.", ""]
    p2 = h.get("p2")
    if p2:
        r, cap, x = p2["row"], p2["row"]["capacity"], p2["row"]["cell"]
        cav = r.get("caveat") or ""
        L += ["P2 hardware capacity (impact-v2-P2). " + cav[:1].upper() + cav[1:]
              + ("" if p2["fa_verified"] else " (b5-fa-control still PENDING)") + f". {x['model']} default (a) {x['default_min']:.1f} "
              f"min per session vs ({x['best']}) {x['best_name']} {x['best_min']:.1f} min (measured), one session at a "
              "time per machine.", "",
              f"| Agent tasks/day (ASSUMED 1 task = 1 turn of a {p2['turns']}-turn session) | Machines, default | "
              "Machines, best | Upfront hardware avoided, USD (list price) | Amortized USD/day (secondary) |",
              "|---|---|---|---|---|",
              f"| impact-v2-P2 profile | {cap['machines_default']} | {cap['machines_best']} | {cap['usd_saved']:,.0f} | "
              f"{cap['usd_day']:.2f} |"]
        for v, c in p2["volumes"].items():
            L.append(f"| {v:,} | {c['machines_default']} | {c['machines_best']} | {c['usd_saved']:,.0f} | "
                     f"{c['usd_day']:.2f} |")
        L += ["", f"Machine price USD {r['machine_usd']:,.0f} (impact-ref-machine-price: "
              f"{r.get('machine_label', price_label(None))}); "
              f"{cap['life_years']:.0f}-year life (ASSUMED) for the amortized column.", ""]
    L.append(HYB_END)
    return "\n".join(L)


CFG_BEGIN, CFG_END = "<!-- t2s_config:begin -->", "<!-- t2s_config:end -->"
CURVE_BEGIN, CURVE_END = "<!-- hybrid_curve:begin -->", "<!-- hybrid_curve:end -->"


def render_t2s_config_block(t) -> str:
    L = [CFG_BEGIN, "", f"Generated by `py -3.12 analysis/impact_model.py --write-hybrid` (register row {T2S_CONFIG_ROW}). "
         "Do not edit by hand.", ""]
    if t["status"] != "VERIFIED":
        return "\n".join(L + [f"PENDING: rows {', '.join(t['pending'])} not VERIFIED.", "", CFG_END])
    models = sorted({m for r in t["rows"] for m in r["savings"]})
    L += [f"Label: {t['label']}. Kept local, {T2S_KEPT_LABEL} = share of R2 sessions with no gated silent failure, "
          "until the evo-t2s B5 results report task accuracy. Savings per month "
          "= B x k - X, B = the measured all-cloud B3 per-task cost of the tier x tasks/day x 30, X = the escalation premium "
          "over the all-cloud price: 0 for a tier costed at its all-cloud price (interim), positive for a tier costed at "
          "the measured full-history redo price (ASSUMPTION: the R2 session share stands for the task share).", "",
          f"| evo-t2s setting | Model | Loaded window | Gated silent failures | Kept local, {T2S_KEPT_LABEL} | "
          + " | ".join(f"Saves USD/month vs {m} at " + " / ".join(f"{v:,}" for v in HYBRID_VOLUMES) for m in models)
          + " |", "|---|---|---|---|---|" + "---|" * len(models)]
    for r in t["rows"]:
        L.append(f"| {r['config']} | {r['model']} | {'/'.join(map(str, r['loaded']))} | {r['silent']}/{r['n']} | "
                 f"{r['n'] - r['silent']}/{r['n']} ({r['k']:.2f}) | "
                 + " | ".join(" / ".join(f"{r['savings'][m][v]['savings']:,.2f}" for v in HYBRID_VOLUMES)
                              if m in r["savings"] else "PENDING" for m in models) + " |")
    return "\n".join(L + ["", CFG_END])


def render_curve_block(c) -> str:
    L = [CURVE_BEGIN, "", f"Generated by `py -3.12 analysis/impact_model.py --write-hybrid` (register row {CURVE_ROW}). "
         "Do not edit by hand.", ""]
    if c["status"] != "VERIFIED":
        return "\n".join(L + [f"PENDING: rows {', '.join(c['pending'])} not VERIFIED.", "", CURVE_END])
    L += [f"savings/month = B x k - X. k and X / B measured on {c['device']} (qwen3:14b, B3 v2 tasks, "
          f"{len(c['seeds'])} shared seeds). Projected points are labelled \"{CURVE_PROJ_LABEL}\"; cloud cost per task from "
          f"cloud-cost-per-real-task, {c['case']} caching (ASSUMPTION hit rate {c['hit_rate']}). k assumes every other "
          f"task is found and escalated ({CEILING_LABEL}).", ""]
    for t in c["tiers"]:
        L += [f"**{t['model']}**: k = {t['kept']}/{t['n']} = {t['k']:.3f}; X / B = {t['x_ratio']:.3f} (escalation cost "
              f"{t['esc_label']}); window-overflow flag at matched quality: "
              + (f"k {t['flag_k']:.3f}" if t["flag_matched"] else "does not reach all-cloud quality (k 0)") + ".", "",
              "| Task size point [label] | Billed input tokens per task | All-cloud USD per task | "
              + " | ".join(f"B / k x B / X / saves, USD/month at {v:,}/day" for v in HYBRID_VOLUMES) + " |",
              "|---|---|---|" + "---|" * len(HYBRID_VOLUMES)]
        for p in t["points"]:
            L.append(f"| {p['name']} [{p['label']}] | {p['size']:,.0f} | {p['bill_task']:.5f} | "
                     + " | ".join(f"{x['bill']:,.2f} / {x['kept']:,.2f} / {x['escalation']:,.2f} / {x['savings']:,.2f}"
                                  for x in p["volumes"].values()) + " |")
        L.append("")
    return "\n".join(L + [CURVE_END])


def _replace_block(text, begin, end, block):
    i, j = text.index(begin), text.index(end) + len(end)
    return text[:i] + block + text[j:]


def write_hybrid_doc(repo=REPO, block=None, extra=()):
    """Replace the hybrid block, and each (begin, end, block) in extra, in docs/IMPACT.md."""
    path = Path(repo) / "docs" / "IMPACT.md"
    text = _replace_block(path.read_text(encoding="utf-8"), HYB_BEGIN, HYB_END, block)
    for b, e, blk in extra:
        text = _replace_block(text, b, e, blk)
    path.write_text(text, encoding="utf-8")


def main(argv=None):
    argv = argv or sys.argv[1:]
    if "--write-hybrid" in argv:
        h = hybrid_savings(REPO, compute_v2(REPO)["rows"]["P2"])
        print(hybrid_savings_text(h))
        c = hybrid_curve(REPO, h) if h["status"] == "VERIFIED" else {"status": "PENDING", "pending": h["pending"]}
        t = t2s_config_comparison(REPO, c)
        tb = hybrid_tradeoff(h, c)
        print(hybrid_tradeoff_text(tb))
        late = late_escalation_vs_fresh(REPO, tb)
        rt = route_at_start_full(REPO, h, tb, t)
        print(late_text(late))
        print(route_text(rt))
        write_hybrid_doc(block=render_hybrid_block(h), extra=((CFG_BEGIN, CFG_END, render_t2s_config_block(t)),
                                                              (CURVE_BEGIN, CURVE_END, render_curve_block(c)),
                                                              (TB_BEGIN, TB_END, render_tradeoff_block(tb)),
                                                              (ROUTE_BEGIN, ROUTE_END, render_route_block(rt)),
                                                              (OCT29_BEGIN, OCT29_END, render_oct29_block(tb, late, rt))))
        print("wrote docs/IMPACT.md route at start, trade-off, Oct 29 text, hybrid savings, evo-t2s configuration and curve blocks")
        return
    res = compute_all()
    p = res["profile"]
    for k in ROW_FUNCS:
        print(f"{k} {PROBLEMS[k]}: {row_value(res['base'][k], p)}")
    at = res["audit_total"]
    print(f"total: measured-only USD/day {_fmt_usd(at['measured_usd_day'])}, range {_rng(at['low'], at['high'])} "
          f"(not added {at['excluded_assumed']})")
    print("ranking:", " > ".join(res["ranking"]))
    print("scenario:", scenario_value(res))
    print("session time:", session_time_value(res))
    if "--write" in argv:
        write_doc(block=render_block(res, escalation_table()))
        print("wrote docs/IMPACT.md generated block")


if __name__ == "__main__":
    main()
