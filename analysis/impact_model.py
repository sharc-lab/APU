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


def main(argv=None):
    argv = argv or sys.argv[1:]
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
