"""Business metrics (offline): what a customer team pays today, and what the router would change, per customer
profile. Plan v2 Part C1 (operator plan, 2026-10-13), revised by the operator decisions of 2026-10-14 (A to G).

No machine time, no API call. Built only from the impact model (plan item A3, analysis/impact_model.py), whose
functions are reused unchanged:

  - per-session time costs P2, P4, P5 and the P1/P7 rates at a given device and window: impact_model.p1 .. p7;
  - cloud cost of a whole session sent to the cloud model, provider caching on (main case) and off (upper bound):
    impact_model.r2_cloud_overspend, on the shared Anthropic table (claude-sonnet-5-5);
  - the router's own decisions: impact_model.router_escalation (R2 sessions replayed through src/dse/router.py,
    escalations E1-E4; each escalated step priced with caching on between escalated steps, and without caching as
    the upper bound; output at the step's full answer budget).

Operator decisions applied here:
  B  caching on is the main case for every cloud figure; no caching is the upper bound only.
  C  local compute time is valued from 0 (nobody waits) to the full hour value (someone waits); P1 is not measured
     (no B3 task runs yet), so no main-table figure contains a P1 dollar value.
  D  no headline figure rests on the +A credit: every router saving (vs the naive default, vs all-cloud, vs today's
     mix, and the hardware recommendation) prints "pending B5" in the main table. The +A computation is kept only in
     the clearly labelled SCENARIO section (scenario_*), with the assumptions it rests on.
  F  the 8 GB laptop profile is dropped (results/business_profiles_20261014.json, "dropped_profiles").
  G  "steps kept local" is traced: at num_ctx 4096 and 8192 the replay escalates the same 16 of 80 steps in every
     session (the 8 canary-check turns, both calls each; src/dse/r2_replay.py needs_turns), a property of the R2
     session script, not of the profile. steps_local_trace() states it from the replay itself.

The +A assumption (scenario only): at a window where the runtime alone loses context, the router's residual P1 and
P7 are zero, credited only where the pre-registered mitigation measured 'recovered' for that model and window
(register row R2-mitigation-v1-verdict). The scenario also values P1 with the impact model's ASSUMED rerun proxy and
time at the full hour value.

Profiles: results/business_profiles_20261014.json. Every profile field is an ASSUMPTION. The hour value is register
row impact-hour-value; its range uses the same BLS page's percentiles.

Labels: measured = per-session quantity from result files; projection = measured per-session quantity x assumed
volumes; scenario = additionally rests on +A, the P1 rerun proxy, or time at the full hour value as a saving.

Usage: py -3.12 analysis/business_metrics.py            prints the per-profile values
       py -3.12 analysis/business_metrics.py --write    also rewrites docs/BUSINESS_METRICS.md's generated block
"""
from __future__ import annotations

import copy
import json
import sys
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "analysis", REPO / "harness"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import impact_model as im  # noqa: E402

PROFILES_FILE = "results/business_profiles_20261014.json"
SESSION_ROWS = ("P1", "P2", "P4", "P5", "P7")
TIME_ROWS = ("P2", "P4", "P5")
ROUTER_ZERO = ("P1", "P7")   # residual cost assumed zero under the router (scenario only)
NOT_MEASURED = im.NOT_MEASURED
PENDING = "pending B5"


# ── profiles ────────────────────────────────────────────────────────────────────────────────────────────────────

def load_profiles(repo=REPO, path=None) -> dict:
    return json.loads(((Path(path) if path else repo / PROFILES_FILE)).read_text(encoding="utf-8"))


def hour_values(doc, repo=REPO) -> dict:
    """{"p10", "median", "p90"} USD/h: the median is impact-hour-value; p10/p90 are the page's annual percentiles
    divided by the hours per year implied by the page's own median pair (derivation in the JSON)."""
    med = im.hour_value(im.load_profile(repo))
    r = doc["hour_value"]["range"]
    hours = r["group_median_annual_usd"] / r["group_median_hourly_usd"]
    return {"p10": r["p10_annual_usd"] / hours, "median": med, "p90": r["p90_annual_usd"] / hours,
            "hours_per_year": hours}


def sessions_day(a) -> float:
    return a["team_size"] * a["agent_sessions_per_dev_per_day"]


def _cloud_prof(a) -> dict:
    return {"hour_value": {"usd_per_hour": 1.0},
            "assumptions": {"cloud_model": a["cloud_model"], "cloud_tier": a["cloud_tier"],
                            "cloud_price_file": a.get("cloud_price_file", im.PRICES_FILE)}}


def _impact_profile(a, hv, device, tier) -> dict:
    """An impact_model profile dict for one (device, window) at the business profile's volumes."""
    keys = ("team_size", "agent_sessions_per_dev_per_day", "turns_per_session", "single_shot_calls_per_dev_per_day",
            "workdays_per_month", "co_runner_share", "cloud_session_share", "model", "cloud_model", "cloud_tier")
    pa = {k: a[k] for k in keys}
    pa.update(device=device, runtime=tier, scenario_runtime=tier,
              cloud_price_file=a.get("cloud_price_file", im.PRICES_FILE))
    return {"hour_value": {"usd_per_hour": hv}, "assumptions": pa}


# ── shared inputs (computed once per repo) ──────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=2)
def _inputs(repo_str: str) -> dict:
    repo = Path(repo_str)
    wall = im.session_wallclock(repo)
    rates = im.context_loss_rates(repo)
    b3 = im.b3_task_windows(repo)
    return {"P1": {"rates": rates, "wall": wall, "b3": b3}, "P2": {"deltas": im.oversize_turn_delta(repo)},
            "P4": {"rates": im.decode_rates(repo), "wall": wall}, "P5": {"ratios": im.co_runner_ratios(repo),
                                                                        "wall": wall},
            "P7": {"rates": rates}}


@lru_cache(maxsize=32)
def _escalation(repo_str: str, model: str, tier: str, cloud_model: str, cloud_tier: str, price_file: str) -> dict:
    prof = _cloud_prof({"cloud_model": cloud_model, "cloud_tier": cloud_tier, "cloud_price_file": price_file})
    return im.router_escalation(Path(repo_str), model, tier, prof)


@lru_cache(maxsize=4)
def _cloud_sessions(repo_str: str, cloud_model: str, cloud_tier: str, price_file: str) -> dict:
    prof = _cloud_prof({"cloud_model": cloud_model, "cloud_tier": cloud_tier, "cloud_price_file": price_file})
    return im.r2_cloud_overspend(Path(repo_str), prof)


def escalation(a, tier, repo=REPO) -> dict:
    return _escalation(str(repo), a["model"], tier, a["cloud_model"], a["cloud_tier"],
                       a.get("cloud_price_file", im.PRICES_FILE))


def cloud_sessions(a, repo=REPO) -> dict | None:
    return _cloud_sessions(str(repo), a["cloud_model"], a["cloud_tier"],
                           a.get("cloud_price_file", im.PRICES_FILE)).get(a["model"])


# ── pure arithmetic (synthetic-testable) ────────────────────────────────────────────────────────────────────────

def policy_total(rows: dict, cloud_usd_day: float = 0.0) -> dict:
    """Sum of the priced rows plus cloud spend; rows that are None are listed, never filled in."""
    known = {k: v for k, v in rows.items() if v is not None}
    return {"usd_day": sum(known.values()) + cloud_usd_day, "time_usd_day": sum(known.values()),
            "cloud_usd_day": cloud_usd_day, "not_measured": sorted(k for k, v in rows.items() if v is None)}


def time_range(rows: dict) -> tuple:
    """(low, high) USD/day of the local compute-time rows: 0 (nobody waits) to their full-hour-value sum. None when
    no time row is priced."""
    known = [rows[k] for k in TIME_ROWS if rows.get(k) is not None]
    if not known:
        return None, None
    return 0.0, sum(known)


def router_rows(naive_rows: dict, mitigation: str | None) -> tuple[dict, bool]:
    """SCENARIO ONLY. Router rows at the same device and window. P1 and P7 are set to 0 only when that needs no
    assumption (the naive value is already 0) or when the pre-registered mitigation at this (model, window) measured
    'recovered' (register row R2-mitigation-v1-verdict); then assumption_used is True. Otherwise the router gets no
    credit: P1 and P7 keep their naive values. Returns (rows, assumption_used)."""
    out = dict(naive_rows)
    needs = any(out.get(k) != 0 for k in ROUTER_ZERO if k in out)
    if not needs:
        return out, False
    if mitigation != "recovered":
        return out, False
    for k in ROUTER_ZERO:
        if k in out:
            out[k] = 0.0
    return out, True


@lru_cache(maxsize=2)
def _mitigation(repo_str: str) -> dict:
    """{(model, tier): verdict} from the R2 mitigation report behind register row R2-mitigation-v1-verdict."""
    import numbers_register as nr
    try:
        _mr, rep = nr._r2_mitigation_report(Path(repo_str))
    except FileNotFoundError:
        return {}
    out = {}
    for key, c in rep["cells"].items():
        tier, model = key.split("|")
        out[(model, im.tier_of(tier))] = c["verdict"]["verdict"]
    return out


def mix_usd_day(share, all_cloud_day, naive_day) -> float:
    """Today's mix: `share` of sessions on the cloud model, the rest local at the runtime default."""
    return share * all_cloud_day + (1.0 - share) * naive_day


def steps_local_share(esc) -> float | None:
    if not esc or not esc.get("steps_per_session"):
        return None
    return 1.0 - esc["cloud_steps_per_session"] / esc["steps_per_session"]


def steps_local_trace(esc) -> str:
    """Where a 'steps kept local' share comes from: the replay's per-session cloud-step counts. When every session
    escalated the same number of steps, say so (a constant of the R2 session script, not of the profile)."""
    if not esc or not esc.get("n"):
        return "no replay"
    cv, sv = esc.get("cloud_steps_values") or [], esc.get("steps_values") or []
    if cv == [0]:
        return f"no step escalated in any of {esc['n']} replayed sessions"
    if len(cv) == 1 and len(sv) == 1:
        return f"{cv[0]} of {sv[0]} steps escalated in every one of {esc['n']} replayed sessions (same count each)"
    return f"cloud steps per session {cv} of {sv} over {esc['n']} replayed sessions"


# ── one configuration ───────────────────────────────────────────────────────────────────────────────────────────

def session_rows(a, hv, device, tier, repo=REPO, scenario=False) -> dict:
    """USD/day of P1, P2, P4, P5, P7 at (device, tier) for the profile's volumes, via impact_model's row functions.
    P1 is None (not measured) unless scenario=True, which uses the impact model's ASSUMED rerun proxy. Time rows are
    at the full hour value (the upper end of their 0-to-full range)."""
    inp = _inputs(str(repo))
    own = device == "evo-x2"
    sess_dev = device if own else a.get("session_proxy_device", device)
    prof = _impact_profile(a, hv, sess_dev, tier)
    out = {}
    r1 = im.p1(repo, prof, None, inp["P1"])
    out["P1"] = (r1["cell"] or {}).get("scenario_usd_day") if scenario else r1["usd_day"]
    for k in ("P2", "P7"):
        out[k] = im.ROW_FUNCS[k](repo, prof, None, inp[k])["usd_day"]
    out["P5"] = im.p5(repo, prof, None, inp["P5"])["usd_day"] if own else None
    p4_dev = device if own else f"{device}@{im.tier_ctx(tier)}"
    out["P4"] = im.p4(repo, _impact_profile(a, hv, p4_dev, tier), None, inp["P4"])["usd_day"]
    return {k: out[k] for k in SESSION_ROWS}


def config(a, hv, device, tier, repo=REPO, scenario=False) -> dict:
    """Naive (runtime alone) rows and the router replay at one (device, window); with scenario=True also the +A
    router rows and totals."""
    naive = session_rows(a, hv, device, tier, repo, scenario)
    esc = escalation(a, tier, repo)
    spd = sessions_day(a)
    out = {"device": device, "tier": tier, "naive_rows": naive, "naive": policy_total(naive),
           "time_range": time_range(naive), "escalation": esc, "router_ok": esc["usd_per_session"] is not None,
           "local_share": steps_local_share(esc), "local_trace": steps_local_trace(esc),
           "router_cloud_day": None if esc["usd_per_session"] is None else spd * esc["usd_per_session"],
           "router_cloud_day_uncached": None if esc["usd_per_session_uncached"] is None else
           spd * esc["usd_per_session_uncached"], "proxy": device != "evo-x2"}
    if scenario:
        mit = _mitigation(str(repo)).get((a["model"], tier))
        rrows, used = router_rows(naive, mit)
        out.update(mitigation=mit, router_rows=rrows, router=policy_total(rrows, out["router_cloud_day"] or 0.0),
                   assumption_used=used or device != "evo-x2")
    return out


def hours(usd_day, hv):
    return None if usd_day is None else usd_day / hv


# ── one profile: measured / projection facts (main table) ───────────────────────────────────────────────────────

def single_shot(repo=REPO) -> dict:
    """pareto-sample-recommendation-v2's choice and its saving per 1k single-shot steps vs the all-cloud STUB."""
    import numbers_register as nr
    P, pts = nr._pareto_points(repo)
    rec = P.recommend(pts, **nr.PARETO_SAMPLE, max_timeout_share=nr.PARETO_SAMPLE_MAX_TIMEOUT_SHARE)
    ch = rec.chosen
    return {"config": None if ch is None else f"{ch.machine} {ch.model} {ch.runtime}",
            "usd_per_1k_saved": rec.savings_vs_all_cloud_usd_per_1k, "stub": rec.stub}


def profile_metrics(a, hv, repo=REPO) -> dict:
    """Main-table facts for one profile: no +A, no P1 dollar value, caching on (no caching as the upper bound),
    local time as a 0-to-full-hour-value range. Router savings are PENDING (plan item B5)."""
    spd = sessions_day(a)
    wd = a["workdays_per_month"]
    cs = cloud_sessions(a, repo)
    naive = config(a, hv, a["device"], a["default_tier"], repo)
    windows = [config(a, hv, a["device"], t, repo) for t in a["router_tiers"]]
    m = lambda v: None if v is None else v * wd  # noqa: E731
    lo, hi = naive["time_range"]
    share = a["cloud_session_share"]
    all_c = None if not cs else spd * cs["cached"]
    all_u = None if not cs else spd * cs["uncached"]
    return {
        "assumptions": a, "hour_value": hv, "sessions_month": spd * wd, "all_cloud_per_session": cs,
        "naive": naive, "windows": windows,
        "month": {
            "all_cloud_cached": m(all_c), "all_cloud_uncached": m(all_u),
            "naive_time_low": m(lo), "naive_time_high": m(hi), "naive_hours": m(hours(hi, hv)),
            # today's mix: cloud fees on the cloud share (caching on / off) plus local time on the rest (0 to full)
            "mix_cloud_cached": m(None if all_c is None else share * all_c),
            "mix_cloud_uncached": m(None if all_u is None else share * all_u),
            "mix_time_high": m(None if hi is None else (1 - share) * hi),
        },
        "savings": {"vs_naive": PENDING, "vs_all_cloud": PENDING, "vs_mix": PENDING, "recommendation": PENDING},
    }


# ── SCENARIO (+A): kept only in the labelled scenario section ───────────────────────────────────────────────────

def scenario_metrics(a, hv, repo=REPO, ss=None) -> dict:
    """SCENARIO: the router window minimising (time at the full hour value + router cloud, caching on), with the +A
    credit and P1 at the ASSUMED rerun proxy. Every figure here rests on those assumptions."""
    spd = sessions_day(a)
    wd = a["workdays_per_month"]
    cs = cloud_sessions(a, repo)
    naive = config(a, hv, a["device"], a["default_tier"], repo, scenario=True)
    nm = set(naive["naive"]["not_measured"])
    ok = lambda c: c["router_ok"] and set(c["router"]["not_measured"]) <= nm  # noqa: E731
    own = [c for c in (config(a, hv, a["device"], t, repo, scenario=True) for t in a["router_tiers"]) if ok(c)]
    best = min(own, key=lambda c: c["router"]["usd_day"])
    alts = [c for c in (config(a, hv, d, t, repo, scenario=True) for d, ts in (a.get("alternatives") or {}).items()
                        for t in ts) if ok(c)]
    rec = min(own + alts, key=lambda c: c["router"]["usd_day"])
    n_day, r_day = naive["naive"]["usd_day"], best["router"]["usd_day"]
    share = a["cloud_session_share"]
    all_c = None if not cs else spd * cs["cached"]
    all_u = None if not cs else spd * cs["uncached"]
    mix_c = None if all_c is None else mix_usd_day(share, all_c, n_day)
    mix_u = None if all_u is None else mix_usd_day(share, all_u, n_day)
    rc, ru = best["router_cloud_day"], best["router_cloud_day_uncached"]
    ss = ss or single_shot(repo)
    m = lambda v: None if v is None else v * wd  # noqa: E731
    return {"assumptions": a, "hour_value": hv, "naive": naive, "router": best, "candidates": own,
            "recommendation": rec, "alternatives": alts, "single_shot": ss,
            "month": {"save_vs_naive": m(n_day - r_day),
                      # the same comparison if nobody waits for local compute: time is worth 0 on both sides, the
                      # P1 rerun proxy drops out with it, and the router is left with its cloud fees
                      "save_vs_naive_if_nobody_waits": m(-(rc or 0.0)),
                      "hours_saved_vs_naive": m(hours(naive["naive"]["time_usd_day"] - best["router"]["time_usd_day"], hv)),
                      "cloud_avoided_cached": m(None if all_c is None else all_c - rc),
                      "cloud_avoided_uncached": m(None if all_u is None else all_u - ru),
                      "save_vs_mix_cached": m(None if mix_c is None else mix_c - r_day),
                      "save_vs_mix_uncached": m(None if mix_u is None else mix_u - (best["router"]["time_usd_day"] + ru)),
                      "rec_save_vs_naive": m(n_day - rec["router"]["usd_day"]),
                      "single_shot_saved": None if ss["usd_per_1k_saved"] is None else
                      a["team_size"] * a["single_shot_calls_per_dev_per_day"] * wd * ss["usd_per_1k_saved"] / 1000.0}}


def sensitivity(doc, hvs, repo=REPO, ss=None) -> list:
    """SCENARIO: one assumption varied at a time around the base profile."""
    s = doc["sensitivity"]
    base = doc["profiles"][s["base_profile"]]["assumptions"]
    out = []
    for param in ("team_size", "agent_sessions_per_dev_per_day", "cloud_session_share", "hour_value"):
        for v in s[param]:
            a = copy.deepcopy(base)
            hv = hvs["median"]
            if param == "hour_value":
                hv = hvs[v]
            else:
                a[param] = v
            r = scenario_metrics(a, hv, repo, ss)
            out.append({"param": param, "value": v if param != "hour_value" else f"{v} (USD {hv:.2f}/h)",
                        "router_tier": r["router"]["tier"], **{k: r["month"][k] for k in (
                            "save_vs_naive", "save_vs_naive_if_nobody_waits", "cloud_avoided_cached", "cloud_avoided_uncached",
                            "save_vs_mix_cached", "save_vs_mix_uncached", "hours_saved_vs_naive")}})
    return out


def compute_all(repo=REPO, doc=None) -> dict:
    doc = doc or load_profiles(repo)
    hvs = hour_values(doc, repo)
    ss = single_shot(repo)
    res = {name: profile_metrics(p["assumptions"], hvs["median"], repo) for name, p in doc["profiles"].items()}
    scen = {name: scenario_metrics(p["assumptions"], hvs["median"], repo, ss) for name, p in doc["profiles"].items()}
    return {"doc": doc, "hour_values": hvs, "profiles": res, "scenario": scen, "single_shot": ss,
            "sensitivity": sensitivity(doc, hvs, repo, ss)}


# ── formatting ──────────────────────────────────────────────────────────────────────────────────────────────────

def usd(v, nd=0):
    return NOT_MEASURED if v is None else f"{v:,.{nd}f}"


def rng(lo, hi, nd=0):
    if lo is None or hi is None:
        return NOT_MEASURED
    return f"{usd(lo, nd)} to {usd(hi, nd)}"


def _window(r, tier):
    return next((c for c in r["windows"] if c["tier"] == tier), None)


def profile_value(name, r) -> str:
    """One register value string for a profile (business-* rows): measured and projection facts only; router
    savings pending B5."""
    a, m, nv = r["assumptions"], r["month"], r["naive"]
    cs = r["all_cloud_per_session"] or {}
    wins = "; ".join(
        f"{c['tier']}: {usd(None if c['local_share'] is None else 100 * c['local_share'], 1)}% steps local ({c['local_trace']}), "
        f"router cloud USD {usd(c['escalation']['usd_per_session'], 3)}/session caching on, "
        f"{usd(c['escalation']['usd_per_session_uncached'], 3)} no caching"
        for c in r["windows"] if c["router_ok"])
    return (f"{name} (ASSUMED team {a['team_size']}, {a['agent_sessions_per_dev_per_day']} sessions/dev/day, cloud "
            f"share {a['cloud_session_share']}; {a['device']} {a['model']}; cloud {a['cloud_model']}): naive default "
            f"{nv['tier']}; all-cloud USD/session {usd(cs.get('cached'), 3)} caching on, {usd(cs.get('uncached'), 3)} "
            f"no caching (measured, n={cs.get('n')}); USD/month (projection): all-cloud {usd(m['all_cloud_cached'])} "
            f"caching on (no caching {usd(m['all_cloud_uncached'])}); naive local compute time {rng(m['naive_time_low'], m['naive_time_high'])} "
            f"(0 = nobody waits, upper = full hour value; {usd(m['naive_hours'], 1)} h; P1 not measured, P7 "
            f"{usd(nv['naive_rows']['P7'], 2)}); router replay per window (measured): {wins}; router savings vs naive, vs "
            f"all-cloud, vs today's mix and the hardware recommendation: {PENDING}")


def scenario_value(res) -> str:
    out = []
    for name, s in res["scenario"].items():
        m = s["month"]
        out.append(f"{name}: router window {s['router']['tier']}, vs naive {usd(m['save_vs_naive'])} (if nobody waits "
                   f"{usd(m['save_vs_naive_if_nobody_waits'])}), hours "
                   f"{usd(m['hours_saved_vs_naive'], 1)}, cloud avoided vs all-cloud {usd(m['cloud_avoided_cached'])} "
                   f"caching on ({usd(m['cloud_avoided_uncached'])} no caching), vs today's mix "
                   f"{usd(m['save_vs_mix_cached'])} caching on, recommended {s['recommendation']['device']} "
                   f"{s['recommendation']['tier']} saves {usd(m['rec_save_vs_naive'])}, +A used "
                   f"{'yes' if s['router']['assumption_used'] else 'no'}")
    return ("SCENARIO, not a measurement (rests on the +A residual-P1/P7 = 0 credit, P1 at the assumed rerun proxy, "
            "and local time valued at the full hour value; plan item B5 has not run). USD/month: " + "; ".join(out))


def sensitivity_value(rows) -> str:
    return "SCENARIO (+A, full hour value): " + "; ".join(
        f"{x['param']}={x['value']}: vs naive {usd(x['save_vs_naive'])} (if nobody waits "
        f"{usd(x['save_vs_naive_if_nobody_waits'])}), cloud avoided {usd(x['cloud_avoided_cached'])} "
        f"caching on ({usd(x['cloud_avoided_uncached'])} no caching), vs mix {usd(x['save_vs_mix_cached'])} "
        f"({x['router_tier']})" for x in rows)


BEGIN, END = "<!-- business_metrics:begin -->", "<!-- business_metrics:end -->"


def render_block(res) -> str:
    hv = res["hour_values"]
    P = res["profiles"]
    out = [BEGIN, "", "Generated by `analysis/business_metrics.py --write` from results/business_profiles_20261014.json "
           "and the impact model's functions. USD per month unless stated. Labels: measured = per-session quantity "
           "from result files; projection = measured per-session quantity x ASSUMED profile volumes. Cloud: "
           "claude-sonnet-5-5 from the shared Anthropic table, caching on (no caching = upper bound). Local compute "
           "time: 0 (nobody waits) to the full hour value (someone waits).", "",
           f"Hour values: median USD {hv['median']:.2f} (impact-hour-value); p10 USD {hv['p10']:.2f} and p90 USD "
           f"{hv['p90']:.2f} (business-hour-range, derived from BLS annual percentiles at {hv['hours_per_year']:.1f} "
           "h/year).", "", "### Per-profile facts (register rows business-mid-team, business-mid-team-qwen3-14b)", "",
           "| quantity | label | " + " | ".join(P) + " |", "|---|---|" + "---:|" * len(P)]

    def row(q, lab, f):
        out.append(f"| {q} | {lab} | " + " | ".join(f(r) for r in P.values()) + " |")

    row("team size; sessions/dev/day; cloud share today", "assumption",
        lambda r: f"{r['assumptions']['team_size']}; {r['assumptions']['agent_sessions_per_dev_per_day']}; "
                  f"{r['assumptions']['cloud_session_share']}")
    row("device; model", "assumption", lambda r: f"{r['assumptions']['device']}; {r['assumptions']['model']}")
    row("sessions / month", "assumption", lambda r: usd(r["sessions_month"]))
    row("naive local default window", "measured", lambda r: r["naive"]["tier"])
    row("cloud USD / session, whole session to the cloud: caching on (no caching)", "measured",
        lambda r: "n/a" if not r["all_cloud_per_session"] else
        f"{r['all_cloud_per_session']['cached']:.3f} ({r['all_cloud_per_session']['uncached']:.3f})")
    row("all-cloud spend: caching on (no caching)", "projection",
        lambda r: f"{usd(r['month']['all_cloud_cached'])} ({usd(r['month']['all_cloud_uncached'])})")
    row("naive local: compute time cost (0 to full hour value)", "projection",
        lambda r: rng(r["month"]["naive_time_low"], r["month"]["naive_time_high"]))
    row("naive local: compute hours / month", "projection", lambda r: usd(r["month"]["naive_hours"], 1))
    row("naive local: silent wrong results (P1)", "not measured", lambda r: "not measured (no B3 task runs)")
    row("today's mix: cloud fees caching on (no caching); local time up to", "projection",
        lambda r: f"{usd(r['month']['mix_cloud_cached'])} ({usd(r['month']['mix_cloud_uncached'])}); "
                  f"{usd(r['month']['mix_time_high'])}")
    row("router saving vs naive local default", "", lambda r: PENDING)
    row("router saving vs all-cloud", "", lambda r: PENDING)
    row("router saving vs today's mix", "", lambda r: PENDING)
    row("hardware recommendation", "", lambda r: PENDING)
    out += ["", "### Router replay per window (measured: which steps escalate and what they cost)", "",
            "| profile | window | steps kept local | where the share comes from | router cloud USD / session caching on (no caching) | cache-hit share | router cloud USD / month caching on (no caching), projection |",
            "|---|---|---:|---|---:|---:|---:|"]
    for name, r in P.items():
        wd = r["assumptions"]["workdays_per_month"]
        for c in r["windows"]:
            e = c["escalation"]
            out.append(f"| {name} | {c['tier']} | {'n/a' if c['local_share'] is None else f'{c['local_share']:.1%}'} | "
                       f"{c['local_trace']} | {usd(e['usd_per_session'], 3)} ({usd(e['usd_per_session_uncached'], 3)}) | "
                       f"{'n/a' if e['cache_hit_share'] is None else f'{e['cache_hit_share']:.0%}'} | "
                       f"{usd(None if c['router_cloud_day'] is None else c['router_cloud_day'] * wd)} "
                       f"({usd(None if c['router_cloud_day_uncached'] is None else c['router_cloud_day_uncached'] * wd)}) |")
    out += ["", "## SCENARIO (not a measurement): router savings under the +A credit", "",
            "Everything in this section rests on three assumptions, none measured: (1) +A, the router leaves no silent "
            "failure where the pre-registered mitigation recovered [R2-mitigation-v1-verdict]; (2) P1 valued at the "
            "impact model's rerun proxy (each silent rule/recall failure = one wrong result costing one full session "
            "rerun); (3) local compute time saved is worth the full hour value. Plan item B5 replaces all three. "
            "Register rows business-scenario-plus-a and business-sensitivity.", "",
            "| quantity | " + " | ".join(res["scenario"]) + " |", "|---|" + "---:|" * len(res["scenario"])]
    S = res["scenario"]

    def srow(q, f):
        out.append(f"| {q} | " + " | ".join(f(s) for s in S.values()) + " |")

    srow("router window (minimises time + router cloud, caching on)", lambda s: s["router"]["tier"])
    srow("+A credit used", lambda s: "yes" if s["router"]["assumption_used"] else "no")
    srow("saving vs naive default", lambda s: usd(s["month"]["save_vs_naive"]))
    srow("saving vs naive default if nobody waits (time worth 0; router cloud fees remain)",
         lambda s: usd(s["month"]["save_vs_naive_if_nobody_waits"]))
    srow("hours saved vs naive / month", lambda s: usd(s["month"]["hours_saved_vs_naive"], 1))
    srow("cloud spend avoided vs all-cloud: caching on (no caching)",
         lambda s: f"{usd(s['month']['cloud_avoided_cached'])} ({usd(s['month']['cloud_avoided_uncached'])})")
    srow("saving vs today's mix, caching on", lambda s: usd(s["month"]["save_vs_mix_cached"]))
    srow("hardware recommendation (hardware cost not priced)",
         lambda s: f"{s['recommendation']['device']} {s['recommendation']['tier']}: {usd(s['month']['rec_save_vs_naive'])}")
    srow("single-shot calls on the pareto choice: cloud avoided (STUB cloud reference)",
         lambda s: usd(s["month"]["single_shot_saved"], 2))
    ss = res["single_shot"]
    out += ["", f"Single-shot choice (pareto-sample-recommendation-v2): {ss['config']}, saving USD "
            f"{usd(ss['usd_per_1k_saved'], 2)} per 1k steps vs the all-cloud reference"
            + (" [STUB: cloud quality not measured]" if ss["stub"] else "") + "."]
    out += ["", f"### Scenario sensitivity ({res['doc']['sensitivity']['base_profile']}, one assumption varied at a "
            "time; USD/month; register row business-sensitivity)", "",
            "| assumption | value | router window | saving vs naive | saving vs naive if nobody waits | cloud avoided "
            "vs all-cloud, caching on (no caching) | saving vs today's mix, caching on | hours saved vs naive |",
            "|---|---|---|---:|---:|---:|---:|---:|"]
    for x in res["sensitivity"]:
        out.append(f"| {x['param']} | {x['value']} | {x['router_tier']} | {usd(x['save_vs_naive'])} | "
                   f"{usd(x['save_vs_naive_if_nobody_waits'])} | "
                   f"{usd(x['cloud_avoided_cached'])} ({usd(x['cloud_avoided_uncached'])}) | "
                   f"{usd(x['save_vs_mix_cached'])} | {usd(x['hours_saved_vs_naive'], 1)} |")
    out += ["", END]
    return "\n".join(out)


def write_doc(block, repo=REPO):
    path = repo / "docs" / "BUSINESS_METRICS.md"
    text = path.read_text(encoding="utf-8")
    i, j = text.index(BEGIN), text.index(END) + len(END)
    path.write_text(text[:i] + block + text[j:], encoding="utf-8")


def main(argv=None):
    argv = argv or sys.argv[1:]
    res = compute_all()
    for name, r in res["profiles"].items():
        print(profile_value(name, r))
    print("scenario:", scenario_value(res))
    print("sensitivity:", sensitivity_value(res["sensitivity"]))
    if "--write" in argv:
        write_doc(render_block(res))
        print("wrote docs/BUSINESS_METRICS.md generated block")


if __name__ == "__main__":
    main()
