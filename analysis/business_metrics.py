"""Business metrics (offline): what the router saves a customer team per month, vs all-cloud, vs the runtime's naive
local default, and vs today's mix, for a few customer profiles. Plan v2 Part C1 (operator plan, 2026-10-13).

No machine time, no API call. Built only from the impact model (plan item A3, analysis/impact_model.py), whose
functions are reused unchanged:

  - per-session time and quality costs P1, P2, P4, P5, P7 at a given device and window: impact_model.p1 .. p7
    (register rows impact-P1 .. impact-P7 are the same functions at the impact profile);
  - cloud cost of a whole session sent to the cloud model, without and with implicit prefix caching:
    impact_model.r2_cloud_overspend (the input of register row impact-P3);
  - the router's own decisions: impact_model.router_escalation (R2 sessions replayed through src/dse/router.py,
    escalations E1-E4, each escalated step priced uncached with output at the step's answer budget, so an upper
    bound on the router's cloud spend; the "Escalation cost of the P1 fix" table of docs/IMPACT.md).

The policy comparison of plan item B5 has NOT run. So the router's effect is only what the replay computes (which
steps go to the cloud, at what price) plus one ASSUMPTION, stated wherever it is used: at a window where the runtime
alone loses context (P1 or P7 nonzero or not measured), the router's residual P1 and P7 are zero, because the router
never sends an overflowing request (client-side trimming keeps the system prompt) and escalates a step whose needed
turn would be trimmed (E4). The assumption is applied only where the pre-registered mitigation measured 'recovered'
for that model and window (register row R2-mitigation-v1-verdict: llama3.1:8b at num_ctx 4096 and 8192, qwen3:14b at
8192; qwen3:14b at 4096 'not_recovered'); elsewhere the router gets no credit for P1/P7. Every saving that rests on
it is labelled +A.

Profiles: results/business_profiles_20261014.json. Every profile field is an ASSUMPTION. The hour value is register
row impact-hour-value; its range uses the same BLS page's percentiles (source, quotes and the derivation in the JSON).

Labels used in the output:
  measured    a per-session or per-step quantity read from result files (R2 sessions, Router replay, price table)
  projection  a per-month figure: measured per-session quantities times the assumed volumes of a profile
  +A          additionally rests on the residual-P1/P7 = 0 router assumption, or on the session proxy device

Usage: py -3.12 analysis/business_metrics.py            prints the per-profile table
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
ROUTER_ZERO = ("P1", "P7")   # residual cost assumed zero under the router (see module doc)
NOT_MEASURED = im.NOT_MEASURED


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


def _impact_profile(a, hv, device, tier) -> dict:
    """An impact_model profile dict for one (device, window) at the business profile's volumes."""
    keys = ("team_size", "agent_sessions_per_dev_per_day", "turns_per_session", "single_shot_calls_per_dev_per_day",
            "workdays_per_month", "co_runner_share", "cloud_session_share", "model", "cloud_model", "cloud_tier")
    pa = {k: a[k] for k in keys}
    pa.update(device=device, runtime=tier, scenario_runtime=tier)
    return {"hour_value": {"usd_per_hour": hv}, "assumptions": pa}


# ── shared inputs (computed once per repo) ──────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=2)
def _inputs(repo_str: str) -> dict:
    repo = Path(repo_str)
    wall = im.session_wallclock(repo)
    rates = im.context_loss_rates(repo)
    return {"P1": {"rates": rates, "wall": wall}, "P2": {"deltas": im.oversize_turn_delta(repo)},
            "P4": {"rates": im.decode_rates(repo), "wall": wall}, "P5": {"ratios": im.co_runner_ratios(repo),
                                                                        "wall": wall},
            "P7": {"rates": rates}}


@lru_cache(maxsize=32)
def _escalation(repo_str: str, model: str, tier: str, cloud_model: str, cloud_tier: str) -> dict:
    prof = {"hour_value": {"usd_per_hour": 1.0},
            "assumptions": {"cloud_model": cloud_model, "cloud_tier": cloud_tier}}
    return im.router_escalation(Path(repo_str), model, tier, prof)


@lru_cache(maxsize=4)
def _cloud_sessions(repo_str: str, cloud_model: str, cloud_tier: str) -> dict:
    prof = {"hour_value": {"usd_per_hour": 1.0},
            "assumptions": {"cloud_model": cloud_model, "cloud_tier": cloud_tier}}
    return im.r2_cloud_overspend(Path(repo_str), prof)


# ── pure arithmetic (synthetic-testable) ────────────────────────────────────────────────────────────────────────

def policy_total(rows: dict, cloud_usd_day: float = 0.0) -> dict:
    """Sum of the measured rows plus cloud spend; rows that are None are listed, never filled in."""
    known = {k: v for k, v in rows.items() if v is not None}
    return {"usd_day": sum(known.values()) + cloud_usd_day, "time_usd_day": sum(known.values()),
            "cloud_usd_day": cloud_usd_day, "not_measured": sorted(k for k, v in rows.items() if v is None)}


def router_rows(naive_rows: dict, mitigation: str | None) -> tuple[dict, bool]:
    """Router rows at the same device and window. P1 and P7 are set to 0 only when that needs no assumption (the
    naive value is already a measured 0) or when the pre-registered mitigation at this (model, window) measured
    'recovered' (register row R2-mitigation-v1-verdict); then assumption_used is True. Otherwise (mitigation
    'not_recovered' or not run) the router gets no credit: P1 and P7 keep their naive values.
    Returns (rows, assumption_used)."""
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


# ── one configuration ───────────────────────────────────────────────────────────────────────────────────────────

def session_rows(a, hv, device, tier, repo=REPO) -> dict:
    """USD/day of P1, P2, P4, P5, P7 at (device, tier) for the profile's volumes, via impact_model's row functions.
    A device without its own R2 sessions uses a['session_proxy_device'] for P1/P2/P7 (assumption, see the JSON)
    and prints P5 as not measured; P4 uses the device's own decode rate ('blade@<ctx>' for the laptop)."""
    inp = _inputs(str(repo))
    own = device == "evo-x2"
    sess_dev = device if own else a.get("session_proxy_device", device)
    prof = _impact_profile(a, hv, sess_dev, tier)
    out = {}
    for k in ("P1", "P2", "P7"):
        out[k] = im.ROW_FUNCS[k](repo, prof, None, inp[k])["usd_day"]
    out["P5"] = im.p5(repo, prof, None, inp["P5"])["usd_day"] if own else None
    p4_dev = device if own else f"{device}@{im.tier_ctx(tier)}"
    out["P4"] = im.p4(repo, _impact_profile(a, hv, p4_dev, tier), None, inp["P4"])["usd_day"]
    return {k: out[k] for k in SESSION_ROWS}


def config(a, hv, device, tier, repo=REPO) -> dict:
    """Naive (runtime alone) and router costs at one (device, window)."""
    naive = session_rows(a, hv, device, tier, repo)
    mit = _mitigation(str(repo)).get((a["model"], tier))
    rrows, used = router_rows(naive, mit)
    esc = _escalation(str(repo), a["model"], tier, a["cloud_model"], a["cloud_tier"])
    spd = sessions_day(a)
    esc_day = None if esc["usd_per_session"] is None else spd * esc["usd_per_session"]
    return {"device": device, "tier": tier, "naive_rows": naive, "naive": policy_total(naive),
            "router_rows": rrows, "router": policy_total(rrows, esc_day or 0.0), "escalation": esc,
            "router_ok": esc_day is not None, "proxy": device != "evo-x2", "mitigation": mit,
            "assumption_used": used or device != "evo-x2",
            "local_share": steps_local_share(esc)}


def hours(usd_day, hv):
    return None if usd_day is None else usd_day / hv


# ── one profile ─────────────────────────────────────────────────────────────────────────────────────────────────

def single_shot(repo=REPO) -> dict:
    """pareto-sample-recommendation-v2's choice and its saving per 1k single-shot steps vs the all-cloud STUB."""
    import numbers_register as nr
    P, pts = nr._pareto_points(repo)
    rec = P.recommend(pts, **nr.PARETO_SAMPLE, max_timeout_share=nr.PARETO_SAMPLE_MAX_TIMEOUT_SHARE)
    ch = rec.chosen
    return {"config": None if ch is None else f"{ch.machine} {ch.model} {ch.runtime}",
            "usd_per_1k_saved": rec.savings_vs_all_cloud_usd_per_1k, "stub": rec.stub}


def profile_metrics(a, hv, repo=REPO, ss=None) -> dict:
    spd = sessions_day(a)
    wd = a["workdays_per_month"]
    cs = _cloud_sessions(str(repo), a["cloud_model"], a["cloud_tier"]).get(a["model"])
    all_cloud = {"cached_day": None if not cs else spd * cs["cached"],
                 "uncached_day": None if not cs else spd * cs["uncached"], "per_session": cs}
    naive = config(a, hv, a["device"], a["default_tier"], repo)
    # a candidate is comparable only if it has a Router replay and no row is unpriced that the naive default prices
    # (otherwise leaving a cost out would make it look cheaper)
    nm = set(naive["naive"]["not_measured"])
    ok = lambda c: c["router_ok"] and set(c["router"]["not_measured"]) <= nm  # noqa: E731
    own = [c for c in (config(a, hv, a["device"], t, repo) for t in a["router_tiers"]) if ok(c)]
    best = min(own, key=lambda c: c["router"]["usd_day"])
    alts = [c for c in (config(a, hv, d, t, repo) for d, ts in (a.get("alternatives") or {}).items() for t in ts)
            if ok(c)]
    rec = min(own + alts, key=lambda c: c["router"]["usd_day"])
    n_day, r_day = naive["naive"]["usd_day"], best["router"]["usd_day"]
    share = a["cloud_session_share"]
    mix = {k: (None if all_cloud[f"{k}_day"] is None else mix_usd_day(share, all_cloud[f"{k}_day"], n_day))
           for k in ("cached", "uncached")}
    ss = ss or single_shot(repo)
    ss_month = (None if ss["usd_per_1k_saved"] is None else
                a["team_size"] * a["single_shot_calls_per_dev_per_day"] * wd * ss["usd_per_1k_saved"] / 1000.0)
    m = lambda v: None if v is None else v * wd  # noqa: E731
    return {
        "assumptions": a, "hour_value": hv, "sessions_month": spd * wd,
        "all_cloud": all_cloud, "naive": naive, "router": best, "router_candidates": own,
        "recommendation": rec, "alternatives": alts,
        "month": {
            "all_cloud_cached": m(all_cloud["cached_day"]), "all_cloud_uncached": m(all_cloud["uncached_day"]),
            "naive_total": m(n_day), "naive_hours": m(hours(naive["naive"]["time_usd_day"], hv)),
            "router_total": m(r_day), "router_cloud": m(best["router"]["cloud_usd_day"]),
            "router_time": m(best["router"]["time_usd_day"]),
            "router_hours": m(hours(best["router"]["time_usd_day"], hv)),
            "cloud_avoided_vs_cached": m(None if all_cloud["cached_day"] is None else
                                         all_cloud["cached_day"] - best["router"]["cloud_usd_day"]),
            "cloud_avoided_vs_uncached": m(None if all_cloud["uncached_day"] is None else
                                           all_cloud["uncached_day"] - best["router"]["cloud_usd_day"]),
            "save_vs_naive": m(n_day - r_day),
            "hours_saved_vs_naive": m(hours(naive["naive"]["time_usd_day"] - best["router"]["time_usd_day"], hv)),
            "mix_cached": m(mix["cached"]), "mix_uncached": m(mix["uncached"]),
            "save_vs_mix_cached": m(None if mix["cached"] is None else mix["cached"] - r_day),
            "save_vs_mix_uncached": m(None if mix["uncached"] is None else mix["uncached"] - r_day),
            "rec_save_vs_naive": m(n_day - rec["router"]["usd_day"]),
            "single_shot_saved": ss_month,
        },
        "single_shot": ss,
    }


def compute_all(repo=REPO, doc=None) -> dict:
    doc = doc or load_profiles(repo)
    hvs = hour_values(doc, repo)
    ss = single_shot(repo)
    res = {name: profile_metrics(p["assumptions"], hvs["median"], repo, ss) for name, p in doc["profiles"].items()}
    return {"doc": doc, "hour_values": hvs, "profiles": res, "sensitivity": sensitivity(doc, hvs, repo, ss)}


def sensitivity(doc, hvs, repo=REPO, ss=None) -> list:
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
            r = profile_metrics(a, hv, repo, ss)
            out.append({"param": param, "value": v if param != "hour_value" else f"{v} (USD {hv:.2f}/h)",
                        "router_tier": r["router"]["tier"], **{k: r["month"][k] for k in (
                            "save_vs_naive", "cloud_avoided_vs_cached", "cloud_avoided_vs_uncached",
                            "save_vs_mix_cached", "save_vs_mix_uncached", "hours_saved_vs_naive")}})
    return out


# ── formatting ──────────────────────────────────────────────────────────────────────────────────────────────────

def usd(v, nd=0):
    return NOT_MEASURED if v is None else f"{v:,.{nd}f}"


def rng(lo, hi, nd=0):
    if lo is None or hi is None:
        return NOT_MEASURED
    return f"{usd(lo, nd)} to {usd(hi, nd)}"


def label(r) -> str:
    return "projection +A" if r["router"]["assumption_used"] or r["naive"]["proxy"] else "projection"


def profile_value(name, r) -> str:
    """One register value string for a profile (business-* rows)."""
    a, m, rt, nv = r["assumptions"], r["month"], r["router"], r["naive"]
    rec = r["recommendation"]
    return (f"{name} (ASSUMED team {a['team_size']}, {a['agent_sessions_per_dev_per_day']} sessions/dev/day, cloud "
            f"share {a['cloud_session_share']}; {a['device']} {a['model']}; hour USD {r['hour_value']:.2f}): "
            f"router setting {rt['device']} {rt['tier']}, steps kept local "
            f"{'n/a' if rt['local_share'] is None else f'{rt['local_share']:.1%}'} (measured, Router replay n="
            f"{rt['escalation']['n']}); USD/month (projection): router total {usd(m['router_total'])} (cloud "
            f"{usd(m['router_cloud'])} upper bound, time {usd(m['router_time'])}); naive default "
            f"{nv['tier']} {usd(m['naive_total'])} (not measured: {', '.join(nv['naive']['not_measured']) or 'none'}); "
            f"all-cloud {rng(m['all_cloud_cached'], m['all_cloud_uncached'])} (cached to uncached); saving vs naive "
            f"{usd(m['save_vs_naive'])}, hours saved {usd(m['hours_saved_vs_naive'], 1)}; cloud spend avoided vs "
            f"all-cloud {rng(m['cloud_avoided_vs_cached'], m['cloud_avoided_vs_uncached'])}; saving vs today's mix "
            f"{rng(m['save_vs_mix_cached'], m['save_vs_mix_uncached'])}; recommended {rec['device']} {rec['tier']} "
            f"saves {usd(m['rec_save_vs_naive'])} vs naive; label {label(r)}")


def sensitivity_value(rows) -> str:
    return "; ".join(f"{x['param']}={x['value']}: vs naive {usd(x['save_vs_naive'])}, vs all-cloud "
                     f"{rng(x['cloud_avoided_vs_cached'], x['cloud_avoided_vs_uncached'])}, vs mix "
                     f"{rng(x['save_vs_mix_cached'], x['save_vs_mix_uncached'])} ({x['router_tier']})" for x in rows)


BEGIN, END = "<!-- business_metrics:begin -->", "<!-- business_metrics:end -->"


def render_block(res) -> str:
    hv = res["hour_values"]
    P = res["profiles"]
    out = [BEGIN, "", "Generated by `analysis/business_metrics.py --write` from results/business_profiles_20261014.json "
           "and the impact model's functions. USD per month unless stated. Labels: measured = per-session quantity "
           "from result files; projection = measured per-session quantity x ASSUMED profile volumes; +A = also rests "
           "on the residual-P1/P7 = 0 router assumption or on the session proxy device (see How to read).", "",
           f"Hour values: median USD {hv['median']:.2f} (impact-hour-value); p10 USD {hv['p10']:.2f} and p90 USD "
           f"{hv['p90']:.2f} (business-hour-range, derived from BLS annual percentiles at {hv['hours_per_year']:.1f} "
           "h/year).", "", "### Per-profile savings (register rows business-*)", "",
           "| quantity | label | " + " | ".join(P) + " |", "|---|---|" + "---:|" * len(P)]

    def row(q, lab, f):
        out.append(f"| {q} | {lab} | " + " | ".join(f(r) for r in P.values()) + " |")

    row("team size; sessions/dev/day; cloud share today", "assumption",
        lambda r: f"{r['assumptions']['team_size']}; {r['assumptions']['agent_sessions_per_dev_per_day']}; "
                  f"{r['assumptions']['cloud_session_share']}")
    row("device; model", "assumption", lambda r: f"{r['assumptions']['device']}; {r['assumptions']['model']}")
    row("sessions / month", "assumption", lambda r: usd(r["sessions_month"]))
    row("naive local default window", "measured", lambda r: r["naive"]["tier"])
    row("router setting (window minimising router total)", "measured inputs", lambda r: r["router"]["tier"])
    row("steps kept local under the router", "measured (Router replay)",
        lambda r: "n/a" if r["router"]["local_share"] is None else f"{r['router']['local_share']:.1%}")
    row("cloud USD / session: all-cloud cached; uncached", "measured",
        lambda r: "n/a" if not r["all_cloud"]["per_session"] else
        f"{r['all_cloud']['per_session']['cached']:.3f}; {r['all_cloud']['per_session']['uncached']:.3f}")
    row("cloud USD / session: router escalations (upper bound)", "measured (Router replay)",
        lambda r: usd(r["router"]["escalation"]["usd_per_session"], 3))
    row("all-cloud spend (cached to uncached)", "projection",
        lambda r: rng(r["month"]["all_cloud_cached"], r["month"]["all_cloud_uncached"]))
    row("naive local: time and quality cost", "projection",
        lambda r: usd(r["month"]["naive_total"]) + (f" (not measured: {', '.join(r['naive']['naive']['not_measured'])})"
                                                   if r["naive"]["naive"]["not_measured"] else ""))
    row("router: cloud spend (upper bound)", "projection", lambda r: usd(r["month"]["router_cloud"]))
    row("router: time cost", "projection" , lambda r: usd(r["month"]["router_time"]) + (
        f" (not measured: {', '.join(r['router']['router']['not_measured'])})" if r["router"]["router"]["not_measured"]
        else ""))
    row("cloud spend avoided vs all-cloud", "projection",
        lambda r: rng(r["month"]["cloud_avoided_vs_cached"], r["month"]["cloud_avoided_vs_uncached"]))
    row("time saved vs all-cloud", "not measured", lambda r: "not measured (cloud latency of these sessions)")
    row("saving vs naive local default", "", lambda r: f"{usd(r['month']['save_vs_naive'])} ({label(r)})")
    row("developer hours saved vs naive / month", "", lambda r: f"{usd(r['month']['hours_saved_vs_naive'], 1)} ({label(r)})")
    row("saving vs today's mix (cached to uncached)", "", lambda r: f"{rng(r['month']['save_vs_mix_cached'], r['month']['save_vs_mix_uncached'])} ({label(r)})")
    row("hardware recommendation", "measured inputs",
        lambda r: f"{r['recommendation']['device']} {r['recommendation']['tier']} + router")
    row("recommendation saves vs naive default", "",
        lambda r: f"{usd(r['month']['rec_save_vs_naive'])} ({'projection +A' if r['recommendation']['assumption_used'] or r['naive']['proxy'] else 'projection'}; hardware cost not priced)")
    row("single-shot calls on the pareto choice: cloud spend avoided", "projection, STUB cloud reference",
        lambda r: usd(r["month"]["single_shot_saved"], 2))
    ss = next(iter(P.values()))["single_shot"]
    out += ["", f"Single-shot choice (pareto-sample-recommendation-v2): {ss['config']}, saving USD "
            f"{usd(ss['usd_per_1k_saved'], 2)} per 1k steps vs the all-cloud reference"
            + (" [STUB: cloud quality not measured]" if ss["stub"] else "") + "."]
    out += ["", "### Router candidates per profile (window: router total USD/month = time + cloud; steps local)", "",
            "| profile | device | window | naive total | router time | router cloud (upper bound) | router total | "
            "steps local | not measured | +A |", "|---|---|---|---:|---:|---:|---:|---:|---|---|"]
    for name, r in P.items():
        wd = r["assumptions"]["workdays_per_month"]
        for c in r["router_candidates"] + r["alternatives"]:
            out.append(f"| {name} | {c['device']} | {c['tier']} | {usd(c['naive']['usd_day'] * wd)} | "
                       f"{usd(c['router']['time_usd_day'] * wd)} | {usd(c['router']['cloud_usd_day'] * wd)} | "
                       f"{usd(c['router']['usd_day'] * wd)} | "
                       f"{'n/a' if c['local_share'] is None else f'{c['local_share']:.1%}'} | "
                       f"{', '.join(c['router']['not_measured']) or 'none'} | {'yes' if c['assumption_used'] else 'no'} |")
    out += ["", f"### Sensitivity ({res['doc']['sensitivity']['base_profile']}, one assumption varied at a time; "
            "USD/month, all projection; register row business-sensitivity)", "",
            "| assumption | value | router window | saving vs naive | cloud avoided vs all-cloud | saving vs today's mix | "
            "hours saved vs naive |", "|---|---|---|---:|---:|---:|---:|"]
    for x in res["sensitivity"]:
        out.append(f"| {x['param']} | {x['value']} | {x['router_tier']} | {usd(x['save_vs_naive'])} | "
                   f"{rng(x['cloud_avoided_vs_cached'], x['cloud_avoided_vs_uncached'])} | "
                   f"{rng(x['save_vs_mix_cached'], x['save_vs_mix_uncached'])} | {usd(x['hours_saved_vs_naive'], 1)} |")
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
    print("sensitivity:", sensitivity_value(res["sensitivity"]))
    if "--write" in argv:
        write_doc(render_block(res))
        print("wrote docs/BUSINESS_METRICS.md generated block")


if __name__ == "__main__":
    main()
