"""Session-level bootstrap CIs for the headline proportions that were reported with a task-level (Wilson) CI or with
no CI (operator request, 2026-10-26: "redo every headline CI by bootstrapping over sessions (not turns)").

Unit. A session is one local run of 30 task turns: one (tier, seed) of results/x2_evalset_tasks_real_v2.jsonl. Task
turns inside one session share an environment and a conversation history, so they are not independent; a Wilson CI
over task turns treats them as if they were. The session bootstrap here resamples whole sessions with replacement,
stratified by tier where there is more than one tier, 10000 resamples, one random.Random(20261014) driving all
resamples, percentile 95% CI: the same procedure as harness/x2_r2_evalset.session_bootstrap_excess and
analysis/r2_real_report.excess_over_default, applied to a pooled proportion (sum successes / sum n over the resampled
sessions) instead of an excess. A 0/n or n/n proportion has no bootstrap variance, so its session CI is a point.

Rows: headline-ci-session-rescue (the 197 wrong local tasks: full-history redo, rollback redo, combined rescue, and
the surfaced/silent split) and headline-ci-session-hybrid (hybrid-route-at-start-measured at the 8192 setting with
claude-sonnet-5-5: 3 sessions, one per seed, so the session CI is very coarse).
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BOOT_N = 10000
BOOT_SEED = 20261014
REAL_V2 = "results/x2_evalset_tasks_real_v2.jsonl"


def _quantile(xs, q):
    if not xs:
        return None
    i = q * (len(xs) - 1)
    lo = int(i)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def session_bootstrap_prop(cells: dict, n_boot: int = BOOT_N, seed: int = BOOT_SEED) -> dict:
    """Pure. cells {stratum: [(successes, n) per session]}. Pooled proportion and its 95% percentile CI, each resample
    drawing, with replacement, as many sessions as each stratum has from that stratum."""
    cells = {k: v for k, v in cells.items() if v}
    k = sum(a for v in cells.values() for a, _ in v)
    n = sum(b for v in cells.values() for _, b in v)
    if not n:
        raise ValueError("no units")
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        sa = sb = 0
        for xs in cells.values():
            for _ in xs:
                a, b = xs[rng.randrange(len(xs))]
                sa += a
                sb += b
        draws.append(sa / sb)
    draws.sort()
    return {"k": k, "n": n, "p": k / n, "ci95": (_quantile(draws, 0.025), _quantile(draws, 0.975)),
            "n_sessions": sum(len(v) for v in cells.values()), "n_strata": len(cells), "n_resamples": n_boot,
            "seed": seed}


def cells_from_units(units: list, ok) -> dict:
    """Pure. units: dicts with 'tier' and 'seed'; ok(unit) -> bool. {tier: [(ok count, n) per (tier, seed)]}."""
    by = {}
    for u in units:
        by.setdefault((u["tier"], u["seed"]), []).append(1 if ok(u) else 0)
    cells = {}
    for (t, _s), xs in sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        cells.setdefault(t, []).append((sum(xs), len(xs)))
    return cells


def _wilson(k, n):
    sys.path.insert(0, str(REPO / "analysis"))
    import numbers_register as nr
    return nr.wilson_ci(k, n)


def _fmt(b: dict, unit_note: str) -> str:
    lo, hi = _wilson(b["k"], b["n"])
    return (f"{b['k']}/{b['n']} ({100 * b['p']:.1f}%); task-level 95% Wilson CI [{lo:.1%}, {hi:.1%}] (unit task, "
            f"treats the {b['n']} as independent); session-level 95% bootstrap CI [{b['ci95'][0]:.1%}, "
            f"{b['ci95'][1]:.1%}] ({b['n_sessions']} sessions{unit_note})")


# -- the 197 wrong local tasks ---------------------------------------------------------------------------------------

def rescue_session_ci(repo: Path = REPO) -> dict:
    sys.path.insert(0, str(Path(repo) / "analysis"))
    import own_rollback as orb
    _cr, _cd, d, merged = orb.redo_diagnosis(repo)
    rb = orb.own_rollback_redo(repo)
    key = lambda r: (r["tier"], r["seed"], r["turn_idx"])  # noqa: E731
    repair = {key(p) for p in d["per_task"] if p["cls"] == "earlier_state_repair"}
    rb_ok = {key(t): bool(t.get("cloud_end_state_ok")) for t in rb["tasks"]}
    rb_fixed = {k for k in repair if rb_ok.get(k)}
    fh = cells_from_units(merged, lambda r: bool(r.get("cloud_end_state_ok")))
    comb = cells_from_units(merged, lambda r: bool(r.get("cloud_end_state_ok")) or key(r) in rb_fixed)
    rbu = [t for t in rb["tasks"] if key(t) in repair]
    rbc = cells_from_units(rbu, lambda t: bool(t.get("cloud_end_state_ok")))
    return {"full_history": session_bootstrap_prop(fh), "combined": session_bootstrap_prop(comb),
            "rollback": session_bootstrap_prop(rbc),
            "per_tier": {t: session_bootstrap_prop({t: v}) for t, v in fh.items()}}


def surfaced_split(repo: Path = REPO) -> dict:
    """Every wrong local task turn of the B3 v2 real run (qwen3:14b, all tiers and seeds): surfaced (an HTTP error or
    error field on any call of the turn, the definition of x2_r2_evalset.tasks_real_v2_outcome) vs silent."""
    ag = _r2_agent(repo)
    rows = ag.read_rows(Path(repo) / REAL_V2)
    sessions = [x for x in ag.completed_sessions(rows) if x.get("mode") == "real" and x["model_id"] == "qwen3:14b"]
    att = {x["attempt_id"]: x for x in sessions}
    tier_of = lambda arm: arm.replace("_call2_notools", "")  # noqa: E731
    wrong = [{"tier": tier_of(att[r["attempt_id"]]["arm_id"]), "seed": att[r["attempt_id"]]["seed"],
              "surfaced": bool(r.get("any_http_error"))}
             for r in rows if r.get("record") == "r2a_turn" and r.get("attempt_id") in att
             and r.get("item_kind") == "task" and r.get("end_state_ok") is False]
    surf = cells_from_units(wrong, lambda u: u["surfaced"])
    per_tier = {}
    for u in wrong:
        per_tier[u["tier"]] = per_tier.get(u["tier"], 0) + 1
    return {"b": session_bootstrap_prop(surf), "n_sessions_total": len(sessions), "per_tier": per_tier}


def _r2_agent(repo):
    sys.path.insert(0, str(Path(repo) / "harness"))
    import x2_r2_agent
    return x2_r2_agent


def rescue_text(r: dict) -> str:
    note = ", stratified by tier; session = one (tier, seed) local run with at least one wrong task"
    tiers = "; ".join(f"{t} {v['k']}/{v['n']}, session-level [{v['ci95'][0]:.1%}, {v['ci95'][1]:.1%}] "
                      f"({v['n_sessions']} sessions)" for t, v in r["per_tier"].items())
    return ("session-level bootstrap of the cloud rescue of the wrong local tasks (B3 v2 real run, qwen3:14b, 5 tiers "
            "x 5 seeds; claude-sonnet-5-5; 10000 resamples, seed 20261014, percentile CI): full-history redo fixed "
            + _fmt(r["full_history"], note) + f"; per tier (full-history redo): {tiers}"
            + "; rollback redo of the earlier_state_repair tasks fixed " + _fmt(r["rollback"], note)
            + " (STAND-IN: earlier wrong turns replaced by reference calls; n/n has no bootstrap variance)"
            + "; combined rescue " + _fmt(r["combined"], note) + " (same STAND-IN for its rollback part)")


def surfaced_text(r: dict) -> str:
    s = r["b"]
    lo, hi = _wilson(s["k"], s["n"])
    tiers = ", ".join(f"{t} {v}" for t, v in sorted(r["per_tier"].items()))
    return (f"B3 v2 real run, qwen3:14b, {r['n_sessions_total']} sessions (5 tiers x 5 seeds, 30 task turns each): "
            f"wrong local task turns {s['n']} (per tier {tiers}); surfaced an HTTP error or error field on any call "
            f"{s['k']}/{s['n']} ({100 * s['p']:.1f}%), silent {s['n'] - s['k']}/{s['n']} "
            f"({100 * (s['n'] - s['k']) / s['n']:.1f}%); task-level 95% Wilson CI for the surfaced share [{lo:.1%}, "
            f"{hi:.1%}] (unit task turn; the turns are nested in {s['n_sessions']} sessions, so not independent); "
            f"session-level bootstrap [{s['ci95'][0]:.1%}, {s['ci95'][1]:.1%}] ({s['n_sessions']} sessions, stratified "
            f"by tier, 10000 resamples, seed 20261014; 0/n has no bootstrap variance); 'surfaced' is what the client "
            f"sees (HTTP status or error field in the reply), not the Ollama server log")


# -- hybrid route at start, 8192, claude-sonnet-5-5 -----------------------------------------------------------------

HYB_TIER = "ollama_ctx_8192"
HYB_MODEL = "claude-sonnet-5-5"


def hybrid_sessions(repo: Path = REPO, tier: str = HYB_TIER, model: str = HYB_MODEL) -> list:
    """Per seed (session): n, hybrid correct and USD, all-cloud correct and USD, naive local correct."""
    sys.path.insert(0, str(Path(repo) / "analysis"))
    import impact_model as im
    d = im.route_at_start_measured(repo)
    if d["status"] != "VERIFIED":
        raise ValueError(f"hybrid-route-at-start-measured PENDING: {d.get('pending')}")
    cell = next(c for c in d["cells"] if c["tier"] == tier and c["model"] == model)
    local = im.replay_local_tasks(repo)[tier]
    cloud = im.replay_cloud_tasks(repo)["models"][model]
    tasks = cell["hybrid"]["tasks"]
    out = []
    for s in sorted({k[0] for k in tasks}):
        ks = [k for k in tasks if k[0] == s]
        out.append({"seed": s, "n": len(ks), "hyb_ok": sum(bool(tasks[k]["ok"]) for k in ks),
                    "hyb_usd": sum(tasks[k]["usd"] for k in ks), "ac_ok": sum(bool(cloud[k]["ok"]) for k in ks),
                    "ac_usd": sum(cloud[k]["usd"] for k in ks), "nv_ok": sum(bool(local[k]["ok"]) for k in ks)})
    return out


def hybrid_bootstrap(sess: list, n_boot: int = BOOT_N, seed: int = BOOT_SEED) -> dict:
    """Pure. Paired session bootstrap: each resample draws len(sess) seeds with replacement and carries all three
    policies of a drawn seed together. Accuracies, saving vs all-cloud (1 - hybrid USD / all-cloud USD) and the
    wrong-count ratio naive / hybrid (resamples with 0 hybrid wrong are counted and left out of the ratio)."""
    def agg(xs):
        n = sum(x["n"] for x in xs)
        hw = n - sum(x["hyb_ok"] for x in xs)
        nw = n - sum(x["nv_ok"] for x in xs)
        return {"hyb": sum(x["hyb_ok"] for x in xs) / n, "ac": sum(x["ac_ok"] for x in xs) / n,
                "nv": sum(x["nv_ok"] for x in xs) / n,
                "saving": 1 - sum(x["hyb_usd"] for x in xs) / sum(x["ac_usd"] for x in xs),
                "ratio": nw / hw if hw else None}
    point = agg(sess)
    rng = random.Random(seed)
    draws = {k: [] for k in point}
    undefined = 0
    for _ in range(n_boot):
        a = agg([sess[rng.randrange(len(sess))] for _ in sess])
        for k, v in a.items():
            if v is None:
                undefined += 1
            else:
                draws[k].append(v)
    ci = {k: (_quantile(sorted(v), 0.025), _quantile(sorted(v), 0.975)) for k, v in draws.items()}
    return {"point": point, "ci95": ci, "ratio_undefined": undefined, "n_sessions": len(sess),
            "n": sum(x["n"] for x in sess), "hyb_ok": sum(x["hyb_ok"] for x in sess),
            "ac_ok": sum(x["ac_ok"] for x in sess), "nv_ok": sum(x["nv_ok"] for x in sess),
            "n_resamples": n_boot, "seed": seed}


def hybrid_text(b: dict) -> str:
    p, c, n = b["point"], b["ci95"], b["n"]

    def acc(label, k, key):
        lo, hi = _wilson(k, n)
        return (f"{label} {k}/{n} ({100 * k / n:.1f}%), task-level Wilson [{lo:.1%}, {hi:.1%}], session-level "
                f"[{c[key][0]:.1%}, {c[key][1]:.1%}]")
    return (f"session-level bootstrap of hybrid-route-at-start-measured, evo-x2 {HYB_TIER} (memory setting 8192) vs "
            f"{HYB_MODEL}, replay of recorded runs; {b['n_sessions']} sessions (one per seed, 30 tasks each), paired "
            f"(a drawn seed carries all three policies), 10000 resamples, seed 20261014, percentile CI; with "
            f"{b['n_sessions']} sessions the session bootstrap is very coarse (at most 10 distinct resamples), so read "
            f"it as a sensitivity check, not a precise interval: "
            + acc("route at start", b["hyb_ok"], "hyb") + "; " + acc("all cloud", b["ac_ok"], "ac") + "; "
            + acc("naive local (8192)", b["nv_ok"], "nv")
            + f"; wrong: route at start {n - b['hyb_ok']}/{n} ({100 * (n - b['hyb_ok']) / n:.1f}%), naive local "
            f"{n - b['nv_ok']}/{n} ({100 * (n - b['nv_ok']) / n:.1f}%)"
            + f"; saving vs all-cloud {100 * p['saving']:.1f}%, session-level [{100 * c['saving'][0]:.1f}%, "
            f"{100 * c['saving'][1]:.1f}%]; wrong naive / wrong route at start {n - b['nv_ok']}/{n - b['hyb_ok']} = "
            f"{p['ratio']:.1f}x, session-level [{c['ratio'][0]:.1f}x, {c['ratio'][1]:.1f}x] "
            f"({b['ratio_undefined']} of 10000 resamples had 0 route-at-start wrong and are left out of the ratio)")
