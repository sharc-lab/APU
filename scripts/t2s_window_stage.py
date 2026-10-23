"""Plan and checks for the evo-t2s window 2026-10-26 to 2026-11-08 (pending Zach's confirmation). NOT loaded anywhere.

Canonical staged file: queues/t2s_oct26_queue.json, built by scripts/t2s_week_queue.py build-oct26 (one builder, gates
on its own t2s_o26_preflight, loaded at the window with `python t2s_week_queue.py load-window t2s_oct26_queue.json`).
The earlier second file queues/t2s_window_20261026_staged.json (B5 for llama3.1:8b on the iGPU only, 3 seeds, and a
3-seed iGPU-only B6, gated on the previous window's t2s_wk_* ids) was retired on 2026-10-22: two files for one window
with different B6 designs and gates pointing at a finished queue invited loading the wrong one. B5 is now in the
canonical file (t2s_week_queue.oct26_b5_cells: since 2026-10-21 the two llama3.1:8b cells at 5 seeds, then qwen3:14b
on the CPU at 4096; since the 2026-10-21 22:30 amendment at 2 seeds and only if qwen_decision says it fits), and the B6
entries follow B5. Since the amendment of 2026-10-21 (later): policy set on evo-t2s a, b, c, d, e, f
(t2s_week_queue.OCT26_B5_POLICIES; since 2026-10-22 per cell: the qwen3:14b CPU cell runs OCT26_QWEN_POLICIES = a, c,
d, e, f, see oct26_cell_policies; every B5 hour here is estimated for the cell's own set), qwen3:14b at 3 seeds, a launch
budget of the available hours minus SLACK_H = 24 h before the no-launch time, and B6 filling what is left of that
budget (select_b6) with hours from evo-x2 H-scale minutes times a measured evo-t2s / evo-x2 factor (b6_estimates).

  py -3.12 scripts/t2s_window_stage.py build      # rebuilds queues/t2s_oct26_queue.json (same as build-oct26)
  py -3.12 scripts/t2s_window_stage.py dry-run    # every entry through the week queue's checks; no machine, no SSH
  py -3.12 scripts/t2s_window_stage.py plan       # hours per entry (ESTIMATES) and what fits in the window

Order of the window (docs/T2S_WEEK_PLAN.md "Staged for 2026-10-26"; process rule 2026-10-20, docs/STATE.md):
  1. On the CONTROLLER, first: `py -3.12 harness/t2s_deadline.py open-window --no-launch-after <NO_LAUNCH_AFTER_UTC>
     --hard-deadline <HARD_DEADLINE_UTC> --reason "<handover to Zach ...; resume only in the next confirmed window>"`.
     It runs the handover check on evo-t2s (HANDOVER_DONE.json, the pause flag naming its releaser, no process of
     ours; refuses and changes nothing on failure), then widens the results sync to both hosts (an expiring window
     file; the APU-SyncResults task keeps --host evo-x2), arms the handover deadline, waits to the hard deadline, and
     removes the sync window file in its finally block. The same job that arms the deadline is the one that switches
     the sync and switches it back.
  2. Deploy the window's files, then `python t2s_week_queue.py load-window t2s_oct26_queue.json` (one read-modify-write
     under the queue pause, cleared in its finally block: this is the release of the handover pause).
  3. The queue runs: preflight, A5, B5 (the staged cells), then B6. The B5 job stops at a session boundary at the deadline.

HOURS ARE ESTIMATES. B5: harness/policy_compare.estimate_t2s, from MEASURED evo-t2s session minutes
(register row t2s-r2-session-minutes); B6: no timing of those models on evo-t2s, no estimate.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO / "scripts"))

import t2s_week_queue as wq  # noqa: E402

CANONICAL = wq.OCT26_PATH
RETIRED = REPO / "queues" / "t2s_window_20261026_staged.json"
WINDOW = ("2026-10-26", "2026-11-08")
WINDOW_START_UTC = "2026-10-26T04:00:00+00:00"       # 2026-10-26 00:00 EDT
NO_LAUNCH_AFTER_UTC = "2026-11-09T04:00:00+00:00"    # 2026-11-08 23:00 EST (DST ends 2026-11-01)
HARD_DEADLINE_UTC = "2026-11-09T04:30:00+00:00"      # 2026-11-08 23:30 EST
START_ALLOWANCE_H = 1.0   # ASSUMED: handover check, deploy, load and preflight before the first launch (not measured)
NOMINAL_H = {wq.OCT26_PREFIX + "window_override_methods": 0.3}   # the entry's own nominal (no evo-t2s timing)
SLACK_H = 24.0            # operator 2026-10-21 (later): every estimated entry ends at least this long before no-launch
# B6 (amendment 2026-10-21 later): hours from evo-x2 H-scale measured minutes x an evo-t2s / evo-x2 factor
B6_MODELS = tuple(wq.OCT26_MODELS.split(","))
B6_REAL_SEEDS = 5                                   # --plan strong (x2_r2_agent.SEEDS_STRONG)
B6_IGPU_WINDOW_ASSUMED = 32768                      # ASSUMED: not measured for these models on evo-t2s
B6_PULL_NOMINAL_H = 1.0                             # nominal, not measured
B6_X2_REAL = "x2_r2_scale_real.jsonl"
B6_X2_VAL = "x2_r2_scale_validation.jsonl"


def _ts(s):
    return datetime.fromisoformat(s).timestamp()


def available_h() -> float:
    return (_ts(NO_LAUNCH_AFTER_UTC) - _ts(WINDOW_START_UTC)) / 3600.0 - START_ALLOWANCE_H


def budget_h() -> float:
    """The launch budget: available hours minus the 24 h slack before the no-launch time."""
    return available_h() - SLACK_H


def _t2s_x2_factor(platform) -> dict:
    """evo-t2s / evo-x2 session-minute ratio per model at the same loaded window, from measured R2 sessions
    (evo-t2s: t2s_r2_real_{cpu,igpu}_v1 medians; evo-x2: x2_r2_real_v1/v1b medians). The factor used is the largest
    ratio (conservative for fitting)."""
    import policy_compare as pc
    t2s = pc._t2s_measured(REPO, platform)
    x2 = pc._x2_session_minutes(REPO)
    ratios = {m: e["default_min"] / x2[(m, e["D"])] for m, e in t2s.items()
              if e.get("default_min") and x2.get((m, e["D"]))}
    window = {m: e["D"] for m, e in t2s.items()}
    return {"ratios": ratios, "windows": window, "factor": max(ratios.values()) if ratios else None}


def _x2_scale_minutes() -> dict:
    """{(model, arm): median minutes of a completed 40-turn session} and {model: validation minutes (first to last
    ts_utc of the model's rows)} from the evo-x2 H-scale files."""
    import policy_compare as pc
    ag = pc.ag
    acc = {}
    for r in ag.read_rows(REPO / "results" / B6_X2_REAL):
        if r.get("record") == "r2a_turn":
            a = acc.setdefault((r["model_id"], r["arm_id"], r["attempt_id"]), [0.0, 0])
            a[0] += r.get("turn_wall_s") or 0.0
            a[1] += 1
    by = {}
    for (m, arm, _a), (sec, n) in acc.items():
        if n == pc.TURNS:
            by.setdefault((m, arm), []).append(sec / 60.0)
    import statistics as st
    real = {k: st.median(v) for k, v in by.items()}
    span = {}
    for r in ag.read_rows(REPO / "results" / B6_X2_VAL):
        if r.get("model_id") and r.get("ts_utc"):
            t = _ts(r["ts_utc"])
            lo, hi = span.get(r["model_id"], (t, t))
            span[r["model_id"]] = (min(lo, t), max(hi, t))
    return {"real": real, "val": {m: (hi - lo) / 60.0 for m, (lo, hi) in span.items()}}


def b6_estimates() -> dict:
    """{key: {"hours", "basis"}} for the B6 entries (key without the t2s_o26_ prefix). ESTIMATES: evo-x2 H-scale
    measured minutes x the evo-t2s / evo-x2 factor of the backend (CPU at 4096, iGPU at the default window); the iGPU
    window of these models is ASSUMED 32768. hours None = no measured rate for every model of the entry."""
    cpu, igpu = _t2s_x2_factor("evo-t2s-cpu"), _t2s_x2_factor("evo-t2s")
    x2 = _x2_scale_minutes()
    sfx = "_call2_notools"

    def fmt(f):
        return (f"factor {f['factor']:.2f} = largest evo-t2s/evo-x2 R2 session ratio at the same window ("
                + ", ".join(f"{m} {r:.2f} at {f['windows'][m]}" for m, r in sorted(f["ratios"].items())) + ")")

    def real(arm, f, label):
        mins = [x2["real"].get((m, arm + sfx)) for m in B6_MODELS]
        if None in mins or f["factor"] is None:
            return {"hours": None, "basis": "none"}
        return {"hours": B6_REAL_SEEDS * sum(mins) * f["factor"] / 60.0,
                "basis": f"ESTIMATE: {B6_REAL_SEEDS} seeds x evo-x2 {B6_X2_REAL} {arm} median session minutes ("
                         + ", ".join(f"{m} {v:.1f}" for m, v in zip(B6_MODELS, mins)) + f") x {fmt(f)}" + label}

    def val(f):
        mins = [x2["val"].get(m) for m in B6_MODELS]
        if None in mins or f["factor"] is None:
            return {"hours": None, "basis": "none"}
        return {"hours": sum(mins) * f["factor"] / 60.0,
                "basis": f"ESTIMATE: evo-x2 {B6_X2_VAL} minutes per model ("
                         + ", ".join(f"{m} {v:.1f}" for m, v in zip(B6_MODELS, mins)) + f") x {fmt(f)}"}

    return {"b6_model_pull": {"hours": B6_PULL_NOMINAL_H, "basis": "nominal (not measured)"},
            "b6_val_cpu": val(cpu), "b6_val_igpu": val(igpu),
            "b6_real_cpu_4096": real("ollama_ctx_4096", cpu, ""),
            "b6_real_igpu_default": real(f"ollama_ctx_{B6_IGPU_WINDOW_ASSUMED}", igpu,
                                         f"; iGPU-enabled window ASSUMED {B6_IGPU_WINDOW_ASSUMED}")}


# B3 v2 Intel laptop task accuracy (docs/FINDINGS.md "PRE-REGISTRATION: Intel laptop task accuracy", 2026-10-22):
# hours from MEASURED evo-t2s R2 session minutes (row t2s-r2-session-minutes) scaled by the B3 v2 task-turn profile,
# i.e. evo-x2 qwen3:14b B3 v2 task session minutes (validation: x2_evalset_tasks_val_v2; real: x2_evalset_tasks_real_v2
# at the matching window) x the qwen3:14b evo-t2s / evo-x2 R2 session ratio of the backend at its default window.
B3_X2_VAL = "x2_evalset_tasks_val_v2.jsonl"
B3_X2_REAL = "x2_evalset_tasks_real_v2.jsonl"
B3_MODEL = "qwen3:14b"
# evo-x2 tier whose task sessions stand for the evo-t2s default window: CPU 4096 = the same window; iGPU 32768 = the
# evo-x2 default (40960), the smallest measured window above 32768 (the task session's projected largest prompt is
# about 23,100 tokens, so neither truncates)
B3_X2_TIER = {"cpu": "ollama_ctx_4096", "igpu": "ollama_default"}
B3_PLATFORM = {"cpu": "evo-t2s-cpu", "igpu": "evo-t2s"}
B3_CPU_LONG_CTX_HIGH = 2.0   # ASSUMED (not measured): CPU validation negative control runs at qwen3:14b's 40960
                             # window, where CPU attention cost per token is above the 4096-window ratio used


def _x2_task_session_s(name, model, arm) -> list[float]:
    import policy_compare as pc
    ag = pc.ag
    rows = ag.read_rows(REPO / "results" / name)
    att = {s["attempt_id"] for s in ag.completed_sessions(rows) if s["model_id"] == model and s["arm_id"] == arm}
    walls = {}
    for r in rows:
        if r.get("record") == "r2a_turn" and r.get("attempt_id") in att:
            walls[r["attempt_id"]] = walls.get(r["attempt_id"], 0.0) + (r.get("turn_wall_s") or 0.0)
    return sorted(walls.values())


def b3_estimates() -> dict:
    """{key: {"hours", "high_h", "basis"}} for t2s_o26_b3_{val,real}_{cpu,igpu} (key without the prefix). ESTIMATES:
    point = evo-x2 qwen3:14b task session seconds x the qwen3:14b evo-t2s/evo-x2 R2 ratio of the backend; high = the
    slowest sessions x the largest ratio of the backend (both models), and for the CPU validation also x
    B3_CPU_LONG_CTX_HIGH (ASSUMED)."""
    import statistics as st
    import x2_r2_evalset as ev
    sfx = "_call2_notools"
    e = ev.tasks_v2_hours_estimate(ev.ag.read_rows(REPO / "results" / B3_X2_VAL), model=B3_MODEL)
    neg, pos = e["neg_session_s"], e["pos_session_s"]
    n = len(ev.TASKS_V2_T2S_SEEDS)
    out = {}
    for be in ("cpu", "igpu"):
        f = _t2s_x2_factor(B3_PLATFORM[be])
        r, rmax, win = f["ratios"].get(B3_MODEL), f["factor"], f["windows"].get(B3_MODEL)
        real = _x2_task_session_s(B3_X2_REAL, B3_MODEL, B3_X2_TIER[be] + sfx)
        if r is None or not real or not neg or not pos:
            out[f"b3_val_{be}"] = out[f"b3_real_{be}"] = {"hours": None, "high_h": None, "basis": "none"}
            continue
        extra = B3_CPU_LONG_CTX_HIGH if be == "cpu" else 1.0
        fac = (f"qwen3:14b evo-t2s/evo-x2 R2 session ratio {r:.2f} at {win} (row t2s-r2-session-minutes over the "
               f"evo-x2 R2 median); high = slowest x largest ratio {rmax:.2f}")
        out[f"b3_val_{be}"] = {
            "hours": (sum(neg) + sum(pos)) * r / 3600, "high_h": (max(neg) * len(neg) + max(pos)) * rmax * extra / 3600,
            "basis": f"ESTIMATE: evo-x2 {B3_X2_VAL} qwen3:14b sessions (negative control {sum(neg) / 60:.1f} min over "
                     f"{len(neg)}, positive control {sum(pos) / 60:.1f} min) x {fac}"
                     + (f" x {B3_CPU_LONG_CTX_HIGH} ASSUMED for the 40960-window negative control on the CPU"
                        if be == "cpu" else "")}
        out[f"b3_real_{be}"] = {
            "hours": n * st.mean(real) * r / 3600, "high_h": n * max(real) * rmax / 3600,
            "basis": f"ESTIMATE: {n} sessions x evo-x2 {B3_X2_REAL} qwen3:14b {B3_X2_TIER[be]} session (mean "
                     f"{st.mean(real) / 60:.1f} min, slowest {max(real) / 60:.1f} min, {len(real)} sessions) x {fac}; "
                     "runs only if the strict preflight admits qwen3:14b (a refusal takes minutes)"}
    return out


def b3_h() -> float:
    """Point-estimate hours of the staged B3 entries (0 for an unestimated one)."""
    return sum(v["hours"] or 0.0 for v in b3_estimates().values())


def b5_end_h() -> float:
    """Estimated end of A5 and the staged B5 cells (hours from the window start, after the start allowance)."""
    q = qwen_decision()
    return q["before_qwen_h"] + (q["qwen_h"] if q["fits"] else 0.0)


def select_b6(specs, start_h=None) -> dict:
    """Amendment 2026-10-21 (later): B6 entries in order; an estimated entry is staged if its gate is staged (or is
    not a B6 entry) and its estimated end is within budget_h(); one that does not fit is not staged; unestimated entries
    (and entries gated on one) are staged after every estimated one. Returns {"staged": [ids in queue order], "not_staged": {id: reason}}."""
    cum = b5_end_h() if start_h is None else start_h
    budget = budget_h()
    b6_ids = {s["id"] for s in specs}
    staged, later, skipped = [], [], {}
    for s in specs:
        g = s.get("gate")
        if g in b6_ids and g not in staged and g not in later:
            skipped[s["id"]] = f"gate {g} not staged"
            continue
        h = s.get("est")
        if h is None or g in later:     # unestimated, or gated on an unestimated entry: after every estimated one
            later.append(s["id"])
        elif cum + h <= budget:
            staged.append(s["id"])
            cum += h
        else:
            skipped[s["id"]] = (f"does not fit: {h} h from {round(cum, 1)} h ends at {round(cum + h, 1)} h, budget "
                                f"{round(budget, 1)} h")
    return {"staged": staged + later, "not_staged": skipped, "end_h": round(cum, 1)}


def qwen_decision() -> dict:
    """Amendment 2026-10-21 (later): after the llama3.1:8b cells at 5 seeds (and A5 before them), does qwen3:14b at 3
    seeds on the Intel CPU (default window 4096) end within the launch budget (no-launch time minus 24 h slack)?
    Policy set a, b, c, d, e, f. ESTIMATES from measured evo-t2s session minutes (harness/policy_compare.estimate_t2s,
    row t2s-r2-session-minutes)."""
    import policy_compare as pc
    pols = tuple(wq.OCT26_B5_POLICIES.split(","))
    qpols = tuple(wq.oct26_cell_policies(*wq.OCT26_QWEN_CELL).split(","))
    seeds5 = tuple(int(x) for x in wq.OCT26_B5_SEEDS.split(","))
    seeds3 = tuple(int(x) for x in wq.OCT26_QWEN_SEEDS.split(","))
    llama = {f"{p} {m}": round(pc.estimate_t2s(p, [m], seeds=seeds5, policies=pols)["models"][m]["total_h"], 1)
             for p, m in wq.OCT26_B5_CELLS}
    b3 = round(sum(round(v["hours"], 1) for v in b3_estimates().values() if v["hours"] is not None), 1)
    before = sum(NOMINAL_H.values()) + b3 + sum(llama.values())
    plat, model = wq.OCT26_QWEN_CELL
    q = pc.estimate_t2s(plat, [model], seeds=seeds3, policies=qpols)["models"][model]
    qwen_h = round(q["total_h"], 1)
    left = budget_h() - before
    return {"label": "ESTIMATE (hours)", "available_h": round(available_h(), 1), "slack_h": SLACK_H,
            "budget_h": round(budget_h(), 1), "a5_h": sum(NOMINAL_H.values()), "b3_h": b3,
            "llama_h": llama, "before_qwen_h": round(before, 1), "left_h": round(left, 1),
            "qwen_cell": f"{plat} {model}", "qwen_seeds": len(seeds3), "qwen_per_seed_h": round(q["per_seed_h"], 1),
            "qwen_h": qwen_h, "fits": qwen_h <= left, "policies": list(qpols), "llama_policies": list(pols)}


def plan(items=None) -> dict:
    """Entries in queue order with estimated hours, cumulative start/end, and whether each fits before the no-launch
    time. A B5 entry that starts but does not finish gets the number of whole seeds that fit (the job stops at a
    session boundary at the deadline; a cell counts for the verdict at 5 seeds, or at the first 3)."""
    import policy_compare as pc
    items = items if items is not None else json.loads(CANONICAL.read_text(encoding="utf-8"))
    cells = {wq.OCT26_PREFIX + "b5_" + wq._b5_slug(p, m): (p, m) for p, m in (*wq.OCT26_B5_CELLS, wq.OCT26_QWEN_CELL)}
    cell_pols = {i: tuple(wq.oct26_cell_policies(p, m).split(",")) for i, (p, m) in cells.items()}
    est = {i: pc.estimate_t2s(p, [m], policies=cell_pols[i]) for i, (p, m) in cells.items()}
    avail, budget, cum, rows = available_h(), budget_h(), 0.0, []
    for it in items:
        h = it.get("est_hours")
        basis = it.get("est_basis")
        if h is None and it["id"] in NOMINAL_H:
            h, basis = NOMINAL_H[it["id"]], "nominal"
        r = {"id": it["id"], "est_hours": h, "basis": basis, "start_h": round(cum, 1)}
        if "--policies" in it["cmd"]:
            r["policies"] = it["cmd"][it["cmd"].index("--policies") + 1]
        if h is None:
            r.update(fits="no estimate" if cum < avail else "not reached", end_h=None)
        else:
            end = cum + h
            r["end_h"] = round(end, 1)
            r["slack_h"] = round(avail - end, 1)
            if end <= budget:
                r["fits"] = "yes"
            elif end <= avail:
                r["fits"] = "inside the slack"
            elif cum >= avail:
                r["fits"] = "not reached"
            else:
                r["fits"] = "partial"
            if it["id"] in cells:
                p, m = cells[it["id"]]
                e = est[it["id"]]["models"][m]
                fixed = e["probe_h"]
                n = len(it["cmd"][it["cmd"].index("--seeds") + 1].split(","))
                k = 0 if cum >= budget else min(n, max(0, math.floor((budget - cum - fixed) / e["per_seed_h"])))
                r.update(per_seed_h=round(e["per_seed_h"], 1), seeds_that_fit=k, seeds_staged=n,
                         cell_counts=("yes (5 seeds)" if k >= 5 else "yes (first 3 seeds)" if k >= 3 else
                                      "no (reported only, below 3 seeds)" if k > 0 else "no"))
            cum = end
        rows.append(r)
    b5 = [r for r in rows if r["id"] in cells]
    b6_all = [s for s in wq.oct26_all_specs() if "_b6_" in s["id"]]
    b6_est = b6_estimates()
    for s in b6_all:
        e = b6_est.get(s["id"].removeprefix(wq.OCT26_PREFIX)) or {}
        s.update(est=None if e.get("hours") is None else round(e["hours"], 1), basis=e.get("basis", "none"))
    sel = select_b6(b6_all)
    not_staged = [{"id": s["id"], "est_hours": s["est"], "basis": s["basis"], "reason": sel["not_staged"][s["id"]]}
                  for s in b6_all if s["id"] in sel["not_staged"]]
    return {"label": "ESTIMATES (hours), not measurements", "window": WINDOW, "window_start_utc": WINDOW_START_UTC,
            "no_launch_after_utc": NO_LAUNCH_AFTER_UTC, "hard_deadline_utc": HARD_DEADLINE_UTC,
            "start_allowance_h": START_ALLOWANCE_H, "available_h": round(avail, 1), "slack_h": SLACK_H,
            "budget_h": round(budget, 1), "entries": rows, "not_staged": not_staged,
            "end_of_estimated_h": max((r["end_h"] for r in rows if r.get("end_h") is not None), default=0.0),
            "b5_total_h": round(sum(r["est_hours"] for r in b5), 1),
            "b5_cells_counting": [r["id"] for r in b5 if str(r.get("cell_counts", "no")).startswith("yes")],
            "b5_policies": {i: ",".join(v) for i, v in cell_pols.items()},
            "qwen_decision": qwen_decision(),
            "estimate_basis": pc.T2S_ESTIMATE_ASSUMPTIONS}


def plan_value(p=None) -> str:
    p = p or plan()
    parts = []
    for r in p["entries"]:
        s = f"{r['id']}{(' [policies ' + r['policies'] + ']') if r.get('policies') else ''} " \
            f"{('n/a' if r['est_hours'] is None else r['est_hours'])} h ({r['fits']}"
        if r.get("slack_h") is not None:
            s += f"; ends {r['end_h']} h, {r['slack_h']} h before no-launch"
        if "seeds_that_fit" in r:
            s += (f"; {r['per_seed_h']} h/seed, {r['seeds_that_fit']} of {r['seeds_staged']} seeds fit, cell counts "
                  f"{r['cell_counts']}")
        parts.append(s + ")")
    return (f"evo-t2s window {p['window'][0]} to {p['window'][1]} ESTIMATE: available {p['available_h']} h "
            f"({p['window_start_utc']} to no-launch {p['no_launch_after_utc']} minus an assumed "
            f"{p['start_allowance_h']} h start allowance); launch budget {p['budget_h']} h ({p['slack_h']} h slack "
            f"before no-launch); B5 policy sets per cell on evo-t2s (amendment 2026-10-22): llama3.1:8b cells "
            f"{wq.OCT26_B5_POLICIES}, qwen3:14b CPU cell {wq.OCT26_QWEN_POLICIES}; B5 staged cells {p['b5_total_h']} h; in queue order: "
            + "; ".join(parts) + f"; B5 cells that count: {', '.join(p['b5_cells_counting']) or 'none'}; "
            + qwen_text(p["qwen_decision"]) + "; B6 not staged: "
            + ("; ".join(f"{x['id']} {x['est_hours']} h ({x['reason']})" for x in p["not_staged"]) or "none")
            + "; B6 bases: " + "; ".join(f"{r['id']}: {r['basis']}" for r in p["entries"]
                                          if "_b6_" in r["id"] and r.get("basis"))
            + "; B5 basis: " + p["estimate_basis"])


def qwen_text(q) -> str:
    return (f"qwen3:14b decision (amendments 2026-10-21 later and 2026-10-22): after A5 {q['a5_h']} h, the B3 v2 task "
            f"entries {q['b3_h']} h and the llama3.1:8b cells at 5 "
            "seeds (" + ", ".join(f"{k} {v} h" for k, v in q["llama_h"].items()) + f"), {q['left_h']} h are left of "
            f"the {q['budget_h']} h launch budget; {q['qwen_cell']} at {q['qwen_seeds']} seeds on policies "
            f"{','.join(q['policies'])} needs {q['qwen_h']} h "
            f"({q['qwen_per_seed_h']} h/seed): " + ("fits, staged after the llama3.1:8b cells" if q["fits"] else
                                                   "does not fit, not staged; qwen3:14b on Intel is future work"))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "plan"
    if cmd == "build":
        CANONICAL.write_text(json.dumps(wq.build_oct26(), indent=1) + "\n", encoding="utf-8")
        print(f"wrote {CANONICAL} (staged, NOT loaded)")
        return 0
    if cmd == "dry-run":
        items = json.loads(CANONICAL.read_text(encoding="utf-8"))
        rep = wq.dry_run(items, prefix=wq.OCT26_PREFIX)
        rep["fresh_build_matches_file"] = items == wq.build_oct26()
        rep["retired_file_absent"] = not RETIRED.exists()
        print(json.dumps(rep, indent=1, default=str))
        return 0 if rep["ok"] and rep["fresh_build_matches_file"] and rep["retired_file_absent"] else 1
    if cmd == "plan":
        print(json.dumps(plan(), indent=1))
        return 0
    raise SystemExit(f"unknown command {cmd!r}")


if __name__ == "__main__":
    sys.exit(main())
