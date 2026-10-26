"""own-dse-v1: rank {hardware, model, window, KV, backend, routing policy} by total monthly cost for a workload.

Pre-registered in docs/EXPERIMENTS.md "2026-10-23: own-profiler ... and own-dse-v1" (Part B); row own-dse-v1.

Input: a workload (task mix, tasks/day), an accuracy target and a monthly cloud budget. Output: the candidates that
meet the target and the budget ranked by total monthly cost, and the quality x cloud spend x hardware cost Pareto
frontier.

Cost per 30-day month (cost_month, pure):
  machines    = ceil(tasks/day x local share x seconds per local task / 86,400), at least 1 if any task runs locally
                (one task at a time per machine; there is no measured concurrency row)
  hardware    = machines x list price / 36 months (impact-ref-machine-price; a device with no cited price cannot be
                priced and is not a candidate)
  electricity = tasks/month x local share x seconds per local task / 3600 x watts / 1000 x USD/kWh (PX2-power watts,
                energy-price-eia-commercial); active time only
  cloud       = tasks/month x cloud USD per task

The DSE ranks on PREDICTED quality and seconds (each Candidate's `pred`), validation scores the pick on its MEASURED
values (`meas`) against the brute-force best over measured candidates (validate_cell). Pure except build_b3_candidates,
build_long_candidates and own_dse_v1, which read committed result files through analysis/impact_model.py,
analysis/numbers_register.py and src/dse/own_profiler.py. Long and mixed workloads: see build_long_candidates_pure
and mix_candidates.
"""
from __future__ import annotations

import math
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

DAYS_PER_MONTH = 30
LIFE_MONTHS = 36
SECONDS_PER_DAY = 86400.0
VOLUMES = (100, 1000, 10000)
ACCURACY_TARGET = 0.90
CLOUD_BUDGET_USD_MONTH = 500.0
TARGET_GAP = 0.10
LONG_ROW = "long-task-local-accuracy"


@dataclass
class Estimate:
    """Per-task quality and resource use of one candidate on one workload."""
    quality: float              # share of tasks correct
    local_share: float          # share of tasks that run on the local machine
    local_s: float              # seconds of local machine time per LOCAL task
    cloud_usd_task: float       # cloud USD per task (averaged over all tasks)


@dataclass
class Candidate:
    hardware: str               # device name, or "none" for all-cloud
    model: str
    window: str
    kv: str
    backend: str
    policy: str
    pred: Estimate
    meas: Estimate | None = None
    hw_usd: float | None = None  # cited list price of one machine (None when no local hardware is used)
    watts: float | None = None
    notes: list = field(default_factory=list)
    name: str | None = None      # label override (mixed candidates: "short [<label>] + long [<label>]")

    @property
    def label(self) -> str:
        if self.name:
            return self.name
        return f"{self.hardware}/{self.model}/{self.window}/{self.kv}/{self.backend}/{self.policy}"


def cost_month(est: Estimate, tasks_day: float, hw_usd, watts, usd_kwh, days=DAYS_PER_MONTH,
               life_months=LIFE_MONTHS) -> dict:
    """Monthly cost parts of one estimate at a volume. Raises if local work needs a machine with no price."""
    local_s_day = tasks_day * est.local_share * est.local_s
    machines = 0
    if est.local_share > 0:
        if hw_usd is None or watts is None:
            raise ValueError("local share > 0 but no hardware price or watts")
        machines = max(1, math.ceil(local_s_day / SECONDS_PER_DAY))
    hardware = machines * (hw_usd or 0.0) / life_months
    electricity = local_s_day * days / 3600.0 * (watts or 0.0) / 1000.0 * usd_kwh
    cloud = tasks_day * days * est.cloud_usd_task
    return {"machines": machines, "hardware": hardware, "electricity": electricity, "cloud": cloud,
            "total": hardware + electricity + cloud}


def evaluate(c: Candidate, tasks_day, usd_kwh, which="pred") -> dict:
    est = c.pred if which == "pred" else c.meas
    return {"quality": est.quality, **cost_month(est, tasks_day, c.hw_usd, c.watts, usd_kwh)}


def feasible(ev: dict, accuracy_target, budget) -> bool:
    return ev["quality"] >= accuracy_target and ev["cloud"] <= budget


def rank(cands, tasks_day, accuracy_target, budget, usd_kwh, which="pred"):
    """[(candidate, evaluation)] of the feasible candidates, cheapest total first."""
    out = []
    for c in cands:
        if which == "meas" and c.meas is None:
            continue
        ev = evaluate(c, tasks_day, usd_kwh, which)
        if feasible(ev, accuracy_target, budget):
            out.append((c, ev))
    # ties: higher quality, then the simpler setup (all-cloud before a hybrid that keeps nothing local), then label
    out.sort(key=lambda x: (round(x[1]["total"], 9), -x[1]["quality"], 0 if x[0].hardware == "none" else 1,
                            x[0].label))
    return out


def pareto(cands, tasks_day, usd_kwh, which="pred"):
    """Non-dominated candidates on (quality max, cloud USD/month min, hardware USD/month min)."""
    evs = [(c, evaluate(c, tasks_day, usd_kwh, which)) for c in cands if which == "pred" or c.meas is not None]

    def dom(a, b):  # a dominates b
        ge = a["quality"] >= b["quality"] and a["cloud"] <= b["cloud"] and a["hardware"] <= b["hardware"]
        gt = a["quality"] > b["quality"] or a["cloud"] < b["cloud"] or a["hardware"] < b["hardware"]
        return ge and gt

    return [(c, e) for c, e in evs if not any(dom(e2, e) for _c2, e2 in evs)]


def takes_probe_input(c: Candidate) -> bool:
    """True when the candidate's predicted values come (at least partly) from the probe: any local machine. The
    all-cloud candidates are priced, not probed (pred is meas)."""
    return c.hardware != "none" or c.pred != c.meas


def cost_floor(c: Candidate, tasks_day, days=DAYS_PER_MONTH, life_months=LIFE_MONTHS) -> float:
    """Lowest predicted monthly total a local candidate can have under ANY probe input: one machine's hardware plus
    its predicted cloud USD (from the task-definition projection and the all-cloud rows, not the probe); electricity
    0 (pre-registered, docs/EXPERIMENTS.md 'own-dse-v1 trivial cells')."""
    machines = 1 if c.pred.local_share > 0 else 0
    return machines * (c.hw_usd or 0.0) / life_months + tasks_day * days * c.pred.cloud_usd_task


def cell_counts(cands, tasks_day, accuracy_target, budget, usd_kwh) -> dict:
    """Candidates, eligible on measured values (accuracy target and cloud budget), and eligible using a local machine."""
    elig = rank(cands, tasks_day, accuracy_target, budget, usd_kwh, "meas")
    return {"n_candidates": len(cands), "n_eligible": len(elig),
            "n_eligible_local": sum(1 for _c, e in elig if e["machines"] > 0)}


def trivial_cell(cands, tasks_day, accuracy_target, budget, usd_kwh, tol=TARGET_GAP, local_by="share") -> dict:
    """Pre-registered (docs/EXPERIMENTS.md "own-dse-v1 trivial cells", before any count was computed): a computed cell
    is TRIVIAL if (1) at most one candidate is eligible on measured values, or (2) the brute-force best takes no probe
    input and every candidate using a local machine has a cost floor above the best's measured total by more than tol.
    local_by "share" (the pre-registered wording, used for the row) or "label" (hardware != "none", the first
    implementation; reported next to it, not counted). Returns {"trivial": bool, "reason": str, "floor_min"}."""
    brute = rank(cands, tasks_day, accuracy_target, budget, usd_kwh, "meas")
    if len(brute) <= 1:
        return {"trivial": True, "reason": f"{len(brute)} eligible", "floor_min": None}
    best_c, best_ev = brute[0]
    # "uses a local machine" = predicted local share > 0 (at least one machine; the share comes from the task-definition
    # projection, not the probe), the same reading as the per-cell count (machines > 0). A per-task hybrid that routes
    # every task to the cloud (share 0, 0 machines) does not use a local machine and its prediction takes no probe input.
    local = [c for c in cands if (c.pred.local_share > 0 if local_by == "share" else c.hardware != "none")]
    floors = [cost_floor(c, tasks_day) for c in local]
    fmin = min(floors) if floors else None
    if not takes_probe_input(best_c) and all(f > best_ev["total"] * (1 + tol) for f in floors):
        return {"trivial": True, "floor_min": fmin,
                "reason": (f"best takes no probe input and every local candidate's cost floor (lowest USD "
                           f"{fmin:,.2f}/month) is more than {tol * 100:.0f}% above it" if floors else
                           "best takes no probe input and no candidate uses a local machine")}
    if takes_probe_input(best_c):
        why = "the brute-force best takes probe input"
    else:
        why = (f"lowest local cost floor USD {fmin:,.2f}/month is not more than {tol * 100:.0f}% above the best's "
               f"measured USD {best_ev['total']:,.2f}, so a probe input could change the pick")
    arg = min(range(len(local)), key=lambda i: floors[i]) if floors else None
    lowest = None if arg is None else {"label": local[arg].label, "pred_quality": local[arg].pred.quality,
                                       "meas_quality": local[arg].meas.quality if local[arg].meas else None}
    return {"trivial": False, "floor_min": fmin, "reason": why, "lowest_floor": lowest}


def validate_cell(cands, tasks_day, accuracy_target, budget, usd_kwh) -> dict:
    """validate_cell_core plus the per-cell counts and the trivial-cell label (not counted as validation if trivial).
    A cell with no candidate eligible on measured values is computed (the counts exist) and TRIVIAL by rule 1 of the
    pre-registered definition ("at most one candidate is eligible"); it has no pick and no gap."""
    v = validate_cell_core(cands, tasks_day, accuracy_target, budget, usd_kwh)
    v.update(cell_counts(cands, tasks_day, accuracy_target, budget, usd_kwh))
    if v["status"] == "no measured config meets the target and budget":
        v.update({"status": "computed", "no_eligible": True, "trivial": True, "trivial_reason": "0 eligible",
                  "local_floor_min": None, "trivial_by_label": True})
        return v
    if v["status"] == "DSE found no feasible candidate":
        v.update({"status": "computed", "dse_none": True})
    if v["status"] == "computed":
        t = trivial_cell(cands, tasks_day, accuracy_target, budget, usd_kwh)
        v.update({"trivial": t["trivial"], "trivial_reason": t["reason"], "local_floor_min": t["floor_min"],
                  "lowest_floor": t.get("lowest_floor"),
                  "trivial_by_label": trivial_cell(cands, tasks_day, accuracy_target, budget, usd_kwh,
                                                   local_by="label")["trivial"]})
    return v


def validate_cell_core(cands, tasks_day, accuracy_target, budget, usd_kwh) -> dict:
    """DSE pick (predicted ranking) scored on measured values vs the brute-force best over measured candidates."""
    pred_rank = rank(cands, tasks_day, accuracy_target, budget, usd_kwh, "pred")
    brute = rank(cands, tasks_day, accuracy_target, budget, usd_kwh, "meas")
    if not brute:
        return {"status": "no measured config meets the target and budget", "met": False}
    best_c, best_ev = brute[0]
    if not pred_rank:
        return {"status": "DSE found no feasible candidate", "met": False, "best": best_c.label,
                "best_total": best_ev["total"]}
    pick, pick_pred = pred_rank[0]
    pm = evaluate(pick, tasks_day, usd_kwh, "meas")
    ok = feasible(pm, accuracy_target, budget)
    gap = (pm["total"] - best_ev["total"]) / best_ev["total"] if best_ev["total"] > 0 else (
        0.0 if pm["total"] == 0 else math.inf)
    loc = [(c, e) for c, e in brute if e["machines"] > 0]
    return {"status": "computed", "best_local": loc[0][0].label if loc else None,
            "best_local_total": loc[0][1]["total"] if loc else None, "pick": pick.label, "pick_pred_total": pick_pred["total"],
            "pick_meas_total": pm["total"], "pick_meas_quality": pm["quality"], "pick_meas_feasible": ok,
            "best": best_c.label, "best_total": best_ev["total"], "best_quality": best_ev["quality"],
            "gap": gap, "met": ok and gap <= TARGET_GAP,
            "pareto_pred": [c.label for c, _e in pareto(cands, tasks_day, usd_kwh, "pred")]}


# ───────────────────────────────────────────────────────────── candidate builder (reads committed files)
def _impact(repo):
    a = str(Path(repo) / "analysis")
    if a not in sys.path:
        sys.path.insert(0, a)
    import impact_model as im
    import numbers_register as nr
    return im, nr


def _window(tier, loaded):
    if tier == "ollama_default":
        return f"{loaded} (Ollama default)" if loaded else "Ollama default"
    return str(tier.split("_")[2])


def build_b3_candidates(repo) -> tuple:
    """(candidates, inputs) for the short B3 v2 workload: evo-x2 qwen3:14b local tiers, all-cloud tiers, and the
    measured per-task hybrids. Predictions from the own-profiler probe (seed 20260901 prefix to 600 s, ratios from
    evo-t2s configs only); measurements on the matched seeds of hybrid-route-at-start-measured."""
    from src.dse import own_profiler as op
    im, nr = _impact(repo)
    repo = Path(repo)
    price = im.ref_machine(repo)
    watts_by = nr.compute_px2_package_power(repo)["detail"]
    watts = watts_by["qwen3-14b"]
    import json
    eia = json.loads((repo / nr.ENERGY_PRICE_EIA_FILE).read_text(encoding="utf-8"))
    usd_kwh = float(eia["value_cents_per_kwh"]) / 100.0
    route = im.route_at_start_measured(repo)
    if route["status"] != "VERIFIED":
        raise FileNotFoundError("hybrid-route-at-start-measured not VERIFIED")
    seeds = set(route["seeds"])
    local = im.replay_local_tasks(repo)
    ac = im.replay_cloud_tasks(repo)

    sess = op.session_feats(op.load_session_configs(repo))
    ratios = op.fit_ratios([f for f in sess if f["device"] == "evo-t2s"])
    probe = {f["arm"]: f for f in sess if f["family"] == "b3v2"}

    cloud_meas = {}
    for m, tasks in ac["models"].items():
        xs = [v for k, v in tasks.items() if k[0] in seeds]
        cloud_meas[m] = Estimate(sum(v["ok"] for v in xs) / len(xs), 0.0, 0.0,
                                 statistics.mean(v["usd"] for v in xs))
    cands = []
    for m, est in sorted(cloud_meas.items()):
        cands.append(Candidate("none", m, "provider", "provider", "Anthropic API", "all-cloud", pred=est, meas=est,
                               notes=["cloud quality and USD from all_cloud_b3_v1 (priced, not probed)"]))
    pred_local = {}
    for tier in im.HYBRID_ALL_TIERS:
        if tier not in local or tier not in probe:
            continue
        xs = [v for k, v in local[tier].items() if k[0] in seeds]
        loaded = next((v["loaded"] for v in xs if v.get("loaded")), None)
        meas = Estimate(sum(v["ok"] for v in xs) / len(xs), 1.0, statistics.mean(v["wall_s"] for v in xs), 0.0)
        ps, pa = op.predict(probe[tier], ratios)
        pred = Estimate(pa, 1.0, ps, 0.0)
        pred_local[tier] = pred
        policy = "local, num_keep" if tier.endswith("numkeep") else "local only"
        cands.append(Candidate("evo-x2", im.B3_V2_MODEL, _window(tier, loaded), "f16 (Ollama default)", "Ollama",
                               policy, pred=pred, meas=meas, hw_usd=price["usd"], watts=watts))
    for cell in route["cells"]:
        tier, m, h = cell["tier"], cell["model"], cell["hybrid"]
        if tier not in pred_local:
            continue
        local_tasks = [t for t in h["tasks"].values() if t["route"] == "local"]
        share_m = len(local_tasks) / h["n"]
        local_s = statistics.mean(t["seconds"] for t in local_tasks) if local_tasks else 0.0
        meas = Estimate(h["quality"], share_m, local_s, h["usd_task"])
        share_p = 1.0 - h["routed"] / h["n"]  # projected from task definitions before any call
        pl, cl = pred_local[tier], cloud_meas[m]
        pred = Estimate(share_p * pl.quality + (1 - share_p) * cl.quality, share_p, pl.local_s,
                        (1 - share_p) * cl.cloud_usd_task)
        cands.append(Candidate("evo-x2", f"{im.B3_V2_MODEL} + {m}", tier.split("_")[2], "f16 (Ollama default)",
                               "Ollama + Anthropic API", "per-task route at start + window flag", pred=pred, meas=meas,
                               hw_usd=price["usd"], watts=watts))
    inputs = {"usd_kwh": usd_kwh, "hw_usd": price["usd"], "hw_label": price["label"], "watts": watts,
              "seeds": sorted(seeds), "ratios": ratios}
    return cands, inputs


# ───────────────────────────────────────────────────────────── long-task and mixed candidates
LONG_CLOUD_ROW = "long-task-cloud-priced-vs-billed"
MIX_LONG_SHARE = 0.5   # pre-registered: "Mixed: 50% short B3 v2 tasks and 50% long tasks by count"
LONG_TIERS = {         # x2_longtask ARMS: window label and KV label per arm
    "ollama_default": ("40960 (Ollama default)", "f16 (Ollama default)"),
    "kv_q8_0": ("40960", "q8_0 (flash attention on)"),
}


def build_long_candidates_pure(local: dict, cloud: dict, proj: dict, windows: dict, hw_usd, watts) -> tuple:
    """Pure. (candidates, notes) for the long workload.

    local {arm: {(seed, task): {ok, wall_s, flag}}} (flag = the task's prompt exceeded the loaded window);
    cloud {model: {(seed, task): {ok, usd}}} (usd = billed); proj {(seed, task): projected prompt in model tokens from
    the task definition (oracle transcript)}; windows {arm: loaded window}.

    Local-only: measured accuracy and seconds over every task run of the arm. There is no long-task probe (the
    own-profiler probe covers B3 short-task configs only and the pre-registration names none for long tasks), so the
    DSE sees the measured values (pred is meas). All-cloud: measured accuracy and mean billed USD over the tasks the
    cloud run measured (priced, not probed). Per-task hybrid: the pre-registered policy (route at start: a task whose
    projected prompt exceeds the arm's window goes to the cloud from its first call; window flag: a local task whose
    prompt exceeded the loaded window is escalated to the cloud), over every local task run; a routed or flagged task
    needs that task's cloud row, else the hybrid is not built (noted). Predicted share = 1 - routed/n."""
    cands, notes = [], []
    cloud_est = {}
    for m in sorted(cloud):
        xs = list(cloud[m].values())
        if not xs:
            continue
        cloud_est[m] = Estimate(sum(v["ok"] for v in xs) / len(xs), 0.0, 0.0, statistics.fmean(v["usd"] for v in xs))
        cands.append(Candidate("none", m, "provider", "provider", "Anthropic API", "all-cloud", pred=cloud_est[m],
                               meas=cloud_est[m], notes=[f"long-task cloud run, {len(xs)} tasks, billed USD"]))
    for arm in [a for a in LONG_TIERS if a in local] + sorted(a for a in local if a not in LONG_TIERS):
        tasks = local[arm]
        if not tasks:
            continue
        win_label, kv_label = LONG_TIERS.get(arm, (str(windows.get(arm)), "unknown"))
        meas = Estimate(sum(v["ok"] for v in tasks.values()) / len(tasks), 1.0,
                        statistics.fmean(v["wall_s"] for v in tasks.values()), 0.0)
        cands.append(Candidate("evo-x2", "qwen3:14b", win_label, kv_label, "Ollama", "local only", pred=meas,
                               meas=meas, hw_usd=hw_usd, watts=watts,
                               notes=["no long-task probe: DSE sees measured values"]))
        missing_proj = [k for k in tasks if k not in proj]
        if missing_proj:
            notes.append(f"{arm}: no projected prompt for {len(missing_proj)} task runs, hybrids not built")
            continue
        w = windows[arm]
        n = len(tasks)
        routed = {k for k in tasks if proj[k] > w}
        flagged = {k for k in tasks if k not in routed and tasks[k]["flag"]}
        for m in sorted(cloud_est):
            need = [k for k in routed | flagged if k not in cloud[m]]
            if need:
                notes.append(f"{arm} + {m}: {len(need)} routed or flagged task runs have no cloud row, hybrid not built")
                continue
            ok = usd = 0.0
            kept = []
            for k, v in tasks.items():
                if k in routed or k in flagged:
                    ok += cloud[m][k]["ok"]
                    usd += cloud[m][k]["usd"]
                else:
                    ok += v["ok"]
                if k not in routed:
                    kept.append(v["wall_s"])
            meas_h = Estimate(ok / n, len(kept) / n, statistics.fmean(kept) if kept else 0.0, usd / n)
            share_p = 1.0 - len(routed) / n
            cl = cloud_est[m]
            pred_h = Estimate(share_p * meas.quality + (1 - share_p) * cl.quality, share_p, meas.local_s,
                              (1 - share_p) * cl.cloud_usd_task)
            cands.append(Candidate("evo-x2", f"qwen3:14b + {m}", win_label, kv_label, "Ollama + Anthropic API",
                                   "per-task route at start + window flag", pred=pred_h, meas=meas_h, hw_usd=hw_usd,
                                   watts=watts, notes=[f"routed {len(routed)}/{n}, flagged {len(flagged)}/{n}"]))
        notes.append(f"{arm} (window {w}): routed {len(routed)}/{n}, flagged {len(flagged)}/{n}")
    return cands, notes


def _mix_est(a: Estimate, b: Estimate, long_share=MIX_LONG_SHARE) -> Estimate:
    sa, sb = (1 - long_share) * a.local_share, long_share * b.local_share
    share = sa + sb
    return Estimate((1 - long_share) * a.quality + long_share * b.quality, share,
                    (sa * a.local_s + sb * b.local_s) / share if share > 0 else 0.0,
                    (1 - long_share) * a.cloud_usd_task + long_share * b.cloud_usd_task)


def mix_candidates(short: list, long: list, long_share=MIX_LONG_SHARE) -> list:
    """Pure. Every (short candidate, long candidate) pair as one mixed candidate: the task type is known from the task
    definition at the start, short tasks run on the short candidate's setup and long tasks on the long one's (both on
    the same evo-x2 machines when both are local; machines from the pooled local seconds)."""
    out = []
    for a in short:
        for b in long:
            hw = a.hardware if a.hardware != "none" else b.hardware
            out.append(Candidate(
                hw, f"short {a.model} | long {b.model}", f"short {a.window} | long {b.window}",
                f"short {a.kv} | long {b.kv}", f"short {a.backend} | long {b.backend}",
                f"short {a.policy} | long {b.policy}", pred=_mix_est(a.pred, b.pred, long_share),
                meas=_mix_est(a.meas, b.meas, long_share), hw_usd=a.hw_usd or b.hw_usd, watts=a.watts or b.watts,
                name=f"short [{a.label}] + long [{b.label}]"))
    return out


def build_long_candidates(repo, hw_usd, watts) -> tuple:
    """(candidates, inputs) for the long workload from results/x2_longtask_local_v1.jsonl and
    results/longtask_cloud_v1.jsonl (the attempt that wrote run_end), through analysis/numbers_register.py's readers.
    Projected prompt per task = the largest step of its oracle transcript (x2_longtask.oracle_prompt_steps, model
    tokens), the long-task analogue of the B3 v2 task-definition projection."""
    _im, nr = _impact(repo)
    repo = Path(repo)
    lt, xl = nr._longtask_modules()
    ag = nr._r2_agent_module()
    rows = nr._finished_attempt_rows(ag.read_rows(repo / nr.LONGTASK_LOCAL_FILE))
    local, windows = {}, {}
    for arm in xl.ARM_ORDER:
        g = xl.task_rows(rows, xl.ARMS[arm]["arm_id"])
        if not g:
            continue
        loaded = {r.get("loaded_context") for r in g}
        if len(loaded) != 1 or None in loaded:
            raise ValueError(f"{arm}: loaded window {sorted(map(str, loaded))} not one value")
        windows[arm] = loaded.pop()
        local[arm] = {(r["seed"], r["task_id"]): {"ok": bool(r["correct"]), "wall_s": float(r["task_wall_s"]),
                                                   "flag": bool(r.get("over_loaded_window"))} for r in g}
    seeds = sorted({k[0] for t in local.values() for k in t})
    proj = {}
    for s in seeds:
        for (task, _env), steps in zip(lt.build_tasks(s), xl.oracle_prompt_steps(s)):
            proj[(s, task.task_id)] = max(steps)
    crow = nr._finished_attempt_rows(nr._read_jsonl(repo / nr.LONGTASK_CLOUD_FILE))
    cloud = {}
    for r in crow:
        if r.get("record") == "longtask_cloud_task":
            cloud.setdefault(r["model_id"], {})[(r["seed"], r["task_id"])] = {"ok": bool(r["correct"]),
                                                                              "usd": float(r["usd_billed"])}
    cands, notes = build_long_candidates_pure(local, cloud, proj, windows, hw_usd, watts)
    return cands, {"seeds": seeds, "windows": windows, "notes": notes,
                   "cloud_seeds": sorted({k[0] for t in cloud.values() for k in t}),
                   "cloud_n": {m: len(t) for m, t in cloud.items()}, "max_proj": max(proj.values())}


def own_dse_v1(repo, volumes=VOLUMES, accuracy_target=ACCURACY_TARGET, budget=CLOUD_BUDGET_USD_MONTH) -> dict:
    """Everything row own-dse-v1 reports: the short B3 cells, and the long and mixed cells when rows
    long-task-local-accuracy and long-task-cloud-priced-vs-billed are VERIFIED (PENDING otherwise)."""
    im, _nr = _impact(repo)
    cands, inputs = build_b3_candidates(repo)
    cells = {}
    for v in volumes:
        cells[("short", v)] = validate_cell(cands, v, accuracy_target, budget, inputs["usd_kwh"])
    missing = [r for r in (LONG_ROW, LONG_CLOUD_ROW) if not im.register_has(repo, r)]
    long_c, mixed_c, long_inputs = [], [], None
    if missing:
        for w in ("long", "mixed"):
            for v in volumes:
                cells[(w, v)] = {"status": f"PENDING: row {', '.join(missing)} not VERIFIED", "met": False}
    else:
        long_c, long_inputs = build_long_candidates(repo, inputs["hw_usd"], inputs["watts"])
        mixed_c = mix_candidates(cands, long_c)
        for w, cs in (("long", long_c), ("mixed", mixed_c)):
            for v in volumes:
                cells[(w, v)] = validate_cell(cs, v, accuracy_target, budget, inputs["usd_kwh"])
    return {"cells": cells, "inputs": inputs, "n_candidates": len(cands), "candidates": cands,
            "long_candidates": long_c, "mixed_candidates": mixed_c, "long_inputs": long_inputs,
            "accuracy_target": accuracy_target, "budget": budget, "volumes": list(volumes)}


def _cell_text(w, v, c) -> str:
    head = f"{w} {v:,}/day: "
    counts = (f"{c['n_candidates']} candidates, {c['n_eligible']} eligible (measured accuracy >= target and cloud "
              f"within budget), eligible using a local machine: {c['n_eligible_local']}")
    if c.get("no_eligible"):
        return head + f"TRIVIAL (0 eligible; not counted as validation); {counts}; no pick, no gap"
    label = (f"TRIVIAL ({c['trivial_reason']}; not counted as validation)" if c["trivial"]
             else f"NON-TRIVIAL ({c['trivial_reason']}"
             + (f"; lowest-floor local candidate {c['lowest_floor']['label']}, predicted accuracy "
                f"{c['lowest_floor']['pred_quality']:.3f}, measured {c['lowest_floor']['meas_quality']:.3f}"
                if c.get('lowest_floor') and c['lowest_floor']['meas_quality'] is not None else "") + ")")
    alt = "TRIVIAL" if c["trivial_by_label"] else "NON-TRIVIAL"
    if c.get("dse_none"):
        return (head + f"{label}; {counts}; under the hardware-label reading (first implementation, not the "
                f"pre-registered wording) {alt}; DSE found no feasible candidate on predicted values; brute-force best "
                f"{c['best']} USD {c['best_total']:,.2f} -> NOT within 10%")
    return (head + f"{label}; {counts}; "
            f"under the hardware-label reading (first implementation, not the pre-registered wording) {alt}; "
            f"DSE pick {c['pick']} (predicted USD {c['pick_pred_total']:,.2f}/month, measured "
            f"USD {c['pick_meas_total']:,.2f}, measured accuracy {c['pick_meas_quality']:.3f}); brute-force "
            f"best {c['best']} USD {c['best_total']:,.2f} (accuracy {c['best_quality']:.3f}); gap "
            f"{c['gap'] * 100:.1f}% -> {'within 10%' if c['met'] else 'NOT within 10%'}; cheapest measured "
            f"feasible option using a local machine: "
            + (f"{c['best_local']} USD {c['best_local_total']:,.2f}" if c["best_local"] else "none"))


def _long_inputs_text(d) -> str:
    li = d.get("long_inputs")
    if not li:
        return ""
    return (f"long workload: {len(d['long_candidates'])} candidates (evo-x2 qwen3:14b local arms "
            + ", ".join(f"{a} window {w}" for a, w in li["windows"].items())
            + f", seeds {', '.join(map(str, li['seeds']))}, seconds per local task = measured task wall-clock mean; "
            "all-cloud "
            + ", ".join(f"{m} ({n} tasks)" for m, n in sorted(li["cloud_n"].items()))
            + f", seed {', '.join(map(str, li['cloud_seeds']))}, billed USD per task; per-task route at start + window "
            f"flag at the arm's window, largest projected prompt {li['max_proj']:,} model tokens: "
            + "; ".join(li["notes"])
            + "); no long-task probe exists (own-profiler-probe covers B3 short-task configs only and the "
            "pre-registration names none), so the long local candidates' predicted values are their measured values "
            "(LABELLED); mixed workload: 50% short and 50% long tasks by count (pre-registered), "
            f"{len(d['mixed_candidates'])} candidates = every short candidate for the short tasks x every long candidate "
            "for the long tasks (task type known at start; LABELLED: the pre-registration does not fix the mixed "
            "candidate set); ")


def own_dse_v1_text(d) -> str:
    inp = d["inputs"]
    parts = []
    for (w, v), c in d["cells"].items():
        if c.get("status") != "computed":
            parts.append(f"{w} {v:,}/day: {c['status']}")
            continue
        parts.append(_cell_text(w, v, c))
    computed = [c for c in d["cells"].values() if c.get("status") == "computed"]
    nontriv = [c for c in computed if not c["trivial"]]
    met = sum(1 for c in nontriv if c["met"])
    pending = len(d["cells"]) - len(computed)
    if nontriv:
        verdict = (f"pre-registered target within 10% in every non-trivial computed cell: "
                   f"{'MET' if met == len(nontriv) else 'MISSED'} ({met}/{len(nontriv)} non-trivial cells")
    else:
        verdict = "validation: 0 non-trivial cells, target not tested ("
    verdict += (f"; {len(computed) - len(nontriv)} TRIVIAL computed cells not counted; {pending} PENDING); the earlier "
                f"'MET (3/3 computed cells, 6 PENDING)' counted trivial cells")
    return (f"accuracy target {d['accuracy_target']:.2f}, cloud budget USD {d['budget']:,.0f}/month, 30-day month; "
            f"short workload: {d['n_candidates']} measured candidates (evo-x2 qwen3:14b local tiers, all-cloud, "
            f"per-task hybrids; seeds {', '.join(map(str, inp['seeds']))}); "
            + _long_inputs_text(d)
            + f"hardware USD {inp['hw_usd']:,.0f} ({inp['hw_label']}, "
            f"impact-ref-machine-price) / 36 months; electricity {inp['watts']:.1f} W (PX2-power qwen3-14b, measured "
            f"on evo-t2s, applied to evo-x2: ASSUMED) x USD {inp['usd_kwh']:.4f}/kWh (energy-price-eia-commercial), "
            f"active time only; one task at a time per machine (ASSUMED); DSE ranks on own-profiler probe predictions "
            f"(ratios from evo-t2s), validation on measured values; trivial cell (pre-registered 2026-10-23): at most "
            f"one eligible, or the brute-force best takes no probe input and every candidate using a local machine "
            f"(predicted local share > 0) has a cost floor (one machine USD {inp['hw_usd'] / LIFE_MONTHS:,.2f}/month "
            f"plus its predicted cloud USD) more than 10% above it; "
            + "; ".join(parts) + "; " + verdict)
