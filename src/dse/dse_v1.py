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
values (`meas`) against the brute-force best over measured candidates (validate_cell). Pure except build_b3_candidates
and own_dse_v1, which read committed result files through analysis/impact_model.py and src/dse/own_profiler.py.
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

    @property
    def label(self) -> str:
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


def validate_cell(cands, tasks_day, accuracy_target, budget, usd_kwh) -> dict:
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


def own_dse_v1(repo, volumes=VOLUMES, accuracy_target=ACCURACY_TARGET, budget=CLOUD_BUDGET_USD_MONTH) -> dict:
    """Everything row own-dse-v1 reports: the short B3 cells computed, long and mixed PENDING unless the long-task
    local row is VERIFIED (and even then PENDING until a long-task candidate builder exists in this module)."""
    im, _nr = _impact(repo)
    cands, inputs = build_b3_candidates(repo)
    cells = {}
    for v in volumes:
        cells[("short", v)] = validate_cell(cands, v, accuracy_target, budget, inputs["usd_kwh"])
    long_ok = im.register_has(repo, LONG_ROW)
    for w in ("long", "mixed"):
        for v in volumes:
            cells[(w, v)] = {"status": (f"PENDING: row {LONG_ROW} not VERIFIED" if not long_ok else
                                        "PENDING: long-task candidate builder not written yet"), "met": False}
    return {"cells": cells, "inputs": inputs, "n_candidates": len(cands), "candidates": cands,
            "accuracy_target": accuracy_target, "budget": budget, "volumes": list(volumes)}


def own_dse_v1_text(d) -> str:
    inp = d["inputs"]
    parts = []
    for (w, v), c in d["cells"].items():
        if c.get("status") != "computed":
            parts.append(f"{w} {v:,}/day: {c['status']}")
            continue
        parts.append(f"{w} {v:,}/day: DSE pick {c['pick']} (predicted USD {c['pick_pred_total']:,.2f}/month, measured "
                     f"USD {c['pick_meas_total']:,.2f}, measured accuracy {c['pick_meas_quality']:.3f}); brute-force "
                     f"best {c['best']} USD {c['best_total']:,.2f} (accuracy {c['best_quality']:.3f}); gap "
                     f"{c['gap'] * 100:.1f}% -> {'within 10%' if c['met'] else 'NOT within 10%'}; cheapest measured "
                     f"feasible option using a local machine: "
                     + (f"{c['best_local']} USD {c['best_local_total']:,.2f}" if c["best_local"] else "none"))
    computed = [c for c in d["cells"].values() if c.get("status") == "computed"]
    met = sum(1 for c in computed if c["met"])
    pending = len(d["cells"]) - len(computed)
    return (f"accuracy target {d['accuracy_target']:.2f}, cloud budget USD {d['budget']:,.0f}/month, 30-day month; "
            f"{d['n_candidates']} measured candidates (evo-x2 qwen3:14b local tiers, all-cloud, per-task hybrids; seeds "
            f"{', '.join(map(str, inp['seeds']))}); hardware USD {inp['hw_usd']:,.0f} ({inp['hw_label']}, "
            f"impact-ref-machine-price) / 36 months; electricity {inp['watts']:.1f} W (PX2-power qwen3-14b, measured "
            f"on evo-t2s, applied to evo-x2: ASSUMED) x USD {inp['usd_kwh']:.4f}/kWh (energy-price-eia-commercial), "
            f"active time only; one task at a time per machine (ASSUMED); DSE ranks on own-profiler probe predictions "
            f"(ratios from evo-t2s), validation on measured values; "
            + "; ".join(parts)
            + f"; pre-registered target within 10% in every computed cell: "
              f"{'MET' if computed and met == len(computed) else 'MISSED'} ({met}/{len(computed)} computed cells, "
              f"{pending} PENDING)")
