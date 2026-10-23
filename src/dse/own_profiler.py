"""own-profiler: predict a configuration's task accuracy and mean seconds per task from a probe of at most 600 s.

Pre-registered in docs/EXPERIMENTS.md "2026-10-23: own-profiler ... and own-dse-v1" (Part A); row own-profiler-loo.

A configuration is one (device, run family, model, arm) cell of a committed session run. The probe is the turn prefix
of the lowest committed seed's completed session whose cumulative turn_wall_s is at most PROBE_BUDGET_S (the first
turn's wall clock includes model load, so load is inside the budget). No dedicated probe job was ever run: offline the
probe is simulated by that prefix, which is exactly what a real probe would record on the same harness and seed.

Predictor: seconds per task = probe seconds per turn x R_s, accuracy = min(1, probe accuracy x R_a). R_s and R_a are
medians of (actual / probe) over the TRAINING device's configurations in the same stratum ("overflow" if the probe's
projected final prompt exceeds the loaded window, else "fits"), pooled over all training configurations if the
stratum has none. Targets are computed from the sessions whose seed is not the probe seed.

Everything here is pure except the loaders (load_session_configs, load_single_configs, load_scale_probe), which read
committed result files only.
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

PROBE_BUDGET_S = 600.0

# (device, family, files merged in order, grade field). Files that do not exist are skipped by the loader; the
# register row lists every one as a data file, so a missing file makes the row PENDING rather than silently smaller.
SESSION_FAMILIES = [
    ("evo-x2", "b3v2", ["results/x2_evalset_tasks_real_v2.jsonl"], "end_state_ok"),
    ("evo-x2", "r2", ["results/x2_r2_real_v1.jsonl", "results/x2_r2_real_v1b.jsonl",
                      "results/x2_r2_qwen3_8b_seeds_06_10.jsonl"], "task_tool_ok"),
    ("evo-x2", "b5", ["results/x2_policy_compare.jsonl", "results/x2_policy_compare_fa_control.jsonl"], "task_tool_ok"),
    ("evo-x2", "scale", ["results/x2_r2_scale_real.jsonl"], "task_tool_ok"),
    ("evo-x2", "mistral", ["results/x2_r2_family_mistral_small32_real.jsonl"], "task_tool_ok"),
    ("evo-t2s", "r2_cpu", ["results/t2s_r2_real_cpu_v1.jsonl"], "task_tool_ok"),
    ("evo-t2s", "r2_igpu", ["results/t2s_r2_real_igpu_v1.jsonl"], "task_tool_ok"),
]
SESSION_MODES = ("real", "b5")
SINGLE_FILES = [("evo-x2", "results/x2_outcome_table_v3.jsonl"), ("evo-t2s", "results/t2s_outcome_table_full.jsonl")]
SCALE_PROBE_FILE = "results/x2_scale_timing_probe.jsonl"
BLADE_TASK_FILE = "results/blade_r2_real_v1.jsonl"
SESSION_DATA_FILES = [f for _d, _fam, fs, _g in SESSION_FAMILIES for f in fs]
TARGET_PCT = 10.0
BASELINE_ROW = "ttft-few-point-calibration"


def _rows(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ───────────────────────────────────────────────────────────── loaders
def sessions_from_rows(rows, grade, modes=SESSION_MODES):
    """{(model, arm): {seed: [turn dicts sorted by turn_idx]}} over completed sessions (attempt with an r2a_session
    record), latest attempt per seed in row order. Turn dict: wall, ok (None if ungraded), prompt, loaded, idx."""
    done = {r.get("attempt_id") for r in rows if r.get("record") == "r2a_session"}
    att_turns, att_key, order = {}, {}, []
    for r in rows:
        if r.get("record") != "r2a_turn" or r.get("mode") not in modes or r.get("attempt_id") not in done:
            continue
        a = r["attempt_id"]
        if a not in att_turns:
            att_turns[a] = []
            att_key[a] = (r["model_id"], r["arm_id"].replace("_call2_notools", ""), r["seed"])
            order.append(a)
        g = r.get(grade)
        att_turns[a].append({"wall": float(r["turn_wall_s"]), "ok": None if g is None else bool(g),
                             "prompt": r.get("final_call_prompt_tokens_calibrated"),
                             "loaded": r.get("loaded_context"), "idx": r["turn_idx"]})
    out = {}
    for a in order:  # later attempts overwrite earlier ones for the same seed
        model, arm, seed = att_key[a]
        out.setdefault((model, arm), {})[seed] = sorted(att_turns[a], key=lambda t: t["idx"])
    return out


def load_session_configs(repo, families=SESSION_FAMILIES):
    """List of configs {device, family, model, arm, sessions: {seed: turns}} merged across each family's files."""
    repo = Path(repo)
    configs = []
    for device, family, files, grade in families:
        merged = {}
        for rel in files:
            p = repo / rel
            if not p.exists():
                raise FileNotFoundError(p)
            for key, seeds in sessions_from_rows(_rows(p), grade).items():
                merged.setdefault(key, {}).update(seeds)
        for (model, arm), seeds in sorted(merged.items()):
            configs.append({"device": device, "family": family, "model": model, "arm": arm, "sessions": seeds})
    return configs


def load_single_configs(repo, files=SINGLE_FILES):
    """Single-shot outcome-table configs {device, family 'single', model, arm (config), items: [{wall, ok}]} in file
    order; latest row per item wins (position of its first appearance kept); rows without latency_s are skipped."""
    repo = Path(repo)
    configs = []
    for device, rel in files:
        cells = {}
        for r in _rows(repo / rel):
            if "item_id" not in r or r.get("score") is None or r.get("latency_s") is None:
                continue
            c = cells.setdefault((r.get("model_id"), r.get("config")), {})
            c[r["item_id"]] = {"wall": float(r["latency_s"]), "ok": float(r["score"])}
        for (model, cfg), items in sorted(cells.items(), key=str):
            configs.append({"device": device, "family": "single", "model": model, "arm": cfg,
                            "items": list(items.values())})
    return configs


def load_scale_probe(repo):
    """{(model, arm): [turns]} from the real 2-turn scale timing probe (mode probe)."""
    rows = _rows(Path(repo) / SCALE_PROBE_FILE)
    out = {}
    for key, seeds in sessions_from_rows(rows, "task_tool_ok", modes=("probe",)).items():
        seed = min(seeds)
        out[key] = seeds[seed]
    return out


def blade_task_turns(repo) -> int:
    p = Path(repo) / BLADE_TASK_FILE
    if not p.exists():
        return 0
    return sum(1 for r in _rows(p) if r.get("record") == "r2a_turn")


# ───────────────────────────────────────────────────────────── probe and targets (pure)
def probe_prefix(turns, budget_s=PROBE_BUDGET_S):
    """Longest prefix of turns whose cumulative wall clock is <= budget_s (empty if the first turn exceeds it)."""
    out, cum = [], 0.0
    for t in turns:
        if cum + t["wall"] > budget_s:
            break
        cum += t["wall"]
        out.append(t)
    return out


def projected_final_prompt(probe, n_turns):
    """Least-squares line of prompt tokens against turn index over the probe, evaluated at the last turn index
    (n_turns - 1 relative to the first probe turn's index); zero slope for one usable point."""
    pts = [(t["idx"], t["prompt"]) for t in probe if t.get("prompt") is not None]
    if not pts:
        return None
    if len(pts) == 1 or len({x for x, _ in pts}) == 1:
        return float(pts[-1][1])
    mx = statistics.mean(x for x, _ in pts)
    my = statistics.mean(y for _, y in pts)
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    slope = sum((x - mx) * (y - my) for x, y in pts) / sxx
    last = probe[0]["idx"] + n_turns - 1
    return my + slope * (last - mx)


def _acc(turns):
    g = [t["ok"] for t in turns if t["ok"] is not None]
    return (sum(float(x) for x in g) / len(g)) if g else None


def session_features(cfg, budget_s=PROBE_BUDGET_S):
    """Probe features and held-out targets of one session config. probe None when the first turn exceeds the
    budget; target None when no seed other than the probe seed exists."""
    seeds = sorted(cfg["sessions"])
    pseed = seeds[0]
    pturns = cfg["sessions"][pseed]
    probe = probe_prefix(pturns, budget_s)
    held = [t for s in seeds[1:] for t in cfg["sessions"][s]]
    out = {"probe_seed": pseed, "n_seeds": len(seeds), "n_probe_turns": len(probe), "first_turn_s": pturns[0]["wall"]}
    if probe:
        loaded = next((t["loaded"] for t in probe if t.get("loaded")), None)
        proj = projected_final_prompt(probe, len(pturns))
        out.update({"probe_s": statistics.mean(t["wall"] for t in probe), "probe_acc": _acc(probe),
                    "loaded": loaded, "proj_prompt": proj,
                    "stratum": "overflow" if (loaded and proj is not None and proj > loaded) else "fits"})
    else:
        out.update({"probe_s": None, "probe_acc": None, "stratum": None})
    if held:
        out.update({"actual_s": statistics.mean(t["wall"] for t in held), "actual_acc": _acc(held),
                    "n_target_turns": len(held)})
    else:
        out.update({"actual_s": None, "actual_acc": None, "n_target_turns": 0})
    return out


def single_features(cfg, budget_s=PROBE_BUDGET_S):
    items = cfg["items"]
    probe = probe_prefix(items, budget_s)
    held = items[len(probe):]
    out = {"n_probe_turns": len(probe), "first_turn_s": items[0]["wall"] if items else None, "stratum": "single"}
    out.update({"probe_s": statistics.mean(t["wall"] for t in probe) if probe else None,
                "probe_acc": _acc(probe) if probe else None})
    out.update({"actual_s": statistics.mean(t["wall"] for t in held) if held else None,
                "actual_acc": _acc(held) if held else None, "n_target_turns": len(held)})
    return out


# ───────────────────────────────────────────────────────────── predictor (pure)
def _ratio(a, p):
    if a is None or p is None or p <= 0:
        return None
    return a / p


def fit_ratios(train):
    """{stratum or '_all': (R_s, R_a)} medians of actual/probe over training feature dicts with both."""
    by = {}
    for f in train:
        if f.get("probe_s") is None or f.get("actual_s") is None:
            continue
        rs = _ratio(f["actual_s"], f["probe_s"])
        ra = _ratio(f.get("actual_acc"), f.get("probe_acc"))
        for k in (f["stratum"], "_all"):
            d = by.setdefault(k, {"s": [], "a": []})
            if rs is not None:
                d["s"].append(rs)
            if ra is not None:
                d["a"].append(ra)
    out = {}
    for k, d in by.items():
        out[k] = (statistics.median(d["s"]) if d["s"] else None, statistics.median(d["a"]) if d["a"] else None)
    return out


def predict(feat, ratios, raw=False):
    """(pred_s, pred_acc); None for either when there is no probe."""
    if feat.get("probe_s") is None:
        return None, None
    if raw:
        rs, ra = 1.0, 1.0
    else:
        rs, ra = ratios.get(feat["stratum"], (None, None))
        alls, alla = ratios.get("_all", (1.0, 1.0))
        rs = rs if rs is not None else (alls if alls is not None else 1.0)
        ra = ra if ra is not None else (alla if alla is not None else 1.0)
    ps = feat["probe_s"] * rs
    pa = None if feat.get("probe_acc") is None else min(1.0, feat["probe_acc"] * ra)
    return ps, pa


def ape(pred, actual):
    """Absolute percentage error; infinity when there is no prediction (a miss), None when actual is missing/0."""
    if actual is None or actual == 0:
        return None
    if pred is None:
        return math.inf
    return abs(pred - actual) / abs(actual) * 100.0


def evaluate_loo(feats, raw=False):
    """feats: list of feature dicts each with 'device'. Leave-one-device-out: for each device D, ratios fit on the
    other devices, predictions scored on D's configs that have a target. Returns per-config results and per-fold
    medians."""
    devices = sorted({f["device"] for f in feats})
    results = []
    for d in devices:
        train = [f for f in feats if f["device"] != d]
        ratios = fit_ratios(train)
        for f in feats:
            if f["device"] != d or f.get("actual_s") is None:
                continue
            ps, pa = predict(f, ratios, raw=raw)
            results.append({**{k: f[k] for k in ("device", "family", "model", "arm", "stratum")},
                            "pred_s": ps, "actual_s": f["actual_s"], "pred_acc": pa, "actual_acc": f["actual_acc"],
                            "err_s": ape(ps, f["actual_s"]), "err_acc": ape(pa, f["actual_acc"]),
                            "ratios": ratios.get(f["stratum"]) if f["stratum"] else None})
    return results


def _med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def summarize(results):
    out = {"n": len(results), "med_s": _med([r["err_s"] for r in results]),
           "med_acc": _med([r["err_acc"] for r in results]), "folds": {}}
    for d in sorted({r["device"] for r in results}):
        rr = [r for r in results if r["device"] == d]
        out["folds"][d] = {"n": len(rr), "med_s": _med([r["err_s"] for r in rr]),
                           "med_acc": _med([r["err_acc"] for r in rr]),
                           "misses": sum(1 for r in rr if r["err_s"] == math.inf)}
    return out


def _tag(cfg):
    return {k: cfg[k] for k in ("device", "family", "model", "arm")}


def session_feats(configs, budget_s=PROBE_BUDGET_S):
    return [{**_tag(c), **session_features(c, budget_s)} for c in configs]


def single_feats(configs, budget_s=PROBE_BUDGET_S):
    return [{**_tag(c), **single_features(c, budget_s)} for c in configs]


def scale_probe_check(repo, feats):
    """Real 2-turn probe vs the full scale run: predictions with ratios fit on evo-t2s configs only (the other
    device), scored against the matching evo-x2 scale configs' held-out targets."""
    probe = load_scale_probe(repo)
    ratios = fit_ratios([f for f in feats if f["device"] == "evo-t2s"])
    out = []
    for f in feats:
        if f["family"] != "scale" or f.get("actual_s") is None:
            continue
        turns = probe.get((f["model"], f["arm"]))
        if not turns:
            continue
        loaded = next((t["loaded"] for t in turns if t.get("loaded")), None)
        proj = projected_final_prompt(turns, 40)
        pf = {"probe_s": statistics.mean(t["wall"] for t in turns), "probe_acc": _acc(turns),
              "stratum": "overflow" if (loaded and proj is not None and proj > loaded) else "fits"}
        ps, pa = predict(pf, ratios)
        rs, _ = predict(pf, ratios, raw=True)
        out.append({"model": f["model"], "arm": f["arm"], "err_s": ape(ps, f["actual_s"]),
                    "err_s_raw": ape(rs, f["actual_s"]), "err_acc": ape(pa, f["actual_acc"])})
    return out


def own_profiler_loo(repo):
    """Everything row own-profiler-loo reports."""
    sess = session_feats(load_session_configs(repo))
    res = evaluate_loo(sess)
    raw = evaluate_loo(sess, raw=True)
    single = single_feats(load_single_configs(repo))
    return {"session": summarize(res), "session_raw": summarize(raw), "results": res,
            "single": summarize(evaluate_loo(single)), "scale_probe": scale_probe_check(repo, sess),
            "n_configs": len(sess), "n_by_device": {d: sum(1 for f in sess if f["device"] == d)
                                                     for d in sorted({f["device"] for f in sess})},
            "not_target": [f"{f['device']} {f['model']} {f['arm']}" for f in sess if f.get("actual_s") is None],
            "blade_turns": blade_task_turns(repo)}


def _p(x):
    if x is None:
        return "n/a"
    if x == math.inf:
        return "inf"
    return f"{x:.1f}%"


def own_profiler_text(d) -> str:
    s, r = d["session"], d["session_raw"]
    met_s = s["med_s"] is not None and s["med_s"] < TARGET_PCT
    met_a = s["med_acc"] is not None and s["med_acc"] < TARGET_PCT
    folds = "; ".join(f"test {k} (trained on the other device): n={v['n']}, seconds median APE {_p(v['med_s'])}, "
                      f"accuracy median APE {_p(v['med_acc'])}, no-probe misses {v['misses']}"
                      for k, v in s["folds"].items())
    sp = d["scale_probe"]
    spt = (f"real 2-turn scale probe (x2_scale_timing_probe) vs x2_r2_scale_real, ratios from evo-t2s: n={len(sp)}, "
           f"seconds median APE {_p(_med([x['err_s'] for x in sp]))} (raw probe {_p(_med([x['err_s_raw'] for x in sp]))}), "
           f"accuracy median APE {_p(_med([x['err_acc'] for x in sp]))}") if sp else "real scale probe: no matching pair"
    si = d["single"]
    return (f"probe = first turns of the lowest committed seed's session up to 600 s wall clock (simulated from "
            f"committed runs; no dedicated probe job was run); predictor = probe x median actual/probe ratio of the "
            f"other device per stratum (fits / overflow of the projected final prompt vs loaded window); session domain "
            f"{d['n_configs']} configs ({', '.join(f'{k} {v}' for k, v in d['n_by_device'].items())}), "
            f"{s['n']} scored leave-one-device-out: seconds per task median APE {_p(s['med_s'])}, accuracy median APE "
            f"{_p(s['med_acc'])} (raw probe without ratios: {_p(r['med_s'])}, {_p(r['med_acc'])}); {folds}; "
            f"pre-registered target < 10% on both: {'MET' if met_s and met_a else 'MISSED'} "
            f"(seconds {'met' if met_s else 'missed'}, accuracy {'met' if met_a else 'missed'}); baseline "
            f"{BASELINE_ROW} k=2 median 22.6%; secondary: {spt}; single-shot outcome tables (probe = items to 600 s): "
            f"n={si['n']}, seconds median APE {_p(si['med_s'])}, accuracy median APE {_p(si['med_acc'])}; not scored "
            f"(only the probe seed): {len(d['not_target'])}; Blade: NOT RUN (blade_r2_real_v1 has {d['blade_turns']} "
            f"task turns; a Blade test needs a probe and a reference task run on the Blade, never run)")
