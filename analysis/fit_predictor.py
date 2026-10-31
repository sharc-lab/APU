"""fit-predictor-v1: "will this task overflow the device window", predicted at session start (operator brief
2026-10-29; pre-registered in docs/EXPERIMENTS.md "PRE-REGISTRATION (2026-10-29): fit-predictor-v1").

Predictor: src/dse/fits_locally.SessionGrowthPredictor (k = 3, linear growth from the first 3 turns' prompt sizes,
against the measured ceiling where the run records one, else the loaded window). Baseline: ProjectionBaseline on the
task-definition projection the routers use today (impact_model.projected_task_prompts), where one exists.

Units, series, ground truth and exclusions are exactly as pre-registered; see load_units(). Offline: reads committed
result files only.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
K = 3
TARGET_PRECISION = 0.90
TARGET_RECALL = 0.90
LONGTASK_CALIB = 1.20            # x2_longtask.PROJ_CALIB (= x2_r2_evalset.PROJ_CALIB)
LLAMASERVER_OVERFLOW = ("flagged_reject", "silent_cut", "ctx_stop")

# Session files (r2a_turn rows), all with run_end. evo-t2s files start with t2s_; every other file is evo-x2.
SESSION_FILES = (
    "results/x2_evalset_tasks_real_v2.jsonl",
    "results/x2_own_compaction_v1_a.jsonl",
    "results/x2_own_compaction_v1_b.jsonl",
    "results/x2_own_config_probe_q8_15360.jsonl",
    "results/x2_own_config_probe_q4_28928.jsonl",
    "results/x2_own_profiler_probe_v1_ollama_default.jsonl",
    "results/x2_own_profiler_probe_v1_ollama_ctx_16384.jsonl",
    "results/x2_own_profiler_probe_v1_ollama_ctx_15360.jsonl",
    "results/x2_own_profiler_probe_v1_ollama_ctx_28928.jsonl",
    "results/x2_own_profiler_probe_v1_ollama_ctx_8192.jsonl",
    "results/x2_own_profiler_probe_v1_ollama_ctx_4096.jsonl",
    "results/x2_r2_real_v1.jsonl",
    "results/x2_r2_real_v1b.jsonl",
    "results/t2s_r2_real_cpu_v1.jsonl",
    "results/t2s_r2_real_igpu_v1.jsonl",
    "results/xruntime_llamacpp_qwen3_14b.jsonl",
)
LONGTASK_FILE = "results/x2_longtask_local_v1.jsonl"
DEMO_FILE = "results/demo_detector_check_v2.jsonl"
DATA_FILES = SESSION_FILES + (LONGTASK_FILE, DEMO_FILE)


def _paths(repo):
    for sub in ("", "analysis", "harness"):
        p = str(Path(repo) / sub) if sub else str(Path(repo))
        if p not in sys.path:
            sys.path.insert(0, p)


def _rows(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def device_of(rel: str) -> str:
    return "evo-t2s" if Path(rel).name.startswith("t2s_") else "evo-x2"


FAMILIES = (("x2_evalset_tasks_real_v2", "B3 v2"), ("x2_own_compaction", "own-compaction and client trim (client policy)"),
            ("x2_own_config_probe", "own-config probes"), ("x2_own_profiler_probe", "own-profiler probes"),
            ("r2_real", "R2 sessions"), ("longtask", "long tasks"), ("xruntime_llamacpp", "llama-server"),
            ("demo_detector", "demo detector check v2"))


def family(rel: str) -> str:
    return next((name for key, name in FAMILIES if key in rel), rel)


def ground_truth(r: dict):
    """(overflowed, kind) or (None, reason) for a turn row, in the pre-registered order."""
    if "context_event" in r:
        ev = r.get("context_event")
        if ev == "error_other":
            return None, "llama-server error_other (indeterminate)"
        if ev is not None:
            return ev in LLAMASERVER_OVERFLOW, "v2"
    if r.get("cut_v2_flagged") is not None:
        return bool(r["cut_v2_flagged"]), "v2"
    if r.get("over_loaded_window") is not None:
        return bool(r["over_loaded_window"]), "v1"
    return None, "no ground truth"


def turn_prompt(r: dict):
    v = r.get("sent_final_prompt_proj")
    return v if v is not None else r.get("final_call_prompt_tokens_calibrated")


def _baseline_proj(repo, cache: dict, r: dict):
    """Task-definition projection (impact_model.projected_task_prompts) for B3 v2-design task turns, else None."""
    if r.get("design") != "v2" or r.get("item_kind") != "task" or not r.get("task_id"):
        return None
    s = r["seed"]
    if s not in cache:
        import impact_model as im
        with contextlib.redirect_stdout(io.StringIO()):
            cache[s] = im.projected_task_prompts(repo, s)
    v = cache[s].get(r["task_id"])
    return v[1] if v else None


def session_units(repo, rel: str, rows: list, cache: dict, excluded: dict) -> list:
    # a session counts once its closing record is written (r2a_session; probe_session for the own-profiler probes,
    # which stop at their 600 s budget by design)
    done = {(r.get("arm_id"), r.get("seed"), r.get("attempt_id")) for r in rows
            if r.get("record") in ("r2a_session", "probe_session")}
    sess = {}
    for r in rows:
        if r.get("record") == "r2a_turn" and (r.get("arm_id"), r.get("seed"), r.get("attempt_id")) in done:
            sess.setdefault((r.get("arm_id"), r.get("seed"), r.get("attempt_id")), []).append(r)
    out = []
    for key, turns in sorted(sess.items(), key=lambda kv: str(kv[0])):
        turns.sort(key=lambda r: r["turn_idx"])
        first = [(t["turn_idx"], turn_prompt(t)) for t in turns[:K]]
        ceil_src = turns[0].get("server_n_ctx")
        window = turns[0].get("loaded_context")
        if len(first) < K or any(p is None for _i, p in first) or (window is None and ceil_src is None):
            excluded["session without k turns, prompt sizes or window"] = (
                excluded.get("session without k turns, prompt sizes or window", 0) + 1)
            continue
        prof = {"window": window, "measured_ceiling": ceil_src}
        for t in turns[K:]:
            if t.get("item_kind") not in ("task", None):
                continue
            gt, kind = ground_truth(t)
            if gt is None:
                excluded[kind] = excluded.get(kind, 0) + 1
                continue
            out.append({"file": rel, "device": device_of(rel), "arm": key[0], "seed": key[1],
                        "turn_idx": t["turn_idx"], "task_id": t.get("task_id"),
                        "ctx": {"first_turns": first, "target_turn": t["turn_idx"]}, "profile": prof,
                        "gt": gt, "gt_kind": kind, "baseline_proj": _baseline_proj(repo, cache, t)})
    return out


def longtask_units(rel: str, rows: list, excluded: dict) -> list:
    out = []
    for r in rows:
        if r.get("record") != "longtask_task":
            continue
        ps = [p for p in (r.get("prompt_tokens_est") or [])]
        if len(ps) <= K:
            excluded["long task with k calls or fewer"] = excluded.get("long task with k calls or fewer", 0) + 1
            continue
        if r.get("over_loaded_window") is None or r.get("loaded_context") is None:
            excluded["no ground truth"] = excluded.get("no ground truth", 0) + 1
            continue
        first = [(i, p * LONGTASK_CALIB) for i, p in enumerate(ps[:K])]
        out.append({"file": rel, "device": device_of(rel), "arm": r.get("arm_id"), "seed": r.get("seed"),
                    "turn_idx": len(ps) - 1, "task_id": r.get("task_id"),
                    "ctx": {"first_turns": first, "target_turn": len(ps) - 1},
                    "profile": {"window": r["loaded_context"]}, "gt": bool(r["over_loaded_window"]), "gt_kind": "v1",
                    "baseline_proj": None})
    return out


def demo_units(rel: str, rows: list, excluded: dict) -> list:
    out = []
    for r in rows:
        if r.get("record") != "detector_task":
            continue
        if r.get("num_ctx") is None:
            excluded["demo default arm (window not recorded)"] = (
                excluded.get("demo default arm (window not recorded)", 0) + 1)
            continue
        out.append({"file": rel, "device": "evo-x2", "arm": r.get("arm"), "seed": r.get("seed"),
                    "turn_idx": r.get("turn_idx"), "task_id": r.get("task_id"),
                    "ctx": {"projected_tokens": r["projected_prompt"]}, "profile": {"window": r["num_ctx"]},
                    "gt": bool(r["flagged_v2"]), "gt_kind": "v2", "baseline_proj": r["projected_prompt"]})
    return out


def load_units(repo=REPO) -> tuple:
    _paths(repo)
    cache, excluded, units = {}, {}, []
    for rel in SESSION_FILES:
        units += session_units(repo, rel, _rows(Path(repo) / rel), cache, excluded)
    units += longtask_units(LONGTASK_FILE, _rows(Path(repo) / LONGTASK_FILE), excluded)
    units += demo_units(DEMO_FILE, _rows(Path(repo) / DEMO_FILE), excluded)
    return units, excluded


def confusion(pairs) -> dict:
    """pairs: [(predicted overflow, did overflow)]; precision / recall with 95% Wilson intervals (None when empty)."""
    _paths(REPO)
    from numbers_register import wilson_ci
    tp = sum(1 for p, g in pairs if p and g)
    fp = sum(1 for p, g in pairs if p and not g)
    fn = sum(1 for p, g in pairs if not p and g)
    tn = sum(1 for p, g in pairs if not p and not g)
    out = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": tp + fp + fn + tn}
    for name, num, den in (("precision", tp, tp + fp), ("recall", tp, tp + fn)):
        out[name] = num / den if den else None
        out[name + "_ci"] = wilson_ci(num, den) if den else None
    return out


def score(units: list, predictor=None) -> dict:
    """Session-start predictor and baseline on the same units: pooled, by ground-truth kind, device, device x
    ceiling."""
    _paths(REPO)
    from src.dse.fits_locally import ProjectionBaseline, SessionGrowthPredictor
    pred = predictor or SessionGrowthPredictor(k=K)
    base = ProjectionBaseline(margin=0.0)
    for u in units:
        r = pred.predict(u["ctx"], u["profile"])
        u["pred"], u["proj"], u["ceiling"] = not r["fits"], r["projected_tokens"], r["ceiling_tokens"]
        u["base_pred"] = (None if u["baseline_proj"] is None else
                          not base.predict({"projected_tokens": u["baseline_proj"]}, {"window": u["ceiling"]})["fits"])

    def cell(us):
        b = [u for u in us if u["base_pred"] is not None]
        return {"pred": confusion([(u["pred"], u["gt"]) for u in us]),
                "base": confusion([(u["base_pred"], u["gt"]) for u in b]) if b else None,
                "pred_on_base_units": confusion([(u["pred"], u["gt"]) for u in b]) if b else None}

    groups = {"pooled": cell(units)}
    for kind in ("v2", "v1"):
        groups[f"ground truth {kind}"] = cell([u for u in units if u["gt_kind"] == kind])
    devs = sorted({u["device"] for u in units})
    by_dev = {d: cell([u for u in units if u["device"] == d]) for d in devs}
    by_dw = {(d, c): cell([u for u in units if u["device"] == d and u["ceiling"] == c])
             for d in devs for c in sorted({u["ceiling"] for u in units if u["device"] == d})}
    p = groups["pooled"]["pred"]
    met = (p["precision"] or 0) >= TARGET_PRECISION and (p["recall"] or 0) >= TARGET_RECALL
    fams = sorted({family(u["file"]) for u in units})
    by_fam = {f: cell([u for u in units if family(u["file"]) == f]) for f in fams}
    return {"groups": groups, "by_device": by_dev, "by_device_window": by_dw, "by_family": by_fam, "target_met": met,
            "n_units": len(units), "n_sessions": len({(u["file"], u["arm"], u["seed"]) for u in units}),
            "source": pred.source}


def fit_predictor_v1(repo=REPO) -> dict:
    units, excluded = load_units(repo)
    return {**score(units), "excluded": excluded}


def _pr(c) -> str:
    if c is None:
        return "no current estimate"

    def one(name):
        v = c[name]
        return "n/a" if v is None else f"{v:.3f} [{c[name + '_ci'][0]:.3f}, {c[name + '_ci'][1]:.3f}]"
    return (f"precision {one('precision')}, recall {one('recall')} (TP {c['tp']}, FP {c['fp']}, FN {c['fn']}, "
            f"TN {c['tn']})")


def _cell_text(c) -> str:
    s = f"predictor {_pr(c['pred'])}"
    if c["base"] is not None:
        s += (f"; baseline (current per-task estimate, n={c['base']['n']}) {_pr(c['base'])}; predictor on those "
              f"units {_pr(c['pred_on_base_units'])}")
    else:
        s += "; baseline: no current estimate"
    return s


def _pr_one(c, name) -> str:
    v = c[name]
    return "n/a" if v is None else f"{v:.3f} [{c[name + '_ci'][0]:.3f}, {c[name + '_ci'][1]:.3f}]"


def fit_predictor_v1_text(d: dict) -> str:
    """Row text. Operator decision 2026-10-30: the headline is the detector-v2 (runtime log) result, the verdict is
    MISSED, and the pooled target is labelled as proposed by the agent, not the operator; pooled and v1 numbers follow
    as secondary."""
    g = d["groups"]
    p = g["pooled"]["pred"]
    v2 = g["ground truth v2"]["pred"]
    parts = [f"HEADLINE (operator 2026-10-30): against the runtime log (ground truth detector v2 / llama-server "
             f"context event), the session-start predictor ({d['source']}: first {K} turns' prompts, linear growth to "
             f"the task's turn, vs measured ceiling else loaded window, margin 0) has precision {_pr_one(v2, 'precision')}"
             f", recall {_pr_one(v2, 'recall')} (TP {v2['tp']}, FP {v2['fp']}, FN {v2['fn']}, TN {v2['tn']}); "
             f"verdict MISSED (operator: treat as missed); 95% Wilson intervals (units are turns within sessions: too "
             f"narrow)",
             "detector v2 cell in full: " + _cell_text(g["ground truth v2"]),
             f"SECONDARY (not the headline): {d['n_units']} units in {d['n_sessions']} sessions",
             "pooled (secondary): " + _cell_text(g["pooled"]),
             "ground truth v1 (secondary; over_loaded_window, the same series the predictor extrapolates): "
             + _cell_text(g["ground truth v1"])]
    for dv, c in d["by_device"].items():
        parts.append(f"{dv} (secondary): " + _cell_text(c))
    for (dv, w), c in d["by_device_window"].items():
        parts.append(f"{dv} ceiling {w} (n={c['pred']['n']}): " + _cell_text(c))
    for f, c in d["by_family"].items():
        parts.append(f"run family {f} (n={c['pred']['n']}): " + _cell_text(c))
    ex = ", ".join(f"{k} {v}" for k, v in sorted(d["excluded"].items())) or "none"
    parts.append(f"excluded: {ex}")
    parts.append(f"pooled target precision >= {TARGET_PRECISION:.2f} and recall >= {TARGET_RECALL:.2f} (proposed by "
                 f"the agent, not the operator): pooled point estimates precision {p['precision']:.3f}, recall "
                 f"{p['recall']:.3f} would clear it, but the pooled cell is mostly v1 ground truth; verdict MISSED "
                 f"(operator 2026-10-30)")
    return " / ".join(parts)

if __name__ == "__main__":
    print(fit_predictor_v1_text(fit_predictor_v1()))
