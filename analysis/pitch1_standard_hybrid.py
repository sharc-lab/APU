"""pitch1: a standard hybrid vs the post-run-check pipeline on the same 90 jobs, same local window per comparison
(external reviewer request, operator approved; pre-registered in docs/EXPERIMENTS.md "PRE-REGISTRATION (2026-11-03):
pitch1, standard hybrid vs the post-run-check pipeline on the same 90 jobs").

Replay of recorded runs only: no call is made. Jobs: B3 v2, qwen3:14b, evo-x2 (results/x2_evalset_tasks_real_v2.jsonl),
seeds 20260901-03, 30 task turns each. Windows: default (ollama_default, loaded 40960) and 8192 (ollama_ctx_8192).
Cloud: claude-sonnet-5-5, every cloud row priced with the rolling cache breakpoint (analysis/rolling_cache_reprice.py).

Arms per window:
  all-cloud      each job's all_cloud_b3_v1 row.
  local-only     each job's local turn accepted as is.
  standard       Ollama local-first + LiteLLM-style cloud fallback + a prompt-length check against the window:
                 estimate > window before the task's first local call (primary; every call as a sensitivity) sends the
                 task to the cloud (its all-cloud row, as a pipeline route); otherwise local, with a cloud fallback only
                 on a hard error (HTTP error, call outcome not "ok", empty final response), taking the pipeline's
                 escalation source (own-rollback redo, else full-history redo, else IMPUTED). No correctness check.
                 Estimate: chars/4, the client's recorded pre-call prompt_tokens_est (primary), or the model's own
                 tokenizer count, prompt_tokens_est x the session's token_calib_ratio (sensitivity).
  pipeline       hybrid_full_pipeline.run_pipeline unchanged, L3 at CONFIG_TUNED (primary) or CONFIG_V1.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WINDOWS = {"default": "f16 default", "8192": "f16 8192"}   # window name -> hybrid_full_pipeline config
ESTIMATES = ("chars4", "tokenizer")
DATA_FILES = ["results/x2_evalset_tasks_real_v2.jsonl", "results/all_cloud_b3_v1.jsonl",
              "results/cloud_redo_full_context_v1.jsonl", "results/cloud_redo_full_context_v1_part2.jsonl",
              "results/cloud_redo_full_context_v1_part3.jsonl", "results/own_rollback_redo_v1.jsonl",
              "results/cloud_prices_anthropic_20261014.json"]


def _paths(repo):
    for sub in ("analysis", "src", "harness", "scripts", ""):
        p = str(Path(repo) / sub) if sub else str(Path(repo))
        if p not in sys.path:
            sys.path.insert(0, p)


# ── pure ───────────────────────────────────────────────────────────────────────────────────────────

def hard_error(row: dict) -> bool:
    """A failure a LiteLLM-style fallback sees: HTTP error, a call outcome other than ok (timeout, runtime refusal,
    error), or an empty final response. A wrong but well-formed answer is not one."""
    return (bool(row.get("any_http_error")) or any((c.get("outcome") or "ok") != "ok" for c in row.get("calls") or [])
            or bool(row.get("final_content_empty")))


def call_estimate(call: dict, row: dict, est: str) -> int | None:
    """The pre-call prompt estimate of one local call. chars4: the client's recorded prompt_tokens_est (len // 4 of each
    message's content, 4 per message, chars/4 of tool-call and tools JSON). tokenizer: that x the session's
    token_calib_ratio (server prompt_eval_count / chars/4 on the session's first call; 1.0 if not recorded)."""
    p = call.get("prompt_tokens_est")
    if p is None:
        return None
    if est == "chars4":
        return int(p)
    if est == "tokenizer":
        return round(int(p) * (row.get("token_calib_ratio") or 1.0))
    raise ValueError(est)


def over_window(row: dict, window: int, est: str = "chars4", every_call: bool = False) -> bool:
    calls = row.get("calls") or []
    calls = calls if every_call else calls[:1]
    return any((e := call_estimate(c, row, est)) is not None and e > window for c in calls)


def standard_hybrid(tasks, local, window, cloud, esc, est="chars4", every_call=False) -> dict:
    """Pure. local {(seed, task): turn row}; cloud {(seed, task): {ok, usd, seconds}} (all-cloud rows); esc
    {(seed, task): {ok, usd, seconds, src}} (the pipeline's escalation source)."""
    res = {"n": 0, "correct": 0, "usd": 0.0, "window_sent": 0, "fallback": 0, "kept_local": 0, "wrong_kept_local": 0,
           "imputed": 0, "per_task": {}, "seconds": {}}
    for k in tasks:
        row = local[k]
        res["n"] += 1
        if over_window(row, window, est, every_call):
            res["window_sent"] += 1
            res["correct"] += bool(cloud[k]["ok"])
            res["usd"] += cloud[k]["usd"]
            res["per_task"][k], res["seconds"][k] = "window", cloud[k]["seconds"]
        elif hard_error(row):
            e = esc[k]
            res["fallback"] += 1
            res["imputed"] += e["src"] == "imputed"
            res["correct"] += bool(e["ok"])
            res["usd"] += e["usd"]
            res["per_task"][k], res["seconds"][k] = "fallback", float(row["turn_wall_s"]) + e["seconds"]
        else:
            ok = row["end_state_ok"] is True
            res["kept_local"] += 1
            res["correct"] += ok
            res["wrong_kept_local"] += not ok
            res["per_task"][k], res["seconds"][k] = "local", float(row["turn_wall_s"])
    return _finish(res, res["window_sent"] + res["fallback"])


def _finish(res: dict, sent: int) -> dict:
    n = res["n"]
    res["sent"] = sent
    res["wrong"] = n - res["correct"]
    res["usd_job"] = res["usd"] / n if n else None
    res["median_s"] = statistics.median(res["seconds"].values()) if res["seconds"] else None
    return res


def pipeline_seconds(per_task: dict, local: dict, cloud: dict, esc: dict) -> dict:
    """Per job seconds of a run_pipeline result: routed = the all-cloud row; escalated = local wall + the escalation
    source row; local = local wall. The post-run check's compute time is not added."""
    out = {}
    for k, how in per_task.items():
        if how == "routed":
            out[k] = cloud[k]["seconds"]
        elif how == "escalated":
            out[k] = float(local[k]["turn_wall_s"]) + esc[k]["seconds"]
        else:
            out[k] = float(local[k]["turn_wall_s"])
    return out


def reference_arms(tasks, local, cloud) -> dict:
    allc = {"n": len(tasks), "correct": sum(bool(cloud[k]["ok"]) for k in tasks),
            "usd": sum(cloud[k]["usd"] for k in tasks), "kept_local": 0, "wrong_kept_local": 0, "imputed": 0,
            "seconds": {k: cloud[k]["seconds"] for k in tasks}}
    loc = {"n": len(tasks), "correct": sum(local[k]["end_state_ok"] is True for k in tasks), "usd": 0.0,
           "kept_local": len(tasks), "imputed": 0, "seconds": {k: float(local[k]["turn_wall_s"]) for k in tasks}}
    loc["wrong_kept_local"] = len(tasks) - loc["correct"]
    return {"all_cloud": _finish(allc, len(tasks)), "local_only": _finish(loc, 0)}


def targets(spill: dict, std: dict) -> dict:
    t2 = spill["correct"] > std["correct"]
    t3 = spill["usd_job"] <= std["usd_job"] + 1e-12
    return {"T1": t2 and t3, "T2": t2, "T3": t3}


# ── replay inputs ──────────────────────────────────────────────────────────────────────────────────

def _load_check(repo, config):
    p = Path(repo) / "analysis" / "post_run_check.py"
    spec = importlib.util.spec_from_file_location("post_run_check_pitch1", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.make_post_run_check(Path(repo), config)


def cloud_inputs(repo=REPO) -> dict:
    """All-cloud, full-history redo and own-rollback redo rows at their rolling USD with recorded seconds, and the
    imputed escalation USD (highest tier-mean full-history redo rolling USD, as rolling_cache_reprice does)."""
    _paths(repo)
    import hybrid_full_pipeline as hfp
    import impact_model as im
    import rolling_cache_reprice as rr
    ac = rr.all_cloud_rolling(repo)
    rd = rr.redo_rolling(repo)
    with contextlib.redirect_stdout(io.StringIO()):
        redo_raw, _m, _n, _p = im.redo_rows_and_model(repo)
    redo_s = {(r["tier"], r["seed"], r["task_id"]): float(r["seconds"]) for r in redo_raw}
    rb_s = {(r["tier"], r["seed"], r["task_id"]): float(r["seconds"])
            for r in hfp._rows(Path(repo) / hfp.ROLLBACK_FILE)
            if r.get("record") == "cloud_redo_task" and r.get("mode") == "executed"}
    cloud = {k: {"ok": v["ok"], "usd": v["usd_new"], "seconds": v["seconds"]}
             for k, v in ac[hfp.CLOUD_MODEL].items() if k[0] in hfp.SEEDS}
    redo = {k: {"ok": v["ok"], "usd": v["usd_new"]} for k, v in rd["redo"].items()}
    rb = {k: {"ok": v["ok"], "usd": v["usd_new"]} for k, v in rd["rb"].items()}
    tier_mean = {t: statistics.fmean(v["usd"] for (tr, _s, _t), v in redo.items() if tr == t)
                 for t in sorted({tr for tr, _s, _t in redo})}
    return {"cloud": cloud, "redo": redo, "rb": rb, "redo_s": redo_s, "rb_s": rb_s,
            "imputed_usd": max(tier_mean.values())}


def escalations_with_seconds(repo, tier, ci) -> dict:
    import hybrid_full_pipeline as hfp
    esc = hfp.escalation_outcomes(repo, tier, ci["cloud"], ci["redo"], ci["rb"], ci["imputed_usd"])
    for k, e in esc.items():
        if e["src"] == "rollback_redo":
            e["seconds"] = ci["rb_s"][(tier, *k)]
        elif e["src"] == "full_history_redo":
            e["seconds"] = ci["redo_s"][(tier, *k)]
        else:
            e["seconds"] = ci["cloud"][k]["seconds"]
    return esc


def compare(repo=REPO, ci=None) -> dict:
    """Every arm at both windows, plus the sensitivity cells (tokenizer estimate, every-call check, CONFIG_V1)."""
    _paths(repo)
    import hybrid_full_pipeline as hfp
    from src.dse import l3_invariants as L3
    ci = ci or cloud_inputs(repo)
    cloud = ci["cloud"]
    checks = {"tuned": _load_check(repo, L3.CONFIG_TUNED), "v1": _load_check(repo, None)}
    out = {"imputed_usd": ci["imputed_usd"], "windows": {}}
    for wname, cfg in WINDOWS.items():
        _rel, _arm, tier = hfp.CONFIGS.get(cfg) or hfp.COMPARE_CONFIGS[cfg]
        local, first, ceiling = hfp.config_sessions(repo, cfg)
        tasks = sorted(k for k in local if k in cloud)
        esc = escalations_with_seconds(repo, tier, ci)
        w = {"config": cfg, "tier": tier, "window": ceiling, "n": len(tasks), **reference_arms(tasks, local, cloud)}
        allc = w["all_cloud"]["usd_job"]
        for est in ESTIMATES:
            for every in (False, True):
                r = standard_hybrid(tasks, local, ceiling, cloud, esc, est, every)
                r["pct_bill"] = 100 * r["usd_job"] / allc
                w[f"standard_{est}_{'every' if every else 'first'}"] = r
        for name, chk in checks.items():
            r = hfp.run_pipeline(tasks, local, first, ceiling, cloud, esc, chk, hfp.load_task_budget(repo))
            r["imputed"] = r["imputed_escalations"]
            r["seconds"] = pipeline_seconds(r["per_task"], local, cloud, esc)
            _finish(r, r["routed"] + r["escalated"])
            r["pct_bill"] = 100 * r["usd_job"] / allc
            w[f"pipeline_{name}"] = r
        for arm in ("all_cloud", "local_only"):
            w[arm]["pct_bill"] = 100 * w[arm]["usd_job"] / allc
        w["targets"] = targets(w["pipeline_tuned"], w["standard_chars4_first"])
        w["sensitivity"] = {
            "tokenizer estimate": targets(w["pipeline_tuned"], w["standard_tokenizer_first"]),
            "every-call check": targets(w["pipeline_tuned"], w["standard_chars4_every"]),
            "tokenizer + every call": targets(w["pipeline_tuned"], w["standard_tokenizer_every"]),
            "L3 CONFIG_V1": targets(w["pipeline_v1"], w["standard_chars4_first"])}
        out["windows"][wname] = w
    return out


# ── text ───────────────────────────────────────────────────────────────────────────────────────────

ARM_LABELS = (("all_cloud", "all-cloud"), ("local_only", "local-only"),
              ("standard_chars4_first", "standard hybrid (chars/4, first-call check)"),
              ("pipeline_tuned", "post-run-check pipeline (L3 tuned)"))
SENS_LABELS = (("standard_tokenizer_first", "standard hybrid (tokenizer count, first-call check)"),
               ("standard_chars4_every", "standard hybrid (chars/4, every-call check)"),
               ("standard_tokenizer_every", "standard hybrid (tokenizer count, every-call check)"),
               ("pipeline_v1", "post-run-check pipeline (L3 v1)"))


def _split(r) -> str:
    if "window_sent" in r:
        return f" (window check {r['window_sent']}, hard-error fallback {r['fallback']})"
    if "routed" in r:
        return (f" (routed at start {r['routed']}, escalated by the check {r['escalated']}, budget refusals "
                f"{r['budget_refused_route'] + r['budget_refused_escalation']})")
    return ""


def _cell(label, r) -> str:
    return (f"{label}: correct {r['correct']}/{r['n']}, wrong local accepted {r['wrong_kept_local']}, wrong total "
            f"{r['wrong']}, cloud USD {r['usd_job']:.5f}/job ({r['pct_bill']:.1f}% of the all-cloud bill), sent to "
            f"cloud {r['sent']}{_split(r)}, imputed {r['imputed']}, median {r['median_s']:.1f} s/job")


def extra_usd_per_extra_correct(spill: dict, std: dict) -> float | None:
    """Descriptive, not a target: the pipeline's extra cloud USD over the standard hybrid per extra correct job."""
    dc = spill["correct"] - std["correct"]
    return (spill["usd"] - std["usd"]) / dc if dc > 0 else None


def _extra(sp, std) -> str:
    x = extra_usd_per_extra_correct(sp, std)
    if x is None:
        return ""
    return (f" / descriptive, not pre-registered: the pipeline spends USD {sp['usd'] - std['usd']:.4f} more cloud over "
            f"the {sp['n']} jobs for {sp['correct'] - std['correct']} more correct, USD {x:.4f} per extra correct job")


def _mm(b: bool) -> str:
    return "MET" if b else "MISSED"


def window_text(d: dict, wname: str) -> str:
    w = d["windows"][wname]
    t = w["targets"]
    std, sp = w["standard_chars4_first"], w["pipeline_tuned"]
    sens = "; ".join(f"{k}: T1 {_mm(v['T1'])}" for k, v in w["sensitivity"].items())
    return (f"replay of recorded runs, no call made; B3 v2, qwen3:14b, evo-x2 {w['tier']} (loaded window "
            f"{w['window']}, the same window for every arm), seeds 20260901-03, claude-sonnet-5-5 at the rolling cache "
            f"breakpoint (billed basis); standard hybrid = Ollama local-first, prompt-length check (estimate > window "
            f"sends the task to the cloud, its all-cloud row), cloud fallback only on a hard error (HTTP error, call "
            f"outcome not ok, empty response; escalation source), no correctness check; post-run-check pipeline = "
            f"hybrid_full_pipeline.run_pipeline unchanged (route at start, post-run check L1 to L3, rollback, cloud "
            f"redo, USD 0.25 per-task budget), L3 CONFIG_TUNED (tuned on seeds 20260901-04, in-sample here); imputed "
            f"escalation USD {d['imputed_usd']:.5f} (highest tier-mean full-history redo rolling USD, outcome = the "
            f"all-cloud row); latency = recorded local turn_wall_s plus the cloud source row's seconds (check compute "
            f"not added) / " + " / ".join(_cell(lbl, w[k]) for k, lbl in ARM_LABELS)
            + " / sensitivity: " + " / ".join(_cell(lbl, w[k]) for k, lbl in SENS_LABELS)
            + f" / pre-registered T1-{wname} (pipeline correct above the standard hybrid at equal or lower USD/job): "
            f"{_mm(t['T1'])} ({sp['correct']} vs {std['correct']} correct, USD {sp['usd_job']:.5f} vs "
            f"{std['usd_job']:.5f}/job); T2 accuracy {_mm(t['T2'])}; T3 cost {_mm(t['T3'])}; T1 under sensitivity: "
            f"{sens}" + _extra(sp, std))


def main():
    d = compare(REPO)
    for wname in WINDOWS:
        print(f"== {wname}\n{window_text(d, wname)}\n")


if __name__ == "__main__":
    main()
