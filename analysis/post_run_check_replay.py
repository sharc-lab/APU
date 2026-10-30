"""Post-run check: replay on recorded runs (rows post-run-check-4096, -8192, -default, post-run-check-layer-attribution)
and the live estimate (row post-run-check-live-estimate). Pre-registered in docs/EXPERIMENTS.md "PRE-REGISTRATION
(2026-10-29): post-run check and per-task cloud budget". Offline, no network.

`py analysis/post_run_check_replay.py` writes results/post_run_check_replay_v1.jsonl (run_start, one prc_task row per task
turn, one prc_cell row per tier, prc_attribution rows, run_end). The register rows read only that file.

What the check sees per task turn (src/dse/post_run_check.check_record): the recorded calls (errors, prompt counts,
tool calls), the final text, the over_loaded_window flag, the session's system prompt and the task's user text (rebuilt
from the seed by harness/x2_r2_evalset.build_session; the rebuilt system prompt's sha256 is asserted equal to the one
every recorded call carries). The grader's end_state_ok is read only AFTER the check, to score it.

Escalated tasks are graded as in hybrid-local-first-rollback (analysis/own_rollback.py::local_first_policy): the
own-rollback redo row, else the full-history redo row of the same tier, seed and task, else (a task right locally,
no redo row) IMPUTED from the Sonnet all-cloud run (same seed and task; seeds without an all-cloud row: that task's
all-cloud pass fraction over the shared seeds) at the tier's mean redo USD. Per-task cap: USD 0.25 against the task's
all-input-uncached cost; a task over it is budget_exhausted and counted wrong.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = "results/post_run_check_replay_v1.jsonl"
SOURCE = "results/x2_evalset_tasks_real_v2.jsonl"
OWN_COMPACTION = ("results/x2_own_compaction_v1_a.jsonl", "results/x2_own_compaction_v1_b.jsonl")
MECH = "results/x2_r2_mechanism.jsonl"
ROLLBACK_FILE = "results/own_rollback_redo_v1.jsonl"
TIERS = ("ollama_ctx_4096", "ollama_ctx_8192", "ollama_default")
ROW_OF = {"ollama_ctx_4096": "post-run-check-4096", "ollama_ctx_8192": "post-run-check-8192",
          "ollama_default": "post-run-check-default"}
MODEL = "qwen3:14b"
CLOUD_MODEL = "claude-sonnet-5-5"
TASK_CAP_USD = 0.25
TARGET_RECALL, TARGET_FALSE_ESC, TARGET_ACC = 0.80, 0.10, 0.97
LAYER_ORDER = ("L1_errors", "L1_proxy", "L2_proxy", "L3")
LAYER_LABEL = {"L1_errors": "L1 recorded call errors (real part of L1)",
               "L1_proxy": "L1 PROXY (recorded over_loaded_window; no server log kept)",
               "L2_proxy": "L2 PROXY (server prompt_eval_count below the chars/4 estimate sent; no render kept)",
               "L3": "L3 task-rule invariants (real)"}
L4_TOKENS_OUT, L4_TOKENS_IN = 16, 64   # L4 design: num_predict 16; the check message is about 64 tokens
GIN_WAIT_S = 3.0         # harness/x2_r2_mechanism.GIN_WAIT_S: the log-slice wait per call (a code constant)


def _load(rel: str, name: str, repo: Path = REPO):
    spec = importlib.util.spec_from_file_location(name, Path(repo) / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, m)
    spec.loader.exec_module(m)
    return sys.modules[name]


def _paths(repo: Path):
    for p in (str(repo), str(repo / "harness"), str(repo / "analysis")):
        if p not in sys.path:
            sys.path.insert(0, p)


def _rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _git_blob(path: Path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _q(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    return statistics.median(xs) if p == 50 else statistics.quantiles(xs, n=10, method="inclusive")[8]


# ── inputs ─────────────────────────────────────────────────────────────────────────────────────

def sessions(repo: Path, seeds) -> dict:
    """{seed: (system_prompt, {turn_idx: user_text}, tools_schema)} rebuilt from the seed (task spec only)."""
    _paths(repo)
    import x2_r2_evalset as ev
    out = {}
    for s in seeds:
        with contextlib.redirect_stdout(io.StringIO()):
            es = ev.build_session(s, workload="evalset", guardrails=False, design="v2")
        out[s] = (es.system_prompt, {t.idx: t.user_text for t in es.turns}, ev.tools_payload("v2"))
    return out


def local_turns(repo: Path, path: str, tiers=TIERS, arm_suffix="_call2_notools") -> dict:
    _paths(repo)
    import x2_r2_evalset as ev
    rows = ev.ag.read_rows(Path(repo) / path)
    att = {s["attempt_id"] for s in ev.ag.completed_sessions(rows)
           if s.get("mode") == "real" and s["model_id"] == MODEL}
    out = {}
    for r in rows:
        if (r.get("record") == "r2a_turn" and r.get("attempt_id") in att and r.get("item_kind") == "task"
                and r.get("end_state_ok") is not None):
            tier = r["arm_id"].replace(arm_suffix, "")
            if tier in tiers:
                out.setdefault(tier, []).append(r)
    return out


def cloud_sources(repo: Path) -> dict:
    _paths(repo)
    import impact_model as im
    merged, model, _n, _p = im.redo_rows_and_model(repo)
    redo = {(r["tier"], r["seed"], r["task_id"]): {"ok": bool(r.get("cloud_end_state_ok")), "usd": float(r["usd"]),
                                                   "upper": float(r["usd_all_uncached"]), "seconds": r.get("seconds")}
            for r in merged if r.get("record") == "cloud_redo_task"}
    rb = {}
    for r in _rows(Path(repo) / ROLLBACK_FILE):
        if r.get("record") == "cloud_redo_task" and r.get("mode") == "executed":
            rb[(r["tier"], r["seed"], r["task_id"])] = {"ok": bool(r.get("cloud_end_state_ok")), "usd": float(r["usd"]),
                                                        "upper": float(r["usd_all_uncached"]),
                                                        "seconds": r.get("seconds")}
    ac = im.replay_cloud_tasks(repo)
    return {"redo": redo, "redo_model": model, "rb": rb, "ac": ac["models"][CLOUD_MODEL], "ac_seeds": ac["seeds"]}


# ── grading of one decision ─────────────────────────────────────────────────────────────────────

def grade(esc: bool, local_ok: bool, key: tuple, src: dict, tier_mean: dict, cap: float = TASK_CAP_USD) -> dict:
    """Pure. key (tier, seed, task_id). Returns {ok (0..1), usd, upper, src, budget_exhausted}."""
    if not esc:
        return {"ok": 1.0 if local_ok else 0.0, "usd": 0.0, "upper": 0.0, "src": "local", "budget_exhausted": False}
    tier, seed, task = key
    if key in src["rb"]:
        c, s = src["rb"][key], "rollback_redo"
    elif key in src["redo"]:
        c, s = src["redo"][key], "full_history_redo"
    else:
        if not local_ok:
            raise ValueError(f"wrong local task {key} has no redo row")
        ac = src["ac"]
        if (seed, task) in ac:
            ok = 1.0 if ac[(seed, task)]["ok"] else 0.0
            s = "imputed_same_seed"
        else:
            vals = [v["ok"] for (sd, t), v in ac.items() if t == task]
            ok = sum(vals) / len(vals)
            s = "imputed_task_fraction"
        c = {"ok": ok, "usd": tier_mean["usd"], "upper": tier_mean["upper"]}
    if c["upper"] > cap:
        return {"ok": 0.0, "usd": min(c["usd"], cap), "upper": cap, "src": s, "budget_exhausted": True}
    return {"ok": float(c["ok"]), "usd": c["usd"], "upper": c["upper"], "src": s, "budget_exhausted": False}


def policy(tasks: list[dict], layers: tuple, src: dict, tier_mean: dict) -> dict:
    """Pure over prc_task rows: escalate iff any of `layers` fired."""
    n = len(tasks)
    wrong = [t for t in tasks if not t["local_ok"]]
    right = [t for t in tasks if t["local_ok"]]
    esc = lambda t: any(l in t["fired"] for l in layers)  # noqa: E731
    g = [grade(esc(t), t["local_ok"], (t["tier"], t["seed"], t["task_id"]), src, tier_mean) for t in tasks]
    tp, fp = sum(esc(t) for t in wrong), sum(esc(t) for t in right)
    srcs = {}
    for x in g:
        srcs[x["src"]] = srcs.get(x["src"], 0) + 1
    correct = sum(x["ok"] for x in g)
    return {"layers": list(layers), "n": n, "wrong": len(wrong), "right": len(right), "caught": tp,
            "false_esc": fp, "recall": tp / len(wrong) if wrong else None,
            "false_esc_rate": fp / len(right) if right else None, "kept_local": n - tp - fp,
            "correct": correct, "accuracy": correct / n, "usd_task": sum(x["usd"] for x in g) / n,
            "upper_task": sum(x["upper"] for x in g) / n, "sources": srcs,
            "budget_exhausted": sum(x["budget_exhausted"] for x in g),
            "imputed": srcs.get("imputed_same_seed", 0) + srcs.get("imputed_task_fraction", 0),
            "fractional_imputed": srcs.get("imputed_task_fraction", 0)}


def targets(p: dict) -> dict:
    return {"recall": p["recall"] is not None and p["recall"] >= TARGET_RECALL,
            "false_esc": p["false_esc_rate"] is not None and p["false_esc_rate"] <= TARGET_FALSE_ESC,
            "accuracy": p["accuracy"] >= TARGET_ACC}


# ── replay ─────────────────────────────────────────────────────────────────────────────────────

def replay(repo: Path = REPO, out: str = OUT) -> list[dict]:
    repo = Path(repo)
    _paths(repo)
    from src.dse import post_run_check as prc
    turns = local_turns(repo, SOURCE)
    seeds = sorted({r["seed"] for v in turns.values() for r in v})
    sess = sessions(repo, seeds)
    src = cloud_sources(repo)
    rows = [{"record": "run_start", "ts_utc": datetime.now(timezone.utc).isoformat(), "harness": "post_run_check_replay",
             "source": SOURCE, "source_git_blob": _git_blob(repo / SOURCE),
             "rollback_file": ROLLBACK_FILE, "redo_model": src["redo_model"], "cloud_model": CLOUD_MODEL,
             "all_cloud_seeds": src["ac_seeds"], "seeds": seeds, "tiers": list(TIERS), "task_cap_usd": TASK_CAP_USD,
             "targets": {"recall_min": TARGET_RECALL, "false_esc_max": TARGET_FALSE_ESC, "accuracy_min": TARGET_ACC},
             "layers": LAYER_LABEL, "check_module_git_blob": _git_blob(repo / "src/dse/post_run_check.py"),
             "l3_module_git_blob": _git_blob(repo / "src/dse/l3_invariants.py")}]
    sha_mismatch = 0
    tasks_by_tier = {}
    for tier in TIERS:
        for r in sorted(turns.get(tier, []), key=lambda x: (x["seed"], x["turn_idx"])):
            sp, users, schema = sess[r["seed"]]
            if any(c.get("system_prompt_sha256") not in (None, _sha(sp)) for c in r.get("calls") or []):
                sha_mismatch += 1
            d = prc.check_record(r, sp, users[r["turn_idx"]], schema)
            row = {"record": "prc_task", "tier": tier, "seed": r["seed"], "turn_idx": r["turn_idx"],
                   "task_id": r["task_id"], "fired": d["fired"], "decision": d["decision"],
                   "l3_failed": d["layers"]["L3"]["failed"], "l3_kind": d["layers"]["L3"]["kind"],
                   "check_seconds": d["check_seconds"], "n_calls": len(r.get("calls") or []),
                   "turn_wall_s": r.get("turn_wall_s"),
                   "local_ok": r["end_state_ok"] is True}   # grader outcome, read after the check, for scoring only
            rows.append(row)
            tasks_by_tier.setdefault(tier, []).append(row)
    if sha_mismatch:
        raise RuntimeError(f"{sha_mismatch} task turns: rebuilt system prompt differs from the recorded sha256")
    for tier in TIERS:
        ts = tasks_by_tier[tier]
        tr = [v for (t, _s, _k), v in src["redo"].items() if t == tier]
        tier_mean = {"usd": statistics.fmean(v["usd"] for v in tr), "upper": statistics.fmean(v["upper"] for v in tr)}
        head = policy(ts, LAYER_ORDER, src, tier_mean)
        per_layer = {l: policy(ts, (l,), src, tier_mean) for l in LAYER_ORDER}
        unique = {l: sum(1 for t in ts if not t["local_ok"] and t["fired"] == [l]) for l in LAYER_ORDER}
        unique_fp = {l: sum(1 for t in ts if t["local_ok"] and t["fired"] == [l]) for l in LAYER_ORDER}
        l3_checks = {}
        for t in ts:
            for f in t["l3_failed"]:
                k = l3_checks.setdefault(f, [0, 0])
                k[0 if not t["local_ok"] else 1] += 1
        cs = [t["check_seconds"] for t in ts]
        rows.append({"record": "prc_cell", "tier": tier, "headline": head, "targets_met": targets(head),
                     "tier_mean_redo": tier_mean, "check_seconds_median": _q(cs, 50), "check_seconds_p90": _q(cs, 90),
                     "local_turn_wall_s_median": _q([t["turn_wall_s"] for t in ts], 50)})
        rows.append({"record": "prc_attribution", "tier": tier, "per_layer": per_layer, "unique_catches": unique,
                     "unique_false_esc": unique_fp, "l3_checks_wrong_right": l3_checks,
                     "l3_alone_targets_met": targets(per_layer["L3"])})
    rows.append({"record": "prc_own_compaction", **own_compaction_attribution(repo)})
    rows.append({"record": "run_end", "ts_utc": datetime.now(timezone.utc).isoformat(), "n_tasks":
                 sum(len(v) for v in tasks_by_tier.values()), "system_prompt_sha_mismatch": sha_mismatch})
    p = repo / out
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    return rows


def own_compaction_attribution(repo: Path) -> dict:
    """Detection only on the own_compaction / client_trim arms: L1 real (recorded detector v2) vs L3."""
    _paths(repo)
    from src.dse import post_run_check as prc
    out = {}
    for f in OWN_COMPACTION:
        rows = _rows(Path(repo) / f)
        tasks = [r for r in rows if r.get("record") == "r2a_turn" and r.get("item_kind") == "task"
                 and r.get("mode") == "real" and r.get("end_state_ok") is not None]
        sess = sessions(repo, sorted({r["seed"] for r in tasks}))
        for r in tasks:
            sp, users, schema = sess[r["seed"]]
            arm = r["arm_id"].replace("_call2_notools", "")
            c = out.setdefault(arm, {"n": 0, "wrong": 0, "l1_wrong": 0, "l1_right": 0, "l3_wrong": 0, "l3_right": 0,
                                     "any_wrong": 0, "any_right": 0, "no_log_turns": 0, "sha_mismatch": 0,
                                     "file": f})
            if any(x.get("system_prompt_sha256") not in (None, _sha(sp)) for x in r.get("calls") or []):
                c["sha_mismatch"] += 1
            wrong = r["end_state_ok"] is not True
            l1 = bool(r.get("cut_v2_flagged"))
            l3 = prc.l3_record(r, sp, users[r["turn_idx"]], schema)["fail"]
            c["n"] += 1
            c["wrong"] += wrong
            c["no_log_turns"] += bool(r.get("cut_v2_no_log_calls"))
            k = "wrong" if wrong else "right"
            c["l1_" + k] += l1
            c["l3_" + k] += l3
            c["any_" + k] += l1 or l3
    return {"arms": out}


# ── register text ──────────────────────────────────────────────────────────────────────────────

def load(repo: Path = REPO, path: str = OUT) -> dict:
    rows = _rows(Path(repo) / path)
    return {"start": next(r for r in rows if r["record"] == "run_start"),
            "end": next((r for r in rows if r["record"] == "run_end"), None),
            "cells": {r["tier"]: r for r in rows if r["record"] == "prc_cell"},
            "attr": {r["tier"]: r for r in rows if r["record"] == "prc_attribution"},
            "oc": next((r for r in rows if r["record"] == "prc_own_compaction"), None),
            "tasks": [r for r in rows if r["record"] == "prc_task"]}


def _pc(k, n):
    return f"{k:g}/{n} ({100 * k / n:.1f}%)" if n else f"{k}/0"


def _mm(ok):
    return "MET" if ok else "MISSED"


def policy_text(p: dict) -> str:
    return (f"recall {_pc(p['caught'], p['wrong'])} of known-wrong local answers, false escalations "
            f"{_pc(p['false_esc'], p['right'])} of correct ones, kept local {_pc(p['kept_local'], p['n'])}, final "
            f"accuracy {p['correct']:.2f}/{p['n']} ({100 * p['accuracy']:.1f}%), cloud USD {p['usd_task']:.5f}/task "
            f"billed (cache-priced), {p['upper_task']:.5f}/task upper bound (all input uncached); escalated graded "
            f"from the own-rollback redo {p['sources'].get('rollback_redo', 0)}, the full-history redo "
            f"{p['sources'].get('full_history_redo', 0)}, IMPUTED {p['imputed']} (right locally, no redo row: Sonnet "
            f"all-cloud outcome, {p['fractional_imputed']} of them the task's all-cloud pass fraction over seeds "
            f"20260901-03; tier mean redo USD); budget_exhausted at the USD {TASK_CAP_USD:.2f} per-task cap "
            f"{p['budget_exhausted']}")


def cell_text(d: dict, tier: str) -> str:
    c = d["cells"][tier]
    h, m = c["headline"], c["targets_met"]
    return (f"measured by replay of recorded runs (B3 v2 real run, {MODEL}, evo-x2 {tier}, seeds "
            f"{', '.join(map(str, d['start']['seeds']))}, {h['n']} task turns); check = any FAIL of "
            f"{'; '.join(LAYER_LABEL[l] for l in LAYER_ORDER)}; L1 server-log and L2 render layers NOT computable "
            f"from this run's files (no log, no render kept), so L1 and L2 enter as PROXIES; {policy_text(h)}; "
            f"targets (pre-registered): recall >= 80% {_mm(m['recall'])}, false escalations <= 10% "
            f"{_mm(m['false_esc'])}, final accuracy >= 97% {_mm(m['accuracy'])}; added latency (the check's own "
            f"compute time per task, replay) median {1000 * c['check_seconds_median']:.2f} ms, p90 "
            f"{1000 * c['check_seconds_p90']:.2f} ms (live L1 log slice and L2 render not included: row "
            f"post-run-check-live-estimate); replay assumes a task's local outcome does not depend on earlier "
            f"escalations")


def attribution_text(d: dict) -> str:
    parts = []
    for tier in TIERS:
        a = d["attr"][tier]
        lay = "; ".join(f"{LAYER_LABEL[l]} alone: caught {a['per_layer'][l]['caught']}/{a['per_layer'][l]['wrong']}, "
                        f"false escalations {a['per_layer'][l]['false_esc']}/{a['per_layer'][l]['right']}, only layer "
                        f"firing on {a['unique_catches'][l]} caught and {a['unique_false_esc'][l]} false"
                        for l in LAYER_ORDER)
        l3 = a["per_layer"]["L3"]
        m = a["l3_alone_targets_met"]
        chk = ", ".join(f"{k} {v[0]} wrong / {v[1]} right" for k, v in sorted(a["l3_checks_wrong_right"].items()))
        parts.append(f"{tier}: {lay}; L3 alone (the only real layer here): {policy_text(l3)}; targets recall "
                     f"{_mm(m['recall'])}, false escalations {_mm(m['false_esc'])}, accuracy {_mm(m['accuracy'])}; "
                     f"L3 checks that fired (on wrong / right tasks): {chk or 'none'}")
    oc = []
    for arm, c in sorted((d["oc"] or {}).get("arms", {}).items()):
        right = c["n"] - c["wrong"]
        oc.append(f"{arm} ({c['n']} tasks, {c['wrong']} wrong): L1 real (recorded detector v2) caught "
                  f"{c['l1_wrong']}/{c['wrong']}, false {c['l1_right']}/{right}; L3 caught {c['l3_wrong']}/{c['wrong']}, "
                  f"false {c['l3_right']}/{right}; L1 or L3 caught {c['any_wrong']}/{c['wrong']}, false "
                  f"{c['any_right']}/{right}; turns with a call lacking its log slice {c['no_log_turns']}")
    return ("measured by replay of recorded runs; which layer caught what, B3 v2 qwen3:14b / " + " / ".join(parts)
            + " / own-compaction runs (managed context, detection only, no cloud redo exists): " + "; ".join(oc))


def live_estimate(repo: Path = REPO, path: str = OUT) -> dict:
    """ESTIMATE for the live job: cloud spend from the replay's measured per-task costs, hours from the B3 v2 turn times
    plus the check's live overhead (log-slice wait per call, median prompt-check duration per task) and the redo
    seconds of escalated tasks; L4 cost from the recorded per-call decode and prompt-eval rates."""
    d = load(repo, path)
    _paths(repo)
    n_tasks = sum(c["headline"]["n"] for c in d["cells"].values())
    billed = sum(c["headline"]["usd_task"] * c["headline"]["n"] for c in d["cells"].values())
    upper = sum(c["headline"]["upper_task"] * c["headline"]["n"] for c in d["cells"].values())
    esc = sum(c["headline"]["caught"] + c["headline"]["false_esc"] for c in d["cells"].values())
    local_s = sum(t["turn_wall_s"] for t in d["tasks"])
    n_calls = sum(t["n_calls"] for t in d["tasks"])
    mech = [r["check_duration_s"] for r in _rows(Path(repo) / MECH) if r.get("record") == "r2m_prompt_check"
            and r.get("check_duration_s") is not None]
    chk = statistics.median(mech)
    import impact_model as im
    merged, *_ = im.redo_rows_and_model(repo)
    redo_s = statistics.fmean(float(r["seconds"]) for r in merged if r.get("seconds") is not None)
    hours = (local_s + n_calls * GIN_WAIT_S + n_tasks * chk + esc * redo_s) / 3600
    src_rows = _rows(Path(repo) / SOURCE)
    l4 = {}
    for tier in TIERS:
        dec, pe, per = [], [], []
        for r in src_rows:
            if r.get("record") == "r2a_turn" and r.get("arm_id", "").replace("_call2_notools", "") == tier:
                for c in r.get("calls") or []:
                    if c.get("completion_tokens") and c.get("eval_duration_s"):
                        dec.append(c["eval_duration_s"] / c["completion_tokens"])
                    if c.get("prompt_eval_duration_s"):
                        pe.append(c["prompt_eval_duration_s"])
                        if c.get("prompt_eval_count_info_only"):
                            per.append(c["prompt_eval_duration_s"] / c["prompt_eval_count_info_only"])
        md = statistics.median(dec)
        l4[tier] = {"point_s": md * L4_TOKENS_OUT + statistics.median(per) * L4_TOKENS_IN, "decode_s_per_token": md,
                    "upper_s": md * L4_TOKENS_OUT + statistics.median(pe)}
    return {"n_tasks": n_tasks, "escalated": esc, "billed": billed, "upper": upper, "cap_ceiling": n_tasks *
            TASK_CAP_USD, "esc_ceiling": esc * TASK_CAP_USD, "hours": hours, "local_hours": local_s / 3600,
            "n_calls": n_calls, "prompt_check_s": chk, "n_mech": len(mech), "redo_s": redo_s, "l4": l4,
            "over_ask_first": billed > 2.0 or upper > 2.0}


def live_estimate_text(e: dict) -> str:
    l4 = "; ".join(f"{t}: {v['point_s']:.2f} s/task warm cache (16 tokens at the median decode "
                   f"{v['decode_s_per_token']:.3f} s/token plus {L4_TOKENS_IN} prompt tokens at the median "
                   f"prompt-eval rate), {v['upper_s']:.2f} s/task if the prompt is re-evaluated"
                   for t, v in e["l4"].items())
    return (f"ESTIMATE, not measured: live job x2_post_run_check_v1 (B3 v2 qwen3:14b, 3 tiers x 5 seeds x 30 tasks = "
            f"{e['n_tasks']} tasks, check live, rollback and Sonnet escalation): escalations expected {e['escalated']} "
            f"(the replay's headline decisions); projected cloud spend USD {e['billed']:.2f} billed (cache-priced), "
            f"USD {e['upper']:.2f} upper bound (all input uncached), from the replay's measured per-task costs; hard "
            f"ceiling at the USD {TASK_CAP_USD:.2f} per-task cap: USD {e['esc_ceiling']:.2f} for {e['escalated']} "
            f"escalations, USD {e['cap_ceiling']:.2f} if every task escalated; "
            + ("ABOVE the USD 2 per-job ask-first limit: the operator must approve it at their own console"
               if e["over_ask_first"] else "within the USD 2 per-job limit")
            + f"; hours {e['hours']:.1f} (local turns {e['local_hours']:.1f} h from the B3 v2 run, plus {e['n_calls']} "
            f"calls x {GIN_WAIT_S:.0f} s log-slice wait, {e['n_tasks']} tasks x {e['prompt_check_s']:.2f} s median "
            f"prompt check (results/x2_r2_mechanism.jsonl, {e['n_mech']} checks), {e['escalated']} escalations x "
            f"{e['redo_s']:.2f} s mean redo); L4 self-check cost (design only, not run): {l4}")


if __name__ == "__main__":
    rs = replay()
    d = load()
    for t in TIERS:
        print(ROW_OF[t], cell_text(d, t))
    print(attribution_text(d))
    print(live_estimate_text(live_estimate()))
