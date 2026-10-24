"""Detector false-alarm analysis (operator request, 2026-10-23; analysis only, no new runs).

1. Per call of the demo runs (results/demo_detector_check.jsonl, results/demo_live_run_local_v1.jsonl) the server log
   slice (results/demo_live_logs/...) gives the ground truth: a cut is an Ollama message drop (its truncation line
   with fewer messages kept than sent, read from the call's own evidence, which used the request's message count), a
   token cut ("truncating input prompt"), a llama.cpp context shift, or an exceed-context error. Two hypotheses for
   every call where detector v1 fired:
     H1 prompt-cache reuse: the token count the detector compared (the server's task.n_tokens) misses the cached
        prefix. Tested by task.n_tokens == first "cached n_tokens" + the tokens evaluated anew ("prompt eval time ...
        / N tokens"), and prompt_eval_count == task.n_tokens.
     H2 our size estimate vs the server's own tokenization: the detector's own size (chars/4 estimate x 1.2022) vs
        task.n_tokens, the server's tokenization of the rendered chat template; on a ground-truth uncut call the
        ratio task.n_tokens / chars/4 estimate is the true tokenizer ratio.
2. Re-score of row hybrid-route-at-start-measured with detector v1 as the escalation flag (B3 v2 per-call data:
   prompt_tokens_est and prompt_eval_count_info_only, which in Ollama 0.34.4 is the server total; no server logs
   were kept for that run, so ground truth there is inferred, labelled).
3. Confusion counts of v1 and of the proposed v2 against the log ground truth: the R2 mechanism run (400 calls,
   llama3.1:8b, tiers ollama_default (131072), 32768, 16384, 8192, 4096, 80 calls each; ground truth = the
   mechanism job's own per-call reading) and the demo runs.

Detector definitions (harness/demo_overflow.py):
  v1 (as run): cut = a server cut line, or task.n_tokens < 0.9 x (chars/4 estimate x ratio), ratio 1.2022 for
      qwen3:14b (route_profiles.json); for the llama3.1:8b mechanism calls, the session's own token_calib_ratio (the
      same calibration method).
  v2 (proposed, not yet re-checked on evo-x2): cut = a server cut line only (H2 shows the chars/4-based size is the
      problem; the server's own report is preferred). Against the log ground truth it agrees by construction; the
      independent check is the mechanism job's render-only prompt (render_and_log_agree).
  v2_count_ratio1 (the variant first proposed, rejected here): a server cut line, or task.n_tokens < 0.9 x the chars/4
      estimate (ratio 1.0); it fails on llama3.1:8b, whose server/estimate ratio is 0.70 to 0.86 uncut.
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RATIO_V1 = 1.2022
TOL = 0.10
DEMO_LOGS = "results/demo_live_logs"

_NEW = re.compile(r"new prompt, n_ctx_slot = (\d+), n_keep = (-?\d+), task\.n_tokens = (\d+)")
_CACHED = re.compile(r"cached n_tokens = (\d+), memory_seq_rm")
_EVAL = re.compile(r"prompt eval time = .*?/\s*(\d+) tokens")
_STOP = re.compile(r"stop processing: n_tokens = (\d+), truncated = (\d+)")
_DROP = re.compile(r'msg="truncating input messages which exceed context length" truncated=(\d+)')
_TOKCUT = re.compile(r'msg="truncating input prompt"')
_SHIFT = re.compile(r"context shift, n_keep")
_EXCEED = re.compile(r"exceeds the available context size|is larger than the max context size")


def read_jsonl(p):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(x) for x in fh if x.strip()]


def parse_log(text: str) -> list[dict]:
    """One record per llama-server task, in order. Lines from the previous task's stop to this task's new prompt
    (Ollama's message-drop / token-cut lines are written before the request) belong to this task."""
    out, pre = [], {"drop": None, "tokcut": False}
    cur = None
    for line in text.splitlines():
        m = _DROP.search(line)
        if m:
            pre["drop"] = int(m.group(1))
            continue
        if _TOKCUT.search(line):
            pre["tokcut"] = True
            continue
        m = _NEW.search(line)
        if m:
            cur = {"n_ctx_slot": int(m.group(1)), "n_tokens": int(m.group(3)), "cached": None, "evaluated": None,
                   "shift": 0, "exceed": False, "stop_truncated": None, "drop_kept": pre["drop"],
                   "tokcut": pre["tokcut"]}
            out.append(cur)
            pre = {"drop": None, "tokcut": False}
            continue
        if cur is None:
            continue
        m = _CACHED.search(line)
        if m and cur["cached"] is None:
            cur["cached"] = int(m.group(1))
        m = _EVAL.search(line)
        if m:
            cur["evaluated"] = int(m.group(1))
        if _SHIFT.search(line):
            cur["shift"] += 1
        if _EXCEED.search(line):
            cur["exceed"] = True
        m = _STOP.search(line)
        if m:
            cur["stop_truncated"] = int(m.group(2))
    return out


def demo_calls(repo=REPO) -> list[dict]:
    """Every local call of the demo runs with its log task, in run order per server."""
    out = []
    det = read_jsonl(repo / "results" / "demo_detector_check.jsonl")
    for arm, log in (("num_ctx_4096", "serve_4096.log"), ("default", "serve_default.log")):
        calls = [dict(c, task_id=r["task_id"], seed=r["seed"], lane=arm)
                 for r in det if r.get("record") == "detector_task" and r["arm"] == arm for c in r["calls"]]
        out += _attach(calls, repo / DEMO_LOGS / "detector_check" / log, "detector check")
    acc = read_jsonl(repo / "results" / "demo_live_run_local_v1.jsonl")
    for ctx in (4096, 8192, 32768):
        calls = [r for r in acc if r.get("record") == "call" and r.get("num_ctx") == ctx]
        out += _attach(calls, repo / DEMO_LOGS / "acceptance" / f"serve_{ctx}.log", "acceptance run")
    return out


def _attach(calls, log_path, source):
    tasks = parse_log(Path(log_path).read_text(encoding="utf-8", errors="replace"))
    if len(tasks) != len(calls):
        raise ValueError(f"{log_path}: {len(tasks)} server tasks vs {len(calls)} calls")
    out = []
    for c, t in zip(calls, tasks):
        if c.get("used_tokens") is not None and c["used_tokens"] != t["n_tokens"]:
            raise ValueError(f"{log_path}: call/log order mismatch ({c['used_tokens']} vs {t['n_tokens']})")
        drop = any(str(e).startswith("older messages dropped") for e in c.get("evidence") or [])
        truth = bool(drop or t["tokcut"] or t["shift"] or t["exceed"])
        est = c["own_tokens"] / RATIO_V1
        out.append({"source": source, "server": Path(log_path).name, "lane": c.get("lane"), "task_id": c.get("task_id"),
                    "seed": c.get("seed"), "n": c.get("n"), "est": est, "own_v1": c["own_tokens"],
                    "n_tokens": t["n_tokens"], "n_ctx": t["n_ctx_slot"], "cached": t["cached"],
                    "evaluated": t["evaluated"], "prompt_eval_count": c.get("prompt_eval_count"),
                    "drop": drop, "tokcut": t["tokcut"], "shift": t["shift"], "exceed": t["exceed"],
                    "stop_truncated": t["stop_truncated"], "truth_cut": truth, "v1": bool(c.get("cut")),
                    "v1_count_only": bool(c.get("cut")) and not truth,
                    "v2": truth, "v2_count_ratio1": truth or t["n_tokens"] < (1 - TOL) * est})
    return out


def hypotheses(c: dict) -> dict:
    h1_total = c["cached"] is not None and c["evaluated"] is not None and c["n_tokens"] == c["cached"] + c["evaluated"]
    h1 = {"cached": c["cached"], "evaluated": c["evaluated"], "n_tokens": c["n_tokens"],
          "n_tokens_includes_cache": h1_total, "prompt_eval_count_eq_total": c["prompt_eval_count"] == c["n_tokens"],
          "explains": not (h1_total and c["prompt_eval_count"] == c["n_tokens"])}
    ratio = c["n_tokens"] / c["est"]
    h2 = {"ratio_server_over_chars4": round(ratio, 3), "v1_threshold_ratio": round(RATIO_V1 * (1 - TOL), 4),
          "explains": (not c["truth_cut"]) and ratio < RATIO_V1 * (1 - TOL)}
    return {"H1": h1, "H2": h2, "explained_by": ("H2" if h2["explains"] else "") + ("H1" if h1["explains"] else "")
            or ("real cut" if c["truth_cut"] else "neither")}


def mechanism_calls(repo=REPO) -> list[dict]:
    """The 400 R2 mechanism calls with v1 / v2 computed from the call's own chars/4 estimate (the r2a_turn call
    records) and task.n_tokens; ground truth = the mechanism job's reading (message drop, context shift, token cut,
    exceed)."""
    rows = read_jsonl(repo / "results" / "x2_r2_mechanism.jsonl")
    turns = {(r["arm_id"], r["seed"], r["turn_idx"]): r for r in rows if r.get("record") == "r2a_turn"}
    out = []
    for r in rows:
        if r.get("record") != "r2m_call":
            continue
        t = turns[(r["arm_id"], r["seed"], r["turn_idx"])]
        call = t["calls"][r["call_idx"] - 1]
        est = call["prompt_tokens_est"]
        ratio = t.get("token_calib_ratio") or 1.0
        n_tok = r["new_prompt"][-1]["n_tokens"]
        truth = bool(r["message_truncation_in_log"] or r["context_shifts"] or r["token_cut"] or r["exceed_context_error"])
        out.append({"source": "mechanism", "tier": r["arm_id"].replace("_call2_notools", ""), "est": est,
                    "n_tokens": n_tok, "truth_cut": truth, "v1": truth or n_tok < (1 - TOL) * est * ratio,
                    "v2": truth, "v2_count_ratio1": truth or n_tok < (1 - TOL) * est, "calib_ratio": ratio,
                    "pec_eq_total": r["prompt_eval_count"] == n_tok})
    return out


def confusion(calls, key) -> dict:
    tp = sum(1 for c in calls if c[key] and c["truth_cut"])
    fp = sum(1 for c in calls if c[key] and not c["truth_cut"])
    fn = sum(1 for c in calls if not c[key] and c["truth_cut"])
    tn = sum(1 for c in calls if not c[key] and not c["truth_cut"])
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None, "n": len(calls)}


def b3_rescore(repo=REPO) -> dict:
    """Row hybrid-route-at-start-measured re-scored with detector v1 as the escalation flag. Ground truth (no server
    logs for that run, INFERRED): a call is cut for certain if its chars/4 estimate alone exceeds the loaded window
    (every observed uncut qwen3:14b call has server/estimate >= 1.021), and cannot be cut if its server total plus
    nothing more is far below the window and the estimate x 1.21 (the largest observed ratio) still fits."""
    sys.path.insert(0, str(repo / "analysis"))
    import impact_model as im
    rows = read_jsonl(repo / "results" / "x2_evalset_tasks_real_v2.jsonl")
    base = im.route_at_start_measured(repo)
    local = im.replay_local_tasks(repo)
    ac = im.replay_cloud_tasks(repo)
    redo, redo_model = im.replay_redo_tasks(repo)
    seeds = ac["seeds"]
    proj = {(s, tid): p for s in seeds for tid, (_i, p) in im.projected_task_prompts(repo, s).items()}
    turns = {(r["arm_id"].replace("_call2_notools", ""), r["seed"], r["task_id"]): r for r in rows
             if r.get("record") == "r2a_turn" and r.get("item_kind") == "task" and r.get("model_id") == "qwen3:14b"}
    out = {"label": "re-score by replay with detector v1 as the flag; ground truth INFERRED (no server logs kept)",
           "cells": []}
    for tier in ("ollama_ctx_8192", "ollama_default"):
        w = im.tier_window(tier) if tier != "ollama_default" else 40960
        v1 = {}
        info = {"fired": 0, "fired_certain_cut": 0, "fired_no_cut_possible": 0, "fired_undetermined": 0}
        for k in local[tier]:
            r = turns.get((tier, k[0], k[1]))
            if r is None:
                continue
            fired = certain = possible = False
            for c in r["calls"]:
                pec, est = c.get("prompt_eval_count_info_only"), c.get("prompt_tokens_est")
                if pec is None or not est:
                    continue
                if pec < (1 - TOL) * RATIO_V1 * est:
                    fired = True
                if est > w:
                    certain = True
                if est * 1.21 > w:
                    possible = True
            v1[k] = dict(local[tier][k], flag=fired)
            if fired and k[0] in set(seeds) and proj.get(k, 10 ** 9) <= w:
                info["fired"] += 1
                info["fired_certain_cut" if certain else ("fired_undetermined" if possible
                                                          else "fired_no_cut_possible")] += 1
        t_redo = {(s, t): v for (tr, s, t), v in redo.items() if tr == tier}
        t_usd = statistics.fmean(v["usd"] for v in t_redo.values()) if t_redo else None
        for m in sorted(ac["models"]):
            measured = m == redo_model
            kw = dict(redo=t_redo if measured else None, redo_tier_usd=t_usd,
                      esc_cloud=ac["models"][redo_model] if measured else ac["models"][m])
            if measured and not t_redo:   # no redo rows at this tier: impute at the 8192 tier's mean, labelled
                r8 = {(s, t): v for (tr, s, t), v in redo.items() if tr == "ollama_ctx_8192"}
                kw = dict(redo={}, redo_tier_usd=statistics.fmean(v["usd"] for v in r8.values()),
                          esc_cloud=ac["models"][redo_model])
            res = im.route_measured_policy(v1, ac["models"][m], proj, w, seeds, **kw)
            row = next((c for c in base["cells"] if c["tier"] == tier and c["model"] == m), None)
            out["cells"].append({"tier": tier, "window": w, "model": m, "v1": {
                k2: res[k2] for k2 in ("n", "correct", "usd_task", "kept_local", "escalated", "routed")},
                "row": ({k2: row["hybrid"][k2] for k2 in ("n", "correct", "usd_task", "kept_local", "escalated",
                                                         "routed")} if row else None),
                "fired_on_local_routed": info, "imputed_note": (
                    "no full-history redo rows at this tier: escalations imputed at the 8192 tier's mean redo USD"
                    if measured and not t_redo else None)})
    return out


def summary(repo=REPO) -> dict:
    demo = demo_calls(repo)
    mech = mechanism_calls(repo)
    fired = [dict(c, **hypotheses(c)) for c in demo if c["v1"] and not c["truth_cut"]]
    by_src = {}
    for src, calls in (("mechanism", mech), ("demo detector check", [c for c in demo if c["source"] == "detector check"]),
                       ("demo acceptance run", [c for c in demo if c["source"] == "acceptance run"])):
        by_src[src] = {"v1": confusion(calls, "v1"), "v2": confusion(calls, "v2"),
                       "v2_count_ratio1": confusion(calls, "v2_count_ratio1")}
    pooled = mech + demo
    by_src["pooled"] = {"v1": confusion(pooled, "v1"), "v2": confusion(pooled, "v2"),
                        "v2_count_ratio1": confusion(pooled, "v2_count_ratio1")}
    rows = read_jsonl(repo / "results" / "x2_r2_mechanism.jsonl")
    agree = [r.get("render_and_log_agree") for r in rows if r.get("record") == "r2m_call"]
    mech_default = [c for c in mech if c["tier"] == "ollama_default"]
    return {"false_alarm_calls": fired, "confusion": by_src,
            "render_log_agreement": (sum(1 for a in agree if a is True), sum(1 for a in agree if a is not None)),
            "mechanism_uncut_ratio_range": (min(c["n_tokens"] / c["est"] for c in mech if not c["truth_cut"]),
                                            max(c["n_tokens"] / c["est"] for c in mech if not c["truth_cut"])),
            "mechanism_cut_ratio_max": max(c["n_tokens"] / c["est"] for c in mech if c["truth_cut"]),
            "mechanism_pec_eq_total": {t: (sum(c["pec_eq_total"] for c in mech if c["tier"] == t),
                                           sum(1 for c in mech if c["tier"] == t))
                                       for t in sorted({c["tier"] for c in mech})},
            "mechanism_default_v1_fires": sum(c["v1"] for c in mech_default),
            "demo_uncut_ratio_range": (min(c["n_tokens"] / c["est"] for c in demo if not c["truth_cut"]),
                                       max(c["n_tokens"] / c["est"] for c in demo if not c["truth_cut"])),
            "demo_cut_ratio_max": max(c["n_tokens"] / c["est"] for c in demo if c["truth_cut"])}


if __name__ == "__main__":
    s = summary()
    print(json.dumps({k: v for k, v in s.items() if k != "false_alarm_calls"}, indent=1, default=str))
    for c in s["false_alarm_calls"]:
        print(c["source"], c["server"], c["lane"], c["task_id"], c["n"], "n_tokens", c["n_tokens"], "own_v1",
              c["own_v1"], "cached", c["cached"], "eval", c["evaluated"], "pec", c["prompt_eval_count"],
              "ratio", c["H2"]["ratio_server_over_chars4"], "->", c["explained_by"])
