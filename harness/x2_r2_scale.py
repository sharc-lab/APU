"""H-scale (pre-registered in docs/FINDINGS.md, 2026-10-14): R2 at 30B to 70B on evo-x2. Read that section first.

Three queue jobs, one per mode, all on the R2 harness (harness/x2_r2_agent.py: session plan, two-call turn, scoring,
canaries, gates) with call-2 mode "off" (tools withheld on a turn's second call, the protocol of every Ollama R2 run
since 2026-10-08):

  --mode probe       queue id x2_scale_timing_probe, results/x2_scale_timing_probe.jsonl. One 2-turn session per
                     model present in the store and per tier (num_ctx 4096, num_ctx 32768, Ollama default), seed
                     20260901. Then project_all(): minutes per 40-turn session per model and tier from a reference
                     model's completed sessions at the same tier (x2_r2_real_v1/v1b) with prompt-eval and eval time
                     rescaled by the probe's measured rates (see project_session_minutes), the validation job's minutes
                     the same way from x2_r2_validation_v2, and the mechanical llama3.3:70b default-tier decision
                     (projected 3 sessions <= DEFAULT_TIER_CAP_H). Writes a scale_projection record and .report.json.
  --mode validation  queue id x2_r2_scale_validation, results/x2_r2_scale_validation.jsonl. x2_r2_agent.validation_plan
                     ("off") exactly as v2/v2b (negative control 131072 3 x 10, positive control 8192 1 x 15, diagnostic
                     arm 3 x 10) for every model present; gates by x2_r2_agent.evaluate_gates / validation_preflight.
  --mode real        queue id x2_r2_scale_real, results/x2_r2_scale_real.jsonl. Per-model refusal through
                     x2_r2_agent.validation_preflight_per_model on the validation file; 4096 and 32768 x SEEDS x 40
                     turns for every passing model (seed-major), then Ollama default for DEFAULT_TIER_MODELS and for
                     llama3.3:70b only if the probe file's projection fits (default_tier_decision). Resumable: a
                     completed (model, arm, seed) in the output file is not rerun. Report: h_scale_verdict + the R2
                     kill criterion.

A model not in the Ollama store (llama3.3:70b until its library tag is pulled; see the pre-registration: only a GGUF is
on disk) gets a missing_model row and no model call; nothing else changes.

Every mode calls t2s_queue.advance() exactly once on exit (finally) and sets no pause. Not for a bare run on evo-x2
while a queued job runs (advance() refuses a caller the queue did not launch). Modules are imported from this file's
directory (C:\\apu\\ovn when deployed); run_start records the git blob of the x2_r2_agent.py actually imported.

Two operator utilities (no model call, no server start):
  --import-check     import, parse, print the plan of every mode and the models' expected think values; exit.
  --queue-insert     the one read-modify-write of evo-x2's queue for these three jobs (insert_jobs, QUEUE_ORDER) under
                     t2s_queue.set_pause with a reason naming its release, clear_pause() in a finally block.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
if (_here.parent / "harness").is_dir():
    sys.path.insert(0, str(_here.parent / "harness"))

import x2_r2_agent as ag  # noqa: E402

# ── configuration (fixed by the pre-registration) ───────────────────────────────────────────────────

SCALE_MODELS = ("qwen3:32b", "qwen3-30b-a3b-2507", "llama3.3:70b")
# think sent per model (None = field omitted): qwen3 hybrid models get false; Llama has no thinking mode.
SCALE_THINK = {"qwen3:32b": False, "qwen3-30b-a3b-2507": False, "llama3.3:70b": None}
for _m, _t in SCALE_THINK.items():
    ag.MODELS.setdefault(_m, _t)
REFERENCE_MODEL = {"qwen3:32b": "qwen3:14b", "qwen3-30b-a3b-2507": "qwen3:14b", "llama3.3:70b": "llama3.1:8b"}
CALL2 = "off"
SFX = ag.MODE_SUFFIX[CALL2]
TIER_4096, TIER_32768, TIER_DEFAULT = "ollama_ctx_4096", "ollama_ctx_32768", "ollama_default"
PROBE_TIERS = (TIER_4096, TIER_32768, TIER_DEFAULT)
PROBE_TURNS = 2
PROBE_SEED = ag.SEEDS[0]
RATE_TURNS = (1, 2)
DECIDING_TIERS = (TIER_4096, TIER_32768)
DEFAULT_TIER_MODELS = ("qwen3:32b", "qwen3-30b-a3b-2507")
CONDITIONAL_DEFAULT_MODELS = ("llama3.3:70b",)
DEFAULT_TIER_CAP_H = 24.0
REAL_SEEDS = tuple(ag.SEEDS)
REAL_TURNS = 40
REFERENCE_REAL_FILES = ("x2_r2_real_v1.jsonl", "x2_r2_real_v1b.jsonl")
REFERENCE_VALIDATION_FILE = "x2_r2_validation_v2.jsonl"
PROBE_FILE = "x2_scale_timing_probe.jsonl"
VALIDATION_FILE = "x2_r2_scale_validation.jsonl"
REAL_FILE = "x2_r2_scale_real.jsonl"
JOB_IDS = {"probe": "x2_scale_timing_probe", "validation": "x2_r2_scale_validation", "real": "x2_r2_scale_real"}
# Queue order on evo-x2 (operator, 2026-10-13; several agents insert): a job goes immediately before the first pending
# entry that comes later in this list, otherwise at the end. A trailing "*" matches any id with that prefix.
QUEUE_ORDER = ("x2_num_keep_guardrail_test", "x2_window_override_methods", "x2_scale_timing_probe",
               "x2_r2_scale_validation", "x2_r2_scale_real", "x2_r2_family_*", "x2_policy_compare")
METRIC_FIELD = {"tool_validity": "tool_validity_all", "tool_args": "tool_args_all", "recall": "recall_ok"}


def arm(tier: str) -> str:
    return tier + SFX


def git_blob_sha1(path) -> str | None:
    try:
        b = Path(path).read_bytes()
    except OSError:
        return None
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


# ── projection (pure) ───────────────────────────────────────────────────────────────────────────────

def _turns_by_attempt(rows):
    out = {}
    for r in rows:
        if r.get("record") == "r2a_turn":
            out.setdefault(r.get("attempt_id"), []).append(r)
    for v in out.values():
        v.sort(key=lambda r: r["turn_idx"])
    return out


def rates(turn_rows, turns=RATE_TURNS) -> dict:
    """Prompt-eval and eval rates (tokens per second, totals over the calls of the given turns) and the turn-1 call-1
    load time. Calls without both a count and a duration are left out of that rate."""
    pe_tok = pe_s = ev_tok = ev_s = 0.0
    load = None
    for r in turn_rows:
        if r["turn_idx"] not in turns:
            continue
        for i, c in enumerate(r.get("calls") or []):
            if c.get("prompt_eval_count_info_only") and c.get("prompt_eval_duration_s"):
                pe_tok += c["prompt_eval_count_info_only"]
                pe_s += c["prompt_eval_duration_s"]
            if c.get("completion_tokens") and c.get("eval_duration_s"):
                ev_tok += c["completion_tokens"]
                ev_s += c["eval_duration_s"]
            if r["turn_idx"] == min(turns) and i == 0:
                load = c.get("load_duration_s")
    return {"pe_tok": pe_tok, "pe_s": pe_s, "ev_tok": ev_tok, "ev_s": ev_s,
            "pe_rate": pe_tok / pe_s if pe_s else None, "ev_rate": ev_tok / ev_s if ev_s else None,
            "load_s": load}


def pooled_rates(sessions_turns) -> dict:
    """rates() pooled over several sessions (token and second totals summed, load = mean of the sessions' loads)."""
    parts = [rates(t) for t in sessions_turns]
    pe_tok, pe_s = sum(p["pe_tok"] for p in parts), sum(p["pe_s"] for p in parts)
    ev_tok, ev_s = sum(p["ev_tok"] for p in parts), sum(p["ev_s"] for p in parts)
    loads = [p["load_s"] for p in parts if p["load_s"] is not None]
    return {"pe_tok": pe_tok, "pe_s": pe_s, "ev_tok": ev_tok, "ev_s": ev_s,
            "pe_rate": pe_tok / pe_s if pe_s else None, "ev_rate": ev_tok / ev_s if ev_s else None,
            "load_s": sum(loads) / len(loads) if loads else None}


def project_session_minutes(ref_turns, ref_rates: dict, new_rates: dict) -> float:
    """Minutes the new model would take for the reference session: the reference turn wall times, with every call's
    prompt-eval duration multiplied by k_pe = ref pe_rate / new pe_rate and its eval duration by k_ev = ref ev_rate /
    new ev_rate (all other time kept), and the reference session's turn-1 load replaced by the new model's load."""
    k_pe = ref_rates["pe_rate"] / new_rates["pe_rate"]
    k_ev = ref_rates["ev_rate"] / new_rates["ev_rate"]
    total = 0.0
    for r in ref_turns:
        total += r.get("turn_wall_s") or 0.0
        for c in r.get("calls") or []:
            total += (c.get("prompt_eval_duration_s") or 0.0) * (k_pe - 1.0)
            total += (c.get("eval_duration_s") or 0.0) * (k_ev - 1.0)
    ref_load = rates(ref_turns)["load_s"] or 0.0
    total += (new_rates.get("load_s") or 0.0) - ref_load
    return total / 60.0


def reference_sessions(rows, model, arm_id, mode, turns=None) -> list[list[dict]]:
    """Turn rows of each completed session (latest per (mode, model, arm, seed)) of `model` on `arm_id`, optionally
    only sessions with exactly `turns` turn rows."""
    tba = _turns_by_attempt(rows)
    out = []
    for s in ag.completed_sessions(rows):
        if s["model_id"] != model or s["arm_id"] != arm_id or s.get("mode") != mode:
            continue
        t = tba.get(s["attempt_id"], [])
        if turns is None or len(t) == turns:
            out.append(t)
    return out


def probe_sessions(probe_rows) -> dict:
    """{(model, tier): turn rows} of the probe's completed sessions."""
    tba = _turns_by_attempt(probe_rows)
    out = {}
    for s in ag.completed_sessions(probe_rows):
        if s.get("mode") == "probe":
            out[(s["model_id"], ag.base_arm(s["arm_id"]))] = tba.get(s["attempt_id"], [])
    return out


def project_all(probe_rows, ref_real_rows, ref_val_rows, seeds=len(REAL_SEEDS)) -> dict:
    """Per model with probe sessions: per tier the projected minutes per 40-turn session (mean over the reference
    sessions) and hours for `seeds` sessions; the validation job's projected minutes (sum over the reference model's
    validation sessions, default-tier probe rates); the llama3.3:70b default-tier decision; the real job's hours."""
    probes = probe_sessions(probe_rows)
    out = {"models": {}, "method": "reference sessions rescaled by probe rates (docs/FINDINGS.md H-scale)",
           "default_tier_cap_h": DEFAULT_TIER_CAP_H}
    for model in SCALE_MODELS:
        ref_model = REFERENCE_MODEL[model]
        mres = {"reference_model": ref_model, "tiers": {}}
        for tier in PROBE_TIERS:
            pt = probes.get((model, tier))
            if not pt:
                continue
            refs = reference_sessions(ref_real_rows, ref_model, arm(tier), "real", turns=REAL_TURNS)
            new = rates(pt)
            entry = {"probe_rates": new, "probe_loaded_context": pt[-1].get("loaded_context"),
                     "probe_thinking_turns": sum(1 for r in pt if r.get("thinking_present")),
                     "probe_turn_wall_s": [r.get("turn_wall_s") for r in pt], "n_reference_sessions": len(refs)}
            if refs and new["pe_rate"] and new["ev_rate"]:
                ref = pooled_rates(refs)
                mins = [project_session_minutes(t, ref, new) for t in refs]
                entry.update({"reference_rates": ref,
                              "reference_loaded_context": sorted({r.get("loaded_context") for t in refs for r in t
                                                                  if r.get("loaded_context")}),
                              "k_pe": ref["pe_rate"] / new["pe_rate"], "k_ev": ref["ev_rate"] / new["ev_rate"],
                              "minutes_per_session": sum(mins) / len(mins),
                              "hours_for_seeds": seeds * (sum(mins) / len(mins)) / 60.0})
            mres["tiers"][tier] = entry
        if not mres["tiers"]:
            continue
        dp = probes.get((model, TIER_DEFAULT))
        vrefs = [t for s_arm in {a for a, _, _ in ag.validation_plan(CALL2)}
                 for t in reference_sessions(ref_val_rows, ref_model, s_arm, "validation")]
        if dp and vrefs:
            new = rates(dp)
            if new["pe_rate"] and new["ev_rate"]:
                ref = pooled_rates(vrefs)
                mres["validation_minutes"] = sum(project_session_minutes(t, ref, new) for t in vrefs)
                mres["n_validation_reference_sessions"] = len(vrefs)
        out["models"][model] = mres
    for model in SCALE_MODELS:
        if model in out["models"]:
            out["models"][model]["default_tier"] = default_tier_decision(out, model)
    out["real_hours_if_all_pass"] = real_hours(out, list(out["models"]))
    out["validation_hours"] = sum((m.get("validation_minutes") or 0.0) for m in out["models"].values()) / 60.0
    return out


def default_tier_decision(projection, model) -> dict:
    """Whether the real job runs Ollama default for `model`: always for DEFAULT_TIER_MODELS; for
    CONDITIONAL_DEFAULT_MODELS only if the projected hours for its seeds are <= DEFAULT_TIER_CAP_H; never otherwise."""
    if model in DEFAULT_TIER_MODELS:
        return {"include": True, "reason": "planned default tier (operator)"}
    if model not in CONDITIONAL_DEFAULT_MODELS:
        return {"include": False, "reason": "not planned"}
    m = (projection or {}).get("models", {}).get(model) or {}
    h = ((m.get("tiers") or {}).get(TIER_DEFAULT) or {}).get("hours_for_seeds")
    if h is None:
        return {"include": False, "reason": "no probe projection for the default tier"}
    ok = h <= DEFAULT_TIER_CAP_H
    return {"include": ok, "projected_hours": h,
            "reason": f"projected {h:.1f} h for {len(REAL_SEEDS)} default sessions "
                      f"{'<=' if ok else '>'} cap {DEFAULT_TIER_CAP_H} h"}


def real_hours(projection, models) -> dict:
    """Projected hours of the real job per model (4096 + 32768, plus default when default_tier_decision includes
    it) and in total, for the given models."""
    per = {}
    for model in models:
        m = projection["models"].get(model) or {}
        tiers = list(DECIDING_TIERS) + ([TIER_DEFAULT] if default_tier_decision(projection, model)["include"] else [])
        hs = [((m.get("tiers") or {}).get(t) or {}).get("hours_for_seeds") for t in tiers]
        per[model] = {"tiers": tiers, "hours": None if any(h is None for h in hs) else sum(hs)}
    known = [v["hours"] for v in per.values() if v["hours"] is not None]
    return {"per_model": per, "total_known": sum(known), "complete": all(v["hours"] is not None for v in per.values())}


# ── real-run plan (pure) ────────────────────────────────────────────────────────────────────────────

def real_items(models, include_default: dict, seeds=REAL_SEEDS, turns=REAL_TURNS) -> list[tuple]:
    """[(model, arm, seed, turns)]: every 4096 and 32768 session first, seed-major (seed, then model, then tier), then
    the default sessions (seed, then model) of the models include_default marks True."""
    items = [(m, arm(t), s, turns) for s in seeds for m in models for t in DECIDING_TIERS]
    items += [(m, arm(TIER_DEFAULT), s, turns) for s in seeds for m in models if include_default.get(m)]
    return items


def run_items(runtime, items, mode, out_path: Path, emit, log):
    """run_plan's loop for an explicit item list: a session completed in out_path under this mode is skipped."""
    done = {(s["mode"], s["model_id"], s["arm_id"], s["seed"]) for s in ag.completed_sessions(ag.read_rows(out_path))}
    for model, arm_id, seed, turns in items:
        if (mode, model, arm_id, seed) in done:
            log(f"skip (done): {model} {arm_id} seed={seed}")
            continue
        emit({"record": "heartbeat", "mode": mode, "model_id": model, "arm_id": arm_id, "seed": seed,
              "call2_mode": ag.ARMS[arm_id]["call2_mode"], "ts_utc": ag.utc_iso()})
        log(f"session: {model} {arm_id} seed={seed} turns={turns}")
        s = ag.run_session(runtime, model, arm_id, seed, turns, emit, log, mode)
        log(f"session done: first_failure={s['first_failure_turn']} {s['first_failure_reasons']} "
            f"truncation={s['truncation_detected_turn']} canary_misses={s['canary_sys_misses']}+"
            f"{s['canary_hist_misses']}")


# ── H-scale verdict (pure; register row H-scale-verdict) ───────────────────────────────────────────

def _first(turns, pred):
    return next((r["turn_idx"] for r in turns if pred(r)), None)


def session_flags(turns, rules, gated_fields=None) -> dict:
    """Per session: first over-window turn, first error, first rule-in-use failure, whether a silent rule-in-use
    failure happened at or after the first over-window turn (the first rule failure at/after it, with no error surfaced
    on an earlier turn), whether a canary was missed on a canary turn over the loaded window, and the same silent
    after-window check for the full gated field set (reported, not decisive)."""
    turns = sorted(turns, key=lambda r: r["turn_idx"])
    first_over = _first(turns, lambda r: r.get("over_loaded_window"))
    first_err = _first(turns, lambda r: r.get("any_http_error"))

    def silent_after(fields):
        if first_over is None:
            return False, None
        t = _first(turns, lambda r: r["turn_idx"] >= first_over and any(r.get(f) is False for f in fields))
        if t is None:
            return False, None
        return not (first_err is not None and first_err < t), t

    rule_silent, rule_turn = silent_after(rules)
    gated_silent, _ = silent_after(gated_fields if gated_fields is not None else rules)
    return {"n_turns": len(turns), "first_over_window": first_over, "first_error": first_err,
            "first_rule_failure": _first(turns, lambda r: any(r.get(f) is False for f in rules)),
            "rule_failures_within_window": sum(1 for r in turns if not r.get("over_loaded_window")
                                               and any(r.get(f) is False for f in rules)),
            "rule_failure_after_window": rule_turn is not None,
            "silent_rule_failure_after_window": rule_silent, "silent_rule_failure_turn": rule_turn,
            "silent_gated_failure_after_window": gated_silent,
            "canary_miss_over_window": ag.truncation_turn(turns) is not None,
            "first_canary_miss": _first(turns, lambda r: r.get("canary_check") and (
                r.get("canary_sys_ok") is False or r.get("canary_hist_ok") is False))}


def gated_fields_from(baseline_entry, rules) -> list[str]:
    """Rules in use plus each non-rule metric whose validation baseline reached the 90% gate (R2-real-v1-gated-kill)."""
    e = baseline_entry or {}
    return list(rules) + [f for m, f in METRIC_FIELD.items()
                          if ((e.get(m) or {}).get("rate") or 0) >= ag.BASELINE_THRESHOLD]


def tier_counts(flags_by_seed: dict) -> dict:
    return {"n": len(flags_by_seed),
            "exceeded_window": sum(1 for f in flags_by_seed.values() if f["first_over_window"] is not None),
            "silent_rule_after_window": sum(1 for f in flags_by_seed.values() if f["silent_rule_failure_after_window"]),
            "any_rule_after_window": sum(1 for f in flags_by_seed.values() if f["rule_failure_after_window"]),
            "silent_gated_after_window": sum(1 for f in flags_by_seed.values()
                                             if f["silent_gated_failure_after_window"]),
            "history_lost_rules_held": sum(1 for f in flags_by_seed.values()
                                           if f["canary_miss_over_window"] and f["first_rule_failure"] is None)}


def model_verdict(c4096: dict, c32768: dict) -> dict:
    """The pre-registered per-model verdict from the two tiers' counts (3 complete sessions each). "Resists" also
    needs no rule-in-use failure after the window at all, silent or not (amendment of 2026-10-14, before any data)."""
    r, h = c4096["silent_rule_after_window"], c32768["history_lost_rules_held"]
    if r >= 2:
        at4096 = "rule loss holds"
    elif r == 0 and c4096["exceeded_window"] == c4096["n"] and c4096["any_rule_after_window"] == 0:
        at4096 = "resists rule loss"
    else:
        at4096 = "mixed"
    if h >= 2:
        at32768 = "history loss with rules holding holds"
    elif c32768["exceeded_window"] < 2:
        at32768 = "inconclusive (window exceeded in fewer than 2 sessions)"
    else:
        at32768 = "does not hold"
    if at4096 == "rule loss holds" and at32768 == "history loss with rules holding holds":
        v = "H-scale holds"
    elif at4096 == "resists rule loss":
        v = "alternative (resists rule loss)"
    else:
        v = "mixed"
    return {"verdict": v, "at_4096": at4096, "at_32768": at32768, "R4096": r, "H32768": h}


def h_scale_verdict(real_rows, validation_rows, models=SCALE_MODELS) -> dict:
    """Per model: missing / refused (not tested) / inconclusive (incomplete) / the model_verdict; overall verdict."""
    starts = [r for r in real_rows if r.get("record") == "run_start" and r.get("mode") == "real"]
    finished = any(r.get("record") == "run_end" for r in real_rows)
    refused = {}
    for r in real_rows:
        if r.get("record") == "refused_model":
            refused[r["model_id"]] = r
    missing = {r["model_id"] for r in real_rows if r.get("record") == "missing_model"}
    rules_in_use = {}
    for s in starts:
        rules_in_use.update(s.get("rules_in_use") or {})
    table = ag.evaluate_gates(validation_rows, call2_tools=CALL2)["baseline_table"] if validation_rows else {}
    tba = _turns_by_attempt(real_rows)
    sessions = [s for s in ag.completed_sessions(real_rows) if s.get("mode") == "real"]
    per = {}
    for model in models:
        if model in missing and model not in rules_in_use:
            per[model] = {"verdict": "missing", "reason": "not in the Ollama store"}
            continue
        if model in refused:
            per[model] = {"verdict": "refused (not tested)", "category": refused[model].get("refusal_category"),
                          "reasons": refused[model].get("reasons")}
            continue
        if model not in rules_in_use:
            per[model] = {"verdict": "not run", "reason": "no rules in use for it in any real run_start"}
            continue
        rules = list(rules_in_use[model])
        gf = gated_fields_from(table.get(model), rules)
        tiers, problems = {}, []
        for tier in DECIDING_TIERS + (TIER_DEFAULT,):
            fl = {}
            for s in sessions:
                if s["model_id"] == model and s["arm_id"] == arm(tier):
                    fl[s["seed"]] = session_flags(tba.get(s["attempt_id"], []), rules, gf)
            if tier in DECIDING_TIERS:
                complete = sorted(sd for sd, f in fl.items() if f["n_turns"] == REAL_TURNS)
                if complete != sorted(REAL_SEEDS):
                    problems.append(f"{tier}: complete 40-turn sessions for seeds {complete}, "
                                    f"expected {list(REAL_SEEDS)}")
            if fl:
                tiers[tier] = {"counts": tier_counts(fl), "per_seed": fl}
        if not finished:
            problems.append("no run_end in the real file")
        entry = {"rules_in_use": rules, "gated_fields": gf, "tiers": tiers}
        if problems:
            entry.update({"verdict": "inconclusive (incomplete)", "problems": problems})
        else:
            entry.update(model_verdict(tiers[TIER_4096]["counts"], tiers[TIER_32768]["counts"]))
        per[model] = entry
    tested = [m for m, e in per.items() if e["verdict"] in ("H-scale holds", "alternative (resists rule loss)", "mixed")]
    if tested and all(per[m]["verdict"] == "H-scale holds" for m in tested):
        overall = "H-scale holds"
    elif tested and all(per[m]["verdict"] == "alternative (resists rule loss)" for m in tested):
        overall = "alternative supported"
    elif tested:
        overall = "mixed"
    else:
        overall = "no model tested"
    return {"overall": overall, "tested": tested, "per_model": per, "finished": finished}


# ── queue insertion (pure part tested; the CLI runs it on evo-x2) ──────────────────────────────────

def _order_index(job_id: str):
    for i, pat in enumerate(QUEUE_ORDER):
        if pat.endswith("*") and job_id.startswith(pat[:-1]):
            return i
        if job_id == pat:
            return i
    return None


def insert_jobs(items: list[dict], new_items: list[dict]) -> list[dict]:
    """Each new item goes immediately before the first PENDING existing item that comes later in QUEUE_ORDER,
    otherwise at the end. An id already present is not inserted again (ValueError: a second insert is a mistake)."""
    items = list(items)
    have = {it["id"] for it in items}
    for new in new_items:
        if new["id"] in have:
            raise ValueError(f"{new['id']} is already in the queue")
        mine = _order_index(new["id"])
        pos = len(items)
        for i, it in enumerate(items):
            oi = _order_index(it["id"])
            if it.get("status") == "pending" and oi is not None and mine is not None and oi > mine:
                pos = i
                break
        items.insert(pos, new)
        have.add(new["id"])
    return items


# The interpreter running --queue-insert on evo-x2 (the same Python 3.12 every queue entry there uses), so no user
# path is written into source.
PYTHON_EXE = sys.executable


def queue_entries(results_dir=r"C:\apu\ovn\results") -> list[dict]:
    """The three queue entries, in run order."""
    def cmd(mode, fname):
        return [PYTHON_EXE, "x2_r2_scale.py", "--mode", mode, "--out", f"{results_dir}\\{fname}"]
    common = "H-scale, pre-registered in docs/FINDINGS.md 2026-10-14. Releases itself via tq.advance() on exit; sets " \
             "no pause."
    return [
        {"id": JOB_IDS["probe"], "cmd": cmd("probe", PROBE_FILE), "status": "pending", "projected_duration_h": None,
         "note": f"2-turn timing probe per model x tier (4096/32768/default), projections for the validation and "
                 f"real jobs, llama3.3:70b default-tier decision. {common}"},
        {"id": JOB_IDS["validation"], "cmd": cmd("validation", VALIDATION_FILE), "status": "pending",
         "projected_duration_h": None,
         "note": f"R2 validation (v2 plan, call-2 off) for qwen3:32b, qwen3-30b-a3b-2507, llama3.3:70b (missing until "
                 f"pulled). Hours: see the probe's scale_projection. {common}"},
        {"id": JOB_IDS["real"], "cmd": cmd("real", REAL_FILE), "status": "pending", "projected_duration_h": None,
         "note": f"R2 real 4096/32768 x 3 seeds x 40 turns (+ default per the plan), per-model refusal on "
                 f"{VALIDATION_FILE}. Hours: see the probe's scale_projection. {common}"},
    ]


def queue_insert(log=print) -> list[dict]:
    import t2s_queue as tq
    tq.set_pause("x2_r2_scale --queue-insert: one read-modify-write inserting x2_scale_timing_probe, "
                 "x2_r2_scale_validation, x2_r2_scale_real; released by clear_pause() in this same script's finally "
                 "block within seconds")
    try:
        items = tq.read_queue()
        new = insert_jobs(items, queue_entries())
        tq.write_queue(new)
        ids = [it["id"] for it in new]
        log(json.dumps({jid: ids.index(jid) for jid in JOB_IDS.values()}))
        return new
    finally:
        tq.clear_pause()


# ── job driver ──────────────────────────────────────────────────────────────────────────────────────

def build_arg_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("probe", "validation", "real"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--results-dir", default=None,
                    help="where the reference, probe and validation files are read (default: --out's directory)")
    ap.add_argument("--models", default=",".join(SCALE_MODELS))
    ap.add_argument("--import-check", action="store_true")
    ap.add_argument("--queue-insert", action="store_true")
    return ap


def import_check() -> dict:
    import host_config  # noqa: F401
    import t2s_queue  # noqa: F401
    return {"agent_file": str(Path(ag.__file__).resolve()), "agent_blob": git_blob_sha1(ag.__file__),
            "think": {m: ag.MODELS.get(m) for m in SCALE_MODELS},
            "probe_arms": [arm(t) for t in PROBE_TIERS], "validation_plan": ag.validation_plan(CALL2),
            "real_items_all_default": real_items(list(SCALE_MODELS), {m: True for m in SCALE_MODELS})[:8],
            "queue_entries": [e["cmd"] for e in queue_entries()]}


def main(argv=None, advance=True):
    ap = build_arg_parser()
    args = ap.parse_args(argv)
    if args.import_check:
        print(json.dumps(import_check(), indent=1, default=str))
        return "import-check"
    if args.queue_insert:
        queue_insert()
        return "queue-insert"
    if not args.mode or not args.out:
        ap.error("--mode and --out are required for a job")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rdir = Path(args.results_dir) if args.results_dir else out_path.parent
    models = [m for m in args.models.split(",") if m]

    def log(msg):
        print(f"[{ag.utc_iso()}] {msg}", flush=True)

    def emit(row):
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    note = "completed"
    import host_config as hc
    started = False
    try:
        import socket
        hostname = socket.gethostname()
        try:
            host_cfg = hc.require_host(hostname)
        except SystemExit as e:
            raise RuntimeError(str(e)) from None
        if hostname.upper() != "EVO-X2":
            raise RuntimeError(f"x2_r2_scale: host {hostname!r} is not EVO-X2")
        interactive = hc.enforce_or_record_interactive_session(host_cfg)
        preflight, refused, include_default, projection = None, {}, {}, None
        rules_in_use = None
        if args.mode == "real":
            vrows = ag.read_rows(rdir / VALIDATION_FILE)
            preflight = ag.validation_preflight_per_model([(VALIDATION_FILE, vrows)], CALL2, models)
            refused, rules_in_use = preflight["refused"], preflight["rules_in_use"]
            prows = ag.read_rows(rdir / PROBE_FILE)
            projection = (next((r["projection"] for r in reversed(prows) if r.get("record") == "scale_projection"),
                               None) if prows else None)
            include_default = {m: default_tier_decision(projection, m) for m in models}
        emit({"record": "run_start", "mode": args.mode, "models_requested": models, "call2_mode": CALL2,
              "agent_file": str(Path(ag.__file__).resolve()), "agent_blob": git_blob_sha1(ag.__file__),
              "think": {m: ag.MODELS.get(m) for m in models}, "rules_in_use": rules_in_use,
              "validation_preflight": preflight, "default_tier": include_default, "probe_projection": projection,
              "hw_id": host_cfg.get("hw_id"), "interactive_session": interactive,
              "max_tokens_per_call": ag.MAX_TOKENS_PER_CALL, "keep_alive": ag.KEEP_ALIVE, "ts_utc": ag.utc_iso()})
        present = {VALIDATION_FILE: (rdir / VALIDATION_FILE).exists()}
        fin = {VALIDATION_FILE: any(r.get("record") == "run_end" for r in ag.read_rows(rdir / VALIDATION_FILE))}
        for m, reasons in refused.items():
            emit({"record": "refused_model", "model_id": m, "reasons": reasons, "call2_mode": CALL2,
                  "refusal_category": ag.refusal_category(reasons, present, fin), "ts_utc": ag.utc_iso()})
            log(f"refused model {m}: {'; '.join(reasons)}")
        if preflight is not None and not preflight["ok"]:
            note = ("completed; refused to start, no model passed validation in " + VALIDATION_FILE + ": "
                    + "; ".join(r for rs in refused.values() for r in rs))[:400]
            emit({"record": "refused", "reasons": refused, "ts_utc": ag.utc_iso()})
            emit({"record": "run_end", "note": note, "ts_utc": ag.utc_iso()})
            log(note)
            return note
        ag._start_server(hc)
        started = True
        if not hc.wait_for_ollama_ready(timeout_s=60):
            hc.stop_ollama_server()
            ag._start_server(hc)
            if not hc.wait_for_ollama_ready(timeout_s=90):
                raise RuntimeError("ollama did not become ready")
        runtime = ag.OllamaRuntime()
        avail = runtime.available_models()
        run_models = [m for m in models if m not in refused]
        missing = [m for m in run_models if not any(ag.same_tag(m, a) for a in avail)]
        for m in missing:
            emit({"record": "missing_model", "model_id": m, "available": avail, "ts_utc": ag.utc_iso()})
            log(f"missing model {m}: not in this Ollama store (no model call)")
        run_models = [m for m in run_models if m not in missing]
        emit({"record": "runtime", "ollama_version": runtime.version(), "models_run": run_models,
              "ts_utc": ag.utc_iso()})
        if args.mode == "probe":
            items = [(m, arm(t), PROBE_SEED, PROBE_TURNS) for m in run_models for t in PROBE_TIERS]
            run_items(runtime, items, "probe", out_path, emit, log)
            ref_real = [r for f in REFERENCE_REAL_FILES for r in ag.read_rows(rdir / f)]
            proj = project_all(ag.read_rows(out_path), ref_real, ag.read_rows(rdir / REFERENCE_VALIDATION_FILE))
            emit({"record": "scale_projection", "projection": proj, "ts_utc": ag.utc_iso()})
            report = proj
        elif args.mode == "validation":
            ag.run_plan(runtime, ag.validation_plan(CALL2), run_models, "validation", out_path, log=log, emit=emit)
            rows = ag.read_rows(out_path) + [{"record": "run_end"}]
            report = ag.evaluate_gates(rows, call2_tools=CALL2)
            report["preflight_per_model"] = {m: ag.validation_preflight(rows, CALL2, [m]) for m in run_models}
            failed = {m: p["reasons"] for m, p in report["preflight_per_model"].items() if not p["ok"]}
            note = (f"completed; validation passed: {[m for m in run_models if m not in failed]}; failed: "
                    + "; ".join(r for rs in failed.values() for r in rs)
                    + (f"; missing: {missing}" if missing else ""))[:400]
            log("\n" + ag.format_baseline_markdown(report))
        else:
            inc = {m: include_default[m]["include"] for m in run_models}
            run_items(runtime, real_items(run_models, inc), "real", out_path, emit, log)
            rows = ag.read_rows(out_path)
            sessions = [s for s in ag.completed_sessions(rows) if s["mode"] == "real"]
            report = {"rules_in_use": rules_in_use, "refused_models": refused, "missing": missing,
                      "default_tier": include_default,
                      "kill_criterion": ag.kill_criterion(sessions, rules_in_use, rows),
                      "h_scale": h_scale_verdict(rows + [{"record": "run_end"}],
                                                 ag.read_rows(rdir / VALIDATION_FILE), models)}
            if refused or missing:
                note = (f"completed; refused: {sorted(refused)}; missing: {missing}")[:400]
        Path(str(out_path) + ".report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
        emit({"record": "run_end", "note": note, "ts_utc": ag.utc_iso()})
    except Exception as e:
        import traceback
        note = f"stopped: {e!r}"[:400]
        log(note)
        traceback.print_exc()
    finally:
        if started:
            try:
                hc.stop_ollama_server()
            except Exception as e:
                log(f"ollama stop failed: {e!r}")
        if advance:
            try:
                import t2s_queue as tq
                tq.advance(note)
            except Exception as e:
                log(f"queue advance failed: {e!r}")
    return note


if __name__ == "__main__":
    main()
