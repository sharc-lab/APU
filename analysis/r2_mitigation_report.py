"""R2 mitigation report (x2_r2_mitigation_v1, pre-registered in docs/FINDINGS.md "PRE-REGISTRATION: R2 mitigation
x2_r2_mitigation_v1", 2026-10-08): client-side trimming at num_ctx 4096 and 8192 vs the same tiers, models and seeds in
the real runs (4096: results/x2_r2_real_v1.jsonl; 8192: results/x2_r2_real_v1b.jsonl).

Per tier x model, for the mitigation run and for the real run's cell:
  - rule survival per rule in use: compliance rate over the cell's turns (rule 4 over its length turns), first failure
    turn per seed, sessions in which the rule held for all turns;
  - first failure of any rule in use per seed;
  - canary misses, system-prompt canary C and first-user-message canary H separately (H misses are expected under
    trimming: memory loss, not a rule failure);
  - surfaced errors (turns with a non-200 status, transport error or error field), thinking present, think sent;
mitigation only (observed from the OLLAMA_DEBUG log, x2_r2_client_trim.summarize_calls): context shifts, Ollama
token-level cuts, calls on which Ollama still dropped messages, exceed-context errors, incomplete log slices, client
token counting method, over-budget calls, client exact count vs llama-server's own prompt token count.

Verdict (pre-registered, mechanical; verdict_cell / overall_verdict):
  valid       : all expected seeds completed, 0 context shifts, 0 token cuts, 0 Ollama message drops, 0 exceed-context
                errors, a complete log slice for every call, no thinking, think false sent on every row of a qwen3
                model; otherwise "invalid";
  recovered   : valid and every rule in use >= 90% (ra.BASELINE_THRESHOLD) over the cell's turns;
  not_recovered : valid and some rule in use < 90%.
  Overall (4096 cells): supported if both models are recovered; refuted if either is valid and not recovered;
  otherwise inconclusive. 8192 is computed the same way, as a control.

Usage: py -3.12 analysis/r2_mitigation_report.py [results/x2_r2_mitigation_v1.jsonl]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO / "analysis"))

import x2_r2_agent as ra  # noqa: E402
import x2_r2_client_trim as ct  # noqa: E402
import r2_real_report as rr  # noqa: E402

MITIGATION_FILE = "results/x2_r2_mitigation_v1.jsonl"
REAL_V1_FILE = "results/x2_r2_real_v1.jsonl"
REAL_V1B_FILE = "results/x2_r2_real_v1b.jsonl"
VALIDATION_FILE = "results/x2_r2_validation_v2.jsonl"
COMPARISON_FILE = {"num_ctx_4096": REAL_V1_FILE, "num_ctx_8192": REAL_V1B_FILE}
VERDICT_TIER = "num_ctx_4096"
EXPECTED_SEEDS = tuple(ra.SEEDS)
THRESHOLD = ra.BASELINE_THRESHOLD


def _rows(path: Path) -> list[dict]:
    return ra.read_rows(Path(path))


def _rules_in_use(rows) -> dict:
    start = next((r for r in rows if r.get("record") == "run_start"), {})
    riu = start.get("rules_in_use") or {}
    return riu if isinstance(riu, dict) else {}


def _turns_by_session(rows, sessions):
    keep = {s["attempt_id"]: s for s in sessions}
    out = {}
    for r in rows:
        if r.get("record") == "r2a_turn" and r.get("attempt_id") in keep:
            out.setdefault(r["attempt_id"], []).append(r)
    return [(keep[a], sorted(t, key=lambda r: r["turn_idx"])) for a, t in out.items()]


def cell_metrics(sessions_turns: list[tuple[dict, list[dict]]], rules) -> dict:
    """Rule survival, canaries, errors and thinking over a cell's sessions [(session row, its turn rows)]."""
    st = sorted(sessions_turns, key=lambda x: x[0]["seed"])
    all_turns = [t for _, ts in st for t in ts]
    out = {"seeds": [s["seed"] for s, _ in st], "n_sessions": len(st), "n_turns": len(all_turns), "rules": {}}
    for rule in rules:
        vals = [t.get(rule) for t in all_turns if t.get(rule) is not None]
        n_ok = sum(1 for v in vals if v)
        firsts = [next((t["turn_idx"] for t in ts if t.get(rule) is False), None) for _, ts in st]
        out["rules"][rule] = {"rate": n_ok / len(vals) if vals else None, "n": len(vals), "n_ok": n_ok,
                              "first_fail_per_seed": firsts,
                              "sessions_held_all_turns": sum(1 for f in firsts if f is None)}
    out["first_rule_failure_per_seed"] = [
        min((t["turn_idx"] for t in ts if any(t.get(r) is False for r in rules)), default=None) for _, ts in st]
    can = [t for t in all_turns if t.get("canary_check")]
    out["canary_checks"] = len(can)
    out["canary_sys_misses"] = sum(1 for t in can if t.get("canary_sys_ok") is False)
    out["canary_hist_misses"] = sum(1 for t in can if t.get("canary_hist_ok") is False)
    out["error_turns"] = sum(1 for t in all_turns if t.get("any_http_error"))
    out["thinking_turns"] = sum(1 for t in all_turns if t.get("thinking_present") is True)
    out["think_sent"] = sorted({str(t.get("think")) for t in all_turns})
    out["think_false_rows"] = sum(1 for t in all_turns if t.get("think") is False)
    out["rules_all_at_threshold"] = (all(v["rate"] is not None and v["rate"] >= THRESHOLD
                                         for v in out["rules"].values()) if out["rules"] else None)
    return out


def observed_metrics(rows, sessions_turns) -> dict:
    """Mitigation only: x2_r2_client_trim.summarize_calls over the cell's calls, plus r2m_call rows present per call."""
    recs = [c.get("client_trim") for _, ts in sessions_turns for t in ts for c in t.get("calls") or []]
    n_calls = len(recs)
    recs = [r for r in recs if r is not None]
    o = ct.summarize_calls(recs)
    o["calls_without_client_trim"] = n_calls - len(recs)
    keys = {(s["model_id"], s["arm_id"], s["seed"]) for s, _ in sessions_turns}
    o["r2m_call_rows"] = sum(1 for r in rows if r.get("record") == "r2m_call" and r.get("mode") == "mitigation"
                             and (r.get("model_id"), r.get("arm_id"), r.get("seed")) in keys)
    o["calls_total"] = n_calls
    return o


def verdict_cell(model: str, m: dict, obs: dict, expected_seeds=EXPECTED_SEEDS) -> dict:
    reasons = []
    if sorted(m["seeds"]) != sorted(expected_seeds):
        reasons.append(f"sessions completed for seeds {m['seeds']}, expected {list(expected_seeds)}")
    for key, label in (("context_shift_events", "context shifts"), ("token_cut_calls", "token-level cuts"),
                       ("ollama_message_drop_calls", "calls with Ollama message drops"),
                       ("exceed_context_error_calls", "exceed-context errors"),
                       ("log_slice_incomplete_calls", "incomplete log slices"),
                       ("calls_without_client_trim", "calls without a client-trim record")):
        if obs.get(key):
            reasons.append(f"{obs[key]} {label}")
    if obs.get("observed_calls", 0) < obs.get("calls_total", 0):
        reasons.append(f"{obs['calls_total'] - obs.get('observed_calls', 0)} calls without an observed log record")
    if obs.get("r2m_call_rows", 0) < obs.get("calls_total", 0):
        reasons.append(f"r2m_call rows {obs.get('r2m_call_rows')} < calls {obs.get('calls_total')}")
    if m["thinking_turns"]:
        reasons.append(f"thinking present on {m['thinking_turns']} turns")
    if model.startswith("qwen3") and m["think_false_rows"] != m["n_turns"]:
        reasons.append(f"think false sent on {m['think_false_rows']}/{m['n_turns']} rows")
    if reasons:
        return {"verdict": "invalid", "valid": False, "reasons": reasons}
    failing = [r for r, v in m["rules"].items() if v["rate"] is None or v["rate"] < THRESHOLD]
    return {"verdict": "not_recovered" if failing else "recovered", "valid": True, "reasons": [],
            "rules_below_threshold": failing}


def overall_verdict(cells: dict, tier=VERDICT_TIER, models=ra.DEFAULT_MODELS) -> str:
    vs = {m: (cells.get(f"{tier}|{m}") or {}).get("verdict", {}).get("verdict") for m in models}
    if any(v == "not_recovered" for v in vs.values()):
        return "refuted"
    if all(v == "recovered" for v in vs.values()):
        return "supported"
    return "inconclusive"


def _real_cells(tier: str, path: Path):
    """{model: [(session, turns)]} for the real run's completed v1-protocol sessions at `tier` (mode real)."""
    rows = _rows(path)
    sessions = [s for s in ra.completed_sessions(rows) if s.get("mode") == "real"
                and rr.call2_mode_of(s) == rr.POOLED_CALL2_MODE and rr.tier_of(s["arm_id"]) == tier]
    by = {}
    for s, ts in _turns_by_session(rows, sessions):
        by.setdefault(s["model_id"], []).append((s, ts))
    return by, _rules_in_use(rows)


def build(mitigation_path=None, real_paths=None, validation_path=None, tiers=None) -> dict:
    """real_paths: {tier: path} (default COMPARISON_FILE under REPO); a missing file makes that comparison
    "not available". tiers: restrict to these tiers (default every tier in the mitigation file)."""
    mpath = Path(mitigation_path or REPO / MITIGATION_FILE)
    rows = _rows(mpath)
    if not rows:
        raise FileNotFoundError(mpath)
    riu = _rules_in_use(rows)
    sessions = [s for s in ra.completed_sessions(rows) if s.get("mode") == "mitigation"]
    cells_raw = {}
    for s, ts in _turns_by_session(rows, sessions):
        cells_raw.setdefault((rr.tier_of(s["arm_id"]), s["model_id"]), []).append((s, ts))
    real_paths = real_paths or {t: REPO / f for t, f in COMPARISON_FILE.items()}
    vpath = Path(validation_path or REPO / VALIDATION_FILE)
    baseline = rr._baseline_tables([vpath]) if vpath.exists() else {}
    real_cache = {}
    cells = {}
    order = {t: i for i, t in enumerate(rr.TIER_ORDER)}
    for (tier, model), st in sorted(cells_raw.items(), key=lambda kv: (order.get(kv[0][0], 99), kv[0][1])):
        if tiers and tier not in tiers:
            continue
        rules = list(riu.get(model) or ra.RULE_IDS)
        m = cell_metrics(st, rules)
        obs = observed_metrics(rows, st)
        cell = {"rules_in_use": rules, "mitigation": m, "observed": obs, "verdict": verdict_cell(model, m, obs)}
        b = baseline.get(model)
        cell["baseline"] = ({r: {"rate": b["rules"][r]["rate"], "n": b["rules"][r]["n"]} for r in rules}
                            if b else None)
        rp = real_paths.get(tier)
        if rp is None or not Path(rp).exists():
            cell["real"] = {"available": False, "file": None if rp is None else Path(rp).name}
        else:
            if tier not in real_cache:
                real_cache[tier] = _real_cells(tier, Path(rp))
            by, real_riu = real_cache[tier]
            seeds = set(m["seeds"])
            rst = [(s, ts) for s, ts in by.get(model, []) if s["seed"] in seeds]
            real_rules = list(real_riu.get(model) or rules)
            rm = cell_metrics(rst, rules)
            cell["real"] = {"available": True, "file": Path(rp).name, "metrics": rm,
                            "rules_in_use_same": real_rules == rules}
        cells[f"{tier}|{model}"] = cell
    return {"cells": cells, "overall_verdict": overall_verdict(cells), "verdict_tier": VERDICT_TIER,
            "threshold": THRESHOLD, "rules_in_use": riu, "file": mpath.name,
            "n_sessions": sum(c["mitigation"]["n_sessions"] for c in cells.values()),
            "n_turns": sum(c["mitigation"]["n_turns"] for c in cells.values())}


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _f(vals):
    return "/".join("-" if v is None else str(v) for v in vals)


def rule_summary(m: dict) -> str:
    return ", ".join(f"{r.split('_')[0]} {_pct(v['rate'])} (n={v['n']}, first fail {_f(v['first_fail_per_seed'])})"
                     for r, v in m["rules"].items())


def format_markdown(rep: dict) -> str:
    out = [f"Overall (pre-registered, {rep['verdict_tier']}): {rep['overall_verdict']}", "",
           "| tier | model | run | seeds | rule survival per rule in use | first rule failure | canary C / H misses | "
           "errors | context shifts | token cuts | Ollama drops | verdict |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, c in rep["cells"].items():
        tier, model = key.split("|")
        m, o = c["mitigation"], c["observed"]
        out.append(f"| {tier} | {model} | mitigation | {_f(m['seeds'])} | {rule_summary(m)} | "
                   f"{_f(m['first_rule_failure_per_seed'])} | {m['canary_sys_misses']} / {m['canary_hist_misses']} of "
                   f"{m['canary_checks']} | {m['error_turns']} | {o['context_shift_events']} | {o['token_cut_calls']} | "
                   f"{o['ollama_message_drop_calls']} | {c['verdict']['verdict']} |")
        if c["real"]["available"]:
            rm = c["real"]["metrics"]
            out.append(f"| {tier} | {model} | real ({c['real']['file']}) | {_f(rm['seeds'])} | {rule_summary(rm)} | "
                       f"{_f(rm['first_rule_failure_per_seed'])} | {rm['canary_sys_misses']} / "
                       f"{rm['canary_hist_misses']} of {rm['canary_checks']} | {rm['error_turns']} | not observed | "
                       f"not observed | not observed | - |")
        else:
            out.append(f"| {tier} | {model} | real | not available ({c['real']['file']} not synced) | | | | | | | | |")
    return "\n".join(out)


if __name__ == "__main__":
    a = sys.argv[1:]
    rep = build(REPO / a[0] if a else None)
    print(format_markdown(rep))
    print(json.dumps({k: {"verdict": c["verdict"], "observed": c["observed"]} for k, c in rep["cells"].items()},
                     indent=1, default=str)[:6000])
