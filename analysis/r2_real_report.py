"""R2 real run report (x2_r2_real_v1): per context tier and model, the turn of first canary miss, of first failure
per baseline-passing rule, and of first invalid tool call; whether any error surfaced before them; survival
curves; and the pre-registered kill criterion (harness/x2_r2_agent.kill_criterion).

Rules counted are the run's own `rules_in_use` from its run_start record (the validation-v2 gate-passing rules per
model), never all five. "First canary miss" is any canary-check turn where either canary was missed, whether or not
the final call's prompt exceeded the loaded window; the over-window flag is reported next to it.

Usage: py -3.12 analysis/r2_real_report.py [results/x2_r2_real_v1.jsonl]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))

import x2_r2_agent as ra  # noqa: E402

DEFAULT_FILE = "results/x2_r2_real_v1.jsonl"
CHECKPOINTS = (5, 10, 20, 30, 40)
TIER = {"ollama_default_call2_notools": "ollama_default", "ollama_ctx_32768_call2_notools": "num_ctx_32768",
        "ollama_ctx_4096_call2_notools": "num_ctx_4096"}


def load(path: Path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    start = next(r for r in rows if r.get("record") == "run_start")
    return rows, start["rules_in_use"]


def _first(rows, pred):
    return next((r["turn_idx"] for r in rows if pred(r)), None)


def session_events(turns: list[dict], rules: list[str]) -> dict:
    turns = sorted(turns, key=lambda r: r["turn_idx"])
    miss = lambda r: r.get("canary_check") and (r.get("canary_sys_ok") is False or r.get("canary_hist_ok") is False)
    ev = {"first_canary_miss": _first(turns, miss),
          "first_invalid_tool_call": _first(turns, lambda r: r.get("tool_validity_all") is False),
          "first_error": _first(turns, lambda r: r.get("any_http_error")),
          "first_over_window": _first(turns, lambda r: r.get("over_loaded_window"))}
    m = next((r for r in turns if miss(r)), None)
    ev["first_canary_miss_over_window"] = None if m is None else bool(m.get("over_loaded_window"))
    for rule in rules:
        ev[f"first_fail_{rule}"] = _first(turns, lambda r, rule=rule: r.get(rule) is False)
    any_fail = [v for k, v in ev.items() if k.startswith("first_fail_") and v is not None]
    for k in ("first_invalid_tool_call", "first_canary_miss"):
        if ev[k] is not None:
            any_fail.append(ev[k])
    ev["first_any_failure"] = min(any_fail) if any_fail else None
    ev["error_before_first_failure"] = (None if ev["first_any_failure"] is None
                                       else ev["first_error"] is not None and ev["first_error"] < ev["first_any_failure"])
    ev["n_turns"] = len(turns)
    return ev


VALIDATION_FILE = "results/x2_r2_validation_v2.jsonl"
_METRIC_FIELD = {"tool_validity": "tool_validity_all", "tool_args": "tool_args_all", "recall": "recall_ok"}


def gated_metrics(rules_in_use: dict, validation_path: Path = REPO / VALIDATION_FILE) -> dict:
    """Per model: the rules in use plus each non-rule metric (tool validity, tool args, recall) whose validation-v2
    baseline reached the 90% gate. The pre-registered kill criterion counts every non-rule metric regardless of its
    baseline; this stricter variant only counts what the model could reliably do with no context pressure at all."""
    v2 = ra.read_rows(validation_path)
    bt = ra.evaluate_gates(v2, call2_tools=False)["baseline_table"]
    out = {}
    for model, rules in rules_in_use.items():
        ok = [_METRIC_FIELD[m] for m in _METRIC_FIELD if ((bt[model].get(m) or {}).get("rate") or 0) >= ra.BASELINE_THRESHOLD]
        out[model] = {"fields": list(rules) + ok,
                      "excluded": [m for m in _METRIC_FIELD if _METRIC_FIELD[m] not in ok]}
    return out


def gated_session(turns: list[dict], fields: list[str]) -> dict:
    turns = sorted(turns, key=lambda r: r["turn_idx"])
    first_fail = _first(turns, lambda r: any(r.get(f) is False for f in fields))
    first_err = _first(turns, lambda r: r.get("any_http_error"))
    first_over = _first(turns, lambda r: r.get("over_loaded_window"))
    silent = first_fail is not None and not (first_err is not None and first_err < first_fail)
    return {"first_gated_failure": first_fail, "silent": silent,
            "after_window_exceeded": silent and first_over is not None and first_fail >= first_over}


def build(path: Path = REPO / DEFAULT_FILE) -> dict:
    rows, rules_in_use = load(path)
    gm = gated_metrics(rules_in_use)
    turns = [r for r in rows if r.get("record") == "r2a_turn"]
    sessions = ra.completed_sessions(rows)
    cells = {}
    for s in sessions:
        st = [t for t in turns if t.get("attempt_id") == s["attempt_id"]]
        ev = session_events(st, rules_in_use[s["model_id"]])
        ev.update(gated_session(st, gm[s["model_id"]]["fields"]))
        ev["seed"] = s["seed"]
        cells.setdefault((TIER.get(s["arm_id"], s["arm_id"]), s["model_id"]), []).append(ev)
    table = {}
    for (tier, model), evs in sorted(cells.items()):
        evs.sort(key=lambda e: e["seed"])
        # Intact = no gated failure (rules in use + 90%-baseline metrics) and no canary miss, up to and including t.
        def end(e):
            v = [x for x in (e["first_gated_failure"], e["first_canary_miss"]) if x is not None]
            return min(v) if v else None
        surv = {t: sum(1 for e in evs if end(e) is None or end(e) > t) for t in CHECKPOINTS}
        table[f"{tier}|{model}"] = {"sessions": evs, "n": len(evs), "survival": surv,
                                   "rules_in_use": rules_in_use[model]}
    kill = ra.kill_criterion(sessions, rules_in_use, turn_rows=rows)
    gated = {}
    for key, c in table.items():
        tier = key.split("|")[0]
        g = gated.setdefault(tier, {"n_sessions": 0, "silent": 0, "silent_after_window_exceeded": 0})
        g["n_sessions"] += c["n"]
        g["silent"] += sum(e["silent"] for e in c["sessions"])
        g["silent_after_window_exceeded"] += sum(e["after_window_exceeded"] for e in c["sessions"])
    return {"table": table, "kill": kill, "gated_kill": gated, "gated_metrics": gm,
            "n_sessions": len(sessions), "n_turn_rows": len(turns)}


def _fmt(vals):
    return "/".join("-" if v is None else str(v) for v in vals)


def format_markdown(rep: dict) -> str:
    out = ["| tier | model | first canary miss | " + "first rule failure (per rule) | first invalid tool call | "
           "first surfaced error | sessions intact at turn 5/10/20/30/40 |", "|---|---|---|---|---|---|---|"]
    for key, c in rep["table"].items():
        tier, model = key.split("|")
        s = c["sessions"]
        rules = "; ".join(f"{r.split('_')[0]} {_fmt(e[f'first_fail_{r}'] for e in s)}" for r in c["rules_in_use"])
        out.append(f"| {tier} | {model} | {_fmt(e['first_canary_miss'] for e in s)} | {rules} | "
                   f"{_fmt(e['first_invalid_tool_call'] for e in s)} | "
                   f"{_fmt(e['first_error'] for e in s)} | "
                   f"{'/'.join(str(c['survival'][t]) for t in CHECKPOINTS)} of {c['n']} |")
    return "\n".join(out)


if __name__ == "__main__":
    rep = build(REPO / (sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FILE))
    print(format_markdown(rep))
    print(json.dumps(rep["kill"], indent=1, default=str)[:3000])
