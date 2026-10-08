"""R2 real run report: per context tier and model, the turn the prompt first exceeded the loaded window, of first
canary miss, of first failure per baseline-passing rule, and of first invalid tool call; whether any error surfaced
before each of them; survival curves; and the pre-registered kill criterion (harness/x2_r2_agent.kill_criterion)
plus its gated variant.

Two entry points:
  build(path)            : one file, the v1 report (results/x2_r2_real_v1.jsonl, checkpoints 5/10/20/30/40). The
                           register rows R2-real-v1-* use this and their values do not change.
  build_combined(paths)  : the strengthened run (2026-10-08): results/x2_r2_real_v1.jsonl plus
                           results/x2_r2_real_v1b.jsonl, five tiers (Ollama default, num_ctx 32768/16384/8192/4096),
                           four models, up to five seeds per cell, checkpoints 5/10/15/20/25/30/35/40. Sessions are
                           keyed by (model, arm, seed); a key completed in an earlier file wins over a later one
                           (the v1b run never reruns a v1 cell, so a duplicate is reported, not silently pooled).
                           Only sessions of one call-2 mode are pooled (v1 protocol, "off"); any other is reported
                           as excluded.

Rules counted are each file's own `rules_in_use` from its run_start record (the validation gate-passing rules per
model: x2_r2_validation_v2.jsonl for llama3.1:8b and qwen3:14b, x2_r2_validation_v2b.jsonl for qwen3-4b-2507 and
qwen3:8b), never all five. "First canary miss" is any canary-check turn where either canary was missed, whether or
not the final call's prompt exceeded the loaded window; the over-window flag is reported next to it.

Usage: py -3.12 analysis/r2_real_report.py [results/x2_r2_real_v1.jsonl [results/x2_r2_real_v1b.jsonl ...]]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))

import x2_r2_agent as ra  # noqa: E402

DEFAULT_FILE = "results/x2_r2_real_v1.jsonl"
V1B_FILE = "results/x2_r2_real_v1b.jsonl"
COMBINED_FILES = (DEFAULT_FILE, V1B_FILE)
CHECKPOINTS = (5, 10, 20, 30, 40)                     # v1 report (register rows R2-real-v1-*)
CHECKPOINTS_FULL = (5, 10, 15, 20, 25, 30, 35, 40)    # strengthened report
TIER = {"ollama_default_call2_notools": "ollama_default", "ollama_ctx_32768_call2_notools": "num_ctx_32768",
        "ollama_ctx_4096_call2_notools": "num_ctx_4096"}
TIER_ORDER = ("ollama_default", "num_ctx_32768", "num_ctx_16384", "num_ctx_8192", "num_ctx_4096")
POOLED_CALL2_MODE = "off"


def tier_of(arm: str) -> str:
    """Arm id -> tier label: the call-2 mode suffix removed, num_ctx arms as num_ctx_<n>."""
    if arm in TIER:
        return TIER[arm]
    base = arm
    for sfx in sorted((s for s in ra.MODE_SUFFIX.values() if s), key=len, reverse=True):
        if base.endswith(sfx):
            base = base[: -len(sfx)]
            break
    return base.replace("ollama_ctx_", "num_ctx_")


def call2_mode_of(row: dict) -> str:
    """Rows written before call2_mode existed (x2_r2_real_v1) carry call2_tools only: False is mode "off"."""
    m = row.get("call2_mode")
    if m:
        return m
    return "on" if row.get("call2_tools", True) else "off"


def load(path: Path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    start = next(r for r in rows if r.get("record") == "run_start")
    return rows, start["rules_in_use"]


def _first(rows, pred):
    return next((r["turn_idx"] for r in rows if pred(r)), None)


_EVENT_KEYS_FOR_ERROR_CHECK = ("first_over_window", "first_canary_miss", "first_invalid_tool_call")


def session_events(turns: list[dict], rules: list[str]) -> dict:
    turns = sorted(turns, key=lambda r: r["turn_idx"])
    miss = lambda r: r.get("canary_check") and (r.get("canary_sys_ok") is False or r.get("canary_hist_ok") is False)
    ev = {"first_canary_miss": _first(turns, miss),
          "first_invalid_tool_call": _first(turns, lambda r: r.get("tool_validity_all") is False),
          "first_error": _first(turns, lambda r: r.get("any_http_error")),
          "first_over_window": _first(turns, lambda r: r.get("over_loaded_window")),
          "first_thinking": _first(turns, lambda r: r.get("thinking_present") is True)}
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
    # per event: did an error (non-200, transport error, error field) surface on an earlier turn? None = no event
    for k in _EVENT_KEYS_FOR_ERROR_CHECK + tuple(f"first_fail_{r}" for r in rules):
        v = ev[k]
        ev[f"error_before_{k[len('first_'):]}"] = (None if v is None
                                                   else ev["first_error"] is not None and ev["first_error"] < v)
    ev["n_turns"] = len(turns)
    return ev


VALIDATION_FILE = "results/x2_r2_validation_v2.jsonl"
VALIDATION_FILES = ("results/x2_r2_validation_v2.jsonl", "results/x2_r2_validation_v2b.jsonl")
_METRIC_FIELD = {"tool_validity": "tool_validity_all", "tool_args": "tool_args_all", "recall": "recall_ok"}


def _baseline_tables(validation_paths) -> dict:
    """{model: baseline table entry} from the arm-b (call-2 mode "off") validation files; a model in several files
    takes the last one that has it (the per-model preflight's rule)."""
    out = {}
    for p in validation_paths:
        p = Path(p)
        if not p.exists():
            continue
        bt = ra.evaluate_gates(ra.read_rows(p), call2_tools=False)["baseline_table"]
        out.update(bt)
    return out


def gated_metrics(rules_in_use: dict, validation_path: Path = REPO / VALIDATION_FILE, validation_paths=None) -> dict:
    """Per model: the rules in use plus each non-rule metric (tool validity, tool args, recall) whose validation
    baseline reached the 90% gate. The pre-registered kill criterion counts every non-rule metric regardless of its
    baseline; this stricter variant only counts what the model could reliably do with no context pressure at all."""
    bt = _baseline_tables(validation_paths if validation_paths is not None else [validation_path])
    out = {}
    for model, rules in rules_in_use.items():
        entry = bt.get(model, {})
        ok = [_METRIC_FIELD[m] for m in _METRIC_FIELD
              if ((entry.get(m) or {}).get("rate") or 0) >= ra.BASELINE_THRESHOLD]
        out[model] = {"fields": list(rules) + ok,
                      "excluded": [m for m in _METRIC_FIELD if _METRIC_FIELD[m] not in ok],
                      "baseline_found": model in bt}
    return out


def gated_session(turns: list[dict], fields: list[str]) -> dict:
    turns = sorted(turns, key=lambda r: r["turn_idx"])
    first_fail = _first(turns, lambda r: any(r.get(f) is False for f in fields))
    first_err = _first(turns, lambda r: r.get("any_http_error"))
    first_over = _first(turns, lambda r: r.get("over_loaded_window"))
    silent = first_fail is not None and not (first_err is not None and first_err < first_fail)
    return {"first_gated_failure": first_fail, "silent": silent,
            "after_window_exceeded": silent and first_over is not None and first_fail >= first_over}


def _intact_end(e):
    v = [x for x in (e["first_gated_failure"], e["first_canary_miss"]) if x is not None]
    return min(v) if v else None


def survival(evs: list[dict], checkpoints) -> dict:
    """Sessions intact at each checkpoint t: no gated failure (rules in use + 90%-baseline metrics) and no canary
    miss up to and including t."""
    return {t: sum(1 for e in evs if _intact_end(e) is None or _intact_end(e) > t) for t in checkpoints}


def _collect(paths):
    """Rows of every file, the merged per-model rules in use, and the deduplicated completed sessions."""
    all_rows, rules_in_use, seen, sessions, dups, excluded, conflicts = [], {}, {}, [], [], [], []
    for p in paths:
        rows, riu = load(Path(p))
        for m, rules in (riu or {}).items():
            if m in rules_in_use and list(rules_in_use[m]) != list(rules):
                conflicts.append({"model": m, "kept": rules_in_use[m], "other": rules, "file": Path(p).name})
                continue
            rules_in_use.setdefault(m, list(rules))
        for s in ra.completed_sessions(rows):
            if s.get("mode") != "real":
                continue
            key = (s["model_id"], s["arm_id"], s["seed"])
            if call2_mode_of(s) != POOLED_CALL2_MODE:
                excluded.append({"key": list(key), "call2_mode": call2_mode_of(s), "file": Path(p).name})
                continue
            if key in seen:
                dups.append({"key": list(key), "kept_from": seen[key], "duplicate_in": Path(p).name})
                continue
            seen[key] = Path(p).name
            sessions.append(s)
        all_rows.extend(rows)
    return all_rows, rules_in_use, sessions, {"duplicates": dups, "excluded_other_call2_mode": excluded,
                                              "rules_in_use_conflicts": conflicts}


def build_combined(paths=None, checkpoints=CHECKPOINTS_FULL, validation_paths=None, tier_order=True) -> dict:
    paths = [Path(p) for p in (paths or [REPO / f for f in COMBINED_FILES])]
    validation_paths = validation_paths or [REPO / f for f in VALIDATION_FILES]
    rows, rules_in_use, sessions, issues = _collect(paths)
    gm = gated_metrics(rules_in_use, validation_paths=validation_paths)
    keep = {s["attempt_id"] for s in sessions}
    turns_by_attempt = {}
    for r in rows:
        if r.get("record") == "r2a_turn" and r.get("attempt_id") in keep:
            turns_by_attempt.setdefault(r["attempt_id"], []).append(r)
    cells = {}
    for s in sessions:
        st = turns_by_attempt.get(s["attempt_id"], [])
        rules = rules_in_use.get(s["model_id"], list(ra.RULE_IDS))
        ev = session_events(st, rules)
        ev.update(gated_session(st, gm.get(s["model_id"], {"fields": rules})["fields"]))
        ev["seed"] = s["seed"]
        cells.setdefault((tier_of(s["arm_id"]), s["model_id"]), []).append(ev)
    order = {t: i for i, t in enumerate(TIER_ORDER)} if tier_order else {}
    table = {}
    for (tier, model), evs in sorted(cells.items(), key=lambda kv: (order.get(kv[0][0], 99), kv[0][0], kv[0][1])):
        evs.sort(key=lambda e: e["seed"])
        rules = rules_in_use.get(model, list(ra.RULE_IDS))
        table[f"{tier}|{model}"] = {
            "sessions": evs, "n": len(evs), "seeds": [e["seed"] for e in evs],
            "survival": survival(evs, checkpoints), "rules_in_use": rules,
            "any_error_before_any_event": any(
                e.get(k) for e in evs for k in e if k.startswith("error_before_")),
            "thinking_turn_sessions": sum(1 for e in evs if e.get("first_thinking") is not None)}
    turn_rows = [r for a in turns_by_attempt.values() for r in a]
    kill = ra.kill_criterion(sessions, rules_in_use, turn_rows=turn_rows)
    gated = {}
    for key, c in table.items():
        tier = key.split("|")[0]
        g = gated.setdefault(tier, {"n_sessions": 0, "silent": 0, "silent_after_window_exceeded": 0})
        g["n_sessions"] += c["n"]
        g["silent"] += sum(e["silent"] for e in c["sessions"])
        g["silent_after_window_exceeded"] += sum(e["after_window_exceeded"] for e in c["sessions"])
    # the gated kill verdict per the pre-registered rule's shape: killed only if no tier has a silent gated failure
    gated_killed = all(g["silent"] == 0 for g in gated.values()) if gated else None
    return {"table": table, "kill": kill, "gated_kill": gated, "gated_killed": gated_killed, "gated_metrics": gm,
            "n_sessions": len(sessions), "n_turn_rows": len(turn_rows), "rules_in_use": rules_in_use,
            "files": [p.name for p in paths], "checkpoints": list(checkpoints), "issues": issues}


def build(path: Path = REPO / DEFAULT_FILE) -> dict:
    """The v1 report on one file (register rows R2-real-v1-*), checkpoints 5/10/20/30/40."""
    return build_combined([path], checkpoints=CHECKPOINTS, validation_paths=[REPO / VALIDATION_FILE],
                          tier_order=False)


def _fmt(vals):
    return "/".join("-" if v is None else str(v) for v in vals)


def _yn(vals):
    return "/".join("-" if v is None else ("y" if v else "n") for v in vals)


def format_markdown(rep: dict) -> str:
    cps = rep.get("checkpoints") or list(CHECKPOINTS)
    out = ["| tier | model | seeds | first over window | first canary miss | first rule failure (per rule) | "
           "first invalid tool call | first surfaced error | error before first failure | sessions intact at turn "
           + "/".join(str(t) for t in cps) + " |", "|---|---|---|---|---|---|---|---|---|---|"]
    for key, c in rep["table"].items():
        tier, model = key.split("|")
        s = c["sessions"]
        rules = "; ".join(f"{r.split('_')[0]} {_fmt(e[f'first_fail_{r}'] for e in s)}" for r in c["rules_in_use"])
        out.append(f"| {tier} | {model} | {c['n']} | {_fmt(e['first_over_window'] for e in s)} | "
                   f"{_fmt(e['first_canary_miss'] for e in s)} | {rules} | "
                   f"{_fmt(e['first_invalid_tool_call'] for e in s)} | {_fmt(e['first_error'] for e in s)} | "
                   f"{_yn(e['error_before_first_failure'] for e in s)} | "
                   f"{'/'.join(str(c['survival'][t]) for t in cps)} of {c['n']} |")
    return "\n".join(out)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) <= 1:
        rep = build(REPO / (args[0] if args else DEFAULT_FILE))
    else:
        rep = build_combined([REPO / a for a in args])
    print(format_markdown(rep))
    print(json.dumps({"kill": rep["kill"], "gated_kill": rep["gated_kill"], "issues": rep["issues"]},
                     indent=1, default=str)[:4000])
