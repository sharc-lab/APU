"""R2: what the model never saw, and how rule-following changes once the system prompt is gone (operator plan v2,
2026-10-13, Part A2).

Sources (committed files only):
  observed   results/x2_r2_mechanism.jsonl (evo-x2) and results/blade_r2_mechanism_4096_v2.jsonl (Blade): Ollama debug
             log + render per call, llama3.1:8b, ONE session per tier (seed 20260901). r2m_call rows give the message
             level trim (message_level_truncation, from the render/log analysis), the Ollama token-level cut
             (token_cut: limit, prompt, keep, new) and the llama.cpp context shifts (context_shifts: n_keep,
             n_discard). x2_r2_real_v1b has no mechanism logging, so the observed parts stay one session per tier.
  predicted  results/x2_r2_real_v1.jsonl + results/x2_r2_real_v1b.jsonl (evo-x2, every model x tier x seed): no server
             log. Context shifts are PREDICTED with analysis/r2_shift_predictor.py (P + C > n_ctx, register row
             R2-shift-predictor-confusion; P = prompt_eval_count_info_only, C = completion_tokens, n_ctx =
             loaded_context) and token cuts are INFERRED from the cut signature (P == n_ctx - (n_ctx - 4) // 2).
             Blade has no R2 real data (results/blade_r2_real_v1.jsonl is a refused run with no turns).

Per call:
  full     tokens of the whole transcript the client sent, in the model's tokenizer, ESTIMATED: the call's
           prompt_tokens_est (chars/4) x the model's tokens-per-estimate ratio at that transcript size. The ratio is
           the median P / prompt_tokens_est of the 15 nearest (by prompt_tokens_est) untrimmed calls of the same
           model in all input files, leaving out the call's own session (untrimmed: a mechanism call with no message
           trim and no token cut; a real call with P < n_ctx / 2, which no trimmed prompt reaches: Ollama trims only
           until the prompt fits, and a token cut leaves n_ctx / 2 + 2). The same leave-session-out estimate on the
           untrimmed calls themselves is the estimator check (est_check).
  transcript share never seen
           observed: 0 if the call was not message-trimmed, else 1 - P_msg / full, P_msg = the prompt after the
           message trim (token_cut.prompt when a token cut followed, else llama.cpp's new-prompt n_tokens).
           predicted: 0 when P < n_ctx / 2, else 1 - P / full (message drops and any token cut together).
  system-prompt share never seen
           Ollama's message trim always keeps the system message (system_kept on every trimmed mechanism call), so it
           removes none. A token cut keeps the first `keep` (4) prompt tokens and drops the next prompt - new; a
           context shift keeps the first n_keep tokens and discards the next n_discard. The system prompt's own text
           sits right after the chat-template header, so the share lost is the overlap of the dropped span with
           [h, h + S), over S. S = the session's system prompt (harness/x2_r2_agent.build_agent_session(seed)) in
           chars/4 x the session's token_calib_ratio (estimate). h = header tokens before the text: llama3.1 5 in
           llama.cpp's view (BOS 128000, 128006, 9125, 128007 as in the prompt checks' render_ids_head, then the
           "\\n\\n" token) and 4 in Ollama's token-cut view (no BOS); qwen3 3 (<|im_start|>, "system", "\\n"; assumed,
           no qwen render is logged). A shift drops the system prompt for the rest of that call's generation; a token
           cut drops it for the whole call. Turn level = the largest share over the turn's calls.
Per session (rules in use = each model's validation-gated rules from the real runs' run_start, never all five; all
five rule types are still counted and marked in use or not):
  L        first turn with system-prompt share > 0 (observed or predicted);
  violation turn  any rule in use False on that turn;
  before / after  violation rate on turns < L vs turns >= L;
  lag      first violation turn >= L minus L (0 = same turn);
  by rule  False counts before / after L per rule type: format = rule1_json_keys, required audit action =
           rule2_log_event_called, banned content = rule3_no_zebra, units = rule4_metres, session code =
           rule5_session_code.

Usage: py -3.12 analysis/r2_info_loss.py
"""
from __future__ import annotations

import contextlib
import io
import json
import statistics
import sys
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
import r2_shift_predictor as sp  # noqa: E402

MECH_FILES = (("evo-x2", "results/x2_r2_mechanism.jsonl"), ("Blade", "results/blade_r2_mechanism_4096_v2.jsonl"))
REAL_FILES = ("results/x2_r2_real_v1.jsonl", "results/x2_r2_real_v1b.jsonl")
REAL_MACHINE = "evo-x2"
TIER_ORDER = ("ollama_default", "num_ctx_32768", "num_ctx_16384", "num_ctx_8192", "num_ctx_4096")
CHECKPOINTS = (5, 10, 20, 30, 40)
RULES = ("rule1_json_keys", "rule2_log_event_called", "rule3_no_zebra", "rule4_metres", "rule5_session_code")
RULE_TYPE = {"rule1_json_keys": "format", "rule2_log_event_called": "audit action", "rule3_no_zebra": "banned content",
             "rule4_metres": "units", "rule5_session_code": "session code"}
KNN = 15
OBSERVED_LABEL = ("observed, one session per tier (mechanism runs, Ollama debug log + render, llama3.1:8b; "
                  "x2_r2_real_v1b has no mechanism logging, so nothing is added to the observed cells)")
PREDICTED_LABEL = ("predicted (shift predictor P + C > n_ctx, R2-shift-predictor-confusion; token cuts inferred from "
                   "the cut signature; transcript share estimated)")


def tier_of(arm: str) -> str:
    return sp.tier_of(arm)


def header_tokens(model: str, view: str) -> int:
    """Chat-template tokens before the system prompt text. view 'llama.cpp' (context shift) or 'ollama' (token cut)."""
    if model.startswith("llama3.1"):
        return 5 if view == "llama.cpp" else 4
    return 3


def overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def system_share_lost(model: str, S: int, shifts=(), cut=None) -> float:
    """Share of the S system-prompt tokens inside the span a token cut (keep, dropped) or context shifts
    [(n_keep, n_discard), ...] remove. Consecutive shifts in one call are capped at S (the first already covers the
    whole system prompt whenever n_discard >= S)."""
    if S <= 0:
        return 0.0
    lost = 0
    if cut is not None:
        keep, dropped = cut
        h = header_tokens(model, "ollama")
        lost = max(lost, overlap(keep, keep + dropped, h, h + S))
    h = header_tokens(model, "llama.cpp")
    shifted = sum(overlap(k, k + d, h, h + S) for k, d in shifts)
    return min(S, max(lost, shifted)) / S


# ── loading ─────────────────────────────────────────────────────────────────────────────────────────────

def read_rows(path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def mechanism_sessions(rows: list[dict], machine: str, source: str) -> dict:
    """{key: session} from one mechanism file; calls carry the OBSERVED trim, cut and shifts."""
    turns = {(r["model_id"], r["arm_id"], r["seed"], r["turn_idx"]): r for r in rows if r.get("record") == "r2a_turn"}
    out = {}
    for r in rows:
        if r.get("record") != "r2m_call":
            continue
        t = turns.get((r["model_id"], r["arm_id"], r["seed"], r["turn_idx"]))
        if t is None:
            continue
        key = (machine, source, r["model_id"], tier_of(r["arm_id"]), r["seed"])
        s = out.setdefault(key, {"key": key, "kind": "observed", "model": r["model_id"], "turns": {},
                                 "calib": t.get("token_calib_ratio")})
        s["turns"].setdefault(r["turn_idx"], {"row": t, "calls": {}})
        c = (t.get("calls") or [])[r["call_idx"] - 1]
        np_ = (r.get("new_prompt") or [{}])[-1]
        P = np_.get("n_tokens", r.get("prompt_eval_count"))
        tc = r.get("token_cut")
        s["turns"][r["turn_idx"]]["calls"][r["call_idx"]] = {
            "est": c.get("prompt_tokens_est"), "P": P, "P_msg": tc["prompt"] if tc else P,
            "n_ctx": np_.get("n_ctx_slot"), "msg_trunc": bool(r.get("message_level_truncation")),
            "system_kept": r.get("system_kept"),
            "cut": (tc["keep"], tc["prompt"] - tc["new"]) if tc else None,
            "shifts": [(x["n_keep"], x["n_discard"]) for x in (r.get("context_shifts") or [])]}
    return out


def real_sessions(rows: list[dict], source: str) -> dict:
    """{key: session} from one real-run file; shifts PREDICTED, token cuts INFERRED (lower-bound span)."""
    out = {}
    for t in rows:
        if t.get("record") != "r2a_turn":
            continue
        model, N = t["model_id"], t.get("loaded_context")
        key = (REAL_MACHINE, source, model, tier_of(t["arm_id"]), t["seed"])
        s = out.setdefault(key, {"key": key, "kind": "predicted", "model": model, "turns": {},
                                 "calib": t.get("token_calib_ratio")})
        calls = {}
        for i, c in enumerate(t.get("calls") or [], start=1):
            P, C = c.get("prompt_eval_count_info_only"), c.get("completion_tokens")
            ok = None not in (P, C, N) and c.get("outcome") in (None, "ok")
            sig = ok and P == sp.token_cut_new_len(N)
            calls[i] = {"est": c.get("prompt_tokens_est"), "P": P, "P_msg": P, "n_ctx": N,
                        "msg_trunc": None, "system_kept": None,
                        "cut": (sp.OLLAMA_KEEP, (N - sp.OLLAMA_KEEP) // 2) if sig else None,
                        "shifts": ([(x["n_keep"], x["n_discard"]) for x in
                                    sp.simulate_shifts(P, C, N, sp.n_keep_for(model))] if ok else [])}
        s["turns"][t["turn_idx"]] = {"row": t, "calls": calls}
    return out


def rules_in_use(repo=REPO, files=REAL_FILES) -> dict:
    out = {}
    for f in files:
        start = next((r for r in read_rows(Path(repo) / f) if r.get("record") == "run_start"), {})
        for m, rules in (start.get("rules_in_use") or {}).items():
            out.setdefault(m, list(rules))
    return out


# ── full-transcript estimate ─────────────────────────────────────────────────────────────────────────

def _untrimmed(s: dict, c: dict) -> bool:
    if c["est"] in (None, 0) or c["P"] is None:
        return False
    if s["kind"] == "observed":
        return not c["msg_trunc"] and c["cut"] is None
    return c["n_ctx"] is not None and c["P"] < c["n_ctx"] / 2


def reference_points(sessions: dict) -> dict:
    """{model: [(est, ratio, session key)]} from every untrimmed call."""
    out = {}
    for s in sessions.values():
        for t in s["turns"].values():
            for c in t["calls"].values():
                if _untrimmed(s, c):
                    out.setdefault(s["model"], []).append((c["est"], c["P"] / c["est"], s["key"]))
    return out


def ratio_at(ref: dict, model: str, est: int, exclude_key=None, k: int = KNN):
    pts = [(abs(e - est), r) for e, r, key in ref.get(model, []) if key != exclude_key]
    if not pts:
        return None
    pts.sort()
    return statistics.median(r for _, r in pts[:k])


def full_tokens(ref, s, c):
    r = ratio_at(ref, s["model"], c["est"], exclude_key=s["key"])
    return None if r is None or not c["est"] else c["est"] * r


def estimator_check(sessions: dict, ref: dict) -> dict:
    """Leave-session-out relative error of the full-transcript estimate on untrimmed calls, per model."""
    out = {}
    for s in sessions.values():
        for t in s["turns"].values():
            for c in t["calls"].values():
                if _untrimmed(s, c):
                    f = full_tokens(ref, s, c)
                    if f:
                        out.setdefault(s["model"], []).append(abs(f - c["P"]) / c["P"])
    return {m: {"n": len(v), "median": statistics.median(v), "max": max(v)} for m, v in out.items()}


# ── per call / turn / session ─────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=None)
def _system_prompt_est(seed: int) -> int:
    sys.path.insert(0, str(REPO / "harness"))
    import x2_r2_agent as a
    with contextlib.redirect_stdout(io.StringIO()):
        return a.est_tokens(a.build_agent_session(seed, 40).system_prompt)


def system_tokens(s: dict, ref: dict, sys_est_fn=None) -> int:
    seed = s["key"][-1]
    est = (sys_est_fn or _system_prompt_est)(seed)
    ratio = s.get("calib") or ratio_at(ref, s["model"], est) or 1.0
    return round(est * ratio)


def transcript_share(s: dict, c: dict, ref: dict):
    if s["kind"] == "observed":
        if not c["msg_trunc"]:
            return 0.0
        P = c["P_msg"]
    else:
        if c["P"] is None or c["n_ctx"] is None:
            return None
        if c["P"] < c["n_ctx"] / 2:
            return 0.0
        P = c["P"]
    f = full_tokens(ref, s, c)
    if not f:
        return None
    return min(1.0, max(0.0, 1.0 - P / f))


def session_table(s: dict, ref: dict, rules: list[str], sys_est_fn=None) -> dict:
    """Per turn: transcript share (final-answer call), system share (largest over calls), violation flags; then L,
    before/after rates, lag and per-rule counts."""
    S = system_tokens(s, ref, sys_est_fn)
    turns = []
    for idx in sorted(s["turns"]):
        t = s["turns"][idx]
        calls = [t["calls"][i] for i in sorted(t["calls"])]
        sys_sh = max((system_share_lost(s["model"], S, c["shifts"], c["cut"]) for c in calls), default=0.0)
        tr = transcript_share(s, calls[-1], ref) if calls else None
        row = t["row"]
        turns.append({"turn": idx, "transcript_share": tr, "system_share": sys_sh,
                      "rules": {r: row.get(r) for r in RULES},
                      "violation": any(row.get(r) is False for r in rules)})
    L = next((t["turn"] for t in turns if t["system_share"] > 0), None)
    before = [t for t in turns if L is None or t["turn"] < L]
    after = [t for t in turns if L is not None and t["turn"] >= L]
    first_v_after = next((t["turn"] for t in after if t["violation"]), None)
    by_rule = {r: {"in_use": r in rules,
                   "before": sum(1 for t in before if t["rules"][r] is False),
                   "before_n": sum(1 for t in before if t["rules"][r] is not None),
                   "after": sum(1 for t in after if t["rules"][r] is False),
                   "after_n": sum(1 for t in after if t["rules"][r] is not None)} for r in RULES}
    return {"key": s["key"], "kind": s["kind"], "S": S, "turns": turns, "L": L,
            "before_n": len(before), "before_v": sum(t["violation"] for t in before),
            "after_n": len(after), "after_v": sum(t["violation"] for t in after),
            "first_violation": next((t["turn"] for t in turns if t["violation"]), None),
            "lag": None if first_v_after is None else first_v_after - L, "by_rule": by_rule,
            "system_loss_turns": sum(1 for t in turns if t["system_share"] > 0)}


def _tier_rank(t):
    return TIER_ORDER.index(t) if t in TIER_ORDER else len(TIER_ORDER)


def build(repo=REPO, mech_files=MECH_FILES, real_files=REAL_FILES, sys_est_fn=None) -> dict:
    """Cells (machine, kind, model, tier) -> session tables. Real files are pooled; a (model, tier, seed) already
    seen in an earlier real file keeps the earlier session."""
    repo = Path(repo)
    sessions = {}
    for machine, f in mech_files:
        if (repo / f).exists():
            sessions.update(mechanism_sessions(read_rows(repo / f), machine, Path(f).stem))
    seen = set()
    for f in real_files:
        if not (repo / f).exists():
            continue
        for key, s in real_sessions(read_rows(repo / f), Path(f).stem).items():
            k = (key[2], key[3], key[4])
            if k in seen:
                continue
            seen.add(k)
            sessions[key] = s
    ref = reference_points(sessions)
    rules = rules_in_use(repo, [f for f in real_files if (repo / f).exists()])
    cells = {}
    for key, s in sessions.items():
        machine, _src, model, tier, _seed = key
        tab = session_table(s, ref, rules.get(model, list(RULES)), sys_est_fn)
        cells.setdefault((machine, s["kind"], model, tier), []).append(tab)
    order = sorted(cells, key=lambda c: (c[1] != "observed", c[0], c[2], _tier_rank(c[3])))
    sys_kept = [c["system_kept"] for s in sessions.values() if s["kind"] == "observed"
                for t in s["turns"].values() for c in t["calls"].values() if c["msg_trunc"]]
    return {"cells": {c: sorted(cells[c], key=lambda t: t["key"][-1]) for c in order}, "rules_in_use": rules,
            "est_check": estimator_check(sessions, ref),
            "system_kept_on_trimmed": (sum(1 for x in sys_kept if x), len(sys_kept))}


# ── register strings ───────────────────────────────────────────────────────────────────────────────────

def _pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def _cell_label(kind, tabs):
    if kind == "observed":
        return f"observed, one session per tier, n {len(tabs)}"
    return f"predicted, n {len(tabs)}"


def _turn_val(tabs, turn, field):
    return _mean([next((t[field] for t in tab["turns"] if t["turn"] == turn), None) for tab in tabs])


def info_loss_str(rep: dict) -> str:
    parts = []
    for (machine, kind, model, tier), tabs in rep["cells"].items():
        tr = "/".join(_pct(_turn_val(tabs, k, "transcript_share")) for k in CHECKPOINTS)
        sy = "/".join(_pct(_turn_val(tabs, k, "system_share")) for k in CHECKPOINTS)
        loss_turns = sum(t["system_loss_turns"] for t in tabs)
        n_turns = sum(len(t["turns"]) for t in tabs)
        firsts = ",".join(str(t["L"]) if t["L"] is not None else "-" for t in tabs)
        S = ",".join(str(t["S"]) for t in tabs)
        parts.append(f"{machine} {model} {tier} [{_cell_label(kind, tabs)}]: "
                     f"transcript never seen at turns {'/'.join(map(str, CHECKPOINTS))} {tr}; system prompt never seen "
                     f"{sy}; turns with system-prompt loss {loss_turns}/{n_turns}; first loss turn per session {firsts}; "
                     f"system prompt tokens (est) {S}")
    ec = "; ".join(f"{m} median {_pct(v['median'])} max {_pct(v['max'])} (n {v['n']})" for m, v in rep["est_check"].items())
    k, n = rep["system_kept_on_trimmed"]
    return (" / ".join(parts) + f" / labels: observed = {OBSERVED_LABEL}; predicted = {PREDICTED_LABEL} / "
            f"full-transcript estimator leave-session-out error on untrimmed calls: {ec} / system message kept on "
            f"{k}/{n} observed message-trimmed calls")


def _rate(v, n):
    return f"{v}/{n} ({_pct(v / n) if n else 'n/a'})"


def reliability_str(rep: dict) -> str:
    parts = []
    for (machine, kind, model, tier), tabs in rep["cells"].items():
        bv, bn = sum(t["before_v"] for t in tabs), sum(t["before_n"] for t in tabs)
        av, an = sum(t["after_v"] for t in tabs), sum(t["after_n"] for t in tabs)
        per = ",".join(f"L {t['L'] if t['L'] is not None else '-'} lag {t['lag'] if t['lag'] is not None else '-'}"
                       for t in tabs)
        rules = []
        for r in RULES:
            b = sum(t["by_rule"][r]["before"] for t in tabs)
            bn_r = sum(t["by_rule"][r]["before_n"] for t in tabs)
            a = sum(t["by_rule"][r]["after"] for t in tabs)
            an_r = sum(t["by_rule"][r]["after_n"] for t in tabs)
            in_use = tabs[0]["by_rule"][r]["in_use"]
            rules.append(f"{RULE_TYPE[r]}{'' if in_use else ' (not in use)'} {b}/{bn_r} -> {a}/{an_r}")
        parts.append(f"{machine} {model} {tier} [{_cell_label(kind, tabs)}]: "
                     f"violation turns before loss {_rate(bv, bn)}, from loss on {_rate(av, an)}; per session {per}; "
                     f"by rule before -> after: " + ", ".join(rules))
    return (" / ".join(parts) + f" / labels: observed = {OBSERVED_LABEL}; predicted = {PREDICTED_LABEL}; L = first "
            "turn with system-prompt loss, lag = turns from L to the first violation (rules in use); rules in use "
            + "; ".join(f"{m} {','.join(v)}" for m, v in rep["rules_in_use"].items()))


def main(argv=None) -> int:
    rep = build()
    print(info_loss_str(rep).replace(" / ", "\n"))
    print()
    print(reliability_str(rep).replace(" / ", "\n"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
