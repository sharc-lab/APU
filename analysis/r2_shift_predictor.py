"""R2: predict llama.cpp context shifts per call, and count orphaned messages after Ollama's message-level trim.

Source of the trigger (llama.cpp b11081, the version Ollama 0.34.4 bundles; tools/server/server-context.cpp):

    void pre_decode() {
        iterate(slots, [&](server_slot & slot) {
            if (slot.state == SLOT_STATE_GENERATING && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
                ...
                int n_keep = slot.task->params.n_keep < 0 ? slot.task->n_tokens() : slot.task->params.n_keep;
                if (add_bos_token) { n_keep += 1; }
                n_keep = std::min(slot.n_ctx - 4, n_keep);
                const int n_left    = slot.prompt.n_tokens() - n_keep;
                int       n_discard = slot.task->params.n_discard ? slot.task->params.n_discard : (n_left / 2);
                n_discard = std::clamp(n_discard, 0, std::max(0, n_left - 1));
                SLT_WRN(slot, "slot context shift, n_keep = %d, n_left = %d, n_discard = %d\\n", ...);

The check runs once per decode step, before the token sampled on the previous step is appended to slot.prompt
(server_slot::add_token_to_batch pushes it after pre_decode). A call whose rendered prompt is P tokens and which
returns C generated tokens (Ollama's eval_count = slot.stats.n_gen) decodes C - 1 of them: the last token (EOS or
the num_predict limit) sets has_next_token = false and is never decoded. So the check sees n_tokens = P, P + 1, ...,
P + C - 2, and a shift fires iff

    P + C - 2 + 1 >= n_ctx   <=>   P + C > n_ctx            (C >= 1)

n_keep does not enter the trigger at all; it only sets how much is discarded (n_left = n_tokens - n_keep, n_discard
= n_left / 2), and so whether a second shift can follow in the same call. With n_keep = 4 (Ollama's --keep) + 1 for
the BOS of llama3.1, n_ctx 4096 gives n_discard 2045 and 32768 gives 16381, both far more than the 384-token
per-call budget, so at most one shift per call is possible in these runs; simulate_shifts() handles the general case.
The C - 1 decoded tokens are checked on every mechanism call: the "stop processing: n_tokens = N" line of llama.cpp
equals P + C - 1 - sum(n_discard) (decoded_tokens_check in the summary).

Inputs per call (mechanism file results/x2_r2_mechanism.jsonl):
  P: for calls with a matching fresh prompt check (r2m_prompt_check, match True), the check's prompt_eval_count
     (render-validated, model's own tokenizer); otherwise llama.cpp's own "new prompt ... task.n_tokens" from the
     call's log slice. Both are after any message-level trim and any Ollama token-level cut.
  C: the call's completion_tokens (eval_count) from the r2a_turn row's calls list.
  n_ctx: n_ctx_slot from the same "new prompt" line.
Real run (results/x2_r2_real_v1.jsonl, no server log): P = prompt_eval_count_info_only (the mechanism file shows it
equals task.n_tokens on every call, so it is the post-trim prompt the model saw), C = completion_tokens, n_ctx =
loaded_context. The calibrated transcript estimate (transcript_tokens_calibrated) is the pre-trim size and is not
what llama.cpp holds, so it is not used for the prediction. Real-run shifts are PREDICTED; nothing was observed.

Orphans (mechanism file only; the real run has no message-level data): over message-truncated calls, the first kept
non-system message's role from the log (first_kept_message_role_log). An assistant or tool message there has no
user message before it in the kept window. A tool message there is an orphaned tool result (its assistant tool call
was dropped). The message list of each call is rebuilt from the r2a_turn rows exactly as x2_r2_agent.run_turn
builds it, to tell a call-1 tool-call assistant from a final-answer assistant; the rebuilt role at the logged index
must equal the logged role (role_check in the summary).

Usage: py -3.12 analysis/r2_shift_predictor.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MECH_FILE = "results/x2_r2_mechanism.jsonl"
REAL_FILE = "results/x2_r2_real_v1.jsonl"
NUM_PREDICT = 384          # max_tokens_per_call in both runs' run_start
OLLAMA_KEEP = 4            # Ollama's --keep for llama-server (num_keep default 4)
BOS_MODELS = ("llama3.1",)  # models whose GGUF adds BOS (n_keep + 1); qwen3 adds none


def tier_of(arm: str) -> str:
    return arm.replace("_call2_notools", "").replace("ollama_ctx_", "num_ctx_")


def predict(P: int, C: int, n_ctx: int) -> bool:
    """The derived trigger: a context shift fires during the call iff P + C > n_ctx."""
    return P + C > n_ctx


# Refinements tried against the observed shifts (none is fitted; each is one fixed alternative reading).
VARIANTS = {
    "exact P+C>n_ctx": lambda P, C, N: P + C > N,
    "off-by-one P+C>=n_ctx (also: one extra BOS token counted)": lambda P, C, N: P + C >= N,
    "off-by-one P+C>n_ctx+1": lambda P, C, N: P + C > N + 1,
    "budget P+num_predict>n_ctx": lambda P, C, N: P + NUM_PREDICT > N,
    "prompt only P>=n_ctx-1": lambda P, C, N: P + 1 >= N,
}


def token_cut_new_len(n_ctx: int, keep: int = OLLAMA_KEEP) -> int:
    """Prompt length after Ollama's token-level cut (llm/llama_server.go completionPromptForRequest: limit =
    num_ctx - (num_ctx - keep) / 2; keep is num_keep, 4). A call whose prompt_eval_count equals this exactly is
    flagged as carrying the cut's signature (inferred, not observed, in the real run)."""
    return n_ctx - (n_ctx - keep) // 2


def simulate_shifts(P: int, C: int, n_ctx: int, n_keep: int) -> list[dict]:
    """Shift events of one call, following pre_decode step by step (n_keep already includes the BOS +1)."""
    n, out = P, []
    keep = min(n_ctx - 4, n_keep)
    for _ in range(max(C - 1, 0)):
        if n + 1 >= n_ctx:
            n_left = n - keep
            disc = max(0, min(n_left // 2, max(0, n_left - 1)))
            out.append({"n_keep": keep, "n_left": n_left, "n_discard": disc})
            n -= disc
        n += 1
    return out


def n_keep_for(model: str, logged_keep: int | None = None) -> int:
    k = OLLAMA_KEEP if logged_keep is None else logged_keep
    return k + (1 if any(model.startswith(b) for b in BOS_MODELS) else 0)


def read_rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


# ── mechanism file: predictor vs observed ───────────────────────────────────────────────────────────

def _stop_n_tokens(row: dict):
    import re
    for line in reversed(row.get("log_excerpt") or []):
        m = re.search(r"stop processing: n_tokens = (\d+)", line)
        if m:
            return int(m.group(1))
    return None


def mechanism_calls(rows: list[dict]) -> list[dict]:
    """One dict per r2m_call: P (and its source), C, n_ctx, observed and predicted shifts."""
    turns = {(r["model_id"], r["arm_id"], r["seed"], r["turn_idx"]): r for r in rows if r.get("record") == "r2a_turn"}
    checks = {(r["model_id"], r["arm_id"], r["seed"], r["turn_idx"], r["call_idx"]): r for r in rows
              if r.get("record") == "r2m_prompt_check" and r.get("match") is True}
    out = []
    for r in rows:
        if r.get("record") != "r2m_call":
            continue
        key = (r["model_id"], r["arm_id"], r["seed"], r["turn_idx"])
        t = turns.get(key)
        calls = (t or {}).get("calls") or []
        c = calls[r["call_idx"] - 1] if len(calls) >= r["call_idx"] else {}
        np_ = (r.get("new_prompt") or [None])[-1]
        chk = checks.get(key + (r["call_idx"],))
        if chk is not None:
            P, src = chk["check_prompt_eval_count"], "fresh_check"
        elif np_ is not None:
            P, src = np_["n_tokens"], "new_prompt"
        else:
            P, src = None, None
        C = c.get("completion_tokens")
        N = np_["n_ctx_slot"] if np_ else None
        obs = r.get("context_shifts") or []
        rec = {"tier": tier_of(r["arm_id"]), "model": r["model_id"], "seed": r["seed"], "turn": r["turn_idx"],
               "call": r["call_idx"], "P": P, "P_source": src, "C": C, "n_ctx": N,
               "new_prompt_n_tokens": np_["n_tokens"] if np_ else None,
               "prompt_eval_count": r.get("prompt_eval_count"), "observed": bool(obs), "observed_events": len(obs),
               "observed_shifts": obs, "token_cut": r.get("token_cut"), "stop_n_tokens": _stop_n_tokens(r)}
        if None in (P, C, N):
            rec["predicted"] = None
        else:
            rec["predicted"] = predict(P, C, N)
            rec["margin"] = P + C - N
            rec["cut_signature"] = P == token_cut_new_len(N)
            sim = simulate_shifts(P, C, N, n_keep_for(r["model_id"], np_.get("n_keep")))
            rec["predicted_events"] = len(sim)
            rec["predicted_shifts"] = sim
            if rec["stop_n_tokens"] is not None:
                rec["decoded_ok"] = rec["stop_n_tokens"] == P + C - 1 - sum(s["n_discard"] for s in obs)
        out.append(rec)
    return out


def confusion(calls: list[dict], fn=None) -> dict:
    fn = fn or (lambda P, C, N: predict(P, C, N))
    tp = fp = fn_ = tn = skipped = 0
    for c in calls:
        if c.get("predicted") is None:
            skipped += 1
            continue
        p = fn(c["P"], c["C"], c["n_ctx"])
        o = c["observed"]
        tp += p and o
        fp += p and not o
        fn_ += o and not p
        tn += (not p) and (not o)
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn_) if tp + fn_ else None
    return {"tp": tp, "fp": fp, "fn": fn_, "tn": tn, "n": tp + fp + fn_ + tn, "skipped": skipped,
            "precision": prec, "recall": rec}


def by_tier(calls: list[dict]) -> dict:
    out = {}
    for c in calls:
        out.setdefault(c["tier"], []).append(c)
    return out


def mechanism_summary(rows: list[dict]) -> dict:
    calls = mechanism_calls(rows)
    tiers = by_tier(calls)
    per_tier = {}
    for tier, cs in tiers.items():
        conf = confusion(cs)
        ok = [c for c in cs if c.get("predicted") is not None]
        conf["events_observed"] = sum(c["observed_events"] for c in cs)
        conf["events_predicted"] = sum(c.get("predicted_events") or 0 for c in ok)
        conf["n_discard_match"] = all([s["n_discard"] for s in c["observed_shifts"]] ==
                                      [s["n_discard"] for s in c["predicted_shifts"]] for c in ok if c["observed"])
        margins = [c["margin"] for c in ok]
        pos = [m for m in margins if m > 0]
        neg = [m for m in margins if m <= 0]
        conf["closest_margin_shifted"] = min(pos) if pos else None
        conf["closest_margin_not_shifted"] = max(neg) if neg else None
        per_tier[tier] = conf
    misses = [c for c in calls if c.get("predicted") is not None and c["predicted"] != c["observed"]]
    variants = {name: confusion(calls, fn) for name, fn in VARIANTS.items()}
    dec = [c.get("decoded_ok") for c in calls if "decoded_ok" in c]
    src = {}
    for c in calls:
        src[c["P_source"]] = src.get(c["P_source"], 0) + 1
    return {"per_tier": per_tier, "overall": confusion(calls), "misses": misses, "variants": variants,
            "decoded_tokens_check": {"n": len(dec), "ok": sum(1 for d in dec if d)},
            "P_sources": src,
            "P_equals_prompt_eval_count": sum(1 for c in calls if c["P"] is not None and c["P"] == c["prompt_eval_count"]),
            "P_equals_new_prompt": sum(1 for c in calls if c["P"] is not None and c["P"] == c["new_prompt_n_tokens"]),
            "token_cut_signature": {
                "signature_calls": sum(1 for c in calls if c.get("cut_signature")),
                "observed_cuts": sum(1 for c in calls if c.get("token_cut")),
                "both": sum(1 for c in calls if c.get("cut_signature") and c.get("token_cut"))},
            "n_calls": len(calls)}


# ── real run: predicted only ────────────────────────────────────────────────────────────────────────

def real_calls(rows: list[dict]) -> list[dict]:
    out = []
    for t in rows:
        if t.get("record") != "r2a_turn" or t.get("mode", "real") != "real":
            continue
        N = t.get("loaded_context")
        for i, c in enumerate(t.get("calls") or [], start=1):
            P, C = c.get("prompt_eval_count_info_only"), c.get("completion_tokens")
            rec = {"tier": tier_of(t["arm_id"]), "model": t["model_id"], "seed": t["seed"], "turn": t["turn_idx"],
                   "call": i, "P": P, "C": C, "n_ctx": N, "attempt_id": t.get("attempt_id")}
            if None in (P, C, N) or c.get("outcome") not in (None, "ok"):
                rec["predicted"] = None
            else:
                rec["predicted"] = predict(P, C, N)
                rec["predicted_events"] = len(simulate_shifts(P, C, N, n_keep_for(t["model_id"])))
                rec["cut_signature"] = P == token_cut_new_len(N)
            out.append(rec)
    return out


def _first_rule_failure(turns: list[dict], rules: list[str]):
    return next((r["turn_idx"] for r in sorted(turns, key=lambda r: r["turn_idx"])
                 if any(r.get(rule) is False for rule in rules)), None)


def real_summary(rows: list[dict], rules_in_use: dict | None = None) -> dict:
    """Per tier x model: calls and shift events predicted, sessions with any; per session the first predicted shift
    turn next to the first rule failure (rules in use) and how they line up."""
    if rules_in_use is None:
        start = next((r for r in rows if r.get("record") == "run_start"), {})
        rules_in_use = start.get("rules_in_use") or {}
    calls = real_calls(rows)
    turns_by_session = {}
    for t in rows:
        if t.get("record") == "r2a_turn":
            turns_by_session.setdefault((tier_of(t["arm_id"]), t["model_id"], t["seed"]), []).append(t)
    cells = {}
    for c in calls:
        cell = cells.setdefault((c["tier"], c["model"]), {"calls": 0, "pred_calls": 0, "pred_events": 0,
                                                          "pred_call1": 0, "pred_call2": 0, "skipped": 0,
                                                          "cut_sig_calls": 0, "max_margin": None, "sessions": {}})
        cell["calls"] += 1
        if c["predicted"] is None:
            cell["skipped"] += 1
            continue
        m = c["P"] + c["C"] - c["n_ctx"]
        cell["max_margin"] = m if cell["max_margin"] is None else max(cell["max_margin"], m)
        s = cell["sessions"].setdefault(c["seed"], {"first_pred_shift": None, "pred_calls": 0,
                                                    "first_cut_signature": None})
        if c.get("cut_signature"):
            cell["cut_sig_calls"] += 1
            if s["first_cut_signature"] is None or c["turn"] < s["first_cut_signature"]:
                s["first_cut_signature"] = c["turn"]
        if c["predicted"]:
            cell["pred_calls"] += 1
            cell["pred_events"] += c["predicted_events"]
            cell[f"pred_call{c['call']}"] += 1
            s["pred_calls"] += 1
            if s["first_pred_shift"] is None or c["turn"] < s["first_pred_shift"]:
                s["first_pred_shift"] = c["turn"]
    for (tier, model), cell in cells.items():
        rules = list(rules_in_use.get(model) or [])
        for seed, s in cell["sessions"].items():
            ff = _first_rule_failure(turns_by_session.get((tier, model, seed), []), rules)
            s["first_rule_failure"] = ff
            fp = s["first_pred_shift"]
            if ff is None or fp is None:
                s["lineup"] = "no shift predicted" if fp is None else "no rule failure"
            elif ff == fp:
                s["lineup"] = "same turn"
            elif ff > fp:
                s["lineup"] = "failure after first shift"
            else:
                s["lineup"] = "failure before first shift"
            starts = [v for v in (fp, s["first_cut_signature"]) if v is not None]
            s["first_prompt_start_loss"] = min(starts) if starts else None
        cell["sessions"] = dict(sorted(cell["sessions"].items()))
        cell["sessions_with_pred_shift"] = sum(1 for s in cell["sessions"].values() if s["pred_calls"])
    return {"cells": dict(sorted(cells.items())), "rules_in_use": rules_in_use, "n_calls": len(calls)}


# ── orphans after the message-level trim ────────────────────────────────────────────────────────────

def rebuild_roles(turn_rows: list[dict]) -> dict:
    """{(turn, call): [message kinds]} for the messages sent on each call, rebuilt as x2_r2_agent.run_turn builds
    the transcript: system; per turn user, then (if call 1 parsed tool calls) assistant tool-call message, one tool
    message per call, the final assistant, and tool messages for any call-2 tool calls; else the final assistant.
    Kinds: system, user, assistant_toolcall, assistant_final, tool."""
    msgs = ["system"]
    out = {}
    for t in sorted(turn_rows, key=lambda r: r["turn_idx"]):
        calls = t.get("calls") or []
        msgs.append("user")
        out[(t["turn_idx"], 1)] = list(msgs)
        c1 = (calls[0].get("tool_calls_parsed") or []) if calls else []
        if c1:
            msgs.append("assistant_toolcall")
            msgs.extend(["tool"] * len(c1))
            out[(t["turn_idx"], 2)] = list(msgs)
            msgs.append("assistant_final")
            c2 = (calls[1].get("tool_calls_parsed") or []) if len(calls) > 1 else []
            msgs.extend(["tool"] * len(c2))
        else:
            msgs.append("assistant_final")
    return out


def orphan_summary(rows: list[dict]) -> dict:
    turns = {}
    for r in rows:
        if r.get("record") == "r2a_turn":
            turns.setdefault((r["model_id"], r["arm_id"], r["seed"]), []).append(r)
    roles = {k: rebuild_roles(v) for k, v in turns.items()}
    per_tier = {}
    for r in rows:
        if r.get("record") != "r2m_call":
            continue
        tier = tier_of(r["arm_id"])
        d = per_tier.setdefault(tier, {"calls": 0, "truncated": 0, "first_user": 0, "first_assistant_toolcall": 0,
                                       "first_assistant_final": 0, "first_tool": 0, "first_unknown": 0,
                                       "role_check_ok": 0, "role_check_bad": 0, "length_check_bad": 0})
        d["calls"] += 1
        if not r.get("message_level_truncation"):
            continue
        d["truncated"] += 1
        idx = r.get("first_kept_message_index_log")
        kinds = roles.get((r["model_id"], r["arm_id"], r["seed"]), {}).get((r["turn_idx"], r["call_idx"]))
        if kinds is None or len(kinds) != r.get("n_messages_sent"):
            d["length_check_bad"] += 1
        kind = kinds[idx] if kinds is not None and idx is not None and idx < len(kinds) else None
        logged = r.get("first_kept_message_role_log")
        if kind is not None:
            if kind.split("_")[0] == logged:
                d["role_check_ok"] += 1
            else:
                d["role_check_bad"] += 1
        if kind is None or kind.split("_")[0] != logged:
            kind = {"user": "user", "tool": "tool"}.get(logged, "unknown")
        d[f"first_{kind}"] += 1
    for d in per_tier.values():
        n = d["truncated"]
        orph = n - d["first_user"]
        d["orphan_first_kept"] = orph
        d["orphan_rate"] = orph / n if n else None
        d["orphan_tool_rate"] = d["first_tool"] / n if n else None
    return per_tier


# ── report ──────────────────────────────────────────────────────────────────────────────────────────

def _pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def main(argv=None) -> int:
    mrows = read_rows(REPO / MECH_FILE)
    ms = mechanism_summary(mrows)
    print("Trigger: shift iff P + C > n_ctx (llama.cpp pre_decode: n_tokens + 1 >= n_ctx, C - 1 tokens decoded)")
    print(f"mechanism calls {ms['n_calls']}; P sources {ms['P_sources']}; P == prompt_eval_count on "
          f"{ms['P_equals_prompt_eval_count']}, == new_prompt n_tokens on {ms['P_equals_new_prompt']}; decoded C-1 "
          f"check {ms['decoded_tokens_check']}")
    print("| tier | TP | FP | FN | TN | precision | recall | events obs/pred | closest margins (shift / no shift) |")
    for t, c in ms["per_tier"].items():
        print(f"| {t} | {c['tp']} | {c['fp']} | {c['fn']} | {c['tn']} | {_pct(c['precision'])} | {_pct(c['recall'])} | "
              f"{c['events_observed']}/{c['events_predicted']} | {c['closest_margin_shifted']} / "
              f"{c['closest_margin_not_shifted']} |")
    print("token cut signature (mechanism):", ms["token_cut_signature"])
    print("misses:", [{k: m[k] for k in ("tier", "turn", "call", "P", "C", "n_ctx", "observed")} for m in ms["misses"]])
    for name, c in ms["variants"].items():
        print(f"  variant {name}: TP {c['tp']} FP {c['fp']} FN {c['fn']} TN {c['tn']}")
    rs = real_summary(read_rows(REPO / REAL_FILE))
    print("\nReal run (PREDICTED shifts, no observation):")
    for (tier, model), c in rs["cells"].items():
        sess = "; ".join(f"{seed}: shift {s['first_pred_shift']} cut-sig {s['first_cut_signature']} fail {s['first_rule_failure']} ({s['lineup']})"
                         for seed, s in c["sessions"].items())
        print(f"  {tier} {model}: {c['pred_calls']}/{c['calls']} calls (call1 {c['pred_call1']}, call2 "
              f"{c['pred_call2']}), {c['pred_events']} events, {c['sessions_with_pred_shift']} sessions, {c['cut_sig_calls']} cut-signature calls; {sess}")
    print("\nOrphans:")
    for t, d in orphan_summary(mrows).items():
        print(f"  {t}: {d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
