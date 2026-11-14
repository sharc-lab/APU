"""Before/after replay of the spend-reservation token count (operator decision 2026-11-13: replace the whitespace word
count in src/cloud/client.py with a conservative upper bound, upper_bound_input_tokens). Replay of recorded files only:
no network, no model call, no cloud spend.

Part R1, measured tokenizer truth: the 40 Anthropic count_tokens results in
results/anthropic_token_calibration_20261021.jsonl (claude-sonnet-5-5, real agent prompts). The texts are public
dataset text and are not committed; scripts/trace_output_calibration.py --extract rebuilds them into a scratch
directory, and this part reads them from --texts (each matched to its row by sha256). Writes the per-text counts (no
text) to results/token_reservation_check_v1.jsonl.

Part R2, the live 8192 run (row post-run-check-live-8192, results/x2_post_run_check_live_v1.jsonl): for every
escalated task, the first cloud request is rebuilt exactly as scripts/post_run_check_cloud.py built it
(cloud_redo.rebuild from results/x2_post_run_check_local_v1.jsonl, the same tools, max_tokens and cache_control), and
its old and new reservation are set against the billed input tokens of that call (cloud_calls[0].input_tokens).
Later calls are not rebuildable (the model's replies are not stored), so for them the reservation is ESTIMATED as
the first call's bound scaled by billed input_tokens(call k) / input_tokens(call 1). The per-task cap (USD 0.25) is
then replayed with src/cloud/task_budget.py's counting rule (a settled call counts at max(billed, upper bound,
reservation); before a call, counted + reservation must be <= the cap).

Run: py -3.12 analysis/token_fix_replay.py [--texts DIR]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "harness", REPO / "src", REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CALIB = REPO / "results" / "anthropic_token_calibration_20261021.jsonl"
CHECK_OUT = REPO / "results" / "token_reservation_check_v1.jsonl"
LIVE = REPO / "results" / "x2_post_run_check_live_v1.jsonl"
LOCAL = REPO / "results" / "x2_post_run_check_local_v1.jsonl"
TASK_CAP_USD = 0.25
MAX_TOKENS = 2048


def _rows(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def old_reservation_tokens(messages, kwargs) -> int:
    """The pre-2026-11-13 CloudClient._reservation input count (whitespace words of message content + of tools)."""
    from cloud import client as cc
    n = cc._approx_token_count(messages)
    if kwargs.get("tools"):
        n += max(1, len(json.dumps(kwargs["tools"]).split()))
    return n


def _q(xs):
    xs = sorted(xs)
    return xs[0], statistics.median(xs), xs[-1]


def r1(texts: Path | None, out: list[str]) -> list[dict]:
    from cloud import client as cc
    cal = {r["text_sha256"]: r for r in _rows(CALIB)}
    recs = []
    if texts is not None and texts.exists():
        for f in sorted(texts.glob("*.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            t = d.get("text") or ""
            h = hashlib.sha256(t.encode("utf-8")).hexdigest()
            if h not in cal:
                continue
            m = [{"role": "user", "content": t}]
            real = int(cal[h]["anthropic_tokens"])
            recs.append({"record": "reservation_check", "text_sha256": h, "dataset": cal[h]["dataset"],
                         "chars": len(t), "utf8_bytes": len(t.encode("utf-8")), "anthropic_tokens": real,
                         "old_word_count": old_reservation_tokens(m, {}),
                         "new_upper_bound": cc.upper_bound_input_tokens(m)})
        recs.sort(key=lambda r: r["text_sha256"])
        CHECK_OUT.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in recs), encoding="utf-8",
                             newline="\n")
    elif CHECK_OUT.exists():
        recs = _rows(CHECK_OUT)
    out.append("PART R1: reservation count vs Anthropic count_tokens (claude-sonnet-5-5), real agent prompts")
    if not recs:
        out.append("  no texts and no committed check file; run with --texts")
        return recs
    o = [r["old_word_count"] / r["anthropic_tokens"] for r in recs]
    n = [r["new_upper_bound"] / r["anthropic_tokens"] for r in recs]
    out.append(f"  n={len(recs)} texts (of {len(cal)} calibration rows)")
    out.append("  old word count / real: min %.3f median %.3f max %.3f; under the real count on %d of %d"
               % (*_q(o), sum(x < 1 for x in o), len(o)))
    out.append("  new upper bound / real: min %.3f median %.3f max %.3f; under the real count on %d of %d"
               % (*_q(n), sum(x < 1 for x in n), len(n)))
    return recs


def r2(out: list[str]) -> list[dict]:
    import cloud_redo as CR
    import x2_r2_evalset as EV
    from cloud import client as cc
    live = [r for r in _rows(LIVE) if r.get("record") == "prc_live_task" and r.get("decision") == "escalate"]
    src = CR.load_rows(LOCAL)
    table = CR.load_price_table()
    model = CR.escalation_model(table)
    cache, res = {}, []
    for lr in live:
        r = next(x for x in src if x.get("record") == "r2a_turn" and x["seed"] == lr["seed"]
                 and x["turn_idx"] == lr["turn_idx"] and x["task_id"] == lr["task_id"])
        es = CR.build_session_for(r, cache)
        prior = [x for x in CR.session_rows(src, r["attempt_id"]) if x["turn_idx"] < r["turn_idx"]]
        built = CR.rebuild(es, prior, r["turn_idx"])
        kw = {"tools": EV.tools_payload(es.design), "max_tokens": MAX_TOKENS, "custom_llm_provider": "anthropic"}
        th = CR.THINKING_OFF.get(model)
        if th is not None:
            kw["thinking"] = th
        msgs = CR.with_cache_control(copy.deepcopy(built["messages"]))
        ttl = cc.request_cache_ttl(msgs, kw) if cc._uses_cache_control(msgs, kw) else None
        old1, new1 = old_reservation_tokens(msgs, kw), cc.upper_bound_input_tokens(msgs, kw)
        calls = (lr.get("cloud") or {}).get("cloud_calls") or []
        if not calls:
            continue
        in1 = calls[0]["input_tokens"]
        sim = {}
        # old / new: task_budget's current rule (a settled call counts at max(billed, upper, reservation)).
        # new_settle_upper: the new pre-call bound, but a settled call counts at its post-call upper bound (every
        # billed input token at the uncached rate, from the call's own usage), not at the pre-call reservation.
        # count_tokens_settle_upper: the design adopted 2026-11-13 (reserve with Anthropic count_tokens x
        # COUNT_TOKENS_MARGIN + COUNT_TOKENS_PAD, settle at the post-call upper bound). count_tokens is not called in
        # this $0 replay: each call's billed input tokens stand in for its count (the same request on the same
        # tokenizer; count_tokens is documented as an estimate, which the margin covers). Exact for every call.
        for name, first, settle_resv in (("old", old1, True), ("new", new1, True), ("new_settle_upper", new1, False),
                                         ("count_tokens_settle_upper", None, False)):
            counted, sent, stop = 0.0, 0, None
            for k, c in enumerate(calls):
                if first is None:
                    tok = math.ceil(c["input_tokens"] * cc.COUNT_TOKENS_MARGIN) + cc.COUNT_TOKENS_PAD
                else:
                    tok = first if k == 0 else round(first * c["input_tokens"] / in1)
                resv = cc.reservation_cost_usd(model, tok, MAX_TOKENS, ttl)
                if counted + resv > TASK_CAP_USD + 1e-12:
                    stop = k
                    break
                upper = cc.estimate_cost_usd(model, c["input_tokens"], c["output_tokens"])
                counted += max(upper, resv) if settle_resv else upper
                sent += 1
            sim[name] = {"first_reservation_tokens": first, "calls_sent": sent, "refused_at_call": stop,
                         "counted_usd": round(counted, 6)}
        res.append({"seed": lr["seed"], "turn_idx": lr["turn_idx"], "task_id": lr["task_id"], "n_calls": len(calls),
                    "billed_in_call1": in1, "old_call1": old1, "new_call1": new1, "sim": sim,
                    "recorded_outcome": lr["cloud"].get("outcome")})
    out.append("")
    out.append(f"PART R2: live 8192 run, escalated tasks with cloud calls n={len(res)} (model {model}, per-task cap "
               f"USD {TASK_CAP_USD}, max_tokens {MAX_TOKENS})")
    out.append("  task | calls | billed in, call 1 | old count | new bound | old/billed | new/billed | "
               "old sim: sent/refused-at | new sim: sent/refused-at")
    for x in res:
        out.append(f"  {x['seed']}:{x['turn_idx']}:{x['task_id']} | {x['n_calls']} | {x['billed_in_call1']} | "
                   f"{x['old_call1']} | {x['new_call1']} | {x['old_call1'] / x['billed_in_call1']:.3f} | "
                   f"{x['new_call1'] / x['billed_in_call1']:.3f} | {x['sim']['old']['calls_sent']}/"
                   f"{x['sim']['old']['refused_at_call']} | {x['sim']['new']['calls_sent']}/"
                   f"{x['sim']['new']['refused_at_call']}")
    if res:
        o = [x["old_call1"] / x["billed_in_call1"] for x in res]
        n = [x["new_call1"] / x["billed_in_call1"] for x in res]
        out.append("  call 1 (exact rebuild): old / billed min %.3f median %.3f max %.3f, under on %d of %d"
                   % (*_q(o), sum(v < 1 for v in o), len(o)))
        out.append("  call 1 (exact rebuild): new / billed min %.3f median %.3f max %.3f, under on %d of %d"
                   % (*_q(n), sum(v < 1 for v in n), len(n)))
        for name in ("old", "new", "new_settle_upper", "count_tokens_settle_upper"):
            ex = [x for x in res if x["sim"][name]["refused_at_call"] is not None]
            basis = ("every call's billed input tokens stand in for count_tokens, exact; max counted per task USD "
                     f"{max(x['sim'][name]['counted_usd'] for x in res):.4f}" if name.startswith("count_tokens")
                     else "calls after call 1 ESTIMATED")
            out.append(f"  per-task cap replay, {name} reservation: {len(ex)} of {len(res)} tasks would stop at the "
                       f"cap before their recorded last call ({basis})")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", type=Path, default=None,
                    help="scratch dir from scripts/trace_output_calibration.py --extract (not committed)")
    a = ap.parse_args(argv)
    out: list[str] = []
    r1(a.texts, out)
    r2(out)
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
