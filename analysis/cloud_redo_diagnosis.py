"""Offline diagnosis of the live cloud redo (results/cloud_redo_v1.jsonl, row cloud-redo-cost-per-wrong-task).

No network call. Answers three questions from the rows plus a rebuild of each task's transcript with
scripts/cloud_redo.py::rebuild (the same function the live run used):

(a) What the redo sent. rebuild() makes the local session's client-side transcript (system prompt; every prior user
    message rebuilt exactly; every prior call's assistant content and parsed tool calls from the source row; every tool
    result recomputed on the environment as the local model left it) and appends the wrong turn's user message. That is
    the history the local harness built client-side before Ollama truncated it server-side to num_ctx. Two checks per
    task: the longest prior assistant content (the source rows keep content[:4000], so a value of 4000 would mean a
    truncated message) and the rebuilt prompt's harness token estimate (x2_r2_agent.prompt_tokens_est) against the
    local run's own estimate for its first call of that turn (calls[0].prompt_tokens_est, the untrimmed client-side
    prompt). Environment: env_before = the state after the local model's earlier turns (its earlier wrong calls
    included), not the wrong turn's own calls and not a clean state.

(b) Model and tier: run_start model_id plus the price file's selection.

(c) Every cloud failure (cloud_end_state_ok false) in exactly one class, first match wins:
    1 thinking_empty_at_cap (redo protocol: thinking consumed the output budget, empty final answer): the loop ended on
      a tools-offered call (n_calls <= MAX_TOOL_ROUNDS) with empty content and no tool call. The live run sent
      max_tokens = x2_r2_agent.MAX_TOKENS_PER_CALL (384, the local num_predict) and no thinking parameter;
      claude-sonnet-5-5 thinks by default and thinking tokens count against max_tokens (found by the all-cloud B3 job,
      scripts/all_cloud_b3.py THINKING_OFF). v1 rows do not record finish_reason, so the class rests on output tokens:
      where the turn is a single call, output_tokens equals the cap exactly with nothing visible (the only output the
      harness cannot see is thinking); for multi-call turns only the turn total is recorded (consistent, not confirmed
      per call).
    2 earlier_state_repair: the end-state diff touches only records outside the task's reference calls (the cloud did the
      task and also changed state left by the local model's earlier turns; the checker's expected state is env_before
      plus the reference, so a repair counts as a diff).
    3 tool_round_cap: the turn used all MAX_TOOL_ROUNDS tool rounds and the final call (tool_choice none) did not
      complete the task.
    4 grader_issue: the answer value is in the final text in another form (numeric string, cm vs m, separators) but
      answer_ok is false (evalset_env.answer_has on the raw final text, plus the cm form).
    5 missing_info: a needed value is absent from the transcript sent (a truncated prior message).
    6 cloud_model_error: anything else (the model did the wrong thing with the full transcript and room to finish).
    Task ambiguity is not a mechanical class; no failure reached classes 4 to 6 in v1 (see FINDINGS).
"""
from __future__ import annotations

import collections
import contextlib
import importlib.util
import io
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REDO = "results/cloud_redo_v1.jsonl"
CLASSES = ("thinking_empty_at_cap", "earlier_state_repair", "tool_round_cap", "grader_issue", "missing_info",
           "cloud_model_error")


def _cloud_redo(repo: Path):
    spec = importlib.util.spec_from_file_location("cloud_redo", repo / "scripts" / "cloud_redo.py")
    cr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cr)
    return cr


def _ref_records(task) -> set[str]:
    out = set()
    for c in task.reference:
        a = c.get("arguments") or {}
        for k in ("record_id", "sku", "account_id", "event_id", "path", "from_account", "to_account", "to"):
            if a.get(k):
                out.add(str(a[k]))
    return out


def _diff_record(key: str) -> str:
    parts = key.split("/")
    return parts[1] if len(parts) > 1 else key


def classify(row: dict, task, max_rounds: int, cap: int, maxc: int, trunc_chars: int, E) -> str:
    fin = row.get("cloud_final_text") or ""
    if "cloud_calls" in row:   # runs that record finish_reason per call (cloud_redo.py --thinking-off)
        if row.get("empty_at_cap"):
            return "thinking_empty_at_cap"
    elif not fin and row["n_calls"] <= max_rounds:
        return "thinking_empty_at_cap"
    diff = row.get("cloud_end_state_diff") or []
    refs = _ref_records(task)
    if diff and refs and not any(_diff_record(d) in refs for d in diff):
        return "earlier_state_repair"
    if row["n_calls"] > max_rounds:
        return "tool_round_cap"
    if row.get("cloud_answer_ok") is False and task.answer_values:
        loose = all(E.answer_has(fin, v) or (isinstance(v, float) and E.answer_has_length(fin, v))
                    for v in task.answer_values)
        if loose:
            return "grader_issue"
    if maxc >= trunc_chars:
        return "missing_info"
    return "cloud_model_error"


def compute(repo: Path = REPO, path: str = REDO) -> dict:
    cr = _cloud_redo(repo)
    E, ag = cr.E, cr.ag
    rows = cr.load_rows(repo / path)
    start = next(r for r in rows if r.get("record") == "run_start")
    tasks = [r for r in rows if r.get("record") == "cloud_redo_task"]
    src = cr.load_rows(repo / start["source"])
    wrong = {(r["attempt_id"], r["turn_idx"]): r for r in cr.wrong_tasks(src)}
    cap = ag.MAX_TOKENS_PER_CALL
    cache, per, ratios, maxcs, n_msgs = {}, [], [], [], []
    with contextlib.redirect_stdout(io.StringIO()):
        for t in tasks:
            r = wrong[(t["attempt_id"], t["turn_idx"])]
            es = cr.build_session_for(r, cache)
            prior = [x for x in cr.session_rows(src, r["attempt_id"]) if x["turn_idx"] < r["turn_idx"]]
            b = cr.rebuild(es, prior, r["turn_idx"])
            rebuilt = ag.prompt_tokens_est(b["messages"], cr.EV.tools_payload(es.design))
            ratios.append(rebuilt / r["calls"][0]["prompt_tokens_est"])
            maxc = max([len(c.get("content") or "") for x in prior for c in x.get("calls") or []] or [0])
            maxcs.append(maxc)
            n_msgs.append(len(b["messages"]))
            if t["cloud_end_state_ok"]:
                continue
            k = classify(t, b["item"].task, cr.MAX_TOOL_ROUNDS, cap, maxc, cr.TRUNC_CHARS, E)
            per.append({"cls": k, "task_id": t["task_id"], "tier": t["tier"], "seed": t["seed"],
                        "turn_idx": t["turn_idx"], "n_calls": t["n_calls"], "output_tokens": t["output_tokens"],
                        "diff": t["cloud_end_state_diff"], "final": (t["cloud_final_text"] or "")[:200]})
    counts = collections.Counter(p["cls"] for p in per)
    single = [p for p in per if p["cls"] == "thinking_empty_at_cap" and p["n_calls"] == 1]
    multi = [p for p in per if p["cls"] == "thinking_empty_at_cap" and p["n_calls"] > 1]
    table = json.loads((repo / start["price_file"]).read_text(encoding="utf-8"))
    return {
        "n_tasks": len(tasks), "n_correct": sum(1 for t in tasks if t["cloud_end_state_ok"]), "n_fail": len(per),
        "model_id": start["model_id"],
        "tier": next((k for k, v in table["selection"].items() if v == start["model_id"]), None),
        "max_tokens": cap, "history_truncated_rows": sum(1 for t in tasks if t["history_truncated_messages"]),
        "max_prior_content_chars": max(maxcs), "trunc_chars": cr.TRUNC_CHARS,
        "tok_ratio_min": min(ratios), "tok_ratio_max": max(ratios),
        "msgs_min": min(n_msgs), "msgs_max": max(n_msgs),
        "counts": {c: counts.get(c, 0) for c in CLASSES},
        "cap_single_n": len(single), "cap_single_at_cap": sum(1 for p in single if p["output_tokens"] == cap),
        "cap_multi_n": len(multi), "cap_multi_out_min": min((p["output_tokens"] for p in multi), default=None),
        "cap_multi_out_max": max((p["output_tokens"] for p in multi), default=None),
        "per_task": per,
    }


def value_text(d: dict) -> str:
    c = d["counts"]
    return (f"{d['n_fail']} of {d['n_tasks']} cloud redo tasks failed ({d['n_correct']} fixed); model {d['model_id']} "
            f"({d['tier']}), max_tokens {d['max_tokens']} per call; transcript sent = the full client-side history "
            f"(0 of {d['n_tasks']} rows with a prior message at the {d['trunc_chars']}-char row limit, longest prior "
            f"assistant content {d['max_prior_content_chars']} chars; rebuilt prompt estimate / local untrimmed estimate "
            f"{d['tok_ratio_min']:.3f} to {d['tok_ratio_max']:.3f}; {d['msgs_min']} to {d['msgs_max']} messages); "
            f"env = state after the local model's earlier turns; failure classes: thinking_empty_at_cap (default thinking "
            f"spent the output budget, empty answer; finish_reason not recorded in v1) {c['thinking_empty_at_cap']} "
            f"({d['cap_single_at_cap']} of {d['cap_single_n']} single-call turns at exactly {d['max_tokens']} output "
            f"tokens, {d['cap_multi_n']} multi-call turns with {d['cap_multi_out_min']} to {d['cap_multi_out_max']} "
            f"output tokens in total), earlier_state_repair {c['earlier_state_repair']}, tool_round_cap "
            f"{c['tool_round_cap']}, grader_issue {c['grader_issue']}, missing_info {c['missing_info']}, "
            f"cloud_model_error {c['cloud_model_error']}")


if __name__ == "__main__":
    d = compute()
    print(value_text(d))
    for p in d["per_task"]:
        if p["cls"] != "thinking_empty_at_cap":
            print(p)
