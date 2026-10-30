"""Adapters from this repo's own recorded runs to audit requests (apu-audit-session/1), for testing the audit offline.

demo_detector_requests  results/demo_detector_check*.jsonl: one request per recorded call (detector_task records), in
                        run order per lane, so the committed server logs results/demo_live_logs/<run>/serve_*.log can
                        be sliced against them. Messages were not recorded: n_messages is None (the audit then takes
                        Ollama's message-drop line as a drop) and prompt_tokens is the server's own count (used_tokens).
r2a_turn_requests       x2_r2_evalset r2a_turn rows that carry a detector v2 verdict per turn (cut_v2_flagged; the own
                        compaction runs): one request per task turn of a completed session, with that verdict as
                        recorded_verdict (the server logs themselves were not kept). prompt_tokens is the turn's
                        final_call_prompt_tokens_calibrated (an estimate, labelled), for the cloud-cost columns.
"""
from __future__ import annotations

DEMO_LANES = {"num_ctx_4096": "ollama_ctx_4096", "default": "ollama_default"}


def demo_detector_requests(rows: list[dict], lane: str) -> list[dict]:
    out = []
    for r in rows:
        if r.get("record") != "detector_task" or r.get("arm") != lane:
            continue
        sid = f"{r['seed']}:{r['task_id']}"
        calls = r["calls"]
        for i, c in enumerate(calls):
            out.append({"session_id": sid, "task_id": r["task_id"], "request_idx": c["n"] - 1, "n_messages": None,
                        "prompt_tokens": c.get("used_tokens"), "prompt_tokens_basis": "server count",
                        "http_status": 200 if c.get("outcome") == "ok" else None,
                        "error": None if c.get("outcome") == "ok" else c.get("outcome"),
                        "ran_at": {"config": DEMO_LANES.get(lane, lane), "window": c["loaded_window"]},
                        "outcome": {"ok": r["end_state_ok"] is True} if i == len(calls) - 1 else None,
                        "recorded_cut_v2": c.get("cut_v2")})
    return out


def r2a_turn_requests(rows: list[dict]) -> list[dict]:
    done = {(r["arm_id"], r["seed"], r.get("attempt_id")) for r in rows if r.get("record") == "r2a_session"}
    out = []
    for r in rows:
        if r.get("record") != "r2a_turn" or r.get("item_kind") != "task":
            continue
        if (r["arm_id"], r["seed"], r.get("attempt_id")) not in done:
            continue
        cfg = r["arm_id"].replace("_call2_notools", "")
        out.append({"session_id": f"{cfg}:{r['seed']}:{r.get('attempt_id')}", "task_id": f"{r['turn_idx']}:{r['task_id']}",
                    "request_idx": r["turn_idx"], "ran_at": {"config": cfg, "window": r.get("loaded_context")},
                    "outcome": {"ok": r.get("end_state_ok") is True},
                    "prompt_tokens": r.get("final_call_prompt_tokens_calibrated"),
                    "prompt_tokens_basis": "calibrated estimate of the final call as sent",
                    "recorded_verdict": {"flagged": bool(r.get("cut_v2_flagged")),
                                         "no_log": bool(r.get("cut_v2_no_log_calls")),
                                         "evidence": r.get("cut_v2_evidence") or []}})
    return out
