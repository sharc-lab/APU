"""Task subsets of B3 v2 (harness/x2_r2_evalset.py, design v2, workload evalset, guardrails off) and their content hash.

The task set is generated from code and a seed, not stored, so its identity is a hash of what the model is actually
sent: the system prompt, the tool schema and every user message in session order, plus each item's id and kind.
Two checkouts that produce the same hash ran the same tasks. The projected task-turn prompt (calibrated tokens from
the task definition alone, as analysis/impact_model.projected_task_prompts) is included per task; the fake runtime
and the fits-locally baseline use it.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

SUBSETS = {
    # id: (seeds, description)
    "b3v2-small": ((20260901,), "B3 v2, 1 session (seed 20260901), 30 task turns"),
    "b3v2-3seed": ((20260901, 20260902, 20260903), "B3 v2, 3 sessions (seeds 20260901-03), 90 task turns"),
}
DESIGN = "v2"
GENERATOR = "harness/x2_r2_evalset.py build_session(seed, workload='evalset', guardrails=False, design='v2')"


def _ev():
    h = str(REPO / "harness")
    if h not in sys.path:
        sys.path.insert(0, h)
    import x2_r2_evalset as ev
    return ev


def session_spec(seed: int) -> dict:
    """The full content of one B3 v2 session as sent to the model, plus the per-task projections."""
    ev = _ev()
    with contextlib.redirect_stdout(io.StringIO()):
        es = ev.build_session(seed, workload="evalset", guardrails=False, design=DESIGN)
    ag = ev.ag
    tools = ev.tools_payload(es.design)
    static = ag.est_tokens(es.system_prompt) + ag.est_tokens(json.dumps(tools))
    cum, turns = 0, []
    for t, it in zip(es.turns, es.items):
        msg = ev.user_message(t, es)
        cum += ag.est_tokens(msg) + ag.PER_MESSAGE_OVERHEAD_TOKENS
        proj = round((static + cum + t.idx * ev.PROJ_NONFILLER_PER_TURN) * ev.PROJ_CALIB) if it.kind == "task" else None
        turns.append({"turn_idx": t.idx, "item_id": it.item_id, "kind": it.kind, "user_message": msg,
                      "task_type": getattr(it.task, "ttype", None), "projected_prompt_tokens": proj})
    return {"seed": seed, "design": es.design, "system_prompt": es.system_prompt, "tools": tools, "turns": turns,
            "proj_calib": ev.PROJ_CALIB, "proj_nonfiller_per_turn": ev.PROJ_NONFILLER_PER_TURN}


def build(subset: str) -> dict:
    if subset not in SUBSETS:
        raise ValueError(f"unknown subset {subset!r}; known: {', '.join(SUBSETS)}")
    seeds, desc = SUBSETS[subset]
    sessions = [session_spec(s) for s in seeds]
    canon = json.dumps([{k: s[k] for k in ("seed", "design", "system_prompt", "tools")}
                        | {"turns": [{k: t[k] for k in ("turn_idx", "item_id", "kind", "user_message")}
                                     for t in s["turns"]]} for s in sessions],
                       sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    n_tasks = sum(1 for s in sessions for t in s["turns"] if t["kind"] == "task")
    return {"id": subset, "description": desc, "design": DESIGN, "seeds": list(seeds), "n_task_turns": n_tasks,
            "sha256": hashlib.sha256(canon.encode("utf-8")).hexdigest(), "generator": GENERATOR,
            "sessions": sessions}
