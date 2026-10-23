"""Live demo engine (2026-10-22 operator brief; decisions of 2026-10-23): one B3 v2 task at its real position in its
session, run LIVE in three lanes, graded by the B3 v2 end-state grader.

Task: (seed, task_id) of the B3 v2 task real run (results/x2_evalset_tasks_real_v2.jsonl, qwen3:14b, design v2,
guardrails off). The session is rebuilt with x2_r2_evalset.build_session; the history before the task turn is rebuilt
from the recorded default-window session of that seed (REPLAYED on screen): each prior user message exactly, each
prior call's assistant message as the local run sent it (content, and the native tool calls when the call's tool calls
were native), each tool result recomputed with evalset_env.apply_call on the environment as that run left it. Two
shapes are built from the same rows: the Ollama shape (x2_r2_agent._assistant_msg / x2_r2_evalset._tool_msgs) for the
local lanes and the OpenAI shape (scripts/cloud_redo.rebuild) for the cloud. history_check compares the Ollama-shape
transcript estimate with the recorded row's transcript_tokens_est (equal unless a recorded content hit the 4000-char cap).

Lanes (scenarios in demo/live/scenarios.json):
  default   every task locally at the scenario's window (num_ctx sent), the runtime's defaults otherwise;
  ours      src/dse/route_at_start.route on the scenario's device profile: local at the chosen window, or the whole
            task on the cloud from its first call; a local task whose detector fired is escalated: the task turn is
            re-run on the cloud with the full history (the full-history redo protocol of scripts/cloud_redo);
  cloud     the whole task on the cloud model, live through src/cloud/client.py, or REPLAYED from
            results/all_cloud_b3_v1.jsonl when a live call is not possible (no key, cap, budget, provider error).

Local turns: the B3 v2 tool loop (x2_r2_evalset._calls_tool_loop, same tools, rounds, options and grader) against the
lane's own Ollama server (LaneServer: own port, own log, OLLAMA_DEBUG=1 for that process only). After every call the
server log slice of that call goes to demo_overflow.detect.

Pure parts are unit-tested with fake runtimes (tests/test_demo_live.py). No cloud key is read here; the cloud client
is passed in by the caller.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

_here = Path(__file__).resolve().parent
if _here.name == "harness" and (_here.parent / "src" / "dse").is_dir():
    REPO = _here.parent
    _front = [str(_here), str(REPO / "scripts"), str(REPO / "src" / "dse"), str(REPO / "src"), str(REPO)]
else:                                   # deployed side directory on evo-x2 (C:\apu\ovn\<dir>), parent holds the rest
    REPO = _here
    _front = [str(_here), str(_here.parent)]
# in this order, ahead of everything else: the side directory's x2_r2_evalset / evalset_env win over the parent's
sys.path[:] = _front + [p for p in sys.path if p not in _front]

import demo_overflow as DO  # noqa: E402
import evalset_env as E  # noqa: E402
import x2_r2_evalset as EV  # noqa: E402  (before x2_r2_agent, which puts its own directory first on sys.path: in a
import x2_r2_agent as ag  # noqa: E402     side directory this keeps the side directory's x2_r2_evalset; see EXPERIMENTS.md
                          #                "AMENDMENT (2026-10-22 ...)" of the long-task job)

MODEL = "qwen3:14b"
DEFAULT_ARM = "ollama_default_call2_notools"
B3_FILE = "x2_evalset_tasks_real_v2.jsonl"
ALL_CLOUD_FILE = "all_cloud_b3_v1.jsonl"
LOG_SETTLE_S = 3.0
LOG_POLL_S = 0.2

LABEL_LIVE, LABEL_REPLAYED, LABEL_MEASURED = "LIVE", "REPLAYED", "MEASURED"


# ── tasks (pure given the rows) ────────────────────────────────────────────────────────────────────────

def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def session_rows(rows: list[dict], seed: int, arm: str = DEFAULT_ARM) -> list[dict]:
    att = {s["attempt_id"] for s in ag.completed_sessions(rows)
           if s.get("mode") == "real" and s["model_id"] == MODEL and s["arm_id"] == arm and s["seed"] == seed}
    if len(att) != 1:
        raise ValueError(f"expected one completed {arm} session for seed {seed}, found {len(att)}")
    a = att.pop()
    return sorted((r for r in rows if r.get("record") == "r2a_turn" and r.get("attempt_id") == a),
                  key=lambda r: r["turn_idx"])


def projected_task_prompt(es, task_id: str) -> tuple[int, int]:
    """(turn_idx, projected final-call prompt) from the task definition alone: the per-turn form of
    x2_r2_evalset.projected_max_prompt, as analysis/impact_model.projected_task_prompts (the headline row
    hybrid-route-at-start-measured routes on exactly this number)."""
    static = ag.est_tokens(es.system_prompt) + ag.est_tokens(json.dumps(EV.tools_payload(es.design)))
    cum = 0
    for t, it in zip(es.turns, es.items):
        cum += ag.est_tokens(EV.user_message(t, es)) + ag.PER_MESSAGE_OVERHEAD_TOKENS
        if it.kind == "task" and it.item_id == task_id:
            return t.idx, round((static + cum + t.idx * EV.PROJ_NONFILLER_PER_TURN) * EV.PROJ_CALIB)
    raise KeyError(task_id)


def _ollama_history(es, prior: list[dict]):
    by_idx = {t.idx: t for t in es.turns}
    messages = [{"role": "system", "content": es.system_prompt}]
    env = copy.deepcopy(es.env0)
    capped = 0
    for r in prior:
        messages.append({"role": "user", "content": EV.user_message(by_idx[r["turn_idx"]], es)})
        for c in r.get("calls") or []:
            content = c.get("content") or ""
            capped += len(content) >= 4000
            calls = c.get("tool_calls_parsed") or []
            resp = {"message": content, "tool_calls": c.get("native_tool_calls")}
            method = c.get("tool_detection_method") or ("native" if c.get("native_tool_calls") else "none")
            messages.append(ag._assistant_msg(resp, method if calls else "none"))
            if calls:
                messages.extend(EV._tool_msgs(calls, [E.apply_call(env, x) for x in calls]))
    return messages, env, capped


def build_task(rows: list[dict], seed: int, task_id: str, with_cloud: bool = True) -> dict:
    """Everything a lane needs for one task. The session's prior history is REPLAYED from the recorded run.
    with_cloud=False (evo-x2, no cloud client there) skips the OpenAI-shape rebuild ("built": None)."""
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):     # build_session prints its filler calibration
        es = EV.build_session(seed, None, "evalset", False, MODEL, None, "v2")
    turn_idx, proj = projected_task_prompt(es, task_id)
    srows = session_rows(rows, seed)
    prior = [r for r in srows if r["turn_idx"] < turn_idx]
    rec = next(r for r in srows if r["turn_idx"] == turn_idx)
    if rec.get("task_id") != task_id:
        raise ValueError(f"seed {seed} turn {turn_idx} is {rec.get('task_id')}, not {task_id}")
    msgs, env, capped = _ollama_history(es, prior)
    recorded_est = prior[-1]["transcript_tokens_est"] if prior else ag.transcript_tokens(msgs[:1])
    turn = next(t for t in es.turns if t.idx == turn_idx)
    item = next(it for t, it in zip(es.turns, es.items) if t.idx == turn_idx)
    built = None
    if with_cloud:
        import cloud_redo as CR
        built = CR.rebuild(es, prior, turn_idx)    # OpenAI shape, same rows, for the cloud lanes
        if built["env_before"] != env:
            raise ValueError("the two history rebuilds disagree on env_before")
    return {"seed": seed, "task_id": task_id, "turn_idx": turn_idx, "projected_prompt": proj, "es": es,
            "turn": turn, "item": item, "history": msgs, "env_before": env, "built": built,
            "history_turns": len(prior), "history_check": {"rebuilt_est": ag.transcript_tokens(msgs),
                                                           "recorded_est": recorded_est, "capped_contents": capped},
            "task_type": item.task.ttype if item.task else None, "task_text": item.text,
            "recorded": {a: _recorded_outcome(rows, seed, task_id, a) for a in
                         sorted({r["arm_id"] for r in rows if r.get("record") == "r2a_turn"})}}


def _recorded_outcome(rows, seed, task_id, arm):
    for r in rows:
        if (r.get("record") == "r2a_turn" and r.get("arm_id") == arm and r.get("seed") == seed
                and r.get("task_id") == task_id and r.get("model_id") == MODEL):
            return r.get("end_state_ok")
    return None


# ── local lane ─────────────────────────────────────────────────────────────────────────────────────────

class LaneRuntime(ag.OllamaRuntime):
    """x2_r2_agent's runtime pointed at a lane's own server; every call's server log slice goes to the detector and
    the verdict rides on the response (resp["overflow"]), from where on_call() streams it."""

    def __init__(self, base: str, server_log: str | None, ratio: float, on_call=None, clock=time.monotonic,
                 sleep=time.sleep):
        super().__init__(base=base)
        self.server_log, self.ratio, self.on_call = server_log, ratio, on_call
        self.clock, self.sleep = clock, sleep
        self.calls = []

    def _log_size(self):
        try:
            return Path(self.server_log).stat().st_size if self.server_log else 0
        except OSError:
            return 0

    def _slice(self, start: int) -> str | None:
        if not self.server_log:
            return None
        deadline = self.clock() + LOG_SETTLE_S
        text = ""
        while True:
            with open(self.server_log, "rb") as fh:
                fh.seek(start)
                text = fh.read().decode("utf-8", errors="replace")
            if "stop processing:" in text or "/api/chat" in text or self.clock() >= deadline:
                return text
            self.sleep(LOG_POLL_S)

    def chat(self, model, messages, num_ctx, tools, think, extra=None):
        start = self._log_size()
        t0 = self.clock()
        resp = self._post_native(ag.native_chat_body(model, messages, num_ctx, tools, think, extra))
        own = round(ag.prompt_tokens_est(messages, tools) * self.ratio)
        loaded = None
        try:
            loaded = self.loaded_context(model)
        except Exception:
            pass
        v = DO.detect(own, resp.get("prompt_eval_count"), self._slice(start), loaded, n_messages=len(messages))
        rec = {"n": len(self.calls) + 1, "seconds": round(self.clock() - t0, 2), "own_tokens": own,
               "used_tokens": v["used_tokens"], "loaded_window": loaded, "cut": v["cut"], "evidence": v["evidence"], "cut_v2": v.get("cut_v2"), "evidence_v2": v.get("evidence_v2"),
               "method": v["method"], "prompt_eval_count": resp.get("prompt_eval_count"),
               "tool_calls": [tc.get("function", {}).get("name") for tc in (resp.get("tool_calls") or [])],
               "outcome": resp.get("outcome")}
        self.calls.append(rec)
        resp["overflow"] = v
        if self.on_call:
            self.on_call(rec)
        return resp

    def _ps(self) -> list[dict]:
        """GET /api/ps on this lane's server (x2_r2_agent.OllamaRuntime.client is fixed to port 11434)."""
        import urllib.request
        with urllib.request.urlopen(self.base + "/api/ps", timeout=10) as r:
            return json.loads(r.read()).get("models", [])

    def loaded_context(self, model):
        for m in self._ps():
            if ag.same_tag(m.get("name") or m.get("model"), model):
                ctx = m.get("context_length")
                return int(ctx) if ctx is not None else None
        return None

    def unload(self, model, wait_s=30):
        body = json.dumps({"model": model, "keep_alive": 0}).encode()
        import urllib.request
        try:
            req = urllib.request.Request(self.base + "/api/generate", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                r.read()
        except Exception:
            return False
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            if not any(ag.same_tag(m.get("name") or m.get("model"), model) for m in self._ps()):
                return True
            time.sleep(0.5)
        return False

    def recover(self):
        return False


def run_local(rt: LaneRuntime, task: dict, num_ctx: int | None, model: str = MODEL) -> dict:
    """The task turn in the B3 v2 tool loop on rt; graded with the end-state grader."""
    es, turn, item = task["es"], task["turn"], task["item"]
    messages = copy.deepcopy(task["history"])
    messages.append({"role": "user", "content": EV.user_message(turn, es)})
    rt.calls = []
    t0 = time.monotonic()
    recs, calls_by_call, final_text, extra = EV._calls_tool_loop(rt, model, num_ctx, ag.MODELS.get(model), es,
                                                                 task["env_before"], messages, lambda *_: None)
    scored = EV.score_turn(es, item, turn, task["env_before"], calls_by_call, final_text)
    scored.pop("_env_after", None)
    calls = rt.calls
    return {"route": "local", "num_ctx": num_ctx, "end_state_ok": scored["end_state_ok"],
            "answer_ok": scored["answer_ok"], "state_ok": scored["state_ok"], "flagged": any(c["cut"] for c in calls), "flagged_v2": any(c.get("cut_v2") for c in calls),
            "calls": calls, "n_calls": len(recs), "tool_calls": [c.get("name") for cs in calls_by_call for c in cs],
            "final_text": (final_text or "")[:1000], "seconds": round(time.monotonic() - t0, 2),
            "max_used_tokens": max((c["used_tokens"] or 0 for c in calls), default=0),
            "max_own_tokens": max((c["own_tokens"] for c in calls), default=0),
            "any_http_error": any(c["outcome"] != "ok" for c in calls), **extra}


# ── cloud ──────────────────────────────────────────────────────────────────────────────────────────────

def run_cloud(client, task: dict, model: str, rates: dict) -> dict:
    """The task turn on the cloud with the full (REPLAYED) history, the full-history redo protocol of
    scripts/cloud_redo (tools on every call, at most 4 tool rounds, thinking off, max_tokens 2048). LIVE."""
    import cloud_redo as CR
    t0 = time.monotonic()
    r = CR.live_turn(client, model, task["es"], task["built"], rates, 2048, CR.THINKING_OFF.get(model))
    return {"route": "cloud", "model": model, "label": LABEL_LIVE, "end_state_ok": r["cloud_end_state_ok"],
            "answer_ok": r["cloud_answer_ok"], "state_ok": r["cloud_state_ok"], "usd_ledger": r["usd_ledger"],
            "usd_cache_priced": r["usd"], "seconds": round(time.monotonic() - t0, 2), "provider_seconds": r["seconds"],
            "n_calls": r["n_calls"], "final_text": (r["cloud_final_text"] or "")[:1000],
            "tool_calls_n": r["cloud_n_tool_calls"], "empty_at_cap": r["empty_at_cap"]}


def replayed_cloud(all_cloud_rows: list[dict], task: dict, model: str) -> dict | None:
    """The recorded all-cloud result of the same (seed, task) on the same model (all_cloud_b3_v1; that run kept the
    cloud's own history, not the local one): REPLAYED."""
    done = {s.get("attempt_id") for s in all_cloud_rows if s.get("record") == "cloud_session"}
    for r in all_cloud_rows:
        if (r.get("record") == "r2a_turn" and r.get("attempt_id") in done and r.get("seed") == task["seed"]
                and r.get("task_id") == task["task_id"] and r.get("model_id") == model):
            return {"route": "cloud", "model": model, "label": LABEL_REPLAYED, "end_state_ok": r.get("end_state_ok"),
                    "usd_ledger": r.get("usd_ledger"), "usd_cache_priced": r.get("usd"),
                    "seconds": r.get("seconds"), "final_text": (r.get("final_text") or "")[:1000],
                    "source": "results/" + ALL_CLOUD_FILE}
    return None
