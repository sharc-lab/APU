"""View D: send one wrong task from the B3 v2 task run to the cloud through the capped client.

The task. build_escalation() (build time) picks one wrong task listed in results/cloud_redo_v1.jsonl (the wrong tasks
of results/x2_evalset_tasks_real_v2.jsonl): tier ollama_ctx_4096, a state-changing task whose reference is writes only,
smallest local prompt. Only the task list is used: the v1 redo itself was a broken protocol run (thinking consumed the
output budget; operator 2026-10-22), so none of its costs, seconds or answers are shown. The rebuild (transcript before
the wrong turn, environment as the local model left it) is scripts/cloud_redo.py's, reused unchanged.

The call. Escalator.run() makes AT MOST ONE provider request per demo run (one Escalator per server process):
  - only when DEMO_CLOUD_LIVE=1 and a real-mode client can be built (ANTHROPIC_API_KEY in this process's environment,
    host allowed by src/cloud/client.py, e.g. CLOUD_CLIENT_HOST_ROLE=controller);
  - through src/cloud/client.py's CloudClient(mode="real") with spend_cap_usd = DEMO_CLOUD_CAP_USD (default 10, never
    above 10) and its own ledger, demo/ledger/cloud_ledger_demo.jsonl (gitignored), not results/cloud_ledger.jsonl;
  - one request with the task's tools offered (no tool round trips); the page shows its cost, seconds and answer, no
    graded result (the redo's fix rate is under diagnosis, operator 2026-10-21).
Anything else (no flag, no key, host refused, cap reached, provider error, a second click) returns the stub: nothing
is sent; the page shows this task's recorded full-history redo (results/cloud_redo_full_context_v1*.jsonl, row
cloud-redo-full-context: USD, seconds, answer, measured) or, without one, "being measured", with the reason. The key is never printed,
logged, written or returned; error text is scrubbed of it before it reaches the page.

CLI (development; one live call at most, same rules):
    py -3.12 -m demo.dashboard.escalate            # stub
    py -3.12 -m demo.dashboard.escalate --live     # needs DEMO_CLOUD_LIVE=1 and the key in this process only
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LEDGER = Path(__file__).resolve().parent.parent / "ledger" / "cloud_ledger_demo.jsonl"
SOURCE = "results/x2_evalset_tasks_real_v2.jsonl"
REDO = "results/cloud_redo_v1.jsonl"
REGISTER_ID = "cloud-redo-cost-per-wrong-task"
TIER = "ollama_ctx_4096"
CAP_CEILING_USD = 10.0  # this agent's cap (operator 2026-10-21); env may lower it, never raise it
ENV_LIVE = "DEMO_CLOUD_LIVE"
ENV_CAP = "DEMO_CLOUD_CAP_USD"
KEY_ENV = "ANTHROPIC_API_KEY"
PRICED = "priced, not executed"
MEASURED = "measured"
BEING_MEASURED = "being measured"
RECORDED = "recorded"
FULL_MAX_TOKENS = 2048  # the full-history redo protocol (row cloud-redo-full-context)
FIX_COST_TEXT = "Cost to fix a wrong answer: being measured"
V1_NOTE = ("The earlier cloud redo (results/cloud_redo_v1.jsonl) was a broken protocol run: thinking consumed the "
           "output budget (docs/FINDINGS.md, cloud redo diagnosis). Its cost and answers are not shown.")


def cap_usd(env=None) -> float:
    env = os.environ if env is None else env
    try:
        v = float(env.get(ENV_CAP, CAP_CEILING_USD))
    except (TypeError, ValueError):
        v = CAP_CEILING_USD
    return max(0.0, min(v, CAP_CEILING_USD))


def _cloud_redo():
    """scripts/cloud_redo.py (and the harness modules it imports), quietly: the session builder prints progress."""
    for p in (REPO / "scripts", REPO / "harness", REPO / "src", REPO):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    with contextlib.redirect_stdout(io.StringIO()):
        import cloud_redo
    return cloud_redo


def _rebuild(cr, src_rows: list[dict], attempt_id: str, turn_idx: int):
    row = next(r for r in src_rows if r.get("record") == "r2a_turn" and r.get("attempt_id") == attempt_id
               and r.get("turn_idx") == turn_idx)
    with contextlib.redirect_stdout(io.StringIO()):
        es = cr.build_session_for(row, {})
        prior = [x for x in cr.session_rows(src_rows, attempt_id) if x["turn_idx"] < turn_idx]
        built = cr.rebuild(es, prior, turn_idx)
    return row, es, built


def _line_of(path: Path, pred) -> int | None:
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            if line.strip() and pred(json.loads(line)):
                return i
    return None


def build_escalation(repo: Path = REPO) -> dict:
    """Pick the task and snapshot what the page shows before any call (task text, local wrong answer)."""
    if not (repo / REDO).exists() or not (repo / SOURCE).exists():
        return {"available": False, "reason": f"{REDO} or {SOURCE} missing"}
    cr = _cloud_redo()
    redo = [r for r in cr.load_rows(repo / REDO) if r.get("record") == "cloud_redo_task" and r.get("tier") == TIER]
    src_rows = cr.load_rows(repo / SOURCE)
    best = None
    turn_rows = {(x.get("attempt_id"), x.get("turn_idx")): x for x in src_rows if x.get("record") == "r2a_turn"}

    def local_prompt(r):  # the local run's own prompt tokens: stable whether the redo file is priced or executed
        return sum(p for p, _o in cr.local_call_tokens(turn_rows.get((r["attempt_id"], r["turn_idx"]), {})))

    for r in sorted(redo, key=lambda r: (local_prompt(r), r["seed"], r["turn_idx"])):
        if r.get("history_truncated_messages"):
            continue
        row, es, built = _rebuild(cr, src_rows, r["attempt_id"], r["turn_idx"])
        t = built["item"].task
        if t is None or t.is_info or not t.reference or any(c["name"] in cr.E.READ_ONLY for c in t.reference):
            continue
        best = (r, row, es, built)
        break
    if best is None:
        return {"available": False, "reason": "no single-call gradable wrong task in the redo file"}
    r, row, es, built = best
    local_calls = [c for rec in row.get("calls") or [] for c in rec.get("tool_calls_parsed") or []]
    return {
        "available": True, "tier": r["tier"], "seed": r["seed"], "turn_idx": r["turn_idx"], "task_id": r["task_id"],
        "attempt_id": r["attempt_id"], "local_model": r["local_model_id"], "cloud_model": r["model_id"],
        "user_message": built["messages"][-1]["content"], "n_messages": len(built["messages"]),
        "expected_calls": [{"name": c["name"], "arguments": c.get("arguments")} for c in built["item"].task.reference],
        "local": {"final_text": (row.get("final_text") or "")[:1500], "tool_calls": local_calls,
                  "end_state_ok": row.get("end_state_ok"), "end_state_diff": row.get("end_state_diff"),
                  "any_http_error": row.get("any_http_error"), "loaded_context": row.get("loaded_context"),
                  "src": {"file": SOURCE, "line": _line_of(repo / SOURCE, lambda x: x.get("record") == "r2a_turn"
                                                          and x.get("attempt_id") == r["attempt_id"]
                                                          and x.get("turn_idx") == r["turn_idx"])}},
        # No figure, answer or source line from the v1 redo (operator 2026-10-22: a broken protocol run, thinking
        # consumed the output budget); the v1 file only supplies the list of wrong tasks. The task's redo shown is the
        # full-history redo (row cloud-redo-full-context), or "being measured" if it has no row for this task.
        "redo": full_redo_of(repo, r["attempt_id"], r["turn_idx"]),
        "selection": f"tier {TIER}, state-changing task with a writes-only reference (one call is gradable), "
                     "smallest local prompt among the redo file's wrong tasks",
    }


FULL_REDO_FILES = ("results/cloud_redo_full_context_v1.jsonl", "results/cloud_redo_full_context_v1_part2.jsonl",
                   "results/cloud_redo_full_context_v1_part3.jsonl")
FULL_REDO_NOTE = ("Full-history redo on the cloud model (row cloud-redo-full-context): the whole transcript, thinking "
                  "between tools, up to 2048 output tokens per call (the local run had 384).")


def full_redo_of(repo: Path, attempt_id: str, turn_idx: int) -> dict:
    """This task's row in the full-history redo files (the last one wins: later parts supersede earlier rows):
    USD (cache-priced), seconds and answer, labelled measured; 'being measured' if there is none."""
    hit = None
    for rel in FULL_REDO_FILES:
        p = repo / rel
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("record") == "cloud_redo_task" and r.get("attempt_id") == attempt_id \
                        and r.get("turn_idx") == turn_idx and r.get("mode") == "executed":
                    hit = (rel, i, r)
    if not hit:
        return {"label": BEING_MEASURED, "text": FIX_COST_TEXT, "note": V1_NOTE}
    rel, line, r = hit
    return {"mode": "executed", "label": MEASURED, "usd": r.get("usd"), "seconds": r.get("seconds"),
            "cloud_final_text": (r.get("cloud_final_text") or "")[:1500] or None, "model": r.get("model_id"),
            "note": FULL_REDO_NOTE, "src": {"file": rel, "line": line}}


def _scrub(text: str, env=None) -> str:
    env = os.environ if env is None else env
    s = str(text)
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "API_KEY"):
        k = env.get(var)
        if k and len(k) >= 8:
            s = s.replace(k, "[redacted]")
    return s[:400]


def _default_client_factory(cap: float, ledger: Path):
    from src.cloud import client as cc  # noqa: PLC0415
    return cc.CloudClient(mode="real", spend_cap_usd=cap, ledger_path=ledger)


def ledger_total(ledger: Path = LEDGER) -> float:
    """Sum of cost_usd over the demo ledger's call rows (alerts excluded), as CloudClient counts it."""
    if not ledger.exists():
        return 0.0
    total = 0.0
    for line in ledger.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("record") != "alert":
            total += float(row.get("cost_usd") or 0.0)
    return total


class Escalator:
    """One per server process (= one demo run): at most `max_live_calls` provider requests over its lifetime."""

    def __init__(self, task: dict, repo: Path = REPO, ledger: Path = LEDGER, env=None, client_factory=None,
                 max_live_calls: int = 1):
        self.task = task
        self.repo = Path(repo)
        self.ledger = Path(ledger)
        self.env = os.environ if env is None else env
        self.client_factory = client_factory or _default_client_factory
        self.max_live_calls = max_live_calls
        self.live_calls = 0
        self.last_live = None  # the run's live result, returned again on a later click
        self._lock = threading.Lock()

    def live_requested(self) -> bool:
        return str(self.env.get(ENV_LIVE, "")).strip() == "1"

    def status(self) -> dict:
        return {"live_requested": self.live_requested(), "key_present": bool(self.env.get(KEY_ENV)),
                "cap_usd": cap_usd(self.env), "live_calls_used": self.live_calls,
                "live_calls_allowed": self.max_live_calls, "ledger": self._rel(self.ledger),
                "ledger_total_usd": ledger_total(self.ledger)}

    def _rel(self, p: Path) -> str:
        try:
            return p.resolve().relative_to(self.repo.resolve()).as_posix()
        except ValueError:
            return p.name

    def stub(self, reason: str) -> dict:
        """No live call: the task's full-history redo (measured) if there is one, else 'being measured'. Nothing from
        the v1 redo is ever shown."""
        rd = self.task.get("redo") or {}
        if rd.get("usd") is not None:
            return {"mode": "recorded redo", "label": MEASURED,
                    "reason": f"{reason}; showing this task's recorded full-history cloud redo",
                    "answer": rd.get("cloud_final_text"), "tool_calls": None, "seconds": rd.get("seconds"),
                    "usd": rd["usd"], "usd_src": rd.get("src"), "usd_basis": rd.get("note"), **self.status()}
        return {"mode": BEING_MEASURED, "label": BEING_MEASURED, "reason": f"{reason}; {FIX_COST_TEXT}",
                "answer": None, "tool_calls": None, "seconds": None, "usd": None, "usd_src": None,
                "usd_basis": FIX_COST_TEXT, **self.status()}

    def run(self, protocol: str = "single") -> dict:
        """One live escalation per demo run, shared by view D (protocol 'single': one request, no tool round trips)
        and story step 4's button (protocol 'full': the full-history redo protocol of row cloud-redo-full-context,
        scripts/cloud_redo.py live_turn with thinking between tools and max_tokens 2048). A later click returns the
        result already obtained (or the recorded fallback if the one attempt failed)."""
        if not self.task.get("available"):
            return self.stub(self.task.get("reason") or "no task")
        if not self.live_requested():
            return self.stub(f"live calls are off ({ENV_LIVE} is not 1)")
        with self._lock:
            if self.live_calls >= self.max_live_calls:
                if self.last_live is not None:
                    return {**self.last_live, "reason": "the one live call for this demo run was already made; "
                                                        "showing its result", "repeat": True, **self.status()}
                return self.stub(f"the one live call for this demo run is used ({self.live_calls}/"
                                 f"{self.max_live_calls}); restart the dashboard for another")
            self.live_calls += 1  # counted before sending: a failed attempt still uses the run's one call
        try:
            res = self._live_full() if protocol == "full" else self._live()
        except Exception as e:  # noqa: BLE001 - every failure falls back to the stub
            return self.stub(f"live call failed: {type(e).__name__}: {_scrub(e, self.env)}")
        if res.get("mode") == "executed":
            self.last_live = res
        return res

    def _live_full(self) -> dict:
        """The full-history redo protocol: the task's whole transcript, the tool loop of scripts/cloud_redo.py
        (live_turn), thinking per THINKING_OFF (between tools on claude-sonnet-5-5) and max_tokens 2048."""
        client = self.client_factory(cap_usd(self.env), self.ledger)
        if getattr(client, "stub_mode", True):
            return self.stub("no real-mode client (no key in this process)")
        cr = _cloud_redo()
        t = self.task
        src_rows = cr.load_rows(self.repo / SOURCE)
        _row, es, built = _rebuild(cr, src_rows, t["attempt_id"], t["turn_idx"])
        if built["item"].task is None or built["item"].task.task_id != t["task_id"]:
            raise RuntimeError("rebuilt task does not match the cached task")
        model = t["cloud_model"]
        rates = cr.model_rates(cr.load_price_table(), model)
        res = cr.live_turn(client, model, es, built, rates, FULL_MAX_TOKENS, cr.THINKING_OFF.get(model))
        return {"mode": "executed", "label": MEASURED, "protocol": "full",
                "reason": f"one live fix through src/cloud/client.py ({FULL_REDO_NOTE})",
                "answer": res.get("cloud_final_text"), "tool_calls": None, "seconds": res.get("seconds"),
                "usd": res.get("usd"), "usd_ledger": res.get("usd_ledger"), "n_calls": res.get("n_calls"),
                "usd_basis": "cache-priced on the provider's usage (as row cloud-redo-full-context)",
                **self.status()}

    def recorded(self, reason: str) -> dict:
        """Story step 4's fallback: this task's recorded full-history redo, tagged 'recorded'."""
        out = self.stub(reason)
        if out.get("usd") is not None:
            out = {**out, "label": RECORDED, "mode": "recorded redo"}
        return out

    def _live(self) -> dict:
        client = self.client_factory(cap_usd(self.env), self.ledger)
        if getattr(client, "stub_mode", True):
            return self.stub("no real-mode client (no key in this process)")
        cr = _cloud_redo()
        t = self.task
        src_rows = cr.load_rows(self.repo / SOURCE)
        _row, es, built = _rebuild(cr, src_rows, t["attempt_id"], t["turn_idx"])
        if built["item"].task is None or built["item"].task.task_id != t["task_id"]:
            raise RuntimeError("rebuilt task does not match the cached task")
        tools = cr.EV.tools_payload(es.design)
        prepared = client.prepare(model_id=t["cloud_model"], messages=cr.with_cache_control(built["messages"]),
                                  tools=tools, max_tokens=cr.ag.MAX_TOKENS_PER_CALL, custom_llm_provider="anthropic")
        t0 = time.perf_counter()
        raw = prepared.send()
        seconds = time.perf_counter() - t0
        res = prepared.settle(raw)
        resp = res.raw
        content, calls = cr._parse_reply(resp)
        # Not graded on screen: the redo's fix rate is under diagnosis (operator 2026-10-21); cost, seconds, answer.
        u = cr._usage(resp)
        rates = cr.model_rates(cr.load_price_table(), t["cloud_model"])
        usd_cache = cr.call_cost(rates, u["uncached_input_tokens"], u["cache_write_tokens"], u["cache_read_tokens"],
                                 u["output_tokens"], u["input_tokens"])
        return {"mode": "executed", "label": MEASURED, "reason": "one live call through src/cloud/client.py",
                "answer": content, "tool_calls": calls, "seconds": seconds, "usd": res.cost_usd,
                "usd_basis": "ledger cost (client.py estimate_cost on the provider's usage, input uncached)",
                "usd_with_cache_pricing": usd_cache, "usage": u, **self.status()}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--live", action="store_true", help="one live call (needs DEMO_CLOUD_LIVE=1 and the key)")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(REPO))
    cache = Path(__file__).resolve().parent / "cache" / "escalation.json"
    task = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else build_escalation()
    env = dict(os.environ)
    if not a.live:
        env.pop(ENV_LIVE, None)
    out = Escalator(task, env=env).run()
    show = {k: out.get(k) for k in ("mode", "reason", "seconds", "usd", "usd_with_cache_pricing",
                                    "cap_usd", "ledger", "ledger_total_usd", "live_calls_used")}
    show["answer"] = (out.get("answer") or "")[:300]
    show["tool_calls"] = out.get("tool_calls")
    print(json.dumps(show, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
