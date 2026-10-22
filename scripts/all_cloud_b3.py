"""All-cloud baseline on the B3 v2 tasks: the 30 tasks run fully on the cloud model (all_cloud_b3_v1).

Pre-registered in docs/FINDINGS.md "PRE-REGISTRATION: all-cloud baseline on the B3 v2 tasks ..." (2026-10-21). The
all-cloud reference quality and cost for the hybrid savings model (savings = all-cloud cost minus hybrid cost at
matched quality).

Protocol: the B3 v2 tasks plan session (harness/x2_r2_evalset.py::build_session, workload evalset, guardrails off,
design v2: 30 task turns, 150 filler tokens per turn) run turn by turn with x2_r2_evalset.run_turn unchanged (tool loop,
tools on every call, MAX_TOOL_ROUNDS tool-bearing calls then one final call; scoring by score_turn, so "correct" is
evalset_env.check_task's end_state_ok). CloudRuntime stands in for the Ollama runtime: it converts the harness
transcript to the provider's tool-call shape and sends it through src/cloud/client.py (litellm, custom_llm_provider
"anthropic"). Declared differences, forced by the provider: the final call after the round cap keeps the tool
definitions with tool_choice "none"; no context window applies. max_tokens 384 (as local), temperature 0 (as local).

Prompt caching on: breakpoints on the system prompt and the last user message (as scripts/cloud_redo.py). USD per task
is priced from the provider usage at the price file's Standard rates (uncached input, 5-minute cache write, cache hit,
output); usd_ledger is client.py's estimate (all input at the input rate), the basis of the cap check.

Budget: CloudClient spend_cap_usd = ledger running total at job start + --job-cap (15.0). Before any measured turn: one
warm-up call per model (excluded), then a pilot (turn 1 of the first seed, a fresh session, excluded) per model, and a
ledger-basis projection of the full job from the pilot (see project()). Over the job cap: stop, report, no reduced plan.

Operator command (controller laptop, from the repo root; the key is loaded into this one process only):
    $env:ANTHROPIC_API_KEY = [Environment]::GetEnvironmentVariable('ANTHROPIC_API_KEY','User')
    $env:CLOUD_CLIENT_HOST_ROLE = 'controller'
    py -3.12 scripts/all_cloud_b3.py --live --ledger <main checkout>/results/cloud_ledger.jsonl
        --redo-file <main checkout>/results/cloud_redo_v1.jsonl
    Remove-Item Env:\\ANTHROPIC_API_KEY
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import statistics
import sys
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "harness", REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import x2_r2_agent as ag  # noqa: E402
import x2_r2_evalset as EV  # noqa: E402
from cloud import client as cc  # noqa: E402

_spec = importlib.util.spec_from_file_location("cloud_redo", REPO / "scripts" / "cloud_redo.py")
CR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CR)

OUT = REPO / "results" / "all_cloud_b3_v1.jsonl"
PRICE_FILE = REPO / "results" / "cloud_prices_anthropic_20261014.json"
LOCAL_SOURCE = REPO / "results" / "x2_evalset_tasks_real_v2.jsonl"
REDO_FILE = REPO / "results" / "cloud_redo_v1.jsonl"
HARNESS = "all_cloud_b3"
SEEDS = (20260901, 20260902, 20260903)
JOB_CAP_USD = 15.0
MARGIN = 1.5
LOCAL_REF_ARM = "ollama_default_call2_notools"
DESIGN = "v2"
ARM_ID = "cloud_all"
KEY_ENV = cc.ENV_VAR_ANTHROPIC_API_KEY
COST_KEYS = ("input_tokens", "uncached_input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens",
             "usd", "usd_all_uncached", "usd_ledger", "seconds")


def models_from(table: dict) -> tuple[str, str]:
    """(cheapest, mid tier) from the shared price file's selection."""
    return table["selection"]["cheap_small"], table["selection"]["escalation_mid_tier"]


# ── transcript conversion ───────────────────────────────────────────────────────────────────────────

def to_provider_messages(messages: list[dict]) -> list[dict]:
    """Harness transcript (Ollama shape) to the OpenAI shape litellm takes. Each native assistant tool call gets a
    deterministic id (message index, call index) so the prefix is byte-identical across calls (cache hits); its tool
    results follow it in order. A tool result with no pending native call (a call written as text) becomes a user
    message carrying the result."""
    out, pending = [], []
    for i, m in enumerate(messages):
        role = m.get("role")
        if role == "assistant":
            tcs = []
            for j, tc in enumerate(m.get("tool_calls") or []):
                call = ag._as_call(tc) or {"name": "unknown", "arguments": {}}
                args = call.get("arguments")
                tcs.append({"id": f"call_{i}_{j}", "type": "function",
                            "function": {"name": call.get("name") or "unknown",
                                         "arguments": json.dumps(args if isinstance(args, dict) else {})}})
            msg = {"role": "assistant", "content": m.get("content") or ""}
            if tcs:
                msg["tool_calls"] = tcs
            out.append(msg)
            pending = [t["id"] for t in tcs]
        elif role == "tool":
            if pending:
                out.append({"role": "tool", "tool_call_id": pending.pop(0), "content": m.get("content") or ""})
            else:
                out.append({"role": "user", "content": f"Tool result ({m.get('tool_name') or 'tool'}): "
                                                       f"{m.get('content') or ''}"})
        else:
            out.append({"role": role, "content": m.get("content") or ""})
            pending = []
    return out


def has_tool_calls(pmsgs: list[dict]) -> bool:
    return any(m.get("tool_calls") or m.get("role") == "tool" for m in pmsgs)


# ── cloud runtime (stands in for x2_r2_agent.OllamaRuntime) ─────────────────────────────────────────

class CloudRuntime:
    """chat() has OllamaRuntime.chat's signature and return shape. Every call goes through client.prepare / send /
    settle; per-call usage, priced cost and send() seconds accumulate in self.calls until drain()."""

    def __init__(self, client, model_id: str, rates: dict, max_tokens: int = ag.MAX_TOKENS_PER_CALL,
                 temperature: float | None = 0.0, log=print, retry_wait_s: float = 10.0):
        self.client, self.model_id, self.rates = client, model_id, rates
        self.max_tokens, self.temperature, self.log, self.retry_wait_s = max_tokens, temperature, log, retry_wait_s
        self.calls: list[dict] = []

    def drain(self) -> list[dict]:
        out, self.calls = self.calls, []
        return out

    def recover(self):
        return False

    def loaded_context(self, model):
        return None

    def unload(self, model, wait_s=30):
        return None

    def _send(self, pmsgs, kw):
        prepared = self.client.prepare(model_id=self.model_id, messages=pmsgs, **kw)
        t0 = time.perf_counter()
        raw = prepared.send()
        secs = time.perf_counter() - t0
        return prepared.settle(raw), secs

    def chat(self, model, messages, num_ctx, tools, think, extra=None):
        pmsgs = CR.with_cache_control(to_provider_messages(messages))
        kw = {"max_tokens": self.max_tokens, "custom_llm_provider": "anthropic"}
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        if tools is not None:
            kw["tools"] = tools
        elif has_tool_calls(pmsgs):   # final call after the round cap: tools stay defined, choice none
            kw["tools"] = EV.tools_payload(DESIGN)
            kw["tool_choice"] = "none"
        err = None
        for attempt in (1, 2):
            try:
                res, secs = self._send(pmsgs, kw)
                break
            except cc.SpendCapExceeded:
                raise
            except Exception as e:   # provider error: one retry, then an error outcome (the turn records it)
                err = f"{type(e).__name__}: {str(e)[:300]}"
                self.log(f"  provider error on attempt {attempt}: {err}")
                if attempt == 1:
                    time.sleep(self.retry_wait_s)
        else:
            return {"outcome": "error", "status": 0, "error": err, "duration_s": None}
        resp = res.raw
        u = CR._usage(resp)
        usd = CR.call_cost(self.rates, u["uncached_input_tokens"], u["cache_write_tokens"], u["cache_read_tokens"],
                           u["output_tokens"], u["input_tokens"])
        usd_unc = CR.call_cost(self.rates, u["input_tokens"], 0, 0, u["output_tokens"], u["input_tokens"])
        self.calls.append({**u, "usd": usd, "usd_all_uncached": usd_unc, "usd_ledger": res.cost_usd,
                           "seconds": secs, "tool_choice": kw.get("tool_choice")})
        choice = (resp.get("choices") or [{}])[0] or {}
        msg = choice.get("message") or {}
        native = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            native.append({"function": {"name": fn.get("name"), "arguments": ag._norm_args(fn.get("arguments"))}})
        return {"outcome": "ok", "status": 200, "message": msg.get("content") or "", "tool_calls": native or None,
                "prompt_eval_count": u["input_tokens"], "eval_count": u["output_tokens"], "duration_s": secs,
                "raw": {"message": {"content": msg.get("content") or ""}, "done_reason": choice.get("finish_reason")}}


# ── session ─────────────────────────────────────────────────────────────────────────────────────────

def sum_calls(calls: list[dict]) -> dict:
    return {k: sum(c[k] for c in calls) for k in COST_KEYS}


def run_cloud_session(runtime: CloudRuntime, seed: int, emit, log, max_turns: int | None = None,
                      record: str = "r2a_turn", spend: dict | None = None) -> dict:
    """One B3 v2 tasks session on the cloud: x2_r2_evalset.run_turn per turn, transcript carried across turns."""
    es = EV.build_session(seed, max_turns, "evalset", False, runtime.model_id, None, DESIGN)
    attempt = uuid.uuid4().hex[:12]
    base = {"mode": "real", "model_id": runtime.model_id, "arm_id": ARM_ID, "num_ctx_requested": None, "seed": seed,
            "think": None, "call2_mode": "off", "attempt_id": attempt, "session_code": es.session_code,
            "harness": HARNESS, "workload": "evalset", "guardrails": False, "design": DESIGN,
            "turn_protocol": EV.TURN_PROTOCOL[DESIGN], "prompt_caching": True}
    spend = spend if spend is not None else {"usd": 0.0, "usd_ledger": 0.0}
    messages = [{"role": "system", "content": es.system_prompt}]
    env = copy.deepcopy(es.env0)
    rows = []
    runtime.drain()
    for turn, item in zip(es.turns, es.items):
        t0 = time.monotonic()
        row, env = EV.run_turn(runtime, runtime.model_id, None, None, es, turn, item, env, messages, log, "off")
        calls = runtime.drain()
        cost = sum_calls(calls)
        spend["usd"] += cost["usd"]
        spend["usd_ledger"] += cost["usd_ledger"]
        row.update(base)
        row.update(cost)
        row.update({"record": record, "ts_utc": ag.utc_iso(), "turn_wall_s": time.monotonic() - t0,
                    "cloud_calls": calls, "job_spend_usd": spend["usd"], "job_spend_usd_ledger": spend["usd_ledger"],
                    "ledger_total_usd": runtime.client.running_total_usd, "over_loaded_window": False,
                    "loaded_context": None})
        emit(row)
        rows.append(row)
        log(f"  {runtime.model_id} seed={seed} turn {turn.idx}/{len(es.turns)} {item.item_id} "
            f"end_state={row['end_state_ok']} calls={row['n_calls']} task ${cost['usd']:.4f} "
            f"(ledger ${cost['usd_ledger']:.4f}) {cost['seconds']:.1f}s | job ${spend['usd']:.4f} "
            f"(ledger ${spend['usd_ledger']:.4f}) | ledger total ${runtime.client.running_total_usd:.4f}")
    task_rows = [r for r in rows if r.get("item_kind") == "task"]
    s = {**base, "record": "cloud_session" if record == "r2a_turn" else record + "_session", "ts_utc": ag.utc_iso(),
         "n_turns": len(rows), "n_task_turns": len(task_rows),
         "n_end_state_ok": sum(1 for r in task_rows if r.get("end_state_ok")),
         **{k: sum(r[k] for r in rows) for k in COST_KEYS}}
    emit(s)
    return s


# ── projection ──────────────────────────────────────────────────────────────────────────────────────

def local_sessions(rows: list[dict], seeds=SEEDS, arm=LOCAL_REF_ARM) -> dict:
    """seed -> list of (prompt_tokens_est, completion_tokens) per call of the local qwen3:14b ollama_default session."""
    done = {s["attempt_id"] for s in ag.completed_sessions(rows) if s.get("mode") == "real"}
    out = {}
    for r in rows:
        if (r.get("record") == "r2a_turn" and r.get("arm_id") == arm and r.get("seed") in seeds
                and r.get("attempt_id") in done):
            out.setdefault(r["seed"], []).extend((int(c.get("prompt_tokens_est") or 0),
                                                  int(c.get("completion_tokens") or 0)) for c in r["calls"])
    return out


def pilot_ratio(pilot_rows: list[dict]) -> float:
    """Provider input tokens over the harness's own prompt estimate, summed over the pilot's calls."""
    est = sum(c["prompt_tokens_est"] for r in pilot_rows for c in r["calls"] if c.get("outcome") == "ok")
    prov = sum(r["input_tokens"] for r in pilot_rows)
    return prov / est if est else float("nan")


def project(model_id: str, ratio: float, local: dict, margin: float = MARGIN) -> float:
    """Ledger-basis USD of the full job for one model: each local call's prompt estimate x ratio as input, its
    completion tokens as output, priced by client.py's estimate (all input at the input rate), x margin."""
    return margin * sum(cc.estimate_cost_usd(model_id, round(p * ratio), o) for calls in local.values()
                        for p, o in calls)


# ── summary ─────────────────────────────────────────────────────────────────────────────────────────

def _cost_stats(vals: list[float]) -> dict:
    sv = sorted(vals)
    return {"n": len(vals), "total": sum(vals), "mean": statistics.fmean(vals) if vals else None,
            "median": statistics.median(vals) if vals else None, "p90": EV._quantile(sv, 0.9)}


def summarize(rows: list[dict]) -> dict:
    turns = [r for r in rows if r.get("record") == "r2a_turn" and r.get("item_kind") == "task"]
    out = {}
    for m in sorted({r["model_id"] for r in turns}):
        tr = [r for r in turns if r["model_id"] == m]
        graded = [r for r in tr if r.get("end_state_ok") is not None]
        per_seed, per_type = {}, {}
        for r in graded:
            for d, k in ((per_seed, r["seed"]), (per_type, r.get("task_type"))):
                d.setdefault(k, [0, 0])
                d[k][0] += bool(r["end_state_ok"])
                d[k][1] += 1
        out[m] = {"correct": sum(1 for r in graded if r["end_state_ok"]), "graded": len(graded),
                  "per_seed": {str(k): v for k, v in sorted(per_seed.items())},
                  "per_task_type": {str(k): v for k, v in sorted(per_type.items())},
                  "any_http_error_turns": sum(1 for r in tr if r.get("any_http_error")),
                  "usd": _cost_stats([r["usd"] for r in tr]),
                  "usd_ledger_total": sum(r["usd_ledger"] for r in tr),
                  "usd_all_uncached_total": sum(r["usd_all_uncached"] for r in tr),
                  "seconds_mean": statistics.fmean([r["seconds"] for r in tr]) if tr else None,
                  **{f"{k}_total": sum(r[k] for r in tr) for k in ("input_tokens", "uncached_input_tokens",
                                                                    "cache_read_tokens", "cache_write_tokens",
                                                                    "output_tokens")},
                  "n_calls_mean": statistics.fmean([r["n_calls"] for r in tr]) if tr else None}
    return out


# ── main ────────────────────────────────────────────────────────────────────────────────────────────

def redo_finished(path: Path) -> bool:
    """True once the cloud redo file holds an executed run (cloud_redo writes the whole file at the end)."""
    if not Path(path).exists():
        return False
    rows = CR.load_rows(Path(path))
    return (any(r.get("record") == "run_start" and r.get("mode") == CR.MODE_LIVE for r in rows)
            and any(r.get("record") == "run_end" for r in rows))


def _rel(p: Path) -> str:
    p = Path(p)
    return p.relative_to(REPO).as_posix() if p.is_relative_to(REPO) else p.name


def run(client, out: Path = OUT, price_file: Path = PRICE_FILE, local_source: Path = LOCAL_SOURCE, seeds=SEEDS,
        models=None, job_cap: float = JOB_CAP_USD, margin: float = MARGIN, max_turns: int | None = None,
        ledger_total_at_start: float | None = None, log=print, retry_wait_s: float = 10.0) -> dict:
    table = CR.load_price_table(price_file)
    models = list(models or models_from(table))
    out.parent.mkdir(parents=True, exist_ok=True)
    fh = open(out, "w", encoding="utf-8", newline="\n")
    rows: list[dict] = []

    def emit(row):
        rows.append(row)
        fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        fh.flush()

    try:
        local = local_sessions(CR.load_rows(local_source), tuple(seeds))
        emit({"record": "run_start", "harness": HARNESS, "ts_utc": ag.utc_iso(), "mode": "executed",
              "script": "scripts/all_cloud_b3.py", "script_git_blob": CR.git_blob(Path(__file__)),
              "harness_git_blob": CR.git_blob(REPO / "harness" / "x2_r2_evalset.py"),
              "evalset_env_git_blob": CR.git_blob(REPO / "harness" / "evalset_env.py"),
              "price_file": _rel(price_file), "price_file_git_blob": CR.git_blob(price_file),
              "price_file_fetched_utc": table.get("fetched_utc"), "models": models, "seeds": list(seeds),
              "design": DESIGN, "workload": "evalset", "guardrails": False, "max_turns": max_turns,
              "max_tokens": ag.MAX_TOKENS_PER_CALL, "temperature": 0.0, "prompt_caching": True,
              "cache_breakpoints": "system prompt and last user message", "job_cap_usd": job_cap,
              "ledger_total_at_start_usd": ledger_total_at_start, "spend_cap_usd": client.spend_cap_usd,
              "ledger": client.ledger_path.name, "projection_margin": margin,
              "local_reference": {"file": _rel(local_source), "arm": LOCAL_REF_ARM,
                                  "seeds_found": sorted(local)}})
        if sorted(local) != sorted(seeds):
            raise SystemExit(f"refused: local reference sessions found for seeds {sorted(local)}, need {list(seeds)}")
        spend = {"usd": 0.0, "usd_ledger": 0.0}
        runtimes = {m: CloudRuntime(client, m, CR.model_rates(table, m), log=log, retry_wait_s=retry_wait_s)
                    for m in models}
        # warm-up (excluded) and pilot (excluded), per model
        decision, stopped = "stop", None
        try:
            proj = {}
            for m in models:
                w = CR.warmup(client, m)
                spend["usd_ledger"] += w["usd_ledger"]
                spend["usd"] += w["usd_ledger"]   # no cache on the warm-up call
                emit({**w, "model_id": m, "ts_utc": ag.utc_iso()})
                pilot_rows = []
                run_cloud_session(runtimes[m], seeds[0], lambda r: (pilot_rows.append(r), emit(r)), log,
                                  max_turns=1, record="pilot_turn", spend=spend)
                ratio = pilot_ratio([r for r in pilot_rows if r["record"] == "pilot_turn"])
                proj[m] = {"ratio_provider_over_estimate": ratio,
                           "projected_usd_ledger": project(m, ratio, local, margin)}
                log(f"pilot {m}: ratio {ratio:.3f}, projected ledger-basis ${proj[m]['projected_usd_ledger']:.4f}")
            total_proj = spend["usd_ledger"] + sum(p["projected_usd_ledger"] for p in proj.values())
            pilot_errors = sum(1 for r in rows if r.get("record") == "pilot_turn" and r.get("any_http_error"))
            decision = "run" if total_proj <= job_cap and not pilot_errors else "stop"
            emit({"record": "projection", "ts_utc": ag.utc_iso(), "per_model": proj, "spend_so_far_usd_ledger":
                  spend["usd_ledger"], "spend_so_far_usd": spend["usd"], "projected_total_usd_ledger": total_proj,
                  "job_cap_usd": job_cap, "decision": decision, "pilot_error_turns": pilot_errors,
                  "fits_per_model": {m: p["projected_usd_ledger"] <= job_cap - spend["usd_ledger"]
                                     for m, p in proj.items()}})
            log(f"projection: ${total_proj:.4f} ledger-basis for the full job vs cap ${job_cap:.2f} -> {decision}")
            stopped = None
            if decision == "run":
                try:
                    for m in models:
                        for s in seeds:
                            run_cloud_session(runtimes[m], s, emit, log, max_turns=max_turns, spend=spend)
                except cc.SpendCapExceeded as e:
                    stopped = f"SpendCapExceeded: {e}"
                    log(stopped)
            else:
                stopped = ("pilot turn had a provider error; no measured turn run" if pilot_errors
                           else "projection over the job cap; no measured turn run")
        except cc.SpendCapExceeded as e:   # reached during warm-up or pilot
            stopped = f"SpendCapExceeded: {e}"
            log(stopped)
        summary = summarize(rows)
        emit({"record": "run_end", "harness": HARNESS, "ts_utc": ag.utc_iso(), "decision": decision,
              "stopped": stopped, "job_spend_usd": spend["usd"], "job_spend_usd_ledger": spend["usd_ledger"],
              "ledger_total_usd": client.running_total_usd, "summary": summary})
        log(f"end: job ${spend['usd']:.4f} (ledger ${spend['usd_ledger']:.4f}); ledger total "
            f"${client.running_total_usd:.4f}; stopped={stopped}")
        return {"rows": rows, "decision": decision, "stopped": stopped, "summary": summary}
    finally:
        fh.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--live", action="store_true", help="execute on the cloud (needs ANTHROPIC_API_KEY)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--ledger", type=Path, default=cc.DEFAULT_LEDGER_PATH)
    ap.add_argument("--redo-file", type=Path, default=REDO_FILE)
    ap.add_argument("--job-cap", type=float, default=JOB_CAP_USD)
    a = ap.parse_args(argv)
    if not a.live:
        print("refused: this job only runs live (--live); nothing was sent", file=sys.stderr)
        return 2
    if not os.environ.get(KEY_ENV):
        print(f"refused: --live needs {KEY_ENV} set in the environment; nothing was sent", file=sys.stderr)
        return 2
    if not redo_finished(a.redo_file):
        print(f"refused: the cloud redo has not finished ({a.redo_file.name} has no executed run); nothing was sent",
              file=sys.stderr)
        return 3
    total0 = cc.CloudClient(mode="stub", ledger_path=a.ledger)._read_running_total()
    client = cc.CloudClient(mode="real", spend_cap_usd=total0 + a.job_cap, ledger_path=a.ledger)
    print(f"ledger total at start ${total0:.4f}; job cap ${a.job_cap:.2f}; client cap ${client.spend_cap_usd:.4f}")
    res = run(client, a.out, job_cap=a.job_cap, ledger_total_at_start=total0)
    return 0 if res["decision"] == "run" and res["stopped"] is None else 4


if __name__ == "__main__":
    raise SystemExit(main())
