"""Long-task local accuracy on evo-x2 (pre-registered in docs/EXPERIMENTS.md, 2026-10-22, "Long-task local accuracy";
pointer in docs/FINDINGS.md). Tasks: harness/longtask_env.py, "long-horizon synthetic, trace-length matched" work
orders in the B3 v2 tool environment (never SWE-Gym).

Arms (qwen3:14b, think false, as the B3 v2 runs):
  ollama_default : the job's own `ollama serve` with the base env (B5 policy (a)), no num_ctx sent (Ollama loads 40960
                   for qwen3:14b on evo-x2, b5_resolution in results/x2_policy_compare.jsonl)
  kv_q8_0        : B5 policy (g): the job's own server with OLLAMA_KV_CACHE_TYPE=q8_0 and OLLAMA_FLASH_ATTENTION=1,
                   num_ctx = L(q8_0) = 40960 (the same b5_resolution row). The server log's KV type is checked
                   (kv_check); an arm whose check fails is not run (recorded).
Seeds 20260901-03; order per seed: ollama_default then kv_q8_0, the 20 tasks in LONG_TASKS order each. After each
server start: 1 warm-up call (excluded), then the tasks.

One task = one fresh conversation: system prompt, the user message naming the work order file, then the tool loop:
tools on every call, every tool call of a response executed in order (numeric strings coerced as in B3 v2, text-written
calls parsed with x2_r2_evalset.extract_calls_v2); the task ends at the first response without a tool call, or after
step_cap tool-bearing calls (1.5 x reference steps + 5) with one final call without tools. Graded by
longtask_env.check_long_task on the environment as executed.

Per task row (record "longtask_task"): end_state_ok, outcome (correct / wrong_silent / wrong_surfaced; surfaced = an
HTTP or transport error or the step cap; thinking present = wrong), task_wall_s, calls, tool calls, the per-call prompt
profile, loaded window. The run report (<out>.report.json) holds summarize().

    python longtask\\x2_longtask.py --out C:\\apu\\ovn\\results\\x2_longtask_local_v1.jsonl
    python longtask\\x2_longtask.py --check-plan      (no server, no model call, no queue access)

Queue job: calls t2s_queue.advance() exactly once on exit (finally); sets no pause. --stop-after-utc: no new task is
started after that time (pre-registered value in the queue entry); the run_end note says so.
"""
from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
if not (_here / "t2s_queue.py").exists():
    sys.path.insert(1, str(_here.parent))

# Import order matters on evo-x2: x2_r2_agent (taken from C:\apu\ovn) puts its own directory first on sys.path, so
# this side directory's modules are imported before it (x2_r2_evalset imports x2_r2_agent itself, after them).
import evalset_env as E  # noqa: E402
import longtask_env as L  # noqa: E402
import x2_r2_evalset as EV  # noqa: E402
import x2_r2_agent as ag  # noqa: E402

HARNESS = "x2_longtask"
JOB_ID = "x2_longtask_local_v1"
MODEL = "qwen3:14b"
SEEDS = (20260901, 20260902, 20260903)
KV_Q8_NUM_CTX = 40960     # L(q8_0) for qwen3:14b, b5_resolution row of results/x2_policy_compare.jsonl
ARMS = {
    "ollama_default": {"arm_id": "longtask_ollama_default", "env": {}, "num_ctx": None, "kv": None,
                       "b5_policy": "a"},
    "kv_q8_0": {"arm_id": "longtask_kv_q8_0", "env": {"OLLAMA_KV_CACHE_TYPE": "q8_0", "OLLAMA_FLASH_ATTENTION": "1"},
                "num_ctx": KV_Q8_NUM_CTX, "kv": "q8_0", "b5_policy": "g"},
}
ARM_ORDER = ("ollama_default", "kv_q8_0")
SERVER_LOG = r"C:\apu\ovn\ollama_serve_longtask.log"
CALL_CONTENT_KEEP = 600
PROJ_CALIB = EV.PROJ_CALIB      # 1.20, est -> model tokens, as the B3 v2 projection
STOP_AFTER_UTC = "2026-10-27T16:00:00+00:00"


def utc_iso():
    return datetime.now(timezone.utc).isoformat()


# ── one task (runtime-agnostic: Ollama runtime locally, scripts/longtask_cloud.py's cloud runtime) ──────

def run_long_task(runtime, model, num_ctx, think, task: L.LongTask, env_before: dict, log, on_call=None) -> dict:
    """Run one task to its end; returns the scored row (transport fields of each call included)."""
    tools = L.tools_payload()
    messages = [{"role": "system", "content": L.SYSTEM_PROMPT}, {"role": "user", "content": L.user_message(task)}]
    live = copy.deepcopy(env_before)
    recs, n_tool, n_err, n_coerced, n_strict_invalid, n_not_offered, final_text, final_text_calls = \
        [], 0, 0, 0, 0, 0, "", 0
    cap = task.step_cap
    t0 = time.monotonic()
    for i in range(cap + 1):
        offer = i < cap
        resp, rec = ag._do_call(runtime, model, messages, num_ctx, tools if offer else None, think, log)
        if on_call is not None:
            on_call(rec)
        if offer:
            raw, method = EV.extract_calls_v2(resp.get("message"), resp.get("tool_calls"))
        else:
            raw, method = [], "none"
            final_text_calls = len(EV.text_call_objects(resp.get("message") or ""))
        calls = []
        for c in raw:
            c2, n = E.coerce_call(c)
            calls.append(c2)
            n_coerced += n
        n_strict_invalid += sum(1 for c in raw if not E.tool_call_valid(c))
        rec.update({"tool_calls_parsed": calls, "tool_detection_method": method, "tools_offered": offer})
        rec["content"] = (rec.get("content") or "")[:CALL_CONTENT_KEEP]
        recs.append(rec)
        if calls:
            results = []
            for c in calls:
                res = L.execute(live, c)
                results.append(res)
                n_tool += 1
                n_err += res.get("status") != "ok"
                n_not_offered += (c or {}).get("name") not in L.OFFERED_TOOLS
            messages.append(ag._assistant_msg(resp, method))
            messages.extend(L.tool_message(c, r) for c, r in zip(calls, results))
            continue
        messages.append(ag._assistant_msg(resp, "none"))
        final_text = resp.get("message") or ""
        break
    wall = time.monotonic() - t0
    chk = L.check_long_task(task, env_before, live)
    http_err = any((r["http_status"] not in (200, None)) or r["outcome"] != "ok" or r["error"] for r in recs)
    thinking = any(r.get("thinking_present") or r.get("think_tag_in_content") for r in recs)
    cap_hit = len(recs) > cap
    correct = bool(chk["end_state_ok"]) and not thinking
    outcome = "correct" if correct else ("wrong_surfaced" if (http_err or cap_hit) else "wrong_silent")
    prompts = [r.get("prompt_tokens_est") for r in recs]
    pec = [r.get("prompt_eval_count_info_only") for r in recs]
    return {
        "task_id": task.task_id, "stratum": task.stratum, "target_steps": task.target_steps,
        "trajectory_id": task.trajectory_id, "ref_steps": task.ref_steps, "step_cap": cap,
        "n_items": len(task.items), "n_reference_writes": len(task.reference_writes), "task_label": L.LABEL,
        "n_calls": len(recs), "n_tool_calls": n_tool, "n_tool_errors": n_err, "n_not_offered_calls": n_not_offered,
        "n_strict_invalid_calls": n_strict_invalid, "n_args_coerced": n_coerced, "step_cap_hit": cap_hit,
        "final_text": final_text[:2000], "final_text_tool_calls": final_text_calls,
        "final_admits_problem": L.admits_problem(final_text),
        "end_state_ok": chk["end_state_ok"], "end_state_diff": chk["end_state_diff"], "n_diff": chk["n_diff"],
        "any_http_error": http_err, "thinking_present": thinking, "correct": correct, "outcome": outcome,
        "silent_wrong": outcome == "wrong_silent", "task_wall_s": wall,
        "call_seconds": [r.get("duration_s") for r in recs], "prompt_tokens_est": prompts,
        "prompt_eval_count": pec, "completion_tokens": [r.get("completion_tokens") for r in recs],
        "max_prompt_tokens_est": max((p for p in prompts if p), default=None),
        "max_prompt_eval_count": max((p for p in pec if p), default=None),
        "calls": recs,
    }


# ── summaries (pure) ────────────────────────────────────────────────────────────────────────────────

def task_rows(rows, arm_id=None) -> list[dict]:
    return [r for r in rows if r.get("record") == "longtask_task" and (arm_id is None or r.get("arm_id") == arm_id)]


def _wilson(k, n, z=1.959964):
    if not n:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return c - h, c + h


def arm_summary(rows) -> dict:
    """Accuracy, silent-failure share, minutes per task, per stratum and per seed, for one arm's task rows."""
    n = len(rows)
    k = sum(1 for r in rows if r.get("correct"))
    wrong = [r for r in rows if not r.get("correct")]
    silent = [r for r in wrong if r.get("outcome") == "wrong_silent"]
    lo, hi = _wilson(k, n)
    walls = [r["task_wall_s"] for r in rows]
    out = {"n_tasks": n, "correct": k, "accuracy": k / n if n else None, "wilson95": [lo, hi],
           "wrong": len(wrong), "silent_wrong": len(silent), "surfaced_wrong": len(wrong) - len(silent),
           "silent_share_of_wrong": len(silent) / len(wrong) if wrong else None,
           "silent_share_of_all": len(silent) / n if n else None,
           "step_cap_hits": sum(1 for r in rows if r.get("step_cap_hit")),
           "http_error_tasks": sum(1 for r in rows if r.get("any_http_error")),
           "thinking_tasks": sum(1 for r in rows if r.get("thinking_present")),
           "minutes_per_task_mean": statistics.fmean(walls) / 60 if walls else None,
           "minutes_per_task_median": statistics.median(walls) / 60 if walls else None,
           "calls_per_task_mean": statistics.fmean(r["n_calls"] for r in rows) if rows else None,
           "max_prompt_tokens_est": max((r.get("max_prompt_tokens_est") or 0) for r in rows) if rows else None,
           "loaded_context": sorted({r.get("loaded_context") for r in rows if r.get("loaded_context")})}
    for key in ("stratum", "seed"):
        out[f"by_{key}"] = {}
        for v in sorted({r[key] for r in rows}, key=str):
            g = [r for r in rows if r[key] == v]
            out[f"by_{key}"][str(v)] = {"n": len(g), "correct": sum(1 for r in g if r.get("correct")),
                                        "silent_wrong": sum(1 for r in g if r.get("outcome") == "wrong_silent"),
                                        "minutes_mean": statistics.fmean(r["task_wall_s"] for r in g) / 60}
    return out


def summarize(rows) -> dict:
    return {a: arm_summary(task_rows(rows, ARMS[a]["arm_id"])) for a in ARM_ORDER
            if task_rows(rows, ARMS[a]["arm_id"])}


def per_task_wall(rows) -> list[dict]:
    """(arm, seed, task, wall seconds, calls, correct) for every task row: the input of a later latency row."""
    return [{"arm_id": r["arm_id"], "seed": r["seed"], "task_id": r["task_id"], "task_wall_s": r["task_wall_s"],
             "n_calls": r["n_calls"], "correct": r["correct"]} for r in task_rows(rows)]


# ── duration ESTIMATE from measured per-call times (pure) ───────────────────────────────────────────

B5_ARM_OF = {"ollama_default": "b5_a_default_call2_notools", "kv_q8_0": "b5_g_kv_q8_0_call2_notools"}
BAND = 10000


def call_time_bands(pc_rows, model=MODEL) -> dict:
    """{arm: {band: sorted call seconds}} from the B5 qwen3:14b sessions of results/x2_policy_compare.jsonl (calls
    whose turn was not over the loaded window), band = prompt_eval_count // 10000."""
    out = {}
    for r in pc_rows:
        if r.get("record") != "r2a_turn" or r.get("model_id") != model or r.get("over_loaded_window"):
            continue
        arm = next((a for a, b in B5_ARM_OF.items() if r.get("arm_id") == b), None)
        if arm is None:
            continue
        for c in r.get("calls") or []:
            pc, d = c.get("prompt_eval_count_info_only"), c.get("duration_s")
            if pc and d:
                out.setdefault(arm, {}).setdefault(pc // BAND, []).append(d)
    return {a: {b: sorted(v) for b, v in bs.items()} for a, bs in out.items()}


def _q(sv, q):
    k = (len(sv) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(sv) - 1)
    return sv[lo] + (sv[hi] - sv[lo]) * (k - lo)


def oracle_prompt_steps(seed) -> list[list[int]]:
    """Per task: the model-token prompt length at each reference step (oracle transcript, est x PROJ_CALIB)."""
    tools = L.tools_payload()
    out = []
    for task, env in L.build_tasks(seed):
        msgs, _ = L.oracle_transcript(task, env)
        # call k sees the system + user message and every assistant/tool pair before it
        steps = [round(ag.prompt_tokens_est(msgs[:2 + 2 * k], tools) * PROJ_CALIB) for k in range(task.ref_steps + 1)]
        out.append(steps)
    return out


def hours_estimate(pc_rows, seeds=SEEDS, high_calls_factor=L.STEP_CAP_FACTOR, overhead_s_per_arm=120.0) -> dict:
    """ESTIMATE per arm and total: every reference step plus the final answer is one call, timed at the B5 median call
    time of its prompt band for that arm (point); high = 1.5 x the calls at the band's 75th percentile."""
    bands = call_time_bands(pc_rows)
    out = {}
    for arm in ARM_ORDER:
        tb = bands.get(arm) or {}
        if not tb:
            raise ValueError(f"no B5 call times for {arm}")
        top = max(tb)

        def t(p, q):
            return _q(tb[min(p // BAND, top)], q)
        pt = hi = 0.0
        for s in seeds:
            for steps in oracle_prompt_steps(s):
                pt += sum(t(p, 0.5) for p in steps)
                hi += high_calls_factor * sum(t(p, 0.75) for p in steps)
        pt += overhead_s_per_arm * len(seeds)
        hi += overhead_s_per_arm * len(seeds)
        out[arm] = {"point_h": pt / 3600, "high_h": hi / 3600,
                    "bands": {b: {"n": len(v), "median_s": _q(v, 0.5), "p75_s": _q(v, 0.75)} for b, v in tb.items()}}
    out["total"] = {"point_h": sum(out[a]["point_h"] for a in ARM_ORDER),
                    "high_h": sum(out[a]["high_h"] for a in ARM_ORDER)}
    return out


# ── KV-type check on the server log (pure; same rule as policy_compare.kv_check) ───────────────────

def kv_check(log_text: str, requested: str | None) -> dict:
    import re
    if requested is None:
        return {"requested": None, "valid": True}
    cfg = [ln for ln in (log_text or "").splitlines() if "server config" in ln]
    has_env = any(f"OLLAMA_KV_CACHE_TYPE:{requested}" in ln for ln in cfg)
    fa = any(re.search(r"OLLAMA_FLASH_ATTENTION:(true|1)", ln) for ln in cfg)
    rx = (re.compile(r"type_k\s*=\s*'?([a-z0-9_]+)", re.I), re.compile(r"\bK \(([a-z0-9_]+)\)"),
          re.compile(r"cache_type_k[\"=: ]+([a-z0-9_]+)", re.I))
    observed = sorted({m.group(1).lower() for x in rx for m in x.finditer(log_text or "")})
    bad = [t for t in observed if t != requested]
    return {"requested": requested, "config_has_env": has_env, "config_flash_attention": fa, "observed": observed,
            "valid": has_env and fa and not bad, "verified": bool(observed) and not bad}


# ── server (evo-x2: host_config's `ollama serve`, as policy_compare.EvoServer) ──────────────────────

class Server:
    def __init__(self, hc, log_path=SERVER_LOG):
        self.hc, self.log_path, self.env = hc, log_path, None

    def start(self, env: dict):
        self.hc.stop_ollama_server()
        try:
            Path(self.log_path).unlink()
        except OSError:
            pass
        ag._start_server(self.hc, dict(env) or None, self.log_path)
        if not self.hc.wait_for_ollama_ready(timeout_s=60):
            self.hc.stop_ollama_server()
            ag._start_server(self.hc, dict(env) or None, self.log_path)
            if not self.hc.wait_for_ollama_ready(timeout_s=90):
                raise RuntimeError("ollama did not become ready")
        self.env = dict(env)

    def stop(self):
        self.hc.stop_ollama_server()

    def log_text(self) -> str:
        try:
            return Path(self.log_path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def runtime(self):
        rt = ag.OllamaRuntime()
        rt.server_env, rt.server_log = (dict(self.env) or None), self.log_path
        return rt


def warmup(runtime, model, num_ctx, think) -> dict:
    """The excluded warm-up call (loads the model), then the window Ollama reports."""
    t0 = time.monotonic()
    resp = runtime.chat(model, [{"role": "user", "content": "Reply with OK."}], num_ctx, None, think)
    return {"outcome": resp.get("outcome"), "http_status": resp.get("status"), "error": resp.get("error"),
            "seconds": time.monotonic() - t0, "loaded_context": runtime.loaded_context(model)}


# ── plan / provenance ───────────────────────────────────────────────────────────────────────────────

def git_blob_sha1(path) -> str | None:
    import hashlib
    try:
        data = Path(path).read_bytes()
    except OSError:
        return None
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def provenance() -> dict:
    return {"harness_file": str(Path(__file__).resolve()), "harness_blob": git_blob_sha1(__file__),
            "longtask_env_blob": git_blob_sha1(L.__file__), "evalset_env_blob": git_blob_sha1(E.__file__),
            "x2_r2_evalset_blob": git_blob_sha1(EV.__file__), "agent_file": str(Path(ag.__file__).resolve()),
            "agent_blob": git_blob_sha1(ag.__file__)}


def check_plan(seeds=SEEDS, arms=ARM_ORDER) -> dict:
    """The plan and the oracle self-check (every task's reference solution passes the grader); no server, no call."""
    per = {}
    for s in seeds:
        for task, env in L.build_tasks(s):
            msgs, after = L.oracle_transcript(task, env)
            ok = L.check_long_task(task, env, after)["end_state_ok"]
            per.setdefault(task.task_id, []).append({"seed": s, "ref_steps": task.ref_steps, "cap": task.step_cap,
                                                     "items": len(task.items), "oracle_ok": ok,
                                                     "final_prompt_est": ag.prompt_tokens_est(msgs[:-1],
                                                                                              L.tools_payload())})
    return {"model": MODEL, "think": ag.MODELS.get(MODEL), "seeds": list(seeds),
            "arms": {a: ARMS[a] for a in arms}, "n_tasks": len(L.LONG_TASKS), "label": L.LABEL,
            "oracle_all_ok": all(x["oracle_ok"] for v in per.values() for x in v),
            "ref_steps_total_per_seed": {s: sum(x["ref_steps"] for v in per.values() for x in v if x["seed"] == s)
                                         for s in seeds},
            "tasks": per, "provenance": provenance()}


# ── main ────────────────────────────────────────────────────────────────────────────────────────────

def build_arg_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=r"C:\apu\ovn\results\x2_longtask_local_v1.jsonl")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--arms", default=",".join(ARM_ORDER))
    ap.add_argument("--tasks", default=None, help="comma-separated task ids (default: all 20)")
    ap.add_argument("--stop-after-utc", default=STOP_AFTER_UTC)
    ap.add_argument("--check-plan", action="store_true")
    return ap


def main(argv=None, advance=True, deps=None) -> str:
    args = build_arg_parser().parse_args(argv)
    deps = deps or {}
    seeds = tuple(int(s) for s in args.seeds.split(",") if s)
    arms = tuple(a for a in args.arms.split(",") if a)
    if any(a not in ARMS for a in arms):
        raise SystemExit(f"unknown arm in {arms}")
    task_ids = tuple(t for t in (args.tasks or "").split(",") if t) or None
    if args.check_plan:
        print(json.dumps(check_plan(seeds, arms), indent=1, default=str))
        return "check-plan"
    log = deps.get("log", print)
    note, server, fh = "completed", None, None
    stop_at = datetime.fromisoformat(args.stop_after_utc) if args.stop_after_utc else None
    now = deps.get("now", lambda: datetime.now(timezone.utc))
    try:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and any(r.get("record") == "run_end" for r in ag.read_rows(out_path)):
            raise RuntimeError(f"{out_path.name} already holds a finished run; refusing to append another")
        fh = open(out_path, "a", encoding="utf-8", newline="\n")
        rows = []
        attempt = uuid.uuid4().hex[:12]

        def emit(r):
            r = {**r, "attempt_id": attempt}
            rows.append(r)
            fh.write(json.dumps(r, default=str) + "\n")
            fh.flush()
        hc = deps.get("hc")
        if hc is None:
            import host_config as hc
        hostname = deps.get("hostname") or __import__("socket").gethostname()
        host_cfg = hc.require_host(hostname)
        if hostname.upper() != "EVO-X2":
            raise RuntimeError(f"{HARNESS}: host {hostname!r} is not EVO-X2")
        interactive = None
        try:
            interactive = hc.enforce_or_record_interactive_session(host_cfg)
        except SystemExit as e:
            note = f"stopped: {e}"[:400]
            return note
        if deps.get("server") is None:
            import x2_num_keep as nk
            if nk.tray_app_running():
                raise RuntimeError("the Ollama tray app is running; this job starts its own servers and does not stop "
                                   "the tray app")
        server = deps.get("server") or Server(hc)
        plan = check_plan(seeds, arms)
        if not plan["oracle_all_ok"]:
            raise RuntimeError("oracle self-check failed: a reference solution does not pass the grader")
        emit({"record": "run_start", "harness": HARNESS, "job_id": JOB_ID, "ts_utc": utc_iso(), "model_id": MODEL,
              "think": ag.MODELS.get(MODEL), "seeds": list(seeds), "arms": {a: ARMS[a] for a in arms},
              "arm_order": list(arms), "tasks": [list(t) for t in L.LONG_TASKS if not task_ids or t[0] in task_ids],
              "task_label": L.LABEL, "offered_tools": list(L.OFFERED_TOOLS), "step_cap_rule":
              f"{L.STEP_CAP_FACTOR} x ref_steps + {L.STEP_CAP_EXTRA}", "max_tokens_per_call": ag.MAX_TOKENS_PER_CALL,
              "keep_alive": ag.KEEP_ALIVE, "stop_after_utc": args.stop_after_utc,
              "ref_steps_total_per_seed": plan["ref_steps_total_per_seed"], "interactive_session": interactive,
              **provenance()})
        stopped = False
        for s in seeds:
            if stopped:
                break
            built = L.build_tasks(s, task_ids)
            for a in arms:
                cfg = ARMS[a]
                if stop_at is not None and now() >= stop_at:
                    stopped = True
                    break
                server.start(cfg["env"])
                rt = deps.get("runtime") or server.runtime()
                w = warmup(rt, MODEL, cfg["num_ctx"], ag.MODELS.get(MODEL))
                chk = kv_check(server.log_text(), cfg["kv"])
                emit({"record": "warmup", "excluded_from_stats": True, "seed": s, "arm_id": cfg["arm_id"],
                      "kv_check": chk, "server_env": cfg["env"], "ts_utc": utc_iso(), **w})
                if not chk["valid"] or w["outcome"] != "ok":
                    emit({"record": "arm_skipped", "seed": s, "arm_id": cfg["arm_id"], "kv_check": chk,
                          "warmup": w, "ts_utc": utc_iso()})
                    log(f"arm {a} seed {s} skipped: kv_check {chk} warmup {w['outcome']}")
                    continue
                for task, env in built:
                    if stop_at is not None and now() >= stop_at:
                        stopped = True
                        break
                    row = run_long_task(rt, MODEL, cfg["num_ctx"], ag.MODELS.get(MODEL), task, env, log)
                    loaded = rt.loaded_context(MODEL)
                    row.update({"record": "longtask_task", "harness": HARNESS, "model_id": MODEL,
                                "arm_id": cfg["arm_id"], "arm": a, "seed": s, "num_ctx_requested": cfg["num_ctx"],
                                "loaded_context": loaded, "kv": cfg["kv"], "ts_utc": utc_iso(),
                                "over_loaded_window": bool(loaded and row["max_prompt_tokens_est"]
                                                           and row["max_prompt_tokens_est"] * PROJ_CALIB > loaded)})
                    emit(row)
                    log(f"{a} seed {s} {task.task_id}: {row['outcome']} calls={row['n_calls']} "
                        f"ref={task.ref_steps} {row['task_wall_s'] / 60:.1f} min")
        summary = summarize(rows)
        Path(str(out_path) + ".report.json").write_text(json.dumps({"summary": summary,
                                                                     "per_task_wall": per_task_wall(rows)},
                                                                    indent=1, default=str), encoding="utf-8")
        if stopped:
            note = f"completed; stopped at --stop-after-utc {args.stop_after_utc} before every task ran"
        emit({"record": "run_end", "note": note, "summary": summary, "ts_utc": utc_iso()})
    except Exception as e:
        import traceback
        note = f"stopped: {e!r}"[:400]
        log(note)
        traceback.print_exc()
    finally:
        if fh is not None:
            fh.close()
        if server is not None:
            try:
                server.stop()
            except Exception as e:
                log(f"ollama stop failed: {e!r}")
        if advance:
            try:
                tq = deps.get("tq")
                if tq is None:
                    import t2s_queue as tq
                tq.advance(note)
            except Exception as e:
                log(f"queue advance failed: {e!r}")
    return note


if __name__ == "__main__":
    main()
