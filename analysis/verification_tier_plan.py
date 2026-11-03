"""Verification tier, planning figures (docs/VERIFICATION_TIER_PROPOSAL.md; operator decisions 2026-11-03). Offline, no
network, no API key, USD 0.

Builds the verifier payloads exactly as the live Run V1 would (src/dse/verification_tier.py, arm (a): claude-haiku-5-5
accept/reject verdict), only for LOCAL results that L1 to L3 accept, and never for cloud-escalated results:

  - B3 v2 (results/x2_evalset_tasks_real_v2.jsonl, qwen3:14b, evo-x2): the task turns the post-run check replay
    accepts (results/post_run_check_replay_v1.jsonl, decision "accept"; tiers ollama_ctx_4096, ollama_ctx_8192,
    ollama_default, seeds 20260901 to 20260905; L1 and L2 as recorded PROXIES, L3 as replayed there);
  - the q4_0 28928 operating point (results/x2_own_config_probe_q4_28928.jsonl, seeds 20260901 to 20260903): the task
    turns hybrid-full-pipeline-v1's replay keeps local (route at start did not route them and the post-run check did
    not escalate them; analysis/hybrid_full_pipeline.py::run_pipeline per_task "local").

Payload inputs (client_view): the task text with the recall-probe suffix STRIPPED, the standing instructions and rules
6 to 8, the pre-turn state (directory format, api_token values redacted), the turn's tool calls with their results (each
call applied in order to a copy of the pre-turn state, as the local harness did: evalset_env.apply_call, no fact
store), the final answer, and the records the turn changed (before vs after). The turn rows are passed through
hybrid_full_pipeline.strip_grader first, and the session's Task objects are never read, so the grader, the reference
calls and end_state_diff cannot reach the payload. The grader's fields are read AFTER the payload is built, only to say
which accepted turns are wrong and why the error is visible in the payload (accepted_wrong).

Token estimator (stated with every figure): tiktoken o200k_base count of the payload text x a provider/o200k ratio
measured on the full-history Sonnet redo's FIRST calls (results/cloud_redo_full_context_v1*.jsonl: provider usage
input_tokens of call 1 vs the o200k_base count of that call's rebuilt prompt, scripts/cloud_redo.py::rebuild, system
prompt + every message's text + tool calls as JSON + the tools schema as JSON). Point figures use the median ratio,
the high and upper-bound figures the largest ratio observed. ASSUMPTION: the ratio measured on the redo prompts
(filler-heavy chat transcripts, claude-sonnet-5-5) transfers to the verifier payload (tabular state and JSON) and to
claude-haiku-5-5 (same model generation, same price-table tokenizer note).

Pricing: results/cloud_prices_anthropic_20261014.json, rolling cache breakpoint (src/cloud/prompt_cache.py): the system
prompt (instruction + standing instructions + rules, identical within a session) is written on a session's first
verifier call and read on every later one; the user message (the turn's own part) is written on every call. ASSUMED:
consecutive verifier calls of one session are under 5 minutes apart; output 100 tokens per verdict at the point, the
cap (256) at high and upper bound.

Labels: every figure here is PLANNING (sizes and prices), ESTIMATE (the Run V1 cost) or CEILING (perfect verifier);
none of them is a verifier measurement. Nothing in recorded data stands in for a verdict.

Usage: py -3.12 analysis/verification_tier_plan.py   (prints the four register texts)
"""
from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

B3_FILE = "results/x2_evalset_tasks_real_v2.jsonl"
Q4_FILE = "results/x2_own_config_probe_q4_28928.jsonl"
PRC_FILE = "results/post_run_check_replay_v1.jsonl"
PRICE_FILE = "results/cloud_prices_anthropic_20261014.json"
ALL_CLOUD_FILE = "results/all_cloud_b3_v1.jsonl"
REDO_FILES = ["results/cloud_redo_full_context_v1.jsonl", "results/cloud_redo_full_context_v1_part2.jsonl",
              "results/cloud_redo_full_context_v1_part3.jsonl"]
ROLLBACK_FILE = "results/own_rollback_redo_v1.jsonl"
DATA_FILES = [B3_FILE, Q4_FILE, PRC_FILE, PRICE_FILE, ALL_CLOUD_FILE, *REDO_FILES, ROLLBACK_FILE]
Q4_CONFIG = "q4_0 28928"
MODEL = "qwen3:14b"
OUT_TOKENS_POINT = 100      # ASSUMED: one JSON verdict with a one-sentence reason
WARMUP_CALLS = 1            # statistics convention: 1 warm-up + N measured calls
SNAPSHOT_HEADER_FROM = "Workspace directory (state at the start of this session):"
SNAPSHOT_HEADER_TO = "Workspace directory (state before this turn):"

_CACHE: dict = {}


def _paths(repo: Path):
    for sub in ("", "analysis", "harness", "scripts"):
        p = str(Path(repo) / sub) if sub else str(Path(repo))
        if p not in sys.path:
            sys.path.insert(0, p)


def _rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def _load_module(repo: Path, rel: str, name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(repo) / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def _q(xs, p):
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    return statistics.median(xs) if p == 50 else statistics.quantiles(xs, n=10, method="inclusive")[8]


# ── sessions and the client's view of one turn ────────────────────────────────────────────────────────

def session_for(row: dict, cache: dict):
    """The rebuilt evalset session of a turn row (seed, design, guardrails, workload, model)."""
    _paths(REPO)
    import x2_r2_evalset as EV
    key = (row["seed"], row.get("design", "v2"), bool(row.get("guardrails")), row.get("workload", "evalset"),
           row["model_id"])
    if key not in cache:
        with contextlib.redirect_stdout(io.StringIO()):
            cache[key] = EV.build_session(row["seed"], None, key[3], key[2], row["model_id"], None, key[1])
    return cache[key]


def _record_of(table: str, path_rest: str, ids) -> str | None:
    """The record id a diff path names (file ids contain slashes: the longest id that is the path or its prefix)."""
    hits = [k for k in ids if path_rest == k or path_rest.startswith(k + "/")]
    return max(hits, key=len) if hits else None


def changed_records(E, before: dict, after: dict) -> list[dict]:
    out = []
    for p in E.env_diff(before, after, limit=10_000):
        if p == "outbox":
            out.append({"outbox_added": after["outbox"][len(before["outbox"]):]})
            continue
        table, rest = p.split("/", 1)
        rid = _record_of(table, rest, set(before[table]) | set(after[table]))
        rec = {"record": f"{table}/{rid}", "before": before[table].get(rid), "after": after[table].get(rid)}
        if rec not in out:
            out.append(rec)
    return out


def client_view(es, prior: list[dict], row: dict) -> dict:
    """What the client holds at the end of the turn. prior: the session's earlier turn rows; row: this turn. Every row
    must already be grader-stripped (hybrid_full_pipeline.strip_grader); only "calls", "final_text" and "turn_idx"
    are read. Returns {env_before, env_after, trace, changes, turn}."""
    _paths(REPO)
    import evalset_env as E
    env = copy.deepcopy(es.env0)
    for r in prior:
        for c in r.get("calls") or []:
            for tc in c.get("tool_calls_parsed") or []:
                E.apply_call(env, tc)
    before = copy.deepcopy(env)
    trace = []
    for c in row.get("calls") or []:
        for tc in c.get("tool_calls_parsed") or []:
            trace.append({"call": tc, "result": E.apply_call(env, tc)})
    turn = {t.idx: t for t in es.turns}[row["turn_idx"]]
    return {"env_before": before, "env_after": env, "trace": trace, "changes": changed_records(E, before, env),
            "turn": turn}


def state_text(E, env: dict) -> str:
    return E.directory_text(env).replace(SNAPSHOT_HEADER_FROM, SNAPSHOT_HEADER_TO, 1)


def state_json(E, env: dict) -> str:
    """Arm (c)'s state: every table with every field (phone numbers, file contents) plus the outbox."""
    return json.dumps({k: env[k] for k in (*E.TABLES, "outbox")}, sort_keys=True)


def payloads_for(es, prior: list[dict], row: dict) -> dict:
    """Arm (a) and arm (c) payloads of one grader-stripped turn row, and the client view they came from."""
    _paths(REPO)
    import evalset_env as E
    from src.dse import verification_tier as VT
    v = client_view(es, prior, row)
    t = v["turn"]
    secrets = E.secrets_of(v["env_before"])
    a = VT.build_verdict_payload(t.user_text, t.recall_key if t.is_recall else None, es.standing, E.RULES_678_TEXT,
                                 state_text(E, v["env_before"]), v["trace"], row.get("final_text"), v["changes"],
                                 secrets)
    c = VT.build_rederive_payload(t.user_text, t.recall_key if t.is_recall else None, es.standing, E.RULES_678_TEXT,
                                  state_json(E, v["env_before"]), secrets)
    return {"a": a, "c": c, "view": v}


# ── which local results L1 to L3 accept ─────────────────────────────────────────────────────────────

def _completed_turns(rows: list[dict], arm_ok) -> dict:
    """{attempt_id: [turn rows sorted]} of completed real qwen3:14b sessions whose arm passes arm_ok."""
    _paths(REPO)
    import x2_r2_agent as ag
    att = {s["attempt_id"] for s in ag.completed_sessions(rows)
           if s.get("mode") == "real" and s["model_id"] == MODEL and arm_ok(s["arm_id"])}
    out = {}
    for r in rows:
        if r.get("record") == "r2a_turn" and r.get("attempt_id") in att:
            out.setdefault(r["attempt_id"], []).append(r)
    for v in out.values():
        v.sort(key=lambda r: r["turn_idx"])
    return out


def accepted_b3(repo: Path) -> dict:
    """{(tier, seed, turn_idx): task_id} the post-run check replay accepts."""
    return {(r["tier"], r["seed"], r["turn_idx"]): r["task_id"] for r in _rows(Path(repo) / PRC_FILE)
            if r.get("record") == "prc_task" and r.get("decision") == "accept"}


def q4_pipeline(repo: Path) -> dict:
    """hybrid-full-pipeline-v1's replay of the q4_0 28928 config: its run_pipeline result, local rows and escalation
    outcomes (for the CEILING)."""
    _paths(repo)
    import hybrid_full_pipeline as hfp
    check, budget = hfp.load_post_run_check(repo), hfp.load_task_budget(repo)
    refs = hfp.replay_refs(repo)
    local, first, ceiling = hfp.config_sessions(repo, Q4_CONFIG)
    tasks = sorted(k for k in local if k in refs["cloud"])
    esc = hfp.escalation_outcomes(repo, hfp.CONFIGS[Q4_CONFIG][2], refs["cloud"], refs["redo"], refs["rb"],
                                  refs["imputed_usd"])
    res = hfp.run_pipeline(tasks, local, first, ceiling, refs["cloud"], esc, check, budget)
    if res["budget_refused_escalation"]:
        raise ValueError("q4_0 28928: an escalation was refused by the budget; kept-local is not L1-L3 accepted")
    return {"res": res, "local": local, "esc": esc, "n": len(tasks)}


def accepted_turns(repo: Path) -> list[dict]:
    """Every accepted local task turn as {config, tier, seed, turn_idx, task_id, es, prior, row (grader-stripped),
    full (the recorded row, for scoring only)}, in run order (config, tier, seed, turn)."""
    _paths(repo)
    import hybrid_full_pipeline as hfp
    cache, out = {}, []
    acc = accepted_b3(repo)
    b3 = _completed_turns(_rows(Path(repo) / B3_FILE), lambda a: a.endswith("_call2_notools"))
    for att, turns in b3.items():
        tier = turns[0]["arm_id"].replace("_call2_notools", "")
        for i, r in enumerate(turns):
            if r.get("item_kind") == "task" and (tier, r["seed"], r["turn_idx"]) in acc:
                if acc[(tier, r["seed"], r["turn_idx"])] != r["task_id"]:
                    raise ValueError(f"task id mismatch at {(tier, r['seed'], r['turn_idx'])}")
                out.append({"config": "B3 v2", "tier": tier, "seed": r["seed"], "turn_idx": r["turn_idx"],
                            "task_id": r["task_id"], "es": session_for(r, cache),
                            "prior": [hfp.strip_grader(p) for p in turns[:i]], "row": hfp.strip_grader(r), "full": r})
    q4 = q4_pipeline(repo)
    kept = {k for k, v in q4["res"]["per_task"].items() if v == "local"}
    arm = hfp.CONFIGS[Q4_CONFIG][1]
    q4t = _completed_turns(_rows(Path(repo) / Q4_FILE), lambda a: a == arm)
    for att, turns in q4t.items():
        for i, r in enumerate(turns):
            if r.get("item_kind") == "task" and (r["seed"], r["task_id"]) in kept:
                out.append({"config": Q4_CONFIG, "tier": "ollama_ctx_28928 (q4_0)", "seed": r["seed"],
                            "turn_idx": r["turn_idx"], "task_id": r["task_id"], "es": session_for(r, cache),
                            "prior": [hfp.strip_grader(p) for p in turns[:i]], "row": hfp.strip_grader(r), "full": r})
    n_b3 = sum(1 for t in out if t["config"] == "B3 v2")
    if n_b3 != len(acc):
        raise ValueError(f"accepted B3 v2 turns found {n_b3}, replay file has {len(acc)}")
    if sum(1 for t in out if t["config"] == Q4_CONFIG) != len(kept):
        raise ValueError("q4_0 28928 kept-local turns not all found")
    out.sort(key=lambda t: (t["config"] != "B3 v2", t["tier"], t["seed"], t["turn_idx"]))
    return out


# ── token estimator ────────────────────────────────────────────────────────────────────────────────

def _enc():
    import tiktoken
    return tiktoken.get_encoding("o200k_base")


def calibration(repo: Path) -> dict:
    """provider input tokens / o200k_base tokens of the full-history redo's first calls (module docstring)."""
    _paths(repo)
    import impact_model as im
    import x2_r2_evalset as EV
    cr = _load_module(repo, "scripts/cloud_redo.py", "cloud_redo")
    enc = _enc()
    merged, model, _n, _p = im.redo_rows_and_model(repo)
    local = _rows(Path(repo) / B3_FILE)
    by_att = {}
    for r in local:
        if r.get("record") == "r2a_turn":
            by_att.setdefault(r["attempt_id"], []).append(r)
    tools_tok = len(enc.encode(json.dumps(EV.tools_payload("v2"))))
    cache, ratios = {}, []
    for d in merged:
        if d.get("record") != "cloud_redo_task" or not d.get("cloud_calls"):
            continue
        rs = sorted(by_att[d["attempt_id"]], key=lambda r: r["turn_idx"])
        prior = [r for r in rs if r["turn_idx"] < d["turn_idx"]]
        es = session_for(rs[0], cache)
        with contextlib.redirect_stdout(io.StringIO()):
            built = cr.rebuild(es, prior, d["turn_idx"])
        n = tools_tok
        for m in built["messages"]:
            n += len(enc.encode(m.get("content") or ""))
            if m.get("tool_calls"):
                n += len(enc.encode(json.dumps(m["tool_calls"])))
        ratios.append(d["cloud_calls"][0]["input_tokens"] / n)
    return {"model": model, "n": len(ratios), "median": statistics.median(ratios), "min": min(ratios),
            "max": max(ratios)}


# ── pricing ──────────────────────────────────────────────────────────────────────────────────────

def rates(repo: Path, model: str) -> dict:
    p = json.loads((Path(repo) / PRICE_FILE).read_text(encoding="utf-8"))
    m = p["models"][model]
    return {**m["standard"]["short"], "min_cacheable": m["min_cacheable_tokens"]}


def call_usd(rt: dict, read: float, write: float, uncached: float, out: float) -> float:
    from src.cloud.prompt_cache import split_cost
    return split_cost(rt, [(read, write, uncached)], [read + write + uncached], [out])


def split_for(rt: dict, static: float, turn: float, warm: bool) -> tuple:
    """(read, write, uncached) of one verifier call under the rolling breakpoint: the system prompt is read when the
    session's cache is warm, else written (or sent uncached below the minimum cacheable prefix); the user message is
    written (the whole prompt is the second breakpoint's prefix)."""
    mc = rt["min_cacheable"]
    if static < mc:
        return (0, 0, static + turn) if static + turn < mc else (0, static + turn, 0)
    return (static, turn, 0) if warm else (0, static + turn, 0)


def per_task_prices(rt: dict, sizes: dict, out_point: float, out_cap: float) -> dict:
    """billed point (static read, median turn part written, out_point output, median ratio); high (cold cache, p90 turn
    part, cap output, max ratio); upper bound (largest prompt all written, cap output, max ratio)."""
    pt = call_usd(rt, *split_for(rt, sizes["static_med"], sizes["turn_med"], True), out_point)
    hi = call_usd(rt, *split_for(rt, sizes["static_hi"], sizes["turn_p90_hi"], False), out_cap)
    ub = call_usd(rt, 0, sizes["max_hi"], 0, out_cap)
    return {"point": pt, "high": hi, "upper": ub}


def run_cost(rt: dict, calls: list[dict], key: str, out_point: float, out_cap: float) -> dict:
    """Run V1 cost of one arm over the ordered calls (cache group = (config, tier, seed) session; the first call of a
    group is cold) plus WARMUP_CALLS cold warm-up calls on the first payload."""
    pt, hi, prev = 0.0, 0.0, None
    for c in calls:
        grp = (c["config"], c["tier"], c["seed"])
        pt += call_usd(rt, *split_for(rt, c[key]["static"], c[key]["turn"], grp == prev), out_point)
        hi += call_usd(rt, *split_for(rt, c[key]["static_hi"], c[key]["turn_hi"], False), out_cap)
        prev = grp
    f = calls[0][key]
    pt += WARMUP_CALLS * call_usd(rt, *split_for(rt, f["static"], f["turn"], False), out_point)
    hi += WARMUP_CALLS * call_usd(rt, *split_for(rt, f["static_hi"], f["turn_hi"], False), out_cap)
    mx = max(c[key]["static_hi"] + c[key]["turn_hi"] for c in calls)
    ub = (len(calls) + WARMUP_CALLS) * call_usd(rt, 0, mx, 0, out_cap)
    return {"point": pt, "high": hi, "upper": ub, "n_calls": len(calls) + WARMUP_CALLS, "max_prompt_hi": mx}


def haiku_allcloud_output_per_task(repo: Path) -> dict:
    rows = [r for r in _rows(Path(repo) / ALL_CLOUD_FILE)
            if r.get("record") == "r2a_turn" and r.get("item_kind") == "task" and r.get("model_id") == "claude-haiku-5-5"]
    tot = sum(int(r["output_tokens"]) for r in rows)
    return {"n": len(rows), "output_tokens": tot, "per_task": tot / len(rows)}


# ── accepted-wrong turns and why each is visible ──────────────────────────────────────────────────────

def _snapshot_line(state: str, rid: str) -> str | None:
    for ln in state.splitlines():
        if ln.strip().startswith(f"{rid} |"):
            return ln.strip()
    return None


def why_visible(t: dict, p: dict) -> dict:
    """Scoring-side analysis (reads the grader's end_state_diff and the task's expected answer values AFTER the payload
    is built; neither is in the payload): where in the verifier's inputs the error can be read."""
    _paths(REPO)
    import evalset_env as E
    full, v, user = t["full"], p["view"], p["a"]["user"]
    reasons, rids = [], []
    for path in full.get("end_state_diff") or []:
        if path == "outbox":
            reasons.append("outbox differs: the sent email is in the change set")
            continue
        table, rest = path.split("/", 1)
        rid = _record_of(table, rest, set(v["env_before"][table]) | set(v["env_after"][table]) | {rest.split("/")[0]})
        fld = rest[len(rid) + 1:] if rid and len(rest) > len(rid) else None
        rids.append(rid)
        in_changes = any(c.get("record") == f"{table}/{rid}" for c in v["changes"])
        if in_changes:
            b = (v["env_before"][table].get(rid) or {}).get(fld) if fld else None
            a = (v["env_after"][table].get(rid) or {}).get(fld) if fld else None
            line = _snapshot_line(state_text(E, v["env_before"]), rid)
            reasons.append(f"{table}/{rid} is in the change set ({fld}: {b!r} -> {a!r})"
                           + (f", pre-turn state line: {line}" if line else ", a new record"))
        else:
            line = _snapshot_line(state_text(E, v["env_before"]), rid)
            reasons.append(f"{table}/{rid} is in the pre-turn state ({line}) and absent from the change set" if line
                           else f"{table}/{rid}/{fld} should have changed and is absent from the change set")
    if full.get("answer_ok") is False:
        task = t["es"].tasks[t["task_id"]]
        vals = [str(x) for x in (task.answer_values or ())]
        shown = all(E.answer_has(state_text(E, v["env_before"]), x) for x in task.answer_values or ())
        reasons.append(f"the final answer lacks {', '.join(vals)}; the pre-turn state shows it: {'yes' if shown else 'no'}")
    errs = [x["result"].get("error") or x["result"].get("message") for x in v["trace"] if x["result"].get("status") != "ok"]
    if errs:
        reasons.append(f"{len(errs)} tool result(s) with an error in the trace ({'; '.join(map(str, errs))[:120]})")
    return {"reasons": reasons, "record_ids_in_payload": all(r in user for r in rids)}


# ── the plan ───────────────────────────────────────────────────────────────────────────────────────

def plan(repo: Path = REPO) -> dict:
    repo = Path(repo)
    key = str(repo)
    if key in _CACHE:
        return _CACHE[key]
    _paths(repo)
    import post_run_check_replay as R
    enc = _enc()
    cal = calibration(repo)
    turns = accepted_turns(repo)
    calls = []
    for t in turns:
        p = payloads_for(t["es"], t["prior"], t["row"])
        rec = {k: t[k] for k in ("config", "tier", "seed", "turn_idx", "task_id")}
        for arm in ("a", "c"):
            s = len(enc.encode(p[arm]["system"]))
            u = len(enc.encode(p[arm]["user"]))
            rec[arm] = {"static_o200k": s, "turn_o200k": u, "static": s * cal["median"], "turn": u * cal["median"],
                        "static_hi": s * cal["max"], "turn_hi": u * cal["max"],
                        "chars": len(p[arm]["system"]) + len(p[arm]["user"])}
        rec["local_ok"] = t["full"]["end_state_ok"] is True     # scoring only, read after the payload is built
        if not rec["local_ok"]:
            rec["why"] = why_visible(t, p)
        rec["recall_probe_stripped"] = bool(t["full"].get("is_recall"))
        calls.append(rec)

    def sizes(arm):
        st = [c[arm]["static"] for c in calls]
        tu = [c[arm]["turn"] for c in calls]
        tot = [c[arm]["static"] + c[arm]["turn"] for c in calls]
        return {"static_med": _q(st, 50), "static_hi": max(c[arm]["static_hi"] for c in calls),
                "turn_med": _q(tu, 50), "turn_p90": _q(tu, 90), "turn_max": max(tu),
                "turn_p90_hi": _q([c[arm]["turn_hi"] for c in calls], 90),
                "input_med": _q(tot, 50), "input_max": max(tot),
                "max_hi": max(c[arm]["static_hi"] + c[arm]["turn_hi"] for c in calls),
                "o200k_med": _q([c[arm]["static_o200k"] + c[arm]["turn_o200k"] for c in calls], 50)}

    sz = {"a": sizes("a"), "c": sizes("c")}
    out_c = haiku_allcloud_output_per_task(repo)
    from src.dse import verification_tier as VT
    prices = {}
    for m in (VT.VERIFIER_MODEL, VT.COMPARISON_MODEL):
        rt = rates(repo, m)
        prices[m] = {"a": per_task_prices(rt, sz["a"], OUT_TOKENS_POINT, VT.MAX_TOKENS_VERDICT),
                     "c": per_task_prices(rt, sz["c"], out_c["per_task"], VT.MAX_TOKENS_REDERIVE)}
    rt = rates(repo, VT.VERIFIER_MODEL)
    run = {"a": run_cost(rt, calls, "a", OUT_TOKENS_POINT, VT.MAX_TOKENS_VERDICT),
           "c": run_cost(rt, calls, "c", out_c["per_task"], VT.MAX_TOKENS_REDERIVE)}

    # CEILING: a perfect verifier rejects exactly the accepted wrong turns; each takes its recorded escalation outcome
    src = R.cloud_sources(repo)
    prc = [r for r in _rows(repo / PRC_FILE) if r.get("record") == "prc_task"]
    cells = {r["tier"]: r["headline"] for r in _rows(repo / PRC_FILE) if r.get("record") == "prc_cell"}
    ceil = []
    for tier in R.TIERS:
        tr = [v for (t, _s, _k), v in src["redo"].items() if t == tier]
        tier_mean = {"usd": statistics.fmean(v["usd"] for v in tr), "upper": statistics.fmean(v["upper"] for v in tr)}
        aw = [r for r in prc if r["tier"] == tier and r["decision"] == "accept" and not r["local_ok"]]
        g = [(r, R.grade(True, False, (tier, r["seed"], r["task_id"]), src, tier_mean)) for r in aw]
        h = cells[tier]
        ceil.append({"config": "B3 v2", "tier": tier, "n": h["n"], "now": h["correct"], "accepted_wrong": len(aw),
                     "fixed": sum(x["ok"] for _r, x in g), "ceiling": h["correct"] + sum(x["ok"] for _r, x in g),
                     "usd": sum(x["usd"] for _r, x in g), "upper": sum(x["upper"] for _r, x in g),
                     "srcs": sorted({x["src"] for _r, x in g}),
                     "not_fixed": [f"{r['task_id']} seed {r['seed']} turn {r['turn_idx']}" for r, x in g if x["ok"] < 1],
                     "budget_exhausted": sum(x["budget_exhausted"] for _r, x in g)})
    q4 = q4_pipeline(repo)
    wk = [k for k, v in q4["res"]["per_task"].items() if v == "local" and q4["local"][k]["end_state_ok"] is not True]
    ceil.append({"config": Q4_CONFIG, "tier": "ollama_ctx_28928 (q4_0)", "n": q4["n"], "now": q4["res"]["correct"],
                 "accepted_wrong": len(wk), "fixed": sum(q4["esc"][k]["ok"] for k in wk),
                 "ceiling": q4["res"]["correct"] + sum(q4["esc"][k]["ok"] for k in wk),
                 "usd": sum(q4["esc"][k]["usd"] for k in wk), "upper": None,
                 "srcs": sorted({q4["esc"][k]["src"] for k in wk}),
                 "not_fixed": [f"{k[1]} seed {k[0]}" for k in wk if not q4["esc"][k]["ok"]], "budget_exhausted": 0})
    d = {"cal": cal, "calls": calls, "sizes": sz, "prices": prices, "run": run, "out_c": out_c, "ceiling": ceil,
         "n_recall_stripped": sum(c["recall_probe_stripped"] for c in calls)}
    _CACHE[key] = d
    return d


# ── register texts ─────────────────────────────────────────────────────────────────────────────────

def _cal_text(cal: dict) -> str:
    return (f"token estimator: tiktoken o200k_base x provider/o200k ratio {cal['median']:.4f} (median; range "
            f"{cal['min']:.4f} to {cal['max']:.4f}, max used at high and upper bound) over the first calls of "
            f"{cal['n']} full-history redo tasks ({cal['model']}); ASSUMED to transfer to the verifier payload")


def _by_config(calls):
    out = {}
    for c in calls:
        k = c["config"] if c["config"] != "B3 v2" else f"B3 v2 {c['tier']}"
        o = out.setdefault(k, [0, 0])
        o[0 if not c["local_ok"] else 1] += 1
    return out


def payload_cost_text(d: dict) -> str:
    a, c = d["sizes"]["a"], d["sizes"]["c"]
    h, s = d["prices"]["claude-haiku-5-5"], d["prices"]["claude-sonnet-5-5"]
    n = len(d["calls"])
    return (f"PLANNING (priced, not run): verifier payloads rebuilt for the {n} local task turns L1 to L3 accept "
            f"(no cloud-escalated result), recall-probe suffix stripped from {d['n_recall_stripped']} of them; "
            f"{_cal_text(d['cal'])}. Arm (a) verdict: input tokens median {a['input_med']:,.0f} (static system part "
            f"{a['static_med']:,.0f}, turn part median {a['turn_med']:,.0f}, p90 {a['turn_p90']:,.0f}, max "
            f"{a['turn_max']:,.0f}), max prompt {a['input_max']:,.0f}; output {OUT_TOKENS_POINT} ASSUMED (cap 256). "
            f"USD per verified task, rolling cache breakpoint (static read, turn part written): claude-haiku-5-5 point "
            f"{h['a']['point']:.6f}, high {h['a']['high']:.6f}, upper bound {h['a']['upper']:.6f}; claude-sonnet-5-5 "
            f"point {s['a']['point']:.6f}, high {s['a']['high']:.6f}, upper bound {s['a']['upper']:.6f}. Arm (c) "
            f"re-derivation (comparison): input median {c['input_med']:,.0f}, max {c['input_max']:,.0f}; output "
            f"{d['out_c']['per_task']:.0f} (claude-haiku-5-5 all-cloud B3 v2 mean, {d['out_c']['output_tokens']:,} over "
            f"{d['out_c']['n']} tasks; cap 1024); claude-haiku-5-5 point {h['c']['point']:.6f}, high {h['c']['high']:.6f}, "
            f"upper bound {h['c']['upper']:.6f}; claude-sonnet-5-5 point {s['c']['point']:.6f}, high "
            f"{s['c']['high']:.6f}, upper bound {s['c']['upper']:.6f}")


def accepted_wrong_text(d: dict) -> str:
    bc = _by_config(d["calls"])
    head = "; ".join(f"{k}: {w} wrong + {r} correct accepted" for k, (w, r) in bc.items())
    rows = []
    for c in d["calls"]:
        if not c["local_ok"]:
            rows.append(f"[{c['tier']} seed {c['seed']} turn {c['turn_idx']} {c['task_id']}: "
                        + "; ".join(c["why"]["reasons"]) + "]")
    nw = sum(w for w, _r in bc.values())
    nr = sum(r for _w, r in bc.values())
    return (f"PLANNING: local task turns L1 to L3 accept (the verifier's stream), wrong (end_state_ok false, read after "
            f"the payload is built, scoring only) vs correct: {head}; total {nw} wrong + {nr} correct = {nw + nr}. "
            f"Why each wrong one is visible from the verifier inputs (where the grader's differing record or value sits "
            f"in the payload; the grader is never in it): " + " ".join(rows))


def ceiling_text(d: dict) -> str:
    parts = []
    for c in d["ceiling"]:
        nf = f", not fixed: {', '.join(c['not_fixed'])}" if c["not_fixed"] else ""
        up = f", upper bound USD {c['upper']:.4f}" if c["upper"] is not None else ""
        parts.append(f"{c['config']} {c['tier']}: {c['now']:.0f}/{c['n']} now -> {c['ceiling']:.0f}/{c['n']} "
                     f"({c['fixed']:.0f} of {c['accepted_wrong']} accepted wrong fixed by escalation, sources "
                     f"{'+'.join(c['srcs']) or 'none'}{nf}; escalation USD {c['usd']:.4f} billed{up})")
    return ("CEILING, not a verifier measurement: a perfect verifier after L1 to L3 (rejects exactly the accepted wrong "
            "local results, accepts every correct one, so 0 false escalations); each rejected turn takes its recorded "
            "escalation outcome (own-rollback redo, else full-history redo; q4_0 28928 IMPUTED from the all-cloud "
            "outcome at the imputed USD, as hybrid-full-pipeline-v1); " + "; ".join(parts))


def run_v1_text(d: dict) -> str:
    a, c = d["run"]["a"], d["run"]["c"]
    from src.dse import verification_tier as VT
    return (f"ESTIMATE, not run (waits for the operator's go after the 8192 cloud phase): Run V1, arm (a) "
            f"claude-haiku-5-5 verdict (approved 2026-11-03) on the {len(d['calls'])} accepted local task turns + "
            f"{WARMUP_CALLS} warm-up = {a['n_calls']} calls: billed point USD {a['point']:.4f}, high USD {a['high']:.4f}, "
            f"upper bound USD {a['upper']:.4f}; job cap USD {VT.JOB_CAP_USD:.2f} billed (client spend cap = ledger total "
            f"+ 2.00), per-task cap USD {VT.PER_TASK_CAP_USD:.3f}. Comparison arm (c) claude-haiku-5-5 re-derivation "
            f"(proposal's cheap comparison, NOT approved): {c['n_calls']} calls, point USD {c['point']:.4f}, high USD "
            f"{c['high']:.4f}, upper bound USD {c['upper']:.4f}; (a)+(c) total point USD {a['point'] + c['point']:.4f}, "
            f"high USD {a['high'] + c['high']:.4f}, upper bound USD {a['upper'] + c['upper']:.4f}. Rolling cache "
            f"breakpoint, cache group = one (config, tier, seed) session, first call cold; {_cal_text(d['cal'])}")


def main():
    d = plan(REPO)
    for f in (payload_cost_text, accepted_wrong_text, ceiling_text, run_v1_text):
        print(f.__name__, ":", f(d), "\n")


if __name__ == "__main__":
    main()
