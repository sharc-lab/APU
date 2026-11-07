"""Verification tier Run V1, controller cloud step (docs/EXPERIMENTS.md "PRE-REGISTRATION (2026-11-03):
verification-tier-v1"; design docs/VERIFICATION_TIER_PROPOSAL.md). Operator decision 2026-11-03: "Verifier: write the
live run script, $0.005 per-task cap, no re-derivation arm. Live run waits for my go after the 8192 cloud phase."

What it does: one claude-haiku-5-5 accept/reject verdict (arm (a), src/dse/verification_tier.py, prompt frozen) on each
of the 260 LOCAL task turns L1 to L3 accept (analysis/verification_tier_plan.py::accepted_turns, run order config,
tier, seed, turn), plus 1 warm-up call on the first payload, excluded from statistics (as the pre-registration and the
estimate row verification-tier-run-v1-estimate state). Cloud-escalated results are not verified. Arm (c), the
re-derivation, is NOT in Run V1: any arm other than "a" is refused (adding it needs an amendment filed first).

Payloads are built from recorded files only, through the plan module's client view (grader-stripped rows, recall-probe
suffix stripped, secrets redacted); the grader's end_state_ok is read AFTER the payload is built, for scoring and the
futility rule only, never sent.

Default (no --live): a DRY RUN. No client, no key, no network, nothing written. Builds every payload, prints the call
count, the estimate (the plan module's figures, the same as row verification-tier-run-v1-estimate), the largest
worst-case USD of one call against the per-task cap, and the exact live command.

--live is refused unless: --operator-approval "<text>" is given (always required here, even under USD 2: the
operator's go; console_gate.validate_approval checks it names the job id and an amount >= the job cap, and the
operator confirms it at the console); ANTHROPIC_API_KEY is set in this process (read only by the client, never
printed or written); results/console_usage_check.json exists in the main checkout (then console_gate.pre_job runs the
full Console check, fail closed); the 8192 cloud phase file (results/x2_post_run_check_live_v1.jsonl, row
post-run-check-live-8192) exists with run_end; the job cap is at most USD 2.00; the out file does not exist.

Caps (none of them can be raised here):
  - job cap: a real-mode CloudClient with spend_cap_usd = ledger total + job cap (default and maximum USD 2.00); a
    client refusal stops the job and every task not yet run gets a not_run_job_cap row.
  - per-task cap USD 0.005, two checks before sending: (1) this script's worst case of the call (every payload byte
    counted as one input token at the highest input rate of the price table, cache write included, plus max_tokens at
    the output price) must be <= 0.005, else the call is refused (row refused_task_cap, nothing sent); (2)
    src/cloud/task_budget.BudgetedClient's reservation check (row budget_exhausted). After the call, a BILLED cost above
    0.005 stops the job (row outcome task_cap_exceeded_billed, run_end stopped).
  - stop after 5 replies with no parseable verdict or 5 provider errors (each failed send counts, retries included).
    Amendment 2026-11-07: one retry on a provider timeout, HTTP 429 or 5xx, through the same per-task checks; no
    retry on any other error or on an unparseable reply.
  - futility: stop once false escalations on the development-seed negatives (seeds 20260901 to 20260904) pass 10% of
    those negatives (the count only grows, so this is the pre-registered check, applied as soon as it is decided).

Verdict parsing: after one surrounding code fence is stripped (amendment 2026-11-07), the reply must be exactly one
JSON object with "verdict" "accept" or "reject"; anything else is unparseable and counts as a reject (fail safe),
reported separately. A task with no reply (refused, provider error,
not run) counts as neither caught nor a false escalation and makes the run incomplete; reported separately.

Writes results/verification_tier_run_v1.jsonl: run_start, vt_warmup, vt_call per task, run_end (summary). score()
scores it against the pre-registered targets (row verification-tier-run-v1, PENDING until run_end exists), including
the replay of the q4_0 28928 operating point (operating_point_replay, amendment 2026-11-07) for new correct/90 and the
share of the Sonnet all-cloud bill.

This file starts no process of its own (console_gate reads the git head through harness/proc_util.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "harness", REPO / "src", REPO / "scripts", REPO / "analysis"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cloud import client as cc  # noqa: E402
from cloud import console_gate  # noqa: E402
from cloud import task_budget as TB  # noqa: E402
from src.dse import verification_tier as VT  # noqa: E402

JOB_ID = "verification_tier_run_v1"
OUT = REPO / "results" / "verification_tier_run_v1.jsonl"
PHASE_8192_FILE = "results/x2_post_run_check_live_v1.jsonl"     # row post-run-check-live-8192
PRICE_FILE = REPO / "results" / "cloud_prices_anthropic_20261014.json"
ARMS_ALLOWED = ("a",)          # arm (c), the re-derivation, is not in Run V1 (operator 2026-11-03)
MODEL = VT.VERIFIER_MODEL
MAX_TOKENS = VT.MAX_TOKENS_VERDICT
THINKING = VT.THINKING
TASK_CAP_USD = VT.PER_TASK_CAP_USD     # 0.005
JOB_CAP_MAX_USD = VT.JOB_CAP_USD       # 2.00
WARMUP_CALLS = 1
HELDOUT_SEED = 20260905
FRAMING_TOKENS = 64            # role and message framing, added to the worst case
STOP_UNPARSEABLE = 5
STOP_PROVIDER_ERRORS = 5
RETRIES_PROVIDER_ERROR = 1     # amendment 2026-11-07: one retry on timeout, 429 or 5xx; none on an unparseable reply
FUTILITY_FRACTION = 0.10
TARGETS = {"caught_min": 14, "positives": 16, "false_esc_max": 12, "negatives": 244, "usd_per_task_max": 0.001,
           "latency_median_max_s": 3.0, "n_calls": 261, "correct_90_min": 88, "pct_bill_max": 75.0}
KEY_ENV = cc.ENV_VAR_ANTHROPIC_API_KEY
RECORD = "vt_call"


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_blob(path: Path) -> str:
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def check_arms(arms) -> None:
    extra = [a for a in (arms or ()) if a not in ARMS_ALLOWED]
    if extra or not arms:
        raise SystemExit(f"refused: Run V1 has arm (a) only (claude-haiku-5-5 verdict); arm(s) {extra or arms} are not "
                         "in the run (operator 2026-11-03: no re-derivation arm). Adding one needs the operator's "
                         "decision and an amendment filed in docs/EXPERIMENTS.md first; nothing was sent")


# ── payloads (recorded files only) ─────────────────────────────────────────────────────────────────

def _plan():
    import verification_tier_plan as P  # noqa: PLC0415
    return P


def verdict_payload(es, prior: list[dict], row: dict) -> dict:
    """Arm (a) payload of one grader-stripped turn row, exactly as analysis/verification_tier_plan.py::payloads_for
    builds it (tests/test_verification_tier_cloud.py checks the two agree)."""
    P = _plan()
    import evalset_env as E  # noqa: PLC0415
    v = P.client_view(es, prior, row)
    t = v["turn"]
    return VT.build_verdict_payload(t.user_text, t.recall_key if t.is_recall else None, es.standing, E.RULES_678_TEXT,
                                    P.state_text(E, v["env_before"]), v["trace"], row.get("final_text"), v["changes"],
                                    E.secrets_of(v["env_before"]))


def build_calls(repo: Path = REPO, arms=ARMS_ALLOWED) -> list[dict]:
    """Every Run V1 call (warm-up excluded), in run order. local_ok is the recorded grade, read after the payload is
    built, for scoring and the futility rule only."""
    check_arms(arms)
    calls = []
    for t in _plan().accepted_turns(repo):
        payload = verdict_payload(t["es"], t["prior"], t["row"])
        calls.append({"config": t["config"], "tier": t["tier"], "seed": t["seed"], "turn_idx": t["turn_idx"],
                      "task_id": t["task_id"], "payload": payload,
                      "local_ok": t["full"]["end_state_ok"] is True})
    return calls


def plan_estimate(repo: Path = REPO) -> dict:
    """The estimate of row verification-tier-run-v1-estimate (arm (a) only)."""
    r = _plan().plan(repo)["run"]["a"]
    return {"point": r["point"], "high": r["high"], "upper": r["upper"], "n_calls": r["n_calls"],
            "source": "analysis/verification_tier_plan.py::plan run a (row verification-tier-run-v1-estimate)"}


def load_prices(path: Path = PRICE_FILE) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def worst_case_usd(payload: dict, prices: dict, model: str = MODEL, max_tokens: int = MAX_TOKENS) -> float:
    """Upper bound of one call's cost: every UTF-8 byte of the payload as one input token (a token is at least one
    byte) plus framing, at the highest input rate of the model's tier (uncached, 5 m or 1 h cache write), plus
    max_tokens at the output price."""
    m = prices["models"][model]
    n_in = len((payload["system"] + payload["user"]).encode("utf-8")) + FRAMING_TOKENS
    above = m.get("long_context_above")
    r = m["standard"]["long" if above and n_in > above else "short"]
    in_rate = max(float(r[k]) for k in ("input", "cache_write", "cache_write_1h") if k in r)
    return (n_in * in_rate + max_tokens * float(r["output"])) / 1e6


def payload_sha(payload: dict) -> str:
    return hashlib.sha256((payload["system"] + "\0" + payload["user"]).encode("utf-8")).hexdigest()


def strip_fence(text: str) -> str:
    """One surrounding code fence removed (amendment 2026-11-07): ```json ... ``` or ``` ... ``` around the whole
    reply. Only one; anything outside it, or a second fence, leaves the text as it was (then unparseable)."""
    t = text.strip()
    if not (t.startswith("```") and t.endswith("```") and len(t) >= 6):
        return t
    first_nl = t.find("\n")
    if first_nl < 0:
        return t
    head, body = t[3:first_nl].strip(), t[first_nl + 1:-3]
    if head not in ("", "json") or "```" in body:
        return t
    return body.strip()


def parse_verdict(content: str | None) -> tuple[str | None, str | None]:
    """(verdict, reason): verdict None when the reply, after one surrounding code fence is stripped, is not exactly one
    JSON object with verdict accept or reject."""
    try:
        d = json.loads(strip_fence(content or ""))
    except (ValueError, TypeError):
        return None, None
    if not isinstance(d, dict) or d.get("verdict") not in ("accept", "reject"):
        return None, None
    return d["verdict"], (str(d.get("reason")) if d.get("reason") is not None else None)


def retryable_error(e: BaseException) -> bool:
    """A timeout, HTTP 429 or HTTP 5xx from the provider (the only errors retried, once)."""
    if "timeout" in type(e).__name__.lower():
        return True
    code = getattr(e, "status_code", None)
    if code is None:
        code = getattr(getattr(e, "response", None), "status_code", None)
    try:
        code = int(code)
    except (TypeError, ValueError):
        return False
    return code == 429 or 500 <= code <= 599


def _usage(res) -> dict:
    u = getattr(res, "usage", None) or {}
    return {k: u.get(k) for k in ("input_tokens", "uncached_input_tokens", "cache_read_tokens", "cache_write_tokens",
                                  "output_tokens")}


# ── the run ────────────────────────────────────────────────────────────────────────────────────────

def _one_call(lane, c: dict, key: str, clock) -> dict:
    """One verdict call through the per-task checks. Returns the call fields; raises SpendCapExceeded (job cap)."""
    out = {"worst_case_usd": c["worst_case_usd"], "payload_sha256": payload_sha(c["payload"])}
    if c["worst_case_usd"] > TASK_CAP_USD + 1e-12:
        return {**out, "outcome": "refused_task_cap", "sent": False,
                "reason": f"worst case USD {c['worst_case_usd']:.6f} > per-task cap USD {TASK_CAP_USD:.3f}"}
    lane.start_task(key)
    errors = []
    for attempt in range(1 + RETRIES_PROVIDER_ERROR):
        try:
            prepared = lane.prepare(model_id=MODEL, messages=VT.messages(c["payload"]), max_tokens=MAX_TOKENS,
                                    thinking=THINKING, custom_llm_provider="anthropic")
        except TB.TaskBudgetExhausted as e:
            return {**out, "outcome": TB.OUTCOME_EXHAUSTED, "sent": False, "task_budget": e.record,
                    "provider_errors": errors, **lane.task_state()}
        t0 = clock()
        try:
            raw = prepared.send()
            break
        except Exception as e:  # noqa: BLE001 - send() released the reservation; no ledger row was written
            errors.append({"error": f"{type(e).__name__}: {str(e)[:300]}", "retryable": retryable_error(e)})
            if not errors[-1]["retryable"] or attempt == RETRIES_PROVIDER_ERROR:
                return {**out, "outcome": "provider_error", "sent": False, "error": errors[-1]["error"],
                        "provider_errors": errors}
    dt = clock() - t0
    res = prepared.settle(raw)
    verdict, reason = parse_verdict(res.content)
    rec = {**out, "outcome": "sent", "sent": True, "verdict": verdict, "parse_ok": verdict is not None,
           "fenced": (res.content or "").strip().startswith("```"), "provider_errors": errors,
           "reason": (reason or "")[:500], "reply": (res.content or "")[:2000] if verdict is None else None,
           "usd_billed": float(res.cost_usd), "usd_upper_bound": res.cost_usd_upper_bound, "latency_s": dt,
           "tokens": _usage(res), **lane.task_state()}
    if float(res.cost_usd) > TASK_CAP_USD + 1e-12:
        rec["outcome"] = "task_cap_exceeded_billed"
    return rec


def run(out: Path = OUT, live: bool = False, client=None, job_cap: float = JOB_CAP_MAX_USD,
        ledger: Path | None = None, approval: str | None = None, arms=ARMS_ALLOWED, calls: list[dict] | None = None,
        estimate: dict | None = None, gate_fn=None, emit_print=print, clock=time.perf_counter,
        prices: dict | None = None) -> list[dict]:
    """Live Run V1 (live=True) or a dry run (live=False: no client used, nothing written, returns the run_start)."""
    check_arms(arms)
    if not (0 < job_cap <= JOB_CAP_MAX_USD + 1e-9):
        raise SystemExit(f"refused: job cap ${job_cap:.2f} is not in (0, {JOB_CAP_MAX_USD:.2f}]; nothing was sent")
    prices = prices or load_prices()
    calls = build_calls() if calls is None else calls
    estimate = estimate or plan_estimate()
    for c in calls:
        c["worst_case_usd"] = worst_case_usd(c["payload"], prices)
    dev_neg = sum(1 for c in calls if c["local_ok"] and c["seed"] != HELDOUT_SEED)
    start = {"record": "run_start", "harness": JOB_ID, "ts_utc": utc_iso(), "live": live, "model_id": MODEL,
             "arms": list(arms), "max_tokens": MAX_TOKENS, "thinking": THINKING, "task_cap_usd": TASK_CAP_USD,
             "job_cap_usd": job_cap, "warmup_calls": WARMUP_CALLS, "n_tasks": len(calls),
             "n_calls": len(calls) + WARMUP_CALLS, "positives": sum(1 for c in calls if not c["local_ok"]),
             "negatives": sum(1 for c in calls if c["local_ok"]), "dev_negatives": dev_neg,
             "heldout_seed": HELDOUT_SEED, "estimate": estimate, "targets": TARGETS,
             "max_worst_case_usd": max((c["worst_case_usd"] for c in calls), default=0.0),
             "verifier_instruction_sha256": hashlib.sha256(VT.VERIFIER_INSTRUCTION.encode("utf-8")).hexdigest(),
             "script_git_blob": git_blob(Path(__file__)),
             "vt_module_git_blob": git_blob(REPO / "src" / "dse" / "verification_tier.py"),
             "stop_rules": {"unparseable": STOP_UNPARSEABLE, "provider_errors": STOP_PROVIDER_ERRORS,
                            "futility_fraction_dev_negatives": FUTILITY_FRACTION}}
    start["matches_estimate_calls"] = start["n_calls"] == estimate["n_calls"]
    if not live:
        return [start]
    if not start["matches_estimate_calls"]:
        raise SystemExit(f"refused: {start['n_calls']} calls built, the estimate has {estimate['n_calls']}; nothing "
                         "was sent")
    if client is None or getattr(client, "stub_mode", True):
        raise SystemExit("refused: --live needs a real-mode client; nothing was sent")
    if not (approval and str(approval).strip()):
        raise SystemExit("refused: the live run needs the operator's go on the command line "
                         f"({console_gate.approval_syntax(JOB_ID, job_cap)}); nothing was sent")
    if Path(out).exists():
        raise SystemExit(f"refused: {Path(out).name} exists (would be overwritten); nothing was sent")
    gate = (gate_fn or console_gate.pre_job)(
        client, job_id=JOB_ID, estimate_usd=job_cap, estimate_basis="job cap (Run V1 estimate upper bound "
        f"USD {estimate['upper']:.4f}, row verification-tier-run-v1-estimate)",
        projection_billed_usd=min(estimate["high"], job_cap), projection_upper_usd=estimate["upper"],
        approval=approval)
    start["console_check"] = gate
    lane = TB.BudgetedClient(client, TASK_CAP_USD, ledger_path=ledger)
    spend_cap_errors = tuple({cc.SpendCapExceeded, _src_spend_cap()})
    rows: list[dict] = []
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        def emit(row):
            rows.append(row)
            fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            fh.flush()

        emit(start)
        stopped, n_unparse, n_err, fe_dev = None, 0, 0, 0
        for w in range(WARMUP_CALLS):
            c = calls[0]
            try:
                r = _one_call(lane, c, f"warmup:{w}", clock)
            except spend_cap_errors as e:
                r = {"outcome": "not_run_job_cap", "sent": False, "reason": str(e)[:300]}
                stopped = f"job cap: {e}"
            emit({"record": "vt_warmup", "excluded_from_stats": True, "on_payload_of": _key(c), **r})
            n_err += len(r.get("provider_errors") or ())
            if r.get("outcome") == "task_cap_exceeded_billed":
                stopped = f"per-task cap: warm-up billed USD {r['usd_billed']:.6f} > USD {TASK_CAP_USD:.3f}"
        for c in calls:
            base = {"record": RECORD, **{k: c[k] for k in ("config", "tier", "seed", "turn_idx", "task_id")},
                    "heldout": c["seed"] == HELDOUT_SEED}
            if stopped:
                r = {"outcome": "not_run_job_cap" if stopped.startswith("job cap") else "not_run_stopped",
                     "sent": False}
            else:
                try:
                    r = _one_call(lane, c, _key(c), clock)
                except spend_cap_errors as e:
                    stopped = f"job cap: {e}"
                    r = {"outcome": "not_run_job_cap", "sent": False, "reason": str(e)[:300]}
            row = {**base, **r, "local_ok": c["local_ok"]}   # grader outcome, read after the call, scoring only
            emit(row)
            if stopped:
                continue
            n_err += len(r.get("provider_errors") or ())
            n_unparse += r.get("sent", False) and not r.get("parse_ok")
            if r["outcome"] == "task_cap_exceeded_billed":
                stopped = f"per-task cap: {_key(c)} billed USD {r['usd_billed']:.6f} > USD {TASK_CAP_USD:.3f}"
            elif n_unparse >= STOP_UNPARSEABLE:
                stopped = f"stop rule: {n_unparse} replies with no parseable verdict"
            elif n_err >= STOP_PROVIDER_ERRORS:
                stopped = f"stop rule: {n_err} provider errors"
            if c["local_ok"] and not row["heldout"] and _rejects(row):
                fe_dev += 1
                if fe_dev > FUTILITY_FRACTION * dev_neg:
                    stopped = stopped or (f"futility: {fe_dev} false escalations on the {dev_neg} development-seed "
                                          f"negatives, above {FUTILITY_FRACTION:.0%}")
        tasks = [x for x in rows if x["record"] == RECORD]
        emit({"record": "run_end", "ts_utc": utc_iso(), "stopped": stopped, "summary": summarize(rows),
              "n_tasks_recorded": len(tasks), "ledger_total_usd": getattr(client, "running_total_usd", None)})
    emit_print(json.dumps(rows[-1]["summary"], default=str))
    return rows


def _src_spend_cap():
    from src.cloud import client as scc  # noqa: PLC0415  (the same class under its other import name)
    return scc.SpendCapExceeded


def _key(c: dict) -> str:
    return f"{c['config']}:{c['tier']}:{c['seed']}:{c['turn_idx']}:{c['task_id']}"


# ── scoring ────────────────────────────────────────────────────────────────────────────────────────

def _rejects(t: dict) -> bool:
    """A sent call that rejects, or whose reply has no parseable verdict (fail safe: counts as a reject)."""
    return bool(t.get("sent")) and t.get("verdict") != "accept"


def _counts(tasks: list[dict]) -> dict:
    pos = [t for t in tasks if not t["local_ok"]]
    neg = [t for t in tasks if t["local_ok"]]
    return {"positives": len(pos), "negatives": len(neg), "caught": sum(map(_rejects, pos)),
            "false_esc": sum(map(_rejects, neg))}


def summarize(rows: list[dict]) -> dict:
    """caught/positives, false escalations/negatives, added USD and latency against the pre-registered targets; the
    held-out seed's figures; unparseable replies; tasks with no verdict. Warm-up excluded from every statistic."""
    tasks = [r for r in rows if r.get("record") == RECORD]
    warm = [r for r in rows if r.get("record") == "vt_warmup"]
    sent = [t for t in tasks if t.get("sent")]
    usd = [float(t.get("usd_billed") or 0.0) for t in sent]
    lat = [float(t["latency_s"]) for t in sent if t.get("latency_s") is not None]
    allc = _counts(tasks)
    held = _counts([t for t in tasks if t.get("heldout")])
    by = {}
    for t in tasks:
        k = t["config"] if t["config"] != "B3 v2" else f"B3 v2 {t['tier']}"
        by[k] = _counts([x for x in tasks if (x["config"] if x["config"] != "B3 v2" else f"B3 v2 {x['tier']}") == k])
    no_verdict = [t for t in tasks if not t.get("sent")]
    usd_task = (sum(usd) / len(sent)) if sent else None
    lat_med = statistics.median(lat) if lat else None
    s = {"n_tasks": len(tasks), **allc, "unparseable": sum(1 for t in sent if not t.get("parse_ok")),
         "no_verdict": len(no_verdict),
         "no_verdict_by_outcome": {o: sum(1 for t in no_verdict if t.get("outcome") == o)
                                   for o in sorted({t.get("outcome") for t in no_verdict})},
         "sent": len(sent), "usd_billed_total": sum(usd), "usd_billed_per_task_mean": usd_task,
         "latency_median_s": lat_med, "latency_p90_s": (statistics.quantiles(lat, n=10, method="inclusive")[8]
                                                        if len(lat) > 1 else (lat[0] if lat else None)),
         "warmup_usd_billed": sum(float(w.get("usd_billed") or 0.0) for w in warm),
         "usd_billed_incl_warmup": sum(usd) + sum(float(w.get("usd_billed") or 0.0) for w in warm),
         "heldout": held, "by_config": by}
    s["targets_met"] = {
        "caught": allc["caught"] >= TARGETS["caught_min"],
        "false_esc": allc["false_esc"] <= TARGETS["false_esc_max"],
        "usd_per_task": usd_task is not None and usd_task <= TARGETS["usd_per_task_max"],
        "latency_median": lat_med is not None and lat_med <= TARGETS["latency_median_max_s"]}
    s["complete"] = not no_verdict and allc["positives"] == TARGETS["positives"] \
        and allc["negatives"] == TARGETS["negatives"]
    return s


def operating_point_replay(rows: list[dict], repo: Path = REPO) -> dict:
    """Amendment 2026-11-07: the q4_0 28928 operating point through hybrid-full-pipeline-v1's replay at the rolling
    cache breakpoint (rolling_cache_reprice.operating_point_rolling, usd_new: the 85/90 and 43.4% of row
    operating-point-q4-28928-rolling-cache), then the verifier's verdicts on the turns it keeps local: a rejected
    (or unparseable) turn takes its recorded escalation outcome and USD (own-rollback redo, else full-history redo,
    else IMPUTED, as that row); a turn with no reply stays local. The verifier's billed USD on those turns is added to
    the pipeline's bill. The per-task budget is not re-applied to the new escalations (as the CEILING row)."""
    P = _plan()
    P._paths(repo)
    import hybrid_full_pipeline as hfp  # noqa: PLC0415
    import rolling_cache_reprice as rc  # noqa: PLC0415
    ac, rd = rc.all_cloud_rolling(repo), rc.redo_rolling(repo)
    op = rc.operating_point_rolling(repo, ac=ac, rd=rd, configs=(P.Q4_CONFIG,))["usd_new"]
    cell, allc = op["cells"][P.Q4_CONFIG], op["all_cloud"]
    cloud = {k: {"ok": v["ok"], "usd": v["usd_new"]} for k, v in ac[hfp.CLOUD_MODEL].items() if k[0] in set(hfp.SEEDS)}
    redo = {k: {"ok": v["ok"], "usd": v["usd_new"]} for k, v in rd["redo"].items()}
    rb = {k: {"ok": v["ok"], "usd": v["usd_new"]} for k, v in rd["rb"].items()}
    esc = hfp.escalation_outcomes(repo, hfp.CONFIGS[P.Q4_CONFIG][2], cloud, redo, rb, op["imputed_usd"])
    local, _first, _ceil = hfp.config_sessions(repo, P.Q4_CONFIG)
    vt = {(t["seed"], t["task_id"]): t for t in rows if t.get("record") == RECORD and t.get("config") == P.Q4_CONFIG}
    kept = [k for k, v in cell["per_task"].items() if v == "local"]
    rejected = [k for k in kept if k in vt and _rejects(vt[k])]
    correct = cell["correct"] + sum(esc[k]["ok"] - (local[k]["end_state_ok"] is True) for k in rejected)
    usd_esc = sum(esc[k]["usd"] for k in rejected)
    usd_vt = sum(float(t.get("usd_billed") or 0.0) for t in vt.values() if t.get("sent"))
    n = cell["n"]
    share = 100 * (cell["usd"] + usd_esc + usd_vt) / n / allc
    return {"config": P.Q4_CONFIG, "n": n, "correct_before": cell["correct"], "pct_bill_before": cell["pct_bill"],
            "kept_local": len(kept), "verified": sum(1 for k in kept if k in vt and vt[k].get("sent")),
            "no_reply": sum(1 for k in kept if not (k in vt and vt[k].get("sent"))), "rejected": len(rejected),
            "rejected_fixed": sum(esc[k]["ok"] for k in rejected),
            "rejected_were_correct": sum(1 for k in rejected if local[k]["end_state_ok"] is True),
            "imputed": sum(1 for k in rejected if esc[k].get("src") == "imputed"), "correct": correct,
            "usd_escalations_added": usd_esc, "usd_verifier": usd_vt, "all_cloud_usd_task": allc, "pct_bill": share}


def load_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(x) for x in fh if x.strip()]


def score(path: Path = OUT, repo: Path = REPO) -> dict:
    """The summary of a finished Run V1 file (needs run_end; the register row stays PENDING without it)."""
    rows = load_rows(path)
    end = [r for r in rows if r.get("record") == "run_end"]
    if not end:
        raise RuntimeError(f"{Path(path).name} has no run_end; Run V1 not finished")
    s = summarize(rows)
    s["stopped"] = end[-1].get("stopped")
    s["operating_point"] = op = operating_point_replay(rows, repo)
    s["targets_met"]["correct_90"] = op["correct"] >= TARGETS["correct_90_min"]
    s["targets_met"]["pct_bill"] = op["pct_bill"] <= TARGETS["pct_bill_max"]
    return s


def score_text(s: dict) -> str:
    """Register text: counts only (never a percentage of one hundred for a catch figure)."""
    t, h = s["targets_met"], s["heldout"]
    met = lambda k: "met" if t[k] else "missed"  # noqa: E731
    usd = s["usd_billed_per_task_mean"]
    lat = s["latency_median_s"]
    by = "; ".join(f"{k}: caught {v['caught']}/{v['positives']}, false escalations {v['false_esc']}/{v['negatives']}"
                   for k, v in s["by_config"].items())
    return (f"Run V1, claude-haiku-5-5 verdict (arm (a)) on the local task turns L1 to L3 accept, {s['n_tasks']} tasks "
            f"+ 1 warm-up (excluded): failures caught {s['caught']}/{s['positives']} (target >= "
            f"{TARGETS['caught_min']}/{TARGETS['positives']}, {met('caught')}); false escalations {s['false_esc']}/"
            f"{s['negatives']} (target <= {TARGETS['false_esc_max']}/{TARGETS['negatives']}, {met('false_esc')}); added "
            f"USD per verified task billed {'n/a' if usd is None else f'{usd:.6f}'} (target <= "
            f"{TARGETS['usd_per_task_max']:.3f}, {met('usd_per_task')}); added latency median "
            f"{'n/a' if lat is None else f'{lat:.2f}'} s (target <= {TARGETS['latency_median_max_s']:.0f} s, "
            f"{met('latency_median')}); billed total USD {s['usd_billed_incl_warmup']:.4f} incl warm-up; unparseable "
            f"replies {s['unparseable']} (counted as rejects); tasks with no verdict {s['no_verdict']} "
            f"{s['no_verdict_by_outcome']}; held-out seed {HELDOUT_SEED}: caught {h['caught']}/{h['positives']}, false "
            f"escalations {h['false_esc']}/{h['negatives']}; per config: {by}; stopped: {s.get('stopped')}; "
            f"complete: {s['complete']}. " + operating_point_text(s))


def operating_point_text(s: dict) -> str:
    o, t = s["operating_point"], s["targets_met"]
    met = lambda k: "met" if t[k] else "missed"  # noqa: E731
    return (f"REPLAY of the {o['config']} operating point with these verdicts (rolling cache breakpoint, escalations "
            f"from recorded redo rows or IMPUTED as hybrid-full-pipeline-v1): {o['correct_before']}/{o['n']} -> "
            f"{o['correct']}/{o['n']} correct (target >= {TARGETS['correct_90_min']}/{o['n']}, {met('correct_90')}); "
            f"{o['rejected']} of {o['kept_local']} kept-local turns rejected ({o['rejected_fixed']} escalations "
            f"correct, {o['rejected_were_correct']} were correct locally, {o['imputed']} imputed; {o['no_reply']} with "
            f"no reply stay local); share of the Sonnet all-cloud bill {o['pct_bill_before']:.1f}% -> "
            f"{o['pct_bill']:.1f}% incl the verifier's USD {o['usd_verifier']:.4f} (target <= "
            f"{TARGETS['pct_bill_max']:.0f}%, {met('pct_bill')})")


# ── command line ───────────────────────────────────────────────────────────────────────────────────

def live_command(job_cap: float = JOB_CAP_MAX_USD) -> str:
    return (f"py -3.12 scripts/verification_tier_cloud.py --live --ledger results/cloud_ledger.jsonl --job-cap "
            f"{job_cap:.2f} {console_gate.APPROVAL_FLAG} \"approve {JOB_ID} up to USD {job_cap:.2f}: Run V1 go after "
            "the 8192 cloud phase\"")


def dry_run_text(start: dict) -> str:
    e = start["estimate"]
    return "\n".join([
        f"DRY RUN (no client, no key, no network, nothing written): {JOB_ID}, model {MODEL}, arm (a) only",
        f"calls: {start['n_tasks']} accepted local task turns + {WARMUP_CALLS} warm-up = {start['n_calls']} calls "
        f"(positives {start['positives']}, negatives {start['negatives']}, development-seed negatives "
        f"{start['dev_negatives']})",
        f"estimate (row verification-tier-run-v1-estimate, {e['n_calls']} calls): billed point USD {e['point']:.4f}, "
        f"high USD {e['high']:.4f}, upper bound USD {e['upper']:.4f}; call count matches the estimate: "
        f"{'yes' if start['matches_estimate_calls'] else 'NO'}",
        f"caps: job cap USD {start['job_cap_usd']:.2f} billed (client spend cap = ledger total + job cap), per-task cap "
        f"USD {TASK_CAP_USD:.3f}; largest worst-case call USD {start['max_worst_case_usd']:.6f} "
        f"({'under' if start['max_worst_case_usd'] <= TASK_CAP_USD else 'OVER'} the per-task cap)",
        "live run waits for the operator's go after the 8192 cloud phase (row post-run-check-live-8192); from the "
        "main checkout, with ANTHROPIC_API_KEY set and results/console_usage_check.json fresh:",
        "  " + live_command(start["job_cap_usd"])])


def _has_run_end(path: Path) -> bool:
    try:
        return any(json.loads(x).get("record") == "run_end" for x in path.read_text(encoding="utf-8").splitlines()
                   if x.strip())
    except (OSError, ValueError):
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--live", action="store_true", help="send the verdict calls (needs ANTHROPIC_API_KEY)")
    ap.add_argument("--ledger", type=Path, default=cc.DEFAULT_LEDGER_PATH,
                    help="the shared ledger (pass the main checkout's results/cloud_ledger.jsonl)")
    ap.add_argument("--job-cap", type=float, default=JOB_CAP_MAX_USD, help="USD billed, at most 2.00")
    ap.add_argument("--arm", action="append", default=None, help="only 'a' is accepted (no re-derivation arm)")
    ap.add_argument(console_gate.APPROVAL_FLAG, dest="operator_approval", default=None)
    a = ap.parse_args(argv)
    arms = tuple(a.arm) if a.arm else ARMS_ALLOWED

    def refuse(msg: str) -> int:
        print(f"refused: {msg}; nothing was sent", file=sys.stderr)
        return 2
    try:
        check_arms(arms)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 2
    if not (0 < a.job_cap <= JOB_CAP_MAX_USD + 1e-9):
        return refuse(f"job cap ${a.job_cap:.2f} is not in (0, {JOB_CAP_MAX_USD:.2f}]")
    if not a.live:
        start = run(live=False, job_cap=a.job_cap, arms=arms)[0]
        print(dry_run_text(start))
        return 0
    if not (a.operator_approval and a.operator_approval.strip()):
        return refuse(f"--live needs the operator's go: {console_gate.approval_syntax(JOB_ID, a.job_cap)}")
    if not os.environ.get(KEY_ENV):
        return refuse(f"--live needs {KEY_ENV} set in the environment")
    check_path = console_gate.default_check_path()
    if not check_path.exists():
        return refuse(f"--live needs the Console check file {check_path} (console_gate.TEMPLATE), filled and fresh")
    if not _has_run_end(REPO / PHASE_8192_FILE):
        return refuse(f"--live waits for the 8192 cloud phase: {PHASE_8192_FILE} has no run_end here")
    if Path(a.out).exists():
        return refuse(f"{Path(a.out).name} exists (would be overwritten)")
    total0 = cc.CloudClient(mode="stub", ledger_path=a.ledger)._read_running_total()
    client = cc.CloudClient(mode="real", spend_cap_usd=total0 + min(a.job_cap, JOB_CAP_MAX_USD), ledger_path=a.ledger)
    rows = run(a.out, True, client, a.job_cap, a.ledger, a.operator_approval, arms)
    return 0 if rows[-1].get("stopped") is None else 4


if __name__ == "__main__":
    raise SystemExit(main())
