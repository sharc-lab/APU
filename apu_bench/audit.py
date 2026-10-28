"""apu_bench audit ("bring your own logs"): replay recorded agent sessions against a device profile and report the
hidden failure rate and the cloud cost. Design: docs/BYOL_AUDIT_DESIGN.md.

Input (schema apu-audit-session/1): JSONL, one request per line (an optional first line {"schema": ...} is a header):
  session_id        str, required
  request_idx       int, order within the session, required
  task_id           str, optional: groups requests into a task (default: the whole session is one task)
  messages          OpenAI-style chat messages as sent [{"role", "content", ...}] (may be omitted when redacted)
  tools             the tool schema sent, optional
  n_messages        int, when messages are redacted (needed only to read Ollama's message-drop line exactly)
  prompt_chars      int, when messages are redacted (chars of messages + tools, for the chars/4 estimate)
  prompt_tokens, completion_tokens, cached_prompt_tokens, cache_write_tokens   provider usage, optional
  completion        str, optional (chars/4 of it when completion_tokens is absent)
  http_status, error   what the client saw, optional (absent = 200, no error)
  outcome           {"ok": bool}, optional: the task's graded outcome (on any request of the task; last one wins)
  ran_at            {"config", "window"}, optional: where the request was actually served (needed for the recorded
                    modes; config is a free label, e.g. "ollama_ctx_4096")
  recorded_verdict  {"flagged": bool, "no_log": bool}, optional: a detector v2 verdict already computed from the
                    server log at run time (our own harness writes these)

Ground truth for "the runtime cut the context", per request, in order of preference:
  server_log        the runtime's own debug log of the run (Ollama OLLAMA_DEBUG=1): sliced per /api/chat request at
                    the [GIN] access line, then detector v2 (harness/own_compaction.cut_v2_verdict): a message-drop
                    line ("truncating input messages which exceed context length"), a token cut ("truncating input
                    prompt"), a llama.cpp context shift, or an exceed-context error. With messages redacted and no
                    n_messages, a message-drop line counts as a drop (Ollama writes it only when it truncates);
  recorded_verdict  the verdict written at run time from the same log lines (the log itself was not kept);
  estimate          no log: projected tokens (chars/4 x calibration) vs the window, through src/dse/fits_locally
                    (PREDICTED, not ground truth; labelled so in the report).
Hidden = cut by the runtime while the client saw HTTP 200 and no error. A task is a hidden failure when any of its
requests is a hidden cut; wrong tasks are split into those with a hidden cut and those without.

Cloud cost (src/cloud/prices.py, the committed price tables), per session on --cloud-model:
  upper bound  every input token at the input rate + output at the output rate (no caching);
  floor        the last prompt at the input rate, every earlier prompt token at the cache-read rate (as
               analysis/r2_session_cost.session_cost "best caching"; ignores cache-write surcharges);
  billed       when every request carries cached_prompt_tokens (provider usage): uncached input at the input rate,
               cached at the cache-read rate, cache writes (if given) at the cache-write rate, output at output rate.
Token basis: provider usage when present, else chars/4 (labelled).

Live replay (sending the sessions to a runtime) is not implemented: docs/BYOL_AUDIT_DESIGN.md says what it needs.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

from apu_bench import AUDIT_SCHEMA, BENCH_VERSION, SESSION_SCHEMA

REPO = Path(__file__).resolve().parents[1]
_GIN = re.compile(r"\[GIN\].*?\|\s*(\d{3})\s*\|.*?\"(/api/[\w/]+)\"")
CHAT_PATHS = ("/api/chat", "/v1/chat/completions")
EST_CHARS_PER_TOKEN = 4


def _mech():
    h = str(REPO / "harness")
    if h not in sys.path:
        sys.path.insert(0, h)
    import x2_r2_mechanism as mech
    return mech


# ── input ────────────────────────────────────────────────────────────────────────────────────────────

def load_requests(path) -> list[dict]:
    out = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        r = json.loads(line)
        if "schema" in r and "session_id" not in r:
            if r["schema"] != SESSION_SCHEMA:
                raise ValueError(f"unknown session schema {r['schema']!r}")
            continue
        for k in ("session_id", "request_idx"):
            if k not in r:
                raise ValueError(f"line {i + 1}: missing {k}")
        out.append(r)
    return out


# ── ground truth from the runtime log ────────────────────────────────────────────────────────────────

def slice_server_log(text: str) -> list[dict]:
    """One slice per chat request, in order: the lines since the previous access ([GIN]) line up to and including
    this request's own access line (Ollama writes the drop / cut lines before the request's access line)."""
    out, buf = [], []
    for line in text.splitlines():
        buf.append(line)
        m = _GIN.search(line)
        if m:
            if m.group(2) in CHAT_PATHS:
                out.append({"http_status": int(m.group(1)), "path": m.group(2), "text": "\n".join(buf)})
            buf = []
    return out


def log_verdict(text: str, n_messages: int | None) -> dict:
    """Detector v2 on one request's log slice (own_compaction.cut_v2_verdict's rules, with n_messages optional)."""
    mech = _mech()
    ev = mech.parse_call_log(text)
    evidence = []
    if ev.get("msg_truncation_value") is not None:
        if n_messages is None:
            evidence.append("older messages dropped (message count unknown: truncation line taken as a drop)")
        else:
            idx = mech.first_kept_index(ev, n_messages)
            if idx is not None and idx > 0:
                evidence.append(f"older messages dropped (first kept: message {idx} of {n_messages})")
    if ev.get("token_cut"):
        evidence.append("prompt tokens cut ({prompt} -> {new})".format(**ev["token_cut"]))
    if ev.get("context_shifts"):
        evidence.append(f"context shift x{len(ev['context_shifts'])}")
    if ev.get("exceed_context_error"):
        evidence.append("exceeds the context size")
    np_ = ev.get("new_prompt") or []
    return {"method": "server_log", "cut": bool(evidence), "evidence": evidence,
            "used_tokens": np_[-1]["n_tokens"] if np_ else None}


# ── tokens and estimate ──────────────────────────────────────────────────────────────────────────────

def prompt_chars(r: dict) -> int | None:
    if r.get("prompt_chars") is not None:
        return int(r["prompt_chars"])
    if r.get("messages") is None:
        return None
    n = sum(len(m.get("content") or "") if isinstance(m.get("content"), str) else len(json.dumps(m.get("content")))
            for m in r["messages"])
    return n + (len(json.dumps(r["tools"])) if r.get("tools") else 0)


def prompt_tokens(r: dict) -> tuple[int | None, str]:
    if r.get("prompt_tokens") is not None:
        return int(r["prompt_tokens"]), "provider usage"
    c = prompt_chars(r)
    return (math.ceil(c / EST_CHARS_PER_TOKEN), "chars/4") if c is not None else (None, "none")


def completion_tokens(r: dict) -> tuple[int, str]:
    if r.get("completion_tokens") is not None:
        return int(r["completion_tokens"]), "provider usage"
    if r.get("completion"):
        return math.ceil(len(r["completion"]) / EST_CHARS_PER_TOKEN), "chars/4"
    return 0, "none"


def estimate_verdict(r: dict, window: int, calib: float, predictor=None) -> dict:
    from src.dse.fits_locally import DEFAULT_PREDICTOR
    c = prompt_chars(r)
    if c is None:
        return {"method": "estimate", "cut": None, "evidence": ["no prompt text or prompt_chars"]}
    p = (predictor or DEFAULT_PREDICTOR).predict({"est_tokens": math.ceil(c / EST_CHARS_PER_TOKEN), "calib": calib},
                                                 {"window": window})
    return {"method": "estimate", "cut": not p["fits"], "projected_tokens": p["projected_tokens"],
            "ceiling_tokens": p["ceiling_tokens"], "source": p["source"],
            "evidence": [] if p["fits"] else [f"projected {p['projected_tokens']} > ceiling {p['ceiling_tokens']}"]}


# ── cost ─────────────────────────────────────────────────────────────────────────────────────────────

def price_session(reqs: list[dict], rates: dict) -> dict | None:
    from src.cloud import prices as cp
    ps, cs, bases = [], [], set()
    for r in sorted(reqs, key=lambda x: x["request_idx"]):
        p, b = prompt_tokens(r)
        if p is None:
            return None
        c, b2 = completion_tokens(r)
        ps.append(p)
        cs.append(c)
        bases |= {b, b2} - {"none"}
    rt = cp.rate_for(rates, max(ps))
    cached_rate = rt["cached_input"] if rt.get("cached_input") is not None else rt["input"]
    out_usd = sum(cs) * rt["output"]
    upper = (sum(ps) * rt["input"] + out_usd) / 1e6
    floor = (ps[-1] * rt["input"] + (sum(ps) - ps[-1]) * cached_rate + out_usd) / 1e6
    billed = None
    if all(r.get("cached_prompt_tokens") is not None for r in reqs):
        wr = rt.get("cache_write") if rt.get("cache_write") is not None else rt["input"]
        billed = sum((p - int(r["cached_prompt_tokens"]) - int(r.get("cache_write_tokens") or 0)) * rt["input"]
                     + int(r["cached_prompt_tokens"]) * cached_rate + int(r.get("cache_write_tokens") or 0) * wr
                     for p, r in zip(ps, sorted(reqs, key=lambda x: x["request_idx"]))) / 1e6 + out_usd / 1e6
    return {"upper_bound_usd": upper, "floor_usd": floor, "billed_usd": billed, "input_tokens": sum(ps),
            "output_tokens": sum(cs), "token_basis": sorted(bases) or ["none"]}


# ── the audit ────────────────────────────────────────────────────────────────────────────────────────

def _task_key(r):
    return (r["session_id"], r.get("task_id"))


def audit_group(reqs: list[dict], window: int, slices: list[dict] | None, calib: float, rates: dict | None,
                predictor=None) -> dict:
    """Verdicts, hidden failures and cost for the requests served at one (config, window)."""
    reqs = sorted(reqs, key=lambda r: (r.get("_order", 0), r["session_id"], r["request_idx"]))
    if slices is not None:
        method = "server_log"
        if len(slices) != len(reqs):
            raise ValueError(f"window {window}: {len(slices)} chat requests in the log vs {len(reqs)} in the sessions")
        verdicts = []
        for r, s in zip(reqs, slices):
            n = r.get("n_messages", len(r["messages"]) if r.get("messages") is not None else None)
            v = log_verdict(s["text"], n)
            v["log_http_status"] = s["http_status"]
            verdicts.append(v)
    elif reqs and all(r.get("recorded_verdict") is not None for r in reqs):
        method = "recorded_verdict"
        verdicts = [{"method": method, "cut": bool(r["recorded_verdict"].get("flagged")),
                     "no_log": bool(r["recorded_verdict"].get("no_log")), "evidence": r["recorded_verdict"].get("evidence") or []}
                    for r in reqs]
    else:
        method = "estimate"
        verdicts = [estimate_verdict(r, window, calib, predictor) for r in reqs]
    tasks = {}
    for r, v in zip(reqs, verdicts):
        t = tasks.setdefault(_task_key(r), {"requests": 0, "cut": 0, "hidden": 0, "ok": None})
        t["requests"] += 1
        status = r.get("http_status", v.get("log_http_status", 200)) or 200
        surfaced = status != 200 or bool(r.get("error"))
        t["cut"] += bool(v["cut"])
        t["hidden"] += bool(v["cut"]) and not surfaced
        if r.get("outcome") is not None and r["outcome"].get("ok") is not None:
            t["ok"] = bool(r["outcome"]["ok"])
    graded = [t for t in tasks.values() if t["ok"] is not None]
    wrong = [t for t in graded if not t["ok"]]
    sessions = {}
    for r in reqs:
        sessions.setdefault(r["session_id"], []).append(r)
    cost = None
    if rates is not None:
        per = [price_session(v, rates) for v in sessions.values()]
        if per and all(p is not None for p in per):
            cost = {"sessions": len(per),
                    "upper_bound_usd": sum(p["upper_bound_usd"] for p in per),
                    "floor_usd": sum(p["floor_usd"] for p in per),
                    "billed_usd": (sum(p["billed_usd"] for p in per) if all(p["billed_usd"] is not None for p in per)
                                   else None),
                    "token_basis": sorted({b for p in per for b in p["token_basis"]})}
    n = len(tasks)
    hidden_tasks = sum(1 for t in tasks.values() if t["hidden"])
    return {"window": window, "method": method, "ground_truth": method != "estimate",
            "requests": len(reqs), "requests_cut": sum(bool(v["cut"]) for v in verdicts),
            "requests_hidden_cut": sum(t["hidden"] for t in tasks.values()),
            "requests_no_log": sum(bool(v.get("no_log")) for v in verdicts),
            "tasks": n, "tasks_hidden_failure": hidden_tasks,
            "hidden_failure_rate": hidden_tasks / n if n else None,
            "tasks_graded": len(graded), "wrong": len(wrong),
            "wrong_with_hidden_cut": sum(1 for t in wrong if t["hidden"]),
            "wrong_without_cut": sum(1 for t in wrong if not t["cut"]),
            "sessions": len(sessions), "cloud_cost": cost, "verdicts": verdicts}


def run_audit(requests: list[dict], prof: dict, logs: dict | None = None, cloud_model: str | None = None,
              calib: float = 1.0, predictor=None) -> dict:
    """logs: {(config, window) or window: server log text}. Requests with ran_at are audited at that window (only
    windows in the profile); requests without ran_at are replayed by estimate against every profile window."""
    from apu_bench import profile as P
    missing = P.validate(prof)
    rates = None
    model = cloud_model or prof.get("cloud_model")
    if model:
        from src.cloud import prices as cp
        idx = cp.pricing_index(REPO)
        if model not in idx:
            raise ValueError(f"cloud model {model!r} not in the committed price tables")
        rates = idx[model]["rates"]
    windows = [w["window"] for w in prof["windows"]]
    for i, r in enumerate(requests):
        r.setdefault("_order", i)
    groups = {}
    for r in requests:
        ra = r.get("ran_at")
        if ra:
            if ra.get("window") in windows:
                groups.setdefault((ra.get("config") or str(ra["window"]), ra["window"]), []).append(r)
        else:
            for w in windows:
                groups.setdefault(("estimate", w), []).append(r)
    logs = logs or {}
    out = []
    for (cfg, w), reqs in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        text = logs.get((cfg, w), logs.get(w)) if cfg != "estimate" else None
        g = audit_group(reqs, w, slice_server_log(text) if text is not None else None, calib, rates, predictor)
        g["config"] = cfg
        out.append(g)
    return {"schema": AUDIT_SCHEMA, "bench_version": BENCH_VERSION, "device_profile": prof,
            "missing_tags": missing, "cloud_model": model, "calib": calib, "groups": out}


def report_md(res: dict) -> str:
    p = res["device_profile"]
    L = [f"# APU audit report: {p['profile_id']}", "",
         f"Runtime {p['runtime']['name']} {p['runtime'].get('version') or ''}, model {p['model']['name']}, cloud "
         f"model {res['cloud_model'] or 'none'}.", "",
         "| config | window | ground truth | tasks | hidden failures | rate | requests cut | wrong | wrong with hidden cut | "
         "wrong without cut | cloud USD floor | cloud USD upper | cloud USD billed |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for g in res["groups"]:
        c = g["cloud_cost"] or {}
        rate = f"{100 * g['hidden_failure_rate']:.1f}%" if g["hidden_failure_rate"] is not None else "n/a"
        gt = g["method"] + ("" if g["ground_truth"] else " (PREDICTED)")

        def usd(k):
            return f"{c[k]:.4f}" if c.get(k) is not None else "n/a"
        L.append(f"| {g['config']} | {g['window']} | {gt} | {g['tasks']} | {g['tasks_hidden_failure']} | {rate} | "
                 f"{g['requests_cut']}/{g['requests']} | {g['wrong']}/{g['tasks_graded']} | {g['wrong_with_hidden_cut']} | "
                 f"{g['wrong_without_cut']} | {usd('floor_usd')} | {usd('upper_bound_usd')} | {usd('billed_usd')} |")
    L += ["", "Hidden failure: the runtime cut the context (its own log line) while the client saw HTTP 200 and no "
          "error. PREDICTED rows come from the chars/4 estimate, not from a log.", ""]
    return "\n".join(L)
