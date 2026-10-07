"""Outcome-table run for evo-x2: for each workload-pack item (trace-weighted priority order, r2_sessions
excluded, 340 items), for each of --models, runs two configurations:
  1. ollama_default   -- stock Ollama, no env override, `"think": false` per request
  2. llama_server     -- llama.cpp llama-server, Vulkan backend, -c sized to this item's own prompt_tokens
                         (+256 headroom), thinking disabled (see THINKING below)
Order is item-major: for a given item, both configs run for model 1, then model 2, ..., and all models for
that item complete before the next item starts. qwen3-32b runs only on a fixed 100-item trace-weighted
subset (qwen32b_subset(): weighted sampling without replacement, seed QWEN32B_SUBSET_SEED, ids recorded in
a "qwen32b_subset" record at the top of --out).

VALIDITY OVERHAUL (2026-10-07). The weekend run (results/x2_outcome_table_weekend.jsonl) was invalid in
three ways, each fixed here:
  * THINKING. qwen3-8b/14b/32b on llama_server spent their whole 256-token budget on hidden reasoning
    (llama-server log: "chat template, thinking = 1", n_gen = 256 on most calls) and scored 0.05 to 0.22.
    llama_server now starts with LLAMA_SERVER_THINKING_ARGS and sends LLAMA_REQUEST_THINKING_FIELDS on
    every request; Ollama sends think=false. Every row records `thinking_setting` (the exact mechanism),
    `reasoning_chars`, `content_chars` and `thinking_leak` (reasoning text present, or a <think> tag in
    content). The mechanisms were verified live by harness/x2_thinking_verify.py (--verify-thinking runs
    it first, once, writing results/x2_thinking_verify.jsonl).
  * INFRA ERRORS SCORED AS MODEL OUTCOMES. Every error is classified by cause (classify_error_cause:
    context_overflow / timeout / connection / other, plus error_subcause), never one flat error rate.
  * NO VALIDITY GATE. Each (model, config) first runs 10 canary items (5 shortest gsm8k + 5 shortest
    function_calling). Canary non-overflow error rate > 10% or mean score < 0.5 halts that (model, config)
    for the rest of the run and writes an ALERT record (surfaced at the top of RESULTS_DIGEST.md by
    analysis/results_digest.py). A rolling non-overflow error rate above 10% (n >= 10) also ALERTs.

RESUME AND CACHE. An (item_id, model, config) key with a VALID row in --out is skipped. A row is valid
unless it carries an invalid_* tag, or its cause is "connection" (an infra failure, always retried).
--reuse-from PATH copies the valid rows of an earlier run into --out once (marked reused_from), after
tagging that file's invalid rows in place (tag_invalid_rows, idempotent).

ITEM-BOUNDARY YIELD CONTRACT (for other queue jobs, e.g. the R2 harness, that need the machine):
  1. The other job queues its own item (status "pending") somewhere after this job's running item.
  2. It creates the file C:\\apu\\ovn\\yield_x2_outcome_table (contents ignored).
  3. Between items (never mid-item, never mid-canary), this runner sees the file, deletes it, inserts a
     copy of its own queue item (status "pending", same cmd, id suffixed _resumeN with N one higher than
     any existing _resume suffix) immediately AFTER the first pending item that follows its own item in
     the queue (directly after its own item if nothing is pending after it), writes the queue, then exits
     cleanly via t2s_queue.advance(note="yielded at item boundary"). advance() marks this entry done and
     launches the next pending entry, i.e. the other job; the resume copy runs after it.
  4. On resume, every valid cached row (including canaries) is reused, so nothing measured is re-paid.
The yield check happens at most once per item, so worst-case latency is one item (all models x both
configs), typically a few minutes, up to ~30 min on an 80K-token item.

QUEUE EXIT. When launched by the queue (APU_QUEUE_JOB_ID set), main() always ends with t2s_queue.advance():
"completed ..." / "deadline reached ..." / "yielded at item boundary" / "stopped: <exception>".

Usage:
  py -3.12 x2_outcome_table.py --out results/x2_outcome_table_v3.jsonl \\
      [--reuse-from results/x2_outcome_table_weekend.jsonl] [--verify-thinking] [--smoke-n 10] \\
      [--deadline-h 120] [--call-timeout-s 900]
  py -3.12 x2_outcome_table.py --out results/x2_outcome_table_weekend.jsonl --tag-invalid
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

_here = Path(__file__).resolve().parent
if (_here / "analysis").is_dir():
    REPO = _here  # flat deployment on evo-x2
else:
    REPO = _here.parents[0]  # repo layout
sys.path.insert(0, str(_here))
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO))

import host_config as hc  # noqa: E402
import t2s_k1_ollama as k1  # noqa: E402

# model_id -> (ollama tag, local gguf path for the llama-server leg)
MODEL_MAP = {
    "llama3.1:8b": ("llama3.1:8b", r"C:\apu\models\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"),
    "qwen3-8b": ("qwen3:8b", r"C:\apu\models\Qwen3-8B-Q4_K_M.gguf"),
    "qwen3-4b-2507": ("qwen3-4b-2507", r"C:\apu\models\qwen3-4b-instruct-85e4a5b7.gguf"),
    "qwen3-14b": ("qwen3:14b", r"C:\apu\models\Qwen3-14B-Q4_K_M.gguf"),
    # 2026-10-02: "qwen3:30b-a3b-instruct-2507" does not exist on the Ollama registry (404), so the weekend run
    # used plain "qwen3:30b-a3b". 2026-10-07, found by the canary gate: that registry tag is NOT the same model
    # as the llama_server leg's Instruct-2507 GGUF. Its template unconditionally opens "<think>\n" after the
    # assistant turn (read from the manifest's template blob on evo-x2), so "think": false cannot turn it off
    # and its reasoning lands in content (10/10 canaries: finish=length, content = "Okay, let's see...",
    # mean score 0.0). The Ollama leg now uses a local model created from the SAME GGUF as the llama_server
    # leg (OLLAMA_CREATE_FROM_GGUF, the same route qwen3-4b-2507 already uses), so both configs measure
    # identical weights.
    "qwen3-30b-a3b": ("qwen3-30b-a3b-2507", r"C:\apu\models\Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"),
    "qwen3-32b": ("qwen3:32b", r"C:\apu\models\Qwen3-32B-Q4_K_M.gguf"),
}
# Ollama tags this harness creates itself (ollama create from the model's own llama_server GGUF) if missing.
OLLAMA_CREATE_FROM_GGUF = {"qwen3-30b-a3b-2507": r"C:\apu\models\Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"}
# Ollama tag a row was measured with, for rows written before rows recorded it (all weekend/early-v3 rows).
LEGACY_OLLAMA_TAG = {"qwen3-30b-a3b": "qwen3:30b-a3b"}
LLAMA_SERVER_EXE = r"C:\apu\bin\llama-b10970\llama-server.exe"
LLAMA_SERVER_PORT = 58299
DEFAULT_MODELS = ["llama3.1:8b", "qwen3-4b-2507", "qwen3-8b", "qwen3-14b", "qwen3-30b-a3b", "qwen3-32b"]
CONFIGS = ("ollama_default", "llama_server")
MAX_TOKENS = 256

# ------------------------------------------------------------------------------------------------ THINKING
# Server flag + per-request kwarg, both applied (belt and braces): harness/x2_thinking_verify.py records which
# one(s) suppress reasoning on their own on this exact build (b10970); applying both costs nothing for models
# with no reasoning mode and means one mechanism silently regressing in a future build cannot re-contaminate.
LLAMA_SERVER_THINKING_ARGS = ["--reasoning-budget", "0"]
LLAMA_SERVER_REASONING_BUDGET_ARGS = LLAMA_SERVER_THINKING_ARGS  # back-compat name
LLAMA_REQUEST_THINKING_FIELDS = {"chat_template_kwargs": {"enable_thinking": False}}
THINKING_SETTING_LLAMA = ("llama_server: server flag --reasoning-budget 0 + per-request "
                          "chat_template_kwargs.enable_thinking=false")
THINKING_SETTING_OLLAMA = "ollama: per-request \"think\": false"
_THINK_TAG_RE = re.compile(r"<think>.*?(</think>|$)", re.S)

# models that default to hybrid thinking on this build (llama3.1:8b has no reasoning mode; qwen3-4b-2507 and
# qwen3-30b-a3b are -Instruct-2507 releases, non-thinking by default, confirmed by their weekend n_gen ~10)
THINKING_BY_DEFAULT_MODELS = {"qwen3-8b", "qwen3-14b", "qwen3-32b"}

# ------------------------------------------------------------------------------------------------ GATES
CANARY_ERROR_RATE_THRESHOLD = 0.10
CANARY_MIN_MEAN_SCORE = 0.5
ROLLING_NON_OVERFLOW_ERROR_RATE_THRESHOLD = 0.10
ROLLING_MIN_N = 10

QWEN32B_SUBSET_N = 100
QWEN32B_SUBSET_SEED = 20261007

YIELD_FLAG = Path(r"C:\apu\ovn\yield_x2_outcome_table")
LLAMA_LOG_DIR = Path(r"C:\apu\ovn\results")


# ================================================================================================ classification
ERROR_CAUSES = ("context_overflow", "timeout", "connection", "other")


def classify_error_cause(row):
    """One of: none (HTTP 200, scored normally), context_overflow (HTTP 400 exceeds-context, expected for an
    item larger than the config can hold), timeout, connection (server unreachable / never came up), other.
    Classified from the row's raw fields only; invalid_* tags are a separate validity axis (row_is_valid),
    so a tagged race row still reports its real cause, connection. Never a single flat 'error' bucket: the
    weekend run's flat error rate hid a server-start failure underneath it."""
    status = row.get("http_status")
    if status == 200:
        return "none"
    error = (row.get("error") or "").lower()
    if status == 400 and ("context" in error or "exceed" in error):
        return "context_overflow"
    if row.get("chat_outcome") == "infra_not_ready" or any(m in error for m in (
            "actively refused", "10061", "connection refused", "did not open its port", "never left 'loading'",
            "server did not become ready", "connection reset", "remote end closed", "10054", "forcibly closed")):
        return "connection"
    if "timed out" in error or "timeout" in error:
        return "timeout"
    return "other"


def error_subcause(row):
    """Finer label inside a cause, for the report (e.g. other/oom vs other/model_not_found)."""
    cause = classify_error_cause(row)
    if cause == "none":
        return None
    error = (row.get("error") or "").lower()
    if cause == "other":
        if any(m in error for m in ("out of memory", "out-of-memory", "cudamalloc failed", "rocm error",
                                    "failed to allocate", "alloc_buffer: failed", "bad_alloc")):
            return "oom"
        if "not found" in error:
            return "model_not_found"
        if "guard" in error or "port" in error and "listener" in error:
            return "stale_server_guard"
        if "terminated" in error or "exit status" in error:
            return "runner_terminated"
        return f"http_{row.get('http_status')}"
    if cause == "connection":
        if row.get("chat_outcome") == "infra_not_ready":
            return "ollama_not_ready"
        if "did not open its port" in error or "never left" in error:
            return "llama_server_not_ready"
        if "refused" in error or "10061" in error:
            return "refused"
        return "dropped"
    return cause


def invalid_tags(row):
    return sorted(k for k, v in row.items() if k.startswith("invalid_") and v is True)


def row_is_valid(row):
    """A row is reusable as a cached result unless it carries an invalid_* tag or is an infra failure
    (cause connection). Context overflow, timeouts and other errors are real measured outcomes."""
    if row.get("family") == "gsm8k" and row.get("http_status") == 200 and row.get("scorer_version", 1) < 2:
        return False  # scored by the pre-fix final_number_match path (always 0); rerun
    model = row.get("model_id")
    if row.get("config") == "ollama_default" and model in MODEL_MAP:
        measured_tag = row.get("ollama_tag") or LEGACY_OLLAMA_TAG.get(model, MODEL_MAP[model][0])
        if measured_tag != MODEL_MAP[model][0]:
            return False  # measured a different Ollama model than the current mapping
    return not invalid_tags(row) and classify_error_cause(row) != "connection"


# ================================================================================================ canaries
def canary_items(items):
    """5 shortest gsm8k + 5 shortest function_calling items, by prompt_tokens, from the already-loaded,
    trace-weighted item list (so this never re-reads the pack separately or risks a different ordering)."""
    gsm8k = sorted((it for it in items if it["family"] == "gsm8k"), key=lambda it: it["prompt_tokens"])[:5]
    fcall = sorted((it for it in items if it["family"] == "function_calling"), key=lambda it: it["prompt_tokens"])[:5]
    return gsm8k + fcall


def canary_gate_check(canary_rows):
    """Returns (passed, error_rate, mean_score). A canary error is any cause other than none/context_overflow
    (an overflow on a short canary would be a canary-selection bug, not an infra problem), or a thinking leak.
    Mean score is over all canary rows."""
    if not canary_rows:
        return True, 0.0, 1.0
    errors = sum(1 for r in canary_rows
                 if classify_error_cause(r) not in ("none", "context_overflow") or r.get("thinking_leak"))
    scores = [r.get("score") or 0.0 for r in canary_rows]
    error_rate = errors / len(canary_rows)
    mean_score = sum(scores) / len(scores)
    passed = error_rate <= CANARY_ERROR_RATE_THRESHOLD and mean_score >= CANARY_MIN_MEAN_SCORE
    return passed, error_rate, mean_score


class RollingErrorMonitor:
    """Rolling non-overflow error rate per (model, config). Alerts once when the rate crosses above the
    threshold (n >= ROLLING_MIN_N), re-arms when it drops back to or below it, so a sustained problem yields
    one ALERT, not one per item."""

    def __init__(self, threshold=ROLLING_NON_OVERFLOW_ERROR_RATE_THRESHOLD, min_n=ROLLING_MIN_N):
        self.threshold, self.min_n = threshold, min_n
        self.state = {}

    def add(self, key, row):
        st = self.state.setdefault(key, {"n": 0, "errors": 0, "alerting": False})
        st["n"] += 1
        if classify_error_cause(row) not in ("none", "context_overflow"):
            st["errors"] += 1
        if st["n"] < self.min_n:
            return None
        rate = st["errors"] / st["n"]
        if rate > self.threshold and not st["alerting"]:
            st["alerting"] = True
            return {"rate": rate, "n": st["n"], "errors": st["errors"]}
        if rate <= self.threshold:
            st["alerting"] = False
        return None


# ================================================================================================ items
def utc_iso():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load_weights(repo):
    weights_path = repo / "results" / "workload_pack" / "item_weights_trace_weighted.json"
    if not weights_path.exists():
        raise FileNotFoundError(str(weights_path))
    return json.loads(weights_path.read_text(encoding="utf-8"))


def load_items_trace_weighted(repo):
    weight_by_id = load_weights(repo)
    sys.path.insert(0, str(repo))
    from analysis import trace_weighted_pack as twp
    items = twp.load_pack_items(repo)
    pairs = [(it, weight_by_id.get(it["item_id"], 0.0)) for it in items if it["family"] != "r2_sessions"]
    pairs.sort(key=lambda p: p[1], reverse=True)
    return [p[0] for p in pairs]


def qwen32b_subset(items, weight_by_id, n=QWEN32B_SUBSET_N, seed=QWEN32B_SUBSET_SEED):
    """Weighted sampling without replacement (Efraimidis-Spirakis: key = u ** (1 / w), take the n largest),
    weights from analysis/trace_weighted_pack.py's item_weights_trace_weighted.json, fixed seed. Items with
    weight 0 are never drawn. Returns a set of item ids; deterministic for a given (items, weights, n, seed)."""
    rng = random.Random(seed)
    keyed = []
    for it in sorted(items, key=lambda it: it["item_id"]):  # stable order before drawing
        w = weight_by_id.get(it["item_id"], 0.0)
        u = rng.random()
        if w > 0:
            keyed.append((u ** (1.0 / w), it["item_id"]))
    keyed.sort(reverse=True)
    return {iid for _, iid in keyed[:n]}


def load_graders():
    sys.path.insert(0, str(REPO / "results" / "workload_pack"))
    import grade as g
    return g


def strip_thinking(text):
    return _THINK_TAG_RE.sub("", text or "").strip()


def score_response(grade_module, item, response_text):
    method = item["grading"]["method"]
    oracle = item["oracle_answer"]
    fn = grade_module.GRADERS[method]
    try:
        if method == "exact_substring":
            return fn(oracle, response_text, case_sensitive=item["grading"].get("case_sensitive", True))
        if method in ("exact_dict_match", "session_rule_recall"):
            m = re.search(r"\{.*\}", response_text, re.S)
            parsed = json.loads(m.group(0)) if m else {}
            return fn(oracle, parsed)
        if method == "final_number_match":
            # 2026-10-07 bug found live (x2_thinking_verify): grade.py's grader compares the oracle against a bare
            # number, but this used to pass the whole response, so every gsm8k answer scored 0 (e.g. a correct
            # "...\n#### 3" vs oracle "3"). Extract the number after the LAST "####", the exact answer format the
            # prompt asks for and the pack builder's own FINAL_RE (results/workload_pack/scripts/build_c_gsm8k.py).
            found = _FINAL_NUMBER_RE.findall(response_text or "")
            return fn(oracle, found[-1]) if found else 0.0
    except Exception:
        return 0.0
    return 0.0


_FINAL_NUMBER_RE = re.compile(r"####\s*\**\s*(-?[\d,]+(?:\.\d+)?)")
# Rows scored before the final_number_match fix carry no scorer_version; they are not valid cached rows.
SCORER_VERSION = 2


# ================================================================================================ results file
def read_rows(path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def valid_cached_rows(out_path):
    """(item_id, model_id, config) -> the LAST valid outcome row for that key in out_path."""
    cached = {}
    for r in read_rows(out_path):
        if r.get("record") == "outcome_row" and row_is_valid(r):
            cached[(r["item_id"], r["model_id"], r["config"])] = r
    return cached


def already_done_keys(out_path):
    return set(valid_cached_rows(out_path))


def emit(out_path, row):
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


# ================================================================================================ invalid tagging
_CONNECTION_REFUSED_MARKERS = ("actively refused", "winerror 10061", "connection refused")

# Verified cause per time window for the weekend run's connection-refused ollama_default rows (2026-10-07
# re-examination; evidence in docs/FINDINGS.md). Process lineage from C:\apu\ovn\watchdog.log: the watchdog
# (SYSTEM) requeued x2_outcome_table_v2 and launched x2_k2_v2 at 2026-10-04T20:12:05Z; x2_k2_v2's own
# advance() then relaunched x2_outcome_table_v2 (pid 15964) under SYSTEM. SYSTEM's PATH has no ollama.exe and
# its LOCALAPPDATA is the system profile, so start_ollama_server launched a bare "ollama" that never ran:
# 405/405 ollama calls refused, contiguous, each after a full 60s readiness wait. A stop/start race would
# leave intermittent successes (the next call's start sees no process and launches fresh); there were none.
# The next process (pid 21552, launched from an SSH user session at 2026-10-07T00:31Z) had 0 refused.
RACE_ATTRIBUTION_WINDOWS = [
    ("2026-10-05T00:00:00", "2026-10-07T00:31:00", "system_ollama_exe_resolution",
     "pid 15964 ran as SYSTEM (watchdog -> x2_k2_v2 -> advance()); bare 'ollama' not on SYSTEM PATH; "
     "server never started; 0 successes in the contiguous block"),
    ("2026-10-02T07:40:00", "2026-10-02T08:05:00", "concurrent_job_contention_unverified",
     "x2_model_pulls was launched 2026-10-02T07:43:39Z while this job (pid 7468) was still running; both "
     "start/stop Ollama; not verified beyond timing"),
]


def race_cause_for(ts):
    for start, end, cause, evidence in RACE_ATTRIBUTION_WINDOWS:
        if ts and start <= ts[:19] <= end:
            return cause, evidence
    return "unattributed", "no lineage evidence for this window"


def tag_invalid_rows(path):
    """Post-processing only, no re-run, additive and idempotent. Tags every outcome row of an earlier run:
      invalid_race        -- ollama_default row whose error matches connection refused. The operator's tag
                             name is kept; the verified cause is in invalid_race_cause / _evidence.
      invalid_thinking    -- llama_server row for a THINKING_BY_DEFAULT_MODELS model with no thinking_setting
                             (measured with the server's default thinking-on chat template).
      invalid_infra_oom   -- ollama_default row whose runner died out of memory. Clustered in time across all
                             model sizes (4B included) while the same item lengths succeeded at other times:
                             treated as machine-state contamination and rerun, not reused as an outcome.
    Returns (n_race, n_thinking, n_oom)."""
    raw = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                raw.append(json.loads(line))
            except Exception:
                raw.append(line)
    n_race = n_thinking = n_oom = 0
    for r in raw:
        if not isinstance(r, dict) or r.get("record") != "outcome_row":
            continue
        error_text = (r.get("error") or "").lower()
        if r.get("config") == "ollama_default" and any(m in error_text for m in _CONNECTION_REFUSED_MARKERS):
            r["invalid_race"] = True
            r["invalid_race_cause"], r["invalid_race_evidence"] = race_cause_for(r.get("ts_utc"))
            n_race += 1
        if (r.get("config") == "llama_server" and r.get("model_id") in THINKING_BY_DEFAULT_MODELS
                and not r.get("thinking_setting") and not r.get("thinking_disabled")):
            r["invalid_thinking"] = True
            r["invalid_thinking_cause"] = ("llama-server default chat template with thinking on (log: 'chat template, "
                                           "thinking = 1'); budget spent on reasoning_content")
            n_thinking += 1
        if r.get("config") == "ollama_default" and error_subcause(r) == "oom":
            r["invalid_infra_oom"] = True
            n_oom += 1
    with open(path, "w", encoding="utf-8") as f:
        for r in raw:
            f.write((json.dumps(r, default=str) if isinstance(r, dict) else r) + "\n")
    return n_race, n_thinking, n_oom


def seed_from_previous_run(out_path, reuse_from):
    """Once per --out: tag reuse_from's invalid rows in place, then copy its valid outcome rows into out_path
    (reused_from set, thinking_setting filled for the record). Idempotent via a 'reuse_seed' record."""
    if any(r.get("record") == "reuse_seed" for r in read_rows(out_path)):
        return None
    counts = tag_invalid_rows(reuse_from)
    n_copied = 0
    for r in read_rows(reuse_from):
        if r.get("record") != "outcome_row" or not row_is_valid(r):
            continue
        r = dict(r)
        r["reused_from"] = reuse_from.name
        if "thinking_setting" not in r:
            r["thinking_setting"] = (THINKING_SETTING_OLLAMA if r["config"] == "ollama_default" else
                                     "llama_server: none applied (weekend row; model non-thinking by default)")
        emit(out_path, r)
        n_copied += 1
    rec = {"record": "reuse_seed", "ts_utc": utc_iso(), "reuse_from": str(reuse_from), "n_copied": n_copied,
           "tagged": {"invalid_race": counts[0], "invalid_thinking": counts[1], "invalid_infra_oom": counts[2]}}
    emit(out_path, rec)
    return rec


def ollama_manifest_path(tag, models_dir):
    name, _, version = tag.partition(":")
    return Path(models_dir) / "manifests" / "registry.ollama.ai" / "library" / name / (version or "latest")


def ensure_ollama_custom_models(models, out_path, run_fn=None, models_dir=None):
    """Creates any OLLAMA_CREATE_FROM_GGUF tag used by `models` that is not yet in the host's Ollama store
    (`ollama create <tag> -f Modelfile` with "FROM <gguf>", the same route qwen3-4b-2507 was made by). Runs inside
    the queue job, before any measurement, so it never overlaps one. Records an "ollama_create" row either way."""
    import tempfile
    models_dir = models_dir or hc._this_host_entry().get("ollama_models")
    for model_key in models:
        tag = MODEL_MAP[model_key][0]
        gguf = OLLAMA_CREATE_FROM_GGUF.get(tag)
        if not gguf:
            continue
        if models_dir and ollama_manifest_path(tag, models_dir).exists():
            continue
        rec = {"record": "ollama_create", "ts_utc": utc_iso(), "tag": tag, "gguf": gguf}
        hc.start_ollama_server()
        try:
            if not hc.wait_for_ollama_ready(timeout_s=90):
                rec.update({"outcome": "error", "error": "ollama server not ready"})
            else:
                exe = hc._resolve_ollama_exe_for_serve()
                with tempfile.TemporaryDirectory() as td:
                    mf = Path(td) / "Modelfile"
                    mf.write_text(f"FROM {gguf}\n", encoding="utf-8")
                    run = run_fn or (lambda argv: subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                                                                 errors="replace", timeout=3600))
                    t0 = time.monotonic()
                    res = run([exe, "create", tag, "-f", str(mf)])
                rec.update({"outcome": "ok" if res.returncode == 0 else "error", "returncode": res.returncode,
                            "elapsed_s": time.monotonic() - t0, "stdout_tail": (res.stdout or "")[-500:],
                            "stderr_tail": (res.stderr or "")[-500:]})
        except Exception as e:
            rec.update({"outcome": "error", "error": repr(e)[:400]})
        finally:
            hc.stop_ollama_server()
        emit(out_path, rec)
        print(f"ollama create {tag}: {rec.get('outcome')}", flush=True)


# ================================================================================================ runtimes
def run_one_ollama(item, model_key, ollama_tag, out_path, call_timeout_s, canary=False):
    hc.start_ollama_server()
    row = {"record": "outcome_row", "item_id": item["item_id"], "family": item["family"],
           "config": "ollama_default", "model_id": model_key, "ts_utc": utc_iso(),
           "thinking_setting": THINKING_SETTING_OLLAMA, "canary": canary, "scorer_version": SCORER_VERSION,
           "ollama_tag": ollama_tag}
    try:
        # 2026-10-06: a server that never came up must be recorded as infra_not_ready, not scored as a model
        # outcome. One forced restart-and-rewait before giving up.
        ready = hc.wait_for_ollama_ready(timeout_s=60)
        if not ready:
            hc.stop_ollama_server()
            hc.start_ollama_server()
            ready = hc.wait_for_ollama_ready(timeout_s=60)
        if not ready:
            row.update({"http_status": None, "score": 0.0, "chat_outcome": "infra_not_ready",
                        "error": "ollama server did not become ready within 60s, even after one forced restart"})
            emit(out_path, row)
            return row
        ollama = k1.OllamaClient()
        t0 = time.monotonic()
        resp = ollama.chat(ollama_tag, "", num_ctx=None, messages=[{"role": "user", "content": item["prompt"]}],
                           max_tokens=MAX_TOKENS, think=False, keep_alive="2m", timeout=call_timeout_s)
        dt = time.monotonic() - t0
        content = resp.get("message") or ""
        raw_msg = ((resp.get("raw") or {}).get("message") or {})
        reasoning = raw_msg.get("thinking") or ""
        leak = bool(reasoning) or "<think>" in content
        score = score_response(load_graders(), item, strip_thinking(content))
        sent = item["prompt_tokens"]
        processed = resp.get("prompt_eval_count")
        row.update({"http_status": resp.get("status"), "score": score, "latency_s": dt,
                    "sent_tokens": sent, "processed_tokens": processed,
                    "silently_truncated": bool(processed is not None and processed < sent),
                    "chat_outcome": resp.get("outcome"), "error": resp.get("error"),
                    "completion_tokens": resp.get("eval_count"),
                    "finish_reason": (resp.get("raw") or {}).get("done_reason"),
                    "content_chars": len(content), "reasoning_chars": len(reasoning), "thinking_leak": leak,
                    "output_text": content[:500]})
    except Exception as e:
        row.update({"http_status": None, "score": 0.0, "error": f"driver exception: {e!r}"[:400]})
    finally:
        hc.stop_ollama_server()
    emit(out_path, row)
    return row


def safe_name(s):
    """Filesystem-safe: 'llama3.1:8b' in a Windows path wrote into an NTFS alternate data stream (weekend bug:
    every llama3.1:8b llama-server log went to the stream 'x2_outcome_table_llamaserver_llama3.1')."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)


class LlamaServer:
    """Start/verify/stop one llama-server. Uses harness/server_guard.py when importable (port free before
    start, /props model path + n_ctx + slot count after start, listener pid before the request)."""

    def __init__(self, gguf_path, n_ctx, log_path, extra_args=(), port=LLAMA_SERVER_PORT):
        self.gguf_path, self.n_ctx, self.log_path, self.port = gguf_path, n_ctx, log_path, port
        self.cmd = ([LLAMA_SERVER_EXE, "-m", gguf_path, "--port", str(port), "--host", "127.0.0.1", "--no-webui",
                     "-c", str(n_ctx), "-np", "1", "-t", "8", "--log-verbosity", "4", "-ngl", "99"] + list(extra_args))
        self.proc = None
        self.guard_record = {}
        self.url = f"http://127.0.0.1:{port}"

    def _guard(self):
        try:
            import server_guard
            return server_guard
        except Exception:
            return None

    def start(self, call_timeout_s=900):
        """Returns None on success, else an error string (row error text)."""
        import socket
        sg = self._guard()
        if sg is not None:
            deadline = time.monotonic() + 30
            while True:
                try:
                    sg.assert_port_free(self.port)
                    break
                except sg.GuardError as e:
                    if time.monotonic() > deadline:
                        self.guard_record = e.record
                        return f"stale server guard: {e}"[:400]
                    time.sleep(2)
        with open(self.log_path, "w", encoding="utf-8") as logfh:
            self.proc = subprocess.Popen(self.cmd, stdout=logfh, stderr=logfh)
        t0 = time.monotonic()
        ready = False
        while time.monotonic() - t0 < 180:
            if self.proc.poll() is not None:
                return f"server did not open its port in time (process exited rc={self.proc.returncode})"
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=1):
                    ready = True
                    break
            except OSError:
                time.sleep(1)
        if not ready:
            return "server did not open its port in time"
        t1 = time.monotonic()
        while time.monotonic() - t1 < call_timeout_s:
            try:
                with urllib.request.urlopen(f"{self.url}/health", timeout=10) as r:
                    if r.status == 200:
                        break
            except urllib.error.HTTPError as e:
                if e.code != 503:
                    break
            except Exception:
                pass
            time.sleep(2)
        else:
            return "server never left 'loading' state"
        if sg is not None:
            try:
                self.guard_record = sg.assert_server_matches(self.url, self.port, self.proc.pid,
                                                             {"model_path": self.gguf_path, "n_ctx": self.n_ctx, "np": 1})
            except sg.GuardError as e:
                self.guard_record = e.record
                return f"server guard mismatch: {e}"[:400]
        return None

    def request_guard_ok(self):
        sg = self._guard()
        if sg is None or self.proc is None:
            return True, None
        ok, lp = sg.RequestGuard(self.port, self.proc.pid).check()
        return ok, lp

    def chat(self, body, timeout):
        req = urllib.request.Request(f"{self.url}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    def stop(self):
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()
                try:
                    self.proc.wait(timeout=10)
                except Exception:
                    pass


def llama_message_fields(data):
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    return content, reasoning, choice.get("finish_reason"), data.get("usage") or {}


def run_one_llama_server(item, model_key, gguf_path, out_path, call_timeout_s, canary=False):
    n_ctx = ((item["prompt_tokens"] + MAX_TOKENS + 255) // 256) * 256
    log_path = str(LLAMA_LOG_DIR / f"x2_outcome_table_llamaserver_{safe_name(model_key)}_{safe_name(item['item_id'])}.log")
    srv = LlamaServer(gguf_path, n_ctx, log_path, extra_args=LLAMA_SERVER_THINKING_ARGS)
    row = {"record": "outcome_row", "item_id": item["item_id"], "family": item["family"],
           "config": "llama_server", "model_id": model_key, "ts_utc": utc_iso(), "requested_n_ctx": n_ctx,
           "thinking_setting": THINKING_SETTING_LLAMA, "canary": canary, "scorer_version": SCORER_VERSION}
    try:
        err = srv.start(call_timeout_s=call_timeout_s)
        if srv.guard_record:
            row["server_guard"] = {k: srv.guard_record.get(k) for k in ("guard", "props_model_path", "props_n_ctx",
                                                                         "props_build_info", "problems", "listeners")
                                   if k in srv.guard_record}
        if err:
            row.update({"http_status": None, "score": 0.0, "error": err})
            emit(out_path, row)
            return row
        ok, lp = srv.request_guard_ok()
        if not ok:
            row.update({"http_status": None, "score": 0.0, "invalid_stale_server": True,
                        "error": f"request guard: listener pids {lp} != started pid {srv.proc.pid}"})
            emit(out_path, row)
            return row
        body = {"model": "x", "messages": [{"role": "user", "content": item["prompt"]}],
                "max_tokens": MAX_TOKENS, "temperature": 0}
        body.update(LLAMA_REQUEST_THINKING_FIELDS)
        t2 = time.monotonic()
        try:
            data = srv.chat(body, timeout=call_timeout_s)
            dt = time.monotonic() - t2
            content, reasoning, finish, usage = llama_message_fields(data)
            leak = bool(reasoning) or "<think>" in content
            score = score_response(load_graders(), item, strip_thinking(content))
            sent = item["prompt_tokens"]
            processed = usage.get("prompt_tokens")
            row.update({"http_status": 200, "score": score, "latency_s": dt, "sent_tokens": sent,
                        "processed_tokens": processed,
                        "silently_truncated": bool(processed is not None and processed < sent),
                        "completion_tokens": usage.get("completion_tokens"), "finish_reason": finish,
                        "content_chars": len(content), "reasoning_chars": len(reasoning), "thinking_leak": leak,
                        "output_text": content[:500]})
        except urllib.error.HTTPError as e:
            row.update({"http_status": e.code, "score": 0.0, "error": e.read().decode(errors="replace")[:400]})
        except Exception as e:
            row.update({"http_status": None, "score": 0.0, "error": str(e)[:400]})
    finally:
        srv.stop()
    emit(out_path, row)
    return row


# ================================================================================================ yield contract
def _queue_module():
    import t2s_queue
    return t2s_queue


def own_job_id():
    return os.environ.get("APU_QUEUE_JOB_ID")


def _base_id(job_id):
    return re.sub(r"_resume\d+$", "", job_id)


def build_yield_queue(items, own_id):
    """Pure function (unit-tested): returns (new_items, resume_entry). Inserts a pending copy of own_id's entry
    (same cmd, id <base>_resumeN) immediately after the first pending entry that follows own_id; directly after
    own_id if none follows. Never modifies or reorders any other entry."""
    items = [dict(it) for it in items]
    idx = next(i for i, it in enumerate(items) if it["id"] == own_id)
    own = items[idx]
    base = _base_id(own_id)
    nums = [int(m.group(1)) for it in items for m in [re.match(re.escape(base) + r"_resume(\d+)$", it["id"])] if m]
    resume_id = f"{base}_resume{max(nums, default=0) + 1}"
    resume = {"id": resume_id, "cmd": list(own["cmd"]), "status": "pending",
              "note": f"resume copy inserted by {own_id} at an item boundary (yield contract, x2_outcome_table.py "
                      f"docstring); reuses every valid cached row in the same --out"}
    insert_at = idx + 1
    for j in range(idx + 1, len(items)):
        if items[j].get("status") == "pending":
            insert_at = j + 1
            break
    items.insert(insert_at, resume)
    return items, resume


def check_and_yield(flag_path=None, queue=None, job_id=None):
    """Item boundary hook. Returns the resume entry if this run yielded (caller must then stop and advance),
    else None. No-op outside a queue launch (no APU_QUEUE_JOB_ID): the flag is left in place for the real job."""
    flag_path = Path(flag_path) if flag_path is not None else YIELD_FLAG
    if not flag_path.exists():
        return None
    job_id = job_id or own_job_id()
    if not job_id:
        return None
    tq = queue or _queue_module()
    items = tq.read_queue()
    if not any(it["id"] == job_id for it in items):
        return None
    try:
        flag_path.unlink()
    except FileNotFoundError:
        pass
    new_items, resume = build_yield_queue(items, job_id)
    tq.write_queue(new_items)
    return resume


# ================================================================================================ main loop
class Yielded(Exception):
    pass


def run(out_path, models, smoke_n=None, deadline_h=24.0, call_timeout_s=900, reuse_from=None,
        verify_thinking=False, runners=None, yield_hook=None, items=None, weights=None):
    """runners / yield_hook / items / weights are injection points for tests. Returns a status string."""
    runners = runners or {"ollama_default": run_one_ollama, "llama_server": run_one_llama_server}
    yield_hook = yield_hook or check_and_yield
    if items is None:
        items = load_items_trace_weighted(REPO)
    if weights is None:
        weights = load_weights(REPO)
    if smoke_n:
        items = items[:smoke_n]
    if reuse_from:
        rec = seed_from_previous_run(out_path, Path(reuse_from))
        if rec:
            print(f"seeded from {reuse_from}: {rec['n_copied']} valid rows copied, tagged {rec['tagged']}")
    if verify_thinking:
        import x2_thinking_verify as tv
        tv.ensure_verified(out_path.parent / "x2_thinking_verify.jsonl", items, models)

    if runners.get("ollama_default") is run_one_ollama:  # real runtimes only (tests inject fakes)
        ensure_ollama_custom_models(models, out_path)

    subset = qwen32b_subset(items, weights, n=QWEN32B_SUBSET_N) if "qwen3-32b" in models else set()
    if subset and not any(r.get("record") == "qwen32b_subset" for r in read_rows(out_path)):
        emit(out_path, {"record": "qwen32b_subset", "ts_utc": utc_iso(), "seed": QWEN32B_SUBSET_SEED,
                        "n": len(subset), "item_ids": sorted(subset)})

    canaries = canary_items(items)
    cached = valid_cached_rows(out_path)
    t_start = time.monotonic()
    deadline_s = deadline_h * 3600
    gated_off = set()
    gate_checked = set()
    monitor = RollingErrorMonitor()

    def call(config, item, model_key, canary=False):
        ollama_tag, gguf_path = MODEL_MAP[model_key]
        target = ollama_tag if config == "ollama_default" else gguf_path
        row = runners[config](item, model_key, target, out_path, call_timeout_s, canary=canary)
        cached_key = (item["item_id"], model_key, config)
        if row_is_valid(row):
            cached[cached_key] = row
        return row

    def run_canary_gate(model_key, config):
        rows = []
        for c in canaries:
            key = (c["item_id"], model_key, config)
            rows.append(cached[key] if key in cached else call(config, c, model_key, canary=True))
        passed, error_rate, mean_score = canary_gate_check(rows)
        causes = {}
        for r in rows:
            causes[classify_error_cause(r)] = causes.get(classify_error_cause(r), 0) + 1
        emit(out_path, {"record": "canary_gate", "model_id": model_key, "config": config, "ts_utc": utc_iso(),
                        "passed": passed, "error_rate": error_rate, "mean_score": mean_score, "n": len(rows),
                        "causes": causes, "thinking_leaks": sum(1 for r in rows if r.get("thinking_leak")),
                        "item_ids": [c["item_id"] for c in canaries]})
        if not passed:
            print(f"ALERT: canary gate FAILED for {model_key}/{config}: error_rate={error_rate:.1%} "
                  f"mean_score={mean_score:.3f} -- halting this (model, config) for the rest of the run")
            emit(out_path, {"record": "alert", "source": "x2_outcome_table", "model_id": model_key, "config": config,
                            "ts_utc": utc_iso(), "reason": "canary_gate_failed", "error_rate": error_rate,
                            "mean_score": mean_score})
        return passed

    n_items_done = 0
    for item in items:
        if time.monotonic() - t_start > deadline_s:
            print(f"deadline reached ({deadline_h}h), stopping at {n_items_done} items done this run")
            return f"deadline reached ({deadline_h}h) after {n_items_done} items this run"
        resume = yield_hook()
        if resume is not None:
            emit(out_path, {"record": "yield", "ts_utc": utc_iso(), "resume_id": resume["id"],
                            "before_item": item["item_id"], "items_done_this_run": n_items_done})
            print(f"yield flag seen: inserted {resume['id']}, exiting at item boundary")
            return "yielded at item boundary"
        emit(out_path, {"record": "heartbeat", "item_id": item["item_id"], "ts_utc": utc_iso()})
        for model_key in models:
            if model_key == "qwen3-32b" and item["item_id"] not in subset:
                continue
            for config in CONFIGS:
                if (model_key, config) in gated_off:
                    continue
                if (model_key, config) not in gate_checked:
                    gate_checked.add((model_key, config))
                    if not run_canary_gate(model_key, config):
                        gated_off.add((model_key, config))
                        continue
                key = (item["item_id"], model_key, config)
                if key in cached:
                    continue
                t0 = time.monotonic()
                row = call(config, item, model_key)
                dt = time.monotonic() - t0
                cause = classify_error_cause(row)
                alert = monitor.add((model_key, config), row)
                if alert:
                    print(f"ALERT: rolling non-overflow error rate for {model_key}/{config} is {alert['rate']:.1%} "
                          f"(n={alert['n']})")
                    emit(out_path, {"record": "alert", "source": "x2_outcome_table", "model_id": model_key,
                                    "config": config, "ts_utc": utc_iso(), "reason": "rolling_error_rate", **alert})
                print(f"{item['item_id']} {model_key} {config}: {dt:.1f}s http={row.get('http_status')} "
                      f"score={row.get('score')} cause={cause}", flush=True)
        n_items_done += 1
    print(f"done: {n_items_done} items this run, output {out_path}")
    return f"completed: {n_items_done} items this run, gated off: {sorted(gated_off)}"


def _advance(note):
    if not own_job_id():
        print(f"(not launched by the queue; would advance with note: {note})")
        return
    _queue_module().advance(note)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--smoke-n", type=int, default=None)
    ap.add_argument("--deadline-h", type=float, default=24.0)
    ap.add_argument("--call-timeout-s", type=int, default=900)
    ap.add_argument("--reuse-from", default=None, help="earlier run's jsonl: tag its invalid rows, reuse valid ones")
    ap.add_argument("--verify-thinking", action="store_true",
                    help="run harness/x2_thinking_verify.py's live check first (once; skipped if already recorded)")
    ap.add_argument("--tag-invalid", action="store_true",
                    help="post-process --out in place: tag invalid_race/invalid_thinking/invalid_infra_oom, no re-run")
    args = ap.parse_args(argv)
    if args.tag_invalid:
        n_race, n_thinking, n_oom = tag_invalid_rows(Path(args.out))
        print(f"tagged {n_race} invalid_race, {n_thinking} invalid_thinking, {n_oom} invalid_infra_oom rows in {args.out}")
        return
    note = "stopped: unknown"
    try:
        note = run(Path(args.out), args.models.split(","), smoke_n=args.smoke_n, deadline_h=args.deadline_h,
                   call_timeout_s=args.call_timeout_s, reuse_from=args.reuse_from,
                   verify_thinking=args.verify_thinking)
    except BaseException as e:  # noqa: BLE001 -- the queue must advance whatever happened
        note = f"stopped: {type(e).__name__}: {e}"[:300]
        raise
    finally:
        _advance(note)


if __name__ == "__main__":
    main()
