"""Outcome-table run: for each workload-pack item (trace-weighted priority order, r2_sessions
excluded, 340 items), runs llama3.1:8b under three configurations before moving to the next item:
  1. ollama_default      -- stock Ollama, no env override (collapses to the 4096 default on evo-t2s)
  2. ollama_igpu_enable   -- Ollama with OLLAMA_IGPU_ENABLE=1 set only inside this job's own server
                            start (never persisted; restored to absent on every server stop)
  3. llama_server_vulkan  -- llama.cpp llama-server, Vulkan backend, -c sized to this item's own
                            prompt_tokens (+256 headroom for the response)

Per (item, config): score (reusing results/workload_pack/grade.py's real GRADERS, against the
model's real response, not the oracle), TTFT, total latency, tokens sent vs processed, HTTP status,
the silent-truncation flag (processed < sent), effective context (from /api/ps for Ollama configs,
from the startup log's n_ctx for llama-server), and a machine-state snapshot (which runtime/device).

Resumable: a real item_id+config key already present in the output file's own rows is skipped.
Heartbeat: one row per (item, config) flushed immediately (append, not buffered) -- a partial night
is still fully usable data, per the explicit 2026-10-02 instruction this script was built to satisfy.
Every call is capped at --call-timeout-s (default 900 = 15 min). The whole run respects
--deadline-h, checked at the start of each item (not mid-item), so the job stops itself cleanly
rather than overrunning into the handover.

T2S SUBSET RUN (2026-10-08, docs/T2S_WEEK_PLAN.md step 5). With --models and/or --subset-n the run is generalised
the way harness/x2_outcome_table.py already is, without changing what the plain invocation above does:
  * --models llama3.1:8b,qwen3-8b (MODEL_MAP below: Ollama tag and the llama-server GGUF of each);
  * --subset-n 100 --subset-seed 20261007: a fixed trace-weighted subset drawn exactly as x2_outcome_table's
    qwen32b_subset (Efraimidis-Spirakis weighted sampling without replacement; trace_weighted_subset below, parity
    tested), recorded once in a "t2s_subset" record at the top of --out. Seed 20261007 is QWEN32B_SUBSET_SEED, so the
    T2S subset is the same 100 items as evo-x2's qwen3-32b subset;
  * --canary-gate: before any measured item, each (model, config) runs the 10 canary items (5 shortest gsm8k + 5
    shortest function_calling, as x2_outcome_table.canary_items); a canary error rate above 10% or a mean canary
    score below 0.5 halts that (model, config) for the rest of the run with an ALERT record;
  * every row of this path records model_key, thinking_setting and thinking_leak; qwen3 on llama-server starts with
    --reasoning-budget 0 and sends chat_template_kwargs.enable_thinking=false (the x2_outcome_table mechanism);
  * GSM8K: the fixed #### scorer (SCORER_VERSION 2) and the lenient column, as for every row of this file.
QUEUE EXIT: when launched by the queue (APU_QUEUE_JOB_ID set), main() always ends with one t2s_queue.advance(note).

Usage:
  py -3.12 harness/t2s_outcome_table.py --out results/t2s_outcome_table_<stem>.jsonl \\
      [--smoke-n 10] [--deadline-h 5] [--call-timeout-s 900]
  py -3.12 harness/t2s_outcome_table.py --out results/t2s_outcome_subset_v1.jsonl --models llama3.1:8b,qwen3-8b \\
      --subset-n 100 --subset-seed 20261007 --canary-gate --deadline-h 30
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Deployed two ways: inside this repo (harness/t2s_outcome_table.py, REPO is its grandparent), and
# flat on evo-t2s itself (C:\apu\ovn\t2s_outcome_table.py, with analysis/, results/, t2s_k1_ollama.py
# etc. all siblings or sub-dirs of that same C:\apu\ovn directory -- confirmed against this project's
# real remote layout, same fix as t2s_handover.py needed). Detect which layout this is by checking
# for a sibling "analysis" directory next to this file first.
_here = Path(__file__).resolve().parent
if (_here / "analysis").is_dir():
    REPO = _here  # flat deployment: this file's own directory IS the effective repo root
else:
    REPO = _here.parents[0]  # repo layout: harness/t2s_outcome_table.py -> repo root
sys.path.insert(0, str(_here))
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO))

import host_config as hc  # noqa: E402
import t2s_k1_ollama as k1  # noqa: E402

MODEL_TAG_OLLAMA = "llama3.1:8b"
GGUF_PATH = r"C:\apu\models\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
LLAMA_SERVER_EXE = r"C:\apu\bin\llama-b10970\llama-server.exe"
LLAMA_SERVER_PORT = 58199

CONFIGS = ("ollama_default", "ollama_igpu_enable", "llama_server_vulkan")

# model_key -> (Ollama tag, llama-server GGUF, llama-server row model_id). llama3.1:8b keeps the model ids the
# earlier T2S rows used, so old and new rows join. GGUF file names are the ones t2s_overnight.MODEL_FILES verified.
MODEL_MAP = {
    "llama3.1:8b": (MODEL_TAG_OLLAMA, GGUF_PATH, "llama3.1-8b-gguf"),
    "qwen3-8b": ("qwen3:8b", r"C:\apu\models\Qwen3-8B-Q4_K_M.gguf", "qwen3-8b-gguf"),
}
DEFAULT_MODEL = "llama3.1:8b"
# Same values as harness/x2_outcome_table.py (parity tested in tests/test_t2s_outcome_subset.py).
LLAMA_SERVER_THINKING_ARGS = ["--reasoning-budget", "0"]
LLAMA_REQUEST_THINKING_FIELDS = {"chat_template_kwargs": {"enable_thinking": False}}
THINKING_SETTING_LLAMA = ("llama_server: server flag --reasoning-budget 0 + per-request "
                          "chat_template_kwargs.enable_thinking=false")
THINKING_SETTING_OLLAMA = "ollama: per-request \"think\": false"
CANARY_ERROR_RATE_THRESHOLD = 0.10
CANARY_MIN_MEAN_SCORE = 0.5
SUBSET_SEED_DEFAULT = 20261007  # x2_outcome_table.QWEN32B_SUBSET_SEED


def trace_weighted_subset(items, weight_by_id, n, seed):
    """Byte-for-byte the same draw as x2_outcome_table.qwen32b_subset (Efraimidis-Spirakis: key = u ** (1 / w),
    the n largest keys win; items sorted by id before drawing; weight-0 items never drawn but still consume one
    random number). Duplicated, not imported, because this file is deployed flat to evo-t2s on its own."""
    import random
    rng = random.Random(seed)
    keyed = []
    for it in sorted(items, key=lambda it: it["item_id"]):
        w = weight_by_id.get(it["item_id"], 0.0)
        u = rng.random()
        if w > 0:
            keyed.append((u ** (1.0 / w), it["item_id"]))
    keyed.sort(reverse=True)
    return {iid for _, iid in keyed[:n]}


def canary_items(items):
    """5 shortest gsm8k + 5 shortest function_calling items by prompt_tokens (x2_outcome_table.canary_items)."""
    gsm8k = sorted((it for it in items if it["family"] == "gsm8k"), key=lambda it: it["prompt_tokens"])[:5]
    fcall = sorted((it for it in items if it["family"] == "function_calling"), key=lambda it: it["prompt_tokens"])[:5]
    return gsm8k + fcall


def classify_error_cause(row):
    """none / context_overflow / timeout / connection / other, as x2_outcome_table.classify_error_cause."""
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


def canary_gate_check(canary_rows):
    """(passed, error_rate, mean_score), x2_outcome_table.canary_gate_check's rule."""
    if not canary_rows:
        return True, 0.0, 1.0
    errors = sum(1 for r in canary_rows
                 if classify_error_cause(r) not in ("none", "context_overflow") or r.get("thinking_leak"))
    scores = [r.get("score") or 0.0 for r in canary_rows]
    error_rate = errors / len(canary_rows)
    mean_score = sum(scores) / len(scores)
    return (error_rate <= CANARY_ERROR_RATE_THRESHOLD and mean_score >= CANARY_MIN_MEAN_SCORE), error_rate, mean_score


def thinking_leak(text):
    return "<think>" in (text or "")


def chat_template_fields(ollama_tag=None, gguf=None):
    """chat_template_source / chat_template_sha256 (harness/chat_template_source.py) for the Ollama tag or the
    llama-server GGUF of a row (2026-10-08). Informational only; never raises; rows without it stay reusable."""
    try:
        import chat_template_source as cts
        info = cts.ollama_tag_template(ollama_tag) if ollama_tag else cts.llama_server_template(gguf)
        return {"chat_template_source": info["chat_template_source"],
                "chat_template_sha256": info["chat_template_sha256"]}
    except Exception as e:
        return {"chat_template_source": None, "chat_template_sha256": None,
                "chat_template_error": repr(e)[:200]}


def utc_iso():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load_items_trace_weighted(repo):
    """Real 340 non-r2_sessions items, sorted by real trace-weight descending (so a partial run is
    still representative, per the explicit instruction). Uses the precomputed per-item weights file
    (results/workload_pack/item_weights_trace_weighted.json, generated by
    analysis/trace_weighted_pack.py::compute_item_weights and committed) rather than recomputing from
    the parquet directly -- found live 2026-10-02: evo-t2s's Python has no pyarrow/fastparquet
    installed, so pandas.read_parquet is not usable there at all. The precomputed file is the exact
    same real weights, just computed once (locally, where pyarrow is available) instead of per-run."""
    sys.path.insert(0, str(repo))
    weights_path = repo / "results" / "workload_pack" / "item_weights_trace_weighted.json"
    if not weights_path.exists():
        raise FileNotFoundError(str(weights_path))
    weight_by_id = json.loads(weights_path.read_text(encoding="utf-8"))
    from analysis import trace_weighted_pack as twp
    items = twp.load_pack_items(repo)
    pairs = [(it, weight_by_id.get(it["item_id"], 0.0)) for it in items if it["family"] != "r2_sessions"]
    pairs.sort(key=lambda p: p[1], reverse=True)
    return [p[0] for p in pairs]


def load_graders():
    sys.path.insert(0, str(REPO / "results" / "workload_pack"))
    import grade as g
    return g


def score_response(grade_module, item, response_text):
    method = item["grading"]["method"]
    oracle = item["oracle_answer"]
    fn = grade_module.GRADERS[method]
    try:
        if method == "exact_substring":
            return fn(oracle, response_text, case_sensitive=item["grading"].get("case_sensitive", True))
        if method in ("exact_dict_match", "session_rule_recall"):
            # the grader expects a parsed dict; a real model response is free text -- best-effort
            # parse, and a response that cannot be parsed as the expected structure scores 0.0
            # (a real failure mode, not an error to hide).
            m = re.search(r"\{.*\}", response_text, re.S)
            parsed = json.loads(m.group(0)) if m else {}
            return fn(oracle, parsed)
        if method == "final_number_match":
            # Same bug and fix as harness/x2_outcome_table.py (commit 1349e6b): grade.py's grader compares the
            # oracle against a bare number, but this used to pass the whole response, so every gsm8k answer
            # scored 0. Extract the number after the LAST "####" (the answer format the prompt asks for).
            found = _FINAL_NUMBER_RE.findall(response_text or "")
            return fn(oracle, found[-1]) if found else 0.0
    except Exception:
        return 0.0
    return 0.0


# Kept byte-identical in behaviour to harness/x2_outcome_table.py (tests/test_gsm8k_lenient_score.py checks
# parity); duplicated rather than imported because this file is deployed flat to evo-t2s on its own.
_FINAL_NUMBER_RE = re.compile(r"####\s*\**\s*(-?[\d,]+(?:\.\d+)?)")
# Rows written before the final_number_match fix carry no scorer_version; their gsm8k score is always 0.
SCORER_VERSION = 2
_THINK_TAG_RE = re.compile(r"<think>.*?(</think>|$)", re.S)
# Lenient gsm8k score: see harness/x2_outcome_table.py (gsm8k_last_sentence) for the exact definition.
_SENTENCE_SPLIT_RE = re.compile(r"\n+|(?<=[.!?])\s+")
_LENIENT_NUMBER_RE = re.compile(r"(?:(?<![\w.])-)?\$?(?:\d{1,3}(?:,\d{3})+(?!\d)|\d+)(?:\.\d+)?")


def strip_thinking(text):
    return _THINK_TAG_RE.sub("", text or "").strip()


def gsm8k_last_sentence(response_text):
    pieces = [p.strip() for p in _SENTENCE_SPLIT_RE.split(strip_thinking(response_text))]
    pieces = [p for p in pieces if p]
    return pieces[-1] if pieces else ""


def gsm8k_lenient_score(oracle, response_text):
    try:
        want = float(str(oracle).replace(",", "").replace("$", "").strip())
    except ValueError:
        return 0.0
    for tok in _LENIENT_NUMBER_RE.findall(gsm8k_last_sentence(response_text)):
        try:
            if float(tok.replace("$", "").replace(",", "")) == want:
                return 1.0
        except ValueError:
            continue
    return 0.0


def gsm8k_score_fields(grade_module, item, response_text):
    """score_strict / format_ok / score_lenient for a final_number_match item, {} otherwise."""
    if (item.get("grading") or {}).get("method") != "final_number_match":
        return {}
    try:
        text = strip_thinking(response_text)
        return {"score_strict": score_response(grade_module, item, text),
                "format_ok": bool(_FINAL_NUMBER_RE.search(text or "")),
                "score_lenient": gsm8k_lenient_score(item.get("oracle_answer"), text)}
    except Exception:
        return {}  # the extra columns must never turn a measured row into a driver exception


def scored_fields(item, output_text):
    """Every score-related field a scored row carries: score (strict for gsm8k), scorer_version, the gsm8k
    strict/lenient/format columns, and the response text (head and tail, 500 chars each) so a row can be
    rescored later without rerunning it."""
    grade_module = load_graders()
    out = {"score": score_response(grade_module, item, output_text), "scorer_version": SCORER_VERSION,
           "output_text": output_text[:500], "output_tail": output_text[-500:]}
    out.update(gsm8k_score_fields(grade_module, item, output_text))
    return out


def row_is_reusable(row):
    """A gsm8k row scored 200 by the pre-fix final_number_match path (no scorer_version) always scored 0 and is
    rerun; every other row is reused on resume exactly as before."""
    return not (row.get("family") == "gsm8k" and row.get("http_status") == 200
                and row.get("scorer_version", 1) < SCORER_VERSION)


def already_done_keys(out_path):
    done = set()
    if not out_path.exists():
        return done
    for line in out_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("record") == "outcome_row" and row_is_reusable(r):
            done.add((r["item_id"], r["config"], r.get("model_key") or DEFAULT_MODEL))
    return done


def emit(out_path, row):
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def run_ollama_call(ollama, model_tag, prompt, num_ctx, call_timeout_s):
    t0 = time.monotonic()
    resp = ollama.chat(model_tag, "", num_ctx=num_ctx,
                       messages=[{"role": "user", "content": prompt}],
                       max_tokens=256, think=False, keep_alive="2m", timeout=call_timeout_s)
    dt = time.monotonic() - t0
    return resp, dt


def get_ollama_ps_context_length(model_tag):
    try:
        req = urllib.request.Request("http://127.0.0.1:11434/api/ps")
        with urllib.request.urlopen(req, timeout=30) as r:
            ps = json.loads(r.read())
        for m in ps.get("models", []):
            if m.get("name") == model_tag or m.get("model") == model_tag:
                return m.get("context_length")
    except Exception:
        pass
    return None


def run_one_item_ollama(item, config_label, igpu_enable, out_path, call_timeout_s, model_key=None, canary=None):
    """model_key None: the original single-model row exactly as before. A model_key (the T2S subset path) selects
    the MODEL_MAP entry and adds model_key / thinking_setting / thinking_leak / canary to the row."""
    tag = MODEL_MAP[model_key][0] if model_key else MODEL_TAG_OLLAMA
    exe = hc._resolve_ollama_exe_for_serve()
    if igpu_enable:
        cmd = (f"$cmd = 'cmd.exe /c set OLLAMA_KEEP_ALIVE=0 && set OLLAMA_IGPU_ENABLE=1 && \"{exe}\" serve > "
               f"C:\\apu\\ovn\\ollama_serve_outcome_table.log 2>&1'; "
               "$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$cmd}; "
               "'pid=' + $r.ProcessId")
        subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=30)
    else:
        hc.start_ollama_server()
    row = {"record": "outcome_row", "item_id": item["item_id"], "family": item["family"],
          "config": config_label, "model_id": tag, "ts_utc": utc_iso()}
    if model_key:
        row.update({"model_key": model_key, "thinking_setting": THINKING_SETTING_OLLAMA, "canary": bool(canary)})
    row.update(chat_template_fields(ollama_tag=tag))
    try:
        hc.wait_for_ollama_ready(timeout_s=60)
        ollama = k1.OllamaClient()
        resp, dt = run_ollama_call(ollama, tag, item["prompt"], None, call_timeout_s)
        effective_ctx = get_ollama_ps_context_length(tag)
        output_text = resp.get("message") or ""
        sent = item["prompt_tokens"]
        processed = resp.get("prompt_eval_count")
        row.update(scored_fields(item, output_text))
        if model_key:
            row["thinking_leak"] = thinking_leak(output_text)
        row.update({
            "http_status": resp.get("status"), "ttft_s": None,  # non-streaming call: no separate TTFT
            "latency_s": dt, "sent_tokens": sent, "processed_tokens": processed,
            "silently_truncated": bool(processed is not None and processed < sent),
            "effective_context": effective_ctx, "machine_state": config_label,
            "chat_outcome": resp.get("outcome"), "error": resp.get("error"),
        })
    except Exception as e:
        row.update({"http_status": None, "score": 0.0, "error": f"driver exception: {e!r}"[:400]})
    finally:
        hc.stop_ollama_server()
    emit(out_path, row)
    return row


def run_one_item_llama_server(item, out_path, call_timeout_s, model_key=None, canary=None):
    """model_key None: the original single-model row exactly as before. A model_key (the T2S subset path) selects
    the MODEL_MAP GGUF, starts the server with LLAMA_SERVER_THINKING_ARGS, sends LLAMA_REQUEST_THINKING_FIELDS, and
    names the per-item server log by model so two models never overwrite each other's log."""
    import socket
    gguf = MODEL_MAP[model_key][1] if model_key else GGUF_PATH
    model_id = MODEL_MAP[model_key][2] if model_key else "llama3.1-8b-gguf"
    n_ctx = ((item["prompt_tokens"] + 256 + 255) // 256) * 256  # round up to a multiple of 256, +256 headroom
    if model_key:
        log_path = fr"C:\apu\ovn\results\t2s_outcome_subset_llamaserver_{model_id}_{item['item_id']}.log"
    else:
        log_path = fr"C:\apu\ovn\results\t2s_outcome_table_llamaserver_{item['item_id']}.log"
    cmd = [LLAMA_SERVER_EXE, "-m", gguf, "--port", str(LLAMA_SERVER_PORT), "--host", "127.0.0.1",
          "--no-webui", "-c", str(n_ctx), "-np", "1", "-t", "4", "--log-verbosity", "4", "-ngl", "99"]
    if model_key:
        cmd += LLAMA_SERVER_THINKING_ARGS
    row = {"record": "outcome_row", "item_id": item["item_id"], "family": item["family"],
          "config": "llama_server_vulkan", "model_id": model_id, "ts_utc": utc_iso(),
          "requested_n_ctx": n_ctx}
    if model_key:
        row.update({"model_key": model_key, "thinking_setting": THINKING_SETTING_LLAMA, "canary": bool(canary)})
    row.update(chat_template_fields(gguf=gguf))
    proc = None
    try:
        with open(log_path, "w", encoding="utf-8") as logfh:
            proc = subprocess.Popen(cmd, stdout=logfh, stderr=logfh)
        t0 = time.monotonic()
        ready = False
        while time.monotonic() - t0 < 180:
            try:
                with socket.create_connection(("127.0.0.1", LLAMA_SERVER_PORT), timeout=1):
                    ready = True
                    break
            except OSError:
                time.sleep(1)
        if not ready:
            row.update({"http_status": None, "score": 0.0, "error": "server did not open its port in time"})
            emit(out_path, row)  # 2026-10-07: was returned unrecorded, so startup failures left no row
            return row
        # wait for actual model-loaded readiness (port open != loaded, found live 2026-10-01)
        t1 = time.monotonic()
        loaded = False
        while time.monotonic() - t1 < call_timeout_s:
            try:
                req = urllib.request.Request(f"http://127.0.0.1:{LLAMA_SERVER_PORT}/v1/chat/completions",
                                             data=json.dumps({"model": "x", "messages": [{"role": "user", "content": "hi"}],
                                                              "max_tokens": 1}).encode(),
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    r.read()
                loaded = True
                break
            except urllib.error.HTTPError as e:
                if e.code != 503:
                    loaded = True
                    break
                time.sleep(3)
            except Exception:
                time.sleep(3)
        if not loaded:
            row.update({"http_status": None, "score": 0.0, "error": "server never left 'loading' state"})
            emit(out_path, row)  # 2026-10-07: was returned unrecorded, so startup failures left no row
            return row
        t2 = time.monotonic()
        req_body = {"model": "x", "messages": [{"role": "user", "content": item["prompt"]}],
                    "max_tokens": 256, "temperature": 0}
        if model_key:
            req_body.update(LLAMA_REQUEST_THINKING_FIELDS)
        body = json.dumps(req_body).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{LLAMA_SERVER_PORT}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=call_timeout_s) as r:
                data = json.loads(r.read())
            dt = time.monotonic() - t2
            output_text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
            usage = data.get("usage") or {}
            sent = item["prompt_tokens"]
            processed = usage.get("prompt_tokens")
            row.update(scored_fields(item, output_text))
            if model_key:
                msg = (data.get("choices") or [{}])[0].get("message", {}) or {}
                row["reasoning_chars"] = len(msg.get("reasoning_content") or "")
                row["thinking_leak"] = bool(row["reasoning_chars"]) or thinking_leak(output_text)
            row.update({
                "http_status": 200, "ttft_s": None, "latency_s": dt,
                "sent_tokens": sent, "processed_tokens": processed,
                "silently_truncated": bool(processed is not None and processed < sent),
                "effective_context": n_ctx, "machine_state": "llama_server_vulkan",
            })
        except urllib.error.HTTPError as e:
            row.update({"http_status": e.code, "score": 0.0, "error": e.read().decode(errors="replace")[:400]})
        except Exception as e:
            row.update({"http_status": None, "score": 0.0, "error": str(e)[:400]})
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
    emit(out_path, row)
    return row


def _default_runners(out_path, call_timeout_s):
    """config -> fn(item, model_key, canary) -> row. model_key None is the original single-model path."""
    return {
        "ollama_default": lambda it, mk, c: run_one_item_ollama(it, "ollama_default", False, out_path,
                                                               call_timeout_s, model_key=mk, canary=c),
        "ollama_igpu_enable": lambda it, mk, c: run_one_item_ollama(it, "ollama_igpu_enable", True, out_path,
                                                                   call_timeout_s, model_key=mk, canary=c),
        "llama_server_vulkan": lambda it, mk, c: run_one_item_llama_server(it, out_path, call_timeout_s,
                                                                          model_key=mk, canary=c),
    }


def cached_rows(out_path):
    """(item_id, config, model_key) -> the latest reusable outcome row in out_path."""
    out = {}
    if not out_path.exists():
        return out
    for line in out_path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("record") == "outcome_row" and row_is_reusable(r):
            out[(r["item_id"], r["config"], r.get("model_key") or DEFAULT_MODEL)] = r
    return out


def run(out_path, smoke_n=None, deadline_h=5.0, call_timeout_s=900, models=None, subset_n=None,
        subset_seed=SUBSET_SEED_DEFAULT, canary_gate=False, runners=None, items=None, weights=None):
    """models None: the original run (llama3.1:8b, rows exactly as before). models given (keys of MODEL_MAP): the
    T2S subset path, item-major, each item runs every model under every config. Returns a short exit note."""
    out_path = Path(out_path)
    if items is None:
        items = load_items_trace_weighted(REPO)
    all_items = list(items)
    if subset_n:
        if weights is None:
            weights = json.loads((REPO / "results" / "workload_pack" / "item_weights_trace_weighted.json")
                                 .read_text(encoding="utf-8"))
        subset = trace_weighted_subset(all_items, weights, subset_n, subset_seed)
        if not any(r.get("record") == "t2s_subset" for r in _read_records(out_path)):
            emit(out_path, {"record": "t2s_subset", "ts_utc": utc_iso(), "seed": subset_seed, "n": len(subset),
                            "method": "x2_outcome_table.qwen32b_subset (Efraimidis-Spirakis, trace weights)",
                            "item_ids": sorted(subset)})
        items = [it for it in all_items if it["item_id"] in subset]  # keeps the trace-weighted priority order
    if smoke_n:
        items = items[:smoke_n]
    model_keys = list(models) if models else [None]
    for mk in model_keys:
        if mk is not None and mk not in MODEL_MAP:
            raise ValueError(f"unknown model {mk!r}; known: {sorted(MODEL_MAP)}")
    runners = runners or _default_runners(out_path, call_timeout_s)
    cache = cached_rows(out_path)
    halted = set()

    def call(config, item, mk, canary=False):
        key = (item["item_id"], config, mk or DEFAULT_MODEL)
        if key in cache:
            return cache[key]
        t0 = time.monotonic()
        row = runners[config](item, mk, canary)
        cache[key] = row
        print(f"{item['item_id']} {mk or DEFAULT_MODEL} {config}: {time.monotonic() - t0:.1f}s "
              f"http={row.get('http_status')} score={row.get('score')}", flush=True)
        return row

    if canary_gate:
        for mk in model_keys:
            for config in CONFIGS:
                rows = [call(config, c, mk, canary=True) for c in canary_items(all_items)]
                passed, error_rate, mean_score = canary_gate_check(rows)
                emit(out_path, {"record": "canary_gate", "model_key": mk or DEFAULT_MODEL, "config": config,
                                "passed": passed, "error_rate": error_rate, "mean_score": mean_score,
                                "n": len(rows), "ts_utc": utc_iso()})
                if not passed:
                    halted.add((mk, config))
                    emit(out_path, {"record": "ALERT", "reason": "canary_gate_failed", "model_key": mk or DEFAULT_MODEL,
                                    "config": config, "error_rate": error_rate, "mean_score": mean_score,
                                    "ts_utc": utc_iso()})
                    print(f"ALERT: canary gate FAILED for {mk or DEFAULT_MODEL}/{config}: error_rate={error_rate:.1%} "
                          f"mean_score={mean_score:.2f}; halted for the rest of this run", flush=True)
    t_start = time.monotonic()
    deadline_s = deadline_h * 3600
    n_items_done = 0
    for item in items:
        if time.monotonic() - t_start > deadline_s:
            print(f"deadline reached ({deadline_h}h), stopping at {n_items_done} items done this run")
            return f"deadline reached after {n_items_done} items"
        emit(out_path, {"record": "heartbeat", "item_id": item["item_id"], "ts_utc": utc_iso()})
        for mk in model_keys:
            for config in CONFIGS:
                if (mk, config) in halted:
                    continue
                call(config, item, mk)
        n_items_done += 1
    print(f"done: {n_items_done} items this run, output {out_path}")
    gated = sorted(f"{mk or DEFAULT_MODEL}/{c}" for mk, c in halted)
    return f"completed {n_items_done} items" + (f"; canary gate halted {gated}" if gated else "")


def _read_records(path):
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def _advance(note, tq=None):
    """One t2s_queue.advance(note), only when the queue launched this process (APU_QUEUE_JOB_ID set)."""
    import os
    if not os.environ.get("APU_QUEUE_JOB_ID"):
        print(f"(not launched by the queue; would advance with note: {note})")
        return
    if tq is None:
        import t2s_queue as tq
    tq.advance(note)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke-n", type=int, default=None)
    ap.add_argument("--deadline-h", type=float, default=5.0)
    ap.add_argument("--call-timeout-s", type=int, default=900)
    ap.add_argument("--models", default=None, help=f"comma-separated keys of MODEL_MAP: {','.join(MODEL_MAP)}")
    ap.add_argument("--subset-n", type=int, default=None)
    ap.add_argument("--subset-seed", type=int, default=SUBSET_SEED_DEFAULT)
    ap.add_argument("--canary-gate", action="store_true")
    args = ap.parse_args(argv)
    note = "completed"
    try:
        if args.models or args.subset_n or args.canary_gate:
            import socket
            host_cfg = hc.require_host(socket.gethostname())
            try:
                hc.enforce_or_record_interactive_session(host_cfg)
            except SystemExit as e:
                note = f"stopped: {e}"[:400]
                return note
            import t2s_week_preflight as pf
            print(f"ollama pin: {pf.apply_ollama_pin()}", flush=True)  # side-by-side 0.34.4, if pinned
        models =[m for m in args.models.split(",") if m] if args.models else None
        note = run(Path(args.out), smoke_n=args.smoke_n, deadline_h=args.deadline_h,
                   call_timeout_s=args.call_timeout_s, models=models, subset_n=args.subset_n,
                   subset_seed=args.subset_seed, canary_gate=args.canary_gate)
    except BaseException as e:  # noqa: BLE001 -- the queue must advance whatever happened
        note = f"stopped: {e!r}"[:400]
        print(note, flush=True)
    finally:
        try:
            _advance(note)
        except Exception as e:
            print(f"queue advance failed: {e!r}", flush=True)
    return note


if __name__ == "__main__":
    main()
