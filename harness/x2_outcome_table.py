"""Outcome-table run for evo-x2: for each workload-pack item (trace-weighted priority order,
r2_sessions excluded, 340 items), for each of --models (default: llama3.1:8b, qwen3-8b,
qwen3-4b-2507, qwen3-14b, qwen3-30b-a3b, qwen3-32b), runs two configurations:
  1. ollama_default   -- stock Ollama, no env override
  2. llama_server      -- llama.cpp llama-server, Vulkan backend, -c sized to this item's own
                          prompt_tokens (+256 headroom)
Order is model-major: for a given item, both configs run for model 1 before moving to model 2, and
all models for that item complete before moving to the next item -- matching the same per-item
priority-order resumability as the T2S outcome table.

Resumable (an (item_id, model, config) key already present in --out is skipped), heartbeat per item,
rows flushed per (item, model, config). Every call capped at --call-timeout-s. --deadline-h checked
at the start of each item.

Usage:
  py -3.12 harness/x2_outcome_table.py --out results/x2_outcome_table_<stem>.jsonl \\
      [--smoke-n 10] [--deadline-h 24] [--call-timeout-s 900]
"""
from __future__ import annotations

import argparse
import json
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
    # 2026-10-02 bug found live: "qwen3:30b-a3b-instruct-2507" does not exist on the Ollama registry (confirmed via
    # the registry's own manifest endpoint: 404) -- every ollama_default call for this model 404'd all night,
    # silently producing zero real data for this model's ollama leg. The real tag (also confirmed against the
    # registry: 200) is plain "qwen3:30b-a3b"; the GGUF path for the llama_server leg is unaffected.
    "qwen3-30b-a3b": ("qwen3:30b-a3b", r"C:\apu\models\Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"),
    "qwen3-32b": ("qwen3:32b", r"C:\apu\models\Qwen3-32B-Q4_K_M.gguf"),
}
LLAMA_SERVER_EXE = r"C:\apu\bin\llama-b10970\llama-server.exe"
LLAMA_SERVER_PORT = 58299
DEFAULT_MODELS = list(MODEL_MAP.keys())

# 2026-10-06 bug found live: qwen3-8b/14b/32b (base Qwen3, hybrid-thinking-by-default) burned their entire
# n_predict budget on hidden reasoning_content with content="" on llama_server -- confirmed with a raw
# response dump (finish_reason="length", content="", reasoning_content=full budget). --reasoning-budget 0
# at server startup fixes it (verified live: same model, same prompt, finish_reason="stop", correct
# content). Applied unconditionally -- harmless for models with no reasoning mode (llama3.1:8b) or that
# already default to non-thinking (qwen3-4b-2507, qwen3-30b-a3b are both -Instruct-2507 releases, confirmed
# by their own high pre-fix scores: 0.844-1.000 vs 0.055-0.222 for the hybrid-thinking models).
LLAMA_SERVER_REASONING_BUDGET_ARGS = ["--reasoning-budget", "0"]

# Canary gate (1c): 5 shortest gsm8k + 5 shortest function_calling items, run before the full block for
# every (model, config). A config that fails the gate is skipped for the rest of the run, never silently
# included in the full block.
CANARY_ERROR_RATE_THRESHOLD = 0.10
CANARY_MIN_MEAN_SCORE = 0.5
ROLLING_NON_OVERFLOW_ERROR_RATE_THRESHOLD = 0.10


def classify_error_cause(row):
    """Buckets a row's failure into one of: none (score counted normally), context_overflow (HTTP 400,
    expected), timeout, connection, other. Never a single flat 'error' bucket -- the validity overhaul
    this function exists for was triggered by exactly that flattening hiding a real race underneath a
    generic error rate."""
    if row.get("http_status") == 200 and not row.get("invalid_race") and not row.get("invalid_thinking"):
        return "none"
    if row.get("invalid_race"):
        return "connection"
    if row.get("invalid_thinking"):
        return "thinking_contamination"
    status = row.get("http_status")
    error = (row.get("error") or "")
    if status == 400 and ("context" in error.lower() or "exceed" in error.lower()):
        return "context_overflow"
    if "connection" in error.lower() or "refused" in error.lower() or row.get("chat_outcome") == "infra_not_ready":
        return "connection"
    if "timeout" in error.lower() or "timed out" in error.lower():
        return "timeout"
    return "other"


def canary_items(items):
    """5 shortest gsm8k + 5 shortest function_calling items, by prompt_tokens, from the already-loaded,
    trace-weighted item list (so this never re-reads the pack separately or risks a different ordering)."""
    gsm8k = sorted((it for it in items if it["family"] == "gsm8k"), key=lambda it: it["prompt_tokens"])[:5]
    fcall = sorted((it for it in items if it["family"] == "function_calling"), key=lambda it: it["prompt_tokens"])[:5]
    return gsm8k + fcall


def canary_gate_check(canary_rows):
    """Returns (passed, error_rate, mean_score). A row counts as a canary error if its error_cause is not
    'none' and not 'context_overflow' (an overflow on a short canary item would itself be a real bug in the
    canary selection, not an infra problem) -- matches item (d)'s 'never a single error rate' framing."""
    if not canary_rows:
        return True, 0.0, 1.0
    errors = sum(1 for r in canary_rows if classify_error_cause(r) not in ("none", "context_overflow"))
    scores = [r.get("score", 0.0) for r in canary_rows]
    error_rate = errors / len(canary_rows)
    mean_score = sum(scores) / len(scores)
    passed = error_rate <= CANARY_ERROR_RATE_THRESHOLD and mean_score >= CANARY_MIN_MEAN_SCORE
    return passed, error_rate, mean_score


def utc_iso():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load_items_trace_weighted(repo):
    weights_path = repo / "results" / "workload_pack" / "item_weights_trace_weighted.json"
    if not weights_path.exists():
        raise FileNotFoundError(str(weights_path))
    weight_by_id = json.loads(weights_path.read_text(encoding="utf-8"))
    sys.path.insert(0, str(repo))
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
            import re
            m = re.search(r"\{.*\}", response_text, re.S)
            parsed = json.loads(m.group(0)) if m else {}
            return fn(oracle, parsed)
        if method == "final_number_match":
            return fn(oracle, response_text)
    except Exception:
        return 0.0
    return 0.0


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
        if r.get("record") == "outcome_row":
            done.add((r["item_id"], r["model_id"], r["config"]))
    return done


def emit(out_path, row):
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def run_one_ollama(item, model_key, ollama_tag, out_path, call_timeout_s):
    hc.start_ollama_server()
    row = {"record": "outcome_row", "item_id": item["item_id"], "family": item["family"],
          "config": "ollama_default", "model_id": model_key, "ts_utc": utc_iso()}
    try:
        # 2026-10-06 bug found live: this used to discard wait_for_ollama_ready's own return value and call
        # ollama.chat() regardless, which is exactly how a server that never actually came up (see
        # host_config.stop_ollama_server's own docstring for the race) produced "connection refused" on 409
        # of ~650 real ollama_default calls over the weekend. One forced restart-and-rewait before giving up
        # distinguishes a genuine infra failure from a model/content outcome, rather than silently scoring it
        # as a model fabrication (score=0.0 either way, but "infra_not_ready" is now a distinct chat_outcome).
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
                           max_tokens=256, think=False, keep_alive="2m", timeout=call_timeout_s)
        dt = time.monotonic() - t0
        output_text = resp.get("message") or ""
        grade_module = load_graders()
        score = score_response(grade_module, item, output_text)
        sent = item["prompt_tokens"]
        processed = resp.get("prompt_eval_count")
        row.update({"http_status": resp.get("status"), "score": score, "latency_s": dt,
                   "sent_tokens": sent, "processed_tokens": processed,
                   "silently_truncated": bool(processed is not None and processed < sent),
                   "chat_outcome": resp.get("outcome"), "error": resp.get("error"),
                   "thinking_disabled": True, "output_text": output_text[:500]})
    except Exception as e:
        row.update({"http_status": None, "score": 0.0, "error": f"driver exception: {e!r}"[:400]})
    finally:
        hc.stop_ollama_server()
    emit(out_path, row)
    return row


def run_one_llama_server(item, model_key, gguf_path, out_path, call_timeout_s):
    import socket
    n_ctx = ((item["prompt_tokens"] + 256 + 255) // 256) * 256
    log_path = fr"C:\apu\ovn\results\x2_outcome_table_llamaserver_{model_key}_{item['item_id']}.log"
    cmd = ([LLAMA_SERVER_EXE, "-m", gguf_path, "--port", str(LLAMA_SERVER_PORT), "--host", "127.0.0.1",
          "--no-webui", "-c", str(n_ctx), "-np", "1", "-t", "8", "--log-verbosity", "4", "-ngl", "99"]
          + LLAMA_SERVER_REASONING_BUDGET_ARGS)
    row = {"record": "outcome_row", "item_id": item["item_id"], "family": item["family"],
          "config": "llama_server", "model_id": model_key, "ts_utc": utc_iso(), "requested_n_ctx": n_ctx}
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
            return row
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
            return row
        t2 = time.monotonic()
        body = json.dumps({"model": "x", "messages": [{"role": "user", "content": item["prompt"]}],
                           "max_tokens": 256, "temperature": 0}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{LLAMA_SERVER_PORT}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=call_timeout_s) as r:
                data = json.loads(r.read())
            dt = time.monotonic() - t2
            output_text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
            usage = data.get("usage") or {}
            grade_module = load_graders()
            score = score_response(grade_module, item, output_text)
            sent = item["prompt_tokens"]
            processed = usage.get("prompt_tokens")
            row.update({"http_status": 200, "score": score, "latency_s": dt, "sent_tokens": sent,
                       "processed_tokens": processed,
                       "silently_truncated": bool(processed is not None and processed < sent),
                       "thinking_disabled": True, "output_text": output_text[:500],
                       "finish_reason": (data.get("choices") or [{}])[0].get("finish_reason")})
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


QWEN32B_SUBSET_N = 100  # (e): qwen3-32b runs on a 100-item trace-weighted subset, not the full 340


def run(out_path, models, smoke_n=None, deadline_h=24.0, call_timeout_s=900):
    items = load_items_trace_weighted(REPO)
    if smoke_n:
        items = items[:smoke_n]
    canaries = canary_items(items)
    done = already_done_keys(out_path)
    t_start = time.monotonic()
    deadline_s = deadline_h * 3600

    gated_off = set()  # (model_key, config) pairs that failed their canary gate -- skipped for the rest of this run
    rolling = {}  # (model_key, config) -> {"n", "non_overflow_errors"} for the rolling-error-rate alert

    def run_canary_gate(model_key, config, ollama_tag, gguf_path):
        """(c): 10 canaries (5 short gsm8k + 5 short function_calling) before the first real item for this
        (model, config). Reuses an already-done row for a canary item id instead of re-running it (e):
        resumable runs never re-pay for a canary this file already has a valid row for."""
        rows = []
        for canary in canaries:
            key = (canary["item_id"], model_key, config)
            if key in done:
                rows.append({"score": 1.0, "http_status": 200})  # cached-valid stand-in for the gate check
                continue
            if config == "ollama_default":
                row = run_one_ollama(canary, model_key, ollama_tag, out_path, call_timeout_s)
            else:
                row = run_one_llama_server(canary, model_key, gguf_path, out_path, call_timeout_s)
            rows.append(row)
        passed, error_rate, mean_score = canary_gate_check(rows)
        if not passed:
            print(f"ALERT: canary gate FAILED for {model_key}/{config}: error_rate={error_rate:.1%} "
                 f"mean_score={mean_score:.3f} -- skipping this (model, config) for the rest of the run")
            emit(out_path, {"record": "alert", "model_id": model_key, "config": config, "ts_utc": utc_iso(),
                           "reason": "canary_gate_failed", "error_rate": error_rate, "mean_score": mean_score})
        return passed

    n_items_done = 0
    for item in items:
        if time.monotonic() - t_start > deadline_s:
            print(f"deadline reached ({deadline_h}h), stopping at {n_items_done} items done this run")
            break
        emit(out_path, {"record": "heartbeat", "item_id": item["item_id"], "ts_utc": utc_iso()})
        for model_key in models:
            ollama_tag, gguf_path = MODEL_MAP[model_key]
            for config in ("ollama_default", "llama_server"):
                if (model_key, config) in gated_off:
                    continue
                if model_key == "qwen3-32b" and n_items_done >= QWEN32B_SUBSET_N:
                    continue
                key = (item["item_id"], model_key, config)
                if key in done:
                    continue
                if (model_key, config) not in rolling:
                    rolling[(model_key, config)] = {"n": 0, "non_overflow_errors": 0}
                    if not run_canary_gate(model_key, config, ollama_tag, gguf_path):
                        gated_off.add((model_key, config))
                        continue
                t0 = time.monotonic()
                if config == "ollama_default":
                    row = run_one_ollama(item, model_key, ollama_tag, out_path, call_timeout_s)
                else:
                    row = run_one_llama_server(item, model_key, gguf_path, out_path, call_timeout_s)
                dt = time.monotonic() - t0
                cause = classify_error_cause(row)
                st = rolling[(model_key, config)]
                st["n"] += 1
                if cause not in ("none", "context_overflow"):
                    st["non_overflow_errors"] += 1
                if st["n"] >= 10:
                    rate = st["non_overflow_errors"] / st["n"]
                    if rate > ROLLING_NON_OVERFLOW_ERROR_RATE_THRESHOLD:
                        print(f"ALERT: rolling non-overflow error rate for {model_key}/{config} is {rate:.1%} (n={st['n']})")
                        emit(out_path, {"record": "alert", "model_id": model_key, "config": config, "ts_utc": utc_iso(),
                                       "reason": "rolling_error_rate", "rate": rate, "n": st["n"]})
                print(f"{item['item_id']} {model_key} {config}: {dt:.1f}s http={row.get('http_status')} "
                     f"score={row.get('score')} cause={cause}")
        n_items_done += 1
    print(f"done: {n_items_done} items this run, output {out_path}")


# (b): models that default to hybrid thinking mode on this build, pre-fix -- any llama_server row from one
# of these without "thinking_disabled": true was measured before --reasoning-budget 0 existed and is
# suspect. llama3.1:8b has no reasoning mode; qwen3-4b-2507/qwen3-30b-a3b are -Instruct-2507 releases,
# already non-thinking by default (confirmed by their own high pre-fix scores).
THINKING_BY_DEFAULT_MODELS = {"qwen3-8b", "qwen3-14b", "qwen3-32b"}

_CONNECTION_REFUSED_MARKERS = ("actively refused", "winerror 10061", "connection refused")


def tag_invalid_rows(path):
    """(b): post-processing only, no re-run. Tags every already-written weekend row:
      invalid_race     -- an ollama_default row whose error text matches the connection-refused race
                           (host_config/x2_outcome_table fix, 2026-10-06).
      invalid_thinking  -- a llama_server row for a THINKING_BY_DEFAULT_MODELS model with no
                           "thinking_disabled": true marker (measured before --reasoning-budget 0 existed).
    Both tags are additive booleans on the existing row; nothing else is changed. Idempotent. Returns
    (n_race, n_thinking)."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                rows.append(line)
    n_race = n_thinking = 0
    for r in rows:
        if not isinstance(r, dict) or r.get("record") != "outcome_row":
            continue
        error_text = (r.get("error") or "").lower()
        if r.get("config") == "ollama_default" and any(m in error_text for m in _CONNECTION_REFUSED_MARKERS):
            r["invalid_race"] = True
            n_race += 1
        if (r.get("config") == "llama_server" and r.get("model_id") in THINKING_BY_DEFAULT_MODELS
                and not r.get("thinking_disabled")):
            r["invalid_thinking"] = True
            n_thinking += 1
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write((json.dumps(r, default=str) if isinstance(r, dict) else r) + "\n")
    return n_race, n_thinking


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--smoke-n", type=int, default=None)
    ap.add_argument("--deadline-h", type=float, default=24.0)
    ap.add_argument("--call-timeout-s", type=int, default=900)
    ap.add_argument("--tag-invalid", action="store_true",
                    help="post-process --out in place: tag invalid_race/invalid_thinking rows, no re-run")
    args = ap.parse_args(argv)
    if args.tag_invalid:
        n_race, n_thinking = tag_invalid_rows(Path(args.out))
        print(f"tagged {n_race} invalid_race rows, {n_thinking} invalid_thinking rows in {args.out}")
        return
    models = args.models.split(",")
    run(Path(args.out), models, smoke_n=args.smoke_n, deadline_h=args.deadline_h, call_timeout_s=args.call_timeout_s)


if __name__ == "__main__":
    main()
