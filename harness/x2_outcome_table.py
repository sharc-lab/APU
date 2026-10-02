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
    "qwen3-30b-a3b": ("qwen3:30b-a3b-instruct-2507", r"C:\apu\models\Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"),
    "qwen3-32b": ("qwen3:32b", r"C:\apu\models\Qwen3-32B-Q4_K_M.gguf"),
}
LLAMA_SERVER_EXE = r"C:\apu\bin\llama-b10970\llama-server.exe"
LLAMA_SERVER_PORT = 58299
DEFAULT_MODELS = list(MODEL_MAP.keys())


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
        hc.wait_for_ollama_ready(timeout_s=60)
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
                   "chat_outcome": resp.get("outcome"), "error": resp.get("error")})
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
    cmd = [LLAMA_SERVER_EXE, "-m", gguf_path, "--port", str(LLAMA_SERVER_PORT), "--host", "127.0.0.1",
          "--no-webui", "-c", str(n_ctx), "-np", "1", "-t", "8", "--log-verbosity", "4", "-ngl", "99"]
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
                       "silently_truncated": bool(processed is not None and processed < sent)})
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


def run(out_path, models, smoke_n=None, deadline_h=24.0, call_timeout_s=900):
    items = load_items_trace_weighted(REPO)
    if smoke_n:
        items = items[:smoke_n]
    done = already_done_keys(out_path)
    t_start = time.monotonic()
    deadline_s = deadline_h * 3600
    n_items_done = 0
    for item in items:
        if time.monotonic() - t_start > deadline_s:
            print(f"deadline reached ({deadline_h}h), stopping at {n_items_done} items done this run")
            break
        emit(out_path, {"record": "heartbeat", "item_id": item["item_id"], "ts_utc": utc_iso()})
        for model_key in models:
            ollama_tag, gguf_path = MODEL_MAP[model_key]
            for config in ("ollama_default", "llama_server"):
                key = (item["item_id"], model_key, config)
                if key in done:
                    continue
                t0 = time.monotonic()
                if config == "ollama_default":
                    row = run_one_ollama(item, model_key, ollama_tag, out_path, call_timeout_s)
                else:
                    row = run_one_llama_server(item, model_key, gguf_path, out_path, call_timeout_s)
                dt = time.monotonic() - t0
                print(f"{item['item_id']} {model_key} {config}: {dt:.1f}s http={row.get('http_status')} score={row.get('score')}")
        n_items_done += 1
    print(f"done: {n_items_done} items this run, output {out_path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--smoke-n", type=int, default=None)
    ap.add_argument("--deadline-h", type=float, default=24.0)
    ap.add_argument("--call-timeout-s", type=int, default=900)
    args = ap.parse_args(argv)
    models = args.models.split(",")
    run(Path(args.out), models, smoke_n=args.smoke_n, deadline_h=args.deadline_h, call_timeout_s=args.call_timeout_s)


if __name__ == "__main__":
    main()
