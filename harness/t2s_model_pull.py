"""Queue job (evo-t2s): pull the Ollama library tags the T2S week needs and are missing from the store, verify each
loads with a 1-token call, exit. Found 2026-10-15 at the start of the window: the store
(<profile>\\.ollama\\models, HOSTS["EVO-T2S"]) has llama3.1:8b and qwen3:8b but no qwen3:14b, which every R2 entry of
queues/t2s_week_queue.json uses and t2s_week_preflight.py hard-checks. Loaded ahead of t2s_wk_preflight
(queues/t2s_week_pulls.json), so the preflight then sees the model or fails on its absence.

Safe to re-run: a tag /api/tags already reports is never pulled again (ollama pull also resumes partial blobs). Uses the
pinned Ollama (t2s_week_preflight.apply_ollama_pin, C:\\apu\\ovn\\ollama_pin.json) for both the server and the pull
client, so the store is written by the same 0.34.4 the week runs. Refuses while anyone is logged in (interactive_guard).
Output: results/t2s_model_pull_v1.jsonl. Calls t2s_queue.advance() exactly once on exit; sets no pause.

Usage (C:\\apu\\ovn): python t2s_model_pull.py [--tags qwen3:14b] [--out results/t2s_model_pull_v1.jsonl]
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
if (_here.parent / "harness").is_dir():
    sys.path.insert(0, str(_here.parent / "harness"))

DEFAULT_TAGS = ("qwen3:14b",)
HEARTBEAT_EVERY_S = 60
PULL_TIMEOUT_S = 3 * 3600


def utc_iso():
    return datetime.now(timezone.utc).isoformat()


def _norm(tag):
    return tag[:-len(":latest")] if tag.endswith(":latest") else tag


def present_tags(base="http://127.0.0.1:11434"):
    with urllib.request.urlopen(base + "/api/tags", timeout=10) as r:
        return {_norm(m["name"]) for m in json.loads(r.read()).get("models", [])}


def pull_one(exe, tag, emit, log, popen=None):
    """Streams `ollama pull` (drained by a thread so the child never blocks on a full pipe, x2_model_pulls' fix) and
    emits a heartbeat row every HEARTBEAT_EVERY_S so the queue log stays fresh."""
    if popen is None:
        from proc_util import popen_hidden as popen
    import subprocess
    t0 = time.monotonic()
    proc = popen([exe, "pull", tag], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8",
                 errors="replace")
    tail = []
    th = threading.Thread(target=lambda: [tail.append(c) for c in iter(lambda: proc.stdout.read(4096), "")],
                          daemon=True)
    th.start()
    last = time.monotonic()
    while proc.poll() is None:
        time.sleep(2)
        if time.monotonic() - last >= HEARTBEAT_EVERY_S:
            el = time.monotonic() - t0
            log(f"heartbeat: pulling {tag} ({el:.0f}s)")
            emit({"record": "heartbeat", "model_tag": tag, "elapsed_s": round(el, 1), "ts_utc": utc_iso()})
            last = time.monotonic()
        if time.monotonic() - t0 > PULL_TIMEOUT_S:
            proc.kill()
            return {"pull_exit_code": None, "pull_error": f"killed after {PULL_TIMEOUT_S}s"}
    th.join(timeout=10)
    return {"pull_exit_code": proc.returncode, "pull_s": round(time.monotonic() - t0, 1),
            "pull_error": None if proc.returncode == 0 else "".join(tail)[-2000:]}


def verify_one(tag):
    import t2s_k1_ollama as k1
    resp = k1.OllamaClient().chat(tag, "hi", max_tokens=1)
    return {"verify_ok": resp.get("outcome") == "ok", "verify_error": resp.get("error")}


def run(tags, emit, log):
    import host_config as hc
    import t2s_week_preflight as pf
    pin = pf.apply_ollama_pin()
    exe = hc._resolve_ollama_exe_for_serve()
    emit({"record": "run_start", "tags": list(tags), "pin": pin, "ollama_exe": exe, "ts_utc": utc_iso()})
    hc.stop_ollama_server()
    hc.start_ollama_server()
    try:
        if not hc.wait_for_ollama_ready(timeout_s=60):
            raise RuntimeError("ollama did not become ready")
        have = present_tags()
        rows = []
        for tag in tags:
            row = {"record": "model_pull_result", "model_tag": tag, "already_present": tag in have}
            if not row["already_present"]:
                log(f"{tag}: pulling")
                row.update(pull_one(exe, tag, emit, log))
            row.update(verify_one(tag))
            row["ts_utc"] = utc_iso()
            emit(row)
            log(json.dumps(row)[:400])
            rows.append(row)
        ok = all(r["verify_ok"] for r in rows)
        emit({"record": "run_end", "all_verified": ok, "ts_utc": utc_iso()})
        return ok, rows
    finally:
        hc.stop_ollama_server()


def main(argv=None, tq=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", default=",".join(DEFAULT_TAGS))
    ap.add_argument("--out", default="results/t2s_model_pull_v1.jsonl")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def emit(row):
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def log(msg):
        print(f"[{utc_iso()}] {msg}", flush=True)

    note = "completed"
    try:
        import host_config as hc
        cfg = hc.require_host(socket.gethostname())
        try:
            hc.enforce_or_record_interactive_session(cfg)
        except SystemExit as e:
            note = f"stopped: {e}"[:400]
            return note
        ok, rows = run([t for t in args.tags.split(",") if t], emit, log)
        if not ok:
            note = "stopped: models failed verification: " + ",".join(r["model_tag"] for r in rows if not r["verify_ok"])
    except BaseException as e:   # SystemExit from require_host included: always reach advance()
        note = f"stopped: {e!r}"[:400]
        log(note)
    finally:
        try:
            if tq is None:
                import t2s_queue as tq
            tq.advance(note)
        except Exception as e:
            log(f"queue advance failed: {e!r}")
    return note


if __name__ == "__main__":
    main()
