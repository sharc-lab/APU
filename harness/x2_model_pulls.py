"""Queue job (evo-x2): pulls every Ollama model x2_outcome_table.py needs, verifies each one actually loads
with a real 1-token call, then exits. Queued directly ahead of x2_outcome_table_v2 so a missing model can
never again silently stall or corrupt that job's own data (2026-10-02: qwen3:30b-a3b-instruct-2507 did not
exist on the registry and 404'd for a whole night before this was caught; qwen3:32b was simply never pulled).

Self-healing: re-running this job (or a future one like it) is safe and fast once models are present --
already-present models are confirmed via a real /api/tags query (not a manifest-file guess) and never
re-pulled. This is also why x2_outcome_table.py itself does not need its own pull-on-demand logic: this job
is the single place that owns "do we have every model we need" and runs before the outcome table every time,
so the outcome table can assume its models are already there.

The model list is read from x2_outcome_table.MODEL_MAP (single source of truth -- no separate hardcoded list
here that could drift from the real one).

Resumable at the model level: if the process is killed mid-run (e.g. by someone else's stop_ollama_server()),
a fresh run skips every model /api/tags already reports present and resumes the pull for whatever is still
missing (ollama pull itself also resumes from partial blobs, so even a half-downloaded model is not restarted
from zero).

Usage: py -3.12 harness/x2_model_pulls.py [--out results/x2_model_pulls_<stem>.jsonl]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
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
import t2s_queue as tq  # noqa: E402
import x2_outcome_table as x2ot  # noqa: E402

OLLAMA_EXE = hc._resolve_ollama_exe_for_serve()
HEARTBEAT_EVERY_S = 60  # keep queue_<id>.log fresh during a long pull, well under the 30 min staleness threshold
PULL_TIMEOUT_S = 3600  # generous: a 20 GB model at a slow link should still finish well inside this


def utc_iso():
    return datetime.now(timezone.utc).isoformat()


def emit(out_path, row):
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()


def required_tags():
    """De-duplicated ollama tags from x2_outcome_table.MODEL_MAP, in that dict's own order."""
    seen = []
    for ollama_tag, _gguf in x2ot.MODEL_MAP.values():
        if ollama_tag not in seen:
            seen.append(ollama_tag)
    return seen


def currently_present_tags():
    """Real /api/tags query against a running server -- never a manifest-file guess (see host_config's own
    wait_for_ollama_ready docstring for why the port being open is not the same question as a model being
    loadable)."""
    try:
        req = urllib.request.Request("http://127.0.0.1:11434/api/tags")
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        return {m["name"] for m in data.get("models", [])}
    except Exception:
        return set()


def pull_one(tag, out_path, log):
    """Streams `ollama pull <tag>`'s own output while emitting a heartbeat line at least every
    HEARTBEAT_EVERY_S seconds regardless of the child's own output cadence (ollama's progress bar writes
    carriage-return-only updates that do not reliably flush as discrete lines when redirected)."""
    t0 = time.monotonic()
    proc = subprocess.Popen([OLLAMA_EXE, "pull", tag], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    last_heartbeat = time.monotonic()
    while proc.poll() is None:
        time.sleep(2)
        if time.monotonic() - last_heartbeat >= HEARTBEAT_EVERY_S:
            elapsed = time.monotonic() - t0
            log(f"heartbeat: still pulling {tag} ({elapsed:.0f}s elapsed)")
            emit(out_path, {"record": "heartbeat", "model_tag": tag, "elapsed_s": round(elapsed, 1), "ts_utc": utc_iso()})
            last_heartbeat = time.monotonic()
        if time.monotonic() - t0 > PULL_TIMEOUT_S:
            proc.kill()
            return {"pull_exit_code": None, "pull_error": f"killed after exceeding {PULL_TIMEOUT_S}s"}
    rc = proc.returncode
    tail = (proc.stdout.read() or "")[-2000:] if proc.stdout else ""
    return {"pull_exit_code": rc, "pull_error": None if rc == 0 else tail}


def verify_one(tag):
    client = k1.OllamaClient()
    resp = client.chat(tag, "hi", max_tokens=1)
    return {"verify_ok": resp.get("outcome") == "ok", "verify_error": resp.get("error")}


def run(out_path, log=print):
    hc.start_ollama_server()
    hc.wait_for_ollama_ready(timeout_s=60)
    try:
        present = currently_present_tags()
        results = []
        for tag in required_tags():
            row = {"record": "model_pull_result", "model_tag": tag, "ts_utc": utc_iso()}
            if tag in present:
                row.update({"already_present": True, "pulled": False, "pull_exit_code": None, "pull_error": None})
                log(f"{tag}: already present, skipping pull")
            else:
                log(f"{tag}: not present, pulling")
                pull_result = pull_one(tag, out_path, log)
                row.update({"already_present": False, "pulled": pull_result.get("pull_exit_code") == 0})
                row.update(pull_result)
            verify = verify_one(tag)
            row.update(verify)
            log(f"{tag}: pulled={row.get('pulled')} already_present={row.get('already_present')} "
                f"verify_ok={verify['verify_ok']}")
            emit(out_path, row)
            results.append(row)
        all_ok = all(r["verify_ok"] for r in results)
        emit(out_path, {"record": "run_end", "all_models_verified": all_ok, "n_models": len(results), "ts_utc": utc_iso()})
        return all_ok, results
    finally:
        hc.stop_ollama_server()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = Path(args.out) if args.out else REPO / "results" / f"x2_model_pulls_{stamp}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"logging to {out_path}")

    note = "completed"
    try:
        all_ok, results = run(out_path)
        if not all_ok:
            failed = [r["model_tag"] for r in results if not r["verify_ok"]]
            note = f"stopped: models failed verification: {failed}"
        print(f"done: {len(results)} models checked, all_ok={all_ok}")
    except Exception as e:
        note = f"stopped: {e!r}"[:400]
        print(note)
    finally:
        try:
            tq.advance(note)
        except Exception as e:
            print(f"queue advance failed: {e!r}")


if __name__ == "__main__":
    main()
