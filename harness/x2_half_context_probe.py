"""Half-context three-marker probe (evo-x2, never run before 2026-10-06). Tests whether content survives
at HALF the loaded context window (not overflow), position-dependently: three distinct markers placed at
the start, middle, and end of a filler document sized to roughly half of num_ctx, plus a system-level rule
("end every answer with CODE-<code>"), library llama3.1:8b, num_ctx=8192.

One call per rep: system prompt carries the rule and the three markers' existence is not announced (the
model is simply asked, at the end of the filler, to repeat all three markers verbatim and follow the rule).
Which of the three positions survive, and whether the rule survives, is the real result -- this is a
position-in-context question, not a context-overflow question (the prompt is sized to roughly half the
window, deliberately not near it).

Usage: py -3.12 harness/x2_half_context_probe.py --out results/x2_half_context_probe.jsonl [--reps 5]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_here = Path(__file__).resolve().parent
if (_here / "analysis").is_dir():
    REPO = _here
else:
    REPO = _here.parents[0]
sys.path.insert(0, str(_here))
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO))

import host_config as hc  # noqa: E402
import t2s_k1_ollama as k1  # noqa: E402

MODEL_TAG = "llama3.1:8b"
NUM_CTX = 8192
TARGET_FILLER_TOKENS = NUM_CTX // 2  # "half context" -- deliberately well short of overflow
CHARS_PER_TOKEN_ESTIMATE = 4  # this repo's own established filler estimate; real token count is confirmed
                              # by the server's own prompt_eval_count, never assumed
RULE_CODE = "RULE-7F3Q"
MARKERS = {"start": "MARKER-ALPHA-1829", "middle": "MARKER-BRAVO-5502", "end": "MARKER-CHARLIE-9164"}


def utc_iso():
    return datetime.now(timezone.utc).isoformat()


def emit(out_path, row):
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def build_prompt():
    """Filler text with the three markers at roughly the start, middle, and end of the filler span (not of
    the whole prompt -- the instruction block that follows the filler is not part of the measured span)."""
    unit = "The warehouse inventory system logs each pallet movement with a timestamp and operator id. "
    filler_len_chars = TARGET_FILLER_TOKENS * CHARS_PER_TOKEN_ESTIMATE
    body = (unit * (filler_len_chars // len(unit) + 1))[:filler_len_chars]
    third = len(body) // 3
    # splice each marker into its own third of the filler, near that third's own midpoint, so "start"
    # really is early, "middle" really is in the middle, "end" really is late
    body = (body[:third // 2] + f" [{MARKERS['start']}] " + body[third // 2:third]
           + body[third:third + third // 2] + f" [{MARKERS['middle']}] " + body[third + third // 2:2 * third]
           + body[2 * third:2 * third + third // 2] + f" [{MARKERS['end']}] " + body[2 * third + third // 2:])
    system = (f"System rule, applies to every answer this session: end your answer with the exact code "
             f"{RULE_CODE}, on its own line.")
    instruction = ("\n\nAbove is a warehouse log excerpt. Three bracketed marker codes are embedded "
                   "somewhere in it, in the form [MARKER-...]. List all marker codes you can find, "
                   "verbatim, each on its own line.")
    return system, body + instruction


def run_one_rep(rep, out_path, log):
    system, user = build_prompt()
    hc.start_ollama_server()
    row = {"record": "half_context_probe_result", "rep": rep, "model_id": MODEL_TAG, "num_ctx": NUM_CTX,
          "target_filler_tokens": TARGET_FILLER_TOKENS, "ts_utc": utc_iso()}
    try:
        ready = hc.wait_for_ollama_ready(timeout_s=60)
        if not ready:
            hc.stop_ollama_server()
            hc.start_ollama_server()
            ready = hc.wait_for_ollama_ready(timeout_s=60)
        if not ready:
            row.update({"started": False, "error": "ollama did not become ready"})
            emit(out_path, row)
            return row
        ollama = k1.OllamaClient()
        t0 = time.monotonic()
        resp = ollama.chat(MODEL_TAG, "", num_ctx=NUM_CTX, max_tokens=256, think=False,
                           messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                           keep_alive="2m", timeout=180)
        dt = time.monotonic() - t0
        output = resp.get("message") or ""
        survived = {name: (marker in output) for name, marker in MARKERS.items()}
        rule_survived = RULE_CODE in output
        row.update({"started": True, "http_status": resp.get("status"), "chat_outcome": resp.get("outcome"),
                   "latency_s": dt, "prompt_eval_count": resp.get("prompt_eval_count"),
                   "output_text": output[:1000], "marker_start_survived": survived["start"],
                   "marker_middle_survived": survived["middle"], "marker_end_survived": survived["end"],
                   "rule_survived": rule_survived})
    except Exception as e:
        row.update({"started": False, "error": f"driver exception: {e!r}"[:400]})
    finally:
        hc.stop_ollama_server()
    emit(out_path, row)
    log(f"rep {rep}: start={row.get('marker_start_survived')} middle={row.get('marker_middle_survived')} "
        f"end={row.get('marker_end_survived')} rule={row.get('rule_survived')} "
        f"prompt_tokens={row.get('prompt_eval_count')}")
    return row


def already_done_reps(out_path):
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
        if r.get("record") == "half_context_probe_result":
            done.add(r["rep"])
    return done


def run(out_path, reps, log=print):
    done = already_done_reps(out_path)
    for rep in range(reps):
        if rep in done:
            continue
        emit(out_path, {"record": "heartbeat", "rep": rep, "ts_utc": utc_iso()})
        run_one_rep(rep, out_path, log)
    rows = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines()
           if l.strip() and json.loads(l).get("record") == "half_context_probe_result"]
    n = len(rows)
    for name in ("start", "middle", "end"):
        survived = sum(1 for r in rows if r.get(f"marker_{name}_survived"))
        log(f"marker {name}: {survived}/{n} survived")
    rule_ok = sum(1 for r in rows if r.get("rule_survived"))
    log(f"rule: {rule_ok}/{n} survived")
    emit(out_path, {"record": "run_end", "ts_utc": utc_iso()})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args(argv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"logging to {out_path}")

    note = "completed"
    try:
        run(out_path, args.reps)
    except Exception as e:
        import traceback
        note = f"stopped: {e!r}"[:400]
        print(note)
        traceback.print_exc()
    finally:
        try:
            import t2s_queue as tq
            tq.advance(note)
        except Exception as e:
            print(f"queue advance failed: {e!r}")


if __name__ == "__main__":
    main()
