"""Q0 token-length calibration check, added after the common_words_extraction bug (a 2000-token-intended prompt
actually tokenized to 12,657 tokens, see commit 47ef612): that bug was specific to one task's padding scheme, but the
same class of miscalibration (build_task's char-based length estimate vs. the real tokenizer) could exist, undetected,
in any other task at any target length. This script builds every quality_suite.TASK_TYPES task at every target length
in TARGET_LENGTHS, tokenizes each via a live server's real /tokenize endpoint (not the char-based estimate build_task
itself uses), and reports actual/target for each (task_type, target) cell. No chat call is made -- this only checks
prompt length, so it is much cheaper than a full Q0 run and safe to run before trusting any task at any length.

Usage (deployed to C:\\apu\\ovn on either machine):
  python q0_token_calibration.py --expect-blobs expected_blobs.json --model qwen3-8b
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import types
from pathlib import Path

DEPLOY = Path(__file__).resolve().parent
sys.path.insert(0, str(DEPLOY))
import host_config as hc  # noqa: E402
import quality_suite as qs  # noqa: E402
import run_provenance as rp  # noqa: E402
import t2s_lab as L  # noqa: E402
import t2s_overnight as ov  # noqa: E402
import t2s_queue as tq  # noqa: E402
from t2s_lab import log, utc_iso  # noqa: E402

TARGET_LENGTHS = (2000, 8000, 24000, 96000)
SEED = 42
OK_LOW, OK_HIGH = 0.95, 1.05


def check_all(srv, lab, model_id):
    rows = []
    for task_type in qs.TASK_TYPES:
        for target in TARGET_LENGTHS:
            task = qs.build_task(task_type, target, SEED)
            actual = srv.tokenize(task.prompt)
            ratio = actual / target
            ok = OK_LOW <= ratio <= OK_HIGH
            row = {"record": "q0_token_calibration", "task_type": task_type, "target_tokens": target,
                  "actual_tokens": actual, "ratio": round(ratio, 4), "ok": ok, "model_id": model_id,
                  "hw_id": lab.identity.get("hw_id"), "ts_utc": utc_iso()}
            rows.append(row)
            lab.emit(row)
            log(f"q0 calibration {task_type} target={target}: actual={actual} ratio={ratio:.3f} {'OK' if ok else 'OUT OF RANGE'}")
    return rows


def summarize(rows):
    by_task = {}
    for r in rows:
        by_task.setdefault(r["task_type"], []).append(r)
    return {t: {"all_ok": all(r["ok"] for r in rs), "worst_ratio": max(rs, key=lambda r: abs(r["ratio"] - 1.0))["ratio"],
               "cells": [{"target": r["target_tokens"], "actual": r["actual_tokens"], "ratio": r["ratio"], "ok": r["ok"]} for r in rs]}
            for t, rs in by_task.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect-blobs", required=True)
    ap.add_argument("--model", default="qwen3-8b")
    args = ap.parse_args()
    host_cfg = hc.require_host(socket.gethostname())
    hc.enforce_or_record_interactive_session(host_cfg)
    prov = rp.verify_deployed_blobs(DEPLOY, args.expect_blobs)
    ns = types.SimpleNamespace(smoke=False, deadline_h=2.0, resume=None, stem_prefix="q0_calibration", only=None,
                               reserve_min=0, no_cap_arm=True, max_items=0, gpu_vendor=host_cfg["gpu_vendor"])
    lab = ov.Lab(ns, prov)
    lab.identity["hw_id"] = host_cfg["hw_id"]
    lab.track_console = not host_cfg.get("interactive_guard", True)
    ov.read_downloads(lab)
    fn, hyb, mx, yf = ov.MODEL_FILES[args.model]
    sha = ov.KNOWN_4B_SHA if args.model == "qwen3-4b-2507" else lab.dl_sha.get(fn)
    if sha is None:
        raise SystemExit(f"{args.model} ({fn}) not yet in downloads.jsonl")
    mi = L.ModelInfo(args.model, str(Path(L.MODELS_DIR) / fn), sha, hyb, mx, yf)
    lab.tele.start()
    note = "completed"
    try:
        srv = L.Server(lab, mi, max(TARGET_LENGTHS) + 4096, tag="q0_calibration")
        lab.resources["server"] = srv
        info = srv.start(timeout=1800)
        ov.start_row(lab, srv, mi, "Q0CAL", "q0_calibration_start", info, {})
        if not info.get("ok"):
            raise RuntimeError(f"server failed to start: {info.get('error')}")
        rows = check_all(srv, lab, args.model)
        srv.stop()
        lab.resources["server"] = None
        summary = summarize(rows)
        overall_ok = all(s["all_ok"] for s in summary.values())
        lab.emit({"record": "q0_calibration_summary", "summary": summary, "overall_ok": overall_ok, "ts_utc": utc_iso()})
        log(f"Q0 TOKEN CALIBRATION {'PASS' if overall_ok else 'OUT OF RANGE ON SOME TASKS'}: {json.dumps(summary, indent=1)}")
    except Exception as e:
        note = f"stopped: {e!r}"[:400]
        log(note)
    finally:
        ov.cleanup(lab)
        lab.emit({"record": "run_end", "note": note, "ts_utc": utc_iso()})
        (Path(lab.out_dir) / f"{lab.stem}.DONE").write_text(note + "\n")
        try:
            tq.advance(note)
        except Exception as e:
            log(f"queue advance failed: {e!r}")


if __name__ == "__main__":
    main()
