"""Q0 positive control: every quality_suite task type must score >= 90% on qwen3-8b at 2k tokens with the full
prompt intact, and clearly lower with 50% of the prompt removed from the front. Required before K1 or K2 use Q0 for
anything real (a suite that can't pass its own control isn't trustworthy as a quality signal).

Usage (deployed to C:\\apu\\ovn on either machine):
  python q0_positive_control.py --expect-blobs expected_blobs.json --model qwen3-8b
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import socket
import sys
import time
from pathlib import Path

DEPLOY = Path(__file__).resolve().parent
sys.path.insert(0, str(DEPLOY))
import host_config as hc  # noqa: E402
import quality_suite as qs  # noqa: E402
import run_provenance as rp  # noqa: E402
import t2s_lab as L  # noqa: E402
import t2s_overnight as ov  # noqa: E402
from t2s_lab import log, utc_iso  # noqa: E402

TARGET_TOKENS = 2000
SEED = 42
PASS_THRESHOLD = 0.90


def truncate_front(prompt, frac_removed=0.5):
    keep_from = int(len(prompt) * frac_removed)
    return prompt[keep_from:]


def run_control(lab, srv, mi):
    rows = []
    for task_type in qs.TASK_TYPES:
        task = qs.build_task(task_type, TARGET_TOKENS, SEED)
        full = qs.run_task(srv, task, TARGET_TOKENS, SEED)
        full["condition"] = "full"
        rows.append(full)

        trunc_task = dataclasses.replace(task, prompt=truncate_front(task.prompt, 0.5))
        trunc = qs.run_task(srv, trunc_task, TARGET_TOKENS, SEED)
        trunc["condition"] = "truncated_50pct_front"
        rows.append(trunc)

        for r in (full, trunc):
            r["record"] = "q0_control"
            r["model_id"] = mi.model_id
            r["hw_id"] = lab.identity.get("hw_id")
            r["ts_utc"] = utc_iso()
            lab.emit(r)
        log(f"q0 control {task_type}: full={full['score']} truncated={trunc['score']}")
    return rows


def summarize(rows):
    by_type = {}
    for r in rows:
        by_type.setdefault(r["task_type"], {})[r["condition"]] = r["score"]
    summary = {}
    for task_type, conds in by_type.items():
        full, trunc = conds.get("full"), conds.get("truncated_50pct_front")
        full_ok = full is not None and full >= PASS_THRESHOLD
        trunc_lower = trunc is not None and full is not None and trunc < full
        summary[task_type] = {"full_score": full, "truncated_score": trunc, "full_pass": full_ok,
                              "truncated_clearly_lower": trunc_lower, "control_ok": full_ok and trunc_lower}
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect-blobs", required=True)
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--out-dir", default=r"C:\apu\ovn\results")
    args = ap.parse_args()
    host_cfg = hc.require_host(socket.gethostname())
    hc.enforce_or_record_interactive_session(host_cfg)  # raises on evo-t2s if occupied; never raises on evo-x2
    prov = rp.verify_deployed_blobs(DEPLOY, args.expect_blobs)
    import types
    ns = types.SimpleNamespace(smoke=False, deadline_h=2.0, resume=None, stem_prefix="q0_control", only=None,
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
        srv = L.Server(lab, mi, 8192, tag="q0_control")
        lab.resources["server"] = srv
        info = srv.start(timeout=1800)
        ov.start_row(lab, srv, mi, "Q0CTRL", "q0_control_start", info, {})
        if not info.get("ok"):
            raise RuntimeError(f"server failed to start: {info.get('error')}")
        rows = run_control(lab, srv, mi)
        srv.stop()
        lab.resources["server"] = None
        summary = summarize(rows)
        overall_ok = all(s["control_ok"] for s in summary.values())
        lab.emit({"record": "q0_control_summary", "summary": summary, "overall_ok": overall_ok, "ts_utc": utc_iso()})
        log(f"Q0 CONTROL {'PASS' if overall_ok else 'FAIL'}: {json.dumps(summary, indent=1)}")
    except Exception as e:
        note = f"stopped: {e!r}"[:400]
        log(note)
    finally:
        ov.cleanup(lab)
        lab.emit({"record": "run_end", "note": note, "ts_utc": utc_iso()})
        (Path(lab.out_dir) / f"{lab.stem}.DONE").write_text(note + "\n")


if __name__ == "__main__":
    main()
