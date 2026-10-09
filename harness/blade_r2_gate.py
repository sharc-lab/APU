"""Blade night 1 gate (operator 2026-10-08): reads the R2 validation rerun and decides which branch the night takes.

  branch "real"       the negative control (BLADE_NEG_ARM at 32768, call-2 mode off) has 0 canary misses over all of
                      its sessions: the R2 real tiers and the mitigation for llama3.1:8b run next (each still applies
                      its own --require-validation-gates preflight; rules in use come from the same validation file,
                      so rules 2 and 5 drop out as on evo-x2 when they fail the baseline).
  branch "mechanism"  any canary miss: real and mitigation are skipped ("skipped: negative-control canary misses N")
                      and the mechanism run at the 4096 tier runs instead; the Blade recall result is recorded here as
                      an open finding.

No model, no server: it reads one jsonl and writes one json (--out). Scoring is x2_r2_agent.evaluate_gates with the
Blade negative-control arm, unchanged. harness/blade_queue.run_night reads the decision file after this job exits 0
and stores it in the queue state, so the conditional jobs are resumable: a rerun of the night reads the same decision.

Exit codes: 0 decision written (either branch); 2 no decision (validation unfinished, or fewer negative-control
sessions than --neg-sessions); nothing is written then, and the dependent jobs stay pending.

Usage:
  py -3.12 harness/blade_r2_gate.py --validation results/blade_r2_validation_v2.jsonl --out results/blade_r2_gate_v2.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))

import blade_common as bc  # noqa: E402
import blade_r2  # noqa: E402
import x2_r2_agent as agent  # noqa: E402

EXIT_OK, EXIT_ERR = 0, 2
BRANCH_REAL, BRANCH_MECHANISM = "real", "mechanism"


def neg_sessions_of(rows: list[dict], model: str) -> list[dict]:
    arm = blade_r2.BLADE_NEG_ARM + agent.MODE_SUFFIX[blade_r2.CALL2]
    ss = [s for s in agent.completed_sessions(rows)
          if s["mode"] == "validation" and s["model_id"] == model and s["arm_id"] == arm]
    return sorted(ss, key=lambda s: s["seed"])


def decide(rows: list[dict], model: str = "llama3.1:8b", neg_sessions: int = 5, validation: str = "") -> dict:
    """{"ok": bool, ...}. ok False = no decision (reasons). ok True: "branch", per-session misses, the gate report's
    controls and rules in use, and for the mechanism branch "skipped_reason" and "open_finding"."""
    reasons = []
    if not any(r.get("record") == "run_end" for r in rows):
        reasons.append("validation file has no run_end record (validation did not finish)")
    ss = neg_sessions_of(rows, model)
    if len(ss) < neg_sessions:
        reasons.append(f"{model}: {len(ss)} of {neg_sessions} negative-control sessions completed")
    if reasons:
        return {"ok": False, "reasons": reasons, "validation": validation, "model": model}
    rep = agent.evaluate_gates(rows, call2_tools=blade_r2.CALL2, neg_arm=blade_r2.BLADE_NEG_ARM)
    g = rep["gates"]
    per = [{"seed": s["seed"], "canary_checks": s["canary_checks"], "canary_sys_misses": s["canary_sys_misses"],
            "canary_hist_misses": s["canary_hist_misses"],
            "canary_misses": s["canary_sys_misses"] + s["canary_hist_misses"]} for s in ss]
    misses = sum(p["canary_misses"] for p in per)
    neg = g["negative_control"].get(model) or {}
    if neg.get("canary_misses") is not None and neg["canary_misses"] != misses:
        raise AssertionError(f"per-session misses {misses} != evaluate_gates {neg['canary_misses']}")
    table = rep["baseline_table"].get(model) or {}
    branch = BRANCH_REAL if misses == 0 else BRANCH_MECHANISM
    out = {"ok": True, "record": "blade_r2_gate", "validation": validation, "model": model,
           "neg_arm": blade_r2.BLADE_NEG_ARM, "num_ctx": blade_r2.BLADE_VALIDATION_CTX, "call2_mode": blade_r2.CALL2,
           "n_neg_sessions": len(per), "neg_sessions_required": neg_sessions, "per_session": per,
           "canary_misses": misses, "canary_checks": sum(p["canary_checks"] for p in per), "branch": branch,
           "rules_in_use": (g["baseline"].get(model) or {}).get("rules_in_use"),
           "failing_rules": (g["baseline"].get(model) or {}).get("failing_rules"),
           "positive_control": g["positive_control"].get(model), "recall": table.get("recall"),
           "skipped_reason": None, "open_finding": None, "decided_utc": bc.utc_iso()}
    if branch == BRANCH_MECHANISM:
        out["skipped_reason"] = f"skipped: negative-control canary misses {misses}"
        rec = table.get("recall") or {}
        out["open_finding"] = (f"Blade {model} negative control at num_ctx {blade_r2.BLADE_VALIDATION_CTX}: {misses} "
                               f"canary misses in {len(per)} sessions (recall rate {rec.get('rate')} over "
                               f"{rec.get('n')} turns), with no truncation expected at that window; real and "
                               "mitigation not run, mechanism run at 4096 instead. Open: why the canaries are lost "
                               "on the Blade.")
    else:
        out["skipped_reason"] = "skipped: negative control clean (0 canary misses); real and mitigation run instead"
    return out


def synthetic_validation_rows(model="llama3.1:8b", neg_sessions=5, misses=0, finished=True) -> list[dict]:
    """Stub and test fixture: the rows decide() reads (run_start, r2a_session per negative-control seed and the
    positive control, run_end). `misses` canary misses are put on the last negative-control session."""
    arm = blade_r2.BLADE_NEG_ARM + agent.MODE_SUFFIX[blade_r2.CALL2]
    pos = agent.POS_ARM + agent.MODE_SUFFIX[blade_r2.CALL2]
    rows = [{"record": "run_start", "mode": "validation", "call2_mode": blade_r2.CALL2}]
    base = {"record": "r2a_session", "mode": "validation", "model_id": model, "loaded_context_values": [32768],
            "max_transcript_tokens_calibrated": 15000, "canary_sys_misses": 0, "canary_hist_misses": 0,
            "truncation_detected_turn": None, "first_over_window_turn": None}
    for i, seed in enumerate(blade_r2.neg_seeds(neg_sessions)):
        rows.append({**base, "arm_id": arm, "seed": seed, "attempt_id": f"neg{seed}", "canary_checks": 2,
                     "canary_hist_misses": misses if i == neg_sessions - 1 else 0})
    rows.append({**base, "arm_id": pos, "seed": agent.SEEDS[0], "attempt_id": "pos", "canary_checks": 3,
                 "loaded_context_values": [8192], "truncation_detected_turn": 10, "first_over_window_turn": 6})
    if finished:
        rows.append({"record": "run_end", "note": "completed"})
    return rows


def build_arg_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validation", required=True, help="the validation jsonl (blade_r2.py --mode validation)")
    ap.add_argument("--out", required=True, help="decision json, blade_ prefixed")
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--neg-sessions", type=int, default=5)
    return ap


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    out = bc.out_path_ok(args.out, False)
    if not Path(args.validation).name.startswith("blade_"):
        raise SystemExit(f"--validation must be a Blade file (blade_*): {args.validation}")
    rows = agent.read_rows(Path(args.validation))
    d = decide(rows, args.model, args.neg_sessions, validation=args.validation)
    if not d["ok"]:
        print(f"[{bc.utc_iso()}] no decision: {d['reasons']}", flush=True)
        return EXIT_ERR
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1, default=str), encoding="utf-8")
    tmp.replace(out)
    print(f"[{bc.utc_iso()}] branch {d['branch']}: canary misses {d['canary_misses']} over {d['n_neg_sessions']} "
          f"sessions -> {out}", flush=True)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
