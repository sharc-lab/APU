"""Paired W-3 int4 vs int8 BFCL quality analysis (no measurement).

McNemar on per-turn (turns present in both arms) and trajectory (n=200),
entry-level trajectory diffs, failure-bucket side-by-side, force_terminated
cross-arm note, independence-model residual at n=200.

Writes derived/bfcl_feasibility/w3_weight_quality/paired_analysis_<int4>_<int8>.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scipy.stats import binomtest, chi2

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "derived" / "bfcl_feasibility" / "w3_weight_quality"

INT4_DEFAULT = "6225d6e1-4e0a-41c9-90bb-695ecc5fbe0a"
INT8_DEFAULT = "1d8db970-4c18-4bcf-824d-d9c141b6eb22"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_arm(session_id: str) -> dict[str, Any]:
    d = BASE / session_id
    summary = _read(d / "summary.json")
    plan = _read(d / "plan.json") if (d / "plan.json").is_file() else {}
    entries = {str(e["id"]): e for e in summary.get("per_entry") or []}
    return {
        "session_id": session_id,
        "dir": d,
        "summary": summary,
        "plan": plan,
        "entries": entries,
        "model_spec": summary.get("model_spec") or plan.get("model_spec"),
        "ir_sha256": summary.get("ir_sha256") or plan.get("ir_sha256"),
        "traj": summary.get("accuracy_trajectory") or {},
        "per_turn": summary.get("accuracy_per_turn_f2") or {},
    }


def _mcnemar(b: int, c: int) -> dict[str, Any]:
    """Exact McNemar on discordant counts b, c (two-sided binomial).

    b = int4 pass / int8 fail
    c = int4 fail / int8 pass
    Statistic: continuity-corrected chi-square (|b-c|-1)^2 / (b+c) when b+c>0.
    OR = b/c with exact CI from Clopper-Pearson on p=b/(b+c) transformed to OR=p/(1-p).
    """
    n_disc = b + c
    if n_disc == 0:
        return {
            "b": b,
            "c": c,
            "n_discordant": 0,
            "statistic_chi2_cc": None,
            "p_exact": 1.0,
            "odds_ratio": None,
            "odds_ratio_ci95": None,
            "note": "no discordant pairs",
        }
    # Continuity-corrected McNemar chi-square (Edwards 1948)
    stat = ((abs(b - c) - 1) ** 2) / n_disc if n_disc else None
    # Exact two-sided: binom under H0 p=0.5 on min(b,c) of n_disc
    p_exact = float(binomtest(min(b, c), n_disc, 0.5, alternative="two-sided").pvalue)
    if c == 0:
        or_point: float | None = math.inf if b > 0 else None
        or_ci: list[float | None] | None = None
    elif b == 0:
        or_point = 0.0
        or_ci = None
    else:
        or_point = b / c
        # Clopper-Pearson on proportion p = b/(b+c); OR = p/(1-p)
        bt = binomtest(b, n_disc, 0.5)
        lo_p, hi_p = bt.proportion_ci(confidence_level=0.95, method="exact")

        def _or(p: float) -> float | None:
            if p <= 0.0:
                return 0.0
            if p >= 1.0:
                return math.inf
            return p / (1.0 - p)

        or_ci = [_or(float(lo_p)), _or(float(hi_p))]
    return {
        "b": b,
        "c": c,
        "n_discordant": n_disc,
        "statistic_chi2_cc": stat,
        "statistic_chi2_cc_note": "(|b-c|-1)^2/(b+c); reference only — p from exact binomial",
        "p_exact": p_exact,
        "odds_ratio": or_point,
        "odds_ratio_ci95": or_ci,
        "odds_ratio_definition": "b/c = (int4-pass&int8-fail)/(int4-fail&int8-pass)",
        "chi2_p_approx": float(chi2.sf(stat, 1)) if stat is not None else None,
    }


def _independence(per_turn: dict[str, Any], traj: dict[str, Any], n_entries: int) -> dict[str, Any]:
    """README_feasibility independence: predicted traj = p_turn ^ mean_turns_per_entry."""
    correct = int(per_turn.get("correct") or 0)
    n_turns = int(per_turn.get("n") or 0)
    p = (correct / n_turns) if n_turns else float("nan")
    mean_turns = (n_turns / n_entries) if n_entries else float("nan")
    predicted = p**mean_turns if n_turns and n_entries else float("nan")
    observed = float(traj.get("accuracy") if traj.get("accuracy") is not None else float("nan"))
    if traj.get("correct") is not None and traj.get("n"):
        observed = int(traj["correct"]) / int(traj["n"])
    residual = observed - predicted
    return {
        "per_turn_correct": correct,
        "per_turn_n": n_turns,
        "per_turn_accuracy": p,
        "n_entries": n_entries,
        "mean_turns_per_entry": mean_turns,
        "predicted_trajectory": predicted,
        "observed_trajectory": observed,
        "observed_trajectory_fraction": f"{traj.get('correct')}/{traj.get('n')}",
        "residual_obs_minus_pred": residual,
        "residual_ratio_obs_over_pred": (observed / predicted) if predicted else None,
        "model": "p_turn ** mean_turns_per_entry (README_feasibility independence claim)",
    }


def analyze(int4_id: str, int8_id: str) -> dict[str, Any]:
    a4 = _load_arm(int4_id)
    a8 = _load_arm(int8_id)
    ids4 = set(a4["entries"])
    ids8 = set(a8["entries"])
    if ids4 != ids8:
        raise SystemExit(
            f"entry id sets differ: only4={sorted(ids4-ids8)[:5]} only8={sorted(ids8-ids4)[:5]}"
        )
    ids = sorted(ids4, key=lambda x: int(x.rsplit("_", 1)[-1]))

    # --- per-turn McNemar: turns present in BOTH ---
    both_a = both_b = both_c = both_d = 0
    paired_turns: list[dict[str, Any]] = []
    for eid in ids:
        turns4 = {int(t["turn"]): t for t in (a4["entries"][eid].get("per_turn") or [])}
        turns8 = {int(t["turn"]): t for t in (a8["entries"][eid].get("per_turn") or [])}
        for turn in sorted(set(turns4) & set(turns8)):
            p4 = bool(turns4[turn].get("pass"))
            p8 = bool(turns8[turn].get("pass"))
            if p4 and p8:
                both_a += 1
            elif p4 and not p8:
                both_b += 1
            elif (not p4) and p8:
                both_c += 1
            else:
                both_d += 1
            paired_turns.append({"id": eid, "turn": turn, "int4_pass": p4, "int8_pass": p8})

    mcn_turn = _mcnemar(both_b, both_c)
    mcn_turn["contingency"] = {
        "a_both_pass": both_a,
        "b_int4_pass_int8_fail": both_b,
        "c_int4_fail_int8_pass": both_c,
        "d_both_fail": both_d,
        "n_paired_turns": both_a + both_b + both_c + both_d,
    }

    # --- trajectory McNemar ---
    ta = tb = tc = td = 0
    traj_rows: list[dict[str, Any]] = []
    int4_only_pass: list[str] = []
    int8_only_pass: list[str] = []
    for eid in ids:
        t4 = bool(a4["entries"][eid].get("trajectory_pass"))
        t8 = bool(a8["entries"][eid].get("trajectory_pass"))
        if t4 and t8:
            ta += 1
        elif t4 and not t8:
            tb += 1
            int4_only_pass.append(eid)
        elif (not t4) and t8:
            tc += 1
            int8_only_pass.append(eid)
        else:
            td += 1
        traj_rows.append({"id": eid, "int4_pass": t4, "int8_pass": t8})

    mcn_traj = _mcnemar(tb, tc)
    mcn_traj["contingency"] = {
        "a_both_pass": ta,
        "b_int4_pass_int8_fail": tb,
        "c_int4_fail_int8_pass": tc,
        "d_both_fail": td,
        "n": 200,
    }

    # --- failure buckets ---
    buckets4 = Counter(str(a4["entries"][e].get("failure_bucket") or "PASS") for e in ids)
    buckets8 = Counter(str(a8["entries"][e].get("failure_bucket") or "PASS") for e in ids)
    # trajectory pass should not use failure_bucket as PASS wrongly — check
    for e in ids:
        if a4["entries"][e].get("trajectory_pass"):
            buckets4["PASS_trajectory"] += 0  # ensure key visibility via rewrite

    # Rebuild: use PASS when trajectory_pass else failure_bucket
    def _bucket(arm: dict[str, Any], eid: str) -> str:
        row = arm["entries"][eid]
        if row.get("trajectory_pass"):
            return "(trajectory_pass)"
        return str(row.get("failure_bucket") or "unknown")

    buckets4 = Counter(_bucket(a4, e) for e in ids)
    buckets8 = Counter(_bucket(a8, e) for e in ids)
    all_buckets = sorted(set(buckets4) | set(buckets8))
    bucket_table = [
        {
            "bucket": b,
            "int4": int(buckets4.get(b, 0)),
            "int8": int(buckets8.get(b, 0)),
        }
        for b in all_buckets
    ]

    # --- force_terminated cross-arm ---
    force4 = [
        eid
        for eid in ids
        if a4["entries"][eid].get("force_terminated")
        or str(a4["entries"][eid].get("failure_bucket") or "") == "multi_turn:force_terminated"
    ]
    force_cross = []
    for eid in force4:
        r4 = a4["entries"][eid]
        r8 = a8["entries"][eid]
        force_cross.append(
            {
                "id": eid,
                "in_first_20": int(eid.rsplit("_", 1)[-1]) < 20,
                "int4": {
                    "failure_bucket": r4.get("failure_bucket"),
                    "force_terminated": r4.get("force_terminated"),
                    "trajectory_pass": r4.get("trajectory_pass"),
                    "n_completed_turns": r4.get("n_completed_turns"),
                    "n_user_turns": r4.get("n_user_turns"),
                },
                "int8": {
                    "failure_bucket": r8.get("failure_bucket"),
                    "force_terminated": r8.get("force_terminated"),
                    "trajectory_pass": r8.get("trajectory_pass"),
                    "n_completed_turns": r8.get("n_completed_turns"),
                    "n_user_turns": r8.get("n_user_turns"),
                },
            }
        )
    n_force_first20 = sum(1 for x in force_cross if x["in_first_20"])
    rate_first20 = n_force_first20 / 20.0
    rate_rest = (len(force4) - n_force_first20) / 180.0 if len(force4) > n_force_first20 else 0.0
    overrep = (rate_first20 / rate_rest) if rate_rest > 0 else None

    # --- independence ---
    ind4 = _independence(a4["per_turn"], a4["traj"], 200)
    ind8 = _independence(a8["per_turn"], a8["traj"], 200)

    return {
        "kind": "w3_paired_quality_analysis",
        "analyzed_utc": _utc_now(),
        "arms": {
            "int4": {
                "session_id": int4_id,
                "model_spec": a4["model_spec"],
                "ir_sha256": a4["ir_sha256"],
                "trajectory": a4["traj"],
                "per_turn_f2": a4["per_turn"],
                "n_force_terminated": a4["summary"].get("n_force_terminated"),
            },
            "int8": {
                "session_id": int8_id,
                "model_spec": a8["model_spec"],
                "ir_sha256": a8["ir_sha256"],
                "trajectory": a8["traj"],
                "per_turn_f2": a8["per_turn"],
                "n_force_terminated": a8["summary"].get("n_force_terminated"),
            },
        },
        "mcnemar_per_turn": mcn_turn,
        "mcnemar_trajectory": mcn_traj,
        "entry_trajectory_diffs": {
            "int4_pass_int8_fail": int4_only_pass,
            "int8_pass_int4_fail": int8_only_pass,
            "n_int4_only": len(int4_only_pass),
            "n_int8_only": len(int8_only_pass),
            "n_both_pass": ta,
            "n_both_fail": td,
        },
        "failure_buckets": {
            "int4_session": int4_id,
            "int8_session": int8_id,
            "n_entries": 200,
            "table": bucket_table,
        },
        "force_terminated_cross": {
            "int4_force_terminated_ids": force4,
            "n": len(force4),
            "n_in_first_20": n_force_first20,
            "rate_first_20": rate_first20,
            "rate_entries_20_199": rate_rest,
            "overrepresentation_first20_vs_rest": overrep,
            "note": (
                f"{n_force_first20} of {len(force4)} int4 force_terminated fell in the "
                "first 20 entries; pilot overrepresented that failure mode "
                f"~{overrep:.0f}x vs the remaining 180"
                if overrep
                else "force_terminated cross-arm detail"
            ),
            "rows": force_cross,
        },
        "independence_model_n200": {
            "readme_claim": (
                "docs/README_feasibility.md: independence model reproduces trajectory "
                "exactly (product of per-turn competences). At n=20 it held; recompute "
                "at n=200 as p_turn ** mean_turns_per_entry."
            ),
            "int4": ind4,
            "int8": ind8,
        },
        "generation_note": "greedy do_sample=False; one-entry diffs are not sampling noise",
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--int4", default=INT4_DEFAULT)
    p.add_argument("--int8", default=INT8_DEFAULT)
    args = p.parse_args(argv)
    out = analyze(args.int4, args.int8)
    out_path = BASE / f"paired_analysis_{args.int4}_{args.int8}.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": True,
                "path": str(out_path),
                **{
                    "mcnemar_per_turn": {
                        "b": out["mcnemar_per_turn"]["b"],
                        "c": out["mcnemar_per_turn"]["c"],
                        "p": out["mcnemar_per_turn"]["p_exact"],
                        "or": out["mcnemar_per_turn"]["odds_ratio"],
                    },
                    "mcnemar_traj": {
                        "b": out["mcnemar_trajectory"]["b"],
                        "c": out["mcnemar_trajectory"]["c"],
                        "p": out["mcnemar_trajectory"]["p_exact"],
                        "or": out["mcnemar_trajectory"]["odds_ratio"],
                    },
                    "independence": {
                        "int4_resid": out["independence_model_n200"]["int4"][
                            "residual_obs_minus_pred"
                        ],
                        "int8_resid": out["independence_model_n200"]["int8"][
                            "residual_obs_minus_pred"
                        ],
                    },
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
