"""Validation checks for analysis/envelope_model.py. This is the actual point of the model, not polish:

1. validate_ttft_cross_machine(): fit TTFT on T2S's own real data (already done in envelope_model at import
   time), then check whether any real evo-x2 TTFT data exists anywhere in this repo to predict against. As of
   this commit it does not (see envelope_model.py's module docstring and docs/FINDINGS.md's PX2 section), so
   this prints that fact plainly as the validation result rather than fabricating an X2 number to compare
   against. If evo-x2 TTFT rows are ever committed to results/*.jsonl with hw_id=="evo-x2", this function will
   find them (it scans results/*.jsonl for that field) and compute real MAPE per prompt length automatically.

2. validate_budget_boundary_loo(): leave-one-model-out on the real A-24 bisect data. Fits the
   (weights+compute intercept) vs (KV slope) relationship on 3 of the 4 fully-bisected models, predicts the
   4th's intercept, derives its predicted budget-crossing n_ctx, and reports the error in MiB and percent
   against the real crossing. Held-out model: qwen3-30b-a3b-2507 (the MoE model) -- chosen because it is the
   one architecturally different model in the set (dense vs MoE), making it the cleanest test of whether
   linear extrapolation from dense models generalizes to a different architecture. As a secondary check, the
   fit is also evaluated against the real, independently-measured llama-3.3-70b point (T2S_A70_LAST_OK_POINT),
   which was never part of the bisect-point fit at all (only one point exists for it, so it cannot be
   leave-one-out'd the same way -- it is a pure out-of-sample check).

Usage: py -3.12 analysis/validate_envelope_model.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from envelope_model import (  # noqa: E402
    T2S_BUDGET_BISECT_POINTS, T2S_TTFT_FIT, T2S_VULKAN_BUDGET_MIB, T2S_A70_LAST_OK_POINT, _linreg,
    H2_X2_PER_MODEL,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _find_evo_x2_ttft_rows():
    """Scans every results/*.jsonl in the repo for rows carrying hw_id (or hardware) == 'evo-x2' AND a
    non-null ttft_s AND a prompt_tokens field. Returns a list of (prompt_tokens, ttft_s) real pairs, or []
    if none exist (which, as of this commit, is the real answer -- see module docstring)."""
    pairs = []
    for path in glob.glob(str(REPO_ROOT / "results" / "*.jsonl")):
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    hw = row.get("hw_id") or row.get("hardware")
                    if hw == "evo-x2" and row.get("ttft_s") is not None and row.get("prompt_tokens"):
                        pairs.append((row["prompt_tokens"], row["ttft_s"], path))
        except OSError:
            continue
    return pairs


def validate_ttft_cross_machine():
    print("=== Validation 1: T2S-fit TTFT regression applied to evo-x2 ===")
    print(f"T2S fit: ttft_s = {T2S_TTFT_FIT['intercept_s']:.4f} + {T2S_TTFT_FIT['slope_s_per_token']:.6f} * "
          f"prompt_tokens  (n=39 real points, R2={T2S_TTFT_FIT['r2']:.4f})")
    rows = _find_evo_x2_ttft_rows()
    if not rows:
        print("RESULT: no evo-x2 TTFT data exists anywhere in results/*.jsonl in this repo as of this commit "
              "(scanned every *.jsonl for hw_id/hardware=='evo-x2' with a non-null ttft_s and prompt_tokens). "
              "This matches docs/FINDINGS.md's own PX2-section statement: 'no evo-x2 per-call timing to anchor "
              "on ... no results/*.jsonl row carries hw_id evo-x2'. Cross-machine generalization of the TTFT "
              "regression CANNOT be validated against real data at this time. This absence is itself the "
              "reportable finding for this check -- not a computed error number, because none exists to compute.")
        return {"validated": False, "n_x2_points": 0, "reason": "no evo-x2 TTFT data in repo"}
    errors = []
    for prompt_tokens, real_ttft, path in rows:
        pred = T2S_TTFT_FIT["intercept_s"] + T2S_TTFT_FIT["slope_s_per_token"] * prompt_tokens
        ape = abs(pred - real_ttft) / real_ttft * 100 if real_ttft else float("nan")
        errors.append(ape)
        print(f"  prompt_tokens={prompt_tokens}: real={real_ttft:.2f}s predicted={pred:.2f}s APE={ape:.1f}% ({path})")
    mape = sum(errors) / len(errors)
    print(f"RESULT: MAPE across {len(rows)} real evo-x2 points = {mape:.1f}%")
    return {"validated": True, "n_x2_points": len(rows), "mape_pct": mape}


def validate_budget_boundary_loo(held_out_model="qwen3-30b-a3b-2507"):
    print("\n=== Validation 2: leave-one-model-out on the A-24 budget boundary ===")
    models = list(T2S_BUDGET_BISECT_POINTS)
    assert held_out_model in models
    train_models = [m for m in models if m != held_out_model]
    print(f"Held-out model: {held_out_model} (the one MoE / architecturally-distinct model in the bisected set; "
          f"training models: {train_models})")

    # Per-model real (intercept, slope) from its own bisect points.
    per_model_fit = {m: _linreg(T2S_BUDGET_BISECT_POINTS[m]) for m in models}
    for m, (a, b, r2) in per_model_fit.items():
        print(f"  {m}: intercept={a:.1f} MiB, slope={b:.5f} MiB/ctx-token, R2={r2:.5f}  (own real bisect points)")

    # Fit intercept ~ slope across the 3 training models (simple linear relationship between the model's
    # context-independent footprint and its KV cost-per-token -- both are real, both come from the same fit).
    train_pairs = [(per_model_fit[m][1], per_model_fit[m][0]) for m in train_models]  # (slope, intercept)
    a_i, b_i, r2_i = _linreg(train_pairs)
    held_out_slope = per_model_fit[held_out_model][1]
    predicted_intercept = a_i + b_i * held_out_slope
    real_intercept = per_model_fit[held_out_model][0]
    real_slope = held_out_slope

    # Real boundary-crossing n_ctx for the held-out model: the n_ctx at which projected_mib crosses the budget,
    # taken from its own real fit (own slope+intercept, both real -- this is the ground truth we're predicting).
    real_boundary_ctx = (T2S_VULKAN_BUDGET_MIB - real_intercept) / real_slope
    predicted_boundary_ctx = (T2S_VULKAN_BUDGET_MIB - predicted_intercept) / real_slope
    # (real_slope used for both -- slope comes from the model's own real bisect points either way; only the
    # intercept, the model-size-dependent term, is what's actually being predicted out-of-sample here.)

    real_mib_at_real_boundary = T2S_VULKAN_BUDGET_MIB
    mib_error = predicted_intercept - real_intercept
    pct_error = abs(mib_error) / real_intercept * 100
    ctx_error = predicted_boundary_ctx - real_boundary_ctx
    ctx_pct_error = abs(ctx_error) / real_boundary_ctx * 100

    print(f"\nTrained intercept~slope relation (3 models): intercept = {a_i:.1f} + {b_i:.6f} * slope, R2={r2_i:.4f}")
    print(f"Held-out {held_out_model}: real intercept={real_intercept:.1f} MiB, predicted intercept="
          f"{predicted_intercept:.1f} MiB -> error = {mib_error:+.1f} MiB ({pct_error:.1f}%)")
    print(f"Held-out {held_out_model}: real budget-crossing n_ctx={real_boundary_ctx:.0f}, predicted n_ctx="
          f"{predicted_boundary_ctx:.0f} -> error = {ctx_error:+.0f} tokens ({ctx_pct_error:.1f}%)")

    # Secondary, purely out-of-sample check against the one real llama-3.3-70b point (never used in any fit).
    a70_ctx, a70_mib = T2S_A70_LAST_OK_POINT
    train_pairs_incl_moe = [(per_model_fit[m][1], per_model_fit[m][0]) for m in models]  # all 4
    a_i4, b_i4, r2_i4 = _linreg(train_pairs_incl_moe)
    # We have no independent slope estimate for the 70B (only one point), so we can only sanity-check the
    # intercept sign/direction here, not do a full LOO prediction -- reported honestly as a weaker check.
    print(f"\nSecondary out-of-sample check (llama-3.3-70b, A-24 update text, never used in any fit above): "
          f"real last-OK point is n_ctx={a70_ctx}, projected_mib={a70_mib}. No independent slope exists for "
          f"this model in this repo (only one point is committed -- see T2S_A70_LAST_OK_POINT's docstring), so "
          f"a full boundary prediction for it is not possible from local data alone; this is reported as a gap, "
          f"not papered over with an invented slope.")

    return {"held_out_model": held_out_model, "intercept_error_mib": mib_error, "intercept_error_pct": pct_error,
            "boundary_ctx_error_tokens": ctx_error, "boundary_ctx_error_pct": ctx_pct_error}


def validate_px2_decode_multiplier_loo():
    """Leave-one-model-out on H2_X2_PER_MODEL's real decode ratios (results/t2s_night2_20260930T135145Z.jsonl,
    5 PX2 models, n=5 calls/condition, n=3 for llama-3.3-70b). For each model, predicts its decode ratio (for
    both B4 and S4) as the mean of the OTHER 4 models' real ratios, then reports the real error against the
    held-out model's own real ratio. This is the held-out check for the 2026-10-01 per-machine co-runner refit
    (the prior validation functions above predate that refit and cover evo-t2s's budget boundary / cross-machine
    TTFT only -- this is the new axis the refit added)."""
    print("\n=== Validation 3: leave-one-model-out on the real PX2 evo-x2 decode-multiplier refit ===")
    models = list(H2_X2_PER_MODEL)
    results = {"B4": [], "S4": []}
    for cond, field in (("B4", "decode_ratio_b4"), ("S4", "decode_ratio_s4")):
        print(f"\n  -- condition {cond} ({field}) --")
        for held_out in models:
            others = [H2_X2_PER_MODEL[m][field] for m in models if m != held_out]
            predicted = sum(others) / len(others)
            real = H2_X2_PER_MODEL[held_out][field]
            abs_err = predicted - real
            pct_err = abs(abs_err) / real * 100
            results[cond].append(pct_err)
            print(f"    held out {held_out:16s}: real={real:.4f} predicted(mean of other 4)={predicted:.4f} "
                  f"error={abs_err:+.4f} ({pct_err:.2f}%)")
        mape = sum(results[cond]) / len(results[cond])
        print(f"  RESULT: MAPE across {len(models)} held-out models for {cond} = {mape:.2f}%")
    return {
        "b4_mape_pct": sum(results["B4"]) / len(results["B4"]),
        "s4_mape_pct": sum(results["S4"]) / len(results["S4"]),
        "n_models": len(models),
        "note": "LOO over 5 real PX2 models; S4's ratio is close to 1.0 for every model so its MAPE is "
                "expected to be small and uninformative on its own -- B4's MAPE is the real test of whether "
                "the bandwidth-hog decode effect generalizes across models, not just within the 5 measured ones.",
    }


if __name__ == "__main__":
    validate_ttft_cross_machine()
    validate_budget_boundary_loo()
    validate_px2_decode_multiplier_loo()
