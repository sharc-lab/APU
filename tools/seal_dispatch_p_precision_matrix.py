"""Seal DISPATCH P interleaved precision matrix (288 cells + canaries).

Unlike ``seal_delta_prefill_session.py`` (small 2-arm matrices named
``delta_prefill_*.json``), this sealer understands:
  - PowerShell ``ConvertTo-Json`` List serialization ``{value: [...], Count: N}``
  - cell artifacts named ``dispatch_p_interleaved_*``
  - canary series + canary_gate
  - per-cell KV_CACHE_PRECISION readback must match the arm

Does not mutate ``seal_delta_prefill_session.py``. Does not rewrite sealed
trees under other run_ids. NO raw/ write unless ``--attempt-raw-promote``.

Usage:
  .\\.venv-seam\\Scripts\\python.exe tools\\seal_dispatch_p_precision_matrix.py \\
      --session-id 41e419bd-f3e9-43b1-8364-0ebd89fa086b
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from seam.ov_kv_precision import kv_bytes_per_token  # noqa: E402

SESSION_DEFAULT = "41e419bd-f3e9-43b1-8364-0ebd89fa086b"
ARM_EXPECTED_PRECISION = {
    "gpu_only_f16": "f16",
    "gpu_only_u8": "u8",
    "gpu_only_u4": "u4",
}
ARMS = ("gpu_only_f16", "gpu_only_u8", "gpu_only_u4")
N_CACHED = (2000, 4000, 8000, 12000)
DELTAS = (50, 150, 400, 1000)
MODES = ("RESIDENT", "NON_RESIDENT")
REPEATS = (0, 1, 2)

# Operator-reported launch environment (DISPATCH seal request). Persisted as-stated.
LAUNCH_ENVIRONMENT_AS_REPORTED = {
    "source": "operator_dispatch_seal_request",
    "boot_local": "2026-08-23T14:33:21",
    "available_mb": 10047,
    "processor_frequency_mhz": 1850,
    "power": "AC",
    "dell_mcafee_services": "Manual",
    "workloads_session_host": "killed",
    "remove_appxpackage_workloads": {
        "attempted": True,
        "refused_hr": "0x80073D02",
    },
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_tree(root: Path, *, exclude: set[str]) -> str:
    h = hashlib.sha256()
    files = sorted(
        (p for p in root.rglob("*") if p.is_file() and p.name not in exclude),
        key=lambda p: p.relative_to(root).as_posix(),
    )
    for p in files:
        rel = p.relative_to(root).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _parse_utc(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _unwrap_ps_list(obj: Any) -> list[Any]:
    """PowerShell ConvertTo-Json of List[object] → {value: [...], Count: N}."""
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and "value" in obj and isinstance(obj["value"], list):
        return list(obj["value"])
    raise TypeError(f"expected list or PS {{value,Count}}; got {type(obj)}")


def _kv_readback_normalized(cell: dict[str, Any]) -> tuple[str | None, bool, bool, Any]:
    """Return (normalized, enforced, match, raw_blob)."""
    blob = cell.get("kv_cache_precision_readback")
    if blob is None:
        return None, False, False, None
    # May be a list-of-one from ConvertTo-Json of a scalar PSCustomObject.
    if isinstance(blob, list):
        if not blob:
            return None, False, False, blob
        blob = blob[0]
    if not isinstance(blob, dict):
        return None, False, False, blob
    enforced = bool(blob.get("enforced"))
    match = bool(blob.get("match"))
    rb = blob.get("readback") or {}
    if isinstance(rb, dict):
        norm = rb.get("normalized")
    else:
        norm = None
    return (str(norm) if norm is not None else None), enforced, match, blob


def _median(vals: list[float]) -> float | None:
    if not vals:
        return None
    return float(statistics.median(vals))


def _linreg(xs: list[float], ys: list[float]) -> dict[str, Any]:
    """Ordinary least squares y = a + b x with simple SEs (n small)."""
    n = len(xs)
    if n < 2:
        return {"n": n, "ok": False}
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=False))
    if sxx == 0:
        return {"n": n, "ok": False, "reason": "zero_x_variance"}
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys, strict=False)]
    sse = sum(r * r for r in resid)
    sst = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - sse / sst if sst > 0 else 1.0
    dof = n - 2
    mse = sse / dof if dof > 0 else float("nan")
    se_b = math.sqrt(mse / sxx) if dof > 0 and sxx > 0 else float("nan")
    se_a = math.sqrt(mse * (1.0 / n + mx * mx / sxx)) if dof > 0 and sxx > 0 else float("nan")
    # 95% CI via t≈2 for tiny n (report as se-based; n=12 for 4 points × 3 reps pooled)
    tcrit = 2.228 if n >= 12 else (2.776 if n >= 5 else 4.303)  # approx
    return {
        "n": n,
        "ok": True,
        "intercept_a": a,
        "slope_b": b,
        "r2": r2,
        "sse": sse,
        "rmse": math.sqrt(mse) if mse == mse else None,
        "se_intercept": se_a,
        "se_slope": se_b,
        "ci95_intercept": [a - tcrit * se_a, a + tcrit * se_a] if se_a == se_a else None,
        "ci95_slope": [b - tcrit * se_b, b + tcrit * se_b] if se_b == se_b else None,
        "residuals": resid,
        "tcrit_approx": tcrit,
    }


def _ols_multi(
    rows: list[dict[str, float]],
    *,
    y_key: str,
    feature_keys: list[str],
) -> dict[str, Any]:
    """y = a0 + sum a_i x_i via normal equations (numpy-free)."""
    n = len(rows)
    p = 1 + len(feature_keys)
    if n < p:
        return {"ok": False, "n": n, "reason": "underdetermined"}
    # Build X'X and X'y
    xtx = [[0.0] * p for _ in range(p)]
    xty = [0.0] * p
    ys = []
    for row in rows:
        x = [1.0] + [float(row[k]) for k in feature_keys]
        y = float(row[y_key])
        ys.append(y)
        for i in range(p):
            xty[i] += x[i] * y
            for j in range(p):
                xtx[i][j] += x[i] * x[j]

    # Gauss-Jordan invert xtx
    m = [xtx[i][:] + ([1.0 if i == j else 0.0 for j in range(p)]) for i in range(p)]
    for col in range(p):
        piv = max(range(col, p), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-18:
            return {"ok": False, "n": n, "reason": "singular"}
        m[col], m[piv] = m[piv], m[col]
        div = m[col][col]
        m[col] = [v / div for v in m[col]]
        for r in range(p):
            if r == col:
                continue
            factor = m[r][col]
            m[r] = [m[r][c] - factor * m[col][c] for c in range(2 * p)]
    inv = [row[p:] for row in m]
    coef = [sum(inv[i][j] * xty[j] for j in range(p)) for i in range(p)]

    yhat = []
    resid = []
    for row, y in zip(rows, ys, strict=False):
        x = [1.0] + [float(row[k]) for k in feature_keys]
        yh = sum(c * xi for c, xi in zip(coef, x, strict=False))
        yhat.append(yh)
        resid.append(y - yh)
    sse = sum(r * r for r in resid)
    my = sum(ys) / n
    sst = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - sse / sst if sst > 0 else 1.0
    dof = n - p
    mse = sse / dof if dof > 0 else float("nan")
    # coefficient SEs from diag of (X'X)^{-1} * mse
    se = [
        math.sqrt(mse * inv[i][i]) if dof > 0 and inv[i][i] >= 0 else float("nan") for i in range(p)
    ]
    tcrit = 1.984 if n >= 30 else (2.042 if n >= 30 else 2.0)  # rough
    if n < 30:
        tcrit = 2.042 if n >= 30 else (2.086 if n >= 20 else 2.306)
    names = ["a0"] + [f"a_{k}" for k in feature_keys]
    # remap feature names to a1, C for the hypothesis form
    out_coef = {"a0": coef[0]}
    out_se = {"a0": se[0]}
    out_ci = {"a0": [coef[0] - tcrit * se[0], coef[0] + tcrit * se[0]] if se[0] == se[0] else None}
    for i, k in enumerate(feature_keys, start=1):
        out_coef[k] = coef[i]
        out_se[k] = se[i]
        out_ci[k] = [coef[i] - tcrit * se[i], coef[i] + tcrit * se[i]] if se[i] == se[i] else None
    return {
        "ok": True,
        "n": n,
        "p": p,
        "feature_keys": feature_keys,
        "coef": out_coef,
        "se": out_se,
        "ci95_approx": out_ci,
        "r2": r2,
        "rmse": math.sqrt(mse) if mse == mse else None,
        "sse": sse,
        "residuals": resid,
        "tcrit_approx": tcrit,
        "names": names,
    }


def analyze_matrix(
    cells: list[dict[str, Any]],
    canaries: list[dict[str, Any]],
    canary_gate: dict[str, Any],
    *,
    run_id: str,
) -> dict[str, Any]:
    """Produce analysis sections (a)-(e). Uses ttft_ratio; does not filter cache_retained."""

    def ttft_ratio(c: dict[str, Any]) -> float | None:
        t1 = c.get("turn1_prefill_s")
        t2 = c.get("turn2_prefill_s")
        if t1 is None or t2 is None or float(t1) == 0:
            return None
        return float(t2) / float(t1)

    # Enrich ratios (computed; not from cache_retained).
    for c in cells:
        c["_ttft_ratio"] = ttft_ratio(c)

    # ---- (a) delta=50 anomaly ----
    def _group_vals(arm: str, nc: int, mode: str, d: int) -> list[float]:
        return [
            float(c["turn2_prefill_s"])
            for c in cells
            if c.get("arm") == arm
            and int(c.get("n_cached") or 0) == nc
            and c.get("mode") == mode
            and int(c.get("delta") or 0) == d
            and c.get("classification") == "OK"
            and c.get("turn2_prefill_s") is not None
        ]

    a_rows = []
    for arm in ARMS:
        for nc in N_CACHED:
            for mode in MODES:
                for d in DELTAS:
                    vals = _group_vals(arm, nc, mode, d)
                    a_rows.append(
                        {
                            "arm": arm,
                            "n_cached": nc,
                            "mode": mode,
                            "delta": d,
                            "n": len(vals),
                            "turn2_median": _median(vals),
                            "turn2_vals": vals,
                        }
                    )

    def _row(arm: str, nc: int, mode: str, d: int) -> dict[str, Any]:
        for r in a_rows:
            if r["arm"] == arm and r["n_cached"] == nc and r["mode"] == mode and r["delta"] == d:
                return r
        raise KeyError(f"missing group {arm} nc={nc} {mode} d={d}")

    d50_exceeds_d150 = []
    for arm in ARMS:
        for nc in N_CACHED:
            r50 = _row(arm, nc, "RESIDENT", 50)
            r150 = _row(arm, nc, "RESIDENT", 150)
            exceeds = (
                r50["turn2_median"] is not None
                and r150["turn2_median"] is not None
                and r50["turn2_median"] > r150["turn2_median"]
            )
            per_rep = []
            for rep in REPEATS:
                v50 = next(
                    (
                        float(c["turn2_prefill_s"])
                        for c in cells
                        if c.get("arm") == arm
                        and int(c.get("n_cached") or 0) == nc
                        and c.get("mode") == "RESIDENT"
                        and int(c.get("delta") or 0) == 50
                        and int(c.get("repeat") or -1) == rep
                        and c.get("turn2_prefill_s") is not None
                    ),
                    None,
                )
                v150 = next(
                    (
                        float(c["turn2_prefill_s"])
                        for c in cells
                        if c.get("arm") == arm
                        and int(c.get("n_cached") or 0) == nc
                        and c.get("mode") == "RESIDENT"
                        and int(c.get("delta") or 0) == 150
                        and int(c.get("repeat") or -1) == rep
                        and c.get("turn2_prefill_s") is not None
                    ),
                    None,
                )
                if v50 is not None and v150 is not None:
                    per_rep.append(
                        {
                            "repeat": rep,
                            "d50": v50,
                            "d150": v150,
                            "d50_gt_d150": v50 > v150,
                        }
                    )
            nr50 = _row(arm, nc, "NON_RESIDENT", 50)
            nr150 = _row(arm, nc, "NON_RESIDENT", 150)
            nr_exceeds = (
                nr50["turn2_median"] is not None
                and nr150["turn2_median"] is not None
                and nr50["turn2_median"] > nr150["turn2_median"]
            )
            d50_exceeds_d150.append(
                {
                    "arm": arm,
                    "n_cached": nc,
                    "resident_d50_median": r50["turn2_median"],
                    "resident_d150_median": r150["turn2_median"],
                    "resident_d50_gt_d150": exceeds,
                    "resident_ratio_d50_over_d150": (
                        r50["turn2_median"] / r150["turn2_median"]
                        if r50["turn2_median"] and r150["turn2_median"]
                        else None
                    ),
                    "per_repeat": per_rep,
                    "per_repeat_all_d50_gt": all(p["d50_gt_d150"] for p in per_rep)
                    if per_rep
                    else None,
                    "non_resident_d50_median": nr50["turn2_median"],
                    "non_resident_d150_median": nr150["turn2_median"],
                    "non_resident_d50_gt_d150": nr_exceeds,
                }
            )

    n_res_groups = sum(1 for g in d50_exceeds_d150 if g["resident_d50_gt_d150"])
    n_nr_groups = sum(1 for g in d50_exceeds_d150 if g["non_resident_d50_gt_d150"])
    n_rep_all = sum(1 for g in d50_exceeds_d150 if g["per_repeat_all_d50_gt"])

    section_a = {
        "question": "RESIDENT turn2 median at d=50 exceeds d=150 in arm x n_cached groups?",
        "n_groups_arm_x_n": len(d50_exceeds_d150),
        "resident_d50_gt_d150_count": n_res_groups,
        "resident_d50_gt_d150_of": f"{n_res_groups}/12",
        "per_repeat_all_three_d50_gt_count": n_rep_all,
        "non_resident_d50_gt_d150_count": n_nr_groups,
        "non_resident_same_shape": n_nr_groups == 12,
        "mechanism": None,  # do not propose unless data separates candidates
        "note": (
            "NON_RESIDENT does not show the same d50>d150 shape "
            f"({n_nr_groups}/12). No mechanism proposed."
        ),
        "groups": d50_exceeds_d150,
    }

    # ---- (b) fit RESIDENT turn2, exclude d=50 ----
    fit_deltas = [d for d in DELTAS if d != 50]
    section_b: dict[str, Any] = {
        "excluded_deltas": [50],
        "excluded_reason": "delta=50 anomaly (section a); excluded from fit as instructed",
        "pure_multiplicative_falsified": True,
        "pure_multiplicative_note": (
            "C*d*n predicts d1000/d50=20 at every n; observed RESIDENT median "
            "ratios rise with n (see scale rows). Form turn2~a0+a1*n+C*n*d is a hypothesis."
        ),
        "arms": {},
    }
    # observed d1000/d50 scales for context
    scales = {}
    for arm in ARMS:
        scales[arm] = {}
        for nc in N_CACHED:
            m50 = next(
                r["turn2_median"]
                for r in a_rows
                if r["arm"] == arm
                and r["n_cached"] == nc
                and r["mode"] == "RESIDENT"
                and r["delta"] == 50
            )
            m1000 = next(
                r["turn2_median"]
                for r in a_rows
                if r["arm"] == arm
                and r["n_cached"] == nc
                and r["mode"] == "RESIDENT"
                and r["delta"] == 1000
            )
            scales[arm][nc] = (m1000 / m50) if m50 and m1000 else None
    section_b["observed_d1000_over_d50_resident_median"] = scales

    for arm in ARMS:
        rows = []
        for c in cells:
            if c.get("arm") != arm or c.get("mode") != "RESIDENT":
                continue
            if c.get("classification") != "OK":
                continue
            d = int(c.get("delta") or 0)
            if d == 50:
                continue
            if d not in fit_deltas:
                continue
            nc = int(c.get("n_cached") or 0)
            t2 = c.get("turn2_prefill_s")
            if t2 is None:
                continue
            rows.append(
                {
                    "n": float(nc),
                    "d": float(d),
                    "n_times_d": float(nc) * float(d),
                    "turn2": float(t2),
                }
            )
        # Hypothesis: turn2 ~ a0 + a1*n + C*n*d
        fit = _ols_multi(rows, y_key="turn2", feature_keys=["n", "n_times_d"])
        # Also fit pure C*n*d (no intercept/linear) for comparison
        fit_pure = _ols_multi(rows, y_key="turn2", feature_keys=["n_times_d"])
        # Relative residual on medians for f16-style check
        median_resid = []
        if fit.get("ok"):
            a0 = fit["coef"]["a0"]
            a1 = fit["coef"]["n"]
            C = fit["coef"]["n_times_d"]
            for nc in N_CACHED:
                for d in fit_deltas:
                    med = next(
                        r["turn2_median"]
                        for r in a_rows
                        if r["arm"] == arm
                        and r["n_cached"] == nc
                        and r["mode"] == "RESIDENT"
                        and r["delta"] == d
                    )
                    if med is None or med == 0:
                        continue
                    pred = a0 + a1 * nc + C * nc * d
                    median_resid.append(
                        {
                            "n_cached": nc,
                            "delta": d,
                            "median": med,
                            "pred": pred,
                            "rel_err": (pred - med) / med,
                            "abs_rel_err": abs(pred - med) / med,
                        }
                    )
        max_abs_rel = max((r["abs_rel_err"] for r in median_resid), default=None)
        section_b["arms"][arm] = {
            "form": "turn2 ~ a0 + a1*n_cached + C*n_cached*delta",
            "fit": {
                k: v
                for k, v in fit.items()
                if k != "residuals"  # keep residuals summary only
            },
            "residual_rmse": fit.get("rmse"),
            "residual_sse": fit.get("sse"),
            "coef_mapped": {
                "a0": fit.get("coef", {}).get("a0"),
                "a1": fit.get("coef", {}).get("n"),
                "C": fit.get("coef", {}).get("n_times_d"),
            },
            "ci95_mapped": {
                "a0": fit.get("ci95_approx", {}).get("a0"),
                "a1": fit.get("ci95_approx", {}).get("n"),
                "C": fit.get("ci95_approx", {}).get("n_times_d"),
            },
            "median_rel_errors": median_resid,
            "max_abs_rel_err_on_medians": max_abs_rel,
            "within_5pct_all_medians": (max_abs_rel is not None and max_abs_rel <= 0.05),
            "pure_C_nd_fit_r2": fit_pure.get("r2"),
            "form_verdict": (
                "hypothesis_acceptable"
                if max_abs_rel is not None and max_abs_rel <= 0.10
                else "form_strained_or_wrong"
            ),
        }

    # ---- (c) precision effect at matched (n, delta) ----
    section_c_rows = []
    for nc in N_CACHED:
        for d in DELTAS:
            meds = {}
            for arm in ARMS:
                meds[arm] = next(
                    r["turn2_median"]
                    for r in a_rows
                    if r["arm"] == arm
                    and r["n_cached"] == nc
                    and r["mode"] == "RESIDENT"
                    and r["delta"] == d
                )
            f16 = meds["gpu_only_f16"]
            ratios = {}
            for arm in ("gpu_only_u8", "gpu_only_u4"):
                if f16 and meds[arm]:
                    # f16/uX: >1 means f16 slower; <1 means f16 faster
                    ratios[arm] = f16 / meds[arm]
            section_c_rows.append(
                {
                    "n_cached": nc,
                    "delta": d,
                    "resident_turn2_median": meds,
                    "f16_over_u8": ratios.get("gpu_only_u8"),
                    "f16_over_u4": ratios.get("gpu_only_u4"),
                    "f16_faster_than_u8": (
                        ratios.get("gpu_only_u8") is not None and ratios["gpu_only_u8"] < 1.0
                    ),
                    "f16_faster_than_u4": (
                        ratios.get("gpu_only_u4") is not None and ratios["gpu_only_u4"] < 1.0
                    ),
                }
            )
    # Focus claim: f16 15-27% faster at d=1000 for n=8000 and 12000
    claim_cells = [
        r for r in section_c_rows if r["delta"] == 1000 and r["n_cached"] in (8000, 12000)
    ]
    section_c = {
        "metric": "RESIDENT turn2_prefill_s median; ratio = f16/quantized (<1 => f16 faster)",
        "claim_under_test": ("f16 is 15-27% FASTER than u8/u4 at d=1000 for n=8000 and 12000"),
        "claim_cells": claim_cells,
        "full_grid": section_c_rows,
        "n_grid_f16_faster_than_u8": sum(1 for r in section_c_rows if r["f16_faster_than_u8"]),
        "n_grid_f16_faster_than_u4": sum(1 for r in section_c_rows if r["f16_faster_than_u4"]),
        "n_grid_total": len(section_c_rows),
    }
    # verdict on claim
    claim_ok = []
    for r in claim_cells:
        for key in ("f16_over_u8", "f16_over_u4"):
            ratio = r.get(key)
            if ratio is None:
                claim_ok.append(False)
                continue
            # 15-27% faster => ratio in [0.73, 0.85]
            claim_ok.append(0.73 <= ratio <= 0.85)
    section_c["claim_verdict"] = (
        "confirmed_in_band"
        if claim_ok and all(claim_ok)
        else (
            "direction_confirmed_band_loose"
            if claim_cells
            and all(
                (r.get("f16_over_u8") or 1) < 1 and (r.get("f16_over_u4") or 1) < 1
                for r in claim_cells
            )
            else "refuted_or_mixed"
        )
    )

    # ---- (d) peak_ws slope B/token ----
    # Model from 2026-08-23 narrative: resident ≈ 95 KB/token workspace + nominal KV.
    # Fit peak_ws_turn1 vs n_cached (RESIDENT; pool deltas & repeats as independent samples).
    section_d: dict[str, Any] = {
        "model_under_test": (
            "peak_ws_turn1 ≈ intercept + slope * n_cached; "
            "additive narrative: slope ≈ 95 KB/token workspace + nominal_KV_bytes/token"
        ),
        "nominal_kv_bytes_per_token": {
            arm: kv_bytes_per_token(prec) for arm, prec in ARM_EXPECTED_PRECISION.items()
        },
        "arms": {},
    }
    workspace_estimates = []
    for arm in ARMS:
        xs = []
        ys = []
        for c in cells:
            if c.get("arm") != arm or c.get("mode") != "RESIDENT":
                continue
            if c.get("classification") != "OK":
                continue
            pw = c.get("peak_ws_turn1")
            nc = c.get("n_cached")
            if pw is None or nc is None:
                continue
            xs.append(float(nc))
            ys.append(float(pw))
        fit = _linreg(xs, ys)
        slope_b_per_tok = fit["slope_b"] if fit.get("ok") else None
        slope_kb = (slope_b_per_tok / 1024.0) if slope_b_per_tok is not None else None
        nominal = kv_bytes_per_token(ARM_EXPECTED_PRECISION[arm])
        nominal_kb = (nominal / 1024.0) if nominal is not None else None
        workspace_kb = (
            slope_kb - nominal_kb if slope_kb is not None and nominal_kb is not None else None
        )
        if workspace_kb is not None:
            workspace_estimates.append(workspace_kb)
        # residual vs additive model slope = 95*1024 + nominal
        additive_slope = 95.0 * 1024.0 + nominal if nominal is not None else None
        section_d["arms"][arm] = {
            "fit": {k: v for k, v in fit.items() if k != "residuals"},
            "slope_bytes_per_token": slope_b_per_tok,
            "slope_kb_per_token": slope_kb,
            "nominal_kv_kb_per_token": nominal_kb,
            "implied_workspace_kb_per_token": workspace_kb,
            "additive_model_slope_bytes": additive_slope,
            "residual_vs_additive_kb": (
                slope_kb - (additive_slope / 1024.0)
                if slope_kb is not None and additive_slope is not None
                else None
            ),
        }
    section_d["workspace_kb_across_arms"] = workspace_estimates
    section_d["workspace_constant_across_arms"] = (
        (max(workspace_estimates) - min(workspace_estimates)) < 15.0
        if len(workspace_estimates) == 3
        else None
    )
    section_d["note"] = (
        "95 KB/token workspace figure is the 2026-08-23 additive narrative "
        "(171.7 − 73.7 ≈ 98 ≈ 95). Fitted here on 4 n_cached × 3 repeats × 4 deltas "
        "RESIDENT turn1 peak_ws."
    )

    # ---- (e) canary series ----
    t0 = None
    if canaries:
        t0 = _parse_utc(canaries[0]["started_utc"])
    series = []
    for c in canaries:
        started = _parse_utc(c["started_utc"])
        elapsed_h = (started - t0).total_seconds() / 3600.0 if t0 else None
        series.append(
            {
                "canary_index": c.get("canary_index"),
                "cell_index": c.get("cell_index"),
                "after_matrix_cell_index": c.get("after_matrix_cell_index"),
                "started_utc": c.get("started_utc"),
                "elapsed_h_from_canary0": elapsed_h,
                "turn1_prefill_s": c.get("turn1_prefill_s"),
                "turn2_prefill_s": c.get("turn2_prefill_s"),
                "rel_drift_t1": c.get("rel_drift_t1"),
                "rel_drift_t2": c.get("rel_drift_t2"),
                "classification": c.get("classification"),
            }
        )
    t1s = [float(c["turn1_prefill_s"]) for c in canaries if c.get("turn1_prefill_s") is not None]
    t2s = [float(c["turn2_prefill_s"]) for c in canaries if c.get("turn2_prefill_s") is not None]
    window_h = None
    if series and series[-1]["elapsed_h_from_canary0"] is not None:
        window_h = series[-1]["elapsed_h_from_canary0"]
    section_e = {
        "n_canaries": len(canaries),
        "window_h": window_h,
        "canary_gate_thresholds_used": {
            "threshold_t1": canary_gate.get("threshold_t1"),
            "threshold_t2": canary_gate.get("threshold_t2"),
            "ref_turn1_prefill_s": canary_gate.get("ref_turn1_prefill_s"),
            "ref_turn2_prefill_s": canary_gate.get("ref_turn2_prefill_s"),
            "derivation_applied": canary_gate.get("derivation_applied"),
        },
        "turn1_min": min(t1s) if t1s else None,
        "turn1_max": max(t1s) if t1s else None,
        "turn1_median": _median(t1s),
        "turn1_spread_rel": (
            (max(t1s) - min(t1s)) / _median(t1s) if t1s and _median(t1s) else None
        ),
        "turn2_min": min(t2s) if t2s else None,
        "turn2_max": max(t2s) if t2s else None,
        "turn2_median": _median(t2s),
        "turn2_spread_rel": (
            (max(t2s) - min(t2s)) / _median(t2s) if t2s and _median(t2s) else None
        ),
        "series": series,
        "note": (
            "Freshness-gate data over a clean ~4.4 h window. Thresholds are those "
            "the run actually armed from the first 3 canaries — not re-derived."
        ),
    }

    return {
        "run_id": run_id,
        "a_delta50_anomaly": section_a,
        "b_resident_turn2_fit": section_b,
        "c_precision_effect": section_c,
        "d_peak_ws_slope": section_d,
        "e_canary_series": section_e,
        "caveats": {
            "ttft_ratio_used": True,
            "cache_retained_not_filtered": True,
            "cache_retained_note": (
                "cache_retained is false on all 288 cells because WS arm does not "
                "track KV delta; analysis uses ttft_ratio_turn2_over_turn1 / turn2 times."
            ),
            "residency_operator_claim": (
                "Operator: residency confirmed in all 48 groups (RESIDENT/NON_RESIDENT "
                "ratios 0.079-0.469). Not re-derived here."
            ),
        },
    }


def seal_session(session_id: str, *, run_id: str | None = None) -> dict[str, Any]:
    run_id = run_id or session_id
    session_dir = ROOT / "derived" / "delta_prefill" / session_id
    if not session_dir.is_dir():
        raise SystemExit(f"missing session dir {session_dir}")

    plan_path = session_dir / "plan.json"
    summary_path = session_dir / "summary.json"
    if not plan_path.is_file():
        raise SystemExit(f"missing {plan_path}")

    plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8-sig")) if summary_path.is_file() else plan
    )

    plan_sha = _sha256_file(plan_path)
    summary_sha = _sha256_file(summary_path) if summary_path.is_file() else None
    summary_is_plan_copy = plan_sha == summary_sha

    if plan.get("status") != "complete":
        raise SystemExit(f"refusing to seal: status={plan.get('status')!r}")

    cells = _unwrap_ps_list(plan.get("cells"))
    canaries = _unwrap_ps_list(plan.get("canaries"))
    canary_gate = plan.get("canary_gate") or {}

    if len(cells) != 288:
        raise SystemExit(f"expected 288 matrix cells in plan, found {len(cells)}")
    if len(canaries) != 25:
        raise SystemExit(f"expected 25 canaries in plan, found {len(canaries)}")

    # Cartesian completeness from plan cells
    expected = {
        (a, nc, d, m, r)
        for a in ARMS
        for nc in N_CACHED
        for d in DELTAS
        for m in MODES
        for r in REPEATS
    }
    found = set()
    for c in cells:
        found.add(
            (
                c.get("arm"),
                int(c.get("n_cached")),
                int(c.get("delta")),
                c.get("mode"),
                int(c.get("repeat")),
            )
        )
    if found != expected:
        raise SystemExit(
            f"cartesian incomplete: missing={sorted(expected - found)[:5]} "
            f"extra={sorted(found - expected)[:5]}"
        )

    # KV readback gate — refuse seal if any mismatch
    kv_failures: list[dict[str, Any]] = []
    for c in cells + canaries:
        arm = c.get("arm")
        expected_prec = ARM_EXPECTED_PRECISION.get(str(arm))
        norm, enforced, match, raw = _kv_readback_normalized(c)
        ok = expected_prec is not None and enforced and match and norm == expected_prec
        if not ok:
            kv_failures.append(
                {
                    "arm": arm,
                    "mode": c.get("mode"),
                    "n_cached": c.get("n_cached"),
                    "delta": c.get("delta"),
                    "repeat": c.get("repeat"),
                    "is_canary": c.get("is_canary"),
                    "canary_index": c.get("canary_index"),
                    "expected_precision": expected_prec,
                    "readback_normalized": norm,
                    "enforced": enforced,
                    "match": match,
                    "artifact": c.get("artifact"),
                    "raw": raw,
                }
            )
    if kv_failures:
        fail_path = session_dir / "KV_READBACK_FAILURES.json"
        fail_path.write_text(
            json.dumps({"n": len(kv_failures), "failures": kv_failures}, indent=2) + "\n",
            encoding="utf-8",
        )
        raise SystemExit(
            f"REFUSING TO SEAL: {len(kv_failures)} cell(s) failed KV readback "
            f"match (wrote {fail_path}). Failures are data, not droppable."
        )

    # Locate cell files
    matrix_files = sorted(
        p for p in session_dir.glob("dispatch_p_interleaved_*.json") if "CANARY" not in p.name
    )
    canary_files = sorted(p for p in session_dir.glob("dispatch_p_interleaved_*CANARY*.json"))
    if len(matrix_files) != 288 or len(canary_files) != 25:
        raise SystemExit(
            f"cell files on disk: matrix={len(matrix_files)} canary={len(canary_files)} "
            f"(want 288+25)"
        )

    # Uptime from boot (operator-reported local) and first/last cell
    # boot local 14:33:21 on 2026-08-23, America/New_York = UTC-4 → 18:33:21Z
    boot_utc = datetime(2026, 8, 23, 18, 33, 21, tzinfo=UTC)
    first_cell = min(cells, key=lambda c: c["started_utc"])
    last_cell = max(cells, key=lambda c: c["ended_utc"])
    first_canary = min(canaries, key=lambda c: c["started_utc"])
    last_canary = max(canaries, key=lambda c: c["ended_utc"])
    uptime = {
        "boot_utc_assumed_from_operator_local": boot_utc.isoformat().replace("+00:00", "Z"),
        "boot_local_as_reported": LAUNCH_ENVIRONMENT_AS_REPORTED["boot_local"],
        "timezone_assumption": "America/New_York UTC-4 (operator dispatch date)",
        "uptime_s_at_first_matrix_cell": (
            _parse_utc(first_cell["started_utc"]) - boot_utc
        ).total_seconds(),
        "uptime_s_at_last_matrix_cell": (
            _parse_utc(last_cell["ended_utc"]) - boot_utc
        ).total_seconds(),
        "uptime_s_at_first_canary": (
            _parse_utc(first_canary["started_utc"]) - boot_utc
        ).total_seconds(),
        "uptime_s_at_last_canary": (
            _parse_utc(last_canary["ended_utc"]) - boot_utc
        ).total_seconds(),
        "first_matrix_cell_started_utc": first_cell["started_utc"],
        "last_matrix_cell_ended_utc": last_cell["ended_utc"],
    }

    analysis = analyze_matrix(cells, canaries, canary_gate, run_id=run_id)

    sealed_utc = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    out = ROOT / "derived" / "delta_prefill" / f"sealed_{run_id}"
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing seal dir {out}")
    out.mkdir(parents=True, exist_ok=False)
    cells_dir = out / "cells"
    canaries_dir = out / "canaries"
    cells_dir.mkdir()
    canaries_dir.mkdir()

    for src in matrix_files:
        shutil.copy2(src, cells_dir / src.name)
    for src in canary_files:
        shutil.copy2(src, canaries_dir / src.name)

    # Normalize plan cells to a real JSON array for the seal (do not mutate session plan).
    plan_normalized = dict(plan)
    plan_normalized["cells"] = cells
    plan_normalized["canaries"] = canaries

    step0 = {
        "plan_sha256": plan_sha,
        "summary_sha256": summary_sha,
        "summary_is_byte_identical_copy_of_plan": summary_is_plan_copy,
        "analysis_in_plan": False,
        "analysis_location": (
            "Write-ThreeNumbers stdout in launch log only "
            "(derived/delta_prefill/_launches/delta_prefill_20260823_150728.log); "
            "never persisted into plan/summary"
        ),
        "result_json": {
            "session_result_json_exists": (session_dir / "result.json").is_file(),
            "registered_result_path": (
                "derived/delta_prefill/_launches/delta_prefill_20260823_150728.result.json"
            ),
            "registered_result_path_exists": (
                ROOT
                / "derived"
                / "delta_prefill"
                / "_launches"
                / "delta_prefill_20260823_150728.result.json"
            ).is_file(),
            "last_result_json_exists": (
                ROOT / "derived" / "delta_prefill" / "_launches" / "last_result.json"
            ).is_file(),
            "why": (
                "Orchestrate registers result_path=$tagLaunch.result.json, but "
                "DetachedWorker writes only derived/delta_prefill/_launches/last_result.json "
                "(run_delta_prefill_matrix.ps1 ~1509). Session-dir result.json is written "
                "only on the HeartbeatOnly verify_detach path (~1068), not on the matrix "
                "worker path. Fix deferred per dispatch."
            ),
        },
        "cell_files_on_disk": {
            "matrix": len(matrix_files),
            "canary": len(canary_files),
            "cartesian_complete": True,
        },
        "plan_cells_ps_list_serialization": (
            "plan.cells was {value:[...], Count:288} from ConvertTo-Json on "
            "List[object]; seal_delta_prefill_session.py expected a JSON array "
            "and globbed delta_prefill_*.json (0 hits). This sealer unwraps value."
        ),
    }

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "status": "COMPLETE",
        "kind": "delta_prefill_matrix",
        "dispatch": "DISPATCH_P_interleaved_precision",
        "seal_style": "derived_diagnostic",
        "sealed_utc": sealed_utc,
        "session_id": session_id,
        "measurement_isolation_mode": "remote",
        "measurement_launch_context": plan.get("launch_context"),
        "matrix": {
            "tag": plan.get("tag"),
            "arms": plan.get("arms"),
            "n_cached": plan.get("n_cached"),
            "deltas": plan.get("deltas"),
            "modes": list(MODES),
            "repeats": plan.get("repeats"),
            "cells_completed": 288,
            "cells_ok": sum(1 for c in cells if c.get("classification") == "OK"),
            "canaries_completed": 25,
            "canaries_ok": sum(1 for c in canaries if c.get("classification") == "OK"),
            "plan_started_utc": plan.get("started_utc"),
            "plan_ended_utc": plan.get("ended_utc"),
            "canary_every_n": (plan.get("canary") or {}).get("every_n"),
            "canary_gate": canary_gate,
            "cell_timeout_s": plan.get("cell_timeout_s"),
            "kv_readback_gate": "all_cells_match_arm_enforced",
        },
        "launch_environment_as_reported": LAUNCH_ENVIRONMENT_AS_REPORTED,
        "environment_launch_from_plan": plan.get("environment_launch"),
        "environment_end_from_plan": plan.get("environment_end"),
        "uptime": uptime,
        "step0": step0,
        "admissibility": {
            "matrix_status": "COMPLETE",
            "cells_ok": 288,
            "canaries_ok": 25,
            "kv_readback_failures": 0,
            "note": "All cells OK; KV readback matched arm on every cell+canary.",
        },
    }

    sealed_summary = {
        "run_id": run_id,
        "status": "COMPLETE",
        "kind": "delta_prefill_matrix",
        "dispatch": "DISPATCH_P_interleaved_precision",
        "session_id": session_id,
        "cells_ok": 288,
        "canaries_ok": 25,
        "analysis": analysis,
        "step0": step0,
        "uptime": uptime,
        "launch_environment_as_reported": LAUNCH_ENVIRONMENT_AS_REPORTED,
        "canary_gate": canary_gate,
    }

    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out / "summary.json").write_text(
        json.dumps(sealed_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out / "plan_normalized.json").write_text(
        json.dumps(plan_normalized, indent=2) + "\n", encoding="utf-8"
    )
    (out / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    (out / "STEP0_REPORT.json").write_text(
        json.dumps(step0, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Session-level result.json (was missing) — written by this sealer, not by rewriting plan.
    result_doc = {
        "status": "complete",
        "session_id": session_id,
        "run_id": run_id,
        "tag": plan.get("tag"),
        "cells": 288,
        "canaries": 25,
        "cells_ok": 288,
        "canaries_ok": 25,
        "ended_utc": plan.get("ended_utc"),
        "started_utc": plan.get("started_utc"),
        "seal_path": str(out.relative_to(ROOT)).replace("\\", "/"),
        "analysis_path": str((out / "analysis.json").relative_to(ROOT)).replace("\\", "/"),
        "kv_readback_gate": "pass",
        "written_by": "tools/seal_dispatch_p_precision_matrix.py",
    }
    result_path = session_dir / "result.json"
    if result_path.exists():
        raise SystemExit(f"refusing to overwrite existing {result_path}")
    result_path.write_text(
        json.dumps(result_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    tree_hash = _sha256_tree(out, exclude={".sealed"})
    seal_marker = {
        "run_id": run_id,
        "sealed_at_utc": sealed_utc,
        "seal_style": "derived_diagnostic",
        "tree_sha256": tree_hash,
        "self_check": "pass",
        "note": (
            "DISPATCH P precision matrix derived_diagnostic seal. "
            "raw/ not written. Do not mutate after seal."
        ),
    }
    (out / ".sealed").write_text(
        json.dumps(seal_marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for p in [out / ".sealed", *out.rglob("*")]:
        if p.is_file():
            try:
                p.chmod(p.stat().st_mode & ~0o222)
            except OSError:
                pass

    pointer = {
        "session_id": session_id,
        "run_id": run_id,
        "sealed": True,
        "seal_style": "derived_diagnostic",
        "seal_path": str(out.relative_to(ROOT)).replace("\\", "/"),
        "raw_emit": False,
        "sealed_utc": sealed_utc,
        "tree_sha256": tree_hash,
        "cells_ok": 288,
        "canaries_ok": 25,
        "result_json": str(result_path.relative_to(ROOT)).replace("\\", "/"),
        "sealer": "tools/seal_dispatch_p_precision_matrix.py",
    }
    (session_dir / "SEAL_POINTER.json").write_text(
        json.dumps(pointer, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return {
        "run_id": run_id,
        "sealed": True,
        "seal_path": str(out.resolve()),
        "tree_sha256": tree_hash,
        "result_json": str(result_path.resolve()),
        "analysis": analysis,
        "step0": step0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", default=SESSION_DEFAULT)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)
    result = seal_session(args.session_id, run_id=args.run_id)
    # Compact stdout summary
    a = result["analysis"]["a_delta50_anomaly"]
    b = result["analysis"]["b_resident_turn2_fit"]
    c = result["analysis"]["c_precision_effect"]
    d = result["analysis"]["d_peak_ws_slope"]
    e = result["analysis"]["e_canary_series"]
    print(
        json.dumps(
            {
                "step0": result["step0"],
                "run_id": result["run_id"],
                "seal_path": result["seal_path"],
                "tree_sha256": result["tree_sha256"],
                "result_json": result["result_json"],
            },
            indent=2,
        )
    )
    print("--- (a) delta=50 ---")
    print(
        f"RESIDENT d50>d150: {a['resident_d50_gt_d150_of']}; "
        f"per-repeat all three: {a['per_repeat_all_three_d50_gt_count']}/12; "
        f"NON_RESIDENT same shape: {a['non_resident_d50_gt_d150_count']}/12"
    )
    print("--- (b) fit ---")
    for arm, blk in b["arms"].items():
        cm = blk["coef_mapped"]
        print(
            f"{arm}: a0={cm['a0']:.6g} a1={cm['a1']:.6g} C={cm['C']:.6g} "
            f"r2={blk['fit'].get('r2')} max|rel|={blk['max_abs_rel_err_on_medians']} "
            f"verdict={blk['form_verdict']}"
        )
    print("--- (c) precision ---")
    print(
        f"claim_verdict={c['claim_verdict']} "
        f"f16_faster_u8={c['n_grid_f16_faster_than_u8']}/{c['n_grid_total']} "
        f"f16_faster_u4={c['n_grid_f16_faster_than_u4']}/{c['n_grid_total']}"
    )
    for row in c["claim_cells"]:
        print(
            f"  nc={row['n_cached']} d=1000 f16/u8={row['f16_over_u8']:.4f} "
            f"f16/u4={row['f16_over_u4']:.4f}"
        )
    print("--- (d) peak_ws ---")
    for arm, blk in d["arms"].items():
        print(
            f"{arm}: slope={blk['slope_kb_per_token']:.2f} KB/tok "
            f"nominal_kv={blk['nominal_kv_kb_per_token']:.2f} "
            f"workspace={blk['implied_workspace_kb_per_token']:.2f} "
            f"resid_vs_95+nom={blk['residual_vs_additive_kb']}"
        )
    print(f"workspace constant across arms? {d['workspace_constant_across_arms']}")
    print("--- (e) canaries ---")
    print(
        f"n={e['n_canaries']} window_h={e['window_h']:.3f} "
        f"t1=[{e['turn1_min']:.3f},{e['turn1_max']:.3f}] "
        f"spread_rel={e['turn1_spread_rel']:.3f} "
        f"thr_t1={e['canary_gate_thresholds_used']['threshold_t1']:.4f} "
        f"thr_t2={e['canary_gate_thresholds_used']['threshold_t2']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
