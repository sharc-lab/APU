"""Task BR week-1 amendments: DERIVED predictions, fixed before any of their runs.

Writes derived/br/BR_{INV1,XRT1,SESS1,STAT1,M2B,HYB1}_PREDICTIONS.json, registered by
docs/registrations/BR_{INV,XRT,SESS,STAT,8B,HYB1}_2026-10-26.md. Runners never read them.

Constants (sources in the registrations):
- Compiled KV bytes per token (OpenVINO 2026.2.1 GPU plugin layout; u4 keys per channel with
  scale/zero-point per 16-token block, values per token): f16 147,456; i8 84,096; u4 47,232.
  The nominal values used by the earlier registrations are f16 147,456; i8 78,336; u4 41,472.
- a = 87,520 B/token, precision-independent (A20 f16 slope, e4e797e7; the u4 slope 93,280,
  ae16fcd5, minus the 5,760 B/token u4 scale/zero-point bytes).
- B rule (BR): every unballooned XPS cell at the 57% override that has a largest-held n (one
  draw each) gives an implied non-weight budget B_i = P + k*L + a*min(L, c), with c the step cap
  (none = L) and P a fixed pre-allocated pool (cache_size cells: P = pool bytes, k*L dropped
  because the pool already holds the KV). B_BR = the median of the B_i; the band is their
  [min, max]. IN-SAMPLE by construction; the new predictions below are out of sample.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

GIB = 2**30
MIB = 2**20
K_COMPILED = {"f16": 147_456, "i8": 84_096, "u4": 47_232}
K_NOMINAL = {"f16": 147_456, "i8": 78_336, "u4": 41_472}
A = 87_520
A_RUN = "e4e797e7"
WINDOW = 40_960
CAP_NONE = 0  # marker: unchunked (the whole prompt in one step)

# (run_id, arm, cap tokens or CAP_NONE, largest held n, fixed pool bytes or 0, source cell)
B_CELLS: list[tuple[str, str, int, int, int, str]] = [
    ("5943c992", "u4", CAP_NONE, 25_000, 0, "mech_u4_unlx"),
    ("d6616dfd", "u4", CAP_NONE, 25_000, 0, "bal_g0_u4"),
    ("3cd03df7", "u4", CAP_NONE, 24_250, 0, "a33_u4_cache0"),
    ("aa914c4c", "f16", CAP_NONE, 14_000, 0, "mech_f16_unlx"),
    ("ba7a1e89", "f16", CAP_NONE, 13_250, 0, "bal_g0_f16"),
    ("fd074be1", "i8", CAP_NONE, 18_000, 0, "mech_8_unlx"),
    ("b038e362", "i8", CAP_NONE, 19_750, 0, "a21_8bit_rep"),
    ("f8df3847", "u4", 256, 70_000, 0, "mech_u4_c256"),
    ("d782cf80", "f16", 256, 22_000, 0, "mech_f16_c256"),
    ("1e9671a0", "i8", 256, 38_000, 0, "mech_8_c256"),
    ("f7542f68", "u4", 1024, 67_000, 0, "mech_u4_c1024"),
    ("b48e4cfe", "f16", 1024, 21_000, 0, "mech_f16_c1024"),
    ("3e45f554", "i8", 1024, 36_000, 0, "mech_8_c1024"),
    ("aaadf214", "u4", CAP_NONE, 13_000, 2 * GIB, "a33_u4_cache2"),
]
# Sessions (CC-GROW, v3b window 20261024_175932): last HELD turn's prompt tokens. The failing
# turn is the next one (failure_mode grow_turn<k>, k = number of held turns), so the history
# before the failing turn is about this value (+ the 1-token reply and its template).
GROW_LAST_HELD = {
    ("f16", 1024): ("e99bb2ad", 21_736),
    ("f16", 2048): ("71b7d9d8", 18_532),
    ("f16", 4096): ("9cb921fd", 16_429),
    ("u4", 1024): ("0c7ae0a3", 65_206),
    ("u4", 2048): ("d7e8b43e", 61_771),
    ("u4", 4096): ("10cb79e4", 57_499),
}
GROW_C256_LAST_HELD = {
    ("f16", 2048): ("2bfae783", 16_473),
    ("u4", 2048): ("08e52da9", 65_889),
}
# Shared GPU memory per slider setting (A19 DXGI records, ullTotalPhys 16,752,234,496 B).
SHARED_BYTES = {50: 8_376_117_248, 57: 9_548_773_662, 74: 12_396_653_527}
SHARED_RUNS = {
    50: "window_20261025_174345",
    57: "window_20261025_204059",
    74: "window_20261025_185526",
}
SLIDER_CELLS = {
    50: {"u4": ("a2a7c763", 22_750), "f16": ("769c5a03", 13_250)},
    57: {"u4": ("f4abaf6e", 21_250), "f16": ("c461f165", 12_500)},
    74: {"u4": ("9e31d4cb", 17_750), "f16": ("39d38c85", None)},
}
BAND_HELDOUT = 0.15
QWEN3_4B = {"hidden_size": 2560, "intermediate_size": 9728}
QWEN3_8B = {"hidden_size": 4096, "intermediate_size": 12288}
W_4B_IR = 2_290_768_181  # configs/models/Qwen3-4B-int4-ov.yaml ir_bytes
W_8B_IR = 4_882_865_352  # configs/models/Qwen3-8B-int4-ov.v2.yaml ir_bytes


def implied_budget(arm: str, cap: int, n: int, pool: int) -> int:
    """B_i = P + k*L + a*min(L, c); with a fixed pool P the KV term is inside P."""
    step = n if cap == CAP_NONE else min(n, cap)
    kv = 0 if pool else K_COMPILED[arm] * n
    return pool + kv + A * step


def budget_rule() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    vals: list[int] = []
    for run, arm, cap, n, pool, cell in B_CELLS:
        b = implied_budget(arm, cap, n, pool)
        rows.append(
            {
                "run_id": run,
                "cell": cell,
                "arm": arm,
                "cap": "none" if cap == CAP_NONE else cap,
                "largest_held": n,
                "pool_bytes": pool,
                "implied_B_bytes": b,
            }
        )
        vals.append(b)
    med = int(statistics.median(vals))
    return {
        "rule": "B_i = P + k_compiled*L + a*min(L, cap); B_BR = median(B_i); band [min, max]",
        "cells": rows,
        "B_BR_bytes": med,
        "B_BR_GB": round(med / 1e9, 3),
        "B_BR_GiB": round(med / GIB, 3),
        "band_bytes": [min(vals), max(vals)],
        "spread_pct": round(100 * (max(vals) - min(vals)) / 2 / med, 1),
        "label": "IN-SAMPLE (57% override, windows 20261023_140349_cc_v3 and 20261024_175932_cc_v3b)",
    }


def b_br() -> int:
    return int(budget_rule()["B_BR_bytes"])


def band(x: float, frac: float = BAND_HELDOUT) -> list[int]:
    return [round(x * (1 - frac)), round(x * (1 + frac))]


def session_law(arm: str, t: int, *, cap: int = CAP_NONE, budget: int | None = None) -> float:
    """Corrected session law: L_hist = (B - a*step)/k - T, step = T unchunked, else min(T, cap).

    The scheduler grows the KV pool to cover the whole turn before prefilling it, so the
    footprint at the failing turn is k*(L_hist + T) + a*step."""
    b = b_br() if budget is None else budget
    step = t if cap == CAP_NONE else min(t, cap)
    return (b - A * step) / K_COMPILED[arm] - t


def sub_turn_rule() -> str:
    return (
        "x* = (B_BR - k*H) / (k + a) for the final-turn size that still holds on history H "
        "(H = that run's own last held prompt, a measured input); predicted x* is computed "
        "from H at analysis time with this formula and B_BR, nothing refitted"
    )


def build_inv() -> dict[str, Any]:
    adv = {pct: round((0.94 * SHARED_BYTES[pct] - 450 * MIB) / GIB, 3) for pct in SHARED_BYTES}
    sp = {
        pct: round(0.90 * (0.94 * SHARED_BYTES[pct] - 450 * MIB) / GIB, 2) for pct in SHARED_BYTES
    }
    return {
        "id": "BR_INV1_PREDICTIONS",
        "registration": "docs/registrations/BR_INV_2026-10-26.md",
        "generator": "tools/br_predict.py",
        "runner_must_not_read_this_file": True,
        "observed_trigger": {
            "slider_cells": {str(k): v for k, v in SLIDER_CELLS.items()},
            "shared_bytes": {str(k): v for k, v in SHARED_BYTES.items()},
            "a19_records": SHARED_RUNS,
            "same_setting_drift": {
                "57pct_fri": ["5943c992", 25_000],
                "57pct_sun": ["f4abaf6e", 21_250],
            },
        },
        "advertised_device_size_GiB": {
            "formula": "0.94 x SharedSystemMemory - 450 MiB (Intel compute-runtime source, "
            "LITERATURE via the GPU-share plan); check: GPU_DEVICE_TOTAL_MEM_SIZE at every boot "
            "within 1 MiB of it",
            "values": {str(k): v for k, v in adv.items()},
            "known_57pct": "8,503,984,128 B = 7.920 GiB (d7e8b43e load record)",
        },
        "shape_predictor_threshold_GiB": {str(k): v for k, v in sp.items()},
        "per_boot_set": {
            "f16_uncapped": {"points": [4096, 6144, 8192], "repeats": 3},
            "u4_uncapped": {"points": [16384, 18432], "repeats": 3},
            "order": "seeded shuffle per boot and arm (seed = 2610260 + boot); arm order seeded "
            "too; recorded in the queue log before the first trial",
        },
        "predictions": {
            "P-INV-1 (H1)": "usm_device bytes at u4 16,384 after turn 1 (gpu_mem_stats) are >= 15% "
            "higher at 74% than at 50%, in both ABAB pairs",
            "P-INV-2 (H1 vs H2)": "raw allocate-until-failure ceiling (USM device, touched, 256 MiB "
            "chunks) at 74% >= its 50% value (H1). H2: it falls by more than 15%, or device USM "
            "fails before host USM",
            "P-INV-3 (H1)": "llama.cpp (XRT-1 E2) ceiling is non-decreasing from 50% to 74%; H2: it "
            "also inverts (falls > 15%)",
            "P-INV-4 (H3)": "a PINNED 1 GiB GPU balloon (kept resident by a 10 ms kernel) or a 2 GiB "
            "VirtualLock CPU balloon lowers the u4 hold count at 16,384 / 18,432 versus the same "
            "boot's unballooned trials (H3); H1 predicts no change, as with the idle balloons "
            "(598e882d, f9953743, 0014465c, 62d59ab8)",
            "P-INV-5 (inversion reproduces)": "at u4 18,432 the two 50% boots hold >= 5 of 6 and "
            "the two 74% boots hold <= 1 of 6 (from L* 22,750 a2a7c763 vs 17,750 9e31d4cb); u4 "
            "16,384 and f16 4,096 / 6,144 hold at both settings (controls)",
            "P-INV-6 (f16 collapse)": "f16 8,192 fails >= 2 of 3 in at least one 74% boot and "
            "holds >= 5 of 6 at 50%. If 8,192 holds at 74%, the 39d38c85 low-rung failure is "
            "read as a single-boot event (its loaded Available was 3,492 MB vs 7,150-8,061 MB "
            "in the other slider children)",
            "P-INV-7 (H4)": "if boot variance explains the inversion, the 74/50 difference in hold "
            "counts changes sign between the two ABAB pairs",
            "not_scheduled": "f16 cap256 inversion within drift (from the plan's INV row): no "
            "INV-1 cell; scored only if a f16 cap256 cell runs at both settings",
        },
        "decision_rules": {
            "inconclusive": "ceiling differences under 15% (raw allocation, llama.cpp, any L*) "
            "are inconclusive by rule",
            "reproduced": "P-INV-5 holds and one-sided Fisher exact p < 0.05 on the pooled u4 "
            "18,432 counts (50% vs 74%), with the same sign in each ABAB pair",
            "mechanism_class": "H1 if P-INV-1 and P-INV-2 hold; H2 if P-INV-2's H2 branch holds; "
            "H3 if P-INV-4 holds; Gate A (Wed Nov 4) decides",
        },
    }


def build_xrt() -> dict[str, Any]:
    kv = {
        "f16": 36 * 2 * 8 * 128 * 2,
        "q8_0": 36 * 2 * 8 * 128 * 34 // 32,
        "q4_0": 36 * 2 * 8 * 128 * 18 // 32,
    }
    return {
        "id": "BR_XRT1_PREDICTIONS",
        "registration": "docs/registrations/BR_XRT_2026-10-26.md",
        "generator": "tools/br_predict.py",
        "runner_must_not_read_this_file": True,
        "kv_bytes_per_token_llamacpp": kv,
        "e1_census": {
            "grid": "ctx 4096/16384/32768 x ubatch 128/256/512/1024/2048/4096 x KV f16/q8_0/q4_0 "
            "x FA on/off, SYCL and Vulkan builds, --fit off",
            "P-XRT-1": "KV buffer MiB = n_ctx x k / 2^20 within 1% for every ready combination",
            "P-XRT-2": "FA on: compute buffer flat in n_ctx (max/min across 4K/16K/32K <= 1.05) at "
            "each ubatch and KV type",
            "P-XRT-3": "FA off: compute buffer grows with n_ctx at fixed ubatch (32K / 4K >= 2.0)",
            "P-XRT-4": "FA off with a quantized V cache is refused at load (recorded, not a crash)",
        },
        "e2_boundary": {
            "cells": "f16 and q4_0, FA on, ubatch 512, at slider 50% and 74%, both builds",
            "P-XRT-5": "q4_0 / f16 ceiling ratio >= 3.0 at ubatch 512 (k ratio 3.56)",
            "P-XRT-6 (=P-INV-3)": "ceiling at 74% >= ceiling at 50% for each KV type and build "
            "(H1); a fall > 15% is the H2 branch; under 15% inconclusive",
        },
        "rithwik_side": {
            "P-XRT-7 (E3, RTX 4070, FA off)": "ceiling proportional to 1 / (k + 4 x n_head x "
            "ubatch) bytes (n_head 32: 4 x 32 x 512 = 65,536 B/token at ubatch 512)",
            "P-XRT-8 (E5, Strix Halo)": "ceiling linear in the GPU share with slope 1/k (R^2 >= "
            "0.98 over >= 3 points); any drop at high share generalizes the inversion",
            "status": "CONDITIONAL on Rithwik (docs/RITHWIK_ASKS_2026-10-26.md)",
        },
    }


def build_sess() -> dict[str, Any]:
    b = b_br()
    pred: dict[str, Any] = {}
    for arm in ("f16", "u4"):
        for t in (3072, 6144):
            v = session_law(arm, t)
            pred[f"{arm}_T{t}"] = {"L_hist": round(v), "band": band(v)}
    insample: dict[str, Any] = {}
    for (arm, t), (run, last) in GROW_LAST_HELD.items():
        v = session_law(arm, t)
        old = (b - A * t) / K_COMPILED[arm]
        insample[f"{arm}_T{t}"] = {
            "run_id": run,
            "last_held_prompt": last,
            "corrected_law": round(v),
            "old_law_total": round(old),
            "residual_pct": round(100 * (last - v) / v, 1),
        }
    chunk: dict[str, Any] = {}
    for arm in ("f16", "u4"):
        for t in (2048, 6144):
            v = session_law(arm, t, cap=256)
            chunk[f"{arm}_c256_T{t}"] = {"L_hist": round(v), "band": band(v)}
    for (arm, t), (run, last) in GROW_C256_LAST_HELD.items():
        v = session_law(arm, t, cap=256)
        chunk[f"{arm}_c256_T{t}"]["in_sample"] = {
            "run_id": run,
            "last_held_prompt": last,
            "residual_pct": round(100 * (last - v) / v, 1),
        }
    reserved: dict[str, Any] = {}
    for arm in ("f16", "u4"):
        total = session_law(arm, 2048) + 2048
        blocks = int(total // 16)
        reserved[f"{arm}_T2048"] = {
            "num_kv_blocks": blocks,
            "pool_bytes": blocks * 16 * K_COMPILED[arm],
            "block_bytes": 16 * K_COMPILED[arm],
            "rule": "blocks = floor((L_hist_pred + T) / 16) at T = 2048 (covers the predicted "
            "failing turn exactly)",
        }
    return {
        "id": "BR_SESS1_PREDICTIONS",
        "registration": "docs/registrations/BR_SESS_2026-10-26.md",
        "generator": "tools/br_predict.py",
        "runner_must_not_read_this_file": True,
        "law": "L_hist = (B_BR - a*T)/k_compiled - T (unchunked); chunked: a*min(T, 256)",
        "B_BR_bytes": b,
        "a": A,
        "k_compiled": K_COMPILED,
        "logged_length_check": {
            "finding": "the CC-GROW records' last HELD turn is the one before the failing turn; "
            "the length to compare with L_hist is that turn's prompt_tokens_reported. The "
            "lengths in TASK_BK addendum 2 and the GPU-share plan (e.g. d7e8b43e 59,712) are the "
            "turn before that (one turn short): d7e8b43e held turn 29 at 61,771 and failed at "
            "turn 30",
            "evidence": "failure_mode grow_turn<k> with k = number of recorded (held) turns",
        },
        "held_out": pred,
        "in_sample_check_corrected_lengths": insample,
        "chunked_arm": chunk,
        "reserved_pool_arm": reserved,
        "sub_turn_bisection": {"rule": sub_turn_rule(), "resolution": 256},
        "phase_rule": "failure phase from the 10 Hz per-type trace: 'growth' if the last cl_mem "
        "sample before the exception is below 0.98 x k*(L_hist + T); 'prefill' otherwise; "
        "'unknown' with fewer than 2 samples in the failing turn",
        "predictions": {
            "P-SESS-1": "held-out L_hist within +-15% (session drift) of the values above",
            "P-SESS-2": "failure phase 'prefill' in >= 2/3 of failing unchunked turns (the pool "
            "grows first, then a*T is allocated); 'growth' in most turns favours copy "
            "transients",
            "P-SESS-3": "reserved-pool arm: last held history >= the dynamic arm's at the same T "
            "minus one turn; if it holds more turns than the dynamic arm, copy transients are "
            "the residual (H-copy)",
            "P-SESS-4": "chunked arm within +-15% of its law values (the a*T term replaced by "
            "a*256, the -T term kept)",
            "P-SESS-5": "sub-turn bisection: the measured largest final turn within +-1 grid "
            "step (256) of x* from the rule",
        },
    }


def law_limit(arm: str, cap: int = CAP_NONE) -> float:
    b = b_br()
    if cap == CAP_NONE:
        return b / (K_COMPILED[arm] + A)
    return (b - A * cap) / K_COMPILED[arm]


def build_stat() -> dict[str, Any]:
    anchors: dict[str, dict[str, Any]] = {
        "f16_unl": {"arm": "f16", "cap": "none", "bisection_max": ["aa914c4c", 14_000]},
        "u4_unl": {"arm": "u4", "cap": "none", "bisection_max": ["5943c992", 25_000]},
        "f16_c256": {"arm": "f16", "cap": 256, "bisection_max": ["d782cf80", 22_000]},
    }
    for v in anchors.values():
        cap = CAP_NONE if v["cap"] == "none" else int(v["cap"])
        lim = law_limit(str(v["arm"]), cap)
        v["law_L50"] = round(lim)
        v["law_band"] = band(lim)
    u4c = law_limit("u4", 256)
    return {
        "id": "BR_STAT1_PREDICTIONS",
        "registration": "docs/registrations/BR_STAT_2026-10-26.md",
        "generator": "tools/br_predict.py",
        "runner_must_not_read_this_file": True,
        "design": {
            "rule": "biased-coin up-and-down targeting hold 0.9 (b = 1/9), fresh child per "
            "trial, step 250, >= 30 trials per anchor over >= 3 sessions (one session = one "
            "boot), anchor blocks in seeded random order each session",
            "estimator": "centered isotonic regression (tools/stat_cir.py) for L90 and L50; 90% "
            "CI by a two-stage bootstrap (sessions, then trials within session), 2,000 draws, "
            "seed 20261026; reversal count reported",
            "covariates": "session as a random effect (secondary: logistic GLMM with pre-trial "
            "Available MB); exclusion only for a logged perturbing event",
        },
        "anchors": anchors,
        "u4_c256_binomial": {
            "level": 63_000,
            "level_rule": "0.9 x 70,000 (f8df3847, the bisection max; 0/4 at 70,000 in 0580176a "
            "so L50 < 70,000)",
            "trials": 20,
            "law_limit": round(u4c),
            "prediction": ">= 19 of 20 hold (20/20 gives reliability >= 0.861 at 95% one-sided)",
        },
        "predictions": {
            "P-STAT-1": "each anchor's L50 lies inside its law band (B_BR, compiled k)",
            "P-STAT-2": "each anchor's bisection max is >= its L50 (selection bias of one-draw "
            "bisection)",
            "P-STAT-3": "L90 < L50 for each anchor with a CI width (90%) <= 10% of L50",
            "P-STAT-4": "u4 cap256 at 63,000: >= 19 of 20 hold",
        },
    }


def a_rules(cfg: dict[str, int]) -> dict[str, int]:
    d, i = cfg["hidden_size"], cfg["intermediate_size"]
    return {"f16_live_7(d+I)": 7 * (d + i), "fp32_(2I+d)x4": (2 * i + d) * 4}


def build_m2b() -> dict[str, Any]:
    r8 = a_rules(QWEN3_8B)
    r4 = a_rules(QWEN3_4B)
    a8 = r8["f16_live_7(d+I)"]
    b8 = b_br() - (W_8B_IR - W_4B_IR)
    unc = {p: b8 / (K_COMPILED[p] + a8) for p in K_COMPILED}
    cap = (b8 - 256 * a8) / K_COMPILED["u4"]
    return {
        "id": "BR_M2B_PREDICTIONS",
        "registration": "docs/registrations/BR_8B_2026-10-26.md",
        "generator": "tools/br_predict.py",
        "runner_must_not_read_this_file": True,
        "a_derivation": {
            "qwen3_4b": {"rules": r4, "measured": [A, A_RUN]},
            "qwen3_8b": {"rules": r8, "point": a8, "band_10pct": band(a8, 0.10)},
            "note": "for the 8B both readings equal 114,688 B/token; for the 4B they bracket the "
            "measured 87,520 (-1.7% / +0.6%)",
        },
        "budget": {
            "rule": "B_8B = B_BR - (W_8B - W_4B), W = IR bytes (the BO rule with B_BR in place of "
            "the 3.412 GiB pair fit)",
            "B_BR_bytes": b_br(),
            "B_8B_bytes": b8,
            "B_8B_GiB": round(b8 / GIB, 3),
        },
        "uncapped_limits": {p: {"point": round(v), "band": band(v)} for p, v in unc.items()},
        "u4_cap256_limit": {"point": round(cap), "band": band(cap)},
        "registered_bo_values_kept": "CC_M2_PREDICTIONS.json (a 131,072; limits 3,846 / 5,116 / "
        "6,209; u4 cap256 25,022) stay registered and are scored as registered",
        "early_warning": "retro sealed 8B f16 3,156 (322b2f86) and u8 4,062 (69234ed5), older "
        "driver and tool, not a test of this amendment",
    }


def build_hyb1() -> dict[str, Any]:
    b = b_br()
    phantom = {p: round((K_COMPILED[p] + A) / K_COMPILED[p], 3) for p in K_COMPILED}
    kv_only = {p: round(b / K_COMPILED[p]) for p in K_COMPILED}
    law_unc = {p: round(b / (K_COMPILED[p] + A)) for p in K_COMPILED}
    return {
        "id": "BR_HYB1_PREDICTIONS",
        "registration": "docs/registrations/BR_HYB1_2026-10-26.md",
        "generator": "tools/br_predict.py",
        "runner_must_not_read_this_file": True,
        "no_trace_row_opened": True,
        "constants": {"B_BR_bytes": b, "a": A, "k_compiled": K_COMPILED, "window": WINDOW},
        "kv_only_phantom_ratio": phantom,
        "kv_only_phantom_ceiling": kv_only,
        "law_unchunked_ceiling": law_unc,
        "u4_c256_memory_ceiling": round(law_limit("u4", 256)),
        "band_16gb": {
            "default_i8_unchunked": [19_750, WINDOW],
            "default_source": "b038e362 (A21 8-bit, the true-default KV type i8)",
            "u4_unchunked_comparison": [24_250, WINDOW],
            "u4_unchunked_source": "3cd03df7",
            "identity": "dS = F(min(L_c, W)) - F(L_u) per call (no SLO); an upper bound per "
            "session under sticky escalation",
        },
        "band_32gb_64gb": "empty (window-bound under the default; T2S CC-WINDOW 7e249f7e / "
        "6dfd1a12 / 2d329db2 at 40,448)",
        "hard_fail_bound_per_1k_calls": {
            "copilot_mean_40.6_calls": round(1000 / 40.6, 1),
            "tracelab_about_83_steps": round(1000 / 83, 1),
        },
        "hypotheses": {
            "H-NARROW": "TraceLab calls with n_ctx <= 40,960 are < 25% of calls and < 5% of input "
            "tokens",
            "H-BAND": "P1 - P0 local-feasible share (no SLO) equals F(min(L_c, W)) - F(L_u) on "
            "every tier; zero on 32 and 64 GB",
            "H-CONTRAST": "Exgentic feasible share >= 3x TraceLab's on 16 GB",
            "H-P2": "P2 hard failures + silent overruns <= 10% of P0's, at <= 5 pp local-share "
            "loss versus P4 (m = L90)",
            "H-KV": "unchunked: P2-KV hard failures >= 5x P2's; chunk-256: within +-10% of P2's",
            "H-SILENT": "conditional on the window probe running silently past 40,960: P1 incurs "
            "silent overruns in every session crossing 40,960 below the memory ceiling; P0 "
            "fails loudly; P2 has none",
            "H-RANK": "P4 >= P3 >= P2 > P0 on failure waste and local-within-SLO share in >= 90% "
            "of sensitivity settings; at a 20 s SLO the TTFT roof binds for most calls that "
            "clear memory and window",
        },
        "window_probe": {
            "cell": "u4 chunk-256 at 42,000 and 45,000 (one trial each, XPS, 57%)",
            "prediction": "silent: both hold with no error (GenAI enforces no position limit; "
            "BB found 43 local generates above 40,960 in 8 runs)",
        },
        "replay_criteria": {
            "R-fit": ">= 95% fit/fail agreement outside the +-10% band (in-band reported)",
            "R-ttft": "median absolute TTFT error <= 15%",
            "R-guard": "zero hard failures under P2 at m = L90",
        },
    }


BUILDERS = {
    "BR_INV1_PREDICTIONS.json": build_inv,
    "BR_XRT1_PREDICTIONS.json": build_xrt,
    "BR_SESS1_PREDICTIONS.json": build_sess,
    "BR_STAT1_PREDICTIONS.json": build_stat,
    "BR_M2B_PREDICTIONS.json": build_m2b,
    "BR_HYB1_PREDICTIONS.json": build_hyb1,
}


def build_all() -> dict[str, dict[str, Any]]:
    common = budget_rule()
    out: dict[str, dict[str, Any]] = {}
    for name, fn in BUILDERS.items():
        doc = fn()
        doc["budget_rule"] = common
        out[name] = doc
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args(argv)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for name, doc in build_all().items():
        (a.out_dir / name).write_text(json.dumps(doc, indent=2) + "\n", encoding="ascii")
        print(name, "written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
