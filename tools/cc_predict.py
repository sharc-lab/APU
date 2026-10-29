"""CC cells (Task BJ): DERIVED predictions for the context-ceiling cells, before any CC run.

Writes derived/cc/CC_PREDICTIONS.json, registered by
docs/registrations/CC_CONTEXT_CEILING_2026-10-23.md.

Inputs (each labelled):
- SEALED XPS resident limits, Qwen3-4B INT4, plain LLMPipeline (uncapped step): f16 15,000
  (553b3a5c), u8 20,000 (cf3555d5), u4 26,500 (e4a22dac). All three failed in TURN 1 (the n-token
  prefill) with CL_OUT_OF_RESOURCES; turn 2 appends only 183 tokens (boot4_session.DELTA_TOKENS).
- k_KV per token from the cache shapes (36 layers x 8 KV heads x 128 x K and V): f16 147,456 B
  (the sealed records log kv_bytes_per_token 147456); u8 78,336 B and u4 41,472 B with scales
  (DERIVED in the strategy report, ~76.5 / ~40.5 KiB).
Model: at the limit, budget B = L x (k_KV + c), with c the non-KV bytes per context token of
the uncapped prefill. Each precision pair gives one (c, B); the three pairs bound c.
Three hypotheses for a step cap T (max_num_batched_tokens), all DERIVED:
- H-act: c is a prefill working set proportional to the step's tokens, so a cap leaves
  T x c of it: L* = (B - T c) / k_KV.
- H-mixed (strategy report): a MIXED-step fp32 scratch of 64 T bytes per context token:
  L* = B / (k_KV + 64 T).
- H-grow: with a cap the KV grows across many steps by reallocate-and-copy, so the peak holds
  about g x KV, g in [1.5, 2]: L* = B / (g k_KV).
Point = H-mixed (the registered report's model); band = the min and max over the hypotheses and
the three (c, B) pairs. The ratio of the pairs is reported too.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

SEALED_XPS = {"f16": (15000, "553b3a5c"), "u8": (20000, "cf3555d5"), "u4": (26500, "e4a22dac")}
K_KV = {"f16": 147_456, "u8": 78_336, "u4": 41_472}
CAPS = (1024, 256)
GROW = (1.5, 2.0)
MIXED_B_PER_T = 64
# T2S: the model window, not memory, bounds the probe (Qwen3-4B max_position_embeddings 40,960).
QWEN3_4B_WINDOW = 40_960
T2S_GPU_VISIBLE_MB = 48_634  # DEMO-2 audit figure, attribution to the T2S unconfirmed (report)
T2S_WEIGHTS_RESERVE_B = 4.0e9  # ASSUMED: weights ~2.4 GB + runtime reserve, rounded up


def pair_fit(a: str, b: str) -> tuple[float, float]:
    la, lb = SEALED_XPS[a][0], SEALED_XPS[b][0]
    c = (lb * K_KV[b] - la * K_KV[a]) / (la - lb)
    return c, la * (K_KV[a] + c)


def fits() -> dict[str, tuple[float, float]]:
    return {f"{a}/{b}": pair_fit(a, b) for a, b in itertools.combinations(SEALED_XPS, 2)}


def limit(model: str, p: str, cap: int, c: float, budget: float, g: float = 1.0) -> float:
    k = K_KV[p]
    if model == "act":
        return (budget - cap * c) / k
    if model == "mixed":
        return budget / (k + MIXED_B_PER_T * cap)
    if model == "grow":
        return budget / (g * k)
    raise ValueError(model)


def xps_limits() -> dict[str, Any]:
    f = fits()
    c_ref, b_ref = f["f16/u4"]
    out: dict[str, Any] = {}
    for p in K_KV:
        for cap in CAPS:
            vals = [limit("act", p, cap, c, b) for c, b in f.values()]
            vals += [limit("mixed", p, cap, c, b) for c, b in f.values()]
            vals += [limit("grow", p, cap, c, b, g) for c, b in f.values() for g in GROW]
            out[f"{p}:c{cap}"] = {
                "point": round(limit("mixed", p, cap, c_ref, b_ref)),
                "band": [round(min(vals)), round(max(vals))],
                "by_hypothesis_f16u4_pair": {
                    "act": round(limit("act", p, cap, c_ref, b_ref)),
                    "mixed": round(limit("mixed", p, cap, c_ref, b_ref)),
                    "grow_g1.5": round(limit("grow", p, cap, c_ref, b_ref, 1.5)),
                    "grow_g2": round(limit("grow", p, cap, c_ref, b_ref, 2.0)),
                },
            }
        out[f"{p}:unl_explicit"] = {
            "point": SEALED_XPS[p][0],
            "band": [SEALED_XPS[p][0] - 1000, SEALED_XPS[p][0] + 1000],
            "basis": "explicit SchedulerConfig with an unlimited step = the sealed plain-pipeline "
            "limit within 2 rungs, if the scheduler object itself changes nothing",
        }
    return out


def ratios(lim: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"sealed_u4_over_f16": round(26500 / 15000, 3)}
    for cap in CAPS:
        u4, f16 = lim[f"u4:c{cap}"], lim[f"f16:c{cap}"]
        vals = []
        for c, b in fits().values():
            for model, g in (("act", 1.0), ("mixed", 1.0), ("grow", 1.5), ("grow", 2.0)):
                vals.append(limit(model, "u4", cap, c, b, g) / limit(model, "f16", cap, c, b, g))
        out[f"c{cap}_u4_over_f16"] = {
            "point": round(u4["point"] / f16["point"], 2),
            "band": [round(min(vals), 2), round(max(vals), 2)],
        }
    return out


def t2s() -> dict[str, Any]:
    c_lo = min(c for c, _ in fits().values())
    c_hi = max(c for c, _ in fits().values())
    budget = T2S_GPU_VISIBLE_MB * 1e6 - T2S_WEIGHTS_RESERVE_B
    ceil = {p: round(budget / (K_KV[p] + c_hi)) for p in K_KV}
    return {
        "uncapped_ceiling_tokens_DERIVED": ceil,
        "holds_at_40960_every_arm": all(v > QWEN3_4B_WINDOW for v in ceil.values()),
        "settled_slope_B_per_token": {
            p: {"point": K_KV[p], "band": [round(0.75 * K_KV[p]), round(1.25 * K_KV[p])]}
            for p in K_KV
        },
        "peak_minus_settled_slope_B_per_token": {
            "unl": {"band": [round(c_lo), round(c_hi)]},
            "c1024": {"band": [0, MIXED_B_PER_T * 1024]},
            "c256": {"band": [0, MIXED_B_PER_T * 256]},
        },
    }


def build() -> dict[str, Any]:
    f = fits()
    lim = xps_limits()
    return {
        "id": "CC_PREDICTIONS",
        "status": "registered with docs/registrations/CC_CONTEXT_CEILING_2026-10-23.md before any "
        "CC run; DERIVED",
        "generator": "tools/cc_predict.py",
        "runner_must_not_read_this_file": True,
        "inputs": {
            "sealed_xps_limits": {
                p: {"tokens": v[0], "run_id": v[1]} for p, v in SEALED_XPS.items()
            },
            "k_kv_B_per_token": K_KV,
            "caps": list(CAPS),
            "grow_factors": list(GROW),
            "mixed_B_per_T_per_context_token": MIXED_B_PER_T,
            "t2s_gpu_visible_mb": T2S_GPU_VISIBLE_MB,
            "t2s_weights_reserve_B_ASSUMED": T2S_WEIGHTS_RESERVE_B,
        },
        "pair_fits": {
            k: {"c_B_per_token": round(c), "budget_GiB": round(b / 2**30, 3)}
            for k, (c, b) in f.items()
        },
        "xps_resident_limits": lim,
        "xps_ratios": ratios(lim),
        "xps_gate": {
            "rule": "mechanism gate (F1): HIT if the u4 c256 limit >= 40,000 tokens (1.5x the "
            "sealed 26,500) in the median of r=3 limit-point checks",
            "predicted": "HIT",
        },
        "t2s": t2s(),
        "cliff_xps": {
            "below_limit": "every point <= sealed L* holds; TTFT grows with n (exponent 1.8-2.4)",
            "above_limit": "points >= L* + 500 fail LOUD (CL_OUT_OF_RESOURCES or an allocation "
            "error) in turn 1, as all 3 sealed limit probes did; no point above L* holds with a "
            ">10x TTFT step (no silent paging in this probe)",
        },
        "scoring": "a limit is HIT if the measured largest holding n lies in band; a ratio is HIT "
        "if inside its band; a TIME_BOUND probe (child timeout with no allocation or OpenCL error) "
        "is not a memory limit and leaves that cell UNRESOLVED, not MISS",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    doc = build()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=2) + "\n", encoding="ascii")
    print(json.dumps({"pair_fits": doc["pair_fits"], "ratios": doc["xps_ratios"]}, indent=1))
    for k, v in doc["xps_resident_limits"].items():
        print(k, v["point"], v["band"])
    print(json.dumps(doc["t2s"]["uncapped_ceiling_tokens_DERIVED"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
