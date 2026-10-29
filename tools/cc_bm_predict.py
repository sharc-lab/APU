"""CC amendment c (Task BM, A19-A36): DERIVED predictions for the Context Roofline cells.

Writes derived/cc/CC_BM_PREDICTIONS.json, registered by docs/registrations/CC_AMEND_2026-10-23c.md.
Adds to CC_PREDICTIONS.json and CC_BL_PREDICTIONS.json, which are unchanged.

Inputs: H-act a and budget B (tools/cc_bl_predict, f16/u4 pair of the sealed XPS limits);
k per token (tools/cc_predict); the TTFT law F(L) = 7.2e9 L + 2.95e5 L^2 FLOP with F_eff
back-solved from the sealed 10,000-token / 10 s point (c2246b1f); the T2S context headroom
of CC_BL (43.0 GiB). Source prediction texts are recorded verbatim beside the arithmetic.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import cc_bl_predict as bl  # noqa: E402
from tools import cc_predict as base  # noqa: E402

GIB = 2**30
MIB = 2**20
K = base.K_KV
K8_NO_SCALES = 73_728  # 8-bit KV bytes per token without scales (A21 band edge)
FLOP_LIN = 7.2e9
FLOP_QUAD = 2.95e5
FLOP_TURN_CTX = 589_824  # 4 x 36 layers x 4,096 query dims, per context token per new token
SEALED_TTFT_TOKENS = 10_000  # c2246b1f, 10 s SLO
SLO_S = 10.0


def f_full(n: float) -> float:
    return FLOP_LIN * n + FLOP_QUAD * n * n


def f_eff() -> float:
    return f_full(SEALED_TTFT_TOKENS) / SLO_S


def l_ttft(seconds: float) -> float:
    c = -f_eff() * seconds
    return (-FLOP_LIN + math.sqrt(FLOP_LIN**2 - 4 * FLOP_QUAD * c)) / (2 * FLOP_QUAD)


def turn_ttft(delta: float, ctx: float) -> float:
    return delta * (FLOP_LIN + FLOP_TURN_CTX * ctx) / f_eff()


def slider_slope(m_bytes: float, k: float, a: float) -> float:
    return 0.01 * m_bytes / (k + a)


def build() -> dict[str, Any]:
    a, budget = bl.h_act()
    per_gib = {p: GIB / (K[p] + a) for p in ("f16", "u4")}
    t2s = bl.t2s()
    avail = t2s["available_for_context_GiB"]
    kv32k = bl.CONC_TOKENS * K["f16"] / GIB
    # A33: explicit cache_size S GiB, u4. GenAI 2026.2.1.0 counts block bytes with the
    # compiled element type's size(), which rounds u4 up to 1 byte per element (two per byte
    # really stored), so a budget holds half the u4 tokens its bytes could.
    s_gib = 2
    counted = 2 * K["u4"]
    kv_cap = s_gib * GIB / counted
    act_cap = (budget - kv_cap * K["u4"]) / a
    slider = {
        f"M_{m}GiB": {
            "f16_law": round(slider_slope(m * GIB, K["f16"], a)),
            "u4_law": round(slider_slope(m * GIB, K["u4"], a)),
            "f16_kv_only": round(slider_slope(m * GIB, K["f16"], 0)),
            "u4_kv_only": round(slider_slope(m * GIB, K["u4"], 0)),
        }
        for m in (15.5, 16.0)
    }
    return {
        "id": "CC_BM_PREDICTIONS",
        "status": "registered with docs/registrations/CC_AMEND_2026-10-23c.md before any CC run; "
        "DERIVED; adds to CC_PREDICTIONS.json and CC_BL_PREDICTIONS.json (unchanged)",
        "generator": "tools/cc_bm_predict.py",
        "runner_must_not_read_this_file": True,
        "inputs": {"a_B_per_token": round(a), "budget_GiB": round(budget / GIB, 3), "k": K},
        "A19_shared_system_memory": {
            "T2S": "SharedSystemMemory = ullTotalPhys - 16,384 MiB (+-256 MiB)",
            "XPS": "SharedSystemMemory = 0.50 x ullTotalPhys (+-256 MiB), or slider x ullTotalPhys "
            "if the Intel override is set",
            "tolerance_MiB": 256,
            "t2s_48124_check": "48,124 MiB + 16,384 MiB = 64,508 MiB of OS-visible memory implied",
        },
        "A20_allocator_a": {
            "predict_kB_per_token": [93, 98],
            "falsified_if_kB": "< 70 or > 130, or f16 vs u4 slopes differ > 10%",
            "fitted_a_B": round(a),
        },
        "A21_8bit_rebisect": {
            "predict": [round(budget / (K["u8"] + a)), round(budget / (K8_NO_SCALES + a))],
            "law_fails_outside": [19_900, 22_600],
            "note": "8-bit arm requests i8 (the sealed u8 request compiled i8, CONFIG_PROVENANCE)",
        },
        "A22_balloon": {
            "gpu_dL_per_GiB": {"f16": -round(per_gib["f16"]), "u4": -round(per_gib["u4"])},
            "slope_ratio_u4_over_f16": round(per_gib["u4"] / per_gib["f16"], 3),
            "ratio_band": [1.59, 1.95],
            "cpu_balloon_abs_dL_max_frac": 0.05,
        },
        "A23_grow_ttft": {
            "formula": "turn TTFT = delta x (7.2e9 + 589,824 L) / F_eff",
            "F_eff_flop_per_s": round(f_eff()),
            "example_L20000_delta2048_s": round(turn_ttft(2048, 20_000), 2),
            "failure": "device_alloc at L_mem, not TTFT > 10 s, for delta <= 2K",
        },
        "A24_ttft_law": {
            "L_TTFT": {f"{s}s": round(l_ttft(s)) for s in (5, 20, 30)},
            "L_TTFT_20s_tolerance": 0.15,
            "sealed_10000_point": "c2246b1f: cold single generate, one child per repeat "
            "(derived/audit/CONFIG_PROVENANCE.md), i.e. full-prompt prefill",
        },
        "A25_A36_slider": {
            "rule": "slope tokens per point = 0.01 x M / (k + a), M = measured ullTotalPhys; "
            "print the KV-only 0.01 x M / k beside it; never a bare 681",
            "at": slider,
            "ratio": round((K["f16"] + a) / (K["u4"] + a), 3),
            "falsified_if": "R^2 < 0.98, or x-intercepts differ > 5% by precision, or ratio "
            "outside 1.77 +- 10%",
        },
        "A26_energy": "fit E(N) = a N + b N^2 + c, J per token; no numeric prediction",
        "A27_residency_power": "DischargeRate delta (~3 GiB KV resident vs none) within meter "
        "resolution",
        "A29_t2s_balloon_conc_ii": {
            f"balloon_{g}GiB_fails_at_N": int((avail - g) // kv32k) + 1 for g in (0, 4.5, 9)
        },
        "A30_ollama_default": {
            "XPS": 4096,
            "T2S": 32768,
            "basis": "Ollama VRAM tiers; T2S 47.0 GiB < 48 GiB",
        },
        "A33_cache_size": {
            "source_prediction": "u4 uncapped ceilings equal (+-256) with cache_size 0 and "
            "cache_size set (activation law)",
            "cache_size_GiB": s_gib,
            "source_read": "openvino.genai 2026.2.1.0: the budget assertion runs only when "
            "cache_size > 0; block bytes use the compiled element type's size()",
            "derived_counted_u4_B_per_token": counted,
            "derived_kv_capacity_tokens": round(kv_cap),
            "derived_activation_bound_tokens": round(act_cap),
            "derived_ceiling": round(min(kv_cap, act_cap)),
            "derived_note": "the source prediction is expected to MISS by about 600 tokens: a "
            "fixed 2 GiB cache holds only about 25,900 u4 tokens under 1-byte accounting",
        },
        "A34_cap256": {
            "bookkeeping": "no ceiling change at max_num_batched_tokens 256",
            "activation_law": {"f16_c256": round(bl.act_limit("f16", 256)), "u4_c256_min": 29_000},
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    doc = build()
    a.out.write_text(json.dumps(doc, indent=2) + "\n", encoding="ascii")
    print(json.dumps(doc, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
