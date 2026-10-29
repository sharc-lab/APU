"""CC amendment (Task BL): DERIVED secondary predictions, before any CC run.

Writes derived/cc/CC_BL_PREDICTIONS.json, registered by docs/registrations/CC_AMEND_2026-10-23b.md.
It adds to derived/cc/CC_PREDICTIONS.json and changes none of it (the primary u4 c256 63,319
[36,239, 99,204] and the gate >= 40,000 stay as registered).

Inputs: the sealed XPS limits and k_KV from tools/cc_predict.py. Hypotheses (all DERIVED):
- H-act: a linear prefill working set a B per context token in the uncapped step, a and the
  budget B from the f16/u4 pair (a = 96,768); a cap T leaves T x a: L* = (B - T a) / k_KV.
- H-quad: a term c x L^2 that the cap does not remove; c and B from the f16/u4 pair
  (c ~ 2.33 B per token^2); capped limits stay near the uncapped ones.
- MIXED (64 T per context token) is retired for the sealed limits and kept only for CC-GROW,
  where each turn appends T = 2,048 tokens to a resident session.
T2S tier 48,124 MiB (Rithwik's pilot load failure, PILOT) minus ASSUMED 2.5 GiB weights and
1.5 GiB runtime reserve.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import cc_predict as base  # noqa: E402

GIB = 2**30
L16, L4 = base.SEALED_XPS["f16"][0], base.SEALED_XPS["u4"][0]
K = base.K_KV
T2S_TIER_MIB = 48_124
T2S_WEIGHTS_GIB = 2.5
T2S_RESERVE_GIB = 1.5
CONC_TOKENS = 32_768
GROW_T = 2_048


def h_act() -> tuple[float, float]:
    a, budget = base.pair_fit("f16", "u4")
    return a, budget


def h_quad() -> tuple[float, float]:
    c = (L16 * K["f16"] - L4 * K["u4"]) / (L4**2 - L16**2)
    return c, L16 * K["f16"] + c * L16**2


def act_limit(p: str, cap: int) -> float:
    a, budget = h_act()
    return (budget - cap * a) / K[p]


def mixed_limit(p: str, t: int) -> float:
    _, budget = h_act()
    return budget / (K[p] + base.MIXED_B_PER_T * t)


def t2s() -> dict[str, Any]:
    a, _ = h_act()
    avail = T2S_TIER_MIB * 2**20 - (T2S_WEIGHTS_GIB + T2S_RESERVE_GIB) * GIB
    kv = CONC_TOKENS * K["f16"]
    act = CONC_TOKENS * (K["f16"] + a)
    single = {p: round(avail / (K[p] + a)) for p in K}
    return {
        "available_for_context_GiB": round(avail / GIB, 2),
        "single_session_ceiling_uncapped_tokens": single,
        "window_headroom_f16": round(single["f16"] / 40_960, 2),
        "cc_window": "40,448 holds at f16 / u8 / u4 uncapped (r = 3); the window, not memory, binds",
        "byte_slope_B_per_token": {
            "uncapped_band": [round(0.75 * a), round(1.25 * a)],
            "capped_max": round(0.25 * a),
            "basis": "uncapped within +-25% of the XPS fit a; capped about 0 (<= 0.25 a)",
        },
        "conc": {
            "per_session_kv_GiB": round(kv / GIB, 2),
            "ii_dynamic_cap256_fails_at_N": int(avail // kv) + 1,
            "ii_N_star": round(avail / kv, 2),
            "i_unbounded_simultaneous_fails_at_N": int(avail // act) + 1,
            "i_N_star": round(avail / act, 2),
            "iii_fixed_40GiB_cap256_preemption_from_N": int(40 * GIB // kv) + 1,
            "iv_fixed_50GiB": "construction fails (50 GiB > the 48,124 MiB tier)",
            "u4_capped_N_star_not_reached": round(avail / (CONC_TOKENS * K["u4"]), 1),
        },
        "slider_A15": "each +4.5 GiB of tier adds one capped 32,768-token f16 session",
    }


def build() -> dict[str, Any]:
    a, b_act = h_act()
    c, b_quad = h_quad()
    act = {f"{p}:c{cap}": round(act_limit(p, cap)) for p in K for cap in (256, 1024)}
    ratio = act_limit("u4", 256) / act_limit("f16", 256)
    per_gib = {
        "f16_default": round(GIB / (K["f16"] + a)),
        "f16_bounded": round(GIB / K["f16"]),
        "u4_default": round(GIB / (K["u4"] + a)),
        "u4_bounded": round(GIB / K["u4"]),
    }
    return {
        "id": "CC_BL_PREDICTIONS",
        "status": "registered with docs/registrations/CC_AMEND_2026-10-23b.md before any CC run; "
        "DERIVED; adds to CC_PREDICTIONS.json, which is unchanged",
        "generator": "tools/cc_bl_predict.py",
        "runner_must_not_read_this_file": True,
        "hypotheses": {
            "H_act": {"a_B_per_token": round(a), "budget_GiB": round(b_act / GIB, 3)},
            "H_quad": {"c_B_per_token2": round(c, 3), "budget_GiB": round(b_quad / GIB, 3)},
            "MIXED_64T": "retired for the sealed limits; CC-GROW only",
        },
        "cc_mech_xps_secondary": {
            "H_act_limits": act,
            "H_act_capped_u4_over_f16": round(ratio, 2),
            "H_quad": "each capped limit within 10% of the same precision's uncapped limit",
            "report_u8_capped_band": [46_000, 48_000],
            "report_u8_note": "the source report gives u8 capped 46-48K; this generator gives "
            "c256 and c1024 above; the decision rule does not use u8",
            "rule_H_act_confirmed": "f16 c256 >= 21,000 AND u4 c256 >= 60,000 AND |c1024 - c256| "
            "<= 0.10 x c256 (per precision measured)",
            "rule_H_act_refuted": "u4 c256 <= 29,000",
            "primary_unchanged": "CC_PREDICTIONS.json u4:c256 63,319 [36,239, 99,204]; gate >= 40,000",
        },
        "cc_ttft_xps": {"u4_c256_penalty_at_16384_max": 0.15},
        "cc_grow_xps": {
            "H_act_kv_bound": {
                "f16": round(act_limit("f16", 256)),
                "u4": round(act_limit("u4", 256)),
            },
            "MIXED_T2048": {
                "f16_unl": round(mixed_limit("f16", GROW_T)),
                "u4_unl": round(mixed_limit("u4", GROW_T)),
                "u4_c256": round(mixed_limit("u4", 256)),
            },
        },
        "cc_rt_xps": {"q4_0_over_f16_ceiling_min": 3.0, "openvino_default_sealed": 1.77},
        "cc_valid_xps": {
            "anchor_rule": "2507 uncapped f16 and u4 within 1,000 tokens of 15,000 and 26,500",
            "u4_c256_L_valid_min": 40_000,
        },
        "t2s": t2s(),
        "tokens_per_GiB": per_gib,
        "slider_A18_xps": {
            "f16_unl_tokens_per_GiB": per_gib["f16_default"],
            "plus_4p5_GiB_from_15000": round(L16 + 4.5 * per_gib["f16_default"]),
        },
        "scoring": "each item HIT/MISS by its own rule; censored (timeout, no device error) "
        "probes leave a limit UNRESOLVED, not MISS",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    doc = build()
    a.out.write_text(json.dumps(doc, indent=2) + "\n", encoding="ascii")
    print(json.dumps({k: doc[k] for k in ("hypotheses", "cc_mech_xps_secondary", "t2s")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
