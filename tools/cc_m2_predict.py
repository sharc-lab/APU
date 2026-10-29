"""Second-model cells (Task BO, Qwen3-8B INT4 on the XPS): DERIVED predictions before any run.

Writes derived/cc/CC_M2_PREDICTIONS.json, registered by docs/registrations/CC_AMEND_2026-10-23d_MODEL2.md.

Config (huggingface.co/Qwen/Qwen3-8B config.json, read 2026-10-23): 36 layers, hidden 4,096,
intermediate 12,288, 32 query heads, 8 KV heads, head_dim 128, max_position_embeddings 40,960,
rope_scaling null, untied embeddings. KV shapes equal Qwen3-4B's, so k per precision is the same.
Config-predicted a: the same rule that gives 98,304 B/token for Qwen3-4B, one layer's f32 FFN
gate + up buffers plus two f32 hidden buffers: a = 4 x (2 x intermediate + 2 x hidden).
B: the XPS budget for Qwen3-4B fitted from the sealed f16/u4 pair (H-act) minus the weight-size
difference of the two IRs (configs/models/*.yaml ir_bytes). The fixed overhead c0 and the OS
tier cancel, so no refit is needed. The substitution rule for the measured A19 value is in the
output.
Bands: +-15% (the held-out rule, PLAN_LOCK R8).
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

from tools import cc_bl_predict as bl  # noqa: E402
from tools import cc_predict as base  # noqa: E402

GIB = 2**30
CFG_8B = {
    "num_hidden_layers": 36,
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "max_position_embeddings": 40960,
}
CFG_4B = {"hidden_size": 2560, "intermediate_size": 9728}
W_4B_IR = 2_290_768_181  # configs/models/Qwen3-4B-int4-ov.yaml ir_bytes
W_8B_IR = 4_882_865_352  # configs/models/Qwen3-8B-int4-ov.v2.yaml ir_bytes
K = {"f16": base.K_KV["f16"], "i8": base.K_KV["u8"], "u4": base.K_KV["u4"]}
BAND = 0.15


def a_config(cfg: dict[str, int]) -> int:
    return 4 * (2 * cfg["intermediate_size"] + 2 * cfg["hidden_size"])


def budget_8b() -> float:
    _, b4 = bl.h_act()
    return b4 - (W_8B_IR - W_4B_IR)


def band(x: float) -> list[int]:
    return [round(x * (1 - BAND)), round(x * (1 + BAND))]


def build() -> dict[str, Any]:
    a8 = a_config({**CFG_8B})
    a4_cfg = a_config(CFG_4B)
    a4_fit, b4 = bl.h_act()
    b8 = budget_8b()
    unc = {p: b8 / (K[p] + a8) for p in K}
    cap = {p: (b8 - 256 * a8) / K[p] for p in ("f16", "u4")}
    return {
        "id": "CC_M2_PREDICTIONS",
        "status": "registered with docs/registrations/CC_AMEND_2026-10-23d_MODEL2.md before any "
        "Qwen3-8B CC run; DERIVED",
        "generator": "tools/cc_m2_predict.py",
        "runner_must_not_read_this_file": True,
        "inputs": {
            "config_8b": CFG_8B,
            "k_B_per_token": K,
            "a_rule": "4 x (2 x intermediate + 2 x hidden) bytes per token",
            "a_rule_on_4b": a4_cfg,
            "a_fitted_4b": round(a4_fit),
            "budget_4b_GiB": round(b4 / GIB, 3),
            "weights_ir_bytes": {"4b": W_4B_IR, "8b": W_8B_IR},
            "budget_8b_GiB": round(b8 / GIB, 3),
            "budget_rule": "B_8B = B_4B(fit) - (W_8B - W_4B). If the A19 record at the M2 run shows a "
            "different GPU-shared tier S than the A19 record at the CC v3 queue start, use "
            "B_8B + (S_m2 - S_v3) with no refit",
        },
        "alloc_slope_a": {
            "f16": {"point": a8, "band": band(a8)},
            "u4": {"point": a8, "band": band(a8)},
            "f16_u4_max_diff": 0.10,
        },
        "uncapped_limits": {p: {"point": round(v), "band": band(v)} for p, v in unc.items()},
        "u4_cap256_limit": {"point": round(cap["u4"]), "band": band(cap["u4"])},
        "f16_cap256_limit_info": round(cap["f16"]),
        "kv_compression_bound": {
            "G_max": round((K["f16"] + a8) / a8, 3),
            "basis": "(k_f16 + a) / a",
        },
        "u4_over_f16_uncapped": round(unc["u4"] / unc["f16"], 3),
        "scoring": "HIT if the measured value lies in its band; a censored (timeout) probe leaves "
        "the cell UNRESOLVED",
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
