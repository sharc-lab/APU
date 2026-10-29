"""Fig 5: the KV-compression bound. Context gain G(s) = (k + a) / (k/s + a) vs KV compression s,
for Qwen3-4B (a fitted) and Qwen3-8B (a from config); asymptote (k + a) / a."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from tools.figs._common import CC_DIR, DEFAULT_OUT, Inputs, save, setup


def make(out_dir: Path) -> dict[str, Any]:
    plt = setup()
    inp = Inputs()
    bl = inp.json(CC_DIR / "CC_BL_PREDICTIONS.json")
    base = inp.json(CC_DIR / "CC_PREDICTIONS.json")
    m2 = inp.json(CC_DIR / "CC_M2_PREDICTIONS.json")
    k = base["inputs"]["k_kv_B_per_token"]["f16"]
    a4 = bl["hypotheses"]["H_act"]["a_B_per_token"]
    a8 = m2["alloc_slope_a"]["f16"]["point"]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ss = [1 + i / 10 for i in range(0, 151)]
    for a, name, c in ((a4, "Qwen3-4B", "C0"), (a8, "Qwen3-8B", "C1")):
        ax.plot(
            ss,
            [(k + a) / (k / s + a) for s in ss],
            color=c,
            label=f"{name}: max {(k + a) / a:.2f}x",
        )
        ax.axhline((k + a) / a, color=c, ls=":", lw=0.8)
    ax.plot(ss, ss, color="0.6", lw=0.8, ls="--", label="KV-only (gain = s)")
    ax.scatter(
        [k / 41472], [26500 / 15000], color="k", zorder=5, label="sealed u4 1.77x (in-sample)"
    )
    ax.set_ylim(1, 4)
    ax.set_xlabel("KV bytes compression s (f16 = 1)")
    ax.set_ylabel("context gain")
    ax.legend(fontsize=6)
    ax.set_title("What KV compression can buy (unchunked prefill; DERIVED)", fontsize=8)
    return save(
        fig, "fig5_kv_bound", inp, out_dir, ["the 1.77x point is in-sample (a fitted to it)"]
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    make(ap.parse_args(argv).out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
