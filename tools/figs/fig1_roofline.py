"""Fig 1: the Context Roofline. L_mem = B / (k + a) vs GPU-visible budget B, per KV precision,
with the KV-only lines (B / k), the Qwen3-4B window, and the sealed XPS limits (in-sample)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from tools.figs._common import (
    CC_DIR,
    DEFAULT_OUT,
    Inputs,
    measured,
    placeholder,
    save,
    sealed_limits,
    setup,
)

GIB = 2**30


def make(out_dir: Path) -> dict[str, Any]:
    plt = setup()
    inp = Inputs()
    bl = inp.json(CC_DIR / "CC_BL_PREDICTIONS.json")
    base = inp.json(CC_DIR / "CC_PREDICTIONS.json")
    a = bl["hypotheses"]["H_act"]["a_B_per_token"]
    b_fit = bl["hypotheses"]["H_act"]["budget_GiB"]
    k = base["inputs"]["k_kv_B_per_token"]
    sealed = sealed_limits(inp)
    meas = measured(inp)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    bs = [i / 10 for i in range(1, 121)]
    for p, c in (("f16", "C0"), ("u8", "C1"), ("u4", "C2")):
        ax.plot(bs, [b * GIB / (k[p] + a) for b in bs], color=c, label=f"{p}: B/(k+a)")
        ax.plot(
            bs, [b * GIB / k[p] for b in bs], color=c, ls=":", lw=0.8, label=f"{p}: KV-only B/k"
        )
    ax.axhline(40960, color="k", lw=0.8, ls="--", label="Qwen3-4B window 40,960")
    for rid, s in sealed.items():
        ax.scatter([b_fit], [s["largest_n_cached"]], color="k", zorder=6, s=12)
        ax.annotate(
            f"{rid} ({s['precision']}, in-sample)",
            (b_fit, s["largest_n_cached"]),
            fontsize=6,
            xytext=(4, -2),
            textcoords="offset points",
        )
    a19 = meas.get("a19_budget_GiB")
    if a19 is None:
        placeholder(ax, 7.0, 30000, "A19 measured budget")
    ax.set_xlabel("GPU-visible budget for context B (GiB)")
    ax.set_ylabel("resident context (tokens)")
    ax.set_ylim(0, 120000)
    ax.legend(fontsize=6, ncol=2)
    ax.set_title("Context Roofline, Qwen3-4B INT4 (a fitted; DERIVED)", fontsize=8)
    return save(
        fig,
        "fig1_roofline",
        inp,
        out_dir,
        ["a and B fitted to the f16/u4 pair: the sealed points are in-sample"],
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    make(ap.parse_args(argv).out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
