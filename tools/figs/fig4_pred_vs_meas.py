"""Fig 4: predicted vs measured ceilings, law vs KV-only baseline, with the +-15% band.
Sealed XPS points are in-sample (open markers); sealed 8B points are retrospective; weekend
held-out cells come from the closeout file."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from tools.figs._common import (
    CC_DIR,
    DEFAULT_OUT,
    ROOT,
    Inputs,
    measured,
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
    m2 = inp.json(CC_DIR / "CC_M2_PREDICTIONS.json")
    sealed = sealed_limits(inp)
    k = base["inputs"]["k_kv_B_per_token"]
    a = bl["hypotheses"]["H_act"]["a_B_per_token"]
    b = bl["hypotheses"]["H_act"]["budget_GiB"] * GIB
    s8 = {
        "322b2f86": (
            "f16",
            inp.json(
                ROOT / "derived/c2_ttft/sealed_322b2f86-9571-46ae-be6a-ba1cec44e018/summary.json"
            ),
        ),
        "69234ed5": (
            "i8",
            inp.json(
                ROOT / "derived/c2_ttft/sealed_69234ed5-a716-4274-b851-33509bdb9972/summary.json"
            ),
        ),
    }
    held = measured(inp).get("heldout", [])
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    lim = 100000
    ax.plot([0, lim], [0, lim], color="k", lw=0.8)
    ax.fill_between([0, lim], [0, 0.85 * lim], [0, 1.15 * lim], color="0.9", label="+-15%")
    kk = {"f16": k["f16"], "u8->i8": k["u8"], "u4": k["u4"]}
    # KV-only baseline fitted the same way the law was: its budget from the sealed f16 point.
    b_kv = sealed["553b3a5c"]["largest_n_cached"] * k["f16"]
    for rid, s in sealed.items():
        p = s["precision"]
        ax.scatter(
            [b / (kk[p] + a)],
            [s["largest_n_cached"]],
            facecolor="none",
            edgecolor="C0",
            label="sealed 4B (in-sample)" if rid == "553b3a5c" else None,
        )
        ax.scatter(
            [b_kv / kk[p]],
            [s["largest_n_cached"]],
            marker="s",
            facecolor="none",
            edgecolor="C3",
            label="KV-only baseline (B fit to f16)" if rid == "553b3a5c" else None,
        )
    for rid, (p, summ) in s8.items():
        meas8 = summ["arm_results"][0]["ceiling_n_cached"]
        ax.scatter(
            [m2["uncapped_limits"][p]["point"]],
            [meas8],
            marker="^",
            color="C4",
            label="sealed 8B (retrospective)" if rid == "322b2f86" else None,
        )
    for h in held:
        ax.scatter(
            [h["predicted"]],
            [h["measured"]],
            color="C2",
            edgecolor="k",
            label="held-out" if h is held[0] else None,
        )
    if not held:
        ax.text(
            0.05 * lim, 0.9 * lim, "held-out cells: weekend (placeholder)", fontsize=6, color="0.4"
        )
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("predicted ceiling (tokens)")
    ax.set_ylabel("measured ceiling (tokens)")
    ax.legend(fontsize=6, loc="lower right")
    ax.set_title("Law vs KV-only, predicted vs measured", fontsize=8)
    return save(
        fig,
        "fig4_pred_vs_meas",
        inp,
        out_dir,
        ["only held-out points test the law; sealed 4B points are in-sample, 8B retrospective"],
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    make(ap.parse_args(argv).out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
