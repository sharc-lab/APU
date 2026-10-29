"""Fig 3: held-out slopes. Ceiling vs GPU-shared balloon size (A22), predicted lines through the
sealed limits, with the measured balloon and slider points when present; slope ratio 1.767."""

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


def make(out_dir: Path) -> dict[str, Any]:
    plt = setup()
    inp = Inputs()
    bm = inp.json(CC_DIR / "CC_BM_PREDICTIONS.json")
    sealed = sealed_limits(inp)
    meas = measured(inp).get("balloon", {})
    slope = bm["A22_balloon"]["gpu_dL_per_GiB"]
    base = {
        "f16": sealed["553b3a5c"]["largest_n_cached"],
        "u4": sealed["e4a22dac"]["largest_n_cached"],
    }
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    xs = [0, 1, 2]
    for p, c in (("f16", "C0"), ("u4", "C2")):
        ax.plot(
            xs,
            [base[p] + slope[p] * x for x in xs],
            color=c,
            label=f"{p} predicted ({slope[p]}/GiB)",
        )
        pts = meas.get(p)
        if pts:
            ax.scatter(
                [q["gib"] for q in pts], [q["limit"] for q in pts], color=c, edgecolor="k", zorder=5
            )
        else:
            placeholder(ax, 1, base[p] + slope[p], f"{p} balloon")
    ax.set_xlabel("GPU-shared balloon (GiB)")
    ax.set_ylabel("resident limit (tokens)")
    ax.legend(fontsize=6)
    ax.set_title(
        f"Balloon slopes, ratio {bm['A22_balloon']['slope_ratio_u4_over_f16']} (DERIVED)",
        fontsize=8,
    )
    return save(
        fig, "fig3_slopes", inp, out_dir, ["lines DERIVED (A22); points from the weekend closeout"]
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    make(ap.parse_args(argv).out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
