"""Fig 6: Windows graphics-memory formula vs installed memory, with the T2S 48,124 MiB point
(PILOT, back-solved until A19) and the A19 records when present."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from tools.figs._common import DEFAULT_OUT, Inputs, measured, placeholder, save, setup


def tier(m: float) -> float:
    return min(0.8 * m, max(m - 16.0, 0.5 * m))


def make(out_dir: Path) -> dict[str, Any]:
    plt = setup()
    inp = Inputs()
    a19 = measured(inp).get("a19", {})
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ms = [m / 2 for m in range(8, 257)]
    ax.plot(ms, [tier(m) for m in ms], color="C0", label="MIN(80%, MAX(M-16, 50%))")
    ax.plot(ms, [0.5 * m for m in ms], color="0.6", ls=":", label="vendor FAQ 50%")
    ax.scatter([64508 / 1024], [48124 / 1024], color="C3", zorder=5, label="T2S 48,124 MiB (PILOT)")
    for host, rec in a19.items():
        ax.scatter([rec["total_GiB"]], [rec["shared_GiB"]], color="k", zorder=6)
        ax.annotate(f"{host} A19", (rec["total_GiB"], rec["shared_GiB"]), fontsize=6)
    if not a19:
        placeholder(ax, 15.5, 7.75, "XPS A19")
    ax.set_xlabel("OS-visible memory M (GiB)")
    ax.set_ylabel("graphics-usable tier (GiB)")
    ax.legend(fontsize=6)
    ax.set_title("The OS formula sets the tier", fontsize=8)
    return save(
        fig,
        "fig6_os_formula",
        inp,
        out_dir,
        ["the T2S point is a consistency check until A19 records ullTotalPhys"],
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    make(ap.parse_args(argv).out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
