"""Fig 7: which ceiling binds, per device and regime (DERIVED map; cells turn measured as the
closeout fills analysis/cc_20261026/measured.json "binding")."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from tools.figs._common import DEFAULT_OUT, Inputs, measured, save, setup

ROWS = [
    ("16 GB iGPU, full prefill, 10 s SLO", "TTFT", "DERIVED; 10,000 SEALED c2246b1f"),
    ("16 GB iGPU, cached KV, delta <= 2K", "memory", "DERIVED; CC-GROW"),
    ("16 GB, CPU path", "TTFT", "468 SEALED 7f232f86"),
    ("64 GB, one session", "window", "DERIVED; CC-WINDOW"),
    ("64 GB, N x 32K sessions", "memory (aggregate)", "DERIVED; CC-CONC"),
    ("32 GB iGPU at 50%", "window", "DERIVED"),
    ("8 GB iGPU at 50%", "does not load", "ASSUMED c0"),
    ("RTX 4070 laptop", "spill ramp", "PILOT (Rithwik)"),
]
COLORS = {
    "TTFT": "C1",
    "memory": "C0",
    "memory (aggregate)": "C9",
    "window": "C2",
    "does not load": "C3",
    "spill ramp": "C4",
}


def make(out_dir: Path) -> dict[str, Any]:
    plt = setup()
    inp = Inputs()
    meas = measured(inp).get("binding", {})
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    ax.axis("off")
    for i, (row, bind, src) in enumerate(ROWS):
        got = meas.get(row)
        y = len(ROWS) - i
        ax.text(0.0, y, row, fontsize=7, va="center")
        ax.text(0.55, y, bind, fontsize=7, va="center", color=COLORS.get(bind, "k"), weight="bold")
        ax.text(0.75, y, (f"measured: {got}" if got else src), fontsize=6, va="center", color="0.3")
    ax.set_ylim(0, len(ROWS) + 1)
    ax.set_xlim(0, 1)
    ax.set_title("Which ceiling binds (predicted; measured where available)", fontsize=8)
    return save(
        fig, "fig7_binding_map", inp, out_dir, ["rows from the award report's regime map (DERIVED)"]
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    make(ap.parse_args(argv).out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
