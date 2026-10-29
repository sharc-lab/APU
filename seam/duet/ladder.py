"""Run the DUET ladder on synthetic shapes U/H/T for Sonnet/Haiku/Opus; write a markdown table."""

from __future__ import annotations

from pathlib import Path

from seam.duet import params as prm
from seam.duet.policies import simulate
from seam.duet.shapes import shape_h, shape_t, shape_u

RUNGS = ("D0", "D1-stateless", "D1-resident", "D3", "DUET-HW", "DUET-full", "D1-bg", "D3-bg",
         "config-only-bg", "DUET-bg")  # fmt: skip
SHAPES = {"U": shape_u, "H": shape_h, "T": shape_t}
PRICES = (prm.SONNET, prm.HAIKU, prm.OPUS)


def table() -> str:
    lines = [
        "# DUET ladder on SYNTHETIC shapes (no trace data; $ per 1,000 sessions; DERIVED)",
        "",
        "Shapes U/H/T are ASSUMED (seam/duet/shapes.py). Constants are PLACEHOLDER until the Nov 22 freeze.",
        "",
        "| price | shape | " + " | ".join(RUNGS) + " |",
        "|---|---|" + "---|" * len(RUNGS),
    ]
    for p in PRICES:
        for name, fn in SHAPES.items():
            vals = [f"{simulate(r, fn(), p).usd * 1000:.2f}" for r in RUNGS]
            lines.append(f"| {p.name} | {name} | " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def write(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(table(), encoding="ascii")


if __name__ == "__main__":
    write(Path("derived/duet/LADDER_SYNTHETIC_2026-10-29.md"))
