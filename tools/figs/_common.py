"""Shared helpers for the meeting-pack figures (Task BO).

Inputs are only sealed run records (derived/**/sealed_*/summary.json) and committed prediction
JSONs (derived/cc/*.json). Weekend cells come from analysis/cc_20261026/measured.json once Monday's
closeout writes it; until then each figure draws labelled placeholders. Every figure writes PNG and
PDF plus a manifest with the sha256 of every input it read.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CC_DIR = ROOT / "derived" / "cc"
MEASURED = ROOT / "analysis" / "cc_20261026" / "measured.json"
DEFAULT_OUT = ROOT / "analysis" / "meeting_20261027" / "figs"
SEALED_LIMITS = {  # run_id -> (precision, sealed dir)
    "553b3a5c": ("f16", "derived/delta_prefill/sealed_553b3a5c-dd68-4502-bb3f-1b6a10a61eb1"),
    "cf3555d5": ("u8->i8", "derived/delta_prefill/sealed_cf3555d5-1361-434f-b1fb-745fc17e6843"),
    "e4a22dac": ("u4", "derived/delta_prefill/sealed_e4a22dac-6301-4dfa-ae12-882328b6ba7c"),
}


class Inputs:
    """Tracks every file read so the manifest can hash it."""

    def __init__(self) -> None:
        self.paths: list[Path] = []

    def json(self, path: Path) -> Any:
        self.paths.append(path)
        return json.loads(path.read_text(encoding="utf-8"))

    def manifest(self) -> dict[str, str]:
        out = {}
        for p in self.paths:
            h = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "absent"
            out[str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)] = h
        return out


def sealed_limits(inp: Inputs) -> dict[str, dict[str, Any]]:
    out = {}
    for rid, (prec, rel) in SEALED_LIMITS.items():
        s = inp.json(ROOT / rel / "summary.json")
        out[rid] = {"precision": prec, "largest_n_cached": s.get("largest_n_cached")}
    return out


def measured(inp: Inputs) -> dict[str, Any]:
    """Monday's closeout values (with run_ids), or {} so figures draw placeholders."""
    if MEASURED.is_file():
        return inp.json(MEASURED)
    inp.paths.append(MEASURED)  # recorded as "absent" in the manifest
    return {}


def setup() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "figure.dpi": 150})
    return plt


def save(fig: Any, name: str, inp: Inputs, out_dir: Path, notes: list[str]) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        paths.append(str(p))
    man = {"figure": name, "outputs": paths, "inputs_sha256": inp.manifest(), "notes": notes}
    (out_dir / f"{name}.manifest.json").write_text(
        json.dumps(man, indent=1) + "\n", encoding="ascii"
    )
    return man


def placeholder(ax: Any, x: float, y: float, label: str) -> None:
    ax.scatter([x], [y], marker="x", color="0.6", zorder=5)
    ax.annotate(
        label + " (weekend; placeholder)",
        (x, y),
        fontsize=6,
        color="0.4",
        xytext=(4, 4),
        textcoords="offset points",
    )
