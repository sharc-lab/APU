"""Shared helpers for the Paper 1 figure scripts (analysis/figures/fig_*.py).

Every value a figure draws comes from a numbers-register row (analysis/numbers_register.py, NUMBER_ENTRIES) that is
VERIFIED in docs/NUMBERS_REGISTER.md, either through the row's own compute function or through the helper functions
that compute function calls on the row's own data files. `row()` refuses a row that is not VERIFIED there (PENDING,
UNSUPPORTED, CORRECTED, or missing), so a figure cannot silently draw an unverified number.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "docs" / "paper1" / "figures"
REGISTER_MD = REPO / "docs" / "NUMBERS_REGISTER.md"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from analysis import numbers_register as nr  # noqa: E402

# Okabe-Ito order (colour-blind safe); markers carry identity as well, so nothing is colour-only.
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
MUTED = "#6b6b6b"


def register_status(claim_id: str) -> str:
    """Status column of `claim_id` in docs/NUMBERS_REGISTER.md, or 'UNREGISTERED' if it has no row."""
    pat = re.compile(r"^\|\s*`?" + re.escape(claim_id) + r"`?\s*\|\s*([A-Z]+)\s*\|")
    for line in REGISTER_MD.read_text(encoding="utf-8").splitlines():
        m = pat.match(line)
        if m:
            return m.group(1)
    return "UNREGISTERED"


def entry(claim_id: str) -> dict:
    """The NUMBER_ENTRIES dict for a VERIFIED row; raises if the row is missing or not VERIFIED."""
    status = register_status(claim_id)
    if status != "VERIFIED":
        raise RuntimeError(f"register row {claim_id!r} is {status}, not VERIFIED; a figure may not draw it")
    for e in nr.NUMBER_ENTRIES:
        if e["claim_id"] == claim_id:
            return e
    raise RuntimeError(f"register row {claim_id!r} is in NUMBERS_REGISTER.md but not in NUMBER_ENTRIES")


def row(claim_id: str) -> dict:
    """Compute a VERIFIED row fresh (same call the register builder makes)."""
    return entry(claim_id)["compute"](nr.REPO)


# Printed widths (IEEEtran conference, 10 pt): docs/paper1/md2tex.py includes a column figure at 0.95\columnwidth
# (0.95 x 252 pt) and a full-width figure at 0.95\textwidth (0.95 x 516 pt); 72.27 pt per inch. Each script sets its
# figsize width to one of these and saves without a tight bounding box, so the PDF is included at scale 1.0 and the
# font sizes below are the printed sizes.
COL_IN = 0.95 * 252 / 72.27
TEXT_IN = 0.95 * 516 / 72.27
MIN_PT = 8   # smallest printed font size allowed in any figure


def style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": MIN_PT,
        "axes.titlesize": MIN_PT,
        "axes.labelsize": MIN_PT,
        "xtick.labelsize": MIN_PT,
        "ytick.labelsize": MIN_PT,
        "legend.fontsize": MIN_PT,
        "legend.title_fontsize": MIN_PT,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.02,
        "figure.constrained_layout.w_pad": 0.02,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e3e3e3",
        "grid.linewidth": 0.5,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.2,
        "pdf.fonttype": 42,
        "savefig.bbox": None,
    })


def save(fig, name: str, out_dir: Path | None = None) -> Path:
    out = Path(out_dir or OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    small = [x for x in fig.findobj(lambda o: hasattr(o, "get_fontsize") and hasattr(o, "get_text"))
             if x.get_text().strip() and x.get_visible() and x.get_fontsize() < MIN_PT - 1e-6]
    if small:
        raise RuntimeError(f"{name}: text below {MIN_PT} pt: {[x.get_text() for x in small][:5]}")
    # Fixed metadata so a rerun on the same data gives the same file.
    fig.savefig(path, format="pdf", metadata={"CreationDate": None, "ModDate": None, "Creator": None,
                                               "Producer": None, "Author": None})
    plt.close(fig)
    return path
