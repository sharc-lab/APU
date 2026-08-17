#!/usr/bin/env python3
"""Plot score vs context depth for each probe, highlighting depth-sensitive ones.

Usage:
    py -3.11 analysis/plot_per_probe.py                     # latest run
    py -3.11 analysis/plot_per_probe.py results/run_XYZ.jsonl
    py -3.11 analysis/plot_per_probe.py --out figs/per_probe.png
    py -3.11 analysis/plot_per_probe.py --min-range 0.3     # vary threshold
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"

CATEGORY_COLORS = {
    "reasoning_heavy": "#e41a1c",
    "code_heavy": "#377eb8",
    "structured_output": "#4daf4a",
    "rag_heavy": "#984ea3",
    "search_heavy": "#ff7f00",
    "long_horizon": "#a65628",
    "chained_tools": "#f781bf",
    "fan_out": "#999999",
}


def load_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", nargs="?")
    parser.add_argument("--out", default=None)
    parser.add_argument("--min-range", type=float, default=0.2,
                        help="Min(max-mean - min-mean) to label a probe as depth-sensitive")
    args = parser.parse_args()

    if args.results_file:
        path = Path(args.results_file)
    else:
        files = sorted(RESULTS_DIR.glob("run_*.jsonl"))
        if not files:
            raise SystemExit("No results files found in results/")
        path = files[-1]
        print(f"Using {path.name}")

    rows = [r for r in load_rows(path) if r.get("score") is not None]

    probe_cat: dict[str, str] = {}
    probe_depth: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        pid = r["probe_id"]
        probe_cat[pid] = r["category"]
        probe_depth[pid][r["depth"]].append(float(r["score"]))

    depths_all = sorted({r["depth"] for r in rows})

    # Identify depth-sensitive probes
    sensitive = []
    stable = []
    for pid, dm in probe_depth.items():
        means = [np.mean(dm[d]) for d in depths_all if d in dm]
        if max(means) - min(means) >= args.min_range:
            sensitive.append(pid)
        else:
            stable.append(pid)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Draw stable probes in light grey first
    for pid in stable:
        dm = probe_depth[pid]
        depths = [d for d in depths_all if d in dm]
        means = [np.mean(dm[d]) for d in depths]
        ax.plot(depths, means, color="#cccccc", linewidth=0.8, alpha=0.6, zorder=1)

    # Draw sensitive probes on top with category colour
    for pid in sensitive:
        dm = probe_depth[pid]
        cat = probe_cat[pid]
        color = CATEGORY_COLORS.get(cat, "#333333")
        depths = [d for d in depths_all if d in dm]
        means = [np.mean(dm[d]) for d in depths]
        ax.plot(depths, means, color=color, linewidth=2.0, marker="o", markersize=4,
                label=f"{pid} ({cat})", zorder=2)

    ax.set_xlabel("Filler Context Depth (tokens)", fontsize=11)
    ax.set_ylabel("Score (0-1)", fontsize=11)
    ax.set_title(
        f"Per-probe score vs depth  |  {len(sensitive)} sensitive / {len(stable)} stable  "
        f"(range >= {args.min_range})",
        fontsize=11,
    )
    ax.set_ylim(-0.05, 1.05)
    ax.set_xscale("symlog", linthresh=500)
    ax.set_xticks(depths_all)
    ax.set_xticklabels([str(d) for d in depths_all], rotation=30)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.85, ncol=2)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
