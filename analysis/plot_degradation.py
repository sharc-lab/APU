#!/usr/bin/env python3
"""Plot score vs context depth, one line per category, error bars from reps.

Usage:
    py -3.11 analysis/plot_degradation.py                     # latest run
    py -3.11 analysis/plot_degradation.py results/run_XYZ.jsonl
    py -3.11 analysis/plot_degradation.py --out figs/deg.png
"""

from __future__ import annotations

import argparse
import json
import sys
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
    parser.add_argument("results_file", nargs="?", help="run_*.jsonl path")
    parser.add_argument("--out", default=None, help="Save figure to this path")
    parser.add_argument(
        "--difficulty", choices=["easy", "medium", "hard", "all"], default="all"
    )
    parser.add_argument(
        "--filler-mode", default="unlabelled", choices=["unlabelled", "labelled", "all"],
        help="Filter to rows with this filler_mode (default: unlabelled)",
    )
    args = parser.parse_args()

    if args.results_file:
        path = Path(args.results_file)
    else:
        files = sorted(RESULTS_DIR.glob("run_*.jsonl"))
        if not files:
            print("No results files found in results/", file=sys.stderr)
            sys.exit(1)
        path = files[-1]
        print(f"Using {path.name}")

    all_rows = load_rows(path)
    rows = [r for r in all_rows if r.get("score") is not None]
    if args.difficulty != "all":
        rows = [r for r in rows if r.get("difficulty") == args.difficulty]
    if args.filler_mode != "all":
        rows = [r for r in rows if r.get("filler_mode", "unlabelled") == args.filler_mode]

    if not rows:
        print("No scoreable rows after filtering.", file=sys.stderr)
        sys.exit(1)

    # category -> depth -> [scores]
    data: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        data[r["category"]][r["depth"]].append(float(r["score"]))

    depths_all = sorted({r["depth"] for r in rows})

    fig, ax = plt.subplots(figsize=(11, 6))

    for category, depth_map in sorted(data.items()):
        depths = [d for d in depths_all if d in depth_map]
        means = [np.mean(depth_map[d]) for d in depths]
        stds = [np.std(depth_map[d]) for d in depths] if len(depth_map[depths[0]]) > 1 else [0] * len(depths)
        color = CATEGORY_COLORS.get(category, "#333333")
        ax.errorbar(
            depths, means, yerr=stds,
            label=category, marker="o", capsize=4,
            linewidth=1.8, markersize=5, color=color,
        )

    n_probes = len({r["probe_id"] for r in rows})
    n_reps = max(len(v) for dm in data.values() for v in dm.values())
    ax.set_xlabel("Filler Context Depth (tokens)", fontsize=12)
    ax.set_ylabel("Score (0–1)", fontsize=12)
    title = f"Quality Degradation vs Context Depth — {n_probes} probes, up to {n_reps} reps"
    if args.difficulty != "all":
        title += f" [{args.difficulty}]"
    ax.set_title(title, fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xscale("symlog", linthresh=500)
    ax.set_xticks(depths_all)
    ax.set_xticklabels([str(d) for d in depths_all], rotation=30)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved → {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
