"""Quality vs USD per 1000 steps frontier plot, one panel column per machine, from src/dse/pareto.py.

Reads only committed files (outcome tables, item weights, cloud price file); no API is called and nothing is run.
Writes:
  figs/pareto_frontier_<tag>_<date>.png and .svg   (tag "x2" by default)
  results/pareto_points_<date>.json                 (every point, the frontier per machine, the sample
                                                     recommendation, inputs and the STUB list)

Each machine column has two panels: quality vs USD per 1000 steps (the frontier, a step line through the frontier
points), and quality vs p50 latency (the constraint the recommender applies). Colour = runtime family (Ollama,
llama-server, cloud). Hollow markers = bare-template cells (Ollama tag created from a bare GGUF: install-path
result, kept off the frontier). STUB cloud models have a priced cost but no measured quality or latency, so they
are drawn as dashed vertical lines labelled STUB in the cost panel, never as points.

Usage:
  py -3.12 analysis/plot_pareto_frontier.py                                   # evo-x2 v3 table, stub cloud
  py -3.12 analysis/plot_pareto_frontier.py --file evo-x2=results/x2_outcome_table_v3.jsonl \\
      --file evo-t2s=results/t2s_outcome_table_full.jsonl --tag x2_t2s --cloud-source auto
  py -3.12 analysis/plot_pareto_frontier.py --hw-usd-per-hour evo-x2=0.05     # amortized hardware cost
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import NullFormatter, ScalarFormatter  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.dse import pareto as P  # noqa: E402

DEFAULT_FILES = [("evo-x2", "results/x2_outcome_table_v3.jsonl")]
SAMPLE = {"budget_usd": 50.0, "quality_floor": 0.9, "latency_target_ms": 30_000.0, "hardware": ["evo-x2"]}

# Reference palette (dataviz skill references/palette.md, light mode), first three categorical slots: the only
# slots validated all-pairs for scatter. Other runtimes fold into neutral gray.
SURFACE, INK, INK_2, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
RUNTIME_COLORS = {"ollama": "#2a78d6", "llama_server": "#eb6834", "cloud": "#1baf7a", "other": "#8a8984"}
RUNTIME_NAMES = {"ollama": "Ollama", "llama_server": "llama-server", "cloud": "cloud (measured)",
                 "other": "other runtime"}


def runtime_family(p: P.Point) -> str:
    if p.is_cloud:
        return "cloud"
    if p.runtime.startswith("ollama"):
        return "ollama"
    if p.runtime.startswith("llama_server"):
        return "llama_server"
    return "other"


def short_model(m: str) -> str:
    return m.replace("llama3.1:8b", "llama3.1-8b")


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK_MUTED)
    ax.tick_params(colors=INK_2, labelsize=8)


def _scatter(ax, p, x, on_frontier):
    color = RUNTIME_COLORS[runtime_family(p)]
    hollow = p.bare_template
    ax.scatter([x], [p.quality], s=64, marker="o", facecolors="none" if hollow else color,
               edgecolors=color if hollow else SURFACE, linewidths=2 if hollow else 1.5, zorder=3)
    if on_frontier:
        ax.scatter([x], [p.quality], s=200, marker="o", facecolors="none", edgecolors=INK, linewidths=1.2, zorder=4)


def plot(points, out_png, out_svg, title_note="", latency_target_s=None):
    machines = sorted({p.machine for p in points if not p.is_cloud and p.quality is not None})
    real_cloud = [p for p in points if p.is_cloud and p.quality is not None]
    stub_cloud = [p for p in points if p.is_cloud and p.quality is None]
    n = max(1, len(machines))
    fig, axes = plt.subplots(2, n, figsize=(6.4 * n, 9.0), squeeze=False, facecolor=SURFACE)
    qs = [p.quality for p in points if p.quality is not None]
    ylo = (min(qs) if qs else 0.0) - 0.02
    yhi = 1.0 + 0.01
    xmax = max([p.usd_per_1k_steps for p in points] + [1.0])
    for col, m in enumerate(machines):
        mine = [p for p in points if p.machine == m and p.quality is not None] + real_cloud
        front = P.frontier(mine)
        on = {p.config_id for p in front}
        ax, axl = axes[0][col], axes[1][col]
        _style(ax), _style(axl)
        for p in mine:
            _scatter(ax, p, p.usd_per_1k_steps, p.config_id in on)
            if p.latency_p50_ms is not None:
                _scatter(axl, p, p.latency_p50_ms / 1000, p.config_id in on)
        if front:
            xs, ys = [], []
            for p in front:
                if xs:
                    xs.append(p.usd_per_1k_steps)
                    ys.append(ys[-1])
                xs.append(p.usd_per_1k_steps)
                ys.append(p.quality)
            xs.append(xmax * 1.5)
            ys.append(ys[-1])
            ax.plot(xs, ys, color=INK, linewidth=1.2, linestyle="-", zorder=2)
            for p in front:
                ax.annotate(f"{short_model(p.model)} {p.runtime}\nq {p.quality:.3f} (n={p.quality_n})",
                            (p.usd_per_1k_steps, p.quality), xytext=(10, -26), textcoords="offset points",
                            fontsize=8, color=INK, zorder=5)
                if p.latency_p50_ms is not None:
                    axl.annotate(f"{short_model(p.model)} {p.runtime}", (p.latency_p50_ms / 1000, p.quality),
                                 xytext=(8, 6), textcoords="offset points", fontsize=8, color=INK)
        bare = sorted((p for p in mine if p.bare_template and p.latency_p50_ms is not None),
                      key=lambda p: p.latency_p50_ms)
        for i, p in enumerate(bare):
            axl.annotate(f"{short_model(p.model)} {p.runtime} [bare template]", (p.latency_p50_ms / 1000, p.quality),
                         xytext=(6, -14 - 11 * i), textcoords="offset points", fontsize=7, color=INK_2,
                         arrowprops={"arrowstyle": "-", "color": INK_MUTED, "linewidth": 0.6})
        if latency_target_s is not None:
            axl.axvline(latency_target_s, color=INK_MUTED, linestyle=(0, (2, 2)), linewidth=1.2, zorder=1)
            axl.text(latency_target_s, ylo + 0.004, f" sample latency target {latency_target_s:.0f} s",
                     fontsize=7.5, color=INK_2, va="bottom", ha="left")
        for i, s in enumerate(stub_cloud):
            ax.axvline(s.usd_per_1k_steps, color=RUNTIME_COLORS["cloud"], linestyle=(0, (4, 3)), linewidth=1.5,
                       zorder=1)
            ax.text(s.usd_per_1k_steps, ylo + 0.004 + 0.012 * i,
                    f" {s.model} {P.STUB}\n USD {s.usd_per_1k_steps:.2f}/1k steps\n quality not measured",
                    fontsize=7.5, color=INK_2, va="bottom", ha="left")
        ax.set_xscale("symlog", linthresh=0.1)
        ax.set_xlim(-0.02, xmax * 1.5)
        ax.set_ylim(ylo, yhi)
        axl.set_xscale("log")
        axl.xaxis.set_major_formatter(ScalarFormatter())
        axl.xaxis.set_minor_formatter(NullFormatter())
        lats = [p.latency_p50_ms / 1000 for p in mine if p.latency_p50_ms]
        if lats:
            lo, hi = min(lats) * 0.8, max(lats) * 1.3
            if latency_target_s:
                lo, hi = min(lo, latency_target_s * 0.8), max(hi, latency_target_s * 1.3)
            axl.set_xticks([t for t in (1, 2, 5, 10, 20, 30, 50, 100, 200, 500, 1000) if lo <= t <= hi])
            axl.set_xlim(lo, hi)
        axl.set_ylim(ylo, yhi)
        n_cells = len([p for p in mine if not p.is_cloud])
        ax.set_title(f"{m}: {n_cells} configs", fontsize=11, color=INK, loc="left")
        ax.set_xlabel("USD per 1000 steps (local: 0 marginal API cost)", fontsize=9, color=INK_2)
        axl.set_xlabel("p50 latency per call, s (completed calls)", fontsize=9, color=INK_2)
        for a in (ax, axl):
            a.set_ylabel("quality (trace-weighted score)", fontsize=9, color=INK_2)
    handles = [Line2D([], [], marker="o", linestyle="", markerfacecolor=RUNTIME_COLORS[k], markeredgecolor=SURFACE,
                      markersize=8, label=RUNTIME_NAMES[k])
               for k in ("ollama", "llama_server", "cloud", "other")
               if any(runtime_family(p) == k for p in points if p.quality is not None)]
    handles += [Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=INK_2,
                       markeredgewidth=2, markersize=8, label="bare template (off frontier)"),
                Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=INK,
                       markersize=13, label="on frontier"),
                Line2D([], [], color=RUNTIME_COLORS["cloud"], linestyle=(0, (4, 3)), label=f"cloud {P.STUB} cost")]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 3 * n), frameon=False, fontsize=8.5,
               labelcolor=INK_2)
    fig.suptitle("Quality vs cost frontier per machine" + (f"  ({title_note})" if title_note else ""),
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, facecolor=SURFACE)
    fig.savefig(out_svg, facecolor=SURFACE)
    plt.close(fig)


def _kv(s):
    k, v = s.split("=", 1)
    return k, v


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", action="append", type=_kv, help="machine=path (repeatable)")
    ap.add_argument("--cloud-source", default="stub")
    ap.add_argument("--hw-usd-per-hour", action="append", type=_kv, default=[], help="machine=USD per hour")
    ap.add_argument("--tag", default="x2")
    ap.add_argument("--date", default=datetime.date.today().strftime("%Y%m%d"))
    a = ap.parse_args(argv)
    files = a.file or DEFAULT_FILES
    hw = {k: float(v) for k, v in a.hw_usd_per_hour}
    points = P.load_points(files, cloud_source=a.cloud_source, hw_usd_per_hour=hw)
    rec = P.recommend(points, **SAMPLE)
    png = REPO / "figs" / f"pareto_frontier_{a.tag}_{a.date}.png"
    plot(points, png, png.with_suffix(".svg"),
         title_note="cloud points are STUB" if any(p.stub for p in points) else "",
         latency_target_s=SAMPLE["latency_target_ms"] / 1000)
    table = P.points_table(points)
    table.update({"inputs": {"files": [list(f) for f in files], "cloud_source": a.cloud_source,
                             "hw_usd_per_hour": hw, "weights": P.WEIGHTS_FILE, "prices": P.PRICES_FILE},
                  "sample_recommendation": rec.to_dict(), "figure": str(png.relative_to(REPO)).replace("\\", "/")})
    out = REPO / "results" / f"pareto_points_{a.date}.json"
    out.write_text(json.dumps(table, indent=1) + "\n", encoding="utf-8")
    print("\n".join(P.format_table(points)))
    print(f"\nsample {SAMPLE}: {rec.reason}; savings vs all-cloud USD/1k steps "
          f"{rec.savings_vs_all_cloud_usd_per_1k if rec.savings_vs_all_cloud_usd_per_1k is None else round(rec.savings_vs_all_cloud_usd_per_1k, 2)}"
          + (f" [{P.STUB}]" if rec.stub else ""))
    print(f"wrote {png.relative_to(REPO)}, {png.with_suffix('.svg').relative_to(REPO)}, {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
