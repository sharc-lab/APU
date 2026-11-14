"""Figure FH1 (headline): the cost of crossing each platform's memory boundary, two panels (AMD APU, Intel mini PC).

x axis: memory in use at a window as a fraction of the platform's boundary (the spec's "window as a fraction of the
boundary", expressed in memory because each boundary is a memory size).

Panel (a) AMD APU. Register rows fh1-amd-boundary-llama-3.3-70b and fh1-amd-boundary-qwen3-32b, through their compute
functions numbers_register.compute_fh1_amd_boundary_llama70b / compute_fh1_amd_boundary_qwen3_32b (result files
results/mx2_spill_stats.jsonl, results/mx2_validation.jsonl, results/t2s_night2_20261001T151340Z.jsonl,
results/x2_70b_edge_reps.jsonl). x = per-process GPU memory / the dedicated block (`frac_dedicated`, boundary
`dedicated_mib`); y = decode rate and TTFT at that window divided by the same model's smallest window from the same
source file. Points are labelled with the served window; clamped points (served window < requested) are hollow
and labelled requested -> served. Labels sit at evenly spaced x positions with leader lines to their points, so no
two labels overlap at the printed 8 pt size. Each point's
overflow path is read from its own `spill` field; the "spills to shared memory" region is shaded only if every point
agrees with it. The hard crash far beyond the boundary (the row's bisection record) is stated in the paper text
(Section 4.4), not drawn: its memory figure is the server's logged allocation, a different measure from the x counter.

Panel (b) Intel iGPU. Register row A-24-effective-heap-limit, through numbers_register._a24_model_logs /
_a24_model_parameters / _a24_heap_demand_points, the helpers its compute function calls (result files
results/t2s_amech_20260926T181456Z.jsonl, results/t2s_amech_20260926T181456Z_srv_bis_*.txt,
results/t2s_night2_20260929T034014Z_srv_bis_a70_*.txt). x = shared-heap demand at each server start ('started') or
refused device-compute allocation ('device compute refused'), divided by the upper end of the fitted limit L
(`tightest_upper`); y = start or refuse, one row per model. The fitted bracket on L is shaded. The row's pinned
host-compute bounds are not drawn (they do not decide whether the server starts).

The NVIDIA laptop GPU is not drawn: its placement (register row blade-k1-layer-split) is in the paper text, and the
cost of crossing its boundary is not in verified results, so the figure has two panels.

Measurement note (caption): each AMD point is one call per fresh server start without a warm-up call (5 or 3 calls
per point), which departs from the paper's 1 warm-up + N convention.

Usage: py -3.12 analysis/figures/fig_fh1.py   (writes docs/paper1/figures/fig_fh1.pdf)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import COLORS, MUTED, TEXT_IN, nr, plt, row, save, style  # noqa: E402

ROWS = ["fh1-amd-boundary-llama-3.3-70b", "fh1-amd-boundary-qwen3-32b", "A-24-effective-heap-limit"]
NAME = "fig_fh1.pdf"
NL = "\n"


def amd_points():
    meas, details = [], {}
    for cid in ROWS[:2]:
        d = row(cid)["detail"]
        details[d["model_id"]] = d
        base = {p["source"]: p for p in d["points"] if p["actual"] == p["baseline_ctx"] and not p["clamped"]}
        for p in d["points"]:
            b = base.get(p["source"])
            if b is None:
                continue
            meas.append(dict(model=d["model_id"], req=p["requested"], served=p["actual"], clamped=p["clamped"],
                             spill=p["spill"], x=p["frac_dedicated"], decode=p["decode"] / b["decode"],
                             ttft=p["ttft_ms"] / b["ttft_ms"]))
    return meas, details


def intel_points():
    lim = row("A-24-effective-heap-limit")["detail"]
    l_lo, l_hi = lim["tightest_lower"][0], lim["tightest_upper"][0]
    pts = {}
    for model, logs in nr._a24_model_logs(nr.REPO).items():
        p = nr._a24_model_parameters(model, logs)
        lo, hi = nr._a24_heap_demand_points(p, logs)
        pts[model] = ([v for v, _, k in lo if k == "started"], [v for v, _, k in hi if k == "device compute refused"])
    return l_lo, l_hi, pts


def _fan(ax, pts, x_lo, x_hi, rows):
    """Label points (x, y, text) without overlap: horizontal labels at evenly spaced x positions in [x_lo, x_hi] (in
    point order), alternating between the label rows `rows` (y values), each joined to its point by a leader line."""
    pts = sorted(pts)
    n = len(pts)
    for k, (x, y, txt) in enumerate(pts):
        xt = x_lo + (x_hi - x_lo) * (k / (n - 1) if n > 1 else 0.5)
        yt = rows[k % len(rows)]
        ax.plot([x, xt], [y, yt], color="#9a9a9a", lw=0.5, zorder=1)
        ax.text(xt, yt, txt, ha="center", va="top", color="#333333", zorder=4,
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))


def draw(out_dir=None):
    from matplotlib.lines import Line2D
    style()
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_IN, 2.45), sharex=True,
                             gridspec_kw={"width_ratios": [1.55, 1.0]})
    a, b = axes
    xlim = (0.0, 1.85)

    # (a) AMD APU
    meas, details = amd_points()
    models = list(details)
    ded = details["llama-3.3-70b"]["dedicated_mib"]
    shade_ok = all(p["spill"] == (p["x"] > 1.0) for p in meas)
    a.axvline(1.0, color=MUTED, lw=0.8, ls="--")
    a.axhline(1.0, color="#bbbbbb", lw=0.6)
    if shade_ok:
        a.axvspan(1.0, xlim[1], color="#f0f0f0", zorder=0)
        a.text(1.83, 1.155, "spills to shared" + NL + "memory, no error", color="#555555", va="top", ha="right")
    for i, m in enumerate(models):
        col = COLORS[i]
        for p in [q for q in meas if q["model"] == m]:
            fc = "white" if p["clamped"] else col
            a.plot(p["x"], p["decode"], "o", ms=4.0, mfc=fc, mec=col, mew=0.9, zorder=3)
            a.plot(p["x"], p["ttft"], "^", ms=4.0, mfc=fc, mec=col, mew=0.9, zorder=3)
    seen = {}
    for p in sorted(meas, key=lambda q: q["x"]):
        seen.setdefault((p["model"], p["req"], p["served"]), p)
    lab = [(p["x"], min(p["decode"], p["ttft"]) - 0.004,
            f"{p['served']}" if not p["clamped"] else f"{p['req']}→{p['served']}") for p in seen.values()]
    _fan(a, lab, 0.30, 1.58, (0.945, 0.905))
    a.set_ylim(0.86, 1.16)
    a.set_yticks([0.95, 1.0, 1.05, 1.10])
    a.set_title("(a) AMD APU: soft spill", loc="left")
    a.set_ylabel("ratio to the smallest window")
    a.set_xlabel("GPU memory in use /" + NL + f"{ded:.0f} MiB dedicated block")
    h = [Line2D([], [], color=COLORS[i], marker="s", ls="", ms=4, label=m) for i, m in enumerate(models)]
    h += [Line2D([], [], color="#444444", marker="o", ls="", ms=4, label="decode"),
          Line2D([], [], color="#444444", marker="^", ls="", ms=4, label="TTFT"),
          Line2D([], [], color="#444444", marker="o", mfc="white", ls="", ms=4, label="clamped")]
    a.legend(handles=h, loc="upper left", ncol=3, columnspacing=0.6, frameon=False, handletextpad=0.2, labelspacing=0.2,
             borderaxespad=0.1)

    # (b) Intel mini PC iGPU
    l_lo, l_hi, ipts = intel_points()
    order = sorted(ipts)
    b.axvspan(l_lo / l_hi, 1.0, color="#cfcfcf", zorder=0)
    b.axvline(1.0, color=MUTED, lw=0.8, ls="--")
    for j, m in enumerate(order):
        st, rf = ipts[m]
        b.plot([v / l_hi for v in st], [j - 0.08] * len(st), "o", ms=3.4, mfc=COLORS[2], mec="white", mew=0.4,
               zorder=3)
        b.plot([v / l_hi for v in rf], [j - 0.08] * len(rf), "x", ms=4.0, color=COLORS[1], mew=1.1, zorder=3)
        b.text(0.03, j + 0.36, m, color="#333333", va="center")
    b.set_yticks([])
    b.set_ylim(-0.5, len(order) + 0.75)
    b.grid(axis="y", visible=False)
    b.set_title("(b) Intel mini PC: hard wall", loc="left")
    b.set_xlabel("shared-heap demand / L" + NL + f"(L ≤ {l_hi:.1f} MiB)")
    b.legend(handles=[Line2D([], [], color=COLORS[2], marker="o", ls="", ms=4, label="started"),
                      Line2D([], [], color=COLORS[1], marker="x", ls="", ms=4, label="refused")],
             loc="upper right", ncol=2, frameon=False, handletextpad=0.2, columnspacing=0.6, borderaxespad=0.1)

    for ax in axes:
        ax.set_xlim(*xlim)
        ax.set_xticks([0.0, 0.5, 1.0, 1.5])
    return save(fig, NAME, out_dir)


def main(out_dir=None):
    return draw(out_dir)


if __name__ == "__main__":
    print(main())
