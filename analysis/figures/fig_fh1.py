"""Figure FH1 (headline): the cost of crossing each platform's memory boundary, three panels on one x axis.

x axis: memory in use at a window as a fraction of the platform's boundary (the spec's "window as a fraction of the
boundary", expressed in memory because each boundary is a memory size).

Panel (a) AMD APU. Register rows fh1-amd-boundary-llama-3.3-70b and fh1-amd-boundary-qwen3-32b, through their compute
functions numbers_register.compute_fh1_amd_boundary_llama70b / compute_fh1_amd_boundary_qwen3_32b (result files
results/mx2_spill_stats.jsonl, results/mx2_validation.jsonl, results/t2s_night2_20261001T151340Z.jsonl,
results/x2_70b_edge_reps.jsonl). x = per-process GPU memory / the dedicated block (`frac_dedicated`, boundary
`dedicated_mib`); y = decode rate and TTFT at that window divided by the same model's smallest window from the same
source file. Points are labelled with the served window; clamped points (served window < requested) are hollow
and labelled requested -> served. Each point's
overflow path is read from its own `spill` field; the "spills to shared memory" region is shaded only if every point
agrees with it. The hard crash far beyond the boundary is stated as panel text from the row's bisection record, not
placed on the axis (its memory figure is the server's logged allocation, a different measure from the x counter).

Panel (b) Intel iGPU. Register row A-24-effective-heap-limit, through numbers_register._a24_model_logs /
_a24_model_parameters / _a24_heap_demand_points, the helpers its compute function calls (result files
results/t2s_amech_20260926T181456Z.jsonl, results/t2s_amech_20260926T181456Z_srv_bis_*.txt,
results/t2s_night2_20260929T034014Z_srv_bis_a70_*.txt). x = shared-heap demand at each server start ('started') or
refused device-compute allocation ('device compute refused'), divided by the upper end of the fitted limit L
(`tightest_upper`); y = start or refuse, one row per model. The fitted bracket on L is shaded. The row's pinned
host-compute bounds are not drawn (they do not decide whether the server starts).

Panel (c) NVIDIA dGPU. Register row blade-k1-layer-split, through results/blade_k1_v2.summary.json (the one file its
compute function reads): x = loaded model size / device VRAM at each tested window, labelled with the VRAM share. The
cost of crossing (slowdown against spilled share) is NOT in verified results, so the panel draws placement only and
marks the cost PENDING.

Measurement note (caption): each AMD point is one call per fresh server start without a warm-up call (5 or 3 calls
per point), which departs from the paper's 1 warm-up + N convention.

Usage: py -3.12 analysis/figures/fig_fh1.py   (writes docs/paper1/figures/fig_fh1.pdf)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import COLORS, MUTED, REPO, entry, nr, plt, row, save, style  # noqa: E402

ROWS = ["fh1-amd-boundary-llama-3.3-70b", "fh1-amd-boundary-qwen3-32b", "A-24-effective-heap-limit",
        "blade-k1-layer-split"]
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


def nvidia_points():
    files = entry("blade-k1-layer-split")["data_files"]
    s = json.loads((REPO / files[0]).read_text(encoding="utf-8"))
    vram_b = s["device"]["vram_total_mib"] * 2 ** 20
    out = {}
    for m, v in s["models"].items():
        if not v.get("default_ctx"):
            continue
        out[m] = [(o["num_ctx"], o["size"] / vram_b, o["size_vram"] / o["size"]) for o in v.get("overflow_window", [])
                  if o.get("size")]
    return s["device"]["vram_total_mib"], out


def draw(out_dir=None):
    from matplotlib.lines import Line2D
    style()
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.7), sharex=True,
                             gridspec_kw={"width_ratios": [1.3, 1.0, 0.9], "wspace": 0.12})
    a, b, c = axes
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
        a.text(1.02, 1.045, "spills to shared memory, no error", fontsize=4.6, color="#555555", rotation=90,
               va="top", ha="left")
    for i, m in enumerate(models):
        col = COLORS[i]
        for p in [q for q in meas if q["model"] == m]:
            fc = "white" if p["clamped"] else col
            a.plot(p["x"], p["decode"], "o", ms=4.0, mfc=fc, mec=col, mew=0.9, zorder=3)
            a.plot(p["x"], p["ttft"], "^", ms=4.0, mfc=fc, mec=col, mew=0.9, zorder=3)
    seen = {}
    for p in sorted(meas, key=lambda q: q["x"]):
        seen.setdefault((p["model"], p["req"], p["served"]), p)
    for k, p in enumerate(sorted(seen.values(), key=lambda q: q["x"])):
        txt = f"{p['served']}" if not p["clamped"] else f"{p['req']}→" + NL + f"{p['served']}"
        y = min(p["decode"], p["ttft"])
        dy = (-4, -11, -18)[k % 3]
        a.annotate(txt, (p["x"], y), xytext=(0, dy), textcoords="offset points", fontsize=4.6, ha="center",
                   va="top", color="#333333")
    a.set_ylim(0.932, 1.068)
    bis = details["llama-3.3-70b"]["bisection"]
    a.text(0.99, 0.02, f"llama-3.3-70b hard limit: request {bis['first_fail']['n_ctx']}" + NL
           + f"crashes the server (exit {bis['first_fail']['exit_code']});" + NL
           + f"last start: request {bis['last_start']['n_ctx']}"
           + (" (clamped)" if bis["last_start"]["clamped"] else ""),
           transform=a.transAxes, fontsize=4.8, color="#333333", va="bottom", ha="right")
    a.set_title("(a) AMD APU: soft spill", loc="left")
    a.set_ylabel("ratio to the smallest window")
    a.set_xlabel("GPU memory in use /" + NL + f"{ded:.0f} MiB dedicated block")
    h = [Line2D([], [], color=COLORS[i], marker="s", ls="", ms=3.5, label=m) for i, m in enumerate(models)]
    h += [Line2D([], [], color="#444444", marker="o", ls="", ms=3.5, label="decode"),
          Line2D([], [], color="#444444", marker="^", ls="", ms=3.5, label="TTFT"),
          Line2D([], [], color="#444444", marker="o", mfc="white", ls="", ms=3.5, label="clamped")]
    a.legend(handles=h, loc="upper left", ncol=1, frameon=False, handletextpad=0.2, labelspacing=0.25)

    # (b) Intel iGPU
    l_lo, l_hi, ipts = intel_points()
    order = sorted(ipts)
    b.axvspan(l_lo / l_hi, 1.0, color="#cfcfcf", zorder=0)
    b.axvline(1.0, color=MUTED, lw=0.8, ls="--")
    for j, m in enumerate(order):
        st, rf = ipts[m]
        b.plot([v / l_hi for v in st], [j + 0.1] * len(st), "o", ms=3.0, mfc=COLORS[2], mec="white", mew=0.4,
               zorder=3)
        b.plot([v / l_hi for v in rf], [j - 0.1] * len(rf), "x", ms=3.4, color=COLORS[1], mew=1.0, zorder=3)
        b.text(0.03, j + 0.32, m, fontsize=5, color="#333333", va="center")
    b.set_yticks([])
    b.set_ylim(-0.6, len(order) + 0.25)
    b.grid(axis="y", visible=False)
    b.set_title("(b) Intel iGPU: start or refuse", loc="left")
    b.set_xlabel("shared-heap demand / L" + NL + f"(L ≤ {l_hi:.1f} MiB)")
    b.legend(handles=[Line2D([], [], color=COLORS[2], marker="o", ls="", ms=3.2, label="started"),
                      Line2D([], [], color=COLORS[1], marker="x", ls="", ms=3.2, label="refused")],
             loc="upper left", ncol=2, frameon=False, handletextpad=0.2, columnspacing=0.6)

    # (c) NVIDIA dGPU: placement measured, cost pending
    vram, npts = nvidia_points()
    c.axvline(1.0, color=MUTED, lw=0.8, ls="--")
    for i, (m, ps) in enumerate(npts.items()):
        y = 0.22 + 0.3 * i
        c.plot([x for _, x, _ in ps], [y] * len(ps), "s", ms=3.2, color=COLORS[3 + i], zorder=3)
        c.text(0.03, y + 0.1, m, fontsize=5, color="#333333", va="center")
        for k, (ctx, x, share) in enumerate(ps):
            c.annotate(f"{ctx}" + NL + f"{share:.2f}", (x, y), xytext=(0, 3 if k % 2 else -3),
                       textcoords="offset points", fontsize=4.3, ha="center", va="bottom" if k % 2 else "top",
                       color="#333333")
    c.set_ylim(0, 1)
    c.set_yticks([])
    c.grid(axis="y", visible=False)
    c.text(0.5, 0.87, "cost of crossing: PENDING" + NL + "(not in verified results)", transform=c.transAxes,
           ha="center", va="center", fontsize=5.6, color="#8a3b00",
           bbox=dict(boxstyle="round,pad=0.3", fc="#fff4e5", ec="#e0b080", lw=0.6))
    c.text(0.03, 0.03, "labels: window / VRAM share", transform=c.transAxes, fontsize=4.6, color="#555555")
    c.set_title("(c) NVIDIA dGPU: placement only", loc="left")
    c.set_xlabel("loaded model size /" + NL + f"{vram} MiB VRAM")
    for ax in axes:
        ax.set_xlim(*xlim)
    return save(fig, NAME, out_dir)


def main(out_dir=None):
    return draw(out_dir)


if __name__ == "__main__":
    print(main())
