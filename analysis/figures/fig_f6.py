"""Figure F6: session survival per window tier, and the after-window silent-failure excess per tier.

Register rows R2-real-v1b-survival and R2-v1b-excess-over-default. Both compute functions
(numbers_register.compute_r2_real_v1b_survival / compute_r2_v1b_excess_over_default) build the same report with
numbers_register._r2_real_v1b_report (analysis/r2_real_report.py::build_combined over results/x2_r2_real_v1.jsonl,
results/x2_r2_real_v1b.jsonl, results/x2_r2_validation_v2.jsonl, results/x2_r2_validation_v2b.jsonl); the excess
row adds r2_real_report.excess_over_default. This script calls the same two functions and draws their structured
output.

Panel (a): intact sessions (no gated failure, no canary miss) at each checkpoint turn, summed over the three models
per tier (`table[tier|model]["survival"]`, checkpoints `checkpoints`), out of the tier's session count. Only the
checkpoint turns the row reports are drawn; no turn-0 point is added.
Panel (b): per tier, after-window gated silent-failure rate minus the Ollama-default tier's rate, overall (with its
95% percentile bootstrap CI as stored in the row) and per model (small hollow markers, CI not drawn).

Usage: py -3.12 analysis/figures/fig_f6.py   (writes docs/paper1/figures/fig_f6.pdf)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import COLORS, MUTED, entry, nr, plt, save, style  # noqa: E402

ROWS = ["R2-real-v1b-survival", "R2-v1b-excess-over-default"]
NAME = "fig_f6.pdf"


def tier_label(t):
    return "Ollama default" if t == "ollama_default" else t.replace("num_ctx_", "")


def load():
    for cid in ROWS:
        entry(cid)  # refuses a row that is not VERIFIED
    rr, rep = nr._r2_real_v1b_report(nr.REPO)
    x = rr.excess_over_default(rep)
    return rep, x


def draw(out_dir=None):
    style()
    rep, x = load()
    cps = rep["checkpoints"]
    tiers = []
    for key in rep["table"]:
        t = key.split("|")[0]
        if t not in tiers:
            tiers.append(t)
    fig, (a, b) = plt.subplots(1, 2, figsize=(3.45, 2.2), gridspec_kw={"width_ratios": [1.45, 1.0], "wspace": 0.42})
    markers = ["o", "s", "D", "^", "v"]
    for i, t in enumerate(tiers):
        cells = [c for k, c in rep["table"].items() if k.split("|")[0] == t]
        n = sum(c["n"] for c in cells)
        surv = [sum(c["survival"][cp] for c in cells) for cp in cps]
        a.plot(cps, surv, marker=markers[i % 5], ms=3.0, color=COLORS[i], lw=1.1,
               label=f"{tier_label(t)} (n={n})")
    a.set_xlabel("turn")
    a.set_ylabel("intact sessions")
    a.set_xticks(cps)
    a.set_title("(a) survival", loc="left")
    a.set_yticks([0, 5, 10, 15])
    a.legend(loc="upper left", bbox_to_anchor=(-0.25, -0.24), ncol=3, frameon=False, fontsize=5, handlelength=1.6,
             labelspacing=0.2, columnspacing=0.8, title="window tier (sessions)", title_fontsize=5)

    xt = list(x["per_tier"])
    models = sorted(k for k in x["per_tier"][xt[0]] if k != "overall")
    for j, t in enumerate(xt):
        v = x["per_tier"][t]["overall"]
        lo, hi = v["ci95"]
        b.errorbar(j, v["excess"], yerr=[[v["excess"] - lo], [hi - v["excess"]]], fmt="o", ms=3.4, color="#222222",
                   capsize=2, lw=0.9, zorder=3)
        for k, m in enumerate(models):
            b.plot(j + 0.17 + 0.07 * k, x["per_tier"][t][m]["excess"], marker=markers[k], ms=2.6, ls="",
                   mfc="white", mec=MUTED, mew=0.7, zorder=2)
    b.axhline(0, color=MUTED, lw=0.6)
    b.set_xticks(range(len(xt)))
    b.set_xticklabels([tier_label(t) for t in xt], rotation=35, fontsize=5.5)
    b.set_xlabel("window tier")
    b.set_ylabel("excess over default")
    b.set_xlim(-0.5, len(xt) - 0.3)
    b.set_title("(b) silent-failure excess", loc="left")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", color="#222222", ls="", ms=3, label="all (95% CI)")]
    h += [Line2D([], [], marker=markers[k], mfc="white", mec=MUTED, mew=0.7, ls="", ms=2.8, label=m)
          for k, m in enumerate(models)]
    b.legend(handles=h, loc="upper left", frameon=False, fontsize=4.8, labelspacing=0.2, handletextpad=0.2)
    return save(fig, NAME, out_dir)


def main(out_dir=None):
    return draw(out_dir)


if __name__ == "__main__":
    print(main())
