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
from _common import COL_IN, COLORS, MUTED, entry, nr, plt, save, style  # noqa: E402

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
    from matplotlib.lines import Line2D
    style()
    rep, x = load()
    cps = rep["checkpoints"]
    tiers = []
    for key in rep["table"]:
        t = key.split("|")[0]
        if t not in tiers:
            tiers.append(t)
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL_IN, 2.85), gridspec_kw={"width_ratios": [1.25, 1.0]})
    markers = ["o", "s", "D", "^", "v"]
    h = []
    for i, t in enumerate(tiers):
        cells = [c for k, c in rep["table"].items() if k.split("|")[0] == t]
        n = sum(c["n"] for c in cells)
        surv = [sum(c["survival"][cp] for c in cells) for cp in cps]
        a.plot(cps, surv, marker=markers[i % 5], ms=3.2, color=COLORS[i], lw=1.1)
        h.append(Line2D([], [], marker=markers[i % 5], ms=3.6, color=COLORS[i], lw=1.1,
                        label=f"{tier_label(t).replace('Ollama ', '')} ({n})"))
    a.set_xlabel("turn")
    a.set_ylabel("intact sessions")
    a.set_xticks([c for c in cps if c % 10 == 0])
    a.set_title("(a) survival", loc="left")
    a.set_yticks([0, 5, 10, 15])

    xt = list(x["per_tier"])
    models = sorted(k for k in x["per_tier"][xt[0]] if k != "overall")
    for j, t in enumerate(xt):
        v = x["per_tier"][t]["overall"]
        lo, hi = v["ci95"]
        b.errorbar(j, v["excess"], yerr=[[v["excess"] - lo], [hi - v["excess"]]], fmt="o", ms=3.6, color="#222222",
                   capsize=2, lw=0.9, zorder=3)
        for k, m in enumerate(models):
            b.plot(j + 0.2 + 0.1 * k, x["per_tier"][t][m]["excess"], marker=markers[k], ms=3.0, ls="",
                   mfc="white", mec=MUTED, mew=0.7, zorder=2)
    b.axhline(0, color=MUTED, lw=0.6)
    b.set_xticks(range(len(xt)))
    b.set_xticklabels([tier_label(t) for t in xt], rotation=50, ha="right", rotation_mode="anchor")
    b.set_xlabel("window tier")
    b.set_ylabel("excess over default")
    b.set_xlim(-0.5, len(xt) - 0.2)
    b.set_title("(b) excess", loc="left")
    h.append(Line2D([], [], marker="o", color="#222222", ls="", ms=3.6, label="all (95% CI)"))
    h += [Line2D([], [], marker=markers[k], mfc="white", mec=MUTED, mew=0.7, ls="", ms=3.6, label=m)
          for k, m in enumerate(models)]
    fig.legend(handles=h, loc="outside lower center", ncol=3, frameon=False, handlelength=1.0, labelspacing=0.2,
               columnspacing=0.5, handletextpad=0.3, title="(a) tier (sessions); (b) excess, all and per model")
    return save(fig, NAME, out_dir)


def main(out_dir=None):
    return draw(out_dir)


if __name__ == "__main__":
    print(main())
