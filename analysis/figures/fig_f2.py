"""Figure F2: Intel iGPU flip point per model, measured against three accountings.

Register rows A-24-flip-point-per-model and A-24-flip-point-prediction, through their compute functions
numbers_register.compute_a24_flip_point_per_model / compute_a24_flip_point_prediction (result files
results/t2s_amech_20260926T181456Z.jsonl, results/t2s_night2_20260929T034014Z.jsonl,
results/t2s_amech_20260926T181456Z_srv_bis_*.txt, results/t2s_night2_20260929T034014Z_srv_bis_a70_*.txt).
Both rows return the same per-model `detail`; the script checks that they agree and draws it.

One row per model (all five), ordered by measured flip point. x = predicted flip point minus the measured first refusal
(`measured_first_fail`), in tokens. The measured bracket (last start, first refusal] is the shaded interval
(`measured_last_ok` - `measured_first_fail`, 0]. Markers: accounting A (model file size), B (device memory only),
C (shared heap, limit fitted leave-one-model-out). The model's measured bracket is printed beside each row.

Usage: py -3.12 analysis/figures/fig_f2.py   (writes docs/paper1/figures/fig_f2.pdf)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import COLORS, MUTED, plt, row, save, style  # noqa: E402

ROWS = ["A-24-flip-point-per-model", "A-24-flip-point-prediction"]
NAME = "fig_f2.pdf"
ACC = [("A_file_size", "A: file size", "s", COLORS[1]),
       ("B_device_only", "B: device only", "D", COLORS[4]),
       ("C_shared_heap_loo", "C: shared heap (LOO)", "o", COLORS[0])]


def load():
    per_model = row(ROWS[0])["detail"]
    pred = row(ROWS[1])["detail"]
    for m, d in per_model.items():
        if pred[m]["pred"] != d["pred"] or pred[m]["measured_first_fail"] != d["measured_first_fail"]:
            raise RuntimeError(f"{ROWS[0]} and {ROWS[1]} disagree for {m}")
    return per_model


def draw(out_dir=None):
    style()
    d = load()
    models = sorted(d, key=lambda m: d[m]["measured_first_fail"])
    fig, ax = plt.subplots(figsize=(3.4, 2.15))
    ax.axvline(0, color=MUTED, lw=0.7)
    for j, m in enumerate(models):
        r = d[m]
        lo = r["measured_last_ok"] - r["measured_first_fail"]
        ax.add_patch(plt.Rectangle((lo, j - 0.32), -lo, 0.64, color="#cfcfcf", lw=0, zorder=1))
        for k, (key, _, mk, col) in enumerate(ACC):
            ax.plot(r["pred"][key] - r["measured_first_fail"], j + (k - 1) * 0.16, mk, ms=3.6, color=col,
                    mec="white", mew=0.4, zorder=3)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([f"{m}\n({d[m]['measured_last_ok']}, {d[m]['measured_first_fail']}]" for m in models],
                       fontsize=5.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("predicted minus measured first refusal (tokens)")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = [Line2D([], [], marker=mk, color=col, ls="", ms=3.6, label=lab) for _, lab, mk, col in ACC]
    h.append(Patch(color="#cfcfcf", label="measured (last start, first refusal]"))
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False,
              columnspacing=0.8, handletextpad=0.3)
    return save(fig, NAME, out_dir)


def main(out_dir=None):
    return draw(out_dir)


if __name__ == "__main__":
    print(main())
