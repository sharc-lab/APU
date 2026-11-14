"""Figure F5: the default window each platform's runtime loads, over the CDF of agent step lengths in public traces.

Register rows:
- trace-fraction-exceeds-4096 (numbers_register.compute_trace_fraction_exceeds_4096; result file
  results/traces/agent_step_lengths.parquet). The CDF is drawn from the same rows that function selects (the two
  uncensored trace sources, `tokens_qwen`). The script recomputes the row's two percentages from its own selection
  and refuses to draw if they differ from the row's value string, so the curve and the row cannot drift apart.
- T2S-vs-X2-default-ctx (numbers_register.compute_t2s_x2_default_ctx_table; result files
  results/apu_results__t2s_k1_ollama_evo-t2s_20261001T074622Z.jsonl, results/t2s_k1_ollama_evo-x2_20260930T205515Z.jsonl):
  Ollama default window per model on the Intel mini PC (CPU) and on the AMD APU (iGPU).
- blade-k1-layer-split (results/blade_k1_v2.summary.json, the one file its compute function reads): Ollama default
  window per model on the NVIDIA laptop GPU. A model with no default recorded there (not present on that machine) is
  not drawn.

Usage: py -3.12 analysis/figures/fig_f5.py   (writes docs/paper1/figures/fig_f5.pdf)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import COLORS, REPO, entry, plt, row, save, style  # noqa: E402

ROWS = ["trace-fraction-exceeds-4096", "T2S-vs-X2-default-ctx", "blade-k1-layer-split"]
NAME = "fig_f5.pdf"
NL = "\n"
UNCENSORED = ["nebius/SWE-agent-trajectories", "SWE-Gym/SWE-Gym-Trajectories"]


def step_lengths():
    import pandas as pd
    e = entry(ROWS[0])
    reg = e["compute"](REPO)
    df = pd.read_parquet(REPO / e["data_files"][0])
    if set(df["dataset"].unique()) & {UNCENSORED[1]}:
        sel = df[df["dataset"].isin(UNCENSORED)]
    else:
        sel = df[df["dataset"] == UNCENSORED[0]]
    tok = sel["tokens_qwen"].to_numpy()
    if len(tok) != reg["n"]:
        raise RuntimeError(f"step selection n={len(tok)} differs from {ROWS[0]} n={reg['n']}")
    pcts = [float(x) for x in re.findall(r"([\d.]+)% ", reg["value"])]
    mine = [round((tok > 4096).mean() * 100, 1), round((tok > 40960).mean() * 100, 1)]
    if pcts[:2] != mine:
        raise RuntimeError(f"recomputed {mine} differs from {ROWS[0]} {pcts[:2]}")
    return tok


def defaults():
    t = row(ROWS[1])["detail"]
    s = json.loads((REPO / entry(ROWS[2])["data_files"][0]).read_text(encoding="utf-8"))
    nv = {m: v["default_ctx"] for m, v in s["models"].items() if v.get("default_ctx")}
    return ({m: v["t2s_default_ctx"] for m, v in t.items()}, {m: v["x2_default_ctx"] for m, v in t.items()}, nv)


def draw(out_dir=None):
    style()
    tok = np.sort(step_lengths())
    intel, amd, nv = defaults()
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    y = np.arange(1, len(tok) + 1) / len(tok)
    ax.step(tok, y, where="post", color="#333333", lw=1.1)
    ax.set_xscale("log", base=2)
    ax.set_xlim(2 ** 10, 2 ** 19)
    ax.set_ylim(0, 1.02)

    # group models by (platform, window) so coinciding defaults share one line
    lines = {}
    for plat, d in (("Intel CPU", intel), ("NVIDIA GPU", nv), ("AMD iGPU", amd)):
        for m, w in d.items():
            lines.setdefault(w, []).append((plat, m))
    pcol = {"Intel CPU": COLORS[2], "NVIDIA GPU": COLORS[3], "AMD iGPU": COLORS[0]}
    for w, items in sorted(lines.items()):
        frac = (tok <= w).mean()
        plats = sorted({p for p, _ in items})
        col = pcol[plats[0]] if len(plats) == 1 else "#444444"
        ax.axvline(w, color=col, lw=1.0, ls="--", zorder=1)
        ax.plot([w], [frac], "o", ms=3.2, color=col, zorder=3)
        who = [f"{p}: " + ", ".join(m for q, m in items if q == p) for p in plats]
        longer = f"{(1 - frac) * 100:.1f}% of steps longer"
        if len(plats) > 1:  # shared default: horizontal block in the empty upper-left area
            ax.text(w * 1.1, 0.98, NL.join([f"default {w}"] + who + [longer]), fontsize=4.8, color="#333333",
                    ha="left", va="top", zorder=4,
                    bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.9))
        else:
            ax.text(w * 1.08, 0.03, f"default {w}: {'; '.join(who)}" + NL + longer, rotation=90,
                    fontsize=4.6, color="#333333", ha="left", va="bottom")
    ticks = [2 ** k for k in range(10, 20, 2)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("agent step length (tokens, log scale)")
    ax.set_ylabel(f"CDF of steps (n = {len(tok)})")
    return save(fig, NAME, out_dir)


def main(out_dir=None):
    return draw(out_dir)


if __name__ == "__main__":
    print(main())
