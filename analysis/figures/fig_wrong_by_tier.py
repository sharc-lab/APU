"""Figure FW (headline): wrong task turns by window tier, split silent vs loud, for three task workloads.

Register rows x2-evalset-tasks-real-v2-wrong-rate, xruntime-llamacpp-wrong-rate and pkg1109-t4-8b. Each group is drawn
from the same structured output its row's compute function formats:

- Ollama, qwen3:14b, AMD APU: x2-evalset-tasks-real-v2-wrong-rate, through harness/x2_r2_evalset.py::
  tasks_real_v2_outcome on results/x2_evalset_tasks_real_v2.jsonl (numbers_register.
  compute_x2_evalset_tasks_real_v2_wrong_rate). Per tier: `wrong` of `n_task_turns`, split into `silent_wrong` (no
  HTTP error or error field on the session's calls) and `surfaced_wrong`; the bar is labelled with the tier's
  `loaded_context_values`. The num_keep tier is not a window tier and is not drawn (it is in the paper text).
- llama-server, qwen3:14b, AMD APU: xruntime-llamacpp-wrong-rate, through harness/xruntime_llamacpp.py::outcome on
  results/xruntime_llamacpp_qwen3_14b.jsonl (numbers_register.compute_xruntime_llamacpp_wrong_rate). Arms "default"
  and "4096": `wrong` of `n_task_turns`, silent = `harness_silent_wrong` (no HTTP error on any call, the Ollama row's
  definition), loud = the rest. Arm "fixed" (-c 40960) loads the same window as the default and is not drawn.
- Ollama, qwen3:8b, Intel mini PC (second workload): pkg1109-t4-8b, through analysis/zach_package_20261109.py
  (t4_recompute over the raw rows, and bridge1_scores.json's D_silent_failures). Silence is recorded only for the
  default arm (D, loaded 4096); the 24576 arm (P) has no silent/loud split in the row, so its wrong turns are drawn as
  "not recorded", never as silent or loud.

Each bar is labelled with wrong / task turns (n per bar). Bars of one group share a workload, model and runtime;
groups are never pooled.

Usage: py -3.12 analysis/figures/fig_wrong_by_tier.py   (writes docs/paper1/figures/fig_wrong_by_tier.pdf)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import COL_IN, COLORS, entry, nr, plt, save, style  # noqa: E402

ROWS = ["x2-evalset-tasks-real-v2-wrong-rate", "xruntime-llamacpp-wrong-rate", "pkg1109-t4-8b"]
NAME = "fig_wrong_by_tier.pdf"
SILENT, LOUD, UNKNOWN = COLORS[1], COLORS[0], "#bdbdbd"


def _win(values, default):
    w = values[0] if len(values) == 1 else values
    return f"{w} (default)" if default else f"{w}"


def load():
    for cid in ROWS:
        entry(cid)  # refuses a row that is not VERIFIED
    groups = []
    # Ollama, qwen3:14b
    nr._r2_agent_module()
    import x2_r2_evalset as ev
    o = ev.tasks_real_v2_outcome(ev.ag.read_rows(nr.REPO / nr.X2_EVALSET_TASKS_REAL_V2), models=(nr.V2_PROTOCOL_MODEL,))
    d = o["models"][nr.V2_PROTOCOL_MODEL]
    bars = []
    for t, i in d["tiers"].items():
        if "numkeep" in t:
            continue
        assert i["silent_wrong"] + i["surfaced_wrong"] == i["wrong"]
        bars.append(dict(label=_win(i["loaded_context_values"], t == "ollama_default"), n=i["n_task_turns"],
                         wrong=i["wrong"], silent=i["silent_wrong"], loud=i["surfaced_wrong"], unknown=0))
    groups.append((f"Ollama, {nr.V2_PROTOCOL_MODEL}, AMD APU", bars))
    # llama-server, qwen3:14b
    xrt = nr._xrt_module()
    o = xrt.outcome(xrt.ag.read_rows(nr.REPO / nr.XRT_FILE))
    bars = []
    for k in ("default", "4096"):
        a = o["arms"][k]
        s = a["harness_silent_wrong"]
        bars.append(dict(label=_win(a["loaded_n_ctx"], k == "default"), n=a["n_task_turns"], wrong=a["wrong"],
                         silent=s, loud=a["wrong"] - s, unknown=0))
    groups.append((f"llama-server, {nr.V2_PROTOCOL_MODEL}, AMD APU", bars))
    # Ollama, qwen3:8b, second workload
    z = nr._zach_pkg()
    sc = z.jload("1_t4_8b/scores/bridge1_scores.json")
    rc = z.t4_recompute()
    bars = []
    for arm, default in (("bridge1_P", False), ("bridge1_D", True)):
        r, s = rc[arm], sc["arms"][arm]
        assert (r["wrong"], r["turns"]) == (s["wrong"], s["task_turns"])
        if default:
            assert sc["D_failures"] == r["wrong"]
            sil = sc["D_silent_failures"]
            b = dict(silent=sil, loud=r["wrong"] - sil, unknown=0)
        else:
            b = dict(silent=0, loud=0, unknown=r["wrong"])
        bars.append(dict(label=_win(s["loaded_context"], default), n=r["turns"], wrong=r["wrong"], **b))
    groups.append(("Ollama, qwen3:8b, Intel mini PC", bars))
    return groups


def draw(out_dir=None):
    from matplotlib.patches import Patch
    style()
    groups = load()
    fig, ax = plt.subplots(figsize=(COL_IN, 2.75))
    y, ticks, labels, heads = 0.0, [], [], []
    for title, bars in groups:
        heads.append((y, title))
        y += 0.85
        for b in bars:
            n = b["n"]
            left = 0.0
            for key, col, hatch in (("silent", SILENT, None), ("loud", LOUD, "////"), ("unknown", UNKNOWN, "...")):
                w = 100.0 * b[key] / n
                if w > 0:
                    ax.barh(y, w, left=left, height=0.72, color=col, hatch=hatch, edgecolor="white", lw=0.4)
                    left += w
            ax.text(left + 1.5, y, f"{b['wrong']}/{n}", va="center", ha="left")
            ticks.append(y)
            labels.append(b["label"])
            y += 1.0
        y += 0.15
    for yh, title in heads:
        ax.text(0.5, yh, title, ha="left", va="center", fontweight="bold")
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(y - 0.6, -0.6)
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("task turns ended wrong (%)")
    ax.set_ylabel("loaded window (tokens)")
    fig.legend(handles=[Patch(fc=SILENT, ec="white", label="silent"),
                       Patch(fc=LOUD, ec="white", hatch="////", label="loud (HTTP error)"),
                       Patch(fc=UNKNOWN, ec="white", hatch="...", label="silence not recorded")],
              loc="outside upper center", ncol=2, frameon=False, handlelength=1.2, columnspacing=0.8, labelspacing=0.2,
              handletextpad=0.3)
    return save(fig, NAME, out_dir)


def main(out_dir=None):
    return draw(out_dir)


if __name__ == "__main__":
    print(main())
