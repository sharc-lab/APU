"""Figure F1: platform and memory-architecture diagram (a drawing; every memory size on it comes from a register row).

One horizontal lane per platform on a common MiB scale, showing the memory the GPU is meant to use, where weights,
KV cache and host-visible buffers live, what happens past the boundary, and the runtime backend.

Register rows and what is read from each:
- fh1-amd-boundary-llama-3.3-70b (numbers_register.compute_fh1_amd_boundary_llama70b; results/mx2_spill_stats.jsonl,
  results/mx2_validation.jsonl, results/t2s_night2_20261001T151340Z.jsonl, results/x2_70b_edge_reps.jsonl):
  `detail.dedicated_mib`, `detail.device_local_mib`, `detail.vulkan_total_mib` for the AMD APU lane, and the
  per-point `spill` field (spill goes to shared memory past the dedicated block).
- A-24-effective-heap-limit (numbers_register.compute_a24_effective_heap_limit; results/t2s_amech_20260926T181456Z.jsonl,
  results/t2s_amech_20260926T181456Z_srv_bis_*.txt, results/t2s_night2_20260929T034014Z_srv_bis_a70_*.txt):
  the fitted limit bracket `detail.tightest_lower` / `detail.tightest_upper`, and the driver budget from
  numbers_register._a24_vulkan_budget_mib (the helper that compute function calls), for the Intel iGPU lane.
- blade-k1-layer-split (results/blade_k1_v2.summary.json, the one file its compute function reads): device VRAM
  `device.vram_total_mib`, runtime library `device.libraries`, and the smallest tested window at which a model is
  no longer fully in VRAM, for the NVIDIA lane.
- x2-device-detect-mechanism (numbers_register.compute_x2_device_detect_mechanism; results/x2_template_mechanism_test.jsonl):
  qualitative only (Ollama uses the AMD iGPU via ROCm; the Intel machine falls through to the CPU). The script
  checks the row's value string names ROCm and CPU before printing those labels.

Usage: py -3.12 analysis/figures/fig_f1.py   (writes docs/paper1/figures/fig_f1.pdf)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import COLORS, MUTED, REPO, entry, nr, plt, row, save, style  # noqa: E402

ROWS = ["fh1-amd-boundary-llama-3.3-70b", "A-24-effective-heap-limit", "blade-k1-layer-split",
        "x2-device-detect-mechanism"]
NAME = "fig_f1.pdf"
NL = "\n"
GPU = "#cfe3f3"     # memory the GPU is meant to use
SHARED = "#f6dccb"  # overflow / host memory


def load():
    amd = row(ROWS[0])["detail"]
    if not any(p["spill"] for p in amd["points"]):
        raise RuntimeError("AMD row has no spill point; the shared-memory arrow would not be backed")
    lim = row(ROWS[1])["detail"]
    budget, _ = nr._a24_vulkan_budget_mib(nr.REPO)
    s = json.loads((REPO / entry(ROWS[2])["data_files"][0]).read_text(encoding="utf-8"))
    first_off = min(o["num_ctx"] for v in s["models"].values() for o in v.get("overflow_window", [])
                    if o.get("size") and o["size_vram"] < o["size"])
    det = row(ROWS[3])["value"]
    if "ROCm" not in det or "CPU" not in det:
        raise RuntimeError(f"{ROWS[3]} no longer names ROCm and CPU: {det}")
    return amd, lim, budget, s["device"], first_off


def lane(ax, y, segs, h=0.5, fs=4.5):
    for x0, x1, fc, txt in segs:
        ax.add_patch(plt.Rectangle((x0, y - h / 2), x1 - x0, h, fc=fc, ec="#555555", lw=0.6, zorder=2))
        if txt:
            ax.text((x0 + x1) / 2, y, txt, ha="center", va="center", fontsize=fs, color="#222222", zorder=3, linespacing=1.0)


def tick(ax, x, y, txt, h=0.5, above=True):
    ax.plot([x, x], [y - h / 2 - 0.06, y + h / 2 + 0.06], color="#222222", lw=0.9, zorder=4)
    ax.text(x, y + (h / 2 + 0.09 if above else -h / 2 - 0.09), txt, ha="center", va="bottom" if above else "top",
            fontsize=4.6, color="#222222")


def draw(out_dir=None):
    style()
    amd, lim, budget, dev, first_off = load()
    ded, dl, tot = amd["dedicated_mib"], amd["device_local_mib"], amd["vulkan_total_mib"]
    l_lo, l_hi = lim["tightest_lower"][0], lim["tightest_upper"][0]
    vram = dev["vram_total_mib"]
    lib = ", ".join(dev.get("libraries") or [])

    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    ax.set_xlim(-1500, tot * 1.03)
    ax.set_ylim(-0.5, 3.75)
    ax.axis("off")
    fs = 4.6

    # AMD APU (top)
    y = 3.0
    ax.text(0, y + 0.47, "AMD APU: unified memory", fontsize=5.8, weight="bold", va="bottom")
    lane(ax, y, [(0, ded, GPU, "dedicated block: weights + KV cache"),
                 (ded, tot, SHARED, "shared memory:" + NL + "spill past the block," + NL + "no error")])
    tick(ax, ded, y, f"{ded:.0f}")
    tick(ax, dl, y, f"device-local heap {dl:.0f}", above=False)
    tick(ax, tot, y, f"{tot:.0f}")
    ax.text(0, y - 0.52, "llama-server: Vulkan; Ollama: iGPU via ROCm", fontsize=fs, color=MUTED, va="top")

    # Intel iGPU (middle)
    y = 1.6
    ax.text(0, y + 0.33, "Intel iGPU: unified memory, one shared heap", fontsize=5.8, weight="bold", va="bottom")
    lane(ax, y, [(0, l_hi, GPU, "one heap: weights + KV cache +" + NL + "compute + host-visible buffers")])
    ax.add_patch(plt.Rectangle((l_lo, y - 0.25), l_hi - l_lo, 0.5, fc="#333333", lw=0, zorder=3))
    ax.text(l_hi + 1500, y + 0.06, f"budget {budget:.0f}; limit L \u2264 {l_hi:.0f}", va="bottom", fontsize=fs)
    ax.plot([l_hi, l_hi], [y - 0.31, y + 0.31], color="#222222", lw=0.9, zorder=4)
    ax.text(l_hi + 1500, y - 0.04, "past L: start refused", fontsize=fs, va="top", color="#222222")
    ax.text(0, y - 0.34, "llama-server: Vulkan; Ollama: CPU as shipped", fontsize=fs, color=MUTED, va="top")

    # NVIDIA dGPU (bottom)
    y = 0.3
    ax.text(0, y + 0.33, "NVIDIA laptop dGPU: discrete VRAM", fontsize=5.8, weight="bold", va="bottom")
    lane(ax, y, [(0, vram, GPU, "")])
    ax.plot([vram, vram], [y - 0.31, y + 0.31], color="#222222", lw=0.9, zorder=4)
    ax.text(vram + 2000, y - 0.05, f"VRAM {vram}", fontsize=fs, va="top")
    ax.annotate("", xy=(vram + 54000, y), xytext=(vram + 1200, y),
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#222222"))
    lane(ax, y, [(vram + 54000, vram + 90000, SHARED, "CPU RAM (not to scale)")])
    ax.text(vram + 2000, y + 0.05, f"layers move at window \u2265 {first_off}", fontsize=fs, va="bottom")
    ax.text(0, y - 0.34, f"VRAM: weights + KV cache; Ollama: {lib}", fontsize=fs, color=MUTED, va="top")

    ax.text(tot * 1.03, -0.48, "bars in MiB, common scale", fontsize=fs, color=MUTED, ha="right", va="bottom")
    return save(fig, NAME, out_dir)


def main(out_dir=None):
    return draw(out_dir)


if __name__ == "__main__":
    print(main())
