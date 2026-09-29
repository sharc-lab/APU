"""One figure per claim, generated straight from committed/pulled result JSONL files, no new analysis beyond what
docs/CLAIMS_LEDGER.md already states. Every figure is labelled with hardware, model(s), and n. Run locally (matplotlib
only, no server, no SSH):

  py -3.12 analysis/make_paper_figures.py

Writes PNGs to figures/ and a short caption/data-source line per figure to figures/MANIFEST.md, which
docs/PAPER_OUTLINE.md links to. Each fig_* function takes an explicit input path (not a hardcoded results/ file) so
it works against a locally-committed copy or a copy pulled from evo-t2s/evo-x2, whichever exists.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
FIGURES = REPO / "figures"
FIGURES.mkdir(exist_ok=True)

MANIFEST_LINES: list[str] = []


def _load(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def _med(vals):
    vals = [v for v in vals if v is not None]
    return st.median(vals) if vals else None


def _note(fig_name, caption, source):
    MANIFEST_LINES.append(f"- **{fig_name}** -- {caption} Source: `{source}`.")


# ---------------------------------------------------------------- B1+B3: power effect (6 models)
def fig_power_effect(b1b3_path):
    rows = _load(b1b3_path)
    b1_models = ["qwen3-4b-2507", "qwen3-8b"]
    b3_models = ["llama31-8b", "qwen3-14b", "qwen3-30b-a3b-2507", "qwen3-32b"]
    coresets = ["none", "p4", "e4", "nonp12"]

    fig, ax = plt.subplots(figsize=(7, 5))
    for mid in b1_models + b3_models:
        section = "B1" if mid in b1_models else "B3"
        xs, ys = [], []
        base = None
        for co in coresets:
            calls = [r for r in rows if r.get("section") == section and r.get("kind") == "call"
                     and not r.get("warmup") and r.get("model_id") == mid and r.get("co_runner") == co]
            ttft = _med([r.get("ttft_s") for r in calls])
            if ttft is None:
                continue
            if base is None:
                base = ttft
            xs.append(co)
            ys.append(ttft / base)
        if ys:
            ax.plot(xs, ys, marker="o", label=mid)
    ax.set_ylabel("TTFT ratio vs none")
    ax.set_xlabel("co-runner core set")
    ax.set_title("evo-t2s: CPU co-runner slows the iGPU (power-budget effect), 6 models\n(night2 B1+B3, n=5/call, hw=evo-t2s Intel Arc B390 Vulkan)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIGURES / "fig_power_effect_b1_b3.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    _note("fig_power_effect_b1_b3.png",
          "TTFT ratio vs `none`, across `none/p4/e4/nonp12` co-runner core sets, 6 models (qwen3-4b-2507, qwen3-8b "
          "from B1; llama31-8b, qwen3-14b, qwen3-30b-a3b-2507, qwen3-32b from B3). hw=evo-t2s (Intel Arc B390, Vulkan).",
          b1b3_path.name)


# ---------------------------------------------------------------- B2: duty-cycle dose-response
def fig_b2_duty_cycle(b1b2_path):
    rows = _load(b1b2_path)
    models = ["qwen3-4b-2507", "qwen3-8b"]
    duties = [0, 25, 50, 75, 100]  # 0 = the B1 "none" condition, same server/session pattern
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    for mid in models:
        xs, ttft_ratios, dec_ratios = [], [], []
        base_calls = [r for r in rows if r.get("section") == "B1" and r.get("kind") == "call" and not r.get("warmup")
                      and r.get("model_id") == mid and r.get("co_runner") == "none"]
        base_ttft, base_dec = _med([r.get("ttft_s") for r in base_calls]), _med([r.get("decode_tok_s") for r in base_calls])
        for d in duties:
            if d == 0:
                calls = base_calls
            else:
                calls = [r for r in rows if r.get("section") == "B2" and r.get("kind") == "call" and not r.get("warmup")
                         and r.get("model_id") == mid and r.get("item_id") == f"B2_{mid}_duty{d}"]
            ttft, dec = _med([r.get("ttft_s") for r in calls]), _med([r.get("decode_tok_s") for r in calls])
            if ttft is None or base_ttft is None:
                continue
            xs.append(d)
            ttft_ratios.append(ttft / base_ttft)
            dec_ratios.append(dec / base_dec if dec and base_dec else None)
        ax1.plot(xs, ttft_ratios, marker="o", label=mid)
        ax2.plot(xs, dec_ratios, marker="o", label=mid)
    for ax, ylabel in ((ax1, "TTFT ratio vs 0% duty"), (ax2, "decode ratio vs 0% duty")):
        ax.set_xlabel("nonp12 co-runner duty cycle (%)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("evo-t2s: duty-cycle dose-response, 2 models (night2 B1/B2, n=5/call)")
    fig.tight_layout()
    out = FIGURES / "fig_b2_duty_cycle.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    _note("fig_b2_duty_cycle.png",
          "TTFT and decode ratio vs the nonp12 co-runner's duty cycle (0/25/50/75/100%), qwen3-4b-2507 and qwen3-8b. "
          "hw=evo-t2s (Intel Arc B390, Vulkan).", b1b2_path.name)


# ---------------------------------------------------------------- B4: E-core threshold
def fig_b4_ecore_threshold(b4_path):
    rows = _load(b4_path)
    b4 = [r for r in rows if r.get("section") == "B4" and r.get("kind") == "call" and not r.get("warmup")]
    conds = [("none", 0), ("e2", 2), ("lp4", 4), ("e4_clusterA", 4), ("e4_clusterB", 4), ("e4_split", 4),
             ("e6", 6), ("e8", 8), ("e8_lp4", 12)]
    base = [r for r in b4 if r["item_id"] == "B4_qwen3-8b_none"]
    base_ttft = _med([r.get("ttft_s") for r in base])
    xs, ys, labels = [], [], []
    for label, n_cores in conds:
        calls = [r for r in b4 if r["item_id"] == f"B4_qwen3-8b_{label}"]
        ttft = _med([r.get("ttft_s") for r in calls])
        if ttft is None or base_ttft is None:
            continue
        xs.append(n_cores)
        ys.append(ttft / base_ttft)
        labels.append(label)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xs, ys)
    for x, y, l in zip(xs, ys, labels):
        ax.annotate(l, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.axvspan(0, 5, alpha=0.08, color="green")
    ax.axvspan(5, 12, alpha=0.08, color="red")
    ax.set_xlabel("active E-core count")
    ax.set_ylabel("TTFT ratio vs none")
    ax.set_title("evo-t2s: iGPU clock/TTFT threshold at >=6 active E-cores\n(qwen3-8b, night3 B4, n=5/call, hw=evo-t2s Intel Arc B390 Vulkan)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIGURES / "fig_b4_ecore_threshold.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    _note("fig_b4_ecore_threshold.png",
          "TTFT ratio vs active E-core count across the 9 B4 conditions, qwen3-8b, n=1 model, 5 calls/condition. "
          "hw=evo-t2s (Intel Arc B390, Vulkan). Verified real L2-cache cluster layout (P0-3 individual, E4-7/E8-11/"
          "LP-E12-15 each shared).", b4_path.name)


# ---------------------------------------------------------------- C1: 8B memory lock
def fig_c1_memory_lock(c1_path):
    rows = _load(c1_path)
    # c1 and c1b share the same "C1" section tag in the data (only the item_id namespace and model differ); night3's
    # c1b only got one cell in before it was aborted (qwen3-14b, headroom 0) -- excluded here by model_id so this
    # figure is exactly the 18-cell qwen3-8b C1 sweep, not 18 + that 1 stray row.
    starts = {r["item_id"]: r for r in rows if r.get("section") == "C1" and r.get("kind") == "start"
             and r.get("model_id") == "qwen3-8b"}
    resp = [r for r in rows if r.get("record") == "c1_responsiveness"]
    resp_by_item = {r["item_id"]: r for r in resp}
    by_headroom = {0: {"ok": 0, "fail": 0, "resp": []}, -1: {"ok": 0, "fail": 0, "resp": []}, -2: {"ok": 0, "fail": 0, "resp": []}}
    for item_id, s in starts.items():
        hg = s.get("mem_headroom_gb")
        if hg not in by_headroom:
            continue
        bucket = by_headroom[hg]
        if s.get("valid"):
            bucket["ok"] += 1
        else:
            bucket["fail"] += 1
        r = resp_by_item.get(item_id)
        if r and r.get("resp_max_s") is not None:
            bucket["resp"].append(r["resp_max_s"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    hgs = [0, -1, -2]
    ok_counts = [by_headroom[h]["ok"] for h in hgs]
    fail_counts = [by_headroom[h]["fail"] for h in hgs]
    ax1.bar([str(h) for h in hgs], ok_counts, label="started ok", color="tab:green")
    ax1.bar([str(h) for h in hgs], fail_counts, bottom=ok_counts, label="device-lost crash", color="tab:red")
    ax1.set_xlabel("headroom (GB)")
    ax1.set_ylabel("cells (n)")
    ax1.set_title("start outcome")
    ax1.legend(fontsize=8)

    resp_max_by_hg = [by_headroom[h]["resp"] for h in hgs]
    ax2.boxplot(resp_max_by_hg, tick_labels=[str(h) for h in hgs])
    ax2.set_xlabel("headroom (GB)")
    ax2.set_ylabel("resp_max_s (local subprocess-launch latency)")
    ax2.set_title("responsiveness during the cell")
    fig.suptitle("evo-t2s: qwen3-8b memory lock, mmap on/off x 3 reps (night3 C1)\nhw=evo-t2s (Intel Arc B390, Vulkan)")
    fig.tight_layout()
    out = FIGURES / "fig_c1_memory_lock.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    _note("fig_c1_memory_lock.png",
          "Start outcome (ok vs ggml_vulkan device-lost crash) and responsiveness (resp_max_s) by memory headroom "
          "(0/-1/-2 GB), qwen3-8b, mmap on/off x 3 reps = 6 cells/headroom, n=18 cells total. hw=evo-t2s (Intel Arc "
          "B390, Vulkan).", c1_path.name)


# ---------------------------------------------------------------- A-24: budget boundary, 4 models
def fig_a24_budget_boundary(amech_path):
    rows = _load(amech_path)
    all_boundaries = [r for r in rows if r.get("record") == "bisect_result"]
    # Keep the LAST bisect_result per label (chronologically). qwen3-8b has two: an earlier probe that hit its
    # max_ctx_native ceiling (131072/131328, ~23,100 MiB -- a different boundary, not the shared memory budget) before
    # a later, correctly-configured probe reached the real memory-limited boundary at 305152/305408 (~47,770 MiB, the
    # value docs/CLAIMS_LEDGER.md A-24 actually cites); qwen3-32b's two entries are identical repeats confirming
    # stability. Taking the last one per label reproduces exactly the ledger's four cited numbers, nothing new.
    boundaries, seen = [], {}
    for r in all_boundaries:
        seen[r.get("label")] = r
    boundaries = list(seen.values())
    fig, ax = plt.subplots(figsize=(7, 5))
    labels, los, his = [], [], []
    for r in boundaries:
        labels.append(r.get("label") or r.get("model_id"))
        los.append(r.get("projected_mib_last_ok"))
        his.append(r.get("projected_mib_first_fail"))
    budget = boundaries[0].get("budget_B_mib") if boundaries else 47865.0
    idx = range(len(labels))
    ax.scatter(idx, los, color="tab:green", label="last ok (projected MiB)", zorder=3)
    ax.scatter(idx, his, color="tab:red", label="first fail (projected MiB)", zorder=3)
    for i in idx:
        ax.plot([i, i], [los[i], his[i]], color="gray", alpha=0.6, zorder=1)
    ax.axhline(budget, color="black", linestyle="--", label=f"Vulkan heap budget ({budget:.0f} MiB)")
    ax.set_xticks(list(idx))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("llama.cpp-projected device memory (MiB)")
    ax.set_title("evo-t2s: n_ctx bisection crosses the same memory budget, 4 models\n(t2s_amech, 256-token step, hw=evo-t2s Intel Arc B390 Vulkan)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIGURES / "fig_a24_budget_boundary.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    _note("fig_a24_budget_boundary.png",
          f"Last-ok/first-fail llama.cpp-projected device memory at the n_ctx bisection boundary (256-token step), "
          f"4 models (qwen3-32b, qwen3-8b, qwen3-30b-a3b-2507, llama31-8b), n=1 bisection each (32B repeated 3-4x, "
          f"stable). Budget line = vulkaninfo heap budget ({budget:.0f} MiB). hw=evo-t2s (Intel Arc B390, Vulkan).",
          amech_path.name)


# ---------------------------------------------------------------- Blade M1/C1: discrete-GPU spill onset curve
def fig_blade_c1_spill(blade_path):
    rows = _load(blade_path)
    calls = [r for r in rows if r.get("status") == "ok" and r.get("ttft_s") is not None]
    by_ctx = {}
    for r in calls:
        by_ctx.setdefault(r.get("ctx_size"), []).append(r["ttft_s"])
    ctxs = sorted(by_ctx)
    ttfts = [_med(by_ctx[c]) for c in ctxs]
    base = ttfts[0] if ttfts else None
    ratios = [t / base if base else None for t in ttfts]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ctxs, ratios, marker="o")
    ax.set_xlabel("context (tokens)")
    ax.set_ylabel(f"TTFT ratio vs ctx={ctxs[0] if ctxs else '?'}")
    ax.set_title("Blade RTX 4070 (discrete, OFF-TARGET): silent driver spillover onset\n(blade_m1_vram_spill, n=%d ctx points)" % len(ctxs))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIGURES / "fig_blade_c1_spill.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    _note("fig_blade_c1_spill.png",
          f"TTFT ratio vs context size, {len(ctxs)} context points ({', '.join(str(c) for c in ctxs)}). "
          f"hw=blade_rtx4070 (discrete RTX 4070 Laptop GPU, OFF-TARGET). Onset of the driver's silent Sysmem "
          f"Fallback spillover (A-20).", blade_path.name)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--night2", required=True, help="night2 results JSONL (has sections B1/B2/B3): "
                    "results/t2s_night2_20260928T004924Z.jsonl or a locally pulled copy of it")
    ap.add_argument("--night3", required=True, help="night3 results JSONL (has sections B4/C1): "
                    "results/t2s_night2_20260928T200748Z.jsonl or a locally pulled copy of it")
    ap.add_argument("--amech", required=True, help="t2s_amech results JSONL (A-24's bisect_result records): "
                    "results/t2s_amech_20260926T181456Z.jsonl, pulled from evo-t2s (not yet committed)")
    ap.add_argument("--blade-c1", default=str(REPO / "results" / "blade_m1_vram_spill_20260925T041415Z.jsonl"),
                    help="Blade M1 VRAM spill onset sweep JSONL, the file with real ttft_s/ctx_size rows "
                         "(already committed, this is its default path)")
    args = ap.parse_args()

    fig_power_effect(Path(args.night2))
    fig_b2_duty_cycle(Path(args.night2))
    fig_b4_ecore_threshold(Path(args.night3))
    fig_c1_memory_lock(Path(args.night3))
    fig_a24_budget_boundary(Path(args.amech))
    fig_blade_c1_spill(Path(args.blade_c1))

    manifest = FIGURES / "MANIFEST.md"
    manifest.write_text(
        "# Figures generated from committed/pulled result JSONL (no new analysis beyond docs/CLAIMS_LEDGER.md)\n\n"
        "Regenerate with `py -3.12 analysis/make_paper_figures.py`.\n\n" + "\n".join(MANIFEST_LINES) + "\n",
        encoding="utf-8")
    print(f"wrote {len(MANIFEST_LINES)} figures to {FIGURES}")


if __name__ == "__main__":
    main()
