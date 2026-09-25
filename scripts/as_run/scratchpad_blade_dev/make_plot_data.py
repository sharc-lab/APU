"""Compute Fig 6.1 plot JSON from full-run JSONL."""
import json, sys
from collections import defaultdict
from pathlib import Path

src = Path("C:/Users/rithw/OneDrive/Documents/GitHub/APU/results/fig61_full_20260922T060942Z.jsonl")
rows = [json.loads(l) for l in src.read_text("utf-8").splitlines() if l.strip()]

RATIOS = [1.20, 1.00, 0.85, 0.70, 0.55, 0.40]
ARMS   = ["LATE", "EARLY"]
PROBES = ["rag_01", "rag_02", "rag_05", "sea_01", "sea_04"]
CATS   = {"rag": ["rag_01", "rag_02", "rag_05"], "search": ["sea_01", "sea_04"]}

def cell_mean(rows, probe=None, arm=None, ratio=None, cat=None, pc_filter=False):
    sel = [r for r in rows
           if (probe is None or r["probe_id"] == probe)
           and (arm is None or r["arm"] == arm)
           and (ratio is None or abs(r["budget_ratio"] - ratio) < 0.01)
           and (cat is None or r["probe_id"] in CATS[cat])
           and r["score"] is not None
           and (not pc_filter or r["positive_control_ok"])]
    if not sel:
        return None, 0
    return sum(r["score"] for r in sel) / len(sel), len(sel)

# ── aggregated by arm × ratio ──────────────────────────────────────────────────
agg_all = {}
for ratio in RATIOS:
    for arm in ARMS:
        mean, n = cell_mean(rows, arm=arm, ratio=ratio)
        agg_all[f"{ratio:.2f}_{arm}"] = {"mean": round(mean, 4), "n": n}

# ── per-category by arm × ratio ───────────────────────────────────────────────
agg_cat = {}
for cat in CATS:
    agg_cat[cat] = {}
    for ratio in RATIOS:
        for arm in ARMS:
            mean, n = cell_mean(rows, arm=arm, ratio=ratio, cat=cat)
            agg_cat[cat][f"{ratio:.2f}_{arm}"] = {"mean": round(mean, 4), "n": n}

# ── per-probe by arm × ratio ──────────────────────────────────────────────────
per_probe = {}
for probe in PROBES:
    per_probe[probe] = {}
    for ratio in RATIOS:
        for arm in ARMS:
            mean, n = cell_mean(rows, probe=probe, arm=arm, ratio=ratio)
            per_probe[probe][f"{ratio:.2f}_{arm}"] = {"mean": round(mean, 4), "n": n}

# ── positive control summary ──────────────────────────────────────────────────
pc_total = len(rows)
pc_ok    = sum(1 for r in rows if r["positive_control_ok"])
pc_fail_rows = [
    {"probe_id": r["probe_id"], "arm": r["arm"],
     "budget_ratio": r["budget_ratio"], "rep": r["rep"],
     "intended": r["intended_budget_tokens"],
     "actual": r["n_prompt_tokens_actual"]}
    for r in rows if not r["positive_control_ok"]
]

# ── summary table for display ─────────────────────────────────────────────────
print(f"\n{'Ratio':>8}  {'LATE_all':>10}  {'EARLY_all':>10}  {'delta':>8}")
for ratio in RATIOS:
    lm = agg_all[f"{ratio:.2f}_LATE"]["mean"]
    em = agg_all[f"{ratio:.2f}_EARLY"]["mean"]
    print(f"{ratio:>8.2f}  {lm:>10.3f}  {em:>10.3f}  {em-lm:>+8.3f}")

print(f"\n--- RAG probes (rag_01, rag_02, rag_05) ---")
print(f"{'Ratio':>8}  {'LATE':>8}  {'EARLY':>8}  {'delta':>8}")
for ratio in RATIOS:
    lm = agg_cat["rag"][f"{ratio:.2f}_LATE"]["mean"]
    em = agg_cat["rag"][f"{ratio:.2f}_EARLY"]["mean"]
    print(f"{ratio:>8.2f}  {lm:>8.3f}  {em:>8.3f}  {em-lm:>+8.3f}")

print(f"\n--- Search probes (sea_01, sea_04) ---")
print(f"{'Ratio':>8}  {'LATE':>8}  {'EARLY':>8}  {'delta':>8}")
for ratio in RATIOS:
    lm = agg_cat["search"][f"{ratio:.2f}_LATE"]["mean"]
    em = agg_cat["search"][f"{ratio:.2f}_EARLY"]["mean"]
    print(f"{ratio:>8.2f}  {lm:>8.3f}  {em:>8.3f}  {em-lm:>+8.3f}")

print(f"\nPositive control: {pc_ok}/{pc_total} OK")
if pc_fail_rows:
    print("PC failures:")
    for r in pc_fail_rows:
        devpct = abs(r["actual"] - r["intended"]) / max(r["intended"], 1) * 100
        print(f"  {r['probe_id']} {r['arm']} ratio={r['budget_ratio']} rep={r['rep']}: intended={r['intended']} actual={r['actual']} dev={devpct:.1f}%")

# ── output JSON ────────────────────────────────────────────────────────────────
out = {
    "source_file":    "fig61_full_20260922T060942Z.jsonl",
    "run_ts":         "20260922T060942Z",
    "n_rows":         len(rows),
    "n_probes":       len(PROBES),
    "probe_ids":      PROBES,
    "budget_ratios":  RATIOS,
    "arms":           ARMS,
    "reps":           3,
    "hardware":       "evo-t2s / Intel Arrow Lake / unified LPDDR5X 64 GB / NOT the Strix Halo BOM target",
    "platform":       "evo-t2s",
    "backend":        "llama-server b10970 Vulkan",
    "model":          "Qwen3-4B-Q4_K_M.gguf",
    "n_ctx_slot":     8192,
    "filler_tokens":  3997,
    "count_method":   "llamaserver_tokenize",
    "reasoning_budget": 0,
    "positive_control": {"ok": pc_ok, "total": pc_total, "failures": pc_fail_rows},
    "note_late_arm":  "LATE = [filler][artifact][question]; artifact is at END of prompt after 4000-token filler",
    "note_early_arm": "EARLY = [artifact][filler][question]; artifact is at START of prompt",
    "note_signal":    (
        "LATE outperforms EARLY aggregated (search probes drive this: records adjacent to question at end). "
        "RAG probes: LATE fails at all ratios (artifact buried at position ~4000 not attended to without CoT). "
        "Search probes: LATE=1.0 at ratios 0.85-0.55 (records near question), EARLY collapses at first truncating ratio. "
        "Both categories collapse to 0 when artifact is removed by truncation. "
        "Effect is probe-category dependent: position-near-question beats position-at-start for search; "
        "position-at-start may beat burial for RAG when CoT is disabled."
    ),
    "aggregated_all":        {k: v for k, v in agg_all.items()},
    "aggregated_by_category": agg_cat,
    "per_probe":             per_probe,
}

dest = Path("C:/Users/rithw/OneDrive/Documents/GitHub/APU/results/fig61_plot_data.json")
dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(f"\nPlot data written to {dest}")
