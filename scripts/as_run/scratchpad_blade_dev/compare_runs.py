"""Compare 203557Z (new, cache-off) vs 191031Z (old, cache-on)."""
import json, sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from collections import defaultdict

REPO = Path("C:/Users/rithw/OneDrive/Documents/GitHub/APU")
OLD  = REPO / "results/fig61_stagec_full_20260922T191031Z.jsonl"
NEW  = REPO / "results/fig61_stagec_full_20260922T203557Z.jsonl"

def load(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

old_rows = load(OLD)
new_rows = load(NEW)
print(f"Old: {len(old_rows)} rows   New: {len(new_rows)} rows")

# ── 1. Score comparison cell-by-cell ─────────────────────────────────────────
# Key: (probe_id, arm, budget_ratio, rep)
def key(r):
    return (r["probe_id"], r["arm"], r["budget_ratio"], r["rep"])

old_by_key = {key(r): r for r in old_rows}
new_by_key = {key(r): r for r in new_rows}

diffs = []
for k, nr in new_by_key.items():
    or_ = old_by_key.get(k)
    if or_ is None:
        continue
    if or_["score"] != nr["score"]:
        diffs.append((k, or_["score"], nr["score"], nr.get("output","")[:60], or_.get("output","")[:60]))

print(f"\n── Score comparison ──────────────────────────────────────")
if not diffs:
    print("  Scores match OLD cell-for-cell (0 differences)")
else:
    print(f"  {len(diffs)} cells differ:")
    for (pid, arm, ratio, rep), os_, ns_, nout, oout in sorted(diffs):
        print(f"  {pid} {arm} r={ratio:.2f} rep={rep}: old={os_} new={ns_}")
        print(f"    old_out={repr(oout)}  new_out={repr(nout)}")

# ── 2. PC pass rate ───────────────────────────────────────────────────────────
print(f"\n── Positive control ──────────────────────────────────────")
pc_pass = sum(1 for r in new_rows if r["positive_control_ok"])
pc_fail_rows = [r for r in new_rows if not r["positive_control_ok"]]
print(f"  Pass: {pc_pass}/{len(new_rows)}")
if pc_fail_rows:
    print(f"  Failures ({len(pc_fail_rows)}):")
    for r in pc_fail_rows[:20]:
        pct = abs(r["n_prompt_tokens_actual"] - r["intended_budget_tokens"]) / max(r["intended_budget_tokens"],1)*100
        print(f"    {r['probe_id']} {r['arm']} r={r['budget_ratio']:.2f} rep={r['rep']}: "
              f"actual={r['n_prompt_tokens_actual']} intended={r['intended_budget_tokens']} "
              f"err={pct:.1f}%")

# ── 3. tokens_in_api present? ─────────────────────────────────────────────────
api_nonzero = sum(1 for r in new_rows if r.get("tokens_in_api", 0) > 0)
print(f"\n── API usage tokens returned ─────────────────────────────")
print(f"  tokens_in_api > 0: {api_nonzero}/{len(new_rows)}")
if api_nonzero:
    sample = next(r for r in new_rows if r.get("tokens_in_api", 0) > 0)
    gap = sample["tokens_in_api"] - sample["n_prompt_tokens_actual"]
    print(f"  Sample: actual={sample['n_prompt_tokens_actual']} api={sample['tokens_in_api']} gap={gap:+d} (chat template)")

# ── 4. TTFT by ratio and arm ──────────────────────────────────────────────────
print(f"\n── TTFT by budget_ratio and arm (new run, ms) ────────────")
ttft_cells: dict[tuple, list] = defaultdict(list)
lat_cells:  dict[tuple, list] = defaultdict(list)
for r in new_rows:
    k2 = (r["budget_ratio"], r["arm"])
    if r.get("ttft_ms"):
        ttft_cells[k2].append(r["ttft_ms"])
    if r.get("latency_ms"):
        lat_cells[k2].append(r["latency_ms"])

print(f"  {'ratio':>6}  {'arm':>5}  {'ttft_mean':>10}  {'ttft_min':>9}  {'ttft_max':>9}  {'lat_mean':>9}")
for ratio in [1.20, 1.00, 0.85, 0.70, 0.55, 0.40]:
    for arm in ["LATE", "EARLY"]:
        vals = ttft_cells.get((ratio, arm), [])
        lats = lat_cells.get((ratio, arm), [])
        if vals:
            print(f"  {ratio:>6.2f}  {arm:>5}  {sum(vals)/len(vals):>10.0f}  "
                  f"{min(vals):>9.0f}  {max(vals):>9.0f}  "
                  f"{sum(lats)/len(lats):>9.0f}")

# ── 5. Wall clock ─────────────────────────────────────────────────────────────
print(f"\n── Wall-clock estimate ───────────────────────────────────")
total_lat = sum(r["latency_ms"] for r in new_rows if r.get("latency_ms"))
print(f"  Sum of all latency_ms: {total_lat/1000:.0f}s  ({total_lat/60000:.1f} min)")
print(f"  (Does not include filler-build or cache-check overhead)")
