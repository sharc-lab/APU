"""Trace-weighted reporting for results/workload_pack/.

grade.py reports the pack's p50/p90/p99 prompt-token lengths as a FLAT average over the 400 items (every
item counts once, regardless of how often a real agent step of that length actually happens). This module
re-weights each item by how often a step of about that many tokens actually occurs in the real, uncensored
agent-trace data, so the reported statistics reflect real agent-step demand instead of however the pack
generator happened to allocate items across length buckets.

METHOD (read this before trusting a number below -- every step is a real, documented choice):

1. Trace mass source: step-level `tokens_qwen` from BOTH real UNCENSORED sources in
   results/traces/agent_step_lengths.parquet --
     - nebius/SWE-rebench-openhands-trajectories (Qwen3-Coder-480B-A35B-Instruct, 256K native)
     - SWE-Gym/OpenHands-Sampled-Trajectories (gpt-4o-2024-08-06 / claude-3-5-sonnet-20241022, 128K/200K
       native)
   The other two dataset values present in that same parquet (nebius/SWE-agent-trajectories and
   Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k) are CENSORED at their generating model's own
   context ceiling -- see analysis/agent_traces.py's module docstring and its measured context-exit rate
   (results/traces/exit_status_sample.parquet) for the evidence -- and are deliberately excluded from the
   trace mass used here. Combined at STEP granularity, not deduplicated to one value per trajectory, because
   the question being answered is "how often does a step of about this many tokens occur", and every row in
   the parquet is one real occurrence of a step.

2. Buckets: log-spaced histogram bin edges (log-spaced, not linear, because token lengths span roughly 50 to
   120,000 tokens -- about 3.5 orders of magnitude -- and linear bins would put essentially all trace mass
   and all pack items into a single first bin). N_BINS = 30 edges running from min(trace_min, pack_min) to
   max(trace_max, pack_max), both with a 1% margin, so every one of the pack's 400 items falls inside the
   edge range even where it falls outside the trace data's own observed min/max (such an item lands in an
   edge bin that holds ~0 real trace mass, which correctly drives its weight toward 0 rather than erroring).

3. Per-item weight: each pack item's `prompt_tokens` places it in exactly one bin. That bin's trace
   probability mass (real uncensored step count in the bin / total real uncensored step count) is split
   EQUALLY across every pack item that falls in the same bin. This is a histogram-bucket density estimator,
   chosen over a KDE because every number in the computation is a plain, auditable count (bin edges, step
   counts per bin, item counts per bin) and because the pack's own items already cluster tightly within each
   family's own narrow token range (see grade.py's per-family output), so a smoother density would not move
   these numbers meaningfully.

4. Normalization: the raw per-item masses from step 3 are renormalized to sum to exactly 1.0 across the 400
   items. This step matters because some bins that hold real trace mass contain zero pack items -- that
   mass has no item to attach to and is simply absent from the 400-item sample -- so the raw per-item masses
   sum to LESS than 1 before this final renormalization.

5. Weighted statistic: standard weighted-percentile definition (sort by value, build the cumulative weight
   fraction, linearly interpolate between the two bracketing points) -- the same definition grade.py's own
   `pct()` uses for the unweighted case, so the two tables in compute_pack_stats() are comparable side by
   side.

Usage: py -3.12 analysis/trace_weighted_pack.py   (prints both tables; also importable)
"""
from __future__ import annotations

import json
from pathlib import Path

N_BINS = 30
UNCENSORED_DATASETS = (
    "nebius/SWE-rebench-openhands-trajectories",
    "SWE-Gym/OpenHands-Sampled-Trajectories",
)
FAMILY_FILES = [
    "a_longdoc_qa.jsonl",
    "b_function_calling.jsonl",
    "c_gsm8k.jsonl",
    "d_r2_sessions.jsonl",
    "e_trace_mix.jsonl",
]


def load_pack_items(repo: Path) -> list[dict]:
    items_dir = repo / "results" / "workload_pack" / "items"
    items = []
    for fname in FAMILY_FILES:
        path = items_dir / fname
        if not path.exists():
            raise FileNotFoundError(str(path))
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
    if not items:
        raise FileNotFoundError(f"no items loaded from {items_dir}")
    return items


def load_uncensored_trace_tokens(repo: Path):
    """Real step-level tokens_qwen from both real uncensored sources, combined. See module docstring step 1
    for which sources and why."""
    import pandas as pd

    path = repo / "results" / "traces" / "agent_step_lengths.parquet"
    if not path.exists():
        raise FileNotFoundError(str(path))
    df = pd.read_parquet(path)
    uncensored = df[df["dataset"].isin(UNCENSORED_DATASETS)]
    if uncensored.empty:
        raise FileNotFoundError(f"no rows matched {UNCENSORED_DATASETS} in {path}")
    return uncensored["tokens_qwen"].to_numpy(dtype=float)


def _log_bin_edges(trace_tokens, pack_tokens, n_bins=N_BINS):
    import numpy as np

    lo = min(float(trace_tokens.min()), float(min(pack_tokens))) * 0.99
    hi = max(float(trace_tokens.max()), float(max(pack_tokens))) * 1.01
    lo = max(lo, 1.0)
    return np.logspace(np.log10(lo), np.log10(hi), n_bins + 1)


def compute_item_weights(repo: Path):
    """Returns (items, weights) -- items is the real 400-item list (each dict unmodified, in load order),
    weights is a list of the same length, same order, of trace-mass-derived per-item weights summing to
    1.0 across all items. See module docstring for the exact method."""
    import numpy as np

    items = load_pack_items(repo)
    pack_tokens = [it["prompt_tokens"] for it in items]
    trace_tokens = load_uncensored_trace_tokens(repo)

    edges = _log_bin_edges(trace_tokens, pack_tokens)
    trace_bin_idx = np.clip(np.digitize(trace_tokens, edges) - 1, 0, N_BINS - 1)
    trace_counts = np.bincount(trace_bin_idx, minlength=N_BINS)
    total_trace_steps = int(trace_counts.sum())
    bin_mass = trace_counts / total_trace_steps  # sums to 1.0 over bins

    item_bin_idx = np.clip(np.digitize(pack_tokens, edges) - 1, 0, N_BINS - 1)
    items_per_bin = np.bincount(item_bin_idx, minlength=N_BINS)

    raw_weights = []
    for b in item_bin_idx:
        n_items_in_bin = items_per_bin[b]
        raw_weights.append(bin_mass[b] / n_items_in_bin if n_items_in_bin > 0 else 0.0)
    raw_weights = np.array(raw_weights, dtype=float)
    total = raw_weights.sum()
    if total <= 0:
        raise ValueError("all raw per-item weights are zero -- no pack item's bin overlaps any real trace mass")
    weights = raw_weights / total  # renormalize to sum to 1.0 across the 400 items (step 4)
    return items, weights.tolist(), {
        "bin_edges": edges.tolist(), "n_bins": N_BINS,
        "total_uncensored_trace_steps": total_trace_steps,
        "raw_weight_sum_before_renorm": float(total),
    }


def _grade_pct_module(repo: Path):
    """Imports results/workload_pack/grade.py's own pct() by file path, so the FLAT table in this module
    uses the exact same nearest-rank-interpolation percentile function grade.py already reports (not a
    different interpolation that would make the "flat" column not actually match grade.py's output)."""
    import importlib.util

    path = repo / "results" / "workload_pack" / "grade.py"
    spec = importlib.util.spec_from_file_location("workload_pack_grade", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def weighted_percentile(values, weights, p):
    """Standard weighted-percentile: sort by value, build the cumulative weight fraction, linearly
    interpolate. Matches grade.py's unweighted pct() when all weights are equal."""
    import numpy as np

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.sum() <= 0:
        # Every item in this subset carries zero real trace mass (its token length falls in a bin the real
        # uncensored traces never visit at that length) -- the weighted percentile is genuinely undefined,
        # not a bug, so this is reported as NaN rather than silently falling back to an unweighted value.
        return float("nan")
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w) - 0.5 * w
    cw = cw / w.sum()
    return float(np.interp(p, cw, v))


def compute_pack_stats(repo: Path):
    """Both tables side by side: flat (unweighted, matches grade.py's own pct() exactly, imported from
    grade.py so there is no risk of a second, subtly different percentile definition) and trace-weighted,
    overall and per family."""
    grade_mod = _grade_pct_module(repo)
    items, weights, meta = compute_item_weights(repo)
    families = sorted({it["family"] for it in items})

    def stats_for(idxs):
        toks = [items[i]["prompt_tokens"] for i in idxs]
        ws = [weights[i] for i in idxs]
        flat = {p: grade_mod.pct(toks, p) for p in (0.50, 0.90, 0.99)}
        traced = {p: weighted_percentile(toks, ws, p) for p in (0.50, 0.90, 0.99)}
        return {
            "n": len(idxs),
            "flat_p50": flat[0.50], "flat_p90": flat[0.90], "flat_p99": flat[0.99],
            "traced_p50": traced[0.50], "traced_p90": traced[0.90], "traced_p99": traced[0.99],
            "weight_mass": sum(ws),
        }

    overall = stats_for(range(len(items)))
    by_family = {}
    for fam in families:
        idxs = [i for i, it in enumerate(items) if it["family"] == fam]
        by_family[fam] = stats_for(idxs)

    return {"overall": overall, "by_family": by_family, "meta": meta}


def _fmt_row(label, s):
    return (f"  [{label}] n={s['n']} weight_mass={s['weight_mass']:.4f}  "
            f"flat p50={s['flat_p50']:.0f} p90={s['flat_p90']:.0f} p99={s['flat_p99']:.0f}  |  "
            f"traced p50={s['traced_p50']:.0f} p90={s['traced_p90']:.0f} p99={s['traced_p99']:.0f}")


def main():
    repo = Path(__file__).resolve().parents[1]
    result = compute_pack_stats(repo)
    print(f"bins={result['meta']['n_bins']} (log-spaced), "
          f"real uncensored trace steps={result['meta']['total_uncensored_trace_steps']}")
    print("\nFLAT (unweighted, matches grade.py) vs TRACE-WEIGHTED prompt-token percentiles:")
    print(_fmt_row("overall", result["overall"]))
    for fam, s in result["by_family"].items():
        print(_fmt_row(fam, s))


if __name__ == "__main__":
    main()
