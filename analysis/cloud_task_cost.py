"""Cloud cost per real agent task (offline, no API call, no machine time).

Per trajectory of the real agent traces in results/traces/agent_step_lengths.parquet (the two uncensored sources
already used by row uncensored-trace-32k-crossing: nebius/SWE-rebench-openhands-trajectories and
SWE-Gym/OpenHands-Sampled-Trajectories), every step re-sends its full prompt (tokens_qwen is the full prompt at that
step). Billed input = sum of the step prompts; billed output = steps x a per-step output stand-in. USD at two tiers of
the shared Anthropic price file (cheapest claude-haiku-5-5, mid claude-sonnet-5-5; Standard, 5-minute cache writes)
for three caching cases:
  none:       every input token at the uncached rate;
  realistic:  ASSUMPTION, a hit rate h (default 0.7) of the re-sent prefix is read from cache, the rest of the
              re-sent prefix and all new tokens are written at the cache-write premium;
  best:       every re-sent prefix is read from cache, only the new tokens are written.
Re-sent prefix of a step = min(previous prompt, this prompt) when the previous prompt reaches the model's minimum
cacheable prefix (cloud_budget_forecast.resent_tokens); a prompt under that minimum is never written to cache.
Output tokens per step are NOT in the trace data. Stand-in: the median per-call reply length (o200k_base) of the
measured R2 agent sessions, results/x2_r2_real_v1.jsonl (cloud_budget_forecast.output_lengths' r2_sessions_per_turn
basis). Token basis: the qwen tokenizer of the trace file, not Anthropic's tokenizer (an assumption).
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
import cloud_budget_forecast as cb  # noqa: E402

TRACE_FILE = "results/traces/agent_step_lengths.parquet"
PRICE_FILE = "results/cloud_prices_anthropic_20261014.json"
R2_FILE = "results/x2_r2_real_v1.jsonl"
TRACE_SETS = (("nebius/SWE-rebench-openhands-trajectories", "nebius-rebench-openhands"),
              ("SWE-Gym/OpenHands-Sampled-Trajectories", "SWE-Gym"))
TIERS = ("claude-haiku-5-5", "claude-sonnet-5-5")
CASES = ("none", "realistic", "best")
HIT_RATE = 0.7  # ASSUMPTION: share of the re-sent prefix read from cache in the realistic case


def load_prices(repo=REPO) -> dict:
    return json.loads((Path(repo) / PRICE_FILE).read_text(encoding="utf-8"))


def output_standin(repo=REPO) -> dict:
    """Per-step output tokens stand-in: median o200k_base reply length per call in the measured R2 sessions."""
    vals = [cb.r2_call_output_tokens(c) for r in cb._jsonl(Path(repo) / R2_FILE) if r.get("record") == "r2a_turn"
            for c in r["calls"]]
    if not vals:
        raise FileNotFoundError(Path(repo) / R2_FILE)
    return {"tokens": statistics.median(vals), "n_calls": len(vals), "basis": f"median per-call reply, {R2_FILE}"}


def task_cost(prices, model, prompts, out_per_step, case, hit_rate=HIT_RATE) -> float:
    """USD for one trajectory: prompts = full prompt tokens per step in order."""
    mc = cb.min_cacheable(prices, model)
    resent = cb.resent_tokens([(p, 0) for p in prompts], mc)
    total = 0.0
    for p, r in zip(prompts, resent):
        new = p - r
        if case == "none" or p < mc:
            cached, write = 0, 0
        elif case == "best":
            cached, write = r, new
        elif case == "realistic":
            cached, write = hit_rate * r, new + (1.0 - hit_rate) * r
        else:
            raise ValueError(case)
        total += cb.call_cost(prices, model, "standard", p, out_per_step, cached=cached, write=write)
    return total


def _q(vals, q):
    s = sorted(vals)
    if not s:
        return None
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(trajs: dict, prices, out_per_step, hit_rate=HIT_RATE) -> dict:
    """trajs: {label: [[prompt tokens per step], ...]} -> per set: tokens and USD median/p90 per tier per case."""
    out = {}
    for label, plist in trajs.items():
        bin_ = [sum(p) for p in plist]
        bout = [len(p) * out_per_step for p in plist]
        e = {"n_tasks": len(plist), "steps_median": _q([len(p) for p in plist], 0.5),
             "input_median": _q(bin_, 0.5), "input_p90": _q(bin_, 0.9),
             "output_median": _q(bout, 0.5), "output_p90": _q(bout, 0.9), "usd": {}}
        for m in TIERS:
            for c in CASES:
                v = [task_cost(prices, m, p, out_per_step, c, hit_rate) for p in plist]
                e["usd"][(m, c)] = {"median": _q(v, 0.5), "p90": _q(v, 0.9)}
        out[label] = e
    return out


def load_trajectories(repo=REPO) -> dict:
    import pandas as pd
    df = pd.read_parquet(Path(repo) / TRACE_FILE)
    out = {}
    for ds, label in TRACE_SETS:
        sub = df[df["dataset"] == ds].sort_values(["trajectory_id", "step_idx"])
        if sub.empty:
            raise FileNotFoundError(f"no rows for {ds}")
        out[label] = [g["tokens_qwen"].astype(int).tolist() for _, g in sub.groupby("trajectory_id")]
    return out


def compute(repo=REPO, hit_rate=HIT_RATE) -> dict:
    so = output_standin(repo)
    return {"standin": so, "hit_rate": hit_rate,
            "sets": summarize(load_trajectories(repo), load_prices(repo), so["tokens"], hit_rate)}


def value_text(res) -> str:
    so, parts = res["standin"], []
    for label, e in res["sets"].items():
        segs = []
        for m in TIERS:
            segs.append(f"{m}: " + ", ".join(
                f"{c} ${e['usd'][(m, c)]['median']:.4f} / ${e['usd'][(m, c)]['p90']:.4f}" for c in CASES))
        parts.append(f"{label} (n={e['n_tasks']} tasks, median {e['steps_median']:.0f} steps; billed input tokens "
                     f"median {e['input_median']:,.0f} p90 {e['input_p90']:,.0f}; billed output median "
                     f"{e['output_median']:,.0f} p90 {e['output_p90']:,.0f}): " + "; ".join(segs))
    return ("USD per real agent task, median / p90, priced, not executed (no API call): " + " | ".join(parts)
            + f"; cases: none = all input uncached, realistic = ASSUMPTION cache hit rate {res['hit_rate']:.1f} of the "
              f"re-sent prefix (rest written at the 5-minute cache-write premium), best = every re-sent prefix read "
              f"from cache, only new tokens written; each step re-sends its full prompt; output tokens per step are "
              f"not in the trace data, STAND-IN {so['tokens']:.0f} tokens per step ({so['basis']}, n={so['n_calls']} "
              f"calls); prices {PRICE_FILE} Standard; token basis qwen tokenizer, not Anthropic's (assumption)")


if __name__ == "__main__":
    print(value_text(compute()))
