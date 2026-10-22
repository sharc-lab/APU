"""Cloud cost per real agent task (offline pricing, no machine time).

Per trajectory of the real agent traces in results/traces/agent_step_lengths.parquet (the two uncensored sources
already used by row uncensored-trace-32k-crossing: nebius/SWE-rebench-openhands-trajectories and
SWE-Gym/OpenHands-Sampled-Trajectories), every step re-sends its full prompt (tokens_qwen is the full prompt at that
step). Billed input = sum of the step prompts; billed output = sum of the MEASURED output tokens of each step (what
the trace's assistant generated at that step: content + tool_calls, results/traces/agent_step_outputs.parquet, from
scripts/trace_output_calibration.py --extract). USD at two tiers of the shared Anthropic price file (cheapest
claude-haiku-5-5, mid claude-sonnet-5-5; Standard, 5-minute cache writes) for three caching cases:
  none:       every input token at the uncached rate;
  realistic:  ASSUMPTION, a hit rate h (default 0.7) of the re-sent prefix is read from cache, the rest of the
              re-sent prefix and all new tokens are written at the cache-write premium;
  best:       every re-sent prefix is read from cache, only the new tokens are written.
Re-sent prefix of a step = min(previous prompt, this prompt) when the previous prompt reaches the model's minimum
cacheable prefix (cloud_budget_forecast.resent_tokens); a prompt under that minimum is never written to cache.
Token basis: the trace file counts are qwen3 tokens; every count (input and output) is multiplied by the trace set's
median Anthropic/qwen ratio measured with Anthropic's messages.count_tokens endpoint on a sample of 20 trajectories
per set (results/anthropic_token_calibration_20261021.jsonl).
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
OUTPUT_FILE = "results/traces/agent_step_outputs.parquet"
CALIB_FILE = "results/anthropic_token_calibration_20261021.jsonl"
PRICE_FILE = "results/cloud_prices_anthropic_20261014.json"
TRACE_SETS = (("nebius/SWE-rebench-openhands-trajectories", "nebius-rebench-openhands"),
              ("SWE-Gym/OpenHands-Sampled-Trajectories", "SWE-Gym"))
TIERS = ("claude-haiku-5-5", "claude-sonnet-5-5")
CASES = ("none", "realistic", "best")
HIT_RATE = 0.7  # ASSUMPTION: share of the re-sent prefix read from cache in the realistic case


def load_prices(repo=REPO) -> dict:
    return json.loads((Path(repo) / PRICE_FILE).read_text(encoding="utf-8"))


def calibration(records) -> dict:
    """{label: {median, min, max, p10, p90, n}} of anthropic_tokens / qwen_tokens per trace set, plus "all"."""
    labels = dict(TRACE_SETS)
    by = {}
    for r in records:
        if r.get("record") == "count_tokens" and r.get("qwen_tokens"):
            ratio = r["anthropic_tokens"] / r["qwen_tokens"]
            by.setdefault(labels.get(r["dataset"], r["dataset"]), []).append(ratio)
            by.setdefault("all", []).append(ratio)
    return {k: {"median": statistics.median(v), "min": min(v), "max": max(v), "p10": _q(v, 0.1), "p90": _q(v, 0.9),
                "n": len(v)} for k, v in by.items()}


def load_calibration(repo=REPO) -> dict:
    p = Path(repo) / CALIB_FILE
    if not p.exists():
        raise FileNotFoundError(p)
    return calibration([json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()])


def task_cost(prices, model, prompts, outputs, case, hit_rate=HIT_RATE) -> float:
    """USD for one trajectory: prompts = full prompt tokens per step in order, outputs = output tokens per step."""
    mc = cb.min_cacheable(prices, model)
    resent = cb.resent_tokens([(p, 0) for p in prompts], mc)
    total = 0.0
    for p, o, r in zip(prompts, outputs, resent):
        new = p - r
        if case == "none" or p < mc:
            cached, write = 0, 0
        elif case == "best":
            cached, write = r, new
        elif case == "realistic":
            cached, write = hit_rate * r, new + (1.0 - hit_rate) * r
        else:
            raise ValueError(case)
        total += cb.call_cost(prices, model, "standard", p, o, cached=cached, write=write)
    return total


def _q(vals, q):
    s = sorted(vals)
    if not s:
        return None
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(trajs: dict, prices, factors: dict, hit_rate=HIT_RATE) -> dict:
    """trajs: {label: [(prompts per step, outputs per step), ...]} in qwen tokens; factors: {label: Anthropic/qwen}.
    Returns per set: calibrated tokens and USD median/p90 per tier per case."""
    out = {}
    for label, tlist in trajs.items():
        f = factors[label]
        cal = [([p * f for p in ps], [o * f for o in os_]) for ps, os_ in tlist]
        bin_ = [sum(ps) for ps, _ in cal]
        bout = [sum(os_) for _, os_ in cal]
        step_out = [o for _, os_ in cal for o in os_]
        e = {"n_tasks": len(cal), "factor": f, "steps_median": _q([len(ps) for ps, _ in cal], 0.5),
             "input_median": _q(bin_, 0.5), "input_p90": _q(bin_, 0.9),
             "output_median": _q(bout, 0.5), "output_p90": _q(bout, 0.9),
             "output_per_step_median": _q(step_out, 0.5), "usd": {}}
        for m in TIERS:
            for c in CASES:
                v = [task_cost(prices, m, ps, os_, c, hit_rate) for ps, os_ in cal]
                e["usd"][(m, c)] = {"median": _q(v, 0.5), "p90": _q(v, 0.9)}
        out[label] = e
    return out


def load_trajectories(repo=REPO) -> dict:
    """{label: [(prompt tokens per step, measured output tokens per step)]} in qwen tokens."""
    import pandas as pd
    df = pd.read_parquet(Path(repo) / TRACE_FILE)
    outs = pd.read_parquet(Path(repo) / OUTPUT_FILE)
    m = df.merge(outs, on=["dataset", "trajectory_id", "step_idx"], how="left")
    out = {}
    for ds, label in TRACE_SETS:
        sub = m[m["dataset"] == ds].sort_values(["trajectory_id", "step_idx"])
        if sub.empty:
            raise FileNotFoundError(f"no rows for {ds}")
        if sub["output_tokens_qwen"].isna().any():
            raise ValueError(f"{ds}: {int(sub['output_tokens_qwen'].isna().sum())} steps without a measured output")
        out[label] = [(g["tokens_qwen"].astype(int).tolist(), g["output_tokens_qwen"].astype(int).tolist())
                      for _, g in sub.groupby("trajectory_id")]
    return out


def compute(repo=REPO, hit_rate=HIT_RATE) -> dict:
    cal = load_calibration(repo)
    factors = {label: cal[label]["median"] for _, label in TRACE_SETS}
    return {"calibration": cal, "hit_rate": hit_rate,
            "sets": summarize(load_trajectories(repo), load_prices(repo), factors, hit_rate)}


def calibration_text(cal) -> str:
    return "; ".join(f"{k} median {v['median']:.3f} (range {v['min']:.3f} to {v['max']:.3f}, p10 {v['p10']:.3f} p90 "
                     f"{v['p90']:.3f}, n={v['n']})" for k, v in cal.items())


def value_text(res) -> str:
    parts = []
    for label, e in res["sets"].items():
        segs = []
        for m in TIERS:
            segs.append(f"{m}: " + ", ".join(
                f"{c} ${e['usd'][(m, c)]['median']:.4f} / ${e['usd'][(m, c)]['p90']:.4f}" for c in CASES))
        parts.append(f"{label} (n={e['n_tasks']} tasks, median {e['steps_median']:.0f} steps; Anthropic tokens = qwen "
                     f"x {e['factor']:.3f}; billed input tokens median {e['input_median']:,.0f} p90 "
                     f"{e['input_p90']:,.0f}; billed output median {e['output_median']:,.0f} p90 "
                     f"{e['output_p90']:,.0f}, median {e['output_per_step_median']:,.0f} per step): " + "; ".join(segs))
    return ("USD per real agent task, median / p90, priced, not executed (no generation call): " + " | ".join(parts)
            + f"; cases: none = all input uncached, realistic = ASSUMPTION cache hit rate {res['hit_rate']:.1f} of the "
              f"re-sent prefix (rest written at the 5-minute cache-write premium), best = every re-sent prefix read "
              f"from cache, only new tokens written; each step re-sends its full prompt; output tokens per step "
              f"MEASURED from the traces (assistant content + tool calls per step, {OUTPUT_FILE}); token calibration "
              f"Anthropic/qwen from messages.count_tokens on the final-step prompt of 20 sampled tasks per set "
              f"({CALIB_FILE}): " + calibration_text(res["calibration"])
            + f"; prices {PRICE_FILE} Standard")


if __name__ == "__main__":
    print(value_text(compute()))
