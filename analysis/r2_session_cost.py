"""R2 real sessions: wall-clock time, tokens and cloud cost per session (operator plan v2, 2026-10-13, Part A1).

Inputs (committed result files only):
  - results/x2_r2_real_v1.jsonl and results/x2_r2_real_v1b.jsonl (evo-x2, r2a_turn rows, 40 turns per session);
  - the shared price tables results/cloud_prices_*.json, the latest file per provider (src/cloud/prices.py, the same
    loader src/cloud/client.py uses). Today that is results/cloud_prices_20261008.json (OpenAI); when an Anthropic
    file (results/cloud_prices_anthropic_<date>.json) lands it is picked up with no code change and both providers
    are reported. Per provider the priced models are the file's "selection" (strong / mid tier and cheap / small
    tier), Standard prices, short or long context from the session's largest prompt.

Per session (model x arm x seed, keyed per file):
  wall_min        sum of the session's turn_wall_s / 60 (both calls of every turn, as the harness timed them);
  billed          max session_tokens_billed_cumulative. The harness's counter (x2_r2_agent._call_tokens) adds every
                  call's prompt_tokens_est (chars/4 of the whole message list the client sent) PLUS that call's
                  completion tokens, so it is input and output together;
  output          sum of completion_tokens over every call (chars/4 of the content when completion_tokens is null,
                  the same fallback the harness counter uses);
  input           billed - output = sum of prompt_tokens_est over every call (the tokens a cloud API would bill as
                  input, at the chars/4 basis; no cloud tokenizer is applied here);
  final_prompt    prompt_tokens_est of the session's last call (the final transcript the client sent);
  prefill_s, decode_s  sums of prompt_eval_duration_s / eval_duration_s over every call (inputs of the impact model).
Cost per session, per cloud model, list (Standard) prices:
  no caching      input x input price + output x output price;
  best caching    final_prompt x input price + (input - final_prompt) x cache-read price + output x output price.
                  This is a floor, not a forecast: it ignores cache-write surcharges and assumes every earlier
                  prompt token is served from cache. A model with no cache-read price gets no discount.

Token basis: chars/4 (the harness's own counter). analysis/cloud_budget_forecast.py converts R2 prompts to
o200k_base; this module does not, so its token counts and costs are on the row estimate basis.

Usage: py -3.12 analysis/r2_session_cost.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from src.cloud import prices as cp  # noqa: E402

V1_FILE = "results/x2_r2_real_v1.jsonl"
V1B_FILE = "results/x2_r2_real_v1b.jsonl"
PRICES_GLOB = "results/" + cp.PRICE_GLOB   # register data_files entry
TIER_ORDER = ("ollama_default", "num_ctx_32768", "num_ctx_16384", "num_ctx_8192", "num_ctx_4096")


def tier_of(arm: str) -> str:
    return arm.replace("_call2_notools", "").replace("ollama_ctx_", "num_ctx_")


def read_rows(path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def load_prices(repo=REPO) -> dict:
    """{provider: {"path", "source_url", "fetched_utc", "models": (strong, cheap), "rates": {model: rates}}} from
    the latest price file per provider."""
    out = {}
    for provider, t in cp.load_tables(repo).items():
        table = t["table"]
        tiers = cp.tier_models(table)
        models = tuple(m for m in (tiers.get("strong"), tiers.get("cheap")) if m)
        rates = cp.model_rates(table)
        out[provider] = {"path": t["path"], "source_url": table.get("source_url"),
                         "fetched_utc": table.get("fetched_utc"), "models": models,
                         "rates": {m: rates[m] for m in models}}
    if not out:
        raise FileNotFoundError(f"no price file matching {PRICES_GLOB} under {repo}")
    return out


def priced_models(prices: dict) -> list[tuple[str, str]]:
    """[(provider, model)] in report order."""
    return [(p, m) for p, d in prices.items() for m in d["models"]]


def _est_tokens(text) -> int:
    """chars/4, the harness's est_tokens (r2._approx_token_count): max(1, len // 4) for non-empty text."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def _completion(call) -> int:
    c = call.get("completion_tokens")
    return c if c is not None else _est_tokens(call.get("content"))


def session_metrics(turns: list[dict]) -> dict:
    """Per-session metrics from one session's r2a_turn rows (any order). Shared with analysis/impact_model.py."""
    ts = sorted(turns, key=lambda r: r["turn_idx"])
    calls = [c for t in ts for c in (t.get("calls") or [])]
    output = sum(_completion(c) for c in calls)
    billed = max((t.get("session_tokens_billed_cumulative") or 0 for t in ts), default=0)
    wall_s = sum(t.get("turn_wall_s") or 0.0 for t in ts)
    return {"n_turns": len(ts), "wall_s": wall_s, "wall_min": wall_s / 60.0,
            "billed": billed, "output": output, "input": billed - output,
            "input_sum_check": sum(c.get("prompt_tokens_est") or 0 for c in calls),
            "final_prompt": (calls[-1].get("prompt_tokens_est") or 0) if calls else 0,
            "max_prompt": max((c.get("prompt_tokens_est") or 0 for c in calls), default=0),
            "prefill_s": sum(c.get("prompt_eval_duration_s") or 0.0 for c in calls),
            "decode_s": sum(c.get("eval_duration_s") or 0.0 for c in calls)}


def sessions(rows: list[dict]) -> dict:
    """{(model, tier, seed): metrics} for every session with r2a_turn rows in one file."""
    by = {}
    for r in rows:
        if r.get("record") == "r2a_turn":
            by.setdefault((r["model_id"], tier_of(r["arm_id"]), r["seed"]), []).append(r)
    return {key: session_metrics(ts) for key, ts in by.items()}


def session_cost(rates: dict, s: dict) -> dict:
    """USD per session for one model's rates (src/cloud/prices.model_rates entry): no caching and best-case
    caching (see module docstring)."""
    r = cp.rate_for(rates, s["max_prompt"])
    cached = r["cached_input"] if r["cached_input"] is not None else r["input"]
    out_usd = s["output"] * r["output"]
    no_cache = (s["input"] * r["input"] + out_usd) / 1e6
    best = (s["final_prompt"] * r["input"] + (s["input"] - s["final_prompt"]) * cached + out_usd) / 1e6
    return {"no_cache": no_cache, "best_cache": best}


def cells(file_sessions: dict, prices: dict) -> dict:
    """{(model, tier): {"seeds": {seed: metrics + costs}, "mean": {...}}}, sessions pooled over the given dict.
    Costs are keyed by cloud model id."""
    pm = priced_models(prices)
    out = {}
    for (model, tier, seed), s in sorted(file_sessions.items(), key=lambda kv: (kv[0][0], _tier_rank(kv[0][1]),
                                                                                 kv[0][2])):
        s = dict(s)
        s["cost"] = {m: session_cost(prices[p]["rates"][m], s) for p, m in pm}
        out.setdefault((model, tier), {"seeds": {}})["seeds"][seed] = s
    for c in out.values():
        ss = list(c["seeds"].values())
        c["mean"] = {k: statistics.mean(s[k] for s in ss) for k in ("wall_min", "billed", "output", "input")}
        c["mean"]["cost"] = {m: {k: statistics.mean(s["cost"][m][k] for s in ss) for k in ("no_cache", "best_cache")}
                             for _p, m in pm}
        c["n"] = len(ss)
    return out


def _tier_rank(t):
    return TIER_ORDER.index(t) if t in TIER_ORDER else len(TIER_ORDER)


def build(repo=REPO, files=(V1_FILE, V1B_FILE)) -> dict:
    """Per file and combined. Combined pools every session of every file; a (model, tier, seed) key present in more
    than one file keeps the first file's session and is listed under duplicates (v1b skips v1's completed keys, so
    none is expected)."""
    repo = Path(repo)
    prices = load_prices(repo)
    per_file, combined, dups = {}, {}, []
    for f in files:
        ss = sessions(read_rows(repo / f))
        per_file[f] = cells(ss, prices)
        for k, v in ss.items():
            if k in combined:
                dups.append((f,) + k)
            else:
                combined[k] = v
    return {"per_file": per_file, "combined": cells(combined, prices), "duplicates": dups,
            "prices": {p: {k: d[k] for k in ("path", "source_url", "fetched_utc", "models")}
                       for p, d in prices.items()}}


# ── operator expectations (plan v2, 2026-10-13), compared, never adjusted ───────────────────────────────

EXPECTED = (
    # (label, model, tier, metric, expected value, decimals the expectation is stated at)
    ("llama3.1:8b default wall-clock min", "llama3.1:8b", "ollama_default", "wall_min", 43, 0),
    ("llama3.1:8b 4096 wall-clock min", "llama3.1:8b", "num_ctx_4096", "wall_min", 7, 0),
    ("qwen3:14b default wall-clock min", "qwen3:14b", "ollama_default", "wall_min", 211, 0),
    ("qwen3:14b 4096 wall-clock min", "qwen3:14b", "num_ctx_4096", "wall_min", 13, 0),
    ("gpt-6.1-sol best-caching USD per llama3.1:8b default session", "llama3.1:8b", "ollama_default",
     ("gpt-6.1-sol", "best_cache"), 0.41, 2),
    ("gpt-6.1-sol no-caching USD per llama3.1:8b default session", "llama3.1:8b", "ollama_default",
     ("gpt-6.1-sol", "no_cache"), 5.63, 2),
)


def _metric(s, metric):
    if isinstance(metric, tuple):
        return s["cost"][metric[0]][metric[1]]
    return s[metric]


def expectation_check(rep: dict) -> list[dict]:
    """Each expectation against the v1 mean, the combined (v1 + v1b) mean and the per-seed range (combined). A
    computed value matches when it rounds to the expected value at the expectation's own precision. An expectation
    on a cloud model no current price file prices is reported as n/a."""
    out = []
    for label, model, tier, metric, exp, dec in EXPECTED:
        res = {"label": label, "expected": exp}
        for name, cs in (("v1", rep["per_file"].get(V1_FILE, {})), ("combined", rep["combined"])):
            c = cs.get((model, tier))
            if c is None or (isinstance(metric, tuple) and metric[0] not in c["mean"]["cost"]):
                res[name] = None
                continue
            mean = (c["mean"]["cost"][metric[0]][metric[1]] if isinstance(metric, tuple) else c["mean"][metric])
            vals = [_metric(s, metric) for s in c["seeds"].values()]
            res[name] = {"mean": mean, "min": min(vals), "max": max(vals), "n": len(vals),
                         "match": round(mean, dec) == round(exp, dec)}
        out.append(res)
    return out


# ── register strings ───────────────────────────────────────────────────────────────────────────────────

def _sections(rep):
    return list(rep["per_file"].items()) + [("combined v1+v1b", rep["combined"])]


def _name(f):
    return Path(f).stem if f.endswith(".jsonl") else f


def wallclock_str(rep: dict) -> str:
    parts = []
    for f, cs in _sections(rep):
        cell = []
        for (model, tier), c in cs.items():
            seeds = ", ".join(f"{s['wall_min']:.1f}" for s in c["seeds"].values())
            m = c["mean"]
            cell.append(f"{model} {tier}: mean {m['wall_min']:.1f} min ({seeds}; n {c['n']}), billed in+out "
                        f"{m['billed']:.0f}, input {m['input']:.0f}, output {m['output']:.0f}")
        parts.append(f"[{_name(f)}] " + "; ".join(cell))
    return " / ".join(parts)


def cost_str(rep: dict) -> str:
    pm = [(p, m) for p, d in rep["prices"].items() for m in d["models"]]
    parts = []
    for f, cs in _sections(rep):
        cell = []
        for (model, tier), c in cs.items():
            costs = []
            for p, m in pm:
                mc = c["mean"]["cost"][m]
                lo = min(s["cost"][m]["best_cache"] for s in c["seeds"].values())
                hi = max(s["cost"][m]["no_cache"] for s in c["seeds"].values())
                costs.append(f"{m} USD {mc['best_cache']:.2f}-{mc['no_cache']:.2f} (seed range {lo:.2f}-{hi:.2f})")
            cell.append(f"{model} {tier} (n {c['n']}): " + ", ".join(costs))
        parts.append(f"[{_name(f)}] " + "; ".join(cell))
    src = "; ".join(f"{p}: {d['path']} ({', '.join(d['models'])}; fetched {d['fetched_utc']})"
                    for p, d in rep["prices"].items())
    return (" / ".join(parts) + " / basis: chars/4 row estimate, Standard list prices, latest price file per "
            f"provider ({src}); range = best-case caching (final prompt at input price, rest at cache-read price) "
            "to no caching; output at output price")


def expectation_str(rep: dict, kind: str | None = None) -> str:
    """kind 'wallclock' or 'cost' keeps only that kind of expectation; None keeps all."""
    out = []
    for e in expectation_check(rep):
        is_cost = "USD" in e["label"]
        if (kind == "wallclock" and is_cost) or (kind == "cost" and not is_cost):
            continue
        bits = []
        for name in ("v1", "combined"):
            r = e[name]
            if r is None:
                bits.append(f"{name} n/a")
            else:
                bits.append(f"{name} {r['mean']:.2f} [{r['min']:.2f}-{r['max']:.2f}, n {r['n']}] "
                            f"{'match' if r['match'] else 'MISMATCH'}")
        out.append(f"{e['label']}: expected {e['expected']}, " + ", ".join(bits))
    return "; ".join(out)


def main(argv=None) -> int:
    rep = build()
    print(wallclock_str(rep).replace(" / ", "\n\n"))
    print()
    print(cost_str(rep).replace(" / ", "\n\n"))
    print()
    print(expectation_str(rep).replace("; ", "\n"))
    if rep["duplicates"]:
        print("duplicate session keys (first file kept):", rep["duplicates"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
