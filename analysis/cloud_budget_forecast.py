"""Cloud budget forecast (offline): what the planned cloud work costs against the USD 50 code cap.

No API is called. Every input is a committed file:
  - prices: results/cloud_prices_20261008.json (transcribed from OpenAI's official pricing page and prompt-caching
    guide; source URLs and fetch time are inside the file);
  - workload pack: results/workload_pack/items/*.jsonl (400 items) and item_weights_trace_weighted.json;
  - measured outputs: results/x2_outcome_table_v3.jsonl (rows with output_text);
  - R2 agent sessions: results/x2_r2_real_v1.jsonl (evo-x2 real run, 18 sessions x 40 turns x 2 calls);
  - demo volumes: docs/DEMO_SPEC.md (failure scenarios counted from section 4).

Token counting: tiktoken o200k_base everywhere text is available. The R2 rows do not store the messages sent, only
the harness's chars/4 estimate per call (prompt_tokens_est). Those are converted to o200k_base with a per-seed ratio
measured by regenerating that seed's deterministic session text (system prompt, tool schema, the 40 user messages,
via harness/x2_r2_agent.py's own builders) and tokenizing it both ways. Model replies inside the transcript are not
in that ratio. Per-call output tokens are re-tokenized from the recorded reply text (or tool-call JSON).

Line items:
  (a) cheap model, Standard, one pass over the 400 pack items, no caching;
  (b) mid-tier model, Batch, one pass over the 400 pack items, no caching;
  (c) 12 R2-shaped agent sessions (40 turns, 2 calls per turn) on the mid-tier model, Standard, implicit prompt
      caching: each call's cached prefix is the previous call's prompt (cached tokens = min(prev, cur), only if
      prev >= 1024, the page's minimum cacheable prefix); every other prompt token is a cache write (1.25x input)
      when the prompt is >= 1024 tokens, else plain input. Cost = mean over the 18 measured sessions x 12;
  (d) demo: (100 live validation + 5 rehearsals x 50) items, half to the cloud, items drawn trace-weighted from
      the pack, mid-tier Standard; plus the DEMO_SPEC failure scenario(s), one cloud call each, run live + per
      rehearsal;
  (e) 30% rerun/debug allowance on (a)-(d).
Pessimistic: no caching, no Batch, mid-tier wherever a model choice exists ((a) stays the cheap model by
definition; (c) and (d) are already mid-tier).

Usage: py -3.12 analysis/cloud_budget_forecast.py   (prints the table and writes nothing)
"""
from __future__ import annotations

import contextlib
import io
import json
import re
import statistics
import sys
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PRICES_FILE = "results/cloud_prices_20261008.json"
PACK_DIR = "results/workload_pack/items"
WEIGHTS_FILE = "results/workload_pack/item_weights_trace_weighted.json"
OUTCOME_FILE = "results/x2_outcome_table_v3.jsonl"
R2_FILE = "results/x2_r2_real_v1.jsonl"
DEMO_SPEC = "docs/DEMO_SPEC.md"

CHEAP, MID = "gpt-6-luna", "gpt-6.1-sol"
sys.path.insert(0, str(REPO))
from src.cloud.client import HARD_SPEND_CAP_USD  # noqa: E402  (one source for the cap)

CAP_USD = HARD_SPEND_CAP_USD
N_AGENT_SESSIONS = 12
DEMO_LIVE_ITEMS, DEMO_REHEARSALS, DEMO_REHEARSAL_ITEMS = 100, 5, 50
CLOUD_SHARE = 0.5
RERUN_ALLOWANCE = 0.30
MIN_CACHEABLE = 1024
LONG_CONTEXT_ABOVE = 272_000
# DEMO_SPEC section 4 states the failure scenario's runbook as "~14,000-token"; used as its cloud call's input.
SCENARIO_INPUT_TOKENS = 14_000


@lru_cache(maxsize=1)
def _enc():
    import tiktoken
    return tiktoken.get_encoding("o200k_base")


def ntok(text) -> int:
    if not text:
        return 0
    return len(_enc().encode(text if isinstance(text, str) else json.dumps(text), disallowed_special=()))


def _jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_prices(repo=REPO):
    return json.loads((repo / PRICES_FILE).read_text(encoding="utf-8"))


def _rate(prices, model, tier, prompt_tokens):
    ctx = "long" if prompt_tokens > LONG_CONTEXT_ABOVE else "short"
    return prices["models"][model][tier][ctx]


def call_cost(prices, model, tier, prompt, output, cached=0, write=0):
    """USD for one call: prompt = total input tokens, of which `cached` at the cached rate and `write` at the
    cache-write rate; the rest at the uncached input rate."""
    r = _rate(prices, model, tier, prompt)
    plain = prompt - cached - write
    return (plain * r["input"] + cached * r["cached_input"] + write * r["cache_write"] + output * r["output"]) / 1e6


# ── workload pack ────────────────────────────────────────────────────────────────────────────────────

def pack_items(repo=REPO):
    items = []
    for f in sorted((repo / PACK_DIR).glob("*.jsonl")):
        for r in _jsonl(f):
            items.append({"item_id": r["item_id"], "family": r["family"], "n_turns": int(r.get("n_turns") or 1),
                          "input_tokens": ntok(r["prompt"]),
                          "oracle_tokens": ntok(r.get("reference_solution") or str(r.get("oracle_answer", "")))})
    return items


def r2_call_output_tokens(call) -> int:
    if call.get("content"):
        return ntok(call["content"])
    if call.get("native_tool_calls"):
        return ntok(json.dumps(call["native_tool_calls"]))
    return 0


def output_lengths(repo=REPO):
    """Expected output tokens per pack family. Basis: median o200k_base length of real measured replies
    (x2_outcome_table_v3 rows with output_text and finish_reason 'stop', all models). trace_length_mix has its
    own measured rows (same needle-QA task as longdoc_qa). r2_sessions has no outcome-table rows; its basis is the
    median per-call reply length in the R2 real run, times the item's n_turns (one reply per turn)."""
    by_fam = {}
    for r in _jsonl(repo / OUTCOME_FILE):
        if r.get("output_text") is not None and r.get("finish_reason") == "stop" and r.get("family"):
            by_fam.setdefault(r["family"], []).append(ntok(r["output_text"]))
    out = {f: {"median": statistics.median(v), "n": len(v), "basis": "x2_outcome_table_v3 measured replies"}
           for f, v in by_fam.items()}
    r2_calls = [r2_call_output_tokens(c) for r in _jsonl(repo / R2_FILE) if r.get("record") == "r2a_turn"
                for c in r["calls"]]
    out["r2_sessions_per_turn"] = {"median": statistics.median(r2_calls), "n": len(r2_calls),
                                   "basis": "x2_r2_real_v1 per-call replies"}
    return out


def item_output(item, outs):
    if item["family"] == "r2_sessions":
        return outs["r2_sessions_per_turn"]["median"] * item["n_turns"]
    return outs[item["family"]]["median"]


def pack_summary(repo=REPO):
    items = pack_items(repo)
    outs = output_lengths(repo)
    fams = {}
    for it in items:
        it["output_tokens"] = item_output(it, outs)
        f = fams.setdefault(it["family"], {"n": 0, "input_tokens": 0, "output_tokens": 0, "oracle_tokens": 0})
        f["n"] += 1
        f["input_tokens"] += it["input_tokens"]
        f["output_tokens"] += it["output_tokens"]
        f["oracle_tokens"] += it["oracle_tokens"]
    return items, fams, outs


# ── R2 sessions ──────────────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=None)
def o200k_per_est_ratio(seed: int, turns: int = 40) -> float:
    """o200k_base tokens / harness chars/4 estimate, on the seed's regenerated deterministic session text."""
    sys.path.insert(0, str(REPO / "harness"))
    import x2_r2_agent as a
    with contextlib.redirect_stdout(io.StringIO()):
        s = a.build_agent_session(seed, turns)
        texts = [s.system_prompt, json.dumps(a.r2.ollama_tools_payload())] + [a.user_message(t, s) for t in s.spec.turns]
    return sum(ntok(t) for t in texts) / sum(a.est_tokens(t) for t in texts)


def r2_sessions(repo=REPO, token_basis="o200k"):
    """Per session: ordered calls [(prompt_o200k, output_o200k)], plus per-turn sent tokens. token_basis="row_est"
    uses the rows' own chars/4 prompt estimates unconverted (sensitivity only)."""
    sessions = {}
    for r in _jsonl(repo / R2_FILE):
        if r.get("record") != "r2a_turn":
            continue
        sessions.setdefault((r["model_id"], r["arm_id"], r["seed"]), []).append(r)
    out = []
    for key, rows in sorted(sessions.items()):
        rows.sort(key=lambda r: r["turn_idx"])
        ratio = o200k_per_est_ratio(key[2]) if token_basis == "o200k" else 1.0
        calls, per_turn, est_total = [], [], 0
        for r in rows:
            sent = 0
            for c in r["calls"]:
                p = round(c["prompt_tokens_est"] * ratio)
                calls.append((p, r2_call_output_tokens(c)))
                sent += p
            per_turn.append(sent)
            est_total = r["session_tokens_billed_cumulative"]
        out.append({"key": key, "ratio": ratio, "calls": calls, "per_turn_sent": per_turn,
                    "prompt_total": sum(p for p, _ in calls), "output_total": sum(o for _, o in calls),
                    "row_est_total_incl_output": est_total})
    return out


def session_cost(prices, calls, model, tier, caching):
    total, prev = 0.0, 0
    for p, o in calls:
        if caching:
            cached = min(prev, p) if prev >= MIN_CACHEABLE else 0
            write = (p - cached) if p >= MIN_CACHEABLE else 0
            total += call_cost(prices, model, tier, p, o, cached=cached, write=write)
        else:
            total += call_cost(prices, model, tier, p, o)
        prev = p
    return total


# ── demo volumes ─────────────────────────────────────────────────────────────────────────────────────

def count_failure_scenarios(repo=REPO) -> int:
    """Scripted failure scenarios in DEMO_SPEC.md: the numbered '## N. Live-run script' sections."""
    text = (repo / DEMO_SPEC).read_text(encoding="utf-8")
    return len(re.findall(r"^## \d+\.\s+Live-run script", text, flags=re.M))


# ── forecast ─────────────────────────────────────────────────────────────────────────────────────────

def forecast(repo=REPO, output_multiplier=1.0, demo_weighting="trace", r2_token_basis="o200k"):
    prices = load_prices(repo)
    items, fams, outs = pack_summary(repo)
    weights = json.loads((repo / WEIGHTS_FILE).read_text(encoding="utf-8"))
    om = output_multiplier

    def pack_pass(model, tier):
        return sum(call_cost(prices, model, tier, it["input_tokens"], it["output_tokens"] * om) for it in items)

    def item_mean(model, tier):
        if demo_weighting == "flat":
            w = {it["item_id"]: 1 / len(items) for it in items}
        else:
            w = weights
        return sum(w[it["item_id"]] * call_cost(prices, model, tier, it["input_tokens"], it["output_tokens"] * om)
                   for it in items)

    sess = r2_sessions(repo, token_basis=r2_token_basis)
    scaled = lambda s: [(p, o * om) for p, o in s["calls"]]  # noqa: E731
    c_cached = statistics.mean(session_cost(prices, scaled(s), MID, "standard", True) for s in sess) * N_AGENT_SESSIONS
    c_plain = statistics.mean(session_cost(prices, scaled(s), MID, "standard", False) for s in sess) * N_AGENT_SESSIONS

    n_scen = count_failure_scenarios(repo)
    demo_items = (DEMO_LIVE_ITEMS + DEMO_REHEARSALS * DEMO_REHEARSAL_ITEMS) * CLOUD_SHARE
    scen_calls = n_scen * (1 + DEMO_REHEARSALS)
    scen_out = outs["longdoc_qa"]["median"] * om
    d = demo_items * item_mean(MID, "standard") + scen_calls * call_cost(prices, MID, "standard",
                                                                        SCENARIO_INPUT_TOKENS, scen_out)

    base = {"a": pack_pass(CHEAP, "standard"), "b": pack_pass(MID, "batch"), "c": c_cached, "d": d}
    pess = {"a": pack_pass(CHEAP, "standard"), "b": pack_pass(MID, "standard"), "c": c_plain, "d": d}
    for t in (base, pess):
        t["e"] = RERUN_ALLOWANCE * (t["a"] + t["b"] + t["c"] + t["d"])
        t["total"] = sum(t[k] for k in "abcde")

    turns = len(sess[0]["per_turn_sent"])
    mean_turn = [statistics.mean(s["per_turn_sent"][i] for s in sess) for i in range(turns)]
    cum, acc = [], 0
    for v in mean_turn:
        acc += v
        cum.append(acc)
    return {
        "base": base, "pessimistic": pess, "cap_usd": CAP_USD,
        "models": {"cheap": CHEAP, "mid": MID}, "price_source": prices["source_url"],
        "fetched_utc": prices["fetched_utc"],
        "pack": {"n_items": len(items), "families": fams,
                 "input_tokens_total": sum(it["input_tokens"] for it in items),
                 "output_tokens_total": sum(it["output_tokens"] for it in items)},
        "outputs": outs,
        "r2": {"n_sessions": len(sess), "turns": turns,
               "ratio_o200k_per_est": sorted({round(s["ratio"], 4) for s in sess}),
               "session_prompt_tokens": [s["prompt_total"] for s in sess],
               "session_output_tokens": [s["output_total"] for s in sess],
               "session_row_est_totals": [s["row_est_total_incl_output"] for s in sess],
               "mean_per_turn_sent": mean_turn, "mean_cumulative_sent": cum,
               "per_session": [{"model": s["key"][0], "arm": s["key"][1], "seed": s["key"][2],
                                "prompt": s["prompt_total"], "output": s["output_total"],
                                "row_est": s["row_est_total_incl_output"]} for s in sess]},
        "demo": {"failure_scenarios": n_scen, "scenario_cloud_calls": scen_calls,
                 "cloud_items": demo_items, "weighting": demo_weighting},
    }


def _fmt(v):
    return f"{v:.2f}"


def table_lines(f):
    b, p = f["base"], f["pessimistic"]
    rows = [("(a) cheap model, full pack pass (400 items), Standard", "a"),
            ("(b) mid-tier, full pack pass (400 items), Batch", "b"),
            ("(c) 12 R2-shaped agent sessions x 40 turns, mid-tier, cached", "c"),
            ("(d) demo: 175 cloud items + failure-scenario calls, mid-tier", "d"),
            ("(e) 30% rerun/debug allowance on (a)-(d)", "e"),
            ("Total", "total")]
    out = ["| Line | Forecast USD | Pessimistic USD |", "|---|---:|---:|"]
    out += [f"| {name} | {_fmt(b[k])} | {_fmt(p[k])} |" for name, k in rows]
    return out


def main():
    f = forecast()
    print("\n".join(table_lines(f)))
    print(f"\nmodels: cheap={f['models']['cheap']} mid={f['models']['mid']}; prices {f['price_source']} "
          f"fetched {f['fetched_utc']}; cap USD {f['cap_usd']:.0f}")
    print(f"pack: {f['pack']['n_items']} items, input {f['pack']['input_tokens_total']} o200k tokens, "
          f"expected output {f['pack']['output_tokens_total']:.0f}")
    for fam, v in f["pack"]["families"].items():
        print(f"  {fam}: n={v['n']} input={v['input_tokens']} output={v['output_tokens']:.0f} "
              f"oracle={v['oracle_tokens']}")
    for k, v in f["outputs"].items():
        print(f"  output basis {k}: median {v['median']} (n={v['n']}, {v['basis']})")
    r2 = f["r2"]
    print(f"R2: {r2['n_sessions']} sessions, ratio o200k/est {r2['ratio_o200k_per_est']}")
    for s in r2["per_session"]:
        print(f"  {s['model']} {s['arm']} {s['seed']}: prompt {s['prompt']} output {s['output']} (row est {s['row_est']})")
    print("  mean cumulative sent by turn: " + ", ".join(f"t{i+1}={v:.0f}" for i, v in enumerate(r2["mean_cumulative_sent"])))
    print(f"demo: {f['demo']}")
    for label, kw in (("flat-weighted demo items", {"demo_weighting": "flat"}),
                      ("output tokens x5 (hidden reasoning)", {"output_multiplier": 5.0}),
                      ("R2 prompts at the rows' chars/4 estimate", {"r2_token_basis": "row_est"})):
        g = forecast(**kw)
        print(f"sensitivity {label}: total {_fmt(g['base']['total'])}, pessimistic {_fmt(g['pessimistic']['total'])}")


if __name__ == "__main__":
    main()
