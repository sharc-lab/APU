# Demo Spec — Hybrid Execution Layer / DSE Tool

Freeze date: Oct 28 2026. This spec covers the investor/professor demo built
on top of Paper 1's measured hardware envelope (see docs/PLAN_PAPER1_DEMO.md
section 1 for the claim-to-feature map). The differentiator stated in the
program brief: the router treats effective context, silent failure modes,
vendor-dependent co-runner penalties, and memory budget walls as *measured,
live state* rather than assuming hardware just works, which is what
RouteLLM and RouterBench do not do.

---

## 1. Inputs

| Input | Type | Example | Backing data |
|---|---|---|---|
| Workload description | Free-text + category tag (RAG / agentic search / multi-turn chat / tool-chain) | "Customer support agent that retrieves order records from a ~15k-token knowledge snippet and answers a follow-up question" | Maps to the closest probe category in `evaluation/probes/*.jsonl` (art_*, rag_*, sea_*, cha_*, lon_*) so the tool can look up a measured outcome rather than guess |
| Cloud budget | USD total | $50 | Enforced by `src/cloud/client.py`'s hard spend cap (see B2): USD 50 total over the whole ledger, not per month; a real-key client cannot raise it; alerts at 50%, 75% and 90% (2026-10-07) |
| Quality floor | 0.0-1.0 (exact-match score threshold) | 0.90 | Compared against per-probe-category score curves from `results/outcome_table_*.json` (docs/PLAN_PAPER1_DEMO.md Oct 14 deliverable) |
| Latency target | ms, p50 or p99 | p50 <= 3000ms | Compared against `http_client_ns` / TTFT fields recorded per-call (Claim B-04) |
| Hardware options | Multi-select from `configs/hardware/*.yaml` | 16gb_no_npu, 32gb_npu, evox2_strix_halo_128gb, evo_t2s | Each option's feasibility is read from its own measured envelope, not a shared assumption |

---

## 2. Outputs

1. **Recommended hardware** — one of `configs/hardware/*.yaml`, chosen as the
   cheapest option whose measured envelope satisfies the quality floor and
   latency target for the stated workload category at the stated budget
   (the J-01 function: f(workload type, quality floor) -> minimum provisioned
   GB, then mapped to the cheapest config meeting that GB figure).
2. **Recommended local model** — a specific (model, quantization, context
   limit) tuple, e.g. "Qwen3-4B-Instruct, q4_0 KV, ctx <= 32768" — picked so
   that the required-fraction headroom (Claim A-04/A-05) covers the
   workload's typical artifact size without crossing the measured extinction
   threshold (Claim A-02/A-03) for that category.
3. **Routing policy** — one of `routing/policies/*` (static_category,
   cascade, budget_aware_cascade, speculative, or learned_router), with the
   specific thresholds it will use (budget_ratio cutoff, escalation
   trigger) stated in plain language, not just a policy name.
4. **Pareto frontier** — a plot of quality (y) vs. cloud spend (x), with
   hardware BOM cost as point size/color (Fig 4.1 / 4.2 in
   docs/PAPER_OUTLINE_DSE.md), dominated configurations grayed out, the
   recommended point highlighted.

---

## 3. Screens

### Screen 1 — Input form

Four fields plus hardware multi-select, as in section 1. A "use example
workload" button loads one of the workload-pack examples (Oct 7 deliverable,
docs/PLAN_PAPER1_DEMO.md) so the presenter never types live.

### Screen 2 — Frontier view

The Pareto frontier plot (output #4), with the recommended (hardware, model,
policy) triple marked as a star. Hovering a point shows which claim IDs
(A-NN / J-01) back that point's position, so a technical reviewer can click
through to the measured data instead of trusting a marketing number.

### Screen 3 — Live-run view

Runs the workload (or a cached replay of it, see `harness/replay/cache.py`)
under two conditions side by side: a naive hybrid baseline and this tool's
recommended policy. Shows live token stream, a running cost meter, a context
budget_ratio gauge, and a failure-mode label (`correct` / `format_failure` /
`parametric_default` / `free_fabrication` / `silent_truncation`) pulled from
the same taxonomy used in docs/FINDINGS.md (Stage 1.4).

---

## 4. Live-run script — "naive hybrid silently fails vs. ours succeeds"

**Example workload** (from the workload pack): a support-ticket triage agent.
The user pastes a ~14,000-token internal runbook (contains a specific
escalation SLA buried about two-thirds of the way through the document) and
asks: *"Per our runbook, what's the SLA for a P1 ticket raised by an
enterprise customer after hours?"* The correct answer is a specific number
("4 business hours") that appears once, in a table near the end of the
runbook.

Hardware on stage: a 16 GB laptop with no discrete GPU (`16gb_no_npu.yaml`),
running a 4B local model with an 8192-token context window at default
settings — a plausible SME setup, not a strawman.

**Side A — naive hybrid (what most tools, including RouteLLM-style routers, do today):**

1. The router sees the prompt is "mostly local-sized" (14k tokens is within
   what the local model's 8192 ctx *could* approximately cover after lossy
   compression) and picks the local model to save cost, because its policy
   only looks at prompt length vs. advertised context window — not measured
   behavior.
2. The runbook plus question exceeds 8192 tokens, so the serving runtime
   silently left-truncates from the start of the prompt (the same mechanism
   measured in Claim A-02/A-03). The SLA table, sitting two-thirds of the way
   through the original document, survives truncation in this case — but the
   *escalation clause explaining the after-hours enterprise exception*, which
   sits earlier in the document, does not.
3. The local model, now missing the exception clause but still seeing a
   table with SLA numbers, does not say "I don't have enough information."
   Per Claim A-08/A-09, abstention is rare and an explicit "say so if unsure"
   instruction in the system prompt actually *drives fabrication rate to
   100%* in this regime — so the naive side's prompt (which does include such
   an instruction, because that is standard practice) backfires exactly as
   measured. The model answers confidently: **"2 business hours"** — the
   *standard* SLA, not the after-hours enterprise exception, because the
   clause distinguishing them was silently dropped.
4. On screen: a confident, fast, cheap, wrong answer. No error. No warning.
   The cost meter shows $0.00. The latency gauge is green. Nothing in the
   naive tool's own telemetry flags this as a problem.

**Side B — this tool's router:**

1. Before dispatch, the router checks the workload category (long-document
   QA / RAG-style) against that hardware+model's measured extinction
   threshold for this category (Claim A-02/A-03: per-probe thresholds in the
   0.444-0.761 artifact-fraction range) and the required-fraction data
   (Claim A-04/A-05: only ~21% of an artifact's tokens are typically
   load-bearing, but *which* 21% matters and position-dependent loss is not
   something a length check alone can see).
2. The router computes that a 14,000-token input against an 8192-token local
   window means the budget_ratio for this workload is below 1.0, and — per
   Claim A-13 — this is exactly the regime where artifact position
   determines whether the load-bearing span survives. Because it cannot
   verify the exception clause's position without either a bigger context or
   a span-aware retrieval step, it does **not** silently truncate and guess.
3. The router escalates: either (a) routes to the cloud model (within the
   stated $50 total cap, logging the call to `results/cloud_ledger.jsonl`),
   or (b) if the budget is exhausted, returns a labeled refusal —
   **"silent_truncation risk: escalation clause may be outside the surviving
   context window at this budget_ratio; answer withheld rather than guessed"**
   — instead of fabricating.
4. On screen: either a correct answer ("4 business hours," from the cloud
   escalation, cost shown transparently against the budget meter) or an
   honest, labeled refusal — never a confident wrong answer.

**The punchline on screen:** both sides show their failure-mode label.
Side A's label, after the fact, is revealed as `silent_truncation ->
free_fabrication`. Side B's label is `correct` or `refused (budget_ratio
below extinction threshold)`. The investor/professor sees the same
underlying measured mechanism (Claim A-02/A-03/A-08/A-09/A-13) produce two
different outcomes depending on whether the router has that mechanism as
live state.

---

## 5. Success metrics for the meeting

What would make the professor/investor say yes:

1. **The side-by-side is not staged with a toy example** — the workload pack
   example above is representative of a real SME support/ops workload, and
   the naive failure is a real, measured mechanism (cited claim IDs), not a
   contrived prompt-injection trick.
2. **Every number on the frontier screen traces to a file** — clicking any
   Pareto point or routing threshold shows the `results/*.json` file and
   claim ID behind it. No number on screen should be unsourced.
3. **The recommender changes its answer when inputs change**, live, in front
   of the audience — raising the quality floor should visibly move the
   recommended hardware/model/policy, demonstrating the tool is a live
   function over measured data, not a canned demo path.
4. **The failure-mode taxonomy reads as domain expertise**, not a generic
   "error" — `format_failure`, `parametric_default`, `free_fabrication`,
   `silent_truncation` are distinct, named, and each has a measured example
   the presenter can point to.
5. **A direct question about RouteLLM/RouterBench gets a crisp answer**: those
   systems route by model capability/benchmark score; this tool routes by
   measured hardware-conditioned behavior (memory budget walls, co-runner
   penalties, silent truncation thresholds) that does not show up in a
   static benchmark table at all. The demo should make this distinction
   land without the presenter having to explain it verbally — the frontier
   screen's hover-to-claim feature is designed to make this self-evident.
6. **No live cloud spend surprises** — the demo must run entirely on the
   stub provider or a pre-capped, pre-logged real-key run (see
   `src/cloud/client.py` spend cap), so a "what's your unit economics"
   question can be answered from `results/cloud_ledger.jsonl` rather than a
   verbal estimate.
