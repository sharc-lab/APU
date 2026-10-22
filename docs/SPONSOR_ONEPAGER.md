# Local AI agents fail quietly: sponsor one-pager

Measured cost today: USD 3.60/day for the reference team; the larger costs depend on the wrong-task rate (pending B3
task runs) and on whether anyone waits for compute. ("measured-only USD/day 3.60" [impact-total], projection for the
assumed 10-developer reference team; the measured per-session facts behind it follow.)

Rithwik Sharma, 2026-10-14 (rewritten after the impact audit of the same day). Plan v2 Part C2. Every number below is
pasted verbatim from a row of docs/NUMBERS_REGISTER.md (row id in brackets) and tagged: **measured** (read from result
files of real runs), **projection** (measured per-session data times an assumed team size and workload), or
**scenario** (also rests on an assumption that has not been measured). Full tables: docs/IMPACT.md and
docs/BUSINESS_METRICS.md.

## What we measured, per agent session (40-turn sessions on a workstation-class AI PC)

- **The smaller window is faster, and wrong without saying so.** Median time per session, llama3.1:8b:
  "ollama_default median 43.3 min/session (n=5), num_ctx_4096 median 7.0 min/session (n=5)"
  [impact-session-time-by-window] (measured). At num_ctx 4096 sessions fail silently: "num_ctx_4096 10/15 silent,
  10 after window exceeded", against "ollama_default 3/15 silent, 0 after window exceeded" over three models
  [R2-real-v1b-gated-kill] (measured); for llama3.1:8b at the default window, "R2 silent rule/recall failures after the
  window 0/5 sessions" [impact-P1] (measured). Every call returns normally; nothing tells the user. With tools
  on every call, qwen3:14b at num_ctx 4096 gets "excess +30.7 points (session-level 95% CI +26.0 to +34.7)" more
  wrong tasks than at the default window; for the reference team that is "time USD/day 0.00 to 126.85" plus "cloud
  redo USD/day 10.33 (USD 0.0281 per wrong task, priced, not executed ...)", "profile cell USD/day 10.33 to 137.18"
  [impact-v2-P1] (mixed: measured rate and wall time, cloud redo priced from token counts but not executed, assumed
  volume and waiting). Fixing one wrong answer in the cloud costs about USD 0.03 (priced, not executed); the cost is
  that the failure is silent. Ranking "P1 > P5 > P3 > P4; unranked: P2, P6, P7" [impact-ranking-v2]: the
  configuration saving (P2) is now valued as machine capacity, "3 saved x USD 2,000 (impact-ref-machine-price,
  list price, Liliputing, 2025-04-15) / 3-year life (ASSUMED) = USD 5.48/day" with waiting time only an upper bound
  [impact-v2-P2]: half the hardware, and "the flash-attention control attributes the speedup to KV precision, not
  flash attention (b5-fa-control)" [impact-v2-P2]. (The ranking quoted above predates that control; the current one is
  in row impact-ranking-v2.)
- **At the smallest window the runtime cuts the system prompt.** "ollama_ctx_4096: 76/80 calls message-truncated; 34
  context shifts (first turn/call (4, 2), n_keep [5], n_discard [2045], 29 on the final-answer call); 1 token cuts
  (token cut limit 2050 keep 4 prompt 4097 -> 2050)" [R2-mechanism-lowlevel] (measured): on a context shift only the
  first 5 tokens are kept, so the instructions that hold the agent's rules are dropped.
- **Sending the whole session to a cloud model is cheap only with prompt caching on.** Claude Sonnet 5.5
  at Anthropic's published prices: "whole session to the cloud USD 0.3358 caching on, 3.7302 no caching (n=5)"
  [impact-P3] (measured token counts, official price page [cloud-prices-anthropic]). With caching on, re-sending the
  history costs "USD 0.1799 caching on, 3.5978 no caching" per session [impact-P3] (measured).
- **The other runtime trades one failure for another.** On a 32B model, llama-server silently truncated "0/74" rows
  but timed out 11 times; Ollama's default silently truncated "15/97" [qwen3-32b-truncation-vs-latency] (measured).

So no single setting is safe: the default window is slow, the small window loses history and rules silently, and
the cloud is only cheap with caching on. The fix is a decision per step (keep it local, trim it, or send it to the
cloud), which is what our router does (escalations E1 to E4 in docs/IMPACT.md).

## What it means in money (and how much of that is measured)

- For an assumed 10-developer team at the runtime default, the measured part of the cost is "measured-only USD/day
  3.60 (USD/month 75.55)", and the full range is "USD/day 3.60 to 192.81" [impact-total] (projection). The low end is
  cloud fees with caching on; everything above it is developer time, which is worth money only if someone is waiting
  for the machine. Silent wrong results are not in either number: no task-level run has measured them yet.
- **Wage sensitivity (scenario).** If the router's local time savings are valued at a wage, its monthly saving for an
  assumed 25-developer team is "hour_value=p10 (USD 39.64/h): vs naive 1,601 (if nobody waits -538)" and
  "hour_value=p90 (USD 103.20/h): vs naive 5,029 (if nobody waits -538)" [business-sensitivity] (scenario: assumes the
  router leaves no silent failures where a pre-registered mitigation recovered, not yet tested end to end). If nobody
  waits for local compute, the router saves nothing and costs its cloud fees, at every wage. (The first version of
  this page reported a saving of 0 at the 10th-percentile wage; that figure priced the router's cloud calls without
  caching and no longer holds.)
- Router savings against the runtime default, against all-cloud and against today's mix are **pending** the policy
  comparison (plan item B5). Hardware and power are not priced.

## Evidence status

| number | status |
|---|---|
| silent-failure counts, truncation mechanism, session times, decode rates | measured on real sessions and runs |
| cloud cost per session with and without caching | measured token counts at the official Anthropic price page |
| any USD per day or month for a team | projection (assumed team size, workload, cloud share) |
| developer time as money | assumption: valued from 0 (nobody waits) to the full median wage |
| cost of a silent wrong result | not measured (needs task-level runs and rework time) |
| router savings | pending B5; scenario figures only |

## What we need

1. **Run time for the policy comparison (B5)** on the two workstations: turns every "pending" router saving into a
   measurement.
2. **Task-level runs (B3) under at least two window settings, plus real-session rework time**: the only way to put a
   measured price on a silent wrong result.
3. **From Zach** (docs/IMPACT.md, plan item A6): access to the repository holding commits 180dfbb9, 78fc3478 and
   b172b287; the raw data files behind each; and his hybrid demo code (routing rule and cost calculator), priced on
   our shared price table and escalation definitions so the two results can be compared.
4. **Cited hardware and power prices**, so savings can be stated net of hardware.
