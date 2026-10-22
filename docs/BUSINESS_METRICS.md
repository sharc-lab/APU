# Business metrics: what a customer team pays today, and what the router would change

Plan v2 Part C1 (operator plan, 2026-10-13), revised by the operator decisions of 2026-10-14 (impact audit and
C-track review, items A to G). Built only from the impact model (plan item A3, docs/IMPACT.md,
analysis/impact_model.py) and its Router replay. Every number on this page is in the generated block below or in a
register row of docs/NUMBERS_REGISTER.md (row ids in brackets): business-mid-team, business-mid-team-qwen3-14b,
business-hour-range for the facts, and business-scenario-plus-a and business-sensitivity for the scenario section.

## What changed on 2026-10-14

- **Caching is the default (B).** Cloud fees use the shared Anthropic table [cloud-prices-anthropic], escalation
  model claude-sonnet-5-5, with provider prompt caching on as the main case, for all-cloud and for the router's
  escalated steps alike. No caching is shown only as the upper bound, in brackets.
- **Waiting time (C).** Local compute time is shown as a range: 0 (nobody waits for it) to the full hour value
  (someone waits for every second). P1 (silent wrong results) had no dollar value in this profile (v1, no B3 task runs then); impact v2 prices it for a qwen3:14b team at num_ctx 4096, "profile cell USD/day 10.33 to 137.18" [impact-v2-P1] (mixed; cloud part priced, not executed: the cloud redo per wrong task is priced from token counts, not run), ranking "P2 > P1 > P5 > P3 > P4" [impact-ranking-v2], where P2 is the measured "139.4 min saved per session" by configuration (B5, flash-attention control PENDING) at the assumed volume (docs/IMPACT.md).
- **No +A in any headline (D).** Every router saving (vs the naive default, vs all-cloud, vs today's mix) and the
  hardware recommendation print "pending B5" in the main table. The +A numbers are kept only in the labelled
  SCENARIO section at the end of the generated block.
- **Nothing built on an assumed rework cost is called a lower bound (E).** The scenario section says what it rests on.
- **The 8 GB laptop profile is dropped (F).** No R2 session completed on that laptop; every session outcome in that
  profile had been transferred from the workstation, so nothing measured supported its totals. The laptop's own
  measured facts stay in their rows ([blade-k1-layer-split], [blade-dryrun-rates]).
- **"Steps kept local" traced (G).** See the next section.

## Where "80.0% steps kept local" comes from

It is a replay, not a measurement of a team. The evo-x2 R2 sessions of each model and window (5 per cell) are replayed
through src/dse/router.py, and the share is 1 minus (cloud steps / steps). At num_ctx 4096 and 8192 the router
escalates exactly 16 of 80 steps in every replayed session, for both models (the "Router replay per window" table
below prints the per-session counts). The 16 are the R2 script's 8 canary-check turns (every 5th turn asks for a tag
that lives only in turn 1, src/dse/r2_replay.py `needs_turns`), both calls each: once turn 1 has been trimmed, each of
them escalates under E4. So 80.0% is a constant of the R2 session script at those windows, the same for every profile
that uses them; it is not a property of the team, the workload volumes or the model. The two workstation profiles
share it because they use the same kind of replay at windows below the point where turn 1 still fits. Larger windows
keep turn 1 longer and escalate fewer canary turns (the per-window table). A real workload with a different share of
steps that reach back to old history would give a different number.

## Status of the inputs

- **Plan item B5 (policy comparison) has not run.** There is no measured end-to-end run of the router against the
  other policies on real sessions, so no router saving is stated outside the scenario section. Which steps the router
  sends to the cloud, and what they cost, is computed from the replay (measured inputs); whether its answers are as
  good as the alternatives' is not measured.
- **Profiles are assumptions.** results/business_profiles_20261014.json holds two customer profiles (team size,
  sessions per developer per day, turns per session, cloud share today, workdays, co-runner share, machine, model);
  every field is an ASSUMPTION. The hour value is the cited BLS median [impact-hour-value]; its range (p10, p90) is
  derived from the same BLS page's annual percentiles [business-hour-range].
- Time saved vs all-cloud is not measured (cloud latency for these sessions was never measured). Hardware purchase
  cost and power are not priced anywhere in this repo. No market-size figure is used.

## Policies compared

| policy | cloud spend per session | local cost per session |
|---|---|---|
| all-cloud | the whole R2 session sent to the escalation model, caching on (no caching as the upper bound); the input of [impact-P3] | none; cloud latency not measured |
| naive local default | 0 | compute time of rows P2, P4, P5 at the runtime's default window, from 0 to the full hour value; P1 not measured; P7 as measured |
| today's mix | the profile's cloud share of sessions at the all-cloud price | the rest at the naive local default |
| router | the Router replay's escalated steps at a window, caching on between escalated steps (no caching as the upper bound), output at the full answer budget | pending B5 |

<!-- business_metrics:begin -->

Generated by `analysis/business_metrics.py --write` from results/business_profiles_20261014.json and the impact model's functions. USD per month unless stated. Labels: measured = per-session quantity from result files; projection = measured per-session quantity x ASSUMED profile volumes. Cloud: claude-sonnet-5-5 from the shared Anthropic table, caching on (no caching = upper bound). Local compute time: 0 (nobody waits) to the full hour value (someone waits).

Hour values: median USD 64.44 (impact-hour-value); p10 USD 39.64 and p90 USD 103.20 (business-hour-range, derived from BLS annual percentiles at 2080.1 h/year).

### Per-profile facts (register rows business-mid-team, business-mid-team-qwen3-14b)

| quantity | label | mid_team | mid_team_qwen3_14b |
|---|---|---:|---:|
| team size; sessions/dev/day; cloud share today | assumption | 25; 4; 0.5 | 25; 4; 0.5 |
| device; model | assumption | evo-x2; llama3.1:8b | evo-x2; qwen3:14b |
| sessions / month | assumption | 2,100 | 2,100 |
| naive local default window | measured | ollama_default | ollama_default |
| cloud USD / session, whole session to the cloud: caching on (no caching) | measured | 0.336 (3.730) | 0.363 (3.859) |
| all-cloud spend: caching on (no caching) | projection | 705 (7,833) | 763 (8,104) |
| naive local: compute time cost (0 to full hour value) | projection | 0 to 3,857 | 0 to 15,759 |
| naive local: compute hours / month | projection | 59.9 | 244.6 |
| naive local: silent wrong results (P1) | not measured in this profile | not measured (v1 profile; v2 at qwen3:14b num_ctx 4096: USD/day 10.33 to 137.18, cloud part priced, not executed [impact-v2-P1]) | not measured (v1 profile) |
| today's mix: cloud fees caching on (no caching); local time up to | projection | 353 (3,917); 1,929 | 382 (4,052); 7,879 |
| router saving vs naive local default |  | pending B5 | pending B5 |
| router saving vs all-cloud |  | pending B5 | pending B5 |
| router saving vs today's mix |  | pending B5 | pending B5 |
| hardware recommendation |  | pending B5 | pending B5 |

### Router replay per window (measured: which steps escalate and what they cost)

| profile | window | steps kept local | where the share comes from | router cloud USD / session caching on (no caching) | cache-hit share | router cloud USD / month caching on (no caching), projection |
|---|---|---:|---|---:|---:|---:|
| mid_team | num_ctx_4096 | 80.0% | 16 of 80 steps escalated in every one of 5 replayed sessions (same count each) | 0.256 (1.122) | 94% | 538 (2,356) |
| mid_team | num_ctx_8192 | 80.0% | 16 of 80 steps escalated in every one of 5 replayed sessions (same count each) | 0.255 (1.112) | 94% | 535 (2,335) |
| mid_team | num_ctx_16384 | 85.0% | 12 of 80 steps escalated in every one of 5 replayed sessions (same count each) | 0.234 (1.003) | 92% | 491 (2,107) |
| mid_team | num_ctx_32768 | 90.0% | 8 of 80 steps escalated in every one of 5 replayed sessions (same count each) | 0.433 (0.784) | 62% | 910 (1,646) |
| mid_team | ollama_default | 100.0% | no step escalated in any of 5 replayed sessions | 0.000 (0.000) | n/a | 0 (0) |
| mid_team_qwen3_14b | num_ctx_4096 | 80.0% | 16 of 80 steps escalated in every one of 5 replayed sessions (same count each) | 0.280 (1.254) | 94% | 588 (2,634) |
| mid_team_qwen3_14b | num_ctx_8192 | 80.0% | 16 of 80 steps escalated in every one of 5 replayed sessions (same count each) | 0.281 (1.259) | 94% | 590 (2,644) |
| mid_team_qwen3_14b | num_ctx_16384 | 82.5% | 14 of 80 steps escalated in every one of 5 replayed sessions (same count each) | 0.799 (1.224) | 53% | 1,677 (2,570) |
| mid_team_qwen3_14b | num_ctx_32768 | 87.5% | 10 of 80 steps escalated in every one of 5 replayed sessions (same count each) | 0.689 (1.039) | 50% | 1,448 (2,183) |
| mid_team_qwen3_14b | ollama_default | 90.0% | 8 of 80 steps escalated in every one of 5 replayed sessions (same count each) | 0.590 (0.891) | 50% | 1,240 (1,872) |

## SCENARIO (not a measurement): router savings under the +A credit

Everything in this section rests on three assumptions, none measured: (1) +A, the router leaves no silent failure where the pre-registered mitigation recovered [R2-mitigation-v1-verdict]; (2) P1 valued at the impact model's rerun proxy (each silent rule/recall failure = one wrong result costing one full session rerun); (3) local compute time saved is worth the full hour value. Plan item B5 replaces all three. Register rows business-scenario-plus-a and business-sensitivity.

| quantity | mid_team | mid_team_qwen3_14b |
|---|---:|---:|
| router window (minimises time + router cloud, caching on) | num_ctx_4096 | num_ctx_8192 |
| +A credit used | yes | no |
| saving vs naive default | 2,938 | 13,976 |
| saving vs naive default if nobody waits (time worth 0; router cloud fees remain) | -538 | -590 |
| hours saved vs naive / month | 53.9 | 226.0 |
| cloud spend avoided vs all-cloud: caching on (no caching) | 167 (5,478) | 173 (5,461) |
| saving vs today's mix, caching on | 1,362 | 6,478 |
| hardware recommendation (hardware cost not priced) | evo-x2 num_ctx_4096: 2,938 | evo-x2 num_ctx_8192: 13,976 |
| single-shot calls on the pareto choice: cloud avoided (STUB cloud reference) | 9.72 | 9.72 |

Single-shot choice (pareto-sample-recommendation-v2): evo-x2 qwen3-30b-a3b llama_server, saving USD 1.85 per 1k steps vs the all-cloud reference [STUB: cloud quality not measured].

### Scenario sensitivity (mid_team, one assumption varied at a time; USD/month; register row business-sensitivity)

| assumption | value | router window | saving vs naive | saving vs naive if nobody waits | cloud avoided vs all-cloud, caching on (no caching) | saving vs today's mix, caching on | hours saved vs naive |
|---|---|---|---:|---:|---:|---:|---:|
| team_size | 3 | num_ctx_4096 | 353 | -65 | 20 (657) | 163 | 6.5 |
| team_size | 10 | num_ctx_4096 | 1,175 | -215 | 67 (2,191) | 545 | 21.6 |
| team_size | 25 | num_ctx_4096 | 2,938 | -538 | 167 (5,478) | 1,362 | 53.9 |
| team_size | 50 | num_ctx_4096 | 5,877 | -1,076 | 335 (10,955) | 2,724 | 107.9 |
| agent_sessions_per_dev_per_day | 2 | num_ctx_4096 | 1,469 | -269 | 84 (2,739) | 681 | 27.0 |
| agent_sessions_per_dev_per_day | 4 | num_ctx_4096 | 2,938 | -538 | 167 (5,478) | 1,362 | 53.9 |
| agent_sessions_per_dev_per_day | 8 | num_ctx_4096 | 5,877 | -1,076 | 335 (10,955) | 2,724 | 107.9 |
| cloud_session_share | 0.25 | num_ctx_4096 | 2,938 | -538 | 167 (5,478) | 2,150 | 53.9 |
| cloud_session_share | 0.5 | num_ctx_4096 | 2,938 | -538 | 167 (5,478) | 1,362 | 53.9 |
| cloud_session_share | 0.75 | num_ctx_4096 | 2,938 | -538 | 167 (5,478) | 574 | 53.9 |
| hour_value | p10 (USD 39.64/h) | num_ctx_4096 | 1,601 | -538 | 167 (5,478) | 767 | 53.9 |
| hour_value | median (USD 64.44/h) | num_ctx_4096 | 2,938 | -538 | 167 (5,478) | 1,362 | 53.9 |
| hour_value | p90 (USD 103.20/h) | num_ctx_4096 | 5,029 | -538 | 167 (5,478) | 2,293 | 53.9 |

<!-- business_metrics:end -->

## How to read the scenario section

The scenario repeats the first version of this page's arithmetic on the new cloud basis: the router window is the
one that minimises (local time at the full hour value + router cloud fees, caching on), P1 is valued at the impact
model's assumed rerun proxy, and the router is credited with zero residual P1 and P7 where the pre-registered
mitigation recovered [R2-mitigation-v1-verdict] (llama3.1:8b at num_ctx 4096 and 8192, qwen3:14b at 8192; not
qwen3:14b at 4096). Its "if nobody waits" line is the same comparison with local time worth 0: the router is then left
with its cloud fees and saves nothing. The wage sensitivity in the scenario table moves only the upper end.

## What would change these numbers

1. Plan item B5 (the policy comparison on real sessions): replaces every "pending B5" with a measured result and the
   +A credit with a measured residual P1/P7 under the router.
2. B3 task runs under at least two windows, and developer rework time per wrong result: needed before P1 has any
   dollar value (docs/IMPACT.md, "Single most decisive missing measurement"). At num_ctx 4096 the silent rule/recall
   rate is "10/15 silent, 10 after window exceeded" over three models [R2-real-v1b-gated-kill].
3. Who waits for local compute: decides where in the 0-to-full-hour-value range the time rows sit.
4. Cloud latency for whole sessions: needed before any time saving vs all-cloud can be stated.
5. Hardware and power prices from a citable source: needed to turn any saving into a net saving.
