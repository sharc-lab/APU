# Business metrics: what the router saves a customer team

Plan v2 Part C1 (operator plan, 2026-10-13). Built only from the impact model (plan item A3, docs/IMPACT.md,
analysis/impact_model.py) and its Router replay. Every number on this page is in the generated block below or in a
register row of docs/NUMBERS_REGISTER.md (row ids in brackets); the headline per-profile figures are register rows
business-small-team, business-mid-team, business-mid-team-qwen3-14b, business-sensitivity and business-hour-range.

## Status of the inputs

- **Plan item B5 (policy comparison) has not run** (its agent was stopped). So there is no measured end-to-end run of
  the router against the other policies on real sessions. What this page uses instead is the impact model's Router
  replay: the evo-x2 R2 sessions replayed through src/dse/router.py (escalations E1 to E4), each escalated step
  priced with the shared price table (docs/IMPACT.md, "Escalation cost of the P1 fix"). Which steps the router
  sends to the cloud, and what that costs, is computed. Whether the router's answers are then as good as the
  alternatives' is not measured, except through the mitigation verdict below.
- **Profiles are assumptions.** results/business_profiles_20261014.json holds three customer profiles as parameter
  sets (team size, sessions per developer per day, turns per session, cloud share today, workdays, co-runner share,
  machine, model). Every one of those fields is an ASSUMPTION. The hour value is the cited BLS median
  [impact-hour-value]; its range (p10, p90) is derived from the same BLS page's annual percentiles
  [business-hour-range] (quotes, URL, fetch date and the conversion are in the JSON).
- No market-size figure is used: none was fetched from a citable source for this page.

## Policies compared

All per session, then scaled by the profile's assumed volumes (sessions per month = team x sessions per developer
per day x workdays).

| policy | cloud spend per session | time and quality cost per session |
|---|---|---|
| all-cloud | the whole R2 session sent to the escalation model, priced with implicit prefix caching (low end) and without it (high end); the input of [impact-P3] | not measured (cloud latency for these sessions was never measured) |
| naive local default | 0 | impact model rows P1, P2, P4, P5, P7 at the runtime's own default window on that machine (the A3 status quo) |
| today's mix | the profile's cloud share of sessions at the all-cloud price, the rest at the naive local default | as for its two parts |
| router | the Router replay's escalated steps at that window, priced uncached with the step's full answer budget as output: an upper bound | impact model rows at the router's window, with P1 and P7 credited to zero only where allowed (next section) |

The router's window is the replayed window with the lowest router total (time cost plus cloud spend); a window is
only a candidate if it prices every row the naive default prices, so no window looks cheaper by leaving a cost out.
P3 (re-sent history on cloud sessions) and P6 (single-shot failures) are outside these per-session policies; P6's
router effect is not measured. Single-shot calls appear once, through [pareto-sample-recommendation-v2].

## The one router assumption (+A)

At a window where the runtime alone loses context (P1 or P7 nonzero or not measured), the router is credited with a
residual P1 and P7 of zero, because it never sends an overflowing request (client-side trimming of whole oldest turns
keeps the system prompt) and escalates a step whose needed turn would be trimmed (E4). This credit is given only
where the pre-registered mitigation measured "recovered" for that model and window [R2-mitigation-v1-verdict]:
llama3.1:8b at num_ctx 4096 and 8192, qwen3:14b at 8192. Where it measured "not_recovered" (qwen3:14b at 4096) or did
not run (any 16384 window), the router gets no credit and keeps the naive P1 and P7. Every figure that rests on the
credit, or on the session proxy device below, is labelled "+A".

The small-team profile runs on laptop-class machines (the NVIDIA laptop of the Blade arm). Its default window is
measured, not assumed: "default ctx 4096" for llama3.1:8b, with over-length prompts answered HTTP 200 at the half
window [blade-k1-layer-split]. But no R2 session completed on that laptop, so its session outcomes are taken from
evo-x2 sessions at the same window (ASSUMPTION; supported in part by the equal decode rate at 4096,
[blade-dryrun-rates]). Its co-runner cost (P5) is not transferred and prints "not measured", so its totals omit P5.

## How to read the results

- **Measured** per-session quantities (cloud USD per session, steps kept local, the hours behind each USD figure)
  come from result files. **Projection** figures multiply them by assumed volumes. Nothing per month is measured.
- Cloud spend avoided vs all-cloud is a range because all-cloud is priced with and without prefix caching. The low
  end is negative in every profile: with prefix caching, sending whole sessions to the cloud costs less in cloud
  fees than the router's escalated steps priced at their upper bound (uncached, full answer budget). The router
  saves cloud spend only against all-cloud without caching. The router's own escalation price is an upper bound, so
  the true low end is less negative than shown, by an amount that is not measured.
- Savings vs the naive local default are the A3 cost rows at the default window minus the same rows at the router's
  window plus the router's cloud spend. For the mid-size llama3.1:8b team the default window is very large ("131072"
  for llama3.1:8b, [R2-real-v1-loaded-ctx]), so nothing is lost silently and the saving is time only; it rests on
  the +A credit at num_ctx 4096 and disappears at the p10 hour value (sensitivity table: the router then keeps the
  default window). For qwen3:14b ("40960", [R2-real-v1-loaded-ctx]) the saving at num_ctx 8192 needs no +A credit:
  P1 and P7 are measured zero there.
- Time saved vs all-cloud is not measured. Hardware purchase cost and power are not priced anywhere in this repo,
  so every "saves vs" figure is gross of hardware. The contention assumption (every session at single-user speed;
  how many machines a team needs is not modelled) is in the profile JSON.

<!-- business_metrics:begin -->

Generated by `analysis/business_metrics.py --write` from results/business_profiles_20261014.json and the impact model's functions. USD per month unless stated. Labels: measured = per-session quantity from result files; projection = measured per-session quantity x ASSUMED profile volumes; +A = also rests on the residual-P1/P7 = 0 router assumption or on the session proxy device (see How to read).

Hour values: median USD 64.44 (impact-hour-value); p10 USD 39.64 and p90 USD 103.20 (business-hour-range, derived from BLS annual percentiles at 2080.1 h/year).

### Per-profile savings (register rows business-*)

| quantity | label | small_team | mid_team | mid_team_qwen3_14b |
|---|---|---:|---:|---:|
| team size; sessions/dev/day; cloud share today | assumption | 3; 4; 0.5 | 25; 4; 0.5 | 25; 4; 0.5 |
| device; model | assumption | blade; llama3.1:8b | evo-x2; llama3.1:8b | evo-x2; qwen3:14b |
| sessions / month | assumption | 252 | 2,100 | 2,100 |
| naive local default window | measured | num_ctx_4096 | ollama_default | ollama_default |
| router setting (window minimising router total) | measured inputs | num_ctx_4096 | num_ctx_4096 | num_ctx_8192 |
| steps kept local under the router | measured (Router replay) | 80.0% | 80.0% | 80.0% |
| cloud USD / session: all-cloud cached; uncached | measured | 0.336; 3.730 | 0.336; 3.730 | 0.363; 3.859 |
| cloud USD / session: router escalations (upper bound) | measured (Router replay) | 1.122 | 1.122 | 1.259 |
| all-cloud spend (cached to uncached) | projection | 85 to 940 | 705 to 7,833 | 763 to 8,104 |
| naive local: time and quality cost | projection | 11,708 (not measured: P5, P7) | 3,857 | 15,759 |
| router: cloud spend (upper bound) | projection | 283 | 2,356 | 2,644 |
| router: time cost | projection | 0 (not measured: P5) | 381 | 1,193 |
| cloud spend avoided vs all-cloud | projection | -198 to 657 | -1,651 to 5,478 | -1,881 to 5,461 |
| time saved vs all-cloud | not measured | not measured (cloud latency of these sessions) | not measured (cloud latency of these sessions) | not measured (cloud latency of these sessions) |
| saving vs naive local default |  | 11,426 (projection +A) | 1,120 (projection +A) | 11,922 (projection) |
| developer hours saved vs naive / month |  | 181.7 (projection +A) | 53.9 (projection +A) | 226.0 (projection) |
| saving vs today's mix (cached to uncached) |  | 5,614 to 6,042 (projection +A) | -456 to 3,108 (projection +A) | 4,424 to 8,095 (projection) |
| hardware recommendation | measured inputs | blade num_ctx_4096 + router | evo-x2 num_ctx_4096 + router | evo-x2 num_ctx_8192 + router |
| recommendation saves vs naive default |  | 11,426 (projection +A; hardware cost not priced) | 1,120 (projection +A; hardware cost not priced) | 11,922 (projection; hardware cost not priced) |
| single-shot calls on the pareto choice: cloud spend avoided | projection, STUB cloud reference | 1.17 | 9.72 | 9.72 |

Single-shot choice (pareto-sample-recommendation-v2): evo-x2 qwen3-30b-a3b llama_server, saving USD 1.85 per 1k steps vs the all-cloud reference [STUB: cloud quality not measured].

### Router candidates per profile (window: router total USD/month = time + cloud; steps local)

| profile | device | window | naive total | router time | router cloud (upper bound) | router total | steps local | not measured | +A |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| small_team | blade | num_ctx_4096 | 11,708 | 0 | 283 | 283 | 80.0% | P5 | yes |
| small_team | blade | num_ctx_32768 | 726 | 726 | 197 | 924 | 90.0% | P5 | yes |
| small_team | evo-x2 | num_ctx_4096 | 11,754 | 46 | 283 | 328 | 80.0% | none | yes |
| small_team | evo-x2 | num_ctx_8192 | 9,442 | 75 | 280 | 356 | 80.0% | none | yes |
| small_team | evo-x2 | num_ctx_16384 | 4,814 | 4,814 | 253 | 5,067 | 85.0% | none | no |
| small_team | evo-x2 | num_ctx_32768 | 182 | 182 | 197 | 380 | 90.0% | none | no |
| small_team | evo-x2 | ollama_default | 463 | 463 | 0 | 463 | 100.0% | none | no |
| mid_team | evo-x2 | num_ctx_4096 | 97,952 | 381 | 2,356 | 2,737 | 80.0% | none | yes |
| mid_team | evo-x2 | num_ctx_8192 | 78,684 | 628 | 2,335 | 2,963 | 80.0% | none | yes |
| mid_team | evo-x2 | num_ctx_16384 | 40,114 | 40,114 | 2,107 | 42,221 | 85.0% | none | no |
| mid_team | evo-x2 | num_ctx_32768 | 1,518 | 1,518 | 1,646 | 3,164 | 90.0% | none | no |
| mid_team | evo-x2 | ollama_default | 3,857 | 3,857 | 0 | 3,857 | 100.0% | none | no |
| mid_team_qwen3_14b | evo-x2 | num_ctx_8192 | 1,193 | 1,193 | 2,644 | 3,837 | 80.0% | none | no |
| mid_team_qwen3_14b | evo-x2 | num_ctx_16384 | 2,854 | 2,854 | 2,570 | 5,424 | 82.5% | none | no |
| mid_team_qwen3_14b | evo-x2 | num_ctx_32768 | 5,488 | 5,488 | 2,183 | 7,670 | 87.5% | none | no |
| mid_team_qwen3_14b | evo-x2 | ollama_default | 15,759 | 15,759 | 1,872 | 17,631 | 90.0% | none | no |

### Sensitivity (mid_team, one assumption varied at a time; USD/month, all projection; register row business-sensitivity)

| assumption | value | router window | saving vs naive | cloud avoided vs all-cloud | saving vs today's mix | hours saved vs naive |
|---|---|---|---:|---:|---:|---:|
| team_size | 3 | num_ctx_4096 | 134 | -198 to 657 | -55 to 373 | 6.5 |
| team_size | 10 | num_ctx_4096 | 448 | -660 to 2,191 | -182 to 1,243 | 21.6 |
| team_size | 25 | num_ctx_4096 | 1,120 | -1,651 to 5,478 | -456 to 3,108 | 53.9 |
| team_size | 50 | num_ctx_4096 | 2,241 | -3,301 to 10,955 | -912 to 6,217 | 107.9 |
| agent_sessions_per_dev_per_day | 2 | num_ctx_4096 | 560 | -825 to 2,739 | -228 to 1,554 | 27.0 |
| agent_sessions_per_dev_per_day | 4 | num_ctx_4096 | 1,120 | -1,651 to 5,478 | -456 to 3,108 | 53.9 |
| agent_sessions_per_dev_per_day | 8 | num_ctx_4096 | 2,241 | -3,301 to 10,955 | -912 to 6,217 | 107.9 |
| cloud_session_share | 0.25 | num_ctx_4096 | 1,120 | -1,651 to 5,478 | 332 to 2,114 | 53.9 |
| cloud_session_share | 0.5 | num_ctx_4096 | 1,120 | -1,651 to 5,478 | -456 to 3,108 | 53.9 |
| cloud_session_share | 0.75 | num_ctx_4096 | 1,120 | -1,651 to 5,478 | -1,244 to 4,102 | 53.9 |
| hour_value | p10 (USD 39.64/h) | ollama_default | 0 | 705 to 7,833 | -834 to 2,730 | 0.0 |
| hour_value | median (USD 64.44/h) | num_ctx_4096 | 1,120 | -1,651 to 5,478 | -456 to 3,108 | 53.9 |
| hour_value | p90 (USD 103.20/h) | num_ctx_4096 | 3,211 | -1,651 to 5,478 | 475 to 4,039 | 53.9 |

<!-- business_metrics:end -->

## What would change these numbers

1. Plan item B5 (the policy comparison on real sessions): replaces the +A credit with a measured residual P1/P7
   under the router, and gives the router's real cloud spend in place of the replay's upper bound.
2. Cloud latency for whole sessions: needed before any time saving vs all-cloud can be stated.
3. Developer time lost per silent wrong result (docs/IMPACT.md, "Single most decisive missing measurement"): P1 is
   priced at a rerun-time lower bound, and P1 is almost all of the small-team naive cost (at num_ctx 4096, "10/15 silent, 10 after window exceeded" over three
   models, [R2-real-v1b-gated-kill]).
4. R2 sessions on the laptop arm: removes the session proxy device from the small-team profile.
5. Hardware and power prices from a citable source: needed to turn "saves vs" into a net saving.
