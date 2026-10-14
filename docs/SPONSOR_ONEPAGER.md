# Local AI agents fail quietly, and it costs money: sponsor one-pager

Rithwik Sharma, 2026-10-14. Plan v2 Part C2. Every number below is quoted from a row of docs/NUMBERS_REGISTER.md
(row id in brackets). "Projection" means measured per-session data times an assumed team size and workload;
"assumption" marks a modelling choice. Full tables: docs/IMPACT.md and docs/BUSINESS_METRICS.md.

## The problem in dollars

A 10-developer team that runs agent sessions half on a workstation-class AI PC at the runtime's default settings and
half on a cloud model loses an estimated "USD/month 3,963.66" across seven problems, two of them priced at zero at
this setting (projection; team size, sessions per day and cloud share are assumptions; the hour value is the BLS
median wage, [impact-hour-value]) [impact-total]. The largest
single item is paying the cloud to re-read the same history: "overspend USD 3.3944/session" without prefix caching
[impact-P3].

## Why it is hard: no single setting fixes it

- **The obvious fix makes it worse.** Changing one flag (a smaller context window, which removes the oversized-window
  time cost) moves the same team to "USD/month 40,606.25" (two rows not measured, so a lower bound), because history
  is then lost silently [impact-total].
- **The loss is silent.** At small windows agent sessions fail with no error: at num_ctx 4096, "10/15 silent, 10
  after window exceeded" [R2-real-v1b-gated-kill]. At the smallest window the runtime also cuts the system prompt
  (context shifts and a token cut, [R2-mechanism-lowlevel]).
- **The default depends on the machine.** The same runtime loads "131072" tokens for one model on one machine
  [R2-real-v1-loaded-ctx] and "default ctx 4096" on an 8 GB laptop GPU, where over-length prompts still return
  HTTP 200 [blade-k1-layer-split].
- **The other runtime trades one failure for another.** On a 32B model, llama-server silently truncated "0/74" rows
  but timed out 11 times; Ollama's default silently truncated "15/97" [qwen3-32b-truncation-vs-latency].

So the fix is a decision per step (keep it local, trim it, or send it to the cloud), not a flag. That is what our
router does (escalations E1 to E4 in docs/IMPACT.md).

## What we save (per month, router vs alternatives)

| team (assumption) | vs runtime default | vs all-cloud: cloud spend avoided | steps kept local | status |
|---|---|---|---|---|
| 3 devs, 8 GB laptops, llama3.1:8b | USD 11,426 | USD -198 to 657 | 80.0% | projection, rests on +A [business-small-team] |
| 25 devs, workstation, llama3.1:8b | USD 1,120 | USD -1,651 to 5,478 | 80.0% | projection, rests on +A [business-mid-team] |
| 25 devs, workstation, qwen3:14b | USD 11,922 | USD -1,881 to 5,461 | 80.0% | projection [business-mid-team-qwen3-14b] |

How to read it, without overclaiming:

- **Against the runtime default the router saves money in every profile**, mainly by avoiding silent context loss
  (small team) or oversized-window time, including the co-runner slowdown that grows with it (the two workstation
  teams). "+A" means the figure assumes the router leaves no
  silent failures where a pre-registered mitigation recovered [R2-mitigation-v1-verdict]; that has not yet been
  measured end to end (plan item B5 has not run).
- **Against all-cloud, the router is not cheaper in cloud fees if the cloud provider caches the history.** The
  negative low end is all-cloud with prefix caching against our escalations priced at an upper bound. The router
  saves cloud spend only against uncached all-cloud. Time saved vs all-cloud is not measured.
- At the lowest wage percentile the workstation llama3.1:8b saving falls to "vs naive 0" [business-sensitivity];
  the savings scale linearly with team size and sessions per day.
- Hardware and power are not priced in any figure.

## Evidence status

| number | status |
|---|---|
| silent-failure rates, truncation mechanism, default windows, decode rates | measured on real sessions and runs |
| cloud cost per session, share of steps the router keeps local | measured (token counts, public price table, Router replay of 5 sessions per window) |
| USD per month for any team | projection (assumed team size, workload, cloud share) |
| router leaves no silent failures | assumption, partly measured [R2-mitigation-v1-verdict]; end-to-end test (B5) not run |
| laptop-team session outcomes | assumption: transferred from the workstation at the same window (no laptop session completed) |
| developer time per silent wrong result | not measured (priced at one rerun, a lower bound) |
| time saved vs all-cloud; hardware cost | not measured; not priced |

## What we need

1. **Run time for the policy comparison (B5)** on the two workstations: it turns every "+A" above into a measurement.
2. **Real-session data on rework time** (how long a developer loses to one silent wrong result): the single input that
   decides whether context loss is the top cost or a minor one (docs/IMPACT.md).
3. **From Zach** (docs/IMPACT.md, plan item A6): access to the repository holding commits 180dfbb9, 78fc3478 and
   b172b287; the raw data files behind each (not summary tables); and his hybrid demo code (routing rule and cost
   calculator), priced on our price table and escalation definitions so the two results can be compared.
4. **Cited hardware and power prices**, so savings can be stated net of hardware.
