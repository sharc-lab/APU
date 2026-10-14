# Impact: what each problem costs, ranked

Plan v2 Part A3 and A6 (operator plan, 2026-10-13), answering the 2026-10-13 ask in docs/MEETINGS.md: quantify every
problem in money and time and rank the problems by cost.

How to read this page:

- Everything between the two generated markers below is written by `py -3.12 analysis/impact_model.py --write` from
  committed result files. The profile table rows are the register rows impact-P1 to impact-P7 and impact-total in
  docs/NUMBERS_REGISTER.md (same functions, same files); the hour value is register row impact-hour-value.
- The hour value is a cited public figure: the BLS Occupational Outlook Handbook median pay per hour for Software
  Developers, Quality Assurance Analysts, and Testers (May 2025 data, fetched 2026-10-14). Source URL, table and the
  figure as quoted are in results/impact_profile_20261014.json.
- Every other profile parameter (team size, sessions per developer per day, turns per session, single-shot calls,
  workdays, co-runner share, cloud share, the configuration priced) is an ASSUMPTION, stored in the same JSON under
  `assumptions` and printed as such in the generated block. Change the JSON to price a different customer.
- A row whose per-event cost is not measured prints "not measured", never a number. A measured rate of zero prints
  zero.
- Per model size and per device: the grid at the end of the generated block is computed from every R2 file matching
  `results/x2_r2_real*.jsonl` (evo-x2) or `results/blade_r2_real*.jsonl` (Blade), the decode-rate and co-runner
  register rows, and the two single-shot outcome tables. Plan v2 Part B results (R2 at 30B to 70B) enter by being
  written under those names; no code change is needed.

## Rows registered in parallel (not on main when this was written)

Another plan item registers r2-session-wallclock, r2-session-cloud-cost, info-loss-by-window and
reliability-after-guardrail-loss. This model does not wait for them: the inputs it needs (session wall time, prefill
and decode seconds, output tokens, session cloud cost) are computed here by small helpers
(`session_wallclock`, `r2_cloud_overspend` in analysis/impact_model.py) from the same committed files. Where a row
cites one of those ids it is marked "PENDING on main". Once they land, the two computations should agree; if they do
not, the register rows win and this model is the one to fix.

## Price table

This model prices every cloud call with results/cloud_prices_20261008.json through
analysis/cloud_budget_forecast.py `call_cost` (the same table and function as the cloud-budget-* rows and the Pareto
frontier).

Disagreement inside our own code, recorded here so it is fixed and not cited around: src/cloud/client.py still carries
its own older pricing table and model map (a cheap and a strong model that are not in cloud_prices_20261008.json), and
src/dse/router.py's `est_cost_usd` and demo/dashboard/fakes.py use that table. This model therefore takes only the
Router's decisions (which steps escalate) and reprices them with the shared table. Moving src/cloud/client.py onto
cloud_prices_20261008.json is a separate change (it touches the live spend cap path) and is not made here.

## Zach's KV results and hybrid demo (plan item A6)

Status on 2026-10-14: not found, nothing registered.

Searched, read-only:

- this repository, every branch and worktree (`git log --all`; `git cat-file -t` on each hash): commits 180dfbb9,
  78fc3478 and b172b287 are not objects in this repository;
- every git repository directly under the GitHub folder on this laptop (19 repositories, including this one),
  `git -C <repo> log --all --oneline` filtered for the three hashes and for "kv", "npu", "zach": no match;
- results/zachary/: holds only replication_remote_search_v3.json (the earlier orchestration-overhead replication,
  claim B-03), no KV residency or NPU export data.

So the three register rows are not created, and no number from Zach's 2026-10-13 update (docs/MEETINGS.md, quoted
there as reported) is used anywhere in this model. Pending, with the row id each would feed:

| pending row id | what it would hold | file needed from Zach |
|---|---|---|
| zach-kv-residency-xps | longest resident context on the XPS per KV dtype (f16, u8, u4) and the int8 vs int4 decode comparison (commit 180dfbb9) | the raw per-run results file(s) behind commit 180dfbb9, with the script that wrote them |
| zach-kv-residency-t2s | the same residency limits on evo-t2s for the 8B and 4B models | the raw per-run results file(s) for the evo-t2s runs |
| zach-npu-export | NPU vs GPU agreement for int4 g128 (commit 78fc3478) and int8 channel-wise (commit b172b287) | the per-prompt NPU and GPU outputs and the comparison script for both commits |

What is needed from Zach, so the files can be copied into results/zachary/ with provenance (source repository URL,
commit, path, sha256) and registered through analysis/numbers_register.py:

1. Push access or a link to the repository that holds commits 180dfbb9, 78fc3478 and b172b287 (they are not in any
   repository on this laptop).
2. For each commit, the data file(s) the numbers come from, not a summary table.
3. His hybrid demo code (the routing rule and the cost calculator), so the comparison below can be made against code
   rather than against his summary.

Agreement needed before his hybrid demo numbers and this model's can be put side by side (to check against his code
once it is available; nothing below is a measured disagreement yet):

- Price table: the demo must price cloud calls with results/cloud_prices_20261008.json (same models, same Standard
  vs Batch tier, same short/long context split, same cache-write and cached-input rules) or say which table it uses.
- Escalation definitions: the demo's "keeps a turn local" must mean no E1-E4 escalation in the sense of the
  generated block below (memory, fit, latency, quality, in that order, with the Router's CONTEXT_MARGIN on the
  window). His reported "share of turns kept local" is only comparable to this model if a turn that is trimmed locally and loses a needed turn
  counts as escalated (E4), not as local.
- Unit of cost: his demo reports USD per task and a total; this model reports USD per session and per day for a
  stated profile. The task definition (turns, tokens per turn) has to be stated to convert one into the other.

<!-- impact_model:begin -->

Generated by `analysis/impact_model.py --write`. Profile: hour value USD 64.44 (impact-hour-value); ASSUMPTIONS: team 10, 4 agent sessions per developer per day, 40 turns per session, 10 single-shot calls per developer per day, 21 workdays per month, co-runner share 0.25, cloud session share 0.5, configuration evo-x2 llama3.1:8b ollama_default, escalation model gpt-6.1-sol (standard).

### Profile table (status quo: Ollama default window)

| row | problem | USD/day | USD/month | measured | confidence | register row |
|---|---|---:|---:|---|---|---|
| P1 | silent wrong results from context loss | 0.00 | 0.00 | measured | low | impact-P1 |
| P2 | oversized-window time cost | 23.27 | 488.67 | measured | medium | impact-P2 |
| P3 | cloud overspend from re-sent history | 67.89 | 1,425.64 | measured | high | impact-P3 |
| P4 | hardware mismatch | 0.87 | 18.23 | measured | medium | impact-P4 |
| P5 | co-runner slowdown | 49.34 | 1,036.09 | measured | low | impact-P5 |
| P6 | failed starts and retries | 47.38 | 995.03 | measured | medium | impact-P6 |
| P7 | guardrail removal | 0.00 | 0.00 | measured | low | impact-P7 |
| total | measured rows only (P1, P2, P3, P4, P5, P6, P7) | 188.75 | 3,963.66 | excluded: none | | impact-total |

### Same team after one flag: `ollama_ctx_4096`

| row | USD/day | USD/month | measured |
|---|---:|---:|---|
| P1 | 1,858.48 | 39,028.06 | measured |
| P2 | 0.00 | 0.00 | measured |
| P3 | 67.89 | 1,425.64 | measured |
| P4 | 0.87 | 18.23 | measured |
| P5 | 6.40 | 134.33 | measured |
| P6 | not measured | not measured | not measured |
| P7 | not measured | not measured | not measured |
| total (measured rows) | 1,933.63 | 40,606.25 | excluded: P6, P7 |

### Rows: formula, measured inputs, missing inputs, fix, escalation

| row | formula | measured inputs (register ids) | missing inputs | fix | needs cloud escalation? |
|---|---|---|---|---|---|
| P1 | sessions/day x P(gated silent failure at/after the first over-window turn) x redo hours (median session wall time at Ollama's default window) x USD/h; lower bound | R2-real-v1b-gated-kill, R2-real-v1b-survival, R2-mechanism-verdict, x2-v3-truncation-all-models, r2-session-wallclock (PENDING on main; same files read here), info-loss-by-window (PENDING on main) | developer time to notice a silent wrong result and redo it, and the cost of one that is not noticed (not measured; the redo-time lower bound assumes the developer waits for one rerun); R2 sessions on evo-t2s and the Blade | client-side trimming of whole oldest turns to the usable window (never send an overflowing request), escalate on E4 | yes when a needed turn would be trimmed (E4); cost = Router replay, see the escalation table |
| P2 | sessions/day x max(0, median per-turn wall delta between the configured window and the next smaller measured window, on turns that fit both) x turns/session x USD/h | r2-session-wallclock (PENDING on main; same files read here), R2-real-v1-loaded-ctx, R2-real-v1b-first-events | a right-sized window for a full R2 session (every smaller R2 tier is outgrown before the last turn, R2-real-v1b-first-events); session timings on evo-t2s and the Blade | set num_ctx to the session's need, not the runtime default; combined with the P1 fix so a smaller window cannot silently cut history | no for the window change itself; E4 escalations of the P1 fix apply at the smaller window |
| P3 | sessions/day x cloud share x mean(session cost uncached - session cost with implicit caching), price table cloud_prices_20261008, R2 session token counts | cloud-budget-tokens, cloud-budget-inputs, r2-session-cloud-cost (PENDING on main; same computation here) | how much of a real customer's cloud history is a stable cacheable prefix (R2 sessions are append-only, the best case for caching) | keep the history an append-only prefix so implicit caching applies; trim history client-side | no (the fix lowers cloud spend) |
| P4 | sessions/day x output tokens/session x (1/decode rate on this device - 1/best measured rate for the model) x USD/h; decode only (prefill excluded), lower bound | decode-rate-per-machine, blade-dryrun-rates, blade-k1-layer-split, r2-session-wallclock (output tokens; PENDING on main) | session prefill time on evo-t2s and the Blade; the XPS (Zach's) rates are not in this repo | route each model to the device whose envelope fits it (src/dse/pareto.py recommend) | no between local devices; E1/E2 only when no local device fits |
| P5 | sessions/day x co-runner share x (session prefill s x (TTFT ratio - 1) + session decode s x (1/decode ratio - 1)) x USD/h | PX2-full-ratio-table, B3-corunner-6model, PX2-decode-ratio, r2-session-wallclock (PENDING on main) | R2 session timings on evo-t2s (B3 is TTFT only); how often customers run a heavy co-runner (assumed share) | co-runner-aware latency prediction in the router (co_runner flag) | only under a latency target (E3); cost depends on the customer's target, not modelled |
| P6 | single-shot calls/day x (P(timeout) x per-call timeout s + P(allocation failure) x seconds to fail) x USD/h | x2-v3-usability, t2s-outcome-allocation-failures, t2s-llamaserver-load-failures, A-24-flip-point-prediction | operator time to notice and restart a stuck runtime (not measured); llama-server start failures need the gitignored logs (register row only) | pre-check the memory fit (A-24 shared-heap model) and the latency before sending; route the rest | yes for calls that cannot fit or finish locally (E1/E3); cost = P(event) x price of those items |
| P7 | sessions/day x P(a rule in use first broken at/after the first over-window turn) x cost per violation | R2-real-v1b-first-events, R2-mechanism-lowlevel, R2-orphan-rates, R2-mitigation-v1-verdict, reliability-after-guardrail-loss (PENDING on main) | cost of one guardrail violation (not measured; depends on what the rule protects) | never send an overflowing request (the system prompt is always kept by client-side trimming) | no for system-prompt rules (the router never trims the system prompt) |

### Escalation definitions (src/dse/router.py Router.decide, in decision order)

- E1: memory: a measured boundary says the local configuration will not start (A-24 last n_ctx that started) -> cloud if the budget covers it, else refuse
- E2: fit: system prompt + current turn + answer budget exceed the usable window (num_ctx x (1 - CONTEXT_MARGIN)) even after trimming every older turn -> cloud, else refuse
- E3: latency: the request fits untrimmed but the predicted latency is over the caller's target -> cloud if affordable
- E4: quality: trimming whole oldest turns would drop a turn the step needs (needs_turns or an identifier in the current user text) and quality_floor > 0 (or require_full_history) -> cloud, else local_trimmed with quality_floor_met=False

### Ranking (status quo profile)

1. P3 cloud overspend from re-sent history: USD/day 67.89; confidence high (token counts and public prices only; the daily figure scales linearly with the assumed cloud share)
2. P5 co-runner slowdown: USD/day 49.34; confidence low (co-runner ratios come from single-shot microbenchmarks applied to session timings, and the co-runner share is an assumption)
3. P6 failed starts and retries: USD/day 47.38; confidence medium (rates are measured on the workload pack; the daily volume of single-shot calls is an assumption)
4. P2 oversized-window time cost: USD/day 23.27; confidence medium (paired same-seed turns that fit both windows isolate the window size, but the per-turn delta is scaled to a whole session by assumption and the right-sized window was not run)
5. P4 hardware mismatch: USD/day 0.87; confidence medium (decode rates are measured per machine, but on different workloads (the Blade's from short dry runs), and prefill is excluded, so it is a lower bound)
6. P1 silent wrong results from context loss: USD/day 0.00; confidence low (the per-session rate is measured on few sessions per cell (n in the grid), but the cost per event is a rerun-time lower bound; detection and rework time are not measured)
7. P7 guardrail removal: USD/day 0.00; confidence low (the rate is measured but the cost per violation is not; a nonzero rate prints not measured)

Single most decisive missing measurement: Developer time lost per silent wrong result (time to notice plus rework, and how often it is never noticed), measured on real agent sessions. It is the one input P1 and P7 lack: today P1 is priced at a rerun-time lower bound and P7 is not priced at all, and at any small window both rates are high (R2-real-v1b-gated-kill), so a measured per-event cost decides whether context loss ranks first or last.

### Escalation cost of the P1 fix (Router replay, E1-E4, priced at the price table)

| model | window | cloud steps / session | steps / session | USD / session | sessions |
|---|---|---:|---:|---:|---:|
| llama3.1:8b | num_ctx_4096 | 16.0 | 80.0 | 1.1218 | 5 |
| llama3.1:8b | ollama_default | 0.0 | 80.0 | 0.0000 | 5 |
| qwen3:14b | num_ctx_4096 | 16.0 | 80.0 | 1.2543 | 5 |
| qwen3:14b | ollama_default | 8.0 | 80.0 | 0.8915 | 5 |
| qwen3:8b | num_ctx_4096 | 15.6 | 79.4 | 1.2741 | 5 |
| qwen3:8b | ollama_default | 8.0 | 80.0 | 0.9175 | 5 |

### Per model and device grid (USD/day at the profile volumes)

| row | model@device/window | USD/day | inputs |
|---|---|---:|---|
| P1 | llama3.1:8b@num_ctx_16384 | 743.39 | 2/5 sessions, redo 0.72 h |
| P1 | llama3.1:8b@num_ctx_32768 | 0.00 | 0/5 sessions, redo 0.72 h |
| P1 | llama3.1:8b@num_ctx_4096 | 1,858.48 | 5/5 sessions, redo 0.72 h |
| P1 | llama3.1:8b@num_ctx_8192 | 1,486.78 | 4/5 sessions, redo 0.72 h |
| P1 | llama3.1:8b@ollama_default | 0.00 | 0/5 sessions, redo 0.72 h |
| P1 | qwen3:14b@num_ctx_16384 | 0.00 | 0/5 sessions, redo 3.37 h |
| P1 | qwen3:14b@num_ctx_32768 | 0.00 | 0/5 sessions, redo 3.37 h |
| P1 | qwen3:14b@num_ctx_4096 | 6,953.80 | 4/5 sessions, redo 3.37 h |
| P1 | qwen3:14b@num_ctx_8192 | 0.00 | 0/5 sessions, redo 3.37 h |
| P1 | qwen3:14b@ollama_default | 0.00 | 0/5 sessions, redo 3.37 h |
| P1 | qwen3:8b@num_ctx_16384 | 0.00 | 0/5 sessions, redo 1.19 h |
| P1 | qwen3:8b@num_ctx_32768 | 0.00 | 0/5 sessions, redo 1.19 h |
| P1 | qwen3:8b@num_ctx_4096 | 612.59 | 1/5 sessions, redo 1.19 h |
| P1 | qwen3:8b@num_ctx_8192 | 2,450.37 | 4/5 sessions, redo 1.19 h |
| P1 | qwen3:8b@ollama_default | 0.00 | 0/5 sessions, redo 1.19 h |
| P2 | llama3.1:8b@num_ctx_16384 | 0.00 | vs num_ctx_8192: -0.09 s/turn (n=25) |
| P2 | llama3.1:8b@num_ctx_32768 | 2.23 | vs num_ctx_16384: 0.08 s/turn (n=55) |
| P2 | llama3.1:8b@num_ctx_4096 | 0.00 | smallest measured window |
| P2 | llama3.1:8b@num_ctx_8192 | 0.00 | vs num_ctx_4096: -0.27 s/turn (n=10) |
| P2 | llama3.1:8b@ollama_default | 23.27 | vs num_ctx_32768: 0.81 s/turn (n=110) |
| P2 | qwen3:14b@num_ctx_16384 | 8.28 | vs num_ctx_8192: 0.29 s/turn (n=20) |
| P2 | qwen3:14b@num_ctx_32768 | 0.00 | vs num_ctx_16384: -0.11 s/turn (n=45) |
| P2 | qwen3:14b@num_ctx_4096 | 0.00 | smallest measured window |
| P2 | qwen3:14b@num_ctx_8192 | 0.00 | vs num_ctx_4096: -0.26 s/turn (n=10) |
| P2 | qwen3:14b@ollama_default | 81.42 | vs num_ctx_32768: 2.84 s/turn (n=97) |
| P2 | qwen3:8b@num_ctx_16384 | 1.55 | vs num_ctx_8192: 0.05 s/turn (n=20) |
| P2 | qwen3:8b@num_ctx_32768 | 0.00 | vs num_ctx_16384: -0.02 s/turn (n=45) |
| P2 | qwen3:8b@num_ctx_4096 | 0.00 | smallest measured window |
| P2 | qwen3:8b@num_ctx_8192 | 0.00 | vs num_ctx_4096: -0.01 s/turn (n=10) |
| P2 | qwen3:8b@ollama_default | 32.66 | vs num_ctx_32768: 1.14 s/turn (n=94) |
| P3 | llama3.1:8b | 67.89 | uncached 3.7302, cached 0.3358 USD/session (n=5) |
| P3 | qwen3:14b | 69.92 | uncached 3.8592, cached 0.3634 USD/session (n=5) |
| P3 | qwen3:8b | 71.84 | uncached 3.9853, cached 0.3933 USD/session (n=5) |
| P4 | llama-33-70b@evo-t2s | not measured | 2.32 tok/s vs best 6.42; output n/a (no R2 session) tok/session |
| P4 | llama-33-70b@evo-x2 | not measured | 6.42 tok/s vs best 6.42; output n/a (no R2 session) tok/session |
| P4 | llama31-8b@blade@32768 | 113.08 | 13.58 tok/s vs best 46.40; output 3032 tok/session |
| P4 | llama31-8b@blade@4096 | 0.00 | 46.40 tok/s vs best 46.40; output 3032 tok/session |
| P4 | llama31-8b@evo-t2s | 90.48 | 15.81 tok/s vs best 46.40; output 3032 tok/session |
| P4 | llama31-8b@evo-x2 | 0.87 | 45.55 tok/s vs best 46.40; output 3032 tok/session |
| P4 | qwen3-14b@evo-t2s | 175.56 | 11.69 tok/s vs best 27.43; output 4996 tok/session |
| P4 | qwen3-14b@evo-x2 | 0.00 | 27.43 tok/s vs best 27.43; output 4996 tok/session |
| P4 | qwen3-30b-a3b-2507@evo-t2s | not measured | 31.74 tok/s vs best 85.84; output n/a (no R2 session) tok/session |
| P4 | qwen3-30b-a3b-2507@evo-x2 | not measured | 85.84 tok/s vs best 85.84; output n/a (no R2 session) tok/session |
| P4 | qwen3-32b@evo-t2s | not measured | 4.57 tok/s vs best 12.43; output n/a (no R2 session) tok/session |
| P4 | qwen3-32b@evo-x2 | not measured | 12.43 tok/s vs best 12.43; output n/a (no R2 session) tok/session |
| P4 | qwen3-4b-2507@evo-t2s | not measured | 16.04 tok/s vs best 77.07; output n/a (no R2 session) tok/session |
| P4 | qwen3-4b-2507@evo-x2 | not measured | 77.07 tok/s vs best 77.07; output n/a (no R2 session) tok/session |
| P4 | qwen3-8b@evo-t2s | 148.72 | 19.83 tok/s vs best 50.11; output 6819 tok/session |
| P4 | qwen3-8b@evo-x2 | 0.00 | 50.11 tok/s vs best 50.11; output 6819 tok/session |
| P5 | llama3.1:8b@evo-x2@num_ctx_16384 | 19.82 | TTFT x1.111 |
| P5 | llama3.1:8b@evo-x2@num_ctx_32768 | 25.81 | TTFT x1.111 |
| P5 | llama3.1:8b@evo-x2@num_ctx_4096 | 6.40 | TTFT x1.111 |
| P5 | llama3.1:8b@evo-x2@num_ctx_8192 | 11.10 | TTFT x1.111 |
| P5 | llama3.1:8b@evo-x2@ollama_default | 49.34 | TTFT x1.111 |
| P5 | qwen3:14b@evo-x2@num_ctx_16384 | 46.09 | TTFT x1.100 |
| P5 | qwen3:14b@evo-x2@num_ctx_32768 | 104.52 | TTFT x1.100 |
| P5 | qwen3:14b@evo-x2@num_ctx_4096 | 11.04 | TTFT x1.100 |
| P5 | qwen3:14b@evo-x2@num_ctx_8192 | 22.73 | TTFT x1.100 |
| P5 | qwen3:14b@evo-x2@ollama_default | 218.75 | TTFT x1.100 |
| P5 | qwen3:8b@evo-x2@num_ctx_16384 | 26.20 | TTFT x1.123 |
| P5 | qwen3:8b@evo-x2@num_ctx_32768 | 46.84 | TTFT x1.123 |
| P5 | qwen3:8b@evo-x2@num_ctx_4096 | 8.63 | TTFT x1.123 |
| P5 | qwen3:8b@evo-x2@num_ctx_8192 | 14.44 | TTFT x1.123 |
| P5 | qwen3:8b@evo-x2@ollama_default | 89.50 | TTFT x1.123 |
| P5 | llama3.1:8b@evo-t2s@ollama_default | not measured | no R2 session timings on evo-t2s |
| P6 | llama31-8b@evo-t2s@ollama_default | 0.47 | timeouts 0, allocation failures 18 of 40; escalation USD 1.29/day |
| P6 | llama31-8b@evo-t2s@ollama_igpu_enable | 1.94 | timeouts 0, allocation failures 19 of 40; escalation USD 1.41/day |
| P6 | llama31-8b-gguf@evo-t2s@llama_server_vulkan | 0.00 | timeouts 0, allocation failures 0 of 20; escalation USD 0.00/day |
| P6 | llama31-8b@evo-x2@llama_server | 189.53 | timeouts 40, allocation failures 0 of 340; escalation USD 2.12/day |
| P6 | llama31-8b@evo-x2@ollama_default | 47.38 | timeouts 10, allocation failures 0 of 340; escalation USD 0.71/day |
| P6 | qwen3-14b@evo-x2@llama_server | 0.00 | timeouts 0, allocation failures 0 of 340; escalation USD 0.00/day |
| P6 | qwen3-14b@evo-x2@ollama_default | 0.00 | timeouts 0, allocation failures 0 of 340; escalation USD 0.00/day |
| P6 | qwen3-30b-a3b@evo-x2@llama_server | 47.38 | timeouts 10, allocation failures 0 of 340; escalation USD 0.71/day |
| P6 | qwen3-30b-a3b@evo-x2@ollama_default | 0.00 | timeouts 0, allocation failures 0 of 340; escalation USD 0.00/day |
| P6 | qwen3-32b@evo-x2@llama_server | 177.21 | timeouts 11, allocation failures 0 of 100; escalation USD 0.70/day |
| P6 | qwen3-32b@evo-x2@ollama_default | 0.00 | timeouts 0, allocation failures 0 of 100; escalation USD 0.00/day |
| P6 | qwen3-4b-2507@evo-x2@llama_server | 189.53 | timeouts 40, allocation failures 0 of 340; escalation USD 2.12/day |
| P6 | qwen3-4b-2507@evo-x2@ollama_default | 47.38 | timeouts 10, allocation failures 0 of 340; escalation USD 0.71/day |
| P6 | qwen3-8b@evo-x2@llama_server | 0.00 | timeouts 0, allocation failures 0 of 340; escalation USD 0.00/day |
| P6 | qwen3-8b@evo-x2@ollama_default | 0.00 | timeouts 0, allocation failures 0 of 340; escalation USD 0.00/day |
| P7 | llama3.1:8b@num_ctx_16384 | 0.00 | 0/5 sessions |
| P7 | llama3.1:8b@num_ctx_32768 | 0.00 | 0/5 sessions |
| P7 | llama3.1:8b@num_ctx_4096 | not measured | 4/5 sessions |
| P7 | llama3.1:8b@num_ctx_8192 | not measured | 1/5 sessions |
| P7 | llama3.1:8b@ollama_default | 0.00 | 0/5 sessions |
| P7 | qwen3:14b@num_ctx_16384 | 0.00 | 0/5 sessions |
| P7 | qwen3:14b@num_ctx_32768 | 0.00 | 0/5 sessions |
| P7 | qwen3:14b@num_ctx_4096 | not measured | 4/5 sessions |
| P7 | qwen3:14b@num_ctx_8192 | 0.00 | 0/5 sessions |
| P7 | qwen3:14b@ollama_default | 0.00 | 0/5 sessions |
| P7 | qwen3:8b@num_ctx_16384 | 0.00 | 0/5 sessions |
| P7 | qwen3:8b@num_ctx_32768 | 0.00 | 0/5 sessions |
| P7 | qwen3:8b@num_ctx_4096 | not measured | 1/5 sessions |
| P7 | qwen3:8b@num_ctx_8192 | not measured | 4/5 sessions |
| P7 | qwen3:8b@ollama_default | 0.00 | 0/5 sessions |

<!-- impact_model:end -->
