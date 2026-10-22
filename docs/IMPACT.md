# Impact: what each problem costs, ranked

Measured cost today: USD 3.60/day for the reference team; the larger costs depend on the wrong-task rate (pending B3
task runs) and on whether anyone waits for compute. (Register row impact-total: "measured-only USD/day 3.60";
projection at the assumed 10-developer reference team.)

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
  zero, except P1, which prints "not measured" at every window until B3 task runs exist (see the audit below).
- Per model size and per device: the grid at the end of the generated block is computed from every R2 file matching
  `results/x2_r2_real*.jsonl` (evo-x2) or `results/blade_r2_real*.jsonl` (Blade), the decode-rate and co-runner
  register rows, and the two single-shot outcome tables. Plan v2 Part B results (R2 at 30B to 70B) enter by being
  written under those names; no code change is needed.

## Impact v2: P1 from the measured wrong-task rate, one stated profile (2026-10-22; P1 framing and P2 valuation 2026-10-21 operator decisions)

**Fixing a wrong answer in the cloud costs about USD 0.03 (priced, not executed); the cost is that the failure is
silent, so you do not know which answers to fix.** Row cloud-redo-cost-per-wrong-task: "USD per wrong task mean
$0.0283" (stub mode, priced from the local run's token counts, no API call), and at the profile cell "ollama_ctx_4096:
wrong 63/150 (42.0%) (silent 63, surfaced 0)" [x2-evalset-tasks-real-v2-wrong-rate]: every wrong task was silent.
Detection, not the redo, is what is worth paying for (measured detector below).

Operator priority override of 2026-10-21. This section supersedes the v1 P1 "not measured" and the v1 ranking for
decisions; the v1 generated block below is kept unchanged for traceability. Code: `analysis/impact_model.py`
(`p1_v2`, `p2_v2`, `p2_capacity`, `measured_detection`, `v2_rows`, `compute_v2`); register rows impact-v2-P1,
impact-v2-P2, impact-v3-detection-value-measured, impact-ref-machine-price and impact-ranking-v2
(impact-v2-detection-value still computes but is superseded).

**Profile (one, stated).** evo-x2, qwen3:14b, Ollama at num_ctx 4096. Why: qwen3:14b is the model with a measured
wrong-task rate (B3 v2 task real run), and 4096 is the window Ollama loads by default on a CPU-only machine
([t2s-r2-real-cpu]: "loaded window [4096]"), so a CPU-only team is in this cell without choosing it. Volumes (team 10,
4 agent sessions per developer per day, cloud share 0.5, co-runner share 0.25) are ASSUMPTIONS from
results/impact_profile_20261014.json; the hour value is cited ([impact-hour-value]).

**P1 formula.** sessions/day x task turns per session x excess wrong-task rate over the default window x (cloud redo
USD per wrong task + wall time of the wrong task turn x hour value x share of that time someone waits, 0 to 1).
- Measured: the excess rate per tier and its session-level 95% CI ([x2-evalset-tasks-real-v2-wrong-rate],
  [x2-evalset-tasks-real-v2-ci-units]); task turns per session; the wall time of the wrong turns (same file).
- Priced, not executed: cloud redo USD per wrong task, per tier, register row cloud-redo-cost-per-wrong-task (stub
  mode: the local run's token counts priced at the cloud escalation model; cloud correctness and seconds not
  measured). P1's cloud part carries that label, so P1 is mixed, not measured; it is no longer a lower bound. When a
  live redo replaces the stub file, regeneration relabels the cloud part measured with no code change.
- Assumed: volumes; who waits; that a wrong task costs one cloud redo plus the wasted turn (time to notice a wrong
  result is not counted).

Pasted from [impact-v2-P1]: "status mixed, cloud part priced, not executed; profile cell USD/day 10.33 to 137.18";
per tier: "ollama_ctx_4096: excess +30.7 points (session-level 95% CI +26.0 to +34.7), 30 task turns/session, 368.0
excess wrong tasks/day, wrong-turn wall 19.3 s; time USD/day 0.00 to 126.85 (CI 107.55 to 143.39); cloud redo USD/day
10.33 (USD 0.0281 per wrong task, priced, not executed; CI 8.76 to 11.68)"; "ollama_ctx_8192: ... time USD/day 0.00
to 157.24 (CI 122.30 to 192.19); cloud redo USD/day 8.72"; "ollama_ctx_4096_numkeep: ... time USD/day 0.00 to 48.84
(CI 41.87 to 55.82); cloud redo USD/day 6.03"; "ollama_ctx_16384: excess +1.3 points (session-level 95% CI -1.3 to
+4.0) ... time USD/day 0.00 to 3.57 (CI 0.00 to 10.72); cloud redo USD/day 0.43".

**Value of detection (MEASURED detector, row impact-v3-detection-value-measured; operator item 4, 2026-10-22).** This
replaces the earlier projection (row impact-v2-detection-value, now marked superseded), which borrowed a context-shift
predictor's precision and recall from a different model and job. The detector is now scored on the B3 v2 real run
itself (qwen3:14b, five tiers, 150 task turns per tier): flag a task when its turn's prompt exceeded the loaded window
(over_loaded_window), and count it right when the task ended wrong (end_state_ok false). Cloud USD compares escalating
every task with escalating only the flagged ones, at the tier's cloud redo USD per task (row
cloud-redo-cost-per-wrong-task, priced, not executed). Pasted from [impact-v3-detection-value-measured]:
- "ollama_ctx_4096: TP 63 FP 84 FN 0 TN 3, precision 42.9%, recall 100.0%, wrong tasks missed 0/63 (0.0%); escalate
  all 150 USD 4.2098 vs flagged 147 USD 4.1256 (saves USD 0.0842 and leaves 0 wrong unfixed; USD 0.0281 per task)"
- "ollama_ctx_4096_numkeep: TP 45 FP 102 FN 0 TN 3, precision 30.6%, recall 100.0%, wrong tasks missed 0/45 (0.0%);
  escalate all 150 USD 4.0355 vs flagged 147 USD 3.9548 (saves USD 0.0807 and leaves 0 wrong unfixed; USD 0.0269 per
  task)"
- "ollama_ctx_8192: TP 48 FP 58 FN 5 TN 39, precision 45.3%, recall 90.6%, wrong tasks missed 5/53 (9.4%); escalate
  all 150 USD 4.5423 vs flagged 106 USD 3.2099 (saves USD 1.3324 and leaves 5 wrong unfixed; USD 0.0303 per task)"
- "ollama_ctx_16384: TP 4 FP 22 FN 15 TN 109, precision 15.4%, recall 21.1%, wrong tasks missed 15/19 (78.9%);
  escalate all 150 USD 4.0764 vs flagged 26 USD 0.7066 (saves USD 3.3698 and leaves 15 wrong unfixed; USD 0.0272 per
  task)"
- "ollama_default: TP 0 FP 0 FN 17 TN 133, precision n/a, recall 0.0%, wrong tasks missed 17/17 (100.0%)"

Reading: at the profile cell (num_ctx 4096) the window check catches every wrong task, but it flags 147 of 150 tasks,
so it saves almost nothing over escalating everything; it is a "this whole configuration is unsafe" signal, not a
per-task filter. At 8192 it is a useful filter (90.6% recall on 106 of 150 tasks). At 16384 and the default window
most wrong tasks happen inside the window, so the check misses them (78.9% and 100.0%). Assumed: a redo of any task
costs the tier's mean redo of a wrong one; a fixed task is right after the redo (cloud correctness is not measured in
stub mode). When a live cloud redo file exists, the USD labels switch to measured automatically (the same
cloud_redo_data stub flag as P1).

**P2 valuation: machine capacity (operator decision 2026-10-21; row impact-v2-P2, PENDING).** The measured session
minutes of the runtime default (B5 policy a) and the fastest measured B5 setting (row cost-cut-config-time; the
flash-attention control joins once synced) give sessions per machine per day (24 h / session minutes, one session at
a time: there is no measured concurrency row), the machines needed for the profile's 40 sessions/day (ceil), and the
USD of machines saved, amortized per day over an ASSUMED 3-year life. Reference machine price: row
impact-ref-machine-price, "ASSUMPTION, must be verified: USD 2,000 for evo-x2 class: AMD Ryzen AI Max+ 395, 128 GB
LPDDR5X mini PC (GMKtec EVO-X2, 128 GB / 2 TB configuration)", a launch list price quoted by Liliputing (source in
results/hardware_price_evox2_20261021.json), not a purchase record; the repo had no cited price
(configs/hardware/evox2_strix_halo_128gb.yaml `bom_cost_usd: null`). Waiting time at the hour value is now only an
UPPER BOUND, applied to an ASSUMED interactive-session share of 0.25.

| qwen3:14b, 40 sessions/day | default (a) | fastest measured (g) kv_q8_0 |
|---|---|---|
| minutes per session (MEASURED) | 214.8 | 75.4 |
| sessions per machine per day | 6.70 | 19.11 |
| machines needed (ceil) | 6 | 3 |
| machines saved x USD 2,000 / 3 years | | USD 5.48/day |
| upper bound: waiting time x interactive share 0.25 | | USD 1,497.62/day on top |

Pasted from [impact-v2-P2]: "PENDING (values computed, not ranked) until row b5-fa-control is VERIFIED; machine
capacity: qwen3:14b default (a) 214.8 min per session = 6.70 sessions/machine/day vs (g) kv_q8_0 75.4 min = 19.11
... 6 default vs 3 best (ceil), 3 saved x USD 2,000 ... = USD 5.48/day; upper bound: waiting time 139.4 min saved per
session x hour value x interactive-session share 0.25 (ASSUMED) = USD 1,497.62/day on top". Status PENDING because
the B5 default ran without flash attention, so the saving may be flash attention or KV precision; the flash-attention
control (row b5-fa-control) decides which. Until that row is VERIFIED, P2 is shown with its computed values but is not
ranked. **P3** as in v1. **P4 to P7**: the v1 functions at the v2 profile.

**Ranking and status split**, pasted from [impact-ranking-v2]: "ranking by main-case USD/day (P2: machine capacity;
the other rows: time at the full hour value; a PENDING row is not ranked): P1 > P5 > P3 > P4; unranked: P2, P6, P7";
"P1 137.18 (range 10.33 to 137.18; mixed, cloud part priced, not executed); P5 11.04 (range 0.00 to 11.04; mixed); P3
3.70 (range 3.70 to 74.10; measured); P4 0.00 (range 0.00; mixed); P2 PENDING (computed 5.48, not ranked) (range 5.48
to 1,503.10; ...)"; "split: PENDING P2, measured P3, mixed P1/P4/P5, not measured P6/P7; PENDING inputs:
b5-fa-control".

| | v1 (impact-P1..P7, impact-total; llama3.1:8b, Ollama default) | v2 (impact-ranking-v2; qwen3:14b, num_ctx 4096) |
|---|---|---|
| P1 | not measured (status assumed) | 10.33 to 137.18 USD/day (cloud part priced, not executed) |
| P2 | 0.00 to 23.27 USD/day (oversized-window time) | PENDING: 5.48 USD/day machine capacity, waiting upper bound to 1,503.10 |
| ranking | P5 > P6 > P2 > P3 > P4 > P7 > P1 (P1 unranked) | P1 > P5 > P3 > P4; P2 (PENDING), P6, P7 unranked |
| measured | P3, P7 | P3 |
| mixed | P2, P4, P5, P6 | P1, P4, P5 |
| assumed | P1 | none |
| PENDING / not measured | none | P2 PENDING (b5-fa-control); P6, P7 not measured at 4096 |

Why P6 and P7 drop out: the single-shot failure table exists only at the Ollama default window, and at 4096 qwen3:14b
breaks a rule after the window, so P7's nonzero rate needs a cost per violation that is not measured. P2 is a
different quantity in v2 (machines saved by configuration, B5) than in v1 (oversized-window time). Valued as machine
capacity it is small (USD 5.48/day at the assumed price and life); the earlier v2 figure that ranked it first priced
every saved minute at the full hour value, which now survives only as the upper bound on the interactive share. The
v1 ranking (P5 first) was priced at a different profile (llama3.1:8b at the default window, where P1 had no number),
so the two rankings are not a like-for-like change of one input.

**Text prepared for docs/OCT29_COST_SUMMARY.md (file not created).** "Fixing a wrong answer in the cloud costs about
USD 0.03 (priced, not executed); the cost is that the failure is silent, so you do not know which answers to fix. At
num_ctx 4096, qwen3:14b got 42.0% of tasks wrong and every one was silent [x2-evalset-tasks-real-v2-wrong-rate]. In
that cell, a check for "the prompt exceeded the loaded window" caught every wrong task (recall 100.0%) but flagged 147
of 150 tasks (precision 42.9%), so escalating only flagged tasks costs about the same as escalating all of them (USD
4.1256 vs 4.2098 for the run's 150 tasks); measured on the same run [impact-v3-detection-value-measured]. The configuration saving is PENDING the flash-attention control: valued as machines, it
is 3 of 6 machines (USD 5.48/day at an assumed USD 2,000 price and 3-year life), with waiting time an upper bound of
USD 1,497.62/day if a quarter of sessions are interactive [impact-v2-P2]."

## Hybrid savings vs all-cloud at matched quality (2026-10-22; operator item 3, framing per later decisions)

Savings are measured against running every task on the cloud, at matched quality, per device and model, on the 30 B3
v2 tasks. A task counts as "kept local at matched quality" only where the local run is correct AND the all-cloud tier
is correct on the same task and seed. The all-cloud run has seeds 20260901 to 20260903 and the local run has 5 seeds,
so the table uses only the 3 shared seeds. Every number below is from register row hybrid-savings-v1
(analysis/impact_model.py::hybrid_savings).

Policies per device: (1) all-cloud, the reference; (2) naive local: USD 0 cloud spend, every wrong task silent; (3)
hybrid: escalate every task the window-overflow flag marks (the task turn's prompt exceeded the loaded window); (4)
hybrid with a perfect detector: escalate exactly the tasks that end wrong locally. Policy (4) is an UPPER BOUND, not a
detector that exists. An escalated task is assumed to take the all-cloud outcome of the same task and seed.

**Window overflow is not a general wrong-answer detector.** It only sees tasks whose prompt outgrew the loaded window.
It misses every wrong task that stays inside the window: at evo-x2's default window it flags nothing, so policy (3)
there is the same as naive local, and every default-window wrong task stays silent [hybrid-savings-v1,
impact-v3-detection-value-measured]. Where the window is small (4096) almost every task overflows, so the flag catches
the wrong tasks but escalates nearly everything and saves little against all-cloud [hybrid-savings-v1].

Inputs and their labels:

- All-cloud quality and USD per task, per task and seed: MEASURED, rows all-cloud-b3-quality and all-cloud-b3-cost
  (results/all_cloud_b3_v1.jsonl; claude-haiku-5-5 and claude-sonnet-5-5, prompt caching on).
- Local outcome per tier, task and seed (end_state_ok), qwen3:14b on evo-x2: MEASURED, row
  x2-evalset-tasks-real-v2-wrong-rate (results/x2_evalset_tasks_real_v2.jsonl). The best evo-x2 setting is the tier
  with the lowest local wrong rate on all 5 seeds, which is the default.
- Window-overflow flag per task: MEASURED, row impact-v3-detection-value-measured (same file).
- Escalation cost per escalated task: INTERIM, the all-cloud per-task cost of the same task at the same tier ("interim:
  all-cloud per-task cost, full-history redo pending"). It switches automatically to row cloud-redo-full-context once
  that row is VERIFIED and its run finished unstopped; that row is PENDING (the operator must run it). Row
  cloud-redo-cost-per-wrong-task and results/cloud_redo_v1.jsonl are NOT used: that run is a broken-protocol run (see
  the cloud redo diagnosis in docs/FINDINGS.md).
- evo-t2s Intel CPU at its 4096 default: PROJECTED. There is no B3 v2 run on evo-t2s, so evo-x2's ollama_ctx_4096 tier
  stands in for it (same model, same window, different backend); the loaded window is from row t2s-r2-real-cpu.
- Blade: not measured.
- Volume: ASSUMED parameter range of 100, 1,000 and 10,000 agent tasks/day, 30-day month; not a claim about demand.
- P2 hardware capacity: session minutes MEASURED (impact-v2-P2, from cost-cut-config-time); the speedup is KV
  precision, not flash attention (b5-fa-control); machine price an ASSUMPTION (impact-ref-machine-price); 3-year life
  ASSUMED; one agent task = one turn of a 40-turn R2 session ASSUMED for the per-volume machine counts.

<!-- hybrid_savings:begin -->

Generated by `py -3.12 analysis/impact_model.py --write-hybrid` (register row hybrid-savings-v1). Do not edit by hand.

Matched on the 3 seeds both runs share (20260901, 20260902, 20260903); the local run has 5 seeds and the other 2 are not used in the table. 90 tasks per cell. Escalated task cost: interim: all-cloud per-task cost, full-history redo pending. Monthly savings: 30-day month at 100, 1,000, 10,000 agent tasks/day (a stated parameter range, not a claim). Labels: measured = evo-x2 B3 v2 run; projected = evo-x2's ollama_ctx_4096 tier standing in for evo-t2s.

| Device (setting) [label] | Cloud tier | Policy | Correct | vs all-cloud | USD/task | Saves USD/task | Saves USD/month at 100 / 1,000 / 10,000 tasks/day | Escalated | Matched quality |
|---|---|---|---|---|---|---|---|---|---|
| evo-x2 (ollama_default, default = best) [measured] | claude-haiku-5-5 | all-cloud (reference) | 87/90 (96.7%) | +0.0 pts | 0.00067 | 0.00000 | 0.00 / 0.00 / 0.00 | 90 | yes |
| evo-x2 (ollama_default, default = best) [measured] | claude-haiku-5-5 | naive local | 80/90 (88.9%) | -7.8 pts | 0.00000 | 0.00067 | 2.01 / 20.12 / 201.25 | 0 | no |
| evo-x2 (ollama_default, default = best) [measured] | claude-haiku-5-5 | hybrid: escalate window-overflow flagged tasks | 80/90 (88.9%) | -7.8 pts | 0.00000 | 0.00067 | 2.01 / 20.12 / 201.25 | 0 | no |
| evo-x2 (ollama_default, default = best) [measured] | claude-haiku-5-5 | hybrid: escalate every wrong task (perfect detector, UPPER BOUND) | 90/90 (100.0%) | +3.3 pts | 0.00011 | 0.00056 | 1.67 / 16.70 / 167.03 | 10 | yes |
| evo-x2 (ollama_default, default = best) [measured] | claude-sonnet-5-5 | all-cloud (reference) | 90/90 (100.0%) | +0.0 pts | 0.01020 | 0.00000 | 0.00 / 0.00 / 0.00 | 90 | yes |
| evo-x2 (ollama_default, default = best) [measured] | claude-sonnet-5-5 | naive local | 80/90 (88.9%) | -11.1 pts | 0.00000 | 0.01020 | 30.61 / 306.10 / 3,060.99 | 0 | no |
| evo-x2 (ollama_default, default = best) [measured] | claude-sonnet-5-5 | hybrid: escalate window-overflow flagged tasks | 80/90 (88.9%) | -11.1 pts | 0.00000 | 0.01020 | 30.61 / 306.10 / 3,060.99 | 0 | no |
| evo-x2 (ollama_default, default = best) [measured] | claude-sonnet-5-5 | hybrid: escalate every wrong task (perfect detector, UPPER BOUND) | 90/90 (100.0%) | +0.0 pts | 0.00192 | 0.00829 | 24.86 / 248.57 / 2,485.72 | 10 | yes |
| evo-t2s Intel CPU (default, loaded window 4096) [projected] | claude-haiku-5-5 | all-cloud (reference) | 87/90 (96.7%) | +0.0 pts | 0.00067 | 0.00000 | 0.00 / 0.00 / 0.00 | 90 | yes |
| evo-t2s Intel CPU (default, loaded window 4096) [projected] | claude-haiku-5-5 | naive local | 53/90 (58.9%) | -37.8 pts | 0.00000 | 0.00067 | 2.01 / 20.12 / 201.25 | 0 | no |
| evo-t2s Intel CPU (default, loaded window 4096) [projected] | claude-haiku-5-5 | hybrid: escalate window-overflow flagged tasks | 87/90 (96.7%) | +0.0 pts | 0.00066 | 0.00001 | 0.04 / 0.44 / 4.41 | 88 | yes |
| evo-t2s Intel CPU (default, loaded window 4096) [projected] | claude-haiku-5-5 | hybrid: escalate every wrong task (perfect detector, UPPER BOUND) | 89/90 (98.9%) | +2.2 pts | 0.00032 | 0.00035 | 1.06 / 10.65 / 106.47 | 37 | yes |
| evo-t2s Intel CPU (default, loaded window 4096) [projected] | claude-sonnet-5-5 | all-cloud (reference) | 90/90 (100.0%) | +0.0 pts | 0.01020 | 0.00000 | 0.00 / 0.00 / 0.00 | 90 | yes |
| evo-t2s Intel CPU (default, loaded window 4096) [projected] | claude-sonnet-5-5 | naive local | 53/90 (58.9%) | -41.1 pts | 0.00000 | 0.01020 | 30.61 / 306.10 / 3,060.99 | 0 | no |
| evo-t2s Intel CPU (default, loaded window 4096) [projected] | claude-sonnet-5-5 | hybrid: escalate window-overflow flagged tasks | 90/90 (100.0%) | +0.0 pts | 0.00994 | 0.00026 | 0.79 / 7.90 / 79.03 | 88 | yes |
| evo-t2s Intel CPU (default, loaded window 4096) [projected] | claude-sonnet-5-5 | hybrid: escalate every wrong task (perfect detector, UPPER BOUND) | 90/90 (100.0%) | +0.0 pts | 0.00482 | 0.00538 | 16.14 / 161.38 / 1,613.84 | 37 | yes |
| Blade | n/a | all | not measured (no B3 v2 run on the Blade) | | | | | | |

Per cell: tasks kept local at matched quality (local correct and the all-cloud tier correct on the same task and seed), and what the window-overflow flag sees:

- evo-x2 (ollama_default) vs claude-haiku-5-5: kept local at matched quality 77/90; local wrong 10; flagged 0 (of them wrong 0); wrong tasks inside the window, missed by the flag: 10.
- evo-x2 (ollama_default) vs claude-sonnet-5-5: kept local at matched quality 80/90; local wrong 10; flagged 0 (of them wrong 0); wrong tasks inside the window, missed by the flag: 10.
- evo-t2s Intel CPU (ollama_ctx_4096) vs claude-haiku-5-5: kept local at matched quality 51/90; local wrong 37; flagged 88 (of them wrong 37); wrong tasks inside the window, missed by the flag: 0.
- evo-t2s Intel CPU (ollama_ctx_4096) vs claude-sonnet-5-5: kept local at matched quality 53/90; local wrong 37; flagged 88 (of them wrong 37); wrong tasks inside the window, missed by the flag: 0.

Local wrong rate per evo-x2 tier on all 5 seeds (picks the best tier): ollama_default 11.3%, ollama_ctx_16384 12.7%, ollama_ctx_8192 35.3%, ollama_ctx_4096 42.0%, ollama_ctx_4096_numkeep 30.0%; best = ollama_default.

P2 hardware capacity (impact-v2-P2). The speedup is KV precision, not flash attention (b5-fa-control). qwen3:14b default (a) 214.8 min per session vs (g) kv_q8_0 75.4 min (measured), one session at a time per machine.

| Agent tasks/day (ASSUMED 1 task = 1 turn of a 40-turn session) | Machines, default | Machines, best | Upfront hardware avoided, USD (ASSUMPTION price) | Amortized USD/day (secondary) |
|---|---|---|---|---|
| impact-v2-P2 profile | 6 | 3 | 6,000 | 5.48 |
| 100 | 1 | 1 | 0 | 0.00 |
| 1,000 | 4 | 2 | 4,000 | 3.65 |
| 10,000 | 38 | 14 | 48,000 | 43.84 |

Machine price USD 2,000 (impact-ref-machine-price: ASSUMPTION, must be verified); 3-year life (ASSUMED) for the amortized column.

<!-- hybrid_savings:end -->

## Rows registered in parallel (not on main when this was written)

Another plan item registers r2-session-wallclock, r2-session-cloud-cost, info-loss-by-window and
reliability-after-guardrail-loss. This model does not wait for them: the inputs it needs (session wall time, prefill
and decode seconds, output tokens, session cloud cost) are computed here by small helpers
(`session_wallclock`, `r2_cloud_overspend` in analysis/impact_model.py) from the same committed files. Where a row
cites one of those ids it is marked "PENDING on main". Once they land, the two computations should agree; if they do
not, the register rows win and this model is the one to fix.

## Inputs added later

- **2026-10-21: cost of redoing a silent wrong task on the cloud** (register row cloud-redo-cost-per-wrong-task,
  scripts/cloud_redo.py, results/cloud_redo_v1.jsonl). Every B3 v2 wrong task (qwen3:14b, all tiers) priced on the
  escalation model with prompt caching on. Mode: priced, not executed (no API key on the controller): token counts are
  the local run's, the output is a stand-in, and cloud correctness and seconds are not measured. Not yet wired into
  analysis/impact_model.py; it is a candidate per-event cost for P1 (the redo price, not the cost of a correct answer)
  once a live run measures whether the cloud answer is correct.

## Impact audit, 2026-10-14 (operator decisions A to C)

Before any impact number leaves the repo, the model was audited. What changed:

- **Prices (A).** Every cloud call is priced with the shared Anthropic table
  results/cloud_prices_anthropic_20261014.json, transcribed from Anthropic's official pricing page (URL, fetch time,
  model IDs, input / cache write / cache read / output per MTok are in the file; register row cloud-prices-anthropic).
  The escalation model is claude-sonnet-5-5, the middle price tier. Token counts stay o200k_base (or the R2 harness's
  local estimate for the Router replay); Anthropic's own tokenizer counts are not measured, so the token basis is an
  assumption, listed in the P3 row.
- **Caching is the default (B).** Every cloud figure is priced with provider prompt caching on: P3 is now the
  re-sent history at the cache-read price (main case), with the same tokens at the uncached input price as the upper
  bound only. The Router's escalated steps are priced with caching between escalated steps of a session (a hit only
  when the previous escalated step started within the 5-minute default cache lifetime in the recorded session).
- **Audit split (C).** Each row's USD/day is split into a measured part and an assumed part, with every assumption
  listed (value and source) and a status. Compute time is valued at the full hour value only as an upper bound; the
  lower bound is 0 (nobody waits for it), so every time row is "mixed" and contributes 0 to the measured-only total.
  P1 prints "not measured": no B3 task run (tool tasks with end-state checks) exists under two or more windows yet
  (results/x2_evalset_tasks_val_v1.jsonl is the validation gate run, not a measurement, and is excluded), and a silent
  rule or recall failure is not converted into a wrong task. impact-total is now a measured-only total plus a range.

How the ranking changed (status quo profile). Before the audit (register row impact-total as it stood before this
change, OpenAI table, P3 = uncached minus cached session cost, time at the full hour value): P3 > P5 > P6 > P2 > P4 >
P1 > P7, with P3 first at USD 67.89/day. After (caching on, time at its upper bound): see "Ranking" in the generated
block. P3 falls from first to fourth because with caching on the re-sent history is billed at the cache-read price;
P1 drops to last because it is no longer priced at all. The three time rows (P5, P6, P2) now lead, but each is
"mixed": their measured part is 0 and their whole value rests on the assumption that someone waits.

Where the earlier USD 1,933.63/day (USD 40,606.25/month) at num_ctx 4096 came from is laid out term by term in the
generated block ("Scenario"), recomputed from the same files (register row impact-scenario-one-flag): almost all of
it (USD 1,858.48/day) is P1 under the assumption that each of the 5/5 silent rule/recall failures is a wrong result
costing one full 0.72 h rerun at the full wage. It is a SCENARIO that depends on assumed rework time per silent wrong
result (not measured), and it is not a lower bound. Its USD/month form, USD 40,606.25, is the same figure.

No doc cites an impact number whose status is "assumed".

## Price table

This model prices every cloud call with the shared Anthropic table (above) through analysis/cloud_budget_forecast.py
`call_cost` / `session_cost`, which now read a per-model minimum cacheable prefix and long-context threshold from the
price file when it has them (the OpenAI table has neither, so the cloud-budget-* rows are unchanged).

Disagreement inside our own code, recorded here so it is fixed and not cited around: src/cloud/client.py still carries
its own older pricing table and model map, and src/dse/router.py's `est_cost_usd` and demo/dashboard/fakes.py use that
table. This model therefore takes only the Router's decisions (which steps escalate) and reprices them with the shared
table. Moving src/cloud/client.py onto the shared price file is a separate change made in parallel, not here.

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

- Price table: the demo must price cloud calls with results/cloud_prices_anthropic_20261014.json (same model, same
  Standard vs Batch tier, same cache-write and cache-read rules, caching on as the main case) or say which table it
  uses.
- Escalation definitions: the demo's "keeps a turn local" must mean no E1-E4 escalation in the sense of the
  generated block below (memory, fit, latency, quality, in that order, with the Router's CONTEXT_MARGIN on the
  window). His reported "share of turns kept local" is only comparable to this model if a turn that is trimmed locally and loses a needed turn
  counts as escalated (E4), not as local.
- Unit of cost: his demo reports USD per task and a total; this model reports USD per session and per day for a
  stated profile. The task definition (turns, tokens per turn) has to be stated to convert one into the other.

<!-- impact_model:begin -->

Generated by `analysis/impact_model.py --write`. Profile: hour value USD 64.44 (impact-hour-value); ASSUMPTIONS: team 10, 4 agent sessions per developer per day, 40 turns per session, 10 single-shot calls per developer per day, 21 workdays per month, co-runner share 0.25, cloud session share 0.5, configuration evo-x2 llama3.1:8b ollama_default. Cloud: claude-sonnet-5-5 (standard) from results/cloud_prices_anthropic_20261014.json, provider prompt caching ON (main case); no caching is the upper bound only. Compute time: lower bound 0 (nobody waits), upper bound at the full hour value (someone waits).

### Impact audit (status quo: Ollama default window)

Status: measured = USD follows from measured per-session quantities and cited prices at the profile volumes; mixed = a measured quantity whose USD also rests on an assumption beyond volume (who waits, co-runner share); assumed = the per-event cost itself is not measured; not measured = no number. Every USD/day is a projection: it scales with the assumed team size and sessions per day.

| row | problem | measured USD/day | assumed USD/day | range USD/day | status | assumptions (value; source) | register row |
|---|---|---:|---:|---|---|---|---|
| P1 | silent wrong results from context loss | not measured | 0.00 | not measured | assumed | team size = 10 (assumption (profile JSON)); agent sessions per developer per day = 4 (assumption (profile JSON)); a silent rule/recall failure is a wrong task result = 0/5 sessions treated as wrong (not measured: B3 task runs under >= 2 windows needed; found none (validation files excluded)); rework per wrong result = one full session rerun = median session wall at the default window (0.72 h) (assumption (not measured)); developer waits for this compute = upper bound values it at the full hour value USD 64.44/h (impact-hour-value); lower bound 0 (nobody waits) (assumption: who waits was not measured) | impact-P1 |
| P2 | oversized-window time cost | 0.00 | 23.27 | 0.00 to 23.27 | mixed | team size = 10 (assumption (profile JSON)); agent sessions per developer per day = 4 (assumption (profile JSON)); turns per session = 40 (R2 session length (scales a per-turn delta)); developer waits for this compute = upper bound values it at the full hour value USD 64.44/h (impact-hour-value); lower bound 0 (nobody waits) (assumption: who waits was not measured) | impact-P2 |
| P3 | cloud overspend from re-sent history | 3.60 | 0.00 | 3.60 to 71.96 | measured | team size = 10 (assumption (profile JSON)); agent sessions per developer per day = 4 (assumption (profile JSON)); cloud session share = 0.5 (assumption (profile JSON)); cloud model = claude-sonnet-5-5 standard, caching on (results/cloud_prices_anthropic_20261014.json (official Anthropic pricing page)); token basis = o200k_base counts (assumption: Anthropic tokenizer counts not measured); re-sent history is a cacheable prefix = R2 sessions are append-only (best case for caching; real histories not measured) | impact-P3 |
| P4 | hardware mismatch | 0.00 | 0.87 | 0.00 to 0.87 | mixed | team size = 10 (assumption (profile JSON)); agent sessions per developer per day = 4 (assumption (profile JSON)); decode only = prefill excluded (lower bound on the time itself); developer waits for this compute = upper bound values it at the full hour value USD 64.44/h (impact-hour-value); lower bound 0 (nobody waits) (assumption: who waits was not measured) | impact-P4 |
| P5 | co-runner slowdown | 0.00 | 49.34 | 0.00 to 49.34 | mixed | team size = 10 (assumption (profile JSON)); agent sessions per developer per day = 4 (assumption (profile JSON)); co-runner share = 0.25 (assumption (profile JSON)); developer waits for this compute = upper bound values it at the full hour value USD 64.44/h (impact-hour-value); lower bound 0 (nobody waits) (assumption: who waits was not measured) | impact-P5 |
| P6 | failed starts and retries | 0.00 | 47.38 | 0.00 to 47.38 | mixed | team size = 10 (assumption (profile JSON)); agent sessions per developer per day = 4 (assumption (profile JSON)); single-shot calls per developer per day = 10 (assumption (profile JSON)); developer waits for this compute = upper bound values it at the full hour value USD 64.44/h (impact-hour-value); lower bound 0 (nobody waits) (assumption: who waits was not measured) | impact-P6 |
| P7 | guardrail removal | 0.00 | 0.00 | 0.00 | measured | team size = 10 (assumption (profile JSON)); agent sessions per developer per day = 4 (assumption (profile JSON)); cost per guardrail violation = not needed: measured rate 0/5 sessions (measured zero rate) | impact-P7 |

impact-total: measured-only total USD/day 3.60 (USD/month 75.55); range USD/day 3.60 to 192.81 (USD/month 75.55 to 4,049.08) over P2, P3, P4, P5, P6, P7; not added (assumed or not measured): P1.

### Ranking (status quo profile, caching on, time at its upper bound)

1. P5 co-runner slowdown: USD/day 49.34 (range 0.00 to 49.34, mixed); confidence low (co-runner ratios come from single-shot microbenchmarks applied to session timings, and the co-runner share is an assumption)
2. P6 failed starts and retries: USD/day 47.38 (range 0.00 to 47.38, mixed); confidence medium (rates are measured on the workload pack; the daily volume of single-shot calls is an assumption)
3. P2 oversized-window time cost: USD/day 23.27 (range 0.00 to 23.27, mixed); confidence medium (paired same-seed turns that fit both windows isolate the window size, but the per-turn delta is scaled to a whole session by assumption and the right-sized window was not run)
4. P3 cloud overspend from re-sent history: USD/day 3.60 (range 3.60 to 71.96, measured); confidence high (token counts and the official price page only; the daily figure scales linearly with the assumed cloud share)
5. P4 hardware mismatch: USD/day 0.87 (range 0.00 to 0.87, mixed); confidence medium (decode rates are measured per machine, but on different workloads (the Blade's from short dry runs), and prefill is excluded, so it is a lower bound)
6. P7 guardrail removal: USD/day 0.00 (range 0.00, measured); confidence low (the rate is measured but the cost per violation is not; a nonzero rate prints not measured)
7. P1 silent wrong results from context loss: USD/day not measured (range not measured, assumed); confidence none (USD not measured: no B3 task run yet, so no wrong-task rate; the R2 silent rule/recall rate (n in the grid) is not a wrong-task rate, and rework time is not measured)

Single most decisive missing measurement: Two inputs P1 lacks: (1) a wrong-task rate per window, from B3 task runs with end-state checks under at least two window settings (not yet run), and (2) developer time lost per wrong result (time to notice plus rework, and how often it is never noticed). P7 lacks the cost of one guardrail violation. At any small window the silent rule/recall rates are high (R2-real-v1b-gated-kill), so these inputs decide whether context loss ranks first or last; until then it is not ranked.

### Scenario: the same team after one flag, `ollama_ctx_4096` (register row impact-scenario-one-flag)

SCENARIO, depends on assumed rework time per silent wrong result (not measured). Not a lower bound.

| row | problem | measured USD/day | assumed USD/day | range USD/day | status |
|---|---|---:|---:|---|---|
| P1 | silent wrong results from context loss | not measured | 1,858.48 | not measured | assumed |
| P2 | oversized-window time cost | 0.00 | 0.00 | 0.00 | mixed |
| P3 | cloud overspend from re-sent history | 3.60 | 0.00 | 3.60 to 71.96 | measured |
| P4 | hardware mismatch | 0.00 | 0.87 | 0.00 to 0.87 | mixed |
| P5 | co-runner slowdown | 0.00 | 6.40 | 0.00 to 6.40 | mixed |
| P6 | failed starts and retries | not measured | not measured | not measured | not measured |
| P7 | guardrail removal | not measured | not measured | not measured | not measured |

Measured-only USD/day 3.60; range 3.60 to 79.22 over P2, P3, P4, P5; not added: P1, P6, P7.

Where the earlier USD 1,933.63/day (USD 40,606.25/month) came from (first impact model, before the 2026-10-14 audit; recomputed here from the same files):

| term | USD/day | inputs | status under the audit |
|---|---:|---|---|
| P1 | 1,858.48 | 5/5 sessions with a silent rule/recall failure after the window, each treated as a wrong result, x 40 sessions/day x one full rerun 0.7210 h (median session wall at the default window) x USD 64.44/h | assumed (rework per wrong result not measured; rule breaks are not wrong tasks) |
| P2 | 0.00 | smallest measured window | measured |
| P3 | 67.89 | gpt-6.1-sol (results/cloud_prices_20261008.json): uncached minus cached USD 3.3944/session x 40 sessions/day x cloud share 0.5 | replaced: caching on is now the main case (P3 row above) |
| P4 | 0.87 | decode-rate gap at the full hour value | mixed (time) |
| P5 | 6.40 | co-runner slowdown at the full hour value, co-runner share 0.25 | mixed (time) |
| not included | | P6, P7 not measured at this window | |

### Rows: formula, measured inputs, missing inputs, fix, escalation

| row | formula | measured inputs (register ids) | missing inputs | fix | needs cloud escalation? |
|---|---|---|---|---|---|
| P1 | NOT MEASURED until B3 task runs (end-state checked tool tasks) exist under >= 2 windows; then sessions/day x P(wrong task end state after the window is exceeded) x rework hours x USD/h. Scenario only (ASSUMED): the first model's sessions/day x P(gated silent rule/recall failure at/after the first over-window turn) x one full session rerun (median session wall at Ollama's default window) x USD/h | R2-real-v1b-gated-kill, R2-real-v1b-survival, R2-mechanism-verdict, x2-v3-truncation-all-models, r2-session-wallclock (PENDING on main; same files read here), info-loss-by-window (PENDING on main) | developer time to notice a silent wrong result and redo it, and the cost of one that is not noticed (not measured; the redo-time lower bound assumes the developer waits for one rerun); R2 sessions on evo-t2s and the Blade | client-side trimming of whole oldest turns to the usable window (never send an overflowing request), escalate on E4 | yes when a needed turn would be trimmed (E4); cost = Router replay, see the escalation table |
| P2 | sessions/day x max(0, median per-turn wall delta between the configured window and the next smaller measured window, on turns that fit both) x turns/session x USD/h | r2-session-wallclock (PENDING on main; same files read here), R2-real-v1-loaded-ctx, R2-real-v1b-first-events | a right-sized window for a full R2 session (every smaller R2 tier is outgrown before the last turn, R2-real-v1b-first-events); session timings on evo-t2s and the Blade | set num_ctx to the session's need, not the runtime default; combined with the P1 fix so a smaller window cannot silently cut history | no for the window change itself; E4 escalations of the P1 fix apply at the smaller window |
| P3 | sessions/day x cloud share x mean(re-sent history tokens per session x cache-read price) with provider caching on (main case); the same tokens x uncached input price is the no-caching upper bound; shared Anthropic table cloud_prices_anthropic_20261014 (claude-sonnet-5-5), R2 session token counts (o200k_base) | cloud-budget-tokens, cloud-budget-inputs, r2-session-cloud-cost (PENDING on main; same computation here) | how much of a real customer's cloud history is a stable cacheable prefix (R2 sessions are append-only, the best case for caching); Anthropic tokenizer counts (o200k_base used) | keep the history an append-only prefix and turn provider caching on; trim history client-side | no (the fix lowers cloud spend) |
| P4 | sessions/day x output tokens/session x (1/decode rate on this device - 1/best measured rate for the model) x USD/h; decode only (prefill excluded), lower bound | decode-rate-per-machine, blade-dryrun-rates, blade-k1-layer-split, r2-session-wallclock (output tokens; PENDING on main) | session prefill time on evo-t2s and the Blade; the XPS (Zach's) rates are not in this repo | route each model to the device whose envelope fits it (src/dse/pareto.py recommend) | no between local devices; E1/E2 only when no local device fits |
| P5 | sessions/day x co-runner share x (session prefill s x (TTFT ratio - 1) + session decode s x (1/decode ratio - 1)) x USD/h | PX2-full-ratio-table, B3-corunner-6model, PX2-decode-ratio, r2-session-wallclock (PENDING on main) | R2 session timings on evo-t2s (B3 is TTFT only); how often customers run a heavy co-runner (assumed share) | co-runner-aware latency prediction in the router (co_runner flag) | only under a latency target (E3); cost depends on the customer's target, not modelled |
| P6 | single-shot calls/day x (P(timeout) x per-call timeout s + P(allocation failure) x seconds to fail) x USD/h | x2-v3-usability, t2s-outcome-allocation-failures, t2s-llamaserver-load-failures, A-24-flip-point-prediction | operator time to notice and restart a stuck runtime (not measured); llama-server start failures need the gitignored logs (register row only) | pre-check the memory fit (A-24 shared-heap model) and the latency before sending; route the rest | yes for calls that cannot fit or finish locally (E1/E3); cost = P(event) x price of those items |
| P7 | sessions/day x P(a rule in use first broken at/after the first over-window turn) x cost per violation | R2-real-v1b-first-events, R2-mechanism-lowlevel, R2-orphan-rates, R2-mitigation-v1-verdict, reliability-after-guardrail-loss (PENDING on main) | cost of one guardrail violation (not measured; depends on what the rule protects) | never send an overflowing request (the system prompt is always kept by client-side trimming) | no for system-prompt rules (the router never trims the system prompt) |

### Escalation definitions (src/dse/router.py Router.decide, in decision order)

- E1: memory: a measured boundary says the local configuration will not start (A-24 last n_ctx that started) -> cloud if the budget covers it, else refuse
- E2: fit: system prompt + current turn + answer budget exceed the usable window (num_ctx x (1 - CONTEXT_MARGIN)) even after trimming every older turn -> cloud, else refuse
- E3: latency: the request fits untrimmed but the predicted latency is over the caller's target -> cloud if affordable
- E4: quality: trimming whole oldest turns would drop a turn the step needs (needs_turns or an identifier in the current user text) and quality_floor > 0 (or require_full_history) -> cloud, else local_trimmed with quality_floor_met=False

### Escalation cost of the P1 fix (Router replay, E1-E4, claude-sonnet-5-5)

Caching on: an escalated step reads the previous escalated step's prompt from cache when that step started at most 300 s earlier in the recorded local session (the default cache lifetime). Output is priced at the step's full answer budget, an upper bound.

| model | window | cloud steps / session | steps / session | USD / session caching on | USD / session no caching | cache-hit share | sessions |
|---|---|---:|---:|---:|---:|---:|---:|
| llama3.1:8b | num_ctx_4096 | 16.0 | 80.0 | 0.2561 | 1.1218 | 94% | 5 |
| llama3.1:8b | ollama_default | 0.0 | 80.0 | 0.0000 | 0.0000 | n/a | 5 |
| qwen3:14b | num_ctx_4096 | 16.0 | 80.0 | 0.2802 | 1.2543 | 94% | 5 |
| qwen3:14b | ollama_default | 8.0 | 80.0 | 0.5904 | 0.8915 | 50% | 5 |
| qwen3:8b | num_ctx_4096 | 15.6 | 79.4 | 0.2863 | 1.2741 | 94% | 5 |
| qwen3:8b | ollama_default | 8.0 | 80.0 | 0.6079 | 0.9175 | 50% | 5 |

### Per model and device grid (USD/day at the profile volumes; time rows 0 to the full hour value)

| row | model@device/window | USD/day | inputs |
|---|---|---:|---|
| P1 | llama3.1:8b@num_ctx_16384 | not measured | 2/5 sessions silent rule/recall failure after the window; scenario (ASSUMED rerun 0.72 h): USD 743.39 |
| P1 | llama3.1:8b@num_ctx_32768 | not measured | 0/5 sessions silent rule/recall failure after the window; scenario (ASSUMED rerun 0.72 h): USD 0.00 |
| P1 | llama3.1:8b@num_ctx_4096 | not measured | 5/5 sessions silent rule/recall failure after the window; scenario (ASSUMED rerun 0.72 h): USD 1,858.48 |
| P1 | llama3.1:8b@num_ctx_8192 | not measured | 4/5 sessions silent rule/recall failure after the window; scenario (ASSUMED rerun 0.72 h): USD 1,486.78 |
| P1 | llama3.1:8b@ollama_default | not measured | 0/5 sessions silent rule/recall failure after the window; scenario (ASSUMED rerun 0.72 h): USD 0.00 |
| P1 | qwen3:14b@num_ctx_16384 | not measured | 0/5 sessions silent rule/recall failure after the window; scenario (ASSUMED rerun 3.37 h): USD 0.00 |
| P1 | qwen3:14b@num_ctx_32768 | not measured | 0/5 sessions silent rule/recall failure after the window; scenario (ASSUMED rerun 3.37 h): USD 0.00 |
| P1 | qwen3:14b@num_ctx_4096 | not measured | 4/5 sessions silent rule/recall failure after the window; scenario (ASSUMED rerun 3.37 h): USD 6,953.80 |
| P1 | qwen3:14b@num_ctx_8192 | not measured | 0/5 sessions silent rule/recall failure after the window; scenario (ASSUMED rerun 3.37 h): USD 0.00 |
| P1 | qwen3:14b@ollama_default | not measured | 0/5 sessions silent rule/recall failure after the window; scenario (ASSUMED rerun 3.37 h): USD 0.00 |
| P1 | qwen3:8b@num_ctx_16384 | not measured | 0/5 sessions silent rule/recall failure after the window; scenario (ASSUMED rerun 1.19 h): USD 0.00 |
| P1 | qwen3:8b@num_ctx_32768 | not measured | 0/5 sessions silent rule/recall failure after the window; scenario (ASSUMED rerun 1.19 h): USD 0.00 |
| P1 | qwen3:8b@num_ctx_4096 | not measured | 1/5 sessions silent rule/recall failure after the window; scenario (ASSUMED rerun 1.19 h): USD 612.59 |
| P1 | qwen3:8b@num_ctx_8192 | not measured | 4/5 sessions silent rule/recall failure after the window; scenario (ASSUMED rerun 1.19 h): USD 2,450.37 |
| P1 | qwen3:8b@ollama_default | not measured | 0/5 sessions silent rule/recall failure after the window; scenario (ASSUMED rerun 1.19 h): USD 0.00 |
| P2 | llama3.1:8b@num_ctx_16384 | 0.00 to 0.00 | vs num_ctx_8192: -0.09 s/turn (n=25) |
| P2 | llama3.1:8b@num_ctx_32768 | 0.00 to 2.23 | vs num_ctx_16384: 0.08 s/turn (n=55) |
| P2 | llama3.1:8b@num_ctx_4096 | 0.00 to 0.00 | smallest measured window |
| P2 | llama3.1:8b@num_ctx_8192 | 0.00 to 0.00 | vs num_ctx_4096: -0.27 s/turn (n=10) |
| P2 | llama3.1:8b@ollama_default | 0.00 to 23.27 | vs num_ctx_32768: 0.81 s/turn (n=110) |
| P2 | qwen3:14b@num_ctx_16384 | 0.00 to 8.28 | vs num_ctx_8192: 0.29 s/turn (n=20) |
| P2 | qwen3:14b@num_ctx_32768 | 0.00 to 0.00 | vs num_ctx_16384: -0.11 s/turn (n=45) |
| P2 | qwen3:14b@num_ctx_4096 | 0.00 to 0.00 | smallest measured window |
| P2 | qwen3:14b@num_ctx_8192 | 0.00 to 0.00 | vs num_ctx_4096: -0.26 s/turn (n=10) |
| P2 | qwen3:14b@ollama_default | 0.00 to 81.42 | vs num_ctx_32768: 2.84 s/turn (n=97) |
| P2 | qwen3:8b@num_ctx_16384 | 0.00 to 1.55 | vs num_ctx_8192: 0.05 s/turn (n=20) |
| P2 | qwen3:8b@num_ctx_32768 | 0.00 to 0.00 | vs num_ctx_16384: -0.02 s/turn (n=45) |
| P2 | qwen3:8b@num_ctx_4096 | 0.00 to 0.00 | smallest measured window |
| P2 | qwen3:8b@num_ctx_8192 | 0.00 to 0.00 | vs num_ctx_4096: -0.01 s/turn (n=10) |
| P2 | qwen3:8b@ollama_default | 0.00 to 32.66 | vs num_ctx_32768: 1.14 s/turn (n=94) |
| P3 | llama3.1:8b | 3.60 to 71.96 | re-sent 1,798,884 tok/session: USD 0.1799 caching on, 3.5978 no caching; whole session 0.3358 / 3.7302 (n=5) |
| P3 | qwen3:14b | 3.70 to 74.10 | re-sent 1,852,395 tok/session: USD 0.1852 caching on, 3.7048 no caching; whole session 0.3634 / 3.8592 (n=5) |
| P3 | qwen3:8b | 3.81 to 76.13 | re-sent 1,903,321 tok/session: USD 0.1903 caching on, 3.8066 no caching; whole session 0.3933 / 3.9853 (n=5) |
| P4 | llama-33-70b@evo-t2s | not measured | 2.32 tok/s vs best 6.42; output n/a (no R2 session) tok/session |
| P4 | llama-33-70b@evo-x2 | not measured | 6.42 tok/s vs best 6.42; output n/a (no R2 session) tok/session |
| P4 | llama31-8b@blade@32768 | 0.00 to 113.08 | 13.58 tok/s vs best 46.40; output 3032 tok/session |
| P4 | llama31-8b@blade@4096 | 0.00 to 0.00 | 46.40 tok/s vs best 46.40; output 3032 tok/session |
| P4 | llama31-8b@evo-t2s | 0.00 to 90.48 | 15.81 tok/s vs best 46.40; output 3032 tok/session |
| P4 | llama31-8b@evo-x2 | 0.00 to 0.87 | 45.55 tok/s vs best 46.40; output 3032 tok/session |
| P4 | qwen3-14b@evo-t2s | 0.00 to 175.56 | 11.69 tok/s vs best 27.43; output 4996 tok/session |
| P4 | qwen3-14b@evo-x2 | 0.00 to 0.00 | 27.43 tok/s vs best 27.43; output 4996 tok/session |
| P4 | qwen3-30b-a3b-2507@evo-t2s | not measured | 31.74 tok/s vs best 85.84; output n/a (no R2 session) tok/session |
| P4 | qwen3-30b-a3b-2507@evo-x2 | not measured | 85.84 tok/s vs best 85.84; output n/a (no R2 session) tok/session |
| P4 | qwen3-32b@evo-t2s | not measured | 4.57 tok/s vs best 12.43; output n/a (no R2 session) tok/session |
| P4 | qwen3-32b@evo-x2 | not measured | 12.43 tok/s vs best 12.43; output n/a (no R2 session) tok/session |
| P4 | qwen3-4b-2507@evo-t2s | not measured | 16.04 tok/s vs best 77.07; output n/a (no R2 session) tok/session |
| P4 | qwen3-4b-2507@evo-x2 | not measured | 77.07 tok/s vs best 77.07; output n/a (no R2 session) tok/session |
| P4 | qwen3-8b@evo-t2s | 0.00 to 148.72 | 19.83 tok/s vs best 50.11; output 6819 tok/session |
| P4 | qwen3-8b@evo-x2 | 0.00 to 0.00 | 50.11 tok/s vs best 50.11; output 6819 tok/session |
| P5 | llama3.1:8b@evo-x2@num_ctx_16384 | 0.00 to 19.82 | TTFT x1.111 |
| P5 | llama3.1:8b@evo-x2@num_ctx_32768 | 0.00 to 25.81 | TTFT x1.111 |
| P5 | llama3.1:8b@evo-x2@num_ctx_4096 | 0.00 to 6.40 | TTFT x1.111 |
| P5 | llama3.1:8b@evo-x2@num_ctx_8192 | 0.00 to 11.10 | TTFT x1.111 |
| P5 | llama3.1:8b@evo-x2@ollama_default | 0.00 to 49.34 | TTFT x1.111 |
| P5 | qwen3:14b@evo-x2@num_ctx_16384 | 0.00 to 46.09 | TTFT x1.100 |
| P5 | qwen3:14b@evo-x2@num_ctx_32768 | 0.00 to 104.52 | TTFT x1.100 |
| P5 | qwen3:14b@evo-x2@num_ctx_4096 | 0.00 to 11.04 | TTFT x1.100 |
| P5 | qwen3:14b@evo-x2@num_ctx_8192 | 0.00 to 22.73 | TTFT x1.100 |
| P5 | qwen3:14b@evo-x2@ollama_default | 0.00 to 218.75 | TTFT x1.100 |
| P5 | qwen3:8b@evo-x2@num_ctx_16384 | 0.00 to 26.20 | TTFT x1.123 |
| P5 | qwen3:8b@evo-x2@num_ctx_32768 | 0.00 to 46.84 | TTFT x1.123 |
| P5 | qwen3:8b@evo-x2@num_ctx_4096 | 0.00 to 8.63 | TTFT x1.123 |
| P5 | qwen3:8b@evo-x2@num_ctx_8192 | 0.00 to 14.44 | TTFT x1.123 |
| P5 | qwen3:8b@evo-x2@ollama_default | 0.00 to 89.50 | TTFT x1.123 |
| P5 | llama3.1:8b@evo-t2s@ollama_default | not measured | no R2 session timings on evo-t2s |
| P6 | llama31-8b@evo-t2s@ollama_default | 0.00 to 0.47 | timeouts 0, allocation failures 18 of 40; escalation USD 1.29/day |
| P6 | llama31-8b@evo-t2s@ollama_igpu_enable | 0.00 to 1.94 | timeouts 0, allocation failures 19 of 40; escalation USD 1.41/day |
| P6 | llama31-8b-gguf@evo-t2s@llama_server_vulkan | 0.00 to 0.00 | timeouts 0, allocation failures 0 of 20; escalation USD 0.00/day |
| P6 | llama31-8b@evo-x2@llama_server | 0.00 to 189.53 | timeouts 40, allocation failures 0 of 340; escalation USD 2.12/day |
| P6 | llama31-8b@evo-x2@ollama_default | 0.00 to 47.38 | timeouts 10, allocation failures 0 of 340; escalation USD 0.71/day |
| P6 | qwen3-14b@evo-x2@llama_server | 0.00 to 0.00 | timeouts 0, allocation failures 0 of 340; escalation USD 0.00/day |
| P6 | qwen3-14b@evo-x2@ollama_default | 0.00 to 0.00 | timeouts 0, allocation failures 0 of 340; escalation USD 0.00/day |
| P6 | qwen3-30b-a3b@evo-x2@llama_server | 0.00 to 47.38 | timeouts 10, allocation failures 0 of 340; escalation USD 0.71/day |
| P6 | qwen3-30b-a3b@evo-x2@ollama_default | 0.00 to 0.00 | timeouts 0, allocation failures 0 of 340; escalation USD 0.00/day |
| P6 | qwen3-32b@evo-x2@llama_server | 0.00 to 177.21 | timeouts 11, allocation failures 0 of 100; escalation USD 0.70/day |
| P6 | qwen3-32b@evo-x2@ollama_default | 0.00 to 0.00 | timeouts 0, allocation failures 0 of 100; escalation USD 0.00/day |
| P6 | qwen3-4b-2507@evo-x2@llama_server | 0.00 to 189.53 | timeouts 40, allocation failures 0 of 340; escalation USD 2.12/day |
| P6 | qwen3-4b-2507@evo-x2@ollama_default | 0.00 to 47.38 | timeouts 10, allocation failures 0 of 340; escalation USD 0.71/day |
| P6 | qwen3-8b@evo-x2@llama_server | 0.00 to 0.00 | timeouts 0, allocation failures 0 of 340; escalation USD 0.00/day |
| P6 | qwen3-8b@evo-x2@ollama_default | 0.00 to 0.00 | timeouts 0, allocation failures 0 of 340; escalation USD 0.00/day |
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
