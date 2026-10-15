# evo-x2 queue plan (2026-10-15)

Operator decisions of 2026-10-14 (the first set at about 02:45Z on 2026-10-15, revised at 21:40 PT = 04:40Z), applied
by the queue steward. Hours come from committed result files where a measured rate exists; every other figure is
labelled estimated and says what it is scaled from. All times UTC.

## Applied order (operator decision of 21:40 PT)

Pending items were reordered in read-modify-writes of `C:\apu\ovn\queue_state.json` under `t2s_queue.set_pause`
(each reason names its release: `clear_pause()` in the same script's finally block); the pause flag was confirmed
absent after each. Only pending items moved; finished entries were not touched.

1. `x2_num_keep_guardrail_test` (A4): done, 04:44 to 05:07Z, note "completed" (its own advance() worked)
2. `x2_r2_llama_val_32k_offload` (new; another agent inserts it here)
3. `x2_evalset_tasks_real_v1` (new B3 task real runs; another agent inserts it here)
4. `x2_model_pulls_scale` (llama3.3:70b and mistral-small3.2:24b library pulls), running since 05:07Z
5. `x2_family_smoke_mistral_small32` (B2)
6. `x2_scale_timing_probe` (B1)
7. `x2_r2_scale_validation` (B1)
8. `x2_r2_scale_real` (B1)
9. `x2_r2_family_mistral_small32_validation` (B2)
10. `x2_r2_family_mistral_small32_real` (B2)
11. `x2_evalset_replay_val_v1` (B3 validation, ahead of the B3 real runs that read it)
12. B3 guardrail and replay real runs (not built yet)
13. B5 `x2_policy_compare` (being built)
14. `x2_window_override_methods` (last unless told otherwise)

Items other agents add later should follow this order. At the last check (05:10Z) items 2 and 3 were not yet in the
queue. num_keep finished first, so the pulls launched; items 2 and 3 run right after the pulls once they are inserted
ahead of the smoke test. Every item present matched this order.

## Finished jobs

| job | queue status | queue note | result file | register rows |
|---|---|---|---|---|
| x2_r2_real_v1b | done, 2026-10-10 13:25 to 2026-10-11 23:42 | completed; refused models (validation gates): qwen3-4b-2507-tools (failed) | results/x2_r2_real_v1b.jsonl | R2-real-v1b-kill-criterion, -gated-kill, -first-events, -survival |
| x2_r2_mitigation_v1 | done, 2026-10-11 23:53 to 2026-10-12 02:15 | completed | results/x2_r2_mitigation_v1.jsonl | R2-mitigation-v1-verdict, -4096, -8192, -observed |
| x2_evalset_guard_val_v1 | done, 2026-10-14 14:13:40 to 2026-10-15 04:40:34 (run_end) | finished before the 08:45Z stop deadline; not stopped, no requeue copy | results/x2_evalset_guard_val_v1.jsonl (on evo-x2; next results sync) | none yet |

**x2_evalset_guard_val_v1 closeout.** All 8 sessions completed (3 x 40-turn negative control and 1 x 15-turn positive
control per model) and the job wrote run_end and its report. Its own `advance()` then refused to act ("this process is
not a descendant of pid 23416", the WMI wrapper), so the queue still showed it running with no process left. The
steward confirmed that no python or Ollama process remained, marked it done with that note, and launched the next item
with `t2s_queue.launch_next` under one pause. Without that, the watchdog would have marked it crashed after the
30-minute heartbeat timeout and requeued it.

**Validation gates, from the job's own log and report (not yet a register row).** In both evalset validations
(`x2_evalset_tasks_val_v1.jsonl.report.json` committed; `x2_evalset_guard_val_v1.jsonl.report.json` on evo-x2), the
baseline, negative-control and task-tool gates fail for llama3.1:8b and qwen3:14b. In the guardrail validation, rules
6, 7 and 8 also fail for both models. Positive control, content non-empty and thinking off pass. A real run that
preflights on these files (`validation_preflight_ext`) is therefore expected to refuse both models. That includes
`x2_evalset_tasks_real_v1` and the B3 guardrail real run. Such a refusal costs minutes, not the hours in the table
below. This is for the operator to decide on; the queue was not changed because of it.

## Measured rates used

Per-session wall clock is the heartbeat record to the `r2a_session` record in the committed files.

| source file | model | arm | sessions | h per session |
|---|---|---|---|---|
| x2_r2_real_v1 + v1b | llama3.1:8b | num_ctx 4096 / 8192 / 16384 / 32768 / default | 5 each (8192, 16384 v1b only) | 0.11 to 0.12 / 0.18 / 0.31 / 0.38 to 0.39 / 0.71 to 0.72 |
| x2_r2_real_v1 + v1b | qwen3:14b | num_ctx 4096 / 8192 / 16384 / 32768 / default | 5 each (8192, 16384 v1b only) | 0.22 / 0.40 / 0.76 / 1.53 to 1.75 / 3.30 to 3.52 |
| x2_r2_real_v1b | qwen3:8b | num_ctx 4096 / 8192 / 16384 / 32768 / default | 5 each | 0.18 / 0.26 / 0.40 / 0.66 / 1.19 |
| x2_r2_mitigation_v1 | llama3.1:8b / qwen3:14b | num_ctx 4096, client trim | 3 each | 0.10 / 0.17 |
| x2_r2_validation_v2 | llama3.1:8b / qwen3:14b | whole validation (3 arms, 7 sessions) | 7 each | 0.16 / 0.38 total |
| x2_r2_llama_val_32k | llama3.1:8b | whole job | 9 | 0.19 file wall clock |
| x2_evalset_tasks_val_v1 | both | whole job, 30-turn sessions | 8 | 7.00 file wall clock; neg control 0.25 (llama3.1:8b), 2.03 (qwen3:14b) |
| x2_evalset_guard_val_v1 | both | whole job, 40-turn sessions | 8 | 14.45 queue wall clock; neg control 0.83 to 0.89 (llama3.1:8b, 75 to 80 s per turn), 3.77 to 3.97 (qwen3:14b, 249 to 357 s per turn) |
| x2_model_pulls | qwen3:32b pull | 20.2 GB | 1 | 26.7 min, about 12.6 MB/s |

The timing probe has not run yet. Its `scale_projection` record replaces every estimated H-scale and family figure
below once synced (register row `x2-scale-probe-hours`).

## Hours table (through 2026-11-10)

Back to back from the pulls' launch at 2026-10-15 05:07Z, in the order the jobs will actually run (items 2 and 3 after
the pulls, see above); queue handoff gaps (seconds through `advance()`, up to 10 minutes through the watchdog) are
ignored. Rows marked "if it runs" assume the job passes its own preflight.

| job | basis | hours | projected start | projected finish |
|---|---|---|---|---|
| x2_num_keep_guardrail_test | measured (actual queue wall clock; projected 0.5 h from llama3.1:8b num_ctx 4096 sessions) | 0.38 | 10-15 04:44 | 10-15 05:07 (done) |
| x2_model_pulls_scale (running) | measured pull rate 12.6 MB/s x about 42.5 GB (llama3.3:70b Q4_K_M size on disk) + about 15 GB (mistral-small3.2:24b, estimated), plus hash checks | 1.5 | 10-15 05:07 | 10-15 06:37 |
| x2_r2_llama_val_32k_offload (not yet queued) | estimated: its sibling x2_r2_llama_val_32k took 0.19 h (measured) | 0.2 | 10-15 06:37 | 10-15 06:49 |
| x2_evalset_tasks_real_v1 (not yet queued; if it runs, see the gate note) | estimated: v1b per-session hours summed over the 5 R2 tiers, scaled by the measured tasks-validation to R2-default ratio (llama3.1:8b 0.35, qwen3:14b 0.59), x 3 seeds, both models | 13.3 | 10-15 06:49 | 10-15 20:07 |
| x2_family_smoke_mistral_small32 | estimated: 5 early turns of a 24B model at num_ctx 32768 | 0.2 | 10-15 20:07 | 10-15 20:19 |
| x2_scale_timing_probe | estimated: 3 models x 3 tiers x 2-turn sessions, load dominated | 1.0 | 10-15 20:19 | 10-15 21:19 |
| x2_r2_scale_validation | estimated: measured validation_v2 totals (qwen3:14b 0.38 h, llama3.1:8b 0.16 h) x the operator's per-session size ratios (qwen3:32b 2.0x, llama3.3:70b 18x); qwen3:14b's total as proxy for qwen3-30b-a3b-2507 | 4.1 | 10-15 21:19 | 10-16 01:25 |
| x2_r2_scale_real | estimated: operator per-session estimates (qwen3:32b 26 / 210 / 420 min at 4096 / 32768 / default: 32.8 h; llama3.3:70b 50 / 420 min at 4096 / 32768: 23.5 h, default skipped unless the probe projects at most 24 h) + qwen3:14b measured sessions as proxy for qwen3-30b-a3b-2507 (15.9 h) | 72.2 | 10-16 01:25 | 10-19 01:37 |
| x2_r2_family_mistral_small32_validation (if the smoke passes) | estimated: qwen3:14b validation_v2 0.38 h x 1.7 (24B/14B) | 0.7 | 10-19 01:37 | 10-19 02:19 |
| x2_r2_family_mistral_small32_real (if it passes) | estimated: qwen3:14b measured 4096 + 32768 sessions (0.22 + 1.66 h) x 1.7 x 3 seeds | 9.6 | 10-19 02:19 | 10-19 11:55 |
| x2_evalset_replay_val_v1 | measured proxy: the guard validation's wall clock (same models, arms, seeds, 40-turn sessions) | 14.5 | 10-19 11:55 | 10-20 02:25 |
| B3 guardrail real (not built; if it runs) | estimated: v1b per-tier sums x measured guard-validation to R2-default ratio (llama3.1:8b 1.23, qwen3:14b 1.15) x 3 seeds | 28.6 | 10-20 02:25 | 10-21 07:01 |
| B3 replay real (not built; if it runs) | estimated: as B3 guardrail real | 28.6 | 10-21 07:01 | 10-22 11:37 |
| x2_policy_compare (B5, being built) | no basis yet | ? | 10-22 11:37 | ? |
| x2_window_override_methods | estimated: design estimate (a few short requests and three server restarts) | 0.3 | after B5 | B5 finish + 0.3 h |

All listed work ends by about 2026-10-22 11:40 plus B5's hours. evo-x2 sits idle from then to 2026-11-10 unless more
work is queued. If both evalset real runs refuse at preflight, as the gate note above expects, their about 42 h come
off and everything after them moves earlier by that much (the B3 replay real run would then end about 2026-10-20 17:40). The scale
rows are the largest uncertainty: 72.2 h for scale real rests on the operator's per-session estimates until the probe
has run.
