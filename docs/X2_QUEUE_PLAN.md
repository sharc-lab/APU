# evo-x2 queue plan

## Queue fill of 2026-10-21 (to 2026-11-10)

evo-x2's queue was empty (`queue_empty.flag`, every earlier job done) and the machine idle from about 2026-10-20 06:00Z
until the first job below launched. Each job was pre-registered in its own docs/FINDINGS.md section, committed before
it was deployed or queued; the deployment and queueing details are in "AMENDMENT (2026-10-21 ...): queue fill of
2026-10-21 deployed and queued on evo-x2". Hours are ESTIMATEs from measured session times in committed files, register
row `x2-queue-fill-1020-hours`; dates are back to back from the first launch (2026-10-21 18:03:40Z, watchdog log),
ignoring handoff gaps (seconds through `advance()`, up to 10 minutes through the watchdog). All times UTC.

| pos | queue id | pre-registration | ESTIMATE h (high) | projected start | projected finish (high case) |
|---|---|---|---|---|---|
| 71 | x2_num_keep_qwen3_14b (running) | 0c0c6e4 | 0.64 (0.65) | 10-21 18:03 | 10-21 18:42 (18:42) |
| 72 | x2_evalset_tasks_val_v2_qwen3_8b | e13ee1c | 0.22 (0.47) | 10-21 18:42 | 10-21 18:55 (19:10) |
| 73 | x2_evalset_tasks_real_v2_qwen3_8b (gate: requires_done 72; v2_strict preflight) | e13ee1c | 2.11 (3.16) | 10-21 18:55 | 10-21 21:02 (22:20) |
| 74 | x2_r2_qwen3_8b_seeds_06_10 | 8c2f686 | 2.19 (2.40) | 10-21 21:02 | 10-21 23:13 (10-22 00:44) |

Bases, pasted from `x2-queue-fill-1020-hours`: num_keep "3 x qwen3:14b num_ctx 4096 R2 median 12.2 min x num_keep
overhead 1.05"; tasks validation "qwen3:14b v2 tasks validation sessions ... x qwen3:8b/qwen3:14b R2 median ratio
(default 0.35, 8192 0.62); high = unscaled"; real run "qwen3:14b x2_evalset_tasks_real_v2 measured per tier ... x the
qwen3:8b/qwen3:14b R2 median ratio of the tier ...; high = unscaled (refusal at preflight would take minutes)"; seeds
06-10 "5 x qwen3:8b v1b median at 8192 (15.6 min) + 5 x at 4096 (10.7 min)".

**Total, slack and idle capacity** (same row, pasted): "total ESTIMATE 5.2 h (high 6.7 h) from 2026-10-21T18:03Z; queue
ends 10-21 23:13Z (high 10-22 00:44Z); slack before 2026-11-10 00:00Z 456.8 h (high case 455.3 h); idle capacity left
after the 24 h reserve 432.8 h (high case 431.3 h)". The 24 h reserve before 2026-11-10 00:00Z holds by a wide margin;
after about 2026-10-22 evo-x2 is idle again unless more work is queued.

**Gates.** Job 73 cannot start before job 72 is done (queue gate), and its preflight refuses unless qwen3:8b passes
every gate of job 72's validation with no waiver (the qwen3:14b 80/90 waiver does not extend). A refusal ends it in
minutes with no result, and job 74 then starts that much earlier.

**Decisions recorded here.**
- **d. B5 extensions: none.** The B5 verdict is already refuted on evo-x2's complete cells (register row `b5-verdict`,
  pasted: "refuted (router dominated in evo-x2/llama3.1:8b, evo-x2/qwen3:14b); evo-x2 complete (router on frontier in
  every complete cell: False ...)"). More evo-x2 B5 cells cannot change a refutation, so none are queued.
- **e. qwen3:32b default tier: not queued, already run.** It was pre-registered in H-scale (2026-10-14) and ran inside
  `x2_r2_scale_real` (register row `x2-scale-real-default-tier`, pasted: "qwen3:32b: 3 default-tier sessions completed
  (seed 20260901: 40 turns, 6.18 h, ...; seed 20260902: 40 turns, 6.01 h, ...; seed 20260903: 40 turns, 4.66 h, ...)").
  A rerun would duplicate completed cells. llama3.3:70b resisted rule loss (`H-scale-verdict`), so no default-tier run
  for it either.

## Earlier plan (2026-10-15)

Operator decisions of 2026-10-14 (the first set at about 02:45Z on 2026-10-15, revised at 21:40 PT = 04:40Z), applied
by the queue steward. Hours come from committed result files where a measured rate exists; every other figure is
labelled estimated and says what it is scaled from. All times UTC.

### Applied order (operator decision of 21:40 PT)

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

### Finished jobs

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

### Measured rates used

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

### Hours table (through 2026-11-10)

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
