# Oct 29 Cost Summary

One page, 2026-10-24, reframed 2026-10-26. Every number is pasted from a VERIFIED row of docs/NUMBERS_REGISTER.md,
cited by row id in brackets. Measured parts first; projected and assumed parts are labelled as such. Every percentage
carries its count, as "42.0% (63/150)". Every 95% CI names its unit: "task-level" (Wilson or bootstrap over tasks,
treating tasks as independent) or "session-level" (bootstrap over whole sessions, 10000 resamples, seed 20261014).
A session is one 30-task local run (one memory setting and one seed); tasks inside a session share an environment
and a history, so they are not independent, and the session-level CI is the one to read.

## 1. Lead: correctness first, cost as a hard cap

**Reliability and budget control for local and hybrid agents: correctness is the hook, cost is a hard cap.** On the
B3 v2 tool-using tasks with qwen3:14b on evo-x2 (AMD), a local model that runs out of memory gets tasks wrong without
saying so, and routing plus a cloud fallback removes most of those wrong answers while costing less than sending
everything to the cloud.

- **16x fewer wrong answers, at the 8192 memory setting.** Naive local was wrong on 36.7% (33/90) tasks; route at
  start (claude-sonnet-5-5 as the cloud model) was wrong on 2.2% (2/90): 33 vs 2 = 16.5x
  [hybrid-route-at-start-measured] [headline-ci-session-hybrid]. Session-level 95% CI for the ratio [6.0x, 19.0x], over only 3 sessions (one per
  seed), so very coarse; 3003 of 10000 resamples had 0 route-at-start wrong and are left out of the ratio
  [headline-ci-session-hybrid]. The setting matters: 8192 is the setting the hybrid also uses, and at the default
  window naive local was wrong on 11.1% (10/90) tasks [hybrid-savings-v1] [cost-target-scenario].
- **0/197 warnings.** Of the 197 wrong local task turns of the B3 v2 real run (qwen3:14b, 5 memory settings x 5
  seeds = 25 sessions), 0.0% (0/197) surfaced an HTTP error or an error field to the client; 100.0% (197/197) were
  silent. Task-level 95% Wilson CI for the surfaced share [0.0%, 1.9%] (task turns are nested in sessions, so not
  independent); the session-level bootstrap is a point at 0 (0/n has no bootstrap variance)
  [x2-evalset-tasks-real-v2-silent-total]. "Surfaced" is what the client sees, not the Ollama server log.
- **99% rescue with rollback, under a stand-in.** The cloud (claude-sonnet-5-5) rescued 99.0% (195/197) of those
  wrong tasks: 167 by a redo with the full history plus 28 by rolling the environment back and redoing
  [cloud-rescue-combined]. Task-level 95% Wilson CI [96.4%, 99.7%] [cloud-rescue-combined]; session-level 95% CI
  [97.5%, 100.0%] (25 sessions, stratified by memory setting) [headline-ci-session-rescue]. Stand-in caveat: the
  rollback redo replaced every earlier wrong task turn by its reference calls, i.e. it assumes each earlier
  escalation succeeded; it is not a live hybrid [cloud-rescue-combined].
- **Cost is measured at 20% below all-cloud (19.7%, n = 90 tasks), quality not matched.** Route at start costs USD 0.00819 per task vs
  all-cloud USD 0.01020, 19.7% lower, at 97.8% (88/90) correct vs 100.0% (90/90) [hybrid-route-at-start-measured];
  session-level 95% CI for the saving [16.8%, 22.4%] over 3 sessions, very coarse [headline-ci-session-hybrid].
  Section 2 gives the setup.
- **75% is a target scenario, not measured** (section 3; 75.3%, n = 90 tasks, under a perfect detector that does not exist).

## 2. Measured cost headline: route at start (measured by replay of recorded runs)

Setup [hybrid-route-at-start-measured]: B3 v2 tool-using tasks, qwen3:14b on evo-x2 with a memory setting (loaded
context window) of 8192, seeds 20260901 to 20260903, 90 tasks. Each task is routed at its start: if its projected
prompt fits the window it runs locally, otherwise the whole task goes to the cloud from its first call. A local task
whose recorded prompt exceeded the window would be escalated (none was). Measured by replay of recorded runs; the
replay assumes a task's outcome does not depend on whether earlier tasks ran local or on the cloud.

| Policy (memory setting 8192) | Correct | Cloud USD per task | Median s per task |
|---|---|---|---|
| Route at start, cloud = claude-sonnet-5-5 | 97.8% (88/90) | 0.00819 | 4.84 |
| All cloud, claude-sonnet-5-5 | 100.0% (90/90) | 0.01020 | 4.59 |
| Naive local, qwen3:14b | 63.3% (57/90) | 0 | 27.27 |

Accuracy and cost: [hybrid-route-at-start-measured]. Latency (median; p90 11.45 / 6.27 / 41.78 s):
[task-latency-by-policy]. With claude-haiku-5-5 as the cloud model the same policy gives 94.4% (85/90) at USD 0.00055
per task vs all-cloud 96.7% (87/90) at USD 0.00067 [hybrid-route-at-start-measured].

CIs for the table (8192, claude-sonnet-5-5) [headline-ci-session-hybrid]: route at start task-level Wilson [92.3%,
99.4%], session-level [93.3%, 100.0%]; all cloud task-level [95.9%, 100.0%], session-level [100.0%, 100.0%] (90/90
has no bootstrap variance); naive local task-level [53.0%, 72.6%], session-level [56.7%, 73.3%]. There are 3 sessions
(one per seed, 30 tasks each), so the session bootstrap has at most 10 distinct resamples: read it as a sensitivity
check, not a precise interval.

**Read this plainly.** Only 23.3% (21/90) tasks were kept local (19 of the 21 correct locally); 69 went to the cloud
from their first call [hybrid-route-at-start-measured]. The saving over all-cloud comes from those 21 tasks. Raising
the local share is the roadmap. At a memory setting of 4096 no task fits, so the policy is all-cloud (kept local
0.0% (0/90)) [hybrid-route-at-start-measured].

This result does not depend on a cut-off detector: the flag is the recorded over-window condition. Detector v1 was
retired for routing (pooled precision 0.588 [detector-precision-recall]; docs/FINDINGS.md "DECISION (2026-10-23)").

## 3. Target scenario, not measured: 75% below all-cloud

The "75%" is the ceiling policy of [hybrid-savings-v1] expressed as a share of the all-cloud cost
[cost-target-scenario]: at evo-x2's default window with claude-sonnet-5-5, keep local the 88.9% (80/90) tasks
qwen3:14b got right and escalate exactly the 10 it got wrong, for USD 0.00252 per task vs all-cloud USD 0.01020,
75.3% below all-cloud (n = 90 tasks), at 90/90. **Target scenario, not measured.** Its assumptions [cost-target-scenario]: (1) a
detector that flags exactly the wrong local tasks and no right ones, which does not exist (the window flag caught 0 of
the 10 wrong tasks at this window); (2) each escalated task costs the measured full-history redo mean; (3) each
escalated task takes the all-cloud outcome of the same task (assumed; the measured redo did not fix every wrong task
[cloud-redo-full-context]); (4) a replay of recorded runs on short B3 v2 tasks only. The measured figure is the 19.7% (n = 90 tasks)
of section 2.

## 4. Supporting measurements (briefly)

- **Naive local is silently wrong** [x2-evalset-tasks-real-v2-wrong-rate] (unit: task turns, 5 seeds, so not the
  same denominator as the table): qwen3:14b wrong 11.3% (17/150) at the default window, 35.3% (53/150) at 8192 and
  42.0% (63/150) at 4096; every wrong task silent (surfaced 0). The excess over the default (+24.0 points at 8192,
  +30.7 at 4096) has session-level 95% CIs [+18.7, +29.3] and [+26.0, +34.7] (5 vs 5 sessions); the task-turn-level
  CIs are wider here, [+14.7, +33.3] and [+21.3, +40.0] [x2-evalset-tasks-real-v2-ci-units].
- **Late escalation is costly and incomplete** [late-escalation-cost-vs-fresh] [cloud-redo-full-context]: handing a
  wrong local task's full history to claude-sonnet-5-5 costs mean $0.0227 vs a fresh all-cloud run mean $0.0102
  (2.22x on the mean; 1.76x on 119 matched rows), and fixed 84.8% (167/197), task-level 95% Wilson CI 79.1% to 89.1%
  [cloud-redo-full-context]; session-level 95% CI 79.4% to 90.4% (25 sessions) [headline-ci-session-rescue]. Of the
  30 unfixed, 28 inherited the local model's earlier executed actions.
- **Rollback fixes those 28, under a stand-in** [own-rollback-redo]: rolling the environment back and redoing the 28
  earlier_state_repair tasks on claude-sonnet-5-5 fixed 100.0% (28/28), task-level 95% Wilson CI [87.9%, 100.0%]
  (pre-registered target at least 25/28: met); session-level CI [100.0%, 100.0%] (12 sessions; 28/28 has no bootstrap
  variance) [headline-ci-session-rescue]. Stand-in caveat: every earlier
  wrong task turn was replaced by its reference calls, i.e. the redo assumes each earlier escalation succeeded; it is
  not a live hybrid. Combined with the full-history redo, the cloud rescued 99.0% (195/197), task-level 95% Wilson CI
  [96.4%, 99.7%] [cloud-rescue-combined], session-level [97.5%, 100.0%] [headline-ci-session-rescue]: 167 fixed by the full-history redo plus these 28. The 2 still wrong (file_write,
  ollama_default, seed 20260904, turn 24; cal_default_len, ollama_ctx_4096, seed 20260904, turn 6) are class
  cloud_model_error, the cloud model's own errors [cloud-rescue-combined].
- **Local first with rollback, oracle vs measured signal** [hybrid-local-first-rollback] (8192, claude-sonnet-5-5,
  replay): with an oracle signal (escalate iff the task ended wrong; a ceiling, no such signal exists) 100.0% (90/90)
  at USD 0.00805 per task, kept local 63.3% (57/90). With the measured stand-in signal (the recorded over-window
  condition) 96.7% (87/90) at USD 0.01485 per task, kept local 30.0% (27/90), which costs more than all-cloud (100.0%
  (90/90), USD 0.01020).
- **The runtime's own cut-off signal passes its check** [demo-detector-check-v2]: detector v2 is Ollama's own
  truncation log lines (message drop, token cut, context shift, exceed), not our invention. Pre-registered rule met:
  fired on 5/5 tasks at num_ctx 4096 and 0/5 at the runtime default.
- **Recorded miss: the profiler** [own-profiler-loo]: predicting seconds per task from a short probe, leave one device
  out, gave median APE 70.2% against the pre-registered target under 10% (MISSED; accuracy median APE 1.5%, met;
  APE is an error size, not a share of units, so it has no k/n). The
  measured replacement, own-profiler-probe, is queued and PENDING [own-profiler-probe].
- **Cloud cost per long agent task, priced not executed** [cloud-cost-per-real-task]: nebius-rebench-openhands
  (n=36 tasks, median 62 steps), median per task with realistic caching: claude-sonnet-5-5 $2.5307, claude-haiku-5-5
  $0.1354. "Realistic" assumes a 0.7 cache hit rate (ASSUMPTION).
- **Billed vs priced** [long-task-cloud-priced-vs-billed]: on 5 long-horizon synthetic, trace-length matched tasks
  (not SWE-Gym), claude-haiku-5-5 billed USD 0.1132 vs realistic priced 0.0726, "billed / realistic 1.559", so the
  real bill was about 1.6x the realistic priced figure (within [best, none] on 5/5 tasks).

## 5. Still projected or assumed

- Cloud cost per long agent task is priced from traces, not executed, and its caching case is an assumption
  [cloud-cost-per-real-task].
- Everything for the Intel laptop (evo-t2s) is projected from evo-x2's 4096 tier, not measured; on that projection a
  window-flag hybrid escalates 88 of 90 tasks and costs more than all-cloud [late-escalation-cost-vs-fresh].
- The route-at-start result is a replay of recorded runs, not a live hybrid deployment (replay assumption above;
  a real deployment would also pay a cache write on the first routed task) [hybrid-route-at-start-measured].
- The 195/197 rescue and the 28/28 rollback redo rest on a stand-in (earlier wrong turns replaced by their reference
  calls), and the local-first figures are a replay [cloud-rescue-combined] [hybrid-local-first-rollback].
- Long-task local accuracy and long-task latency are PENDING (no results on main) [task-latency-by-policy].
- The 75% cost figure is a target scenario that needs a perfect failure detector, not a result [cost-target-scenario].

## 6. Runs that replace projections (evo-t2s, 2026-10-26 to 2026-11-08, window pending confirmation)

Staged, not loaded (docs/T2S_WEEK_PLAN.md): Intel laptop task accuracy on the B3 v2 tasks (qwen3:14b, Intel CPU at its
4096 default and Intel iGPU at 32768, validation then real run each), then the B5 context-policy cells on the Intel
CPU and iGPU. These replace the projected Intel figures in section 5 with measured ones.
