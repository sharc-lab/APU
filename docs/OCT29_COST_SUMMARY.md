# Oct 29 Cost Summary

One page, 2026-10-23. Every number is pasted from a VERIFIED row of docs/NUMBERS_REGISTER.md, cited by row id in
brackets. Measured parts first; projected and assumed parts are labelled as such.

## 1. Headline: route at start (measured by replay of recorded runs)

Setup [hybrid-route-at-start-measured]: B3 v2 tool-using tasks, qwen3:14b on evo-x2 with a memory setting (loaded
context window) of 8192, seeds 20260901 to 20260903, 90 tasks. Each task is routed at its start: if its projected
prompt fits the window it runs locally, otherwise the whole task goes to the cloud from its first call. A local task
whose recorded prompt exceeded the window would be escalated (none was). Measured by replay of recorded runs; the
replay assumes a task's outcome does not depend on whether earlier tasks ran local or on the cloud.

| Policy (memory setting 8192) | Correct | Cloud USD per task | Median s per task |
|---|---|---|---|
| Route at start, cloud = claude-sonnet-5-5 | 88/90 (97.8%) | 0.00819 | 4.84 |
| All cloud, claude-sonnet-5-5 | 90/90 (100.0%) | 0.01020 | 4.59 |
| Naive local, qwen3:14b | 57/90 (63.3%) | 0 | 27.27 |

Accuracy and cost: [hybrid-route-at-start-measured]. Latency (median; p90 11.45 / 6.27 / 41.78 s):
[task-latency-by-policy]. With claude-haiku-5-5 as the cloud model the same policy gives 85/90 (94.4%) at USD 0.00055
per task vs all-cloud 87/90 (96.7%) at USD 0.00067 [hybrid-route-at-start-measured].

**Read this plainly.** Only 21/90 (23.3%) tasks were kept local (19 of the 21 correct locally); 69 went to the cloud
from their first call [hybrid-route-at-start-measured]. The saving over all-cloud comes from those 21 tasks. Raising
the local share is the roadmap. Any "75%" local share is a **target, needs failure detection**; no row measures it. At
a memory setting of 4096 no task fits, so the policy is all-cloud (kept local 0/90) [hybrid-route-at-start-measured].

This result does not depend on a cut-off detector: the flag is the recorded over-window condition. Detector v1 was
retired for routing (row detector-precision-recall, on branch demo/oct28 until it merges, expected by 2026-10-27; see
docs/FINDINGS.md "DECISION (2026-10-23)").

## 2. Supporting measurements (briefly)

- **Naive local is silently wrong** [x2-evalset-tasks-real-v2-wrong-rate] (unit: task turns, 5 seeds, so not the
  same denominator as the table): qwen3:14b wrong 17/150 (11.3%) at the default window, 53/150 (35.3%) at 8192 and
  63/150 (42.0%) at 4096; every wrong task silent (surfaced 0).
- **Late escalation is costly and incomplete** [late-escalation-cost-vs-fresh] [cloud-redo-full-context]: handing a
  wrong local task's full history to claude-sonnet-5-5 costs mean $0.0227 vs a fresh all-cloud run mean $0.0102
  (2.22x on the mean; 1.76x on 119 matched rows), and fixed 167/197 (84.8%), 95% Wilson CI 79.1% to 89.1%. Of the 30
  unfixed, 28 inherited the local model's earlier executed actions.
- **Cloud cost per long agent task, priced not executed** [cloud-cost-per-real-task]: nebius-rebench-openhands
  (n=36 tasks, median 62 steps), median per task with realistic caching: claude-sonnet-5-5 $2.5307, claude-haiku-5-5
  $0.1354. "Realistic" assumes a 0.7 cache hit rate (ASSUMPTION).
- **Billed vs priced** [long-task-cloud-priced-vs-billed]: on 5 long-horizon synthetic, trace-length matched tasks
  (not SWE-Gym), claude-haiku-5-5 billed USD 0.1132 vs realistic priced 0.0726, "billed / realistic 1.559", so the
  real bill was about 1.6x the realistic priced figure (within [best, none] on 5/5 tasks).

## 3. Still projected or assumed

- Cloud cost per long agent task is priced from traces, not executed, and its caching case is an assumption
  [cloud-cost-per-real-task].
- Everything for the Intel laptop (evo-t2s) is projected from evo-x2's 4096 tier, not measured; on that projection a
  window-flag hybrid escalates 88 of 90 tasks and costs more than all-cloud [late-escalation-cost-vs-fresh].
- The route-at-start result is a replay of recorded runs, not a live hybrid deployment (replay assumption above;
  a real deployment would also pay a cache write on the first routed task) [hybrid-route-at-start-measured].
- Long-task local accuracy and long-task latency are PENDING (no results on main) [task-latency-by-policy].
- A 75% local share is a target that needs failure detection, not a result.

## 4. Runs that replace projections (evo-t2s, 2026-10-26 to 2026-11-08, window pending confirmation)

Staged, not loaded (docs/T2S_WEEK_PLAN.md): Intel laptop task accuracy on the B3 v2 tasks (qwen3:14b, Intel CPU at its
4096 default and Intel iGPU at 32768, validation then real run each), then the B5 context-policy cells on the Intel
CPU and iGPU. These replace the projected Intel figures in section 3 with measured ones.
