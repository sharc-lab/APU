# X-2 per-turn decomposition (holdout)

Generated: 2026-09-08T23:04:30.420041+00:00

No parameter fitted to X-2.

## (a) sum(ttft+decode_time) vs sealed session wall

| session | placement | residency | sealed s | sum ttft+dec s | gap s | gap/turn s |
|---|---|---|---:|---:|---:|---:|
| cb781dbf | cpu-p | NON_RESIDENT | 8148.5 | 3689.3 | 4459.2 | 63.70 |
| 9fdedb46 | cpu-p | RESIDENT | 1576.4 | 1067.6 | 508.8 | 7.27 |
| afd1aa21 | gpu_only | NON_RESIDENT | 750.5 | 354.9 | 395.6 | 5.65 |
| 0963168f | gpu_only | RESIDENT | 489.5 | 164.9 | 324.6 | 4.77 |

## (b) predicted − measured error series

| session | residency | ttft mean err s | ttft rmse | decode mean err tok/s | decode rmse |
|---|---|---:|---:|---:|---:|
| cb781dbf | NON_RESIDENT | -1.511 | 3.363 | 13.05 | 13.06 |
| 9fdedb46 | RESIDENT | 4.290 | 8.272 | 13.24 | 13.25 |
| afd1aa21 | NON_RESIDENT | 1.646 | 2.179 | 0.58 | 1.98 |
| 0963168f | RESIDENT | 1.169 | 1.877 | 0.12 | 2.07 |

## RESIDENT miss carrier

RESIDENT session under-prediction is dominated by (a): measured ttft+decode_time already leaves a large gap to sealed entry/session wall (tool exec + inter-generate harness). (b-prefill)/(b-decode) errors are secondary for RESIDENT cells.

Mean uncovered s/turn (RESIDENT): 6.021361061457956
Looks like: tool execution + tokenizer/template + KV bookkeeping + harness between generate() calls (entry wall >> sum of step walls)
Independent measure: Instrument a sealed session_residency cell with per-phase timers around execute_multi_turn_func_call and message/template build (bfcl_feasibility_probe generate_turn loop). A dedicated microbench on the same BFCL entries with generate stubbed would isolate tool exec; X-2 sealed steps already expose generate()-only wall_s.
Not added to the model.

