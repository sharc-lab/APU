# NPU amendment 2026-10-14: canary calibration state (pre-run, window 2)

Adds to docs/amendments/NPU_AMEND_2026-10-13b_INT8_ENVELOPE.md; edits
neither it nor any earlier amendment. Committed before any further guarded
NPU session. Runners do not open this file.

## Scope
- Guarded NPU-INT8 envelope sessions only: tools/run_npu_profile.py
  --envelope without --no-canary (launch_boot1.ps1 profile npu-1 with
  -Envelope).
- Unchanged:
  - the int4 npu-1 path
  - load-only sessions
  - natstop / F1 cells (unguarded by design; this includes NPU-1-CW)
  - every GPU-only profile (P1, H1, V5-CAL, resident, t2s-boot1)

## Evidence
analysis/canary/CANARY_DRIFT_20261014.{md,json}. Every guarded NPU-INT8
session that reached a checked canary had its first checked canary
(after rung 64 and the high rung) with t1 below the calibration median:

| session | ref t1 | first check t1 | signed | thr t1 | outcome |
|---|---|---|---|---|---|
| 1024 1e79daa3 | 1.1239 | 0.9853 | -12.3% | 0.1757 | pass (wide threshold) |
| 2048a 0fd9cd8c | 1.0896 | 0.9941 | -8.8% | 0.0771 | TRIP t1 |
| 4096 63d35555 | 1.1002 | 1.0231 | -7.0% | 0.0771 | pass |
| 2048c 4f6a9102 | 1.1732 | 1.0907 | -7.0% | 0.0771 | TRIP t2 (+57.0%) |
| 8192 (2026-10-14, Zach) | 1.1177 | 1.0235 | -8.4% | 0.0771 | TRIP t1 |

The 8192 row comes from Zach's report of the refusal (rel_drift 0.084232).
It is added to the table when its evidence is sealed. Later canaries
returned to the calibration level: 1024 canary 4 at -2.7%; 4096 canaries
4-6 at +1.9% to +2.2%.

The calibration canaries are taken right after the GPU warm-up generate,
before any NPU rung. The checked canaries are taken after NPU rungs. So the
reference is measured in a different session state from the values it is
compared with.

## Change (protocol state, not threshold)
Before the C = 3 calibration canaries, and after the GPU pipeline warm-up
(2026-10-08b):
1. One record-only canary, using the existing warm-up flag: it is
   discarded from calibration and never trips. It records the
   pre-NPU-rung state.
2. One untimed warm-up NPU rung at the low rung (64 tokens). It makes the
   same calls as a measured rung: a paired 64-token NPU and GPU generate,
   then two NPU 1-token generates. It is not a probe and not scored, and
   bisection still visits rung 64 as a measured rung. Recorded in
   plan.canary_warmup_rung.
3. The 3 calibration canaries, then probes as before.

Per-canary telemetry, recorded only and never gating
(tools/canary_telemetry.py), is taken just before each canary cell:
- CPU clock (psutil.cpu_freq)
- GPU clock (no portable reader; recorded as not read)
- the active power plan (powercfg /getactivescheme)
- free memory
- process count and names, with the change since the previous canary

Unchanged:
- the canary cell
- C = 3
- ref = median of the calibration canaries
- threshold = max(2 x calibration spread, floor)
- floors t1 0.0771 and t2 0.1643
- the drift rule and the trip action (FAIL_CANARY_DRIFT)
- the 2026-10-13b predictions, validity rule, content check and wall
  estimates

Why this is not a threshold change: no number in the drift rule moves, and
no session can pass a check it would otherwise fail because a limit was
widened. Only the moment the reference is sampled moves: from "before any
NPU work" to "after one NPU rung". That puts it in the same kind of state
as the checks.

One side effect to state plainly: the old canary 0, often the slowest one
(1024: 1.2226; 2048c: 1.2076), no longer sits in calibration. Sessions like
1024 therefore lose the wide threshold that a slow first canary gave them.
The threshold formula is unchanged.

## Replay from the recorded series (approximate)
No session ran with this schedule, so the replay rests on two
assumptions. Under A, the post-rung state is the "fast" state seen at each
first check. Under B, the fast reading was a one-time effect and
calibration after the warm-up rung lands at the later level.

| session | A: ref ~ first check, later canaries | B: ref ~ later level |
|---|---|---|
| 1024 | ref 0.9853; canary 4 1.0939 = +11.0% > 0.0771: TRIP (was pass) | pass |
| 2048a | ref 0.9941; no later canary: unknown | first-check trip removed only if the fast reading does not recur |
| 4096 | ref 1.0231; canary 4 1.1216 = +9.6%: TRIP (was pass) | pass |
| 2048c | t2 host event (+57%) is not addressed; unknown | unknown |
| 8192 | ref 1.0235; no later canary: unknown | as 2048a |

The recorded series cannot tell A from B. Under A, this change would turn
the two passing sessions into trips. Under B, it would remove the t1 trips.

## Pre-stated reading rule (record only; no gate)
Per amended session, compare the record-only canary's t1 (pre-rung) with
the calibration median (post-rung):
- If the median is at least 7% below the record-only canary, the
  post-rung state is the fast one (A).
- If it is within 3%, it is B.
- Anything between is inconclusive.

The reading goes in the session's closeout. Any further protocol change
needs its own amendment and Zach's OK. Thresholds and floors stay.

## Risks
1. Under A, sessions that would have passed now refuse. That costs T2S
   time, not data validity: the checks still compare like with like.
2. Extra wall time: about 213 s per session
   (run_npu_profile.canary_warmup_rung_estimate_s(64)). This is not added
   to the registered 2026-10-13b estimates. The largest planned session
   (8192, 3823 s) plus 213 s still fits the 7200 s bound.
3. The NPU and GPU pipelines run one more paired generate before
   calibration. A failure there is recorded in plan.canary_warmup_rung and
   does not stop the session. The measured rung 64 still decides validity.
4. Telemetry adds a process listing and a powercfg call (5 s timeout)
   before each canary cell. This is outside the timed cell.
5. The change does not address host activity (2048c's t2). The process
   names in telemetry are what would attribute it.
6. The int8 GPU canary floors on the T2S are still borrowed from aipc-c1.

## Code
- tools/run_npu_profile.py: CANARY_STATE_AMENDMENT, canary_warmup_rung,
  canary_warmup_rung_estimate_s; _arm_canary(before_calibration=,
  telemetry=); the envelope path wires both.
- tools/ttft_slo_canary.py: optional `telemetry` on TtftSloCanaryGuard.
  It is recorded on each canary record and is None by default, so every
  other caller is unchanged.
- tools/canary_telemetry.py.
- tests/test_npu_canary_state.py.
