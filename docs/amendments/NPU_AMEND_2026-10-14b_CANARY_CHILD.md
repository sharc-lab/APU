# NPU amendment 2026-10-14b: canary cell in a child process (pre-run, window 2)

Approved by Zach 2026-10-14 (about 3:30 PM ET), option C of
docs/amendments/DRAFT_NPU_CANARY_STATE_2_window2.md. Committed before any
guarded NPU session that uses it. Adds to
docs/amendments/NPU_AMEND_2026-10-13b_INT8_ENVELOPE.md and
docs/amendments/NPU_AMEND_2026-10-14_CANARY_STATE.md; edits neither.
Runners do not open this file.

## Scope
- Guarded NPU-INT8 envelope sessions (run_npu_profile --envelope without
  --no-canary).
- Unchanged:
  - every other canary user (XPS P1, V5-CAL, H1 opening canaries, the
    ceiling runner)
  - load-only sessions
  - natstop / F1 cells
  - every launch gate

## Unchanged (registered elsewhere)
- Canary cell: gpu_only_f16, 4,000 cached + 400 delta.
- C = 3; ref = calibration median.
- Threshold = max(2 x calibration spread, floor); floors t1 0.0771 and t2
  0.1643.
- The two-sided drift rule and FAIL_CANARY_DRIFT.
- An OTHER canary (no usable turn1/turn2) still refuses after the gate
  arms.
- The 2026-10-14 schedule: record-only canary, warm rung at 64, three
  calibration canaries, telemetry.
- Every 2026-10-13b prediction, validity rule, content check and estimate.

## Change
1. Every canary cell of these sessions runs in a fresh child process,
   spawned per canary: record-only, calibration, checks and closing. The
   child loads its own GPU pipeline, writes its result as JSON to a file in
   the run's work/canaries directory, and exits. It shares no OpenVINO
   context with the worker's NPU pipeline or GPU reference pipeline.
2. The worker records, per canary: child pid, exit code, wall time,
   timeout, whether it timed out, and the stderr tail.
   - A child that exits non-zero, times out or writes no result is
     classified OTHER, with the exit code or timeout in execute_error.
   - The gate then behaves as registered (OTHER after arming refuses).
   - The worker itself never ran the failing GPU work, so it refuses
     cleanly.
3. Transition record (record-only, never gating): in the first two
   sessions under this amendment, the worker also runs the canary cell
   in-process at two points and stores the values next to the child's:
   - the record-only point
   - the first post-arm point
   The in-process run comes after the gating child run at the same point.
   It is switched by configs/npu_canary_child.yaml
   (inprocess_transition_record). A commit sets it to false after the
   second session.

## Pre-stated reading rule (per session, at the first post-arm point)
- **C-confirmed:** child t1 within 3% of the child calibration median, and
  in-process t1 at least 6% below the in-process record-only value. The dip
  is an in-process coupling, and the child process removes it.
- **C-not-sufficient:** child t1 at least 6% below its calibration median.
  The dip is host- or GPU-level. Option H of the draft would be next; it
  needs its own amendment.
- **Anything else:** inconclusive, and recorded.
- Two sessions are read before any further change. No threshold changes in
  any branch.

## Risks
- About +10 s per canary for process start and imports (estimate). Every
  canary also loads its own pipeline; today's in-process cell already loads
  one per canary.
- The child timeout is 180 s. A hang after CL_OUT_OF_RESOURCES (a known
  driver behavior) becomes OTHER at the timeout instead of hanging the
  worker.
- The first two sessions run two extra in-process canaries, which can
  perturb what follows; they come after the gating child canary at the same
  point.
