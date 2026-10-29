# DRAFT (not registered, not implemented): NPU canary state 2, for window 2

Status: DRAFT, 2026-10-14 (task AJ). It becomes an amendment only when Zach
approves it and it is committed under a dated NPU_AMEND_ name before any
window-2 guarded NPU session. Runners do not open this file.

Keeps unchanged: canary cell (gpu_only_f16, 4,000 + 400), C = 3,
ref = calibration median, threshold = max(2 x calibration spread, floor),
floors t1 0.0771 / t2 0.1643, the two-sided drift rule, FAIL_CANARY_DRIFT,
and every NPU-INT8 2026-10-13b prediction.

## Why (analysis/canary/CANARY_POSTRUNG_20261014.md)

- In 7 of 7 guarded NPU-INT8 sessions with a value, the first post-arm
  canary has t1 6.3-13.2% below the calibration median. Its absolute level
  is about 1.02 s; calibration is about 1.12 s.
- The shift does not scale with the high rung (rho 0.17), the gap since
  calibration (rho -0.09), or host uptime.
- Later canaries return to the calibration level. The dip is a transient
  at the first check, not a persistent post-rung state.
- The 2026-10-14 warm rung at 64 tokens did not reproduce the dip state.
  The first check still dips under it: 2048d -6.3%, 4096 0965a50b -13.2%.
- At the 0.0771 t1 floor this trips the first check in about half the
  sessions. That is systematic, not random.

## Option C (recommended): canary in a child process

- Every canary cell (record-only, calibration, checks, closing) runs in a
  fresh child process. It loads its own GPU pipeline and exits, and shares
  no OpenVINO context with the worker's NPU pipeline or GPU reference
  pipeline.
- Calibration and checks therefore run in the same state by construction:
  a fresh process with the worker idle while the canary runs.
- It also isolates the CL_OUT_OF_RESOURCES teardown crash seen in
  c759d72f. A canary failure stays in the child; the gate still classifies
  OTHER and still refuses, as registered.
- Transition record (record-only, never gating): in the first two window-2
  sessions, at the record-only point and at the first post-arm point, also
  run the in-process canary as today. Store both values.
- Cost: about +10 s per canary for process start and imports (estimate,
  not measured). That is about +1-2 min per session.

Pre-stated reading rule (per session, at the first post-arm point):
- **C-confirmed:** child t1 within 3% of the child calibration median, and
  in-process t1 at least 6% below the in-process record-only value. The dip
  is an in-process coupling, and option C removes it.
- **C-not-sufficient:** child t1 at least 6% below its calibration median.
  The dip is host- or GPU-level, so try option H in the next session.
- Anything else is inconclusive and recorded. Two sessions are read before
  any further change. No threshold changes in any branch.

## Option H (fallback): warm the full first window before calibration

- Before calibration, run the session's first bisection window untimed:
  rung 64 x3, then the high rung x3, with the paired NPU and GPU 64-token
  validity generate at each, exactly as the first measured window.
- Then run one record-only canary, which is expected to show the
  transient, then the C = 3 calibration canaries, then probes. Bisection
  then starts at rung 64 as registered.
- Risk, stated in advance: calibration taken inside the transient would
  read about 1.02, and later checks about 1.10, that is +8% and a trip. The
  record-only canary after the warm window shows whether the transient is
  still present when calibration starts.
- Cost: one window of NPU time. At 2048 that is about 2.5 min on the
  post-reboot T2S; at 8192 more (the 8128 rung is about 21 s prefill plus
  64-token decode).

Pre-stated reading rule (per session):
- **H-ok:** the calibration median is within 3% of the first post-arm t1.
- **H-transient:** the record-only canary after the warm window is at least
  6% below the calibration median. The transient decayed before
  calibration, which is the intended state.
- **H-fails:** the first post-arm t1 is still at least 6% below the
  calibration median. Stop; record; no further protocol change without
  Zach.

## Order and scope
- Window 2, guarded NPU-INT8 envelope sessions only. Start with option C at
  2048 (a setting with a citable limit, 9acabbf7), then 4096 and 8192.
- The 2026-10-14 amendment (canary state) would be superseded for the
  sessions this applies to. Its record-only canary and telemetry stay.
- Implementation scope for C: the guard's _run_cell calls a child process
  (python -m tools.smoke_delta_prefill ... with JSON on stdout) instead of
  calling run_cell in-process. About 40 lines plus tests. The XPS P1 and
  V5-CAL canaries are untouched.
