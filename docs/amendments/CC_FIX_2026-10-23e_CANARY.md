# CC fix 2026-10-23e: canary arming for one-step CC series and one-point CC cells

Recorded Fri Oct 23 2026, about 4:25 PM ET (Task BP). Mac only, $0. Procedural bug fix under the
PLAN LOCK (docs/PLAN_LOCK_2026-10-23.md): it lets registered cells run. No cell, arm, prediction or
threshold changed. Runners do not read this file.

## 1. What failed
- XPS CC v3 window_20261023_132334_cc_v3 (head 241e4d3): rehearsal grow reh_grow
  4bc203d8-4125-4e63-92a4-196b2acf480e ended REFUSED_CANARY ("canary gate armed==false; refuse to
  seal"). The queue stopped at 13:40:56 at its rehearsal gate, as designed.
- Relaunch window_20261023_134421_cc_v3 (FROM=12): the one-point limit rehearsals reh_i8
  84453830-667d-4c4e-b9e4-24bdcd5cfef1 (point 1,024 passed; compiled KV i8; A31 on) and reh_bal
  b8b71ce7-2282-45d6-9500-9b553cfbdf08 (0.5 GiB balloon verified 541 MB; point 1,024 passed) ended
  REFUSED_CANARY for the same reason.
- Logged deviations: the XPS queue was relaunched FROM=14 UNTIL=79 (rehearsals 12-13 skipped; their
  measured evidence above stands), and the T2S CC v3 queue FROM=14 after the held-out. A31 stays on;
  the 8-bit cells use i8 as registered.

## 2. Cause (confirmed by reading the code and by a test against the 241e4d3 code)
- The guard needs C = 3 OK calibration canaries after one discarded warm-up.
- tools/boot4_session.py opened the guard with warm-up + 1 calibration canary, then fired at most
  one canary per probe (guard.after_probe). Arming therefore needs at least 2 probes.
- CC-GROW is one step. CC-CONC stops at its first failing step, and conc_iv is one step. One-point
  limit cells are one probe. A bisection whose low rung fails also stops after one probe. None of
  these can arm, so the seal refuses.

## 3. Fix
- Rule: calibrate before the first probe. This is option (ii) of
  docs/amendments/NPU_AMEND_2026-10-08_GUARD.md (the guard's existing arm_before_probes schedule).
  docs/CANARY_PROTOCOL.md section 3 only requires the first C successful canaries to calibrate
  and the gate not to arm before them; it does not require them to be interleaved with probes.
- run_cc_series (CC-GROW, CC-CONC): always warm-up + C = 3 calibration canaries before the first
  step.
  - The gate must be armed after them, or the run refuses before any step (status REFUSED_CANARY).
    This is NPU 2026-10-08 rule 1.
  - The after-step canary remains the drift check, followed by a closing check if any step ran since
    the last canary.
  - Canaries run in the parent process between measured children. measured_repeat waits for (or
    reaps) the child before it returns, so no canary runs while a growing session or concurrent
    sessions are resident.
- run_resident_limit: the same schedule only for fixed-point cells with fewer than C - 1 = 2
  points (one-point cells). Bisections and cells with 2 or more points are unchanged: warm-up + 1,
  then a canary every N probes, the same planned count and the same N. Test:
  tests/test_cc_bp_canary.py::test_limit_paths_that_already_arm_are_unchanged, which also passes
  against the 241e4d3 code.
- Honest budget: the planned probe count is the real count (1 for grow and one-point cells, len(sessions)
  for conc), not max(4, n).
  - With calibration up front, only one armed check has to fit, so the preflight needs planned >= 1
    and n_budget = max(1, floor(planned / (C + 1))).
  - That gives the same N as before for every planned count.
  - New guard option up_front_budget, default off. NPU and other callers are unchanged.
- Residual, not fixed: a bisection whose low rung fails still stops after one probe and cannot arm.
  It writes summary.json with that point, unsealed.

## 4. Head
This commit is the new EXPECT_HEAD for the grow cells (~/seam_closeout_1013/xps/run_queue_xps_cc_v3b.sh).
