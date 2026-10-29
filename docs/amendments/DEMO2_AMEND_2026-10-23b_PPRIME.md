# DEMO-2 note: XPS P' rows for the BF0 secondary contrasts

Registered Fri Oct 23 2026, about 10:50 AM ET (Task BJ part A), before any held-out run. Mac only,
no ssh, $0. Approval: docs/amendments/APPROVALS_2026-10-23.md ("P' rows, approved 2026-10-23").
Runners do not read this file.

Adds to, and edits none of: DEMO2_FREEZE_2026-10-23.md, DEMO2_PREFREEZE_2026-10-22.md,
DEMO2_AMEND_2026-10-23_OBJECTIVES.md ("OBJ"), DEMO2_BF0_PREDICTIONS.json.

## 0. Guard: no held-out record exists (re-checked Fri Oct 23 about 10:41 AM ET)
- ~/seam_closeout_1013 (all window_* dirs, all cells.txt / run_ids.txt / steps.txt): no `ho_*`
  line and no `ho_*` or `*heldout*` directory. The newest window is window_20261022_190155_be_v2
  (calibration only).
- derived/, raw/, analysis/: no run.json or plan.json names DEMO2_HELDOUT_60, and no `ho_*` or
  `*heldout*` directory.
- Correction to OBJ section 0: the 4 `NOT_RUN_BEFORE_FREEZE` matches are the guard lines in the 4
  queue scripts (run_queue_{t2s,xps}_be{,_v2}.sh), not lines in calibration window logs. No window
  log contains one. The conclusion (no held-out record) is unchanged.

## 1. What this adds
- OBJ section 1 found that on the XPS, P is P' (PSPEC Qwen3-4B INT8, FREEZE 2b), but BF0's XPS rows
  "P vs R-full", "P vs I-4K" and "P vs C" use P's 8B INT4 multiplier (1.4 x p_A).
- This note adds the matching XPS rows for P', computed by the same function
  (tools/demo2_bf0_predict.contrast) with BF0's own P' multiplier (lo 1.0 / point 1.1 / hi 1.2 x p_A).
- Generator: tools/demo2_bj_pprime_predict.py. File:
  derived/h1_hybrid/demo2/DEMO2_BJ_PPRIME_PREDICTIONS.json
  (sha256 9c8a8e6019b81afd40dd5d0c30ef12305655fac94c6a6c84253972d933340c70). Test:
  tests/test_demo2_bj.py.

| XPS contrast (DERIVED) | dC_iso_rel point | band | dC_iso point, $/task |
|---|---|---|---|
| P' vs R-full | 1.5% | [0.0, 3.0] | 0.0018 |
| P' vs I-4K | 1.5% | [1.3, 1.8] | 0.0036 |
| P' vs C | 1.5% | [0.0, 3.0] | 0.0018 |

- P' vs R-full and P' vs C equal BF0's XPS "P' vs A" row (1.5% [0.0, 3.0]), because the model gives
  R-full, C and A the same p and f (BF0 docstring).

## 2. How the XPS rows are scored
- On the XPS, the contrasts that compare the frozen P arm with R-full, I-4K or C are scored against
  these P' rows. BF0's XPS "P vs ..." rows stay in their file, unchanged, and are reported as
  "8B multiplier, not the frozen XPS arm".
- These are secondary contrasts (PF 2). The confirmatory dollar primary, its fixed sequence
  (T2S then XPS), F2, arms, pins, budgets, PSPEC, HO_CLOUD_CAP and the replay are unchanged.

## 3. Schedule note (dated, not a rule change)
- Fri Oct 23 2026, about 10:45 AM ET, Zach: the T2S held-out runs first this weekend; the XPS
  held-out is deferred until after the Tue Oct 27 meeting.
- The frozen rules do not date the XPS test. PF 2's fixed sequence counts the XPS test only if the
  T2S test rejects, so the order stays T2S then XPS. The XPS held-out must still run from a head
  that contains this note and the freeze note, and no DEMO-2 rule may change before it runs.
