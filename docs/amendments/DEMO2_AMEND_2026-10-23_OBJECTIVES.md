# DEMO-2 amendment: P vs I-4K check, and confirmatory family F2 (non-dollar objectives)

Registered Thu Oct 22 2026, late evening ET (Task BH), before any held-out run. Mac only, no ssh, $0.
Approval: docs/amendments/APPROVALS_2026-10-23.md (Zach, about 10:20 PM ET, "ok register everything
new"). Cited below as "OBJ".

Adds to, and edits none of:
- DEMO2_FREEZE_2026-10-23.md (the freeze, values only);
- DEMO2_PREFREEZE_2026-10-22.md ("PF", sha256 e24f4ce9...);
- DEMO2_ISO_QUALITY_2026-10-21.md, DEMO2_ARMS_REVISED_2026-10-21.md, DEMO2_PER_DEVICE_2026-10-21.md.

Where this note and PF 2 differ, this note governs for the four F2 members only. Runners do not read it.

## 0. Guard: no held-out record exists (checked before writing)
Searched on the Mac, Thu Oct 22 2026 about 10:30 PM ET:
- ~/seam_closeout_1013/xps/window_* and ~/seam_closeout_1013/t2s/window_*: no `ho_*` line in any
  run_ids.txt, cells.txt or steps.txt. There are 4 `NOT_RUN_BEFORE_FREEZE` lines, from the
  calibration windows. The ho_cloud once-only check (`grep -l '^ho_cloud' .../run_ids.txt`) finds
  nothing.
- No directory named `ho_*`, `*heldout*` or `*park*` under ~/seam_closeout_1013, derived/, raw/ or
  analysis/.
- No run.json, plan.json or .sealed under derived/ or raw/ names DEMO2_HELDOUT_60. The only
  repo files that name it are:
  - queue and window scripts, which restore the pin and check its hash;
  - the pin's own .meta.json;
  - the BC and TRUE_DEFAULT prediction files.
- Result: no held-out record. Nothing in this task touches DEMO2_HELDOUT_60.

## 1. A1: P vs I-4K and C vs I-4K are identical on both hosts. NOT a bug; nothing changed
- **Observation.** DEMO2_BF0_PREDICTIONS.json gives these values on both hosts:
  - P vs I-4K: 2.3% [1.3, 4.2];
  - C vs I-4K: 1.3% [0.6, 1.9];
  - both with C_R 0.2374 and Q* 0.5933.
- **Cause (tools/demo2_bf0_predict.py).** With I-4K as the reference, no input depends on the host:
  - p_R = 0.5 x p_A, and p_A = 4/60 on both hosts (tools/demo2_be_predict.P_A).
  - f_R = I4K_F = 48/60 (`f_ref`). This is BC's registered escalation count. BC has I4K-X = I4K-T =
    48 escalations, $0.3564/task (DEMO2_BC_PREDICTIONS.json): at a 4,096 window, the window sets
    every escalation, not the host's TTFT edge.
  - The pass multipliers of X (P 1.4, C 1.0) are the same on both hosts.
  - c and q come from BASE_30 (1acc8c58) and are shared.
  - The only host input in the model is f_A (A's registered $/task). It enters only the contrasts
    whose reference is A, B, C, R-full or P, so those differ by host and the I-4K rows do not.
  - Check: Q* = 1/30 + 0.8 x (22/30 - 1/30) = 0.5933, C_R = 0.8 x 0.2968 = 0.2374.
- **Why it is not a bug.** The generator does what its docstring says ("48/60 for I-4K (its
  registered BC escalation count at the point)"). It also regenerates the committed file byte for
  byte (sha256 c250131d..., checked in this task; test tests/test_demo2_bh.py).
- **What changed: nothing.** No DEMO2_BH_PREDICTIONS.json was made for A1. The F2 predictions are
  in a separately named file (section 4).
- **Related gap, not changed (decision for Zach).**
  - On the XPS, "P" is P' (PSPEC Qwen3-4B INT8, FREEZE 2b). BF0's XPS rows "P vs R-full / I-4K / C"
    were computed with P's 8B INT4 multiplier (1.4).
  - The matching P' rows (multiplier 1.1) do not exist; only "P' vs A" does.
  - Adding them would be a new prediction. A1 does not allow that, so it is not done here.

## 2. F2 member definitions (fields the held-out ledgers already log)
- Arms per host: A (true default), P (T2S: 8B INT8; XPS: P'), C.
- Run dirs: the same local_only dirs the iso-quality replay reads, with hung runs and their
  continuations included.
- Every held-out local cell runs with `-StepLog`, `-DebugRawText`, `-StallWatchdogS` and
  `-TurnWatchdogS`. The frozen v2 queue scripts set them in COMMON / WD. XPS stall 3,600 s, turn
  14,400 s; T2S stall 900 s, turn 3,600 s.

Fields:
- `policies/local_only/turn_ledger.json`: `entries[].entry_id`, `local_pass`, `stop_reason`,
  `turns[].placement`, `turns[].turn`, `turns[].turn_wall_s`.
- `policies/local_only/entry_quality.json`: rows with `status` "skipped_hang", `hang_ref.turn`.
- `hang_events.json`: `events[].entry_id`, `outcome` (hang / stall / turn_budget_exceeded),
  `elapsed_s`.
- `step_log.jsonl`: `pre_generate.prompt_tokens`, and `post_generate.generated_tokens`, `ok`,
  `error` and `wall_s`.
- `plan.json`: `local_limits.context_window.tokens`, or `context_window.tokens` (the arm's window).

Per entry:
- positions of one local generate = prompt_tokens + generated_tokens. If the generate never
  returned, use prompt_tokens alone.
- cap = min(40,960, the arm's window). For A, which has no window, cap = 40,960.
- loud = any of these:
  - stop_reason generation_error or ctx_overflow;
  - a post_generate with ok false or a non-null error;
  - a hang event or skipped_hang row for the entry.
  - `max_steps` is not an error report. The runtime returned normally; it is counted in member d.

The four members:
- **a. silent overrun:** at least one generate with positions > cap, and the entry is not loud.
- **b. wall-clock per entry:** the sum of the entry's local `turn_wall_s`.
  - For a skipped_hang entry (no ledger row): the hang event's `elapsed_s` plus the step_log
    `wall_s` of its turns before t_h. This is a lower bound and is counted as such.
  - Watchdog and stall time is therefore included.
- **c. time-to-failure:** b, on entries with local_pass false (skipped_hang counts as a fail).
- **d. hang/stall/loop:** a hang event or skipped_hang row (turn or stall watchdog fired), or
  stop_reason `max_steps` (the agent-loop step limit, MAXIMUM_STEP_LIMIT).

## 3. Tests, family and claim rule
- **Contrasts:** A vs P and A vs C, per host.
  - A vs R-full is NOT added. Deliverable B (approvals 2026-10-22 item 2) is P vs R-full.
  - A vs R-full answers no deliverable.
  - R-full's 40,960 window makes its member a true by construction.
  - P vs R-full stays a PF 2 secondary.
- **Tests:**
  - a and d: exact two-sided McNemar on the paired binary (b = A only, c = X only).
  - b: two-sided Wilcoxon signed-rank on wall_A - wall_X over all 60 pin entries (zeros dropped).
  - c: the same, on entries where both arms fail locally.
  - b and c report the median paired difference; positive means X is faster.
- **Family F2, per host:** (b, c, d) x (A vs P, A vs C), so **m = 6**.
  - Holm at alpha 0.05. Hosts are separate families.
  - F2 is NOT gated on the dollar primary, and the dollar primary does not depend on F2.
- **By construction:** member a, in both contrasts.
  - P and C run `-ContextWindow W -CtxOverflowLocalFail`, with W = 8,704 on the XPS and 16,896 on
    the T2S. The runner does not generate when input + 512 > W.
  - So X cannot overrun its cap, and X = 0 is a mechanism, not a finding.
  - Member a is tested and reported as a mechanism check, outside the Holm family. It makes no
    claim.
  - A's own count of silent overruns is reported descriptively.
- **Claim rule (per member and contrast):**
  - "X improves on A for k on host H" if the Holm-adjusted p < 0.05 and the data favour X:
    - McNemar: A-only > X-only;
    - Wilcoxon: R+ > R-.
  - "A is better than X for k" if p_Holm < 0.05 and the data favour A.
  - Otherwise there is no claim.
- **If an arm never fails or never fires:**
  - c with no both-fail entry: "not testable", p = 1, and m stays 6.
  - d or a with 0 vs 0: McNemar p = 1, reported as "0 vs 0".
  - If A never overruns: "mechanism not exercised".
- **Power, stated plainly:** exact McNemar cannot reach p < 0.05 unless the A-only count is >= 6.
  Member d is expected to have 0-4 A-only entries, so it is predicted NOT to reject.
- **Code:** tools/demo2_f2_objectives.py (sibling of the replay; reuses its holm, mcnemar_exact and
  refusals). Tests are in tests/test_demo2_bh.py. No runner or queue script changed.
- **Command after held-out, per host:**
  `python tools/demo2_f2_objectives.py --pin derived/h1_hybrid/demo2/DEMO2_HELDOUT_60.json --arm A=<ho_a dirs> --arm P=<ho_p dirs> --arm C=<ho_c dirs> --host <T2S|XPS> --out analysis/demo2_pilots/<task>/f2_<host>.json`
  (on the XPS, P = the ho_p cell, which runs PSPEC).

## 4. DERIVED predictions (derived/h1_hybrid/demo2/DEMO2_BH_F2_PREDICTIONS.json)
Generator: tools/demo2_bh_predict.py. Seed 20261023, 2,000 draws. Only pre-held-out records are used.

Inputs:
- Calibration (DEMO2_CALIB_30, base entries; paired per-entry walls, passes and stops):
  - T2S: A 75fa4d1c, C dd62e520, P c91bc055;
  - XPS: A 7cf9b18d, C a200bd1a, P' 4407accc.
- Long-context priors: arm A on the disjoint DEMO2_HELDOUT_60_B long half.
  - T2S: 1a634610 + daa6c026 (29 entries); XPS: 15367c3a (15 entries).
  - Max prompt per entry from analysis/demo2_pilots/BB/context_over_limit.json.
- Entry 123:
  - T2S A cddcd9fd: 21 steps to 98,165 tokens, stop max_steps, 26.2 min, no watchdog fire, so a
    SILENT LOOP under the held-out T2S limits (stall 900 s; step 1 TTFT 342.7 s; turn 3,600 s).
  - T2S X e66a2a89: a cap of 40,960 gives ctx_overflow at step 1.
  - XPS A: modelled as a stall fire at 3,600 s (841b84e4 stalled, UNSEALED; X0-XPS d0dd5a37 is
    still pending).

Facts behind the priors:
- Past-40,960 generates: 43 = XPS 17 / T2S 26.
  - XPS: 841b84e4 4, 162a9b5c 4, 15367c3a 9.
  - T2S: 1a634610 4, 3aa565da 4, daa6c026 9, 108d5f43 4, 8af10211 5.
  - Source: BB context_over_limit.json, recomputed here.
- Every calibration cell had 0 watchdog fires (FREEZE 2f).

Model assumptions (stated, not measured):
- X stops at the first turn whose step-0 n_ctx + 512 > W. If no step-0 prompt crosses W but the
  entry's max prompt does, X runs half of A's wall.
- X's run part is scaled by a calibration X/A wall ratio drawn at random.
- X takes A's path on long entries.
- XPS long draws use only 16 pool entries, so entry 123 has weight 1/16 there.
- The bands below are sampling bands under this model. They do not include model error, so they
  are narrow, and a MISS is likely even when the direction is right.

Predictions per host and contrast:
- a: A's silent overruns of 60, vs X = 0 by construction.
- b and c: median paired diff A - X in s (positive = X faster), band [p05, p95], predicted
  outcome at Holm level, with P(p < 0.05/6) in brackets.
- d: A count of 60, vs X = 0.

| host | contrast | a | b wall | c time-to-failure | d hang/loop |
|---|---|---|---|---|---|
| T2S | A vs P | 5 [2, 8] | -20.5 [-29.3, -14.3], REJECT, A better (1.00) | -24.8 [-35.3, -17.6], REJECT, A better (1.00) | 1 [0, 3], NOT REJECT |
| T2S | A vs C | 5 [2, 8] | -0.012 [-0.030, 0.012], NOT REJECT (0.03) | -0.012 [-0.026, 0.013], NOT REJECT (0.05) | 1 [0, 3], NOT REJECT |
| XPS | A vs P' | 4 [1, 7] | -6.87 [-8.43, -3.82], REJECT, A better (0.80) | -6.93 [-9.25, -4.27], REJECT, A better (0.80) | 2 [0, 4], NOT REJECT |
| XPS | A vs C | 4 [1, 7] | +0.24 [0.06, 0.65], REJECT, C better (0.85) | +0.31 [0.06, 0.76], REJECT, C better (0.88) | 2 [0, 4], NOT REJECT |

- Long entries where X overflows (predicted): T2S 12 of 30, XPS 17 of 30.
- In words:
  - P is predicted SLOWER than A per entry on both hosts. P is the larger or higher-precision model
    (T2S: 8B INT8; XPS: 4B INT8). Calibration medians are 25.9 s vs 12.2 s on the T2S and 19.0 s
    vs 16.8 s on the XPS.
  - C is predicted faster than A only on the XPS, where its 8,704 window cuts more long entries.
  - On the T2S, C vs A is predicted null.
  - The default's silent overruns (a) and loops (d) are real but rare. A has 4-5 silent overruns of
    60 (DERIVED), and d is too rare to test.
- Scoring (in the file): HIT if the measured statistic is inside [p05, p95] and the measured test
  outcome equals the predicted one. Partial if only one of the two holds.

## 5. Disclosure (why F2 exists, and what was read)
- F2 was added AFTER the research reports flagged small dollar effects:
  - "Dollar win against the default";
  - "Win cost where device defaults break" (DEMO-2's registered savings are 6.2% XPS / 19.6% T2S,
    "cents per task").
  - The deliverable is "$X and/or other objectives", and PF 2 made every non-dollar outcome
    secondary and gated.
  - That is the reason for F2.
- The calibration ledgers were already read in BF0/BF (FREEZE section 2).
  - In this task the author read them again for the F2 prediction inputs above.
  - The author also ran tools/demo2_f2_objectives.py on T2S calibration as a pipeline check:
    - T2S P slower than A, p_Holm 3.4e-8 (b) and 1.2e-6 (c);
    - C vs A p_Holm 0.66;
    - 0 overruns and 0 hang/loop in every arm.
  - The F2 predictions were written knowing those calibration walls. Calibration is base-only, so
    it says nothing about member a, d, or the long half.
- The long-context priors come from pilot runs on DEMO2_HELDOUT_60_B. That pin is disjoint from
  DEMO2_HELDOUT_60 (0 shared ids, checked).
- Members a and d were chosen knowing that A overran 40,960 and looped on the B pin. That is the
  mechanism they measure. a is labelled by construction for that reason.

## 6. A3: unchanged, and an exploratory note
- Unchanged: arms, pins, budgets (T2S 16,384, XPS 8,192), PSPEC, HO_CLOUD_CAP $20, the replay, the
  dollar primary and its fixed sequence (P vs A dC_iso, T2S then XPS), and every BF0/BE/BC
  prediction.
- Exploratory note, not an arm:
  - DEMO-2 runs the original Qwen3-4B, not the 2507 checkpoints.
  - Published BFCL multi-turn is about 19% for Qwen3-4B (19.4%, BFCL v4 multi-turn, Fission-GRPO
    arXiv 2601.15625, via the research report). It is about 52% for Qwen3-4B-Thinking-2507 (figure
    as given in the Task BH brief; source not re-checked in this task).
  - A newer checkpoint is model choice, not configuration. Disclose it as a control only.

## 7. Held-out commands (EXPECT_HEAD = the last Task BH commit, see git log)
- The v2 gate still passes at that head:
  - docs/amendments/DEMO2_FREEZE_2026-10-23.md is in the tree (`ls-tree ... grep
    '^docs/amendments/DEMO2_FREEZE_'`);
  - tools/h1_cloud_accounting.py has `def price_request`;
  - the DEMO2_HELDOUT_60 hash is unchanged.
- Same environment variables as FREEZE section 5:
  - `! git -C ~/gnn-hls-accel push origin main`
  - T2S:
    `REBOOT_OK=1 BUDGET=16384 HELDOUT_OK=1 EXPECT_HEAD=<BH head> FROM=30 bash ~/seam_closeout_1013/xps/run_queue_t2s_be_v2.sh`
  - XPS:
    `HELDOUT_OK=1 EXPECT_HEAD=<BH head> PSPEC='configs\models\Qwen3-4B-int8-ov.yaml' HO_CLOUD_CAP=20 FROM=30 bash ~/seam_closeout_1013/xps/run_queue_xps_be_v2.sh`
- The held-out commands, with the head filled in, are in the Task BH report.
