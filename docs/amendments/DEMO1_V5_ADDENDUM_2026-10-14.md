# DEMO-1 V5 addendum 2026-10-14 (committed before any V5 run)

Adds to docs/amendments/DEMO1_AMEND_2026-10-13c_CONFIG.md (ed8ffb2) and
docs/amendments/DEMO1_V5CAL_PREREG_2026-10-14.md (d469942). Edits neither.
Runners do not open this file.

## Inputs (sealed)
- V5-CAL, XPS GPU 4B, RESIDENT, turn-2 TTFT, measured P (common prefix) and
  D (tokens after it), 3 repeats, median:
  - u8 a178e9c5 (MATCH_MANIFEST): P 2,194 / 3,314 / 4,074 / 6,284 / 7,374 at
    D 8,511: 22.41 / 31.81 / 35.49 / 28.57 / 31.70 s.
  - u4 e1ed06b4 (MATCH_MANIFEST): same P and D: 17.34 / 20.98 / 22.78 /
    31.02 / 35.09 s.
  - 15993163 (V5-path, u8, P 4,076, D 8,515): 32.64 s. UNGUARDED (canary
    never armed), recorded, not cited, not used.
- Script: tools/demo1_v5_addendum.py (sha256
  53ecec141a7cec8bee8062185a4f19b9eaf3d64535a2fa8e089c3dc488223914).
  Output: analysis/demo1/V5_ADDENDUM_PREDICTIONS.json (sha256
  774c5d5a6ff6a5e41f7d7537e1452145a8ad1114a88fbd33bb88e29447e57140).

## Findings that change the method
1. Delta prefill on a cached prefix is slower than a cold prefill of the
   same full prompt at D about 8,500: u8 P 2,194 D 8,511 takes 22.4 s, while
   the sealed cold curve gives about 11 s for 10,705 tokens (c2246b1f). The
   amendment's cold bound ("a delta prefill does no more work than a cold
   prefill of the same prompt") is false on this stack and is dropped.
   TTFT is also not monotone in P at fixed D (u8: 35.5 s at 4,074, 28.6 s at
   6,284).
2. The H1 runner escalates on step 0 of each user turn only:
   tools/bfcl_feasibility_probe.py sets turn_ttft_s at step == 0, and
   tools/run_h1_hybrid.py decide_slo_escalate reads that value. A large
   observation consumed by a later step inside the turn is never checked.
   The amendment's per-step rule predicted triggers the runner does not
   evaluate. DEMO_PREREG_v2's per-turn model matched the runner.

## Prediction 1: registered method (amendment, per step, cold bound dropped)
Monotone bound on all sealed warm points plus V5-CAL; block bounds 0 / +256.
- V5a u8: 15/15 escalate, local share 0.036, $0.801/task (both bounds).
- V5b u4: 15/15 escalate, local share 0.036 to 0.000, $0.801 to 0.811/task.
- Every escalation is now determined (0 outside the grid). The jump steps
  are bounded by measured points at 17.3 to 35.5 s. Steps with D 401 to
  8,000 (17 within-turn steps) are bounded only by the D 8,511 points, so
  those bounds are loose.
- Pre-stated rule (V5-CAL prereg): "V5 is not worth running if all 15
  entries are still predicted to escalate at both precisions." They are.
  Verdict under the registered method: V5 is NOT worth running.

## Prediction 2: runner trigger (per turn), labelled companion
Entry step 0 cold; step 0 of a later turn is a delta prefill of the
turn-boundary D (29 to 118 tokens, median 50.5) on the cached prefix.
- Block 0: V5a and V5b 0/15 escalate, local share 1.000, $0.
- Block +256: up to 15/15 escalate (u8 local share 0.309, $0.732/task; u4
  0.273, $0.742). Driver: D + 256 = 285 to 374 is above the dense small-D
  points. u8 is then bounded by one sealed point, 48779b5a P 12,000 D 400 at
  15.56 s (other P 12,000 points are 1.0 to 1.6 s); u4 has no point with
  D' >= 299 above P 7,374 (9 outside the grid).
- All 15 entries flip between the block bounds (boundary). The real KV
  block size is not recorded; with a block of 32 tokens or less, every
  boundary D stays within the D 150 / 183 points and the prediction is the
  block-0 one.
- Unchecked within-turn steps: 53 (u8) and 55 (u4) are predicted above
  10 s, up to 31.7 s (u8) and 35.1 s (u4).
- Reading: under the runner's trigger, a high V5 local share would come
  from the trigger not looking at within-turn steps, not from fast
  residency. The jump prefill would still take 17 to 35 s inside the turn.

## Live-trajectory caveat
V4 XPS a678926e (measured) scored MISS on every prereg v2 band: local share
0.673 (pred 0.436), $0.184/task (pred 0.660-0.715), 9 escalations (pred 15).
Six of 15 entries never escalated although their gold trajectory has a jump
past the 10,000 edge. Four of them never made the large get_stock_info call
locally; two did (134, 136) but their live step-0 n_ctx stayed at or below
6,326. Five escalations happened at step-0 n_ctx 6,804 to 9,042 with TTFT
10.4 to 16.1 s, below the sealed cold edge. So the gold-trajectory model
over-predicted escalation for V4 XPS. Both V5 predictions are gold-path
bounds, not live forecasts.
- A live-conditioned V5 prediction is not computable from sealed data: the
  V4 XPS turn ledger records each turn's step-0 n_ctx only, not the
  within-turn step prompts that set the RESIDENT turn-boundary prefix and D.
  None is registered.

## Decision record
- By the pre-stated rule applied to the registered method, V5 is not worth
  running. The $30 reserve approved by Zach for V5a/V5b (2026-10-13) stays
  unspent.
- If a residency contrast is still wanted, the trigger semantics (step 0
  only, or every step) must be fixed first in its own amendment. Under the
  current trigger, V5 would mostly measure that the trigger ignores
  within-turn steps.
- Spend so far on DEMO-1: about $4.21 (V4 XPS $2.77, V3 T2S $1.44, pings).
