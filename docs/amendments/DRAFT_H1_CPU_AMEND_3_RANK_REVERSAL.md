# DRAFT H1-CPU amendment 3: rank-reversal criterion and matched-rate random baseline

DRAFT, NOT REGISTERED (Task AQ, 2026-10-15). Must be approved by Zach and
registered (sha256 in the H1-CPU registry row) BEFORE the first H1-CPU boot.
Adds to derived/h1_hybrid/H1_CPU_PREREG.json and H1_CPU_AMEND_2.json; edits
neither. Runners do not open this file. No run, cell, entry, cap or
existing prediction changes.

## Why

H1-CPU is the test of H-C (docs/THESIS.md): which router wins depends on
the device configuration, with the model held fixed. The 2026-10-15
literature check requires two things next to any rank-reversal claim
(EC2-Bench 2610.17816, STEPGATE 2610.07816, CARGO 2607.20481;
docs/RELATED_SEARCHES.md): the model held fixed, and a random-escalation
baseline at a matched rate. The registered prereg and amendment 2 state
per-policy agreement predictions (at least 24 of 30 entries agree with
d482c621). They state no ranking criterion and no random baseline.

## 1. Comparison (fixed before the run)

- Model: Qwen3-4B int4 (OpenVINO/Qwen3-4B-int4-ov b467368d), the same as
  d482c621. Configuration moved: placement GPU (d482c621) -> CPU (cpu-p,
  H1-CPU). Residency RESIDENT in both.
- Entry set for ranking: DEMO1_BASE_30 (seed 20260930), the only set on
  which all four arms run under amendment 2. slo_escalate also runs on the
  full W-3 200; its 200-entry result is reported, not ranked.
- Arms ranked: slo_escalate, emission_escalate, full_signal_bounceback and
  local_only.
- Rank key: hybrid passes on the 30 entries. Ties are broken by lower
  cloud $/task.
- GPU reference ranks, from d482c621 restricted to BASE_30
  (analysis/aq_20261015/LEVER_SIZE.json): emission 7 passes ($0.052/task);
  slo_escalate 4 ($0); bounceback 4 ($0.028). local_only was not run on GPU
  in d482c621. Its GPU stand-in is slo_escalate, which made 0 cloud turns.

## 2. Reversal criterion (proposal; Zach to confirm or change before registration)

A rank reversal is recorded for an ordered pair (A, B) when both hold:
1. A ranks above B on GPU and below B on CPU.
2. On CPU, the pass difference |passes(B) - passes(A)| lies outside what
   matched-rate random escalation gives for B. B's passes must fall
   outside the exact 95% interval of the random baseline at B's own
   escalation count.

The amendment-2 agreement predictions are scored separately and are
unchanged.

## 3. Matched-rate random baseline

- For each arm, k = the number of BASE_30 entries it escalated.
- Baseline: escalate k of the 30 entries uniformly at random at turn 0.
  An escalated entry takes its cloud outcome and the rest take their local
  outcome. Use the exact distribution, as in tools/aq_sealed_analyses.py
  (_random_baseline).
- Local outcomes on CPU: the H1-CPU local_only arm. Cloud outcomes: XPS V2
  1acc8c58 (BASE_30, cloud from turn 0).
- GPU baseline, already computed from sealed data
  (analysis/aq_20261015/RANDOM_BASELINE.json): at k = 6, expected 6.8
  passes, 95% interval [4, 9].

## 4. Prediction (from sealed data; proposal)

On CPU, slo_escalate escalates 30/30 at turn 0 (amendment 2, 7f232f86). It
then behaves as cloud-only: about 22 passes, as in V2 1acc8c58. On GPU,
slo_escalate is tied last (4 passes, 0 escalations). So slo_escalate is
predicted to move from tied last to first, reversing against both
emission_escalate and full_signal_bounceback. For each of those pairs the
criterion in section 2 needs the CPU gap to clear the random interval.

## 5. Not changed

Caps, boots, entries, order, the amendment-2 predictions and the spend
approval requirement are unchanged.
