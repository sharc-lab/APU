# DEMO-2 revised arms, controls, predictions and $0 ladder (Task BC, 2026-10-21)

Registered before any held-out run (no DEMO2_HELDOUT_60 / _B cell has run).
Writing it was approved by Zach 2026-10-21 ("write it"). Held-out runs and
every paid cell still need his spend approval.
Basis: research reports ~/seam_closeout_1013/research_20261021/ ("Tuned config
optimization DEMO2", "Typical device configs hybrid LLMs"); repo facts
analysis/demo2_pilots/BC/REPO_FACTS.md.

Supersedes, without editing them: the arm B and arm C definitions in
DEMO2_AMEND_2026-10-16_TRUE_DEFAULT.md section 2, and the B/C values in
DRAFT_DEMO2_ARM_SPEC_ADDENDUM.md. Arm A, DEMO2_HELDOUT_60, hybrid_full, the
router and every trigger threshold, the primary test, the non-inferiority
margin and test, the decomposition, the material-win rule and the pivot rule of
that amendment are unchanged.

## 1. Facts this rests on (REPO_FACTS.md)
- The IR is the hybrid Qwen3-4B (official OpenVINO/Qwen3-4B-int4-ov at its
  current revision b467368d), 40,960 positions, no RoPE scaling.
- Thinking is off in every arm, through the chat template, enforced per step.
- The IR rt_info says AWQ True, scale estimation False (the card says True).
- The 18,500 error (a5d19dcc) is an OpenCL event error in turn 2;
  19,000-31,000 fail in turn 1 with CL_OUT_OF_RESOURCES.
- Every arm passes max_new_tokens 512. Not-at-cap steps: p99 301, max 482.
- No DEMO-2 arm uses CACHE_DIR.

## 2. Arms (launch_h1.ps1 flags; all arms -Policy hybrid_full -Placement gpu_only
## -Residency RESIDENT -Kv default -RequireTrueDefault unless stated)
| arm | what differs from A | flags added |
|---|---|---|
| A | true default (unchanged) | none |
| B (revised, device-only) | prompt budget P per device: a step whose prompt exceeds P is not generated and escalates (existing sticky ctx_overflow) | -ContextWindow P+512 |
| C (revised) | B + retention: each tool output cut at insertion to K = 2,048 model tokens, JSON-aware (h70t30 fallback), marker "[... N tokens elided ...]"; lazy age-masking when a step would exceed the window, oldest first, never the last 3 tool messages, frozen once masked | B + -RetentionK 2048 -RetentionShape json -RetentionAgeMask |
| I-4K (= registered D, relabelled) | 4,096 window, escalate on overflow | -ContextWindow 4096 |
| R-full (new) | f16 KV, explicit complete SchedulerConfig (prefix caching on, max_num_batched_tokens SIZE_MAX, dynamic split-fuse on), loud escalation at 40,960, no context management | -Kv f16 -SchedPrefixCaching on -SchedMaxBatchedTokens 18446744073709551615 -SchedDynamicSplitFuse on -CompiledReadback -ContextWindow 40960 (instead of -Kv default -RequireTrueDefault) |

- P: XPS 8,192 (-ContextWindow 8704), T2S 16,384 (-ContextWindow 16896).
  The ceiling P + 512 is within the 9,216 / 17,408 totals. The guard
  P + 512 <= 32,768 holds by construction.
- Output cap: min(512, ceiling - prompt) = 512 inside every budget. This is
  the shared cap from the data (REPO_FACTS 5); no arm changes it.
- I-4K escalates; it does not truncate. Code: tools/context_window.py
  context_overflow, which sets stop_reason ctx_overflow in
  tools/bfcl_feasibility_probe.py; the hybrid caller escalates sticky.
  R-4K (silent oldest-message drop) is optional and not built.
- Escalation in this runner is not a transcript handoff. The cloud agent runs
  the entry with its own context (AnthropicCloudBackend), and retention applies
  to its tool outputs exactly as to local ones. The raw-payload sensitivity
  cell C_raw is C + -RetentionCloudRaw: the cloud agent sees FULL outputs.
- Dropped from C: u4 KV, prompt lookup, chunk cap, OBS_MASK K=512/1024 as
  registered.
- Compiled-state checks: A, B, C, I-4K need the arm A verdict (PA, no SDPA
  fallback, i8, no IR override). R-full needs a PA backend and a compiled KV
  f16 readback, else REFUSED.

## 3. Fairness split
- **May differ:** P, the window/guard and the cap derived from them; K and
  the age-mask trigger (C only). Policy shape, marker and freeze rule are fixed
  study-wide.
- **Identical in every arm:** IR and hash, GenAI/OpenVINO 2026.2.1, GPU,
  driver per host, power scheme, Shared GPU Memory Override value (recorded
  from the next window), chat template, thinking off, greedy sampling, parser,
  max_new_tokens 512, 20-step limit, no retries, router and thresholds (10 s
  TTFT, 6 tok/s, n_steps), cloud model, prices and caching policy (none), task
  ids and order, tokenizer.
- **Shared baseline corrections:** decisions for Zach, reported separately from
  any tuned gain, NOT adopted here:
  - (a) the official-harness differences 1-6 of REPO_FACTS 9;
  - (b) thinking off: already true in every arm (no change);
  - (c) an AWQ+SE IR for all arms: moot, since we run the official IR; a v3 pin
    that records scale_estimation False is a decision.

## 4. Metrics and controls (added to section 4 of the 2026-10-16 amendment)
1. Headline reported figure: $ per passed task = total cloud $ / hybrid
   passes, with the pass count and mean $/task beside it. The primary test
   stays the registered one-sided Wilcoxon on paired $/task.
2. Paired per-task table (C vs A, B vs A, C vs B) on $ and pass.
3. Pass non-inferiority C vs A: as registered (Newcombe paired, one-sided 95%
   lower bound > -0.15).
4. Decomposition: as registered, `$C - $A = (E_C - E_A) x p_A + E_C x (p_C - p_A)`.
   It is also reported for C vs B (retention's share).
5. Matched-rate random-escalation control: computed offline with
   tools/aq_sealed_analyses._random_baseline at C's escalated-entry count.
   It needs per-entry local-only and cloud-only pass records for the held-out
   entries. If no cloud-only record exists, the control is reported NOT_RUN;
   it is never estimated.
6. C_raw sensitivity: C with -RetentionCloudRaw, run only if C runs and spend
   allows. It bounds the payload term.
7. Context: report both the physical tokens resident and the raw tool-output
   tokens absorbed before the first forced escalation. Claim only the second
   for C. Generates past 32,768 are counted per arm; any in B or C voids that
   cell.

## 5. Predictions (DERIVED, before any run)
- File: derived/h1_hybrid/demo2/DEMO2_BC_PREDICTIONS.json, sha256
  7256755a30bd1a804bf0c51d1ce4b71dc59c31180024a9aed0aa1ab2b75d47be.
- Generator: tools/demo2_bc_predict.py. It imports the registered generator
  unchanged and reproduces arm A exactly.
- $/task lo / point / hi, then escalations at the point:

| cell | XPS | T2S |
|---|---|---|
| A | 0.098 / 0.119 / 0.161 (3) | 0.026 / 0.052 / 0.157 (0) |
| B | 0.114 / 0.134 / 0.174 (7) | 0.098 / 0.119 / 0.161 (3) |
| C | 0.084 / 0.104 / 0.144 (7) | 0.069 / 0.090 / 0.132 (3) |
| C_raw | = B | = B |
| I-4K | 0.352 / 0.356 / 0.365 (48), registered D-X band | same model value (new, DERIVED) |
| R-full | 0.098 / 0.119 / 0.161 (3) | 0.040 / 0.078 / 0.157 (0.7) |

- C vs A at the point: XPS -12%, under the 25% material floor. T2S +73%: C
  costs more, because the 16,384 budget forces 3 escalations that A does not
  make.
- The model cannot price what C removes. Gold trajectories never exceed 34k
  tokens, so A's runaway turns (87-98k, 8af10211, cddcd9fd) are not in it. The
  model predicts 8 gold local turns over 32,768 for A-T at the point.
- So the registered prediction is that revised C does not materially beat A
  on $ on either host. Any C claim must come from failure behaviour, latency or
  context absorbed (material-win criteria 2-3), or be a MISS.
- Caps (1.5 x hi total; I-4K-X the registered D-X band), one replicate:
  A-X 14.49, B-X 15.62, C-X 12.98, I4K-X 32.80, RF-X 14.49, A-T 14.14,
  B-T 14.49, C-T 11.83, RF-T 14.14. Sum $144.98 of the $275 ceiling. Two
  replicates would be $289.96, over the ceiling: decision for Zach.

## 6. $0 ladder (local-only; DEMO-1 pins or scripts; never DEMO2_HELDOUT_60 / _B)
Every rule is fixed now. A value changes only by these rules.
- **L0, compiled state and host record** (both hosts):
  - tools/kv_compiled_readback.py --mode all --cache-dir <empty dir>;
  - the window scripts' host record (drivers, power scheme, dxdiag display
    memory, Intel GPU registry keys).
  - Rule: arm A stays valid only if PA, i8 and the latency defaults are
    confirmed. If CACHE_DIR changes the key_cache.* port type, it is excluded
    from every arm (as now); if only the property changes, record "reporting
    artifact".
- **L1, ground-truth replay** (tools/demo2_ladder_l1.py, Mac):
  - Rule in the script docstring: the smallest K in {512, 1024, 2048} with
    shape json or h70t30 that keeps 100% of required values; head is never
    chosen.
  - If none reaches 100%: K = 2048 json only if it keeps the most values;
    otherwise retention is dropped (C = B).
  - K and shape in section 2 are replaced by the L1 choice.
- **L4, TTFT and decode vs prompt length** (tools/demo2_ladder_l4.py; arm A
  runtime; grid 1,024-16,384 XPS, 1,024-32,768 T2S; 3 cold + 1 warm per point):
  - P = the largest of {4,096, 6,144, 8,192, 10,240, 12,288, 16,384, 24,576}
    whose cold TTFT is <= 9.0 s at every repeat and decode >= 12 tok/s.
  - XPS: P is capped at 8,192 (the research value). T2S: capped at 16,384
    unless L6 passes 24,576.
  - DERIVED expectation, XPS: 8,192 at ~8.9 s, from c2246b1f's 6.80 s at
    8,000 scaled by a5d19dcc/c2246b1f at 12,000 (16.96 / 13.04 s). This is
    near the 9.0 s line, so an XPS P of 6,144 is possible.
- **L5, output and memory:**
  - The output cap is fixed at 512 by data (REPO_FACTS 5).
  - The XPS memory probe text is read from L4's first failing point, if any,
    and from a5d19dcc.
  - No fixed KV pool is adopted (cache_size is not in any arm).
- **L6, local-only C vs A on DEMO-1 pins** (BASE_30 + LONGCTX_30; interleaved
  local_only, -ContextWindow P+512 -CtxOverflowLocalFail, K and shape from L1,
  age-mask on and off; -StallWatchdogS 900 -TurnWatchdogS 3600):
  - Arm A records are the sealed local-only runs:
    - XPS: 39c1c131 (BASE_30); 841b84e4 + 15367c3a (LONGCTX), with 123
      counted a local fail;
    - T2S: b90f1d5b; 1a634610 + daa6c026.
  - Keep age-masking only if, with it, ctx_overflow events fall >= 25% vs
    without, with no local pass loss and loud parse failures within +-2.
  - C proceeds to held-out only if its local passes are >= A's - 1 on each
    pin and its silent-failure share does not rise (capacity-paradox guard).
  - Generates past 32,768 must be 0.
  - T2S also runs P = 24,576 once (sensitivity).
- **L2/L3:** the harness-parity correction and A/A self-variance are not run
  unless Zach adopts the section 3 corrections.
- **L7:** a small paid hybrid pilot on DEMO-1 pins stays a decision for Zach
  (P7 caps $27.54 approved 2026-10-16 for arm A x2).

## 7. Order
- L0 and L1 first. L1 runs on the Mac, after this note is committed.
- Then L4 and L6 per host (window scripts run_window_xps_ladder.sh and
  run_window_t2s_ladder.sh).
- Then an arm-value note with the ladder values (new dated file).
- Then the P7/L7 decision, then held-out cells. Held-out cells never start
  before that value note is committed.
