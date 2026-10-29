# DEMO-2 dollar win at iso-quality: shared corrections, resident arm A, device-provisioned arm P (Task BE)

Registered 2026-10-21 (ET), BEFORE any run it governs: no calibration log, held-out
local-only log, cloud-only log or P cell has run. Approvals:
docs/amendments/APPROVALS_2026-10-21.md (Zach, 9:45 PM ET). Basis:
~/seam_closeout_1013/research_20261021/"Dollar win against the default.md".
Siblings, none edited:
- DEMO2_AMEND_2026-10-16_TRUE_DEFAULT.md;
- DEMO2_ARMS_REVISED_2026-10-21.md;
- DEMO2_PER_DEVICE_2026-10-21.md.

What changes:
- The primary evaluation of the DEMO-2 dollar claim becomes iso-quality $, by offline replay
  (section 4). The registered live-hybrid endpoints (Wilcoxon on paired $/task, Newcombe
  non-inferiority, decomposition, material-win rule) are kept as secondary endpoints for any
  live hybrid cell that runs.
- Arm A becomes RESIDENT.
- Arm P is added.
- Every other registered rule, threshold, pin and arm definition is unchanged.

## 1. Models (facts; analysis/demo2_pilots/BE/MODEL_IRS.md)
| IR | revision | rt_info recipe | card | hybrid |
|---|---|---|---|---|
| Qwen3-4B-int4-ov (A, B, C, I-4K, R-full) | b467368d | INT4_SYM g128, AWQ True, SE False, dataset null | AWQ + SE + wikitext2 | yes |
| Qwen3-8B-int4-ov (P on XPS) | 5c47abf4 | INT4_ASYM g128, AWQ False, SE True, dataset wikitext2 | same | yes |
| Qwen3-4B-int8-ov (P' on XPS) | fd9c3cc8 | INT8_ASYM per-channel, data-free | INT8_ASYM | yes |
| Qwen3-8B-int8-ov (P on T2S) | f82c978d | INT8_ASYM per-channel, data-free | INT8_ASYM | yes |

- All four use 40,960 positions, no RoPE scaling, and byte-identical chat_template.jinja,
  tokenizer.json and tokenizer_config.json, so the rendered prompts have the same token
  counts.
- The 8B INT4 recipe differs from the 4B INT4 (ASYM, no AWQ, SE on). This is part of the
  "device-provisioned" label. No published 8B IR matches the 4B recipe.

## 2. Shared corrections (identical in every arm; reported separately, never as an arm's gain)
- Resident pipeline: one LLMPipeline per run (--resident-pipeline). One 1-token warm-up on
  each tool set's prefix before its first entry (--warm-toolset-prefix; toolset_warmups.jsonl).
- Run order grouped by tool set (--order-by-toolset; stable, first-appearance order),
  identical across arms.
- Prompt caching on: --caching-policy ephemeral.
- Cloud-side tool-output retention K = 1,024 h70t30 (the L1 choice):
  - arms without local retention (A, B, I-4K, R-full, P): --retention-k 1024
    --retention-shape h70t30 --retention-cloud-only;
  - C: its registered local retention, which also applies on the cloud side.
- Unchanged: cloud model, price sheet, sampling, max_new_tokens 512, the 20-step limit and
  router thresholds.
- Safety bound (BD rule, registered now): a stall (stall watchdog) or a turn-budget overrun
  in any arm is a local error at that turn, a sticky mechanical trigger in replay.
  - Limits: stall 900 s on the T2S and 3,600 s on the XPS; turn budget 3,600 s on the T2S
    and 14,400 s on the XPS (as P11-B).
  - The queue resumes past the entry with --skip-hang-record.

## 3. Arms per host (local-only passive logs for every arm; live hybrid cells are stretch)
| arm | XPS | T2S |
|---|---|---|
| A (resident) | 4B INT4, true default + section 2 | same |
| B | A + budget (-ContextWindow 8704) | A + budget (R1 rule; -ContextWindow B+512) |
| C | B + local retention 1,024 h70t30 + age-mask | same |
| P (device-provisioned) | **Qwen3-8B INT4**, C's device settings | **Qwen3-8B INT8**, C's device settings |
| P' (XPS, by rule) | Qwen3-4B INT8, C's device settings | - |
| I-4K | A + -ContextWindow 4096 | A + -ContextWindow 4096 |
| R-full | f16, explicit scheduler, -ContextWindow 40960 | same |
| P-think (T2S secondary) | - | NOT_RUN: no thinking-budget harness |

- **P' rule** (fixed now; calibration pin DEMO2_CALIB_30 only):
  - XPS P' replaces P if P's count of calibration entries with a passive step-0 TTFT trigger
    (`ttft>10.0` on any turn) is greater than A-resident's count on the same entries.
  - Otherwise P stays. The rule is applied once, at the freeze, and both counts are recorded.
- Label everywhere: "device-provisioned (model tier/precision), not model-fixed".
- P is never folded into C.

## 4. Primary evaluation: $ at iso-quality by offline replay (tools/demo2_iso_quality_replay.py)
- One shared cloud-only log of DEMO2_HELDOUT_60:
  - single-policy cloud_only, collected once, after the freeze;
  - caching ephemeral, cloud retention as section 2, the registered cloud model.
- One local-only passive log per arm per host (interleaved local_only, --passive-router),
  run to the end. Hangs count as section 2's local error.
- Replay policy (identical for every arm):
  - base = escalate sticky at the first local turn with any passive trigger;
  - the merged trajectory (local turns before t + cloud turns from t) is scored with the
    BFCL checker;
  - cost = the cloud log's cost of turns t..end; no trigger = the local-only outcome at $0.
  - The frontier adds "up" (extra entries escalated at turn 0, random order) and "down"
    (triggered entries thinned, random order), seed 20261023.
- Endpoints, per host, vs A (resident):
  - **primary dC_iso = C_A(Q*) - C_X(Q*)**, Q* = A's base pass rate;
  - dQ_iso = Q_X(C_A) - Q* with exact McNemar at the matched-cost point;
  - $ per passed task at base points (only where pass >= Q*); the area difference on a
    50-point cost grid.
  - Paired bootstrap, B = 10,000, the same entry indices for every arm and the cloud log.
    Two-sided p for dC_iso from the bootstrap.
  - Holm over all arms x endpoints x hosts (one family per host file, combined in the
    analysis).
- **Claim rule:** "X is cheaper than A at A's quality" only if dC_iso > 0 with its Holm-adjusted
  p < 0.05 and the 95% CI excluding 0. Otherwise report the point estimate and CI as
  inconclusive. "Frontier dominance" is never claimed from pointwise intervals.
- No router score threshold is tuned. The policy family has no free score, so only the P'
  rule uses calibration.
- Calibration entries (DEMO2_CALIB_30, sha256 28f00f54..., 15 + 15 from DEMO2_CONFIRM_200,
  0 overlap with held-out) never enter a held-out number.
- Replay validity:
  - the A/A discordance floor (A twice on calibration);
  - live-vs-replay agreement on the stretch live hybrid subset, if run (reported, not gated).

## 5. What the logs must contain (refused at replay if missing)
- Local, per turn: decoded_steps, passive_triggers (mechanical, quality, ttft_s, decode_tok_s,
  n_steps, generation_error, parse and tool-error flags), ctx_overflow. Per step
  (step_log, -StepLog -DebugRawText): prompt tokens, TTFT, generated tokens, raw text.
  Per entry: local_pass, stop_reason, status (hang/stall). Run: toolset_warmups.jsonl and the
  plan (model spec, ir_sha256, compiled readback, runtime options, task_be modes, order).
- Cloud, per turn: decoded_steps, cloud_usd, cloud_requests (input / cache-read / cache-write /
  output tokens, model id), the price-sheet date and caching policy in the plan. Per entry:
  hybrid_pass.
- Manifest: driver, power scheme, host record (R0), git head, seeds.

## 6. Predictions (DERIVED; derived/h1_hybrid/demo2/DEMO2_BE_PREDICTIONS.json, sha256
f54b5ad7990edef86adf486c4857f4da068c5662b171c03c7265a8cc9f40e293; tools/demo2_be_predict.py)
- Inputs: p_A = 4/60 per host; cloud pass 22/30 at $0.2968/task (1acc8c58); f_A from the
  registered A $/task.
- XPS: Q* 0.334.
  - P at 1.4x p_A: dC_iso 6% ($0.007/task), dQ +1.6 pp.
  - P at 2x p_A: 17% ($0.020), +4.0 pp.
- T2S: Q* 0.184.
  - P at 1.4x: 20% ($0.010), +2.2 pp.
  - P at 2x: 52% ($0.027), +5.5 pp.
- Arm A resident (gold generator, cold edge removed): XPS 0.098 / 0.119 / 0.161;
  T2S 0.026 / 0.052 / 0.103. Same point as registered.
- Odds (judgement, adopted from the research report):
  - point iso-quality win for P: T2S 65-75%, XPS 55-60%;
  - CI excluding 0 at n = 60: T2S 25-35%, XPS 15-20%;
  - a raw $/task win by the registered C on the T2S: < 5%.
- Registered expectation: P's dC_iso point estimate > 0 on both hosts; CI excluding 0 not
  expected at n = 60.

## 7. Spend
- Calibration cloud-only log: cap $20.
- Held-out cloud-only log: cap $20, once.
- Live hybrid cells: only within the registered 1-replicate caps, and only if time remains.
- Ceiling $275 unchanged.
