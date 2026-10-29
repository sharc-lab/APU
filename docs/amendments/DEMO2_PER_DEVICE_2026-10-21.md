# DEMO-2 per-device profiles, 2x2 cross-transfer, emulated caps, T2S ladder R0-R6 (Task BC addendum 2)

Registered 2026-10-21, BEFORE any run it governs (no R0-R6, P_T2S-cap, 2x2 or
emulated-cap cell has run). Approved by Zach 2026-10-21 ("yes").
Basis: ~/seam_closeout_1013/research_20261021/"Hardware specific tuned profiles.md".
Sibling of DEMO2_ARMS_REVISED_2026-10-21.md (17a102e); neither file is edited.

Order, stated plainly: this request arrived after DEMO2_ARMS_REVISED was
committed and after ladder L1 ran (65bb249). This note changes nothing L1
decided. Its R1 budget rule replaces that note's L4 budget rule (section 6, L4)
on both hosts; no L4 point has run.

## 1. Profiles (OpenVINO GenAI 2026.2.1 pinned on both hosts)
Budget rule, the same for both hosts. B = the largest multiple of 1,024 whose
cold first-prefill TTFT p90 (3 repeats, nearest rank, so the maximum) is below
9.0 s with decode >= 12 tok/s. It is capped at 8,192 on the XPS, at 16,384 on
the T2S (quality), and at 32,768 everywhere (hard). Measured by R1 with
tools/demo2_ladder_l4.py --refine-ttft-s 9.0; the window is
-ContextWindow B+512.

| profile | definition |
|---|---|
| P_XPS | revised arm C on the XPS: true default runtime, budget by the rule (cap 8,192), output cap 512, retention from L1 (K 1,024 h70t30), lazy age-mask, i8 KV |
| P_T2S-fit | revised arm C on the T2S: the same with the T2S budget by the rule (cap 16,384) |
| P_T2S-cap | P_T2S-fit + -Kv u4 + -PromptLookup -NumAssistantTokens 3 -MaxNgramSize 3 (P4's best: 4d8e2fb1, 1.212x) |

- K: the addendum text says K = 2,048. Registered L1 chose 1,024 h70t30, and
  every profile inherits arm C's K. Going back to 2,048 is a decision for Zach.
- **u4 and prompt lookup failed arm C's registered rules and do NOT re-enter
  arm C.** u4 failed keep_u4 (P8, T2S 27/59 vs 26/59 silent); prompt lookup
  failed byte identity (P4, 20/77 and 19/58 steps differ). They appear only in
  P_T2S-cap, a new arm with its own rule.
- **P_T2S-cap acceptance** (task-level equivalence vs P_T2S-fit, on
  DEMO2_CONFIRM_200, paired, local-only):
  - (a) pass non-inferiority: Newcombe paired 95% lower bound of
    pass(cap) - pass(fit) > -0.05;
  - (b) McNemar exact (two-sided) on the discordant pairs, reported; a
    lopsided b - c >= 3 entries in fit's favour drops the arm;
  - (c) A/A noise floor from R2: arm A run twice; per-entry discordance and the
    step-level AST mismatch rate;
  - (d) per-call errors (wrong function name, wrong argument) within the A/A
    floor;
  - (e) speed: median per-step decode time ratio fit/cap >= 1.10, paired
    bootstrap 95% CI excluding 1.0 (seed 20261021);
  - (f) zero generates past 32,768.
  - Byte identity is a diagnostic only. If (a)-(f) do not all hold, the T2S
    profile is P_T2S-fit.
- Compiled-state rule (R0), before any cell:
  - P_T2S-cap builds read back the compiled KV (u4) and the full PA property
    dump; every other field must equal arm A's dump on the same host, else
    REFUSED.
  - The same check applies to a build with prompt_lookup only and with u4 only,
    so a silent scheduler or property change from either is caught.

## 2. Separate axes (exploratory; never mixed into A/B/C/I-4K/R-full)
- **Model tier.**
  - 4B and 8B INT4 on both hosts as the control.
  - 14B and 30B-A3B INT4 are T2S-only. The XPS cell is marked INFEASIBLE after
    one attempted load, with its error recorded.
  - Only 4B and 8B are pinned. The 8B pin is INT4_ASYM, AWQ false, SE true, so
    its recipe differs from 4B.
  - 14B and 30B-A3B: not pinned, no download without Zach's OK; NOT_RUN until
    then.
- **Thinking budget** B in {0, 128, 256, 512}, through the chat template, with
  one rule on both hosts. The two-phase budget harness is **not built**, so R3
  and R4 are NOT_BUILT and skipped.
- **Observation, not an arm:** OpenVINO 2026.3+ makes u4 the default for INT4
  PagedAttention models. Our data: u4 helps decode on the T2S at 12k (+8%,
  29209e26 vs 0a55c316) and slows it on the XPS (-4-6%, P8/AY). "The next
  default helps one chip and hurts the other." The version stays pinned at
  2026.2.1.

## 3. 2x2 cross-transfer (device x profile)
- Cells: {XPS, T2S} x {P_XPS, P_T2S-fit, P_T2S-cap}, plus arm A per device.
- Run on DEMO2_CONFIRM_200, local-only (interleaved local_only,
  -CtxOverflowLocalFail for the budget profiles), identical flags apart from
  the profile.
- Per-entry metrics: median local step wall time; count of router-trigger
  events computed offline from the step records by the registered rules (step-0
  TTFT > 10 s, decode < 6 tok/s, ctx_overflow); local pass. Cloud $ only if a
  hybrid cell is approved later.
- **Crossing** holds if, on each device, the own-profile cell beats the other
  device's profile on at least one of:
  - median step wall time: paired log-ratio, 95% bootstrap CI excluding 0;
  - trigger-event count: paired Wilcoxon one-sided p < 0.05;
  - and it is not worse on the other (CI or p in the opposite direction).
- Passes and $ are reported beside the crossing, never used to decide it.
- The XPS half runs after the T2S hand-back on the same versions, entries and
  rules.

## 4. Emulated memory caps on the T2S (R5; labelled EMULATED)
- Arm A runtime on DEMO2_GROWTH_10 (sha256 a223ebd1...), local_only, with a
  complete explicit SchedulerConfig:
  - -SchedPrefixCaching on -SchedMaxBatchedTokens 18446744073709551615
    -SchedDynamicSplitFuse on -SchedCacheSize {4, 8, 12} -CompiledReadback;
  - plus dynamic (arm A, no config).
- Per entry: the failure type (loud error with its text / silent generate past
  32,768 / loop at 512 x 5 / ok).
- "Guardrail mechanism supported" if the share of entries with a silent past-
  32,768 generate does not fall as the cap rises (4 <= 8 <= 12 <= dynamic) and
  dynamic > 4 GiB.
- An out-of-pool error here is not the XPS allocation error.

## 5. Power and entry sets (DERIVED)
- 3/30 has Wilson 95% [0.035, 0.256]: a 30-entry slice cannot detect < ~15
  points.
- 200 paired entries (McNemar): +10 points (2% lost, 12% gained) at ~0.97
  power; +5 points (2% vs 7%) at ~0.59.
- The device-gain difference-in-differences needs >= 10 points for 80% at
  n = 200.
- Latency and count metrics detect a 1.1x step-time ratio at n = 60.
- DEMO2_CONFIRM_200: sha256
  fdd3d404f474888fa86b8c3b0de8cd11d1f2a7b1f6cbb720192da279773b8753
  (tools/demo2_per_device_pins.py, seed 20261021).
  - 100 base of 110 eligible and 100 long of 103 eligible.
  - It excludes DEMO2_HELDOUT_60, DEMO2_HELDOUT_60_B and every DEMO-1 pin, the
    entries that motivated u4 (P8) and prompt lookup (P4). Overlap checked: 0.
- DEMO2_GROWTH_10: the 10 DEMO1_LONGCTX_30 entries with the largest T2S arm A
  prompts: 123, 138, 117, 135, 112, 137, 116, 114, 101, 132.

## 6. T2S $0 ladder R0-R6 (~/seam_closeout_1013/xps/run_window_t2s_ladder.sh)
Schedule (ET):
- start Thu Oct 22 9 AM; finish by Sat Oct 24 10 AM;
- registration freeze Sat noon;
- logs off the T2S by Sun 6 PM; hard stop 9:44 PM Sun Oct 25.
- Drop order if time slips: R5, then R6's fourth arm (P_XPS-on-T2S), then R4.
- Execution order in the script: R0, R1 (i8, then u4), R2, R6 arms 1-3, R6 arm 4, R5. Running R5
  last makes the drop order automatic. A cell starts only if its reserve ends before FINISH.
- B is read from R1's run.json (refined_largest_pass, capped at 16,384). P_XPS-on-T2S uses the
  XPS cap, 8,192 (window 8,704), because the XPS R1 has not run.
- The R6 P_T2S-cap levers are those R2 keeps, passed as CAP_ARGS, which must name -Kv. Without
  CAP_ARGS the cap arm is NOT_RUN; it is never guessed.

| rung | what | time (DERIVED) | rule |
|---|---|---|---|
| R0 | read-only host record (drivers, power, CPU name, dxdiag, Intel GPU keys), kv_compiled_readback --mode all --cache-dir, L4 --readback-only builds: A, u4, prompt_lookup, u4+prompt_lookup | 0.5 h | section 1 compiled-state rule; A valid only with PA, i8, latency scheduler |
| R1 | L4 cold/warm TTFT + decode at 2k, 4k, 8k, 12k, 16k, 24k, 32k, then the 1,024 refinement; i8, then u4; 3 repeats; 2-min warm-up | 1.5-2.5 h | budget rule (section 1). u4 provisional if decode at the budget depth >= +5% and cold TTFT <= +3% vs i8 |
| R2 | 60 DEMO-1 pins (BASE_30 + LONGCTX_30) local-only: A, A again, P_T2S-fit, +u4, +PL3, +PL5, +u4+PL3 | 8-9 h | keep PL at the K with the best median step tok/s if >= +10% and AST mismatch <= 2x the A/A floor; keep u4 if per-step gain >= 5%; drop any lever with b - c >= 3 entries |
| R3 | cost microbenchmark (N samples, thinking B) | - | NOT_BUILT |
| R4 | BASE_30 at the realized B | - | NOT_BUILT |
| R5 | section 4 | 2-3 h | section 4 |
| R6 | DEMO2_CONFIRM_200 local-only: A, P_T2S-fit, P_T2S-cap (with the R2 levers), P_XPS-on-T2S | 12-14 h | section 1 acceptance; 2x2 T2S half |

- The A/A floor and P_T2S-cap's levers come from R2.
- If R2 keeps neither lever, P_T2S-cap = P_T2S-fit and R6 runs three arms.

## 7. Facts
- The 43 generates past 40,960 (BB, context_over_limit.json):
  - XPS 17: 841b84e4 4, 162a9b5c 4, 15367c3a 9;
  - T2S 26: 1a634610 4, 3aa565da 4, daa6c026 9, 108d5f43 4, 8af10211 5.
  - 2026-10-21 added 40 more on the T2S: cddcd9fd 20, 02ade252 20 (steps 1-20,
    87,392-98,165).
- T2S CPU model: NOT YET MEASURED. Get-CimInstance Win32_Processor is in R0's
  read-only step.

## 8. XPS ladder (next XPS window; ~/seam_closeout_1013/xps/run_window_xps_ladder.sh)
- R0-XPS: host record and readback.
- R1-XPS: L4 at 2k-16k with the 1,024 refinement (i8; u4 for the observation).
- L6 on the XPS: section 6 of DEMO2_ARMS_REVISED, with P_XPS.
- R2/R6-XPS and the 2x2 XPS half come after the T2S hand-back, same rules.
