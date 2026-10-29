# DEMO-2 SEARCH phase (contingency), registered 2026-10-16

Status: REGISTERED 2026-10-16 (Task AS), before any DEMO-2 run. Adds to
docs/amendments/DEMO2_AMEND_2026-10-16_TRUE_DEFAULT.md; edits nothing.
Runners do not read this file. Its sha256 is in the DEMO2-SEARCH registry
row.

## 1. When it starts

Only under the pivot rule of the true-default amendment (section 6): on a
host where full tuned (C) does not materially beat the true default (A),
DEMO-2 records "vendor defaults capture most of the single-device gain;
remaining levers: <measured>", and this phase starts for that host. The
rule is fixed now and is not changed after results are seen.

## 2. Exploration rules

- Entries: DEMO-1 pins (DEMO1_BASE_30, DEMO1_LONGCTX_30, DEMO1_LONGCTX_15,
  DEMO1_LONGCTX_OBSK1024_15, DEMO1_PAIRED_15) and any other entry that is
  in neither DEMO2_HELDOUT_60 nor DEMO2_HELDOUT_60_B. The base half of
  DEMO2_HELDOUT_60_B is not yet drawn (registry DEMO-2 row, rules R1-R3).
  Until it is, exploration on multi_turn_base uses DEMO1_BASE_30 only. If R1
  is chosen, those 30 ids are excluded from exploration and no W-3 /
  d482c621 per-entry outcome for them is read.
- Any number of configurations, in any order, on either host.
- Every exploration run is logged as EXPLORATORY: its plan.json and the
  registry say so, and no exploratory number is reported as a DEMO-2
  result. Cloud spend still needs Zach's approval per run.

## 3. Confirmation

Exactly one run of the chosen profile and one run of arm A on
DEMO2_HELDOUT_60_B (the pin's own entries, paired), with the same tests
as DEMO-2: the non-inferiority test (Newcombe paired score interval,
one-sided 95% lower bound > -0.15) and the material-win criteria of the
true-default amendment (section 6), with the noise terms from P7.
Launches use `-Entries <DEMO2_HELDOUT_60_B> -ConfirmationRun`; the runner
refuses that pin otherwise. No second confirmation run on the same pin.

Chosen profile, fixed now: among explored configurations whose exploration
runs pass non-inferiority against A on the same entries, the one with the
largest margin over the first material-win criterion it meets; ties go to
the configuration that changes fewer settings from A. The choice and the
exploration evidence are committed before the confirmation launch.

## 4. Angles to search, ranked

Axis tags: DEVICE (device configuration), AGENT (agent/software),
CLOUD (cloud-side). "Policy change" = the angle changes hybrid_full, so it
is a separate cell and is never credited to configuration alone.

| rank | angle | needs | reuses | axis | policy change |
|---|---|---|---|---|---|
| 1 | Chunk cap + preallocated KV (`max_num_batched_tokens`, `cache_size`) to lift context held | P3 limits and cold-TTFT cost; scheduler flags (Task AS part 6) | P3, NONRES-CHUNK, RESIDENT-LIMIT | DEVICE | no |
| 2 | OBS_MASK K and handoff retention | P5 K sweep; obs_mask ledger | P5, OBS-COST, DEMO-1 V4 | DEVICE (retention; a reviewer may call it AGENT, so B is reported beside it) | no |
| 3 | Asymmetric retention: full context kept locally, compacted context sent to the cloud on escalation (Handoff Tax 2608.24358) | runner option for a separate cloud-side view of history; compaction rule | OBS_MASK code, P5 | AGENT | yes: it changes what an escalation sends. Credited only as a two-arm sensitivity (A + compaction vs profile + compaction) |
| 4 | Per-step adaptive configuration: retention and window chosen from predicted n_ctx at each step ([private] joint decision in miniature); KV precision per entry, since it needs a reload | n_ctx predictor; per-step switch in the session | [private]-PREDICTIVE-ROUTER, P6 n_ctx tail, hybrid_full ctx_overflow pre-check | DEVICE (dynamic) | yes |
| 5 | Pre-emptive escalation on a predicted observation jump (feasibility boundary; [private]) | jump predictor from tool name and arguments | P6, ENV-ATLAS gold jumps, hybrid_full | AGENT (routing) | yes |
| 6 | Prompt lookup and output-length control | P1 output breakdown; P4 speedup and identity | P1, P4 | prompt lookup DEVICE (runtime); output length AGENT, applied to every arm | no |
| 7 | Local competence: model tier (8B on the T2S) and thinking budget | Q-TIER-CLEAN (8B quality), P1-quality thinking line | Q-TIER-CLEAN, P1 A4 | DEVICE (model; breaks "model held fixed", so reported apart from the H-C claim) and AGENT (thinking budget) | no, but a separate cell |

## 5. Rejected, with reasons

- f16 KV as the default tuned setting: with prefix caching a warm turn
  prefills a few hundred tokens, so f16 saves 0.05-0.3 s of TTFT and loses
  0.8-2.6 s of decode per 200-token turn (DERIVED); it lowers the XPS
  resident limit to 15,000 (553b3a5c) against the default's 20,000
  (cf3555d5).
- INT8 weights: decode 30-34% slower (180dfbb9), about +1.76 GB, and fewer
  loud failures (no parseable call on 7 entries vs 32 for INT4;
  1d8db970 vs 6225d6e1), which removes bounces that buy cloud passes.
- XAttention: f16/i8 only (no u4), no speculative decoding, and about half
  of attention is dense at 4k (source, 2026.2.1).
- Cache eviction (and TriShape): eviction turns off micro-SDPA prefill and
  does not cut prefill; TriShape does nothing with unlimited chunks.
- Grammar-constrained decoding: raised well-formed-but-wrong tool calls
  from 8.5% to 52% (Constraint Tax 2605.26128); it would blind the
  no-parse trigger.

## 6. Sensitivity arms only (applied to both arms or neither)

- Cloud prompt caching (Anthropic cache_control): cuts the re-sent
  histories of A more than those of a masked profile.
- Compact handoff: same direction.

Each is reported as a labeled sensitivity pair and is never part of a
primary comparison.
