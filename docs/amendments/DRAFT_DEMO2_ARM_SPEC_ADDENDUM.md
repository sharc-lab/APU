# DRAFT: DEMO-2 arm-specification addendum (to the 2026-10-16 true-default amendment)

Status: DRAFT 2026-10-20 (Task AX). NOT registered. It edits no registered
file, and no registered prediction or threshold changes until Zach approves a
final version and its sha256 is put in the registry. Runners do not read it.
It collects what the pilots have decided so far and the changes that are
pending.

Parent: docs/amendments/DEMO2_AMEND_2026-10-16_TRUE_DEFAULT.md (sha256
d76c1b08...). Pilots: docs/amendments/DEMO2_PILOTS_2026-10-16.md (sha256
eac93a00...).

## 1. Decided by Zach (2026-10-20, in chat)

1. **-CacheDir is dropped for every arm (A, B, C, D) and every pilot.**
   - Evidence: refusal 3a03221e-c54d-4c57-8ee6-c5811a2cf497 (2026-10-16). With
     CACHE_DIR set, a later load in the same worker printed compiled
     KV_CACHE_PRECISION dynamic and DYNAMIC_QUANTIZATION_GROUP_SIZE 0, and the
     arm A check refused (ARM_A_INVALID). Loads without CACHE_DIR print i8 and
     18446744073709551615.
   - It supersedes "CACHE_DIR set in every arm" (amendment section 1), the
     `-CacheDir <dir>` in the A and B launch flags (section 2), and
     `-CacheDir <C>` in pilots steps 5, 7, 8, 12-14 and 25-30.
   - Parity: no arm sets it, so load time is not part of any contrast.
2. **P1 citation superseded.** Pilots row P1 cites "130-255 output tokens
   per local turn". Measured (XPS 39c1c131-f09b-4357-bde3-f39e36947300, arm A
   local_only, DEMO1_BASE_30, 118 turns): median 91, p90 157. The three
   literature notes were corrected in place on 2026-10-20; the registered
   pilots doc is not edited.
3. **P3 before P4.** XPS window 2 runs P3 first, then P4.
4. **Added P3 cell 0 (plan change, not a rule change): true-default resident
   limit.**
   - Command: `run_resident_limit.py --arm gpu_only --high 50000`, with no
     scheduler override, so it loads a plain `LLMPipeline(model, "GPU")`: i8,
     latency scheduler, the same load as arm A.
   - Reason: arm A held step-0 prompts up to 43,961 tokens locally in
     841b84e4-0eb8-436d-89f6-c5b558848991 (UNSEALED partial) and completed 7
     entries above 34,000. The amendment's XPS A trigger uses 20,000.
   - P3's adoption rules are unchanged; they compare against the sealed
     limits as registered.

## 2. Pilot results so far (feed this addendum)
- P0: XPS PASSED (P0a 597f40a9, P0b 640ec474-cb2e-43d0-a0e2-7e41bfe10a0c).
  T2S NOT YET RUN.
- P9: XPS drift +1.52% (687bbb9f-83db-4b9f-8a8e-e994969a7721), so no warm-up
  rule.
- P1: the output-length lever exists. Non-tool-call share 0.652 on BASE_30
  and 0.777 with LONGCTX 14/30 (ESTIMATE). Rule: applied to every arm or none.
- P6: not applied yet (LONGCTX 14/30); the remaining 16 entries run next.
- P8: i8 side recorded; u4 not run.
- Still to fill: u4 kept (P8); chunk cap, cache_size and XPS window (P3);
  prompt lookup (P4); OBS_MASK K (P5); replicates (P7).

## 3. Correction to the premise of cell 0

The 20,000 limit (cf3555d5-1361-434f-b1fb-745fc17e6843, 2026-09-30, arm
gpu_only_u8) was not measured under a bare SchedulerConfig:
- seam/tools/_delta_n_child.py at that date (0e8aa5a) builds
  `LLMPipeline(model_dir, device, **properties)` with no SchedulerConfig. So it
  ran the latency scheduler with a KV u8 property, which compiles i8 (P0a).
- The bare SchedulerConfig is the NON_RESIDENT DEMO-2 control in
  tools/bfcl_feasibility_probe.py, not this tool.
- So the 20,000 vs 43,961 difference is not explained by the scheduler.
  Other differences:
  - workload: a synthetic two-turn resident prompt with a one-shot turn-1
    prefill, vs the H1 agent loop;
  - date: 2026-09-30 vs 2026-10-19; cf3555d5's plan does not record the
    OpenVINO version.
- Cell 0 measures the limit on today's stack with the same tool.

## 4. PENDING, decision for Zach: if the XPS arm A limit is above 20,000

Recomputed with the registered generator (tools/demo2_true_default_predict.py
predict_cost, DEMO2_HELDOUT_60 gold trajectories, registered cost model).
Only LIMITS["xps_i8_resident"] changes. The 20,000 run reproduces the
registered A-X exactly. All values are DERIVED.

| XPS A memory limit | A-X $/task lo / point / hi | A-X mech. escalations (point) |
|---|---|---|
| 20,000 (registered) | 0.098 / 0.119 / 0.161 | 3.0 |
| 26,500 or 30,000 | unchanged | 3.0 |
| 36,000 | 0.085 / 0.107 / 0.160 | 2.0 |
| 40,960 | 0.040 / 0.078 / 0.157 | 0.7 |
| >= 44,000 (covers 43,961 observed) or none | 0.026 / 0.052 / 0.157 | 0.0 |

All three registered A-X point escalations (entries 105, 108, 147; gold
max step 34,084-40,781) come from the 20,000 memory trigger. The hi band
keeps 3 because it applies the 10,000 cold edge on every turn. Measured in
841b84e4 (unsealed): with prefix caching, step-0 TTFT at 34-44k-token turns
was 1.4-12.2 s, and 2 of 17 such turns were above 10 s (entries 101, 117).
So the hi bound overstates later-turn triggers.

What it does to each registered piece. None of these is changed here:
1. **A's mechanical escalations (ctx/memory):** 3.0 -> 0 at point once the
   limit is >= 41,293 (the largest gold step + 512 on the pin).
2. **P6 capacity-paradox band (A limit, B/C window]:** (20,000, 23,850]
   holds 0 of 60 entries on the gold trajectories even at 20,000. With
   A >= 23,850 the band is empty and the rule cannot fire. The reverse case
   then appears: B/C's window escalates entries A keeps local. That is 3
   entries at the 23,850 window and 1 at 36,864 (gold max 34,596, 36,281,
   41,293).
3. **B/C window advantage:** gone on the XPS.
   - B-X point $0.119 vs A-X $0.052 is 2.3x: B costs more, as the amendment
     already predicts on the T2S.
   - Context held: A >= 43,961 (unsealed observation) vs the B/C window
     <= 40,960, so C holds less context than A.
4. **Material-win path (section 6):**
   - $: C-X/A-X = 0.046/0.052 = -11% at point (lo -11%, hi -29%), against a
     needed >= max(25%, 2 x d_usd). The registered point was -62%.
   - Context: cannot hold (item 3).
   - Latency (5-25% DERIVED, floor 20%): unchanged.
   - So on the XPS, a material win would rest on latency or the hi band
     only. Under the pivot rule that points to the SEARCH phase for the XPS.

Proposed addendum changes, for Zach to decide after P3 cell 0 is sealed:
- (a) Set XPS A's memory trigger to the sealed cell-0 limit.
- (b) Recompute the A-X band and the C/A band with (a).
- (c) Restate the P6 rule for the reverse case, B/C escalating where A
  stays local, or record it as not applicable.
- (d) Set B/C's XPS window to min(40,960, max(P3 limit, A limit) - 10%), or
  no window, so B/C do not hold less than A.
- (e) Record whether context held can still serve as a material-win
  criterion on the XPS.
Nothing above changes until a final addendum is registered.

## 5. Arm D label (decided by Zach 2026-10-21, Task BA decision 4)
This section supersedes the arm D label of the registered amendment when the
final addendum is registered. It does not edit that file.
- Registered text being superseded:
  DEMO2_AMEND_2026-10-16_TRUE_DEFAULT.md L63 ("A plus a 4096 window, labeled
  'the configuration typical hybrid studies inherit'") and L89-92 ("ECO-LLM
  ... and Minions run Ollama locally and state no context setting").
  DEMO2_DESIGN_2026-10-14.md L78 (the unversioned Ollama FAQ "4096" quote).
- New label: **"the 4096 window Minions configures for its local model
  (Ollama's current default on a < 24 GiB machine)"**.
- Basis: docs/literature/2026-10-20/OLLAMA_DEFAULT_INHERITANCE.md.
  - Minions' OllamaClient sends num_ctx 4096 at publication (5c2c9c2f,
    minions/clients/ollama.py L16, L59-66).
  - Ollama's default is 4096 below 24 GiB VRAM from v0.15.5 (0334ffa6), and
    4096 for v0.6.7-v0.15.4.
- Arm D's configuration (A plus a 4096 window, XPS only) is unchanged. Only
  its label and justification change. The word "inherit" is not used.

## 6. Pilot results added 2026-10-20 (Task AZ; analysis/demo2_pilots/AZ/)
- P6 NO_CHANGE (0 of 60 in the band).
- The XPS true-default resident limit is 18,000 (a5d19dcc). Item 4's premise
  ("A limit above 20,000") does not hold, so changes (a)-(e) use 18,000.
- T2S:
  - arm A valid;
  - P8 keep_u4 false;
  - P4 no prompt lookup;
  - P9 no warm-up;
  - P2 u4 gain supported (1.082 at 12k).
