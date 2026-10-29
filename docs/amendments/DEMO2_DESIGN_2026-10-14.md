# DEMO-2 design (2026-10-14): tuned configuration vs default hybrid

Design only. $0, no runs, no launch-path change. Not a prereg: the
predictions, the held-out pin and the policy code come in a prereg
committed before any DEMO-2 run. Runners do not open this file. Times are
ET.

Question: on the same two hosts (XPS, T2S), does a configuration picked from
sealed limits beat the default hybrid setup on $/task, without losing task
passes?

Sizing data: analysis/demo2/DEMO2_SIZING_20261014.json, made by
tools/demo2_sizing.py from sealed and committed data only.

## 1. Existing policies (tools/run_h1_hybrid.py at 6898ee8)

| policy | triggers | checked when | stickiness |
|---|---|---|---|
| slo_escalate | ttft_s > 10 or decode_tok_s < 6 (decide_slo_escalate) | step 0 of each user turn | sticky: rest of the entry on cloud |
| emission_escalate | no parseable tool call; at turn end, stop reason no_tool_call or empty_execute | each turn | sticky |
| full_signal_bounceback (R2c) | tool_exec_error, or n_steps >= 5 (BOUNCE_STEP_BUDGET), or no parseable tool call (decide_bounceback) | each turn | one-turn bounce, then local resumes |

- Quality triggers (no_parseable, step_budget) live only in
  emission_escalate and R2c.
- Mechanical triggers (TTFT, decode) live only in slo_escalate.
- No policy combines both.
- A context-overflow trigger existed before H1-3POLICY. It is now only
  decide_slo_escalate_legacy_with_ctx (ctx > CTX_LIMIT_COLD_START 10,000),
  "for sealed-ledger diffs only", and never called.
- A local generation failure ends the entry with no escalation under every
  policy. emission_escalate returns "no escalate" for stop reasons
  generation_error and max_steps. slo_escalate gets no TTFT from a failed
  generate.
- Measured trigger rates (d482c621, multi_turn_base 200, XPS GPU 4B u8):
  - slo_escalate: 0 cloud turns.
  - emission_escalate: 59/200 entries (0.295); 25.0% of turns on cloud;
    triggered by no_parseable 59.
  - R2c: 66/200 entries (0.33); 12.5% of turns bounced; 81 no_parseable,
    11 step_budget, 0 tool_exec.

Minimal composition needed (not implemented): a policy `hybrid_full`. Per
turn, evaluate in this order:
1. Mechanical. Window overflow (any step prompt + max_new_tokens > the
   configured window, checked before each local generate), or ttft_s > 10
   or decode_tok_s < 6 at step 0 (as slo_escalate, keeping the DEMO-1
   decision; the per-step trigger remains row PER-STEP-SLO), or a local
   generation_error. Sticky, because context only grows.
2. Quality. The R2c triggers (no_parseable, step_budget, tool_exec_error).
   One-turn bounce, then local resumes.

Each turn records one reason, mechanical first. Cost caps, the ledger,
cloud-dead handling and seals stay as they are.

## 2. Context-window cap

There is none. The CLI has no window flag: --obs-mask-k, --placement,
--residency and --kv exist, but no window. Overflow today surfaces only as
a local failure (CL_OUT_OF_RESOURCES / generation_error), which ends the
entry.

Scope:
- `--context-window N` (default: off, so every existing profile is
  unchanged).
- Before each local generate, tokenize the rendered prompt (the runner
  already computes step prompt tokens).
- If prompt + max_new_tokens > N, do not generate; escalate with reason
  `ctx>N` (sticky).
- Record N in plan.json and every trigger in the ledger.
- Size: about 40 lines in run_h1_hybrid plus tests (overflow escalates;
  off = byte-identical plan for existing policies; runner does not read the
  prereg). It also needs one launcher passthrough line, plus a smoke on
  each host.

## 3. Default hybrid control (same on XPS and T2S)

| setting | control value | source |
|---|---|---|
| context window | 4096 tokens | Ollama FAQ: "By default, Ollama uses a context window size of 4096 tokens." |
| KV precision | runtime default (no KV_CACHE_PRECISION passed); read back at load and recorded | OpenVINO 2025.1 notes: int8 KV "by default on all GPU platforms"; 2026.2.1 default NOT CONFIRMED |
| backend | LLMPipeline default (PagedAttention per 2025.1 notes); no SchedulerConfig | OpenVINO 2025.1 release notes |
| prefix caching | runtime default, read back if exposed | default NOT FOUND (docs/RELATED_SEARCHES.md) |
| placement | open decision (Zach). Literal default: CPU, per the GenAI chat sample `default="CPU"`. Recommended: GPU, so the control is not a CPU strawman | openvino.genai chat_sample.py |
| retention | FULL (no masking) | no runtime transforms observations |
| residency | NON_RESIDENT (fresh prompt each turn; no session reuse) | the H1 runner's non-session path |
| weights | Qwen3-4B int4 (OpenVINO org IR, as DEMO-1) | configs/models/Qwen3-4B-int4-ov.yaml |
| policy | hybrid_full (section 1), window 4096 | this design |

Searches are logged in docs/RELATED_SEARCHES.md ("DEMO-2 default-control
runtime settings", 2026-10-14). The KV and prefix-caching defaults must be
read back on each host (a property read at load, $0) before the prereg is
frozen.

## 4. Sizing from sealed data (estimates, not predictions)

Gold trajectories (ENV-ATLAS). Overflow = any step prompt + 512 > window.
TTFT edge = step-0 n_ctx > sealed cold edge (XPS 10,000 c2246b1f; T2S
18,500 d2d5cc4a). Tuned window = 40,960 (max_position_embeddings). Sticky.
Mechanical triggers only.

| workload / retention | control 4096: entries / turns on cloud | tuned XPS | tuned T2S |
|---|---|---|---|
| multi_turn_base FULL (200) | 0.685 / 0.616 (all overflow) | 0 / 0 | 0 / 0 |
| multi_turn_base OBS_MASK K=1024 | 0.685 / 0.616 | 0 / 0 | 0 / 0 |
| multi_turn_long_context FULL (200) | 0.950 / 0.905 (all overflow) | 0.205 / 0.136 (33 TTFT, 8 overflow) | 0.200 / 0.132 |
| multi_turn_long_context OBS_MASK K=1024 | 0.950 / 0.905 | 0.200 / 0.131 (40 TTFT) | 0.015 / 0.005 (3 TTFT) |

- Base turn 0 already needs 4,193 tokens on multi_turn_base_0, so a 4096
  window overflows before any tool output.
- Quality triggers come on top (d482c621): 0.295-0.33 of base entries.

Cloud input tokens per escalated entry:

| | atlas estimate (n_ctx x 7.11) | measured (DEMO-1 ledgers) |
|---|---|---|
| base FULL, control | median 99,931 | V2 1acc8c58: median 91,286 (30 billed) |
| long FULL, tuned | median 554,097 (XPS) | V3 T2S a4284ab0: median 122,071 (4 billed) |
| long OBS_MASK K=1024, tuned | median 224,612 (XPS) / 132,225 (T2S) | V4 XPS a678926e: median 70,111 (9); V4 T2S 30caa847: 62,907 (1) |

- The atlas factor agrees with the measured value on base (within 10%).
  It overstates long_context 2-4.5x. The live 4B often does not make the
  large-observation call (Task X counts: 12/16 non-escalated jump entries
  never made it).
- FULL vs OBS_MASK on long_context: masking cuts per-escalation tokens
  about 2.5x (atlas) and about 1.7x (measured median, 122k vs 70k), on
  different hosts and entries.

## 5. Selection rule for the tuned profile (frozen before any run)

Inputs are sealed limits only, never held-out entries. Per device:
- **Placement:** GPU, because the XPS GPU cold limit is 21.4x the CPU's
  (c2246b1f 10,000 vs 7f232f86 468). NPU excluded: the only citable
  NPU-INT8 limit is 1984 (9acabbf7).
- **Weights:** int4 (int8 decodes 30-34% slower, 180dfbb9).
- **Window:** the model window, 40,960 (max_position_embeddings).
- **KV precision:** among precisions with a sealed limit on that device,
  take the one with the largest RESIDENT limit; break ties on warm turn-2
  TTFT.
  - XPS: u4 26,500 (e4a22dac) > u8 20,000 (cf3555d5) > f16 15,000
    (553b3a5c). Warm turn-2 at 12k: u4 0.946 s, u8 1.003 s. So u4.
  - T2S: u8 is the only sealed precision (d2d5cc4a). So u8.
- **Retention:** OBS_MASK K=1024 for any workload whose atlas has an
  observation above K tokens (long_context: yes; base: no, so FULL). This
  is a category property, not a per-entry fit.
- **Residency:** RESIDENT only if the workload's turn-boundary delta D is
  within the sealed small-D warm points (D <= 400; 72776603, 1587d2f4,
  dd2b0779); otherwise NON_RESIDENT.
  - V5-CAL (a178e9c5, e1ed06b4): at D about 8,500 a delta prefill is
    slower than a cold prefill.
  - The DEMO-1 turn-boundary D is 29-118 (median 50.5; V5 addendum), so
    RESIDENT.
  - Within-turn large observations are not checked by the step-0 trigger
    (PER-STEP-SLO); stated, not changed.
- **Policy:** hybrid_full with the tuned window.

## 6. Cells

Held-out pin DEMO2_HELDOUT_60 (new seed, written in the prereg):
- 30 multi_turn_base entries not in DEMO1_BASE_30.
- 30 multi_turn_long_context entries not in DEMO1_PAIRED_15, stratified so
  the share with a gold turn above 18,500 tokens matches the category
  (19.5% of entries, ENV-ATLAS).
- No entry is used by the selection rule.
- d482c621 used all 200 base entries. That run gave trigger rates only,
  not selection.

| cell | host | profile |
|---|---|---|
| D-X | XPS | default control |
| T-X | XPS | tuned (XPS) |
| D-T | T2S | default control |
| T-T | T2S | tuned (T2S) |
| A-win | XPS | tuned, window -> 4096 |
| A-ret | XPS | tuned, retention -> FULL |
| A-kv | XPS | tuned, KV -> runtime default |
| A-res | XPS | tuned, residency -> NON_RESIDENT |
| A-wt | XPS | tuned, weights -> int8 |

The A cells are leave-one-out ablations on XPS.

Outcomes:
- **Primary:** $/task, paired by entry. Tuned minus default per host:
  one-sided Wilcoxon signed-rank on per-entry $, plus a bootstrap 95% CI.
- **Gate on the primary: pass non-inferiority**, paired. The lower bound of
  the one-sided 95% CI for (tuned - default) hybrid pass rate must be above
  -0.15. A $ win counts only if this holds.
- **Margin justification:** 0.15 is about half the smallest
  configuration-driven pass change seen so far (V4 XPS -> T2S, -4/15 =
  -0.27; placement V1 -> V2, +0.63). Power at n = 60 is about 80% if the
  discordance rate is 0.2 and the true difference is 0. At discordance 0.4
  it falls to about 50%. A margin of 0.10 would need n of about 124-247.
  This is an open decision.
- **Secondary:** local share, TTA per turn, cloud input tokens per task,
  per-trigger counts.

## 7. Spend and schedule

Upper bound per cell: per entry, mechanical-escalated entries at the
all-cloud $/task (prereg v2 references: base $0.347, long FULL $2.081, long
OBS_MASK $0.878), and other entries at 12.5% of it (R2c bounce share).
Gold trajectories, so this is a ceiling. The live long_context realized
about 5% of its gold prediction in V3 T2S.

| cell | upper bound (n = 60) |
|---|---|
| D-X, D-T (each) | $67 (base $7.5 + long $59.7) |
| T-X | $9 |
| T-T | $5 |
| A-win | $33 |
| A-ret | $20 |
| A-kv, A-res, A-wt | up to $18 each (no sealed edge per variant on this workload; capped at 2 x T-X) |
| main 2x2 | about $149 |
| ablations | about $108 |
| all | about $257 |

- Remaining DEMO-1 ceiling: $200 - about $13.30 spent - $45 V3 XPS a/b
  caps in flight (V3a launched about 1:25 PM ET today) = about $142 at
  worst.
- The full design exceeds it, and the $55 reserve is not separate money.
  DEMO-2 needs its own spend approval or a smaller n.
- At n = 40 the upper bounds scale to about $99 (main) + $72 (ablations).

Schedule (ET):
- Oct 15-16: implement --context-window and hybrid_full with tests; KV and
  prefix-cache readback on both hosts ($0); write the prereg and pin;
  commit before any run.
- Oct 17: XPS rehearsal (spawn_detached, REHEARSAL_COMPLETE) and T2S
  rehearsal.
- Oct 18-24: XPS cells, one per cold boot, about 1-2 h each: D-X and T-X
  first, then the ablations in priority order A-win, A-ret, A-kv, A-res,
  A-wt.
- Oct 19-25, T2S block: D-T then T-T. The T2S is shared with Rithwik; each
  launch needs your OK. Windows Update may reboot it (as on Oct 14 and
  Sep 21); defer updates for the block.
- Oct 25: closeout and scoring.
- Oct 26: showing.

## Open decisions for Zach

1. Control placement: literal default CPU (GenAI sample) or GPU
   (recommended).
2. Pin size and NI margin: n = 60 with margin 0.15 (recommended), or a
   larger n for 0.10.
3. Spend: approve a DEMO-2 ceiling (main 2x2 about $149 upper bound;
   everything about $257), or cut the ablations.
4. Whether hybrid_full checks TTFT per step (PER-STEP-SLO) or keeps step 0
   as in DEMO-1. This design keeps step 0.
5. Whether to build --context-window and hybrid_full now (about 40 + 60
   lines plus tests) so the Oct 17 rehearsals can run.
6. Whether to pause Windows Update on the T2S for Oct 19-25. This needs
   Rithwik, since the host is shared.
