# DEMO-1 amendment 2026-10-13c: configuration on fixed hardware is the primary axis; V5 (residency, KV precision)

Registered before any V5 run and before any V5 code the runner reads (the
-Kv / --kv selection is built after this file is committed). It adds to
derived/h1_hybrid/demo1/DEMO_PREREG_v2.json and docs/DEMO1_DESIGN.md and
edits neither prediction. tools/run_h1_hybrid.py and tools/launch_h1.ps1 do
not open this file (tests/test_h1_demo1.py). Decision on gate 3a: Zach,
2026-10-13, option A.

## 1. Thesis (replaces the DEMO1_DESIGN.md headline)

"On a fixed device, the inference configuration (placement, residency, KV
precision, weight precision, retention, model export) sets the context range
in which a local turn meets the latency SLO. A hybrid agent's routing, cloud
spend and latency follow from where its trajectory lands relative to that
range. Configuration changes on the same hardware move hybrid outcomes by
amounts comparable to a hardware upgrade, and the size of each effect,
including when it is zero, can be predicted before the run from sealed
limits and the workload's trajectory."

Hypothesis, not a finding: "these knobs are commonly held fixed or
unreported in hybrid-execution studies". It needs search evidence logged in
docs/RELATED_SEARCHES.md before any paper use (registry RELATED-LOG).

## 2. Design

Primary axis: one-factor configuration contrasts on the XPS GPU 4B.
- Placement: V1 XPS GPU vs V2 XPS CPU (DEMO1_BASE_30).
- Retention: V3 (FULL) vs V4 (OBS_MASK K=1024) on DEMO1_PAIRED_15, on the XPS
  and on the T2S (retention x platform interaction).
- Residency, then KV precision: V4 XPS (NON_RESIDENT u8) -> V5a (RESIDENT
  u8) -> V5b (RESIDENT u4). All on DEMO1_PAIRED_15, slo_escalate, OBS_MASK
  K=1024, XPS GPU 4B int4.
Yardstick: platform pairs at matched configuration (V1, V3, V4: XPS vs T2S).
No new T2S cell in this amendment. Follow-on for "does configuration help on
the stronger box too": RESIDENT-T2 and SPACE-SHAPE.

## 3. Gate checks

3a, prefix stability (redefined; Zach 2026-10-13, option A). OBS_MASK is
append-only (no break). The Qwen3 template drops the empty 4-token think
block from earlier assistant messages when thinking is off, so the prefix
holds up to (previous prompt - 4 tokens) at every step: 0/95 exact, 95/95
within 4 tokens (analysis/demo1/PREFIX_CHECK_K1024.json, sha256
5922fdec7d8cbda3d163d9e88781b7e1ca322e3384593e4b69ab880a5ca3d99a, 7326990).
Finding: residency's payoff depends on a chat-template detail.

3b, RESIDENT = CB prefix caching ON (load_arm_pipeline(enable_prefix_caching=
None)); the GenAI ChatHistory is rendered with the same template. Within an
entry, cached blocks are reused up to the divergence; the previous reply,
the 4 think tokens, the new observations and the partial block are
recomputed. begin_entry builds a new LLMPipeline (reload_prefix_cache), so
nothing crosses entries and each entry's step 0 is a cold prefill.

3c, KV precision: arm_id_for_h1 maps gpu_only u8/u4/f16 and the KV readback
refuses a mismatch. Built after this file: run_h1_hybrid --kv, launch_h1
-Kv u8|u4|f16, default u8 unchanged.

## 4. Predictions

Script: tools/demo1_v5_predict.py (sha256
ff8e8cffaae3b7d42b29c320c7e8674ecac4429e1a6abc6cacc40dcbd26230af). Output:
analysis/demo1/V5_PREDICTIONS.json (sha256
f76ea63afcf2e3c8d30103e23b1c2ee595a26c66ec74b9628847a9253044abe9). Sealed
data only.

Inputs (sealed run_ids):
- Cold prefill: u8 c2246b1f (8,000-12,000) + CAP-4 2b3316b6 (12,000+); u4
  c647f0c7 (8,000-12,000) + 2b3316b6.
- RESIDENT turn-2 (delta prefill on a cached prefix), grid (P cached, D new):
  u8 a784f5ec (P 2,000/5,000/12,000; D 150), 48779b5a (P 2,000/4,000/12,000;
  D 400), 1587d2f4 (12,000; 183), cf3555d5 (12,000-20,000; 183); u4 dd2b0779
  (12,000, 24,000; 183), e4a22dac (12,000-26,500; 183). Excluded: 9f38eb15,
  d5c98342 (superseded, KV not pinned).
- Resident limits: u8 20,000 (cf3555d5), u4 26,500 (e4a22dac).
- Decode: sealed >= 14.4 tok/s in the limit runs (DEMO_PREREG_v2
  trigger_mix); the decode trigger is not predicted to fire.
- The sealed warm points used exact prefixes; the reply and think recompute
  is modeled by D, not by the warm data.

Rule:
- Entry step 0: cold prefill at its prompt tokens (log-log between measured
  cold medians; below the smallest, bounded by it; above the largest,
  outside the grid -> escalate, flagged).
- Every later step: delta prefill of D = (prompt - common prefix) on a cached
  prefix of P = common prefix, both from PREFIX_CHECK_K1024.json. TTFT is
  bounded by the smallest sealed turn-2 TTFT among points with P' >= P and
  D' >= D at the same KV precision. The measured points are not a
  rectangular grid and most D lie below the smallest measured D, so
  "interpolation within the grid" is implemented as this monotone upper
  bound (delta prefill is non-decreasing in P and D).
- Outside the warm grid: bounded by the cold prefill of the full prompt (a
  delta prefill does no more work than a cold prefill of the same prompt).
  If that bound is also > 10 s or outside the cold grid, the step escalates
  and is flagged. Variant "literal" (outside the warm grid always escalates)
  is reported, not registered.
- Bracket per step: TTFT in [cold(D), cold(P + D)] from the sealed cold
  curve. Entry "local" if every upper bound <= 10 s, "escalates" if some
  lower bound > 10 s, else "indeterminate".
- Prompt above the resident limit: cold step. Escalate if TTFT > 10 s;
  slo_escalate is sticky per entry.
- Block rounding: the KV block size is not recorded in any sealed load meta
  (checked: d482c621 plan load_meta has scheduler_config only), so the high
  bound recomputes one extra block of 256 tokens (a cover-all for GPU block
  sizes); the low bound 0. V5 records the readback if the API exposes it.
- Cost and local share as DEMO_PREREG_v2 (per user turn; cloud $ = 3.20e-6
  x 7.11 x n_ctx). Passes are recorded, not predicted.

Result (both block bounds identical):

| cell | local share | $/task (x1.083) | escalations | bracket | local prefill bound (s, total) |
|---|---|---|---|---|---|
| V5a RESIDENT u8 | 0.164 | 0.766 (0.829) | 15/15, all flagged | 15 indeterminate | 153.7 to 207.2 |
| V5b RESIDENT u4 | 0.164 | 0.766 (0.829) | 15/15, all flagged | 15 indeterminate | 157.1 to 229.1 |
| literal variant, V5a | 0.036 | 0.801 | 15/15 | - | - |
| reference: V4 XPS step model (every step cold u8) | 0.164 | 0.766 | 15/15 (not flagged) | 15 escalate | - |

Jump step per entry (largest D; it is the escalating step in every entry):
D 8,109 to 8,208 new tokens (one masked observation) on P 2,159 to 7,356
cached tokens, prompt 10,291 to 15,561. Bracket u8: lower 6.95 to 7.09 s
(cold(D)), upper 10.25 to 20.83 s (cold(P + D)); u4 lower 7.11 to 7.26 s,
upper 10.53 to 20.90 s. The XPS 10 s cold edge is 10,000 to 10,250 tokens
(c2246b1f); there is no sealed warm edge for D > 400 (u8) or D > 183 (u4).
Every entry straddles 10 s: the V5 outcome is not determined by sealed
data. Registered point prediction (upper bound): V5a and V5b escalate in
15/15 entries at the same step as the V4 step model, local share 0.164,
$0.766/task; flagged. No entry flips between the block bounds
(boundary: none at block resolution; all 15 are indeterminate on the
bracket).

Per entry (u8; u4 identical in step and turn): 100 step 1 turn 0; 101 1/0;
110 3/1; 119 1/0; 120 1/0; 125 3/1; 126 3/1; 129 1/0; 132 1/0; 134 1/0; 136
1/0; 137 5/2; 141 5/2; 144 1/0; 146 5/2 (multi_turn_long_context_<id>).

Disclosure: DEMO_PREREG_v2 predicted V4 XPS local share 0.436 with a
per-turn model (step-0 n_ctx of each turn only). The per-step model here
gives 0.164 for the same cell: within-turn steps that receive a large
observation pass the cold edge before the next turn. The measured V4 XPS
(a678926e) decides between the two; it is not used to set any V5 number.

What would make V5 predictable: sealed RESIDENT turn-2 points at D about
8,200 on P 2,000 / 4,000 / 7,500 at u8 and u4 (local-only, $0, one XPS boot).

## 5. Comparison rule (fixed now)

Anchors: V4 T2S f7de976b (complete, 15/15, $0.00, unsealed: seal first) and
V4 XPS a678926e (running at amendment time). Both anchors were observed
before this amendment; V5 predictions use only sealed limit data, not V4
outcomes.
- Recovery R_y = (y_V5 - y_V4xps) / (y_V4t2s - y_V4xps) for y = local share
  and y = -$/task; undefined when the denominator is 0 (then raw deltas).
- Labels: R >= 0.9 matches the upgrade; 0.5-0.9 recovers most; 0.1-0.5
  recovers some; < 0.1 does not.
- Per-entry paired table (V4 XPS, V5a, V5b, V4 T2S): routed or not, step of
  escalation, $, pass.
- Quality: McNemar on passes V5x vs V4 XPS and vs V4 T2S; n = 15 is
  underpowered; a local-share gain with fewer passes is a trade-off, not a
  win.
- One table: each knob's effect next to the upgrade effect (placement V1 vs
  V2; retention V3 vs V4; residency V4 XPS vs V5a; KV precision V5a vs V5b;
  upgrade V4 XPS vs V4 T2S), with sealed run_ids.

## 6. Spend

Prereg v2: caps $145 (V1 5+5, V2 15, V3 45+45, V4 10+20) + reserve $55 =
ceiling $200. Spent so far: V3 T2S b323f0e5 $0 (aborted, 401), V4 T2S
f7de976b $0.00, V4 XPS a678926e running (spend not yet known here). Remaining
planned caps: V2 XPS $15, V3 XPS a/b $24/$21, V3 T2S rerun $45. V5 caps by
the prereg v2 method (hard per-config cap at or above the cell's most-cloud
cost): the all-cloud reference for DEMO1_PAIRED_15 K=1024 is $0.811-0.878 per
task, $13.17 for 15 at the high end, so $15 per cell: V5a $15, V5b $15,
$30 total. The $145 caps are fully allocated to V1-V4, so V5's $30 comes from
the $55 reserve: needs Zach's written OK.

## 7. Scope and limits

One model family (Qwen3-4B int4); one benchmark (BFCL multi-turn base and
long_context); one SLO (10 s; SLO-SENS edges at 5 and 20 s); two Intel
platforms; n = 30 / 15 entries; residency changes some outputs (DET-PROBE:
outputs change under residency, 4/20; RES-QUALITY); the V5 outcome rests on
an unmeasured warm delta prefill of about 8,200 tokens (section 4).

## 8. Launch (XPS, each after a cold reboot; needs Zach's OK and the reserve OK)

V5a (RESIDENT u8):
  ssh <user>@<host> 'cd C:\Users\<user>\Projects\gnn-hls-accel; git pull --ff-only; .\tools\launch_h1.ps1 -Policy slo_escalate -Residency RESIDENT -Kv u8 -ObsMaskK 1024 -Entries derived\h1_hybrid\demo1\DEMO1_PAIRED_15.json -MaxUsd 15'
V5b (RESIDENT u4):
  ssh <user>@<host> 'cd C:\Users\<user>\Projects\gnn-hls-accel; git pull --ff-only; .\tools\launch_h1.ps1 -Policy slo_escalate -Residency RESIDENT -Kv u4 -ObsMaskK 1024 -Entries derived\h1_hybrid\demo1\DEMO1_PAIRED_15.json -MaxUsd 15'
Dry runs: the same with -DryRun appended (one cloud ping, under $0.001,
unless -NoCloudPing).
