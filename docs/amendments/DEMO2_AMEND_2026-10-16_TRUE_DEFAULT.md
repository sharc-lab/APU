# DEMO-2 amendment 2026-10-16: the true runtime default as the control (pre-run redesign)

Status: REGISTERED 2026-10-16 (Task AS), before any DEMO-2 run. Nothing has
run: the only DEMO-2 launch, rehearsal be91e69d, was refused at the
entry-pin check. This file supersedes the control profile, the cells and the
predictions of derived/h1_hybrid/demo2/DEMO2_PREREG_v1.json (sha256
05330d6ba823acc721a57c15b41075ae383a116b6fa9df1c165eafee7587ded2) and of
docs/amendments/DEMO2_DESIGN_2026-10-14.md (sha256
abd4d687778b83638c8c26d380af9c98e4b129ec83e225051bf98f9a4cebbeca). It edits
neither. Runners do not read this file. Its sha256 is in the DEMO-2
registry row.

Predictions file (registered with this amendment):
derived/h1_hybrid/demo2/DEMO2_AMEND_TRUE_DEFAULT_PREDICTIONS.json, sha256
9e7834e89b57e28f11724d6dc9adfd6a5939d7065373b154ba6dec8b793c9825, generated
by tools/demo2_true_default_predict.py from sealed and committed data only.

## 0. Why

Source read at the openvino / openvino.genai 2026.2.1 tags
(docs/literature/2026-10-16/DEMO2_CONTROL/openvino_defaults.md):

- `LLMPipeline(model, "GPU")` with nothing passed runs PagedAttention with
  prefix caching on and unlimited batch tokens, a dynamic KV cache and no
  context cap. It falls back to SDPA only if PA setup throws (one WARN line).
- With no KV property a PA model compiles i8 KV. A u8 request compiles the
  same i8. Automatic u4 arrives only in 2026.3.x.
- The registered control (NON_RESIDENT) passes a bare SchedulerConfig():
  prefix caching off and 256-token chunks. That is not the runtime default.

Against the true default, three registered tuned knobs are no longer
differences: RESIDENT (prefix caching is already on), the 40,960 window (no
cap, trained max 40,960) and the T2S u8 KV (= i8). What still differs is
OBS_MASK retention on both hosts and u4 KV. The registered ~8x prediction
rested on 48 control escalations from a 4096 window that the true default
does not have.

## 1. Common to every arm

- Policy hybrid_full exactly as registered (mechanical sticky: ctx_overflow
  if a window is set, step-0 TTFT > 10 s or decode < 6 tok/s, local
  generation error; quality one-turn bounce: no parseable call, tool
  execution error, n_steps >= 5).
- Weights Qwen3-4B INT4 IR (configs/models/Qwen3-4B-int4-ov.yaml), device
  GPU, openvino and openvino_genai 2026.2.1 pinned. plan.json records both
  versions; a different version invalidates the cell.
- Greedy local decoding (do_sample false), max_new_tokens 512.
- `CACHE_DIR` set in every arm (`-CacheDir`; affects load time only).
- Compiled-property readback on in every cell (`-CompiledReadback`): the
  backend (PA or SDPA) and the compiled KV type of every pipeline load are
  written to plan.json and checked again at each reload.
- Entries DEMO2_HELDOUT_60 (sha256 cc74a37e...). DEMO2_HELDOUT_60_B is
  reserved for the SEARCH confirmation only
  (docs/amendments/DEMO2_SEARCH_PHASE_2026-10-16.md).

## 2. Arms

| arm | name | settings | hosts |
|---|---|---|---|
| A | true default (primary control) | `LLMPipeline(model, "GPU")`: no SchedulerConfig, no KV property, no window; FULL retention | XPS, T2S |
| B | device-only tuned | u4 KV; explicit `SchedulerConfig(enable_prefix_caching=True, max_num_batched_tokens=<P3>, cache_size=<sized>)`; prompt lookup per P4; window min(40,960, P3 resident limit - 10%) on the XPS, 40,960 on the T2S; FULL retention, **no OBS_MASK** | XPS, T2S |
| C | full tuned | B plus OBS_MASK at the K chosen by P5 (default 1024) | XPS, T2S |
| D | ecosystem default | A plus a 4096 window, labeled "the configuration typical hybrid studies inherit" | XPS only |

Launch flags (values in <> are filled by the pilot addendum, section 8):

- A: `-Policy hybrid_full -Placement gpu_only -Kv default -Residency RESIDENT -CacheDir <dir> -CompiledReadback -RequireTrueDefault`
  with no `-ContextWindow`, no `-Sched*`, no `-PromptLookup`, no `-ObsMaskK`.
  run_h1_hybrid already builds exactly `LLMPipeline(path, "GPU")` for
  RESIDENT + `-Kv default` (tools/bfcl_feasibility_probe.py
  load_arm_pipeline: no SchedulerConfig, gpu_only has no properties).
- B: `-Kv u4 -Residency RESIDENT -SchedPrefixCaching on -SchedMaxBatchedTokens <P3> -SchedCacheSize <GB> -PromptLookup -NumAssistantTokens <P4> -MaxNgramSize 3 -ContextWindow <window> -CacheDir <dir> -CompiledReadback`
- C: B plus `-ObsMaskK <P5>`.
- D: A plus `-ContextWindow 4096` (D keeps `-RequireTrueDefault`; the window
  is a runner check, not a runtime property).

The explicit SchedulerConfig in B and C restates prefix caching and the
chunk size because it replaces the default; it is not merged with it.

**Arm A validity (stop rule).** If the "Falling back to SDPA backend" warning
appears, or the compiled KV is not i8, or the backend cannot be read, or
the IR rt_info sets KV_CACHE_PRECISION, arm A is invalid. The runner
refuses at load (`-RequireTrueDefault`); the cell stops and is reported.
No arm-A result is used until pilot P0 confirms all of this on that host.

Framing. u4 on 2026.2.1 is "the vendor's 2026.3 default, applied early",
not a discovery. The comparison pins 2026.2.1 and says so.

Arm D. Ollama's context default is tiered by VRAM: < 24 GiB 4k, 24-48 GiB
32k, >= 48 GiB 256k (docs.ollama.com/context-length). ECO-LLM (2507.09003)
and Minions run Ollama locally and state no context setting
(docs/RELATED_SEARCHES.md, 2026-10-16). D is not run on the T2S: Ollama
would give a 64 GB machine 256k, above the model's 40,960, so there is no
4k "ecosystem" configuration to represent there.

Superseded before running and kept registered as DEFERRED: the bare
SchedulerConfig control (cells D-X, D-T of DEMO2_PREREG_v1) and the old
tuned cells T-X, T-T. The control's own question (does 256-token chunking
slow prefill?) is answered by registry row NONRES-CHUNK. New cell ids: A-X,
B-X, C-X, D-X2 (XPS); A-T, B-T, C-T (T2S). Run order per host: A, C, B,
then D-X2 on the XPS (A first, so the validity rule is checked first).

## 3. Predictions (sealed data only; DERIVED unless a run_id is given)

The registered ~8x prediction ($0.046 vs $0.356 per task) is withdrawn as
a prediction against A. It is kept as the prediction against D: D-X2
carries the DEMO2_PREREG_v1 D-X band unchanged.

$ per task on DEMO2_HELDOUT_60 (registered cost model of DEMO2_PREREG_v1
with per-arm triggers; bands lo / point / hi; mechanical escalations at the
point):

| cell | lo | point | hi | mech. esc. (point) |
|---|---|---|---|---|
| A-X | 0.098 | 0.119 | 0.161 | 3.0 |
| B-X | 0.071 | 0.119 | 0.161 | 3.0 |
| C-X | 0.023 | 0.046 | 0.112 | 0.0 |
| D-X2 (of record) | 0.352 | 0.356 | 0.365 | 48.0 |
| A-T | 0.026 | 0.052 | 0.157 | 0.0 |
| B-T | 0.040 | 0.078 | 0.157 | 0.7 |
| C-T | 0.023 | 0.046 | 0.091 | 0.0 |

Triggers used: XPS A: memory limit 20,000 (cf3555d5, i8) and cold edge
10,000 (c2246b1f); T2S A: no memory limit at 64 GB, cold edge 18,500
(d2d5cc4a); XPS B/C window 23,850 (u4 limit 26,500 e4a22dac - 10%) at
point and hi, 36,864 at lo (the DERIVED P3 leap); T2S B/C window 40,960.
c647f0c7's u4 edge 9,750 is HELD and not used.

C/A ratio, registered as a band across two estimates:

| host | model point | sealed proxy FULL -> OBS_MASK | registered C/A band |
|---|---|---|---|
| XPS | 0.384 | 0.597 ($0.308 -> $0.184; 71d33f7a+ab88f47f -> a678926e) | [0.38, 0.60] |
| T2S | 0.886 | 0.135 ($0.096 -> $0.013; a4284ab0 -> 30caa847) | [0.13, 0.89] |

Flag: the proxy cells ran NON_RESIDENT (prefix caching off, 256-token
chunks), u8 (= i8), slo_escalate, no window. Arm A runs prefix caching on,
hybrid_full and no window. The proxy's KV is the same; its scheduler,
policy and window differ.

B vs A ($): XPS within +-10% of A (model 1.00: the cost moves through
escalations and payload, which u4 and the runtime levers barely touch). T2S
B >= A (model 1.52): B's 40,960 window turns entries that A keeps local past
the trained length (silently) into paid escalations.

Local-turn latency (local turns only):
- XPS arm A: TTFT p50 0.30 s, p90 2.52 s; turn generate p50 4.56 s, p90
  9.22 s; n_ctx p50 3,936. Source: d482c621 slo_escalate local turns, the
  closest sealed analogue (GPU u8 = i8, RESIDENT, no SchedulerConfig; W-3
  base entries).
- T2S arm A: turn generate p50 3.47 s (DERIVED: XPS / 1.312, the V1 decode
  ratio b026f3fd vs f7343708). TTFT NOT YET MEASURED.
- B and C vs A: turn generate time 5% to 25% lower (DERIVED: decode share
  0.92 of generate time in d482c621; u4/i8 decode gain 1.058 at n_ctx 3,936
  from the bandwidth model, effective ~70 GB/s DERIVED from d482c621;
  prompt lookup x1.0 to x1.3 on decode).

Memory (DERIVED): KV bytes at n_ctx 3,936: i8 0.29 GB, u4 0.15 GB; at
40,960: i8 3.02 GB, u4 1.51 GB. Peak working set: NOT YET MEASURED
(telemetry added in Task AS part 6).

Context held: XPS A 20,000 (cf3555d5); XPS B/C 26,500 (e4a22dac), up to
40,960 only if the DERIVED P3 leap holds; XPS D 4,096; T2S A not bound by
memory (no window); T2S B/C 40,960.

Passes (sealed rates, Wilson 95%):
- base half, all arms: 0.105 to 0.150 (d482c621 W-3 200, bounceback to slo).
- long half, XPS: A and B 4/15 = 0.27 [0.11, 0.52] (V3); C 6/15 = 0.40
  [0.20, 0.64] (V4 a678926e).
- long half, T2S: A and B 3/15 = 0.20 [0.07, 0.45] (V3 a4284ab0); C 2/15 =
  0.13 [0.04, 0.38] (V4 30caa847).
- Direction C - A: up on the XPS, down on the T2S. Non-inferiority is most
  at risk on the T2S. Capacity paradox: raising local capacity keeps
  entries local that A would escalate, at 0-10% local pass rates.

## 4. Outcomes, all reported

1. $/task with the pre-registered decomposition. Let E = escalated cloud
   requests per task and p = mean cloud $ per request. Then exactly
   `$C - $A = (E_C - E_A) x p_A + E_C x (p_C - p_A)`, i.e.
   delta $/task = delta(escalations) x payload + escalations x
   delta(payload). Reported overall and by trigger_class (mechanical,
   quality), for C vs A and B vs A.
2. Local-turn latency, never pooled with cloud turns: TTFT p50/p90 and turn
   generate time p50/p90.
3. Memory: KV bytes (compiled KV type x n_ctx) and peak working set per turn
   (`-MemTelemetry`).
4. Context held: the largest n_ctx a local generate completed at, per entry
   and per cell.
5. Quality: hybrid passes; paired non-inferiority as registered (Newcombe
   paired score interval, one-sided 95% lower bound > -0.15, worded
   "within 15 points").
6. Escalation reasons by class.

The primary test of DEMO2_PREREG_v1 (one-sided Wilcoxon on paired $/task,
bootstrap CI of the mean difference, seed 20261014) applies to C vs A per
host.

## 5. Replicates

Pilot P7 runs arm A twice on the same entries. If A-vs-A discordance exceeds
about 15 of 60 (25% of entries), plan 2 replicates per arm; otherwise 1.

## 6. Material-win threshold and pivot rule (fixed now)

Noise terms from P7 (per host): d_usd = |$A1 - $A2| / mean($A1, $A2) and
d_lat = |p50_1 - p50_2| / mean(p50_1, p50_2) of local-turn generate time.

C materially beats A on a host if passes are inside the margin (the
non-inferiority test holds) and at least one of:
- $/task lower by >= max(25%, 2 x d_usd);
- p50 local-turn latency lower by >= max(20%, 2 x d_lat);
- context held higher by >= 20%.

Derivation (recorded before any data):
- The 2x noise term makes a win larger than twice the run-to-run change of
  the same configuration (P7), so one replicate's luck cannot produce it.
- 25% ($): about 0.6 of the smallest sealed proxy gain (XPS 40%,
  71d33f7a+ab88f47f -> a678926e), leaving room for the proxy's different
  scheduler and policy.
- 20% (latency): inside the DERIVED 5-25% band of section 3, so only the
  upper part of the predicted gain counts as material.
- 20% (context): below the sealed u4-over-i8 resident gain (+32.5%,
  e4a22dac 26,500 vs cf3555d5 20,000), so the device lever alone can clear
  it; the window alone (B/C at 23,850 vs A at 20,000, +19%) cannot.
- The floors and the 2x factor were set by Zach in the Task AS instruction
  (2026-10-16); the derivations above are recorded with them.

Pivot rule. If no criterion holds on a host, DEMO-2 records "vendor
defaults capture most of the single-device gain; remaining levers:
<measured>" for that host, and the study enters the registered SEARCH
phase (docs/amendments/DEMO2_SEARCH_PHASE_2026-10-16.md). This rule is
decided in advance and is not changed after results are seen.

## 7. Spend

Caps per cell = 1.5 x the cell's hi-band total (D-X2: the registered D-X
hi band): A-X $14.49, B-X $14.49, C-X $10.06, D-X2 $32.80, A-T $14.14,
B-T $14.14, C-T $8.23. Sum $108.35 with one replicate per arm, inside the
$150 ceiling (headroom $41.65). With 2 replicates per arm (if P7 says so)
the sum is $216.70: **shortfall $66.70 against the approved $150. Decision
for Zach; approval is not assumed.** Pilot spend (P5, P7) is separate and
needs its own approval (docs/amendments/DEMO2_PILOTS_2026-10-16.md).

## 8. Sequencing

1. Pilots P0-P9 (DEMO-1 pins only; never a held-out pin).
2. Final arm specification committed as an addendum to this amendment,
   with the values the pilot rules select: u4 kept or not (P8), chunk cap
   and cache_size (P3), prompt lookup settings (P4), OBS_MASK K (P5), XPS
   window (P3), replicates (P7). After the pilots and before any held-out
   run.
3. Rehearsals (2 entries, $1 cap, DEMO-1 entries).
4. Held-out cells in the run order of section 2.

## 9. Unchanged

DEMO2_HELDOUT_60, the hybrid_full policy, the non-inferiority margin and
test, the primary statistical test, the T2S Windows Update arrangement, and
every registered threshold.
