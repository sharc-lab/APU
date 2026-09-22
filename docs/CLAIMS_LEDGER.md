# Claims Ledger — Paper 1

Every claim Paper 1 intends to make, with supporting evidence, hardware origin, and
current status. Status definitions:

- **SUPPORTED-ON-TARGET** — supported by data from Strix Halo EVO-X2 (AMD Ryzen AI Max+ 395,
  unified LPDDR5X, Ubuntu). None yet; EVO-X2 not yet provisioned.
- **OFF-TARGET-ONLY** — supported by committed data, but only from Razer Blade 14
  (RTX 4070 discrete, Windows) or evo-t2s (Intel Arrow Lake, Vulkan, unified LPDDR5X).
  Neither is the BOM target. Claim stands as a directional finding; target replication required
  before submission citation.
- **UNSUPPORTED** — no committed data supports this claim. Either the experiment was not run
  or the result file contains no valid measurements for this claim.
- **PENDING** — data collection is in progress now; status will be updated when committed.

No status field is valid without a cited file path.

---

## Section 4 — Axis A: Memory vs. Correctness

### Claim A-01 (Fig 4.1, SUPPORTING — negative result)
**Quality is flat across depth (0–32k tokens) at fixed memory budget** when filler is
semantically inert; depth alone is not the productive axis for quality degradation.

- Files: `results/run_20260813T021516Z.jsonl` (1100 rows, blade_rtx4070, qwen3:4b-instruct)
- Hardware: blade_rtx4070 (discrete, OFF-TARGET)
- **Status: OFF-TARGET-ONLY**

---

### Claim A-02 (Fig 4.3, CORE)
**Artifact truncation causes a sharp cliff in task score** — score drops from ~1.0 to ~0.0
as the artifact survival fraction falls below a per-probe extinction threshold.

- Files: `results/art_truncation.json`, `results/art_truncation_analysis.json`
- Hardware: blade_rtx4070 (discrete, OFF-TARGET — inferred from commit date)
- **Status: OFF-TARGET-ONLY**

---

### Claim A-03 (Fig 4.4, APPENDIX)
**The extinction threshold is probe-specific and occurs in the range 0.444–0.761 artifact
fraction** across a fine-grained 1.00 → 0.85 ratio sweep.

- Files: `results/partial_truncation.json`
- Hardware: blade_rtx4070 (inferred)
- **Status: OFF-TARGET-ONLY**

---

### Claim A-04 (Fig 4.6, SUPPORTING)
**Mean required artifact fraction for correct retrieval is 0.211** (range 0.064–0.433 across
10 probes); 78.9% of artifact tokens are evictable without loss.

- Files: `results/span_ablation.json`, `results/span_ablation.jsonl`
- Hardware: blade_rtx4070 (inferred)
- **Status: OFF-TARGET-ONLY**

---

### Claim A-05 (Fig 4.6 / Table A, SUPPORTING)
**Positional waste gap is 0.255–0.551 artifact-fraction** for 3 probes: the prefix-truncation
survival threshold exceeds the targeted-span retention requirement by this margin.

- Files: `results/span_ablation.json`, `results/artifact_ratio_sweep.json`
- Hardware: blade_rtx4070
- **Status: OFF-TARGET-ONLY**

---

### Claim A-06 (Fig 4.6, SUPPORTING)
**For 9 of 10 probes, the answer span alone (without header or surrounding context) is
sufficient for correct retrieval** (answer_no_header condition).

- Files: `results/span_ablation.jsonl`
- Hardware: blade_rtx4070
- **Status: OFF-TARGET-ONLY**

---

### Claim A-07 (Fig 4.6, SUPPORTING)
**art_04 requires an adjacent log-entry as a format exemplar** in addition to the answer
span; span-only retrieval succeeds semantically but fails on format (DELETE vs. deleted).

- Files: `results/span_ablation.jsonl`
- Hardware: blade_rtx4070
- **Status: OFF-TARGET-ONLY**

---

### Claim A-08 (Fig 4.14, APPENDIX)
**100% fabrication rate and 0% abstention in the 36 classifiable extinct-context rows**
(budget_ratio ≤ 0.85, artifact absent, unclassifiable rows removed from denominator).

- Files: `results/stage_a_scale.json`
- Hardware: blade_rtx4070 (inferred); cloud model gpt-oss:120b
- **Status: OFF-TARGET-ONLY**

---

### Claim A-09 (Fig 4.7 / 4.9, APPENDIX)
**An abstention instruction inverts its intent**: arm2 (abstention_instruction) drives
fabrication to 100% and eliminates the 20% abstention rate present in the baseline arm.

- Files: `results/selfreport_arms.json`
- Hardware: blade_rtx4070 (explicit field); model qwen3:4b-instruct
- **Status: OFF-TARGET-ONLY**

---

### Claim A-10 (Fig 4.9, APPENDIX)
**Schema collision null result**: zero filler lifts when the artifact is present, across
F-NUM, F-TYPED, and F-SCHEMA filler with 3 models (qwen3:4b-instruct, llama3.1:8b,
gpt-oss:120b).

- Files: `results/schema_collision.json`
- Hardware: blade_rtx4070 (inferred)
- **Status: OFF-TARGET-ONLY**

---

### Claim A-11 (Fig 4.8 / Fig 4.14, APPENDIX)
**Type-matched filler interference (art_02/F-TYPED/120B) is a position/recency effect**:
at r=1.20, artifact present, the 120B model retrieves a filler value rather than the artifact
value in 5 of 6 runs.

- Files: `results/stage_a_scale.json`, `results/interference_r120.json`
- Hardware: blade_rtx4070 (inferred); cloud model gpt-oss:120b
- **Status: OFF-TARGET-ONLY**

---

### Claim A-12 (Fig 4.11, SUPPORTING)
**Measured KV cache memory reduction: f16 → q8_0 = 1.77×, f16 → q4_0 = 3.24×** —
shallower than architectural predictions (2.00× and 4.00×) due to per-block quantization
metadata overhead and SWA default on Qwen3.

- Files: `results/llamaserver_feasibility.json` (reference); `results/gate1_kv_precision.json`
  (documents Ollama API limitation; contains no valid cross-precision measurements)
- Hardware: blade14_rtx4070 (explicit field); llama-server b1-f8def7fe1, CUDA
- **Status: OFF-TARGET-ONLY** — must re-run via llama-server on Strix Halo with
  `--cache-type-k` flags verified effective; unified memory subtraction method differs.

---

### Claim A-13 (Fig 4.12, SUPPORTING)
**Budget_ratio × artifact position (EARLY vs. LATE) is the productive axis**: under
harness-side left-char truncation, the LATE arm retains its artifact because filler precedes
it; the EARLY arm loses its artifact first.

- Files: `results/stage_c_20260818T040408Z.jsonl` (396 rows, blade_rtx4070,
  qwen3:4b-instruct), `results/position_pressure_analysis.json`;
  `results/fig61_stagec_full_20260922T203557Z.jsonl` (396 rows, evo-t2s,
  Intel Arrow Lake, Vulkan b10970, qwen3-4b-instruct-85e4a5b7.gguf, matched checkpoint);
  `results/fig61_stagec_full_20260922T230133Z.jsonl` (396 rows, blade_rtx4070,
  CUDA b10970, same checkpoint, like-for-like replication)
- Hardware: blade_rtx4070 (discrete, OFF-TARGET) + evo-t2s (unified LPDDR5X, OFF-TARGET)
- **Three-architecture replication note:** All three runs agree on the position-pressure
  effect (LATE > EARLY at r<1 for probes where artifact is intact). One cell disagrees
  between CUDA and Vulkan backends: sea_01 LATE at r=1.0 and r=1.2 — CUDA outputs "C8"
  (wrong), Vulkan outputs "A9" (correct). Both Blade runs (Ollama and CUDA llama-server)
  agree on "C8". This is backend numerical sensitivity on one borderline probe; 130/132
  cells match. The disagreement does not affect the position-pressure claim.
- **Status: OFF-TARGET-ONLY** — three-architecture replication complete (blade Ollama,
  blade CUDA, evo-t2s Vulkan); Strix Halo EVO-X2 run required for submission.

---

### Claim A-14 (Fig 6.1, CORE — primary contribution)
**The joint feasibility envelope**: a point in (budget_ratio, quality_score, latency) space
is feasible iff score ≥ quality floor AND latency ≤ budget, with EARLY/LATE arm determining
which envelope face the workload lies on.

- Files: `results/fig61_stagec_full_20260922T203557Z.jsonl` (396 rows, evo-t2s,
  Intel Arrow Lake, Vulkan b10970, qwen3-4b-instruct-85e4a5b7.gguf, matched checkpoint,
  streaming with TTFT, cache_prompt=false, max_tokens=128, temp=0, filler=4000 tok seed=42).
  Diagnostic-only predecessor 060942Z superseded (model confound); 191031Z superseded (timing invalid).
- Target hardware: Strix Halo EVO-X2 — NOT YET PROVISIONED (arrives ~Dec 3, 2026).
- **Status: OFF-TARGET-ONLY** — evo-t2s replication complete; Strix Halo run required for submission.

---

### Claim A-15 (Fig 4.13, SUPPORTING)
**Quality-depth curves are model-dependent**: llama3.1:8b shows different per-probe failure
modes from qwen3:4b-instruct, establishing the curves are not harness artifacts.

- Files: `results/model2_truncation.json`, `results/crossmodel_baseline.json`,
  `results/run_20260818T000746Z.jsonl`
- Hardware: blade_rtx4070 (inferred)
- **Status: OFF-TARGET-ONLY**

---

### Claim A-16 (Fig 4.2, SUPPORTING — mechanism note)
**cha_04 failure mechanism is uncharacterized**: the config-parameter substitution hypothesis
is disconfirmed by ablation; the probe returns 600 (wrong) at depth even without the retries
field present.

- Files: `results/ablation_cha04_20260817.jsonl`, `results/run_20260813T021516Z.jsonl`
- Hardware: blade_rtx4070
- Note: the score drop (1.0 → 0.0 at d=2000) is confirmed; the mechanism is not.
- **Status: OFF-TARGET-ONLY** — the mechanism claim must NOT be asserted.

---

### Claim A-17 (Fig 4.2, SUPPORTING — mechanism note)
**lon_02 format failure couples with correct computation at d=16000**: reps that compute
correctly at depth output "130/5=26" (format failure); reps that output a bare digit at
d=32000 often output the wrong digit.

- Files: `results/run_20260813T021516Z.jsonl`
- Hardware: blade_rtx4070
- **Status: OFF-TARGET-ONLY**

---

### Claim A-18 (Table 4.1 / APPENDIX)
**Parametric default failure class**: models emit canonical field-type sentinels (8080, 0,
2147483647) when the answer span is absent — drawn from parametric knowledge, not context.

- Files: `results/span_ablation.jsonl`
- Hardware: blade_rtx4070
- **Status: OFF-TARGET-ONLY**

---

### Claim A-19 (Fig 4.15, SUPPORTING — blocked)
**Multi-turn recall degrades as a function of artifact distance, intervening schema, and
artifact size** under partial KV eviction.

- Files: NONE — harness not yet written; data not yet collected.
- **Status: UNSUPPORTED**

---

## Section 5 — Axis B: Orchestration / Throughput vs. Latency

### Claim B-01 (Table 5.1, SUPPORTING)
**SDK and LangGraph orchestration decompose into measurable span categories**
(ORCH_SETUP, HTTP_CLIENT, TOOL_COMPUTE, FRAMEWORK, RESIDUAL) across 14 task types.

- Files: `results/claude_code_characterization.json` — **gitignored, local only**
- Hardware: UNKNOWN (gitignored; not committed)
- **Status: UNSUPPORTED** — file not in repo; cannot be independently reproduced.
  Must commit or re-run before submission.

---

### Claim B-02 (Fig 5.1, SUPPORTING)
**Tail-latency (p50/p99) distributions across 14 task types under 3 concurrency conditions**
identify outlier tasks that constrain system design.

- Files: `results/tail_latency_results.json` — **gitignored, local only**
- Hardware: UNKNOWN (gitignored)
- **Status: UNSUPPORTED** — same issue as B-01.

---

### Claim B-03 (Fig 5.2, SUPPORTING)
**Independent replication (Zachary Johnson) cross-validates span attribution methodology**
for the remote-search LangGraph task.

- Files: `results/zachary/replication_remote_search_v3.json` — **IN REPO**
- Hardware: UNKNOWN (external replication; hardware not documented)
- **Status: OFF-TARGET-ONLY** — hardware not documented; label as external replication.

---

### Claim B-04 (Fig 6.1, CORE — per-call join)
**TTFT and http_client_ns are measured on the same call as quality score**, enabling
per-call (not mean-of-means) joint envelope points.

- Files: `results/fig61_stagec_full_20260922T203557Z.jsonl` (evo-t2s, streaming, TTFT valid,
  cache_prompt=false verified, n_prompt_tokens_actual from /tokenize, tokens_in_api 396/396).
  Previous run 191031Z scores-valid but timing-invalid (superseded; see RESULT_PROVENANCE.md).
- **What the data shows:** TTFT varies with prompt length (budget_ratio), not with arm.
  LATE and EARLY are within 3% of each other at every ratio (e.g., r=1.20: LATE 7305ms vs
  EARLY 7434ms). TTFT falls from ~7.4s at r=1.20 to ~2.0s at r=0.40, consistent with
  prefill scaling linearly with token count. B-04 supports the claim that TTFT is validly
  measured per call in the same row as quality score. It does not support a claim that
  artifact position affects latency.
- **Status: OFF-TARGET-ONLY** — per-call TTFT measured on evo-t2s; Strix Halo required for submission envelope.

---

## Section 6 — Joint Envelope (Primary Contribution)

### Claim J-01 (PRIMARY CLAIM)
**f(workload type, quality floor) → minimum provisioned GB**: the minimum context memory
needed to meet a quality floor is a function of workload category (RAG vs. search) and
artifact position, measurable empirically from budget_ratio × position sweeps.

- Files: `results/stage_c_20260818T040408Z.jsonl` (Blade, off-target, non-streaming);
  `results/fig61_stagec_full_20260922T203557Z.jsonl` (evo-t2s, off-target, streaming,
  matched checkpoint, TTFT valid, cache_prompt=false). Strix Halo EVO-X2 run required.
- **Status: OFF-TARGET-ONLY** — two-architecture replication complete; no target-class data yet.

---

## Claims requiring Strix Halo data before submission

The following claims are in CORE or SUPPORTING figures and must be reproduced on
AMD Ryzen AI Max+ 395 (EVO-X2, unified LPDDR5X) before submission. All are currently
OFF-TARGET-ONLY or PENDING.

| Priority | Claim | Figure | Blocking issue |
|---|---|---|---|
| 1 | J-01 (joint envelope) | Fig 6.1 (CORE) | EVO-X2 not provisioned; evo-t2s replication complete (off-target) |
| 2 | A-02 (truncation cliff) | Fig 4.3 (CORE) | Discrete RTX 4070; bandwidth difference may shift cliff |
| 3 | A-01 (quality flat at depth) | Fig 4.1 (CORE) | Same as A-02 |
| 4 | A-12 (KV quantization ratios) | Fig 4.11 (SUPPORTING) | gate1 Ollama path invalid; llama-server rerun on Strix Halo AMD unified memory path unvalidated |
| 5 | A-04 / A-05 (span ablation) | Fig 4.6 (SUPPORTING) | Unified memory bandwidth may change required fraction |
| 6 | A-13 (position pressure) | Fig 4.12 (SUPPORTING) | Two-architecture replication complete (Blade + evo-t2s); Strix Halo required for submission |
| 7 | A-15 (multi-model comparison) | Fig 4.13 (SUPPORTING) | Blade only |
| 8 | A-19 (multi-turn recall) | Fig 4.15 (SUPPORTING) | Harness not written; EVO-X2 required |
| 9 | B-01 / B-02 (Axis B span data) | Table 5.1, Fig 5.1 | Files gitignored; must commit or re-run |

**APPENDIX figures** (A-03, A-06, A-07, A-09, A-10, A-11, A-14, A-16, A-17, A-18, B-03)
are acceptable labeled as "Blade 14 / RTX 4070" with a note that Strix Halo re-runs are
planned; they do not block submission if CORE figures 6.1, 4.3, and 4.1 are reproduced.
