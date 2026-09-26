# Per-Probe Findings

Mechanistic observations from sweep runs that are not captured by scores alone.
These are observations about *how* the model fails, not just *that* it fails.
Each entry should name the failure mode precisely so that a score drop at any
context depth can be attributed to a mechanism rather than reported as generic
"degradation."

---

## KV Cache Quantization — Measured Gains vs Architectural Prediction

**Experiment:** `results/llamaserver_feasibility.json`  
**Date:** 2026-08-26, blade14_rtx4070, qwen3:4b-instruct (Q4_K-Medium), llama-server build b1-f8def7fe1  
**Method:** VRAM delta between ctx=4096 and ctx=32768, n_slots=1, n_gpu_layers=99

### Finding: quantization delivers less memory reduction than the architectural calculation predicts, and the shortfall increases with quantization depth

| Precision | Measured B/tok | Architectural B/tok | Ratio meas/arch | KV reduction vs f16 (meas) | KV reduction vs f16 (arch) |
|-----------|----------------|---------------------|-----------------|---------------------------|---------------------------|
| f16  | 144,530 | 147,456 | 0.980 | 1.00× (baseline) | 1.00× (baseline) |
| q8_0 |  81,490 |  73,728 | 1.105 | 1.77× | 2.00× |
| q4_0 |  44,626 |  36,864 | 1.211 | 3.24× | 4.00× |

The architectural values assume pure precision: f16 = 2 bytes/element, q8_0 = 1 byte/element, q4_0 = 0.5 bytes/element. Measured values differ for two reasons:

**f16 (0.98 ratio):** Qwen3 uses sliding window attention (SWA) by default (`--swa-full` not set). SWA layers maintain a smaller KV window, reducing total KV allocation below the full-context architectural value. This ratio is build-specific — a build with SWA-full or a non-SWA model would give a different number. Record the build identifier (b1-f8def7fe1) alongside any use of this figure.

**q8_0 and q4_0 (>1.0 ratio):** Per-block quantization metadata (scale factors, block headers) is stored at full precision alongside the quantized elements. This overhead is constant per block regardless of element precision; as precision drops and elements per byte increase, the metadata fraction of total KV grows. At q4_0, overhead accounts for an additional ~21% above the element-only prediction.

### Implication for the provisioning table

The published token-precision literature reasons in architectural bytes-per-token (e.g., "q4 KV is 4× smaller than f16"). This study cannot confirm that claim at its stated ratio: measured f16→q4_0 reduction is **3.24×, not 4×**. A provisioning calculation that uses the architectural ratio will overestimate the memory reduction that quantization delivers.

The provisioning table must carry two columns: architectural and measured. The gap between them is the point — it is what a device OEM would need to correct for when sizing memory from published token-precision data.

Quantization still provides meaningful reduction (3.24× for q4_0 vs 1× for f16 measured — 144,530 B/tok, build b1-f8def7fe1, SWA enabled) and the flags take effect. The finding is not that quantization is broken but that the reduction is shallower than often stated, and the shortfall is larger at higher compression.

### Note on the 144,530 B/tok f16 figure

This figure is used in Stage 1.4 reclaimable-MB estimates. It is from a specific build (SWA enabled by default) on a specific host (Blade14/RTX4070). Architectural f16 is 147,456 B/tok. For the provisioning table, both are relevant: architectural gives the hardware-maximum cost, measured gives the observed cost on this specific runtime.

---

## Stage A — gpt-oss:120b-cloud Type-Match Results and Disconfirmed Claim

**Experiment:** `harness/stage_a_scale.py` → `results/stage_a_scale.json`  
**Date:** 2026-08-22, git hash 70d8663 (initial run) + bda6894 (rerun)  
**Machine:** blade14_rtx4070

### Confirmed findings

**Headroom failure / interference — art_02/F-TYPED:** At r=1.20 (artifact fully
present, af=1.00), the 120B model outputs `"0.0147"` — a ppb value from
type-matched filler — in 5 of 6 runs (the sixth rep also showed this on rerun).
The model retrieves the type-matched filler value in preference to the artifact
value even when the artifact is present and untruncated. This is a
recency/position interference effect, not a truncation effect.
art_02/F-TYPED is the headline cell for the interference experiment.

**F-NUM fabrication:** 100% fabrication rate under dissimilar filler (F-NUM)
across all 4 probes at both extinct ratios. Same behaviour as qwen3:4b. No
abstention. This matches the pattern established at smaller model sizes.

**F-TYPED lifting (art_06):** ~50% lift rate across r=0.85 and r=0.40. Model
retrieves surnames from filler records rather than denying. Same mechanism as
at smaller models.

### Disconfirmed claim: implicit abstention

**Status: DISCONFIRMED.**

The Stage A commit described "novel failure mode: art_06/F-NUM and art_07
(both fillers) produce empty string outputs (implicit abstention) rather than
fabricating." This was wrong. The empty outputs were **thinking-budget
exhaustion**, not deliberate abstention.

**Evidence:** On rerun at MIN_PREDICT=1024 (`harness/stage_a_rerun.py`,
commit bda6894), the 21 originally-empty rows split as:

- `done_reason='length'` (budget exhausted): **12 rows** — art_07 across
  both fillers and both ratios (5/6 reps even at 1024), plus art_06/F-TY
  r=0.85 rep=1. These rows remain without output at 1024.
- Became non-empty: **9 rows**. Of these, 8 are fabrications (surnames,
  wrong versions, lifted ppb values); 1 is a denial ('NOTFOUND').

**art_06/F-NUM r=0.85 specifically:** All three reps that were empty at 512
tokens produced fabricated surnames at 1024 (`'Miller'`, `'Smith'`,
`'Smith'`). The 120B model fabricates under dissimilar filler like qwen3:4b;
it produced no novel failure mode there. The apparent empty-output behaviour
was entirely an artefact of token budget.

**Corrected reading:** Under F-NUM filler, gpt-oss:120b fabricates (like
qwen3:4b), not abstains (unlike llama3.1:8b). The cross-model abstention
contrast (qwen/120b fabricate, llama abstains) holds without qualification
from 120B data.

### art_07 — genuinely long-thinking at scale

art_07 is a version-number probe (ORM release notes, CVE lookup). At
MIN_PREDICT=1024, art_07 rows remain `done_reason='length'` in **5 of 6**
rerun reps across both fillers and both extinct ratios (r=0.85 and r=0.40).
This is not under-budgeting — 1024 tokens is a substantial thinking budget.
The model's search through long version-history filler appears to require more
than 1024 thinking tokens when the artifact is extinct.

**Implication for the Stage A table:** The art_07 extinct rows are largely
missing outcomes. In `results/stage_a_scale.json`, art_07 rows at r=0.85 and
r=0.40 should be treated as `outcome=None` / not classifiable for the
fabrication-rate analysis. They are flagged with `classification_method:
"unavailable"` (pre-fix rows) or `rerun: true` + `done_reason: "length"`
(rerun rows). Any aggregate fabrication rate that includes art_07 extinct rows
is inflated (those rows are classified `outcome="incorrect"` by the scorer
when `output=""`, but the true outcome is unknown).

The art_07 r=1.20 headroom rows (artifact fully present) produced correct
outputs (`'3.11.9'`) in all 6 reps at 512 tokens — so the model can retrieve
the correct value quickly when no search is needed. The long-thinking issue
is specific to the extinct context (full filler scan required).

---

## Fabrication rates, unclassifiable rows excluded

**Source:** `results/stage_a_scale.json` (gpt-oss:120b-cloud, 4 probes × 2 fillers × 3 ratios × 3 reps = 72 rows) and `results/selfreport_arms.json` (qwen3:4b-instruct, self-report arms).
**Script:** `analysis/unclassifiable_rows.py`

### Why a correction is needed

12 rows in `stage_a_scale.json` have `done_reason == 'length'` and `output == ''`:
the model exhausted its thinking budget (MIN_PREDICT=1024) before producing any
output token.  The scorer records `outcome = "incorrect"` for an empty string,
but the true outcome is unknown — the model may have been mid-computation.
Including these rows in the fabrication-rate denominator inflates the count.
They are distinct from the 51 rows with `classification_method == 'unavailable'`,
which are pre-rerun rows that do have non-empty outputs and valid recorded outcomes.

The per-probe rates stated in the confirmed findings above (e.g., "100% fabrication
under F-NUM") are unaffected for probes other than art_07 because no other probe
has budget-exhausted rows.  The correction matters for any stated aggregate rate
over all probes, and for the honest denominator claim at extinct conditions.

### Table 1 — Unclassifiable row inventory (stage_a_scale.json)

| Criterion | Count | Notes |
|-----------|------:|-------|
| `done_reason == 'length'` | 12 | art_07 (11 rows) and art_06/F-TYPED/r=0.85 (1 row); output empty; outcome unknown |
| `classification_method == 'unavailable'` | 51 | Pre-rerun rows; all 51 have non-empty outputs and valid recorded outcomes — **not** unclassifiable |
| `output == ''` | 12 | Identical to `done_reason == 'length'` set |
| **Truly unclassifiable (used for correction)** | **12** | `done_reason == 'length'` only |

### Table 2 — Budget-exhausted rows by probe / filler / ratio

| probe | filler | ratio | budget-exhausted rows | cell total |
|-------|--------|------:|----------------------:|-----------:|
| art_06 | F-TYPED | 0.85 | 1 | 3 |
| art_07 | F-NUM | 0.40 | 2 | 3 |
| art_07 | F-NUM | 0.85 | 3 | 3 |
| art_07 | F-TYPED | 0.40 | 3 | 3 |
| art_07 | F-TYPED | 0.85 | 3 | 3 |

All other (probe, filler, ratio) cells have 0 budget-exhausted rows.

### Table 3 — Fabrication / abstention / correct rates: uncorrected vs corrected

**Uncorrected** = budget-exhausted rows counted as incorrect (scorer behaviour, `outcome='incorrect'`).
**Corrected** = 12 budget-exhausted rows removed from denominator entirely.
Fabrication (corrected) = `outcome == 'incorrect'` with non-empty output.
No abstentions were observed in any classifiable row of this file.

| Subset | n | correct | fabrication | abstention | note |
|--------|--:|--------:|------------:|-----------:|------|
| All rows — **uncorrected** | 72 | 29.2% | 70.8% | 0.0% | 12 budget-exhausted counted as incorrect |
| All rows — **corrected** | 60 | 35.0% | 65.0% | 0.0% | 12 budget-exhausted removed |
| Extinct (r≤0.85) — **uncorrected** | 48 | 0.0% | 100.0% | 0.0% | rate unchanged; denominator claim changes |
| Extinct (r≤0.85) — **corrected** | 36 | 0.0% | 100.0% | 0.0% | 12 budget-exhausted removed |
| Extinct + F-NUM — **uncorrected** | 24 | 0.0% | 100.0% | 0.0% | |
| Extinct + F-NUM — **corrected** | 19 | 0.0% | 100.0% | 0.0% | 5 removed |
| Extinct + F-TYPED — **uncorrected** | 24 | 0.0% | 100.0% | 0.0% | |
| Extinct + F-TYPED — **corrected** | 17 | 0.0% | 100.0% | 0.0% | 7 removed |
| Headroom (r=1.20) — uncorr. = corr. | 24 | 87.5% | 12.5% | 0.0% | no budget-exhausted rows |

The correction changes the all-rows fabrication rate from 70.8% to 65.0%.
At extinct conditions the rate is 100% in both cases; what changes is the
denominator: 36 classifiable responses, not 48.

### Table 4 — selfreport_arms.json: aggregate rates by arm (no correction possible)

Individual row data not stored; `done_reason` values unavailable.
Probes: rag_01, rag_02, rag_05, sea_01, sea_04 (qwen3:4b-instruct, all extinct,
ratios 0.85 / 0.70 / 0.55 / 0.40, n=60 per arm).

| arm | n | correct | fabrication | abstention |
|-----|--:|--------:|------------:|-----------:|
| arm1_baseline (no prompt change) | 60 | 0.0% | 80.0% | 20.0% |
| arm2_abstention_instruction | 60 | 0.0% | 100.0% | 0.0% |
| arm3_self_report | 60 | 0.0% | 76.7% | 23.3% |

**Note:** the abstention_instruction arm (arm2) eliminates abstention entirely and
drives fabrication to 100%, the inverse of its intent.  arm3 self-report
reduces fabrication modestly (76.7% vs 80.0% baseline) with similar abstention gain.

### Corrected headline rate

Among the 36 classifiable extinct-context responses in `results/stage_a_scale.json`
(budget_ratio ≤ 0.85; 12 thinking-budget-exhausted art_07 and art_06/F-TYPED rows
excluded from denominator), gpt-oss:120b fabricated or retrieved a filler value in
all 36 cases (100%); abstention was not observed in any classifiable row.

---

## cha_04 — Mechanism Disconfirmed by Ablation (chained_tools, hard)

**Status: DISCONFIRMED.** The config-parameter substitution mechanism proposed
after the main sweep was ruled out by ablation on 2026-08-17. See
`results/ablation_cha04_20260817.jsonl` and `evaluation/probes/ablation.jsonl`.

**Sweep:** run_20260813T021516Z.jsonl, blade14_rtx4070, depths 0–32000

**Score pattern:** 1.0 at d=0 (5/5 reps), 0.0 at d=2000+ (with one stochastic
pass at d=16000 r=2). The sharpest 1→0 transition in the run.

**Prompt (abbreviated):**
```
Tool results, in call order:
get_config() -> {"retries": 3, "backoff_ms": 200, "jitter": false}
get_attempt_log() -> ["fail", "fail", "success"]
With no jitter and constant backoff, how many total milliseconds were
spent waiting in backoff before the successful attempt?
```

Expected answer: 400 (2 failures × 200 ms).

**Observed outputs (main sweep):**
- d=0: "400" — matches expected.
- d=2000–32000: "600" (dominant), "4200" (one outlier rep at d=2000).

**Original hypothesis (substitution):** The model uses `retries=3` from
get_config() instead of counting the 2 failure events in get_attempt_log(),
yielding 3 × 200 = 600. This appeared to explain the score drop at depth.

**Ablation design (2026-08-17, 45 calls, depths 0/8000/32000, 5 reps):**
- `cha_04_ablate`: `retries` field removed entirely from get_config().
  Expected: removing the supposed distractor restores correct counting.
- `cha_04_swap`: `retries` changed from 3 to 7.
  Expected: if substituting the field, model would output 7 × 200 = 1400.

**Ablation results:**

| Probe | Depth | Output | Count |
|---|---|---|---|
| cha_04 (control) | 0 | 400 | 5/5 |
| cha_04 (control) | 8000 | 600 | 5/5 |
| cha_04 (control) | 32000 | 600 | 5/5 |
| cha_04_ablate (no retries) | 0 | 1200 | 5/5 |
| cha_04_ablate (no retries) | 8000 | 600 | 5/5 |
| cha_04_ablate (no retries) | 32000 | 200 (×3), 200200 (×2) | 5/5 |
| cha_04_swap (retries=7) | 0 | 1200 | 5/5 |
| cha_04_swap (retries=7) | 8000 | 200000 (×2), 200 (×2), 2000 (×1) | 5/5 |
| cha_04_swap (retries=7) | 32000 | "200"×46 str (×3), 200000 (×1), 600 (×1) | 5/5 |

**Why substitution is ruled out:**
The decisive result is `cha_04_ablate` at d=8000: it returns 600 even though
the `retries` field is absent. If the model were substituting the labeled field
for a log count, removing the field should change the answer. It does not. The
answer 600 is derivable from the log alone (e.g., 3 total entries × 200 ms),
independent of whether `retries` is present.

`cha_04_swap` does not return 1400 (7 × 200) at any depth. Instead it returns
unstable values (200, 2000, 200000, repeated "200" strings) inconsistent with
any simple field-substitution reading.

**What remains unexplained:**
Both ablation variants fail at d=0 with output "1200" (5/5 reps each). The
original probe at d=0 returns "400" correctly, but the ablation shows this is
not robust: changing or removing `retries` produces a consistently wrong answer
at d=0, suggesting the d=0 correctness for cha_04 is not general log-counting
ability. "1200" is not directly derivable from the stated values under any
obvious formula; no current hypothesis explains it.

The `cha_04_swap` behavior at depth (chaotic, non-reproducible) also has no
clean explanation.

**Current status:** The failure mode at depth is confirmed (score 0.0,
wrong numerical answer), but the mechanism is not characterized. The probe
remains valid for detecting this class of failure; the mechanism claim in the
paper must not be asserted without further evidence.

---

## Stage 1.1 — Schema Collision (135 calls)

**Experiment:** `harness/schema_collision.py` → `results/schema_collision.json`  
**Date:** 2026-08-24, blade14_rtx4070  
**Design:** 3 probes (art_01, art_06, art_07) × {F-NUM, F-TYPED, F-SCHEMA} × 3 models × 5 reps at r=1.20 (artifact fully present, af=1.00)

### Result: disconfirmed — zero genuine filler lifts in 135 rows

With the artifact present at full context (r=1.20), no filler condition — including F-SCHEMA, which replicates the artifact's own record format with different entity identifiers — induces retrieval failure on any probe or model. F-NUM, F-TYPED, and F-SCHEMA all produce mean_score=1.00 across the 27 cells that score clean.

**The one non-trivial failure pattern** is art_06/F-TYPED/qwen3:4b-instruct, 5/5 reps, output='Herrera', score=0.00. Reclassified as **field confusion**, not filler lift: 'Herrera' appears in the artifact itself as the mover (Motion: V. Herrera); the correct answer is the seconder (T. Blum). The model reads the Motion field instead of the Second field. F-TYPED filler also contains Herrera as seconder, making lift classification ambiguous from output string alone — but field confusion is the more parsimonious explanation. The same cell at llama3.1:8b and gpt-oss:120b-cloud scores 1.00.

**Discriminating case:** art_07/F-SCHEMA uses Ferrite ORM in both artifact and filler (different CVEs and version numbers). Score=1.00 across all three models. No version confusion induced by same-schema competing records. This is the case that would have confirmed the hypothesis if any effect existed.

### Mechanism implication

The attentional competition mechanism that drives 75% lifting under type-match (established by filler_composition and type_match experiments) requires the answer span to be **absent**, not merely outnumbered by competing schema-matched context. With the artifact present, models retrieve correctly regardless of filler format. Consequently:

- Stage 1.2 (distractor density) is deprioritised. A density curve has no anchor if schema collision does not operate with the artifact present.
- This result is the control the main interference finding (art_02/F-TYPED/120B) requires. That result holds at r=1.20 and af=1.00 — but interference in that cell is a recency/position effect, not schema collision, and is isolated to one probe/model cell.

### Latency anomaly (analytical non-issue)

art_01/F-SCHEMA/qwen3 reps 0 and 1 show latencies 95.609s and 95.61s (1ms apart). Both are the first successful call in their respective runs during a cold-load event (rep=0 is from run-2 after run-1 had a call error; rep=1 is from run-1). Outputs identical (51847), scores identical (1.0). No analytical impact; documented in schema_collision.json.

---

## Stage 1.4 — Resident Set Measurement (250 calls)

**Experiment:** `harness/span_ablation.py` → `results/span_ablation.json`, `results/span_ablation.jsonl`  
**Date:** 2026-08-24/25, blade14_rtx4070, qwen3:4b-instruct  
**Design:** 10 probes × 5 conditions (baseline, no_answer, answer_plus_header, answer_no_header, answer_plus_adjacent) × 5 reps

### What was measured

The minimum set of artifact spans that must be present in the context for correct retrieval, determined by ablating spans individually rather than truncating from one end. The unit is **artifact_tokens_required / artifact_tokens_total**, with preamble (question + instruction) excluded from both terms.

### Required fraction: headline result

Mean required fraction across 10 probes: **0.211** (range 0.064–0.433).

Excluding art_06 (see probe-design note below): mean 0.227, range 0.118–0.433.

The complement, 1 − required_fraction, is the evictable fraction — the portion of the artifact that contributes no tokens to correct retrieval and can be dropped from KV cache without accuracy loss.

| probe | art_total | art_req | required_fraction | evictable_fraction |
|-------|-----------|---------|-------------------|--------------------|
| art_01 | 95 | 20 | 0.211 | 0.790 |
| art_02 | 101 | 20 | 0.198 | 0.802 |
| art_03 | 153 | 25 | 0.163 | 0.837 |
| art_04 | 127 | 55 | 0.433 | 0.567 |
| art_05 | 170 | 37 | 0.218 | 0.782 |
| art_06 | 234 | 15 | 0.064 | 0.936 |
| art_07 | 196 | 67 | 0.342 | 0.658 |
| art_08 | 206 | 34 | 0.165 | 0.835 |
| art_09 | 144 | 17 | 0.118 | 0.882 |
| art_10 | 158 | 31 | 0.196 | 0.804 |

**art_06 note:** required_fraction=0.064 is a probe-design consequence: the artifact contains four agenda items, one of which is relevant. The high evictable fraction reflects item multiplicity, not a general retrieval property. It should not be averaged as if it were evidence about typical task structure.

**At realistic session sizes** (f16, Qwen3-4B, 147,456 B/token architectural):

- 10k token artifact, mean fraction 0.211 → 2,110 required tokens → **311 MB** KV
- 10k token artifact, art_04-like outlier 0.433 → 4,330 required tokens → **638 MB** KV
- 100k token artifact, mean fraction 0.211 → 21,100 required tokens → **3.11 GB** KV

These figures scale linearly in artifact size and are independent of hardware. The Blade14 measurement at 118,784 B/token is not used; see note in span_ablation.json.

### Table A: Positional waste relative to prefix-truncation (3 probes)

This table measures a different quantity from Table B above: the gap between what prefix-truncation must retain to pass and what targeted span retention requires. It is meaningful only for probes with fine-grained truncation sweep data (artifact_ratio_sweep.json). The two quantities are now in the same unit.

| probe | trunc_threshold | required_fraction | gap (positional waste) | reclaimable tokens | reclaimable MB |
|-------|----------------|-------------------|------------------------|-------------------|----------------|
| art_01 | 0.761 | 0.211 | 0.551 | 52 | 7.71 |
| art_07 | 0.597 | 0.342 | 0.255 | 50 | 7.37 |
| art_08 | 0.444 | 0.165 | 0.279 | 58 | 8.48 |

**Gap interpretation:** prefix-truncation must retain everything up to the answer span's position; targeted retention keeps only the span. The gap (0.255–0.551 artifact-fraction) is the positional waste — tokens the model holds in KV cache because the answer is embedded deep in the artifact, not because those tokens are retrievally necessary. At toy probe scale (95–206 artifact tokens) the reclaimable absolute amount is 7–9 MB; the fraction is the transferable quantity.

Do not average Table A and Table B. They answer different questions. Table A requires sweep data that exists for only 3 probes. Table B covers all 10 probes but has no truncation baseline.

### Minimum sufficient condition per probe

For 9 of 10 probes, `answer_no_header` is sufficient: the answer span alone, without any artifact header or surrounding context, produces correct retrieval. art_04 is the exception.

### Format dependency (art_04)

art_04 requires an adjacent log entry as a format exemplar in addition to the answer span. The answer span alone ('14:02  DELETE  tcosta') yields correct value in natural language ('deleted') but incorrect format in all conditions that lack a neighboring log line. Only `answer_plus_adjacent` — which adds '10:33  QUERY   tcosta' — produces the correct output token 'DELETE'. Marginal cost of the format exemplar: ~12 tokens.

**Three failure modes, not two:**

- **retrieval_failure:** answer span evicted → model abstains or fabricates ('No action' in no_answer condition)
- **format_failure:** answer span retained, exemplar evicted → correct value, wrong format ('deleted' or 'deleted the system')
- **correct:** both spans retained → 'DELETE'

A KV eviction policy that retains spans by answer-value proximity will produce format_failure silently — the output is semantically right but structurally wrong, and exact-match scoring marks it correct for 'deleted'/'DELETE' equivalence only if the scorer normalizes. In a system expecting a structured log token, format_failure is an undetected error. Scope: format dependency is confirmed for art_04 (access log, structured token output). The other 9 probes pass on the answer span alone, including art_10 which is also a structured-value probe (integer config field).

### Parametric default failure class

Three probes (art_01, art_09, art_10) emit canonical field-type sentinels when the answer span is absent, with zero variance across 5 reps at temperature 0:

- art_01: '8080' (de-facto HTTP alternative port)
- art_09: '0' (canonical null/disabled keepalive sentinel)
- art_10: '2147483647' (INT_MAX = 2^31−1, canonical unlimited sentinel for signed integer config fields)

These values are drawn from model parametric knowledge about the field domain, not from any visible context. This distinguishes parametric_default from free fabrication (art_02: '12400', no canonical basis) and from neighbor-based fabrication (art_07: '3.12.0', next plausible semver; art_08: 'PN-38847', preceding table row). Parametric_default values are deterministic and field-type-specific; they are the model's prior for that field when context is absent.

art_07/'0.0.0' (null semver) also qualifies under art_truncation conditions but appears under full filler replacement rather than the cleaner span-ablation context.

---

## lon_02 — Format Compliance Couples with Correct Computation at d=16000

**Sweep:** run_20260813T021516Z.jsonl, blade14_rtx4070, depths 0–32000

**Score pattern:** 0.0 at d=0 (wrong answer 63), 1.0 at d=2000–8000 (correct
answer 26), 0.0 at d=16000+ (either format failure or wrong answer 126).

**Failure mode taxonomy:**
- d=0: arithmetic error. Model outputs 63 deterministically. No format issue;
  the answer is simply wrong.
- d=2000–8000: correct computation, correct format. Output: "26".
- d=16000 (4/5 reps): correct computation, format failure. Output: "130/5=26".
  The model shows its final division step. The exact-match scorer rejects this.
- d=16000 (1/5 reps) and d=32000 (majority): wrong answer 126. Model computes
  incorrectly; this is not a format issue.

**Observation:** At d=16000, the reps that compute correctly also leak their
working. The reps that fail format at d=16000 are not the same reps that get
wrong answers at d=32000. This suggests that format compliance and arithmetic
correctness are not independent at larger context depths: the model that
attempts a careful step-by-step computation at d=16000 tends to output that
process rather than distilling to a final answer, while the model that produces
a bare answer at d=32000 may be taking a less careful path that sometimes
produces the wrong number. The coupling is a confound: score=0.0 at d=16000
for lon_02 mixes format failures (model is right) and arithmetic errors (model
is wrong) in a way that makes the per-depth mean misleading.

---

## Bandwidth Saturation — evo-t2s Unified LPDDR5X

**Experiment:** `results/bw_saturation_20260923T045844Z.jsonl` (f16, 5 clean ctx points + 1 partial);
`results/bw_saturation_20260923T065604Z.jsonl` (q8_0 and q4_0, complete for ctx 8192–131072; 262144/524288 rows present but failed or killed — see `_note`/`status` per row)  
**Date:** 2026-09-23, evo-t2s (Intel Arrow Lake, Vulkan b10970-bfdc32183, unified LPDDR5X)  
**Method:** 90% fill prompts, sweep ctx=[8192, 16384, 32768, 65536, 131072] at f16, q8_0, q4_0 KV precisions; measure prefill and decode tok/s per config

### Finding: on unified LPDDR5X, decode throughput scales as 1/N with context length; no knee anywhere across a 16× range and three precisions; the binding constraint is memory bandwidth, not capacity

**f16 — five context points:**

| ctx | prefill tok/s | decode tok/s | Ratio vs prev doubling (prefill) |
|-----|--------------|-------------|----------------------------------|
| 8,192 | 402.6 | 15.8 | — |
| 16,384 | 207.3 | 10.2 | 1.94× |
| 32,768 | 106.3 | 5.8 | 1.95× |
| 65,536 | 53.7 | 3.2 | 1.98× |
| 131,072 | 27.3 | 1.7 | 1.97× |

Prefill throughput halves per context doubling to within 2–3% across a 16× range.
This is pure memory-bandwidth-limited operation: attention over N tokens requires
reading all previous KV for each new token (O(N²) total reads), so throughput ∝ 1/N
when bandwidth is the bottleneck. No capacity cliff appears at any point. The unified
LPDDR5X pool accepted ctx=262144 (~40 GiB allocated) before inference was interrupted
by SSH disconnection; no allocation failure occurred at any tested context length.

At ctx=131072, decode=1.66 tok/s. This is unusable in interactive settings regardless
of whether the context fits in memory. On unified memory the binding constraint is
bandwidth, not capacity; capacity determines which contexts are reachable, but bandwidth
determines whether those contexts are usable.

### Storage vs. throughput: KV bytes are not the decode bottleneck on this build

**Storage (allocator, from `results/kv_val_vulkan_20260923.json`, three cold-start memory deltas at ctx=32768):** q8_0 is 1.999× smaller than f16, q4_0 is 3.997× smaller than f16 — both at essentially exact architectural ratios (2×, 4×). The precision flags take effect at the allocator.

**Throughput does not scale with that storage ratio.** Cross-precision decode and prefill at matched ctx, from `results/bw_saturation_20260923T065604Z.jsonl`:

| ctx | f16 decode | q8_0 decode | q8_0/f16 | q4_0 decode | q4_0/f16 |
|-----|-----------:|------------:|---------:|------------:|---------:|
| 8,192 | 15.84 | 14.19 | 0.90× | 18.07 | 1.14× |
| 16,384 | 10.21 | (incomplete — see row) | — | 11.92 | 1.17× |
| 32,768 | 5.76 | 5.98 | 1.04× | 7.09 | 1.23× |
| 65,536 | 3.18 | 3.27 | 1.03× | 3.93 | 1.24× |
| 131,072 | 1.66 | 1.72 | 1.04× | 2.02 | 1.22× |

| ctx | f16 prefill | q4_0 prefill | q4_0/f16 |
|-----|-----------:|-------------:|---------:|
| 8,192 | 402.57 | 452.15 | 1.12× |
| 16,384 | 207.29 | 257.26 | 1.24× |
| 32,768 | 106.29 | 104.89 | 0.99× |
| 65,536 | 53.73 | 50.00 | 0.93× |
| 131,072 | 27.29 | 24.85 | **0.91×** |

A 2× (q8_0) and 4× (q4_0) reduction in resident KV bytes produces only a ~3–4% decode speedup at q8_0 and ~14–24% at q4_0 — nowhere near proportional to the storage reduction. q4_0 prefill is *faster* than f16 at small ctx (8,192–16,384) but crosses over to *slower* than f16 from ctx=32,768 upward, reaching 0.91× of f16 at ctx=131,072. Because TTFT is dominated by prefill at these context lengths, q4_0's slower prefill outweighs its faster decode for typical response lengths (e.g. ~128 output tokens) at ctx=131,072: q4_0 end-to-end (TTFT + decode time) is slower than f16 there, despite quantized KV being 4× smaller and decode-tok/s being ~1.22× faster in isolation.

**KV bytes resident in memory are not the decode bottleneck on this build.** Reducing KV size by 2–4× does not proportionally reduce attention cost, and at large ctx it can *increase* prefill cost for q4_0. Something other than raw KV byte count — most plausibly attention/dequantization compute overhead on the Vulkan backend — dominates at these context lengths.

### Mechanism: flash attention is forced on for quantized KV; cited from source, not the runtime log

The runtime log (`kv_val_f16.txt`, `--log-verbosity 3`) contains no flash-attention statement and no KV buffer-size line at any precision — the Vulkan server build only logs `n_ctx_slot`/`kv_unified` at load, nothing about the attention path. That question was resolved from llama.cpp source at the exact commit the binary was built from, not from the log:

- Build `b10970-bfdc32183` corresponds to upstream commit `bfdc32183d57f1e35bacf35c47d6311e2028bbbc` (`gh api repos/ggml-org/llama.cpp/commits/bfdc32183`, dated 2026-09-14).
- `bw_saturation_sweep.py`'s server launch command (`start_server()`) never passes `-fa`/`--flash-attn`. Default is `LLAMA_FLASH_ATTN_TYPE_AUTO` (`common/common.h:499` at that commit).
- `src/llama-context.cpp:3704-3707` at that commit: when the V-cache type is quantized (`ggml_is_quantized(params.type_v)`) and flash-attn is `AUTO`, llama.cpp force-enables it, logging `"enabling flash_attn since it is required for quantized V cache"` — the non-flash attention path cannot consume a quantized V cache at all, so it is not an available fallback.

So the q8_0 and q4_0 runs in this sweep ran with flash attention forced on by this code path (not by an explicit flag). The f16 runs had an unquantized V cache, so this specific force-enable branch never triggered for them; whether AUTO also resolved to flash-attn there was left open in this section originally and has since been resolved from source (both citations below, same commit `bfdc32183`):

- `src/llama-context.cpp:505-545`, function `resolve_fused_ops()`: when `cparams.auto_fa` is true (AUTO, not forced by the quantized-V branch above), it calls `resolve(llm_fused_op_flash_attn_probe, cparams.flash_attn)`. `flash_attn` starts `true` for AUTO (line 229: `cparams.flash_attn = params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED`). `resolve()` builds a graph reservation and checks only whether the fused Flash-Attention op node lands on the same backend device as the model's layers (`device_fused != device_layer`) — it disables FA only on a *device* mismatch (e.g. `--no-kv-offload`-style splits), not on KV precision. A single-GPU `-ngl 99` run has no such mismatch by construction.
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp:19286-19332`, function `ggml_backend_vk_device_supports_op()`, case `GGML_OP_FLASH_ATTN_EXT`: the KV-type allow-list `fa_kv_ok()` (line ~19307) explicitly includes `GGML_TYPE_F16` alongside `Q8_0`/`Q5_1`/`Q5_0`/`Q4_1`/`Q4_0`/`IQ4_NL`/`F32`/`BF16` — f16 is a fully eligible KV type for the Vulkan flash-attention kernel, not just the quantized types. The op additionally requires `device->coopmat2` or `(device->subgroup_shuffle && device->subgroup_vote)` (line ~19328); this is a live GPU/driver capability query, not something source alone fixes. It is empirically known to be satisfied on evo-t2s's GPU/driver for this build, because the q8_0/q4_0 runs in the same sweep required and got flash-attn (a hard requirement per `llama-context.cpp:3709-3710` — the server refuses to start otherwise) and started successfully.

**Conclusion:** source shows f16 is an eligible KV type for the Vulkan FA kernel, and the one runtime-dependent gate (subgroup/coopmat2 capability) is empirically confirmed present on this exact device via the successful quantized runs. AUTO therefore very likely also resolved flash-attn to enabled for the f16 runs in this sweep. This is not a certainty in the way the quantized case is (that one is forced unconditionally by a different code path with no capability dependency) — it depends on `resolve_fused_ops()`'s live graph-based decision, which this build's log never states — but source plus the empirical corroboration converge on "yes." Whether the Vulkan flash-attn kernel dequantizes KV to f16 internally per block versus operating on quantized bytes directly was not determined and would require reading `ggml_vk_flash_attn()`'s shader dispatch, not just the capability-check function inspected here.

**This measures throughput only.** Whether KV quantization costs task quality at
these context lengths — specifically whether retrieving a target span from a 131K-token
context is less accurate with q4_0 KV than with f16 — is the open question this sweep
does not address. The throughput measurement establishes the hardware operating point;
the accuracy measurement requires the probe suite.

---

## KV Precision Quality Sweep — evo-t2s, ctx=8192/32768: no measurable cost, but the test is at ceiling

**Experiment:** `results/kv_quality_20260923T181840Z_full.jsonl` (+ per-ctx files)
**Date:** 2026-09-23/24, evo-t2s, qwen3-4b-instruct Q4_K_M, b10970-bfdc32183
**Design:** 2 ctx (8192, 32768) × 3 KV precisions (f16, q8_0, q4_0) × 2 artifact positions (ADJACENT = artifact immediately before the question; START = artifact at the very beginning, filler after) × 10 single-fact artifact-retrieval probes (art_01–art_10, exact-match scored) = 120 calls.

### Result: no measurable quality cost from q8_0 or q4_0 KV on single-fact artifact retrieval at 8k and 32k, at either position — but f16 scored 40/40, so this test cannot detect degradation even if it existed

f16 scored 1.0 on all 40 of its rows (both ctx, both positions). Because the baseline is already perfect, this run is **at ceiling**: there is no headroom for q8_0 or q4_0 to score worse than f16 in a way this design could register short of an outright wrong answer. **The null result here is absence of evidence, not evidence of absence.** A quality cost that only shows up as reduced *margin* — e.g. correct but less confident, or correct at r=1 reps but occasionally wrong under resampling — is invisible to a single-rep exact-match design where the ceiling is already 1.0.

119 of 120 rows scored 1.0. The sole non-1.0 row: **art_06 / ADJACENT / ctx=8192 / q4_0**, output `"Herrera"` instead of `"Blum"`, score 0.0. Its START counterpart at the identical config scored 1.0, and art_06 scored 1.0 at every other (ctx, precision, position) cell in the grid. This is not treated as a quantization effect: it is very likely a recurrence of the **pre-existing, precision-independent field-confusion failure mode already documented above** (Stage 1.1 — Schema Collision), where the model reads art_06's Motion field ("V. Herrera") instead of the Second field ("T. Blum") — a confusion demonstrated at f16 in that earlier 135-call sweep, unrelated to filler or KV precision. One data point is not enough to rule out a genuine q4_0-specific effect on this probe, but the prior, better-powered evidence for a precision-independent cause makes that the more parsimonious reading.

**What this run does and does not establish:** it rules out a *gross* quantization failure on short single-fact retrieval at these two context lengths. It does not establish that quantization is quality-neutral in general — the design has no sensitivity to partial degradation, and per the caveat already on record in the bandwidth-saturation section above, **this null result must not be extended to ctx=131072 or beyond**: softmax quantization error accumulates with key count, so long context is exactly where an effect would be expected to appear, and this run's longest context (32768) does not test that range. A provisioning comparison (q4_0@131072 vs f16@32768, same ~7.5–8.6 GiB resident budget) is in progress on evo-t2s at the time of writing and is the intended follow-up for that range; results not yet available.

### Precision-check failure: memory-delta rows lack per-row proof for 5 of 6 configs

Per-config `sys_free_before_server_mib`/`sys_free_after_server_mib` deltas were meant to be a KV-precision sanity proxy (see script docstring). Four of the six configs produced a **negative** delta: `ctx=32768/q4_0` = −7523 MiB, `ctx=32768/q8_0` = −7065 MiB, `ctx=8192/f16` = −8205 MiB, `ctx=8192/q4_0` = −8125 MiB; `ctx=8192/q8_0` = −2691 MiB (5 of 6 negative; only the very first config, `ctx=32768/f16`, the first server ever started in the run, has a clean positive delta of +7629 MiB).

**Cause:** `kill_server()` sends `Stop-Process -Force` and sleeps 3 seconds before the next config's "before" memory snapshot. For a ~7.5–10 GiB Vulkan-backed process, 3 seconds is not always enough for the OS/driver to finish reclaiming pages before that snapshot is taken — so "before" is read while the *previous* server's memory is still draining, making it read artificially low. By the time "after" is read (post health-check), the prior memory has now fully released, making "after" look higher than "before." This produces a negative delta and means **these five rows carry no valid per-row memory-based precision confirmation** — the sign of the arithmetic itself proves the measurement window was contaminated by the previous process's teardown, not that KV allocation shrank.

**Backfill attempted, not possible:** the per-config server log files (`kv_qual_*.txt` on evo-t2s) were checked for command-line or precision confirmation during this same investigation; `parse_kv_log()` returned `{}` for every config in the run (confirmed against the captured run log), and this build's log format (established already in this section) does not print the server's invocation arguments at any verbosity — only `n_ctx_slot`/`kv_unified` at load. There is no alternate log-based source to backfill precision proof for these five rows from.

**Fix for all future runs (not yet applied to any completed sweep):** before taking the "before" memory/process snapshot for a new config, wait for the *previous* `llama-server` PID to fully exit (poll `Get-Process` until it returns nothing, not a fixed sleep). Then, instead of system-wide free memory (which conflates model weights, KV, compute buffers, and OS-level reclaim timing across process boundaries), record the **new server's own process command line and process-private memory** directly: `Get-CimInstance Win32_Process -Filter "ProcessId=<pid>"` for `CommandLine` (proves which `-ctk`/`-ctv` flags actually took effect for that process, closing the log gap entirely) plus `Get-Process -Id <pid>` for `WorkingSet64`/`PrivateMemorySize64` (a single-process measurement, immune to the previous process's teardown timing). This is applied in `harness/kv_neardup_sweep.py` (designed, not yet run — see below) and should be backported to any future rerun of the quality or provisioning scripts.

---

## Blade host-interference and VRAM-spill leads: gate decisions (2026-09-25)

**Hardware arm:** Blade 14 only (RTX 4070 Laptop, 8188 MiB discrete VRAM, CUDA b10970, driver 610.88, Windows 11 Home 10.0.26200, AC power, Balanced power plan). Not the BOM target and never pooled with unified-memory data.
**Sources:** `results/blade_m1_vram_spill_20260925T041415Z.jsonl` + `results/m1_telemetry_*.csv`; `results/blade_m2_host_interference_20260925T043921Z.jsonl` + `results/m2_dmon_*.txt` + `results/m2_gpumem_*.csv`. The M2 table is regenerated by `analysis/blade_m2_evidence_table.py`.
**Value of the Windows CUDA Sysmem Fallback Policy during these runs:** UNKNOWN. nvidia-smi does not expose it and it was not read from the driver profile. It is presumed to be the driver default because it was never changed, but that is unverified.

### M2 evidence (ctx 8192, 4B instruct Q4_K_M, f16 KV, n=3 calls per condition, so no confidence interval is reported)

Telemetry medians are taken over the rows of the 1 s `nvidia-smi dmon -s pucvmt` log that fall inside the request windows. The dmon file has no timestamp column, so rows were placed in time by file mtime minus row index, which is accurate to roughly 2 s. Utilization is the dmon `sm` column and SM clock is the dmon `pclk` column.

| condition | TTFT s | decode tok/s | GPU util % | SM clock MHz | power W | PCIe rx MB/s | PCIe tx MB/s | max Shared Usage |
|---|---|---|---|---|---|---|---|---|
| none | 1.77 | 57.8 | 94 | 2565 | 93.0 | 14.5 | 61.0 | 98 MiB (4 samples) |
| memcpy 16, server -t 4 | 8.34 | 10.3 | 27 | 1110 | 15.0 | 3.0 | 2.0 | NOT CAPTURED |
| spin 16, server -t 4 | 1.92 | 32.2 | 56 | 2565 | 65.0 | 7.5 | 37.5 | 98 MiB (3 samples) |
| memcpy 16, server -t 1 | 5.98 | 21.7 | 37 | 2565 | 49.5 | 9.5 | 21.5 | NOT CAPTURED |
| memcpy 16, server -t 8 | 5.43 | 20.5 | 33 | 2550 | 49.0 | 8.0 | 34.0 | NOT CAPTURED |

Slowdown vs the none row: memcpy 16 -t 4 is 4.71x on TTFT and 5.59x on decode; spin 16 is 1.09x TTFT and 1.79x decode; memcpy 16 -t 1 is 3.37x and 2.66x; memcpy 16 -t 8 is 3.07x and 2.82x. The dmon power-violation and thermal-violation columns read 0 in every row that was checked (none, memcpy 16 -t 4, memcpy 16 -t 1, spin 16).

**Shared Usage was not captured under load.** The Windows counter poll ran as a PowerShell subprocess and returned empty strings for every memcpy row (Part A memcpy, Part B, Part C), most likely because it was starved by the same CPU contention it was meant to observe (cause not verified). The only Shared Usage values in M2 are 98 MiB from the no-load and spin conditions. That 98 MiB is nonzero with no load, so the literal condition "Shared Usage = 0" cannot be met on this GPU: the driver holds a pinned host allocation that grows with ctx (98, 106, 114, 122 MiB at ctx 8192, 16384, 24576, 32768 in M1). The meaningful test is "no growth above that baseline", and for the memcpy rows it is untested.

### Verdict per hypothesis (rule: "ruled out" only if the telemetry was recorded and stayed within 5% of baseline)

- **Host on the critical path** (prediction: GPU utilization falls while SM clock stays steady). Utilization fell 57 to 67 points in every memcpy row. SM clock stayed within 1% of baseline in the -t 1, -t 8 and spin rows, but fell to 43% of baseline in the -t 4 memcpy row. Status: **supported in the -t 1, -t 8 and spin rows, contradicted on the clock criterion in the -t 4 memcpy row.** The requested -t 1 vs -t 8 check showed no thread-count dependence (TTFT 5.98 vs 5.43, decode 21.7 vs 20.5, both within about 10%). That neither confirms nor refutes the hypothesis, because 16 hog threads occupy all 16 logical cores under either setting. The mechanism is not established.
- **Shared power budget** (prediction: SM clock and power fall). Power fell to 16% to 53% of baseline in every memcpy row, but power and utilization fell together, so the two cannot be separated. SM clock stayed within 1% in the -t 1 and -t 8 rows, which carry the same memcpy 16 load and a 2.7x to 3.4x slowdown, and the violation columns read 0. Status: **not supported as the cause in the -t 1 and -t 8 rows; NOT ruled out for the -t 4 memcpy row, where the clock moved by 57%.**
- **PCIe bus contention** (prediction: PCIe throughput changes). PCIe rx fell to 21% to 66% and tx to 3% to 61% of baseline, so it moved by more than 5% and is **not ruled out by the rule**. The change is a decrease that tracks the token rate (decode 18% of baseline and rx 21% of baseline in the -t 4 memcpy row), and the absolute values (2 to 61 MB/s) are far from a saturated link. That is the signature of an idle GPU, not of a congested bus. Status: **no evidence for it, not formally ruled out, and no PCIe load was applied to test it.**

### M2 Part C (dose response) is invalid for quantitative use

The llama-server started for the 5 GB/s dose was never terminated. The 10, 15 and 20 GB/s servers logged "listening on 127.0.0.1:8385" but never received requests: their logs are about 1 KB against 15.8 KB for the 5 GB/s server, and PID 111316 (the 5 GB/s server) was found still holding port 8385 and 3753 MiB of VRAM afterwards. Every call for the 10, 15 and 20 GB/s doses therefore went to the stale 5 GB/s server, the per-PID Shared Usage query tracked the wrong process (hence `gpu_mem_initial` empty on all 12 rows), and framebuffer use read 7500 MiB (two resident servers). The likely reason the kill failed is a swallowed taskkill timeout under the co-runner load, but that was not verified. The stale server was ended on 2026-09-25 after its command line (which named the 5 GB/s log file) was checked. The achieved-GB/s column is still a valid positive control for the co-runner; the TTFT and decode columns are real measurements of a server under that co-runner, but the rows must not be used to fit a dose-response curve.

### Lead A gate (memcpy 16 at ctx 8192): NOT MET. Section A is skipped.

| criterion | result | met |
|---|---|---|
| memcpy 16 slowdown >= 2x on TTFT or decode | 4.71x TTFT, 5.59x decode (-t 4) | yes |
| Shared Usage = 0 throughout | baseline is 98 MiB with no load; not captured under any memcpy row | NO (unverifiable) |
| GPU utilization drops >= 20 points | -67 points | yes |
| SM clock within 5% | 43% of baseline in the -t 4 row (within 1% in the -t 1 and -t 8 rows) | NO |

Lead A failed because the Shared Usage criterion could not be verified and the SM-clock criterion failed on the specified -t 4 condition. What would reopen it: one memcpy 16 rerun at -t 4 with a Shared Usage sampler that is not starved by the load, plus the thermal gate and randomized ordering now required.

### M1 evidence (no co-runner, 3 calls per ctx, telemetry medians over active rows where utilization > 0)

| ctx | prompt tokens | max Shared MiB | dedicated MiB | memory.used MiB | SM clock MHz (median, min) | throttle reasons on active rows | TTFT s (median) | decode tok/s | TTFT ratio vs previous ctx | prompt length ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| 8192 | 7369 | 98 | 3753 | 3753 | 2565, 2340 | 0x0, 0x4 | 1.99 | 58.2 | n/a | n/a |
| 16384 | 14742 | 106 | 4913 | 4913 | 2400, 360 | 0x0, 0x1, 0x4 | 5.18 | 46.3 | 2.60 | 2.00 |
| 24576 | 22114 | 114 | 6073 | 6073 | 2565, 2310 | 0x0, 0x4 | 11.17 | 35.7 | 2.16 | 1.50 |
| 32768 | 29482 | 122 | 7233 | 7233 | 2558, 210 | 0x0, 0x1, 0x4 | 19.54 | 28.7 | 1.75 | 1.33 |
| 40960 | 36861 | 598 | 7926 | 7925 | 2565, 2550 | 0x0 | 118.95 | 6.25 | 6.09 | 1.25 |

Throttle bit 0x1 is GPU idle and 0x4 is software power cap; the low minimum clocks at 16384 and 32768 are single ramp rows at call boundaries. At 40960 power falls to a median 47 W (from about 97 W) while utilization stays at 100%, and no throttle bit other than 0x0 is set on any active row.

### Lead C gate (ctx 40960): MET. Section C proceeds.

Shared Usage is 598 MiB against 122 MiB at the previous ctx (an excess of about 470 MiB over the linear baseline growth). TTFT rose 6.09x for a 1.25x longer prompt, so the length-scaled ratio is 4.9x, above the 1.5x threshold (the earlier steps ran 1.3x to 1.4x on the same measure). SM clock median 2565 MHz, min 2550 MHz, within 5%.

Note for C1: the raw fraction Shared / (Dedicated + Shared) includes the roughly 100 MiB pinned baseline that exists with no spill, so C1 reports both the raw fraction and the excess over the fitted baseline.

---

## Phase D: locked-memory sweep on evo-t2s (Intel Arc B390, unified memory, Vulkan): no silent slowdown, one loud crash, and the crash is not monotonic in the memory limit (2026-09-25)

**Hardware arm:** evo-t2s only (Core Ultra X7 358H, unified memory, Vulkan b10970). Not the BOM target, never pooled with the Blade.
**Sources:** `results/ramlock_evo-t2s_20260925T010739Z.jsonl`, its manifest, and `results/ramlock_phaseD_telemetry/` (per-level balloon CSV, GPU sampler CSV, server log, run log).
**Design:** an AWE locked balloon holds physical pages that Windows cannot trim or page, leaving S GB available. Server qwen3-4b-instruct, ctx 32768, f16 KV, 90% fill, one throughput call plus 5 correctness probes per level. D2 was a smoke gate at S=7 that passed its three literal checks (held pages constant, available within S+250 MB, no free or resize logged). Classes: runs_normally (TTFT within 1.25x of D1 and correctness unchanged), pages_and_slows, fails_loudly, fails_silently.

| level | available before server MB | server load s | TTFT s (x D1) | decode tok/s | probes correct | class |
|---|---|---|---|---|---|---|
| D1 baseline, no balloon | n/a | 2.6 | 277.7 (1.00) | 5.85 | 5/5 | runs_normally |
| S=12 | 12228 | 4.6 | 277.4 (1.00) | 5.81 | 5/5 | runs_normally |
| S=10 | 10146 | 2.6 | 278.7 (1.00) | 5.83 | 5/5 | runs_normally |
| S=9 | 9155 | 4.5 | 282.6 (1.02) | 5.74 | 5/5 | runs_normally |
| S=8 | 8082 | 6.6 | 287.9 (1.04) | 5.65 | 5/5 | runs_normally |
| S=7.5 | 7594 | 6.6 | 289.7 (1.04) | 5.59 | 5/5 | runs_normally |
| S=7 (also the D2 smoke, 303.4 s) | 7096 | 10.9 | 300.1 (1.08) | 5.18 | 5/5 | runs_normally |
| S=6 | 5983 | 154.6 | 296.8 (1.07) | 5.40 | 5/5 | runs_normally |
| S=5 | 0 (no server) | 46.8 then crash | none | none | none | **fails_loudly** |
| S=4 | 3944 | 150.9 | 291.1 (1.05) | 5.53 | 5/5 | runs_normally |

### What the data show

1. **There is no gradual-degradation regime in the tested range.** The largest TTFT ratio is 1.08x (S=7) against the 1.25x threshold, and every level whose server started scored 5/5. Decode fell at most 11% (5.85 to 5.18 tok/s at S=7).
2. **The single failure is a Vulkan device-lost crash during model load, not a memory-exhaustion message.** At S=5 the server exited with code 3221226505 (0xC0000409) and its log ends with `ggml_vulkan: device lost on Vulkan0` about 43 s after thread-pool init. Available memory bottomed at 4.9 MB in that level.
3. **The failure is not monotonic.** S=4, with less memory than S=5, started and ran normally. Each level was run once and none was repeated, so no threshold can be stated. Available memory reached about 5 to 15 MB during server load at every level from S=7 down (S=7 about 14 MB, S=6 14.9 MB, S=5 4.9 MB, S=4 15.0 MB), so whether a load survives at that floor may be close to chance. This is a hypothesis, not a result.
4. **Load time is where the pressure shows.** Server load took 2.6 to 6.6 s down to S=8, 10.9 s at S=7, then 154.6 s at S=6 and 150.9 s at S=4 (about 50x the baseline). The likely cause is the weights being evicted and re-read from the SSD, but disk reads were not recorded (see limits), so this is inferred.
5. **Paging evidence exists only for S=7.** During the first 30 s after the server started, hard faults (Pages/sec) reached a peak of 277,470/s, with a median of 74,278/s over the window and about 981,000 pages (roughly 3.7 GB) read in total, while Available fell to 14 to 87 MB and pagefile use rose from 7.9% to 11.8%. At S=12 through S=8 the same window shows medians of 10 to 16 pages/s. For S=6, S=5 and S=4 hard faults after server start are UNKNOWN (see limits).

### Why S=7 ran normally although the server needs about 7.6 GB

GPU shared usage was 7489 MiB at every level (dedicated usage 0, as expected for unified memory), which is the source of the roughly 7.6 GB estimate. That figure treats everything as non-evictable. Splitting it:

- The GGUF is **2382 MiB of file-backed mapped weights**: evictable and re-readable from disk.
- The f16 KV cache at ctx 32768 is 36 layers x 2 x 8 KV heads x 128 x 2 bytes x 32768 tokens = **4608 MiB** (computed from the model architecture; it matches the roughly 145 KiB per token growth seen on the Blade in M1).
- The remainder, 7489 - 2382 - 4608 = **about 499 MiB**, is compute buffers by subtraction (inferred, not measured).

So the non-evictable footprint is about 5.1 GB and the weights are the part Windows can drop. S=7 leaves about 1.9 GB above that, and even S=6 is above it, which fits both runs being normal, the load slowdown, and the S=7 page-in storm. This reading assumes the weights are the mapped pages and the KV and compute buffers are the anonymous ones. It does **not** explain S=4, which is below 5.1 GB and ran normally, and it does not explain the S=5 crash on its own. The server's process **private bytes** (11,837 MB, its commit charge) were identical at all levels and are not informative. Its **working set** swung from a median of 0 MB (max 79 MB) at S=7 to 9,011 MB at S=6 and S=4, because Windows trims it, so working set is not a valid measure of what the server needs.

### Limits of this run

- The balloon CSV covers only about the first 100 s of each level. After the server starts, its sampling interval stretched to about 20 s and `pages_per_sec` and `pagefile_pct_usage` read exactly 0 at S=6, S=5 and S=4, which is implausible and is treated as a counter failure under starvation. Hard faults and pagefile use for those three levels after server start are UNKNOWN.
- The balloon CSV timestamps are local time (UTC-7) labelled with a false Z. The analysis shifted them by 7 h. The GPU sampler CSV timestamps are true UTC.
- The balloon CSV server columns are 0 (the PID is unknown to the balloon). Server private bytes and working set come from the separate GPU sampler CSV.
- `\PhysicalDisk(_Total)\Disk Read Bytes/sec` was not recorded, so weight re-reads from the SSD are not directly visible. `harness/memory_balloon_awe.py` now records it, a counter-read duration column, and true UTC timestamps (commit 2b317ef); no completed run used that version.
- One run per level. The S=5 crash and the S=4 success are single observations.
- Stale-server check: every level whose server started has exactly 7 request markers in its log, and the S=5 log has 0 plus the device-lost line, so no level was answered by another server.

### Bearing on the paper claim (PENDING)

On this unified-memory Windows/Vulkan stack the regime under locked-memory shortage is: no measurable silent degradation down to 4 GB available, a large load-time cost, and an occasional loud crash. On the discrete-GPU Blade the regime under VRAM shortage is a silent 6x TTFT cliff (M1). Both observations are consistent with the failure regime being set by driver and runtime defaults, but Phase D alone does not establish that; the S=5 result needs a repeat with the guard and the disk-read counter before it can support a claim.

---

## M3, evo-t2s: CPU compute slows the iGPU through a shared package power limit, but the P-core prediction was wrong (2026-09-25)

**Hardware arm:** evo-t2s only (Core Ultra X7 358H: 4 P-cores plus 12 E and LP-E cores, Arc B390 iGPU, unified memory, Vulkan b10970). Not the BOM target, never pooled with the Blade.
**Sources:** `results/t2s_m3_power_coupling_20260925T075348Z.jsonl`, its manifest, `_sysman.csv`, `_wincounters.jsonl` and the per-condition hog files. Table from `analysis/t2s_m3_analysis.py`. Scripts committed before the run (commit 144f792) and verified on the machine against their committed git blobs (manifest `script_provenance`, git head 2ce23fd).
**Design:** ctx 8192, qwen3-4b-instruct, f16 KV, 90% fill, one discarded warm-up then 3 measured calls per condition, co-runner started 5 s before and pinned by affinity mask, order of the three spin conditions shuffled with logged seed 20260925 (realised order: none, spinE, spin16, spinP, none_end). Co-runner is a pure integer loop with no memory traffic. Telemetry at 1 s: Level Zero Sysman iGPU frequency, power-limited frequency, throttle reasons and energy counter (`ZES_ENABLE_SYSMAN=1`), plus Windows GPU Engine, RAPL Energy Meter and processor counters.

| condition | cores busy | TTFT s (x none) | decode tok/s (slowdown) | iGPU MHz median (samples below 2500) | iGPU power W median | RAPL package W median (min to max) | throttle bit 2 set on |
|---|---|---|---|---|---|---|---|
| none | 0 | 18.30 (1.00) | 16.04 (1.00) | 2500 (0%) | 18.0 | 23.8 (20.8 to 34.0) | 0% of samples |
| spinE, logical 4 to 15 | 12 | 26.08 (1.42) | 12.98 (1.24) | 1650 (100%) | 8.4 | 44.9 (44.2 to 45.1) | 100% |
| spin16, all cores | 16 | 25.85 (1.41) | 12.89 (1.24) | 1650 (100%) | 8.4 | 44.9 (44.9 to 45.1) | 100% |
| spinP, logical 0 to 3 | 4 | 19.07 (1.04) | 16.00 (1.00) | 2500 (32%) | 17.7 | 44.9 (42.5 to 45.1) | 30% |
| none_end | 0 | 18.31 (1.00) | 16.04 (1.00) | 2500 (0%) | 18.1 | 23.7 (21.1 to 36.6) | 0% |

n = 3 calls per condition, so ranges rather than confidence intervals: TTFT ranges are within 0.7% of the median in every condition, and the two no-co-runner conditions bracket the run without drift (18.30 s and 18.31 s). The baseline is the mean of none and none_end. Positive controls: every spin row has hog iterations per second above 0 and every worker reported exactly the requested affinity mask (65535, 65520, 15); no row was invalid, no request failed, and no process other than ours was above 5% CPU before the start.

### What the data show

1. **The slowdown tracks a package power limit.** In all three spin conditions the RAPL package power sits at about 45 W (median 44.9 W, worst-case range 42.5 to 45.1 W), against 23.8 W with no co-runner. Where the effect appears, the iGPU is held at Sysman's power-limited frequency (1650 MHz against a 2500 MHz maximum) on 100% of samples with throttle-reason bit 2 set, and its own power falls from 18.0 W to 8.4 W. No thermal bit appears in any condition.
2. **The prediction that P-core spin would hurt more is rejected.** It hurt least. Pinning the spin to the 4 P-cores gave TTFT 1.04x and no decode loss, while the 12 E and LP-E cores gave 1.42x and 1.24x, the same as all 16 cores. Under spinP the iGPU dipped below 2500 MHz on 32% of samples but its median stayed at 2500 MHz.
3. **A consistent reading, with an approximation.** The package sits at the same cap in all three spin conditions, so what differs is how the cap is shared. By subtraction (package minus iGPU power, which assumes the RAPL package domain includes the iGPU and that the two counters are comparable) the CPU side drew about 36.5 W under spinE, 27.2 W under spinP and 5.8 W with no co-runner. Twelve E-cores pulled more of the shared budget than four P-cores, leaving the iGPU less. The hog's per-core rate is consistent with a CPU-side limit too: 14.6 million iterations/s per core on P, 10.5 on E, and 7.8 averaged over 16 cores; the 16-core total (124 M/s) is no higher than the 12-core total (126 M/s), so adding the 4 P-cores bought nothing.
4. **Prefill suffers more than decode** (1.42x vs 1.24x), which fits a frequency-limited compute-bound phase against a bandwidth-bound one.

### Limits

- Three calls per condition and one run per condition; the two no-co-runner conditions agree closely, but there is no repeat of the spin conditions.
- The E set is 12 cores because Windows reports the 8 E and 4 LP-E cores in one efficiency class here, so E and LP-E effects cannot be separated.
- The package limit of about 45 W is read off the telemetry. The BIOS PL1 and PL2 settings were not queried.
- Throttle-reason bit meanings and the Sysman struct layouts follow `zes_api.h` as recalled and could not be checked against a header on the machine. The values are physically sensible (iGPU range 100 to 2500 MHz, power limit 25 W, readings up to 28 W), and bit 2 tracks the observed slowdown, but the label "burst power cap" is not independently confirmed.
- The Windows processor-frequency counter read a constant 1600 MHz and is not informative. The RAPL per-core counters summed to 0.
- Other mechanisms are not excluded: shared uncore or ring clocks and contention among the 12 busy cores are not separable from a power limit with this design. The co-runner is a synthetic integer loop, not a realistic workload.
- AC power is assumed for this mini PC, not measured.

### Bearing on the paper claim (PENDING)

This is the unified-memory half of the "failure regime is selected by driver and runtime policy" claim only in a loose sense: on this SoC a CPU-side load degrades iGPU inference through shared power allocation, a mechanism that has no analogue on the discrete Blade. It does not by itself confirm the claim.

---

## C1, Blade: silent VRAM spill, quantified. A sharp onset between ctx 36864 and 38912, then decode slowdown proportional to the spilled fraction (2026-09-25)

**Hardware arm:** Blade 14 only (RTX 4070 Laptop, 8188 MiB discrete VRAM, CUDA b10970, driver 610.88, AC power, Balanced plan). CUDA Sysmem Fallback Policy value UNKNOWN (never changed, not readable). Not the BOM target, never pooled with the evo-t2s data.
**Sources:** `results/blade_c1_spill_sweep_20260925T053651Z.jsonl`, its manifest and telemetry (`_smi.csv`, `_dmon.txt`, `_winctr.csv`), and `results/blade_c1_server_logs/`. Table from `analysis/blade_c1_spill_analysis.py` (commit bf32465). The run used harness commit 1c17f5d.
**Design:** qwen3-4b-instruct, f16 KV, `-fa on -ngl 99 -np 1 -t 4`, no co-runner, prompt filled to 90% of ctx. ctx 32768 (anchor) then 34816 to 47104 in steps of 2048, order of the 7 grid points shuffled with seed 20260925, the anchor re-measured after every 3 grid points. Per ctx: 1 discarded warm-up and 5 measured calls (128 tokens, `ignore_eos`, `cache_prompt` false), thermal gate before every call (up to 5 min to reach within 3 C of idle, waits logged per call). Telemetry every 1 s: nvidia-smi, `dmon` with PCIe, and Windows GPU Process Memory for the server PID.

| ctx | prompt tokens | TTFT s, median [IQR] | slowdown vs no-spill expectation [bootstrap 95% CI] | decode tok/s (slowdown) | max Shared MiB | excess Shared MiB | excess spilled fraction | SM clock MHz | power W |
|---|---|---|---|---|---|---|---|---|---|
| 32768 (4 anchors, n=20) | 29482 | 19.52 [19.48, 19.59] | 1.00 [1.00, 1.01] | 29.0 (0.96) | 122 | 0 | 0 | 2565 | 97.2 |
| 34816 | 31328 | 21.98 [21.95, 21.99] | 1.01 [1.00, 1.01] | 28.0 (1.00) | 124 | 0 | 0 | 2565 | 95.2 |
| 36864 | 33179 | 24.02 [24.01, 24.05] | 0.98 [0.98, 0.99] | 26.9 (1.04) | 126 | 0 | 0 | 2565 | 95.5 |
| **38912** | 35018 | **100.23** [100.22, 100.24] | **3.71** [3.71, 3.71] | 16.4 (1.70) | 306 | 178 | 0.022 | 2550 | 50.9 |
| 40960 | 36861 | 114.81 [113.86, 117.48] | 3.85 [3.79, 3.95] | 6.5 (4.33) | 598 | 468 | 0.055 | 2565 | 46.3 |
| 43008 | 38702 | 167.00 [166.90, 167.03] | 5.10 [5.10, 5.10] | 4.5 (6.24) | 890 | 758 | 0.086 | 2565 | 41.6 |
| 45056 | 40546 | 179.04 [179.02, 179.32] | 5.00 [5.00, 5.02] | 3.2 (8.67) | 1178 | 1044 | 0.115 | 2565 | 39.0 |
| 47104 | 42391 | 245.99 [244.95, 246.68] | 6.30 [6.26, 6.36] | 2.6 (10.78) | 1474 | 1338 | 0.142 | 2565 | 36.2 |

The no-spill expectation is a quadratic fit of TTFT against prompt length on the five clean points (M1 at ctx 8192 to 24576 plus the anchor). The bootstrap intervals resample the 5 measured calls and so cover call-to-call noise only, not fit uncertainty. Excess Shared is Shared Usage minus a baseline fitted on the three clean ctx (90 MiB plus 1 MiB per 1024 ctx), because the driver holds a pinned host allocation of about 100 to 130 MiB with no spill. Excess spilled fraction is that excess divided by (dedicated plus shared).

### What the data show

1. **The onset is a step, not a slope.** Slowdown is 0.98x to 1.01x at ctx 34816 and 36864 (CI includes 1.0) and 3.71x at 38912 (CI excludes 1.0), a 4.2x jump in TTFT (24.0 s to 100.2 s) across a single 2048-token step. Dedicated memory reaches 7925 to 7931 MiB (VRAM full) from 38912 on, and Shared Usage above the baseline appears in the same step. The onset therefore lies between ctx 36864 and 38912; the 2048 step is the resolution.
2. **It is spill, not throttling.** SM clock stays at 2550 to 2565 MHz in every spilled row, no thermal throttle bit is set on any active row, power falls from about 95 W to 36 to 51 W, and the anchors re-measured after every block drift by only +0.5%, +0.3% and +0.2%. Maximum GPU temperature was 68 to 82 C.
3. **Spill grows by about 290 MiB per 2048 ctx** (178, 468, 758, 1044, 1338 MiB excess), which equals the KV growth per step (147,456 B per token x 2048 = 288 MiB): once VRAM is full, all further KV lands in system memory.
4. **Decode slowdown is proportional to the spilled fraction; TTFT is a fixed penalty plus a ramp.** Over the five spilled points, decode slowdown = 0.08 + 74.6 x excess fraction (R2 0.998, n=5); TTFT slowdown = 3.03 + 20.95 x excess fraction (R2 0.888, n=5). Five points is a small fit, but the decode relation is nearly exact.
5. **No failure was reached.** The server started and answered at ctx 47104 with 1338 MiB (14.2%) of the allocation outside VRAM. The 10x stop rule was not triggered (the largest length-scaled ratio was about 8.7x), so the point of failure lies above 47104 and was not found.

### Limits

- One server and 5 measured calls per ctx; the onset resolution is 2048 tokens.
- Shared Usage is a Windows per-process counter, sampled at 1 s and taken as the maximum over each call window.
- The nvidia-smi power reading can glitch (values near 590 W were seen at idle); readings above 200 W were dropped.
- C1 ran before the stale-server guard existed. Its own start-time check passed, each of the 11 servers logged exactly 6 requests (1 warm-up and 5 measured), and each row records its server PID.
- The C1 manifest does not record script SHAs (added afterwards); the script version is commit 1c17f5d.

---

## C2, Blade: correctness under spill. No answer changed (2026-09-25)

**Hardware arm:** Blade 14 only (RTX 4070 Laptop, 8188 MiB VRAM, CUDA b10970). Not the BOM target, never pooled with evo-t2s.
**Sources:** `results/blade_c2_spill_correctness_20260925T110739Z.jsonl`, its manifest, telemetry and `results/blade_c2_server_logs/`. Scripts committed before the run and recorded in the manifest (git head 9bfd881, `harness/blade_spill_sweep.py` last changed in c6ab4a9). Run with the stale-server guard active.
**Design:** the five artifact probes art_01 to art_05, filler first, then the artifact, then the question (artifact adjacent to the question), filler sized to 90% of ctx, `cache_prompt` false, max_tokens 32, scored by `score()` in `evaluation/probes/scorers.py`. Two contexts: ctx 36864, the last clean context from C1 (0.98x), and ctx 47104, the largest spilled context (6.30x, 1338 MiB spilled).

| probe | expected answer | ctx 36864 (33.2k tokens, no spill): output, score | ctx 47104 (42.4k tokens, 14.2% spilled): output, score |
|---|---|---|---|
| art_01 | 51847 | 51847, 1.0 | 51847, 1.0 |
| art_02 | 0.0073 | 0.0073, 1.0 | 0.0073, 1.0 |
| art_03 | 8.9 | 8.9, 1.0 | 8.9, 1.0 |
| art_04 | DELETE | DELETE, 1.0 | DELETE, 1.0 |
| art_05 | (text answer) | scored 1.0 | scored 1.0 (identical output text) |

10 of 10 probes correct, and every output string is identical between the two contexts. TTFT was 24.2 to 25.7 s at ctx 36864 and 240.3 to 243.5 s at ctx 47104, consistent with C1. No row is invalid, the guard records show no problems for either server, and each server log holds exactly 5 request markers.

### Reading and limits

- The silent spill cost time, not correctness, on these probes: the driver moved KV cache into system RAM and the model still returned the same answers. That is the expected outcome, because spill changes where the KV lives, not the arithmetic.
- The probes are single-fact retrieval and were already at ceiling in earlier sweeps, so this cannot detect subtle degradation. It is a null result on gross errors only.
- One pass per ctx (temperature 0, so a repeat would add little), and the two contexts differ in prompt length (33.2k vs 42.4k tokens) because filler is sized to the context. An answer change could not have been attributed to spill alone, but none occurred.

### Phase D correction (2026-09-25, found while preparing the overnight run): the lock was not held after load at S=6 and probably S=4

Re-reading the per-level balloon logs showed three problems with the Phase D constraint.

1. **The balloon released its memory at S=6.** `ramlock_balloon_D3_S6_20260925T010739Z.csv` ends with a `safety_valve_low_available` row at 215.3 s. The balloon's valve releases the lock when Available memory stays under 1024 MB for more than 120 s, and Available was 15 to 700 MB from 88 s onward because the server was loading (load took 154.6 s and started about 65 s into the level). The valve fired at about the moment the server finished loading, so the S=6 throughput and correctness phases ran with no lock.
2. **S=4 shows the same signature without a logged release.** Its balloon log ends at 193 s with no valve row, and its server working set is a constant 9,011 MB for the last two thirds of the level, identical to S=6 and higher than at S=12. At S=12, 10, 9, 8, 7.5 and 7 the working set settles lower at each lower S (8.4, 5.3, 2.2, 0.4, 0.06 and 0.00 GB) and stays there, which is what a held lock looks like. So S=4 also appears unconstrained after load. This is an inference from the working set, not a logged event.
3. **The balloon log stops early in every level** (17 to 25 rows, about 100 s), which fits the balloon's stdout pipe (opened by the orchestrator and never read) filling and blocking its `print`. The balloon would keep holding while blocked, and the working-set traces for S=12 to S=7 show the pressure lasting the whole level, but this was not logged. The D2 gate checked only the first 105 s of a level.

**Effect on the results above.** Valid as constrained measurements: S=12, 10, 9, 8, 7.5 and 7 (TTFT at most 1.08x, 5 of 5 probes correct). **Invalid as post-load measurements: S=6 and S=4** (their "runs_normally" means only that an unconstrained server is normal). The S=5 crash stands as a load-phase event under real pressure (Available 4.9 MB), and S=4 survived its load under real pressure (Available about 15 MB), so a load-phase difference between S=5 and S=4 remains, one run each. The statement that there is no silent slowdown down to 4 GB is withdrawn and replaced by: no silent slowdown down to 7 GB. The load-time rise at S=6 and S=4 (about 150 s) is real load-phase behavior under pressure, before the release.

**Fix for later runs.** The overnight orchestrator sends balloon output to a file, disables the low-available valve for levels where Available is expected to fall toward zero during load, samples Available itself every 5 s for the whole level, and marks a level invalid if the balloon is not alive and holding at the end of the measured phase.

### Phase D second correction (2026-09-26): the weights were not file-backed mapped pages

The Phase D section above explains why S=7 ran normally by splitting the 7489 MiB of GPU shared usage into 2382 MiB of "file-backed mapped weights (evictable, re-readable)" and about 5.1 GB of KV and compute buffers. That split assumed the server mapped the GGUF. It did not. On this Vulkan device the default llama-server load mode (`--load-mode auto`) behaves exactly like `--load-mode none`: in a control on the 8B (`results/t2s_overnight_20260926T011744Z.jsonl`, record `mmap_control`) the default and `none` starts have identical private bytes (6,243 MiB) and working set (6,178 MiB), while an explicit `--load-mode mmap` start differs (5,916 MiB private, 10,297 MiB working set, an increase of 4,120 MiB). Phase D used the default, so its weights were read into ordinary process memory, not mapped from the file.

What still holds: the arithmetic that about 5.1 GB is KV and compute, and that the working set fell with S. What does not: the claim that the weights were evictable and re-read from the file. The 50x load-time rise at S=6 and S=4 and the hard-fault storm at S=7 are still real observations, but their explanation is open (the weights may have been paged out through the pagefile instead). The explicit mmap-versus-none comparison under a held lock is in tonight's Section C. The same default applies to every other llama-server run in this repository on this build and device.
