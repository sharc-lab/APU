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

## M3, evo-t2s: CPU compute slows the iGPU and drops its clock; mechanism open; the P-core prediction was wrong (2026-09-25)

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

This is the unified-memory half of the "failure regime is selected by driver and runtime policy" claim only in a loose sense: on this SoC a CPU-side load degrades iGPU inference and drops its clock, by a mechanism not established here (see the overnight Section B e4/p4 update below, which rules out package power alone), with no analogue on the discrete Blade. It does not by itself confirm the claim.

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


---

## evo-t2s overnight, Section A: crossing the iGPU memory budget. Nothing slowed below the budget, the server did not start above it (2026-09-26)

**Hardware arm:** evo-t2s only (Core Ultra X7 358H, Panther Lake, Arc B390, 63.49 GB unified, Vulkan b10970). Never pooled with the Blade.
**Sources:** `results/t2s_overnight_20260926T011744Z.jsonl` and its server logs. Table from `analysis/t2s_overnight_analysis.py` (commit 5ebdd01, run on the final file). Script versions per row in `docs/RESULT_PROVENANCE.md`. Full report and self-review: `docs/T2S_OVERNIGHT_REPORT.md`.
**Design:** Qwen3-32B Q4_K_M, f16 KV, `-fa on -ngl 99`, YaRN flags (`--rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 32768`) at every point, prompt of 4,527 tokens (60 s call cap), 1 discarded warm-up and 5 measured calls per point, anchor (65,792) re-measured every 3 grid points, order shuffled with seed 20260925. B = 47,865 MiB (llama-server free-memory figure; `vulkaninfo` heap budget 47,866 MiB).

| n_ctx | starts ok | TTFT s median [IQR] | decode tok/s | max Shared MiB |
|---|---|---|---|---|
| 65,792 (4 anchors) | 5/5 | 31.97 [31.75, 32.09] | 4.6 | 35,751 |
| 111,104 | 1/1 | 31.88 [31.87, 31.88] | 4.6 | 47,190 |
| 117,248 to 125,440 (5 points) | 0/5 | none | none | none |

### What the data show

1. **No slowdown below the budget:** 1.00 [0.99, 1.00] at 111,104 against the 65,792 anchors, with 47,190 MiB of Shared Usage.
2. **Above the budget the failure is loud:** five of five grid points from 117,248 to 125,440 exited with code 1 at start (`failed to allocate buffer for kv cache`). No silent slow regime and no wrong answers were produced at this grid (the 256-token boundary is in the A-mech section below).
3. **YaRN confounds the ladder.** The YaRN check on the same 14,747-token prompts: 5 of 5 probes correct without YaRN, 4 of 5 with (art_03: 8.9 versus 33.6). The 4 of 5 at 111,104 is a YaRN effect.

### Limits

- One model; the 30B-A3B, 14B, 8B and 4B ladders were trimmed by the planner, as were 12 of the 32B items.
- Prompts were 4,527 tokens, so the allocated KV was about 4% filled. Full-KV behavior is not measured (see the A-mech filled check).
- One start per failing grid point. Timing rows used a RAPL package-power proxy gate only.

---

## evo-t2s overnight, Section B: the co-runner slowdown replicates on the 4B and the 8B, and package power does not explain it (2026-09-26)

**Hardware arm:** evo-t2s only. **Sources:** `results/t2s_overnight_20260926T011744Z.jsonl`, hog affinity and iteration files, Sysman and Windows counter files. Table from `analysis/t2s_overnight_analysis.py`.
**Design:** context 8,192, 7,368-token prompt, 1 warm-up and 5 measured calls per condition, `none` measured 10 times, co-runner (integer spin loop) pinned by affinity mask 5 s before the first call, order per model shuffled with the run seed. Qwen3-4B-2507 and Qwen3-8B. The PROCTHROTTLEMAX arm was not run (positive control failed in smoke: 44.93 W at cap 50 against 44.94 W at cap 100).

| model | co-runner | TTFT slowdown [boot 95%] | decode slowdown | iGPU MHz | package W |
|---|---|---|---|---|---|
| 4B-2507 | e4 / p4 / nonp12 / all16 | 1.02 / 1.04 / 1.40 / 1.39 | 1.00 / 1.00 / 1.24 / 1.24 | 2500 / 2500 / 1650 / 1650 | 39.1 / 44.9 / 44.9 / 44.9 |
| 8B | e4 / p4 / nonp12 / all16 | 1.05 / 1.06 / 1.42 / 1.43 | 1.00 / 1.01 / 1.20 / 1.20 | 2500 / 2450 / 1600 / 1600 | 41.1 / 44.9 / 44.9 / 44.9 |

`none`: 18.20 s and 22.8 W (4B), 20.16 s and 25.8 W (8B). Every interval excludes 1.0 except the `none` reference.

### What the data show

1. **The slowdown replicates** (1.39x to 1.43x TTFT, 1.20x to 1.24x decode) on two more sizes and matches the earlier M3 figure of 1.42x.
2. **The iGPU frequency falls to 1600 to 1650 MHz exactly when the slowdown appears.**
3. **Package power is not the explanation on its own.** The 4 P-core co-runner sits at 44.9 W like the 12-core one but leaves the iGPU near 2500 MHz and costs only 1.04x to 1.06x. Across the five conditions the correlation of package W with iGPU MHz is -0.53 in each model.
4. **What separates the conditions is which cores are busy:** the 12 E and LP-E cores cost 1.40x to 1.42x, the 4 E-cores 1.02x to 1.05x.

### Limits

- Two sizes, one family (Qwen3). One run per condition.
- The thermal gate is a package-power proxy and it released by timeout in every co-runner condition (48 rows), so each measurement followed at least 120 s of load. No temperature was recorded (Sysman has no sensor), so thermal throttling is not ruled out.
- Causal test not run; the mechanism claim in A-23 is downgraded to UNVERIFIED.

---

## evo-t2s overnight, Section C: zero and negative headroom did not slow TTFT; the one failure was the non-mmap arm at -1 GB (2026-09-26)

**Hardware arm:** evo-t2s only. **Sources:** `results/t2s_overnight_20260926T011744Z.jsonl`, balloon logs and telemetry. Table from `analysis/t2s_overnight_analysis.py`.
**Design:** headroom = free memory after the lock minus (weights + KV + compute), 8B at context 16,384 (12,588-token prompt) and 32B at 16,384 (4,527-token prompt); the balloon held the lock for the whole level (valve disabled, own Available sampler, level marked invalid if the balloon died; 0 of 6 levels invalid); arms `--load-mode mmap` and `--load-mode none` (the default `auto` behaves like `none` here); pairs per headroom in randomised order; anchor at +8 GB.

| model | load mode | headroom GB | load s | TTFT s | decode tok/s | probes | max pages in/s | Available min MB |
|---|---|---|---|---|---|---|---|---|
| 8B | mmap | +8 | 5.6 | 55.7 | 9.33 | | 960 | 3,702 |
| 8B | mmap | 0 | 19.9 | 56.5 | 8.67 | 5/5 | 1,152 | 113 |
| 8B | mmap | -1 | 23.0 | 56.7 | 8.66 | 5/5 | 56,571 | 172 |
| 8B | none | 0 | 7.7 | 56.8 | 8.54 | 5/5 | 25,082 | 50 |
| 8B | none | -1 | failed after 1,115 s, Vulkan device lost (0xc0000409) | | | | | 1 |
| 32B | none | 0 | 18.9 | 32.8 | 4.45 | 5/5 | 26,733 | 3 |

### What the data show

1. **TTFT was flat under the lock:** +1.4% to +2.0% of the +8 GB anchor at zero and -1 GB; the 32B at 3 MB available was 1.03x of its Section A anchors. Correctness held (5 of 5 probes in every cell that started).
2. **Decode fell 7% to 8.5%** (9.33 to 8.54 to 8.67 tok/s), outside the 5% band: a small decode cost is not ruled out. One anchor, no repeats.
3. **Paging happened (pages input up to 56,571 per second) without changing TTFT**, so the weight and KV traffic under pressure is at load, not in the timed prefill.
4. **The load-mode arms differ only at load and only once:** at -1 GB the mmap arm loaded in 23 s and the non-mmap arm stalled for 18.6 minutes and lost the device. One observation each.

### Limits

- One cell per level (35 of the 40 planned 8B cells and all 14B and 32B cells but one were trimmed). The 32B has a single cell.
- The mmap-off failure is one start and is UNVERIFIED as an effect of the load mode.
- Headroom is defined against the model's own requirement, so levels are not comparable to the Available-memory levels S of Phase D.

---

## evo-t2s overnight, Section D: not run (2026-09-26)

The SYCL section (`D_prepare` and three cells) was trimmed by the planner and never started. No SYCL claim is made. It is a candidate for the backfill added in commit 272cb43.

---

## PX2, evo-x2: pre-registered readout for separating the co-runner power effect from the memory-bandwidth effect (2026-09-29, RUN 2026-09-30, real results below the pre-registration)

**Status (original, kept for the record):** pre-registration only. No PX2 data exists. Nothing below is a result.
Recorded here before the run so the readout cannot be chosen after seeing the numbers.

**Hardware arm:** evo-x2 only (Ryzen AI Max+ 395, Strix Halo, Radeon 8060S, 2 CCDs, 8 physical cores and 16 logical
CPUs per CCD, 32 MiB L3 per CCD, no P/E split). PX2 rows must never be pooled with the evo-t2s B1/B2/B3/B4 rows: a
different vendor and a different core taxonomy make them separate arms.

**Question.** B1/B3 on evo-t2s showed a CPU co-runner slows iGPU inference, and B4 showed the effect tracks the E-core
grouping rather than the core count alone. Two mechanisms remain confounded there: the co-runner draws package power
that the shared budget then denies the iGPU, and it also moves DRAM traffic that competes with the iGPU on a unified
memory controller. On evo-x2 the two can be separated, because a spin co-runner and a STREAM-triad co-runner on the
same four cores draw comparable CPU power but move very different amounts of DRAM traffic.

**Design.** One llama-server per model, held up across all 9 conditions (only a model change restarts it, the B3/P70
pattern), context 8,192, prompt 2,048 tokens, 128 output tokens. Order per model is N0 first, then the 7 co-runner
conditions in an order shuffled with a logged seed (`PX2_SEED + crc(model_id)`, recorded on every row so the order is
reproducible from the data), then N1 last. The server is pinned to logical CPUs 0 to 3 in every condition including the
baselines, so N0 and S28 differ only in the hog.

| condition | co-runner | cores | placement |
|---|---|---|---|
| N0 | none | | clean baseline, always first |
| S2 | spin | 2 physical | hog CCD only |
| S4 | spin | 4 physical | hog CCD only |
| S8 | spin | 8 physical | hog CCD only (all of it) |
| S8x | spin | 8 physical | 4 per CCD, same core count as S8, traffic split across both L3 domains |
| S14 | spin | 14 physical | every core not held by the server |
| S28 | spin | 28 logical | both SMT threads of all 14 free cores |
| B4 | bandwidth | the same 4 physical cores as S4 | hog CCD only |
| N1 | none | | drift check, always last |

Core sets are derived from real cache topology (`harness/win_cpu_topology.py`, GetLogicalProcessorInformationEx with
RelationCache): an L3 group is a CCD, and an L2 group is one physical core's SMT sibling pair on Zen. Nothing is
hardcoded. Any physical core holding one of the server's logical CPUs is excluded whole, so no hog thread ever lands on
the SMT sibling of a core the server dispatches from. If the part's cache layout does not support that reading (a
shared-L2 cluster part such as evo-t2s, or a single-CCD part) the phase records `px2_disabled` with the reason and runs
nothing, rather than measuring the wrong core sets.

**Models.** qwen3-8b, qwen3-14b, qwen3-32b, llama-3.3-70b, llama31-8b. 5 measured calls plus 1 warm-up per condition,
reduced to 3 measured for llama-3.3-70b by the cut rule so the largest model cannot eat the deadline. 252 planned call
shapes in total (9 conditions x 6 calls x 4 models, plus 9 x 4 for the 70B). `measured_with_extra` cuts a condition to
3 measured calls anyway if its warm-up exceeds 90 s, so 252 is the planned maximum, not a guarantee.

**Pre-registered criteria, decided before the run.**

1. **Bandwidth is the dominant mechanism** if B4's median TTFT is at least 10% worse than S4's. Same cores, same core
   count, similar CPU power draw, so a gap of that size is attributable to DRAM traffic rather than to power.
2. **Power is the dominant mechanism** if B4 and S4 agree within 5% while both are at least 10% worse than N0. The
   co-runner then costs the same whether or not it touches memory.
3. **Neither is separable at this sensitivity** if B4 and S4 differ by between 5% and 10%, or if the S-ladder shows no
   monotone dose response from S2 to S14.
4. **L3 or CCD placement matters** if S8 and S8x differ by more than 5% at an identical core count. This is the AMD
   analogue of B4's E-cluster result on Intel.
5. **SMT contributes beyond core occupancy** if S28 is more than 5% worse than S14.

**Validity gates, all recorded per condition.**

- **Hog saturation:** every pinned logical CPU must read at least 95% "% Processor Time" at the end of the settle
  window (`hog_cpus_all_at_95`, `hog_min_core_pct`). A condition whose hog did not saturate its mask is not evidence.
- **Bandwidth positive control:** `bw_hog.py` is run solo once per session on B4's own cores before the sweep, and its
  achieved GB/s is recorded as `bw_gbps_solo` on every row. B4's in-sweep rate is read against that ceiling. A B4
  condition whose in-sweep rate is near the solo ceiling is definitely saturating the controller; the reported figure
  uses the STREAM 3-array convention and understates real traffic, so it is a lower bound.
- **Session drift:** N1 versus N0 median TTFT per model. Flagged when they differ by more than 3%. A flagged model's
  condition comparisons are suspect regardless of what the criteria above say.
- **Thermal settle:** a pre-condition thermal gate (3 C tolerance, 180 s cap) before the hog starts, recorded as
  `px2_gate_released_by` and `px2_gate_wait_s`, plus `do_call`'s own per-call gate on every row.

**Sensor availability on evo-x2, checked against the real `t2s_lab.Telemetry` and its AMD LibreHardwareMonitor feeder
rather than assumed.**

| field | on evo-x2 | source |
|---|---|---|
| `igpu_mhz` | available | Radeon GPU Core Clock, mapped by the LHM feeder into the same freq rows Sysman uses |
| `igpu_power_w` | **present but unusable for a contention/throttle signal** (2026-10-01) | Radeon GPU Core Power. Confirmed via the real `_lhm.jsonl` samples (257 PX2 rows' worth of window): reads near zero (median 0W, max 24-36W) in every single condition, including ones with heavy co-runner contention -- this 8B-class Vulkan-offloaded inference workload keeps the iGPU itself barely loaded throughout, so the sensor has nothing to report regardless of whether the package is at its power ceiling. A flat-zero reading here means "this workload doesn't load the iGPU," not "the iGPU is unaffected by contention" -- do not read a steady igpu_power_w as evidence either way |
| `temp_c_max` | available | Ryzen CPU Temperature. This is also what gives `lab.idle_temp` a real value, so the thermal gate uses its temperature path instead of Intel's package-power proxy |
| `igpu_temp_c_max` | available | Radeon temperature |
| `pkg_power_w`, `rapl_pp0_w`, `rapl_pp1_w` | expected null | Windows "Energy Meter" counter set only, whose instance names are Intel RAPL ones. The LHM loop maps no CPU power sensor into the per-call rows |
| `igpu_throttle_bits` | present but uninformative | the LHM feeder hardcodes 0 on the freq rows it synthesises, so a 0 here does not mean "not throttling" |
| `cpu_p_pct_perf`, `cpu_e_pct_perf`, `cpu_lpe_pct_perf` | meaningless here | Intel P/E/LP-E class groupings; this part has no P/E split. PX2 reads real per-logical-CPU "% Processor Time" instead |
| PPT, STAPM, power limits, throttle-reason registers | not captured | no plumbing for these exists anywhere in the repo |

Consequence for criterion 1: the iGPU clock side of the claim does have a sensor, because `igpu_mhz` is real on this
part. What is missing is CPU package power in the per-call row, so "the co-runner raised CPU package power by X W"
cannot be stated from the rows alone. `harness/lhm_sensors.ps1` does collect every CPU-group Power sensor into
`<prefix>_lhm.jsonl`, so CPU package power is recoverable offline from that raw file if LibreHardwareMonitor exposes it
on this part, which is itself unverified. The S4-versus-B4 contrast does not depend on that field: it separates the two
mechanisms by construction, through matched cores and different DRAM traffic, not by attributing a power number.

**Hour estimate.** 5.3 h by the harness's own `estimate_hours`, anchored on the real per-call medians measured on
evo-t2s in `results/t2s_night2_20260928T004924Z.jsonl` and `results/t2s_night2_20260928T200748Z.jsonl` (per-model
`load_s` and non-warmup `e2e_s`, plus a 70 s median thermal-gate wait). This is an over-estimate for two reasons and
should not be treated as a measurement: those medians were taken at a 7,368-token prompt while PX2 uses 2,048, and
llama-3.3-70b has never produced a timed call on either machine, so its calls fall back to a flat 45 s guess. There is
no evo-x2 per-call timing to anchor on: as of this commit no `results/*.jsonl` row carries `hw_id` `evo-x2`, and no
R1, A70 or P70 phase has produced rows on any machine yet. The estimate will be re-derived from evo-x2's own smoke and
N0 rows once they exist.

**Deploy note.** PX2 needs `bw_hog.py` and `win_cpu_topology.py` in the `scripts/deploy_evo.py` path list alongside
`spin_hog_affinity.py`, or the bandwidth arm cannot launch and the topology read fails on the machine.

### PX2 real results (ran 2026-09-30, `x2_px2`, completed)

File: `results/t2s_night2_20260930T135145Z.jsonl` (the main stem; `PX2` section rows plus `px2_model_summary`,
`px2_condition_done`, `px2_bw_calibration` records). All 5 models completed (llama-3.3-70b at the reduced 3
measured calls per the cut rule, the other 4 at the full 5).

**Verdicts against the 5 pre-registered criteria: none fire as literally worded, on TTFT, for any model.**
1. Bandwidth-dominant (B4 >=10% worse than S4 on TTFT): not met. Real gap is 2-3% across all 5 models (e.g.
   qwen3-8b S4 1.098x, B4 1.123x).
2. Power-dominant (B4 and S4 agree within 5%, both >=10% worse than N0): not met. S4 never reaches 10% worse than
   N0 for any model (max 9.8%, qwen3-8b), even where B4 does.
3. Neither separable (5-10% B4/S4 gap, or no monotone S2-S14 dose response): does not fire either -- the real gap
   (2-3%) sits below even this criterion's own 5% floor, and the dose response is monotone (not absent).
4. L3/CCD placement (S8 vs S8x >5%): not met, any model (max observed gap 1.2%, qwen3-32b).
5. SMT beyond occupancy (S28 >5% worse than S14): not met, any model (max observed gap 0.9%, llama31-8b).

**Validity gates:** drift (N0 vs N1) passes for all 5 models, |drift_pct| <=0.78%, well under the 3% threshold.
bw_hog solo calibration 32.212 GB/s; B4's in-sweep achieved rate (llama-3.3-70b) 24.159 GB/s, 75% of the solo
ceiling. Hog-saturation gate (`hog_cpus_all_at_95`) not independently re-checked in this write-up pass.

**Real finding, restated plainly, with the earlier "2-3% gap" phrasing corrected:** the 2-3% figure in the
criteria verdicts above is the gap **between B4 and S4**, not the gap of either one from the N0 baseline. Both
co-runners actually raise TTFT substantially vs N0 on evo-x2 (worked example, qwen3-8b: S4 1.098x, i.e. 9.8%
worse than N0; B4 1.123x, 12.3% worse than N0 -- a difference of about 2.3 percentage points between the two
hogs, "S4 max 9.8% worse than N0" being this same ceiling across the 5 models, with B4 exceeding 10% worse than
N0 for at least one model). So the evo-t2s-style "co-runner slows TTFT" headline effect is **not absent** on
evo-x2 -- TTFT does get worse under either hog, by a similar amount. What fails to replicate is the specific
claim PX2 was designed to test: that a bandwidth hog and a power/compute hog can be told apart **on TTFT**. They
can't -- B4 and S4 move TTFT almost identically (within 2-3% of each other), so criterion 2 (power-dominant)
comes within a hair of firing (S4 falls just short of the 10%-worse-than-N0 bar on the model where it gets
closest) while criterion 1 (bandwidth-dominant, which needs B4 to clearly beat S4) does not fire at all.
TTFT alone is the wrong instrument for this question on evo-x2.

The mechanism that does separate the two hogs cleanly lives in **decode throughput**: B4 (bandwidth hog) cuts
decode 7-10% below N0 consistently across all 5 models (ratio range 0.905-0.928x), while S4 (compute/power hog,
same 4 physical cores, comparable CPU occupancy) leaves decode essentially untouched (ratio range 0.987-1.000x).
The original design's 5 criteria are all written against TTFT, so they miss this distinction entirely even
though the underlying bandwidth-vs-power separation the experiment set out to find is real -- it just shows up in
a different metric. Read together with A-23 (evo-t2s, Intel, TTFT-visible, mechanism never established there
either), this makes the co-runner interference mechanism **vendor-dependent in where it is visible**: on AMD
Strix Halo the bandwidth/power distinction appears in decode throughput and not in TTFT, where the two hogs are
nearly indistinguishable; on Intel evo-t2s the only metric measured is TTFT and no bandwidth-vs-power separation
was ever attempted there, so the two platforms are not even answering the same question yet.

**n and ranges, per model, from the prose numbers already extracted above** (this pass did not recompute these
from the raw rows -- see the data-availability note below): n = 5 measured calls per condition for qwen3-8b,
qwen3-14b, qwen3-32b and llama31-8b, n = 3 for llama-3.3-70b (cut rule). Decode-throughput ratio (B4/N0) spans
0.905x to 0.928x across the 5 models; (S4/N0) spans 0.987x to 1.000x. TTFT ratio B4-vs-S4 gap spans the 2-3%
band for all 5 models, with qwen3-8b given as the worked example (S4 1.098x vs N0, B4 1.123x vs N0). No
per-model bootstrap CI is available from this write-up pass; at n=3-5 per cell a real CI would be wide, so these
are reported as ranges across the 5 models, not within-model confidence intervals, and should not be read as
such.

**Power/thermal readout (2026-10-01 correction): the file and its `_lhm.jsonl` sidecar exist on evo-x2 and were
pulled and committed** (`results/t2s_night2_20260930T135145Z.jsonl`, `..._lhm.jsonl`, `..._manifest.json`,
`..._sysman.csv`). The earlier claim in this section that package power is "expected null on evo-x2 by sensor
design" was wrong -- it was reasoning from the pre-registration's sensor table without the file in hand.
`pkg_power_w` is a real, populated field on every one of the 257 PX2 measurement rows (all 5 models, all 9
conditions), computed directly here (not from the prose above):

| model | N0 | B4 | S2 | S4 | S8 | S8x | S14 | S28 |
|---|---|---|---|---|---|---|---|---|
| qwen3-8b | 88.9 | 104.7 | 111.4 | 111.0 | 111.4 | 86.6 | 83.8 | 84.0 |
| llama31-8b | 91.8 | 105.8 | 112.0 | 111.6 | 111.7 | 88.9 | 84.3 | 84.0 |
| qwen3-14b | 93.3 | 87.7 | 95.2 | 95.4 | 91.6 | 83.7 | 83.6 | 83.7 |
| qwen3-32b | 83.6 | 83.6 | 83.6 | 83.6 | 83.6 | 83.6 | 83.6 | 83.6 |
| llama-3.3-70b | 83.6 | 83.6 | -- | -- | -- | 83.6 | 83.6 | -- |

(median package power, watts, per condition; n = 5 calls/cell for the 4 smaller models, n = 3 or 4 for
llama-3.3-70b per the cut rule; llama-3.3-70b's S2/S4/S8 cells were not run under the cut rule, shown as `--`.)

**Real finding this table adds:** qwen3-32b and llama-3.3-70b sit flat at ~83.6W across every single condition,
co-runner or not -- their own decode compute already saturates whatever headroom the co-runner hogs could use, so
the hogs add essentially nothing measurable to package power for these two models. qwen3-8b and llama31-8b, by
contrast, show a real +20W jump under the compute/bandwidth hogs (S2/S4/S8, ~111-112W) vs their own N0 baseline
(~89-92W) and vs the occupancy-only hogs (S8x/S14/S28, ~84-89W, close to N0). qwen3-14b sits in between. This is
a real, size-dependent interference signature in power draw that the TTFT/decode-throughput criteria do not
capture at all: smaller models leave real power headroom for a co-runner to consume; the two largest do not.

**iGPU clock and CPU temperature: confirmed available only for a subset of qwen3-8b's own conditions, from the
raw `_lhm.jsonl` sidecar directly (not the row-stamped fields, which are even sparser).** Joining the sidecar's
continuous sensor stream to each condition's real time window (bounded by that condition's own row timestamps)
gives LHM samples only for qwen3-8b/{N0, B4, S2, S4, S14} -- zero samples fall inside the time windows for
qwen3-8b/{S8, S28, S8x, N1} or for any of the other 4 models at all. The background LHM collector stopped
producing samples partway through qwen3-8b's own sweep and never ran again for the rest of the PX2 run -- a
real, now-precisely-bounded feeder gap (not "4 of 5 models never populated," but "the feeder died partway
through model 1 of 5 and the run proceeded without it"), worth fixing (keep-alive/restart on the LHM collector
process) before the next PX2-style run.

What the surviving qwen3-8b samples do show, median [max], now with iGPU power and package power alongside:

| condition | CPU temp C | iGPU clock MHz | iGPU power W | Package power W |
|---|---|---|---|---|
| N0 | 33.3 [51.6] | 602 | 0 [36] | 3.4 |
| B4 | 47.6 [56.3] | 601 | 0 [31] | 26.3 |
| S2 | 65.8 [66.9] | 602 | 0 [30] | 19.7 |
| S4 | 64.8 [65.9] | 602 | 0 [31] | 19.7 |
| S14 | **98.0 [98.1]** | 601 | 0 [24] | 81.5 |

(Package power medians here are computed over the same LHM-sample window as the other three columns, so they
differ from the row-stamped `pkg_power_w` medians reported earlier in this section, which are computed over
the full condition rather than just the window where LHM happened to be sampling -- both are real, they are
just two different, both-legitimate windows over the same quantity.)

**Real thermal-limit finding (qwen3-8b only; cannot be generalized to the other 4 models given the sampling
gap above):** the S14 condition (SMT/occupancy hog) drives CPU temp to a tight 98.0-98.1C plateau -- a real
thermal ceiling, not noise (the tight range across the whole condition window is itself the signature of a
throttle plateau, not a transient spike). No other condition for qwen3-8b gets within 30C of this.

**Does the iGPU clock hold while package power is at the ceiling (the "AMD firmware protects the iGPU under
contention" question)?** Yes, as far as this data can show it: iGPU clock stays flat at ~601-602 MHz across
every condition, including S14 where package power peaks at 81.5W median and CPU temp plateaus at 98C -- a
roughly 24x package-power range (3.4W to 81.5W) with essentially zero change in iGPU clock. But this result
needs a real caveat, not a clean confirmation: **iGPU power itself stays near zero (median 0W, max 24-36W)
in every condition**, meaning the iGPU is barely loaded throughout this whole workload -- consistent with
this 8B model's Vulkan-offloaded inference being memory-bandwidth-bound rather than iGPU-compute-bound (the
same mechanism PX2's main decode-throughput finding points to). A clock that never moves because the iGPU is
never under real load of its own is not the same evidence as a clock that holds steady while the iGPU itself
is under load and the package is at its ceiling -- this data shows the former, not the latter. The
"firmware protects the iGPU" claim is **not tested here**, only "the iGPU clock is unaffected by a co-runner
that stresses the CPU package while leaving the iGPU idle," which is a weaker and different claim.
`igpu_throttle_bits` remains unreliable (hardcoded 0 on LHM-synthesized rows per the pre-registration's own
sensor table) and was not used as evidence here; the CPU temp plateau and the power/clock table above are
the evidence.

**Reconstructing the missing coverage (S8, S28, S8x, N1, and all 4 other models):** no further LHM collector
data exists anywhere for this run -- confirmed by scanning the entire `_lhm.jsonl` sidecar's timestamp range
(it ends well before the S8/S28/S8x/N1 conditions or any non-qwen3-8b model's conditions even started) rather
than just the per-condition windows checked above. This is marked a real, unrecoverable gap for this run, not
reconstructed from anything else -- the next PX2-style run needs the LHM collector kept alive (or restarted)
across the whole sweep, not just model 1 of 5.

**The firmware-protects-the-iGPU question, retested with the hog's own achieved rate instead of the unusable
iGPU power sensor (2026-10-01).** Since `igpu_power_w` reads near-zero regardless of real contention (see the
sensor-availability table above), the test instead uses the co-runner hog's own measured throughput --
`hog_rate_during_calls` in `px2_condition_done`, real GB/s for B4 (bandwidth hog) and real iterations/s for
S14 (SMT/occupancy hog) -- as the signal: if the firmware is reserving bandwidth/cycles for the GPU, the
hog's own rate should drop when co-running with a model whose package sits at the power ceiling (qwen3-32b,
llama-3.3-70b) compared to a model with real headroom (qwen3-8b, llama31-8b), or compared to the hog running
alone.

B4 (bandwidth hog), GB/s, solo-alone calibration 32.212 GB/s:

| model | package regime | GB/s | % of solo |
|---|---|---|---|
| qwen3-32b | at ceiling (~83.6W) | 25.776 | 80.0% |
| llama-3.3-70b | at ceiling (~83.6W) | 24.159 | 75.0% |
| qwen3-14b | intermediate | 24.159 | 75.0% |
| llama31-8b | below ceiling (real headroom) | 28.186 | 87.5% |
| qwen3-8b | below ceiling (real headroom) | 22.411 | 69.6% |

S14 (SMT/occupancy hog), iterations/s, no solo-alone baseline exists for this condition in this run:

| model | package regime | iterations/s |
|---|---|---|
| qwen3-32b | at ceiling | 98,600,000 |
| llama-3.3-70b | at ceiling | 99,950,000 |
| qwen3-14b | intermediate | 101,800,000 |
| llama31-8b | below ceiling | 102,073,006 |
| qwen3-8b | below ceiling | 95,506,128 |

**n and CIs:** each cell above is a single real measurement (`hog_rate_during_calls` is one aggregate value
per model/condition in this run, not a per-call series) -- n=1 per cell, no confidence interval is computable
from this data, stated plainly rather than invented.

**Reading: this does not support "firmware prioritizes the GPU under contention."** For B4, the at-ceiling
group (75.0%, 80.0% of solo) is not consistently lower than the below-ceiling group: llama31-8b (87.5% of
solo, the least suppressed of all 5) fits the hypothesis, but qwen3-8b (69.6% of solo, the most suppressed of
all 5, even more than either at-ceiling model) directly contradicts it. For S14, the at-ceiling mean
(~99.3M/s) and below-ceiling mean (~98.8M/s) differ by under 0.5% -- no meaningful separation at all. With
n=1 per cell this is not a clean, directional pattern in either direction; the honest conclusion is that this
test is inconclusive on the firmware-protection question, not that it confirms or refutes it. A real answer
would need multiple real measurements per cell (to get an actual CI) and, ideally, a condition where the
iGPU itself is genuinely under load (which `igpu_power_w`'s near-zero reading says this workload never
produces) so the "protect the GPU" framing has something to protect in the first place.

This supersedes the "NOT YET RUN" status above. It also supersedes any downstream document that still cites PX2
as unrun, including `analysis/envelope_model.py`'s H2/co-runner multiplier, which is currently substituted from
claim A-23's evo-t2s CPU-co-runner TTFT ratio (1.39x-1.43x) and needs a refit. The refit is not a simple
drop-in of a PX2 TTFT number in place of A-23's: evo-x2's own TTFT ratios (S4 ~1.098x, B4 ~1.123x in the worked
example) are actually close in magnitude to A-23's evo-t2s range, so the multiplier's size may not need to move
much -- what needs to change is the model's structure, from one scalar TTFT multiplier to a representation that
also carries the decode-throughput effect (B4-specific, 0.905x-0.928x, bandwidth-attributable) that TTFT alone
does not capture on evo-x2. See the updated H2 entry in `docs/CLAIMS_LEDGER.md` (claim A-23) and the flag left in
`analysis/envelope_model.py`.

## R1b evaluation audit, both machines, from the live Oct-1-cut runs (2026-09-30)

Analysis of `t2s_night2_20260929T202603Z.jsonl` (evo-t2s) and `t2s_night2_20260929T205109Z.jsonl` (evo-x2), the
currently-running r1b_r1d jobs, pulled live and analyzed locally (`analysis/r1b_wrong_answer_audit.py`, added this
commit). All art_* probes, baseline (`arm1_baseline`) and self-report (`arm3_self_report`) arms only.

**Per-probe correctness grid (baseline arm).** Clean and consistent across both machines: every probe scores 0.0 at
ratios 0.4 and 0.85 (`artifact_fraction_retained=0`, the answer material never survives that much truncation) and
1.0 at ratio 1.2 (`artifact_fraction_retained=1`, full context, answer intact). Ratio 0.98 is the only ratio where
probes split: art_08/art_09/art_10 score 1.0 (their answer material sits early enough in the prompt to survive a 2%
cut) while art_01-art_07 score 0.0 (their answer material sits later, in the part that gets cut first). This
confirms the standing explanation -- correctness at a given ratio is decided by whether that probe's specific answer
position survives the truncation, not by anything else -- using the harness's own `artifact_fraction_retained` field
directly rather than re-deriving it.

**Refusal-classified outputs.** 18 refusals out of 480 t2s rows, 51 out of 872 x2 rows, all concentrated on a single
probe (art_05, a grid-reference-coordinates question) whose wrong-answer phrasing ("The provided text does not
mention any grid reference coordinates...") a narrow refusal regex initially missed entirely (first pass found 0 on
both machines; broadening the pattern to catch "no X was/were provided/recorded/mentioned" phrasing, not just
"no secret/code/value", found these). This is itself worth flagging: a heuristic classifier's true positive rate
depends heavily on exact phrasing and is not safe to trust uninspected -- exactly the reason the addendum's
tri-rater (heuristic / hand-label / LLM judge) validation exists rather than citing the heuristic alone.

**Refusal share of wrong answers, by model (the pre-planned "does scale change fabricate vs refuse" question) --
CORRECTED 2026-10-01 against the number register (docs/NUMBERS_REGISTER.md), real files now committed.**
This section's numbers below were written while `x2_r1b_r1d` was still a live, growing job (the original prose
here cited "872 x2 rows" total; the file has since grown to 1440 rows once llama31-8b and llama-3.3-70b were
added and the job completed/was cut) -- recomputed fresh from the final, now-committed files
(`results/t2s_night2_20260929T202603Z.jsonl`, `..._20260929T205109Z.jsonl`) via the real
`analysis/r1b_wrong_answer_audit.py::refusal_share_by_model` function:
t2s: qwen3-8b 9/192 (5%), qwen3-14b 9/198 (5%). x2: qwen3-4b-2507 12/198 (6%), qwen3-8b 11/192 (6%), qwen3-14b 9/198
(5%), llama31-8b 9/174 (5%), llama-3.3-70b 9/183 (5%), **qwen3-32b 41/199 (21%)**. The qwen3-32b figure was
reported as "19/111 (17%)" earlier today, itself a reconciliation of an even earlier hand-typed "41/82 (50%)"
that had no backing file at all -- that reconciliation was trust-based on this section's own prior internal
consistency (the four smaller-model counts summing to a stated x2 total), not an actual fresh recomputation,
since the source file was not available to recompute from at the time. It is available now, and the real
number is 41/199 (21%), not 19/111 (17%). qwen3-32b's refusal share is still roughly 4x the smaller models'
(not 3x as the stale figure suggested), still entirely driven by a single probe (art_05) in the current data,
still not a general pattern across probes -- now with llama family and 70B data present, and neither shows an
elevated refusal share (both at 5%, in line with the smaller models), so the effect looks qwen3-32b-specific
so far, not a general large-model effect. Treat this as a lead, not a finding, until it replicates on a second
probe or holds up under the tri-rater validation.

**Self-report truncation-awareness -- CORRECTED 2026-10-01.** 0 out of 240 (t2s) and **0 out of 720** (x2, not
435 as earlier reported -- same live-job/file-growth explanation as the refusal-share correction above) =
**0/960** self-report outputs mention that the input looked incomplete or truncated, at any ratio. The
conclusion is unchanged by the correction (still zero, now over a larger real sample): the self-report arm's
"AVAILABLE: yes/no" framing never once produces a model saying anything like "this context looks cut off" --
when the model reports the answer is unavailable, it says so as if the information were simply absent from a
complete document, never as evidence of truncation.

**Self-report scoring: the running jobs are currently writing wrong scores to disk, live, right now.** The
`strip_available_prefix` fix (commit `3f7b1f4`) is correctly present in the on-disk `scorers.py` on both machines
(confirmed: `C:\apu\ovn\scorers.py` on evo-t2s has the function; verified by reading the file directly). But the
`t2s_night2.py` processes that are actively running `r1b_r1d` on both machines were launched before that fix was
deployed and hold their own already-imported, pre-fix copy of the module in memory -- a running Python process never
re-reads a module from disk. Their live `score` field for the self-report arm is therefore still wrong, in real
time, on both machines: t2s reports self_report@1.2 mean 0.200, x2 reports 0.150. Rescoring the exact same raw
`output` strings from these same rows, offline, with the correct (disk) scorer gives t2s 0.950 and x2 0.900 at ratio
1.2 -- 57 of 222 t2s self-report rows and 111 of 402 x2 self-report rows change score under correct scoring, all of
them wrong-to-right. Ratios 0.4 and 0.85 are unaffected either way (0.000 under both the stale and correct scorer --
those are real truncation failures, not artifacts). Ratio 0.98 goes from 0.000 to 0.222 (t2s) / 0.194 (x2) under
correct scoring.

**No data loss and no rerun needed.** The raw `output` text is preserved regardless of which scorer wrote the
`score` field, so this is fully recoverable by rescoring from the saved JSONL once each run finishes -- restarting
either running job would lose its `--resume` continuity for no benefit, since nothing here depends on the process
being fixed mid-run. **Action needed before this run's self-report numbers are used anywhere:** any analysis of
these two specific stems (`t2s_night2_20260929T202603Z`, `t2s_night2_20260929T205109Z`) must rescore the self-report
arm from raw `output` via the corrected `score()` dispatcher rather than trust the stored `score` field, until that
rescoring is done once and the corrected values are written back or cached separately. Future runs are unaffected
once the process that reads scorers.py is itself started fresh after a deploy (already true for anything queued
after this point).

**CONFUSION category (item 4f): not computed this pass.** The visible post-truncation prompt text is not stored
per-row in these result JSONL files (only `full_tokens`/`target_tokens`/`chars_dropped` counts, not the prompt
string itself), so checking whether a wrong answer exactly matches another value present in the visible prompt needs
the exact visible prompt re-derived per row (deterministic from `item_id`/seed/ratio via the same builder R1b used),
which was not done in this pass. Flagged rather than skipped silently.

---

## A3/A6 audit: reconciling the qwen3-32b refusal-share discrepancy, real row examples, and a citation check (2026-09-30)

**SUPERSEDED 2026-10-01, see the R1b evaluation audit section above and docs/NUMBERS_REGISTER.md.** The source
file this section could not reach on 2026-09-30 (`results/t2s_night2_20260929T205109Z.jsonl`) has since been
pulled and committed. The real, freshly-computed number is **41/199 (21%)**, not 19/111 (17%) as this
section's own verdict below concluded. That verdict was itself trust-based (internal consistency against this
doc's own prior prose), not a fresh file computation, since the file was unavailable at the time -- stated
plainly in this section's own text below ("19/111 is real but can only be confirmed by provenance... not
independently recomputed from files in the repo today"). The rest of this section (A6's verbatim samples and
citation check) is unaffected and stands as written.

### A3 (historical, see supersession note above): 41/82 (50%) vs 19/111 (17%) -- which is real

**Verdict: 19/111 (17%) is the only one of the two numbers that is grounded in this repository.
41/82 (50%) does not appear anywhere in this project's committed history, on any branch, in any
doc, result file, or analysis script, and cannot be reconstructed from any committed row data.**

What was checked (`git log --all -p` over the whole repo, plus a literal search of every tracked
file at the current tip): the string "19/111" occurs in exactly one place anywhere in this
project's history -- the paragraph added by commit `6368afa` ("R1b audit: per-probe grid, refusal
dumps, self-report rescore"), which is the same paragraph present today in the section directly
above this one:

> t2s: qwen3-8b 9/192 (5%), qwen3-14b 9/198 (5%). x2: qwen3-4b-2507 12/198 (6%), qwen3-8b 11/192
> (6%), qwen3-14b 9/198 (5%), qwen3-32b 19/111 (17%).

That number's stated source is `analysis/r1b_wrong_answer_audit.py` run against
`results/t2s_night2_20260929T202603Z.jsonl` (evo-t2s) and `results/t2s_night2_20260929T205109Z.jsonl`
(evo-x2) -- described in the same section as "the currently-running r1b_r1d jobs, pulled live and
analyzed locally." Those two exact filenames do not exist anywhere in this repository (checked
at every commit that touches `results/` and at the current tip; no file with either timestamp was
ever committed). The only `t2s_night2_*.jsonl` files actually committed are
`results/t2s_night2_20260928T004924Z.jsonl` (300 lines) and `results/t2s_night2_20260928T200748Z.jsonl`
(225 lines) -- running `analysis/r1b_wrong_answer_audit.py` against both of these directly gives
**0 R1b art_* rows in either file** (verified by loading both files and filtering for
`kind == "call"` and `probe_id` starting with `art_`: zero matches, zero `arm1_baseline`/
`arm3_self_report` rows). These two files are from the night2 B1/B2/C1 core-set-sweep phases
(the work landed in commit `13a5be2`, before R1b existed), not from the R1b phase at all. So the
19/111 number is real and correctly attributed to its own analysis, but its two raw source files
were never pulled off the remote machines into this repository -- they existed only transiently
("pulled live" means read off evo-t2s/evo-x2 directly during the live run, not saved) -- which
means the number cannot be independently recomputed from any file in this repo today. It can only
be confirmed by provenance (one script, one commit, internally consistent numbers) rather than by
rerunning the script on committed data.

Internal-consistency check on the 19/111 paragraph itself: the same paragraph states "51 [refusals]
out of 872 x2 rows" for all models combined on evo-x2. Summing the four per-model x2 refusal counts
it also gives -- qwen3-4b-2507 12, qwen3-8b 11, qwen3-14b 9, qwen3-32b 19 -- gives 12+11+9+19 = 51,
exactly matching the stated x2 total. This is consistent with 19/111 being a real tabulation from
one real run, not a transcription error or a guess.

The one row-level artifact that *is* committed and does reference qwen3-32b wrong answers is
`results/labeling/r1b_wrong_sample.csv` (and its blinded twin), a 150-row stratified sample built
by `analysis/build_r1b_wrong_sample.py` (commit `b6c8cae`, refreshed in `c7cb214`) from a
different, later pull (the commit message says "from x2_r1b_scale") that is also not itself
committed as a raw JSONL. That sample contains only 20 qwen3-32b rows (counted directly from the
CSV), of which 5 are heuristically labeled REFUSAL and 15 FABRICATION -- 5/20 = 25%. This is a
*stratified sample* (round-robin across model/ratio/arm/machine strata, capped at 150 rows total),
not the full wrong-answer population, so 25% from 20 rows is not a competing estimate of the true
share and is not directly comparable to 19/111 from (apparently) 111 full-population wrong rows;
it is cited here only because it is the real source used for the row examples in the next section.

**41/82 (50%) traces to nothing.** No commit, in any branch reachable from `main`, ever added the
string "41/82," "41 / 82," or "50%" anywhere near "refusal" or "qwen3-32b." It is not an earlier,
narrower-regex version of the same computation either: the FINDINGS text itself records that the
*narrow* (pre-fix) regex found 0 refusals on both machines for the R1b run, not 41, and the
narrow-vs-broad regex history in `analysis/r1b_wrong_answer_audit.py` / `analysis/build_r1b_wrong_sample.py`
only ever contains the one (broadened) pattern as committed -- there is no narrower committed
version that would yield 41. It does not match the stratified sample's qwen3-32b subset (20 rows,
not 82) or any arithmetic combination of the other per-model counts in the same paragraph. The
most likely explanation is that 41/82 is a misremembered or miscommunicated figure from outside
this repository's committed record -- possibly a conflation with a different model, a different
phase's row count, or a verbal approximation -- not a stale-but-real number from an earlier
classifier version or an earlier scope. **Conclusion: 19/111 (17%) stands as the project's number
for qwen3-32b's refusal share of wrong answers; 41/82 (50%) should not be cited and has no
identifiable source file.**

### A6(a): real qwen3-32b refusal and fabrication rows

Source: `results/labeling/r1b_wrong_sample.csv` (committed, commit `c7cb214`), filtered to
`model == qwen3-32b`. This file contains every qwen3-32b row available anywhere in the repository
with a verbatim model `output` string -- **the full population only has 5 rows this file's own
heuristic labels as REFUSAL**, not 10. Asking for 10 real refusal rows cannot be satisfied without
inventing 5 of them, which this audit will not do; all 5 real REFUSAL rows are listed below in
full, followed by 5 of the 15 real FABRICATION rows. None of the 15 outputs contain `<think>`
tags or any other reasoning-mode marker (checked by literal substring search across the whole
column) -- qwen3-32b's outputs in this sample are short, direct answers with no visible chain of
thought, consistent with either non-thinking mode being used or the harness storing only the
post-thinking final text.

**Refusal rows (5 of 5 available):**

| id | probe | expected | thinking tags? | ratio/arm | verbatim output |
|---|---|---|---|---|---|
| r1bw_029_evo-x2_qwen3-32b_art_06 | art_06 | `Blum` | no | 0.85 / arm1_baseline | "The provided text does not mention who seconded the motion to retire the DELPHI-2 indexing service." |
| r1bw_045_evo-x2_qwen3-32b_art_06 | art_06 | `Blum` | no | 0.4 / arm1_baseline | "The provided text does not mention who seconded the motion to retire the DELPHI-2 indexing service." |
| r1bw_085_evo-x2_qwen3-32b_art_02 | art_02 | `0.0073` | no | 0.85 / arm1_baseline | "The information provided does not include the alert_threshold_ppb for unit CHROM-7." |
| r1bw_094_evo-x2_qwen3-32b_art_02 | art_02 | `0.0073` | no | 0.98 / arm1_baseline | "The information provided does not specify the alert_threshold_ppb for unit CHROM-7." |
| r1bw_100_evo-x2_qwen3-32b_art_02 | art_02 | `0.0073` | no | 0.4 / arm1_baseline | "The provided text does not contain information about the alert_threshold_ppb for unit CHROM-7." |

Note: all 5 real refusal rows come from only 2 of the 10 art_* probes (art_02 and art_06), both on
evo-x2, all in `arm1_baseline`. This is too thin a sample (5 rows, 2 probes, 1 machine) to say
anything about qwen3-32b's refusal behavior in general; it is reported here only because it is
every real refusal-labeled row this project has committed for this model.

**Fabrication rows (5 of 15 available, chosen to show the range of patterns present):**

| id | probe | expected | thinking tags? | ratio/arm | verbatim output |
|---|---|---|---|---|---|
| r1bw_041_evo-x2_qwen3-32b_art_03 | art_03 | `8.9` | no | 1.2 / arm1_baseline | "33.6" |
| r1bw_038_evo-x2_qwen3-32b_art_04 | art_04 | `DELETE` | no | 0.98 / arm1_baseline | "User tcosta did not perform any action at 14:02." |
| r1bw_140_evo-x2_qwen3-32b_art_07 | art_07 | `3.11.9` | no | 0.85 / arm1_baseline | "None of the provided administrative log entries mention a version number or reference to CVE-2024-51022. Therefore, it is not possible to determine which version number patched CVE-2024-51022 from the" [excerpted: row value ends mid-sentence in the source CSV] |
| r1bw_086_evo-x2_qwen3-32b_art_06 | art_06 | `Blum` | no | 1.2 / arm3_self_report | "AVAILABLE: yes, Blum" |
| r1bw_141_evo-x2_qwen3-32b_art_08 | art_08 | `PN-38901` | no | 1.2 / arm3_self_report | "AVAILABLE: yes, PN-38901" |

The last two rows above are not fabrications in the ordinary sense -- the model's output literally
contains the correct expected value (`Blum`, `PN-38901`) -- and their presence in a "wrong answer"
sample is itself evidence for a finding already on record in this file: the self-report arm's
on-disk `score` field was stale for exactly this run due to the in-process scorer bug described
above ("Self-report scoring: the running jobs are currently writing wrong scores to disk, live,
right now"), which the FINDINGS text says flips exactly these kinds of rows from wrong-to-right
under correct rescoring. These two rows were sampled as "wrong" by `score != 1.0` at build time;
they are consistent with being two more instances of that same stale-scorer artifact rather than
real model fabrications, though this was not independently re-verified per-row here.

### A6(b): citation check, `docs/PRIOR_ART.md`

**arXiv 2609.19169 ("SiliconBench").** `docs/PRIOR_ART.md` cites this twice (claim (b), line ~372,
and claim (c), line ~424) as "SiliconBench (arXiv:2609.19169, Sept 2026) evaluates 'speed, memory,
and fidelity' together" on Apple Silicon serving engines. **Verified directly** (fetched
`arxiv.org/abs/2609.19169`): the paper exists and its real title is **"SiliconBench: Speed, Memory,
and Fidelity for LLM Serving on Unified-Memory Desktops"** (Zhang, Fan, Munhá Correia, Cheema,
Zhang). It evaluates nine Apple Silicon serving engines on speed, memory consumption, and output
fidelity, and its abstract/results include the "explicit memory budgets do not guarantee memory
headroom" finding that `PRIOR_ART.md` quotes verbatim. The citation is real, correctly titled (by
inference from the fetched content, not restated verbatim from a title field), and used
accurately -- `PRIOR_ART.md`'s characterization matches the paper's actual content.

**GitHub issues/PRs referenced in `docs/PRIOR_ART.md`.** All 18 distinct references were fetched
directly and checked against `PRIOR_ART.md`'s description of each. All 18 resolved to real,
accessible issues whose actual titles and content match what `PRIOR_ART.md` claims for them:

| reference | real title | what it establishes (per PRIOR_ART.md's use of it) |
|---|---|---|
| ggml-org/llama.cpp#17284 | Eval bug: Server fails with HTTP 400 (context size exceeded) instead of truncating chat history | server hard-fails on context overflow rather than degrading gracefully |
| ggml-org/llama.cpp#11577 | Feature Request: resize an existing context | no built-in way to resize context without save/reload |
| ggml-org/llama.cpp#18889 | Eval bug: `llama_model_fit` results in zero context size | context-fit estimation has a real bug path |
| ggml-org/llama.cpp#19745 | llama-server/llama-cli hang/crash during RPC tensor upload for large models (AMD/HIP) | AMD path has its own distinct OOM/hang failure mode |
| ggml-org/llama.cpp#18946 | ErrorOutOfDeviceMemory -- critical out-of-device-memory and memory-accounting failures (Intel Vulkan/SYCL, Lunar Lake) | Intel Vulkan path has its own distinct OOM failure mode, including bad memory accounting |
| ggml-org/llama.cpp#1866 | CUDA out of memory -- but there's plenty of memory | NVIDIA path OOMs despite free VRAM being reported |
| espetro/llama.cpp#2 (fork) | State silently truncated to `row_cap`, no usage field in the API response to detect it | real repo, real issue, confirmed to be a fork of ggml-org/llama.cpp with a custom addition ("Kev System One"); the silent-truncation-with-no-usage-field shape is real but is specific to the fork's custom endpoint, not core llama.cpp -- `PRIOR_ART.md`'s own framing ("relevant to a/d framing") already treats it as an analogy rather than a core-llama.cpp bug, which matches |
| lmstudio-ai/lmstudio-bug-tracker#2404 | Context length silently clamped to ~6,656 tokens regardless of model size, free memory, or parallelism (Apple Silicon, 24GB) | silent context clamping on Apple unified memory, no error surfaced |
| ollama/ollama#14073 | New default context lengths will break | tiered VRAM-based default overflows VRAM on a real 52 GiB system |
| ollama/ollama#14116 | Tiered context length can exhaust VRAM | `OLLAMA_NUM_PARALLEL` not factored into the VRAM tier calculation |
| ollama/ollama#12353 | Feature Request: Auto-size num_ctx to a user VRAM budget (and recalc on model switch) | users asking for auto-fit because over-guessing silently overflows VRAM |
| ollama/ollama#9774 | Estimate of VRAM needs based on context length and quantization | users have no way to predict VRAM needs from context size |
| ollama/ollama#9890 | Large context size completely breaks the usability of the model | real user-reported unresponsiveness when raising context size |
| ollama/ollama#14173 | Ollama 0.15.6 ignores requested context size | requested context size silently not honored |
| ollama/ollama#11964 | context size larger than set | requested (smaller) context size silently overridden to a larger one |
| ollama/ollama#18229 | Loaded context length: source not shown (default/env/Modelfile/request) | no visibility into which precedence level set the active context length |
| ollama/ollama#11659 | Allow users to set context length like we used to instead of terrible GUI presets every launch | env-var override broke, confirms tiering is often unwanted |
| ollama/ollama#18242 | Can't set custom values for Context Length | custom `OLLAMA_CONTEXT_LENGTH` silently rejected, falls back to default |

All 18 GitHub references and the arXiv paper were verified directly (via live fetch of the actual
page content), not inferred from memory or from `PRIOR_ART.md`'s own description. 19/19 citations
checked resolve to real, relevant content; 0 are unresolved. No citation in this list needed to be
reported as "unverified, no web access" -- web access was available for this audit.

---

## K2 arm (c), everyday_apps: pre-registered kill criterion (2026-09-29, NOT YET RUN)

**Motivation.** K2's existing pressure arms (AWE-locked balloon, ordinary pageable touch, see
`harness/t2s_k2_pressure.py`) are both synthetic memory-pressure mechanisms. A real agent session on a user's own
machine is far more likely to lose memory to ordinary background applications (a browser with several tabs open)
than to either of those two mechanisms. Arm (c), everyday_apps, replaces the synthetic pressure source with a
headless browser holding 20 local static pages open, each allocating roughly 150 MB of JS heap on load (see
`harness/browser_pressure.py`), started at a fixed session turn and held for 10 turns before a clean process-tree
kill, run against Ollama (default fit/tier) and llama-server (default fit) on llama3.1:8b and qwen3:4b-instruct-2507
(4 combinations).

**Claim under test.** Quality degrades silently under everyday-app memory pressure: a quality drop can occur during
the everyday_apps arm with no corresponding surfaced error.

**Pre-registered kill criterion, decided before the arm's run code was written (only the page-generation and
browser-lifecycle plumbing existed at the time this section was written; wiring the arm into a real K2 run did
not).** The "degrades silently under everyday-app memory pressure" claim fails if every quality drop observed
during the everyday_apps arm coincides with a surfaced error (a non-200 HTTP status, or a non-null error field) in
that same call. A silent, error-free quality drop is required to support the claim -- a quality drop that always
shows up alongside a loud, surfaced failure is not silent degradation, it is an ordinary failure the caller can
already detect and is not evidence for this claim.

Only calibration-passing Q0 tasks are eligible for this check (`quality_suite.load_calibration_pass_set`,
excluding common_words_extraction-style tasks per the 2026-09-29 calibration-gate fix) -- a score change on a task
known to be miscalibrated for a given model is not usable evidence either way.

---

## R2, agent session growth: pre-registration (2026-09-29)

**Status:** PRE-REGISTRATION. No R2 run exists yet. This section is written before any session-growth data is collected, per this repo's standing rule that a kill criterion is recorded before the run that could satisfy or fail it.

**Design:** `harness/t2s_r2_session_growth.py` builds deterministic, seeded agent sessions (a turn-0 system prompt carrying 5 checkable rules, a 2-tool schema and 3 recall facts, followed by up to 80 turns each requiring one tool call plus a ~1.5k-token simulated tool result, with a recall question every 4th turn) and scores each turn's rule compliance, tool-call validity, fact recall and sent-vs-processed token gap. Five runtime arms (Ollama default / Ollama num_ctx=131072 / Ollama num_ctx=32768 on evo-x2 only / llama-server default fit / llama-server -c 131072) are crossed with two memory conditions (as-is, and an occupying process holding ~40 GB) on llama3.1:8b, qwen3:4b-instruct-2507 and qwen3:8b, 3 seeds per cell, temperature 0.

**The claim this phase is built to support:** on a memory-constrained machine, a long-running agent session silently loses track of its own system-prompt rules and recall ability as it grows, purely because the runtime silently truncates or shifts context once it exceeds whatever tier the runtime picked for the available memory, with no error ever surfacing.

### Kill criterion

The silent-failure claim for R2 fails if, in every arm, the first rule or tool failure happens only after an explicit error (non-200 HTTP status, or an error field in the response) had already surfaced in that same session.

This is checked mechanically by `evaluate_kill_criterion()`: for each arm, a session counts as showing a silent failure only if its first rule/tool/recall failure turn exists and no error (HTTP non-200 or an error field) was recorded strictly before that turn in the same session. If every arm's sessions fail only after such an error has already surfaced, R2's central claim is killed -- the failures would be a known-and-reported degradation, not a silent one, and would not support the memory -> runtime-chosen context -> silent truncation -> lost rules argument this phase exists to make.

### What would NOT kill it

An arm where the model's very first rule violation (e.g. it stops appending the session code, or answers a length in feet) or its first invalid tool call happens with no prior 400/500 response and no error field anywhere earlier in that session. A single such session in a single arm is enough to keep the claim alive; the criterion only fires if it is absent from every arm.

### Limits of this pre-registration

- No run has been executed against this criterion yet; this section only fixes the rule in advance.
- The rule is evaluated per session, then aggregated per arm as "does at least one session in this arm show a silent failure" -- it does not require a majority of seeds in an arm to be silent, only at least one, since even one clean demonstration of silent failure is sufficient evidence that the runtime can lose rules with no error.
- `error_surfaced_before_failure` as computed by `score_session()` only sees errors captured on turns up to and including the first failure turn; an error on a later turn does not retroactively satisfy the criterion, since the failure being silently preceded by nothing is exactly what is being tested for.

---

## K2 arm (d), pause_resume: pre-registered predictions (2026-09-30, NOT YET RUN)

**Motivation.** K2's existing pressure arms (AWE-locked balloon, ordinary pageable touch, everyday_apps) all test
memory pressure applied DURING active inference: the model is loaded and answering calls the whole time pressure is
present. A different, equally realistic scenario is not covered by any of them: the model sits idle between agent
steps, something else (a browser) loads memory in the background while it is idle, Ollama's own keep_alive timer
expires and unloads the model, and when the agent resumes, the model reloads under the NEW memory conditions --
possibly with a different GPU-layer placement or a different context size than it started the session with, silently,
with no error. Arm (d), pause_resume, exists to test this reload path specifically (see `harness/t2s_k2_pressure.py`,
`run_pause_resume_run`/`phase_k2_pause_resume`, and `harness/browser_pressure.py`'s page-generation code, reused
rather than reimplemented for the background app load).

**Design summary.** Session turns 1-10 run normally against Ollama with its DEFAULT keep_alive (5 minutes -- a
deliberate, arm-specific exception to the 0-keep_alive convention every other K1/K2 Ollama call site uses for
contamination-avoidance; recorded on every row of this arm precisely because it is an exception, not the norm). The
Chromium memory load from `browser_pressure.py` then opens, scaled in steps of 0/8/16/24/32 GB total browser memory.
The session then idles for 6 minutes (real wall-clock on a live run, injectable via a `sleep_fn` parameter for tests)
-- longer than the 5-minute keep_alive, so the model actually unloads. Turns 11-30 then continue. At both the
initial load (turn 1) and the reload (turn 11), this arm records: the offloaded GPU layer count and the context size
actually loaded (from Ollama's server.log and GET /api/ps), `size`/`size_vram` from GET /api/ps, and the server.log
placement lines verbatim. Run on both evo-x2 (unified memory -- dedicated and shared GPU usage, where the telemetry
layer exposes it) and evo-t2s (Intel path; see the code-level note in `t2s_k2_pressure.py` on which telemetry fields
are actually available on each vendor -- they are not the same fields, and this file does not assume they are).

**Claims under test (two competing predictions, not one).**

- **(P1)** Mid-session, with the model already loaded, opening apps does NOT change the context or quality; any
  effect is speed (paging) or a crash. Under P1, the 0 GB and higher-GB steps should look identical at both the
  initial load and the reload: same layer placement, same context, and the only difference (if any) between GB
  steps is TTFT/decode speed, not correctness or the presence of an error.
- **(P2)** After an idle gap longer than keep_alive, the model reloads under the new memory state, and the reload
  silently changes placement (CPU-offloaded GPU-layer count, read from server.log) and/or context, with a quality
  and/or speed change and no error surfaced. Under P2, some GB step's reload placement/context should differ from
  its own initial-load placement/context (or from the 0 GB step's reload), with no HTTP error and no error field
  anywhere in that step's rows.

**This run is designed to determine which of P1/P2 holds** -- or neither (e.g. every placement/context change
coincides with a surfaced error, the way R2's and everyday_apps' own kill criteria are framed), or both under
different conditions (e.g. P1 at low GB steps, P2 only past some threshold GB step). The report from an eventual
real run must state explicitly, per step and overall, which prediction held; `pause_resume_report()` (one function,
one call, the whole per-step table: placement before/after, context before/after, TTFT/decode before/after, quality
before/after, any error surfaced) and `pause_resume_prediction_verdict()` (P1/P2/inconclusive per step) exist so
that determination is mechanical, not a judgment call made after the fact.

**Control.** The spec asks for "the same pause-and-resume sequence with no app load at all, so the reload itself is
not the cause." The 0 GB step in the 0/8/16/24/32 GB sweep already is exactly this: at 0 GB, `run_pause_resume_run`
takes the identical code path (same turn counts, same keep_alive, same idle `sleep_fn` wait) and simply never calls
`EverydayAppsPressure.start()` at all -- no browser process is launched, so there is no separate "no app load"
variant to build. **Conclusion: the 0 GB step IS the control**, not a separate arm variant; a placement/context
change observed at 0 GB (if any) isolates "does merely unloading and reloading, with nothing else going on, itself
change anything" exactly as the spec's control wording asks for, and any change observed only at GB > 0 isolates the
app-load effect specifically.

**Status:** PRE-REGISTRATION. No pause_resume run has been executed against these predictions yet; this section is
written before any of arm (d)'s run-execution code exists, per this repo's standing rule (see the everyday_apps and
R2 pre-registrations above) that a kill criterion or prediction set is recorded before the run that could satisfy or
fail it.

## K1 v3, evo-t2s: the 4096 default context is Ollama's own integrated-GPU policy, not an Intel hardware limit (2026-10-01)

**Hardware arm:** evo-t2s only (Intel Core Ultra X7 358H, Arc iGPU, Ollama 0.33.2). Not the BOM target, never
pooled with evo-x2.

**This replaces the earlier "measurement gap" framing of this result with a mechanism finding.** The prior report
(same K1 v3 probe set) treated T2S's `ollama_default_ctx == 4096` for all three models as an unexplained host-level
divergence from evo-x2, where the same three models land at their own native/uncapped context. It is not a gap:
the real `ollama_serve.log` from a live, read-only `/api/ps` check (pulled under the maintenance lock, 2026-10-01)
shows the exact mechanism, verbatim:

```
time=2026-10-01T12:59:00.253-07:00 level=INFO source=runner.go:405 msg="dropping integrated GPU; to enable, set OLLAMA_IGPU_ENABLE=1" id=0 library=Vulkan compute=0.0 name=Vulkan0 description="Intel(R) Arc(TM) B390 GPU" pci_id=""
time=2026-10-01T12:59:00.254-07:00 level=INFO source=types.go:50 msg="inference compute" id=cpu library=cpu compute="" name=cpu description=cpu libdirs=ollama driver="" pci_id="" type="" total="63.5 GiB" available="50.5 GiB"
time=2026-10-01T12:59:00.254-07:00 level=INFO source=routes.go:2058 msg="vram-based default context" total_vram="0 B" default_num_ctx=4096
```

Ollama's own GPU discovery correctly identifies the iGPU (`Vulkan0`, "Intel(R) Arc(TM) B390 GPU") and then explicitly
drops it by policy, falling back to CPU. Because no GPU is counted, `total_vram="0 B"`, and Ollama's own
vram-based default-context sizing floors at 4096 in that case. `OLLAMA_IGPU_ENABLE` was confirmed unset in the
full `OLLAMA_*` env-var dump from the same session (no entry for it at all, i.e. Ollama's own default applies).

**Live-verified fix, read-only in the sense that it changes nothing on disk, only the server's own env var for one
test invocation:** starting `ollama serve` with `OLLAMA_IGPU_ENABLE=1` and loading llama3.1:8b again:

| | OLLAMA_IGPU_ENABLE unset (default) | OLLAMA_IGPU_ENABLE=1 |
|---|---|---|
| size_vram | 0 | 9189206261 (8.56 GiB, full model) |
| context_length | 4096 | 32768 |

Setting one environment variable moves the model fully onto the iGPU and raises the default context 8x. **This
confirms the hypothesis: the 4096 ceiling on evo-t2s is Ollama's own integrated-GPU opt-out default, not a hardware
or driver limitation of the Arc iGPU.**

**evo-x2's own server.log, quoted verbatim (2026-10-01, no new run, `C:\apu\ovn\ollama_serve.log` on evo-x2):**

```
level=INFO source=runner.go:405 msg="dropping integrated GPU; to enable, set OLLAMA_IGPU_ENABLE=1" id=0 library=Vulkan compute=0.0 name=Vulkan0 description="AMD Radeon(TM) 8060S Graphics" pci_id=""
level=INFO source=types.go:32 msg="inference compute" id=0 filter_id=0 library=ROCm compute=gfx1151 name=ROCm0 description="AMD Radeon(TM) 8060S Graphics" libdirs=ollama,rocm_v7_1 driver=0.0 pci_id=0000:c5:00.0 type=iGPU total="99.7 GiB" available="99.6 GiB"
level=INFO source=routes.go:2115 msg="vram-based default context" total_vram="99.7 GiB" default_num_ctx=262144
```

This settles the T2S vs X2 gap directly, no elimination required: the AMD iGPU **is** dropped by the identical
Vulkan-backend opt-out policy Intel's iGPU hits. But Ollama then separately discovers the same physical device
through a second backend, ROCm, which carries no such opt-out -- so evo-x2 ends up using the iGPU anyway, via
ROCm, while evo-t2s (no ROCm path exists for Intel hardware in Ollama) has no fallback once Vulkan drops it and
falls through to CPU. The gap is not an AMD-vs-Intel capability difference; it is that Ollama ships a second,
opt-out-free discovery path for AMD and not for Intel.

**Half-context overflow rule (2026-10-01, `ollama-overflow-keeps-half` in the register).** Across four real,
independently-checked configurations, an over-length prompt's processed tokens after overflow equal
`floor(num_ctx / 2) + 2`, every time, with HTTP 200 (not an error):

| num_ctx | processed after overflow | config |
|---|---|---|
| 4096 | 2050 | Ollama stock default, evo-t2s |
| 32768 | 16386 | Ollama `OLLAMA_IGPU_ENABLE=1`, evo-t2s |
| 40960 | 20482 | Ollama stock default, evo-x2 |
| 8192 | 4098 | live template-mechanism test, evo-x2 |

Mechanism, cited rather than guessed: Ollama's own internal `llama-server` invocation carries `--context-shift
--keep 4` on every real command line captured this session (both hosts). This is llama.cpp's own context-shift
feature: once the context fills, the oldest tokens are discarded (keeping only the first `--keep` tokens, 4 here,
negligible) and the window slides to retain the most recent content -- a tail-keeping sliding window, not a
model-specific or host-specific quirk. A live marker probe (front/middle/end markers planted in a ~12,000-word
prompt, num_ctx=8192) is consistent with the front marker not surviving, but the model also failed to report an
end-adjacent marker that plausibly fell inside the retained tail -- reported as inconclusive on exact survival,
most likely a model-recall limitation at this context pressure rather than evidence against tail-retention.

**With `OLLAMA_IGPU_ENABLE=1`, prompts up to the full reported default context pass intact** (no truncation,
marker found, HTTP 200); only prompts exceeding it get cut to this half-window, still returning HTTP 200 rather
than erroring -- this corrects the earlier table's wording, which described the flag's ceiling without stating
that sub-ceiling prompts are unaffected.

**T2S vs X2 default-context table** (from the K1 v3 register rows, `ollama_default_ctx` per model):

| model | evo-t2s default ctx | evo-x2 default ctx | native_ctx |
|---|---|---|---|
| qwen3-4b-2507 | 4096 | 262144 | 262144 |
| llama3.1:8b | 4096 | 131072 | 131072 |
| qwen3:8b | 4096 | 40960 | 40960 |

**Engine/source mechanism (item 1b): the overflow behavior follows how the model was loaded into Ollama, not the
model itself.** At the same 4096 default, live-tested:

| model | source | overflow behavior at 4096 default |
|---|---|---|
| qwen3-4b-2507 | `ollama create` from local GGUF | HTTP 400 (hard error, `exceed_context_size_error`) |
| llama3.1:8b | `ollama pull` (library) | HTTP 200, silent truncation (`token_truncated=True`) |
| llama3.1:8b | `ollama create` from its own local GGUF (same file) | HTTP 400 (hard error, identical message) |
| qwen3:8b | `ollama pull` (library) | HTTP 200, silent truncation, pinned at 2050 tokens processed (half of 4096) |

The same model (llama3.1:8b) produces opposite overflow behavior depending on creation method alone: a bare
`FROM <gguf>` Modelfile (no other directives) hard-errors, while the library-pulled version (whose Modelfile
Ollama ships with additional template/parameter directives not present in a bare GGUF import) silently truncates.
This settles item 1b: **the behavior follows the Modelfile/creation path, not the model architecture or the
underlying llama-server engine** (both paths use the same `llama-server.exe` binary per `ollama_serve.log`).

**Register additions:** `T2S-vs-X2-default-ctx` (the table above) and the fraction-of-trace-steps-exceeding-4096
row are added to `analysis/numbers_register.py` (see item 1c in the 2026-10-01 report for the computed fraction
and its data file).

## PRE-REGISTRATION: K1 v3 probe set under IPEX-LLM Ollama and llama.cpp Vulkan llama-server (not yet run)

**Hypothesis under test:** the default context and overflow semantics found above are set by the *runtime's own
device-detection and default-sizing policy*, not by the Arc iGPU hardware itself. The stock-Ollama
`OLLAMA_IGPU_ENABLE=1` result above is one data point for this; this experiment tests it under two more runtimes
that are not stock Ollama's CPU/Vulkan-with-iGPU-disabled default path.

**Design, pre-registered before any of this runs:**
- **(a) Intel's GPU-enabled Ollama build (IPEX-LLM Ollama).** Installed side-by-side with stock Ollama, never
  replacing it: a separate install directory and a separate `OLLAMA_MODELS`/port, so the existing K1 v3/R2/PX2
  queue entries that depend on stock Ollama are never put at risk. Version and source URL recorded verbatim at
  install time. A revert procedure (uninstall steps, confirmation stock Ollama still resolves and serves
  afterward) is written into `docs/T2S_CHANGELOG.md` before the install runs, not after.
- **(b) llama.cpp's own `llama-server`, Vulkan backend, with no `-c` flag** -- i.e. whatever its own default
  context policy is with the context size unspecified, not this repo's usual explicit `-c <n>`.
- Same three models (qwen3-4b-2507, llama3.1:8b, qwen3:8b), same five probe lengths (16K/32K/48K/96K/128K) as the
  existing K1 v3 probe set, for direct comparison against the stock-Ollama-CPU and stock-Ollama-iGPU-enabled rows
  already on record.
- **Recorded per runtime:** the detected device (verbatim log line, the same way `tier_v3_device_detect` already
  records it for stock Ollama), the default context before any override, and the overflow behavior per length
  (HTTP status, sent vs processed tokens) -- the identical schema as the existing `tier_v3_probe` record, so the
  four runtimes (stock Ollama CPU, stock Ollama+IGPU_ENABLE, IPEX-LLM Ollama, llama.cpp Vulkan) are directly
  comparable rows in one table.
- **Prediction:** if the hypothesis holds, IPEX-LLM Ollama's device detection should see the iGPU and move its
  default context off the 4096 tier (the way `OLLAMA_IGPU_ENABLE=1` already did for stock Ollama above); llama.cpp
  Vulkan's own no-`-c` default is an independent third data point on whether a non-Ollama runtime's default
  context policy also depends on detecting the iGPU, or is architected differently (e.g. llama.cpp is known to
  default to the model's own trained context rather than a VRAM-scaled value, which would make this a genuinely
  different comparison, not just a replication).

**Status:** PRE-REGISTRATION. Install and run not yet started as of this writing.

## Workload pack: trace-weighted vs flat prompt-token statistics (2026-10-01)

**Why this exists.** `results/workload_pack/grade.py` reports the 400-item pack's prompt-token p50/p90/p99 as a
flat average -- every item counts once, regardless of family size or how often a real agent step of that length
actually happens. The real distribution is heavily skewed short (overall p50=356, p90=50642, p99=120010), far
below the real agent-trace step length (the uncensored trace sources' own p50 is in the thousands -- see
`x2-truncation-cliff-qwen3-8b` and `uncensored-trace-32k-crossing` in `docs/NUMBERS_REGISTER.md`). This section
re-weights the pack's own reported statistics by how often a step of each item's approximate length actually
occurs in the real uncensored trace data, so the "typical" item length reported is the one a real agent workload
would actually spend most of its steps at, not whatever length the pack generator happened to allocate the most
items to.

**Method** (implemented in `analysis/trace_weighted_pack.py`, full docstring there):
1. **Trace mass.** Step-level `tokens_qwen` from both real UNCENSORED sources in
   `results/traces/agent_step_lengths.parquet` -- `nebius/SWE-rebench-openhands-trajectories` (Qwen3-Coder-480B,
   256K native) and `SWE-Gym/OpenHands-Sampled-Trajectories` (gpt-4o/claude-3.5-sonnet, 128K/200K native),
   9,900 real steps combined. The other two dataset values in that parquet
   (`nebius/SWE-agent-trajectories`, `Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k`) are excluded:
   both are CENSORED at their generating model's own context ceiling (see `analysis/agent_traces.py`'s module
   docstring and the measured `trace-context-exit-rate` entry -- 29.85% of `nebius/SWE-agent-trajectories` runs
   hit their own generator's context limit mid-task), so including them would undercount long steps as an
   artifact of the generator, not a real absence of demand.
2. **Buckets.** 30 log-spaced histogram bin edges covering both the trace data's observed range (455-88,987
   tokens) and the pack's own observed range (50-120,017 tokens), log-spaced because the combined range spans
   roughly 3.5 orders of magnitude.
3. **Per-item weight.** Each pack item's `prompt_tokens` places it in one bin; that bin's real trace probability
   mass (real steps in the bin / 9,900 total) is split equally across every pack item landing in the same bin.
4. **Normalization.** Renormalized to sum to exactly 1.0 across the 400 items (some bins carrying real trace mass
   have zero pack items, so the raw per-item masses sum to less than 1 before this step).
5. **Statistic.** Standard weighted-percentile (sort by value, cumulative weight fraction, linear interpolation),
   the weighted generalization of `grade.py`'s own `pct()`; the flat column below is computed by importing
   `grade.py`'s `pct()` directly, so it matches `grade.py`'s own printed output exactly, not a second
   re-implementation that could silently drift from it.

**Flat vs trace-weighted prompt-token percentiles** (from `pack-trace-weighted-stats` in
`docs/NUMBERS_REGISTER.md`, computed by `analysis/trace_weighted_pack.compute_pack_stats`):

| scope | n | weight mass | flat p50 | flat p90 | flat p99 | traced p50 | traced p90 | traced p99 |
|---|---|---|---|---|---|---|---|---|
| overall (400 items) | 400 | 1.0000 | 356 | 50,642 | 120,010 | 16,000 | 32,013 | 64,012 |
| longdoc_qa | 120 | 0.6294 | 28,006 | 96,012 | 120,014 | 18,746 | 48,007 | 80,002 |
| trace_length_mix | 60 | 0.3706 | 3,684 | 14,262 | 38,498 | 9,205 | 17,875 | 46,543 |
| function_calling | 80 | 0.0000 | 347 | 363 | 367 | n/a | n/a | n/a |
| gsm8k | 80 | 0.0000 | 80 | 117 | 150 | n/a | n/a | n/a |
| r2_sessions | 60 | 0.0000 | 177 | 205 | 209 | n/a | n/a | n/a |

**Reading.** Trace-weighting moves the overall median UP by about 45x (356 -> 16,000 tokens): once items are
weighted by how often a real agent step of that length happens, the pack's effective typical length is governed
almost entirely by `longdoc_qa` (63% of the trace-matched weight mass) and `trace_length_mix` (37%), not by the
240 short items in `function_calling`, `gsm8k`, and `r2_sessions` (80+80+60=220 of the 400 items, 55% of the pack
by item count). Those three families' prompt lengths (80-367 tokens) all fall below the real uncensored trace
data's own minimum observed step length (455 tokens) -- real agent steps essentially never run that short -- so
every item in them lands in a bin with zero real trace mass and gets a weight of exactly 0 after normalization.
This is not a computation error (the function returns `NaN` for a percentile over an all-zero-weight subset
rather than silently falling back to an unweighted value); it is the direct, intended consequence of the
weighting scheme: a flat per-category average overstates how often an agent workload actually spends time at
these three families' token lengths by counting each item equally, while the trace-weighted number says the real
uncensored agent data essentially never visits that range.

**Other per-category numbers checked for a trace-weighted recomputation (step 5 of this task).** Searched
`docs/PLAN_PAPER1_DEMO.md`, `docs/FINDINGS.md`, and `analysis/numbers_register.py` for any other already-computed
"per family" / "per category" workload-pack numbers. Found none beyond `grade.py`'s own flat p50/p90/p99 output
addressed above -- `docs/PLAN_PAPER1_DEMO.md` references the workload pack only as an input to the demo build and
to the (not yet committed) baseline-policy runs, with no per-family statistic reported in prose anywhere else in
this repo as of this writing.

**Register addition:** `pack-trace-weighted-stats` in `docs/NUMBERS_REGISTER.md`, computed by
`analysis/numbers_register.py::compute_pack_trace_weighted_stats`, which calls
`analysis/trace_weighted_pack.compute_pack_stats`. Data files: `results/workload_pack/items/*.jsonl`,
`results/traces/agent_step_lengths.parquet`.
