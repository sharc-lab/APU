# Per-Probe Findings

Mechanistic observations from sweep runs that are not captured by scores alone.
These are observations about *how* the model fails, not just *that* it fails.
Each entry should name the failure mode precisely so that a score drop at any
context depth can be attributed to a mechanism rather than reported as generic
"degradation."

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
