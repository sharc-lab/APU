# Stage C: Position under Memory Pressure

**Run:** stage_c_20260818T040408Z  
**Model:** qwen3:4b-instruct (instruct variant, thinking_enabled=false)  
**Hardware:** blade_rtx4070 (discrete VRAM, 8188 MiB) — NOT the BOM target  
**Date:** 2026-08-18

---

## Design

Two arms, fixed 4,000-token filler, 6 budget ratios, 11 eligible probes × 3 reps.

| Arm | Prompt structure |
|---|---|
| LATE (control) | [filler][artifact][question] |
| EARLY (treatment) | [artifact][filler][question] |

Budget ratios: 1.20, 1.00, 0.85, 0.70, 0.55, 0.40 × full prompt length (filler + probe).  
Full prompt ≈ 4,161–4,304 tokens (varies by probe). Eligible probes: rag_01–06, sea_01–06 with artifact ≥ 120 tokens. cha_* and lon_* excluded (artifacts < 50 tokens).

**Truncation method:** Harness-side left-char truncation (see Gate 2 below).

**Prediction (stated before run):** Under LATE, truncation consumes filler and score holds as budget shrinks. Under EARLY, truncation consumes the artifact and score collapses once the budget falls below the point where the artifact survives.

---

## Gate 2: Truncation Behavior

Ollama 0.32.9 returns HTTP 400 when `num_ctx < prompt length`. Model-side context truncation is not available.

**Fallback:** Harness-side left-character truncation. At each ratio, `chars_to_keep = round(len(full_prompt) × target_tokens / full_tokens)` characters are kept from the right of the full prompt string. This produces the same logical outcome as model-side left-truncation: LATE loses filler first, EARLY loses artifact first.

---

## Results

### Summary by arm × ratio (all 11 probes, 3 reps)

| Ratio | LATE | EARLY | Δ (EARLY−LATE) | Truncating? |
|---|---|---|---|---|
| 1.20 | 0.455 | 0.545 | +0.091 | No |
| 1.00 | 0.455 | 0.545 | +0.091 | No |
| 0.85 | **0.545** | **0.182** | **−0.364** | Yes |
| 0.70 | 0.545 | 0.182 | −0.364 | Yes |
| 0.55 | 0.545 | 0.182 | −0.364 | Yes |
| 0.40 | 0.545 | 0.182 | −0.364 | Yes |

### By probe family

| Family | Arm | r=1.20 | r=1.00 | r=0.85 | r=0.40 |
|---|---|---|---|---|---|
| RAG | LATE | 0.667 | 0.667 | 0.667 | 0.667 |
| RAG | EARLY | 0.667 | 0.667 | 0.333 | 0.333 |
| SEA | LATE | 0.200 | 0.200 | 0.400 | 0.400 |
| SEA | EARLY | 0.400 | 0.400 | 0.000 | 0.000 |

---

## Per-Probe Classification

| Probe | Classification | LATE@1.20 | LATE@0.85 | EARLY@1.20 | EARLY@0.85 |
|---|---|---|---|---|---|
| rag_01 | **clean_collapse** | 1.00 | 1.00 | 1.00 | 0.00 |
| rag_02 | **clean_collapse** | 1.00 | 1.00 | 1.00 | 0.00 |
| rag_03 | floor | 0.00 | 0.00 | 0.00 | 0.00 |
| rag_04 | perverse_truncation_benefit | 0.00 | 0.00 | 0.00 | 1.00 |
| rag_05 | **clean_collapse** | 1.00 | 1.00 | 1.00 | 0.00 |
| rag_06 | ceiling_artifact_independent | 1.00 | 1.00 | 1.00 | 1.00 |
| sea_01 | **flip** | 0.00 | 1.00 | 1.00 | 0.00 |
| sea_03 | floor | 0.00 | 0.00 | 0.00 | 0.00 |
| sea_04 | **clean_collapse** | 1.00 | 1.00 | 1.00 | 0.00 |
| sea_05 | floor | 0.00 | 0.00 | 0.00 | 0.00 |
| sea_06 | floor | 0.00 | 0.00 | 0.00 | 0.00 |

### Classification definitions

- **clean_collapse:** LATE=1.0 stable at all ratios; EARLY=1.0 at no-truncation → 0.0 at first truncating ratio. Prediction confirmed.
- **flip:** LATE=0.0 at no-truncation → 1.0 at truncation; EARLY=1.0 at no-truncation → 0.0 at truncation. Both effects observed.
- **floor:** Both arms fail at all ratios. Probe uninformative for this experiment.
- **ceiling_artifact_independent:** Both arms pass at all ratios including full truncation. Artifact not required to answer correctly; probe confounded.
- **perverse_truncation_benefit:** EARLY passes at truncation but fails without artifact. Model's wrong answer at full context is replaced by a correct guess when the artifact is dropped.

---

## Findings

### F-C1: Prediction confirmed on 4/11 probes (clean_collapse)

rag_01, rag_02, rag_05, sea_04 show exactly the predicted pattern: LATE arm is stable from ratio 1.20 through 0.40; EARLY arm collapses from 1.00 to 0.00 at the first truncating ratio (0.85). The effect is sharp (not gradual) because harness-level truncation removes the entire artifact in one step at 0.85×.

### F-C2: Full flip on sea_01 demonstrates lost-in-middle effect at no-truncation

sea_01 (search_heavy, easy — "Which lot has the highest defect_rate?") shows an unexpected result at no-truncation: LATE=0.0, EARLY=1.0. The 4,000-token filler block placed before the record table (LATE arm) causes the model to fail a task it succeeds on when the records come first (EARLY arm). This is consistent with the "lost in the middle" phenomenon: artifacts buried after long preambles receive degraded attention.

When truncation begins at ratio=0.85, filler is removed from the front of the LATE arm, records become more prominent, and LATE=1.0. The EARLY arm loses its records and collapses to 0.0. Both effects are as predicted.

**Implication:** The LATE arm at full context is not a neutral baseline — it is a degraded condition for probes where the artifact follows 4,000 tokens of filler. The LATE/EARLY comparison at ratio=1.00 is not a test of truncation; it is a test of whether artifact-first or filler-first placement is better attended to at full context.

### F-C3: 5/11 probes are floor probes unresponsive to position or budget

rag_03, sea_03, sea_05, sea_06 fail in both arms at all ratios. These probes were already at floor in the main depth sweep. Their inclusion in the eligible set (artifact ≥ 120 tokens) was based on structural criteria, not performance variability. Floor probes do not contribute information about position-pressure interaction.

### F-C4: Two probes are confounded and must be excluded from conclusions

**rag_06** (ceiling_artifact_independent): The model answers correctly with or without the artifact. The question asks about sampling rate of Meridian-4 in 2021; the documents say the firmware revision "did not alter sampling rate," suggesting the model can answer from that statement regardless of artifact presence. Score=1.0 at all conditions including full truncation of EARLY arm.

**rag_04** (perverse_truncation_benefit): LATE fails at all ratios (the model applies policies incorrectly). EARLY also fails at no-truncation but passes at truncation — the model guesses the correct answer ("1") when the policy documents are absent. The artifact is actively harming performance at full context. This is a probe-design issue, not a truncation effect.

### F-C5: Score transition is step-function, not gradual

The aggregate EARLY score transitions from 0.545 (at ratio=1.00) to 0.182 (at ratio=0.85) and stays flat through ratio=0.40. There is no gradual degradation across the four truncating ratios. This is because harness-level char truncation removes the entire artifact in one step: once the dropped chars exceed the artifact length (~730–1,460 chars), all artifact content is gone. A gradual degradation would require partial artifact survival, which would require finer-grained budgets or token-level truncation.

### F-C6: LATE arm shows unexpected improvement under truncation

Aggregate LATE score increases from 0.455 (no-truncation) to 0.545 (truncation). This is attributable to sea_01 (LATE goes from 0.0 to 1.0 as filler is removed). The LATE arm is not a neutral control at full context: long filler preceding the artifact hurts some probes (lost-in-middle). Truncation of the filler paradoxically improves LATE performance for this probe.

---

## Threats to Validity

**T-C01: Harness-level truncation is not equivalent to model-level context truncation.** The experiment uses character-proportional truncation because Ollama 0.32.9 rejects requests where `num_ctx < prompt length`. The truncation is deterministic and consistent across reps, but it may not reflect how llama.cpp would handle actual context overflow (e.g., sliding-window attention or page-level eviction). The experimental manipulation is valid as a test of position sensitivity, but is not a simulation of memory-pressure inference on device.

**T-C02: Dynamic range problem reappears.** 5/11 probes are at floor in both arms. The effective sample for clean comparisons is 5 probes (4 clean_collapse + 1 flip). All five are from the easy/medium difficulty tier; hard probes in these families are at floor before the experiment begins.

**T-C03: Artifact-independence confound.** rag_06 is answerable without its artifact. Any probe whose answer is recoverable from model priors will not show a truncation effect even if the artifact is removed. Pre-screening probes for artifact-dependence is needed in future work.

**T-C04: LATE baseline is already degraded for some probes.** The LATE arm at full context is not a pure "no-pressure" baseline for probes affected by lost-in-middle. For sea_01, the control arm fails at full context. This means the LATE score is not a stable reference point for measuring EARLY degradation.

**T-C05: Step function limits sensitivity.** The 0.85× budget already fully removes the artifact from EARLY for all probes. Finer budget steps (e.g., 0.92, 0.88, 0.85) would be needed to measure at what exact truncation depth the artifact first becomes partially incomplete. The current design can only confirm that collapse occurs before 0.85×, not where the threshold is.

**T-C06: Single-rep truncation confirmation.** The same physical truncation is applied to all 3 reps of a given probe×arm×ratio cell. Rep-to-rep variation within a cell reflects inference stochasticity, not truncation stochasticity. At temperature=0 the model is deterministic, so the 3 reps are identical by design.

**T-C07: Hardware mismatch.** All data collected on blade_rtx4070 (discrete VRAM, GDDR6). Target hardware is AMD Strix Halo (unified LPDDR5x). Position-pressure effects may differ under hardware-induced memory pressure (actual KV eviction), which this experiment cannot simulate.

---

## Recommended Phase 2 Changes

1. Screen probes for artifact-independence before inclusion in position experiments.
2. Use finer budget steps around the expected collapse point (probe-specific, based on artifact_tokens / full_tokens).
3. Replace char-proportional truncation with token-level truncation via direct llama-server invocation (bypassing Ollama's num_ctx restriction).
4. Include at least one probe from each category that passes at no-truncation in both arms at baseline.
5. Add a d=256 (no-filler) control arm to measure the artifact-alone baseline without any filler effect.
