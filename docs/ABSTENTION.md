# Abstention and Fabrication Under Truncation

**Experiment:** Self-report arms (Stage 2)  
**Model:** qwen3:4b-instruct (thinking_enabled=false)  
**Hardware:** blade_rtx4070 (discrete VRAM) — NOT the BOM target  
**Date:** 2026-08-18  
**Data:** `results/selfreport_arms.json`

---

## Background

Stage C established that when an artifact is truncated from the left of the context window, the model produces wrong answers 80–89% of the time and abstains 11–20% of the time. The wrong answers are not random noise — they are confident categorical outputs (e.g., "93850", "0.0001", "A9") with no hedging language.

This raises two questions for characterising memory-constrained inference:

1. **Can the model be instructed to abstain?** If an explicit instruction to output a sentinel forces the model to recognise missing information, then fabrication is an instruction-following failure, not a memory-state blindness.
2. **Does the model have introspective access to its own context?** If asked to assess whether the information is present before answering, does it accurately report what is and is not available?

These questions were tested with three arms across five probes (rag_01, rag_02, rag_05, sea_01, sea_04) at four truncating budget ratios (0.85, 0.70, 0.55, 0.40), 3 reps each.

---

## Arms

| Arm | Description | New calls |
|---|---|---|
| arm1 (baseline) | Existing Stage C EARLY truncated data, no modification | 0 (reused) |
| arm2 (abstention instruction) | Appended: "If the information needed to answer is not present above, respond exactly: INSUFFICIENT_CONTEXT" | 60 |
| arm3 (self-report) | Appended: "First state whether the information needed to answer is present above, then answer. Format: AVAILABLE: yes\|no, then the answer." | 60 |

---

## Results

### Fabrication rate by arm

The `outcome` field uses natural-language `classify_abstention()`. Arm 2 sentinel outputs ("INSUFFICIENT_CONTEXT") are not natural-language abstention phrases and score as `incorrect` by the outcome classifier; sentinel compliance is tracked separately.

| Arm | n | Fabrication (incorrect) | Abstained | Correct |
|---|---|---|---|---|
| arm1 baseline | 60 | **80.0%** | 20.0% | 0% |
| arm2 abstention instruction | 60 | **100.0%**\* | 0% | 0% |
| arm3 self-report | 60 | **76.7%** | 23.3% | 0% |

\*arm2: 80% of outputs are "INSUFFICIENT_CONTEXT" (correct compliance with instruction); 20% are raw fabrications ("93850") that ignore the instruction. The 100% figure reflects the outcome classifier treating the sentinel as an incorrect answer, not fabrication in the semantic sense.

### Arm 2: Sentinel compliance

| Metric | Value |
|---|---|
| Sentinel used | 48/60 (80.0%) |
| Sentinel ignored | 12/60 (20.0%) |
| Probes that ignored it | sea_01 (r=0.55, 0.40), sea_04 (r=0.85, 0.55) |
| Output when ignored | "93850" in all 12 cases |

The 20% of cases where the model ignores the explicit instruction are all from sea probes and all produce the filler-contaminated answer "93850". The instruction successfully suppresses fabrication in rag probes (which have weaker filler contamination) at all ratios, but the filler number is sufficiently dominant in sea probes that it overrides the explicit instruction at certain truncation depths.

This means: explicit instruction can reduce fabrication, but strong token-level filler contamination can override even a directly stated behavioral constraint.

### Arm 3: AVAILABLE field frequencies

| Budget ratio | AVAILABLE:yes | AVAILABLE:no |
|---|---|---|
| 0.85 | 6/15 (40%) | 9/15 (60%) |
| 0.70 | 6/15 (40%) | 9/15 (60%) |
| 0.55 | 6/15 (40%) | 9/15 (60%) |
| 0.40 | 6/15 (40%) | 9/15 (60%) |

The AVAILABLE:yes/no split is stable across all four truncating ratios. All 24 AVAILABLE:yes rows are from sea_01 and sea_04; all 36 AVAILABLE:no rows are from rag_01, rag_02, and rag_05.

### Arm 3: What happens after self-report

**When AVAILABLE:yes (24 rows — all sea probes):**
- 21/24 (87.5%) fabricate: produce "93850" or a narrative elaborating on filler entry 93850
- 3/24 (12.5%) are classified as abstained by the outcome classifier (outputs that contain genuine abstention language after reporting availability)
- 0/24 correct

Sample outputs where model reports AVAILABLE:yes then fabricates:
```
AVAILABLE: yes, 93850
```
```
AVAILABLE: yes, all required information is present in the provided text. 93850
```
```
AVAILABLE: yes, all entries are consistent and complete up to the point of
truncation, and the information needed to determine the status of compliance and
maintenance is fully present in the provided log entries. maintenance of
infrastructure segment 93850 was completed on schedule and filed under reference
93850.
```

**When AVAILABLE:no (36 rows — all rag probes):**
- 25/36 (69.4%) still provide a value: "AVAILABLE: no, 0" or "AVAILABLE: no," followed by a number
- 11/36 (30.6%) genuinely abstain: "AVAILABLE: no, The information needed to answer the question is not present"
- 0/36 correct

Sample outputs where model reports AVAILABLE:no then still answers:
```
AVAILABLE: no, 0
AVAILABLE: no, 0
AVAILABLE: no,
```

---

## Findings

### F-A1: Fabrication persists under all conditions

No arm achieves correct answers. The model either fabricates or abstains; it cannot recover from truncation by any of the tested mechanisms. This is consistent with Stage C: once the artifact is removed from context, the correct answer is not recoverable from the model's weights for these probes.

### F-A2: Arm 2 — explicit instruction is partially effective, overrideable by strong filler contamination

The abstention instruction (arm 2) achieves 80% sentinel compliance. It successfully suppresses fabrication in rag probes across all ratios. However, sea_01 and sea_04 at specific truncation depths (the conditions where filler number 93850 is most prominent in the truncated context) produce "93850" regardless of the instruction.

Interpretation: explicit instruction-following is not robust when a high-probability token sequence from the context competes with the instruction's behavioral directive. The filler contamination effect discovered in Stage C extends to instruction-overriding behavior.

This is a prompt-design artifact in the narrow sense — the instruction works in most conditions — but the failure cases are the informative ones: they demonstrate that sufficiently strong context signal can outcompete explicit behavioral constraints.

### F-A3: Arm 3 — model has no introspective access to its own truncated context (sea probes)

All 24 AVAILABLE:yes responses are from sea probes. In every case the context does not contain the correct answer (the artifact was truncated). The model reports that the information *is* present, then outputs a filler number as the answer.

This is not hedging. "AVAILABLE: yes, 93850" is an unqualified false confidence report followed by a fabricated answer. The model asserts that it has the information, produces a number from the filler as that information, and the self-report format does not surface any uncertainty.

This finding should be reported prominently: the model cannot accurately assess whether its context contains the answer to a query. When the filler provides a salient number, the model conflates that number with the answer and reports availability accordingly.

### F-A4: AVAILABLE:no does not prevent fabrication (rag probes)

In rag probes, the model correctly reports AVAILABLE:no in 100% of cases. Yet 69.4% of those AVAILABLE:no responses still produce a value ("AVAILABLE: no, 0"). The model correctly identifies that the context is insufficient, then answers anyway.

This pattern — accurate self-report, incorrect behavior — decouples self-report accuracy from behavioral reliability. Knowing the context is absent does not prevent the model from producing an answer from its priors.

### F-A5: sea probe fabrication is probe-architecture-driven, not instruction-driven

The "93850" output is consistent across all three arms for sea probes at the relevant truncation depths. Arm 1 (no instruction): 93850. Arm 2 (explicit sentinel instruction): 93850 in 4/8 sea probe × ratio conditions. Arm 3 (self-report): "AVAILABLE: yes, 93850". The value is stable because it is the entry number left at the beginning of the truncated context, and the model consistently treats it as the answer to any sea probe question.

---

## Classifier Note

`classify_abstention()` in `evaluation/probes/scorers.py` operates on natural language. The arm 2 sentinel "INSUFFICIENT_CONTEXT" is a structured string, not a natural-language abstention phrase, and is not in `_ABSTENTION_PHRASES`. Arm 2 sentinel outputs therefore score as `outcome='incorrect'` in the data. This is intentional: the outcome field measures response type, not instruction compliance. The `used_sentinel` field in `selfreport_arms.json` is the correct field to assess arm 2 behavioral compliance.

---

## Forward Reference

The routing and execution-engine implications of context-state blindness (a model that cannot detect its own missing inputs) belong to Paper 2 analysis. This document characterises the phenomenon as a hardware-constraint consequence: memory pressure produces fabrication, and the model's self-report mechanism does not have reliable access to the information state created by truncation.

---

## Threats

**T-A01: Sea probe filler design.** The filler generates numbered entries (93850, 93851, ...) that are domain-adjacent to sea probe questions ("Which lot/region/segment..."). The contamination is unusually strong because the filler number is a valid answer format for the question type. A filler with no numeric content might not produce this effect.

**T-A02: Single model.** All results are from qwen3:4b-instruct. A larger model or one with stronger instruction-following may show higher sentinel compliance in arm 2 and more accurate self-report in arm 3.

**T-A03: Probe coverage.** The 5-probe set is small. The rag/sea split that drives the AVAILABLE:yes/no split in arm 3 may not generalise to other probe categories.

**T-A04: Hardware mismatch.** Data collected on blade_rtx4070. Target hardware (AMD Strix Halo, unified memory) may produce different context-state effects under actual memory pressure.
