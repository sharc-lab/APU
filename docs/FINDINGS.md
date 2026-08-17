# Per-Probe Findings

Mechanistic observations from sweep runs that are not captured by scores alone.
These are observations about *how* the model fails, not just *that* it fails.
Each entry should name the failure mode precisely so that a score drop at any
context depth can be attributed to a mechanism rather than reported as generic
"degradation."

---

## cha_04 — Mechanism Disconfirmed by Ablation (chained_tools, hard)

**Status: DISCONFIRMED.** The config-parameter substitution mechanism proposed
after the main sweep was ruled out by ablation on 2026-08-17. See
`results/ablation_cha04_20260817.jsonl` and `evaluation/probes/ablation.jsonl`.

**Sweep:** run_20260813T021516Z.jsonl, blade_rtx4070, depths 0–32000

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

## lon_02 — Format Compliance Couples with Correct Computation at d=16000

**Sweep:** run_20260813T021516Z.jsonl, blade_rtx4070, depths 0–32000

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
