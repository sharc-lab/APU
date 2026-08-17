# Per-Probe Findings

Mechanistic observations from sweep runs that are not captured by scores alone.
These are observations about *how* the model fails, not just *that* it fails.
Each entry should name the failure mode precisely so that a score drop at any
context depth can be attributed to a mechanism rather than reported as generic
"degradation."

---

## cha_04 — Config-Parameter Substitution (chained_tools, hard)

**Sweep:** run_20260813T021516Z.jsonl, blade_rtx4070, depths 0–32000

**Score pattern:** 1.0 at d=0 (5/5 reps), 0.0 at d=2000+ (with one stochastic
pass at d=16000 r=2). The sharpest 1→0 transition in the run.

**Prompt (abbreviated):**
```
Tool results, in call order:
get_config() -> {"retries": 3, "backoff_ms": 200, "jitter": false}
get_attempt_log() -> [["fail", "fail", "success"]
With no jitter and constant backoff, how many total milliseconds were
spent waiting in backoff before the successful attempt?
```

Expected answer: 400 (2 failures × 200 ms).

**Observed outputs:**
- d=0: "400" — model correctly counts 2 failures from the log, computes 2 × 200.
- d=2000–32000: "600" (dominant), "4200" (one outlier rep at d=2000).
  - 600 = retries × backoff_ms = 3 × 200. The model uses the `retries` field
    from get_config() instead of counting actual failures from get_attempt_log().
  - 4200 is not obviously derivable from the given values and is likely an
    internal computation error compounded by the wrong substitution.

**Failure mode: config-parameter substitution.**
The model does not fail randomly. It makes a specific, internally consistent
error: it substitutes the `retries=3` configuration parameter for the
count that should be derived from reading `get_attempt_log()`. The attempt log
shows `["fail", "fail", "success"]` — two failures — but the model treats the
retry count as an authoritative count of failure events. This is a
schema-over-observation error: the model prefers the typed, labeled field
(`retries: 3`) over the event sequence that must be interpreted (`["fail",
"fail", "success"]`).

**Why this matters for score interpretation:**
A score of 0.0 at d=2000+ is not evidence that the model lost access to the
tool results. The model clearly read get_config() correctly (it uses the
backoff_ms=200 value accurately). The failure is in preferring the config
schema to the log — a reasoning shortcut that selects the most salient labeled
number rather than performing the required multi-step inference (count
occurrences in the log, then multiply). Context depth may increase the
saliency of the config block relative to the log, or it may simply lower the
model's willingness to perform the counting step when a labeled number is
available. Either way, the failure mode is substitution, not access failure.

**Implication for Phase 2 design:**
Probes that have both a labeled configuration parameter and a derived count
that differs from it will detect substitution-vs-derivation failures. These
are a distinct capability from simple artifact retrieval (can you find X in
the context?). The chained_tools category should include at least one probe
where the correct answer requires overriding a labeled number with a
computed one, and the score should be annotated with the actual output to
enable post-hoc failure mode classification.

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
