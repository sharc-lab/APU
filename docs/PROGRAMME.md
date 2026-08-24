# Experimental programme: the required resident set

## The unifying claim

An agentic task succeeds on a memory-constrained device if and only if its
**required resident set** — the specific spans that must be present for the task
to be answerable — fits within the device's key-value budget. The programme has
three obligations:

1. **Characterize the resident set.** What must be present, and what determines
   it. This is the mechanism work.
2. **Characterize failure when it does not fit.** What the model does instead,
   whether that is detectable, and what governs the failure mode.
3. **Predict.** Given a workload and a quality floor, compute the minimum memory
   required, and validate that prediction on hardware where memory is genuinely
   constrained.

An experiment earns a place only by serving one of these. Everything below is
mapped to one.

Already established, and where it sits:

| Result | Obligation |
|---|---|
| Recovery depends on whether the answer span survived, not on how much artifact remains | 1 |
| Partial-regime width is geometric in artifact size over prompt size; no gradient inside it | 1 |
| Type-matched residual context eliminates abstention (llama 100% → 25%) | 2 |
| Lifting requires type-match; absent in 48 dissimilar trials, 75% under type-match | 2 |
| Models differ in whether refusal exists at all; all agree type-match removes it | 2 |
| Three mitigations fail; self-assessment suppresses correct refusal | 2 |
| Format-correct fabrication from schema memory | 2 |
| Schema collision disconfirmed: with artifact present, no filler condition induces failure (135 calls, 0 lifts) | 1 — control |
| Interference at full context on one probe, one model | 1, unresolved |

**Schema collision result (1.1) as a control statement.** With the artifact present at r=1.20, no filler condition — not F-NUM, not F-TYPED, not F-SCHEMA (exact record format, same entity namespace) — induces retrieval failure on any of the three models tested. The 5/5 failure pattern on art_06/qwen3/F-TYPED is reclassified as field confusion within the artifact (model reads Motion field instead of Second; Herrera appears in the artifact as mover, Blum as seconder). Zero genuine filler lifts in 135 rows. This is the control the main interference result required: the attentional competition mechanism that drives 75% lifting under type-match requires the answer span to be *absent*, not merely outnumbered by competing schema-matched context. Consequently, 1.2 (distractor density) is deprioritised — if schema collision does not operate at all with the artifact present, a density curve has no anchor.

Obligation 3 has no results. That is the gap that keeps this a characterization
rather than a contribution.

---

## Phase 1 — What determines the resident set

### 1.1 Schema collision (135 calls)

The interference result is one probe on one model. The hypothesis is that
`art_02` differs because its type-matched filler replicates the artifact's
*record structure* — calibration records with unit IDs and ppb fields — while
the other probes' fillers supply matched values in a different container.

Build **F-SCHEMA** for `art_01`, `art_06`, `art_07`: filler replicating each
artifact's own record format with different entity identifiers. Run 3 probes ×
{F-NUM, F-TYPED, F-SCHEMA} × 3 models × 5 reps at r=1.20 only.

If F-SCHEMA induces failure where F-TYPED did not, the mechanism is schema
collision, and it applies directly to agentic context where every tool output
shares a schema. If not, `art_02` is idiosyncratic and interference is reported
as a single-probe observation.

### 1.2 Distractor density (180 calls)

Schema collision, if real, should scale with the number of competing records.
Vary the count of schema-matched distractors — 1, 4, 16, 64 — at fixed total
filler tokens, so density changes while length does not.

The output is a **collision curve**: accuracy against distractor count. That
curve is a term in the predicate, because a device holding more session history
holds more distractors.

### 1.3 Multi-turn accumulation (~400 calls)

Every result so far is single-turn. The claim that agentic context is
homogeneous, and therefore permanently in the type-matched condition, is
currently an argument rather than a measurement.

Build 12 multi-turn probes: an early turn produces a tool result, intervening
turns accumulate unrelated tool results **of the same schema**, and a final turn
requires the early result. This makes distractor accumulation a natural
consequence of the trajectory rather than injected filler.

Measure: accuracy against turn count, and whether failure arises from eviction
of the early result or from collision with later same-schema results. Those are
different causes with different provisioning consequences, and separating them
requires holding one fixed while varying the other.

### 1.4 Resident set measurement (~250 calls)

The direct measurement of obligation 1. For each probe, determine the minimum
set of spans that must be present for a correct answer, by ablating spans
individually and in combination rather than by truncating from one end.

Output per probe: required span count, token extent, and position within the
artifact. This is the quantity the predicate consumes, and no result to date
measures it directly — the answer-location thresholds are a one-dimensional
shadow of it.

---

## Phase 2 — Real memory constraint

### 2.1 Runtime eviction via llama-server (~300 calls)

All results emulate memory constraint by truncating prompts. `llama-server`
performs context shift under a fixed `--ctx-size`, evicting by policy when a
session exceeds the window. That is genuine eviction under a real allocation.

First determine what the policy actually keeps: prefix preservation, oldest-
first, sink retention. That behaviour is currently assumed. Then repeat the
central experiments under real eviction and compare.

If results match prompt truncation, every earlier result stands and the
methodological objection is answered. If they diverge, the earlier results
characterize the wrong thing and must be restated.

### 2.2 KV precision (~120 calls)

Ollama silently ignored `OLLAMA_KV_CACHE_TYPE` because it never passed the flags
through. `llama-server` accepts `--cache-type-k` and `--cache-type-v` directly,
so the precision axis may be recoverable.

Measure bytes per token at f16, q8_0, q4_0 to confirm the flags take effect,
then measure quality at fixed *byte* budgets rather than fixed token budgets.
This is the axis that connects to the published token-precision literature, and
it is the one an OEM actually controls: at fixed memory, lower precision buys
more resident tokens.

### 2.3 Unified memory (~100 calls)

The Radeon 780M shares system memory, so weights and KV compete in one pool with
the operating system. Gate carefully — ROCm support for gfx1103 on Windows is
inconsistent and a silent CPU fallback would produce plausible wrong results.

If usable, this removes the "requires hardware not in hand" limitation and gives
a discrete-versus-unified comparison on one machine.

### 2.4 Contention (~150 calls)

A shipping device runs other software. Vary background memory pressure and
measure whether the effective budget, and therefore the failure threshold,
moves. A provisioning number that assumes an idle machine is not a provisioning
number.

---

## Phase 3 — The predicate

This is the contribution. Everything above is input.

### 3.1 Construction

Given a workload — task type, artifact size distribution, turn count, schema
homogeneity — and a quality floor, compute the minimum KV budget in gigabytes.

Inputs, each measured above: required resident set size (1.4), distractor
collision curve (1.2), turn accumulation rate (1.3), bytes per token at a given
precision (2.2), eviction policy retention (2.1), and contention headroom (2.4).

### 3.2 Validation

Held-out probes not used in construction. Predict the minimum budget, run at
that budget and one increment below, and check that quality clears the floor at
the prediction and fails below it.

A predicate that is right on its training probes and wrong on held-out ones is
worthless, and this is the step that distinguishes a model from a curve fit.

### 3.3 The provisioning table

The output an OEM would use: for each memory configuration and model, the
workload envelope it sustains at a stated quality floor. This is what the BOM
frontier was always meant to produce, now grounded in measurement rather than
arithmetic.

---

## Cost and sequencing

Roughly 1,900 model calls across the programme, most of them local and cheap.
Compute is not the constraint; probe authoring and harness work are.

**Dependencies.** 1.1 gates 1.2, since density only matters if collision is
real. 1.3 and 1.4 are independent and can run in parallel. Phase 2 is
independent of Phase 1 and can proceed whenever `llama-server` is verified.
Phase 3 requires 1.2, 1.3, 1.4, 2.1, and 2.2.

**Honest scoping.** The full programme is a semester. If it must be cut, the
minimum that still yields a contribution rather than a description is: 1.1, 1.3,
1.4, 2.1, and Phase 3. That drops precision, unified memory, and contention,
which weakens the hardware side but preserves the predicate.

**What is already sufficient** for a characterization paper on its own: the
existing results plus 1.1 and 1.3. The predicate is what makes it more than
that.