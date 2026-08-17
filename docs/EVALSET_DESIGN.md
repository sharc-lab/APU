# Eval Set Design Constraints

This file records the binding decisions for the Paper 1 quality-axis eval set.
`evaluation/probes/setup_evalset.py` is the single source of truth for all probe
definitions, schema files, unit tests, and the `prompts.jsonl` manifest.

---

## 1. Context Independence

Every probe must be self-contained: all information needed to answer is in the
prompt itself. Filler context injected by the harness must never be required to
answer correctly. Rationale: a score drop must be attributable to context
degradation (the model failing to use what it could previously use), not to
retrieval failure (information that was never provided). Probes that depend on
filler content would conflate these two failure modes.

Enforcement: `tests/test_context_independence.py` asserts that the expected
answer string for every probe is absent from generated filler at all tested
depths and seeds. RAG probes embed their own documents directly in the prompt.

## 2. Deterministic-Primary Scoring

At least 80% of probes must be mechanically checkable (exact, schema,
span_match, unit_test scorers). Judge-scored probes are reported in a separate
column and excluded from the primary quality axis. Rationale: judge variance
adds noise that cannot be distinguished from context degradation; separating the
columns lets a reviewer discount judge scores without affecting the primary signal.

## 3. Stratified Difficulty

Probes are distributed across easy, medium, and hard difficulties so that the
run yields a capability boundary rather than just an average score. A set with
only hard probes would floor immediately; only easy probes would ceiling
everywhere.

## 4. Short, Bounded Outputs

Probe outputs are short and deterministically bounded (`max_tokens` set per
probe, typically 64–800 tokens). This ensures context depth, not output length,
is the independent variable. A probe whose output length varies with context
depth would confound the measurement.

---

## Phase 1 Sweep Configuration (blade_rtx4070, 2026-08-13)

Run: `results/run_20260813T021516Z.jsonl`

- Depths: [0, 2000, 8000, 16000, 32000] tokens of unlabelled filler
- Probes: 44 (50 defined, 6 judge-only excluded from primary axis)
- Reps: 5 per (depth, probe) cell
- Model: qwen3:4b-instruct, thinking_enabled=false
- Hardware: Razer Blade 15, RTX 4070 Laptop, 8188 MiB discrete VRAM

Findings: mean score flat at 0.48–0.52 across all depths. Two populations
dominate: 17 probes at ceiling (1.0 all depths), 14 at floor (0.0 all depths).
Depth-sensitive probes: cha_04 (1.0→0.0 at d=2000), rea_07 (degrades at d=32000),
str_03 and lon_02 (reverse pattern — see THREATS.md Threat 9).

---

## Phase 2 Sweep Configuration (binding constraints)

**Minimum baseline depth: 256 tokens** (not 0).

Rationale: d=0 is structurally different from all d>0 conditions. At d=0 the
model receives only the probe prompt; at any d>0 it receives filler + probe,
which changes the model's inference mode independently of depth (see THREATS.md
Threat 9 for evidence and mechanism). Using a nonzero minimum baseline ensures
filler is always present, removing the presence-vs-absence confound and letting
depth effects be measured as a gradient rather than a step change.

Additional Phase 2 constraints carried forward from Phase 1:
- Filler mode: unlabelled (no framing or ignore instruction)
- Probe shuffle: randomize probe order within each (depth, rep) cell using
  seed = depth * 100 + rep
- Position tracking: record position_in_cell (0-indexed) on every row as a
  scheduling-order covariate
- count_fn calibration: use /api/chat with num_predict=1 and
  num_ctx=len(filler)//4+512 to avoid allocating full native_ctx KV
  (see THREATS.md Threat 8 note on KV persistence)
- Format-aware scorer: add extraction of last numeric token before exact-match
  to separate format compliance from arithmetic correctness
  (see THREATS.md Threat 8 for motivation)
