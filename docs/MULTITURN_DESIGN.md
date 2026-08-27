# Multi-Turn Probe Design

## Purpose

Single-turn artifact probes (Stage 1 / `artifact.jsonl`) confirm that a model
can retrieve a specific value from an artifact presented in the same prompt.
They do not test whether the model retains that value across a conversation,
nor whether competing intervening content displaces it.

Multi-turn probes close both gaps. They serve two goals:

1. **Close the single-turn limitation.** In real use the artifact is introduced
   early in a session; retrieval is requested much later. Single-turn probes
   cannot measure this.

2. **Provide an accumulation mechanism for KV eviction.** Stage 2 probes use
   filler content and truncation ratios to stress KV-cache eviction. Multi-turn
   probes create the same pressure organically: each intervening exchange pair
   adds tokens to the KV cache, and at sufficient distance the total trajectory
   exceeds the model's practical context window, forcing eviction or degraded
   recall.

## File

`evaluation/probes/multiturn.jsonl` — 36 probes, one JSON object per line.
36 = 12 cells × 3 probes per cell.  3 probes per cell are required so that
per-cell results are separable from probe identity.

## Schema

| Field | Type | Description |
|---|---|---|
| `id` | str | `mt_NN` |
| `category` | str | `"multi_turn"` |
| `workload_regime` | str | `"multi_turn"` |
| `artifact_distance` | int | Exchange pairs between artifact response and final question |
| `artifact_turn` | int | 0-indexed position in `turns[]` where artifact is introduced |
| `final_turn_index` | int | `len(turns)` — where the question appended at inference time |
| `intervening_schema` | str | `"same"` or `"different"` |
| `artifact_size` | str | `"small"` or `"large"` |
| `artifact_tokens` | int | Approximate token count of artifact text (~4 chars/token) |
| `question` | str | Final retrieval question (appended after `turns[]`) |
| `expected` | str | Unique correct answer |
| `negative_example` | str | Plausible wrong answer (scorer must reject) |
| `scorer_type` | str | `"exact"` for all 12 probes |
| `max_tokens` | int | 32 |
| `turns` | list | Prior conversation history |

`turns[]` structure for a probe with `artifact_distance = d`:

```
turns[0]         — user:      introduces artifact + discussion question
turns[1]         — assistant: reviews artifact, references key values
turns[2..2d+1]   — d exchange pairs of intervening conversation
```

`len(turns) == 2 + 2 * artifact_distance == final_turn_index`

The final `question` is not in `turns[]`. The inference harness appends it as
a new user message.

## Three Design Dimensions

### Distance (artifact_distance ∈ {2, 5, 9})

Number of full exchange pairs between the artifact response and the final
retrieval question. 12 probes per distance value (3 per cell × 4 cells per
distance).

| distance | len(turns) |
|---|---|
| 2 | 6 |
| 5 | 12 |
| 9 | 20 |

### Eviction pressure — option (b): vary ctx-size, not trajectory length

**Only large probes at distance=9 naturally exceed 4096 tokens.** In all other
cells the trajectory fits a standard window, so no eviction fires and those
probes measure long-context recall rather than memory pressure.

Two remedies were considered:

**(a) Scale trajectory length** — add filler content so every cell exceeds
the target ctx-size. Rejected: large-trajectory distance=2 probes would
require thousands of tokens of padding, distorting the intervening-turn
character and making the schema × size × distance decomposition uninterpretable.

**(b) Vary ctx-size, keep trajectory length fixed** ← chosen. The harness sets
`num_ctx` to a target fraction of the full prompt token count per cell. All
cells are subjected to the same eviction ratios (e.g., 100%, 75%, 50%); the
model sees the same conversation at each ratio, but the effective context
window shrinks. This cleanly separates eviction pressure (the independent
variable) from trajectory length (the probe content), and makes eviction ratio
a first-class experimental parameter without regenerating probe content.

Each probe records `artifact_tokens`; the harness measures `full_tokens` at
setup time and derives `num_ctx = round(full_tokens * eviction_ratio)` before
each call.

### Intervening Schema (intervening_schema ∈ {"same", "different"})

Controls whether the intervening exchange pairs discuss artifacts of the same
domain (potential semantic interference) or unrelated topics (neutral filler).

- **same**: each intervening pair presents a different service config, manifest,
  calibration record, or inventory entry of the same structural type as the
  probe artifact. Tests whether semantically similar content displaces the
  target value.
- **different**: each intervening pair discusses code review, system design, or
  project planning topics — unrelated to the artifact domain. Tests pure
  temporal displacement.

### Artifact Size (artifact_size ∈ {"small", "large"})

- **small**: ~140–200 artifact tokens. Fits comfortably in any context window.
  Tests recall quality under intervening semantic pressure.
- **large**: ~1040–1320 artifact tokens. Consumes a large fraction of the
  context budget at distance=9, directly triggering eviction under realistic
  KV-cache constraints.

## Probe Inventory

36 probes, 12 cells × 3 probes per cell. Probes within a cell share the same
(d, schema, size) but use distinct artifacts with distinct expected values.

| IDs | d | schema | size | artifact type | expected values |
|---|---|---|---|---|---|
| mt_01–03 | 2 | same | small | service config JSON | `37419`, `62183`, `54207` (listen_port) |
| mt_04–06 | 2 | same | large | deployment manifests | `182`, `0.273`, `2730` |
| mt_07–09 | 2 | different | small | calibration/monitoring records | `0.00419`, `28614`, `0.00731` |
| mt_10–12 | 2 | different | large | ORM release notes, inventory, field survey | `5.2.1`, `PN-47203`, `VX-8840-C` |
| mt_13–15 | 5 | same | small | service config JSON | `37419`, `62183`, `54207` |
| mt_16–18 | 5 | same | large | deployment manifests | `182`, `0.273`, `2730` |
| mt_19–21 | 5 | different | small | calibration/monitoring records | `0.00419`, `28614`, `0.00731` |
| mt_22–24 | 5 | different | large | ORM release notes, inventory, field survey | `5.2.1`, `PN-47203`, `VX-8840-C` |
| mt_25–27 | 9 | same | small | service config JSON | `37419`, `62183`, `54207` |
| mt_28–30 | 9 | same | large | deployment manifests | `182`, `0.273`, `2730` |
| mt_31–33 | 9 | different | small | calibration/monitoring records | `0.00419`, `28614`, `0.00731` |
| mt_34–36 | 9 | different | large | ORM release notes, inventory, field survey | `5.2.1`, `PN-47203`, `VX-8840-C` |

All expected values are numeric strings or alphanumeric codes that are unique
within each probe. All 36 probes pass the artifact-deletion check: score drops
below 1.0 when turns[0] and turns[1] are removed (verified against
qwen3:4b-instruct). Results in `results/artifact_deletion_check.json`.

## Validation Checks

`evaluation/probes/validate.py` runs three classes of check against
`multiturn.jsonl`:

### POSITIVE

```python
score_exact(p["expected"], p["expected"]) == 1.0
```

The scorer must accept the correct answer as correct. This catches scorer
misconfiguration.

### NEGATIVE

Two sub-checks:

**Scorer rejection**: `score_exact(p["negative_example"], p["expected"]) < 1.0`

Each probe carries a `negative_example`: a plausible wrong answer
(e.g., a common default port `"8080"` for a probe whose expected port is
`"37419"`). The scorer must reject it. This check is load-bearing: a scorer
that always returns 1.0 is strictly worse than no scorer, because it masks
model failure. The NEGATIVE check catches that regression.

**Leak check**: the expected answer must not appear verbatim in any intervening
turn (turn index `> artifact_turn + 1`). Turn 0 (artifact) and turn 1
(immediate artifact response) are exempt — it is expected that the reviewing
assistant quotes key values from the artifact it just read. Turns 2 onwards are
the intervening conversation; the expected value must not leak there, otherwise
the question is answerable without reading the original artifact.

Negative example choices per probe:

| expected | negative_example | rationale |
|---|---|---|
| `37419` | `8080` | default HTTP port |
| `62183` | `9090` | common Prometheus port |
| `54207` | `8080` | default HTTP port |
| `182` | `75` | common keepalive value |
| `0.273` | `0.5` | default fraction |
| `2730` | `2048` | power-of-two default |
| `0.00419` | `0.0073` | adjacent magnitude |
| `28614` | `10000` | round number |
| `0.00731` | `0.0050` | adjacent magnitude |
| `5.2.1` | `5.2.0` | prior version (also mentioned in same artifact) |
| `PN-47203` | `PN-47218` | co-discontinued SKU in same report |
| `VX-8840-C` | `VX-8830-B` | adjacent serial |

### STRUCTURAL

Schema integrity checks for each probe:

- `workload_regime == "multi_turn"` and `category == "multi_turn"`
- `final_turn_index == len(turns)`
- `artifact_turn < final_turn_index`
- `turns[]` strictly alternates user/assistant starting with user
- `turns[artifact_turn]["role"] == "user"`
- `artifact_tokens > 0`
- `artifact_size == "small"` ↔ `artifact_tokens <= 500`
- `artifact_size == "large"` ↔ `artifact_tokens > 500`
- `artifact_distance == (final_turn_index - 2) // 2`
- `0 < max_tokens <= 800`
- `intervening_schema ∈ {"same", "different"}`
- `negative_example` present and `!= expected`

## Relationship to Existing Probes

The `workload_regime` field distinguishes probe classes:

- Single-turn probes (`prompts.jsonl`, `artifact.jsonl`, `segments.jsonl`):
  `workload_regime = "single_turn"` (documented in harness JSONL rows)
- Multi-turn probes (`multiturn.jsonl`): `workload_regime = "multi_turn"`

Existing harnesses already emit `workload_regime: "single_turn"` in their
result rows, added as part of the Stage 3 backfill. The field enables unified
analysis across workload types without re-running old experiments.
