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

`evaluation/probes/multiturn.jsonl` — 12 probes, one JSON object per line.

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
retrieval question. Four probes per distance value.

| distance | len(turns) | Context budget check (large artifact, 4 char/tok) |
|---|---|---|
| 2 | 6 | ~1200 + 4×120 tok ≈ 1680 tok — below 4096, tests recall not eviction |
| 5 | 12 | ~1200 + 10×180 tok ≈ 3000 tok — approaching 4096 |
| 9 | 20 | ~1300 + 18×200 tok ≈ 4900 tok — exceeds 4096, eviction expected |

Design requirement: **large probes at distance=9 must total >4096 tokens**, so
trajectories actually test eviction rather than ordinary recall. Large probe
artifact_tokens range 1041–1318; at distance=9 the full trajectory clears 4096
tokens with margin.

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

| ID | d | schema | size | artifact type | expected |
|---|---|---|---|---|---|
| mt_01 | 2 | same | small | service config | `37419` (listen_port) |
| mt_02 | 2 | same | large | deployment manifest | `182` (keepalive_timeout_s) |
| mt_03 | 2 | different | small | calibration record | `0.00419` (alert_threshold_ppb) |
| mt_04 | 2 | different | large | ORM release notes | `5.2.1` (CVE patch version) |
| mt_05 | 5 | same | small | service config | `62183` (listen_port) |
| mt_06 | 5 | same | large | ML inference manifest | `0.273` (kv_cache_fraction) |
| mt_07 | 5 | different | small | monitoring record | `28614` (alert_count_threshold) |
| mt_08 | 5 | different | large | inventory review | `PN-47203` (discontinued SKU) |
| mt_09 | 9 | same | small | service config | `54207` (listen_port) |
| mt_10 | 9 | same | large | batch processor manifest | `2730` (max_shard_size_mb) |
| mt_11 | 9 | different | small | service config | `41739` (listen_port) |
| mt_12 | 9 | different | large | field survey report | `VX-8840-C` (instrument serial) |

All expected values are numeric strings or alphanumeric codes that are unique
within each probe and not guessable from domain priors.

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

| ID | negative_example | rationale |
|---|---|---|
| mt_01 | `8080` | default HTTP port |
| mt_02 | `75` | common keepalive value |
| mt_03 | `0.0073` | adjacent magnitude |
| mt_04 | `5.2.0` | prior version |
| mt_05 | `9090` | common Prometheus port |
| mt_06 | `0.5` | default fraction |
| mt_07 | `10000` | round number |
| mt_08 | `PN-38901` | plausible nearby SKU |
| mt_09 | `8080` | default HTTP port |
| mt_10 | `2048` | power-of-two default |
| mt_11 | `8080` | default HTTP port |
| mt_12 | `VX-8830-B` | adjacent serial |

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
