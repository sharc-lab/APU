# Paper 1 Outline — Hardware Characterization

This file maps Paper 1 sections to repository artifacts.
Paper 1 characterizes consumer-class AI PC hardware along two co-equal axes:
(a) memory pressure vs. task correctness, and (b) orchestration overhead vs. latency.
The joint feasibility envelope over both axes is the primary contribution.

Paper 2 (DSE + hybrid engine) is outlined in [docs/PAPER_OUTLINE_DSE.md](docs/PAPER_OUTLINE_DSE.md).

---

## Section 1 — Introduction

No figures. States the two-axis framing and the joint-envelope contribution.

---

## Section 2 — Related Work

Sources: [docs/RELATED_WORK.md](docs/RELATED_WORK.md), [docs/references.bib](docs/references.bib).

---

## Section 3 — Methodology

### 3A — Memory vs. Correctness

- Probe suite: [evaluation/probes/prompts.jsonl](evaluation/probes/prompts.jsonl),
  [evaluation/probes/artifact.jsonl](evaluation/probes/artifact.jsonl),
  [evaluation/probes/segments.jsonl](evaluation/probes/segments.jsonl),
  [evaluation/probes/multiturn.jsonl](evaluation/probes/multiturn.jsonl)
- Harness entry point: [harness/runner.py](harness/runner.py)
- Telemetry: [harness/telemetry.py](harness/telemetry.py)
- Context / filler generation: [harness/context.py](harness/context.py)
- Scorers: [evaluation/probes/scorers.py](evaluation/probes/scorers.py)
- Hardware guard: [harness/gpu_guard.py](harness/gpu_guard.py)
- Sources: [docs/METHODOLOGY.md](docs/METHODOLOGY.md), [docs/SCHEMA.md](docs/SCHEMA.md)

### 3B — Orchestration / Throughput vs. Latency

- SDK vs. LangGraph instrumentation:
  [harness/adapters/sdk_direct.py](harness/adapters/sdk_direct.py)
- Span categories: [harness/instrumentation/](harness/instrumentation)
- Tail-latency instrument: [harness/tail_latency_instrument.py](harness/tail_latency_instrument.py)
- Zachary Johnson's category decomposition baseline:
  [results/zachary/replication_remote_search_v3.json](results/zachary/replication_remote_search_v3.json)
- Data: `results/claude_code_characterization.json` (**gitignored**, local only),
  `results/tail_latency_results.json` (**gitignored**, local only)

---

## Section 4 — Memory vs. Correctness Results (Axis A)

### Table 4.1 — Probe suite summary

**Shows:** probe count and category breakdown across prompts.jsonl, artifact.jsonl,
segments.jsonl, multiturn.jsonl; workload_regime field values.

- Script: none (derived from probe files directly)
- Data: probe JSONL files — **IN REPO**
- Status: ready to produce

---

### Figure 4.1 — Quality vs. context depth by category

**Shows:** mean score ± stderr by filler depth (0 / 2k / 8k / 16k / 32k / 64k tokens)
for each probe category; reveals depth at which each category first degrades.

- Script: [analysis/plot_degradation.py](analysis/plot_degradation.py)
- Data: `results/run_*.jsonl` — **gitignored, local only**;
  produced by `python -m harness.runner`
- Status: data exists locally; must re-run `harness/runner.py` to regenerate
  (results not committed; gitignored by `results/run_*.jsonl` pattern)

---

### Figure 4.2 — Per-probe quality vs. depth

**Shows:** score vs. depth for every individual probe; highlights probes with
non-monotonic degradation or early cliff; sensitive probes labeled.

- Script: [analysis/plot_per_probe.py](analysis/plot_per_probe.py)
- Data: `results/run_*.jsonl` — **gitignored, local only** (same as Fig. 4.1)
- Status: same as Fig. 4.1

---

### Figure 4.3 — Artifact truncation cliff (artifact_fraction vs. score)

**Shows:** mean score vs. artifact survival fraction as left-truncation removes
progressively more of the leading artifact; cliff point identified per probe.

- Script: [harness/art_truncation.py](harness/art_truncation.py) (data collection),
  [harness/art_truncation_analysis.py](harness/art_truncation_analysis.py) (analysis)
- Data: `results/art_truncation.json`, `results/art_truncation_analysis.json` — **IN REPO**
- Status: ready to plot

---

### Figure 4.4 — Partial truncation sweep (fine-grained 1.00 → 0.85 ratio)

**Shows:** score by probe × budget ratio at 9 fine-grained truncation levels
(1.00, 0.98, 0.96, …, 0.85); artifact_fraction on second axis; extinction
ratio per probe marked.

- Script: [harness/stage2_partial_truncation.py](harness/stage2_partial_truncation.py) (data),
  no dedicated plot script yet — needs one
- Data: `results/partial_truncation.json` — **IN REPO**
- Status: data ready; plot script needed

---

### Figure 4.5 — Artifact ratio sweep (artifact proportion vs. score)

**Shows:** score as the fraction of context occupied by the artifact varies;
disentangles artifact-size effect from filler-density effect.

- Script: [harness/artifact_ratio_sweep.py](harness/artifact_ratio_sweep.py) (data)
- Data: `results/artifact_ratio_sweep.json` — **IN REPO**
- Status: data ready; plot script needed

---

### Figure 4.6 — Span ablation (filler type effect on score)

**Shows:** score × depth for different filler token compositions (numeric,
semantic, structural); establishes that score drops are artifact-truncation-driven,
not filler-type-driven.

- Script: [harness/span_ablation.py](harness/span_ablation.py) (data)
- Data: `results/span_ablation.json`, `results/span_ablation.jsonl` — **IN REPO**
- Status: data ready; plot script needed

---

### Figure 4.7 — Filler composition control (F-NUM vs. F-PROSE vs. F-STRUCT-NONNUM)

**Shows:** fabrication rate, abstention rate, and score by filler variant × arm
(baseline vs. self-report) × truncation ratio; self-report arm availability
classification.

- Script: [harness/filler_composition_sweep.py](harness/filler_composition_sweep.py) (data),
  no dedicated plot script yet
- Data: `results/filler_composition.json` — **IN REPO**
- Status: data ready; plot script needed

---

### Figure 4.8 — Type-matched filler control

**Shows:** score with numeric filler replaced by type-matched non-numeric filler;
controls for numeric-echo confound in fabrication classification.

- Script: [harness/type_match_experiment.py](harness/type_match_experiment.py) (data)
- Data: `results/type_match.json` — **IN REPO**
- Status: data ready; plot script needed

---

### Figure 4.9 — Schema collision (same-domain interference on score)

**Shows:** score on target artifact question when context contains a same-schema
competing artifact at varying overlap; quantifies semantic collision effect.

- Script: [harness/schema_collision.py](harness/schema_collision.py) (data)
- Data: `results/schema_collision.json`, `results/schema_collision.jsonl` — **IN REPO**
- Status: data ready; plot script needed

---

### Figure 4.10 — Artifact headroom (context budget consumed by artifact)

**Shows:** fraction of context window occupied by artifact × filler at each
depth level; identifies memory headroom remaining for model output at each depth.

- Script: [harness/headroom_check.py](harness/headroom_check.py) (data)
- Data: `results/art_headroom.json` — **IN REPO**
- Status: data ready; plot script needed

---

### Figure 4.11 — KV precision gate (KV quantization effect on score)

**Shows:** score degradation as KV-cache quantization bits decrease from FP16 →
INT8 → INT4; establishes minimum precision floor for target correctness.

- Script: [harness/stage_a_kv_precision.py](harness/stage_a_kv_precision.py) (data)
- Data: `results/gate1_kv_precision.json` — **IN REPO**
- Status: data ready; plot script needed

---

### Figure 4.12 — Position pressure (artifact position × depth interaction)

**Shows:** score as function of artifact position (EARLY/MID/LATE) at multiple
filler depths; identifies lost-in-the-middle degradation on target hardware.

- Script: [harness/stage_c_position_pressure.py](harness/stage_c_position_pressure.py) (data),
  no dedicated plot script
- Data: `results/stage_c_20260818T040408Z.jsonl`,
  `results/stage_c_20260818T040408Z_gate2.json`,
  `results/position_pressure_analysis.json` — **IN REPO**
- Status: data ready; plot script needed

---

### Figure 4.13 — Multi-model comparison (qwen3:4b vs. second model at depth)

**Shows:** score vs. depth for two model sizes; establishes that quality-depth
curves are model-dependent, not harness artifacts.

- Script: [harness/model2_truncation.py](harness/model2_truncation.py) (data)
- Data: `results/model2_truncation.json` — **IN REPO**
- Status: data ready; plot script needed

---

### Figure 4.14 — Scale experiment (120-model-call throughput + correctness)

**Shows:** quality scores and latency distribution over 120 sustained calls;
confirms model correctness under sustained GPU load.

- Script: [harness/stage_a_scale.py](harness/stage_a_scale.py) (data),
  [harness/interference_r120.py](harness/interference_r120.py) (interference arm)
- Data: `results/stage_a_scale.json`, `results/interference_r120.json` — **IN REPO**
- Status: data ready; plot script needed

---

### Figure 4.15 — Multi-turn recall vs. distance × schema × artifact size

**Shows:** score on final retrieval question vs. artifact_distance (2 / 5 / 9),
intervening_schema (same / different), and artifact_size (small / large);
eviction ratios (100% / 75% / 50%) as subplot dimension.

- Script: multi-turn inference harness (NOT YET WRITTEN — consumes
  `evaluation/probes/multiturn.jsonl`)
- Data: NOT YET PRODUCED — requires new harness run
- Status: **blocked on harness implementation**

---

## Section 5 — Orchestration / Throughput vs. Latency Results (Axis B)

### Table 5.1 — SDK vs. LangGraph orchestration breakdown

**Shows:** mean batch_host_cpu_ms and category percentages (ORCH, FRAMEWORK,
TOOL_COMPUTE, HTTP_CLIENT, RESIDUAL_UNATTRIBUTED) for OpenAI SDK vs. LangGraph
across 14 task types.

- Script: [harness/adapters/sdk_direct.py](harness/adapters/sdk_direct.py) (data collection)
- Data: `results/claude_code_characterization.json` —
  **gitignored by name** (`.gitignore` line 54), **local only**
- Status: data exists locally; not in repo; re-run with
  `python -m harness.adapters.sdk_direct` on a machine with OPENAI_API_KEY

---

### Figure 5.1 — Tail-latency distributions (p50 / p99 by task × concurrency)

**Shows:** p50 and p99 latency across 14 task types under single / chained /
fan-out concurrency conditions (1,260 probes); identifies tail-latency
outliers that constrain end-to-end system design.

- Script: [harness/tail_latency_instrument.py](harness/tail_latency_instrument.py)
- Data: `results/tail_latency_results.json` —
  **gitignored by name** (`.gitignore` line 55), **local only**
- Status: data exists locally; not in repo; re-run with
  `python -m harness.tail_latency_instrument`

---

### Figure 5.2 — Category decomposition from Zachary's remote-search baseline

**Shows:** per-category timing breakdown (ORCH_SETUP, HTTP_CLIENT, TOOL_COMPUTE,
FRAMEWORK, …) from an independent replication of the LangGraph remote-search task;
cross-validates the span attribution methodology.

- Script: [results/zachary/replication_remote_search_v3.json](results/zachary/replication_remote_search_v3.json)
  (data file, no separate script; raw spans in JSON)
- Data: `results/zachary/replication_remote_search_v3.json` — **IN REPO** (811 KB)
- Status: ready to plot

---

## Section 6 — Joint Feasibility Envelope (Primary Contribution)

### Figure 6.1 — Joint feasibility envelope (both axes on shared hardware)

**Shows:** the feasibility region in (context_depth, quality_score) space with
latency contours from the same probe calls; a point is feasible if it satisfies
both the quality floor (Axis A: score ≥ threshold) and the latency budget
(Axis B: http_client_ns ≤ budget). This is the envelope Paper 2 consumes.
The join is **per-call** — both axes are recorded in the same result row,
not pooled across separate experiments.

- Script: [harness/runner.py](harness/runner.py) (span instrumentation now
  active; every result row carries `orch_setup_ns`, `http_client_ns`,
  `tool_compute_ns` alongside `score` — **no separate join step required**)
- Analysis / plot: **must be written** (reads `run_*.jsonl`, plots
  (depth, score, http_client_ns) per probe, overlays feasibility boundary)
- Data: **NOT YET COLLECTED on target hardware.**
  Existing `run_*.jsonl` rows now carry span fields but were produced on
  Razer Blade 14 (discrete RTX 4070, off-target). A target-class run on
  Strix Halo EVO-X2 is required.

**Experiment specification — minimum run for this figure:**

| Parameter | Value | Rationale |
|---|---|---|
| Hardware | Strix Halo EVO-X2, 128 GB unified | target class; Blade 14 data NOT usable for envelope |
| Model | `qwen3:4b-instruct` via Ollama | same as all Axis A Blade runs |
| Probes | all 10 artifact probes (`art_01`–`art_10`) | highest quality signal at depth |
| Depths | 0, 2000, 8000, 16000, 32000, 64000 | 6 depth cells |
| Reps | 5 | matches existing Blade runs |
| **Total calls** | **10 × 6 × 5 = 300** | — |
| Estimated wall clock | 75–150 min | 15–30 s/call at depth≥32k on Strix Halo (TBD) |
| Row schema | `probe_id, depth, rep, score, score_detail, latency_ms, ttft_ms,` | both axes in same row |
| | `orch_setup_ns, http_client_ns, tool_compute_ns, hardware_config` | span breakdown |
| Hardware config flag | `--hardware-config evox2_strix_halo_128gb --memory-architecture unified` | required for cross-hw isolation |

**Why per-call pairing matters:** a mean-of-means join across depth cells would
allow the envelope plot to be driven by cells with different sample compositions
(e.g., quality sample set ≠ latency sample set). Per-call pairing eliminates
this confound — each point on the envelope plot is a single (score, latency)
pair from a single probe execution.

**Prerequisite:** `configs/hardware/evox2_strix_halo_128gb.yaml` fields
`reserved_gb`, `achievable_pool_gb`, `bandwidth_gb_s` must be populated from
telemetry before running (run `scripts/verify_platform.py` first).

---

## Section 7 — Threats to Validity

Sources: [docs/THREATS.md](docs/THREATS.md), [docs/DECISIONS.md](docs/DECISIONS.md),
[docs/SCHEMA.md](docs/SCHEMA.md).

### Threat 7.1 — Hardware scope

All Axis A experiments run on Razer Blade 14 (RTX 4070 Laptop, 8188 MiB discrete
VRAM). Target hardware is AMD Strix Halo (unified memory). Blade data is not
pooled with AI PC data. All result rows carry a `hardware` field;
unified-memory results require a separate run on target hardware.

### Threat 7.2 — Gitignored data files

Two Axis B data files are not committed:
`results/claude_code_characterization.json` and `results/tail_latency_results.json`
are gitignored by name (`.gitignore` lines 54–55). They exist locally.
Results in this paper derived from those files cannot be independently
reproduced without re-running the instrumentation harnesses.

### Threat 7.3 — Filler token distribution

Filler content is machine-generated numeric (F-NUM), prose, or structural
(F-STRUCT-NONNUM). Filler-composition sweep (Fig. 4.7) tests sensitivity;
real-world context distributions may differ.
