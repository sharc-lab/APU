# Result Provenance

This file records the hardware each committed result was produced on,
whether that hardware is on-target-class, and which figures it feeds.
It exists because the Razer Blade 14 (the development machine) is
**not the BOM target**; results from it must not be silently pooled
with future Strix Halo measurements.

**Target hardware:** AMD Ryzen AI Max+ 395 (Strix Halo), unified LPDDR5X,
Ubuntu. See `configs/hardware/evox2_strix_halo_{64,128}gb.yaml`.

**Off-target hardware used so far:** Razer Blade 14 RZ09-0508,
RTX 4070 Laptop GPU (discrete, 8188 MiB VRAM), Windows 11.
Identified in result files as `blade_rtx4070`, `blade14_rtx4070`,
or `hardware_config: blade_rtx4070`.

---

## Committed Results — Provenance Table

| File | Hardware | On-target? | Figures fed | Re-run needed? |
|---|---|---|---|---|
| `results/run_20260813T021516Z.jsonl` | blade_rtx4070, discrete | **NO** | Fig 4.1 (CORE), Fig 4.2 (SUPPORTING) — primary quality-vs-depth sweep | Yes — re-run on Strix Halo required for Fig 6.1 |
| `results/run_20260818T000746Z.jsonl` | blade_rtx4070, discrete | **NO** | Fig 4.13 (SUPPORTING) — llama3.1:8b cross-model baseline | Yes — re-run on Strix Halo for Axis A completeness |
| `results/run_20260812T*.jsonl` (11 files) | blade_rtx4070, discrete | **NO** | Exploratory/development runs; not directly cited in paper | Not required for paper |
| `results/ablation_cha04_20260817.jsonl` | blade_rtx4070, discrete | **NO** | Fig 4.2 per-probe detail (SUPPORTING) | Yes if cha_04 ablation figure is included |
| `results/stage_c_20260818T040408Z.jsonl` | blade_rtx4070, discrete | **NO** | Fig 4.12 position pressure (SUPPORTING) | Yes — re-run on Strix Halo |
| `results/span_ablation.jsonl` | blade_rtx4070, discrete | **NO** | Fig 4.6 span ablation (SUPPORTING) | Yes — re-run on Strix Halo |
| `results/art_truncation.json` | blade_rtx4070, discrete (inferred) | **NO** | Fig 4.3 (CORE) — artifact truncation cliff | Yes — re-run on Strix Halo required for envelope |
| `results/art_headroom.json` | blade_rtx4070, discrete (inferred) | **NO** | Fig 4.10 (APPENDIX) — headroom measurement | No (appendix; Blade data acceptable if labeled) |
| `results/art_truncation_analysis.json` | blade_rtx4070, discrete (inferred) | **NO** | Supports Fig 4.3 analysis | Yes, if Fig 4.3 re-run |
| `results/artifact_ratio_sweep.json` | blade_rtx4070, discrete (inferred) | **NO** | Fig 4.5 (APPENDIX) — artifact ratio sweep | No (appendix) |
| `results/crossmodel_baseline.json` | blade_rtx4070, discrete (inferred) | **NO** | Fig 4.13 (SUPPORTING) — cross-model comparison | Yes — re-run on Strix Halo |
| `results/filler_composition.json` | blade_rtx4070 (explicit field) | **NO** | Fig 4.7 (APPENDIX) — filler composition confound | No (appendix) |
| `results/model2_truncation.json` | blade_rtx4070, discrete (inferred) | **NO** | Fig 4.13 (SUPPORTING) — llama3.1:8b truncation | Yes — re-run on Strix Halo |
| `results/partial_truncation.json` | blade_rtx4070, discrete (inferred) | **NO** | Fig 4.4 (APPENDIX) — partial truncation fine sweep | No (appendix) |
| `results/position_pressure_analysis.json` | blade_rtx4070 (explicit field) | **NO** | Fig 4.12 (SUPPORTING) | Yes — re-run on Strix Halo |
| `results/selfreport_arms.json` | blade_rtx4070 (explicit field) | **NO** | Fig 4.9 (APPENDIX) — schema collision | No (appendix) |
| `results/span_ablation.json` | blade_rtx4070, discrete (inferred) | **NO** | Fig 4.6 (SUPPORTING) — span ablation | Yes — re-run on Strix Halo |
| `results/token_aligned_rerun.json` | blade_rtx4070, discrete (inferred) | **NO** | Quality sweep validation; not a primary figure | No |
| `results/type_match.json` | blade_rtx4070, discrete (inferred) | **NO** | Fig 4.8 (APPENDIX) — type-matched filler | No (appendix) |
| `results/schema_collision.json` | blade_rtx4070, discrete (inferred) | **NO** | Fig 4.9 (APPENDIX) | No (appendix) |
| `results/interference_r120.json` | blade_rtx4070, discrete (inferred) | **NO** | Fig 4.10 (APPENDIX) — headroom interference | No (appendix) |
| `results/gate1_kv_precision.json` | blade_rtx4070 (explicit in data) | **NO** | Fig 4.11 (SUPPORTING) — KV precision gate | **Yes — must re-run via llama-server directly** (Ollama path cannot set KV precision; see note below) |
| `results/llamaserver_feasibility.json` | blade14_rtx4070 (explicit) | **NO** | Reference/validation; no direct figure | No |
| `results/stage_a_scale.json` | blade_rtx4070, discrete (inferred) | **NO** | Fig 4.14 (APPENDIX) — scale experiment | No (appendix; uses cloud model gpt-oss:120b) |
| `results/fig61_full_20260922T060942Z.jsonl` | evo-t2s (evox2_evo-t2s, Intel Arrow Lake, Vulkan, unified LPDDR5X) | **NO** | **DIAGNOSTIC ONLY — DO NOT CITE** | Yes — model confound (hybrid Qwen3-4B-Q4_K_M.gguf used instead of qwen3:4b-instruct); output budget confound (max_tokens=128, all LATE RAG rows finish_reason=length); TTFT absent (non-streaming). See THREATS.md §19. Replaced by `fig61_stagec_full_20260922T191031Z.jsonl`. |
| `results/fig61_stagec_full_20260922T191031Z.jsonl` | evo-t2s (Intel Arrow Lake, Vulkan b10970-bfdc32183, unified LPDDR5X) | **NO** | Fig 6.1 position-pressure sweep — **SCORES VALID, TIMING INVALID, PC UNVERIFIED** | **Do not cite TTFT or latency from this file.** Two defects: (1) prefix KV cache was NOT disabled — rep 0 ttft≈5177ms, rep 1 ttft≈75ms on identical prompt (70x spread is cache state, not prompt variation); (2) n_prompt_tokens_actual=0 for all 396 rows (streaming returned no usage; direct /tokenize measurement not used). Scores are valid: correct checkpoint, correct scorer, correct truncation. Superseded by `fig61_stagec_full_20260922T203557Z.jsonl`. |
| `results/fig61_stagec_full_20260922T203557Z.jsonl` | evo-t2s (Intel Arrow Lake, Vulkan b10970-bfdc32183, unified LPDDR5X) | **NO** | Fig 6.1 position-pressure sweep — **VALID** | Scores match 191031Z cell-for-cell (0 differences). cache_prompt=false verified (7640ms vs 5846ms, ratio=1.3x at startup). n_prompt_tokens_actual from /tokenize on truncated prompt string. stream_options honored: tokens_in_api populated for 396/396 rows (gap=+8 tokens, chat template). PC: 387/396 pass; 9 failures at r=0.40 LATE for sea_01/sea_05/sea_06 (5.0–5.3% over threshold, char-truncation rounding, conservative — delivers slightly more context than intended). TTFT scales linearly with budget_ratio, LATE≈EARLY within 3% at each ratio. Wall clock: 37.5 min inference. |

**"Inferred"** = no explicit `hardware` field in JSON; inferred from commit date,
model name (`qwen3:4b-instruct` + Ollama), and the Blade 14 being the only
machine with Ollama access during these commits.

---

## gate1_kv_precision.json — Provenance note

**Produced on:** Razer Blade 14 RZ09-0508, Windows 11, Ollama 0.32.9/0.32.6,
RTX 4070 Laptop GPU (NVIDIA discrete VRAM, 8188 MiB).

**Key result:** None. All four conditions (f16, q8_0, q4_0, f16_no_flash) ran
at f16 — Ollama 0.32.x reads `OLLAMA_KV_CACHE_TYPE` at startup but does not
propagate it to the llama-server command line as `--cache-type-k`. The flag
is silently ignored; every condition produced identical KV log lines
(`K (f16): 2304 MiB, V (f16): 2304 MiB`). The file records f16 KV
allocation on Ollama 0.32.9 (118,784 B/tok at ctx=32768, CUDA0 buffer) and
documents the API limitation. It is **not** a source of quantization
reduction ratios. The figures previously cited here (≈1.83× and ≈3.76×)
were incorrect and have been removed.

**Reference for KV quantization ratios:** `results/llamaserver_feasibility.json`
(llama-server b1-f8def7fe1, flags confirmed effective). Measured reductions:
f16→q8_0 = 1.77×, f16→q4_0 = 3.24×. See `docs/KV_MEASUREMENT.md` for the
full reconciliation.

**Off-target-class status:** The Blade 14 uses discrete NVIDIA VRAM.
The KV cache precision test measures GPU-side memory allocation via
`nvidia-smi`. On Strix Halo (AMD unified memory, no discrete VRAM):
- `nvidia-smi` is unavailable; the script now falls back to `rocm-smi`
  and then to `/proc/meminfo` (unified pool, less precise for KV-only)
- VRAM allocation semantics differ: on unified memory, weights + KV + OS
  all share one pool; the subtraction method (`total_vram - weight_vram`)
  may include OS/driver allocations not present on discrete hardware

**Status:** TARGET-CLASS RE-RUN PENDING.
The committed result is valid for the Blade 14 configuration and documents
the expected KV ratio for NVIDIA discrete hardware. Before citing this
figure in the paper for the BOM device, the experiment must be re-run on
Strix Halo EVO-X2 with a unified-memory-compatible measurement method.

**Reproducibility:** `harness/stage_a_kv_precision.py` is now fully portable
(OLLAMA_BIN / PATH resolution; platform-aware kill, log path, GPU query).
Run on EVO-X2 after verifying `scripts/verify_platform.py` passes.

---

## Rep determinism note (applies to all fig61_stagec_* files and stage_c_20260818T040408Z.jsonl)

At temperature=0 with an identical prompt, reps are near-deterministic. Per-cell variance
across reps (N=3) is not a meaningful error estimate — the three values are produced by the
same deterministic path. This matches stage C behavior. Reps exist to detect non-determinism
(e.g., sampling glitches) and to confirm stability, not to provide a variance estimate.

---

## fig61_full_20260922T060942Z.jsonl — Diagnostic run note

**Produced on:** evo-t2s (Intel Arrow Lake, Vulkan build b10970, unified LPDDR5X).
**Date:** 2026-09-22.

**Status: DIAGNOSTIC ONLY. Do not cite in paper. Do not pool with stage C.**

Two confounds make this run non-comparable to stage C (`stage_c_20260818T040408Z.jsonl`):

1. **Model confound (THREATS.md §19a):** The run used `Qwen3-4B-Q4_K_M.gguf`
   (hybrid thinking model, `--reasoning-budget 0 --reasoning-format deepseek`).
   Stage C used `qwen3:4b-instruct` (Ollama, sha256:85e4a5b7..., instruct-tuned,
   no thinking capability). These are different checkpoints with different output
   behavior (EARLY outputs begin `</think>\n\n`; LATE RAG outputs are verbose).

2. **Output budget confound (THREATS.md §19b):** `max_tokens=128` was used
   (correct, matching stage C), but the hybrid model produces longer reasoning
   outputs, causing all LATE RAG rows to hit `finish_reason: length` before
   the answer is reached. Score=0.0 at non-truncating ratios for these rows
   is an output budget artifact, not a position effect.

**Affected rows:** All 54 rag_01/rag_02/rag_05 LATE rows. All rag_02 LATE rows
show non-monotonic scores (score=0.0 at r=1.20, score=1.0 at r=0.55) —
this is output budget artifact, not a real position signal.

**Valid rows:** EARLY RAG rows and all SEA rows are usable for internal
comparison purposes, but cannot be compared to stage C (different checkpoint).

**Required fix:** Re-run with `qwen3-4b-instruct-85e4a5b7.gguf` (no reasoning
flags), streaming (for TTFT), matched stage C generation params.

---

## Block 1 — Required re-runs on Strix Halo (EVO-X2)

These committed results feed CORE or SUPPORTING figures and must be
reproduced on target-class hardware before submission.

| Priority | File | Figure | Why needed on target |
|---|---|---|---|
| 1 | Any `run_*.jsonl` (10-probe artifact sweep) | Fig 6.1 (CORE) | Joint envelope requires Strix Halo data for both axes |
| 2 | `art_truncation.json` | Fig 4.3 (CORE) | Truncation cliff may shift with different memory bandwidth |
| 3 | `run_20260813T021516Z.jsonl` (main quality sweep) | Fig 4.1 (CORE), Fig 4.2 | Primary quality-vs-depth curve |
| 4 | `gate1_kv_precision.json` | Fig 4.11 (SUPPORTING) | Discrete NVIDIA measurement; AMD unified path unvalidated |
| 5 | `span_ablation.json` + `span_ablation.jsonl` | Fig 4.6 (SUPPORTING) | Span ablation on unified memory may show different proportions |
| 6 | `position_pressure_analysis.json` + `stage_c_*.jsonl` | Fig 4.12 (SUPPORTING) | Position pressure at depth may differ on unified memory bandwidth |
| 7 | `crossmodel_baseline.json`, `model2_truncation.json` | Fig 4.13 (SUPPORTING) | Cross-model comparison needs same-hardware baseline |

Results in **APPENDIX** figures produced on Blade 14 are acceptable labeled
as "Blade 14 / discrete RTX 4070" with a note that Strix Halo re-runs
are planned. They do not block submission if the CORE figures are reproduced.

---

## Path Redaction Record

**Date:** 2026-09-14  
**Author:** Rithwik Sharma  
**Nature of change:** Username redaction only. No numeric value, measurement, or
analysis-relevant field was modified. All five edits replace a literal Windows
username path prefix with the platform-token equivalent so the repo does not
leak the developer's username.

| File | JSON key path | Before (prefix) | After |
|---|---|---|---|
| `results/gate1_kv_precision.json` | `["llama_server_direct"]["path"]` | `C:\Users\rithw\AppData\Local\...` | `%LOCALAPPDATA%\...` |
| `results/llamaserver_feasibility.json` | `["binary"]["path"]` | `C:\Users\rithw\AppData\Local\...` | `%LOCALAPPDATA%\...` |
| `results/llamaserver_feasibility.json` | `["gguf_location"]["manifest_path"]` | `C:\Users\rithw\.ollama\...` | `%USERPROFILE%\.ollama\...` |
| `results/llamaserver_feasibility.json` | `["gguf_location"]["blob_path"]` | `C:\Users\rithw\.ollama\models\blobs\sha256-...` | `%USERPROFILE%\.ollama\models\blobs\sha256-...` (digest preserved) |
| `results/llamaserver_feasibility.json` | `["launch_recipe"]["example"]` | `cd C:/Users/rithw/AppData/Local/...` | `cd %LOCALAPPDATA%/...` |

None of these fields is load-bearing for any analysis script or figure. They are
provenance metadata (binary location, model blob path, launch example). The
SHA-256 model digest in `blob_path` is preserved verbatim.

Tests that verify no username appears in committed JSON result files are in
`tests/test_no_absolute_paths.py` (the `test_results_no_username` test).
