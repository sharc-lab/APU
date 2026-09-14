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
