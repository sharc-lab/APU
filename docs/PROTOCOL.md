# Measurement Protocol

This is a shared normative document. Both researchers MUST follow it so that
result rows can be joined across platforms. Where a requirement uses MUST, that
means non-negotiable for a row to be pooled. Where something is needed but no
field yet exists in the schema, it is listed under §10 (Proposed additions).

---

## §1 Requirements

**R1 — Positive-control obligation (every configuration flag must prove it took effect)**

Before recording any row against a configuration knob, you MUST obtain a log
line or verifiable artifact proving the flag was applied by the runtime, not
just passed on the command line.

Acceptable evidence:
- An Ollama or llama-server log line naming the value (e.g.,
  `llama_kv_cache: K (q8_0): ...`, `--cache-type-k q8_0`)
- A tool output whose value changes predictably when the flag is toggled (e.g.,
  `CUDA0 KV buffer size` shrinks by ~2× when going from f16 to q8_0)
- A side-by-side diff of two consecutive server invocations (flag on / flag off)
  with the memory readout changing as expected

**Worked example of why this requirement exists** — `docs/KV_MEASUREMENT.md`:
`OLLAMA_KV_CACHE_TYPE=q8_0` was passed correctly on the command line for the
entire gate1 sweep. Ollama 0.32.x accepts the variable but does not propagate it
to the `llama-server` call as `--cache-type-k`. Every condition ran at f16 with
no error or warning. The failure was only detectable by inspecting the Ollama
server log: `K (f16): 2304.00 MiB, V (f16): 2304.00 MiB` appeared identically
across all four intended precisions. The gate1 file is therefore NOT a source of
KV quantization reduction ratios (`docs/RESULT_PROVENANCE.md`). A 30-second log
check before the sweep would have caught it.

**R2 — Hardware segregation**

Blade14 (blade14_rtx4070, discrete NVIDIA VRAM) and Strix Halo (unified LPDDR5X)
rows MUST NEVER be silently pooled. Every result file MUST carry `hardware_config`
(the key from `configs/hardware/`) and `memory_architecture` on each row. These
two fields are the join key; any cross-hardware comparison that does not filter or
stratify on both is invalid. See `docs/RESULT_PROVENANCE.md` for the Blade14
off-target-class status.

**R3 — Memory source segregation**

Rows with `gpu_mem_source == "unified-psutil"` measure whole-system RAM, not GPU
allocation. They MUST NOT be pooled with rows whose `gpu_mem_source` is
`"nvidia-smi"`, `"rocm-smi"`, or `"intel-level-zero"`. Filter by source before
any memory-pressure analysis. See THREATS.md §13.

---

## §2 Run Identity

The following fields MUST appear on every result row. They are row-level fields
in the schema (as of 2026-09-14); all five are emitted by the runner and
are filled with None by `normalize_result_row()` on old rows.

| Field | Type | What it records |
|---|---|---|
| `git_sha` | str \| null | Full 40-char SHA of the repo at sweep start; null if git unavailable |
| `run_seed` | int \| null | Top-level RNG seed (`--seed N`); null if not specified |
| `hostname` | str | Machine hostname at run time (`socket.gethostname()`) |
| `operator` | str \| null | Researcher identifier from `APU_OPERATOR` env var; null if unset |
| `done_reason` | str \| null | Ollama stop reason for this call: `"stop"` (normal), `"length"` (hit generation budget); null on cache hits from before this field was added, and on error rows |

`timestamp` is not a row-level field; it is encoded in the JSONL filename
(`run_<timestamp>.jsonl`). The `utc_start` field in the sibling
`_MANIFEST.json` file is the authoritative per-run timestamp.

The existing `config_hash` field is a SHA-256 prefix of the sweep
configuration parameters, not the git SHA. It identifies sweep-parameter
identity but not code state. Both `config_hash` and `git_sha` are required.

When joining rows from two researchers, all four identity fields MUST match
for rows to be considered from the same configuration. A `git_sha` mismatch
means the code differed; a `hostname` difference is expected (that is the
point); `run_seed` and `operator` distinguish deliberate replications from
the same lab.

---

## §3 Hardware

Every result row MUST carry:

| Field (row-level) | Values | Source |
|---|---|---|
| `hardware_config` | Key from `configs/hardware/` — e.g. `blade14_rtx4070`, `evox2_strix_halo_64gb` | Runner `--hardware-config` argument |
| `memory_architecture` | `"discrete"` or `"unified"` | Copied from the hardware YAML's `memory_architecture` field |

Before running a sweep on new hardware, a hardware YAML MUST exist in
`configs/hardware/` with the following fields populated:
`name`, `memory_architecture`, `os`, `cpu`, `gpu`, `memory_gb`, `reserved_gb`,
`achievable_pool_gb`, `quant`.

`achievable_pool_gb` MUST be measured, not calculated. For discrete hardware,
this is `gpu_vram_mib / 1024 − reserved_gb`. For unified hardware, measure idle
system RAM consumption and subtract from total pool; the OS share is variable
and must not be assumed from specs alone.

---

## §4 Runtime

The following MUST be recorded before the sweep begins (in the run-level meta
file described in §2 until row-level fields exist):

| Item | What to record |
|---|---|
| Runtime name | `"ollama"` or `"llama-server"` or `"openvino"` |
| Build identifier | The full version string — e.g. `"Ollama 0.32.9"`, `"b1-f8def7fe1"` (llama-server git hash), `"OpenVINO 2024.1"` |
| Log evidence | The log line(s) proving each non-default configuration flag took effect (R1) |

For Ollama: copy the relevant `llama-server` invocation line from the server log
(shows actual flags passed) and at minimum the `llama_kv_cache:` line showing the
K and V types that were applied.

For llama-server direct: copy the startup banner lines that confirm `--cache-type-k`,
`--cache-type-v`, `--n-gpu-layers`, and `--ctx-size`.

For OpenVINO: record the `ov::device::full_name` and `ov::intel_gpu::execution_units_count`
properties from `ov::Core().get_property()`, and the precision reported in the
model's XML manifest.

---

## §5 Configuration Vector

For every sweep, record the configuration vector that identifies where in the
design space the run sits. Fields marked "row-level (existing)" are already on
each result row. The rest MUST go in the run-level meta file until §10 additions
are implemented.

### Placement — where model layers execute

| Concept | llama.cpp / Ollama | OpenVINO |
|---|---|---|
| GPU layer count | `--n-gpu-layers N` (llama-server) / `OLLAMA_GPU_LAYERS` (Ollama) | `"GPU"` as device string to `ov::Core().compile_model()` |
| CPU-only offload | `--n-gpu-layers 0` | `"CPU"` as device string |
| Mixed (partial offload) | `--n-gpu-layers K` where K < total layers | Not natively supported per-layer; use `MULTI:GPU,CPU` for whole-model split |

Record the actual value used, not the intended value. Prove via log (R1):
`llm_load_tensors:` lines show how many layers landed on GPU vs CPU.

### Residency — what pool weights and KV live in at runtime

| Concept | llama.cpp / Ollama | OpenVINO |
|---|---|---|
| Full VRAM (discrete) | `--n-gpu-layers 99` with sufficient VRAM | Compile to `GPU` with model fits in device memory |
| Partial offload to RAM | `--n-gpu-layers K` < total | Not directly applicable |
| Paged / swapped | Onset observable via latency spike and `gpu_mem_mb` plateau | System swap visible in `mem_rss_mb` growth |

### KV precision

| Concept | llama.cpp (confirmed working) | Ollama (broken ≤0.32.x) | OpenVINO |
|---|---|---|---|
| Full precision KV | `--cache-type-k f16 --cache-type-v f16` | `OLLAMA_KV_CACHE_TYPE=f16` (silently ignored ≤0.32.x) | Default; no separate KV precision control |
| Int8 KV | `--cache-type-k q8_0 --cache-type-v q8_0` | Same variable, same failure mode | N/A |
| Int4 KV | `--cache-type-k q4_0 --cache-type-v q4_0` | Same | N/A |

Log evidence required (R1): `llama_kv_cache: K (q8_0):` must appear in the
server log. A line reading `K (f16):` when q8_0 was intended means the flag
failed — do not record rows.

### Weight precision

| Concept | llama.cpp / Ollama | OpenVINO |
|---|---|---|
| Quantization | GGUF quant type embedded in model file: q4_k_m, q8_0, f16, etc. | Model XML precision field; INT8/INT4 via NNCF export |

Record the quant type as `weight_precision`. For GGUF models, `ollama show <model>`
prints the quant type. For OpenVINO, read `<rt_info><Config>` in the XML.

### Model tier

Record the model family and parameter count, e.g. `qwen3_4b`, `llama3_8b`.
This is separate from `model_variant` (instruct vs. reasoning), which is already
a row-level field.

### Context budget

`max_tokens` is a row-level field (the generation budget per call). The
**sweep depth** is the filler token target, recorded as `depth` (row-level).
Both are required. `ctx_suspect` (row-level) flags rows where `tokens_in < depth × 0.9`.

---

## §6 Regime

For each run, classify the operating regime and record it with the evidence that
determined it. This is not yet a row-level field; record it in the run-level meta
or per-depth summary until §10 is implemented.

| Regime | Meaning | Detection |
|---|---|---|
| `NOT_REACHED` | Model completed generation normally | `tokens_out >= 1`, no latency spike, `ctx_suspect == false` |
| `TRUNCATION` | Output was cut by the `max_tokens` generation budget | `tokens_out == max_tokens` consistently across reps |
| `PAGING` | Context is loaded but KV eviction is occurring | Latency spike at a specific depth without OOM; `gpu_mem_mb` plateau at an earlier depth; throughput drop |
| `OOM` | Model failed to load or inference errored due to memory exhaustion | Error row present (`error` field), or Ollama server log shows OOM/CUDA error |

A run that spans multiple depths may exhibit different regimes at different
depths. The regime MUST be recorded per depth stratum, not per run as a whole.

`done_reason` is a row-level field (as of 2026-09-14). A value of `"length"`
is the primary signal for TRUNCATION regime; `"stop"` means the model finished
normally. Old rows before this date will have `done_reason == null` after
`normalize_result_row()` is applied — regime for those rows must be inferred
from `tokens_out == max_tokens` as a fallback.

---

## §7 Outcome

Every result row MUST carry all four outcome fields. `score` has been present
since the first sweep. The three classifier fields were added in the
outcome-classifier wiring (2026-09-14) and are produced by `evaluation/outcome.py`
for every probe call.

| Field | Type | Values |
|---|---|---|
| `score` | float \| null | 0.0–1.0 (null on error rows only) |
| `outcome_class` | str \| null | `CORRECT`, `REFUSED`, `FABRICATED`, `UNCLASSIFIABLE` (null on pre-classifier rows and error rows) |
| `classification_method` | str \| null | `score`, `last_token`, `refused_sentinel`, `refused_abstention`, `unclassifiable` |
| `format_compliant` | bool \| null | True = scorer accepted as-is; False = last-token match (model showed working); None = not applicable |

Old rows that predate the classifier wiring will have `outcome_class == null`.
When loading JSONL rows for analysis, call `evaluation.outcome.normalize_result_row(d)`
to fill missing fields with null rather than raising a KeyError.

`UNCLASSIFIABLE` rows (empty output, `done_reason == "length"` or
`"budget_exhausted"`) MUST be excluded from fabrication rate denominators. The
corrected rate for `results/stage_a_scale.json` is documented in
`docs/FINDINGS.md`. The per-arm rates in `results/selfreport_arms.json` cannot
receive this correction because that file stores only aggregate statistics —
see THREATS.md §14.

---

## §8 Telemetry

Every result row MUST carry both of the following memory fields together:

| Field | Type | Semantics |
|---|---|---|
| `gpu_mem_mb` | float \| null | Memory used (MiB). Null when no backend succeeded; never 0.0 as a proxy for unavailability in rows written ≥ 2026-09-07. Old rows used 0.0 ambiguously. |
| `gpu_mem_source` | str \| null | How the value was measured; null in rows written before 2026-09-07 (old rows used `gpu_mem_method`). |

Valid `gpu_mem_source` literals and what they actually measure:

| Literal | Measures | Comparable to |
|---|---|---|
| `nvidia-smi` | Discrete NVIDIA VRAM currently in use by all processes | Other `nvidia-smi` rows on the same hardware config only |
| `rocm-smi` | AMD GPU driver's allocation within the shared pool | Other `rocm-smi` rows on the same hardware config only |
| `intel-level-zero` | GPU-allocated shared DRAM via Sysman API (sum of all modules) | Other `intel-level-zero` rows on the same hardware config only |
| `intel-sysfs` | Intel i915 GEM object total (debugfs; rarely available) | Treat as equivalent to `intel-level-zero` within same config |
| `unified-psutil` | **Whole-system RAM in use** (OS + all processes + GPU combined) | **Only other `unified-psutil` rows on the same hardware config. NOT GPU allocation.** |
| `unavailable:*` | Nothing — measurement failed | Do not use in memory-pressure analysis |

The `unified-psutil` literal is a whole-system proxy, not a GPU allocation figure.
On a Strix Halo machine at idle it was ≈7.0 GB of 63.5 GB total — dominated by
OS and background processes, not GPU activity. See THREATS.md §13.

The `mem_rss_mb` field (process RSS) is a separate measurement from `gpu_mem_mb`.
It reflects harness process memory, not model VRAM.

---

## §9 Cost

For sweeps that call remote APIs (e.g., `stage_a_scale.json` used `gpt-oss:120b`
via OpenAI), record:

| Item | What to record |
|---|---|
| `tokens_in` | Prompt tokens as reported by the API (row-level field, already present) |
| `tokens_out` | Completion tokens (row-level field, already present) |
| API pricing snapshot | Model name + USD/M-token rates at time of run, in run-level meta |
| Total billed cost | tokens_in × input_rate + tokens_out × output_rate, summed across the run |

For local-only runs, omit cost fields entirely. Do not record $0.00.

---

## §10 Proposed Additions

The following fields are required by this protocol but do not currently exist in
the row schema. They MUST be recorded manually (in run notes or a companion
`results/run_<timestamp>_meta.json` file) until the runner is updated.

| Field | Where needed | Reason not yet in row schema |
|---|---|---|
| `timestamp` | §2 | Present in filename only, not in any row field |
| `placement` | §5 | Not captured; no `n_gpu_layers` or equivalent field |
| `residency` | §5 | Not captured; inferred indirectly from `gpu_mem_mb` curve |
| `kv_precision` | §5 | Not emitted; must be read from runtime log and recorded manually |
| `weight_precision` | §5 | In hardware YAML (`quant` field) but not propagated to result rows |
| `model_tier` | §5 | No canonical field; derivable from `model` string but not normalised |
| `regime` | §6 | No row-level field; currently derived post-hoc from `done_reason` and latency patterns |

Fields promoted out of this section on 2026-09-14:
`git_sha`, `run_seed`, `hostname`, `operator`, `done_reason` — all now
row-level fields in `harness/runner.py`; backward-compat None fill in
`evaluation/outcome.normalize_result_row()`.

---

## Change Log

| Date | Author | Change |
|---|---|---|
| 2026-09-14 | Rithwik Sharma | Initial draft. Derives from SCHEMA.md, telemetry.py, THREATS.md, KV_MEASUREMENT.md, and RESULT_PROVENANCE.md. |
| 2026-09-14 | Rithwik Sharma | Promote `git_sha`, `run_seed`, `hostname`, `operator`, `done_reason` from §10 (proposed) to normative §2 and §6. All five are now row-level fields emitted by harness/runner.py; backward-compat fill in evaluation/outcome.normalize_result_row(). |
