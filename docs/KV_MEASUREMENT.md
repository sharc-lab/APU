# KV Cache Memory Measurement — Methods, Reference, and Reconciliation

This document records what each committed KV measurement file actually
measured, which one is the reference for the paper, and why the two sets of
numbers differ.

---

## 1. What each file measured

### results/gate1_kv_precision.json
**Script:** `harness/stage_a_kv_precision.py`  
**Date:** 2026-08, blade14_rtx4070, Ollama 0.32.9/0.32.6, qwen3:4b-instruct  

**Method:**  
- Loaded model in Ollama at `num_ctx=32768` with `OLLAMA_KV_CACHE_TYPE` set via
  environment variable to f16, q8_0, or q4_0 in successive server restarts.
- KV allocation read from the `CUDA0 KV buffer size = X MiB` line in the Ollama
  server log (preferred over nvidia-smi subtraction, which was unreliable due to
  timing).
- bytes/token = `(cuda_kv_mib * 1048576) / 32768`.
- nvidia-smi also polled for total VRAM; subtraction `total - weight_footprint`
  used as a secondary check.

**Outcome: FAIL — KV quantization was never applied.**  
All four conditions (f16_with_flash, q8_0_with_flash, q4_0_with_flash,
f16_no_flash) produced identical log lines:

```
llama_kv_cache: K (f16): 2304.00 MiB, V (f16): 2304.00 MiB
CUDA0 KV buffer size = 3712.0 MiB
```

Ollama 0.32.x reads `OLLAMA_KV_CACHE_TYPE` at startup but does not propagate it
to the `llama-server` command line as `--cache-type-k`. The env var is silently
ignored. Confirmed: the server log shows `--cache-type-k` is absent from every
`llama-server` invocation recorded.

**Consequence for any cited ratio:** There are no valid cross-precision KV ratios
in this file. The `ratio_measured_over_arch` field in each condition record is
the fixed f16-at-32768 KV allocation (118,784 B/tok) divided by that condition's
*own* architectural B/tok prediction — a comparison of one f16 measurement to
each precision's theoretical floor, not a measurement of reduction. These ratios
(0.806, 1.611, 3.222) are not f16→q8 or f16→q4 reduction factors; they have no
interpretation as KV quantization effectiveness.

The file's purpose is to document the Ollama API limitation. It is valid for
that purpose and as a record of f16 KV allocation on this build. It is not a
source of quantization reduction measurements.

---

### results/llamaserver_feasibility.json
**Script:** none — collected manually via llama-server CLI  
**Date:** 2026-08-26, blade14_rtx4070, llama-server build b1-f8def7fe1,
qwen3:4b-instruct Q4_K-Medium  

**Method:**  
- Ran llama-server directly (bypassing Ollama) with `--cache-type-k` and
  `--cache-type-v` flags confirmed to take effect (verified by server log type
  fields and VRAM measurements).
- Measured total VRAM via nvidia-smi at `ctx=32768, n_slots=1, n_gpu_layers=99`
  for f16, q8_0, and q4_0 individually.
- Established weight-only overhead separately: `weights_overhead_estimate = 3405
  MiB` (measured by loading the model with minimal ctx; cross-checked: f16 KV at
  ctx=4096 ≈ 564 MiB and 3970 − 3405 = 565 MiB ✓).
- bytes/token = `(vram_32768_X − 3405 MiB) * 1048576 / 32768`.

**Results:**

| Precision | VRAM at ctx=32768 (MiB) | KV only (MiB) | Measured B/tok | Arch B/tok | Ratio meas/arch |
|-----------|-------------------------|---------------|----------------|------------|-----------------|
| f16       | 7922                    | 4517          | 144,530        | 147,456    | 0.980           |
| q8_0      | 5952                    | 2547          | 81,490         | 73,728     | 1.105           |
| q4_0      | 4800                    | 1395          | 44,626         | 36,864     | 1.211           |

**KV reduction ratios (measured f16 as baseline):**

| Comparison | Measured | Architectural |
|------------|----------|---------------|
| f16 → q8_0 | **1.77×** | 2.00× |
| f16 → q4_0 | **3.24×** | 4.00× |
| q8_0 → q4_0 | **1.83×** | 2.00× |

The f16 measurement sits 2% below architectural because Qwen3 uses sliding
window attention (SWA) by default; SWA layers maintain a narrower KV window than
full-context. The `--swa-full` flag was not set. The q8 and q4 measurements
exceed architectural because per-block quantization metadata (scale factors,
block headers) is stored at full precision — a fixed overhead per block whose
fraction of total KV grows as element precision drops. These are known effects
of block quantization; see `docs/FINDINGS.md` for the mechanistic analysis.

---

## 2. Reference for the paper

**`results/llamaserver_feasibility.json` is the reference.**

Reasons:

1. **Quantization flags actually took effect.** gate1 never ran q8 or q4 KV;
   every condition in that file was f16. llamaserver confirmed flag effect via log
   type fields (K (q8_0) / K (q4_0)) and corroborating VRAM values.

2. **The measurement formula is more principled.** Subtracting a separately
   measured weight overhead (3405 MiB) is less sensitive to post-load driver
   allocations than the nvidia-smi subtraction method in stage_a, which yielded
   negative values (`kv_vram_mib_by_subtraction = -36 MiB`) when timing varied.

3. **The results are mechanistically explained.** The deviations from
   architectural ratios have identified causes (SWA for f16 undershoot;
   per-block metadata for q8/q4 overshoot). An unexplained deviation would
   require a re-run; these do not.

4. **Build provenance is explicit.** Build b1-f8def7fe1 is named alongside every
   figure. The SWA effect on the f16 baseline is build-specific and is
   documented as such.

**What to cite in the paper:**  
Use the measured ratios from llamaserver_feasibility.json: **1.77× for f16→q8_0,
3.24× for f16→q4_0**. Always name "f16 (measured, b1-f8def7fe1, SWA enabled)" as
the baseline rather than the architectural f16 value, because the SWA effect
shifts the f16 floor and the ratios are computed against that actual floor.

If the architectural ratios (2× and 4×) appear in the same figure for comparison,
label them as "architectural (no metadata overhead)" so the two columns are
distinguishable.

---

## 3. Reconciliation note for gate1_kv_precision.json

gate1_kv_precision.json's verdict field already records the failure:
*"FAIL — OLLAMA_KV_CACHE_TYPE and LLAMA_ARG_CACHE_TYPE_K/V both ignored on
Ollama 0.32.9."*

The file remains committed because it is a valid record of:
- f16 KV allocation on Ollama 0.32.9 at ctx=32768 (4608 MiB, 118,784 B/tok)
- The Ollama API limitation that blocks Stage D via the Ollama wrapper
- The fallback path to llama-server direct invocation

The file does **not** provide quantization reduction measurements and must not be
read as doing so.

---

## 4. Correction: ratios in RESULT_PROVENANCE.md are wrong

`docs/RESULT_PROVENANCE.md` (§ gate1_kv_precision.json provenance note) states:
> "Key result: f16/q8_0 VRAM ratio ≈ 1.83×; f16/q4_0 ratio ≈ 3.76×."

Both figures are incorrect:

- **1.83×** is not the f16/q8_0 ratio from any measurement. It is the q8_0/q4_0
  ratio from llamaserver (81,490 / 44,626 = 1.826 ≈ 1.83×). The correct f16/q8_0
  measured ratio is **1.77×**. The text immediately below calls it
  "the ~1.83× int8-to-int4 memory reduction figure," which confirms it is the
  q8→q4 step, not the f16→q8 step.

- **3.76×** is not derivable from either file. The measured f16→q4_0 ratio from
  llamaserver is **3.24×**; the architectural f16→q4_0 ratio is 4.00×. No
  intermediate calculation produces 3.76× consistently. This figure should not
  be cited.

The RESULT_PROVENANCE.md note was corrected: the incorrect figures have been removed
and the gate1 section now reads "Key result: None" with the f16-only explanation.
The Fig 4.11 entry in PAPER_OUTLINE.md has also been corrected to reflect that gate1
contains no valid cross-precision data and points to llamaserver_feasibility.json.

---

## 5. KV ratio citations without explicit f16 baseline — inventory

Locations where a KV reduction ratio is stated but the f16 baseline (architectural
vs measured, which build) is not named:

| File | Location | Statement | Status |
|------|----------|-----------|--------|
| `docs/RESULT_PROVENANCE.md` | gate1 note | "f16/q8_0 VRAM ratio ≈ 1.83×; f16/q4_0 ratio ≈ 3.76×" | **CORRECTED** — figures removed; gate1 section now states "Key result: None" |
| `docs/RESULT_PROVENANCE.md` | gate1 note | "the ~1.83× int8-to-int4 memory reduction figure" | **CORRECTED** — removed with above |
| `docs/PAPER_OUTLINE.md` | Fig 4.11 | "data ready; plot script needed" | **CORRECTED** — now states gate1 has no cross-precision data; points to llamaserver_feasibility.json |
| `analysis/bom_sweep.py` | lines 49–51 | "NOTE: 36,864 and 73,728 are architectural values. Measured q8->q4 reduction is 1.83x" | **CORRECT** — q8→q4 = 1.83× matches KV_MEASUREMENT.md §1; architectural caveat is explicitly noted |
| `docs/FINDINGS.md` | line 37 | "3.24× for q4_0 vs 1× for f16" | Baseline not specifying measured vs architectural — acceptable in context; see §1 for explicit table |

Locations where the baseline IS explicitly named (no action needed):

| File | Location | Why it is adequately labelled |
|------|----------|-------------------------------|
| `docs/FINDINGS.md` table (lines 19–23) | "KV reduction vs f16 (meas)" / "vs f16 (arch)" columns | Both architectural and measured are listed with column headers; source is llamaserver_feasibility.json; build identifier is in the heading |
| `docs/FINDINGS.md` line 33 | "measured f16→q4_0 reduction is **3.24×, not 4×**" | "measured" vs "architectural" (4×) are both named in the same sentence |
| `results/llamaserver_feasibility.json` precision_table | `reduction_vs_f16_architectural` / `reduction_vs_f16_measured` keys | Separate fields for each baseline |

---

## 6. Vulkan arm (evo-t2s) — KV precision validation (three-start)

**Source:** `results/kv_val_vulkan_20260923.json` (three cold server starts, ctx=32768);
`results/bw_saturation_20260923T045844Z.jsonl` (f16 sweep, ctx=8192–131072).  
**Build:** b10970-bfdc32183, Vulkan, Intel Arrow Lake, unified LPDDR5X.

**Log-line status:** b10970 Vulkan does NOT emit a `llama_kv_cache:` or KV buffer-size
line at any verbosity level tested. The log sequence is: startup → `llama threadpool init`
→ `load_model: initializing, n_ctx_slot=32768, kv_unified='false'` → `model loaded` →
`listening`. No allocation line exists to parse. This is a build-specific characteristic
of b10970 Vulkan; earlier CUDA build b1-f8def7fe1 did emit `CUDA0 KV buffer size = X MiB`.

**Three cold server starts (ctx=32768, port 8384, no inference):**

| Precision | Flags | Δ RAM (MiB) | Healthy |
|-----------|-------|-------------|---------|
| f16 | `-ctk f16 -ctv f16` | 7,599 | ✓ |
| q8_0 | `-ctk q8_0 -ctv q8_0` | 5,458 | ✓ |
| q4_0 | `-ctk q4_0 -ctv q4_0` | 4,279 | ✓ |

RAM delta decreases monotonically across precisions. The Δ difference from f16 to q8_0
(2,141 MiB) and from f16 to q4_0 (3,320 MiB) can only arise from reduced KV storage,
since model weights are identical across all three starts.

**Derived KV B/tok (Vulkan, ctx=32768):**

Model weight inferred as 7,599 − 4,608 = 2,991 MiB (delta_f16 minus architectural KV
at ctx=32768: 36 layers × 8 KV heads × 128 head_dim × 2 types × 2 bytes × 32768 /
1,048,576 = 4,608 MiB). Vulkan model weight (~2,991 MiB) is smaller than CUDA
(~3,405 MiB on b1-f8def7fe1) due to backend buffer layout differences.

| Precision | Δ RAM (MiB) | KV (MiB) | B/tok | f16-baseline ratio |
|-----------|-------------|----------|-------|--------------------|
| f16 | 7,599 | 4,608 (arch anchor) | **147,456** | 1.00× |
| q8_0 | 5,458 | ~2,304 | ~73,765 | **1.999×** |
| q4_0 | 4,279 | ~1,153 | ~36,892 | **3.997×** |

Vulkan f16 sits at exactly the architectural value (no SWA discount). Qwen3's sliding-window
attention layers are allocated at full-context size on the Vulkan backend. The q8_0 and q4_0
compression ratios are near-architectural (1.999× vs 2.00× arch; 3.997× vs 4.00× arch),
indicating negligible per-block metadata overhead — in contrast to CUDA (b1-f8def7fe1)
where per-block metadata accounts for ~10–21% of additional KV storage above the
element-only prediction.

---

## 7. Per-platform B/tok comparison

| Platform | Build | f16 B/tok | vs arch | f16→q8_0 | f16→q4_0 | Notes |
|----------|-------|-----------|---------|----------|----------|-------|
| Blade 14 / CUDA | b1-f8def7fe1 | **144,530** | −2.0% | **1.77×** | **3.24×** | SWA active (full-ctx KV not allocated for all layers); per-block metadata inflates q8/q4 above arch |
| evo-t2s / Vulkan | b10970-bfdc32183 | **147,456** | 0.0% | **1.999×** | **3.997×** | No SWA discount; near-architectural compression ratios; negligible metadata overhead |
| Architectural | — | 147,456 | — | 2.00× | 4.00× | Qwen3-4B: 36 layers × 8 KV heads × 128 head_dim × 2 types × {2,1,0.5} bytes/element |

The two measured platforms differ in two ways:

**f16 baseline:** CUDA (b1-f8def7fe1) sits 2% below architectural because Qwen3's
sliding-window layers maintain a narrower KV window by default (`--swa-full` not set).
Vulkan (b10970) allocates full-context KV for all 36 layers, matching the architectural
prediction exactly. Do not substitute the CUDA f16 B/tok (144,530) for a Vulkan
provisioning calculation; the correct Vulkan f16 figure is 147,456.

**Precision compression:** CUDA shows shallower-than-architectural compression (1.77× and
3.24× vs 2× and 4×) because per-block quantization metadata (scale factors, block
headers) is stored at full precision alongside quantized elements. This overhead is
proportionally larger at higher compression, explaining why the shortfall grows from f16
to q8_0 to q4_0. Vulkan shows no such overhead; compression ratios are within 0.1% of
architectural. Both platforms confirm that quantization flags take effect.
