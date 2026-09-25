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
| `results/fig61_stagec_full_20260922T230133Z.jsonl` | blade_rtx4070 (Razer Blade 14, RTX 4070, CUDA b10970-bfdc32183, Windows 11) | **NO** | Fig 6.1 position-pressure sweep — **VALID (Blade CUDA replication)** | Gate: 0/22 disagreements with stage_c_20260818T040408Z.jsonl. cache_prompt=false verified (1.5x ratio at startup). PC: 387/396 pass; same 9 failures at r=0.40 LATE sea_01/sea_05/sea_06 (5.0–5.3%) as evo-t2s run. Score comparison vs 203557Z: 130/132 cells match. 2 disagreements: sea_01 LATE at r=1.0 and r=1.2 — Blade/CUDA outputs "C8" (wrong), evo-t2s/Vulkan outputs "A9" (correct). Both Blade runs (Ollama stage_c + CUDA llama-server) agree on "C8"; this is a backend numerical sensitivity on a borderline probe, not a data integrity issue. Position-pressure effect (LATE > EARLY at r<1 for retained-artifact probes) holds across 130/132 cells on both architectures. TTFT at full context ~1.2s on RTX 4070 discrete vs ~7.4s on evo-t2s unified (6× difference consistent with discrete vs unified memory bandwidth). |
| `results/fig61_toktrunc_full_20260922T233232Z.jsonl` | evo-t2s (Intel Arrow Lake, Vulkan b10970-bfdc32183, unified LPDDR5X) | **NO** | **TRUNCATION-METHOD VALIDATION ARM — not a replacement for 203557Z** | Token-accurate truncation via `/tokenize` binary search (`left_truncate_tokens`). PC: **396/396 pass** (all 9 marginal r=0.40 LATE sea failures from 203557Z are resolved — token-accurate delivery confirmed). Score diff vs 203557Z: **0/132 cells** — char-truncation over-delivery was conservative; no score changes result from fixing it. sea_01 LATE r=1.0 and r=1.2 still score 1.0 (out="A9") on Vulkan backend — CUDA/Vulkan divergence is a compute-backend property, not a truncation artifact. cache_prompt=false verified (7499ms vs 5854ms, ratio=1.3x). Use 203557Z as primary for score comparisons; use this file to bound THREATS §20. |
| `results/bw_saturation_20260923T045844Z.jsonl` | evo-t2s (Intel Arrow Lake, Vulkan b10970-bfdc32183, unified LPDDR5X) | **NO** | **Bandwidth saturation sweep — f16 arm** | qwen3-4b-instruct Q4_K_M, f16 KV, ctx=[8192, 16384, 32768, 65536, 131072], ~90% fill prompts. 5 rows status=ok. Row 6 (ctx=262144) status=http_200, inference aborted by SSH disconnect (WinError 10054), not RAM exhaustion. See note below. |
| `results/bw_saturation_20260923T065604Z.jsonl` | evo-t2s (Intel Arrow Lake, Vulkan b10970-bfdc32183, unified LPDDR5X) | **NO** | **Bandwidth saturation sweep — q8_0 and q4_0 arms** | qwen3-4b-instruct Q4_K_M, q8_0 and q4_0 KV. Complete (status=ok) for ctx=[8192, 32768, 65536, 131072, 262144] q8_0 and ctx=[8192, 16384, 32768, 65536, 131072] q4_0. ctx=16384 q8_0 has no usable metrics (status=incomplete_no_metrics — http_200 but per-config log block never printed, corrupted by terminal CR overwrite). ctx=262144 q4_0 was manually killed mid-run (status=killed_connection_reset, before/after memory only, no timing). ctx=524288 failed server-side ctx cap at both precisions (status=fail_http_400). All 15 rows reconstructed from `bw_sweep_run.log` (JSONL writer had a bug under `-NoNewWindow` redirect: `sys.__stdout__` is `None`, so the Tee class silently dropped writes to the JSONL path while the log file received everything). Timing values for ok rows cross-checked against the script's own final merged summary table printed at the end of the log run. |
| `results/kv_val_vulkan_20260923.json` | evo-t2s (Intel Arrow Lake, Vulkan b10970-bfdc32183, unified LPDDR5X) | **NO** | **KV precision validation — three cold starts** | Three server starts at ctx=32768 (f16, q8_0, q4_0) on port 8384, no inference. Memory delta confirms KV is compressed: f16=7599 MiB, q8_0=5458 MiB, q4_0=4279 MiB. b10970 Vulkan does not emit llama_kv_cache log line. Derived B/tok: f16=147,456 (arch), q8_0~73,765 (1.999× reduction), q4_0~36,892 (3.997× reduction). See docs/KV_MEASUREMENT.md §6–7. |
| `results/kv_quality_20260923T181840Z_ctx32768.jsonl`, `..._ctx8192.jsonl`, `..._full.jsonl` | evo-t2s (Intel Arrow Lake, Vulkan b10970-bfdc32183, unified LPDDR5X) | **NO** | **KV precision quality sweep — does quantized KV cost retrieval accuracy?** | 12 configs (ctx∈{32768,8192} × prec∈{f16,q4_0,q8_0}) × 2 positions (ADJACENT = artifact immediately before question; START = artifact at the beginning, filler after) × 10 artifact-retrieval probes (art_01–art_10, exact-match) = 120 calls, all `status=ok`. **119/120 score 1.0.** Sole non-1.0: `art_06`/ADJACENT/ctx=8192/q4_0 scores 0.0 (output "Herrera", expected "Blum") while the START counterpart at the same config, and art_06 at every other config, score 1.0 — this exact confusion (Motion field vs. Second field) is already documented as a pre-existing model-level failure mode in this doc's Stage 1.1 section (`schema_collision.json`), independent of KV precision, so this is more likely a recurrence of that known bug than a new quantization effect — one data point, not confirmed. **Do not extend this null result to ctx=131072 or beyond**: softmax quantization error accumulates with key count, and long context is exactly where an effect would be expected to appear; this run does not test that range. KV precision is confirmed only via `sys_free` memory delta per config (`kv_log_note` field states this explicitly in every row) — b10970 Vulkan emits no KV buffer-size log line at any verbosity, as established above. Known imprecision: filler is sized against the probe with the largest artifact+question token count per ctx, then trimmed shorter per probe; probes below that maximum are filled to slightly under the 90% target (up to ~150 tokens short — ~2% at ctx=8192, ~0.5% at ctx=32768). Actual achieved token count is recorded per row (`n_prompt_tokens_actual`). Wall clock: 316.5 min. Script: `harness/kv_quality_sweep.py`. **Deviations from the validated Stage C config** (`fig61_stagec_full_20260922T203557Z.jsonl`, which records `reasoning_flags: "none"` — no reasoning override passed, server/model default reasoning behavior in effect): this sweep's server launch passes `--reasoning-format deepseek --reasoning-budget 0` (copied from `bw_saturation_sweep.py`, which used these flags for a throughput sweep where reasoning tokens were unwanted noise), forcing reasoning off/zero-budget rather than leaving it at whatever qwen3-4b-instruct's default is. If the model's default includes chain-of-thought before the final answer, Stage C measured that behavior and this sweep measured behavior with reasoning suppressed — these are not directly comparable configs, and any comparison between this sweep's scores/timing and Stage C's should account for that. **Precision-check failure:** `sys_free_before/after_server_mib` deltas are negative for 5 of 6 configs (only the first-ever config in the run, ctx=32768/f16, has a clean positive delta) — see `docs/FINDINGS.md` "KV Precision Quality Sweep" section for the full diagnosis (kill_server()'s fixed 3s sleep is not always enough for a ~7.5-10 GiB process's memory to fully release before the next config's "before" snapshot) and the fix applied going forward (wait for previous PID exit; record the new server's own process command line + private memory via `Get-CimInstance`/`Get-Process`, not system-wide free memory). These five rows carry no valid per-row memory-based precision confirmation; no backfill was possible from the per-config server logs (same log-format limitation as established elsewhere in this document — no command-line or KV-size line at any verbosity). |

| `results/kv_provisioning_20260924T003514Z_ctx32768_f16.jsonl` | evo-t2s (Intel Arrow Lake, Vulkan b10970-bfdc32183, unified LPDDR5X) | **NO** | **Provisioning comparison — f16@32768 leg (fast baseline); q4_0@131072 leg ABORTED, no data** | Only leg completed: START position, 4 probes (art_01, art_05, art_06, art_09), 1 rep each, 4/4 score 1.0. The q4_0@131072 leg was killed by explicit instruction ~1h38m into an estimated ~5.3h run (2026-09-23 ~7:33 PM, on probe 1-2 of 4) — the project's own retrospective judgment that easy-probe retrieval accuracy at ceiling under abundant free memory is not informative, in favor of the memory-pressure experiments below. **No partial data exists for this leg**: the script only writes its per-config JSONL after all 4 probes in that config finish, and stdout was fully buffered (non-tty redirect) with nothing flushed to the log before the kill — there is nothing to reconstruct, unlike the bw_saturation sweep's kill (which had a final merged-summary table). Script: `harness/kv_provisioning_sweep.py`. Same deviations as the quality sweep above apply (`--reasoning-format deepseek --reasoning-budget 0`). |
| ~~`results/kv_neardup_*.jsonl`~~ | — | — | **DISCARDED before running** | `harness/kv_neardup_sweep.py` was designed but never run, then discarded (commit `dcf6f31`) — it would have restated the existing type-matched interference finding already in FINDINGS.md rather than extending it. No file exists at this path. |
| `results/memory_pressure_20260924T025611Z.jsonl` | evo-t2s (Intel Arrow Lake, Vulkan b10970-bfdc32183, unified LPDDR5X) | **NO** | **INVALID as a memory-constraint experiment — balloon design flaw** | 7-level "balloon" sweep (16/10/8/7/6/5/4 GB) intended to test llama-server behavior under real memory pressure. The balloon (`memory_balloon.py`) held **Available MBytes at a moving target**: its control loop measured current available memory and grew/shrank itself every tick to keep available pinned at the target. When llama-server allocated memory, available dropped below target, and the balloon's own control loop released memory to bring available back up — i.e. the balloon got out of the server's way every time the server actually needed something. The server was never memory-constrained at any level; it always got what it asked for. This is why all 7 levels scored `runs_normally` with flat ~277-299s TTFT and 10/10 correctness: nothing was ever actually squeezed. Compounding this, the 4 GB level's own `available_mb_after_stabilize` field (9003 MB, not ~4096 MB) shows the balloon didn't even hold its pre-server-start target reliably at the most extreme level. **Do not cite this file for anything beyond "the harness plumbing works end-to-end" (server starts, throughput/correctness calls succeed, positive-control queries succeed) — it demonstrates none of the memory-pressure behavior it was built to measure.** Superseded by a fixed-allocation balloon design (allocate once before the server starts, never resize during the level) — see the corrected experiment once run. |
| `results/memory_pressure_v2_20260924T034948Z.jsonl` | evo-t2s (Intel Arrow Lake, Vulkan b10970-bfdc32183, unified LPDDR5X) | **NO** | **INVALID as a memory-constraint experiment — unlocked balloon got paged out by Windows** | Single smoke level at X=7 GB budget using the v2 fixed-allocation balloon (`memory_balloon.py`, never resizes itself during the level — that specific bug from the v1 file above is fixed). But the balloon's own log shows `balloon_own_working_set_mb` dropping from 57866.9 to as low as 28712.5 MB against a `balloon_size_mb_fixed` that stayed exactly 57850.0 throughout: the balloon's **committed allocation** never changed, but Windows **trimmed its resident working set** once llama-server needed memory, handing the freed physical pages to the server anyway — the same net effect as the v1 bug (server never truly constrained), just caused by the OS's own memory manager instead of this script's control loop. `pages_per_sec_hard_faults` (real disk-based paging, not the noisier Page Faults/sec) spiked to 380,105/s, 237,371/s, and 221,820/s at various points, confirming real thrashing occurred — but it was the balloon fighting the OS to keep its own pages resident, not the server being denied memory it needed. `lock_status=UNLOCKED` in the balloon's own log header states this plainly: VirtualLock was attempted and failed because the `sharc` account does not hold `SeLockMemoryPrivilege` (`AdjustTokenPrivileges` returned `ERROR_NOT_ALL_ASSIGNED`). **All 10 correctness probes scored 1.0** and throughput/correctness TTFT stayed close to the unconstrained baseline (~297-345s vs. 277s baseline) — the server was not measurably slowed by this contaminated pressure. **Cost-read correction**: the ~58-minute total wall clock for this one level is NOT balloon-thrashing overhead — 10 full 90%-fill ctx=32768 correctness probes at ~331-345s TTFT each sum to ~3364.5s (~56 min) on their own, matching the established unconstrained baseline for calls of this size; the balloon's thrashing added wall-clock noise between calls (tick intervals stretching to 130-250s at times) but the correctness-probe time itself is baseline-consistent, not pressure-inflated. **Superseded by two follow-up methods**: primary — a boot-time RAM cap via `bcdedit truncatememory` on EVO-X2 (`docs/RAM_CAP_PROTOCOL.md`), which removes the OS's ability to hand contested pages to either party since the physical ceiling is enforced below the OS memory manager entirely; secondary (no-reboot, fine-grained) — a genuinely locked balloon, designed in `docs/RAM_CAP_PROTOCOL.md` but not yet built or run, requiring `SeLockMemoryPrivilege` to be granted first. **UPDATE 2026-09-24: the locked-balloon secondary method is now built** (`harness/memory_balloon_awe.py`, AWE-based via `AllocateUserPhysicalPages`), `SeLockMemoryPrivilege` has been granted to `sharc` on evo-t2s (exact procedure and revert command in `docs/RAM_CAP_PROTOCOL.md`), and the primary boot-time RAM-cap protocol plus this locked balloon are both queued to run (RAM-cap on EVO-X2 per instruction; the locked balloon as Phase D of an overnight run on evo-t2s). |
| `results/contention_blade_rtx4070_20260924T171907Z.jsonl` | blade_rtx4070 (Razer Blade 14, RTX 4070, CUDA b10970-bfdc32183, Windows 11) | **NO** | **Contention sweep — VALID, corrected model** | Supersedes an earlier (never committed) run that used `Qwen3-4B-Q4_K_M.gguf` (a hybrid-thinking checkpoint) with `--reasoning-format deepseek --reasoning-budget 0` — a real confound (a thinking-capable model forced not to think, versus a model that never thinks) discovered before that run's data was ever saved. This run uses the same plain instruct GGUF as evo-t2s (sha256 `85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25fce54b18b9`, verified at startup), no reasoning flags. 14 conditions (baseline x3 at first/middle/last, memcpy at 4/8/12/16 threads, spin at 4/8/12/16 threads, pytest, pandas_groupby, compile_proxy) x 2 ctx (8192, 32768) = 28 rows, 3 reps each, all `outcome=ok`, no unexpected errors. Positive controls: memcpy achieved 21.5-25.9 GB/s across all conditions (well above the 1 GB/s STOP floor); spin achieved 18M-53M iterations/s. **Real baseline drift observed**: ctx=32768 baseline TTFT climbed 19.4s (first) -> 35.5s (middle) -> 44.6s (last) over the ~50-minute tier — plausible thermal throttling on a laptop chassis under sustained load, not a measurement artifact (memcpy/spin positive controls stayed consistent throughout, so the co-runners themselves were not degrading). memcpy_16 (100% cores) and compile_proxy independently pushed ctx=32768 TTFT to ~53s, comparable severity from a synthetic saturation load and a realistic tool workload. Manifest: `results/contention_blade_rtx4070_20260924T171907Z_manifest.json`. |

| `results/contention_evo-t2s_20260924T171913Z.jsonl` | evo-t2s (Intel Arrow Lake, Vulkan b10970-bfdc32183, unified LPDDR5X) | **NO** | **Contention sweep — VALID, unified-memory contrast to Blade above** | Same design and corrected model/flags as the Blade run above (14 conditions x 2 ctx = 28 rows, 3 reps each, all `outcome=ok`, no unexpected errors). Positive controls: memcpy achieved 34.5-42.6 GB/s; spin achieved 44M-127M iterations/s. **Zero baseline drift**: ctx=32768 baseline TTFT held at 277.1s (first) / 277.1s (middle) / 278.5s (last) across the ~2h17min tier — sharp contrast to Blade's significant thermal drift, consistent with a non-laptop chassis. **memcpy saturates fast**: TTFT plateaus at ~413s from 8 threads (50% of cores) onward at ctx=32768 (8/12/16 threads all ~412-414s) — additional bandwidth-contending threads beyond that point don't matter once the memory bus itself is saturated. **spin (core-only contention) also shows a real, separate effect**: TTFT climbs to ~392s at 12-16 threads (75-100% of cores) at ctx=32768, well above the 277s baseline, despite zero memory-bandwidth traffic -- confirming core contention alone (not just bandwidth) measurably slows this workload, which is exactly why the memcpy/spin split exists (memcpy_slowdown minus spin_slowdown at the same thread count isolates the bandwidth-specific component from ordinary core contention). **pytest and pandas_groupby show almost no effect** (276.8s and 278.1s vs. 277.1s baseline) -- neither co-runner meaningfully competes with the model server on this hardware. compile_proxy shows a small-moderate effect (282.0s). Manifest: `results/contention_evo-t2s_20260924T171913Z_manifest.json`. |

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

## bw_saturation_20260923T045844Z.jsonl — Bandwidth sweep note

**Produced on:** evo-t2s (Intel Arrow Lake, Vulkan b10970-bfdc32183, unified LPDDR5X).
**Date:** 2026-09-23. **Script:** `C:\apu\bw_saturation_sweep.py` (not checked into repo).
**Model:** `qwen3-4b-instruct-85e4a5b7.gguf` (Q4_K_M, same checkpoint as fig61 runs).

**Results (f16, rows 1–5):**

| ctx | KV (GiB, computed) | prefill tok/s | decode tok/s | TTFT (s) |
|---|---|---|---|---|
| 8,192 | 1.1 | 402.6 | 15.8 | 18.3 |
| 16,384 | 2.2 | 207.3 | 10.2 | 71.1 |
| 32,768 | 4.4 | 106.3 | 5.8 | 277.4 |
| 65,536 | 8.9 | 53.7 | 3.2 | 1,097.8 |
| 131,072 | 18.1 | 27.3 | 1.7 | 4,322.0 |

**Key finding:** Prefill throughput scales as exact 1/N across a 16× context range
(8K→131K): 402 → 207 → 106 → 54 → 27 tok/s, ratio per doubling = 2.00× ± 0.02.
This is pure memory-bandwidth-limited operation with no capacity cliff on unified memory.
Decode follows the same trend (halving per doubling of ctx), but from a lower absolute rate.
At ctx=131072, decode=1.66 tok/s is unusable regardless of whether memory fits.

**Row 6 (ctx=262144) — aborted, not a RAM error:**
Server loaded successfully (sys_free dropped from 59526 to 18767 MiB = ~40 GiB allocated,
consistent with model + 36 GiB KV). Inference was in progress (235,929-token prompt was
built; HTTP 200 received) when the SSH session carrying the Python process was killed.
Exact error: `[WinError 10054] An existing connection was forcibly closed by the remote host`.
**No config in this file failed due to genuine RAM exhaustion.** The "9 GiB allocation cap"
hypothesis from the design phase was incorrect; the allocator placed 40 GiB with no refusal.

**What was NOT in the original "no-ceiling" observation:** The earlier run that loaded
ctx=262144 f16 (from the design phase) showed only that the allocator does not refuse.
It did not measure throughput. The present sweep supplies throughput measurements.

**KV constant validation:** `measured_kv_mib` is null in all rows (server log overwritten
each restart; log parser returned `{}`). Indirect estimate from incremental memory deltas
gives ~152,078 B/tok (upper bound; includes compute buffers). See `docs/KV_MEASUREMENT.md §6`.

**q8_0 and q4_0 arms:** Complete in `results/bw_saturation_20260923T065604Z.jsonl`
for ctx=8192–131072 (q4_0 also complete at 262144 before the run was killed for
262144 f16-equivalent cost reasons — see row status). ctx=16384 q8_0 has no usable
metrics (per-config log block corrupted by terminal CR overwrite; only the
computed/theoretical KV size is known for that row, not a measurement). ctx=524288
failed a server-side ctx cap (server refuses n_ctx > 262144 regardless of requested
value) at both precisions — this is not a capacity/OOM measurement.

**Flash attention / attention-path mechanism (evo-t2s Vulkan, b10970-bfdc32183):**
The runtime log at `--log-verbosity 3` does not print a flash-attention or KV-buffer
line at any point — this cannot be determined from the log. It was determined instead
from llama.cpp source at the exact upstream commit the binary was built from
(`bfdc32183d57f1e35bacf35c47d6311e2028bbbc`, confirmed via `gh api
repos/ggml-org/llama.cpp/commits/bfdc32183`): the sweep script never passes `-fa`, so
`flash_attn_type` defaults to `AUTO` (`common/common.h:499`); when the V-cache type is
quantized, `llama-context.cpp:3704-3707` force-enables flash attention under `AUTO`
because the non-flash path cannot consume quantized V at all. So the q8_0 and q4_0 rows
in this sweep ran with flash attention forced on by this code path. Whether AUTO also
resolves to flash-attn for the f16 rows (unquantized V, so the force-enable branch does
not trigger), and whether the Vulkan flash-attn kernel dequantizes KV per block or
operates on quantized bytes directly, was not determined — that requires reading
`ggml/src/ggml-vulkan/ggml-vulkan.cpp`'s flash-attention dispatch, which has not been
done. See `docs/FINDINGS.md` "Bandwidth Saturation" section for the full citation and
the throughput data this explains.

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
