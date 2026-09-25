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

### Corrections to the two contention entries above (2026-09-25, no reruns — reading rules only)

1. **Manifest `hw_config` field is wrong for both `contention_evo-t2s_*_manifest.json` files**: it reads `"evox2_evo-t2s"`, a copy-paste artifact from an earlier script's constant. **Read this as `evo-t2s`** (Intel Arrow Lake, Vulkan b10970-bfdc32183, unified LPDDR5X) — the same machine identified as `evo-t2s` everywhere else in this document. This is a label bug in the manifest JSON, not a different hardware configuration; the manifest file itself is not being edited to fix it (per instruction — the wrong string is a fact about what the script produced, not something to retroactively correct in place).

2. **"Bandwidth component = memcpy slowdown minus spin slowdown" is retracted.** It assumed the two effects are additive (core contention and bandwidth contention sum linearly), which was never established and may not hold — they can act through different, non-additive mechanisms (e.g. scheduling interference vs. memory-controller queuing). **Report memcpy slowdown and spin slowdown as two separate ratios going forward, not a subtracted "component."** Any occurrence of the subtracted quantity in earlier analysis in this document or in chat should be disregarded.

3. **Blade `ctx=32768` results are INVALID for slowdown-ratio analysis**, not merely caveated: baseline TTFT drifted 19.4s → 35.5s → 44.6s across the tier (thermal), so every "slowdown vs. baseline" figure at that ctx is computed against a moving, contaminated reference. The raw per-condition TTFT values themselves are real measurements and remain in the committed file; only the *derived ratios* against baseline are invalid at this ctx on this platform. Blade `ctx=8192` baselines were stable (1.8s / 1.8s / 1.8s) and its ratios there are trustworthy.

4. **Blade `memcpy_12` at `ctx=8192` has no achieved GB/s recorded** (`memcpy_achieved_gbps: null` in the committed row) — the positive control is missing for that one condition specifically. The TTFT measurement itself was recorded (11.1s), but there is no verification that real memcpy load was actually running during that specific condition the way the STOP-gated check requires elsewhere. Treat that one row's load as unverified, not as evidence of a specific GB/s figure.

| `results/blade_m1_vram_spill_20260925T041415Z.jsonl` | blade_rtx4070 (Razer Blade 14, RTX 4070 Laptop, 8188 MiB VRAM, CUDA b10970-bfdc32183) | **NO** | **M1 — VRAM spill confirmed, cliff between ctx=32768 and ctx=40960** | Same GGUF/flags/no-co-runner design as elsewhere (sha256-verified, f16, `-fa on`, no reasoning flags). ctx in {8192,16384,24576,32768,40960}, 3 throughput calls per ctx, 60s cooldown, 1s telemetry (`results/m1_telemetry_<ctx>_*.csv`: nvidia-smi memory/clocks/power/temp/utilization/throttle-reasons + Windows GPU Process Memory Dedicated/Shared Usage for the server PID). **Prior art**: this replicates NVIDIA's well-documented "CUDA Sysmem Fallback Policy" (driver 536.40+) on this specific model/context combination — not a novel mechanism, a local confirmation of where it kicks in here. **Confirmed spill, not thermal**: at ctx=40960, `memory.used` sits at 7919-7925/8188 MiB (VRAM effectively full), `gpu_shared_usage` jumps to ~627 MB from a ~120-128 MB baseline at every lower ctx (24576: 119.5 MB; 32768: 127.9 MB) — a genuine ~500 MB spill into system RAM, not baseline driver overhead. SM clocks stay pegged at 2550-2565 MHz throughout (never throttled) and `clocks_throttle_reasons.active` reads `0x0` — ruling out thermal/power throttling as the cause. Power draw instead **drops** from ~95-103 W (full compute, ctx≤32768) to ~27-56 W at ctx=40960 while `utilization.gpu` still reads 99-100% — the documented Sysmem Fallback signature exactly (GPU reports busy while actually idling on PCIe transfers for spilled data, not computing). **TTFT discontinuity matches**: 1.8s→4.9s→11.0s→19.5s (smooth ~1.3-1.8x per ctx-doubling step) then **19.5s→117-119s at the next step (6x)** — a cliff, not a continuation of the trend; decode_tps correspondingly crashes 28.7→~6.3 tok/s. Queued follow-up: flip NVIDIA Control Panel → Manage 3D Settings → CUDA - Sysmem Fallback Policy → "Prefer No Sysmem Fallback" (exact known fix per prior art) and rerun ctx=40960 only, to confirm it now fails loudly (OOM) instead of silently degrading. |

| `results/blade_m2_host_interference_20260925T043921Z.jsonl` | blade_rtx4070 (Razer Blade 14, RTX 4070 Laptop, 8188 MiB VRAM, CUDA b10970-bfdc32183) | **NO** | **M2 — host-critical-path (CPU starvation) supported; shared-power and PCIe-bus-contention NOT supported** | ctx=8192 throughout (verified `gpu_mem_initial.shared`=~98MB, i.e. no VRAM spill confound at this ctx). 27 rows: Part A (`none`/`memcpy_16`/`spin_16` at server `-t 4`, 3 calls each), Part B (`memcpy_16` rerun at server `-t 1` and `-t 8`, 3 calls each), Part C (dose-response `memcpy` at 16 threads throttled toward 5/10/15/20 GB/s targets, server `-t 4`, 3 calls each). 1s `nvidia-smi dmon -s pucvmt` telemetry per condition (`results/m2_dmon_<condition>_*.txt`) plus GPU Process Memory polling (`results/m2_gpumem_<condition>_*.csv`). **Verdict: host-critical-path.** Under `A_memcpy16`, GPU power collapses to 1-25 W (vs. 93-104 W baseline) and SM clock (`pclk`) drops to 210-1920 MHz (vs. a steady 2550-2565 MHz baseline) *together with* utilization dropping to 0-50% (vs. 92-100% baseline) — the GPU is genuinely idle, not stalled-but-busy. This is a different signature from M1's VRAM-spill cliff (where utilization stayed pegged at 99-100% while only power dropped) and rules out a shared-power-budget explanation (power fell as a *consequence* of the GPU having no work queued, not as a cause of slower compute). `rxpci`/`txpci` (PCIe MB/s) stayed near-zero (single-digit to low tens of MB/s) throughout `A_memcpy16` — the same order of magnitude as baseline — ruling out PCIe-bus-contention (`memcpy_16` is a pure host-RAM operation that never touches the PCIe bus). The remaining explanation is that the memcpy hog, by saturating all 16 logical cores on this machine (confirmed via `nproc`-equivalent), starves llama-server's own CPU-side work (HTTP handling, sampling/dispatch, feeding the GPU), leaving the GPU with nothing to compute — consistent with TTFT 7.3-16.7s vs. 1.8s baseline. **`A_spin16` (pure CPU compute, no host memory-bandwidth traffic) shows a materially different signature**: SM clocks mostly stay near-baseline (2265-2565 MHz) and power stays higher (58-98 W vs. memcpy's 1-25 W), yet decode_tps still drops ~40-47% (30.8-37.0 vs. ~58 tok/s baseline) despite TTFT barely moving (1.9-2.3s vs. 1.8s baseline) — spin appears to compete specifically with per-token CPU-side work (sampling/detokenization loop) rather than blocking the GPU from being fed work at all, a distinct mechanism from memcpy's bandwidth-based starvation. **Part B (thread-count discriminator) is inconclusive, not negative**: `-t 1` (TTFT 5.7-6.4s) and `-t 8` (TTFT 5.4-5.5s) show near-identical, and both notably *better* than `A_memcpy16`'s own `-t 4` run (7.3-16.7s) — but `A_memcpy16` itself shows a strong within-condition warm-up trend (16.7s → 8.3s → 7.3s across its 3 calls) of similar magnitude to the across-condition gap, so this comparison is confounded by run-order/thermal drift and should not be read as "more or fewer server threads helps." The most likely reason `-t1` vs `-t8` shows no real difference: the co-runner already saturates all 16 logical cores regardless of how many the server requests, so there is no spare core for the server to gain by asking for more (or fewer) threads — consistent with, not contradicting, the host-critical-path verdict. **Part C dose-response data quality issue**: `achieved_gbps` (the true positive control, always used instead of the requested target) is noisy and sometimes missing — `C_dose_10gbps` achieved 1.404/8.517/8.174 GB/s against a 10 GB/s target (one call barely loaded the bus at all), `C_dose_15gbps` has two `null` achieved-rate readings (calls 0 and 1) and only one real reading (7.597 GB/s on call 2), and `C_dose_20gbps` consistently undershot at 15.6-16.6 GB/s. Treat Part C as a qualitative confirmation only (higher realized load broadly tracks with worse decode_tps and TTFT) — it is **not** a clean quantitative dose-response curve, and any bandwidth-vs-slowdown regression fit from this table would be fit to badly-conditioned x-values. **`gpu_mem_initial` is an empty `{}` for every row in Part C** (all four dose conditions) — the Windows GPU Process Memory PowerShell counter query returned nothing at server-startup time for these four conditions specifically (Parts A and B all populated correctly with `{'shared': 102760448, 'dedicated': 3928981504}`). This is a real telemetry gap, most likely the counter-query subprocess itself getting starved/timed-out by the same CPU contention Part C was intentionally creating — a plausible but unconfirmed self-referential artifact. Not a crash; the throughput measurements for Part C are unaffected, only the GPU-memory-at-startup field is missing for those four rows. |

| `results/ramlock_evo-t2s_20260925T010739Z.jsonl` (+ manifest, `.DONE`, `results/ramlock_phaseD_telemetry/`) | evo-t2s (Intel Core Ultra X7 358H, Arc B390, Vulkan b10970, unified memory) | **NO** | **Phase D locked-balloon sweep, complete: VALID with stated limits** | AWE-locked balloon (`SeLockMemoryPrivilege` granted), S = 12, 10, 9, 8, 7.5, 7, 6, 5, 4 GB plus a D1 baseline and a D2 smoke at S=7. All levels whose server started ran normally (TTFT at most 1.08x D1, 5/5 correct); S=5 crashed at load with a Vulkan device-lost (exit 0xC0000409) while S=4 ran normally, so the failure is non-monotonic and unrepeated. Server load time rose from 2.6 s to about 150 s at S=6 and S=4. Full table, the file-backed vs private memory decomposition and all limits are in `docs/FINDINGS.md` (Phase D section). Data caveats: balloon CSV timestamps are local time labelled Z; `pages_per_sec` and `pagefile_pct_usage` read exactly 0 after server start at S=6, 5 and 4 (counter failure, values UNKNOWN); no disk-read counter; balloon CSV covers about the first 100 s per level. Stale-server check: 7 request markers in every started level's server log, 0 in the S=5 log. Script: `scripts/as_run/blade_C_apu/ramlock_evo_t2s.py` with `harness/memory_balloon_awe.py` (deployed copy identical to the pre-2b317ef version). |

### Corrections to the M2 entry above (2026-09-25, no reruns, reading rules only)

The M2 row above was written before its telemetry had been tabulated. Read it with these corrections; the tabulated evidence and per-hypothesis verdicts are in `docs/FINDINGS.md` under "Blade host-interference and VRAM-spill leads".

1. **Retracted: "verified `gpu_mem_initial.shared` ~98MB, i.e. no VRAM spill confound at this ctx".** That reading was taken before load and with no co-runner. Shared Usage was NOT captured under any memcpy condition (empty strings on every memcpy row), so the absence of a spill during the memcpy rows is unverified, not confirmed.
2. **Retracted: "shared-power-budget and PCIe-bus-contention NOT supported / ruled out", and the commit title that said so.** Under the rule that a hypothesis is ruled out only when its telemetry was recorded and stayed within 5% of baseline, neither is ruled out: power fell to 16% to 53% of baseline and PCIe rx/tx fell to 3% to 66% of baseline in the memcpy rows. Shared power is not supported as the cause in the -t 1 and -t 8 rows (SM clock within 1% at the same slowdown) but is not ruled out for the -t 4 row (SM clock at 43% of baseline). PCIe has no evidence for it and was not tested with a bus load.
3. **Part C rows are invalid for dose-response use.** The 5 GB/s server was never terminated and served all of the 10, 15 and 20 GB/s calls; per-PID Shared Usage tracked the wrong process. The "noisy achieved GB/s" description in the M2 row understated this: the noise came from a co-runner run against a stale server, not only from pacing error. Achieved-GB/s values remain valid as a co-runner positive control.
3a. `gpu_mem_initial = {}` on all Part C rows is explained by item 3; the earlier "starved counter query" guess in the M2 row is a partial explanation at best. The Part A and B memcpy gpumem CSVs are empty for the starvation reason only.
4. **Retracted: "confirmed via nproc-equivalent" as evidence for the explanation of the -t 1 vs -t 8 result.** The machine has 16 logical processors and the co-runner uses 16 threads, but that explains why the test cannot discriminate; it does not confirm the host-critical-path hypothesis.
5. **Item 3 in the Blade contention corrections above still applies:** Blade ctx=32768 baseline drift makes its slowdown ratios invalid.

### Stale-server exposure audit (2026-09-25)

**Failure mode.** On Windows two processes can bind the same port. A server that was never terminated keeps answering while a newly started server logs "listening" and receives nothing. This invalidated `blade_m2_host_interference` Part C (10, 15 and 20 GB/s doses answered by the 5 GB/s server).

**Standing guard (from 2026-09-25, `harness/server_guard.py`, tests in `tests/test_server_guard.py`).** Every harness that launches llama-server must (1) confirm the port is free before any start and, if not, record the listener PID and command line and STOP; (2) after each start query `/props` and assert model path, n_ctx and slot count, assert KV type, flash-attn and the other launch flags from the process command line (b10970 `/props` does not expose KV type or flash-attn, so those two are asserted from the command line and the record says so), and record the server PID; (3) before every request assert the listening PID equals the started PID, and on mismatch mark the row invalid and stop that condition. The guard was tested read-only against a live server (pass path, wrong n_ctx, wrong KV type, wrong PID, and the port-in-use STOP path all behaved as specified). Applied to `harness/blade_spill_sweep.py` (used by C2 and any C1 rerun) and required in M3.

**Audit method.** A healthy server log contains one `launch_slot_` line per request. Requests answered by a different process leave the new server's log with `launch_slot_` count 0 (M2 Part C: 12 markers in the 5 GB/s log, 0 in the 10, 15 and 20 GB/s logs). Where retained server logs exist, request counts were reconciled against result rows. Where they do not, the verdict rests on the design argument stated, and is labelled as such.

| result file(s) | harness and checks it had | evidence | exposure |
|---|---|---|---|
| `blade_m2_host_interference_20260925T043921Z.jsonl` | none (health poll only; taskkill errors swallowed) | Part A and B logs: 3 markers each, own PID. Part C: 12 / 0 / 0 / 0 markers | **AFFECTED: Part C, 9 rows for 10, 15 and 20 GB/s.** Part A and B not affected. |
| `blade_m1_vram_spill_20260925T041415Z.jsonl` | none | Server logs `m1_srv_*` are no longer on disk, so no request-count audit. Design argument: ctx ascends and every prompt is 90% of ctx, so a stale earlier-tier server (smaller n_ctx) could not accept the next tier's prompt, yet all 15 rows are `status ok` with 128 completion tokens. The ctx 8192 tier could only have been answered by a same-config server. | Not affected by design argument; not log-verified. |
| `contention_blade_rtx4070_20260924T171907Z.jsonl` | none (`contention_v3.py`, fixed port 8385, no listener check) | One server per ctx tier (2 distinct PIDs across 28 rows). Tier logs hold 56 markers each = 14 conditions x (1 warm-up + 3 reps). | Not affected (log-verified). |
| `contention_evo-t2s_20260924T171913Z.jsonl` | none (same script) | One server per ctx tier (PIDs 7256 and 7940). Tier logs not found on evo-t2s, so no request-count audit. The 32768 tier cannot have been answered by the 8192-tier server (29k-token prompts exceed n_ctx 8192). | 32768 tier not affected by design argument; **8192 tier unaudited**. |
| `ramlock_evo-t2s_20260925T010739Z.jsonl` (Phase D) | none (no port check; unique log per level; PID recorded) | Every completed level log holds exactly 7 markers (D1, D2, S12, S10, S9, S8, S7.5). D3_S7 in progress. | Not affected for completed levels (log-verified). Remaining levels to be audited the same way when Phase D ends. |
| `kv_quality_20260923T181840Z_*.jsonl` | `harness/kv_quality_sweep.py`: kills all llama-server by name and sleeps 3 s before each start, then health poll only; no listener, PID or `/props` check | Logs `kv_qual_*.txt` not located, so not audited. The negative before-start memory deltas already recorded in FINDINGS show the previous server was still being torn down when the next one started. | **Exposure cannot be excluded**; per-row server identity not recorded. |
| `kv_provisioning_20260924T003514Z_ctx32768_f16.jsonl` | generating script not found in the repo or in `C:\apu` on this machine | none | **UNKNOWN** |
| `bw_saturation_20260923T045844Z.jsonl`, `bw_saturation_20260923T065604Z.jsonl`, `memory_pressure_20260924T025611Z.jsonl`, `memory_pressure_v2_20260924T034948Z.jsonl` | generating scripts not in the repo | Retained logs on evo-t2s do not reconcile with the rows (`bw_srv_log.txt`: 1 marker for 20 rows across two files; each `mem_pressure_srv_*` log: 2 markers) | **UNKNOWN**. The two memory_pressure files are already flagged invalid above for the balloon defect. |
| `fig61_stagec_full_20260922T191031Z.jsonl`, `..._203557Z.jsonl`, `..._230133Z.jsonl` (Blade), `fig61_stagec_gate_*`, `fig61_stagec_smoke_*`, `fig61_toktrunc_full_*`, `fig61_full_20260922T060942Z.jsonl` | `harness/fig61_stagec_sweep.py`, `harness/fig61_sweep.py`: server started outside the harness on port 8383; only `/props` for build id; no model, ctx, PID or port assertion | Cross-checks that are not a guard: 203557Z matches 191031Z cell for cell, and the Blade run matches evo-t2s in 130 of 132 cells. `fig61_full_20260922T060942Z` is a documented case of the wrong model (hybrid checkpoint) answering undetected. | **No check.** A wrong or stale server on 8383 cannot be excluded from the harness record; the cross-checks make it unlikely for the three `stagec_full` files. |
| results produced through `harness/llama_server.py` (`runner.py`, `stage_a_kv_precision.py`, `gate1_kv_precision.json`) | fixed port per config, `/health` poll only | not audited | **No check; unaudited.** |
| Ollama-based results (`run_*.jsonl`, `stage_c_20260818T040408Z.jsonl`, and similar) | not a llama-server path | n/a | Not applicable to this failure mode. |

**Live jobs.** The C1 sweep (code at commit `1c17f5d`) was already running and was not interrupted. It has a start-time listener check (no listener on 8385 before start, listener PID equals the started PID after health) but no `/props` assertion and no per-request check. C1 is therefore audited after the fact with the same log method (6 markers per server: 1 warm-up + 5 measured) and its `server_start` rows record each PID. Phase D on evo-t2s was likewise not touched.

### Result file to generating script map (2026-09-25)

Rule from now on: no result is generated by a script that is not committed first, and every manifest records the script git SHA (`harness/run_provenance.py`). Before this rule most experiment scripts lived only in `C:\apu` on each machine. Byte-exact copies were committed in df7f4b3 under `scripts/as_run/` (with `-text` in `.gitattributes` so the stored bytes equal the originals). Evo-t2s file times are PDT, the same zone as the Blade. "Version" evidence is by (a) exact hash match to a committed blob, (b) the deployed file's modification time bracketing the run, or (c) a script naming its own output. Nothing below was re-run.

| result file(s) | generating script | version evidence and git SHA | status |
|---|---|---|---|
| `ablation_cha04_20260817.jsonl` | `harness/ablation_cha04.py` | result added in 6034e1a; script last changed in 6034e1a; blob 7c0ac36; differs from HEAD (as of result commit: git show 6034e1a:harness/ablation_cha04.py) | identified by name, mention or co-commit; version taken at the result's commit |
| `art_headroom.json` | `harness/headroom_check.py` | result added in 940eada; script last changed in 940eada; blob 9669385; differs from HEAD (as of result commit: git show 940eada:harness/headroom_check.py) | identified by name, mention or co-commit; version taken at the result's commit |
| `art_truncation.json` | `harness/art_truncation.py` | result added in f2b686f; script last changed in f2b686f; blob 9b54852; differs from HEAD (as of result commit: git show f2b686f:harness/art_truncation.py) | identified by name, mention or co-commit; version taken at the result's commit |
| `art_truncation_analysis.json` | `harness/art_truncation_analysis.py` | result added in f2b686f; script last changed in f2b686f; blob 004ffcc; same as HEAD | identified by name, mention or co-commit; version taken at the result's commit |
| `artifact_ratio_sweep.json` | `harness/artifact_ratio_sweep.py` | result added in 23c1eba; script last changed in 23c1eba; blob 7d8837d; differs from HEAD (as of result commit: git show 23c1eba:harness/artifact_ratio_sweep.py) | identified by name, mention or co-commit; version taken at the result's commit |
| `filler_composition.json` | `harness/filler_composition_sweep.py` | result added in d1633f6; script last changed in d1633f6; blob 503aa43; differs from HEAD (as of result commit: git show d1633f6:harness/filler_composition_sweep.py) | identified by name, mention or co-commit; version taken at the result's commit |
| `interference_r120.json` | `harness/interference_r120.py` | result added in 20a52bb; script last changed in 110893a; blob 38e6187; differs from HEAD (as of result commit: git show 20a52bb:harness/interference_r120.py) | identified by name, mention or co-commit; version taken at the result's commit |
| `model2_truncation.json` | `harness/model2_truncation.py` | result added in 47840bf; script last changed in 47840bf; blob b4eb3e9; differs from HEAD (as of result commit: git show 47840bf:harness/model2_truncation.py) | identified by name, mention or co-commit; version taken at the result's commit |
| `partial_truncation.json` | `harness/stage2_partial_truncation.py` | result added in 41f5988; script last changed in 41f5988; blob 4ec1976; differs from HEAD (as of result commit: git show 41f5988:harness/stage2_partial_truncation.py) | identified by name, mention or co-commit; version taken at the result's commit |
| `schema_collision.json`, `schema_collision.jsonl` | `harness/schema_collision.py` | result added in 722232a; script last changed in 4aa61be; blob 2496b29; differs from HEAD (as of result commit: git show 722232a:harness/schema_collision.py) | identified by name, mention or co-commit; version taken at the result's commit |
| `selfreport_arms.json` | `analysis_selfreport.py` | result added in a509f73; script last changed in a509f73; blob bcae0b1; differs from HEAD (as of result commit: git show a509f73:analysis_selfreport.py) | identified by name, mention or co-commit; version taken at the result's commit |
| `span_ablation.json`, `span_ablation.jsonl`, `span_ablation_err.log`, `span_ablation_run.log` | `harness/span_ablation.py` | result added in a7056d5; script last changed in b78e09d; blob c2bae7d; differs from HEAD (as of result commit: git show a7056d5:harness/span_ablation.py) | identified by name, mention or co-commit; version taken at the result's commit |
| `stage_a_scale.json` | `harness/stage_a_scale.py` | result added in 70d8663; script last changed in 70d8663; blob 5aafabd; differs from HEAD (as of result commit: git show 70d8663:harness/stage_a_scale.py) | identified by name, mention or co-commit; version taken at the result's commit |
| `stage_c_20260818T040408Z.jsonl`, `stage_c_20260818T040408Z_gate2.json`, `position_pressure_analysis.json` | `harness/stage_c_position_pressure.py` | result added in 743e83e; script last changed in 743e83e; blob 3a57980; differs from HEAD (as of result commit: git show 743e83e:harness/stage_c_position_pressure.py) | identified by name, mention or co-commit; version taken at the result's commit |
| `type_match.json` | `harness/type_match_experiment.py` | result added in 23c1eba; script last changed in 23c1eba; blob 22af4db; differs from HEAD (as of result commit: git show 23c1eba:harness/type_match_experiment.py) | identified by name, mention or co-commit; version taken at the result's commit |
| `gate1_kv_precision.json` | `harness/stage_a_kv_precision.py` | result added in 0b67b24; script last changed in 0b67b24; blob 3f1bc2d; differs from HEAD (as of result commit: git show 0b67b24:harness/stage_a_kv_precision.py) | identified by name, mention or co-commit; version taken at the result's commit |
| `fig61_smoke_20260922T060616Z.jsonl`, `fig61_smoke_20260922T060616Z_manifest.json`, `fig61_full_20260922T060942Z.jsonl`, `fig61_full_20260922T060942Z_manifest.json` | `harness/fig61_sweep.py` | result added in 254bbf9; script last changed in 254bbf9; blob 14d31b1; same as HEAD | identified by name, mention or co-commit; version taken at the result's commit |
| `fig61_stagec_smoke_20260922T185743Z.jsonl`, `fig61_stagec_gate_20260922T190239Z.jsonl`, `fig61_stagec_full_20260922T191031Z.jsonl`, `fig61_stagec_manifest_20260922T191031Z.json` | `scripts/as_run/evo_t2s_C_apu/fig61_stagec_sweep.py` | as-run copy added in df7f4b3; sha256 b4331d9787; never committed in this form before. The evo-t2s deployed file was last modified 2026-09-22 10:37 PDT, before these runs (11:57 to 12:10 PDT) | identified by file time; strong. The manifest was backfilled after the run by `scripts/as_run/scratchpad_blade_dev/backfill_manifests.py` |
| `fig61_stagec_smoke_20260922T203405Z.jsonl`, `fig61_stagec_full_20260922T203557Z.jsonl`, `fig61_stagec_manifest_20260922T203557Z.json` | `harness/fig61_stagec_sweep.py` | nearest committed version is 0fd749a (blob ad072e3281). The run (13:34 to 13:35 PDT) preceded that commit (15:44 PDT) and the evo-t2s file was later overwritten by the a8a0140 version, so the exact as-run bytes were not preserved | VERSION UNCERTAIN (not SCRIPT LOST: the nearest committed version is known). Manifest backfilled by `backfill_manifests.py` |
| `fig61_stagec_smoke_20260922T230030Z.jsonl`, `fig61_stagec_gate_20260922T230052Z.jsonl`, `fig61_stagec_full_20260922T230133Z.jsonl` and their manifests (Blade) | `harness/fig61_stagec_sweep.py` | commit 88958cd (blob 9fd7cff061), made about 9 minutes after these runs (16:00 to 16:01 PDT); the manifests were written by the script's own write_manifest | likely exact; not proven |
| `fig61_toktrunc_full_20260922T233232Z.jsonl`, `fig61_toktrunc_manifest_20260922T233232Z.json` | `harness/fig61_stagec_sweep.py` | commit a8a0140 (blob 3381d6dfe6), same minute as the run (16:32 PDT); evo-t2s copy identical | identified; strong |
| `fig61_plot_data.json` | `scripts/as_run/scratchpad_blade_dev/make_plot_data.py` | dev copy added in df7f4b3; sha256 bad27c8c7e; reads `fig61_full_20260922T060942Z.jsonl` | identified by content; not confirmed as the exact file that ran |
| `kv_quality_20260923T181840Z_ctx8192.jsonl`, `..._ctx32768.jsonl`, `..._full.jsonl` | `scripts/as_run/evo_t2s_C_apu/kv_quality_sweep.py` | as-run copy added in df7f4b3; sha256 cd369263a4; deployed file modified 11:07 PDT, run started 11:18 PDT. The committed `harness/kv_quality_sweep.py` (blob 6288954, commit 4c4c009) is a LATER, different version | identified by file time; strong |
| `kv_provisioning_20260924T003514Z_ctx32768_f16.jsonl`, `kv_val_vulkan_20260923.json` | `scripts/as_run/evo_t2s_C_apu/kv_provisioning_sweep.py` | as-run copy added in df7f4b3; sha256 0617ed10bf; deployed file modified 17:30 PDT, provisioning run started 17:35 PDT; `kv_val_vulkan` is attributed by the script naming it, not by time | provisioning file strong; kv_val_vulkan by mention only |
| `bw_saturation_20260923T065604Z.jsonl` | `scripts/as_run/evo_t2s_C_apu/bw_saturation_sweep.py` | as-run copy added in df7f4b3; sha256 e34758d8e0; deployed file modified 23:54 PDT, run started 23:56 PDT (grid reduced to ctx up to 131072 and q8_0/q4_0) | identified by file time; strong |
| `bw_saturation_20260923T045844Z.jsonl` | probable: `scripts/as_run/scratchpad_blade_dev/bw_saturation_sweep.py` | dev copy added in df7f4b3; sha256 205b77a029; differs from the deployed file only in the CTX_SIZES and KV_PRECISIONS constants (full grid, three precisions). The run started 21:58 PDT, before the deployed file was edited at 23:54 PDT, so the exact as-run bytes were overwritten | VERSION UNCERTAIN, probable variant preserved |
| `memory_pressure_v2_20260924T034948Z.jsonl` | `scripts/as_run/evo_t2s_C_apu/memory_balloon.py` and `scripts/as_run/evo_t2s_C_apu/memory_pressure_experiment.py` | as-run copies added in df7f4b3; sha256 79392c4f66 and ecd2702297; deployed files modified 20:47 and 20:49:36 PDT, run started 20:49:48 PDT | identified by file time; strong |
| `memory_pressure_20260924T025611Z.jsonl` (v1) | none | the v1 resizing balloon and its experiment script were overwritten by the v2 versions (evo-t2s files are dated after the v1 run at 19:56 PDT); no copy exists in the repo, C:\apu on either machine, or the dev scratchpad | **SCRIPT LOST** |
| `contention_blade_rtx4070_20260924T171907Z.jsonl` and manifest, `contention_evo-t2s_20260924T171913Z.jsonl` and manifest | `scripts/as_run/blade_C_apu/contention_v3.py` with co-runners `scripts/as_run/blade_C_apu/bandwidth_hog2.py`, `scripts/as_run/blade_C_apu/spin_hog2.py`, `scripts/as_run/blade_C_apu/pytest_hog.py`, `scripts/as_run/blade_C_apu/pandas_groupby_hog.py`, `scripts/as_run/blade_C_apu/compile_proxy_hog.py` | as-run copies added in df7f4b3; contention_v3 sha256 a9ab68290b; identical bytes on the Blade, on evo-t2s and in the dev scratchpad | identified; strong |
| `blade_m1_vram_spill_20260925T041415Z.jsonl`, `m1_telemetry_*.csv` | `scripts/as_run/blade_C_apu/blade_m1_vram_spill.py` | as-run copy added in df7f4b3; sha256 a155ab0301 | identified; strong |
| `blade_m2_host_interference_20260925T043921Z.jsonl`, `m2_dmon_*.txt`, `m2_gpumem_*.csv` | `scripts/as_run/blade_C_apu/blade_m2_host_interference.py` with `scripts/as_run/blade_C_apu/bandwidth_hog_throttled.py`, `scripts/as_run/blade_C_apu/bandwidth_hog2.py`, `scripts/as_run/blade_C_apu/spin_hog2.py` | as-run copies added in df7f4b3; sha256 7977d78da1 | identified; strong. Table script: `analysis/blade_m2_evidence_table.py` (commit 4841bf5) |
| `ramlock_evo-t2s_20260925T010739Z.jsonl` (Phase D, in progress) | `scripts/as_run/blade_C_apu/ramlock_evo_t2s.py` and `harness/memory_balloon_awe.py` | as-run copy added in df7f4b3; sha256 da92b7daa2; identical to the file running on evo-t2s; balloon module blob 1350f9f identical to the deployed copy | identified; strong |
| `blade_c1_spill_sweep_20260925T053651Z*` (C1, in progress) | `harness/blade_spill_sweep.py`, `harness/blade_telemetry.py` at commit 1c17f5d | the manifest does not record the SHA (the recording was added afterwards); `harness/blade_telemetry.py`, `harness/context.py` and `evaluation/probes/scorers.py` are unchanged since 1c17f5d, and `blade_spill_sweep.py` was edited after launch | identified; strong |
| `crossmodel_baseline.json` | none | added in commit 0b67b24; no script that names it exists in the repo, `C:\apu`, `C:\apu\APU` or the dev scratchpad | **SCRIPT LOST** |
| `token_aligned_rerun.json` | none | added in commit 574b1c0 with no script; nothing names it | **SCRIPT LOST** |
| `artifact_deletion_check.json` | none | added in commit cd9fc11 with no script; nothing names it | **SCRIPT LOST** |
| `llamaserver_feasibility.json` | none | added in commit 7efacd1 with no script; only analysis code reads it | **SCRIPT LOST** |
| `zachary/replication_remote_search_v3.json` | none | produced by Zachary's harness outside this repository; not ours to recover here | **SCRIPT LOST** |

Not needed for any committed result and therefore not preserved: `contention_experiment.py` (superseded, its run was never saved), `balloon_correctness_rerun.py` (empty on evo-t2s), the various measurement, probe and PowerShell monitoring helpers in the dev scratchpad.

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
