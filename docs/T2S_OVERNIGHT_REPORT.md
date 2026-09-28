# evo-t2s overnight run and A-mech (2026-09-26)

**Hardware arm:** evo-t2s only (`Intel(R) Core(TM) Ultra X7 358H`, Arc B390 iGPU, 63.49 GB unified memory, Windows 11, Vulkan llama-server b10970, driver 32.0.101.8509). Never pooled with the Blade (discrete RTX 4070, CUDA).
**CPU family note:** the CPU string recorded in every row is `Intel(R) Core(TM) Ultra X7 358H`. That is a Panther Lake part, not Arrow Lake. Any slide or text that says Arrow Lake for evo-t2s is wrong and should say Panther Lake.
**Sources:** `results/t2s_overnight_20260926T011744Z.jsonl` (+ manifest, model table, schedule, telemetry, server logs) and `results/t2s_amech_20260926T181456Z.jsonl` (+ manifest, vulkaninfo, server logs). Tables come from `analysis/t2s_overnight_analysis.py`, `analysis/t2s_overnight_audit.py` and `analysis/t2s_amech_analysis.py`, all committed before they were run on the final data. Script versions per row are in `docs/RESULT_PROVENANCE.md`.

---

## Brutal self-review (read this first)

### What is new, and what is not

- **Not new:** a server that asks for more memory than the device has fails to start. Every allocator does that. The failure itself is not a finding.
- **New on this device (measured here, start-only):** the operative limit is the Vulkan memory heap *budget* (47,866 MiB, from `vulkaninfo`), not the heap *size* (37,060 MiB) that llama.cpp prints as the device total. A 32B server allocated 39.8 GB and ran, so 37,060 is not a limit. The measured boundary sits within 0.5% of the budget for four models from two families (Section A-mech), so it is set by total memory, not by the model or by a single-allocation limit (`maxMemoryAllocationSize` is 4,096 MiB and the refused buffer was 964 MiB).
- **Not new, and a hostile reviewer will say so:** llama.cpp's own `--fit` step (default on) projects device memory and can shrink `-ngl`. Our runtime-policy arm (below) measures that documented feature on one iGPU. The contribution is the measurement and the decomposition (memory arithmetic, budget, policy), not the mechanism.
- **Replication, not novelty:** the CPU co-runner slowdown (1.40x to 1.43x TTFT) reproduces the M3 result (1.42x) on two more model sizes.

### What confounds each result

| result | confound or hole |
|---|---|
| A: 32B ladder | Every 32B point above 32768 used YaRN (`--rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 32768`). YaRN alone changed one probe answer (art_03: 8.9 without, 33.6 with, same 14,747-token prompt). The probe result at 111104 (4 of 5) is therefore a YaRN effect, not a memory effect. Calls used prompts of 4,527 tokens (a 60 s call cap), so the KV cache was allocated but only about 4% filled. One start per failing config. |
| A-mech | Start-only: proves what starts, says nothing about speed or correctness near the boundary. `projected_mib` is llama.cpp's own estimate and differs from the logged buffers by the host-side buffers (about 545 MiB). Spread between models is 295 MiB (0.6%), so the boundary is close to, not exactly at, one number. |
| B: co-runner | The thermal gate is a RAPL package-power proxy only (Sysman exposes no temperature sensor). In every co-runner condition the proxy could not be met and the gate released by its 120 s timeout (48 of the 72 Section B call rows, plus 2 in Section C; 50 of 151 call rows in all), so each co-runner measurement began after at least 120 s of load and the conditions had no thermal control. The `none` baseline sat at 18.20 s and 20.16 s with a tight spread in both models, and no drift was seen, but thermal throttling is **not ruled out**: no temperature was recorded. The PROCTHROTTLEMAX causal arm was not run (positive control failed in smoke). |
| C: memory lock | One cell per level, no repeats (the repeats were trimmed). Only one anchor (+8 GB) per model. The 32B has one cell (mmap off, zero headroom). The mmap-off failure at -1 GB is a single observation. |
| D: SYCL | Not run (trimmed). |

### Does it hold on at least 3 model sizes and 2 families?

| claim | sizes and families | verdict |
|---|---|---|
| boundary set by total memory (start-only) | 8B and 32B Qwen3 dense, 30B-A3B Qwen3 MoE, 8B Llama-3.1 | **Yes** (3 sizes, 2 families) |
| no slowdown below budget, loud failure above (with calls) | 32B only | **No.** One model. |
| co-runner slowdown | 4B and 8B, both Qwen3 | **No.** Two sizes, one family. |
| memory lock, mmap arm | 8B (5 cells), 32B (1 cell) | **No.** Not repeated. |
| runtime-policy arm | 32B only | **No.** One model. |

### What a hostile ISPASS reviewer will say

1. "The failure is an out-of-memory error; the boundary is the driver's budget. What is the systems insight?" The insight is only that the number to plan against is the budget, not the reported device total, and that the default runtime policy converts the crash into slower operation. Both are single-platform observations.
2. "Prompt fills of 4.5K tokens on a 110K context measure nothing about full-KV behavior." Correct. A 90% fill of 111,104 tokens is estimated at about 3.6 hours per call from our own quadratic prefill fit (see the fill feasibility record), so the filled checks below use a capped fill and say so.
3. "n=1 per failing config, one machine, one driver." Correct. The boundary flips were repeated 3 to 4 times for the 32B and are stable; nothing else was repeated.
4. "YaRN confounds the whole 32B ladder." Correct, and shown directly by the YaRN check (5 of 5 without, 4 of 5 with).
5. "Your power-coupling mechanism is a story." Partly: package power reads 44.9 W in the P-core-only condition too, where the iGPU is not throttled, so package power alone does not predict the slowdown (Pearson r = -0.53 across five conditions in each model). The causal knob failed its positive control.
6. "Thermal throttling is not excluded." Correct, see B.

### Failures, trims and invalid rows

- **First launch stopped after Section 0** with `TypeError: Lab.row() got multiple values for keyword argument 'load_mode'` (a duplicate keyword in the mmap control). Fixed and resumed. The file therefore contains two `schedule` records and two `run_end` records (the first is that error, the second is `completed`).
- **Three code versions** produced the rows: 54 Section 0 rows at git 328a0d7, 5 rows at ba56eae, 192 rows at 2819d83 (all Section A, B and C rows). Every row carries its own `git_sha` and `script_sha`.
- **Invalid rows (`valid` false), 6:** five Section A starts (117,248 to 125,440, expected loud failures, listed below) and one Section C start (8B, mmap off, -1 GB: device lost). No item error, no invalid call row, no `n_reduced` call.
- **Schedule trims:** the second plan kept 19 of 208 items (8.12 h planned of 8.24 h) and trimmed 189: Section A for 4B, 8B and 30B-A3B and 12 of the 32B grid points, all of Section C for the 14B and 32B except one cell, 35 of 40 8B cells, one Section B group each for Llama, 14B, 30B-A3B and 32B, and all of Section D. The run finished about 3 hours before its deadline because the estimates were high; the planner had no backfill (added afterwards, commit 272cb43, see the last section).
- **Bisection first attempt (A-mech):** the stale-server guard rejected 8B starts above n_ctx 131072 and 30B-A3B starts above 262144 (the server reports a capped `/props` n_ctx although the full KV buffer was allocated). Those 34 start rows are superseded by a rerun that counts a created context as a start; they stay in the file.
- **Arm (b) first attempt (A-mech):** the guard rejected a server that had started with 62 of 65 layers on the GPU because it expected `-ngl 99` on the command line. Fixed (commit 1a75ac6) and rerun under new item ids; the rejected rows stay in the file.

### Unsupported or downgraded claims

- **Mechanism: OPEN** (updated 2026-09-27, do not call this a shared power budget effect): p4 reaches the same 44.9 W package power as nonp12 but keeps the iGPU at 2,450 to 2,500 MHz and costs only 1.04x to 1.06x, so package power at the cap is not sufficient on its own; the causal knob had no effect; no temperature was recorded.
- **UNVERIFIED:** "mmap-off crashes where mmap-on survives at -1 GB headroom." One start each.
- **Not established:** correctness near the memory boundary (probes ran at 4.5K to 14.7K tokens, never at a full context).
- **Doc error found, not fixed here:** `docs/THREATS.md` section 11 describes the EVO-T2S as a discrete-memory system. evo-t2s is a unified-memory iGPU machine. It needs correcting by the author of that section.
- **Cross-dataset join:** Zachary's `replication_remote_search_v3.json` comes from a different machine (Ultra 5 325, 8 logical cores, WSL2), so no per-call join to these rows is possible (see `docs/JOIN_KEY.md`).

---

## Summary tables (one screen per section)

### Section 0: smoke and positive controls (all passed)

| check | result |
|---|---|
| six models load, probes, no `<think>` | all six load; probes 5/5 (30B-A3B 4/5); KV bytes per token from logs equal GGUF metadata |
| paging positive control (strict: 4 GB available, random 2 GB read) | passed 3 times: 321,784, 458,411 and 302,734 pages input/s |
| mmap control (8B) | `--load-mode mmap`: private 5,916 MiB, working set 10,297 MiB; `none` and `auto`: 6,243 and 6,178 MiB. `auto` behaves like `none` on this device, so all earlier runs used non-file-backed weights |
| PROCTHROTTLEMAX control (smoke) | **failed**: spin package power 44.93 W at cap 50, 44.94 W at cap 100; causal arm disabled |
| temperature sensor (Sysman) | none exposed (count 0); ACPI thermal zone constant 28 C |

### Section A: crossing the iGPU memory budget (32B, YaRN flags, n_ctx above 32768)

| n_ctx | llama.cpp projected MiB (need MiB, our estimate) | starts ok | TTFT s median [IQR] (n) | decode tok/s | max Shared Usage MiB |
|---|---|---|---|---|---|
| 65,792 (4 anchors) | (35,511) | 5/5 | 31.97 [31.75, 32.09] (20) | 4.6 | 35,751 |
| 111,104 | 46,494 (46,839) | 1/1 | 31.88 [31.87, 31.88] (5) | 4.6 | 47,190 |
| 117,248 | 48,036 (48,375) | 0/1 | none | none | none |
| 119,296 | (48,887) | 0/1 | none | none | none |
| 121,344 | (49,399) | 0/1 | none | none | none |
| 123,392 | 49,578 (49,911) | 0/1 | none | none | none |
| 125,440 | (50,423) | 0/1 | none | none | none |

Slowdown of 111,104 against the 65,792 anchors: 1.00 [0.99, 1.00] (bootstrap 95%). Prompt was 4,527 tokens at every point (fill capped). B = 47,865 MiB (llama-server free-memory figure; the Vulkan heap budget is 47,866). Below budget nothing slowed and nothing degraded silently; above budget the server did not start. There was no slow or wrong-answer regime in between at this resolution (the bisection below refines the boundary to 256 tokens).

**YaRN check (5 art probes, 16,384 context, 14,747-token prompt):** without YaRN 5/5 correct; with YaRN 4/5 (art_03 answers 33.6 instead of 8.9). At 111,104 with YaRN and a 4,532-token prompt art_03 again answers 33.6. YaRN alone changes that answer, so the 4/5 at 111,104 is not a memory effect.

### Section B: CPU co-runner coupling (context 8,192, 7,368-token prompt, 5 measured calls per condition, `none` n=10)

| model | co-runner | TTFT s | slowdown [boot 95%] | decode slowdown [boot 95%] | iGPU MHz | package W |
|---|---|---|---|---|---|---|
| 4B-2507 | none | 18.20 | 1.00 | 1.00 | 2500 | 22.8 |
| | e4 (4 E-cores) | 18.53 | 1.02 [1.01, 1.02] | 1.00 [0.99, 1.00] | 2500 | 39.1 |
| | p4 (4 P-cores) | 18.86 | 1.04 [1.03, 1.04] | 1.00 [0.99, 1.00] | 2500 | 44.9 |
| | nonp12 | 25.43 | 1.40 [1.33, 1.44] | 1.24 [1.18, 1.24] | 1650 | 44.9 |
| | all16 | 25.32 | 1.39 [1.39, 1.44] | 1.24 [1.24, 1.24] | 1650 | 44.9 |
| 8B | none | 20.16 | 1.00 | 1.00 | 2500 | 25.8 |
| | e4 | 21.21 | 1.05 [1.05, 1.05] | 1.00 [1.00, 1.00] | 2500 | 41.1 |
| | p4 | 21.28 | 1.06 [1.05, 1.06] | 1.01 [1.01, 1.01] | 2450 | 44.9 |
| | nonp12 | 28.65 | 1.42 [1.40, 1.44] | 1.20 [1.19, 1.21] | 1600 | 44.9 |
| | all16 | 28.75 | 1.43 [1.40, 1.45] | 1.20 [1.20, 1.21] | 1600 | 44.9 |

Reading: (1) the 12-core and 16-core co-runners cost 1.39x to 1.43x TTFT and 1.20x to 1.24x decode on both sizes and reproduce the M3 figure (1.42x). (2) The iGPU falls from 2500 to 1600 to 1650 MHz in exactly those conditions. (3) The P-core-only co-runner also drives package power to 44.9 W but the iGPU stays at 2450 to 2500 MHz and TTFT rises only 1.04x to 1.06x, so package power at the cap is necessary at most, not sufficient. (4) Package power is not a good predictor across conditions (Pearson r = -0.53, n = 5 per model). (5) The PROCTHROTTLEMAX causal arm was not run. Thermal throttling is not excluded.

### Section C: memory lock and the mmap arm (8B, context 16,384, 12,588-token prompt; 32B, 4,527-token prompt)

Headroom is free memory after the lock minus (weights + KV + compute) in GiB; 0 means the lock leaves exactly what the server needs. One cell per level; every level marked valid (lock held, balloon alive).

| model | load mode | headroom | load s | TTFT s (n=5) | decode tok/s | probes | max pages in/s | Available min MB |
|---|---|---|---|---|---|---|---|---|
| 8B | mmap | +8 (anchor) | 5.6 | 55.7 | 9.33 | | 960 | 3,702 |
| 8B | mmap | 0 | 19.9 | 56.5 | 8.67 | 5/5 | 1,152 | 113 |
| 8B | mmap | -1 | 23.0 | 56.7 | 8.66 | 5/5 | 56,571 | 172 |
| 8B | none | 0 | 7.7 | 56.8 | 8.54 | 5/5 | 25,082 | 50 |
| 8B | none | -1 | **failed at 1,115 s: Vulkan device lost (0xc0000409)** | none | none | none | none | 1 |
| 32B | none | 0 | 18.9 | 32.8 | 4.45 | 5/5 | 26,733 | 3 |

Reading: at zero and negative headroom nothing slowed in TTFT beyond +1.4% to +2.0% of the anchor and correctness held, but decode fell 7% to 8.5% (9.33 to 8.54 to 8.67 tok/s), which is outside the 5% "ruled out" band, so a small decode cost is **not ruled out** (one anchor, no repeats). The 32B at zero headroom with only 3 MB left available started in 18.9 s and answered 5/5 probes at 32.8 s TTFT (1.03x of the 31.97 s median of the Section A no-pressure anchors). The only failure is the mmap-off 8B at -1 GB, which stalled for 18.6 minutes at load and lost the device, while the same level with mmap on loaded in 23 s. One observation each; it does not establish that mmap prevents the crash.

### Section D: SYCL

Not run (trimmed by the planner; no `D_prepare`).

---

## A-mech: why the 32B fails above the budget, and where the boundary is

### Step 0: rope scaling is YaRN, not linear

- The failing 32B log prints `print_info: rope scaling = linear` and `llama_context: freq_scale = 0.25`. The first line is the model file's trained setting, printed before the context is built; the no-YaRN arm of the YaRN check prints the same line with `freq_scale = 1`, and the YaRN arm prints `freq_scale = 0.25`. llama-server does not print the context's rope type, so **the log cannot confirm yarn or linear by itself**.
- Command line of every overnight 32B Section A point and of the YaRN check "on" arm (reconstructed from `Server._cmd()` at git 2819d83 plus the `rope_flags` recorded in each row; the overnight rows did not store the command line verbatim): `C:\apu\bin\llama-b10970\llama-server.exe -m C:\apu\models\Qwen3-32B-Q4_K_M.gguf --port 8385 -c <n_ctx> -ctk f16 -ctv f16 -fa on -ngl 99 -np 1 -t 4 --no-context-shift --log-file <path> --log-verbosity 4 --rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 32768`. The stale-server guard asserted `-ctk`, `-ctv`, `-fa`, `-ngl`, `-np`, `-t`, `-c` and the model path on the live process before every condition.
- **Behavioural test (A-mech rope phase, 32B, 4,096 context, one 1,502-token prompt, greedy):** top-5 first-token log-probabilities differ between yarn and linear at the same `freq_scale` 0.25 by up to 0.448 nats, between yarn and no scaling by up to 0.214, and between linear and no scaling by up to 0.395. The generated text of the yarn arm equals the no-scaling text, and the linear arm's text is garbled. The process command line of the yarn arm contains `--rope-scaling yarn`. The overnight flags therefore produced YaRN behavior, not linear scaling, and no 32B row needs relabelling. Data: `kind: rope_probe` and `record: rope_verdict` rows.

### Steps 1 and 2: which layer refused, and the device limits

First failing configuration in time: 32B, n_ctx 123,392 (`A_qwen3-32b_grid_123392_4`). Full log, load to exit (metadata dump, tokenizer and `print_info` lines elided; every memory, fit, `llama_context` and Vulkan line kept):

```
0.00.046.238 I cmn  common_param:   - Vulkan0 : Intel(R) Arc(TM) B390 GPU (37060 MiB, 47865 MiB free)
0.00.046.244 I cmn  common_param:   - CPU     : Intel(R) Core(TM) Ultra X7 358H (65018 MiB, 60923 MiB free)
0.00.055.238 I cmn  common_init_: fitting params to device memory ...
0.00.258.971 I common_memory_breakdown_print: | memory breakdown [MiB]  | total    free     self   model   context   compute    unaccounted |
0.00.258.979 I common_memory_breakdown_print: |   - Vulkan0 (Intel(R) Arc(TM) B390 GPU) | 37060 = 47864 + (49578 = 18423 + 30848 + 306) + -60382 |
0.00.258.979 I common_memory_breakdown_print: |   - Host | 557 = 417 + 0 + 140 |
0.00.273.701 I common_params_fit_impl: projected to use 49578 MiB of device memory vs. 47864 MiB of free device memory
0.00.273.711 I common_params_fit_impl: cannot meet free memory target of 1024 MiB, need to reduce device memory by 2737 MiB
0.00.273.712 I common_params_fit_impl: context size set by user to 123392 -> no change
0.00.276.036 W common_fit_params: failed to fit params to free device memory: n_gpu_layers already set by user to 99, abort
0.00.347.552 I llama_prepare_model_devices: using device Vulkan0 (Intel(R) Arc(TM) B390 GPU) (unknown id) - 47865 MiB free
0.00.429.471 I load_tensors: loading model tensors, this can take a while... (load_mode = none)
0.01.116.487 I load_tensors: offloaded 65/65 layers to GPU
0.01.116.491 I load_tensors:      Vulkan0 model buffer size = 18423.65 MiB
0.01.116.492 I load_tensors:  Vulkan_Host model buffer size =   417.30 MiB
0.03.979.873 I llama_context: constructing llama_context
0.03.979.882 I llama_context: n_ctx                 = 123392  (n_ctx_seq = 123392, n_batch 2048, n_ubatch 512)
0.03.979.884 I llama_context: flash_attn            = enabled  (kv_unified = false)
0.03.979.888 I llama_context: freq_base             = 1000000.0
0.03.979.890 I llama_context: freq_scale            = 0.25
0.03.979.892 W llama_context: n_ctx_seq (123392) > n_ctx_train (40960) -- possible training context overflow
0.03.980.368 I llama_context: Vulkan_Host  output buffer size =     0.58 MiB
0.09.315.880 E alloc_tensor_range: failed to allocate Vulkan0 buffer of size 1010827264
0.11.629.176 E llama_init_from_model: failed to initialize the context: failed to allocate buffer for kv cache
0.11.629.189 E cmn  common_init_: failed to create context with model 'C:\apu\models\Qwen3-32B-Q4_K_M.gguf'
0.11.629.194 E srv    load_model: failed to create_context with model 'C:\apu\models\Qwen3-32B-Q4_K_M.gguf'
0.11.629.397 I srv    operator(): operator(): cleaning up before exit...
0.11.631.549 E srv  llama_server: exiting due to model loading error
```

The Vulkan result is on stderr and is in the `.stdout.txt` capture, not in the `--log-file`: `ggml_vulkan: vk::Device::allocateMemory: ErrorOutOfDeviceMemory`. Process exit code 1.

**Which layer refused.** Two layers looked at the same arithmetic and only the second one refused:

1. **llama.cpp's memory pre-check (`common_params_fit_impl`) projected the overflow** (49,578 MiB against 47,864 MiB free, 2,737 MiB over its 1,024 MiB margin) **and did nothing**, because `-ngl 99` had been pinned on the command line ("n_gpu_layers already set by user to 99, abort") and `-c` was set by the user ("context size set by user -> no change"). It warned and continued.
2. **The Vulkan driver refused:** `vkAllocateMemory` returned `ErrorOutOfDeviceMemory` while llama.cpp was allocating the KV cache. ggml then reported `alloc_tensor_range: failed to allocate Vulkan0 buffer of size 1010827264` (964 MiB, two layers of K and V) and the context was not created. About 5.3 s of successful KV allocation preceded the failure (0.03.98 to 0.09.32), so the earlier chunks fit and this one did not.

**Device limits (`vulkaninfo`, Intel Arc B390, API 1.4.335, driver 101.8509):**

| limit | value |
|---|---|
| `maxMemoryAllocationSize` | 0xffff0000 = 4,096 MiB (4 GiB minus 64 KiB) |
| `maxBufferSize` | 0x100000000 = 4,096 MiB |
| memory heaps | one: `size` 38,860,499,189 B (37,060 MiB, 36.19 GiB), `budget` 50,191,138,816 B (47,866 MiB, 46.74 GiB), flags `MEMORY_HEAP_DEVICE_LOCAL_BIT` |
| memory types | four, all on heap 0 |

**Single allocation or total budget?** The refused buffer (964 MiB) is 4.3 times smaller than the single-allocation limit (4,096 MiB), and the 18,423 MiB model buffer is itself larger than that limit, so ggml necessarily splits it into allocations below the limit and those succeeded; this is **not** a single-allocation limit. It matches the **total budget**: the heap budget is 47,866 MiB, and the server's own free-memory figure is 47,865 MiB. The 37,060 MiB heap `size` is not the limit: it was exceeded by successful runs (32B at 111,104 used 47,190 MiB of GPU Shared Usage).

### Steps 3 and 4: the boundary by bisection (server start only, no prompts)

Search: exponential then bisection on n_ctx in steps of 256 (the server pads to 256), so the resolution is the KV size of 256 tokens: 64 MiB for the 32B, 36 MiB for the Qwen3 8B, 24 MiB for the 30B-A3B and 32 MiB for Llama-3.1-8B. "Projected" is llama.cpp's own device-memory projection printed at start (`projected to use N MiB`). B = 47,865 MiB.

| model | rope flags | last passing n_ctx | projected MiB | first failing n_ctx | projected MiB | last passing / B |
|---|---|---|---|---|---|---|
| Qwen3-32B | YaRN factor 4 | 115,712 | 47,650 | 115,968 | 47,714 | 0.9955 |
| Qwen3-8B | YaRN factor 4 | 305,152 | 47,753 | 305,408 | 47,789 | 0.9977 |
| Qwen3-30B-A3B-2507 (MoE) | none | 320,256 | 47,945 | 320,512 | 47,969 | 1.0017 |
| Llama-3.1-8B | none | 343,808 | 47,815 | 344,064 | 47,847 | 0.9990 |

All four flip within 47,650 to 47,969 MiB of projected memory, while their contexts differ by a factor of 3 (115,712 to 343,808 tokens) and their weights by a factor of 4 (4.7 to 18.4 GiB). The boundary follows total required memory, not the model. The 32B boundary configs were started 3 to 4 times each and gave the same answer every time (115,712 passed 4 of 4; 115,968 failed 3 of 3). All 9 genuine failing starts of the rerun show `ErrorOutOfDeviceMemory` in their stderr capture. Caveats: the 30B-A3B passes at 47,945 MiB, slightly above B, so llama.cpp's projection is an estimate (spread between models 295 MiB, 0.6% of B); all boundary contexts are beyond the trained context (the 8B, 30B-A3B and Llama runs are memory tests, not usable contexts); the 8B (YaRN, cap 131,072 in `/props`) and Llama (native cap 131,072) allocated the full requested KV although `/props` reports a smaller n_ctx.

**The fit margin is conservative by about 1 GiB:** the llama.cpp fit target is B minus 1,024 MiB = 46,841 MiB, but every model started at least 800 MiB above it. On this device `--fit` (default on) therefore acts before the real boundary.

### Step 5: Intel Shared GPU Memory Override

Read only. No `HKLM\SOFTWARE\Intel\GMM` key; the `IntelGraphicsSoftware` tree has no Shared, Memory, Override or Segment values; the Arc B390 display-class key has `IncreaseFixedSegment = 0` and `GMM_Debug_Lvl = 1` (present before this work, not touched). No override is set. Nothing was changed; the override is to be raised with Zach before any change.

@@ARMS@@

---

## Planner backfill (commit 272cb43, test `tests/test_planner_backfill.py`)

When work finishes ahead of its estimates, trimmed items are pulled back in by priority until the deadline. After every item the queue's estimated remaining time is compared with the time left; trimmed items are pulled in priority order while they fit (a SYCL cell only with `D_prepare`). Plan-only test on evo-t2s using last night's model table, an 11 h deadline and 45 min reserve, simulating actual time at 0.6x the estimate: 29 items kept and 185 trimmed at plan time; the simulation logs 27 backfill events and runs 59 items by 10.16 h of a 10.25 h budget (155 still trimmed). Unit tests: on-time work pulls nothing; at half the estimate the highest-priority trimmed items are pulled first and the run ends within budget; a SYCL cell is not pulled without `D_prepare`. The backfill is applied in the overnight queue loop; it was not exercised in a real run.
