# CC: context-ceiling cells (registration)

Registered Fri Oct 23 2026, about 11 AM ET (Task BJ part C), before any CC cell runs. Mac only, no
ssh, $0. Approval: Zach's Task BJ brief (TASK_BJ_REGISTRY_REBUILD.md, priority override about 10:45 AM
ET: "measured evidence for 'Not the KV cache: memory tiers set the context ceiling' on TWO devices").
Launching any cell still needs Zach's go (launch command in the Task BJ report). Runners do not read
this file. Predictions: derived/cc/CC_PREDICTIONS.json (DERIVED; tools/cc_predict.py;
tests/test_cc_predict.py). Paper plan: ~/seam_closeout_1013/research_20261021/
Memory_hierarchy_thesis_and_paper_strategy.md ("strategy report"), findings F1-F10.

This is a separate experiment track. It changes nothing in DEMO-2 or CTXKV-E1E2.

## 0. Summary
| Cell | Machine | Window | Status |
|---|---|---|---|
| CC-PRECHECK | Mac | done in part now (section 1); rest Mon Oct 26 | PARTIAL |
| CC-REH-{T2S,XPS} | each host | first cell of each CC queue | REGISTERED |
| CC-BYTES-XPS, CC-MECH-XPS, CC-CLIFF-XPS | XPS | this weekend (XPS held-out deferred) | REGISTERED |
| CC-MECH-T2S (= CC-BYTES-T2S, CC-CEIL-T2S data) | T2S | after the T2S held-out, to Sun 8:44 PM | REGISTERED |
| CC-CLIFF-T2S | T2S | not this weekend (section 3.5) | BLOCKED |
| CC-CEIL-XPS | XPS | analysis of CC-BYTES-XPS + CC-CLIFF-XPS data | REGISTERED |
| CC-BUDGET-XPS | XPS | after Oct 27 (Shared GPU Memory Override, reboot per setting) | REGISTERED, not scheduled |
| CC-XRT-T2S | T2S | only with Rithwik's written OK to install a pinned llama.cpp | CONDITIONAL |
| CC-XRT-XPS | XPS | after Oct 27 | REGISTERED, not scheduled |
| E2-T2S (CTXKV) | T2S | only with Rithwik's written OK | CONDITIONAL |
| E1-XPS (CTXKV) | XPS | after Oct 27 (Ollama install, version pinned) | REGISTERED in CTXKV |
| CC-AMD-*, CC-4070-SPILL, CC-AMD-ZEROCOPY | Rithwik's machines | his schedule | PROPOSED |

## 1. CC-PRECHECK (Mac, read-only): what the sealed limit runs actually ran
Done in this task from the committed records (derived/delta_prefill/*, derived/c2_ttft/*):
- **KV precision.** In every probe of the XPS resident limits 553b3a5c (f16), cf3555d5 (u8) and
  e4a22dac (u4), the GPU plugin's KV_CACHE_PRECISION read back after compile equals the request
  (`kv_cache_precision_readback`: requested f16/u8/u4, normalized f16/u8/u4, match true). This also
  holds for the T2S TTFT-bound limits 5c714535 and 051d2681 (f16) and d2d5cc4a and 56c116a6 (u8).
  - The compiled PA type (GenAI's load-log dump, seam/genai_runtime.py) was not captured in those
    runs. The property was set explicitly, and auto-u4 is off on 2026.2.1 by source (genai_runtime
    docstring). So "the f16 arm ran another precision" is unlikely but not excluded.
  - Every CC cell now logs the compiled type (`--compiled-readback`).
- **The 8-bit arm is u8, not i8.** The sealed 20,000 is u8 (cf3555d5). The DEMO-2 true default
  compiles i8 (a5d19dcc, 18,000, plain pipeline). CC cells use f16 / u8 / u4 to match the sealed arms.
  An i8 arm would need a new config arm; it is not added.
- **Where the sealed limits failed.** All three XPS limit probes failed in TURN 1, the n-token
  prefill: 15,500 f16 (CL_OUT_OF_RESOURCES, then a 1,800 s hang), 20,500 u8 and 27,000 u4
  (`turn1:RuntimeError`, CL_OUT_OF_RESOURCES).
  - Turn 2 appends only 183 tokens (tools/boot4_session.DELTA_TOKENS).
  - So the strategy report's candidate mechanism cannot be what binds in these probes: a MIXED-step
    fp32 scratch of 64 x T bytes per context token, with T = agent-turn tokens, gives T = 183 ->
    about 11.7 KB per token here, not 95-126 KiB.
  - The non-KV term must sit in the uncapped turn-1 prefill (one step of n tokens under the latency
    scheduler).
  - **Wording gate (F1), until CC-MECH reports:** say "a context-proportional non-KV term in the
    uncapped prefill step", not "MIXED-step attention scratch".
- **T2S limits are a different kind.** 5c714535 / 051d2681 / d2d5cc4a / 56c116a6 are
  `c2_ttft_bound_limit` runs (TTFT at the 10 s SLO; 5c714535 first fail 18,875 at prefill 10.08 s),
  not resident-memory limits. No resident-memory limit exists for the T2S.
- **Left for Mon Oct 26:**
  - an OpenVINO 2026.2.1 tag diff of the PA prefill path (which buffers scale with step tokens);
  - the run_id behind "XPS silent paging at 0 MB available, up to 44,742 tokens" (strategy report
    F6). Not found in this pass, so F6 is not citable until it is found;
  - whether the T2S's 48,634 MB GPU-visible figure is the T2S's.

## 2. Settings shared by every CC cell (logged per call)
- Runner tools/run_resident_limit.py; arms gpu_only_f16 / gpu_only_u8 / gpu_only_u4
  (configs/delta_n.yaml, explicit KV_CACHE_PRECISION); model configs/models/Qwen3-4B-int4-ov.yaml.
- New options, all default off (tests/test_cc_runner.py):
  - `--cc-cell` id, required with any other CC option;
  - `--low` / `--resolution` (bisection grid);
  - `--points` (fixed points, each attempted once, in order);
  - `--turn2-new-tokens` / `--turn2-repeats`;
  - `--gpu-mem-stats`;
  - `--compiled-readback`.
- **Step cap.** The step cap is `--sched-max-batched-tokens N`: an explicit SchedulerConfig with
  prefix caching on and max_num_batched_tokens N. "unl" is the plain LLMPipeline (latency
  scheduler, unlimited step), as in the sealed runs. "unlx" is an explicit SchedulerConfig with N =
  2^64 - 1, a control for "the scheduler object itself changes the limit".
  - CHUNK-CAP-LEAP's "never use 256" was a DEMO TTFT rule. Here 256 is a mechanism arm, not a
    deployment setting.
- **Logged per probe (child, outside the timed generate), at load, after turn 1 and after each turn 2:**
  - GPU_MEMORY_STATISTICS read through the load's Core. Whether it sees GenAI's allocations is
    UNVERIFIED; the rehearsal shows it.
  - the GPU process Shared/Dedicated Usage counters for the child's pid (seam/turn_memory);
  - system memory (available, rss, used);
  - the compiled backend and KV type from the load log;
  - the KV property readback (as before);
  - the scheduler config;
  - turn-1 TTFT, turn-2 TTFT, turn-2 wall and tokens (decode tok/s = (tokens - 1) / (wall - TTFT)).
- **Logged per cell (host sidecar tools/cc_mem_sidecar.ps1, about 1 Hz, CSV outside the repo):**
  - GPU Adapter Memory Shared/Dedicated Usage;
  - GPU Process Memory Shared Usage per pid;
  - Available MBytes, Page Reads/sec (hard-fault reads), Pages Input/sec, page file % usage.
  - Summaries come from tools/cc_mem_trace.py. On an iGPU the Dedicated counters read about 0
    (KB 4490156).
- Gates: the full measurement gates (uptime included), so each cell starts after a reboot when uptime
  is high (T2S: reboot before every cell; XPS: when uptime >= 6,900 s). One canary per probe as in
  the sealed runs. Child timeout 1,800 s per probe (configs/delta_n.yaml).
- Probe pass rule (unchanged): turn 1 completes and every turn 2 runs.

## 3. Cells
### 3.1 CC-REH-{T2S,XPS} (rehearsal, first cell of each queue)
- u4, points 1,024 and 2,048, every CC flag, launched via tools\spawn_detached.ps1 exactly as the
  real cells.
- Stop rule: the queue stops unless the cell completes and its log has the CC start line
  (REHEARSAL_COMPLETE).
- A missing sidecar CSV only alerts, because the child-side records remain.
- About 5 min.

### 3.2 CC-BYTES-{XPS,T2S}: where context bytes go (stacked-bar headline; first on each machine)
- Question: at a fixed context n, how many bytes are weights, KV, and other GPU allocations, per KV
  precision and step cap?
- XPS:
  - n = 4,096, 12,288 and the sealed limit minus about 1,000 (f16 14,000; u8 19,000; u4 25,500);
  - {f16, u8, u4} x {unl, c256}, plus u4 c1024;
  - 7 cells, turn 2 = 64 tokens x 3.
- T2S: the CC-MECH-T2S cells (section 3.3) carry the same records.
- Decomposition per n:
  - W = after-load Shared Usage;
  - KV = the settled (after-turn-1) GPU memory minus W, compared with n x k_KV(p) from the shapes,
    which also verifies the precision;
  - other = the turn-1 peak (sidecar) minus settled.
- Prediction (CC_PREDICTIONS t2s; the same rule holds on the XPS):
  - settled slope per token within +-25% of k_KV (f16 147,456 / u8 78,336 / u4 41,472 B);
  - uncapped peak-minus-settled slope in [71,956, 129,024] B per token (the sealed pair fits);
  - c1024 <= 65,536, c256 <= 16,384 B per token.
- Reps: memory r = 1 per n per cell (the T2S headline cells are repeated, section 3.3); TTFT and
  decode r = 3 (turn-2 repeats).
- Run time (DERIVED): about 10-25 min per XPS cell (3 probes; turn-1 prefill 55 s at 26,500 u4,
  e4a22dac) plus a reboot when needed. About 3 h for the 7 cells.

### 3.3 CC-MECH-{XPS,T2S}: does a step cap move the ceiling, and which non-KV term binds?
- **XPS:** resident-limit bisection (low 12,000, resolution 500), {f16, u8, u4} x {c256, c1024,
  unlx}: 9 cells.
  - Ladder tops: u4 100,000; u8 54,000; f16 30,000 (unlx: f16 20,000; u8 24,000; u4 30,000).
  - Then repeats at the found limit L* (points L*, L*+500, each repeated): u4 c256 r = 5; f16 c256,
    u4 unlx, f16 unlx r = 3; the other caps r = 1.
- **T2S:** the 64 GB tier is predicted not to bind below the model window (uncapped f16 ceiling
  about 161,000 tokens, DERIVED). So the T2S measures bytes, not a limit.
  - Fixed points:
    - unl DENSE: 2,048 / 4,096 / 8,192 / 12,288 / 16,384 / 24,576 / 32,768 / 40,448;
    - capped SPARSE: 4,096 / 12,288 / 24,576 / 40,448;
    - 40,448 + 183 + 64 stays inside 40,960 positions.
  - Cells: {f16, u8, u4} x {unl, c256, c1024} plus u4 unlx.
  - Headline repeats: u4 unl and u4 c256 to r = 5; f16 unl and f16 c256 to r = 3. 22 cells in all.
- **Hypotheses (DERIVED; each predicts the XPS limits below, f16/u4 pair):**
  - H-act: the prefill working set scales with the step's tokens, so a cap removes it.
    u4 c256 87,700; f16 c256 24,700.
  - H-mixed (strategy report): 64 x T bytes per context token.
    u4 c256 63,300; c1024 34,200; f16 c256 22,400.
  - H-grow: with a cap the KV grows by reallocate-and-copy, so the peak holds 1.5-2x KV.
    u4 c256 44,200-58,900.
- **Registered point and band (CC_PREDICTIONS xps_resident_limits)** (the point is H-mixed; the band
  spans all hypotheses and the three pair fits):

  | Precision | c256 | c1024 | unlx |
  |---|---|---|---|
  | u4 | 63,319 [36,239, 99,204] | 34,234 [28,090, 96,814] | 26,500 [25,500, 27,500] |
  | u8 | 38,676 [19,186, 52,520] | 25,463 [19,186, 51,255] | 20,000 [19,000, 21,000] |
  | f16 | 22,359 [10,192, 27,901] | 17,200 [10,192, 27,229] | 15,000 [14,000, 16,000] |

  - u4/f16 ratio at c256: 2.83 [2.83, 3.56]; at c1024: 1.99 [1.99, 3.56]. Sealed: 1.77.
- **Mechanism gate (F1, registered):** HIT if the median of the u4 c256 limit-point checks is
  >= 40,000 tokens (1.5x the sealed 26,500). Predicted HIT.
- **Discriminating rules (applied in this order, on XPS limits and T2S slopes):**
  - c1024 within 10% of c256 and both near H-act -> H-act;
  - c1024 near 34K while c256 is near 63K -> H-mixed;
  - capped settled slope about 1.5-2x k_KV on the T2S -> H-grow;
  - none -> "non-KV term moves with the step cap, cause open".
- **TIME_BOUND rule.** A probe that hits the 1,800 s child timeout with no allocation or OpenCL
  error is TIME_BOUND, not a memory limit. The cell is then UNRESOLVED above that n, not a MISS.
  - The XPS uncapped prefill extrapolates to about 900 s at 100,000 tokens (exponent 2.12,
    2b3316b6); capped steps may be slower. So u4 c256 near its top is at risk.
- **Run time (DERIVED):**
  - XPS: f16 cells about 45-75 min, u8 about 1-1.5 h, u4 capped about 2-3 h (about 10 probes, the
    high ones 5-30 min each). About 11 h of bisection plus about 4 h of repeats.
  - T2S: DENSE about 20-30 min, SPARSE about 15-25 min (prefill about 10 s at 18,875 tokens f16,
    5c714535; exponent 1.85, 65e33de8; about 90 s per-probe overhead for load and canary; reboot
    about 5 min). 22 cells, about 8-10 h.

### 3.4 CC-CEIL-{T2S,XPS}: ceiling microbenchmarks
- Data:
  - T2S: DENSE unl cells (8 context points) plus SPARSE capped cells;
  - XPS: CC-BYTES-XPS + CC-CLIFF-XPS points.
- Outputs:
  - W (after-load), k_KV (settled slope), k_ws(T) (peak-minus-settled slope at T in {unl, 1024,
    256}), R0 (intercept);
  - decode tok/s vs n against the bandwidth line tok/s <= eta x BW / (W + n x k_KV) (P2 found it
    holds within 3.3% on the T2S, 29209e26);
  - the KV-vs-weight traffic crossover L_x = W / k_KV. DERIVED for Qwen3-4B INT4 weights about
    2.35 GB (P2): f16 about 16,000; u8 about 30,000; u4 about 57,000.
- Prediction: decode tok/s within +-10% of the bandwidth line calibrated at 4,096 on each device. On
  the T2S the f16 point at 40,448 is past L_x, so KV bytes per token exceed weight bytes.
- No extra machine time beyond 3.2 / 3.3 / 3.5.

### 3.5 CC-CLIFF-{XPS,T2S}: what happens past the ceiling
- **XPS:** plain uncapped arms f16 and u4 at 0.8 / 0.9 / 0.95 / 1.0 / 1.05 / 1.25 / 1.5 x the sealed
  L*:
  - f16: 12,000 / 13,500 / 14,250 / 15,000 / 15,750 / 18,750 / 22,500;
  - u4: 21,200 / 23,850 / 25,175 / 26,500 / 27,825 / 33,125 / 39,750.
  - Turn 2 = 64 tokens x 3 (TTFT and decode r = 3 at the points that hold). Each point above L* is
    attempted once (r = 1): a loud failure can hang up to the 1,800 s child timeout, as 553b3a5c's
    15,500 did.
  - Logged: sidecar available memory and hard faults.
- **Prediction:**
  - points <= L* hold, with TTFT growing per the prefill exponent;
  - points >= L* + 500 fail LOUD in turn 1 (CL_OUT_OF_RESOURCES or an allocation error), as all 3
    sealed probes did;
  - no point holds with a >10x TTFT step (no silent paging in this probe). The strategy report's
    "UMA step" (F6) is therefore predicted NOT to appear in the resident probe.
- **T2S: BLOCKED this weekend.** The 4B ceiling is predicted far above the model window, and the
  installed 8B models do not reach it either.
  - A T2S cliff needs a lower GPU budget (Shared GPU Memory Override or bcdedit removememory). Both
    need a reboot and a setting change on Rithwik's host, so his written OK. Not scheduled.
- Run time (DERIVED): about 1-2.5 h per precision (the over-limit points dominate).

### 3.6 Conditional and later cells (not in this weekend's queues)
- **CC-XRT-T2S (CONDITIONAL on Rithwik's written OK; no script yet).**
  - Pinned llama.cpp build (Vulkan or SYCL) in C:\Users\<user>\seam_tools\llama.cpp_<tag>.
  - Same Qwen3-4B (GGUF Q4_K_M, digest pinned): f16 / q8_0 KV x flash attention on/off; resident
    limit by allocate-until-failure, and bytes at the SPARSE points.
  - It breaks the platform x runtime confound.
  - Prediction: with flash attention the non-KV slope is near flat (strategy report), so the u4 or
    q4 / f16 limit ratio is near the byte ratio.
  - Never touches APU-QueueWatchdog or Rithwik's files.
- **E2-T2S (CTXKV-E1E2; CONDITIONAL on Rithwik's written OK).** OpenVINO i8 vs u4 with 1 / 2 / 4
  concurrent sessions, as registered in docs/registrations/CTXKV_E1E2_2026-10-23.md.
- **CC-BUDGET-XPS (after Oct 27).**
  - Intel Shared GPU Memory Override (default 57%), one setting per reboot: default / 50% / max.
  - u4 c256 and f16 unl resident limits per setting.
  - F2 prediction: L* moves linearly with budget, slope 1 / (k_KV + k_ws) tokens per byte.
  - Needs Zach to set the driver option by hand.
- **CC-XRT-XPS (after Oct 27).** The CC-XRT design on the XPS (llama.cpp Vulkan).
- **E1-XPS (after Oct 27).** As registered in CTXKV (Ollama true default on a 16 GB tier).
- **Rithwik-side (PROPOSED, pending his OK):**
  - CC-AMD-BYTES / CC-AMD-LIMIT: Strix Halo 128 GB, llama.cpp, f16 and q8_0 KV with flash attention
    on/off, VGM/GTT tier emulation 8/16/32/64 GB;
  - CC-AMD-ZEROCOPY: device-local vs host-visible allocation tok/s at 1K/8K/32K with UMC/CS
    counters;
  - CC-4070-SPILL: RTX 4070 laptop, spill fraction f in {0, 0.05, 0.1, 0.2, 0.4}, sysmem fallback
    default vs "Prefer No Sysmem Fallback", link width read. Predicted ramp (1 - f) + f x r with
    r = 16.3 (DERIVED, x8 Gen4 assumed): 1.8x / 2.5x / 4.1x / 7.1x.

## 4. Queues and stop rules
- **T2S:** ~/seam_closeout_1013/xps/run_queue_t2s_cc.sh (new). It waits for the T2S lock and starts
  only if the held-out window logged "T2S BE v2 queue done".
  - Cell order: rehearsal, CC-MECH-T2S main cells (u4, f16, u8 at unl and c256, then c1024, then
    unlx), then repeats.
  - fits() before every cell; PAST_FINISH after 20:44 ET Sun; our worker only is stopped at
    21:15 ET; hand-back check (no python of ours, no sidecar, clean tree) by 21:44 ET Sun Oct 25.
- **XPS:** ~/seam_closeout_1013/xps/run_queue_xps_cc.sh (new).
  - Cell order: rehearsal, CC-BYTES-XPS, CC-MECH-XPS, repeats at L*, CC-CLIFF-XPS.
  - Copy-off by Mon Oct 26 08:00 ET.
- **Both:** stop on a failed rehearsal, a banned or foreign process, a dirty tree, a park failure or
  a head mismatch. A failed cell is recorded and the queue continues (XPS reboots after an error).

## 5. Pitfalls (registered)
- **Positions.** Points above 40,960 (XPS capped bisections up to 100,000) are past Qwen3-4B's
  trained positions. They measure memory only; no quality is claimed there.
- **Sidecar overhead.** The sidecar costs CPU (one Get-Counter per second). TTFT and decode at the
  same n are compared only within the sidecar regime.
- **GPU_MEMORY_STATISTICS.** It may read 0 or fail through a separate Core. The rehearsal decides;
  the PDH counters are the fallback, not a substitute claim.
- **Batch-size label.** The task brief names `--max-num-batched-tokens`; the runner flag is
  `--sched-max-batched-tokens` (same SchedulerConfig field).
- **Failure classes.** A cell's failures are classed from the child exception text (allocation or
  OpenCL error vs timeout). They are never retried silently.

## 6. Owners and dates
| What | Owner | Due |
|---|---|---|
| Launch T2S CC queue after held-out; launch XPS CC queue | Zach | Fri Oct 23 |
| CC-PRECHECK remainder (tag diff, 44,742 run_id, T2S GPU-visible figure) | Claude Code (Task BK) | Mon Oct 26 |
| Closeout, values, meeting pack (TASK_BK_MEETING27.md) | Claude Code | Mon Oct 26 |
| OK for CC-XRT-T2S / E2-T2S / any T2S setting change | Rithwik | before Sun Oct 25 noon to fit |
| Mechanism go/no-go (fix prototype vs context-guard fallback) | Zach | Fri Nov 6 |

## Gate text reconciliation (2026-10-23, Task BO; wording only)
The Nov 6 mechanism go/no-go uses one gate wording everywhere:
- held-out median absolute error <= 10% (target);
- > 15% median = red flag R8 (docs/PLAN_LOCK_2026-10-23.md);
- at least 30 held-out cells;
- the binding ceiling called correctly in >= 90%.
Source: docs/registrations/CC_AMEND_2026-10-23d_MODEL2.md section 4.
