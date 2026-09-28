# Claims Ledger — Paper 1

Every claim Paper 1 intends to make, with supporting evidence, hardware origin, and
current status. Status definitions:

- **SUPPORTED-ON-TARGET** — supported by data from Strix Halo EVO-X2 (AMD Ryzen AI Max+ 395,
  unified LPDDR5X, Ubuntu). None yet; EVO-X2 not yet provisioned.
- **OFF-TARGET-ONLY** — supported by committed data, but only from Razer Blade 14
  (RTX 4070 discrete, Windows) or evo-t2s (Intel Arrow Lake, Vulkan, unified LPDDR5X).
  Neither is the BOM target. Claim stands as a directional finding; target replication required
  before submission citation.
- **UNSUPPORTED** — no committed data supports this claim. Either the experiment was not run
  or the result file contains no valid measurements for this claim.
- **PENDING** — data collection is in progress now; status will be updated when committed.
- **UNVERIFIED** (added 2026-09-25, applied on top of the status above, never replacing it) - the claim rests on a result file whose stale-server exposure cannot be excluded, or whose generating script is lost. The claim is neither confirmed nor withdrawn.

No status field is valid without a cited file path.

---

## Section 4 — Axis A: Memory vs. Correctness

### Claim A-01 (Fig 4.1, SUPPORTING — negative result)
**Quality is flat across depth (0–32k tokens) at fixed memory budget** when filler is
semantically inert; depth alone is not the productive axis for quality degradation.

- Files: `results/run_20260813T021516Z.jsonl` (1100 rows, blade_rtx4070, qwen3:4b-instruct)
- Hardware: blade_rtx4070 (discrete, OFF-TARGET)
- **Status: OFF-TARGET-ONLY**

---

### Claim A-02 (Fig 4.3, CORE)
**Artifact truncation causes a sharp cliff in task score** — score drops from ~1.0 to ~0.0
as the artifact survival fraction falls below a per-probe extinction threshold.

- Files: `results/art_truncation.json`, `results/art_truncation_analysis.json`
- Hardware: blade_rtx4070 (discrete, OFF-TARGET — inferred from commit date)
- **Status: OFF-TARGET-ONLY**

---

### Claim A-03 (Fig 4.4, APPENDIX)
**The extinction threshold is probe-specific and occurs in the range 0.444–0.761 artifact
fraction** across a fine-grained 1.00 → 0.85 ratio sweep.

- Files: `results/partial_truncation.json`
- Hardware: blade_rtx4070 (inferred)
- **Status: OFF-TARGET-ONLY**

---

### Claim A-04 (Fig 4.6, SUPPORTING)
**Mean required artifact fraction for correct retrieval is 0.211** (range 0.064–0.433 across
10 probes); 78.9% of artifact tokens are evictable without loss.

- Files: `results/span_ablation.json`, `results/span_ablation.jsonl`
- Hardware: blade_rtx4070 (inferred)
- **Status: OFF-TARGET-ONLY**

---

### Claim A-05 (Fig 4.6 / Table A, SUPPORTING)
**Positional waste gap is 0.255–0.551 artifact-fraction** for 3 probes: the prefix-truncation
survival threshold exceeds the targeted-span retention requirement by this margin.

- Files: `results/span_ablation.json`, `results/artifact_ratio_sweep.json`
- Hardware: blade_rtx4070
- **Status: OFF-TARGET-ONLY**

---

### Claim A-06 (Fig 4.6, SUPPORTING)
**For 9 of 10 probes, the answer span alone (without header or surrounding context) is
sufficient for correct retrieval** (answer_no_header condition).

- Files: `results/span_ablation.jsonl`
- Hardware: blade_rtx4070
- **Status: OFF-TARGET-ONLY**

---

### Claim A-07 (Fig 4.6, SUPPORTING)
**art_04 requires an adjacent log-entry as a format exemplar** in addition to the answer
span; span-only retrieval succeeds semantically but fails on format (DELETE vs. deleted).

- Files: `results/span_ablation.jsonl`
- Hardware: blade_rtx4070
- **Status: OFF-TARGET-ONLY**

---

### Claim A-08 (Fig 4.14, APPENDIX)
**100% fabrication rate and 0% abstention in the 36 classifiable extinct-context rows**
(budget_ratio ≤ 0.85, artifact absent, unclassifiable rows removed from denominator).

- Files: `results/stage_a_scale.json`
- Hardware: blade_rtx4070 (inferred); cloud model gpt-oss:120b
- **Status: OFF-TARGET-ONLY**

---

### Claim A-09 (Fig 4.7 / 4.9, APPENDIX)
**An abstention instruction inverts its intent**: arm2 (abstention_instruction) drives
fabrication to 100% and eliminates the 20% abstention rate present in the baseline arm.

- Files: `results/selfreport_arms.json`
- Hardware: blade_rtx4070 (explicit field); model qwen3:4b-instruct
- **Status: OFF-TARGET-ONLY**

---

### Claim A-10 (Fig 4.9, APPENDIX)
**Schema collision null result**: zero filler lifts when the artifact is present, across
F-NUM, F-TYPED, and F-SCHEMA filler with 3 models (qwen3:4b-instruct, llama3.1:8b,
gpt-oss:120b).

- Files: `results/schema_collision.json`
- Hardware: blade_rtx4070 (inferred)
- **Status: OFF-TARGET-ONLY**

---

### Claim A-11 (Fig 4.8 / Fig 4.14, APPENDIX)
**Type-matched filler interference (art_02/F-TYPED/120B) is a position/recency effect**:
at r=1.20, artifact present, the 120B model retrieves a filler value rather than the artifact
value in 5 of 6 runs.

- Files: `results/stage_a_scale.json`, `results/interference_r120.json`
- Hardware: blade_rtx4070 (inferred); cloud model gpt-oss:120b
- **Status: OFF-TARGET-ONLY**

---

### Claim A-12 (Fig 4.11, SUPPORTING)
**Measured KV cache memory reduction: f16 → q8_0 = 1.77×, f16 → q4_0 = 3.24×** —
shallower than architectural predictions (2.00× and 4.00×) due to per-block quantization
metadata overhead and SWA default on Qwen3.

- Files: `results/llamaserver_feasibility.json` (reference); `results/gate1_kv_precision.json`
  (documents Ollama API limitation; contains no valid cross-precision measurements)
- Hardware: blade14_rtx4070 (explicit field); llama-server b1-f8def7fe1, CUDA
- **Status: OFF-TARGET-ONLY** — must re-run via llama-server on Strix Halo with
  `--cache-type-k` flags verified effective; unified memory subtraction method differs.

- **UNVERIFIED (2026-09-25):** `results/llamaserver_feasibility.json` is SCRIPT LOST, and `results/gate1_kv_precision.json` was run through `harness/llama_server.py`, which has no stale-server check, so exposure cannot be excluded. See `docs/RESULT_PROVENANCE.md`, sections "Stale-server exposure audit" and "Result file to generating script map". The status line above is kept unchanged; nothing was removed.

---

### Claim A-13 (Fig 4.12, SUPPORTING)
**Budget_ratio × artifact position (EARLY vs. LATE) is the productive axis**: under
harness-side left-char truncation, the LATE arm retains its artifact because filler precedes
it; the EARLY arm loses its artifact first.

- Files: `results/stage_c_20260818T040408Z.jsonl` (396 rows, blade_rtx4070,
  qwen3:4b-instruct), `results/position_pressure_analysis.json`;
  `results/fig61_stagec_full_20260922T203557Z.jsonl` (396 rows, evo-t2s,
  Intel Arrow Lake, Vulkan b10970, qwen3-4b-instruct-85e4a5b7.gguf, matched checkpoint);
  `results/fig61_stagec_full_20260922T230133Z.jsonl` (396 rows, blade_rtx4070,
  CUDA b10970, same checkpoint, like-for-like replication)
- Hardware: blade_rtx4070 (discrete, OFF-TARGET) + evo-t2s (unified LPDDR5X, primary target)
- **Three-architecture replication note:** All three runs agree on the position-pressure
  effect (LATE > EARLY at r<1 for probes where artifact is intact). One cell disagrees
  between CUDA and Vulkan backends: sea_01 LATE at r=1.0 and r=1.2 — CUDA outputs "C8"
  (wrong), Vulkan outputs "A9" (correct). Both Blade runs (Ollama and CUDA llama-server)
  agree on "C8". This is backend numerical sensitivity on one borderline probe; 130/132
  cells match. The disagreement does not affect the position-pressure claim.
- **Harness change confirmed neutral:** The Blade CUDA gate (`fig61_stagec_gate_20260922T230052Z.jsonl`)
  compared llama-server CUDA b10970 against `stage_c_20260818T040408Z.jsonl` (Ollama) on
  the same hardware (Blade 14, RTX 4070) and found **0/22 disagreements**. This confirms
  that the switch from Ollama to llama-server introduced no score change on the Blade.
  The evo-t2s vs. Blade score differences (1 cell: sea_01 LATE) can therefore be
  attributed to architecture (discrete CUDA vs. unified Vulkan), not to harness or
  runtime changes.
- **Status: ON-TARGET (evo-t2s)** for the Vulkan leg; the Blade legs (Ollama and CUDA) stay OFF-TARGET-ONLY. Three-architecture
  replication complete; EVO-X2 (the other primary platform) run still required for submission.

- **UNVERIFIED (2026-09-25):** for the parts resting on the `fig61_*` replications: the `fig61_*` files were produced with a server started outside the harness and with no model, ctx, PID or port check, so stale or wrong-server exposure cannot be excluded. The Ollama-based `stage_c_20260818T040408Z.jsonl` is not exposed to this failure mode. See `docs/RESULT_PROVENANCE.md`, sections "Stale-server exposure audit" and "Result file to generating script map". The status line above is kept unchanged; nothing was removed.

---

### Claim A-14 (Fig 6.1, CORE — primary contribution)
**The joint feasibility envelope**: a point in (budget_ratio, quality_score, latency) space
is feasible iff score ≥ quality floor AND latency ≤ budget, with EARLY/LATE arm determining
which envelope face the workload lies on.

- Files: `results/fig61_stagec_full_20260922T203557Z.jsonl` (396 rows, evo-t2s,
  Intel Arrow Lake, Vulkan b10970, qwen3-4b-instruct-85e4a5b7.gguf, matched checkpoint,
  streaming with TTFT, cache_prompt=false, max_tokens=128, temp=0, filler=4000 tok seed=42).
  Diagnostic-only predecessor 060942Z superseded (model confound); 191031Z superseded (timing invalid).
- Target hardware: EVO-X2 (Strix Halo), NOT YET PROVISIONED (arrives ~Dec 3, 2026); evo-t2s is the other primary platform and has run.
- **Status: ON-TARGET (evo-t2s).** EVO-X2 run still required for submission.

- **UNVERIFIED (2026-09-25):** the `fig61_*` files were produced with a server started outside the harness and with no model, ctx, PID or port check, so stale or wrong-server exposure cannot be excluded. `fig61_stagec_full_20260922T203557Z.jsonl` also has an uncertain script version (nearest committed version 0fd749a). See `docs/RESULT_PROVENANCE.md`, sections "Stale-server exposure audit" and "Result file to generating script map". The status line above is kept unchanged; nothing was removed.

---

### Claim A-15 (Fig 4.13, SUPPORTING)
**Quality-depth curves are model-dependent**: llama3.1:8b shows different per-probe failure
modes from qwen3:4b-instruct, establishing the curves are not harness artifacts.

- Files: `results/model2_truncation.json`, `results/crossmodel_baseline.json`,
  `results/run_20260818T000746Z.jsonl`
- Hardware: blade_rtx4070 (inferred)
- **Status: OFF-TARGET-ONLY**

- **UNVERIFIED (2026-09-25):** for the part resting on `results/crossmodel_baseline.json`: SCRIPT LOST. See `docs/RESULT_PROVENANCE.md`, sections "Stale-server exposure audit" and "Result file to generating script map". The status line above is kept unchanged; nothing was removed.

---

### Claim A-16 (Fig 4.2, SUPPORTING — mechanism note)
**cha_04 failure mechanism is uncharacterized**: the config-parameter substitution hypothesis
is disconfirmed by ablation; the probe returns 600 (wrong) at depth even without the retries
field present.

- Files: `results/ablation_cha04_20260817.jsonl`, `results/run_20260813T021516Z.jsonl`
- Hardware: blade_rtx4070
- Note: the score drop (1.0 → 0.0 at d=2000) is confirmed; the mechanism is not.
- **Status: OFF-TARGET-ONLY** — the mechanism claim must NOT be asserted.

---

### Claim A-17 (Fig 4.2, SUPPORTING — mechanism note)
**lon_02 format failure couples with correct computation at d=16000**: reps that compute
correctly at depth output "130/5=26" (format failure); reps that output a bare digit at
d=32000 often output the wrong digit.

- Files: `results/run_20260813T021516Z.jsonl`
- Hardware: blade_rtx4070
- **Status: OFF-TARGET-ONLY**

---

### Claim A-18 (Table 4.1 / APPENDIX)
**Parametric default failure class**: models emit canonical field-type sentinels (8080, 0,
2147483647) when the answer span is absent — drawn from parametric knowledge, not context.

- Files: `results/span_ablation.jsonl`
- Hardware: blade_rtx4070
- **Status: OFF-TARGET-ONLY**

---

### Claim A-19 (Fig 4.15, SUPPORTING — blocked)
**Multi-turn recall degrades as a function of artifact distance, intervening schema, and
artifact size** under partial KV eviction.

- Files: NONE — harness not yet written; data not yet collected.
- **Status: UNSUPPORTED**

---

### Claim A-20 (Fig 6.2 candidate, SUPPORTING)
**On a discrete GPU with the driver-default Sysmem Fallback Policy, exceeding VRAM causes a silent slowdown with a sharp onset, no error and no clock throttling.** On the Blade the onset lies between ctx 36864 (0.98x) and 38912 (3.71x TTFT); decode slowdown is proportional to the spilled fraction (0.08 + 74.6 x excess fraction, R2 0.998, n=5); 10 of 10 correctness probes are unchanged under 14.2% spill. Whether the setting "Prefer No Sysmem Fallback" turns this into a hard out-of-memory failure (C3) is not yet tested.

- Files: `results/blade_m1_vram_spill_20260925T041415Z.jsonl`, `results/blade_c1_spill_sweep_20260925T053651Z.jsonl`, `results/blade_c2_spill_correctness_20260925T110739Z.jsonl`
- Hardware: blade_rtx4070 (discrete, OFF-TARGET)
- **Status: OFF-TARGET-ONLY.** The forced-failure half is PENDING (C3).

---

### Claim A-21 (Fig 6.2 candidate, SUPPORTING, mechanism open)
**A host-side memory-bandwidth co-runner slows a fully VRAM-resident model on the Blade** by 3.1x to 4.7x on TTFT and 2.7x to 5.6x on decode (memcpy at 16 threads, ctx 8192) while GPU utilization falls 57 to 67 points. A pure-compute co-runner (spin, 16 processes) costs 1.09x TTFT and 1.79x decode. The mechanism is not established: host-on-critical-path is supported in three of four rows, shared power is not ruled out for the default-thread memcpy row, PCIe was not tested, and Shared Usage was not captured under load.

- Files: `results/blade_m2_host_interference_20260925T043921Z.jsonl` (Parts A and B only; Part C is invalid)
- Hardware: blade_rtx4070 (discrete, OFF-TARGET)
- **Status: OFF-TARGET-ONLY** for the magnitude. Mechanism: not established.

---

### Claim A-22 (Fig 6.2 candidate, SUPPORTING)
**On unified-memory evo-t2s (Vulkan), locking available memory from 12 GB down to 4 GB produced no silent slowdown** (TTFT at most 1.08x, 5 of 5 probes correct at every level whose server started) **and one loud crash at 5 GB** (Vulkan device lost, exit 0xC0000409) that did not recur at 4 GB. Server load time rose from about 3 s to about 150 s at 6 and 4 GB. One run per level; the crash is unrepeated.

- Files: `results/ramlock_evo-t2s_20260925T010739Z.jsonl`, `results/ramlock_phaseD_telemetry/`
- Hardware: evo-t2s (unified memory, primary target)
- **Status: ON-TARGET (evo-t2s).** Hard-fault and pagefile data after server start are UNKNOWN at 6, 5 and 4 GB. EVO-X2 not yet run.
- **UNVERIFIED (2026-09-25, correction):** the balloon's safety valve released the lock during load at 6 GB and the 4 GB level shows an unconstrained working set after load, so "no slowdown at 6 and 4 GB" is not a valid constrained result. The supported range is no silent slowdown from 12 GB down to 7 GB. The crash at 5 GB is a load-phase event under real pressure and stays one unrepeated run. See `docs/FINDINGS.md` (Phase D correction). The status line above is kept unchanged.
- **Update (2026-09-26, overnight Section C):** a second, independent lock design (8B, context 16,384, headroom measured against weights + KV + compute) found no TTFT slowdown at zero and -1 GB headroom (TTFT +1.4% to +2.0% of the +8 GB anchor, 5 of 5 probes correct) with the file-backed load mode, and at zero headroom without it. A decode cost of 7% to 8.5% is **not ruled out** (outside the 5% band; one anchor, no repeats). The 32B at zero headroom with 3 MB available ran normally (TTFT 1.03x of Section A anchors, 5 of 5 probes). One more loud failure: 8B without mmap at -1 GB stalled 1,115 s at load and lost the device (n=1). Same design limits as above: one cell per level. See `docs/T2S_OVERNIGHT_REPORT.md`.

---

### Claim A-23 (Fig 6.2 candidate, SUPPORTING)
**On evo-t2s a CPU-only co-runner slows iGPU inference and drops its clock, by a mechanism that is not established.** With 12 or 16 cores spinning, RAPL package power sits at about 45 W, the iGPU is held at 1650 MHz (from 2500 MHz) and TTFT rises 1.42x and 1.41x; with only the 4 P-cores spinning TTFT rises 1.04x. The prediction that P-core spin would hurt more was wrong.

- Files: `results/t2s_m3_power_coupling_20260925T075348Z.jsonl` (+ Sysman and Windows counter files)
- Hardware: evo-t2s (unified memory, primary target)
- **Status: ON-TARGET (evo-t2s).** 3 calls per condition, one run. **Mechanism: OPEN.** EVO-X2 not yet run.
- **Update (2026-09-26, overnight Section B):** the magnitude **replicates** on two more model sizes (4B-2507: 1.40x and 1.39x TTFT, 1.24x decode; 8B: 1.42x and 1.43x, 1.20x), iGPU 1650 and 1600 MHz, 5 measured calls per condition, bootstrap intervals excluding 1.0. Section B adds an e4 arm (4 E-cores) between p4 and nonp12: 4B-2507 1.02x TTFT at 39.1 W package power, 8B 1.05x at 41.1 W, both with the iGPU at 2450 to 2500 MHz, in between the p4 result (44.9 W, iGPU near 2500 MHz, 1.04x to 1.06x) and nonp12/all16 (44.9 W, iGPU 1600 to 1650 MHz, 1.39x to 1.43x). **p4 reaches the same 44.9 W package power as nonp12 but keeps the iGPU at 2,450 to 2,500 MHz, so package power alone does not explain the iGPU clock drop or the slowdown** (Pearson r of package W against iGPU MHz across the five conditions is -0.53 in each model). **Mechanism: OPEN.** Do not describe this as a shared power budget effect. The causal arm (PROCTHROTTLEMAX) failed its positive control (44.93 W at cap 50, 44.94 W at cap 100) and was not run; the co-runner conditions ran without an effective thermal gate (release by 120 s timeout) and no temperature sensor exists on this SoC through Level Zero, so thermal throttling is not ruled out. Two sizes, both Qwen3. See `docs/T2S_OVERNIGHT_REPORT.md`.

---

### Claim A-24 (Fig 6.2 candidate, SUPPORTING)
**On evo-t2s (Vulkan b10970) a llama-server start fails loudly when the required device memory exceeds the Vulkan memory-heap budget, and the boundary is set by total memory, not by the model.** The budget is 47,866 MiB (`vulkaninfo` heap budget; llama-server free figure 47,865 MiB), not the heap size (37,060 MiB, exceeded by successful runs) and not the single-allocation limit (`maxMemoryAllocationSize` 4,096 MiB; the refused buffer was 964 MiB). Bisecting n_ctx to a 256-token step, four models flip between 47,650 and 47,969 MiB of llama.cpp-projected memory: Qwen3-32B (last pass 115,712, first fail 115,968), Qwen3-8B (305,152 / 305,408), Qwen3-30B-A3B MoE (320,256 / 320,512) and Llama-3.1-8B (343,808 / 344,064). The refusal is `vkAllocateMemory` returning `ErrorOutOfDeviceMemory` during KV allocation; llama.cpp's fit pre-check projected the overflow and only warned because `-ngl` was pinned.

- Files: `results/t2s_amech_20260926T181456Z.jsonl` (+ `_vulkaninfo.txt`, server logs), `results/t2s_overnight_20260926T011744Z.jsonl` (Section A starts)
- Hardware: evo-t2s (Intel Arc B390 unified memory, Vulkan; primary target)
- **Status: ON-TARGET (evo-t2s).** Start-only (no prompts); the model-independence spread is 295 MiB (0.6%); all boundary contexts are beyond the trained context (memory tests, not usable contexts); EVO-X2 unmeasured. The 32B boundary flips were repeated 3 to 4 times each and were stable; the other models were probed once per point.

---

### Claim A-25 (Fig 6.2 candidate, SUPPORTING)
**Below the budget the 32B shows no TTFT slowdown as the allocated context grows from 65,792 to 111,104 tokens** (1.00 [0.99, 1.00], bootstrap 95%, 5 to 20 calls per point, 47,190 MiB of GPU Shared Usage at the larger point). **Above the budget, with `-ngl 99` pinned, only start failures appeared, no slow or wrong-answer regime** (5 of 5 grid points from 117,248 to 125,440 failed at start). **Under the default runtime policy (`-ngl` unset, `-fit on`), the picture changes: the 32B ran at the first failing context with 3 layers offloaded to CPU, at a 17% decode cost** (matched 12,802-token prompt and rope flags: 2.71 tok/s at 62/65 layers on GPU versus 3.25 tok/s at 65/65 layers on GPU at the last passing context, TTFT ratio 1.01x) **, and crashed outright one step higher** (see the fit-policy map, A-mech).

- Files: `results/t2s_overnight_20260926T011744Z.jsonl`, `results/t2s_amech_20260926T181456Z.jsonl` (arms and map phases)
- Hardware: evo-t2s (primary target)
- **Status: ON-TARGET (evo-t2s).** **UNVERIFIED for full-KV behavior:** calls used 4,527-token prompts on the pinned-`-ngl` ladder (the KV was allocated but about 4% filled); a 90% fill of these contexts is estimated at hours per call; the matched decode comparison used a 12,802-token prompt (10% fill), not 90%. One model. All points use YaRN factor 4 (see A-26). EVO-X2 not yet run.

---

### Claim A-26 (threat to A-25, SUPPORTING)
**YaRN alone changed a probe answer on the 32B.** With identical prompts at context 16,384 (14,747 tokens), art_03 answers 8.9 without YaRN (5 of 5 probes correct) and 33.6 with `--rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 32768` (4 of 5). At context 111,104 with YaRN and a 4,532-token prompt art_03 again answers 33.6. The flags produced YaRN behavior (first-token log-probabilities differ from linear scaling at the same `freq_scale` by up to 0.448 nats and from no scaling by up to 0.214; the process command line carries `--rope-scaling yarn`).

- Files: `results/t2s_overnight_20260926T011744Z.jsonl` (YaRN check), `results/t2s_amech_20260926T181456Z.jsonl` (rope phase)
- Hardware: evo-t2s (primary target)
- **Status: ON-TARGET (evo-t2s).** One probe, one model, one run; the 4 of 5 at 111,104 must not be read as a memory effect. EVO-X2 not yet run.

---

### Claim A-27 (Fig 6.2 candidate, SUPPORTING)
**On evo-t2s the default load mode is not file-backed on this Vulkan device, and an explicit `--load-mode mmap` is.** For the 8B: `--load-mode mmap` gives private bytes 5,916 MiB and working set 10,297 MiB; `none` and `auto` give 6,243 and 6,178 MiB. All earlier evo-t2s runs (including Phase D) therefore used non-file-backed weights. In the lock design of A-22, at -1 GB headroom, the mmap arm loaded in 23.0 s (pages input peak 56,571/s) and answered normally, while the non-mmap arm ran for 1,115 s and then exited with 0xc0000409 (device lost). At 0 GB headroom both arms loaded normally (mmap 19.9 s, pages input peak 1,152/s; non-mmap 7.7 s, pages input peak 25,082/s) with no TTFT difference. One start per arm per level (n=1); UNVERIFIED as an mmap effect.

- Files: `results/t2s_overnight_20260926T011744Z.jsonl` (`mmap_control` record, Section C)
- Hardware: evo-t2s (primary target)
- **Status: ON-TARGET (evo-t2s).** EVO-X2 not yet run.

---

## Section 5 — Axis B: Orchestration / Throughput vs. Latency

### Claim B-01 (Table 5.1, SUPPORTING)
**SDK and LangGraph orchestration decompose into measurable span categories**
(ORCH_SETUP, HTTP_CLIENT, TOOL_COMPUTE, FRAMEWORK, RESIDUAL) across 14 task types.

- Files: `results/claude_code_characterization.json` — **IN REPO** (committed 2026-09-28,
  force-added past a stale `.gitignore` entry; see `docs/RESULT_PROVENANCE.md`)
- Hardware: laptop (AMD Ryzen 9 8945HS, 8c/16t, 31.28 GB), OFF-TARGET
- **Status: OFF-TARGET-ONLY** — real data, reproducible, but on laptop hardware via the OpenAI
  API (`gpt-4o-mini`), not evo-t2s/evo-x2 or a llama.cpp backend.

---

### Claim B-02 (Fig 5.1, SUPPORTING)
**Tail-latency (p50/p99) distributions across 14 task types under 3 concurrency conditions**
identify outlier tasks that constrain system design.

- Files: `results/tail_latency_results.json` — **IN REPO** (committed 2026-09-28,
  force-added past a stale `.gitignore` entry; see `docs/RESULT_PROVENANCE.md`)
- Hardware: laptop (AMD Ryzen 9 8945HS, 8c/16t, 31.28 GB), OFF-TARGET
- **Status: OFF-TARGET-ONLY** — same basis as B-01: real data, reproducible, laptop/OpenAI-API,
  not evo-t2s/evo-x2.

---

### Claim B-03 (Fig 5.2, SUPPORTING)
**Independent replication (Zachary Johnson) cross-validates span attribution methodology**
for the remote-search LangGraph task.

- Files: `results/zachary/replication_remote_search_v3.json` — **IN REPO**
- Hardware: UNKNOWN (external replication; hardware not documented)
- **Status: OFF-TARGET-ONLY** — hardware not documented; label as external replication.

- **UNVERIFIED (2026-09-25):** `results/zachary/replication_remote_search_v3.json` was produced by a script outside this repository, so no generating script is available here (SCRIPT LOST from this repo's point of view). See `docs/RESULT_PROVENANCE.md`, sections "Stale-server exposure audit" and "Result file to generating script map". The status line above is kept unchanged; nothing was removed.

---

### Claim B-04 (Fig 6.1, CORE — per-call join)
**TTFT and http_client_ns are measured on the same call as quality score**, enabling
per-call (not mean-of-means) joint envelope points.

- Files: `results/fig61_stagec_full_20260922T203557Z.jsonl` (evo-t2s, streaming, TTFT valid,
  cache_prompt=false verified, n_prompt_tokens_actual from /tokenize, tokens_in_api 396/396).
  Previous run 191031Z scores-valid but timing-invalid (superseded; see RESULT_PROVENANCE.md).
- **What the data shows:** TTFT varies with prompt length (budget_ratio), not with arm.
  LATE and EARLY are within 3% of each other at every ratio (e.g., r=1.20: LATE 7305ms vs
  EARLY 7434ms). TTFT falls from ~7.4s at r=1.20 to ~2.0s at r=0.40, consistent with
  prefill scaling linearly with token count. B-04 supports the claim that TTFT is validly
  measured per call in the same row as quality score. It does not support a claim that
  artifact position affects latency.
- **Status: ON-TARGET (evo-t2s).** Per-call TTFT measured on evo-t2s; EVO-X2 required for submission envelope.

---

## Section 6 — Joint Envelope (Primary Contribution)

- **UNVERIFIED (2026-09-25):** the `fig61_*` files were produced with a server started outside the harness and with no model, ctx, PID or port check, so stale or wrong-server exposure cannot be excluded. `fig61_stagec_full_20260922T203557Z.jsonl` also has an uncertain script version. See `docs/RESULT_PROVENANCE.md`, sections "Stale-server exposure audit" and "Result file to generating script map". The status line above is kept unchanged; nothing was removed.

### Claim J-01 (PRIMARY CLAIM)
**f(workload type, quality floor) → minimum provisioned GB**: the minimum context memory
needed to meet a quality floor is a function of workload category (RAG vs. search) and
artifact position, measurable empirically from budget_ratio × position sweeps.

- Files: `results/stage_c_20260818T040408Z.jsonl` (Blade, OFF-TARGET, non-streaming);
  `results/fig61_stagec_full_20260922T203557Z.jsonl` (evo-t2s, primary target, streaming,
  matched checkpoint, TTFT valid, cache_prompt=false). EVO-X2 run required.
- **Status: ON-TARGET (evo-t2s).** The Blade leg stays OFF-TARGET-ONLY. Two-architecture replication complete; EVO-X2 (the other primary platform) data not yet collected.

---

## Claims requiring Strix Halo data before submission

The following claims are in CORE or SUPPORTING figures and must also be reproduced on
AMD Ryzen AI Max+ 395 (EVO-X2, unified LPDDR5X, the other primary platform) before submission.
evo-t2s and EVO-X2 are both primary target platforms; only the Razer Blade 14 (discrete RTX 4070) is
OFF-TARGET. A claim resting only on evo-t2s is ON-TARGET but incomplete, not OFF-TARGET-ONLY.

| Priority | Claim | Figure | Blocking issue |
|---|---|---|---|
| 1 | J-01 (joint envelope) | Fig 6.1 (CORE) | EVO-X2 not provisioned; evo-t2s (primary) replication complete |
| 2 | A-02 (truncation cliff) | Fig 4.3 (CORE) | Discrete RTX 4070; bandwidth difference may shift cliff |
| 3 | A-01 (quality flat at depth) | Fig 4.1 (CORE) | Same as A-02 |
| 4 | A-12 (KV quantization ratios) | Fig 4.11 (SUPPORTING) | gate1 Ollama path invalid; llama-server rerun on Strix Halo AMD unified memory path unvalidated |
| 5 | A-04 / A-05 (span ablation) | Fig 4.6 (SUPPORTING) | Unified memory bandwidth may change required fraction |
| 6 | A-13 (position pressure) | Fig 4.12 (SUPPORTING) | Two-architecture replication complete (Blade off-target + evo-t2s primary); EVO-X2 required for submission |
| 7 | A-15 (multi-model comparison) | Fig 4.13 (SUPPORTING) | Blade only |
| 8 | A-19 (multi-turn recall) | Fig 4.15 (SUPPORTING) | Harness not written; EVO-X2 required |
| 9 | B-01 / B-02 (Axis B span data) | Table 5.1, Fig 5.1 | Files gitignored; must commit or re-run |
| 10 | A-20 / A-21 (Blade VRAM spill and host interference) | Fig 6.2 candidates | Discrete GPU behaviour; Strix Halo has unified memory, so the regime must be measured there |
| 11 | A-22 / A-23 / A-24 to A-27 (evo-t2s memory lock, power coupling, budget boundary, runtime policy) | Fig 6.2 candidates | ON-TARGET on evo-t2s (Intel Arc B390 unified memory, Vulkan); EVO-X2 path unmeasured |

**APPENDIX figures** (A-03, A-06, A-07, A-09, A-10, A-11, A-14, A-16, A-17, A-18, B-03)
are acceptable labeled as "Blade 14 / RTX 4070" with a note that Strix Halo re-runs are
planned; they do not block submission if CORE figures 6.1, 4.3, and 4.1 are reproduced.

- **UNVERIFIED (2026-09-25):** for the parts resting on the `fig61_*` replications: the `fig61_*` files were produced with a server started outside the harness and with no model, ctx, PID or port check, so stale or wrong-server exposure cannot be excluded. The Ollama-based `stage_c_20260818T040408Z.jsonl` is not exposed to this failure mode. See `docs/RESULT_PROVENANCE.md`, sections "Stale-server exposure audit" and "Result file to generating script map". The status line above is kept unchanged; nothing was removed.
