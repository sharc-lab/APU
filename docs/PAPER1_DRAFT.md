# Paper 1 Draft Skeleton (ISPASS 2027 target, anonymized)

Status: outline for advisor review, 2026-10-08. Not a submission. Double-blind is assumed: this file names no
author, affiliation, machine owner or repository. Author and internal labels are in `docs/PAPER1_COVER_NOTE.md`.

Rules this draft follows:

- Every measured number is pasted verbatim from `docs/NUMBERS_REGISTER.md` and carries its register row id in
  square brackets, e.g. `[A-24-flip-point-prediction]`. A number with no register row is not used in the text; the
  claim goes under Gaps (Section 8 of this file).
- Platform labels. INTEL-iGPU = Intel Core Ultra X7 358H (Panther Lake) with Arc B390 iGPU, unified memory.
  AMD-iGPU = AMD Ryzen AI Max+ 395 (Strix Halo) with Radeon 8060S iGPU, unified memory with a firmware-reserved GPU
  block. NVIDIA-dGPU = laptop with an RTX 4070 Laptop GPU, discrete VRAM. Register values are pasted verbatim and
  some of them contain the internal host labels `evo-t2s` or `T2S` (= INTEL-iGPU) and `evo-x2` or `X2`
  (= AMD-iGPU), as do result file names; these are replaced in the LaTeX version. Queue-job and plan-item names
  (x2_r2_real_v1b, "T2S week item N", "Blade night N (x)") are internal tracking labels for this outline only and do
  not appear in the submission.
- Deadline: UNCONFIRMED. `docs/TIMELINE.md` estimates an abstract deadline around 2026-12-07 and a full paper around
  2026-12-14/15 (no 2027 CFP was out when it was written); `docs/PLAN_PAPER1_DEMO.md` uses 2026-12-02. This draft
  does not resolve the conflict. Page limit: assumed 10 pages plus references, UNCONFIRMED until the 2027 CFP is
  published.

---

## 1. Working title and abstract

**Working title:** *Silent by Default: How Memory Architecture and Runtime Policy Shape Local LLM Agents on AI PCs*

Alternatives:

1. *Where Local Agents Break: Memory Boundaries, Runtime Defaults and Silent Context Loss on Three AI PC Platforms*
2. *The Runtime Picks Your Context Window: A Cross-Vendor Characterization of Local LLM Inference on AI PCs*
3. *No Error, No Rules: Characterizing Silent Failure of Local LLM Agents Across Unified and Discrete Memory*

**Abstract (150 words, counted with `wc -w` on the paragraph below; contains no numbers by design):**

Local agent stacks on consumer AI PCs fail in ways their users cannot see. We characterize three platforms from
three GPU vendors spanning unified and discrete memory: an Intel integrated GPU, an AMD Strix Halo integrated GPU,
and an NVIDIA laptop GPU. First, memory architecture sets hard boundaries: on the Intel integrated GPU a model start
succeeds or fails at one shared-heap limit that also counts host-visible buffers, and a first-principles accounting
predicts the measured flip point; on discrete memory, overflow becomes a silent spill. Second, runtime
policy, not hardware, sets the effective context window: one runtime picks very different default windows on two
integrated GPUs because of device detection, and truncates overflow silently. Third, these defaults reach
agents: in tool-using sessions, history is lost without any error, and at the smallest window the system
prompt is discarded. We give a predictor for that loss and evaluate a pre-registered mitigation.

Claims behind each abstract sentence (status in brackets):

- Shared-heap boundary and first-principles flip point: `[A-24-effective-heap-limit]`, `[A-24-flip-point-prediction]`
  (verified, INTEL-iGPU only).
- Discrete overflow is a silent spill: NO register row yet (pending; Gap G4). The sentence must be cut if G4 does not
  land.
- Device-detection policy sets the default window: `[T2S-vs-X2-default-ctx]`, `[x2-device-detect-mechanism]`
  (verified, INTEL-iGPU and AMD-iGPU).
- Silent overflow truncation: `[ollama-overflow-keeps-half]` (verified).
- History lost without error; system prompt discarded at the smallest window: `[R2-real-v1-gated-kill]`,
  `[R2-mechanism-lowlevel]` (verified on AMD-iGPU only; cross-platform pending x2_r2_real_v1b, T2S week item 2,
  Blade night 1 (b)).
- Predictor: no data yet (at risk; Gap G6). Mitigation: pre-registered, not run (pending x2_r2_mitigation_v1,
  T2S week item 3, Blade night 1 (c)). The last abstract sentence is contingent on both.

---

## 2. Contributions

Each item is tagged **verified now**, **pending run** (with the job that fills it) or **at risk**.

1. **A cross-vendor map of where memory runs out and how it fails.** On INTEL-iGPU the start/refuse boundary is
   one shared heap that includes host-visible buffers, and a first-principles accounting (weights + KV + compute)
   predicts the flip point: "C shared-heap LOO -257 to -26 tokens (-0.11% to -0.01%)" `[A-24-flip-point-prediction]`.
   - INTEL-iGPU half: **verified now.**
   - AMD-iGPU spill/crash regimes: **pending run** x2_70b_edge_reps, plus register rows for the existing
     `results/mx2_validation.jsonl` and `results/mx2_spill_stats.jsonl` (Gap G5).
   - NVIDIA-dGPU silent spill and the forced-failure driver setting: **pending run** Blade night 2 (d) (C3 sysmem
     fallback), plus register rows for the existing C1/C2 files (Gap G4).
2. **Runtime policy, not hardware, sets the effective context window, and overflow is silent.** The same runtime
   chooses a small default on INTEL-iGPU and model-native defaults on AMD-iGPU because of device detection
   `[T2S-vs-X2-default-ctx]` `[x2-device-detect-mechanism]`; overflow keeps "num_ctx/2 + 2" tokens with no error
   `[ollama-overflow-keeps-half]`; the install path of the same weights changes tool-call behaviour
   `[R2-install-path-4b-comparison]`.
   - INTEL-iGPU and AMD-iGPU: **verified now.**
   - NVIDIA-dGPU default context: **pending run** Blade night 1 (a) (K1).
3. **These defaults reach agents, silently, and we explain the mechanism.** In real tool-using sessions history is
   lost with no error at every truncating window, and rules are lost only at the smallest one
   `[R2-real-v1-gated-kill]`; message-level trimming keeps the system prompt, and llama.cpp context shift plus a
   token-level cut discard it at the smallest window `[R2-mechanism-lowlevel]`.
   - AMD-iGPU, small model set and seed count: **verified now.**
   - More models and seeds: **pending run** x2_r2_real_v1b. INTEL-iGPU: **pending run** T2S week item 2 (R2 native
     Intel) and T2S week item 6 (mechanism run). NVIDIA-dGPU: **pending run** Blade night 1 (b) (R2 native) and
     Blade night 2 (e) (mechanism).
4. **A predictor for system-prompt loss and a pre-registered mitigation.**
   - Context-shift predictor: being built, no result file: **at risk.**
   - Mitigation: **pending run** x2_r2_mitigation_v1 (queued after x2_r2_real_v1b), T2S week item 3 (mitigation at
     4096), Blade night 1 (c) (mitigation).

---

## 3. Platforms and versions

Configuration record, not measured results. Source files are given per cell. None of these values has a register
row yet; Gap G13 adds a `platform-config` row so the paper can cite them the same way as results.

| field | INTEL-iGPU | AMD-iGPU | NVIDIA-dGPU |
|---|---|---|---|
| CPU / GPU | Core Ultra X7 358H (Panther Lake), Arc B390 iGPU (`docs/HARDWARE.md`, `docs/T2S_OVERNIGHT_REPORT.md`) | Ryzen AI Max+ 395 (Strix Halo), Radeon 8060S iGPU (`docs/HARDWARE.md`) | RTX 4070 Laptop GPU (`docs/FINDINGS.md`, Blade sections) |
| memory architecture | unified, one Vulkan heap that all memory types map to (`[A-24-effective-heap-limit]`: "4 memory types all on heap 0") | unified, with a firmware-reserved dedicated GPU block plus a driver-shared extension (`docs/HARDWARE.md`) | discrete VRAM; driver may spill to system RAM (`docs/FINDINGS.md`, C1) |
| GPU driver | 32.0.101.8509 (`docs/T2S_OVERNIGHT_REPORT.md`) | 32.0.31007.1017 (`docs/HARDWARE.md`, `docs/X2_CHANGELOG.md`) | 610.88 (`docs/FINDINGS.md`) |
| llama-server (direct runs) | b10970, Vulkan (`docs/FINDINGS.md`) | b10970, commit bfdc32183, Vulkan (`docs/HARDWARE.md`) | b10970, CUDA (`docs/FINDINGS.md`) |
| Ollama | 0.33.2 (`docs/T2S_CHANGELOG.md`; K1 v3 run) | 0.34.4, bundled llama.cpp b11081 (`docs/R2_DESIGN.md`, `docs/X2_CHANGELOG.md`) | 0.34.0, earlier 0.32.9 (`docs/THREATS.md` section 18) |
| Ollama GPU backend actually used | none: iGPU dropped by policy, CPU fallback, unless `OLLAMA_IGPU_ENABLE=1` (`[x2-device-detect-mechanism]`) | ROCm (`[x2-device-detect-mechanism]`) | CUDA (assumed; not recorded) |
| OS | Windows 11 (build MISSING) | Windows build 10.0.26100 (`docs/HARDWARE.md`) | Windows 11 Home 10.0.26200 (`docs/FINDINGS.md`) |

Missing and must be recorded before submission (Gap G13):

- INTEL-iGPU Windows build; the llama.cpp build bundled in Ollama 0.33.2.
- NVIDIA-dGPU: the CUDA Sysmem Fallback Policy value during C1/C2 ("UNKNOWN" in `docs/FINDINGS.md`); the Ollama
  backend actually used; whether Ollama 0.34.0 there passes the same context-shift flags.
- Version skew: INTEL-iGPU Ollama results are 0.33.2 and AMD-iGPU results are 0.34.4. The cross-platform default
  context comparison (`[T2S-vs-X2-default-ctx]`) therefore crosses an Ollama version as well as a device. T2S week
  item 1 (re-enable checks) should record the current version; a re-run on a matched version closes it.

---

## 4. Section outline and page budget

Target: 10 pages plus references (UNCONFIRMED). Ordering: durable architecture findings first, then runtime policy
and agent consequences, then predictor and mitigation.

| # | Section | Pages |
|---|---|---|
| 1 | Introduction | 1.00 |
| 2 | Background and related work | 0.75 |
| 3 | Platforms, harness and methodology | 1.25 |
| 4 | Memory-architecture boundaries | 2.00 |
| 5 | Runtime policy | 1.50 |
| 6 | Agent consequences | 1.50 |
| 7 | Predictor and mitigation | 1.00 |
| 8 | Threats to validity | 0.50 |
| 9 | Conclusion | 0.50 |
| | **Total** | **10.00** |

For each section below: the one-sentence claim, the evidence (register rows and result files), the figure, and the
gaps with the job that fills each.

### 4.1 Introduction (1.00 page)

- **Claim:** local agent stacks on AI PCs fail silently, and which failure a user gets is set by the memory
  architecture and by runtime defaults, not by the model alone.
- **Evidence for the motivation:** real agent trajectories run long enough to hit small defaults: "81.1% of steps
  exceed evo-t2s's 4096 default; 0.0% exceed evo-x2's smallest model default (40,960)"
  `[trace-fraction-exceeds-4096]`; "29.85%" of runs in one censored trace source end at their generator's own context
  limit `[trace-context-exit-rate]`; the models themselves do not report the loss: "0/240 (T2S) + 0/720 (X2) = 0/960"
  `[self-report-truncation-awareness]`. Files: `results/traces/agent_step_lengths.parquet`,
  `results/traces/exit_status_sample.parquet`, the two `t2s_night2_20260929T20*` files.
- **Figure:** F1 (platform diagram) only.
- **Gaps:** none beyond those of the sections it previews.

### 4.2 Background and related work (0.75 page)

- **Claim:** prior work characterizes either serving efficiency at fixed memory, or single-vendor out-of-memory
  behaviour, or abstention under missing retrieval; none measures the chain from memory architecture through a
  runtime's default to agent behaviour, across vendors.
- **Positioning** (sources in `docs/RELATED_WORK.md`, `docs/PRIOR_ART.md`, `docs/references.bib`):
  - LLM routing: RouteLLM (`routellm`), Martian (`martian`), NotDiamond (`notdiamond`), RouterBench (named in
    `docs/DEMO_SPEC.md`, NOT yet in `references.bib`; Gap G14). They route on model capability and cost; none conditions
    on a measured, hardware- and runtime-dependent effective context.
  - Cloud-edge routing (`cloud_edge_survey`, `edge_llm_serving`): static placement or per-request offload.
  - Memory-system DSE: MemExplorer (arXiv:2604.16007) sizes memory technology for throughput at fixed correctness.
  - Apple-silicon serving: SiliconBench (arXiv:2609.19169), one memory architecture.
  - KV eviction and compression (H2O, SnapKV, StreamingLLM, LLMLingua): choose what to keep at fixed memory, as a
    designed policy. We measure what a shipped runtime actually discards by default.
  - Missing context and abstention: Sufficient Context (arXiv:2411.06037), Lost in the Middle (arXiv:2307.03172).
  - Runtime issue trackers (ollama #14073, #14116, #12353; llama.cpp #18946, #19745, #1866): the mechanisms exist as
    anecdotes, one vendor and one bug at a time.
- **Figure:** none.
- **Gaps:** RouterBench bib entry (G14); StreamingLLM and Scissorhands are "not independently verified" in
  `docs/PRIOR_ART.md` and need citation checks (G14).

### 4.3 Platforms, harness and methodology (1.25 pages)

- **Claim:** the measurements are reproducible and guarded against the specific artifacts that invalidated earlier
  runs (stale servers, thinking-mode contamination, launch races).
- **Evidence:** platform table (Section 3 of this file); stale-server guard on every llama-server harness
  (`harness/server_guard.py`); validity gates before every agent run: R2 rule and control gates
  `[R2-validation-v2-baseline]`, task-tool gate `[R2-task-tool-gate]`; render-only prompt check for the mechanism run,
  "render/log disagreements 0" `[R2-mechanism-verdict]`; outcome-table canary gates `[x2-v3-canary-gates]`;
  cross-machine output agreement "96.1% text, 99.7% score (combined)" `[R1-check-agreement]`. Statistics convention:
  1 warm-up plus N measured calls unless a harness says otherwise.
- **Figure:** none (a small table of gates per experiment).
- **Gaps:** platform-config register row (G13).

### 4.4 Memory-architecture boundaries (2.00 pages)

**4.4a INTEL-iGPU shared-heap budget and first-principles flip point.**

- **Claim:** a model start on INTEL-iGPU succeeds or fails at one shared-heap limit that also counts host-visible
  buffers, and weights + KV + compute accounting against that limit predicts the measured flip point.
- **Evidence:** "budget boundary 47482-47969 MiB projected, 5 models" `[A-24-budget-boundary]`; "shared-heap L in
  (48112.7, 48124.0] MiB, consistent across all 5 models (92 lower / 22 upper bounds), +247 MiB vs the 47866 MiB
  budget; device-only accounting is inconsistent (max start 47945.2 > min refusal 47562.5 MiB)"
  `[A-24-effective-heap-limit]`; prediction error "A file-size -2890 to -820 tokens (-3.48% to -0.60%); B device-only
  -1092 to +968 tokens (-0.34% to +4.11%); C shared-heap LOO -257 to -26 tokens (-0.11% to -0.01%)"
  `[A-24-flip-point-prediction]`; per-model inputs and results `[A-24-flip-point-inputs]`,
  `[A-24-flip-point-per-model]`. Files: `results/t2s_amech_20260926T181456Z.jsonl`,
  `results/t2s_night2_20260929T034014Z.jsonl` and their `_srv_bis_*` logs.
- **Figure:** F2. Existing `figures/fig_a24_budget_boundary.png` predates the llama-3.3-70b bisection and shows
  projected memory only; needs regeneration with all models of `[A-24-budget-boundary]` ("5 models") and the three
  accountings (A/B/C) against the measured interval.
- **Gaps:** start-only, and beyond the trained context for qwen3-8b, qwen3-30b-a3b-2507 and llama31-8b (the flip is
  the memory flip, not usable context);
  AMD-iGPU equivalent not measured (G5). One driver and boot state; the margin over the reported budget is not
  claimed to transfer.

**4.4b AMD-iGPU spill and crash regimes (MX2).**

- **Claim (not yet citable):** on AMD-iGPU, large contexts pass through a silent-spill regime before a hard failure,
  and requests above the trained context are clamped without error.
- **Evidence:** result files exist (`results/mx2_validation.jsonl`, `results/mx2_spill_stats.jsonl`), FINDINGS
  sections "PRE-REGISTRATION: MX2 validation" and "CORRECTION (2026-10-06)", but NO register row. Nothing from MX2 is
  stated numerically here.
- **Figure:** needs figure: decode tok/s and TTFT with 95% CI versus n_ctx for llama-3.3-70b at the spill-stats
  points, with per-process dedicated/shared GPU memory on a second axis; regime bands FITS / SILENT_SPILL / HARD_FAIL.
- **Gaps:** register rows for MX2 regime boundaries and spill cost (G5); the llama-3.3-70b crash boundary is
  non-deterministic and needs x2_70b_edge_reps (warm-read vs standby-purge, ABAB).

**4.4c NVIDIA-dGPU discrete VRAM spill (C1/C2).**

- **Claim (not yet citable):** on the discrete GPU with the default driver policy, exceeding VRAM causes a silent
  slowdown with a sharp onset, no error and unchanged answers.
- **Evidence:** result files exist (`results/blade_m1_vram_spill_20260925T041415Z.jsonl`,
  `results/blade_c1_spill_sweep_20260925T053651Z.jsonl`, `results/blade_c2_spill_correctness_20260925T110739Z.jsonl`),
  FINDINGS sections C1 and C2, but NO register row.
- **Figure:** F3. Existing `figures/fig_blade_c1_spill.png` (M1 ladder); needs regeneration from the C1 sweep with
  bootstrap CIs and the excess-shared-memory axis, plus the C3 forced-failure point.
- **Gaps:** register rows for C1/C2 (G4); the driver Sysmem Fallback Policy value was not recorded; the
  forced-failure half: Blade night 2 (d) (C3 sysmem fallback).

**4.4d Unified-memory co-runner effects (B3, PX2, M3).**

- **Claim:** on unified memory a CPU-side co-runner slows LLM inference, more on INTEL-iGPU than on AMD-iGPU, and the
  metric where it shows differs by vendor.
- **Evidence:**
  - INTEL-iGPU TTFT under the 12-core co-runner: "mean 1.429x (+42.9%)" `[B3-corunner-6model]`; llama-3.3-70b "1.42x"
    `[P70-TTFT-ratio]`; graded by E-core count, "e6:+14.3%, e8:+30.2%, e8_lp4:+41.4%" `[b4_32b-dose-response]`.
  - AMD-iGPU: "max TTFT ratio on evo-x2: 1.129x (qwen3-8b/S28)" `[PX2-full-ratio-table]`; bandwidth versus compute hog
    gap on TTFT only "2.1-2.6% (B4 vs S4 TTFT gap range across 5 models)" `[PX2-TTFT-gap]`, but separable on decode:
    "B4 0.905x-0.928x of N0; S4 0.987x-1.000x of N0" `[PX2-decode-ratio]`; "qwen3-32b/llama-3.3-70b flat ~83.6W
    across every condition" `[PX2-power]`.
  - Files: `results/t2s_night2_20260928T004924Z.jsonl`, `results/t2s_night2_20260929T034014Z.jsonl`,
    `results/t2s_night2_20260930T135145Z.jsonl`.
- **Figure:** F4. Existing `figures/fig_power_effect_b1_b3.png` and `figures/fig_b4_ecore_threshold.png` (INTEL-iGPU
  only); needs a combined cross-vendor panel: TTFT ratio and decode ratio per co-runner condition, INTEL-iGPU vs
  AMD-iGPU.
- **Gaps:** INTEL-iGPU mechanism is open (package power alone does not explain it; M3 and Section B have no register
  row, so their numbers are not used); INTEL-iGPU has no bandwidth hog, so the two vendors are not measured on the
  same metric: T2S week item 4 (Intel bandwidth hog). One run per condition, small n.

### 4.5 Runtime policy (1.50 pages)

**4.5a Default context selection per device (K1 v3).**

- **Claim:** the same runtime selects the default context window from its own device-detection policy, so two
  integrated GPUs get very different windows for the same model.
- **Evidence:** `[T2S-vs-X2-default-ctx]`: {"qwen3-4b-2507": {"t2s_default_ctx": 4096, "x2_default_ctx": 262144},
  "llama3.1:8b": {"t2s_default_ctx": 4096, "x2_default_ctx": 131072}, "qwen3:8b": {"t2s_default_ctx": 4096,
  "x2_default_ctx": 40960}}; mechanism `[x2-device-detect-mechanism]`: "Vulkan backend drops the AMD iGPU by the same
  policy as Intel, but Ollama also discovers it via ROCm (no opt-out), so evo-x2 uses the iGPU via ROCm while evo-t2s
  (no ROCm path for Intel) falls through to CPU once Vulkan drops it"; `[K1v3-X2-table]`. Files:
  `results/apu_results__t2s_k1_ollama_evo-t2s_20261001T074622Z.jsonl`,
  `results/t2s_k1_ollama_evo-x2_20260930T205515Z.jsonl`, `results/x2_template_mechanism_test.jsonl`.
- **Figure:** F5: default window per (platform, model) overlaid on the CDF of real agent step lengths
  (`figures/fig_trace_cdf.png` exists for the CDF; the overlay is new).
- **Gaps:** NVIDIA-dGPU default: Blade night 1 (a) (K1). Ollama version differs between the two iGPU platforms
  (Section 3). The `OLLAMA_IGPU_ENABLE=1` control on INTEL-iGPU has no register row (G12).

**4.5b Overflow behaviour.**

- **Claim:** when a prompt exceeds the loaded window, the runtime returns success and silently keeps half the window.
- **Evidence:** "processed = num_ctx/2 + 2 in all 4 checked cases: 4096->2050; 32768->16386; 40960->20482;
  8192->4098. Mechanism: Ollama's own '--context-shift --keep 4' llama-server flag" `[ollama-overflow-keeps-half]`.
- **Figure:** folded into F7.
- **Gaps:** NVIDIA-dGPU: Blade night 1 (a).

**4.5c Install-path effects.**

- **Claim:** the same weights behave differently as an agent depending on how they were installed into the runtime.
- **Evidence:** `[R2-install-path-4b-comparison]`: "change bare -> library template: call-1 native 11/30 -> 2/30,
  text tool calls 30/30 -> 30/30, task tool correct 5/30 -> 5/30"; bare tag alone `[R2-install-path-4b-bare]`;
  template source per outcome row `[x2-chat-template-source]`. Files: `results/x2_r2_validation_v2b.jsonl`,
  `results/x2_r2_validation_v2c.jsonl`, `results/x2_chat_template_sources.jsonl`.
- **Figure:** none (a small table).
- **Gaps:** both install paths fail the task-tool gate for this model, so the finding is "behaviour follows the
  install path", not "one path works". The K1 item-1b result (bare create hard-errors on overflow, library pull
  truncates silently, same file) has no register row (G12).

**4.5d TTFT first-call stall.**

- **Claim:** the first request on a new prompt waits for the previous request's KV state to be moved, at a fixed
  per-machine rate, which dominates TTFT error on AMD-iGPU.
- **Evidence:** "evo-x2: 658/919 rep-0 calls >1.5x their rep 1, stall median 15.7-40.8 s, displaced KV state
  29.0-29.3 MiB/s" and "evo-t2s: 242/424 rep-0 calls >1.5x their rep 1, stall median 1.9-4.0 s, displaced KV state
  134.0-141.0 MiB/s" `[ttft-first-call-stall]`; removing those rows: "drop_first_call_rep0 0.95-1.00 (6 models)"
  `[ttft-x2-refit-by-hypothesis]`; out-of-sample check "median abs % error 0.4-2.3% with the stall term vs 48.3-89.6%
  without, 6 models" `[ttft-x2-section0-stall-check]`; per cell `[ttft-first-call-stall-per-cell]`.
- **Figure:** optional (cut first if space is short).
- **Gaps:** the mechanism (llama-server host-RAM prompt cache) is inferred, not confirmed: a `--cache-ram 0` control
  is needed and NO queue job exists for it (G11). `[ttft-cross-machine-transfer]` and `[ttft-few-point-calibration]`
  were computed on pooled rows and must be recomputed on rep>=1 rows before they are cited (G11).

### 4.6 Agent consequences (1.50 pages)

**4.6a R2 real run.**

- **Claim:** in multi-turn tool-using sessions the runtime drops conversation history with no error at every
  truncating window, and rule compliance is lost only at the smallest window.
- **Evidence:** "killed=False; ollama_default 3/6 silent; ollama_ctx_32768 4/6 silent; ollama_ctx_4096 6/6 silent"
  `[R2-real-v1-kill-criterion]`; gated on metrics at or above the baseline bar: "num_ctx_32768 1/6 silent, 0 after
  window exceeded; num_ctx_4096 6/6 silent, 6 after window exceeded; ollama_default 1/6 silent, 0 after window
  exceeded" `[R2-real-v1-gated-kill]`; "num_ctx_4096 llama3.1:8b 0/0/0/0/0 of 3; num_ctx_4096 qwen3:14b 0/0/0/0/0 of 3"
  `[R2-real-v1-survival]`; no error field in any of "18 sessions, 720 turns" (n column) `[R2-real-v1-first-events]`.
  File: `results/x2_r2_real_v1.jsonl`.
- **Figure:** F6: per-tier session survival curve (intact sessions vs turn), v1 now, v1b when synced.
- **Gaps:** AMD-iGPU only, models llama3.1:8b and qwen3:14b, sessions "of 3" per cell `[R2-real-v1-survival]`.
  Fill: x2_r2_real_v1b (rows `R2-real-v1b-*` PENDING), T2S week
  item 2 (R2 native Intel), Blade night 1 (b) (R2 native).

**4.6b Mechanism.**

- **Claim:** the runtime trims whole messages and keeps the system prompt; at the smallest window llama.cpp context
  shift and a token-level cut then discard the start of the prompt, system prompt included, still with success
  status.
- **Evidence:** `[R2-mechanism-lowlevel]`: "ollama_ctx_16384: 54/80 calls message-truncated; 0 context shifts; 0 token
  cuts" and "ollama_ctx_4096: 76/80 calls message-truncated; 34 context shifts (first turn/call (4, 2), n_keep [5],
  n_discard [2045], 29 on the final-answer call); 1 token cuts (token cut limit 2050 keep 4 prompt 4097 -> 2050)";
  `[R2-mechanism-verdict]`: "token-level cut seen True, context shift seen True, render/log disagreements 0".
  File: `results/x2_r2_mechanism.jsonl`.
- **Figure:** F7: per-turn event raster per tier (message drops, context shifts, token cut), one row per tier.
- **Gaps:** one model, one session per tier (`[R2-mechanism-verdict]` n column: "5 sessions"), AMD-iGPU. Fill: T2S week item 6 (mechanism run), Blade night 2 (e)
  (mechanism); more seeds through x2_r2_real_v1b's mechanism logging.

**4.6c Outcome table.**

- **Claim (pending):** the runtime and its defaults change task-level outcomes for the same model on the same
  machine.
- **Evidence so far:** "total 1078/3600" `[x2-v3-progress]` (moving value; re-paste at each register regeneration); partial per-cell scores `[x2-v3-scores]` and error causes
  `[x2-v3-error-causes]`. File: `results/x2_outcome_table_v3.jsonl`. Not citable as a result until complete.
- **Figure:** none until complete (candidate replacement for F6 if R2 v1b slips).
- **Gaps:** x2_outcome_table_v3 completion; T2S week item 5 (outcome-table subset).

### 4.7 Predictor and mitigation (1.00 page)

- **Claim (at risk):** a predictor from the loaded window, the per-turn prompt growth and the runtime's trimming rule
  flags the turn at which the system prompt will be lost, before it happens.
- **Evidence:** none yet. Inputs exist: `[R2-mechanism-lowlevel]`, `[ollama-overflow-keeps-half]`.
- **Claim (pending):** the pre-registered mitigation keeps rule compliance at the smallest window.
- **Evidence:** none yet; pre-registration in `docs/FINDINGS.md` / the job's own docstring.
- **Figure:** F8: rule-compliant session survival at the smallest window, with vs without mitigation.
- **Gaps:** predictor build and its validation against x2_r2_real_v1b (G6); x2_r2_mitigation_v1; T2S week item 3
  (mitigation at 4096); Blade night 1 (c) (mitigation).

### 4.8 Threats to validity (0.50 page)

- **Scorer validity:** held-out agreement "kappa 0.840 (analytic 95% CI 0.760-0.920, bootstrap 95% CI 0.755-0.915);
  raw agreement 136/150" `[kappa-heldout]`. The earlier in-sample agreement after scorer fixes has no register row
  and is not cited; it was partly correlated error between two raters: "scorer vs classify_human_label: kappa 0.988"
  `[kappa-heldout-fn-annotator]`. Residual errors are one-directional: "14 disagreements: ambiguous 4, scorer_bug 10;
  direction (scorer->annotator) FABRICATED->CORRECT 1, FABRICATED->REFUSED 13" `[kappa-heldout-disagreements]`.
  Proposed scorer fixes are not applied (applying them makes this sample in-sample). One annotator channel; a second
  human rater is pending (G10).
- **Single seeds and small n:** R2 mechanism one seed per tier; co-runner runs one run per condition; K2 quality at
  ceiling: "316/316 scored items = 1.0 (1 unscored)" `[K2-x2-score-ceiling]` and responsiveness resolution "max
  step/+8GB ratio 1.937x (tolerance 2.0x)" `[K2-x2-responsiveness-resolution]`, so "13 kill-criterion evaluations, 13
  passed, 0 violated" `[K2-x2-kill-criterion]` is a weak null.
- **Thinking contamination, found and fixed:** reasoning models without suppression produced no answer:
  "llama_server/qwen3-8b/baseline: works=False reasoning=[980, 774, 766] content=[0, 0, 0]" vs
  "llama_server/qwen3-8b/production: works=True reasoning=[0, 0, 0]" `[x2-thinking-verify]`.
- **Invalid-race tagging:** the invalidated weekend run is tagged, not deleted: "invalid_race 412
  (system_ollama_exe_resolution 405, concurrent_job_contention_unverified 7); invalid_thinking 330; invalid_infra_oom
  207; valid reusable 373 of 1327 outcome rows" `[x2-weekend-tag-counts]`.
- **Infrastructure failures on INTEL-iGPU:** time-ordered allocation failures, not a prompt-length boundary
  `[t2s-outcome-allocation-failures]`, `[t2s-llamaserver-load-failures]`; cause unverified until the machine is back.
- **Version skew and backend numerics:** Ollama versions differ across platforms (Section 3); CUDA and Vulkan differ
  on one borderline probe (`docs/THREATS.md` section 21, no register row).
- **Install path as a confound:** bare-GGUF Ollama legs are marked "[bare template]" in `[x2-v3-scores]`.

### 4.9 Conclusion (0.50 page)

- **Claim:** memory architecture sets where local inference stops, runtime defaults set what a long agent session
  silently loses, and both are measurable and predictable before deployment.
- **Gaps:** depends on 4.4b, 4.4c, 4.6 replication and 4.7.

---

## 5. Claim table

Platforms: I = INTEL-iGPU, A = AMD-iGPU, N = NVIDIA-dGPU. Status: verified / pending (job named) / at risk.

| id | claim | evidence (register rows) | platforms | seeds and CI status | likely reviewer objection | our answer | status |
|---|---|---|---|---|---|---|---|
| P1-01 | INTEL-iGPU start boundary is one shared heap incl. host-visible buffers; first-principles flip prediction | A-24-budget-boundary, A-24-effective-heap-limit, A-24-flip-point-prediction, A-24-flip-point-inputs, A-24-flip-point-per-model | I | deterministic start/refuse bisection; bounds, no CI; one boot state | "Running out of memory fails; obvious." | Device-only accounting cannot separate starts from refusals ("max start 47945.2 > min refusal 47562.5 MiB" `[A-24-effective-heap-limit]`); only the shared-heap accounting predicts the flip | verified |
| P1-02 | AMD-iGPU silent-spill then hard-fail regimes; clamping above trained context | none yet | A | spill-stats reps exist but are unregistered; 70B boundary non-deterministic (FINDINGS) | "One machine, unclear crash." | Report the 70B boundary as a distribution after edge reps | pending (x2_70b_edge_reps; register rows G5) |
| P1-03 | NVIDIA-dGPU silent spill with sharp onset, answers unchanged; driver setting flips it to a hard failure | none yet | N | C1 has per-context bootstrap CIs in FINDINGS, unregistered; C3 not run | "Known driver feature." | The cross-vendor contrast with unified memory is the contribution, plus the measured onset and the C3 flip | pending (Blade night 2 (d); register rows G4) |
| P1-04 | CPU co-runner slows INTEL-iGPU inference, graded by E-core count | B3-corunner-6model, P70-TTFT-ratio, b4_32b-dose-response | I | small n per condition, one run; bootstrap CIs narrow because n is small | "Thermal or power throttling, trivial." | Package power alone does not explain it (Section B, no register row yet); mechanism stated as open | verified (magnitude); mechanism open |
| P1-05 | On AMD-iGPU the co-runner TTFT cost is smaller and bandwidth vs compute separate only on decode | PX2-full-ratio-table, PX2-TTFT-gap, PX2-decode-ratio, PX2-power, PX2-thermal | A | small n per condition, one run; n column "5" models `[PX2-decode-ratio]` | "Different metrics per vendor is not a comparison." | Agreed; T2S week item 4 adds the Intel bandwidth hog on the same metrics | verified (A); cross-vendor pending T2S week item 4 |
| P1-06 | First call on a new prompt stalls at a fixed per-machine rate set by displaced KV state | ttft-first-call-stall, ttft-first-call-stall-per-cell, ttft-x2-refit-by-hypothesis, ttft-x2-section0-stall-check | I, A | many calls; out-of-sample check on held-out section; no control run | "Prompt-cache behaviour is a configuration detail." | It explains the TTFT model error on one platform; but the mechanism needs the control | at risk (no queued `--cache-ram 0` control) |
| P1-07 | Runtime device policy sets the default window: small on INTEL-iGPU, model-native on AMD-iGPU | T2S-vs-X2-default-ctx, x2-device-detect-mechanism, K1v3-X2-table | I, A | deterministic configuration reads; n column "3" models `[T2S-vs-X2-default-ctx]` | "Documented Ollama behaviour, bug report level." | Issue trackers document the tiering; nobody traced it through device detection to agent loss across vendors | verified (I, A); N pending Blade night 1 (a) |
| P1-08 | Overflow is silent: success status, half the window kept | ollama-overflow-keeps-half | I, A | "all 4 checked cases" `[ollama-overflow-keeps-half]`; deterministic | "Context shift is a llama.cpp feature." | Yes; the finding is that the shipped default applies it silently and what it removes | verified |
| P1-09 | Install path of identical weights changes tool-call behaviour | R2-install-path-4b-comparison, R2-install-path-4b-bare, x2-chat-template-source | A | "30 + 30 baseline turns" `[R2-install-path-4b-comparison]`; one model | "You used the wrong template." | That is the finding: the install path picks the template silently; both paths fail the gate | verified (narrow, one model) |
| P1-10 | Models do not report that context was lost | self-report-truncation-awareness | I, A | "0/960" `[self-report-truncation-awareness]` | "Models can't know what they don't see." | Agreed; it is why the failure is silent end to end | verified |
| P1-11 | Real agent sessions lose history silently at every truncating window; rules lost only at the smallest | R2-real-v1-kill-criterion, R2-real-v1-gated-kill, R2-real-v1-survival, R2-real-v1-first-events | A | llama3.1:8b and qwen3:14b, sessions "of 3" per cell `[R2-real-v1-survival]`, no CI | "Few models, one machine." | x2_r2_real_v1b adds models and seeds; T2S and Blade replicate | verified (A); generalization pending x2_r2_real_v1b, T2S week item 2, Blade night 1 (b) |
| P1-12 | Mechanism: message trimming keeps the system prompt; context shift and token cut discard it at the smallest window | R2-mechanism-verdict, R2-mechanism-lowlevel | A | one model, one session per tier (n column "5 sessions" `[R2-mechanism-verdict]`); render/log check passed | "Single seed." | The mechanism is deterministic given the prompt sizes; replication across platforms is queued | verified (A); pending T2S week item 6, Blade night 2 (e) |
| P1-13 | Real agent workloads exceed small defaults | trace-fraction-exceeds-4096, trace-context-exit-rate, uncensored-trace-32k-crossing, pack-trace-weighted-stats | n/a (trace data) | full trace sets | "Synthetic agents." | Public real trajectories, censoring handled | verified |
| P1-14 | Runtime choice changes task outcomes for the same model and machine | x2-v3-progress, x2-v3-scores, x2-v3-error-causes | A (I subset pending) | incomplete | "Incomplete table." | Report only when complete | pending (x2_outcome_table_v3; T2S week item 5) |
| P1-15 | Co-running app memory pressure does not degrade quality before a loud failure | K2-x2-kill-criterion, K2-x2-score-ceiling, K2-x2-responsiveness-resolution, K2-x2-clean-failures | A | measures at ceiling | "Your test could not detect degradation." | Correct; reported as a weak null only | at risk (x2_k2_pause_resume_v1; harder task set unplanned) |
| P1-16 | A predictor flags the turn at which the system prompt will be lost | none yet | (A first) | no data | "Post hoc." | Validate on x2_r2_real_v1b sessions not used to build it | at risk (being built) |
| P1-17 | Pre-registered mitigation keeps rule compliance at the smallest window | none yet | A, I, N planned | pre-registered | "Mitigation is just a bigger window." | Pre-registration fixes the comparison before data | pending (x2_r2_mitigation_v1; T2S week item 3; Blade night 1 (c)) |
| P1-18 | A single TTFT fit does not transfer across machines | ttft-cross-machine-transfer, ttft-few-point-calibration | I, A | pooled rows include the stall | "Confounded by the stall." | Must recompute on rep>=1 rows before citing | at risk (recompute, G11) |

Tally: verified 10 (P1-01, 04, 05, 07, 08, 09, 10, 11, 12, 13), pending 4 (P1-02, 03, 14, 17), at risk 4 (P1-06,
15, 16, 18). Of the 10 verified, P1-05, 07, 11 and 12 are verified on fewer platforms than the paper claims and need
the named runs to support the cross-vendor wording.

---

## 6. "Obvious?" defense

One sentence per headline finding on why `docs/PRIOR_ART.md` does not already show it.

- **Shared-heap boundary (P1-01):** llama.cpp #18946 reports Intel Vulkan `ErrorOutOfDeviceMemory` as an anecdote, and
  MemExplorer sizes memory technology for throughput, but no prior work identifies which buffers count against an
  iGPU heap or predicts the start boundary from first principles.
- **Cross-vendor failure regimes (P1-02, P1-03):** `docs/PRIOR_ART.md` claim (c) finds only a single-vendor NVIDIA
  blog benchmark and an Apple-only paper (SiliconBench), so a common-protocol matrix across unified and discrete
  memory from three vendors is not shown; the contribution is the matrix, not any single failure mode.
- **Co-runner effects on unified memory (P1-04, P1-05):** prior co-runner and memory-pressure work measures speed on
  one platform; none contrasts where the effect appears (TTFT vs decode) across two unified-memory vendors.
- **Default window from device detection (P1-07):** the Ollama tiering is documented in issue trackers (#14073,
  #14116, #12353) as user pain, but no measured study traces device detection to the default window to agent
  behaviour, and none shows two iGPUs diverging through a backend-specific opt-out.
- **Silent overflow and system-prompt loss (P1-08, P1-11, P1-12):** KV eviction work (H2O, SnapKV, StreamingLLM)
  studies designed retention policies at fixed memory, and Sufficient Context studies information never retrieved;
  neither measures what a shipped runtime's default discards in a live agent session, nor that the system prompt is
  what goes.
- **Install path (P1-09):** no prior work found treats the runtime's install path of identical weights as an
  independent variable for agent behaviour.
- **Mitigation and predictor (P1-16, P1-17):** routers (RouteLLM, Martian, NotDiamond, RouterBench) route on model
  capability and cost; none conditions on a predicted, runtime-specific context loss.

Caveat: StreamingLLM and Scissorhands are marked "not independently verified" in `docs/PRIOR_ART.md`; check before
citing (G14).

---

## 7. Figure list (8)

| fig | content | data source | data exists? |
|---|---|---|---|
| F1 | Platform and memory-architecture diagram: three platforms, where KV, weights and host-visible buffers live, which runtime backend is used | `docs/HARDWARE.md`, Section 3 of this file | yes (drawing only) |
| F2 | INTEL-iGPU flip point: measured (last_ok, first_fail] vs accountings A/B/C, all models in `[A-24-budget-boundary]` | `[A-24-flip-point-per-model]`, `[A-24-effective-heap-limit]` | yes; existing `figures/fig_a24_budget_boundary.png` needs regeneration |
| F3 | NVIDIA-dGPU spill onset: TTFT and decode slowdown vs context with spilled fraction; C3 point | C1/C2 files (no register row); Blade night 2 (d) | partly: C1/C2 yes, C3 no; existing `figures/fig_blade_c1_spill.png` is the M1 ladder |
| F4 | Co-runner effects by vendor: TTFT and decode ratio per condition, INTEL-iGPU vs AMD-iGPU | `[B3-corunner-6model]`, `[b4_32b-dose-response]`, `[PX2-full-ratio-table]`, `[PX2-decode-ratio]`; T2S week item 4 | partly: Intel decode under a bandwidth hog missing |
| F5 | Default window per platform and model over the agent step-length CDF | `[T2S-vs-X2-default-ctx]`, `[trace-fraction-exceeds-4096]`; Blade night 1 (a) | partly: I and A yes, N no; `figures/fig_trace_cdf.png` exists |
| F6 | R2 session survival by tier | `[R2-real-v1-survival]`; x2_r2_real_v1b; T2S week item 2; Blade night 1 (b) | partly: v1 yes, v1b and other platforms no |
| F7 | Per-turn truncation-event raster by tier (message drop, context shift, token cut) | `[R2-mechanism-lowlevel]`; T2S week item 6; Blade night 2 (e) | yes for A, one session per tier |
| F8 | Mitigation: rule-compliant survival at the smallest window, with vs without | x2_r2_mitigation_v1; T2S week item 3; Blade night 1 (c) | no |

Backup figure if one slips: first-call TTFT stall per cell (`[ttft-first-call-stall-per-cell]`, data exists).

---

## 8. Gaps: what must land before 2026-11-01, ordered by impact on acceptance

1. **G1. Agent result on more than one platform and more seeds.** Without it the headline is one AMD machine and
   two models (llama3.1:8b, qwen3:14b). Fill: x2_r2_real_v1b (rows `R2-real-v1b-*` PENDING); T2S week item 2 (R2 native Intel); Blade night 1 (b)
   (R2 native). Prerequisite for the Intel runs: T2S week item 1 (re-enable checks).
2. **G2. Mechanism on more than one platform and seed.** Fill: T2S week item 6 (mechanism run); Blade night 2 (e)
   (mechanism).
3. **G3. Mitigation result.** Contribution 4 is empty without it. Fill: x2_r2_mitigation_v1 (after x2_r2_real_v1b);
   T2S week item 3 (mitigation at 4096); Blade night 1 (c) (mitigation).
4. **G4. Discrete-memory half of the architecture claim.** Register rows for existing C1/C2 files (analysis only),
   plus Blade night 2 (d) (C3 sysmem fallback). Without it, "three vendors, two memory architectures" is not
   supported in the text and the abstract sentence on discrete spill is cut.
5. **G5. AMD-iGPU boundary rows.** Register rows for `results/mx2_validation.jsonl` and `results/mx2_spill_stats.jsonl`
   (analysis only), plus x2_70b_edge_reps for the non-deterministic 70B boundary.
6. **G6. Predictor built and validated** on sessions not used to build it (x2_r2_real_v1b). No queue job; analysis
   work. At risk on time.
7. **G7. Outcome table.** x2_outcome_table_v3 completion; T2S week item 5 (outcome-table subset).
8. **G8. NVIDIA-dGPU default context and overflow.** Blade night 1 (a) (K1).
9. **G9. Same-metric co-runner comparison.** T2S week item 4 (Intel bandwidth hog).
10. **G10. Second human rater** for the held-out kappa subset (labels pending), and a decision on the proposed scorer
    fixes (needs a third fresh sample if applied).
11. **G11. TTFT stall control and recompute.** A `--cache-ram 0` control (no queue job exists); recompute
    `[ttft-cross-machine-transfer]` and `[ttft-few-point-calibration]` on rep>=1 rows.
12. **G12. Register rows for findings stated in FINDINGS but not registered:** K1 item 1b (bare create hard-errors,
    library pull truncates), the `OLLAMA_IGPU_ENABLE=1` control, M3 / overnight Section B co-runner mechanism rows,
    Phase D and C1-lock rows.
13. **G13. Platform-config register row** and the missing version fields (Section 3); matched-version re-check of
    the default-context comparison during T2S week item 1.
14. **G14. Citations:** RouterBench into `docs/references.bib`; verify StreamingLLM and Scissorhands.
15. **G15. K2 strengthening** (harder task set, sub-tick responsiveness probe) and x2_k2_pause_resume_v1. Lowest
    priority for this paper; P1-15 can be dropped without hurting the story.

Not resolved here: the submission deadline (Dec 2 vs mid-December estimate) and the 2027 page limit.

---

## Appendix: claim-to-evidence index

| claim | section | register rows | figure | status | gap |
|---|---|---|---|---|---|
| P1-01 shared-heap boundary | 4.4a | A-24-budget-boundary, A-24-effective-heap-limit, A-24-flip-point-prediction, A-24-flip-point-inputs, A-24-flip-point-per-model | F2 | supported (I) | AMD equivalent (G5) |
| P1-02 AMD spill/crash | 4.4b | none | needs figure (MX2 regimes) | pending x2_70b_edge_reps | G5 |
| P1-03 NVIDIA silent spill | 4.4c | none | F3 | pending Blade night 2 (d) | G4 |
| P1-04 Intel co-runner | 4.4d | B3-corunner-6model, P70-TTFT-ratio, b4_32b-dose-response | F4 | supported (magnitude) | mechanism rows (G12) |
| P1-05 AMD co-runner by metric | 4.4d | PX2-full-ratio-table, PX2-TTFT-gap, PX2-decode-ratio, PX2-power, PX2-thermal | F4 | supported (A) | T2S week item 4 (G9) |
| P1-06 first-call stall | 4.5d | ttft-first-call-stall, ttft-first-call-stall-per-cell, ttft-x2-refit-by-hypothesis, ttft-x2-section0-stall-check | backup | partial | G11 |
| P1-07 default window by device policy | 4.5a | T2S-vs-X2-default-ctx, x2-device-detect-mechanism, K1v3-X2-table | F5 | supported (I, A) | Blade night 1 (a) (G8), version skew (G13) |
| P1-08 silent overflow | 4.5b | ollama-overflow-keeps-half | F7 | supported | Blade night 1 (a) |
| P1-09 install path | 4.5c | R2-install-path-4b-comparison, R2-install-path-4b-bare, x2-chat-template-source | none | supported (one model) | K1 1b row (G12) |
| P1-10 no self-report | 4.1 | self-report-truncation-awareness | none | supported | none |
| P1-11 R2 silent loss | 4.6a | R2-real-v1-kill-criterion, R2-real-v1-gated-kill, R2-real-v1-survival, R2-real-v1-first-events | F6 | partial (A only) | G1 |
| P1-12 mechanism | 4.6b | R2-mechanism-verdict, R2-mechanism-lowlevel | F7 | partial (A, one seed) | G2 |
| P1-13 workload relevance | 4.1 | trace-fraction-exceeds-4096, trace-context-exit-rate, uncensored-trace-32k-crossing, pack-trace-weighted-stats | F5 | supported | none |
| P1-14 outcome table | 4.6c | x2-v3-progress, x2-v3-scores, x2-v3-error-causes | none | pending x2_outcome_table_v3 | G7 |
| P1-15 app memory pressure | 4.8 | K2-x2-kill-criterion, K2-x2-score-ceiling, K2-x2-responsiveness-resolution, K2-x2-clean-failures | none | partial (weak null) | G15 |
| P1-16 predictor | 4.7 | none | F8 (shared) | pending (being built) | G6 |
| P1-17 mitigation | 4.7 | none | F8 | pending x2_r2_mitigation_v1 | G3 |
| P1-18 TTFT transfer | 4.5d | ttft-cross-machine-transfer, ttft-few-point-calibration | none | partial | G11 |
