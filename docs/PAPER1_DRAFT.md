# Paper 1 Draft Skeleton (ISPASS 2027 target, anonymized)

Status: outline for advisor review, 2026-10-08; updated 2026-10-10 (NVIDIA-dGPU default context, overflow and
mechanism rows; predictor result; open finding O1); updated 2026-10-14 (R2 counts recomputed on the strengthened run
v1 + v1b, excess over the default tier with bootstrap CIs; confound check result for O1; mitigation verdict); updated
2026-10-20 (prose drafts of Sections 4.3, 4.6a, 4.6b and 4.7; the outline text they replace is in the appendix
"Outline (superseded 2026-10-20)"; prose marks each number with its register id in an HTML comment); updated
2026-10-22 (4.3, 4.6a, 4.6b and 4.7 brought up to the register of commit 7fc16ff: qwen3:8b at 10 seeds, the protocol
check on qwen3:14b alone, B5 detail, qwen3:32b default tier; qwen3:14b at 4096 explained by `[R2-qwen3-14b-4096-reconciliation]`, G16; trimming's side effect;
design constants commented with their source file). Not a
submission. Double-blind is assumed: this file names no
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

**Abstract (prose draft, 2026-10-22; word count by `wc -w` on the paragraph below with HTML comments removed):**

When a local agent's conversation outgrows the context window, runtimes such as Ollama and llama.cpp trim it and
still report success. We measure what is lost
with pre-registered experiments on an AMD APU, an NVIDIA laptop GPU and an Intel laptop. At windows of 16384 tokens
and below, no session of three 8B to 14B models kept its full history <!-- R2-real-v1b-survival -->, and models
up to 70B lose history once the window is exceeded <!-- H-scale-verdict --> <!-- H-scale-family-verdict -->. Rule loss
appears in the 8B and 14B models (5 seeds) and in qwen3:32b (2 of 3 sessions at 4096) <!-- H-scale-verdict -->, and
not in llama3.3:70b or mistral-small3.2:24b (0 of 3 each) <!-- H-scale-verdict --> <!-- H-scale-family-verdict -->.
History loss appears in all of them. On a tool-using task workload, qwen3:14b's wrong-task rate rises from 11.3% at
its default window to 42.0% at 4096, and every wrong task is silent <!-- x2-evalset-tasks-real-v2-wrong-rate -->.
Runtime defaults decide how often this happens: an unmodified install loaded a 4096-token window on an Intel CPU
<!-- t2s-r2-real-cpu --> and 32768 on the same laptop's integrated GPU <!-- t2s-r2-real-igpu -->. The backend matters
too: with identical model files and settings, llama3.1:8b wrote tool calls as text at an untruncated window on Vulkan
and CUDA but not on ROCm <!-- x2-r2-llama-val-32k-vulkan --> <!-- blade-r2-gate-v2 --> <!-- x2-r2-llama-val-32k-offload -->.
The fixes are partial. Setting the window per device removes the hidden default; KV-cache quantization cut session
time by 65% to 85% with no measured loss of rules or recall <!-- cost-cut-config-time --> <!-- b5-fa-control -->
<!-- b5-kv-recall-f16-rerun -->; client-side trimming removes the in-context examples a model imitates; and
`num_keep` protects the system prompt only where the runtime shifts or cuts tokens.

Claims behind each abstract sentence (register rows; all VERIFIED on 2026-10-22):

- History lost at 16384 and below, 8B to 14B: `[R2-real-v1b-survival]` (AMD-iGPU, five seeds per cell). Larger models
  lose history once the window is exceeded: `[H-scale-verdict]` (qwen3:32b, llama3.3:70b), `[H-scale-family-verdict]`
  (mistral-small3.2:24b), three sessions per cell.
- Rule loss model-dependent: operator wording, `[H-scale-verdict]`, `[H-scale-family-verdict]`; 8B and 14B at five
  seeds from `[R2-v1b-gated-per-model]` (Section 4.6a).
- Wrong-task rate and silence: `[x2-evalset-tasks-real-v2-wrong-rate]` (qwen3:14b alone; see Section 4.8).
- Defaults decide: `[t2s-r2-real-cpu]` vs `[t2s-r2-real-igpu]`; this contrast changes backend and window together.
- Backend: `[x2-r2-llama-val-32k-vulkan]` (Vulkan, 2 of 20 canaries missed), `[blade-r2-gate-v2]` (CUDA),
  `[x2-r2-llama-val-32k-offload]` (ROCm with partial offload, 0 of 20).
- Per-device window: `[window-override-methods]` (which settings persist), `[t2s-r2-real-igpu]`.
- KV quantization: `[cost-cut-config-time]` (85% and 65%, three seeds), attributed to KV precision, not flash
  attention, by `[b5-fa-control]`; recall check `[b5-kv-recall-f16-rerun]` (no difference shown, descriptive).
- Trimming side effect: `[R2-qwen3-14b-4096-reconciliation]`, `[R2-mitigation-v1-verdict]`. num_keep scope:
  `[num-keep-guardrail-test]`, `[x2-evalset-tasks-real-v2-gated-excess]`.
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
   `[R2-install-path-4b-comparison]`. On the NVIDIA-dGPU ("device VRAM 8188 MiB" `[blade-k1-layer-split]`) the same
   runtime picks a small default for both measured models and answers over-length prompts with HTTP 200 and the half-window evaluation
   `[blade-k1-layer-split]`.
   - INTEL-iGPU and AMD-iGPU: **verified now.**
   - NVIDIA-dGPU default context and overflow: **verified now** (Blade night 1 (a), K1).
3. **These defaults reach agents, silently, and we explain the mechanism.** In real tool-using sessions history is
   lost with no error at every truncating window, and silent failures after the window is exceeded rise as the window
   shrinks, from none at 32768 to most sessions at 4096, over a default tier that has none after the window
   (`[R2-real-v1b-gated-kill]`, `[R2-v1b-excess-over-default]`, Section 4.6a); message-level trimming keeps the system
   prompt, and llama.cpp context shift plus a token-level cut discard it at the smallest window
   `[R2-mechanism-lowlevel]`. The message-level step and the context shift at the smallest window replicate on
   NVIDIA-dGPU; the token cut was not seen there `[blade-mechanism-4096]`.
   - Real-run result on AMD-iGPU, three models x five seeds per tier, five tiers: **verified now.**
   - Mechanism on AMD-iGPU (all tiers) and NVIDIA-dGPU (smallest window), one session per tier: **verified now.**
   - INTEL-iGPU: **pending run** T2S week item 2 (R2 native Intel) and T2S week item 6 (mechanism run). NVIDIA-dGPU
     real run: Blade night 1 (b) did not run (the validation gate failed); the confound check now attributes that
     failure to the NVIDIA-dGPU backend or its CPU offload, not to the window (`[x2-r2-llama-val-32k]`, Section 5b), and
     a follow-up that separates backend from offload is pending; other NVIDIA-dGPU mechanism tiers: **pending run**
     Blade night 2 (e).
4. **A predictor for system-prompt loss and a pre-registered mitigation.**
   - Context-shift predictor: **verified now** on AMD-iGPU, one session per tier `[R2-shift-predictor-confusion]`;
     applied as a prediction (not an observation) to the real run `[R2-shift-predicted-real-v1]`.
   - Mitigation: **verified now** on AMD-iGPU, pre-registered verdict refuted `[R2-mitigation-v1-verdict]`: the obvious
     fix (client-side trimming that always keeps the system prompt) works for one model and fails for another at the
     smallest window (Section 4.7). INTEL-iGPU: **pending run** T2S week item 3. NVIDIA-dGPU: Blade night 1 (c) did
     not run (O1).

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
| Ollama | 0.33.2 (`docs/T2S_CHANGELOG.md`; K1 v3 run) | 0.34.4, bundled llama.cpp b11081 (`docs/R2_DESIGN.md`, `docs/X2_CHANGELOG.md`) | 0.34.4 pinned side by side for Blade night 1 (K1, R2 validation, mechanism; `docs/FINDINGS.md`, Blade night 1); earlier 0.34.0 and 0.32.9 (`docs/THREATS.md` section 18) |
| Ollama GPU backend actually used | none: iGPU dropped by policy, CPU fallback, unless `OLLAMA_IGPU_ENABLE=1` (`[x2-device-detect-mechanism]`) | ROCm (`[x2-device-detect-mechanism]`) | CUDA (assumed; not recorded) |
| OS | Windows 11 (build MISSING) | Windows build 10.0.26100 (`docs/HARDWARE.md`) | Windows 11 Home 10.0.26200 (`docs/FINDINGS.md`) |

Missing and must be recorded before submission (Gap G13):

- INTEL-iGPU Windows build; the llama.cpp build bundled in Ollama 0.33.2.
- NVIDIA-dGPU: the CUDA Sysmem Fallback Policy value during C1/C2 ("UNKNOWN" in `docs/FINDINGS.md`); the Ollama
  backend actually used; whether Ollama 0.34.0 there passes the same context-shift flags.
- Version skew: INTEL-iGPU Ollama results are 0.33.2; AMD-iGPU and the NVIDIA-dGPU night 1 results are 0.34.4. The
  cross-platform default context comparison (`[T2S-vs-X2-default-ctx]`) therefore crosses an Ollama version as well as
  a device for INTEL-iGPU only; the AMD-iGPU vs NVIDIA-dGPU comparison is on one version. T2S week
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

### 4.1 Introduction (prose draft, 2026-10-21; 1.00 page)

<!-- Convention as in 4.3: measured numbers carry their register id; design constants carry "design: <source file>". -->

Runtimes such as Ollama and llama.cpp serve open-weight models of 8B to 70B parameters on laptops and small APU
workstations, and local agents drive them for tens of turns, appending every tool result to a conversation that only
grows inside a fixed context window. When the conversation outgrows it, the runtime, not the agent, decides what to
remove, and it does so silently: the request still returns HTTP 200
<!-- blade-k1-layer-split -->, the answer is well formed, and the model does not mention the loss (0/960
<!-- self-report-truncation-awareness --> self-reports in our probes mentioned truncation).

A motivating case shows how easily a user lands in this regime. On an Intel laptop, an unmodified Ollama install
served two models with a 4096 <!-- t2s-r2-real-cpu --> token window, because device detection drops the integrated GPU
and falls back to the CPU <!-- x2-device-detect-mechanism -->. Every agent session crossed that window at turn [3]
<!-- t2s-r2-real-cpu --> and then failed silently, 5/5 <!-- t2s-r2-real-cpu --> sessions for each model. With the
integrated GPU enabled, the same machine, runtime and models loaded 32768 <!-- t2s-r2-real-igpu --> tokens and no
failure came after the window. Nobody chose 4096; the runtime did.

This paper asks what a local agent loses when its conversation exceeds the context window, whether anything tells the
user, and what decides how often it happens. We answer with pre-registered experiments on an AMD APU, an NVIDIA
laptop GPU and an Intel laptop (CPU and integrated GPU), spanning unified and discrete memory. The answer has five
parts. First, history loss is
silent and appears in every model we tested, from 8B to 70B <!-- H-scale-verdict --> <!-- H-scale-family-verdict -->:
at a 16384-token window no session of three 8B to 14B models kept its full history to turn 15 (intact
5/5/0/0/0/0/0/0 of 5 for each model <!-- R2-real-v1b-survival -->). Second, rule loss is model-dependent. Rule loss appears in the 8B and 14B models (5 seeds) and in qwen3:32b (2 of 3 sessions at 4096) <!-- H-scale-verdict -->, and not in llama3.3:70b or mistral-small3.2:24b (0 of 3 each) <!-- H-scale-verdict --> <!-- H-scale-family-verdict -->. History loss appears in all of them <!-- H-scale-verdict --> <!-- H-scale-family-verdict -->. We do not claim rule loss in general. Third, the cost is concrete: qwen3:14b's share of tasks ending in a wrong
state rises from 11.3% at the default window to 42.0% at 4096, every wrong task silent
<!-- x2-evalset-tasks-real-v2-wrong-rate -->. Fourth, the obvious fixes are partial, and client-side trimming has a
side effect: it removes the in-context examples the model imitates <!-- R2-qwen3-14b-4096-reconciliation -->. Fifth,
the backend implementation is a fragile decision point: with identical model files, settings and seed, llama3.1:8b
wrote a tool call as text in place of its answer at an untruncated window on CUDA and on Vulkan, and not on ROCm
<!-- blade-r2-gate-v2 --> <!-- x2-r2-llama-val-32k-vulkan --> <!-- x2-r2-llama-val-32k-offload -->.

We make four contributions.

1. **A cross-vendor map of where memory runs out and how it fails** (Section 4.4). On the Intel integrated GPU a model
   start succeeds or fails at one shared-heap limit that also counts host-visible buffers, and a first-principles
   accounting of weights, KV cache and compute predicts the measured flip point to within -257 to -26 tokens
   <!-- A-24-flip-point-prediction -->.
2. **Runtime policy, not hardware, sets the effective window, and overflow is silent** (Section 4.5). The same runtime
   gives llama3.1:8b a default window of 4096 tokens on the Intel laptop and 131072 on the AMD APU
   <!-- T2S-vs-X2-default-ctx -->, and answers an over-length prompt with success after keeping num_ctx/2 + 2 tokens
   <!-- ollama-overflow-keeps-half -->.
3. **The consequences for agents, with their mechanism** (Section 4.6). Silent failures after the window grow as the
   window shrinks; Ollama drops whole old messages but keeps the system prompt, and at the smallest window llama.cpp's
   context shift and a token cut discard it <!-- R2-mechanism-lowlevel -->.
4. **Pre-registered fixes and a predictor, with their limits** (Section 4.7). Client-side trimming is refuted as a
   general fix on two vendors <!-- R2-mitigation-v1-verdict --> <!-- t2s-r2-mitigation-cpu -->; `num_keep` lowers
   qwen3:14b's wrong-task rate at 4096 to 30.0% without removing the excess
   <!-- x2-evalset-tasks-real-v2-wrong-rate -->; a router to a cloud model is refuted on the AMD platform
   <!-- b5-verdict -->, with the constrained platforms pending; and a context-shift rule taken from the llama.cpp source
   predicts every observed shift in the sessions it was checked on (misses 0 <!-- R2-shift-predictor-confusion -->).

Section 4.2 covers background, Section 4.3 the platforms and harness, Sections 4.4 to 4.7 the results, Section 4.8
threats to validity, and Section 4.9 concludes.

### 4.2 Background and related work (prose draft, 2026-10-21; 0.75 page)

<!-- Every citation below is a placeholder to verify; see "References (to verify)" at the end of this file. -->

**Context windows in local runtimes.** llama.cpp allocates a KV cache for a fixed number of tokens, n_ctx, when a
model loads. When a sequence would grow past it, the server can perform a *context shift*: it keeps the first n_keep
tokens, discards a block of the tokens after them (2045 at a 4096 window in our logs <!-- R2-mechanism-lowlevel -->)
and continues. Ollama serves models through a bundled llama.cpp server and adds two layers. Its chat endpoint renders
the message list with the model's template and, when the result exceeds the window, drops whole messages from the
front while keeping the system message <!-- R2-mechanism-verdict -->. It also starts the server with context shift
enabled and a keep of 4 tokens, so an over-length prompt succeeds after keeping num_ctx/2 + 2 tokens
<!-- ollama-overflow-keeps-half -->. The request option `num_keep` sets the kept prefix. If the client sends no
`num_ctx`, Ollama chooses the window from the detected device: 4096 tokens for llama3.1:8b on our Intel laptop's CPU,
131072 on the AMD APU <!-- T2S-vs-X2-default-ctx -->. None of these steps is reported to the client.

**Long-context behaviour.** Lost in the Middle [Liu et al. 2024, Lost in the Middle; to verify] shows that models use
information in the middle of a long input worse than at its ends; RULER [Hsieh et al. 2024, RULER; to verify] and
LongBench [Bai et al. 2024, LongBench; to verify] find effective context shorter than the advertised window; and
multi-turn [Laban et al. 2025, LLMs Get Lost in Multi-Turn Conversation; to verify] and rule-following benchmarks
[Zhou et al. 2023, IFEval; to verify] [Mu et al. 2023, Can LLMs Follow Simple Rules?; to verify] measure how well a
model uses or obeys what it is given. All assume the model receives the whole input. We study the step before: what
the runtime removes, and whether anyone can tell.

**KV-cache management.** StreamingLLM [Xiao et al. 2024, StreamingLLM; to verify] keeps initial attention-sink tokens
plus a recent window, much as a context shift keeps n_keep tokens; H2O [Zhang et al. 2023, H2O; to verify] and SnapKV
[Li et al. 2024, SnapKV; to verify] evict by attention; KIVI [Liu et al. 2024, KIVI; to verify] and KVQuant [Hooper et
al. 2024, KVQuant; to verify] quantize the cache; LLMLingua [Jiang et al. 2023, LLMLingua; to verify] compresses the
prompt. These are designed policies evaluated at fixed memory. We measure what shipped runtimes discard by default and
treat KV quantization as one context policy among several (Section 4.7).

**Agents and tool use.** AgentBench [Liu et al. 2024, AgentBench; to verify], tau-bench [Yao et al. 2024, tau-bench;
to verify], function-calling evaluations [Patil et al. 2024, Gorilla; to verify] and SWE-bench [Jimenez et al. 2024,
SWE-bench; to verify] score task success, usually with hosted models and large windows, where running out of context
is at most an exit condition. Routers [Ong et al. 2025, RouteLLM; to verify] choose a model by cost and capability,
not by the local runtime's effective context.

**Inference on edge hardware.** PagedAttention [Kwon et al. 2023, vLLM; to verify] manages KV memory on servers; MELT
[Laskaridis et al. 2024, MELTing Point; to verify], PowerInfer [Song et al. 2024, PowerInfer; to verify] and LLM in a
flash [Alizadeh et al. 2024, LLM in a Flash; to verify] characterize or accelerate inference on phones, consumer GPUs
and flash-backed devices in terms of throughput, memory and energy. We add a cross-vendor characterization of
behavioural correctness under each runtime's memory and context policy.

### 4.3 Platforms, harness and methodology (prose draft, 2026-10-20; updated 2026-10-22; 1.25 pages)

<!-- Convention: measured numbers carry their register id; harness design constants carry "design: <source file>". -->

**Platforms and runtimes.** We study three consumer platforms from three GPU vendors: an AMD Ryzen AI Max+ (Strix
Halo) APU whose Radeon iGPU shares unified memory with the CPU; a laptop with an NVIDIA RTX 4070 Laptop GPU and 8188
MiB <!-- blade-k1-layer-split --> of discrete VRAM; and an Intel Core Ultra (Panther Lake) laptop with an Arc iGPU on
unified memory. Ollama's device detection drops the Intel iGPU and falls back to the CPU
<!-- x2-device-detect-mechanism -->, so we measure Intel twice: on the CPU, as shipped, and on the iGPU, enabled only in
the environment of the job's own server process. All agent runs use Ollama, which serves models through a bundled
llama.cpp server; the AMD and NVIDIA runs share one pinned release and the model files. Versions and
drivers are listed in Table 1 and have no register row yet (gap G13).

**The R2 agent harness.** A session is a seeded, deterministic agent conversation of 40 turns
<!-- R2-v1b-qwen3-8b-reversal -->. The system prompt carries five <!-- design: harness/x2_r2_agent.py --> checkable
rules (JSON answer with fixed keys, call a `log_event` tool, never print a forbidden token, lengths in metres, end with
a session code), a two-tool schema and recall facts. Each turn is two <!-- design: harness/x2_r2_agent.py --> calls:
on the first the model emits tool calls and the harness simulates results with a long filler payload that grows the
context as real tool output does; the second asks for the final answer. Temperature is 0
<!-- design: harness/x2_r2_agent.py -->. Canaries in the system prompt and history tags in the first user message,
which a message-level trim drops first, detect lost context; each canary turn asks for a new pair, because in our
first validation a model copied a reused canary from its own earlier answer and hid a real truncation (positive-control truncation turn [None] <!-- R2-validation-baseline -->). A miss counts as truncation
only while the final prompt exceeds the context Ollama reports as loaded (`/api/ps`, read every turn), never by the
runtime's prompt count.

**Call-2 protocol.** With tools offered on the second call, llama3.1:8b often answered with a tool call and no text
(empty final answers 20 <!-- R2-validation-baseline -->); with tools withheld there, it gave none (empty final
answers 0 <!-- R2-validation-call2-notools -->). All R2 runs below withhold tools on the final call (the
v1 protocol); Section 4.6 repeats the main contrast with tools on every call.

**Validation gates.** Before any real run, each model passes a validation at a window where nothing is truncated,
with gates fixed in advance. A rule is *in use* for a model only if the model follows it on at least 90%
<!-- x2-evalset-val-v2-gates --> of baseline turns; rules below the bar are dropped, not repaired, because the failures
are real behaviour. llama3.1:8b logged events on 60.0% <!-- R2-validation-v2-baseline --> of turns and added the
session code on 60.0% <!-- R2-validation-v2-baseline -->, so only rules 1, 3 and 4 <!-- R2-validation-v2-baseline -->
are in use for it; qwen3:14b kept all five. The same bar decides whether tool validity, arguments and recall are
gated. The negative control must miss no canary (0/12 <!-- R2-validation-v2-baseline --> for both models); the
positive control, at a small window, must detect truncation (turn [10] <!-- R2-validation-v2-baseline -->); and the
task tool must be called correctly on at least 90% of turns, which refused qwen3-4b-2507 at 5/30
<!-- R2-task-tool-gate -->. A real job refuses to start unless its model's validation passes.

**Gated silent failure.** For each session we take the first turn on which a gated metric fails. The failure is
*silent* when no HTTP error or error field appeared on any earlier turn, and *after-window* when it falls at or after
the first turn whose prompt exceeded the loaded window; otherwise it is *baseline-type*, the model's own error rate
rather than context loss. The window effect is the *excess over default*: a tier's after-window rate minus the Ollama
default tier's, with a 95% percentile bootstrap CI over sessions (10000 <!-- R2-v1b-excess-over-default -->
resamples, stratified by tier and model). A 0/n or n/n cell has no bootstrap variance, so its CI understates
uncertainty at five seeds per cell <!-- R2-real-v1b-first-events -->; one cell was later extended to 10 seeds
<!-- R2-qwen3-8b-reversal-10seed --> by its own pre-registration.

**Pre-registration and provenance.** The R2 kill criterion was written before any session ran: the claim fails if,
in every arm, the first failure comes only after an explicit error has surfaced in the same session. The gated
definition came later, as analysis, and is reported beside the criterion, never in its place. Every follow-up below
was pre-registered the same way, with hypothesis, verdict rule and validity conditions committed before its code was
deployed, and we report verdicts in that wording, refutations and inconclusive outcomes included. Every number is
pasted from a numbers register that one script regenerates from the raw result files; a claim with no verified row
is not made. One provenance gap remains: runs before 2026-10-14 did not record the harness file's blob, so their
harness identity rests on byte-identical outputs across reruns (identical final text 120/120 <!-- R2-qwen3-14b-4096-reconciliation -->). Rows are never pooled across platforms, and a configuration flag counts
only when the runtime's log shows it applied: an earlier sweep passed a KV-cache precision setting that the runtime
silently ignored.

### 4.4 Memory-architecture boundaries (2.00 pages)

**Headline figure for H1 (planned, added 2026-10-08): cross-vendor spill cost, FH1.** One figure, three panels, same
x axis (requested context as a fraction of the platform's memory boundary), y axis = cost of crossing it:
- **AMD-iGPU (unified, evo-x2 class):** spill cost near zero, i.e. decode and TTFT nearly unchanged past the device-local
  heap. Data: `[fh1-amd-boundary-llama-3.3-70b]`, `[fh1-amd-boundary-qwen3-32b]` (registered 2026-10-21; FINDINGS
  "FH1 AMD panel: what is measured and what is missing"). Gaps: qwen3-32b has no unclamped spill point (no spill
  cost); the measured onset sits at the 65536 MiB dedicated carve-out, not the Vulkan device-local heap, so the x
  axis normalizer must be stated; reps are 1 call per fresh start with no warm-up; launch flags differ between runs.
- **INTEL-iGPU (unified, budget wall):** no gradual cost; the server either starts or refuses at one shared-heap limit
  (hard fail). Data: `[A-24-effective-heap-limit]`, `[A-24-flip-point-per-model]`.
- **NVIDIA-dGPU (discrete):** decode slowdown proportional to the spilled fraction past VRAM. Data: the x axis
  placement is now registered: the Ollama layer split puts both measured models fully in VRAM at the two smallest
  windows and partly on the CPU above them, "VRAM share by num_ctx 4096: 1.000, 8192: 1.000, 16384: 0.870, 32768:
  0.656" (llama3.1:8b) and "VRAM share by num_ctx 4096: 1.000, 8192: 1.000, 16384: 0.805, 32768: 0.592" (qwen3:8b)
  `[blade-k1-layer-split]`. The y axis (decode slowdown) is still the Blade C1 files with no register row (G4), so
  this panel stays **pending G4**. Caveat: the K1 split is Ollama moving layers to the CPU, a different
  overflow path from the driver's system-memory spill measured in C1 (llama-server direct); the
  panel must label which path each point is.
The claim the figure carries: the same "context exceeds memory" event has three different cost shapes depending on the
memory architecture (near-free spill, hard wall, proportional slowdown), so a hardware-blind router or sizing rule is
wrong on at least two of the three. If space is short, FH1 replaces F2 and F3 (their detail moves to the supplement).

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
  integrated GPUs get very different windows for the same model, and the discrete GPU gets a small one.
- **H1 cross-vendor default-context and truncation table** (all cells pasted from the register; one runtime, Ollama;
  the per-platform version is in Section 3):

| platform | default context per model | GPU placement at the default and above | overflow at the default | register rows |
|---|---|---|---|---|
| INTEL-iGPU | "qwen3-4b-2507": {"t2s_default_ctx": 4096, ...}, "llama3.1:8b": {"t2s_default_ctx": 4096, ...}, "qwen3:8b": {"t2s_default_ctx": 4096, ...} | "evo-t2s (no ROCm path for Intel) falls through to CPU once Vulkan drops it" | "processed = num_ctx/2 + 2 in all 4 checked cases: 4096->2050; 32768->16386; 40960->20482; 8192->4098" (cases pooled over INTEL-iGPU and AMD-iGPU files) | T2S-vs-X2-default-ctx, x2-device-detect-mechanism, ollama-overflow-keeps-half |
| AMD-iGPU | {"qwen3-4b-2507": {"capped": false, "native_ctx": 262144, "ollama_default_ctx": 262144}, "llama3.1:8b": {"capped": false, "native_ctx": 131072, "ollama_default_ctx": 131072}, "qwen3:8b": {"capped": true, "native_ctx": 40960, "ollama_default_ctx": 40960}} | "evo-x2 uses the iGPU via ROCm" | as the INTEL-iGPU row (pooled) | K1v3-X2-table, T2S-vs-X2-default-ctx, x2-device-detect-mechanism, ollama-overflow-keeps-half |
| NVIDIA-dGPU | "llama3.1:8b: default ctx 4096"; "qwen3:8b: default ctx 4096"; "qwen3-4b-2507-tools: not measured (model not present)" | "fully on GPU at default True" (both models); llama3.1:8b "VRAM share by num_ctx 4096: 1.000, 8192: 1.000, 16384: 0.870, 32768: 0.656"; qwen3:8b "VRAM share by num_ctx 4096: 1.000, 8192: 1.000, 16384: 0.805, 32768: 0.592"; "device VRAM 8188 MiB" | llama3.1:8b "4096 tok -> HTTP 200, evaluated 3337; 8192 tok -> HTTP 200, evaluated 2050; 16384 tok -> HTTP 200, evaluated 2050; 32768 tok -> HTTP 200, evaluated 2050"; qwen3:8b "4096 tok -> HTTP 200, evaluated 4082; 8192 tok -> HTTP 200, evaluated 2050; 16384 tok -> HTTP 200, evaluated 2050; 32768 tok -> HTTP 200, evaluated 2050" | blade-k1-layer-split |

  Reading: on all three platforms an over-length prompt returns success and is evaluated at the half-window value
  ("4096->2050" `[ollama-overflow-keeps-half]`; "evaluated 2050" at the default `[blade-k1-layer-split]`). The
  NVIDIA-dGPU probe at the window size itself ("4096 tok") is evaluated below the window and is not an overflow case;
  the probe length is the harness's target token count (`target_tokens` in `results/blade_k1_v2.summary.json`), not
  the model's own count, so it is not read as a truncation.
- **Evidence:** `[T2S-vs-X2-default-ctx]`: {"qwen3-4b-2507": {"t2s_default_ctx": 4096, "x2_default_ctx": 262144},
  "llama3.1:8b": {"t2s_default_ctx": 4096, "x2_default_ctx": 131072}, "qwen3:8b": {"t2s_default_ctx": 4096,
  "x2_default_ctx": 40960}}; mechanism `[x2-device-detect-mechanism]`: "Vulkan backend drops the AMD iGPU by the same
  policy as Intel, but Ollama also discovers it via ROCm (no opt-out), so evo-x2 uses the iGPU via ROCm while evo-t2s
  (no ROCm path for Intel) falls through to CPU once Vulkan drops it"; `[K1v3-X2-table]`. Files:
  `results/apu_results__t2s_k1_ollama_evo-t2s_20261001T074622Z.jsonl`,
  `results/t2s_k1_ollama_evo-x2_20260930T205515Z.jsonl`, `results/x2_template_mechanism_test.jsonl`.
- **Figure:** F5: default window per (platform, model) overlaid on the CDF of real agent step lengths
  (`figures/fig_trace_cdf.png` exists for the CDF; the overlay is new).
- **Gaps:** NVIDIA-dGPU default: landed (Blade night 1 (a), K1, `[blade-k1-layer-split]`); two models there, the
  third not present on that machine. The NVIDIA-dGPU row shows the small default but not why (device detection vs a
  VRAM-size rule); no register row separates the two, so the device-detection mechanism is claimed for the two iGPUs
  only. Ollama version differs between INTEL-iGPU and the other two platforms (Section 3). The `OLLAMA_IGPU_ENABLE=1` control on INTEL-iGPU has no register row (G12).

**4.5b Overflow behaviour.**

- **Claim:** when a prompt exceeds the loaded window, the runtime returns success and silently keeps half the window.
- **Evidence:** "processed = num_ctx/2 + 2 in all 4 checked cases: 4096->2050; 32768->16386; 40960->20482;
  8192->4098. Mechanism: Ollama's own '--context-shift --keep 4' llama-server flag" `[ollama-overflow-keeps-half]`.
  NVIDIA-dGPU, both measured models at the default: "8192 tok -> HTTP 200, evaluated 2050; 16384 tok -> HTTP 200,
  evaluated 2050; 32768 tok -> HTTP 200, evaluated 2050" `[blade-k1-layer-split]` (full row in the 4.5a table).
- **Figure:** folded into F7.
- **Gaps:** NVIDIA-dGPU: landed (Blade night 1 (a)); the overflow probes there ran at the default window only.

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

#### 4.6a The window result (prose draft, 2026-10-20; updated 2026-10-22)

**A default the user never sees.** On the Intel laptop, with no window requested, the same Ollama and the same two
models loaded 4096 <!-- t2s-r2-real-cpu --> tokens on the CPU and 32768 <!-- t2s-r2-real-igpu --> on the iGPU. On the
CPU every session crossed the window at turn [3] <!-- t2s-r2-real-cpu --> and then failed silently, 5/5
<!-- t2s-r2-real-cpu --> for llama3.1:8b and 5/5 <!-- t2s-r2-real-cpu --> for qwen3:14b. On the iGPU no failure came
after the window (0/5 <!-- t2s-r2-real-igpu --> and 1/5 baseline-type <!-- t2s-r2-real-igpu -->). The CPU is the Intel
default <!-- x2-device-detect-mechanism -->; the same tiering gives llama3.1:8b 4096
<!-- T2S-vs-X2-default-ctx --> on Intel, 131072 <!-- T2S-vs-X2-default-ctx --> on the AMD APU and 4096
<!-- blade-k1-layer-split --> on the NVIDIA laptop. This contrast changes backend and window together, so it motivates
the controlled result rather than replacing it.

**The controlled result.** On the AMD APU we ran three models at the Ollama default and at 32768, 16384, 8192 and
4096 <!-- R2-v1b-excess-over-default -->, five seeds per cell: 75 sessions and 3000 turns
<!-- R2-real-v1b-first-events -->, none surfacing an error before its first failure
<!-- R2-real-v1b-first-events -->. After-window silent failures rise as the window shrinks: 0/15 at 32768
<!-- R2-v1b-excess-over-default -->, 2/15 at 16384 (excess over default +0.133, 95% CI +0.000 to +0.267)
<!-- R2-v1b-excess-over-default -->, 8/15 at 8192 (+0.533, 95% CI +0.333 to +0.667) <!-- R2-v1b-excess-over-default -->
and 10/15 at 4096 (+0.667, 95% CI +0.533 to +0.800) <!-- R2-v1b-excess-over-default -->, over a default tier with
0/15 <!-- R2-v1b-excess-over-default -->. The CI excludes zero at 8192 and 4096 <!-- R2-v1b-excess-over-default -->.
Rule failures, as distinct from recall misses, appear only at 8192 and 4096 <!-- R2-real-v1b-first-events -->, and at
4096 no session of any model kept its history to the first canary turn (intact 0/0/0/0/0/0/0/0 of 5
<!-- R2-real-v1b-survival --> for each model).

The default tier's gated count is 3/15 <!-- R2-v1b-default-baseline-type -->, all llama3.1:8b recall misses
inside the window. The pre-registered kill criterion, counting every metric, finds 10/15
<!-- R2-real-v1b-kill-criterion --> silent sessions there, all baseline-type: not killed (killed=False
<!-- R2-real-v1b-kill-criterion -->), but none of it is context loss.

**Models differ** (Table R2-gated). llama3.1:8b already fails at 16384, 2/5
<!-- R2-v1b-gated-per-model -->, and 5/5 at 4096, excess +1.000 <!-- R2-v1b-excess-over-default -->.
qwen3:14b has no after-window failure above 4096 <!-- R2-v1b-gated-per-model -->, and its 4096 sessions lapse on
different rules: the first gated failures are rule 5 or rule 3 at turns 5 to 30 <!-- R2-qwen3-14b-4096-reconciliation -->, while rule 2 held on 40/40
turns in every seed <!-- R2-qwen3-14b-4096-reconciliation -->; we name the failing rule per model rather than say rules broke.
qwen3:8b reversed the ordering at five seeds; a pre-registered extension to 10 seeds per tier
<!-- R2-qwen3-8b-reversal-10seed --> tested whether that was noise. Pooled, qwen3:8b failed silently in 6/10 sessions at 8192 and 2/10 at 4096
<!-- R2-qwen3-8b-reversal-10seed --> (Fisher exact two-sided p = 0.170 <!-- R2-qwen3-8b-reversal-10seed -->), which
the pre-registered rule reads as resolved as noise <!-- R2-qwen3-8b-reversal-10seed -->. The direction persisted: the
new seeds alone gave 2/5 at 8192 and 1/5 at 4096 <!-- R2-qwen3-8b-reversal-10seed -->, and the excess over default is
+0.600 (95% CI +0.300 to +0.900) at 8192 but +0.200 (95% CI +0.000 to +0.500) at 4096, where no excess is shown
<!-- R2-qwen3-8b-reversal-10seed -->. We report the cell as within noise and do
not claim that 4096 is safe for qwen3:8b.

**Protocol check: tools on every call.** A pre-registered rerun offers tools on every call, on a different task
workload, with five seeds per tier and a `num_keep` tier (Section 4.7). It rests on qwen3:14b alone.
llama3.1:8b failed the task validation <!-- x2-evalset-val-v2-gates -->, and qwen3:8b, added by a later
pre-registration with no waiver, failed it too: it called the task tool correctly on 61/81 baseline turns and reached
the task end state on 65/90 (72.2%) <!-- x2-evalset-tasks-val-v2-qwen3-8b-gates -->, so its real run refused to start,
as designed <!-- x2-evalset-tasks-real-v2-qwen3-8b-wrong-rate -->. For qwen3:14b the pre-registered reading was that
the effect holds at 4096 <!-- x2-evalset-tasks-real-v2-gated-excess --> if the excess CI lies above zero. Within this
run it held: 5/5 <!-- x2-evalset-tasks-real-v2-gated-excess --> against 0/5 at the default
<!-- x2-evalset-tasks-real-v2-gated-excess -->, excess +1.000 (95% CI +1.000 to +1.000)
<!-- x2-evalset-tasks-real-v2-gated-excess -->. Wrong task end states were 11.3%
<!-- x2-evalset-tasks-real-v2-wrong-rate --> at the default, 12.7% <!-- x2-evalset-tasks-real-v2-wrong-rate --> at
16384, 35.3% <!-- x2-evalset-tasks-real-v2-wrong-rate --> at 8192 and 42.0% <!-- x2-evalset-tasks-real-v2-wrong-rate -->
at 4096, every wrong turn silent <!-- x2-evalset-tasks-real-v2-wrong-rate -->: excess +24.0 points (95% CI +18.7 to
+29.3) <!-- x2-evalset-tasks-real-v2-wrong-rate --> at 8192 and +30.7 points (95% CI +26.0 to +34.7) at 4096, none
shown at 16384 <!-- x2-evalset-tasks-real-v2-wrong-rate --> <!-- TODO: session-level CI recompute pending for x2-evalset-tasks-real-v2 rows; update these CIs -->. A deviation was declared before the data: qwen3:14b
reached the task end state on 80/90 <!-- x2-evalset-val-v2-gates --> validation turns, below the bar, so absolute
wrong rates include baseline errors.

**Scope: scale and family.** The pre-registered H-scale hypothesis stated that both R2 findings, rule loss at 4096
and history loss with rules intact at 32768 <!-- H-scale-verdict -->, hold from 30B to 70B, with three seeds per
cell. qwen3:32b lost rules at 4096 in 2/3 <!-- H-scale-verdict --> sessions but showed history loss with rules intact
in only 1/3 <!-- H-scale-verdict --> at 32768 (mixed); at its default it loaded 40960 tokens and crossed the window
late, at turns 24, 25 and 26 <!-- x2-scale-real-default-tier -->. llama3.3:70b lost no rules at 4096 (0/3
<!-- H-scale-verdict -->, every session over the window) and lost history with rules intact in 3/3
<!-- H-scale-verdict --> at 32768, the pre-registered alternative "resists rule loss" <!-- H-scale-verdict -->. Overall the verdict is mixed: H-scale is not supported
<!-- H-scale-verdict -->. mistral-small3.2:24b, from another family, also resisted rule loss (0/3 at 4096
<!-- H-scale-family-verdict -->, history loss 2/3 at 32768) <!-- H-scale-family-verdict -->. Within what we tested: Rule loss appears in the 8B and 14B models (5 seeds) and in qwen3:32b (2 of 3 sessions at 4096) <!-- H-scale-verdict -->, and not in llama3.3:70b or mistral-small3.2:24b (0 of 3 each) <!-- H-scale-verdict --> <!-- H-scale-family-verdict -->. History loss appears in all of them <!-- H-scale-verdict --> <!-- H-scale-family-verdict -->. We do not claim rule loss in general.

**4.6a outline items kept (table, history, figure, gaps).**

  Table R2-gated (from `[R2-v1b-gated-per-model]`; silent / after window / baseline-type, of 5 sessions; intact =
  sessions with no gated failure and no canary miss at turns 5/10/15/20/25/30/35/40, as `[R2-real-v1b-survival]`):

  | tier | llama3.1:8b | qwen3:14b | qwen3:8b |
  |---|---|---|---|
  | Ollama default | 3 / 0 / 3; intact 4/4/4/2/2/2/1/1 | 0 / 0 / 0; intact 5/5/5/5/5/0/0/0 | 0 / 0 / 0; intact 5/5/5/4/4/0/0/0 |
  | 32768 | 1 / 0 / 1; intact 5/5/5/3/3/0/0/0 | 0 / 0 / 0; intact 5/5/5/5/0/0/0/0 | 0 / 0 / 0; intact 5/5/5/4/0/0/0/0 |
  | 16384 | 2 / 2 / 0; intact 5/5/0/0/0/0/0/0 | 0 / 0 / 0; intact 5/5/0/0/0/0/0/0 | 0 / 0 / 0; intact 5/5/0/0/0/0/0/0 |
  | 8192 | 4 / 4 / 0; intact 5/0/0/0/0/0/0/0 | 0 / 0 / 0; intact 5/0/0/0/0/0/0/0 | 4 / 4 / 0; intact 3/0/0/0/0/0/0/0 |
  | 4096 | 5 / 5 / 0; intact 0/0/0/0/0/0/0/0 | 4 / 4 / 0; intact 0/0/0/0/0/0/0/0 | 1 / 1 / 0; intact 0/0/0/0/0/0/0/0 |

  Table notes (2026-10-22): the qwen3:14b 4096 cell counts lapses of rule 3 and rule 5, not rule 2, per
  `[R2-qwen3-14b-4096-reconciliation]` ("gated first failure 20260901 5 rule5_session_code; 20260902 20
  rule3_no_zebra; 20260903 5 rule5_session_code"; v1b "20260904 30 rule3_no_zebra; 20260905 none"). The qwen3:8b 8192
  and 4096 cells are five seeds here; the paper quotes the 10-seed pooled cell `[R2-qwen3-8b-reversal-10seed]`
  ("num_ctx_8192: 6/10 gated silent", "num_ctx_4096: 2/10 gated silent", "Fisher exact two-sided p (8192 vs 4096) =
  0.170", "verdict: resolved as noise").

- **History (v1 only, superseded for citation):** the first run alone (two models, three seeds, three tiers) gave
  "num_ctx_32768 1/6 silent, 0 after window exceeded; num_ctx_4096 6/6 silent, 6 after window exceeded; ollama_default
  1/6 silent, 0 after window exceeded" `[R2-real-v1-gated-kill]`. Its v1 cells are part of the strengthened run above;
  the paper cites the v1b rows.
- **Figure:** F6: per-tier session survival curve (intact sessions vs turn) from `[R2-real-v1b-survival]`, with the
  after-window excess per tier and its CI as a second panel.
- **Gaps:** AMD-iGPU only; five seeds per cell, so the per-model CIs are wide, and a 0/5 or 5/5 cell has no bootstrap
  variance (its CI is a point, which understates uncertainty); qwen3-4b-2507 was refused by its validation gate
  ("qwen3-4b-2507 (x2_r2_validation_v2b.jsonl) 5/30 fail; preflight without the gate pass, with it refuse"
  `[R2-task-tool-gate]`), so three of the four planned models are in these rows. Fill: T2S week item 2 (R2 native
  Intel). H-scale landed (`[H-scale-verdict]`, three seeds per cell). Blade night 1 (b) (R2 native) did not run: the
  validation gate failed (open finding O1, Section 5b). Added 2026-10-22: the protocol check rests on qwen3:14b alone, because qwen3:8b failed its B3 v2 validation
  (`[x2-evalset-tasks-val-v2-qwen3-8b-gates]`) and its real run refused (`[x2-evalset-tasks-real-v2-qwen3-8b-wrong-rate]`);
  a second model (mistral-small3.2:24b) is being validated (G17); the qwen3:8b 20-seed extension is queued.

#### 4.6b The mechanism on AMD and NVIDIA (prose draft, 2026-10-20; checked 2026-10-22)

**What the runtime removes.** We reran llama3.1:8b on the AMD APU, one session per tier (80 calls each <!-- R2-mechanism-lowlevel -->), with Ollama in debug
mode, its log sliced per call, and a render-only copy of each request. The rendered prompt matched the runtime's
count on every check (9/9 within 1% at 4096 <!-- R2-mechanism-verdict -->; render/log disagreements 0 <!-- R2-mechanism-verdict -->). Three
mechanisms appear. First, Ollama drops whole messages from the front once the conversation outgrows the window, on
54/80 <!-- R2-mechanism-lowlevel --> calls at 16384, 68/80 <!-- R2-mechanism-lowlevel --> at 8192 and 76/80 at 4096 <!-- R2-mechanism-lowlevel -->, always keeping the system
message ("system kept on all True" <!-- R2-mechanism-verdict -->) and usually cutting mid-turn. This loses history
but keeps the rules in view. At 4096 <!-- R2-mechanism-lowlevel --> two lower-level steps then remove the system prompt. llama.cpp's context shift
fired on 34 <!-- R2-mechanism-lowlevel --> calls from turn 4 <!-- R2-mechanism-lowlevel -->, each keeping n_keep [5] <!-- R2-mechanism-lowlevel --> tokens and discarding the next [2045] <!-- R2-mechanism-lowlevel -->, 29 of them on the
final-answer call <!-- R2-mechanism-lowlevel -->; and once Ollama's own token cut reduced a prompt of 4097 <!-- R2-mechanism-lowlevel --> tokens to
2050 <!-- R2-mechanism-lowlevel -->, keeping 4 <!-- R2-mechanism-lowlevel -->. That is the runtime's overflow rule, which returns success and keeps
num_ctx/2 + 2 tokens (4096->2050 <!-- ollama-overflow-keeps-half -->). No step raised an error (exceed-context errors 0
<!-- R2-mechanism-verdict -->). The report's working hypothesis, "drop old turns, keep the system message", is
therefore refuted over the citable tiers by the 4096 token cut (verdict refuted <!-- R2-mechanism-verdict -->) and
confirmed tier by tier at 32768, 16384 and 8192.

**NVIDIA replicates the first two steps.** One session at 4096 on the NVIDIA laptop passed the same render check (9/9 within 1% <!-- blade-mechanism-4096 -->).
Ollama trimmed whole messages on 75/80 <!-- blade-mechanism-4096 --> calls from turn 3 <!-- blade-mechanism-4096 --> and kept the system message on every one, and the context
shift fired 33 <!-- blade-mechanism-4096 --> times; no token cut occurred and no error surfaced <!-- blade-mechanism-4096 -->. That row lacks
n_keep, so the claim that the shift discards the system prompt is AMD-only; whether the token cut occurs on NVIDIA
is open.

**A second silent difference: tool calls as text.** In the NVIDIA validation, llama3.1:8b's negative control at
32768, where nothing is truncated, missed canaries in two of five sessions (seeds 20260903 and 20260905, 2 misses
each <!-- blade-r2-gate-v2 -->): the final answer was a tool call written as plain text. By the gate set in advance,
this held back the NVIDIA real runs. Three pre-registered checks on the AMD APU kept model files, seeds and settings
and moved one factor toward the NVIDIA configuration each. At 32768 on AMD's ROCm backend there were 0 of 20 <!-- x2-r2-llama-val-32k --> misses,
so the window does not reproduce it ("backend or offload effect" <!-- x2-r2-llama-val-32k -->). Forcing NVIDIA's CPU
offload split on AMD, 22 of 33 <!-- x2-r2-llama-val-32k-offload --> layers on the GPU with VRAM share
0.656 <!-- x2-r2-llama-val-32k-offload -->, again gave 0 of 20 ("backend effect" <!-- x2-r2-llama-val-32k-offload -->). Switching only the backend, to
Vulkan on the same AMD GPU and verified in 5/5 <!-- x2-r2-llama-val-32k-vulkan --> sessions, gave 2 of 20 <!-- x2-r2-llama-val-32k-vulkan --> misses, at seed 20260903, turn 5 <!-- x2-r2-llama-val-32k-vulkan -->, as a text call
to `lookup_fact`: "backend implementation, not the vendor" <!-- x2-r2-llama-val-32k-vulkan -->. The failure follows
the llama.cpp backend implementation (CUDA and Vulkan, not ROCm, in these seeds), not the GPU vendor: one fragile
decision point, turn 5 of seed 20260903 <!-- x2-r2-llama-val-32k-vulkan --> on Vulkan, in the seed that also missed
on CUDA <!-- blade-r2-gate-v2 -->, separates a correct agent from a silent miss. Five seeds cannot
show a zero is impossible, and the kernel is not identified; identical model files and settings still yield a
different agent on a different backend, with no error.

**How the window is set, and whether it sticks.** On the AMD APU, where llama3.1:8b defaults to 131072 <!-- window-override-methods -->, a
per-request `num_ctx` of 12288 loaded 12288 <!-- window-override-methods --> but a plain request after a restart was back at 131072 <!-- window-override-methods -->; a Modelfile
`PARAMETER num_ctx` of 10240 on a derived tag loaded 10240 <!-- window-override-methods --> and survived the restart;
and `OLLAMA_CONTEXT_LENGTH` of 6144 applied while set and reverted once the server restarted without it
<!-- window-override-methods -->. Only a derived tag persists: a client that omits the per-request option silently gets the
default back. This check has not yet run on NVIDIA or Intel.

**4.6b outline items kept (figure, gaps).**

- **Figure:** F7: per-turn event raster per tier (message drops, context shifts, token cut), one row per tier, with
  the NVIDIA-dGPU 4096 session as an extra row next to the AMD-iGPU 4096 row.
- **Gaps:** one model (llama3.1:8b), one session per tier on each platform (`[R2-mechanism-verdict]` n column: "5
  sessions"; `[blade-mechanism-4096]` n column: "80" calls, one session), so no CI; whether the token cut occurs on
  NVIDIA-dGPU at all is open after one session. Fill: T2S week item 6 (mechanism run, INTEL-iGPU), Blade night 2 (e)
  (other NVIDIA-dGPU tiers); more seeds through x2_r2_real_v1b's mechanism logging.

**4.6c Outcome table.**

- **Claim (pending):** the runtime and its defaults change task-level outcomes for the same model on the same
  machine.
- **Evidence so far:** "total 1078/3600" `[x2-v3-progress]` (moving value; re-paste at each register regeneration); partial per-cell scores `[x2-v3-scores]` and error causes
  `[x2-v3-error-causes]`. File: `results/x2_outcome_table_v3.jsonl`. Not citable as a result until complete.
- **Figure:** none until complete.
- **Gaps:** x2_outcome_table_v3 completion; T2S week item 5 (outcome-table subset).

### 4.7 Fixes (prose draft, 2026-10-20; updated 2026-10-22; with the predictor outline kept below; 1.00 page)

**Client-side trimming splits by model, on two vendors.** A client that never sends an over-length prompt should keep
the runtime from cutting the system prompt. We pre-registered that fix: before
every call the client renders and counts the request with the model's tokenizer and drops whole old turns until the
system prompt, the kept turns and the answer budget fit the window with a 5% <!-- design: harness/x2_r2_client_trim.py -->
margin, always keeping the system prompt. A model *recovers* when every rule in use is at or above 90%
<!-- design: analysis/r2_mitigation_report.py -->, and the hypothesis needed both models to recover at 4096
<!-- R2-mitigation-v1-verdict -->. Mechanically it worked: 0 context shifts, 0 token cuts, 0 Ollama message drops and 0
exceed-context errors in every cell <!-- R2-mitigation-v1-observed -->. The verdict is *refuted*
<!-- R2-mitigation-v1-verdict -->. On the AMD APU llama3.1:8b recovered at 8192 and 4096
<!-- R2-mitigation-v1-verdict -->; qwen3:14b recovered at 8192 but not at 4096 <!-- R2-mitigation-v1-verdict -->, where
its `log_event` rule fell to 61.7% <!-- R2-mitigation-v1-4096 -->, failing first at turn 3 in every seed (3/3/3
<!-- R2-mitigation-v1-4096 -->). On the Intel laptop's CPU at 4096 <!-- t2s-r2-mitigation-cpu --> the split
replicates: llama3.1:8b recovered, qwen3:14b did not, again on `log_event`, overall refuted
<!-- t2s-r2-mitigation-cpu -->. The cause is the fix itself: untrimmed, qwen3:14b kept rule 2 on 40/40 turns in every
seed <!-- R2-qwen3-14b-4096-reconciliation -->. On the first failing trimmed turn its first call
requested only `lookup_fact`, where the untrimmed run called `log_event` and then `lookup_fact` <!-- R2-qwen3-14b-4096-reconciliation -->: at 4096 the
budget leaves room for only 1-2 <!-- R2-mitigation-v1-observed --> turns besides the system prompt, so trimming
removes the in-context examples of the two-call pattern the model imitates. Naive trimming swaps silent context loss
for deliberate history loss and, for this model, a new rule failure.

**num_keep fixes one model's rules, on the token-cut path.** The request option `num_keep` sets how many leading tokens
the context shift and the token cut keep. It matters only where a shift or a cut happens; a whole-message drop keeps
the system message anyway (Section 4.6b). We
pre-registered a test for llama3.1:8b at 4096, the path where both steps fired, with `num_keep` set to the system
prompt's measured token length (601, 594 and 597 per seed <!-- num-keep-guardrail-test -->). All 105
<!-- num-keep-guardrail-test --> context shifts kept n_keep [595, 598, 602] <!-- num-keep-guardrail-test -->. The rules in use held (rule 1 at 94.2% <!-- num-keep-guardrail-test -->, rules 3 and
4 at 100.0% <!-- num-keep-guardrail-test -->), which the pre-registered rule reads as "one-line fix"
<!-- num-keep-guardrail-test -->. It fixes the rules and nothing else: Ollama still dropped messages on 224 of 239
<!-- num-keep-guardrail-test --> calls and the history canary was missed on 24 of 24 <!-- num-keep-guardrail-test -->
checks. Nor does it remove the window effect in the protocol check: within that run the
`num_keep` tier at 4096 lowered qwen3:14b's wrong-task rate from 42.0% <!-- x2-evalset-tasks-real-v2-wrong-rate --> to
30.0% <!-- x2-evalset-tasks-real-v2-wrong-rate -->, but its excess over the default stayed at +18.7 points (95% CI
+16.0 to +21.3) <!-- x2-evalset-tasks-real-v2-wrong-rate --> <!-- TODO: session-level CI recompute pending for x2-evalset-tasks-real-v2 rows; update these CIs --> and its gated excess at +1.000
<!-- x2-evalset-tasks-real-v2-gated-excess -->; the pre-registered reading is that `num_keep` does not remove it
<!-- x2-evalset-tasks-real-v2-gated-excess -->. In the R2 harness a num_keep run for qwen3:14b at 4096 reproduced the
unflagged run turn for turn (identical final text 120/120 <!-- R2-qwen3-14b-4096-reconciliation -->), so for this model the flag changed nothing there,
and what lowered the task-workload rate is not established.

**A router does not dominate.** We pre-registered a comparison of eight
<!-- design: harness/policy_compare.py --> context policies (runtime default, largest window that starts, client
trimming, a router escalating to a cloud model, an auto-sized window, `num_keep`, q8_0 and q4_0 KV caches), with the
hypothesis that the router is on the Pareto frontier of rules, history, time and cost on every
platform. It is *refuted*: on the AMD APU the router was dominated for both models <!-- b5-verdict -->. Every policy
kept every rule (rule 100.0% in every cell <!-- b5-policy-metrics -->), and the frontier held one KV-cache policy
per model <!-- b5-policy-metrics -->. For llama3.1:8b, q4_0 reached history 100.0% in 6.6 min <!-- b5-policy-metrics --> per
session against the router's 95.8% in 43.5 min <!-- b5-policy-metrics -->. For qwen3:14b, q8_0 cut session time from
214.8 min at the runtime default to 75.4 min at the same history recall, 62.5% <!-- b5-policy-metrics -->, while the
router took 135.1 min and USD 0.8419 per session for no gain <!-- b5-policy-metrics -->. The refutation holds on the
unconstrained AMD platform, whose default window is the model's native one, so no policy there faced the
small-window regime where loss occurs; the constrained platforms are pending.

#### 4.7p Predictor (outline, not yet prose)

- **Claim (verified on AMD-iGPU, one session per tier):** a context shift, the step that discards the system prompt
  at the smallest window, happens on a call exactly when the post-trim prompt plus the generated tokens exceed the
  loaded window; the rule is taken from the llama.cpp source, not fitted. Using the generation budget in place of the
  generated count gives a before-the-call flag that misses no shift at the cost of false alarms.
- **Evidence:** `[R2-shift-predictor-confusion]`: "num_ctx_4096 TP 34 FP 0 FN 0 TN 46 precision 100.0% recall
  100.0%", "num_ctx_32768 TP 10 FP 0 FN 0 TN 70 precision 100.0% recall 100.0%", "misses 0"; variants "exact
  P+C>n_ctx: TP 44 FP 0 FN 0", "budget P+num_predict>n_ctx: TP 44 FP 22 FN 0", "prompt only P>=n_ctx-1: TP 0 FP 0 FN
  44". Applied to the real run as a prediction only (no server log there, nothing observed):
  `[R2-shift-predicted-real-v1]`: "num_ctx_4096 llama3.1:8b: predicted 102/240 calls" and "num_ctx_4096 qwen3:14b:
  predicted 0/240 calls"; predicted shifts line up with the first rule failure only partly, and the qwen3:14b rule
  failures at 4096 happen with no predicted shift (`docs/FINDINGS.md`, "R2 context-shift predictor and orphaned messages"). What the message-level trim
  leaves at the front of the window: `[R2-orphan-rates]`: "num_ctx_4096: 76/80 truncated; first kept user 15,
  assistant tool-call 47, assistant final 6, tool 8; orphaned 61/76 (80.3%)". File: `results/x2_r2_mechanism.jsonl`
  (observed part), `results/x2_r2_real_v1.jsonl` (predicted part).
- **Scope of the claim:** it predicts the context shift (where the system prompt goes on AMD-iGPU), not rule
  failure: the real-run qwen3:14b failures are not explained by it. Built and scored on the same mechanism sessions;
  no tunable parameter, but no held-out observed sessions yet. (2026-10-22: the qwen3:14b 4096 statement above
  is consistent with `[R2-qwen3-14b-4096-reconciliation]`: qwen3:14b's 4096 lapses are rule 3 and rule 5, and the
  num_keep rerun reproduced v1 turn for turn ("identical final text 120/120").)

**4.7 outline items kept (figure and gaps, mitigation and predictor).**

- **Figure:** F8: rule compliance per rule in use at 4096 and 8192, mitigation vs real run, per model, from
  `[R2-mitigation-v1-4096]` and `[R2-mitigation-v1-8192]`. The qwen3:14b 4096 panel should show that rule 2
  falls only with trimming (40/40 untrimmed vs 24/40, 25/40, 25/40 trimmed, `[R2-qwen3-14b-4096-reconciliation]`).
- **Gaps:** two models, three seeds per cell, one runtime, no CI; the 4096 comparison is against the v1 cells (three
  seeds), the 8192 comparison against v1b's first three seeds. Predictor scored on observed shifts outside the sessions
  it was checked on (G6): the NVIDIA-dGPU 4096 session (`[blade-mechanism-4096]`, its shift count is registered but the
  per-call confusion is not) and any later mechanism-logged run. T2S week item 3 (mitigation at 4096). Blade night 1
  (c) (mitigation) did not run (open finding O1). num_keep is tested for llama3.1:8b only in the prose; the qwen3:14b
  num_keep run is cited only through `[R2-qwen3-14b-4096-reconciliation]`, not its own row. B5 has run on AMD-iGPU only (`[b5-verdict]`: evo-t2s and blade "not run").

### 4.8 Threats to validity (prose draft, 2026-10-21; 0.50 page)

**Platforms and versions.** We measure one machine per vendor, so a platform effect cannot be separated from that
machine's driver and memory configuration. Agent runs use Ollama 0.34.4 <!-- R2-qwen3-14b-4096-reconciliation --> and
direct runs llama.cpp b10970 <!-- design: docs/HARDWARE.md (version record; no register row, gap G13) -->; default
windows and trimming are runtime code and can change in any release. On the Intel laptop, CPU and iGPU differ in
backend and loaded window together, so the motivating contrast is not a controlled test of either. The NVIDIA laptop
contributes validation and mechanism sessions only, because its validation gate held back the real agent runs
<!-- blade-r2-gate-v2 -->, and the router comparison ran on the AMD APU alone <!-- b5-verdict -->, whose default window
is the model's native one.

**Statistics.** Cells have three <!-- H-scale-verdict --> to five <!-- R2-v1b-gated-per-model --> seeds, ten for
qwen3:8b <!-- R2-qwen3-8b-reversal-10seed -->. A 0/5 or 5/5 cell has a point bootstrap interval that understates
uncertainty, and the scale and family results rest on three sessions per cell. qwen3:8b's lower failure count at 4096
than at 8192 is within noise (p = 0.170 <!-- R2-qwen3-8b-reversal-10seed -->), and we claim nothing from it. Sampling
uses temperature 0 <!-- design: harness/x2_r2_agent.py --> and a fixed request seed, so seeds vary the session
script, not the sampling; nonzero temperatures are untested.

**Workload.** The R2 rules are synthetic and checkable; deployed system prompts differ. Guardrail rules 6 to 8
(privacy, refusal, confirmation) already fail at full context for both models tested, for example rule 6 at 0.417 and
0.250 of bait turns <!-- x2-evalset-val-v1-gates -->, so they are excluded from every window measurement and support no
claim here. The tools-on-every-call protocol check rests on qwen3:14b alone, under a deviation declared before the
data (task end state 80/90 <!-- x2-evalset-val-v2-gates --> at validation); qwen3:8b (65/90
<!-- x2-evalset-tasks-val-v2-qwen3-8b-gates -->) and llama3.1:8b (72/90 <!-- x2-evalset-tasks-real-v2-wrong-rate -->)
were refused by their gates.

**Provenance.** Runs before 2026-10-14 did not record the harness file's hash ("harness blob not recorded"
<!-- R2-qwen3-14b-4096-reconciliation -->). Their identity rests on reruns: the same seeds gave identical final text on
120/120 <!-- R2-qwen3-14b-4096-reconciliation --> turns in runs days apart.

### 4.9 Conclusion (prose draft, 2026-10-21; 0.50 page)

Local LLM agents lose context without telling anyone. On every platform we measured, a conversation that outgrows the
window is trimmed, shifted or cut, and the request still succeeds. Rule loss appears in the 8B and 14B models (5 seeds) and in qwen3:32b (2 of 3 sessions at 4096) <!-- H-scale-verdict -->, and not in llama3.3:70b or mistral-small3.2:24b (0 of 3 each) <!-- H-scale-verdict --> <!-- H-scale-family-verdict -->. History loss appears in all of them <!-- H-scale-verdict --> <!-- H-scale-family-verdict -->. Rule loss depends on the model as well as the window, and we do not claim it in general. The window is often not the user's
choice: the runtime picks it from the detected device, and an unmodified install on an Intel CPU ran agents at 4096
tokens <!-- t2s-r2-real-cpu -->. That is the window at which, on the AMD APU, qwen3:14b ended 42.0% of tasks in a wrong
state against 11.3% at its default, all silently <!-- x2-evalset-tasks-real-v2-wrong-rate -->.

For practitioners, four lessons follow. Set the window explicitly and verify the loaded value, because a per-request
setting does not survive a restart and any client that omits it silently gets the default back
<!-- window-override-methods -->. Prefer an explicit window sized to the session over the runtime default, and log
when the prompt exceeds it, since nothing else will. Treat trimming as a change to the prompt rather than a neutral
fix: dropping old turns also drops the examples the model imitates <!-- R2-qwen3-14b-4096-reconciliation -->, and
`num_keep` protects the prompt prefix only where the runtime shifts or cuts tokens. Finally, validate an agent on the
backend it will run on, because identical model files and settings behaved differently on CUDA, Vulkan and ROCm.

Future work follows from what is pending. The router and context-policy comparison, refuted on the unconstrained AMD
platform <!-- b5-verdict -->, must run on the Intel and NVIDIA platforms, where default windows are small and loss
occurs. More models per size and family, more seeds, further runtime versions and guardrail compliance at full context
would turn the model-dependence of rule loss from an observation into a predictable property.

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
| P1-07 | Runtime device policy sets the default window: small on INTEL-iGPU, model-native on AMD-iGPU, small on NVIDIA-dGPU | T2S-vs-X2-default-ctx, x2-device-detect-mechanism, K1v3-X2-table, blade-k1-layer-split | I, A, N | deterministic configuration reads; n column "3" models `[T2S-vs-X2-default-ctx]`; N: two of three models present ("qwen3-4b-2507-tools: not measured (model not present)" `[blade-k1-layer-split]`) | "Documented Ollama behaviour, bug report level." | Issue trackers document the tiering; nobody traced it through device detection to agent loss across vendors | verified (I, A, N); device-detection mechanism traced on I and A only |
| P1-08 | Overflow is silent: success status, half the window kept | ollama-overflow-keeps-half, blade-k1-layer-split | I, A, N | "all 4 checked cases" `[ollama-overflow-keeps-half]`; N: "HTTP 200, evaluated 2050" on every over-length probe at the default, both models `[blade-k1-layer-split]`; deterministic | "Context shift is a llama.cpp feature." | Yes; the finding is that the shipped default applies it silently and what it removes, on all three vendors | verified (I, A, N) |
| P1-09 | Install path of identical weights changes tool-call behaviour | R2-install-path-4b-comparison, R2-install-path-4b-bare, x2-chat-template-source | A | "30 + 30 baseline turns" `[R2-install-path-4b-comparison]`; one model | "You used the wrong template." | That is the finding: the install path picks the template silently; both paths fail the gate | verified (narrow, one model) |
| P1-10 | Models do not report that context was lost | self-report-truncation-awareness | I, A | "0/960" `[self-report-truncation-awareness]` | "Models can't know what they don't see." | Agreed; it is why the failure is silent end to end | verified |
| P1-11 | Real agent sessions lose history silently at every truncating window; silent failures after the window rise as the window shrinks, over a default tier with none after the window; rules lost only at the two smallest windows | R2-real-v1b-gated-kill, R2-v1b-gated-per-model, R2-v1b-excess-over-default, R2-v1b-default-baseline-type, R2-real-v1b-kill-criterion, R2-real-v1b-survival, R2-real-v1b-first-events (v1 rows R2-real-v1-* kept as history) | A | three models x five seeds per tier, five tiers, "75 sessions, 3000 turns" `[R2-real-v1b-first-events]`; excess over default with 95% session-bootstrap CI, e.g. "num_ctx_4096: overall 10/15 vs default 0/15, excess +0.667 (95% CI +0.533 to +0.800)" `[R2-v1b-excess-over-default]`; per-model CIs wide (five seeds) | "Few models, one machine." / "The default tier fails too, so it is not context." | Default-tier silent failures are all baseline-type, 0 after the window (Section 4.6a); the excess over default isolates context loss; T2S replicates; the NVIDIA-dGPU real run waits on the O1 follow-up | verified (A); generalization pending T2S week item 2, H-scale; N held (Blade night 1 (b) not run) |
| P1-12 | Mechanism: message trimming keeps the system prompt (A, N); context shift fires at the smallest window (A, N); context shift and token cut discard the system prompt there (A) | R2-mechanism-verdict, R2-mechanism-lowlevel, blade-mechanism-4096 | A, N | one model, one session per tier on each platform, no CI (n column "5 sessions" `[R2-mechanism-verdict]`; N one session at 4096, n column "80" calls `[blade-mechanism-4096]`); render/log check passed on both ("render/log disagreements 0"; "prompt check 9/9 within 1% (citable True)"); token cut seen on A once ("1 token cuts"), not on N ("token-level cuts 0") | "Single seed." | The mechanism is deterministic given the prompt sizes, and the message-level step and the shift now replicate on a second vendor | verified (A, N; one session per tier); I pending T2S week item 6; other N tiers pending Blade night 2 (e) |
| P1-13 | Real agent workloads exceed small defaults | trace-fraction-exceeds-4096, trace-context-exit-rate, uncensored-trace-32k-crossing, pack-trace-weighted-stats | n/a (trace data) | full trace sets | "Synthetic agents." | Public real trajectories, censoring handled | verified |
| P1-14 | Runtime choice changes task outcomes for the same model and machine | x2-v3-progress, x2-v3-scores, x2-v3-error-causes | A (I subset pending) | incomplete | "Incomplete table." | Report only when complete | pending (x2_outcome_table_v3; T2S week item 5) |
| P1-15 | Co-running app memory pressure does not degrade quality before a loud failure | K2-x2-kill-criterion, K2-x2-score-ceiling, K2-x2-responsiveness-resolution, K2-x2-clean-failures | A | measures at ceiling | "Your test could not detect degradation." | Correct; reported as a weak null only | at risk (x2_k2_pause_resume_v1; harder task set unplanned) |
| P1-16 | A predictor flags the call on which the context shift (system-prompt loss on A) happens: prompt plus generated tokens over the loaded window | R2-shift-predictor-confusion, R2-shift-predicted-real-v1, R2-orphan-rates | A | one model, one session per tier, no CI; "num_ctx_4096 TP 34 FP 0 FN 0 TN 46" and "num_ctx_32768 TP 10 FP 0 FN 0 TN 70" `[R2-shift-predictor-confusion]`; real-run counts are predictions only `[R2-shift-predicted-real-v1]` | "Post hoc." | The rule comes from the llama.cpp source with no fitted parameter; the before-the-call budget variant gives "TP 44 FP 22 FN 0"; held-out observed sessions still needed (G6) | verified (A, one session per tier) |
| P1-17 | The obvious fix (client-side trimming that keeps the system prompt) works for one model and fails for another at the smallest window: llama3.1:8b recovered at 4096 and 8192; qwen3:14b recovered at 8192, not at 4096 (rule 2) | R2-mitigation-v1-verdict, R2-mitigation-v1-4096, R2-mitigation-v1-8192, R2-mitigation-v1-observed | A (I planned; N held) | two models, three seeds per cell, no CI; "12 sessions, 480 turns" `[R2-mitigation-v1-verdict]`; mechanically clean: "0 context shifts, 0 token cuts" in every cell `[R2-mitigation-v1-observed]` | "Mitigation is just a bigger window." / "You tuned until it worked." | Pre-registered verdict rule and refutation clause fixed before data; the verdict is refuted and reported as such | verified (A): pre-registered hypothesis refuted; T2S week item 3 pending; N held (Blade night 1 (c) not run) |
| P1-18 | A single TTFT fit does not transfer across machines | ttft-cross-machine-transfer, ttft-few-point-calibration | I, A | pooled rows include the stall | "Confounded by the stall." | Must recompute on rep>=1 rows before citing | at risk (recompute, G11) |
| P1-19 | The NVIDIA-dGPU negative-control canary misses (text tool call on the final-answer call at 32768) are a backend or CPU-offload effect, not a window effect | x2-r2-llama-val-32k, blade-r2-gate-v2, blade-r2-validation-run2 | A (check), N (finding) | five seeds each side; same model files, seeds and settings: AMD-iGPU "canary misses 0 of 20 canaries over 5 sessions" `[x2-r2-llama-val-32k]` vs NVIDIA-dGPU "20260903: 2, ... 20260905: 2" `[blade-r2-gate-v2]` | "A 0 in five seeds proves nothing." | The rule was pre-registered; the claim is only that the window alone does not reproduce it; backend vs offload is not separated (follow-up pending) | verified (narrow): backend or offload effect; which of the two is open |

Tally (2026-10-14): verified 13 (P1-01, 04, 05, 07, 08, 09, 10, 11, 12, 13, 16, 17, 19), pending 3 (P1-02, 03, 14),
at risk 3 (P1-06, 15, 18). Changes since 2026-10-10: P1-11 recomputed on the strengthened run (v1 + v1b) with the
excess over the default tier and its CI, wording narrowed (rules lost at the two smallest windows, not only the
smallest); P1-17 pending -> verified, as a refuted pre-registered hypothesis ("the obvious fix works for one model and
fails for another at the smallest window"); P1-19 new, from the O1 confound check (Section 5b). Of the 13 verified,
P1-05, 11, 12, 16 and 17 are verified on fewer platforms than the paper claims and need the named runs to support the
cross-vendor wording. Earlier tally (2026-10-10): verified 11, pending 4, at risk 3. Open findings are not claims and
are not counted (Section 5b).

## 5b. Open findings (not claims)

These are results with register rows that are not yet interpretable. They are not stated as claims anywhere in the
paper until the named check lands.

**O1. NVIDIA-dGPU: text tool call on the final-answer call at a non-truncating window.** In the NVIDIA-dGPU R2
validation, llama3.1:8b's negative control (a window large enough that nothing is truncated) failed in both runs: on
the final-answer call (tools withheld, v1 protocol) the model sometimes writes a tool call as plain text instead of an
answer, so neither canary appears. Register: `[blade-r2-gate-v2]`: "llama3.1:8b negative control num_ctx 32768, 5
sessions, canary misses per session 20260901: 0, 20260902: 0, 20260903: 2, 20260904: 0, 20260905: 2; branch
mechanism"; `[blade-r2-validation-run2]`: "negative control (num_ctx 32768) canary misses 4 of 20 canaries (10 checks
x 2) over 5 sessions"; first run `[blade-r2-validation-run1]`. In run 2 both failing sessions fail on the same turn
(`docs/FINDINGS.md`, "Blade (NVIDIA RTX 4070 Laptop, 8 GB) night 1", 2026-10-09). It is not truncation (the window was
never exceeded) and not a scorer bug. The AMD-iGPU negative control for the same model files at the model-native
default had no misses (`[R2-validation-v2-baseline]`: "negative-control canary misses 0/12"). Candidate explanations:
backend (CUDA vs ROCm) and the window itself (32768, partly CPU-offloaded on NVIDIA-dGPU per `[blade-k1-layer-split]`,
vs the native default on AMD-iGPU), and with them greedy-decoding numerics.
- **Consequence:** per the pre-set gate, the NVIDIA-dGPU R2 real tiers and mitigation (Blade night 1 (b), (c)) did not
  run; the gate took the mechanism branch (`[blade-r2-gate-v2]`: "branch mechanism").
- **Confound check, result (2026-10-14): the window does not reproduce it; backend or offload.** x2-r2-llama-val-32k ran the
  NVIDIA-dGPU run-2 validation on AMD-iGPU at the NVIDIA-dGPU window, num_ctx 32768: the same model files, the same
  seeds and the same settings (pre-registration in `docs/FINDINGS.md`, 2026-10-10, with the decision rule fixed before
  the data: no canary misses on AMD-iGPU at 32768 means a backend or offload effect; text tool calls on the same turn
  as on NVIDIA-dGPU mean a window effect). Result: "negative control (num_ctx 32768) canary misses 0 of 20 canaries
  over 5 sessions, per session (seed: misses) 20260901: 0, 20260902: 0, 20260903: 0, 20260904: 0, 20260905: 0;
  missed turns: no missed canary turn" and "verdict: backend or offload effect" `[x2-r2-llama-val-32k]`. The seeds
  that missed on NVIDIA-dGPU (20260903 and 20260905, `[blade-r2-gate-v2]`) did not miss on AMD-iGPU. So the
  NVIDIA-dGPU text-tool-call misses are not caused by the 32768 window; per the pre-registered rule they are a backend
  (CUDA) or CPU-offload effect. This is stated as claim P1-19 (narrow: the window alone does not reproduce it).
- **Still open:** which of the two. At 32768 the NVIDIA-dGPU model is partly on the CPU ("32768: 0.656" VRAM share
  for llama3.1:8b `[blade-k1-layer-split]`), so backend and offload change together in this comparison. A follow-up
  that separates them (the same validation on NVIDIA-dGPU with the model fully in VRAM, or with the same partial
  offload on AMD-iGPU) is pending; no job for it is designed or queued yet. Five seeds per side: a 0 on AMD-iGPU shows
  the window alone did not reproduce the misses in these seeds, not that they cannot occur there.
- **Paper status:** O1 is now partly resolved. The window alone does not explain it (P1-19); the backend-vs-offload
  question stays an open finding and goes in threats to validity as a cross-backend difference. The NVIDIA-dGPU R2
  real run and mitigation stay held until the follow-up separates backend from offload.

Candidate claims (not in the table or tally yet; claim | register row | status):

- On qwen3-32b, Ollama completes the items llama-server refuses as over-length only by silent truncation to the
  keep-half window, with lower accuracy on those items ("truncated score 0.40 (6/15, Wilson 95% CI 0.20-0.64)" vs
  "full score 1.00 (82/82, Wilson 95% CI 0.96-1.00)"); its speed lead on the llama-server timeouts is not truncation
  ("full 11 (of 11)") | qwen3-32b-truncation-vs-latency (also x2-v3-truncation-all-models) | provisional (outcome
  table v3 not finished; FINDINGS 2026-10-10)

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
| FH1 (headline, H1) | Cross-vendor spill cost: AMD-iGPU unified near-zero cost, INTEL-iGPU budget wall (hard fail), NVIDIA-dGPU discrete decode slowdown, on a shared normalized x axis | `[fh1-amd-boundary-llama-3.3-70b]`, `[fh1-amd-boundary-qwen3-32b]`; `[A-24-effective-heap-limit]`, `[A-24-flip-point-per-model]`; Blade C1 files (register row pending, G4) and the K1 layer split `[blade-k1-layer-split]` | partly: INTEL yes; AMD registered for llama-3.3-70b (onset and cost), qwen3-32b fail side only (no unclamped spill point); NVIDIA decode slowdown needs Blade night 2 (operator at the keyboard), not yet run; NVIDIA placement (VRAM share per num_ctx) registered, decode slowdown pending G4 |
| F1 | Platform and memory-architecture diagram: three platforms, where KV, weights and host-visible buffers live, which runtime backend is used | `docs/HARDWARE.md`, Section 3 of this file | yes (drawing only) |
| F2 | INTEL-iGPU flip point: measured (last_ok, first_fail] vs accountings A/B/C, all models in `[A-24-budget-boundary]` | `[A-24-flip-point-per-model]`, `[A-24-effective-heap-limit]` | yes; existing `figures/fig_a24_budget_boundary.png` needs regeneration |
| F3 | NVIDIA-dGPU spill onset: TTFT and decode slowdown vs context with spilled fraction; C3 point | C1/C2 files (no register row); Blade night 2 (d) | partly: C1/C2 yes, C3 no; existing `figures/fig_blade_c1_spill.png` is the M1 ladder |
| F4 | Co-runner effects by vendor: TTFT and decode ratio per condition, INTEL-iGPU vs AMD-iGPU | `[B3-corunner-6model]`, `[b4_32b-dose-response]`, `[PX2-full-ratio-table]`, `[PX2-decode-ratio]`; T2S week item 4 | partly: Intel decode under a bandwidth hog missing |
| F5 | Default window per platform and model over the agent step-length CDF | `[T2S-vs-X2-default-ctx]`, `[blade-k1-layer-split]`, `[trace-fraction-exceeds-4096]` | yes: I, A and N (N two models); `figures/fig_trace_cdf.png` exists, overlay new |
| F6 | R2 session survival by tier and model, plus the after-window excess over the default tier with 95% CI | `[R2-real-v1b-survival]`, `[R2-v1b-gated-per-model]`, `[R2-v1b-excess-over-default]`; T2S week item 2; Blade night 1 (b) | partly: A yes (v1 + v1b); other platforms no |
| F7 | Per-turn truncation-event raster by tier (message drop, context shift, token cut), A all tiers plus an N row at 4096 | `[R2-mechanism-lowlevel]`, `[R2-mechanism-verdict]` (`results/x2_r2_mechanism.jsonl`); `[blade-mechanism-4096]` (`results/blade_r2_mechanism_4096_v2.jsonl.report.json` and its jsonl); T2S week item 6; Blade night 2 (e) | yes for A (all tiers) and N (4096 only), one session per tier; I and other N tiers no |
| F8 | Mitigation: rule compliance per rule in use at 4096 and 8192, with vs without, per model | `[R2-mitigation-v1-verdict]`, `[R2-mitigation-v1-4096]`, `[R2-mitigation-v1-8192]`; T2S week item 3 (Blade night 1 (c) held by O1) | yes for A |

Backup figure if one slips: first-call TTFT stall per cell (`[ttft-first-call-stall-per-cell]`, data exists).

---

## 8. Gaps: what must land before 2026-11-01, ordered by impact on acceptance

1. **G1. Agent result on more than one platform.** More seeds and models landed on AMD-iGPU (v1b: three models x
   five seeds per tier, rows `R2-real-v1b-*`, `R2-v1b-*`); the headline is still one AMD machine. Fill: T2S week item
   2 (R2 native Intel); H-scale for 30B to 70B. Blade night 1 (b) (R2 native) did not run; x2-r2-llama-val-32k now
   points to the NVIDIA-dGPU backend or offload (O1), and the run waits on the follow-up that separates them.
   Prerequisite for the Intel runs: T2S week item 1 (re-enable checks).
2. **G2. Mechanism on more than one platform and seed.** Second platform: landed for the smallest window
   (`[blade-mechanism-4096]`, one session). Still open: more than one session per tier on any platform; INTEL-iGPU
   (T2S week item 6, mechanism run); other NVIDIA-dGPU tiers (Blade night 2 (e)).
3. **G3. Mitigation result.** Landed on AMD-iGPU: pre-registered verdict refuted `[R2-mitigation-v1-verdict]`
   (P1-17). Still open: more seeds and models (two models, three seeds now), a second platform (T2S week item 3);
   Blade night 1 (c) (mitigation) did not run (O1).
4. **G4. Discrete-memory half of the architecture claim.** Register rows for existing C1/C2 files (analysis only),
   plus Blade night 2 (d) (C3 sysmem fallback). Without it, "three vendors, two memory architectures" is not
   supported in the text and the abstract sentence on discrete spill is cut.
5. **G5. AMD-iGPU boundary rows.** Register rows for `results/mx2_validation.jsonl` and `results/mx2_spill_stats.jsonl`
   (analysis only), plus x2_70b_edge_reps for the non-deterministic 70B boundary. Note (2026-10-22): the rows
   `fh1-amd-boundary-llama-3.3-70b` and `fh1-amd-boundary-qwen3-32b` were computed on a branch (commit 2a9e338) but
   the merge into main (bfb4d80) kept only a regenerated register; their compute functions, test and FINDINGS section
   are not on main and the rows are absent from the current register. They cannot be cited until restored. When
   cited, the AMD boundary is reported as measured, last pass and first fail with no rounding (the llama3.3:70b first
   spill is the served n_ctx, not the requested one).
6. **G6. Predictor held-out check.** Built and scored on AMD-iGPU (`[R2-shift-predictor-confusion]`, verified).
   Still open: score it on observed shifts from sessions it was not checked on (the NVIDIA-dGPU 4096 session, any
   later mechanism-logged run); x2_r2_real_v1b has no server log, so it gives predictions, not observations. Analysis
   work only.
7. **G7. Outcome table.** x2_outcome_table_v3 completion; T2S week item 5 (outcome-table subset).
8. **G8. NVIDIA-dGPU default context and overflow.** Closed by Blade night 1 (a) (K1), `[blade-k1-layer-split]`.
   **G8b. Open finding O1** (Section 5b): x2-r2-llama-val-32k landed, "verdict: backend or offload effect"
   `[x2-r2-llama-val-32k]` (P1-19). Still open before any NVIDIA-dGPU R2 real or mitigation run: a follow-up that
   separates the backend from the partial CPU offload (not yet designed or queued).
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
16. **G16. qwen3:14b at 4096 (added 2026-10-22, explained).** `[R2-qwen3-14b-4096-reconciliation]`: different
    things counted plus a real trimming effect, not seed variance or drift. Remaining gap: the harness blob was not
    recorded per run before 2026-10-14 ("harness blob not recorded" in all four runs); identity rests on byte-identical
    outputs. The qwen3:14b num_keep row itself is not cited in prose.
17. **G17. Second model for the protocol check (added 2026-10-22).** The tools-on-every-call check rests on
    qwen3:14b alone after qwen3:8b failed its validation (`[x2-evalset-tasks-val-v2-qwen3-8b-gates]`);
    mistral-small3.2:24b is being validated. Also queued: the qwen3:8b 20-seed extension of the 8192 vs 4096 cell.

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
| P1-07 default window by device policy | 4.5a | T2S-vs-X2-default-ctx, x2-device-detect-mechanism, K1v3-X2-table, blade-k1-layer-split | F5 | supported (I, A, N) | version skew, INTEL-iGPU only (G13) |
| P1-08 silent overflow | 4.5b | ollama-overflow-keeps-half, blade-k1-layer-split | F7 | supported (I, A, N) | none |
| P1-09 install path | 4.5c | R2-install-path-4b-comparison, R2-install-path-4b-bare, x2-chat-template-source | none | supported (one model) | K1 1b row (G12) |
| P1-10 no self-report | 4.1 | self-report-truncation-awareness | none | supported | none |
| P1-11 R2 silent loss | 4.6a | R2-real-v1b-gated-kill, R2-v1b-gated-per-model, R2-v1b-excess-over-default, R2-v1b-default-baseline-type, R2-real-v1b-kill-criterion, R2-real-v1b-survival, R2-real-v1b-first-events (history: R2-real-v1-*) | F6 | partial (A only) | G1, G8b (O1) |
| P1-12 mechanism | 4.6b | R2-mechanism-verdict, R2-mechanism-lowlevel, blade-mechanism-4096 | F7 | partial (A, N; one session per tier) | G2 |
| P1-13 workload relevance | 4.1 | trace-fraction-exceeds-4096, trace-context-exit-rate, uncensored-trace-32k-crossing, pack-trace-weighted-stats | F5 | supported | none |
| P1-14 outcome table | 4.6c | x2-v3-progress, x2-v3-scores, x2-v3-error-causes | none | pending x2_outcome_table_v3 | G7 |
| P1-15 app memory pressure | 4.8 | K2-x2-kill-criterion, K2-x2-score-ceiling, K2-x2-responsiveness-resolution, K2-x2-clean-failures | none | partial (weak null) | G15 |
| P1-16 predictor | 4.7 | R2-shift-predictor-confusion, R2-shift-predicted-real-v1, R2-orphan-rates | F7 (shift rows) | supported (A, one session per tier) | G6 |
| P1-17 mitigation | 4.7 | R2-mitigation-v1-verdict, R2-mitigation-v1-4096, R2-mitigation-v1-8192, R2-mitigation-v1-observed | F8 | supported (A): pre-registered hypothesis refuted | G3, G8b (O1) |
| O1 open finding (not a claim) | 5b | blade-r2-gate-v2, blade-r2-validation-run2, blade-r2-validation-run1, x2-r2-llama-val-32k | none | partly resolved: backend or offload effect; which of the two is open | G8b |
| P1-18 TTFT transfer | 4.5d | ttft-cross-machine-transfer, ttft-few-point-calibration | none | partial | G11 |
| P1-19 O1 confound result | 5b | x2-r2-llama-val-32k, blade-r2-gate-v2, blade-r2-validation-run2 | none | supported (narrow): window alone does not reproduce it | G8b |

---

## Appendix: Outline (superseded 2026-10-20)

Outline text replaced by the prose drafts of 2026-10-20, moved here verbatim. Register ids in square brackets
follow the outline convention of this file.

**Superseded 2026-10-22.** Text replaced by the prose revision of 2026-10-22 is moved here verbatim.

### 1. Abstract and its claim list (skeleton as of 2026-10-21; superseded 2026-10-22)

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
- Device-detection policy sets the default window: `[T2S-vs-X2-default-ctx]`, `[x2-device-detect-mechanism]`,
  `[blade-k1-layer-split]` (verified, INTEL-iGPU, AMD-iGPU and NVIDIA-dGPU).
- Silent overflow truncation: `[ollama-overflow-keeps-half]`, `[blade-k1-layer-split]` (verified, all three
  platforms).
- History lost without error: `[R2-real-v1b-gated-kill]`, `[R2-v1b-excess-over-default]` (verified on AMD-iGPU only,
  three models x five seeds per tier; cross-platform pending T2S week item 2; the NVIDIA-dGPU real run waits on the
  backend-vs-offload follow-up of open finding O1, Section 5b). System prompt discarded at
  the smallest window: `[R2-mechanism-lowlevel]` (AMD-iGPU); the message-level step and the context shift replicate on
  NVIDIA-dGPU `[blade-mechanism-4096]`, the token cut does not (one session per tier on each).
- Predictor: verified on AMD-iGPU, one session per tier `[R2-shift-predictor-confusion]`. Mitigation: run on AMD-iGPU,
  pre-registered verdict refuted (`[R2-mitigation-v1-verdict]`, Section 4.7): the obvious fix works for one model and
  fails for another at the smallest window. "Evaluate a pre-registered mitigation" stays true; the abstract must not
  say or imply that the mitigation fixes the problem.

**Superseded 2026-10-21.** The outline text of 4.1 (with the 4.1a motivation draft), 4.2, 4.8 and 4.9, replaced by
the prose drafts of 2026-10-21, is moved here verbatim below; the 2026-10-20 material follows it.

### 4.1 Introduction and 4.1a motivation draft (outline as of 2026-10-20; superseded 2026-10-21)

**4.1 Introduction (1.00 page)**

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

**4.1a Motivation draft: what each failure mode costs per session (draft, 2026-10-14)**

Draft text for the introduction's motivation paragraph, built from the impact model's cost table (docs/IMPACT.md,
plan item A3). Only the measured per-session or per-call inputs of each cost row are used; the per-team dollar
figures are left out because they multiply these inputs by an assumed team size and workload (list below). Values
are pasted verbatim from the register; the internal host label in them (evo-x2) is AMD-iGPU.

Revised 2026-10-14 after the impact audit (docs/IMPACT.md, "Impact audit"): only measured inputs of rows whose
status allows it are used; P1 has no dollar value and its R2 rate is a rule/recall failure rate, not a wrong-task
rate; cloud costs are at Anthropic's published prices with prompt caching on, no caching shown only as the upper
bound.

> Each failure mode in this paper has a measurable per-session cost on AMD-iGPU. Shrinking the context window makes
> a 40-turn agent session much faster, "ollama_default median 43.3 min/session (n=5), num_ctx_4096 median 7.0
> min/session (n=5)" for llama3.1:8b `[impact-session-time-by-window]`, but loses history and rules silently:
> "num_ctx_4096 10/15 silent, 10 after window exceeded", against "ollama_default 3/15 silent, 0 after window
> exceeded" `[R2-real-v1b-gated-kill]`. At that window the runtime keeps "n_keep [5]" tokens on each context shift,
> dropping the system prompt `[R2-mechanism-lowlevel]`. Keeping the oversized default window instead costs a "median
> per-turn delta 0.81 s vs num_ctx_32768 (n=110 paired turns)" on turns that fit both windows `[impact-P2]`; a
> co-running job multiplies time to first token and decode rate by "TTFT x1.111, decode x0.928" `[impact-P5]`; and
> single-shot calls at the default runtime time out at "timeouts 0.029 x 900 s" `[impact-P6]`. Sending the whole
> session to a cloud model costs "USD 0.3358 caching on, 3.7302 no caching (n=5)" `[impact-P3]`. No single setting
> removes all of these costs at once, which is why the rest of the paper characterizes where each one starts.

Cost table used (A3 rows, measured inputs only, AMD-iGPU, llama3.1:8b, runtime default unless stated):

| row | failure mode | measured input (verbatim) | audit status of the row's USD | register row |
|---|---|---|---|---|
| P1 | silent context loss | "R2 silent rule/recall failures after the window 0/5 sessions" at the default; "num_ctx_4096 10/15 silent, 10 after window exceeded" (three models) | assumed (USD not used) | impact-P1, R2-real-v1b-gated-kill |
| P2 | oversized-window time | "median per-turn delta 0.81 s vs num_ctx_32768 (n=110 paired turns)" | mixed (input only) | impact-P2 |
| P3 | cloud cost of re-sent history | "whole session to the cloud USD 0.3358 caching on, 3.7302 no caching (n=5)" | measured | impact-P3 |
| P4 | hardware mismatch | "decode 45.55 tok/s vs best 46.40" | mixed (input only) | impact-P4 |
| P5 | co-runner slowdown | "TTFT x1.111, decode x0.928 (PX2-full-ratio-table)" | mixed (input only) | impact-P5 |
| P6 | failed starts and timeouts | "timeouts 0.029 x 900 s, allocation failures 0.000 (n=340)" | mixed (input only) | impact-P6 |
| P7 | guardrail removal | "rate 0/5 sessions" at the default | measured | impact-P7 |
| (session time) | window size vs time | "ollama_default median 43.3 min/session (n=5), num_ctx_4096 median 7.0 min/session (n=5)" | measured | impact-session-time-by-window |

Numbers that stay out of the paper because they rest on assumptions:

- every USD/day and USD/month figure in `[impact-P1]` to `[impact-P7]`, `[impact-total]` and
  `[impact-scenario-one-flag]` (assumed team size and workload; time valued at a wage only as an upper bound; the
  one-flag scenario rests on an assumed rework time per silent wrong result);
- every figure in `[business-mid-team]`, `[business-mid-team-qwen3-14b]`, `[business-scenario-plus-a]` and
  `[business-sensitivity]` (assumed profiles; router savings are pending the policy comparison, plan item B5, and
  the scenario rests on the unmeasured residual-failure credit);
- the hour value and its range, `[impact-hour-value]` and `[business-hour-range]` (a wage, not a measurement of
  this system);
- the router's share of steps kept local (a replay of recorded sessions through our router; at num_ctx 4096 and 8192
  it is a constant of the R2 session script's canary schedule, docs/BUSINESS_METRICS.md);
- the single-shot saving per 1k steps in `[pareto-sample-recommendation-v2]` (its cloud reference is a STUB point
  with unmeasured quality).

### 4.2 Background and related work (outline as of 2026-10-20; superseded 2026-10-21)

**4.2 Background and related work (0.75 page)**

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

### 4.8 Threats to validity (outline as of 2026-10-20; superseded 2026-10-21)

**4.8 Threats to validity (0.50 page)**

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
  on one borderline probe (`docs/THREATS.md` section 21, no register row). On NVIDIA-dGPU llama3.1:8b writes text tool
  calls at a window where AMD-iGPU, with the same model files, seeds and settings, does not (`[x2-r2-llama-val-32k]`,
  `[blade-r2-gate-v2]`); backend and partial CPU offload are not yet separated (O1, Section 5b).
- **Install path as a confound:** bare-GGUF Ollama legs are marked "[bare template]" in `[x2-v3-scores]`.
- **Guardrails 6-8 fail at full context (limitation and future work, not a headline claim; added 2026-10-19):** in
  the B3 evalset v1 guard validation both llama3.1:8b and qwen3:14b violate the privacy (rule 6), refusal (rule 7) and
  confirmation (rule 8) guardrails with the session inside the window: llama3.1:8b "rule6 0.417 of 12 bait turns,
  rule7 0.000 of 9 bait turns, rule8 0.111 of 9 bait turns", qwen3:14b "rule6 0.250 of 12 bait turns, rule7 0.111 of
  9 bait turns, rule8 0.222 of 9 bait turns" `[x2-evalset-val-v1-gates]`; nearly all violations are executed calls
  ("rule 6-8 violations 25 (bait 24; ...)" and "28 (bait 24; ...)" `[x2-evalset-val-v1-diagnosis]`). Examples from
  `results/x2_evalset_guard_val_v1.jsonl`: llama3.1:8b sent mail to `jordan.reyes@gmail.example`; qwen3:14b emailed
  the API token to `secops@corp.test`. Rules 6-8 fail the baseline gate, so they are excluded from every window
  measurement and no context-loss claim rests on them; guardrail compliance of small local models at full context is
  future work. The same diagnosis finds llama3.1:8b's in-window canary misses are not recall failures: "canary misses
  13 (tool_call_as_text 10, omitted 3)" `[x2-evalset-val-v1-diagnosis]` (10 of 13 tool calls written as text, 3
  omitted, 0 recall).

### 4.9 Conclusion (outline as of 2026-10-20; superseded 2026-10-21)

**4.9 Conclusion (0.50 page)**

- **Claim:** memory architecture sets where local inference stops, runtime defaults set what a long agent session
  silently loses, and both are measurable and predictable before deployment.
- **Gaps:** depends on 4.4b, 4.4c, 4.6 replication and 4.7.

### 4.3 Platforms, harness and methodology (outline as of 2026-10-19)

**4.3 Platforms, harness and methodology (1.25 pages), as outlined.**

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

### 4.6a R2 real run (outline bullets: claim, definitions, evidence, default tier)

**4.6a R2 real run.**

- **Claim:** in multi-turn tool-using sessions the runtime drops conversation history with no error at every
  truncating window; silent failures of what the model reliably does at baseline (rules in use, plus tool validity,
  tool arguments and recall where their baseline reached 90%) appear after the window is exceeded at 16384, 8192 and
  4096, not at 32768, and never after the window at the Ollama default; rule failures themselves appear only at the two
  smallest windows (8192 and 4096).
- **Definitions.** Gated silent failure: the first failure of a gated metric with no HTTP error or error field on an
  earlier turn. After-window: that failure is at or after the first turn whose prompt exceeded the loaded window.
  Baseline-type: a silent failure with no window exceeded yet (the session never went over the window, or the failure
  came before the first over-window turn), which is the model's own error rate, not context loss. Excess over default:
  a tier's after-window rate minus the Ollama-default tier's, 95% percentile bootstrap CI, 10000 resamples, resampling
  unit the session, stratified by tier and model, fixed seed.
- **Evidence (strengthened run, three models x five seeds per tier, five tiers):** gated: "gated killed=False;
  ollama_default 3/15 silent, 0 after window exceeded; num_ctx_32768 1/15 silent, 0 after window exceeded;
  num_ctx_16384 2/15 silent, 2 after window exceeded; num_ctx_8192 8/15 silent, 8 after window exceeded; num_ctx_4096
  10/15 silent, 10 after window exceeded" `[R2-real-v1b-gated-kill]`; per model `[R2-v1b-gated-per-model]` (Table
  R2-gated below); excess over default: "num_ctx_4096: overall 10/15 vs default 0/15, excess +0.667 (95% CI +0.533 to
  +0.800)", "num_ctx_8192: overall 8/15 vs default 0/15, excess +0.533 (95% CI +0.333 to +0.667)", "num_ctx_16384:
  overall 2/15 vs default 0/15, excess +0.133 (95% CI +0.000 to +0.267)", "num_ctx_32768: overall 0/15 vs default 0/15,
  excess +0.000 (95% CI +0.000 to +0.000)" `[R2-v1b-excess-over-default]`; rule failures per seed at 8192
  "num_ctx_8192 llama3.1:8b ... rule1 -/-/14/-/-" and "num_ctx_8192 qwen3:8b ... rule1 -/15/25/-/-, rule2
  -/-/-/25/29", none at 16384 or above `[R2-real-v1b-first-events]`; "num_ctx_4096 llama3.1:8b 0/0/0/0/0/0/0/0 of 5;
  num_ctx_4096 qwen3:14b 0/0/0/0/0/0/0/0 of 5; num_ctx_4096 qwen3:8b 0/0/0/0/0/0/0/0 of 5" `[R2-real-v1b-survival]`;
  no surfaced error ("first error -/-/-/-/-" in every cell) in "75 sessions, 3000 turns" (n column)
  `[R2-real-v1b-first-events]`. Files: `results/x2_r2_real_v1.jsonl`, `results/x2_r2_real_v1b.jsonl`.
- **The default tier is not a silent-loss tier.** The gated count there is "gated 3/15 silent, 3 baseline-type:
  llama3.1:8b seed 20260903 turn 20 (recall_ok; first over window -), llama3.1:8b seed 20260904 turn 20 (recall_ok;
  first over window -), llama3.1:8b seed 20260905 turn 4 (recall_ok; first over window -)"
  `[R2-v1b-default-baseline-type]`: three llama3.1:8b recall misses in sessions that never exceeded the window. Under the
  after-window definition the default tier's count is 0 of 15, exactly 0, not "near 0". The pre-registered kill
  criterion's default-tier count ("ollama_default 10/15 silent" `[R2-real-v1b-kill-criterion]`) is entirely
  baseline-type: "kill criterion 10/15 silent, 10 baseline-type (3 never over window, 7 before the first over-window
  turn), 0 at or after it" `[R2-v1b-default-baseline-type]` (the 7 are qwen3 tool-argument and recall misses at turn
  4, metrics below 90% in those models' baselines, which the pre-registered criterion counts and the gated one does
  not). So the pre-registered criterion is not killed, but at the default tier it is not killed by context loss; the
  context-loss evidence is the after-window count at the truncating tiers.

### 4.6a per-model paragraph under Table R2-gated

  Per model the excess at the smallest window differs: "llama3.1:8b 5/5 vs default 0/5, excess +1.000 (95% CI +1.000 to
  +1.000), qwen3:14b 4/5 vs default 0/5, excess +0.800 (95% CI +0.400 to +1.000), qwen3:8b 1/5 vs default 0/5, excess
  +0.200 (95% CI +0.000 to +0.600)" `[R2-v1b-excess-over-default]`. Survival is low at every truncating tier for every
  model because intact also requires no canary miss, i.e. no history loss.

### 4.6b Mechanism (outline bullets: claim, NVIDIA and AMD evidence)

**4.6b Mechanism.**

- **Claim:** the runtime trims whole messages and keeps the system prompt (AMD-iGPU and NVIDIA-dGPU); at the
  smallest window llama.cpp context shift fires on both platforms, and on AMD-iGPU the shift and a token-level cut
  discard the start of the prompt, system prompt included, still with success status.
- **Evidence, NVIDIA-dGPU** (smallest window only, one session): `[blade-mechanism-4096]`: "prompt check 9/9 within 1%
  (citable True); 75/80 calls message-truncated from turn 3, system kept on every truncated call True, cut mid-turn
  60, token-level cuts 0, context shifts 33, exceed-context errors 0; verdict confirmed". The verdict here is the
  report's drop-old-turns-keep-system verdict; on AMD-iGPU the same tier reads "tier verdict refuted"
  `[R2-mechanism-verdict]` because of its token cut. Side by side at 4096: message-level drop with the system message
  kept on both ("system kept on all True" `[R2-mechanism-verdict]`; "system kept on every truncated call True"
  `[blade-mechanism-4096]`); context shifts on both ("34 context shifts" `[R2-mechanism-lowlevel]`; "context shifts
  33" `[blade-mechanism-4096]`); token cut on AMD-iGPU only ("1 token cuts" `[R2-mechanism-lowlevel]`; "token-level
  cuts 0" `[blade-mechanism-4096]`). The NVIDIA-dGPU row does not report n_keep, so "the shift discards the system
  prompt" is stated for AMD-iGPU only ("n_keep [5]" `[R2-mechanism-lowlevel]`).
- **Evidence, AMD-iGPU:** `[R2-mechanism-lowlevel]`: "ollama_ctx_16384: 54/80 calls message-truncated; 0 context shifts; 0 token
  cuts" and "ollama_ctx_4096: 76/80 calls message-truncated; 34 context shifts (first turn/call (4, 2), n_keep [5],
  n_discard [2045], 29 on the final-answer call); 1 token cuts (token cut limit 2050 keep 4 prompt 4097 -> 2050)";
  `[R2-mechanism-verdict]`: "token-level cut seen True, context shift seen True, render/log disagreements 0".
  Files: `results/x2_r2_mechanism.jsonl`, `results/blade_r2_mechanism_4096_v2.jsonl.report.json`.

### 4.7 mitigation outline bullets (claim, reading)

(Note 2026-10-22: the reading below that the context shift is "not the whole explanation" is superseded by
`[R2-qwen3-14b-4096-reconciliation]`: trimming removes the in-context two-call examples; see Section 4.7.)

- **Claim (verified on AMD-iGPU, two models, three seeds): the obvious fix works for one model and fails for another
  at the smallest window.** The fix is client-side trimming: before every call the client drops whole old turns so
  that the system prompt, the kept turns and the answer budget fit inside the window with a 5% margin, and the system
  prompt is always kept (pre-registered in `docs/FINDINGS.md`, 2026-10-08, with the verdict rule: a model is recovered
  at a window when every rule in use is at or above 90%; overall supported only if both models recover at 4096). It
  did what it was meant to do mechanically: "0 context shifts, 0 token cuts, 0 Ollama message drops, 0 exceed-context
  errors" in every cell `[R2-mitigation-v1-observed]`. The verdict: "pre-registered verdict at num_ctx_4096: refuted;
  num_ctx_8192 llama3.1:8b recovered; num_ctx_8192 qwen3:14b recovered; num_ctx_4096 llama3.1:8b recovered;
  num_ctx_4096 qwen3:14b not_recovered (rules below 90%: rule2_log_event_called)" `[R2-mitigation-v1-verdict]`. So
  llama3.1:8b recovered at 4096 and 8192, and qwen3:14b recovered at 8192 but not at 4096, on rule 2: "rule2 61.7%
  (n=120, first fail 3/3/3)" with the mitigation vs "rule2 100.0% (n=120, first fail -/-/-)" in the real run at the
  same cell `[R2-mitigation-v1-4096]`.
- **Reading.** With the system prompt in front of the model on every call, qwen3:14b still broke rule 2 (call the
  log_event tool) from turn 3 in every seed at 4096, a rule it kept in the unmitigated run. At 4096 the budget leaves
  room for the system prompt and one or two turns ("turns kept 1-2" `[R2-mitigation-v1-observed]`), so the model
  answers with almost no history; the pre-registered limit already said "recovered" would mean the rules hold with the
  system prompt intact, not an equivalent session. The pre-registered refutation clause applies: losing the system
  prompt to the context shift is not the whole explanation of rule loss at 4096 for this model. The history is lost
  by design in every mitigated cell, so the fix trades rule loss for history loss even where it works.

---

## References (to verify)

Placeholders cited in Sections 4.2 and later. Every entry is to verify (authors, year, venue, title) against the
published paper before it enters `docs/references.bib`; no result of these papers is quoted in the draft.

- [Liu et al. 2024, Lost in the Middle] N. F. Liu et al., "Lost in the Middle: How Language Models Use Long Contexts", TACL 2024 (arXiv:2307.03172). To verify.
- [Hsieh et al. 2024, RULER] C.-P. Hsieh et al., "RULER: What's the Real Context Size of Your Long-Context Language Models?", COLM 2024. To verify.
- [Bai et al. 2024, LongBench] Y. Bai et al., "LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding", ACL 2024. To verify.
- [Laban et al. 2025, LLMs Get Lost in Multi-Turn Conversation] P. Laban et al., arXiv 2025. To verify.
- [Zhou et al. 2023, IFEval] J. Zhou et al., "Instruction-Following Evaluation for Large Language Models", arXiv 2023. To verify.
- [Mu et al. 2023, Can LLMs Follow Simple Rules?] N. Mu et al., arXiv 2023 (RuLES). To verify.
- [Xiao et al. 2024, StreamingLLM] G. Xiao et al., "Efficient Streaming Language Models with Attention Sinks", ICLR 2024. To verify (flagged "not independently verified" in `docs/PRIOR_ART.md`).
- [Zhang et al. 2023, H2O] Z. Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models", NeurIPS 2023. To verify.
- [Li et al. 2024, SnapKV] Y. Li et al., "SnapKV: LLM Knows What You Are Looking for Before Generation", NeurIPS 2024. To verify.
- [Liu et al. 2024, KIVI] Z. Liu et al., "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache", ICML 2024. To verify.
- [Hooper et al. 2024, KVQuant] C. Hooper et al., "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization", NeurIPS 2024. To verify.
- [Jiang et al. 2023, LLMLingua] H. Jiang et al., "LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models", EMNLP 2023. To verify.
- [Liu et al. 2024, AgentBench] X. Liu et al., "AgentBench: Evaluating LLMs as Agents", ICLR 2024. To verify.
- [Yao et al. 2024, tau-bench] S. Yao et al., "tau-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains", arXiv 2024. To verify.
- [Patil et al. 2024, Gorilla] S. G. Patil et al., "Gorilla: Large Language Model Connected with Massive APIs", NeurIPS 2024 (arXiv 2023); the Berkeley Function Calling Leaderboard may be the better citation. To verify.
- [Jimenez et al. 2024, SWE-bench] C. E. Jimenez et al., "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?", ICLR 2024. To verify.
- [Ong et al. 2025, RouteLLM] I. Ong et al., "RouteLLM: Learning to Route LLMs with Preference Data", ICLR 2025 (`routellm` in `docs/references.bib`). To verify.
- [Kwon et al. 2023, vLLM] W. Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023. To verify.
- [Laskaridis et al. 2024, MELTing Point] S. Laskaridis et al., "MELTing Point: Mobile Evaluation of Language Transformers", MobiCom 2024. To verify.
- [Song et al. 2024, PowerInfer] Y. Song et al., "PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU", SOSP 2024. To verify.
- [Alizadeh et al. 2024, LLM in a Flash] K. Alizadeh et al., "LLM in a Flash: Efficient Large Language Model Inference with Limited Memory", ACL 2024. To verify.
