# Paper 1 Draft (ISPASS 2027 target, anonymized)

Status (2026-10-22): advisor version. This file is the paper only: title, abstract, Sections 4.1 to 4.9 (paper
Sections 1 to 9), figure and table specs, and references. Internal tracking material (status history, claim table,
open findings, "Obvious?" defense, figure-source list, gaps, claim-to-evidence index, superseded outlines) is in
`docs/paper1/INTERNAL_TRACKING.md`; author and label mapping in `docs/PAPER1_COVER_NOTE.md`. The typeset version is
`docs/paper1/paper1.tex`. Every measured number carries its register id in an HTML comment (a LaTeX comment in the
typeset version); design constants carry "design: <source>".

---

## 1. Working title and abstract

**Working title:** *Silent by Default on Ollama: How Memory Architecture and Runtime Policy Shape Local LLM Agents on AI PCs*

Alternatives:

1. *Where Local Agents Break: Memory Boundaries, Runtime Defaults and Silent Context Loss on Ollama Across Three AI PC Platforms*
2. *The Runtime Picks Your Context Window: A Cross-Vendor Characterization of Local LLM Inference on AI PCs*
3. *No Error, No Rules: Characterizing Silent Failure of Ollama-Served LLM Agents Across Unified and Discrete Memory*

**Abstract.**

When a local agent's conversation outgrows the context window, Ollama trims it and still reports success, while a
stock llama-server at the same small window fails loudly, with an HTTP error on every wrong task turn <!-- xruntime-llamacpp-wrong-rate -->. We measure what is lost
with pre-registered experiments on an AMD APU, an NVIDIA laptop GPU and an Intel mini PC. At windows of 16384 tokens
and below, no session of three 8B to 14B models kept its full history <!-- R2-real-v1b-survival -->, and models
up to 70B lose history once the window is exceeded <!-- H-scale-verdict --> <!-- H-scale-family-verdict -->. Rule loss
appears in the 8B and 14B models (5 seeds) and in qwen3:32b (2 of 3 sessions at 4096) <!-- H-scale-verdict -->, and
not in llama3.3:70b or mistral-small3.2:24b (0 of 3 each) <!-- H-scale-verdict --> <!-- H-scale-family-verdict -->.
These cross-model results are R2 rule-following evidence, not B3 task accuracy.
On a tool-using task workload (B3), where qwen3:14b is the one model that passed validation, its wrong-task rate rises from 11.3% (17/150) at
its default window to 42.0% (63/150) at 4096 on Ollama, and every wrong task is silent <!-- x2-evalset-tasks-real-v2-wrong-rate -->;
on llama-server the 4096 effect replicates (+92.0 points, 95% CI +88.7 to +94.7) but no wrong task is silent <!-- xruntime-llamacpp-wrong-rate -->.
Runtime defaults decide how often this happens: an unmodified install loaded a 4096-token window on an Intel CPU
<!-- t2s-r2-real-cpu --> and 32768 on the same machine's integrated GPU <!-- t2s-r2-real-igpu -->, and in one session of
five, switching only the GPU backend made llama3.1:8b write a tool call as text <!-- x2-r2-llama-val-32k-vulkan --> <!-- blade-r2-gate-v2 -->
<!-- x2-r2-llama-val-32k-offload -->. The fixes are partial: KV-cache quantization cut session time by 65% to 85%
(139.4 of 214.8 and 37.1 of 43.7 min per session, 3 sessions per setting) with no measured loss of rules or recall <!-- cost-cut-config-time --> <!-- b5-fa-control -->
<!-- b5-kv-recall-f16-rerun -->, while client-side trimming and `num_keep` each repair only part of the loss.

---

### 4.1 Introduction

<!-- Convention as in 4.3: measured numbers carry their register id; design constants carry "design: <source file>". -->

Runtimes such as Ollama and llama.cpp serve open-weight models of 8B to 70B parameters on laptops and small APU
workstations, and local agents drive them for tens of turns, appending every tool result to a conversation that only
grows inside a fixed context window. When the conversation outgrows it, the runtime, not the agent, decides what to
remove, and on Ollama it does so silently: the request still returns HTTP 200
<!-- blade-k1-layer-split -->; llama-server at the same window fails loudly instead, with HTTP errors <!-- xruntime-llamacpp-wrong-rate -->. The
answer is well formed, and no response field reports the loss.

**Why run locally, and what it costs.** Agent tasks are expensive in the cloud because every step re-sends the whole
conversation. Priced on public agent traces (not executed, with token counts recalibrated to the cloud tokenizer), a
median task of a long-horizon coding-agent set costs USD 2.5307 on a mid-tier cloud model and USD 0.1354 on a small
one with realistic caching <!-- cloud-cost-per-real-task -->. Our own tool-using tasks are short: run fully in the
cloud, the small and mid-tier models were correct on 87/90 and 90/90 tasks <!-- all-cloud-b3-quality --> at a mean
USD 0.00067 and USD 0.01020 per task <!-- all-cloud-b3-cost -->. Running the same tasks locally on qwen3:14b at its
default window removes the bill but not the risk: 80/90 tasks were correct, and nothing marked the 10 that were not
<!-- hybrid-savings-v1 -->. The runtime's own window-overflow flag does not find them either (Section 4.7p). A perfect
detector would reach 90/90 by escalating 10 tasks <!-- hybrid-savings-v1 -->, but that is a ceiling, and no such
detector exists; fixing a wrong answer afterwards in the cloud fixed 84.8% (167/197) <!-- cloud-redo-full-context -->
at a mean of USD 0.0227 per task <!-- cloud-redo-full-context -->, but a plain redo cannot undo actions already
executed (Section 4.7).

A motivating case shows how easily a user lands in this regime. On an Intel mini PC, an unmodified Ollama install
served two models with a 4096 <!-- t2s-r2-real-cpu --> token window, because device detection drops the integrated GPU
and falls back to the CPU <!-- x2-device-detect-mechanism -->, and every agent session then failed silently; with
the integrated GPU enabled, none failed after the window (Section 4.5). Nobody chose 4096; the runtime did.

This paper asks what a local agent loses when its conversation exceeds the context window, whether anything tells the
user, and what decides how often it happens. We answer with pre-registered experiments on an AMD APU, an NVIDIA
laptop GPU and an Intel mini PC (CPU and integrated GPU), spanning unified and discrete memory. First, on Ollama
history loss is silent and appears in every model we tested, from 8B to 70B <!-- H-scale-verdict -->
<!-- H-scale-family-verdict -->. Second, rule loss is model-dependent, and we do not claim it in general; these first
two parts are R2 rule-following evidence across models, not B3 task accuracy. Third, the cost is concrete: on the B3
task workload, where qwen3:14b is the one model that passed validation, its share of tasks ending in a wrong state
rises from 11.3% (17/150) at the default window to 42.0% (63/150) at 4096, every wrong task silent on Ollama
<!-- x2-evalset-tasks-real-v2-wrong-rate -->. Fourth, the runtime's defaults and backend decide how often this
happens. Fifth, the fixes are partial.

We make four contributions.

1. **A cross-vendor map of where memory runs out and how it fails** (Section 4.4). On the Intel integrated GPU a model
   start succeeds or fails at one shared-heap limit that also counts host-visible buffers, and a first-principles
   accounting of weights, KV cache and compute predicts the measured flip point to within -257 to -26 tokens
   <!-- A-24-flip-point-prediction -->.
2. **Runtime policy, not hardware, sets the effective window, and on Ollama overflow is silent** (Section 4.5). The same runtime
   gives llama3.1:8b a default window of 4096 tokens on the Intel mini PC and 131072 on the AMD APU
   <!-- T2S-vs-X2-default-ctx -->, and answers an over-length prompt with success after keeping num_ctx/2 + 2 tokens
   <!-- ollama-overflow-keeps-half -->.
3. **The consequences for agents, with their mechanism** (Section 4.6). On Ollama, silent failures after the window grow as the
   window shrinks; Ollama drops whole old messages but keeps the system prompt, and at the smallest window llama.cpp's
   context shift and a token cut discard it <!-- R2-mechanism-lowlevel -->.
4. **Pre-registered fixes, with their limits, and a characterization of the runtime's shift rule** (Section 4.7). Client-side trimming is refuted as a
   general fix on two vendors <!-- R2-mitigation-v1-verdict --> <!-- t2s-r2-mitigation-cpu -->; `num_keep` lowers
   qwen3:14b's wrong-task rate at 4096 to 30.0% (45/150) without removing the excess
   <!-- x2-evalset-tasks-real-v2-wrong-rate -->; a router to a cloud model is refuted on the AMD platform
   <!-- b5-verdict -->, where a quantized KV cache was on the frontier instead <!-- b5-policy-metrics -->, while under a
   simulated window cap the router joins the frontier beside client trimming <!-- b5-sim-constraint -->;
   and the runtime's own context-shift rule, transcribed from the llama.cpp source, matches every observed shift in the
   sessions it was checked on (misses 0 <!-- R2-shift-predictor-confusion -->); we characterize it to explain
   system-prompt loss, not as a detector, and it does not predict wrong answers.

### 4.2 Background and related work

<!-- Every citation below is a placeholder to verify; see "References (to verify)" at the end of this file. -->

**Context windows in local runtimes.** llama.cpp allocates a KV cache for a fixed number of tokens, n_ctx, when a
model loads. When a sequence would grow past it, the server can perform a *context shift*: it keeps the first n_keep
tokens, discards a block of the tokens after them and continues. Ollama serves models through a bundled llama.cpp
server and adds two layers. Its chat endpoint drops whole messages from the front of an over-length conversation
while keeping the system message <!-- R2-mechanism-verdict -->, and it starts the server with context shift enabled
and a keep of 4 tokens, so an over-length prompt succeeds after keeping num_ctx/2 + 2 tokens
<!-- ollama-overflow-keeps-half -->. The request option `num_keep` sets the kept prefix, which matters only on the shift
and cut paths. If the client sends no `num_ctx`, Ollama chooses the window from the detected device (Section 4.5).
None of these steps is reported to the client.

**Long-context behaviour.** Lost in the Middle [Liu et al. 2024, Lost in the Middle; to verify] finds the middle of a
long input used worse than its ends; RULER [Hsieh et al. 2024, RULER; to verify] and LongBench [Bai et al.
2024, LongBench; to verify] find effective context shorter than the advertised window; and multi-turn [Laban et al.
2025, LLMs Get Lost in Multi-Turn Conversation; to verify] and rule-following benchmarks [Zhou et al. 2023, IFEval; to
verify] [Mu et al. 2023, Can LLMs Follow Simple Rules?; to verify] measure how well a model uses or obeys its input.
All assume the model receives the whole input; we study what the runtime removes before that.

**KV-cache management.** StreamingLLM [Xiao et al. 2024, StreamingLLM; to verify] keeps initial attention-sink tokens
plus a recent window, much as a context shift keeps n_keep tokens; H2O [Zhang et al. 2023, H2O; to verify] and SnapKV
[Li et al. 2024, SnapKV; to verify] evict by attention; KIVI [Liu et al. 2024, KIVI; to verify] and KVQuant [Hooper et
al. 2024, KVQuant; to verify] quantize the cache; LLMLingua [Jiang et al. 2023, LLMLingua; to verify] compresses the
prompt. These are designed policies evaluated at fixed memory; we measure what shipped runtimes discard by default,
and treat the runtime's built-in KV quantization as one context policy among several (Section 4.7).

**Agents, routing and verification.** AgentBench [Liu et al. 2024, AgentBench; to verify], tau-bench [Yao et al. 2024,
tau-bench; to verify], function-calling evaluations [Patil et al. 2024, Gorilla; to verify] and SWE-bench [Jimenez et
al. 2024, SWE-bench; to verify] score task success, usually with hosted models and large windows; we vary the window
under a fixed model, runtime and seed. Routers [Ong et al. 2025, RouteLLM; to verify] [Ding et al. 2024, Hybrid LLM; to
verify] pick a small or a large model from a learned estimate of difficulty, cascades [Chen et al. 2023, FrugalGPT; to
verify] escalate when a scorer distrusts the cheap answer, and self-verification [Weng et al. 2023, Self-Verification;
to verify] and sampling-consistency checks [Manakul et al. 2023, SelfCheckGPT; to verify] let a model test its own
answer. All need a signal that correlates with a wrong answer. The runtime's own over-window signal, which we
characterize and do not contribute, is blind to wrong answers produced inside the window (Section 4.7p).

**Edge-cloud delegation and model fit.** EdgeWise [Shi et al. 2026, EdgeWise; to verify] learns which steps of a long
agent task to delegate to a cloud model; EC2-Bench [Huang et al. 2026, EC2-Bench; to verify] benchmarks edge-cloud
collaboration strategies; Minions [Narayan et al. 2025, Minions; to verify] has a cloud model decompose tasks for an
on-device model; Arch-Router [Tran et al. 2025, Arch-Router; to verify] routes by user-defined preferences; and llmfit
[llmfit; to verify] estimates which models fit a machine's memory. An open runtime issue [Ollama issue #14259; to
verify] reports that chat-history truncation is silent. None of these measures how the window a runtime loads changes
an agent's correctness.

**Inference on edge hardware.** PagedAttention [Kwon et al. 2023, vLLM; to verify] manages KV memory on servers; MELT
[Laskaridis et al. 2024, MELTing Point; to verify], PowerInfer [Song et al. 2024, PowerInfer; to verify] and LLM in a
flash [Alizadeh et al. 2024, LLM in a Flash; to verify] characterize or accelerate inference on phones, consumer GPUs
and flash-backed devices for throughput, memory and energy. We add behavioural correctness under each runtime's memory
and context policy.

### 4.3 Platforms, harness and methodology

<!-- Convention: measured numbers carry their register id; harness design constants carry "design: <source file>". -->

**Platforms and runtimes.** We study three consumer platforms from three GPU vendors: an AMD Ryzen AI Max+ (Strix
Halo) APU whose Radeon iGPU shares unified memory with the CPU; a laptop with an NVIDIA RTX 4070 Laptop GPU and 8188
MiB <!-- blade-k1-layer-split --> of discrete VRAM; and an Intel Core Ultra (Panther Lake) mini PC with an Arc iGPU on
unified memory. Ollama's device detection drops the Intel iGPU and falls back to the CPU
<!-- x2-device-detect-mechanism -->, so we measure Intel twice: on the CPU, as shipped, and on the iGPU, enabled only in
the environment of the job's own server process. All agent runs use Ollama; the AMD and NVIDIA runs share one pinned release and the model files. Versions and
drivers are listed in Table 1.

**The R2 agent harness.** A session is a seeded, deterministic agent conversation of 40 turns
<!-- R2-v1b-qwen3-8b-reversal -->. The system prompt carries five <!-- design: harness/x2_r2_agent.py --> checkable
rules (JSON answer with fixed keys, call a `log_event` tool, never print a forbidden token, lengths in metres, end with
a session code), a two-tool schema and recall facts. Each turn is two <!-- design: harness/x2_r2_agent.py --> calls:
on the first the model emits tool calls and the harness simulates results with a long filler payload that grows the
context as real tool output does; the second asks for the final answer. Temperature is 0
<!-- design: harness/x2_r2_agent.py -->. Canaries in the system prompt and history tags in the first user message,
which a message-level trim drops first, detect lost context; each canary turn asks for a new pair, so a model cannot copy a reused canary from its own earlier answer
<!-- R2-validation-baseline -->. A miss counts as truncation
only while the final prompt exceeds the context Ollama reports as loaded (`/api/ps`, read every turn), never by the
runtime's prompt count. All R2 runs withhold tools on the final call (the v1 protocol), because with tools offered there llama3.1:8b often
answered with a tool call and no text; Section 4.6a repeats the main contrast with tools on every call.

**Validation gates.** Before any real run, each model passes a validation at a window where nothing is truncated,
with gates fixed in advance. A rule is *in use* for a model only if the model follows it on at least 90%
<!-- x2-evalset-val-v2-gates --> of baseline turns; rules below the bar are dropped, not repaired, because the failures
are real behaviour. For llama3.1:8b only rules 1, 3 and 4 <!-- R2-validation-v2-baseline --> are in use; qwen3:14b kept all five.
Negative and positive controls and a task-tool gate complete the list, and a real job refuses to start unless its
model's validation passes.

**Gated silent failure.** For each session we take the first turn on which a gated metric fails. The failure is
*silent* when no HTTP error or error field appeared on any earlier turn, and *after-window* when it falls at or after
the first turn whose prompt exceeded the loaded window; otherwise it is *baseline-type*, the model's own error rate
rather than context loss. The window effect is the *excess over default*: a tier's after-window rate minus the Ollama
default tier's, with a 95% percentile bootstrap CI over sessions (10000 <!-- R2-v1b-excess-over-default -->
resamples, stratified by tier and model). A 0/n or n/n cell has no bootstrap variance, so where both cells are
0/n or n/n we report a one-sided Fisher exact test with the session as the unit instead <!-- session-exact-tests -->; one cell was later extended to 10 seeds
<!-- R2-qwen3-8b-reversal-10seed --> by its own pre-registration.

**Pre-registration and provenance.** The R2 kill criterion was written before any session ran: the claim fails if,
in every arm, the first failure comes only after an explicit error has surfaced in the same session. The gated
definition came later, as analysis, and is reported beside the criterion, never in its place. Every follow-up was
pre-registered the same way (hypothesis, verdict rule and validity conditions committed before deployment), and we
report verdicts in that wording, refutations included. Every number is
pasted from a numbers register that one script regenerates from the raw result files; a claim with no verified row
is not made. One provenance gap remains: runs before 2026-10-14 did not record the harness file's blob, so their
harness identity rests on byte-identical outputs across reruns (identical final text 120/120 <!-- R2-qwen3-14b-4096-reconciliation -->). Rows are never pooled across platforms, and a configuration flag counts
only when the runtime's log shows it applied: an earlier sweep passed a KV-cache precision setting that the runtime
silently ignored.

**Table 1 and figure spec.**

- **Table 1.** Platforms and versions: CPU and GPU, memory architecture, GPU driver, llama.cpp build for direct runs,
  Ollama version and the GPU backend Ollama actually used, OS build, per platform. Configuration record, not a measured
  result; the values come from the platform configuration record and are pasted when its register row lands.
- **F1.** Platform and memory-architecture diagram: the three platforms, where KV cache, weights and host-visible
  buffers live, and which runtime backend is used (drawing only).

### 4.4 Memory-architecture boundaries

**Claim.** The same event, a context window whose KV cache no longer fits the memory the GPU is meant to use, has a
different cost shape on each memory architecture: on the AMD APU the server keeps running past the boundary at almost
unchanged speed, on the Intel iGPU the server either starts or refuses at one shared-heap limit, and on the discrete
GPU the cost is expected to grow with the share of the model pushed out of VRAM. Figure FH1 carries this claim; the AMD and Intel
panels are measured and the NVIDIA panel is not yet complete.

**AMD APU: a soft spill, then a hard crash far beyond it.** We ran llama-3.3-70b and qwen3-32b in llama-server with
the Vulkan backend and every layer on the GPU (`-ngl 99` <!-- fh1-amd-boundary-llama-3.3-70b -->), stepping the
window and recording per-process dedicated and shared GPU memory, decode rate and time to first token (TTFT). The
platform offers three candidate boundaries: a firmware-reserved dedicated block of 65536 MiB, a Vulkan device-local
heap of 76218 MiB, and a Vulkan total of 114327 MiB <!-- fh1-amd-boundary-llama-3.3-70b -->. We call a point a spill
when shared GPU usage grows by more than 1024 MiB over the smallest window <!-- fh1-amd-boundary-llama-3.3-70b -->.
For llama-3.3-70b the last window without spill is 65536 and the first window with spill is 100096, the window the
server reported for a request of 100000 <!-- fh1-amd-boundary-llama-3.3-70b -->. At 100096 the process held 1.122 of
the dedicated block but only 0.965 of the device-local heap <!-- fh1-amd-boundary-llama-3.3-70b -->: the onset
follows the dedicated block, not the heap the Vulkan driver advertises, so FH1 normalizes the AMD x axis by the
dedicated block and says so. With one 1855-token prompt at every window <!-- fh1-amd-boundary-llama-3.3-70b -->, crossing the onset
is nearly free: the decode ratio across it is 0.983 and the TTFT ratio 1.004 <!-- fh1-amd-boundary-llama-3.3-70b -->; decode fell from 5.284 +/- 0.004 tok/s at 65536 to 5.196 +/- 0.010
tok/s at 100096, and at 131072, with 18540 MiB more in shared memory than at the smallest window, it was still 5.176
+/- 0.015 tok/s <!-- fh1-amd-boundary-llama-3.3-70b -->. Nothing in the server's response tells the client it has
spilled.

The hard boundary sits much further out and is reached only through clamping. The model's trained context is 131072
<!-- fh1-amd-boundary-llama-3.3-70b -->, and a request above it is served at that value: the last window that started
was 221440, clamped to 131072, and the first that failed was 221696, with exit code 3221226505
<!-- fh1-amd-boundary-llama-3.3-70b -->. So the AMD platform has three regimes for this model: it fits, it spills
silently at almost no cost (llama-server, Vulkan <!-- fh1-amd-boundary-llama-3.3-70b -->), and, for requests far above the trained context, it crashes.

For qwen3-32b no unclamped spill point was measured, so the row has no spill cost
<!-- fh1-amd-boundary-qwen3-32b -->. A dedicated qwen3-32b spill sweep (llama-server b10970, Vulkan, `-ngl 99`, 1 warm-up + 5 calls per point)
from 20480 to 40960 found no unclamped spill point up to 40960: shared GPU memory grew by at most 20 MiB over 20480,
and decode stayed between 11.271 and 11.368 tok/s <!-- fh1-amd-qwen3-32b-spill-sweep -->. The qwen3-32b onset is
therefore reported as "above 40960" for unclamped windows only.

**Intel iGPU: a hard wall that first-principles accounting predicts.** On the Intel iGPU there is no soft regime: a
model either starts or the runtime refuses it. Across five models, the refusal point is consistent with one
shared-heap limit L in (48112.7, 48124.0] MiB, 247 MiB above the 47866 MiB budget the driver reports, from 92 lower
and 22 upper bounds <!-- A-24-effective-heap-limit -->. The limit counts host-visible buffers as well as device memory:
accounting for device memory alone is inconsistent, because the largest start (47945.2 MiB) exceeds the smallest
refusal (47562.5 MiB) <!-- A-24-effective-heap-limit -->. Counting weights, KV cache and compute buffers against the
shared heap, with L fitted leave-one-model-out, predicts each model's flip point within -257 to -26 tokens (-0.11% to
-0.01%), against -2890 to -820 tokens for model file size alone and -1092 to +968 tokens for device-only accounting
<!-- A-24-flip-point-prediction -->. The flip points span an order of magnitude: (23296, 23552] tokens for
llama-3.3-70b and (115712, 115968] for qwen3-32b <!-- A-24-flip-point-per-model -->. Two limits apply. These are
start tests only, and for several models the flip point lies beyond the trained context, so it is a memory boundary,
not a usable window. And the result is for one driver and boot state; we do not claim the 247 MiB margin transfers.

**NVIDIA discrete GPU: placement measured, cost pending.** With 8188 MiB of VRAM <!-- blade-k1-layer-split -->, Ollama keeps llama3.1:8b fully on the GPU
at windows of 4096 and 8192 and moves layers to the CPU above them, with VRAM shares of 0.870 at 16384 and 0.656 at
32768; for qwen3:8b the shares are 0.805 and 0.592 <!-- blade-k1-layer-split -->. The slowdown as a function of the spilled share is not yet in our verified results, so the NVIDIA panel of FH1 is
pending.

**Figure specs.**

- **FH1 (headline).** Three panels, shared x axis = requested or served window as a fraction of the platform's
  boundary (AMD: the 65536 MiB dedicated block, stated in the caption; Intel: the shared-heap limit; NVIDIA: VRAM),
  y axis = cost of crossing it (decode and TTFT ratio; for Intel, start or refuse). Clamped points are drawn but
  marked; each point is labelled with its launch and its overflow path. Measurement note for the caption: each AMD
  point is one call per fresh server start without a warm-up call (5 or 3 calls per point), which departs from this
  paper's 1 warm-up + N convention. If space is short, FH1 replaces F2.
- **F2.** Intel flip point: measured (last start, first refusal] per model against accountings A (file size), B
  (device only) and C (shared heap), all five models <!-- A-24-flip-point-per-model --> <!-- A-24-flip-point-prediction -->.

### 4.5 Runtime policy

**Claim.** The window an agent actually gets is set by the runtime's policy for the device it detects, not by the
model or the memory, and when a prompt outgrows that window Ollama reports success and silently keeps half of
it; llama-server at a 4096 window instead fails loudly, with HTTP errors <!-- xruntime-llamacpp-wrong-rate -->.

**Default windows follow the detected device.** Ollama discovers the AMD iGPU through ROCm, with no opt-out, so the AMD APU runs on its iGPU while the
Intel mini PC, which has no ROCm path, falls through to the CPU <!-- x2-device-detect-mechanism -->. The default window
follows. On the Intel mini PC the runtime loaded 4096 tokens for each of three models, while on the AMD APU it loaded
262144 for qwen3-4b-2507, 131072 for llama3.1:8b and 40960 for qwen3:8b <!-- T2S-vs-X2-default-ctx -->, in each case
the model's native window (capped only for qwen3:8b <!-- K1v3-X2-table -->). The discrete GPU also got 4096 for both
models we could measure there <!-- blade-k1-layer-split -->. The runtime's own log reports the rule behind the two
small values: a default sized from detected VRAM, 4096 for the discrete GPU's 8.0 GiB and 32768 for the Intel iGPU's
36.2 GiB when that iGPU is enabled <!-- gpu-memory-readonly -->. Table 2 lists the values.

| platform (device used) | default window, three models | over-length prompt at the default |
|---|---|---|
| AMD APU (iGPU via ROCm) | 262144 / 131072 / 40960 <!-- T2S-vs-X2-default-ctx --> | success, half kept <!-- ollama-overflow-keeps-half --> |
| Intel mini PC (CPU) | 4096 / 4096 / 4096 <!-- T2S-vs-X2-default-ctx --> | success, half kept <!-- ollama-overflow-keeps-half --> |
| NVIDIA laptop (discrete GPU) | not measured / 4096 / 4096 <!-- blade-k1-layer-split --> | HTTP 200, evaluated 2050 <!-- blade-k1-layer-split --> |

Table 2: default context window per platform for qwen3-4b-2507 / llama3.1:8b / qwen3:8b, one runtime (Ollama).

**The default decides the outcome.** On the Intel mini PC we ran the same agent sessions twice, once as shipped on the
CPU and once with the iGPU enabled for the job's own server process. On the CPU both models loaded a 4096-token window,
crossed it at turn 3, and failed silently on Ollama 0.34.4 in 5/5 sessions each, every failure after the window was exceeded
<!-- t2s-r2-real-cpu -->. On the iGPU the same models loaded 32768, crossed it only at turn 23 (llama3.1:8b) and turns 20 and 21 (qwen3:14b), and no
failure came after the window: 0/5 for llama3.1:8b and, for qwen3:14b, 1/5 of baseline type
<!-- t2s-r2-real-igpu -->. The pre-registered kill criterion was not triggered on either backend
<!-- t2s-r2-real-cpu --> <!-- t2s-r2-real-igpu -->. This contrast changes backend and window together, so it shows
that the default decides whether a session fails, not that the window alone does; Section 4.6a isolates the window.

**On Ollama, overflow is silent.** When a prompt exceeds the loaded window, Ollama returns success and processes
num_ctx/2 + 2 tokens, in all 4 checked cases (4096->2050, 8192->4098, 32768->16386, 40960->20482), through its own
`--context-shift --keep 4` server flag <!-- ollama-overflow-keeps-half -->. The discrete GPU behaved the same (Table 2). No error, warning or response field tells the client on Ollama.

**Changing the window, and what persists.** On the AMD APU, a per-request `num_ctx` and `OLLAMA_CONTEXT_LENGTH`
both applied but reverted to the default of 131072 after a restart without them, while a Modelfile `PARAMETER
num_ctx` on a derived tag persisted <!-- window-override-methods -->. Only the derived tag survives a restart without
the client's help; the other two platforms have not run this check <!-- window-override-methods -->.

**Within the window, KV precision is the lever.** We pre-registered a comparison of eight context policies on the
Pareto frontier of rules, history, time and cost (Section 4.7 gives the router verdict). On the AMD APU every policy
kept every rule, and the frontier held one KV-cache policy per model <!-- b5-policy-metrics -->. For llama3.1:8b a
q4_0 KV cache took 6.6 min per session at history 100.0% (24/24), against 43.7 min at 95.8% (23/24) for the runtime
default; for qwen3:14b a q8_0 cache took 75.4 min against 214.8 min, at the same history recall of 62.5% (15/24)
<!-- b5-policy-metrics --> <!-- paper-percentage-counts -->. Our quantized-cache policies also turn on flash attention
<!-- design: harness/policy_compare.py -->, so we ran a control with
flash attention on and the cache at f16: 43.4 min for llama3.1:8b and 206.6 min for qwen3:14b, within tolerance of
the default, so the speedup is KV precision, not flash attention <!-- b5-fa-control -->. A rerun of the f16 recall
check for llama3.1:8b found 11/24 needles at f16 against
13/24 at q8_0 (Fisher p = 0.773) and 14/24 at q4_0 (p = 0.564), no difference shown, with needles within a call not
independent <!-- b5-kv-recall-f16-rerun -->. All of this is on the unconstrained AMD APU, where the default is the
native window. In the simulated-constraint version, with the window capped at 8192 and 4096 on the same platform,
the router (policy d) was on the frontier in all four capped cells, and it was the only policy that kept history:
80.0% and 62.5% at 8192 and 97.5% and 62.5% at 4096 for llama3.1:8b and qwen3:14b, against at most 12.5% for every
other policy <!-- b5-sim-constraint -->. The pre-registered verdict is still *refuted*, only because fixed setting (c),
client trimming, was also on the frontier in every capped cell <!-- b5-sim-constraint -->. The constrained platforms
have not run this comparison <!-- b5-verdict -->.

**What the KV setting loads, and whether output changes.** For qwen3:14b at its 40960 window, the server log reported the K
and V types each setting names, and the default setting's `/api/ps` size equals that of f16 with flash attention to
the byte, and the size differences give KV buffers
consistent with 16, 8.5 and 4.5 bits per element <!-- kv-precision-evidence -->. Under greedy decoding, over turns whose first-call prompt was identical on both
sides, f16 and f16 with flash attention matched exactly on 120/120 turns; f16 and q8_0 matched exactly on 103/105
comparable turns (98.1%), with tool calls agreeing on 104/105 (99.0%) and a first divergence at turn 30 and turn 35
in two seeds and none in the third <!-- kv-output-agreement -->. q4_0 changed the trajectory early: against f16 only
5 of 120 turns were comparable, with the first divergence at turn 1 or 2 in every seed, and 2/5 matched exactly
<!-- kv-output-agreement -->. On this workload q8_0 left the output nearly unchanged and q4_0 did not. A pre-registered
follow-up on llama-server fills the cell Ollama cannot run, a quantized KV cache with flash attention off; it has not
run, and we cite no result from it <!-- pending row: kv-fa-interaction-llamacpp, no number cited -->.

**Figure spec.**

- **F5.** Default window per platform and model over the CDF of agent step lengths in public traces
  <!-- T2S-vs-X2-default-ctx --> <!-- blade-k1-layer-split --> <!-- trace-fraction-exceeds-4096 -->.

### 4.6 Agent consequences

#### 4.6a The window result

**The controlled result.** On the AMD APU we ran three models at the Ollama default and at 32768, 16384, 8192 and
4096 <!-- R2-v1b-excess-over-default -->, five seeds per cell: 75 sessions and 3000 turns
<!-- R2-real-v1b-first-events -->, none surfacing an error before its first failure
<!-- R2-real-v1b-first-events -->. On Ollama, after-window silent failures rise as the window shrinks: 0/15 at 32768
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
<!-- R2-v1b-gated-per-model -->, and 5/5 at 4096, excess +1.000 <!-- R2-v1b-excess-over-default --> (one-sided Fisher exact, 5/5 vs 0/5 sessions,
p = 0.00397 <!-- session-exact-tests -->).
qwen3:14b has no after-window failure above 4096 <!-- R2-v1b-gated-per-model -->, and its 4096 sessions lapse on
different rules: the first gated failures are rule 5 or rule 3 at turns 5 to 30 <!-- R2-qwen3-14b-4096-reconciliation -->, while rule 2 held on 40/40
turns in every seed <!-- R2-qwen3-14b-4096-reconciliation -->; we name the failing rule per model rather than say rules broke.
qwen3:8b reversed the ordering at five seeds; a pre-registered extension to 10 seeds per tier
<!-- R2-qwen3-8b-reversal-10seed --> tested whether that was noise. Pooled, qwen3:8b failed silently in 6/10 sessions at 8192 and 2/10 at 4096
<!-- R2-qwen3-8b-reversal-10seed --> (Fisher exact two-sided p = 0.170 <!-- R2-qwen3-8b-reversal-10seed -->), which
the pre-registered rule reads as resolved as noise <!-- R2-qwen3-8b-reversal-10seed -->. The excess over default is
+0.600 (95% CI +0.300 to +0.900) at 8192 but +0.200 (95% CI +0.000 to +0.500) at 4096, where no excess is shown
<!-- R2-qwen3-8b-reversal-10seed -->. We report the cell as within noise and do
not claim that 4096 is safe for qwen3:8b.

**Protocol check: tools on every call, on two runtimes.** A pre-registered rerun offers tools on every call, on a
different task workload, with five seeds per tier and a `num_keep` tier (Section 4.7). It rests on qwen3:14b alone:
llama3.1:8b and qwen3:8b failed the task validation <!-- x2-evalset-val-v2-gates -->
<!-- x2-evalset-tasks-val-v2-qwen3-8b-gates -->, so their real runs refused to start, as designed. For qwen3:14b the
pre-registered reading was that the effect holds at 4096 <!-- x2-evalset-tasks-real-v2-gated-excess --> if the
excess CI lies above zero. Within this run it held: 5/5 <!-- x2-evalset-tasks-real-v2-gated-excess --> against 0/5
at the default <!-- x2-evalset-tasks-real-v2-gated-excess -->, excess +1.000
<!-- x2-evalset-tasks-real-v2-gated-excess -->, one-sided Fisher exact p = 0.00397 over sessions <!-- session-exact-tests -->. Wrong task end states were 11.3% (17/150)
<!-- x2-evalset-tasks-real-v2-wrong-rate --> at the default, 12.7% (19/150) <!-- x2-evalset-tasks-real-v2-wrong-rate --> at
16384, 35.3% (53/150) <!-- x2-evalset-tasks-real-v2-wrong-rate --> at 8192 and 42.0% (63/150) <!-- x2-evalset-tasks-real-v2-wrong-rate -->
at 4096, every wrong turn silent on Ollama 0.34.4 <!-- x2-evalset-tasks-real-v2-wrong-rate -->: excess +24.0 points
(95% CI +18.7 to +29.3) <!-- x2-evalset-tasks-real-v2-wrong-rate --> at 8192 and +30.7 points (95% CI +26.0 to
+34.7) at 4096, none shown at 16384 <!-- x2-evalset-tasks-real-v2-wrong-rate --> <!-- x2-evalset-tasks-real-v2-ci-units: these CIs are the session-level bootstrap (10000 resamples, sessions resampled within tier) -->
(session-level 95% CIs over 5 vs 5 sessions). A deviation was declared before the data: qwen3:14b
reached the task end state on 80/90 <!-- x2-evalset-val-v2-gates --> validation turns, below the bar, so absolute
wrong rates include baseline errors. On a stock llama-server (llama.cpp), five sessions per arm, the default loaded
40960. Wrong task turns were 8.0% (12/150) at the default
and at 40960, and 100.0% (150/150) at 4096, +92.0 points over the default (95% CI +88.7 to +94.7, session bootstrap)
<!-- xruntime-llamacpp-wrong-rate -->. At 4096 no wrong turn was silent: by the server's log, 60 were rejected as
over the window, 3 stopped at the context limit and 87 ended in another error, and 3.3% of task turns completed a
tool round, against 97.3% at the default <!-- xruntime-llamacpp-wrong-rate -->. Silence is counted by the harness's
definition, a wrong turn with no HTTP error on any call: 0 at 4096 and 12 at the default
<!-- xruntime-llamacpp-wrong-rate -->. With the window fixed at 40960, llama-server matches Ollama's default-window
rate, -3.3 points against Ollama's default tier (95% CI -6.7 to +0.0, 5 vs 5 sessions), with 0/150 context-event
turns <!-- xruntime-default-fixed -->. The window effect replicates across the two runtimes; the silence at 4096 is
Ollama's.

**Scope: scale and family.** The pre-registered H-scale hypothesis stated that both R2 findings, rule loss at 4096
and history loss with rules intact at 32768 <!-- H-scale-verdict -->, hold from 30B to 70B, with three seeds per
cell. qwen3:32b lost rules at 4096 in 2/3 <!-- H-scale-verdict --> sessions but showed history loss with rules intact
in only 1/3 <!-- H-scale-verdict --> at 32768 (mixed). llama3.3:70b lost no rules at 4096 (0/3
<!-- H-scale-verdict -->, every session over the window) and lost history with rules intact in 3/3
<!-- H-scale-verdict --> at 32768, the pre-registered alternative "resists rule loss" <!-- H-scale-verdict -->. Overall the verdict is mixed: H-scale is not supported
<!-- H-scale-verdict -->. mistral-small3.2:24b, from another family, also resisted rule loss (0/3 at 4096
<!-- H-scale-family-verdict -->, history loss 2/3 at 32768) <!-- H-scale-family-verdict -->. Within what we tested: Rule loss appears in the 8B and 14B models (5 seeds) and in qwen3:32b (2 of 3 sessions at 4096) <!-- H-scale-verdict -->, and not in llama3.3:70b or mistral-small3.2:24b (0 of 3 each) <!-- H-scale-verdict --> <!-- H-scale-family-verdict -->. History loss appears in all of them <!-- H-scale-verdict --> <!-- H-scale-family-verdict -->. We do not claim rule loss in general.
Every cross-model result in this subsection is R2 rule-following evidence, not B3 task accuracy. That includes the
10-seed R2 extensions: for llama3.1:8b, 10/10 sessions failed silently after the window at 4096 against 0/10 at the
default (excess +1.000, one-sided Fisher exact p = 5.41e-06 <!-- session-exact-tests -->), with an excess of +0.500 (95% CI +0.200 to +0.800) at 8192 and none shown at 16384
<!-- R2-v1b-excess-10seed -->, and the qwen3:8b cell above <!-- R2-qwen3-8b-reversal-10seed -->. B3 task accuracy has
one valid model, qwen3:14b (Section 4.8).

**Long tasks, inside the window.** On 20 synthetic long-horizon tasks whose trace lengths are matched to public agent
traces (10 to SWE-Gym step quantiles and 10 to OpenHands traces on nebius-rebench; the tasks are ours, not from those
suites), qwen3:14b on the AMD APU, three seeds each, completed 5/60 tasks (8.3%, Wilson 95% 3.6% to 18.1%) at the
Ollama default (loaded window 40960) and 4/60 (6.7%, 2.6% to 15.9%) with a q8_0 KV cache; every wrong task was
silent <!-- long-task-local-accuracy -->. On long-horizon tasks, qwen3:14b and qwen3:32b solved none of 30, and
llama3.3:70b 3 of 30 after a pre-registered tool-template fix <!-- long-task-scale-local --> <!-- longtask-70b-toolfix -->;
at a window that held the whole prompt, llama3.3:70b still failed 3 of 3 token-matched tasks on wrong actions, so on
these tasks capability, not memory, fails first <!-- longtask-tokens-v1 -->.

**Table R2-gated and figure spec.**

Table R2-gated: silent / after window / baseline-type gated failures, of 5 sessions per cell; intact = sessions with no
gated failure and no canary miss at turns 5/10/15/20/25/30/35/40 <!-- R2-v1b-gated-per-model --> <!-- R2-real-v1b-survival -->.

| tier | llama3.1:8b | qwen3:14b | qwen3:8b |
|---|---|---|---|
| Ollama default | 3 / 0 / 3; intact 4/4/4/2/2/2/1/1 | 0 / 0 / 0; intact 5/5/5/5/5/0/0/0 | 0 / 0 / 0; intact 5/5/5/4/4/0/0/0 |
| 32768 | 1 / 0 / 1; intact 5/5/5/3/3/0/0/0 | 0 / 0 / 0; intact 5/5/5/5/0/0/0/0 | 0 / 0 / 0; intact 5/5/5/4/0/0/0/0 |
| 16384 | 2 / 2 / 0; intact 5/5/0/0/0/0/0/0 | 0 / 0 / 0; intact 5/5/0/0/0/0/0/0 | 0 / 0 / 0; intact 5/5/0/0/0/0/0/0 |
| 8192 | 4 / 4 / 0; intact 5/0/0/0/0/0/0/0 | 0 / 0 / 0; intact 5/0/0/0/0/0/0/0 | 4 / 4 / 0; intact 3/0/0/0/0/0/0/0 |
| 4096 | 5 / 5 / 0; intact 0/0/0/0/0/0/0/0 | 4 / 4 / 0; intact 0/0/0/0/0/0/0/0 | 1 / 1 / 0; intact 0/0/0/0/0/0/0/0 |

Table note: the qwen3:14b 4096 cell counts lapses of rule 3 and rule 5, not rule 2
<!-- R2-qwen3-14b-4096-reconciliation -->. The qwen3:8b 8192 and 4096 cells are five seeds here; the text quotes the
10-seed pooled cell <!-- R2-qwen3-8b-reversal-10seed -->.

- **F6.** Per-tier session survival curve (intact sessions against turn) <!-- R2-real-v1b-survival -->, with the
  after-window excess per tier and its 95% CI as a second panel <!-- R2-v1b-excess-over-default -->.

#### 4.6b The mechanism on AMD and NVIDIA

**What the runtime removes.** We reran llama3.1:8b on the AMD APU, one session per tier (80 calls each <!-- R2-mechanism-lowlevel -->), with Ollama in debug
mode, its log sliced per call, and a render-only copy of each request. The rendered prompt matched the runtime's
count on every check (9/9 within 1% at 4096 <!-- R2-mechanism-verdict -->; render/log disagreements 0 <!-- R2-mechanism-verdict -->). Three
mechanisms appear. First, Ollama drops whole messages from the front once the conversation outgrows the window, on
54/80 <!-- R2-mechanism-lowlevel --> calls at 16384, 68/80 <!-- R2-mechanism-lowlevel --> at 8192 and 76/80 at 4096 <!-- R2-mechanism-lowlevel -->, always keeping the system
message ("system kept on all True" <!-- R2-mechanism-verdict -->) and usually cutting mid-turn. This loses history
but keeps the rules in view. At 4096 <!-- R2-mechanism-lowlevel --> two lower-level steps then remove the system prompt. llama.cpp's context shift
fired on 34 <!-- R2-mechanism-lowlevel --> calls from turn 4 <!-- R2-mechanism-lowlevel -->, each keeping n_keep 5 <!-- R2-mechanism-lowlevel --> tokens and discarding the next 2045 <!-- R2-mechanism-lowlevel -->, 29 of them on the
final-answer call <!-- R2-mechanism-lowlevel -->; and once Ollama's own token cut reduced a prompt of 4097 <!-- R2-mechanism-lowlevel --> tokens to
2050 <!-- R2-mechanism-lowlevel -->, keeping 4 <!-- R2-mechanism-lowlevel -->. That is the runtime's overflow rule, which returns success and keeps
num_ctx/2 + 2 tokens (4096->2050 <!-- ollama-overflow-keeps-half -->). No step raised an error (exceed-context errors 0
<!-- R2-mechanism-verdict -->). The working hypothesis, "drop old turns, keep the system message", is therefore
refuted by the 4096 token cut (verdict refuted <!-- R2-mechanism-verdict -->) and confirmed tier by tier at 32768,
16384 and 8192.

**NVIDIA replicates the first two steps.** One session at 4096 on the NVIDIA laptop passed the same render check (9/9 within 1% <!-- blade-mechanism-4096 -->).
Ollama trimmed whole messages on 75/80 <!-- blade-mechanism-4096 --> calls from turn 3 <!-- blade-mechanism-4096 --> and kept the system message on every one, and the context
shift fired 33 <!-- blade-mechanism-4096 --> times; no token cut occurred and no error surfaced <!-- blade-mechanism-4096 -->. That row lacks
n_keep, so the claim that the shift discards the system prompt is AMD-only; whether the token cut occurs on NVIDIA
is open.

**A second silent difference on Ollama: tool calls as text.** In the NVIDIA validation, llama3.1:8b's negative control at
32768, where nothing is truncated, missed canaries in two of five sessions (seeds 20260903 and 20260905, 2 misses
each <!-- blade-r2-gate-v2 -->): the final answer was a tool call written as plain text. Three pre-registered checks on the AMD APU kept model files, seeds and settings
and moved one factor toward the NVIDIA configuration each. At 32768 on AMD's ROCm backend there were 0 of 20 <!-- x2-r2-llama-val-32k --> misses,
so the window does not reproduce it ("backend or offload effect" <!-- x2-r2-llama-val-32k -->). Forcing NVIDIA's CPU
offload split on AMD, 22 of 33 <!-- x2-r2-llama-val-32k-offload --> layers on the GPU with VRAM share
0.656 <!-- x2-r2-llama-val-32k-offload -->, again gave 0 of 20 ("backend effect" <!-- x2-r2-llama-val-32k-offload -->). Switching only the backend, to
Vulkan on the same AMD GPU and verified in 5/5 <!-- x2-r2-llama-val-32k-vulkan --> sessions, gave 2 of 20 <!-- x2-r2-llama-val-32k-vulkan --> misses, at seed 20260903, turn 5 <!-- x2-r2-llama-val-32k-vulkan -->, as a text call
to `lookup_fact`: "backend implementation, not the vendor" <!-- x2-r2-llama-val-32k-vulkan -->. The failure follows
the llama.cpp backend implementation (CUDA and Vulkan, not ROCm, in these seeds), not the GPU vendor, but on Vulkan
that rests on one session. Five seeds cannot
show a zero is impossible, and the kernel is not identified; identical model files and settings still yield a
different agent on a different backend, with no error from Ollama.

**Figure spec.**

- **F7.** Per-turn event raster per tier (message drops, context shifts, token cut), one row per tier, with the NVIDIA
  4096 session as an extra row next to the AMD 4096 row <!-- R2-mechanism-lowlevel --> <!-- blade-mechanism-4096 -->.

### 4.7 Fixes

**Client-side trimming splits by model, on two vendors.** We pre-registered a client-side fix: before every call the client renders and counts the request with the model's tokenizer and drops whole old turns until the
system prompt, the kept turns and the answer budget fit the window with a 5% <!-- design: harness/x2_r2_client_trim.py -->
margin, always keeping the system prompt. A model *recovers* when every rule in use is at or above 90%
<!-- design: analysis/r2_mitigation_report.py -->, and the hypothesis needed both models to recover at 4096
<!-- R2-mitigation-v1-verdict -->. Mechanically it worked: 0 context shifts, 0 token cuts, 0 Ollama message drops and 0
exceed-context errors in every cell <!-- R2-mitigation-v1-observed -->. The verdict is *refuted*
<!-- R2-mitigation-v1-verdict -->. On the AMD APU llama3.1:8b recovered at 8192 and 4096
<!-- R2-mitigation-v1-verdict -->; qwen3:14b recovered at 8192 but not at 4096 <!-- R2-mitigation-v1-verdict -->, where
its `log_event` rule fell to 61.7% (74/120) <!-- R2-mitigation-v1-4096 --> <!-- paper-percentage-counts -->, failing first at turn 3 in every seed (3/3/3
<!-- R2-mitigation-v1-4096 -->). On the Intel mini PC's CPU at 4096 <!-- t2s-r2-mitigation-cpu --> the split
replicates: llama3.1:8b recovered, qwen3:14b did not, again on `log_event`, overall refuted
<!-- t2s-r2-mitigation-cpu -->. The cause is the fix itself: untrimmed, qwen3:14b kept rule 2 on 40/40 turns in every
seed <!-- R2-qwen3-14b-4096-reconciliation -->. On the first failing trimmed turn its first call
requested only `lookup_fact`, where the untrimmed run called `log_event` and then `lookup_fact` <!-- R2-qwen3-14b-4096-reconciliation -->: at 4096 the
budget leaves room for only 1-2 <!-- R2-mitigation-v1-observed --> turns besides the system prompt, so trimming
removes the in-context examples of the two-call pattern the model imitates.

**num_keep fixes one model's rules, on the token-cut path.** We
pre-registered a test for llama3.1:8b at 4096, the path where both the context shift and the token cut fired, with
`num_keep` set to the system prompt's measured token length (601, 594 and 597 per seed <!-- num-keep-guardrail-test -->). All 105
<!-- num-keep-guardrail-test --> context shifts kept n_keep 595, 598 or 602 <!-- num-keep-guardrail-test -->. The rules in use held (rule 1 at 94.2% (113/120) <!-- num-keep-guardrail-test --> <!-- paper-percentage-counts -->, rules 3 and
4 at 100.0% (120/120 and 60/60) <!-- num-keep-guardrail-test --> <!-- paper-percentage-counts -->), which the pre-registered rule reads as "one-line fix"
<!-- num-keep-guardrail-test -->. It fixes the rules and nothing else: Ollama still dropped messages on 224 of 239
<!-- num-keep-guardrail-test --> calls and the history canary was missed on 24 of 24 <!-- num-keep-guardrail-test -->
checks. Nor does it remove the window effect in the protocol check: within that run the
`num_keep` tier at 4096 lowered qwen3:14b's wrong-task rate from 42.0% (63/150) <!-- x2-evalset-tasks-real-v2-wrong-rate --> to
30.0% (45/150) <!-- x2-evalset-tasks-real-v2-wrong-rate -->, but its excess over the default stayed at +18.7 points (95% CI
+16.0 to +21.3) <!-- x2-evalset-tasks-real-v2-wrong-rate --> <!-- x2-evalset-tasks-real-v2-ci-units: these CIs are the session-level bootstrap (10000 resamples, sessions resampled within tier) --> and its gated excess at +1.000
<!-- x2-evalset-tasks-real-v2-gated-excess --> (5/5 vs 0/5 sessions, p = 0.00397 <!-- session-exact-tests -->); the pre-registered reading is that `num_keep` does not remove it
<!-- x2-evalset-tasks-real-v2-gated-excess -->. What lowered the task-workload rate is not established: in the R2 harness the
same flag changed nothing for qwen3:14b at 4096 <!-- R2-qwen3-14b-4096-reconciliation -->.

**A router does not dominate.** We pre-registered a comparison of eight
<!-- design: harness/policy_compare.py --> context policies ((a) runtime default, (b) largest window that starts, (c)
client trimming, (d) a router escalating to a cloud model, (e) an auto-sized window, (f) `num_keep`, (g) q8_0 and (h)
q4_0 KV caches), with the hypothesis that the router is on the Pareto frontier of rules, history, time and cost on
every platform. It is *refuted*: on the AMD APU the router was dominated for both models <!-- b5-verdict -->. Every policy
kept every rule (rule 100.0% in every cell, 300/300 for llama3.1:8b and 540/540 for qwen3:14b <!-- b5-policy-metrics -->
<!-- paper-percentage-counts -->). The router took 43.5 min per session for llama3.1:8b at history 95.8% (23/24), and
135.1 min and USD 0.8419 per session for qwen3:14b, against faster KV-cache policies at the same or higher history
recall (Section 4.5) <!-- b5-policy-metrics --> <!-- paper-percentage-counts -->. Under the simulated cap of Section 4.5
the router joined the frontier in all four capped cells, and the verdict stays refuted only because client trimming
did too <!-- b5-sim-constraint -->; the constrained platforms are pending.

**Recovering after a silent failure on Ollama is costly and incomplete.** A cloud model given the whole client-side
history late, on every wrong task of the protocol-check task run (Section 4.6a), fixed 84.8% (167/197) of
them, task-level 95% Wilson CI [79.1%, 89.1%] <!-- cloud-redo-full-context -->, which treats the 197 tasks as
independent; resampling whole sessions instead (25 sessions, stratified by tier, 10000 resamples) gives 95% CI
[79.4%, 90.4%] <!-- headline-ci-session-rescue -->, at a mean of USD 0.0227 per task
<!-- cloud-redo-full-context -->. Matched on the same task and seed, the late redo cost 1.76x a fresh run of
that task from the start (mean USD 0.0226 against USD 0.0129 over 119 matched rows) <!-- late-escalation-cost-vs-fresh -->.
A plain redo also cannot undo what the local model already did: of the 30 tasks still wrong, 28 needed a repair of
state the local model's earlier actions had changed, and 2 were cloud-model errors <!-- cloud-redo-full-context -->. Deciding at session start instead, with the rule "run locally if local accuracy at the recommended setting is
within a tolerance of the cloud model's", no measured device qualifies at matched quality: the AMD APU at its best
setting is 7.8 points below the small cloud model and 11.1 below the mid-tier one, and the Intel iGPU, measured by
session pass rate rather than task accuracy, is 16.7 and 20.0 points below <!-- hybrid-route-at-start -->. At
matched quality the measured answer is therefore the cloud from the start; step-level escalation stays a narrow
middle case, since window overflow does not pick out the wrong steps (Section 4.7p).

**Rolling back executed actions before the redo (stand-in).** On simulated tools, cascaded rollback fixed 28/28
(195/197 overall) <!-- own-rollback-redo --> <!-- cloud-rescue-combined -->. In a cascaded rollback, applied to the 28 tasks that needed a repair
of earlier state, the environment is snapshotted and
every earlier wrong task turn of the session is rolled back and replaced by its reference calls, and the cloud model
then redoes the task: it fixed 28/28 (95% Wilson CI [87.9%, 100.0%]) against a pre-registered target of 25/28, at
USD 0.8926 billed for the 28 <!-- own-rollback-redo -->. Combined with the full-history redo, 195/197 wrong local
tasks were fixed (99.0%, 95% Wilson CI [96.4%, 99.7%]); the 2 still wrong were the cloud model's own errors
<!-- cloud-rescue-combined -->. Replacing earlier turns by their reference calls is a stand-in for a successful
escalation of each earlier turn, so these 28 are fixed under that assumption, not in a live hybrid deployment
<!-- own-rollback-redo --> <!-- cloud-rescue-combined -->.

**No reliable detector.** Every pre-registered attempt to find or avoid the wrong tasks missed its target. On its
held-out seed at 4096 the tuned post-run check caught 14/14 wrong tasks but escalated 15/16 correct ones, against a
target of at most 10%; its rule-level checks alone caught 6/14 <!-- post-run-check-tuned -->
<!-- post-run-check-l3-rule-breakdown -->. A session-start fit predictor reached precision 0.350 against the runtime
log (MISSED) <!-- fit-predictor-v1 -->, and the full route, check and redo pipeline kept 81/90 tasks local at 94.4%
accuracy, below its 97% target (MISSED) <!-- hybrid-full-pipeline-v1 -->. Our own compaction, which summarizes old
turns, left 16.0% and 23.3% of tasks wrong at 8192 and 4096, against targets of 15% and 20% <!-- own-compaction-8192 -->
<!-- own-compaction-4096 -->, and no route-at-start policy met its target <!-- hybrid-route-at-start-v2 -->. On the
Intel mini PC the strict gate refused qwen3:14b (task end state 80/90 on the CPU, 79/90 on the iGPU)
<!-- t2s-evalset-tasks-cpu --> <!-- t2s-evalset-tasks-igpu -->, so no Intel task accuracy exists.

#### 4.7p The runtime's shift rule and the window flag, characterized

**The runtime's context-shift trigger, characterized.** The step that discards the system prompt at the smallest window
is llama.cpp's context shift (Section 4.6b), and its trigger can be read from the llama.cpp source rather than fitted:
a shift happens on a call when the prompt left after the message-level trim, plus the tokens generated, exceeds the
loaded window. We scored that rule on the AMD mechanism sessions, one per tier, against the shifts the server log
recorded. At 4096 it gave TP 34 FP 0 FN 0 TN 46, and at 32768 TP 10 FP 0 FN 0 TN 70, precision and recall 100.0% in
both (34/34 at 4096, 10/10 at 32768), with misses 0 over all tiers <!-- R2-shift-predictor-confusion -->. The generated count is not known before the
call, so a client substitutes the generation budget: P+num_predict>n_ctx gives TP 44 FP 22 FN 0
<!-- R2-shift-predictor-confusion -->: this characterizes the runtime's rule as it can be evaluated before the call,
missing no shift at the cost of false alarms; it is not a flag we contribute. It has no
tunable parameter, but it was built and scored on the same sessions; no held-out observed sessions exist
yet.

**It predicts the shift, not the failure.** Applied to the real run as a prediction only, where no server log exists,
the rule flags 102/240 calls for llama3.1:8b at 4096 and 0/240 for qwen3:14b <!-- R2-shift-predicted-real-v1 -->, yet
qwen3:14b also loses rules at 4096. For that model the loss runs through what the message-level trim removes, not
through the shift <!-- R2-qwen3-14b-4096-reconciliation -->. What the trim leaves at the front of the window is usually a fragment: at
4096, 80.3% (61/76) of truncated calls began with an orphaned message <!-- R2-orphan-rates -->.

**Window overflow is not a general wrong-answer detector.** We measured the flag "this task turn's prompt exceeded
the loaded window" as a predictor of the outcome "the task ended wrong" on the protocol-check task run of Section 4.6a
(qwen3:14b, AMD APU) <!-- impact-v3-detection-value-measured -->. At the runtime default no turn overflowed, and the flag
missed 17/17 wrong tasks (recall 0.0% (0/17)). At 16384 it reached precision 15.4% (4/26) and recall 21.1% (4/19); at
8192 precision 45.3% (48/106) and recall 90.6% (48/53); at 4096 recall 100.0% (63/63) but precision 42.9% (63/147),
flagging 147 of 150 turns (TP 63 FP 84)
<!-- impact-v3-detection-value-measured -->. Where the window is large, wrong answers have other causes that the flag
cannot see; where it is small, nearly every step overflows and the flag no longer separates good steps from bad. The
shift rule and the window flag explain system-prompt loss; they are not contributed detectors or a cheap escalation
filter.

### 4.8 Threats to validity

**Platforms and versions.** We measure one machine per vendor, so a platform effect cannot be separated from that
machine's driver and memory configuration. Agent runs use Ollama 0.34.4 <!-- R2-qwen3-14b-4096-reconciliation --> and
direct runs llama.cpp b10970 <!-- design: docs/HARDWARE.md (version record; no register row, gap G13) -->; default
windows and trimming are runtime code and can change in any release. The NVIDIA laptop contributes validation and
mechanism sessions only, because its validation gate held back the real agent runs <!-- blade-r2-gate-v2 -->, and the
router and context-policy comparison ran on the AMD APU alone <!-- b5-verdict -->, whose default window is the model's
native one.

**Statistics and provenance.** Cells have three <!-- H-scale-verdict --> to five <!-- R2-v1b-gated-per-model --> seeds, ten for
qwen3:8b <!-- R2-qwen3-8b-reversal-10seed -->. A 0/5 or 5/5 cell has a point bootstrap interval, so such cells carry a
session-level Fisher exact test instead <!-- session-exact-tests -->, and the scale and family results rest on three sessions per cell. Sampling
uses temperature 0 <!-- design: harness/x2_r2_agent.py --> and a fixed request seed, so seeds vary the session
script, not the sampling; nonzero temperatures are untested. Where a script times or scores repeated calls it follows
one warm-up call plus N measured calls (N = 3 for the KV-precision recall check <!-- b5-kv-recall-f16-rerun -->, whose
needles within a call are not independent, so that check is descriptive); session-time comparisons are over three
seeds <!-- cost-cut-config-time -->. Runs before 2026-10-14 rest on byte-identical reruns for their harness identity
(Section 4.3).

**Workload.** The R2 rules are synthetic and checkable; deployed system prompts differ. Guardrail rules 6 to 8
(privacy, refusal, confirmation) already fail at full context for both models tested, for example rule 6 at 0.417 and
0.250 of bait turns <!-- x2-evalset-val-v1-gates -->, so they are excluded from every window measurement and support no
claim here. The tools-on-every-call task workload rests on qwen3:14b alone, under the deviation of Section 4.6a;
qwen3:8b, llama3.1:8b, mistral-small3.2:24b and llama3.3:70b failed its validation
<!-- x2-evalset-tasks-val-v2-qwen3-8b-gates --> <!-- x2-evalset-tasks-real-v2-wrong-rate -->
<!-- x2-evalset-tasks-val-v2-mistral-small32-gates --> <!-- x2-evalset-tasks-val-v2-llama33-70b-gates -->. B3 task
accuracy therefore has one valid model; the cross-model results are R2 rule-following evidence, not B3 task accuracy.

**Cost figures.** Cloud cost per real agent task is priced from traces, not executed <!-- cloud-cost-per-real-task -->,
and its caching case is an assumption. Our first cloud redo of wrong local tasks was a broken protocol: 109
<!-- cloud-redo-diagnosis --> of its 118 <!-- cloud-redo-diagnosis --> failures were turns where the model's default
thinking spent the whole 384-token <!-- cloud-redo-diagnosis --> output budget and returned an empty answer, so its fix
rate measured the token budget, not the cloud model. We cite no fix rate or cost from it; the corrected full-history redo (thinking limited to between tool calls,
2048 output tokens per call against the local run's 384) is the one reported in Section 4.7 <!-- cloud-redo-full-context -->.

### 4.9 Conclusion

On Ollama, local LLM agents lose context without telling anyone. On every platform we measured, Ollama trims, shifts
or cuts a conversation that outgrows the window, and the request still succeeds. A stock llama-server at a 4096 window
instead fails loudly, with an HTTP error on every wrong task turn, while the 4096 effect on wrong tasks replicates
there (+92.0 points, 95% CI +88.7 to +94.7) <!-- xruntime-llamacpp-wrong-rate -->. Rule loss appears in the 8B and 14B models (5 seeds) and in qwen3:32b (2 of 3 sessions at 4096) <!-- H-scale-verdict -->, and not in llama3.3:70b or mistral-small3.2:24b (0 of 3 each) <!-- H-scale-verdict --> <!-- H-scale-family-verdict -->. History loss appears in all of them <!-- H-scale-verdict --> <!-- H-scale-family-verdict -->. These cross-model results are R2 rule-following evidence, not B3 task accuracy, which rests on qwen3:14b alone. The window is often not the user's
choice: the runtime picks it from the detected device, and an unmodified install on an Intel CPU ran agents at 4096
tokens <!-- t2s-r2-real-cpu -->. That is the window at which, on the AMD APU, qwen3:14b ended 42.0% (63/150) of tasks in a wrong
state against 11.3% (17/150) at its default, all silently on Ollama 0.34.4 <!-- x2-evalset-tasks-real-v2-wrong-rate -->.

For practitioners, five lessons follow. Set the window explicitly for each device and verify the loaded value
<!-- window-override-methods -->. Surface the runtime's own cut-off log lines [Ollama issue #14259; to verify], but do
not treat them as a correctness check. Try a quantized KV cache before a smaller window: on the AMD APU it cut session
time by 85% (37.1 of 43.7 min) and 65% (139.4 of 214.8 min, 3 sessions per setting) with no measured loss of rules or recall <!-- cost-cut-config-time --> <!-- b5-kv-recall-f16-rerun -->.
Treat trimming as a change to the prompt, since dropping old turns also drops the examples the model imitates
<!-- R2-qwen3-14b-4096-reconciliation -->. Validate an agent on the backend it will run on, because identical model files
and settings behaved differently on CUDA, Vulkan and ROCm <!-- blade-r2-gate-v2 --> <!-- x2-r2-llama-val-32k-vulkan -->
<!-- x2-r2-llama-val-32k-offload -->.

Future work follows from what is pending. The router and context-policy comparison must run on the Intel mini PC and the NVIDIA
laptop, whose default windows are small enough that loss occurs <!-- b5-verdict -->. A hybrid deployment needs a
general verifier for wrong answers produced inside the window: a late cloud redo with the full history fixed 84.8%
(167/197) of wrong tasks <!-- cloud-redo-full-context --> at 1.76x the cost of a fresh cloud run of the same task <!-- late-escalation-cost-vs-fresh -->,
and a plain redo could not undo executed actions; on simulated tools, cascaded rollback fixed 28/28 (195/197 overall)
<!-- own-rollback-redo --> <!-- cloud-rescue-combined -->, a stand-in in which earlier wrong turns are replaced by
reference calls, so a live rollback remains future work. More models, seeds and runtime versions would turn the
model-dependence of rule loss from an observation into a predictable property.

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
- [Ding et al. 2024, Hybrid LLM] D. Ding et al., "Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing", ICLR 2024. To verify (added 2026-10-22).
- [Chen et al. 2023, FrugalGPT] L. Chen, M. Zaharia, J. Zou, "FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance", arXiv 2023 (TMLR 2024?). To verify (added 2026-10-22).
- [Weng et al. 2023, Self-Verification] Y. Weng et al., "Large Language Models are Better Reasoners with Self-Verification", Findings of EMNLP 2023. To verify (added 2026-10-22).
- [Manakul et al. 2023, SelfCheckGPT] P. Manakul, A. Liusie, M. J. F. Gales, "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models", EMNLP 2023. To verify (added 2026-10-22).
- [Shi et al. 2026, EdgeWise] Z. Shi et al., "EdgeWise: Bringing Long-Horizon Agents to the Edge via Selective Cloud Delegation", arXiv:2610.17560, 2026. To verify (added 2026-10-23; arXiv page checked 2026-10-23).
- [Huang et al. 2026, EC2-Bench] P. Huang et al., "Empowering Edge Agents: A Systematic Analysis of Edge-Cloud Collaboration in LLM Agent Execution", arXiv:2610.17816, 2026. To verify (added 2026-10-23; arXiv page checked 2026-10-23).
- [Narayan et al. 2025, Minions] A. Narayan et al., "Minions: Cost-efficient Collaboration Between On-device and Cloud Language Models", arXiv:2502.15964, 2025. To verify (added 2026-10-23; arXiv page checked 2026-10-23).
- [Tran et al. 2025, Arch-Router] C. Tran et al., "Arch-Router: Aligning LLM Routing with Human Preferences", arXiv:2506.16655, 2025. To verify (added 2026-10-23; arXiv page checked 2026-10-23).
- [llmfit] llmfit, open-source model-fit tool (GitHub repository README checked 2026-10-23). To verify (added 2026-10-23).
- [Ollama issue #14259] "Chat history and embedding truncation happens silently with no user-visible indication", Ollama issue tracker, opened 2026-02-14. To verify (added 2026-10-23; issue page checked 2026-10-23).
- [Kwon et al. 2023, vLLM] W. Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023. To verify.
- [Laskaridis et al. 2024, MELTing Point] S. Laskaridis et al., "MELTing Point: Mobile Evaluation of Language Transformers", MobiCom 2024. To verify.
- [Song et al. 2024, PowerInfer] Y. Song et al., "PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU", SOSP 2024. To verify.
- [Alizadeh et al. 2024, LLM in a Flash] K. Alizadeh et al., "LLM in a Flash: Efficient Large Language Model Inference with Limited Memory", ACL 2024. To verify.
