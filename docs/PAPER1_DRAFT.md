# Paper 1 Draft (ISPASS 2027 target, anonymized)

Status (2026-10-22): advisor version. This file is the paper only: title, abstract, Sections 4.1 to 4.9 (paper
Sections 1 to 9), figure and table specs, and references. Internal tracking material (status history, claim table,
open findings, "Obvious?" defense, figure-source list, gaps, claim-to-evidence index, superseded outlines) is in
`docs/paper1/INTERNAL_TRACKING.md`; author and label mapping in `docs/PAPER1_COVER_NOTE.md`. The typeset version is
`docs/paper1/paper1.tex`. Every measured number carries its register id in an HTML comment (a LaTeX comment in the
typeset version); design constants carry "design: <source>".

---

## 1. Working title and abstract

**Working title:** *Silent by Default: How Runtime Policy and Memory Architecture Break Local LLM Agents on AI PCs*

Alternatives:

1. *Where Local Agents Break: Memory Boundaries, Runtime Defaults and Silent Context Loss on Ollama Across Three AI PC Platforms*
2. *The Runtime Picks Your Context Window: A Cross-Vendor Characterization of Local LLM Inference on AI PCs*
3. *No Error, No Rules: Characterizing Silent Failure of Ollama-Served LLM Agents Across Unified and Discrete Memory*

**Abstract.**

Local LLM agents on AI PCs rely on a runtime to choose the context window and manage memory. We measure what happens
when either runs out, with pre-registered experiments on an AMD APU, an NVIDIA laptop GPU and two Intel machines.
First, the failure is silent. When a prompt outgrows the window, Ollama keeps about half and reports success
<!-- ollama-overflow-keeps-half -->. On tool-using tasks, qwen3:14b ended 42.0% (63/150) of tasks wrong at a 4096-token
window against 11.3% (17/150) at its default <!-- x2-evalset-tasks-real-v2-wrong-rate -->, and qwen3:8b 80.4% (193/240) of turns at its
4096-token default against 25.0% (60/240) at 24576 <!-- pkg1109-t4-8b -->, every wrong answer at 4096 silent. Second, the
device and runtime set the window: Ollama loaded 4096 tokens for llama3.1:8b on one machine and 131072 on another
<!-- T2S-vs-X2-default-ctx -->, and serving stacks answer an over-length prompt with silent truncation or an HTTP error
<!-- pkg1109-contract1 --> <!-- xruntime-llamacpp-wrong-rate -->. Third, the memory figures runtimes report miss the real limit. On the AMD APU,
spill began past the 65536 MiB dedicated block, inside the advertised 76218 MiB heap <!-- fh1-amd-boundary-llama-3.3-70b -->; on an
Intel laptop, lowering the GPU memory share did not move the paging onset <!-- pkg1109-w3-50 --> <!-- pkg1109-x1 -->, and free memory
after load overstated usable memory by a median 1,238 MiB (99 probes, exploratory) <!-- pkg1109-os-reserve -->. No spill or
paging point raised an error <!-- fh1-amd-boundary-llama-3.3-70b --> <!-- pkg1109-x1 -->.

---

### 4.1 Introduction

<!-- Convention as in 4.3: measured numbers carry their register id; design constants carry "design: <source file>". -->

Runtimes such as Ollama and llama.cpp serve open-weight models of 8B to 70B parameters on laptops and small APU
workstations, and local agents drive them for tens of turns, appending every tool result to a conversation that grows
inside a fixed context window. When it outgrows the window, the runtime decides what to remove. On Ollama this is
silent: the request returns HTTP 200 <!-- blade-k1-layer-split --> with a well-formed answer and no field reporting
the loss; llama-server at the same window fails loudly, with HTTP errors <!-- xruntime-llamacpp-wrong-rate -->
(Figure FW).

- **FW (headline).** Wrong task turns by loaded window, split into silent (no HTTP error) and loud (HTTP error), with
  wrong / task turns on each bar: Ollama and llama-server serving qwen3:14b on the AMD APU (first task workload, 5
  seeds per tier) and Ollama serving qwen3:8b on the Intel mini PC (second task workload, 8 seeds per arm; silence is
  recorded only for its default arm) <!-- x2-evalset-tasks-real-v2-wrong-rate --> <!-- xruntime-llamacpp-wrong-rate --> <!-- pkg1109-t4-8b -->.

**Why run locally, and what it costs.** Priced on public agent traces (not executed; token counts recalibrated to the
cloud tokenizer), a median long-horizon coding-agent task costs USD 2.5307 on a mid-tier cloud model and USD 0.1354
on a small one with realistic caching <!-- cloud-cost-per-real-task -->. Our own tool-using tasks are short: in the
cloud the small and mid-tier models were correct on 87/90 and 90/90 <!-- all-cloud-b3-quality --> at a mean USD
0.00067 and USD 0.01020 per task <!-- all-cloud-b3-cost -->. Run locally on qwen3:14b at its default window, the bill
disappears, but 10 of 90 tasks were wrong and nothing marked them <!-- hybrid-savings-v1 -->; a perfect detector would
reach 90/90 by escalating those 10 <!-- hybrid-savings-v1 -->, but no such detector exists (Section 4.7).

The window is often not the user's choice: on an Intel mini PC, Ollama's device detection drops the integrated GPU
<!-- x2-device-detect-mechanism -->, so an unmodified install served two models with a 4096 <!-- t2s-r2-real-cpu -->
token window and every agent session failed silently (Section 4.5).

We ask what a local agent loses when its conversation outgrows the window or its memory, whether anything tells the
user, and what decides how often it happens, with pre-registered experiments on an AMD APU, an NVIDIA laptop GPU and
two Intel machines (a mini PC and a laptop), spanning unified and discrete memory. We make four contributions.

1. **Where memory runs out, and how** (Section 4.4). The AMD APU spills past its dedicated block, not its advertised
   heap, at almost no cost <!-- fh1-amd-boundary-llama-3.3-70b -->; on the Intel mini PC's iGPU a first-principles
   accounting predicts the start-or-refuse flip point to within -257 to -26 tokens <!-- A-24-flip-point-prediction -->,
   while on an Intel laptop the same kind of accounting misses the silent paging onset (0 of 3) <!-- pkg1109-x1 -->.
2. **The runtime and serving stack set the window and the overflow behaviour** (Section 4.5). Ollama gives
   llama3.1:8b 4096 tokens on the Intel mini PC and 131072 on the AMD APU <!-- T2S-vs-X2-default-ctx --> and answers an
   over-length prompt with success after keeping num_ctx/2 + 2 tokens <!-- ollama-overflow-keeps-half -->; other
   stacks reject it or, like OpenVINO (provisionally), also run silently past the window <!-- pkg1109-contract1 -->.
3. **The consequences for agents, with their mechanism** (Section 4.6). History loss is silent in every model tested,
   8B to 70B <!-- H-scale-verdict --> <!-- H-scale-family-verdict -->; on two task workloads a 4096-token window more
   than triples wrong task end states, all silent on Ollama; at 4096 a context shift and a token cut discard the
   system prompt <!-- R2-mechanism-lowlevel -->.
4. **What to set, and what still fails** (Sections 4.7 and 4.7p). Set the window per device in a Modelfile and check
   the loaded value <!-- window-override-methods -->, prefer a quantized KV cache to a smaller window
   <!-- b5-policy-metrics -->, and set `num_keep` to the system prompt's length <!-- num-keep-guardrail-test -->. History
   is still lost at small windows <!-- num-keep-guardrail-test -->, trimming still fails qwen3:14b
   <!-- R2-mitigation-v1-verdict --> <!-- t2s-r2-mitigation-cpu -->, and no check we tested finds the silent wrong
   answers; the runtime's shift rule matches every shift (misses 0 <!-- R2-shift-predictor-confusion -->) but not them.

### 4.2 Background and related work

<!-- Citations below were checked against their sources on 2026-11-13 (bib: docs/paper1/references.bib). A citation still marked "to verify" has an open question listed under References. -->

**Context windows in local runtimes.** llama.cpp allocates a KV cache for n_ctx tokens when a model loads. When a
sequence would grow past it, the server can perform a *context shift*: keep the first n_keep tokens, discard a block
after them and continue. Ollama serves models through a bundled llama.cpp server and adds two layers: its chat
endpoint drops whole messages from the front of an over-length conversation while keeping the system message
<!-- R2-mechanism-verdict -->, and it starts the server with context shift on and a keep of 4, so an over-length prompt
succeeds after keeping num_ctx/2 + 2 tokens <!-- ollama-overflow-keeps-half -->; the option `num_keep` sets the kept
prefix. Without a client `num_ctx`, Ollama picks the window from the detected device. None of this is reported.

**Long-context behaviour.** Lost in the Middle [Liu et al. 2024, Lost in the Middle] finds the middle of a long input
used worse than its ends; RULER [Hsieh et al. 2024, RULER] finds effective context shorter than the advertised window;
LongBench [Bai et al. 2024, LongBench] notes that most LLMs "can only handle texts a few thousand tokens long" and
finds that GPT-3.5-Turbo-16k "still struggles on longer contexts"; and multi-turn [Laban et al. 2025, LLMs Get Lost in
Multi-Turn Conversation] and rule-following benchmarks [Zhou et al. 2023, IFEval] [Mu et al. 2023, Can LLMs Follow
Simple Rules?] measure how well a model uses or obeys its input. All assume the model receives the whole input.

**KV-cache management.** StreamingLLM [Xiao et al. 2024, StreamingLLM] keeps initial attention-sink tokens plus a
recent window, much as a context shift keeps n_keep tokens; H2O [Zhang et al. 2023, H2O] and SnapKV [Li et al. 2024,
SnapKV] evict by attention; KIVI [Liu et al. 2024, KIVI] and KVQuant [Hooper et al. 2024, KVQuant] quantize the cache;
LLMLingua [Jiang et al. 2023, LLMLingua] compresses the prompt. We measure what shipped runtimes discard by default.

**Agents, routing and delegation.** AgentBench [Liu et al. 2024, AgentBench], tau-bench [Yao et al. 2024, tau-bench],
function-calling evaluations [Patil et al. 2024, Gorilla] and SWE-bench [Jimenez et al. 2024, SWE-bench] score task
success, usually with hosted models and large windows.
RouteLLM [Ong et al. 2025, RouteLLM] learns from preference data the probability that the strong model wins and routes
on it, Hybrid LLM [Ding et al. 2024, Hybrid LLM] routes on predicted query difficulty, cascades [Chen et al. 2023,
FrugalGPT] escalate when a scorer distrusts the cheap answer, and self-verification [Weng et al. 2023,
Self-Verification] and sampling-consistency checks [Manakul et al. 2023, SelfCheckGPT] let a model test its own answer;
all need a signal that correlates with a wrong answer. EdgeWise [Shi et al. 2026, EdgeWise] learns which steps of a
long agent task to delegate to the cloud, EC2-Bench [Huang et al. 2026, EC2-Bench] benchmarks edge-cloud
collaboration, Minions [Narayan et al. 2025, Minions] has a cloud model decompose tasks for an on-device model,
Arch-Router [Tran et al. 2025, Arch-Router] routes by user preferences, and llmfit [llmfit] estimates which models fit
a machine's memory. None measures how the window a runtime loads changes an agent's correctness.

**Silent truncation is known; its rate, cause and device dependence are not.** An issue open since 2024 asks Ollama to
report truncation from exceeding the context size [Ollama issue #3839]. A 2026 issue [Ollama issue #14259] documents
that chat-history truncation is silent and that its only log line is at debug level (`slog.Debug`), invisible at the
default level; a comment reports the same on v0.34.0 with ROCm. A practitioner write-up [Desbiens 2026, ContextPaw]
shows Ollama returning success after evaluating 16,387 tokens of a 160,689-token prompt at `num_ctx` 32768, the model
confabulating the lost fact, while llama-server at a 4096 window rejected an over-length request with
`exceed_context_size_error`; with only a marker saying content was gone, its compaction recovered 0 of 3 facts. These
are single cases, without rates, device dependence or memory analysis; we add measured rates on agent tasks, the
device-dependent default windows and the mechanism.

**Inference on edge hardware.** PagedAttention [Kwon et al. 2023, vLLM] manages KV memory on servers; MELT [Laskaridis
et al. 2024, MELTing Point], PowerInfer [Song et al. 2024, PowerInfer] and LLM in a flash [Alizadeh et al. 2024, LLM in
a Flash] characterize or accelerate inference on phones, consumer GPUs and flash-backed devices, not behavioural
correctness under a runtime's memory and context policy.

### 4.3 Platforms, harness and methodology

<!-- Convention: measured numbers carry their register id; harness design constants carry "design: <source file>". -->

**Platforms and runtimes.** We study four consumer machines from three GPU vendors (Table 1): an AMD Ryzen AI Max+
(Strix Halo) APU whose Radeon iGPU shares unified memory with the CPU; a laptop with an NVIDIA RTX 4070 Laptop GPU and
8188 MiB <!-- blade-k1-layer-split --> of VRAM; an Intel Core Ultra (Panther Lake) mini PC with an Arc iGPU on unified
memory; and an Intel laptop whose iGPU draws on a configurable share of system memory (Sections 4.4 and 4.5). Ollama's
device detection drops the Intel iGPU and falls back to the CPU <!-- x2-device-detect-mechanism -->, so we measure the
mini PC twice: on the CPU, as shipped, and on the iGPU, enabled only for the job's own server process. All agent runs
use Ollama; the AMD and NVIDIA runs share one pinned release and the model files.

**The R2 agent harness.** A session is a seeded, deterministic agent conversation of 40 turns
<!-- R2-v1b-qwen3-8b-reversal -->. The system prompt carries five <!-- design: harness/x2_r2_agent.py --> checkable
rules (JSON answer with fixed keys, call a `log_event` tool, never print a forbidden token, lengths in metres, end with
a session code), a two-tool schema and recall facts. Each turn is two <!-- design: harness/x2_r2_agent.py --> calls:
the first returns tool calls, answered with a long filler payload that grows the context as real tool output does;
the second asks for the final answer, at temperature 0 <!-- design: harness/x2_r2_agent.py -->. Canaries in the system
prompt and history tags in the first user message detect lost context; each canary turn asks for a new pair
<!-- R2-validation-baseline -->. A miss counts as truncation only while the final prompt exceeds the context Ollama
reports as loaded (`/api/ps`, read every turn). R2 runs withhold tools on the final call; Section 4.6a repeats the
main contrast with tools on every call.

**Validation gates.** Each model first passes a validation, with gates fixed in advance, at a window where nothing is
truncated. A rule is *in use* for a model only if the model follows it on at least 90%
<!-- x2-evalset-val-v2-gates --> of baseline turns; rules below the bar are dropped, not repaired. For llama3.1:8b only
rules 1, 3 and 4 <!-- R2-validation-v2-baseline --> are in use; qwen3:14b kept all five. A real job refuses to start
unless its model's validation passes.

**Gated silent failure.** For each session we take the first turn on which a gated metric fails. The failure is
*silent* when no HTTP error or error field appeared on any earlier turn, and *after-window* when it falls at or after
the first turn whose prompt exceeded the loaded window; otherwise it is *baseline-type*. The window effect is the
*excess over default*: a tier's after-window rate minus the Ollama default tier's, with a 95% percentile bootstrap CI
over sessions (10000 <!-- R2-v1b-excess-over-default --> resamples, stratified by tier and model); where both cells are
0/n or n/n we report a one-sided Fisher exact test over sessions instead <!-- session-exact-tests -->.

**Pre-registration and provenance.** The R2 kill criterion was written before any session ran: the claim fails if, in
every arm, the first failure comes only after an explicit error in the same session. The gated definition came later,
as analysis, and is reported beside the criterion. Every follow-up was pre-registered (hypothesis, verdict rule and
validity conditions committed before deployment), and we report verdicts in that wording, refutations included. The
second task workload, the Intel laptop and the serving-stack probes use a separate harness whose predictions were
sealed before each run; we report its verdicts as sealed, misses included. Every number is pasted from a numbers
register that one script regenerates from the raw result files. Runs before 2026-10-14 rest on byte-identical reruns
for their harness identity (identical final text 120/120 <!-- R2-qwen3-14b-4096-reconciliation -->). Rows are never
pooled across platforms, and a configuration flag counts only when the runtime's log shows it applied.

**Table 1 and figure spec.**

- **Table 1.** Platforms and versions: CPU and GPU, memory architecture, the device the runtime used, GPU driver, OS
  build and runtime versions, per platform. Configuration record, not a measured result; each version is copied from a
  recorded file (docs/paper1/md2tex.py, TABLE1, names the source of each), and a dash marks one not recorded.

### 4.4 Memory-architecture boundaries

**Claim.** Running out of GPU memory has a different shape on each architecture (Figure FH1), and on the AMD APU and
the Intel laptop the boundary did not follow the memory figure the platform reports.

**The AMD APU spills past its dedicated block, not its advertised heap, at almost no cost.** We ran llama-3.3-70b and
qwen3-32b in llama-server with the Vulkan backend and every layer on the GPU (`-ngl 99`
<!-- fh1-amd-boundary-llama-3.3-70b -->), stepping the window and recording per-process GPU memory, decode rate and time
to first token (TTFT); a point spills when shared GPU usage grows by more than 1024 MiB over the smallest window
<!-- fh1-amd-boundary-llama-3.3-70b -->. For llama-3.3-70b spill began between windows 65536 and 100096, where the
process held 1.122 of the 65536 MiB dedicated block but only 0.965 of the 76218 MiB device-local heap the driver
advertises <!-- fh1-amd-boundary-llama-3.3-70b -->, so Figure FH1 normalizes the AMD axis by the dedicated block. Crossing the
onset is nearly free, decode ratio 0.983 and TTFT ratio 1.004, and nothing in the response reports the spill
<!-- fh1-amd-boundary-llama-3.3-70b -->; the server failed only far beyond, past the trained context (Table 3). For
qwen3-32b no unclamped spill point was measured: its onset lies above 40960 <!-- fh1-amd-boundary-qwen3-32b -->
<!-- fh1-amd-qwen3-32b-spill-sweep -->.

**The Intel mini PC's iGPU has a hard wall that first-principles accounting predicts.** A model either starts or the
runtime refuses it, and across five models the refusal point is consistent with one shared-heap limit L in (48112.7,
48124.0] MiB, 247 MiB above the 47866 MiB budget the driver reports <!-- A-24-effective-heap-limit -->; the limit
counts host-visible buffers as well as device memory (Table 3). Counting weights, KV cache and compute buffers against
L, fitted leave-one-model-out, predicts each model's flip point within -257 to -26 tokens, against -2890 to -820 tokens
for model file size alone <!-- A-24-flip-point-prediction --> (Figure F2). The flip points range from (23296, 23552]
tokens for llama-3.3-70b to (343808, 344064] for llama31-8b, with qwen3-32b at (115712, 115968] and the other two at
(305152, 305408] (qwen3-8b) and (320256, 320512] (qwen3-30b-a3b-2507) <!-- A-24-flip-point-per-model -->. These are
start tests on one driver and boot state; where the flip point lies beyond the trained context it is a memory boundary,
not a usable window.

**The Intel laptop pages silently, and accounting does not predict where.** On the laptop we served Qwen3-4B through
OpenVINO on the iGPU. A prompt length is a *paging onset* when the first turn completes but host free memory reaches a
fixed floor and prefill runs well above its cold-cache time; we located it by bisection for 16-, 8- and 4-bit KV caches
(f16, i8, u4) <!-- pkg1109-x1 -->. No paging point raised an error, and the allocation per token matched the
accounting of KV cache and activations, but all three onset predictions missed, and so did dividing the free memory
reported after load by that allocation (0 of 3) <!-- pkg1109-x1 -->: the allocation is right, the memory it is divided
into is not. Lowering the iGPU's share of memory to 50% did not lower the onset <!-- pkg1109-w3-50 -->. An exploratory
fit finds a reserve of a median 1,238 MiB between the free memory reported after load and what a prompt can use, but it
differs by model <!-- pkg1109-os-reserve -->, and a sealed per-model predictor of the largest prompt that holds hit 5
of 12 cells <!-- pkg1109-e2 -->. Paging, not prompt length, causes the break: the 64 GB Intel mini PC held prompts of up
to 97,000 tokens that the laptop took 2,913 s to prefill <!-- pkg1109-t1 -->, and chunked prefill moved the laptop's i8
onset to about 79,000 to 80,000 tokens <!-- pkg1109-c256 --> (Table 3).

**On the NVIDIA laptop GPU, Ollama moves layers to the CPU above 8192 tokens.** With 8188 MiB of VRAM
<!-- blade-k1-layer-split -->, Ollama keeps llama3.1:8b fully on the GPU at windows of 4096 and 8192 and moves layers
to the CPU above them, with VRAM shares of 0.870 at 16384 and 0.656 at 32768; for qwen3:8b the shares are 0.805 and
0.592 <!-- blade-k1-layer-split -->.

| measurement | value |
|---|---|
| AMD: candidate boundaries | dedicated block 65536 MiB; device-local heap 76218 MiB; Vulkan total 114327 MiB <!-- fh1-amd-boundary-llama-3.3-70b --> |
| AMD llama-3.3-70b: last window without spill; first with | 65536; 100096, served for a request of 100000 <!-- fh1-amd-boundary-llama-3.3-70b --> |
| AMD: decode at 65536 / 100096 / 131072, one 1855-token prompt | 5.284 +/- 0.004 / 5.196 +/- 0.010 / 5.176 +/- 0.015 tok/s; 18540 MiB more shared memory at 131072 <!-- fh1-amd-boundary-llama-3.3-70b --> |
| AMD: past the trained context of 131072 (clamped) | last start 221440, first failure 221696, exit code 3221226505 <!-- fh1-amd-boundary-llama-3.3-70b --> |
| AMD qwen3-32b, 20480 to 40960 (b10970, 1 warm-up + 5 calls) | shared memory growth at most 20 MiB; decode 11.271 to 11.368 tok/s <!-- fh1-amd-qwen3-32b-spill-sweep --> |
| Intel mini PC: limit L | from 92 lower and 22 upper bounds; device-only accounting inconsistent: largest start 47945.2 MiB, smallest refusal 47562.5 MiB <!-- A-24-effective-heap-limit --> |
| Intel mini PC: flip-point error | shared heap -257 to -26 tokens (-0.11% to -0.01%); device only -1092 to +968; file size -2890 to -820 <!-- A-24-flip-point-prediction --> |
| Intel laptop: onset f16 / i8 / u4 (23 points each) | (35,500, 36,000] / (44,000, 45,000] / (48,000, 59,000] tokens; 234,957 / 171,615 / 134,760 bytes per token; floor reached at or below the onset in 3 of 3 <!-- pkg1109-x1 --> |
| Intel laptop at a 50% share, i8 / u4 | (45,000, 46,000] / (57,000, 58,000]; "the share does not matter" HIT, "the share moves the onset" MISS <!-- pkg1109-w3-50 --> |
| Intel laptop: reserve after load (exploratory) | median 1,238 MiB (IQR 1,085 to 1,506; n = 99; 277 probes, 140 in the fit); 793 to 2,264 MiB by model; static error cut 2-11x <!-- pkg1109-os-reserve --> |
| Intel laptop: chunked prefill | i8 about 79,000 to 80,000 (79,000 paged 2/4, 80,000 paged 3/7 repeats); u4 held to 80,000 (paged 0/1); the i8 location and three prefill and free-memory predictions missed <!-- pkg1109-c256 --> |

Table 3: memory-boundary measurements behind Section 4.4 and Figure FH1.

**Figure specs.**

- **FH1.** Cost of crossing the memory boundary on the AMD APU and the Intel mini PC. x: memory in use over the
  boundary (AMD: the 65536 MiB dedicated block; Intel mini PC: the shared-heap limit L). (a) decode and TTFT ratio to the
  smallest window, labelled with the served window (requested -> served when clamped); (b) start or refuse per model.
  Each AMD point is one call per fresh server start without a warm-up call (5 or 3 calls per point), a departure from
  this paper's 1 warm-up + N convention.
- **F2.** Intel mini PC flip point per model: measured (last start, first refusal] against accountings A (file size),
  B (device only) and C (shared heap, leave-one-model-out) <!-- A-24-flip-point-per-model --> <!-- A-24-flip-point-prediction -->.

### 4.5 Runtime policy

**Claim.** The runtime's policy for the detected device sets the window, and the serving stack decides what happens
past it.

**Default windows follow the detected device.** Ollama discovers the AMD iGPU through ROCm, so the AMD APU runs on its
iGPU, while the Intel mini PC, with no ROCm path, falls through to the CPU <!-- x2-device-detect-mechanism -->. On the
Intel mini PC the runtime loaded 4096 tokens for each of three models; on the AMD APU it loaded 262144 for
qwen3-4b-2507, 131072 for llama3.1:8b and 40960 for qwen3:8b <!-- T2S-vs-X2-default-ctx -->, each the model's native
window (capped only for qwen3:8b <!-- K1v3-X2-table -->). The discrete GPU also got 4096 for both models measured there
<!-- blade-k1-layer-split -->. The runtime's log reports the rule: a default sized from detected VRAM, 4096 for the
discrete GPU's 8.0 GiB and 32768 for the Intel iGPU's 36.2 GiB when that iGPU is enabled <!-- gpu-memory-readonly -->
(Table 2, Figure F5).

| platform (device used) | default window, three models | over-length prompt at the default |
|---|---|---|
| AMD APU (iGPU via ROCm) | 262144 / 131072 / 40960 <!-- T2S-vs-X2-default-ctx --> | success, half kept <!-- ollama-overflow-keeps-half --> |
| Intel mini PC (CPU) | 4096 / 4096 / 4096 <!-- T2S-vs-X2-default-ctx --> | success, half kept <!-- ollama-overflow-keeps-half --> |
| NVIDIA laptop (discrete GPU) | not measured / 4096 / 4096 <!-- blade-k1-layer-split --> | HTTP 200, evaluated 2050 <!-- blade-k1-layer-split --> |

Table 2: default context window per platform for qwen3-4b-2507 / llama3.1:8b / qwen3:8b, one runtime (Ollama).

**The default decides the outcome.** On the Intel mini PC as shipped (CPU), both models loaded 4096 tokens, crossed it
at turn 3 and failed silently on Ollama 0.34.4 in 5/5 sessions each, every failure after the window
<!-- t2s-r2-real-cpu -->. On the iGPU they loaded 32768, crossed it only at turn 23 (llama3.1:8b) and turns 20 and 21
(qwen3:14b), and no failure came after the window: 0/5 for llama3.1:8b and, for qwen3:14b, 1/5 of baseline type
<!-- t2s-r2-real-igpu -->. The pre-registered kill criterion was not triggered on either backend <!-- t2s-r2-real-cpu -->
<!-- t2s-r2-real-igpu -->. This contrast changes backend and window together; Section 4.6a isolates the window.

**On Ollama, overflow is silent.** When a prompt exceeds the loaded window, Ollama returns success and processes
num_ctx/2 + 2 tokens in all 4 checked cases (4096->2050, 8192->4098, 32768->16386, 40960->20482), through its own
`--context-shift --keep 4` server flag <!-- ollama-overflow-keeps-half -->; the discrete GPU behaved the same (Table 2).

**Whether overflow is silent depends on the serving stack.** For each stack we pre-registered what an over-length prompt
returns and probed seven prompt lengths, three repeats each, under the stack's own defaults (Table 4). On the Intel
laptop Ollama and, provisionally, OpenVINO returned success past the window, while llama-server and LM Studio rejected over-length
prompts with an error <!-- pkg1109-contract1 -->; on the AMD APU with 96 GB assigned to the GPU, Ollama truncated
silently and llama-server on ROCm rejected <!-- pkg1109-evo-x2-split96 -->. Silence on overflow is therefore not unique
to Ollama, and loud stacks answer with an HTTP error. We report no Vulkan outcome at 96 GB: the GPU vendor's installer
silently installed a new driver during that step <!-- pkg1109-evo-x2-split96 -->.

| stack (platform) | over-length prompt |
|---|---|
| Ollama (Intel laptop) | matched 7 of 7 probes: the first fit, and the six above the window returned success after evaluating 2,050 tokens <!-- pkg1109-contract1 --> |
| llama-server, SYCL (Intel laptop) | auto-fitted a slot of 37,376 tokens and rejected prompts over it; matched every probe it reached (one only conditionally on that slot; the third repeat stopped after four probes) <!-- pkg1109-contract1 --> |
| llama-server, Vulkan (Intel laptop) | reached only three probes in one repeat <!-- pkg1109-contract1 --> |
| OpenVINO (Intel laptop) | matched the first four probes; on the last three, predicted to fail loudly out of memory, it returned OK, ran silently past the window and returned OK, 3/3 each; its past-the-window mode ran silently past the window on all 12 of its sealed repeats that ran (12 of 20, provisional) <!-- pkg1109-contract1 --> |
| LM Studio (Intel laptop) | predicted silent on probes 3 to 7; rejected each with HTTP 400, 3/3 per probe <!-- pkg1109-contract1 --> |
| Ollama (AMD APU, 96 GB to the GPU; Qwen3-4B Q4_K_M, 40,960-token window) | OK through 33,024 tokens; silent truncation at 41,216 and 45,056 (3/3 each) <!-- pkg1109-evo-x2-split96 --> |
| llama-server, ROCm (same) | OK through 33,024; rejected at 41,216 and 45,056 (3/3 each) <!-- pkg1109-evo-x2-split96 --> |

Table 4: overflow contract per serving stack (pre-registered; seven prompt lengths, three repeats).

**Only a Modelfile window setting survived a restart.** On the AMD APU, a per-request `num_ctx` and
`OLLAMA_CONTEXT_LENGTH` both applied but reverted to the default of 131072 after a restart, while a Modelfile
`PARAMETER num_ctx` on a derived tag persisted <!-- window-override-methods -->.

**Within the window, KV precision is the lever.** On the AMD APU, a pre-registered comparison of eight context policies
on the Pareto frontier of rules, history, time and cost (Section 4.7) found that every policy kept every rule and that
the frontier held one quantized-KV-cache policy per model <!-- b5-policy-metrics -->: a q4_0 cache took 6.6 min per
llama3.1:8b session against 43.7 min for the default, and a q8_0 cache 75.4 min per qwen3:14b session against 214.8
min, at equal or higher history recall (Table 5). The speedup is KV precision, not the flash attention our quantized
policies also turn on <!-- design: harness/policy_compare.py -->: a control with flash attention and an f16 cache stayed
within tolerance of the default <!-- b5-fa-control -->. With the window capped at 8192 and 4096 (a simulated
constraint), the router (policy d) was the only policy that kept history, but client trimming was also on the frontier
in every capped cell, so the pre-registered verdict is *refuted* <!-- b5-sim-constraint --> <!-- b5-verdict -->.

**An 8-bit cache leaves output nearly unchanged; a 4-bit cache does not, but keeps long-context quality.** For qwen3:14b
at 40960 the server log reported the K and V types each setting names, and the loaded sizes match them
<!-- kv-precision-evidence -->. Under greedy decoding an 8-bit cache matched the f16 output on almost every comparable
turn, while a 4-bit cache diverged within the first two turns of every seed <!-- kv-output-agreement --> (Table 5). In a
sealed comparison on the Intel mini PC's iGPU under OpenVINO, with Qwen3-4B-Instruct-2507, greedy decoding and RULER
items at three context lengths, a 4-bit cache was non-inferior to a reference cache (16-bit, or 8-bit at the length
where 16-bit does not fit) within the pre-registered 5-point margin
<!-- design: sealed prediction file; reference f16 at the two shorter lengths, i8 at the longest --> <!-- pkg1109-qual1 -->.
On multi-turn function calling (BFCL) the pre-registered test lacked power, and we draw no conclusion about tool use
under a 4-bit cache <!-- pkg1109-qual1 -->.

| measurement | value |
|---|---|
| llama3.1:8b, q4_0 vs default | 6.6 vs 43.7 min per session; history 100.0% (24/24) vs 95.8% (23/24) <!-- b5-policy-metrics --> <!-- paper-percentage-counts --> |
| qwen3:14b, q8_0 vs default | 75.4 vs 214.8 min; history 62.5% (15/24) for both <!-- b5-policy-metrics --> <!-- paper-percentage-counts --> |
| flash attention, f16 cache | 43.4 min (llama3.1:8b), 206.6 min (qwen3:14b) <!-- b5-fa-control --> |
| router under the cap, llama3.1:8b / qwen3:14b | history 80.0% / 62.5% at 8192 and 97.5% / 62.5% at 4096; every other policy at most 12.5%; on the frontier in all four capped cells <!-- b5-sim-constraint --> |
| KV buffers loaded (qwen3:14b, 40960) | 16, 8.5 and 4.5 bits per element; the default's `/api/ps` size equals f16 with flash attention to the byte <!-- kv-precision-evidence --> |
| output agreement, identical first-call prompts | f16 with vs without flash attention 120/120 turns; f16 vs q8_0 103/105 (98.1%), tool calls 104/105 (99.0%), first divergence at turn 30 and 35 in two seeds, never in the third; q4_0 diverged at turn 1 or 2 in every seed, 5 of 120 turns comparable, 2/5 matching <!-- kv-output-agreement --> |
| RULER, 4-bit vs reference, 270 paired items | -2.2 points, 95% CI [-5.0, +0.6]: non-inferior (HIT) <!-- pkg1109-qual1 --> |
| BFCL multi-turn, 8-bit vs 4-bit | 16% vs 13% over n = 100 paired items, +3.0 points (95% CI -5.9 to +12.0); missed for lack of power <!-- pkg1109-qual1 --> |

Table 5: KV-cache policy, output and quality measurements behind Section 4.5.

**Figure spec.**

- **F5.** Default window per platform and model over the CDF of agent step lengths in public traces
  <!-- T2S-vs-X2-default-ctx --> <!-- blade-k1-layer-split --> <!-- trace-fraction-exceeds-4096 -->.

### 4.6 Agent consequences

#### 4.6a The window result

**Silent failures rise as the window shrinks.** On the AMD APU we ran three models at the Ollama default and at 32768,
16384, 8192 and 4096 <!-- R2-v1b-excess-over-default -->, five seeds per cell: 75 sessions and 3000 turns
<!-- R2-real-v1b-first-events -->, none surfacing an error before its first failure <!-- R2-real-v1b-first-events -->.
After-window silent failures went from 0/15 at the default and at 32768 to 10/15 at 4096, and the excess over the
default excludes zero at 8192 and 4096 <!-- R2-v1b-excess-over-default --> (Table 6, Figure F6). Rule failures, as
distinct from recall misses, appear only at 8192 and 4096 <!-- R2-real-v1b-first-events -->, and at 4096 no session of
any model kept its history to the first canary turn <!-- R2-real-v1b-survival -->. The default tier's 3/15 gated
failures <!-- R2-v1b-default-baseline-type --> are llama3.1:8b recall misses inside the window; the pre-registered kill
criterion, counting every metric, finds 10/15 silent sessions there, all baseline-type, so it is not triggered
(killed=False <!-- R2-real-v1b-kill-criterion -->).

**Models differ** (Table R2-gated). llama3.1:8b fails first: 2/5 at 16384 <!-- R2-v1b-gated-per-model --> and 5/5 at
4096, and 10/10 at 4096 against 0/10 at the default when extended to 10 seeds <!-- session-exact-tests -->. qwen3:14b
has no after-window failure above 4096 <!-- R2-v1b-gated-per-model -->; at 4096 its sessions lapse on rule 5 or rule 3
at turns 5 to 30, while rule 2 held on 40/40 turns in every seed <!-- R2-qwen3-14b-4096-reconciliation -->. qwen3:8b
reversed the ordering: extended to 10 seeds per tier it failed silently in 6/10 sessions at 8192 and 2/10 at 4096, which
the pre-registered rule reads as noise <!-- R2-qwen3-8b-reversal-10seed -->. We do not claim that 4096 is safe for
qwen3:8b.

**On tool-using tasks the window effect holds on two runtimes: silent on Ollama, loud on llama-server** (Figure FW). A
pre-registered rerun offers tools on every call, on a task workload (B3), five seeds per tier plus a `num_keep` tier
(Section 4.7). It rests on qwen3:14b alone, which ran under a validation exception declared before the data (Section
4.8): llama3.1:8b and qwen3:8b failed the task validation <!-- x2-evalset-val-v2-gates -->
<!-- x2-evalset-tasks-val-v2-qwen3-8b-gates -->, so their real runs refused to start. On Ollama 0.34.4 qwen3:14b ended
42.0% of task turns wrong at 4096 against 11.3% at the default, every wrong turn silent
<!-- x2-evalset-tasks-real-v2-wrong-rate -->, and its gated failures rose as pre-registered
<!-- x2-evalset-tasks-real-v2-gated-excess -->; the wrong rates include the baseline errors the exception admits. On a
stock llama-server, five sessions per arm, the default loaded 40960 and the same model ended 8.0% of task turns wrong
there and 100.0% at 4096, but no wrong turn at 4096 was silent: the server rejected prompts over the window, stopped at
the context limit or failed with another error, and only 3.3% of task turns completed a tool round
<!-- xruntime-llamacpp-wrong-rate -->. With the window fixed at 40960, llama-server matches Ollama's default-window rate
<!-- xruntime-default-fixed --> (Table 6). The window effect replicates across the two runtimes; at 4096 llama-server was
loud where Ollama was silent, though silence is not Ollama's alone (Section 4.5).

**A second task workload, at the Intel default, gives the same result.** A separate pre-registered task workload, with
its own harness and validation, ran qwen3:8b on the Intel mini PC under Ollama 0.40.14, 8 seeds per arm. At the default
the runtime loaded 4096 tokens and 80.4% of task turns ended wrong, every one silently; with 24576 loaded, 25.0% did,
lower in all 8 seeds <!-- pkg1109-t4-8b --> (Figure FW, Table 6). The larger window lowered wrong task turns but not
recall errors, against the sealed prediction, and still left a quarter of turns wrong <!-- pkg1109-t4-8b -->. qwen3:8b
failed our B3 validation on the AMD APU <!-- x2-evalset-tasks-val-v2-qwen3-8b-gates -->, so this workload stands beside
B3, not as B3 task accuracy. An off-the-shelf LiteLLM failover router, or LiteLLM's pre-call check against the declared
window, in front of the local model escalated no turn and left the wrong rate unchanged <!-- pkg1109-pitch-ll -->.

| measurement | value |
|---|---|
| R2, three models, after-window silent at 16384 / 8192 / 4096 (of 15) | 2 / 8 / 10; excess over default +0.133 (95% CI +0.000 to +0.267) / +0.533 (+0.333 to +0.667) / +0.667 (+0.533 to +0.800) <!-- R2-v1b-excess-over-default --> |
| R2 llama3.1:8b at 4096, 5 seeds | 5/5 vs 0/5 sessions, excess +1.000 <!-- R2-v1b-excess-over-default -->, one-sided Fisher exact p = 0.00397 <!-- session-exact-tests --> |
| R2 llama3.1:8b, 10 seeds | 4096: 10/10 vs 0/10, p = 5.41e-06 <!-- session-exact-tests -->; 8192: +0.500 (95% CI +0.200 to +0.800); 16384: none shown <!-- R2-v1b-excess-10seed --> |
| R2 qwen3:8b, 10 seeds | 8192: 6/10, +0.600 (95% CI +0.300 to +0.900); 4096: 2/10, +0.200 (95% CI +0.000 to +0.500), none shown; Fisher exact two-sided p = 0.170 <!-- R2-qwen3-8b-reversal-10seed --> |
| B3 Ollama, qwen3:14b, wrong task turns | default 11.3% (17/150); 16384 12.7% (19/150), none shown; 8192 35.3% (53/150), excess +24.0 points (95% CI +18.7 to +29.3); 4096 42.0% (63/150), +30.7 points (95% CI +26.0 to +34.7); session-level CIs, 5 vs 5 sessions <!-- x2-evalset-tasks-real-v2-wrong-rate --> |
| B3 Ollama, gated failures at 4096 | 5/5 vs 0/5 sessions, excess +1.000 <!-- x2-evalset-tasks-real-v2-gated-excess -->, one-sided Fisher exact p = 0.00397 <!-- session-exact-tests --> |
| B3 llama-server, qwen3:14b | default (40960) 8.0% (12/150), 12 silent, tool round on 97.3%; 4096 100.0% (150/150), +92.0 points (95% CI +88.7 to +94.7, session bootstrap), 0 silent: 60 rejected over the window, 3 stopped at the context limit, 87 other errors; tool round on 3.3% <!-- xruntime-llamacpp-wrong-rate --> |
| long tasks, qwen3:14b, AMD APU, 3 seeds | default 5/60 (8.3%, Wilson 95% 3.6% to 18.1%); q8_0 KV 4/60 (6.7%, 2.6% to 15.9%) <!-- long-task-local-accuracy --> |
| B3 llama-server at a fixed 40960 | -3.3 points vs Ollama's default (95% CI -6.7 to +0.0, 5 vs 5 sessions); 0/150 context-event turns <!-- xruntime-default-fixed --> |
| second workload, qwen3:8b, Intel mini PC, 8 seeds | default (4096) 80.4% (193/240), 193 silent; 24576 25.0% (60/240); paired difference +0.554 (95% t interval +0.445 to +0.664), positive in 8 of 8 seeds (sign test p = 0.0078); cluster bootstrap +0.467 to +0.633 (exploratory) <!-- pkg1109-t4-8b --> |

Table 6: window results with their intervals (R2: rule-following harness; B3 and the second workload: tool-using tasks).

**Scope: history loss is general, rule loss is not.** A pre-registered extension to 30B to 70B models (three seeds per
cell) is not supported (mixed): qwen3:32b lost rules at 4096 in 2/3 sessions but history with rules intact in only 1/3
at 32768, and llama3.3:70b lost no rules at 4096 (0/3, every session over the window) but history in 3/3 at 32768
<!-- H-scale-verdict -->; mistral-small3.2:24b also resisted rule loss (0/3 at 4096, history loss 2/3 at 32768)
<!-- H-scale-family-verdict -->. These are rule-following results, not task accuracy.

**On long tasks capability, not memory, fails first.** On 20 synthetic long-horizon tasks step-matched to public agent
traces (10 to SWE-Gym quantiles, 10 to OpenHands traces), qwen3:14b on the AMD APU completed 5/60 at the Ollama default
(loaded window 40960) and 4/60 with a q8_0 KV cache, every wrong task silent <!-- long-task-local-accuracy --> (Table
6); qwen3:14b and qwen3:32b solved none of 30 long-horizon tasks and llama3.3:70b 3 of 30 after a pre-registered
tool-template fix <!-- long-task-scale-local --> <!-- longtask-70b-toolfix -->, and at a window that held the whole
prompt llama3.3:70b still failed 3 of 3 trace-sized tasks (prompts about 1.6x the traces
<!-- longtask-tokens-v1-sizing -->) on wrong actions <!-- longtask-tokens-v1 -->.

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

- **F6.** Session survival per tier (intact sessions against turn) <!-- R2-real-v1b-survival -->, and the after-window
  excess per tier with its 95% CI <!-- R2-v1b-excess-over-default -->.

#### 4.6b The mechanism on AMD and NVIDIA

**Ollama removes history first and, at 4096, the system prompt.** We reran llama3.1:8b on the AMD APU, one session per
tier (80 calls each <!-- R2-mechanism-lowlevel -->), with Ollama in debug mode, its log sliced per call, and a
render-only copy of each request; the rendered prompt matched the runtime's count on every check (9/9 within 1% at
4096; render/log disagreements 0 <!-- R2-mechanism-verdict -->). Ollama drops whole messages from the front once the
conversation outgrows the window, on 54/80 calls at 16384, 68/80 at 8192 and 76/80 at 4096
<!-- R2-mechanism-lowlevel -->, always keeping the system message <!-- R2-mechanism-verdict --> and usually cutting
mid-turn: history goes, the rules stay in view. At 4096 <!-- R2-mechanism-lowlevel --> two lower-level steps then
remove the system prompt. llama.cpp's context shift fired on 34 calls from turn 4, each keeping n_keep 5 tokens and
discarding the next 2045, 29 of them on the final-answer call; and once Ollama's own token cut reduced a prompt of 4097
tokens to 2050, keeping 4 <!-- R2-mechanism-lowlevel -->, the num_ctx/2 + 2 overflow rule
<!-- ollama-overflow-keeps-half -->. No step raised an error (exceed-context errors 0 <!-- R2-mechanism-verdict -->). The
hypothesis "drop old turns, keep the system message" is therefore confirmed at 32768, 16384 and 8192 and refuted at
4096 by the token cut <!-- R2-mechanism-verdict -->.

**NVIDIA replicates the first two steps.** One session at 4096 on the NVIDIA laptop passed the same render check (9/9
within 1%); Ollama trimmed whole messages on 75/80 calls from turn 3 and kept the system message on every one, and the
context shift fired 33 times; no token cut occurred and no error surfaced <!-- blade-mechanism-4096 -->. That row lacks
n_keep, so the claim that the shift discards the system prompt is AMD-only.

**Tool calls written as text, a second silent failure, follow the llama.cpp backend.** In the NVIDIA validation,
llama3.1:8b's negative control at 32768, where nothing is truncated, missed canaries in two of five sessions
<!-- blade-r2-gate-v2 -->: the final answer was a tool call written as text. Three pre-registered AMD checks each moved
one factor toward the NVIDIA configuration: ROCm at 32768 gave 0 of 20 misses <!-- x2-r2-llama-val-32k -->, forcing
NVIDIA's CPU offload split (22 of 33 layers on the GPU, VRAM share 0.656) again 0 of 20
<!-- x2-r2-llama-val-32k-offload -->, and switching only the backend, to Vulkan on the same GPU, 2 of 20, as a text call
to `lookup_fact` <!-- x2-r2-llama-val-32k-vulkan -->. The failure follows the backend (CUDA and Vulkan, not ROCm, in
these seeds), not the GPU vendor, but on Vulkan that rests on one session, and the kernel is not identified.

### 4.7 Fixes

**Client-side trimming fixes llama3.1:8b but not qwen3:14b, on two vendors.** Our pre-registered client fix counts each
request with the model's tokenizer and drops whole old turns until the system prompt, the kept turns and the answer
budget fit the window with a 5% <!-- design: harness/x2_r2_client_trim.py --> margin. A model *recovers* when every
rule in use is at or above 90% <!-- design: analysis/r2_mitigation_report.py -->, and the hypothesis needed both models
to recover at 4096 <!-- R2-mitigation-v1-verdict -->. Mechanically it worked: 0 context shifts, 0 token cuts, 0 Ollama
message drops and 0 exceed-context errors in every cell <!-- R2-mitigation-v1-observed -->. The verdict is *refuted*
<!-- R2-mitigation-v1-verdict -->: on the AMD APU llama3.1:8b recovered at 8192 and 4096, and qwen3:14b at 8192 but not
at 4096 <!-- R2-mitigation-v1-verdict -->, where its `log_event` rule fell to 61.7% (74/120)
<!-- R2-mitigation-v1-4096 --> <!-- paper-percentage-counts -->; the Intel mini PC's CPU replicates the split
<!-- t2s-r2-mitigation-cpu -->. The cause is the fix itself: at 4096 the budget leaves room for only 1-2
<!-- R2-mitigation-v1-observed --> turns besides the system prompt, so trimming removes the examples of the two-call
pattern the model imitates: on the first failing trimmed turn qwen3:14b skipped the `log_event` call that the
untrimmed run made first <!-- R2-qwen3-14b-4096-reconciliation -->.

**num_keep keeps one model's rules in view, but not its history or task accuracy.** We pre-registered a test for
llama3.1:8b at 4096 with `num_keep` set to the system prompt's measured length (601, 594 and 597 tokens per seed). All
105 context shifts kept n_keep 595, 598 or 602, and the rules in use held, which the pre-registered rule reads as
"one-line fix" <!-- num-keep-guardrail-test -->. It fixes the rules and nothing else: Ollama still dropped messages on
224 of 239 calls and the history canary was missed on 24 of 24 checks <!-- num-keep-guardrail-test -->. In the protocol
check the `num_keep` tier lowered qwen3:14b's wrong-task rate at 4096 from 42.0% to 30.0%, but its excess over the
default stayed positive <!-- x2-evalset-tasks-real-v2-wrong-rate --> <!-- x2-evalset-tasks-real-v2-gated-excess -->
(Table 7). What lowered that rate is not established: in the R2 harness the same flag changed nothing for qwen3:14b at
4096 <!-- R2-qwen3-14b-4096-reconciliation -->.

**A cloud router does not dominate.** The eight <!-- design: harness/policy_compare.py --> policies were the default,
the largest window that starts, client trimming, a cloud router, an auto-sized window, `num_keep`, and q8_0 and q4_0 KV
caches; the hypothesis was that the router is on the Pareto frontier on every platform. It is *refuted*: on the AMD APU
the router was dominated for both models <!-- b5-verdict -->. Every policy kept every rule (300/300 for llama3.1:8b and
540/540 for qwen3:14b <!-- b5-policy-metrics -->), and the router was slower or costlier than the KV-cache policies at
the same or higher recall (Section 4.5, Table 7); under the simulated cap it joined the frontier, beside client trimming
<!-- b5-sim-constraint -->.

**Recovering after a silent failure is costly and incomplete.** A cloud model given the whole client-side history late,
on every wrong task of the protocol check, fixed 84.8% (167/197) <!-- cloud-redo-full-context --> at 1.76x the cost of a
fresh run of the same task <!-- late-escalation-cost-vs-fresh -->, and it cannot undo what the local model did: of the
30 tasks still wrong, 28 needed a repair of changed state <!-- cloud-redo-full-context -->. Rolling back the executed
actions first, on simulated tools, fixed all 28 against a pre-registered target of 25/28 <!-- own-rollback-redo -->,
and 195/197 wrong local tasks overall with the full-history redo <!-- cloud-rescue-combined -->; replacing earlier
turns by reference calls stands in for a successful escalation of each, not a live deployment. Deciding at session start
instead, no measured device qualifies at matched quality <!-- hybrid-route-at-start --> (Table 7).

**No detector we tested finds the wrong tasks.** Every pre-registered attempt to find or avoid the wrong tasks missed
its target (Table 7): a tuned post-run check <!-- post-run-check-tuned -->, a session-start fit predictor
<!-- fit-predictor-v1 -->, a route, check and redo pipeline <!-- hybrid-full-pipeline-v1 -->, our own compaction, which
summarizes old turns <!-- own-compaction-8192 --> <!-- own-compaction-4096 -->, and every route-at-start policy
<!-- hybrid-route-at-start-v2 -->. On the Intel mini PC the strict gate refused qwen3:14b <!-- t2s-evalset-tasks-cpu -->
<!-- t2s-evalset-tasks-igpu -->, so Intel task accuracy comes from the second workload (Section 4.6a).

| measurement | value |
|---|---|
| client trimming, qwen3:14b at 4096 | `log_event` 61.7% (74/120), failing first at turn 3 in every seed (3/3/3) <!-- R2-mitigation-v1-4096 --> <!-- paper-percentage-counts --> |
| num_keep, llama3.1:8b at 4096 | rule 1 94.2% (113/120); rules 3 and 4 100.0% (120/120 and 60/60) <!-- num-keep-guardrail-test --> <!-- paper-percentage-counts --> |
| num_keep, qwen3:14b at 4096 (B3) | wrong 30.0% (45/150) vs 42.0% (63/150); excess +18.7 points (95% CI +16.0 to +21.3) <!-- x2-evalset-tasks-real-v2-wrong-rate -->; gated excess +1.000 (5/5 vs 0/5 sessions, p = 0.00397) <!-- x2-evalset-tasks-real-v2-gated-excess --> <!-- session-exact-tests --> |
| cloud router, per session | llama3.1:8b 43.5 min at history 95.8% (23/24); qwen3:14b 135.1 min and USD 0.8419 <!-- b5-policy-metrics --> <!-- paper-percentage-counts --> |
| late cloud redo, full history | 84.8% (167/197) fixed, task-level 95% Wilson CI [79.1%, 89.1%] <!-- cloud-redo-full-context -->; session resampling (25 sessions, stratified by tier, 10000 resamples) [79.4%, 90.4%] <!-- headline-ci-session-rescue -->; mean USD 0.0227 per task <!-- cloud-redo-full-context --> |
| rollback, then redo (stand-in) | 28/28 (95% Wilson CI [87.9%, 100.0%]), target 25/28; USD 0.8926 billed for the 28 <!-- own-rollback-redo --> |
| tuned post-run check, held-out seed at 4096 | caught 14/14 wrong tasks but escalated 15/16 correct ones (target at most 10%); rule-level checks alone 6/14 <!-- post-run-check-tuned --> <!-- post-run-check-l3-rule-breakdown --> |
| session-start fit predictor | precision 0.350 against the runtime log (MISSED) <!-- fit-predictor-v1 --> |
| route, check and redo pipeline | 81/90 tasks local at 94.4% accuracy, target 97% (MISSED) <!-- hybrid-full-pipeline-v1 --> |
| own compaction at 8192 / 4096 | 16.0% / 23.3% of tasks wrong, targets 15% / 20% <!-- own-compaction-8192 --> <!-- own-compaction-4096 --> |
| window overflow as a wrong-task flag (B3) | default: missed 17/17 wrong (recall 0.0% (0/17)); 16384: precision 15.4% (4/26), recall 21.1% (4/19); 8192: 45.3% (48/106), 90.6% (48/53); 4096: recall 100.0% (63/63), precision 42.9% (63/147), 147 of 150 turns flagged (TP 63 FP 84) <!-- impact-v3-detection-value-measured --> |

Table 7: fixes, recovery and detectors (Section 4.7).

#### 4.7p The runtime's shift rule and the window flag, characterized

**The runtime's shift rule predicts every shift.** The shift that discards the system prompt (Section 4.6b) has a
trigger readable from the llama.cpp source: the prompt left after the message-level trim, plus the tokens generated,
exceeds the loaded window. Scored against the shifts the server log recorded in the AMD mechanism sessions, it gave TP
34 FP 0 FN 0 TN 46 at 4096 and TP 10 FP 0 FN 0 TN 70 at 32768 (34/34 and 10/10), misses 0 over all tiers
<!-- R2-shift-predictor-confusion -->. The generated count is not known before the call; substituting the generation
budget, P+num_predict>n_ctx gives TP 44 FP 22 FN 0 <!-- R2-shift-predictor-confusion -->, missing no shift at the cost
of false alarms; it has no tunable parameter but was scored on the sessions it was built from.

**It predicts the shift, not the failure.** Applied to the real run as a prediction, the rule flags 102/240 calls for
llama3.1:8b at 4096 and 0/240 for qwen3:14b <!-- R2-shift-predicted-real-v1 -->, yet qwen3:14b also loses rules at
4096, through what the message-level trim removes <!-- R2-qwen3-14b-4096-reconciliation -->: at 4096, 80.3% (61/76) of
truncated calls began with an orphaned message <!-- R2-orphan-rates -->.

**Window overflow is not a general wrong-answer detector.** We measured "this task turn's prompt exceeded the loaded
window" as a predictor of "the task ended wrong" on the protocol check (qwen3:14b, AMD APU): it missed every wrong task
at the default, where no turn overflowed, and at 4096 it flagged nearly every turn
<!-- impact-v3-detection-value-measured --> (Table 7). With a large window wrong answers have other causes; with a small
one nearly every step overflows. The shift rule and the window flag explain system-prompt loss; they are not detectors
or a cheap escalation filter.

### 4.8 Threats to validity

**Platforms and versions.** We measure one machine per vendor (two for Intel), so a platform effect cannot be
separated from that machine's driver and memory configuration; on the AMD APU an unrequested driver install during one
serving-stack step confounds its Vulkan cells (Section 4.5). Agent runs use Ollama 0.34.4
<!-- R2-qwen3-14b-4096-reconciliation --> (0.40.14 for the second workload <!-- pkg1109-t4-8b -->) and direct runs
llama.cpp b10970 <!-- design: docs/HARDWARE.md (version record; no register row, gap G13) -->; default windows and
trimming are runtime code and can change in any release. The NVIDIA laptop contributes validation and mechanism
sessions only, because its validation gate held back the real agent runs <!-- blade-r2-gate-v2 -->, and the router
comparison ran on the AMD APU alone <!-- b5-verdict -->, whose default window is the model's native one.

**Statistics.** Cells have three <!-- H-scale-verdict --> to five <!-- R2-v1b-gated-per-model --> seeds, ten for
extended cells <!-- R2-qwen3-8b-reversal-10seed --> and eight for the second workload <!-- pkg1109-t4-8b -->; 0/5 and
5/5 cells carry a session-level Fisher exact test <!-- session-exact-tests -->, and the scale and family results rest
on three sessions per cell. Sampling uses temperature 0 <!-- design: harness/x2_r2_agent.py --> and a fixed request
seed, so seeds vary the session script, not the sampling. Timed or scored calls follow one warm-up plus N measured
calls (N = 3 for the descriptive KV recall check <!-- b5-kv-recall-f16-rerun -->); session times are over three seeds
<!-- cost-cut-config-time -->.

**Workload.** The R2 rules are synthetic and checkable; deployed system prompts differ. Guardrail rules 6 to 8
(privacy, refusal, confirmation) already fail at full context, for example rule 6 at 0.417 and 0.250 of bait turns
<!-- x2-evalset-val-v1-gates -->, so they support no claim here. B3 task accuracy rests on qwen3:14b, which passed the
task validation only through a declared exception: it passed every gate except task end state, 80/90 (88.9%) against a
90% bar, and an exception declared before its real run waived that gate for exactly that count
<!-- x2-evalset-val-v2-gates --> <!-- x2-evalset-val-v2-task-end-state -->. qwen3:8b failed the same validation (task end
state 65/90, 72.2%; task tool 61/81) <!-- x2-evalset-tasks-val-v2-qwen3-8b-gates -->, and so did llama3.1:8b,
mistral-small3.2:24b and llama3.3:70b
<!-- x2-evalset-tasks-val-v2-qwen3-8b-gates --> <!-- x2-evalset-tasks-real-v2-wrong-rate -->
<!-- x2-evalset-tasks-val-v2-mistral-small32-gates --> <!-- x2-evalset-tasks-val-v2-llama33-70b-gates -->; the second
workload adds qwen3:8b on the Intel mini PC under its own harness and validation.

**Open items.** Still open are the AMD APU's Vulkan cells at 96 GB, to be rerun on a fixed driver
<!-- pkg1109-evo-x2-split96 -->, the cost of crossing the NVIDIA GPU's memory boundary, whose placement alone we measured
<!-- blade-k1-layer-split -->, the router comparison on the small-default platforms <!-- b5-verdict -->, and the sealed
Intel laptop cells not yet scored <!-- pkg1109-e2 --> <!-- pkg1109-w3-50 -->, and OpenVINO's provisional overflow result <!-- pkg1109-contract1 -->.

**Cost figures.** Cloud cost per real agent task is priced from traces, not executed, with caching assumed
<!-- cloud-cost-per-real-task -->. A first cloud redo spent its whole 384-token output budget on thinking in 109 of 118
failures <!-- cloud-redo-diagnosis -->; we cite only the corrected redo (2048 output tokens per call, Section 4.7)
<!-- cloud-redo-full-context -->.

### 4.9 Conclusion

On every platform we measured, Ollama trims, shifts or cuts a conversation that outgrows the window and still
succeeds; llama-server at 4096 fails loudly <!-- xruntime-llamacpp-wrong-rate -->. An unmodified Intel CPU install ran
agents at 4096 tokens <!-- t2s-r2-real-cpu -->, where 42.0% of qwen3:14b's tasks <!-- x2-evalset-tasks-real-v2-wrong-rate --> and
80.4% of qwen3:8b's task turns <!-- pkg1109-t4-8b --> went silently wrong. Memory runs out silently too, and not where
the reported figures say (Section 4.4).

Five lessons follow. Set the window explicitly for each device and verify the loaded value
<!-- window-override-methods -->. Surface the runtime's cut-off log lines (Ollama writes them only at debug level
[Ollama issue #14259]), but do not treat them as a correctness check.
Try a quantized KV cache before a smaller window: on the AMD APU it cut session time by 85% (37.1 of 43.7 min) and 65%
(139.4 of 214.8 min, 3 sessions per setting) with no measured loss of rules or recall <!-- cost-cut-config-time -->
<!-- b5-kv-recall-f16-rerun -->. Treat trimming as a change to the prompt: dropping old turns drops the examples
the model imitates <!-- R2-qwen3-14b-4096-reconciliation -->. Validate an agent on its target backend: identical model
files and settings behaved differently on CUDA, Vulkan and ROCm <!-- blade-r2-gate-v2 -->
<!-- x2-r2-llama-val-32k-vulkan --> <!-- x2-r2-llama-val-32k-offload -->.

## References

Cited in Sections 4.2 and later. Each entry was checked on 2026-11-13 against the source named in its
`docs/paper1/references.bib` entry (title, authors, venue, year, DOI or URL, and the sentence that cites it); no
result of these papers is quoted in the draft. An entry marked "To verify" has an open question stated with it.

- [Liu et al. 2024, Lost in the Middle] N. F. Liu et al., "Lost in the Middle: How Language Models Use Long Contexts", TACL 12:157-173, 2024, doi:10.1162/tacl_a_00638 (arXiv:2307.03172). Verified 2026-11-13.
- [Hsieh et al. 2024, RULER] C.-P. Hsieh et al., "RULER: What's the Real Context Size of Your Long-Context Language Models?", COLM 2024 (arXiv:2404.06654). Verified 2026-11-13.
- [Bai et al. 2024, LongBench] Y. Bai et al., "LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding", ACL 2024, pp. 3119-3137, doi:10.18653/v1/2024.acl-long.172. Verified 2026-11-13; the citing sentence quotes the abstract ("can only handle texts a few thousand tokens long"; GPT-3.5-Turbo-16k "still struggles on longer contexts").
- [Laban et al. 2025, LLMs Get Lost in Multi-Turn Conversation] P. Laban, H. Hayashi, Y. Zhou, J. Neville, "LLMs Get Lost In Multi-Turn Conversation", ICLR 2026 (arXiv:2505.06120). Verified 2026-11-13.
- [Zhou et al. 2023, IFEval] J. Zhou et al., "Instruction-Following Evaluation for Large Language Models", arXiv:2311.07911, 2023. Verified 2026-11-13.
- [Mu et al. 2023, Can LLMs Follow Simple Rules?] N. Mu et al., "Can LLMs Follow Simple Rules?", arXiv:2311.04235, 2023 (RuLES). Verified 2026-11-13.
- [Xiao et al. 2024, StreamingLLM] G. Xiao, Y. Tian, B. Chen, S. Han, M. Lewis, "Efficient Streaming Language Models with Attention Sinks", ICLR 2024 (arXiv:2309.17453). Verified 2026-11-13.
- [Zhang et al. 2023, H2O] Z. Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models", NeurIPS 2023, doi:10.52202/075280-1506. Verified 2026-11-13.
- [Li et al. 2024, SnapKV] Y. Li et al., "SnapKV: LLM Knows What You are Looking for Before Generation", NeurIPS 2024, doi:10.52202/079017-0722. Verified 2026-11-13.
- [Liu et al. 2024, KIVI] Z. Liu et al., "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache", ICML 2024, PMLR 235:32332-32344. Verified 2026-11-13.
- [Hooper et al. 2024, KVQuant] C. Hooper et al., "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization", NeurIPS 2024, doi:10.52202/079017-0040. Verified 2026-11-13.
- [Jiang et al. 2023, LLMLingua] H. Jiang et al., "LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models", EMNLP 2023, doi:10.18653/v1/2023.emnlp-main.825. Verified 2026-11-13.
- [Liu et al. 2024, AgentBench] X. Liu et al., "AgentBench: Evaluating LLMs as Agents", ICLR 2024 (arXiv:2308.03688). Verified 2026-11-13.
- [Yao et al. 2024, tau-bench] S. Yao, N. Shinn, P. Razavi, K. Narasimhan, "tau-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains", ICLR 2025 (arXiv:2406.12045, 2024). Verified 2026-11-13; published venue is ICLR 2025.
- [Patil et al. 2024, Gorilla] S. G. Patil, T. Zhang, X. Wang, J. E. Gonzalez, "Gorilla: Large Language Model Connected with Massive APIs", NeurIPS 2024, doi:10.52202/079017-4020 (arXiv 2023). Verified 2026-11-13; the Berkeley Function Calling Leaderboard may still be the better citation for "function-calling evaluations" (operator choice).
- [Jimenez et al. 2024, SWE-bench] C. E. Jimenez et al., "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?", ICLR 2024 (arXiv:2310.06770). Verified 2026-11-13.
- [Ong et al. 2025, RouteLLM] I. Ong et al., "RouteLLM: Learning to Route LLMs from Preference Data", ICLR 2025 (arXiv:2406.18665 titles it "with Preference Data"; `routellm` in `docs/paper1/references.bib` now points to the paper, not the repository). Verified 2026-11-13.
- [Ding et al. 2024, Hybrid LLM] D. Ding et al., "Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing", ICLR 2024 (arXiv:2404.14618). Verified 2026-11-13.
- [Chen et al. 2023, FrugalGPT] L. Chen, M. Zaharia, J. Zou, "FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance", TMLR 2024 (arXiv:2305.05176, 2023). Verified 2026-11-13.
- [Weng et al. 2023, Self-Verification] Y. Weng et al., "Large Language Models are Better Reasoners with Self-Verification", Findings of EMNLP 2023, doi:10.18653/v1/2023.findings-emnlp.167. Verified 2026-11-13.
- [Manakul et al. 2023, SelfCheckGPT] P. Manakul, A. Liusie, M. J. F. Gales, "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models", EMNLP 2023, doi:10.18653/v1/2023.emnlp-main.557. Verified 2026-11-13.
- [Shi et al. 2026, EdgeWise] Z. Shi et al., "EdgeWise: Bringing Long-Horizon Agents to the Edge via Selective Cloud Delegation", arXiv:2610.17560, 2026 (v1 2026-10-13). Verified 2026-11-13.
- [Huang et al. 2026, EC2-Bench] P. Huang et al., "Empowering Edge Agents: A Systematic Analysis of Edge-Cloud Collaboration in LLM Agent Execution", arXiv:2610.17816, 2026 (v1 2026-10-13). Verified 2026-11-13.
- [Narayan et al. 2025, Minions] A. Narayan et al., "Minions: Cost-efficient Collaboration Between On-device and Cloud Language Models", ICML 2025 (arXiv:2502.15964). Verified 2026-11-13.
- [Tran et al. 2025, Arch-Router] C. Tran, S. Paracha, A. Hafeez, S. Chen, "Arch-Router: Aligning LLM Routing with Human Preferences", arXiv:2506.16655, 2025. Verified 2026-11-13.
- [llmfit] A. Jones and contributors, llmfit, GitHub repository https://github.com/AlexsJones/llmfit (created 2026-02-15; v1.1.20, 2026-11-03). Verified 2026-11-13.
- [Ollama issue #14259] "Chat history and embedding truncation happens silently with no user-visible indication", ollama/ollama issue #14259, opened 2026-02-14, open as of 2026-11-13 (label "documentation"; warning PRs #14262 and #17145 open, not merged). States the only log line is at `slog.Debug` level in server/prompt.go; a 2026-09-12 comment reports silent truncation on 0.34.0 (Linux, ROCm). Verified 2026-11-13.
- [Ollama issue #3839] "Feature Request: Detect Truncation Due to Exceeding Context Size", ollama/ollama issue #3839, opened 2024-04-23, open as of 2026-11-13 (label "feature request"). Verified 2026-11-13.
- [Desbiens 2026, ContextPaw] Y. Desbiens, "Your LLM server lies to you when the context window overflows", blog post, 2026-07-11, https://yandesbiens.com/blog/contextpaw-silent-truncation/, code at https://github.com/Linutesto/contextpaw. Ollama: HTTP 200, prompt_eval_count 16387 of a 160,689-token prompt at num_ctx 32768, confabulated answer; llama-server at n_ctx 4096: exceed_context_size_error for a 60,001-token request (the post calls it the same prompt, but the token counts differ); marker-only compaction 0/3 facts recovered. No repeated runs or rates, one GPU (RTX 4090), no memory analysis. Verified 2026-11-13.
- [Kwon et al. 2023, vLLM] W. Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023, pp. 611-626, doi:10.1145/3600006.3613165. Verified 2026-11-13.
- [Laskaridis et al. 2024, MELTing Point] S. Laskaridis, K. Katevas, L. Minto, H. Haddadi, "MELTing Point: Mobile Evaluation of Language Transformers", MobiCom 2024, pp. 890-907, doi:10.1145/3636534.3690668. Verified 2026-11-13.
- [Song et al. 2024, PowerInfer] Y. Song, Z. Mi, H. Xie, H. Chen, "PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU", SOSP 2024, pp. 590-606, doi:10.1145/3694715.3695964. Verified 2026-11-13.
- [Alizadeh et al. 2024, LLM in a Flash] K. Alizadeh et al., "LLM in a Flash: Efficient Large Language Model Inference with Limited Memory", ACL 2024, pp. 12562-12584, doi:10.18653/v1/2024.acl-long.678 (bib key `edge_llm_serving`, whose author field was wrong and is fixed). Verified 2026-11-13.
