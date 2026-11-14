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
the loss; llama-server at the same window fails loudly, with HTTP errors <!-- xruntime-llamacpp-wrong-rate -->.

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
   stacks reject it or, provisionally, also run silently past the window <!-- pkg1109-contract1 -->.
3. **The consequences for agents, with their mechanism** (Section 4.6). History loss is silent in every model tested,
   8B to 70B <!-- H-scale-verdict --> <!-- H-scale-family-verdict -->; on two task workloads a 4096-token window more
   than triples wrong task end states, all silent on Ollama; at 4096 a context shift and a token cut discard the
   system prompt <!-- R2-mechanism-lowlevel -->.
4. **Pre-registered fixes, with their limits** (Sections 4.7 and 4.7p). Client-side trimming is refuted on two vendors
   <!-- R2-mitigation-v1-verdict --> <!-- t2s-r2-mitigation-cpu -->; `num_keep` does not remove the window effect
   <!-- x2-evalset-tasks-real-v2-gated-excess -->; a cloud router is refuted on the AMD platform <!-- b5-verdict -->;
   and the runtime's context-shift rule matches every observed shift (misses 0 <!-- R2-shift-predictor-confusion -->)
   but does not predict wrong answers.

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

- **Table 1.** Platforms and versions: CPU and GPU, memory architecture, GPU driver, llama.cpp build for direct runs,
  Ollama version and the GPU backend Ollama actually used, OS build, per platform. Configuration record, not a measured
  result; the values come from the platform configuration record and are pasted when its register row lands.

### 4.4 Memory-architecture boundaries

**Claim.** Running out of GPU memory has a different shape on each architecture (Figure FH1): the AMD APU keeps
running past the boundary at almost unchanged speed, the Intel mini PC's iGPU starts or refuses at one shared-heap
limit, and the Intel laptop serves until the host pages, silently. On the AMD APU and the Intel laptop the boundary did
not follow the memory figure the platform reports.

**AMD APU: a soft spill, then a hard crash far beyond it.** We ran llama-3.3-70b and qwen3-32b in llama-server with the
Vulkan backend and every layer on the GPU (`-ngl 99` <!-- fh1-amd-boundary-llama-3.3-70b -->), stepping the window and
recording per-process GPU memory, decode rate and time to first token (TTFT). Candidate boundaries are a firmware-reserved dedicated block of 65536 MiB, a Vulkan device-local heap of 76218 MiB
and a Vulkan total of 114327 MiB <!-- fh1-amd-boundary-llama-3.3-70b -->. A point spills when shared GPU usage grows by
more than 1024 MiB over the smallest window <!-- fh1-amd-boundary-llama-3.3-70b -->. For llama-3.3-70b the last window
without spill is 65536 and the first with spill is 100096, served for a request of 100000
<!-- fh1-amd-boundary-llama-3.3-70b -->. There the process held 1.122 of the dedicated block but only 0.965 of the
device-local heap <!-- fh1-amd-boundary-llama-3.3-70b -->: the onset follows the dedicated block, not the heap the
driver advertises, so FH1 normalizes the AMD axis by the dedicated block. With one 1855-token prompt at every window
<!-- fh1-amd-boundary-llama-3.3-70b -->, crossing the onset is nearly free: decode ratio 0.983 and TTFT ratio 1.004
<!-- fh1-amd-boundary-llama-3.3-70b -->; decode fell from 5.284 +/- 0.004 tok/s at 65536 to 5.196 +/- 0.010 tok/s at
100096, and at 131072, with 18540 MiB more in shared memory than at the smallest window, it was still 5.176 +/- 0.015
tok/s <!-- fh1-amd-boundary-llama-3.3-70b -->, and nothing in the response reports the spill. Requests above the
trained context of 131072 <!-- fh1-amd-boundary-llama-3.3-70b --> are clamped to it; the last that started was 221440
and the first that failed 221696, exit code 3221226505 <!-- fh1-amd-boundary-llama-3.3-70b -->. For qwen3-32b no unclamped spill point was measured
<!-- fh1-amd-boundary-qwen3-32b -->: a dedicated sweep (llama-server b10970, Vulkan, `-ngl 99`, 1 warm-up + 5 calls per
point) from 20480 to 40960 found shared GPU memory growing by at most 20 MiB and decode between 11.271 and 11.368 tok/s
<!-- fh1-amd-qwen3-32b-spill-sweep -->, so its onset is "above 40960" for unclamped windows.

**Intel mini PC iGPU: a hard wall that first-principles accounting predicts.** On the mini PC's iGPU there is no soft
regime: a model either starts or the runtime refuses it. Across five models the refusal point is consistent with one
shared-heap limit L in (48112.7, 48124.0] MiB, 247 MiB above the 47866 MiB budget the driver reports, from 92 lower
and 22 upper bounds <!-- A-24-effective-heap-limit -->. The limit counts host-visible buffers as well as device memory:
device-only accounting is inconsistent, because the largest start (47945.2 MiB) exceeds the smallest refusal (47562.5
MiB) <!-- A-24-effective-heap-limit -->. Counting weights, KV cache and compute buffers against the shared heap, with L
fitted leave-one-model-out, predicts each model's flip point within -257 to -26 tokens (-0.11% to -0.01%), against
-2890 to -820 tokens for model file size alone and -1092 to +968 tokens for device-only accounting
<!-- A-24-flip-point-prediction -->, across flip points from (23296, 23552] tokens for llama-3.3-70b to (115712, 115968]
for qwen3-32b <!-- A-24-flip-point-per-model --> (Figure F2). These are start tests on one driver and boot state; for
several models the flip point lies beyond the trained context, so it is a memory boundary, not a usable window.

**Intel laptop iGPU: a silent paging onset that accounting does not predict.** On the laptop we served Qwen3-4B through
OpenVINO on the iGPU. A prompt length is a *paging onset* when the first turn completes but host free memory reaches a
fixed floor and prefill runs well above its cold-cache time; we located it by bisection for 16-, 8- and 4-bit KV
caches (f16, i8, u4), 23 points each <!-- pkg1109-x1 -->. What held: the onsets ordered f16 < i8 < u4, in brackets
(35,500, 36,000], (44,000, 45,000] and (48,000, 59,000] tokens; free memory fell below the floor at or below the onset
in 3 of 3 cells; no paging point raised an error; and the allocation per token matched the accounting of KV cache and
activations, 234,957, 171,615 and 134,760 bytes per token <!-- pkg1109-x1 -->. What missed: all three onset
predictions, and the formula dividing the free memory reported after load by that allocation, which put the onset in
no bracket (0 of 3) <!-- pkg1109-x1 -->. The accounting that predicts the mini PC's flip point thus fails here: the
allocation is right, the memory it is divided into is not. Lowering the iGPU's share of memory to 50% did not lower
the onset (i8 (45,000, 46,000], u4 (57,000, 58,000]; "the share does not matter" HIT and "the share moves the onset"
MISS in both cells) <!-- pkg1109-w3-50 -->; cells at a further share are not yet scored. An exploratory fit over 277
load probes (140 in the fit) finds a reserve between the free memory reported after load and what a prompt can use:
median 1,238 MiB (IQR 1,085 to 1,506; n = 99), 793 to 2,264 MiB by model; subtracting it cut the static predictor's
error 2-11x <!-- pkg1109-os-reserve -->, but it differs by model. A sealed predictor of the largest prompt that holds,
calibrated per model with a 16-bit cache (Qwen3-4B-Instruct-2507 22,500, Qwen3-1.7B 50,000, Phi-4-mini 30,000 and
Qwen3-8B 21,000 tokens, the last two flagged for missing edge repeats) <!-- pkg1109-e1 -->, then applied to other cache
precisions and to a pinned memory ballast, hit 5 of 12 cells, missed 6 and left 1 unscored <!-- pkg1109-e2 -->. A
static memory model does not yet predict the onset on this machine.

The onset also moves with a runtime setting. With chunked prefill the i8 onset moved from (44,000, 45,000]
<!-- pkg1109-x1 --> to about 79,000 to 80,000 tokens (79,000 paged 2/4 and 80,000 paged 3/7 repeats), and u4 held to
80,000, censored there (paged 0/1) <!-- pkg1109-c256 -->. That it would move was predicted; where it landed was not (the
i8 cell, predicted censored at 80,000, missed), and three prefill-time and free-memory predictions also missed
<!-- pkg1109-c256 -->. Paging, not prompt length, causes the break: on the 64 GB Intel mini PC under OpenVINO, prompts
of 63,000, 86,000 and 97,000 tokens all held, with 94.3, 183.0 and 238.9 s of first-turn prefill, at least 39,262 MB
free and prefill ratios of 1.94 and 2.53 against predicted 1.83 and 2.32 (inside +-10%), at 171,597 bytes per token,
while the laptop took 327, 1,102 and 2,913 s for the same sizes <!-- pkg1109-t1 -->.

**NVIDIA discrete GPU: placement measured, cost pending.** With 8188 MiB of VRAM <!-- blade-k1-layer-split -->, Ollama
keeps llama3.1:8b fully on the GPU at windows of 4096 and 8192 and moves layers to the CPU above them, with VRAM shares
of 0.870 at 16384 and 0.656 at 32768; for qwen3:8b the shares are 0.805 and 0.592 <!-- blade-k1-layer-split -->. The
slowdown against the spilled share is not yet in our verified results.

**Figure specs.**

- **FH1 (headline).** Cost of crossing each platform's memory boundary. x: memory in use over the boundary (AMD: the
  65536 MiB dedicated block; Intel mini PC: the shared-heap limit L; NVIDIA: VRAM). (a) decode and TTFT ratio to the
  smallest window, labelled with the served window (requested -> served when clamped); (b) start or refuse per model;
  (c) placement only. Each AMD point is one call per fresh server start without a warm-up call (5 or 3 calls per
  point), a departure from this paper's 1 warm-up + N convention.
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
<!-- t2s-r2-real-cpu -->. On the iGPU they loaded 32768, crossed
it only at turn 23 (llama3.1:8b) and turns 20 and 21 (qwen3:14b), and no failure came after the window: 0/5 for
llama3.1:8b and, for qwen3:14b, 1/5 of baseline type <!-- t2s-r2-real-igpu -->. The pre-registered kill criterion was
not triggered on either backend <!-- t2s-r2-real-cpu --> <!-- t2s-r2-real-igpu -->. This contrast changes backend and
window together; Section 4.6a isolates the window.

**On Ollama, overflow is silent.** When a prompt exceeds the loaded window, Ollama returns success and processes
num_ctx/2 + 2 tokens in all 4 checked cases (4096->2050, 8192->4098, 32768->16386, 40960->20482), through its own
`--context-shift --keep 4` server flag <!-- ollama-overflow-keeps-half -->; the discrete GPU behaved the same (Table 2).

**The overflow contract is a property of the serving stack.** For each stack we pre-registered what an over-length
prompt returns and probed seven prompt lengths, three repeats each, under the stack's own defaults. On the Intel
laptop, Ollama matched on 7 of 7 probes: the first fit, and the six above the window returned success after
evaluating 2,050 tokens <!-- pkg1109-contract1 -->. llama-server with SYCL auto-fitted a slot of 37,376 tokens and
rejected prompts over it, matching on every probe it reached (one only conditionally on that slot; the third repeat
stopped after four probes); with Vulkan it reached only three probes in one repeat <!-- pkg1109-contract1 -->. OpenVINO
matched on the first four probes and missed on the last three, where we had predicted a loud out-of-memory failure: it
returned OK, ran silently past the window, and returned OK, 3/3 each, a result we hold as provisional; its
past-the-window mode ran silently past the window on all 12 of 20 sealed repeats run so far <!-- pkg1109-contract1 -->.
LM Studio missed the other way: predicted silent on probes 3 to 7, it rejected each with HTTP 400, 3/3 per probe
<!-- pkg1109-contract1 -->. On the AMD APU with 96 GB assigned to the GPU, Qwen3-4B (Q4_K_M) at a 40,960-token
window, Ollama answered OK through 33,024 tokens and truncated silently at 41,216 and 45,056 (3/3 each), and
llama-server on ROCm answered OK through 33,024 and rejected at 41,216 and 45,056 (3/3 each)
<!-- pkg1109-evo-x2-split96 -->. The Vulkan result at 96 GB is confounded: the GPU vendor's installer silently installed
a new driver during that step, so we report no Vulkan outcome; it is being rerun <!-- pkg1109-evo-x2-split96 -->.
Silence on overflow is therefore not unique to Ollama (OpenVINO, provisionally), and loud stacks answer with an HTTP
error.

**Changing the window, and what persists.** On the AMD APU, a per-request `num_ctx` and `OLLAMA_CONTEXT_LENGTH` both
applied but reverted to the default of 131072 after a restart, while a Modelfile `PARAMETER num_ctx` on a derived tag
persisted <!-- window-override-methods -->; the other platforms have not run this check
<!-- window-override-methods -->.

**Within the window, KV precision is the lever.** We pre-registered a comparison of eight context policies on the
Pareto frontier of rules, history, time and cost (Section 4.7). On the AMD APU every policy kept every rule, and the
frontier held one KV-cache policy per model <!-- b5-policy-metrics -->: for llama3.1:8b a q4_0 cache took 6.6 min per
session at history 100.0% (24/24), against 43.7 min at 95.8% (23/24) for the default; for qwen3:14b a q8_0 cache took
75.4 min against 214.8 min, at the same history recall of 62.5% (15/24) <!-- b5-policy-metrics -->
<!-- paper-percentage-counts -->. Our quantized-cache policies also turn on flash attention
<!-- design: harness/policy_compare.py -->; a control with flash attention on and an f16 cache took 43.4 min for
llama3.1:8b and 206.6 min for qwen3:14b, within tolerance of the default, so the speedup is KV precision
<!-- b5-fa-control -->. A rerun of the f16 recall check for llama3.1:8b found 11/24 needles at f16 against 13/24 at q8_0
(Fisher p = 0.773) and 14/24 at q4_0 (p = 0.564), no difference shown, with needles within a call not independent
<!-- b5-kv-recall-f16-rerun -->. With the window capped at 8192 and 4096 on the same platform (a simulated constraint),
the router (policy d) was on the frontier in all four capped cells and was the only policy that kept history: 80.0% and
62.5% at 8192 and 97.5% and 62.5% at 4096 for llama3.1:8b and qwen3:14b, against at most 12.5% for every other policy
<!-- b5-sim-constraint -->. The verdict is still *refuted*, because client trimming was also on the frontier in every
capped cell <!-- b5-sim-constraint -->; the constrained platforms have not run this comparison <!-- b5-verdict -->.

**What the KV setting loads, and whether output changes.** For qwen3:14b at 40960, the server log reported the K and
V types each setting names, the default's `/api/ps` size equals f16 with flash attention to the byte, and the size
differences give KV buffers of 16, 8.5 and 4.5 bits per element <!-- kv-precision-evidence -->. Under greedy decoding,
on turns with identical first-call prompts, f16 with and without flash attention matched on 120/120 turns; f16 and q8_0
on 103/105 (98.1%), tool calls agreeing on 104/105 (99.0%), first diverging at turn 30 and 35 in two seeds and never in
the third; q4_0 diverged at turn 1 or 2 in every seed, leaving 5 of 120 turns comparable, 2/5 matching
<!-- kv-output-agreement -->. A llama-server follow-up with flash attention off has not run
<!-- pending row: kv-fa-interaction-llamacpp, no number cited -->.

**Long-context quality under a 4-bit KV cache is non-inferior.** A sealed comparison on the Intel mini PC's iGPU under
OpenVINO, Qwen3-4B-Instruct-2507 with greedy decoding, scored RULER items at three context lengths with a 4-bit KV
cache against a reference cache (16-bit, or 8-bit at the length where 16-bit does not fit)
<!-- design: sealed prediction file; reference f16 at the two shorter lengths, i8 at the longest -->. Over 270 paired
items the reference-minus-4-bit difference was -2.2 points, 95% CI [-5.0, +0.6]: the upper bound is below the
pre-registered 5-point margin, so the 4-bit cache is non-inferior (HIT) <!-- pkg1109-qual1 -->. 8-bit against 16-bit
gave -0.6 points (95% CI -4.3 to +3.2, n = 180). With the pilot items added, a sensitivity analysis and not the primary
test, the differences are -2.6 points (-4.6 to -0.6, n = 540) for 4-bit and -0.8 points (-3.4 to +1.7, n = 360) for
8-bit <!-- pkg1109-qual1 -->. On multi-turn function calling (BFCL) the 8-bit and 4-bit caches scored 16% and 13% over
n = 100 paired items, +3.0 points (95% CI -5.9 to +12.0); that pre-registered test missed for lack of power, and we draw
no conclusion about tool use under a 4-bit cache <!-- pkg1109-qual1 -->.

**Figure spec.**

- **F5.** Default window per platform and model over the CDF of agent step lengths in public traces
  <!-- T2S-vs-X2-default-ctx --> <!-- blade-k1-layer-split --> <!-- trace-fraction-exceeds-4096 -->.

### 4.6 Agent consequences

#### 4.6a The window result

**The controlled result.** On the AMD APU we ran three models at the Ollama default and at 32768, 16384, 8192 and 4096
<!-- R2-v1b-excess-over-default -->, five seeds per cell: 75 sessions and 3000 turns <!-- R2-real-v1b-first-events -->,
none surfacing an error before its first failure <!-- R2-real-v1b-first-events -->. After-window silent failures rise
as the window shrinks: 0/15 at 32768 <!-- R2-v1b-excess-over-default -->, 2/15 at 16384 (excess +0.133, 95% CI +0.000
to +0.267) <!-- R2-v1b-excess-over-default -->, 8/15 at 8192 (+0.533, 95% CI +0.333 to +0.667)
<!-- R2-v1b-excess-over-default --> and 10/15 at 4096 (+0.667, 95% CI +0.533 to +0.800)
<!-- R2-v1b-excess-over-default -->, over a default tier with 0/15 <!-- R2-v1b-excess-over-default -->; the CI excludes
zero at 8192 and 4096 <!-- R2-v1b-excess-over-default --> (Figure F6). Rule failures, as distinct from recall misses,
appear only at 8192 and 4096 <!-- R2-real-v1b-first-events -->, and at 4096 no session of any model kept its history to
the first canary turn (intact 0/0/0/0/0/0/0/0 of 5 <!-- R2-real-v1b-survival --> for each model). The default tier's
gated count is 3/15 <!-- R2-v1b-default-baseline-type -->, all llama3.1:8b recall misses inside the window; the
pre-registered kill criterion, counting every metric, finds 10/15 <!-- R2-real-v1b-kill-criterion --> silent sessions
there, all baseline-type: not killed (killed=False <!-- R2-real-v1b-kill-criterion -->), but none is context loss.

**Models differ** (Table R2-gated). llama3.1:8b already fails at 16384, 2/5 <!-- R2-v1b-gated-per-model -->, and 5/5
at 4096, excess +1.000 <!-- R2-v1b-excess-over-default --> (one-sided Fisher exact, 5/5 vs 0/5 sessions, p = 0.00397
<!-- session-exact-tests -->); extended to 10 seeds, 10/10 at 4096 against 0/10 at the default (p = 5.41e-06
<!-- session-exact-tests -->), +0.500 (95% CI +0.200 to +0.800) at 8192 and none shown at 16384
<!-- R2-v1b-excess-10seed -->. qwen3:14b has no after-window failure above 4096 <!-- R2-v1b-gated-per-model -->, and its
4096 sessions lapse on rule 5 or rule 3 at turns 5 to 30 <!-- R2-qwen3-14b-4096-reconciliation -->, while rule 2 held
on 40/40 turns in every seed <!-- R2-qwen3-14b-4096-reconciliation -->. qwen3:8b reversed the ordering at five seeds; a
pre-registered extension to 10 seeds per tier failed silently in 6/10 sessions at 8192 and 2/10 at 4096 (Fisher exact
two-sided p = 0.170), which the pre-registered rule reads as noise; the excess is +0.600 (95% CI +0.300 to +0.900) at
8192 but +0.200 (95% CI +0.000 to +0.500) at 4096, where none is shown <!-- R2-qwen3-8b-reversal-10seed -->. We do not
claim that 4096 is safe for qwen3:8b.

**Protocol check: tools on every call, on two runtimes.** A pre-registered rerun offers tools on every call, on a task
workload (B3), five seeds per tier plus a `num_keep` tier (Section 4.7). It rests on qwen3:14b alone: llama3.1:8b and
qwen3:8b failed the task validation <!-- x2-evalset-val-v2-gates --> <!-- x2-evalset-tasks-val-v2-qwen3-8b-gates -->,
so their real runs refused to start. For qwen3:14b the effect held at 4096 as pre-registered: 5/5
<!-- x2-evalset-tasks-real-v2-gated-excess --> against 0/5 at the default, excess +1.000
<!-- x2-evalset-tasks-real-v2-gated-excess -->, one-sided Fisher exact p = 0.00397 <!-- session-exact-tests -->. Wrong
task end states were 11.3% (17/150) at the default, 12.7% (19/150) at 16384, 35.3% (53/150) at 8192 and 42.0%
(63/150) at 4096, every wrong turn silent on Ollama 0.34.4: excess +24.0 points (95% CI +18.7 to +29.3) at 8192 and
+30.7 points (95% CI +26.0 to +34.7) at 4096, none shown at 16384 <!-- x2-evalset-tasks-real-v2-wrong-rate -->
<!-- x2-evalset-tasks-real-v2-ci-units: these CIs are the session-level bootstrap (10000 resamples, sessions resampled within tier) -->
(session-level 95% CIs, 5 vs 5 sessions). By a deviation declared before the data, qwen3:14b ran although it reached
the task end state on only 80/90 <!-- x2-evalset-val-v2-gates --> validation turns, so wrong rates include baseline
errors. On a stock llama-server, five sessions per arm, the default loaded 40960; wrong task turns were 8.0%
(12/150) at the default and at 40960, and 100.0% (150/150) at 4096, +92.0 points (95% CI +88.7 to +94.7, session
bootstrap) <!-- xruntime-llamacpp-wrong-rate -->. At 4096 no wrong turn was silent (no HTTP error on any call): by the
server's log, 60 were rejected as over the window, 3 stopped at the context limit and 87 ended in another error, and
3.3% of task turns completed a tool round, against 97.3% at the default; 12 wrong turns at the default were silent
<!-- xruntime-llamacpp-wrong-rate -->. With the window fixed at 40960, llama-server matches Ollama's default-window
rate, -3.3 points (95% CI -6.7 to +0.0, 5 vs 5 sessions), with 0/150 context-event turns
<!-- xruntime-default-fixed -->. The window effect replicates across the two runtimes; at 4096 llama-server was loud
where Ollama was silent, though silence is not Ollama's alone (Section 4.5).

**A second task workload, at the Intel default.** A separate pre-registered task workload, with its own harness and
validation, ran qwen3:8b on the Intel mini PC under Ollama 0.40.14, 8 seeds per arm. At the default the runtime loaded
4096 tokens and 193/240 task turns (80.4%) ended wrong; with 24576 loaded, 60/240 (25.0%) did; all 193 wrong turns at
the default were silent <!-- pkg1109-t4-8b -->. The paired difference over seeds was +0.554 (95% t interval +0.445 to
+0.664), positive in 8 of 8 seeds (sign test p = 0.0078); an exploratory cluster bootstrap gives +0.467 to +0.633
<!-- pkg1109-t4-8b -->. The sealed recall prediction missed: recall errors were 28/56 (0.50) at the default and 37/56
(0.66) at 24576, the opposite of the predicted direction <!-- pkg1109-t4-8b -->, so the larger window lowered wrong
task turns but not recall errors, and it still left 25.0% of turns wrong. qwen3:8b failed our B3 validation on the AMD
APU (task end state 65/90, 72.2%) <!-- x2-evalset-tasks-val-v2-qwen3-8b-gates -->, so this workload stands beside B3,
not as B3 task accuracy. A further sealed run on the same model, machine and runtime, 8 seeds per arm, put an
off-the-shelf LiteLLM failover router, or LiteLLM's pre-call check against the declared window, in front of the local
model: both arms ended 193/240 (80.4%) of task turns wrong, all silent, with 0 cloud turns and USD 0.0000 spent
<!-- pkg1109-pitch-ll -->; neither escalated a single turn.

**Scope: scale and family.** H-scale pre-registered that rule loss at 4096 and history loss with rules intact at
32768 <!-- H-scale-verdict --> hold from 30B to 70B, three seeds per cell.
qwen3:32b lost rules at 4096 in 2/3 sessions but showed history loss with rules intact in only 1/3 at 32768;
llama3.3:70b lost no rules at 4096 (0/3, every session over the window) and lost history with rules intact in 3/3 at
32768, the pre-registered alternative "resists rule loss"; overall H-scale is not supported (mixed)
<!-- H-scale-verdict -->. mistral-small3.2:24b, from another family, also resisted rule loss (0/3 at 4096, history loss
2/3 at 32768) <!-- H-scale-family-verdict -->. Rule loss thus appears in the 8B and 14B models and qwen3:32b, history
loss in all <!-- H-scale-verdict --> <!-- H-scale-family-verdict -->; we do not claim rule loss in general. These are R2
rule-following results, not task accuracy, which rests on qwen3:14b (B3) and the second workload's qwen3:8b.

**Long tasks, inside the window.** On 20 synthetic long-horizon tasks of our own, step-matched to public agent traces
(10 to SWE-Gym quantiles, 10 to OpenHands traces), qwen3:14b on the AMD APU, three seeds each, completed 5/60 tasks (8.3%, Wilson 95% 3.6% to 18.1%) at the Ollama default (loaded
window 40960) and 4/60 (6.7%, 2.6% to 15.9%) with a q8_0 KV cache, every wrong task silent
<!-- long-task-local-accuracy -->. On long-horizon tasks qwen3:14b and qwen3:32b solved none of 30, and llama3.3:70b 3 of
30 after a pre-registered tool-template fix <!-- long-task-scale-local --> <!-- longtask-70b-toolfix -->; at a window
that held the whole prompt, llama3.3:70b still failed 3 of 3 long tasks sized from trace token counts (their prompts
were about 1.6x the traces <!-- longtask-tokens-v1-sizing -->) on wrong actions, so on these tasks capability, not
memory, fails first <!-- longtask-tokens-v1 -->.

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

**What the runtime removes.** We reran llama3.1:8b on the AMD APU, one session per tier (80 calls each
<!-- R2-mechanism-lowlevel -->), with Ollama in debug mode, its log sliced per call, and a render-only copy of each
request; the rendered prompt matched the runtime's count on every check (9/9 within 1% at 4096; render/log
disagreements 0 <!-- R2-mechanism-verdict -->). Three mechanisms appear (Figure F7). First, Ollama drops whole messages
from the front once the conversation outgrows the window, on 54/80 calls at 16384, 68/80 at 8192 and 76/80 at 4096
<!-- R2-mechanism-lowlevel -->, always keeping the system message ("system kept on all True"
<!-- R2-mechanism-verdict -->) and usually cutting mid-turn: history goes, the rules stay in view. At 4096
<!-- R2-mechanism-lowlevel --> two lower-level steps then remove the system prompt. llama.cpp's context shift fired on 34
calls from turn 4, each keeping n_keep 5 tokens and discarding the next 2045, 29 of them on the final-answer call; and
once Ollama's own token cut reduced a prompt of 4097 tokens to 2050, keeping 4 <!-- R2-mechanism-lowlevel -->, the
num_ctx/2 + 2 overflow rule (4096->2050 <!-- ollama-overflow-keeps-half -->). No step raised an error (exceed-context
errors 0 <!-- R2-mechanism-verdict -->). The hypothesis "drop old turns, keep the system message" is therefore refuted
by the 4096 token cut (verdict refuted <!-- R2-mechanism-verdict -->) and confirmed at 32768, 16384 and 8192.

**NVIDIA replicates the first two steps.** One session at 4096 on the NVIDIA laptop passed the same render check (9/9
within 1%); Ollama trimmed whole messages on 75/80 calls from turn 3 and kept the system message on every one, and the
context shift fired 33 times; no token cut occurred and no error surfaced <!-- blade-mechanism-4096 -->. That row lacks
n_keep, so the claim that the shift discards the system prompt is AMD-only.

**A second silent difference on Ollama: tool calls as text.** In the NVIDIA validation, llama3.1:8b's negative control
at 32768, where nothing is truncated, missed canaries in two of five sessions (seeds 20260903 and 20260905, 2 misses
each <!-- blade-r2-gate-v2 -->): the final answer was a tool call written as text. Three pre-registered AMD checks each
moved one factor toward the NVIDIA configuration. At 32768 on
ROCm there were 0 of 20 misses ("backend or offload effect" <!-- x2-r2-llama-val-32k -->). Forcing NVIDIA's CPU offload
split, 22 of 33 layers on the GPU with VRAM share 0.656, again gave 0 of 20 ("backend effect"
<!-- x2-r2-llama-val-32k-offload -->). Switching only the backend, to Vulkan on the same GPU and verified in 5/5
sessions, gave 2 of 20 misses, at seed 20260903, turn 5, as a text call to `lookup_fact`: "backend implementation, not
the vendor" <!-- x2-r2-llama-val-32k-vulkan -->. The failure follows the llama.cpp backend (CUDA and Vulkan, not ROCm,
in these seeds), not the GPU vendor, but on Vulkan that rests on one session, and the kernel is not identified.

**Figure spec.**

- **F7.** Per-turn events per tier (message drops, context shifts, token cut), llama3.1:8b, one session per tier, with
  the NVIDIA 4096 session as an extra row <!-- R2-mechanism-lowlevel --> <!-- blade-mechanism-4096 -->.

### 4.7 Fixes

**Client-side trimming splits by model, on two vendors.** Our pre-registered client fix counts each request with the
model's tokenizer and drops whole old turns until the system prompt, the kept turns and the answer budget fit the window
with a 5% <!-- design: harness/x2_r2_client_trim.py --> margin. A model
*recovers* when every rule in use is at or above 90% <!-- design: analysis/r2_mitigation_report.py -->, and the
hypothesis needed both models to recover at 4096 <!-- R2-mitigation-v1-verdict -->. Mechanically it worked: 0 context
shifts, 0 token cuts, 0 Ollama message drops and 0 exceed-context errors in every cell <!-- R2-mitigation-v1-observed -->.
The verdict is *refuted* <!-- R2-mitigation-v1-verdict -->: on the AMD APU llama3.1:8b recovered at 8192 and 4096, and
qwen3:14b at 8192 but not at 4096 <!-- R2-mitigation-v1-verdict -->, where its `log_event` rule fell to 61.7% (74/120),
failing first at turn 3 in every seed (3/3/3) <!-- R2-mitigation-v1-4096 --> <!-- paper-percentage-counts -->. On the
Intel mini PC's CPU at 4096 the split replicates: llama3.1:8b recovered, qwen3:14b did not, again on `log_event`
<!-- t2s-r2-mitigation-cpu -->. The cause is the fix itself: untrimmed, qwen3:14b kept rule 2 on 40/40 turns in every
seed, and on the first failing trimmed turn it requested only `lookup_fact`, where the untrimmed run called
`log_event` and then `lookup_fact` <!-- R2-qwen3-14b-4096-reconciliation -->: at 4096 the budget leaves room for only
1-2 <!-- R2-mitigation-v1-observed --> turns besides the system prompt, so trimming removes the examples of the
two-call pattern the model imitates.

**num_keep fixes one model's rules, on the token-cut path.** We pre-registered a test for llama3.1:8b at 4096 with
`num_keep` set to the system prompt's measured length (601, 594 and 597 tokens per seed). All 105 context shifts kept
n_keep 595, 598 or 602, and the rules in use held (rule 1 at 94.2% (113/120), rules 3 and 4 at 100.0% (120/120 and
60/60)), which the pre-registered rule reads as "one-line fix" <!-- num-keep-guardrail-test -->
<!-- paper-percentage-counts -->. It fixes the rules and nothing else: Ollama still dropped messages on 224 of 239 calls
and the history canary was missed on 24 of 24 checks <!-- num-keep-guardrail-test -->. In the protocol check the
`num_keep` tier at 4096 lowered qwen3:14b's wrong-task rate from 42.0% (63/150) to 30.0% (45/150), but its excess over
the default stayed at +18.7 points (95% CI +16.0 to +21.3) <!-- x2-evalset-tasks-real-v2-wrong-rate --> and its gated
excess at +1.000 (5/5 vs 0/5 sessions, p = 0.00397 <!-- session-exact-tests -->), so `num_keep` does not remove it
<!-- x2-evalset-tasks-real-v2-gated-excess -->. What lowered that rate is not established: in the R2 harness the same
flag changed nothing for qwen3:14b at 4096 <!-- R2-qwen3-14b-4096-reconciliation -->.

**A router does not dominate.** The eight <!-- design: harness/policy_compare.py --> policies were the default, the
largest window that starts, client trimming, a cloud router, an auto-sized window, `num_keep`, and q8_0 and q4_0 KV
caches; the hypothesis was that the router is on the Pareto frontier on every platform. It is *refuted*: on the AMD APU the router was dominated for both models <!-- b5-verdict -->. Every
policy kept every rule (300/300 for llama3.1:8b and 540/540 for qwen3:14b <!-- b5-policy-metrics -->). The router took
43.5 min per session for llama3.1:8b at history 95.8% (23/24), and 135.1 min and USD 0.8419 per session for qwen3:14b,
against faster KV-cache policies at the same or higher recall (Section 4.5) <!-- b5-policy-metrics -->
<!-- paper-percentage-counts -->; under the simulated cap it joined the frontier, beside client trimming
<!-- b5-sim-constraint -->.

**Recovering after a silent failure is costly and incomplete.** A cloud model given the whole client-side history late,
on every wrong task of the protocol check, fixed 84.8% (167/197), task-level 95% Wilson CI [79.1%, 89.1%]
<!-- cloud-redo-full-context -->; resampling whole sessions instead (25 sessions, stratified by tier, 10000 resamples)
gives [79.4%, 90.4%] <!-- headline-ci-session-rescue -->, at a mean of USD 0.0227 per task
<!-- cloud-redo-full-context -->. Matched on task and seed, the late redo cost 1.76x a fresh run of that task (mean USD
0.0226 against USD 0.0129 over 119 matched rows) <!-- late-escalation-cost-vs-fresh -->. A plain redo also cannot undo
what the local model did: of the 30 tasks still wrong, 28 needed a repair of changed state and 2 were cloud-model
errors <!-- cloud-redo-full-context -->. Deciding at session start instead, no measured device qualifies at matched
quality: the AMD APU at
its best setting is 7.8 points below the small cloud model and 11.1 below the mid-tier one, and the Intel iGPU,
measured by session pass rate rather than task accuracy, 16.7 and 20.0 points below <!-- hybrid-route-at-start -->.

**Rolling back executed actions before the redo (stand-in).** On simulated tools, for the 28 tasks that needed a
repair, we snapshotted the environment, rolled back every earlier wrong task turn of the session, replaced it by its
reference calls, and had the cloud model redo the task: it fixed 28/28 (95% Wilson CI [87.9%, 100.0%]) against a
pre-registered target of 25/28, at USD 0.8926 billed for the 28 <!-- own-rollback-redo -->. With the full-history
redo, 195/197 wrong local tasks were fixed (99.0%, 95% Wilson CI [96.4%, 99.7%]); the 2 still wrong were the cloud
model's own errors <!-- cloud-rescue-combined -->. Replacing earlier turns by reference calls stands in for a
successful escalation of each, not a live deployment.

**No reliable detector.** Every pre-registered attempt to find or avoid the wrong tasks missed its target. On its
held-out seed at 4096 the tuned post-run check caught 14/14 wrong tasks but escalated 15/16 correct ones, against a
target of at most 10%; its rule-level checks alone caught 6/14 <!-- post-run-check-tuned -->
<!-- post-run-check-l3-rule-breakdown -->. A session-start fit predictor reached precision 0.350 against the runtime log
(MISSED) <!-- fit-predictor-v1 -->, and the full route, check and redo pipeline kept 81/90 tasks local at 94.4%
accuracy, below its 97% target (MISSED) <!-- hybrid-full-pipeline-v1 -->. Our own compaction, which summarizes old
turns, left 16.0% and 23.3% of tasks wrong at 8192 and 4096, against targets of 15% and 20% <!-- own-compaction-8192 -->
<!-- own-compaction-4096 -->, and no route-at-start policy met its target <!-- hybrid-route-at-start-v2 -->. On the
Intel mini PC the strict gate refused qwen3:14b (task end state 80/90 on the CPU, 79/90 on the iGPU)
<!-- t2s-evalset-tasks-cpu --> <!-- t2s-evalset-tasks-igpu -->, so no Intel B3 task accuracy exists; Intel task
accuracy comes from the second workload (Section 4.6a).

#### 4.7p The runtime's shift rule and the window flag, characterized

**The context-shift trigger.** The shift that discards the system prompt (Section 4.6b) has a trigger readable from the
llama.cpp source: the prompt left after the message-level trim, plus the tokens generated, exceeds the loaded window. Scored against the shifts the
server log recorded in the AMD mechanism sessions, it gave TP 34 FP 0 FN 0 TN 46 at 4096 and TP 10 FP 0 FN 0 TN 70 at
32768 (34/34 and 10/10), misses 0 over all tiers <!-- R2-shift-predictor-confusion -->. The generated count is not
known before the call; substituting the generation budget, P+num_predict>n_ctx gives TP 44 FP 22 FN 0
<!-- R2-shift-predictor-confusion -->, missing no shift at the cost of false alarms; it has no tunable parameter but
was scored on the sessions it was built from.

**It predicts the shift, not the failure.** Applied to the real run as a prediction, the rule flags 102/240 calls for
llama3.1:8b at 4096 and 0/240 for qwen3:14b <!-- R2-shift-predicted-real-v1 -->, yet qwen3:14b also loses rules at
4096, through what the message-level trim removes <!-- R2-qwen3-14b-4096-reconciliation -->: at 4096, 80.3% (61/76) of
truncated calls began with an orphaned message <!-- R2-orphan-rates -->.

**Window overflow is not a general wrong-answer detector.** We measured "this task turn's prompt exceeded the loaded
window" as a predictor of "the task ended wrong" on the protocol check (qwen3:14b, AMD APU)
<!-- impact-v3-detection-value-measured -->. At the default no turn overflowed, and the flag missed 17/17 wrong tasks
(recall 0.0% (0/17)). At 16384 it reached precision 15.4% (4/26) and recall 21.1% (4/19); at 8192 precision 45.3%
(48/106) and recall 90.6% (48/53); at 4096 recall 100.0% (63/63) but precision 42.9% (63/147), flagging 147 of 150
turns (TP 63 FP 84) <!-- impact-v3-detection-value-measured -->. With a large window wrong answers have other causes;
with a small one nearly every step overflows. The shift rule and the window flag explain system-prompt loss; they are
not detectors or a cheap escalation filter.

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
<!-- x2-evalset-val-v1-gates -->, so they support no claim here. The B3 workload rests on qwen3:14b alone, under the
deviation of Section 4.6a; qwen3:8b, llama3.1:8b, mistral-small3.2:24b and llama3.3:70b failed its validation
<!-- x2-evalset-tasks-val-v2-qwen3-8b-gates --> <!-- x2-evalset-tasks-real-v2-wrong-rate -->
<!-- x2-evalset-tasks-val-v2-mistral-small32-gates --> <!-- x2-evalset-tasks-val-v2-llama33-70b-gates -->; the second
workload adds qwen3:8b on the Intel mini PC under its own harness and validation.

**Cost figures.** Cloud cost per real agent task is priced from traces, not executed <!-- cloud-cost-per-real-task -->,
and its caching case is an assumption. A first cloud redo was broken (109 of its 118 failures spent the whole
384-token output budget on thinking <!-- cloud-redo-diagnosis -->); we cite nothing from it, and Section 4.7 reports
the corrected redo (2048 output tokens per call) <!-- cloud-redo-full-context -->.

### 4.9 Conclusion

On every platform we measured, Ollama trims, shifts or cuts a conversation that outgrows the window and the request
still succeeds; llama-server at 4096 fails loudly, with the same window effect on wrong tasks (+92.0 points, 95% CI
+88.7 to +94.7) <!-- xruntime-llamacpp-wrong-rate -->. The window is often not the user's choice: an unmodified install on an Intel CPU
ran agents at 4096 tokens <!-- t2s-r2-real-cpu -->, the window at which qwen3:14b on the AMD APU ended 42.0% (63/150) of
tasks wrong against 11.3% (17/150) at its default <!-- x2-evalset-tasks-real-v2-wrong-rate -->, and at which qwen3:8b
on the Intel mini PC ended 80.4% (193/240) of task turns wrong <!-- pkg1109-t4-8b -->, all silently. Memory runs out
silently too, and not where the reported memory figures say (Section 4.4).

Five lessons follow. Set the window explicitly for each device and verify the loaded value
<!-- window-override-methods -->. Surface the runtime's own cut-off log lines, which Ollama writes only at debug level
(`slog.Debug`), invisible at the default log level [Ollama issue #14259], but do not treat them as a correctness check.
Try a quantized KV cache before a smaller window: on the AMD APU it cut session time by 85% (37.1 of 43.7 min) and 65%
(139.4 of 214.8 min, 3 sessions per setting) with no measured loss of rules or recall <!-- cost-cut-config-time -->
<!-- b5-kv-recall-f16-rerun -->. Treat trimming as a change to the prompt, since dropping old turns also drops the
examples the model imitates <!-- R2-qwen3-14b-4096-reconciliation -->. Validate an agent on the backend it will run
on, because identical model files and settings behaved differently on CUDA, Vulkan and ROCm <!-- blade-r2-gate-v2 -->
<!-- x2-r2-llama-val-32k-vulkan --> <!-- x2-r2-llama-val-32k-offload -->.

Pending: the router comparison on the small-default platforms <!-- b5-verdict -->, a verifier for wrong answers
inside the window, a live rollback, the Vulkan rerun at 96 GB, and more models, seeds and runtime versions.

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
