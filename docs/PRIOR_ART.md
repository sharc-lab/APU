# Prior Art — Paper 1

The claim under test is: **task-level correctness as a function of provisioned context memory,
expressed as f(workload type, quality floor) → GB** — the minimum memory needed to meet a
correctness floor for a given class of agentic workload.

Each section below identifies the closest existing work, states what it establishes, and
argues whether the claim survives as stated or must be narrowed. The reviewer's side is argued
first.

References cross-checked against `docs/RELATED_WORK.md` and `docs/references.bib`. Real
arXiv/DOI identifiers are used only where the paper was confirmed to exist in search results.
Unverified papers are noted as such.

---

## a. Positional Waste Gap

**Against: KV-cache eviction** (H2O, SnapKV, Scissorhands, StreamingLLM)
**and context compression** (LLMLingua and successors)

### Closest existing work

- **H2O — Heavy-Hitter Oracle** (Zhang et al., NeurIPS 2023; arXiv:2306.14048): evicts KV
  entries by keeping "heavy hitter" tokens (those with large cumulative attention scores) plus
  a recent window. Reduces KV cache size by up to 20× with modest quality loss on OPT and
  LLaMA.
- **SnapKV** (Li et al., NeurIPS 2024; proceedings.neurips.cc/2024/28ab418...): selects
  clustered important positions from an observation window at the end of the prompt, enabling
  efficient long-context inference.
- **Scissorhands** (Liu et al., 2023; not independently verified in search — cited as
  "persistence of importance hypothesis for KV cache compression at test time"): evicts KV
  entries whose importance does not persist across generation steps.
- **StreamingLLM** (Xiao et al., 2023; not independently verified): retains only the most
  recent tokens plus a small set of sink tokens; enables infinite-length inference at fixed
  memory.
- **LLMLingua** (Jiang et al., EMNLP 2023) and **LongLLMLingua** (Jiang et al., ACL 2024;
  aclanthology.org/2024.acl-long.91): compress the prompt by dropping low-perplexity tokens,
  achieving up to 20× compression with minimal quality degradation on GSM8K and BBH.

### What they establish

All of these works accept the memory budget as fixed and optimize what to keep within it.
H2O and SnapKV select which KV entries to evict; LLMLingua selects which prompt tokens to
drop. The optimization objective is throughput or serving efficiency, not the minimum memory
needed for correct retrieval on a specified task.

### Reviewer argument

A reviewer will argue: "Your result — that the artifact must be present to avoid fabrication
— is a direct corollary of what the KV eviction literature already established. H2O shows
that retaining 20% of KV entries preserves quality; your claim is the same idea expressed in
capacity terms."

### Why the claim survives

The distinction holds in one sentence: **KV eviction optimizes a retention policy at fixed
hardware; our work sizes hardware against a correctness floor.**

More precisely:
1. Eviction policies (H2O, SnapKV, etc.) presuppose that a capable serving system already
   exists and ask what portion of KV can be safely dropped during inference. The hardware
   capacity is the constraint, not the output.
2. Our f(workload, quality floor) → GB inverts the optimization: the quality floor and the
   workload are fixed, and the required GB is the output. This is a procurement-time question,
   not a serving-time question.
3. The existing works measure quality degradation as a function of *fraction retained* at fixed
   total capacity. Our work measures quality degradation as a function of *total available
   capacity* (budget_ratio), which is a different curve and answers a different question:
   not "how much of the KV can I evict and still pass?" but "how much KV do I need to buy to
   pass at all?"

The claim cannot be narrowed away by the eviction literature because that literature does not
produce a GB figure — it produces a retention fraction. Converting a retention fraction to a
GB figure requires the total capacity as input, which is precisely what we are trying to derive.

**Caveat the paper must state:** Our result is established on Qwen3-4B with synthetic filler
and a small probe suite. The curve shape (cliff vs. graceful degradation) and the GB figure
are model-specific and workload-specific. We characterize the curve; we do not claim the
specific GB figure generalizes to other models or workloads without replication.

---

## b. Silent Fabrication Under Truncation

### Closest existing work

- **"Characterizing LLM Abstention Behavior in Science QA with Context Perturbations"**
  (Feng et al., Findings EMNLP 2024; arXiv:2404.12452): studies how perturbations to context
  (noise injection, context replacement) affect whether models abstain or answer. Found that
  Claude models abstain more conservatively than GPT under distractors.
- **"Know Your Limits: A Survey of Abstention in LLMs"** (Tonmoy et al., TACL 2025;
  MIT Press doi:10.1162/tacl_a_00754): broad survey of abstention mechanisms. Focuses on
  model confidence calibration and prompting strategies, not context loss.
- Both works study abstention under *perturbed but present* context (noisy, replaced, or
  contradictory). Neither measures abstention when the answer span is **physically absent**
  from the context window.

### What they establish

The existing work establishes that abstention rates vary with context noise level and model
family. It does not establish what happens to abstention when the relevant information is
simply gone — evicted, truncated, or never provided.

### Reviewer argument

"Fabrication under missing context is a well-known LLM property. The 'lost-in-the-middle'
and general hallucination literatures already establish that models confabulate when context
is insufficient. Your 100% fabrication rate is not a new finding."

### Why the claim survives — and where it must be narrowed

**Survives on specificity:** We measure *abstention rate specifically when context is absent*
(artifact_fraction_retained = 0), not when it is noisy or insufficient. The answer span is
completely gone from the context. The finding — 0% abstention, 100% fabrication, even at
gpt-oss:120B scale — is a claim about *absence* producing confident wrong answers at a rate
the literature does not appear to have measured directly and reported as a function of truncation
ratio.

**Must be narrowed as follows:** The paper cannot claim this is a novel finding in the general
sense (fabrication when context is lost). The specific contribution is: (1) the fabrication
rate at the cliff is total (not partial), (2) the abstention-instruction intervention inverts
rather than improves behavior, and (3) the abstention rate is measured as a *function of
budget_ratio* — i.e., the transition from 0% fabrication to 100% fabrication is sharp, not
gradual. This last point connects to the step-function claim (§c below) and is where the
novelty resides. The fabrication claim alone without the cliff shape is not strong.

---

## c. Step-Function Collapse Rather Than Graceful Degradation

### Closest existing work

- **"GSM-Infinite"** (arXiv:2502.05252; verified in search): tests models on problems of
  increasing complexity/length; finds sigmoid-like performance decay with a sharp cliff past
  a critical threshold.
- **"Context Length Alone Hurts LLM Performance Despite Perfect Retrieval"** (EMNLP 2025;
  aclanthology.org/2025.findings-emnlp.1264): shows that adding context degrades reasoning
  even when the relevant information is correctly retrieved — a form of context dilution.
- **LLMLingua compression cliff** (from search results): accuracy degrades gently up to
  ~10–30× compression, then drops sharply.

### What they establish

GSM-Infinite and similar work establish sigmoid degradation as complexity grows. The LLMLingua
work establishes a compression cliff at extreme compression ratios. Context-dilution work
establishes that longer contexts harm reasoning even at perfect retrieval.

### Reviewer argument

"The step-function collapse you observe is just the sigmoid cliff found by every long-context
evaluation, observed at the particular artifact extinction threshold of your probes."

### Why the claim survives

Our cliff is mechanistically distinct from both the sigmoid complexity cliff and the
compression cliff:

1. **Our cliff is per-probe and artifact-fraction driven, not context-length driven.** We
   observe it as a function of *artifact survival fraction* (Fig 4.3), not total context
   length. The same total context length can score 0.0 or 1.0 depending on which tokens
   were truncated. This rules out a length-driven attention failure as the mechanism.

2. **The cliff position is probe-specific** (0.444–0.761 artifact fraction), meaning a
   single hardware configuration can be above the cliff for one workload category and below
   it for another. The shape is not a global property of the context window; it is a function
   of where the answer span appears in the artifact.

3. **The fabrication / non-abstention behavior at the cliff bottom** (claim §b) does not
   appear in the compression literature, which measures accuracy, not the fabrication/abstention
   split.

**Must be narrowed:** We cannot claim to have characterized the mechanism of the cliff.
The cha_04 ablation in `docs/FINDINGS.md` explicitly disconfirms one proposed mechanism.
The claim must be stated observationally — "we observe step-function collapse at the per-probe
artifact extinction threshold" — without asserting why the cliff is sharp rather than gradual.

---

## d. Position Effects Under Truncation

**Against: Liu et al. "Lost in the Middle" and successors**

### Closest existing work

- **"Lost in the Middle: How Language Models Use Long Contexts"** (Liu et al., TACL 2024;
  arXiv:2307.03172): Multi-document QA performance follows a U-shaped function of relevant
  document position — highest at beginning and end, lowest in the middle. Full context;
  no truncation.
- **"Positional Biases Shift as Inputs Approach Context Window Limits"** (arXiv:2508.07479,
  confirmed in search): position biases change as prompts grow toward context window limits.
- Successor work on long-context position: a large literature on recency/primacy bias in
  attention. Most assumes full context is present.

### What they establish

The existing work establishes that LLM attention is U-shaped with respect to position *when
the full context is present*. Information in the middle receives less attention than information
at the boundaries. The mechanism is architectural (RoPE decay, softmax concentration).

### Reviewer argument

"Your EARLY/LATE result is just the recency and primacy effects from Liu et al.: LATE arm
works because the artifact is at the end (recency), EARLY arm fails at depth because the
artifact is in the middle once filler is added. This is not a new finding."

### Why the claim survives — and where it must be narrowed

**Survives on mechanism:** Our position effect is *truncation-driven*, not attention-driven.
In the LATE arm, the artifact survives truncation because filler precedes it; in the EARLY
arm, the artifact is truncated away first. This is a hardware-level constraint (available GB
determines what fits in the context window), not an attention mechanism. A model with
infinite memory would score EARLY = LATE; a model with limited memory scores differently
because of *which* tokens are physically absent, not because of where attention falls.

The test is unambiguous: in the EARLY arm at truncating ratios, artifact_fraction_retained
approaches 0. The score drop is explained by the artifact being gone, not by the model
failing to attend to it. The stage C data confirms this: at non-truncating ratios (1.00, 1.20),
where the full artifact is present in both arms, the position effect is different from the
truncating-ratio effect.

**Must be narrowed:** At non-truncating ratios (ratio ≥ 1.00), the EARLY/LATE score
difference that remains is attributable to attention effects (lost-in-the-middle), not
truncation. Our claim must distinguish clearly between the attention-position effect (present
but not novel) and the truncation-position effect (the main contribution). The paper should
not conflate the two.

**Naming note:** The term "position pressure" is not standard. Define it explicitly as
"the effect of left-char truncation on which arm loses its artifact first," not as a synonym
for lost-in-the-middle attention effects.

---

## e. MemExplorer (arXiv:2604.16007)

### What MemExplorer establishes

MemExplorer (arXiv:2604.16007, Microsoft Research, April 2026; confirmed in search) is a
hardware design-space exploration framework for agentic inference NPUs. It jointly optimizes
NPU compute configuration and heterogeneous memory technology selection (choosing among HBM,
LPDDR, HBF, 3D-stacked SRAM) to maximize throughput and energy efficiency under a fixed
power budget. Reported results: up to 2.3× better energy efficiency than baseline NPU under
the same power budget for agentic workloads.

### Overlap with our claim

Both MemExplorer and this paper are concerned with how memory affects agentic LLM inference
performance. MemExplorer explicitly motivates its work by noting that "agentic LLM workloads
are driving rapidly growing demand on memory capacity and bandwidth." It sizes memory
technology choices for throughput and power; it does not measure how task-level correctness
varies with context memory capacity.

### Two-sentence explicit distinction

MemExplorer co-designs NPU compute and memory technology hierarchy to maximize throughput
and energy efficiency under a fixed power budget, treating task correctness as a property of
the model weights that is independent of the memory configuration. Our work measures how
task-level correctness varies with provisioned context memory capacity — specifically, the
minimum GB needed to preserve a quality floor on defined workloads — which is the input
constraint that MemExplorer would need to set KV cache capacity requirements at design time.

### Can this distinction be stated in one sentence?

Yes: **MemExplorer sizes memory technology to maximize throughput at fixed correctness;
our work sizes memory capacity to preserve correctness at fixed quality floor.**

### Reviewer threat

MemExplorer is the closest hardware-side relative and was published 5 months before this
writing. A reviewer may argue that our contribution is the empirical measurement that
MemExplorer takes as an assumption. This is a fair characterization — and it is exactly
the contribution we are claiming: MemExplorer needs the correctness-vs-capacity curve as
an input to its design space; we provide the measurement methodology to produce it. The
papers are complementary, not competing.

**The paper should cite MemExplorer and state explicitly that it provides the empirical
characterization methodology that hardware design-space tools like MemExplorer treat as
an input.**

---

# Prior Art Check -- Memory-Dependent Runtime Behavior Claims (a)-(d)

Scope note: this section is a separate prior-art pass, done for four specific claims about
local LLM runtime memory behavior (default context sizing, co-running memory pressure,
cross-vendor OOM handling, fabrication vs. refusal under truncation). It is distinct from the
"Paper 1" positional/fabrication claims reviewed above, though claim (d) below overlaps with
claim (b) above (Silent Fabrication Under Truncation) and reuses some of the same sources.
Search performed across arXiv, ACM DL, IEEE Xplore, USENIX/MLSys proceedings, technical blogs,
and the llama.cpp / Ollama GitHub issue trackers, September 2026.

## Claim (a): Memory configuration silently changes the default context window, degrading agent/instruction-following quality once real usage exceeds the silently-chosen ceiling

**Restatement:** VRAM/unified-memory amount (or a BIOS/UEFI memory setting, or a device's
memory "tier") changes the default context window a runtime auto-selects, and this causes
agent/instruction-following quality loss once usage exceeds that silently-chosen ceiling, not
just a throughput hit.

**Verdict: PARTIALLY SHOWN**

- The mechanism half of this claim -- that a runtime changes its default context window based on
  detected VRAM -- is real, documented, and already causing user-facing problems, but this is
  engineering/issue-tracker evidence, not a paper.
  - Ollama shipped exactly this behavior in v0.15.5: tiers of 4,096 tokens below 24 GiB VRAM,
    32,768 tokens from 24-48 GiB, 262,144 tokens at 48 GiB+
    (`ollama/docs/context-length.mdx`, github.com/ollama/ollama/blob/main/docs/context-length.mdx).
  - **ollama/ollama#14073** ("New default context lengths will break," opened 2026-02-04):
    a user with 52 GiB VRAM reports the new tiered default (262,144 tokens) overflows their
    VRAM and Ollama becomes unresponsive as it spills to CPU, exactly the "silently-chosen
    ceiling causes a real-usage failure" shape of this claim, though the reported failure mode
    there is a responsiveness collapse, not a measured instruction-following score drop.
  - **ollama/ollama#14116** ("Tiered context length can exhaust VRAM"): documents that
    `OLLAMA_NUM_PARALLEL` is not accounted for in the tier calculation, so total VRAM used is
    `num_ctx * num_parallel`, causing spill even when the single-context tier looked safe.
  - **ollama/ollama#12353** ("Feature Request: Auto-size num_ctx to a user VRAM budget"):
    users are explicitly asking for a `--fit-vram` flag because they must currently guess a safe
    `num_ctx`, and an over-guess causes VRAM overflow "with no clear feedback."
  - llama.cpp shows the same failure shape from a different direction: **lmstudio-ai/lmstudio-bug-tracker#2404**
    reports context length silently clamped to ~6,656 tokens on a 24 GB unified-memory Apple
    Silicon machine "regardless of model size or available memory," with no error surfaced.
  - vLLM's `--max-model-len auto` explicitly ties context length to free GPU memory at load
    time (docs.vllm.ai/en/stable/configuration/engine_args), confirming the general pattern
    across a third runtime.
- The quality-degradation half is separately well established in the general "context rot" /
  lost-in-the-middle literature, but not tied back to *runtime-auto-selected* ceilings:
  - Chroma Research, "Context Rot: How Increasing Input Tokens Impacts LLM Performance"
    (research.trychroma.com/context-rot, 2026): shows performance degrading as input length
    grows even within a model's supported context, i.e. degradation is not merely a hard-cutoff
    event.
  - Liu et al., "Lost in the Middle" (TACL 2024, arXiv:2307.03172) and "Context Length Alone
    Hurts LLM Performance Despite Perfect Retrieval" (EMNLP Findings 2025,
    aclanthology.org/2025.findings-emnlp.1264): both already cited above under claims (c)/(d)
    of the Paper 1 section; they measure quality loss as a function of context length/position,
    not as a function of a runtime's *automatically chosen* default ceiling.
  - Multiple practitioner write-ups ("Your Agent's Context Window Overflowed and It Answered
    Anyway," dev.to; various "context degradation" / "instruction drift" blog posts) describe
    the intuitive mechanism (instructions become a smaller fraction of a full context, agents
    silently drop earlier constraints) but are not measured studies with a controlled
    memory-tier independent variable.
- **What is missing, specifically:** no source found runs the full causal chain end to end --
  i.e., varies available VRAM/memory tier as the independent variable, observes the runtime's
  auto-selected default context window change as a mediating variable, and then measures
  agent/instruction-following task success (not raw perplexity or throughput) as the dependent
  variable. The GitHub issues establish the first link (memory tier to default context) is real
  and already causing operational pain; the academic literature establishes the second link
  (context overflow/length to quality loss) in isolation. Nobody has published the chained
  measurement with memory tier as the entry point.

**How this paper's planned measurement differs:** this paper would be the first to hold the
memory constraint as the independent variable and trace it through the runtime's default-context
decision to a measured agent/instruction-following score, rather than treating context length as
a directly-set experimental knob (as the "context rot" and "lost in the middle" work does) or
treating the VRAM-to-default-context link as an unmeasured bug report (as the GitHub issues do).
It would not be redundant with either body of work, but it also should not claim to be the first
to observe the mechanism, since the Ollama issue tracker already documents users hitting it in
production.

## Claim (b): Memory pressure from ordinary co-running consumer apps degrades LLM OUTPUT QUALITY specifically, not just speed

**Restatement:** A browser, office apps, etc. running alongside an active local LLM session
consume enough memory to degrade the LLM's correctness / instruction-following / hallucination
rate, as distinct from just slowing it down.

**Verdict: NO PRIOR ART FOUND**

- Everything found in this area measures throughput/latency effects of memory pressure, not
  output-quality effects:
  - General consumer-GPU discussion (dev.to, various blogs): "when models exceed available
    VRAM, the GPU starts thrashing... causing inference times to skyrocket" -- speed only.
  - The dev.to "vLLM vs llama.cpp vs Ollama" piece (see claim (c) below) measures tokens/sec
    degradation under forced RAM spillover, not answer correctness.
  - SiliconBench (arXiv:2609.19169, Sept 2026) evaluates "speed, memory, and fidelity" together
    on Apple Silicon serving engines and does use a classification task to check for quality
    regressions against an NVIDIA reference, which is the closest thing found to a quality
    metric under memory-constrained serving. But its "memory" axis is engine memory-budget
    configuration and steady-state headroom under concurrency, not a second, independent,
    ordinary consumer application competing for the same pool of memory during the session.
    This is the single closest paper and is worth reading in full before claiming novelty here,
    but it does not test the specific scenario in claim (b).
  - Nothing found tests the scenario as stated: an LLM session already running, with a browser
    or office app opened or made memory-hungry *during* that session, scoring the LLM's answers
    for correctness/hallucination before and after.
- This is a real, clean gap. No paper, blog benchmark, or GitHub issue was found that isolates
  co-running consumer application memory pressure as the manipulated variable and output quality
  (rather than tokens/sec or time-to-first-token) as the outcome.

**How this paper's planned measurement differs:** it would be genuinely new on the axis that
matters (quality, not speed) and on the specific stressor (ordinary consumer apps, not a
second inference workload or a synthetic memory hog). Worth stating this plainly to the person
running the project: this is the strongest, least-covered claim of the four, and probably the
best place to spend measurement time if only one of the four claims can be pursued.

## Claim (c): Systematic cross-vendor, cross-runtime classification of OOM behavior (hard error / silent truncation / memory spill / crash / hang)

**Restatement:** A classification of how llama.cpp, Ollama, vLLM (etc.) behave when they run
out of memory, compared across at least two GPU vendors and at least two runtimes.

**Verdict: PARTIALLY SHOWN**

- Anecdotal, per-vendor evidence for each failure mode exists in llama.cpp's own issue tracker,
  but nobody has assembled it into a systematic cross-vendor x cross-runtime matrix:
  - NVIDIA/CUDA: **ggml-org/llama.cpp#1866** ("CUDA out of memory - but there's plenty of
    memory") and general reports that llama-server crashes on CUDA OOM "without indication of
    how much context to remove or how many layers to move."
  - AMD/ROCm-HIP: **ggml-org/llama.cpp#19745** ("llama-server and llama-cli hang/crash during
    RPC tensor upload") documents llama-server hanging indefinitely at `load_tensors` in one
    path and llama-cli crashing with a HIP/HSA runtime fault in another, i.e. two different
    failure modes on the same vendor depending on binary.
  - Intel/Vulkan-SYCL: **ggml-org/llama.cpp#18946** ("Misc. bug: ErrorOutOfDeviceMemory -
    Critical Out of Device Memory") reports the Vulkan backend erroring out on Intel hardware
    both with and without flash attention enabled.
  - Apple Silicon (unified memory, not a discrete GPU vendor but a relevant fourth memory
    architecture): **lmstudio-ai/lmstudio-bug-tracker#2404**, cited above, shows silent
    context clamping rather than an explicit OOM error.
  - Cross-runtime, single-vendor (NVIDIA only) comparison: the dev.to article "vLLM vs
    llama.cpp vs Ollama: What Happens When Your Model Doesn't Fit in 24GB VRAM"
    (dev.to/sikamikanikobg) is a real, reasonably careful blog benchmark, not peer-reviewed,
    that found: vLLM OOMs outright at a fixed ~22.1-22.2 GB used / <700 MB free threshold
    regardless of quantization; llama.cpp and Ollama both spill to system RAM and continue
    generating at single-digit tok/s rather than failing; llama.cpp's manual layer offload beats
    Ollama's automatic split by 37x on time-to-first-token during spill, with similar
    steady-state decode speed. This is the closest thing to the claim found anywhere, but it is
    one vendor (NVIDIA), a blog post rather than a paper, and does not test AMD or Intel.
  - Cross-runtime, Apple-Silicon-only comparison: SiliconBench (arXiv:2609.19169) evaluates
    nine serving engines and finds that "explicit memory budgets do not guarantee memory
    headroom: two stacks complete every request while memory use approaches physical capacity
    and throughput declines" -- again a real cross-engine memory-behavior comparison, but confined
    to one memory architecture (Apple unified memory), not cross-vendor discrete GPUs.
- No source found runs the same OOM-inducing protocol against NVIDIA, AMD, and Intel GPUs with
  the same set of runtimes and classifies the resulting behavior into a common taxonomy (hard
  error / silent truncation / spill / crash / hang). The pieces exist scattered across GitHub
  issues (one vendor and one bug at a time) and two single-vendor comparative benchmarks
  (NVIDIA-only blog, Apple-only paper); nobody has stitched them into the cross-vendor matrix
  this claim describes.

**How this paper's planned measurement differs:** the novelty here is entirely in the
cross-vendor, common-protocol, common-taxonomy design, not in discovering that any individual
failure mode exists (all five modes named in the claim are already independently documented
somewhere in the llama.cpp tracker or the two blog/paper benchmarks above). This paper should
cite the dev.to piece and SiliconBench explicitly and frame its contribution as "the matrix
nobody assembled," not as "the discovery that these runtimes fail differently under memory
pressure."

## Claim (d): Fabrication vs. refusal specifically under context truncation, as distinct from the general hallucination/"I don't know" literature

**Restatement:** Whether models fabricate a specific wrong answer versus refuse/express
uncertainty when the information they need has been truncated from context, studied
specifically as a truncation phenomenon, not the broader (and much larger) hallucination/abstention
literature in general.

**Verdict: PARTIALLY SHOWN -- and the two bodies of work need to be kept explicitly separate**

Two distinct bodies of work exist and should not be conflated:

1. **The broad hallucination/abstention/"know your limits" literature** (large, general,
   mostly not about truncation): surveys such as Tonmoy et al., "Know Your Limits: A Survey of
   Abstention in LLMs" (TACL 2025, doi:10.1162/tacl_a_00754), and studies of abstention under
   noisy or contradictory (but present) context such as Feng et al., "Characterizing LLM
   Abstention Behavior in Science QA with Context Perturbations" (Findings EMNLP 2024,
   arXiv:2404.12452) -- both already catalogued under claim (b) of the Paper 1 section above.
   This body of work is not about physical absence of information from a truncated context
   window; it studies abstention under confidence miscalibration, distractors, or contradiction.

2. **The narrower "insufficient/missing retrieved context" literature**, which is closer to but
   still not identical to a truncation-specific setting:
   - Joren et al. (Google Research), "Sufficient Context: A New Lens on Retrieval Augmented
     Generation Systems" (arXiv:2411.06037, 2024): directly measures whether models answer
     incorrectly instead of abstaining when retrieved context is *insufficient* to answer the
     query. Key finding: larger/stronger models (Gemini 1.5 Pro, GPT-4o, Claude 3.5) "excel at
     answering queries when the context is sufficient, but often output incorrect answers
     instead of abstaining when the context is not [sufficient]," while weaker models abstain
     or hallucinate even with sufficient context. This is the single closest paper to claim (d)
     found anywhere in this search. The gap: "insufficient context" here means the retriever
     did not surface the needed document at all; it is a retrieval-quality problem, not a
     context-window-capacity/truncation problem. The information was never fetched, as opposed
     to being fetched and then cut off by a memory-driven context-size ceiling.
   - "Prompt-Based Abstention Fails Under Misleading Context" (arXiv:2608.22228, GRAB-RAG-style
     study): finds prompted abstention treats missing and misleading evidence alike, and models
     still answer 41.6% of misleading questions even under explicit abstention instructions.
     Same gap as above: missing evidence via retrieval failure, not truncation of present
     evidence via a memory-limited context window.
   - "Distractor-Aware Truncation: Disentangling Context-Length Effects from Signal Loss in
     Long-Context LLM Benchmarks" (arXiv:2608.03297): the title uses the word "truncation," and
     it is worth reading directly, but from the search summary it is aimed at separating
     context-length effects from signal-loss effects in benchmark design, not at classifying
     model responses into fabrication vs. refusal buckets as the outcome variable.
   - "Efficient Solutions For An Intriguing Failure of LLMs: Long Context Windows Yet Poor..."
     (arXiv:2408.01866): also about long-context failure generally, not a fabrication/refusal
     split under truncation specifically as far as the search results show.
- No source found sets up the exact experiment this claim describes: take a context that
  contains the needed information, truncate it (by a memory/context-size mechanism, not a
  retrieval failure), and measure the resulting split between "fabricates a specific wrong
  answer" and "refuses/expresses uncertainty," as a function of how much was truncated.

**How this paper's planned measurement differs:** the Sufficient Context paper is close enough
that this paper must cite it and explain the distinction clearly: retrieval-insufficiency
(information never retrieved) versus capacity-truncation (information retrieved/present, then
cut by a memory-constrained context window). If this paper's probe design can't cleanly
separate "the model never had the evidence" from "the model had the evidence and it got cut,"
a reviewer will reasonably ask why this differs from Joren et al. The distinction has to be
made mechanically explicit in the methodology, not just asserted, or the claim will not survive
review.

## GitHub issues found (relevant to claims a-d)

**llama.cpp (ggml-org/llama.cpp unless noted):**
- `#17284` -- Eval bug: server returns HTTP 400 (context size exceeded) instead of handling it
  gracefully (relevant to a/c).
- `#11577` -- Feature request: resize an existing context (relevant to a).
- `#18889` -- Eval bug: `llama_model_fit` results in zero context size (relevant to a).
- `espetro/llama.cpp#2` (fork) -- state silently truncated to `row_cap`, no usage field in the
  server response to detect it (relevant to a/d framing: silent, undetectable truncation).
- `lmstudio-ai/lmstudio-bug-tracker#2404` -- context length silently clamped to ~6,656 tokens on
  24 GB unified memory regardless of available memory (relevant to a).
- `#19745` -- llama-server/llama-cli hang vs. crash during RPC tensor upload, AMD/HIP path
  (relevant to c).
- `#18946` -- `ErrorOutOfDeviceMemory` on Intel Vulkan backend (relevant to c).
- `#1866` -- "CUDA out of memory - but there's plenty of memory" (relevant to c).

**Ollama (ollama/ollama):**
- `#14073` -- "New default context lengths will break": tiered VRAM-based default context
  overflows VRAM on a 52 GiB machine, causes CPU-spill unresponsiveness (relevant to a).
- `#14116` -- "Tiered context length can exhaust VRAM": `OLLAMA_NUM_PARALLEL` not factored into
  the tier's VRAM estimate (relevant to a).
- `#12353` -- Feature request: `--fit-vram` to auto-size `num_ctx` to a VRAM budget, filed
  because users currently must guess and over-guessing overflows VRAM silently (relevant to a).
- `#9774` -- Request to estimate VRAM needs from context length and quantization (relevant to a).
- `#9890` -- "Large context size completely breaks the usability of the model" (relevant to a).
- `#14173` -- Ollama 0.15.6 ignores requested context size (relevant to a).
- `#11964` -- "context size larger than set" (relevant to a).
- `#18229` -- "Loaded context length: source not shown," i.e. no way to tell whether the active
  `num_ctx` came from default/env/Modelfile (relevant to a: opacity of the silently-chosen
  ceiling).
- `#11659` / `#18242` -- users asking to set context length directly instead of the tiered
  auto-selection (relevant to a: confirms the tiering is often unwanted/unexpected).

These are all issue-tracker reports, not measured studies. They confirm the mechanism in claim
(a) is real and already causing production problems, but none of them measure downstream
agent/instruction-following quality, which remains the open part of claim (a).
