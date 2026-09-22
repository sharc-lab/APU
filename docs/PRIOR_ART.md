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
