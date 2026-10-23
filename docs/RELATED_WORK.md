# Related Work

This document is structured as the future paper Section 2.

## 2.1 LLM Routing

### RouteLLM

- Citation: see [references.bib](references.bib), key `routellm`.
- One-line summary: RouteLLM studies query-level routing between smaller and larger LLMs to trade quality for cost.
- Gap we exploit: Existing routing decisions are query-level/static; our repo studies step-level, budget-stateful routing with per-step replayable traces.

### Martian

- Citation: see [references.bib](references.bib), key `martian`.
- One-line summary: Martian-style systems optimize model selection policies to improve cost-performance tradeoffs for requests.
- Gap we exploit: Prior framing is largely per-query policy optimization, not action-level routing under a depleting budget state.

### NotDiamond

- Citation: see [references.bib](references.bib), key `notdiamond`.
- One-line summary: NotDiamond emphasizes practical model routing for production APIs using request-level selectors.
- Gap we exploit: We focus on intra-task step granularity, where routing choices happen within one task trajectory and become router-training data.

### Cloud-Edge Routing Literature

- Citation: see [references.bib](references.bib), keys `cloud_edge_survey`, `edge_llm_serving`.
- One-line summary: Cloud-edge routing work typically partitions work by static placement rules or per-request offloading heuristics.
- Gap we exploit: We treat remaining budget as an explicit state variable and evaluate dynamic escalation/rollback behavior across task steps.

## 2.2 Host-Overhead Characterization

### TaxBreak (arXiv:2603.12465)

- Citation: see [references.bib](references.bib), key `taxbreak2026`.
- One-line summary: TaxBreak characterizes compute/serving overhead with emphasis on system-level serving efficiency.
- Gap we exploit: The literature is largely GPU/serving-centric; we instrument host CPU orchestration overhead at fine category granularity for agentic workflows.

### CPU-Centric Agentic Profiling (arXiv:2511.00739)

- Citation: see [references.bib](references.bib), key `cpuagent2025`.
- One-line summary: This line of work profiles agentic execution on host resources and decomposes overhead sources.
- Gap we exploit: We tie host-overhead categories directly to routing-policy outcomes and replay-stable decision logs for policy learning.

## 2.3 Agent Benchmarking Cost

### Large-Scale Cost Studies (arXiv:2603.23749)

- Citation: see [references.bib](references.bib), key `agentcost2026`.
- One-line summary: Recent benchmark economics papers quantify steep evaluation costs, including the cited ~$40k HALO finding.
- Gap we exploit: We operationalize cost control through replay caching, sampled certification, and zero-API-cost distillation/evaluation loops.

## 2.4 Speculative Execution

### Token-Level Speculative Decoding

- Citation: see [references.bib](references.bib), keys `leviathan2023specdec`, `specdec_followup`.
- One-line summary: Speculative decoding accelerates token generation by validating draft tokens from a smaller model.
- Gap we exploit: Prior evidence is token-level; our repository studies action-level speculative execution (step outputs, agreement checks, rollback accounting).

## 2.5 Edge-Cloud Agent Delegation, Routing and Model Fit (added 2026-10-23)

Fetched 2026-10-23 unless marked "not verified". Summaries are limited to what each abstract or README says. None of
these is in references.bib yet. Common difference: this work measures how a local runtime's memory setting (the loaded
context window) silently cuts an agent task's context and what that does to correctness and cloud cost, and routes a
whole task at its start on a window-fit check; it does not contribute a learned router or a cut-off detector (the
cut-off signal used as a safety net is Ollama's own log lines).

### EdgeWise (arXiv:2610.17560)

- Citation: Z. Shi, J. Jiang, R. Hachiuma, et al. (13 authors), "EdgeWise: Bringing Long-Horizon Agents to the Edge
  via Selective Cloud Delegation", arXiv:2610.17560, v1 2026-10-13 (v2 2026-10-15). https://arxiv.org/abs/2610.17560
- One-line summary: a small local model runs a long agent task and delegates critical steps to a cloud LLM, with the
  delegation decision learned by step-level reinforcement learning (abstract reports cost reductions on SWE-Rebench and
  Terminal-Bench 2.1).
- How this work differs: no learned delegation policy; we characterize the memory-setting failure that makes local
  steps silently wrong and measure whole-task routing at start and the cost of late escalation.

### EC2-Bench (arXiv:2610.17816)

- Citation: P. Huang, Z. Liu, S. Ren, S. Yu, S. Yao, Y. Yan, C. Xiao, M. Sun, "Empowering Edge Agents: A Systematic
  Analysis of Edge-Cloud Collaboration in LLM Agent Execution", arXiv:2610.17816, v1 2026-10-13 (v2 2026-10-18).
  https://arxiv.org/abs/2610.17816
- One-line summary: a benchmark scoring edge-cloud collaboration strategies for long-horizon agent tasks on task
  performance, resource usage and privacy exposure.
- How this work differs: not a strategy benchmark; we measure one runtime mechanism (the loaded window chosen from
  memory) on real unified-memory hardware and its silent wrong answers.

### Minions (arXiv:2502.15964)

- Citation: A. Narayan, D. Biderman, S. Eyuboglu, A. May, S. Linderman, J. Zou, C. Re, "Minions: Cost-efficient
  Collaboration Between On-device and Cloud Language Models", arXiv:2502.15964, 2025-02-21.
  https://arxiv.org/abs/2502.15964
- One-line summary: an on-device model and a cloud model collaborate on long-document tasks; in MinionS the cloud
  model decomposes the task into subtasks over shorter chunks that run locally in parallel.
- How this work differs: tool-using agent tasks rather than long-document reading, and no decomposition protocol; the
  local side is a fixed runtime whose context window is the variable under study.

### Arch-Router (arXiv:2506.16655)

- Citation: C. Tran, S. Paracha, A. Hafeez, S. Chen, "Arch-Router: Aligning LLM Routing with Human Preferences",
  arXiv:2506.16655, 2025-06-19. https://arxiv.org/abs/2506.16655
- One-line summary: a 1.5B-parameter router that maps queries to user-defined domain and action preferences to pick a
  model.
- How this work differs: our route depends on whether the task fits the loaded window, not on query semantics or
  preferences, and we report its measured accuracy and cost against all-cloud and naive local.

### ContextPaw (not verified)

- Citation: not verified. Web searches on 2026-10-23 found no project, paper or repository by this name; no details
  are recorded here until a source is supplied.
- How this work differs: to be written once the source is confirmed.

### llmfit (GitHub AlexsJones/llmfit)

- Citation: AlexsJones, "llmfit", GitHub repository, MIT license, https://github.com/AlexsJones/llmfit (README fetched
  2026-10-23; no release date taken).
- One-line summary: a terminal tool that scans the machine's RAM, CPU and GPU and rates models on what runs on that
  hardware, including a context dimension estimated from context length, parameter count and quantization.
- How this work differs: llmfit estimates fit before running; we measure the correctness and cost consequences of the
  window a runtime actually loads, on agent tasks.

### Ollama issue #14259

- Citation: akuligowski9, "Chat history and embedding truncation happens silently with no user-visible indication",
  ollama/ollama issue #14259, opened 2026-02-14, open (label documentation).
  https://github.com/ollama/ollama/issues/14259
- One-line summary: asks Ollama to raise its truncation log line from debug to warning level, with the number of
  dropped messages and the active context length, and to warn likewise on the embed endpoint.
- How this work differs: an independent report of the silent cut we measure; our safety net reads the same runtime
  lines (the runtime's signal, not ours), and we add measured effects on agent correctness and cloud cost.
