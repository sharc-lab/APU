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
