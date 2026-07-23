# Contributions

This document enumerates the five novelty claims and where they are substantiated in this repository.

## 1. Budget-as-State Routing

- Claim statement: Routing quality/cost improves when remaining budget is an explicit policy state rather than a fixed threshold.
- Nearest prior work: Query-level model routers (RouteLLM/Martian/NotDiamond) and static cloud-edge dispatch policies.
- Substantiating experiment in this repo: budget-aware cascade sweep and trajectory histogram in [evaluation/sweep.py](evaluation/sweep.py), with decision-level theta logging.
- Commercialization note: Enables spend-safe SLA control for enterprise deployments with predictable budget exhaustion behavior.

## 2. Step-Level Granularity

- Claim statement: Intra-task step routing yields finer and more effective control than per-query routing.
- Nearest prior work: Per-request model selection and static endpoint routing.
- Substantiating experiment in this repo: per-step routing-decision artifacts in [results/pareto_results.json](results/pareto_results.json) produced by [evaluation/sweep.py](evaluation/sweep.py).
- Commercialization note: Supports product features like premium escalation only on difficult steps, reducing blended inference cost.

## 3. Speculative Action Execution

- Claim statement: Running local/cloud speculatively at action level with agreement gating can reduce effective cloud billing for repeated categories.
- Nearest prior work: Token-level speculative decoding work; little published action-level rollback accounting.
- Substantiating experiment in this repo: speculative policy/reporting in [routing/policies/speculative.py](routing/policies/speculative.py) and [evaluation/sweep.py](evaluation/sweep.py), including `SPEC_ROLLBACK` timing category.
- Commercialization note: Provides a practical bridge to low-latency UX while preserving quality fallback pathways.

## 4. Trace-Distilled Router Flywheel

- Claim statement: Routing logs + replay traces can continuously train better routers without incremental API spend.
- Nearest prior work: Offline policy learning from logs; production router heuristics without standardized replay traces.
- Substantiating experiment in this repo: [analysis/distill_router.py](analysis/distill_router.py) and [routing/policies/learned_router.py](routing/policies/learned_router.py) replay-mode evaluation loop.
- Commercialization note: Converts routine benchmark operations into proprietary model-selection data assets.

## 5. Sampled Quality Certification

- Claim statement: Sampling-based cloud verification with Wilson intervals yields deployable confidence bounds on local quality.
- Nearest prior work: Full-cost exhaustive judging and static benchmark reporting.
- Substantiating experiment in this repo: sampled-verification pipeline in [evaluation/certify.py](evaluation/certify.py) outputting certified quality per (policy, budget).
- Commercialization note: Supports governance and compliance reporting with tunable audit spend.
