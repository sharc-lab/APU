# Threats to Validity

This file seeds Section 6 of the paper and tracks planned mitigations.

## 1. Windows Clock Quantization

- Threat: `process_time_ns` granularity on Windows can create per-seed timing artifacts and unstable residual attribution at short span durations.
- Impact: Inflated variance in per-seed residual metrics and fragile threshold-based validity checks.
- Planned resolution on AMD Linux machine: Re-run instrumentation with high-resolution Linux timing counters and compare category distributions against Windows runs.

## 2. Mock-Tool Inflation

- Threat: Local mock tool execution can dominate TOOL_COMPUTE, potentially overstating orchestration-relative compute shares for real remote-tool deployments.
- Impact: Cross-study comparability is reduced when workload locality differs.
- Planned resolution on AMD Linux machine: Add matched remote-tool and mixed-tool profiles to isolate orchestration vs tool-compute contributions under realistic latency/cost mixes.

## 3. Seed Variance

- Threat: Small numbers of seeds can produce unstable quality/cost frontiers due to stochastic task/model effects.
- Impact: Apparent policy ranking differences may be sampling noise.
- Planned resolution on AMD Linux machine: Increase seed count, compute confidence intervals on policy deltas, and report sensitivity to seed subsets.

## 4. Low Per-Task n

- Threat: Limited per-task repetitions under each policy/budget condition can underpower tail-latency and quality comparisons.
- Impact: p95/p99 estimates and per-task quality conclusions may be noisy.
- Planned resolution on AMD Linux machine: Increase per-task repetitions and introduce stratified resampling/bootstrapped uncertainty reporting for tails and task-level quality.
