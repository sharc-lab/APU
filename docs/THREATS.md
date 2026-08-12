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

## 5. RLIMIT Enforcement Unavailable on Windows

- Threat: `score_unit_test` uses `resource.setrlimit` to cap CPU time and address space on POSIX hosts. On Windows, `RLIMIT_NPROC` does not exist and `RLIMIT_AS` is unsupported; the guard in `scorers.py` catches these silently.
- Impact: On Windows, a runaway or deliberately hostile model-generated submission can consume unbounded CPU and memory. The `timeout=30` argument to `subprocess.run` is the **sole** resource cap available on Windows.
- Planned resolution: Run sweeps on the Linux box where all four `setrlimit` calls succeed. The Windows path is development-only; do not treat its results as production measurements for CPU-sensitive probes.
- Note: The sandbox's env scrubbing (`safe_env` allowlist, exclusion of API keys) is platform-independent and is verified by `tests/test_scorer_sandbox.py` on both platforms.

## 6. Synthetic Filler vs Real Agent Context

- Threat: The harness injects synthetic prose (numbered administrative log entries) as filler context. Real accumulated agent context has structure — tool call results, prior model reasoning, partial plans — that may affect attention patterns differently from uniform synthetic text.
- Impact: Measured quality degradation curves may understate or overstate degradation relative to a real agent workload at the same nominal token depth.
- Mitigation: The filler is semantically inert (contains no information that could help or hinder any probe answer) and is added in the `unlabelled` mode by default, so the model receives no banner telling it to ignore the block. This approximates the worst-case real scenario more closely than labelled filler. The gap between synthetic and real context structure is a known confound; sweeps on real accumulated agent traces are a future planned experiment.
- Test: `tests/test_context_independence.py` asserts that the expected answer string for every probe is absent from generated filler at all depths across multiple seeds.

## 7. Probe Scores and Task Scores Are Separate Measurement Tracks

- Threat: Attempting to score the 14 live agent tasks in `harness/adapters/sdk_direct.py` using the probe answer keys from `evaluation/probes/prompts.jsonl` produces confidently-wrong numbers. Each probe is a self-contained item: prompt + fixed expected value. A probe's expected value answers *that probe's prompt*, not any agent task prompt. Applying a probe scorer to an agent task output (e.g., scoring a transformer-architecture explanation against a Meridian-sensor document answer key) guarantees a near-zero score regardless of output quality. The number looks like a real measurement but has no relationship to the task's actual quality.
- Impact: Cross-scoring would silently corrupt the routing study's quality axis with systematic false-low scores. Downstream policy comparisons and Pareto frontiers would rank all policies equally badly, hiding real quality differences that the judge correctly distinguishes.
- Rule: **Do not create any mapping from task IDs to probe IDs for the purpose of scoring task outputs.** The two tracks must remain independent:
  - `harness/runner.py` runs probes against their own prompts and scores with `evaluation/probes/scorers.py`. This is the Paper 1 quality axis (local-model degradation under context growth).
  - `evaluation/quality.py` scores agent task outputs using the LLM judge (and the CN-01 programmatic scorer, which is task-specific, not probe-derived). This is the Paper 2 routing study quality axis.
- If deterministic scoring is needed for additional agent tasks, write a task-specific programmatic scorer inside `evaluation/quality.py` (as was done for CN-01), not a probe answer key.
