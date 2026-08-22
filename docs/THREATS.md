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

## 8. Format Compliance Scored as Quality Failure

- Threat: Exact-match scorers reject outputs where the correct answer is present but surrounded by additional text (e.g., "130/5=26" when "26" is expected; "1130+612+765=2507" when "2507" is expected). These are format failures — the model computed the right answer but violated the "ONLY the final answer" instruction — but the scorer records them as quality failures (score=0), inflating apparent degradation.
- Measurement: Scanning run_20260813T021516Z.jsonl for exact-scoring rows where score=0.0 and the expected value appears in the output as a complete token (not a coincidental digit substring) found 12 true format failures across 2 probes out of 676 exact-scoring rows (1.8%). False-positive rate from naive substring matching was high (29 additional coincidental matches, mostly rea_05 "1200" containing "120"), so the boundary between format failure and correct answer requires word-boundary matching.
- Affected probes and counts by depth:
  - sea_03 (search_heavy): outputs "A+B+C=2507" instead of "2507". Occurs at d=2000 (2/5 reps), d=8000 (2/5), d=16000 (1/5), d=32000 (2/5). Flat across depths — stochastic non-compliance with the "ONLY" instruction, not depth-driven.
  - lon_02 (long_horizon): outputs "130/5=26" instead of "26" at d=16000 (4/5 reps) and d=32000 (1/5 reps). Notably, these are the reps that compute the arithmetic correctly; at d=0 the model outputs a wrong answer, and at d=32000 some reps output "126" (genuinely wrong). The format failure and the correct computation are coupled: the model that computes step-by-step correctly at d=16000 also shows its working.
- Impact: 12 rows scored 0.0 that should be classified as format-fail rather than quality-fail. This slightly inflates apparent degradation at d=16000 (5 rows, rate 3.8% of exact-scoring attempts) and d=32000 (3 rows, 2.3%). The overall effect on aggregate mean scores is small (<0.01 absolute), but for per-probe analysis lon_02 at d=16000 shows apparent score=0.0 when the model is computing correctly.
- Note: rea_05 consistently outputs "1200" (wants "120") at all depths — this is a unit error (divides by 100 instead of 1000), not a format failure. "120" appears as a substring of "1200" but 1200 is a genuinely wrong answer.
- Mitigation: For Phase 2, add a format-aware scorer mode that extracts the last numeric token from outputs before exact-matching. This separates format compliance (did the model follow the ONLY instruction?) from arithmetic correctness (did it compute the right answer?), which are distinct capabilities.

## 9. d=0 Baseline Confound: Prompt Presence vs. Context Depth

- Threat: The d=0 condition is structurally different from every d>0 condition. At d=0, the model receives only the probe prompt (typically 30–200 tokens). At d=2000, it receives ~2000 tokens of filler followed by the probe — a prompt that is 10–50× longer in aggregate. This structural difference (presence vs. absence of any filler) changes model inference behaviour independently of the depth being measured.
- Evidence: Two reverse-pattern probes illuminate the mechanism.
  - **str_03** (structured_output): at d=0, the model outputs a non-approved status value for a ticket described as "still being worked." At d=2000+, it uses "open" exactly as required by the schema. The filler contains none of the probe's vocabulary ("open", "closed", "pending", "status", "json") — confirmed by substring search. The mechanism is prompt-length-induced compliance: a longer prompt causes the model to read the explicit constraint list more carefully rather than improvising a natural-language synonym.
  - **lon_02** (long_horizon): at d=0, the model returns 63 (a deterministic arithmetic error, consistent across all 5 reps). At d=2000–8000, it returns 26 (correct). No numbers from the arithmetic problem appear in the filler. At d=16000+, the model's arithmetic is correct but it leaks its working ("130/5=26"), failing the exact-match scorer. The mechanism is prompt-length-induced deliberateness: a shorter prompt leads to a quicker, less careful computation.
- Impact: The aggregate mean score appears flat (0.48–0.52 across d=0–32000) in part because reverse-pattern probes (str_03, lon_02 gaining ≈1.0 at d=2000) cancel genuine degradation probes (cha_04 losing 1.0 at d=2000). Depth-effect curves for individual probes that improve from d=0 to d=2000 cannot be interpreted as depth effects — they reflect a prompt-presence step change, not a depth gradient.
- Mitigation planned: For future sweeps, use a nonzero minimum baseline (e.g., d=256 or d=512 tokens) so that filler is always present and the presence-vs-absence confound is removed. The genuine depth effect is then measured relative to that baseline, not relative to the no-filler condition.
- Note: This does not invalidate the "semantically inert" claim about filler *content*. The filler's content does not help or hinder any probe answer. The confound is structural, arising from the d=0 no-filler condition, not from what the filler says.

## 9. KV Prefix Cache Not Persistent Between API Calls (Discrete Host)

- Threat: The depth→rep→probe loop order was originally motivated partly by the hypothesis that consecutive probe calls at the same depth would benefit from Ollama's KV prefix cache (shared filler prefix = cache hit). A two-call persistence test at d=64,000 tokens on the Razer Blade (RTX 4070, Ollama 0.9.x) showed that the cache is flushed between `/api/chat` calls: both calls took ~124 s (consistent cold load), not ~3.5 s (expected cache-hit latency). No KV persistence was observed.
- Impact: `position_in_cell` tracks scheduling position within a (depth, rep) cell, not warm/cold cache status. Any latency regression of position on score is a scheduling-order confound, not a cache effect. Latency analyses should treat position_in_cell as a nuisance covariate, not a cache indicator.
- Mitigation: The field `cache_state` was removed from result rows (it would have silently labelled position 0 as "cold" and the rest as "warm" — false data). `position_in_cell` is retained as-is; its causal interpretation is ambiguous (it could reflect anything from thermal throttling to scheduler jitter) and must not be treated as a cache signal.
- Note: KV caching behaviour is host-specific. Sweeps on the AMD Strix Halo unified-memory host should re-run the persistence test before treating probe ordering as a cache confound.

## 9. Extreme Latency Outliers at d=32,000 rep=1 (Discrete Host)

- Threat: Three probes in the d=32,000, rep=1 cell showed anomalous latencies: sea_01 (6,703 s ≈ 112 min), sea_04 (3,831 s ≈ 64 min), lon_02 (3,641 s ≈ 61 min). All three produced 2–4 output tokens despite num_predict=800. The preceding probe in that cell was cod_07, which generated 714/800 tokens over ~1,085 s at d=32,000 rep=0, likely leaving the Ollama server in a degraded state.
- Impact: The three outlier rows have valid scores and must not be excluded from quality analysis. Their latencies, however, reflect server-state degradation rather than model inference speed and must be excluded from any latency or throughput analysis.
- Identification: A row is a latency outlier if latency_s > 1800 and tokens_out < 10. Three rows match this criterion in run_20260813T021516Z.jsonl (probes sea_01, sea_04, lon_02 at d=32000, r=1).
- Mitigation: Add a server health check (e.g., a ping call with num_predict=1 and timeout=30 s) between cells when the previous cell's maximum latency exceeded a threshold (e.g., 600 s). This would detect and surface degraded state before the next cell begins.

## 10. Artifact Probe Selection Bias Toward Simple Retrieval

- Threat: Two of the ten artifact-bearing probes (art_03 and art_04) were redesigned during authoring because their original formulations required multi-step inference — art_03 asked for the maximum-CPU_HOURS job (argmax over a table) and art_04 asked which user performed a specified action (reverse actor lookup). Both models failed these questions reliably with 4,000 tokens of filler present, even when the artifact was intact, and redesign was necessary to achieve headroom. The final suite therefore consists entirely of direct retrieval lookups: given a key, return the value at that key from the artifact.
- Impact: The valid artifact suite is selected for probe types that both models can handle under the experimental condition. Multi-step inference over an artifact (filtering, argmax, reverse lookup) is not represented. The truncation result — that score drops to zero when the artifact is absent from context — is established for **retrieval-type artifact dependence**, not for inference-over-artifact dependence. The claim that truncation causes fabrication is sound within this scope but does not extend to inference tasks without a separate experiment showing the same probes pass headroom.
- Implications for the discreteness argument: the structured-vs-narrative contrast in the art_* suite tests whether different artifact *formats* behave differently under truncation, while both being retrieval tasks. It does not test whether structured artifacts require more inference steps than narrative ones, which is a separate question.
- Planned future work: Author a multi-step inference variant of the art_* suite (argmax, filter, chain-of-lookup) and repeat the headroom + truncation experiment. If these also show abrupt collapse, the discreteness argument extends to inference. If they show graded degradation even at partial-artifact, the mechanism differs and must be characterised separately.
