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
- Runtime truncation clarification (2026-09-15): No runtime tested in this study produces silent prompt truncation. Both llama-server b10970 and Ollama 0.32.9 return HTTP 400 (`exceed_context_size_error`) when the prompt exceeds the configured context window, regardless of the `--context-shift` flag. Prior harness-level truncation (Stage C) therefore models application-layer context management, not runtime behaviour. See `docs/RUNTIME_EVICTION.md` for the empirical characterisation.

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

- Threat: The depth→rep→probe loop order was originally motivated partly by the hypothesis that consecutive probe calls at the same depth would benefit from Ollama's KV prefix cache (shared filler prefix = cache hit). A two-call persistence test at d=64,000 tokens on the Razer Blade 14 RZ09-0508 (RTX 4070 dGPU, Ollama 0.9.x) showed that the cache is flushed between `/api/chat` calls: both calls took ~124 s (consistent cold load), not ~3.5 s (expected cache-hit latency). No KV persistence was observed.
- Impact: `position_in_cell` tracks scheduling position within a (depth, rep) cell, not warm/cold cache status. Any latency regression of position on score is a scheduling-order confound, not a cache effect. Latency analyses should treat position_in_cell as a nuisance covariate, not a cache indicator.
- Mitigation: The field `cache_state` was removed from result rows (it would have silently labelled position 0 as "cold" and the rest as "warm" — false data). `position_in_cell` is retained as-is; its causal interpretation is ambiguous (it could reflect anything from thermal throttling to scheduler jitter) and must not be treated as a cache signal.
- Note: KV caching behaviour is host-specific. The blade14_780m iGPU (AMD Radeon 780M, unified LPDDR5) is available on the measurement host for an architecture-comparison replication; it is not target-class. Sweeps on a target-class unified-memory device (AMD Strix Halo) should re-run the persistence test before treating probe ordering as a cache confound on that hardware.

## 9. Extreme Latency Outliers at d=32,000 rep=1 (Discrete Host)

- Threat: Three probes in the d=32,000, rep=1 cell showed anomalous latencies: sea_01 (6,703 s ≈ 112 min), sea_04 (3,831 s ≈ 64 min), lon_02 (3,641 s ≈ 61 min). All three produced 2–4 output tokens despite num_predict=800. The preceding probe in that cell was cod_07, which generated 714/800 tokens over ~1,085 s at d=32,000 rep=0, likely leaving the Ollama server in a degraded state.
- Impact: The three outlier rows have valid scores and must not be excluded from quality analysis. Their latencies, however, reflect server-state degradation rather than model inference speed and must be excluded from any latency or throughput analysis.
- Identification: A row is a latency outlier if latency_s > 1800 and tokens_out < 10. Three rows match this criterion in run_20260813T021516Z.jsonl (probes sea_01, sea_04, lon_02 at d=32000, r=1).
- Mitigation: Add a server health check (e.g., a ping call with num_predict=1 and timeout=30 s) between cells when the previous cell's maximum latency exceeded a threshold (e.g., 600 s). This would detect and surface degraded state before the next cell begins.

## 11. Unified vs. Discrete Memory: Non-Equivalence of Nominal Pool Size

- Threat: A 64 GB unified LPDDR5X system (AMD Strix Halo) and a 64 GB discrete
  GPU VRAM system (e.g., Intel EVO-T2S) share the same nominal memory figure but
  provide materially different inference headroom. On discrete memory, the GPU VRAM
  pool is fully available for model weights and KV cache; system RAM is a separate
  pool. On Strix Halo, the 64 GB is the total shared budget for OS, all processes,
  KV cache, weights, and GPU workloads combined. OS and driver overhead typically
  consumes 8–12 GB at idle, reducing effective inference headroom to ~52–56 GB.
- Impact: Direct comparison of quality-vs-depth curves, KV eviction onset, or
  fabrication rate between `evox2_strix_halo_64gb` and any discrete-memory 64 GB
  config conflates two different effective memory constraints and will produce
  misleading conclusions. A discrete system with nominally the same size reaches
  eviction later (more headroom) and shows a different curve shape.
- Scope: Applies to any cross-architecture comparison where both sides are
  identified by total nominal GB. Does not affect comparisons within the same
  architecture (e.g., 64 GB vs. 128 GB Strix Halo, or two discrete platforms).
- Mitigation: Always compare by `achievable_pool_gb` (telemetry-populated field
  in configs/hardware/), not by `memory_gb`. The `memory_architecture` field
  (`unified` vs. `discrete`) must be a covariate in any cross-architecture
  regression. Cross-architecture comparisons must correct for the headroom
  difference or be restricted to matched achievable-pool values.
- Config flags: `evox2_strix_halo_64gb.yaml` carries this caveat inline.
  Analysis code reading configs/hardware/*.yaml should check `memory_architecture`
  before pooling results from configs with the same `memory_gb`.

## 12. Replay Cache: Partial Reproducibility of Axis B Measurements

- Threat: The `ReplayCache` (modes AUTO / RECORD / REPLAY) reproduces `recorded_latency_ms`
  (the actual inference time measured during recording) and all quality-axis fields
  (`score`, `score_detail`, model output text) from disk without re-running inference.
  However, span timing fields — `orch_setup_ns`, `http_client_ns`, `tool_compute_ns` —
  are **live wall-clock measurements** taken at replay time, not stored in the trace.
  They reflect the cost of reading the trace from disk plus local harness overhead,
  not the original inference latency.
- Impact: A reader reproducing Fig 6.1 from a replay trace will recover the quality
  axis (score vs. depth) and the `recorded_latency_ms` latency axis faithfully.
  The span-breakdown fields in the same rows are not replayed values — they are fresh
  measurements of the replay code path, which is dominated by disk I/O rather than
  model inference. Any analysis that aggregates or plots span fields from replayed rows
  alongside live-inference rows will conflate two different measurement populations.
- Mitigation: Rows emitted during replay carry `replayed: true` in the session output.
  Any span-level analysis must filter to `replayed == false` rows. The `recorded_latency_ms`
  field is safe to use from both live and replayed rows.
- Scope: Applies to all Axis B span data (Fig 6.1 latency contours, Table 5.1 category
  decomposition). Axis A quality scores are fully reproducible from replay.

## 12a. Call Cache Key Covers Full Assembled Prompt — Hit Cannot Return a Response for Different Inputs

- Finding: The content-addressed call cache in `harness/cache.py` keys on
  SHA-256 of `{model, prompt, params}` where `prompt` is the fully assembled
  text returned by `context.wrap_prompt()` — the complete filler block plus the
  probe text.  Both the filler depth and the probe question therefore participate
  in the key.  A cache hit can only occur when the model, the full assembled
  prompt, and all inference parameters (max_tokens, temperature, filler_mode,
  model_variant) are byte-for-byte identical to a prior call.  It is not possible
  for a hit to silently return a response produced for a different probe or a
  different depth.
- Relevance: A test failure during outcome-classifier integration showed that
  `_run_cell_with_span` was not patching `cache.get`/`cache.put`, so disk-cached
  results from an earlier test call were returned regardless of the mock's
  `fake_output` value.  This looked like the cache returning a response for
  different inputs, but the mechanism was different: the mock replaced
  `_call_ollama_streaming` while the disk cache bypassed it entirely.  The
  failure is an **isolation concern** specific to the test setup, not a
  correctness concern in the production path.  In production there is no
  substitute output; `cache.put` stores only the actual model response to a
  real (model, prompt, params) triple.
- Rule: Test helpers that mock `_call_ollama_streaming` MUST also patch
  `cache.get` (returning None) and `cache.put` (no-op) to ensure the mock is
  actually reached.  See `tests/test_runner_span_scores.py` for the corrected
  pattern.  Production code requires no change.

## 13. unified-psutil Measures Whole-System Memory, Not GPU Allocation

- Threat: When no vendor-specific GPU tool (nvidia-smi, rocm-smi, intel-level-zero)
  is available, the telemetry layer falls back to `psutil.virtual_memory()` and
  emits `gpu_mem_source = "unified-psutil"`.  This value is **total system RAM in
  use** (OS + all processes + any GPU workloads combined), not a GPU-specific
  allocation counter.  On EVO-T2S at idle the figure was ≈7.0 GB of 63.5 GB
  total — reflecting OS and background process footprint, not GPU activity.
- Impact: Any result row carrying `gpu_mem_source == "unified-psutil"` will be
  silently misinterpreted as a GPU memory reading if it is pooled with rows from
  `nvidia-smi`, `rocm-smi`, or `intel-level-zero`.  Cross-source comparisons of
  `gpu_mem_mb` would mix fundamentally different quantities: VRAM-in-use (discrete)
  vs. whole-system-RAM-in-use (unified fallback).
- Rule: **Do not pool or directly compare `gpu_mem_mb` values across different
  `gpu_mem_source` literals.**  Filter result rows by `gpu_mem_source` before any
  memory-pressure analysis.  Rows with `gpu_mem_source == "unified-psutil"` may
  be used as a system-load proxy within the same hardware config but must never
  be treated as a GPU allocation measurement.
- Scope: Applies to any hardware config where `memory_architecture = unified` and
  no Intel Level Zero / rocm-smi path succeeds.  On EVO-T2S the Level Zero library
  loads but requires `ZES_ENABLE_SYSMAN=1` at `zeInit` time to expose memory
  modules; when that succeeds, `intel-level-zero` replaces `unified-psutil` as the
  source and this threat does not apply to those rows.
- Config flag: `memory_architecture` in `configs/hardware/*.yaml` is the first
  gate.  The `gpu_mem_source` field in each result row is the authoritative label
  for what the accompanying `gpu_mem_mb` value actually measures.

## 14. selfreport_arms.json — Arm Rates Cannot Receive the Unclassifiable-Row Correction

- Threat: `results/selfreport_arms.json` stores only pre-computed aggregate statistics
  (fabrication_rate, abstention_rate, n_total per arm per ratio). No individual row records
  are stored; the file contains no `done_reason`, `output`, or `classification_method` fields.
  The budget-exhausted rows identified in `results/stage_a_scale.json` (12 rows with
  `done_reason='length'` and `output=''`) allowed the fabrication rate for that file to be
  corrected by removing an identifiable set of truly-unclassifiable rows from the denominator.
  No equivalent correction is possible for selfreport_arms.json.
- Impact: The per-arm fabrication rates reported in `docs/FINDINGS.md` (arm1=80%, arm2=100%,
  arm3=76.7%) are based on the same scorer behaviour as the uncorrected stage_a_scale rates:
  `outcome='incorrect'` counts all wrong-answer rows including any budget-exhausted ones.
  If any arm's runs produced budget-exhausted rows, those rows inflate the fabrication count
  without the denominator correction that could have been applied had row-level data been
  stored. The arm2 result (100% fabrication, 0% abstention) is the most sensitive to this:
  if arm2 had budget-exhausted rows, the true rate could be below 100%, though the direction
  of arm2's perverse effect (eliminating abstention) would remain.
- Rule: Do not apply the unclassifiable-row correction to selfreport_arms.json rates. State
  them as reported (uncorrected) with a note that individual row data is unavailable.
- Mitigation: For any future self-report arm experiment, emit row-level result files alongside
  aggregate summaries so that `done_reason` and `output` are available for post-hoc correction.

## 10. Artifact Probe Selection Bias Toward Simple Retrieval

- Threat: Two of the ten artifact-bearing probes (art_03 and art_04) were redesigned during authoring because their original formulations required multi-step inference — art_03 asked for the maximum-CPU_HOURS job (argmax over a table) and art_04 asked which user performed a specified action (reverse actor lookup). Both models failed these questions reliably with 4,000 tokens of filler present, even when the artifact was intact, and redesign was necessary to achieve headroom. The final suite therefore consists entirely of direct retrieval lookups: given a key, return the value at that key from the artifact.
- Impact: The valid artifact suite is selected for probe types that both models can handle under the experimental condition. Multi-step inference over an artifact (filtering, argmax, reverse lookup) is not represented. The truncation result — that score drops to zero when the artifact is absent from context — is established for **retrieval-type artifact dependence**, not for inference-over-artifact dependence. The claim that truncation causes fabrication is sound within this scope but does not extend to inference tasks without a separate experiment showing the same probes pass headroom.
- Implications for the discreteness argument: the structured-vs-narrative contrast in the art_* suite tests whether different artifact *formats* behave differently under truncation, while both being retrieval tasks. It does not test whether structured artifacts require more inference steps than narrative ones, which is a separate question.
- Planned future work: Author a multi-step inference variant of the art_* suite (argmax, filter, chain-of-lookup) and repeat the headroom + truncation experiment. If these also show abrupt collapse, the discreteness argument extends to inference. If they show graded degradation even at partial-artifact, the mechanism differs and must be characterised separately.

## 15. thinking_enabled Label Not Operative in runner.py — Phase 1 Sweep Unconstrained

- Threat: `harness/runner.py` computes `thinking_enabled = model_variant == "reasoning"` (line
  711) and records the value in every result row. Before 2026-09-14 it never transmitted this
  as an API control: the payload sent to Ollama `/api/chat` did not include `"think": false`,
  even when `thinking_enabled=False`. The field was a metadata annotation, not a transmitted
  instruction.
- Affected data: three committed JSONL files produced by runner.py:
  `results/run_20260813T011126Z.jsonl` (40 rows, pilot, qwen3:4b-instruct),
  `results/run_20260813T021516Z.jsonl` (1100 rows, Phase 1 sweep, qwen3:4b-instruct),
  `results/run_20260818T000746Z.jsonl` (132 rows, Phase 1 replication, llama3.1:8b).
  Total: 1272 rows.
- Documented claims that rely on the label: `docs/EVALSET_DESIGN.md` §"Phase 1 Sweep
  Configuration" asserts `thinking_enabled=false` for `run_20260813T021516Z.jsonl`.
  This is the source file for Figures 4.1 and 4.2 (`PAPER_OUTLINE.md`) and the cha_04
  and lon_02 findings in `FINDINGS.md`. The claim is that thinking mode was inactive during
  these sweeps; the payload evidence says the mode was at the model's discretion.
- Evidence that the label can be wrong: `results/stage_a_scale.json` shows that
  `gpt-oss:120b-cloud` produced 1826–3637 characters of `message.thinking` content per call
  despite `"think": False` being present in those payloads. That is a different model and
  backend, but it demonstrates the flag is not universally effective.
- Impact on figures and findings: For qwen3:4b-instruct (1140 rows), the raw-chunk probe
  at warm model showed no thinking content for short prompts. Whether thinking fired on
  complex reasoning probes (rea_*, cha_*) at d=8000–32000 is unknown — the 1272 rows
  carry no `thinking_chars` field and thinking content was not captured. Score-based
  findings (categorical 1→0 cliff at d=2000 for cha_04, format failures at d=16000 for
  lon_02) are behavioral observations that are unlikely to be explained by a thinking phase
  on a 4B model, but the mode-consistency claim cannot be verified from existing data.
- Code fix (2026-09-14, forward only): runner.py now sends `"think": false` in the API
  payload when `thinking_enabled=False`, matching all other harness scripts. The row label
  and the transmitted flag are now derived from the same `thinking_enabled` value and
  cannot diverge again. A `thinking_chars` field is recorded on every new result row
  (total chars of `message.thinking` across all streamed chunks; 0 means suppression
  succeeded; None on cache hits).
- Existing rows (1272, unverifiable): the three committed JSONL files listed above carry
  `thinking_chars=None` because thinking content was not captured when they were recorded.
  The `thinking_enabled=False` label in those rows is a metadata annotation whose
  correctness cannot be checked retrospectively. It cannot be corrected by re-running
  because the original model-load state, seed ordering, and context depths would need to
  be replicated exactly.
- Rule: Any paper claim of the form "thinking was disabled for this sweep" MUST reference
  either (a) rows from 2026-09-14 onwards where `thinking_chars == 0` (verified at the
  API level), or (b) rows predating 2026-09-14 with a qualifying footnote that the claim
  rests on the row label only and was not verified at the API level.

## 16. Backend Confound — Mitigation by Vulkan Standardization

- Threat (original): The Blade 14 previously ran only llama.cpp build b1-f8def7fe1 (CUDA
  backend) while evo-t2s runs build b10970 (Vulkan backend). Different release versions,
  different compute backends, and different quantization implementations meant that
  cross-platform comparisons conflated hardware differences with kernel differences.
- Mitigation (2026-09-14): Vulkan build b10970 has been installed on the Blade 14 alongside
  the existing CUDA path. The CUDA path is retained. Both machines now share an identical
  Vulkan binary: llama-server.exe SHA256
  `0c8338ae5694f31db394ad3ea9578ba3de4a3ab35095d3ca6f16d74a02bdae35`; zip SHA256
  `f17091a433feb686d9e17378a8a2fc53a1437d64c1bf302ab6fb3072b4afcf0d`.
  All future cross-platform measurements will use the Vulkan backend on both machines.
- Residual confound (quantified): The CUDA backend is retained on the Blade to measure
  the backend delta directly: identical model, identical prompts, identical flags, CUDA vs
  Vulkan on fixed hardware. This converts the backend effect from an uncontrolled confound
  into a quantified term that can be subtracted from cross-platform comparisons.
- Prior Blade data: All measurements recorded under CUDA build b1-f8def7fe1 — including
  KV cache compression ratios 1.77× (f16→q8_0) and 3.24× (f16→q4_0) documented in
  docs/KV_MEASUREMENT.md, and all Stage C / ABSTENTION / filler-composition results — were
  taken under the CUDA backend and MUST NOT be compared directly to Vulkan b10970
  measurements without applying the measured backend delta. The KV ratios must be
  re-measured under Vulkan b10970 before being cited for any platform comparison.
- Rule: Every llama-server result row MUST record `build_id`, `backend` (one of: "cuda",
  "vulkan"), and `platform` (one of: "blade14", "evo-t2s"). These fields must never be
  omitted or defaulted; they are the only basis for separating arms in analysis.

## 17. Ollama Blob vs HF GGUF — Unverified Weight Identity

- Threat: The Ollama blob used for all existing repo results
  (SHA-256 `85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25fce54b18b9`,
  2,497,280,480 bytes) and the canonical HuggingFace GGUF used for all
  llama-server results
  (SHA-256 `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5`,
  2,497,280,256 bytes) differ by 224 bytes. The two files are not byte-identical.
- The difference is likely in GGUF metadata fields (e.g. tokenizer vocabulary
  strings, model card text, or key-value metadata entries) rather than in tensor
  weights, since a 224-byte difference is too small to alter any weight block in a
  Q4_K_M quantisation. However, this has not been verified by diffing the decoded
  GGUF structures. It is possible, though unlikely, that a metadata field influences
  inference behaviour (e.g. a rope_freq_base or context_length override embedded in
  the file).
- Affected data: All Ollama-path rows (runner.py sweeps, Stage C, ABSTENTION,
  filler-composition) used the blob. All llama-server rows (evo-t2s sentinel test,
  future sweeps) use the HF file. Any claim that compares scores or latencies across
  these two sets is comparing results from files that are not verified as identical.
- Rule: Until the 224-byte difference is characterised (e.g. by `gguf-dump` or
  equivalent), any cross-path comparison MUST carry a footnote citing this threat.
  The footnote must not assert the difference is harmless.

## 19. Fig 6.1 Model and Inference-Mode Confound — Stage C vs evo-t2s Run

### 19a. Different checkpoint: qwen3:4b-instruct vs Qwen3-4B hybrid

- Threat: Stage C (`stage_c_20260818T040408Z.jsonl`, 396 rows, blade_rtx4070) used Ollama `qwen3:4b-instruct`
  (blob sha256:85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25fce54b19b9, Ollama template has no
  `<think>` block — a dedicated non-reasoning instruct model).
  The evo-t2s Fig 6.1 sweep (`fig61_full_20260922T060942Z.jsonl`, 180 rows) used
  `Qwen3-4B-Q4_K_M.gguf` (filename has no `-Instruct` suffix — the hybrid reasoning model)
  via llama-server b10970 with `--reasoning-budget 0 --reasoning-format deepseek`.
  Evidence: every EARLY arm output in the Fig 6.1 run begins with `</think>\n\n`
  (the closing tag bleeds through even when `--reasoning-budget 0` suppresses thinking content),
  confirming the GGUF is the hybrid model. Stage C outputs for the same prompts are
  direct single tokens with no thinking artefact ("0.15", "A9").

- Impact: The two runs cannot be compared as replications. They differ in
  (a) checkpoint weights (qwen3:4b-instruct ≠ Qwen3-4B base), (b) inference mode
  (Ollama default temperature vs llama-server temperature=0), and (c) CoT suppression
  mechanism (none in Stage C vs `--reasoning-budget 0` in Fig 6.1). Any claim of the form
  "evo-t2s agrees with / differs from Stage C (Blade 14)" is comparing across two confounds
  simultaneously and is uninterpretable.

- Rule: Cross-platform comparisons between Stage C Blade data and evo-t2s Fig 6.1 data
  MUST NOT be made until a replication run exists on evo-t2s using the same checkpoint
  family (qwen3:4b-instruct or qwen3:4b with matched inference mode) as Stage C.

### 19b. max_tokens=128 output budget — LATE RAG scores are not position measurements

- Threat: The Fig 6.1 sweep set `max_tokens=128`. LATE arm RAG probes (rag_01, rag_02, rag_05)
  receive a full ~4000-token filler block followed by the artifact and question.
  The hybrid model with `--reasoning-budget 0` externalizes its suppressed reasoning as visible
  prose in the output ("Okay, let's see. The question is asking about…") and exhausts the
  128-token budget before emitting an answer. All LATE RAG rows at all budget_ratios
  including 1.20 (no truncation) have `finish_reason: "length"` and `tokens_out: 128`.
  Scorer takes the last non-empty line of the truncated prose, which is not the answer.

- Evidence that this is output-budget exhaustion, not a position effect: rag_02 LATE at
  ratio=0.55 scores 1.0 — a truncating condition where a shorter input prompt allows the
  model to reach the answer within 128 tokens. Score=0.0 at ratio=1.00 and score=1.0 at
  ratio=0.55 for the same arm on the same probe is non-monotonic and physically inconsistent
  with a position-pressure explanation; it is explained by the output budget decreasing with
  prompt length.

- Impact: The RAG LATE column in the Fig 6.1 run is not a measurement of position-driven
  quality degradation. LATE RAG scores=0.0 at ratios 1.20 and 1.00 should not be cited as
  evidence that LATE arm fails on RAG probes. The run does not establish whether the LATE
  arm would fail or succeed on RAG probes with adequate output budget.

- Rule: Do not use Fig 6.1 LATE RAG scores from `fig61_full_20260922T060942Z.jsonl`
  as a position-effect measurement for RAG probes. Rerun with max_tokens ≥ 512
  (or ideally uncapped) before drawing conclusions. In any figure or table citing
  this run, annotate all LATE RAG cells as "invalid — output budget exhaustion (128 tok)".

- Affected rows: all rag_01, rag_02, rag_05 LATE rows (54 of 180 rows total).
  EARLY RAG rows are not affected (finish_reason="stop", tokens_out=8-17 for most).
  Note: rag_05 EARLY also fails at all ratios (finish_reason="length"), same mechanism.

## 20. Char-Truncation Bias at Low Budget Ratios

- **Threat:** The harness truncates prompts by character count using a fixed heuristic of
  5.03 chars/token (from `context.py`). The actual chars/token ratio varies with content:
  filler (sequential integers) is token-dense; artifact and question text is less so. At
  low budget ratios the tail of the LATE-arm prompt after truncation contains proportionally
  more artifact and question tokens, which are denser than the dropped filler prefix.
  This causes the delivered token count to exceed the intended budget.
- **Observed:** In `fig61_stagec_full_20260922T203557Z.jsonl`, 9 of 396 rows fail the 5%
  positive-control tolerance: all at `r=0.40`, all LATE arm, all search (sea) probes
  (sea_01, sea_05, sea_06), over-delivering by 5.0–5.3% (~89 tokens over target of ~1716).
  RAG probes and higher ratios pass, consistent with filler comprising a smaller fraction
  of remaining context at those ratios.
- **Directionality:** Conservative — the model receives slightly more context than the
  nominal budget, so any observed score degradation at low ratios is not an artefact of
  under-delivering context.
- **Both arms carry this bias equally:** Stage C on the Blade used the same char-based
  truncation with the same heuristic. The evo-t2s replication used the same `context.py`
  `build_filler` and `left_truncate` methods. Cross-run score comparisons are therefore
  not confounded by this bias.
- **Planned fix:** Replace char-based truncation with token-accurate truncation via a
  `/tokenize` call per prompt in a future run. This was not done in stage C or the
  evo-t2s replication to preserve comparability.

## 18. Ollama Version Change — 0.32.9 → 0.34.0

- Threat: All KV-measurement and position-pressure results cited in `KV_MEASUREMENT.md`
  and `POSITION_PRESSURE.md` were produced on Ollama 0.32.9. The Blade 14 installation
  has since been updated to Ollama 0.34.0. Re-runs on the Blade will use a different
  runtime than the originals, and results cannot be compared without confirming
  version-to-version reproducibility.
- Known behaviour differences confirmed between 0.32.9 and 0.34.0 on Blade:
  - KV cache type bug: 0.32.9 did not propagate `OLLAMA_KV_CACHE_TYPE` to the bundled
    llama-server. 0.34.0 propagates it correctly as `--cache-type-k` and `--cache-type-v`.
    Any KV-cache-type sweep result from 0.32.9 must be treated as having used the default
    KV type regardless of what `OLLAMA_KV_CACHE_TYPE` was set to.
  - Context shift: 0.34.0 passes `--context-shift --keep 4` to the bundled llama-server
    by default. Whether 0.32.9 did the same has not been verified from a captured
    invocation.
- Rule: Do not run new Ollama measurements on the Blade and compare them to prior
  0.32.9 results without explicitly recording both versions and noting that version
  parity has not been established. Tag every result row with the Ollama version as
  measured at run time (e.g. `ollama --version` output), not assumed from install history.
