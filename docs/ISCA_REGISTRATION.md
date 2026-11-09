# ISCA abstract registration (draft, not submitted)

Prepared 2026-11-09 from docs/PAPER1_DRAFT.md and the numbers register. Nothing has been submitted. Double-blind:
no author, lab or company names below.

## Title

Silent by Default: How Local LLM Agents Fail When Context and Memory Run Out on AI PCs

Alternatives:
1. Where Local Agents Break: Context Overflow, Memory Pressure and Silent Failure on AI PCs
2. No Error, Wrong Answer: Characterizing and Catching Silent Failures of Local LLM Agents

## Abstract (about 200 words)

When a local LLM agent's conversation outgrows the context window, Ollama trims it and still reports success, while
llama-server at the same window returns an error. We characterize this failure mode with pre-registered experiments on
an AMD APU, an NVIDIA laptop GPU and an Intel mini PC, spanning unified and discrete memory. On a tool-using task
workload, qwen3:14b's wrong-task rate rises from 11.3% (17/150) at its default window to 42.0% (63/150) at 4096
tokens, and every wrong task is silent. Runtime defaults, not hardware, set the window: the same runtime loaded 4096
tokens on one machine and 131072 on another. Memory pressure on unified memory does not degrade gracefully: decode
stayed within 2% of baseline, and the one failure in 317 requests was a GPU device-lost crash, not a slowdown. In a
collaborator's comparison against a LiteLLM-based fallback stack, wrong turns were 80% versus 15%. We then evaluate a
post-run check that detects silent failures and escalates only those tasks to a cloud model: on 90 jobs at an 8192
window it reaches 88/90 correct against 84/90 for a standard prompt-length hybrid, and a live run caught 9 of 9
failures (29/30 correct). On long-horizon tasks, no local model up to 70B solved any of 30.

## Suggested topics

Match these to the submission site's own topic list before registering.

- Machine learning systems: inference serving and runtimes
- Performance evaluation and workload characterization
- Memory systems: unified memory and capacity limits
- Mobile, edge and client computing (AI PCs)
- Reliability and fault detection

## Sources for every number

| claim in the abstract | register row (docs/NUMBERS_REGISTER.md) |
|---|---|
| Ollama trims and reports success; llama-server returns an error | xruntime-llamacpp-wrong-rate, blade-k1-layer-split |
| 11.3% (17/150) to 42.0% (63/150), every wrong task silent | x2-evalset-tasks-real-v2-wrong-rate |
| 4096 on one machine, 131072 on another | T2S-vs-X2-default-ctx |
| decode within 2% under memory pressure (0.984x to 1.000x, 7 runs per arm) | K2-x2-decode-by-arm |
| one failure in 317 requests, Vulkan device lost | K2-x2-clean-failures |
| LiteLLM comparison, 80% vs 15% wrong turns | **UNREGISTERED**: Zach's result; no run IDs yet. Remove or replace with a register row before final submission |
| 88/90 vs 84/90 at 8192, 90 jobs | pitch1-standard-hybrid-8192 (replay of recorded runs) |
| live run 9/9 caught, 29/30 correct | post-run-check-live-8192 (29/30 not regraded; turn 24 is a handoff artifact) |
| no local model up to 70B solved any of 30 long tasks | long-task-scale-local |

Notes: the memory-pressure numbers come from the AMD APU (evo-x2 K2 run). The Intel mini PC memory-pressure findings in
docs/FINDINGS.md (flat TTFT, loud device-lost failure) are not register rows and are not used here. Under the
"reliability first" positioning, short-task savings are not in the abstract, and long-task savings are 0% to date.
