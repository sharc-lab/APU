# Next after Zach's evo-x2 runs

Operator queue, 2026-11-07. Starts after Zach's evo-x2 runs (Parts 1 to 4 at his splits) are finished and zipped and
the X2 is restored (64 GB split in the BIOS, Ollama 0.34.4, Remote Desktop off). Each item that runs anything is
pre-registered in docs/EXPERIMENTS.md with its target before it runs; numbers come from files only.

## Queue

1. **Long-task diagnosis (traces only).** How many of the 30 long tasks pass 40,960 tokens; the failure cause per
   model (truncation, wrong action, format).
   Status 2026-11-07: done, row `long-task-failure-diagnosis`. All 10 source OpenHands traces pass 40,960 Qwen tokens
   (9 of 10 in Llama tokens), but the synthetic tasks match step counts, not token counts, and no run reached its
   40,960 window (max prompt 12,447 tokens for qwen3:32b, 10,749 for llama3.3:70b). qwen3:32b: wrong action 30/30.
   llama3.3:70b: format 30/30. No truncation.
   Update 2026-11-09: the 70B "format" failures were caused by Ollama's llama3.3 chat template (tool definitions
   rendered only on a trailing user message, so the model saw no tool schema after step 1), not by the model. With
   the pre-registered harness fix (`--tool-prompt persistent`), llama3.3:70b solved 3/30 held-out long tasks; its
   remaining failures are wrong actions [REG longtask-70b-toolfix].
2. **If truncation: llama3.3:70b big-window rerun** at a 96 GB split, Ollama 0.34.4 (to match the earlier runs), local
   only, registered next to the 40,960 row (`long-task-scale-local`).
   Status 2026-11-07: not triggered (item 1 found no truncation on this task set). A task set matched on tokens, not
   steps, would need its own pre-registration before this could apply.
   Update 2026-11-09: the token-matched set exists (longtask-tokens-v1, exploratory, 3 tasks): llama3.3:70b at a
   window holding the whole prompt (131,072 loaded) failed 3/3 on wrong actions, so for 70B capability fails first;
   the 32B fitting arm was not a test (Ollama 0.34.4 loaded 40,960 for a 98,304 request) [REG longtask-tokens-v1].
   The real memory-vs-capability test is longtask-tokens-v2 (a 256K-native model; pre-registration filed, not run).
3. **NEW: per-turn routing inside long tasks.** Small appends stay local; large tool outputs go to the cloud. Compared
   with route-at-start and all-cloud on the long-task set. Replay first, USD 0; pre-register before computing.
   Goal: any long-task saving at equal accuracy.
4. **Cut the 52% false escalations.** CLOSED 2026-11-09, no change adopted. In the live 8192 run (11/21 right tasks
   escalated, row `post-run-check-live-8192`) L1 never fired alone: L2 (the earliest-example render check, which reads
   turn 1, the first turn the window drops) fired on all 11 [REG l1-false-escalation-candidates]. Every candidate rule
   that reaches the 10% target on the held-out seed loses 4 of the 9 caught failures (final 26/30 instead of 29/30);
   after the overflow, 8 of 19 local turns were wrong and 11 right, and no layer separates them. Levers left: stay
   under the window (the operating point) or a stronger L3.
5. **Haiku verification tier live run** (Run V1, about USD 0.11 estimated, USD 2 job cap). Needs the operator's go.
6. **XPS 16 GB comparison with Zach's kit** (docs/XPS_DEMO_KIT.md): local default, every-call standard hybrid, Spillway.

## Positioning for all write-ups

- Reliability first: silent failures caught. At the 8,192 window the pipeline gets 88/90 against 84/90 for the
  standard hybrid (every-call check) [REG pitch1-standard-hybrid-8192].
- Savings only on short tasks so far.
- Long-task savings: 0% to date. qwen3:14b and qwen3:32b solved 0/30 long tasks [REG long-task-scale-local];
  llama3.3:70b 3/30 after the pre-registered tool-template fix [REG longtask-70b-toolfix], and 0/3 token-matched
  tasks at a window that held the whole prompt [REG longtask-tokens-v1]. No long-task saving is claimed.
- False escalations are not reduced: 11/21 in the live 8192 run, and no rule change cuts them without losing caught
  failures [REG l1-false-escalation-candidates].
- The live 8192 post-run check is reported as 29/30 with the turn 24 handoff explanation, not regraded
  [REG post-run-check-live-8192].
