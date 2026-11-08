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
2. **If truncation: llama3.3:70b big-window rerun** at a 96 GB split, Ollama 0.34.4 (to match the earlier runs), local
   only, registered next to the 40,960 row (`long-task-scale-local`).
   Status 2026-11-07: not triggered (item 1 found no truncation on this task set). A task set matched on tokens, not
   steps, would need its own pre-registration before this could apply.
3. **NEW: per-turn routing inside long tasks.** Small appends stay local; large tool outputs go to the cloud. Compared
   with route-at-start and all-cloud on the long-task set. Replay first, USD 0; pre-register before computing.
   Goal: any long-task saving at equal accuracy.
4. **Cut the 52% false escalations.** In the live 8192 run, L1 fires on every turn after the window overflows (11/21
   right tasks escalated, row `post-run-check-live-8192`). Propose a fix, evaluated on a held-out seed only.
5. **Haiku verification tier live run** (Run V1, about USD 0.11 estimated, USD 2 job cap). Needs the operator's go.
6. **XPS 16 GB comparison with Zach's kit** (docs/XPS_DEMO_KIT.md): local default, every-call standard hybrid, Spillway.

## Positioning for all write-ups

- Reliability first: silent failures caught. At the 8,192 window the pipeline gets 88/90 against 84/90 for the
  standard hybrid (every-call check) [REG pitch1-standard-hybrid-8192].
- Savings only on short tasks so far.
- Long-task savings: 0% to date (no local model has solved a long task: qwen3:14b, qwen3:32b and llama3.3:70b each
  0/30 [REG long-task-scale-local]).
- The live 8192 post-run check is reported as 29/30 with the turn 24 handoff explanation, not regraded
  [REG post-run-check-live-8192].
