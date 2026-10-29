# Added pilot P11-HANG-REGIME: from one prompt to a regime (EXPLORATORY)

Registered 2026-10-20 (Task AZ), before any P11 run. NOT YET APPROVED to run
(each window needs Zach's OK; T2S cells also need Rithwik's block, which ends
midnight ET 2026-10-25). $0, local only. Runners do not read this file. It adds
a pilot; it changes no registered prediction, threshold, arm, entry set or
decision rule of DEMO-2, of pilots P0-P9 or of P10.

## Facts at registration
Four turn-watchdog-class stalls, all at multi_turn_long_context_123 turn 0, on
two devices, two GPUs and two drivers:
- XPS (Intel Graphics, Xe3, driver 32.0.101.8860; read 2026-10-20) x3:
  841b84e4 (2026-10-19, no watchdog, killed), 24c1abc0 (1,800 s watchdog),
  027a07b7 (P10 C1, prefix caching off, 600 s watchdog).
- T2S (Arc B390, driver 32.0.101.8509; read 2026-10-20) x1: bbdd1c0d (600 s).
- py-spy (24c1abc0, 027a07b7): main thread in clEnqueueNDRangeKernel ->
  igdrcl64.dll waiting on a GPU sync object, no error.
- Which step of turn 0 stalls is UNKNOWN. Gold path: step 0 = 2,189 tokens
  (prompt sha256 d3b347bc...), step 1 = 34,050 tokens (sha256 4e965f37...),
  whose new part is get_stock_info's result, about 31,665 tokens of
  comma-separated floats (Qwen splits digits, about 1.2 chars per token).
- Not a hang: the synthetic resident probe fails at 18,500 cached tokens with a
  turn-2 RuntimeError (a5d19dcc). In 15367c3a arm A held step prompts of
  34,417 / 43,245 / 67,947 tokens on other entries without a stall
  (descriptive; those entries were not size-matched).

## Instruments (built in Task AZ, default off)
- Runner: `--step-log` (launch_h1 -StepLog). Before each local generate one
  JSON line (entry, turn, step, prompt tokens, estimated new tokens) is
  written, flushed and fsynced to step_log.jsonl and to stderr.
- Repro: `tools/repro_hang123.py`. `build` (Mac) wrote
  derived/repro_hang123/inputs_e123_t0.json, sha256
  d12142f89702f0ab92a48148abaa034760bb20c87558b58945a92a29a405f2f0 (two gold
  steps; step 0 matches the Task AY pin). `run` (host) is a plain
  `LLMPipeline(model, "GPU")`, no property, no SchedulerConfig, greedy,
  max_new_tokens 512, apply_chat_template false; one flushed PRE_GENERATE line
  before each generate. Pass `--inputs-sha256` so a changed file refuses.
- Wall clock: the window script waits 600 s per cell; on expiry it runs
  `py-spy dump --native` (XPS only; py-spy is not approved on the T2S), then
  stops the process, records outcome hang, and reboots before the next cell.

## Cells
Let S be the stalling step and N the new-token count at S (prompt tokens at S
minus prompt tokens at S-1 minus the tokens generated at S-1; for S = 0,
N = prompt tokens). The variant inputs below are built on the Mac by a
`variants` subcommand of tools/repro_hang123.py, to be added and committed,
with every variant's sha256 listed in a follow-up file, before any phase-M or
phase-S cell runs. Its definitions are fixed here. Every variant keeps the
real prompt up to the start of the new part at S and replaces only the new
part's body (inside the tool response for S >= 1). The length is measured
with the pinned HF tokenizer and trimmed at a token boundary to the target
count, +-0.5%.

Phase L (localisation; real input, both hosts):
- L1 repro, all steps. L2 runner, arm A, entry 123 alone, -StepLog
  -TurnWatchdogS 600 (+ py-spy on the XPS).
- S = the step of the last PRE_GENERATE / STEP_PRE_GENERATE line with no
  POST_GENERATE / completed step. If L1 completes and L2 hangs, S comes
  from L2 and phases M/S use the repro anyway (recorded as a limit).

Phase M (same length at S; target N tokens):
- M1-M3 other real BFCL content. Sources fixed by rule: the 3 entries of
  BFCL multi_turn_long_context (BFCL file order, excluding 123) whose gold
  path has the largest single tool result. Their results are concatenated in
  call order and repeated if shorter than N.
- M4 random tokens. Token ids drawn uniformly from the HF vocabulary,
  excluding special and added tokens, seed 20261020. Decoded, re-tokenized
  and trimmed.
- M5 random floats in 123's own format, seed 20261020. A JSON object with the
  same keys as 123's get_stock_info result; every float list is replaced by
  uniform random values in [200, 250] with 2 decimals, list lengths scaled to
  hit N.
- M6 the real input again (repeat of L1 at S only, `--only-step S`).

Phase S (length sweep at S): fractions f in {0.5, 0.75, 0.9, 1.0, 1.1, 1.25,
1.5} of N.
- Real: 123's own new part, truncated at f < 1, or extended at f > 1 by
  appending its own float lists again.
- Random: the M4 generator.
- 14 cells.

## Hosts and order (non-hanging first)
- T2S (preferred for hang-prone cells; it rebooted in about 90 s on
  2026-10-20). L1, L2, then M4, M1, M2, M3, M5, then S in ascending f (random
  before real at each f), then M6 last.
- XPS (post-hang reboots took 15+ min and once did not come back): L1, L2,
  M4, M5, M6 only.
- A hang is followed by a reboot before the next cell (T2S: REBOOT_OK=1;
  Tailscale unattended worked on 2026-10-20).

## Classification (fixed now; per host, then across hosts)
- Size class = 0.9 N to 1.1 N.
- "regime" if, at the same S, hangs occur for >= 2 distinct inputs (M1, M2,
  M3, M5, the real input and the random sweep count as distinct), or for
  random text (M4 or the random sweep) in the size class.
- "input-specific driver bug" if the real input hangs (L1 or M6) and every
  control in the size class (M1-M5, random 0.9/1.0/1.1) completes.
- "not reproduced" if neither L1 nor M6 hangs on that host. No verdict on
  that host; the runner-only hang (L2) is recorded.
- Mixed results (for example, one control hangs out of five) count as
  "regime" under the >= 2 distinct inputs rule only when the real input
  also hangs. Otherwise they are reported cell by cell, with no class.
- Size threshold (descriptive): the smallest f that hangs in each sweep.
- Across hosts: the same class on both hosts is stated as such; different
  classes are stated per device and driver.
- Every hang counts once, with its py-spy dump where taken. No cell is
  retried.

## Cost (DERIVED)
A completing cell is about 4 min (load about 5 s; the 34k prefill about
140 s on the XPS, scaled from 2b3316b6; 512-token decode). A hanging cell is
600 s plus a reboot (T2S about 2 min, XPS 15+ min).
- T2S: 23 cells, about 2.5-4 h by the number of hangs.
- XPS: 5 cells, about 1-1.5 h.

## Framing (Prof. Hao)
- Driver/vendor behavior is evidence, not the headline. The paper states
  the regime (a silent device stall that only a wall-clock deadline
  detects) and its scope.
- The repro and inputs are kept for an Intel driver report after submission.
  It is not filed before the data freeze.
