# State

A handoff note, updated at the end of every working session, so the project survives a context reset
without re-deriving status from memory. Read this first. Everything here is checked against real files
as of the date on each entry; it is not carried forward from a prior session's own claims without
re-verification.

Machines: evo-t2s (Intel Core Ultra X7 358H, Arc iGPU, Vulkan) and evo-x2 (AMD Ryzen AI Max+ 395 Strix
Halo, Radeon 8060S iGPU). As of 2026-10-06: evo-t2s is with Zach, no SSH, watchdog disabled, excluded
from sync, until told otherwise. All machine-time work is on evo-x2.

## What exists (built, real, file-backed)

- **Workload pack** (`results/workload_pack/`): 400 items across 5 families (longdoc_qa 120,
  function_calling 80, gsm8k 80, r2_sessions 60, trace_length_mix 60). Self-check (`grade.py`, run
  directly 2026-10-06): `GRADE-CHECK PASS: 400/400`. Trace-weighted prompt-length stats registered
  (`pack-trace-weighted-stats`, `docs/NUMBERS_REGISTER.md`). Known limitation on record in the pack's own
  README: families (d) r2_sessions and (e) trace_length_mix were built with independently-written
  generators rather than reusing `harness/t2s_r2_session_growth.py` / the real trace parquet -- a real
  follow-up, not a blocker.
- **X2 outcome-table harness** (`harness/x2_outcome_table.py`): running block covering the workload pack
  (minus r2_sessions) across 6 models x 2 runtimes (ollama_default, llama_server). As of 2026-10-06, 108/340
  items done, but **found invalid this session**: thinking-mode contamination on reasoning models and a
  server-race producing false connection-refused errors (see "Open decisions" and the 2026-10-06 session log
  below) -- stopped for an overhaul, not currently running.
- **MX2 validation** (`harness/mx2_validation.py`, `results/mx2_validation.jsonl`): done, reclassified
  with the correct per-process GPU Shared Usage counter. llama-3.3-70b: SILENT_SPILL confirmed at
  n_ctx=115200 (+13,397 MiB over baseline). qwen3-32b: no clean (non-clamped) spill point exists below its
  effective ceiling. Crash boundary did not reproduce for either model in isolation within a 60s start
  timeout -- re-test with a longer timeout before calling it a hang (see corrections, item 4a below).
- **K2** (`harness/t2s_k2_pressure.py`): `awe_balloon`/`pageable_touch` arms ran 18h on evo-x2
  (2026-10-04/05), 3/4 models complete, 13 kill-criterion hits recorded. Pause-and-resume arm (arm d)
  never ran -- it is gated behind the per-model pressure-arm loop finishing for every model, which the
  deadline cut off first.
- **Kappa study** (`results/labeling/`): 0.55 (moderate) on the original 150-row sample, 1.0 after two
  real scorer fixes in `evaluation/outcome.py` -- in-sample on the same rows the fixes were derived from
  (see corrections, item 4b: a new, independent sample is needed).
- **Physics-based budget model** (`analysis/numbers_register.py`): `ttft-physical-fit-per-machine`,
  `ttft-cross-machine-transfer`, `ttft-few-point-calibration` all VERIFIED in the register. Does not yet
  cover the first-principles A-24 budget boundary (weights + KV bytes/token x n_ctx + compute buffers vs
  47,866 MiB, predicted vs measured flip point) -- that is a distinct, not-yet-built analysis (corrections,
  item 4c).
- **R1 cross-machine agreement**: 96.1% text, 99.7% score (evo-t2s vs evo-x2, 337 paired rows).
- **PX2 full ratio table**: VERIFIED, max TTFT ratio 1.129x.
- **Demo spec** (`docs/DEMO_SPEC.md`) and **Paper 1/demo plan** (`docs/PLAN_PAPER1_DEMO.md`): both exist.
  The plan doc is dated 2026-09-30 and is stale on several items (see "Corrected from the stale plan doc"
  below) -- read it for structure, not for current status.
- **Cloud client stub** (`src/cloud/client.py`): exists, `CLOUD_API_KEY` unset -> `BLOCKED: cloud key`.

## What is specified but not built

- **R2 (two-step agent-session design)**: does not exist as a harness. `harness/t2s_r2_session_growth.py`
  is a DIFFERENT, earlier single-call-per-turn design; it is not the two-step (tool-call turn, then scored
  final-answer turn) design item 2 of the 2026-10-06 session specifies. Spec is being written now to
  `docs/R2_DESIGN.md`.
- **x2_llm_judge**: no harness file exists anywhere in the repo. `blocked_not_built` in queue_state.json is
  accurate.
- **Half-context three-marker probe**: no committed result file matches this description anywhere
  (`results/*.jsonl`, `docs/*.md` searched). An earlier session's claim that this was "already done" does
  not hold up against a file search -- never actually run as a distinct artifact.
- **Local outcome tables, both machines** (`results/outcome_table_*.json`, Oct 14 deliverable per
  `docs/PLAN_PAPER1_DEMO.md`): do not exist.
- **Demo-specific workload-description set for the input form** (distinct from the 400-item workload pack
  above, which is real): `docs/PLAN_PAPER1_DEMO.md`'s own Oct 7 "workload pack" line item refers to this
  demo-form artifact, not the 400-item pack -- not started. This was the source of a milestone-table error
  corrected 2026-10-06 (see below).

## Running / in progress as of 2026-10-06

See the 2026-10-06 session log at the bottom for the detailed replan. In short: outcome table stopped for
an overhaul (thinking mode, race tagging, validity gate, error-cause breakdown); queued next, in order:
the half-context probe, MX2 spill-cost-with-statistics, R2 validation, then the fixed outcome table.

## Open decisions (need a person, not a file search)

- Whether `docs/TIMELINE.md`'s ~Dec 14-15 ISPASS estimate or `docs/PLAN_PAPER1_DEMO.md`'s given Dec 2 date
  is the authoritative submission deadline.
- Whether to keep `kappa_sample`'s `annotator_claude` labels as a standing second channel, or replace them
  entirely once `ritz_spotcheck_30.csv` is labeled (still 0/30 as of 2026-10-06).
- Whether evo-t2s stays excluded from sync/SSH past this session (currently: yes, until told otherwise).

## 2026-10-06 session log

- Corrected a milestone-table error: the "workload pack" was reported "not started" from a stale reading
  of `docs/PLAN_PAPER1_DEMO.md`'s Oct 7 table, which is itself talking about a different (demo-form)
  artifact. The real 400-item pack is built and self-check passes 400/400 -- verified live by running
  `results/workload_pack/grade.py` directly, not assumed.
- Found and fixed a real race in `host_config.stop_ollama_server`/`x2_outcome_table.run_one_ollama`: 409 of
  ~650 weekend `ollama_default` calls hit connection-refused because the server's own kill was not
  confirmed before the next call's start check. Fixed, deployed, restarted once, then stopped again for
  the broader validity overhaul below once thinking-mode contamination was also found.
- Found thinking-mode contamination on reasoning models (qwen3-8b/14b/32b) via a live raw-response check
  (`llama-server` built with no reasoning-suppression flag) -- see the commit introducing this file's
  sibling fix for the exact mechanism and the fix applied.
- Stopped `x2_outcome_table_v2` for a full validity overhaul (thinking-mode disable, race/thinking row
  tagging, a canary validity gate, per-cause error classification, a trimmed model list).
- Began writing `docs/R2_DESIGN.md` for the real two-step R2 design (not yet in a prior session).
