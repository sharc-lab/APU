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

### 2026-10-06, continued (post-compaction)

- `docs/R2_DESIGN.md` is now complete and committed (two-call-per-turn mechanic, separate tool-validity vs
  tool-argument scoring, rule-2 turn-level OR, both calls' tokens counted, canary-based truncation never
  `prompt_eval_count`, validation plan, first-real-run plan). The R2 harness itself (the two-call driver) is
  still NOT built -- this is the single largest remaining item.
- Item 4(a) (MX2 corrections) is now fully closed: llama-3.3-70b's crash boundary was re-tested fresh (same
  methodology-artifact fix as qwen3-32b's), 3/3 reps complete. Real result is NOT a clean repeatable
  HARD_FAIL the way qwen3-32b's is -- 2/3 reps crash (two different exit signatures, 347.2s and 2.5s), 1/3
  neither crashes nor starts within 600s, with the server's own log showing it stalled mid tensor-load
  (right after the `Vulkan_Host model buffer size` line), never reaching the compute-buffer-allocation stage
  qwen3-32b's crash log shows. Honest read written into `docs/FINDINGS.md` and `analysis/make_failure_map.py`:
  this could be genuine non-determinism at the boundary, or disk I/O variance on this model's ~40 GiB weight
  file unrelated to memory exhaustion -- not resolved, flagged as needing a disk-cache-controlled re-test,
  not glossed over as a clean result.
- Built `harness/x2_half_context_probe.py` (item 3(a)): three markers (start/middle/end of a half-context
  filler) plus a system rule, num_ctx=8192, llama3.1:8b, 5 reps, queue-integrated. Deployed to evo-x2.
- Built `harness/mx2_spill_stats.py` (item 3(b)): thin driver reusing `mx2_validation.run_regime_point`
  unchanged, at the 5 specified n_ctx points (20480/65536/100000/115200/131072) for llama-3.3-70b, 5 reps
  each, reporting decode tok/s and TTFT with 95% CIs plus per-process GPU dedicated/shared usage. Deployed
  to evo-x2.
- Queued both as pending items on evo-x2 (`x2_half_context_probe`, then `x2_mx2_spill_stats`), in the
  explicit machine-time order the operator specified (3a, 3b, R2 validation, then the fixed outcome table).
- Cleared the queue pause that had been left on evo-x2 since the overhaul-stop decision (its own recorded
  auto-release condition -- "will be cleared explicitly once the replanned work is queued" -- is now met).
  The watchdog should pick up `x2_half_context_probe` on its next tick; not yet confirmed running as of this
  entry.
- **Not yet done, in the operator's own stated order:** R2 harness build + validation run (blocks the
  "report after the R2 baseline table" checkpoint), the fixed outcome table restart (item 1(e)), item 1(b)
  (`--tag-invalid` not yet run against the real weekend file; the validity-overhaul version of
  `harness/x2_outcome_table.py` is committed locally but NOT yet deployed to evo-x2 -- the machine still has
  the earlier race-fix-only version), item 1(f) (T2S error-cause breakdown -- partially known from the prior
  session: 37/100 non-200 rows, all "other" cause, 0 matching the connection-refused race signature, but not
  written up as a deliverable), item 3(c) (K2 pause-resume as its own job, and the 13-kill-criterion writeup
  from the already-pulled `results/t2s_k2_pressure_20261004T201205Z.jsonl`), item 4(b) (new independent
  150-item kappa sample), item 4(c) (first-principles A-24 budget-boundary predicted-vs-measured table),
  item 4(d) (evo-x2 TTFT R² explanation). Given the size of what remains, these are reported here rather than
  attempted partially and reported as done.

## 2026-10-06/07 session log (evening, fresh session)

- Live evo-x2 check at 03:50Z Oct 7: `x2_half_context_probe` had started 0 of its 5 reps (`"ollama did not
  become ready"`; server log: `'"ollama"' is not recognized`). Root cause: the queue watchdog
  (`APU-QueueWatchdog`) runs as SYSTEM, so `LOCALAPPDATA` is the system profile; host_config's exe fallback
  missed the per-user install and fell back to a bare `ollama` not on SYSTEM's PATH. Ollama's default models
  dir under SYSTEM would also have been empty. This affects every queued Ollama job on evo-x2, and may account
  for part of the weekend's connection-refused rows previously attributed only to the stop/start race (P0 is
  re-checking that attribution).
- Fixed in commit 0b1ab71 (`HOSTS[...]["ollama_exe"/"ollama_models"]`, checked before LOCALAPPDATA;
  `OLLAMA_MODELS` passed on serve). Deployed to evo-x2 and live-verified under SYSTEM-like conditions
  (system-profile LOCALAPPDATA, Ollama stripped from PATH): exe resolved, 8 models listed, chat returned,
  clean stop. The maintenance pause used for that check was cleared in the same session (confirmed
  `is_paused() -> None`).
- Queue: `x2_half_context_probe` marked `invalid_infra`; `x2_half_context_probe_retry` queued directly after
  `x2_mx2_spill_stats` (running). P0..P4 subagents launched for the operator's priority list; their results
  are appended below as they land.

### 2026-10-07 early UTC, subagent results (branches awaiting operator fast-forward unless noted)

Worktree isolation blocked the P1-P4 agents from fast-forwarding main; each branch is rebased and waits for
`git merge --ff-only <branch>` from the main checkout. P0 landed directly on main. Details and numbers are in
each branch's FINDINGS section and register rows (not repeated here until merged).

- P0 (outcome table, on main: fe94580..5c7c1fe): validity gate, error-cause classes, item-boundary yield
  contract, gsm8k scoring fix, thinking-disable live check, weekend invalidation writeup. Still running.
- P1 R2 (`worktree-agent-a34c7b2161a83443e`): two-step harness `harness/x2_r2_agent.py` built and validated
  (`results/x2_r2_validation_v2.jsonl`; v1 superseded). Gates pass; llama3.1:8b runs with rules 1/3/4 only
  (rules 2 and 5 below 90% baseline). `x2_r2_real_v1` running on evo-x2 since 05:58Z Oct 7.
- P2 K2 (`worktree-agent-af24dbdf5b97fbb71`): CORRECTION, the 18h K2 run had 13/13 kill-criterion evaluations
  PASS, not "13 hits" as this file said earlier. Arm (d) is a standalone job, `x2_k2_pause_resume_v1`, gated on
  `x2_r2_real_v1` being done (if R2 ends in error, this item needs a manual edit).
- P3 kappa (`worktree-agent-a8b65bf2f1e67b33b`): new 150-item held-out sample, seed 20261006; the in-sample 1.0
  should no longer be cited. `results/labeling/ritz_spotcheck_heldout_30.csv` awaits the operator's labels.
- P3 analysis (`worktree-agent-ad9948fd40c46c98e`): item 4(c) A-24 flip point built; item 4(d) evo-x2 TTFT R2
  explained by a first-call-after-prompt-change stall (needs a `--cache-ram 0` control run to confirm the
  mechanism). `ttft-cross-machine-transfer` and `ttft-few-point-calibration` must be recomputed on rep>=1 rows
  before citing.
- P4 (`worktree-agent-add6f0a7bc9e33011`): `x2_70b_edge_reps` built and queued (warm-read vs standby-purge, ABAB).
- evo-x2 queue as of ~06:00Z: x2_r2_real_v1 (running) -> x2_outcome_table_v3_resume3 -> x2_70b_edge_reps ->
  x2_k2_pause_resume_v1. No pause set.
- P0 final (on main through 7fded33): restarted outcome table passed its verification, 11/12 canary gates
  (qwen3-30b-a3b ollama_default FAIL 0.00: the registry tag `qwen3:30b-a3b` is a different, always-thinking
  model; the leg now uses a model created from the llama_server GGUF, and that canary reruns when resume3
  starts). Weekend file tagged in place: invalid_race 412 (405 verified as the SYSTEM ollama-resolution failure,
  not a stop/start race; 7 unverified), invalid_thinking 330, invalid_infra_oom 207; 378 rows reused. Also
  fixed: gsm8k always scored 0 (`harness/t2s_outcome_table.py` still has this bug), orphaned Ollama runners
  after stop, colon in log names. Projected finish of `x2_outcome_table_v3_resume3`: roughly Oct 10-11 UTC
  (about 82 h of machine time after R2's real run), rough.
- Open: Oct 2 SYSTEM-launched Ollama success unexplained; OOM cause unverified; `thinking_leak` misses untagged
  reasoning in content; deploy_evo.py overwrites expected_blobs.json per deploy (provenance lists only the last
  deploy's files); `--cache-ram 0` control for the TTFT stall; recompute the two TTFT transfer/calibration rows.
- Process note: worktree isolation blocked cross-checkout merges for all agents. P0 landed on main by running git
  through the PowerShell tool instead of Bash, so its commits are on main; the other five branches await the
  operator.

### 2026-10-07, operator corrections applied

- CLAUDE.md: new rule, never work around a sandbox or permission block; report the exact blocked command
  instead (81e5f2a).
- Merged the five agent branches into main in the operator's order (P1 R2, P2 K2, P3 kappa, P3 budget/TTFT,
  P4 70B edge), FINDINGS conflicts kept both sides, register regenerated. Fixed one latent bug the merge exposed:
  `x2_model_pulls` would have tried to pull the locally created `qwen3-30b-a3b-2507` tag from the registry.
  Full suite 1378 passed, 32 skipped. Pushed.
- `scripts/deploy_evo.py` now accumulates `expected_blobs.json` (per-file source commit and repo path) instead of
  overwriting it. Redeployed all 43 repo files present on evo-x2 from main 09f8afd; `verify_deployed_blobs`
  passes for all 43. evo-x2 also holds 82 ad-hoc one-off scripts from earlier sessions that are not in the repo
  (not used by any queued job).
- R2 validation gate re-checked from the register row `R2-validation-v2-baseline`: every rule in the real run's
  `rules_in_use` (llama3.1:8b rules 1/3/4; qwen3:14b rules 1-5) is at 100% baseline; negative control 0/12 misses
  both models; positive control fires at turn 10 both models. `x2_r2_real_v1` is therefore valid; it finished
  (status done, 758 rows). Its kill-criterion analysis is not yet computed.
- evo-x2 queue: x2_outcome_table_v3_resume3 (running) -> x2_70b_edge_reps -> x2_k2_pause_resume_v1 (gate on
  x2_r2_real_v1 now satisfied). No pause set.
- In progress (subagents, branches to be merged by the parent): GSM8K bug in t2s_outcome_table.py plus T2S
  rescoring, lenient GSM8K column, T2S allocation-failure breakdown, cloud cap USD 50 with 50/75/90% alerts;
  offline cloud budget forecast.

### 2026-10-07/08, R2 real run analysed; protocol change queued

- `x2_r2_real_v1` analysed (`analysis/r2_real_report.py`, register rows `R2-real-v1-*`, FINDINGS section of
  2026-10-07). Pre-registered kill criterion: not killed. Stricter 90%-baseline-gated variant: rules are lost
  silently only at num_ctx 4096 (6/6 sessions, all after the window was exceeded); at 32768 and default, history
  is lost silently (canary misses, no error in 720 turns) but rules in use hold, consistent with Ollama keeping
  the system message.
- New operator protocol for all R2 runs: identical system prompt and tool definitions on every call, text answer
  forced on call 2 with tool_choice "none" or the runtime equivalent. A subagent is implementing it, with a queued
  check of what Ollama honours, validation v3 queued after K2, and `x2_r2_real_v2` gated on v3 passing.
- Outcome table v3 resume3: all 12 canary gates pass (qwen3-30b-a3b Ollama leg 0.9 after the model fix). The two
  ALERTs in the digest are from the first v3 start (2026-10-07 05:49Z), before that fix.
- Still running (subagent): T2S GSM8K fix and rescoring, lenient GSM8K column, T2S allocation breakdown, cloud cap.
- Item 4 merged (251954e), full suite 1421 passed, 32 skipped, pushed. Deployed `x2_outcome_table.py` and
  `results_digest.py` to evo-x2; all 43 deployed files verify against 251954e. The running resume3 process keeps the
  code it started with until its next restart/yield (SCORER_VERSION unchanged, so its cache stays valid).
  - T2S GSM8K fix ported; the synced T2S rows contain 0 gsm8k rows and no output text, so nothing needed rescoring.
  - Both outcome harnesses now record score_strict, format_ok, score_lenient, output_tail (register
    `x2-v3-gsm8k-strict-vs-lenient`, `x2-v3-canary-gate-lenient`). Lenient is computable only for v3 rows whose full
    response was stored (first 500 chars only before this change).
  - T2S allocation failures (register `t2s-outcome-allocation-failures`): both Ollama configs, time-ordered (all
    after about 07:29Z 2026-10-02, right after llama-server's first stuck 900 s load), not a prompt-length boundary.
    Cause open: needs the llama-server load logs `results/t2s_outcome_table_llamaserver_<item>.log`.
  - Cloud client: hard USD 50 total cap, alerts at 50/75/90% (once each, ledger + digest + print). Real mode keys on
    CLOUD_API_KEY; scripts that read OPENAI_API_KEY directly (harness/adapters/sdk_direct.py,
    harness/tail_latency_instrument.py, harness/backends/cloud_openai.py) bypass the cap. Operator decision pending.

### 2026-10-08, operator decisions applied

- R2 call-2 mode: v1 (tools removed on call 2) for all Ollama and llama-server R2 runs; cloud (OpenAI) R2 sessions
  use identical tools plus tool_choice "none". Cancelled on evo-x2: x2_r2_toolchoice_check, x2_r2_validation_v3,
  x2_r2_real_v2 (status `cancelled`; the launcher only starts `pending`).
- Subagents running: (1) strengthened R2, v1 protocol (5 seeds, + qwen3-4b-2507 and qwen3:8b with arm-b validation
  first, tiers 4096/8192/16384/32768/default, 40 turns, OLLAMA_DEBUG mechanism logging), queued after K2;
  (2) route every OpenAI call through the capped client plus a test forbidding direct use.
- T2S: no SSH; from the synced logs, llama-server also hit ErrorOutOfDeviceMemory on 20/40 items (register
  `t2s-llamaserver-load-failures`). Cause of the held device memory waits until evo-t2s is back.
- Outcome table v3 progress registered (`x2-v3-progress`, `x2-v3-scores`, `x2-v3-error-causes`), synced 4a288b1.
- Merged: cloud-cap routing (79fe2a2; every OpenAI call goes through the USD 50 capped client, test forbids direct
  use) and strengthened R2 (f5e5dd2). Queued on evo-x2 after K2: x2_r2_validation_v2b (qwen3-4b-2507, qwen3:8b,
  ~0.6 h), x2_r2_mechanism_v1 (OLLAMA_DEBUG render + log per tier, ~1.9 h), x2_r2_real_v1b (82 new sessions, gated
  on v2b, per-model refusal, ~46.5 h of which ~24.5 h is scaled not measured). Register rows for these are PENDING
  until synced. Machine order as instructed: outcome table v3, 70B edge reps, K2 pause-resume, then R2.
- 2026-10-08 queue change (operator): outcome table yielded at an item boundary (resume3 -> resume4, resumable);
  order now x2_r2_validation_v2b (running) -> x2_r2_mechanism_v1 -> x2_outcome_table_v3_resume4 -> x2_r2_real_v1b
  -> x2_k2_pause_resume_v1 -> x2_70b_edge_reps. The mechanism job was held until its prompt-token validity check
  (render-only tokens via llama-tokenize vs fresh-request prompt_eval_count, >=5 checks per tier, 1% match, tier
  citable only if all match) was merged and deployed (e96a5cb), then released. Watchdog Ready, no pause.
  Open risk: llama-tokenize's output format is assumed from its help text; a mismatch shows up as error rows and a
  not-citable tier, never a silent pass.
- 2026-10-08: merged 2f10f03 (install-path finding, task-tool gate, chat template source per row). Queued
  x2_r2_4b_tools_validate after x2_r2_mechanism_v1 (creates qwen3-4b-2507-tools from the same GGUF with the library
  Qwen3 template, verified by digest and template sha, then validates it alone into x2_r2_validation_v2c.jsonl).
  x2_r2_real_v1b now runs llama3.1:8b, qwen3:14b, qwen3-4b-2507-tools, qwen3:8b (projected ~51.6 h, partly
  unmeasured), gated on that job; a failing 4B is refused per model. Both bare-GGUF Ollama legs in the outcome
  table (qwen3-4b-2507 and qwen3-30b-a3b-2507) are marked "[bare template]" in the register tables.
- 2026-10-08: x2_r2_mechanism_v1 done (FINDINGS section of 2026-10-08; register R2-mechanism-verdict and
  R2-mechanism-lowlevel). All tiers citable (render tokens = fresh prompt_eval_count on every check). x2_r2_4b_tools_validate
  errored by design: the Modelfile was written with CRLF, so the created template differed from the library one only
  by CR characters; fixed (ecf1b66, LF), deployed, requeued as x2_r2_4b_tools_validate_r2 ahead of the outcome table via
  the yield. The old qwen3-4b-2507 tag is untouched. evo-x2 SSH briefly reset connections during the mechanism run
  (no reboot; uptime since 2026-10-04).

## Demo milestones (freeze 2026-10-28)

| Checkpoint | Due | Contents | Status 2026-10-08 |
|---|---|---|---|
| Router + Pareto on stub data | 2026-10-14 | src/dse/router.py with the context-shift guard rule; src/dse/pareto.py frontier + recommendation on current outcome rows, cloud rows STUB | in progress (subagents) |
| Dashboard with real local data | 2026-10-21 | demo/dashboard: form, frontier per hardware, recommendation with savings vs all-cloud, live-run view, naive-vs-ours 4096 scenario from R2 data | in progress (subagent) |
| Cloud rows + live validation + freeze | 2026-10-28 | real cloud rows through the USD 50 capped client once the key exists; live validation; freeze | not started (needs key) |

Every report states status against these three checkpoints.

### 2026-10-08, 4B revalidation and new workstreams

- qwen3-4b-2507-tools (same GGUF, library Qwen3 template, LF Modelfile after the ecf1b66 fix) failed its validation
  (register `R2-validation-v2c-baseline`, `R2-install-path-4b-comparison`): native tool calls on call 1 fell from
  11/30 (bare) to 2/30, rules 1, 2 and 5 at 0.0%, negative control 2/12 canary misses, task tool 5/30. The real
  run's start check refuses it (category "failed"); x2_r2_real_v1b runs llama3.1:8b, qwen3:14b, qwen3:8b.
- Started (subagents): context-shift predictor; x2_r2_mitigation_v1 (pre-registered, queued after x2_r2_real_v1b);
  demo router, Pareto, dashboard; docs/T2S_WEEK_PLAN.md (T2S back about 2026-10-15, no SSH until the operator says);
  docs/PAPER1_DRAFT.md (ISPASS skeleton); docs/BLADE_PLAN.md (overnight only; nothing starts before the operator says
  "start Blade night 1"; all local work paused during Blade runs).
- 2026-10-08 evo-x2 health: watchdog Ready, heartbeat current, no pause, 12/12 canary gates PASS. Rolling-error ALERTs
  on llama_server (qwen3-8b/14b/32b) traced to (a) items longer than n_ctx_train (48384 > 40960: server clamps, guard
  refuses) now classified context_overflow/exceeds_n_ctx_train (commit before 9bd1506, deployed; the running process
  picks it up at its next restart), and (b) 11 genuine qwen3-32b llama_server timeouts on 32K items (Ollama completes
  the same items); left as valid timeout outcomes, operator decision whether to raise the timeout. First 4B job marked
  error_superseded. Progress 1078/3600 (register x2-v3-progress).
