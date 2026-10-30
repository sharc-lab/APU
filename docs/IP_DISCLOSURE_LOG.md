# IP disclosure log

Written 2026-10-26, read-only on history. Nothing here changes repo visibility or rewrites history; the
operator raises the question with Prof. Hao.

## Method

- "First public" = the earliest commit containing the concept that is reachable from `origin/main` (remote
  `https://github.com/sharc-lab/APU.git`). Found with `git log --diff-filter=A --reverse origin/main -- <path>` for
  the files and `git log -S <term> --reverse origin/main` (pickaxe) for the concept terms in code and docs.
- `origin/main` is the local tracking ref as of 2026-10-26, not re-fetched (this task was offline). At that time it
  was 383e498, the same as local `main`.
- Dates: author date and commit date, both from git. The commit date is not the date the commit became public.
- Push date: the authoritative push time is only in GitHub (the repository events API, or the remote's own reflog,
  neither read here). As a local substitute, the column "first local push record" is the earliest entry of this
  laptop's reflog for `refs/remotes/origin/main` (all entries read "update by push") whose commit contains the
  concept commit. It records pushes made from this laptop only. A push from another clone, or of another branch
  (for example the demo branch `demo/oct28`, which is not present as a remote-tracking ref here), would not appear,
  so a concept can have become public earlier than this column says, never later than it says (assuming the
  remote has not been force-pushed since).
- Visibility history (when the repo became public) is not visible from git; the dates below are the dates the
  commits reached `origin/main`, which equals public disclosure only for the period the repo was public.
- All times are as git recorded them (PT, -0700, except the July commits at -0400).

## Table

| Concept | First code commit (hash, author date / commit date) | First local push record | First description commit (hash, date, file) | Notes |
|---|---|---|---|---|
| Route-at-start (decide local vs cloud per device and task at session start) | 7c179c0, 2026-10-22 16:11 / 16:11 (`demo/dashboard/views.py`, `demo/dashboard/cache/device.json`: per-device decision at session start); eca13d6, 2026-10-22 16:12 / 16:12 (`analysis/impact_model.py` route_at_start, `tests/test_route_at_start.py`); 92c02cd, 2026-10-22 18:38 / 18:38 (`src/dse/route_at_start.py` added) | 7c179c0 and 92c02cd: 2026-10-23 21:03 (came to main with the demo branch); eca13d6: 2026-10-22 16:40 | eca13d6, 2026-10-22 16:12, `docs/IMPACT.md` (session-start section); 7482b9a, 2026-10-22 16:22, `docs/FINDINGS.md`; a567687, 2026-10-22 16:42, `docs/PAPER1_DRAFT.md`; 1826e66, 2026-10-22 18:05, `docs/EXPERIMENTS.md` | Earliest public: eca13d6 (code and IMPACT.md together), local push 2026-10-22 16:40. Precursor: 998e7d1, 2026-09-30 23:12, `src/dse/router.py` (route before running: a blocked local request goes to cloud within budget, then compaction, then refuse), local push 2026-10-01 23:39. |
| Rollback (transactional snapshots, pre-task and cascaded rollback before a cloud redo) | df10229, 2026-10-23 12:03 / 12:03 (`SnapshotLog` in `harness/evalset_env.py`, `scripts/cloud_redo.py --rollback pre_task`); 5d270af, 2026-10-23 12:06 / 12:06 (`analysis/own_rollback.py` added) | 2026-10-23 12:54 (both) | 5d270af, 2026-10-23 12:06, `docs/EXPERIMENTS.md` and `docs/FINDINGS.md` (pre-registration); d03c2cc, 2026-10-23 13:34, `docs/EXPERIMENTS.md` (cascaded variant). Not in `docs/PAPER1_DRAFT.md` (pickaxe on "rollback": no commit). | Earlier, different sense of the word: febf980, 2026-07-23 09:48 (-0400), `routing/policies/speculative.py`, `docs/CONTRIBUTIONS.md` ("action-level rollback accounting" for speculative routing, SPEC_ROLLBACK timing), local push 2026-07-23 09:48. That is speculative-execution rollback cost, not environment snapshots; listed because a reader could argue it as prior disclosure. |
| Compaction (client-side context manager) and the earlier client trim | Client trim: 4b46765, 2026-10-08 14:00 / 14:10 (`harness/x2_r2_client_trim.py` added, `x2_r2_agent --client-trim`). Compaction: 0387a4f, 2026-10-23 12:09 / 12:09 (`harness/own_compaction.py` added) | Client trim: 2026-10-08 14:17. Compaction: 2026-10-23 13:09 | Client trim: 8297347, 2026-10-08 13:47 (commit 14:10), `docs/FINDINGS.md` (pre-registration of x2_r2_mitigation_v1). Compaction: 6fd58f4, 2026-10-23 12:21, `docs/EXPERIMENTS.md` and `docs/FINDINGS.md`. Not in `docs/PAPER1_DRAFT.md` (pickaxe on "compaction": no commit). | Precursor: "compaction" as a named fallback in 998e7d1, 2026-09-30 23:12, `src/dse/router.py` (local push 2026-10-01 23:39) and in `docs/STATE.md` 49a0e24, 2026-10-06 20:43. Neither implements it. |
| Overflow detector: window flag | e22ea9b, 2026-10-07 20:54 / 20:54 (`analysis/r2_real_report.py`: over-window flag, final prompt exceeded the loaded window); 4d0cb7b, 2026-10-22 18:17 (per-task window flag in `analysis/impact_model.py`) | e22ea9b: 2026-10-07 20:55; 4d0cb7b: 2026-10-22 19:03 | 9de656a, 2026-10-14 00:12 (commit 09:08), `docs/FINDINGS.md` ("over-window"); 6b09282, 2026-10-14 09:07, `docs/IMPACT.md`; 73fc53c, 2026-10-22 14:52, `docs/PAPER1_DRAFT.md` ("window-flag limits") | |
| Overflow detector: context-shift rule (P + C > n_ctx predicts a shift) | 07682d8, 2026-10-08 13:56 / 13:56 (`analysis/r2_shift_predictor.py` added) | 2026-10-08 14:12 | de77e91, 2026-10-08 13:46, `docs/STATE.md` (name only); 07682d8, 2026-10-08 13:56, `docs/FINDINGS.md` (description); aa61631, 2026-10-21 20:13, `docs/PAPER1_DRAFT.md` ("context-shift rule taken from the llama.cpp source") | The rule is derived from the llama.cpp source, as the paper draft itself says. |
| Overflow detector v1 (cut lines, or token count below the calibrated estimate) | 92c02cd, 2026-10-22 18:38 / 18:38 (`harness/demo_overflow.py` added, `tests/test_demo_overflow.py`) | 2026-10-23 21:03 (with the demo branch) | 9268b76, 2026-10-22 20:19, `docs/EXPERIMENTS.md` and `docs/FINDINGS.md` (demo live pre-registration); 44e868b, 2026-10-22 14:49, `docs/PAPER1_DRAFT.md` (an overflow detector discussed generally, before v1 code) | Retired for routing on 2026-10-23 (FINDINGS "DECISION (2026-10-23)"). |
| Overflow detector v2 (the runtime's own cut-off log lines only) | 2505a1e, 2026-10-23 11:10 / 11:10 (`harness/demo_overflow.py`, `harness/x2_queue_fill.py`); also used per call in 0387a4f, 2026-10-23 12:09 (`harness/own_compaction.py`) | 2505a1e: 2026-10-23 21:03; 0387a4f: 2026-10-23 13:09 | 13aa6e8, 2026-10-23 10:28, `docs/EXPERIMENTS.md` (proposed); 2505a1e, 2026-10-23 11:10, `docs/FINDINGS.md`; 43dd034, 2026-10-23 11:58, `docs/PAPER1_DRAFT.md` | FINDINGS states v2 is the runtime's own signal and not a contribution of this work. Earliest public of v2 code by local record is 0387a4f (13:09), earlier than 2505a1e's (21:03). |
| Cost model: impact model | 6b09282, 2026-10-14 09:07 / 09:07 (`analysis/impact_model.py` added) | 2026-10-14 09:13 | 6b09282, 2026-10-14 09:07, `docs/IMPACT.md`; cbb7b48, 2026-10-14 10:32 (commit 21:47), `docs/PAPER1_DRAFT.md` (motivation from the A3 measured inputs) | |
| Cost model: src/cloud pricing | a257a20, 2026-09-30 20:07 / 20:07 (`src/cloud/client.py`: spend cap and ledger); efea7bf, 2026-10-14 21:43 / 21:43 (`src/cloud/prices.py`, shared price loader); d0ed237, 2026-10-14 21:43 (client prices from the shared files) | a257a20: 2026-10-01 23:39; efea7bf and d0ed237: 2026-10-14 21:56 | efea7bf, 2026-10-14 21:43 (R2 session cost) | Prices are public vendor list prices; the loader is the disclosed part. |
| Cost model: DSE v1 (cost ranking, Pareto, brute-force validation) | bc6996c, 2026-10-23 12:06 / 12:06 (`src/dse/dse_v1.py` added) | 2026-10-23 12:40 | 92aade5, 2026-10-23 12:02, `docs/EXPERIMENTS.md` (pre-registration, "cost model") | |

## Update 2026-10-29: first public commit per concept

Read from the table above (same method, no re-fetch; offline task). "First public" is the earliest of the code and
description commits by first local push record.

| Concept | First public commit | First local push record | Status |
|---|---|---|---|
| Route-at-start | eca13d6, 2026-10-22 16:12 (code and `docs/IMPACT.md`) | 2026-10-22 16:40 | public |
| Rollback (environment snapshots before a cloud redo) | df10229, 2026-10-23 12:03 (code); 5d270af, 2026-10-23 12:06 (pre-registration) | 2026-10-23 12:54 | public |
| Post-run check | none | none | not yet public: being written on 2026-10-29, not on `main` or `origin/main`. Earlier, different sense: 93ffd20, 2026-10-08 14:15, a "post-run runtime check" of the loaded context per turn in the T2S week plan; listed only because a reader could argue it as prior disclosure. |
| Compaction (client-side context manager) | 0387a4f, 2026-10-23 12:09 (code); client-trim precursor 4b46765, 2026-10-08 14:00 | 2026-10-23 13:09 (client trim 2026-10-08 14:17) | public |
| Detector (window flag; shift rule; v1; v2) | window flag e22ea9b, 2026-10-07 20:54; shift rule 07682d8, 2026-10-08 13:56; v1 92c02cd, 2026-10-22 18:38; v2 0387a4f, 2026-10-23 12:09 | 2026-10-07 20:55; 2026-10-08 14:12; 2026-10-23 21:03; 2026-10-23 13:09 | public (v2 is the runtime's own signal, not claimed) |
| Cost model (pricing; impact model; DSE v1) | a257a20, 2026-09-30 20:07; 6b09282, 2026-10-14 09:07; bc6996c, 2026-10-23 12:06 | 2026-10-01 23:39; 2026-10-14 09:13; 2026-10-23 12:40 | public |

Draft disclosures held privately by the operator.

## Committed personal usernames and paths

Searched separately (all of history reachable from `origin/main`, and the current tree). The findings are reported
to the operator directly and are deliberately not repeated in this file, since this file is itself public and
listing them here would disclose them again.
