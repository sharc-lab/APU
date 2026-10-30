# apu_bench: open release plan (plan only)

Written 2026-10-29. Nothing here has been done: no package was built, no repository was created, no visibility was
changed. The harness and its format are in `apu_bench/` and `docs/APU_BENCH_RFP_MAPPING.md`; the audit design is
`docs/BYOL_AUDIT_DESIGN.md`.

## Constraint: double-blind review

The paper (ISPASS: abstract Dec 11, paper Dec 18) is double-blind. A named, public release before or during review
would identify the authors. The release therefore happens in two stages:

1. **During review:** an anonymous artifact only, if the venue allows one (an anonymized archive or an anonymous
   repository link in the submission). It contains the stripped tree below, with no author, lab, machine or account
   names. See `docs/paper1/DOUBLE_BLIND_PLAN.md` for what identifies the authors today.
2. **After the decision (or after the paper is public, whichever the venue's policy allows first):** the named
   public release, steps below.

## What to strip (both stages)

- Everything outside the benchmark: `analysis/`, `demo/`, `docs/` except the benchmark's own docs, `queues/`,
  `routing/`, `scripts/` (deployment and sync scripts reference machines and accounts), `results/` except the sample
  inputs the tests need.
- Machine and account identifiers: hostnames in device profiles (`apu_bench/profiles/*.json` "hardware.hostname",
  "sources"), Tailscale addresses, user names and home-directory paths in any file or log, SSH and scheduled-task
  configuration.
- Recorded server logs used as test fixtures: re-check every kept log for paths, hostnames and user names; replace
  with a scrubbed copy and record the scrub in a manifest (the audit tests must still pass on the scrubbed logs).
- Internal planning text: register rows, operator decisions, cost ledgers, IP and disclosure notes.
- Git history: release from a fresh repository with one initial commit made from the stripped tree. Do not publish
  this repository's history (author emails and names are in it). Do not rewrite this repository's history.
- Cloud price tables: keep (public list prices), with their source dates.

What stays: `apu_bench/` (run, audit, onepage, profile, taskset, adapters), the B3 v2 task definitions and grader it
imports (`harness/evalset_env.py` and the minimum of `harness/x2_r2_evalset.py` the live mode needs, or a vendored
copy), `src/dse/fits_locally.py`, `src/cloud/prices.py`, the device-profile and session schemas, the example fake
profile, the tests that run offline, and a README.

## Steps after submission (named release)

| step | what | hours (estimate) |
|---|---|---|
| 1 | Freeze the benchmark version used in the paper: tag `apu-bench-1.0.0` in the private tree; record task-set sha256 and schema versions | 1 |
| 2 | Build the stripped tree with a script (allow-list of paths, not a deny-list), plus a manifest of every file and its sha256 | 4 |
| 3 | Vendor or decouple the harness imports so `apu_bench` runs without the rest of the repo; keep the fake runtime as the default | 6 |
| 4 | Scrub kept fixture logs; rerun the audit tests on the scrubbed copies | 3 |
| 5 | Identity sweep on the stripped tree: grep for the categories in DOUBLE_BLIND_PLAN.md, plus a manual read of every doc | 2 |
| 6 | README: what it measures, the two commands, the schemas, the "FAKE RUNTIME" label, how to submit a result, citation (added only after de-anonymization) | 3 |
| 7 | Licence files and headers (below); third-party notices for anything vendored | 1 |
| 8 | Anonymous artifact archive for review (stage 1), if the venue allows | 1 |
| 9 | After the decision: new public repository under the lab's account, single initial commit, release tag, archive DOI | 2 |
| 10 | Packaging: `pyproject.toml` for `apu-bench`, Python 3.12, optional extras for live mode (`httpx`) | 2 |
| | total | 25 |

## Versioning

- Semantic versioning on the benchmark as a whole, `MAJOR.MINOR.PATCH`, starting at `1.0.0` for the paper's version.
- MAJOR: any change to the task set, grader, ground-truth rule (what counts as a hidden cut) or scoring; results are
  comparable only within one MAJOR.
- MINOR: new runtimes, adapters, report fields or optional schema fields, with old results still valid.
- PATCH: fixes that do not change any score on the committed fixtures (the tests pin this).
- Schemas carry their own versions (`apu-device-profile/1`, `apu-audit-session/1`, `apu-bench-result/1`); a result
  file records the benchmark version, the schema versions and the task-set sha256, so a reader can check
  comparability from the file.

## Licence

- Code: Apache-2.0 (patent grant and notice requirements suit a benchmark that vendors may run and extend). MIT is
  the simpler alternative; the choice is the operator's and the lab's, and must be checked against the university's
  technology-transfer policy and any open invention disclosure before release.
- Task definitions and fixture data: CC BY 4.0, or the same Apache-2.0 if one licence is preferred.
- Third-party: keep the licences of anything vendored (for example llama.cpp-derived constants or tokenizer files, if
  any are kept); list them in `THIRD_PARTY_NOTICES`.

## Open questions for the operator

- Whether the venue allows an anonymous artifact during review, and in what form.
- Licence choice and the tech-transfer check (an invention disclosure for rollback may be filed; the benchmark itself
  does not include rollback).
- Which org account hosts the named release.
