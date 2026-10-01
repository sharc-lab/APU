# ritz_spotcheck_30.csv -- blinded 30-row spot-check sample

A fresh, blinded 30-row spot-check sample of real R1b `art_*` probe rows, independent of the 150-row
kappa-agreement sample in `kappa_sample.csv` (that sample's source rows come from a different,
uncommitted raw file -- see `KAPPA_STUDY_NOTE.md` -- so there is no row overlap by construction).

## Provenance

Sampled by `analysis/build_ritz_spotcheck_30.py`, seed `20261001`, from the real, committed R1b
result files:

- `results/t2s_night2_20260929T202603Z.jsonl` (evo-t2s, 480 real `art_*` `arm1_baseline` /
  `arm3_self_report` rows)
- `results/t2s_night2_20260929T205109Z.jsonl` (evo-x2, 1440 real `art_*` `arm1_baseline` /
  `arm3_self_report` rows)

per `docs/FINDINGS.md`, "R1b evaluation audit, both machines, from the live Oct-1-cut runs
(2026-09-30)". All 1920 real rows from both files are pooled, shuffled with the fixed seed above, and
the first 30 are taken -- re-running the script reproduces the exact same 30 rows.

## Columns

`id`, `question`, `expected`, `output`, `label` (empty -- for the rater to fill in).

## Allowed values for the `label` column

Same four-category scheme used throughout this kappa-agreement study (`CORRECT` / `FABRICATED` /
`REFUSED`; `UNCLASSIFIABLE` did not occur in this sample since every row has non-empty output):

- `CORRECT` -- output gives the expected answer (exactly, or embedded in otherwise format-noncompliant
  text).
- `FABRICATED` -- output gives a specific, confident, wrong-looking answer with no genuine refusal
  language.
- `REFUSED` -- output declines to answer, or declares the needed information unavailable, without
  then giving a concrete value anyway.
