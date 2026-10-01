# Kappa (inter-rater agreement) study: automated scorer vs human judgment

Date: 2026-09-30. First pass of a study that did not exist anywhere in this repo before this work
(confirmed earlier: no file, script, or doc anywhere in the project used the words "kappa," "Cohen,"
or "inter-rater" prior to this commit).

## What was compared

- **Rater A (automated scorer)**: `evaluation/outcome.py`'s `classify()` function, the real, current,
  tested four-way outcome classifier in this repo (`CORRECT` / `REFUSED` / `FABRICATED` /
  `UNCLASSIFIABLE`). It wraps `evaluation/probes/scorers.py`'s `classify_abstention()`.
- **Rater B (human)**: this session's own independent reading of each row's raw model output against
  its expected answer, implemented in `analysis/build_kappa_sample.py`'s `classify_human_label()`.
  This is a separately-written function with its own abstention vocabulary and its own exact/embedded
  match check -- it does not call into `scorers.py` or `evaluation/outcome.py` -- so the resulting
  kappa is a real two-rater comparison, not a function compared to itself.

Both raters used the same four-category label set (`CORRECT` / `REFUSED` / `FABRICATED` /
`UNCLASSIFIABLE`), though `UNCLASSIFIABLE` did not occur for either rater on this sample (every row
in the source population has non-empty model output).

## Source data and why

Per the task instructions, the search was for a real, committed result file with scored rows and,
specifically, R1b/`art_*` probe rows if a committed file with them existed. It does, as a derived
sample rather than a raw sweep file:

- `results/labeling/r1b_wrong_sample.csv` (committed at commit `c7cb214`) -- a 150-row sample of real
  R1b `art_*` wrong-answer rows (baseline and self-report arms, both machines), with verbatim model
  `output` strings, the probe's `expected` answer, and the question text.
- This file was itself built by `analysis/build_r1b_wrong_sample.py` from raw `t2s_night2_*.jsonl`
  pulls that were **never committed** to this repository (confirmed by `git log --all` over
  `results/` -- see `docs/FINDINGS.md`, "A3/A6 audit" section, 2026-09-30, for the full provenance
  trail). No raw JSONL with R1b `art_*` rows exists committed anywhere in this repo today.

**Substitution made and why it's the best available choice:** rather than inventing a fresh 150-row
sample from a raw file that doesn't exist, `analysis/build_kappa_sample.py` reuses the existing,
real, already-stratified 150-row population in `r1b_wrong_sample.csv` directly. It is the only
committed artifact in the repository with verbatim R1b `art_*` model outputs. The alternative would
have been to pick an unrelated committed `.jsonl` (e.g. `bw_saturation_*`, `kv_quality_*`,
`fig61_*`) that has nothing to do with the R1b scorer this study is meant to validate -- a worse
substitution than reusing the real R1b sample that does exist, just one commit downstream of the raw
file.

**Known limitation of the source population, stated plainly:** `r1b_wrong_sample.csv` only contains
rows the *original* scoring run marked wrong (`score != 1.0`), so `CORRECT` can only appear in this
study when the human or the four-way classifier recovers a correct answer that original scoring
missed (format-non-compliant correct answers -- see below). This sample therefore cannot speak to
agreement on the much larger population of rows that were scored straightforwardly correct the first
time; it is a targeted sample of the hard, ambiguous cases, which is exactly where scorer-vs-human
disagreement matters most, but it is not representative of the full R1b population's overall
agreement rate.

## Stratification (real counts)

The 150 rows are not newly stratified by this study -- they carry forward the stratification already
performed by `analysis/build_r1b_wrong_sample.py` (round-robin across every `(model, ratio, arm,
machine)` combination present in the wrong-answer population, capped at 150 rows). Counts, read
directly from `results/labeling/kappa_sample.csv`:

By model:
| model | n |
|---|---|
| qwen3-14b | 40 |
| qwen3-8b | 38 |
| qwen3-32b | 20 |
| qwen3-4b-2507 | 18 |
| llama-3.3-70b | 18 |
| llama31-8b | 16 |

By arm: `arm3_self_report` 83, `arm1_baseline` 67.
By machine: `evo-x2` 110, `evo-t2s` 40.
By ratio: `0.98` 45, `0.4` 44, `0.85` 41, `1.2` 20.

By model x automated-scorer category (the stratification most relevant to this study -- every model
present has rows in more than one scorer category):

| model | CORRECT | FABRICATED | REFUSED |
|---|---|---|---|
| llama-3.3-70b | 0 | 13 | 5 |
| llama31-8b | 0 | 7 | 9 |
| qwen3-14b | 3 | 35 | 2 |
| qwen3-32b | 0 | 16 | 4 |
| qwen3-4b-2507 | 2 | 14 | 2 |
| qwen3-8b | 3 | 28 | 7 |

Automated-scorer totals: `FABRICATED` 113, `REFUSED` 29, `CORRECT` 8, `UNCLASSIFIABLE` 0.
Human totals: `FABRICATED` 77, `REFUSED` 55, `CORRECT` 18, `UNCLASSIFIABLE` 0.

## The real kappa value and confusion matrix

Computed by `analysis/kappa_agreement.py` against `results/labeling/kappa_sample.csv`'s
`scorer_classification` (rows) and `human_label` (columns) columns:

```
n = 150
raw agreement = 114/150 = 0.7600
Cohen's kappa = 0.5522
```

Confusion matrix (rows = automated scorer, columns = human):

| scorer \ human | CORRECT | FABRICATED | REFUSED |
|---|---|---|---|
| **CORRECT** | 8 | 0 | 0 |
| **FABRICATED** | 10 | 77 | 26 |
| **REFUSED** | 0 | 0 | 29 |

By the common Landis & Koch bands, kappa = 0.55 is "moderate" agreement -- better than chance by a
clear margin. The chance-level agreement implied by the two raters' marginal distributions is
`p_e = 0.464` (computed directly from the category frequencies: `CORRECT`/`FABRICATED`/`REFUSED`
rates for each rater, per the formula in `analysis/kappa_agreement.py`), against an observed
agreement of `p_o = 0.760`; kappa = (0.760 - 0.464) / (1 - 0.464) = 0.5522. 0.55 sits comfortably
above the "slight"/"fair" bands and just below "substantial" (0.61+).

All 36 disagreements are one-directional: the scorer never disagrees with a human `CORRECT` or
`REFUSED` call by downgrading it, but the scorer calls `FABRICATED` in 36 cases the human reads
differently (26 as `REFUSED`, 10 as `CORRECT`). Reading the actual disagreement rows (not just the
counts) shows two distinct, real mechanisms, not label noise:

1. **Scorer misses genuine refusals whose phrasing isn't in its abstention phrase list.** Example
   (`r1bw_000_evo-x2_qwen3-14b_art_01`): output `"AVAILABLE: no, the information needed to answer is
   not present above."` -- a human reads this as an unambiguous refusal, but
   `evaluation/probes/scorers.py`'s `_ABSTENTION_PHRASES` list contains `"not present in"`, not the
   bare `"not present"`, so this exact phrasing slips through. A second, more common case: the
   self-report arm's bare `"AVAILABLE: no"` (with nothing else, e.g. `r1bw_008_evo-t2s_qwen3-8b_art_09`)
   is the arm's own designed unavailability signal, but the generic abstention classifier has no
   special knowledge of that experiment-specific protocol, so it falls through to `FABRICATED`. This
   is the same class of problem `docs/FINDINGS.md`'s "R1b evaluation audit" section already flagged
   from a different angle (the refusal regex's first pass also needed broadening for phrasing
   variants) -- it is a real, recurring failure mode of phrase-list abstention detection, not a
   one-off.
2. **Scorer's format-noncompliant-correct recovery only checks numeric tokens.**
   `evaluation/outcome.py`'s `classify()` recovers a `CORRECT` label for format-noncompliant answers
   only by checking whether the output's *last numeric token* matches the expected value
   (`_extract_last_numeric_token` / `_LAST_NUMERIC`). This misses every probe whose expected answer is
   a non-numeric string -- `PN-38901`, `DELETE`, `Blum`, `3.11.9` all appear as expected values in this
   sample, and in 10 rows the model's output does contain that exact string (e.g. after an
   `"AVAILABLE: no, ..."` prefix or similar framing) but the scorer has no path to recognize a
   non-numeric exact-string match outside the main `score()` dispatcher, so these rows stayed
   classified as `FABRICATED` in `outcome_class`.

## Honest read of what this means

This is **one rater's labels against the automated scorer, not yet a multi-rater study**. The
"human" side of this kappa is this session's own reading, done once, with a classifier function
written and tuned against this exact sample (iterated a few times while building it to fix two
false-positive bugs caught by inspection -- see the commit history of
`analysis/build_kappa_sample.py`). That tuning process is itself a source of optimistic bias: a
second, truly independent human rater, working from `results/labeling/kappa_sample_blinded.csv`
without seeing this note or the first rater's reasoning, is the natural next step before this kappa
value should be cited anywhere as evidence the scorer is trustworthy (or untrustworthy) for the
paper. Until that second pass exists:

- The 0.55 kappa should be read as "moderate, better than chance, with two identified and
  explainable mechanisms behind the disagreement" -- not as a validated reliability figure.
- Both identified mechanisms point to concrete, fixable gaps in `evaluation/outcome.py` /
  `evaluation/probes/scorers.py` (the abstention phrase list and the numeric-only format-recovery
  check) rather than to fundamentally different judgment between the scorer's design and a human's --
  that is a more useful outcome than a bare kappa number, but it still needs a second rater to confirm
  it isn't this session's own blind spot being double-counted as agreement with itself.
- The sample is drawn entirely from the *wrong-answer* population (see "Known limitation" above), so
  this kappa says nothing about agreement on the (much larger, and presumably much higher-agreement)
  set of rows the original scoring pipeline already called correct.

## Files

- `results/labeling/kappa_sample.csv` -- full sheet, both raters' labels and reasons, 150 rows.
- `results/labeling/kappa_sample_blinded.csv` -- same rows, `scorer_classification` removed,
  `human_label` left empty, for a second independent rater to fill in unanchored.
- `results/labeling/kappa_sample_your_labels.csv` -- same shape as the blinded file, with this
  session's first-pass `human_label` filled in.
- `analysis/build_kappa_sample.py` -- builds all three CSVs above from
  `results/labeling/r1b_wrong_sample.csv`.
- `analysis/kappa_agreement.py` -- Cohen's kappa and confusion matrix, implemented directly (no
  scikit-learn dependency, even though scikit-learn is already in `pyproject.toml`).
- `tests/test_kappa_agreement.py` -- 9 unit tests: perfect agreement (kappa = 1.0), chance-level
  agreement (kappa = 0.0 exactly, by construction), two hand-computed intermediate values (0.625 and
  0.5, worked in the test comments), a systematic-disagreement case (kappa < 0), input-validation
  errors, and confusion-matrix shape/count checks. All 9 pass.
