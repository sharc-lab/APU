# Thesis amendment 2026-10-15: project thesis and paper wording

Decided by Zach and Claude (Cowork) 2026-10-14/15; recorded in Task AQ.
Adds to the registered files below; edits none of them. Runners do not open
this file.

## 1. What supersedes what

From 2026-10-15 the project thesis is docs/THESIS.md, and the paper-level
wording is in:
- docs/PAPER_[private]_ABSTRACT.md ([private];
  current priority)
- docs/[private]_ABSTRACT.md ([private], planned after [private])
- docs/PAPER_[private]_ABSTRACT.md ([private], Zach's configuration line; numbers dated
  2026-09-07, to refresh)

These supersede the thesis, headline and main-claim wording in:
- docs/amendments/DEMO1_AMEND_2026-10-13c_CONFIG.md section 1 ("Thesis")
- derived/h1_hybrid/demo1/DEMO_PREREG_v2.json field "thesis"
- docs/EXPERIMENT_REGISTRY.md "Unifying thesis (2026-10-12)" (kept below
  the new summary, marked superseded)
- docs/DEMO1_DESIGN.md "Thesis" headline (kept for the record)
- docs/SEAM_research_blueprint.md section 1 "Thesis and main line" (a
  one-line note is added; pin updated in GOVERNING_DOCS.sha256 under this
  amendment)
- docs/MEETING_BRIEF_hybrid_execution.md section 1 headline (one-line note)
- every other thesis or headline line found by the 2026-10-15 search
  (listed in the Task AQ report), whether or not it carries a note

The registered files keep their bytes. Where their wording disagrees with
docs/THESIS.md, docs/THESIS.md governs what the project claims.

## 2. What does not change

- Every prediction, band, test, threshold and verdict in every registered
  prereg and amendment is unchanged. That includes DEMO_PREREG_v2,
  DEMO1_AMEND_2026-10-13c, the V5 addendum, V5-CAL, DEMO2_PREREG_v1, every
  NPU amendment, H1_CPU_AMEND_2 and the P1 amendments.
- Every sealed run and every score stays as recorded.
- No registry row is dropped. Claims C1-C5 stay in the registry, relabeled
  as mechanism / evidence under the new thesis.

## 3. What the new thesis adds (summary; docs/THESIS.md is authoritative)

- Configuration on fixed hardware is the design space; hardware change is a
  yardstick.
- Four objectives: turn latency, memory cost, task quality, context held
  before failure; cloud $ is reported next to them for hybrid cells.
- Six axes: placement, session residency, KV precision, weight precision,
  model tier/size, context window/retention (naming open for Zach).
- H-A to H-D are hypotheses with evidence status, not findings.
- Literature-required wording from the 2026-10-15 sweep
  (docs/RELATED_SEARCHES.md).
- Decision rules for every future experiment (also in CLAUDE.md).
