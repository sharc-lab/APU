# Trim plan: Paper 1 to the ISPASS 2027 page limit

Written 2026-10-31. Plan only: nothing has been cut yet (operator: "trim to the page limit once known").

## The limit

ISPASS 2027 (New York University, May 9-11, 2027; abstract Dec 11, 2026, full paper Dec 18, 2026, AoE), from the
official site, read 2026-10-31:

- https://ispass.org/2027/submission/ : "The content of the submission is limited to nine (9) single-spaced,
  double-column pages", 10-point type on 8.5x11 inch pages (IEEE conference style), with "an unlimited number of
  additional pages for references". "All content other than references (figures, appendices and so on) must fit
  within the first nine pages." Reviews are double-blind; own work is cited in the third person.
- https://ispass.org/2027/cfp/ : the deadlines above; the submission site had not opened yet.

So references do not count, but **an appendix does count**. Material that leaves the body goes to the anonymous
artifact (`docs/APU_BENCH_RELEASE_PLAN.md`, stage 1), not to an appendix.

## Where the draft stands

Build of 2026-10-31 (`docs/paper1/paper1.tex`, IEEEtran `conference`, 10 pt): 12 pages in total. The body ends on
page 12 and the references start partway down that page, so the body is about 11.3 pages. It must come down to 9,
a cut of about 2.3 pages; the plan aims at about 2.8 pages to leave room for real figures, which are placeholders
today (9 figure boxes of 3.6 to 4.2 cm).

Words (md2tex count, register comments excluded): abstract 324, body prose 10,439, figure and table specs about 725.
At about 925 body words per page (with the current figures and three tables), 9 pages hold about 7,700 words with six
figures.

## Rules for the cut

- Cut prose only. No number is changed or dropped from the record: a sentence that carries a number either stays
  verbatim with its row ID or moves, verbatim with its row ID, to the artifact's supplementary text.
- The five open operator decisions stay as written: b5-sim-constraint in 4.5, the rollback citation wording beyond
  the stand-in sentence (the stand-in sentence itself stays), the borderline detector wording, the pitch and demo fixes
  (not in the paper text), and ContextPaw (now resolved: removed on 2026-10-31).
- Every cross-model claim keeps its "R2 rule-following evidence, not B3 task accuracy" label.

## Cuts by section

| section | words now | target | cut | what goes |
|---|---|---|---|---|
| Abstract | 324 | 250 | 75 | Fold the backend sentence into one clause; drop the list of fixes to one sentence. |
| 4.1 Introduction | 1147 | 800 | 350 | The five-part answer repeats results stated in 4.4 to 4.7: keep one sentence per part and the contribution list; shorten "Why run locally". |
| 4.2 Background | 949 | 650 | 300 | One or two sentences per paragraph; keep every citation; merge "Agents and tool use" into "Routing". |
| 4.3 Methodology | 830 | 650 | 180 | Call-2 protocol and the gate list move to the artifact; keep the definitions of gated silent failure and pre-registration. |
| 4.4 Memory boundaries | 1143 | 850 | 290 | "Co-running work on unified memory" moves to the artifact with F4; NVIDIA "cost pending" to two sentences. |
| 4.5 Runtime policy | 1458 | 1050 | 400 | "Install path and first-call stall" moves to the artifact; "What the KV setting loads" keeps the size and agreement sentences, its rate and per-seed timing sentences move; the planned llama-server 2x2 paragraph shrinks to one sentence. |
| 4.6a Window result | 1378 | 1050 | 330 | Merge the llama-server replication into the protocol-check paragraph; "A default the user never sees" overlaps 4.5 and shrinks to two sentences; the long-task paragraph to three sentences. |
| 4.6b Mechanism | 663 | 520 | 140 | "How the window is set" overlaps 4.5 ("Changing the window"): keep one. |
| 4.7 Fixes | 1246 | 950 | 300 | Shorten num_keep and the router paragraph; the recovery paragraph keeps its numbers and drops the explanation that repeats 4.6a. |
| 4.7p Shift rule | 519 | 420 | 100 | Tighten the characterization paragraph; F8 moves to the artifact. |
| 4.8 Threats | 554 | 450 | 100 | Merge "Provenance" into "Statistics"; drop restatements of 4.6a deviations. |
| 4.9 Conclusion | 552 | 400 | 150 | Five lessons to one sentence each; future work to three sentences. |
| total | 10,763 | 8,040 | 2,715 | (abstract included) |

## Figures and tables

Today: 9 figures (FH1, F1 to F8) and 3 tables (Table 1, Table 2, Table R2-gated). Keep six figures: FH1 (headline),
F1 (platforms), F2 (Intel flip point), F5 (default windows over the trace CDF), F6 (survival), F7 (event raster).
Move F3 (NVIDIA spill), F4 (co-runners) and F8 (rule compliance under trimming) to the artifact. Table 1 can become
part of F1's caption if space is still short.

## Order of work and estimate

1. Move the listed paragraphs and figures to an artifact supplement file, verbatim with row IDs (1 h).
2. Section cuts in the order 4.5, 4.1, 4.6a, 4.7, 4.4, 4.2, then the rest, rebuilding after each (6 h).
3. Check every number in the trimmed draft against the register: no automated paper-number check exists in `tests/`
   today, so either diff the numbers of the old and new draft by script or check by hand (2 h).
4. Draw the six figures and rebuild; adjust the cut if the real figures differ in height from the placeholders (time in the figure plan, not here).

About 9 hours of writing once the operator approves the plan.

## Applied 2026-10-31

Built from `docs/PAPER1_DRAFT.md` with `python md2tex.py ../PAPER1_DRAFT.md paper1.tex` (run in docs/paper1), then
pdflatex, bibtex, pdflatex, pdflatex (no LaTeX errors; one 3.8 pt overfull box, present before the trim too).

- **Pages:** 10 in total. The body ends on page 9 (right column, about two thirds down) and the references start
  there and run onto page 10, so everything other than references fits in the first nine pages. Before the trim: 12
  pages, references starting on page 12.
- **Figures and tables:** six figures (FH1, F1, F2, F5, F6, F7) and Tables 1, 2 and R2-gated. F3, F4 and F8 moved to
  `SUPPLEMENT.md` verbatim with their row ids.
- **Source file:** the trim was made in a copy and then copied over `docs/PAPER1_DRAFT.md`, which is again the
  single source (the tex built from it is identical). The pre-trim source is saved as
  `docs/paper1/preview/PAPER1_DRAFT_pretrim_20261031.md`.
- **Audit decisions applied first:** b5-sim-constraint result in 4.5 (router on the frontier in all four capped cells;
  verdict refuted only because client trimming is also on the frontier) with the dependent sentences in 4.1 and 4.7
  updated; "a plain redo" wording for the redo that cannot undo executed actions in 4.1, 4.7 and 4.9, and "on
  simulated tools, cascaded rollback fixed 28/28 (195/197 overall)" in 4.7 and 4.9 (not in the abstract); 4.7p's
  before-the-call sentence reworded to say it characterizes the runtime's rule.
- **Number check:** every number of the pre-trim abstract and body was extracted by script and compared with the
  trimmed abstract and body. 70 distinct numbers no longer appear in the paper; each one is in `SUPPLEMENT.md`
  together with the register id it carried. Numbers new to the paper: 80.0%, 97.5% and 12.5%, all from row
  b5-sim-constraint (audit decision 1).

Words per section (md2tex count, register comments excluded; figure and table spec words not included):

| section | before | after | cut |
|---|---|---|---|
| Abstract | 324 | 282 | 42 |
| 4.1 Introduction | 1147 | 812 | 335 |
| 4.2 Background | 949 | 722 | 227 |
| 4.3 Methodology | 830 | 714 | 116 |
| 4.4 Memory boundaries | 1143 | 830 | 313 |
| 4.5 Runtime policy | 1458 | 1082 | 376 |
| 4.6a Window result | 1378 | 1052 | 326 |
| 4.6b Mechanism | 663 | 520 | 143 |
| 4.7 Fixes | 1246 | 1029 | 217 |
| 4.7p Shift rule | 519 | 468 | 51 |
| 4.8 Threats | 554 | 409 | 145 |
| 4.9 Conclusion | 552 | 440 | 112 |
| total | 10,763 | 8,360 | 2,403 |

Figure spec words in the body went from 725 to 523 (F3, F4 and F8 removed). The audit decisions added about 110
words, included in the "after" column. The body ends about a third of a column before the end of page 9, so real
figures taller than the placeholders would need a further cut.
