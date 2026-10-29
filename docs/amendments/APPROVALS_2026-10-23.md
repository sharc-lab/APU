# Approvals recorded 2026-10-23 (Task BH)

Status: REGISTERED. Written Thu Oct 22 2026, late evening ET, by Claude Code (Task BH). The file is
dated 2026-10-23 because that is the freeze day. It records approvals only, edits no registered
file, and runners do not read it.

Verbatim (Zach, chat, Thu Oct 22 2026, about 10:20 PM ET):

> ok register everything new

Scope, as written in ~/seam_closeout_1013/papers_20261015/TASK_BH_REGISTER.md:
1. The DEMO-2 amendment docs/amendments/DEMO2_AMEND_2026-10-23_OBJECTIVES.md:
   the P vs I-4K prediction check, and a non-dollar confirmatory family F2.
2. Registration of experiments E1 and E2 from the research report
   (~/seam_closeout_1013/research_20261021/Hardware_levers_for_large_cost_wins.md) as
   docs/registrations/CTXKV_E1E2_2026-10-23.md. This is a separate experiment, not part of DEMO-2.

Not approved by this:
- Rithwik has NOT yet agreed to the AMD arms or the T2S arms of E1/E2. They stay pending until he
  confirms in writing and supplies the task source.
- No launch, no reboot, and no cloud spend. E1's cloud-only log needs its own spend approval
  before it runs.
- No change to DEMO-2's arms, pins, budgets, PSPEC, HO_CLOUD_CAP, the replay or the dollar
  primary.

## Task BJ part A (Fri Oct 23 2026, about 10:45 AM ET)
- Zach asked (TASK_BJ_REGISTRY_REBUILD.md) to rebuild the plan and run DEMO-2 held-out. Recorded as:
  "P' rows, approved 2026-10-23". Scope: docs/amendments/DEMO2_AMEND_2026-10-23b_PPRIME.md (XPS P'
  rows for P vs R-full / I-4K / C, DERIVED, same generator) and a dated schedule note (T2S held-out
  first; XPS held-out deferred until after Oct 27). Nothing else changes.
- Not approved by this: any spend beyond the registered held-out cap, a llama.cpp install on the
  T2S, E2-T2S, or anything on the T2S after Oct 25 without Rithwik's written OK.

## Task BQ (Fri Oct 23 2026, about 5:10 PM ET; recorded verbatim from TASK_BQ_BALLOON_GATE.md)
1. R1: "Keep going". Change nothing. R1 is recorded as RED FLAG by the registered range, with the BP
   diagnosis (window-level B shift: u4 and i8 both imply B = 3.22 GiB vs 3.412; ratio holds). Add a
   dated entry to the PLAN_LOCK red-flag log and AUDIT_LOG. No prediction, band or threshold is changed.
2. Balloon gate: "Approve option (a)" from the BP report.
- Scope: option (a) only (docs/amendments/CC_BALLOON_GATE_2026-10-23f.md). The 7,000 MB floor, balloon
  sizes, points, bands and predictions are unchanged.
