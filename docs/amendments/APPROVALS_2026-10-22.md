# Approvals recorded 2026-10-22 (Task BF0)

Status: REGISTERED 2026-10-22. Zach approved the items below in chat on Thu Oct 22 2026 at about
1:15 PM ET ("yes on all"), as written in ~/seam_closeout_1013/papers_20261015/TASK_BF0_FIXPACK.md.
They answer the pre-freeze audit (~/seam_closeout_1013/research_20261021/DEMO2_pre_freeze_audit.md).
This file records approvals only. It edits no registered file, and runners do not read it.
The rules they approve are written in docs/amendments/DEMO2_PREFREEZE_2026-10-22.md.

Verbatim:

1. Arm C / L6 gate (X5): dated deviation. C runs as specified (age-mask on); the L6 gate was
   never evaluated (ladder ran UNTIL=4). No L6 cells are run.
2. Deliverable B and consumer half of A (X6): register P vs R-full and P vs I-4K as SECONDARY
   comparisons (also C vs R-full, P vs C), each with a DERIVED prediction, via `--reference`.
3. Escalation cost model (X7): the warm splice is primary, named "splice (cache-optimistic)";
   full re-run and cold handoff are registered sensitivities. Report the share of escalations
   at turn > 0.
4. Multiplicity (X3): P vs A dC_iso is the ONLY confirmatory test per host, tested in fixed
   sequence T2S then XPS (XPS tested only if T2S rejects; each at alpha 0.05). Everything else
   is secondary, gated on the primary, reported with unadjusted CIs and labelled secondary.
