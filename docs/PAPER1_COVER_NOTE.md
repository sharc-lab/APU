# Paper 1 Cover Note (not anonymized, not part of the submission)

Author: Rithwik Sharma.

This note goes with `docs/PAPER1_DRAFT.md`, which is written double-blind (no author, affiliation, machine owner
or repository named). Everything that would identify the work lives here instead.

## Label mapping used in the draft

| draft label | internal host | notes |
|---|---|---|
| INTEL-iGPU | evo-t2s | Core Ultra X7 358H (Panther Lake), Arc B390 iGPU. The operator brief calls it an "Intel iGPU laptop"; `docs/FINDINGS.md` (M3 section) calls it a mini PC. The draft avoids the form factor until this is confirmed. |
| AMD-iGPU | evo-x2 | Ryzen AI Max+ 395 (Strix Halo), Radeon 8060S iGPU, mini-PC. |
| NVIDIA-dGPU | Blade 14 | RTX 4070 Laptop GPU. |

Plan-item labels in the draft ("T2S week item N", "Blade night N (x)") refer to `docs/T2S_WEEK_PLAN.md` and
`docs/BLADE_PLAN.md`, which other agents were writing at the same time as this draft; the item names follow the
operator's list (T2S items 1-6: re-enable checks, R2 native Intel, mitigation at 4096, Intel bandwidth hog,
outcome-table subset, mechanism run; Blade night 1: (a) K1, (b) R2 native, (c) mitigation; night 2: (d) C3
sysmem fallback, (e) mechanism). If those plans renumber, update the draft's references.

## Before converting to LaTeX

- Replace host labels and result-file names inside pasted register values with the draft labels.
- Remove queue-job and plan-item names; keep only the experiment descriptions.
- The repository link goes in only after review (camera-ready or the artifact appendix, per the CFP's
  double-blind rules).

## Open items for the advisor meeting

- Deadline: `docs/TIMELINE.md` estimates abstract around 2026-12-07 and full paper around 2026-12-14/15;
  `docs/PLAN_PAPER1_DEMO.md` uses 2026-12-02. Not resolved here.
- Page limit: 10 pages plus references assumed, unconfirmed until the ISPASS 2027 CFP is out.
- Scope decision: whether the paper keeps the three-vendor framing if the NVIDIA-dGPU rows (Gap G4) do not land by
  2026-11-01, or narrows to the two unified-memory platforms.
