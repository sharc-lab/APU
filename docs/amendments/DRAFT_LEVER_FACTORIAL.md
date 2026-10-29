# DRAFT prereg: LEVER-FACTORIAL (policy x configuration, same entries)

DRAFT, NOT REGISTERED (Task AQ, 2026-10-15). Tests H-B (docs/THESIS.md):
configuration moves outcomes more than the routing policy does, with the
model held fixed and the entry set shared. A prereg with predictions, caps
and spend approval is needed before any run.

## Grid

Rows are the routing policies: slo_escalate, emission_escalate,
full_signal_bounceback, local_only. Columns are the configurations, all
with Qwen3-4B int4 on the XPS and entries DEMO1_BASE_30 (seed 20260930):

| column | configuration | source |
|---|---|---|
| C1 | GPU, u8, RESIDENT | d482c621 restricted to BASE_30 (sealed; no local_only arm) |
| C2 | CPU (cpu-p), RESIDENT | H1-CPU under amendment 2 (planned) |
| C3 | third configuration, one of the options below | new |

Options for C3 (Zach to pick one):
- Tuned DEMO-2 XPS profile: u4, RESIDENT, OBS_MASK K=1024, window 40,960.
  This shares code with DEMO-2 T-X. It changes four knobs at once, so it
  measures a profile, not one axis.
- OBS_MASK K=1024 alone on GPU u8 RESIDENT. One axis (retention).
  BASE_30 is identical under FULL and K=1024 (DEMO2 design), so on this
  pin the column is expected to equal C1. It is a negative control, not a
  lever.
- GPU u8 NON_RESIDENT. One axis (residency). It also matches the DEMO-1
  scheduler settings (PREFIX-CACHE-AUDIT).

Recommendation: GPU u8 NON_RESIDENT. It moves one axis, reuses DEMO-1
tooling, and separates residency from placement on the same 30 entries.

## Measures

Per cell: hybrid passes, $/task, local share, median local TTFT, peak
memory (needs the telemetry scoped under DEMO-2), and the matched-rate
random baseline (tools/aq_sealed_analyses.py).

H-B is supported if the largest configuration effect within any policy
row, |d passes| and |d $/task|, exceeds the largest policy effect within
any configuration column. That rule needs a threshold fixed in the prereg,
for example outside the random-baseline interval.

## Needs

Cell estimates, spend caps from d482c621 and V2 costs, run order,
predictions, and a C1 rerun if d482c621's session differences
(2026-09, RESIDENT) are judged too large.
