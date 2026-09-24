# Submission Timeline — Paper 1

## ISPASS 2027 deadlines

ISPASS 2026 was held April 26–28, 2026 (Seoul). The 2026 abstract deadline was
December 8, 2025; full-paper deadline was December 15, 2025.
ISPASS 2025: abstract deadline December 9, 2024; full-paper December 16, 2024.

**ISPASS 2027 has not published its call for papers as of 2026-09-22.** No official
dates are available at ispass.org or SIGARCH. Based on the stable historical pattern
(conference in late April, abstract deadline ~first week of December, paper ~one week later):

- **Estimated abstract deadline: ~2026-12-07**
- **Estimated full-paper deadline: ~2026-12-14 or 2026-12-15**

**An abstract submission is required before the full paper can be submitted. Plan
for the abstract deadline; missing it forecloses the full submission.**

These are estimates only. Monitor ispass.org starting October 2026. If ISPASS 2027
shifts to May or later, the effective window extends by ~4 weeks; if the venue
changes, all dates below must be recalculated.

**Conservative planning dates used below: abstract due 2026-12-07; full paper due 2026-12-15.**

---

## EVO-X2 (Strix Halo) availability

**UPDATE 2026-09-24: EVO-X2 is in hand now, not arriving December 3.** The
BOM target (AMD Ryzen AI Max+ 395, Strix Halo, 128 GB unified LPDDR5X,
identified in `configs/hardware/evox2_strix_halo_128gb.yaml`) is physically
available today, roughly **11 weeks before** the estimated full-paper
deadline (~2026-12-15) rather than the ~12 days this document previously
assumed. Every "EVO-X2 BLOCKED" / "cannot realistically run before deadline"
framing below is now stale and should be re-evaluated — CORE figures that
were assumed off-critical-path may now be achievable on-target before
submission. The week-by-week schedule below has been adjusted to start
EVO-X2 setup immediately rather than waiting for a Sep29–Oct5 delivery
window that has already passed; the downstream weeks were NOT otherwise
re-planned in this pass and should be revisited once setup (below) is
confirmed working and initial runtime is known.

Setup is being done via `scripts/setup_evox2.ps1` (Windows, matching the
balloon/WMI-launch tooling already built for evo-t2s, which is
Windows-specific) — see that script and its prerequisites checklist before
running it once at the machine's own keyboard.

**Prior framing (superseded, kept for context):** the paper was planned to
not depend on Strix Halo data for submission, with off-target results
(Blade 14, evo-t2s) as the primary path and EVO-X2 data treated as a
stretch goal. With ~11 weeks of runway now available instead of ~12 days,
on-target CORE data is plausible before the deadline — but this has not yet
been re-confirmed by actually running anything on the machine, so treat the
sections below as not-yet-updated until real EVO-X2 timing data exists.

---

## Week-by-week schedule

| Week | Dates | Tasks | Depends on |
|---|---|---|---|
| **W1** | Sep 22 – Sep 28 | Complete 7-step evo-t2s replication (matched checkpoint, streaming, baseline gate, 396-call full sweep). Commit valid Fig 4.12 evo-t2s data. Update RESULT_PROVENANCE.md. **EVO-X2 arrived during this window (2026-09-24) — setup below starts immediately, not in W2.** | evo-t2s server available |
| **W2** (starts early, ~Sep 24) | Sep 24 – Oct 5 | EVO-X2 setup via `scripts/setup_evox2.ps1` (OpenSSH, Tailscale, Python 3.12 + numpy/pandas/pytest, llama-server b10970 Vulkan build + instruct GGUF with sha256 verification). Run `scripts/verify_platform.py`; populate `evox2_strix_halo_128gb.yaml` fields (reserved_gb, achievable_pool_gb, bandwidth_gb_s). Confirm llama-server Vulkan build runs on the AMD iGPU. | **EVO-X2 now in hand — no longer gated on delivery** |
| **W3** | Oct 6 – Oct 12 | EVO-X2: Fig 4.1 (quality vs depth, main quality sweep — `harness/runner.py`, 11 probes × 7 depths × 5 reps ≈ 385 calls ≈ 4–6 h). Confirm flat-at-depth negative result replicates. **[EVO-X2 BLOCKED]** | W2 complete |
| **W4** | Oct 13 – Oct 19 | EVO-X2: Fig 4.3 (artifact truncation cliff — `harness/art_truncation.py`). Confirm per-probe cliff positions. **[EVO-X2 BLOCKED]** | W3 |
| **W5** | Oct 20 – Oct 26 | EVO-X2: Fig 4.11 (KV precision gate — `harness/stage_a_kv_precision.py` via llama-server, `--cache-type-k` flags verified). Confirm f16→q8→q4 reduction ratios on AMD unified path. **[EVO-X2 BLOCKED]** | W3 |
| **W6** | Oct 27 – Nov 2 | EVO-X2: Fig 6.1 (joint feasibility envelope — `harness/fig61_sweep.py` updated for streaming + matched checkpoint). 11 probes × 6 ratios × 2 arms × 3 reps = 396 calls ≈ 5–8 h. **[EVO-X2 BLOCKED]** | W3 |
| **W7** | Nov 3 – Nov 9 | EVO-X2: Fig 4.6 (span ablation), Fig 4.12 (position pressure replication). Supporting figures. **[EVO-X2 BLOCKED]** | W4 |
| **W8** | Nov 10 – Nov 16 | EVO-X2: Fig 4.13 (cross-model replication). Write multi-turn harness (`evaluation/probes/multiturn.jsonl` consumer). If harness is complete: run Fig 4.15 (multi-turn recall). **[EVO-X2 BLOCKED + HARNESS BLOCKED]** | W7 |
| **W9** | Nov 17 – Nov 23 | Write all plot scripts for figures without them (Fig 4.4, 4.5, 4.7, 4.8, 4.9, 4.15). Produce all figure PDFs. Commit Axis B data (Fig 5.1, 5.2, Table 5.1) from gitignored files or re-run. | all data committed |
| **W10** | Nov 24 – Nov 30 | Full paper draft. Write Section 6 (joint envelope). Cross-check CLAIMS_LEDGER.md — every claim must be SUPPORTED-ON-TARGET or explicitly labeled off-target in the paper text. | W9 |
| **W11** | Dec 1 – Dec 7 | Internal review. Resolve all THREATS.md items that are still open. Update FINDINGS.md with any new disconfirmations. **Submit abstract by ~Dec 7.** | W10 |
| **W12** | Dec 8 – Dec 14 | Revision, author check, IEEE formatting, final submission preparation. EVO-X2 arrives ~Dec 3 — if provisioned fast, can run one CORE figure sweep this week (opportunistic, not on critical path). | W11 + abstract submitted |
| **Submit** | Dec 14–15 | **Full paper submitted.** Abstract prerequisite: must have been submitted by Dec 7. | W12 |

---

## Items that depend on EVO-X2 provisioning (critical path)

| Figure | Claim | Risk if EVO-X2 slips |
|---|---|---|
| Fig 6.1 (CORE) | Joint envelope J-01 | Cannot make primary contribution claim |
| Fig 4.3 (CORE) | Artifact truncation cliff A-02 | Core evidence for fabrication under extinction |
| Fig 4.1 (CORE) | Quality flat at depth A-01 | Supporting negative result that motivates budget_ratio framing |
| Fig 4.11 (SUPPORTING) | KV quantization ratios A-12 | Off-target measurement; AMD unified path unvalidated |
| Fig 4.6 (SUPPORTING) | Span ablation A-04/A-05 | Required fraction may differ on unified memory |
| Fig 4.12 (SUPPORTING) | Position pressure A-13 | evo-t2s replication (W1) is off-target; EVO-X2 needed |
| Fig 4.15 (SUPPORTING) | Multi-turn recall A-19 | Harness also not written — two independent blockers |

**Any EVO-X2 slip past November 2 (end of W6) means Fig 6.1 data arrives after W6, and the
paper draft in W10 is written against placeholder figures.** This is technically possible but
high-risk for the December 15 deadline.

---

## Items with independent blockers (not EVO-X2)

| Figure | Claim | Blocker |
|---|---|---|
| Fig 4.15 (SUPPORTING) | Multi-turn recall A-19 | Harness not written — requires ~1 week implementation before EVO-X2 run |
| Table 5.1 / Fig 5.1 (SUPPORTING) | Axis B spans B-01/B-02 | Data gitignored and not committed; need API key machine or re-run |
| Fig 4.12 (SUPPORTING) | Fig 6.1 matched evo-t2s replication | evo-t2s 7-step sequence not yet run (W1 task) |

---

## Submission options (ordered by preference)

**SUPERSEDED 2026-09-24**: this section was written assuming EVO-X2 arrives ~December 3
with ~12 days of runway, making on-target CORE runs before the deadline unrealistic. EVO-X2
is now in hand with ~11 weeks of runway instead. On-target CORE data before the deadline is
plausible and should be the actively pursued path once setup (`scripts/setup_evox2.ps1`) is
confirmed working, not treated as a stretch goal. The options below are kept for the
no-longer-primary scenario where EVO-X2 setup or runs slip badly enough to eat most of that
runway anyway (e.g. hardware issues, driver/build problems specific to that unit) — re-derive
the actual critical path once real EVO-X2 run timings exist, rather than trusting the
"cannot realistically run before deadline" framing this paragraph previously asserted.

**Primary plan (preferred):** Submit to ISPASS 2027 with off-target data clearly labeled.
All CORE figures (6.1, 4.3, 4.1) are reproduced on evo-t2s and/or Blade 14 with explicit
hardware labels ("off-target: Intel Arrow Lake unified LPDDR5X" / "off-target: RTX 4070
discrete"). The abstract and paper state that Strix Halo re-runs are planned post-submission
and can be presented at the conference. Risk: reviewers may request on-target data as a
condition of acceptance. But the methodology, harness, probe suite, and two-platform
characterization are a complete and reproducible contribution.

**Fallback options (if primary plan is rejected or weakened):**

   a. **Submit with evo-t2s data labeled explicitly as off-target**, with a mandatory
      statement that Strix Halo re-runs are planned. Reviewers may reject on grounds that
      the BOM target claim is unsupported. Risk: moderate — the contribution is real; the
      labeled hardware limitation is honest; reviewers may accept with a revision request.

   b. **Descope to a methodology paper framing**, removing the Strix Halo provisioning
      framing entirely and presenting as a harness + characterization paper. The f(workload,
      quality_floor) → GB claim becomes a methodology for deriving the GB figure, not a
      specific figure on the BOM target. Risk: weaker claim, but stronger evidence base.

   c. **Target a different venue with a later deadline** (e.g., MICRO 2027 abstract
      ~April 2027, or ISCA 2027 ~January 2027 — verify these dates independently).

   d. **Submit as an ISPASS tool and benchmark paper.** ISPASS explicitly solicits tool
      and benchmark papers judged on their potential to enable future research, with the
      tool open-sourced before the conference. The harness (llama-server backend, streaming
      TTFT measurement, probe suite) plus two-architecture coverage (Intel Arrow Lake unified
      LPDDR5X + RTX 4070 discrete) is a stronger tool-track submission than a partial
      full-paper with off-target CORE figures. This is a stronger option than (a) or (b) if
      the primary plan encounters reviewer resistance. The harness is already open; the two
      platforms are already characterized.

**APPENDIX figures are not affected by any option** — all off-target Blade 14 data is
already committed and can be submitted as labeled appendix material.

**The methodology claim survives in all options**: the harness, probe suite, and analysis
infrastructure exist and produce valid measurements on any POSIX host with llama-server.
This is a durable result regardless of hardware delays.

---

## Highest-risk timeline items

1. **SUPERSEDED 2026-09-24 — EVO-X2 is in hand, not arriving December 3.** ~11 weeks of
   runway are available, not ~12 days. On-target CORE data before the deadline is plausible
   and is now the path to actively pursue via `scripts/setup_evox2.ps1`, not a stretch goal
   to be waved off with off-target labeling. The remaining real risk is whether setup and
   the actual CORE sweeps (Fig 6.1, 4.3, 4.1, etc.) run cleanly on this specific unit in
   time — re-assess once setup is confirmed and initial run timings on EVO-X2 are known.

2. **Fig 4.15 multi-turn harness not written** — independent two-week implementation
   required before any run can begin. If this starts in W8 at EVO-X2, it cannot produce
   committed data before W10 when the paper draft must start.

3. **Axis B data gitignored** — Table 5.1 and Fig 5.1 are not reproducible from the
   repo in their current state. If the machine holding those files is lost or the API key
   expires, Axis B has no data. This must be resolved in W9 at the latest; ideally committed
   in W1–W2.
