# Paper 1 / Demo Joint Plan

This file ties the Paper 1 measurement programme (docs/FINDINGS.md,
docs/CLAIMS_LEDGER.md, docs/PAPER_OUTLINE.md) to the investor/professor demo
(docs/DEMO_SPEC.md) on one shared timeline. Paper 1 submits Dec 2 2026. The
demo freezes Oct 28 2026. Both tracks share the same measured envelope: the
demo's router is not allowed to invent a hardware behavior that Paper 1 has
not measured, and every number the demo prints in front of an investor must
be traceable to a row in docs/CLAIMS_LEDGER.md.

A note on naming before the table: this document was written by reading
docs/FINDINGS.md, docs/CLAIMS_LEDGER.md, docs/PAPER_OUTLINE.md and
docs/PAPER_OUTLINE_DSE.md as they exist in this worktree today (2026-09-30).
Two labels used in the milestone spec handed down for this plan -- "R2" and
"MX2" -- do not appear anywhere in this repo's docs or code under those exact
names. Based on the closest matching content:
- **R2** is read here as the multi-turn recall experiment that
  docs/CLAIMS_LEDGER.md tracks as **Claim A-19** (Fig 4.15: score vs.
  artifact_distance x intervening_schema x artifact_size under partial KV
  eviction). That claim's own ledger entry says "Files: NONE -- harness not
  yet written; data not yet collected. Status: UNSUPPORTED," and
  docs/PAPER_OUTLINE.md Figure 4.15 says the harness is "NOT YET WRITTEN." A
  commit message on main outside this worktree ("r2: fix positive-control
  driver reading truncation_detected_turn from the wrong dict level")
  corroborates that an "r2" driver now exists somewhere in the project's
  history and concerns truncation-detection positive control, which lines up
  with the multi-turn/eviction design in docs/MULTITURN_DESIGN.md.
- **MX2** does not match any existing claim, figure, file, or script name in
  this repo. It is carried through the milestone table below as a labeled
  placeholder ("second cross-model or cross-machine control run, exact scope
  TBD") rather than invented. Whoever owns the Oct 14 milestone must pin down
  what MX2 is before that date; this plan cannot do that from repo contents
  alone.
- **K1 / K2**: K1 is confirmed as `harness/t2s_k1_ollama.py` (the evo-t2s
  model-tier/context classification harness; "t2s_k1_tier_v3" appears in main
  branch commit messages). That file does not exist in this worktree's
  checkout (this worktree is branched from an older point on main, commit
  `5ddecb4`, and is missing at least 5 newer commits visible in the
  conversation's git-status context, including the K1 v3 fixes). K2 is not
  named anywhere; it is carried as a placeholder the same way as MX2.
- **"kappa study"**: no file, script, or doc in this repo uses "kappa,"
  "Cohen," or "inter-rater" today. The probe suite's scorers
  (evaluation/probes/scorers.py) and the `classification_method` /
  `done_reason` fields used throughout docs/FINDINGS.md imply an obvious
  candidate -- a Cohen's-kappa agreement study between the automated
  exact-match/classification scorer and a human-labeled sample of the same
  rows (fabrication vs. abstention vs. correct vs. unclassifiable) -- but this
  is this document's inference, not a claim that the study is already scoped
  anywhere in the repo.

Where the table below cites a claim ID (A-NN, B-NN, J-01), that ID and its
file paths come directly from docs/CLAIMS_LEDGER.md. Nothing in this table is
invented; where no existing claim maps to a planned demo feature, the row
says so explicitly.

---

## 1. Claim -> demo feature map

| Paper 1 claim (docs/CLAIMS_LEDGER.md) | What was measured | Demo feature it powers |
|---|---|---|
| **A-02** Artifact truncation causes a sharp cliff in score (Fig 4.3) | Score drops ~1.0 -> ~0.0 once artifact survival fraction crosses a per-probe extinction threshold (`results/art_truncation.json`) | Router's **never-route-into-silent-truncation rule**: before routing a step to a local model, the router checks the step's projected prompt length against that model/runtime's measured extinction threshold and refuses (or escalates to cloud) rather than sending a request the envelope says will silently fail |
| **A-03** Extinction threshold is probe-specific, range 0.444-0.761 artifact fraction (Fig 4.4) | Fine-grained 1.00->0.85 ratio sweep (`results/partial_truncation.json`) | Per-workload-category threshold table the router consults instead of one global cutoff -- the demo's "quality floor" input is checked against the category-specific threshold, not a guessed constant |
| **A-04 / A-05** Mean required artifact fraction 0.211 (range 0.064-0.433); positional waste gap 0.255-0.551 (Fig 4.6) | Span-ablation minimum-sufficient-context measurement (`results/span_ablation.json`) | Context-budget sizing in the hardware recommender: "minimum provisioned GB for this workload" calculation (demo output #1) uses the required fraction, not the full artifact size, so the tool does not over-recommend memory |
| **A-07** art_04 needs an adjacent format exemplar, not just the answer span (Fig 4.6) | Three-way failure taxonomy: retrieval_failure / format_failure / correct (`results/span_ablation.jsonl`) | Failure-mode labeling in the live-run view: when a local run comes back structurally wrong but semantically right, the demo labels it `format_failure`, not a generic "wrong," matching the measured taxonomy instead of a pass/fail scorer |
| **A-08 / A-09** 100% fabrication, 0% abstention once context is extinct; an abstention instruction inverts its own intent (drives fabrication to 100%) (Fig 4.7/4.9/4.14) | Stage A scale run + self-report arms (`results/stage_a_scale.json`, `results/selfreport_arms.json`) | The "naive hybrid silently fails" half of the live-run demo scenario (docs/DEMO_SPEC.md): the naive baseline prompts the local model to "say so if you don't know," which this claim shows backfires, so the demo's naive path is scripted to fabricate with confidence, not to decline -- an accurate dramatization, not a strawman |
| **A-11** Type-matched filler interference is a recency/position effect, not truncation, at full context (Fig 4.8/4.14) | `results/stage_a_scale.json`, `results/interference_r120.json` | A second, independent failure mode the router must guard against beyond truncation: even with the artifact present, position-driven interference can still return a wrong value. Surfaced in the demo's failure-scenario screen as a distinct, labeled risk from truncation |
| **A-12** Measured KV memory reduction f16->q8_0 = 1.77x, f16->q4_0 = 3.24x, shallower than architectural 2x/4x (Fig 4.11) | `results/llamaserver_feasibility.json` (UNVERIFIED, script lost -- see ledger note) | Hardware recommender's memory-budget arithmetic: the tool sizes KV cache from the *measured* ratio, not the architectural one, when recommending a quantization level for a given hardware/model/context combination |
| **A-13** Budget_ratio x artifact position (EARLY vs LATE) is the productive quality-pressure axis, not raw depth (Fig 4.12) | Three-architecture replication, `results/stage_c_20260818T040408Z.jsonl` + two `fig61_stagec_full_*` files | The router's live state variable: instead of tracking "how long is this conversation," it tracks budget_ratio and where the load-bearing artifact sits in the window -- this is the actual signal the router conditions its routing decision on |
| **A-14 / J-01** Joint feasibility envelope: a (budget_ratio, score, latency) point is feasible iff score >= floor AND latency <= budget; f(workload type, quality floor) -> minimum provisioned GB (Fig 6.1, primary contribution) | `results/fig61_stagec_full_20260922T203557Z.jsonl` (evo-t2s, UNVERIFIED pending server-provenance re-check) | **This is the demo's core differentiator.** The Pareto frontier screen (docs/DEMO_SPEC.md) plots exactly this envelope: quality x cost x hardware, with the feasibility boundary taken from measured data, not assumed. This is the claim RouteLLM/RouterBench have no analogue of -- they route by model capability, not by a measured hardware-conditioned feasibility surface |
| **A-20** Discrete-GPU VRAM spill under the driver's Sysmem Fallback Policy: silent slowdown, no error, 10/10 correctness probes unchanged, decode slowdown = 0.08 + 74.6 x excess fraction (Fig 6.2 candidate) | `results/blade_m1_vram_spill_*.jsonl`, `results/blade_c1_spill_sweep_*.jsonl`, `results/blade_c2_spill_correctness_*.jsonl` | Memory-budget-wall warning in the live-run view: on a discrete-GPU hardware option, the demo shows the router refusing to push context past the measured VRAM spill onset rather than silently accepting a 4-10x decode slowdown the user would otherwise not notice until it was too late |
| **A-21** Host-side memory-bandwidth co-runner slows a fully VRAM-resident model 3.1-4.7x TTFT / 2.7-5.6x decode on the Blade; mechanism not established (Fig 6.2 candidate) | `results/blade_m2_host_interference_20260925T043921Z.jsonl` | Co-runner interference penalty applied to the recommended hardware option when the demo's workload description implies a concurrent host-side process (e.g. an agent that also does local file indexing); shown as a vendor-dependent latency multiplier, not a flat discount |
| **A-22** On unified-memory evo-t2s, locking RAM down to 7 GB produced no silent slowdown; one unrepeated crash at 5 GB (Fig 6.2 candidate) | `results/ramlock_evo-t2s_20260925T010739Z.jsonl` | Memory-budget-wall threshold for unified-memory hardware options in the recommender -- a different wall shape from the discrete-GPU case (A-20): no silent degradation region, then a hard crash boundary. The demo must show these as two different failure *shapes* per vendor, which is the point of using measured envelopes instead of one generic "out of memory" assumption |
| **A-23** CPU-only co-runner on evo-t2s drops the iGPU clock 2500->1650 MHz and slows TTFT 1.4x; mechanism open, package-power alone ruled out (Fig 6.2 candidate) | `results/t2s_m3_power_coupling_20260925T075348Z.jsonl` + overnight replication in docs/T2S_OVERNIGHT_REPORT.md | Vendor-dependent co-runner penalty table: on unified-memory Intel hardware, a background CPU load measurably steals iGPU clock headroom even with no package-power ceiling hit. The demo's hardware recommender applies this exact multiplier when scoring an Intel unified-memory option against a workload with a CPU-heavy co-runner, instead of assuming co-runner cost is hardware-independent |
| **A-24** llama-server on evo-t2s fails loudly (not silently) once requested memory exceeds the Vulkan heap *budget* (47,866 MiB), not the heap size or single-allocation limit (Fig 6.2 candidate) | `results/t2s_amech_20260926T181456Z.jsonl` | The exact numeric ceiling the hardware recommender uses for "will this model+context combination even start" on this GPU/runtime pair, instead of a guessed VRAM number from a spec sheet |
| **A-25 / A-26 / A-27** Default runtime fit-policy offloads layers to CPU at a 17% decode cost near the boundary rather than crashing; YaRN alone changes a probe's answer; default load mode is not file-backed/mmap on this Vulkan device, affecting crash behavior under memory pressure (Fig 6.2 candidates) | `results/t2s_overnight_20260926T011744Z.jsonl`, `results/t2s_amech_20260926T181456Z.jsonl` | Runtime-flag-sensitivity warnings surfaced in the demo's recommended routing policy: the tool does not just say "use this model on this box," it also pins the runtime flags (rope scaling, load mode) the envelope was measured under, because this evidence shows the same model+hardware pair behaves differently under different default flags |
| **A-01** Quality flat across depth 0-32k at fixed budget (Fig 4.1, supporting negative result) | `results/run_20260813T021516Z.jsonl` | Justifies *why* the router conditions on budget_ratio x position (A-13) rather than raw context length -- documented in the demo's methodology footnote, not a user-facing feature by itself |
| **A-18** Parametric-default failure class: models emit canonical field-type sentinels (8080, 0, INT_MAX) when the answer span is absent, not fabricated noise (Table 4.1) | `results/span_ablation.jsonl` | Failure-mode classifier in the live-run view distinguishes `parametric_default` from `free_fabrication` -- both are wrong, but a demo audience sees the distinction between "it guessed a plausible-looking default" and "it made something up with no basis," which is a credibility point for the product's diagnostic value |
| **A-19** Multi-turn recall vs. artifact distance / schema / size under partial KV eviction (Fig 4.15) -- **UNSUPPORTED, harness not written** | None -- `docs/PAPER_OUTLINE.md`: "harness not yet written; data not yet collected" | Nothing yet. This is the claim the "R2" milestone label most plausibly refers to (see naming note above). The demo cannot claim a multi-turn eviction-aware routing feature until this exists; until then the demo's routing policy must restrict itself to single-turn/short-horizon claims it can actually back with A-02 through A-14 |
| **J-01** (restated above with A-14) f(workload type, quality floor) -> minimum provisioned GB | Two-architecture replication (Blade off-target, evo-t2s on-target) | Hardware recommender's headline output: "recommended hardware" in docs/DEMO_SPEC.md is this function, evaluated at the user's quality floor and workload category |
| **B-03** Independent replication (Zachary Johnson) cross-validates span-attribution methodology (Fig 5.2) | `results/zachary/replication_remote_search_v3.json` | Not demo-facing directly; backs the orchestration-overhead numbers (ORCH_SETUP/HTTP_CLIENT/TOOL_COMPUTE) that feed the latency-target input on the demo's input form, so the latency budget check is against a validated decomposition, not a single uncorroborated harness |
| **B-04** TTFT and http_client_ns measured on the same call as quality score, enabling per-call joint envelope points (Fig 6.1) | `results/fig61_stagec_full_20260922T203557Z.jsonl` | The methodological guarantee behind the Pareto frontier screen: every point plotted is one real call's (quality, latency, cost) triple, not a mean-of-means composite that could misrepresent the frontier |

Claims **B-01 / B-02** (Axis B span data, gitignored, not committed) and claim
A-15/A-16/A-17 (model-dependence and mechanism notes on cha_04/lon_02) are
Paper-1-internal validity items with no direct demo feature; they are not
listed above because they do not power a demo screen.

---

## 2. Milestone table

Dates and milestone content below are fixed by the task that produced this
document and are reproduced here exactly, with file-level deliverables added
from this repo's actual structure (scripts that exist or that the milestone
implies must exist, and docs that must exist by that date).

### Oct 7 2026

| Deliverable (as specified) | Concrete files expected to exist |
|---|---|
| K1 v3 both machines | `harness/t2s_k1_ollama.py` (confirmed on main outside this worktree; must be merged/present in whatever checkout runs the Oct 7 measurement) run and committed for both evo-t2s and the second "both machines" target (EVO-X2, per docs/TIMELINE.md); `results/t2s_k1_*<timestamp>*.jsonl` + `.DONE` sentinel + manifest, for each machine |
| R2 first run | Multi-turn recall harness (consumes `evaluation/probes/multiturn.jsonl`, does not exist yet per docs/PAPER_OUTLINE.md Fig 4.15) must be written before this is possible; first output would be a `results/multiturn_<timestamp>.jsonl` file. **This is the highest-risk Oct 7 item**: as of today the harness has not been started in this worktree |
| Traces final | `harness/replay/cache.py` trace set frozen; the replay traces `evaluation/sweep.py` and `evaluation/certify.py` depend on should stop changing shape after this date |
| Kappa study | A new script (e.g. `analysis/kappa_agreement.py`) comparing `evaluation/probes/scorers.py` classifications against a human-labeled sample; output `results/kappa_agreement.json` and a short `docs/KAPPA_STUDY.md` note. Nothing with this name exists yet |
| Demo spec | `docs/DEMO_SPEC.md` (this task produces it) |
| Cloud API with cap | `src/cloud/client.py` (this task produces it), `results/cloud_ledger.jsonl` |
| Workload pack | A set of example workload descriptions for the demo's input form and the Pareto-frontier demonstration; does not exist yet -- needs a new `demo/workload_pack/` or similar, scoped by whoever owns the demo build |
| All-cloud pass | `routing/policies/all_cloud.py` (exists) exercised end-to-end against the stub/mock cloud provider in `src/cloud/client.py`, producing a `results/pareto_results.json`-shaped output with the `all_cloud` policy populated (stub values, clearly labeled, since no real key is used per this task's constraints) |

### Oct 14 2026

Paper 1 core experiments complete: R2, K2, MX2, controls, failure-map gaps.

| Deliverable | Concrete files expected |
|---|---|
| R2 complete | `results/multiturn_<timestamp>.jsonl` with full probe grid from `evaluation/probes/multiturn.jsonl`; Claim A-19 updated from UNSUPPORTED to OFF-TARGET-ONLY or ON-TARGET in docs/CLAIMS_LEDGER.md |
| K2 | Not named anywhere in this repo; placeholder for a second evo-t2s/EVO-X2 tier or kappa-style run in the K-series started by K1. Must be scoped before Oct 7 ends |
| MX2 | Not named anywhere in this repo; placeholder, likely a second cross-model (per A-15, Fig 4.13) or cross-machine control run. Must be scoped before Oct 7 ends |
| Controls | Positive-control checks already in the pattern of `docs/RESULT_PROVENANCE.md`'s "stale-server exposure audit" -- every result file produced this cycle must pass `harness/server_guard.py`'s three checks (port-free, /props match, PID match) so it does not join the UNVERIFIED list the way five existing fig61_* files already have |
| Failure-map gaps | Fig 6.2 candidate claims (A-20 through A-27) still have open items per docs/CLAIMS_LEDGER.md: A-20's forced-failure half (C3, driver setting flip) is PENDING; A-23's mechanism is OPEN. These gaps should be closed or explicitly re-labeled by this date |
| Local outcome tables, both machines | A committed `results/outcome_table_blade_rtx4070.json` and `results/outcome_table_evo-t2s.json` (or EVO-X2 equivalent) aggregating score/latency/memory outcomes per probe x hardware, feeding both the paper's headline table (Oct 21) and the demo's recommender lookups |
| Baselines | `routing/policies/all_cloud.py` and `routing/policies/all_local.py` (both exist) run against the workload pack and committed as `results/baseline_all_cloud.json` / `results/baseline_all_local.json` |

### Oct 21 2026

Envelope model validated; Paper 1 headline table measured from the outcome table; router, Pareto, failure scenarios.

| Deliverable | Concrete files expected |
|---|---|
| Envelope model validated | Figure 6.1 analysis script (does not exist yet per docs/PAPER_OUTLINE.md: "Analysis / plot: must be written") -- e.g. `analysis/envelope_model.py` reading the Oct 14 outcome tables and fitting/validating the feasibility boundary |
| Paper 1 headline table | A new `docs/HEADLINE_TABLE.md` or table embedded in the paper draft, built from `results/outcome_table_*.json`, cross-referenced against docs/CLAIMS_LEDGER.md so every headline number carries its claim ID |
| Router | `routing/policies/*` (all seven policies already exist: all_cloud, all_local, static_category, cascade, budget_aware_cascade, speculative, learned_router) wired to read the envelope model's feasibility boundary as live state, not a static threshold -- this is the integration step connecting Paper 1 data to Paper 2/demo code, and does not exist yet |
| Pareto | `analysis/pareto.py` (exists) and `analysis/bom_sweep.py` (exists) run against real outcome-table + cloud-ledger data (not just the stub) to produce `results/pareto_results.json`, which docs/PAPER_OUTLINE_DSE.md currently lists as "ABSENT from repo and disk" |
| Failure scenarios | The live-run "naive hybrid silently fails vs ours succeeds" scenario (docs/DEMO_SPEC.md) implemented against real data rather than the scripted walkthrough drafted for the Oct 7 demo spec |

### Oct 28 2026

All figures final, shared by paper and demo; demo freeze.

| Deliverable | Concrete files expected |
|---|---|
| All figures final | Every `analysis/plot_*.py` script that docs/PAPER_OUTLINE.md marks "needs one" must exist and be run: Figs 4.4, 4.5, 4.7, 4.8, 4.9, 4.15 have no dedicated plot script today. Output PDFs/PNGs land in `figs/` (directory exists, currently sparse) |
| Shared by paper and demo | Figures used in investor materials must be the same artifact files the paper cites -- no separate "demo-only" chart with different numbers. Practically: `figs/` becomes the single source both docs/PAPER_OUTLINE.md and docs/DEMO_SPEC.md point at |
| Demo freeze | No further feature changes to the demo build after this date; `docs/DEMO_SPEC.md` is the frozen spec and any gap between it and the running demo is a bug, not a pending feature |

### Nov 10 2026

Full Paper 1 draft to Callie.

| Deliverable | Concrete files expected |
|---|---|
| Full draft | A `docs/PAPER_DRAFT.md` or equivalent (LaTeX source if the venue requires it) covering all seven sections in docs/PAPER_OUTLINE.md, with every CORE and SUPPORTING claim either ON-TARGET, OFF-TARGET-ONLY with explicit hardware labeling, or removed. Per docs/CLAIMS_LEDGER.md's own tracking table ("Claims requiring Strix Halo data before submission"), J-01, A-02, A-01, A-12, A-04/A-05, A-13 must all be resolved against EVO-X2 by this date or the draft states the off-target limitation explicitly, per the TIMELINE.md fallback-options framing |

### Dec 2 2026

Submit.

| Deliverable | Concrete files expected |
|---|---|
| Submission | Final PDF + supplementary material package. Note: docs/TIMELINE.md currently estimates the ISPASS 2027 full-paper deadline at ~Dec 14-15, not Dec 2 -- this plan uses the Dec 2 date given in this task's instructions, which is earlier than TIMELINE.md's estimate. Whoever owns the submission should reconcile which date is authoritative (a firmer CFP may have appeared since docs/TIMELINE.md was last updated on 2026-09-24) before treating Dec 2 as fixed |

---

## 3. Current status vs. plan (as of 2026-09-30)

This worktree's `git log -1` is commit `5ddecb4` ("harness: fix real bugs
found by tonight's read-only check..."), dated 2026-09-27. The conversation's
git-status context shows at least five newer commits on main
(`655a92a`, `f29d572`, `f9a2c63`, `5c08396`, `314eb56`) that are **not**
present in this worktree's history, including the K1 v3 fixes
("k1: fix the real cause of t2s_k1_tier_v3 measuring zero models on
evo-t2s," "k1 v3: fix truncation labeling... and add GPU device detection").
The status below is reported against what is actually checked out here; the
items marked "ahead on main" are known to exist on main but not in this copy.

### Oct 7 items: what already exists

- **K1 v3, partial / ahead on main**: `harness/t2s_k1_ollama.py` and
  `harness/host_config.py` do not exist in this worktree's checkout. Per the
  commit messages visible in the conversation's git-status context, main has
  already fixed the "zero models measured" bug (PATH resolution for
  `ollama.exe` under WMI-launched/non-interactive SSH sessions, and a
  pull-error-message bug) and added truncation labeling (A1) and GPU device
  detection (A2) fixes, with claimed "65/65" and "58/58" test passes. **Not
  verified from this worktree** -- no K1 result files exist under `results/`
  here (`ls results/ | grep -i k1` returns nothing), so "both machines" data
  collection has not started in what this worktree can see.
- **Traces**: `harness/replay/cache.py` exists and has a test
  (`tests/test_replay_cache.py`); whether the trace set is "final" is a
  content question this plan cannot verify from file presence alone.
- **Demo spec**: did not exist before this task; `docs/DEMO_SPEC.md` is
  produced alongside this document.
- **Cloud API with cap**: did not exist before this task;
  `src/cloud/client.py` is produced by this task (see commit for B2).
- **All-cloud pass**: `routing/policies/all_cloud.py` exists and (per
  `tests/test_cascade_policies.py` / `tests/test_routing_budget_smoke.py`
  naming) appears to have test coverage already, but has not been run
  end-to-end against any cloud client (real or stub) as of this worktree's
  state -- `src/cloud/` did not exist until this task created it.

### Oct 7 items: still needed

- **R2 (multi-turn recall) first run**: not started. docs/PAPER_OUTLINE.md
  Figure 4.15 explicitly states the harness consuming
  `evaluation/probes/multiturn.jsonl` is "NOT YET WRITTEN," and
  docs/CLAIMS_LEDGER.md's Claim A-19 lists "Files: NONE... Status:
  UNSUPPORTED." This is the single highest-risk Oct 7 item: it requires
  writing a harness, not just running an existing one, and docs/TIMELINE.md
  (written before this plan's Dec 2 deadline was communicated) already flags
  it as needing "an independent two-week implementation."
- **Kappa study**: no file or doc in this repo uses "kappa" anywhere. Needs
  to be scoped (this plan proposes: Cohen's kappa between the automated
  scorer in `evaluation/probes/scorers.py` and a human-labeled sample) and
  then built.
- **Workload pack**: no `demo/` directory or workload-description set exists
  in this repo today.
- **K2 / MX2** (named in the Oct 14 milestone but worth flagging now): these
  labels appear nowhere in this repo. They cannot be scheduled against real
  scripts until someone defines what they are; this plan has not invented a
  definition for them.

### results/ and analysis/ state relevant to this plan

`results/` currently holds 116 files/dirs, essentially all from the Paper 1
Axis A/B and Fig 6.2-candidate sweeps on blade_rtx4070 and evo-t2s (dated
2026-08-13 through 2026-09-26), plus three new untracked files from today's
session (`results/blade_m1_vram_spill_20260925T041415Z.DONE`,
`results/contention_blade_rtx4070_20260924T171907Z.DONE`,
`results/t2s_responsiveness_20260928T072710Z.jsonl`). None of these are
multi-turn, K1/K2, MX2, or kappa files. `analysis/` has the Pareto and BOM
sweep scripts (`pareto.py`, `bom_sweep.py`) but `results/pareto_results.json`
does not exist, matching docs/PAPER_OUTLINE_DSE.md's own "ABSENT from repo
and disk" note -- the demo's Pareto frontier screen has no real data to plot
yet and will run against the stub cloud client and synthetic/baseline policy
runs until Oct 14's "baselines" deliverable lands.
