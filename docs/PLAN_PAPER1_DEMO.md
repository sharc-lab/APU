# Paper 1 / Demo Joint Plan

This file ties the Paper 1 measurement programme (docs/FINDINGS.md,
docs/CLAIMS_LEDGER.md, docs/PAPER_OUTLINE.md) to the investor/professor demo
(docs/DEMO_SPEC.md) on one shared timeline. Paper 1 submits Dec 2 2026. The
demo freezes Oct 28 2026. Both tracks share the same measured envelope: the
demo's router is not allowed to invent a hardware behavior that Paper 1 has
not measured, and every number the demo prints in front of an investor must
be traceable to a row in docs/CLAIMS_LEDGER.md.

A note on naming before the table: an earlier pass of this document was
written from a stale worktree checkout (branched from main commit `5ddecb4`,
missing the K1 v3 fixes and everything after), which could not find R2, K2
or MX2 anywhere and treated them as unscoped placeholders. Corrected
2026-10-01 against real main:
- **R2** is `harness/t2s_r2_session_growth.py` -- the multi-turn agent
  session growth experiment (turn-0 system prompt with 5 checkable rules, a
  2-tool schema, 3 recall facts, up to 80 turns, canary-based truncation
  detection comparing cumulative sent tokens against the runtime's actually
  loaded context). It is real, built, and has run this session: its positive
  control (`harness/t2s_r2_positive_control.py`) was confirmed firing
  correctly on both the Ollama and llama-server runtimes (turn 10,
  loaded_context_tokens=8192, for llama31-8b) on 2026-10-01, after fixing two
  real bugs (a `host_cfg['name']` KeyError and a missing `keep_alive` that
  was silently unloading the model between every turn on evo-x2). `x2_r2` is
  currently running on evo-x2 (first-run scope: llama3.1:8b, arms a-d,
  as_is and occupied_40gb, 3 seeds); `t2s_r2` is queued on evo-t2s (as_is
  conditions only). This is a separate, already-built experiment from Claim
  A-19 (the multi-turn KV-eviction recall claim, still genuinely
  UNSUPPORTED/harness-not-written) -- the two should not be conflated; R2
  tests rule-compliance/tool-call/recall decay under session growth plus the
  truncation-detection mechanism itself, not KV-eviction-under-partial-
  context specifically.
- **K2** is `harness/t2s_k2_pressure.py` -- the everyday-apps memory-pressure
  experiment, now including a pause-and-resume arm (arm d): run session
  turns 1-10, open a Chromium-driven browser memory load at a chosen GB
  step, idle past `keep_alive` so the model unloads, resume turns 11-30, and
  record whether the reload silently changes GPU layer placement or context.
  Real, built, run (and once discarded after 18 contaminated rows from an
  operator debugging collision, now fixed via the maintenance lock -- see
  this plan's Oct 7 risk note); `x2_k2` is next in evo-x2's queue after
  `x2_r2` finishes.
- **MX2** is a phase (`--phases mx2`) inside `harness/t2s_night2.py`, queued
  as `x2_mx2` right after `x2_k2` on evo-x2. Not yet run as of 2026-10-01;
  what MX2 itself measures was not re-derived from the phase's own code in
  this pass -- read `t2s_night2.py`'s MX2 phase function docstring directly
  before relying on this plan's description of it elsewhere.
- **"kappa study"**: confirmed still correct that no file, script, or doc in
  this repo uses "kappa," "Cohen," or "inter-rater" as of 2026-10-01. Neither
  the 150-item blinded hand-labeling pass nor `x2_llm_judge` (queued,
  `blocked_not_built`) have been built yet. This is a real gap, not a
  stale-checkout artifact -- it must be scoped and built before the Oct 7
  milestone it's listed under.

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
| **A-19** Multi-turn recall vs. artifact distance / schema / size under partial KV eviction (Fig 4.15) -- **UNSUPPORTED, harness not written** | None -- `docs/PAPER_OUTLINE.md`: "harness not yet written; data not yet collected" | Nothing yet. This is a distinct claim from the "R2" milestone (R2 is `harness/t2s_r2_session_growth.py`, real and running -- see naming note above; A-19 specifically is KV-eviction-under-partial-context, not yet built). The demo cannot claim a multi-turn eviction-aware routing feature until this exists; until then the demo's routing policy must restrict itself to single-turn/short-horizon claims it can actually back with A-02 through A-14 |
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
| K1 v3 both machines | `harness/t2s_k1_ollama.py`. evo-x2 has a real completed tier sweep (`results/t2s_k1_ollama_evo-x2_20260930T205515Z.jsonl`, committed); evo-t2s's first attempt measured zero models (root-caused and fixed 2026-10-01: exe resolution and an Ollama-readiness race), retry queued first in evo-t2s's queue as of today |
| R2 first run | `harness/t2s_r2_session_growth.py` (real, built; see naming note above). Positive control confirmed firing correctly 2026-10-01 after two bug fixes. `x2_r2` running on evo-x2; `t2s_r2` queued on evo-t2s. On track |
| Traces final | `analysis/agent_traces.py` / `results/traces/agent_step_lengths.parquet` -- real, committed, includes the censoring-artifact correction and the uncensored OpenHands-scaffolded sources. Largely done; truncation-cliff numbers confirmed for evo-x2 (per-model), still blocked on evo-t2s pending its K1 v3 retry |
| Kappa study | Not built. Confirmed real gap (not a stale-checkout artifact): a new script (e.g. `analysis/kappa_agreement.py`) comparing the probe scorers' classifications against a human-labeled sample; output `results/kappa_agreement.json`. **Highest-risk Oct 7 item** -- nothing exists yet and it has not been scoped |
| Demo spec | `docs/DEMO_SPEC.md` (this task produces it) |
| Cloud API with cap | `src/cloud/client.py` (this task produces it), `results/cloud_ledger.jsonl` |
| Workload pack | A set of example workload descriptions for the demo's input form and the Pareto-frontier demonstration; does not exist yet -- needs a new `demo/workload_pack/` or similar, scoped by whoever owns the demo build |
| All-cloud pass | `routing/policies/all_cloud.py` (exists) exercised end-to-end against the stub/mock cloud provider in `src/cloud/client.py`, producing a `results/pareto_results.json`-shaped output with the `all_cloud` policy populated (stub values, clearly labeled, since no real key is used per this task's constraints) |

### Oct 14 2026

Paper 1 core experiments complete: R2, K2, MX2, controls, failure-map gaps.

| Deliverable | Concrete files expected |
|---|---|
| R2 complete | Full first-run scope (llama3.1:8b, arms a-d, as_is and occupied_40gb, 3 seeds on evo-x2; as_is only on evo-t2s) finished on both machines, written up in docs/FINDINGS.md |
| K2 | `harness/t2s_k2_pressure.py`, including the pause-and-resume arm (arm d). Queued fresh on evo-x2 (`x2_k2`, right after `x2_r2`) after an earlier run was discarded (18 contaminated rows from an operator-debugging collision, now prevented by the maintenance lock) |
| MX2 | `t2s_night2.py --phases mx2`, queued (`x2_mx2`) right after `x2_k2`. Not yet run as of 2026-10-01; confirm its exact measurement scope from the phase's own docstring before this milestone |
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

(Corrected 2026-10-01 -- an earlier pass of this section, written from a
stale checkout, wrongly treated R2/K2/MX2 as unscoped/not-started. See the
naming note at the top of this document for what each one really is.)

- **R2 first run**: in progress, not at risk -- `x2_r2` is running on
  evo-x2, `t2s_r2` is queued on evo-t2s, and the positive control is
  confirmed working.
- **Kappa study**: genuinely not started, confirmed real (not a
  stale-checkout artifact). Needs to be scoped (this plan proposes: Cohen's
  kappa between the automated scorer in `evaluation/probes/scorers.py` and a
  human-labeled sample) and then built. **This is the single highest-risk
  Oct 7 item.**
- **Workload pack**: no `demo/` directory or workload-description set exists
  in this repo today; this is B3, not yet started.
- **K2 / MX2**: both real and queued (`harness/t2s_k2_pressure.py`,
  `t2s_night2.py --phases mx2`), not at risk for Oct 7 since they are Oct 14
  deliverables; see the Oct 14 table above for their real status.

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
