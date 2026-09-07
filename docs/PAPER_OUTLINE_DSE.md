# Paper 2 Outline — DSE + Hybrid Engine

This file maps Paper 2 sections to repository artifacts.
Paper 2 consumes the joint feasibility envelope produced by Paper 1
(see [docs/PAPER_OUTLINE.md](docs/PAPER_OUTLINE.md), Section 6) and
searches the hardware × routing-policy space to produce a Pareto surface.

The hardware BOM is a first-class axis alongside quality and cloud spend.
Routing code in `routing/policies/*` implements one dimension of the search
space; it is not the primary contribution.

---

## Section 1 — Introduction

No figures. States the DSE framing, the three-axis Pareto surface
(quality × cloud spend × BOM cost), and the dependence on Paper 1's
feasibility envelope.

---

## Section 2 — Related Work

Sources: [docs/RELATED_WORK.md](docs/RELATED_WORK.md), [docs/references.bib](docs/references.bib).

Nearest prior work: RouteLLM / Martian / NotDiamond (query-level routers),
static cloud-edge dispatch policies, speculative decoding (token level).

---

## Section 3 — System Design

Sources: [docs/DECISIONS.md](docs/DECISIONS.md), [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

### 3.1 — Routing policies

| Policy | Implementation |
|---|---|
| all_cloud | baseline — no local inference |
| all_local | baseline — no cloud inference |
| static_category | category-level threshold |
| cascade | quality-conditional escalation |
| budget_aware | remaining-budget as policy state |
| speculative | local + cloud in parallel, agreement gate |
| learned | distilled from replay traces |

Source files: [routing/policies/](routing/policies),
[routing/budget.py](routing/budget.py).

### 3.2 — BOM axis

Hardware configurations as first-class cost axis.
Target configs: 16 GB no-NPU, 32 GB + NPU, EVO-X2 Strix Halo.
Source: [configs/hardware/](configs/hardware).

### 3.3 — Hybrid engine

Replay cache for trace replay: [harness/replay/cache.py](harness/replay/cache.py).
Sweep runner: [evaluation/sweep.py](evaluation/sweep.py).
Sampled quality certification: [evaluation/certify.py](evaluation/certify.py).

---

## Section 4 — DSE Search

### Table 4.1 — Search space

**Shows:** hardware configs × routing policies × budget levels × task suite.
Derived from sweep configuration; no separate data file.

---

### Figure 4.1 — Pareto frontier (quality vs. cloud spend)

**Shows:** Pareto-optimal (policy, hardware) pairs in quality × cloud API spend
space; dominated configurations grayed; selected operating points labeled.

- Script: [analysis/pareto.py](analysis/pareto.py)
- Data: `results/pareto_results.json` — **ABSENT from repo and disk**
- Status: **blocked** — requires `evaluation/sweep.py` run with cloud API key

---

### Figure 4.2 — BOM axis expansion (quality vs. total cost of ownership)

**Shows:** Pareto surface extended to three axes: quality, cloud spend, and
hardware BOM cost; each surface point is a (hardware config, routing policy) pair.

- Script: [analysis/bom_sweep.py](analysis/bom_sweep.py)
- Data: `results/pareto_results.json` — **ABSENT from repo and disk**
- Status: **blocked** — same dependency as Fig. 4.1

---

### Figure 4.3 — Per-task routing breakdown

**Shows:** per-task-type breakdown of cloud vs. local decisions at selected
operating points; identifies which categories dominate cloud spend.

- Script: none yet — would read from `results/pareto_results.json`
- Data: `results/pareto_results.json` — **ABSENT from repo and disk**
- Status: **blocked** — same dependency as Fig. 4.1;
  `reports/pareto_task_breakdown.md` is also absent

---

## Section 5 — Quality Certification

### Figure 5.1 — Certified quality per (policy, budget)

**Shows:** Wilson-interval quality bounds per (routing policy, budget level)
pair; demonstrates that sampling-based certification achieves deployable
confidence bounds at lower audit cost than full evaluation.

- Script: [evaluation/certify.py](evaluation/certify.py)
- Data: `results/certified_quality.json` — **ABSENT from repo and disk**
- Status: **blocked** — requires sweep completion first

---

## Section 6 — Learned Router Flywheel

### Figure 6.1 — Distilled router evaluation

**Shows:** quality × cost for the learned router (distilled from sweep decision
traces) vs. static policies; demonstrates that replay traces produce a viable
router without incremental API spend.

- Script: [analysis/distill_router.py](analysis/distill_router.py),
  [routing/policies/learned_router.py](routing/policies/learned_router.py)
- Data: `results/learned_router_eval.json` — **ABSENT from repo and disk**
- Status: **blocked** — requires sweep + distillation runs

---

## Section 7 — Threats to Validity

All Pareto-surface claims depend on `results/pareto_results.json`, which does
not exist. Until the sweep runs against a live cloud API, the routing-policy
comparison, BOM analysis, learned-router flywheel, and quality certification
are design-level claims without experimental backing.

`routing/policies/*` code is present and tested; the sweep infrastructure
is in [evaluation/sweep.py](evaluation/sweep.py). The blocking requirement
is a cloud API key and budget for the sweep run.

Sources: [docs/THREATS.md](docs/THREATS.md), [docs/DECISIONS.md](docs/DECISIONS.md).
