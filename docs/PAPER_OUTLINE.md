# Paper Outline Skeleton

This file maps future paper sections to repository artifacts.

## Section 2: Related Work

- Source section mapping: Section 2 <- [docs/RELATED_WORK.md](docs/RELATED_WORK.md).
- Repo artifacts that feed this section:
- [docs/RELATED_WORK.md](docs/RELATED_WORK.md)
- [docs/references.bib](docs/references.bib)

## Section 3: Methodology

- Source section mapping: Section 3 <- [docs/METHODOLOGY.md](docs/METHODOLOGY.md).
- Repo artifacts that feed this section:
- [docs/METHODOLOGY.md](docs/METHODOLOGY.md)
- [harness/replay/cache.py](harness/replay/cache.py)
- [routing/policies/](routing/policies)
- [evaluation/quality.py](evaluation/quality.py)
- [evaluation/certify.py](evaluation/certify.py)

## Section 4: System/Design Decisions and Architecture

- Source section mapping: Section 4 <- [docs/DECISIONS.md](docs/DECISIONS.md) + repository architecture.
- Repo artifacts that feed this section:
- [docs/DECISIONS.md](docs/DECISIONS.md)
- [harness/backends/](harness/backends)
- [routing/budget.py](routing/budget.py)
- [routing/policies/](routing/policies)
- [harness/instrumentation/](harness/instrumentation)

## Section 5: Experiments and Results

- Source section mapping: Section 5 <- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) + [results/](results).
- Repo artifacts that feed this section:
- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)
- [results/pareto_results.json](results/pareto_results.json)
- [results/certified_quality.json](results/certified_quality.json)
- [results/learned_router_eval.json](results/learned_router_eval.json)
- [reports/pareto_frontier.png](reports/pareto_frontier.png)
- [reports/pareto_task_breakdown.md](reports/pareto_task_breakdown.md)

## Section 6: Threats to Validity

- Source section mapping: Section 6 <- [docs/THREATS.md](docs/THREATS.md).
- Repo artifacts that feed this section:
- [docs/THREATS.md](docs/THREATS.md)
- [docs/DECISIONS.md](docs/DECISIONS.md)
- [docs/SCHEMA.md](docs/SCHEMA.md)
