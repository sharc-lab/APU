# APU Research — Hybrid Local/Cloud Routing Study

A budget-constrained routing study for hybrid cloud/local agent execution, using orchestration-overhead characterization (SDK vs. LangGraph, tail latency) as supporting instrumentation.

## Thesis

**Primary thesis:** budget-aware, step-level routing is the central contribution in this repository.

The project focuses on how to route agent steps between local and cloud backends under explicit cloud-token budgets while preserving quality and latency targets. The SDK-vs-LangGraph overhead and tail-latency characterization are used as supporting instrumentation: they ground our measurement assumptions, attribution categories, and validity checks for the routing experiments.

## Quick Start

```bash
cd SHARC/APU
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv sync --all-extras

# Run APU characterization
OPENAI_API_KEY=sk-... python -m harness.adapters.sdk_direct

# Run tail-latency study
python -m harness.tail_latency_instrument

# Generate reports
python -m analysis.generate_reports
```

## Directory Structure

```
APU/
├── FINAL_STATUS.md
├── SETUP_SUMMARY.md
├── VERIFICATION.md
├── pyproject.toml
├── README.md
├── .env                          # API keys (not committed)
├── harness/
│   ├── __init__.py
│   ├── instrumentation/
│   │   ├── __init__.py
│   │   ├── categories.py         # Category enum and definitions
│   │   ├── span.py               # Span dataclass and CategoryMetrics
│   │   └── timing.py             # _wall_ns(), _process_cpu_ns() primitives
│   ├── adapters/
│   │   ├── __init__.py
│   │   └── sdk_direct.py          # Supporting orchestration instrumentation adapter
│   └── tail_latency_instrument.py # Supporting tail-latency instrumentation
├── routing/
│   ├── budget.py                  # Budget state tracking + decision logs
│   └── policies/                  # Routing policies (static, cascade, speculative, learned)
├── evaluation/
│   ├── quality.py                 # Task scoring (deterministic + judge)
│   ├── sweep.py                   # Policy-budget sweep runner
│   └── certify.py                 # Sampled quality certification
├── analysis/
│   ├── __init__.py
│   ├── generate_reports.py        # Report generation
│   ├── pareto.py                  # Quality-vs-cost frontier analysis
│   └── distill_router.py          # Router distillation from decision traces
├── docs/
│   ├── DECISIONS.md              # Architectural Decision Records (ADR)
│   ├── METHODOLOGY.md            # Routing and replay methodology
│   └── SCHEMA.md                 # JSON output schema documentation
├── results/
│   ├── .gitkeep
│   ├── claude_code_characterization.json
│   ├── tail_latency_results.json
│   └── zachary/
│       └── replication_remote_search_v3.json
├── reports/
│   ├── project_summary.md
│   └── results_summary.md
└── tests/
    ├── __init__.py
    ├── test_instrumentation.py   # Unit tests for timing primitives
    ├── test_span.py              # Span merge/aggregation tests
    └── test_adapters.py          # Adapter integration tests
```

## Routing Study Outputs

- Budget-constrained policy sweep outputs: `results/pareto_results.json`
- Sampled certification outputs: `results/certified_quality.json`
- Learned router distillation/eval outputs: `results/learned_router_eval.json`

## Supporting Instrumentation Results

| Framework | batch_host_cpu_ms | ORCH % | FRAMEWORK % | TOOL_COMPUTE % |
|---|---|---|---|---|
| **OpenAI SDK** | 2,765 | 0.1 | 0.0 | 94.1 |
| **LangGraph** | 2,067 | 29.5 | 8.9 | 20.5 |

**Methodological role:** These measurements calibrate orchestration categories and latency baselines that inform routing-policy evaluation; they are supporting instrumentation, not the headline contribution.

## Instrumentation Categories

- **ORCH_SETUP** — Prompt assembly, message building
- **ORCH_DISPATCH** — Tool routing, result wiring
- **TOKENIZATION** — Input encoding to tokens
- **SERIALIZATION** — Packing tool results into messages
- **CLIENT_PARSE** — Parsing API JSON response
- **FRAMEWORK** — SDK glue code, object construction
- **TOOL_COMPUTE** — Executing tool implementations
- **HTTP_CLIENT** — Network wait (I/O-bound, cpu=0)
- **RESIDUAL_UNATTRIBUTED** — Unattributed CPU (GC, Python overhead)

## Tail Latency (p50 / p99)

Measured across 14 task types, 3 concurrency conditions (single, chained, fan-out), 30 runs each = 1,260 probes.

| Task | single p50 | single p99 |
|---|---|---|
| CH-01 (code+hybrid) | 3,105 ms | 8,152 ms |
| FO-01 (file/output) | 5,223 ms | 7,712 ms |
| SO-01 (search-only) | 1,701 ms | 8,010 ms |

## References

- [Zachary Johnson's LangGraph baseline](github.com/zjohnson2005/apu-characterization)
- [SHARC DECISIONS.md](docs/DECISIONS.md) — Architecture decision records

## License

Internal research project.
