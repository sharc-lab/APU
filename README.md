# APU Research — Agent Processing Unit Characterization

A reproducible, budget-constrained benchmark and routing study for hybrid cloud/local agent execution, measuring quality, cloud-token cost, and tail latency across orchestration frameworks.

## Thesis

**Orchestration overhead dominates host-CPU cost in AI agent systems.** Direct SDK calls (OpenAI) consume 0.1% host-CPU for orchestration; LangGraph-based agents consume 29.5%. For latency-sensitive applications, framework choice matters more than tool implementation. This work provides reproducible methodology to measure and compare frameworks on any infrastructure.

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
│   │   └── sdk_direct.py         # OpenAI SDK APU adapter (refactored)
│   ├── tasks.py                  # Task definitions (14 tasks)
│   ├── tools.py                  # Tool implementations
│   └── tail_latency_instrument.py # Latency profiler (refactored)
├── analysis/
│   ├── __init__.py
│   ├── generate_reports.py       # MD + PDF report generation
│   └── comparisons.py            # Cross-framework comparison logic
├── docs/
│   ├── DECISIONS.md              # Architectural Decision Records (ADR)
│   └── SCHEMA.md                 # JSON output schema documentation
├── results/
│   ├── claude_code_characterization.json
│   ├── tail_latency_results.json
│   └── zachary/
│       └── replication_remote_search_v3.json
├── reports/
│   ├── results_summary.md
│   └── results_summary.pdf
└── tests/
    ├── __init__.py
    ├── test_instrumentation.py   # Unit tests for timing primitives
    ├── test_span.py              # Span merge/aggregation tests
    └── test_adapters.py          # Adapter integration tests
```

## Key Results

| Framework | batch_host_cpu_ms | ORCH % | FRAMEWORK % | TOOL_COMPUTE % |
|---|---|---|---|---|
| **OpenAI SDK** | 2,765 | 0.1 | 0.0 | 94.1 |
| **LangGraph** | 2,067 | 29.5 | 8.9 | 20.5 |

**Insight:** LangGraph adds ~38 percentage points of orchestration overhead. Raw SDK scales better per-token spent on framework.

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
