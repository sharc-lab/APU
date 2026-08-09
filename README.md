# APU Research — Hybrid Local/Cloud Routing Study

A budget-constrained routing study for hybrid cloud/local agent execution, using orchestration-overhead characterization (SDK vs. LangGraph, tail latency) as supporting instrumentation.

## Thesis

**Primary thesis:** hardware is a decision variable for consumer AI PC provisioning.

The contribution is a design-space exploration (DSE) over hardware configurations × routing policies, producing a Pareto frontier across three axes: agent task quality, cloud API spend, and hardware bill-of-materials (BOM) cost. The question is which combination of local hardware and routing policy delivers acceptable quality at the lowest total cost of ownership for a given workload.

**Paper 1 — Hardware feasibility envelope (characterization).** Measures how sustained inference load, memory pressure, KV-cache eviction, and context growth affect agent task quality on consumer hardware. Produces the constraint model that Paper 2 consumes: at what context depth and memory config does quality fall below a given floor?

**Paper 2 — DSE + hybrid engine.** Searches the hardware × policy space and outputs the Pareto surface. Each point is a (hardware config, routing policy) pair. The hardware BOM is a first-class axis, not a footnote.

**Routing code role.** `routing/policies/*` implements the policy axis of the DSE search space. Routing is not the primary contribution — it is one dimension of the space being searched. The code is retained in full; its framing changes from "the contribution" to "one axis of the DSE."

The SDK-vs-LangGraph overhead and tail-latency characterization are supporting instrumentation: they ground measurement assumptions, attribution categories, and validity checks for the sweep.

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

---

## Quality-Degradation Harness

Measures how local LLM output quality degrades as the context window fills with
filler tokens. Uses the 50-probe eval set in `evaluation/probes/`.

### Quick start (Windows — PowerShell or cmd.exe)

```powershell
# One-time: install Ollama and pull qwen3:4b
bash scripts/setup.sh

# One-time: install deps
py -3.11 -m pip install jsonschema psutil httpx numpy matplotlib pytest

# One-time: validate the eval set
cd evaluation/probes && py -3.11 validate.py && cd ../..

# Run a sweep (44 probes × 6 depths × 5 reps)
py -3.11 -m harness.runner

# Plot results
py -3.11 analysis/plot_degradation.py
```

### Runner options

```
--model     Ollama model (default: qwen3:4b)
--host      Ollama URL   (default: http://localhost:11434)
--reps      N per cell   (default: 5)
--depths    token depths (default: 0 2000 8000 16000 32000 64000)
--probe-ids run subset   (e.g. --probe-ids rea_01 cod_01)
```

Results land in `results/run_<timestamp>.jsonl`, one row per call with:
`probe_id, category, difficulty, depth, rep, score, score_detail,`
`latency_ms, ttft_ms, tokens_in, tokens_out, mem_rss_mb, gpu_mem_mb, config_hash`

Cache in `.cache/calls/` (SHA-256 keyed). Delete to force fresh inference.

---

## References

- [Zachary Johnson's LangGraph baseline](github.com/zjohnson2005/apu-characterization)
- [SHARC DECISIONS.md](docs/DECISIONS.md) — Architecture decision records

## License

Internal research project.
