# JSON Output Schema Documentation

This document describes the output format for APU characterization results.

## Root Schema

```json
{
  "experiment": "replication_batch",
  "result_validity": "publishable" | "residual_too_high_median_X.XXX" | "insufficient_seeds_X_of_Y",
  "generated_utc": "ISO 8601 timestamp",
  "setup_ref": { ... },
  "git": { ... },
  "env": { ... },
  "config": { ... },
  "aggregate": { ... },
  "per_seed_artifacts": [ ... ]
}
```

## Result Validity

Determines whether results meet publishability criteria:
- `"publishable"` — n≥5 seeds AND median residual_fraction < 15%
- `"residual_too_high_median_X.XXX"` — Median residual exceeded threshold
- `"insufficient_seeds_X_of_Y"` — Fewer than required seeds completed

## Aggregate Statistics

All aggregate fields use the stats shape:
```json
{
  "median": number,
  "q1": number,
  "q3": number,
  "iqr": number,
  "min": number,
  "max": number,
  "n": number
}
```

Key aggregate metrics:
- `batch_host_cpu_ms` — Total host CPU per seed (ms)
- `pooled_tool_compute_pct` — % of instrumented CPU spent in TOOL_COMPUTE
- `pooled_framework_pct` — % spent in FRAMEWORK
- `pooled_orch_pct` — % spent in ORCH (setup + dispatch)
- `pooled_harness_strict_pct` — % in harness strict categories
- `pooled_residual_unattributed_pct` — % unattributed

## Per-Seed Artifacts

Each seed contains:
```json
{
  "config": {
    "seed": number,
    "sessions": number,
    "mode": "gpt-4o-mini"
  },
  "batch_wall_s": number,
  "run": {
    "per_category": { "CATEGORY_NAME": CategoryMetrics, ... },
    "per_session": [ Session, ... ],
    "per_session_category": { "agent_N": { "CATEGORY": CategoryMetrics }, ... },
    "residual_cpu_ns": number,
    "residual_fraction": number,
    "totals": {
      "thread_cpu_ns": number,
      "wall_ns": number,
      "instrumented_cpu_ns": number
    },
    "provenance_totals": {
      "measured": number,
      "step_inferred": number,
      "residual": number
    },
    "provenance_detail": { "CATEGORY|agent_N|profile|source": cpu_ns, ... }
  }
}
```

## CategoryMetrics Shape

```json
{
  "cpu_ns": number,
  "wall_ns": number,
  "bytes_in": number,
  "bytes_out": number,
  "count": number
}
```

## Session Shape

```json
{
  "session_id": "agent_N",
  "task_id": "CH-01",
  "category": "code_hybrid",
  "plan_len": number,
  "wall_s": number,
  "thread_cpu_ns": number,
  "process_cpu_ns": number,
  "instrumented_cpu_ns": number,
  "orch_measured_cpu_ns": number,
  "residual_unattributed_cpu_ns": number,
  "tool_call_counts": { "tool_name": count, ... },
  "categories": { "CATEGORY": CategoryMetrics, ... },
  "provenance": {
    "measured": number,
    "step_inferred": number,
    "residual": number
  },
  "search_locality": "local",
  "replay": false
}
```

## Instrumentation Categories (13 total)

| Category | Description | CPU Method |
|---|---|---|
| `ORCH_SETUP` | Prompt assembly, message building | wall proxy |
| `ORCH_DISPATCH` | Tool routing, result wiring | wall proxy |
| `TOKENIZATION` | Input encoding to tokens | wall proxy |
| `SERIALIZATION` | Packing tool results into messages | wall proxy |
| `CLIENT_PARSE` | Parsing API JSON response | wall proxy |
| `FRAMEWORK` | SDK glue code, object construction | wall proxy |
| `TOOL_COMPUTE` | Executing tool implementations | wall proxy |
| `HTTP_CLIENT` | Network wait for LLM response | 0 (I/O) |
| `CLIENT_HTTP` | Network wait (duplicate axis) | 0 (I/O) |
| `PROMPT_ASSEMBLY` | (optional) Prompt template rendering | wall proxy |
| `CONTEXT_MGMT` | (optional) Conversation history management | wall proxy |
| `LOGGING` | (optional) Debug/trace logging overhead | wall proxy |
| `RESIDUAL_UNATTRIBUTED` | Unattributed session CPU | process_time delta |

## Harness Definitions

**Harness Strict** = ORCH_SETUP + ORCH_DISPATCH + TOKENIZATION + SERIALIZATION

**Harness Broad** = Strict + HTTP_CLIENT + PROMPT_ASSEMBLY + CONTEXT_MGMT + LOGGING

## Tail Latency Output Schema

```json
{
  "experiment": "tail_latency_characterization",
  "generated_utc": "ISO 8601 timestamp",
  "setup_ref": { ... },
  "git": { ... },
  "env": { ... },
  "config": {
    "model": "gpt-4o-mini",
    "backend": "openai",
    "min_samples_per_condition": 30,
    "conditions": ["single", "chained", "fan_out"],
    "tasks": ["CH-01", "CH-02", ...]
  },
  "results": [
    {
      "condition": "single" | "chained" | "fan_out",
      "task_id": "CH-01",
      "metric": "tool_dispatch_ms" | "mcp_roundtrip_ms" | "turn_total_ms",
      "p50": number,
      "p95": number,
      "p99": number,
      "n": number,
      "samples": [number, ...]
    },
    ...
  ]
}
```

### Latency Conditions

- **single** — One independent API call per probe (c=1)
- **chained** — Three sequential tool-use turns in one conversation (c=3 serial)
- **fan_out** — Three parallel API calls fired simultaneously (c=3 parallel, MCP latency = max)

### Latency Metrics

- `tool_dispatch_ms` — Time spent executing tool implementations locally
- `mcp_roundtrip_ms` — Wall time of API call(s) to LLM provider
- `turn_total_ms` — End-to-end wall time of entire probe

---

*Last updated: 2026-07-14*
