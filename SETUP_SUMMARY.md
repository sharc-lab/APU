# APU Research Project — Setup Complete

## Summary

Created a new Python research repository (`SHARC/APU`) with:

✅ **pyproject.toml** — Python 3.12, dependencies (openai, httpx, python-dotenv, numpy, matplotlib, pytest)
✅ **README.md** — Project thesis and quick start guide
✅ **docs/DECISIONS.md** — Five ADR entries documenting all Windows/design decisions
✅ **Refactored harness code** — Modular structure with instrumentation split out
✅ **Full test suite** — 10 tests, all passing (✓)

---

## Directory Structure

```
APU/
├── pyproject.toml                      # Python 3.12, uv-compatible
├── README.md                           # Thesis, quick start, results
├── docs/
│   └── DECISIONS.md                    # 5 ADR entries (dated 2026-07-14)
├── harness/
│   ├── __init__.py
│   ├── instrumentation/
│   │   ├── __init__.py
│   │   ├── categories.py               # Category enum + harness definitions
│   │   ├── span.py                     # Span class + merge/accumulate logic
│   │   └── timing.py                   # wall_ns(), process_cpu_ns() primitives
│   ├── adapters/
│   │   ├── __init__.py
│   │   └── sdk_direct.py               # Refactored OpenAI SDK adapter (all tasks, tools, main)
│   └── tail_latency_instrument.py      # p50/p95/p99 profiler (refactored)
├── analysis/
│   └── __init__.py                     # Ready for generate_reports.py
├── results/                            # Output data directory
├── reports/                            # Output reports directory
└── tests/
    ├── __init__.py
    ├── test_instrumentation.py         # Tests timing primitives
    ├── test_span.py                    # Tests Span merge/aggregation
    └── test_adapters.py                # Tests module imports
```

---

## Refactoring Summary

### What Was Split Out

**Instrumentation (`harness/instrumentation/`)**
- `categories.py` — Category enum, CPU_BOUND_CATS, HARNESS_*_CATEGORIES definitions
- `timing.py` — wall_ns(), process_cpu_ns() primitives (moved from adapter)
- `span.py` — Span class + to_dict() / merge() methods

**Adapters (`harness/adapters/`)**
- `sdk_direct.py` — Refactored `claude_code_adapter.py` with:
  - All 14 task definitions (CH-01 through SW-01)
  - Tool implementations (calculator, file_write, search, etc.)
  - Main harness logic (run_seed, _run_session, _check_validity)
  - JSON output schema (matches Zachary's exactly)
  - Entry point: `if __name__ == "__main__": main()`

**Tail Latency**
- `harness/tail_latency_instrument.py` — Refactored with import updates

### Changes Made

1. **Instrumentation imports** — Updated all references:
   - `_wall_ns()` → `wall_ns()`
   - `_process_cpu_ns()` → `process_cpu_ns()`
   - `_CPU_BOUND_CATS` → `CPU_BOUND_CATS`
   - All imports now via: `from harness.instrumentation import ...`

2. **Output paths** — Updated to use Path(__file__).parent calls for correct relative resolution

3. **No functionality changed** — Timing categories, output schema, task definitions, tool implementations all identical

---

## Test Results

All 10 tests pass ✓

```
tests/test_adapters.py::test_adapter_imports PASSED              [ 10%]
tests/test_adapters.py::test_instrumentation_imports PASSED      [ 20%]
tests/test_instrumentation.py::test_wall_ns_monotonic PASSED     [ 30%]
tests/test_instrumentation.py::test_wall_ns_resolution PASSED    [ 40%]
tests/test_instrumentation.py::test_process_cpu_ns_works PASSED  [ 50%]
tests/test_instrumentation.py::test_process_cpu_ns_accumulates PASSED [ 60%]
tests/test_span.py::test_span_record PASSED                      [ 70%]
tests/test_span.py::test_span_merge PASSED                       [ 80%]
tests/test_span.py::test_span_to_dict PASSED                     [ 90%]
tests/test_span.py::test_span_initial_values PASSED              [100%]

============================= 10 passed in 3.17s ==============================
```

---

## Next Steps

```bash
cd SHARC/APU

# Install dependencies
/c/Users/rithw/miniconda3/python.exe -m pip install -e ".[dev]"

# Run adapter (5 seeds × 10 sessions)
OPENAI_API_KEY=sk-... /c/Users/rithw/miniconda3/python.exe -m harness.adapters.sdk_direct

# Run tail-latency study
/c/Users/rithw/miniconda3/python.exe -m harness.tail_latency_instrument

# Generate reports
/c/Users/rithw/miniconda3/python.exe analysis/generate_reports.py

# Run tests
/c/Users/rithw/miniconda3/python.exe -m pytest tests/ -v
```

---

## ADR Topics Documented

1. **Windows Timing Strategy** — Wall-proxy for CPU-bound spans (15.6 ms tick resolution workaround)
2. **Residual Fraction Denominator** — Use instrumented_total for cross-platform validity
3. **Median Residual Gate** — Tolerate single-seed outliers from clock granularity
4. **Mock Tools Over Remote HTTP** — Isolate orchestration overhead from tool latency
5. **Direct SDK vs LangGraph** — Baseline comparison for framework overhead measurement

---

*Project setup complete: 2026-07-14*
