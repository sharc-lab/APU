# Methodology

## Recorded model endpoints.

Model endpoint record/replay is implemented as a backend-level concern so every model invocation goes through the same path.

- Replay key: SHA256 over canonical JSON of `model`, `messages`, `tools`, `temperature`, and `seed`.
- Trace location: `analysis/traces/{model}/{key}.json`.
- `RECORD` mode: always call provider API, then persist trace entry.
- `REPLAY` mode: never call provider API; raise if trace key is missing.
- `AUTO` mode: replay when entry exists; otherwise call API and record.

Each trace entry stores:

- `response`: provider response JSON payload used for downstream parsing.
- `token_counts`: `{prompt_tokens, completion_tokens, total_tokens}` from response usage fields.
- `recorded_latency_ms`: original API latency captured at record time.

Runtime metadata exposes two latency values for each call:

- `recorded_latency_ms`: original provider call latency from the trace.
- `replay_latency_ms`: local cache read latency for the current run.

This separation allows replayed benchmark runs to preserve the original remote latency signal while also quantifying replay overhead.

### Environment controls

- `APU_REPLAY_MODE`: one of `AUTO` (default), `RECORD`, `REPLAY`.
- `APU_TRACES_ROOT`: optional override for trace root directory.

### Backend notes

- `CloudOpenAIBackend` stores traces at `analysis/traces/{model}/{hash}.json`.
- `LocalOllamaBackend` stores traces at `analysis/traces_ollama/{model}/{hash}.json`.
- Both backends emit the same per-turn category keys as the instrumentation enum.

## Router Distillation Flywheel

Each benchmark sweep contributes structured routing decisions and replay-backed model traces.

- Decision logs provide step context, selected backend, budget state, and outcome hints.
- Replay traces provide deterministic output and token metadata at zero additional API cost.
- `analysis/distill_router.py` converts this growing corpus into supervised examples:
	- features: step signals + budget/routing context + trace-derived metadata
	- label: whether local was adequate
- The script trains lightweight classifiers (logistic regression + gradient boosted trees),
	selects the best model, and exports an artifact consumed by `routing/policies/learned_router.py`.

This creates a closed-loop improvement cycle: every run expands training data, and better learned policies can be re-evaluated in replay mode without new model spend.
