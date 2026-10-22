# Cloud models and pricing (src/cloud/client.py)

## Shared price files (2026-10-14)

`src/cloud/client.py` has no price table of its own. At import it builds
`MODEL_PRICING` and the `CLOUD_MODELS` tier map from the shared price files
`results/cloud_prices_*.json`, the latest file per provider, through
`src/cloud/prices.py`. The same loader prices the R2 sessions in
`analysis/r2_session_cost.py` (register row `r2-session-cloud-cost`), so the
cap, the ledger and the analysis all use one set of list prices.

- File naming: `results/cloud_prices_<YYYYMMDD>.json` (provider from the
  file's `provider` field or its source URL; `cloud_prices_20261008.json` is
  the OpenAI table) or `results/cloud_prices_<provider>_<YYYYMMDD>.json`
  (for example `cloud_prices_anthropic_<date>.json`). The latest date wins per
  provider, so a re-fetched price file replaces the old one with no code
  change. Each file carries its vendor source URL and fetch time, and every
  `MODEL_PRICING` row names its `price_file`.
- Rates: Standard tier, short context (long-context rates above the file's
  boundary when it has them). `estimate_cost_usd()` charges every input token
  at the uncached input rate (the upper bound). Since 2026-10-22 the ledger's
  `cost_usd` and `running_total_usd`, the USD 50 cap and the spend alerts use
  the BILLED cost from the call's usage fields (`billed_cost_usd()`: cache
  reads at the hit rate, cache writes at the 5-minute rate unless the call used
  the 1-hour TTL); `cost_usd_upper_bound` and `running_total_usd_upper_bound`
  keep the upper bound on every new row. The pre-call cap check still reserves
  the upper bound (cache hits are unknown before a call; a request with
  `cache_control` reserves the all-cache-write price if that is higher). Rows
  written before 2026-10-22 carry only the upper bound and count at it.
- Tier map: `CLOUD_MODELS = {"cheap", "strong"}` is the preferred provider's
  `selection` block. Anthropic is preferred once its price file exists; until
  then the OpenAI file's selection applies: cheap `gpt-6-luna`, strong
  `gpt-6.1-sol`. Before 2026-10-14 the client carried a hard-coded table
  (cheap `gpt-4o-mini`, strong `claude-sonnet-4-5`, fetched 2026-09-30);
  `gpt-4o-mini` is still priced from the OpenAI file's other rows (same
  0.15 / 0.60 USD per 1M), `claude-sonnet-4-5` is not in any shared file and
  is refused until a price file lists it. The router's cloud model and the
  stub costs in `results/router_sample_decisions.txt` changed with the tier map.
- A model id in no price file raises `UnknownModelError` rather than being
  estimated at $0. With no price file at all both dicts are empty and every
  real call is refused (fail closed).
- No real call has been made against either endpoint from this repo. All
  testing of `src/cloud/client.py` uses the stub provider or an injected fake
  `completion_fn` (`tests/test_cloud_client.py`, `tests/test_cloud_prices.py`).

## Spend cap and alerts (2026-10-07)

- Hard cap: USD 50 total (`HARD_SPEND_CAP_USD`), summed over every call row in
  `results/cloud_ledger.jsonl`, whenever a real key is set. A real-mode client
  may lower it but not raise it (`ValueError`). A request whose projected cost
  would take the total past the cap is refused before anything is sent.
- Alerts at 50%, 75% and 90% of the cap, each fired exactly once per ledger: an
  `{"record": "alert", "source": "cloud_client"}` row in the ledger (surfaced by
  `analysis/results_digest.py`) plus a printed `ALERT:` line. Tests:
  `tests/test_cloud_spend_alerts.py` (fake provider only).
- Unchanged by the 2026-10-14 price-file switch:
  `tests/test_cloud_prices.py::test_hard_cap_unchanged_and_refuses_a_call_that_would_exceed_it`
  spends USD 49 on a shared-price model, then shows a call projected past USD 50
  is refused before anything is sent (no provider call, no ledger row) while a
  call that fits is still allowed.

## Every OpenAI call goes through the capped client (2026-10-08)

Operator decision 2026-10-08: no code path may call OpenAI except through
`src/cloud/client.py`. `tests/test_no_direct_openai_clients.py` scans every
tracked `.py` file and fails on `import openai`, `from openai`, `OpenAI(`,
`AsyncOpenAI(`, a dynamic import of `openai`, `api.openai.com`, or an
`os.environ` / `os.getenv` read of `OPENAI_API_KEY`, anywhere except
`src/cloud/client.py` (and the test itself). Since 2026-10-14 it flags the
same forms for Anthropic (`import anthropic`, `Anthropic(` /
`AsyncAnthropic(`, `api.anthropic.com`, an env read of `ANTHROPIC_API_KEY`).

### Key and mode rule

`CloudClient(mode=...)`; environment variables are read only in real mode.

| mode | Key used | No key |
|---|---|---|
| `"auto"` (default) | the `api_key` argument only; env vars are never read | stub |
| `"stub"` | none, even if `api_key` is passed | stub |
| `"real"` / `CloudClient.real(...)` | `api_key` argument, else `ANTHROPIC_API_KEY`, else `OPENAI_API_KEY`, else `CLOUD_API_KEY` | `MissingApiKeyError` (never a silent stub) |

- 2026-10-14: `ANTHROPIC_API_KEY` is the real-mode provider key and is checked
  first. The client stays a stub until a caller opts in to real mode
  (`CloudClient.real(...)` or `mode="real"`) with a key available; exporting
  `ANTHROPIC_API_KEY` alone flips nothing.
- So `CloudClient(api_key=None)` (how `analysis/pareto.py`, `src/dse/*` and the
  tests build a stub) stays a stub with `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`
  or `CLOUD_API_KEY` exported: exporting a key can never make a sweep spend
  money, and a stub path never reads a key variable
  (`tests/test_cloud_prices.py::test_stub_path_never_reads_key_env`). Behaviour
  change: before 2026-10-08 an exported `CLOUD_API_KEY` did flip such a client
  to real mode; nothing in the repo relied on that.
- The OpenAI path is unchanged: with no `ANTHROPIC_API_KEY`, real mode falls
  back to `OPENAI_API_KEY`. A key taken from `OPENAI_API_KEY` is only used for
  OpenAI models and one from `ANTHROPIC_API_KEY` only for Anthropic models
  (`MODEL_PRICING[...]["provider"]`); a real call to another provider's model
  with it raises `ValueError` before anything is sent. Keys are never printed
  (`redact_key`).
- OpenAI models are sent with the official `openai` SDK (one SDK client per
  `CloudClient`); Anthropic and any other provider go through `litellm`
  (listed in `pyproject.toml` but not installed on the controller as of
  2026-10-14, so it must be installed before an Anthropic real call). No
  direct Anthropic SDK client is built anywhere.
- Extra keywords (`tools`, `tool_choice`, `max_tokens`, `seed`, ...) reach the
  provider unchanged; `tool_choice` is also written to the ledger row. The
  cap projection uses `max_tokens` as the expected output when the caller
  gives no `expected_output_tokens` (else 256), plus the tools JSON as input.

### prepare / send / settle (timed callers)

`call()` is `prepare()` (host check, pricing, cap check, reservation, SDK
client creation) + `send()` (the provider request only) + `settle()` (ledger
row, running total, alerts). In-flight reservations count against the cap
under a lock, so concurrent calls cannot jointly overshoot it. Both harness
backend bases (`harness/adapters/base.py`, `harness/backends/base.py`) call
`prepare()` before their timer starts and `settle()` after it stops, so the
cap and ledger add no work to `recorded_latency_ms`.

### Call sites found and what happened to each

| Call site | Before | Now |
|---|---|---|
| `harness/adapters/sdk_direct.py` (`--backend openai`) | `OpenAI(api_key=os.environ["OPENAI_API_KEY"])` | `OpenAIChatBackend(cloud=CloudClient.real())`; HTTP_CLIENT/CLIENT_HTTP spans unchanged (send only). The cap/ledger work is outside every span, so it lands in the session RESIDUAL. |
| `harness/adapters/sdk_direct.py` (`--backend ollama`) | `OpenAI(base_url=ollama)` | `local_openai_compatible_client(base_url)` from the client module: local, no spend, refuses any `openai.com` host. |
| `harness/tail_latency_instrument.py` | `OpenAI(api_key=os.environ["OPENAI_API_KEY"])` | `OpenAIChatBackend(cloud=CloudClient.real())`. `mcp_roundtrip_ms` is unchanged (send only). `turn_total_ms` now also includes one cap check and one ledger line append per call (under a lock shared by the three `fan_out` threads); rows recorded before 2026-10-08 do not. |
| `harness/backends/cloud_openai.py` (used by `evaluation/quality.py` judge and `evaluation/sweep.py`) | `OpenAI(api_key=api_key or OPENAI_API_KEY)` | `CloudClient.real(api_key=...)`, or an injected `cloud_client`. `recorded_latency_ms` still covers the SDK request plus `model_dump`, as before. Construction without a key raises `MissingApiKeyError` (the bare SDK constructor also raised). |

No other path was found: the remaining `/v1/chat/completions` posts in the
repo go to local llama-server / Ollama, and no file posts to `api.openai.com`.
Every call site could be routed without changing what its latency fields
measure, so none was left refusing to run.

### Cloud R2

There is no cloud runner for the R2 agent sessions: `harness/x2_r2_agent.py`
talks only to local Ollama. The operator's call-2 rule (identical tools plus
`tool_choice` "none", call-2 mode recorded per row) is supported by the client
(`tools` and `tool_choice` pass through unchanged; `tool_choice` is ledgered),
tested in `tests/test_cloud_capped_routing.py`, but no cloud R2 runner was
built.
