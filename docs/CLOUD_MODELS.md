# Cloud models and pricing (src/cloud/client.py)

Two real cloud models are wired into `src/cloud/client.py`'s `MODEL_PRICING`
table and `CLOUD_MODELS` tier map: one cheap, one strong. Prices below were
fetched 2026-09-30 directly from each vendor's own pricing page (not an
aggregator) via WebFetch. Re-verify before trusting these numbers past that
date -- cloud pricing changes without notice.

| Tier | Model id | Input $/1M tok | Output $/1M tok | Source | Fetched |
|---|---|---|---|---|---|
| cheap | `gpt-4o-mini` | 0.15 | 0.60 | https://developers.openai.com/api/docs/pricing (platform.openai.com/docs/pricing redirects here) | 2026-09-30 |
| strong | `claude-sonnet-4-5` | 3.00 | 15.00 | https://claude.com/pricing (anthropic.com/pricing redirects here) | 2026-09-30 |

Notes:
- The claude.com/pricing page lists Claude Sonnet 4.5 under a "Legacy models"
  section as of the fetch date (Claude Sonnet 5 is the current flagship at
  time of writing). It was still chosen as the "strong" tier here because its
  price and model id are clearly published and stable; if the project wants
  the current flagship instead, re-fetch claude.com/pricing for the active
  Sonnet model's id and per-token price and update `MODEL_PRICING` in
  `src/cloud/client.py` accordingly -- do not guess a price.
- GPT-4o-mini also has a lower batch-API rate ($0.075/$0.30 per 1M) and a
  prompt-caching input rate ($0.075/1M); neither is used here because the
  client makes simple synchronous calls with no caching assumption.
- These two model ids are the only ones `estimate_cost_usd()` /
  `CloudClient.call()` will accept; calling with any other model id raises
  `UnknownModelError` rather than silently estimating $0.
- No real call has been made against either endpoint from this repo. All
  testing of `src/cloud/client.py` uses the stub provider or an injected fake
  `completion_fn` — see `tests/test_cloud_client.py`.

## Spend cap and alerts (2026-10-07)

- Hard cap: USD 50 total (`HARD_SPEND_CAP_USD`), summed over every call row in
  `results/cloud_ledger.jsonl`, whenever a real key is set. A real-mode client
  may lower it but not raise it (`ValueError`). A request whose projected cost
  would take the total past the cap is refused before anything is sent.
- Alerts at 50%, 75% and 90% of the cap, each fired exactly once per ledger: an
  `{"record": "alert", "source": "cloud_client"}` row in the ledger (surfaced by
  `analysis/results_digest.py`) plus a printed `ALERT:` line. Tests:
  `tests/test_cloud_spend_alerts.py` (fake provider only).

## Every OpenAI call goes through the capped client (2026-10-08)

Operator decision 2026-10-08: no code path may call OpenAI except through
`src/cloud/client.py`. `tests/test_no_direct_openai_clients.py` scans every
tracked `.py` file and fails on `import openai`, `from openai`, `OpenAI(`,
`AsyncOpenAI(`, a dynamic import of `openai`, `api.openai.com`, or an
`os.environ` / `os.getenv` read of `OPENAI_API_KEY`, anywhere except
`src/cloud/client.py` (and the test itself).

### Key and mode rule

`CloudClient(mode=...)`; environment variables are read only in real mode.

| mode | Key used | No key |
|---|---|---|
| `"auto"` (default) | the `api_key` argument only; env vars are never read | stub |
| `"stub"` | none, even if `api_key` is passed | stub |
| `"real"` / `CloudClient.real(...)` | `api_key` argument, else `OPENAI_API_KEY`, else `CLOUD_API_KEY` | `MissingApiKeyError` (never a silent stub) |

- So `CloudClient(api_key=None)` (how `analysis/pareto.py`, `src/dse/*` and the
  tests build a stub) stays a stub with `OPENAI_API_KEY` or `CLOUD_API_KEY`
  exported: exporting a key can never make a sweep spend money. Behaviour
  change: before 2026-10-08 an exported `CLOUD_API_KEY` did flip such a client
  to real mode; nothing in the repo relied on that.
- A key taken from `OPENAI_API_KEY` is only used for OpenAI models
  (`MODEL_PRICING[...]["provider"] == "openai"`); a real call to
  `claude-sonnet-4-5` with it raises `ValueError` before anything is sent.
- OpenAI models are sent with the official `openai` SDK (one SDK client per
  `CloudClient`); other providers still go through `litellm` (listed in
  `pyproject.toml` but not installed here).
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
