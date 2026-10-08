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
- The client's real-mode key is `CLOUD_API_KEY`. Scripts that read
  `OPENAI_API_KEY` directly (`harness/adapters/sdk_direct.py`,
  `harness/tail_latency_instrument.py`, `harness/backends/cloud_openai.py`) do
  not go through this client, so this cap does not apply to them.
