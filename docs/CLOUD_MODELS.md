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
