"""DSE (design-space exploration) baselines and router for the local/cloud routing demo.

Everything here runs against src/cloud/client.py's STUB provider only (no real cloud API key, no live local
model call) -- see that module's docstring for the stub contract. This is deliberate scope for task B6: a real
run needs machine time and a real or sandboxed cloud key, both out of scope here (see
src/dse/live_validation_stub.py for what a real run would require).

Modules:
  baselines.py  -- all-local, all-cloud (cheap/strong), and a RouteLLM BERT-router baseline (the only RouteLLM
                   router usable without an OpenAI key; see that module's docstring for why mf/sw_ranking are
                   skipped).
  router.py     -- our own router: envelope feasibility/latency/effective-context from
                   analysis/envelope_model.py, a simple difficulty estimate, live remaining-budget state from
                   CloudClient, and a hard rule against ever routing into a silent-failure configuration.
  live_validation_stub.py -- a runnable script demonstrating what each baseline/router decision looks like
                   against the stub provider, and stating plainly what a real (live) validation would need.
"""
