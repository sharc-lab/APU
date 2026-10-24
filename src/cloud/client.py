"""Cloud LLM client scaffolding for the DSE demo router.

HOST RESTRICTION -- READ THIS BEFORE CALLING ANYTHING IN THIS MODULE
---------------------------------------------------------------------
This module is only permitted to make REAL cloud API calls on two hosts:
the controller (the primary development machine driving this project) and
evo-x2 (the Strix Halo target machine). It must NEVER make a real call on
evo-t2s, under any condition, including by accident.

The host check is explicit and allow-list based (see `check_host_allowed`
below) -- it is never inferred from the absence of a deny condition, and it
never branches on hostname in a way that could silently resolve to "allowed"
on a host nobody has reasoned about. Hostnames are matched case-insensitively
against `socket.gethostname()` (or an injected hostname, for testing).

Order of checks, every time:
  1. HARD DENY first, unconditionally: any hostname containing "t2s" is
     refused before anything else runs, even before the allow-list is read.
     This is a belt-and-suspenders check -- even if evo-t2s's hostname were
     accidentally added to an allow-list by a future edit, this line still
     raises.
  2. Explicit role override: the environment variable CLOUD_CLIENT_HOST_ROLE
     may be set to "controller" to explicitly declare "this machine is the
     controller" (this repo has no fixed controller hostname to hard-code,
     so the operator declares it). This is an opt-in, explicit signal, not a
     default.
  3. Allow-list: the environment variable CLOUD_ALLOWED_HOSTS (comma
     separated, case-insensitive hostname substrings) is checked next,
     falling back to a built-in default of ("evo-x2",) if unset.
  4. Anything else is refused loudly (HostRestrictionError), never silently
     downgraded to stub mode. A disallowed host is a bug to fix, not a
     degraded-but-working state.

This function is only invoked on the REAL-call path. Stub mode (below) never
touches the network and is always safe to run anywhere, including evo-t2s,
specifically so the rest of the pipeline can be developed and tested without
tripping the host restriction.

KEY AND MODE RULE (2026-10-08, Anthropic key added 2026-10-14)
----------------------------------------------------------------
This module is the ONLY place in the repo allowed to talk to OpenAI or
Anthropic (tests/test_no_direct_openai_clients.py enforces it). The
constructor's `mode` argument decides stub vs real, and environment
variables are read ONLY when a caller explicitly asks for real mode:

  mode="auto" (the default):
      real iff the caller passes a non-empty `api_key` argument; otherwise
      STUB. No environment variable is consulted, so CloudClient(api_key=None)
      stays a stub even with ANTHROPIC_API_KEY, OPENAI_API_KEY or
      CLOUD_API_KEY exported in the shell. This is what the Pareto sweep,
      src/dse and the tests use, and it is why setting a key can never make a
      sweep spend money by accident.
  mode="stub":
      always stub, even if an api_key is passed.
  mode="real" (or the CloudClient.real(...) factory, the explicit opt-in):
      key precedence: explicit `api_key` argument, then ANTHROPIC_API_KEY
      (the real-mode provider key, operator 2026-10-14), then OPENAI_API_KEY
      (the 2026-10-08 OpenAI path, unchanged), then CLOUD_API_KEY (legacy
      fallback). If none is set (or all are empty), MissingApiKeyError is
      raised at construction: real mode never silently degrades to a stub.
      A key taken from ANTHROPIC_API_KEY is only sent to Anthropic models and
      one from OPENAI_API_KEY only to OpenAI models (MODEL_PRICING
      "provider"); asking it for another provider's model raises ValueError
      before anything is sent. Keys are never read on a stub path and never
      printed (redact_key below).

In stub mode the deterministic STUB PROVIDER answers instead, so the router,
Pareto sweep, and demo can all be exercised end to end without a real key.
Every row the stub produces carries "stub": true and a model id prefixed
"stub-" so a stub result can never be mistaken for a real one downstream.

TIMED CALLERS (prepare / send / settle)
----------------------------------------
call() is prepare() + send() + settle(). A caller that measures latency
(harness/adapters/base.py, harness/backends/base.py) calls the three
separately: prepare() does the host check, pricing lookup, cap check and
reservation and builds the provider SDK client; send() is only the provider
request; settle() writes the ledger row and fires alerts. Only send() goes
inside the timed region, so the cap and ledger add no work to a measured
round trip. Reservations are held under a lock, so concurrent calls (the
fan_out probe) cannot jointly overshoot the cap.

The key is never logged or printed in full. `redact_key()` below shows at
most the first 2 and last 2 characters of any key-shaped string.

SPEND CAP
---------
HARD_SPEND_CAP_USD = 50.00 USD TOTAL (2026-10-07): the hard cap whenever a real
key is set, summed over the whole ledger (not per month). DEFAULT_SPEND_CAP_USD
equals it. A caller may pass a LOWER spend_cap_usd; a real-mode client asking
for a higher one raises ValueError at construction (stub mode spends nothing,
so any cap is accepted there, e.g. for the Pareto sweep).
BILLED COST VS UPPER BOUND (operator 2026-10-22)
The cap, the running total and the spend alerts are on the BILLED cost: what the provider charges for the call,
computed after the call from its usage fields (billed_cost_usd below): uncached input tokens at the input rate,
cache-read tokens at the cache-hit rate, cache-write tokens at the 5-minute cache-write rate (the 1-hour rate for
tokens the usage reports as 1-hour writes, or for all writes when the request asked for ttl "1h" and the usage
gives no split), output tokens at the output rate, all from the shared price files. The UPPER BOUND (every input
token at the uncached input rate, estimate_cost_usd below; what the ledger recorded before 2026-10-22) is kept as
a second column on every new row.
Pre-call reservation: before a call nobody can know its cache hits, so the cap check before every REAL (non-stub)
call still uses the UPPER BOUND projection of that call (the model's list price and the call's estimated token
counts, every input token uncached; reservation_cost_usd below). One refinement: a cache write is billed ABOVE the
uncached input rate (5m writes 1.25x, 1h writes 2x), so the "every input token uncached" figure is not a strict
bound on a call that writes the cache; a request carrying cache_control therefore reserves the larger of that
figure and every input token billed as a cache write. The cost_usd_upper_bound ledger column keeps the plain
"every input token uncached" definition (the pre-2026-10-22 cost_usd), so on a cache-writing call it can be below
cost_usd. If `billed running_total + in-flight reservations + projected upper bound`
would exceed the cap, the call is refused before any request is sent, and the refusal (with its full reasoning)
is returned via `SpendCapExceeded`, never silently retried or downgraded. On success the reservation is released
and the call settles on its billed cost; on failure the reservation is released and nothing is recorded. The
reservation is conservative: it never lets the billed total pass the cap on account of an unexpected cache miss.
The boundary is inclusive: a call that lands exactly on the cap is allowed; a call that would push even
fractionally past it is refused.
Old ledger rows (written before 2026-10-22) carry only cost_usd, which was the upper bound and no usage
breakdown; they stay as they are.
BILLED CAP BASIS AT START (operator 2026-10-23, register row cloud-spend-billed-vs-upper-bound): the billed running
total a client starts from is the register row's billed total, computed by analysis/cloud_spend_billed.py::compute
on this client's ledger: an old row whose call's usage is on file in a cloud job's result file (the job rows'
cache-priced "usd") counts at that billed cost; an old row with no usage on file anywhere counts at its upper bound;
a new row counts cost_usd. The upper-bound total (cost_usd_upper_bound, or an old row's cost_usd) is kept beside it
and printed next to it in every cap message. Before 2026-10-23 every old row counted at its upper bound toward the
billed total, so on the same ledger the billed total, and so the cap headroom, was near the upper bound. If that
computation fails the client falls back to the old per-row basis (conservative) and says so. A new ledger row's
running_total_usd is on this basis, so it steps down at the first row written after the switch; the row's
running_total_basis field names the basis.
Live jobs also pass the CONSOLE CHECK (src/cloud/console_gate.py) before their first call: the operator's latest
Anthropic Console reading must be fresh and agree with the ledger's billed total for the same range, and the billed
headroom must cover the job's projection. Stub mode is never gated.

SPEND ALERTS
------------
After each real call, when the BILLED running total first reaches 50%, 75% and 90%
of the cap (SPEND_ALERT_FRACTIONS), one alert fires per threshold, exactly
once for the life of the ledger. It is appended to the ledger as a
{"record": "alert", "source": "cloud_client", ...} row (no cost_usd, so the
running total is unchanged), which analysis/results_digest.py's
collect_alerts surfaces at the top of RESULTS_DIGEST.md (it scans every
results/*.jsonl for record == "alert"), and it is printed. A threshold already
present as an alert row in the ledger is never fired again, including by a
new client process on the same ledger. One call that crosses several
thresholds fires each of them.

LEDGER
------
Every REAL (non-stub) call appends exactly one call row (plus any spend
alert rows, above) to results/cloud_ledger.jsonl: timestamp (UTC ISO 8601), model id, input tokens (all input,
cache included) and output tokens, the usage breakdown (uncached_input_tokens, cache_read_tokens,
cache_write_tokens, cache_write_1h_tokens), cost_usd (BILLED), cost_usd_upper_bound (every input token uncached),
running_total_usd (billed, after this call), running_total_usd_upper_bound, and cost_basis: "billed_from_usage"
when the provider returned usage, "upper_bound_no_usage" when it did not (then billed = upper bound). Stub
calls are never written to the ledger -- the ledger is a record of real spend only.

CLIENT LIBRARY
---------------
Real calls to an OpenAI model (provider "openai" in MODEL_PRICING) go
through the official `openai` SDK (a project dependency), one SDK client per
CloudClient, created in prepare() and reused. Real calls to any other model
(Anthropic included) go through the `litellm` package
(https://github.com/BerriAI/litellm), listed in pyproject.toml but not
installed on the controller as of 2026-10-14 (`import litellm` raises
ModuleNotFoundError there), so it must be installed before an Anthropic real
call can run. The import is deferred to inside `_default_completion_fn`
specifically so that stub-mode usage and all of this module's unit tests work
without litellm installed at all.

MODELS AND PRICING (2026-10-14: shared price files, nothing hard-coded)
-----------------------------------------------------------------------
MODEL_PRICING and CLOUD_MODELS are built at import from the shared price
tables results/cloud_prices_*.json, the latest file per provider
(src/cloud/prices.py; the same loader analysis/r2_session_cost.py uses).
Each file carries its vendor source URL and fetch time; every row of MODEL_PRICING
names its price_file. CLOUD_MODELS {"cheap", "strong"} is the preferred
provider's "selection" (Anthropic once results/cloud_prices_anthropic_<date>.json
exists, OpenAI before that: gpt-6-luna / gpt-6.1-sol from
results/cloud_prices_20261008.json). A model in no price file is refused with
UnknownModelError, never estimated at $0. See docs/CLOUD_MODELS.md.

TOOLS AND TOOL_CHOICE
----------------------
Any extra keyword (tools, tool_choice, max_tokens, temperature, seed, ...)
is passed to the provider unchanged. tool_choice is also written to the
ledger row when given, so a forced tool_choice "none" call (the R2 call-2
rule) is auditable from the ledger.

USAGE IN SCRIPTS AND REPORTS
------------------------------
Any script or report built on this module must print "BLOCKED: cloud key"
as the first line of its output whenever it is running in stub mode
(`CloudClient.stub_mode` is True), so a human skimming output never mistakes
a stub run for a real one. `print_stub_banner_if_needed()` below does this.
"""

from __future__ import annotations

import json
import os
import socket
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:
    from . import prices as _prices
except ImportError:  # run as a script (py src/cloud/client.py): this directory is sys.path[0]
    import prices as _prices  # type: ignore[no-redef]

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

ENV_VAR_API_KEY = "CLOUD_API_KEY"          # legacy real-mode key, lowest precedence
ENV_VAR_ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"  # real-mode provider key (operator decision 2026-10-14)
ENV_VAR_OPENAI_API_KEY = "OPENAI_API_KEY"  # real-mode key for OpenAI models (operator decision 2026-10-08)
# Real-mode key lookup order after an explicit api_key argument. Only read when mode="real".
REAL_MODE_KEY_ENV_VARS = (ENV_VAR_ANTHROPIC_API_KEY, ENV_VAR_OPENAI_API_KEY, ENV_VAR_API_KEY)
# A key read from one of these variables is only ever sent to that provider's models.
KEY_ENV_PROVIDER = {ENV_VAR_ANTHROPIC_API_KEY: "anthropic", ENV_VAR_OPENAI_API_KEY: "openai"}
CLIENT_MODES = ("auto", "stub", "real")
ENV_VAR_ALLOWED_HOSTS = "CLOUD_ALLOWED_HOSTS"
ENV_VAR_HOST_ROLE = "CLOUD_CLIENT_HOST_ROLE"

_HARD_DENY_SUBSTRINGS = ("t2s",)  # catches evo-t2s, EVO-T2S, t2s-anything -- never overridable
_DEFAULT_ALLOWED_SUBSTRINGS = ("evo-x2",)

HARD_SPEND_CAP_USD = 50.00  # 2026-10-07: hard cap, USD total, whenever a real key is set
DEFAULT_SPEND_CAP_USD = HARD_SPEND_CAP_USD
SPEND_ALERT_FRACTIONS = (0.50, 0.75, 0.90)
DEFAULT_LEDGER_PATH = Path("results/cloud_ledger.jsonl")

def load_model_pricing(repo: str | Path | None = None) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """(MODEL_PRICING, CLOUD_MODELS) from the shared price files (src/cloud/prices.py: results/cloud_prices_*.json,
    the latest file per provider). No price is hard-coded in this module.

    MODEL_PRICING[model_id] = {"provider", "input_per_1m_usd", "output_per_1m_usd", "cached_input_per_1m_usd",
    "cache_write_per_1m_usd", "rates" (short / long context), "fetch_date", "source_url", "price_file", and "tier"
    for the two tier models}. CLOUD_MODELS = {"cheap", "strong"} from the preferred provider's file "selection"
    (Anthropic when its price file exists, else OpenAI). No price file at all gives two empty dicts: every real call
    is then refused with UnknownModelError (fail closed)."""
    root = Path(repo) if repo is not None else _prices.REPO
    pricing: dict[str, dict[str, Any]] = {}
    for model_id, e in _prices.pricing_index(root).items():
        short = e["rates"]["short"]
        pricing[model_id] = {
            "provider": e["provider"],
            "input_per_1m_usd": short["input"],
            "output_per_1m_usd": short["output"],
            "cached_input_per_1m_usd": short["cached_input"],
            "cache_write_per_1m_usd": short["cache_write"],
            "rates": e["rates"],
            "fetch_date": e["fetched_utc"],
            "source_url": e["source_url"],
            "price_file": e["price_file"],
        }
    tiers: dict[str, str] = {}
    provider = _prices.preferred_provider(root)
    if provider is not None:
        tables = _prices.load_tables(root)
        tiers = {t: m for t, m in _prices.tier_models(tables[provider]["table"]).items() if m in pricing}
        for tier, model_id in tiers.items():
            pricing[model_id]["tier"] = tier
    return pricing, tiers


MODEL_PRICING, CLOUD_MODELS = load_model_pricing()


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------


class HostRestrictionError(Exception):
    """Raised when a real cloud call is attempted from a disallowed host."""


class SpendCapExceeded(Exception):
    """Raised when a real call would push running spend past the cap."""


class UnknownModelError(Exception):
    """Raised when a model id has no entry in MODEL_PRICING."""


class MissingApiKeyError(RuntimeError):
    """Raised when mode="real" is requested but no key is available."""


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def redact_key(key: str | None) -> str:
    """Return a redacted form of a key-shaped string: at most first/last 2 chars.

    Never returns more than 4 real characters of the input, regardless of
    length. Safe to print or log.
    """
    if not key:
        return "<none>"
    if len(key) <= 4:
        return "*" * len(key)
    return f"{key[:2]}{'*' * (len(key) - 4)}{key[-2:]}"


def resolve_hostname() -> str:
    """Return the current machine's hostname (socket.gethostname())."""
    return socket.gethostname()


def check_host_allowed(hostname: str | None = None, role_override: str | None = None) -> bool:
    """Explicit, fail-loud host check for the REAL-call path.

    Order: hard deny (t2s) -> explicit role override ("controller") ->
    allow-list (CLOUD_ALLOWED_HOSTS env var, default ("evo-x2",)) -> refuse.

    Returns True if allowed; raises HostRestrictionError otherwise. Never
    returns False -- a disallowed host is always an exception, never a
    silent fallback.
    """
    resolved_hostname = hostname if hostname is not None else resolve_hostname()
    lname = resolved_hostname.lower()

    for bad in _HARD_DENY_SUBSTRINGS:
        if bad in lname:
            raise HostRestrictionError(
                f"cloud client hard-denied on host '{resolved_hostname}' "
                f"(matches permanent deny pattern '{bad}'); this module must "
                "never make real cloud calls on evo-t2s, no override exists for this check"
            )

    resolved_role = role_override if role_override is not None else os.environ.get(ENV_VAR_HOST_ROLE, "")
    if resolved_role.strip().lower() == "controller":
        return True

    allowed_env = os.environ.get(ENV_VAR_ALLOWED_HOSTS, "")
    allowed = tuple(s.strip().lower() for s in allowed_env.split(",") if s.strip()) or _DEFAULT_ALLOWED_SUBSTRINGS
    for ok in allowed:
        if ok in lname:
            return True

    raise HostRestrictionError(
        f"cloud client not allowed on host '{resolved_hostname}'. "
        f"Allowed hostname substrings: {allowed}. Set {ENV_VAR_ALLOWED_HOSTS} "
        f"or {ENV_VAR_HOST_ROLE}=controller to explicitly permit this host."
    )


def _approx_token_count(messages: list[dict[str, Any]]) -> int:
    """Whitespace-based approximate token count. Not a real tokenizer.

    Good enough for spend-cap estimation and stub-mode bookkeeping; do not
    use this for anything that needs exact billing-grade token counts.
    """
    text = " ".join(str(m.get("content", "")) for m in messages)
    return max(1, len(text.split()))


def estimate_cost_usd(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """Project the USD cost of a call from the shared price files (Standard list prices; long-context rates above
    the table's boundary). Every input token is charged at the uncached input rate: cache-read discounts are never
    credited, so the cap check and the ledger are an upper bound on real spend."""
    if model_id not in MODEL_PRICING:
        raise UnknownModelError(
            f"no pricing entry for model '{model_id}' in the shared price files "
            f"(results/{_prices.PRICE_GLOB}); known models: {sorted(MODEL_PRICING)}"
        )
    rates = MODEL_PRICING[model_id].get("rates")
    if rates is not None:
        r = _prices.rate_for(rates, input_tokens)
        inp, outp = r["input"], r["output"]
    else:  # an entry built by hand (tests) with only the flat per-1M fields
        inp, outp = MODEL_PRICING[model_id]["input_per_1m_usd"], MODEL_PRICING[model_id]["output_per_1m_usd"]
    return (input_tokens / 1_000_000) * inp + (output_tokens / 1_000_000) * outp


def usage_breakdown(usage: dict[str, Any] | None, fallback_input_tokens: int = 0,
                    fallback_output_tokens: int = 0) -> dict[str, Any]:
    """Normalize a provider usage dict to {input_tokens (all input, cache included), uncached_input_tokens,
    cache_read_tokens, cache_write_tokens (all writes), cache_write_1h_tokens (the part reported as 1-hour writes),
    output_tokens, has_usage}.

    Shapes read: litellm / OpenAI (prompt_tokens INCLUDES the cache tokens; cache reads under
    cache_read_input_tokens or prompt_tokens_details.cached_tokens; writes under cache_creation_input_tokens) and
    Anthropic's own (input_tokens EXCLUDES the cache tokens; cache_read_input_tokens, cache_creation_input_tokens,
    and an optional cache_creation {ephemeral_5m_input_tokens, ephemeral_1h_input_tokens} split). Same rule as
    scripts/cloud_redo.py::_usage. With no usage at all the fallbacks are used, every input token uncached, and
    has_usage is False."""
    u = usage or {}
    has_usage = bool(u) and any(u.get(k) is not None for k in ("prompt_tokens", "input_tokens", "completion_tokens",
                                                                "output_tokens"))
    if not has_usage:
        inp, out = int(fallback_input_tokens), int(fallback_output_tokens)
        return {"input_tokens": inp, "uncached_input_tokens": inp, "cache_read_tokens": 0, "cache_write_tokens": 0,
                "cache_write_1h_tokens": 0, "output_tokens": out, "has_usage": False}
    read = int(u.get("cache_read_input_tokens") or (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    write = int(u.get("cache_creation_input_tokens") or 0)
    split = u.get("cache_creation") if isinstance(u.get("cache_creation"), dict) else {}
    w5, w1h = int(split.get("ephemeral_5m_input_tokens") or 0), int(split.get("ephemeral_1h_input_tokens") or 0)
    if not write and (w5 or w1h):
        write = w5 + w1h
    w1h = min(w1h, write)
    if u.get("prompt_tokens") is not None:
        prompt = int(u.get("prompt_tokens") or 0)
        unc = prompt - read - write if prompt >= read + write else prompt
    else:
        unc = int(u.get("input_tokens") or 0)
    out_tok = u.get("completion_tokens")
    if out_tok is None:
        out_tok = u.get("output_tokens")
    out = int(out_tok if out_tok is not None else fallback_output_tokens)
    return {"input_tokens": unc + read + write, "uncached_input_tokens": unc, "cache_read_tokens": read,
            "cache_write_tokens": write, "cache_write_1h_tokens": w1h, "output_tokens": out, "has_usage": True}


def request_cache_ttl(messages: Any = None, kwargs: dict[str, Any] | None = None) -> str:
    """'1h' when any cache_control block in the request (messages, system, tools, or a top-level cache_control)
    asks for ttl '1h'; else '5m' (Anthropic's default lifetime)."""
    def _walk(x: Any) -> bool:
        if isinstance(x, dict):
            cc = x.get("cache_control")
            if isinstance(cc, dict) and str(cc.get("ttl", "")).lower() == "1h":
                return True
            return any(_walk(v) for v in x.values())
        if isinstance(x, (list, tuple)):
            return any(_walk(v) for v in x)
        return False
    return "1h" if _walk(messages) or _walk(kwargs or {}) else "5m"


def _model_rates(model_id: str, prompt_tokens: int) -> dict[str, Any]:
    if model_id not in MODEL_PRICING:
        raise UnknownModelError(
            f"no pricing entry for model '{model_id}' in the shared price files "
            f"(results/{_prices.PRICE_GLOB}); known models: {sorted(MODEL_PRICING)}"
        )
    e = MODEL_PRICING[model_id]
    rates = e.get("rates")
    if rates is not None:
        return dict(_prices.rate_for(rates, prompt_tokens))
    # an entry built by hand (tests) with only the flat per-1M fields
    return {"input": e["input_per_1m_usd"], "output": e["output_per_1m_usd"],
            "cached_input": e.get("cached_input_per_1m_usd"), "cache_write": e.get("cache_write_per_1m_usd"),
            "cache_write_1h": e.get("cache_write_1h_per_1m_usd")}


def billed_cost_usd(model_id: str, breakdown: dict[str, Any], cache_ttl: str = "5m") -> float:
    """BILLED USD of one finished call from its usage_breakdown(): uncached input at the input rate, cache reads at
    the cache-hit rate, cache writes at the 5-minute write rate (1-hour writes, and every write when cache_ttl is
    '1h' and the usage gives no 1-hour split, at the 1-hour rate), output at the output rate. Rates come from the
    shared price files; the context tier is chosen by the call's total input tokens. A missing cache rate in the
    price file falls back to a conservative one (read: the input rate; 5m write: the input rate x 1.25; 1h write:
    the input rate x 2), never to zero."""
    r = _model_rates(model_id, int(breakdown["input_tokens"]))
    inp, outp = float(r["input"]), float(r["output"])
    hit = float(r["cached_input"]) if r.get("cached_input") is not None else inp
    w5 = float(r["cache_write"]) if r.get("cache_write") is not None else inp * 1.25
    w1h_rate = float(r["cache_write_1h"]) if r.get("cache_write_1h") is not None else inp * 2.0
    write = int(breakdown.get("cache_write_tokens") or 0)
    w1h = int(breakdown.get("cache_write_1h_tokens") or 0)
    if cache_ttl == "1h" and not w1h:
        w1h = write
    w5_tok = write - w1h
    return (int(breakdown["uncached_input_tokens"]) * inp + int(breakdown.get("cache_read_tokens") or 0) * hit
            + w5_tok * w5 + w1h * w1h_rate + int(breakdown["output_tokens"]) * outp) / 1_000_000


def _uses_cache_control(messages: Any = None, kwargs: dict[str, Any] | None = None) -> bool:
    """True when any cache_control block appears in the request (so the call may write the cache)."""
    def _walk(x: Any) -> bool:
        if isinstance(x, dict):
            return "cache_control" in x or any(_walk(v) for v in x.values())
        if isinstance(x, (list, tuple)):
            return any(_walk(v) for v in x)
        return False
    return _walk(messages) or _walk(kwargs or {})


def reservation_cost_usd(model_id: str, input_tokens: int, output_tokens: int, cache_ttl: str | None = None) -> float:
    """Pre-call reservation for the cap check. Cache hits are unknown before a call, so this is the upper bound
    (estimate_cost_usd, every input token at the uncached rate). When the request carries cache_control
    (cache_ttl given), a cache WRITE costs more than an uncached token (5m: 1.25x, 1h: 2x the input rate), so the
    reservation is the larger of that upper bound and every input token billed as a write at that TTL: the billed
    cost of the call can then never exceed what was reserved for its input."""
    upper = estimate_cost_usd(model_id, input_tokens, output_tokens)
    if cache_ttl is None:
        return upper
    all_write = {"input_tokens": input_tokens, "uncached_input_tokens": 0, "cache_read_tokens": 0,
                 "cache_write_tokens": input_tokens, "cache_write_1h_tokens": input_tokens if cache_ttl == "1h" else 0,
                 "output_tokens": output_tokens}
    return max(upper, billed_cost_usd(model_id, all_write, cache_ttl))


# --------------------------------------------------------------------------
# Result type
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CallResult:
    """Normalized result of a cloud call, real or stub."""

    model_id: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    stub: bool
    content: str
    raw: dict[str, Any] = field(default_factory=dict)
    # Every input token at the uncached rate (the pre-2026-10-22 ledger cost). cost_usd is the BILLED cost.
    cost_usd_upper_bound: float | None = None
    usage: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Provider completion functions -- imports deferred
# --------------------------------------------------------------------------


def _default_completion_fn(*, model_id: str, messages: list[dict[str, Any]], api_key: str, **kwargs: Any) -> dict[str, Any]:
    """Real provider call via litellm (non-OpenAI models). Import is deferred so stub mode and
    unit tests never require litellm to be installed."""
    try:
        import litellm  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised only without the dependency
        raise RuntimeError(
            "litellm is not installed. Add 'litellm' to pyproject.toml's "
            "[project].dependencies before making a real cloud call."
        ) from exc
    response = litellm.completion(model=model_id, messages=messages, api_key=api_key, **kwargs)
    return response.model_dump() if hasattr(response, "model_dump") else dict(response)


def _make_openai_sdk_client(api_key: str) -> Any:
    """The one place in the repo that constructs an OpenAI SDK client for api.openai.com."""
    from openai import OpenAI  # noqa: PLC0415  (deferred: stub mode never needs the SDK)

    return OpenAI(api_key=api_key)


def local_openai_compatible_client(base_url: str, api_key: str = "ollama") -> Any:
    """OpenAI-SDK client for a LOCAL OpenAI-compatible server (Ollama, llama-server). No spend, so no cap.

    Exists so harness/adapters/sdk_direct.py's --backend ollama path does not need its own `OpenAI(...)`
    (tests/test_no_direct_openai_clients.py allows that constructor only in this module). Refuses any
    base_url that points at an openai.com host, so it can never be used to bypass the cap.
    """
    from urllib.parse import urlparse  # noqa: PLC0415

    host = (urlparse(base_url or "").hostname or "").lower()
    if not host:
        raise ValueError(f"local_openai_compatible_client needs an absolute base_url, got {base_url!r}")
    if host == "openai.com" or host.endswith(".openai.com"):
        raise ValueError(
            f"local_openai_compatible_client refuses {base_url!r}: OpenAI itself must go through the capped "
            "CloudClient (mode='real'), not this uncapped local helper"
        )
    from openai import OpenAI  # noqa: PLC0415

    return OpenAI(base_url=base_url, api_key=api_key)


def _response_to_dict(response: Any) -> dict[str, Any]:
    """Normalize a provider response (dict, or an SDK pydantic object) to a dict, the way the harness
    backends always have (model_dump(exclude_unset=False))."""
    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):
        return response.model_dump(exclude_unset=False)
    return dict(response)


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------


@dataclass
class PreparedCall:
    """A real (or stub) call that has passed every pre-send check. See CloudClient.prepare()."""

    client: "CloudClient"
    model_id: str
    messages: list[dict[str, Any]]
    kwargs: dict[str, Any]
    input_tokens_est: int
    expected_output_tokens: int
    reserved_usd: float
    stub: bool
    _done: bool = False

    def send(self) -> Any:
        """The provider request only (the part a latency-measuring caller should time).

        Returns the provider's raw response (an SDK object or a dict). On any exception the reservation is
        released and nothing is written to the ledger.
        """
        if self.stub:
            return self.client._stub_call(self.model_id, self.messages, self.expected_output_tokens).raw
        try:
            return self.client._send_fn(self.model_id)(
                model_id=self.model_id, messages=self.messages, api_key=self.client.api_key, **self.kwargs
            )
        except BaseException:
            self.release()
            raise

    def settle(self, response: Any) -> CallResult:
        """Record the finished call: ledger row, running total, spend alerts. Returns the CallResult."""
        if self.stub:
            return self.client._stub_call(self.model_id, self.messages, self.expected_output_tokens)
        if self._done:
            raise RuntimeError("PreparedCall.settle() called twice")
        return self.client._settle(self, _response_to_dict(response))

    def release(self) -> None:
        """Drop the reservation without recording a call (the request failed or was never sent)."""
        if not self.stub and not self._done:
            self._done = True
            self.client._release(self.reserved_usd)


class CloudClient:
    """Cloud LLM access with a host restriction, stub fallback, spend cap,
    and a spend ledger. See the module docstring for the full contract,
    including the KEY AND MODE RULE.
    """

    def __init__(
        self,
        *,
        spend_cap_usd: float = DEFAULT_SPEND_CAP_USD,
        ledger_path: str | Path = DEFAULT_LEDGER_PATH,
        hostname: str | None = None,
        role_override: str | None = None,
        api_key: str | None = None,
        mode: str = "auto",
        completion_fn: Callable[..., Any] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if mode not in CLIENT_MODES:
            raise ValueError(f"mode must be one of {CLIENT_MODES}, got {mode!r}")
        self.mode = mode
        self.spend_cap_usd = float(spend_cap_usd)
        self.ledger_path = Path(ledger_path)
        self._hostname = hostname
        self._role_override = role_override
        self._completion_fn = completion_fn
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._lock = threading.Lock()
        self._reserved_usd = 0.0
        self._sdk_client: Any = None
        self.cap_basis = "billed_with_recovered_usage"

        # Key resolution: see KEY AND MODE RULE in the module docstring.
        self.api_key, self.api_key_source = self._resolve_key(mode, api_key)
        self.stub_mode = not bool(self.api_key)
        if not self.stub_mode and self.spend_cap_usd > HARD_SPEND_CAP_USD + 1e-9:
            raise ValueError(
                f"spend_cap_usd ${self.spend_cap_usd:.2f} exceeds the hard cap ${HARD_SPEND_CAP_USD:.2f} total; "
                "a real-mode client may only lower the cap"
            )

        self._running_total_usd, self._running_total_upper_usd = self._read_running_totals()
        self._fired_alerts = self._read_fired_alerts()

    @classmethod
    def real(cls, **kwargs: Any) -> "CloudClient":
        """Explicit real-mode factory: CloudClient(mode="real", ...). Raises MissingApiKeyError with no key."""
        return cls(mode="real", **kwargs)

    @staticmethod
    def _resolve_key(mode: str, api_key: str | None) -> tuple[str | None, str | None]:
        if mode == "stub":
            return None, None
        if api_key:
            return api_key, "argument"
        if mode == "auto":
            return None, None  # env vars are deliberately not consulted: a stub stays a stub
        for var in REAL_MODE_KEY_ENV_VARS:
            val = os.environ.get(var)
            if val:
                return val, var
        raise MissingApiKeyError(
            "CloudClient(mode='real') needs a key: pass api_key=..., or set "
            f"{' or '.join(REAL_MODE_KEY_ENV_VARS)} (checked in that order); refusing to fall back to a stub"
        )

    # -- budget bookkeeping -------------------------------------------------

    def _read_running_totals(self) -> tuple[float, float]:
        """(billed, upper bound) running totals over every existing ledger call row (real calls only), on the
        billed cap basis (module docstring, BILLED CAP BASIS AT START): analysis/cloud_spend_billed.py::compute, the
        register row's own computation, so an old row whose usage is on file counts at its billed cost and one
        without counts at its upper bound. Falls back to _read_running_totals_per_row if that fails."""
        if not self.ledger_path.exists():
            return 0.0, 0.0
        try:
            try:
                from . import console_gate as _cg  # noqa: PLC0415
            except ImportError:   # run as a script: load the sibling module by path
                import importlib.util  # noqa: PLC0415
                spec = importlib.util.spec_from_file_location("console_gate", Path(__file__).with_name("console_gate.py"))
                _cg = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(_cg)
            s = _cg.ledger_spend(self.ledger_path)
            self.cap_basis = "billed_with_recovered_usage"
            return s["billed_usd"], s["upper_bound_usd"]
        except Exception as e:  # noqa: BLE001 - any failure falls back to the conservative per-row basis
            print(f"WARNING: billed cap basis unavailable ({type(e).__name__}: {e}); counting every ledger row "
                  "without a billed cost at its upper bound (conservative)", flush=True)
            self.cap_basis = "per_row_old_rows_at_upper_bound"
            return self._read_running_totals_per_row()

    def _read_running_totals_per_row(self) -> tuple[float, float]:
        """Fallback (the pre-2026-10-23 basis): a new row (2026-10-22 on) counts cost_usd (billed) and
        cost_usd_upper_bound. An old row has only cost_usd, which was the upper bound, and no usage breakdown: it
        counts toward both totals (billed stays conservative). Alert rows and unparseable lines are skipped."""
        if not self.ledger_path.exists():
            return 0.0, 0.0
        billed = upper = 0.0
        with self.ledger_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict) or row.get("record") == "alert":
                        continue
                    cost = float(row.get("cost_usd", 0.0))
                    ub = row.get("cost_usd_upper_bound")
                    ub = cost if ub is None else float(ub)
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
                billed += cost
                upper += ub
        return billed, upper

    def _read_running_total(self) -> float:
        """BILLED running total across every existing ledger row (see _read_running_totals for old rows)."""
        return self._read_running_totals()[0]

    def _read_fired_alerts(self) -> set[float]:
        """Spend-alert thresholds (fractions of the cap) already written to the ledger."""
        fired: set[float] = set()
        if not self.ledger_path.exists():
            return fired
        with self.ledger_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
                if isinstance(row, dict) and row.get("record") == "alert" and row.get("source") == "cloud_client":
                    try:
                        fired.add(round(float(row.get("threshold_fraction")), 4))
                    except (TypeError, ValueError):
                        continue
        return fired

    def _fire_spend_alerts(self) -> list[dict[str, Any]]:
        """Fire each not-yet-fired threshold the running total has reached: one ledger row each, and a print.
        Returns the rows written by this call."""
        fired_now = []
        for frac in SPEND_ALERT_FRACTIONS:
            key = round(frac, 4)
            if key in self._fired_alerts or self._running_total_usd + 1e-9 < frac * self.spend_cap_usd:
                continue
            ts = self._clock().isoformat()
            row = {
                "record": "alert",
                "source": "cloud_client",
                "reason": f"cloud spend reached {frac:.0%} of the ${self.spend_cap_usd:.2f} cap",
                "threshold_fraction": frac,
                "threshold_usd": frac * self.spend_cap_usd,
                "cap_usd": self.spend_cap_usd,
                "running_total_usd": self._running_total_usd,
                "running_total_usd_upper_bound": self._running_total_upper_usd,
                "basis": "billed",
                "ts_utc": ts,
                "timestamp": ts,
            }
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with self.ledger_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
            print(
                f"ALERT: cloud spend ${self._running_total_usd:.4f} billed (upper bound "
                f"${self._running_total_upper_usd:.4f}) reached {frac:.0%} of the ${self.spend_cap_usd:.2f} cap",
                flush=True,
            )
            self._fired_alerts.add(key)
            fired_now.append(row)
        return fired_now

    @property
    def running_total_usd(self) -> float:
        """BILLED running total (what the cap and the alerts use)."""
        return self._running_total_usd

    @property
    def running_total_usd_upper_bound(self) -> float:
        """Running total with every input token at the uncached rate (old rows: their cost_usd)."""
        return self._running_total_upper_usd

    @property
    def remaining_budget_usd(self) -> float:
        """Cap headroom on the BILLED basis (what the cap check uses)."""
        return max(0.0, self.spend_cap_usd - self._running_total_usd)

    @property
    def remaining_budget_usd_upper_bound(self) -> float:
        """Cap headroom if the upper-bound total were charged instead; reported beside the billed headroom only."""
        return max(0.0, self.spend_cap_usd - self._running_total_upper_usd)

    def spend_summary(self) -> str:
        """One line, billed and upper bound side by side, for cap messages and pre-job projections."""
        return (f"cap ${self.spend_cap_usd:.2f}: spent billed ${self._running_total_usd:.4f} (upper bound "
                f"${self._running_total_upper_usd:.4f}); headroom billed ${self.remaining_budget_usd:.4f} "
                f"(upper bound ${self.remaining_budget_usd_upper_bound:.4f}); the cap checks the billed figure")

    def estimate_cost(self, model_id: str, input_tokens: int, output_tokens: int) -> float:
        return estimate_cost_usd(model_id, input_tokens, output_tokens)

    # -- ledger ---------------------------------------------------------------

    def _append_ledger(self, result: CallResult, tool_choice: Any = None) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ub = result.cost_usd if result.cost_usd_upper_bound is None else result.cost_usd_upper_bound
        u = result.usage or {}
        row = {
            "timestamp": self._clock().isoformat(),
            "model_id": result.model_id,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "uncached_input_tokens": u.get("uncached_input_tokens", result.input_tokens),
            "cache_read_tokens": u.get("cache_read_tokens", 0),
            "cache_write_tokens": u.get("cache_write_tokens", 0),
            "cache_write_1h_tokens": u.get("cache_write_1h_tokens", 0),
            "cache_ttl": u.get("cache_ttl", "5m"),
            "cost_basis": "billed_from_usage" if u.get("has_usage") else "upper_bound_no_usage",
            "cost_usd": result.cost_usd,
            "cost_usd_upper_bound": ub,
            "running_total_usd": self._running_total_usd + result.cost_usd,
            "running_total_usd_upper_bound": self._running_total_upper_usd + ub,
            "running_total_basis": self.cap_basis,
        }
        if tool_choice is not None:
            row["tool_choice"] = tool_choice
        with self.ledger_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    # -- stub path --------------------------------------------------------------

    def _stub_call(self, model_id: str, messages: list[dict[str, Any]], expected_output_tokens: int) -> CallResult:
        input_tokens = _approx_token_count(messages)
        content = (
            f"[STUB RESPONSE for {model_id}] deterministic fake reply; "
            "no real API call was made; this client is in stub mode (see KEY AND MODE RULE)."
        )
        return CallResult(
            model_id=f"stub-{model_id}",
            input_tokens=input_tokens,
            output_tokens=expected_output_tokens,
            cost_usd=0.0,
            stub=True,
            content=content,
            raw={"stub": True, "model": model_id},
        )

    # -- real path --------------------------------------------------------------

    def _send_fn(self, model_id: str) -> Callable[..., Any]:
        """The provider function for one model. An injected completion_fn (tests) always wins."""
        if self._completion_fn is not None:
            return self._completion_fn
        if MODEL_PRICING[model_id].get("provider") == "openai":
            sdk = self._sdk_client

            def _openai_send(*, model_id: str, messages: list[dict[str, Any]], api_key: str, **kwargs: Any) -> Any:
                return sdk.chat.completions.create(model=model_id, messages=messages, **kwargs)

            return _openai_send
        return _default_completion_fn

    def count_tokens(self, *, model_id: str, messages: list[dict[str, Any]], count_fn: Callable[..., int] | None = None) -> int:
        """Anthropic messages.count_tokens for one request (input tokens on Anthropic's tokenizer). Not billed, so no
        spend reservation and no ledger row; the caller records the call. Real mode only (a stub has no tokenizer to
        ask), host check, Anthropic models only, key provider must match. count_fn (tests) replaces the SDK call."""
        if self.stub_mode:
            raise MissingApiKeyError("count_tokens needs a real-mode client (Anthropic's tokenizer is only remote)")
        check_host_allowed(self._hostname, self._role_override)
        if MODEL_PRICING.get(model_id, {}).get("provider") != "anthropic":
            raise UnknownModelError(f"count_tokens is for Anthropic models in the shared price files, got '{model_id}'")
        key_provider = KEY_ENV_PROVIDER.get(self.api_key_source or "")
        if key_provider is not None and key_provider != "anthropic":
            raise HostRestrictionError(f"key from {self.api_key_source} is not an Anthropic key")
        if count_fn is not None:
            return int(count_fn(model_id=model_id, messages=messages))
        import anthropic  # deferred: only the real count path needs the SDK
        sdk = anthropic.Anthropic(api_key=self.api_key)
        return int(sdk.messages.count_tokens(model=model_id, messages=messages).input_tokens)

    def prepare(
        self,
        *,
        model_id: str,
        messages: list[dict[str, Any]],
        expected_output_tokens: int | None = None,
        **kwargs: Any,
    ) -> PreparedCall:
        """Every pre-send check, done outside any timed region.

        Stub mode: returns a stub PreparedCall (no host check, no network, no ledger).
        Real mode: host check, known-model check, provider/key match, projected cost against the cap
        (counting other in-flight reservations), then reserves the projected cost. The returned
        PreparedCall's send() is the provider request; settle() records it.

        expected_output_tokens defaults to the call's max_tokens / max_completion_tokens if given, else 256.
        """
        if expected_output_tokens is None:
            expected_output_tokens = int(kwargs.get("max_tokens") or kwargs.get("max_completion_tokens") or 256)
        input_tokens = _approx_token_count(messages)
        if self.stub_mode:
            return PreparedCall(self, model_id, messages, kwargs, input_tokens, expected_output_tokens, 0.0, True)

        check_host_allowed(self._hostname, self._role_override)

        if model_id not in MODEL_PRICING:
            raise UnknownModelError(
                f"no pricing entry for model '{model_id}' in the shared price files "
                f"(results/{_prices.PRICE_GLOB}); known models: {sorted(MODEL_PRICING)}"
            )
        provider = MODEL_PRICING[model_id].get("provider")
        key_provider = KEY_ENV_PROVIDER.get(self.api_key_source or "")
        if key_provider is not None and provider != key_provider:
            raise ValueError(
                f"model '{model_id}' is a {provider} model but this client's key came from "
                f"{self.api_key_source} ({key_provider}); pass that provider's key explicitly"
            )

        if kwargs.get("tools"):
            input_tokens += max(1, len(json.dumps(kwargs["tools"]).split()))
        projected_cost = reservation_cost_usd(
            model_id, input_tokens, expected_output_tokens,
            request_cache_ttl(messages, kwargs) if _uses_cache_control(messages, kwargs) else None)

        with self._lock:
            committed = self._running_total_usd + self._reserved_usd
            if committed + projected_cost > self.spend_cap_usd + 1e-9:
                in_flight = (f", plus ${self._reserved_usd:.4f} reserved by in-flight calls"
                             if self._reserved_usd else "")
                reason = (
                    f"refused: projected cost ${projected_cost:.4f} would bring running "
                    f"total to ${committed + projected_cost:.4f}, exceeding "
                    f"the ${self.spend_cap_usd:.2f} cap (current running total "
                    f"${self._running_total_usd:.4f} billed, upper bound ${self._running_total_upper_usd:.4f}"
                    f"{in_flight}; the cap checks the billed total; the projection is an upper bound, every "
                    "input token uncached, since cache hits are unknown before the call)"
                )
                raise SpendCapExceeded(reason)
            self._reserved_usd += projected_cost
            if (self._completion_fn is None and provider == "openai" and self._sdk_client is None):
                try:
                    self._sdk_client = _make_openai_sdk_client(self.api_key)
                except BaseException:
                    self._reserved_usd -= projected_cost
                    raise
        return PreparedCall(self, model_id, messages, kwargs, input_tokens, expected_output_tokens,
                            projected_cost, False)

    def _release(self, reserved_usd: float) -> None:
        with self._lock:
            self._reserved_usd = max(0.0, self._reserved_usd - reserved_usd)

    def _settle(self, prepared: PreparedCall, response_json: dict[str, Any]) -> CallResult:
        """Settle a finished call on its BILLED cost (usage fields); the upper bound is kept beside it."""
        u = usage_breakdown(response_json.get("usage"), prepared.input_tokens_est, prepared.expected_output_tokens)
        u["cache_ttl"] = request_cache_ttl(prepared.messages, prepared.kwargs)
        upper = self.estimate_cost(prepared.model_id, u["input_tokens"], u["output_tokens"])
        billed = billed_cost_usd(prepared.model_id, u, u["cache_ttl"]) if u["has_usage"] else upper

        content = ""
        choices = response_json.get("choices") or []
        if choices:
            content = (choices[0].get("message") or {}).get("content", "") or ""

        result = CallResult(
            model_id=prepared.model_id,
            input_tokens=u["input_tokens"],
            output_tokens=u["output_tokens"],
            cost_usd=billed,
            stub=False,
            content=content,
            raw=response_json,
            cost_usd_upper_bound=upper,
            usage=u,
        )
        with self._lock:
            prepared._done = True
            self._reserved_usd = max(0.0, self._reserved_usd - prepared.reserved_usd)
            self._append_ledger(result, tool_choice=prepared.kwargs.get("tool_choice"))
            self._running_total_usd += result.cost_usd
            self._running_total_upper_usd += upper
            self._fire_spend_alerts()
        return result

    def call(
        self,
        *,
        model_id: str,
        messages: list[dict[str, Any]],
        expected_output_tokens: int | None = None,
        **kwargs: Any,
    ) -> CallResult:
        """Make a (possibly stub) cloud call: prepare() + send() + settle().

        In stub mode: always succeeds, never touches the network or the
        ledger, never checks host restriction (stub mode is safe everywhere,
        including evo-t2s).

        In real mode: enforces the host restriction, then the spend cap,
        before making any request; appends one ledger row on success.
        Extra kwargs (tools, tool_choice, ...) reach the provider unchanged.
        """
        prepared = self.prepare(model_id=model_id, messages=messages,
                                expected_output_tokens=expected_output_tokens, **kwargs)
        if prepared.stub:
            return self._stub_call(model_id, messages, prepared.expected_output_tokens)
        response = prepared.send()
        return prepared.settle(response)


# --------------------------------------------------------------------------
# Reporting helper
# --------------------------------------------------------------------------


def print_stub_banner_if_needed(client: CloudClient) -> None:
    """Print the mandated "BLOCKED: cloud key" banner when in stub mode.

    Call this as the very first output of any script/report built on this
    module, before any other print statement.
    """
    if client.stub_mode:
        print("BLOCKED: cloud key")


# --------------------------------------------------------------------------
# Stub dry run (module self-test, no network, no key)
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        demo_ledger = Path(tmp) / "cloud_ledger_demo.jsonl"
        demo_client = CloudClient(api_key=None, ledger_path=demo_ledger)
        print_stub_banner_if_needed(demo_client)
        print(f"stub_mode={demo_client.stub_mode}")
        print(f"hostname={redact_key(resolve_hostname())}  (redacted for this printout only)")
        for tier, model_id in CLOUD_MODELS.items():
            res = demo_client.call(
                model_id=model_id,
                messages=[{"role": "user", "content": "What is the SLA for a P1 ticket?"}],
                expected_output_tokens=32,
            )
            print(
                f"[{tier}] model_id={res.model_id} stub={res.stub} "
                f"input_tokens={res.input_tokens} output_tokens={res.output_tokens} "
                f"cost_usd={res.cost_usd:.4f}"
            )
            print(f"    content: {res.content}")
        print(f"ledger rows written (should be 0 in stub mode): "
              f"{sum(1 for _ in demo_ledger.open()) if demo_ledger.exists() else 0}")
        print(f"running_total_usd={demo_client.running_total_usd:.4f}  "
              f"remaining_budget_usd={demo_client.remaining_budget_usd:.2f}")
