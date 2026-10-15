"""Shared cloud price tables: results/cloud_prices_*.json, the latest file per provider.

One loader for every consumer that prices a cloud call (src/cloud/client.py's spend cap and ledger,
analysis/r2_session_cost.py's per-session cost). No price is hard-coded here: every number comes from a committed
price file transcribed from the vendor's own pricing page (each file carries its source URL and fetch time).

File naming
-----------
  results/cloud_prices_<YYYYMMDD>.json             provider from the file's "provider" field, else from its
                                                    source_url host (openai.com -> openai; anthropic.com or
                                                    claude.com -> anthropic). results/cloud_prices_20261008.json is
                                                    the OpenAI table.
  results/cloud_prices_<provider>_<YYYYMMDD>.json  provider from the name (e.g. cloud_prices_anthropic_20261014.json).
The latest date wins per provider; a new file for a provider replaces the older one with no code change.

Table shape (the OpenAI file's; the Anthropic file is read the same way)
------------------------------------------------------------------------
  "models": {model_id: {"standard": {"short": RATES, "long": RATES}, "batch": {...}}}
  RATES = {"input", "cached_input", "cache_write", "output"} in USD per 1M tokens.
Tolerated variants, so a differently laid out file still loads: "standard" given as RATES directly (no short/long
split), or the model entry given as RATES directly; the cache-read rate under "cached_input", "cache_read",
"cache_read_input" or "cache_hit"; the cache-write rate under "cache_write", "cache_write_5m" or "cache_creation".
"other_rows_short_context": {"standard": {model_id: [input, cached_input or null, output]}} adds short-context rows.
"long_context_above" (input tokens) sets the short/long boundary; without it the OpenAI page's 272,000 applies to a
table that has long-context rates (cloud_budget_forecast.LONG_CONTEXT_ABOVE, quoted in the OpenAI file).
"selection": {"cheap_small" or "cheap": id, "mid_tier" or "strong": id} names the cheap and strong tier models.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
PRICE_GLOB = "cloud_prices_*.json"
RESULTS_DIR = "results"
DEFAULT_LONG_CONTEXT_ABOVE = 272_000   # quoted in results/cloud_prices_20261008.json "context_tiers"
PREFERRED_PROVIDERS = ("anthropic", "openai")   # real-mode provider order (operator 2026-10-14: Anthropic first)

_NAME_RE = re.compile(r"^cloud_prices_(?:(?P<provider>[a-z][a-z0-9]*)_)?(?P<date>\d{8})\.json$")
_CACHED_KEYS = ("cached_input", "cache_read", "cache_read_input", "cache_hit")
_WRITE_KEYS = ("cache_write", "cache_write_5m", "cache_creation")


class PriceTableError(ValueError):
    """A price file that cannot be read as a price table."""


def _provider_from_table(table: dict) -> str | None:
    p = table.get("provider")
    if p:
        return str(p).lower()
    url = str(table.get("source_url") or "").lower()
    if "openai.com" in url:
        return "openai"
    if "anthropic.com" in url or "claude.com" in url:
        return "anthropic"
    return None


def price_files(repo: str | Path = REPO) -> dict[str, Path]:
    """{provider: path of its latest price file}. Files whose name does not parse are ignored."""
    best: dict[str, tuple[str, Path]] = {}
    for path in sorted((Path(repo) / RESULTS_DIR).glob(PRICE_GLOB)):
        m = _NAME_RE.match(path.name)
        if not m:
            continue
        provider = m.group("provider")
        if provider is None:
            try:
                provider = _provider_from_table(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                provider = None
            if provider is None:
                continue
        if provider not in best or m.group("date") > best[provider][0]:
            best[provider] = (m.group("date"), path)
    return {p: v[1] for p, v in sorted(best.items())}


def load_tables(repo: str | Path = REPO) -> dict[str, dict]:
    """{provider: {"path": repo-relative posix path, "table": the parsed JSON}} for each provider's latest file."""
    out = {}
    for provider, path in price_files(repo).items():
        try:
            rel = path.relative_to(Path(repo)).as_posix()
        except ValueError:
            rel = path.as_posix()
        out[provider] = {"path": rel, "table": json.loads(path.read_text(encoding="utf-8"))}
    return out


def _pick(d: dict, keys) -> Any:
    for k in keys:
        if d.get(k) is not None:
            return d[k]
    return None


def _norm_rates(d: Any, where: str) -> dict:
    if not isinstance(d, dict) or d.get("input") is None or d.get("output") is None:
        raise PriceTableError(f"{where}: expected a rates object with 'input' and 'output', got {d!r}")
    return {"input": float(d["input"]), "output": float(d["output"]),
            "cached_input": None if _pick(d, _CACHED_KEYS) is None else float(_pick(d, _CACHED_KEYS)),
            "cache_write": None if _pick(d, _WRITE_KEYS) is None else float(_pick(d, _WRITE_KEYS))}


def model_rates(table: dict, tier: str = "standard") -> dict[str, dict]:
    """{model_id: {"short": RATES, "long": RATES or None, "long_context_above": int or None}} at one pricing tier."""
    out = {}
    boundary = table.get("long_context_above")
    for model, entry in (table.get("models") or {}).items():
        t = entry.get(tier, entry) if isinstance(entry, dict) else entry
        if isinstance(t, dict) and "short" in t:
            short, long_ = _norm_rates(t["short"], f"{model}.{tier}.short"), t.get("long")
            long_ = None if long_ is None else _norm_rates(long_, f"{model}.{tier}.long")
        else:
            short, long_ = _norm_rates(t, f"{model}.{tier}"), None
        out[model] = {"short": short, "long": long_,
                      "long_context_above": (int(boundary) if boundary else DEFAULT_LONG_CONTEXT_ABOVE)
                      if long_ is not None else None}
    for model, row in ((table.get("other_rows_short_context") or {}).get(tier) or {}).items():
        if model in out or not isinstance(row, (list, tuple)) or len(row) != 3:
            continue
        inp, cached, outp = row
        out[model] = {"short": {"input": float(inp), "output": float(outp),
                                "cached_input": None if cached is None else float(cached), "cache_write": None},
                      "long": None, "long_context_above": None}
    return out


def rate_for(rates: dict, prompt_tokens: int) -> dict:
    """The RATES that apply to a prompt of this size (long-context rates above the boundary when the table has them)."""
    if rates["long"] is not None and prompt_tokens > rates["long_context_above"]:
        return rates["long"]
    return rates["short"]


def tier_models(table: dict) -> dict[str, str]:
    """{"cheap": id, "strong": id} from the table's "selection"; falls back to the cheapest and the most expensive
    model by input price when the file names no selection."""
    sel = table.get("selection") or {}
    cheap, strong = _pick(sel, ("cheap_small", "cheap")), _pick(sel, ("mid_tier", "strong"))
    if cheap and strong:
        return {"cheap": cheap, "strong": strong}
    rates = model_rates(table)
    if not rates:
        return {}
    by_price = sorted(rates, key=lambda m: (rates[m]["short"]["input"], m))
    return {"cheap": cheap or by_price[0], "strong": strong or by_price[-1]}


def pricing_index(repo: str | Path = REPO) -> dict[str, dict]:
    """{model_id: {"provider", "rates", "source_url", "fetched_utc", "price_file"}} over every provider's latest table.
    A model id present in two providers' files keeps the preferred provider's (PREFERRED_PROVIDERS order)."""
    tables = load_tables(repo)
    order = [p for p in PREFERRED_PROVIDERS if p in tables] + [p for p in tables if p not in PREFERRED_PROVIDERS]
    out: dict[str, dict] = {}
    for provider in order:
        t = tables[provider]["table"]
        for model, rates in model_rates(t).items():
            out.setdefault(model, {"provider": provider, "rates": rates, "source_url": t.get("source_url"),
                                   "fetched_utc": t.get("fetched_utc"), "price_file": tables[provider]["path"]})
    return out


def preferred_provider(repo: str | Path = REPO) -> str | None:
    tables = price_files(repo)
    return next((p for p in PREFERRED_PROVIDERS if p in tables), next(iter(tables), None))
