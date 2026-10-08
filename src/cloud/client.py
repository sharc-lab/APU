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

ENV VAR FOR THE KEY
--------------------
CLOUD_API_KEY is read once, and only consulted at all on an allowed host.
If it is unset (or empty), this module falls back to a deterministic STUB
PROVIDER instead of raising, so the router, Pareto sweep, and demo can all be
exercised end to end without a real key. Every row the stub produces carries
"stub": true and a model id prefixed "stub-" so a stub result can never be
mistaken for a real one downstream.

The key is never logged or printed in full. `redact_key()` below shows at
most the first 2 and last 2 characters of any key-shaped string.

SPEND CAP
---------
HARD_SPEND_CAP_USD = 50.00 USD TOTAL (2026-10-07): the hard cap whenever a real
key is set, summed over the whole ledger (not per month). DEFAULT_SPEND_CAP_USD
equals it. A caller may pass a LOWER spend_cap_usd; a real-mode client asking
for a higher one raises ValueError at construction (stub mode spends nothing,
so any cap is accepted there, e.g. for the Pareto sweep).
Before every REAL (non-stub) call, the projected cost is computed from the
model's published per-token price and the call's token counts. If
`running_total + projected_cost` would exceed the cap, the call is refused
before any request is sent, and the refusal (with its full reasoning) is
returned via `SpendCapExceeded`, never silently retried or downgraded. The
boundary is inclusive: a call that lands exactly on the cap is allowed; a
call that would push even fractionally past it is refused.

SPEND ALERTS
------------
After each real call, when the running total first reaches 50%, 75% and 90%
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
alert rows, above) to results/cloud_ledger.jsonl: timestamp (UTC ISO 8601), model id, input and
output tokens, cost in USD, and the running total after that call. Stub
calls are never written to the ledger -- the ledger is a record of real
spend only.

CLIENT LIBRARY
---------------
Real calls go through the `litellm` package (https://github.com/BerriAI/
litellm). As of 2026-09-30, litellm is NOT a dependency of this project
(checked: absent from pyproject.toml's `dependencies` list and not installed
in this environment -- `python -c "import litellm"` raises ModuleNotFoundError).
It must be added to pyproject.toml before the real-call path can run. The
import is deferred to inside `_default_completion_fn` specifically so that
stub-mode usage and all of this module's unit tests work without litellm
installed at all.

MODELS AND PRICING (fetched 2026-09-30, official vendor pages -- see
docs/CLOUD_MODELS.md for the full citation)
-----------------------------------------------------------------------
  cheap  : gpt-4o-mini        $0.15 / $0.60  per 1M input/output tokens
           source: https://developers.openai.com/api/docs/pricing
  strong : claude-sonnet-4-5  $3.00 / $15.00 per 1M input/output tokens
           source: https://claude.com/pricing
Prices change without notice; re-verify against the source URLs before
trusting these numbers past their fetch date.

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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

ENV_VAR_API_KEY = "CLOUD_API_KEY"
ENV_VAR_ALLOWED_HOSTS = "CLOUD_ALLOWED_HOSTS"
ENV_VAR_HOST_ROLE = "CLOUD_CLIENT_HOST_ROLE"

_HARD_DENY_SUBSTRINGS = ("t2s",)  # catches evo-t2s, EVO-T2S, t2s-anything -- never overridable
_DEFAULT_ALLOWED_SUBSTRINGS = ("evo-x2",)

HARD_SPEND_CAP_USD = 50.00  # 2026-10-07: hard cap, USD total, whenever a real key is set
DEFAULT_SPEND_CAP_USD = HARD_SPEND_CAP_USD
SPEND_ALERT_FRACTIONS = (0.50, 0.75, 0.90)
DEFAULT_LEDGER_PATH = Path("results/cloud_ledger.jsonl")

# model_id -> pricing, fetched 2026-09-30 from official vendor pages (see module docstring)
MODEL_PRICING: dict[str, dict[str, Any]] = {
    "gpt-4o-mini": {
        "tier": "cheap",
        "input_per_1m_usd": 0.15,
        "output_per_1m_usd": 0.60,
        "fetch_date": "2026-09-30",
        "source_url": "https://developers.openai.com/api/docs/pricing",
    },
    "claude-sonnet-4-5": {
        "tier": "strong",
        "input_per_1m_usd": 3.00,
        "output_per_1m_usd": 15.00,
        "fetch_date": "2026-09-30",
        "source_url": "https://claude.com/pricing",
    },
}

CLOUD_MODELS = {"cheap": "gpt-4o-mini", "strong": "claude-sonnet-4-5"}


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------


class HostRestrictionError(Exception):
    """Raised when a real cloud call is attempted from a disallowed host."""


class SpendCapExceeded(Exception):
    """Raised when a real call would push running spend past the cap."""


class UnknownModelError(Exception):
    """Raised when a model id has no entry in MODEL_PRICING."""


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
    """Project the USD cost of a call from published per-token pricing."""
    if model_id not in MODEL_PRICING:
        raise UnknownModelError(f"no pricing entry for model '{model_id}'; known models: {list(MODEL_PRICING)}")
    pricing = MODEL_PRICING[model_id]
    return (input_tokens / 1_000_000) * pricing["input_per_1m_usd"] + (
        output_tokens / 1_000_000
    ) * pricing["output_per_1m_usd"]


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


# --------------------------------------------------------------------------
# Default (litellm-backed) completion function -- import deferred
# --------------------------------------------------------------------------


def _default_completion_fn(*, model_id: str, messages: list[dict[str, Any]], api_key: str, **kwargs: Any) -> dict[str, Any]:
    """Real provider call via litellm. Import is deferred so stub mode and
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


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------


class CloudClient:
    """Cloud LLM access with a host restriction, stub fallback, spend cap,
    and a spend ledger. See the module docstring for the full contract.
    """

    def __init__(
        self,
        *,
        spend_cap_usd: float = DEFAULT_SPEND_CAP_USD,
        ledger_path: str | Path = DEFAULT_LEDGER_PATH,
        hostname: str | None = None,
        role_override: str | None = None,
        api_key: str | None = None,
        completion_fn: Callable[..., dict[str, Any]] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.spend_cap_usd = float(spend_cap_usd)
        self.ledger_path = Path(ledger_path)
        self._hostname = hostname
        self._role_override = role_override
        self._completion_fn = completion_fn or _default_completion_fn
        self._clock = clock or (lambda: datetime.now(timezone.utc))

        # api_key resolution: explicit arg wins; else env var; absence -> stub mode.
        self.api_key = api_key if api_key is not None else os.environ.get(ENV_VAR_API_KEY)
        self.stub_mode = not bool(self.api_key)
        if not self.stub_mode and self.spend_cap_usd > HARD_SPEND_CAP_USD + 1e-9:
            raise ValueError(
                f"spend_cap_usd ${self.spend_cap_usd:.2f} exceeds the hard cap ${HARD_SPEND_CAP_USD:.2f} total; "
                "a real-mode client may only lower the cap"
            )

        self._running_total_usd = self._read_running_total()
        self._fired_alerts = self._read_fired_alerts()

    # -- budget bookkeeping -------------------------------------------------

    def _read_running_total(self) -> float:
        """Sum cost_usd across every existing ledger row (real calls only)."""
        if not self.ledger_path.exists():
            return 0.0
        total = 0.0
        with self.ledger_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    if row.get("record") == "alert":
                        continue
                    total += float(row.get("cost_usd", 0.0))
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
        return total

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
                "ts_utc": ts,
                "timestamp": ts,
            }
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with self.ledger_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
            print(
                f"ALERT: cloud spend ${self._running_total_usd:.4f} reached {frac:.0%} of the "
                f"${self.spend_cap_usd:.2f} cap",
                flush=True,
            )
            self._fired_alerts.add(key)
            fired_now.append(row)
        return fired_now

    @property
    def running_total_usd(self) -> float:
        return self._running_total_usd

    @property
    def remaining_budget_usd(self) -> float:
        return max(0.0, self.spend_cap_usd - self._running_total_usd)

    def estimate_cost(self, model_id: str, input_tokens: int, output_tokens: int) -> float:
        return estimate_cost_usd(model_id, input_tokens, output_tokens)

    # -- ledger ---------------------------------------------------------------

    def _append_ledger(self, result: CallResult) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "timestamp": self._clock().isoformat(),
            "model_id": result.model_id,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "cost_usd": result.cost_usd,
            "running_total_usd": self._running_total_usd + result.cost_usd,
        }
        with self.ledger_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    # -- stub path --------------------------------------------------------------

    def _stub_call(self, model_id: str, messages: list[dict[str, Any]], expected_output_tokens: int) -> CallResult:
        input_tokens = _approx_token_count(messages)
        content = (
            f"[STUB RESPONSE for {model_id}] deterministic fake reply; "
            "no real API call was made; CLOUD_API_KEY is unset or empty."
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

    def call(
        self,
        *,
        model_id: str,
        messages: list[dict[str, Any]],
        expected_output_tokens: int = 256,
        **kwargs: Any,
    ) -> CallResult:
        """Make a (possibly stub) cloud call.

        In stub mode (no CLOUD_API_KEY): always succeeds, never touches the
        network or the ledger, never checks host restriction (stub mode is
        safe everywhere, including evo-t2s).

        In real mode: enforces the host restriction, then the spend cap,
        before making any request; appends one ledger row on success.
        """
        if self.stub_mode:
            return self._stub_call(model_id, messages, expected_output_tokens)

        check_host_allowed(self._hostname, self._role_override)

        if model_id not in MODEL_PRICING:
            raise UnknownModelError(f"no pricing entry for model '{model_id}'; known models: {list(MODEL_PRICING)}")

        input_tokens = _approx_token_count(messages)
        projected_cost = self.estimate_cost(model_id, input_tokens, expected_output_tokens)

        if self._running_total_usd + projected_cost > self.spend_cap_usd + 1e-9:
            reason = (
                f"refused: projected cost ${projected_cost:.4f} would bring running "
                f"total to ${self._running_total_usd + projected_cost:.4f}, exceeding "
                f"the ${self.spend_cap_usd:.2f} cap (current running total "
                f"${self._running_total_usd:.4f})"
            )
            raise SpendCapExceeded(reason)

        response_json = self._completion_fn(
            model_id=model_id,
            messages=messages,
            api_key=self.api_key,
            **kwargs,
        )
        usage = response_json.get("usage") or {}
        actual_input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or input_tokens)
        actual_output_tokens = int(
            usage.get("completion_tokens") or usage.get("output_tokens") or expected_output_tokens
        )
        actual_cost = self.estimate_cost(model_id, actual_input_tokens, actual_output_tokens)

        content = ""
        choices = response_json.get("choices") or []
        if choices:
            content = (choices[0].get("message") or {}).get("content", "") or ""

        result = CallResult(
            model_id=model_id,
            input_tokens=actual_input_tokens,
            output_tokens=actual_output_tokens,
            cost_usd=actual_cost,
            stub=False,
            content=content,
            raw=response_json,
        )
        self._append_ledger(result)
        self._running_total_usd += result.cost_usd
        self._fire_spend_alerts()
        return result


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
