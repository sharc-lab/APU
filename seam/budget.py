"""Spend ledger and hard budget ceiling for paid backends.

The ceiling must be **structurally impossible** to exceed, not merely discouraged. That rules out
the obvious design - call the API, add up what it cost, warn when the total gets large - because
by the time the total is known the money is already gone.

The guarantee here comes from three properties:

1. **Every paid call is bounded before it is issued.** A request carries ``max_tokens``, so its
   worst-case cost is computable exactly. :meth:`BudgetGuard.authorize_call` refuses unless the
   remaining budget covers that worst case. No call can therefore overshoot, even if the model
   generates the longest reply it is permitted to.
2. **The ledger is durable and append-only.** Spend is reloaded from disk on construction, so a
   crashed or restarted process resumes against real cumulative spend rather than zero. A
   forgotten ledger is the classic way a "hard" ceiling turns out to be per-process.
3. **Actual cost comes from the provider's own usage fields**, never from a local token estimate.
   Local estimates drive the pre-flight *projection* only, and projected-vs-actual is recorded on
   every call so the projection self-calibrates and its error is reportable.

Refusal is a data point (spec §3.7, AF-006). The guard raises
:class:`~seam.errors.BudgetExceededError`; the caller emits a refusal manifest and stops.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Final, Literal

import yaml

from seam.errors import BudgetExceededError, ConfigError
from seam.jsonlog import log_event

__all__ = [
    "BudgetGuard",
    "BudgetLedger",
    "CacheState",
    "CallUsage",
    "PricingTable",
    "PricingTier",
    "usd_cost",
]

#: Prompt-cache state for *estimate* projections. Authorization always uses the cold/write bound.
CacheState = Literal["cold", "warm", "unknown"]

#: Tokens per pricing unit. Rates in ``configs/pricing/`` are quoted per million tokens.
_MTOK: Final = 1_000_000

#: Cost below which a projection is treated as zero for logging purposes. Well under a cent, so it
#: cannot mask a real charge; exists only to keep float noise out of the calibration record.
_NEGLIGIBLE_USD: Final = 1e-9


# ==================================================================================================
# Pricing
# ==================================================================================================


@dataclass(frozen=True, slots=True)
class PricingTier:
    """One dated pricing regime for one model.

    ``effective_through`` is inclusive and may be ``None`` for the open-ended current tier.
    """

    label: str
    effective_from: date
    effective_through: date | None
    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float
    cache_write_5m_per_mtok: float
    cache_write_1h_per_mtok: float

    def covers(self, when: date) -> bool:
        if when < self.effective_from:
            return False
        return self.effective_through is None or when <= self.effective_through


@dataclass(frozen=True, slots=True)
class PricingTable:
    """Dated pricing for one provider, loaded from ``configs/pricing/<provider>.yaml``."""

    version: str
    retrieved_utc: str
    source: str
    currency: str
    tiers_by_model: dict[str, tuple[PricingTier, ...]]
    pin_convention_by_model: dict[str, str]
    tool_use_overhead_tokens: dict[str, int]

    @classmethod
    def load(cls, path: Path) -> PricingTable:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ConfigError(f"pricing table {path} is not a mapping")

        tiers_by_model: dict[str, tuple[PricingTier, ...]] = {}
        pins: dict[str, str] = {}
        for model, spec in (raw.get("models") or {}).items():
            pins[model] = str(spec.get("pin_convention", "unspecified"))
            parsed: list[PricingTier] = []
            for tier in spec.get("tiers") or []:
                through = tier.get("effective_through")
                parsed.append(
                    PricingTier(
                        label=str(tier["label"]),
                        effective_from=_as_date(tier["effective_from"]),
                        effective_through=_as_date(through) if through else None,
                        input_per_mtok=float(tier["input_per_mtok"]),
                        output_per_mtok=float(tier["output_per_mtok"]),
                        cache_read_per_mtok=float(tier["cache_read_per_mtok"]),
                        cache_write_5m_per_mtok=float(tier["cache_write_5m_per_mtok"]),
                        cache_write_1h_per_mtok=float(tier["cache_write_1h_per_mtok"]),
                    )
                )
            if not parsed:
                raise ConfigError(f"pricing table {path}: model {model!r} declares no tiers")
            tiers_by_model[model] = tuple(sorted(parsed, key=lambda t: t.effective_from))

        if not tiers_by_model:
            raise ConfigError(f"pricing table {path} declares no models")

        return cls(
            version=str(raw["pricing_version"]),
            retrieved_utc=str(raw["pricing_retrieved_utc"]),
            source=str(raw.get("pricing_source", "")),
            currency=str(raw.get("currency", "USD")),
            tiers_by_model=tiers_by_model,
            pin_convention_by_model=pins,
            tool_use_overhead_tokens={
                str(k): int(v) for k, v in (raw.get("tool_use_overhead_tokens") or {}).items()
            },
        )

    def tier_for(self, model: str, when: date) -> PricingTier:
        """Return the tier in force for ``model`` on ``when``.

        A date outside every declared tier is an error, not a fallback to the nearest rate: the
        2026-09-01 increase is exactly the case where silently reusing a stale rate would
        under-report cost.
        """
        tiers = self.tiers_by_model.get(model)
        if tiers is None:
            raise ConfigError(
                f"no pricing declared for model {model!r} (have: "
                f"{sorted(self.tiers_by_model)}). Refusing to guess a rate."
            )
        for tier in tiers:
            if tier.covers(when):
                return tier
        raise ConfigError(
            f"pricing for {model!r} has no tier covering {when.isoformat()}; declared tiers are "
            f"{[(t.label, t.effective_from.isoformat()) for t in tiers]}. Update "
            f"configs/pricing/ rather than extrapolating a rate."
        )


def _as_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


# ==================================================================================================
# Usage and cost
# ==================================================================================================


@dataclass(frozen=True, slots=True)
class CallUsage:
    """Token usage for a single paid call, taken from the provider's response.

    Field names mirror Anthropic's ``usage`` block. ``cache_creation_input_tokens`` is billed at a
    write rate and ``cache_read_input_tokens`` at the (much cheaper) read rate; conflating them
    with plain input tokens misprices cached runs by roughly an order of magnitude in the read
    direction.
    """

    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    #: Cache writes are billed by TTL. The slice uses the 5-minute cache, which pays for itself
    #: after a single read.
    cache_write_ttl: str = "5m"


def usd_cost(usage: CallUsage, tier: PricingTier) -> float:
    """Exact USD cost of one call under ``tier``."""
    write_rate = (
        tier.cache_write_1h_per_mtok
        if usage.cache_write_ttl == "1h"
        else tier.cache_write_5m_per_mtok
    )
    return (
        usage.input_tokens * tier.input_per_mtok
        + usage.output_tokens * tier.output_per_mtok
        + usage.cache_read_input_tokens * tier.cache_read_per_mtok
        + usage.cache_creation_input_tokens * write_rate
    ) / _MTOK


# ==================================================================================================
# Ledger
# ==================================================================================================


@dataclass(slots=True)
class LedgerEntry:
    """One recorded paid call."""

    ts_utc: str
    phase: str
    run_id: str | None
    model: str
    pricing_version: str
    tier_label: str
    projected_usd: float
    actual_usd: float
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int


class BudgetLedger:
    """Durable append-only record of paid calls.

    Persisted outside ``raw/``: ``raw/`` is write-once and sealed per run, whereas the ledger must
    stay writable across every run in the slice. It lives in ``derived/budget/`` instead, and is
    the authority on cumulative spend after a restart.
    """

    def __init__(self, root: Path) -> None:
        self._dir = root
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / "ledger.ndjson"
        self._lock = threading.Lock()
        self._entries: list[LedgerEntry] = []
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            self._entries.append(LedgerEntry(**json.loads(line)))

    @property
    def path(self) -> Path:
        return self._path

    @property
    def entries(self) -> tuple[LedgerEntry, ...]:
        return tuple(self._entries)

    def total_usd(self, *, phase: str | None = None) -> float:
        return sum(e.actual_usd for e in self._entries if phase is None or e.phase == phase)

    def append(self, entry: LedgerEntry) -> None:
        """Append and **fsync** before returning.

        The flush is not incidental. If the process dies between issuing a call and durably
        recording it, the money is spent but the ledger under-reports, and the next process
        starts with a ceiling that is too generous by exactly the unrecorded amount.
        """
        with self._lock:
            self._entries.append(entry)
            with self._path.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(asdict(entry), separators=(",", ":")) + "\n")
                fh.flush()
                os.fsync(fh.fileno())

    def calibration(self) -> dict[str, Any]:
        """Projected-vs-actual error over recorded calls.

        Reported at every checkpoint. A projection that is systematically low is the mechanism by
        which a budget is overrun despite a guard, so its error is tracked as a first-class number
        rather than assumed small.
        """
        priced = [e for e in self._entries if e.projected_usd > _NEGLIGIBLE_USD]
        if not priced:
            return {"n_calls": 0, "mean_rel_error": None, "total_projected_usd": 0.0}
        rel = [(e.actual_usd - e.projected_usd) / e.projected_usd for e in priced]
        return {
            "n_calls": len(priced),
            "total_projected_usd": sum(e.projected_usd for e in priced),
            "total_actual_usd": sum(e.actual_usd for e in priced),
            "mean_rel_error": sum(rel) / len(rel),
            "max_rel_error": max(rel),
            "min_rel_error": min(rel),
        }


# ==================================================================================================
# Guard
# ==================================================================================================


@dataclass(slots=True)
class BudgetGuard:
    """Hard ceiling over a :class:`BudgetLedger`.

    Two ceilings apply simultaneously and the tighter one wins: a per-slice ceiling and a project
    total. Both are checked against durable cumulative spend.
    """

    ledger: BudgetLedger
    pricing: PricingTable
    slice_ceiling_usd: float
    project_ceiling_usd: float
    phase: str = "unassigned"
    #: Set once a refusal has fired, so a caller cannot loop and retry its way past the ceiling.
    tripped: bool = field(default=False, init=False)

    def spent_usd(self) -> float:
        return self.ledger.total_usd()

    def remaining_usd(self) -> float:
        return max(
            0.0,
            min(
                self.slice_ceiling_usd - self.spent_usd(),
                self.project_ceiling_usd - self.spent_usd(),
            ),
        )

    def worst_case_call_usd(
        self, *, model: str, prompt_tokens: int, max_tokens: int, when: date | None = None
    ) -> float:
        """Upper bound on one call's cost.

        Assumes **no** cache hit and a full-length reply, and charges the whole prompt at the cache
        *write* rate, which is the most expensive way the prompt can be billed. Bounding rather
        than estimating is what lets the ceiling be a guarantee.
        """
        return self.projected_call_usd(
            model=model,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            cache_state="cold",
            when=when,
        )

    def projected_call_usd(
        self,
        *,
        model: str,
        prompt_tokens: int,
        max_tokens: int,
        cache_state: CacheState = "unknown",
        uncached_suffix_tokens: int = 0,
        when: date | None = None,
    ) -> float:
        """Estimate one call's cost given known or unknown prompt-cache state.

        * ``cold`` / ``unknown``: charge the prompt at the 5m cache-write rate (worst case).
          ``unknown`` is treated as cold so an uninformed estimate cannot under-project.
        * ``warm``: charge ``prompt_tokens - uncached_suffix_tokens`` at the cache-read rate and
          the suffix at the ordinary input rate. This is the Phase E 0.3 gap: without cache
          state, warm calls over-projected by ~67% because every estimate assumed a write.

        Authorization (:meth:`authorize_call`) always uses :meth:`worst_case_call_usd` (= cold).
        Estimates used for projected-vs-actual calibration should pass the real cache state.
        """
        tier = self.pricing.tier_for(model, when or _today_utc())
        if cache_state == "warm":
            suffix = max(0, min(uncached_suffix_tokens, prompt_tokens))
            cached = max(0, prompt_tokens - suffix)
            prompt_cost = (cached * tier.cache_read_per_mtok + suffix * tier.input_per_mtok) / _MTOK
        else:
            # cold or unknown: write-rate bound
            prompt_cost = (
                prompt_tokens * max(tier.input_per_mtok, tier.cache_write_5m_per_mtok)
            ) / _MTOK
        return prompt_cost + (max_tokens * tier.output_per_mtok) / _MTOK

    def authorize_call(
        self, *, model: str, prompt_tokens: int, max_tokens: int, when: date | None = None
    ) -> float:
        """Refuse unless the remaining budget covers this call's **worst case**.

        Returns the worst-case bound so the caller can log projected-vs-actual.
        """
        bound = self.worst_case_call_usd(
            model=model, prompt_tokens=prompt_tokens, max_tokens=max_tokens, when=when
        )
        remaining = self.remaining_usd()
        if self.tripped or bound > remaining:
            self.tripped = True
            log_event(
                "budget.call_refused",
                severity="error",
                message=(
                    f"refusing paid call: worst case ${bound:.6f} exceeds remaining "
                    f"${remaining:.6f}"
                ),
                phase=self.phase,
                model=model,
                worst_case_usd=bound,
                remaining_usd=remaining,
                spent_usd=self.spent_usd(),
                slice_ceiling_usd=self.slice_ceiling_usd,
                project_ceiling_usd=self.project_ceiling_usd,
            )
            raise BudgetExceededError(
                f"refusing paid call in phase {self.phase!r}: worst-case cost ${bound:.4f} "
                f"exceeds remaining budget ${remaining:.4f} (spent ${self.spent_usd():.4f} of "
                f"slice ceiling ${self.slice_ceiling_usd:.2f} / project ceiling "
                f"${self.project_ceiling_usd:.2f}). Emit a refusal manifest, then stop. The "
                f"guard is not to be raised to complete a run - an unaffordable run is a finding."
            )
        return bound

    def authorize_run(self, *, projected_usd: float, description: str) -> None:
        """Refuse to START a run whose projection exceeds the remaining budget."""
        remaining = self.remaining_usd()
        if projected_usd > remaining:
            log_event(
                "budget.run_refused",
                severity="error",
                message=(
                    f"refusing to start {description}: projected ${projected_usd:.4f} exceeds "
                    f"remaining ${remaining:.4f}"
                ),
                phase=self.phase,
                projected_usd=projected_usd,
                remaining_usd=remaining,
                spent_usd=self.spent_usd(),
            )
            raise BudgetExceededError(
                f"refusing to start {description}: projected cost ${projected_usd:.4f} exceeds "
                f"remaining budget ${remaining:.4f} (spent ${self.spent_usd():.4f}). This is a "
                f"finding, not an obstacle - record the refusal and report it."
            )
        log_event(
            "budget.run_authorized",
            message=f"authorized {description}: projected ${projected_usd:.4f}",
            phase=self.phase,
            projected_usd=projected_usd,
            remaining_usd=remaining,
        )

    def record(
        self,
        *,
        model: str,
        usage: CallUsage,
        projected_usd: float,
        run_id: str | None,
        when: date | None = None,
    ) -> float:
        """Price a completed call from the provider's usage fields and commit it to the ledger."""
        day = when or _today_utc()
        tier = self.pricing.tier_for(model, day)
        actual = usd_cost(usage, tier)
        self.ledger.append(
            LedgerEntry(
                ts_utc=datetime.now(UTC).isoformat(),
                phase=self.phase,
                run_id=run_id,
                model=model,
                pricing_version=self.pricing.version,
                tier_label=tier.label,
                projected_usd=projected_usd,
                actual_usd=actual,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_input_tokens=usage.cache_read_input_tokens,
                cache_creation_input_tokens=usage.cache_creation_input_tokens,
            )
        )
        return actual

    def checkpoint(self) -> dict[str, Any]:
        """Spend summary for a reporting checkpoint."""
        return {
            "phase": self.phase,
            "spent_usd": self.spent_usd(),
            "remaining_usd": self.remaining_usd(),
            "slice_ceiling_usd": self.slice_ceiling_usd,
            "project_ceiling_usd": self.project_ceiling_usd,
            "pricing_version": self.pricing.version,
            "pricing_retrieved_utc": self.pricing.retrieved_utc,
            "n_calls": len(self.ledger.entries),
            "calibration": self.ledger.calibration(),
            "by_phase": {
                p: self.ledger.total_usd(phase=p)
                for p in sorted({e.phase for e in self.ledger.entries})
            },
        }


def _today_utc() -> date:
    return datetime.now(UTC).date()
