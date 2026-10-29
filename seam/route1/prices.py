"""Price sheets as dated data ($ per million tokens)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PriceSheet:
    name: str
    input: float
    write_5m: float
    read: float
    output: float
    date: str
    source: str
    long_ctx_threshold: int | None = None  # whole request reprices above this many input tokens
    long_ctx_mult: float = 1.0

    def multiplier(self, prompt_tokens: int) -> float:
        if self.long_ctx_threshold is not None and prompt_tokens > self.long_ctx_threshold:
            return self.long_ctx_mult
        return 1.0


_HYB1 = "docs/registrations/BR_HYB1_2026-10-26.md (price sheet dated 2026-10-26)"

SONNET = PriceSheet("sonnet-5.5", 2.0, 2.50, 0.10, 10.0, "2026-10-26", _HYB1)
HAIKU = PriceSheet("haiku-5.5", 0.10, 0.125, 0.01, 0.50, "2026-10-26", _HYB1, 100_000, 5.0)
OPUS = PriceSheet("opus-5.5", 4.0, 5.0, 0.20, 20.0, "2026-10-26", _HYB1)
FLASH_LITE = PriceSheet(
    "gemini-3.1-flash-lite",
    0.25,
    0.25,  # cache write = input (ASSUMED)
    0.025,
    1.50,
    "2026-10-29",
    "https://ai.google.dev/gemini-api/docs/pricing (as carried in the ROUTE-1 report)",
)

SHEETS: dict[str, PriceSheet] = {s.name: s for s in (SONNET, HAIKU, OPUS, FLASH_LITE)}


def reingest_penalty(sheet: PriceSheet, tokens: int) -> float:
    """$ of one cold re-ingest vs a warm read of the same prefix: C*(w - r)."""
    return tokens * (sheet.write_5m - sheet.read) / 1e6
