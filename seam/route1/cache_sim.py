"""Prompt-cache ledger under the registered Anthropic rules.

Rules: 5-min TTL from request start, refreshed on hit; exact prefix; 20-block lookback; tools change
invalidates; prompts under 512 tokens are not cached; a model or tier switch is cold (ASSUMED).
"""

from __future__ import annotations

from dataclasses import dataclass

from seam.route1.prices import PriceSheet

TTL_S = 300.0
MIN_CACHE_TOKENS = 512
LOOKBACK_BLOCKS = 20


@dataclass(frozen=True)
class LedgerRow:
    uncached: int
    cache_write: int
    cache_read: int
    output: int
    usd: float


class CloudCache:
    """One provider-side cache for a session's transcript."""

    def __init__(self) -> None:
        self.cached = 0
        self.model = ""
        self.tools_hash = ""
        self.expiry_s = -1.0

    def warm(self, t_start_s: float, model: str, tools_hash: str, new_blocks: int) -> bool:
        return (
            self.cached > 0
            and model == self.model
            and tools_hash == self.tools_hash
            and t_start_s <= self.expiry_s
            and new_blocks <= LOOKBACK_BLOCKS
        )

    def call(
        self,
        sheet: PriceSheet,
        prompt: int,
        output: int,
        t_start_s: float,
        tools_hash: str = "tools-v1",
        new_blocks: int = 1,
    ) -> LedgerRow:
        mult = sheet.multiplier(prompt)
        if prompt < MIN_CACHE_TOKENS:
            usd = mult * (prompt * sheet.input + output * sheet.output) / 1e6
            return LedgerRow(prompt, 0, 0, output, usd)
        read = (
            min(self.cached, prompt)
            if self.warm(t_start_s, sheet.name, tools_hash, new_blocks)
            else 0
        )
        write = prompt - read
        usd = mult * (read * sheet.read + write * sheet.write_5m + output * sheet.output) / 1e6
        self.cached = prompt
        self.model = sheet.name
        self.tools_hash = tools_hash
        self.expiry_s = t_start_s + TTL_S
        return LedgerRow(0, write, read, output, usd)

    def invalidate(self) -> None:
        self.cached = 0
