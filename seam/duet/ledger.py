"""Cloud ledger: reuses seam.route1.cache_sim (read = min(cached, prompt), rest written at the 5m price;
migration invalidates). Adds Haiku x5 above 100K and the compaction executor cost."""

from __future__ import annotations

from seam.duet import params as prm
from seam.route1.cache_sim import CloudCache
from seam.route1.prices import PriceSheet


def sheet(p: prm.Price) -> PriceSheet:
    long_ctx = (100_000, 5.0) if p.name.startswith("haiku") else (None, 1.0)
    return PriceSheet(p.name, p.input, p.write_5m, p.read, p.output, "2026-10-26", "HYB-1 sheet",
                      long_ctx[0], long_ctx[1])  # fmt: skip


class Ledger:
    def __init__(self, price: prm.Price) -> None:
        self.sheet = sheet(price)
        self.cache = CloudCache()
        self.usd = 0.0
        self.calls = 0

    def call(self, prompt: int, out: int) -> float:
        row = self.cache.call(self.sheet, prompt, out, 0.0)
        self.usd += row.usd
        self.calls += 1
        return row.usd

    def reset(self) -> None:
        self.cache.invalidate()

    def compact(self, pc_old: int, keep: float = prm.COMPACT_KEEP) -> int:
        pc_new = int(keep * pc_old)
        self.usd += (pc_old * prm.EXECUTOR.input + pc_new * prm.EXECUTOR.output) / 1e6
        return pc_new
