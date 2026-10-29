"""Ski-rental keep-alive and TTL rules (pure functions). EXPLORATORY."""

from __future__ import annotations

from seam.duet.params import Price

PING_LEAD_S = 30.0  # ping ~30 s before each predicted expiry
TTL_5M_S = 300.0


def ping_threshold(p: Price) -> float:
    """Ping while p_return > r/(w - r)."""
    return p.read / (p.write_5m - p.read)


def max_pings(p: Price) -> float:
    """Stop after (w - r)/r pings: by then pinging has cost one full re-write."""
    return (p.write_5m - p.read) / p.read


def should_ping(p_return: float, pings_so_far: int, p: Price) -> bool:
    return p_return * (p.write_5m - p.read) > p.read and pings_so_far < max_pings(p)


def one_hour_breakeven_min(p: Price) -> float:
    """Idle minutes beyond which a 1h-TTL write beats pinging a 5m entry."""
    pings = (p.write_1h - p.write_5m) / p.read
    return pings * (TTL_5M_S - PING_LEAD_S) / 60.0


def compaction_r_star(rho: float, p: Price) -> float:
    """Compaction mid-segment pays only if remaining turns exceed rho/(1 - rho) * w/r."""
    return rho / (1.0 - rho) * p.write_5m / p.read
