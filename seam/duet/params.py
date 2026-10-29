"""DUET constants with labels. PLACEHOLDER until the Nov 22 params freeze.

kappa unit: resident-turn kappa is in MICROSECONDS per token^2 (s = kappa * n_new * n_ctx), not ns.
"""

from __future__ import annotations

from dataclasses import dataclass

PLACEHOLDER = True

B = 3_267_086_560  # DERIVED: median of 14 in-sample 57% cells (B_BR)
A = {"f16": 87_520, "i8": 87_520, "u4": 93_280}  # MEASURED (BR; u4 per the DUET report)
K = {"f16": 147_456, "i8": 84_096, "u4": 47_232}  # MEASURED (compiled k)
W = 40_960  # model card
WINDOW_HEADROOM = 512
SLO_S = 10.0  # INTERACTIVE class
MARGIN_M = 0.059  # session-law upper error (team-reported)

# Resident extend: TTFT_ext(n, P) = P*(ALPHA + BETA*n)*(P/P_REF)**GAMMA. DERIVED (fit to six anchors,
# conservative +7..+22%): 183@16,384 INV-1; 1K@16,561 5.97 s; 2K@16,473 11.2 s (d7e8b43e);
# 4K@16,429 20.4 s; 1K@33,121 13.0 s; 2K@61,771 53.1 s. i8 uses the u4 fit (ASSUMED) until D-M2.
ALPHA = 0.079e-3  # s per cached token
BETA = 0.339e-6  # s per (new x cached) token pair = 0.339 us/token^2
GAMMA = 0.15
P_REF = 16_500
# Cold prefill: C0 + C1*P + C2*P^2 s. DERIVED (d42040c1, 69f56f0c, INV-1), max error 3.2%.
COLD_C0 = 0.28
COLD_C1 = 0.276e-3
COLD_C2 = 66.7e-9

# Per-cell kappa (us/token^2), DERIVED from the cells' per-turn prefill_s.
KAPPA_US = {
    "u4": {"0c7ae0a3": 0.462, "d7e8b43e": 0.428, "10cb79e4": 0.407},
    "f16": {"e99bb2ad": 0.450, "71b7d9d8": 0.335, "9cb921fd": 0.281},
    "u4_c256": {"08e52da9": 0.578},
    "f16_c256": {"2bfae783": 0.808},
}

STUB_TOKENS = 60  # EXCURSION-STUB payload (EXPLORATORY)
FILTER_KEEP = 0.25  # client tool-output filter keep ratio (EXPLORATORY)
COMPACT_KEEP = 0.25  # compaction at handoff keep ratio (EXPLORATORY)
SMALL_TURN = 700  # next small turn assumed by the excursion check (H shape)


@dataclass(frozen=True)
class Price:
    name: str
    input: float  # $ per M tokens
    write_5m: float
    read: float
    output: float
    write_1h: float


SONNET = Price("sonnet-5.5", 2.0, 2.50, 0.10, 10.0, 4.0)
HAIKU = Price("haiku-5.5", 0.10, 0.125, 0.01, 0.50, 0.20)
OPUS = Price("opus-5.5", 4.0, 5.0, 0.20, 20.0, 8.0)
EXECUTOR = HAIKU  # compaction executor (priced at Haiku input/output)
