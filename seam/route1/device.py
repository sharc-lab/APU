"""Device law (BR constants) and a PLACEHOLDER TTFT model.

L_unchunked = B/(k+a); L_chunked = (B - a*c)/k; L_session_total(T, c) = (B - a*min(T, c))/k.
Every TTFT value here is PLACEHOLDER until the Nov 22 params freeze.
"""

from __future__ import annotations

from dataclasses import dataclass, field

K_BYTES: dict[str, int] = {"f16": 147_456, "i8": 84_096, "u4": 47_232}  # MEASURED (compiled k)
A_BYTES = 87_520  # MEASURED (BR, e4e797e7)
B_BR = 3_267_086_560  # DERIVED (median of 14 in-sample 57% cells)
WINDOW = 40_960  # Qwen3-4B model card
WINDOW_HEADROOM = 512
MARGIN_M = 0.059  # session-law upper error (team-reported band -2.5% / +5.9%)


@dataclass(frozen=True)
class Constants:
    B: float = B_BR
    a: int = A_BYTES
    k: dict[str, int] = field(default_factory=lambda: dict(K_BYTES))
    W: int = WINDOW


def l_unchunked(c: Constants, prec: str) -> float:
    return c.B / (c.k[prec] + c.a)


def l_chunked(c: Constants, prec: str, chunk: int) -> float:
    return (c.B - c.a * chunk) / c.k[prec]


def l_kv_only(c: Constants, prec: str) -> float:
    return c.B / c.k[prec]


def l_session_total(c: Constants, prec: str, t_new: int, chunk: int | None) -> float:
    step = t_new if chunk is None else min(t_new, chunk)
    return (c.B - c.a * step) / c.k[prec]


@dataclass(frozen=True)
class Config:
    name: str
    prec: str
    chunk: int | None
    resident: bool
    passed_xq: bool = True


I8_DEFAULT = Config("i8-default", "i8", None, False)
I8_RESIDENT = Config("i8+resident", "i8", None, True)
I8_C256_RESIDENT = Config("i8-c256+resident", "i8", 256, True)
U4_C256_RESIDENT = Config("u4-c256+resident", "u4", 256, True)
CONFIG_SET: tuple[Config, ...] = (I8_DEFAULT, I8_RESIDENT, I8_C256_RESIDENT, U4_C256_RESIDENT)

# PLACEHOLDER TTFT parameters (s). Cold edge chosen so the i8 default reaches 10 s at 10,000 tokens
# (MEASURED edge c2246b1f); resident turn TTFT is the ASSUMED 2-5 s midpoint (M2b not measured).
TTFT_PARAMS: dict[str, float] = {
    "cold_s_per_token": 1.0e-3,
    "resident_turn_s": 3.5,
    "time_to_fail_s": 0.442,  # MEASURED 00c7fa60 (turn-1 OOM)
    "decode_tok_s": 20.0,  # PLACEHOLDER
    "cloud_ttft_s": 1.0,  # PLACEHOLDER
    "cloud_tok_s": 60.0,  # PLACEHOLDER
    "package_w": 28.0,  # ASSUMED
}


def ttft_s(cfg: Config, prompt: int, new: int, first_turn: bool, p: dict[str, float]) -> float:
    """PLACEHOLDER: cold prefill scales with the prompt; a resident turn is a constant (new is unused
    until M2b measures TTFT(n_new | n_ctx))."""
    del new
    if first_turn or not cfg.resident:
        return prompt * p["cold_s_per_token"]
    return p["resident_turn_s"]


def true_ceiling(
    c: Constants, cfg: Config, new: int, default_measured: int | None = 19_750
) -> float:
    """Memory truth used by the simulator for a local call (no margin).

    The i8 default uses the MEASURED ceiling (19,750, b038e362) when given."""
    if not cfg.resident:
        if cfg.chunk is None:
            if cfg is I8_DEFAULT and default_measured is not None:
                return float(default_measured)
            return l_unchunked(c, cfg.prec)
        return l_chunked(c, cfg.prec, cfg.chunk)
    return l_session_total(c, cfg.prec, new, cfg.chunk)
