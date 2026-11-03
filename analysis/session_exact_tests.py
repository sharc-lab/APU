"""Session-level exact tests for cells whose bootstrap interval collapses to a point.

The excess-over-default rows (analysis/r2_real_report.py::excess_over_default) put a 95% percentile bootstrap CI over
sessions on each tier's after-window gated silent failure rate minus the default tier's. When both cells are 0/n or
n/n (for example 5/5 against 0/5), every resample reproduces the point estimate, so the interval is a point such as
"+1.000 to +1.000" and says nothing about uncertainty. For those cells this module reports a one-sided Fisher exact
test on sessions as the unit instead: the 2x2 table (tier vs default) x (failed after the window vs did not), with the
alternative that the tier fails more often than the default. With k/n against 0/m and k = n, the p-value is
1 / C(n + m, n): 1/252 = 0.00397 for 5/5 against 0/5.

Pure functions only (no file reads), so tests can exercise them on fixtures; analysis/numbers_register.py feeds them
the real reports (row session-exact-tests).
"""
from __future__ import annotations

from math import comb

TEST_NAME = "one-sided Fisher exact test (unit session; alternative: tier rate > default rate)"


def fisher_one_sided_greater(a: int, n1: int, b: int, n0: int) -> float:
    """P(X >= a) for X hypergeometric: a events among n1 tier sessions, b among n0 default sessions, margins fixed."""
    if not (0 <= a <= n1 and 0 <= b <= n0) or n1 + n0 == 0:
        raise ValueError(f"bad counts {a}/{n1} vs {b}/{n0}")
    k, n = a + b, n1 + n0
    total = comb(n, k)
    return sum(comb(n1, x) * comb(n0, k - x) for x in range(a, min(k, n1) + 1)) / total


def is_degenerate(a: int, n1: int, b: int, n0: int) -> bool:
    """True when both cells are 0/n or n/n, so a session bootstrap of the difference has no variance."""
    return a in (0, n1) and b in (0, n0)


def cell_counts(rep: dict) -> dict:
    """{(tier, model): (sessions failing silently after the window, sessions)} from an r2_real_report report."""
    out = {}
    for key, c in rep["table"].items():
        tier, model = key.split("|")
        s = c["sessions"]
        out[(tier, model)] = (sum(1 for e in s if e["after_window_exceeded"]), len(s))
    return out


def exact_tests(rep: dict, default_tier: str = "ollama_default", degenerate_only: bool = True) -> list[dict]:
    """Per tier x model (default tier excluded): counts against the same model's default cell and the one-sided
    Fisher p. With degenerate_only, only cells whose bootstrap CI is a point and whose excess is nonzero."""
    cells = cell_counts(rep)
    out = []
    for (tier, model), (a, n1) in cells.items():
        if tier == default_tier or (default_tier, model) not in cells or n1 == 0:
            continue
        b, n0 = cells[(default_tier, model)]
        if n0 == 0:
            continue
        if degenerate_only and not (is_degenerate(a, n1, b, n0) and a * n0 != b * n1):
            continue
        out.append({"tier": tier, "model": model, "a": a, "n1": n1, "b": b, "n0": n0,
                    "excess": a / n1 - b / n0, "p": fisher_one_sided_greater(a, n1, b, n0)})
    return sorted(out, key=lambda d: (d["model"], d["tier"]))


def format_tests(label: str, tests: list[dict]) -> str:
    if not tests:
        return f"{label}: no degenerate cell"
    return f"{label}: " + ", ".join(
        f"{d['model']} {d['tier']} {d['a']}/{d['n1']} vs default {d['b']}/{d['n0']} sessions, excess "
        f"{d['excess']:+.3f}, p = {d['p']:.3g}" for d in tests)
