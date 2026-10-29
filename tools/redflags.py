"""PLAN LOCK red flags R1-R10 (Task BO): PASS / RED FLAG / NOT YET MEASURED with run_ids.

Reads one closeout JSON (written by the Monday closeout, e.g. analysis/cc_20261026/redflag_inputs.json):
{"R1": {"value": 21040, "run_ids": [...]}, "R2": {"ratios": [1.71], "slider_r2": 0.99, ...}, ...}
A missing key is NOT YET MEASURED. Thresholds are the registered ones
(docs/PLAN_LOCK_2026-10-23.md section 6; CC_AMEND_2026-10-23c.md A19-A35;
CC_CONTEXT_CEILING_2026-10-23.md section 3.3; CC_AMEND_2026-10-23b.md A2).
Usage: python tools/redflags.py <closeout.json> [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

PASS, RED, NYM = "PASS", "RED FLAG", "NOT YET MEASURED"
R1_RANGE = (19_900, 22_600)
RATIO = 1.77
RATIO_TOL = 0.10
R2_MIN = 0.98
A_RANGE_KB = (70.0, 130.0)
SLOPE_DIFF = 0.10
A19_TOL_MIB = 256
GATE_U4_C256 = 40_000
REFUTE_U4_C256 = 29_000
R8_MEDIAN = 0.15
R8_BINDING = 0.90


def _r1(d: dict[str, Any]) -> tuple[str, str]:
    v = d["value"]
    ok = R1_RANGE[0] <= v <= R1_RANGE[1]
    return (PASS if ok else RED), f"8-bit held-out ceiling {v} vs {R1_RANGE[0]}-{R1_RANGE[1]}"


def _r2(d: dict[str, Any]) -> tuple[str, str]:
    lo, hi = RATIO * (1 - RATIO_TOL), RATIO * (1 + RATIO_TOL)
    bad = [r for r in d.get("ratios", []) if not lo <= r <= hi]
    r2 = d.get("slider_r2")
    if r2 is not None and r2 < R2_MIN:
        bad.append(f"R^2 {r2}")
    if not d.get("ratios") and r2 is None:
        return NYM, "no slope ratio yet"
    return (
        RED if bad else PASS
    ), f"ratios {d.get('ratios')} in [{lo:.3f}, {hi:.3f}], slider R^2 {r2}"


def _r3(d: dict[str, Any]) -> tuple[str, str]:
    f, u = d["a_kb"]["f16"], d["a_kb"]["u4"]
    out = [x for x in (f, u) if not A_RANGE_KB[0] <= x <= A_RANGE_KB[1]]
    diff = abs(f - u) / ((f + u) / 2)
    bad = bool(out) or diff > SLOPE_DIFF
    return (RED if bad else PASS), f"a f16 {f} / u4 {u} KB/token, diff {diff:.1%}"


def _r4(d: dict[str, Any]) -> tuple[str, str]:
    no_lift = [
        p for p, v in d.get("cap256_vs_uncapped", {}).items() if v["capped"] <= v["uncapped"]
    ]
    budget = [c for c in d.get("headline_failure_classes", []) if c == "budget_check"]
    bad = bool(no_lift) or bool(budget)
    return (
        RED if bad else PASS
    ), f"no lift: {no_lift or 'none'}; budget-check failures: {len(budget)}"


def _r5(d: dict[str, Any]) -> tuple[str, str]:
    mism = [x for x in d.get("kv", []) if x.get("headline") and x["applied"] != x["requested"]]
    mism += [x for x in d.get("ollama", []) if x["applied"] != x["requested"]]
    return (RED if mism else PASS), f"mismatches: {mism or 'none'}"


def r6_formula_mib(v: dict[str, Any]) -> float:
    """A19 registered prediction for one host (Task BP): the given formula_mib, else the Intel
    override (override_pct x ullTotalPhys, the registered "slider" clause) when recorded, else
    the Windows formula MIN(0.80 M, MAX(M - 16 GiB, 0.50 M)) on ullTotalPhys M (MiB)."""
    if "formula_mib" in v:
        return float(v["formula_mib"])
    m = float(v["total_phys_mib"])
    if v.get("override_pct") is not None:
        return float(v["override_pct"]) / 100.0 * m
    return min(0.80 * m, max(m - 16_384.0, 0.50 * m))


def _r6(d: dict[str, Any]) -> tuple[str, str]:
    bad = {
        h: v
        for h, v in d.items()
        if isinstance(v, dict) and abs(v["shared_mib"] - r6_formula_mib(v)) > A19_TOL_MIB
    }
    return (
        RED if bad else PASS
    ), f"hosts off the formula by > {A19_TOL_MIB} MiB: {sorted(bad) or 'none'}"


def _r7(d: dict[str, Any]) -> tuple[str, str]:
    v = d["u4_c256_median"]
    bad = v < GATE_U4_C256 or v <= REFUTE_U4_C256
    return (
        RED if bad else PASS
    ), f"u4 cap-256 median {v} (gate >= {GATE_U4_C256}; refuted <= {REFUTE_U4_C256})"


def _r8(d: dict[str, Any]) -> tuple[str, str]:
    m, b = d["median_abs_err"], d.get("binding_correct_frac")
    bad = m > R8_MEDIAN or (b is not None and b < R8_BINDING)
    return (
        RED if bad else PASS
    ), f"median abs err {m:.1%} (red > {R8_MEDIAN:.0%}); binding correct {b}; n {d.get('n')}"


def _list_flag(key: str) -> Callable[[dict[str, Any]], tuple[str, str]]:
    def f(d: dict[str, Any]) -> tuple[str, str]:
        hits = [x["item"] for x in d.get("items", []) if x.get(key)]
        return (RED if hits else PASS), f"{key}: {hits or 'none'}"

    return f


RULES: dict[str, tuple[str, Callable[[dict[str, Any]], tuple[str, str]]]] = {
    "R1": ("A21 8-bit held-out ceiling", _r1),
    "R2": ("A22/A25 slope ratio, slider R^2", _r2),
    "R3": ("A20 allocator a", _r3),
    "R4": ("A34 cap-256 lift / A32 failure class", _r4),
    "R5": ("A31/A35 applied KV precision", _r5),
    "R6": ("A19 Windows formula", _r6),
    "R7": ("CC-MECH gate / H-act refutation", _r7),
    "R8": ("held-out law error, binding ceiling", _r8),
    "R9": ("literature kill criteria", _list_flag("kills")),
    "R10": ("integrity of DEMO-2 primary and sealed numbers", _list_flag("failed")),
}


def evaluate(doc: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for rid, (what, fn) in RULES.items():
        d = doc.get(rid)
        if d is None:
            out.append({"id": rid, "what": what, "status": NYM, "detail": "", "run_ids": []})
            continue
        try:
            status, detail = fn(d)
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            status, detail = NYM, f"incomplete input ({type(exc).__name__}: {exc})"
        out.append(
            {
                "id": rid,
                "what": what,
                "status": status,
                "detail": detail,
                "run_ids": d.get("run_ids", []),
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("closeout", type=Path)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    res = evaluate(json.loads(a.closeout.read_text(encoding="utf-8")))
    if a.json:
        print(json.dumps(res, indent=1))
    else:
        for r in res:
            print(
                f"{r['id']:>3}  {r['status']:<16}  {r['what']}: {r['detail']}  run_ids={','.join(r['run_ids']) or '-'}"
            )
    return 1 if any(r["status"] == RED for r in res) else 0


if __name__ == "__main__":
    sys.exit(main())
