"""T2S f16 control band at n=18687.

Pass: the control median prefill_s lies within the 5c714535 median times
(1 +/- tol), where tol = b_t2s * (resolution / n_control). b_t2s is the
sealed T2S prefill exponent from 65e33de8, the gpu_only_f16 fit when
present, otherwise the mean of the three KV fits. That tolerance is the
relative prefill drift that shifts a bisected limit by one 250-token step
under prefill_s = C * n^b.

The within-session repeat window is recorded on the verdict and is not a gate.
The cell estimate is 3 * the 5c714535 median plus the c2246b1f canary
overhead, rounded up to the next whole second.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

REFERENCE_RUN_ID = "5c714535-9f36-4614-a594-698b6cd09296"
EXPONENT_RUN_ID = "65e33de8-ac07-405a-a1f8-53698974afe9"
CONTROL_N = 18687
CONTROL_REPEATS = 3
RESOLUTION = 250
MATCHING_ARM = "gpu_only_f16"
CANARY_OVERHEAD_S = 752.1359013
FORMULA = (
    "tol = b_t2s * (resolution / n_control); "
    "low = reference_median * (1 - tol); "
    "high = reference_median * (1 + tol); "
    "pass iff the control median prefill_s is inside [low, high]"
)
INTERPRETATION = (
    "drift small enough that it cannot move a bisected limit by more than one 250-token step"
)


def _uncold_reason() -> str:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from seam.measurement_gates import UNCOLD_UPTIME_REASON

    return UNCOLD_UPTIME_REASON


def noreboot_deviation() -> dict[str, Any] | None:
    """Deviation stamped into plan and summary when the sequencer sets the env."""
    if os.environ.get("SEAM_NOREBOOT_DEVIATION") != "1":
        return None
    raw = os.environ.get("SEAM_NOREBOOT_UPTIME_S", "").strip()
    uptime: float | None = float(raw) if raw else None
    return {
        "kind": "UNCOLD_UPTIME",
        "uptime_s": uptime,
        "reason": _uncold_reason(),
    }


def _prefill_s(doc: dict[str, Any]) -> float | None:
    generation = doc.get("generation")
    if isinstance(generation, dict) and generation.get("prefill_s") is not None:
        return float(generation["prefill_s"])
    if doc.get("prefill_s") is not None:
        return float(doc["prefill_s"])
    return None


def _result_path(work: Path, repeat: int, n: int = CONTROL_N) -> Path | None:
    name = f"gpu_only_f16.n{n}.r{repeat}.a0.result.json"
    direct = work / name
    if direct.is_file():
        return direct
    matches = sorted(p for p in work.rglob(name) if p.is_file())
    if len(matches) == 1:
        return matches[0]
    return None


def load_prefills(work: Path, n: int = CONTROL_N) -> list[float]:
    """Repeat-order prefill_s. Short when a repeat file or value is missing."""
    values: list[float] = []
    for repeat in range(CONTROL_REPEATS):
        path = _result_path(work, repeat, n)
        if path is None:
            return values
        value = _prefill_s(json.loads(path.read_text(encoding="utf-8")))
        if value is None:
            return values
        values.append(value)
    return values


def reference_work(repo_root: Path) -> Path:
    return repo_root / "derived" / "c2_ttft" / REFERENCE_RUN_ID / "work"


def within_session_window(prefills: list[float]) -> dict[str, Any]:
    """Old gate: [min, max] widened by the max relative repeat spread. Report only."""
    if len(prefills) != CONTROL_REPEATS:
        raise ValueError(f"expected {CONTROL_REPEATS} prefills, got {len(prefills)}")
    median = float(statistics.median(prefills))
    if median <= 0:
        raise ValueError("median prefill_s must be positive")
    low_raw = min(prefills)
    high_raw = max(prefills)
    spread = max(abs(value - median) / median for value in prefills)
    return {
        "reference_run_id": REFERENCE_RUN_ID,
        "n": CONTROL_N,
        "repeats_prefill_s": list(prefills),
        "min": low_raw,
        "max": high_raw,
        "median": median,
        "max_relative_spread": spread,
        "within_session_low": low_raw * (1.0 - spread),
        "within_session_high": high_raw * (1.0 + spread),
    }


def exponent_analysis(repo_root: Path) -> Path:
    return repo_root / "derived" / "cap4" / f"sealed_{EXPONENT_RUN_ID}" / "analysis.json"


def load_exponents(repo_root: Path) -> dict[str, float]:
    doc = json.loads(exponent_analysis(repo_root).read_text(encoding="utf-8"))
    per_arm = doc.get("per_arm")
    if not isinstance(per_arm, dict):
        raise TypeError(f"{EXPONENT_RUN_ID} analysis.json has no per_arm object")
    exponents: dict[str, float] = {}
    for arm, body in per_arm.items():
        if not isinstance(body, dict):
            continue
        fit = body.get("power_law_fit")
        if isinstance(fit, dict) and fit.get("b") is not None:
            exponents[str(arm)] = float(fit["b"])
    if not exponents:
        raise ValueError(f"{EXPONENT_RUN_ID} analysis.json has no power_law_fit.b")
    return exponents


def select_b_t2s(exponents: dict[str, float], arm: str = MATCHING_ARM) -> dict[str, Any]:
    """Matching KV exponent, else the mean of the sealed fits."""
    if not exponents:
        raise ValueError("no sealed prefill exponents")
    mean_b = float(statistics.mean(exponents.values()))
    if arm in exponents:
        chosen = float(exponents[arm])
        source = "matching_kv"
    else:
        chosen = mean_b
        source = "mean_of_arms"
    return {
        "b_t2s": chosen,
        "b_t2s_arm": arm if arm in exponents else None,
        "b_t2s_source": source,
        "b_t2s_by_arm": dict(exponents),
        "b_t2s_mean": mean_b,
        "b_t2s_run_id": EXPONENT_RUN_ID,
    }


def reference_band(repo_root: Path | None = None) -> dict[str, Any]:
    root = repo_root or ROOT
    window = within_session_window(load_prefills(reference_work(root)))
    chosen = select_b_t2s(load_exponents(root), MATCHING_ARM)
    median = float(window["median"])
    tol = float(chosen["b_t2s"]) * (RESOLUTION / CONTROL_N)
    base_s = CONTROL_REPEATS * median
    raw_estimate = base_s + CANARY_OVERHEAD_S
    if raw_estimate == math.floor(raw_estimate):
        estimate_s = int(raw_estimate)
    else:
        estimate_s = math.ceil(raw_estimate)
    return {
        **window,
        **chosen,
        "resolution": RESOLUTION,
        "n_control": CONTROL_N,
        "tol": tol,
        "low": median * (1.0 - tol),
        "high": median * (1.0 + tol),
        "formula": FORMULA,
        "interpretation": INTERPRETATION,
        "base_s": base_s,
        "noreboot_reason": _uncold_reason(),
        "canary_overhead_s": CANARY_OVERHEAD_S,
        "estimate_s": int(estimate_s),
        "estimate_formula": (
            "estimate_s = next whole second of (3 * reference median prefill_s "
            "+ canary overhead 752.1359013 from c2246b1f)"
        ),
    }


def evaluate_control(prefills: list[float], band: dict[str, Any]) -> dict[str, Any]:
    verdict: dict[str, Any] = {
        "reference_run_id": band["reference_run_id"],
        "n": band["n"],
        "low": band["low"],
        "high": band["high"],
        "tol": band["tol"],
        "b_t2s": band["b_t2s"],
        "b_t2s_source": band["b_t2s_source"],
        "b_t2s_run_id": band["b_t2s_run_id"],
        "reference_median": band["median"],
        "within_session_low": band["within_session_low"],
        "within_session_high": band["within_session_high"],
        "max_relative_spread": band["max_relative_spread"],
        "control_prefills": list(prefills),
        "formula": band["formula"],
        "interpretation": band["interpretation"],
    }
    if len(prefills) != CONTROL_REPEATS:
        verdict["pass"] = False
        verdict["median"] = None
        verdict["old_within_session_pass"] = None
        verdict["reason"] = f"expected {CONTROL_REPEATS} control prefills, got {len(prefills)}"
        return verdict
    median = float(statistics.median(prefills))
    ok = float(band["low"]) <= median <= float(band["high"])
    old_ok = float(band["within_session_low"]) <= median <= float(band["within_session_high"])
    verdict["pass"] = ok
    verdict["median"] = median
    verdict["old_within_session_pass"] = old_ok
    verdict["reason"] = "inside_tol" if ok else "median_outside_tol"
    return verdict


def check_work(repo_root: Path, work: Path) -> dict[str, Any]:
    return evaluate_control(load_prefills(work), reference_band(repo_root))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-band", action="store_true")
    parser.add_argument("--check-work", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=None)
    args = parser.parse_args(argv)
    root = (args.repo_root or ROOT).resolve()
    if args.print_band:
        print(json.dumps(reference_band(root), sort_keys=True))
        return 0
    if args.check_work is not None:
        verdict = check_work(root, args.check_work)
        print(json.dumps(verdict, sort_keys=True))
        return 0 if verdict["pass"] else 1
    parser.error("pass --print-band or --check-work")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
