"""STAT-1 (Task BR, docs/registrations/BR_STAT_2026-10-26.md): threshold estimates from
biased-coin up-and-down trials.

- next_start(): the level a new block starts from (the previous block's next level), so a
  queue can chain blocks across sessions.
- cir_quantile(): centered isotonic regression (Oron and Flournoy 2017) of the FAIL
  probability on the level; returns the level where P(fail) = 1 - q (q = 0.9 -> L90,
  q = 0.5 -> L50) by linear interpolation between the centered points, or None when the
  target lies outside the observed range (reported as censored, never extrapolated).
- bootstrap_ci(): two-stage bootstrap (sessions with replacement, then trials within each drawn
  session), percentile interval, fixed seed.
- reversals(): number of direction changes in a level sequence.
Input trials are (session, level, held) triples read from the sealed point records.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

Trial = tuple[str, int, bool]


def pav(levels: Sequence[int], fails: Sequence[int], ns: Sequence[int]) -> list[list[float]]:
    """Pool-adjacent-violators on fail rates (non-decreasing in level).

    Returns blocks [x_center, rate, weight]; x_center is the weight-averaged level."""
    blocks: list[list[float]] = []
    for x, f, n in zip(levels, fails, ns, strict=True):
        blocks.append([float(x) * n, float(f), float(n)])
        while len(blocks) > 1 and blocks[-2][1] / blocks[-2][2] > blocks[-1][1] / blocks[-1][2]:
            xs, fs, ws = blocks.pop()
            blocks[-1][0] += xs
            blocks[-1][1] += fs
            blocks[-1][2] += ws
    return [[xs / w, f / w, w] for xs, f, w in blocks]


def cir_quantile(trials: Sequence[Trial], q: float) -> float | None:
    by: dict[int, list[int]] = {}
    for _, level, held in trials:
        by.setdefault(int(level), []).append(0 if held else 1)
    if not by:
        return None
    levels = sorted(by)
    pts = pav(levels, [sum(by[x]) for x in levels], [len(by[x]) for x in levels])
    target = 1.0 - float(q)
    for (x0, y0, _), (x1, y1, _) in pairwise(pts):
        if y0 <= target <= y1 and y1 > y0:
            return x0 + (target - y0) * (x1 - x0) / (y1 - y0)
    if len(pts) == 1 and abs(pts[0][1] - target) < 1e-12:
        return pts[0][0]
    return None


def bootstrap_ci(
    trials: Sequence[Trial], q: float, *, draws: int = 2000, seed: int = 20261026
) -> dict[str, Any]:
    rng = random.Random(seed)
    sessions: dict[str, list[Trial]] = {}
    for t in trials:
        sessions.setdefault(t[0], []).append(t)
    keys = sorted(sessions)
    est: list[float] = []
    censored = 0
    for _ in range(draws):
        sample: list[Trial] = []
        for _s in range(len(keys)):
            rows = sessions[rng.choice(keys)]
            sample.extend(rng.choice(rows) for _r in range(len(rows)))
        v = cir_quantile(sample, q)
        if v is None:
            censored += 1
        else:
            est.append(v)
    est.sort()
    if not est:
        return {"lo": None, "hi": None, "censored_draws": censored, "draws": draws}
    lo = est[int(0.05 * (len(est) - 1))]
    hi = est[int(0.95 * (len(est) - 1))]
    return {"lo": lo, "hi": hi, "censored_draws": censored, "draws": draws}


def reversals(levels: Sequence[int]) -> int:
    n = 0
    last = 0
    for a, b in pairwise(levels):
        d = (b > a) - (b < a)
        if d and last and d != last:
            n += 1
        if d:
            last = d
    return n


def next_start(summary: dict[str, Any]) -> int | None:
    """The level the next block starts from: the finished block's bcd.next_level."""
    v = (summary.get("bcd") or {}).get("next_level")
    return int(v) if isinstance(v, int) else None


def load_trials(paths: Sequence[Path]) -> list[Trial]:
    """(session tag, level, held) from STAT-1 summary.json files; the session tag is the
    SESSION number in the queue's cell label (stat_<anchor>_s<N>_b<j>), else the summary's
    parent window directory name."""
    import re

    out: list[Trial] = []
    for p in paths:
        d = json.loads(p.read_text(encoding="utf-8"))
        m = re.search(r"_s(\d+)_(b\d+|bin)", str(p))
        tag = f"s{m.group(1)}" if m else p.parents[2].name
        for pt in d.get("points") or []:
            out.append((tag, int(pt["n_cached"]), bool(pt.get("holds"))))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("summaries", nargs="*", type=Path)
    ap.add_argument("--next-start", type=Path, default=None, help="Print one block's next level.")
    a = ap.parse_args(argv)
    if a.next_start is not None:
        v = next_start(json.loads(a.next_start.read_text(encoding="utf-8")))
        print("" if v is None else v)
        return 0
    trials = load_trials(a.summaries)
    out = {
        "n_trials": len(trials),
        "sessions": sorted({t[0] for t in trials}),
        "L90": cir_quantile(trials, 0.9),
        "L50": cir_quantile(trials, 0.5),
        "L90_ci90": bootstrap_ci(trials, 0.9),
        "L50_ci90": bootstrap_ci(trials, 0.5),
        "reversals": reversals([t[1] for t in trials]),
    }
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
