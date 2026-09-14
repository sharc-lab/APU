"""Aggregate derived/gpu_smoke matrix cells into the A-vs-gpu_only report table.

Diagnostic only. Reads JSON written by tools/smoke_gpu_exec.py under a matrix tag
prefix; prints per-cell metrics, CVs, settle-adequacy on available_mb, and
ratios vs prior ladder/smoke answers.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SMOKE_DIR = ROOT / "derived" / "gpu_smoke"
ARMS = ("A", "gpu_only")
REPEATS = 3

# Prior tool answers at n=12000 (means): used only for reproduction statement.
PRIOR = {
    "ladder_prefill_ratio_A_over_gpu": 13.8,  # approximate cited A/gpu_only prefill
    "smoke_prefill_ratio_A_over_gpu": 8.5,
    "smoke_decode_ratio_gpu_over_A": 3.04,
    "ladder_decode_ratio_gpu_over_A": 4.49,
}


def _mean(vals: list[float]) -> float | None:
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _median(vals: list[float]) -> float | None:
    if not vals:
        return None
    return float(statistics.median(vals))


def _cv(vals: list[float]) -> float | None:
    if len(vals) < 2:
        return 0.0 if vals else None
    m = _mean(vals)
    if m is None or m == 0:
        return None
    return float(statistics.stdev(vals) / abs(m))


def _load_cell(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _free_start_mb(rec: dict[str, Any]) -> float | None:
    v = rec.get("free_physical_mb_start")
    if v is not None:
        return float(v)
    rw = (rec.get("diagnostics") or {}).get("rss_window") or {}
    v = rw.get("free_physical_mb_start")
    return float(v) if v is not None else None


def _free_peak_mb(rec: dict[str, Any]) -> float | None:
    free_b = rec.get("free_physical_at_peak")
    if free_b is not None:
        return float(free_b) / (1024.0 * 1024.0)
    rw = (rec.get("diagnostics") or {}).get("rss_window") or {}
    v = rw.get("free_physical_mb_at_peak")
    return float(v) if v is not None else None


def _available_start_mb(rec: dict[str, Any]) -> float | None:
    v = rec.get("available_mb_start")
    if v is not None:
        return float(v)
    env = rec.get("environment_start") or {}
    v = env.get("available_mb")
    if v is not None:
        return float(v)
    rw = (rec.get("diagnostics") or {}).get("rss_window") or {}
    v = rw.get("available_mb_start")
    return float(v) if v is not None else None


def _available_peak_mb(rec: dict[str, Any]) -> float | None:
    env = rec.get("environment_peak") or {}
    v = env.get("available_mb")
    if v is not None:
        return float(v)
    rw = (rec.get("diagnostics") or {}).get("rss_window") or {}
    v = rw.get("available_mb_at_peak")
    return float(v) if v is not None else None


def _peak_ws_gb(rec: dict[str, Any]) -> float | None:
    peak = rec.get("peak_ws_bytes")
    if peak is None:
        return None
    return float(peak) / (1024.0**3)


def collect(
    *,
    launch_context: str,
    tag: str,
    ns: list[int],
    smoke_dir: Path = SMOKE_DIR,
) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for n in ns:
        for repeat in range(REPEATS):
            for arm in ARMS:
                name = f"{tag}_{launch_context}_arm{arm}_n{n}_r{repeat}.json"
                path = smoke_dir / name
                alt = smoke_dir / f"{launch_context}_arm{arm}_n{n}_r{repeat}.json"
                rec = _load_cell(path) or _load_cell(alt)
                cells.append(
                    {
                        "arm": arm,
                        "n_tokens": n,
                        "repeat": repeat,
                        "path": str(path if path.is_file() else alt),
                        "present": rec is not None,
                        "record": rec,
                    }
                )
    return {"launch_context": launch_context, "tag": tag, "ns": ns, "cells": cells}


def settle_adequacy(
    cells: list[dict[str, Any]],
    *,
    n: int,
    threshold_mb: float | None,
    provisional: bool = True,
    citation: str | None = None,
) -> dict[str, Any]:
    by_arm: dict[str, list[float]] = {a: [] for a in ARMS}
    for cell in cells:
        if cell["n_tokens"] != n:
            continue
        rec = cell.get("record")
        if not rec or str(rec.get("classification")) != "OK":
            continue
        avail = _available_start_mb(rec)
        if avail is None:
            continue
        by_arm[cell["arm"]].append(avail)
    all_vals = [v for vs in by_arm.values() for v in vs]
    if len(by_arm["A"]) == 0 or len(by_arm["gpu_only"]) == 0 or len(all_vals) < 2:
        return {
            "evaluated": False,
            "n_tokens": n,
            "metric": "available_mb",
            "by_arm_mb": by_arm,
            "spread_mb": None,
            "threshold_mb": threshold_mb,
            "provisional": provisional,
            "citation": citation,
            "pass": None,
        }
    spread = max(all_vals) - min(all_vals)
    return {
        "evaluated": True,
        "n_tokens": n,
        "metric": "available_mb",
        "by_arm_mb": by_arm,
        "max_mb": max(all_vals),
        "min_mb": min(all_vals),
        "spread_mb": spread,
        "threshold_mb": threshold_mb,
        "provisional": provisional,
        "citation": citation,
        "pass": (spread <= threshold_mb) if threshold_mb is not None else None,
        "rule": "max-min available_mb across arms at same n",
    }


def summarize(
    bundle: dict[str, Any],
    *,
    threshold_mb: float | None = None,
    provisional: bool = True,
    citation: str | None = None,
) -> dict[str, Any]:
    ns: list[int] = list(bundle["ns"])
    per_cell: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    for cell in bundle["cells"]:
        rec = cell.get("record") or {}
        per_cell.append(
            {
                "arm": cell["arm"],
                "n_tokens": cell["n_tokens"],
                "repeat": cell["repeat"],
                "classification": rec.get("classification"),
                "prefill_s": rec.get("prefill_s"),
                "decode_tok_s": rec.get("decode_tok_s"),
                "peak_ws_bytes": rec.get("peak_ws_bytes"),
                "peak_ws_gb": _peak_ws_gb(rec) if rec else None,
                "available_mb_start": _available_start_mb(rec) if rec else None,
                "available_mb_at_peak": _available_peak_mb(rec) if rec else None,
                "free_physical_mb_start": _free_start_mb(rec) if rec else None,
                "free_physical_mb_at_peak": _free_peak_mb(rec) if rec else None,
                "path": cell["path"],
            }
        )

    for arm in ARMS:
        for n in ns:
            ok = [
                c
                for c in per_cell
                if c["arm"] == arm and c["n_tokens"] == n and c.get("classification") == "OK"
            ]
            pref = [float(c["prefill_s"]) for c in ok if c.get("prefill_s") is not None]
            dec = [float(c["decode_tok_s"]) for c in ok if c.get("decode_tok_s") is not None]
            peak = [float(c["peak_ws_gb"]) for c in ok if c.get("peak_ws_gb") is not None]
            astart = [
                float(c["available_mb_start"])
                for c in ok
                if c.get("available_mb_start") is not None
            ]
            apeak = [
                float(c["available_mb_at_peak"])
                for c in ok
                if c.get("available_mb_at_peak") is not None
            ]
            fstart = [
                float(c["free_physical_mb_start"])
                for c in ok
                if c.get("free_physical_mb_start") is not None
            ]
            fpeak = [
                float(c["free_physical_mb_at_peak"])
                for c in ok
                if c.get("free_physical_mb_at_peak") is not None
            ]
            rows.append(
                {
                    "arm": arm,
                    "n_tokens": n,
                    "n_ok": len(ok),
                    "prefill_s_mean": _mean(pref),
                    "prefill_s_cv": _cv(pref),
                    "prefill_s_vals": pref,
                    "decode_tok_s_mean": _mean(dec),
                    "decode_tok_s_cv": _cv(dec),
                    "decode_tok_s_vals": dec,
                    "peak_ws_gb_mean": _mean(peak),
                    "peak_ws_gb_cv": _cv(peak),
                    "available_mb_start_mean": _mean(astart),
                    "available_mb_start_cv": _cv(astart),
                    "available_mb_start_vals": astart,
                    "available_mb_at_peak_mean": _mean(apeak),
                    "free_physical_mb_start_mean": _mean(fstart),
                    "free_physical_mb_start_cv": _cv(fstart),
                    "free_physical_mb_start_vals": fstart,
                    "free_physical_mb_at_peak_mean": _mean(fpeak),
                }
            )

    adequacy = [
        settle_adequacy(
            bundle["cells"],
            n=n,
            threshold_mb=threshold_mb,
            provisional=provisional,
            citation=citation,
        )
        for n in ns
    ]

    ratios: dict[str, Any] = {}
    reproduction: dict[str, Any] = {}
    for n in ns:
        by = {(r["arm"], r["n_tokens"]): r for r in rows}
        a = by.get(("A", n))
        g = by.get(("gpu_only", n))
        if not a or not g:
            continue
        pref_a, pref_g = a.get("prefill_s_mean"), g.get("prefill_s_mean")
        dec_a, dec_g = a.get("decode_tok_s_mean"), g.get("decode_tok_s_mean")
        entry: dict[str, Any] = {"n_tokens": n}
        if pref_a and pref_g:
            entry["prefill_A_over_gpu_only"] = pref_a / pref_g
        if dec_a and dec_g:
            entry["decode_gpu_only_over_A"] = dec_g / dec_a
        ratios[str(n)] = entry

        if (
            n == 12000
            and entry.get("prefill_A_over_gpu_only")
            and entry.get("decode_gpu_only_over_A")
        ):
            pref_r = entry["prefill_A_over_gpu_only"]
            dec_r = entry["decode_gpu_only_over_A"]
            # Distance to prior cited answers
            d_ladder_p = abs(pref_r - PRIOR["ladder_prefill_ratio_A_over_gpu"])
            d_smoke_p = abs(pref_r - PRIOR["smoke_prefill_ratio_A_over_gpu"])
            d_ladder_d = abs(dec_r - PRIOR["ladder_decode_ratio_gpu_over_A"])
            d_smoke_d = abs(dec_r - PRIOR["smoke_decode_ratio_gpu_over_A"])
            reproduction = {
                "n_tokens": 12000,
                "prefill_A_over_gpu_only": pref_r,
                "decode_gpu_only_over_A": dec_r,
                "prefill_closer_to": (
                    "ladder(~13.8x)" if d_ladder_p < d_smoke_p else "smoke(~8.5x)"
                ),
                "decode_closer_to": (
                    "ladder(~4.49x)" if d_ladder_d < d_smoke_d else "smoke(~3.04x)"
                ),
                "distances": {
                    "prefill_to_ladder_13_8": d_ladder_p,
                    "prefill_to_smoke_8_5": d_smoke_p,
                    "decode_to_ladder_4_49": d_ladder_d,
                    "decode_to_smoke_3_04": d_smoke_d,
                },
            }

    return {
        "per_cell": per_cell,
        "rows": rows,
        "settle_adequacy": adequacy,
        "ratios": ratios,
        "reproduction_vs_prior": reproduction,
        "prior_cited": PRIOR,
    }


def _fmt(val: float | None, width: int, prec: int) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return f"{'n/a':>{width}}"
    return f"{val:{width}.{prec}f}"


def render_table(summary: dict[str, Any]) -> str:
    lines = [
        "per-cell:",
        "arm       | N     | r | prefill_s | decode_tok_s | peak_ws_GB | avail_start_MB | free_start_MB | class",
        "----------+-------+---+-----------+--------------+------------+----------------+---------------+------",
    ]
    for c in summary["per_cell"]:
        lines.append(
            f"{c['arm']:<9} | {c['n_tokens']:<5} | {c['repeat']} | "
            f"{_fmt(c.get('prefill_s'), 9, 3)} | "
            f"{_fmt(c.get('decode_tok_s'), 12, 3)} | "
            f"{_fmt(c.get('peak_ws_gb'), 10, 3)} | "
            f"{_fmt(c.get('available_mb_start'), 14, 1)} | "
            f"{_fmt(c.get('free_physical_mb_start'), 13, 1)} | "
            f"{c.get('classification')}"
        )

    lines.append("")
    lines.append("per-arm means + CV:")
    lines.append(
        "arm       | N     | prefill_s (CV)     | decode_tok_s (CV)  | avail_start_MB (CV) | n_ok"
    )
    lines.append(
        "----------+-------+--------------------+--------------------+---------------------+-----"
    )
    for r in summary["rows"]:
        lines.append(
            f"{r['arm']:<9} | {r['n_tokens']:<5} | "
            f"{_fmt(r['prefill_s_mean'], 8, 2)} ({_fmt(r['prefill_s_cv'], 5, 3)}) | "
            f"{_fmt(r['decode_tok_s_mean'], 8, 3)} ({_fmt(r['decode_tok_s_cv'], 5, 3)}) | "
            f"{_fmt(r['available_mb_start_mean'], 8, 1)} ({_fmt(r['available_mb_start_cv'], 5, 3)}) | "
            f"{r['n_ok']}"
        )

    lines.append("")
    lines.append("settle-adequacy (max-min available_mb across arms at same n):")
    for a in summary["settle_adequacy"]:
        if not a.get("evaluated"):
            lines.append(f"  n={a['n_tokens']}: not evaluated (need both arms)")
            continue
        verdict = "PASS" if a.get("pass") else "FAIL"
        prov = " provisional" if a.get("provisional") else ""
        lines.append(
            f"  n={a['n_tokens']}: spread={a['spread_mb']:.1f} MB "
            f"(max={a['max_mb']:.1f} min={a['min_mb']:.1f}) "
            f"threshold={a.get('threshold_mb')} MB{prov} -> {verdict}"
        )
        for arm, vals in (a.get("by_arm_mb") or {}).items():
            lines.append(f"    {arm}: {[round(v, 1) for v in vals]}")

    lines.append("")
    lines.append("ratios (convention: prefill A/gpu_only; decode gpu_only/A speedup):")
    for n, ent in summary.get("ratios", {}).items():
        lines.append(
            f"  n={n}: prefill A:gpu_only = {_fmt(ent.get('prefill_A_over_gpu_only'), 0, 3)} ; "
            f"decode gpu_only:A = {_fmt(ent.get('decode_gpu_only_over_A'), 0, 3)}"
        )

    repro = summary.get("reproduction_vs_prior") or {}
    if repro:
        lines.append("")
        lines.append("reproduction vs prior tool answers @ n=12000:")
        lines.append(
            f"  this run prefill A/gpu_only={repro['prefill_A_over_gpu_only']:.3f} "
            f"-> closer to {repro['prefill_closer_to']}"
        )
        lines.append(
            f"  this run decode gpu_only/A={repro['decode_gpu_only_over_A']:.3f} "
            f"-> closer to {repro['decode_closer_to']}"
        )
    return "\n".join(lines)


def _load_isolation_from_yaml(path: Path) -> dict[str, Any]:
    in_iso = False
    out: dict[str, Any] = {
        "settle_adequacy_max_spread_mb": None,
        "settle_adequacy_provisional": True,
        "settle_adequacy_citation": None,
        "settle_adequacy_metric": "available_mb",
        "pre_run_available_mb_min": None,
    }
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if stripped == "isolation:":
            in_iso = True
            continue
        if in_iso and line and not line[0].isspace() and not line.startswith("#"):
            in_iso = False
        if not in_iso:
            continue
        if ":" not in stripped:
            continue
        key, raw = stripped.split(":", 1)
        key = key.strip()
        raw = raw.strip()
        if raw.startswith('"') and raw.endswith('"'):
            value: Any = raw[1:-1]
        elif raw.startswith("'") and raw.endswith("'"):
            value = raw[1:-1]
        else:
            value = raw.split("#", 1)[0].strip()
        if key == "settle_adequacy_max_spread_mb":
            out[key] = float(value)
        elif (
            key == "settle_adequacy_max_free_mb_spread"
            and out["settle_adequacy_max_spread_mb"] is None
        ):
            # Legacy key; prefer settle_adequacy_max_spread_mb when both present.
            out["settle_adequacy_max_spread_mb"] = float(value)
        elif key == "settle_adequacy_provisional":
            out[key] = str(value).lower() == "true"
        elif key in (
            "settle_adequacy_citation",
            "settle_adequacy_metric",
        ):
            out[key] = str(value)
        elif key == "pre_run_available_mb_min":
            out[key] = float(value)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--launch-context",
        default="ssh_foreground",
        choices=["local_console", "ssh_foreground", "ssh_detached"],
    )
    parser.add_argument("--tag", default="gpu_only_matrix")
    parser.add_argument("--smoke-dir", type=Path, default=SMOKE_DIR)
    parser.add_argument(
        "--ns-list",
        default="2000,12000",
        help="comma-separated N values matching the matrix run",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    ns = [int(x.strip()) for x in args.ns_list.split(",") if x.strip()]
    iso = _load_isolation_from_yaml(ROOT / "configs" / "delta_n.yaml")
    threshold = iso.get("settle_adequacy_max_spread_mb")

    bundle = collect(
        launch_context=args.launch_context,
        tag=args.tag,
        ns=ns,
        smoke_dir=args.smoke_dir,
    )
    summary = summarize(
        bundle,
        threshold_mb=threshold,
        provisional=bool(iso.get("settle_adequacy_provisional", True)),
        citation=iso.get("settle_adequacy_citation"),
    )
    table = render_table(summary)
    out = {
        "bundle": {
            "launch_context": bundle["launch_context"],
            "tag": bundle["tag"],
            "ns": ns,
            "settle_adequacy_metric": iso.get("settle_adequacy_metric"),
            "settle_adequacy_max_spread_mb": threshold,
            "settle_adequacy_provisional": iso.get("settle_adequacy_provisional"),
            "settle_adequacy_citation": iso.get("settle_adequacy_citation"),
            "pre_run_available_mb_min": iso.get("pre_run_available_mb_min"),
            "cells": summary["per_cell"],
        },
        "summary": summary,
        "table": table,
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(table, flush=True)

    # Non-zero if adequacy evaluated and failed (so ReportOnly surfaces the gate).
    for a in summary["settle_adequacy"]:
        if a.get("evaluated") and a.get("pass") is False:
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
