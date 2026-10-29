"""Task BK: registered values and descriptive fits from the sealed weekend records.

Inputs: analysis/cc_20261026/cells.json (collect.py), bytes.json (bytes.py), peak_slopes.json,
provenance.json, the sealed point records, and the committed prediction files. Writes
analysis/cc_20261026/measured.json, which tools/figs/*.py and the meeting-pack figures read.
Every value carries its run_id(s) or a DERIVED / IN-SAMPLE / POST HOC label.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent

from tools import br_predict as br  # noqa: E402

K_C = br.K_COMPILED
A = br.A
GIB = 2**30


def cells() -> dict[str, dict[str, Any]]:
    rows = json.loads((OUT / "cells.json").read_text(encoding="utf-8"))
    return {f"{r['host']}:{r['window'][7:22]}:{r['label']}": r for r in rows if r.get("run_id")}


def by_label(cs: dict[str, dict[str, Any]], label: str, host: str = "xps") -> dict[str, Any]:
    hits = [r for k, r in cs.items() if r["label"] == label and r["host"] == host]
    if len(hits) != 1:
        raise KeyError(f"{host}:{label}: {len(hits)} rows")
    return hits[0]


def point_files(r: dict[str, Any]) -> list[dict[str, Any]]:
    return [json.loads(f.read_text()) for f in sorted((Path(r["dir"]) / "points").glob("*.json"))]


def grow_turns(r: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
    d = point_files(r)[0]
    return list((d.get("generation") or {}).get("turns") or []), d.get("failure_mode")


def fit_quad(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """least squares y = alpha x + beta x^2 (no intercept)."""
    s11 = sum(x * x for x in xs)
    s12 = sum(x**3 for x in xs)
    s22 = sum(x**4 for x in xs)
    t1 = sum(x * y for x, y in zip(xs, ys, strict=True))
    t2 = sum(x * x * y for x, y in zip(xs, ys, strict=True))
    det = s11 * s22 - s12 * s12
    return (t1 * s22 - t2 * s12) / det, (s11 * t2 - s12 * t1) / det


def solve_quad(alpha: float, beta: float, y: float) -> float:
    return (-alpha + math.sqrt(alpha * alpha + 4 * beta * y)) / (2 * beta)


def lin(xs: list[float], ys: list[float]) -> dict[str, float]:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    syy = sum((y - my) ** 2 for y in ys)
    b = sxy / sxx if sxx else 0.0
    r2 = (sxy * sxy / (sxx * syy)) if sxx and syy else 0.0
    return {"slope": b, "intercept": my - b * mx, "r2": r2, "n": n}


def main() -> int:
    cs = cells()
    m: dict[str, Any] = {"generator": "analysis/cc_20261026/measure.py"}

    # --- headline limits (registered L* = bisection largest held) ---
    cells_l = {
        "f16_unl": ("mech_f16_unlx", None),
        "i8_unl": ("mech_8_unlx", None),
        "u4_unl": ("mech_u4_unlx", None),
        "f16_c1024": ("mech_f16_c1024", 1024),
        "i8_c1024": ("mech_8_c1024", 1024),
        "u4_c1024": ("mech_u4_c1024", 1024),
        "f16_c256": ("mech_f16_c256", 256),
        "i8_c256": ("mech_8_c256", 256),
        "u4_c256": ("mech_u4_c256", 256),
    }
    lim = {}
    for k, (lab, cap) in cells_l.items():
        r = by_label(cs, lab)
        lim[k] = {
            "label": lab,
            "run_id": r["run_id"],
            "L_star": r["largest_n_cached"],
            "first_fail": r["first_fail_n_cached"],
            "cap": cap,
        }
    m["xps_limits"] = lim
    m["ratios_u4_over_f16"] = {
        c: round(lim[f"u4_{c}"]["L_star"] / lim[f"f16_{c}"]["L_star"], 3)
        for c in ("unl", "c1024", "c256")
    }
    m["byte_ratios"] = {
        "compiled_f16_over_u4": round(K_C["f16"] / K_C["u4"], 3),
        "nominal_f16_over_u4": round(147456 / 41472, 3),
        "law_unchunked_(kf16+a)/(ku4+a)": round((K_C["f16"] + A) / (K_C["u4"] + A), 3),
        "label": "DERIVED (compiled k from the BR constants; a 87,520 e4e797e7)",
    }

    # --- repeats (ADDENDUM 1 rule): registered L*, repeat pass rate, robust ceiling ---
    # Pooled evidence: every XPS 57%-override resident cell with the same arm and cap class
    # (plain pipeline and explicit-unlimited = uncapped), balloons included, slider 50/74% excluded.
    def cap_class(r: dict[str, Any]) -> str:
        c = (r.get("scheduler") or {}).get("max_num_batched_tokens")
        return "c256" if c == 256 else ("c1024" if c == 1024 else "unl")

    pooled: dict[tuple[str, str], dict[int, list[tuple[bool, str]]]] = {}
    for r in cs.values():
        if r["host"] != "xps" or r.get("kind") != "resident_limit":
            continue
        if r["label"].startswith("slider_") and r["window"].endswith(
            ("174345_cc_v3", "185526_cc_v3")
        ):
            continue
        if r["label"] == "a33_u4_cache2":
            continue
        key = (str(r["arm"]), cap_class(r))
        for n, h, _ in r["points"]:
            pooled.setdefault(key, {}).setdefault(int(n), []).append((bool(h), r["run_id"]))
    reps = {
        "u4_c256": ("mech_u4_c256", "rep_u4_c256", "gpu_only_u4", "c256"),
        "f16_c256": ("mech_f16_c256", "rep_f16_c256", "gpu_only_f16", "c256"),
        "u4_unl": ("mech_u4_unlx", "rep_u4_unlx", "gpu_only_u4", "unl"),
        "f16_unl": ("mech_f16_unlx", "rep_f16_unlx", "gpu_only_f16", "unl"),
        "i8_unl": ("a21_8bit", "a21_8bit_rep", "gpu_only_i8", "unl"),
    }
    rep_out = {}
    for k, (src, rep, arm, cc) in reps.items():
        s = by_label(cs, src)
        r = by_label(cs, rep)
        held: dict[int, list[bool]] = {}
        for n, h, _ in r["points"]:
            held.setdefault(int(n), []).append(bool(h))
        pool = pooled.get((arm, cc), {})
        robust = [n for n, v in pool.items() if len(v) >= 2 and all(h for h, _ in v)]
        rep_out[k] = {
            "registered_L_star": [s["largest_n_cached"], s["run_id"]],
            "repeat_cell": {str(n): f"{sum(v)}/{len(v)}" for n, v in sorted(held.items())},
            "repeat_run_id": r["run_id"],
            "pooled_at_L_star_and_step": {
                str(
                    n
                ): f"{sum(h for h, _ in pool[n])}/{len(pool[n])} ({','.join(sorted({x for _, x in pool[n]}))})"
                for n in sorted(held)
                if n in pool
            },
            "robust_ceiling": max(robust) if robust else "not established",
            "robust_basis": (
                f"{sum(h for h, _ in pool[max(robust)])}/{len(pool[max(robust)])} at {max(robust)} "
                f"({','.join(sorted({x for _, x in pool[max(robust)]}))})"
                if robust
                else "no point with >= 2 attempts held every time"
            ),
            "rule": "highest n held in every attempt with >= 2 attempts, pooled over the cells above",
        }
    m["repeats"] = rep_out

    # --- implied B per cell (IN-SAMPLE, DERIVED; BR rule) ---
    m["implied_B"] = br.budget_rule()

    # --- A20 / bytes (sealed child records) ---
    by = json.loads((OUT / "bytes.json").read_text(encoding="utf-8"))
    pk = json.loads((OUT / "peak_slopes.json").read_text(encoding="utf-8"))
    m["bytes"] = {
        k: {
            "run_id": v["run_id"],
            "arm": v["arm"],
            "cap": v["cap"],
            "settled_kv_slope": v["settled_kv_slope_B_per_token"],
            "usm_device_after_turn1_slope": v["usm_device_after_turn1_slope_B_per_token"],
            "peak10hz_minus_after_turn1": (pk.get(k) or {}).get("peak_minus_after_turn1_slope"),
            "peak10hz_minus_after_turn2": (pk.get(k) or {}).get("peak_minus_after_turn2_slope"),
            "W_shared_mb": next(
                (p["W_shared_mb"] for p in v["points"] if p.get("W_shared_mb")), None
            ),
        }
        for k, v in by.items()
        if v.get("settled_kv_slope_B_per_token") is not None
    }
    # Fig 2 (BO tool) keys: per precision, KiB per context token, uncapped (k + retained a)
    m["bytes_per_token"] = {}
    for prec, key in (
        ("f16", "xps:bytes_f16_unl"),
        ("u8", "xps:bytes_8_unl"),
        ("u4", "xps:bytes_u4_unl"),
    ):
        v = m["bytes"][key]
        m["bytes_per_token"][prec] = {
            "total_KiB": round(
                (v["settled_kv_slope"] + v["usm_device_after_turn1_slope"]) / 1024, 1
            ),
            "run_id": v["run_id"],
        }

    # --- TTFT (A8, A24) ---
    tt = {}
    for lab in ("ttft_u4_unlx", "ttft_u4_c1024", "ttft_u4_c256"):
        r = by_label(cs, lab)
        per: dict[int, list[float]] = {}
        for d in point_files(r):
            if d.get("turn1_prefill_s") is not None:
                per.setdefault(int(d["n_cached"]), []).append(float(d["turn1_prefill_s"]))
        tt[lab] = {
            "run_id": r["run_id"],
            "median_turn1_s": {str(n): sorted(v)[len(v) // 2] for n, v in sorted(per.items())},
        }
    u = tt["ttft_u4_unlx"]["median_turn1_s"]
    c256 = tt["ttft_u4_c256"]["median_turn1_s"]
    c1024 = tt["ttft_u4_c1024"]["median_turn1_s"]
    m["A8_cap_penalty_16384"] = {
        "c256": round(c256["16384"] / u["16384"] - 1, 3),
        "c1024": round(c1024["16384"] / u["16384"] - 1, 3),
        "runs": [tt[x]["run_id"] for x in tt],
    }
    rows = []
    for d in point_files(by_label(cs, "ttft_u4_unlx")):
        if d.get("turn1_prefill_s") is not None:
            rows.append((float(d["n_cached"]), float(d["turn1_prefill_s"])))
    al, be = fit_quad([x for x, _ in rows], [y for _, y in rows])
    m["A24_ttft_law"] = {
        "fit": {"alpha_s_per_token": al, "beta_s_per_token2": be},
        "L_TTFT": {f"{s}s": round(solve_quad(al, be, s)) for s in (5, 20, 30)},
        "predicted": {"5s": 5712, "20s": 16729, "30s": 22164},
        "run_id": tt["ttft_u4_unlx"]["run_id"],
    }
    m["ttft_cells"] = tt

    # --- grow (A9 / A23) with the corrected lengths (ADDENDUM 5) ---
    grow = {}
    for lab, arm, t, cap in (
        ("grow_f16_unl_d1k", "f16", 1024, None),
        ("grow_f16_unl", "f16", 2048, None),
        ("grow_f16_unl_d4k", "f16", 4096, None),
        ("grow_u4_unl_d1k", "u4", 1024, None),
        ("grow_u4_unl", "u4", 2048, None),
        ("grow_u4_unl_d4k", "u4", 4096, None),
        ("grow_f16_c256", "f16", 2048, 256),
        ("grow_u4_c256", "u4", 2048, 256),
    ):
        r = by_label(cs, lab)
        turns, fm = grow_turns(r)
        last = turns[-1]
        law = br.session_law(arm, t, cap=br.CAP_NONE if cap is None else cap)
        a23 = []
        for i in range(1, len(turns)):
            lh = float(turns[i - 1]["prompt_tokens_reported"])
            delta = float(turns[i]["prompt_tokens_reported"]) - lh
            pred = delta * (7.2e9 + 589824 * lh) / 1.015e13
            a23.append(
                [round(lh), round(delta), round(float(turns[i]["prefill_s"]), 2), round(pred, 2)]
            )
        grow[lab] = {
            "run_id": r["run_id"],
            "arm": arm,
            "turn_tokens": t,
            "cap": cap,
            "held_turns": len(turns),
            "failure_mode": fm,
            "last_held_prompt": int(last["prompt_tokens_reported"]),
            "last_held_ttft_s": round(float(last["prefill_s"]), 1),
            "corrected_law_L_hist": round(law),
            "residual_pct": round(100 * (int(last["prompt_tokens_reported"]) - law) / law, 1),
            "above_window_40960": int(last["prompt_tokens_reported"]) > 40960,
            "a23_turns_[L_hist,delta,ttft_s,pred_s]": a23,
        }
    m["grow"] = grow

    # --- Available MB vs L* (ADDENDUM 1.2; descriptive) ---
    # Available after load, before generate (the child's "loaded" record), median over the cell's
    # probes. Unballooned single-shot bisection cells at the 57% override only: balloons move
    # Available by design, cache_size 2 GiB is a different mechanism, 50/74% are the slider.
    excl = ("bal_g1", "bal_g2", "bal_c", "a33_u4_cache2", "slider_")
    av = []
    for key, v in by.items():
        host, lab = key.split(":", 1)
        r = next((x for x in cs.values() if x["host"] == host and x["label"] == lab), None)
        if host != "xps" or r is None or r.get("note") or lab.startswith(excl):
            continue
        if v["cap"] in (256, 1024) or not r.get("largest_n_cached"):
            continue
        a = sorted(p["available_mb_loaded"] for p in v["points"] if p.get("available_mb_loaded"))
        if not a:
            continue
        av.append(
            [
                lab,
                r["run_id"],
                r["arm"],
                r["largest_n_cached"],
                round(a[len(a) // 2]),
                round(a[0]),
                round(a[-1]),
            ]
        )
    m["available_vs_L"] = {
        "rows_[label,run_id,arm,L*,median,min,max_available_mb_after_load]": av,
        "label": "DESCRIPTIVE",
    }
    for arm in ("gpu_only_f16", "gpu_only_u4", "gpu_only_i8"):
        rr = [x for x in av if x[2] == arm]
        if len(rr) >= 3:
            m["available_vs_L"][arm] = lin([x[4] for x in rr], [x[3] for x in rr])
    # every probe (all XPS resident cells incl. slider/balloon): Available after load vs hold
    probes = []
    for key, v in by.items():
        if not key.startswith("xps:"):
            continue
        for p in v["points"]:
            if p.get("available_mb_loaded") and p.get("n"):
                probes.append(
                    [
                        key[4:],
                        v["run_id"],
                        v["arm"],
                        v["cap"],
                        p["n"],
                        bool(p["holds"]),
                        round(p["available_mb_loaded"]),
                    ]
                )
    m["available_probes"] = probes

    # --- slider (A25) and balloons (A22) ---
    m["slider"] = {
        "u4": [[50, 22750, "a2a7c763"], [57, 21250, "f4abaf6e"], [74, 17750, "9e31d4cb"]],
        "f16": [
            [50, 13250, "769c5a03"],
            [57, 12500, "c461f165"],
            [74, None, "39d38c85 UNSEALED, low rung 8,000 failed in turn 2"],
        ],
        "same_setting_fri_u4_57": [25000, "5943c992"],
    }
    m["balloon"] = {
        "f16": [
            {"gib": 0, "limit": 13250, "run_id": "ba7a1e89"},
            {"gib": 1, "limit": 14250, "run_id": "0014465c"},
        ],
        "u4": [
            {"gib": 0, "limit": 25000, "run_id": "d6616dfd"},
            {"gib": 1, "limit": 25000, "run_id": "598e882d"},
            {"gib": 2, "limit": 24750, "run_id": "f9953743"},
        ],
        "cpu_f16": [
            {"gib": 2, "limit": 14000, "run_id": "0f3950a3"},
            {"gib": 4, "limit": 14250, "run_id": "62d59ab8"},
        ],
        "not_run": "bal_g2_f16 98c9a927 (3 quiescence refusals); 241e4d3 floor refusals 5b2ca33c, 75fd18fc, 7cfbb796, 210c6ffd",
    }
    m["a19"] = {
        "xps": {"total_GiB": round(16752234496 / GIB, 3), "shared_GiB": round(9548773662 / GIB, 3)},
        "t2s": {"total_GiB": round(65018 / 1024, 3), "shared_GiB": round(48634 / 1024, 3)},
    }
    m["binding"] = {
        "16 GB, default unchunked": "memory (u4 25,000 5943c992; f16 14,000 aa914c4c)",
        "16 GB, step cap 256": "window (u4 memory 70,000 f8df3847 is a capacity probe above 40,960)",
        "64 GB, one session": "window (40,448 held x3 at f16/i8/u4: 7e249f7e, 6dfd1a12, 2d329db2)",
    }
    (OUT / "measured.json").write_text(json.dumps(m, indent=1) + "\n", encoding="ascii")
    print(
        json.dumps(
            {
                k: m[k]
                for k in ("ratios_u4_over_f16", "A8_cap_penalty_16384", "A24_ttft_law", "repeats")
            },
            indent=1,
        )
    )
    for k, v in grow.items():
        print(
            k,
            v["run_id"],
            v["held_turns"],
            v["last_held_prompt"],
            v["corrected_law_L_hist"],
            v["residual_pct"],
            v["last_held_ttft_s"],
            v["a23_turns_[L_hist,delta,ttft_s,pred_s]"][-1],
        )
    print("avail", {k: v for k, v in m["available_vs_L"].items() if not k.startswith("rows")})
    for x in m["available_vs_L"][
        "rows_[label,run_id,arm,L*,median,min,max_available_mb_after_load]"
    ]:
        print("  ", x)
    return 0


if __name__ == "__main__":
    sys.exit(main())
