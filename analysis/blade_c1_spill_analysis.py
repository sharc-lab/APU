"""Analysis for harness/blade_spill_sweep.py c1 output (Blade, silent VRAM spill).

Usage: py -3.12 analysis/blade_c1_spill_analysis.py <path-to-jsonl-prefix-without-extension> [--m1 <m1 jsonl>]

For every non-warm-up successful call it aligns the 1 s telemetry to the call window [t_start, t_end] (UTC) and
reports per ctx: n, prompt tokens, median/IQR/bootstrap 95% CI of TTFT, max Shared and Dedicated Usage of the
server PID, spilled fraction (raw and excess over the fitted no-spill Shared baseline), median SM clock and power
over active rows, throttle reasons, and TTFT/decode slowdown against a no-spill expectation.

No-spill expectation: TTFT(p) = a*p + b*p^2 fitted to the clean points (M1 ctx 8192..32768 medians plus the C1
anchor). Bootstrap CIs resample the measured calls only (they do not include fit uncertainty). An effect counts
only if the CI excludes 1.0. Power readings above 200 W are nvidia-smi sensor glitches and are dropped.
"""

from __future__ import annotations

import csv
import json
import random
import statistics as st
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

BOOT_N = 10000
BOOT_SEED = 12345


def q(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def boot_ci(xs, fn=st.median, n=BOOT_N, seed=BOOT_SEED):
    if len(xs) < 2:
        return (None, None)
    rng = random.Random(seed)
    vals = sorted(fn([rng.choice(xs) for _ in xs]) for _ in range(n))
    return (vals[int(0.025 * n)], vals[int(0.975 * n) - 1])


def parse_local(s, off_h):
    dt = datetime.strptime(s.strip(), "%Y/%m/%d %H:%M:%S.%f")
    return (dt - timedelta(hours=off_h)).replace(tzinfo=timezone.utc)


def load(prefix):
    p = Path(prefix)
    man = json.loads(Path(str(p) + "_manifest.json").read_text(encoding="utf-8"))
    off = man["environment"]["utc_offset_hours_local"]
    rows = [json.loads(l) for l in Path(str(p) + ".jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    smi = []
    lines = Path(str(p) + "_smi.csv").read_text(encoding="utf-8", errors="replace").splitlines()
    for l in lines[1:]:
        f = [x.strip() for x in l.split(",")]
        if len(f) < 8:
            continue
        try:
            t = parse_local(f[0], off)
            smi.append({"t": t, "util": float(f[1].split()[0]) if f[1][0].isdigit() else None,
                        "sm": float(f[2].split()[0]), "pwr": float(f[4].split()[0]), "temp": float(f[5]),
                        "mem": float(f[6].split()[0]), "thr": f[7]})
        except Exception:
            continue
    win = []
    lines = Path(str(p) + "_winctr.csv").read_text(encoding="utf-8", errors="replace").splitlines()
    for l in lines[1:]:
        f = l.split(",")
        if len(f) < 6 or not f[2] or not f[3]:
            continue
        try:
            win.append({"t": datetime.fromisoformat(f[0].replace("Z", "+00:00")), "pid": int(f[1]),
                        "ded": float(f[2]), "sh": float(f[3])})
        except Exception:
            continue
    return man, rows, smi, win


def between(xs, t0, t1):
    return [x for x in xs if t0 <= x["t"] <= t1]


def fit_expected(points):
    """Least squares TTFT = a*p + b*p^2 through the origin. points: [(p, ttft)]."""
    s11 = sum(p * p for p, _ in points)
    s12 = sum(p ** 3 for p, _ in points)
    s22 = sum(p ** 4 for p, _ in points)
    y1 = sum(p * t for p, t in points)
    y2 = sum(p * p * t for p, t in points)
    det = s11 * s22 - s12 * s12
    a = (y1 * s22 - y2 * s12) / det
    b = (s11 * y2 - s12 * y1) / det
    return lambda p: a * p + b * p * p


def main():
    prefix = sys.argv[1]
    m1_path = None
    if "--m1" in sys.argv:
        m1_path = sys.argv[sys.argv.index("--m1") + 1]
    man, rows, smi, win = load(prefix)
    calls = [r for r in rows if r.get("record") == "call" and not r.get("warmup") and r.get("outcome") == "ok"]
    by_ctx = {}
    for r in calls:
        by_ctx.setdefault(r["ctx"], []).append(r)

    clean = []
    if m1_path:
        m1 = [json.loads(l) for l in Path(m1_path).read_text(encoding="utf-8").splitlines() if l.strip()]
        for ctx in (8192, 16384, 24576, 32768):
            rs = [r for r in m1 if r.get("ctx_size") == ctx and r.get("status") == "ok"]
            if rs:
                clean.append((rs[0]["n_prompt_tokens"], st.median(r["ttft_s"] for r in rs)))
    for ctx in (32768,):
        if ctx in by_ctx:
            clean.append((by_ctx[ctx][0]["n_prompt_tokens"], st.median(r["ttft_s"] for r in by_ctx[ctx])))
    expected = fit_expected(clean) if len(clean) >= 3 else None
    dec_clean = [st.median(r["decode_tps"] for r in by_ctx[32768])] if 32768 in by_ctx else []

    print(f"clean fit points (ptok, ttft): {clean}")
    hdr = ("ctx | n | ptok | TTFT med [q1,q3] | boot95 | slowdown vs expected [boot95] | decode med | max shared MiB | "
           "max dedicated MiB | raw spill frac | SM clk med (active) | power med W (active) | throttle reasons | max temp")
    print(hdr)
    out = []
    shared_baseline = None
    for ctx in sorted(by_ctx):
        rs = by_ctx[ctx]
        ttft = [r["ttft_s"] for r in rs]
        dec = [r["decode_tps"] for r in rs if r.get("decode_tps")]
        ptok = rs[0]["n_prompt_tokens"]
        sh_max = ded_max = None
        sm_act, pw_act, thr, tmax = [], [], set(), None
        for r in rs:
            t0 = datetime.fromisoformat(r["t_start_utc"])
            t1 = datetime.fromisoformat(r["t_end_utc"])
            w = [x for x in between(win, t0, t1) if x["pid"] == r["server_pid"]]
            if w:
                sh_max = max(sh_max or 0, max(x["sh"] for x in w))
                ded_max = max(ded_max or 0, max(x["ded"] for x in w))
            s = between(smi, t0, t1)
            for x in s:
                tmax = max(tmax or 0, x["temp"])
                if x["util"] and x["util"] > 0 and x["pwr"] < 200:
                    sm_act.append(x["sm"])
                    pw_act.append(x["pwr"])
                    thr.add(x["thr"])
        med = st.median(ttft)
        lo, hi = boot_ci(ttft)
        exp = expected(ptok) if expected else None
        sd = med / exp if exp else None
        sd_ci = ((lo / exp, hi / exp) if exp and lo is not None else (None, None))
        raw = (sh_max / (sh_max + ded_max)) if sh_max is not None and ded_max else None
        out.append({"ctx": ctx, "n": len(rs), "ptok": ptok, "ttft_med": med, "ttft_q1": q(ttft, .25),
                    "ttft_q3": q(ttft, .75), "ttft_ci": (lo, hi), "slowdown": sd, "slowdown_ci": sd_ci,
                    "dec_med": st.median(dec) if dec else None, "sh_max_mib": sh_max / 2 ** 20 if sh_max else None,
                    "ded_max_mib": ded_max / 2 ** 20 if ded_max else None, "raw_frac": raw})
        f = lambda v, d=2: "NA" if v is None else f"{v:.{d}f}"
        print(f"{ctx} | {len(rs)} | {ptok} | {f(med)} [{f(q(ttft,.25))},{f(q(ttft,.75))}] | ({f(lo)},{f(hi)}) | "
              f"{f(sd)} [{f(sd_ci[0])},{f(sd_ci[1])}] | {f(st.median(dec) if dec else None,1)} | "
              f"{f(sh_max/2**20 if sh_max else None,0)} | {f(ded_max/2**20 if ded_max else None,0)} | {f(raw,4)} | "
              f"{f(st.median(sm_act) if sm_act else None,0)} | {f(st.median(pw_act) if pw_act else None,1)} | "
              f"{sorted(thr)} | {f(tmax,0)}")

    base = [o for o in out if o["ctx"] == 32768]
    anchors = [(r["tag"], st.median(x["ttft_s"] for x in calls if x["tag"] == r["tag"]))
               for r in rows if r.get("record") == "server_start" and r.get("kind", "") == "" and False]
    print("\nbaseline_check records:", [(r["block"], round(r["drift_frac"], 3), r["flagged"]) for r in rows
                                        if r.get("record") == "baseline_check"])
    print("skipped:", [(r["ctx"], r["reason"]) for r in rows if r.get("record") == "skipped"])
    print("stop_rule:", [(r["ctx"], round(r["factor"], 2)) for r in rows if r.get("record") == "stop_rule"])
    print("failed calls:", [(r["tag"], r["call"], r["outcome"], (r.get("error") or "")[:80]) for r in rows
                            if r.get("record") == "call" and r.get("outcome") != "ok"])

    spilled = [o for o in out if o["raw_frac"] is not None and o["ctx"] >= 34816 and o["slowdown"]]
    if len(spilled) >= 3:
        xs = [o["raw_frac"] for o in spilled]
        for name, ys in (("slowdown", [o["slowdown"] for o in spilled]),
                         ("ln(slowdown)", [__import__("math").log(o["slowdown"]) for o in spilled])):
            mx, my = st.mean(xs), st.mean(ys)
            sxx = sum((x - mx) ** 2 for x in xs)
            if sxx > 0:
                slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
                icpt = my - slope * mx
                ss_res = sum((y - (icpt + slope * x)) ** 2 for x, y in zip(xs, ys))
                ss_tot = sum((y - my) ** 2 for y in ys)
                print(f"fit {name} = {icpt:.3f} + {slope:.3f} * raw_spill_fraction   R2={1 - ss_res / ss_tot:.3f}  n={len(xs)}")
    Path(prefix + "_analysis.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
