"""Tables for the evo-t2s overnight run, generated from the JSONL only.

Usage: py -3.12 analysis/t2s_overnight_analysis.py <results jsonl> [--section A|B|C|D|all]

Rules: warm-up rows (rep -1) and rows with valid false are excluded from statistics; an effect is reported as real only
when the bootstrap 95% CI of its ratio excludes 1.0; n and the n_reduced flag are printed with every cell.
Bootstrap: 10,000 resamples, seed 12345, numerator and baseline resampled independently.
"""

from __future__ import annotations

import json
import random
import statistics as st
import sys
from collections import Counter, defaultdict

BOOT_N, SEED = 10000, 12345


def q(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def ratio_ci(num, den, invert=False):
    """Bootstrap CI of median(num)/median(den) (or its inverse for rates)."""
    if len(num) < 2 or len(den) < 2:
        return None, None, None
    rng = random.Random(SEED)
    vals = []
    for _ in range(BOOT_N):
        a = st.median(rng.choice(num) for _ in num)
        b = st.median(rng.choice(den) for _ in den)
        vals.append((b / a) if invert else (a / b))
    vals.sort()
    pt = (st.median(den) / st.median(num)) if invert else (st.median(num) / st.median(den))
    return pt, vals[int(0.025 * BOOT_N)], vals[int(0.975 * BOOT_N) - 1]


def f(x, d=2):
    return "NA" if x is None else f"{x:.{d}f}"


def load(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return rows


def good(rows):
    return [r for r in rows if r.get("kind") == "call" and not r.get("warmup") and r.get("valid") and r.get("ttft_s") is not None]


def section_a(rows):
    print("\n## Section A: crossing the iGPU memory budget\n")
    for r in rows:
        if r.get("record") == "budget":
            print("budget candidates (MiB):", r.get("candidates_mib"))
        if r.get("record") == "budget_plan":
            print(f"plan {r['model_id']} [{r.get('budget_label')}] B={r['B_mib']:.0f} weights={r['weights_mib']:.0f} "
                  f"n_ctx*={r['n_ctx_star']} max_ctx={r.get('max_ctx_supported')} beyond_trained={r.get('beyond_trained_context_arm')} "
                  f"result={r.get('result')} grid={r.get('grid')}")
    A = [r for r in rows if r.get("section") == "A"]
    by_model = defaultdict(list)
    for r in A:
        by_model[r.get("model_id")].append(r)
    for mid, rs in by_model.items():
        starts = {}
        for r in rs:
            if r.get("kind") == "start" and not (r.get("yarn_check")):
                starts.setdefault(r["n_ctx"], []).append(r)
        calls = defaultdict(list)
        for r in good(rs):
            calls[r["n_ctx"]].append(r)
        print(f"\n### {mid}\n")
        print("n_ctx | starts ok/total | exceeds budgets | need MiB | shared MiB (max) | load s | TTFT med [q1,q3] | decode med | n | n_reduced | error (first)")
        below = [n for n in sorted(calls) if not any((starts[n][0].get("exceeds_budget") or {}).values())]
        base = [c["ttft_s"] for n in below for c in calls[n]]
        for n in sorted(starts):
            s = starts[n]
            ok = sum(1 for x in s if x.get("valid"))
            ex = (s[0].get("exceeds_budget") or {})
            tt = [c["ttft_s"] for c in calls.get(n, [])]
            dd = [c["decode_tok_s"] for c in calls.get(n, []) if c.get("decode_tok_s")]
            err = next((x.get("error") for x in s if x.get("error")), None)
            shared = max((x.get("shared_usage_mib") or 0 for x in calls.get(n, [])), default=None)
            print(f"{n} | {ok}/{len(s)} | {ex} | {f(s[0].get('need_mib'), 0)} | {f(shared, 0)} | {f(s[0].get('load_s'), 1)} | "
                  f"{f(st.median(tt) if tt else None)} [{f(q(tt, .25))},{f(q(tt, .75))}] | {f(st.median(dd) if dd else None, 1)} | "
                  f"{len(tt)} | {any(c.get('n_reduced') for c in calls.get(n, []))} | {str(err)[:90]}")
        if below and len({c['prompt_tokens'] for n in below for c in calls[n]}) <= 2:
            print("\nslowdown vs below-budget median (same fill), ratio [boot95]:")
            for n in sorted(calls):
                if n in below:
                    continue
                tt = [c["ttft_s"] for c in calls[n]]
                pt, lo, hi = ratio_ci(tt, base)
                print(f"  n_ctx {n}: {f(pt)} [{f(lo)},{f(hi)}]")
        probes = defaultdict(dict)
        for r in rs:
            if r.get("kind") == "probe" and r.get("valid"):
                probes[(r["n_ctx"], r.get("yarn_check"))][r["probe_id"]] = (r.get("score"), (r.get("output") or "").strip()[:40])
        for key in sorted(probes, key=lambda k: (k[0], str(k[1]))):
            sc = [v[0] for v in probes[key].values()]
            print(f"probes n_ctx={key[0]} yarn_check={key[1]}: {sum(1 for x in sc if x == 1.0)}/{len(sc)} correct")
        yc = {k[1]: probes[k] for k in probes if k[1]}
        if "on" in yc and "off" in yc:
            diff = [p for p in yc["on"] if yc["on"][p][1] != yc["off"].get(p, (None, None))[1]]
            print(f"YaRN on vs off: outputs differ on {diff if diff else 'none of the probes'}")


def section_b(rows):
    print("\n## Section B: power coupling and causal cap arm\n")
    for r in rows:
        if r.get("record") in ("knob_control", "cap_arm_skipped", "corunner_control", "powercap_restore"):
            print(r["record"], {k: v for k, v in r.items() if k not in ("ts_utc", "affinity")})
    B = [r for r in good([r for r in rows if r.get("section") == "B"])]
    by = defaultdict(lambda: defaultdict(list))
    for r in B:
        by[r["model_id"]][(r.get("cap") or r.get("proc_throttle_max"), r["co_runner"])].append(r)
    for mid, d in by.items():
        print(f"\n### {mid}\n")
        print("cap | co_runner | n | TTFT med [q1,q3] | slowdown vs none at same cap [boot95] | decode slowdown [boot95] | iGPU MHz med | pkg W med | n_reduced")
        pts = []
        for (cap, co), rs in sorted(d.items(), key=lambda kv: (-(kv[0][0] or 0), kv[0][1])):
            tt = [r["ttft_s"] for r in rs]
            dd = [r["decode_tok_s"] for r in rs if r.get("decode_tok_s")]
            base = d.get((cap, "none")) or d.get((100, "none")) or []
            bt = [r["ttft_s"] for r in base]
            bd = [r["decode_tok_s"] for r in base if r.get("decode_tok_s")]
            pt, lo, hi = ratio_ci(tt, bt) if co != "none" else (1.0, None, None)
            dp, dl, dh = ratio_ci(dd, bd, invert=True) if co != "none" else (1.0, None, None)
            mh = [r["igpu_mhz"] for r in rs if r.get("igpu_mhz")]
            pw = [r["pkg_power_w"] for r in rs if r.get("pkg_power_w")]
            if mh and pw:
                pts.append((st.median(pw), st.median(mh)))
            print(f"{cap} | {co} | {len(rs)} | {f(st.median(tt))} [{f(q(tt, .25))},{f(q(tt, .75))}] | {f(pt)} [{f(lo)},{f(hi)}] | "
                  f"{f(dp)} [{f(dl)},{f(dh)}] | {f(st.median(mh) if mh else None, 0)} | {f(st.median(pw) if pw else None, 1)} | "
                  f"{any(r.get('n_reduced') for r in rs)}")
        if len(pts) >= 4:
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            print(f"iGPU MHz vs package W across conditions: Pearson r = {st.correlation(xs, ys):.2f} (n={len(pts)} conditions)")


def section_cd(rows, backend):
    label = "C" if backend == "vulkan" else "D"
    print(f"\n## Section {label}: memory lock, backend {backend}\n")
    valid = {r["item_id"]: r for r in rows if r.get("record") == "level_validity"}
    C = [r for r in rows if r.get("section") in ("C", "D") and r.get("backend") == backend]
    starts = [r for r in C if r.get("kind") == "start"]
    by = defaultdict(list)
    for s in starts:
        by[(s["model_id"], bool(s["mmap"]), s["mem_headroom_gb"])].append(s)
    print("model | mmap | headroom GB | starts (valid level / ok load) | load s med | first error | TTFT med | decode med | probes correct | pages in max | hard faults max | Available min MB")
    for key in sorted(by, key=lambda k: (k[0], not k[1], -(k[2] if k[2] is not None else 99))):
        ss = by[key]
        vs = [s for s in ss if (valid.get(s["item_id"]) or {}).get("valid", True) and not s.get("anchor")]
        ok = [s for s in vs if s.get("valid")]
        ids = {s["item_id"] for s in vs}
        calls = [r for r in good(C) if r["item_id"] in ids]
        pr = [r for r in C if r.get("kind") == "probe" and r.get("valid") and r["item_id"] in ids]
        err = next((s.get("error") for s in vs if s.get("error")), None)
        tt = [r["ttft_s"] for r in calls]
        dd = [r["decode_tok_s"] for r in calls if r.get("decode_tok_s")]
        print(f"{key[0]} | {'on' if key[1] else 'off'} | {key[2]} | {len(vs)} valid of {len(ss)} ({len(ok)} loaded) | "
              f"{f(st.median([s['load_s'] for s in ok]) if ok else None, 1)} | {str(err)[:70]} | {f(st.median(tt) if tt else None, 1)} | "
              f"{f(st.median(dd) if dd else None, 1)} | {sum(1 for r in pr if r.get('score') == 1.0)}/{len(pr)} | "
              f"{f(max((s.get('pages_input_per_s') or 0 for s in ok), default=None), 0)} | "
              f"{f(max((s.get('hard_faults_per_s') or 0 for s in ok), default=None), 0)} | "
              f"{f(min((s.get('avail_mb_min') for s in ok if s.get('avail_mb_min') is not None), default=None), 0)}")
    inval = [v for v in valid.values() if not v.get("valid")]
    print(f"\nlevels marked invalid (lock not held / balloon dead): {len(inval)}")
    for v in inval[:20]:
        print("  ", v["item_id"], v.get("reason"))
    on = [s for s in starts if s.get("mmap") and s.get("valid")]
    off = [s for s in starts if not s.get("mmap") and s.get("valid")]
    for lab, xs in (("mmap on", on), ("mmap off", off)):
        pv = [s.get("private_mib_at_load") for s in xs if s.get("private_mib_at_load")]
        ws = [s.get("working_set_mib_at_load") for s in xs if s.get("working_set_mib_at_load")]
        print(f"{lab}: private MiB at load median {f(st.median(pv) if pv else None, 0)}, working set median {f(st.median(ws) if ws else None, 0)}, n={len(xs)}")


def main():
    path = sys.argv[1]
    sec = sys.argv[sys.argv.index("--section") + 1] if "--section" in sys.argv else "all"
    rows = load(path)
    for r in rows:
        if r.get("record") == "schedule":
            print("schedule: kept", r["kept"], "planned h", f(r["planned_s"] / 3600), "trimmed", len(r["dropped"]))
        if r.get("record") == "run_end":
            print("run end:", r.get("note"))
    if sec in ("A", "all"):
        section_a(rows)
    if sec in ("B", "all"):
        section_b(rows)
    if sec in ("C", "all"):
        section_cd(rows, "vulkan")
    if sec in ("D", "all"):
        section_cd(rows, "sycl")


if __name__ == "__main__":
    main()
