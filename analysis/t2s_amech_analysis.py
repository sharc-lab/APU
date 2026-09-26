"""Tables for the evo-t2s A-mech run (rope check, Vulkan limits, n_ctx bisection, runtime-policy arms, filled-call check).

Usage: py -3.12 analysis/t2s_amech_analysis.py <t2s_amech_*.jsonl> [--quote <server log path>]

Rule: an effect or a boundary is quoted only from rows in the file. Warm-up rows (rep -1, -2) are excluded from timing
statistics. The bisection uses the last record per (label, n_ctx) of type bisect_probe or the start row (first row wins
when a config was probed twice, the repeat is reported separately).
"""

from __future__ import annotations

import json
import re
import statistics as st
import sys
from collections import defaultdict

B = 47865.0
MARGIN = 1024.0


def f(x, d=0):
    return "NA" if x is None else f"{x:.{d}f}"


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def rope(rows):
    print("\n## Step 0: is the 32B rope scaling YaRN?\n")
    for r in rows:
        if r.get("kind") == "rope_probe":
            print(f"- {r['item_id']}: flags {r.get('rope_flags')}; process cmdline tail: ...{(r.get('cmdline_process') or '')[-140:]}")
            print(f"  output: {(r.get('output') or '')[:70]!r}; first-token top-5 (token, logprob): {r.get('first_top')}")
    for r in rows:
        if r.get("record") == "rope_verdict":
            print("\nverdict:", json.dumps({k: v for k, v in r.items() if k not in ("record", "ts_utc")}))


def vulkan(rows):
    print("\n## Vulkan device limits (vulkaninfo)\n")
    for r in rows:
        if r.get("record") == "vulkan_limits":
            for k in ("device_name", "maxMemoryAllocationSize", "maxBufferSize", "maxStorageBufferRange", "maxMemoryAllocationCount"):
                v = r.get(k)
                extra = ""
                if isinstance(v, str) and v.startswith("0x"):
                    extra = f" = {int(v, 16) / 2 ** 20:.0f} MiB"
                print(f"- {k}: {v}{extra}")
            for h in r.get("heaps") or []:
                print(f"- heap {h['index']}: size {h['size_mib']:.0f} MiB, budget {h['budget_mib']:.0f} MiB, usage {h['usage_mib']:.0f} MiB, flags {h['flags']}")
            if not r.get("heaps"):
                print("- heaps (raw block):")
                for l in (r.get("memory_properties_text") or [])[:40]:
                    print("   ", l)


def bisect(rows):
    print("\n## Bisection (server start only, no prompts)\n")
    probes = defaultdict(dict)
    t_v2 = min((r["ts_utc"] for r in rows if r.get("record") == "bisect_probe"), default="9")
    n_v1 = sum(1 for r in rows if r.get("phase") == "bisect" and r.get("kind") == "start" and r["ts_utc"] < t_v2)
    print(f"first attempt (before the guard n_ctx-cap fix, {n_v1} start rows) is superseded and excluded below; its 8B and 30B-A3B rows above "
          "n_ctx 131072 and 262144 were misclassified as failures by the stale-server guard (context was created).\n")
    rows = [r for r in rows if not (r.get("ts_utc", "9") < t_v2 and (r.get("phase") == "bisect" or r.get("record") == "bisect_result"))]
    for r in rows:
        if r.get("phase") == "bisect" and r.get("kind") == "start":
            probes[r["bisect_label"]].setdefault(r["n_ctx"], []).append(r)
    caps = {(r["label"], r["n_ctx"]): r for r in rows if r.get("record") == "bisect_probe"}
    for label, d in probes.items():
        print(f"\n### {label}\n")
        print("n_ctx | started | projected MiB | logged buffers MiB (incl. host) | props n_ctx cap | first error")
        for n in sorted(d):
            for r in d[n]:
                cp = caps.get((label, n)) or {}
                ok = bool(r.get("valid")) or bool(cp.get("started"))
                print(f"{n} | {ok} | {f(r.get('projected_mib'))} | "
                      f"{f(sum(x for x in (r.get('model_buffer_mib_log'), r.get('kv_buffer_mib_log'), r.get('compute_buffer_mib_log')) if x) or None, 0)} | "
                      f"{cp.get('props_n_ctx_cap')} | {str(r.get('error'))[:70] if not ok else ''}")
    print()
    for r in rows:
        if r.get("record") == "bisect_result":
            if r.get("error"):
                print(r["label"], "ERROR", r["error"])
                continue
            a, b = r.get("projected_mib_last_ok"), r.get("projected_mib_first_fail")
            print(f"boundary {r['label']}: last ok n_ctx {r['last_ok_n_ctx']} (projected {f(a)} MiB = {f(100 * a / B, 2)}% of B), first fail "
                  f"{r['first_fail_n_ctx']} (projected {f(b)} MiB = {f(100 * b / B, 2)}% of B); step {r['step_tokens']} tokens = "
                  f"{r['step_mib_of_kv']:.0f} MiB of KV; fit target B-1024 = {B - MARGIN:.0f}; beyond trained ctx {r.get('beyond_trained_ctx')}; "
                  f"probes {r['n_probes']}; first-fail vk {r.get('first_fail_vk')} alloc {r.get('first_fail_alloc')}")
    for r in rows:
        if r.get("record") == "bisect_repeat":
            print(f"repeat {r['label']} n_ctx {r['n_ctx']}: ok {r['ok']} (first time {r['first_time_ok']}), projected {f(r['projected_mib'])}")
    res = [r for r in rows if r.get("record") == "bisect_result" and not r.get("error")]
    if res:
        pl = [r["projected_mib_last_ok"] for r in res]
        pf = [r["projected_mib_first_fail"] for r in res]
        print(f"\nacross {len(res)} model(s): projected MiB at last ok {f(min(pl))} to {f(max(pl))}; at first fail {f(min(pf))} to {f(max(pf))}; "
              f"B = {f(B)}. Highest passing / B = {max(pl) / B:.4f}; lowest failing / B = {min(pf) / B:.4f}")


def arms(rows):
    print("\n## Runtime-policy arms (first three failing 32B grid contexts) and the last-passing filled check\n")
    starts = [r for r in rows if r.get("phase") in ("arms", "fill") and r.get("kind") == "start"]
    print("n_ctx | arm | -ngl | -fit | started | layers GPU/total | projected MiB | free MiB | load s | first error")
    for r in starts:
        print(f"{r['n_ctx']} | {r.get('arm')} | {r.get('ngl')} | {r.get('fit')} | {r.get('valid')} | {r.get('layers_gpu')}/{r.get('layers_total')} | "
              f"{f(r.get('projected_mib'))} | {f(r.get('fit_free_mib'))} | {f(r.get('load_s'), 1)} | {str(r.get('error'))[:60]}")
    for r in starts:
        print(f"\n[{r['item_id']}] breakdown: {r.get('breakdown')}\n   fit: {r.get('fit_lines')}\n   vk: {r.get('vk_errors')} alloc_failed: {r.get('alloc_failed')}")
    print("\nfilled calls:\n")
    print("item | fill used (90% target) | fill_capped | est call s | n | TTFT med [min,max] | decode med | probes correct | shared MiB max")
    by = defaultdict(list)
    for r in rows:
        if r.get("phase") in ("arms", "fill") and r.get("kind") in ("call", "probe") and r.get("item_id"):
            by[r["item_id"]].append(r)
    for item, rs in by.items():
        calls = [r for r in rs if r["kind"] == "call" and not r.get("warmup") and r.get("valid") and r.get("ttft_s") is not None]
        pr = [r for r in rs if r["kind"] == "probe" and r.get("valid")]
        tt = [r["ttft_s"] for r in calls]
        dd = [r["decode_tok_s"] for r in calls if r.get("decode_tok_s")]
        r0 = rs[0]
        print(f"{item} | {r0.get('fill_used')} ({r0.get('fill_target_90pct')}) | {r0.get('fill_capped')} | {f(r0.get('fill_est_call_s'))} | {len(tt)} | "
              f"{f(st.median(tt) if tt else None, 1)} [{f(min(tt) if tt else None, 1)},{f(max(tt) if tt else None, 1)}] | {f(st.median(dd) if dd else None, 2)} | "
              f"{sum(1 for r in pr if r.get('score') == 1.0)}/{len(pr)} | {f(max((r.get('shared_usage_mib') or 0 for r in rs), default=None))}")
    for r in rows:
        if r.get("record") == "fill_feasibility":
            print("\nfill feasibility:", json.dumps({k: v for k, v in r.items() if k not in ("record", "ts_utc")}))


def quote(path):
    txt = open(path, encoding="utf-8", errors="replace").read()
    print(re.sub(r"\x1b\[[0-9;]*m", "", txt))


def main():
    if "--quote" in sys.argv:
        return quote(sys.argv[sys.argv.index("--quote") + 1])
    rows = load(sys.argv[1])
    for r in rows:
        if r.get("record") == "run_end":
            print("run end:", r.get("note"), r.get("ts_utc"))
    rope(rows)
    vulkan(rows)
    bisect(rows)
    arms(rows)


if __name__ == "__main__":
    main()
