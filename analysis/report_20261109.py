"""Tables for docs/reports/2026-11-09.md, from result files only (no network, no model call).

  py -3.12 analysis/report_20261109.py            prints every table as Markdown

Sources: results/x2_longtask_toolfix_v1_llama33_70b.jsonl, results/x2_longtask_scale_v1_llama33_70b.jsonl (job 98),
results/x2_longtask_tokens_v1_*.jsonl, results/x2_mem_20261109/ (5 s Windows counters and the copied Ollama server
logs), results/x2_post_run_check_live_v1.jsonl. Per call: prefill tok/s = prompt_eval_count / prompt_eval_duration
(Ollama counts only tokens it evaluated, not the cached prefix); decode tok/s = completion tokens / eval duration;
TTFT = load + prompt-eval duration (non-streaming request, so TTFT is derived, not observed).
"""
from __future__ import annotations

import csv
import json
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
RES = REPO / "results"
MEM = RES / "x2_mem_20261109"
FILES = {"toolfix 70B": "x2_longtask_toolfix_v1_llama33_70b.jsonl", "job 98 70B": "x2_longtask_scale_v1_llama33_70b.jsonl",
         "tokens 32B": "x2_longtask_tokens_v1_qwen3_32b.jsonl", "tokens 70B": "x2_longtask_tokens_v1_llama33_70b.jsonl"}


def rows(name):
    p = RES / FILES[name]
    if not p.exists():
        return []
    out = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    att = [r for r in out if r.get("record") == "run_start"]
    if att:
        last = att[-1]["attempt_id"]
        out = [r for r in out if r.get("attempt_id") == last]
    return out


def q(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def spread(xs, fmt="{:.1f}"):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "n/a"
    return (f"{fmt.format(q(xs, .5))} [IQR {fmt.format(q(xs, .25))} to {fmt.format(q(xs, .75))}; "
            f"min {fmt.format(min(xs))}, max {fmt.format(max(xs))}; n={len(xs)}]")


def call_perf(tasks):
    """Prefill is DERIVED: Ollama's prompt_eval_count is the whole prompt while prompt_eval_duration covers only the
    part not in its prefix cache, so prefill tok/s = (this call's count - previous call's count - previous output
    tokens) / prompt-eval seconds, for calls after the first of a task (assumes the earlier prefix was cached)."""
    pf, dc, tt, wl = [], [], [], []
    for t in tasks:
        prev = None
        for c in t["calls"]:
            pe, ped = c.get("prompt_eval_count_info_only"), c.get("prompt_eval_duration_s")
            ct, ed = c.get("completion_tokens"), c.get("eval_duration_s")
            if prev and pe and ped:
                delta = pe - prev[0] - (prev[1] or 0)
                if delta > 0:
                    pf.append(delta / ped)
            prev = (pe, ct) if pe else None
            if ct and ed:
                dc.append(ct / ed)
            if ped is not None:
                tt.append((c.get("load_duration_s") or 0) + ped)
            wl.append(c.get("duration_s"))
    return {"prefill": pf, "decode": dc, "ttft": tt, "call_wall": wl}


def task_table(tasks, classify):
    out = ["| seed | task | prompt tokens max (est / server) | output tokens | calls / ref | result | class | min |",
           "|---|---|---|---|---|---|---|---|"]
    for t in tasks:
        est = round((t.get("max_prompt_tokens_est") or 0) * 1.2)
        out.append(f"| {t['seed']} | {t['task_id']} | {est:,} / {t.get('max_prompt_eval_count')} | "
                   f"{sum(x or 0 for x in t['completion_tokens']):,} | {t['n_calls']} / {t['ref_steps']} | "
                   f"{'pass' if t['correct'] else 'fail'} | {classify(t)} | {t['task_wall_s'] / 60:.1f} |")
    return "\n".join(out)


def server_log_memory():
    """Per copied server log: model, n_ctx, KV buffer, compute buffer, layers offloaded (llama.cpp lines)."""
    out = []
    for p in sorted((MEM / "serverlogs").glob("ollama_serve_*.log")):
        s = p.read_text(encoding="utf-8", errors="replace")
        d = {"log": p.name}
        for k, pat in (("n_ctx", r"n_ctx\s*=\s*(\d+)"), ("kv_mib", r"KV self size\s*=\s*([\d.]+) MiB"),
                       ("kv_buf_mib", r"KV buffer size\s*=\s*([\d.]+) MiB"),
                       ("compute_mib", r"compute buffer size\s*=\s*([\d.]+) MiB"),
                       ("offloaded", r"offloaded (\d+/\d+) layers"), ("model", r"general\.name\s+str\s*=\s*(.+)"),
                       ("type_k", r"type_k = '?(\w+)"), ("n_batch", r"n_batch\s*=\s*(\d+)"),
                       ("n_ubatch", r"n_ubatch\s*=\s*(\d+)"), ("backend", r"(ROCm|Vulkan|HIP|CUDA)\d* ?(?:devices|0)")):
            m = re.findall(pat, s)
            d[k] = m[-1].strip() if m else None
        out.append(d)
    return out


def typeperf_peaks(t0: datetime, t1: datetime):
    """Peak GPU dedicated and shared usage (GB, summed over adapters) and min available MB in [t0, t1]."""
    best = {"dedicated_gb": 0.0, "shared_gb": 0.0, "avail_min_mb": None, "n": 0}
    for p in sorted(MEM.glob("typeperf_5s_*.csv")):
        with open(p, newline="", encoding="utf-8-sig", errors="replace") as fh:
            r = csv.reader(fh)
            head = next(r, None)
            if not head:
                continue
            ded = [i for i, h in enumerate(head) if "Dedicated Usage" in h]
            sha = [i for i, h in enumerate(head) if "Shared Usage" in h]
            av = [i for i, h in enumerate(head) if "Available MBytes" in h]
            for row in r:
                try:
                    ts = datetime.strptime(row[0], "%m/%d/%Y %H:%M:%S.%f").astimezone(timezone.utc)
                except (ValueError, IndexError):
                    continue
                if not (t0 <= ts <= t1):
                    continue

                def f(i):
                    try:
                        return float(row[i])
                    except (ValueError, IndexError):
                        return 0.0
                best["n"] += 1
                best["dedicated_gb"] = max(best["dedicated_gb"], sum(f(i) for i in ded) / 1e9)
                best["shared_gb"] = max(best["shared_gb"], sum(f(i) for i in sha) / 1e9)
                a = min((f(i) for i in av), default=None)
                if a is not None:
                    best["avail_min_mb"] = a if best["avail_min_mb"] is None else min(best["avail_min_mb"], a)
    return best


def window(rs):
    ts = [datetime.fromisoformat(r["ts_utc"]) for r in rs if r.get("ts_utc")]
    return (min(ts), max(ts)) if ts else (None, None)


def main():
    import numbers_register as nr
    for name in FILES:
        rs = rows(name)
        tasks = [r for r in rs if r.get("record") == "longtask_task"]
        if not tasks:
            print(f"\n## {name}: no task rows\n")
            continue
        start = next((r for r in rs if r.get("record") == "run_start"), {})
        warm = [r for r in rs if r.get("record") == "warmup"]
        end = next((r for r in rs if r.get("record") == "run_end"), None)
        print(f"\n## {name} ({FILES[name]})\n")
        print(f"model {start.get('model_id')}, arms {start.get('arm_order')}, seeds {start.get('seeds')}, tool_prompt "
              f"{start.get('tool_prompt', 'ollama (field absent)')}, task_set {start.get('task_set', 'steps')}, "
              f"ollama_bin {start.get('ollama_bin')}, max_tokens_per_call {start.get('max_tokens_per_call')}, "
              f"keep_alive {start.get('keep_alive')}; warmups: " +
              "; ".join(f"seed {w.get('seed')} {w.get('arm_id')} loaded {w.get('loaded_context')} kv_check "
                        f"{(w.get('kv_check') or {}).get('valid')}" for w in warm) +
              f"; run_end: {end.get('note') if end else 'NONE (not finished)'}")
        for arm in sorted({t["arm"] for t in tasks}):
            ts = [t for t in tasks if t["arm"] == arm]
            cls = (nr.long_task_failure_class if not name.startswith("tokens")
                   else (lambda t, _m={(r["model"], r["arm"], r["task_id"]): f"{r['class']} (rule) / {r['class_server']} (server)"
                                    for r in nr.longtask_tokens_rows(REPO)}:
                         "correct" if t["correct"] else _m.get((t["model_id"], t["arm"], t["task_id"]), "?")))
            print(f"\n### arm {arm} (loaded windows {sorted({t['loaded_context'] for t in ts})})\n")
            print(task_table(ts, cls))
            p = call_perf(ts)
            print(f"\nper call: prefill tok/s {spread(p['prefill'])}; decode tok/s {spread(p['decode'])}; TTFT s "
                  f"{spread(p['ttft'], '{:.2f}')}; call wall s {spread(p['call_wall'], '{:.2f}')}; task wall min "
                  f"{spread([t['task_wall_s'] / 60 for t in ts])}")
            wu = [w for w in warm if w.get("arm_id") == ts[0]["arm_id"]]
            t0, t1 = window(wu[:1] + ts)
            if t0 and MEM.exists():
                pk = typeperf_peaks(t0, t1)
                print(f"Windows counters over the arm ({t0:%H:%M}Z to {t1:%H:%M}Z, {pk['n']} samples at 5 s): peak GPU "
                      f"dedicated {pk['dedicated_gb']:.2f} GB, peak shared {pk['shared_gb']:.2f} GB, min available "
                      f"{pk['avail_min_mb']} MB")
    print("\n## server logs (copied per server start)\n")
    for d in server_log_memory():
        print("- " + ", ".join(f"{k} {v}" for k, v in d.items()))


if __name__ == "__main__":
    main()
