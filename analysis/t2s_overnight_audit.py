"""Data-quality audit of an evo-t2s overnight or A-mech JSONL: row counts, invalid rows, errors, trims, join-key completeness.

Usage: py -3.12 analysis/t2s_overnight_audit.py <results jsonl>
Prints plain tables from the file only. Nothing is written.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict

JOIN_KEYS = ["hw_id", "cpu_name", "gpu_name", "gpu_driver_version", "backend", "llama_build", "model_id", "model_sha256", "quant",
             "kv_type", "flash_attn", "mmap", "n_ctx", "prompt_tokens", "co_runner", "proc_throttle_max", "mem_headroom_gb", "section",
             "rep", "seed", "git_sha", "script_sha", "load_s", "ttft_s", "decode_tok_s", "e2e_s", "output", "score", "igpu_mhz",
             "pkg_power_w", "shared_usage_mib", "total_committed_mib", "pages_input_per_s", "hard_faults_per_s", "error"]


def main():
    rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
    print(f"rows: {len(rows)}")
    print("record types:", dict(Counter(r.get("record") for r in rows if r.get("record"))))
    data = [r for r in rows if r.get("kind") in ("call", "probe", "start")]
    print(f"data rows (kind call/probe/start): {len(data)}")
    by = Counter((r.get("section"), r.get("kind")) for r in data)
    print("by section/kind:", dict(sorted(by.items(), key=lambda kv: str(kv[0]))))
    print("warm-up rows:", sum(1 for r in data if r.get("warmup")))
    inval = [r for r in data if r.get("valid") is False]
    print(f"rows with valid false: {len(inval)}")
    for r in inval:
        print("  ", r.get("section"), r.get("kind"), r.get("item_id"), r.get("n_ctx"), str(r.get("error"))[:110])
    print("item errors:", [(r["item_id"], r["error"][:100]) for r in rows if r.get("record") == "item_error"])
    print("n_reduced calls:", sum(1 for r in data if r.get("n_reduced")))
    missing = defaultdict(int)
    keys_absent = defaultdict(int)
    for r in data:
        for k in JOIN_KEYS:
            if k not in r:
                keys_absent[k] += 1
            elif r[k] is None:
                missing[k] += 1
    print("join keys absent from a row (count):", dict(keys_absent) or "none")
    calls = [r for r in data if r.get("kind") == "call" and not r.get("warmup")]
    print("call rows null counts (of %d):" % len(calls), {k: sum(1 for r in calls if r.get(k) is None) for k in
          ("ttft_s", "decode_tok_s", "igpu_mhz", "pkg_power_w", "shared_usage_mib", "total_committed_mib", "pages_input_per_s", "hard_faults_per_s", "llama_build")})
    print("git_sha values:", dict(Counter(r.get("git_sha") for r in data)))
    print("script_sha values:", dict(Counter(r.get("script_sha") for r in data)))
    print("cpu_name values:", dict(Counter(r.get("cpu_name") for r in data)))
    print("gate_released_by:", dict(Counter(r.get("gate_released_by") for r in data if r.get("kind") == "call")))
    for r in rows:
        if r.get("record") == "schedule":
            d = r.get("dropped") or []
            c = Counter(x["item_id"].split("_")[0] + "/" + (x["item_id"].split("_")[1] if "_" in x["item_id"] else "") for x in d)
            print(f"schedule kept {r['kept']} planned {r['planned_s'] / 3600:.2f} h budget {r['budget_s'] / 3600:.2f} h; trimmed {len(d)}: {dict(c)}")
        if r.get("record") == "run_end":
            print("run_end:", r.get("note"), r.get("ts_utc"))
        if r.get("record") == "backfill":
            print("backfill:", r.get("after"), [p["item_id"] for p in r.get("pulled", [])])
    print("item_done:", sum(1 for r in rows if r.get("record") == "item_done"))


if __name__ == "__main__":
    main()
