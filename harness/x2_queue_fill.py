"""evo-x2 queue fill of 2026-10-21 (docs/X2_QUEUE_PLAN.md): the queue entries for the jobs pre-registered in
docs/FINDINGS.md on 2026-10-20/21, and the one read-modify-write that appends them.

Run on evo-x2 from a side directory (C:\\apu\\ovn\\qfill), so t2s_queue is imported from the parent C:\\apu\\ovn:

    py -3.12 C:\\apu\\ovn\\qfill\\x2_queue_fill.py --append <job id> [<job id> ...]
    py -3.12 C:\\apu\\ovn\\qfill\\x2_queue_fill.py --show            (prints the entries; no queue access)

--append: t2s_queue.set_pause with a reason naming its release, one read of queue_state.json inside the pause, the
named entries appended at the end in the given order (an id already present, or a gate naming a job that is neither
already in the queue nor appended earlier in the same call, is a ValueError and nothing is written), one write,
clear_pause() in this function's finally block, then a check that the pause flag is gone. Nothing here launches a
job: the watchdog launches the first pending entry on its next tick (t2s_queue.launch_next clears queue_empty.flag).

Every entry's note says it releases itself via tq.advance() on exit and sets no pause.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
if not (_here / "t2s_queue.py").exists():
    sys.path.insert(0, str(_here.parent))

PY = sys.executable   # the machine's own interpreter when run there (no user path in the source)
RESULTS = r"C:\apu\ovn\results"
RELEASE = "Releases itself via tq.advance() on exit; sets no pause."

ENTRIES = {
    "x2_num_keep_qwen3_14b": {
        "cmd": [PY, "numkeep\\x2_r2_sidecar.py", "--mode", "real", "--call2-tools", "off", "--models", "qwen3:14b",
                "--tiers", "4096", "--seeds", "20260901,20260902,20260903", "--rules-from",
                f"{RESULTS}\\x2_r2_validation_v2.jsonl", "--require-validation-gates", "--num-keep", "system",
                "--out", f"{RESULTS}\\x2_num_keep_qwen3_14b.jsonl"],
        "note": "num_keep guardrail flag test for qwen3:14b, pre-registered in docs/FINDINGS.md 2026-10-20: the "
                "llama3.1:8b test's command and side directory (numkeep) with --models qwen3:14b, num_ctx 4096, "
                "3 seeds x 40 turns, verdict one-line fix or not. Hours: register row x2-queue-fill-1020-hours. "
                + RELEASE,
    },    "x2_evalset_tasks_val_v2_qwen3_8b": {
        "projected_duration_h": 0.22,
        "cmd": [PY, "evalv2q8\\x2_r2_evalset.py", "--design", "v2", "--mode", "validation", "--workload", "evalset",
                "--guardrails", "off", "--models", "qwen3:8b", "--out",
                f"{RESULTS}\\x2_evalset_tasks_val_v2_qwen3_8b.jsonl"],
        "note": "B3 v2 tasks validation for qwen3:8b, pre-registered in docs/FINDINGS.md 2026-10-21 (protocol "
                "consistency extended to qwen3:8b): the x2_evalset_tasks_val_v2 plan with --models qwen3:8b. ESTIMATE "
                "0.22 h (register row x2-queue-fill-1020-hours). " + RELEASE,
    },
    "x2_evalset_tasks_real_v2_qwen3_8b": {
        "projected_duration_h": 2.11,
        "gate": {"requires_done": "x2_evalset_tasks_val_v2_qwen3_8b"},
        "cmd": [PY, "evalv2q8\\x2_r2_evalset.py", "--design", "v2", "--mode", "real", "--workload", "evalset",
                "--guardrails", "off", "--models", "qwen3:8b", "--plan", "tasks_v2_qwen3_8b", "--preflight",
                "v2_strict", "--rules-from", f"{RESULTS}\\x2_evalset_tasks_val_v2_qwen3_8b.jsonl", "--out",
                f"{RESULTS}\\x2_evalset_tasks_real_v2_qwen3_8b.jsonl"],
        "note": "B3 v2 task real run for qwen3:8b, pre-registered in docs/FINDINGS.md 2026-10-21: tasks_v2 tiers and "
                "seeds; queue gate requires_done x2_evalset_tasks_val_v2_qwen3_8b, and the v2_strict preflight refuses "
                "unless qwen3:8b passes every gate of that validation (no waiver). ESTIMATE 2.11 h (register row "
                "x2-queue-fill-1020-hours). " + RELEASE,
    },
    "x2_r2_qwen3_8b_seeds_06_10": {
        "projected_duration_h": 2.19,
        "cmd": [PY, "x2_r2_agent.py", "--mode", "real", "--plan", "strong", "--call2-tools", "off", "--models",
                "qwen3:8b", "--tiers", "8192,4096", "--seeds", "20260906,20260907,20260908,20260909,20260910",
                "--allow-new-seeds", "--rules-from", f"{RESULTS}\\x2_r2_validation_v2b.jsonl",
                "--require-validation-gates", "--per-model-refusal", "--order", "seed_major", "--out",
                f"{RESULTS}\\x2_r2_qwen3_8b_seeds_06_10.jsonl"],
        "note": "qwen3:8b at num_ctx 8192 vs 4096, seeds 20260906-10 (pooled with v1b's five), pre-registered in "
                "docs/FINDINGS.md 2026-10-21: Fisher exact on 10 vs 10 gated silent sessions. ESTIMATE 2.19 h "
                "(register row x2-queue-fill-1020-hours). " + RELEASE,
    },
}


def entry(job_id: str) -> dict:
    e = ENTRIES[job_id]
    out = {"id": job_id, "status": "pending", "projected_duration_h": e.get("projected_duration_h"),
           "cmd": list(e["cmd"]), "note": e["note"]}
    if e.get("gate"):
        out["gate"] = dict(e["gate"])
    return out


def append_entries(items: list[dict], job_ids: list[str]) -> list[dict]:
    """Pure: the queue with the named entries appended in order. Every other entry is kept as it is."""
    items = list(items)
    have = {it["id"] for it in items}
    for j in job_ids:
        if j in have:
            raise ValueError(f"{j} is already in the queue")
        e = entry(j)
        req = (e.get("gate") or {}).get("requires_done")
        if req and req not in have:
            raise ValueError(f"{j} is gated on {req}, which is not in the queue ahead of it")
        items.append(e)
        have.add(j)
    return items


def queue_append(job_ids: list[str], log=print) -> list[dict]:
    import t2s_queue as tq
    tq.set_pause(f"x2_queue_fill --append {','.join(job_ids)}: one read-modify-write appending these entries; "
                 "released by clear_pause() in this same script's finally block within seconds")
    try:
        new = append_entries(tq.read_queue(), job_ids)
        tq.write_queue(new)
    finally:
        tq.clear_pause()
    if tq.is_paused() is not None:
        raise RuntimeError("pause flag still present after clear_pause()")
    ids = [it["id"] for it in new]
    log(json.dumps({"positions": {j: ids.index(j) for j in job_ids}, "pause_flag_present": False,
                    "pending_or_running": [(k, it["id"], it["status"]) for k, it in enumerate(new)
                                           if it.get("status") in ("pending", "running")]}))
    return new


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--append", nargs="+", default=None)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args(argv)
    if args.show:
        print(json.dumps({j: entry(j) for j in ENTRIES}, indent=1))
        return 0
    if args.append:
        unknown = [j for j in args.append if j not in ENTRIES]
        if unknown:
            ap.error(f"unknown job id(s) {unknown}")
        queue_append(args.append)
        return 0
    ap.error("nothing to do (--append or --show)")


if __name__ == "__main__":
    raise SystemExit(main())
