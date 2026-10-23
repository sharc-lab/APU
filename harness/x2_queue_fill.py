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
# register row x2-llama33-70b-b3v2-hours (ESTIMATE), copied here because the queue entry is built on evo-x2
L70_HOURS = {"val": 1.76, "val_high": 2.07, "real": 0.97, "real_high": 1.25}
# register row long-task-local-hours (ESTIMATE), copied here for the same reason
LT_HOURS = {"point": 9.63, "high": 16.92}

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
    # ── queue fill of 2026-10-22 (docs/X2_QUEUE_PLAN.md; hours: register row x2-queue-fill-1022-hours) ──
    "x2_b5_fa_control": {
        "projected_duration_h": 4.85,
        "cmd": [PY, "b5fa\\policy_compare.py", "--platform", "evo-x2", "--fa-control", "--out",
                f"{RESULTS}\\x2_policy_compare_fa_control.jsonl"],
        "note": "B5 flash-attention control (policy i: f16 KV + flash attention at L(f16), llama3.1:8b and qwen3:14b, "
                "seeds 20260901-03) and the llama3.1:8b f16 recall rerun (1800 s timeout), pre-registered in "
                "docs/FINDINGS.md 2026-10-22 (a307ca8). ESTIMATE 4.85 h (high 13.92 h; register row "
                "x2-queue-fill-1022-hours). " + RELEASE,
    },
    "x2_b5_simcap_8192": {
        "projected_duration_h": 33.08,
        "cmd": [PY, "b5fa\\policy_compare.py", "--platform", "evo-x2", "--window-cap", "8192", "--seeds",
                "20260901,20260902,20260903,20260904,20260905", "--out",
                f"{RESULTS}\\x2_policy_compare_simcap8192.jsonl"],
        "note": "B5 under a simulated device window of 8192 (simulated constraint), all policies a-h, llama3.1:8b and "
                "qwen3:14b, seeds 20260901-05, pre-registered in docs/FINDINGS.md 2026-10-22. ESTIMATE 33.08 h "
                "(register row x2-queue-fill-1022-hours). " + RELEASE,
    },
    "x2_b5_simcap_4096": {
        "projected_duration_h": 26.11,
        "cmd": [PY, "b5fa\\policy_compare.py", "--platform", "evo-x2", "--window-cap", "4096", "--seeds",
                "20260901,20260902,20260903,20260904,20260905", "--out",
                f"{RESULTS}\\x2_policy_compare_simcap4096.jsonl"],
        "note": "B5 under a simulated device window of 4096 (simulated constraint), all policies a-h, llama3.1:8b and "
                "qwen3:14b, seeds 20260901-05, pre-registered in docs/FINDINGS.md 2026-10-22. ESTIMATE 26.11 h "
                "(register row x2-queue-fill-1022-hours). " + RELEASE,
    },
    "x2_r2_v1b_seeds_06_10": {
        "projected_duration_h": 30.44,
        "cmd": [PY, "r2v1b\\x2_r2_v1b_seeds.py"],
        "note": "R2 at 10 seeds on the v1b harness blob (a4c4a928): reproduction check first (llama3.1:8b num_ctx 4096 "
                "seed 20260904 vs x2_r2_real_v1b, identical final text on every turn, else stop), then llama3.1:8b and "
                "qwen3:14b at default/16384/8192/4096, seeds 20260906-10; pre-registered in docs/FINDINGS.md "
                "2026-10-22 (b19be61). ESTIMATE 30.44 h (register row x2-queue-fill-1022-hours). " + RELEASE,
    },
    "x2_mx2_qwen3_32b_spill_sweep": {
        "projected_duration_h": 0.36,
        "cmd": [PY, "spill32\\mx2_qwen32b_spill_sweep.py", "--out",
                f"{RESULTS}\\x2_mx2_qwen3_32b_spill_sweep.jsonl"],
        "note": "qwen3-32b spill point and spill cost on evo-x2, n_ctx 20480 to 40960 unclamped (llama-server b10970 "
                "Vulkan -ngl 99, own port 58411, stale-server guard, 1 warm-up + 5 calls per point), pre-registered in "
                "docs/FINDINGS.md 2026-10-22 (777ee78). ESTIMATE 0.36 h (high 1.03 h; register row "
                "x2-queue-fill-1022-hours). " + RELEASE,
    },
    "x2_evalset_tasks_val_v2_mistral_small32": {
        "projected_duration_h": 0.35,
        "cmd": [PY, "evalv2ms\\x2_r2_evalset.py", "--design", "v2", "--mode", "validation", "--workload", "evalset",
                "--guardrails", "off", "--models", "mistral-small3.2:24b", "--out",
                f"{RESULTS}\\x2_evalset_tasks_val_v2_mistral_small32.jsonl"],
        "note": "B3 v2 tasks validation for mistral-small3.2:24b, pre-registered in docs/FINDINGS.md 2026-10-22 "
                "(80ce6f8; protocol consistency extended to mistral-small3.2:24b). ESTIMATE 0.35 h (high 0.54 h; "
                "register row x2-queue-fill-1022-hours). " + RELEASE,
    },
    "x2_evalset_tasks_real_v2_mistral_small32": {
        "projected_duration_h": 3.01,
        "gate": {"requires_done": "x2_evalset_tasks_val_v2_mistral_small32"},
        "cmd": [PY, "evalv2ms\\x2_r2_evalset.py", "--design", "v2", "--mode", "real", "--workload", "evalset",
                "--guardrails", "off", "--models", "mistral-small3.2:24b", "--plan", "tasks_v2_mistral_small32",
                "--preflight", "v2_strict", "--rules-from",
                f"{RESULTS}\\x2_evalset_tasks_val_v2_mistral_small32.jsonl", "--out",
                f"{RESULTS}\\x2_evalset_tasks_real_v2_mistral_small32.jsonl"],
        "note": "B3 v2 task real run for mistral-small3.2:24b, pre-registered in docs/FINDINGS.md 2026-10-22 (80ce6f8): "
                "tasks_v2 tiers and seeds; queue gate requires_done x2_evalset_tasks_val_v2_mistral_small32, and the "
                "v2_strict preflight refuses unless mistral passes every gate of that validation (no waiver). "
                "ESTIMATE 3.01 h (high 3.66 h; register row x2-queue-fill-1022-hours). " + RELEASE,
    },
    # ── reserve draw of 2026-10-22 (operator; docs/X2_QUEUE_PLAN.md; hours: register row x2-llama33-70b-b3v2-hours) ──
    "x2_evalset_tasks_val_v2_llama33_70b": {
        "projected_duration_h": L70_HOURS["val"],
        "cmd": [PY, "evalv2l70\\x2_r2_evalset.py", "--design", "v2", "--mode", "validation", "--workload", "evalset",
                "--guardrails", "off", "--models", "llama3.3:70b", "--out",
                f"{RESULTS}\\x2_evalset_tasks_val_v2_llama33_70b.jsonl"],
        "note": "B3 v2 tasks validation for llama3.3:70b, pre-registered in docs/FINDINGS.md 2026-10-22 (a larger local "
                f"option for the route-at-start comparison). ESTIMATE {L70_HOURS['val']:.2f} h (high "
                f"{L70_HOURS['val_high']:.2f} h; register row x2-llama33-70b-b3v2-hours). " + RELEASE,
    },
    "x2_evalset_tasks_real_v2_llama33_70b": {
        "projected_duration_h": L70_HOURS["real"],
        "gate": {"requires_done": "x2_evalset_tasks_val_v2_llama33_70b"},
        "cmd": [PY, "evalv2l70\\x2_r2_evalset.py", "--design", "v2", "--mode", "real", "--workload", "evalset",
                "--guardrails", "off", "--models", "llama3.3:70b", "--plan", "tasks_v2_llama33_70b", "--preflight",
                "v2_strict", "--rules-from", f"{RESULTS}\\x2_evalset_tasks_val_v2_llama33_70b.jsonl", "--out",
                f"{RESULTS}\\x2_evalset_tasks_real_v2_llama33_70b.jsonl"],
        "note": "B3 v2 task real run for llama3.3:70b at the Ollama default window only, seeds 20260901-03, 30 task "
                "turns, pre-registered in docs/FINDINGS.md 2026-10-22 (a larger local option for the route-at-start "
                "comparison); queue gate requires_done x2_evalset_tasks_val_v2_llama33_70b, and the v2_strict "
                "preflight refuses unless llama3.3:70b passes every gate of that validation (no waiver). ESTIMATE "
                f"{L70_HOURS['real']:.2f} h (high {L70_HOURS['real_high']:.2f} h; register row "
                "x2-llama33-70b-b3v2-hours). " + RELEASE,
    },
    # ── long-task local accuracy (2026-10-22; docs/EXPERIMENTS.md "Long-task local accuracy"; hours: register row
    # long-task-local-hours) ──
    "x2_longtask_local_v1": {
        "projected_duration_h": LT_HOURS["point"],
        "cmd": [PY, "longtask\\x2_longtask.py", "--out", f"{RESULTS}\\x2_longtask_local_v1.jsonl",
                "--stop-after-utc", "2026-10-27T16:00:00+00:00"],
        "note": "Long-task local accuracy, pre-registered in docs/EXPERIMENTS.md 2026-10-22: qwen3:14b on 20 "
                "long-horizon synthetic, trace-length matched work orders (not SWE-Gym), ollama_default and kv_q8_0 "
                "(B5 policy g), seeds 20260901-03; no new task after 2026-10-27 16:00Z. ESTIMATE "
                f"{LT_HOURS['point']:.2f} h (high {LT_HOURS['high']:.2f} h; register row long-task-local-hours). "
                + RELEASE,
    },
    # ── live demo (2026-10-23; docs/EXPERIMENTS.md "Live demo: detector check and acceptance run"; operator decision
    # 4: directly after the running B5 job x2_b5_simcap_8192, inserted with --insert-after) ──
    "x2_demo_detector_check": {
        "projected_duration_h": 0.10,
        "cmd": [PY, "demo\\demo_live_x2.py", "--mode", "detector-check", "--picks", "demo\\picks_v1.json",
                "--results-dir", RESULTS, "--out", f"{RESULTS}\\demo_detector_check.jsonl", "--queued"],
        "note": "Live demo cut-off detector check, pre-registered in docs/EXPERIMENTS.md 2026-10-23: the 5 frozen picks "
                "(demo\\picks_v1.json) at num_ctx 4096 and at the runtime default, qwen3:14b, own Ollama servers on "
                "ports 11541/11544 (OLLAMA_DEBUG=1, started and stopped by the job). ESTIMATE 0.10 h (high 0.3 h) "
                "(from the B3 v2 run's own call timings: prefill 637 tok/s median, decode 19.3 tok/s, 2.08 calls per turn; "
                "the frozen picks' prompts 6.6k to 9.1k tokens). "
                + RELEASE,
    },
    "x2_demo_live_acceptance": {
        "projected_duration_h": 0.15,
        "cmd": [PY, "demo\\demo_live_x2.py", "--mode", "acceptance", "--picks", "demo\\picks_v1.json",
                "--scenarios", "amd8192,intel", "--results-dir", RESULTS, "--out",
                f"{RESULTS}\\demo_live_run_local_v1.jsonl", "--queued"],
        "note": "Live demo 5-task acceptance run, local parts (scenarios amd8192 and intel, local lanes one after another), "
                "pre-registered in docs/EXPERIMENTS.md 2026-10-23; cloud parts completed on the controller. Own Ollama "
                "servers on ports 11541-11543, started and stopped by the job. ESTIMATE 0.15 h (high 0.4 h) "
                "(from the B3 v2 run's own call timings: prefill 637 tok/s median, decode 19.3 tok/s, 2.08 calls per turn; "
                "the frozen picks' prompts 6.6k to 9.1k tokens). " + RELEASE,
    },
}


HOLD_ID = "x2_demo_hold"


def hold_entry(release_utc: str, start_utc: str) -> dict:
    """The demo hold (docs/EXPERIMENTS.md "2026-10-23: Demo hold job"): the window and the release time are part of the
    command and of the note, so the queue state itself says when the machine resumes. Added to ENTRIES only by
    --hold-release-utc with --hold-start-utc."""
    from datetime import datetime, timezone

    def utc(s, flag):
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            raise ValueError(f"{flag} needs an explicit UTC offset")
        return dt.astimezone(timezone.utc)
    r, s = utc(release_utc, "--hold-release-utc"), utc(start_utc, "--hold-start-utc")
    if s >= r:
        raise ValueError("the hold window start must be before its release")
    rel, st = r.isoformat(), s.isoformat()
    return {"projected_duration_h": None,
            "cmd": [PY, "demo\\x2_demo_hold.py", "--start-utc", st, "--release-utc", rel, "--out",
                    f"{RESULTS}\\demo_hold.jsonl", "--queued"],
            "note": f"Demo hold for the live demo window {st} to {rel}, pre-registered in docs/EXPERIMENTS.md "
                    f"2026-10-23: touches nothing (no Ollama, no model, no GPU), holds the queue from its launch until "
                    f"{rel} and then releases itself via tq.advance(); if it starts at or after {rel} it releases at "
                    f"once. Sets no pause."}


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


def insert_entries_after(items: list[dict], after: str, job_ids: list[str]) -> list[dict]:
    """Pure: the queue with the named entries inserted, in order, directly after entry `after` (which must be in the
    queue and running or pending). Every other entry is kept as it is."""
    items = list(items)
    ids = [it["id"] for it in items]
    if after not in ids:
        raise ValueError(f"{after} is not in the queue")
    if items[ids.index(after)].get("status") not in ("running", "pending"):
        raise ValueError(f"{after} is {items[ids.index(after)].get('status')}, not running or pending")
    pos = ids.index(after) + 1
    have = set(ids)
    new = []
    for j in job_ids:
        if j in have:
            raise ValueError(f"{j} is already in the queue")
        e = entry(j)
        req = (e.get("gate") or {}).get("requires_done")
        if req and req not in have:
            raise ValueError(f"{j} is gated on {req}, which is not in the queue ahead of it")
        new.append(e)
        have.add(j)
    return items[:pos] + new + items[pos:]


def queue_insert_after(after: str, job_ids: list[str], log=print) -> list[dict]:
    import t2s_queue as tq
    tq.set_pause(f"x2_queue_fill --insert-after {after} {','.join(job_ids)}: one read-modify-write inserting these "
                 "entries; released by clear_pause() in this same script's finally block within seconds")
    try:
        new = insert_entries_after(tq.read_queue(), after, job_ids)
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
    ap.add_argument("--insert-after", nargs="+", default=None, metavar=("AFTER_ID", "JOB_ID"),
                    help="insert the named entries directly after AFTER_ID (running or pending)")
    ap.add_argument("--hold-release-utc", default=None,
                    help=f"defines the {HOLD_ID} entry with this release time (ISO 8601 with offset)")
    ap.add_argument("--hold-start-utc", default=None, help="the demo window start, required with --hold-release-utc")
    args = ap.parse_args(argv)
    if args.hold_release_utc:
        if not args.hold_start_utc:
            ap.error("--hold-release-utc needs --hold-start-utc")
        ENTRIES[HOLD_ID] = hold_entry(args.hold_release_utc, args.hold_start_utc)
    if args.insert_after:
        after, jobs = args.insert_after[0], args.insert_after[1:]
        unknown = [j for j in jobs if j not in ENTRIES]
        if not jobs or unknown:
            ap.error(f"--insert-after needs AFTER_ID and known job id(s); unknown {unknown}")
        queue_insert_after(after, jobs)
        return 0
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
