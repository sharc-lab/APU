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
# register row own-jobs-hours (ESTIMATE; harness/own_compaction.py::hours_estimate), (point, high)
OC_HOURS = {"part_a": (1.04, 3.07), "part_b": (1.89, 2.08), "probe": (0.27, 0.66)}
# register row own-profiler-probe-hours (ESTIMATE; harness/x2_own_profiler_probe.py::hours_estimate), (point, high)
OP_HOURS = (0.78, 1.38)
_OC_COMMON = ["--design", "v2", "--mode", "real", "--workload", "evalset", "--guardrails", "off", "--models",
              "qwen3:14b", "--preflight", "v2_te_deviation", "--rules-from", f"{RESULTS}\\x2_evalset_tasks_val_v2.jsonl"]

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
    # ── own-compaction and own-config-search (2026-10-23; docs/EXPERIMENTS.md "own-compaction", "own-config-search";
    # hours: register row own-jobs-hours; side directory owncomp; placement in docs/X2_QUEUE_PLAN.md) ──
    "x2_own_compaction_v1_a": {
        "projected_duration_h": OC_HOURS["part_a"][0],
        "cmd": [PY, "owncomp\\x2_r2_evalset.py", *_OC_COMMON, "--plan", "own_compaction_v1", "--tiers",
                "own_compaction_8192,own_compaction_4096", "--out", f"{RESULTS}\\x2_own_compaction_v1_a.jsonl"],
        "note": "own-compaction part a, pre-registered in docs/EXPERIMENTS.md 2026-10-23: qwen3:14b B3 v2 tasks, the "
                "client-side context manager at num_ctx 8192 and 4096, seeds 20260901-05, detector v2 on every call "
                f"(own debug server log). ESTIMATE {OC_HOURS['part_a'][0]:.2f} h (high {OC_HOURS['part_a'][1]:.2f} h; "
                "register row own-jobs-hours). " + RELEASE,
    },
    "x2_own_compaction_v1_b": {
        "projected_duration_h": OC_HOURS["part_b"][0],
        "cmd": [PY, "owncomp\\x2_r2_evalset.py", *_OC_COMMON, "--plan", "own_compaction_v1", "--tiers",
                "client_trim_4096,client_trim_8192", "--out", f"{RESULTS}\\x2_own_compaction_v1_b.jsonl"],
        "note": "own-compaction part b (comparison arms), pre-registered in docs/EXPERIMENTS.md 2026-10-23: the "
                "existing client trim (margin 0.05) on B3 v2 at num_ctx 4096 and 8192, seeds 20260901-05, detector "
                f"v2 on every call. ESTIMATE {OC_HOURS['part_b'][0]:.2f} h (high {OC_HOURS['part_b'][1]:.2f} h; "
                "register row own-jobs-hours). " + RELEASE,
    },
    "x2_own_config_probe_q8_15360": {
        "projected_duration_h": OC_HOURS["probe"][0],
        "cmd": [PY, "owncomp\\x2_r2_evalset.py", *_OC_COMMON, "--plan", "own_config_probe_v1", "--tiers",
                "ollama_ctx_15360", "--server-env", "OLLAMA_KV_CACHE_TYPE=q8_0", "--server-env",
                "OLLAMA_FLASH_ATTENTION=1", "--out", f"{RESULTS}\\x2_own_config_probe_q8_15360.jsonl"],
        "note": "own-config-search probe, pre-registered in docs/EXPERIMENTS.md 2026-10-23: qwen3:14b B3 v2 tasks "
                "with KV q8_0 at num_ctx 15360 (the window q8_0 buys in the KV memory f16 uses at 8192), seeds "
                "20260901-03; records the server's llama_kv_cache size line. ESTIMATE "
                f"{OC_HOURS['probe'][0]:.2f} h (high {OC_HOURS['probe'][1]:.2f} h; register row own-jobs-hours). "
                + RELEASE,
    },
    "x2_own_config_probe_q4_28928": {
        "projected_duration_h": OC_HOURS["probe"][0],
        "cmd": [PY, "owncomp\\x2_r2_evalset.py", *_OC_COMMON, "--plan", "own_config_probe_v1", "--tiers",
                "ollama_ctx_28928", "--server-env", "OLLAMA_KV_CACHE_TYPE=q4_0", "--server-env",
                "OLLAMA_FLASH_ATTENTION=1", "--out", f"{RESULTS}\\x2_own_config_probe_q4_28928.jsonl"],
        "note": "own-config-search probe, pre-registered in docs/EXPERIMENTS.md 2026-10-23: qwen3:14b B3 v2 tasks "
                "with KV q4_0 at num_ctx 28928, seeds 20260901-03; records the server's llama_kv_cache size line. "
                f"ESTIMATE {OC_HOURS['probe'][0]:.2f} h (high {OC_HOURS['probe'][1]:.2f} h; register row "
                "own-jobs-hours). " + RELEASE,
    },
    # ── own-profiler-probe (2026-10-23; docs/EXPERIMENTS.md "own-profiler-probe"; hours: register row
    # own-profiler-probe-hours; side directory ownprobe; appended after x2_own_config_probe_q4_28928) ──
    "x2_own_profiler_probe_v1": {
        "projected_duration_h": OP_HOURS[0],
        "cmd": [PY, "ownprobe\\x2_own_profiler_probe.py", "--out", f"{RESULTS}\\x2_own_profiler_probe_v1.jsonl"],
        "note": "own-profiler-probe, pre-registered in docs/EXPERIMENTS.md 2026-10-23: measured probe of qwen3:14b on "
                "B3 v2 tasks, seed 20260901, at most 600 s of session per config, six window x KV configs in turn (f16 "
                "default/16384/8192/4096, q8_0 15360, q4_0 28928), each with the job's own debug server, detector v2 "
                "and the llama_kv_cache line; per-config files x2_own_profiler_probe_v1_<tier>.jsonl. ESTIMATE "
                f"{OP_HOURS[0]:.2f} h (high {OP_HOURS[1]:.2f} h; register row own-profiler-probe-hours). " + RELEASE,
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
