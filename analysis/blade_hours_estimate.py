"""Hour estimates for the Blade nights (docs/BLADE_PLAN.md), computed from result files, never typed in.

Every job's estimate is either "measured" (the same work was timed on the Blade, e.g. C1's per-ctx segments for C3)
or "scaled" (timed on evo-x2 for the same design and multiplied by a stated factor). The factors are assumptions,
stated in FACTORS with the reason; nothing in this file is a Blade measurement unless its source is a blade_* file.

Usage: py -3.12 analysis/blade_hours_estimate.py   (prints JSON; --markdown prints the plan table)
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
X2_REAL = RES / "x2_r2_real_v1.jsonl"
X2_VALID = RES / "x2_r2_validation_v2.jsonl"
X2_MECH = RES / "x2_r2_mechanism.jsonl"
C1 = RES / "blade_c1_spill_sweep_20260925T053651Z.jsonl"
MODEL = "llama3.1:8b"
SFX = "_call2_notools"

FACTORS = {
    # Blade tier fully in 8 GB VRAM (weights 4.9 GB + KV at 4096/8192): decode is bandwidth-bound on both machines
    # (RTX 4070 Laptop 256 GB/s, Strix Halo LPDDR5X about 256 GB/s), so evo-x2's time is used as is.
    "gpu_resident": 1.0,
    # num_ctx 16384: 4.9 GB weights + 2 GiB f16 KV + buffers is at the 8 GB edge; Ollama may move a few layers.
    "edge_16384": 1.5,
    # num_ctx 32768: + 4 GiB KV, does not fit; Ollama offloads layers to the CPU. Unmeasured; K1 measures it.
    "offload_32768": 3.0,
    # validation arm b, num_ctx 131072: 16 GiB KV, most layers on the CPU. Unmeasured.
    "offload_131072": 5.0,
    # qwen3:8b relative to llama3.1:8b (not timed on the same design anywhere in these files).
    "qwen3_8b_vs_llama": 1.2,
    # mitigation: a render-only request plus llama-tokenize before every call.
    "mitigation_overhead": 1.5,
}


def rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def session_minutes(rs: list[dict], model=MODEL) -> dict:
    """{arm: mean minutes per completed session} from r2a_turn turn_wall_s."""
    per = defaultdict(float)
    for r in rs:
        if r.get("record") == "r2a_turn" and r.get("model_id") == model:
            per[(r["arm_id"], r["seed"])] += r.get("turn_wall_s") or 0.0
    by_arm = defaultdict(list)
    for (arm, _), s in per.items():
        by_arm[arm].append(s / 60)
    return {a: sum(v) / len(v) for a, v in by_arm.items()}


def _t(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def c1_segments_minutes(rs: list[dict]) -> dict:
    """{ctx: {"minutes": wall from the previous segment's end to this server's end (load + calls + thermal waits),
    "n_calls": calls in it}} from C1's own rows; for repeated anchors the mean."""
    out = defaultdict(list)
    prev_end = None
    seg_start, ncalls = None, 0
    for r in rs:
        rec = r.get("record")
        if rec == "server_start":
            seg_start, ncalls = (prev_end or _t(r["ts_utc"])), 0
        elif rec == "call":
            ncalls += 1
        elif rec == "server_end" and seg_start is not None:
            end = _t(r["ts_utc"])
            out[r["ctx"]].append(((end - seg_start).total_seconds() / 60, ncalls))
            prev_end, seg_start = end, None
    return {c: {"minutes": sum(m for m, _ in v) / len(v), "n_calls": v[0][1]} for c, v in out.items()}


def estimate() -> dict:
    real = session_minutes(rows(X2_REAL))
    valid = session_minutes(rows(X2_VALID))
    mech = session_minutes(rows(X2_MECH))
    c1 = c1_segments_minutes(rows(C1))
    F = FACTORS
    jobs = []

    def add(night, job, hours, basis, kind, note=""):
        jobs.append({"night": night, "job": job, "hours": round(hours, 2), "kind": kind, "basis": basis,
                     "note": note})

    # K1: no timed equivalent; itemized from Blade C1 (4B prefill) and call counts. Reported as scaled.
    pf_4b = c1.get(32768)
    add(1, "blade_k1_v1", 0.5, "per model: 10 Ollama calls with prompts up to 49K tokens (truncated to the window) "
        "plus one llama-server load; C1's ctx 32768 segment (4B, 6 calls) took "
        f"{pf_4b['minutes']:.1f} min" if pf_4b else "itemized", "scaled",
        "3 models if present; qwen3:8b and the 4B tools tag are not in the Blade store yet")
    v = (3 * valid[f"ollama_ctx_131072{SFX}"] * F["offload_131072"]
         + valid[f"ollama_ctx_8192_positive_control{SFX}"] * F["gpu_resident"]
         + 3 * valid["ollama_ctx_131072"] * F["offload_131072"])
    add(1, "blade_r2_validation_v1 (llama3.1:8b)", v / 60,
        f"x2_r2_validation_v2 llama3.1:8b session minutes: arm b {valid[f'ollama_ctx_131072{SFX}']:.1f}, diagnostic "
        f"{valid['ollama_ctx_131072']:.1f}, positive control {valid[f'ollama_ctx_8192_positive_control{SFX}']:.1f}; "
        f"arm b x{F['offload_131072']}", "scaled")
    d4096 = real[f"ollama_ctx_4096{SFX}"]
    r = 5 * (d4096 * F["gpu_resident"] + d4096 * F["gpu_resident"]
             + real[f"ollama_ctx_32768{SFX}"] * F["offload_32768"])
    add(1, "blade_r2_real_v1 (llama3.1:8b, default/4096/32768 x 5 seeds)", r / 60,
        f"x2_r2_real_v1 llama3.1:8b session minutes: 4096 {d4096:.1f}, 32768 {real[f'ollama_ctx_32768{SFX}']:.1f}; "
        f"Blade default assumed 4096 for the estimate (K1 measures it); 32768 "
        f"x{F['offload_32768']}", "scaled",
        f"if K1 says qwen3:8b fits: + {(v + r) * F['qwen3_8b_vs_llama'] / 60:.1f} h (validation + real, scaled)")
    m = 3 * (d4096 + mech[f"ollama_ctx_8192{SFX}"]) * F["mitigation_overhead"]
    add(1, "blade_r2_mitigation_v1 (llama3.1:8b, 4096/8192 x 3 seeds)", m / 60,
        f"x2_r2_real_v1 4096 {d4096:.1f} min and x2_r2_mechanism 8192 {mech[f'ollama_ctx_8192{SFX}']:.1f} min per "
        f"session, x{F['mitigation_overhead']} for render + tokenize per call", "scaled",
        "depends on mitigation merge; Blade default tier folds into 4096 if K1 measures 4096")
    a = sum(c1[c]["minutes"] for c in (36864, 38912, 40960, 43008))
    b = sum(c1[c]["minutes"] * 4 / c1[c]["n_calls"] for c in (40960, 43008))
    add(2, "blade_c3_sysmem_fallback_v1 half A (upper bound)", a / 60,
        "C1 measured segments (load + 6 calls + thermal waits) at ctx 36864/38912/40960/43008: "
        + ", ".join(f"{c} {c1[c]['minutes']:.1f} min" for c in (36864, 38912, 40960, 43008)), "measured",
        "upper bound: if the policy makes the spilled points fail at load, half A is minutes")
    add(2, "blade_c3_sysmem_fallback_v1 half B (4 of 6 calls)", b / 60,
        "C1 measured segments at 40960/43008 scaled to 1 warm-up + 3 calls", "measured (scaled by call count)",
        "plus operator time at each gate (about 2 min each)")
    mm = (mech[f"ollama_ctx_4096{SFX}"] * 2 + mech[f"ollama_ctx_8192{SFX}"]
          + mech[f"ollama_ctx_16384{SFX}"] * F["edge_16384"] + mech[f"ollama_ctx_32768{SFX}"] * F["offload_32768"])
    add(2, "blade_r2_mechanism_v1 (llama3.1:8b, 5 tiers x 1 session)", mm / 60,
        "x2_r2_mechanism session minutes: " + ", ".join(f"{k.replace(SFX, '')} {v:.1f}" for k, v in sorted(mech.items()))
        + f"; Blade default as 4096, 16384 x{F['edge_16384']}, 32768 x{F['offload_32768']}", "scaled")
    totals = defaultdict(float)
    for j in jobs:
        totals[j["night"]] += j["hours"]
    return {"factors": FACTORS, "jobs": jobs, "night_totals_h": {k: round(v, 2) for k, v in totals.items()},
            "sources": [str(p.relative_to(REPO)) for p in (X2_REAL, X2_VALID, X2_MECH, C1)]}


def markdown(est: dict) -> str:
    lines = ["| night | job | hours | measured/scaled | basis |", "|---|---|---|---|---|"]
    for j in est["jobs"]:
        lines.append(f"| {j['night']} | {j['job']} | {j['hours']:.2f} | {j['kind']} | {j['basis']}"
                     + (f" ({j['note']})" if j["note"] else "") + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    e = estimate()
    print(markdown(e) if "--markdown" in sys.argv else json.dumps(e, indent=1))
    if "--markdown" in sys.argv:
        print("\nnight totals (h):", e["night_totals_h"])
