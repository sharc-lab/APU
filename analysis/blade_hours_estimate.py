"""Hour estimates for the Blade nights (docs/BLADE_PLAN.md), computed from result files, never typed in.

Each job is a list of cells (model, Blade num_ctx, number of sessions, the evo-x2 source of the same session design).
Per cell, in order of preference:

  measured   C3 only: the Blade's own C1 segments at the same ctx (load, calls and thermal waits included).
  rates      the Blade's measured prefill and decode tok/s for that (model, num_ctx), from the 60-second dry-run files
             (results/blade_dryrun/blade_dryrun_*.jsonl), applied to the token workload of the evo-x2 sessions of the
             same design: sum over calls of prompt_eval_count / prefill + completion tokens / decode. Model load and
             harness time are not in it (a lower bound on wall time for that cell).
  scaled     no Blade rate for that (model, num_ctx): evo-x2's wall minutes for the same sessions times a FACTORS entry,
             labelled "scaled (no Blade rate for <model> at <ctx>)". The factors are assumptions.

The Blade "default" tier is estimated as num_ctx 4096 (K1 measures the real value; Ollama sizes its default context by
VRAM). Usage: py -3.12 analysis/blade_hours_estimate.py [--markdown]
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
DRYRUN_DIR = RES / "blade_dryrun"
X2_REAL = "results/x2_r2_real_v1.jsonl"
X2_VALID = "results/x2_r2_validation_v2.jsonl"
X2_VALID_B = "results/x2_r2_validation_v2b.jsonl"
X2_MECH = "results/x2_r2_mechanism.jsonl"
C1 = "results/blade_c1_spill_sweep_20260925T053651Z.jsonl"
SFX = "_call2_notools"
DEFAULT_CTX_ASSUMED = 4096
DRYRUN_FILES = ["results/blade_dryrun/blade_dryrun_k1.jsonl", "results/blade_dryrun/blade_dryrun_r2_validation.jsonl",
                "results/blade_dryrun/blade_dryrun_r2_real.jsonl", "results/blade_dryrun/blade_dryrun_r2_mitigation.jsonl",
                "results/blade_dryrun/blade_dryrun_r2_mechanism.jsonl"]
MIN_PROMPT_FOR_RATE, MIN_GEN_FOR_RATE = 64, 8

FACTORS = {
    # Blade tier fully in 8 GB VRAM: decode is bandwidth-bound on both machines (RTX 4070 Laptop 256 GB/s, Strix Halo
    # LPDDR5X about 256 GB/s), so evo-x2's time is used as is.
    "gpu_resident": 1.0,
    # num_ctx 16384: 4.9 GB of llama3.1:8b weights + 2 GiB f16 KV + buffers is at the 8 GB edge.
    "edge_16384": 1.5,
    # num_ctx 32768: + 4 GiB KV does not fit; Ollama puts layers on the CPU. Unmeasured until K1 / the dry runs.
    "offload_32768": 3.0,
    # mitigation: a render-only request plus llama-tokenize before every call (applied to rate-based cells too).
    "mitigation_overhead": 1.5,
}
CTX_FACTOR = {4096: "gpu_resident", 8192: "gpu_resident", 16384: "edge_16384", 32768: "offload_32768"}


def rows(path) -> list[dict]:
    p = REPO / path if not Path(path).is_absolute() else Path(path)
    if not p.exists():
        return []
    out = []
    for l in p.read_text(encoding="utf-8").splitlines():
        l = l.strip()
        if l:
            try:
                out.append(json.loads(l))
            except json.JSONDecodeError:
                pass
    return out


# ── Blade rates from the dry-run files ──────────────────────────────────────────────────────────────

def call_rate_samples(rs: list[dict]):
    """(model, ctx, prefill_tps|None, decode_tps|None) per call: r2a_turn calls (x2_r2_agent rows) and blade_k1 rows."""
    for r in rs:
        if r.get("record") == "r2a_turn":
            ctx = r.get("num_ctx_requested") or r.get("loaded_context")
            for c in r.get("calls", []):
                pe, ped = c.get("prompt_eval_count_info_only"), c.get("prompt_eval_duration_s")
                ec, ed = c.get("completion_tokens"), c.get("eval_duration_s")
                yield (r.get("model_id"), ctx,
                       pe / ped if pe and ped and pe >= MIN_PROMPT_FOR_RATE else None,
                       ec / ed if ec and ed and ec >= MIN_GEN_FOR_RATE else None)
        elif str(r.get("record", "")).startswith("blade_k1_") and ("prefill_tps" in r or "decode_tps" in r):
            ctx = r.get("num_ctx") or r.get("context_length")
            pe, ec = r.get("prompt_eval_count") or 0, r.get("eval_count") or 0
            yield (r.get("model_tag"), ctx,
                   r.get("prefill_tps") if pe >= MIN_PROMPT_FOR_RATE else None,
                   r.get("decode_tps") if ec >= MIN_GEN_FOR_RATE else None)


def dryrun_rates(files=None) -> dict:
    """{file name: {f"{model}@{ctx}": {"prefill_tps", "decode_tps", "n_prefill", "n_decode"}}} (medians)."""
    out = {}
    for f in (DRYRUN_FILES if files is None else files):  # None = the committed dry-run files; [] = none
        acc = defaultdict(lambda: ([], []))
        for model, ctx, pf, dc in call_rate_samples(rows(f)):
            if model is None or ctx is None:
                continue
            a = acc[f"{model}@{int(ctx)}"]
            if pf:
                a[0].append(pf)
            if dc:
                a[1].append(dc)
        out[Path(f).name] = {k: {"prefill_tps": st.median(p) if p else None, "decode_tps": st.median(d) if d else None,
                                 "n_prefill": len(p), "n_decode": len(d)} for k, (p, d) in acc.items()}
    return out


def rate_index(rates_by_file: dict) -> dict:
    """{(model, ctx): {"prefill_tps", "decode_tps", "source"}} merged over files (medians of the per-file medians)."""
    acc = defaultdict(lambda: ([], [], []))
    for fname, d in rates_by_file.items():
        for key, v in d.items():
            model, ctx = key.rsplit("@", 1)
            a = acc[(model, int(ctx))]
            if v["prefill_tps"]:
                a[0].append(v["prefill_tps"])
            if v["decode_tps"]:
                a[1].append(v["decode_tps"])
            a[2].append(fname)
    return {k: {"prefill_tps": st.median(p) if p else None, "decode_tps": st.median(d) if d else None,
                "source": sorted(set(s))} for k, (p, d, s) in acc.items()}


# ── evo-x2 session workloads ────────────────────────────────────────────────────────────────────────

def x2_sessions(path, model, arm) -> list[dict]:
    """[{"minutes", "prompt_tokens", "gen_tokens"}] per completed (model, arm, seed) session."""
    per = defaultdict(lambda: {"minutes": 0.0, "prompt_tokens": 0, "gen_tokens": 0})
    for r in rows(path):
        if r.get("record") == "r2a_turn" and r.get("model_id") == model and r.get("arm_id") == arm:
            s = per[r["seed"]]
            s["minutes"] += (r.get("turn_wall_s") or 0.0) / 60
            for c in r.get("calls", []):
                s["prompt_tokens"] += c.get("prompt_eval_count_info_only") or 0
                s["gen_tokens"] += c.get("completion_tokens") or 0
    return list(per.values())


def _t(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def c1_segments_minutes(rs: list[dict]) -> dict:
    out = defaultdict(list)
    prev_end = seg_start = None
    ncalls = 0
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


# ── cells and jobs ──────────────────────────────────────────────────────────────────────────────────

def cell_hours(model, ctx, n_sessions, x2_path, x2_model, x2_arm, rates, extra_factor=1.0, model_factor=1.0):
    """One cell: rate-based if the Blade has prefill and decode rates for (model, ctx), else scaled."""
    sess = x2_sessions(x2_path, x2_model, x2_arm)
    if not sess:
        return {"hours": None, "kind": "unknown", "label": f"no evo-x2 sessions for {x2_model} {x2_arm} in {x2_path}"}
    mean = lambda k: sum(s[k] for s in sess) / len(sess)  # noqa: E731
    r = rates.get((model, int(ctx)))
    if r and r.get("prefill_tps") and r.get("decode_tps"):
        sec = mean("prompt_tokens") / r["prefill_tps"] + mean("gen_tokens") / r["decode_tps"]
        return {"hours": n_sessions * sec * extra_factor / 3600, "kind": "rates",
                "label": f"Blade rates {model}@{ctx}: prefill {r['prefill_tps']:.0f}, decode {r['decode_tps']:.1f} tok/s "
                         f"on {x2_arm} token workload ({mean('prompt_tokens'):.0f} prompt + {mean('gen_tokens'):.0f} gen"
                         f" tokens per session)"}
    f = FACTORS[CTX_FACTOR.get(int(ctx), "offload_32768")]
    return {"hours": n_sessions * mean("minutes") * f * extra_factor * model_factor / 60, "kind": "scaled",
            "label": f"scaled (no Blade rate for {model} at {ctx}): evo-x2 {x2_arm} {mean('minutes'):.1f} min/session"
                     f" x{f}" + (f" x{model_factor} (model)" if model_factor != 1.0 else "")
                     + (f" x{extra_factor}" if extra_factor != 1.0 else "")}


def _job(night, name, cells, note=""):
    hs = [c["hours"] for c in cells]
    kinds = sorted({c["kind"] for c in cells})
    return {"night": night, "job": name, "hours": round(sum(h for h in hs if h is not None), 2),
            "kind": kinds[0] if len(kinds) == 1 else "mixed (" + ", ".join(kinds) + ")",
            "basis": "; ".join(c["label"] for c in cells), "note": note}


def estimate(rates_files=None) -> dict:
    rates_by_file = dryrun_rates(rates_files)
    rates = rate_index(rates_by_file)
    D = DEFAULT_CTX_ASSUMED
    L, Q = "llama3.1:8b", "qwen3:8b"
    jobs = []
    c1 = c1_segments_minutes(rows(C1))

    def validation(model, x2_file, x2_model, model_factor=1.0, neg_sessions=3):
        return [cell_hours(model, 32768, neg_sessions, x2_file, x2_model, f"ollama_ctx_131072{SFX}", rates,
                           model_factor=model_factor),
                cell_hours(model, 32768, 3, x2_file, x2_model, "ollama_ctx_131072", rates, model_factor=model_factor),
                cell_hours(model, 8192, 1, x2_file, x2_model, f"ollama_ctx_8192_positive_control{SFX}", rates,
                           model_factor=model_factor)]

    def real(model, model_factor=1.0):
        # the qwen3:8b real-run sessions (x2_r2_real_v1b) are not synced: llama3.1:8b's workload x model factor
        return [cell_hours(model, D, 5, X2_REAL, L, f"ollama_ctx_4096{SFX}", rates, model_factor=model_factor),
                cell_hours(model, 4096, 5, X2_REAL, L, f"ollama_ctx_4096{SFX}", rates, model_factor=model_factor),
                cell_hours(model, 32768, 5, X2_REAL, L, f"ollama_ctx_32768{SFX}", rates, model_factor=model_factor)]

    # night 1 (rerun 2026-10-08): K1, validation with the negative control at 5 sessions, the gate, then EITHER real +
    # mitigation (branch "real") OR the 4096 mechanism session (branch "mechanism"); the night total counts the
    # longer branch, night_totals_by_branch_h has both.
    jobs.append({"night": 1, "job": "blade_k1_v2", "hours": 0.5, "kind": "scaled (itemized)",
                 "basis": "per model 10 Ollama calls with prompts up to 49K tokens (cut to the window) plus one "
                          "llama-server load", "note": ""})
    jobs.append(_job(1, "blade_r2_validation_v2 (llama3.1:8b, arm b at 32768, negative control 5 sessions)",
                     validation(L, X2_VALID, L, neg_sessions=5)))
    jobs.append({"night": 1, "job": "blade_r2_gate_v2", "hours": 0.0, "kind": "measured",
                 "basis": "reads one jsonl, no model", "note": ""})
    jobs.append({**_job(1, "blade_r2_real_v2 (llama3.1:8b, default/4096/32768 x 5 seeds)", real(L)),
                 "branch": "real"})
    jobs.append({**_job(1, "blade_r2_mitigation_v2 (llama3.1:8b, 4096/8192 x 3 seeds)",
                        [cell_hours(L, 4096, 3, X2_REAL, L, f"ollama_ctx_4096{SFX}", rates,
                                    extra_factor=FACTORS["mitigation_overhead"]),
                         cell_hours(L, 8192, 3, X2_MECH, L, f"ollama_ctx_8192{SFX}", rates,
                                    extra_factor=FACTORS["mitigation_overhead"])],
                        "plus one tier if K1's default is not 4096 or 8192"), "branch": "real"})
    jobs.append({**_job(1, "blade_r2_mechanism_4096_v2 (llama3.1:8b, 4096 x 1 session)",
                        [cell_hours(L, 4096, 1, X2_MECH, L, f"ollama_ctx_4096{SFX}", rates)]), "branch": "mechanism"})
    a = sum(c1[c]["minutes"] for c in (36864, 38912, 40960, 43008)) / 60
    b = sum(c1[c]["minutes"] * 4 / c1[c]["n_calls"] for c in (40960, 43008)) / 60
    jobs.append({"night": 2, "job": "blade_c3_sysmem_fallback_v1 (half A upper bound + half B)", "hours": round(a + b, 2),
                 "kind": "measured", "basis": "C1 segments at 36864/38912/40960/43008 (half A, 1+5 calls) and "
                 "40960/43008 scaled to 1+3 calls (half B)", "note": "plus operator time at the two gates"})
    jobs.append(_job(2, "blade_r2_mechanism_v1 (llama3.1:8b, 5 tiers x 1 session)",
                     [cell_hours(L, ctx, 1, X2_MECH, L, f"{arm}{SFX}", rates) for ctx, arm in
                      ((D, "ollama_ctx_4096"), (4096, "ollama_ctx_4096"), (8192, "ollama_ctx_8192"),
                       (16384, "ollama_ctx_16384"), (32768, "ollama_ctx_32768"))]))
    jobs.append(_job(3, "blade_r2_validation_qwen3_8b_v1 (only if it fits)", validation(Q, X2_VALID_B, Q)))
    jobs.append(_job(3, "blade_r2_real_qwen3_8b_v1 (only if it fits)", real(Q, model_factor=1.2),
                     "workload: llama3.1:8b sessions x1.2 (x2_r2_real_v1b, with qwen3:8b, is not synced)"))
    by_branch = defaultdict(lambda: defaultdict(float))     # {night: {branch or "": hours}}
    for j in jobs:
        by_branch[j["night"]][j.get("branch", "")] += j["hours"]
    totals, branch_totals = {}, {}
    for n, d in sorted(by_branch.items()):
        common = d.pop("", 0.0)
        if d:
            branch_totals[n] = {b: round(common + h, 2) for b, h in sorted(d.items())}
            totals[n] = round(common + max(d.values()), 2)
        else:
            totals[n] = round(common, 2)
    return {"factors": FACTORS, "default_ctx_assumed": D, "rates_by_dryrun_file": rates_by_file,
            "rates_used": {f"{m}@{c}": v for (m, c), v in rates.items()}, "jobs": jobs,
            "night_totals_h": totals, "night_totals_by_branch_h": branch_totals}


def markdown(est: dict) -> str:
    lines = ["| night | job | hours | measured/rates/scaled | basis |", "|---|---|---|---|---|"]
    for j in est["jobs"]:
        br = f" [gate branch {j['branch']}]" if j.get("branch") else ""
        lines.append(f"| {j['night']} | {j['job']}{br} | {j['hours']:.2f} | {j['kind']} | {j['basis']}"
                     + (f" ({j['note']})" if j["note"] else "") + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    e = estimate()
    if "--markdown" in sys.argv:
        print(markdown(e))
        print("\nnight totals (h, longer gate branch):", e["night_totals_h"])
        print("night totals by gate branch (h):", e["night_totals_by_branch_h"])
        print("Blade dry-run rates found:", e["rates_used"] or "none (all R2 cells scaled)")
    else:
        print(json.dumps(e, indent=1, default=str))
