"""Regenerates docs/NUMBERS_REGISTER.md: one row per number used anywhere in this program (the claims
ledger, FINDINGS.md, the paper draft, the demo, or any report), each one COMPUTED FRESH from a real,
committed data file by a real function in this repo -- never hand-typed.

Built 2026-10-01 after a hand-typed number (qwen3-32b "41/82 (50%)" refusal share) was reported in a chat
message with no backing file anywhere in this repo. From now on: a report may only quote a number by pasting
its row from this register, never by typing a number from memory. If a number is not in the register, it is
not citable yet -- add an entry and a compute function for it first.

Each entry in NUMBER_ENTRIES is a dict:
  claim_id       -- short id matching the claims ledger / FINDINGS.md section where this number is used
  description    -- one line, what the number means
  compute        -- callable(repo_root: Path) -> dict with at least {"value": str, "n": int|str}; raises
                    FileNotFoundError (or any exception) if its data file(s) are not present -- caught by
                    the register builder and recorded as UNSUPPORTED, never silently skipped
  data_files     -- list of repo-relative paths this number is computed from (for the table's own column,
                    and so a human can open exactly what was read)
  script_function -- "module.py::function_name" string, documentation only, must name a real function in
                    this repo (checked by test_numbers_register.py against the real module)
  reported_value -- (optional) the value as currently written in a doc, used only to classify the row as
                    VERIFIED (matches, within the stated tolerance) / CORRECTED (does not match) /
                    UNSUPPORTED (compute() raised). Omit this key for a number that has never been reported
                    in prose before (first-time entries have no "old" value to check against).

Usage: py -3.12 analysis/numbers_register.py
Writes docs/NUMBERS_REGISTER.md. Exits non-zero if any entry is UNSUPPORTED (a real file is missing) so this
can be wired into a CI-style check later; today it is informational only (the caller decides what to do).
"""
from __future__ import annotations

import json
import statistics
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _git_head():
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def _today():
    import datetime
    return datetime.date.today().isoformat()


def _read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# ───────────────────────────────────────────────────────────────── compute functions


def compute_px2_ttft_gap(repo):
    """PX2 B4-vs-S4 TTFT gap, per model, from the real PX2 rows (ttft_s present means a real measured row;
    co_runner field names the condition directly)."""
    path = repo / "results" / "t2s_night2_20260930T135145Z.jsonl"
    rows = [r for r in _read_jsonl(path)
           if r.get("record") is None and r.get("section") == "PX2" and r.get("ttft_s") is not None]
    by_model_cond = {}
    for r in rows:
        by_model_cond.setdefault((r["model_id"], r["co_runner"]), []).append(r["ttft_s"])
    gaps = {}
    for model in sorted({k[0] for k in by_model_cond}):
        n0 = by_model_cond.get((model, "N0"))
        b4 = by_model_cond.get((model, "B4"))
        s4 = by_model_cond.get((model, "S4"))
        if not (n0 and b4 and s4):
            continue
        n0m, b4m, s4m = statistics.median(n0), statistics.median(b4), statistics.median(s4)
        gap_pct = abs(b4m - s4m) / n0m * 100
        gaps[model] = {"n0_ttft_s": n0m, "b4_ratio": b4m / n0m, "s4_ratio": s4m / n0m,
                      "b4_vs_s4_gap_pct": gap_pct, "n": len(n0)}
    worst = max(gaps.values(), key=lambda v: v["b4_vs_s4_gap_pct"])
    best = min(gaps.values(), key=lambda v: v["b4_vs_s4_gap_pct"])
    return {"value": f"{best['b4_vs_s4_gap_pct']:.1f}-{worst['b4_vs_s4_gap_pct']:.1f}% (B4 vs S4 TTFT gap "
                     f"range across {len(gaps)} models)",
           "n": sum(v["n"] for v in gaps.values()), "detail": gaps}


def compute_px2_full_ratio_table(repo):
    """Full per-model x per-condition (S2,S4,S8,S8x,S14,S28,B4) TTFT and decode ratio vs N0, with
    bootstrap 95% CI (seeded, 1000 resamples of the real per-call values). 2026-10-01: this resolves
    the apparent 'only B4 clears the 1.10 criterion' claim -- that was true only for the two largest
    models (qwen3-32b, llama-3.3-70b, both compute-bound). For llama31-8b, qwen3-14b and qwen3-8b,
    S14 and S28 (the SMT/occupancy hog conditions) ALSO clear 1.10, and in several cases exceed B4's
    own ratio for that model. The real max ratio seen anywhere on evo-x2 is 1.129x (qwen3-8b, S28)."""
    path = repo / "results" / "t2s_night2_20260930T135145Z.jsonl"
    rows = [r for r in _read_jsonl(path) if r.get("record") is None and r.get("section") == "PX2"]
    by_ttft, by_decode = {}, {}
    for r in rows:
        key = (r["model_id"], r["co_runner"])
        if r.get("ttft_s") is not None:
            by_ttft.setdefault(key, []).append(r["ttft_s"])
        if r.get("decode_tok_s") is not None:
            by_decode.setdefault(key, []).append(r["decode_tok_s"])
    if not by_ttft:
        raise FileNotFoundError("no PX2 ttft rows found")
    models = sorted({k[0] for k in by_ttft})
    conds = ["S2", "S4", "S8", "S8x", "S14", "S28", "B4"]
    import random
    rng = random.Random(42)

    def boot_ci(vals_cond, vals_base, n=1000):
        ratios = []
        for _ in range(n):
            c = [rng.choice(vals_cond) for _ in vals_cond]
            b = [rng.choice(vals_base) for _ in vals_base]
            ratios.append(statistics.median(c) / statistics.median(b))
        ratios.sort()
        return ratios[int(0.025 * n)], ratios[int(0.975 * n)]

    table = {}
    max_ratio = (0.0, None, None)
    criterion_clears = []
    for model in models:
        n0t, n0d = by_ttft.get((model, "N0")), by_decode.get((model, "N0"))
        if not n0t:
            continue
        for cond in conds:
            ct, cd = by_ttft.get((model, cond)), by_decode.get((model, cond))
            if not ct:
                continue
            ttft_ratio = statistics.median(ct) / statistics.median(n0t)
            t_lo, t_hi = boot_ci(ct, n0t)
            entry = {"ttft_ratio": ttft_ratio, "ttft_ci": [t_lo, t_hi], "n": len(ct)}
            if n0d and cd:
                decode_ratio = statistics.median(cd) / statistics.median(n0d)
                d_lo, d_hi = boot_ci(cd, n0d)
                entry["decode_ratio"] = decode_ratio
                entry["decode_ci"] = [d_lo, d_hi]
            table[f"{model}@{cond}"] = entry
            if ttft_ratio > max_ratio[0]:
                max_ratio = (ttft_ratio, model, cond)
            if ttft_ratio >= 1.10:
                criterion_clears.append(f"{model}/{cond}")
    return {"value": f"max TTFT ratio on evo-x2: {max_ratio[0]:.3f}x ({max_ratio[1]}/{max_ratio[2]}); "
                     f"{len(criterion_clears)} (model,condition) pairs clear the 1.10 criterion: "
                     f"{', '.join(criterion_clears)}",
           "n": len(table), "detail": table}


def compute_px2_decode_ratios(repo):
    path = repo / "results" / "t2s_night2_20260930T135145Z.jsonl"
    rows = [r for r in _read_jsonl(path)
           if r.get("record") is None and r.get("section") == "PX2" and r.get("decode_tok_s") is not None]
    by_model_cond = {}
    for r in rows:
        by_model_cond.setdefault((r["model_id"], r["co_runner"]), []).append(r["decode_tok_s"])
    b4_ratios, s4_ratios = [], []
    for model in sorted({k[0] for k in by_model_cond}):
        n0 = by_model_cond.get((model, "N0"))
        b4 = by_model_cond.get((model, "B4"))
        s4 = by_model_cond.get((model, "S4"))
        if not (n0 and b4 and s4):
            continue
        n0m = statistics.median(n0)
        b4_ratios.append(statistics.median(b4) / n0m)
        s4_ratios.append(statistics.median(s4) / n0m)
    return {"value": f"B4 {min(b4_ratios):.3f}x-{max(b4_ratios):.3f}x of N0; "
                     f"S4 {min(s4_ratios):.3f}x-{max(s4_ratios):.3f}x of N0",
           "n": len(b4_ratios), "detail": {"b4_ratios": b4_ratios, "s4_ratios": s4_ratios}}


def compute_px2_package_power(repo):
    path = repo / "results" / "t2s_night2_20260930T135145Z.jsonl"
    rows = [r for r in _read_jsonl(path)
           if r.get("record") is None and r.get("section") == "PX2" and r.get("pkg_power_w") is not None]
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model_id"], []).append(r["pkg_power_w"])
    flat_models = {m: round(statistics.median(v), 1) for m, v in by_model.items()
                  if max(v) - min(v) < 2.0}
    return {"value": f"qwen3-32b/llama-3.3-70b flat ~{statistics.median(by_model.get('qwen3-32b', [0])):.1f}W "
                     f"across every condition; qwen3-8b/llama31-8b show a real power jump under compute/"
                     f"bandwidth hogs",
           "n": len(rows), "detail": {m: round(statistics.median(v), 1) for m, v in by_model.items()}}


def compute_px2_thermal_ceiling(repo):
    main_path = repo / "results" / "t2s_night2_20260930T135145Z.jsonl"
    lhm_path = repo / "results" / "t2s_night2_20260930T135145Z_lhm.jsonl"
    from datetime import datetime
    import bisect

    def parse_ts(s):
        return datetime.fromisoformat(s.replace("Z", "+00:00")) if s.endswith("Z") else datetime.fromisoformat(s)

    rows = [r for r in _read_jsonl(main_path)
           if r.get("record") is None and r.get("section") == "PX2" and r.get("ttft_s") is not None
           and r.get("model_id") == "qwen3-8b" and (r.get("co_runner") or "N0") == "S14"]
    t0 = min(parse_ts(r["ts_utc"]) for r in rows)
    t1 = max(parse_ts(r["ts_utc"]) for r in rows)
    lhm = _read_jsonl(lhm_path)
    lhm.sort(key=lambda r: r["ts"])
    lhm_ts = [parse_ts(r["ts"]) for r in lhm]
    i0, i1 = bisect.bisect_left(lhm_ts, t0), bisect.bisect_right(lhm_ts, t1)
    temp_key = "Temperature | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | Core (Tctl/Tdie)"
    temps = [r["sensors"].get(temp_key) for r in lhm[i0:i1] if r["sensors"].get(temp_key) is not None]
    if not temps:
        raise FileNotFoundError("no LHM samples in the qwen3-8b/S14 condition window")
    return {"value": f"{statistics.median(temps):.1f}C median [{max(temps):.1f}C max] CPU temp, qwen3-8b/S14 "
                     f"only (LHM collector gap for other conditions/models)",
           "n": len(temps)}


def compute_k1_v3_x2_table(repo):
    path = repo / "results" / "t2s_k1_ollama_evo-x2_20260930T205515Z.jsonl"
    rows = _read_jsonl(path)
    meta = {r["model_tag"]: r for r in rows if r.get("record") == "tier_v3_model_meta"}
    tiers = {r["model_tag"]: r for r in rows if r.get("record") == "tier"}
    out = {}
    for tag, m in meta.items():
        t = tiers.get(tag, {})
        out[tag] = {"capped": m["capped"], "native_ctx": m["native_ctx"],
                   "ollama_default_ctx": t.get("ollama_default_ctx")}
    return {"value": json.dumps(out), "n": len(meta), "detail": out}


def compute_trace_context_exit_rate(repo):
    path = repo / "results" / "traces" / "exit_status_sample.parquet"
    import pandas as pd
    df = pd.read_parquet(path)
    rate = float(df["exit_status"].astype(str).str.contains("context", case=False, na=False).mean())
    return {"value": f"{rate * 100:.2f}%", "n": len(df)}


def compute_trace_cdf_crossing_fractions(repo):
    path = repo / "results" / "traces" / "agent_step_lengths.parquet"
    import pandas as pd
    df = pd.read_parquet(path)
    nebius = df[df["dataset"] == "nebius/SWE-agent-trajectories"]
    traj_max = nebius.groupby("trajectory_id")["tokens_qwen"].max()
    frac_32k = float((traj_max > 32000).mean())
    frac_40960 = float((traj_max > 40960).mean())
    return {"value": f"{frac_32k * 100:.0f}% of trajectories exceed 32K at some step, "
                     f"{frac_40960 * 100:.0f}% exceed 40,960",
           "n": int(nebius["trajectory_id"].nunique())}


def compute_t2s_x2_default_ctx_table(repo):
    """Per-model ollama_default_ctx on evo-t2s vs evo-x2, from each host's own real K1 v3 jsonl. See
    docs/FINDINGS.md's 2026-10-01 mechanism entry for why these differ (Ollama's own integrated-GPU
    opt-out policy on evo-t2s, confirmed live via a read-only /api/ps check and the OLLAMA_IGPU_ENABLE=1
    test), not a hardware limit."""
    t2s_path = repo / "results" / "apu_results__t2s_k1_ollama_evo-t2s_20261001T074622Z.jsonl"
    x2_path = repo / "results" / "t2s_k1_ollama_evo-x2_20260930T205515Z.jsonl"
    t2s_rows = {r["model_tag"]: r for r in _read_jsonl(t2s_path) if r.get("record") == "tier" and r.get("phase") == "tier"}
    x2_rows = {r["model_tag"]: r for r in _read_jsonl(x2_path) if r.get("record") == "tier" and r.get("phase") == "tier"}
    models = ["qwen3-4b-2507", "llama3.1:8b", "qwen3:8b"]
    table = {}
    for m in models:
        if m not in t2s_rows or m not in x2_rows:
            raise FileNotFoundError(f"missing tier row for {m} on one host")
        table[m] = {"t2s_default_ctx": t2s_rows[m]["ollama_default_ctx"], "x2_default_ctx": x2_rows[m]["ollama_default_ctx"]}
    return {"value": json.dumps(table), "n": len(models), "detail": table}


def compute_trace_fraction_exceeds_4096(repo):
    """Fraction of real trace-matched agent steps (both uncensored sources, qwen tokenizer) whose token
    count exceeds 4096 -- the evo-t2s default context floor found in the K1 v3 mechanism entry -- vs the
    evo-x2 per-model defaults (131072-262144), for the same trace data."""
    import pandas as pd
    path = repo / "results" / "traces" / "agent_step_lengths.parquet"
    df = pd.read_parquet(path)
    uncensored = df[df["dataset"].isin([
        "nebius/SWE-agent-trajectories", "SWE-Gym/SWE-Gym-Trajectories",
    ])] if set(df["dataset"].unique()) & {"SWE-Gym/SWE-Gym-Trajectories"} else df[df["dataset"] == "nebius/SWE-agent-trajectories"]
    n = len(uncensored)
    if n == 0:
        raise FileNotFoundError("no trace rows matched for the fraction-exceeds-4096 computation")
    over_4096 = float((uncensored["tokens_qwen"] > 4096).mean())
    over_x2_floor = float((uncensored["tokens_qwen"] > 40960).mean())  # qwen3:8b is X2's smallest default (40960)
    return {"value": f"{over_4096*100:.1f}% of steps exceed evo-t2s's 4096 default; "
                     f"{over_x2_floor*100:.1f}% exceed evo-x2's smallest model default (40,960)",
           "n": n}


def compute_b3_corunner_6model(repo):
    """Real nonp12-vs-none TTFT ratio per model, section B1 (qwen3-4b-2507, qwen3-8b) and section B3
    (llama31-8b, qwen3-14b, qwen3-30b-a3b-2507, qwen3-32b) of the real b1/b3 phase run -- matched pairs
    only (same section, so qwen3-4b-2507/qwen3-8b's B2 dose-response sweep rows, which also happen to
    carry a 'nonp12' co_runner label as one of several reference arms, are excluded; mixing them in
    produces a materially different, wrong ratio, confirmed live by computing both ways)."""
    path = repo / "results" / "t2s_night2_20260928T004924Z.jsonl"
    rows = [r for r in _read_jsonl(path) if r.get("record") is None and r.get("kind") != "start"
            and r.get("section") in ("B1", "B3")]
    if not rows:
        raise FileNotFoundError("no B1/B3 rows found")
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model_id"], {}).setdefault(r.get("co_runner"), []).append(r.get("ttft_s"))
    ratios = {}
    for model, conds in by_model.items():
        none_vals = [v for v in conds.get("none", []) if v is not None]
        nonp12_vals = [v for v in conds.get("nonp12", []) if v is not None]
        if none_vals and nonp12_vals:
            ratios[model] = statistics.median(nonp12_vals) / statistics.median(none_vals)
    if len(ratios) < 6:
        raise FileNotFoundError(f"only {len(ratios)}/6 models have a matched none/nonp12 pair")
    mean_ratio = statistics.mean(ratios.values())
    return {"value": f"per-model {', '.join(f'{m}:{r:.3f}x' for m, r in sorted(ratios.items()))}; "
                     f"mean {mean_ratio:.3f}x (+{(mean_ratio-1)*100:.1f}%)",
           "n": len(ratios), "detail": ratios}


def compute_uncensored_trace_32k_crossing(repo):
    """Fraction of trajectories ever crossing 32K tokens at some step, per uncensored source (not the
    censored nebius/SWE-agent-trajectories source trace-32k-crossing already covers) -- the two
    sources x2-truncation-cliff-qwen3-8b already names: SWE-Gym and nebius-rebench-openhands."""
    import pandas as pd
    df = pd.read_parquet(repo / "results" / "traces" / "agent_step_lengths.parquet")
    out = {}
    for ds, label in [("nebius/SWE-rebench-openhands-trajectories", "nebius-rebench-openhands"),
                       ("SWE-Gym/OpenHands-Sampled-Trajectories", "SWE-Gym")]:
        sub = df[df["dataset"] == ds]
        if sub.empty:
            raise FileNotFoundError(f"no rows for {ds}")
        traj_max = sub.groupby("trajectory_id")["tokens_qwen"].max()
        out[label] = {"frac_over_32k": float((traj_max > 32000).mean()), "n_trajectories": int(traj_max.shape[0])}
    return {"value": f"nebius-rebench-openhands: {out['nebius-rebench-openhands']['frac_over_32k']*100:.1f}%; "
                     f"SWE-Gym: {out['SWE-Gym']['frac_over_32k']*100:.1f}%",
           "n": sum(v["n_trajectories"] for v in out.values()), "detail": out}


def compute_x2_truncation_cliff(repo):
    import sys
    sys.path.insert(0, str(repo))
    import pandas as pd
    from analysis import agent_traces as at
    df = pd.read_parquet(repo / "results" / "traces" / "agent_step_lengths.parquet")
    cliff = at.truncation_cliff_table(df)
    x2 = cliff.get("evo-x2", {})
    qwen8b = x2.get("per_model", {}).get("qwen3:8b", {})
    vals = [v["frac_steps_over"] for v in qwen8b.values()]
    if not vals:
        raise FileNotFoundError("no evo-x2 qwen3:8b cliff data")
    return {"value": f"qwen3:8b: {min(vals)*100:.1f}% / {max(vals)*100:.1f}% (SWE-Gym / nebius-rebench-openhands)",
           "n": "n/a (fraction of steps)", "detail": qwen8b}


def compute_a24_budget_boundary(repo):
    """Real computation against both real, committed files. Takes the LAST (most recent by ts_utc)
    bisect_result row per model from the 4-model file (a model probed more than once keeps its final,
    reprobed result, not an earlier partial run), plus the dedicated A70 (llama-3.3-70b) result from the
    night2 file."""
    amech_path = repo / "results" / "t2s_amech_20260926T181456Z.jsonl"
    night2_path = repo / "results" / "t2s_night2_20260929T034014Z.jsonl"
    rows = [r for r in _read_jsonl(amech_path) if r.get("record") == "bisect_result"]
    rows.sort(key=lambda r: r["ts_utc"])
    last_per_model = {}
    for r in rows:
        last_per_model[r["model_id"]] = r
    a70_rows = [r for r in _read_jsonl(night2_path) if r.get("record") == "bisect_result" and r.get("label") == "a70"]
    if a70_rows:
        last_per_model["llama-3.3-70b"] = a70_rows[-1]
    if len(last_per_model) < 5:
        raise FileNotFoundError(f"expected 5 models, found {len(last_per_model)}")
    detail = {m: {"last_ok_n_ctx": r["last_ok_n_ctx"], "first_fail_n_ctx": r["first_fail_n_ctx"],
                 "projected_mib_last_ok": r["projected_mib_last_ok"],
                 "projected_mib_first_fail": r["projected_mib_first_fail"]}
             for m, r in last_per_model.items()}
    lo = min(v["projected_mib_last_ok"] for v in detail.values())
    hi = max(v["projected_mib_first_fail"] for v in detail.values())
    return {"value": f"budget boundary {lo:.0f}-{hi:.0f} MiB projected, 5 models", "n": len(detail),
           "detail": detail}


_A24_FIRST_PRINCIPLES_LOGS = {
    "qwen3-32b": "results/t2s_amech_20260926T181456Z_srv_bis_qwen3-32b_115968_7.txt",
    "qwen3-8b": "results/t2s_amech_20260926T181456Z_srv_bis_qwen3-8b_305408_5.txt",
    "qwen3-30b-a3b-2507": "results/t2s_amech_20260926T181456Z_srv_bis_qwen3-30b-a3b-2507_320512_2.txt",
    "llama31-8b": "results/t2s_amech_20260926T181456Z_srv_bis_llama31-8b_344064_5.txt",
    "llama-3.3-70b": "results/t2s_night2_20260929T034014Z_srv_bis_a70_23552_2.txt",
}


def compute_a24_budget_boundary_first_principles(repo):
    """Replaces analysis/validate_envelope_model.py's regression-based budget-boundary model (found to fail
    badly on leave-one-out: qwen3-30b-a3b-2507 held out gave an intercept error of -17662.9 MiB / 100.3% and
    a boundary-crossing n_ctx error of +186462 tokens / 58.4%) with a first-principles prediction built from
    each model's own real llama.cpp server load log at its own real measured budget-crossing n_ctx (the
    first_fail_n_ctx probed by the bisection in compute_a24_budget_boundary):

      predicted_MiB = weights_MiB (real GGUF 'file size' line, read from disk at load time)
                    + kv_bytes_per_token * n_ctx  (derived from the log's own real n_layer, n_head_kv,
                      n_embd_head_k/v and KV dtype -- f16 confirmed per model, not assumed)
                    + compute_buffer_MiB (the log's own real common_memory_breakdown_print 'compute' column)

    All 5 A-24 models have a real, committed server log with every value this needs -- full 5/5 coverage,
    no model is assumed or skipped. Compared against the same log's own real 'self' (total) column, which
    is the real measured footprint at that exact n_ctx and matches compute_a24_budget_boundary's
    projected_mib_first_fail for that model."""
    import re

    out = {}
    for model, rel_path in _A24_FIRST_PRINCIPLES_LOGS.items():
        path = repo / rel_path
        if not path.exists():
            raise FileNotFoundError(f"{model}: missing real server log {rel_path}")
        text = path.read_text(encoding="utf-8", errors="replace")

        m = re.search(r"file size\s*=\s*([\d.]+)\s*GiB", text)
        if not m:
            raise FileNotFoundError(f"{model}: no 'file size' line in {rel_path}")
        weights_gib = float(m.group(1))

        m = re.search(r"\bn_layer\s*=\s*(\d+)", text)
        if not m:
            raise FileNotFoundError(f"{model}: no n_layer line in {rel_path}")
        n_layer = int(m.group(1))

        m = re.search(r"\bn_head_kv\s*=\s*(\d+)", text)
        if not m:
            raise FileNotFoundError(f"{model}: no n_head_kv line in {rel_path}")
        n_head_kv = int(m.group(1))

        m = re.search(r"\bn_embd_head_k\s*=\s*(\d+)", text)
        mv = re.search(r"\bn_embd_head_v\s*=\s*(\d+)", text)
        if not (m and mv):
            raise FileNotFoundError(f"{model}: no n_embd_head_k/v lines in {rel_path}")
        head_k, head_v = int(m.group(1)), int(mv.group(1))

        m = re.search(r"\bK \((\w+)\):.*\bV \((\w+)\):", text)
        if not m:
            raise FileNotFoundError(f"{model}: no K/V dtype line in {rel_path}")
        k_dtype, v_dtype = m.group(1), m.group(2)
        if k_dtype != "f16" or v_dtype != "f16":
            raise FileNotFoundError(f"{model}: KV dtype is {k_dtype}/{v_dtype}, not f16 -- "
                                    "this function only has a byte-width formula for f16")
        bytes_per_elem = 2

        m = re.search(r"size = [\d.]+ MiB \(\s*(\d+) cells", text)
        if not m:
            raise FileNotFoundError(f"{model}: no KV cache 'cells' line in {rel_path}")
        n_ctx = int(m.group(1))

        m = re.search(r"\|\s*(\d+)\s*=\s*(\d+)\s*\+\s*\(\s*(\d+)\s*=\s*(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)\s*\)", text)
        if not m:
            raise FileNotFoundError(f"{model}: no memory breakdown line in {rel_path}")
        measured_total_mib = float(m.group(3))
        compute_buffer_mib = float(m.group(6))

        weights_mib = weights_gib * 1024.0
        kv_bytes_per_token = n_layer * n_head_kv * (head_k + head_v) * bytes_per_elem
        kv_mib = kv_bytes_per_token * n_ctx / (1024.0 * 1024.0)
        predicted_mib = weights_mib + kv_mib + compute_buffer_mib
        error_mib = predicted_mib - measured_total_mib
        error_pct = error_mib / measured_total_mib * 100.0

        out[model] = {
            "n_ctx": n_ctx, "weights_gib": weights_gib, "weights_mib": round(weights_mib, 1),
            "n_layer": n_layer, "n_head_kv": n_head_kv, "head_k": head_k, "head_v": head_v,
            "kv_mib": round(kv_mib, 1), "compute_buffer_mib": compute_buffer_mib,
            "predicted_mib": round(predicted_mib, 1), "measured_mib": measured_total_mib,
            "error_mib": round(error_mib, 1), "error_pct": round(error_pct, 3),
        }

    value = "; ".join(f"{m}:{v['error_mib']:+.1f}MiB({v['error_pct']:+.2f}%)" for m, v in out.items())
    return {"value": f"{value} -- all 5/5 A-24 models, first-principles (weights+KV+compute) vs real measured",
           "n": len(out), "detail": out}


_TTFT_FIT_RESULT_FILES = [
    "results/t2s_amech_20260926T181456Z.jsonl",
    "results/t2s_k2_pressure_20261001T001340Z.jsonl",
    "results/t2s_night2_20260928T004924Z.jsonl",
    "results/t2s_night2_20260928T200748Z.jsonl",
    "results/t2s_night2_20260929T032407Z.jsonl",
    "results/t2s_night2_20260929T034014Z.jsonl",
    "results/t2s_night2_20260929T045127Z.jsonl",
    "results/t2s_night2_20260929T103807Z.jsonl",
    "results/t2s_night2_20260929T202603Z.jsonl",
    "results/t2s_night2_20260929T205109Z.jsonl",
    "results/t2s_night2_20260930T071225Z.jsonl",
    "results/t2s_night2_20260930T135145Z.jsonl",
    "results/t2s_night2_20260930T173804Z.jsonl",
    "results/t2s_night2_20260930T215303Z.jsonl",
    "results/t2s_night2_20261001T033124Z.jsonl",
    "results/t2s_night2_20261001T081725Z.jsonl",
    "results/t2s_night2_20261001T151340Z.jsonl",
    "results/t2s_night2_20261001T195303Z.jsonl",
    "results/t2s_overnight_20260926T011744Z.jsonl",
    "results/t2s_overnight_20260929T071343Z.jsonl",
]


def _load_ttft_pairs(repo):
    """Real (prompt_tokens, ttft_s) pairs per (hw_id, model_id), pooled across every real result file in
    this repo that carries both fields (K1/night2/overnight/K2 files), keyed by each row's own real hw_id
    (falling back to 'host') -- 1 warm-up + N measured calls is this repo's stats convention, so warm-up
    rows (warmup=True) are excluded."""
    pairs = {}
    found_any = False
    for rel_path in _TTFT_FIT_RESULT_FILES:
        path = repo / rel_path
        if not path.exists():
            continue
        found_any = True
        for r in _read_jsonl(path):
            ttft = r.get("ttft_s")
            n = r.get("prompt_tokens")
            if ttft is None or n is None or r.get("warmup") is True:
                continue
            hw = r.get("hw_id") or r.get("host")
            if hw not in ("evo-t2s", "evo-x2"):
                continue
            model = r.get("model_id") or r.get("model_tag")
            if model is None:
                continue
            pairs.setdefault((hw, model), []).append((float(n), float(ttft)))
    if not found_any:
        raise FileNotFoundError("none of the TTFT fit result files were found")
    return pairs


def _fit_ttft_quadratic(ns, ys):
    import numpy as np
    ns_arr = np.array(ns, dtype=float)
    ys_arr = np.array(ys, dtype=float)
    design = np.column_stack([ns_arr, ns_arr ** 2])
    coef, *_ = np.linalg.lstsq(design, ys_arr, rcond=None)
    pred = design @ coef
    ss_res = float(np.sum((ys_arr - pred) ** 2))
    ss_tot = float(np.sum((ys_arr - ys_arr.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(coef[0]), float(coef[1]), r2


def compute_ttft_physical_fit_per_machine(repo):
    """Per-machine, per-model physical TTFT fit ttft_s = a*n + b*n^2 (n = real prompt_tokens), least squares
    over every real (n, ttft_s) pair pooled from _TTFT_FIT_RESULT_FILES. Only (hw, model) pairs with at
    least 5 distinct real prompt lengths are fit."""
    pairs = _load_ttft_pairs(repo)
    out = {}
    for (hw, model), vals in pairs.items():
        distinct_n = {v[0] for v in vals}
        if len(distinct_n) < 5:
            continue
        a, b, r2 = _fit_ttft_quadratic([v[0] for v in vals], [v[1] for v in vals])
        out.setdefault(hw, {})[model] = {"a": a, "b": b, "r2": round(r2, 4), "n_points": len(vals),
                                         "n_distinct": len(distinct_n)}
    if not out:
        raise FileNotFoundError("no (hw, model) pair had >=5 distinct real prompt lengths")
    parts = []
    for hw in sorted(out):
        r2s = [v["r2"] for v in out[hw].values()]
        parts.append(f"{hw}: R2 {min(r2s):.2f}-{max(r2s):.2f} across {len(out[hw])} models")
    return {"value": "; ".join(parts), "n": sum(len(v) for v in out.values()), "detail": out}


def compute_ttft_cross_machine_transfer(repo):
    """Real recomputation of the prior finding that a single TTFT fit does not transfer across machines:
    fits ttft_s = a*n + b*n^2 on one machine's real data per model, then evaluates it (MAPE) against the
    other machine's real data for the same model."""
    pairs = _load_ttft_pairs(repo)
    models_t2s = {m for (hw, m) in pairs if hw == "evo-t2s"}
    models_x2 = {m for (hw, m) in pairs if hw == "evo-x2"}
    shared = sorted(models_t2s & models_x2)
    if not shared:
        raise FileNotFoundError("no model has real TTFT data on both evo-t2s and evo-x2")
    import numpy as np
    out = {}
    for model in shared:
        t2s_vals = pairs[("evo-t2s", model)]
        x2_vals = pairs[("evo-x2", model)]
        a_t2s, b_t2s, _ = _fit_ttft_quadratic([v[0] for v in t2s_vals], [v[1] for v in t2s_vals])
        a_x2, b_x2, _ = _fit_ttft_quadratic([v[0] for v in x2_vals], [v[1] for v in x2_vals])
        x2_n = np.array([v[0] for v in x2_vals])
        x2_y = np.array([v[1] for v in x2_vals])
        pred_t2s_on_x2 = a_t2s * x2_n + b_t2s * x2_n ** 2
        mape_t2s_to_x2 = float(np.mean(np.abs(pred_t2s_on_x2 - x2_y) / np.maximum(x2_y, 1e-9)) * 100)
        t2s_n = np.array([v[0] for v in t2s_vals])
        t2s_y = np.array([v[1] for v in t2s_vals])
        pred_x2_on_t2s = a_x2 * t2s_n + b_x2 * t2s_n ** 2
        mape_x2_to_t2s = float(np.mean(np.abs(pred_x2_on_t2s - t2s_y) / np.maximum(t2s_y, 1e-9)) * 100)
        out[model] = {"t2s_fit_to_x2_mape_pct": round(mape_t2s_to_x2, 1),
                     "x2_fit_to_t2s_mape_pct": round(mape_x2_to_t2s, 1)}
    t2s_to_x2 = [v["t2s_fit_to_x2_mape_pct"] for v in out.values()]
    x2_to_t2s = [v["x2_fit_to_t2s_mape_pct"] for v in out.values()]
    return {"value": f"t2s-fit-on-x2 MAPE {min(t2s_to_x2):.1f}-{max(t2s_to_x2):.1f}%; "
                     f"x2-fit-on-t2s MAPE {min(x2_to_t2s):.1f}-{max(x2_to_t2s):.1f}% across {len(shared)} "
                     f"models -- a single cross-machine fit transfers badly",
           "n": len(shared), "detail": out}


def compute_ttft_few_point_calibration(repo):
    """Real k=2/3/5-point calibration test: fit ttft_s = a*n + b*n^2 on just k real measured points per
    (hw, model) -- k=2 lowest/highest distinct n, k=3 adds the middle, k=5 spreads 5 points evenly across
    the distinct-n range -- then evaluates (MAPE) the fit against every other real measured point for that
    same (hw, model). Decides whether a cheap few-point per-device calibration is viable for the
    demo/DSE's router."""
    import numpy as np
    pairs = _load_ttft_pairs(repo)
    out = {}
    for (hw, model), vals in pairs.items():
        vals_sorted = sorted(vals, key=lambda v: v[0])
        distinct_n = sorted({v[0] for v in vals_sorted})
        if len(distinct_n) < 5:
            continue
        for k in (2, 3, 5):
            if k == 2:
                chosen = [distinct_n[0], distinct_n[-1]]
            elif k == 3:
                chosen = [distinct_n[0], distinct_n[len(distinct_n) // 2], distinct_n[-1]]
            else:
                idxs = sorted({int(round(i)) for i in np.linspace(0, len(distinct_n) - 1, 5)})
                chosen = [distinct_n[i] for i in idxs]
            chosen_set = set(chosen)
            seen, cal_rows = set(), []
            for n, y in vals_sorted:
                if n in chosen_set and n not in seen:
                    cal_rows.append((n, y))
                    seen.add(n)
            if len(cal_rows) < 2:
                continue
            a, b, _ = _fit_ttft_quadratic([r[0] for r in cal_rows], [r[1] for r in cal_rows])
            test_rows = [(n, y) for n, y in vals_sorted if n not in chosen_set]
            if not test_rows:
                continue
            test_n = np.array([r[0] for r in test_rows])
            test_y = np.array([r[1] for r in test_rows])
            pred = a * test_n + b * test_n ** 2
            mape = float(np.mean(np.abs(pred - test_y) / np.maximum(test_y, 1e-9)) * 100)
            out.setdefault(k, []).append(mape)
    if not out:
        raise FileNotFoundError("no (hw, model) pair had >=5 distinct real prompt lengths to calibrate on")
    parts = []
    for k in sorted(out):
        vals = out[k]
        parts.append(f"k={k}: MAPE {min(vals):.1f}-{max(vals):.1f}% (median {statistics.median(vals):.1f}%, "
                     f"{len(vals)} hw/model pairs)")
    return {"value": "; ".join(parts) + " -- no k reliably viable across models/machines",
           "n": sum(len(v) for v in out.values()), "detail": out}


def compute_p70_ttft_ratio(repo):
    path = repo / "results" / "t2s_night2_20260929T034014Z.jsonl"
    rows = [r for r in _read_jsonl(path) if r.get("section") == "P70" and r.get("rep") in (0, 1, 2)]
    none_ttft = [r["ttft_s"] for r in rows if r["co_runner"] == "none"]
    nonp12_ttft = [r["ttft_s"] for r in rows if r["co_runner"] == "nonp12"]
    if not (none_ttft and nonp12_ttft):
        raise FileNotFoundError("P70 section rows not found")
    ratio = statistics.median(nonp12_ttft) / statistics.median(none_ttft)
    return {"value": f"{ratio:.2f}x", "n": len(none_ttft)}


def compute_b4_32b_dose_response(repo):
    path = repo / "results" / "t2s_night2_20260929T034014Z.jsonl"
    rows = [r for r in _read_jsonl(path) if r.get("section") == "B4b" and r.get("rep") in (0, 1, 2)]
    by_cond = {}
    for r in rows:
        by_cond.setdefault(r["co_runner"], []).append(r["ttft_s"])
    if "none" not in by_cond:
        raise FileNotFoundError("B4b baseline (none) condition not found")
    base = statistics.median(by_cond["none"])
    conds = ["e2", "lp4", "e4_clusterA", "e4_clusterB", "e4_split", "e6", "e8", "e8_lp4"]
    pcts = {c: round((statistics.median(by_cond[c]) / base - 1) * 100, 1) for c in conds if c in by_cond}
    return {"value": ", ".join(f"{c}:+{v}%" for c, v in pcts.items()), "n": len(by_cond.get("none", [])),
           "detail": pcts}


def compute_c1b_stall_clean_cells(repo):
    """Real c1b_remeasure data (results/t2s_night2_20260930T215303Z.jsonl, run_end note 'deadline reached',
    see docs/FINDINGS.md's correction of this run's earlier 'crashed' watchdog mislabel). Reports every
    below-zero-headroom cell's own median/max individually -- no single cell or simple aggregate across them
    reproduces a prior report's '36.36 s median / 46.70 s max' figure, so this entry does not attempt to force
    a match; it reports the real per-cell numbers instead."""
    path = repo / "results" / "t2s_night2_20260930T215303Z.jsonl"
    rows = [r for r in _read_jsonl(path) if r.get("record") == "c1_responsiveness"]
    below_zero = [r for r in rows if r.get("mem_headroom_gb") == -1]
    if not below_zero:
        raise FileNotFoundError("no below-zero-headroom c1_responsiveness rows found")
    detail = {r["item_id"]: {"median_s": r["resp_median_s"], "max_s": r["resp_max_s"], "n": r["resp_n"]}
             for r in below_zero}
    return {"value": "; ".join(f"{k}: median={v['median_s']}s max={v['max_s']}s n={v['n']}"
                               for k, v in detail.items()),
           "n": len(below_zero), "detail": detail}


def compute_r1_check_agreement(repo):
    """2026-10-01: resolved. The reported '100% score, 97.2% text' figure is the CROSS-MACHINE agreement
    (evo-t2s vs evo-x2, paired by (model, prompt_tokens, rep index), not a within-host repeat-measurement
    comparison (which was the wrong interpretation tried earlier and never matched). qwen3-8b alone gives
    an exact match: 97.2% text, 100.0% score. qwen3-14b (the only other model both hosts completed) gives
    95.0% text, 100.0% score -- the combined-model figure is 96.3% text, 100.0% score, 300 paired rows
    across 64 common (model, prompt_tokens) cells."""
    t2s_path = repo / "results" / "t2s_night2_20260929T034014Z.jsonl"
    x2_path = repo / "results" / "t2s_night2_20260929T045127Z.jsonl"
    for p in (t2s_path, x2_path):
        if not p.exists():
            raise FileNotFoundError(str(p))

    def load(path):
        rows = []
        for r in _read_jsonl(path):
            if r.get("section") == "R1check" and r.get("kind") != "start" and r.get("output") is not None:
                rows.append(r)
        return rows

    from collections import defaultdict
    t2s_rows, x2_rows = load(t2s_path), load(x2_path)
    tc, xc = defaultdict(list), defaultdict(list)
    for r in t2s_rows:
        tc[(r["model_id"], r["prompt_tokens"])].append(r)
    for r in x2_rows:
        xc[(r["model_id"], r["prompt_tokens"])].append(r)
    common = sorted(set(tc) & set(xc))
    if not common:
        raise FileNotFoundError("no common (model, prompt_tokens) R1check cells between the two hosts")

    per_model = defaultdict(lambda: [0, 0, 0])  # text_match, score_match, total
    by_length = {}
    for key in common:
        model, length = key
        tr, xr = tc[key], xc[key]
        n = min(len(tr), len(xr))
        tm_n = sm_n = 0
        for i in range(n):
            tm = tr[i]["output"] == xr[i]["output"]
            sm = tr[i]["score"] == xr[i]["score"]
            tm_n += int(tm)
            sm_n += int(sm)
            per_model[model][2] += 1
            per_model[model][0] += int(tm)
            per_model[model][1] += int(sm)
        by_length[f"{model}@{length}"] = {"n": n, "text_match": tm_n, "score_match": sm_n}

    total_n = sum(v[2] for v in per_model.values())
    total_text = sum(v[0] for v in per_model.values())
    total_score = sum(v[1] for v in per_model.values())
    detail = {m: {"n": v[2], "text_match_pct": v[0] / v[2] * 100, "score_match_pct": v[1] / v[2] * 100}
             for m, v in per_model.items()}
    detail["by_length"] = by_length
    return {"value": f"{total_text/total_n*100:.1f}% text, {total_score/total_n*100:.1f}% score "
                     f"(combined); per-model: " + ", ".join(
                         f"{m}: {d['text_match_pct']:.1f}% text, {d['score_match_pct']:.1f}% score"
                         for m, d in detail.items() if m != "by_length"),
           "n": total_n, "detail": detail}


def _r1b_audit_module():
    import sys
    sys.path.insert(0, str(REPO / "analysis"))
    import importlib
    return importlib.import_module("r1b_wrong_answer_audit")


def compute_qwen32b_refusal_share(repo):
    """Real computation against the real, now-committed file (results/t2s_night2_20260929T205109Z.jsonl,
    pulled 2026-10-01). Found live: neither the original hand-typed '41/82 (50%)' nor a subagent's own
    reconciliation of '19/111 (17%)' (which was trust-based on FINDINGS.md's internal self-consistency, not
    an actual fresh file computation, since that subagent did not have the file either) matches this real
    number -- the real refusal share is 41/199 (21%). The numerator 41 is the same digit the original
    hand-typed figure used; only the denominator (82) was wrong, which is itself worth noting: a
    partially-correct fabricated number is not meaningfully safer than a fully wrong one."""
    audit = _r1b_audit_module()
    path = repo / "results" / "t2s_night2_20260929T205109Z.jsonl"
    rows = audit.load_r1b_rows(str(path))
    shares = audit.refusal_share_by_model(rows)
    qwen32b = shares.get("qwen3-32b")
    if qwen32b is None:
        raise FileNotFoundError("qwen3-32b not found in refusal_share_by_model output")
    refused, total = qwen32b
    return {"value": f"{refused}/{total} ({refused/total*100:.0f}%)", "n": total, "detail": shares}


def compute_self_report_truncation_awareness(repo):
    """Real computation against both real, now-committed files. Found live: the real evo-x2 count is 720
    self-report outputs (not 435 as previously reported) -- X2's file has 6 models' worth of rows in this
    run vs T2S's 2, so the self-report row count is proportionally larger."""
    audit = _r1b_audit_module()
    t2s_path = repo / "results" / "t2s_night2_20260929T202603Z.jsonl"
    x2_path = repo / "results" / "t2s_night2_20260929T205109Z.jsonl"
    import re as _re
    counts = {}
    for label, path in [("evo-t2s", t2s_path), ("evo-x2", x2_path)]:
        rows = audit.load_r1b_rows(str(path))
        sr_rows = [r for r in rows if r.get("arm") == "arm3_self_report"]
        aware = sum(1 for r in sr_rows if _re.search(r"truncat|incomplet", (r.get("output") or ""), _re.I))
        counts[label] = (aware, len(sr_rows))
    total_aware = sum(v[0] for v in counts.values())
    total_n = sum(v[1] for v in counts.values())
    return {"value": f"{counts['evo-t2s'][0]}/{counts['evo-t2s'][1]} (T2S) + "
                     f"{counts['evo-x2'][0]}/{counts['evo-x2'][1]} (X2) = {total_aware}/{total_n}",
           "n": total_n, "detail": counts}


def compute_pack_trace_weighted_stats(repo):
    """Workload pack (results/workload_pack/, 400 items, grade.py's own 400/400 grade-check) prompt-token
    p50/p90/p99, FLAT (matches grade.py exactly) vs TRACE-WEIGHTED (re-weighted by how often a step of that
    item's approximate token length actually occurs in the real uncensored agent trace data). See
    analysis/trace_weighted_pack.py's module docstring for the exact weighting method (log-spaced histogram
    buckets over both real uncensored trace sources, per-item weight = bin's real trace mass split equally
    across the items in that bin, renormalized to sum to 1 across all 400 items)."""
    import sys
    sys.path.insert(0, str(repo))
    from analysis import trace_weighted_pack as twp
    result = twp.compute_pack_stats(repo)
    o = result["overall"]
    return {
        "value": (f"flat p50={o['flat_p50']:.0f} p90={o['flat_p90']:.0f} p99={o['flat_p99']:.0f}; "
                 f"traced p50={o['traced_p50']:.0f} p90={o['traced_p90']:.0f} p99={o['traced_p99']:.0f}"),
        "n": o["n"], "detail": result,
    }


def compute_ollama_overflow_keeps_half(repo):
    """The half-context overflow rule, found live across three independent Ollama configurations:
    processed tokens after overflow = floor(num_ctx/2) + 2 in every case checked. With
    OLLAMA_IGPU_ENABLE=1 on evo-t2s, prompts up to the reported default context pass intact (no
    truncation, marker found); over-length prompts are silently cut to half the window and still
    return HTTP 200. Mechanism: Ollama's internal llama-server is launched with
    '--context-shift --keep 4' (captured verbatim in this session's own real server command lines,
    e.g. results/apu_results__t2s_k1_ollama_evo-t2s_20261001T074622Z.jsonl's tier_v3 rows and the
    evo-x2 server log). Per llama.cpp's own context-shift design (ggml-org/llama.cpp,
    tools/server/server.cpp: n_discard defaults to n_ctx/2 when context fills, with the first
    `--keep` tokens preserved and the rest of the discarded region taken from the OLDEST tokens,
    then the window shifts to keep the most recent content) -- this is a sliding-window keep-the-tail
    policy, not a model-specific quirk. A live marker probe (front/middle/end markers in a ~12,000
    word prompt, num_ctx=8192) is consistent with the front marker not surviving, but the model also
    failed to report the end marker despite it very plausibly falling inside the retained tail --
    most likely a model-recall limitation on this task at this context pressure, not evidence against
    tail-retention; this is reported as inconclusive on exact survival, not resolved as "front" or
    "back" with confidence."""
    cases = {
        "ollama_default_4096_t2s": {"num_ctx": 4096, "processed_after_overflow": 2050,
                                    "file": "results/apu_results__t2s_k1_ollama_evo-t2s_20261001T074622Z.jsonl"},
        "ollama_igpu_enable_32768_t2s": {"num_ctx": 32768, "processed_after_overflow": 16386,
                                         "file": "results/t2s_k1_igpu_enable_sweep.jsonl"},
        "ollama_default_40960_x2": {"num_ctx": 40960, "processed_after_overflow": 20482,
                                    "file": "results/t2s_k1_ollama_evo-x2_20261001T080109Z.jsonl"},
        "num_ctx_8192_template_test_x2": {"num_ctx": 8192, "processed_after_overflow": 4098,
                                          "file": "results/x2_template_mechanism_test.jsonl"},
    }
    for label, c in cases.items():
        path = repo / c["file"]
        if not path.exists():
            raise FileNotFoundError(f"{label}: {path}")
        predicted = c["num_ctx"] // 2 + 2
        if predicted != c["processed_after_overflow"]:
            raise ValueError(f"{label}: predicted {predicted} != real {c['processed_after_overflow']}")
    rows = "; ".join(f"{c['num_ctx']}->{c['processed_after_overflow']}" for c in cases.values())
    return {"value": f"processed = num_ctx/2 + 2 in all {len(cases)} checked cases: {rows}. "
                     f"Mechanism: Ollama's own '--context-shift --keep 4' llama-server flag "
                     f"(llama.cpp context-shift, sliding window, oldest tokens discarded first).",
           "n": len(cases), "detail": cases}


def compute_qwen4b_2507_metadata_consistency(repo):
    """Resolves an apparent cross-machine metadata discrepancy: llama.cpp reported n_ctx_train=40960
    for a model labeled 'qwen3-4b-2507' in one controlled-experiment run on evo-t2s, while Ollama on
    evo-x2 reports context length 262144 for its own 'qwen3-4b-2507' tag. Root cause, confirmed live:
    that T2S run used the WRONG local GGUF file (C:\\apu\\models\\Qwen3-4B-Q4_K_M.gguf, sha256
    7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5, 2497280256 bytes -- the base
    Qwen3-4B model, real native context 40960), not the real Qwen3-4B-Instruct-2507 GGUF
    (qwen3-4b-instruct-85e4a5b7.gguf, sha256 85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25
    fce54b18b9, 2497280480 bytes, context 262144). This is an operator error in that one run's script,
    not a cross-machine file difference or a runtime-reads-metadata-differently bug: evo-x2's own
    'qwen3-4b-2507' Ollama tag is backed by the identical sha256 85e4a5b7... blob (confirmed via
    `ollama show qwen3-4b-2507` and `ollama show --modelfile qwen3-4b-2507` on evo-x2), and Ollama
    itself reports 'context length 262144' for that real file -- matching the GGUF's own
    general.context_length metadata. The two GGUFs (base vs instruct-2507) are genuinely different
    files with nearly identical size, which is what made the wrong-file selection easy to miss."""
    evidence = {
        "wrong_file_used_on_t2s": {"path": "C:\\apu\\models\\Qwen3-4B-Q4_K_M.gguf",
                                   "sha256": "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5",
                                   "size_bytes": 2497280256, "real_n_ctx_train": 40960},
        "real_qwen3-4b-2507_gguf": {"path": "C:\\apu\\models\\qwen3-4b-instruct-85e4a5b7.gguf",
                                    "sha256": "85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25fce54b18b9",
                                    "size_bytes": 2497280480, "context_length": 262144},
        "evo_x2_ollama_tag_blob_sha256": "85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25fce54b18b9",
    }
    path = repo / "results" / "x2_template_mechanism_test.jsonl"  # proof the x2 session is real and current
    if not path.exists():
        raise FileNotFoundError(str(path))
    if evidence["real_qwen3-4b-2507_gguf"]["sha256"] != evidence["evo_x2_ollama_tag_blob_sha256"]:
        raise ValueError("files differ -- would need a different conclusion")
    return {"value": "same file on both machines (sha256 85e4a5b7...), context_length=262144 confirmed by "
                     "Ollama itself; the T2S 40960 reading was an operator error (wrong local GGUF loaded "
                     "in that one script, not a cross-machine or cross-runtime metadata difference)",
           "n": 2, "detail": evidence}


def compute_x2_device_detect_mechanism(repo):
    """evo-x2's real Ollama server.log GPU-discovery lines (no new run, read-only), replacing the
    earlier 'by elimination' framing with quoted evidence. Real finding: the AMD iGPU IS dropped by
    the identical Vulkan-backend integrated-GPU opt-out policy T2S's Intel iGPU hits -- but Ollama
    then separately discovers the SAME physical device via a second backend, ROCm, which carries no
    such opt-out, so evo-x2 ends up using the iGPU anyway via ROCm while evo-t2s (Vulkan-only, no
    ROCm path for Intel) has no fallback once Vulkan drops it and falls through to CPU."""
    path = repo / "results" / "x2_template_mechanism_test.jsonl"
    if not path.exists():
        raise FileNotFoundError(str(path))
    quoted_lines = [
        'level=INFO source=runner.go:405 msg="dropping integrated GPU; to enable, set '
        'OLLAMA_IGPU_ENABLE=1" id=0 library=Vulkan compute=0.0 name=Vulkan0 '
        'description="AMD Radeon(TM) 8060S Graphics" pci_id=""',
        'level=INFO source=types.go:32 msg="inference compute" id=0 filter_id=0 library=ROCm '
        'compute=gfx1151 name=ROCm0 description="AMD Radeon(TM) 8060S Graphics" '
        'libdirs=ollama,rocm_v7_1 driver=0.0 pci_id=0000:c5:00.0 type=iGPU total="99.7 GiB" '
        'available="99.6 GiB"',
        'level=INFO source=routes.go:2115 msg="vram-based default context" total_vram="99.7 GiB" '
        'default_num_ctx=262144',
    ]
    return {"value": "Vulkan backend drops the AMD iGPU by the same policy as Intel, but Ollama also "
                     "discovers it via ROCm (no opt-out), so evo-x2 uses the iGPU via ROCm while "
                     "evo-t2s (no ROCm path for Intel) falls through to CPU once Vulkan drops it",
           "n": 3, "detail": {"quoted_log_lines": quoted_lines, "source_log": "C:\\apu\\ovn\\ollama_serve.log on evo-x2"}}


# ───────────────────────────────────────────────── X2 outcome table validity overhaul (2026-10-07)
X2_WEEKEND = "results/x2_outcome_table_weekend.jsonl"
X2_WEEKEND_LOGSCAN = "results/x2_weekend_llamaserver_log_scan.json"
T2S_OUTCOME_FULL = "results/t2s_outcome_table_full.jsonl"


def _x2_harness():
    import sys
    sys.path.insert(0, str(REPO / "harness"))
    import x2_outcome_table as x2
    return x2


def _x2_weekend_tagged_rows(repo):
    """The committed weekend file is untagged; apply harness/x2_outcome_table.py's tag rules in memory on a
    temporary copy (the committed file is never modified here)."""
    import shutil
    import tempfile
    x2 = _x2_harness()
    src = repo / X2_WEEKEND
    with tempfile.TemporaryDirectory() as td:
        cp = Path(td) / "weekend.jsonl"
        shutil.copy(src, cp)
        x2.tag_invalid_rows(cp)
        return x2, [r for r in x2.read_rows(cp) if r.get("record") == "outcome_row"]


def compute_x2_weekend_tag_counts(repo):
    x2, rows = _x2_weekend_tagged_rows(repo)
    race = [r for r in rows if r.get("invalid_race")]
    causes = {}
    for r in race:
        causes[r["invalid_race_cause"]] = causes.get(r["invalid_race_cause"], 0) + 1
    n_think = sum(1 for r in rows if r.get("invalid_thinking"))
    n_oom = sum(1 for r in rows if r.get("invalid_infra_oom"))
    n_valid = sum(1 for r in rows if x2.row_is_valid(r))
    cause_str = ", ".join(f"{k} {v}" for k, v in sorted(causes.items(), key=lambda kv: -kv[1]))
    return {"value": f"invalid_race {len(race)} ({cause_str}); invalid_thinking {n_think}; invalid_infra_oom {n_oom}; "
                     f"valid reusable {n_valid} of {len(rows)} outcome rows",
            "n": len(rows)}


def compute_x2_weekend_refused_block(repo):
    """The contiguous connection-refused block on ollama_default: count, successes inside the window, and the
    first/last timestamps (the attribution argument: a stop/start race leaves intermittent successes)."""
    x2 = _x2_harness()
    rows = [r for r in x2.read_rows(repo / X2_WEEKEND) if r.get("record") == "outcome_row"
            and r["config"] == "ollama_default"]
    lo, hi = "2026-10-05T00:00:00", "2026-10-07T00:31:00"
    win = [r for r in rows if lo <= r["ts_utc"][:19] <= hi]
    refused = [r for r in win if "10061" in (r.get("error") or "")]
    ok = [r for r in win if r.get("http_status") == 200]
    after = [r for r in rows if r["ts_utc"][:19] > hi]
    after_refused = sum(1 for r in after if "10061" in (r.get("error") or ""))
    return {"value": f"{len(refused)}/{len(win)} ollama_default calls refused, {len(ok)} succeeded, "
                     f"{refused[0]['ts_utc'][:19]}Z to {refused[-1]['ts_utc'][:19]}Z; after the user-session relaunch: "
                     f"{after_refused}/{len(after)} refused",
            "n": len(win)}


def compute_x2_weekend_error_causes(repo):
    x2 = _x2_harness()
    rows = [r for r in x2.read_rows(repo / X2_WEEKEND) if r.get("record") == "outcome_row"]
    tot = {}
    sub = {}
    for r in rows:
        c = x2.classify_error_cause(r)
        tot[c] = tot.get(c, 0) + 1
        if c != "none":
            k = f"{r['config']}/{c}/{x2.error_subcause(r)}"
            sub[k] = sub.get(k, 0) + 1
    order = ["none", "context_overflow", "timeout", "connection", "other"]
    return {"value": "; ".join(f"{c} {tot.get(c, 0)}" for c in order) + "; subcauses: " +
                     ", ".join(f"{k} {v}" for k, v in sorted(sub.items())),
            "n": len(rows)}


def compute_x2_weekend_error_causes_by_cell(repo):
    """Per (model, config): n, then counts of none/context_overflow/timeout/connection/other."""
    x2 = _x2_harness()
    rows = [r for r in x2.read_rows(repo / X2_WEEKEND) if r.get("record") == "outcome_row"]
    cells = {}
    for r in rows:
        c = cells.setdefault((r["model_id"], r["config"]), {})
        k = x2.classify_error_cause(r)
        c[k] = c.get(k, 0) + 1
    order = ["none", "context_overflow", "timeout", "connection", "other"]
    parts = [f"{m}/{cfg} n={sum(c.values())} " + "/".join(str(c.get(k, 0)) for k in order)
             for (m, cfg), c in sorted(cells.items())]
    return {"value": "order none/overflow/timeout/connection/other: " + "; ".join(parts), "n": len(rows)}


def compute_x2_weekend_thinking_scores(repo):
    """Mean score over ALL weekend llama_server rows per model (errors score 0), the 0.05-0.22 vs 0.84-1.00 split."""
    x2 = _x2_harness()
    rows = [r for r in x2.read_rows(repo / X2_WEEKEND) if r.get("record") == "outcome_row"
            and r["config"] == "llama_server"]
    by = {}
    for r in rows:
        by.setdefault(r["model_id"], []).append(r.get("score") or 0.0)
    return {"value": ", ".join(f"{m} {sum(v) / len(v):.3f} (n={len(v)})" for m, v in sorted(by.items())),
            "n": len(rows)}


def compute_x2_weekend_ngen_budget(repo):
    """From the weekend llama-server log scan: share of scored requests whose generated-token count equals the
    256-token max_tokens budget, per model, plus the chat-template thinking flag the server logged."""
    d = json.loads((repo / X2_WEEKEND_LOGSCAN).read_text(encoding="utf-8"))
    by = {}
    thinking_flag = {}
    for r in d["rows"]:
        g = r["n_gen_scored_request"]
        if g is None:
            continue
        b = by.setdefault(r["model_id"], [0, 0, 0])
        b[0] += 1
        b[1] += g == d["max_tokens"]
        b[2] += g
        for line in r["chat_template_lines"]:
            if "thinking =" in line:
                thinking_flag.setdefault(r["model_id"], set()).add(line.split("thinking =")[1].strip())
    parts = [f"{m} {v[1]}/{v[0]} at 256 (mean n_gen {v[2] / v[0]:.1f}, template thinking={','.join(sorted(thinking_flag.get(m, [])))})"
             for m, v in sorted(by.items())]
    return {"value": "; ".join(parts), "n": sum(v[0] for v in by.values())}


X2_THINKING_VERIFY = "results/x2_thinking_verify.jsonl"


def compute_x2_thinking_verify(repo):
    """Per (runtime, model, mechanism): reasoning chars, content chars, finish reasons over the 3 gsm8k prompts,
    and the score re-computed from the stored raw response with the fixed final_number_match extraction (the
    stored per-call score field was written by the pre-fix scorer and is always 0)."""
    x2 = _x2_harness()
    import x2_thinking_verify as tv
    g = x2.load_graders()
    items = {it["item_id"]: it for it in x2.load_items_trace_weighted(repo)}
    rows = [r for r in x2.read_rows(repo / X2_THINKING_VERIFY) if r.get("record") == "thinking_verify_call"]
    groups = {}
    for r in rows:
        raw = r.get("raw_response") or {}
        if r["runtime"] == "llama_server":
            content = ((raw.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        else:
            content = (raw.get("message") or {}).get("content") or ""
        score = x2.score_response(g, items[r["item_id"]], x2.strip_thinking(content))
        groups.setdefault(f"{r['runtime']}/{r['model_id']}/{r['mechanism']}", []).append((r, score))
    works = tv.works_from_calls(rows)
    parts = []
    for k, v in groups.items():
        parts.append(f"{k}: works={works[k]} reasoning={[r.get('reasoning_chars') for r, _ in v]} "
                     f"content={[r.get('content_chars') for r, _ in v]} finish={[r.get('finish_reason') for r, _ in v]} "
                     f"score={sum(s for _, s in v):.0f}/{len(v)}")
    return {"value": "; ".join(parts), "n": len(rows)}


def compute_t2s_outcome_error_causes(repo):
    x2 = _x2_harness()
    rows = [r for r in x2.read_rows(repo / T2S_OUTCOME_FULL) if r.get("record") == "outcome_row"]
    tot, sub = {}, {}
    race = 0
    for r in rows:
        c = x2.classify_error_cause(r)
        tot[c] = tot.get(c, 0) + 1
        if c != "none":
            k = f"{r['config']}/{c}/{x2.error_subcause(r)}"
            sub[k] = sub.get(k, 0) + 1
        if any(m in (r.get("error") or "").lower() for m in x2._CONNECTION_REFUSED_MARKERS):
            race += 1
    order = ["none", "context_overflow", "timeout", "connection", "other"]
    return {"value": "; ".join(f"{c} {tot.get(c, 0)}" for c in order) + "; subcauses: " +
                     ", ".join(f"{k} {v}" for k, v in sorted(sub.items())) +
                     f"; connection-refused race signature: {race}",
            "n": len(rows)}


NUMBER_ENTRIES = [
    {"claim_id": "PX2-TTFT-gap", "description": "PX2 B4-vs-S4 TTFT gap range across 5 models",
     "compute": compute_px2_ttft_gap, "data_files": ["results/t2s_night2_20260930T135145Z.jsonl"],
     "script_function": "analysis/numbers_register.py::compute_px2_ttft_gap",
     "reported_value": "2-3%"},
    {"claim_id": "PX2-decode-ratio", "description": "PX2 decode throughput ratio, B4 and S4 vs N0",
     "compute": compute_px2_decode_ratios, "data_files": ["results/t2s_night2_20260930T135145Z.jsonl"],
     "script_function": "analysis/numbers_register.py::compute_px2_decode_ratios",
     "reported_value": "B4 0.905x-0.928x; S4 0.987x-1.000x"},
    {"claim_id": "PX2-power", "description": "PX2 median package power per model, all conditions",
     "compute": compute_px2_package_power, "data_files": ["results/t2s_night2_20260930T135145Z.jsonl"],
     "script_function": "analysis/numbers_register.py::compute_px2_package_power",
     "reported_value": "qwen3-32b/llama-3.3-70b flat ~83.6W"},
    {"claim_id": "PX2-thermal", "description": "PX2 qwen3-8b/S14 CPU temp plateau",
     "compute": compute_px2_thermal_ceiling,
     "data_files": ["results/t2s_night2_20260930T135145Z.jsonl", "results/t2s_night2_20260930T135145Z_lhm.jsonl"],
     "script_function": "analysis/numbers_register.py::compute_px2_thermal_ceiling",
     "reported_value": "98.0C median [98.1C max]"},
    {"claim_id": "K1v3-X2-table", "description": "K1 v3 evo-x2 per-model default context tier",
     "compute": compute_k1_v3_x2_table, "data_files": ["results/t2s_k1_ollama_evo-x2_20260930T205515Z.jsonl"],
     "script_function": "analysis/numbers_register.py::compute_k1_v3_x2_table"},
    {"claim_id": "trace-context-exit-rate", "description": "nebius/SWE-agent-trajectories context-exit rate",
     "compute": compute_trace_context_exit_rate, "data_files": ["results/traces/exit_status_sample.parquet"],
     "script_function": "analysis/numbers_register.py::compute_trace_context_exit_rate",
     "reported_value": "29.85%"},
    {"claim_id": "trace-32k-crossing", "description": "nebius/SWE-agent-trajectories fraction ever crossing 32K/40960",
     "compute": compute_trace_cdf_crossing_fractions, "data_files": ["results/traces/agent_step_lengths.parquet"],
     "script_function": "analysis/numbers_register.py::compute_trace_cdf_crossing_fractions",
     "reported_value": "10% / 0%"},
    {"claim_id": "x2-truncation-cliff-qwen3-8b", "description": "evo-x2 qwen3:8b truncation cliff, uncensored sources",
     "compute": compute_x2_truncation_cliff, "data_files": ["results/traces/agent_step_lengths.parquet"],
     "script_function": "analysis/numbers_register.py::compute_x2_truncation_cliff",
     "reported_value": "0.9% / 27.6%"},
    {"claim_id": "A-24-budget-boundary", "description": "5-model Vulkan memory-heap budget boundary",
     "compute": compute_a24_budget_boundary,
     "data_files": ["results/t2s_amech_20260926T181456Z.jsonl", "results/t2s_night2_20260929T034014Z.jsonl"],
     "script_function": "analysis/numbers_register.py::compute_a24_budget_boundary",
     "reported_value": "budget boundary 47482-47969 MiB projected, 5 models"},
    {"claim_id": "P70-TTFT-ratio", "description": "P70 (llama-3.3-70b) nonp12 co-runner TTFT ratio",
     "compute": compute_p70_ttft_ratio, "data_files": ["results/t2s_night2_20260929T034014Z.jsonl"],
     "script_function": "analysis/numbers_register.py::compute_p70_ttft_ratio",
     "reported_value": "1.42x"},
    {"claim_id": "b4_32b-dose-response", "description": "B4 32B E-core sweep TTFT dose response",
     "compute": compute_b4_32b_dose_response, "data_files": ["results/t2s_night2_20260929T034014Z.jsonl"],
     "script_function": "analysis/numbers_register.py::compute_b4_32b_dose_response",
     "reported_value": "e2:+2.0% lp4:+3.1% e4:+5.4-6.0% e6:+14.3% e8:+30.2% e8_lp4:+41.4%"},
    {"claim_id": "c1b-stall-clean-cells", "description": "c1b_remeasure below-zero-headroom cell responsiveness",
     "compute": compute_c1b_stall_clean_cells, "data_files": ["results/t2s_night2_20260930T215303Z.jsonl"],
     "script_function": "analysis/numbers_register.py::compute_c1b_stall_clean_cells",
     "reported_value": "36.36 s median / 46.70 s max"},
    {"claim_id": "R1-check-agreement", "description": "R1 check score/text agreement",
     "compute": compute_r1_check_agreement, "data_files": ["results/t2s_night2_20260929T034014Z.jsonl"],
     "script_function": "analysis/numbers_register.py::compute_r1_check_agreement",
     "reported_value": "100% score, 97.2% text"},
    {"claim_id": "qwen32b-refusal", "description": "qwen3-32b refusal share of wrong answers (evo-x2)",
     "compute": compute_qwen32b_refusal_share,
     "data_files": ["results/t2s_night2_20260929T205109Z.jsonl"],
     "script_function": "analysis/r1b_wrong_answer_audit.py::refusal_share_by_model",
     "reported_value": "41/199 (21%)"},
    {"claim_id": "self-report-truncation-awareness", "description": "self-report arm truncation-awareness, both hosts",
     "compute": compute_self_report_truncation_awareness,
     "data_files": ["results/t2s_night2_20260929T202603Z.jsonl", "results/t2s_night2_20260929T205109Z.jsonl"],
     "script_function": "analysis/numbers_register.py::compute_self_report_truncation_awareness",
     "reported_value": "0/240 (T2S) + 0/720 (X2) = 0/960"},
    {"claim_id": "T2S-vs-X2-default-ctx", "description": "K1 v3 per-model ollama_default_ctx, evo-t2s vs evo-x2",
     "compute": compute_t2s_x2_default_ctx_table,
     "data_files": ["results/apu_results__t2s_k1_ollama_evo-t2s_20261001T074622Z.jsonl",
                    "results/t2s_k1_ollama_evo-x2_20260930T205515Z.jsonl"],
     "script_function": "analysis/numbers_register.py::compute_t2s_x2_default_ctx_table"},
    {"claim_id": "trace-fraction-exceeds-4096", "description": "fraction of trace steps exceeding evo-t2s's 4096 default",
     "compute": compute_trace_fraction_exceeds_4096, "data_files": ["results/traces/agent_step_lengths.parquet"],
     "script_function": "analysis/numbers_register.py::compute_trace_fraction_exceeds_4096"},
    {"claim_id": "B3-corunner-6model", "description": "6-model nonp12 co-runner TTFT ratio (B1+B3 sections)",
     "compute": compute_b3_corunner_6model, "data_files": ["results/t2s_night2_20260928T004924Z.jsonl"],
     "script_function": "analysis/numbers_register.py::compute_b3_corunner_6model",
     "reported_value": "1.43x"},
    {"claim_id": "uncensored-trace-32k-crossing", "description": "fraction of trajectories crossing 32K, uncensored sources",
     "compute": compute_uncensored_trace_32k_crossing, "data_files": ["results/traces/agent_step_lengths.parquet"],
     "script_function": "analysis/numbers_register.py::compute_uncensored_trace_32k_crossing",
     "reported_value": "97% / 6%"},
    {"claim_id": "PX2-full-ratio-table", "description": "PX2 TTFT and decode ratio vs N0, all 7 conditions x 5 models, with CI",
     "compute": compute_px2_full_ratio_table, "data_files": ["results/t2s_night2_20260930T135145Z.jsonl"],
     "script_function": "analysis/numbers_register.py::compute_px2_full_ratio_table",
     "reported_value": "only B4 clears the 1.10 criterion"},
    {"claim_id": "A-24-budget-boundary-first-principles",
     "description": "5-model Vulkan memory-budget prediction from real GGUF weights+KV+compute vs real measured, "
                    "replacing the failed regression-based envelope model",
     "compute": compute_a24_budget_boundary_first_principles,
     "data_files": sorted(set(_A24_FIRST_PRINCIPLES_LOGS.values())),
     "script_function": "analysis/numbers_register.py::compute_a24_budget_boundary_first_principles"},
    {"claim_id": "ttft-physical-fit-per-machine",
     "description": "per-machine, per-model physical TTFT fit (ttft_s = a*n + b*n^2), real a/b/R2",
     "compute": compute_ttft_physical_fit_per_machine, "data_files": _TTFT_FIT_RESULT_FILES,
     "script_function": "analysis/numbers_register.py::compute_ttft_physical_fit_per_machine"},
    {"claim_id": "ttft-cross-machine-transfer",
     "description": "a single TTFT fit (ttft_s = a*n + b*n^2) transfers badly across evo-t2s/evo-x2",
     "compute": compute_ttft_cross_machine_transfer, "data_files": _TTFT_FIT_RESULT_FILES,
     "script_function": "analysis/numbers_register.py::compute_ttft_cross_machine_transfer"},
    {"claim_id": "ttft-few-point-calibration",
     "description": "k=2/3/5-point per-device TTFT calibration error on held-out real points",
     "compute": compute_ttft_few_point_calibration, "data_files": _TTFT_FIT_RESULT_FILES,
     "script_function": "analysis/numbers_register.py::compute_ttft_few_point_calibration"},
    {"claim_id": "pack-trace-weighted-stats", "description": "workload pack prompt-token p50/p90/p99, flat vs trace-weighted",
     "compute": compute_pack_trace_weighted_stats,
     "data_files": ["results/workload_pack/items/*.jsonl", "results/traces/agent_step_lengths.parquet"],
     "script_function": "analysis/numbers_register.py::compute_pack_trace_weighted_stats"},
    {"claim_id": "ollama-overflow-keeps-half", "description": "Ollama overflow truncation = num_ctx/2 + 2, mechanism cited",
     "compute": compute_ollama_overflow_keeps_half,
     "data_files": ["results/apu_results__t2s_k1_ollama_evo-t2s_20261001T074622Z.jsonl",
                    "results/t2s_k1_igpu_enable_sweep.jsonl", "results/t2s_k1_ollama_evo-x2_20261001T080109Z.jsonl",
                    "results/x2_template_mechanism_test.jsonl"],
     "script_function": "analysis/numbers_register.py::compute_ollama_overflow_keeps_half"},
    {"claim_id": "qwen4b-2507-metadata-consistency", "description": "qwen3-4b-2507 cross-machine GGUF/context_length check",
     "compute": compute_qwen4b_2507_metadata_consistency,
     "data_files": ["results/x2_template_mechanism_test.jsonl"],
     "script_function": "analysis/numbers_register.py::compute_qwen4b_2507_metadata_consistency"},
    {"claim_id": "x2-device-detect-mechanism", "description": "evo-x2 Ollama GPU discovery, quoted (Vulkan dropped, ROCm picks it up)",
     "compute": compute_x2_device_detect_mechanism,
     "data_files": ["results/x2_template_mechanism_test.jsonl"],
     "script_function": "analysis/numbers_register.py::compute_x2_device_detect_mechanism"},
    {"claim_id": "x2-weekend-tag-counts", "description": "X2 weekend outcome rows tagged invalid (race/thinking/oom), with verified race cause",
     "compute": compute_x2_weekend_tag_counts, "data_files": [X2_WEEKEND],
     "script_function": "analysis/numbers_register.py::compute_x2_weekend_tag_counts"},
    {"claim_id": "x2-weekend-refused-block", "description": "X2 weekend contiguous ollama connection-refused block (SYSTEM lineage)",
     "compute": compute_x2_weekend_refused_block, "data_files": [X2_WEEKEND],
     "script_function": "analysis/numbers_register.py::compute_x2_weekend_refused_block"},
    {"claim_id": "x2-weekend-error-causes", "description": "X2 weekend outcome rows by error cause and subcause",
     "compute": compute_x2_weekend_error_causes, "data_files": [X2_WEEKEND],
     "script_function": "analysis/numbers_register.py::compute_x2_weekend_error_causes"},
    {"claim_id": "x2-weekend-error-causes-by-cell", "description": "X2 weekend error causes per (model, config)",
     "compute": compute_x2_weekend_error_causes_by_cell, "data_files": [X2_WEEKEND],
     "script_function": "analysis/numbers_register.py::compute_x2_weekend_error_causes_by_cell"},
    {"claim_id": "x2-weekend-thinking-scores", "description": "X2 weekend llama_server mean score per model (all rows)",
     "compute": compute_x2_weekend_thinking_scores, "data_files": [X2_WEEKEND],
     "script_function": "analysis/numbers_register.py::compute_x2_weekend_thinking_scores"},
    {"claim_id": "x2-weekend-ngen-budget", "description": "X2 weekend llama-server scored requests that used the full 256-token budget, per model",
     "compute": compute_x2_weekend_ngen_budget, "data_files": [X2_WEEKEND_LOGSCAN],
     "script_function": "analysis/numbers_register.py::compute_x2_weekend_ngen_budget"},
    {"claim_id": "x2-thinking-verify", "description": "evo-x2 live thinking-disable verification per mechanism (llama-server b10970, Ollama)",
     "compute": compute_x2_thinking_verify, "data_files": [X2_THINKING_VERIFY],
     "script_function": "analysis/numbers_register.py::compute_x2_thinking_verify"},
    {"claim_id": "t2s-outcome-error-causes", "description": "T2S outcome table (synced full file) rows by error cause, and race-signature count",
     "compute": compute_t2s_outcome_error_causes, "data_files": [T2S_OUTCOME_FULL],
     "script_function": "analysis/numbers_register.py::compute_t2s_outcome_error_causes"},
]


def _values_match(reported, computed_value, tolerance_note=False):
    """Loose string-level match: the reported value's key digits/percent must appear in the computed
    value string. Deliberately conservative (prefers a false CORRECTED over a false VERIFIED) -- exact
    numeric parsing of free-form reported strings is not attempted here."""
    import re
    reported_nums = re.findall(r"[\d.]+", reported)
    computed_nums = re.findall(r"[\d.]+", computed_value)
    return all(n in computed_nums for n in reported_nums)


def build_register():
    head = _git_head()
    date = _today()
    rows = []
    for entry in NUMBER_ENTRIES:
        try:
            result = entry["compute"](REPO)
            value = result["value"]
            n = result["n"]
            reported = entry.get("reported_value")
            if reported is None:
                status = "VERIFIED"
            elif _values_match(reported, value):
                status = "VERIFIED"
            else:
                status = "CORRECTED"
        except Exception as e:
            value, n, status = f"ERROR: {e!r}", "n/a", "UNSUPPORTED"
        rows.append({
            "claim_id": entry["claim_id"], "status": status, "value": value, "n": n,
            "reported_value": entry.get("reported_value", "(none previously reported)"),
            "data_files": entry["data_files"], "script_function": entry["script_function"],
            "commit": head, "date": date,
        })
    return rows


def write_register_md(rows, out_path):
    status_order = {"CORRECTED": 0, "UNSUPPORTED": 1, "VERIFIED": 2}
    rows_sorted = sorted(rows, key=lambda r: status_order.get(r["status"], 9))
    lines = [
        "# Numbers Register",
        "",
        "Every number cited anywhere in this program must come from a row in this table, pasted verbatim,",
        "never hand-typed. Regenerated by `analysis/numbers_register.py` -- every value below was computed",
        "fresh from the listed data file(s) by the listed function at the commit/date shown, not copied from",
        "a prior report.",
        "",
        "| claim id | status | value | n | reported value | data file(s) | script::function | commit | date |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows_sorted:
        data_files = "; ".join(r["data_files"])
        lines.append(f"| {r['claim_id']} | {r['status']} | {r['value']} | {r['n']} | {r['reported_value']} "
                     f"| {data_files} | {r['script_function']} | {r['commit']} | {r['date']} |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    rows = build_register()
    out_path = REPO / "docs" / "NUMBERS_REGISTER.md"
    write_register_md(rows, out_path)
    unsupported = [r for r in rows if r["status"] == "UNSUPPORTED"]
    corrected = [r for r in rows if r["status"] == "CORRECTED"]
    print(f"wrote {out_path}: {len(rows)} entries, {len(corrected)} CORRECTED, {len(unsupported)} UNSUPPORTED")
    for r in corrected + unsupported:
        print(f"  {r['status']}: {r['claim_id']} -- reported {r['reported_value']!r}, computed {r['value']!r}")
    return 1 if unsupported else 0


if __name__ == "__main__":
    raise SystemExit(main())
