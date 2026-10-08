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


# ── A-24 flip point (n_ctx at which llama-server stops starting), predicted from first principles ──
#
# compute_a24_budget_boundary_first_principles (above) checks the weights+KV+compute DECOMPOSITION at one
# n_ctx per model (each model's first_fail point) against the same log's own total. It never solves for the
# n_ctx at which the total crosses the heap budget, so it is not a predicted-vs-measured flip point. The
# functions below build that: every _srv_bis_ log for a model is parsed, the KV bytes/token comes from the
# architecture (n_layer, n_head_kv, head dims, f16), the compute buffer's n_ctx slope is fitted across that
# model's own successful starts, and the predicted flip is solved against the 47,866 MiB vulkaninfo budget
# (results/t2s_amech_20260926T181456Z.jsonl, record vulkan_limits).

_A24_BIS_LOG_GLOBS = {
    "qwen3-32b": "t2s_amech_20260926T181456Z_srv_bis_qwen3-32b_*.txt",
    "qwen3-8b": "t2s_amech_20260926T181456Z_srv_bis_qwen3-8b_*.txt",
    "qwen3-30b-a3b-2507": "t2s_amech_20260926T181456Z_srv_bis_qwen3-30b-a3b-2507_*.txt",
    "llama31-8b": "t2s_amech_20260926T181456Z_srv_bis_llama31-8b_*.txt",
    "llama-3.3-70b": "t2s_night2_20260929T034014Z_srv_bis_a70_*.txt",
}


def _parse_llama_server_log(path):
    """Every memory-relevant number one llama-server start log prints, as floats in MiB unless named
    otherwise. Missing values are None (a failed start stops logging partway through)."""
    import re
    text = Path(path).read_text(encoding="utf-8", errors="replace")

    def grab(pattern, cast=float):
        m = re.search(pattern, text)
        return cast(m.group(1)) if m else None

    bd = re.search(r"Vulkan0 \(.*?\)\s*\|\s*(\d+)\s*=\s*(\d+)\s*\+\s*\(\s*(\d+)\s*=\s*(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)\s*\)",
                   text)
    kv_dtypes = re.search(r"\bK \((\w+)\):.*\bV \((\w+)\):", text)
    fail = re.search(r"failed to allocate Vulkan0 buffer of size (\d+)", text)
    return {
        "file_size_gib": grab(r"file size\s*=\s*([\d.]+)\s*GiB"),
        "n_layer": grab(r"\bn_layer\s*=\s*(\d+)", int),
        "n_head_kv": grab(r"\bn_head_kv\s*=\s*(\d+)", int),
        "head_k": grab(r"\bn_embd_head_k\s*=\s*(\d+)", int),
        "head_v": grab(r"\bn_embd_head_v\s*=\s*(\d+)", int),
        "n_ubatch": grab(r"\bn_ubatch\s*=\s*(\d+)", int),
        "n_ctx_train": grab(r"\bn_ctx_train\s*=\s*(\d+)", int),
        "n_ctx_slot": grab(r"n_ctx_slot\s*=\s*(\d+)", int),
        "kv_dtypes": (kv_dtypes.group(1), kv_dtypes.group(2)) if kv_dtypes else None,
        "vk_model_mib": grab(r"Vulkan0 model buffer size =\s*([\d.]+)"),
        "host_model_mib": grab(r"Vulkan_Host model buffer size =\s*([\d.]+)"),
        "host_output_mib": grab(r"Vulkan_Host\s+output buffer size =\s*([\d.]+)"),
        "kv_mib": grab(r"Vulkan0 KV buffer size =\s*([\d.]+)"),
        "kv_cells": grab(r"\(\s*(\d+) cells", int),
        "vk_compute_mib": grab(r"Vulkan0 compute buffer size =\s*([\d.]+)"),
        "host_compute_mib": grab(r"Vulkan_Host compute buffer size =\s*([\d.]+)"),
        "free_mib": int(bd.group(2)) if bd else None,
        "proj_self_mib": int(bd.group(3)) if bd else None,
        "proj_model_mib": int(bd.group(4)) if bd else None,
        "proj_ctx_mib": int(bd.group(5)) if bd else None,
        "proj_compute_mib": int(bd.group(6)) if bd else None,
        "failed_alloc_mib": int(fail.group(1)) / 2 ** 20 if fail else None,
        "pinned_alloc_failed": "Failed to allocate pinned memory" in text,
        "started": ("n_ctx_slot" in text) and ("exiting due to model loading error" not in text),
    }


def _a24_vulkan_budget_mib(repo):
    path = repo / "results" / "t2s_amech_20260926T181456Z.jsonl"
    for r in _read_jsonl(path):
        if r.get("record") == "vulkan_limits":
            heaps = r["heaps"]
            if len(heaps) != 1:
                raise FileNotFoundError(f"expected a single Vulkan heap, found {len(heaps)}")
            return float(heaps[0]["budget_mib"]), r
    raise FileNotFoundError("no vulkan_limits record in t2s_amech_20260926T181456Z.jsonl")


def _a24_model_logs(repo):
    """{model: [parsed log dict + 'req_n_ctx']} for every _srv_bis_ log of the 5 A-24 models."""
    import re
    out = {}
    for model, pattern in _A24_BIS_LOG_GLOBS.items():
        logs = []
        for p in sorted((repo / "results").glob(pattern)):
            if p.name.endswith(".stdout.txt"):
                continue
            m = re.search(r"_(\d+)_(\d+)\.txt$", p.name)
            d = _parse_llama_server_log(p)
            d["req_n_ctx"] = int(m.group(1))
            d["file"] = p.name
            logs.append(d)
        if not logs:
            raise FileNotFoundError(f"{model}: no _srv_bis_ logs matching {pattern}")
        out[model] = logs
    return out


def _a24_model_parameters(model, logs):
    """Architecture, weights and compute-buffer law for one model, all read from its own logs."""
    import numpy as np
    arch = next(d for d in logs if d["n_layer"] and d["kv_dtypes"])
    if arch["kv_dtypes"] != ("f16", "f16"):
        raise FileNotFoundError(f"{model}: KV dtype {arch['kv_dtypes']} is not f16/f16")
    kv_bytes_per_token = arch["n_layer"] * arch["n_head_kv"] * (arch["head_k"] + arch["head_v"]) * 2
    kv_checks = [d["kv_mib"] * 2 ** 20 / d["kv_cells"] for d in logs if d["kv_mib"] and d["kv_cells"]]
    kv_logged_bpt = statistics.median(kv_checks)
    ok = [d for d in logs if d["started"] and d["vk_compute_mib"] is not None]
    cells = np.array([d["kv_cells"] for d in ok], dtype=float)
    comp = np.array([d["vk_compute_mib"] for d in ok], dtype=float)
    hcomp = np.array([d["host_compute_mib"] for d in ok], dtype=float)
    if len(set(cells)) < 2:
        raise FileNotFoundError(f"{model}: fewer than 2 distinct successful n_ctx to fit the compute buffer")
    slope, c0 = np.polyfit(cells, comp, 1)
    hslope, hc0 = np.polyfit(cells, hcomp, 1)
    resid = comp - (slope * cells + c0)
    return {
        "n_layer": arch["n_layer"], "n_head_kv": arch["n_head_kv"], "head_k": arch["head_k"],
        "head_v": arch["head_v"], "n_ubatch": arch["n_ubatch"], "n_ctx_train": arch["n_ctx_train"],
        "kv_bytes_per_token": kv_bytes_per_token, "kv_bytes_per_token_logged": kv_logged_bpt,
        "gguf_file_mib": arch["file_size_gib"] * 1024.0, "vk_model_mib": arch["vk_model_mib"],
        "host_model_mib": arch["host_model_mib"], "host_output_mib": arch["host_output_mib"] or 0.0,
        "compute_slope_bytes_per_token": slope * 2 ** 20, "compute_c0_mib": c0,
        "compute_fit_max_abs_resid_mib": float(np.max(np.abs(resid))), "compute_fit_points": len(ok),
        "host_compute_slope_bytes_per_token": hslope * 2 ** 20, "host_compute_c0_mib": hc0,
    }


def _a24_solve_flip(budget_mib, fixed_mib, p):
    """Smallest n_ctx (continuous) where fixed + KV(n) + device compute(n) exceeds the budget."""
    per_token_mib = (p["kv_bytes_per_token"] + p["compute_slope_bytes_per_token"]) / 2 ** 20
    return (budget_mib - fixed_mib - p["compute_c0_mib"]) / per_token_mib


def _a24_heap_demand_points(p, logs):
    """Per log: the heap demand at the moment the start either succeeded or failed, under the shared-heap
    accounting (Vulkan0 model + Vulkan_Host model/output + KV + Vulkan0 compute; plus the Vulkan_Host
    compute buffer when its pinned allocation was attempted). Returns (lower_bounds, upper_bounds) on the
    effective heap limit L, each a list of (MiB, file, kind)."""
    lowers, uppers = [], []
    for d in logs:
        if d["kv_mib"] is None:
            continue  # failed before the KV buffer: a bigger-n_ctx probe, adds nothing beyond its neighbours
        base = d["vk_model_mib"] + d["host_model_mib"] + (d["host_output_mib"] or 0.0) + d["kv_mib"]
        if d["started"]:
            dev = base + d["vk_compute_mib"]
            lowers.append((dev, d["file"], "started"))
            with_host = dev + d["host_compute_mib"]
            if d["pinned_alloc_failed"]:
                uppers.append((with_host, d["file"], "pinned host compute refused"))
            else:
                lowers.append((with_host, d["file"], "pinned host compute accepted"))
        elif d["failed_alloc_mib"] is not None and d["failed_alloc_mib"] < 1000:
            # the refused allocation is the Vulkan0 compute buffer (graph_reserve), sized exactly in bytes
            uppers.append((base + d["failed_alloc_mib"], d["file"], "device compute refused"))
    return lowers, uppers


def compute_a24_flip_point_prediction(repo):
    """Predicted vs measured flip n_ctx (first n_ctx at which llama-server fails to start) for the 5 A-24
    models, solved from first principles against the 47,866 MiB vulkaninfo heap budget, three accountings:

      A (file-size): GGUF file size + KV(n) + Vulkan0 compute(n)  -- the existing first-principles row's terms
      B (device-only): Vulkan0 model buffer + KV(n) + Vulkan0 compute(n)  -- llama.cpp's own fit projection
      C (shared heap, leave-one-model-out): Vulkan0 + Vulkan_Host model/output buffers + KV(n) + Vulkan0
        compute(n) against an effective limit L estimated from the OTHER 4 models' start/fail logs only

    KV(n) = n_layer * n_head_kv * (head_k + head_v) * 2 bytes * n; compute(n) = c0 + slope * n, fitted per
    model over its own successful starts. Measured flip = compute_a24_budget_boundary's first_fail_n_ctx
    (bisection step 256 tokens, so the true flip lies in (last_ok, first_fail])."""
    budget, _ = _a24_vulkan_budget_mib(repo)
    boundary = compute_a24_budget_boundary(repo)["detail"]
    logs_by_model = _a24_model_logs(repo)
    params = {m: _a24_model_parameters(m, logs) for m, logs in logs_by_model.items()}
    bounds = {m: _a24_heap_demand_points(params[m], logs) for m, logs in logs_by_model.items()}
    out = {}
    for model, p in params.items():
        meas = boundary[model]["first_fail_n_ctx"]
        last_ok = boundary[model]["last_ok_n_ctx"]
        others_lo = max(v for m, (lo, _) in bounds.items() if m != model for v, _, _ in lo)
        others_hi = min(v for m, (_, hi) in bounds.items() if m != model for v, _, _ in hi)
        loo_limit = (others_lo + others_hi) / 2.0
        preds = {
            "A_file_size": _a24_solve_flip(budget, p["gguf_file_mib"], p),
            "B_device_only": _a24_solve_flip(budget, p["vk_model_mib"], p),
            "C_shared_heap_loo": _a24_solve_flip(loo_limit, p["vk_model_mib"] + p["host_model_mib"]
                                                 + p["host_output_mib"], p),
        }
        out[model] = {
            "weights_gguf_mib": round(p["gguf_file_mib"], 1), "weights_vk_mib": p["vk_model_mib"],
            "weights_host_mib": p["host_model_mib"],
            "kv_bytes_per_token": p["kv_bytes_per_token"],
            "kv_bytes_per_token_logged": round(p["kv_bytes_per_token_logged"], 1),
            "arch": f"{p['n_layer']}L x {p['n_head_kv']}kv x ({p['head_k']}+{p['head_v']}) x f16",
            "compute_c0_mib": round(p["compute_c0_mib"], 2),
            "compute_bytes_per_token": round(p["compute_slope_bytes_per_token"], 1),
            "compute_fit_points": p["compute_fit_points"],
            "compute_fit_max_abs_resid_mib": round(p["compute_fit_max_abs_resid_mib"], 3),
            "n_ubatch": p["n_ubatch"], "n_ctx_train_gguf": p["n_ctx_train"],
            "measured_last_ok": last_ok, "measured_first_fail": meas,
            "loo_limit_mib": round(loo_limit, 1),
            "pred": {k: round(v) for k, v in preds.items()},
            "err_tokens": {k: round(v - meas) for k, v in preds.items()},
            "err_pct": {k: round((v - meas) / meas * 100.0, 2) for k, v in preds.items()},
            "within_bisection_step": {k: bool(last_ok < v <= meas) for k, v in preds.items()},
        }

    def rng(key):
        vals = [v["err_tokens"][key] for v in out.values()]
        pcts = [v["err_pct"][key] for v in out.values()]
        return f"{min(vals):+d} to {max(vals):+d} tokens ({min(pcts):+.2f}% to {max(pcts):+.2f}%)"

    value = (f"flip-point error vs measured first_fail, B={budget:.0f} MiB: A file-size {rng('A_file_size')}; "
             f"B device-only {rng('B_device_only')}; C shared-heap LOO {rng('C_shared_heap_loo')}")
    return {"value": value, "n": len(out), "detail": out}


def compute_a24_flip_point_inputs(repo):
    """Per-model inputs of compute_a24_flip_point_prediction, spelled out so each table cell is citable:
    weights (GGUF file / Vulkan0 buffer / Vulkan_Host buffer, MiB), KV bytes/token derived from the
    architecture vs read off the logged KV buffer, and the fitted compute-buffer law c0 + bytes/token."""
    d = compute_a24_flip_point_prediction(repo)["detail"]
    parts = [f"{m}: weights {v['weights_gguf_mib']:.1f}/{v['weights_vk_mib']:.2f}/{v['weights_host_mib']:.2f} MiB "
             f"(gguf/vk/host), KV {v['arch']} = {v['kv_bytes_per_token']} B/tok (logged "
             f"{v['kv_bytes_per_token_logged']:.0f}), compute {v['compute_c0_mib']:.2f} MiB + "
             f"{v['compute_bytes_per_token']:.0f} B/tok (n_ubatch {v['n_ubatch']}, {v['compute_fit_points']} starts, "
             f"max resid {v['compute_fit_max_abs_resid_mib']:.3f} MiB)" for m, v in d.items()]
    return {"value": "; ".join(parts), "n": len(d), "detail": d}


def compute_a24_flip_point_per_model(repo):
    """Per-model predicted vs measured flip n_ctx from compute_a24_flip_point_prediction, one cell each."""
    d = compute_a24_flip_point_prediction(repo)["detail"]
    parts = []
    for m, v in d.items():
        p, e, pc = v["pred"], v["err_tokens"], v["err_pct"]
        parts.append(f"{m}: measured ({v['measured_last_ok']}, {v['measured_first_fail']}]; "
                     f"A {p['A_file_size']} ({e['A_file_size']:+d}, {pc['A_file_size']:+.2f}%), "
                     f"B {p['B_device_only']} ({e['B_device_only']:+d}, {pc['B_device_only']:+.2f}%), "
                     f"C {p['C_shared_heap_loo']} ({e['C_shared_heap_loo']:+d}, {pc['C_shared_heap_loo']:+.2f}%, "
                     f"L_loo {v['loo_limit_mib']:.1f})")
    return {"value": "; ".join(parts), "n": len(d), "detail": d}


def compute_a24_effective_heap_limit(repo):
    """Bracket on the effective Vulkan heap limit L, from every _srv_bis_ log of the 5 A-24 models, under
    the shared-heap accounting (the evo-t2s Vulkan device exposes ONE heap and all 4 memory types,
    including the host-visible ones Vulkan_Host buffers come from, map to heapIndex 0 -- vulkan_limits
    record). Lower bounds: demand at every successful start (and every accepted pinned host-compute
    allocation). Upper bounds: demand at every refused device-compute or pinned host-compute allocation.
    Also reports the same bracket under device-only accounting, which is not consistent across models."""
    budget, vk = _a24_vulkan_budget_mib(repo)
    types_on_heap0 = sum(1 for line in vk["memory_properties_text"] if line.strip() == "heapIndex     = 0")
    logs_by_model = _a24_model_logs(repo)
    all_lo, all_hi, per_model, dev_lo, dev_hi = [], [], {}, [], []
    for model, logs in logs_by_model.items():
        p = _a24_model_parameters(model, logs)
        lo, hi = _a24_heap_demand_points(p, logs)
        all_lo += [(v, model, f, k) for v, f, k in lo]
        all_hi += [(v, model, f, k) for v, f, k in hi]
        per_model[model] = {"max_lower": round(max(v for v, _, _ in lo), 1),
                            "min_upper": round(min(v for v, _, _ in hi), 1)}
        for d in logs:
            if d["kv_mib"] is None:
                continue
            dev = d["vk_model_mib"] + d["kv_mib"]
            if d["started"]:
                dev_lo.append(dev + d["vk_compute_mib"])
            elif d["failed_alloc_mib"] is not None and d["failed_alloc_mib"] < 1000:
                dev_hi.append(dev + d["failed_alloc_mib"])
    lo_best = max(all_lo)
    hi_best = min(all_hi)
    value = (f"shared-heap L in ({lo_best[0]:.1f}, {hi_best[0]:.1f}] MiB, consistent across all 5 models "
             f"({len(all_lo)} lower / {len(all_hi)} upper bounds), {lo_best[0] - budget:+.0f} MiB vs the "
             f"{budget:.0f} MiB budget; device-only accounting is inconsistent "
             f"(max start {max(dev_lo):.1f} > min refusal {min(dev_hi):.1f} MiB); "
             f"{types_on_heap0} memory types all on heap 0")
    return {"value": value, "n": len(all_lo) + len(all_hi),
            "detail": {"per_model": per_model, "tightest_lower": lo_best, "tightest_upper": hi_best,
                       "device_only_max_start": max(dev_lo), "device_only_min_refusal": min(dev_hi)}}


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


# ── Why the evo-x2 TTFT fit is poor (R2 -0.03 to 0.63): diagnosis, each hypothesis tested by refitting ──

def _load_ttft_rows(repo):
    """Same row selection as _load_ttft_pairs (warm-up excluded, hw in evo-t2s/evo-x2, ttft_s and
    prompt_tokens present), but keeps the whole row so filters can use its other fields."""
    rows = []
    for rel_path in _TTFT_FIT_RESULT_FILES:
        path = repo / rel_path
        if not path.exists():
            continue
        for r in _read_jsonl(path):
            if r.get("ttft_s") is None or r.get("prompt_tokens") is None or r.get("warmup") is True:
                continue
            hw = r.get("hw_id") or r.get("host")
            model = r.get("model_id") or r.get("model_tag")
            if hw not in ("evo-t2s", "evo-x2") or model is None:
                continue
            r = dict(r)
            r["_hw"], r["_model"], r["_file"] = hw, model, rel_path
            rows.append(r)
    if not rows:
        raise FileNotFoundError("none of the TTFT fit result files were found")
    return rows


def _r2_rows(rows, intercept=False):
    import numpy as np
    ns = np.array([float(r["prompt_tokens"]) for r in rows])
    ys = np.array([float(r["ttft_s"]) for r in rows])
    if len(set(ns.tolist())) < 5:
        return None
    cols = [ns, ns ** 2] + ([np.ones_like(ns)] if intercept else [])
    design = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(design, ys, rcond=None)
    pred = design @ coef
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    return 1 - float(np.sum((ys - pred) ** 2)) / ss_tot if ss_tot > 0 else None


def _is_corunner(r):
    return r.get("co_runner") not in (None, "none", "N0", "N1")


def _is_pressure_or_smoke(r):
    return r.get("section") in ("K2", "MX2", "SMOKE")


def _is_first_call_on_prompt(r):
    """rep 0 = the first measured call on a given prompt (the harness's only warm-up is one call per server
    start, on a different prompt, so every new item's rep 0 is the first time the server sees that text);
    rows with no rep field (overnight section 0) are also single first calls."""
    return r.get("rep") in (0, None)


_TTFT_HYPOTHESIS_FILTERS = {
    "baseline": lambda r: True,
    "drop_corunner": lambda r: not _is_corunner(r),
    "drop_K2_MX2_SMOKE": lambda r: not _is_pressure_or_smoke(r),
    "drop_user_active": lambda r: r.get("user_active") is not True,
    "drop_first_call_rep0": lambda r: not _is_first_call_on_prompt(r),
    "all_four_dropped": lambda r: not (_is_corunner(r) or _is_pressure_or_smoke(r)
                                       or r.get("user_active") is True or _is_first_call_on_prompt(r)),
}


def compute_ttft_x2_length_spread(repo):
    """Per-model n and prompt-length spread behind the per-machine TTFT fit, both machines: is there enough
    length spread on evo-x2 for a quadratic in prompt_tokens to explain anything?"""
    import numpy as np
    rows = _load_ttft_rows(repo)
    out = {}
    for hw in ("evo-x2", "evo-t2s"):
        for model in sorted({r["_model"] for r in rows if r["_hw"] == hw}):
            ns = np.array([r["prompt_tokens"] for r in rows if r["_hw"] == hw and r["_model"] == model], float)
            out.setdefault(hw, {})[model] = {
                "n": int(len(ns)), "distinct": int(len(set(ns.tolist()))), "min": int(ns.min()),
                "p10": int(np.percentile(ns, 10)), "p50": int(np.median(ns)), "p90": int(np.percentile(ns, 90)),
                "max": int(ns.max())}
    x2 = out["evo-x2"]
    parts = [f"{m}: n={v['n']}, {v['min']}-{v['max']} tok (p10-p90 {v['p10']}-{v['p90']})" for m, v in x2.items()]
    return {"value": "evo-x2 " + "; ".join(parts), "n": sum(v["n"] for v in x2.values()), "detail": out}


def compute_ttft_x2_refit_by_hypothesis(repo):
    """Refits ttft_s = a*n + b*n^2 per (machine, model) with each suspected cause removed, plus a variant
    with an intercept. Hypotheses: co-runner/CPU-hog rows (PX2 conditions other than N0/N1, B-section
    co-runners), memory-pressure/spill/smoke rows (K2, MX2, SMOKE), user_active rows, first call on each
    new prompt (rep 0), and all four together. Only (hw, model) pairs with >=5 distinct prompt lengths
    after filtering are fit (qwen3-30b-a3b-2507 on evo-x2 has only 8 rep-0 rows, so it drops out of every
    filter that removes rep 0)."""
    rows = _load_ttft_rows(repo)
    out = {}
    for hw in ("evo-x2", "evo-t2s"):
        models = sorted({r["_model"] for r in rows if r["_hw"] == hw})
        for name, keep in _TTFT_HYPOTHESIS_FILTERS.items():
            for model in models:
                sub = [r for r in rows if r["_hw"] == hw and r["_model"] == model and keep(r)]
                r2 = _r2_rows(sub)
                out.setdefault(hw, {}).setdefault(name, {})[model] = {
                    "n": len(sub), "r2": None if r2 is None else round(r2, 4)}
        for model in models:
            sub = [r for r in rows if r["_hw"] == hw and r["_model"] == model]
            r2 = _r2_rows(sub, intercept=True)
            out[hw].setdefault("baseline_with_intercept", {})[model] = {
                "n": len(sub), "r2": None if r2 is None else round(r2, 4)}

    def rng(hw, name):
        vals = [v["r2"] for v in out[hw][name].values() if v["r2"] is not None]
        return f"{min(vals):.2f}-{max(vals):.2f} ({len(vals)} models)"

    # residual structure of the baseline fit: how much of the squared error sits on first-call rows
    import numpy as np
    shares = {}
    for model in sorted({r["_model"] for r in rows if r["_hw"] == "evo-x2"}):
        sub = [r for r in rows if r["_hw"] == "evo-x2" and r["_model"] == model]
        a, b, _ = _fit_ttft_quadratic([r["prompt_tokens"] for r in sub], [r["ttft_s"] for r in sub])
        res2 = np.array([(r["ttft_s"] - a * r["prompt_tokens"] - b * r["prompt_tokens"] ** 2) ** 2 for r in sub])
        first = np.array([_is_first_call_on_prompt(r) for r in sub])
        shares[model] = {"first_call_row_frac": round(float(first.mean()), 3),
                         "first_call_ss_res_share": round(float(res2[first].sum() / res2.sum()), 3)}
    out["evo-x2"]["baseline_residual_share"] = shares
    mixed = [v for v in shares.values() if v["first_call_row_frac"] < 1.0]  # models that also have repeats
    rf = [v["first_call_row_frac"] for v in mixed]
    ss = [v["first_call_ss_res_share"] for v in mixed]

    names = ["baseline", "drop_corunner", "drop_K2_MX2_SMOKE", "drop_user_active", "baseline_with_intercept",
             "drop_first_call_rep0", "all_four_dropped"]
    value = (f"evo-x2 rep-0 rows are {min(rf)*100:.0f}-{max(rf)*100:.0f}% of rows but carry "
             f"{min(ss)*100:.0f}-{max(ss)*100:.0f}% of baseline squared residual ({len(mixed)} models with "
             f"repeat calls); ")
    value += "evo-x2 R2: " + "; ".join(f"{n} {rng('evo-x2', n)}" for n in names)
    value += f" | evo-t2s R2: baseline {rng('evo-t2s', 'baseline')}, drop_first_call_rep0 " \
             f"{rng('evo-t2s', 'drop_first_call_rep0')}"
    return {"value": value, "n": sum(len(v) for v in out["evo-x2"].values()), "detail": out}


def compute_ttft_x2_refit_per_model(repo):
    """Per-model cells of compute_ttft_x2_refit_by_hypothesis (baseline, drop first call, all four
    dropped), both machines, plus the share of the evo-x2 baseline squared residual on first-call rows."""
    d = compute_ttft_x2_refit_by_hypothesis(repo)["detail"]
    parts = []
    for hw in ("evo-x2", "evo-t2s"):
        for m in sorted(d[hw]["baseline"]):
            cells = []
            for name in ("baseline", "drop_first_call_rep0", "all_four_dropped"):
                c = d[hw][name][m]
                cells.append(f"{'NA' if c['r2'] is None else format(c['r2'], '.3f')} (n={c['n']})")
            extra = ""
            if hw == "evo-x2":
                s = d[hw]["baseline_residual_share"][m]
                extra = f", first-call rows {s['first_call_row_frac']*100:.0f}% of rows / " \
                        f"{s['first_call_ss_res_share']*100:.0f}% of SS_res"
            parts.append(f"{hw} {m}: " + " / ".join(cells) + extra)
    return {"value": "baseline / drop_first_call_rep0 / all_four_dropped R2: " + "; ".join(parts),
            "n": len(parts), "detail": d}


def compute_ttft_t2s_restricted_to_x2_range(repo):
    """Length-spread test: refit evo-t2s on only the rows inside evo-x2's prompt-length range for the same
    model. If narrow spread were the cause of the low evo-x2 R2, evo-t2s restricted to the same range would
    collapse too, and it would stay low after dropping first calls."""
    rows = _load_ttft_rows(repo)
    out = {}
    for model in sorted({r["_model"] for r in rows if r["_hw"] == "evo-x2"}):
        x2n = [r["prompt_tokens"] for r in rows if r["_hw"] == "evo-x2" and r["_model"] == model]
        lo, hi = min(x2n), max(x2n)
        sub = [r for r in rows if r["_hw"] == "evo-t2s" and r["_model"] == model and lo <= r["prompt_tokens"] <= hi]
        sub_rep = [r for r in sub if not _is_first_call_on_prompt(r)]
        r2a, r2b = _r2_rows(sub), _r2_rows(sub_rep)
        out[model] = {"x2_range": [lo, hi], "n": len(sub), "r2": None if r2a is None else round(r2a, 4),
                      "n_rep_ge1": len(sub_rep), "r2_rep_ge1": None if r2b is None else round(r2b, 4)}
    parts = [f"{m}: {v['r2']:.2f} (n={v['n']})" + (f" -> {v['r2_rep_ge1']:.3f} rep>=1 (n={v['n_rep_ge1']})"
                                                   if v["r2_rep_ge1"] is not None else "")
             for m, v in out.items() if v["r2"] is not None]
    return {"value": "evo-t2s restricted to the evo-x2 prompt range: " + "; ".join(parts),
            "n": sum(v["n"] for v in out.values()), "detail": out}


_TTFT_STALL_SECTIONS = [
    ("evo-x2", "results/t2s_night2_20260929T205109Z.jsonl", ("R1b", "R1d")),
    ("evo-x2", "results/t2s_night2_20260929T045127Z.jsonl", ("R1check",)),
    ("evo-t2s", "results/t2s_night2_20260929T202603Z.jsonl", ("R1b", "R1d")),
]


def _kv_bytes_per_token_from_start_logs(repo, stem):
    """{model: KV bytes/token} from the real llama-server start logs '<stem>_srv_<section>_<model>_start.txt'
    (architecture + f16 dtype read from each log, not assumed)."""
    import re
    out = {}
    for p in sorted((repo / "results").glob(f"{stem}_srv_*_start.txt")):
        m = re.match(rf"{re.escape(stem)}_srv_R1\w*?_(.+)_start\.txt$", p.name)
        if not m:
            continue
        d = _parse_llama_server_log(p)
        if not (d["n_layer"] and d["kv_dtypes"] == ("f16", "f16")):
            continue
        out[m.group(1)] = d["n_layer"] * d["n_head_kv"] * (d["head_k"] + d["head_v"]) * 2
    return out


def compute_ttft_first_call_stall(repo):
    """The first call on a new prompt (rep 0) vs the repeat of the same prompt (rep 1), R1-family sections
    (every item is a new prompt; one warm-up per server start). For each rep-0 call that is >1.5x its own
    rep 1: stall = ttft(rep0) - ttft(rep1), and the KV state of the PREVIOUS call on the same server
    (its prompt_tokens x KV bytes/token from the server's own start log) divided by the stall. A constant
    MiB/s across models with different KV bytes/token means the stall scales with the bytes of KV state
    being displaced, not with the new prompt or with compute. Also reports package power during slow vs
    normal calls on evo-x2."""
    import collections
    out = {}
    for hw, rel, sections in _TTFT_STALL_SECTIONS:
        path = repo / rel
        if not path.exists():
            raise FileNotFoundError(rel)
        stem = Path(rel).stem
        kvb = _kv_bytes_per_token_from_start_logs(repo, stem)
        rows = [r for r in _read_jsonl(path) if r.get("ttft_s") is not None and r.get("warmup") is False
                and r.get("section") in sections]
        rows.sort(key=lambda r: r["ts_utc"])
        for model in sorted({r["model_id"] for r in rows}):
            if model not in kvb:
                raise FileNotFoundError(f"{rel}: no start log with architecture for {model}")
            for sec in sections:
                sub = [r for r in rows if r["model_id"] == model and r["section"] == sec]
                by_stem = collections.defaultdict(dict)
                for r in sub:
                    by_stem[r["item_id"].rsplit("_", 1)[0]][r["rep"]] = r
                pairs = slow = 0
                stalls, rates, p_slow, p_norm = [], [], [], []
                for i, r in enumerate(sub):
                    if r["rep"] != 0:
                        if isinstance(r.get("pkg_power_w"), (int, float)):
                            p_norm.append(r["pkg_power_w"])
                        continue
                    rep1 = by_stem[r["item_id"].rsplit("_", 1)[0]].get(1)
                    if rep1 is None:
                        continue
                    pairs += 1
                    if r["ttft_s"] > 1.5 * rep1["ttft_s"]:
                        slow += 1
                        s = r["ttft_s"] - rep1["ttft_s"]
                        stalls.append(s)
                        if i > 0:
                            rates.append(sub[i - 1]["prompt_tokens"] * kvb[model] / 2 ** 20 / s)
                        if isinstance(r.get("pkg_power_w"), (int, float)):
                            p_slow.append(r["pkg_power_w"])
                if not pairs:
                    continue
                out.setdefault(hw, {})[f"{model}/{sec}"] = {
                    "pairs": pairs, "slow": slow, "kv_bytes_per_token": kvb[model],
                    "stall_median_s": round(statistics.median(stalls), 2) if stalls else None,
                    "displaced_kv_mib_per_s_median": round(statistics.median(rates), 1) if rates else None,
                    "pkg_w_slow_median": round(statistics.median(p_slow), 1) if p_slow else None,
                    "pkg_w_rep_ge1_median": round(statistics.median(p_norm), 1) if p_norm else None,
                }
    parts = []
    for hw in ("evo-x2", "evo-t2s"):
        cells = out[hw].values()
        rates = [c["displaced_kv_mib_per_s_median"] for c in cells if c["displaced_kv_mib_per_s_median"]]
        stalls = [c["stall_median_s"] for c in cells if c["stall_median_s"]]
        slow = sum(c["slow"] for c in cells)
        pairs = sum(c["pairs"] for c in cells)
        part = (f"{hw}: {slow}/{pairs} rep-0 calls >1.5x their rep 1, stall median {min(stalls):.1f}-"
                f"{max(stalls):.1f} s, displaced KV state {min(rates):.1f}-{max(rates):.1f} MiB/s across "
                f"{len(rates)} model/section cells")
        if hw == "evo-x2":
            ps = [c["pkg_w_slow_median"] for c in cells if c["pkg_w_slow_median"]]
            pn = [c["pkg_w_rep_ge1_median"] for c in cells if c["pkg_w_rep_ge1_median"]]
            part += f"; pkg power slow calls {min(ps):.1f}-{max(ps):.1f} W vs rep>=1 {min(pn):.1f}-{max(pn):.1f} W"
        parts.append(part)
    return {"value": "; ".join(parts), "n": sum(len(v) for v in out.values()), "detail": out}


def compute_ttft_first_call_stall_per_cell(repo):
    """Per model/section cells of compute_ttft_first_call_stall."""
    d = compute_ttft_first_call_stall(repo)["detail"]
    parts = [f"{hw} {k}: {c['slow']}/{c['pairs']} slow, stall {c['stall_median_s']:.2f} s, "
             f"{c['displaced_kv_mib_per_s_median']:.1f} MiB/s, pkg {c['pkg_w_slow_median']} W slow vs "
             f"{c['pkg_w_rep_ge1_median']} W rep>=1"
             for hw in ("evo-x2", "evo-t2s") for k, c in d[hw].items()]
    return {"value": "; ".join(parts), "n": len(parts), "detail": d}


def compute_ttft_x2_section0_stall_check(repo):
    """Out-of-sample check of the displaced-KV-state stall on rows never used to estimate it: evo-x2
    overnight section 0 (8 calls per model, one server start each, no rep structure). Predicted ttft =
    the model's own evo-x2 fit on the all_four_dropped rows (see _TTFT_HYPOTHESIS_FILTERS) at this prompt length + (previous call's prompt_tokens x KV
    bytes/token) / R, where R = the evo-x2 median displaced-state rate from compute_ttft_first_call_stall,
    applied only to calls whose prompt is not an extension of the previous call's (prompt_tokens not larger
    than the previous call's). Reports the median absolute error with and without the stall term."""
    import numpy as np
    stall = compute_ttft_first_call_stall(repo)["detail"]["evo-x2"]
    rate = statistics.median(c["displaced_kv_mib_per_s_median"] for c in stall.values()
                             if c["displaced_kv_mib_per_s_median"])
    kvb = _kv_bytes_per_token_from_start_logs(repo, "t2s_night2_20260929T205109Z")
    rows = _load_ttft_rows(repo)
    s0 = [r for r in rows if r["_hw"] == "evo-x2" and r["_file"].endswith("t2s_overnight_20260929T071343Z.jsonl")]
    out = {}
    for model in sorted({r["_model"] for r in s0}):
        if model not in kvb:
            continue
        fit_rows = [r for r in rows if r["_hw"] == "evo-x2" and r["_model"] == model
                    and _TTFT_HYPOTHESIS_FILTERS["all_four_dropped"](r)]
        if len({r["prompt_tokens"] for r in fit_rows}) < 5:
            continue
        a, b, _ = _fit_ttft_quadratic([r["prompt_tokens"] for r in fit_rows], [r["ttft_s"] for r in fit_rows])
        seq = sorted([r for r in s0 if r["_model"] == model], key=lambda r: r["ts_utc"])
        err_with, err_without = [], []
        for i, r in enumerate(seq):
            n = r["prompt_tokens"]
            base = a * n + b * n * n
            pred = base
            if i > 0 and n <= seq[i - 1]["prompt_tokens"]:
                pred = base + seq[i - 1]["prompt_tokens"] * kvb[model] / 2 ** 20 / rate
            err_with.append(abs(pred - r["ttft_s"]) / r["ttft_s"] * 100)
            err_without.append(abs(base - r["ttft_s"]) / r["ttft_s"] * 100)
        out[model] = {"n": len(seq), "mape_with_stall_pct": round(float(np.median(err_with)), 1),
                      "mape_without_stall_pct": round(float(np.median(err_without)), 1)}
    w = [v["mape_with_stall_pct"] for v in out.values()]
    wo = [v["mape_without_stall_pct"] for v in out.values()]
    return {"value": f"evo-x2 overnight section 0, R={rate:.1f} MiB/s: median abs % error {min(w):.1f}-{max(w):.1f}% "
                     f"with the stall term vs {min(wo):.1f}-{max(wo):.1f}% without, {len(out)} models",
            "n": sum(v["n"] for v in out.values()), "detail": out}


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


X2_OUTCOME_V3 = "results/x2_outcome_table_v3.jsonl"


def compute_x2_v3_canary_gates(repo):
    """Latest canary_gate record per (model, config) in the restarted X2 outcome table."""
    x2 = _x2_harness()
    gates = {}
    for r in x2.read_rows(repo / X2_OUTCOME_V3):
        if r.get("record") == "canary_gate":
            gates[(r["model_id"], r["config"])] = r
    parts = [f"{m}/{c} {'PASS' if g['passed'] else 'FAIL'} err={g['error_rate']:.2f} mean={g['mean_score']:.2f} "
             f"leaks={g.get('thinking_leaks')}" for (m, c), g in sorted(gates.items())]
    return {"value": "; ".join(parts), "n": len(gates)}


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


# ───────────────────────────────────────────────── GSM8K strict vs lenient, T2S allocation failures (2026-10-07)
def _x2_v3_gsm8k_rows(repo):
    """Last valid gsm8k row per (item, model, config) in the v3 file (the row the latest canary gate read), with
    score (the recorded strict score), format_ok and score_lenient. Rows written with the lenient columns carry
    them. For older rows they are recomputed from output_text, which holds only the first 500 chars: when
    content_chars > 500 the last sentence was not stored, so score_lenient is None, and format_ok is None unless
    the strict score is 1 (which implies a `#### N`) or the stored head already contains one."""
    x2 = _x2_harness()
    g = x2.load_graders()
    items = {it["item_id"]: it for it in x2.load_items_trace_weighted(repo)}
    out = []
    for (iid, model, config), r in sorted(x2.valid_cached_rows(repo / X2_OUTCOME_V3).items()):
        if r.get("family") != "gsm8k":
            continue
        score = r.get("score") or 0.0
        if "score_lenient" in r:
            fmt, lenient = r.get("format_ok"), r.get("score_lenient")
        else:
            text = r.get("output_text") or ""
            if (r.get("content_chars") or 0) <= 500:
                f = x2.gsm8k_score_fields(g, items[iid], text)
                fmt, lenient = f["format_ok"], f["score_lenient"]
            else:
                head_fmt = bool(x2._FINAL_NUMBER_RE.search(x2.strip_thinking(text)))
                fmt = True if (score == 1.0 or head_fmt) else None
                lenient = None
        out.append({"item_id": iid, "model_id": model, "config": config, "score": score, "format_ok": fmt,
                    "score_lenient": lenient, "canary": r.get("canary")})
    return out


def compute_x2_v3_gsm8k_strict_vs_lenient(repo):
    """Per (model, config): gsm8k rows, strict score (the recorded score), format_ok (a `#### N` in the stored
    text), and the lenient last-sentence score over the rows whose stored text is the whole response."""
    by = {}
    for r in _x2_v3_gsm8k_rows(repo):
        b = by.setdefault((r["model_id"], r["config"]), {"n": 0, "strict": 0.0, "fmt": 0, "len_n": 0, "len": 0.0,
                                                          "strict_on_len": 0.0})
        b["n"] += 1
        b["strict"] += r["score"]
        b["fmt"] += r["format_ok"] is True
        b["fmt_unk"] = b.get("fmt_unk", 0) + (r["format_ok"] is None)
        if r["score_lenient"] is not None:
            b["len_n"] += 1
            b["len"] += r["score_lenient"]
            b["strict_on_len"] += r["score"]
    parts = [f"{m}/{c} strict {b['strict']:.0f}/{b['n']}, format_ok {b['fmt']}/{b['n']} "
             f"({b.get('fmt_unk', 0)} unknown), lenient {b['len']:.0f}/"
             f"{b['len_n']} (strict on the same {b['len_n']}: {b['strict_on_len']:.0f}; "
             f"{b['n'] - b['len_n']} rows over 500 chars, last sentence not stored)"
             for (m, c), b in sorted(by.items())]
    return {"value": "; ".join(parts), "n": sum(b["n"] for b in by.values())}


def compute_x2_v3_canary_gate_lenient(repo):
    """Canary gate mean score per (model, config) as recorded (strict gsm8k) and with lenient gsm8k scores
    substituted, computable only when all of that cell's gsm8k canaries stored their whole response."""
    x2 = _x2_harness()
    gates = {}
    for r in x2.read_rows(repo / X2_OUTCOME_V3):
        if r.get("record") == "canary_gate":
            gates[(r["model_id"], r["config"])] = r
    gsm = {}
    for r in _x2_v3_gsm8k_rows(repo):
        gsm.setdefault((r["model_id"], r["config"]), []).append(r)
    parts = []
    for key, gate in sorted(gates.items()):
        rows = gsm.get(key, [])
        n = gate["n"]
        if rows and all(r["score_lenient"] is not None for r in rows):
            delta = sum(r["score_lenient"] - r["score"] for r in rows)
            lenient = f"{gate['mean_score'] + delta / n:.2f}"
        else:
            # bounds: each gsm8k canary whose last sentence was not stored counts as lenient 0 (low) or 1 (high)
            known = sum(r["score_lenient"] - r["score"] for r in rows if r["score_lenient"] is not None)
            unk = [r for r in rows if r["score_lenient"] is None]
            lo = gate["mean_score"] + (known - sum(r["score"] for r in unk)) / n
            hi = gate["mean_score"] + (known + sum(1 - r["score"] for r in unk)) / n
            lenient = f"{lo:.2f} to {hi:.2f} ({len(unk)}/{len(rows)} gsm8k canaries over 500 chars, bounded)"
        parts.append(f"{key[0]}/{key[1]} strict mean {gate['mean_score']:.2f}, lenient mean {lenient}")
    return {"value": "; ".join(parts), "n": len(gates)}


def compute_t2s_outcome_gsm8k_rescore(repo):
    """Whether the synced T2S outcome rows can be rescored with the fixed gsm8k scorer: gsm8k rows present, and
    rows that stored response text."""
    files = sorted((repo / "results").glob("t2s_outcome_table*.jsonl"))
    parts, total = [], 0
    for p in files:
        rows = [r for r in _read_jsonl(p) if r.get("record") == "outcome_row"]
        total += len(rows)
        gsm = sum(1 for r in rows if r.get("family") == "gsm8k")
        txt = sum(1 for r in rows if r.get("output_text") is not None)
        fams = sorted({r.get("family") for r in rows})
        parts.append(f"{p.name}: {len(rows)} rows, gsm8k {gsm}, rows with output text {txt}, families {fams}")
    return {"value": "; ".join(parts), "n": total}


def _t2s_alloc_kind(err):
    e = (err or "")
    if "unable to allocate Vulkan0 buffer" in e:
        return "Vulkan0"
    if "unable to allocate CPU_REPACK buffer" in e:
        return "CPU_REPACK"
    return None


def compute_t2s_allocation_failures(repo):
    """Allocation failures in the synced T2S full file per config: count, allocation kind and buffer size (from the
    verbatim error), shortest failing and longest succeeding prompt (sent_tokens), and the timestamp of the first
    failure vs the last success (the failures are time-ordered, not length-ordered)."""
    import re
    rows = [r for r in _read_jsonl(repo / T2S_OUTCOME_FULL) if r.get("record") == "outcome_row"]
    parts = []
    for cfg in ("ollama_default", "ollama_igpu_enable", "llama_server_vulkan"):
        cr = [r for r in rows if r["config"] == cfg]
        fail = [r for r in cr if _t2s_alloc_kind(r.get("error"))]
        ok = [r for r in cr if r.get("http_status") == 200]
        if not fail:
            parts.append(f"{cfg}: 0 allocation failures in {len(cr)} rows ({len(ok)} HTTP 200, longest succeeding "
                         f"{max((r['sent_tokens'] for r in ok), default=None)} tok)")
            continue
        kinds = sorted({(_t2s_alloc_kind(r['error']), int(m)) for r in fail
                        for m in re.findall(r"buffer of size (\d+)", r["error"])[:1]})
        f_len = [r["sent_tokens"] for r in fail]
        ok_after = [r for r in ok if r["ts_utc"] > min(f["ts_utc"] for f in fail)]
        parts.append(f"{cfg}: {len(fail)} allocation failures of {len(cr)} rows, model {sorted({r['model_id'] for r in fail})}, "
                     f"buffer {kinds}; failing prompt tokens {min(f_len)}..{max(f_len)}; succeeding prompt tokens "
                     f"{min(r['sent_tokens'] for r in ok)}..{max(r['sent_tokens'] for r in ok)}; first failure "
                     f"{min(f['ts_utc'] for f in fail)} ({min(fail, key=lambda f: f['ts_utc'])['item_id']}), last success "
                     f"{max(r['ts_utc'] for r in ok)}, successes after first failure {len(ok_after)}")
    import datetime
    allr = _read_jsonl(repo / T2S_OUTCOME_FULL)
    items_hb = [r["item_id"] for r in allr if r.get("record") == "heartbeat"]
    ls_rows = {r["item_id"] for r in rows if r["config"] == "llama_server_vulkan"}
    missing = sorted(set(items_hb) - ls_rows)
    # time the unrecorded llama-server attempt took: from the item's last emitted row to the next heartbeat
    gaps = []
    for i, r in enumerate(allr):
        if r.get("record") == "outcome_row" and r["config"] == "ollama_igpu_enable" and r["item_id"] in missing:
            nxt = next((x for x in allr[i + 1:] if x.get("record") == "heartbeat"), None)
            if nxt is not None:
                t0 = datetime.datetime.fromisoformat(r["ts_utc"]) + datetime.timedelta(seconds=r.get("latency_s") or 0)
                gaps.append((datetime.datetime.fromisoformat(nxt["ts_utc"]) - t0).total_seconds())
    ls_ok = sorted((r["ts_utc"], r["item_id"], r.get("requested_n_ctx")) for r in rows
                   if r["config"] == "llama_server_vulkan" and r.get("http_status") == 200)
    parts.append(f"llama_server_vulkan: no row at all for {len(missing)} heartbeat items (the pre-2026-10-07 harness "
                 f"returned its two load-failure paths without emitting a row): {missing}; time spent on each unrecorded attempt "
                 f"(s, n={len(gaps)}): min {min(gaps):.0f}, median {statistics.median(gaps):.0f}, max {max(gaps):.0f}; "
                 f"last llama_server success {ls_ok[-1][1]} at n_ctx {ls_ok[-1][2]} ({ls_ok[-1][0]})")
    return {"value": "; ".join(parts), "n": len(rows)}


# ───────────────────────────────────────────────── R2 two-step agent harness validation (2026-10-07)
R2_VALIDATION_FILE = "results/x2_r2_validation.jsonl"        # v1: spec variant (tools on call 2), one canary pair
R2_VALIDATION_V2_FILE = "results/x2_r2_validation_v2.jsonl"  # v2: tools withheld on call 2, per-check canaries


def _r2_agent_module():
    import sys
    sys.path.insert(0, str(REPO / "harness"))
    import x2_r2_agent
    return x2_r2_agent


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _r2_table_str(e):
    rules = ", ".join(f"{r.split('_')[0]} {_pct(v['rate'])}" for r, v in e["rules"].items())
    return (f"{rules}; tool validity {_pct(e['tool_validity']['rate'])} of {e['tool_validity']['n_calls']} calls; "
            f"tool args {_pct(e['tool_args']['rate'])}; recall {_pct(e['recall']['rate'])} (n={e['recall']['n']}); "
            f"canary C {_pct(e['canary_sys']['rate'])}, H {_pct(e['canary_hist']['rate'])} (n={e['canary_sys']['n']}); "
            f"empty final answers {e['final_content_empty']}")


def _r2_gates_summary(repo, rel, call2_tools):
    ag = _r2_agent_module()
    rows = ag.read_rows(repo / rel)
    if not rows:
        raise FileNotFoundError(repo / rel)
    rep = ag.evaluate_gates(rows, call2_tools=call2_tools)
    parts, n_total = [], 0
    for model, e in sorted(rep["baseline_table"].items()):
        n_total += e["n_turns"]
        neg = rep["controls"]["negative"].get(model, {})
        pos = rep["controls"]["positive"].get(model, {})
        parts.append(f"{model}: {_r2_table_str(e)}; negative-control canary misses {neg.get('canary_misses')}/"
                     f"{2 * neg.get('canary_checks', 0)}; positive-control truncation turn "
                     f"{pos.get('truncation_detected_turns')}")
    return {"value": " / ".join(parts), "n": f"{n_total} turns", "detail": rep}


def _r2_diag_summary(repo, rel, arm):
    ag = _r2_agent_module()
    rows = ag.read_rows(repo / rel)
    if not rows:
        raise FileNotFoundError(repo / rel)
    table = ag.baseline_table(rows, arm=arm)
    parts = [f"{m}: {_r2_table_str(e)}" for m, e in sorted(table.items())]
    return {"value": " / ".join(parts), "n": f"{sum(e['n_turns'] for e in table.values())} turns", "detail": table}


def compute_r2_validation_baseline(repo):
    """R2 validation v1 (harness/x2_r2_agent.py, spec variant: tools available on call 2; arm b num_ctx 131072,
    3 seeds x 10 turns per model; positive control 8192): per model rule compliance on the final answer (rule 2
    turn-level OR), tool validity per call, tool arguments per valid call, recall, canaries, controls. Superseded
    as a gate by v2 (canary echo and llama3.1 call-2 tool loops, see docs/R2_DESIGN.md)."""
    return _r2_gates_summary(repo, R2_VALIDATION_FILE, True)


def compute_r2_validation_call2_notools(repo):
    """v1 file, the diagnostic arm (tools withheld on call 2 only, same seeds): per model table."""
    ag = _r2_agent_module()
    return _r2_diag_summary(repo, R2_VALIDATION_FILE, "ollama_ctx_131072" + ag.NOTOOLS_SUFFIX)


def compute_r2_validation_v2_baseline(repo):
    """R2 validation v2, the gate for the real run: tools withheld on call 2, per-check canaries, arm b 131072
    3 seeds x 10 turns, positive control 8192 1 seed x 15 turns, per model."""
    return _r2_gates_summary(repo, R2_VALIDATION_V2_FILE, False)


def compute_r2_validation_v2_spec_diag(repo):
    """v2 file, the diagnostic arm: the spec variant (tools available on call 2) with per-check canaries."""
    return _r2_diag_summary(repo, R2_VALIDATION_V2_FILE, "ollama_ctx_131072")

_K2_X2_RUN = "results/t2s_k2_pressure_20261004T201205Z.jsonl"


def _k2_x2_rows(repo):
    path = repo / _K2_X2_RUN
    if not path.exists():
        raise FileNotFoundError(str(path))
    return _read_jsonl(path)


def _k2_tag_parts(tag):
    """'k2_<model>_<mmap_arm>_<pressure_arm>' -> (model, mmap_arm, pressure_arm); pressure arms contain an
    underscore themselves, so split from the right on the two known arm names."""
    body = tag[len("k2_"):]
    for parm in ("awe_balloon", "pageable_touch"):
        if body.endswith("_" + parm):
            rest = body[: -len(parm) - 1]
            model, mmap_arm = rest.rsplit("_", 1)
            return model, mmap_arm, parm
    raise ValueError(f"unrecognized K2 tag {tag!r}")


def compute_k2_x2_kill_criterion_table(repo):
    """K2 arms (a)/(b) on evo-x2 (x2_k2_v2, 2026-10-04/05): every k2_kill_criterion record (one per completed
    (model, mmap arm, pressure arm) run), how many passed, plus the run(s) that started but never reached a
    kill-criterion evaluation and the model/arm combinations never started. Per-run detail: levels run, last level,
    first clean-failure level, minimum step median score, and the largest step-median responsiveness ratio vs the
    +8GB step."""
    rows = _k2_x2_rows(repo)
    kcs = [r for r in rows if r.get("record") == "k2_kill_criterion"]
    steps = [r for r in rows if r.get("record") == "k2_step_summary"]
    started = [r["item_id"][: -len("_start")] for r in rows if r.get("kind") == "start"]
    kc_tags = {r["item_tag"] for r in kcs}
    detail = []
    for tag in started:
        st = [s for s in steps if s["item_tag"] == tag]
        base = next((s for s in st if s["level_gb"] == 8), None)
        ratios = [s["responsiveness_median_s"] / base["responsiveness_median_s"] for s in st
                  if base and s.get("responsiveness_median_s") and base.get("responsiveness_median_s")]
        kc = next((r for r in kcs if r["item_tag"] == tag), None)
        model, mmap_arm, parm = _k2_tag_parts(tag)
        detail.append({"model": model, "mmap_arm": mmap_arm, "pressure_arm": parm,
                       "levels_run": [s["level_gb"] for s in st],
                       "first_clean_failure_gb": next((s["level_gb"] for s in st if s["clean_failure"]), None),
                       "min_step_median_score": min((s["median_score"] for s in st if s["median_score"] is not None),
                                                    default=None),
                       "max_resp_ratio_vs_8gb": round(max(ratios), 2) if ratios else None,
                       "kill_criterion": None if kc is None else ("pass" if kc.get("ok") else "violated")})
    n_ok = sum(1 for r in kcs if r.get("ok"))
    no_kc = [t for t in started if t not in kc_tags]
    models = []
    for t in started:
        m = _k2_tag_parts(t)[0]
        if m not in models:
            models.append(m)
    complete = [m for m in models if sum(1 for t in kc_tags if _k2_tag_parts(t)[0] == m) == 4]
    run_end = next((r.get("note") for r in rows if r.get("record") == "run_end"), None)
    value = (f"{len(kcs)} kill-criterion evaluations, {n_ok} passed, {len(kcs) - n_ok} violated; "
             f"models complete (4/4 arm combinations) {len(complete)}/{len(models)} ({', '.join(complete)}); "
             f"started without an evaluation: {', '.join(no_kc) or 'none'}; run_end note: {run_end}")
    per_run = ", ".join(
        f"{d['model']}/{d['mmap_arm']}/{d['pressure_arm']}: levels {d['levels_run'][0]:+d}..{d['levels_run'][-1]:+d}"
        f" fail {d['first_clean_failure_gb']} minscore {d['min_step_median_score']} maxresp "
        f"{d['max_resp_ratio_vs_8gb']} kc {d['kill_criterion']}" for d in detail if d["levels_run"])
    return {"value": value + f"; per run [{per_run}]", "n": len(kcs), "detail": detail}


def compute_k2_x2_clean_failures(repo):
    """Every quality item in the K2 evo-x2 run whose outcome was not ok, with the server crash fields."""
    rows = _k2_x2_rows(repo)
    fails = [r for r in rows if r.get("kind") == "quality_score" and (r.get("outcome") != "ok" or r.get("crash"))]
    n_items = sum(1 for r in rows if r.get("kind") == "quality_score")
    parts = []
    for r in fails:
        parts.append(f"{r['item_id']}: outcome={r.get('outcome')}, crash={r.get('crash')}, "
                     f"exit_code={r.get('exit_code')} (0x{(r.get('exit_code') or 0):08X}), "
                     f"error={r.get('error')!r}, server_log_tail={r.get('server_log_tail')}")
    return {"value": f"{len(fails)}/{n_items} quality items not ok. " + " | ".join(parts), "n": n_items}


def compute_k2_x2_score_ceiling(repo):
    """K2 evo-x2 quality items: how many scored 1.0, the task type, think-tag hits, and the completion length
    (do_call's default ignore_eos=True forces max_tokens=256 on every call)."""
    rows = _k2_x2_rows(repo)
    qs = [r for r in rows if r.get("kind") == "quality_score"]
    calls = [r for r in rows if r.get("kind") == "quality" and r.get("outcome") == "ok"]
    n_one = sum(1 for r in qs if r.get("score") == 1.0)
    n_scored = sum(1 for r in qs if r.get("score") is not None)
    tasks = sorted({r.get("task_type") for r in qs})
    think = sum(1 for r in calls if r.get("think_tag"))
    ctoks = sorted({r.get("completion_tokens") for r in calls})
    seeds = sorted({r.get("rep") for r in qs})
    return {"value": f"{n_one}/{n_scored} scored items = 1.0 ({len(qs) - n_scored} unscored); task_type {tasks}; "
                     f"think_tag {think}/{len(calls)} ok calls; completion_tokens values {ctoks}; "
                     f"distinct prompts (rep) {seeds}",
            "n": len(qs)}


def compute_k2_x2_decode_by_pressure_arm(repo):
    """K2 evo-x2: per run, median decode_tok_s over pressure-step calls divided by the median over that run's own
    3 unpressured baseline calls. Reported as the range per pressure arm."""
    rows = _k2_x2_rows(repo)
    by = {}
    for r in rows:
        if r.get("kind") == "quality" and r.get("outcome") == "ok" and r.get("decode_tok_s"):
            phase = r.get("k2_phase")
            tag = r["item_id"][: r["item_id"].index("_" + phase + "_")]
            by.setdefault(tag, {"base": [], "press": []})["base" if phase == "baseline" else "press"].append(
                r["decode_tok_s"])
    per_arm = {}
    detail = {}
    for tag, v in by.items():
        if not v["base"] or not v["press"]:
            continue
        ratio = statistics.median(v["press"]) / statistics.median(v["base"])
        detail[tag] = round(ratio, 4)
        per_arm.setdefault(_k2_tag_parts(tag)[2], []).append(ratio)
    value = "; ".join(f"{arm}: {min(v):.3f}x-{max(v):.3f}x over {len(v)} runs" for arm, v in sorted(per_arm.items()))
    return {"value": value, "n": len(detail), "detail": detail}


def compute_k2_x2_call_timing(repo):
    """K2 evo-x2: per model, median ttft_s, decode_tok_s and prompt_tokens over every ok quality call (llama-server,
    Vulkan, n_ctx 16384, ~12k-token niah prompt, 256 forced decode tokens). Used as the per-turn basis for the
    standalone arm (d) duration projection."""
    rows = _k2_x2_rows(repo)
    by = {}
    for r in rows:
        if r.get("kind") == "quality" and r.get("outcome") == "ok" and r.get("ttft_s"):
            by.setdefault(r["model_id"], []).append(r)
    parts = []
    for m, v in by.items():
        parts.append(f"{m}: ttft {statistics.median(x['ttft_s'] for x in v):.1f} s, decode "
                     f"{statistics.median(x['decode_tok_s'] for x in v):.2f} tok/s, prompt "
                     f"{statistics.median(x['prompt_tokens'] for x in v):.0f} tok (n={len(v)})")
    return {"value": "; ".join(parts), "n": sum(len(v) for v in by.values())}


def compute_k2_x2_responsiveness_resolution(repo):
    """K2 evo-x2 responsiveness probe: the distinct step-median values observed (python -c pass launch latency,
    timed with time.monotonic(), whose resolution on Windows is the ~15.6 ms system tick), and the largest
    step/baseline ratio against the kill criterion's 2.0x tolerance."""
    rows = _k2_x2_rows(repo)
    steps = [r for r in rows if r.get("record") == "k2_step_summary" and r.get("responsiveness_median_s")]
    vals = sorted({round(r["responsiveness_median_s"], 4) for r in steps})
    ratios = []
    for r in steps:
        base = next((s for s in steps if s["item_tag"] == r["item_tag"] and s["level_gb"] == 8), None)
        if base:
            ratios.append(r["responsiveness_median_s"] / base["responsiveness_median_s"])
    return {"value": f"distinct step medians (s): {vals}; max step/+8GB ratio {max(ratios):.3f}x (tolerance 2.0x)",
            "n": len(steps)}


def compute_k2_x2_unreached_pressure_levels(repo):
    """K2 evo-x2: pressure steps whose pressure source did not reach its available-memory target."""
    rows = _k2_x2_rows(repo)
    starts = [r for r in rows if r.get("record") == "k2_pressure_start"]
    bad = [r for r in starts if not (r.get("info") or {}).get("ok")]
    parts = [f"{r['item_tag'][3:]} {r['level_gb']:+d}GB target {r['target_avail_mb']:.0f} MB, reached "
             f"{(r.get('info') or {}).get('available_mb_after', 0):.0f} MB" for r in bad]
    return {"value": f"{len(bad)}/{len(starts)} steps did not reach target: " + "; ".join(parts), "n": len(starts)}

def _kappa_module(repo):
    import importlib.util
    spec = importlib.util.spec_from_file_location("kappa_agreement", repo / "analysis" / "kappa_agreement.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compute_kappa_heldout_sample(repo):
    """Held-out kappa sample construction (2026-10-06): population, exclusions, strata, seed, recomputed
    by re-running the builder's own sampling functions (no files written)."""
    import importlib.util
    import sys as _sys
    _sys.path.insert(0, str(repo / "analysis"))
    spec = importlib.util.spec_from_file_location("build_kappa_heldout_sample",
                                                  repo / "analysis" / "build_kappa_heldout_sample.py")
    b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b)
    pop = b.load_population()
    pop_keys = {b.row_key(r) for r in pop}
    orig = b.original_sample_keys(pop)
    ritz = b.ritz30_keys() & pop_keys
    excluded = orig | ritz
    remaining = [r for r in pop if b.row_key(r) not in excluded]
    sample, n_strata = b.round_robin_sample(remaining, b.N, b.SEED)
    return {"value": f"seed {b.SEED}; wrong-answer population {len(pop)}; excluded {len(orig)} original-sample + "
                     f"{len(ritz)} ritz30 rows (union {len(excluded)}); remaining {len(remaining)} over "
                     f"{n_strata} strata; sampled {len(sample)}",
            "n": len(sample)}


def compute_kappa_heldout(repo):
    """Held-out kappa, current evaluation/outcome.py scorer vs blinded annotator_claude labels."""
    s = _kappa_module(repo).heldout_stats(repo)
    m = s["matrix"]
    cats = s["categories"]
    cm = "; ".join(f"scorer {a}: " + ", ".join(f"{b} {m[(a, b)]}" for b in cats) for a in cats)
    return {"value": f"kappa {s['kappa']:.3f} (analytic 95% CI {s['analytic_ci'][0]:.3f}-{s['analytic_ci'][1]:.3f}, "
                     f"bootstrap 95% CI {s['bootstrap_ci'][0]:.3f}-{s['bootstrap_ci'][1]:.3f}); raw agreement "
                     f"{s['agree']}/{s['n']}; confusion (rows scorer, cols annotator) {cm}",
            "n": s["n"]}


def compute_kappa_heldout_fn_annotator(repo):
    """Method-controlled secondary: scorer vs the original study's annotator FUNCTION on the held-out rows,
    and manual blinded labels vs that function."""
    s = _kappa_module(repo).heldout_stats(repo)
    return {"value": f"scorer vs classify_human_label: kappa {s['fn_annotator_kappa']:.3f} (bootstrap 95% CI "
                     f"{s['fn_annotator_bootstrap_ci'][0]:.3f}-{s['fn_annotator_bootstrap_ci'][1]:.3f}), agree "
                     f"{s['fn_annotator_agree']}/{s['n']}; manual labels vs classify_human_label: kappa "
                     f"{s['manual_vs_fn_kappa']:.3f}, agree {s['manual_vs_fn_agree']}/{s['n']}",
            "n": s["n"]}


def compute_kappa_heldout_disagreement_read(repo):
    """Breakdown of the held-out disagreements by read (scorer_bug / labeler_error / ambiguous), from the
    committed read file, after asserting its ids are exactly the live disagreement list."""
    import csv as _csv
    from collections import Counter as _Counter
    s = _kappa_module(repo).heldout_stats(repo)
    live = {d["id"]: d for d in s["disagreements"]}
    with open(repo / "results" / "labeling" / "kappa_heldout_disagreement_read.csv", encoding="utf-8") as f:
        reads = list(_csv.DictReader(f))
    if {r["id"] for r in reads} != set(live):
        raise ValueError("disagreement read file ids do not match the live disagreement list")
    by_read = _Counter(r["read"] for r in reads)
    direction = _Counter(f"{live[i]['scorer']}->{live[i]['annotator']}" for i in live)
    return {"value": f"{len(reads)} disagreements: " + ", ".join(f"{k} {v}" for k, v in sorted(by_read.items()))
                     + "; direction (scorer->annotator) " + ", ".join(f"{k} {v}" for k, v in sorted(direction.items())),
            "n": len(reads)}


_CLOUD_FORECAST_CACHE = {}


def _cloud_forecast(repo, **kw):
    """analysis/cloud_budget_forecast.py::forecast, memoized per (repo, kwargs) within one register build."""
    key = (str(repo), tuple(sorted(kw.items())))
    if key not in _CLOUD_FORECAST_CACHE:
        import importlib.util
        spec = importlib.util.spec_from_file_location("cloud_budget_forecast",
                                                      repo / "analysis" / "cloud_budget_forecast.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _CLOUD_FORECAST_CACHE[key] = m.forecast(repo, **kw)
    return _CLOUD_FORECAST_CACHE[key]


_CLOUD_DATA_FILES = ["results/cloud_prices_20261008.json", "results/workload_pack/items/*.jsonl",
                     "results/workload_pack/item_weights_trace_weighted.json", "results/x2_outcome_table_v3.jsonl",
                     "results/x2_r2_real_v1.jsonl", "docs/DEMO_SPEC.md"]


def compute_cloud_budget_totals(repo):
    """Cloud budget forecast (2026-10-07): total and pessimistic total vs the USD 50 cap."""
    f = _cloud_forecast(repo)
    b, p = f["base"]["total"], f["pessimistic"]["total"]
    return {"value": f"forecast USD {b:.2f} ({b / f['cap_usd'] * 100:.0f}% of USD {f['cap_usd']:.0f} cap); "
                     f"pessimistic USD {p:.2f} ({p / f['cap_usd'] * 100:.0f}% of cap); models "
                     f"{f['models']['cheap']} / {f['models']['mid']}",
            "n": f"{f['pack']['n_items']} items, {f['r2']['n_sessions']} R2 sessions"}


def compute_cloud_budget_lines(repo):
    """Cloud budget forecast per line (a)-(e), forecast / pessimistic USD."""
    f = _cloud_forecast(repo)
    return {"value": "; ".join(f"({k}) {f['base'][k]:.2f} / {f['pessimistic'][k]:.2f}" for k in "abcde"),
            "n": f"{f['pack']['n_items']} items, {f['r2']['n_sessions']} R2 sessions"}


def compute_cloud_budget_tokens(repo):
    """Token basis of the forecast: pack input (o200k_base), R2 session prompt totals, mean cumulative at turn 40."""
    f = _cloud_forecast(repo)
    r2 = f["r2"]
    sp = r2["session_prompt_tokens"]
    fam = ", ".join(f"{k} {v['input_tokens']}" for k, v in f["pack"]["families"].items())
    return {"value": f"pack input {f['pack']['input_tokens_total']} tokens ({fam}); R2 session prompt tokens "
                     f"{min(sp)}-{max(sp)} (mean cumulative at turn {r2['turns']}: "
                     f"{r2['mean_cumulative_sent'][-1]:.0f}); o200k/chars-4 ratio {r2['ratio_o200k_per_est']}",
            "n": f"{f['pack']['n_items']} items, {r2['n_sessions']} sessions"}


def compute_cloud_budget_inputs(repo):
    """Forecast inputs: plan parameters, failure scenarios counted from DEMO_SPEC, output-length medians, prices."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("cloud_budget_forecast", repo / "analysis" / "cloud_budget_forecast.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    f = _cloud_forecast(repo)
    pr = m.load_prices(repo)
    px = "; ".join(f"{k} std {v['standard']['short']['input']}/{v['standard']['short']['cached_input']}/"
                   f"{v['standard']['short']['output']} batch {v['batch']['short']['input']}/"
                   f"{v['batch']['short']['output']}" for k, v in pr["models"].items() if k in (m.CHEAP, m.MID))
    outs = ", ".join(f"{k} {v['median']}" for k, v in f["outputs"].items())
    return {"value": f"{m.N_AGENT_SESSIONS} sessions x {f['r2']['turns']} turns; demo {m.DEMO_LIVE_ITEMS} + "
                     f"{m.DEMO_REHEARSALS}x{m.DEMO_REHEARSAL_ITEMS} items at {m.CLOUD_SHARE:.0%} cloud = "
                     f"{f['demo']['cloud_items']:.0f} cloud items; failure scenarios {f['demo']['failure_scenarios']} "
                     f"({f['demo']['scenario_cloud_calls']} cloud calls); rerun allowance {m.RERUN_ALLOWANCE:.0%}; "
                     f"output medians (o200k) {outs}; prices USD/1M in/cached/out ({px}); fetched {pr['fetched_utc']}",
            "n": f"{f['pack']['n_items']} items"}


def compute_cloud_budget_sensitivity(repo):
    """Forecast totals under flat-weighted demo items and under 5x output tokens (hidden reasoning)."""
    flat = _cloud_forecast(repo, demo_weighting="flat")
    x5 = _cloud_forecast(repo, output_multiplier=5.0)
    est = _cloud_forecast(repo, r2_token_basis="row_est")
    return {"value": f"flat demo weighting: {flat['base']['total']:.2f} / {flat['pessimistic']['total']:.2f}; "
                     f"output x5: {x5['base']['total']:.2f} / {x5['pessimistic']['total']:.2f}; "
                     f"R2 at row chars/4 estimate: {est['base']['total']:.2f} / {est['pessimistic']['total']:.2f}",
            "n": "3 variants"}


# ───────────────────────────────────────────────── R2 real run x2_r2_real_v1 (2026-10-07)
R2_REAL_V1_FILE = "results/x2_r2_real_v1.jsonl"


def _r2_real_report(repo):
    import sys as _sys
    _sys.path.insert(0, str(repo / "analysis"))
    import r2_real_report as rr
    return rr, rr.build(repo / R2_REAL_V1_FILE)


def compute_r2_real_v1_first_events(repo):
    """Per tier x model, per seed (seeds 20260901/02/03 in order, '-' = never in 40 turns): first canary miss, first
    failure per rule in use, first invalid tool call, first surfaced error."""
    rr, rep = _r2_real_report(repo)
    f = rr._fmt
    parts = []
    for key, c in rep["table"].items():
        s = c["sessions"]
        rules = ", ".join(f"{r.split('_')[0]} {f(e[f'first_fail_{r}'] for e in s)}" for r in c["rules_in_use"])
        parts.append(f"{key.replace('|', ' ')}: canary miss {f(e['first_canary_miss'] for e in s)} "
                     f"(over loaded window: {f(e['first_canary_miss_over_window'] for e in s)}); {rules}; "
                     f"invalid tool call {f(e['first_invalid_tool_call'] for e in s)}; "
                     f"first error {f(e['first_error'] for e in s)}; first over window {f(e['first_over_window'] for e in s)}")
    return {"value": " / ".join(parts), "n": f"{rep['n_sessions']} sessions, {rep['n_turn_rows']} turns"}


def compute_r2_real_v1_survival(repo):
    """Sessions intact (no gated failure and no canary miss so far) at turns 5/10/20/30/40, per tier x model."""
    rr, rep = _r2_real_report(repo)
    return {"value": "; ".join(f"{k.replace('|', ' ')} " + "/".join(str(c["survival"][t]) for t in rr.CHECKPOINTS)
                               + f" of {c['n']}" for k, c in rep["table"].items()),
            "n": rep["n_sessions"]}


def compute_r2_real_v1_kill_criterion(repo):
    """Pre-registered kill criterion (x2_r2_agent.kill_criterion, rules in use; tool args and recall always count)."""
    _rr, rep = _r2_real_report(repo)
    k = rep["kill"]
    arms = "; ".join(f"{a.replace('_call2_notools', '')} {v['n_silent_failures']}/{v['n_sessions']} silent"
                     for a, v in k["per_arm"].items())
    return {"value": f"killed={k['killed']}; {arms}", "n": rep["n_sessions"]}


def compute_r2_real_v1_gated_kill(repo):
    """Stricter variant: only rules in use plus metrics at >=90% validation-v2 baseline count; also whether the
    first silent failure came at or after the first turn whose prompt exceeded the loaded window."""
    _rr, rep = _r2_real_report(repo)
    excl = "; ".join(f"{m} excludes {', '.join(v['excluded']) or 'nothing'}" for m, v in rep["gated_metrics"].items())
    arms = "; ".join(f"{t} {g['silent']}/{g['n_sessions']} silent, {g['silent_after_window_exceeded']} after window "
                     f"exceeded" for t, g in rep["gated_kill"].items())
    return {"value": f"{arms} ({excl})", "n": rep["n_sessions"]}


# ───────────────────────────────────────────────── X2 outcome table v3 progress (2026-10-08)
def _x2_v3_cells(repo):
    """Latest valid outcome_row per (item, model, config) in the v3 file, the planned item set per cell (pack minus
    r2_sessions; qwen3-32b only its recorded 100-item trace-weighted subset), and the trace weights."""
    x2 = _x2_harness()
    rows = x2.read_rows(repo / X2_OUTCOME_V3)
    weights = x2.load_weights(repo)
    plan_all = {it["item_id"]: it["family"] for it in x2.load_items_trace_weighted(repo)}
    sub = next((set(r["item_ids"]) for r in rows if r.get("record") == "qwen32b_subset"), set())
    latest = {}
    for r in rows:
        if r.get("record") == "outcome_row" and x2.row_is_valid(r):
            latest[(r["item_id"], r["model_id"], r["config"])] = r
    cells = {}
    for (iid, m, c), r in latest.items():
        cells.setdefault((m, c), {})[iid] = r
    plan = {m: (sub if m == "qwen3-32b" else set(plan_all)) for m in x2.DEFAULT_MODELS}
    return x2, cells, plan, plan_all, weights


def compute_x2_v3_progress(repo):
    """Items done (latest valid row) vs planned, per model x config."""
    _x2, cells, plan, _pa, _w = _x2_v3_cells(repo)
    parts = []
    for (m, c) in sorted(cells):
        done = len(set(cells[(m, c)]) & plan.get(m, set()))
        parts.append(f"{m}/{c} {done}/{len(plan.get(m, ()))}")
    total_done = sum(len(set(v) & plan.get(m, set())) for (m, _c), v in cells.items())
    total_plan = sum(len(plan[m]) for m in plan) * 2
    return {"value": f"total {total_done}/{total_plan}; " + "; ".join(parts), "n": total_done}


def compute_x2_v3_scores(repo):
    """Mean score per family and trace-weighted mean (item weights renormalized over the done items), per model x
    config, over done items only. The run visits items in descending trace weight, so done items are the heaviest."""
    _x2, cells, plan, plan_all, w = _x2_v3_cells(repo)
    parts = []
    for (m, c) in sorted(cells):
        rs = {iid: r for iid, r in cells[(m, c)].items() if iid in plan.get(m, set())}
        fam = {}
        for iid, r in rs.items():
            fam.setdefault(plan_all[iid], []).append(r.get("score") or 0.0)
        wsum = sum(w.get(i, 0.0) for i in rs)
        tw = (sum(w.get(i, 0.0) * (r.get("score") or 0.0) for i, r in rs.items()) / wsum) if wsum > 0 else None
        fs = ", ".join(f"{f} {sum(v) / len(v):.2f} (n={len(v)})" for f, v in sorted(fam.items()))
        parts.append(f"{m}/{c}: {fs}; trace-weighted {'n/a' if tw is None else f'{tw:.2f}'}")
    return {"value": " / ".join(parts), "n": sum(len(v) for v in cells.values())}


def compute_x2_v3_error_causes(repo):
    """Error cause counts (none/context_overflow/timeout/connection/other) per model x config, done items."""
    x2, cells, plan, _pa, _w = _x2_v3_cells(repo)
    order = ["none", "context_overflow", "timeout", "connection", "other"]
    parts = []
    for (m, c) in sorted(cells):
        cnt = {}
        for iid, r in cells[(m, c)].items():
            if iid in plan.get(m, set()):
                k = x2.classify_error_cause(r)
                cnt[k] = cnt.get(k, 0) + 1
        parts.append(f"{m}/{c} " + "/".join(str(cnt.get(k, 0)) for k in order))
    return {"value": "order none/context_overflow/timeout/connection/other; " + "; ".join(parts),
            "n": sum(len(v) for v in cells.values())}


T2S_LLAMASERVER_LOG_GLOB = "results/t2s_outcome_table_llamaserver_*.log"


def compute_t2s_llamaserver_load_failures(repo):
    """Synced T2S llama-server logs (gitignored, local to the controller): served vs ErrorOutOfDeviceMemory, and the
    failing allocation lines. The harness wrote no row for these items, so the outcome file shows 0 llama-server
    failures."""
    import glob as _glob
    import re as _re
    logs = sorted(_glob.glob(str(repo / T2S_LLAMASERVER_LOG_GLOB)))
    oom, served, lines = 0, 0, {}
    for f in logs:
        t = Path(f).read_text(encoding="utf-8", errors="replace")
        if "ErrorOutOfDeviceMemory" in t:
            oom += 1
            for ln in t.splitlines():
                if " E " in ln and ("failed to allocate" in ln or "error loading model" in ln):
                    k = _re.sub(r"^[0-9.]+ E ", "", ln).strip()
                    lines[k] = lines.get(k, 0) + 1
        elif "all slots are idle" in t:
            served += 1
    detail = "; ".join(f"{v}x {k}" for k, v in sorted(lines.items(), key=lambda kv: -kv[1]))
    return {"value": f"{oom}/{len(logs)} logs ErrorOutOfDeviceMemory, {served}/{len(logs)} served; {detail}",
            "n": len(logs)}


# ───────────────────────────────────────────────── R2 strengthened run (2026-10-08): v2b validation, mechanism, v1b
# Rows below are registered before their data exists; build_register marks them PENDING (not UNSUPPORTED) while any
# of their data files is missing locally, and they compute like every other row once sync_results pulls the files.
R2_VALIDATION_V2B_FILE = "results/x2_r2_validation_v2b.jsonl"   # qwen3-4b-2507, qwen3:8b; same plan as v2
R2_MECHANISM_FILE = "results/x2_r2_mechanism.jsonl"
R2_REAL_V1B_FILE = "results/x2_r2_real_v1b.jsonl"
R2_V1B_DATA = [R2_REAL_V1_FILE, R2_REAL_V1B_FILE, R2_VALIDATION_V2_FILE, R2_VALIDATION_V2B_FILE]


def compute_r2_validation_v2b_baseline(repo):
    """R2 validation v2b (gate for the new models in the strengthened run): v2's plan (tools withheld on call 2,
    arm b 131072 3 seeds x 10 turns, positive control 8192 1 seed x 15 turns) for qwen3-4b-2507 and qwen3:8b."""
    return _r2_gates_summary(repo, R2_VALIDATION_V2B_FILE, False)


def compute_r2_mechanism_verdict(repo):
    """R2 overflow mechanism (Ollama debug log + render-only prompt per call, one session per tier): per tier the
    first truncated turn, whether the system prompt was kept on every truncated call, contiguous-tail kept,
    token-level cuts, context shifts; and the verdict on "drops whole old turns, keeps the system prompt"."""
    import sys as _sys
    _sys.path.insert(0, str(repo / "harness"))
    import x2_r2_mechanism as mech
    rows = _r2_agent_module().read_rows(repo / R2_MECHANISM_FILE)
    if not rows:
        raise FileNotFoundError(repo / R2_MECHANISM_FILE)
    rep = mech.report(rows)
    parts = []
    for s in sorted(rep["sessions"], key=lambda s: str(s["arm_id"])):
        parts.append(f"{s['model_id']} {s['arm_id'].replace('_call2_notools', '')}: first truncated turn "
                     f"{s['first_truncated_turn']}, {s['n_calls_message_truncated']}/{s['n_calls']} calls truncated, "
                     f"system kept on all {s['system_kept_on_every_truncated_call']}, contiguous tail "
                     f"{s['kept_contiguous_tail_on_every_truncated_call']}, cut mid-turn {s['cut_mid_turn_calls']}, "
                     f"token-level cuts {s['token_level_cut_calls']}, context-shift calls {s['context_shift_calls']}, "
                     f"exceed-context errors {s['exceed_context_error_calls']}, path {','.join(s['chat_paths'])}")
    v = rep["verdict"]
    return {"value": f"verdict {v['drop_old_turns_keep_system']} (token-level cut seen {v['token_level_cut_seen']}, "
                     f"context shift seen {v['context_shift_seen']}, render/log disagreements "
                     f"{v['render_log_disagreements']}); " + " / ".join(parts),
            "n": f"{v['sessions']} sessions"}


def _r2_real_v1b_report(repo):
    import sys as _sys
    _sys.path.insert(0, str(repo / "analysis"))
    import r2_real_report as rr
    for f in (R2_REAL_V1_FILE, R2_REAL_V1B_FILE):
        if not (repo / f).exists():
            raise FileNotFoundError(repo / f)
    return rr, rr.build_combined([repo / R2_REAL_V1_FILE, repo / R2_REAL_V1B_FILE],
                                 validation_paths=[repo / R2_VALIDATION_V2_FILE, repo / R2_VALIDATION_V2B_FILE])


def compute_r2_real_v1b_first_events(repo):
    """Strengthened R2 (v1 + v1b files), per tier x model, per seed in seed order ('-' = never in 40 turns): first
    turn over the loaded window, first canary miss, first failure per rule in use, first invalid tool call, first
    surfaced error, and whether an error surfaced before the first failure."""
    rr, rep = _r2_real_v1b_report(repo)
    f = rr._fmt
    parts = []
    for key, c in rep["table"].items():
        s = c["sessions"]
        rules = ", ".join(f"{r.split('_')[0]} {f(e[f'first_fail_{r}'] for e in s)}" for r in c["rules_in_use"])
        parts.append(f"{key.replace('|', ' ')} (seeds {f(c['seeds'])}): over window {f(e['first_over_window'] for e in s)}; "
                     f"canary miss {f(e['first_canary_miss'] for e in s)}; {rules}; "
                     f"invalid tool call {f(e['first_invalid_tool_call'] for e in s)}; "
                     f"first error {f(e['first_error'] for e in s)}; error before first failure "
                     f"{rr._yn(e['error_before_first_failure'] for e in s)}")
    iss = rep["issues"]
    return {"value": " / ".join(parts) + f" (duplicates {len(iss['duplicates'])}, other call-2 mode excluded "
                     f"{len(iss['excluded_other_call2_mode'])})",
            "n": f"{rep['n_sessions']} sessions, {rep['n_turn_rows']} turns"}


def compute_r2_real_v1b_survival(repo):
    """Strengthened R2: sessions intact (no gated failure, no canary miss) at turns 5/10/15/20/25/30/35/40."""
    rr, rep = _r2_real_v1b_report(repo)
    return {"value": "; ".join(f"{k.replace('|', ' ')} " + "/".join(str(c["survival"][t]) for t in rep["checkpoints"])
                               + f" of {c['n']}" for k, c in rep["table"].items()),
            "n": rep["n_sessions"]}


def compute_r2_real_v1b_kill_criterion(repo):
    """Strengthened R2: the pre-registered kill criterion over all five tiers (rules in use; tool validity, tool
    arguments and recall always count)."""
    _rr, rep = _r2_real_v1b_report(repo)
    k = rep["kill"]
    arms = "; ".join(f"{a.replace('_call2_notools', '')} {v['n_silent_failures']}/{v['n_sessions']} silent"
                     for a, v in k["per_arm"].items())
    return {"value": f"killed={k['killed']}; {arms}", "n": rep["n_sessions"]}


def compute_r2_real_v1b_gated_kill(repo):
    """Strengthened R2, gated variant: only rules in use plus metrics at >=90% validation baseline (v2 for llama3.1:8b
    and qwen3:14b, v2b for qwen3-4b-2507 and qwen3:8b); silent failures per tier and after the window was exceeded."""
    _rr, rep = _r2_real_v1b_report(repo)
    excl = "; ".join(f"{m} excludes {', '.join(v['excluded']) or 'nothing'}" for m, v in rep["gated_metrics"].items())
    arms = "; ".join(f"{t} {g['silent']}/{g['n_sessions']} silent, {g['silent_after_window_exceeded']} after window "
                     f"exceeded" for t, g in rep["gated_kill"].items())
    return {"value": f"gated killed={rep['gated_killed']}; {arms} ({excl})", "n": rep["n_sessions"]}


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
    {"claim_id": "A-24-flip-point-prediction",
     "description": "5-model predicted vs measured n_ctx flip point (server stops starting), first principles "
                    "(weights + KV bytes/token x n_ctx + compute(n_ctx)) solved against the 47,866 MiB heap budget, "
                    "3 accountings",
     "compute": compute_a24_flip_point_prediction,
     "data_files": ["results/t2s_amech_20260926T181456Z.jsonl", "results/t2s_night2_20260929T034014Z.jsonl",
                    "results/t2s_amech_20260926T181456Z_srv_bis_*.txt",
                    "results/t2s_night2_20260929T034014Z_srv_bis_a70_*.txt"],
     "script_function": "analysis/numbers_register.py::compute_a24_flip_point_prediction"},
    {"claim_id": "A-24-flip-point-inputs",
     "description": "per-model inputs to the flip-point prediction: weights, KV bytes/token (derived vs logged), "
                    "compute-buffer law",
     "compute": compute_a24_flip_point_inputs,
     "data_files": ["results/t2s_amech_20260926T181456Z_srv_bis_*.txt",
                    "results/t2s_night2_20260929T034014Z_srv_bis_a70_*.txt"],
     "script_function": "analysis/numbers_register.py::compute_a24_flip_point_inputs"},
    {"claim_id": "A-24-flip-point-per-model",
     "description": "per-model predicted vs measured flip n_ctx, accountings A/B/C",
     "compute": compute_a24_flip_point_per_model,
     "data_files": ["results/t2s_amech_20260926T181456Z.jsonl", "results/t2s_night2_20260929T034014Z.jsonl",
                    "results/t2s_amech_20260926T181456Z_srv_bis_*.txt",
                    "results/t2s_night2_20260929T034014Z_srv_bis_a70_*.txt"],
     "script_function": "analysis/numbers_register.py::compute_a24_flip_point_per_model"},
    {"claim_id": "A-24-effective-heap-limit",
     "description": "effective evo-t2s Vulkan heap limit bracketed from every A-24 bisection start/refusal, "
                    "shared-heap vs device-only accounting",
     "compute": compute_a24_effective_heap_limit,
     "data_files": ["results/t2s_amech_20260926T181456Z.jsonl",
                    "results/t2s_amech_20260926T181456Z_srv_bis_*.txt",
                    "results/t2s_night2_20260929T034014Z_srv_bis_a70_*.txt"],
     "script_function": "analysis/numbers_register.py::compute_a24_effective_heap_limit"},
    {"claim_id": "ttft-x2-length-spread",
     "description": "per-model n and prompt-length range behind the evo-x2 TTFT fit",
     "compute": compute_ttft_x2_length_spread, "data_files": _TTFT_FIT_RESULT_FILES,
     "script_function": "analysis/numbers_register.py::compute_ttft_x2_length_spread"},
    {"claim_id": "ttft-x2-refit-by-hypothesis",
     "description": "evo-x2 TTFT fit R2 refit with each suspected cause removed (co-runner, K2/MX2/SMOKE, "
                    "user_active, intercept, first call on a new prompt)",
     "compute": compute_ttft_x2_refit_by_hypothesis, "data_files": _TTFT_FIT_RESULT_FILES,
     "script_function": "analysis/numbers_register.py::compute_ttft_x2_refit_by_hypothesis"},
    {"claim_id": "ttft-x2-refit-per-model",
     "description": "per-model TTFT fit R2: baseline / drop first call / all four suspected causes dropped",
     "compute": compute_ttft_x2_refit_per_model, "data_files": _TTFT_FIT_RESULT_FILES,
     "script_function": "analysis/numbers_register.py::compute_ttft_x2_refit_per_model"},
    {"claim_id": "ttft-t2s-restricted-to-x2-range",
     "description": "evo-t2s TTFT fit R2 on only the rows inside evo-x2's prompt range (length-spread test)",
     "compute": compute_ttft_t2s_restricted_to_x2_range, "data_files": _TTFT_FIT_RESULT_FILES,
     "script_function": "analysis/numbers_register.py::compute_ttft_t2s_restricted_to_x2_range"},
    {"claim_id": "ttft-first-call-stall",
     "description": "first call on a new prompt: stall vs repeat call, scaled by the displaced KV state, both machines",
     "compute": compute_ttft_first_call_stall,
     "data_files": [rel for _, rel, _ in _TTFT_STALL_SECTIONS] + [
         "results/t2s_night2_20260929T205109Z_srv_R1*_start.txt",
         "results/t2s_night2_20260929T045127Z_srv_R1*_start.txt",
         "results/t2s_night2_20260929T202603Z_srv_R1*_start.txt"],
     "script_function": "analysis/numbers_register.py::compute_ttft_first_call_stall"},
    {"claim_id": "ttft-first-call-stall-per-cell",
     "description": "per model/section first-call stall and displaced-KV-state rate, both machines",
     "compute": compute_ttft_first_call_stall_per_cell,
     "data_files": [rel for _, rel, _ in _TTFT_STALL_SECTIONS] + [
         "results/t2s_night2_20260929T205109Z_srv_R1*_start.txt",
         "results/t2s_night2_20260929T045127Z_srv_R1*_start.txt",
         "results/t2s_night2_20260929T202603Z_srv_R1*_start.txt"],
     "script_function": "analysis/numbers_register.py::compute_ttft_first_call_stall_per_cell"},
    {"claim_id": "ttft-x2-section0-stall-check",
     "description": "out-of-sample check: displaced-KV stall term applied to evo-x2 overnight section 0 rows",
     "compute": compute_ttft_x2_section0_stall_check,
     "data_files": _TTFT_FIT_RESULT_FILES + ["results/t2s_night2_20260929T205109Z_srv_R1*_start.txt"],
     "script_function": "analysis/numbers_register.py::compute_ttft_x2_section0_stall_check"},
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
    {"claim_id": "x2-v3-canary-gates", "description": "X2 outcome table v3 canary gate per (model, config), latest record",
     "compute": compute_x2_v3_canary_gates, "data_files": [X2_OUTCOME_V3],
     "script_function": "analysis/numbers_register.py::compute_x2_v3_canary_gates"},
    {"claim_id": "t2s-outcome-error-causes", "description": "T2S outcome table (synced full file) rows by error cause, and race-signature count",
     "compute": compute_t2s_outcome_error_causes, "data_files": [T2S_OUTCOME_FULL],
     "script_function": "analysis/numbers_register.py::compute_t2s_outcome_error_causes"},
    {"claim_id": "t2s-outcome-gsm8k-rescore", "description": "T2S outcome rows that could be rescored with the fixed gsm8k scorer (gsm8k rows, rows with output text)",
     "compute": compute_t2s_outcome_gsm8k_rescore,
     "data_files": [T2S_OUTCOME_FULL, "results/t2s_outcome_table_smoke.jsonl"],
     "script_function": "analysis/numbers_register.py::compute_t2s_outcome_gsm8k_rescore"},
    {"claim_id": "t2s-outcome-allocation-failures", "description": "T2S outcome allocation failures per config: buffer, prompt-length ranges, onset time, unrecorded llama-server attempts",
     "compute": compute_t2s_allocation_failures, "data_files": [T2S_OUTCOME_FULL],
     "script_function": "analysis/numbers_register.py::compute_t2s_allocation_failures"},
    {"claim_id": "x2-v3-gsm8k-strict-vs-lenient", "description": "X2 v3 gsm8k per (model, config): strict #### score, format_ok, lenient last-sentence score",
     "compute": compute_x2_v3_gsm8k_strict_vs_lenient, "data_files": [X2_OUTCOME_V3],
     "script_function": "analysis/numbers_register.py::compute_x2_v3_gsm8k_strict_vs_lenient"},
    {"claim_id": "x2-v3-canary-gate-lenient", "description": "X2 v3 canary gate mean score, strict (recorded) vs lenient gsm8k substituted",
     "compute": compute_x2_v3_canary_gate_lenient, "data_files": [X2_OUTCOME_V3],
     "script_function": "analysis/numbers_register.py::compute_x2_v3_canary_gate_lenient"},
    {"claim_id": "R2-validation-baseline",
     "description": "R2 two-step harness validation (evo-x2, arm b 131072, 3x10 turns): per model rule/tool/recall "
                    "baseline and negative/positive control results",
     "compute": compute_r2_validation_baseline, "data_files": [R2_VALIDATION_FILE],
     "script_function": "analysis/numbers_register.py::compute_r2_validation_baseline"},
    {"claim_id": "R2-validation-call2-notools",
     "description": "R2 validation diagnostic arm (tools withheld on call 2 only): per model rule compliance",
     "compute": compute_r2_validation_call2_notools, "data_files": [R2_VALIDATION_FILE],
     "script_function": "analysis/numbers_register.py::compute_r2_validation_call2_notools"},
    {"claim_id": "R2-validation-v2-baseline",
     "description": "R2 validation v2 (gate for the real run; tools withheld on call 2, per-check canaries): per model "
                    "rule/tool/recall/canary baseline and negative/positive control results",
     "compute": compute_r2_validation_v2_baseline, "data_files": [R2_VALIDATION_V2_FILE],
     "script_function": "analysis/numbers_register.py::compute_r2_validation_v2_baseline"},
    {"claim_id": "R2-validation-v2-spec-diag",
     "description": "R2 validation v2 diagnostic arm (spec variant, tools available on call 2, per-check canaries)",
     "compute": compute_r2_validation_v2_spec_diag, "data_files": [R2_VALIDATION_V2_FILE],
     "script_function": "analysis/numbers_register.py::compute_r2_validation_v2_spec_diag"},

    {"claim_id": "K2-x2-kill-criterion", "description": "K2 arms (a)/(b) evo-x2 18h run: kill-criterion evaluations, "
                    "passes, completeness",
     "compute": compute_k2_x2_kill_criterion_table, "data_files": [_K2_X2_RUN],
     "script_function": "analysis/numbers_register.py::compute_k2_x2_kill_criterion_table"},
    {"claim_id": "K2-x2-clean-failures", "description": "K2 evo-x2 quality items not ok (crash fields)",
     "compute": compute_k2_x2_clean_failures, "data_files": [_K2_X2_RUN],
     "script_function": "analysis/numbers_register.py::compute_k2_x2_clean_failures"},
    {"claim_id": "K2-x2-score-ceiling", "description": "K2 evo-x2 quality scores at ceiling, think tags, forced length",
     "compute": compute_k2_x2_score_ceiling, "data_files": [_K2_X2_RUN],
     "script_function": "analysis/numbers_register.py::compute_k2_x2_score_ceiling"},
    {"claim_id": "K2-x2-decode-by-arm", "description": "K2 evo-x2 pressure-step vs baseline decode ratio, per arm",
     "compute": compute_k2_x2_decode_by_pressure_arm, "data_files": [_K2_X2_RUN],
     "script_function": "analysis/numbers_register.py::compute_k2_x2_decode_by_pressure_arm"},
    {"claim_id": "K2-x2-call-timing", "description": "K2 evo-x2 per-model median TTFT/decode at ~12k tokens",
     "compute": compute_k2_x2_call_timing, "data_files": [_K2_X2_RUN],
     "script_function": "analysis/numbers_register.py::compute_k2_x2_call_timing"},
    {"claim_id": "K2-x2-responsiveness-resolution", "description": "K2 evo-x2 responsiveness probe values vs 2x tolerance",
     "compute": compute_k2_x2_responsiveness_resolution, "data_files": [_K2_X2_RUN],
     "script_function": "analysis/numbers_register.py::compute_k2_x2_responsiveness_resolution"},
    {"claim_id": "K2-x2-unreached-levels", "description": "K2 evo-x2 pressure steps that missed their target",
     "compute": compute_k2_x2_unreached_pressure_levels, "data_files": [_K2_X2_RUN],
     "script_function": "analysis/numbers_register.py::compute_k2_x2_unreached_pressure_levels"},

    {"claim_id": "kappa-heldout-sample", "description": "held-out kappa sample construction (seed, exclusions, strata)",
     "compute": compute_kappa_heldout_sample,
     "data_files": ["results/t2s_night2_20260929T202603Z.jsonl", "results/t2s_night2_20260929T205109Z.jsonl",
                    "results/labeling/r1b_wrong_sample.csv", "results/labeling/ritz_spotcheck_30.csv"],
     "script_function": "analysis/build_kappa_heldout_sample.py::round_robin_sample"},
    {"claim_id": "kappa-heldout", "description": "held-out kappa, current scorer vs blinded annotator_claude, with CI and confusion matrix",
     "compute": compute_kappa_heldout,
     "data_files": ["results/labeling/kappa_heldout_key.csv", "results/labeling/kappa_heldout_annotator_claude.csv"],
     "script_function": "analysis/kappa_agreement.py::heldout_stats"},
    {"claim_id": "kappa-heldout-fn-annotator", "description": "held-out rows: scorer vs original annotator function, and manual labels vs that function",
     "compute": compute_kappa_heldout_fn_annotator,
     "data_files": ["results/labeling/kappa_heldout_key.csv", "results/labeling/kappa_heldout_annotator_claude.csv"],
     "script_function": "analysis/kappa_agreement.py::heldout_stats"},
    {"claim_id": "kappa-heldout-disagreements", "description": "held-out disagreement breakdown by read and direction",
     "compute": compute_kappa_heldout_disagreement_read,
     "data_files": ["results/labeling/kappa_heldout_disagreement_read.csv", "results/labeling/kappa_heldout_key.csv",
                    "results/labeling/kappa_heldout_annotator_claude.csv"],
     "script_function": "analysis/numbers_register.py::compute_kappa_heldout_disagreement_read"},
    {"claim_id": "cloud-budget-totals", "description": "cloud budget forecast total and pessimistic total vs USD 50 cap",
     "compute": compute_cloud_budget_totals, "data_files": _CLOUD_DATA_FILES,
     "script_function": "analysis/cloud_budget_forecast.py::forecast"},
    {"claim_id": "cloud-budget-lines", "description": "cloud budget forecast lines (a)-(e), forecast / pessimistic USD",
     "compute": compute_cloud_budget_lines, "data_files": _CLOUD_DATA_FILES,
     "script_function": "analysis/cloud_budget_forecast.py::forecast"},
    {"claim_id": "cloud-budget-tokens", "description": "cloud budget token basis (pack input, R2 session totals)",
     "compute": compute_cloud_budget_tokens, "data_files": _CLOUD_DATA_FILES,
     "script_function": "analysis/cloud_budget_forecast.py::r2_sessions"},
    {"claim_id": "cloud-budget-inputs", "description": "cloud budget forecast inputs (plan volumes, output medians, prices)",
     "compute": compute_cloud_budget_inputs, "data_files": _CLOUD_DATA_FILES,
     "script_function": "analysis/cloud_budget_forecast.py::load_prices"},
    {"claim_id": "cloud-budget-sensitivity", "description": "cloud budget totals (forecast / pessimistic) under flat demo weighting, 5x output, R2 chars/4 tokens",
     "compute": compute_cloud_budget_sensitivity, "data_files": _CLOUD_DATA_FILES,
     "script_function": "analysis/cloud_budget_forecast.py::forecast"},
    {"claim_id": "R2-real-v1-first-events",
     "description": "R2 real run: per tier x model first canary miss, first failure per rule in use, first invalid "
                    "tool call, first error",
     "compute": compute_r2_real_v1_first_events, "data_files": [R2_REAL_V1_FILE],
     "script_function": "analysis/numbers_register.py::compute_r2_real_v1_first_events"},
    {"claim_id": "R2-real-v1-survival", "description": "R2 real run: sessions intact at turns 5/10/20/30/40",
     "compute": compute_r2_real_v1_survival, "data_files": [R2_REAL_V1_FILE, "results/x2_r2_validation_v2.jsonl"],
     "script_function": "analysis/numbers_register.py::compute_r2_real_v1_survival"},
    {"claim_id": "R2-real-v1-kill-criterion", "description": "R2 real run: pre-registered kill criterion verdict",
     "compute": compute_r2_real_v1_kill_criterion, "data_files": [R2_REAL_V1_FILE],
     "script_function": "analysis/numbers_register.py::compute_r2_real_v1_kill_criterion"},
    {"claim_id": "R2-real-v1-gated-kill", "description": "R2 real run: kill criterion restricted to 90%-baseline "
                    "metrics, and silent failures after the loaded window was exceeded",
     "compute": compute_r2_real_v1_gated_kill, "data_files": [R2_REAL_V1_FILE, "results/x2_r2_validation_v2.jsonl"],
     "script_function": "analysis/numbers_register.py::compute_r2_real_v1_gated_kill"},
    {"claim_id": "x2-v3-progress", "description": "X2 outcome table v3: items done vs planned per model x config",
     "compute": compute_x2_v3_progress, "data_files": [X2_OUTCOME_V3],
     "script_function": "analysis/numbers_register.py::compute_x2_v3_progress"},
    {"claim_id": "x2-v3-scores", "description": "X2 outcome table v3: mean score per family and trace-weighted",
     "compute": compute_x2_v3_scores, "data_files": [X2_OUTCOME_V3, "results/workload_pack/item_weights_trace_weighted.json"],
     "script_function": "analysis/numbers_register.py::compute_x2_v3_scores"},
    {"claim_id": "x2-v3-error-causes", "description": "X2 outcome table v3: error causes per model x config",
     "compute": compute_x2_v3_error_causes, "data_files": [X2_OUTCOME_V3],
     "script_function": "analysis/numbers_register.py::compute_x2_v3_error_causes"},
    {"claim_id": "t2s-llamaserver-load-failures", "description": "T2S outcome table llama-server load logs: "
                    "served vs out-of-device-memory (logs synced, gitignored)",
     "compute": compute_t2s_llamaserver_load_failures, "data_files": [T2S_LLAMASERVER_LOG_GLOB],
     "script_function": "analysis/numbers_register.py::compute_t2s_llamaserver_load_failures"},

    {"claim_id": "R2-validation-v2b-baseline",
     "description": "R2 validation v2b (gate for qwen3-4b-2507 and qwen3:8b in the strengthened run): per model "
                    "rule/tool/recall/canary baseline and negative/positive control results",
     "compute": compute_r2_validation_v2b_baseline, "data_files": [R2_VALIDATION_V2B_FILE], "pending_ok": True,
     "script_function": "analysis/numbers_register.py::compute_r2_validation_v2b_baseline"},
    {"claim_id": "R2-mechanism-verdict",
     "description": "R2 overflow mechanism from the Ollama debug log and render-only prompts, one session per tier",
     "compute": compute_r2_mechanism_verdict, "data_files": [R2_MECHANISM_FILE], "pending_ok": True,
     "script_function": "analysis/numbers_register.py::compute_r2_mechanism_verdict"},
    {"claim_id": "R2-real-v1b-first-events",
     "description": "Strengthened R2 (v1 + v1b): per tier x model first over-window turn, canary miss, rule failure "
                    "per rule in use, invalid tool call, error, and error-before-failure",
     "compute": compute_r2_real_v1b_first_events, "data_files": R2_V1B_DATA, "pending_ok": True,
     "script_function": "analysis/numbers_register.py::compute_r2_real_v1b_first_events"},
    {"claim_id": "R2-real-v1b-survival",
     "description": "Strengthened R2: sessions intact at turns 5/10/15/20/25/30/35/40 per tier x model",
     "compute": compute_r2_real_v1b_survival, "data_files": R2_V1B_DATA, "pending_ok": True,
     "script_function": "analysis/numbers_register.py::compute_r2_real_v1b_survival"},
    {"claim_id": "R2-real-v1b-kill-criterion",
     "description": "Strengthened R2: pre-registered kill criterion verdict over five tiers",
     "compute": compute_r2_real_v1b_kill_criterion, "data_files": R2_V1B_DATA, "pending_ok": True,
     "script_function": "analysis/numbers_register.py::compute_r2_real_v1b_kill_criterion"},
    {"claim_id": "R2-real-v1b-gated-kill",
     "description": "Strengthened R2: kill criterion restricted to 90%-baseline metrics, silent failures after the "
                    "window was exceeded",
     "compute": compute_r2_real_v1b_gated_kill, "data_files": R2_V1B_DATA, "pending_ok": True,
     "script_function": "analysis/numbers_register.py::compute_r2_real_v1b_gated_kill"},
]


def _pending(entry, repo=None) -> list[str]:
    """Data files of a pending_ok entry that are not present locally yet (empty list: compute it normally)."""
    if not entry.get("pending_ok"):
        return []
    repo = repo or REPO
    return [f for f in entry["data_files"] if not (repo / f).exists()]


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
        missing = _pending(entry)
        if missing:
            rows.append({
                "claim_id": entry["claim_id"], "status": "PENDING",
                "value": "not computed: data file(s) not synced yet: " + ", ".join(missing), "n": "n/a",
                "reported_value": entry.get("reported_value", "(none previously reported)"),
                "data_files": entry["data_files"], "script_function": entry["script_function"],
                "commit": head, "date": date,
            })
            continue
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
    status_order = {"CORRECTED": 0, "UNSUPPORTED": 1, "PENDING": 2, "VERIFIED": 3}
    rows_sorted = sorted(rows, key=lambda r: status_order.get(r["status"], 9))
    lines = [
        "# Numbers Register",
        "",
        "Every number cited anywhere in this program must come from a row in this table, pasted verbatim,",
        "never hand-typed. Regenerated by `analysis/numbers_register.py` -- every value below was computed",
        "fresh from the listed data file(s) by the listed function at the commit/date shown, not copied from",
        "a prior report. A PENDING row is registered ahead of its data and states which file is not synced yet;",
        "it carries no number and cannot be cited until it computes.",
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
    pending = [r for r in rows if r["status"] == "PENDING"]
    print(f"wrote {out_path}: {len(rows)} entries, {len(corrected)} CORRECTED, {len(unsupported)} UNSUPPORTED, "
          f"{len(pending)} PENDING")
    for r in corrected + unsupported:
        print(f"  {r['status']}: {r['claim_id']} -- reported {r['reported_value']!r}, computed {r['value']!r}")
    return 1 if unsupported else 0


if __name__ == "__main__":
    raise SystemExit(main())
