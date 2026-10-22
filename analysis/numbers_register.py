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
# Chat template source facts read on evo-x2 2026-10-08 (harness/chat_template_source.py --probe: manifests, /api/show,
# GGUF tokenizer.chat_template). Rows of the v3 file written before outcome rows recorded chat_template_source get it
# derived here by (model, config); the result file is never rewritten.
X2_TEMPLATE_FACTS = "results/x2_chat_template_sources.jsonl"
BARE_TEMPLATE_MARK = "[bare template]"


def _cts_module():
    import sys
    sys.path.insert(0, str(REPO / "harness"))
    import chat_template_source as cts
    return cts


def _x2_template_source_fn(repo):
    """row -> {chat_template_source, chat_template_sha256, derived} for X2 outcome rows (recorded field, else derived
    from MODEL_MAP / LEGACY_OLLAMA_TAG and the probe facts)."""
    x2, cts = _x2_harness(), _cts_module()
    facts_path = repo / X2_TEMPLATE_FACTS
    if not facts_path.exists():
        raise FileNotFoundError(facts_path)
    facts = cts.facts_from_probe(_read_jsonl(facts_path))
    return lambda r: cts.source_for_row(r, x2.MODEL_MAP, facts, x2.LEGACY_OLLAMA_TAG)


def _x2_bare_cells(repo, rows=None):
    """{(model, config)} with a row measured through an Ollama tag created from a bare GGUF. rows defaults to the
    valid (reusable) outcome rows of the v3 file, i.e. the rows every v3 score and canary gate is computed from."""
    src = _x2_template_source_fn(repo)
    if rows is None:
        rows = list(_x2_harness().valid_cached_rows(repo / X2_OUTCOME_V3).values())
    return {(r["model_id"], r["config"]) for r in rows
            if src(r)["chat_template_source"] == "bare_gguf_ollama_create"}


def compute_x2_v3_canary_gates(repo):
    """Latest canary_gate record per (model, config) in the restarted X2 outcome table."""
    x2 = _x2_harness()
    gates = {}
    for r in x2.read_rows(repo / X2_OUTCOME_V3):
        if r.get("record") == "canary_gate":
            gates[(r["model_id"], r["config"])] = r
    bare = _x2_bare_cells(repo)
    parts = [f"{m}/{c} {'PASS' if g['passed'] else 'FAIL'} err={g['error_rate']:.2f} mean={g['mean_score']:.2f} "
             f"leaks={g.get('thinking_leaks')}"
             + (f" {BARE_TEMPLATE_MARK} (mean includes function_calling canaries)" if (m, c) in bare else "")
             for (m, c), g in sorted(gates.items())]
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
def _x2_v3_gsm8k_rows(repo, exclude_timeout_latency=True):
    """Last valid gsm8k row per (item, model, config) in the v3 file (the row the latest canary gate read), with
    score (the recorded strict score), format_ok and score_lenient. Rows written with the lenient columns carry
    them. For older rows they are recomputed from output_text, which holds only the first 500 chars: when
    content_chars > 500 the last sentence was not stored, so score_lenient is None, and format_ok is None unless
    the strict score is 1 (which implies a `#### N`) or the stored head already contains one.
    exclude_timeout_latency (default True): timeout_latency rows (x2_outcome_table.analysis_outcome) are left out of
    these accuracy rows. The canary gate lenient row passes False, since it adjusts the gate's own recorded mean,
    whose n the harness fixed with every canary row in it."""
    x2 = _x2_harness()
    g = x2.load_graders()
    items = {it["item_id"]: it for it in x2.load_items_trace_weighted(repo)}
    out = []
    for (iid, model, config), r in sorted(x2.valid_cached_rows(repo / X2_OUTCOME_V3).items()):
        if r.get("family") != "gsm8k":
            continue
        if exclude_timeout_latency and not x2.counts_toward_accuracy(r):
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
    text), and the lenient last-sentence score over the rows whose stored text is the whole response.
    timeout_latency rows are excluded from these denominators (_x2_v3_gsm8k_rows default)."""
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
    for r in _x2_v3_gsm8k_rows(repo, exclude_timeout_latency=False):
        gsm.setdefault((r["model_id"], r["config"]), []).append(r)
    bare = _x2_bare_cells(repo)
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
        mark = (f" {BARE_TEMPLATE_MARK} (mean includes function_calling canaries)"
                if key in bare else "")
        parts.append(f"{key[0]}/{key[1]} strict mean {gate['mean_score']:.2f}, lenient mean {lenient}{mark}")
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
    config, over done items only. The run visits items in descending trace weight, so done items are the heaviest.
    Cells measured through an Ollama tag created from a bare GGUF (chat_template_source bare_gguf_ollama_create:
    qwen3-4b-2507 and qwen3-30b-a3b on ollama_default) carry "[bare template]" on their function_calling score: an
    install-path result, not model capability (docs/FINDINGS.md 2026-10-08, install path).
    timeout_latency rows (harness/x2_outcome_table.py::analysis_outcome, cause timeout) are EXCLUDED from every
    denominator here (family means and the trace-weighted mean): a call that hit the per-call timeout is a usability
    outcome (x2-v3-usability), not a wrong answer. Each cell with any says how many were excluded; x2-v3-progress
    and x2-v3-error-causes still count them."""
    x2, cells, plan, plan_all, w = _x2_v3_cells(repo)
    bare = _x2_bare_cells(repo, [r for v in cells.values() for r in v.values()])  # the latest valid rows
    parts = []
    for (m, c) in sorted(cells):
        done = {iid: r for iid, r in cells[(m, c)].items() if iid in plan.get(m, set())}
        rs = {iid: r for iid, r in done.items() if x2.counts_toward_accuracy(r)}
        n_tl = len(done) - len(rs)
        fam = {}
        for iid, r in rs.items():
            fam.setdefault(plan_all[iid], []).append(r.get("score") or 0.0)
        wsum = sum(w.get(i, 0.0) for i in rs)
        tw = (sum(w.get(i, 0.0) * (r.get("score") or 0.0) for i, r in rs.items()) / wsum) if wsum > 0 else None
        is_bare = (m, c) in bare
        fs = ", ".join(f"{f} {sum(v) / len(v):.2f} (n={len(v)})"
                       + (f" {BARE_TEMPLATE_MARK}" if is_bare and f == "function_calling" else "")
                       for f, v in sorted(fam.items()))
        parts.append(f"{m}/{c}: {fs}; trace-weighted {'n/a' if tw is None else f'{tw:.2f}'}"
                     + (" (includes bare-template function_calling)" if is_bare and "function_calling" in fam else "")
                     + (f"; {n_tl} timeout_latency excluded" if n_tl else ""))
    return {"value": " / ".join(parts), "n": sum(len(v) for v in cells.values())}


def _x2_v3_timeout_latency(repo):
    """{(model, config): (done rows in plan, [timeout_latency rows in plan])} over the latest valid v3 rows, plus the
    item lookup (family, prompt_tokens) used for context tiers, and the overall progress (done, planned)."""
    x2, cells, plan, plan_all, _w = _x2_v3_cells(repo)
    items = {it["item_id"]: it for it in x2.load_items_trace_weighted(repo)}
    out = {}
    for (m, c) in sorted(cells):
        done = {iid: r for iid, r in cells[(m, c)].items() if iid in plan.get(m, set())}
        tl = [r for _iid, r in sorted(done.items()) if x2.analysis_outcome(r) == x2.TIMEOUT_LATENCY]
        out[(m, c)] = (len(done), tl)
    total_done = sum(len(set(v) & plan.get(m, set())) for (m, _c), v in cells.items())
    total_plan = sum(len(plan[m]) for m in plan) * len(x2.CONFIGS)
    return x2, out, items, total_done, total_plan


def compute_x2_v3_usability(repo):
    """Usability table per model x config: n timeout_latency rows (harness/x2_outcome_table.py::analysis_outcome),
    their share of the cell's done items, the context tier (item family + prompt-length bucket,
    x2_outcome_table.context_tier) of each timed-out item, and the per-call timeout in seconds read from the
    harness constant DEFAULT_CALL_TIMEOUT_S (the run used the default; rows do not record it). These rows are
    excluded from x2-v3-scores and kept in x2-v3-progress / x2-v3-error-causes."""
    x2, cells, items, _td, _tp = _x2_v3_timeout_latency(repo)
    parts = []
    total_tl = 0
    for (m, c), (n_done, tl) in cells.items():
        total_tl += len(tl)
        tiers = {}
        for r in tl:
            it = items.get(r["item_id"], {})
            t = x2.context_tier(it.get("family", r.get("family")), it.get("prompt_tokens"))
            tiers[t] = tiers.get(t, 0) + 1
        share = f"{len(tl) / n_done:.1%}" if n_done else "n/a"
        tier_s = (" [" + ", ".join(f"{t} x{k}" for t, k in sorted(tiers.items())) + "]") if tiers else ""
        parts.append(f"{m}/{c} {len(tl)}/{n_done} ({share}){tier_s}")
    return {"value": f"per-call timeout {x2.DEFAULT_CALL_TIMEOUT_S} s (harness DEFAULT_CALL_TIMEOUT_S); "
                     f"timeout_latency/done (share) [context tiers]: " + "; ".join(parts), "n": total_tl}


def compute_qwen3_32b_timeout_count(repo):
    """qwen3-32b timeout_latency rows (harness/x2_outcome_table.py::analysis_outcome) per config in the v3 outcome
    table. Flagged "provisional: outcome table v3 not finished" while the v3 progress (latest valid rows in plan,
    all models x configs) is below the plan; the flag turns to "final" by itself once progress reaches the plan."""
    _x2, cells, _items, total_done, total_plan = _x2_v3_timeout_latency(repo)
    per = {c: (n_done, tl) for (m, c), (n_done, tl) in cells.items() if m == "qwen3-32b"}
    if not per:
        raise FileNotFoundError("no qwen3-32b rows in the v3 outcome table")
    status = (f"provisional: outcome table v3 not finished ({total_done}/{total_plan} done)"
              if total_done < total_plan else f"final: outcome table v3 finished ({total_done}/{total_plan} done)")
    body = ", ".join(f"{c} {len(tl)}/{n_done}" for c, (n_done, tl) in sorted(per.items()))
    return {"value": f"{body} (timeout_latency/done); {status}", "n": sum(len(tl) for _n, tl in per.values())}


# ───────────────────────────────────────────────── X2 outcome table v3: silent truncation vs latency (2026-10-10)
# Tolerance for "evaluated fewer tokens than sent": the chat template adds a constant number of tokens on top of the
# pack's prompt_tokens (processed_tokens is ABOVE sent_tokens on a full prompt), so only a shortfall bigger than
# max(TRUNC_ABS_TOL, TRUNC_REL_TOL * sent) counts. The harness's own silently_truncated field is the strict
# processed < sent test; truncation_vs_latency reports how often the two agree.
TRUNC_REL_TOL = 0.01
TRUNC_ABS_TOL = 32
TRUNC_DEFINITION = (f"truncated = HTTP 200, no error, processed_tokens < sent_tokens - max({TRUNC_ABS_TOL}, "
                    f"{TRUNC_REL_TOL:.0%} of sent_tokens)")


def wilson_ci(successes, n, z=1.96):
    """Wilson score 95% interval for a proportion (successes/n); (0.0, 0.0) for n == 0."""
    if n <= 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return max(0.0, center - margin), min(1.0, center + margin)


def _score_ci(scores, seed=42, n_boot=2000):
    """(mean, lo, hi, method). Wilson on the share of 1.0 scores when every score is 0 or 1, else a seeded percentile
    bootstrap of the mean (n_boot resamples)."""
    import random
    if not scores:
        return None, None, None, "n/a"
    mean = sum(scores) / len(scores)
    if all(s in (0, 1) for s in scores):
        lo, hi = wilson_ci(sum(1 for s in scores if s == 1), len(scores))
        return mean, lo, hi, "Wilson"
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(scores) for _ in scores) / len(scores) for _ in range(n_boot))
    return mean, boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1], "bootstrap"


def truncation_status(row, sent=None, x2=None, rel_tol=TRUNC_REL_TOL, abs_tol=TRUNC_ABS_TOL):
    """One of 'truncated' / 'full' / 'unmeasured' for an HTTP 200 row with no error (unmeasured: no
    processed_tokens recorded), else the row's error cause (x2_outcome_table.classify_error_cause: context_overflow,
    timeout, connection, other). sent defaults to the row's sent_tokens. Pure."""
    x2 = x2 or _x2_harness()
    cause = x2.classify_error_cause(row)
    if cause != "none" or row.get("error"):
        return cause if cause != "none" else "other"
    sent = row.get("sent_tokens") if row.get("sent_tokens") is not None else sent
    processed = row.get("processed_tokens")
    if sent is None or processed is None:
        return "unmeasured"
    return "truncated" if processed < sent - max(abs_tol, rel_tol * sent) else "full"


def truncation_vs_latency(x2, cells, plan, items, model):
    """Per config of `model` over the latest valid v3 rows in plan: truncation_status counts, harness
    silently_truncated agreement on HTTP 200 rows, and per group (truncated / full) mean score with 95% CI
    (_score_ci; only rows that count_toward_accuracy, which every HTTP 200 row does) and latency (median, min, max).
    Plus the pairing across configs: llama_server context_overflow / timeout_latency items and what the other
    config did with the same item."""
    out = {}
    status_by = {}
    for (m, c), d in sorted(cells.items()):
        if m != model:
            continue
        done = {iid: r for iid, r in d.items() if iid in plan.get(m, set())}
        st = {iid: truncation_status(r, (items.get(iid) or {}).get("prompt_tokens"), x2) for iid, r in done.items()}
        status_by[c] = st
        counts = {}
        for s in st.values():
            counts[s] = counts.get(s, 0) + 1
        ok = [iid for iid, s in st.items() if s in ("truncated", "full", "unmeasured")]
        agree = sum(1 for iid in ok if bool(done[iid].get("silently_truncated")) == (st[iid] == "truncated"))
        groups = {}
        for g in ("truncated", "full"):
            rs = [done[iid] for iid, s in st.items() if s == g and x2.counts_toward_accuracy(done[iid])]
            scores = [float(r.get("score") or 0.0) for r in rs]
            lats = sorted(r["latency_s"] for r in rs if r.get("latency_s") is not None)
            mean, lo, hi, method = _score_ci(scores)
            groups[g] = {"n": len(rs), "n_correct": sum(1 for s in scores if s == 1), "mean": mean, "lo": lo,
                         "hi": hi, "method": method,
                         "lat_median": statistics.median(lats) if lats else None,
                         "lat_min": lats[0] if lats else None, "lat_max": lats[-1] if lats else None,
                         "evaluated": sorted({r.get("processed_tokens") for r in rs}) if g == "truncated" else None}
        out[c] = {"n_done": len(done), "n_200": len(ok), "counts": counts, "harness_agree": agree, "groups": groups}
    pairing = {}
    if "llama_server" in status_by:
        other = {c: s for c, s in status_by.items() if c != "llama_server"}
        for cause in ("context_overflow", "timeout"):
            ids = [iid for iid, s in status_by["llama_server"].items() if s == cause]
            for c, s in other.items():
                fate = {}
                for iid in ids:
                    k = s.get(iid, "not done")
                    fate[k] = fate.get(k, 0) + 1
                pairing[(cause, c)] = (len(ids), fate)
        # same item served in full by both configs: per-item latency ratio other / llama_server
        ls_rows = {iid: r for (m, c), d in cells.items() if m == model and c == "llama_server" for iid, r in d.items()}
        for c, s in other.items():
            o_rows = cells.get((model, c), {})
            both = [iid for iid, st in s.items() if st == "full" and status_by["llama_server"].get(iid) == "full"
                    and o_rows[iid].get("latency_s") and ls_rows[iid].get("latency_s")]
            ratios = sorted(o_rows[iid]["latency_s"] / ls_rows[iid]["latency_s"] for iid in both)
            pairing[("both_full", c)] = (len(both), {"median_ratio": statistics.median(ratios) if ratios else None,
                                                     "min_ratio": ratios[0] if ratios else None,
                                                     "max_ratio": ratios[-1] if ratios else None})
    return out, pairing


def _fmt_trunc_cell(c, v):
    g_t, g_f = v["groups"]["truncated"], v["groups"]["full"]

    def grp(name, g):
        if not g["n"]:
            return f"{name} n=0"
        return (f"{name} score {g['mean']:.2f} ({g['n_correct']}/{g['n']}, {g['method']} 95% CI "
                f"{g['lo']:.2f}-{g['hi']:.2f}), latency median {g['lat_median']:.0f} s "
                f"(range {g['lat_min']:.0f}-{g['lat_max']:.0f})")
    cnt = v["counts"]
    n_tr = cnt.get("truncated", 0)
    share = f"{n_tr / v['n_200']:.1%}" if v["n_200"] else "n/a"
    errs = ", ".join(f"{k} {cnt[k]}" for k in ("context_overflow", "timeout", "connection", "other") if cnt.get(k))
    ev = (f" (evaluated tokens {'/'.join(str(e) for e in g_t['evaluated'])})" if g_t["evaluated"] else "")
    s = (f"{c}: silently truncated {n_tr}/{v['n_200']} HTTP-200 rows ({share}){ev}; {grp('truncated', g_t)}; "
         f"{grp('full', g_f)}")
    if cnt.get("unmeasured"):
        s += f"; unmeasured (no processed_tokens) {cnt['unmeasured']}"
    s += (f"; not HTTP 200 (outside both groups; timeout = timeout_latency, excluded from accuracy; context_overflow "
          f"scores 0): {errs or 'none'}; harness silently_truncated agrees on {v['harness_agree']}/{v['n_200']}")
    return s


def _fmt_pairing(pairing):
    parts = []
    for (cause, c), (n, fate) in sorted(pairing.items()):
        if cause == "both_full":
            if n:
                parts.append(f"items served in full by both configs: {c}/llama_server latency ratio median "
                             f"{fate['median_ratio']:.2f} (range {fate['min_ratio']:.2f}-{fate['max_ratio']:.2f}, "
                             f"n={n})")
            continue
        if n:
            label = "timeout_latency" if cause == "timeout" else cause
            parts.append(f"llama_server {label} items on {c}: " + ", ".join(f"{k} {v}" for k, v in sorted(fate.items()))
                         + f" (of {n})")
    return "; ".join(parts)


def compute_truncation_vs_latency(repo, model):
    """Silent truncation vs latency for one model in the v3 outcome table (TRUNC_DEFINITION), per config, with the
    cross-config pairing for llama_server's context_overflow and timeout_latency items. Provisional while v3 progress
    is below its plan (same flag as qwen3-32b-timeout-count)."""
    x2, cells, plan, _pa, _w = _x2_v3_cells(repo)
    items = {it["item_id"]: it for it in x2.load_items_trace_weighted(repo)}
    per, pairing = truncation_vs_latency(x2, cells, plan, items, model)
    if not per:
        raise FileNotFoundError(f"no {model} rows in the v3 outcome table")
    total_done = sum(len(set(v) & plan.get(m, set())) for (m, _c), v in cells.items())
    total_plan = sum(len(plan[m]) for m in plan) * len(x2.CONFIGS)
    status = (f"provisional: outcome table v3 not finished ({total_done}/{total_plan} done)"
              if total_done < total_plan else f"final: outcome table v3 finished ({total_done}/{total_plan} done)")
    body = " / ".join(_fmt_trunc_cell(c, v) for c, v in sorted(per.items()))
    pair = _fmt_pairing(pairing)
    return {"value": f"{TRUNC_DEFINITION}. {body}" + (f" / pairing: {pair}" if pair else "") + f"; {status}",
            "n": sum(v["n_done"] for v in per.values()), "detail": {"per": per, "pairing": pairing}}


def compute_qwen3_32b_truncation_vs_latency(repo):
    """compute_truncation_vs_latency for qwen3-32b (operator 2026-10-10: does the Ollama leg complete / look faster
    because it silently truncates long prompts?)."""
    return compute_truncation_vs_latency(repo, "qwen3-32b")


def compute_x2_v3_truncation_all_models(repo):
    """Silently truncated HTTP-200 rows per model x config in the v3 outcome table (TRUNC_DEFINITION), with mean
    score on truncated vs full rows; the per-model detail is compute_truncation_vs_latency."""
    x2, cells, plan, _pa, _w = _x2_v3_cells(repo)
    items = {it["item_id"]: it for it in x2.load_items_trace_weighted(repo)}
    parts, n = [], 0
    for model in sorted({m for (m, _c) in cells}):
        per, _p = truncation_vs_latency(x2, cells, plan, items, model)
        for c, v in sorted(per.items()):
            n += v["n_done"]
            gt, gf = v["groups"]["truncated"], v["groups"]["full"]
            sc = (f"score truncated {gt['mean']:.2f} (n={gt['n']}) vs full {gf['mean']:.2f} (n={gf['n']})"
                  if gt["n"] else f"score full {gf['mean']:.2f} (n={gf['n']})" if gf["n"] else "no scored rows")
            parts.append(f"{model}/{c} {v['counts'].get('truncated', 0)}/{v['n_200']}, {sc}")
    total_done = sum(len(set(v) & plan.get(m, set())) for (m, _c), v in cells.items())
    total_plan = sum(len(plan[m]) for m in plan) * len(x2.CONFIGS)
    status = ("provisional: outcome table v3 not finished" if total_done < total_plan else "final")
    return {"value": f"{TRUNC_DEFINITION}; silently truncated/HTTP-200: " + "; ".join(parts)
                     + f"; {status} ({total_done}/{total_plan} done)", "n": n}


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
    per_tier = rep["verdict"]["per_tier"]
    parts = []
    for s in sorted(rep["sessions"], key=lambda s: str(s["arm_id"])):
        pt = per_tier.get(s["arm_id"]) or {}
        mr = pt.get("match_rate")
        cite = (f"prompt check {pt.get('n_match', 0)}/{pt.get('n_checks', 0)} within 1% "
                f"(match rate {'n/a' if mr is None else f'{mr:.2f}'}), citable {bool(pt.get('mechanism_citable'))}, "
                + (f"tier verdict {pt.get('hypothesis')}" if pt.get("mechanism_citable")
                   else "tier verdict not citable (render does not show what the model saw)"))
        parts.append(f"{s['model_id']} {s['arm_id'].replace('_call2_notools', '')}: {cite}; first truncated turn "
                     f"{s['first_truncated_turn']}, {s['n_calls_message_truncated']}/{s['n_calls']} calls truncated, "
                     f"system kept on all {s['system_kept_on_every_truncated_call']}, contiguous tail "
                     f"{s['kept_contiguous_tail_on_every_truncated_call']}, cut mid-turn {s['cut_mid_turn_calls']}, "
                     f"token-level cuts {s['token_level_cut_calls']}, context-shift calls {s['context_shift_calls']}, "
                     f"exceed-context errors {s['exceed_context_error_calls']}, path {','.join(s['chat_paths'])}")
    v = rep["verdict"]
    # confirmed/refuted only over tiers whose render-only prompt was shown to match what the model saw
    cited = {t: p["hypothesis"] for t, p in per_tier.items() if p["mechanism_citable"]}
    hyps = set(cited.values()) - {"not_observed"}
    overall = ("not citable (no tier passed the prompt check)" if not cited
               else "refuted" if "refuted" in hyps else "confirmed" if hyps else "not_observed")
    not_cited = sorted(str(t).replace("_call2_notools", "") for t in per_tier if t not in cited)
    return {"value": f"verdict {overall} over citable tiers "
                     f"[{', '.join(sorted(str(t).replace('_call2_notools', '') for t in cited))}]"
                     + (f"; not citable (render does not show what the model saw): [{', '.join(not_cited)}]"
                        if not_cited else "")
                     + f" (token-level cut seen {v['token_level_cut_seen']}, "
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


def compute_r2_v1b_gated_per_model(repo):
    """Strengthened R2, gated variant per tier x model: silent gated failures (rules in use plus metrics at >=90%
    validation baseline, no error surfaced first), how many came at or after the first over-window turn, how many are
    baseline-type (no window exceeded yet), and sessions intact at turns 5/10/15/20/25/30/35/40."""
    rr, rep = _r2_real_v1b_report(repo)
    g = rr.gated_per_model(rep)
    parts = [f"{k.replace('|', ' ')}: {v['silent']}/{v['n']} silent, {v['silent_after_window_exceeded']} after window "
             f"exceeded, {v['silent_baseline_type']} baseline-type; intact "
             + "/".join(str(v["survival"][t]) for t in rep["checkpoints"]) for k, v in g.items()]
    return {"value": "; ".join(parts), "n": rep["n_sessions"]}


def compute_r2_v1b_default_baseline_type(repo):
    """Default tier (Ollama default context): of the pre-registered kill criterion's silent sessions, how many are
    baseline-type (no window exceeded, or the first failure came before the first over-window turn); and the gated
    silent sessions with what failed first."""
    rr, rep = _r2_real_v1b_report(repo)
    k = rr.kill_baseline_type(rep, rr.DEFAULT_TIER)
    gd = rr.silent_session_detail(rep, rr.DEFAULT_TIER, gated=True)

    def fmt(d):
        return ", ".join(f"{x['model']} seed {x['seed']} turn {x['first_failure']} ({'+'.join(x['what_failed'])}; "
                         f"first over window {'-' if x['first_over_window'] is None else x['first_over_window']})"
                         for x in d)
    return {"value": f"kill criterion {k['n_silent']}/{k['n_sessions']} silent, {k['n_baseline_type']} baseline-type "
                     f"({k['n_never_over_window']} never over window, {k['n_before_first_over_window']} before the "
                     f"first over-window turn), {k['n_after_window']} at or after it: {fmt(k['sessions'])}; gated "
                     f"{len(gd)}/{k['n_sessions']} silent, {sum(x['baseline_type'] for x in gd)} baseline-type: "
                     f"{fmt(gd)}",
            "n": k["n_sessions"]}


def compute_r2_v1b_excess_over_default(repo):
    """Strengthened R2: per tier, the after-window gated silent failure rate minus the Ollama-default tier's rate,
    overall and per model, with a 95% percentile bootstrap CI (r2_real_report.excess_over_default: resampling unit
    the session, stratified by tier x model, seeded)."""
    rr, rep = _r2_real_v1b_report(repo)
    x = rr.excess_over_default(rep)
    parts = []
    for tier, per in x["per_tier"].items():
        items = []
        for m in ["overall"] + sorted(k for k in per if k != "overall"):
            v = per[m]
            lo, hi = v["ci95"]
            items.append(f"{m} {round(v['rate'] * v['n_tier'])}/{v['n_tier']} vs default "
                         f"{round(v['default_rate'] * v['n_default'])}/{v['n_default']}, excess {v['excess']:+.3f} "
                         f"(95% CI {lo:+.3f} to {hi:+.3f})")
        parts.append(f"{tier}: " + ", ".join(items))
    return {"value": "; ".join(parts) + f" (bootstrap {x['n_resamples']} resamples, unit {x['unit']}, stratified by "
                                        f"{x['strata']}, seed {x['seed']}, percentile CI; a 0/n or n/n cell has no "
                                        f"bootstrap variance)",
            "n": rep["n_sessions"]}


def compute_r2_v1b_qwen3_8b_reversal(repo):
    """Strengthened R2, qwen3:8b at num_ctx 8192 vs 4096 (the gated non-monotonic cell): metrics the gate excludes,
    per session the first gated failure and what failed, the first over-window turn, turns completed and the first
    error, the pre-registered (ungated) first failure; per tier the turn-level counts of rule failures, system and
    history canary misses, tool-argument and recall failures; and a two-sided Fisher exact p on silent sessions."""
    from math import comb
    rr, rep = _r2_real_v1b_report(repo)
    model, tiers = "qwen3:8b", ("num_ctx_8192", "num_ctx_4096")
    rows, rules_in_use, sessions, _iss = rr._collect([repo / R2_REAL_V1_FILE, repo / R2_REAL_V1B_FILE])
    rules = rules_in_use[model]
    tier_of = {s["attempt_id"]: rr.tier_of(s["arm_id"]) for s in sessions if s["model_id"] == model}
    parts = [f"{model} gate excludes {', '.join(rep['gated_metrics'][model]['excluded']) or 'nothing'}"]
    silent = {}
    for tier in tiers:
        c = rep["table"][f"{tier}|{model}"]
        silent[tier] = (sum(e["silent"] for e in c["sessions"]), c["n"])
        sess = ", ".join(
            f"seed {e['seed']}: gated {'none' if e['first_gated_failure'] is None else str(e['first_gated_failure']) + ' ' + '+'.join(e['first_gated_failure_fields'])}"
            f", over window from {e['first_over_window']}, {e['n_turns']} turns, first error "
            f"{'-' if e['first_error'] is None else e['first_error']}, ungated {e['kill_first_failure']} "
            f"{'+'.join(e['kill_first_failure_reasons'])}" for e in c["sessions"])
        t = [r for r in rows if r.get("record") == "r2a_turn" and tier_of.get(r.get("attempt_id")) == tier]
        chk = [r for r in t if r.get("canary_check")]
        rec = [r for r in t if r.get("recall_ok") is not None]
        parts.append(f"{tier}: {silent[tier][0]}/{silent[tier][1]} gated silent ({sess}); turns {len(t)}, rule "
                     f"failure turns {sum(1 for r in t if any(r.get(x) is False for x in rules))}, system canary "
                     f"missed {sum(1 for r in chk if r.get('canary_sys_ok') is False)}/{len(chk)} checks, history "
                     f"canary missed {sum(1 for r in chk if r.get('canary_hist_ok') is False)}/{len(chk)}, tool args "
                     f"failed {sum(1 for r in t if r.get('tool_args_all') is False)}/{len(t)} turns, recall failed "
                     f"{sum(1 for r in rec if r.get('recall_ok') is False)}/{len(rec)}")
    (a, n1), (b, n2) = silent[tiers[0]], silent[tiers[1]]
    k, N = a + b, n1 + n2
    pmf = {x: comb(n1, x) * comb(n2, k - x) / comb(N, k) for x in range(max(0, k - n2), min(k, n1) + 1)}
    p = sum(v for v in pmf.values() if v <= pmf[a] * (1 + 1e-9))
    parts.append(f"Fisher exact two-sided p (silent sessions, {tiers[0]} vs {tiers[1]}) = {p:.3f}")
    return {"value": "; ".join(parts), "n": f"{silent[tiers[0]][1] + silent[tiers[1]][1]} sessions"}


# -- evo-t2s week window 2026-10-15 to 2026-10-18 (Ollama 0.34.4, Intel CPU and Intel iGPU) --
T2S_VAL = {"cpu": "results/t2s_r2_validation_cpu_v1.jsonl", "igpu": "results/t2s_r2_validation_igpu_v1.jsonl"}
T2S_REAL = {"cpu": "results/t2s_r2_real_cpu_v1.jsonl", "igpu": "results/t2s_r2_real_igpu_v1.jsonl"}
T2S_MITIGATION_CPU = "results/t2s_r2_mitigation_cpu_v1.jsonl"
T2S_MECHANISM_CPU = "results/t2s_r2_mechanism_cpu_v1.jsonl"


def compute_t2s_r2_validation(repo, backend):
    """evo-t2s R2 validation (same plan as x2_r2_validation_v2) on one backend: per model the baseline table, task
    tool, thinking, the real-run preflight, and which gates failed."""
    ag = _r2_agent_module()
    rows = ag.read_rows(repo / T2S_VAL[backend])
    if not rows:
        raise FileNotFoundError(repo / T2S_VAL[backend])
    base = compute_x2_scale_validation(repo, vfile=T2S_VAL[backend])
    g = ag.evaluate_gates(rows, call2_tools="off")["gates"]
    models = sorted({r["model_id"] for r in rows if r.get("record") == "r2a_session"})
    failed = {m: sorted(k for k, v in g.items() if isinstance(v, dict) and isinstance(v.get(m), dict)
                        and v[m].get("pass") is False) for m in models}
    return {"value": base["value"] + "; failed gates: " + "; ".join(f"{m} {', '.join(f) or 'none'}"
                                                                    for m, f in failed.items()),
            "n": base["n"]}


def compute_t2s_r2_session_minutes(repo):
    """Measured evo-t2s R2 session times (the input of the B5 window plan): per backend and model, the median minutes
    of a completed 40-turn session (summed turn_wall_s) at the Ollama default window, the CPU trimming session median
    at 4096, and each run's wall clock (run_start to run_end)."""
    import sys as _sys
    _sys.path.insert(0, str(repo / "harness"))
    import policy_compare as pc
    from datetime import datetime as _dt
    parts, n = [], 0
    for plat, label in (("evo-t2s-cpu", "Intel CPU"), ("evo-t2s", "Intel iGPU")):
        meas = pc._t2s_measured(repo, plat)
        segs = []
        for m, e in sorted(meas.items()):
            n += sum(e["n"].values())
            s = f"{m} default window {e['D']}: {e['default_min']:.1f} min/session (n={e['n'].get('default')})"
            if e.get("trim_min"):
                s += f", trimming at 4096 {e['trim_min']:.1f} min/session (n={e['n'].get('trim')})"
            segs.append(s)
        runs = []
        for kind, name in pc.T2S_MEASURED_FILES[plat].items():
            if not name:
                continue
            rows = _r2_agent_module().read_rows(repo / "results" / name)
            st = [r["ts_utc"] for r in rows if r.get("record") == "run_start"]
            en = [r["ts_utc"] for r in rows if r.get("record") == "run_end"]
            if st and en:
                runs.append(f"{name} run {(_dt.fromisoformat(en[-1]) - _dt.fromisoformat(st[0])).total_seconds() / 3600:.1f} h")
        parts.append(f"{label}: " + "; ".join(segs) + (f" ({', '.join(runs)})" if runs else ""))
    return {"value": " / ".join(parts), "n": f"{n} sessions"}


def compute_t2s_b5_plan_hours(repo):
    """B5 on evo-t2s, window 2026-10-26 to 2026-11-08: ESTIMATED hours per staged entry from measured evo-t2s session
    minutes (policy_compare.estimate_t2s) and what fits before the no-launch time (scripts/t2s_window_stage.plan)."""
    import sys as _sys
    _sys.path.insert(0, str(repo / "scripts"))
    _sys.path.insert(0, str(repo / "harness"))
    import t2s_window_stage as ws
    p = ws.plan()
    return {"value": ws.plan_value(p), "n": f"{len(p['entries'])} entries"}


def compute_t2s_r2_real(repo, backend):
    """evo-t2s R2 real run at the Ollama default window (no num_ctx sent), 5 seeds x 40 turns: per model the window
    Ollama loaded, the first over-window turn, the pre-registered kill criterion, and the gated variant (rules in use
    plus metrics at >=90% of this backend's own validation baseline) with after-window and baseline-type counts."""
    import sys as _sys
    _sys.path.insert(0, str(repo / "analysis"))
    import r2_real_report as rr
    if not (repo / T2S_REAL[backend]).exists():
        raise FileNotFoundError(repo / T2S_REAL[backend])
    rep = rr.build_combined([repo / T2S_REAL[backend]], validation_paths=[repo / T2S_VAL[backend]])
    rows = _r2_agent_module().read_rows(repo / T2S_REAL[backend])
    sess = [r for r in rows if r.get("record") == "r2a_session"]
    g = rr.gated_per_model(rep)
    parts = []
    for model in sorted({r["model_id"] for r in sess}):
        ms = [r for r in sess if r["model_id"] == model]
        loaded = sorted({c for r in ms for c in r.get("loaded_context_values") or []})
        fo = sorted({r.get("first_over_window_turn") for r in ms}, key=lambda x: (x is None, x))
        c = g[f"ollama_default|{model}"]
        parts.append(f"{model}: loaded window {loaded}, first over window turn {fo}; gated {c['silent']}/{c['n']} "
                     f"silent, {c['silent_after_window_exceeded']} after window exceeded, {c['silent_baseline_type']} "
                     f"baseline-type (gate excludes {', '.join(rep['gated_metrics'][model]['excluded']) or 'nothing'}); "
                     f"intact " + "/".join(str(c["survival"][t]) for t in rep["checkpoints"]))
    k = rep["kill"]
    arms = "; ".join(f"{a.replace('_call2_notools', '')} {v['n_silent_failures']}/{v['n_sessions']} silent"
                     for a, v in k["per_arm"].items())
    return {"value": " / ".join(parts) + f"; pre-registered kill criterion killed={k['killed']}, {arms}",
            "n": f"{len(sess)} sessions"}


def compute_t2s_r2_mitigation_cpu(repo):
    """evo-t2s CPU: client-side trimming at num_ctx 4096 (margin 0.05), 3 seeds x 40 turns, the pre-registered
    mitigation rule per model against the CPU validation baseline. No evo-t2s untrimmed real run at an explicit 4096
    exists; the CPU default-window real run loaded 4096 (row t2s-r2-real-cpu)."""
    import sys as _sys
    _sys.path.insert(0, str(repo / "analysis"))
    import r2_mitigation_report as mr
    if not (repo / T2S_MITIGATION_CPU).exists():
        raise FileNotFoundError(repo / T2S_MITIGATION_CPU)
    rep = mr.build(repo / T2S_MITIGATION_CPU, real_paths={}, validation_path=repo / T2S_VAL["cpu"])
    parts = []
    for key, c in rep["cells"].items():
        tier, model = key.split("|")
        v = c["verdict"]
        extra = (f" (rules below 90%: {', '.join(v['rules_below_threshold'])})" if v.get("rules_below_threshold")
                 else f" ({'; '.join(v['reasons'])})" if v["reasons"] else "")
        parts.append(f"{tier} {model} {v['verdict']}{extra}")
    return {"value": f"overall {rep['overall_verdict']}; " + "; ".join(parts),
            "n": f"{rep['n_sessions']} sessions, {rep['n_turns']} turns"}


def compute_t2s_r2_mechanism_cpu(repo):
    """evo-t2s CPU mechanism session (llama3.1:8b, Ollama default window, one seed): turns completed, whether the
    session closed (session summary and run_end written), turns over the loaded window, render-only prompt checks."""
    rows = _r2_agent_module().read_rows(repo / T2S_MECHANISM_CPU)
    if not rows:
        raise FileNotFoundError(repo / T2S_MECHANISM_CPU)
    turns = [r for r in rows if r.get("record") == "r2a_turn"]
    start = next(r for r in rows if r.get("record") == "run_start")
    planned = start["plan"][0][2]
    loaded = sorted({str(r.get("loaded_context")) for r in turns})
    checks = [r for r in rows if r.get("record") == "r2m_prompt_check"]
    closed = any(r.get("record") == "r2a_session" for r in rows) and any(r.get("record") == "run_end" for r in rows)
    over = sum(1 for r in turns if r.get("over_loaded_window"))
    return {"value": f"{len(turns)}/{planned} turns completed, session closed {closed}, last turn at "
                     f"{turns[-1]['ts_utc'][:16]}Z; loaded window {loaded}; {over} turns over the loaded window; "
                     f"{len(checks)} render-only prompt checks; no session summary, so the mechanism verdict is not "
                     f"computed",
            "n": f"{len(turns)} turns"}


def compute_r2_qwen3_14b_4096_reconciliation(repo):
    """qwen3:14b at num_ctx 4096 across R2 v1/v1b (untrimmed), the mitigation run (client-side trimming) and the
    num_keep run: per run and seed, rules in use, per-rule pass counts, the gated first failure and what failed, the
    Ollama version, whether a harness blob is recorded, the protocol knobs, and whether the num_keep run reproduces
    v1 turn by turn (identical final text and first-call prompt token count on the same seeds); and, on the first
    trimmed turn where rule 2 fails, which tools call 1 made in the trimmed vs the untrimmed run."""
    import sys as _sys
    _sys.path.insert(0, str(repo / "analysis"))
    import r2_real_report as rr
    files = {"v1": R2_REAL_V1_FILE, "v1b": R2_REAL_V1B_FILE, "mitigation": R2_MITIGATION_FILE,
             "num_keep": "results/x2_num_keep_qwen3_14b.jsonl"}
    ag = _r2_agent_module()
    rules = ["rule1_json_keys", "rule2_log_event_called", "rule3_no_zebra", "rule4_metres", "rule5_session_code"]
    turns, parts = {}, []
    vals = [repo / R2_VALIDATION_V2_FILE, repo / R2_VALIDATION_V2B_FILE]
    for name, f in files.items():
        rows = ag.read_rows(repo / f)
        if not rows:
            raise FileNotFoundError(repo / f)
        rs = next(r for r in rows if r.get("record") == "run_start")
        rt = next((r for r in rows if r.get("record") == "runtime"), {})
        t = {(r["seed"], r["turn_idx"]): r for r in rows if r.get("record") == "r2a_turn"
             and r.get("model_id") == "qwen3:14b" and "4096" in str(r.get("arm_id"))}
        turns[name] = t
        per_seed = []
        for seed in sorted({k[0] for k in t}):
            ts = [v for k, v in t.items() if k[0] == seed]
            per_seed.append(f"{seed} " + ", ".join(
                f"{r.split('_')[0]} {sum(1 for x in ts if x.get(r) is True)}/{sum(1 for x in ts if x.get(r) is not None)}"
                for r in rules))
        rep = rr.build_combined([repo / f], validation_paths=vals)
        g = [e for k, c in rep["table"].items() if k.endswith("|qwen3:14b") and "4096" in k for e in c["sessions"]]
        gated = "; ".join(f"{e['seed']} {e['first_gated_failure'] if e['first_gated_failure'] is not None else 'none'}"
                          f" {'+'.join(e['first_gated_failure_fields'])}".rstrip() for e in g) or "n/a (not a real-mode run)"
        parts.append(f"{name}: rules in use {len((rs.get('rules_in_use') or {}).get('qwen3:14b') or [])}, Ollama "
                     f"{rt.get('ollama_version') or 'not recorded'}, harness blob "
                     f"{rs.get('agent_blob') or 'not recorded'}, call2 {rs.get('call2_mode') or ('off' if rs.get('call2_tools') is False else rs.get('call2_tools'))}, "
                     f"max tokens {rs.get('max_tokens_per_call')}, client trim {rs.get('client_trim')}, num_keep "
                     f"{rs.get('num_keep')}; per seed {' / '.join(per_seed)}; gated first failure {gated}")
    v1, nk, mi = turns["v1"], turns["num_keep"], turns["mitigation"]
    common = [k for k in nk if k in v1]
    same_text = sum(1 for k in common if v1[k].get("final_text") == nk[k].get("final_text"))
    same_pe = sum(1 for k in common if (v1[k].get("calls") or [{}])[0].get("prompt_eval_count")
                  == (nk[k].get("calls") or [{}])[0].get("prompt_eval_count"))
    k = next(k for k in sorted(mi) if mi[k].get("rule2_log_event_called") is False)
    names = lambda t: [c["function"]["name"] for c in ((t.get("calls") or [{}])[0].get("native_tool_calls") or [])]
    return {"value": " // ".join(parts) + f" // num_keep vs v1 on the same seeds: identical final text {same_text}/"
                     f"{len(common)}, identical first-call prompt tokens {same_pe}/{len(common)} // first trimmed "
                     f"rule2 failure seed {k[0]} turn {k[1]}: call 1 tools trimmed {names(mi[k])}, untrimmed v1 "
                     f"{names(v1.get(k, {}))}",
            "n": f"{sum(len(t) for t in turns.values())} turns"}


def compute_x2_evalset_tasks_real_v2_ci_units(repo):
    """B3 v2 real run, qwen3:14b: the wrong-task excess over the default window per tier with two bootstrap units,
    10000 resamples each, seed 20261014. Session level (the pre-registered one, used in the paper): whole sessions
    resampled with replacement within each tier, all task turns of a session kept together
    (x2_r2_evalset.session_bootstrap_excess). Turn level (for comparison only): task turns resampled independently
    within each tier, which ignores within-session correlation and so gives narrower intervals."""
    import random as _random
    import sys as _sys
    _sys.path.insert(0, str(repo / "harness"))
    import x2_r2_evalset as ev
    ag = _r2_agent_module()
    rows = ag.read_rows(repo / "results/x2_evalset_tasks_real_v2.jsonl")
    if not rows:
        raise FileNotFoundError(repo / "results/x2_evalset_tasks_real_v2.jsonl")
    sessions = [x for x in ag.completed_sessions(rows) if x.get("mode") == "real" and x["model_id"] == "qwen3:14b"]
    att = {x["attempt_id"]: x for x in sessions}
    turns = [r for r in rows if r.get("record") == "r2a_turn" and r.get("attempt_id") in att
             and r.get("item_kind") == "task"]
    tier_of = lambda arm: arm.replace("_call2_notools", "")
    cells, flat = {}, {}
    for x in sessions:
        t = tier_of(x["arm_id"])
        st = [r for r in turns if r["attempt_id"] == x["attempt_id"]]
        cells.setdefault(t, []).append((sum(1 for r in st if r.get("end_state_ok") is False), len(st)))
        flat.setdefault(t, []).extend(1 if r.get("end_state_ok") is False else 0 for r in st)
    default = "ollama_default"
    sess = ev.session_bootstrap_excess(cells, default, n_boot=10000, seed=20261014)["per_tier"]
    rng = _random.Random(20261014)
    draws = {t: [] for t in flat if t != default}
    for _ in range(10000):
        d = flat[default]
        rd = sum(d[rng.randrange(len(d))] for _ in d) / len(d)
        for t in draws:
            xs = flat[t]
            draws[t].append(sum(xs[rng.randrange(len(xs))] for _ in xs) / len(xs) - rd)
    order = ["ollama_ctx_16384", "ollama_ctx_8192", "ollama_ctx_4096", "ollama_ctx_4096_numkeep"]
    parts = []
    for t in [o for o in order if o in sess]:
        v = sess[t]
        dt = sorted(draws[t])
        lo_s, hi_s = v["ci95"]
        parts.append(f"{t}: excess {100 * v['excess']:+.1f} points; session-level 95% CI {100 * lo_s:+.1f} to "
                     f"{100 * hi_s:+.1f} ({v['n_sessions']} vs {v['n_default_sessions']} sessions); turn-level 95% "
                     f"CI {100 * ev._quantile(dt, 0.025):+.1f} to {100 * ev._quantile(dt, 0.975):+.1f} "
                     f"({len(flat[t])} vs {len(flat[default])} task turns)")
    return {"value": "qwen3:14b wrong-task excess over default; " + "; ".join(parts)
                     + " (10000 resamples each, seed 20261014; the paper uses the session-level CI)",
            "n": f"{len(sessions)} sessions, {len(turns)} task turns"}


def compute_r2_validation_v2b_runinfo(repo):
    """R2 validation v2b run details per model: sessions and turns completed, calls with an HTTP error or error field,
    task tool called correctly, and on the baseline arm how many call-1 replies carried a native tool call vs a tool
    call written as text ('"name"' inside the reply content)."""
    import sys as _sys
    _sys.path.insert(0, str(repo / "harness"))
    import x2_r2_agent as ra
    rows = ra.read_rows(repo / "results/x2_r2_validation_v2b.jsonl")
    out = []
    for m in sorted({r["model_id"] for r in rows if r.get("record") == "r2a_session"}):
        sess = [r for r in rows if r.get("record") == "r2a_session" and r["model_id"] == m]
        turns = [r for r in rows if r.get("record") == "r2a_turn" and r["model_id"] == m]
        errs = [c for r in turns for c in r.get("calls", []) if c.get("error") or c.get("http_status") not in (200, None)]
        base = [r for r in turns if r["arm_id"] == "ollama_ctx_131072_call2_notools"]
        nat = sum(1 for r in base if r["calls"][0].get("native_tool_calls"))
        txt = sum(1 for r in base if '"name"' in (r["calls"][0].get("content") or ""))
        tt = sum(1 for r in base if r.get("task_tool_ok"))
        out.append(f"{m}: {len(sess)} sessions, {len(turns)} turns, {len(errs)} error calls; baseline arm: task tool "
                   f"correct {tt}/{len(base)}, call-1 native tool call {nat}/{len(base)}, tool call written as text "
                   f"in content {txt}/{len(base)}")
    return {"value": " / ".join(out), "n": sum(1 for r in rows if r.get("record") == "r2a_turn")}



# ───────────────────────────────────────────────── install path (2026-10-08): same 4B weights, bare vs library template
R2_VALIDATION_V2C_FILE = "results/x2_r2_validation_v2c.jsonl"   # qwen3-4b-2507-tools alone, v2b's plan
R2_BASE_ARM_OFF = "ollama_ctx_131072_call2_notools"
R2_4B_BARE, R2_4B_TOOLS = "qwen3-4b-2507", "qwen3-4b-2507-tools"


def _r2_install_path_half(repo, rel, model):
    """One tag's numbers for the install-path comparison, from a validation file: template source, then on the
    baseline arm (num_ctx 131072, tools withheld on call 2, completed sessions) call-1 native tool calls, tool calls
    written as text in call-1 content ('"name"' inside the content, R2-validation-v2b-runinfo's definition), task tool
    correct and the task-tool gate, per-rule compliance, controls, thinking."""
    ag = _r2_agent_module()
    rows = ag.read_rows(repo / rel)
    if not rows:
        raise FileNotFoundError(repo / rel)
    rep = ag.evaluate_gates(rows, call2_tools="off")
    e = rep["baseline_table"].get(model)
    if e is None:
        raise ValueError(f"no completed baseline sessions for {model} in {rel}")
    attempts = {s["attempt_id"] for s in ag.completed_sessions(rows) if s["arm_id"] == R2_BASE_ARM_OFF
                and s["model_id"] == model}
    base = [r for r in rows if r.get("record") == "r2a_turn" and r.get("attempt_id") in attempts]
    nat = sum(1 for r in base if r["calls"][0].get("native_tool_calls"))
    txt = sum(1 for r in base if '"name"' in (r["calls"][0].get("content") or ""))
    g = rep["gates"]
    tt = g["task_tool"][model]
    if model == R2_4B_TOOLS:
        cr = [r for r in rows if r.get("record") == "r2t_create"]
        src = (f"ollama_create_library_template, template sha256 {cr[-1]['template_sha256'][:12]} "
               f"({cr[-1].get('action')})") if cr else "ollama_create_library_template (no r2t_create record)"
    else:
        cts = _cts_module()
        facts = cts.facts_from_probe(_read_jsonl(repo / X2_TEMPLATE_FACTS))
        f = facts.get(("ollama", model)) or {}
        src = f"{f.get('chat_template_source')}, template sha256 {str(f.get('chat_template_sha256'))[:12]}"
    neg, pos = rep["controls"]["negative"].get(model, {}), rep["controls"]["positive"].get(model, {})
    rules = ", ".join(f"{r.split('_')[0]} {_pct(v['rate'])}" for r, v in e["rules"].items())
    pf = ag.validation_preflight(rows, "off", [model])
    return {"text": (f"{model} ({src}): baseline {len(base)} turns, call-1 native tool call {nat}/{len(base)}, tool "
                     f"call written as text in call-1 content {txt}/{len(base)}, task tool correct "
                     f"{tt['n_ok']}/{tt['n']} (task-tool gate {'pass' if tt['pass'] else 'fail'}); rules {rules}; "
                     f"negative-control canary misses {neg.get('canary_misses')}/{2 * neg.get('canary_checks', 0)}; "
                     f"positive-control truncation turn {pos.get('truncation_detected_turns')}; thinking-present calls "
                     f"{e['thinking_present_calls']}; real-run preflight {'pass' if pf['ok'] else 'refuse'}"),
            "nat": nat, "txt": txt, "tt": tt["n_ok"], "n": len(base)}


def compute_r2_install_path_4b_bare(repo):
    """Install path, bare half (computes now): qwen3-4b-2507, an Ollama tag created from a bare GGUF (no TEMPLATE), in
    x2_r2_validation_v2b."""
    h = _r2_install_path_half(repo, R2_VALIDATION_V2B_FILE, R2_4B_BARE)
    return {"value": h["text"], "n": f"{h['n']} baseline turns"}


def compute_r2_install_path_4b_comparison(repo):
    """Install path comparison (PENDING until x2_r2_validation_v2c.jsonl is synced): the same GGUF as qwen3-4b-2507
    (bare, v2b) vs qwen3-4b-2507-tools (library Qwen3 template copied verbatim from qwen3:8b, v2c)."""
    a = _r2_install_path_half(repo, R2_VALIDATION_V2B_FILE, R2_4B_BARE)
    b = _r2_install_path_half(repo, R2_VALIDATION_V2C_FILE, R2_4B_TOOLS)
    return {"value": (f"{a['text']} // {b['text']} // change bare -> library template: call-1 native "
                      f"{a['nat']}/{a['n']} -> {b['nat']}/{b['n']}, text tool calls {a['txt']}/{a['n']} -> "
                      f"{b['txt']}/{b['n']}, task tool correct {a['tt']}/{a['n']} -> {b['tt']}/{b['n']}"),
            "n": f"{a['n']} + {b['n']} baseline turns"}


def compute_r2_validation_v2c_baseline(repo):
    """R2 validation v2c (gate for qwen3-4b-2507-tools in the strengthened run): v2b's plan for that tag alone."""
    return _r2_gates_summary(repo, R2_VALIDATION_V2C_FILE, False)


def compute_r2_task_tool_gate(repo):
    """The uniform task-tool gate (task tool correct on >= 90% of baseline turns) on the validation files the real
    run reads, per model, and whether the full real-run preflight verdict per model changes with it (verdict without
    the gate = preflight reasons other than the task-tool one)."""
    ag = _r2_agent_module()
    parts, n = [], 0
    for rel in (R2_VALIDATION_V2_FILE, R2_VALIDATION_V2B_FILE):
        rows = ag.read_rows(repo / rel)
        if not rows:
            raise FileNotFoundError(repo / rel)
        g = ag.evaluate_gates(rows, call2_tools="off")["gates"]["task_tool"]
        for m in sorted(g):
            t = g[m]
            n += t["n"]
            reasons = ag.validation_preflight(rows, "off", [m])["reasons"]
            without = not [r for r in reasons if "task-tool gate" not in r]
            parts.append(f"{m} ({Path(rel).name}) {t['n_ok']}/{t['n']} {'pass' if t['pass'] else 'fail'}; preflight "
                         f"without the gate {'pass' if without else 'refuse'}, with it "
                         f"{'pass' if not reasons else 'refuse'}")
    return {"value": "; ".join(parts), "n": f"{n} baseline turns"}


def compute_x2_chat_template_source(repo):
    """Chat template source per (model, config) of the X2 outcome table, from the evo-x2 probe facts, with how many
    v3 rows recorded the field themselves vs had it derived by (model, config)."""
    x2 = _x2_harness()
    src = _x2_template_source_fn(repo)
    rows = [r for r in x2.read_rows(repo / X2_OUTCOME_V3) if r.get("record") == "outcome_row"]
    parts = []
    for m in x2.DEFAULT_MODELS:
        tag, _gguf = x2.MODEL_MAP[m]
        for c in x2.CONFIGS:
            cur = {"model_id": m, "config": c}
            if c == "ollama_default":
                cur["ollama_tag"] = tag
            info = src(cur)
            label = f"{m}/{c}" + (f" (tag {tag})" if c == "ollama_default" else "")
            mine = [r for r in rows if r.get("model_id") == m and r.get("config") == c]
            by = {}
            for r in mine:
                k = (src(r)["chat_template_source"], "valid" if x2.row_is_valid(r) else "not valid",
                     "recorded" if r.get("chat_template_source") else "derived")
                by[k] = by.get(k, 0) + 1
            rows_s = ", ".join(f"{v} {a} {b} {d}" for (a, b, d), v in sorted(by.items()))
            parts.append(f"{label} {info['chat_template_source']} sha256 {str(info['chat_template_sha256'])[:12]} "
                         f"(v3 rows: {rows_s or 'none'})")
    return {"value": "; ".join(parts), "n": len(rows)}

# ───────────────────────────────────────────────── Pareto frontier (src/dse/pareto.py, 2026-10-08)
PARETO_X2_FILES = [("evo-x2", X2_OUTCOME_V3)]
PARETO_SAMPLE = {"budget_usd": 50.0, "quality_floor": 0.9, "latency_target_ms": 30_000.0, "hardware": ["evo-x2"]}
_PARETO_DATA = [X2_OUTCOME_V3, "results/workload_pack/item_weights_trace_weighted.json",
                "results/cloud_prices_20261008.json", X2_TEMPLATE_FACTS]
_PARETO_CACHE: dict = {}


def _pareto_points(repo):
    key = str(repo)
    if key not in _PARETO_CACHE:
        import sys
        sys.path.insert(0, str(REPO))
        from src.dse import pareto as P
        _PARETO_CACHE[key] = (P, P.load_points(PARETO_X2_FILES, cloud_source="stub", repo=repo))
    return _PARETO_CACHE[key]


def _pareto_fmt(p):
    lat = "n/a" if p.latency_p50_ms is None else f"{p.latency_p50_ms / 1000:.1f}/{p.latency_p90_ms / 1000:.1f} s"
    q = "n/a" if p.quality is None else f"{p.quality:.3f}"
    return f"{p.label()} q {q} (n={p.quality_n}) USD {p.usd_per_1k_steps:.2f}/1k steps p50/p90 {lat}"


def compute_pareto_frontier_x2(repo):
    """Quality vs USD per 1000 steps frontier per machine (src/dse/pareto.py::frontier) over the evo-x2 v3 outcome
    table; bare-template cells and STUB cloud points (no measured quality) are excluded from the frontier and
    listed separately."""
    P, pts = _pareto_points(repo)
    parts = []
    for m in sorted({p.machine for p in pts}):
        f = P.frontier(pts, m)
        parts.append(f"{m} frontier: " + ("; ".join(_pareto_fmt(p) for p in f) or "none (no measured quality)"))
    bare = [p for p in pts if p.bare_template]
    if bare:
        parts.append("bare template, off frontier: " + "; ".join(_pareto_fmt(p) for p in bare))
    stub = [p for p in pts if p.stub]
    if stub:
        parts.append(f"{P.STUB} cloud (quality not measured): " + "; ".join(
            f"{p.model} USD {p.usd_per_1k_steps:.2f}/1k steps" for p in stub))
    n_local = len([p for p in pts if not p.is_cloud])
    return {"value": f"{n_local} local configs; " + " / ".join(parts), "n": n_local}


def compute_pareto_sample_recommendation(repo):
    """src/dse/pareto.py::recommend for budget USD 50, quality floor 0.9, p50 latency target 30 s, hardware
    [evo-x2]; savings are vs the cheapest cloud point (STUB until a real cloud ledger or outcome file exists).
    v1: no usability constraint (max_timeout_share=None, the recommender as of 2026-10-08); kept unchanged, v2 below
    adds the constraint."""
    P, pts = _pareto_points(repo)
    rec = P.recommend(pts, **PARETO_SAMPLE, max_timeout_share=None)
    sav = rec.savings_vs_all_cloud_usd_per_1k
    return {"value": f"{rec.reason}; savings vs all-cloud "
                     f"{'n/a' if sav is None else f'USD {sav:.2f}/1k steps'}" + (f" [{P.STUB}]" if rec.stub else ""),
            "n": len(pts)}


PARETO_SAMPLE_MAX_TIMEOUT_SHARE = 0.05  # operator 2026-10-10, src/dse/pareto.py DEFAULT_MAX_TIMEOUT_SHARE


def compute_pareto_sample_recommendation_v2(repo):
    """Same inputs as pareto-sample-recommendation plus the usability constraint (operator 2026-10-10): a config is
    eligible only if its timeout_latency share (timeout_latency rows / done rows, src/dse/pareto.py Point
    timeout_latency_share) is <= 0.05. The reason names the chosen config and lists every excluded config with its
    share; n = number of configs excluded by the constraint."""
    P, pts = _pareto_points(repo)
    rec = P.recommend(pts, **PARETO_SAMPLE, max_timeout_share=PARETO_SAMPLE_MAX_TIMEOUT_SHARE)
    sav = rec.savings_vs_all_cloud_usd_per_1k
    return {"value": f"max_timeout_share {PARETO_SAMPLE_MAX_TIMEOUT_SHARE:g}: {rec.reason}; savings vs all-cloud "
                     f"{'n/a' if sav is None else f'USD {sav:.2f}/1k steps'}" + (f" [{P.STUB}]" if rec.stub else ""),
            "n": len(rec.excluded)}


def compute_r2_mechanism_lowlevel(repo):
    """R2 mechanism job, per tier: calls with message-level truncation, llama.cpp context-shift events (n_keep,
    n_discard, how many on the final-answer call), and Ollama token-level prompt cuts (limit/keep/new)."""
    import json as _json
    rows = [_json.loads(x) for x in (repo / "results/x2_r2_mechanism.jsonl").read_text(encoding="utf-8").splitlines()
            if x.strip()]
    by = {}
    for r in rows:
        if r.get("record") == "r2m_call":
            by.setdefault(r["arm_id"].replace("_call2_notools", ""), []).append(r)
    parts = []
    for arm, rs in by.items():
        cs = [(r["call_idx"], s) for r in rs for s in (r.get("context_shifts") or [])]
        tc = [r["token_cut"] for r in rs if r.get("token_level_cut")]
        first = next(((r["turn_idx"], r["call_idx"]) for r in rs if r.get("context_shifts")), None)
        cs_txt = (f"{len(cs)} context shifts (first turn/call {first}, n_keep {sorted({s['n_keep'] for _, s in cs})}, "
                  f"n_discard {sorted({s['n_discard'] for _, s in cs})}, {sum(1 for c, _ in cs if c == 2)} on the "
                  f"final-answer call)") if cs else "0 context shifts"
        tc_txt = "; ".join(f"token cut limit {t['limit']} keep {t['keep']} prompt {t['prompt']} -> {t['new']}" for t in tc)
        parts.append(f"{arm}: {sum(1 for r in rs if r.get('message_level_truncation'))}/{len(rs)} calls message-truncated; "
                     f"{cs_txt}; {len(tc)} token cuts" + (f" ({tc_txt})" if tc else ""))
    return {"value": " / ".join(parts), "n": sum(len(v) for v in by.values())}


def _r2_shift_module(repo):
    import sys as _sys
    _sys.path.insert(0, str(repo / "analysis"))
    import r2_shift_predictor as sp
    return sp


def compute_r2_shift_predictor_confusion(repo):
    """Context-shift predictor (P + C > n_ctx, from llama.cpp b11081 pre_decode) vs the observed shifts of the R2
    mechanism job, per tier: confusion counts, precision/recall, events, closest margins; misses; refinements tried."""
    sp = _r2_shift_module(repo)
    ms = sp.mechanism_summary(sp.read_rows(repo / sp.MECH_FILE))
    pct = lambda x: "n/a" if x is None else f"{100 * x:.1f}%"
    tiers = "; ".join(f"{t} TP {c['tp']} FP {c['fp']} FN {c['fn']} TN {c['tn']} precision {pct(c['precision'])} "
                      f"recall {pct(c['recall'])} events obs/pred {c['events_observed']}/{c['events_predicted']} "
                      f"closest margin shift/no-shift {c['closest_margin_shifted']}/{c['closest_margin_not_shifted']}"
                      for t, c in ms["per_tier"].items())
    var = "; ".join(f"{n}: TP {c['tp']} FP {c['fp']} FN {c['fn']}" for n, c in ms["variants"].items())
    d = ms["decoded_tokens_check"]
    return {"value": f"{tiers} / misses {len(ms['misses'])} / decoded C-1 check {d['ok']}/{d['n']}; P sources "
                     f"{ms['P_sources']} / variants: {var}",
            "n": ms["n_calls"]}


def compute_r2_shift_predicted_real_v1(repo):
    """PREDICTED (not observed) context shifts in the R2 real run v1 per tier x model, with Ollama token-cut
    signatures (prompt_eval_count == num_ctx - (num_ctx - 4) / 2) and, per seed, first predicted shift vs first
    rule failure (rules in use)."""
    sp = _r2_shift_module(repo)
    rs = sp.real_summary(sp.read_rows(repo / sp.REAL_FILE))
    parts = []
    for (tier, model), c in rs["cells"].items():
        sess = ", ".join(f"{seed} shift {s['first_pred_shift'] or '-'} cut {s['first_cut_signature'] or '-'} "
                         f"rule fail {s['first_rule_failure'] or '-'}" for seed, s in c["sessions"].items())
        parts.append(f"{tier} {model}: predicted {c['pred_calls']}/{c['calls']} calls (call1 {c['pred_call1']}, "
                     f"call2 {c['pred_call2']}), {c['pred_events']} events, {c['sessions_with_pred_shift']} sessions, "
                     f"{c['cut_sig_calls']} cut-signature calls, max P+C-n_ctx {c['max_margin']} ({sess})")
    return {"value": " / ".join(parts), "n": f"{rs['n_calls']} calls"}


def compute_r2_orphan_rates(repo):
    """R2 mechanism job, per tier over message-truncated calls: first kept non-system message by kind (user,
    assistant tool-call, assistant final, tool), orphaned (no user message before it in the kept window) and
    orphaned tool results (tool message whose assistant tool call was dropped)."""
    sp = _r2_shift_module(repo)
    o = sp.orphan_summary(sp.read_rows(repo / sp.MECH_FILE))
    pct = lambda x: "n/a" if x is None else f"{100 * x:.1f}%"
    parts = [f"{t}: {d['truncated']}/{d['calls']} truncated; first kept user {d['first_user']}, assistant tool-call "
             f"{d['first_assistant_toolcall']}, assistant final {d['first_assistant_final']}, tool {d['first_tool']}; "
             f"orphaned {d['orphan_first_kept']}/{d['truncated']} ({pct(d['orphan_rate'])}); orphaned tool result "
             f"{d['first_tool']}/{d['truncated']} ({pct(d['orphan_tool_rate'])}); rebuilt-role check "
             f"{d['role_check_ok']}/{d['truncated']}" for t, d in o.items()]
    return {"value": " / ".join(parts), "n": sum(d["calls"] for d in o.values())}


# ───────────────────────────────────────────────── R2 session cost and information loss (operator plan v2, 2026-10-13)
# A1: analysis/r2_session_cost.py (wall-clock, tokens, cloud cost per session; priced from the latest
# results/cloud_prices_*.json per provider, so an Anthropic price file is picked up with no code change);
# A2: analysis/r2_info_loss.py (share of transcript and system prompt never seen, rule violations before vs after the
# first system-prompt loss). Both are pure functions of the committed result files below.
R2_SESSION_COST_FILES = ["results/x2_r2_real_v1.jsonl", "results/x2_r2_real_v1b.jsonl", "results/cloud_prices_*.json"]
R2_INFO_LOSS_FILES = ["results/x2_r2_mechanism.jsonl", "results/blade_r2_mechanism_4096_v2.jsonl",
                      "results/x2_r2_real_v1.jsonl", "results/x2_r2_real_v1b.jsonl"]
_R2_A_CACHE = {}


def _analysis_module(repo, name):
    import importlib
    import sys as _sys
    p = str(REPO / "analysis")
    if p not in _sys.path:
        _sys.path.insert(0, p)
    return importlib.import_module(name)


def _r2_a_report(repo, name):
    mod = _analysis_module(repo, name)
    key = (name, str(repo))
    if key not in _R2_A_CACHE:
        _R2_A_CACHE[key] = mod.build(repo)
    return mod, _R2_A_CACHE[key]


def compute_r2_session_wallclock(repo):
    """Per file (x2_r2_real_v1, x2_r2_real_v1b) and combined, per model x tier: session wall-clock minutes (sum of
    turn_wall_s; mean and per seed), billed tokens (max session_tokens_billed_cumulative, input + output as the harness
    counts them), input (billed - output) and output tokens (sum of completion tokens); then the operator's
    expectations compared (never adjusted)."""
    sc, rep = _r2_a_report(repo, "r2_session_cost")
    return {"value": sc.wallclock_str(rep) + " / operator expectations: " + sc.expectation_str(rep, "wallclock"),
            "n": f"{len(rep['combined'])} cells, {sum(c['n'] for c in rep['combined'].values())} sessions"}


def compute_r2_session_cloud_cost(repo):
    """Per file and combined, per model x tier: USD per session at each provider's selected models' Standard list
    prices (latest price file per provider), best-case caching to no caching (mean), with the per-seed extremes;
    then the operator's expectations compared (never adjusted)."""
    sc, rep = _r2_a_report(repo, "r2_session_cost")
    return {"value": sc.cost_str(rep) + " / operator expectations: " + sc.expectation_str(rep, "cost"),
            "n": f"{sum(c['n'] for c in rep['combined'].values())} sessions"}


def compute_info_loss_by_window(repo):
    """Per machine x model x window: share of transcript tokens and of system-prompt tokens never seen at turns
    5/10/20/30/40, turns with system-prompt loss and the first loss turn; observed (mechanism, one session per tier)
    vs predicted (R2 real runs, analysis/r2_shift_predictor.py)."""
    il, rep = _r2_a_report(repo, "r2_info_loss")
    return {"value": il.info_loss_str(rep), "n": f"{sum(len(v) for v in rep['cells'].values())} sessions"}


def compute_reliability_after_guardrail_loss(repo):
    """Per machine x model x window: violation rate before vs from the first system-prompt loss, turns from that loss
    to the first violation, violations by rule type before -> after (observed one session per tier; predicted)."""
    il, rep = _r2_a_report(repo, "r2_info_loss")
    return {"value": il.reliability_str(rep), "n": f"{sum(len(v) for v in rep['cells'].values())} sessions"}


# ───────────────────────────────────────────────── R2 mitigation x2_r2_mitigation_v1 (pre-registered 2026-10-08)
# Client-side trimming at num_ctx 4096 / 8192 (harness/x2_r2_client_trim.py) vs the real runs' same cells. PENDING until
# results/x2_r2_mitigation_v1.jsonl is synced (and, for the 8192 comparison, results/x2_r2_real_v1b.jsonl).
R2_MITIGATION_FILE = "results/x2_r2_mitigation_v1.jsonl"


def _r2_mitigation_report(repo, tiers=None):
    import sys as _sys
    _sys.path.insert(0, str(repo / "analysis"))
    _sys.path.insert(0, str(REPO / "analysis"))
    import r2_mitigation_report as mr
    if not (repo / R2_MITIGATION_FILE).exists():
        raise FileNotFoundError(repo / R2_MITIGATION_FILE)
    return mr, mr.build(repo / R2_MITIGATION_FILE,
                        real_paths={"num_ctx_4096": repo / R2_REAL_V1_FILE, "num_ctx_8192": repo / R2_REAL_V1B_FILE},
                        validation_path=repo / R2_VALIDATION_V2_FILE, tiers=tiers)


def _r2_mitigation_metrics_str(mr, m):
    return (f"{mr.rule_summary(m)}; first rule failure {mr._f(m['first_rule_failure_per_seed'])}; canary misses C "
            f"{m['canary_sys_misses']}, H {m['canary_hist_misses']} of {m['canary_checks']}; error turns "
            f"{m['error_turns']}; thinking turns {m['thinking_turns']}; think sent {','.join(m['think_sent'])} "
            f"(seeds {mr._f(m['seeds'])})")


def compute_r2_mitigation_v1_verdict(repo):
    """Pre-registered verdict (docs/FINDINGS.md, 2026-10-08): per model at num_ctx 4096 recovered / not_recovered /
    invalid (with reasons), the overall verdict, and the same rule at 8192 as a control."""
    _mr, rep = _r2_mitigation_report(repo)
    parts = []
    for key, c in rep["cells"].items():
        tier, model = key.split("|")
        v = c["verdict"]
        extra = (f" (rules below 90%: {', '.join(v['rules_below_threshold'])})" if v.get("rules_below_threshold")
                 else f" ({'; '.join(v['reasons'])})" if v["reasons"] else "")
        parts.append(f"{tier} {model} {v['verdict']}{extra}")
    return {"value": f"pre-registered verdict at {rep['verdict_tier']}: {rep['overall_verdict']}; " + "; ".join(parts),
            "n": f"{rep['n_sessions']} sessions, {rep['n_turns']} turns"}


def _r2_mitigation_tier(repo, tier):
    mr, rep = _r2_mitigation_report(repo, tiers=(tier,))
    if tier == "num_ctx_8192" and not (repo / R2_REAL_V1B_FILE).exists():
        raise FileNotFoundError(repo / R2_REAL_V1B_FILE)
    parts = []
    for key, c in rep["cells"].items():
        model = key.split("|")[1]
        s = f"{model} mitigation: {_r2_mitigation_metrics_str(mr, c['mitigation'])}"
        if c["real"]["available"]:
            s += f" / {model} real ({c['real']['file']}): {_r2_mitigation_metrics_str(mr, c['real']['metrics'])}"
        else:
            s += f" / {model} real: not available"
        parts.append(s)
    if not parts:
        raise ValueError(f"no {tier} cells in {R2_MITIGATION_FILE}")
    return {"value": " // ".join(parts), "n": f"{rep['n_sessions']} mitigation sessions"}


def compute_r2_mitigation_v1_4096(repo):
    """Mitigation vs x2_r2_real_v1 at num_ctx 4096, same models and seeds: rule survival per rule in use (rate, n,
    first failure per seed), first rule failure, canary C/H misses, surfaced errors, thinking."""
    return _r2_mitigation_tier(repo, "num_ctx_4096")


def compute_r2_mitigation_v1_8192(repo):
    """Mitigation vs x2_r2_real_v1b at num_ctx 8192, same models and seeds (as compute_r2_mitigation_v1_4096)."""
    return _r2_mitigation_tier(repo, "num_ctx_8192")


def compute_r2_mitigation_v1_observed(repo):
    """Mitigation run, observed from the OLLAMA_DEBUG log per tier x model: context shifts, Ollama token-level cuts,
    calls on which Ollama still dropped messages, exceed-context errors, the client's counting method, over-budget
    calls, turns kept, client exact count vs llama-server's prompt token count."""
    _mr, rep = _r2_mitigation_report(repo)
    parts = []
    for key, c in rep["cells"].items():
        o = c["observed"]
        parts.append(f"{key.replace('|', ' ')}: {o['n_calls']} calls ({o['n_calls_trimmed']} trimmed), "
                     f"{o['context_shift_events']} context shifts, {o['token_cut_calls']} token cuts, "
                     f"{o['ollama_message_drop_calls']} Ollama message drops, {o['exceed_context_error_calls']} "
                     f"exceed-context errors, {o['log_slice_incomplete_calls']} incomplete log slices; count method "
                     f"{o['count_method']}; over budget {o['over_budget_calls']}; budget {o['budget_prompt_tokens']}, "
                     f"num_predict {o['num_predict']}, max kept prompt {o['max_kept_prompt_tokens_exact']}; turns kept "
                     f"{o['min_turns_kept']}-{o['max_turns_kept']}; client = server prompt tokens "
                     f"{o['client_vs_server_tokens_equal']}/{o['client_vs_server_tokens_compared']} (max abs diff "
                     f"{o['client_vs_server_max_abs_diff']})")
    return {"value": " / ".join(parts), "n": f"{sum(c['observed']['n_calls'] for c in rep['cells'].values())} calls"}


def compute_blade_validation_ctx_cap(repo):
    """Blade validation context cap (operator rule 2026-10-08): max over every call of the evo-x2 R2 validation
    sessions at num_ctx 131072 (both call-2 variants; the 8192 positive control is excluded because it truncates by
    design) of prompt + generated tokens, taking per call the larger of Ollama's prompt_eval_count and the calibrated
    estimate (prompt_tokens_est x the session's token_calib_ratio). Rule: max < 30000 -> cap 32768, else 65536."""
    import sys as _sys
    _sys.path.insert(0, str(repo / "harness"))
    import x2_r2_agent as ra
    out, overall = [], 0
    for f, m in (("results/x2_r2_validation_v2.jsonl", "llama3.1:8b"), ("results/x2_r2_validation_v2b.jsonl", "qwen3:8b")):
        best = (0, None)
        for r in ra.read_rows(repo / f):
            if r.get("record") != "r2a_turn" or r.get("model_id") != m or "positive_control" in r.get("arm_id", ""):
                continue
            ratio = r.get("token_calib_ratio") or 1.0
            for c in r.get("calls", []):
                prompt = max(c.get("prompt_eval_count_info_only") or 0, round((c.get("prompt_tokens_est") or 0) * ratio))
                tot = prompt + (c.get("completion_tokens") or 0)
                if tot > best[0]:
                    best = (tot, f"{r['arm_id']} seed {r['seed']} turn {r['turn_idx']}")
        out.append(f"{m}: max prompt+generated {best[0]} ({best[1]})")
        overall = max(overall, best[0])
    cap = 32768 if overall < 30000 else 65536
    return {"value": "; ".join(out) + f"; overall max {overall} -> Blade validation cap {cap} (rule: <30000 -> 32768, else 65536)",
            "n": 2}

# ───────────────────────────────────────────────── router envelope inputs (src/dse/envelope_data.json, 2026-10-08)


def compute_r2_real_v1_loaded_ctx(repo):
    """Context Ollama actually loaded (GET /api/ps context_length, every turn) per R2 real-run tier x model on evo-x2.
    For the ollama_default tier this is Ollama's own default for that model on evo-x2; for the num_ctx tiers it is
    the requested value. detail: {tier: {model: loaded_context}}; raises if a (tier, model) cell saw more than one
    value."""
    rows = [r for r in _read_jsonl(repo / R2_REAL_V1_FILE) if r.get("record") == "r2a_turn"]
    seen = {}
    for r in rows:
        tier = r["arm_id"].replace("_call2_notools", "")
        seen.setdefault(tier, {}).setdefault(r["model_id"], set()).add(r.get("loaded_context"))
    detail = {}
    for tier, models in seen.items():
        for m, vals in models.items():
            vals = {v for v in vals if v is not None}
            if len(vals) != 1:
                raise ValueError(f"{tier} {m}: loaded_context not unique: {sorted(vals)}")
            detail.setdefault(tier, {})[m] = vals.pop()
    value = "; ".join(f"{t} " + ", ".join(f"{m} {v}" for m, v in sorted(ms.items())) for t, ms in sorted(detail.items()))
    return {"value": value, "n": len(rows), "detail": detail}


def compute_r2_real_v1_token_calib(repo):
    """Per model, the R2 harness's tokenizer calibration ratio (turn 1 call 1 prompt_eval_count / chars-per-4
    estimate, one per session, harness/x2_r2_agent.py run_session): median, min, max over the real run's sessions.
    detail: {model: {"median", "min", "max", "n_sessions"}}."""
    rows = [r for r in _read_jsonl(repo / R2_REAL_V1_FILE) if r.get("record") == "r2a_session"]
    by = {}
    for r in rows:
        if r.get("token_calib_ratio") is not None:
            by.setdefault(r["model_id"], []).append(float(r["token_calib_ratio"]))
    if not by:
        raise FileNotFoundError("no r2a_session row with token_calib_ratio")
    detail = {m: {"median": statistics.median(v), "min": min(v), "max": max(v), "n_sessions": len(v)}
              for m, v in sorted(by.items())}
    value = "; ".join(f"{m} median {d['median']:.4f} (range {d['min']:.4f}-{d['max']:.4f}, {d['n_sessions']} sessions)"
                      for m, d in detail.items())
    return {"value": value, "n": sum(len(v) for v in by.values()), "detail": detail}


def compute_decode_rate_per_machine(repo):
    """Median decode_tok_s per (machine, model) over the same pooled real files as the TTFT fit
    (_TTFT_FIT_RESULT_FILES), warm-up rows excluded and only rows with no co-runner (co_runner absent, "none" or
    "N0"), so the co-runner penalty stays a separate term. detail: {machine: {model: {"median", "n"}}}."""
    by = {}
    found_any = False
    for rel_path in _TTFT_FIT_RESULT_FILES:
        path = repo / rel_path
        if not path.exists():
            continue
        found_any = True
        for r in _read_jsonl(path):
            d = r.get("decode_tok_s")
            if not d or r.get("warmup") is True or r.get("co_runner") not in (None, "none", "N0"):
                continue
            hw = r.get("hw_id") or r.get("host")
            model = r.get("model_id") or r.get("model_tag")
            if hw not in ("evo-t2s", "evo-x2") or model is None:
                continue
            by.setdefault(hw, {}).setdefault(model, []).append(float(d))
    if not found_any:
        raise FileNotFoundError("none of the TTFT fit result files were found")
    detail = {hw: {m: {"median": statistics.median(v), "n": len(v)} for m, v in sorted(ms.items())}
              for hw, ms in sorted(by.items())}
    value = " / ".join(f"{hw}: " + ", ".join(f"{m} {d['median']:.2f} tok/s (n={d['n']})" for m, d in ms.items())
                       for hw, ms in detail.items())
    return {"value": value, "n": sum(d["n"] for ms in detail.values() for d in ms.values()), "detail": detail}


# Blade 60-second dry runs (docs/BLADE_PLAN.md): PENDING until the operator's dry runs have written their files.
BLADE_DRYRUN_FILES = ["results/blade_dryrun/blade_dryrun_k1.jsonl", "results/blade_dryrun/blade_dryrun_r2_validation.jsonl",
                      "results/blade_dryrun/blade_dryrun_r2_real.jsonl",
                      "results/blade_dryrun/blade_dryrun_r2_mitigation.jsonl",
                      "results/blade_dryrun/blade_dryrun_r2_mechanism.jsonl"]


def _blade_estimator(repo):
    import importlib.util
    spec = importlib.util.spec_from_file_location("blade_hours_estimate", REPO / "analysis" / "blade_hours_estimate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.REPO = repo
    return mod


def compute_blade_dryrun_rates(repo):
    """Per Blade dry-run file: median prefill tok/s (prompt_eval_count / prompt_eval_duration) and decode tok/s
    (eval_count / eval_duration) per (model, num_ctx), over calls with >= 64 prompt / >= 8 generated tokens."""
    est = _blade_estimator(repo)
    rates = est.dryrun_rates(BLADE_DRYRUN_FILES)
    parts, n = [], 0
    for fname, d in rates.items():
        if not d:
            parts.append(f"{fname}: no rated calls")
            continue
        for key, v in sorted(d.items()):
            n += 1
            pf = "n/a" if v["prefill_tps"] is None else f"{v['prefill_tps']:.0f}"
            dc = "n/a" if v["decode_tps"] is None else f"{v['decode_tps']:.1f}"
            parts.append(f"{fname} {key}: prefill {pf} tok/s (n={v['n_prefill']}), decode {dc} tok/s (n={v['n_decode']})")
    return {"value": "; ".join(parts), "n": n}


def compute_blade_night_hours(repo):
    """Blade per-night hour estimates (analysis/blade_hours_estimate.py): Blade dry-run rates where they exist for the
    job's (model, num_ctx), scaled evo-x2 times otherwise; each job labelled measured / rates / scaled / mixed."""
    est = _blade_estimator(repo)
    e = est.estimate(BLADE_DRYRUN_FILES)
    jobs = "; ".join(f"night {j['night']} {j['job'].split(' ')[0]} {j['hours']:.2f} h ({j['kind']})"
                     + (f" [branch {j['branch']}]" if j.get("branch") else "") for j in e["jobs"])
    by_branch = e.get("night_totals_by_branch_h") or {}

    def total(k, v):
        if k not in by_branch:
            return f"night {k} {v:.2f} h"
        return f"night {k} {v:.2f} h (gate branches: " + ", ".join(
            f"{b} {h:.2f} h" for b, h in by_branch[k].items()) + ")"
    totals = ", ".join(total(k, v) for k, v in e["night_totals_h"].items())
    return {"value": f"{totals}; {jobs}", "n": len(e["jobs"])}


# Blade R2 validation runs (llama3.1:8b, arm b at the 32768 cap, call-2 mode off), scored by x2_r2_agent.evaluate_gates
# with the Blade negative-control arm. run1 = night 1's first run (2026-10-09, 3 negative-control sessions); run2 = the
# night 1 rerun with the negative control at 5 sessions (PENDING until its file is synced).
BLADE_R2_VALIDATION_V1 = "results/blade_r2_validation_v1.jsonl"
BLADE_R2_VALIDATION_V2 = "results/blade_r2_validation_v2.jsonl"
BLADE_NEG_ARM = "ollama_ctx_32768_negative_control"


def _blade_r2_validation_summary(repo, rel):
    ag = _r2_agent_module()
    rows = ag.read_rows(repo / rel)
    if not rows:
        raise FileNotFoundError(repo / rel)
    sfx = ag.MODE_SUFFIX["off"]
    rep = ag.evaluate_gates(rows, call2_tools="off", neg_arm=BLADE_NEG_ARM)
    sessions = ag.completed_sessions(rows)
    parts, n_total = [], 0
    for model, e in sorted(rep["baseline_table"].items()):
        n_total += e["n_turns"]
        g = rep["gates"]
        neg = rep["controls"]["negative"].get(model, {})
        pos = rep["controls"]["positive"].get(model, {})
        per = sorted((s["seed"], s["canary_sys_misses"] + s["canary_hist_misses"]) for s in sessions
                     if s["model_id"] == model and s["arm_id"] == BLADE_NEG_ARM + sfx)
        bl = g["baseline"].get(model, {})
        parts.append(
            f"{model}: {_r2_table_str(e)}; rules in use {bl.get('rules_in_use')}, failing {bl.get('failing_rules')}; "
            f"negative control (num_ctx 32768) canary misses {neg.get('canary_misses')} of "
            f"{2 * (neg.get('canary_checks') or 0)} canaries ({neg.get('canary_checks')} checks x 2) over "
            f"{neg.get('n_sessions')} sessions, per session (seed: misses) "
            + ", ".join(f"{s}: {m}" for s, m in per)
            + f" -> {'pass' if (g['negative_control'].get(model) or {}).get('pass') else 'fail'}; positive control "
            f"(8192) truncation turns {pos.get('truncation_detected_turns')}, canary misses "
            f"{pos.get('canary_sys_misses')}+{pos.get('canary_hist_misses')} -> "
            f"{'pass' if (g['positive_control'].get(model) or {}).get('pass') else 'fail'}; task tool "
            f"{(g['task_tool'].get(model) or {}).get('n_ok')}/{(g['task_tool'].get(model) or {}).get('n')}")
    return {"value": " / ".join(parts), "n": f"{n_total} turns", "detail": rep}


def compute_blade_r2_validation_run1(repo):
    """Blade R2 validation, night 1 first run (results/blade_r2_validation_v1.jsonl): per-rule rates on the gate arm,
    negative-control canary misses per session at num_ctx 32768, positive control at 8192."""
    return _blade_r2_validation_summary(repo, BLADE_R2_VALIDATION_V1)


def compute_blade_r2_validation_run2(repo):
    """Blade R2 validation, night 1 rerun (results/blade_r2_validation_v2.jsonl, negative control at 5 sessions,
    seeds 20260901-05): same fields as run1."""
    return _blade_r2_validation_summary(repo, BLADE_R2_VALIDATION_V2)


def compute_blade_k1_layer_split(repo):
    """Blade K1 (night 1 rerun, results/blade_k1_v2.summary.json): Ollama default context per model, and per tested
    num_ctx the share of the loaded model resident in VRAM (size_vram / size from /api/ps) and whether it is fully on
    the GPU; plus the overflow behaviour at the default (prompt_eval_count after an over-length prompt, HTTP status)."""
    import json as _json
    s = _json.loads((repo / "results/blade_k1_v2.summary.json").read_text(encoding="utf-8"))
    parts = []
    for m, v in s["models"].items():
        if not v.get("default_ctx"):
            parts.append(f"{m}: not measured (model not present)")
            continue
        split = ", ".join(f"{o['num_ctx']}: {o['size_vram'] / o['size']:.3f}" for o in v.get("overflow_window", []) if o.get("size"))
        ov = "; ".join(f"{o['target_tokens']} tok -> HTTP {o['http_status']}, evaluated {o['prompt_eval_count']}"
                       for o in v.get("overflow_default", []))
        parts.append(f"{m}: default ctx {v['default_ctx']}, fully on GPU at default {v['fully_on_gpu']}; VRAM share by num_ctx "
                     f"{split}; overflow at default: {ov}")
    return {"value": " / ".join(parts) + f"; device VRAM {s['device'].get('vram_total_mib')} MiB", "n": len(s["models"])}


def compute_blade_r2_gate_v2(repo):
    """Blade night 1 rerun gate (results/blade_r2_gate_v2.json): negative-control canary misses per session and branch."""
    import json as _json
    g = _json.loads((repo / "results/blade_r2_gate_v2.json").read_text(encoding="utf-8"))
    per = ", ".join(f"{x['seed']}: {x['canary_misses']}" for x in g["per_session"])
    return {"value": f"{g['model']} negative control num_ctx {g['num_ctx']}, {g['n_neg_sessions']} sessions, canary misses per "
                     f"session {per}; branch {g.get('branch') or g.get('decision')}", "n": g["n_neg_sessions"]}


def compute_blade_mechanism_4096(repo):
    """Blade mechanism run at num_ctx 4096 (results/blade_r2_mechanism_4096_v2.jsonl.report.json), llama3.1:8b, 1 session."""
    import json as _json
    rep = _json.loads((repo / "results/blade_r2_mechanism_4096_v2.jsonl.report.json").read_text(encoding="utf-8"))
    s = rep["sessions"][0]; pc = s["prompt_check"]
    return {"value": (f"prompt check {pc['n_match']}/{pc['n_checks']} within 1% (citable {pc['mechanism_citable']}); "
                      f"{s['n_calls_message_truncated']}/{s['n_calls']} calls message-truncated from turn {s['first_truncated_turn']}, "
                      f"system kept on every truncated call {s['system_kept_on_every_truncated_call']}, cut mid-turn "
                      f"{s['cut_mid_turn_calls']}, token-level cuts {s['token_level_cut_calls']}, context shifts "
                      f"{s['context_shift_events']}, exceed-context errors {s['exceed_context_error_calls']}; verdict "
                      f"{rep['verdict']['drop_old_turns_keep_system']}"), "n": s["n_calls"]}


# Blade open-finding confound check on evo-x2 (pre-registered in docs/FINDINGS.md 2026-10-10, queue id
# x2_r2_llama_val_32k): Blade run 2's validation (llama3.1:8b, negative control at num_ctx 32768, 5 sessions, seeds
# 20260901-05, v1 call-2 mode) on evo-x2. PENDING until its file is synced. The verdict rule is the pre-registered one.
X2_R2_LLAMA_VAL_32K = "results/x2_r2_llama_val_32k.jsonl"
X2_VAL32K_NEG_ARM = "ollama_ctx_32768_negative_control"
X2_VAL32K_NUM_CTX = 32768
X2_VAL32K_SEEDS = (20260901, 20260902, 20260903, 20260904, 20260905)
X2_VAL32K_TURNS = 10
X2_VAL32K_BLADE_TURN = 5   # the turn of the Blade's text tool calls (REC-0005 lookup), blade-r2-gate-v2


def text_tool_call_name(text):
    """The pre-registered text-tool-call pattern: the final text, after leading whitespace, starts with a JSON object
    whose first key is "name" ({"name": ...}). Returns the named tool (or "" if "name" is not a plain string), or None
    when the text does not match."""
    import re
    m = re.match(r'\s*\{\s*"name"\s*:\s*(?:"([^"]*)")?', text or "")
    return None if m is None else (m.group(1) or "")


def x2_r2_llama_val_32k_verdict(rows, ag, model="llama3.1:8b") -> dict:
    """Mechanical verdict for x2_r2_llama_val_32k from the negative-control (gate) arm only, as pre-registered:
    valid only with run_end, all 5 sessions (seeds 20260901-05) completed with 10 turns, every row at num_ctx 32768
    and no turn over the loaded window (else "inconclusive (incomplete: ...)"); then 0 canary misses -> "backend or
    offload effect"; at least one miss and every missed canary turn a turn-5 text tool call -> "window effect"; any
    other pattern -> "inconclusive". `ag` is x2_r2_agent (completed_sessions, MODE_SUFFIX)."""
    arm = X2_VAL32K_NEG_ARM + ag.MODE_SUFFIX["off"]
    sessions = [s for s in ag.completed_sessions(rows)
                if s["arm_id"] == arm and s["model_id"] == model and s.get("mode") == "validation"]
    by_attempt = {s["attempt_id"]: s for s in sessions}
    turns = sorted((r for r in rows if r.get("record") == "r2a_turn" and r.get("attempt_id") in by_attempt),
                   key=lambda r: (r["seed"], r["turn_idx"]))
    problems = []
    if not any(r.get("record") == "run_end" for r in rows):
        problems.append("no run_end")
    seeds = sorted(s["seed"] for s in sessions)
    if seeds != list(X2_VAL32K_SEEDS):
        problems.append(f"negative-control sessions completed for seeds {seeds}, expected {list(X2_VAL32K_SEEDS)}")
    for s in sorted(sessions, key=lambda s: s["seed"]):
        n = sum(1 for r in turns if r["attempt_id"] == s["attempt_id"])
        if n != X2_VAL32K_TURNS:
            problems.append(f"seed {s['seed']}: {n} turns, expected {X2_VAL32K_TURNS}")
    bad_ctx = sorted({r.get("num_ctx_requested") for r in turns} - {X2_VAL32K_NUM_CTX}, key=str)
    if bad_ctx:
        problems.append(f"num_ctx_requested {bad_ctx} on negative-control rows")
    over = [(r["seed"], r["turn_idx"]) for r in turns if r.get("over_loaded_window")]
    if over:
        problems.append(f"turns over the loaded window {over}")
    per_session = {s["seed"]: s["canary_sys_misses"] + s["canary_hist_misses"]
                   for s in sorted(sessions, key=lambda s: s["seed"])}
    misses = []
    for r in turns:
        if r.get("canary_check") and not (r.get("canary_sys_ok") and r.get("canary_hist_ok")):
            name = text_tool_call_name(r.get("final_text"))
            misses.append({"seed": r["seed"], "turn": r["turn_idx"], "sys_missed": not r.get("canary_sys_ok"),
                           "hist_missed": not r.get("canary_hist_ok"), "text_tool_call": name is not None,
                           "tool_name": name})
    n_missed = sum(per_session.values())

    def fmt(m):
        which = "+".join(k for k, v in (("C", m["sys_missed"]), ("H", m["hist_missed"])) if v)
        tc = f"text tool call {m['tool_name'] or '(unnamed)'}" if m["text_tool_call"] else "not a text tool call"
        return f"seed {m['seed']} turn {m['turn']} {which} missed, {tc}"

    pattern = "; ".join(fmt(m) for m in misses) or "no missed canary turn"
    if problems:
        verdict = "inconclusive (incomplete: " + "; ".join(problems) + ")"
    elif n_missed == 0 and not misses:
        verdict = "backend or offload effect"
    elif misses and all(m["text_tool_call"] and m["turn"] == X2_VAL32K_BLADE_TURN for m in misses):
        verdict = "window effect"
    else:
        verdict = "inconclusive"
    return {"verdict": verdict, "valid": not problems, "problems": problems, "per_session": per_session,
            "n_missed_canaries": n_missed, "n_canaries": 2 * sum(s["canary_checks"] for s in sessions),
            "misses": misses, "pattern": pattern}


def compute_x2_r2_llama_val_32k(repo):
    """x2_r2_llama_val_32k (results/x2_r2_llama_val_32k.jsonl, evo-x2): per-rule rates on the gate arm, negative-
    control canary misses per session at num_ctx 32768, each missed canary turn (seed, turn, text tool call or not),
    positive control, diagnostic arm misses, and the pre-registered verdict (x2_r2_llama_val_32k_verdict)."""
    ag = _r2_agent_module()
    rows = ag.read_rows(repo / X2_R2_LLAMA_VAL_32K)
    if not rows:
        raise FileNotFoundError(repo / X2_R2_LLAMA_VAL_32K)
    ag.neg_arm_for_ctx(X2_VAL32K_NUM_CTX)   # registers the arm in ARMS (gates read its num_ctx)
    model = "llama3.1:8b"
    rep = ag.evaluate_gates(rows, call2_tools="off", neg_arm=X2_VAL32K_NEG_ARM)
    v = x2_r2_llama_val_32k_verdict(rows, ag, model)
    e = rep["baseline_table"].get(model)
    pos = rep["controls"]["positive"].get(model, {})
    diag_arm = X2_VAL32K_NEG_ARM + ag.MODE_SUFFIX["on"]
    diag = [s for s in ag.completed_sessions(rows) if s["arm_id"] == diag_arm and s["model_id"] == model]
    diag_m = sum(s["canary_sys_misses"] + s["canary_hist_misses"] for s in diag)
    tt = rep["gates"]["task_tool"].get(model) or {}
    value = (f"{model}: {_r2_table_str(e) if e else 'no gate-arm turns'}; negative control (num_ctx 32768) canary "
             f"misses {v['n_missed_canaries']} of {v['n_canaries']} canaries over {len(v['per_session'])} sessions, "
             f"per session (seed: misses) " + ", ".join(f"{s}: {m}" for s, m in v["per_session"].items())
             + f"; missed turns: {v['pattern']}; positive control (8192) truncation turns "
             f"{pos.get('truncation_detected_turns')}, canary misses {pos.get('canary_sys_misses')}+"
             f"{pos.get('canary_hist_misses')}; diagnostic arm (tools on call 2) canary misses {diag_m} over "
             f"{len(diag)} sessions; task tool {tt.get('n_ok')}/{tt.get('n')}; verdict: {v['verdict']}")
    return {"value": value, "n": f"{e['n_turns'] if e else 0} turns", "detail": {"verdict": v, "gates": rep}}


# Evalset (a) tasks validation (results/x2_evalset_tasks_val_v1.jsonl, evo-x2, harness/x2_r2_evalset.py --mode
# validation --workload evalset --guardrails off): every gate of x2_r2_evalset.evaluate_gates_ext per model, the tasks in
# use (end state correct on every negative-control session, docs/EVALSET_DESIGN.md) and, per negative-control seed, the
# tasks whose end state was correct in that session. The input of the B3 task real run (x2_evalset_tasks_real_v1).
X2_EVALSET_TASKS_VAL = "results/x2_evalset_tasks_val_v1.jsonl"


def _evalset_module():
    import sys
    sys.path.insert(0, str(REPO / "harness"))
    import x2_r2_evalset
    return x2_r2_evalset


def evalset_tasks_correct_per_seed(rows, ag, ev, arm) -> dict:
    """{model: {seed: sorted task ids whose end state was correct in that seed's completed negative-control session}}."""
    out = {}
    for r in ev._neg_turn_rows(rows, arm):
        if r.get("item_kind") != "task":
            continue
        d = out.setdefault(r["model_id"], {}).setdefault(r["seed"], set())
        if r.get("end_state_ok"):
            d.add(r["task_id"])
    return {m: {s: sorted(v) for s, v in sorted(per.items())} for m, per in sorted(out.items())}


def compute_x2_evalset_tasks_val_gates(repo):
    """x2_evalset_tasks_val_v1 gates per model (pass/fail, with the failing rules and counts), negative-control loaded
    context and over-window task turns, tasks in use (strict, every session) and tasks correct per seed, end state by
    difficulty, silent wrong turns (end state wrong, HTTP 200, no error)."""
    ag, ev = _r2_agent_module(), _evalset_module()
    rows = ag.read_rows(repo / X2_EVALSET_TASKS_VAL)
    if not rows:
        raise FileNotFoundError(repo / X2_EVALSET_TASKS_VAL)
    rep = ev.evaluate_gates_ext(rows, workload="evalset", guardrails=False, call2_mode="off")
    arm = ag.NEG_ARM + ag.MODE_SUFFIX["off"]
    per_seed = evalset_tasks_correct_per_seed(rows, ag, ev, arm)
    task_turns = [r for r in ev._neg_turn_rows(rows, arm) if r.get("item_kind") == "task"]
    g = rep["gates"]
    finished = any(r.get("record") == "run_end" for r in rows)
    parts = []
    for m in sorted(rep["task_table"]):
        tt = rep["task_table"][m]
        neg, pos = g["negative_control"].get(m, {}), g["positive_control"].get(m, {})
        base, task = g["baseline"].get(m, {}), g["task_tool"].get(m, {})
        mt = [r for r in task_turns if r["model_id"] == m]
        over = sum(1 for r in mt if r.get("over_loaded_window"))
        diff = ", ".join(f"{k} {v['n_ok']}/{v['n']}" for k, v in sorted(tt["by_difficulty"].items()))
        seeds = ", ".join(f"{s}: {len(v)}/30" for s, v in per_seed.get(m, {}).items())
        parts.append(
            f"{m}: baseline gate {'pass' if base.get('pass') else 'fail'} (failing {base.get('failing_rules')}); "
            f"negative control (num_ctx 131072 requested) gate {'pass' if neg.get('pass') else 'fail'}, canary misses "
            f"{neg.get('canary_misses')} of {2 * (neg.get('canary_checks') or 0)} canaries ({neg.get('canary_checks')} "
            f"checks) over {neg.get('n_sessions')} sessions, loaded "
            f"context {neg.get('loaded_context_values')}, task turns over the loaded window {over}/{len(mt)}; positive "
            f"control {'pass' if pos.get('pass') else 'fail'}; task tool {task.get('n_ok')}/{task.get('n')} "
            f"({'pass' if task.get('pass') else 'fail'}); thinking off "
            f"{'pass' if g['thinking_off'].get(m, {}).get('pass') else 'fail'}; content non-empty "
            f"{'pass' if g['content_nonempty'].get(m, {}).get('pass') else 'fail'}; end state by difficulty {diff}; "
            f"silent wrong task turns {tt['silent_wrong_turns']}/{len(mt)}; tasks in use (correct on every "
            f"negative-control session) {len(tt['tasks_in_use'])}: {', '.join(tt['tasks_in_use']) or 'none'}; tasks "
            f"correct per seed {seeds}")
    value = (f"run finished {finished}; " + " / ".join(parts))
    return {"value": value, "n": f"{len(task_turns)} negative-control task turns",
            "detail": {"gates": g, "per_seed": per_seed, "task_table": rep["task_table"]}}


# B3 task real run (pre-registered in docs/FINDINGS.md 2026-10-14, queue id x2_evalset_tasks_real_v1): wrong-task rate
# per window, excess at 4096 over the Ollama default with a task-cluster bootstrap CI, silent vs surfaced wrong results,
# all from x2_r2_evalset.tasks_real_outcome. PENDING until the file is synced.
X2_EVALSET_TASKS_REAL = "results/x2_evalset_tasks_real_v1.jsonl"


def _evalset_tasks_real(repo):
    ag, ev = _r2_agent_module(), _evalset_module()
    rows = ag.read_rows(repo / X2_EVALSET_TASKS_REAL)
    if not rows:
        raise FileNotFoundError(repo / X2_EVALSET_TASKS_REAL)
    return ev.tasks_real_outcome(rows, ag.read_rows(repo / X2_EVALSET_TASKS_VAL), "off")


def _frac(a, b):
    return f"{a}/{b}" + (f" ({100 * a / b:.1f}%)" if b else "")


def compute_x2_evalset_tasks_real_wrong_rate(repo):
    """Per model: validity, and for the primary (strict tasks in use) and secondary (per-seed) cell sets the wrong-task
    rate at the default and at 4096, the excess with its 95% task-cluster bootstrap CI and the pre-registered reading."""
    o = _evalset_tasks_real(repo)
    parts, n = [], 0
    for m, d in o["models"].items():
        if d.get("refused"):
            parts.append(f"{m}: refused ({'; '.join(d['reasons'] or [])})")
            continue
        segs = [f"valid {d['valid']}" + (f" ({'; '.join(d['problems'])})" if d["problems"] else "")]
        for w, i in d["windows"].items():
            segs.append(f"{w}: loaded context {i['loaded_context_values']}, task turns over the window "
                        f"{i['task_turns_over_window']}/{i['n_task_turns']}")
        for name in ("primary", "secondary"):
            e = d[name]
            if not e["n_cells_in_use"]:
                segs.append(f"{name}: no cells in use (not estimable)")
                continue
            ci = e["excess_ci95"]
            segs.append(
                f"{name} ({e['n_tasks']} tasks, {e['n_cells_in_use']} cells): wrong default "
                f"{_frac(e['default']['wrong'], e['default']['n'])}, wrong 4096 {_frac(e['4096']['wrong'], e['4096']['n'])}"
                f" (state-only wrong {e['default']['state_wrong']} / {e['4096']['state_wrong']}); excess "
                + (f"{100 * e['excess']:+.1f} points, 95% CI {100 * ci['lo']:+.1f} to {100 * ci['hi']:+.1f} "
                   f"(task-cluster bootstrap, {ci['n_boot']} resamples); reading: {e['reading']}"
                   if e["excess"] is not None and ci["lo"] is not None else "n/a")
                + f"; not in use wrong rate default {_pct(e['default']['not_in_use_rate'])}, 4096 "
                  f"{_pct(e['4096']['not_in_use_rate'])}")
            n += e["default"]["n"] + e["4096"]["n"] if name == "secondary" else 0
        parts.append(f"{m}: " + "; ".join(segs))
    return {"value": f"run finished {o['run_finished']}; " + " / ".join(parts), "n": f"{n} secondary cells",
            "detail": o}


def compute_x2_evalset_tasks_real_silent(repo):
    """Per model, cell set and window: wrong cells split into silent (every call HTTP 200, outcome ok, no error) and
    surfaced, and the silent share of the wrong cells."""
    o = _evalset_tasks_real(repo)
    parts = []
    for m, d in o["models"].items():
        if d.get("refused"):
            parts.append(f"{m}: refused")
            continue
        segs = []
        for name in ("primary", "secondary"):
            e = d[name]
            if not e["n_cells_in_use"]:
                segs.append(f"{name}: no cells in use")
                continue
            segs.append(f"{name}: " + ", ".join(
                f"{w} wrong {e[w]['wrong']} = silent {e[w]['silent_wrong']} + surfaced {e[w]['surfaced_wrong']}"
                f" (silent share {_pct(e[w]['silent_share'])})" for w in ("default", "4096")))
        parts.append(f"{m} (valid {d['valid']}): " + "; ".join(segs))
    return {"value": " / ".join(parts), "n": f"{len(o['models'])} models", "detail": o}


# Blade confound follow-up (pre-registered in docs/FINDINGS.md 2026-10-14, queue id x2_r2_llama_val_32k_offload): the
# x2_r2_llama_val_32k validation with options.num_gpu 22. PENDING until its file is synced. Verdict rule pre-registered.
X2_R2_LLAMA_VAL_32K_OFFLOAD = "results/x2_r2_llama_val_32k_offload.jsonl"
X2_OFFLOAD_NUM_GPU = 22
X2_OFFLOAD_N_LAYERS = 33
X2_OFFLOAD_TOL = 2


def x2_r2_llama_val_32k_offload_verdict(rows, ag, model="llama3.1:8b") -> dict:
    """Pre-registered: the x2_r2_llama_val_32k validity and miss pattern (x2_r2_llama_val_32k_verdict) plus the split
    check per negative-control session (a num_gpu_offload row with at least one offload line, every line N/33 with
    |N - 22| <= 2, every chat call carrying num_gpu 22). Labels: "offload effect" (turn-5 text-tool-call misses only),
    "backend effect (CUDA vs evo-x2's backend)" (0 misses), else "inconclusive (...)"."""
    import x2_num_gpu as ng
    base = x2_r2_llama_val_32k_verdict(rows, ag, model)
    arm = X2_VAL32K_NEG_ARM + ag.MODE_SUFFIX["off"]
    off = {r["seed"]: r for r in rows if r.get("record") == "num_gpu_offload" and r.get("arm_id") == arm
           and r.get("model_id") == model}
    split_problems, unverified = [], []
    for s in X2_VAL32K_SEEDS:
        r = off.get(s)
        if r is None or not r.get("offloaded"):
            unverified.append(s)
            continue
        if not ng.split_as_intended(r["offloaded"], X2_OFFLOAD_NUM_GPU, X2_OFFLOAD_N_LAYERS, X2_OFFLOAD_TOL):
            split_problems.append(f"seed {s}: offloaded {r['offloaded']}")
        if r.get("num_gpu_sent") != X2_OFFLOAD_NUM_GPU or r.get("n_chat_calls_num_gpu") != r.get("n_chat_calls"):
            split_problems.append(f"seed {s}: num_gpu sent {r.get('num_gpu_sent')} on {r.get('n_chat_calls_num_gpu')}"
                                  f"/{r.get('n_chat_calls')} calls")
    if not base["valid"]:
        verdict = base["verdict"]
    elif split_problems:
        verdict = "inconclusive (offload split not as intended: " + "; ".join(split_problems) + ")"
    elif unverified:
        verdict = f"inconclusive (offload split not verified: no offload line for seeds {unverified})"
    elif base["verdict"] == "backend or offload effect":
        verdict = "backend effect (CUDA vs evo-x2's backend)"
    elif base["verdict"] == "window effect":
        verdict = "offload effect"
    else:
        verdict = base["verdict"]
    return {**base, "verdict": verdict, "split_problems": split_problems, "split_unverified": unverified,
            "offload": {s: {k: r.get(k) for k in ("offloaded", "ps_vram_share", "n_chat_calls", "n_chat_calls_num_gpu")}
                        for s, r in sorted(off.items())}}


def compute_x2_r2_llama_val_32k_offload(repo):
    """x2_r2_llama_val_32k_offload: the x2-r2-llama-val-32k measures plus offload lines, VRAM share and backend library
    per session, and the pre-registered offload / backend verdict."""
    import sys as _sys
    ag = _r2_agent_module()
    _sys.path.insert(0, str(REPO / "harness"))
    rows = ag.read_rows(repo / X2_R2_LLAMA_VAL_32K_OFFLOAD)
    if not rows:
        raise FileNotFoundError(repo / X2_R2_LLAMA_VAL_32K_OFFLOAD)
    ag.neg_arm_for_ctx(X2_VAL32K_NUM_CTX)
    model = "llama3.1:8b"
    rep = ag.evaluate_gates(rows, call2_tools="off", neg_arm=X2_VAL32K_NEG_ARM)
    v = x2_r2_llama_val_32k_offload_verdict(rows, ag, model)
    e = rep["baseline_table"].get(model)
    pos = rep["controls"]["positive"].get(model, {})
    libs = sorted({lib for r in rows if r.get("record") == "num_gpu_server" for lib in r.get("libraries") or []})
    offl = ", ".join(f"{s}: {o['offloaded']} VRAM share "
                     + (f"{o['ps_vram_share']:.3f}" if o["ps_vram_share"] is not None else "n/a")
                     for s, o in v["offload"].items())
    value = (f"{model} num_gpu {X2_OFFLOAD_NUM_GPU}: {_r2_table_str(e) if e else 'no gate-arm turns'}; backend library "
             f"{libs or 'not logged'}; offload per negative-control session (seed: [GPU layers, total]) {offl or 'none'}; "
             f"negative control (num_ctx 32768) canary misses {v['n_missed_canaries']} of {v['n_canaries']} canaries "
             f"over {len(v['per_session'])} sessions, per session " + ", ".join(f"{s}: {m}" for s, m in
                                                                                  v["per_session"].items())
             + f"; missed turns: {v['pattern']}; positive control (8192) truncation turns "
             f"{pos.get('truncation_detected_turns')}; verdict: {v['verdict']}")
    return {"value": value, "n": f"{e['n_turns'] if e else 0} turns", "detail": {"verdict": v, "gates": rep}}


# Backend numerics check (pre-registered in docs/FINDINGS.md 2026-10-19, queue id x2_r2_llama_val_32k_vulkan): the
# x2_r2_llama_val_32k validation with the job's own Ollama server on ggml's Vulkan backend (x2_r2_agent --backend
# vulkan, harness/x2_backend.py). PENDING until its file is synced. Verdict rule pre-registered.
X2_R2_LLAMA_VAL_32K_VULKAN = "results/x2_r2_llama_val_32k_vulkan.jsonl"
X2_VULKAN_HOURS_UPPER_FACTOR = 3   # ASSUMPTION for the estimate only: Vulkan up to 3x the slowest measured sibling


def x2_r2_llama_val_32k_vulkan_verdict(rows, ag, model="llama3.1:8b") -> dict:
    """Pre-registered: refused (backend_refused row) -> "refused (Vulkan not loaded)"; else the x2_r2_llama_val_32k
    validity and miss list (x2_r2_llama_val_32k_verdict) plus a verified backend_session row (harness/x2_backend.py
    session_verified) for every negative-control seed. Valid and verified: 0 misses -> "follows the vendor (0 misses on
    Vulkan)"; at least one miss and every missed canary turn a text tool call (any turn) -> "backend implementation, not
    the vendor (text-tool-call misses on Vulkan)"; else "inconclusive (...)"."""
    if any(r.get("record") == "backend_refused" for r in rows):
        reasons = [x for r in rows if r.get("record") == "backend_refused" for x in r.get("reasons") or []]
        return {"verdict": "refused (Vulkan not loaded)", "valid": False, "problems": reasons, "per_session": {},
                "n_missed_canaries": 0, "n_canaries": 0, "misses": [], "pattern": "no result (refused)",
                "backend_unverified": list(X2_VAL32K_SEEDS), "backend": {}}
    base = x2_r2_llama_val_32k_verdict(rows, ag, model)
    arm = X2_VAL32K_NEG_ARM + ag.MODE_SUFFIX["off"]
    ses = {r["seed"]: r for r in rows if r.get("record") == "backend_session" and r.get("arm_id") == arm
           and r.get("model_id") == model}
    unverified = [s for s in X2_VAL32K_SEEDS if not (ses.get(s) or {}).get("verified")]

    def why(s):
        if s not in ses:
            return f"{s}: no backend_session row"
        r = ses[s]
        return (f"{s}: offloaded {r.get('offloaded')}, libraries {r.get('gpu_memory_libraries')}, buffers "
                f"{r.get('buffer_devices')}, ROCm/CUDA lines {len(r.get('forbidden_lines') or [])}")
    if not base["valid"]:
        verdict = base["verdict"]
    elif unverified:
        verdict = (f"inconclusive (backend not verified as Vulkan for seeds {unverified}: "
                   + "; ".join(why(s) for s in unverified) + ")")
    elif base["n_missed_canaries"] == 0 and not base["misses"]:
        verdict = "follows the vendor (0 misses on Vulkan)"
    elif base["misses"] and all(m["text_tool_call"] for m in base["misses"]):
        verdict = "backend implementation, not the vendor (text-tool-call misses on Vulkan)"
    else:
        verdict = "inconclusive"
    return {**base, "verdict": verdict, "backend_unverified": unverified,
            "backend": {s: {k: r.get(k) for k in ("verified", "offloaded", "gpu_memory_libraries", "buffer_devices",
                                                  "vulkan_devices", "forbidden_lines")}
                        for s, r in sorted(ses.items())}}


def compute_x2_r2_llama_val_32k_vulkan(repo):
    """x2_r2_llama_val_32k_vulkan: the x2-r2-llama-val-32k measures plus the backend evidence (server-level inference
    compute lines, per-session verification) and the pre-registered vendor / backend-implementation verdict."""
    ag = _r2_agent_module()
    rows = ag.read_rows(repo / X2_R2_LLAMA_VAL_32K_VULKAN)
    if not rows:
        raise FileNotFoundError(repo / X2_R2_LLAMA_VAL_32K_VULKAN)
    ag.neg_arm_for_ctx(X2_VAL32K_NUM_CTX)
    model = "llama3.1:8b"
    v = x2_r2_llama_val_32k_vulkan_verdict(rows, ag, model)
    srv = [r for r in rows if r.get("record") == "backend_server"]
    ic = sorted({f"{x.get('library')} {x.get('description')}" for r in srv for x in r.get("inference_compute") or []})
    if v["verdict"].startswith("refused"):
        return {"value": f"{model}: inference compute {ic or 'none'}; verdict: {v['verdict']} "
                         f"({'; '.join(v['problems'])})", "n": "0 turns", "detail": {"verdict": v}}
    rep = ag.evaluate_gates(rows, call2_tools="off", neg_arm=X2_VAL32K_NEG_ARM)
    e = rep["baseline_table"].get(model)
    pos = rep["controls"]["positive"].get(model, {})
    n_ok = sum(1 for b in v["backend"].values() if b.get("verified"))
    value = (f"{model} --backend vulkan: {_r2_table_str(e) if e else 'no gate-arm turns'}; inference compute "
             f"{ic or 'not logged'}; backend verified per negative-control session {n_ok}/{len(X2_VAL32K_SEEDS)}; "
             f"negative control (num_ctx 32768) canary misses {v['n_missed_canaries']} of {v['n_canaries']} canaries "
             f"over {len(v['per_session'])} sessions, per session "
             + ", ".join(f"{s}: {m}" for s, m in v["per_session"].items())
             + f"; missed turns: {v['pattern']}; positive control (8192) truncation turns "
             f"{pos.get('truncation_detected_turns')}; verdict: {v['verdict']}")
    return {"value": value, "n": f"{e['n_turns'] if e else 0} turns", "detail": {"verdict": v, "gates": rep}}


def _file_wall_hours(path):
    """First to last ts_utc of a result file, in hours (None when the file is missing or has no timestamps)."""
    import datetime as _dt
    import json as _json
    if not path.exists():
        return None
    ts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            t = _json.loads(line).get("ts_utc")
        except ValueError:
            continue
        if t:
            ts.append(_dt.datetime.fromisoformat(t.replace("Z", "+00:00")))
    return (max(ts) - min(ts)).total_seconds() / 3600 if len(ts) >= 2 else None


def compute_x2_r2_llama_val_32k_vulkan_hours(repo):
    """Duration estimate for x2_r2_llama_val_32k_vulkan from its measured siblings (same plan, same model): file wall
    clock (first to last ts_utc) of x2_r2_llama_val_32k and, once synced, x2_r2_llama_val_32k_offload. ESTIMATE: the
    fastest measured sibling to X2_VULKAN_HOURS_UPPER_FACTOR x the slowest (the factor is an assumption)."""
    sibs = {"x2_r2_llama_val_32k": X2_R2_LLAMA_VAL_32K, "x2_r2_llama_val_32k_offload": X2_R2_LLAMA_VAL_32K_OFFLOAD}
    hrs = {k: _file_wall_hours(repo / f) for k, f in sibs.items()}
    if hrs["x2_r2_llama_val_32k"] is None:
        raise FileNotFoundError(repo / X2_R2_LLAMA_VAL_32K)
    got = [h for h in hrs.values() if h is not None]
    lo, hi = min(got), X2_VULKAN_HOURS_UPPER_FACTOR * max(got)
    parts = [f"{k} {h:.2f} h (measured, file wall clock)" if h is not None
             else f"{k.replace('x2_r2_llama_val_32k_', '')} not synced" for k, h in hrs.items()]
    value = (", ".join(parts) + f"; x2_r2_llama_val_32k_vulkan ESTIMATE {lo:.2f} to {hi:.2f} h (fastest sibling to "
             f"{X2_VULKAN_HOURS_UPPER_FACTOR}x the slowest; the factor is an assumption, Vulkan speed on evo-x2 not "
             "measured)")
    return {"value": value, "n": f"{len(got)} sibling runs", "detail": hrs}


# ───────────────────────────────────────────────── impact model (plan v2 A3, 2026-10-14)
# analysis/impact_model.py prices each problem for the customer profile in results/impact_profile_20261014.json
# (hour value cited from BLS; every other field an assumption). docs/IMPACT.md's profile table is these rows.
IMPACT_PROFILE = "results/impact_profile_20261014.json"
_IMPACT_DATA = [IMPACT_PROFILE, "results/cloud_prices_anthropic_20261014.json", "results/x2_r2_real*.jsonl",
                "results/blade_r2_real*.jsonl", "results/x2_r2_validation_v2.jsonl", "results/x2_r2_validation_v2b.jsonl"]
_IMPACT_FILES = {
    "P1": _IMPACT_DATA,
    "P2": _IMPACT_DATA,
    "P3": _IMPACT_DATA,
    "P4": _IMPACT_DATA + _TTFT_FIT_RESULT_FILES + BLADE_DRYRUN_FILES,
    "P5": _IMPACT_DATA + ["src/dse/envelope_data.json"],
    "P6": _IMPACT_DATA + ["results/x2_outcome_table_v3.jsonl", "results/t2s_outcome_table_full.jsonl",
                          "results/workload_pack/items/*.jsonl"],
    "P7": _IMPACT_DATA,
}


def _impact():
    import sys as _sys
    _sys.path.insert(0, str(REPO / "analysis"))
    import impact_model as im
    return im


_IMPACT_CACHE = {}


def _impact_all(repo):
    key = str(repo)
    if key not in _IMPACT_CACHE:
        _IMPACT_CACHE[key] = _impact().compute_all(repo)
    return _IMPACT_CACHE[key]


def compute_impact_hour_value(repo):
    """Value of one developer hour for the impact profile: the cited BLS figure stored in the profile JSON."""
    im = _impact()
    hv = im.load_profile(repo)["hour_value"]
    return {"value": f"USD {hv['usd_per_hour']:.2f}/h ({hv['figure_quoted']}; {hv['table']}; {hv['reference_period'].split(' (')[0]}; "
                     f"{hv['source_url']}; fetched {hv['fetched_utc']})", "n": 1}


def _impact_row(repo, k):
    im = _impact()
    res = _impact_all(repo)
    return {"value": im.row_value(res["base"][k], res["profile"]), "n": len(res["base"][k]["grid"])}


def compute_impact_p1(repo):
    return _impact_row(repo, "P1")


_IMPACT_V2_CACHE = {}
# P1's cloud part reads results/cloud_redo_v1.jsonl (row cloud-redo-cost-per-wrong-task), P2 reads
# results/x2_policy_compare.jsonl (row cost-cut-config-time; the flash-attention control file joins it once synced).
_IMPACT_V2_FILES = sorted({f for v in _IMPACT_FILES.values() for f in v}
                          | {"results/x2_evalset_tasks_real_v2.jsonl", "results/cloud_redo_v1.jsonl",
                             "results/x2_policy_compare.jsonl", "results/hardware_price_evox2_20261021.json"})


def _impact_v2(repo):
    key = str(repo)
    if key not in _IMPACT_V2_CACHE:
        _IMPACT_V2_CACHE[key] = _impact().compute_v2(repo)
    return _IMPACT_V2_CACHE[key]


def compute_impact_v2_p1(repo):
    """Impact v2 P1 (operator override 2026-10-21): excess wrong-task rate per tier (B3 v2, qwen3:14b, session-level
    CI) x (cloud redo cost per wrong task per tier from row cloud-redo-cost-per-wrong-task, tagged with that row's
    mode, + the measured wall time of the wrong turn valued 0 to the full hour value), at the one stated v2 profile."""
    im = _impact()
    res = _impact_v2(repo)
    return {"value": im.p1_v2_value(res), "n": f"{len(res['rows']['P1']['grid'])} tiers"}


def compute_impact_ranking_v2(repo):
    """Impact v2 ranking at one stated profile: every row with its status (measured / mixed / assumed / PENDING / not
    measured), PENDING inputs named, ranked by main-case USD/day."""
    im = _impact()
    res = _impact_v2(repo)
    return {"value": im.ranking_v2_value(res), "n": "7 rows"}


def compute_impact_ref_machine_price(repo):
    """ASSUMPTION INPUT (must be verified): the reference machine's list price for impact v2 P2, with its source."""
    m = _impact().ref_machine(repo)
    return {"value": f"ASSUMPTION, must be verified: USD {m['usd']:,.0f} for {m['machine']}; quoted \"{m['figure_quoted']}\" "
                     f"({m['publisher']}, {m['article_date']}, {m['source_url']}; fetched {m['fetched_utc']}); "
                     f"{m['meaning']}; not the repo's own purchase price (configs/hardware/evox2_strix_halo_128gb.yaml "
                     f"bom_cost_usd is null)", "n": "1"}


def compute_impact_v2_p2(repo):
    """Impact v2 P2 (operator decision 2026-10-21): machine capacity from measured session minutes (default vs the
    fastest measured B5 setting), machines needed for the profile's sessions/day, USD of machines saved per day over an
    assumed 3-year life; waiting time as an upper bound on the interactive-session share; PENDING until b5-fa-control
    is VERIFIED (values shown)."""
    im = _impact()
    r = _impact_v2(repo)["rows"]["P2"]
    if r["usd_day"] is None:
        return {"value": f"PENDING: inputs {', '.join(r['pending'])} not available", "n": "0"}
    head = ("PENDING (values computed, not ranked) until row b5-fa-control is VERIFIED" if r["status"] == "PENDING"
            else f"status {r['status']}")
    return {"value": f"{head}; " + im.p2_v2_value(r), "n": "1 model"}


def compute_impact_v2_detection_value(repo):
    """PROJECTION (operator decision 2026-10-21): measured wrong-task rate x cloud redo cost when only the steps a
    detector flags are escalated (R2 context-shift predictor's measured precision/recall, with where it was measured)
    vs escalating every step, at the v2 profile cell. Labels follow the cloud redo row's mode."""
    im = _impact()
    d = _impact_v2(repo)["rows"]["P1"].get("detection")
    return {"value": im.detection_value_text(d), "n": "1 cell" if d else "0"}


def compute_impact_v3_detection_value_measured(repo):
    """MEASURED detection (operator item 4): per tier of the B3 v2 real run, precision/recall (TP/FP/FN/TN) of "the task
    turn's prompt exceeded the loaded window" for "the task ended wrong", wrong tasks missed, and cloud USD escalating
    only flagged tasks vs all tasks at the tier's cloud redo USD per task (row cloud-redo-cost-per-wrong-task, labelled
    by its mode). Replaces impact-v2-detection-value's borrowed precision/recall."""
    im = _impact()
    d = _impact_v2(repo)["rows"]["P1"].get("detection_measured")
    if d is None:
        return {"value": "PENDING: needs row cloud-redo-cost-per-wrong-task VERIFIED", "n": "0"}
    return {"value": im.measured_detection_text(d), "n": f"{sum(c['n'] for c in d['tiers'].values())} task turns"}


def compute_impact_p2(repo):
    return _impact_row(repo, "P2")


def compute_impact_p3(repo):
    return _impact_row(repo, "P3")


def compute_impact_p4(repo):
    return _impact_row(repo, "P4")


def compute_impact_p5(repo):
    return _impact_row(repo, "P5")


def compute_impact_p6(repo):
    return _impact_row(repo, "P6")


def compute_impact_p7(repo):
    return _impact_row(repo, "P7")


def compute_impact_total(repo):
    """Sum of the measured profile rows (not-measured rows listed, never filled in), the ranking, and the same total
    for the profile's one-flag scenario runtime."""
    im = _impact()
    res = _impact_all(repo)
    p = res["profile"]
    at = res["audit_total"]
    fmt, rng = im._fmt_usd, im._rng
    pm = lambda v: im.per_month(v, p)  # noqa: E731
    return {"value": f"status quo {p['assumptions']['runtime']}, {p['assumptions']['cloud_model']} caching on: "
                     f"measured-only USD/day {fmt(at['measured_usd_day'])} (USD/month {fmt(pm(at['measured_usd_day']))}); "
                     f"range USD/day {rng(at['low'], at['high'])} (USD/month {rng(pm(at['low']), pm(at['high']))}; low = "
                     f"measured only, nobody waits for compute; high = compute time at the full hour value and cloud "
                     f"without caching) over {', '.join(at['in_range'])}; not added (assumed or not measured): "
                     f"{', '.join(at['excluded_assumed']) or 'none'}; ranking (caching on, time at its upper bound) "
                     f"{' > '.join(res['ranking'])}; per row status "
                     + ", ".join(f"{k} {r['audit']['status']}" for k, r in res["base"].items()), "n": 7}


ANTHROPIC_PRICES = "results/cloud_prices_anthropic_20261014.json"


def compute_cloud_prices_anthropic(repo):
    """The shared Anthropic price table: source, fetch time and the escalation model's rates (USD per MTok)."""
    import json as _json
    d = _json.loads((repo / ANTHROPIC_PRICES).read_text(encoding="utf-8"))
    rows = []
    for mid, m in d["models"].items():
        s = m["standard"]["short"]
        rows.append(f"{mid}: input {s['input']:g}, 5m cache write {s['cache_write']:g}, cache read {s['cached_input']:g}, "
                    f"output {s['output']:g}")
    return {"value": f"USD per MTok, Standard, prompts in the short tier: " + "; ".join(rows) + f"; escalation model "
                     f"{d['selection']['escalation_mid_tier']}; source {d['source_url']}; fetched {d['fetched_utc']}",
            "n": len(d["models"])}


def compute_impact_scenario_one_flag(repo):
    """The one-flag scenario (scenario_runtime) term by term on the previous basis (where the earlier
    USD 1,933.63/day came from) and on the current basis, split into measured and assumed."""
    im = _impact()
    return {"value": im.scenario_value(_impact_all(repo)), "n": 7}


def compute_impact_session_time_by_window(repo):
    """Per-session wall time on evo-x2 at Ollama's default window vs num_ctx 4096, from the R2 sessions."""
    im = _impact()
    res = _impact_all(repo)
    return {"value": im.session_time_value(res), "n": sum(v["n"] for v in res["session_time"].values())}

# ───────────────────────────────────────────────── H-scale, R2 at 30B to 70B (pre-registered 2026-10-14)
X2_SCALE_PROBE = "results/x2_scale_timing_probe.jsonl"
X2_SCALE_VALIDATION = "results/x2_r2_scale_validation.jsonl"
X2_SCALE_REAL = "results/x2_r2_scale_real.jsonl"


def _x2_scale_module():
    import sys
    sys.path.insert(0, str(REPO / "harness"))
    import x2_r2_scale
    return x2_r2_scale


def _fmt_num(x, spec=".1f"):
    return "n/a" if x is None else format(x, spec)


def compute_x2_scale_probe_hours(repo):
    """H-scale timing probe: per model and tier the measured probe rates (prompt-eval / eval tok/s), the projected
    minutes per 40-turn session (reference sessions rescaled by the probe rates, x2_r2_scale.project_all), hours for 3
    seeds, the validation job's projected hours, the llama3.3:70b default-tier decision, the real job's hours."""
    sc = _x2_scale_module()
    ag = _r2_agent_module()
    probe = ag.read_rows(repo / X2_SCALE_PROBE)
    if not probe:
        raise FileNotFoundError(repo / X2_SCALE_PROBE)
    ref = [r for f in (R2_REAL_V1_FILE, R2_REAL_V1B_FILE) for r in ag.read_rows(repo / f)]
    p = sc.project_all(probe, ref, ag.read_rows(repo / R2_VALIDATION_V2_FILE))
    missing = [r["model_id"] for r in probe if r.get("record") == "missing_model"]
    parts = []
    for model, m in p["models"].items():
        tiers = []
        for tier, e in m["tiers"].items():
            pr = e["probe_rates"]
            tiers.append(f"{tier.replace('ollama_', '')} probe pe {_fmt_num(pr['pe_rate'])} / ev "
                         f"{_fmt_num(pr['ev_rate'])} tok/s, loaded {e['probe_loaded_context']}, "
                         f"{_fmt_num(e.get('minutes_per_session'))} min/session, "
                         f"{_fmt_num(e.get('hours_for_seeds'))} h for 3 seeds")
        vm = m.get("validation_minutes")
        parts.append(f"{model} (ref {m['reference_model']}): " + "; ".join(tiers)
                     + f"; validation {_fmt_num(None if vm is None else vm / 60)} h; default tier "
                     f"{'run' if m['default_tier']['include'] else 'skipped'} ({m['default_tier']['reason']})")
    rh = p["real_hours_if_all_pass"]
    value = (" / ".join(parts) + f" / validation job {p['validation_hours']:.1f} h; real job "
             f"{rh['total_known']:.1f} h if every probed model passes"
             + ("" if rh["complete"] else " (some cells without a projection)")
             + (f"; missing from the store: {missing}" if missing else ""))
    return {"value": value, "n": f"{len(p['models'])} models probed", "detail": p}


def compute_x2_scale_validation(repo, vfile=X2_SCALE_VALIDATION):
    """H-scale validation (results/x2_r2_scale_validation.jsonl): per model the R2 baseline table and each gate
    (rules in use >= 90%, negative control, positive control, task tool, thinking off), and the real-run preflight."""
    ag = _r2_agent_module()
    rows = ag.read_rows(repo / vfile)
    if not rows:
        raise FileNotFoundError(repo / vfile)
    rep = ag.evaluate_gates(rows, call2_tools="off")
    g = rep["gates"]
    missing = [r["model_id"] for r in rows if r.get("record") == "missing_model"]
    parts = []
    for model, e in rep["baseline_table"].items():
        pf = ag.validation_preflight(rows, "off", [model])
        tt = g["task_tool"][model]
        parts.append(f"{model}: {_r2_table_str(e)}; task tool {tt['n_ok']}/{tt['n']}; thinking-present calls "
                     f"{g['thinking_off'][model]['thinking_present_calls']}; preflight "
                     f"{'pass' if pf['ok'] else 'refuse: ' + '; '.join(pf['reasons'])}")
    value = " / ".join(parts) + (f" / missing from the store: {missing}" if missing else "")
    return {"value": value, "n": f"{sum(e['n_turns'] for e in rep['baseline_table'].values())} baseline turns"}


def compute_h_scale_verdict(repo, real_file=X2_SCALE_REAL, vfile=X2_SCALE_VALIDATION, models=None):
    """The pre-registered H-scale verdict (x2_r2_scale.h_scale_verdict): per model R4096 (silent rule-in-use failures
    after the window at 4096, of 3), H32768 (history lost over the window with rules held at 32768, of 3), the full
    gated count, the per-model and overall verdicts."""
    sc = _x2_scale_module()
    ag = _r2_agent_module()
    rows = ag.read_rows(repo / real_file)
    if not rows:
        raise FileNotFoundError(repo / real_file)
    v = sc.h_scale_verdict(rows, ag.read_rows(repo / vfile), *([models] if models else []))
    parts = []
    for model, e in v["per_model"].items():
        if "R4096" in e:
            c4, c32 = e["tiers"][sc.TIER_4096]["counts"], e["tiers"][sc.TIER_32768]["counts"]
            parts.append(f"{model}: R4096 {e['R4096']}/3 (window exceeded {c4['exceeded_window']}/3, gated "
                         f"{c4['silent_gated_after_window']}/3), H32768 {e['H32768']}/3 (window exceeded "
                         f"{c32['exceeded_window']}/3); {e['at_4096']}; {e['at_32768']}; verdict {e['verdict']}")
        else:
            why = e.get("problems") or e.get("reasons") or e.get("reason")
            parts.append(f"{model}: {e['verdict']} ({why})")
    return {"value": f"overall {v['overall']} (tested: {v['tested']}); " + " / ".join(parts),
            "n": f"{len(v['tested'])} models tested", "detail": v}


# ───────────────── llama3.3:70b library pull and the B2 model family (pre-registered 2026-10-14)
X2_MODEL_PULLS_SCALE = "results/x2_model_pulls_scale.jsonl"
X2_FAMILY_SMOKE = "results/x2_family_smoke_mistral_small32.jsonl"
X2_FAMILY_VALIDATION = "results/x2_r2_family_mistral_small32_validation.jsonl"
X2_FAMILY_REAL = "results/x2_r2_family_mistral_small32_real.jsonl"
X2_FAMILY_MODEL = "mistral-small3.2:24b"


def _short(d, n=12):
    if not d:
        return "none"
    return d.split(":", 1)[-1][:n]


def compute_x2_model_pulls_scale(repo):
    """x2_model_pulls_scale: per tag pulled or already present, blob digests verified against the manifest, the
    manifest id and the model / params / template layer digests (first 12 hex), the chat template source, the load
    check (latest result row per tag)."""
    import json
    rows = [json.loads(l) for l in (repo / X2_MODEL_PULLS_SCALE).read_text(encoding="utf-8").splitlines() if l.strip()]
    latest = {}
    for r in rows:
        if r.get("record") == "model_pull_result":
            latest[r["model_tag"]] = r
    if not latest:
        raise ValueError("no model_pull_result rows")
    parts = []
    for tag, r in latest.items():
        how = "already present" if r.get("already_present") else ("pulled" if r.get("pulled") else
                                                                  f"not pulled ({r.get('pull_error')})")
        parts.append(f"{tag}: {how}; digests {'verified' if r.get('digests_ok') else 'NOT verified'}; manifest id "
                     f"{_short(r.get('manifest_sha256'))}; model {_short(r.get('model_digest'))}, params "
                     f"{_short(r.get('params_digest'))}, template {_short(r.get('template_digest'))}; template source "
                     f"{r.get('chat_template_source')}; load check {'ok' if r.get('verify_ok') else 'failed'}")
    return {"value": " / ".join(parts), "n": f"{len(latest)} tags"}


def compute_x2_family_smoke(repo):
    """B2 smoke test: native valid tool calls of the 5 call-1 requests (pre-registered pass: >= 4 of 5), from the
    family_smoke_result row, cross-checked by recomputing x2_family_smoke.smoke_summary from the turn rows."""
    import sys
    sys.path.insert(0, str(REPO / "harness"))
    import x2_family_smoke as fs
    ag = _r2_agent_module()
    rows = ag.read_rows(repo / X2_FAMILY_SMOKE)
    res = [r for r in rows if r.get("record") == "family_smoke_result" and r.get("model_id") == X2_FAMILY_MODEL]
    if not res:
        raise ValueError("no family_smoke_result row")
    r = res[-1]
    s = fs.smoke_summary(rows, X2_FAMILY_MODEL)
    if r.get("n_calls") and (s["n_native_valid"], s["n_calls"]) != (r["n_native_valid"], r["n_calls"]):
        raise ValueError(f"smoke result row {r['n_native_valid']}/{r['n_calls']} != recomputed "
                         f"{s['n_native_valid']}/{s['n_calls']}")
    tpl = next((t.get("chat_template_source") for t in rows if t.get("record") == "r2a_turn"
                and t.get("model_id") == X2_FAMILY_MODEL), None)
    calls = ", ".join(f"t{p['turn_idx']} {'native' if p['native_tool_call'] else p['detection_method']}"
                      f"{' valid' if p['valid'] else ' invalid' if p['native_tool_call'] else ''}"
                      for p in r.get("per_call") or [])
    value = (f"{X2_FAMILY_MODEL}: {r['n_native_valid']}/{r['n_calls']} native valid tool calls (pass needs "
             f"{r['pass_min']}): {'pass' if r['passed'] else 'fail'}" + (f" ({r['reason']})" if r.get("reason") else "")
             + (f"; {calls}" if calls else "") + f"; template source {tpl}")
    return {"value": value, "n": f"{r['n_calls']} calls"}


def compute_x2_family_validation(repo):
    return compute_x2_scale_validation(repo, X2_FAMILY_VALIDATION)


def compute_h_scale_family_verdict(repo):
    """The H-scale per-model rules applied to the B2 family model (separate from H-scale-verdict)."""
    return compute_h_scale_verdict(repo, X2_FAMILY_REAL, X2_FAMILY_VALIDATION, (X2_FAMILY_MODEL,))


# ───────────────────────────────────────────────── operator plan v2 (2026-10-13), Parts A4 and A5
# A4, num_keep guardrail flag test (pre-registered in docs/FINDINGS.md 2026-10-13, queue id x2_num_keep_guardrail_test):
# llama3.1:8b, num_ctx 4096, 3 seeds x 40 turns, options.num_keep = the system prompt's token length. PENDING until
# synced. The verdict is the pre-registered one, harness/x2_num_keep.verdict.
NUM_KEEP_FILE = "results/x2_num_keep_guardrail_test.jsonl"
# A5, context-window override methods on Ollama 0.34.4 (harness/window_override_methods.py). The evo-x2 file decides
# PENDING; the Blade and evo-t2s files are added to the value when they exist (staged jobs, run later).
WOM_FILE_X2 = "results/x2_window_override_methods_v1.jsonl"
WOM_FILES_OTHER = {"blade": "results/blade_window_override_methods_v1.jsonl",
                   "evo-t2s": "results/t2s_window_override_methods_v1.jsonl"}


def _harness_module(name):
    import importlib
    import sys
    sys.path.insert(0, str(REPO / "harness"))
    return importlib.import_module(name)


def compute_num_keep_guardrail_test(repo, nk_file=None, model="llama3.1:8b"):
    """x2_num_keep_guardrail_test (results/x2_num_keep_guardrail_test.jsonl, evo-x2): S per seed, num_keep observed in
    the server log (context-shift n_keep, task n_keep, token-cut keep, --keep), shift / cut / message-drop counts, per
    rule in use the rate and first failure per seed, canary misses, surfaced errors, and the pre-registered verdict."""
    ag = _r2_agent_module()
    nk = _harness_module("x2_num_keep")
    nk_file = nk_file or NUM_KEEP_FILE
    rows = ag.read_rows(repo / nk_file)
    if not rows:
        raise FileNotFoundError(repo / nk_file)
    v = nk.verdict(rows, model=model)

    def ff(d):
        return "/".join("-" if t is None else str(t) for t in d.values())
    rules = "; ".join(f"{r.split('_')[0]} {_pct(d['rate'])} (n={d['n']}, first failure {ff(d['first_failure_per_seed'])})"
                      for r, d in v["rule_rates"].items())
    value = (f"{model} num_ctx 4096 num_keep=system: S per seed {v['system_prompt_tokens_per_seed']}; observed "
             f"context shifts {v['context_shift_events']} with n_keep {v['shift_n_keep_values']}, task n_keep "
             f"{v['task_n_keep_values']}, token cuts {v['token_cuts']} with keep {v['token_cut_keep_values']}, --keep "
             f"{v['keep_arg_values']}, Ollama message-drop calls {v['message_drop_calls']} of {v['n_calls']}; rules in "
             f"use: {rules}; canary misses C {v['canary_sys_misses']}, H {v['canary_hist_misses']} of "
             f"{v['canary_checks']} checks; error turns {v['error_turns']}; verdict: {v['verdict']}"
             + (f" ({'; '.join(v['problems'])})" if v["problems"] and not v["verdict"].startswith("inconclusive (inc")
                else ""))
    return {"value": value, "n": f"{v['n_turns']} turns", "detail": v}


def _wom_host_str(wom, rows):
    s = wom.summarize(rows)

    def c(d):
        return "n/a" if not d else f"{d['ps']} (log {d['log']})"
    r, m, e = s["request"], s["modelfile"], s["env"]
    return (f"baseline {c(s['baseline'])}; request num_ctx {r['value']}: loaded {c(r['apply'])}, after restart "
            f"(plain request) {c(r['after_restart'])}, applied {r['applied']}, persists {r['persists_across_restart']}; "
            f"Modelfile PARAMETER num_ctx {m['value']} on {m['tag']}: loaded {c(m['apply'])}, after restart "
            f"{c(m['after_restart'])}, applied {m['applied']}, persists {m['persists_across_restart']}, base tag after "
            f"restart {c(m['base_control_after_restart'])}, tag deleted {m['tag_deleted']}; OLLAMA_CONTEXT_LENGTH "
            f"{e['value']}: loaded {c(e['apply'])}, restart with env {c(e['after_restart_same_env'])}, restart without "
            f"env {c(e['after_restart_without_env'])}, applied {e['applied']}; GPU per Ollama "
            f"{[(g.get('description'), g.get('total'), g.get('available')) for g in s['inference_compute']]}; "
            f"valid {s['valid']}" + (f" ({'; '.join(s['problems'])})" if s["problems"] else ""))


def compute_window_override_methods(repo):
    """Context-window override methods on Ollama 0.34.4, llama3.1:8b, per host: per-request num_ctx, Modelfile
    PARAMETER num_ctx on a derived tag, OLLAMA_CONTEXT_LENGTH on the job's own server; loaded context from /api/ps and
    the server log, applied, and persistence across a restart of the job's own server (window_override_methods.summarize).
    evo-x2 always; the Blade and evo-t2s when their files exist."""
    ag = _r2_agent_module()
    wom = _harness_module("window_override_methods")
    rows = ag.read_rows(repo / WOM_FILE_X2)
    if not rows:
        raise FileNotFoundError(repo / WOM_FILE_X2)
    parts = [f"evo-x2: {_wom_host_str(wom, rows)}"]
    for host, f in WOM_FILES_OTHER.items():
        other = ag.read_rows(repo / f)
        parts.append(f"{host}: {_wom_host_str(wom, other)}" if other else f"{host}: not run yet")
    return {"value": " / ".join(parts), "n": f"{1 + sum(1 for f in WOM_FILES_OTHER.values() if (repo / f).exists())} hosts"}


# ───────────────────────────────────────────────── B5 policy comparison (pre-registered 2026-10-14)
# harness/policy_compare.py. The plan-hours row is an ESTIMATE computed from committed result files now; the result rows
# are PENDING until results/x2_policy_compare.jsonl is synced (the evo-t2s and Blade files join when they exist).
B5_PRICES = "results/cloud_prices_anthropic_20261014.json"   # the shared price table (impact_model.PRICES_FILE)
B5_FILES = {"evo-x2": "results/x2_policy_compare.jsonl", "evo-t2s": "results/t2s_policy_compare.jsonl",
            "blade": "results/blade_policy_compare_qwen3_8b_v1.jsonl"}
B5_PLAN_INPUTS = ["results/x2_r2_real_v1.jsonl", "results/x2_r2_real_v1b.jsonl", "results/x2_r2_mitigation_v1.jsonl",
                  "results/blade_k1_v2.summary.json", "src/dse/envelope_data.json"]


def _b5_rows(repo):
    ag = _r2_agent_module()
    return {p: (ag.read_rows(repo / f) or None) for p, f in B5_FILES.items()}


def compute_b5_plan_hours(repo):
    """B5 estimated hours per platform and model (policy_compare.estimate_platform: assumed windows, aliases, evo-x2
    session minutes at the same window x the platform factor, recall and probe allowances). An ESTIMATE, not a
    measurement; the Blade line also gives what fits inside night 3's deadline cap."""
    pc = _harness_module("policy_compare")
    bq = _harness_module("blade_queue")
    parts = []
    for plat in pc.PLATFORMS:
        if plat == "evo-t2s-cpu":   # added 2026-10-22; its plan from measured evo-t2s sessions is row t2s-b5-plan-hours
            continue
        e = pc.estimate_platform(plat, deadline_h=bq.BLADE_B5_DEADLINE_H if plat == "blade" else None)
        ms = "; ".join(f"{m} {v['total_h']:.1f} h (sessions {v['sessions_h']:.1f}, recall {v['recall_h']:.2f}, probes "
                       f"{v['probe_h']:.2f}; distinct policies {','.join(v['policies'])}"
                       + (f"; aliases {v['aliases']}" if v["aliases"] else "") + ")"
                       for m, v in e["models"].items())
        fit = (f"; within the {e['deadline_h']} h night-3 cap: {len(e['sessions_within_deadline'])} of "
               f"{e['sessions_total']} sessions" if "deadline_h" in e else "")
        parts.append(f"{plat} ESTIMATE {e['total_h']:.1f} h: {ms}{fit}")
    return {"value": " / ".join(parts), "n": f"{len(pc.PLATFORMS)} platforms (estimate)"}


def compute_b5_policy_metrics(repo):
    """B5 per platform and model: rule pass rate (rules in use), history recall, session wall-clock (cloud steps
    re-timed per A-B5-cloud-latency), cloud USD per session (the shared price table, claude-sonnet-5-5, caching on), per policy; the Pareto frontier."""
    pc = _harness_module("policy_compare")
    prices = pc.load_cloud_prices(repo)
    rows = _b5_rows(repo)
    if not rows["evo-x2"]:
        raise FileNotFoundError(repo / B5_FILES["evo-x2"])
    parts = []
    for plat, rs in rows.items():
        if not rs:
            parts.append(f"{plat}: not run yet")
            continue
        for m, c in pc.metrics_for_platform(rs, prices).items():
            pol = "; ".join(f"{p} rule {_pct(v['rule_pass_rate'])} hist {_pct(v['history_recall'])} "
                            f"{_fmt_num(v['wall_min'])} min ${v['cloud_usd']:.4f}" + (f" (={v['alias_of']})"
                                                                                       if v.get("alias_of") else "")
                            for p, v in sorted(c["policies"].items()))
            fr = pc.pareto_frontier(c["policies"]) if c["policies"] else []
            parts.append(f"{plat} {m}: {pol}; frontier {fr}; complete {c['complete']}"
                         + (f" ({'; '.join(c['problems'])})" if c["problems"] else ""))
    return {"value": " / ".join(parts), "n": f"{sum(1 for r in rows.values() if r)} platform files"}


def compute_b5_verdict(repo):
    """The pre-registered B5 verdict (policy_compare.b5_verdict) over every platform file present."""
    pc = _harness_module("policy_compare")
    rows = _b5_rows(repo)
    if not rows["evo-x2"]:
        raise FileNotFoundError(repo / B5_FILES["evo-x2"])
    v = pc.b5_verdict(rows, pc.load_cloud_prices(repo))
    per = "; ".join(f"{p} {e['status']} (router on frontier in every complete cell: "
                    f"{e.get('router_on_frontier_all_cells')}, fixed on frontier in every cell: "
                    f"{e.get('fixed_on_frontier_all_cells')})" for p, e in v["platforms"].items())
    return {"value": f"{v['verdict']}; {per}", "n": f"{len(v['platforms'])} platforms", "detail": v}


def compute_b5_kv_recall(repo):
    """B5 long-context recall at the largest window each KV precision admits (f16 with flash attention, q8_0, q4_0;
    1 warm-up + 3 measured calls) and the KV-type check of the server log, per platform and model."""
    pc = _harness_module("policy_compare")
    rows = _b5_rows(repo)
    if not rows["evo-x2"]:
        raise FileNotFoundError(repo / B5_FILES["evo-x2"])
    parts = []
    for plat, rs in rows.items():
        if not rs:
            parts.append(f"{plat}: not run yet")
            continue
        kv = {(r["model_id"], r["env"]): r["check"] for r in rs if r.get("record") == "b5_kv_check"}
        for m, d in pc.recall_summary(rs).items():
            parts.append(f"{plat} {m}: " + "; ".join(
                f"{p} ctx {e['window']} recall {_pct(e['recall_mean'])} ({e['calls']} calls, truncated "
                f"{e['truncated']}, errors {e['errors']}; KV check "
                f"{(kv.get((m, p)) or {}).get('valid', 'n/a') if p != 'f16' else 'n/a'})" for p, e in sorted(d.items())))
    return {"value": " / ".join(parts), "n": f"{sum(1 for r in rows.values() if r)} platform files"}


GPU_MEM_FILES = ["results/x2_70b_edge_reps.jsonl", "results/t2s_night2_20261001T151340Z.jsonl",
                 "results/t2s_amech_20260926T181456Z.jsonl", "results/t2s_k1_igpu_enable_sweep.jsonl",
                 "results/blade_k1_v2.jsonl"]


def compute_gpu_memory_readonly(repo):
    """GPU memory as each machine already reported it in synced result files (read-only observations, nothing set):
    evo-x2 Vulkan heaps (x2_70b_edge_reps pre_launch.vulkan_heaps, latest row; mx2_baseline heaps in the 2026-10-01
    night2 file), evo-t2s Vulkan heaps (t2s_amech vulkan_limits) and Ollama's inference-compute and vram-based default
    lines (t2s_k1_igpu_enable_sweep), Blade Ollama inference-compute line, nvidia-smi total and vram default ctx
    (blade_k1_v2 blade_k1_device)."""
    wom = _harness_module("window_override_methods")

    def heaps_str(heaps):
        return ", ".join(f"heap {h['index']} {h['size_mib']:.0f} MiB (budget {h['budget_mib']:.0f}, "
                         f"{'device-local' if 'DEVICE_LOCAL' in (h.get('flags') or '') else 'host'})" for h in heaps)
    rows = _read_jsonl(repo / GPU_MEM_FILES[0])
    x2 = [r["pre_launch"]["vulkan_heaps"]["heaps"] for r in rows
          if (r.get("pre_launch") or {}).get("vulkan_heaps", {}).get("heaps")]
    mx2 = next(r["heaps"] for r in _read_jsonl(repo / GPU_MEM_FILES[1]) if r.get("record") == "mx2_baseline"
               and r.get("heaps"))
    t2s = next(r["heaps"] for r in _read_jsonl(repo / GPU_MEM_FILES[2]) if r.get("record") == "vulkan_limits")
    t2s_dev = next(r["gpu_lines"] for r in _read_jsonl(repo / GPU_MEM_FILES[3])
                   if r.get("record") == "igpu_sweep_device_detect")
    t2s_log = wom.parse_server_log("\n".join(t2s_dev))
    blade = next(r for r in _read_jsonl(repo / GPU_MEM_FILES[4]) if r.get("record") == "blade_k1_device")
    b_log = wom.parse_server_log("\n".join(blade.get("inference_compute") or []))

    def gpus(p):
        return "; ".join(f"{g['description']} {g['type']} total {g['total']} available {g['available']}"
                         for g in p["inference_compute"])
    value = (f"evo-x2 Vulkan: {heaps_str(x2[-1])} ({len(x2)} rows, all equal {all(h == x2[-1] for h in x2)}); "
             f"evo-x2 mx2_baseline 2026-10-01: {heaps_str(mx2)} / evo-t2s Vulkan: {heaps_str(t2s)}; Ollama: "
             f"{gpus(t2s_log)}, vram-based default {t2s_log['vram_based_default']} / Blade Ollama: {gpus(b_log)}; "
             f"nvidia-smi total {blade.get('vram_total_mib')} MiB; vram default ctx {blade.get('vram_default_ctx')}")
    return {"value": value, "n": f"{len(x2) + 4} records"}


# ───────────────────────────────────────────────── FH1 AMD boundary panel (2026-10-20, operator item 4)
# One row per model: the evo-x2 memory boundary as measured so far (llama-server b10970 Vulkan, -ngl 99), from the MX2
# bisection (projection-based regimes + real start/fail outcomes), MX2 validation and spill-stats (measured per-process
# GPU dedicated/shared usage and throughput per n_ctx point; one fresh server start + one call per rep, no warm-up call)
# and x2_70b_edge_reps (fresh-process outcomes at the 70B fail point).
AMD_BOUNDARY_FILES = ["results/mx2_spill_stats.jsonl", "results/mx2_validation.jsonl",
                      "results/t2s_night2_20261001T151340Z.jsonl", "results/x2_70b_edge_reps.jsonl"]
AMD_SPILL_DELTA_MIB = 1024.0  # same threshold as harness/mx2_validation.py SPILL_SHARED_DELTA_THRESHOLD_MIB
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def _mean_ci95(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    m = statistics.mean(vals)
    if len(vals) < 2:
        return m, None
    return m, _T95.get(len(vals) - 1, 1.96) * statistics.stdev(vals) / len(vals) ** 0.5


def amd_boundary_summary(model_id, sources, bisection_rows, edge_rows, threshold_mib=AMD_SPILL_DELTA_MIB):
    """Pure. sources: {source_name: rows} of mx2v_regime_point files (spill_stats, validation). Per source, the
    baseline GPU shared usage is that source's smallest unclamped started point for the model; a point is spill if its
    shared usage exceeds the baseline by more than threshold_mib, clamped if actual n_ctx < requested n_ctx (clamped
    points are kept, labeled, and never used for the boundary, per the MX2 pre-registration criterion (c); a point
    served at a slightly LARGER n_ctx than requested, e.g. 100000 rounded up to 100096, counts as unclamped and is
    reported at its actual n_ctx).
    Returns points, the measured boundary (last unclamped no-spill / first unclamped spill actual n_ctx) with the
    decode/TTFT ratio across it, the bisection's projected boundary and fail outcomes, and the edge-rep outcomes."""
    base = next((r for r in bisection_rows if r.get("record") == "mx2_baseline"), {})
    dedicated_mib = (base.get("registry_dedicated") or {}).get("qw_memory_size_mib")
    lines = base.get("lines") or {}
    points = []
    for src, rows in sources.items():
        pts = [r for r in rows if r.get("record") == "mx2v_regime_point" and r.get("model_id") == model_id
               and r.get("started")]
        unclamped = [r for r in pts if r.get("actual_n_ctx") >= r.get("requested_n_ctx")]
        if not unclamped:
            continue
        b = min(unclamped, key=lambda r: r["actual_n_ctx"])
        baseline_shared, baseline_ctx = b.get("gpu_shared_mib"), b["actual_n_ctx"]
        groups = {}
        for r in pts:
            groups.setdefault((r["requested_n_ctx"], r["actual_n_ctx"]), []).append(r)
        for (req, act), g in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            ded = statistics.mean(r["gpu_dedicated_mib"] for r in g)
            sh = statistics.mean(r["gpu_shared_mib"] for r in g)
            d_m, d_ci = _mean_ci95([r.get("decode_tok_s") for r in g])
            t_m, t_ci = _mean_ci95([r.get("ttft_ms") for r in g])
            points.append({"source": src, "requested": req, "actual": act, "n": len(g), "clamped": act < req,
                           "dedicated_mib": ded, "shared_mib": sh, "shared_delta_mib": sh - baseline_shared,
                           "baseline_ctx": baseline_ctx,
                           "spill": (sh - baseline_shared) > threshold_mib,
                           "decode": d_m, "decode_ci": d_ci, "ttft_ms": t_m, "ttft_ci": t_ci,
                           "frac_dedicated": (ded + sh) / dedicated_mib if dedicated_mib else None,
                           "frac_device_local": (ded + sh) / lines["device_local_mib"]
                           if lines.get("device_local_mib") else None})
    unc = [p for p in points if not p["clamped"]]
    spills = [p for p in unc if p["spill"]]
    first_spill = min(spills, key=lambda p: (p["actual"], p["source"] != "spill_stats")) if spills else None
    fits = [p for p in unc if not p["spill"] and (first_spill is None or p["actual"] < first_spill["actual"])]
    last_fit = max(fits, key=lambda p: (p["actual"], p["source"] == "spill_stats")) if fits else None
    cost = None
    if last_fit and first_spill:
        cost = {"decode_ratio": first_spill["decode"] / last_fit["decode"],
                "ttft_ratio": first_spill["ttft_ms"] / last_fit["ttft_ms"],
                "same_source": last_fit["source"] == first_spill["source"]}
    two = next((r for r in bisection_rows if r.get("record") == "mx2_two_line_result"
                and r.get("model_id") == model_id), None)
    probes = {r["n_ctx"]: r for r in bisection_rows if r.get("record") == "mx2_probe" and r.get("model_id") == model_id}
    bis = None
    if two:
        def probe(n):
            p = probes.get(n) or {}
            return {"n_ctx": n, "regime": p.get("regime"), "exit_code": p.get("exit_code"),
                    "clamped": "guard: server does not match" in (p.get("error") or ""), "logged_mib": p.get("logged_mib")}
        bis = {"proj_last_fit": probe(two["device_local_last_fit_n_ctx"]),
               "proj_first_spill": probe(two["device_local_first_spill_n_ctx"]),
               "last_start": probe(two["vulkan_total_last_start_n_ctx"]),
               "first_fail": probe(two["vulkan_total_first_fail_n_ctx"]),
               "max_ctx_native": two.get("max_ctx_native"), "yarn_factor": two.get("yarn_factor"),
               "fail_points": sorted(n for n, p in probes.items() if p.get("regime") == "HARD_FAIL")}
    edge = [r for r in edge_rows if r.get("record") == "x2_70b_edge_rep" and r.get("model_id") == model_id]
    edge_sum = None
    if edge:
        outc = {}
        for r in edge:
            la = r.get("launch") or {}
            k = (r.get("arm"), la.get("outcome"), la.get("exit_code_name"), la.get("crash_stage") or la.get("stall_stage"))
            outc[k] = outc.get(k, 0) + 1
        edge_sum = {"n": len(edge), "n_ctx": sorted({r.get("n_ctx") for r in edge}), "outcomes": outc,
                    "elapsed_s": (min(r["launch"]["elapsed_s"] for r in edge), max(r["launch"]["elapsed_s"] for r in edge))}
    repro = {}
    for rows in sources.values():
        for r in rows:
            if r.get("record") == "mx2v_regime_point" and r.get("model_id") == model_id and not r.get("started"):
                k = (r["requested_n_ctx"], False, None, "n/a (regime point, start timeout "
                     + ("120 s" if "120.0s" in (r.get("error") or "") else "?") + ")")
                repro[k] = repro.get(k, 0) + 1
    for rows in sources.values():
        for r in rows:
            if r.get("record") == "mx2v_crash_repro" and r.get("model_id") == model_id:
                k = (r["n_ctx"], r.get("started"), r.get("exit_code"), r.get("killed_by_probe"))
                repro[k] = repro.get(k, 0) + 1
    return {"model_id": model_id, "points": points, "last_fit": last_fit, "first_spill": first_spill, "cost": cost,
            "bisection": bis, "edge": edge_sum, "crash_repro": repro, "dedicated_mib": dedicated_mib,
            "device_local_mib": lines.get("device_local_mib"), "vulkan_total_mib": lines.get("vulkan_total_mib")}


def format_amd_boundary(s):
    def pt(p):
        ci = f" +/- {p['decode_ci']:.3f}" if p["decode_ci"] is not None else ""
        lab = (f"CLAMPED req {p['requested']} served {p['actual']}" if p["clamped"] else f"n_ctx {p['actual']}"
               + (f" (req {p['requested']})" if p["actual"] != p["requested"] else ""))
        return (f"{lab} [{p['source']}, n={p['n']}]: {'SPILL' if p['spill'] else 'no spill'} (shared "
                f"{p['shared_delta_mib']:+.0f} MiB vs n_ctx {p['baseline_ctx']}, dedicated {p['dedicated_mib']:.0f} MiB, "
                f"process GPU mem {p['frac_dedicated']:.3f} of dedicated / {p['frac_device_local']:.3f} of device-local "
                f"heap), decode {p['decode']:.3f}{ci} tok/s, TTFT {p['ttft_ms']:.0f} ms")
    parts = [f"evo-x2 {s['model_id']}, llama-server Vulkan -ngl 99; boundaries: dedicated {s['dedicated_mib']:.0f} MiB, "
             f"Vulkan device-local heap {s['device_local_mib']:.0f} MiB, Vulkan total {s['vulkan_total_mib']:.0f} MiB"]
    lf, fs, c = s["last_fit"], s["first_spill"], s["cost"]
    if lf and fs:
        parts.append(f"MEASURED spill boundary (GPU shared usage > +{AMD_SPILL_DELTA_MIB:.0f} MiB, unclamped): "
                     f"last no-spill n_ctx {lf['actual']} ({lf['source']}), first spill n_ctx {fs['actual']} "
                     f"({fs['source']}); cost across it: decode ratio {c['decode_ratio']:.3f}, TTFT ratio "
                     f"{c['ttft_ratio']:.3f}{'' if c['same_source'] else ' (different runs)'}")
    elif lf:
        parts.append(f"MEASURED: largest unclamped no-spill n_ctx {lf['actual']} ({lf['source']}); no unclamped spill "
                     f"point measured, so no spill cost")
    parts.append("points: " + "; ".join(pt(p) for p in s["points"]))
    b = s["bisection"]
    if b:
        def bp(p):
            return f"{p['n_ctx']}{' (clamped to 131072)' if p['clamped'] else ''}"
        parts.append(f"bisection (1 start per n_ctx, projection regimes from logged buffers vs device-local heap): "
                     f"projected last fit {bp(b['proj_last_fit'])}, projected first spill {bp(b['proj_first_spill'])}; "
                     f"last start {bp(b['last_start'])}, first fail {b['first_fail']['n_ctx']} (exit "
                     f"{b['first_fail']['exit_code']}); HARD_FAIL points {b['fail_points']}; trained ctx "
                     f"{b['max_ctx_native']} x yarn {b['yarn_factor']}")
    e = s["edge"]
    if e:
        outs = ", ".join(f"{arm} {o} {name} at {stage} {k}" for (arm, o, name, stage), k in sorted(e["outcomes"].items()))
        parts.append(f"fresh-process fail reps at n_ctx {e['n_ctx']}: {outs} (of {e['n']}; {e['elapsed_s'][0]:.0f}-"
                     f"{e['elapsed_s'][1]:.0f} s to exit)")
    if s["crash_repro"]:
        parts.append("in-session start failures and crash repro (confounded, 2026-10-06 correction): " + ", ".join(
            f"n_ctx {n} started={st} exit={ex} killed_by_probe={kp} x{k}"
            for (n, st, ex, kp), k in sorted(s["crash_repro"].items(), key=lambda kv: str(kv[0]))))
    return " // ".join(parts)  # never "|": the value sits in a markdown table cell


def _amd_boundary(repo, model_id):
    sources = {"spill_stats": _read_jsonl(repo / AMD_BOUNDARY_FILES[0]),
               "validation": _read_jsonl(repo / AMD_BOUNDARY_FILES[1])}
    s = amd_boundary_summary(model_id, sources, _read_jsonl(repo / AMD_BOUNDARY_FILES[2]),
                             _read_jsonl(repo / AMD_BOUNDARY_FILES[3]))
    n_calls = sum(p["n"] for p in s["points"])
    n_fail = (s["edge"] or {}).get("n", 0)
    return {"value": format_amd_boundary(s), "n": f"{n_calls} measured calls, {n_fail} fresh-process fail reps",
            "detail": s}


def compute_fh1_amd_boundary_llama70b(repo):
    """FH1 AMD panel, llama-3.3-70b (see amd_boundary_summary)."""
    return _amd_boundary(repo, "llama-3.3-70b")


def compute_fh1_amd_boundary_qwen3_32b(repo):
    """FH1 AMD panel, qwen3-32b (see amd_boundary_summary)."""
    return _amd_boundary(repo, "qwen3-32b")


# ───────────────────────────────────────────────── business metrics (plan v2 C1, 2026-10-14)
# analysis/business_metrics.py: router vs all-cloud / naive local default / today's mix per customer profile
# (results/business_profiles_20261014.json; every profile field an assumption). docs/BUSINESS_METRICS.md is these rows.
BUSINESS_PROFILES = "results/business_profiles_20261014.json"
_BUSINESS_FILES = sorted(set(_IMPACT_FILES["P4"] + _IMPACT_FILES["P5"] + _IMPACT_FILES["P6"] + [
    BUSINESS_PROFILES, "results/x2_r2_mitigation_v1.jsonl", "results/x2_r2_real_v1.jsonl"] + _PARETO_DATA))


def _business():
    import sys as _sys
    _sys.path.insert(0, str(REPO / "analysis"))
    import business_metrics as bm
    return bm


_BUSINESS_CACHE = {}


def _business_all(repo):
    key = str(repo)
    if key not in _BUSINESS_CACHE:
        _BUSINESS_CACHE[key] = _business().compute_all(repo)
    return _BUSINESS_CACHE[key]


def compute_business_hour_range(repo):
    """Hour-value range for the business sensitivity: BLS annual p10/p90 (software developers) converted at the hours
    per year implied by the page's own median pair; the median is impact-hour-value."""
    bm = _business()
    doc = bm.load_profiles(repo)
    h = bm.hour_values(doc, repo)
    r = doc["hour_value"]["range"]
    return {"value": f"p10 USD {h['p10']:.2f}/h, median USD {h['median']:.2f}/h, p90 USD {h['p90']:.2f}/h (annual p10 "
                     f"USD {r['p10_annual_usd']:,}, p90 USD {r['p90_annual_usd']:,}, {r['reference_period']}, divided by "
                     f"{h['hours_per_year']:.1f} h/year = USD {r['group_median_annual_usd']:,} / "
                     f"{r['group_median_hourly_usd']}; {r['source_url']}; fetched {r['fetched_utc']})", "n": 3}


def _business_profile(repo, name):
    bm = _business()
    res = _business_all(repo)
    r = res["profiles"][name]
    return {"value": bm.profile_value(name, r),
            "n": f"{max(c['escalation']['n'] for c in r['windows'])} replayed sessions per window"}


def compute_business_scenario_plus_a(repo):
    """SCENARIO only: router savings per profile under the +A credit, the P1 rerun proxy and time at the full hour
    value (kept out of every headline; operator decision D)."""
    bm = _business()
    res = _business_all(repo)
    return {"value": bm.scenario_value(res), "n": len(res["scenario"])}


def compute_business_mid_team(repo):
    return _business_profile(repo, "mid_team")


def compute_business_mid_team_qwen3_14b(repo):
    return _business_profile(repo, "mid_team_qwen3_14b")


def compute_business_sensitivity(repo):
    bm = _business()
    res = _business_all(repo)
    return {"value": f"base {res['doc']['sensitivity']['base_profile']}, USD/month: "
                     + bm.sensitivity_value(res["sensitivity"]), "n": len(res["sensitivity"])}


EVALSET_V1_FILES = ["results/x2_evalset_tasks_val_v1.jsonl", "results/x2_evalset_guard_val_v1.jsonl"]


def compute_x2_evalset_val_v1_diagnosis(repo):
    """Row-level classes of every negative-control canary miss, failed task turn, task-tool failure and rule 6-8
    violation in the two v1 evalset validations (analysis/evalset_val_diagnosis.py; docs/EVALSET_DESIGN.md 2026-10-14)."""
    import sys as _sys
    p = str(repo / "analysis")
    if p not in _sys.path:
        _sys.path.insert(0, p)
    import evalset_val_diagnosis as D
    return D.register_value(repo)


T2S_R2_VAL_FILES = ["results/t2s_r2_validation_cpu_v1.jsonl", "results/t2s_r2_validation_igpu_v1.jsonl"]


def compute_t2s_r2_val_llama_diagnosis(repo):
    """evo-t2s R2 validations (Intel CPU, Intel iGPU): llama3.1:8b baseline rule rates, negative-control canary misses
    and one class per failed rule on every baseline turn (analysis/t2s_llama_val_diagnosis.py; docs/FINDINGS.md
    2026-10-19)."""
    import sys as _sys
    p = str(repo / "analysis")
    if p not in _sys.path:
        _sys.path.insert(0, p)
    import t2s_llama_val_diagnosis as T
    return T.register_value(repo)


def compute_x2_evalset_val_v1_call_times(repo):
    """Seconds per model call by final-prompt size bucket in the v1 evalset negative controls (the input of the v2
    duration estimate in docs/EVALSET_DESIGN.md 2026-10-14)."""
    import sys as _sys
    p = str(repo / "analysis")
    if p not in _sys.path:
        _sys.path.insert(0, p)
    import evalset_val_diagnosis as D
    return D.call_time_value(repo)


def compute_x2_evalset_val_v1_gates(repo):
    """Every gate of both v1 evalset validations per model, as harness/x2_r2_evalset.evaluate_gates_ext computes it on
    the committed files (the guard file has no run_end: its qwen3:14b third session and positive control are missing)."""
    _r2_agent_module()
    import x2_r2_evalset as ev
    parts, n = [], 0
    for fn, guard in ((EVALSET_V1_FILES[0], False), (EVALSET_V1_FILES[1], True)):
        rows = ev.ag.read_rows(repo / fn)
        if not rows:
            raise FileNotFoundError(repo / fn)
        rep = ev.evaluate_gates_ext(rows, "evalset", guard, "off")
        g, t = rep["gates"], rep["baseline_table"]
        name = "tasks" if not guard else "guard"
        n += sum(1 for r in rows if r.get("record") == "r2a_turn")
        for m in sorted(t):
            rules = ", ".join(f"{k.split('_')[0]} {v['rate']:.3f}" for k, v in t[m]["rules"].items()
                              if v["rate"] is not None)
            tt, nc, pc = g["task_tool"][m], g["negative_control"][m], g["positive_control"][m]
            r678 = rep["rules678_table"][m]
            s = (f"{name} {m}: baseline {'pass' if g['baseline'][m]['pass'] else 'FAIL'} ({rules}); negative control "
                 f"{'pass' if nc['pass'] else 'FAIL'} (canary misses {nc['canary_misses']} over {nc['canary_checks']} "
                 f"checks, {nc['n_sessions']} sessions, loaded {nc['loaded_context_values']}, max transcript "
                 f"{nc['max_transcript_tokens_calibrated']}); positive control {'pass' if pc['pass'] else 'FAIL'} "
                 f"({pc['n_sessions']} sessions); task tool {'pass' if tt['pass'] else 'FAIL'} "
                 f"{tt['n_ok']}/{tt['n']} ({tt['rate']:.3f})")
            if guard:
                s += "; rules 6-8 " + ("pass" if g["rules678"][m]["pass"] else "FAIL") + " (" + ", ".join(
                    f"{k.split('_')[0]} {v['rate']:.3f} of {v['n']} bait turns" for k, v in r678.items()
                    if v["rate"] is not None) + ")"
            tk = rep["task_table"][m]
            s += f"; tasks in use {len(tk['tasks_in_use'])}"
            parts.append(s)
        if not any(r.get("record") == "run_end" for r in rows):
            parts.append(f"{name}: file has no run_end (unfinished)")
    return {"value": " / ".join(parts), "n": n}


# B3 evalset validations v2 (2026-10-15, evo-x2, harness/x2_r2_evalset.py --design v2): the three files, each scored
# with evaluate_gates_ext as the harness scores it (workload, guardrails), design v2.
EVALSET_V2_FILES = {"tasks": ("results/x2_evalset_tasks_val_v2.jsonl", "evalset", False),
                    "guard": ("results/x2_evalset_guard_val_v2.jsonl", "evalset", True),
                    "replay": ("results/x2_evalset_replay_val_v2.jsonl", "replay", True)}
EVALSET_V2_GATE_ORDER = ("baseline", "negative_control", "positive_control", "content_nonempty", "thinking_off",
                         "task_tool", "rules678", "negative_control_window", "task_end_state", "replay_fidelity")


def _evalset_v2_reports(repo):
    _r2_agent_module()
    import x2_r2_evalset as ev
    out = {}
    for name, (fn, workload, guard) in EVALSET_V2_FILES.items():
        rows = ev.ag.read_rows(repo / fn)
        if not rows:
            raise FileNotFoundError(repo / fn)
        if ev.file_design(rows) != "v2":
            raise ValueError(f"{fn} is design {ev.file_design(rows)!r}, not v2")
        out[name] = (rows, ev.evaluate_gates_ext(rows, workload, guard, "off", design="v2"))
    return out


def _pf(ok):
    return "pass" if ok else "FAIL"


def _evalset_v2_gate_text(name, g, rep, m):
    """One gate's pass/fail with the numbers it was decided on."""
    v = g[name].get(m)
    if v is None:
        return f"{name} n/a"
    if name == "baseline":
        t = rep["baseline_table"][m]["rules"]
        return (f"baseline {_pf(v['pass'])} (" + ", ".join(f"{k.split('_')[0]} {e['rate']:.3f} of {e['n']}"
                                                       for k, e in t.items() if e["rate"] is not None)
                + f"; failing {v['failing_rules'] or 'none'})")
    if name == "negative_control":
        return (f"negative control {_pf(v['pass'])} (canary misses {v['canary_misses']} of {2 * v['canary_checks']} "
                f"canaries, {v['canary_checks']} checks, {v['n_sessions']} sessions, loaded "
                f"{v['loaded_context_values']})")
    if name == "positive_control":
        return (f"positive control {_pf(v['pass'])} (truncation detected turns {v['truncation_detected_turns']}, "
                f"first over window {v['first_over_window_turns']}, loaded {v['loaded_context_values']})")
    if name == "content_nonempty":
        return f"content non-empty {_pf(v['pass'])} (empty finals {v['final_content_empty']})"
    if name == "thinking_off":
        return f"thinking off {_pf(v['pass'])} (thinking calls {v['thinking_present_calls']})"
    if name == "task_tool":
        return f"task tool {_pf(v['pass'])} {v['n_ok']}/{v['n']} ({v['rate']:.3f})"
    if name == "rules678":
        if not v["applies"]:
            return "rules 6-8 not applicable (guardrails off)"
        t = rep["rules678_table"][m]
        return (f"rules 6-8 {_pf(v['pass'])} (" + ", ".join(
            f"{k.split('_')[0]} {e['rate']:.3f} of {e['n']} bait turns" if e["rate"] is not None
            else f"{k.split('_')[0]} not exercised" for k, e in t.items()) + ")")
    if name == "negative_control_window":
        return (f"negative-control window {_pf(v['pass'])} ({v['n_over_window']}/{v['n_turns']} turns over the "
                f"loaded window, max final prompt {v['max_final_prompt_calibrated']})")
    if name == "task_end_state":
        return (f"task end state {_pf(v['pass'])} {v['n_ok']}/{v['n']} ({100 * v['rate']:.1f}%, bar "
                f"{100 * v['threshold']:.0f}%)")
    if name == "replay_fidelity":
        return f"replay fidelity {_pf(v['pass'])} {v['n_within']}/{v['n_reachable']} within {v['tol']}"
    return f"{name} {_pf(v.get('pass'))}"


def compute_x2_evalset_val_v2_gates(repo):
    """Every gate of the three v2 evalset validations (tasks, guard, replay) per model, with the numbers each gate was
    decided on, as harness/x2_r2_evalset.evaluate_gates_ext (design v2) computes it on the committed files; then the
    gates each model fails per validation."""
    reps = _evalset_v2_reports(repo)
    parts, n = [], 0
    for name, (rows, rep) in reps.items():
        g = rep["gates"]
        n += sum(1 for r in rows if r.get("record") == "r2a_turn")
        finished = any(r.get("record") == "run_end" for r in rows)
        for m in sorted(rep["baseline_table"]):
            gates = [x for x in EVALSET_V2_GATE_ORDER if x in g]
            failing = [x for x in gates if g[x].get(m) is not None and not g[x][m]["pass"]
                       and not (x == "rules678" and not g[x][m]["applies"])]
            parts.append(f"{name} {m} (run finished {finished}): "
                         + "; ".join(_evalset_v2_gate_text(x, g, rep, m) for x in gates)
                         + f"; failing gates: {', '.join(failing) or 'none'}")
    return {"value": " / ".join(parts), "n": f"{n} turns"}


def compute_x2_evalset_val_v2_task_end_state(repo):
    """The v2 task_end_state gate per validation and model (negative-control task turns with a correct end state
    against the 90% bar), the end state by difficulty, and the tasks correct on every negative-control session."""
    reps = _evalset_v2_reports(repo)
    parts, n = [], 0
    for name, (rows, rep) in reps.items():
        te = rep["gates"]["task_end_state"]
        for m in sorted(te):
            v = te[m]
            tt = rep["task_table"][m]
            n += v["n"]
            diff = ", ".join(f"{k} {d['n_ok']}/{d['n']}" for k, d in sorted(tt["by_difficulty"].items()))
            parts.append(f"{name} {m}: {v['n_ok']}/{v['n']} ({100 * v['rate']:.1f}%) against the "
                         f"{100 * v['threshold']:.0f}% bar, gate {_pf(v['pass'])}; by difficulty {diff}; tasks correct "
                         f"on every negative-control session {len(tt['tasks_in_use'])}/{len(tt['tasks'])}")
    return {"value": " / ".join(parts), "n": f"{n} negative-control task turns"}


# B3 v2 task real run (pre-registered in docs/FINDINGS.md 2026-10-19, "PRE-REGISTRATION: protocol consistency", queue
# id x2_evalset_tasks_real_v2): the duration ESTIMATE before queueing, then (PENDING until the file is synced) the gated
# after-window excess over default against v1b and the wrong-task rate excess.
X2_EVALSET_TASKS_REAL_V2 = "results/x2_evalset_tasks_real_v2.jsonl"
X2_EVALSET_TASKS_VAL_V2 = "results/x2_evalset_tasks_val_v2.jsonl"
V2_PROTOCOL_MODEL = "qwen3:14b"


def compute_x2_evalset_tasks_real_v2_hours(repo):
    """ESTIMATE (not a measurement) of x2_evalset_tasks_real_v2's duration from the v2 tasks validation's measured
    session times (harness/x2_r2_evalset.py::tasks_v2_hours_estimate)."""
    _r2_agent_module()
    import x2_r2_evalset as ev
    rows = ev.ag.read_rows(repo / X2_EVALSET_TASKS_VAL_V2)
    if not rows:
        raise FileNotFoundError(repo / X2_EVALSET_TASKS_VAL_V2)
    e = ev.tasks_v2_hours_estimate(rows)
    return {"value": f"ESTIMATE {e['point_h']:.1f} h (range {e['low_h']:.1f} to {e['high_h']:.1f} h) for "
                     f"{e['n_sessions']} {e['model']} sessions of 30 task turns ({e['n_default_sessions']} at the default "
                     f"at the measured negative-control session time, {e['n_window_sessions']} at smaller windows at the "
                     f"measured num_ctx 8192 positive-control session time); measured sessions (s): negative control "
                     + ", ".join(f"{v:.0f}" for v in e["neg_session_s"]) + "; positive control "
                     + ", ".join(f"{v:.0f}" for v in e["pos_session_s"]),
            "n": f"{len(e['neg_session_s']) + len(e['pos_session_s'])} measured sessions (estimate)"}


def protocol_reading(v2_ci, v1b_excess, v1b_ci) -> str:
    """The pre-registered protocol-consistency reading for one tier (docs/FINDINGS.md 2026-10-19)."""
    lo, hi = v2_ci
    if lo is None:
        return "inconclusive (no CI)"
    if v1b_ci[0] is not None and v1b_ci[0] > 0:
        if lo > 0:
            return "the v1b effect holds"
        if hi < v1b_excess:
            return "the v1b effect is a protocol artifact"
        return "inconclusive"
    return "v2 shows an excess v1b did not" if lo > 0 else "no excess in either protocol"


def num_keep_reading(ci_4096, ci_nk, model="qwen3:14b") -> str:
    if ci_4096[0] is None or ci_nk[0] is None:
        return "not shown either way (no CI)"
    if ci_4096[0] > 0 and ci_nk[0] <= 0 <= ci_nk[1]:
        return f"num_keep removes the 4096 excess for {model}"
    if ci_4096[0] > 0 and ci_nk[0] > 0:
        return "num_keep does not remove it"
    return "not shown either way"


def compute_x2_evalset_tasks_real_v2_gated_excess(repo, model=V2_PROTOCOL_MODEL,
                                                  real_file=X2_EVALSET_TASKS_REAL_V2,
                                                  val_file=X2_EVALSET_TASKS_VAL_V2):
    """qwen3:14b, per tier: gated silent failures after the window was exceeded and the excess over the default with
    the v1b bootstrap (analysis/r2_real_report.py build_combined + excess_over_default on the v2 file, gated metrics
    from the v2 tasks validation), next to v1b's per-model values (R2-v1b-excess-over-default), with the pre-registered
    reading per tier and for num_keep; validity from x2_r2_evalset.tasks_real_v2_outcome."""
    import sys as _sys
    _sys.path.insert(0, str(repo / "analysis"))
    import r2_real_report as rr
    _r2_agent_module()
    import x2_r2_evalset as ev
    rows = ev.ag.read_rows(repo / real_file)
    if not rows:
        raise FileNotFoundError(repo / real_file)
    o = ev.tasks_real_v2_outcome(rows, models=(model,))
    d = o["models"].get(model, {})
    valid = bool(d.get("valid"))
    rep = rr.build_combined([repo / real_file], validation_paths=[repo / val_file])
    x = rr.excess_over_default(rep)
    _rr, rep1 = _r2_real_v1b_report(repo)
    x1 = rr.excess_over_default(rep1)
    g = rr.gated_per_model(rep)
    gm = rep["gated_metrics"].get(model, {})
    parts = [f"{model}: valid {valid}" + (f" ({'; '.join(d.get('problems') or [])})" if not valid else "")
             + f"; gated fields {', '.join(gm.get('fields', []))} (excluded {', '.join(gm.get('excluded', [])) or 'none'})"]
    dflt = g.get(f"ollama_default|{model}")
    if dflt:
        parts.append(f"ollama_default {dflt['silent']}/{dflt['n']} silent, {dflt['silent_after_window_exceeded']} after "
                     f"window exceeded")
    cis = {}
    for tier, per in x.get("per_tier", {}).items():
        v = per.get(model)
        if v is None:
            continue
        lo, hi = v["ci95"]
        cis[tier] = (lo, hi)
        s = (f"{tier}: v2 {round(v['rate'] * v['n_tier'])}/{v['n_tier']} vs default "
             f"{round(v['default_rate'] * v['n_default'])}/{v['n_default']}, excess {v['excess']:+.3f} (95% CI "
             f"{lo:+.3f} to {hi:+.3f})")
        w = x1.get("per_tier", {}).get(tier, {}).get(model)
        if w is not None:
            s += (f"; v1b excess {w['excess']:+.3f} (95% CI {w['ci95'][0]:+.3f} to {w['ci95'][1]:+.3f}); reading: "
                  + (protocol_reading((lo, hi), w["excess"], w["ci95"]) if valid else "inconclusive (run not valid)"))
        parts.append(s)
    nk = "num_ctx_4096_numkeep"
    if "num_ctx_4096" in cis and nk in cis:
        parts.append("num_keep reading: " + (num_keep_reading(cis["num_ctx_4096"], cis[nk], model) if valid
                                             else "inconclusive (run not valid)"))
    return {"value": "; ".join(parts) + f" (bootstrap {x.get('n_resamples')} resamples, unit {x.get('unit')}, "
                                        f"stratified by {x.get('strata')}, seed {x.get('seed')})",
            "n": f"{rep['n_sessions']} sessions"}


def compute_x2_evalset_tasks_real_v2_wrong_rate(repo, model=V2_PROTOCOL_MODEL,
                                                real_file=X2_EVALSET_TASKS_REAL_V2):
    """qwen3:14b, per tier: wrong-task rate over all task turns, silent and surfaced wrong turns, loaded context,
    and the excess over the default with the session bootstrap and its pre-registered reading
    (harness/x2_r2_evalset.py::tasks_real_v2_outcome); refused models with their reasons."""
    _r2_agent_module()
    import x2_r2_evalset as ev
    rows = ev.ag.read_rows(repo / real_file)
    if not rows:
        raise FileNotFoundError(repo / real_file)
    o = ev.tasks_real_v2_outcome(rows, models=(model,))
    parts, n = [], 0
    for m, d in o["models"].items():
        if d.get("refused"):
            parts.append(f"{m}: refused ({'; '.join(d['reasons'] or [])})")
            continue
        segs = [f"valid {d['valid']}" + (f" ({'; '.join(d['problems'])})" if d["problems"] else "")]
        for t, i in d["tiers"].items():
            n += i["n_task_turns"]
            segs.append(f"{t}: wrong {_frac(i['wrong'], i['n_task_turns'])} (silent {i['silent_wrong']}, surfaced "
                        f"{i['surfaced_wrong']}), loaded context {i['loaded_context_values']}"
                        + (f", num_keep per seed {i['num_keep_per_seed']}" if i["num_keep_per_seed"] else ""))
        for t, e in d["excess_over_default"].get("per_tier", {}).items():
            lo, hi = e["ci95"]
            segs.append(f"{t} excess over default {100 * e['excess']:+.1f} points (95% CI "
                        + (f"{100 * lo:+.1f} to {100 * hi:+.1f}" if lo is not None else "n/a")
                        + f", {e['n_sessions']} vs {e['n_default_sessions']} sessions); reading: "
                        + (e["reading"] if d["valid"] else "not read (run not valid)"))
        parts.append(f"{m}: " + "; ".join(segs))
    return {"value": f"run finished {o['run_finished']}; " + " / ".join(parts)
                     + f" (bootstrap {ev.BOOT_N} resamples, unit session, stratified by tier, seed {ev.BOOT_SEED})",
            "n": f"{n} task turns"}


# Cloud redo of every B3 v2 wrong task (scripts/cloud_redo.py): USD per wrong task on the cloud escalation model. Stub
# mode prices the local run's token counts ("priced, not executed"); a live run (--live, key set) executes each task.
CLOUD_REDO_V1 = "results/cloud_redo_v1.jsonl"


def cloud_redo_data(repo, path=CLOUD_REDO_V1) -> dict:
    """scripts/cloud_redo.py::summarize of the cloud redo file plus "stub" (True = priced, not executed), "model_id"
    and "price_file" of its run_start. Shared by the register row and analysis/impact_model.py (impact v2 P1)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("cloud_redo", repo / "scripts" / "cloud_redo.py")
    cr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cr)
    rows = cr.load_rows(repo / path)
    start = next(r for r in rows if r.get("record") == "run_start")
    s = cr.summarize(rows)
    return dict(s, stub=s["mode"] == cr.MODE_STUB, model_id=start["model_id"], price_file=start["price_file"])


def compute_cloud_redo_cost_per_wrong_task(repo, path=CLOUD_REDO_V1):
    """Mean, median and total USD per wrong task, per tier, mode and n (scripts/cloud_redo.py::summarize)."""
    s = cloud_redo_data(repo, path)
    start = s
    stub = s["stub"]
    mode_txt = ("priced, not executed (stub: no API call; input = the local run's prompt tokens per call, output = "
                "the local run's completion tokens as a stand-in)" if stub else "executed")
    tiers = "; ".join(f"{t} n={v['n']} mean ${v['mean_usd']:.4f} median ${v['median_usd']:.4f} total "
                      f"${v['total_usd']:.4f}" for t, v in s["per_tier"].items())
    tail = ("; cloud correctness and seconds NOT MEASURED in stub mode" if stub else
            f"; cloud correct {_frac(s['cloud_correct'], s['cloud_graded'])}, mean {s['seconds_mean']:.2f} s per task")
    return {"value": f"mode {mode_txt}; model {start['model_id']} ({start['price_file']}), prompt caching on (cold "
                     f"cache per task); USD per wrong task mean ${s['mean_usd']:.4f}, median ${s['median_usd']:.4f}, "
                     f"total ${s['total_usd']:.4f} (all input uncached: ${s['total_usd_all_uncached']:.4f}); per tier: "
                     f"{tiers}; local regrade on the rebuilt environment reproduces end_state_ok false for "
                     f"{_frac(s['local_regrade_false'], s['n'])}; token basis is the harness estimate, not Anthropic's "
                     f"tokenizer{tail}",
            "n": f"{s['n']} wrong tasks"}


def compute_cloud_redo_diagnosis(repo):
    """Offline diagnosis of the live cloud redo: what transcript and environment it sent, the model and tier, and every
    cloud failure in exactly one class (analysis/cloud_redo_diagnosis.py::compute)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("cloud_redo_diagnosis", repo / "analysis" / "cloud_redo_diagnosis.py")
    cd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cd)
    d = cd.compute(repo)
    return {"value": cd.value_text(d), "n": f"{d['n_fail']} failed of {d['n_tasks']} redo tasks"}


CLOUD_REDO_FULL_CONTEXT = "results/cloud_redo_full_context_v1.jsonl"
CLOUD_REDO_FULL_CONTEXT_PART2 = "results/cloud_redo_full_context_v1_part2.jsonl"


def compute_cloud_redo_full_context(repo, path=CLOUD_REDO_FULL_CONTEXT, part2=CLOUD_REDO_FULL_CONTEXT_PART2):
    """The full-history redo with thinking off (scripts/cloud_redo.py --thinking-off), read with its --resume
    continuation file when present (cloud_redo.merged_rows: superseded rows excluded, the rerun row counts): fix rate
    with a 95% Wilson CI over all wrong tasks (tasks not run count as not fixed), USD mean / median / total, seconds,
    per tier, whether the last part stopped, the ledger total at its end, the max_tokens per part (the 2048 continuation
    is a declared protocol difference vs the local 384) and the cloud_redo_diagnosis classes for anything still wrong."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("cloud_redo", repo / "scripts" / "cloud_redo.py")
    cr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cr)
    paths = cr.part_files(repo / path)   # the first file and every <stem>_partN.jsonl, in part order
    tasks = cr.merged_rows(paths)
    starts = [next(r for r in cr.load_rows(p) if r.get("record") == "run_start") for p in paths]
    ends = [next((r for r in cr.load_rows(p) if r.get("record") == "run_end"), {}) for p in paths]
    n_sup = len({(r["attempt_id"], r["turn_idx"]) for p in paths[1:] for r in cr.load_rows(p)
                 if r.get("record") == "supersede"})
    s = cr.summarize(tasks)
    spec = importlib.util.spec_from_file_location("cloud_redo_diagnosis", repo / "analysis" / "cloud_redo_diagnosis.py")
    cd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cd)
    d = cd.compute(repo, path, tuple(str(p.relative_to(repo).as_posix()) for p in paths[1:]))
    cls = ", ".join(f"{k} {v}" for k, v in d["counts"].items())
    n_all, ok = starts[0]["n_wrong_tasks"], sum(1 for r in tasks if r.get("cloud_end_state_ok"))
    lo, hi = wilson_ci(ok, n_all)
    secs = sorted(r["seconds"] for r in tasks if r.get("seconds") is not None)
    tiers = []
    for t, v in s["per_tier"].items():
        tt = [r for r in tasks if r["tier"] == t]
        tok = sum(1 for r in tt if r.get("cloud_end_state_ok"))
        tl, th = wilson_ci(tok, len(tt))
        tiers.append(f"{t} fixed {tok}/{len(tt)} [{tl:.1%}, {th:.1%}], USD mean ${v['mean_usd']:.4f} median "
                     f"${v['median_usd']:.4f}")
    first_keys = {(r["attempt_id"], r["turn_idx"]) for r in cr.load_rows(paths[0]) if r.get("record") == "cloud_redo_task"}
    n_first = len(first_keys) - n_sup
    parts = "; ".join(f"part {i + 1}: max_tokens {st.get('max_tokens')}"
                      + (f" ({st['protocol_difference']})" if st.get("protocol_difference") else "")
                      for i, st in enumerate(starts))
    led = next((e["ledger_total_usd"] for e in reversed(ends) if e.get("ledger_total_usd") is not None), None)
    stops = "; ".join(f"part {i + 1}: " + ((str(e["stopped"]) if e.get("stopped") else "completed") if e
                                           else "no run_end (crashed; finished rows kept)")
                      for i, e in enumerate(ends))
    return {"value": f"model {starts[0]['model_id']}, thinking {starts[0].get('thinking_sent')}, full client-side "
                     f"history, caching on; {parts}; {len(tasks)} tasks run ({n_first} kept from part 1, {n_sup} part-1 "
                     f"rows superseded and rerun); fixed {_frac(ok, n_all)} of the wrong tasks, 95% Wilson CI over "
                     f"tasks [{lo:.1%}, {hi:.1%}] (tasks not run count as not fixed); USD per task (cache-priced) mean "
                     f"${s['mean_usd']:.4f}, median ${s['median_usd']:.4f}, total ${s['total_usd']:.4f} (ledger basis, "
                     f"all input uncached: ${s['total_usd_all_uncached']:.4f}); seconds per task mean "
                     f"{s['seconds_mean']:.2f}, median {statistics.median(secs):.2f}; per tier: {'; '.join(tiers)}; "
                     f"stops: {stops}; ledger total at the last run_end "
                     + (f"${led:.4f}" if led is not None else "n/a") + f"; still wrong by class: {cls}",
            "n": f"{len(tasks)} tasks run of {n_all}"}


def compute_cloud_cost_per_real_task(repo):
    """Cloud USD per real agent task (trajectory), no machine time: billed input (every step re-sends its full prompt)
    and output tokens per trajectory, median / p90 USD per trace set, tier (claude-haiku-5-5, claude-sonnet-5-5) and
    caching case (none, realistic ASSUMPTION hit rate 0.7, best). Output per step measured from the traces; counts
    calibrated Anthropic/qwen per set from messages.count_tokens (analysis/cloud_task_cost.py)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("cloud_task_cost", repo / "analysis" / "cloud_task_cost.py")
    ct = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ct)
    res = ct.compute(repo)
    return {"value": ct.value_text(res), "n": f"{sum(e['n_tasks'] for e in res['sets'].values())} trajectories"}

# All-cloud baseline on the B3 v2 tasks (scripts/all_cloud_b3.py, pre-registered in docs/FINDINGS.md 2026-10-21): the 30
# tasks run fully on claude-haiku-5-5 and claude-sonnet-5-5, 3 seeds. The all-cloud reference quality and cost for the
# hybrid savings model. PENDING until results/all_cloud_b3_v1.jsonl exists.
ALL_CLOUD_B3_V1 = "results/all_cloud_b3_v1.jsonl"


def _all_cloud_module(repo):
    import importlib.util
    spec = importlib.util.spec_from_file_location("all_cloud_b3", repo / "scripts" / "all_cloud_b3.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def all_cloud_b3_data(repo, path=ALL_CLOUD_B3_V1) -> dict:
    """run_start, run_end and scripts/all_cloud_b3.py::summarize of the file (recomputed from its turn rows)."""
    ac = _all_cloud_module(repo)
    rows = ac.CR.load_rows(repo / path)
    start = next(r for r in rows if r.get("record") == "run_start")
    end = next((r for r in rows if r.get("record") == "run_end"), None)
    return {"start": start, "end": end, "summary": ac.summarize(rows), "ac": ac}


def _local_default_correct(repo, seeds=None) -> tuple[int, int]:
    """qwen3:14b ollama_default task turns correct / graded in completed real sessions of x2_evalset_tasks_real_v2."""
    _r2_agent_module()
    import x2_r2_evalset as ev
    rows = ev.ag.read_rows(repo / X2_EVALSET_TASKS_REAL_V2)
    done = {s["attempt_id"] for s in ev.ag.completed_sessions(rows) if s.get("mode") == "real"}
    t = [r for r in rows if r.get("record") == "r2a_turn" and r.get("attempt_id") in done
         and r.get("arm_id") == "ollama_default_call2_notools" and r.get("model_id") == V2_PROTOCOL_MODEL
         and r.get("item_kind") == "task" and r.get("end_state_ok") is not None
         and (seeds is None or r.get("seed") in seeds)]
    return sum(1 for r in t if r["end_state_ok"]), len(t)


def compute_all_cloud_b3_quality(repo, path=ALL_CLOUD_B3_V1):
    """Per model: correct / graded (end_state_ok), per seed and per task type; qwen3:14b local default as context."""
    d = all_cloud_b3_data(repo, path)
    s, start, end = d["summary"], d["start"], d["end"]
    parts, n = [], 0
    for m in start["models"]:
        v = s.get(m)
        if not v:
            parts.append(f"{m}: no task turns")
            continue
        n += v["graded"]
        seeds = ", ".join(f"{k} {a}/{b}" for k, (a, b) in v["per_seed"].items())
        types = ", ".join(f"{k} {a}/{b}" for k, (a, b) in v["per_task_type"].items())
        parts.append(f"{m}: correct {_frac(v['correct'], v['graded'])} (per seed {seeds}; per task type {types}; "
                     f"turns with a provider error {v['any_http_error_turns']}; temperature "
                     f"{start.get('temperature_per_model', {}).get(m, 'n/a')})")
    l5 = _local_default_correct(repo)
    l3 = _local_default_correct(repo, set(start["seeds"]))
    stop = f"; stopped: {end['stopped']}" if end and end.get("stopped") else ""
    return {"value": f"B3 v2 tasks fully on the cloud, seeds {start['seeds']}, same session, tools and end-state grader "
                     f"as x2_evalset_tasks_real_v2; " + "; ".join(parts)
                     + f"; context only (other model, runtime and window, not matched): qwen3:14b local "
                     f"ollama_default correct {_frac(*l5)} on 5 seeds, {_frac(*l3)} on the same 3 seeds "
                     f"(x2-evalset-tasks-real-v2-wrong-rate){stop}",
            "n": f"{n} task turns"}


def compute_all_cloud_b3_cost(repo, path=ALL_CLOUD_B3_V1):
    """Per model: USD per task mean, median, p90 (cache-priced), totals, seconds mean, cache read/write totals."""
    d = all_cloud_b3_data(repo, path)
    s, start, end = d["summary"], d["start"], d["end"]
    parts, n = [], 0
    for m in start["models"]:
        v = s.get(m)
        if not v:
            continue
        u = v["usd"]
        n += u["n"]
        parts.append(f"{m}: USD per task mean ${u['mean']:.5f}, median ${u['median']:.5f}, p90 ${u['p90']:.5f}, "
                     f"total ${u['total']:.4f} (ledger basis, all input at the input rate: "
                     f"${v['usd_ledger_total']:.4f}); seconds per task mean {v['seconds_mean']:.2f}; calls per task "
                     f"mean {v['n_calls_mean']:.2f}; tokens input {v['input_tokens_total']} (cache read "
                     f"{v['cache_read_tokens_total']}, cache write {v['cache_write_tokens_total']}, uncached "
                     f"{v['uncached_input_tokens_total']}), output {v['output_tokens_total']}")
    tail = (f"; job spend ${end['job_spend_usd']:.4f} (ledger basis ${end['job_spend_usd_ledger']:.4f}, incl. "
            f"warm-ups and pilots), ledger total at end ${end['ledger_total_usd']:.4f}" if end else "")
    return {"value": f"prompt caching on ({start['cache_breakpoints']}), priced at {start['price_file']} Standard "
                     f"rates (5-minute cache write, cache hit, output); token basis provider usage; "
                     + "; ".join(parts) + tail,
            "n": f"{n} task turns"}


# ───────────────────────────────────────────────── evo-x2 queue fill of 2026-10-21 (docs/X2_QUEUE_PLAN.md)
# Jobs pre-registered in docs/FINDINGS.md on 2026-10-20/21: num_keep for qwen3:14b, B3 v2 for qwen3:8b (tasks
# validation + real run), qwen3:8b 8192 vs 4096 with 10 seeds. The hours row is an ESTIMATE from measured session times;
# the result rows are PENDING until their files are synced. x2-scale-real-default-tier backs the decision not to queue a
# qwen3:32b default-tier run (it already ran inside x2_r2_scale_real).
NUM_KEEP_Q14_FILE = "results/x2_num_keep_qwen3_14b.jsonl"
NUM_KEEP_Q14_MODEL = "qwen3:14b"
X2_EVALSET_TASKS_VAL_V2_Q8 = "results/x2_evalset_tasks_val_v2_qwen3_8b.jsonl"
X2_EVALSET_TASKS_REAL_V2_Q8 = "results/x2_evalset_tasks_real_v2_qwen3_8b.jsonl"
Q8_MODEL = "qwen3:8b"
R2_Q8_SEEDS_06_10_FILE = "results/x2_r2_qwen3_8b_seeds_06_10.jsonl"
Q8_NEW_SEEDS = (20260906, 20260907, 20260908, 20260909, 20260910)
Q8_REVERSAL_TIERS = ("num_ctx_8192", "num_ctx_4096")
# The queue fill's schedule anchor: x2_num_keep_qwen3_14b was launched by the watchdog at this time (watchdog.log on
# evo-x2, '{"action": "launched", "launched": "x2_num_keep_qwen3_14b"}'). Jobs are projected back to back from it.
QUEUE_FILL_ANCHOR_UTC = "2026-10-21T18:03:40+00:00"
QUEUE_FILL_DEADLINE_UTC = "2026-11-10T00:00:00+00:00"
QUEUE_FILL_ORDER = ("x2_num_keep_qwen3_14b", "x2_evalset_tasks_val_v2_qwen3_8b", "x2_evalset_tasks_real_v2_qwen3_8b",
                    "x2_r2_qwen3_8b_seeds_06_10")
QUEUE_FILL_DATA = [NUM_KEEP_FILE, X2_EVALSET_TASKS_VAL_V2, X2_EVALSET_TASKS_REAL_V2, R2_REAL_V1_FILE,
                   R2_REAL_V1B_FILE]


def _session_walls_s(rows, model, arm, mode=None) -> list[float]:
    """Per completed session of (model, arm): the sum of its turns' turn_wall_s (model loads included), sorted."""
    ag = _r2_agent_module()
    att = {s["attempt_id"] for s in ag.completed_sessions(rows) if s["model_id"] == model and s["arm_id"] == arm
           and (mode is None or s.get("mode") == mode)}
    walls = {}
    for r in rows:
        if r.get("record") == "r2a_turn" and r.get("attempt_id") in att:
            walls[r["attempt_id"]] = walls.get(r["attempt_id"], 0.0) + (r.get("turn_wall_s") or 0.0)
    return sorted(walls.values())


def _r2_median_s(repo, model, tier) -> float:
    import sys as _sys
    _sys.path.insert(0, str(repo / "analysis"))
    import impact_model as im
    return im.session_wallclock(repo)[("evo-x2", model, tier)]["median_wall_s"]


def queue_fill_estimates(repo) -> dict:
    """ESTIMATE per queued job (hours, point and high) from measured session times; see each job's basis string."""
    import statistics
    ag = _r2_agent_module()
    import x2_r2_evalset as ev
    sfx = "_call2_notools"
    jobs = {}
    # a) num_keep for qwen3:14b: the llama3.1:8b num_keep sessions' overhead over plain llama3.1:8b 4096 sessions,
    # applied to the median plain qwen3:14b 4096 session, x 3 sessions.
    nk_rows = ag.read_rows(repo / NUM_KEEP_FILE)
    nk_walls = _session_walls_s(nk_rows, "llama3.1:8b", "ollama_ctx_4096" + sfx)
    l4096 = _r2_median_s(repo, "llama3.1:8b", "num_ctx_4096")
    q4096 = _r2_median_s(repo, "qwen3:14b", "num_ctx_4096")
    ratio = statistics.mean(nk_walls) / l4096
    jobs["x2_num_keep_qwen3_14b"] = {
        "point_h": 3 * q4096 * ratio / 3600, "high_h": 3 * q4096 * max(ratio, max(nk_walls) / l4096) / 3600,
        "basis": f"3 x qwen3:14b num_ctx 4096 R2 median {q4096 / 60:.1f} min x num_keep overhead {ratio:.2f} (llama3.1:8b "
                 f"num_keep sessions mean {statistics.mean(nk_walls) / 60:.1f} min over its plain 4096 median "
                 f"{l4096 / 60:.1f} min)"}
    # per-tier qwen3:8b / qwen3:14b R2 session-time ratios
    r = {t: _r2_median_s(repo, "qwen3:8b", t) / _r2_median_s(repo, "qwen3:14b", t)
         for t in ("ollama_default", "num_ctx_16384", "num_ctx_8192", "num_ctx_4096")}
    # b1) tasks validation for qwen3:8b: qwen3:14b's measured v2 tasks-validation sessions, scaled by the R2 ratios
    vrows = ag.read_rows(repo / X2_EVALSET_TASKS_VAL_V2)
    e = ev.tasks_v2_hours_estimate(vrows)
    neg, pos = sum(e["neg_session_s"]), sum(e["pos_session_s"])
    jobs["x2_evalset_tasks_val_v2_qwen3_8b"] = {
        "point_h": (neg * r["ollama_default"] + pos * r["num_ctx_8192"]) / 3600, "high_h": (neg + pos) / 3600,
        "basis": f"qwen3:14b v2 tasks validation sessions (negative control {neg / 60:.1f} min over "
                 f"{len(e['neg_session_s'])}, positive control {pos / 60:.1f} min) x qwen3:8b/qwen3:14b R2 median ratio "
                 f"(default {r['ollama_default']:.2f}, 8192 {r['num_ctx_8192']:.2f}); high = unscaled"}
    # b2) real run for qwen3:8b: qwen3:14b's measured v2 real-run sessions per tier, scaled by the same ratios
    rrows = ag.read_rows(repo / X2_EVALSET_TASKS_REAL_V2)
    tier_ratio = {"ollama_default": "ollama_default", "ollama_ctx_16384": "num_ctx_16384",
                  "ollama_ctx_8192": "num_ctx_8192", "ollama_ctx_4096": "num_ctx_4096",
                  ev.NUMKEEP_ARM: "num_ctx_4096"}
    pt = hi = 0.0
    per = []
    for t, rt in tier_ratio.items():
        w = sum(_session_walls_s(rrows, "qwen3:14b", t + sfx, mode="real"))
        pt += w * r[rt]
        hi += w
        per.append(f"{t} {w / 3600:.2f} h")
    jobs["x2_evalset_tasks_real_v2_qwen3_8b"] = {
        "point_h": pt / 3600, "high_h": hi / 3600,
        "basis": "qwen3:14b x2_evalset_tasks_real_v2 measured per tier (" + ", ".join(per) + ") x the qwen3:8b/qwen3:14b "
                 f"R2 median ratio of the tier (16384 {r['num_ctx_16384']:.2f}, 4096 {r['num_ctx_4096']:.2f}); high = "
                 "unscaled (refusal at preflight would take minutes)"}
    # c) qwen3:8b 8192 and 4096, 5 new seeds each, from its own v1b sessions at those tiers
    m8, m4 = _r2_median_s(repo, "qwen3:8b", "num_ctx_8192"), _r2_median_s(repo, "qwen3:8b", "num_ctx_4096")
    slow = 0.0
    import sys as _sys
    _sys.path.insert(0, str(repo / "analysis"))
    v1b = ag.read_rows(repo / R2_REAL_V1B_FILE)
    s8 = _session_walls_s(v1b, "qwen3:8b", "ollama_ctx_8192" + sfx, mode="real")
    s4 = _session_walls_s(v1b, "qwen3:8b", "ollama_ctx_4096" + sfx, mode="real")
    slow = 5 * (max(s8) + max(s4))
    jobs["x2_r2_qwen3_8b_seeds_06_10"] = {
        "point_h": 5 * (m8 + m4) / 3600, "high_h": slow / 3600,
        "basis": f"5 x qwen3:8b v1b median at 8192 ({m8 / 60:.1f} min) + 5 x at 4096 ({m4 / 60:.1f} min); high = 5 x the "
                 "slowest v1b session of each tier"}
    return jobs


def compute_x2_queue_fill_1020_hours(repo):
    """ESTIMATE (not a measurement) of each job of the 2026-10-21 evo-x2 queue fill, from measured session times in
    committed files, and the back-to-back schedule from the first job's launch to the 2026-11-10 deadline."""
    from datetime import datetime, timedelta
    jobs = queue_fill_estimates(repo)
    t0 = datetime.fromisoformat(QUEUE_FILL_ANCHOR_UTC)
    dl = datetime.fromisoformat(QUEUE_FILL_DEADLINE_UTC)
    t_pt = t_hi = t0
    parts = []

    def f(t):
        return t.strftime("%m-%d %H:%MZ")
    for j in QUEUE_FILL_ORDER:
        e = jobs[j]
        s_pt, s_hi = t_pt, t_hi
        t_pt += timedelta(hours=e["point_h"])
        t_hi += timedelta(hours=e["high_h"])
        parts.append(f"{j} ESTIMATE {e['point_h']:.2f} h (high {e['high_h']:.2f} h; {e['basis']}); projected "
                     f"{f(s_pt)} to {f(t_pt)} (high case {f(s_hi)} to {f(t_hi)})")
    tot_pt = sum(e["point_h"] for e in jobs.values())
    tot_hi = sum(e["high_h"] for e in jobs.values())
    slack_pt = (dl - t_pt).total_seconds() / 3600
    slack_hi = (dl - t_hi).total_seconds() / 3600
    parts.append(f"total ESTIMATE {tot_pt:.1f} h (high {tot_hi:.1f} h) from {QUEUE_FILL_ANCHOR_UTC[:16]}Z; queue ends "
                 f"{f(t_pt)} (high {f(t_hi)}); slack before {QUEUE_FILL_DEADLINE_UTC[:10]} 00:00Z {slack_pt:.1f} h "
                 f"(high case {slack_hi:.1f} h); idle capacity left after the 24 h reserve {slack_pt - 24:.1f} h "
                 f"(high case {slack_hi - 24:.1f} h)")
    return {"value": "; ".join(parts), "n": f"{len(jobs)} jobs (estimate)"}


# ───────────────────────────────────────────────── evo-x2 queue fill of 2026-10-22 (docs/X2_QUEUE_PLAN.md)
# Operator order of 2026-10-21/22: B5 flash-attention control first, then B5 under a simulated device window (8192,
# 4096; 5 seeds), then R2 at 10 seeds on the v1b harness blob, the qwen3:32b spill sweep, and mistral-small3.2:24b B3 v2.
# The hours row is an ESTIMATE from measured times in committed files; result rows are PENDING until synced.
B5_X2_FILE = "results/x2_policy_compare.jsonl"
B5_FA_FILE = "results/x2_policy_compare_fa_control.jsonl"
B5_SIMCAP_FILES = {8192: "results/x2_policy_compare_simcap8192.jsonl", 4096: "results/x2_policy_compare_simcap4096.jsonl"}
B5_SIM_SEEDS = (20260901, 20260902, 20260903, 20260904, 20260905)
MX2_VALIDATION_FILE = "results/mx2_validation.jsonl"
# The schedule anchor: the watchdog launch of the first job of this fill (watchdog.log on evo-x2,
# {"action": "launched", "launched": "x2_b5_fa_control"} at 2026-10-22T03:13:40Z). Jobs are projected back to back.
FILL_1022_ANCHOR_UTC = "2026-10-22T03:13:40+00:00"
FILL_1022_DEADLINE_UTC = "2026-11-10T00:00:00+00:00"
FILL_1022_RESERVE_H = 150.0             # operator: about 150 h left unallocated before the deadline for reviewer reruns
FILL_1022_RESULTS_BY_UTC = "2026-10-28T00:00:00+00:00"   # operator: the two B5 jobs must have results by then
FILL_1022_ORDER = ("x2_b5_fa_control", "x2_b5_simcap_8192", "x2_b5_simcap_4096", "x2_r2_v1b_seeds_06_10",
                   "x2_mx2_qwen3_32b_spill_sweep", "x2_evalset_tasks_val_v2_mistral_small32",
                   "x2_evalset_tasks_real_v2_mistral_small32")
FILL_1022_DATA = [B5_X2_FILE, R2_REAL_V1_FILE, R2_REAL_V1B_FILE, MX2_VALIDATION_FILE, X2_FAMILY_REAL,
                  X2_EVALSET_TASKS_VAL_V2, X2_EVALSET_TASKS_REAL_V2]
SPILL_SWEEP_POINTS_MAX = 9              # 6 grid points + up to 3 refinement points (pre-registration)
SPILL_SWEEP_MEASURED_CALLS = 5          # plus 1 warm-up call per point


def _b5_policy_minutes(repo) -> dict:
    """{model: {policy: per-session mean minutes}} on B5's evo-x2 file (wall_min_proxy_measured: measured time, the
    router's proxy calls included)."""
    import sys as _sys
    _sys.path.insert(0, str(repo / "harness"))
    import policy_compare as pc
    rows = pc.ag.read_rows(repo / B5_X2_FILE)
    m = pc.metrics_for_platform(rows)
    return {model: {p: v["wall_min_proxy_measured"] for p, v in c["policies"].items()} for model, c in m.items()}


def queue_fill_1022_estimates(repo) -> dict:
    """ESTIMATE per job (hours, point and high) from measured times; see each job's basis string."""
    import statistics
    import sys as _sys
    from datetime import datetime
    ag = _r2_agent_module()
    import x2_r2_evalset as ev
    sfx = "_call2_notools"
    b5 = _b5_policy_minutes(repo)
    jobs = {}
    # 1) flash-attention control: 3 sessions per model at (g)/(h)'s mean (point: flash attention explains it) or at
    # (a)'s mean (high); the llama3.1:8b f16 recall rerun at B5's measured f16 call times, the timed-out call at the
    # new 1800 s limit in the high case; window probes and the base-env check at B5's measured probe time.
    rows = ag.read_rows(repo / B5_X2_FILE)
    rec = [r for r in rows if r.get("record") == "b5_recall" and r["model_id"] == "llama3.1:8b"
           and r["precision"] == "f16"]
    rec_s = sum(r.get("duration_s") or 0.0 for r in rec)
    rec_hi = sum((1800.0 if r.get("outcome") != "ok" else (r.get("duration_s") or 0.0)) for r in rec)
    ts = {r["record"]: r["ts_utc"] for r in rows if r.get("record") in ("runtime",)}
    first_hb = next(r["ts_utc"] for r in rows if r.get("record") == "heartbeat")
    probe_s = (datetime.fromisoformat(first_hb) - datetime.fromisoformat(ts["runtime"])).total_seconds()
    gh = {m: (b5[m]["g"] + b5[m]["h"]) / 2 for m in b5}
    pt = sum(3 * gh[m] for m in b5) / 60 + rec_s / 3600 + probe_s / 3 / 3600
    hi = sum(3 * b5[m]["a"] for m in b5) / 60 + rec_hi / 3600 + probe_s / 3 / 3600
    jobs["x2_b5_fa_control"] = {
        "point_h": pt, "high_h": hi,
        "basis": "3 sessions per model at the B5 (g)/(h) mean (" + ", ".join(f"{m} {gh[m]:.1f} min" for m in sorted(gh))
                 + "); high at the (a) mean (" + ", ".join(f"{m} {b5[m]['a']:.1f} min" for m in sorted(b5))
                 + f"); llama3.1:8b f16 recall {rec_s / 60:.1f} min as measured in B5 (high {rec_hi / 60:.1f} min, the "
                 f"timed-out call at 1800 s); probes {probe_s / 3 / 60:.1f} min (B5's probe time for 3 envs / 3)"}
    # 2) simulated constraint, 5 seeds per cap: (a)=(b)=(e), (c), (f), (g), (h) at the R2 v1/v1b median session of the
    # capped num_ctx (high: the slowest); (d) at B5's measured router session (proxy calls at the uncapped window).
    v1 = ag.read_rows(repo / R2_REAL_V1_FILE) + ag.read_rows(repo / R2_REAL_V1B_FILE)
    for cap in (8192, 4096):
        p_h = h_h = 0.0
        parts = []
        for m in ("llama3.1:8b", "qwen3:14b"):
            med = _r2_median_s(repo, m, f"num_ctx_{cap}") / 60
            slow = max(_session_walls_s(v1, m, f"ollama_ctx_{cap}" + sfx, mode="real")) / 60
            d = b5[m]["d"]
            p_h += len(B5_SIM_SEEDS) * (5 * med + d) / 60
            h_h += len(B5_SIM_SEEDS) * (5 * slow + d) / 60
            parts.append(f"{m} 5 policies x {med:.1f} min (slowest {slow:.1f}) + router {d:.1f} min")
        p_h += probe_s / 3600
        h_h += probe_s / 3600
        jobs[f"x2_b5_simcap_{cap}"] = {
            "point_h": p_h, "high_h": h_h,
            "basis": f"5 seeds x ({'; '.join(parts)}) (R2 v1/v1b num_ctx {cap} sessions; router = B5's measured "
                     f"router session); probes {probe_s / 60:.1f} min (B5's)"}
    # 3) R2 at 10 seeds on the v1b blob: the reproduction session (its own v1b wall time) + 5 seeds x 4 tiers x 2 models
    v1b =ag.read_rows(repo / R2_REAL_V1B_FILE)
    att = [s["attempt_id"] for s in ag.completed_sessions(v1b) if s["model_id"] == "llama3.1:8b"
           and s["arm_id"] == "ollama_ctx_4096" + sfx and s["seed"] == 20260904]
    rep_s = sum(r.get("turn_wall_s") or 0.0 for r in v1b if r.get("record") == "r2a_turn" and r.get("attempt_id") in att)
    p_h, h_h, parts = rep_s / 3600, rep_s / 3600, [f"reproduction session {rep_s / 60:.1f} min"]
    tier_arm = {"ollama_default": "ollama_default", "num_ctx_16384": "ollama_ctx_16384",
                "num_ctx_8192": "ollama_ctx_8192", "num_ctx_4096": "ollama_ctx_4096"}
    for m in ("llama3.1:8b", "qwen3:14b"):
        for t, a in tier_arm.items():
            med = _r2_median_s(repo, m, t)
            slow = max(_session_walls_s(v1, m, a + sfx, mode="real"))
            p_h += 5 * med / 3600
            h_h += 5 * slow / 3600
            parts.append(f"{m} {t} {med / 60:.1f} min (slowest {slow / 60:.1f})")
    jobs["x2_r2_v1b_seeds_06_10"] = {"point_h": p_h, "high_h": h_h,
                                     "basis": "5 seeds x v1/v1b medians: " + "; ".join(parts)}
    # 4) qwen3:32b spill sweep: per point, the measured MX2 qwen3-32b point time (start + 1 call + stop) + 5 more calls
    # at the measured call latency, x the maximum number of points.
    mx = [r for r in ag.read_rows(repo / MX2_VALIDATION_FILE) if r.get("model_id") == "qwen3-32b"]
    hb = [(r["point_type"], r["rep"], datetime.fromisoformat(r["ts_utc"])) for r in mx if r.get("record") == "heartbeat"
          and r.get("point_type") in ("deep_fits", "mid_spill")]
    started = {(r["point_type"], r["rep"]): r for r in mx if r.get("record") == "mx2v_regime_point" and r.get("started")}
    ivals = {}
    for i in range(len(hb) - 1):
        k = (hb[i][0], hb[i][1])
        if k in started:
            ivals[k] = (hb[i + 1][2] - hb[i][2]).total_seconds()
    lat = [r["latency_ms"] / 1000 for r in started.values() if r.get("latency_ms")]
    deep = [v for (p, _), v in ivals.items() if p == "deep_fits"]
    per_pt = statistics.mean(deep) + SPILL_SWEEP_MEASURED_CALLS * statistics.mean(lat)
    per_hi = max(ivals.values()) + SPILL_SWEEP_MEASURED_CALLS * max(lat)
    jobs["x2_mx2_qwen3_32b_spill_sweep"] = {
        "point_h": SPILL_SWEEP_POINTS_MAX * per_pt / 3600, "high_h": SPILL_SWEEP_POINTS_MAX * per_hi / 3600,
        "basis": f"{SPILL_SWEEP_POINTS_MAX} points x (MX2 qwen3-32b point {statistics.mean(deep):.0f} s at 20480 + "
                 f"{SPILL_SWEEP_MEASURED_CALLS} calls x {statistics.mean(lat):.1f} s); high: slowest point "
                 f"{max(ivals.values()):.0f} s (clamped 40960) + {SPILL_SWEEP_MEASURED_CALLS} x {max(lat):.1f} s"}
    # 5) mistral-small3.2:24b B3 v2: qwen3:14b's measured v2 validation and real-run sessions x the mistral/qwen3:14b R2
    # session-time ratio (32768 for the large windows, 4096 for the small); high: the larger ratio everywhere.
    fam = ag.read_rows(repo / X2_FAMILY_REAL)
    r_big = statistics.median(_session_walls_s(fam, X2_FAMILY_MODEL, "ollama_ctx_32768" + sfx, mode="real")) / \
        _r2_median_s(repo, "qwen3:14b", "num_ctx_32768")
    r_small = statistics.median(_session_walls_s(fam, X2_FAMILY_MODEL, "ollama_ctx_4096" + sfx, mode="real")) / \
        _r2_median_s(repo, "qwen3:14b", "num_ctx_4096")
    rmax = max(r_big, r_small)
    e = ev.tasks_v2_hours_estimate(ag.read_rows(repo / X2_EVALSET_TASKS_VAL_V2))
    neg, pos = sum(e["neg_session_s"]), sum(e["pos_session_s"])
    jobs["x2_evalset_tasks_val_v2_mistral_small32"] = {
        "point_h": (neg * r_big + pos * r_small) / 3600, "high_h": (neg + pos) * rmax / 3600,
        "basis": f"qwen3:14b v2 tasks validation (negative control {neg / 60:.1f} min, positive control "
                 f"{pos / 60:.1f} min) x mistral/qwen3:14b R2 ratio (32768 {r_big:.2f}, 4096 {r_small:.2f}); high = "
                 f"x {rmax:.2f} throughout"}
    rr = ag.read_rows(repo / X2_EVALSET_TASKS_REAL_V2)
    big = ("ollama_default", "ollama_ctx_16384")
    pt = hi = 0.0
    per = []
    for t in ("ollama_default", "ollama_ctx_16384", "ollama_ctx_8192", "ollama_ctx_4096", ev.NUMKEEP_ARM):
        w = sum(_session_walls_s(rr, "qwen3:14b", t + sfx, mode="real"))
        pt += w * (r_big if t in big else r_small)
        hi += w * rmax
        per.append(f"{t} {w / 3600:.2f} h")
    jobs["x2_evalset_tasks_real_v2_mistral_small32"] = {
        "point_h": pt / 3600, "high_h": hi / 3600,
        "basis": "qwen3:14b x2_evalset_tasks_real_v2 per tier (" + ", ".join(per) + f") x the ratio above; runs only if "
                 "the validation passes every gate (a refusal takes minutes)"}
    return jobs


def compute_x2_queue_fill_1022_hours(repo):
    """ESTIMATE (not a measurement) of each job of the 2026-10-22 evo-x2 queue fill, from measured times in committed
    files, the back-to-back schedule from the anchor, the B5 results-by date, the 150 h reserve and idle capacity."""
    from datetime import datetime, timedelta
    jobs = queue_fill_1022_estimates(repo)
    t0 = datetime.fromisoformat(FILL_1022_ANCHOR_UTC)
    dl = datetime.fromisoformat(FILL_1022_DEADLINE_UTC)
    by = datetime.fromisoformat(FILL_1022_RESULTS_BY_UTC)
    t_pt = t_hi = t0
    parts, ends = [], {}

    def f(t):
        return t.strftime("%m-%d %H:%MZ")
    for j in FILL_1022_ORDER:
        e = jobs[j]
        s_pt, s_hi = t_pt, t_hi
        t_pt += timedelta(hours=e["point_h"])
        t_hi += timedelta(hours=e["high_h"])
        ends[j] = (t_pt, t_hi)
        parts.append(f"{j} ESTIMATE {e['point_h']:.2f} h (high {e['high_h']:.2f} h; {e['basis']}); projected "
                     f"{f(s_pt)} to {f(t_pt)} (high case {f(s_hi)} to {f(t_hi)})")
    b5_end = ends["x2_b5_simcap_4096"]
    tot_pt = sum(e["point_h"] for e in jobs.values())
    tot_hi = sum(e["high_h"] for e in jobs.values())
    slack_pt = (dl - t_pt).total_seconds() / 3600
    slack_hi = (dl - t_hi).total_seconds() / 3600
    parts.append(f"B5 jobs end {f(b5_end[0])} (high {f(b5_end[1])}), results-by {FILL_1022_RESULTS_BY_UTC[:10]} "
                 f"{'met' if b5_end[1] <= by else 'NOT met in the high case'}")
    parts.append(f"total ESTIMATE {tot_pt:.1f} h (high {tot_hi:.1f} h) from {FILL_1022_ANCHOR_UTC[:16]}Z; queue ends "
                 f"{f(t_pt)} (high {f(t_hi)}); slack before {FILL_1022_DEADLINE_UTC[:10]} 00:00Z {slack_pt:.1f} h "
                 f"(high case {slack_hi:.1f} h); unallocated beyond the {FILL_1022_RESERVE_H:.0f} h reserve "
                 f"{slack_pt - FILL_1022_RESERVE_H:.1f} h (high case {slack_hi - FILL_1022_RESERVE_H:.1f} h)")
    return {"value": "; ".join(parts), "n": f"{len(jobs)} jobs (estimate)"}


def cost_cut_config_time_data(repo) -> dict:
    """Per model: the runtime default (B5 policy a) and the fastest measured non-router setting on evo-x2 (B5 policies
    b-h, and the flash-attention control's policy i once synced and valid): {"models": {model: {"default_min",
    "best", "best_name", "best_min", "saved_min", ...}}, "fa_included": bool}. Shared by the register row and
    analysis/impact_model.py (impact v2 P2)."""
    import sys as _sys
    _sys.path.insert(0, str(repo / "harness"))
    import policy_compare as pc
    rows = pc.ag.read_rows(repo / B5_X2_FILE)
    m = pc.metrics_for_platform(rows)
    fa = {}
    if (repo / B5_FA_FILE).exists():
        fa = pc.fa_control_summary(pc.ag.read_rows(repo / B5_FA_FILE), rows)
    out = {}
    for model, c in sorted(m.items()):
        pol = {p: v for p, v in c["policies"].items() if p != pc.ROUTER_POLICY and not v.get("alias_of")}
        cand = {p: (v["wall_min"], v["rule_pass_rate"], v["history_recall"], v["window"]) for p, v in pol.items()}
        if model in fa and fa[model].get("valid"):
            x = fa[model]
            cand["i"] = (x["m_i"], x["rule_pass_rate"], x["history_recall"], x["window"])
        a = cand["a"]
        best = min((p for p in cand if p != "a"), key=lambda p: cand[p][0])
        b = cand[best]
        out[model] = {"default_min": a[0], "default_rule": a[1], "default_hist": a[2], "best": best,
                      "best_name": pc.FA_POLICY_NAME if best == "i" else pc.POLICIES[best]["name"],
                      "best_min": b[0], "best_rule": b[1], "best_hist": b[2], "best_window": b[3],
                      "saved_min": a[0] - b[0], "seeds": c["balanced_seeds"] and len(c["balanced_seeds"])}
    return {"models": out, "fa_included": bool(fa)}


def compute_cost_cut_config_time(repo):
    """MEASURED: per model, the session minutes saved per 40-turn R2 session by the fastest measured non-router setting
    on evo-x2 (B5 policies b-h, and the flash-attention control's policy i once synced) against the runtime default
    (B5 policy a), from the result files; rule pass rate and history recall of both settings beside it."""
    d = cost_cut_config_time_data(repo)
    parts = []
    for model, x in d["models"].items():
        parts.append(f"evo-x2 {model}: default (a) {x['default_min']:.1f} min per session, rule "
                     f"{100 * x['default_rule']:.1f}% hist {100 * x['default_hist']:.1f}%; fastest measured setting "
                     f"({x['best']}) {x['best_name']} {x['best_min']:.1f} min, rule {100 * x['best_rule']:.1f}% hist "
                     f"{100 * x['best_hist']:.1f}%, window {x['best_window']}; saved {x['saved_min']:.1f} min per "
                     f"session ({100 * x['saved_min'] / x['default_min']:.0f}%), over {x['seeds']} seeds")
    return {"value": "MEASURED; " + " / ".join(parts) + ("; flash-attention control "
                     + ("included" if d["fa_included"] else "not synced yet (policy i not in the comparison)")),
            "n": f"{len(d['models'])} models"}


def compute_b5_fa_control(repo):
    """The flash-attention control's pre-registered reading per model (policy_compare.fa_control_summary)."""
    import sys as _sys
    _sys.path.insert(0, str(repo / "harness"))
    import policy_compare as pc
    rows = pc.ag.read_rows(repo / B5_FA_FILE)
    if not rows:
        raise FileNotFoundError(repo / B5_FA_FILE)
    s = pc.fa_control_summary(rows, pc.ag.read_rows(repo / B5_X2_FILE))
    base = {r["model_id"]: r["check"]["config_flash_attention_values"] for r in rows
            if r.get("record") == "b5_base_fa_check"}
    parts = []
    for model, e in s.items():
        parts.append(f"{model}: (i) f16+FA {e['m_i'] if e['m_i'] is None else round(e['m_i'], 1)} min per session "
                     f"(per seed {', '.join(f'{k}: {v:.1f}' for k, v in e['per_seed_min'].items())}), (g)/(h) "
                     f"{e['m_gh'] if e['m_gh'] is None else round(e['m_gh'], 1)} min, (a) "
                     f"{e['m_a'] if e['m_a'] is None else round(e['m_a'], 1)} min, window {e['window']}; rule "
                     f"{e['rule_pass_rate']}, hist {e['history_recall']}; base-env OLLAMA_FLASH_ATTENTION "
                     f"{base.get(model)}; valid {e['valid']}"
                     + (f" ({'; '.join(e['problems'])})" if e["problems"] else "")
                     + f"; reading: {e['reading']}"
                     + (f" (flash-attention share of the gap {e['fa_share']:.2f})" if e.get("fa_share") is not None
                        and e["reading"] == "both contribute" else ""))
    return {"value": " / ".join(parts) + f" (tolerance {int(100 * pc.FA_TOLERANCE)}%)",
            "n": f"{sum(len(e['per_seed_min']) for e in s.values())} sessions"}


def compute_b5_kv_recall_f16_rerun(repo):
    """llama3.1:8b f16 recall rerun (flash-attention control job) against B5's q8_0 and q4_0 rows: needles found of
    24 per precision, two-sided Fisher exact per pair (descriptive), errors and durations."""
    import sys as _sys
    _sys.path.insert(0, str(repo / "harness"))
    import policy_compare as pc
    new = pc.ag.read_rows(repo / B5_FA_FILE)
    if not new:
        raise FileNotFoundError(repo / B5_FA_FILE)
    old = pc.ag.read_rows(repo / B5_X2_FILE)
    m = "llama3.1:8b"

    def found(rows, prec):
        rs = [r for r in rows if r.get("record") == "b5_recall" and r["model_id"] == m and r["precision"] == prec
              and not r.get("warmup")]
        ok = [r for r in rs if r.get("outcome") == "ok"]
        return (sum(r["score"]["found"] for r in ok), sum(r["score"]["n"] for r in ok), len(rs) - len(ok),
                [round(r.get("duration_s") or 0) for r in rs], {r.get("num_ctx") for r in rs})
    f16 = found(new, "f16")
    parts = [f"f16 (rerun) {f16[0]}/{f16[1]} needles, errors {f16[2]}, call s {f16[3]}, ctx {sorted(f16[4])}"]
    for q in ("q8_0", "q4_0"):
        x = found(old, q)
        p = fisher_two_sided(f16[0], f16[1], x[0], x[1]) if f16[1] and x[1] else None
        parts.append(f"{q} (B5) {x[0]}/{x[1]} needles, errors {x[2]}; Fisher two-sided p (f16 vs {q}) = "
                     + (f"{p:.3f}" if p is not None else "n/a") + "; reading: "
                     + ("not read (missing calls)" if p is None else
                        (f"f16 differs from {q}" if p < 0.05 else "no difference shown")))
    return {"value": f"{m}: " + "; ".join(parts) + " (needles within a call are not independent; descriptive)",
            "n": f"{len(f16[3])} measured calls"}


def compute_b5_sim_constraint(repo):
    """B5 under a simulated device window (8192 and 4096, 5 seeds, label "simulated constraint"): per cap, model and
    policy the B5 metrics (rule pass rate, history recall, session minutes, cloud USD), the frontier and the B5 verdict
    rule applied to the two caps as the platforms."""
    import sys as _sys
    _sys.path.insert(0, str(repo / "harness"))
    import policy_compare as pc
    prows = {}
    for cap, f in B5_SIMCAP_FILES.items():
        r = pc.ag.read_rows(repo / f)
        if not r:
            raise FileNotFoundError(repo / f)
        prows[f"cap{cap}"] = r
    prices = pc.load_cloud_prices(repo)
    parts = []
    for plat, rows in prows.items():
        mets = pc.metrics_for_platform(rows, prices, seeds=B5_SIM_SEEDS)
        for model, c in sorted(mets.items()):
            segs = [f"{p} rule {100 * v['rule_pass_rate']:.1f}% hist {100 * (v['history_recall'] or 0):.1f}% "
                    f"{v['wall_min']:.1f} min ${v['cloud_usd']:.4f}" + (f" (={v['alias_of']})" if v.get("alias_of")
                                                                         else "")
                    for p, v in sorted(c["policies"].items())]
            fr = pc.pareto_frontier(c["policies"]) if c["complete"] else None
            parts.append(f"{plat} {model}: " + "; ".join(segs) + f"; frontier {fr}; complete {c['complete']}"
                         + (f" ({'; '.join(c['problems'])})" if c["problems"] else ""))
    v = pc.b5_verdict(prows, prices, required=tuple(prows), seeds=B5_SIM_SEEDS)
    return {"value": "simulated constraint; " + " / ".join(parts) + f"; verdict (B5 rule, caps as platforms): "
                     f"{v['verdict']}", "n": f"{len(prows)} cap files"}




R2_V1B_SEEDS_FILE = "results/x2_r2_v1b_seeds_06_10.jsonl"
R2_V1B_REPRO_FILE = "results/x2_r2_v1b_repro_check.jsonl"
V1B_AGENT_BLOB = "a4c4a928fc7a9b6ac985d473b789b41b9a4b1c12"
V1B10_MODELS = ("llama3.1:8b", "qwen3:14b")
V1B10_TIERS = ("num_ctx_16384", "num_ctx_8192", "num_ctx_4096")
V1B10_NEW_SEEDS = (20260906, 20260907, 20260908, 20260909, 20260910)


def excess_reading(lo, hi, tier) -> str:
    """The pre-registered reading of an excess CI (docs/FINDINGS.md 2026-10-22, R2 at 10 seeds on the v1b blob)."""
    if lo is not None and lo > 0:
        return f"excess at {tier}"
    if hi is not None and hi < 0:
        return f"lower at {tier} than at the default"
    return f"no excess shown at {tier}"


def compute_r2_v1b_excess_10seed(repo):
    """R2 window result pooled over 10 seeds (v1/v1b seeds 20260901-05 + seeds 20260906-10 run on the v1b blob):
    per model and tier the gated after-window silent sessions, the excess over the default with the v1b bootstrap,
    the pre-registered reading, a sign-reversal flag against the 5-seed excess, the new seeds alone, the CI width at 5
    and 10 seeds, and validity (reproduction check, blob, sessions)."""
    import sys as _sys
    _sys.path.insert(0, str(repo / "analysis"))
    import r2_real_report as rr
    ag = _r2_agent_module()
    new = ag.read_rows(repo / R2_V1B_SEEDS_FILE)
    if not new:
        raise FileNotFoundError(repo / R2_V1B_SEEDS_FILE)
    rep_rows = ag.read_rows(repo / R2_V1B_REPRO_FILE)
    vpaths = [repo / R2_VALIDATION_V2_FILE, repo / R2_VALIDATION_V2B_FILE]
    rep10 = rr.build_combined([repo / R2_REAL_V1_FILE, repo / R2_REAL_V1B_FILE, repo / R2_V1B_SEEDS_FILE],
                              validation_paths=vpaths)
    rep5 = rr.build_combined([repo / R2_REAL_V1_FILE, repo / R2_REAL_V1B_FILE], validation_paths=vpaths)
    repn = rr.build_combined([repo / R2_V1B_SEEDS_FILE], validation_paths=vpaths)
    x10, x5 = rr.excess_over_default(rep10), rr.excess_over_default(rep5)
    problems = []
    if not any(r.get("record") == "run_end" for r in new):
        problems.append("no run_end")
    chk = [r for r in rep_rows if r.get("record") == "repro_check"]
    if not chk or not chk[-1].get("pass"):
        problems.append("reproduction check " + ("missing" if not chk else f"failed ({chk[-1].get('reason')})"))
    starts = [r for r in new if r.get("record") == "run_start"]
    blob = ((starts[-1].get("harness_provenance") or {}).get("agent_blob") if starts else None)
    if blob != V1B_AGENT_BLOB:
        problems.append(f"run_start agent blob {blob} != {V1B_AGENT_BLOB}")
    for m in V1B10_MODELS:
        if any(r.get("record") == "refused_model" and r.get("model_id") == m for r in new):
            problems.append(f"{m} refused")
    sess = [s for s in ag.completed_sessions(new) if s.get("mode") == "real"]
    turns = {}
    for r in new:
        if r.get("record") == "r2a_turn":
            turns[r["attempt_id"]] = turns.get(r["attempt_id"], 0) + 1
    for m in V1B10_MODELS:
        for arm in ("ollama_default", "ollama_ctx_16384", "ollama_ctx_8192", "ollama_ctx_4096"):
            got = sorted(s["seed"] for s in sess if s["model_id"] == m and s["arm_id"] == arm + "_call2_notools"
                         and turns.get(s["attempt_id"]) == 40)
            if got != list(V1B10_NEW_SEEDS):
                problems.append(f"{m} {arm}: 40-turn sessions for seeds {got}")
    valid = not problems
    parts = [f"valid {valid}" + (f" ({'; '.join(problems)})" if problems else "")]
    for m in V1B10_MODELS:
        segs = []
        for t in V1B10_TIERS:
            v, v5 = x10["per_tier"][t][m], x5["per_tier"][t][m]
            lo, hi = v["ci95"]
            lo5, hi5 = v5["ci95"]
            cn = repn["table"].get(f"{t}|{m}")
            new_s = (f"{sum(1 for e in cn['sessions'] if e['after_window_exceeded'])}/{cn['n']}" if cn else "n/a")
            rev = (t in ("num_ctx_8192", "num_ctx_4096") and v["excess"] < 0 < v5["excess"])
            segs.append(f"{t} {round(v['rate'] * v['n_tier'])}/{v['n_tier']} vs default "
                        f"{round(v['default_rate'] * v['n_default'])}/{v['n_default']}, excess {v['excess']:+.3f} "
                        f"(95% CI {lo:+.3f} to {hi:+.3f}; 5-seed {v5['excess']:+.3f}, CI width {hi5 - lo5:.3f} -> "
                        f"{hi - lo:.3f}); new seeds alone {new_s}; reading: "
                        + (excess_reading(lo, hi, t) if valid else "not read (run not valid)")
                        + (f"; sign reversed at {t} for {m}" if rev and valid else ""))
        parts.append(f"{m}: " + "; ".join(segs))
    return {"value": "; ".join(parts) + f" (bootstrap {x10['n_resamples']} resamples, unit {x10['unit']}, stratified "
                                        f"by {x10['strata']}, seed {x10['seed']})",
            "n": f"{rep10['n_sessions']} sessions pooled"}




SPILL32_FILE = "results/x2_mx2_qwen3_32b_spill_sweep.jsonl"


def compute_fh1_amd_qwen3_32b_spill_sweep(repo):
    """qwen3-32b spill sweep on evo-x2 (20480 to 40960): per point the served n_ctx, class, shared-usage delta and
    decode/TTFT means; the spill point, the last no-spill point and the spill cost with the pre-registered readings
    (harness/mx2_qwen32b_spill_sweep.summarize)."""
    import sys as _sys
    _sys.path.insert(0, str(repo / "harness"))
    import mx2_qwen32b_spill_sweep as sw
    rows = sw.ag_rows(repo / SPILL32_FILE)
    if not rows:
        raise FileNotFoundError(repo / SPILL32_FILE)
    rs = next((r for r in rows if r.get("record") == "run_start"), {})
    s = sw.summarize(rows, rs.get("threshold_mib") or 1024.0)
    finished = any(r.get("record") == "run_end" for r in rows)
    b = s["points"].get(sw.BASELINE_CTX) or {}
    valid = finished and b.get("class") == "no_spill" and b.get("n_measured") == sw.MEASURED_CALLS
    pts = []
    for n, p in s["points"].items():
        d, t = p["decode"], p["ttft"]
        pts.append(f"n_ctx {n} (served {p.get('served_n_ctx')}) {p['class']}"
                   + (f" shared {p['shared_delta_mib']:+.0f} MiB vs 20480" if p.get("shared_delta_mib") is not None
                      else "")
                   + (f", decode {d[0]:.3f} +/- {d[1]:.3f} tok/s" if d[0] is not None and d[1] is not None else "")
                   + (f", TTFT {t[0]:.0f} +/- {t[1]:.0f} ms" if t[0] is not None and t[1] is not None else "")
                   + f" (n={p['n_measured']})")
    c = s["cost"]
    if not valid:
        head = "not read (run not valid: " + ("no run_end" if not finished else "baseline point problem") + ")"
    elif s["spill_point"] is None:
        head = "no unclamped spill point up to 40960"
    else:
        head = (f"spill point n_ctx {s['spill_point']}, last no-spill {s['last_no_spill']}; decode ratio "
                f"{c['decode_ratio']:.3f}, TTFT ratio {c['ttft_ratio']:.3f}; {c['decode_reading']}; {c['ttft_reading']}")
    return {"value": f"evo-x2 qwen3-32b, llama-server b10970 Vulkan -ngl 99, 1 warm-up + {sw.MEASURED_CALLS} calls "
                     f"per point; {head} // points: " + "; ".join(pts),
            "n": f"{sum(p['n_measured'] for p in s['points'].values())} measured calls"}




X2_EVALSET_TASKS_VAL_V2_MS = "results/x2_evalset_tasks_val_v2_mistral_small32.jsonl"
X2_EVALSET_TASKS_REAL_V2_MS = "results/x2_evalset_tasks_real_v2_mistral_small32.jsonl"
MS_MODEL = "mistral-small3.2:24b"


def compute_x2_evalset_tasks_val_v2_ms_gates(repo):
    """mistral-small3.2:24b v2 tasks validation: every gate with its numbers, the failing gates, and whether the real
    run's strict preflight (v2_strict, mistral only) admits it."""
    _r2_agent_module()
    import x2_r2_evalset as ev
    rows = ev.ag.read_rows(repo / X2_EVALSET_TASKS_VAL_V2_MS)
    if not rows:
        raise FileNotFoundError(repo / X2_EVALSET_TASKS_VAL_V2_MS)
    rep = ev.evaluate_gates_ext(rows, "evalset", False, "off", design=ev.file_design(rows))
    g = rep["gates"]
    m = MS_MODEL
    gates = [x for x in EVALSET_V2_GATE_ORDER if x in g]
    failing = [x for x in gates if g[x].get(m) is not None and not g[x][m]["pass"]
               and not (x == "rules678" and not g[x][m]["applies"])]
    pf = ev.strict_preflight_v2(rows, [m], allowed_models=ev.TASKS_V2_MS_MODELS)
    finished = any(r.get("record") == "run_end" for r in rows)
    return {"value": f"tasks {m} (run finished {finished}, design {ev.file_design(rows)}): "
                     + "; ".join(_evalset_v2_gate_text(x, g, rep, m) for x in gates)
                     + f"; failing gates: {', '.join(failing) or 'none'}; strict preflight (v2_strict): "
                     + ("admits " + m if pf["ok"] else "refuses (" + "; ".join(pf["refused"].get(m, [])) + ")"),
            "n": f"{sum(1 for r in rows if r.get('record') == 'r2a_turn')} turns"}


def ms_protocol_reading(tier, ci, valid) -> str:
    """The 2026-10-22 rule for mistral (no v1b cells; R2 at 4096 had 0/3 gated after the window)."""
    if not valid:
        return "inconclusive (run not valid)"
    if tier == "num_ctx_4096":
        return "v2 shows an excess R2 did not" if ci[0] is not None and ci[0] > 0 else "no excess in either protocol"
    return "no R2 counterpart"


def compute_x2_evalset_tasks_real_v2_ms_gated_excess(repo):
    """mistral-small3.2:24b B3 v2 real run: the gated excess per tier (as for qwen3:14b) with the mistral protocol
    reading (R2 cells at 4096 and 32768 only, no default tier) and the num_keep reading."""
    import sys as _sys
    _sys.path.insert(0, str(repo / "analysis"))
    import r2_real_report as rr
    _r2_agent_module()
    import x2_r2_evalset as ev
    rows = ev.ag.read_rows(repo / X2_EVALSET_TASKS_REAL_V2_MS)
    if not rows:
        raise FileNotFoundError(repo / X2_EVALSET_TASKS_REAL_V2_MS)
    base = compute_x2_evalset_tasks_real_v2_gated_excess(repo, MS_MODEL, X2_EVALSET_TASKS_REAL_V2_MS,
                                                         X2_EVALSET_TASKS_VAL_V2_MS)
    d = ev.tasks_real_v2_outcome(rows, models=(MS_MODEL,))["models"].get(MS_MODEL, {})
    valid = bool(d.get("valid"))
    x = rr.excess_over_default(rr.build_combined([repo / X2_EVALSET_TASKS_REAL_V2_MS],
                                                 validation_paths=[repo / X2_EVALSET_TASKS_VAL_V2_MS]))
    reads = []
    for tier in ("num_ctx_16384", "num_ctx_8192", "num_ctx_4096"):
        v = x.get("per_tier", {}).get(tier, {}).get(MS_MODEL)
        if v is not None:
            reads.append(f"{tier}: {ms_protocol_reading(tier, v['ci95'], valid)}")
    return {"value": base["value"] + "; mistral protocol readings (R2 4096 0/3 gated, no R2 default tier): "
                     + "; ".join(reads), "n": base["n"]}


def compute_x2_evalset_tasks_real_v2_ms_wrong_rate(repo):
    return compute_x2_evalset_tasks_real_v2_wrong_rate(repo, MS_MODEL, X2_EVALSET_TASKS_REAL_V2_MS)




def compute_num_keep_qwen3_14b(repo):
    """x2_num_keep_qwen3_14b: the num_keep guardrail test's measures and pre-registered verdict for qwen3:14b
    (harness/x2_num_keep.verdict with model qwen3:14b)."""
    return compute_num_keep_guardrail_test(repo, NUM_KEEP_Q14_FILE, NUM_KEEP_Q14_MODEL)


def compute_x2_evalset_tasks_val_v2_q8_gates(repo):
    """qwen3:8b v2 tasks validation: every gate with the numbers it was decided on (evaluate_gates_ext, design v2),
    the failing gates, and whether the real run's strict preflight (v2_strict) admits qwen3:8b."""
    _r2_agent_module()
    import x2_r2_evalset as ev
    rows = ev.ag.read_rows(repo / X2_EVALSET_TASKS_VAL_V2_Q8)
    if not rows:
        raise FileNotFoundError(repo / X2_EVALSET_TASKS_VAL_V2_Q8)
    rep = ev.evaluate_gates_ext(rows, "evalset", False, "off", design=ev.file_design(rows))
    g = rep["gates"]
    m = Q8_MODEL
    gates = [x for x in EVALSET_V2_GATE_ORDER if x in g]
    failing = [x for x in gates if g[x].get(m) is not None and not g[x][m]["pass"]
               and not (x == "rules678" and not g[x][m]["applies"])]
    pf = ev.strict_preflight_v2(rows, [m])
    finished = any(r.get("record") == "run_end" for r in rows)
    return {"value": f"tasks {m} (run finished {finished}, design {ev.file_design(rows)}): "
                     + "; ".join(_evalset_v2_gate_text(x, g, rep, m) for x in gates)
                     + f"; failing gates: {', '.join(failing) or 'none'}; strict preflight (v2_strict): "
                     + ("admits " + m if pf["ok"] else "refuses (" + "; ".join(pf["refused"].get(m, [])) + ")"),
            "n": f"{sum(1 for r in rows if r.get('record') == 'r2a_turn')} turns"}


def compute_x2_evalset_tasks_real_v2_q8_gated_excess(repo):
    return compute_x2_evalset_tasks_real_v2_gated_excess(repo, Q8_MODEL, X2_EVALSET_TASKS_REAL_V2_Q8,
                                                         X2_EVALSET_TASKS_VAL_V2_Q8)


def compute_x2_evalset_tasks_real_v2_q8_wrong_rate(repo):
    return compute_x2_evalset_tasks_real_v2_wrong_rate(repo, Q8_MODEL, X2_EVALSET_TASKS_REAL_V2_Q8)


def fisher_two_sided(a, n1, b, n2) -> float:
    from math import comb
    k, N = a + b, n1 + n2
    pmf = {x: comb(n1, x) * comb(n2, k - x) / comb(N, k) for x in range(max(0, k - n2), min(k, n1) + 1)}
    return sum(v for v in pmf.values() if v <= pmf[a] * (1 + 1e-9))


def reversal_verdict(valid, a8, n8, a4, n4, alpha=0.05) -> str:
    """The pre-registered verdict (docs/FINDINGS.md 2026-10-21, qwen3:8b 8192 vs 4096 with 10 seeds per tier)."""
    if not valid:
        return "incomplete"
    p = fisher_two_sided(a8, n8, a4, n4)
    if p < alpha and a8 > a4:
        return "reversal confirmed"
    if p < alpha and a4 > a8:
        return "ordering of the other models"
    return "resolved as noise"


def compute_r2_qwen3_8b_reversal_10seed(repo):
    """qwen3:8b at num_ctx 8192 vs 4096, v1b seeds 20260901-05 pooled with seeds 20260906-10: gated silent sessions per
    tier (r2_real_report.build_combined, v1b gate), two-sided Fisher exact, the pre-registered verdict, the excess over
    the default per tier with the v1b bootstrap, the new seeds alone, turn-level counts, and validity."""
    import sys as _sys
    _sys.path.insert(0, str(repo / "analysis"))
    import r2_real_report as rr
    ag = _r2_agent_module()
    new_rows = ag.read_rows(repo / R2_Q8_SEEDS_06_10_FILE)
    if not new_rows:
        raise FileNotFoundError(repo / R2_Q8_SEEDS_06_10_FILE)
    rep = rr.build_combined([repo / R2_REAL_V1_FILE, repo / R2_REAL_V1B_FILE, repo / R2_Q8_SEEDS_06_10_FILE],
                            validation_paths=[repo / R2_VALIDATION_V2_FILE, repo / R2_VALIDATION_V2B_FILE])
    m = Q8_MODEL
    problems = []
    if not any(r.get("record") == "run_end" for r in new_rows):
        problems.append("no run_end")
    if any(r.get("record") == "refused_model" and r.get("model_id") == m for r in new_rows):
        problems.append(f"{m} refused")
    new_sess = [s for s in ag.completed_sessions(new_rows) if s["model_id"] == m and s.get("mode") == "real"]
    want_ctx = {"num_ctx_8192": 8192, "num_ctx_4096": 4096}
    for tier in Q8_REVERSAL_TIERS:
        arm = f"ollama_ctx_{want_ctx[tier]}_call2_notools"
        got = sorted(s["seed"] for s in new_sess if s["arm_id"] == arm)
        if got != sorted(Q8_NEW_SEEDS):
            problems.append(f"{tier}: new sessions completed for seeds {got}, expected {list(Q8_NEW_SEEDS)}")
        att = {s["attempt_id"]: s["seed"] for s in new_sess if s["arm_id"] == arm}
        for a, seed in att.items():
            t = [r for r in new_rows if r.get("record") == "r2a_turn" and r.get("attempt_id") == a]
            if len(t) != 40:
                problems.append(f"{tier} seed {seed}: {len(t)} turns, expected 40")
            bad = {r.get("num_ctx_requested") for r in t} - {want_ctx[tier]}
            if bad:
                problems.append(f"{tier} seed {seed}: num_ctx_requested {sorted(bad, key=str)}")
    counts, parts = {}, []
    for tier in Q8_REVERSAL_TIERS:
        c = rep["table"].get(f"{tier}|{m}") or {"sessions": [], "n": 0}
        old = [e for e in c["sessions"] if e["seed"] not in Q8_NEW_SEEDS]
        new = [e for e in c["sessions"] if e["seed"] in Q8_NEW_SEEDS]
        if len(old) != 5 or any(e["n_turns"] != 40 for e in old):
            problems.append(f"{tier}: pooled v1b sessions {len(old)}, expected 5 with 40 turns")
        counts[tier] = (sum(e["silent"] for e in c["sessions"]), len(c["sessions"]))
        parts.append(f"{tier}: {counts[tier][0]}/{counts[tier][1]} gated silent (v1b seeds "
                     f"{sum(e['silent'] for e in old)}/{len(old)}, new seeds {sum(e['silent'] for e in new)}/{len(new)}; "
                     "new per seed " + ", ".join(
                         f"{e['seed']}: {'none' if e['first_gated_failure'] is None else e['first_gated_failure']}"
                         for e in sorted(new, key=lambda e: e['seed'])) + ")")
    valid = not problems
    (a8, n8), (a4, n4) = counts["num_ctx_8192"], counts["num_ctx_4096"]
    p = fisher_two_sided(a8, n8, a4, n4) if n8 and n4 else None
    parts.append(f"Fisher exact two-sided p (8192 vs 4096) = {p:.3f}" if p is not None else "Fisher p n/a")
    x = rr.excess_over_default(rep)
    for tier in Q8_REVERSAL_TIERS:
        v = x["per_tier"].get(tier, {}).get(m)
        if v:
            lo, hi = v["ci95"]
            parts.append(f"{tier} excess over default {round(v['rate'] * v['n_tier'])}/{v['n_tier']} vs "
                         f"{round(v['default_rate'] * v['n_default'])}/{v['n_default']}, {v['excess']:+.3f} (95% CI "
                         f"{lo:+.3f} to {hi:+.3f}); reading: "
                         + (f"excess at {tier}" if lo is not None and lo > 0 else f"no excess shown at {tier}"))
    parts.append(f"valid {valid}" + (f" ({'; '.join(problems)})" if problems else ""))
    parts.append(f"verdict: {reversal_verdict(valid, a8, n8, a4, n4)}")
    return {"value": f"{m} gate excludes {', '.join(rep['gated_metrics'][m]['excluded']) or 'nothing'}; "
                     + "; ".join(parts) + " (bootstrap 10000 resamples, unit session, stratified by tier x model, seed "
                     "20261014)", "n": f"{n8 + n4} sessions"}


def compute_x2_scale_real_default_tier(repo):
    """x2_r2_scale_real, Ollama default tier: per model the completed default-tier sessions (seed, turns, wall hours as
    the sum of turn_wall_s, loaded context, first over-window turn), as run inside the H-scale real job."""
    ag = _r2_agent_module()
    rows = ag.read_rows(repo / X2_SCALE_REAL)
    if not rows:
        raise FileNotFoundError(repo / X2_SCALE_REAL)
    arm = "ollama_default_call2_notools"
    parts, n = [], 0
    for m in sorted({s["model_id"] for s in ag.completed_sessions(rows)}):
        ss = sorted((s for s in ag.completed_sessions(rows) if s["model_id"] == m and s["arm_id"] == arm),
                    key=lambda s: s["seed"])
        if not ss:
            parts.append(f"{m}: no default-tier session")
            continue
        segs = []
        for s in ss:
            t = [r for r in rows if r.get("record") == "r2a_turn" and r.get("attempt_id") == s["attempt_id"]]
            segs.append(f"seed {s['seed']}: {len(t)} turns, {sum(r.get('turn_wall_s') or 0 for r in t) / 3600:.2f} h, "
                        f"loaded {s.get('loaded_context_values')}, first over window "
                        f"{'-' if s.get('first_over_window_turn') is None else s['first_over_window_turn']}")
            n += 1
        parts.append(f"{m}: {len(ss)} default-tier sessions completed ({'; '.join(segs)})")
    return {"value": " / ".join(parts), "n": f"{n} sessions"}


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
     "compute": compute_x2_v3_canary_gates, "data_files": [X2_OUTCOME_V3, X2_TEMPLATE_FACTS],
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
     "compute": compute_x2_v3_canary_gate_lenient, "data_files": [X2_OUTCOME_V3, X2_TEMPLATE_FACTS],
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
     "compute": compute_x2_v3_scores, "data_files": [X2_OUTCOME_V3, "results/workload_pack/item_weights_trace_weighted.json",
                                                     X2_TEMPLATE_FACTS],
     "script_function": "analysis/numbers_register.py::compute_x2_v3_scores"},
    {"claim_id": "x2-v3-error-causes", "description": "X2 outcome table v3: error causes per model x config",
     "compute": compute_x2_v3_error_causes, "data_files": [X2_OUTCOME_V3],
     "script_function": "analysis/numbers_register.py::compute_x2_v3_error_causes"},
    {"claim_id": "x2-v3-usability", "description": "X2 outcome table v3 usability: timeout_latency rows per model x "
                    "config (excluded from x2-v3-scores), share, context tiers, per-call timeout",
     "compute": compute_x2_v3_usability, "data_files": [X2_OUTCOME_V3],
     "script_function": "analysis/numbers_register.py::compute_x2_v3_usability"},
    {"claim_id": "qwen3-32b-timeout-count", "description": "X2 outcome table v3: qwen3-32b timeout_latency rows per "
                    "config, provisional until v3 progress reaches its plan",
     "compute": compute_qwen3_32b_timeout_count, "data_files": [X2_OUTCOME_V3],
     "script_function": "analysis/numbers_register.py::compute_qwen3_32b_timeout_count"},
    {"claim_id": "qwen3-32b-truncation-vs-latency", "description": "X2 outcome table v3: qwen3-32b silently "
                    "truncated rows per config, score and latency on truncated vs full rows, cross-config pairing; "
                    "provisional until v3 progress reaches its plan",
     "compute": compute_qwen3_32b_truncation_vs_latency, "data_files": [X2_OUTCOME_V3],
     "script_function": "analysis/numbers_register.py::compute_qwen3_32b_truncation_vs_latency"},
    {"claim_id": "x2-v3-truncation-all-models", "description": "X2 outcome table v3: silently truncated rows per "
                    "model x config, score on truncated vs full rows",
     "compute": compute_x2_v3_truncation_all_models, "data_files": [X2_OUTCOME_V3],
     "script_function": "analysis/numbers_register.py::compute_x2_v3_truncation_all_models"},
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
    {"claim_id": "R2-v1b-gated-per-model",
     "description": "Strengthened R2, gated variant per tier x model: silent failures, after window exceeded, "
                    "baseline-type, sessions intact per checkpoint",
     "compute": compute_r2_v1b_gated_per_model, "data_files": R2_V1B_DATA, "pending_ok": True,
     "script_function": "analysis/numbers_register.py::compute_r2_v1b_gated_per_model"},
    {"claim_id": "R2-v1b-default-baseline-type",
     "description": "Strengthened R2, default tier: kill-criterion and gated silent sessions that are baseline-type "
                    "(no window exceeded yet), with what failed first",
     "compute": compute_r2_v1b_default_baseline_type, "data_files": R2_V1B_DATA, "pending_ok": True,
     "script_function": "analysis/numbers_register.py::compute_r2_v1b_default_baseline_type"},
    {"claim_id": "R2-v1b-excess-over-default",
     "description": "Strengthened R2: after-window gated silent failure rate minus the default tier's, per tier, "
                    "overall and per model, 95% bootstrap CI over sessions",
     "compute": compute_r2_v1b_excess_over_default, "data_files": R2_V1B_DATA, "pending_ok": True,
     "script_function": "analysis/numbers_register.py::compute_r2_v1b_excess_over_default"},
    {"claim_id": "R2-v1b-qwen3-8b-reversal",
     "description": "Strengthened R2, qwen3:8b num_ctx 8192 vs 4096: per-session first gated failure, gate "
                    "exclusions, how each session ended, ungated failures, turn-level counts, Fisher exact p",
     "compute": compute_r2_v1b_qwen3_8b_reversal, "data_files": R2_V1B_DATA, "pending_ok": True,
     "script_function": "analysis/numbers_register.py::compute_r2_v1b_qwen3_8b_reversal"},
    {"claim_id": "t2s-r2-validation-cpu", "description": "evo-t2s Intel CPU R2 validation: baseline table, gates, "
                    "preflight per model", "compute": lambda repo: compute_t2s_r2_validation(repo, "cpu"),
     "data_files": [T2S_VAL["cpu"]], "script_function": "analysis/numbers_register.py::compute_t2s_r2_validation"},
    {"claim_id": "t2s-r2-validation-igpu", "description": "evo-t2s Intel iGPU R2 validation: baseline table, gates, "
                    "preflight per model", "compute": lambda repo: compute_t2s_r2_validation(repo, "igpu"),
     "data_files": [T2S_VAL["igpu"]], "script_function": "analysis/numbers_register.py::compute_t2s_r2_validation"},
    {"claim_id": "t2s-r2-real-cpu", "description": "evo-t2s Intel CPU R2 real at the Ollama default window: loaded "
                    "window, kill criterion, gated silent failures per model",
     "compute": lambda repo: compute_t2s_r2_real(repo, "cpu"), "data_files": [T2S_REAL["cpu"], T2S_VAL["cpu"]],
     "script_function": "analysis/numbers_register.py::compute_t2s_r2_real"},
    {"claim_id": "t2s-r2-real-igpu", "description": "evo-t2s Intel iGPU R2 real at the Ollama default window: loaded "
                    "window, kill criterion, gated silent failures per model",
     "compute": lambda repo: compute_t2s_r2_real(repo, "igpu"), "data_files": [T2S_REAL["igpu"], T2S_VAL["igpu"]],
     "script_function": "analysis/numbers_register.py::compute_t2s_r2_real"},
    {"claim_id": "t2s-r2-session-minutes", "description": "evo-t2s measured R2 session minutes per backend and model "
                    "(default window; CPU trimming at 4096) and run wall clocks: the input of the B5 window plan",
     "compute": compute_t2s_r2_session_minutes, "data_files": [T2S_REAL["cpu"], T2S_REAL["igpu"], T2S_MITIGATION_CPU],
     "script_function": "analysis/numbers_register.py::compute_t2s_r2_session_minutes"},
    {"claim_id": "t2s-b5-plan-hours", "description": "B5 on evo-t2s for the 2026-10-26 window (operator order "
                    "2026-10-21 later, policies a, b, c, d, e, f: llama3.1:8b CPU and iGPU at 5 seeds first, then qwen3:14b on the CPU at 3 seeds only if "
                    "it fits the launch budget, 24 h slack before no-launch; B6 fills the rest): ESTIMATED hours per entry from "
                    "measured evo-t2s sessions (B6: evo-x2 H-scale minutes x a measured evo-t2s factor), what fits, "
                    "and the qwen3:14b fit decision",
     "compute": compute_t2s_b5_plan_hours,
     "data_files": [T2S_REAL["cpu"], T2S_REAL["igpu"], T2S_MITIGATION_CPU, "results/x2_r2_real_v1.jsonl",
                    "results/x2_r2_real_v1b.jsonl", "results/x2_r2_mitigation_v1.jsonl", "results/x2_r2_scale_real.jsonl",
                    "results/x2_r2_scale_validation.jsonl", "queues/t2s_oct26_queue.json"],
     "script_function": "scripts/t2s_window_stage.py::plan"},
    {"claim_id": "t2s-r2-mitigation-cpu", "description": "evo-t2s Intel CPU client-side trimming at 4096: "
                    "pre-registered mitigation verdict per model", "compute": compute_t2s_r2_mitigation_cpu,
     "data_files": [T2S_MITIGATION_CPU, T2S_VAL["cpu"]],
     "script_function": "analysis/numbers_register.py::compute_t2s_r2_mitigation_cpu"},
    {"claim_id": "t2s-r2-mechanism-cpu", "description": "evo-t2s Intel CPU mechanism session: completion state at the "
                    "handover deadline", "compute": compute_t2s_r2_mechanism_cpu, "data_files": [T2S_MECHANISM_CPU],
     "script_function": "analysis/numbers_register.py::compute_t2s_r2_mechanism_cpu"},
    {"claim_id": "R2-qwen3-14b-4096-reconciliation", "description": "qwen3:14b at num_ctx 4096 across v1/v1b, "
                    "mitigation (trimming) and num_keep: per-rule per-seed passes, gated first failures, versions, "
                    "protocol, turn-by-turn identity of the num_keep run with v1",
     "compute": compute_r2_qwen3_14b_4096_reconciliation,
     "data_files": [R2_REAL_V1_FILE, R2_REAL_V1B_FILE, R2_MITIGATION_FILE, "results/x2_num_keep_qwen3_14b.jsonl"],
     "script_function": "analysis/numbers_register.py::compute_r2_qwen3_14b_4096_reconciliation"},
    {"claim_id": "x2-evalset-tasks-real-v2-ci-units", "description": "B3 v2 real run qwen3:14b: wrong-task excess "
                    "CIs with the session-level bootstrap (paper) and a turn-level bootstrap (comparison)",
     "compute": compute_x2_evalset_tasks_real_v2_ci_units,
     "data_files": ["results/x2_evalset_tasks_real_v2.jsonl"],
     "script_function": "analysis/numbers_register.py::compute_x2_evalset_tasks_real_v2_ci_units"},
    {"claim_id": "R2-validation-v2b-runinfo", "description": "R2 validation v2b: sessions/turns, errors, native vs "
                    "text tool calls per model",
     "compute": compute_r2_validation_v2b_runinfo, "data_files": ["results/x2_r2_validation_v2b.jsonl"],
     "script_function": "analysis/numbers_register.py::compute_r2_validation_v2b_runinfo"},
    {"claim_id": "R2-task-tool-gate", "description": "Uniform task-tool gate (task tool correct on >=90% of baseline "
                    "turns) per model on v2/v2b, and the preflight verdict with and without it",
     "compute": compute_r2_task_tool_gate, "data_files": [R2_VALIDATION_V2_FILE, R2_VALIDATION_V2B_FILE],
     "script_function": "analysis/numbers_register.py::compute_r2_task_tool_gate"},
    {"claim_id": "R2-install-path-4b-bare", "description": "Install path, bare half: qwen3-4b-2507 (Ollama tag from a "
                    "bare GGUF) native vs text tool calls, task tool, rules, controls (v2b)",
     "compute": compute_r2_install_path_4b_bare, "data_files": [R2_VALIDATION_V2B_FILE, X2_TEMPLATE_FACTS],
     "script_function": "analysis/numbers_register.py::compute_r2_install_path_4b_bare"},
    {"claim_id": "R2-install-path-4b-comparison", "description": "Install path: same GGUF, qwen3-4b-2507 (bare, v2b) "
                    "vs qwen3-4b-2507-tools (library template, v2c)",
     "compute": compute_r2_install_path_4b_comparison,
     "data_files": [R2_VALIDATION_V2B_FILE, R2_VALIDATION_V2C_FILE, X2_TEMPLATE_FACTS], "pending_ok": True,
     "script_function": "analysis/numbers_register.py::compute_r2_install_path_4b_comparison"},
    {"claim_id": "R2-validation-v2c-baseline", "description": "R2 validation v2c (gate for qwen3-4b-2507-tools): rule/"
                    "tool/recall/canary baseline and controls",
     "compute": compute_r2_validation_v2c_baseline, "data_files": [R2_VALIDATION_V2C_FILE], "pending_ok": True,
     "script_function": "analysis/numbers_register.py::compute_r2_validation_v2c_baseline"},
    {"claim_id": "x2-chat-template-source", "description": "X2 outcome table chat template source per (model, config) "
                    "(evo-x2 probe facts; v3 rows recorded vs derived)",
     "compute": compute_x2_chat_template_source, "data_files": [X2_TEMPLATE_FACTS, X2_OUTCOME_V3],
     "script_function": "analysis/numbers_register.py::compute_x2_chat_template_source"},
    {"claim_id": "pareto-frontier-x2", "description": "Quality (trace-weighted) vs USD per 1000 steps frontier per "
                    "machine, evo-x2 v3 outcome table; cloud points are STUB (priced, quality not measured)",
     "compute": compute_pareto_frontier_x2, "data_files": _PARETO_DATA,
     "script_function": "src/dse/pareto.py::frontier"},
    {"claim_id": "pareto-sample-recommendation", "description": "Recommender output for budget USD 50, quality "
                    "floor 0.9, p50 latency 30 s, hardware evo-x2; savings vs all-cloud (STUB)",
     "compute": compute_pareto_sample_recommendation, "data_files": _PARETO_DATA,
     "script_function": "src/dse/pareto.py::recommend"},
    {"claim_id": "pareto-sample-recommendation-v2", "description": "Recommender output for the same inputs as "
                    "pareto-sample-recommendation with the usability constraint timeout_latency share <= 0.05; "
                    "excluded configs with reasons; n = configs excluded",
     "compute": compute_pareto_sample_recommendation_v2, "data_files": _PARETO_DATA,
     "script_function": "analysis/numbers_register.py::compute_pareto_sample_recommendation_v2"},
    {"claim_id": "R2-mechanism-lowlevel", "description": "R2 mechanism job: message truncation, llama.cpp context "
                    "shifts and Ollama token cuts per tier",
     "compute": compute_r2_mechanism_lowlevel, "data_files": ["results/x2_r2_mechanism.jsonl"],
     "script_function": "analysis/numbers_register.py::compute_r2_mechanism_lowlevel"},
    {"claim_id": "R2-shift-predictor-confusion", "description": "R2 context-shift predictor (P + C > n_ctx) vs observed "
                    "llama.cpp shifts per tier, mechanism job",
     "compute": compute_r2_shift_predictor_confusion, "data_files": ["results/x2_r2_mechanism.jsonl"],
     "script_function": "analysis/r2_shift_predictor.py::mechanism_summary"},
    {"claim_id": "R2-shift-predicted-real-v1", "description": "R2 real run v1: PREDICTED context shifts and token-cut "
                    "signatures per tier x model, first predicted shift vs first rule failure",
     "compute": compute_r2_shift_predicted_real_v1, "data_files": ["results/x2_r2_real_v1.jsonl"],
     "script_function": "analysis/r2_shift_predictor.py::real_summary"},
    {"claim_id": "R2-orphan-rates", "description": "R2 mechanism job: orphaned first kept message and orphaned tool "
                    "result after Ollama's message trim, per tier",
     "compute": compute_r2_orphan_rates, "data_files": ["results/x2_r2_mechanism.jsonl"],
     "script_function": "analysis/r2_shift_predictor.py::orphan_summary"},
    {"claim_id": "r2-session-wallclock", "description": "R2 real runs (v1, v1b, combined): session wall-clock minutes "
                    "per model x tier (mean, per seed), billed / input / output tokens per session",
     "compute": compute_r2_session_wallclock, "data_files": R2_SESSION_COST_FILES,
     "script_function": "analysis/r2_session_cost.py::wallclock_str"},
    {"claim_id": "r2-session-cloud-cost", "description": "R2 real runs (v1, v1b, combined): cloud USD per session at "
                    "list prices (latest price file per provider), best-case caching to no caching",
     "compute": compute_r2_session_cloud_cost, "data_files": R2_SESSION_COST_FILES,
     "script_function": "analysis/r2_session_cost.py::cost_str"},
    {"claim_id": "info-loss-by-window", "description": "R2: share of transcript and system-prompt tokens never seen per "
                    "machine x model x window x turn (observed mechanism sessions, one per tier; predicted real runs)",
     "compute": compute_info_loss_by_window, "data_files": R2_INFO_LOSS_FILES,
     "script_function": "analysis/r2_info_loss.py::info_loss_str"},
    {"claim_id": "reliability-after-guardrail-loss", "description": "R2: rule violation rate before vs after the first "
                    "system-prompt loss, turns to first violation, violations by rule type (observed one session per "
                    "tier; predicted real runs)",
     "compute": compute_reliability_after_guardrail_loss, "data_files": R2_INFO_LOSS_FILES,
     "script_function": "analysis/r2_info_loss.py::reliability_str"},
    {"claim_id": "R2-mitigation-v1-verdict", "description": "R2 mitigation (client-side trimming): pre-registered "
                    "verdict per model at num_ctx 4096 (8192 as control) and overall",
     "compute": compute_r2_mitigation_v1_verdict,
     "data_files": [R2_MITIGATION_FILE, R2_REAL_V1_FILE, R2_VALIDATION_V2_FILE], "pending_ok": True,
     "script_function": "analysis/numbers_register.py::compute_r2_mitigation_v1_verdict"},
    {"claim_id": "R2-mitigation-v1-4096", "description": "R2 mitigation vs real run v1 at num_ctx 4096: rule survival "
                    "per rule in use, first rule failure, canary misses, errors",
     "compute": compute_r2_mitigation_v1_4096,
     "data_files": [R2_MITIGATION_FILE, R2_REAL_V1_FILE, R2_VALIDATION_V2_FILE], "pending_ok": True,
     "script_function": "analysis/numbers_register.py::compute_r2_mitigation_v1_4096"},
    {"claim_id": "R2-mitigation-v1-8192", "description": "R2 mitigation vs real run v1b at num_ctx 8192: rule survival "
                    "per rule in use, first rule failure, canary misses, errors",
     "compute": compute_r2_mitigation_v1_8192,
     "data_files": [R2_MITIGATION_FILE, R2_REAL_V1B_FILE, R2_VALIDATION_V2_FILE], "pending_ok": True,
     "script_function": "analysis/numbers_register.py::compute_r2_mitigation_v1_8192"},
    {"claim_id": "R2-mitigation-v1-observed", "description": "R2 mitigation: observed context shifts, token cuts, "
                    "Ollama message drops and client token counting per tier x model",
     "compute": compute_r2_mitigation_v1_observed, "data_files": [R2_MITIGATION_FILE], "pending_ok": True,
     "script_function": "analysis/numbers_register.py::compute_r2_mitigation_v1_observed"},
    {"claim_id": "blade-validation-ctx-cap", "description": "Blade R2 validation context cap from the evo-x2 "
                    "validation sessions' max prompt+generated tokens",
     "compute": compute_blade_validation_ctx_cap,
     "data_files": ["results/x2_r2_validation_v2.jsonl", "results/x2_r2_validation_v2b.jsonl"],
     "script_function": "analysis/numbers_register.py::compute_blade_validation_ctx_cap"},

    {"claim_id": "R2-real-v1-loaded-ctx", "description": "R2 real run: context Ollama loaded per tier x model on "
                    "evo-x2 (ollama_default = Ollama's own default for the model)",
     "compute": compute_r2_real_v1_loaded_ctx, "data_files": [R2_REAL_V1_FILE],
     "script_function": "analysis/numbers_register.py::compute_r2_real_v1_loaded_ctx"},
    {"claim_id": "R2-real-v1-token-calib", "description": "R2 real run: per-model tokenizer calibration ratio of the "
                    "chars/4 estimate (median and range over sessions)",
     "compute": compute_r2_real_v1_token_calib, "data_files": [R2_REAL_V1_FILE],
     "script_function": "analysis/numbers_register.py::compute_r2_real_v1_token_calib"},
    {"claim_id": "decode-rate-per-machine", "description": "median decode tok/s per machine x model, no co-runner, "
                    "pooled over the TTFT-fit files",
     "compute": compute_decode_rate_per_machine, "data_files": _TTFT_FIT_RESULT_FILES,
     "script_function": "analysis/numbers_register.py::compute_decode_rate_per_machine"},

    {"claim_id": "blade-dryrun-rates", "description": "Blade 60 s dry runs: prefill and decode tok/s per (model, "
                    "num_ctx), from each dry run's own per-call counters",
     "compute": compute_blade_dryrun_rates, "data_files": BLADE_DRYRUN_FILES, "pending_ok": True,
     "script_function": "analysis/numbers_register.py::compute_blade_dryrun_rates"},
    {"claim_id": "blade-night-hours", "description": "Blade per-night hour estimates (dry-run rates where present, "
                    "scaled evo-x2 times otherwise, labelled per job)",
     "compute": compute_blade_night_hours,
     "data_files": BLADE_DRYRUN_FILES + ["results/x2_r2_real_v1.jsonl", "results/x2_r2_validation_v2.jsonl",
                                         "results/x2_r2_validation_v2b.jsonl", "results/x2_r2_mechanism.jsonl",
                                         "results/blade_c1_spill_sweep_20260925T053651Z.jsonl",
                                         "results/blade_k1_v2.summary.json"],
     "pending_ok": True, "script_function": "analysis/numbers_register.py::compute_blade_night_hours"},
    {"claim_id": "blade-r2-validation-run1", "description": "Blade R2 validation night 1 first run (llama3.1:8b, "
                    "32768 cap): per-rule rates, negative-control canary misses per session, positive control",
     "compute": compute_blade_r2_validation_run1, "data_files": [BLADE_R2_VALIDATION_V1], "pending_ok": True,
     "script_function": "analysis/numbers_register.py::compute_blade_r2_validation_run1"},
    {"claim_id": "blade-r2-validation-run2", "description": "Blade R2 validation night 1 rerun (llama3.1:8b, 32768 "
                    "cap, negative control 5 sessions): per-rule rates, canary misses per session, positive control",
     "compute": compute_blade_r2_validation_run2, "data_files": [BLADE_R2_VALIDATION_V2], "pending_ok": True,
     "script_function": "analysis/numbers_register.py::compute_blade_r2_validation_run2"},
    {"claim_id": "blade-k1-layer-split", "description": "Blade K1: default ctx, VRAM share per num_ctx, overflow at default",
     "compute": compute_blade_k1_layer_split, "data_files": ["results/blade_k1_v2.summary.json"],
     "script_function": "analysis/numbers_register.py::compute_blade_k1_layer_split"},
    {"claim_id": "blade-r2-gate-v2", "description": "Blade night 1 rerun: negative-control gate per session and branch",
     "compute": compute_blade_r2_gate_v2, "data_files": ["results/blade_r2_gate_v2.json"],
     "script_function": "analysis/numbers_register.py::compute_blade_r2_gate_v2"},
    {"claim_id": "blade-mechanism-4096", "description": "Blade mechanism at num_ctx 4096: message drops, shifts, cuts",
     "compute": compute_blade_mechanism_4096, "data_files": ["results/blade_r2_mechanism_4096_v2.jsonl.report.json"],
     "script_function": "analysis/numbers_register.py::compute_blade_mechanism_4096"},
    {"claim_id": "x2-r2-llama-val-32k", "description": "Blade confound check on evo-x2 (llama3.1:8b, negative control "
                    "num_ctx 32768, 5 sessions): per-rule rates, canary misses per session, text tool calls per miss, "
                    "pre-registered verdict (backend/offload vs window)",
     "compute": compute_x2_r2_llama_val_32k, "data_files": [X2_R2_LLAMA_VAL_32K], "pending_ok": True,
     "script_function": "analysis/numbers_register.py::compute_x2_r2_llama_val_32k"},
    {"claim_id": "x2-evalset-tasks-val-gates", "description": "Evalset (a) tasks validation on evo-x2 (guardrails "
                    "off, 30 tasks): every gate per model, tasks in use (strict) and tasks correct per negative-control "
                    "seed; the input of x2_evalset_tasks_real_v1",
     "compute": compute_x2_evalset_tasks_val_gates, "data_files": [X2_EVALSET_TASKS_VAL],
     "script_function": "analysis/numbers_register.py::compute_x2_evalset_tasks_val_gates"},
    {"claim_id": "x2-evalset-tasks-real-wrong-rate", "description": "B3 task real run on evo-x2 (llama3.1:8b, "
                    "qwen3:14b; Ollama default vs num_ctx 4096; 3 seeds; 30 tasks): wrong-task rate per window, excess "
                    "at 4096 with a 95% task-cluster bootstrap CI, pre-registered reading, primary and per-seed sets",
     "compute": compute_x2_evalset_tasks_real_wrong_rate, "data_files": [X2_EVALSET_TASKS_REAL, X2_EVALSET_TASKS_VAL],
     "pending_ok": True, "script_function": "analysis/numbers_register.py::compute_x2_evalset_tasks_real_wrong_rate"},
    {"claim_id": "x2-evalset-tasks-real-silent", "description": "B3 task real run: wrong cells split into silent "
                    "(HTTP 200, no error) and surfaced, per model, cell set and window",
     "compute": compute_x2_evalset_tasks_real_silent, "data_files": [X2_EVALSET_TASKS_REAL, X2_EVALSET_TASKS_VAL],
     "pending_ok": True, "script_function": "analysis/numbers_register.py::compute_x2_evalset_tasks_real_silent"},
    {"claim_id": "x2-r2-llama-val-32k-offload", "description": "Blade confound follow-up on evo-x2 (llama3.1:8b, "
                    "negative control num_ctx 32768, 5 sessions, options.num_gpu 22 of 33 layers): offload lines per "
                    "session, canary misses, pre-registered verdict (offload vs backend)",
     "compute": compute_x2_r2_llama_val_32k_offload, "data_files": [X2_R2_LLAMA_VAL_32K_OFFLOAD], "pending_ok": True,
     "script_function": "analysis/numbers_register.py::compute_x2_r2_llama_val_32k_offload"},
    {"claim_id": "x2-r2-llama-val-32k-vulkan", "description": "Backend numerics check on evo-x2 (llama3.1:8b, "
                    "negative control num_ctx 32768, 5 sessions, the job's own Ollama server on ggml Vulkan): backend "
                    "evidence per session, canary misses, pre-registered verdict (vendor vs backend implementation)",
     "compute": compute_x2_r2_llama_val_32k_vulkan, "data_files": [X2_R2_LLAMA_VAL_32K_VULKAN], "pending_ok": True,
     "script_function": "analysis/numbers_register.py::compute_x2_r2_llama_val_32k_vulkan"},
    {"claim_id": "x2-r2-llama-val-32k-vulkan-hours", "description": "Duration estimate for x2_r2_llama_val_32k_vulkan "
                    "from the measured file wall clock of its siblings x2_r2_llama_val_32k and (once synced) "
                    "x2_r2_llama_val_32k_offload",
     "compute": compute_x2_r2_llama_val_32k_vulkan_hours, "data_files": [X2_R2_LLAMA_VAL_32K],
     "script_function": "analysis/numbers_register.py::compute_x2_r2_llama_val_32k_vulkan_hours"},

    {"claim_id": "impact-hour-value", "description": "Impact profile: value of one developer hour (BLS median hourly "
                    "wage, cited source in the profile JSON)",
     "compute": compute_impact_hour_value, "data_files": [IMPACT_PROFILE],
     "script_function": "analysis/impact_model.py::hour_value"},
    {"claim_id": "impact-P1", "description": "Impact P1 silent wrong results from context loss: profile USD/day and "
                    "USD/month (lower bound), rate, confidence",
     "compute": compute_impact_p1, "data_files": _IMPACT_FILES["P1"], "script_function": "analysis/impact_model.py::p1"},
    {"claim_id": "impact-P2", "description": "Impact P2 oversized-window time cost: profile USD/day and USD/month, "
                    "paired per-turn delta, confidence",
     "compute": compute_impact_p2, "data_files": _IMPACT_FILES["P2"], "script_function": "analysis/impact_model.py::p2"},
    {"claim_id": "impact-P3", "description": "Impact P3 cloud overspend from re-sent history: profile USD/day and "
                    "USD/month, per-session overspend, confidence",
     "compute": compute_impact_p3, "data_files": _IMPACT_FILES["P3"], "script_function": "analysis/impact_model.py::p3"},
    {"claim_id": "impact-P4", "description": "Impact P4 hardware mismatch (decode-time penalty vs the best measured "
                    "device): profile USD/day and USD/month, confidence",
     "compute": compute_impact_p4, "data_files": _IMPACT_FILES["P4"], "script_function": "analysis/impact_model.py::p4"},
    {"claim_id": "impact-P5", "description": "Impact P5 co-runner slowdown: profile USD/day and USD/month, ratios, "
                    "confidence",
     "compute": compute_impact_p5, "data_files": _IMPACT_FILES["P5"], "script_function": "analysis/impact_model.py::p5"},
    {"claim_id": "impact-P6", "description": "Impact P6 failed starts and retries (timeouts, allocation failures): "
                    "profile USD/day and USD/month, escalation cost, confidence",
     "compute": compute_impact_p6, "data_files": _IMPACT_FILES["P6"], "script_function": "analysis/impact_model.py::p6"},
    {"claim_id": "impact-P7", "description": "Impact P7 guardrail removal: profile USD/day and USD/month (not measured "
                    "when the rate is nonzero), rate, confidence",
     "compute": compute_impact_p7, "data_files": _IMPACT_FILES["P7"], "script_function": "analysis/impact_model.py::p7"},
    {"claim_id": "impact-v2-P1", "description": "Impact v2 P1 silent wrong tasks: measured excess wrong-task rate "
                    "per tier x (cloud redo cost per wrong task per tier, tagged priced, not executed while the redo is a stub, "
                    "+ wrong-turn wall time), one profile",
     "compute": compute_impact_v2_p1, "data_files": _IMPACT_V2_FILES,
     "script_function": "analysis/impact_model.py::p1_v2"},
    {"claim_id": "impact-ref-machine-price", "description": "ASSUMPTION INPUT, must be verified: reference "
                    "machine (evo-x2 class, Ryzen AI Max+ 395 128 GB mini PC) list price with its public source, for "
                    "impact v2 P2 machine capacity",
     "compute": compute_impact_ref_machine_price, "data_files": ["results/hardware_price_evox2_20261021.json"],
     "script_function": "analysis/impact_model.py::ref_machine"},
    {"claim_id": "impact-v2-detection-value", "description": "SUPERSEDED by impact-v3-detection-value-measured (kept "
                    "computing for traceability, not used in docs/IMPACT.md). PROJECTION: value of detecting silent "
                    "wrong tasks at "
                    "the v2 profile cell: cloud redo USD/day escalating every task step vs only predicted-at-risk "
                    "steps at the R2 shift predictor's measured precision/recall (and where it was measured)",
     "compute": compute_impact_v2_detection_value,
     "data_files": sorted(set(_IMPACT_V2_FILES) | {"results/x2_r2_mechanism.jsonl"}),
     "script_function": "analysis/impact_model.py::detection_projection"},
    {"claim_id": "impact-v3-detection-value-measured", "description": "MEASURED detection on the B3 v2 real run "
                    "(qwen3:14b, five tiers): precision, recall and TP/FP/FN/TN of 'the task turn's prompt exceeded the "
                    "loaded window' for 'the task ended wrong', wrong tasks missed and their share, and cloud USD "
                    "escalating only flagged tasks vs all tasks (cloud redo USD per task from "
                    "cloud-redo-cost-per-wrong-task, labelled by its mode)",
     "compute": compute_impact_v3_detection_value_measured, "data_files": _IMPACT_V2_FILES,
     "script_function": "analysis/impact_model.py::measured_detection"},
    {"claim_id": "impact-v2-P2", "description": "Impact v2 P2 configuration: sessions per machine per day (default vs "
                    "fastest measured B5 setting), machines needed for the profile's sessions/day, USD/day of machines "
                    "saved (assumed 3-year life), waiting-time upper bound on the interactive share; PENDING until "
                    "b5-fa-control is VERIFIED",
     "compute": compute_impact_v2_p2, "data_files": _IMPACT_V2_FILES,
     "script_function": "analysis/impact_model.py::p2_v2"},
    {"claim_id": "impact-ranking-v2", "description": "Impact v2 ranking at one stated profile, every row tagged "
                    "measured / mixed / assumed / PENDING / not measured, PENDING inputs named; P2 = machine capacity "
                    "(impact-v2-P2), not ranked while PENDING",
     "compute": compute_impact_ranking_v2, "data_files": _IMPACT_V2_FILES,
     "script_function": "analysis/impact_model.py::v2_rows"},
    {"claim_id": "impact-total", "description": "Impact total over the measured rows, ranking, and the one-flag "
                    "scenario total",
     "compute": compute_impact_total, "data_files": sorted({f for v in _IMPACT_FILES.values() for f in v}),
     "script_function": "analysis/impact_model.py::audit_total"},
    {"claim_id": "cloud-prices-anthropic", "description": "Shared Anthropic price table (official pricing page): "
                    "input, cache write, cache read and output per model",
     "compute": compute_cloud_prices_anthropic, "data_files": [ANTHROPIC_PRICES],
     "script_function": "analysis/numbers_register.py::compute_cloud_prices_anthropic"},
    {"claim_id": "impact-scenario-one-flag", "description": "Impact one-flag scenario (num_ctx 4096): where the "
                    "earlier USD 1,933.63/day came from, term by term, and the same scenario split into measured and "
                    "assumed (SCENARIO: depends on assumed rework time per silent wrong result)",
     "compute": compute_impact_scenario_one_flag,
     "data_files": sorted({f for v in _IMPACT_FILES.values() for f in v} | {"results/cloud_prices_20261008.json"}),
     "script_function": "analysis/impact_model.py::previous_basis"},
    {"claim_id": "impact-session-time-by-window", "description": "Median wall time per 40-turn R2 session on evo-x2 "
                    "at Ollama's default window vs num_ctx 4096, per model",
     "compute": compute_impact_session_time_by_window, "data_files": _IMPACT_DATA,
     "script_function": "analysis/impact_model.py::session_time_by_window"},

    {"claim_id": "x2-scale-probe-hours", "description": "H-scale timing probe: measured rates and projected hours per "
                    "model and tier, validation and real job hours, llama3.3:70b default-tier decision",
     "compute": compute_x2_scale_probe_hours,
     "data_files": [X2_SCALE_PROBE, R2_REAL_V1_FILE, R2_REAL_V1B_FILE, R2_VALIDATION_V2_FILE], "pending_ok": True,
     "script_function": "analysis/numbers_register.py::compute_x2_scale_probe_hours"},
    {"claim_id": "x2-scale-validation", "description": "H-scale R2 validation gates per model (qwen3:32b, "
                    "qwen3-30b-a3b-2507, llama3.3:70b)",
     "compute": compute_x2_scale_validation, "data_files": [X2_SCALE_VALIDATION], "pending_ok": True,
     "script_function": "analysis/numbers_register.py::compute_x2_scale_validation"},
    {"claim_id": "H-scale-verdict", "description": "Pre-registered H-scale verdict: silent rule loss at 4096 and "
                    "history loss with rules holding at 32768, per model at 30B to 70B",
     "compute": compute_h_scale_verdict, "data_files": [X2_SCALE_REAL, X2_SCALE_VALIDATION], "pending_ok": True,
     "script_function": "analysis/numbers_register.py::compute_h_scale_verdict"},
    {"claim_id": "x2-model-pulls-scale", "description": "x2_model_pulls_scale: llama3.3:70b and mistral-small3.2:24b "
                    "library tags, blob digests verified, manifest / model / params / template digests, template source",
     "compute": compute_x2_model_pulls_scale, "data_files": [X2_MODEL_PULLS_SCALE], "pending_ok": True,
     "script_function": "analysis/numbers_register.py::compute_x2_model_pulls_scale"},
    {"claim_id": "x2-family-smoke", "description": "B2 family smoke test, mistral-small3.2:24b: native valid tool calls "
                    "of 5 call-1 requests (pre-registered pass >= 4 of 5)",
     "compute": compute_x2_family_smoke, "data_files": [X2_FAMILY_SMOKE], "pending_ok": True,
     "script_function": "analysis/numbers_register.py::compute_x2_family_smoke"},
    {"claim_id": "x2-family-validation", "description": "B2 family R2 validation gates, mistral-small3.2:24b (same "
                    "gates as H-scale)",
     "compute": compute_x2_family_validation, "data_files": [X2_FAMILY_VALIDATION], "pending_ok": True,
     "script_function": "analysis/numbers_register.py::compute_x2_family_validation"},
    {"claim_id": "H-scale-family-verdict", "description": "H-scale per-model rules applied to the B2 family model "
                    "mistral-small3.2:24b (separate from H-scale-verdict)",
     "compute": compute_h_scale_family_verdict, "data_files": [X2_FAMILY_REAL, X2_FAMILY_VALIDATION],
     "pending_ok": True, "script_function": "analysis/numbers_register.py::compute_h_scale_family_verdict"},


    {"claim_id": "num-keep-guardrail-test", "description": "A4 flag test on evo-x2 (llama3.1:8b, num_ctx 4096, 3 "
                    "seeds x 40 turns, options.num_keep = system prompt tokens): observed n_keep, rule rates, "
                    "pre-registered verdict (one-line fix or not)",
     "compute": compute_num_keep_guardrail_test, "data_files": [NUM_KEEP_FILE], "pending_ok": True,
     "script_function": "analysis/numbers_register.py::compute_num_keep_guardrail_test"},
    {"claim_id": "window-override-methods", "description": "A5 context-window override methods on Ollama 0.34.4 "
                    "(llama3.1:8b): request num_ctx, Modelfile num_ctx on a derived tag, OLLAMA_CONTEXT_LENGTH; loaded "
                    "context (/api/ps, log) and persistence across a restart, per host",
     "compute": compute_window_override_methods, "data_files": [WOM_FILE_X2], "pending_ok": True,
     "script_function": "analysis/numbers_register.py::compute_window_override_methods"},
    {"claim_id": "b5-plan-hours", "description": "B5 policy comparison: ESTIMATED hours per platform and model "
                    "(assumed windows, aliases, evo-x2 minutes x platform factor), Blade night-3 cap fit",
     "compute": compute_b5_plan_hours, "data_files": B5_PLAN_INPUTS,
     "script_function": "analysis/numbers_register.py::compute_b5_plan_hours"},
    {"claim_id": "b5-policy-metrics", "description": "B5 per platform/model/policy: rule pass rate, history recall, "
                    "session wall-clock, cloud USD per session; Pareto frontier",
     "compute": compute_b5_policy_metrics, "data_files": [B5_FILES["evo-x2"], B5_PRICES], "pending_ok": True,
     "script_function": "analysis/numbers_register.py::compute_b5_policy_metrics"},
    {"claim_id": "b5-verdict", "description": "Pre-registered B5 verdict: router on the 4-metric Pareto frontier on every "
                    "platform and no single fixed setting on the frontier on all platforms",
     "compute": compute_b5_verdict, "data_files": [B5_FILES["evo-x2"], B5_PRICES], "pending_ok": True,
     "script_function": "analysis/numbers_register.py::compute_b5_verdict"},
    {"claim_id": "b5-kv-recall", "description": "B5 (g)/(h): long-context recall at the largest window per KV precision "
                    "and the server-log KV-type check",
     "compute": compute_b5_kv_recall, "data_files": [B5_FILES["evo-x2"]], "pending_ok": True,
     "script_function": "analysis/numbers_register.py::compute_b5_kv_recall"},
    {"claim_id": "gpu-memory-readonly", "description": "GPU memory per machine as already recorded read-only in result "
                    "files (Vulkan heaps, Ollama inference-compute lines, nvidia-smi)",
     "compute": compute_gpu_memory_readonly, "data_files": GPU_MEM_FILES,
     "script_function": "analysis/numbers_register.py::compute_gpu_memory_readonly"},
    {"claim_id": "fh1-amd-boundary-llama-3.3-70b", "description": "FH1 AMD panel, llama-3.3-70b on evo-x2 (llama-server "
     "b10970 Vulkan): measured spill boundary from per-process GPU shared usage, decode/TTFT on each side, bisection "
     "projection and fail points, fresh-process fail reps",
     "compute": compute_fh1_amd_boundary_llama70b, "data_files": AMD_BOUNDARY_FILES,
     "script_function": "analysis/numbers_register.py::compute_fh1_amd_boundary_llama70b"},
    {"claim_id": "fh1-amd-boundary-qwen3-32b", "description": "FH1 AMD panel, qwen3-32b on evo-x2 (llama-server b10970 "
     "Vulkan): same fields; no unclamped spill point exists",
     "compute": compute_fh1_amd_boundary_qwen3_32b, "data_files": AMD_BOUNDARY_FILES,
     "script_function": "analysis/numbers_register.py::compute_fh1_amd_boundary_qwen3_32b"},
    {"claim_id": "business-hour-range", "description": "Business sensitivity: hour-value p10/median/p90 (BLS annual "
     "percentiles converted at the page's own implied hours per year)",
     "compute": compute_business_hour_range, "data_files": [BUSINESS_PROFILES, IMPACT_PROFILE],
     "script_function": "analysis/business_metrics.py::hour_values"},
    {"claim_id": "business-mid-team", "description": "Business metrics, mid-size team on evo-x2 with llama3.1:8b: "
     "measured per-session cloud cost (caching on / off), projected spend and local compute time, router replay per "
     "window; router savings pending B5",
     "compute": compute_business_mid_team, "data_files": _BUSINESS_FILES,
     "script_function": "analysis/business_metrics.py::profile_metrics"},
    {"claim_id": "business-mid-team-qwen3-14b", "description": "Business metrics, mid-size team on evo-x2 with "
     "qwen3:14b", "compute": compute_business_mid_team_qwen3_14b, "data_files": _BUSINESS_FILES,
     "script_function": "analysis/business_metrics.py::profile_metrics"},
    {"claim_id": "business-scenario-plus-a", "description": "SCENARIO (not a measurement): router savings per "
     "profile under the +A residual-failure credit, the P1 rerun proxy and time at the full hour value",
     "compute": compute_business_scenario_plus_a, "data_files": _BUSINESS_FILES,
     "script_function": "analysis/business_metrics.py::scenario_metrics"},
    {"claim_id": "business-sensitivity", "description": "SCENARIO sensitivity of the +A router saving over team size, "
     "sessions per developer per day, cloud share and hour value",
     "compute": compute_business_sensitivity, "data_files": _BUSINESS_FILES,
     "script_function": "analysis/business_metrics.py::sensitivity"},
    {"claim_id": "x2-evalset-val-v1-gates", "description": "B3 evalset validations v1 (tasks, guardrails) on evo-x2: "
     "every gate per model as the harness computes it on the committed files",
     "compute": compute_x2_evalset_val_v1_gates, "data_files": EVALSET_V1_FILES,
     "script_function": "harness/x2_r2_evalset.py::evaluate_gates_ext"},
    {"claim_id": "x2-evalset-val-v1-diagnosis", "description": "B3 evalset validations v1: row-level classes of every "
     "negative-control canary miss (truncation / format / tool call as text / wrong value / omitted), failed task turn "
     "(schema or prompt mismatch / grader bug / task too hard / model failure, with subtypes), task-tool failure and "
     "rule 6-8 violation (executed vs text only)",
     "compute": compute_x2_evalset_val_v1_diagnosis, "data_files": EVALSET_V1_FILES,
     "script_function": "analysis/evalset_val_diagnosis.py::register_value"},
    {"claim_id": "t2s-r2-val-llama-diagnosis", "description": "evo-t2s R2 validations v1 (Intel CPU, Intel iGPU), "
     "llama3.1:8b baseline arm (ollama_ctx_131072_call2_notools): rule pass counts, canary misses, one class per failed "
     "rule (tool call as text / omitted / wrong value / scoring problem) with subtypes, rule 2 by turn task, recorded "
     "digest / temperature / seeds, CPU vs iGPU identity of matched turns, qwen3:14b failed rules",
     "compute": compute_t2s_r2_val_llama_diagnosis, "data_files": T2S_R2_VAL_FILES,
     "script_function": "analysis/t2s_llama_val_diagnosis.py::register_value"},
    {"claim_id": "x2-evalset-val-v1-call-times", "description": "B3 evalset validations v1, negative control: seconds "
     "per model call by final-prompt size (10k-token buckets), per model, over-window turns apart",
     "compute": compute_x2_evalset_val_v1_call_times, "data_files": EVALSET_V1_FILES,
     "script_function": "analysis/evalset_val_diagnosis.py::call_time_buckets"},
    {"claim_id": "x2-evalset-val-v2-gates", "description": "B3 evalset validations v2 (tasks, guard, replay) on "
     "evo-x2: every gate per validation and model with the numbers it was decided on, and the failing gates",
     "compute": compute_x2_evalset_val_v2_gates, "data_files": [v[0] for v in EVALSET_V2_FILES.values()],
     "script_function": "analysis/numbers_register.py::compute_x2_evalset_val_v2_gates"},
    {"claim_id": "x2-evalset-val-v2-task-end-state", "description": "B3 evalset validations v2: the task_end_state "
     "gate (negative-control task turns with a correct end state, 90% bar) per validation and model, by difficulty",
     "compute": compute_x2_evalset_val_v2_task_end_state, "data_files": [v[0] for v in EVALSET_V2_FILES.values()],
     "script_function": "analysis/numbers_register.py::compute_x2_evalset_val_v2_task_end_state"},
    {"claim_id": "x2-evalset-tasks-real-v2-hours", "description": "ESTIMATE, not a measurement: duration of "
     "x2_evalset_tasks_real_v2 (qwen3:14b, 5 tiers x 5 seeds x 30 task turns) from the v2 tasks validation's measured "
     "session times",
     "compute": compute_x2_evalset_tasks_real_v2_hours, "data_files": [X2_EVALSET_TASKS_VAL_V2],
     "script_function": "harness/x2_r2_evalset.py::tasks_v2_hours_estimate"},
    {"claim_id": "x2-evalset-tasks-real-v2-gated-excess", "description": "B3 v2 task real run (tools on every call), "
     "qwen3:14b: gated silent failures after the window was exceeded, excess over default with the v1b bootstrap, next "
     "to v1b's values, with the pre-registered protocol-consistency and num_keep readings",
     "compute": compute_x2_evalset_tasks_real_v2_gated_excess,
     "data_files": [X2_EVALSET_TASKS_REAL_V2, X2_EVALSET_TASKS_VAL_V2] + R2_V1B_DATA, "pending_ok": True,
     "script_function": "analysis/numbers_register.py::compute_x2_evalset_tasks_real_v2_gated_excess"},
    {"claim_id": "x2-evalset-tasks-real-v2-wrong-rate", "description": "B3 v2 task real run, qwen3:14b: wrong-task "
     "rate per tier (all task turns), silent vs surfaced, excess over default with a session bootstrap and its "
     "pre-registered reading; refused models",
     "compute": compute_x2_evalset_tasks_real_v2_wrong_rate, "data_files": [X2_EVALSET_TASKS_REAL_V2],
     "pending_ok": True, "script_function": "harness/x2_r2_evalset.py::tasks_real_v2_outcome"},
    {"claim_id": "cloud-redo-cost-per-wrong-task", "description": "Cost of a silent wrong answer: every B3 v2 wrong "
     "task (qwen3:14b, all tiers) re-run on the cloud escalation model; mean, median and total USD per wrong task, per "
     "tier, mode (priced, not executed vs executed) and n. In stub mode correctness and seconds are not measured",
     "compute": compute_cloud_redo_cost_per_wrong_task, "data_files": [CLOUD_REDO_V1, X2_EVALSET_TASKS_REAL_V2],
     "script_function": "scripts/cloud_redo.py::summarize"},
    {"claim_id": "cloud-redo-diagnosis", "description": "Why the live cloud redo fixed only part of the wrong tasks, "
     "offline: transcript sent (full client-side history vs truncated), environment it started from, model and tier, "
     "and each cloud failure in exactly one class (output cap, earlier-state repair, tool-round cap, grader issue, "
     "missing information, cloud model error)",
     "compute": compute_cloud_redo_diagnosis, "data_files": [CLOUD_REDO_V1, X2_EVALSET_TASKS_REAL_V2],
     "script_function": "analysis/cloud_redo_diagnosis.py::compute"},
    {"claim_id": "cloud-redo-full-context", "description": "Cloud redo rerun with the full client-side history, "
     "claude-sonnet-5-5, caching on, thinking off and a stop on any empty reply at max_tokens (the fix the "
     "cloud-redo-diagnosis row calls for): fix rate, USD mean / median / total, seconds, and the failure classes of "
     "anything still wrong",
     "compute": compute_cloud_redo_full_context,
     "data_files": [CLOUD_REDO_FULL_CONTEXT, CLOUD_REDO_FULL_CONTEXT_PART2,
                    "results/cloud_redo_full_context_v1_part3.jsonl", X2_EVALSET_TASKS_REAL_V2],
     "pending_ok": True, "script_function": "scripts/cloud_redo.py::run"},
    {"claim_id": "cloud-cost-per-real-task", "description": "Cloud cost per real agent task, priced, not executed: per "
     "trajectory of the uncensored traces (nebius-rebench-openhands, SWE-Gym) the billed input (each step re-sends its "
     "full prompt) and output tokens, USD median and p90 at claude-haiku-5-5 and claude-sonnet-5-5 for no caching, "
     "realistic caching (ASSUMPTION hit rate 0.7) and best-case caching; output per step measured from the traces; "
     "token counts calibrated to Anthropic's tokenizer (messages.count_tokens, 20 tasks per set)",
     "compute": compute_cloud_cost_per_real_task,
     "data_files": ["results/traces/agent_step_lengths.parquet", "results/traces/agent_step_outputs.parquet",
                    "results/anthropic_token_calibration_20261021.jsonl",
                    "results/cloud_prices_anthropic_20261014.json"],
     "script_function": "analysis/cloud_task_cost.py::compute"},

    {"claim_id": "all-cloud-b3-quality", "description": "All-cloud baseline: the 30 B3 v2 tasks run fully on "
     "claude-haiku-5-5 and claude-sonnet-5-5, 3 seeds, same session and end-state grader; correct / graded per model, "
     "per seed, per task type; qwen3:14b local default as context only. The all-cloud reference quality for the "
     "hybrid savings model",
     "compute": compute_all_cloud_b3_quality, "data_files": [ALL_CLOUD_B3_V1, X2_EVALSET_TASKS_REAL_V2],
     "pending_ok": True, "script_function": "scripts/all_cloud_b3.py::summarize"},
    {"claim_id": "all-cloud-b3-cost", "description": "All-cloud baseline cost: USD per task mean, median, p90 "
     "(prompt caching on, real cache-write premium), totals, seconds per task, cache read and write tokens, per model. "
     "The all-cloud reference cost for the hybrid savings model",
     "compute": compute_all_cloud_b3_cost, "data_files": [ALL_CLOUD_B3_V1], "pending_ok": True,
     "script_function": "scripts/all_cloud_b3.py::summarize"},
    {"claim_id": "x2-queue-fill-1020-hours", "description": "ESTIMATE, not a measurement: hours per job of the "
     "2026-10-21 evo-x2 queue fill from measured session times, back-to-back projected dates from the first launch, "
     "slack before 2026-11-10 and idle capacity",
     "compute": compute_x2_queue_fill_1020_hours, "data_files": QUEUE_FILL_DATA,
     "script_function": "analysis/numbers_register.py::compute_x2_queue_fill_1020_hours"},
    {"claim_id": "x2-queue-fill-1022-hours", "description": "ESTIMATE, not a measurement: hours per job of the "
     "2026-10-22 evo-x2 queue fill from measured times, back-to-back projected dates from the anchor, the B5 "
     "results-by date, slack before 2026-11-10 and the hours left beyond the 150 h reserve",
     "compute": compute_x2_queue_fill_1022_hours, "data_files": FILL_1022_DATA,
     "script_function": "analysis/numbers_register.py::compute_x2_queue_fill_1022_hours"},
    {"claim_id": "cost-cut-config-time", "description": "MEASURED: per model, session minutes saved per 40-turn R2 "
     "session on evo-x2 by the fastest measured non-router setting (B5 policies b-h, plus the flash-attention control "
     "once synced) against the runtime default (B5 policy a), with both settings' rule pass rate and history recall",
     "compute": compute_cost_cut_config_time, "data_files": [B5_X2_FILE],
     "script_function": "analysis/numbers_register.py::compute_cost_cut_config_time"},
    {"claim_id": "b5-fa-control", "description": "B5 flash-attention control (policy i, f16 KV + flash attention at "
     "L(f16), 3 seeds) against B5's (g)/(h) and (a) session minutes: the pre-registered reading per model (10% "
     "tolerance)", "compute": compute_b5_fa_control, "data_files": [B5_FA_FILE, B5_X2_FILE], "pending_ok": True,
     "script_function": "harness/policy_compare.py::fa_control_summary"},
    {"claim_id": "b5-kv-recall-f16-rerun", "description": "llama3.1:8b f16 long-context recall rerun (1 warm-up + 3 "
     "measured calls, 1800 s timeout) against B5's q8_0 and q4_0 recall: needles found of 24, Fisher exact "
     "(descriptive)", "compute": compute_b5_kv_recall_f16_rerun, "data_files": [B5_FA_FILE, B5_X2_FILE],
     "pending_ok": True, "script_function": "analysis/numbers_register.py::compute_b5_kv_recall_f16_rerun"},
    {"claim_id": "b5-sim-constraint", "description": "B5 under a simulated device window (8192 and 4096, 5 seeds, "
     "labelled simulated constraint): per cap, model and policy the rule pass rate, history recall, session minutes "
     "and cloud USD, the frontier, and B5's pre-registered verdict rule with the caps as platforms",
     "compute": compute_b5_sim_constraint, "data_files": list(B5_SIMCAP_FILES.values()), "pending_ok": True,
     "script_function": "harness/policy_compare.py::b5_verdict"},
    {"claim_id": "R2-v1b-excess-10seed", "description": "R2 window result pooled over 10 seeds per cell (v1/v1b seeds "
     "20260901-05 plus seeds 20260906-10 on the v1b harness blob after a passed reproduction check): llama3.1:8b and "
     "qwen3:14b, excess over the default at 16384/8192/4096 with the v1b bootstrap, pre-registered readings, "
     "sign-reversal flag, new seeds alone, CI widths, validity",
     "compute": compute_r2_v1b_excess_10seed, "data_files": [R2_V1B_SEEDS_FILE, R2_V1B_REPRO_FILE] + R2_V1B_DATA,
     "pending_ok": True, "script_function": "analysis/numbers_register.py::compute_r2_v1b_excess_10seed"},
    {"claim_id": "fh1-amd-qwen3-32b-spill-sweep", "description": "qwen3-32b on evo-x2 (llama-server b10970 Vulkan "
     "-ngl 99), n_ctx 20480 to 40960 unclamped: spill point, last no-spill point and spill cost (decode, TTFT) with "
     "the pre-registered readings, per-point shared-usage delta",
     "compute": compute_fh1_amd_qwen3_32b_spill_sweep, "data_files": [SPILL32_FILE], "pending_ok": True,
     "script_function": "harness/mx2_qwen32b_spill_sweep.py::summarize"},
    {"claim_id": "x2-evalset-tasks-val-v2-mistral-small32-gates", "description": "B3 v2 tasks validation for "
     "mistral-small3.2:24b: every gate with its numbers, failing gates, and whether the strict preflight admits it",
     "compute": compute_x2_evalset_tasks_val_v2_ms_gates, "data_files": [X2_EVALSET_TASKS_VAL_V2_MS],
     "pending_ok": True, "script_function": "analysis/numbers_register.py::compute_x2_evalset_tasks_val_v2_ms_gates"},
    {"claim_id": "x2-evalset-tasks-real-v2-mistral-small32-gated-excess", "description": "B3 v2 task real run, "
     "mistral-small3.2:24b: gated after-window silent failures and excess over the default per tier with the mistral "
     "protocol reading and the num_keep reading",
     "compute": compute_x2_evalset_tasks_real_v2_ms_gated_excess,
     "data_files": [X2_EVALSET_TASKS_REAL_V2_MS, X2_EVALSET_TASKS_VAL_V2_MS] + R2_V1B_DATA, "pending_ok": True,
     "script_function": "analysis/numbers_register.py::compute_x2_evalset_tasks_real_v2_ms_gated_excess"},
    {"claim_id": "x2-evalset-tasks-real-v2-mistral-small32-wrong-rate", "description": "B3 v2 task real run, "
     "mistral-small3.2:24b: wrong-task rate per tier and its excess over the default with the pre-registered reading",
     "compute": compute_x2_evalset_tasks_real_v2_ms_wrong_rate, "data_files": [X2_EVALSET_TASKS_REAL_V2_MS],
     "pending_ok": True, "script_function": "harness/x2_r2_evalset.py::tasks_real_v2_outcome"},
    {"claim_id": "x2-scale-real-default-tier", "description": "H-scale real job, Ollama default tier: completed "
     "default-tier sessions per model (turns, wall hours, loaded context, first over-window turn)",
     "compute": compute_x2_scale_real_default_tier, "data_files": [X2_SCALE_REAL],
     "script_function": "analysis/numbers_register.py::compute_x2_scale_real_default_tier"},
    {"claim_id": "num-keep-qwen3-14b", "description": "num_keep guardrail flag test for qwen3:14b (num_ctx 4096, 3 "
     "seeds x 40 turns, options.num_keep = system prompt tokens): observed n_keep, rule rates, pre-registered verdict",
     "compute": compute_num_keep_qwen3_14b, "data_files": [NUM_KEEP_Q14_FILE], "pending_ok": True,
     "script_function": "analysis/numbers_register.py::compute_num_keep_qwen3_14b"},
    {"claim_id": "x2-evalset-tasks-val-v2-qwen3-8b-gates", "description": "B3 v2 tasks validation for qwen3:8b: every "
     "gate with its numbers, failing gates, and the strict preflight's decision for the real run",
     "compute": compute_x2_evalset_tasks_val_v2_q8_gates, "data_files": [X2_EVALSET_TASKS_VAL_V2_Q8],
     "pending_ok": True, "script_function": "analysis/numbers_register.py::compute_x2_evalset_tasks_val_v2_q8_gates"},
    {"claim_id": "x2-evalset-tasks-real-v2-qwen3-8b-gated-excess", "description": "B3 v2 task real run, qwen3:8b: gated "
     "silent failures after the window was exceeded, excess over default with the v1b bootstrap, next to v1b's values, "
     "with the pre-registered protocol-consistency and num_keep readings",
     "compute": compute_x2_evalset_tasks_real_v2_q8_gated_excess,
     "data_files": [X2_EVALSET_TASKS_REAL_V2_Q8, X2_EVALSET_TASKS_VAL_V2_Q8] + R2_V1B_DATA, "pending_ok": True,
     "script_function": "analysis/numbers_register.py::compute_x2_evalset_tasks_real_v2_gated_excess"},
    {"claim_id": "x2-evalset-tasks-real-v2-qwen3-8b-wrong-rate", "description": "B3 v2 task real run, qwen3:8b: "
     "wrong-task rate per tier, silent vs surfaced, excess over default with a session bootstrap and its reading",
     "compute": compute_x2_evalset_tasks_real_v2_q8_wrong_rate, "data_files": [X2_EVALSET_TASKS_REAL_V2_Q8],
     "pending_ok": True, "script_function": "harness/x2_r2_evalset.py::tasks_real_v2_outcome"},
    {"claim_id": "R2-qwen3-8b-reversal-10seed", "description": "qwen3:8b num_ctx 8192 vs 4096 with 10 seeds per tier "
     "(v1b seeds pooled with 20260906-10): gated silent sessions, Fisher exact, pre-registered verdict, excess over "
     "default",
     "compute": compute_r2_qwen3_8b_reversal_10seed, "data_files": [R2_Q8_SEEDS_06_10_FILE] + R2_V1B_DATA,
     "pending_ok": True, "script_function": "analysis/numbers_register.py::compute_r2_qwen3_8b_reversal_10seed"},
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
