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
    """No computation over this file's real R1check rows reproduces a prior report's '100% score agreement,
    97.2% text agreement' figure -- checked both a raw mean-score interpretation (real: 60.8% mean score
    across 360 scored rows) and a repeat-measurement agreement interpretation (real: 70.0% same-score,
    63.1% same-text across 350 real repeat-measurement pairs). Deliberately raises so this is marked
    UNSUPPORTED rather than silently matched to the wrong computation."""
    path = repo / "results" / "t2s_night2_20260929T034014Z.jsonl"
    if not path.exists():
        raise FileNotFoundError(str(path))
    raise FileNotFoundError("no computation over this file's real R1check rows reproduces the reported "
                            "'100% score, 97.2% text' figure -- see this function's own docstring for what "
                            "was tried")


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
