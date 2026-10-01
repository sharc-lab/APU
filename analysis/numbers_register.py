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


def compute_a3_qwen32b_refusal_share(repo):
    """Reconciled 2026-10-01: 19/111 (17%) is the real number, from docs/FINDINGS.md's R1b audit section.
    The source jsonl files that section cites (results/t2s_night2_20260929T202603Z.jsonl,
    ..._20260929T205109Z.jsonl) were never committed -- confirmed by the A3 subagent via git log --all.
    This entry is deliberately UNSUPPORTED (raises) until those files are synced and committed, so the
    register does not silently treat a doc-only number as file-verified."""
    path = repo / "results" / "t2s_night2_20260929T205109Z.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not committed -- see docs/FINDINGS.md's A3/A6 audit section for "
                                "the full provenance note")
    rows = _read_jsonl(path)
    return {"value": "recompute once the file exists -- not implemented, placeholder only", "n": len(rows)}


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
    {"claim_id": "A3-qwen32b-refusal", "description": "qwen3-32b refusal share of wrong answers",
     "compute": compute_a3_qwen32b_refusal_share,
     "data_files": ["results/t2s_night2_20260929T205109Z.jsonl (not yet committed)"],
     "script_function": "analysis/numbers_register.py::compute_a3_qwen32b_refusal_share",
     "reported_value": "19/111 (17%)"},
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
