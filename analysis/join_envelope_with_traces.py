"""Joins the envelope model against real agent-step prompt-length traces, if they exist yet.

results/traces/agent_step_lengths.parquet is being built by a separate, concurrent part of this session and may
not exist when this script is run. If it doesn't, this prints a clear one-line status and exits 0 (not an error
-- this is an expected, documented state, not a bug). If it does exist, loads it with pandas/pyarrow and computes,
per (machine, runtime_policy) cell, the headline table described in the task:
  - % of real agent steps that would run at full quality (prompt fits effective context)
  - % silently degraded (exceeds effective context, but failure-silence cell is SILENT_SPILL/SILENT_TRUNCATION)
  - % infeasible or over budget at two TTFT budgets (10s, 60s), using predict_latency's own regression

Usage: py -3.12 analysis/join_envelope_with_traces.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from envelope_model import predict_effective_context, predict_failure_silence, predict_latency  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACE_PATH = REPO_ROOT / "results" / "traces" / "agent_step_lengths.parquet"

MACHINES = ["evo-t2s", "evo-x2", "blade_rtx4070"]
RUNTIME_POLICIES = ["llama_ngl99", "llama_default_fit", "llama_fit_off", "ollama_default", "ollama_num_ctx_fixed"]
DEFAULT_MODEL = "qwen3-8b"  # representative model for the headline table; callers can extend per-model later
LATENCY_BUDGETS_S = (10, 60)


def build_headline_table(trace_path: Path = TRACE_PATH, model_id: str = DEFAULT_MODEL):
    if not trace_path.exists():
        print(f"SKIPPED: {trace_path} does not exist yet (it is being built by a separate part of this session). "
              f"The envelope model and its validation are otherwise complete and committed; this join step is "
              f"pending on that file's arrival. Re-run this script once it lands -- no other change is needed.")
        return None

    try:
        import pandas as pd
    except ImportError:
        print("SKIPPED: pandas is not installed in this environment; cannot read the trace parquet. "
              "Install pandas + pyarrow and re-run.")
        return None

    df = pd.read_parquet(trace_path)
    if "prompt_tokens" not in df.columns:
        raise ValueError(f"{trace_path} has no 'prompt_tokens' column; columns present: {list(df.columns)}")

    rows = []
    n_steps = len(df)
    for machine in MACHINES:
        for policy in RUNTIME_POLICIES:
            eff = predict_effective_context(machine, policy, model_id)
            eff_ctx = eff["effective_context"] or 0
            fail = predict_failure_silence(machine, policy)
            degraded_ok = fail["outcome"] in ("SILENT_SPILL", "SILENT_TRUNCATION")

            fits_full = (df["prompt_tokens"] <= eff_ctx).sum()
            over_ctx = n_steps - fits_full
            silently_degraded = over_ctx if degraded_ok else 0
            infeasible_hard = over_ctx if not degraded_ok else 0

            over_budget = {}
            for budget_s in LATENCY_BUDGETS_S:
                count = 0
                for pt in df["prompt_tokens"]:
                    lat = predict_latency(machine, policy, int(pt))
                    if lat["ttft_s"] > budget_s:
                        count += 1
                over_budget[budget_s] = round(100 * count / n_steps, 1) if n_steps else None

            rows.append({
                "machine": machine, "runtime_policy": policy, "model_id": model_id,
                "n_steps": n_steps,
                "pct_full_quality": round(100 * fits_full / n_steps, 1) if n_steps else None,
                "pct_silently_degraded": round(100 * silently_degraded / n_steps, 1) if n_steps else None,
                "pct_infeasible_hard": round(100 * infeasible_hard / n_steps, 1) if n_steps else None,
                "pct_over_10s_budget": over_budget[10],
                "pct_over_60s_budget": over_budget[60],
                "effective_context_flag": eff["flag"],
                "effective_context_measured": eff["measured"],
                "failure_silence_outcome": fail["outcome"],
                "failure_silence_measured": fail["outcome"] != "NOT_MEASURED",
                "latency_measured": machine == "evo-t2s",
            })

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    return out


if __name__ == "__main__":
    build_headline_table()
