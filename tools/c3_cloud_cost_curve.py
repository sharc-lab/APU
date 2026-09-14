"""C-3 cloud cost curve — analysis only (no measurement).

Sources (all sealed / recorded):
  X-2 cells: cb781dbf, 9fdedb46, afd1aa21, 0963168f
  cloud_multi_turn_report.json  ($5.41 / 20 entries, Sonnet 3/15)
  C-2 62395fdb  cold-start TTFT limit 10,000

Escalation rule (explicit): at the first turn where the local arm violates
TTFT > 10 s or decode < 6 tok/s or context > 10,000 tokens, escalate; once
escalated, the session stays on cloud for remaining turns.

Writes derived/bfcl_feasibility/c3_cloud_cost_curve/.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "derived" / "bfcl_feasibility" / "c3_cloud_cost_curve"
CLOUD_REPORT = ROOT / "derived" / "bfcl_feasibility" / "cloud_multi_turn_report.json"
X2_BASE = ROOT / "derived" / "bfcl_feasibility" / "x2_feasibility_table"

USD_PER_MTOK_IN = 3.0
USD_PER_MTOK_OUT = 15.0
TTFT_SLO_S = 10.0
DECODE_SLO_TOK_S = 6.0
COLD_START_CTX_LIMIT = 10_000  # C-2 62395fdb
C2_SESSION = "62395fdb-1899-415f-b708-6adc81a24dda"

CELLS = [
    {
        "session_id": "cb781dbf-3486-4fbc-a69a-34026f801abe",
        "arm": "cpu-p",
        "residency": "NON_RESIDENT",
        "report": "session_residency_A_NON_RESIDENT_report.json",
    },
    {
        "session_id": "9fdedb46-3318-4abc-a56f-50b7d23d25ca",
        "arm": "cpu-p",
        "residency": "RESIDENT",
        "report": "session_residency_A_RESIDENT_report.json",
    },
    {
        "session_id": "afd1aa21-d4b2-4491-81b4-b1b6f4fa681a",
        "arm": "gpu_only",
        "residency": "NON_RESIDENT",
        "report": "session_residency_gpu_only_NON_RESIDENT_report.json",
    },
    {
        "session_id": "0963168f-9144-4d21-99cf-5a77232dd477",
        "arm": "gpu_only",
        "residency": "RESIDENT",
        "report": "session_residency_gpu_only_RESIDENT_report.json",
    },
]

ESCALATION_RULE = {
    "name": "first_local_slo_or_context_breach_then_stay_cloud",
    "escalate_at_first_turn_where": [
        "ttft_s > 10",
        "decode_tok_s < 6",
        f"prompt_tokens (turn context) > {COLD_START_CTX_LIMIT}",
    ],
    "after_escalation": "remaining turns of the entry billed on cloud",
    "cold_start_context_limit_tokens": COLD_START_CTX_LIMIT,
    "cold_start_limit_source": {
        "run_id": C2_SESSION,
        "kind": "c2_ttft_bound_limit",
        "ttft_limits_all_arms": 10_000,
    },
    "slo": {"ttft_s_max": TTFT_SLO_S, "decode_tok_s_min": DECODE_SLO_TOK_S},
    "note": (
        "One policy among many; curve is conditional on this rule. "
        "Local energy estimated 30-45 W, not measured. Pricing is Sonnet "
        f"{USD_PER_MTOK_IN}/{USD_PER_MTOK_OUT} per Mtok in/out — one model, "
        "one price point."
    ),
}


def _utc() -> str:
    return datetime.now(UTC).isoformat()


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    k = (len(ys) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ys[f]
    return ys[f] * (c - k) + ys[c] * (k - f)


def _dist(xs: list[float]) -> dict[str, Any]:
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "min": min(xs),
        "p25": _pct(xs, 0.25),
        "median": statistics.median(xs),
        "p75": _pct(xs, 0.75),
        "p90": _pct(xs, 0.90),
        "max": max(xs),
        "mean": statistics.mean(xs),
    }


def _usd(prompt_tok: int, completion_tok: int) -> float:
    return (prompt_tok / 1e6) * USD_PER_MTOK_IN + (completion_tok / 1e6) * USD_PER_MTOK_OUT


def _load_x2(cell: dict[str, Any]) -> dict[str, Any]:
    sealed = X2_BASE / f"sealed_{cell['session_id']}"
    report = _read(sealed / "artifacts" / cell["report"])
    summary = _read(sealed / "summary.json")
    sealed_marker = _read(sealed / ".sealed")
    return {
        "cell": cell,
        "report": report,
        "summary": summary,
        "tree_sha256": sealed_marker.get("tree_sha256"),
        "entries": (report.get("gpu_probe") or {}).get("per_entry") or [],
    }


def _turn_breach(tm: dict[str, Any]) -> dict[str, Any] | None:
    reasons = []
    ttft = tm.get("ttft_s")
    dec = tm.get("decode_tok_s")
    ctx = tm.get("prompt_tokens")
    if ttft is not None and float(ttft) > TTFT_SLO_S:
        reasons.append(f"ttft_s={ttft}>{TTFT_SLO_S}")
    if dec is not None and float(dec) < DECODE_SLO_TOK_S:
        reasons.append(f"decode_tok_s={dec}<{DECODE_SLO_TOK_S}")
    if ctx is not None and int(ctx) > COLD_START_CTX_LIMIT:
        reasons.append(f"prompt_tokens={ctx}>{COLD_START_CTX_LIMIT}")
    if not reasons:
        return None
    return {"reasons": reasons, "ttft_s": ttft, "decode_tok_s": dec, "prompt_tokens": ctx}


def _escalation_turn(entry: dict[str, Any]) -> dict[str, Any]:
    """Return T = first violating turn index, or None if never escalates."""
    metrics = entry.get("turn_metrics") or []
    for i, tm in enumerate(metrics):
        breach = _turn_breach(tm)
        if breach is not None:
            return {
                "T": i,
                "n_turns": len(metrics),
                "breach": breach,
                "never_escalated": False,
            }
    return {
        "T": None,
        "n_turns": len(metrics),
        "breach": None,
        "never_escalated": True,
    }


def _cloud_by_entry(cloud: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out = {}
    for e in cloud["cloud_arm"]["per_entry_full"]:
        out[e["id"]] = e
    return out


def _bill_from_T(cloud_entry: dict[str, Any] | None, T: int | None) -> dict[str, Any]:
    """Bill cloud calls with turn >= T. T=None => $0 (never escalated). T=0 => full."""
    if cloud_entry is None:
        return {
            "usd": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "n_calls": 0,
            "missing_cloud_entry": True,
        }
    if T is None:
        return {
            "usd": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "n_calls": 0,
            "missing_cloud_entry": False,
            "note": "never escalated; local completes",
        }
    calls = [c for c in cloud_entry.get("calls") or [] if int(c["turn"]) >= int(T)]
    prompt = sum(int(c.get("prompt_tokens") or 0) for c in calls)
    comp = sum(int(c.get("completion_tokens") or 0) for c in calls)
    usd = sum(float(c.get("usd") or 0.0) for c in calls)
    # Prefer per-call usd when present; else recompute from tokens.
    if abs(usd - _usd(prompt, comp)) > 1e-6 and prompt + comp > 0:
        usd_re = _usd(prompt, comp)
    else:
        usd_re = usd if calls else 0.0
        if not calls:
            usd_re = 0.0
        elif usd == 0.0 and (prompt or comp):
            usd_re = _usd(prompt, comp)
    return {
        "usd": float(usd_re if usd == 0.0 and (prompt or comp) else usd),
        "prompt_tokens": prompt,
        "completion_tokens": comp,
        "n_calls": len(calls),
        "missing_cloud_entry": False,
        "T": T,
    }


def _growth_stats(x2_loads: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-turn context length and wall time distributions across configs."""
    per_config = []
    all_deltas: list[float] = []
    all_turn_ctx: list[float] = []
    all_walls: list[float] = []
    first_step_series_deltas: list[float] = []

    for loaded in x2_loads:
        cell = loaded["cell"]
        turn_ctx: list[float] = []
        turn_wall: list[float] = []
        turn_ttft: list[float] = []
        deltas: list[float] = []
        for entry in loaded["entries"]:
            series = entry.get("prompt_tokens_first_step_per_turn") or []
            if len(series) >= 2:
                for a, b in zip(series, series[1:], strict=False):
                    d = float(b) - float(a)
                    deltas.append(d)
                    first_step_series_deltas.append(d)
            for tm in entry.get("turn_metrics") or []:
                if tm.get("prompt_tokens") is not None:
                    turn_ctx.append(float(tm["prompt_tokens"]))
                    all_turn_ctx.append(float(tm["prompt_tokens"]))
                if tm.get("last_wall_s") is not None:
                    turn_wall.append(float(tm["last_wall_s"]))
                    all_walls.append(float(tm["last_wall_s"]))
                if tm.get("ttft_s") is not None:
                    turn_ttft.append(float(tm["ttft_s"]))
            # Also use delta_tokens_vs_prev_turn when present
            for d in entry.get("delta_tokens_vs_prev_turn") or []:
                if d is not None:
                    all_deltas.append(float(d))

        span_note = None
        if turn_ctx:
            span_note = {
                "min_turn_context": min(turn_ctx),
                "max_turn_context": max(turn_ctx),
                "span": max(turn_ctx) - min(turn_ctx),
            }
        per_config.append(
            {
                "arm": cell["arm"],
                "residency": cell["residency"],
                "session_id": cell["session_id"],
                "turn_context_tokens": _dist(turn_ctx),
                "turn_wall_s": _dist(turn_wall),
                "turn_ttft_s": _dist(turn_ttft),
                "per_turn_growth_tokens_first_step_series": _dist(deltas),
                "context_span": span_note,
            }
        )

    # Estimate check: ~1300 from (7743-2598) is a full-session span / implied turns,
    # not a per-turn growth. Report both.
    estimate_1300 = {
        "claimed_approx_tokens_per_turn": 1300,
        "source_span_interpretation": "7743 - 2598 = 5145 over a multi-turn/multi-step span",
        "span_tokens_if_2598_to_7743": 5145,
        "measured_per_turn_growth_first_step_deltas": _dist(first_step_series_deltas),
        "measured_delta_tokens_vs_prev_turn": _dist(all_deltas),
        "verdict": None,
    }
    med = estimate_1300["measured_per_turn_growth_first_step_deltas"].get("median")
    if med is not None:
        estimate_1300["verdict"] = (
            f"median first-step-per-turn growth is {med:.1f} tokens, "
            f"not ~1300; 1300 overstated per-turn growth by treating the "
            f"2598→7743 session span as if it were a single turn delta."
        )

    return {
        "per_configuration": per_config,
        "pooled_turn_context_tokens": _dist(all_turn_ctx),
        "pooled_turn_wall_s": _dist(all_walls),
        "pooled_per_turn_growth_first_step": _dist(first_step_series_deltas),
        "estimate_1300_check": estimate_1300,
        "cloud_report_generation_span": {
            "note": (
                "7743 max appears on per-generation full_prompt in the GPU "
                "multi-turn probe (incl. mid-turn tool steps), not as X-2 "
                "turn-start context."
            ),
            "min_cited": 2598,
            "max_cited": 7743,
        },
    }


def _fit_cost_vs_T(points: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit log(cost) ~ a + b*log(T_eff) for cost>0; report exponent b.

    For the compounding claim, prefer the forced-T curve (cost of billing
    turns >= T on cloud for every entry). Config-level points are secondary.
    T_eff = max_turns - T (remaining cloud turns) when testing compounding
    with remaining horizon; also fit cost vs T directly.
    """
    xs_log = []
    ys_log = []
    xs_lin = []
    ys_lin = []
    xs_remain_log = []
    ys_remain_log = []
    for p in points:
        t = p.get("T")
        usd = p.get("usd_per_session_mean")
        if t is None or usd is None or float(usd) <= 0:
            continue
        xs_lin.append(float(t))
        ys_lin.append(float(usd))
        # cost vs remaining cloud horizon (max_T_index+1 - T)
        remain = p.get("remaining_turns_mean")
        if remain is not None and float(remain) > 0:
            xs_remain_log.append(math.log(float(remain)))
            ys_remain_log.append(math.log(float(usd)))
        t_eff = float(t) + 1.0
        xs_log.append(math.log(t_eff))
        ys_log.append(math.log(float(usd)))

    def _ols(x: list[float], y: list[float]) -> dict[str, Any]:
        n = len(x)
        if n < 2:
            return {"n": n, "ok": False}
        mx = statistics.mean(x)
        my = statistics.mean(y)
        num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y, strict=False))
        den = sum((xi - mx) ** 2 for xi in x)
        if den == 0:
            return {"n": n, "ok": False, "reason": "zero variance in x"}
        b = num / den
        a = my - b * mx
        yhat = [a + b * xi for xi in x]
        ss_res = sum((yi - yh) ** 2 for yi, yh in zip(y, yhat, strict=False))
        ss_tot = sum((yi - my) ** 2 for yi in y)
        r2 = 1.0 - ss_res / ss_tot if ss_tot else None
        return {"n": n, "ok": True, "intercept": a, "slope": b, "r2": r2}

    log_fit = _ols(xs_log, ys_log)
    lin_fit = _ols(xs_lin, ys_lin)
    remain_fit = _ols(xs_remain_log, ys_remain_log)

    # Primary exponent for compounding: cost vs remaining cloud turns.
    primary_exp = remain_fit.get("slope") if remain_fit.get("ok") else None
    claim = {
        "predicted": "quadratic (exponent 2) because cloud context compounds per step",
        "fitted_exponent_cost_vs_remaining_turns": primary_exp,
        "fitted_loglog_exponent_cost_vs_Tplus1": log_fit.get("slope")
        if log_fit.get("ok")
        else None,
        "linear_slope_usd_per_unit_T": lin_fit.get("slope") if lin_fit.get("ok") else None,
        "remaining_turns_r2": remain_fit.get("r2"),
        "loglog_Tplus1_r2": log_fit.get("r2"),
        "linear_r2": lin_fit.get("r2"),
        "verdict": None,
        "remain_fit": remain_fit,
        "log_fit": log_fit,
        "linear_fit": lin_fit,
        "points_used": points,
    }
    exp = primary_exp
    if exp is None:
        claim["verdict"] = "insufficient positive-cost points to fit exponent"
    elif abs(exp - 2.0) <= 0.35:
        claim["verdict"] = (
            f"consistent with quadratic in remaining cloud turns " f"(fitted exponent {exp:.3f})"
        )
    elif abs(exp - 1.0) <= 0.35:
        claim["verdict"] = (
            f"closer to linear in remaining cloud turns "
            f"(fitted exponent {exp:.3f}); quadratic claim not supported"
        )
    else:
        claim["verdict"] = (
            f"fitted exponent vs remaining turns = {exp:.3f} "
            "(neither clearly 1 nor 2); report the number, do not force quadratic"
        )
    return claim


def _forced_T_curve(cloud_entries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean session $ if every entry escalates at forced turn T."""
    max_turn = 0
    for ce in cloud_entries.values():
        for c in ce.get("calls") or []:
            max_turn = max(max_turn, int(c["turn"]))
    points = []
    for T in range(0, max_turn + 1):
        usds = []
        remains = []
        for ce in cloud_entries.values():
            bill = _bill_from_T(ce, T)
            usds.append(float(bill["usd"] or 0))
            turns = {int(c["turn"]) for c in ce.get("calls") or []}
            n_turns = (max(turns) + 1) if turns else 0
            remains.append(max(0, n_turns - T))
        mean_usd = statistics.mean(usds) if usds else 0.0
        points.append(
            {
                "T": T,
                "usd_per_session_mean": mean_usd,
                "usd_total": sum(usds),
                "remaining_turns_mean": statistics.mean(remains) if remains else 0.0,
                "label": f"forced_T={T}",
            }
        )
    return points


def analyze() -> dict[str, Any]:
    cloud = _read(CLOUD_REPORT)
    cloud_entries = _cloud_by_entry(cloud)
    x2_loads = [_load_x2(c) for c in CELLS]
    growth = _growth_stats(x2_loads)

    # Call pattern from cloud (observed)
    calls_per_turn = []
    for e in cloud["cloud_arm"]["per_entry_full"]:
        by_t: Counter[int] = Counter(int(c["turn"]) for c in e["calls"])
        calls_per_turn.extend(by_t.values())

    configs_out = []
    sealed_spend = float(cloud["spend"]["usd"])
    n_completed_cloud = int(cloud["cloud_arm"]["trajectory"]["correct"])

    # Cloud-only synthetic (T=0 always)
    cloud_only_rows = []
    for eid, ce in cloud_entries.items():
        bill = _bill_from_T(ce, 0)
        cloud_only_rows.append(
            {
                "id": eid,
                "T": 0,
                "usd": bill["usd"],
                "trajectory_valid": bool((ce.get("score") or {}).get("valid")),
            }
        )
    usd_sum_cloud = sum(float(r["usd"] or 0) for r in cloud_only_rows)
    cloud_only = {
        "label": "cloud_only",
        "T_policy": "T=0 every entry",
        "n_entries": len(cloud_only_rows),
        "usd_total": usd_sum_cloud,
        "usd_per_session_mean": usd_sum_cloud / len(cloud_only_rows),
        "n_completed_tasks": n_completed_cloud,
        # Amortized: total spend / completed count (matches 5.41/13 ≈ 0.42).
        "usd_per_completed_task": sealed_spend / n_completed_cloud,
        "sanity_targets": {
            "usd_per_entry_approx": 0.27,
            "usd_per_completed_approx": 0.42,
            "completed_count": 13,
        },
        "sanity_observed": {
            "usd_per_entry": usd_sum_cloud / len(cloud_only_rows),
            "usd_per_completed_task_amortized": sealed_spend / n_completed_cloud,
            "usd_mean_among_completed_entries_only": (
                sum(float(r["usd"] or 0) for r in cloud_only_rows if r["trajectory_valid"])
                / n_completed_cloud
            ),
            "n_completed": n_completed_cloud,
            "sealed_spend_usd": sealed_spend,
            "note": (
                "Amortized $/completed = sealed_spend/13. Mean among completed "
                "entries only is lower because failed entries still spend."
            ),
        },
    }

    for loaded in x2_loads:
        cell = loaded["cell"]
        rows = []
        for entry in loaded["entries"]:
            eid = entry["id"]
            esc = _escalation_turn(entry)
            bill = _bill_from_T(cloud_entries.get(eid), esc["T"])
            ce = cloud_entries.get(eid) or {}
            traj = bool((ce.get("score") or {}).get("valid"))
            local_score = entry.get("score") or {}
            rows.append(
                {
                    "id": eid,
                    "T": esc["T"],
                    "never_escalated": esc["never_escalated"],
                    "breach": esc["breach"],
                    "n_local_turns": esc["n_turns"],
                    "usd": bill["usd"],
                    "cloud_prompt_tokens_billed": bill["prompt_tokens"],
                    "cloud_completion_tokens_billed": bill["completion_tokens"],
                    "n_cloud_calls_billed": bill["n_calls"],
                    "cloud_trajectory_valid": traj,
                    "local_turn_contexts": [
                        tm.get("prompt_tokens") for tm in (entry.get("turn_metrics") or [])
                    ],
                    "local_turn_ttft_s": [
                        tm.get("ttft_s") for tm in (entry.get("turn_metrics") or [])
                    ],
                    "local_turn_decode_tok_s": [
                        tm.get("decode_tok_s") for tm in (entry.get("turn_metrics") or [])
                    ],
                    "local_turn_wall_s": [
                        tm.get("last_wall_s") for tm in (entry.get("turn_metrics") or [])
                    ],
                    "local_score_valid": local_score.get("valid"),
                }
            )

        Ts = [r["T"] for r in rows if r["T"] is not None]
        usds = [float(r["usd"] or 0) for r in rows]
        n_completed = sum(1 for r in rows if r["cloud_trajectory_valid"])
        label = f"{cell['arm']}_{cell['residency']}"
        usd_total = sum(usds)
        cfg = {
            "label": label,
            "arm": cell["arm"],
            "residency": cell["residency"],
            "session_id": cell["session_id"],
            "tree_sha256": loaded["tree_sha256"],
            "n_entries": len(rows),
            "n_escalated": sum(1 for r in rows if not r["never_escalated"]),
            "n_never_escalated": sum(1 for r in rows if r["never_escalated"]),
            "T_distribution": _dist([float(t) for t in Ts]) if Ts else {"n": 0},
            "T_mean": statistics.mean(Ts) if Ts else None,
            "T_mean_including_never_as_n_turns": statistics.mean(
                [float(r["T"]) if r["T"] is not None else float(r["n_local_turns"]) for r in rows]
            ),
            "usd_total": usd_total,
            "usd_per_session_mean": statistics.mean(usds) if usds else None,
            "n_completed_tasks_cloud_traj": n_completed,
            "usd_per_completed_task": (usd_total / n_completed if n_completed else None),
            "fraction_turns_slo_ok_sealed": loaded["summary"].get("fraction_turns_slo_ok"),
            "per_entry": rows,
        }
        configs_out.append(cfg)

    forced = _forced_T_curve(cloud_entries)
    quadratic = _fit_cost_vs_T(forced)

    completed_costs = [
        (c["label"], c["usd_per_completed_task"])
        for c in configs_out
        if c["usd_per_completed_task"] is not None
    ]
    completed_costs.sort(key=lambda x: x[1])
    spread = None
    if len(completed_costs) >= 2:
        cheap, expensive = completed_costs[0], completed_costs[-1]
        spread = {
            "cheapest": {"label": cheap[0], "usd_per_completed_task": cheap[1]},
            "most_expensive": {
                "label": expensive[0],
                "usd_per_completed_task": expensive[1],
            },
            "spread_usd_per_completed_task": expensive[1] - cheap[1],
            "ratio_expensive_over_cheap": (expensive[1] / cheap[1] if cheap[1] > 0 else None),
            "hardware_model_note": (
                "Identical Platform A hardware and Qwen3-4B-int4-ov local model; "
                "cloud bill is Sonnet at 3/15. Spread is configuration-only."
            ),
        }

    artifact = {
        "kind": "c3_cloud_cost_curve",
        "generated_utc": _utc(),
        "sources": {
            "x2_cells": [
                {
                    "session_id": c["session_id"],
                    "arm": c["arm"],
                    "residency": c["residency"],
                    "seal": f"derived/bfcl_feasibility/x2_feasibility_table/sealed_{c['session_id']}",
                }
                for c in CELLS
            ],
            "cloud_report": str(CLOUD_REPORT.relative_to(ROOT).as_posix()),
            "cloud_spend_usd": cloud["spend"]["usd"],
            "c2_ttft_limit": {
                "session_id": C2_SESSION,
                "limit_tokens": COLD_START_CTX_LIMIT,
            },
        },
        "escalation_rule": ESCALATION_RULE,
        "pricing": {
            "model": cloud.get("model"),
            "usd_per_mtok_in": USD_PER_MTOK_IN,
            "usd_per_mtok_out": USD_PER_MTOK_OUT,
        },
        "limits": {
            "one_cloud_model_one_price": True,
            "curve_is_sonnet_shaped": True,
            "local_energy_w_estimate": [30, 45],
            "local_energy_measured": False,
            "escalation_rule_is_one_policy_among_many": True,
            "curve_conditional_on_escalation_rule": True,
        },
        "call_pattern_observed": {
            "calls_per_turn_distribution": dict(sorted(Counter(calls_per_turn).items())),
            "calls_per_turn_summary": _dist([float(x) for x in calls_per_turn]),
            "note": (
                "Billing uses sealed cloud per-call usd for turns >= T on the "
                "same entry ids (reproduces cloud-only when T=0). Forced-T curve "
                "varies T to test compounding independent of local SLO."
            ),
        },
        "growth": growth,
        "cloud_only_sanity": cloud_only,
        "configurations": configs_out,
        "forced_T_cost_curve": forced,
        "quadratic_claim_test": quadratic,
        "spread": spread,
    }
    return artifact


def _plot(artifact: dict[str, Any], out_dir: Path) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return f"plot skipped: {exc}"

    forced = artifact.get("forced_T_cost_curve") or []
    xs = [p["T"] for p in forced]
    ys = [p["usd_per_session_mean"] for p in forced]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(xs, ys, "o-", label="forced escalation at T (cloud bills turns>=T)")
    remain = artifact["quadratic_claim_test"].get("remain_fit") or {}
    if remain.get("ok"):
        # reconstruct from remaining-turns fit for annotation only
        ax.set_title(
            "C-3: session cloud cost vs forced escalation turn T\n"
            f"exponent vs remaining turns = {remain.get('slope'):.3f} "
            f"(R2={remain.get('r2'):.3f})"
        )
    else:
        ax.set_title("C-3: session cloud cost vs forced escalation turn T")
    # config markers
    for c in artifact.get("configurations") or []:
        t = c.get("T_mean")
        u = c.get("usd_per_session_mean")
        if t is None or u is None:
            continue
        ax.scatter([t], [u], s=80, marker="x", label=c["label"])
    ax.set_xlabel("escalation turn T")
    ax.set_ylabel("mean cloud USD per session")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    path = out_dir / "cost_vs_T.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path.relative_to(ROOT).as_posix())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    artifact = analyze()
    # ASCII-only growth verdict for Windows consoles
    ev = artifact["growth"]["estimate_1300_check"]
    if ev.get("verdict"):
        ev["verdict"] = ev["verdict"].replace("\u2192", "->")
    plot_path = _plot(artifact, OUT)
    artifact["plot"] = plot_path
    out_json = OUT / "c3_cloud_cost_curve.json"
    out_json.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = []
    lines.append("# C-3 cloud cost curve")
    lines.append("")
    lines.append(f"Generated: {artifact['generated_utc']}")
    lines.append("")
    lines.append("## Escalation rule")
    lines.append("")
    lines.append(
        "Escalate at the first turn where local TTFT > 10 s, decode < 6 tok/s, "
        "or context > 10,000 (C-2). Once escalated, stay on cloud."
    )
    lines.append("")
    lines.append("## Growth (not a mean)")
    g = artifact["growth"]["estimate_1300_check"]
    lines.append(g.get("verdict") or "")
    pg = artifact["growth"]["pooled_per_turn_growth_first_step"]
    lines.append(
        f"Pooled first-step-per-turn delta tokens: n={pg.get('n')} "
        f"min={pg.get('min')} median={pg.get('median')} "
        f"p90={pg.get('p90')} max={pg.get('max')}"
    )
    lines.append("")
    lines.append("## Cloud-only sanity")
    s = artifact["cloud_only_sanity"]["sanity_observed"]
    lines.append(
        f"usd/entry={s['usd_per_entry']:.4f} (target ~0.27); "
        f"usd/completed amortized={s['usd_per_completed_task_amortized']:.4f} "
        f"(target ~0.42); completed={s['n_completed']}/20"
    )
    lines.append("")
    lines.append("## Per configuration (SLO-driven T)")
    lines.append("")
    lines.append("| config | n_esc | T median | $/session | $/completed |")
    lines.append("|---|---:|---:|---:|---:|")
    for c in artifact["configurations"]:
        td = c.get("T_distribution") or {}
        upc = c["usd_per_completed_task"]
        upc_s = f"{upc:.4f}" if upc is not None else "n/a"
        lines.append(
            f"| {c['label']} | {c['n_escalated']}/{c['n_entries']} | "
            f"{td.get('median')} | {c['usd_per_session_mean']:.4f} | {upc_s} |"
        )
    lines.append("")
    lines.append("## Forced-T cost curve (quadratic test)")
    lines.append("")
    lines.append("| T | mean $/session | mean remaining turns |")
    lines.append("|---:|---:|---:|")
    for p in artifact["forced_T_cost_curve"]:
        lines.append(
            f"| {p['T']} | {p['usd_per_session_mean']:.4f} | " f"{p['remaining_turns_mean']:.2f} |"
        )
    q = artifact["quadratic_claim_test"]
    lines.append("")
    lines.append("## Quadratic claim")
    lines.append("")
    lines.append(q.get("verdict") or "")
    lines.append(
        f"exponent vs remaining turns={q.get('fitted_exponent_cost_vs_remaining_turns')}; "
        f"R2={q.get('remaining_turns_r2')}; "
        f"linear cost-vs-T slope={q.get('linear_slope_usd_per_unit_T')}"
    )
    lines.append("")
    sp = artifact.get("spread") or {}
    lines.append("## Spread ($/completed task, amortized)")
    lines.append("")
    if sp:
        lines.append(
            f"Cheapest {sp['cheapest']['label']}: "
            f"${sp['cheapest']['usd_per_completed_task']:.4f}; "
            f"most expensive {sp['most_expensive']['label']}: "
            f"${sp['most_expensive']['usd_per_completed_task']:.4f}; "
            f"spread ${sp['spread_usd_per_completed_task']:.4f}"
        )
    lines.append("")
    lines.append("## Limits")
    lines.append("")
    for k, v in artifact["limits"].items():
        lines.append(f"- {k}: {v}")
    if plot_path:
        lines.append("")
        lines.append(f"Plot: `{plot_path}`")
    (OUT / "C3_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "out": str(out_json), "plot": plot_path}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
