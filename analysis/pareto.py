"""Pareto analysis: quality vs cloud spend vs hardware BOM cost.

Three-axis DSE:
  x  — cloud API tokens spent (mean across seeds)
  y  — quality score          (mean across tasks and seeds)
  z  — hardware BOM cost USD  (from configs/hardware/*.yaml)

2-D projections keep the third axis as marker color/size so all information
is visible in a single figure per hardware config.

Entry-point helpers
-------------------
generate_pareto_artifacts(input_path)        — quality × cloud frontier
generate_bom_artifacts(input_path, hw_dir)   — full three-axis DSE
cheapest_policy_per_floor(results, hw_name)  — policy selection table
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

try:
    import yaml as _yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

REPO_ROOT = Path(__file__).parent.parent
INPUT_PATH = REPO_ROOT / "results" / "pareto_results.json"
HW_DIR = REPO_ROOT / "configs" / "hardware"
OUTPUT_PLOT = REPO_ROOT / "reports" / "pareto_frontier.png"
OUTPUT_BOM_PLOT = REPO_ROOT / "reports" / "pareto_bom_frontier.png"
OUTPUT_TABLE = REPO_ROOT / "reports" / "pareto_task_breakdown.md"
OUTPUT_BOM_TABLE = REPO_ROOT / "reports" / "pareto_bom_breakdown.md"

QUALITY_FLOORS = [6.0, 7.0, 8.0, 9.0]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    return (sum((v - mu) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def _load_hw_configs(hw_dir: Path = HW_DIR) -> dict[str, dict[str, Any]]:
    """Load all *.yaml files from hw_dir.  Returns {} if yaml is unavailable."""
    if not _HAS_YAML or not hw_dir.exists():
        return {}
    configs: dict[str, dict[str, Any]] = {}
    for path in sorted(hw_dir.glob("*.yaml")):
        try:
            cfg = _yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            name = cfg.get("name", path.stem)
            configs[name] = cfg
        except Exception:
            pass
    return configs


def _is_pareto_optimal(points: list[tuple[float, float]]) -> list[bool]:
    """Return a mask of Pareto-optimal points (maximise y, minimise x)."""
    n = len(points)
    dominated = [False] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            xi, yi = points[i]
            xj, yj = points[j]
            if xj <= xi and yj >= yi and (xj < xi or yj > yi):
                dominated[i] = True
                break
    return [not d for d in dominated]


# ---------------------------------------------------------------------------
# Original two-axis frontier (quality × cloud spend)
# ---------------------------------------------------------------------------

def generate_pareto_artifacts(input_path: Path = INPUT_PATH) -> dict:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    rows = payload.get("results", [])

    grouped = defaultdict(list)
    for row in rows:
        key = (row["policy"], row["budget_level"], row["seed"])
        grouped[key].append(row)

    frontier: dict[str, dict] = defaultdict(lambda: {"x": [], "y": [], "yerr": []})
    by_policy_budget: dict[tuple, dict] = defaultdict(lambda: {"quality": [], "cloud": []})

    for (policy, budget_level, seed), recs in grouped.items():
        mean_quality = _mean([float(r["quality"]) for r in recs])
        cloud_spent = sum(int(r["cloud_tokens_spent"]) for r in recs)
        by_policy_budget[(policy, budget_level)]["quality"].append(mean_quality)
        by_policy_budget[(policy, budget_level)]["cloud"].append(float(cloud_spent))

    for (policy, budget_level), stats in sorted(
        by_policy_budget.items(), key=lambda x: (x[0][0], x[0][1])
    ):
        frontier[policy]["x"].append(_mean(stats["cloud"]))
        frontier[policy]["y"].append(_mean(stats["quality"]))
        frontier[policy]["yerr"].append(_stdev(stats["quality"]))

    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for policy, curve in sorted(frontier.items()):
        plt.errorbar(
            curve["x"], curve["y"], yerr=curve["yerr"],
            marker="o", capsize=3, label=policy,
        )

    plt.xlabel("Cloud Tokens Spent (mean across seeds)")
    plt.ylabel("Quality Score (mean across tasks and seeds)")
    plt.title("Pareto Frontier: Quality vs Cloud Token Spend")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT)
    plt.close()

    task_lines = [
        "# Per-Task Breakdown",
        "",
        "| policy | budget_level | task_id | mean_quality | mean_cloud_tokens |",
        "|---|---:|---|---:|---:|",
    ]

    task_group: dict[tuple, dict] = defaultdict(lambda: {"quality": [], "cloud": []})
    for row in rows:
        key = (row["policy"], row["budget_level"], row["task_id"])
        task_group[key]["quality"].append(float(row["quality"]))
        task_group[key]["cloud"].append(float(row["cloud_tokens_spent"]))

    for (policy, budget_level, task_id), stats in sorted(
        task_group.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])
    ):
        task_lines.append(
            f"| {policy} | {budget_level:.2f} | {task_id} | "
            f"{_mean(stats['quality']):.3f} | {_mean(stats['cloud']):.1f} |"
        )

    OUTPUT_TABLE.write_text("\n".join(task_lines) + "\n", encoding="utf-8")

    return {
        "frontier": frontier,
        "plot_path": str(OUTPUT_PLOT),
        "task_table_path": str(OUTPUT_TABLE),
    }


# ---------------------------------------------------------------------------
# Three-axis DSE: quality × cloud spend × hardware BOM cost
# ---------------------------------------------------------------------------

def cheapest_policy_per_floor(
    results: list[dict[str, Any]],
    hw_name: str,
    floors: list[float] = QUALITY_FLOORS,
) -> list[dict[str, Any]]:
    """For each quality floor, return the lowest-cloud-cost (policy, budget_level)
    that meets the floor when running on hw_name hardware.

    Each element of the returned list:
      floor, hw_name, policy, budget_level, mean_quality, mean_cloud_tokens
    """
    hw_rows = [r for r in results if r.get("hardware_config", "") == hw_name]
    if not hw_rows:
        hw_rows = results

    by_pb: dict[tuple, dict] = defaultdict(lambda: {"quality": [], "cloud": []})
    for row in hw_rows:
        key = (row["policy"], row["budget_level"])
        by_pb[key]["quality"].append(float(row["quality"]))
        by_pb[key]["cloud"].append(float(row["cloud_tokens_spent"]))

    candidates = [
        {
            "policy": policy,
            "budget_level": budget_level,
            "mean_quality": _mean(stats["quality"]),
            "mean_cloud_tokens": _mean(stats["cloud"]),
        }
        for (policy, budget_level), stats in by_pb.items()
    ]

    table: list[dict[str, Any]] = []
    for floor in floors:
        feasible = [c for c in candidates if c["mean_quality"] >= floor]
        if not feasible:
            table.append({"floor": floor, "hw_name": hw_name, "policy": None,
                          "budget_level": None, "mean_quality": None,
                          "mean_cloud_tokens": None})
            continue
        best = min(feasible, key=lambda c: c["mean_cloud_tokens"])
        table.append({
            "floor": floor,
            "hw_name": hw_name,
            "policy": best["policy"],
            "budget_level": best["budget_level"],
            "mean_quality": round(best["mean_quality"], 3),
            "mean_cloud_tokens": round(best["mean_cloud_tokens"], 1),
        })
    return table


def generate_bom_artifacts(
    input_path: Path = INPUT_PATH,
    hw_dir: Path = HW_DIR,
) -> dict:
    """Produce quality × cloud frontier plots with BOM cost encoded as marker
    size, one subplot per hardware config.

    Also writes a Markdown table of cheapest-policy-per-quality-floor per hw config.
    """
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    rows = payload.get("results", [])
    hw_configs = _load_hw_configs(hw_dir)

    if not hw_configs:
        return {"bom_plot_path": None, "bom_table_path": None, "hw_configs": {}}

    hw_names = sorted(hw_configs.keys())
    n_hw = len(hw_names)
    fig, axes = plt.subplots(1, n_hw, figsize=(8 * n_hw, 6), squeeze=False)

    # Collect Pareto points per (hw, policy, budget_level)
    def _aggregate(hw_name: str):
        hw_rows = [r for r in rows if r.get("hardware_config", "") == hw_name]
        if not hw_rows:
            hw_rows = rows
        by_pb: dict[tuple, dict] = defaultdict(lambda: {"quality": [], "cloud": []})
        for row in hw_rows:
            key = (row["policy"], row["budget_level"])
            by_pb[key]["quality"].append(float(row["quality"]))
            by_pb[key]["cloud"].append(float(row["cloud_tokens_spent"]))
        return [
            {"policy": p, "budget_level": bl,
             "mean_quality": _mean(s["quality"]),
             "mean_cloud": _mean(s["cloud"])}
            for (p, bl), s in by_pb.items()
        ]

    for col, hw_name in enumerate(hw_names):
        ax = axes[0][col]
        cfg = hw_configs[hw_name]
        bom = float(cfg.get("bom_cost_usd", 0))
        points_by_policy: dict[str, list] = defaultdict(list)
        for pt in _aggregate(hw_name):
            points_by_policy[pt["policy"]].append(pt)

        for policy, pts in sorted(points_by_policy.items()):
            xs = [p["mean_cloud"] for p in pts]
            ys = [p["mean_quality"] for p in pts]
            mask = _is_pareto_optimal(list(zip(xs, ys)))
            sizes = [max(40, min(300, bom / 5)) for _ in xs]
            ax.scatter(
                xs, ys,
                s=sizes,
                alpha=0.7,
                label=policy,
            )
            # highlight Pareto-optimal points
            opt_xs = [x for x, m in zip(xs, mask) if m]
            opt_ys = [y for y, m in zip(ys, mask) if m]
            if opt_xs:
                ax.scatter(opt_xs, opt_ys, s=[sz * 1.6 for sz in sizes[:len(opt_xs)]],
                           edgecolors="black", linewidths=1.2, facecolors="none")

        mem = cfg.get("memory_gb", "?")
        npu = f" NPU {cfg.get('npu_tops')} TOPS" if cfg.get("has_npu") else " no NPU"
        ax.set_title(f"{hw_name}\n{mem} GB{npu} — BOM ${bom:.0f}", fontsize=10)
        ax.set_xlabel("Cloud Tokens Spent")
        ax.set_ylabel("Quality Score")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    fig.suptitle("Quality vs Cloud Spend (marker size ~ BOM cost)\nCircle outline = Pareto-optimal", fontsize=11)
    fig.tight_layout()
    OUTPUT_BOM_PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_BOM_PLOT, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Cheapest-policy-per-floor table
    floor_lines = [
        "# Cheapest Policy per Quality Floor per Hardware Config",
        "",
        "| hardware | quality_floor | policy | budget_level | mean_quality | mean_cloud_tokens |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for hw_name in hw_names:
        for row in cheapest_policy_per_floor(rows, hw_name):
            policy = row["policy"] or "—"
            bl = f"{row['budget_level']:.2f}" if row["budget_level"] is not None else "—"
            mq = f"{row['mean_quality']:.3f}" if row["mean_quality"] is not None else "—"
            mc = f"{row['mean_cloud_tokens']:.1f}" if row["mean_cloud_tokens"] is not None else "—"
            floor_lines.append(
                f"| {hw_name} | {row['floor']:.1f} | {policy} | {bl} | {mq} | {mc} |"
            )

    OUTPUT_BOM_TABLE.write_text("\n".join(floor_lines) + "\n", encoding="utf-8")

    return {
        "bom_plot_path": str(OUTPUT_BOM_PLOT),
        "bom_table_path": str(OUTPUT_BOM_TABLE),
        "hw_configs": {k: {"bom_cost_usd": v.get("bom_cost_usd")} for k, v in hw_configs.items()},
    }


# ---------------------------------------------------------------------------
# Envelope-model Pareto sweep (B6c): quality floor vs cloud spend, on whatever
# REAL data exists in this repo. There is no full outcome table yet (B4 --
# the per-task quality/cost/latency table this would ideally sweep -- has not
# been built; results/pareto_results.json does not exist). This sweep instead
# runs analysis/envelope_model.py's real, already-fit quantities (the A-24
# budget-boundary fit, the R1b truncation-cliff quality-vs-ratio table, and
# the per-machine H2 co-runner terms from B5) through src/dse/router.py at a
# small, EXPLICITLY REAL set of (machine, model, context_ratio) points --
# the same points envelope_model itself was fit/validated on, not a
# synthesized task distribution. What is real: the quality scores at each
# ratio (R1B_BASELINE_SCORE_BY_RATIO, measured), the feasibility status at
# each context length (A-24 fit), and the co-runner multipliers (B5, PX2).
# What is NOT real/complete: there is no measured cloud-side quality number
# anywhere in this repo (no live cloud call has ever been made for this
# project), so "quality if routed to cloud" is not plotted -- only the
# routing outcome (local/cloud/compact/refuse) and the real local quality
# the router would have gotten by routing there, are reported. This is
# stated here, not smoothed over, because synthesizing a cloud-quality
# number would defeat the point of this being a real-data sweep.
# ---------------------------------------------------------------------------

def generate_envelope_pareto_sweep(
    quality_floors: list[float] | None = None,
    spend_caps_usd: list[float] | None = None,
) -> dict[str, Any]:
    """Sweeps (quality_floor, spend_cap) over a small, real set of (machine, model, ratio) points built from
    envelope_model's own real constants, routing each through src/dse/router.EnvelopeRouter against the stub
    CloudClient, and reports the resulting local/cloud/compact/refuse split and achieved quality per cell.

    Returns {"points": [...], "sweep": [...], "n_real_points": N, "note": "..."} -- never writes a plot or table
    file claiming a populated frontier; this is reported as a small/partial real sweep, not padded with
    synthesized outcome rows.
    """
    import sys as _sys
    import tempfile
    from pathlib import Path as _Path

    _sys.path.insert(0, str(REPO_ROOT))
    _sys.path.insert(0, str(REPO_ROOT / "analysis"))
    import envelope_model as _em  # noqa: PLC0415
    from src.cloud.client import CloudClient  # noqa: PLC0415
    from src.dse.router import EnvelopeRouter  # noqa: PLC0415

    quality_floors = quality_floors if quality_floors is not None else [0.0, 0.3, 0.5, 0.8, 1.0]
    spend_caps_usd = spend_caps_usd if spend_caps_usd is not None else [0.0, 1.0, 50.0]

    # Real points: every (ratio -> real score) pair in R1B_BASELINE_SCORE_BY_RATIO (measured, both machines, R1b
    # evaluation audit), each materialized as a task against evo-x2's real measured K1 effective context for
    # qwen3-8b (K1_X2_EFFECTIVE_CTX, 40,960 -- MEASURED, see envelope_model.py) with a full_prompt_tokens chosen
    # (120,000) large enough that every real ratio bucket actually exceeds that effective context (so each
    # point lands in the real truncation_cliff regime and gets its real measured score, not silently shortcut to
    # "full" by a model-native-context coincidence). evo-x2 is used (not evo-t2s) specifically because it has no
    # A-24-equivalent memory-budget bisection, so feasibility there is NOT_MEASURED and does not confound this
    # quality-only sweep with the separate memory-budget failure axis (see predict_feasibility).
    effective_ctx = _em.K1_X2_EFFECTIVE_CTX["qwen3-8b"]
    full_prompt_tokens = 120_000
    assert full_prompt_tokens * min(_em.R1B_BASELINE_SCORE_BY_RATIO) > effective_ctx, \
        "sweep construction invariant: even the smallest real ratio must exceed the real effective context"
    points = []
    for ratio in sorted(_em.R1B_BASELINE_SCORE_BY_RATIO):
        points.append({
            "task_id": f"r1b_ratio_{ratio}",
            "model_id": "qwen3-8b",
            "context_length": full_prompt_tokens,
            "prompt_tokens": int(ratio * full_prompt_tokens),
            "full_prompt_tokens": full_prompt_tokens,
            "prompt": "x",
            "real_score_at_ratio": _em.R1B_BASELINE_SCORE_BY_RATIO[ratio],
        })

    sweep_rows = []
    for floor in quality_floors:
        for cap in spend_caps_usd:
            with tempfile.TemporaryDirectory() as tmp:
                client = CloudClient(api_key=None, spend_cap_usd=cap, ledger_path=_Path(tmp) / "ledger.jsonl")
                router = EnvelopeRouter(client, machine="evo-x2", runtime_policy="ollama_default",
                                        quality_floor=floor)
                targets = [router.route(p).target for p in points]
            n = len(targets)
            sweep_rows.append({
                "quality_floor": floor, "spend_cap_usd": cap,
                "frac_local": targets.count("local") / n,
                "frac_cloud": sum(1 for t in targets if t.startswith("cloud")) / n,
                "frac_compact": targets.count("compact_then_local") / n,
                "frac_refuse": targets.count("refuse") / n,
            })

    return {
        "points": points,
        "sweep": sweep_rows,
        "n_real_points": len(points),
        "note": (
            f"Swept {len(points)} real (ratio -> measured R1b score) points x {len(quality_floors)} quality "
            f"floors x {len(spend_caps_usd)} spend caps against the stub CloudClient. This is NOT a full Pareto "
            f"frontier over a labeled outcome table -- no such table exists yet in this repo (B4 not built, "
            f"results/pareto_results.json absent). It is a real-data sweep of the only measured quality-vs-"
            f"context-ratio relationship this repo has (R1b truncation cliff), through the real router logic."
        ),
    }


def main() -> None:
    out = generate_pareto_artifacts()
    print(f"Pareto plot:  {out['plot_path']}")
    print(f"Task table:   {out['task_table_path']}")

    bom = generate_bom_artifacts()
    if bom.get("bom_plot_path"):
        print(f"BOM plot:     {bom['bom_plot_path']}")
        print(f"BOM table:    {bom['bom_table_path']}")
        for hw, info in bom["hw_configs"].items():
            print(f"  {hw}: BOM ${info['bom_cost_usd']}")


if __name__ == "__main__":
    main()
