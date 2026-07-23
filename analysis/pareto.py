"""Pareto analysis for quality vs cloud-token spend."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


INPUT_PATH = Path("results") / "pareto_results.json"
OUTPUT_PLOT = Path("reports") / "pareto_frontier.png"
OUTPUT_TABLE = Path("reports") / "pareto_task_breakdown.md"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    return (sum((v - mu) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def generate_pareto_artifacts(input_path: Path = INPUT_PATH) -> dict:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    rows = payload.get("results", [])

    grouped = defaultdict(list)
    for row in rows:
        key = (row["policy"], row["budget_level"], row["seed"])
        grouped[key].append(row)

    frontier = defaultdict(lambda: {"x": [], "y": [], "yerr": []})
    by_policy_budget = defaultdict(lambda: {"quality": [], "cloud": []})

    for (policy, budget_level, seed), recs in grouped.items():
        mean_quality = _mean([float(r["quality"]) for r in recs])
        cloud_spent = sum(int(r["cloud_tokens_spent"]) for r in recs)
        by_policy_budget[(policy, budget_level)]["quality"].append(mean_quality)
        by_policy_budget[(policy, budget_level)]["cloud"].append(float(cloud_spent))

    for (policy, budget_level), stats in sorted(by_policy_budget.items(), key=lambda x: (x[0][0], x[0][1])):
        frontier[policy]["x"].append(_mean(stats["cloud"]))
        frontier[policy]["y"].append(_mean(stats["quality"]))
        frontier[policy]["yerr"].append(_stdev(stats["quality"]))

    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for policy, curve in sorted(frontier.items()):
        plt.errorbar(curve["x"], curve["y"], yerr=curve["yerr"], marker="o", capsize=3, label=policy)

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

    task_group = defaultdict(lambda: {"quality": [], "cloud": []})
    for row in rows:
        key = (row["policy"], row["budget_level"], row["task_id"])
        task_group[key]["quality"].append(float(row["quality"]))
        task_group[key]["cloud"].append(float(row["cloud_tokens_spent"]))

    for (policy, budget_level, task_id), stats in sorted(task_group.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        task_lines.append(
            f"| {policy} | {budget_level:.2f} | {task_id} | {_mean(stats['quality']):.3f} | {_mean(stats['cloud']):.1f} |"
        )

    OUTPUT_TABLE.write_text("\n".join(task_lines) + "\n", encoding="utf-8")

    return {
        "frontier": frontier,
        "plot_path": str(OUTPUT_PLOT),
        "task_table_path": str(OUTPUT_TABLE),
    }


def main() -> None:
    out = generate_pareto_artifacts()
    print(f"Pareto plot: {out['plot_path']}")
    print(f"Task table: {out['task_table_path']}")


if __name__ == "__main__":
    main()
