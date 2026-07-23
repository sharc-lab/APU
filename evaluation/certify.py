"""Sampled verification and certified-quality estimation for local outputs."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.quality import QualityEvaluator
from harness.adapters.sdk_direct import TASKS

INPUT_PATH = Path("results") / "pareto_results.json"
OUTPUT_PATH = Path("results") / "certified_quality.json"
TRACES_ROOTS = [Path("analysis") / "traces", Path("analysis") / "traces_ollama"]


def _wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2 * n)) / denom
    margin = (z / denom) * ((p * (1 - p) / n + (z * z) / (4 * n * n)) ** 0.5)
    return max(0.0, center - margin), min(1.0, center + margin)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _extract_content(response: dict[str, Any]) -> str:
    choices = response.get("choices", [])
    if not choices:
        return ""
    msg = choices[0].get("message", {})
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    return str(content)


def _load_hash_to_output_map() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for root in TRACES_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            response = payload.get("response")
            if not isinstance(response, dict):
                continue
            content = _extract_content(response)
            digest = _hash_text(content)
            out[digest] = response
    return out


def certify_local_quality(
    *,
    sample_fraction: float = 0.2,
    pass_threshold: float = 6.0,
    random_seed: int = 7,
    input_path: Path = INPUT_PATH,
) -> dict[str, Any]:
    sample_fraction = min(1.0, max(0.0, sample_fraction))
    data = json.loads(input_path.read_text(encoding="utf-8"))
    rows = data.get("results", [])

    hash_to_response = _load_hash_to_output_map()
    evaluator = QualityEvaluator(replay_mode="AUTO")
    rng = random.Random(random_seed)

    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("policy")), float(row.get("budget_level", 0.0)))].append(row)

    certified_rows: list[dict[str, Any]] = []

    for (policy, budget_level), entries in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        per_category: dict[str, dict[str, Any]] = defaultdict(lambda: {"n": 0, "verified": 0, "success": 0, "scores": []})

        for entry in entries:
            if int(entry.get("local_tokens", 0)) <= 0:
                continue
            category = str(entry.get("category", "unknown"))
            per_category[category]["n"] += 1

            if rng.random() > sample_fraction:
                continue

            local_hash = None
            decisions = entry.get("routing_decisions") or []
            if decisions:
                local_hash = decisions[0].get("local_output_hash")
            response_json = hash_to_response.get(local_hash or "")
            if not response_json:
                continue

            task_id = str(entry.get("task_id"))
            task_prompt = TASKS.get(task_id, {}).get("prompt", "")
            scored = evaluator.score_task(task_id=task_id, task_prompt=task_prompt, response_json=response_json)

            per_category[category]["verified"] += 1
            per_category[category]["scores"].append(float(scored.get("score", 0.0)))
            if float(scored.get("score", 0.0)) >= pass_threshold:
                per_category[category]["success"] += 1

        category_stats: dict[str, Any] = {}
        weighted_lower_sum = 0.0
        weighted_n = 0

        for category, stats in per_category.items():
            verified = int(stats["verified"])
            success = int(stats["success"])
            lower, upper = _wilson_interval(success, verified)
            mean_score = sum(stats["scores"]) / verified if verified else 0.0

            category_stats[category] = {
                "local_samples": int(stats["n"]),
                "verified_samples": verified,
                "pass_rate": (success / verified) if verified else 0.0,
                "wilson_low": lower,
                "wilson_high": upper,
                "mean_verified_score": mean_score,
            }
            weighted_lower_sum += lower * verified
            weighted_n += verified

        certified_quality = (weighted_lower_sum / weighted_n) * 10.0 if weighted_n else 0.0
        certified_rows.append(
            {
                "policy": policy,
                "budget_level": budget_level,
                "sample_fraction": sample_fraction,
                "pass_threshold": pass_threshold,
                "certified_quality": certified_quality,
                "categories": category_stats,
            }
        )

    artifact = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "sample_fraction": sample_fraction,
        "pass_threshold": pass_threshold,
        "rows": certified_rows,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact


def main() -> None:
    artifact = certify_local_quality()
    print(f"Certified rows: {len(artifact['rows'])}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
