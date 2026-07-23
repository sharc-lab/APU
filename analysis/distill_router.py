"""Distill a learned router from routing logs and replay traces."""

from __future__ import annotations

import hashlib
import json
import pickle
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from harness.adapters.sdk_direct import TASKS

INPUT_PATH = Path("results") / "pareto_results.json"
TRACES_ROOTS = [Path("analysis") / "traces", Path("analysis") / "traces_ollama"]
OUTPUT_MODEL = Path("analysis") / "learned_router_model.pkl"
OUTPUT_REPORT = Path("results") / "learned_router_eval.json"

CATEGORY_IDS = sorted({v["category"] for v in TASKS.values()})
CATEGORY_TO_INDEX = {c: i for i, c in enumerate(CATEGORY_IDS)}


def _extract_content(response: dict[str, Any]) -> str:
    choices = response.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return str(content)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _load_trace_features() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
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
            text = _extract_content(response)
            digest = _hash_text(text)
            usage = response.get("usage") or {}
            out[digest] = {
                "trace_output_len": float(len(text)),
                "trace_total_tokens": float(usage.get("total_tokens") or 0),
                "trace_prompt_tokens": float(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
                "trace_completion_tokens": float(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
            }
    return out


def _label_from_row(row: dict[str, Any]) -> int | None:
    if row.get("speculative_agreed") is not None:
        return 1 if bool(row.get("speculative_agreed")) else 0
    if row.get("spec_known_agreement") is not None:
        return 1 if bool(row.get("spec_known_agreement")) else 0
    return None


def _build_dataset(input_path: Path = INPUT_PATH) -> tuple[np.ndarray, np.ndarray, list[str], list[dict[str, Any]]]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    rows = payload.get("results", [])
    trace_features = _load_trace_features()

    feature_names = [
        "category_id",
        "budget_level",
        "remaining_budget",
        "theta",
        "confidence",
        "escalate",
        "cloud_tokens_spent",
        "local_tokens",
        "trace_output_len",
        "trace_total_tokens",
    ]

    x_rows: list[list[float]] = []
    y_rows: list[int] = []
    meta_rows: list[dict[str, Any]] = []

    for row in rows:
        label = _label_from_row(row)
        if label is None:
            continue

        category = str(row.get("category", ""))
        category_id = float(CATEGORY_TO_INDEX.get(category, -1))
        budget_level = float(row.get("budget_level", 0.0))
        cloud_tokens = float(row.get("cloud_tokens_spent", 0.0))
        local_tokens = float(row.get("local_tokens", 0.0))

        decisions = row.get("routing_decisions") or []
        d0 = decisions[0] if decisions else {}
        remaining_budget = float(d0.get("remaining_budget", 0.0) or 0.0)
        theta = float(d0.get("theta", 0.0) or 0.0)
        confidence = float(d0.get("confidence", 0.0) or 0.0)
        escalate = 1.0 if bool(d0.get("escalate", False)) else 0.0

        local_hash = d0.get("local_output_hash")
        tfeat = trace_features.get(local_hash, {})

        features = [
            category_id,
            budget_level,
            remaining_budget,
            theta,
            confidence,
            escalate,
            cloud_tokens,
            local_tokens,
            float(tfeat.get("trace_output_len", 0.0)),
            float(tfeat.get("trace_total_tokens", 0.0)),
        ]

        x_rows.append(features)
        y_rows.append(label)
        meta_rows.append(row)

    return np.array(x_rows, dtype=float), np.array(y_rows, dtype=int), feature_names, meta_rows


def _evaluate_replay_policy(meta_rows: list[dict[str, Any]], preds: np.ndarray) -> dict[str, Any]:
    total_cloud = 0
    total_local = 0
    total_quality = 0.0
    n = len(meta_rows)

    for row, pred in zip(meta_rows, preds.tolist()):
        local_adequate = bool(_label_from_row(row))
        choose_local = bool(pred)

        if choose_local:
            total_local += 1
            if local_adequate:
                total_quality += float(row.get("quality", 0.0))
            else:
                total_quality += max(0.0, float(row.get("quality", 0.0)) - 2.0)
        else:
            total_cloud += 1
            total_cloud += int(row.get("cloud_tokens_spent", 0))
            total_quality += float(row.get("quality", 0.0))

    return {
        "n_samples": n,
        "mean_quality": (total_quality / n) if n else 0.0,
        "cloud_cost_proxy": total_cloud,
        "local_decisions": total_local,
    }


def distill_router(input_path: Path = INPUT_PATH) -> dict[str, Any]:
    x, y, feature_names, meta_rows = _build_dataset(input_path=input_path)
    if len(y) < 10 or len(set(y.tolist())) < 2:
        raise ValueError("Insufficient labeled speculative examples to train router")

    x_train, x_test, y_train, y_test, meta_train, meta_test = train_test_split(
        x,
        y,
        meta_rows,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    lr = LogisticRegression(max_iter=1000)
    gbt = HistGradientBoostingClassifier(random_state=42)

    lr.fit(x_train, y_train)
    gbt.fit(x_train, y_train)

    models = {"logistic_regression": lr, "gradient_boosted_trees": gbt}
    metrics: dict[str, dict[str, float]] = {}

    best_name = None
    best_auc = -1.0

    for name, model in models.items():
        pred = model.predict(x_test)
        proba = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else pred.astype(float)
        auc = roc_auc_score(y_test, proba)
        metrics[name] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "f1": float(f1_score(y_test, pred)),
            "roc_auc": float(auc),
        }
        if auc > best_auc:
            best_auc = auc
            best_name = name

    assert best_name is not None
    best_model = models[best_name]

    payload = {
        "best_model_name": best_name,
        "best_model": best_model,
        "feature_names": feature_names,
        "trained_utc": datetime.now(timezone.utc).isoformat(),
    }
    OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MODEL.write_bytes(pickle.dumps(payload))

    replay_pred = best_model.predict(x_test)
    replay_eval = _evaluate_replay_policy(meta_test, replay_pred)

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "n_rows": int(len(y)),
        "feature_names": feature_names,
        "model_metrics": metrics,
        "best_model_name": best_name,
        "model_path": str(OUTPUT_MODEL),
        "replay_mode_policy_eval": replay_eval,
    }

    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    report = distill_router()
    print(f"Best model: {report['best_model_name']}")
    print(f"Model path: {report['model_path']}")
    print(f"Eval path: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
