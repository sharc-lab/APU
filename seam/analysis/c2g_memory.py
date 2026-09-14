"""C2g: peak memory versus context - linear vs superlinear fit (timed tasks only).



Governing: ``docs/CURSOR_PROMPT_C2g.md`` §2. Points come from per-step RSS / free-memory

instrumentation on timed tasks (discarded warmup excluded). Linear scaling is consistent with

KV plus a constant; superlinear (context^2) implicates attention activation memory.

"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "DEFAULT_NOTE_PATH",
    "collect_memory_context_points",
    "fit_memory_vs_context",
    "memory_vs_context_note_template",
    "write_memory_vs_context_note",
]


DEFAULT_NOTE_PATH = Path("derived/efilter/c2g_memory_vs_context_note.json")


def collect_memory_context_points(per_task: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten timed-task step_memory rows into fit points (skip discarded warmup)."""

    points: list[dict[str, Any]] = []

    for task in per_task:
        if task.get("discarded_warmup"):
            continue

        task_id = task.get("task_id")

        for row in task.get("step_memory") or []:
            ctx = row.get("context_tokens_total")

            rss_peak = row.get("rss_peak_during_generate")

            if ctx is None or rss_peak is None:
                continue

            free_min = row.get("free_memory_mb_min_during_generate")

            free_before = row.get("free_memory_mb_before_generate")

            free_drop = None

            if free_before is not None and free_min is not None:
                free_drop = float(free_before) - float(free_min)

            points.append(
                {
                    "task_id": task_id,
                    "step_idx": row.get("step_idx"),
                    "context_tokens": int(ctx),
                    "rss_peak_bytes": int(rss_peak),
                    "free_memory_mb_min": (float(free_min) if free_min is not None else None),
                    "free_drop_mb": free_drop,
                }
            )

    return points


def _r_squared(y: np.ndarray, y_hat: np.ndarray) -> float | None:
    if y.size < 2:
        return None

    ss_res = float(np.sum((y - y_hat) ** 2))

    ss_tot = float(np.sum((y - np.mean(y)) ** 2))

    if ss_tot <= 0.0:
        return None

    return 1.0 - (ss_res / ss_tot)


def _fit_series(xs: np.ndarray, ys: np.ndarray) -> dict[str, Any]:
    """Fit linear (a + b x) and quadratic (a + b x + c x^2); report R^2 for each."""

    n = int(xs.size)

    out: dict[str, Any] = {
        "n_points": n,
        "context_tokens_min": int(xs.min()) if n else None,
        "context_tokens_max": int(xs.max()) if n else None,
        "linear": None,
        "quadratic": None,
        "verdict": "insufficient_points",
    }

    if n < 2:
        return out

    # Linear: y = a + b x

    coef_lin = np.polyfit(xs, ys, deg=1)

    y_lin = np.polyval(coef_lin, xs)

    r2_lin = _r_squared(ys, y_lin)

    out["linear"] = {
        "form": "a + b*context",
        "a": float(coef_lin[1]),
        "b": float(coef_lin[0]),
        "r_squared": r2_lin,
    }

    if n >= 3:
        coef_quad = np.polyfit(xs, ys, deg=2)

        y_quad = np.polyval(coef_quad, xs)

        r2_quad = _r_squared(ys, y_quad)

        out["quadratic"] = {
            "form": "a + b*context + c*context^2",
            "a": float(coef_quad[2]),
            "b": float(coef_quad[1]),
            "c": float(coef_quad[0]),
            "r_squared": r2_quad,
        }

        # Superlinear if quadratic term is positive and materially improves R^2.

        if (
            r2_lin is not None
            and r2_quad is not None
            and coef_quad[0] > 0
            and (r2_quad - r2_lin) >= 0.02
            and r2_quad >= 0.5
        ):
            out["verdict"] = "superlinear"

        elif r2_lin is not None and r2_lin >= 0.5:
            out["verdict"] = "linear"

        else:
            out["verdict"] = "inconclusive"

    else:
        if r2_lin is not None and r2_lin >= 0.5:
            out["verdict"] = "linear_n_lt_3"

        else:
            out["verdict"] = "inconclusive_n_lt_3"

    return out


def fit_memory_vs_context(per_task: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit RSS_peak and free_drop versus context for timed tasks only."""

    points = collect_memory_context_points(per_task)

    if not points:
        return {
            "n_points": 0,
            "points": [],
            "rss_peak_vs_context": _fit_series(np.array([]), np.array([])),
            "free_drop_vs_context": _fit_series(np.array([]), np.array([])),
            "overall_verdict": "no_points",
            "note": (
                "Timed-task step_memory rows required. Discarded warmup is excluded. "
                "Fill after the next C2g pilot seals."
            ),
        }

    ctx = np.array([p["context_tokens"] for p in points], dtype=float)

    rss = np.array([p["rss_peak_bytes"] for p in points], dtype=float)

    rss_fit = _fit_series(ctx, rss)

    drop_pts = [p for p in points if p.get("free_drop_mb") is not None]

    if drop_pts:
        ctx_d = np.array([p["context_tokens"] for p in drop_pts], dtype=float)

        drop = np.array([p["free_drop_mb"] for p in drop_pts], dtype=float)

        drop_fit = _fit_series(ctx_d, drop)

    else:
        drop_fit = {
            "n_points": 0,
            "linear": None,
            "quadratic": None,
            "verdict": "no_free_drop_points",
        }

    # Prefer RSS peak for the headline verdict (process-local); free_drop is secondary.

    overall = rss_fit.get("verdict") or "inconclusive"

    return {
        "n_points": len(points),
        "points": points,
        "rss_peak_vs_context": rss_fit,
        "free_drop_vs_context": drop_fit,
        "overall_verdict": overall,
        "interpretation": {
            "linear": "KV plus a constant (activation overhead roughly independent of context).",
            "superlinear": (
                "Attention activations scaling with context^2 - capping alone may be "
                "insufficient; chunked prefill (if available) would be the real fix."
            ),
        },
        "note": "Timed tasks only; discarded warmup excluded.",
    }


def memory_vs_context_note_template() -> dict[str, Any]:
    """Empty note filled by the next pilot (or by write_memory_vs_context_note)."""

    return {
        "note_id": "c2g_memory_vs_context",
        "governing": "docs/CURSOR_PROMPT_C2g.md §2",
        "status": "TEMPLATE",
        "run_id": None,
        "question": (
            "Does peak memory (RSS_peak or free_drop) track context linearly or superlinearly?"
        ),
        "analysis": {
            "n_points": 0,
            "points": [],
            "rss_peak_vs_context": None,
            "free_drop_vs_context": None,
            "overall_verdict": "pending_pilot",
        },
        "decides": (
            "Linear -> cap may be sufficient. Superlinear -> prefill activation is the driver; "
            "chunked prefill (if available) is required to bound memory independent of context."
        ),
    }


def write_memory_vs_context_note(
    path: Path,
    *,
    per_task: list[dict[str, Any]] | None = None,
    run_id: str | None = None,
    analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write (or refresh) ``c2g_memory_vs_context_note.json``.



    With no ``per_task`` / ``analysis``, writes the TEMPLATE for the next pilot to fill.

    """

    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    payload = memory_vs_context_note_template()

    if analysis is None and per_task is not None:
        analysis = fit_memory_vs_context(per_task)

    if analysis is not None:
        payload["status"] = "FILLED" if analysis.get("n_points", 0) > 0 else "TEMPLATE"

        payload["analysis"] = analysis

        payload["overall_verdict"] = analysis.get("overall_verdict")

    if run_id is not None:
        payload["run_id"] = run_id

    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )

    return payload
