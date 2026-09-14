"""Fit E-ATTRIB's centered interaction model and render every validity gate.

The chat-mode mechanism spike failed reuse (run
``6b40e3fe-cbc1-4b6a-8867-46650b4ead61``), so the active identification route is:

.. code-block:: text

    t = a + P/R_prefill + n_out*d0 + (P*n_out)*d1

``P`` and ``n_out`` are centered before their interaction is formed.  Reported coefficients are
then transformed back to the physical, uncentered parameterization above.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy import stats  # type: ignore[import-untyped]

from seam.config import resolve_config
from seam.errors import SeamError
from seam.rawstore import open_run_dir, verify_sealed

__all__ = [
    "FitResult",
    "RankDeficientDesignError",
    "analyze",
    "fit_interaction",
    "main",
]

_PARAMETERS = ("a", "inv_R_prefill", "d0", "d1")
_PHYSICAL_PARAMETERS = ("a", "R_prefill", "d0", "d1")


class RankDeficientDesignError(SeamError):
    """The four interaction-model terms cannot be separated."""


@dataclass(frozen=True, slots=True)
class Interval:
    point: float
    lo: float
    hi: float
    confidence: float
    method: str
    n: int

    def excludes_zero(self) -> bool:
        return (self.lo > 0.0 and self.hi > 0.0) or (self.lo < 0.0 and self.hi < 0.0)


@dataclass(frozen=True, slots=True)
class FitResult:
    """One target-by-quantization interaction fit."""

    target: str
    quantization: str
    n: int
    p_center: float
    n_out_center: float
    rank: int
    condition_number: float
    vif: dict[str, float]
    parameters: dict[str, Interval]
    covariance: list[list[float]]
    correlation: list[list[float]]
    max_abs_parameter_correlation: float
    r_squared: float
    adjusted_r_squared: float
    residuals: list[float]
    predictions: list[float]
    bootstrap_valid_draws: int
    bootstrap_rank_deficient_draws: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["parameters"] = {
            name: asdict(interval) for name, interval in self.parameters.items()
        }
        return payload


def _rows(
    records: Sequence[Mapping[str, Any]], *, include_held_out: bool = False
) -> list[Mapping[str, Any]]:
    selected = [
        record for record in records if include_held_out or not bool(record.get("held_out", False))
    ]
    if len(selected) < 4:
        raise RankDeficientDesignError(
            f"interaction fit needs at least four observations, got {len(selected)}"
        )
    return selected


def _design(
    records: Sequence[Mapping[str, Any]],
    *,
    p_center: float | None = None,
    n_center: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    prompt = np.asarray([float(row["total_prompt_tokens"]) for row in records], dtype=float)
    n_out = np.asarray([float(row["n_out"]) for row in records], dtype=float)
    wall = np.asarray([float(row["wall_s"]) for row in records], dtype=float)
    p_mean = float(prompt.mean()) if p_center is None else float(p_center)
    n_mean = float(n_out.mean()) if n_center is None else float(n_center)
    p_c = prompt - p_mean
    n_c = n_out - n_mean
    matrix = np.column_stack((np.ones(len(records)), p_c, n_c, p_c * n_c))
    return matrix, wall, p_mean, n_mean


def _require_full_rank(matrix: np.ndarray, *, where: str) -> None:
    rank = int(np.linalg.matrix_rank(matrix))
    if rank != matrix.shape[1]:
        raise RankDeficientDesignError(
            f"rank-deficient {where} design: rank={rank}, columns={matrix.shape[1]}. "
            "Refusing a pseudo-inverse; a coefficient split from this matrix is arbitrary."
        )


def _standardized_condition_number(matrix: np.ndarray) -> float:
    standardized = matrix.copy()
    for col in range(1, matrix.shape[1]):
        sd = float(np.std(matrix[:, col], ddof=1))
        if sd <= 0.0:
            return math.inf
        standardized[:, col] /= sd
    return float(np.linalg.cond(standardized))


def _vifs(matrix: np.ndarray) -> dict[str, float]:
    names = ("total_prompt_centered", "n_out_centered", "centered_interaction")
    result: dict[str, float] = {}
    for index, name in enumerate(names, start=1):
        response = matrix[:, index]
        others = np.delete(matrix, index, axis=1)
        _require_full_rank(others, where=f"VIF auxiliary for {name}")
        beta = np.linalg.lstsq(others, response, rcond=None)[0]
        residual = response - others @ beta
        ss_res = float(residual @ residual)
        centered = response - float(response.mean())
        ss_tot = float(centered @ centered)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
        result[name] = math.inf if r_squared >= 1.0 else 1.0 / (1.0 - r_squared)
    return result


def _physical_transform(p_center: float, n_center: float) -> np.ndarray:
    """Map centered beta to ``[a, inv_R_prefill, d0, d1]``."""
    return np.asarray(
        [
            [1.0, -p_center, -n_center, p_center * n_center],
            [0.0, 1.0, 0.0, -n_center],
            [0.0, 0.0, 1.0, -p_center],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _fit_once(
    records: Sequence[Mapping[str, Any]],
    *,
    p_center: float | None = None,
    n_center: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    matrix, wall, p_mean, n_mean = _design(records, p_center=p_center, n_center=n_center)
    _require_full_rank(matrix, where="interaction")
    beta = np.linalg.lstsq(matrix, wall, rcond=None)[0]
    predictions = matrix @ beta
    residuals = wall - predictions
    transform = _physical_transform(p_mean, n_mean)
    physical = transform @ beta
    return physical, predictions, residuals, matrix, p_mean, n_mean


def _bootstrap_parameters(
    records: Sequence[Mapping[str, Any]],
    *,
    p_center: float,
    n_center: float,
    resamples: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    rng = random.Random(seed)
    n = len(records)
    valid: list[np.ndarray] = []
    deficient = 0
    for _ in range(resamples):
        draw = [records[rng.randrange(n)] for _ in range(n)]
        try:
            physical, *_ = _fit_once(draw, p_center=p_center, n_center=n_center)
        except RankDeficientDesignError:
            deficient += 1
            continue
        if physical[1] == 0.0:
            continue
        valid.append(
            np.asarray(
                [physical[0], 1.0 / physical[1], physical[2], physical[3]],
                dtype=float,
            )
        )
    if len(valid) < max(100, resamples // 2):
        raise RankDeficientDesignError(
            f"only {len(valid)}/{resamples} bootstrap draws retained full rank; "
            "the design is not stably identifiable"
        )
    return np.vstack(valid), deficient


def _percentile_interval(
    point: float,
    values: np.ndarray,
    *,
    confidence: float,
    method: str,
    n: int,
) -> Interval:
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(values, [alpha, 1.0 - alpha])
    return Interval(
        point=float(point),
        lo=float(lo),
        hi=float(hi),
        confidence=confidence,
        method=method,
        n=n,
    )


def _covariance_and_correlation(
    matrix: np.ndarray,
    residuals: np.ndarray,
    *,
    p_center: float,
    n_center: float,
    inv_r: float,
) -> tuple[np.ndarray, np.ndarray]:
    dof = matrix.shape[0] - matrix.shape[1]
    if dof <= 0:
        raise RankDeficientDesignError("no residual degrees of freedom for coefficient covariance")
    sigma2 = float(residuals @ residuals) / dof
    centered_cov = sigma2 * np.linalg.inv(matrix.T @ matrix)
    transform = _physical_transform(p_center, n_center)
    physical_cov = transform @ centered_cov @ transform.T
    # Delta method maps inv_R to R for A4's physical-parameter correlation.
    jacobian = np.eye(4)
    jacobian[1, 1] = -1.0 / (inv_r * inv_r)
    physical_cov = jacobian @ physical_cov @ jacobian.T
    sd = np.sqrt(np.diag(physical_cov))
    correlation = physical_cov / np.outer(sd, sd)
    return physical_cov, correlation


def fit_interaction(
    records: Sequence[Mapping[str, Any]],
    *,
    target: str = "cpu-p",
    quantization: str = "int4",
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 20260803,
    confidence: float = 0.95,
) -> FitResult:
    """Fit the centered interaction route and return physical parameters.

    A rank-deficient matrix raises :class:`RankDeficientDesignError` before any least-squares
    result is returned.
    """
    training = _rows(records)
    physical, predictions, residuals, matrix, p_center, n_center = _fit_once(training)
    inv_r = float(physical[1])
    r_prefill = math.inf if inv_r == 0.0 else 1.0 / inv_r
    bootstrap, deficient = _bootstrap_parameters(
        training,
        p_center=p_center,
        n_center=n_center,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    points = (float(physical[0]), r_prefill, float(physical[2]), float(physical[3]))
    intervals = {
        name: _percentile_interval(
            point,
            bootstrap[:, index],
            confidence=confidence,
            method="pairs_bootstrap_percentile",
            n=len(training),
        )
        for index, (name, point) in enumerate(zip(_PHYSICAL_PARAMETERS, points, strict=True))
    }
    covariance, correlation = _covariance_and_correlation(
        matrix,
        residuals,
        p_center=p_center,
        n_center=n_center,
        inv_r=inv_r,
    )
    wall = np.asarray([float(row["wall_s"]) for row in training], dtype=float)
    ss_res = float(residuals @ residuals)
    centered_wall = wall - float(wall.mean())
    ss_tot = float(centered_wall @ centered_wall)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    n_obs, n_params = matrix.shape
    adjusted = 1.0 - (1.0 - r_squared) * (n_obs - 1) / (n_obs - n_params)
    off_diagonal = np.abs(correlation - np.eye(4))
    return FitResult(
        target=target,
        quantization=quantization,
        n=len(training),
        p_center=p_center,
        n_out_center=n_center,
        rank=int(np.linalg.matrix_rank(matrix)),
        condition_number=_standardized_condition_number(matrix),
        vif=_vifs(matrix),
        parameters=intervals,
        covariance=covariance.tolist(),
        correlation=correlation.tolist(),
        max_abs_parameter_correlation=float(off_diagonal.max()),
        r_squared=r_squared,
        adjusted_r_squared=adjusted,
        residuals=[float(value) for value in residuals],
        predictions=[float(value) for value in predictions],
        bootstrap_valid_draws=len(bootstrap),
        bootstrap_rank_deficient_draws=deficient,
    )


def predict(fit: FitResult, *, total_prompt_tokens: float, n_out: float) -> float:
    params = fit.parameters
    return (
        params["a"].point
        + total_prompt_tokens / params["R_prefill"].point
        + n_out * params["d0"].point
        + n_out * total_prompt_tokens * params["d1"].point
    )


def _mape(actual: Sequence[float], predicted: Sequence[float]) -> float | None:
    errors = [
        abs(float(pred) - float(obs)) / abs(float(obs))
        for obs, pred in zip(actual, predicted, strict=True)
        if float(obs) != 0.0
    ]
    return statistics.fmean(errors) if errors else None


def _aic(actual: np.ndarray, predicted: np.ndarray, n_params: int) -> float:
    residual = actual - predicted
    rss = max(float(residual @ residual), np.finfo(float).tiny)
    n = len(actual)
    return n * math.log(rss / n) + 2.0 * n_params


def _linear_slope_p(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len({float(value) for value in x}) < 2:
        return None
    return float(stats.linregress(x, y).pvalue)


def _coefficient_cv_by_block(
    records: Sequence[Mapping[str, Any]], *, fit_cfg: Mapping[str, Any]
) -> dict[str, float | None]:
    blocks = sorted({int(row["block_idx"]) for row in records if "block_idx" in row})
    values: dict[str, list[float]] = {name: [] for name in _PHYSICAL_PARAMETERS}
    for block in blocks:
        subset = [row for row in records if int(row.get("block_idx", -1)) == block]
        try:
            block_fit = fit_interaction(
                subset,
                bootstrap_resamples=max(200, int(fit_cfg["bootstrap_resamples"]) // 20),
                bootstrap_seed=int(fit_cfg["bootstrap_seed"]) + block,
                confidence=float(fit_cfg["confidence"]),
            )
        except RankDeficientDesignError:
            continue
        for name in values:
            values[name].append(block_fit.parameters[name].point)
    result: dict[str, float | None] = {}
    for name, observed in values.items():
        if len(observed) < 2 or statistics.fmean(observed) == 0.0:
            result[name] = None
        else:
            result[name] = statistics.stdev(observed) / abs(statistics.fmean(observed))
    return result


def _intervals_overlap(left: Interval, right: Interval) -> bool:
    return max(left.lo, right.lo) <= min(left.hi, right.hi)


def _half_stationarity(
    records: Sequence[Mapping[str, Any]], *, fit_cfg: Mapping[str, Any]
) -> dict[str, Any]:
    ordered = sorted(records, key=lambda row: int(row.get("sequence_idx", 0)))
    midpoint = len(ordered) // 2
    halves = (ordered[:midpoint], ordered[midpoint:])
    try:
        fits = [
            fit_interaction(
                half,
                bootstrap_resamples=max(500, int(fit_cfg["bootstrap_resamples"]) // 10),
                bootstrap_seed=int(fit_cfg["bootstrap_seed"]) + index,
                confidence=float(fit_cfg["confidence"]),
            )
            for index, half in enumerate(halves)
        ]
    except RankDeficientDesignError as exc:
        return {"passed": False, "evaluable": False, "reason": str(exc)}
    overlap = {
        name: _intervals_overlap(fits[0].parameters[name], fits[1].parameters[name])
        for name in _PHYSICAL_PARAMETERS
    }
    return {
        "passed": all(overlap.values()),
        "evaluable": True,
        "overlap": overlap,
        "first_half": fits[0].to_dict(),
        "second_half": fits[1].to_dict(),
    }


def _balanced_split_agreement(
    records: Sequence[Mapping[str, Any]], *, fit_cfg: Mapping[str, Any]
) -> dict[str, Any]:
    """Split each P-by-n_out cell alternately so both halves retain the full design."""
    cells: dict[tuple[float, float], list[Mapping[str, Any]]] = {}
    for record in records:
        key = (float(record["total_prompt_tokens"]), float(record["n_out"]))
        cells.setdefault(key, []).append(record)
    left: list[Mapping[str, Any]] = []
    right: list[Mapping[str, Any]] = []
    allocation: dict[str, list[int]] = {}
    for key, values in sorted(cells.items()):
        ordered = sorted(
            values,
            key=lambda row: (
                int(row.get("block_idx", row.get("repeat_idx", 0))),
                int(row.get("sequence_idx", 0)),
            ),
        )
        left.extend(ordered[::2])
        right.extend(ordered[1::2])
        allocation[f"P{key[0]:g}_n{key[1]:g}"] = [
            len(ordered[::2]),
            len(ordered[1::2]),
        ]
    try:
        fits = [
            fit_interaction(
                half,
                bootstrap_resamples=max(500, int(fit_cfg["bootstrap_resamples"]) // 10),
                bootstrap_seed=int(fit_cfg["bootstrap_seed"]) + 100 + index,
                confidence=float(fit_cfg["confidence"]),
            )
            for index, half in enumerate((left, right))
        ]
    except RankDeficientDesignError as exc:
        return {
            "passed": False,
            "evaluable": False,
            "reason": str(exc),
            "allocation": allocation,
        }
    overlap = {
        name: _intervals_overlap(fits[0].parameters[name], fits[1].parameters[name])
        for name in _PHYSICAL_PARAMETERS
    }
    return {
        "passed": all(overlap.values()),
        "evaluable": True,
        "overlap": overlap,
        "allocation": allocation,
        "left": fits[0].to_dict(),
        "right": fits[1].to_dict(),
    }


def _synthetic_recovery_check(cfg: Mapping[str, Any], *, grid: Mapping[str, Any]) -> dict[str, Any]:
    """Run A7 against the live fitter implementation, including the rank-refusal path."""
    truth = {name: float(value) for name, value in cfg["truth"].items()}
    rng = random.Random(int(cfg["seed"]))
    records: list[dict[str, Any]] = []
    for _repeat in range(int(cfg["repeats"])):
        for prompt in grid["total_prompt_tokens"]:
            for n_out in grid["n_out"]:
                wall = (
                    truth["a"]
                    + float(prompt) / truth["R_prefill"]
                    + float(n_out) * truth["d0"]
                    + float(n_out) * float(prompt) * truth["d1"]
                    + rng.gauss(0.0, float(cfg["noise_sd_s"]))
                )
                records.append(
                    {
                        "total_prompt_tokens": int(prompt),
                        "n_out": int(n_out),
                        "wall_s": wall,
                        "held_out": False,
                    }
                )
    fit = fit_interaction(
        records,
        bootstrap_resamples=int(cfg["bootstrap_resamples"]),
        bootstrap_seed=int(cfg["seed"]),
    )
    recovered = {
        name: fit.parameters[name].lo <= expected <= fit.parameters[name].hi
        for name, expected in truth.items()
    }
    deficient = [
        {
            "total_prompt_tokens": prompt,
            "n_out": prompt // 16,
            "wall_s": 0.1 + prompt / 500.0,
            "held_out": False,
        }
        for prompt in (512, 1024, 2048, 4096)
        for _ in range(3)
    ]
    rank_refused = False
    refusal_message: str | None = None
    try:
        fit_interaction(deficient, bootstrap_resamples=200)
    except RankDeficientDesignError as exc:
        rank_refused = True
        refusal_message = str(exc)
    return {
        "passed": all(recovered.values()) and rank_refused,
        "recovered_within_ci": recovered,
        "rank_deficient_refused": rank_refused,
        "rank_refusal_message": refusal_message,
        "fit": fit.to_dict(),
        "truth": truth,
    }


def _agent_steps(root: Path, run_ids: Sequence[str]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for run_id in run_ids:
        run_dir = open_run_dir(run_id, repo_root=root)
        if not run_dir.is_sealed() or not verify_sealed(run_dir):
            raise SeamError(f"agent-validation run {run_id} is not sealed and verified")
        path = run_dir.path / "steps.ndjson"
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("assigned_target") != "local":
                raise SeamError(f"agent-validation run {run_id} contains a non-local step")
            record["source_run_id"] = run_id
            steps.append(record)
    return steps


def _deadline_grid(t_preds: Sequence[float], cfg: Mapping[str, Any]) -> list[float]:
    values = sorted(float(value) for value in t_preds if float(value) > 0.0)
    if not values:
        return []
    lo = float(np.quantile(values, float(cfg["grid_quantile_lo"])))
    hi = float(np.quantile(values, float(cfg["grid_quantile_hi"])))
    pad = float(cfg["grid_pad_factor"])
    return [float(value) for value in np.geomspace(lo / pad, hi * pad, int(cfg["n_deadlines"]))]


def rotation(
    fit: FitResult, steps: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]
) -> dict[str, Any]:
    t_preds = [float(step["routing"]["t_pred_s"]) for step in steps]
    deadlines = _deadline_grid(t_preds, cfg)
    curve: list[dict[str, Any]] = []
    for deadline in deadlines:
        surviving = [step for step in steps if float(step["routing"]["t_pred_s"]) <= deadline]
        r_prefill = fit.parameters["R_prefill"].point
        compute = sum(float(step["prompt_tokens"]) / (r_prefill * r_prefill) for step in surviving)
        bandwidth = sum(
            float(step["completion_tokens"])
            + float(step["completion_tokens"]) * float(step["prompt_tokens"])
            for step in surviving
        )
        floor = len(surviving) * fit.parameters["a"].point
        sensitivities = {
            "compute": compute,
            "bandwidth": bandwidth,
            "floor": floor,
        }
        argmax = max(sensitivities, key=sensitivities.__getitem__) if surviving else None
        curve.append(
            {
                "deadline_s": deadline,
                "n_surviving": len(surviving),
                "sensitivities": sensitivities,
                "argmax": argmax,
            }
        )
    argmaxes = [point["argmax"] for point in curve if point["argmax"] is not None]
    changed = len(set(argmaxes)) > 1
    crossovers = [
        curve[index]["deadline_s"]
        for index in range(1, len(curve))
        if curve[index]["argmax"] != curve[index - 1]["argmax"]
        and curve[index]["argmax"] is not None
        and curve[index - 1]["argmax"] is not None
    ]
    return {
        "claim_holds": changed,
        "argmax_changes": changed,
        "crossover_deadlines_s": crossovers,
        "curve": curve,
        "interpretation": (
            "The top-ranked resource changes across the deadline grid."
            if changed
            else "NULL: hardware-upgrade ranking is deadline-invariant for this workload."
        ),
    }


def _gate(
    gate_id: str,
    value: Any,
    passed: bool | None,
    criterion: str,
    *,
    note: str | None = None,
) -> dict[str, Any]:
    return {
        "id": gate_id,
        "value": value,
        "passed": passed,
        "evaluable": passed is not None,
        "criterion": criterion,
        "note": note,
    }


def analyze(
    records: Sequence[Mapping[str, Any]],
    *,
    root: Path,
    attrib_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit each target-by-quantization group and return coefficients, A/B/C/D, and rotation."""
    fit_cfg = attrib_cfg["fit"]
    gates_cfg = attrib_cfg["gates"]
    groups = sorted({(str(row["execution_target"]), str(row["quantization"])) for row in records})
    fits: dict[str, FitResult] = {}
    for group_index, (target, quantization) in enumerate(groups):
        subset = [
            row
            for row in records
            if str(row["execution_target"]) == target and str(row["quantization"]) == quantization
        ]
        fits[f"{target}/{quantization}"] = fit_interaction(
            subset,
            target=target,
            quantization=quantization,
            bootstrap_resamples=int(fit_cfg["bootstrap_resamples"]),
            bootstrap_seed=int(fit_cfg["bootstrap_seed"]) + group_index,
            confidence=float(fit_cfg["confidence"]),
        )
    primary_key = "cpu-p/int4" if "cpu-p/int4" in fits else next(iter(fits))
    fit = fits[primary_key]
    training = [
        row
        for row in records
        if not bool(row.get("held_out", False))
        and f"{row['execution_target']}/{row['quantization']}" == primary_key
    ]
    held_out = [
        row
        for row in records
        if bool(row.get("held_out", False))
        and f"{row['execution_target']}/{row['quantization']}" == primary_key
    ]
    held_predictions = [
        predict(
            fit,
            total_prompt_tokens=float(row["total_prompt_tokens"]),
            n_out=float(row["n_out"]),
        )
        for row in held_out
    ]
    held_mape = _mape([float(row["wall_s"]) for row in held_out], held_predictions)

    agent_cfg = attrib_cfg["agent_validation"]
    steps = _agent_steps(root, [str(value) for value in agent_cfg["run_ids"]])
    agent_predictions = [
        predict(
            fit,
            total_prompt_tokens=float(step["prompt_tokens"]),
            n_out=float(step["completion_tokens"]),
        )
        for step in steps
    ]
    agent_actual = [float(step["actual_wall_s"]) for step in steps]
    agent_mape = _mape(agent_actual, agent_predictions)

    actual = np.asarray([float(row["wall_s"]) for row in training])
    prompt = np.asarray([float(row["total_prompt_tokens"]) for row in training])
    n_out = np.asarray([float(row["n_out"]) for row in training])
    no_constant_matrix = np.column_stack((prompt, n_out, prompt * n_out))
    _require_full_rank(no_constant_matrix, where="no-constant comparison")
    no_constant_beta = np.linalg.lstsq(no_constant_matrix, actual, rcond=None)[0]
    no_constant_prediction = no_constant_matrix @ no_constant_beta
    full_aic = _aic(actual, np.asarray(fit.predictions), 4)
    no_constant_aic = _aic(actual, no_constant_prediction, 3)
    quadratic_matrix = np.column_stack(
        (np.ones(len(prompt)), prompt, n_out, prompt * n_out, n_out * prompt * prompt)
    )
    _require_full_rank(quadratic_matrix, where="quadratic comparison")
    quadratic_beta = np.linalg.lstsq(quadratic_matrix, actual, rcond=None)[0]
    quadratic_prediction = quadratic_matrix @ quadratic_beta
    quadratic_aic = _aic(actual, quadratic_prediction, 5)

    levels_p = len({float(row["total_prompt_tokens"]) for row in training})
    levels_n = len({float(row["n_out"]) for row in training})
    interaction_vif = fit.vif["centered_interaction"]
    a3 = {name: fit.parameters[name].excludes_zero() for name in _PHYSICAL_PARAMETERS}
    residual_context_p = _linear_slope_p(prompt.tolist(), fit.residuals)
    repeat_index = [float(row.get("repeat_idx", row.get("block_idx", 0))) for row in training]
    residual_repeat_p = _linear_slope_p(repeat_index, fit.residuals)
    block_cv = _coefficient_cv_by_block(training, fit_cfg=fit_cfg)
    stationarity = _half_stationarity(training, fit_cfg=fit_cfg)
    balanced_agreement = _balanced_split_agreement(training, fit_cfg=fit_cfg)
    synthetic_recovery = _synthetic_recovery_check(
        attrib_cfg["synthetic_recovery"],
        grid=attrib_cfg["interaction_grid"],
    )

    idle_drifts = [
        abs(float(row["idle_baseline_end"]) - float(row["idle_baseline_start"]))
        / abs(float(row["idle_baseline_start"]))
        for row in records
        if row.get("idle_baseline_start") not in (None, 0)
        and row.get("idle_baseline_end") is not None
    ]
    max_idle_drift = max(idle_drifts) if idle_drifts else None
    bandwidth = attrib_cfg.get("_platform_bandwidth_measured")
    lock_overlaps = sum(int(row.get("lock_overlap_count", 0)) for row in records)
    throttle_values = [
        float(row["throttle_fraction"])
        for row in records
        if row.get("throttle_fraction") is not None
    ]
    max_throttle = max(throttle_values) if throttle_values else None

    dashboard = {
        "A1": _gate(
            "A1",
            fit.condition_number,
            fit.condition_number < float(gates_cfg["A1_condition_number_max"]),
            f"condition number < {gates_cfg['A1_condition_number_max']}",
        ),
        "A2": _gate(
            "A2",
            fit.vif,
            all(value < float(gates_cfg["A2_vif_max"]) for value in fit.vif.values()),
            f"every VIF < {gates_cfg['A2_vif_max']}",
        ),
        "A3": _gate(
            "A3",
            {name: asdict(fit.parameters[name]) for name in _PHYSICAL_PARAMETERS},
            all(a3.values()),
            "every 95% CI excludes zero",
        ),
        "A4": _gate(
            "A4",
            fit.max_abs_parameter_correlation,
            fit.max_abs_parameter_correlation
            < float(gates_cfg["A4_max_abs_coefficient_correlation"]),
            f"max absolute coefficient correlation < "
            f"{gates_cfg['A4_max_abs_coefficient_correlation']}",
        ),
        "A5": _gate(
            "A5",
            {
                "route": "interaction",
                "P_levels": levels_p,
                "n_out_levels": levels_n,
                "interaction_vif": interaction_vif,
                "seated_spike_run_id": attrib_cfg["seating"]["mechanism_spike_run_id"],
                "seated_spike_verdict": attrib_cfg["seating"]["mechanism_spike_verdict"],
            },
            levels_p >= int(gates_cfg["A5_interaction_min_levels_each"])
            and levels_n >= int(gates_cfg["A5_interaction_min_levels_each"])
            and interaction_vif < float(gates_cfg["A5_interaction_vif_max"]),
            "P and n_out each have >=3 levels and interaction VIF <5",
        ),
        "A6": _gate(
            "A6",
            {"available_routes": ["interaction"], "seated_route": "failed_reuse"},
            None,
            "95% CIs overlap for a, R_prefill, d0, d1 across both routes",
            note=(
                "not evaluable: the pre-declared mechanism spike failed reuse, so the seated "
                "route was not implemented"
            ),
        ),
        "A6_prime": _gate(
            "A6-prime",
            balanced_agreement,
            (bool(balanced_agreement["passed"]) if balanced_agreement["evaluable"] else False),
            "balanced split-half 95% CIs overlap for all four parameters",
        ),
        "A7": _gate(
            "A7",
            synthetic_recovery,
            bool(synthetic_recovery["passed"]),
            "known coefficients recovered within CI and rank-deficient design refused",
        ),
        "B1": _gate(
            "B1",
            fit.r_squared,
            fit.r_squared >= float(gates_cfg["B1_min_r_squared"]),
            f"R^2 >= {gates_cfg['B1_min_r_squared']}",
        ),
        "B2": _gate(
            "B2",
            {"mape": held_mape, "n": len(held_out)},
            None if held_mape is None else held_mape <= float(gates_cfg["B2_max_held_out_mape"]),
            f"held-out MAPE <= {gates_cfg['B2_max_held_out_mape']}",
        ),
        "B3": _gate(
            "B3",
            {"mape": agent_mape, "n": len(steps), "run_ids": agent_cfg["run_ids"]},
            None
            if agent_mape is None
            else agent_mape <= float(gates_cfg["B3_max_agent_step_mape"]),
            f"agent-step MAPE <= {gates_cfg['B3_max_agent_step_mape']}",
        ),
        "B4": _gate(
            "B4",
            residual_context_p,
            None
            if residual_context_p is None
            else residual_context_p > float(gates_cfg["B4_min_residual_context_slope_p"]),
            f"residual-vs-P slope p > {gates_cfg['B4_min_residual_context_slope_p']}",
        ),
        "B5": _gate(
            "B5",
            residual_repeat_p,
            None
            if residual_repeat_p is None
            else residual_repeat_p > float(gates_cfg["B5_min_residual_repeat_slope_p"]),
            f"residual-vs-repeat slope p > {gates_cfg['B5_min_residual_repeat_slope_p']}",
        ),
        "B6": _gate(
            "B6",
            {
                "aic_full": full_aic,
                "aic_no_constant": no_constant_aic,
                "delta_favouring_constant": no_constant_aic - full_aic,
            },
            no_constant_aic - full_aic >= float(gates_cfg["B6_min_delta_aic_vs_no_constant"]),
            f"delta AIC >= {gates_cfg['B6_min_delta_aic_vs_no_constant']} favouring a",
        ),
        "B7": _gate(
            "B7",
            {
                "aic_linear": full_aic,
                "aic_quadratic": quadratic_aic,
                "quadratic_advantage": full_aic - quadratic_aic,
            },
            None,
            "report both; quadratic wins if advantage >=10",
        ),
        "C1": _gate(
            "C1",
            block_cv,
            None
            if any(value is None for value in block_cv.values())
            else all(
                float(value) < float(gates_cfg["C1_max_coefficient_cv_across_blocks"])
                for value in block_cv.values()
                if value is not None
            ),
            f"every coefficient CV < {gates_cfg['C1_max_coefficient_cv_across_blocks']}",
        ),
        "C2": _gate(
            "C2",
            stationarity,
            bool(stationarity["passed"]) if stationarity["evaluable"] else None,
            "first-half and second-half CIs overlap for every coefficient",
        ),
        "C3": _gate(
            "C3",
            max_idle_drift,
            None
            if max_idle_drift is None
            else max_idle_drift < float(gates_cfg["C3_max_idle_baseline_drift"]),
            f"idle baseline drift < {gates_cfg['C3_max_idle_baseline_drift']}",
        ),
        "C4": _gate(
            "C4",
            bandwidth,
            bandwidth is not None,
            "measured STREAM bandwidth and citing run_id exist",
        ),
        "C5": _gate(
            "C5",
            {"overlap_count": lock_overlaps},
            lock_overlaps == 0,
            "zero lock-window overlap",
        ),
        "C6": _gate(
            "C6",
            max_throttle,
            None
            if max_throttle is None
            else max_throttle < float(gates_cfg["C6_max_throttle_fraction"]),
            f"max throttle fraction < {gates_cfg['C6_max_throttle_fraction']}",
        ),
    }
    abc = [value for key, value in dashboard.items() if key[0] in "ABC" and key != "A6"]
    valid = all(gate["passed"] is True for gate in abc)

    median_agent_wall = statistics.median(agent_actual) if agent_actual else None
    floor_share: float | None = None
    if median_agent_wall is not None and median_agent_wall != 0.0:
        floor_share = fit.parameters["a"].point / median_agent_wall
    short_indices = [
        index
        for index, step in enumerate(steps)
        if int(step["completion_tokens"])
        <= int(attrib_cfg["materiality"]["D6_short_step_n_out_max"])
    ]
    no_constant_agent = [
        float(steps[index]["prompt_tokens"]) * no_constant_beta[0]
        + float(steps[index]["completion_tokens"]) * no_constant_beta[1]
        + float(steps[index]["prompt_tokens"])
        * float(steps[index]["completion_tokens"])
        * no_constant_beta[2]
        for index in short_indices
    ]
    router_bias = (
        statistics.fmean(
            prediction - agent_actual[index]
            for prediction, index in zip(no_constant_agent, short_indices, strict=True)
        )
        if short_indices
        else None
    )
    rotation_result = rotation(fit, steps, attrib_cfg["rotation"])
    d3: dict[str, Any]
    if "cpu-p/int4" in fits and "cpu-p/int8" in fits:
        d3 = {
            "int4_d0": asdict(fits["cpu-p/int4"].parameters["d0"]),
            "int8_d0": asdict(fits["cpu-p/int8"].parameters["d0"]),
            "ratio_int4_over_int8": (
                fits["cpu-p/int4"].parameters["d0"].point
                / fits["cpu-p/int8"].parameters["d0"].point
            ),
        }
    else:
        d3 = {"evaluable": False, "reason": "INT8 IR/run absent"}
    materiality = {
        "D1": {"floor_share_of_median_agent_step": floor_share},
        "D2": {"a_seconds": asdict(fit.parameters["a"])},
        "D3": d3,
        "D4": {
            "context_tokens_where_d1_exceeds_d0": (
                fit.parameters["d0"].point / fit.parameters["d1"].point
                if fit.parameters["d1"].point != 0.0
                else None
            )
        },
        "D5": rotation_result,
        "D6": {
            "mean_signed_no_constant_error_s_on_short_steps": router_bias,
            "n": len(short_indices),
        },
    }
    return {
        "route": "interaction_only",
        "mechanism_spike_run_id": attrib_cfg["seating"]["mechanism_spike_run_id"],
        "valid_for_conclusions": valid,
        "coefficients": {key: value.to_dict() for key, value in fits.items()},
        "validity": dashboard,
        "materiality": materiality,
        "rotation": rotation_result,
    }


def _validity_markdown(validity: Mapping[str, Mapping[str, Any]]) -> str:
    lines = ["| Gate | Value | Pass | Criterion |", "|:--|:--|:--:|:--|"]
    for gate_id, gate in validity.items():
        value = json.dumps(gate["value"], sort_keys=True)
        if len(value) > 180:
            value = value[:177] + "..."
        verdict = (
            "PASS" if gate["passed"] is True else ("FAIL" if gate["passed"] is False else "N/E")
        )
        lines.append(f"| {gate_id} | `{value}` | {verdict} | {gate['criterion']} |")
    return "\n".join(lines) + "\n"


def _write_outputs(result: Mapping[str, Any], *, root: Path, cfg: Mapping[str, Any]) -> None:
    outputs = cfg["outputs"]
    out_dir = root / outputs["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / outputs["coefficients"]).write_text(
        json.dumps(result["coefficients"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / outputs["validity"]).write_text(
        json.dumps(result["validity"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / outputs["validity_table"]).write_text(
        _validity_markdown(result["validity"]), encoding="utf-8"
    )
    (out_dir / outputs["rotation"]).write_text(
        json.dumps(result["rotation"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="sealed E-ATTRIB interaction sweep run")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)
    root = args.root.resolve()
    run_dir = open_run_dir(args.run_id, repo_root=root)
    if not run_dir.is_sealed() or not verify_sealed(run_dir):
        raise SeamError(f"interaction sweep run {args.run_id} is not sealed and verified")
    summary = json.loads((run_dir.path / "summary.json").read_text(encoding="utf-8"))
    records = summary["records"]
    with (root / "configs" / "attrib.yaml").open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    platform = resolve_config([root / "configs" / "platforms" / "aipc-c1.yaml"], repo_root=root)
    cfg["_platform_bandwidth_measured"] = (
        None
        if platform.get("memory.bandwidth_gbps_measured") is None
        else {
            "gbps": platform.get("memory.bandwidth_gbps_measured"),
            "run_id": platform.get("memory.bandwidth_measured_by_run_id"),
        }
    )
    result = analyze(records, root=root, attrib_cfg=cfg)
    result["source_run_id"] = args.run_id
    _write_outputs(result, root=root, cfg=cfg)
    print(json.dumps({"run_id": args.run_id, "valid": result["valid_for_conclusions"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
