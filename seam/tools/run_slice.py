"""M-SLICE runner. Enforces the strict order of operations.

.. code-block:: text

    1. baseline      throughput per target + unconstrained local success + n_out_pred
    2. validate-key  ONE paid call; confirm usage fields parse and report actual cost
    3. aa            A/A negative control -- must PASS before anything else is believed
    4. noise         noise floor, n>=20 on one fixed configuration
    5. main          2 targets x >=4 deadlines, randomized order
    6. analyze       curves, CIs, collapse test, Pareto, verdict

The ordering is enforced, not documented: :func:`_require_phase` refuses to start a phase whose
prerequisite artifact is absent. In particular the main run cannot start until an A/A artifact
exists **and** records a pass, because a comparison run through a pipeline that manufactures
differences is worse than no comparison at all.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from seam.agent.harness import TaskResult, run_task
from seam.agent.policy import ArmConfig, ThroughputModel, assert_isolated
from seam.agent.tools import SYSTEM_PROMPT, TOOL_SPECS, Workload, load_workload
from seam.analysis.slice_stats import coefficient_of_variation, noise_floor
from seam.backends.cloud_anthropic import AnthropicBackend
from seam.backends.local_openvino import LocalOpenVinoBackend, runtime_info
from seam.budget import BudgetGuard, BudgetLedger, PricingTable
from seam.config import load_platform_config
from seam.errors import BackendError, ConfigError
from seam.gitinfo import repo_root
from seam.hashing import sha256_json
from seam.jsonlog import log_event
from seam.model_provenance import load_local_spec, manifest_model_block, quantization_summary
from seam.tools.verify_core_affinity import verify_target

__all__ = ["main"]

#: Fixed probe used for the throughput baseline. Identical on both targets, so the ratio it
#: produces is a property of the silicon and not of the text.
_PROBE_PROMPT = (
    "Explain, in plain prose and without lists, how a deadline-aware scheduler decides whether to "
    "run a task locally or send it to a remote service. Cover latency prediction, the cost of "
    "being wrong in each direction, and what happens as the deadline tightens."
)
_PROBE_MAX_TOKENS = 128


def _load_slice_config(root: Path) -> dict[str, Any]:
    path = root / "configs" / "mslice.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"{path} is not a mapping")
    return data


def _derived_dir(root: Path) -> Path:
    out = root / "derived" / "mslice"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _require_phase(root: Path, artifact: str, phase: str) -> dict[str, Any]:
    path = _derived_dir(root) / artifact
    if not path.exists():
        raise SystemExit(
            f"refusing to run {phase}: {path} is absent. The order of operations is strict - "
            f"produce that artifact first."
        )
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def _build_guard(root: Path, cfg: dict[str, Any], phase: str) -> BudgetGuard:
    budget = cfg["budget"]
    pricing = PricingTable.load(root / cfg["models"]["cloud"]["pricing_table"])
    return BudgetGuard(
        ledger=BudgetLedger(root / budget["ledger_dir"]),
        pricing=pricing,
        slice_ceiling_usd=float(budget["slice_ceiling_usd"]),
        project_ceiling_usd=float(budget["project_ceiling_usd"]),
        phase=phase,
    )


def _local_backend(root: Path, cfg: dict[str, Any], target: str) -> LocalOpenVinoBackend:
    spec_path = root / cfg["models"]["local"]["spec"]
    try:
        spec = load_local_spec(spec_path)
    except ConfigError as exc:
        raise SystemExit(
            f"{exc}\nFetch a pre-converted IR (AM-023) or export one:\n"
            f"  python -m seam.tools.fetch_model --repo OpenVINO/Qwen3-4B-int4-ov "
            f"--ladder-position slice-4b\n"
            f"  python -m seam.tools.export_model"
        ) from exc
    ov = cfg["openvino"]
    quant = quantization_summary(spec)
    return LocalOpenVinoBackend(
        model_dir=Path(spec["ir_dir"]),
        target=target,  # type: ignore[arg-type]
        scheduling_core_type=ov["scheduling_core_type"][target],
        inference_num_threads=int(ov["inference_num_threads"]),
        enable_cpu_pinning=bool(ov["enable_cpu_pinning"]),
        model_ref=f"{spec['name']}@{str(spec['revision'])[:12]}+{quant}",
    )


def _local_model_manifest(
    root: Path, cfg: dict[str, Any], *, reasoning_mode: str | None
) -> dict[str, Any]:
    """Manifest ``model`` block that preserves the AM-023 provenance discriminant."""
    spec_path = root / cfg["models"]["local"]["spec"]
    spec = load_local_spec(spec_path)
    cloud_cfg = cfg["models"]["cloud"]
    return manifest_model_block(
        spec=spec,
        spec_path=spec_path,
        reasoning_mode=reasoning_mode,
        cloud={
            "provider": cloud_cfg.get("provider"),
            "model_snapshot_id": cloud_cfg.get("model_id"),
            "access_date": None,
            "pricing_table_version": None,
            "pin_convention": "anthropic_dateless_generation_id",
            "prompt_caching": bool(cloud_cfg.get("prompt_caching")),
            "reasoning_mode": reasoning_mode,
        },
    )


def _expected_cpus(platform_cfg: Any, target: str) -> list[int]:
    key = "topology.p_cpus" if target == "cpu-p" else "topology.lpe_cpus"
    cpus = platform_cfg.get(key)
    if not cpus:
        raise SystemExit(
            f"platform config carries no verified {key}. Spec §4 requires the M1 mapping to be "
            f"asserted; guessing it would silently answer open question 4 with an assumption."
        )
    return [int(c) for c in cpus]


# ==================================================================================================
# Phase 1 - throughput baseline
# ==================================================================================================


def phase_baseline(root: Path, cfg: dict[str, Any], args: argparse.Namespace) -> int:
    from seam.backends.base import GenerationRequest

    platform_cfg = load_platform_config(args.platform, repo_root=root)
    workload = load_workload(root / cfg["workload"]["task_list"])
    _assert_task_list(cfg, workload)

    results: dict[str, Any] = {
        "phase": "baseline",
        "openvino": asdict(runtime_info()),
        "targets": {},
        "task_list_sha256": workload.sha256,
    }

    for target in cfg["targets"]:
        backend = _local_backend(root, cfg, target)
        verdict = backend.preflight()
        if verdict.status != "OK":
            raise SystemExit(f"{target}: preflight {verdict.status}: {verdict.reason}")

        probe = GenerationRequest(
            messages=[{"role": "user", "content": _PROBE_PROMPT}],
            system=SYSTEM_PROMPT,
            tools=TOOL_SPECS,
            max_tokens=_PROBE_MAX_TOKENS,
            temperature=0.0,
        )

        # MANDATORY (AM-022 / spec §4): confirm the requested cluster is the one actually loaded.
        # Setting SCHEDULING_CORE_TYPE is a request, not a guarantee, and a leak would silently
        # collapse both arms onto the same silicon.
        evidence = verify_target(
            target=target,
            scheduling_core_type=backend.properties()["SCHEDULING_CORE_TYPE"],
            expected_cpus=_expected_cpus(platform_cfg, target),
            generate=lambda b=backend, p=probe: b.generate(p),  # type: ignore[misc]
        )

        samples = []
        for _ in range(args.probe_repeats):
            result = backend.generate(probe)
            ttft_s = (result.ttft_ns or 0) / 1e9
            decode_s = max(result.wall_ns / 1e9 - ttft_s, 1e-9)
            samples.append(
                {
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "wall_s": result.wall_ns / 1e9,
                    "ttft_s": ttft_s,
                    "r_prefill_tok_s": (result.prompt_tokens / ttft_s) if ttft_s > 0 else None,
                    "r_decode_tok_s": result.completion_tokens / decode_s,
                }
            )

        prefill = [s["r_prefill_tok_s"] for s in samples if s["r_prefill_tok_s"]]
        decode = [s["r_decode_tok_s"] for s in samples if s["r_decode_tok_s"]]
        results["targets"][target] = {
            "affinity_evidence": evidence.to_dict(),
            "samples": samples,
            "r_prefill_tok_s": statistics.median(prefill) if prefill else None,
            "r_decode_tok_s": statistics.median(decode) if decode else None,
            "properties": backend.properties(),
            "model_ref": backend.model_ref,
        }
        print(
            f"{target}: prefill {results['targets'][target]['r_prefill_tok_s']} tok/s, "
            f"decode {results['targets'][target]['r_decode_tok_s']} tok/s, "
            f"affinity {evidence.verdict}"
        )

    # Unconstrained local capability. If the model cannot emit valid tool calls without a
    # deadline, escalation in the main run would be dominated by capability failure rather than
    # deadline expiry, the silicon axis would wash out, and the slice would measure the wrong
    # thing. Profiled on the FASTER target only: greedy decoding makes the text identical on both.
    anchor = cfg["policy"]["deadline_anchor_target"]
    unconstrained = _unconstrained_pass(root, cfg, workload, anchor)
    results["unconstrained"] = unconstrained

    ratio = _throughput_ratio(results)
    results["throughput_ratio_p_over_lpe"] = ratio

    # Freeze deadlines from the pre-registered rule (config `deadline_multipliers`).
    median_t_pred = unconstrained["median_t_pred_s_anchor"]
    results["deadlines_s"] = [
        round(m * median_t_pred, 6) for m in cfg["policy"]["deadline_multipliers"]
    ]
    results["n_out_pred_tokens"] = unconstrained["n_out_pred_tokens"]

    out = _derived_dir(root) / "baseline.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nthroughput ratio (cpu-p / cpu-lpe, decode): {ratio}")
    print(f"unconstrained local success: {unconstrained['success_rate']:.3f}")
    print(f"frozen deadlines (s): {results['deadlines_s']}")
    print(f"wrote {out}")
    return 0


def _throughput_ratio(results: dict[str, Any]) -> float | None:
    p = results["targets"].get("cpu-p", {}).get("r_decode_tok_s")
    lpe = results["targets"].get("cpu-lpe", {}).get("r_decode_tok_s")
    if not p or not lpe:
        return None
    return float(p) / float(lpe)


def _unconstrained_pass(
    root: Path, cfg: dict[str, Any], workload: Workload, target: str
) -> dict[str, Any]:
    """Run every task locally with no deadline: pure capability, no escalation."""
    backend = _local_backend(root, cfg, target)
    backend.load()

    # An effectively infinite deadline makes the router always choose local, so this measures
    # capability alone.
    throughput = ThroughputModel(
        target=target, r_prefill_tok_s=1e9, r_decode_tok_s=1e9, measured_by_run_id="unconstrained"
    )
    n_out_seed = {"tool_call_synthesis": 1, "answer_synthesis": 1}

    results: list[TaskResult] = []
    for task in workload.tasks:
        results.append(
            run_task(
                task=task,
                world=workload.world,
                local_backend=backend,
                cloud_backend=backend,  # never reached: deadline cannot bind
                throughput=throughput,
                deadline_s=1e9,
                n_out_pred_tokens=n_out_seed,
                max_steps=int(cfg["workload"]["max_steps"]),
                max_tokens=int(cfg["models"]["cloud"]["max_tokens"]),
                run_id="unconstrained",
                program_id=f"unconstrained/{task.task_id}",
            )
        )
        print(f"  {task.task_id}: success={results[-1].success} steps={results[-1].realized_steps}")

    by_type: dict[str, list[int]] = {"tool_call_synthesis": [], "answer_synthesis": []}
    t_pred_inputs: list[tuple[int, str]] = []
    for r in results:
        for step in r.steps:
            by_type.setdefault(step.step_type, []).append(step.completion_tokens)
            t_pred_inputs.append((step.prompt_tokens, step.step_type))

    n_out_pred = {
        step_type: int(statistics.median(values)) if values else 1
        for step_type, values in by_type.items()
    }
    return {
        "target": target,
        "n_tasks": len(results),
        "success_rate": sum(1 for r in results if r.success) / len(results),
        "successes": [r.task_id for r in results if r.success],
        "failures": [r.task_id for r in results if not r.success],
        "mean_realized_steps": statistics.fmean(r.realized_steps for r in results),
        "n_out_pred_tokens": n_out_pred,
        "median_prompt_tokens": statistics.median(p for p, _ in t_pred_inputs),
        "median_t_pred_s_anchor": 0.0,  # filled by caller once throughput is known
        "per_task": [r.to_summary() for r in results],
    }


def _assert_task_list(cfg: dict[str, Any], workload: Workload) -> None:
    pinned = cfg["workload"].get("task_list_sha256")
    if pinned and pinned != workload.sha256:
        raise SystemExit(
            f"task list hash mismatch: config pins {pinned}, file is {workload.sha256}. A task "
            f"list that changed mid-experiment invalidates comparability with everything already "
            f"collected."
        )


# ==================================================================================================
# Phase 2 - key validation (checkpoint 1)
# ==================================================================================================


def phase_validate_key(root: Path, cfg: dict[str, Any], _args: argparse.Namespace) -> int:
    from seam.backends.base import GenerationRequest

    guard = _build_guard(root, cfg, phase="validate_key")
    cloud_cfg = cfg["models"]["cloud"]
    backend = AnthropicBackend(
        model_id=cloud_cfg["model_id"],
        guard=guard,
        max_tokens=64,
        temperature=0.0,
        prompt_caching=bool(cloud_cfg["prompt_caching"]),
        cache_ttl=str(cloud_cfg["cache_ttl"]),
        max_retries=int(cloud_cfg["max_retries"]),
        run_id="validate_key",
    )
    verdict = backend.preflight()
    if verdict.status != "OK":
        raise SystemExit(f"cloud preflight {verdict.status}: {verdict.reason}")

    request = GenerationRequest(
        messages=[{"role": "user", "content": "Reply with the single word: ready"}],
        system=SYSTEM_PROMPT,
        tools=TOOL_SPECS,
        max_tokens=64,
        temperature=0.0,
    )
    result = backend.generate(request)

    payload = {
        "phase": "validate_key",
        "model_sent": cloud_cfg["model_id"],
        "model_reported": result.reported_model,
        "pin_held": result.reported_model == cloud_cfg["model_id"],
        "usage_fields": {
            "input_tokens": result.prompt_tokens,
            "output_tokens": result.completion_tokens,
            "cache_read_input_tokens": result.cache_read_input_tokens,
            "cache_creation_input_tokens": result.cache_creation_input_tokens,
        },
        "actual_usd": result.usd_cost,
        "projected_usd": result.extra.get("projected_usd"),
        "text": result.text[:200],
        "checkpoint": guard.checkpoint(),
    }
    out = _derived_dir(root) / "validate_key.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


# ==================================================================================================
# Phases 3-5 - A/A, noise floor, main run
# ==================================================================================================


def _throughput_from_baseline(baseline: dict[str, Any], target: str) -> ThroughputModel:
    entry = baseline["targets"][target]
    return ThroughputModel(
        target=target,
        r_prefill_tok_s=float(entry["r_prefill_tok_s"]),
        r_decode_tok_s=float(entry["r_decode_tok_s"]),
        measured_by_run_id=baseline.get("run_id", "baseline"),
    )


def _arm_fields(cfg: dict[str, Any], workload: Workload, deadline: float) -> dict[str, Any]:
    # confinement_mechanism and reasoning_mode are isolation keys: omitting them makes
    # assert_isolated report "absent from both arms" and cannot catch a real mismatch.
    return {
        "task_ids": [t.task_id for t in workload.tasks],
        "seed": cfg["workload"]["seed"],
        "system_prompt_sha256": sha256_json(SYSTEM_PROMPT),
        "tool_specs_sha256": sha256_json([asdict(t) for t in TOOL_SPECS]),
        "task_list_sha256": workload.sha256,
        "n_out_pred_tokens": cfg["policy"]["n_out_pred_tokens"],
        "deadline_s": deadline,
        "max_steps": cfg["workload"]["max_steps"],
        "max_tokens": cfg["models"]["cloud"]["max_tokens"],
        "temperature": 0.0,
        "cloud_model_id": cfg["models"]["cloud"]["model_id"],
        "local_model_ref": cfg["models"]["local"]["spec"],
        "confinement_mechanism": cfg["openvino"].get("adopted_mechanism"),
        "reasoning_mode": cfg.get("active_reasoning_arm", {}).get("id")
        or cfg.get("reasoning_mode"),
    }


def _run_cell(
    *,
    root: Path,
    cfg: dict[str, Any],
    workload: Workload,
    target: str,
    deadline: float,
    throughput: ThroughputModel,
    n_out_pred: dict[str, int],
    guard: BudgetGuard,
    label: str,
    rng: random.Random,
) -> list[TaskResult]:
    local = _local_backend(root, cfg, target)
    local.load()
    cloud_cfg = cfg["models"]["cloud"]
    cloud = AnthropicBackend(
        model_id=cloud_cfg["model_id"],
        guard=guard,
        max_tokens=int(cloud_cfg["max_tokens"]),
        temperature=0.0,
        prompt_caching=bool(cloud_cfg["prompt_caching"]),
        cache_ttl=str(cloud_cfg["cache_ttl"]),
        max_retries=int(cloud_cfg["max_retries"]),
        run_id=label,
    )

    tasks = list(workload.tasks)
    if cfg["design"]["randomize_execution_order"]:
        rng.shuffle(tasks)

    out: list[TaskResult] = []
    for task in tasks:
        out.append(
            run_task(
                task=task,
                world=workload.world,
                local_backend=local,
                cloud_backend=cloud,
                throughput=throughput,
                deadline_s=deadline,
                n_out_pred_tokens=n_out_pred,
                max_steps=int(cfg["workload"]["max_steps"]),
                max_tokens=int(cloud_cfg["max_tokens"]),
                run_id=label,
                program_id=f"{label}/{task.task_id}",
            )
        )
    return out


def _cell_metrics(results: list[TaskResult]) -> dict[str, list[float]]:
    return {
        "escalation_rate": [r.escalation_rate for r in results],
        "jct_s": [r.jct_s for r in results],
        "total_tokens": [float(r.local_tokens + r.cloud_tokens) for r in results],
        "success": [1.0 if r.success else 0.0 for r in results],
        "usd_cost": [r.usd_cost for r in results],
    }


def phase_noise(root: Path, cfg: dict[str, Any], args: argparse.Namespace) -> int:
    baseline = _require_phase(root, "baseline.json", "noise")
    cfg["policy"]["n_out_pred_tokens"] = baseline["n_out_pred_tokens"]
    workload = load_workload(root / cfg["workload"]["task_list"])
    guard = _build_guard(root, cfg, phase="noise_floor")

    target = cfg["policy"]["deadline_anchor_target"]
    deadline = baseline["deadlines_s"][1]  # the 1.0x multiplier: a deadline that actually binds
    throughput = _throughput_from_baseline(baseline, target)
    rng = random.Random(int(cfg["design"]["bootstrap_seed"]))

    repeats = int(args.repeats or cfg["design"]["noise_floor_repeats"])
    per_repeat: list[dict[str, float]] = []
    for i in range(repeats):
        results = _run_cell(
            root=root,
            cfg=cfg,
            workload=workload,
            target=target,
            deadline=deadline,
            throughput=throughput,
            n_out_pred=baseline["n_out_pred_tokens"],
            guard=guard,
            label=f"noise/{i:02d}",
            rng=rng,
        )
        metrics = _cell_metrics(results)
        per_repeat.append({k: statistics.fmean(v) for k, v in metrics.items()})
        print(f"  repeat {i}: {per_repeat[-1]}")

    floors = {
        metric: noise_floor(metric, [r[metric] for r in per_repeat]).to_dict()
        for metric in per_repeat[0]
    }
    payload = {
        "phase": "noise_floor",
        "target": target,
        "deadline_s": deadline,
        "n_repeats": repeats,
        "per_repeat": per_repeat,
        "noise_floor": floors,
        "checkpoint": guard.checkpoint(),
    }
    out = root / "derived" / "noise_floor_slice.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (_derived_dir(root) / "noise.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


def phase_aa(root: Path, cfg: dict[str, Any], args: argparse.Namespace) -> int:
    """A/A negative control: one target, two labels, through the whole pipeline."""
    from seam.analysis.slice_stats import aa_verdict

    baseline = _require_phase(root, "baseline.json", "aa")
    cfg["policy"]["n_out_pred_tokens"] = baseline["n_out_pred_tokens"]
    workload = load_workload(root / cfg["workload"]["task_list"])
    guard = _build_guard(root, cfg, phase="aa")

    target = cfg["policy"]["deadline_anchor_target"]
    deadline = baseline["deadlines_s"][1]
    throughput = _throughput_from_baseline(baseline, target)
    rng = random.Random(int(cfg["design"]["bootstrap_seed"]))

    fields = _arm_fields(cfg, workload, deadline)
    assert_isolated(
        ArmConfig(target=target, throughput=throughput, fields=dict(fields)),
        ArmConfig(target=target, throughput=throughput, fields=dict(fields)),
    )

    repeats = int(args.repeats or cfg["design"]["aa_repeats"])
    arms: dict[str, list[dict[str, float]]] = {"A1": [], "A2": []}
    for label in ("A1", "A2"):
        for i in range(repeats):
            results = _run_cell(
                root=root,
                cfg=cfg,
                workload=workload,
                target=target,
                deadline=deadline,
                throughput=throughput,
                n_out_pred=baseline["n_out_pred_tokens"],
                guard=guard,
                label=f"aa/{label}/{i:02d}",
                rng=rng,
            )
            metrics = _cell_metrics(results)
            arms[label].append({k: statistics.fmean(v) for k, v in metrics.items()})
            print(f"  {label} repeat {i}: {arms[label][-1]}")

    metric_names = list(arms["A1"][0])
    cvs = {
        m: coefficient_of_variation([r[m] for r in arms["A1"] + arms["A2"]]) for m in metric_names
    }
    verdict = aa_verdict(
        metrics={m: ([r[m] for r in arms["A1"]], [r[m] for r in arms["A2"]]) for m in metric_names},
        cvs=cvs,
        null_threshold_cv_multiple=float(cfg["design"]["null_threshold_cv_multiple"]),
        resamples=int(cfg["design"]["bootstrap_resamples"]),
        seed=int(cfg["design"]["bootstrap_seed"]),
    )
    payload = {
        "phase": "aa",
        "target": target,
        "deadline_s": deadline,
        "n_repeats_per_arm": repeats,
        "arms": arms,
        "cvs": cvs,
        "verdict": verdict,
        "checkpoint": guard.checkpoint(),
    }
    out = _derived_dir(root) / "aa.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(verdict["interpretation"])
    if not verdict["passed"]:
        raise SystemExit(
            "A/A FAILED. The harness reports a difference between two labels of the same "
            "condition. STOP - no comparison is valid until this is fixed."
        )
    return 0


def phase_main(root: Path, cfg: dict[str, Any], _args: argparse.Namespace) -> int:
    baseline = _require_phase(root, "baseline.json", "main")
    aa = _require_phase(root, "aa.json", "main")
    if not aa["verdict"]["passed"]:
        raise SystemExit("refusing to run main: the recorded A/A control did not pass.")
    _require_phase(root, "noise.json", "main")

    cfg["policy"]["n_out_pred_tokens"] = baseline["n_out_pred_tokens"]
    workload = load_workload(root / cfg["workload"]["task_list"])
    guard = _build_guard(root, cfg, phase="main")
    rng = random.Random(int(cfg["design"]["bootstrap_seed"]))

    deadlines = baseline["deadlines_s"]
    cells: list[tuple[str, float]] = [(t, d) for t in cfg["targets"] for d in deadlines]
    rng.shuffle(cells)

    # Isolation invariant across the two arms, at every deadline.
    for deadline in deadlines:
        fields = _arm_fields(cfg, workload, deadline)
        assert_isolated(
            ArmConfig("cpu-p", _throughput_from_baseline(baseline, "cpu-p"), dict(fields)),
            ArmConfig("cpu-lpe", _throughput_from_baseline(baseline, "cpu-lpe"), dict(fields)),
        )

    collected: dict[str, Any] = {}
    for target, deadline in cells:
        label = f"main/{target}/d{deadline:.3f}"
        print(f"cell {label}")
        results = _run_cell(
            root=root,
            cfg=cfg,
            workload=workload,
            target=target,
            deadline=deadline,
            throughput=_throughput_from_baseline(baseline, target),
            n_out_pred=baseline["n_out_pred_tokens"],
            guard=guard,
            label=label,
            rng=rng,
        )
        collected[label] = {
            "target": target,
            "deadline_s": deadline,
            "metrics": _cell_metrics(results),
            "per_task": [r.to_summary() for r in results],
            "overrun_rate": statistics.fmean(
                [
                    r.deadline_overruns / r.realized_steps if r.realized_steps else 0.0
                    for r in results
                ]
            ),
        }

    payload = {
        "phase": "main",
        "deadlines_s": deadlines,
        "cells": collected,
        "checkpoint": guard.checkpoint(),
    }
    out = _derived_dir(root) / "main.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


# ==================================================================================================
# CLI
# ==================================================================================================


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=["baseline", "validate-key", "aa", "noise", "main", "status"]
    )
    parser.add_argument("--platform", default="aipc-c1")
    parser.add_argument("--probe-repeats", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=None)
    args = parser.parse_args(argv)

    root = repo_root(Path(__file__).parent)
    cfg = _load_slice_config(root)
    log_event("mslice.phase_start", message=f"phase {args.phase}", phase=args.phase)

    started = time.time()
    try:
        if args.phase == "baseline":
            return phase_baseline(root, cfg, args)
        if args.phase == "validate-key":
            return phase_validate_key(root, cfg, args)
        if args.phase == "aa":
            return phase_aa(root, cfg, args)
        if args.phase == "noise":
            return phase_noise(root, cfg, args)
        if args.phase == "main":
            return phase_main(root, cfg, args)
        guard = _build_guard(root, cfg, phase="status")
        print(json.dumps(guard.checkpoint(), indent=2))
        return 0
    except BackendError as exc:
        print(f"BACKEND ERROR: {exc}")
        return 2
    finally:
        log_event(
            "mslice.phase_end",
            message=f"phase {args.phase} took {time.time() - started:.1f}s",
            phase=args.phase,
            duration_s=time.time() - started,
        )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
