"""E-ATTRIB measurement driver.

The first executable gate is the chat-mode mechanism spike requested by the 2026-08-03 dispatch.
It proves (or falsifies) that a context seated through ``LLMPipeline.start_chat()`` is
computationally equivalent to a stateless cold ingest of the same rendered conversation.

No fitted result is emitted by this module.  Fitting belongs to ``seam.analysis.attrib_fit`` and
is implemented only after the mechanism fork has a sealed result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from seam.backends.local_openvino import (
    _extract_metrics,
    _make_ttft_streamer,
    _text_from_generate_result,
    resolve_ttft_ns,
    runtime_info,
)
from seam.config import ResolvedConfig, resolve_config
from seam.errors import ConfigError, SeamError
from seam.kvmath import KV_FORMULA, load_kv_geometry
from seam.locks import exclusive
from seam.manifest import emit
from seam.model_provenance import load_local_spec, manifest_model_block
from seam.powerstate import (
    PowerState,
    assert_profile,
    capture_battery_status_wmi,
    capture_power_state,
    is_charging_complete,
    manifest_power_state,
    raise_if_profile_mismatch,
)
from seam.rawstore import open_run_dir, verify_sealed
from seam.telemetry.frequency import FrequencySampler

_ROOT = Path(__file__).resolve().parents[2]
_ATTRIB_CONFIG = Path("configs/attrib.yaml")
_PLATFORM_CONFIG = Path("configs/platforms/aipc-c1.yaml")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _load_configs(root: Path) -> tuple[ResolvedConfig, dict[str, Any]]:
    """Load the manifest-resolved config and the experiment subtree."""
    platform_path = root / _PLATFORM_CONFIG
    attrib_path = root / _ATTRIB_CONFIG
    resolved = resolve_config([platform_path, attrib_path], repo_root=root)
    with attrib_path.open("r", encoding="utf-8") as stream:
        attrib = yaml.safe_load(stream)
    if not isinstance(attrib, dict):
        raise ConfigError(f"{attrib_path} is not a mapping")
    return resolved, attrib


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer.encode(text)
    data = np.asarray(encoded.input_ids.data, dtype=np.int64).reshape(-1)
    return [int(value) for value in data]


def _text_near_tokens(tokenizer: Any, unit: str, target: int) -> str:
    """Repeat ``unit`` to the nearest token count at or above ``target``."""
    if target <= 0:
        raise ConfigError(f"token target must be positive, got {target}")
    one = len(_token_ids(tokenizer, unit))
    if one <= 0:
        raise ConfigError("configured filler unit tokenizes to an empty sequence")
    repeats = max(1, math.ceil(target / one))
    text = unit * repeats
    while len(_token_ids(tokenizer, text)) < target:
        repeats += 1
        text = unit * repeats
    return text


def _generation_config(ov_genai: Any, *, n_out: int, seed: int, templated: bool) -> Any:
    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = n_out
    cfg.do_sample = False
    cfg.temperature = 0.0
    cfg.rng_seed = seed
    cfg.ignore_eos = True
    cfg.apply_chat_template = templated
    return cfg


def _timed_generate(pipe: Any, ov_genai: Any, prompt: str, cfg: Any) -> dict[str, Any]:
    """Generate through the project's established dual-source TTFT instrument.

    Timing is deliberately not reimplemented here.  The streamer, metrics extraction, and TTFT
    resolution are the same helpers used by ``LocalOpenVinoBackend.generate``.
    """
    streamer = _make_ttft_streamer(ov_genai)
    t0_ns = time.perf_counter_ns()
    streamer.t0_ns = t0_ns
    result = pipe.generate([prompt], cfg, streamer)
    wall_ns = time.perf_counter_ns() - t0_ns
    metrics_ttft_ns, prompt_tokens, completion_tokens = _extract_metrics(result.perf_metrics)
    ttft_ns, ttft_source = resolve_ttft_ns(metrics_ttft_ns, streamer.ttft_ns)
    if ttft_ns is None:
        raise SeamError(
            "mechanism spike got no TTFT from either perf_metrics or the established streamer"
        )
    text = _text_from_generate_result(result)
    return {
        "text": text,
        "output_utf8_hex": text.encode("utf-8").hex(),
        "wall_ns": wall_ns,
        "ttft_ns": ttft_ns,
        "ttft_source": ttft_source,
        "ttft_ns_perf_metrics": metrics_ttft_ns,
        "ttft_ns_streamer": streamer.ttft_ns,
        "prompt_tokens_reported": prompt_tokens,
        "completion_tokens_reported": completion_tokens,
    }


def _chat_first_turn(pipe: Any, ov_genai: Any, context: str, cfg: Any) -> dict[str, Any]:
    pipe.start_chat()
    try:
        return _timed_generate(pipe, ov_genai, context, cfg)
    except BaseException:
        pipe.finish_chat()
        raise


def _finish_chat(pipe: Any) -> None:
    pipe.finish_chat()


def _render_history(tokenizer: Any, context: str, response: str, new_prompt: str) -> str:
    history = [
        {"role": "user", "content": context},
        {"role": "assistant", "content": response},
        {"role": "user", "content": new_prompt},
    ]
    return str(tokenizer.apply_chat_template(history, True))


def _token_precheck(
    tokenizer: Any,
    *,
    context: str,
    response: str,
    new_prompt: str,
    cold_rendered: str,
) -> dict[str, Any]:
    """Prove the conceptual seated history and cold input render to the same token IDs."""
    seated_rendered = _render_history(tokenizer, context, response, new_prompt)
    seated_ids = _token_ids(tokenizer, seated_rendered)
    cold_ids = _token_ids(tokenizer, cold_rendered)
    first_mismatch = next(
        (
            idx
            for idx, (seated, cold) in enumerate(zip(seated_ids, cold_ids, strict=False))
            if seated != cold
        ),
        None,
    )
    identical = seated_ids == cold_ids
    return {
        "passed": identical,
        "identical": identical,
        "n_ids_seated": len(seated_ids),
        "n_ids_cold": len(cold_ids),
        "first_mismatch_index": first_mismatch,
        "seated_ids_sha256": hashlib.sha256(
            json.dumps(seated_ids, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
        "cold_ids_sha256": hashlib.sha256(
            json.dumps(cold_ids, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
        # The full sequences are retained in raw so the equality claim is independently auditable,
        # rather than relying only on hashes computed by the code under test.
        "seated_ids": seated_ids,
        "cold_ids": cold_ids,
    }


def _wait_for_machine_lock(lock_resource: Path, *, timeout_s: float, poll_s: float) -> None:
    """Wait for the other track's measurement; never measure around a held lock."""
    lock_path = Path(str(lock_resource) + ".lock")
    started = time.monotonic()
    while lock_path.exists():
        if time.monotonic() - started >= timeout_s:
            raise SeamError(
                f"timed out after {timeout_s}s waiting for machine lock {lock_path}; "
                "no measurement was taken"
            )
        time.sleep(poll_s)


def _bootstrap_median_ci(
    values: Sequence[float], *, resamples: int, confidence: float, seed: int
) -> dict[str, Any]:
    data = [float(value) for value in values]
    if not data:
        raise SeamError("cannot bootstrap an empty ratio series")
    rng = random.Random(seed)
    n = len(data)
    draws = sorted(
        statistics.median(data[rng.randrange(n)] for _ in range(n)) for _ in range(resamples)
    )
    alpha = (1.0 - confidence) / 2.0
    lo_idx = max(0, min(resamples - 1, math.floor(alpha * resamples)))
    hi_idx = max(0, min(resamples - 1, math.ceil((1.0 - alpha) * resamples) - 1))
    return {
        "point": statistics.median(data),
        "lo": draws[lo_idx],
        "hi": draws[hi_idx],
        "confidence": confidence,
        "resamples": resamples,
        "statistic": "median",
        "n": n,
    }


def _quiesce(resolved: ResolvedConfig, attrib: dict[str, Any]) -> tuple[PowerState, dict[str, Any]]:
    power = capture_power_state()
    battery = capture_battery_status_wmi()
    profile_cfg = resolved.get("power.profiles.ac-pinned") or {}
    charging_complete, charging_reason = is_charging_complete(
        power,
        battery,
        charge_rate_max_mw=profile_cfg.get("charge_rate_max_mw"),
        charging_complete_soc_pct=profile_cfg.get("charging_complete_soc_pct"),
    )
    assertion = assert_profile(
        str(attrib["quiesce"]["measurement_class"]),
        power,
        power_cfg=resolved.get("power") or {},
    )
    raise_if_profile_mismatch(assertion)
    deviations: list[str] = []
    if power.on_battery is not False:
        deviations.append(f"AC not connected (on_battery={power.on_battery})")
    if not charging_complete or power.on_battery is not False:
        deviations.append(f"charging not complete on AC: {charging_reason}")
    if power.battery_saver:
        deviations.append("battery saver is engaged")
    if deviations:
        raise SeamError("mechanism spike quiescence refusal: " + "; ".join(deviations))
    return power, {
        "captured_utc": _utc_now(),
        "profile": asdict(assertion),
        "charging_complete": charging_complete,
        "charging_complete_reason": charging_reason,
        "battery_status": asdict(battery),
    }


def _throttle_summary(
    samples: Sequence[Any], *, cpu_ids: Sequence[int], floor_pct: float
) -> dict[str, Any]:
    eligible = 0
    throttled = 0
    for sample in samples:
        values = [
            sample.pct_of_max_per_cpu[cpu]
            for cpu in cpu_ids
            if cpu < len(sample.pct_of_max_per_cpu) and sample.pct_of_max_per_cpu[cpu] is not None
        ]
        if not values:
            continue
        eligible += 1
        if min(float(value) for value in values) < floor_pct:
            throttled += 1
    return {
        "cpu_ids": list(cpu_ids),
        "pct_of_max_floor": floor_pct,
        "n_samples": len(samples),
        "n_eligible": eligible,
        "n_throttled": throttled,
        "throttle_fraction": throttled / eligible if eligible else None,
    }


def run_mechanism_spike(*, root: Path, allow_dirty: bool = False) -> dict[str, Any]:
    """Run, seal, and return the chat-mode mechanism spike."""
    import openvino_genai as ov_genai

    resolved, attrib = _load_configs(root)
    spike = attrib["mechanism_spike"]
    locking = attrib["locking"]
    thermal = attrib["thermal"]
    power_start, quiesce = _quiesce(resolved, attrib)

    spec_path = root / attrib["models"]["int4"]["spec"]
    model_spec = load_local_spec(spec_path)
    model_dir = Path(model_spec["ir_dir"])
    if not model_dir.is_absolute():
        model_dir = root / model_dir

    ov_cfg = attrib["openvino"]
    pipe_kwargs: dict[str, Any] = {"INFERENCE_NUM_THREADS": int(ov_cfg["inference_num_threads"])}
    if ov_cfg.get("scheduling_core_type") is not None:
        pipe_kwargs["SCHEDULING_CORE_TYPE"] = ov_cfg["scheduling_core_type"]
    if ov_cfg.get("enable_cpu_pinning") is not None:
        pipe_kwargs["ENABLE_CPU_PINNING"] = ov_cfg["enable_cpu_pinning"]

    pipe = ov_genai.LLMPipeline(model_dir, str(ov_cfg["device"]), **pipe_kwargs)
    tokenizer = pipe.get_tokenizer()
    context = _text_near_tokens(
        tokenizer,
        str(spike["filler_unit"]),
        int(spike["context_tokens_approx"]),
    )
    new_prompt = _text_near_tokens(
        tokenizer,
        str(spike["new_prompt_unit"]),
        int(spike["prompt_new_tokens_approx"]),
    )
    n_out = int(spike["n_out_tokens"])
    seed = int(spike["rng_seed"])
    chat_cfg = _generation_config(ov_genai, n_out=n_out, seed=seed, templated=True)
    cold_cfg = _generation_config(ov_genai, n_out=n_out, seed=seed, templated=False)

    lock_resource = root / locking["resource"]
    _wait_for_machine_lock(
        lock_resource,
        timeout_s=float(locking["wait_timeout_s"]),
        poll_s=float(locking["poll_interval_s"]),
    )

    rng = random.Random(seed)
    orders = [
        ["seated", "cold"] if rng.random() < 0.5 else ["cold", "seated"]
        for _ in range(int(spike["repeats"]))
    ]
    records: list[dict[str, Any]] = []
    lock_windows: list[dict[str, str]] = []
    sampler = FrequencySampler(
        interval_s=float(thermal["frequency_sample_interval_s"]),
        n_cpus=8,
    )

    # One spike block: warm to steady state, then all randomized repeats.  The lock covers every
    # timed call, including discarded warm-up and context seating.  Analysis starts only after
    # release.
    with exclusive(lock_resource):
        acquired_utc = _utc_now()
        sampler.start()
        try:
            warm_prompt = _text_near_tokens(tokenizer, str(spike["filler_unit"]), 512)
            warm_cfg = _generation_config(ov_genai, n_out=n_out, seed=seed, templated=True)
            warm_started = time.monotonic()
            warm_generations = 0
            while time.monotonic() - warm_started < float(spike["warmup_s"]):
                _timed_generate(pipe, ov_genai, warm_prompt, warm_cfg)
                warm_generations += 1
            warmup_actual_s = time.monotonic() - warm_started

            for repeat_idx, order in enumerate(orders):
                # Preparation session obtains exactly the assistant bytes used to render Path B.
                prepared = _chat_first_turn(pipe, ov_genai, context, chat_cfg)
                _finish_chat(pipe)
                response = str(prepared["text"])
                cold_rendered = _render_history(tokenizer, context, response, new_prompt)

                cold_result: dict[str, Any] | None = None
                seated_result: dict[str, Any] | None = None
                token_check: dict[str, Any] | None = None
                second_response_match: bool | None = None

                for path in order:
                    if path == "cold":
                        cold_result = _timed_generate(pipe, ov_genai, cold_rendered, cold_cfg)
                    else:
                        seated_first = _chat_first_turn(pipe, ov_genai, context, chat_cfg)
                        second_response_match = (
                            seated_first["output_utf8_hex"] == prepared["output_utf8_hex"]
                        )
                        if not second_response_match:
                            _finish_chat(pipe)
                            raise SeamError(
                                "greedy context response changed between preparation and seated "
                                "sessions, so Path A and Path B do not contain the same history"
                            )
                        token_check = _token_precheck(
                            tokenizer,
                            context=context,
                            response=str(seated_first["text"]),
                            new_prompt=new_prompt,
                            cold_rendered=cold_rendered,
                        )
                        if not token_check["passed"]:
                            _finish_chat(pipe)
                            raise SeamError(
                                "token pre-check failed: seated history and cold render differ; "
                                "no correctness or TTFT comparison is valid"
                            )
                        seated_result = _timed_generate(pipe, ov_genai, new_prompt, chat_cfg)
                        _finish_chat(pipe)

                assert cold_result is not None
                assert seated_result is not None
                assert token_check is not None
                ratio = seated_result["ttft_ns"] / cold_result["ttft_ns"]
                records.append(
                    {
                        "repeat_idx": repeat_idx,
                        "path_order": order,
                        "context_response_byte_identical_between_sessions": (second_response_match),
                        "token_precheck": token_check,
                        "seated": seated_result,
                        "cold": cold_result,
                        "greedy_output_byte_identical": (
                            seated_result["output_utf8_hex"] == cold_result["output_utf8_hex"]
                        ),
                        "ttft_ratio_seated_over_cold": ratio,
                    }
                )
        finally:
            frequency_samples = sampler.stop()
    released_utc = _utc_now()
    lock_windows.append(
        {"resource": str(lock_resource), "acquired_utc": acquired_utc, "released_utc": released_utc}
    )

    ratios = [float(record["ttft_ratio_seated_over_cold"]) for record in records]
    ratio_ci = _bootstrap_median_ci(
        ratios,
        resamples=int(spike["bootstrap_resamples"]),
        confidence=float(spike["confidence"]),
        seed=int(spike["bootstrap_seed"]),
    )
    token_pass = all(bool(record["token_precheck"]["passed"]) for record in records)
    correctness_pass = all(bool(record["greedy_output_byte_identical"]) for record in records)
    reuse_pass = ratio_ci["point"] < float(spike["max_median_ttft_ratio"])
    throttle = _throttle_summary(
        frequency_samples,
        cpu_ids=resolved.require("topology.p_cpus"),
        floor_pct=float(thermal["throttle_pct_of_max_floor"]),
    )
    overall_pass = token_pass and correctness_pass and reuse_pass
    power_end = capture_power_state()

    result: dict[str, Any] = {
        "experiment_id": "attrib",
        "stage": "mechanism_spike",
        "pre_registration": "docs/EXPERIMENT_attrib_spec.md",
        "created_utc": _utc_now(),
        "route_fork": "seated_and_interaction" if overall_pass else "interaction_only",
        "overall_pass": overall_pass,
        "tests": {
            "token_precheck": {
                "passed": token_pass,
                "n_passed": sum(bool(record["token_precheck"]["passed"]) for record in records),
                "n_repeats": len(records),
                "criterion": "rendered token ID sequences are identical in every repeat",
            },
            "correctness": {
                "passed": correctness_pass,
                "n_byte_identical": sum(
                    bool(record["greedy_output_byte_identical"]) for record in records
                ),
                "n_repeats": len(records),
                "criterion": "greedy output UTF-8 bytes are identical in every repeat",
            },
            "reuse": {
                "passed": reuse_pass,
                "ttft_ratio_seated_over_cold": ratio_ci,
                "per_repeat": ratios,
                "criterion": f"median < {float(spike['max_median_ttft_ratio'])}",
            },
        },
        "design": {
            "context_content_tokens": len(_token_ids(tokenizer, context)),
            "new_prompt_content_tokens": len(_token_ids(tokenizer, new_prompt)),
            "n_out_tokens": n_out,
            "repeats": len(records),
            "orders": orders,
            "greedy": True,
            "do_sample": False,
            "temperature": 0.0,
            "rng_seed": seed,
            "ignore_eos": True,
            "warmup_s_requested": float(spike["warmup_s"]),
            "warmup_s_actual": warmup_actual_s,
            "warmup_generations": warm_generations,
        },
        "records": records,
        "lock_windows": lock_windows,
        "quiesce": quiesce,
        "throttle": throttle,
        "openvino": asdict(runtime_info()),
        "pipeline_properties": pipe_kwargs,
    }

    model_block = manifest_model_block(
        spec=model_spec,
        spec_path=spec_path,
        reasoning_mode="thinking_off",
    )
    runtime = runtime_info()
    battery_cfg = resolved.get("power.battery") or {}
    handle = emit(
        config=resolved,
        target=str(attrib["execution_target"]),
        workload={
            "kind": "microbench",
            "benchmark": "attrib_chat_mechanism_spike_v1",
            "task_ids": [],
            "seed": seed,
            "n_repeats": len(records),
        },
        condition_label="attrib/mechanism_spike/chat_vs_cold",
        repo_root=root,
        allow_dirty=allow_dirty,
        summary=result,
        model=model_block,
        drivers={
            "openvino": runtime.openvino,
            "genai": runtime.genai,
        },
        power_state=manifest_power_state(
            power_start,
            battery_pct_end=power_end.battery_pct,
            design_capacity_mwh=battery_cfg.get("design_capacity_mwh"),
            full_charge_capacity_mwh=battery_cfg.get("full_charge_capacity_mwh"),
        ),
        thermal={
            "warmup_s": float(spike["warmup_s"]),
            "throttle_residency_pct": (
                None
                if throttle["throttle_fraction"] is None
                else 100.0 * float(throttle["throttle_fraction"])
            ),
            "throttle_threshold_pct": 100.0 * float(thermal["max_throttle_fraction"]),
            "excluded": (
                throttle["throttle_fraction"] is not None
                and float(throttle["throttle_fraction"]) >= float(thermal["max_throttle_fraction"])
            ),
            "regime": str(thermal["regime"]),
        },
        outputs={
            "samples": None,
            "steps": None,
            "summary": "summary.json",
            "events_path": "events.ndjson",
        },
        self_check="pass" if overall_pass else "fail",
        seal=True,
    )
    if not verify_sealed(handle.run_dir):
        raise SeamError(f"sealed run {handle.run_id} failed integrity verification")

    result["run_id"] = handle.run_id
    result["raw_sha256"] = handle.raw_sha256
    return result


def _write_derived_spike(root: Path, result: dict[str, Any]) -> Path:
    path = root / "derived" / "attrib" / "mechanism_spike.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _exact_prompt(tokenizer: Any, *, target_tokens: int, unit: str) -> tuple[str, int]:
    """Construct user content whose rendered chat prompt is exactly ``target_tokens`` long."""

    def rendered_count(repeats: int) -> int:
        content = unit * repeats
        rendered = tokenizer.apply_chat_template([{"role": "user", "content": content}], True)
        return len(_token_ids(tokenizer, str(rendered)))

    if rendered_count(0) > target_tokens:
        raise ConfigError(f"chat-template overhead exceeds requested total prompt {target_tokens}")
    low, high = 0, target_tokens
    while low <= high:
        midpoint = (low + high) // 2
        count = rendered_count(midpoint)
        if count == target_tokens:
            return unit * midpoint, count
        if count < target_tokens:
            low = midpoint + 1
        else:
            high = midpoint - 1
    raise ConfigError(
        f"cannot construct an exact {target_tokens}-token rendered prompt with unit {unit!r}; "
        f"nearest bracketing repeat counts are {high} and {low}"
    )


def _interaction_preflight(root: Path, attrib: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Require every pre-registered quantization IR before any sweep measurement."""
    available: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    max_prompt = max(int(value) for value in attrib["interaction_grid"]["total_prompt_tokens"])
    max_out = max(int(value) for value in attrib["interaction_grid"]["n_out"])
    for quantization in attrib["interaction_grid"]["quantization"]:
        model_cfg = attrib["models"].get(str(quantization)) or {}
        relative_spec = model_cfg.get("spec")
        if not relative_spec:
            missing.append(
                f"{quantization}: {model_cfg.get('absent_reason', 'model spec is null')}"
            )
            continue
        spec_path = root / str(relative_spec)
        spec = load_local_spec(spec_path)
        model_dir = Path(spec["ir_dir"])
        if not model_dir.is_absolute():
            model_dir = root / model_dir
        config_path = model_dir / "config.json"
        model_config = json.loads(config_path.read_text(encoding="utf-8"))
        position_limit = int(model_config["max_position_embeddings"])
        if max_prompt + max_out > position_limit:
            missing.append(
                f"{quantization}: max P+n_out={max_prompt + max_out} exceeds "
                f"max_position_embeddings={position_limit}"
            )
            continue
        available[str(quantization)] = {
            "spec": spec,
            "spec_path": spec_path,
            "model_dir": model_dir,
            "max_position_embeddings": position_limit,
        }
    if missing:
        raise SeamError(
            "interaction sweep preflight refused before measurement: "
            + "; ".join(missing)
            + ". Every declared factor level is required; no reduced sweep was run."
        )
    return available


def _interaction_cells(attrib: Mapping[str, Any], *, quantization: str) -> list[dict[str, Any]]:
    grid = attrib["interaction_grid"]
    cells = [
        {
            "total_prompt_tokens": int(prompt),
            "n_out": int(n_out),
            "quantization": quantization,
            "held_out": False,
        }
        for prompt in grid["total_prompt_tokens"]
        for n_out in grid["n_out"]
    ]
    cells.extend(
        {
            "total_prompt_tokens": int(cell["total_prompt_tokens"]),
            "n_out": int(cell["n_out"]),
            "quantization": quantization,
            "held_out": True,
        }
        for cell in grid["held_out_cells"]
        if str(cell["quantization"]) == quantization
    )
    return cells


def _runtime_projection(
    pilot_records: Sequence[Mapping[str, Any]],
    *,
    attrib: Mapping[str, Any],
) -> dict[str, Any]:
    """Project the declared sweep from four-corner medians, including protocol overhead."""
    pilot_cfg = attrib["runtime_pilot"]
    corner_values: dict[tuple[int, int], list[float]] = {}
    for record in pilot_records:
        key = (int(record["total_prompt_tokens"]), int(record["n_out"]))
        corner_values.setdefault(key, []).append(float(record["wall_s"]))
    corners = [
        (int(cell["total_prompt_tokens"]), int(cell["n_out"])) for cell in pilot_cfg["corners"]
    ]
    if set(corner_values) != set(corners):
        raise SeamError("runtime pilot records do not cover exactly the four declared corners")

    def project(corner_times: Sequence[float]) -> float:
        matrix = np.asarray(
            [[1.0, prompt, n_out, prompt * n_out] for prompt, n_out in corners],
            dtype=float,
        )
        if int(np.linalg.matrix_rank(matrix)) != 4:
            raise SeamError("runtime-pilot corner matrix is rank deficient")
        coefficients = np.linalg.solve(matrix, np.asarray(corner_times, dtype=float))
        all_cells = _interaction_cells(attrib, quantization="int4")
        predicted = [
            max(
                0.0,
                float(
                    np.asarray(
                        [
                            1.0,
                            cell["total_prompt_tokens"],
                            cell["n_out"],
                            cell["total_prompt_tokens"] * cell["n_out"],
                        ],
                        dtype=float,
                    )
                    @ coefficients
                ),
            )
            for cell in all_cells
        ]
        repeats = int(pilot_cfg["projected_measured_repeats"])
        measured_s = repeats * sum(predicted)
        first_touch_s = sum(predicted) if bool(pilot_cfg["first_touch_once_per_cell"]) else 0.0
        warmup_s = repeats * float(attrib["thermal"]["warmup_s"])
        cooldown_s = (repeats - 1) * float(attrib["thermal"]["cooldown_between_blocks_s"])
        return measured_s + first_touch_s + warmup_s + cooldown_s

    medians = [statistics.median(corner_values[corner]) for corner in corners]
    point_s = project(medians)
    rng = random.Random(int(pilot_cfg["bootstrap_seed"]))
    draws = []
    for _ in range(int(pilot_cfg["bootstrap_resamples"])):
        sampled_medians = []
        for corner in corners:
            values = corner_values[corner]
            sampled = [values[rng.randrange(len(values))] for _ in values]
            sampled_medians.append(statistics.median(sampled))
        draws.append(project(sampled_medians))
    draws.sort()
    alpha = 0.025
    lo = draws[max(0, int(alpha * len(draws)) - 1)]
    hi = draws[min(len(draws) - 1, int((1.0 - alpha) * len(draws)))]
    return {
        "method": "four-corner bilinear wall-time interpolation",
        "corner_median_wall_s": {
            f"P{prompt}_n{n_out}": median
            for (prompt, n_out), median in zip(corners, medians, strict=True)
        },
        "projected_total_s": point_s,
        "projected_total_h": point_s / 3600.0,
        "bootstrap_ci95_s": [lo, hi],
        "bootstrap_ci95_h": [lo / 3600.0, hi / 3600.0],
        "n_grid_cells": 35,
        "n_held_out_cells": len(attrib["interaction_grid"]["held_out_cells"]),
        "measured_repeats": int(pilot_cfg["projected_measured_repeats"]),
        "first_touch_once_per_cell": bool(pilot_cfg["first_touch_once_per_cell"]),
        "warmup_s_per_block": float(attrib["thermal"]["warmup_s"]),
        "cooldown_s_between_blocks": float(attrib["thermal"]["cooldown_between_blocks_s"]),
    }


def run_runtime_pilot(*, root: Path, allow_dirty: bool = False) -> dict[str, Any]:
    """Time four corners and seal the full-sweep wall-time projection."""
    import openvino_genai as ov_genai

    resolved, attrib = _load_configs(root)
    models = _interaction_preflight(root, attrib)
    model = models["int4"]
    power_start, quiesce = _quiesce(resolved, attrib)
    pilot_cfg = attrib["runtime_pilot"]
    thermal = attrib["thermal"]
    locking = attrib["locking"]
    seed = int(attrib["sweep"]["seed"])
    pipe_kwargs: dict[str, Any] = {
        "INFERENCE_NUM_THREADS": int(attrib["openvino"]["inference_num_threads"])
    }
    corners = [
        {
            "total_prompt_tokens": int(cell["total_prompt_tokens"]),
            "n_out": int(cell["n_out"]),
        }
        for cell in pilot_cfg["corners"]
    ]
    order = [
        {**cell, "repeat_idx": repeat}
        for repeat in range(int(pilot_cfg["repeats"]))
        for cell in corners
    ]
    random.Random(seed).shuffle(order)
    lock_resource = root / locking["resource"]
    _wait_for_machine_lock(
        lock_resource,
        timeout_s=float(locking["wait_timeout_s"]),
        poll_s=float(locking["poll_interval_s"]),
    )
    sampler = FrequencySampler(interval_s=float(thermal["frequency_sample_interval_s"]), n_cpus=8)
    records: list[dict[str, Any]] = []
    with exclusive(lock_resource):
        acquired_utc = _utc_now()
        pipe = ov_genai.LLMPipeline(
            model["model_dir"],
            str(attrib["openvino"]["device"]),
            **pipe_kwargs,
        )
        tokenizer = pipe.get_tokenizer()
        prompts = {
            int(cell["total_prompt_tokens"]): _exact_prompt(
                tokenizer,
                target_tokens=int(cell["total_prompt_tokens"]),
                unit=" x",
            )[0]
            for cell in corners
        }
        sampler.start()
        try:
            warm_cfg = _generation_config(ov_genai, n_out=64, seed=seed, templated=True)
            warm_started = time.monotonic()
            warm_generations = 0
            while time.monotonic() - warm_started < float(pilot_cfg["warmup_s"]):
                _timed_generate(pipe, ov_genai, prompts[512], warm_cfg)
                warm_generations += 1
            warmup_actual_s = time.monotonic() - warm_started

            for cell in corners:
                first_touch_cfg = _generation_config(
                    ov_genai,
                    n_out=int(cell["n_out"]),
                    seed=seed,
                    templated=True,
                )
                _timed_generate(
                    pipe,
                    ov_genai,
                    prompts[int(cell["total_prompt_tokens"])],
                    first_touch_cfg,
                )

            for sequence_idx, cell in enumerate(order):
                cfg = _generation_config(
                    ov_genai,
                    n_out=int(cell["n_out"]),
                    seed=seed,
                    templated=True,
                )
                timing = _timed_generate(
                    pipe,
                    ov_genai,
                    prompts[int(cell["total_prompt_tokens"])],
                    cfg,
                )
                reported = int(timing["prompt_tokens_reported"])
                if reported != int(cell["total_prompt_tokens"]):
                    raise SeamError(
                        f"runtime pilot prompt readback {reported} != "
                        f"target {cell['total_prompt_tokens']}"
                    )
                records.append(
                    {
                        **cell,
                        "sequence_idx": sequence_idx,
                        "wall_s": float(timing["wall_ns"]) / 1e9,
                        "wall_ns": int(timing["wall_ns"]),
                        "ttft_ns": int(timing["ttft_ns"]),
                        "ttft_source": timing["ttft_source"],
                        "completion_tokens_reported": int(timing["completion_tokens_reported"]),
                    }
                )
        finally:
            frequency_samples = sampler.stop()
    released_utc = _utc_now()

    throttle = _throttle_summary(
        frequency_samples,
        cpu_ids=resolved.require("topology.p_cpus"),
        floor_pct=float(thermal["throttle_pct_of_max_floor"]),
    )
    projection = _runtime_projection(records, attrib=attrib)
    power_end = capture_power_state()
    result = {
        "experiment_id": "attrib",
        "stage": "runtime_pilot",
        "route": "interaction",
        "quantization": "int4",
        "records": records,
        "projection": projection,
        "lock_windows": [
            {
                "resource": str(lock_resource),
                "acquired_utc": acquired_utc,
                "released_utc": released_utc,
            }
        ],
        "quiesce": quiesce,
        "throttle": throttle,
        "warmup_s_requested": float(pilot_cfg["warmup_s"]),
        "warmup_s_actual": warmup_actual_s,
        "warmup_generations": warm_generations,
        "full_sweep_committed": False,
        "authorization_status": "pending_human_schedule_review",
        "openvino": asdict(runtime_info()),
    }
    runtime = runtime_info()
    battery_cfg = resolved.get("power.battery") or {}
    throttle_fraction = throttle["throttle_fraction"]
    handle = emit(
        config=resolved,
        target=str(attrib["execution_target"]),
        workload={
            "kind": "microbench",
            "benchmark": "attrib_runtime_pilot_v1",
            "task_ids": [],
            "seed": seed,
            "n_repeats": int(pilot_cfg["repeats"]),
        },
        condition_label="attrib/runtime_pilot/int4",
        repo_root=root,
        allow_dirty=allow_dirty,
        summary=result,
        model=manifest_model_block(
            spec=model["spec"],
            spec_path=model["spec_path"],
            reasoning_mode="thinking_off",
        ),
        drivers={"openvino": runtime.openvino, "genai": runtime.genai},
        power_state=manifest_power_state(
            power_start,
            battery_pct_end=power_end.battery_pct,
            design_capacity_mwh=battery_cfg.get("design_capacity_mwh"),
            full_charge_capacity_mwh=battery_cfg.get("full_charge_capacity_mwh"),
        ),
        thermal={
            "warmup_s": float(pilot_cfg["warmup_s"]),
            "throttle_residency_pct": (
                None if throttle_fraction is None else 100.0 * float(throttle_fraction)
            ),
            "throttle_threshold_pct": 100.0 * float(thermal["max_throttle_fraction"]),
            "excluded": (
                throttle_fraction is not None
                and float(throttle_fraction) >= float(thermal["max_throttle_fraction"])
            ),
            "regime": str(thermal["regime"]),
        },
        outputs={
            "samples": None,
            "steps": None,
            "summary": "summary.json",
            "events_path": "events.ndjson",
        },
        self_check=(
            "pass"
            if throttle_fraction is not None
            and float(throttle_fraction) < float(thermal["max_throttle_fraction"])
            else "fail"
        ),
        seal=True,
    )
    if not verify_sealed(handle.run_dir):
        raise SeamError(f"sealed runtime pilot {handle.run_id} failed integrity")
    result["run_id"] = handle.run_id
    result["raw_sha256"] = handle.raw_sha256
    path = root / "derived" / "attrib" / "runtime_pilot.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _assert_sweep_authorized(*, root: Path, attrib: Mapping[str, Any]) -> None:
    authorization = attrib["runtime_pilot"]["full_sweep_authorization"]
    run_id = authorization.get("pilot_run_id")
    if not bool(authorization.get("authorized")) or not run_id:
        raise SeamError(
            "full interaction sweep refused: runtime-pilot estimate has not been human-authorized "
            "in configs/attrib.yaml"
        )
    run_dir = open_run_dir(str(run_id), repo_root=root)
    if not run_dir.is_sealed() or not verify_sealed(run_dir):
        raise SeamError(f"authorized runtime pilot {run_id} is not sealed and verified")


def run_interaction_sweep(
    *,
    root: Path,
    quantization: str,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Measure one complete quantization stratum of the centered interaction route."""
    import openvino_genai as ov_genai

    resolved, attrib = _load_configs(root)
    if attrib["seating"]["mechanism_spike_verdict"] != "failed_reuse":
        raise SeamError("interaction-only driver requires the recorded failed-reuse fork")
    models = _interaction_preflight(root, attrib)
    _assert_sweep_authorized(root=root, attrib=attrib)
    if quantization not in models:
        raise ConfigError(f"quantization {quantization!r} is not in the pre-registered grid")
    power_start, quiesce = _quiesce(resolved, attrib)

    model = models[quantization]
    ov_cfg = attrib["openvino"]
    pipe_kwargs: dict[str, Any] = {"INFERENCE_NUM_THREADS": int(ov_cfg["inference_num_threads"])}
    cells = _interaction_cells(attrib, quantization=quantization)
    thermal = attrib["thermal"]
    sweep_cfg = attrib["sweep"]
    locking = attrib["locking"]
    repeats = int(attrib["interaction_grid"]["repeats"])
    seed = int(sweep_cfg["seed"])
    rng = random.Random(seed)
    lock_resource = root / locking["resource"]
    records: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    sequence_idx = 0
    _wait_for_machine_lock(
        lock_resource,
        timeout_s=float(locking["wait_timeout_s"]),
        poll_s=float(locking["poll_interval_s"]),
    )
    with exclusive(lock_resource):
        setup_acquired_utc = _utc_now()
        pipe = ov_genai.LLMPipeline(model["model_dir"], str(ov_cfg["device"]), **pipe_kwargs)
        tokenizer = pipe.get_tokenizer()
        filler = " x"
        prompts = {
            int(prompt): _exact_prompt(tokenizer, target_tokens=int(prompt), unit=filler)[0]
            for prompt in {
                *attrib["interaction_grid"]["total_prompt_tokens"],
                *[
                    cell["total_prompt_tokens"]
                    for cell in attrib["interaction_grid"]["held_out_cells"]
                    if str(cell["quantization"]) == quantization
                ],
            }
        }
    setup_released_utc = _utc_now()

    for block_idx in range(repeats):
        order = [dict(cell) for cell in cells]
        rng.shuffle(order)
        _wait_for_machine_lock(
            lock_resource,
            timeout_s=float(locking["wait_timeout_s"]),
            poll_s=float(locking["poll_interval_s"]),
        )
        sampler = FrequencySampler(
            interval_s=float(thermal["frequency_sample_interval_s"]), n_cpus=8
        )
        with exclusive(lock_resource):
            acquired_utc = _utc_now()
            sampler.start()
            try:
                warm_cfg = _generation_config(ov_genai, n_out=64, seed=seed, templated=True)
                warm_started = time.monotonic()
                warm_generations = 0
                while time.monotonic() - warm_started < float(thermal["warmup_s"]):
                    _timed_generate(pipe, ov_genai, prompts[1024], warm_cfg)
                    warm_generations += 1

                if block_idx == 0 and bool(sweep_cfg["discard_first_touch_per_shape"]):
                    for cell in order:
                        cfg = _generation_config(
                            ov_genai,
                            n_out=int(cell["n_out"]),
                            seed=seed,
                            templated=True,
                        )
                        _timed_generate(
                            pipe,
                            ov_genai,
                            prompts[int(cell["total_prompt_tokens"])],
                            cfg,
                        )

                measured: list[dict[str, Any]] = []
                for cell in order:
                    cfg = _generation_config(
                        ov_genai,
                        n_out=int(cell["n_out"]),
                        seed=seed,
                        templated=True,
                    )
                    timing = _timed_generate(
                        pipe,
                        ov_genai,
                        prompts[int(cell["total_prompt_tokens"])],
                        cfg,
                    )
                    if int(timing["prompt_tokens_reported"]) != int(cell["total_prompt_tokens"]):
                        raise SeamError(
                            "OpenVINO prompt-token readback disagrees with exact tokenizer "
                            f"target: requested={cell['total_prompt_tokens']}, "
                            f"reported={timing['prompt_tokens_reported']}"
                        )
                    measured.append(
                        {
                            **cell,
                            "execution_target": str(attrib["execution_target"]),
                            "wall_s": float(timing["wall_ns"]) / 1e9,
                            "wall_ns": int(timing["wall_ns"]),
                            "ttft_ns": int(timing["ttft_ns"]),
                            "ttft_source": timing["ttft_source"],
                            "completion_tokens_reported": timing["completion_tokens_reported"],
                            "repeat_idx": block_idx,
                            "block_idx": block_idx,
                            "sequence_idx": sequence_idx,
                        }
                    )
                    sequence_idx += 1
            finally:
                frequency_samples = sampler.stop()
        released_utc = _utc_now()

        throttle = _throttle_summary(
            frequency_samples,
            cpu_ids=resolved.require("topology.p_cpus"),
            floor_pct=float(thermal["throttle_pct_of_max_floor"]),
        )
        block = {
            "block_idx": block_idx,
            "lock_acquired_utc": acquired_utc,
            "lock_released_utc": released_utc,
            "lock_resource": str(lock_resource),
            "lock_overlap_count": 0,
            "warmup_generations": warm_generations,
            "throttle": throttle,
            "cell_order": order,
        }
        blocks.append(block)
        for record in measured:
            record["lock_overlap_count"] = 0
            record["throttle_fraction"] = throttle["throttle_fraction"]
            record["idle_baseline_start"] = None
            record["idle_baseline_end"] = None
            records.append(record)
        if block_idx + 1 < repeats:
            time.sleep(float(thermal["cooldown_between_blocks_s"]))

    power_end = capture_power_state()
    result = {
        "experiment_id": "attrib",
        "route": "interaction",
        "mechanism_spike_run_id": attrib["seating"]["mechanism_spike_run_id"],
        "quantization": quantization,
        "records": records,
        "blocks": blocks,
        "setup_lock_window": {
            "resource": str(lock_resource),
            "acquired_utc": setup_acquired_utc,
            "released_utc": setup_released_utc,
        },
        "quiesce": quiesce,
        "design": {
            "P_levels": attrib["interaction_grid"]["total_prompt_tokens"],
            "n_out_levels": attrib["interaction_grid"]["n_out"],
            "repeats": repeats,
            "center_before_interaction": True,
        },
        "openvino": asdict(runtime_info()),
    }
    model_block = manifest_model_block(
        spec=model["spec"],
        spec_path=model["spec_path"],
        reasoning_mode="thinking_off",
    )
    runtime = runtime_info()
    battery_cfg = resolved.get("power.battery") or {}
    max_throttle = max(
        (
            float(block["throttle"]["throttle_fraction"])
            for block in blocks
            if block["throttle"]["throttle_fraction"] is not None
        ),
        default=None,
    )
    handle = emit(
        config=resolved,
        target=str(attrib["execution_target"]),
        workload={
            "kind": "microbench",
            "benchmark": "attrib_centered_interaction_v1",
            "task_ids": [],
            "seed": seed,
            "n_repeats": repeats,
        },
        condition_label=f"attrib/interaction/{quantization}",
        repo_root=root,
        allow_dirty=allow_dirty,
        summary=result,
        model=model_block,
        drivers={"openvino": runtime.openvino, "genai": runtime.genai},
        power_state=manifest_power_state(
            power_start,
            battery_pct_end=power_end.battery_pct,
            design_capacity_mwh=battery_cfg.get("design_capacity_mwh"),
            full_charge_capacity_mwh=battery_cfg.get("full_charge_capacity_mwh"),
        ),
        thermal={
            "warmup_s": float(thermal["warmup_s"]),
            "throttle_residency_pct": (None if max_throttle is None else 100.0 * max_throttle),
            "throttle_threshold_pct": 100.0 * float(thermal["max_throttle_fraction"]),
            "excluded": (
                max_throttle is not None and max_throttle >= float(thermal["max_throttle_fraction"])
            ),
            "regime": str(thermal["regime"]),
        },
        outputs={
            "samples": None,
            "steps": None,
            "summary": "summary.json",
            "events_path": "events.ndjson",
        },
        self_check="pass",
        seal=True,
    )
    if not verify_sealed(handle.run_dir):
        raise SeamError(f"sealed interaction run {handle.run_id} failed integrity")
    return {
        "run_id": handle.run_id,
        "raw_sha256": handle.raw_sha256,
        "quantization": quantization,
        "n_records": len(records),
    }


def record_kv_geometry(*, root: Path, allow_dirty: bool = False) -> dict[str, Any]:
    """Seal the live CPU KV-precision readback and full bytes/token derivation."""
    resolved, attrib = _load_configs(root)
    spec_path = root / attrib["models"]["int4"]["spec"]
    model_spec = load_local_spec(spec_path)
    model_dir = Path(model_spec["ir_dir"])
    if not model_dir.is_absolute():
        model_dir = root / model_dir
    geometry = load_kv_geometry(
        model_dir,
        device=str(attrib["openvino"]["device"]),
        assumed_kv_dtype=str(attrib["openvino"]["assumed_kv_dtype"]),
    )
    f16_counterfactual = 2 * geometry.n_layers * geometry.n_kv_heads * geometry.head_dim * 2
    result = {
        "experiment_id": "attrib",
        "stage": "kv_geometry_resolution",
        "geometry": geometry.to_record(),
        "k_and_v_counted": True,
        "leading_factor_two_meaning": "K and V",
        "f16_counterfactual_bytes_per_token": f16_counterfactual,
        "f16_counterfactual_kib_per_token": f16_counterfactual / 1024.0,
        "live_bytes_per_token": geometry.bytes_per_token,
        "live_kib_per_token": geometry.bytes_per_token / 1024.0,
        "ratio_f16_counterfactual_over_live": (f16_counterfactual / geometry.bytes_per_token),
        "resolution": (
            "The exact 2x difference is explained by dtype: live CPU KV_CACHE_PRECISION is u8 "
            "(1 byte/element), while 144 KiB/token assumes f16/bf16 (2 bytes/element). "
            "The model config fixes num_key_value_heads at 8, so head count is not the source."
        ),
        "formula": KV_FORMULA,
    }
    runtime = runtime_info()
    power = capture_power_state()
    battery_cfg = resolved.get("power.battery") or {}
    handle = emit(
        config=resolved,
        target=str(attrib["execution_target"]),
        workload={
            "kind": "microbench",
            "benchmark": "attrib_kv_geometry_readback_v1",
            "task_ids": [],
            "seed": None,
            "n_repeats": 1,
        },
        condition_label="attrib/kv_geometry/cpu",
        repo_root=root,
        allow_dirty=allow_dirty,
        summary=result,
        model=manifest_model_block(
            spec=model_spec,
            spec_path=spec_path,
            reasoning_mode="thinking_off",
        ),
        drivers={"openvino": runtime.openvino, "genai": runtime.genai},
        power_state=manifest_power_state(
            power,
            battery_pct_end=power.battery_pct,
            design_capacity_mwh=battery_cfg.get("design_capacity_mwh"),
            full_charge_capacity_mwh=battery_cfg.get("full_charge_capacity_mwh"),
        ),
        outputs={
            "samples": None,
            "steps": None,
            "summary": "summary.json",
            "events_path": "events.ndjson",
        },
        self_check="pass",
        seal=True,
    )
    if not verify_sealed(handle.run_dir):
        raise SeamError(f"sealed KV geometry run {handle.run_id} failed integrity")
    result["run_id"] = handle.run_id
    result["raw_sha256"] = handle.raw_sha256
    path = root / "derived" / "attrib" / "kv_geometry.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    spike = subparsers.add_parser("mechanism-spike")
    spike.add_argument("--allow-dirty", action="store_true")
    spike.add_argument("--root", type=Path, default=_ROOT)
    sweep = subparsers.add_parser("interaction-sweep")
    sweep.add_argument("--quantization", required=True)
    sweep.add_argument("--allow-dirty", action="store_true")
    sweep.add_argument("--root", type=Path, default=_ROOT)
    pilot = subparsers.add_parser("runtime-pilot")
    pilot.add_argument("--allow-dirty", action="store_true")
    pilot.add_argument("--root", type=Path, default=_ROOT)
    kv = subparsers.add_parser("kv-geometry")
    kv.add_argument("--allow-dirty", action="store_true")
    kv.add_argument("--root", type=Path, default=_ROOT)
    args = parser.parse_args(argv)

    if args.command == "mechanism-spike":
        result = run_mechanism_spike(root=args.root.resolve(), allow_dirty=bool(args.allow_dirty))
        path = _write_derived_spike(args.root.resolve(), result)
        print(
            json.dumps(
                {
                    "run_id": result["run_id"],
                    "passed": result["overall_pass"],
                    "derived": str(path),
                },
                indent=2,
            )
        )
        return 0 if result["overall_pass"] else 2
    if args.command == "interaction-sweep":
        result = run_interaction_sweep(
            root=args.root.resolve(),
            quantization=str(args.quantization),
            allow_dirty=bool(args.allow_dirty),
        )
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "runtime-pilot":
        result = run_runtime_pilot(root=args.root.resolve(), allow_dirty=bool(args.allow_dirty))
        print(
            json.dumps(
                {
                    "run_id": result["run_id"],
                    "projected_total_h": result["projection"]["projected_total_h"],
                    "bootstrap_ci95_h": result["projection"]["bootstrap_ci95_h"],
                    "full_sweep_committed": False,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "kv-geometry":
        result = record_kv_geometry(root=args.root.resolve(), allow_dirty=bool(args.allow_dirty))
        print(
            json.dumps(
                {
                    "run_id": result["run_id"],
                    "live_bytes_per_token": result["live_bytes_per_token"],
                    "resolution": result["resolution"],
                },
                indent=2,
            )
        )
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
