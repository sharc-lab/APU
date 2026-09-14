"""Can an arm's device configuration compile and run in this launch context?

Enumeration (``probe_igpu``) established that the iGPU appears in session 0 and on the console.
Enumeration is not execution. openvino#34390 (``CL_INVALID_WORK_GROUP_SIZE``) manifests at
kernel compilation, which enumeration never exercises.

This smoke loads the measurement model under an arm configuration from
``configs/delta_n.yaml`` (default arm B: historically ``load_sequence: [GPU, CPU]``,
``generate_device: CPU``, with ``openvino.cpu_properties`` applied to the CPU pipeline),
compiles through the same OpenVINO GenAI ``LLMPipeline`` path delta_n uses, and generates a
fixed number of greedy tokens. Completing compile+execute is the question; with ``-N`` it
also records wall / prefill / decode timing against a long enough decode sample.

With ``-N`` / ``--n-tokens``, the prompt is built by the same ``prompt_for`` /
``build_exact_prompt`` path the ladder uses (``filler_unit`` from ``configs/delta_n.yaml``),
with realized token count verified and sha256 recorded, and ``max_new_tokens`` is raised to
64 so decode rate is actually sampled. Without ``-N``, the short fixed prompt and 8-token
bound are retained.

Diagnostic, not a measurement: writes only under ``derived/gpu_smoke/``, seals nothing, and
touches no run ledger. A known defect is classified and recorded; nothing retries.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DELTA_N_CFG = ROOT / "configs" / "delta_n.yaml"
FIXED_PROMPT = "Say hello in one short sentence."
MAX_NEW_TOKENS_FIXED = 8  # short smoke without -N; matches ladder.max_new_tokens
MAX_NEW_TOKENS_WITH_N = 64  # decode must be sampled when measuring rate
_NS_PER_S = 1_000_000_000


def _load_probe_helpers() -> Any:
    """Reuse session / window-station probing from probe_igpu without making tools a package."""
    path = Path(__file__).resolve().parent / "probe_igpu.py"
    spec = importlib.util.spec_from_file_location("_probe_igpu_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load probe helpers from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_probe = _load_probe_helpers()
_describe = _probe._describe
_session_id = _probe._session_id
_window_station = _probe._window_station
_is_interactive = _probe._is_interactive
_path_head = _probe._path_head


def _classify_error(message: str) -> str:
    if "CL_INVALID_WORK_GROUP_SIZE" in message:
        return "UNSUPPORTED"
    return "OTHER"


def _load_delta_n_cfg() -> dict[str, Any]:
    cfg = yaml.safe_load(DELTA_N_CFG.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise TypeError("configs/delta_n.yaml: expected mapping")
    return cfg


def _arm_ids(cfg: dict[str, Any]) -> list[str]:
    arms = cfg.get("arms")
    if not isinstance(arms, list):
        raise TypeError("configs/delta_n.yaml: missing arms list")
    ids: list[str] = []
    for arm in arms:
        if not isinstance(arm, dict) or "id" not in arm:
            raise TypeError("configs/delta_n.yaml: each arm must have an id")
        ids.append(str(arm["id"]))
    return ids


def _max_new_tokens(n_tokens: int | None) -> int:
    return MAX_NEW_TOKENS_WITH_N if n_tokens is not None else MAX_NEW_TOKENS_FIXED


def _arm_device_config(cfg: dict[str, Any], arm_id: str) -> dict[str, Any]:
    arms = cfg.get("arms")
    if not isinstance(arms, list):
        raise TypeError("configs/delta_n.yaml: missing arms list")
    arm = next((a for a in arms if a.get("id") == arm_id), None)
    if arm is None:
        known = _arm_ids(cfg)
        raise ValueError(f"configs/delta_n.yaml: no arm with id {arm_id!r} (known: {known})")
    load_sequence = arm.get("load_sequence")
    generate_device = arm.get("generate_device")
    if not load_sequence or generate_device is None:
        raise ValueError(
            f"configs/delta_n.yaml arm {arm_id}: cannot determine load_sequence / generate_device "
            f"(load_sequence={load_sequence!r}, generate_device={generate_device!r})"
        )
    ov = cfg.get("openvino") or {}
    cpu_properties = dict(ov.get("cpu_properties") or {})
    arm_properties = dict(arm.get("properties") or {})
    # Same construction as smoke_delta_prefill / ceiling_a: cpu props on CPU, then arm props.
    resolved_load: list[dict[str, Any]] = []
    for device in load_sequence:
        props: dict[str, Any] = {}
        if str(device) == "CPU":
            props.update(cpu_properties)
        props.update(arm_properties)
        resolved_load.append({"device": str(device), "properties": props})
    return {
        "arm_id": arm_id,
        "label": arm.get("label"),
        "load_sequence": resolved_load,
        "generate_device": str(generate_device),
        "cpu_properties": cpu_properties,
        "arm_properties": arm_properties,
    }


def _timing_fields(
    *,
    wall_ns: int,
    ttft_ns: int | None,
    tokens_generated: int | None,
) -> dict[str, float | None]:
    """prefill_s / decode_tok_s / wall_s. Missing TTFT or short decode → null rate, not zero."""
    wall_s = wall_ns / _NS_PER_S
    prefill_s = (ttft_ns / _NS_PER_S) if ttft_ns is not None and ttft_ns > 0 else None
    decode_tok_s: float | None = None
    if (
        prefill_s is not None
        and tokens_generated is not None
        and tokens_generated >= 2
        and wall_s > prefill_s
    ):
        decode_tok_s = (tokens_generated - 1) / (wall_s - prefill_s)
    return {
        "prefill_s": prefill_s,
        "decode_tok_s": decode_tok_s,
        "wall_s": wall_s,
    }


def _model_spec_path(cfg: dict[str, Any], override: str | Path | None = None) -> Path:
    """Resolve FetchedModelSpec path. Default is configs/delta_n.yaml openvino.model_spec (4B)."""
    if override is not None:
        path = Path(override)
        return path if path.is_absolute() else ROOT / path
    ov = cfg.get("openvino") or {}
    model_spec = ov.get("model_spec")
    if not model_spec:
        raise ValueError("configs/delta_n.yaml: openvino.model_spec missing")
    return ROOT / str(model_spec)


def _resolve_model(cfg: dict[str, Any], spec_path: str | Path | None = None) -> tuple[str, Path]:
    path = _model_spec_path(cfg, spec_path)
    if not path.is_file():
        raise ValueError(f"model spec not found: {path}")
    model = yaml.safe_load(path.read_text(encoding="utf-8"))
    model_id = str(model.get("name") or path.stem)
    ir_dir = model.get("ir_dir")
    if not ir_dir:
        raise ValueError(f"{path}: ir_dir missing")
    model_dir = Path(str(ir_dir))
    if not model_dir.is_dir():
        raise ValueError(f"model ir_dir does not exist: {model_dir}")
    return model_id, model_dir


def _build_ladder_prompt(
    *, cfg: dict[str, Any], model_dir: Path, n_tokens: int
) -> tuple[str, dict[str, Any]]:
    """Same builder the ladder uses; realized count verified, sha256 recorded."""
    from transformers import AutoTokenizer

    from seam.tools.delta_n import prompt_for

    ladder = cfg.get("ladder") or {}
    unit = ladder.get("filler_unit")
    if not unit:
        raise ValueError("configs/delta_n.yaml: ladder.filler_unit missing")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    entry = prompt_for(
        root=ROOT,
        tokenizer=tokenizer,
        n_tokens=int(n_tokens),
        unit=str(unit),
        cache={},
    )
    realized = int(entry["n_tokens_realized"])
    if realized != int(n_tokens):
        raise RuntimeError(
            f"prompt realized tokens {realized} != requested {n_tokens}; "
            "refusing to measure with a length mismatch"
        )
    text = Path(entry["path"]).read_text(encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if digest != entry["sha256"]:
        raise RuntimeError(f"prompt sha256 mismatch: file {digest} vs entry {entry['sha256']}")
    return text, dict(entry)


def _apply_working_set_lock(cfg: dict[str, Any]) -> dict[str, Any]:
    from seam.tools._winmem import lock_working_set, read_working_set_limits

    wslock = dict(cfg.get("working_set_lock") or {})
    mode = str(wslock.get("mode", "off"))
    if mode == "request":
        return lock_working_set(
            minimum_bytes=int(wslock["minimum_bytes"]),
            maximum_bytes=int(wslock["maximum_bytes"]),
        )
    return {
        "requested": False,
        "granted": False,
        "mode": mode,
        "before": read_working_set_limits(),
    }


def smoke(
    launch_context: str,
    *,
    arm_id: str = "B",
    n_tokens: int | None = None,
    model_spec: str | Path | None = None,
) -> dict[str, Any]:
    pid = os.getpid()
    probe_errors: dict[str, str] = {}
    python_exe = str(Path(sys.executable).resolve())

    max_new = _max_new_tokens(n_tokens)
    record: dict[str, Any] = {
        "hostname": platform.node(),
        "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "launch_context": launch_context,
        "arm_id": arm_id,
        "n_tokens": n_tokens,
        "process_id": pid,
        "session_id": _session_id(pid),
        "window_station": _window_station(probe_errors),
        "is_interactive": _is_interactive(probe_errors),
        "device_config": None,
        "compile_ok": False,
        "compile_error": None,
        "execute_ok": False,
        "execute_error": None,
        "classification": "OTHER",
        "tokens_generated": None,
        "prefill_s": None,
        "decode_tok_s": None,
        "wall_s": None,
        "ov_version": None,
        "python_exe": python_exe,
        "python_version": sys.version,
        "python_in_venv": sys.prefix != sys.base_prefix,
        "model_id": None,
        "path_head": _path_head(6),
        "peak_ws_bytes": None,
        "free_physical_mb_start": None,
        "free_physical_at_peak": None,
        "available_mb_start": None,
        "environment_start": None,
        "environment_peak": None,
        "kv_bytes_expected": None,
        "wslock_granted": None,
        "power_request": None,
        "diagnostics": {
            "context_field_errors": probe_errors,
            "config_path": str(DELTA_N_CFG),
            "prompt": None,
            "prompt_mode": "fixed" if n_tokens is None else "ladder_exact",
            "prompt_meta": None,
            "max_new_tokens": max_new,
            "loads": [],
            "traceback": None,
            "text_head": None,
            "working_set_lock": None,
            "rss_window": None,
            "ttft_source": None,
            "ttft_ns": None,
            "ttft_ns_perf_metrics": None,
            "ttft_ns_streamer": None,
        },
    }

    # Cell-start host snapshot even if compile fails later (Available governs cleanliness).
    try:
        from seam.telemetry.host_environment import capture_host_environment

        env_start = capture_host_environment()
        record["environment_start"] = env_start
        if env_start.get("available_mb") is not None:
            record["available_mb_start"] = float(env_start["available_mb"])
    except Exception as exc:
        record["environment_start"] = {"probe_error": _describe(exc)}

    power_req: Any = None
    try:
        from seam.tools._winpower import assert_system_required

        power_req = assert_system_required(
            reason="SEAM gpu_smoke; PowerRequestSystemRequired for compile+generate",
            role="gpu_smoke",
        )
        record["power_request"] = dict(power_req.record)
    except Exception as exc:
        record["power_request"] = {
            "succeeded": False,
            "error": _describe(exc),
            "role": "gpu_smoke",
        }

    try:
        return _smoke_body(record, arm_id=arm_id, n_tokens=n_tokens, model_spec=model_spec)
    finally:
        if power_req is not None:
            with contextlib.suppress(Exception):
                power_req.release()
                record["power_request"] = dict(power_req.record)


def _smoke_body(
    record: dict[str, Any],
    *,
    arm_id: str,
    n_tokens: int | None,
    model_spec: str | Path | None = None,
) -> dict[str, Any]:
    try:
        cfg = _load_delta_n_cfg()
        device_config = _arm_device_config(cfg, arm_id)
        spec_path = _model_spec_path(cfg, model_spec)
        model_id, model_dir = _resolve_model(cfg, spec_path)
        record["diagnostics"]["model_spec"] = str(spec_path)
    except Exception as exc:
        record["compile_error"] = _describe(exc)
        record["classification"] = "OTHER"
        record["diagnostics"]["traceback"] = traceback.format_exc()[:4000]
        return record

    record["device_config"] = device_config
    record["model_id"] = model_id
    max_new = _max_new_tokens(n_tokens)
    record["diagnostics"]["max_new_tokens"] = max_new

    prompt = FIXED_PROMPT
    prompt_meta: dict[str, Any] | None = None
    realized_tokens: int | None = None
    if n_tokens is not None:
        try:
            prompt, prompt_meta = _build_ladder_prompt(
                cfg=cfg, model_dir=model_dir, n_tokens=int(n_tokens)
            )
            realized_tokens = int(prompt_meta["n_tokens_realized"])
            # kv_bytes_expected filled after KV_CACHE_PRECISION readback (not assumed).
        except Exception as exc:
            record["compile_error"] = _describe(exc)
            record["classification"] = "OTHER"
            record["diagnostics"]["traceback"] = traceback.format_exc()[:4000]
            return record

    record["diagnostics"]["prompt"] = (
        prompt if n_tokens is None else f"<ladder exact n={n_tokens} chars={len(prompt)}>"
    )
    record["diagnostics"]["prompt_meta"] = prompt_meta

    try:
        lock_record = _apply_working_set_lock(cfg)
    except Exception as exc:
        record["compile_error"] = _describe(exc)
        record["classification"] = "OTHER"
        record["diagnostics"]["traceback"] = traceback.format_exc()[:4000]
        return record
    record["diagnostics"]["working_set_lock"] = lock_record
    record["wslock_granted"] = bool(lock_record.get("granted"))

    try:
        import openvino as ov
        import openvino_genai as ov_genai
    except Exception as exc:
        record["compile_error"] = _describe(exc)
        record["classification"] = "OTHER"
        record["diagnostics"]["traceback"] = traceback.format_exc()[:4000]
        return record

    try:
        record["ov_version"] = str(ov.get_version())
    except Exception as exc:
        record["ov_version"] = f"<unavailable: {_describe(exc)}>"

    pipes: dict[str, Any] = {}
    loads: list[dict[str, Any]] = []
    kv_precision_records: list[dict[str, Any]] = []
    from seam.ov_kv_precision import (
        enforce_kv_cache_precision,
        kv_bytes_per_token,
        materialize_pipeline_properties,
        requested_kv_cache_precision,
    )

    # Same load order and property application as seam.tools._delta_n_child / delta_n.run_rung.
    for entry in device_config["load_sequence"]:
        device = str(entry["device"])
        properties_raw = dict(entry.get("properties") or {})
        properties = materialize_pipeline_properties(properties_raw)
        requested_kv = requested_kv_cache_precision(properties_raw)
        try:
            core = ov.Core()
            if requested_kv is not None:
                core.set_property(device, {"KV_CACHE_PRECISION": properties["KV_CACHE_PRECISION"]})
            pipes[device] = ov_genai.LLMPipeline(str(model_dir), device, **properties)
        except Exception as exc:
            err = _describe(exc)
            loads.append(
                {
                    "device": device,
                    "properties": properties_raw,
                    "ok": False,
                    "error": err,
                }
            )
            record["diagnostics"]["loads"] = list(loads)
            record["compile_ok"] = False
            record["compile_error"] = err
            record["classification"] = _classify_error(err)
            record["diagnostics"]["traceback"] = traceback.format_exc()[:4000]
            return record
        kv_check = enforce_kv_cache_precision(device=device, requested=requested_kv, core=core)
        kv_precision_records.append(kv_check)
        loads.append(
            {
                "device": device,
                "properties": properties_raw,
                "ok": True,
                "kv_cache_precision": kv_check,
            }
        )
        record["diagnostics"]["loads"] = list(loads)
        if not kv_check["match"]:
            record["compile_ok"] = False
            record["compile_error"] = kv_check["failure_mode"]
            record["classification"] = "KV_PRECISION_MISMATCH"
            record["execute_error"] = kv_check["failure_mode"]
            return record

    record["compile_ok"] = True
    record["compile_error"] = None
    record["kv_cache_precision_readback"] = list(kv_precision_records)

    generate_device = str(device_config["generate_device"])
    if generate_device not in pipes:
        err = (
            f"generate_device {generate_device!r} not in compiled pipes "
            f"{sorted(pipes)!r}; arm {arm_id} config is inconsistent"
        )
        record["execute_ok"] = False
        record["execute_error"] = err
        record["classification"] = "OTHER"
        return record

    gen_kv = next(
        (
            k
            for k in kv_precision_records
            if (k.get("readback") or {}).get("device") == generate_device
        ),
        kv_precision_records[-1] if kv_precision_records else None,
    )
    readback_name = None
    if gen_kv is not None:
        readback_name = (gen_kv.get("readback") or {}).get("normalized")
    kv_bpt = kv_bytes_per_token(readback_name)
    record["kv_bytes_per_token"] = kv_bpt
    if realized_tokens is not None and kv_bpt is not None:
        record["kv_bytes_expected"] = realized_tokens * kv_bpt
    else:
        record["kv_bytes_expected"] = None

    from seam.backends.local_openvino import (
        _extract_metrics,
        _make_ttft_streamer,
        resolve_ttft_ns,
    )
    from seam.telemetry.host_environment import capture_host_environment
    from seam.telemetry.rss import RssSampler

    pipe = pipes[generate_device]
    sampler = RssSampler(interval_s=0.05)
    gen_cfg = ov_genai.GenerationConfig()
    gen_cfg.max_new_tokens = max_new
    gen_cfg.do_sample = False
    gen_cfg.ignore_eos = True
    gen_cfg.apply_chat_template = False
    streamer = _make_ttft_streamer(ov_genai)
    # Refresh start snapshot immediately before generate (compile may have moved memory).
    env_start = capture_host_environment()
    record["environment_start"] = env_start
    if env_start.get("available_mb") is not None:
        record["available_mb_start"] = float(env_start["available_mb"])
    sampler.start()
    t0 = time.perf_counter_ns()
    streamer.t0_ns = t0
    result: Any = None
    execute_exc: BaseException | None = None
    try:
        result = pipe.generate([prompt], gen_cfg, streamer)
    except Exception as exc:
        execute_exc = exc
    finally:
        wall_ns = time.perf_counter_ns() - t0
        window = sampler.stop()
        record["diagnostics"]["rss_window"] = window.to_record()
        record["peak_ws_bytes"] = int(window.peak_bytes)
        if window.free_physical_mb_start is not None:
            record["free_physical_mb_start"] = float(window.free_physical_mb_start)
        if window.free_physical_mb_at_peak is not None:
            record["free_physical_at_peak"] = int(
                float(window.free_physical_mb_at_peak) * 1024.0 * 1024.0
            )
        # Peak-time Available from the RSS sampler; other host fields from a post-stop snapshot
        # with available_mb overlaid so the governing counter is the true peak-time sample.
        env_peak = capture_host_environment()
        if window.available_mb_at_peak is not None:
            env_peak["available_mb"] = float(window.available_mb_at_peak)
            env_peak["available_mb_source"] = "at_rss_peak"
        else:
            env_peak["available_mb_source"] = "post_stop_snapshot"
        record["environment_peak"] = env_peak
        # Wall always recorded when generate was attempted (success or fail after start).
        record["wall_s"] = wall_ns / _NS_PER_S

    if execute_exc is not None:
        err = _describe(execute_exc)
        record["execute_ok"] = False
        record["execute_error"] = err
        record["classification"] = _classify_error(err)
        record["diagnostics"]["traceback"] = "".join(traceback.format_exception(execute_exc))[:4000]
        return record

    texts = getattr(result, "texts", None)
    text = str(texts[0]) if texts else str(result)
    record["diagnostics"]["text_head"] = text[:200]

    metrics = getattr(result, "perf_metrics", None)
    metrics_ttft_ns, _prompt_tokens, tokens_from_metrics = _extract_metrics(metrics)
    ttft_ns, ttft_source = resolve_ttft_ns(metrics_ttft_ns, streamer.ttft_ns)
    record["diagnostics"]["ttft_source"] = ttft_source
    record["diagnostics"]["ttft_ns"] = ttft_ns
    record["diagnostics"]["ttft_ns_perf_metrics"] = metrics_ttft_ns
    record["diagnostics"]["ttft_ns_streamer"] = streamer.ttft_ns

    tokens_generated: int | None = tokens_from_metrics if tokens_from_metrics else None
    if tokens_generated is None:
        # Fallback: generation completed; report the requested bound rather than invent a count.
        tokens_generated = max_new

    timing = _timing_fields(wall_ns=wall_ns, ttft_ns=ttft_ns, tokens_generated=tokens_generated)
    record["prefill_s"] = timing["prefill_s"]
    record["decode_tok_s"] = timing["decode_tok_s"]
    record["wall_s"] = timing["wall_s"]

    record["execute_ok"] = True
    record["execute_error"] = None
    record["tokens_generated"] = tokens_generated
    record["classification"] = "OK"
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--launch-context",
        required=True,
        choices=["local_console", "ssh_foreground", "ssh_detached"],
        help="how this process was started; passed in rather than inferred",
    )
    parser.add_argument(
        "-N",
        "--n-tokens",
        type=int,
        default=None,
        metavar="TOKENS",
        help=(
            "build an exact-N prompt via seam.tools.delta_n.prompt_for "
            "(filler_unit from configs/delta_n.yaml); raises max_new_tokens to 64; "
            "default: short fixed prompt with 8 tokens"
        ),
    )
    parser.add_argument(
        "--arm",
        default="B",
        help="arm id from configs/delta_n.yaml (any declared arm id; default: B)",
    )
    parser.add_argument(
        "--model-spec",
        default=None,
        help=(
            "FetchedModelSpec YAML. Default: openvino.model_spec in configs/delta_n.yaml "
            "(Qwen3-4B-int4-ov). Pass configs/models/Qwen3-8B-int4-ov.yaml to select 8B. "
            "Does not change the yaml default."
        ),
    )
    parser.add_argument("--out", type=Path, help="also write the object here")
    parser.add_argument(
        "--prompt-only",
        action="store_true",
        help=(
            "build/verify the ladder prompt for -N and exit without compiling the model "
            "(diagnostic; still writes the JSON record)"
        ),
    )
    args = parser.parse_args(argv)

    if args.n_tokens is not None and args.n_tokens < 1:
        parser.error("-N / --n-tokens must be >= 1")

    try:
        cfg = _load_delta_n_cfg()
        known_arms = _arm_ids(cfg)
    except Exception as exc:
        parser.error(f"cannot load arm ids from configs/delta_n.yaml: {exc}")
    if args.arm not in known_arms:
        parser.error(f"--arm {args.arm!r} not in configs/delta_n.yaml arms {known_arms}")

    try:
        if args.prompt_only:
            if args.n_tokens is None:
                parser.error("--prompt-only requires -N / --n-tokens")
            record = _prompt_only_record(
                args.launch_context,
                arm_id=args.arm,
                n_tokens=args.n_tokens,
                model_spec=args.model_spec,
            )
        else:
            record = smoke(
                args.launch_context,
                arm_id=args.arm,
                n_tokens=args.n_tokens,
                model_spec=args.model_spec,
            )
    except BaseException as exc:
        python_exe = str(Path(sys.executable).resolve())
        record = {
            "launch_context": args.launch_context,
            "arm_id": args.arm,
            "n_tokens": args.n_tokens,
            "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "session_id": None,
            "window_station": None,
            "is_interactive": None,
            "device_config": None,
            "compile_ok": False,
            "compile_error": _describe(exc),
            "execute_ok": False,
            "execute_error": None,
            "classification": _classify_error(_describe(exc)),
            "tokens_generated": None,
            "prefill_s": None,
            "decode_tok_s": None,
            "wall_s": None,
            "ov_version": None,
            "python_exe": python_exe,
            "model_id": None,
            "peak_ws_bytes": None,
            "free_physical_mb_start": None,
            "free_physical_at_peak": None,
            "available_mb_start": None,
            "environment_start": None,
            "environment_peak": None,
            "kv_bytes_expected": None,
            "wslock_granted": None,
            "power_request": None,
            "diagnostics": {"traceback": traceback.format_exc()[:4000]},
        }

    rendered = json.dumps(record, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")

    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(rendered, flush=True)

    # Always 0 from Python. INTERPRETER_MISSING is exit 2 from the ps1 before invoke.
    # A negative GPU result is a recorded answer, not a broken tool.
    return 0


def _prompt_only_record(
    launch_context: str,
    *,
    arm_id: str,
    n_tokens: int,
    model_spec: str | Path | None = None,
) -> dict[str, Any]:
    """Argparse / prompt-build check without GPU compile."""
    python_exe = str(Path(sys.executable).resolve())
    record: dict[str, Any] = {
        "hostname": platform.node(),
        "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "launch_context": launch_context,
        "arm_id": arm_id,
        "n_tokens": n_tokens,
        "process_id": os.getpid(),
        "compile_ok": None,
        "execute_ok": None,
        "classification": "PROMPT_ONLY",
        "python_exe": python_exe,
        "peak_ws_bytes": None,
        "free_physical_mb_start": None,
        "free_physical_at_peak": None,
        "available_mb_start": None,
        "environment_start": None,
        "environment_peak": None,
        "kv_bytes_expected": None,
        "wslock_granted": None,
        "power_request": None,
        "device_config": None,
        "model_id": None,
        "diagnostics": {
            "prompt_mode": "ladder_exact",
            "prompt_meta": None,
            "config_path": str(DELTA_N_CFG),
            "traceback": None,
        },
    }
    try:
        cfg = _load_delta_n_cfg()
        record["device_config"] = _arm_device_config(cfg, arm_id)
        spec_path = _model_spec_path(cfg, model_spec)
        model_id, model_dir = _resolve_model(cfg, spec_path)
        record["model_id"] = model_id
        record["diagnostics"]["model_spec"] = str(spec_path)
        _prompt, prompt_meta = _build_ladder_prompt(
            cfg=cfg, model_dir=model_dir, n_tokens=int(n_tokens)
        )
        record["diagnostics"]["prompt_meta"] = prompt_meta
        record["diagnostics"]["max_new_tokens"] = _max_new_tokens(n_tokens)
        # No OV load → no readback; leave null rather than assume a precision.
        record["kv_bytes_per_token"] = None
        record["kv_bytes_expected"] = None
        record["classification"] = "PROMPT_ONLY_OK"
    except Exception as exc:
        record["compile_error"] = _describe(exc)
        record["classification"] = "OTHER"
        record["diagnostics"]["traceback"] = traceback.format_exc()[:4000]
    return record


if __name__ == "__main__":
    raise SystemExit(main())
