"""Delta-prefill under session residency — one cell.

Question: when ``LLMPipeline.start_chat()`` retains a KV cache across ``generate()``
calls, is turn-2 cost the delta prefill (plus decode) rather than a full re-prefill of
``n_cached + delta``?

Design (same warm process for both turns):

* RESIDENT     ``start_chat()``; generate(n_cached); generate(delta-only); ``finish_chat()``
* NON_RESIDENT generate(n_cached) under chat; ``finish_chat()``; generate(n_cached+delta)
               in the same process (warm kernels, no retained KV)

Prompts are ladder exact-N via ``seam.tools.delta_n.prompt_for`` (thinking_off).
``GenerationConfig.apply_chat_template = False`` — same as ``smoke_gpu_exec`` — because
``prompt_for`` already renders the chat template.

Diagnostic only: writes under ``derived/delta_prefill/``, seals nothing.
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
import warnings
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DELTA_N_CFG = ROOT / "configs" / "delta_n.yaml"
OUT_DIR = ROOT / "derived" / "delta_prefill"
MAX_NEW_TOKENS = 64
_NS_PER_S = 1_000_000_000

# Retention judgment thresholds (recorded; not gates that abort the cell).
_TTFT_RATIO_MAX = 0.5  # turn2 / turn1 under RESIDENT, or RESIDENT / NON_RESIDENT peer
_WS_GROWTH_TOLERANCE = 0.5  # |observed - expected_delta_kv| / expected_delta_kv


def _load_probe_helpers() -> Any:
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
            f"configs/delta_n.yaml arm {arm_id}: cannot determine load_sequence / "
            f"generate_device (load_sequence={load_sequence!r}, "
            f"generate_device={generate_device!r})"
        )
    ov = cfg.get("openvino") or {}
    cpu_properties = dict(ov.get("cpu_properties") or {})
    arm_properties = dict(arm.get("properties") or {})
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


def _load_model_pin(cfg: dict[str, Any], spec_path: str | Path | None = None) -> dict[str, Any]:
    path = _model_spec_path(cfg, spec_path)
    if not path.is_file():
        raise ValueError(f"model spec not found: {path}")
    model = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(model, dict):
        raise ValueError(f"{path}: expected mapping")
    model["_spec_path"] = path
    return model


def _resolve_model(cfg: dict[str, Any], spec_path: str | Path | None = None) -> tuple[str, Path]:
    model = _load_model_pin(cfg, spec_path)
    path = model["_spec_path"]
    model_id = str(model.get("name") or path.stem)
    ir_dir = model.get("ir_dir")
    if not ir_dir:
        raise ValueError(f"{path}: ir_dir missing")
    model_dir = Path(str(ir_dir))
    if not model_dir.is_dir():
        raise ValueError(f"model ir_dir does not exist: {model_dir}")
    ir_sha = model.get("ir_sha256")
    if not ir_sha:
        raise ValueError(f"{path}: ir_sha256 missing (cell is not a measurement)")
    return model_id, model_dir


def _build_ladder_prompt(
    *, cfg: dict[str, Any], model_dir: Path, n_tokens: int
) -> tuple[str, dict[str, Any]]:
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


def _timing_fields(
    *,
    wall_ns: int,
    ttft_ns: int | None,
    tokens_generated: int | None,
) -> dict[str, float | None]:
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


def _gen_cfg(ov_genai: Any, *, max_new: int) -> Any:
    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = max_new
    cfg.do_sample = False
    cfg.ignore_eos = True
    # prompt_for already applied the chat template with thinking_off.
    cfg.apply_chat_template = False
    return cfg


def _is_continuous_batching_pipeline(pipe: Any) -> bool:
    return type(pipe).__name__ == "ContinuousBatchingPipeline"


def _normalize_generate_result(result: Any) -> Any:
    """CB generate returns list[GenerationResult]; LLMPipeline returns a single result."""
    if isinstance(result, (list, tuple)):
        if not result:
            raise RuntimeError("generate() returned an empty list")
        return result[0]
    return result


def _run_generate(
    pipe: Any,
    ov_genai: Any,
    prompt: str,
    gen_cfg: Any,
) -> dict[str, Any]:
    from seam.backends.local_openvino import (
        _extract_metrics,
        _make_ttft_streamer,
        resolve_ttft_ns,
    )
    from seam.telemetry.rss import RssSampler, rss_bytes_now

    sampler = RssSampler(interval_s=0.05)
    streamer = _make_ttft_streamer(ov_genai)
    sampler.start()
    t0 = time.perf_counter_ns()
    streamer.t0_ns = t0
    result: Any = None
    exc: BaseException | None = None
    try:
        # ContinuousBatchingPipeline rejects generate([prompt], scalar_cfg, streamer).
        # Use scalar-prompt + scalar GenerationConfig (API form 3), or pass
        # Sequence[GenerationConfig] with a prompt list. Scalar form is enough here.
        if _is_continuous_batching_pipeline(pipe):
            result = pipe.generate(prompt, gen_cfg, streamer)
        else:
            result = pipe.generate([prompt], gen_cfg, streamer)
        result = _normalize_generate_result(result)
    except Exception as e:
        exc = e
    finally:
        wall_ns = time.perf_counter_ns() - t0
        window = sampler.stop()
        end_ws = rss_bytes_now()

    out: dict[str, Any] = {
        "ok": exc is None,
        "error": _describe(exc) if exc is not None else None,
        "wall_ns": wall_ns,
        "peak_ws_bytes": int(window.peak_bytes),
        "end_ws_bytes": int(end_ws),
        "rss_window": window.to_record(),
        "prefill_s": None,
        "decode_tok_s": None,
        "wall_s": wall_ns / _NS_PER_S,
        "tokens_generated": None,
        "ttft_ns": None,
        "ttft_source": None,
        "ttft_ns_perf_metrics": None,
        "ttft_ns_streamer": None,
        "prompt_tokens_reported": None,
        "text_head": None,
        "traceback": None,
    }
    if exc is not None:
        out["traceback"] = "".join(traceback.format_exception(exc))[:4000]
        return out

    texts = getattr(result, "texts", None)
    if texts:
        text = str(texts[0])
    else:
        # ContinuousBatching GenerationResult exposes m_generation_ids, not texts.
        gen_ids = getattr(result, "m_generation_ids", None)
        if gen_ids:
            text = str(gen_ids[0])
        else:
            text = str(getattr(result, "text", None) or result)
    out["text_head"] = text[:200]

    metrics = getattr(result, "perf_metrics", None)
    metrics_ttft_ns, prompt_tokens, tokens_from_metrics = _extract_metrics(metrics)
    ttft_ns, ttft_source = resolve_ttft_ns(metrics_ttft_ns, streamer.ttft_ns)
    out["ttft_source"] = ttft_source
    out["ttft_ns"] = ttft_ns
    out["ttft_ns_perf_metrics"] = metrics_ttft_ns
    out["ttft_ns_streamer"] = streamer.ttft_ns
    out["prompt_tokens_reported"] = prompt_tokens if prompt_tokens else None

    tokens_generated: int | None = tokens_from_metrics if tokens_from_metrics else None
    if tokens_generated is None:
        tokens_generated = int(gen_cfg.max_new_tokens)
    out["tokens_generated"] = tokens_generated

    timing = _timing_fields(wall_ns=wall_ns, ttft_ns=ttft_ns, tokens_generated=tokens_generated)
    out["prefill_s"] = timing["prefill_s"]
    out["decode_tok_s"] = timing["decode_tok_s"]
    out["wall_s"] = timing["wall_s"]
    return out


def _judge_retention(
    *,
    mode: str,
    n_cached: int,
    delta: int,
    turn1: dict[str, Any],
    turn2: dict[str, Any],
    kv_bytes_per_token: float | None,
) -> dict[str, Any]:
    """Boolean retention judgment from TTFT + working-set growth (RESIDENT only)."""
    t1 = turn1.get("prefill_s")
    t2 = turn2.get("prefill_s")
    peak1 = turn1.get("peak_ws_bytes")
    peak2 = turn2.get("peak_ws_bytes")
    kv_unpinned = kv_bytes_per_token is None
    if kv_unpinned:
        kv_delta: float | None = None
        kv_full: float | None = None
    else:
        kv_delta = int(delta) * float(kv_bytes_per_token)
        kv_full = int(n_cached + delta) * float(kv_bytes_per_token)
    ws_growth = None
    if isinstance(peak1, int) and isinstance(peak2, int):
        ws_growth = peak2 - peak1

    ttft_ratio = None
    if isinstance(t1, int | float) and isinstance(t2, int | float) and t1 > 0:
        ttft_ratio = float(t2) / float(t1)

    # Under RESIDENT, turn2 should look like delta prefill: much cheaper than turn1,
    # and WS growth nearer to delta KV than to a full (n_cached+delta) re-seat.
    ttft_suggests = ttft_ratio is not None and ttft_ratio < _TTFT_RATIO_MAX
    ws_suggests: bool | None = False
    ws_note: str | None = None
    if kv_unpinned:
        # Cannot size expected KV without a pinned precision; do not run WS test.
        ws_suggests = None
    elif ws_growth is not None and kv_delta is not None and kv_delta > 0:
        # Growth can include activations / allocator slack; require it is closer to
        # delta KV than to full-context KV, and not a large multiple of full KV.
        dist_delta = abs(ws_growth - kv_delta)
        dist_full = abs(ws_growth - kv_full)
        ws_suggests = dist_delta < dist_full and ws_growth < kv_full * (1.0 + _WS_GROWTH_TOLERANCE)
        ws_note = (
            f"ws_growth={ws_growth} kv_delta={kv_delta} kv_full={kv_full} "
            f"dist_delta={dist_delta} dist_full={dist_full}"
        )

    if mode == "RESIDENT":
        if kv_unpinned:
            # Same TTFT-only path as when WS growth is unavailable.
            retained = bool(ttft_suggests)
            confidence = "ttft_only_kv_precision_unpinned"
            if not ttft_suggests:
                retained = False
        else:
            retained = bool(ttft_suggests and (ws_suggests or ws_growth is None))
            # If WS unavailable, TTFT alone can still suggest; mark confidence.
            confidence = "ttft+ws" if ws_growth is not None else "ttft_only"
            if ttft_suggests and not ws_suggests and ws_growth is not None:
                retained = False
                confidence = "ttft_yes_ws_no"
            elif not ttft_suggests:
                retained = False
                confidence = "ttft_no"
        if kv_unpinned:
            falsification_rule = (
                "RESIDENT turn2_prefill not materially cheaper than turn1 "
                f"(ratio < {_TTFT_RATIO_MAX})"
            )
        else:
            falsification_rule = (
                "RESIDENT turn2_prefill not materially cheaper than turn1 "
                f"(ratio < {_TTFT_RATIO_MAX}) and/or WS growth nearer full context than delta"
            )
    else:
        # NON_RESIDENT is the warm-kernel / no-KV control; retention expected false.
        retained = False
        confidence = "non_resident_control"
        # Still record whether turn2 looked cheap (would falsify the control).
        if ttft_suggests:
            confidence = "non_resident_but_turn2_cheap"
        falsification_rule = (
            "RESIDENT turn2_prefill not materially cheaper than turn1 "
            f"(ratio < {_TTFT_RATIO_MAX}) and/or WS growth nearer full context than delta"
            if not kv_unpinned
            else (
                "RESIDENT turn2_prefill not materially cheaper than turn1 "
                f"(ratio < {_TTFT_RATIO_MAX})"
            )
        )

    return {
        "cache_retained": retained,
        "confidence": confidence,
        "ttft_ratio_turn2_over_turn1": ttft_ratio,
        "ttft_ratio_threshold": _TTFT_RATIO_MAX,
        "ttft_suggests_retention": ttft_suggests,
        "ws_growth_bytes": ws_growth,
        "kv_bytes_expected_delta": kv_delta,
        "kv_bytes_expected_full_turn2": kv_full,
        "ws_suggests_retention": ws_suggests,
        "ws_note": ws_note,
        "falsification_rule": falsification_rule,
    }


def chat_api_check() -> dict[str, Any]:
    """Verify GenAI chat API surface without loading the model."""
    import openvino as ov
    import openvino_genai as ov_genai

    pipe_cls = ov_genai.LLMPipeline
    has_start = callable(getattr(pipe_cls, "start_chat", None))
    has_finish = callable(getattr(pipe_cls, "finish_chat", None))
    has_history = hasattr(ov_genai, "ChatHistory")
    dep_msgs: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Instantiating is heavy; inspect signature / docstring only.
        start_doc = (pipe_cls.start_chat.__doc__ or "").strip() if has_start else None
        finish_doc = (pipe_cls.finish_chat.__doc__ or "").strip() if has_finish else None
        for w in caught:
            dep_msgs.append(str(w.message))

    cfg = ov_genai.GenerationConfig()
    return {
        "ok": has_start and has_finish,
        "ov_version": str(ov.get_version()),
        "genai_version": getattr(ov_genai, "__version__", None),
        "has_start_chat": has_start,
        "has_finish_chat": has_finish,
        "has_ChatHistory": has_history,
        "start_chat_doc": start_doc,
        "finish_chat_doc": finish_doc,
        "apply_chat_template_default": bool(cfg.apply_chat_template),
        "deprecation_warnings_on_inspect": dep_msgs,
        "note": (
            "openvino_genai 2026.2.1 deprecates start_chat/finish_chat in favour of "
            "generate(ChatHistory); this diagnostic still uses start_chat/finish_chat."
        ),
    }


PIPELINE_TYPES = ("llm", "cb_no_eviction")

# Fixed 1-cell CB gate: must pass before any matrix that includes cb_no_eviction.
CB_TOKEN_GATE_N_CACHED = 64
CB_TOKEN_GATE_DELTA = 16
CB_TOKEN_GATE_MAX_NEW = 8


def run_cb_token_gate(
    launch_context: str,
    *,
    arm_id: str,
    model_spec: str | None,
) -> dict[str, Any]:
    """One CB cell that must emit non-empty generated text.

    Hard gate for any matrix that includes ``cb_no_eviction``. Does not measure
    equivalence; only asserts ContinuousBatchingPipeline.generate is callable
    with the smoke signature and returns text.
    """
    record = run_cell(
        launch_context,
        arm_id=arm_id,
        mode="RESIDENT",
        n_cached=CB_TOKEN_GATE_N_CACHED,
        delta=CB_TOKEN_GATE_DELTA,
        max_new_tokens=CB_TOKEN_GATE_MAX_NEW,
        model_spec=model_spec,
        pipeline_type="cb_no_eviction",
    )
    text = ""
    turn1 = record.get("turn1") or {}
    if isinstance(turn1.get("text_head"), str):
        text = turn1["text_head"]
    tokens = turn1.get("tokens_generated")
    tokens_ok = isinstance(tokens, int) and tokens >= 1
    text_ok = bool(text and text.strip())
    gate = {
        "kind": "cb_no_eviction_token_gate",
        "ok": False,
        "classification": record.get("classification"),
        "execute_error": record.get("execute_error"),
        "compile_error": record.get("compile_error"),
        "text_head": text[:200] if text else None,
        "text_nonempty": text_ok,
        "tokens_generated": tokens,
        "tokens_ok": tokens_ok,
        "n_cached": CB_TOKEN_GATE_N_CACHED,
        "delta": CB_TOKEN_GATE_DELTA,
        "max_new_tokens": CB_TOKEN_GATE_MAX_NEW,
        "pipeline_type": "cb_no_eviction",
        "arm_id": arm_id,
        "cell": record,
    }
    if record.get("classification") != "OK":
        gate["fail_reason"] = (
            f"cell classification={record.get('classification')!r} "
            f"execute_error={record.get('execute_error')!r}"
        )
        return gate
    if not text_ok or not tokens_ok:
        gate["fail_reason"] = (
            f"cb_no_eviction output insufficient: text_nonempty={text_ok} "
            f"tokens_generated={tokens!r} (need non-empty text and >=1 token)"
        )
        return gate
    gate["ok"] = True
    gate["fail_reason"] = None
    return gate


def run_cell(
    launch_context: str,
    *,
    arm_id: str,
    mode: str,
    n_cached: int,
    delta: int,
    max_new_tokens: int = MAX_NEW_TOKENS,
    model_spec: str | Path | None = None,
    pipeline_type: str = "llm",
) -> dict[str, Any]:
    mode = mode.upper()
    if mode not in {"RESIDENT", "NON_RESIDENT"}:
        raise ValueError(f"mode must be RESIDENT or NON_RESIDENT, got {mode!r}")
    pipeline_type = str(pipeline_type).strip().lower()
    if pipeline_type not in PIPELINE_TYPES:
        raise ValueError(f"pipeline_type must be one of {PIPELINE_TYPES}, got {pipeline_type!r}")
    # Matrix schedules NON_RESIDENT at every delta (see DEFECT_non_resident_max_delta_only.md).
    if mode == "NON_RESIDENT" and delta != 2000:
        pass

    pid = os.getpid()
    probe_errors: dict[str, str] = {}
    turn2_n = int(delta) if mode == "RESIDENT" else int(n_cached) + int(delta)

    record: dict[str, Any] = {
        "kind": "delta_prefill_cell",
        "hostname": platform.node(),
        "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "launch_context": launch_context,
        "arm_id": arm_id,
        "mode": mode,
        "pipeline_type": pipeline_type,
        "n_cached": int(n_cached),
        "delta": int(delta),
        "turn2_prompt_tokens": turn2_n,
        "max_new_tokens": int(max_new_tokens),
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
        "ov_version": None,
        "model_id": None,
        "python_exe": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "path_head": _path_head(6),
        # Filled from KV_CACHE_PRECISION readback after compile; null until then.
        "kv_bytes_per_token": None,
        "kv_bytes_expected_turn1": None,
        "kv_bytes_expected_turn2": None,
        "environment_start": None,
        "environment_peak": None,
        "wslock_granted": None,
        "power_request": None,
        "turn1": None,
        "turn2": None,
        "cache_retention": None,
        "diagnostics": {
            "context_field_errors": probe_errors,
            "config_path": str(DELTA_N_CFG),
            "prompt_mode": "ladder_exact",
            "prompt_meta_turn1": None,
            "prompt_meta_turn2": None,
            "loads": [],
            "chat_api": None,
            "working_set_lock": None,
            "traceback": None,
        },
    }

    try:
        from seam.telemetry.host_environment import capture_host_environment

        env_start = capture_host_environment()
        record["environment_start"] = env_start
    except Exception as exc:
        record["environment_start"] = {"probe_error": _describe(exc)}

    power_req: Any = None
    try:
        from seam.tools._winpower import assert_system_required

        power_req = assert_system_required(
            reason="SEAM delta_prefill; PowerRequestSystemRequired for compile+generate",
            role="delta_prefill",
        )
        record["power_request"] = dict(power_req.record)
    except Exception as exc:
        record["power_request"] = {
            "succeeded": False,
            "error": _describe(exc),
            "role": "delta_prefill",
        }

    try:
        return _run_cell_body(
            record,
            arm_id=arm_id,
            mode=mode,
            n_cached=n_cached,
            delta=delta,
            max_new_tokens=max_new_tokens,
            model_spec=model_spec,
            pipeline_type=pipeline_type,
        )
    finally:
        if power_req is not None:
            with contextlib.suppress(Exception):
                power_req.release()
                record["power_request"] = dict(power_req.record)


def _run_cell_body(
    record: dict[str, Any],
    *,
    arm_id: str,
    mode: str,
    n_cached: int,
    delta: int,
    max_new_tokens: int,
    model_spec: str | Path | None = None,
    pipeline_type: str = "llm",
) -> dict[str, Any]:
    try:
        cfg = _load_delta_n_cfg()
        device_config = _arm_device_config(cfg, arm_id)
        pin = _load_model_pin(cfg, model_spec)
        spec_path = pin["_spec_path"]
        model_id, model_dir = _resolve_model(cfg, spec_path)
        record["diagnostics"]["model_spec"] = str(spec_path)
        record["diagnostics"]["ir_sha256"] = str(pin.get("ir_sha256") or "")
        if not record["diagnostics"]["ir_sha256"]:
            raise ValueError(f"{spec_path}: ir_sha256 missing (cell is not a measurement)")
        record["diagnostics"]["chat_api"] = chat_api_check()
    except Exception as exc:
        record["compile_error"] = _describe(exc)
        record["diagnostics"]["traceback"] = traceback.format_exc()[:4000]
        return record

    record["device_config"] = device_config
    record["model_id"] = model_id

    turn2_n = int(delta) if mode == "RESIDENT" else int(n_cached) + int(delta)
    try:
        prompt1, meta1 = _build_ladder_prompt(cfg=cfg, model_dir=model_dir, n_tokens=int(n_cached))
        prompt2, meta2 = _build_ladder_prompt(cfg=cfg, model_dir=model_dir, n_tokens=turn2_n)
    except Exception as exc:
        record["compile_error"] = _describe(exc)
        record["diagnostics"]["traceback"] = traceback.format_exc()[:4000]
        return record

    record["diagnostics"]["prompt_meta_turn1"] = meta1
    record["diagnostics"]["prompt_meta_turn2"] = meta2

    try:
        lock_record = _apply_working_set_lock(cfg)
    except Exception as exc:
        record["compile_error"] = _describe(exc)
        record["diagnostics"]["traceback"] = traceback.format_exc()[:4000]
        return record
    record["diagnostics"]["working_set_lock"] = lock_record
    record["wslock_granted"] = bool(lock_record.get("granted"))

    try:
        import openvino as ov
        import openvino_genai as ov_genai
    except Exception as exc:
        record["compile_error"] = _describe(exc)
        record["diagnostics"]["traceback"] = traceback.format_exc()[:4000]
        return record

    try:
        record["ov_version"] = str(ov.get_version())
    except Exception as exc:
        record["ov_version"] = f"<unavailable: {_describe(exc)}>"

    from seam.ov_kv_precision import (
        enforce_kv_cache_precision,
        kv_bytes_per_token,
        materialize_pipeline_properties,
        requested_kv_cache_precision,
    )

    pipes: dict[str, Any] = {}
    loads: list[dict[str, Any]] = []
    kv_precision_records: list[dict[str, Any]] = []
    for entry in device_config["load_sequence"]:
        device = str(entry["device"])
        properties_raw = dict(entry.get("properties") or {})
        properties = materialize_pipeline_properties(properties_raw)
        requested_kv = requested_kv_cache_precision(properties_raw)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                core = ov.Core()
                if requested_kv is not None:
                    core.set_property(
                        device, {"KV_CACHE_PRECISION": properties["KV_CACHE_PRECISION"]}
                    )
                if pipeline_type == "cb_no_eviction":
                    # N-1 / axis-5 gate: ContinuousBatchingPipeline with eviction
                    # explicitly OFF — isolates the CB execution path from plain
                    # LLMPipeline without enabling CacheEvictionConfig.
                    if not hasattr(ov_genai, "ContinuousBatchingPipeline"):
                        raise RuntimeError(
                            "openvino_genai.ContinuousBatchingPipeline missing; "
                            "cannot run cb_no_eviction cell"
                        )
                    sched = ov_genai.SchedulerConfig()
                    sched.use_cache_eviction = False
                    pipes[device] = ov_genai.ContinuousBatchingPipeline(
                        str(model_dir),
                        sched,
                        device,
                        properties,
                    )
                    record["diagnostics"]["pipeline_construction"] = {
                        "class": "ContinuousBatchingPipeline",
                        "use_cache_eviction": False,
                        "scheduler_config_explicit": True,
                    }
                else:
                    pipes[device] = ov_genai.LLMPipeline(str(model_dir), device, **properties)
                    record["diagnostics"]["pipeline_construction"] = {
                        "class": "LLMPipeline",
                        "use_cache_eviction": None,
                        "scheduler_config_explicit": False,
                        "note": "plain LLMPipeline (corpus path); no SchedulerConfig passed",
                    }
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
            record["compile_error"] = err
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
        record["kv_cache_precision_readback"] = list(kv_precision_records)
        if not kv_check["match"]:
            record["compile_error"] = kv_check["failure_mode"]
            record["classification"] = "KV_PRECISION_MISMATCH"
            record["execute_error"] = kv_check["failure_mode"]
            return record

    record["compile_ok"] = True
    record["kv_cache_precision_readback"] = list(kv_precision_records)
    generate_device = str(device_config["generate_device"])
    if generate_device not in pipes:
        err = (
            f"generate_device {generate_device!r} not in compiled pipes "
            f"{sorted(pipes)!r}; arm {arm_id} config is inconsistent"
        )
        record["execute_error"] = err
        return record

    # Bytes/token from generate-device readback (same source as manifest readback).
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
    if kv_bpt is not None:
        record["kv_bytes_expected_turn1"] = int(n_cached) * kv_bpt
        record["kv_bytes_expected_turn2"] = turn2_n * kv_bpt
    else:
        record["kv_bytes_expected_turn1"] = None
        record["kv_bytes_expected_turn2"] = None

    from seam.telemetry.host_environment import capture_host_environment

    pipe = pipes[generate_device]
    gen_cfg = _gen_cfg(ov_genai, max_new=max_new_tokens)

    # Refresh host snapshot immediately before timed turns.
    env_start = capture_host_environment()
    record["environment_start"] = env_start

    turn1: dict[str, Any] | None = None
    turn2: dict[str, Any] | None = None
    chat_started = False

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            pipe.start_chat()
            chat_started = True

        turn1 = _run_generate(pipe, ov_genai, prompt1, gen_cfg)
        turn1["role"] = "turn1_n_cached"
        turn1["prompt_tokens_requested"] = int(n_cached)
        turn1["kv_bytes_expected"] = int(n_cached) * kv_bpt if kv_bpt is not None else None
        record["turn1"] = turn1

        if not turn1["ok"]:
            record["execute_error"] = turn1.get("error")
            record["classification"] = "OTHER"
            return record

        if mode == "NON_RESIDENT":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                pipe.finish_chat()
            chat_started = False
            # Warm kernels, empty KV: full re-prefill of n_cached+delta.
            turn2 = _run_generate(pipe, ov_genai, prompt2, gen_cfg)
        else:
            # RESIDENT: KV retained; prefill only the delta-sized prompt.
            turn2 = _run_generate(pipe, ov_genai, prompt2, gen_cfg)

        turn2["role"] = "turn2_delta_resident" if mode == "RESIDENT" else "turn2_full_non_resident"
        turn2["prompt_tokens_requested"] = (
            int(delta) if mode == "RESIDENT" else int(n_cached) + int(delta)
        )
        turn2["kv_bytes_expected"] = (
            turn2["prompt_tokens_requested"] * kv_bpt if kv_bpt is not None else None
        )
        record["turn2"] = turn2

        if not turn2["ok"]:
            record["execute_error"] = turn2.get("error")
            record["classification"] = "OTHER"
            return record

        record["execute_ok"] = True
        record["classification"] = "OK"
        record["cache_retention"] = _judge_retention(
            mode=mode,
            n_cached=n_cached,
            delta=delta,
            turn1=turn1,
            turn2=turn2,
            kv_bytes_per_token=kv_bpt,
        )
    except Exception as exc:
        record["execute_error"] = _describe(exc)
        record["diagnostics"]["traceback"] = traceback.format_exc()[:4000]
        record["classification"] = "OTHER"
    finally:
        if chat_started:
            with contextlib.suppress(Exception), warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                pipe.finish_chat()
        try:
            env_peak = capture_host_environment()
            # Prefer peak Available from the hotter turn's RSS window when present.
            peak_avail = None
            for tr in (turn2, turn1):
                if not tr:
                    continue
                rw = tr.get("rss_window") or {}
                if rw.get("available_mb_at_peak") is not None:
                    peak_avail = float(rw["available_mb_at_peak"])
                    break
            if peak_avail is not None:
                env_peak["available_mb"] = peak_avail
                env_peak["available_mb_source"] = "at_rss_peak_hotter_turn"
            else:
                env_peak["available_mb_source"] = "post_stop_snapshot"
            record["environment_peak"] = env_peak
        except Exception as exc:
            record["environment_peak"] = {"probe_error": _describe(exc)}

    return record


def _artifact_name(
    *,
    launch_context: str,
    arm_id: str,
    mode: str,
    n_cached: int,
    delta: int,
    repeat: int | None,
    pipeline_type: str = "llm",
) -> str:
    pipe = pipeline_type if pipeline_type != "llm" else "llm"
    base = f"{launch_context}_arm{arm_id}_{mode}_nc{n_cached}_d{delta}_{pipe}"
    if repeat is not None and repeat >= 0:
        base = f"{base}_r{repeat}"
    return f"{base}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--launch-context",
        choices=["local_console", "ssh_foreground", "ssh_detached"],
        help="how this process was started; required except for --chat-api-check",
    )
    parser.add_argument("--arm", default="A", help="arm id from configs/delta_n.yaml")
    parser.add_argument(
        "--model-spec",
        default=None,
        help=(
            "FetchedModelSpec YAML. Default: openvino.model_spec in configs/delta_n.yaml "
            "(Qwen3-4B-int4-ov). Pass configs/models/Qwen3-8B-int4-ov.yaml to select 8B. "
            "Does not change the yaml default."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["RESIDENT", "NON_RESIDENT", "resident", "non_resident"],
        default="RESIDENT",
    )
    parser.add_argument("--n-cached", type=int, default=12000)
    parser.add_argument("--delta", type=int, default=2000)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--repeat", type=int, default=-1, help="repeat index for artifact name")
    parser.add_argument("--out", type=Path, help="write JSON here (default under derived/)")
    parser.add_argument(
        "--tag",
        default="",
        help="optional prefix for default artifact name (matrix tag)",
    )
    parser.add_argument(
        "--chat-api-check",
        action="store_true",
        help="verify start_chat/finish_chat surface and exit (no model load)",
    )
    parser.add_argument(
        "--prompt-only",
        action="store_true",
        help="build/verify ladder prompts for n_cached and turn2 size; no compile",
    )
    parser.add_argument(
        "--pipeline-type",
        choices=list(PIPELINE_TYPES),
        default="llm",
        help=(
            "llm = plain LLMPipeline (corpus path). "
            "cb_no_eviction = ContinuousBatchingPipeline with "
            "SchedulerConfig.use_cache_eviction=False (N-1 equivalence arm)."
        ),
    )
    parser.add_argument(
        "--cb-token-gate",
        action="store_true",
        help=(
            "Run the mandatory 1-cell cb_no_eviction generate smoke (non-empty "
            "text). Exit 0 only if the gate passes. Required before any matrix "
            "that includes cb_no_eviction."
        ),
    )
    args = parser.parse_args(argv)

    if args.chat_api_check:
        result = chat_api_check()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("ok") else 2

    if not args.launch_context:
        parser.error("--launch-context is required (except with --chat-api-check)")

    if args.cb_token_gate:
        try:
            cfg = _load_delta_n_cfg()
            known = _arm_ids(cfg)
        except Exception as exc:
            parser.error(f"cannot load configs/delta_n.yaml: {exc}")
        if args.arm not in known:
            parser.error(f"--arm {args.arm!r} not in {known}")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = (
            args.out
            if args.out
            else OUT_DIR / (f"cb_token_gate_{args.launch_context}_arm{args.arm}.json")
        )
        gate = run_cb_token_gate(
            args.launch_context,
            arm_id=args.arm,
            model_spec=args.model_spec,
        )
        # Persist gate without embedding the full nested cell twice under a
        # different key shape — keep cell for diagnosis.
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary = {
            "event": "cb_no_eviction_token_gate",
            "ok": gate["ok"],
            "path": str(out_path),
            "fail_reason": gate.get("fail_reason"),
            "text_nonempty": gate.get("text_nonempty"),
            "classification": gate.get("classification"),
        }
        print(json.dumps(summary, sort_keys=True), flush=True)
        return 0 if gate["ok"] else 2

    if args.n_cached < 1 or args.delta < 1:
        parser.error("--n-cached and --delta must be >= 1")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be >= 1")

    mode = str(args.mode).upper()
    if mode == "NON_RESIDENT":
        mode = "NON_RESIDENT"
    elif mode == "RESIDENT":
        mode = "RESIDENT"

    try:
        cfg = _load_delta_n_cfg()
        known = _arm_ids(cfg)
    except Exception as exc:
        parser.error(f"cannot load configs/delta_n.yaml: {exc}")
    if args.arm not in known:
        parser.error(f"--arm {args.arm!r} not in {known}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    repeat = args.repeat if args.repeat >= 0 else None
    default_name = _artifact_name(
        launch_context=args.launch_context,
        arm_id=args.arm,
        mode=mode,
        n_cached=args.n_cached,
        delta=args.delta,
        repeat=repeat,
        pipeline_type=args.pipeline_type,
    )
    if args.tag:
        default_name = f"{args.tag}_{default_name}"
    out_path = args.out if args.out else OUT_DIR / default_name

    if args.prompt_only:
        pin = _load_model_pin(cfg, args.model_spec)
        model_id, model_dir = _resolve_model(cfg, pin["_spec_path"])
        turn2_n = args.delta if mode == "RESIDENT" else args.n_cached + args.delta
        p1, m1 = _build_ladder_prompt(cfg=cfg, model_dir=model_dir, n_tokens=args.n_cached)
        p2, m2 = _build_ladder_prompt(cfg=cfg, model_dir=model_dir, n_tokens=turn2_n)
        record = {
            "kind": "delta_prefill_prompt_only",
            "launch_context": args.launch_context,
            "arm_id": args.arm,
            "mode": mode,
            "n_cached": args.n_cached,
            "delta": args.delta,
            "turn2_prompt_tokens": turn2_n,
            "model_id": model_id,
            "model_spec": str(pin["_spec_path"]),
            "ir_sha256": str(pin.get("ir_sha256") or ""),
            "prompt_meta_turn1": m1,
            "prompt_meta_turn2": m2,
            "prompt_chars_turn1": len(p1),
            "prompt_chars_turn2": len(p2),
            "classification": "OK",
        }
        out_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "event": "wrote",
                    "path": str(out_path),
                    **{
                        k: record[k]
                        for k in (
                            "mode",
                            "n_cached",
                            "delta",
                            "turn2_prompt_tokens",
                            "classification",
                        )
                    },
                },
                sort_keys=True,
            )
        )
        return 0

    record = run_cell(
        args.launch_context,
        arm_id=args.arm,
        mode=mode,
        n_cached=args.n_cached,
        delta=args.delta,
        max_new_tokens=args.max_new_tokens,
        model_spec=args.model_spec,
        pipeline_type=args.pipeline_type,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "event": "delta_prefill_cell",
        "path": str(out_path),
        "arm_id": record.get("arm_id"),
        "mode": record.get("mode"),
        "pipeline_type": record.get("pipeline_type"),
        "n_cached": record.get("n_cached"),
        "delta": record.get("delta"),
        "classification": record.get("classification"),
        "turn1_prefill_s": (record.get("turn1") or {}).get("prefill_s"),
        "turn2_prefill_s": (record.get("turn2") or {}).get("prefill_s"),
        "cache_retained": (record.get("cache_retention") or {}).get("cache_retained"),
        "peak_ws_turn1": (record.get("turn1") or {}).get("peak_ws_bytes"),
        "peak_ws_turn2": (record.get("turn2") or {}).get("peak_ws_bytes"),
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if record.get("classification") == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
