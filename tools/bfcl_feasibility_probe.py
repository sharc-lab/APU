#!/usr/bin/env python3
"""BFCL feasibility probe — acquire real BFCL, tokenize, optional gpu_only generate, score.

Diagnostic only. Not a labelling harness. Not a seal path.

Modes:
  acquire_tokenize              — inventory + token profiles + AST/multi_turn gold selftests
  run_gpu                       — A2 mix: single-turn AST + multi_turn first-turn (unscored)
  multi_turn_gold_selftest      — A3: multi_turn_checker gold path on 20 multi_turn_base
  run_gpu_multi_turn            — A3: full agent loop + multi_turn_checker on 20 entries
  run_cloud_multi_turn          — same 20 entries + checker via Anthropic native tool-use
  run_session_residency         — session-level RESIDENT vs NON_RESIDENT on real BFCL multi-turn
  session_residency_cold_control — positive control: NON_RESIDENT turn-2 must be full cold prefill
  session_residency_render_smoke — offline first-turn token equivalence (ChatHistory fix)
  attention_window_inventory    — offline: CacheEvictionConfig / sink+window surface
  kv_precision_property_smoke   — offline: Core set/get KV_CACHE_PRECISION f16/u8/u4
  run_gpu_kv_precision          — gpu_only AST accuracy at KV_CACHE_PRECISION f16/u8/u4
  run_npu_load                  — NPU load ladder for the resolved local IR + MAX_PROMPT_LEN

Default mode (acquire_tokenize / multi_turn_gold_selftest) needs no GPU.
GPU / NPU load modes require a clean host (Cursor/Chrome closed; Available >= 7000 MB).
Optional --model-spec overrides configs/delta_n.yaml openvino.model_spec (4B default).
"""

from __future__ import annotations

import argparse
import ast as py_ast
import contextlib
import ctypes
import hashlib
import json
import os
import re
import sys
import time
import uuid
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.phase_timers import finalize_phase_timers  # noqa: E402

BFCL_UNPACKED = (
    ROOT / "apu_characterization" / "out" / "cap01" / "live_sources" / "bfcl-wheel" / "unpacked"
)
BFCL_DATA = BFCL_UNPACKED / "bfcl_eval" / "data"
BFCL_ANSWERS = BFCL_DATA / "possible_answer"
MODEL_SPEC = ROOT / "configs" / "models" / "Qwen3-4B-int4-ov.yaml"
MODEL_DIR = ROOT / "models" / "Qwen3-4B-int4-ov"
DELTA_N_CFG = ROOT / "configs" / "delta_n.yaml"
DEFAULT_OUT = ROOT / "derived" / "bfcl_feasibility"
DEFAULT_OUT_XLAM = ROOT / "derived" / "bfcl_feasibility_xlam"
# Modes that load a local IR (tokenizer and/or LLMPipeline). --model-spec applies here.
LOCAL_MODEL_MODES = frozenset(
    {
        "acquire_tokenize",
        "run_gpu",
        "run_gpu_multi_turn",
        "run_gpu_multi_turn_xlam",
        "run_session_residency",
        "session_residency_cold_control",
        "session_residency_render_smoke",
        "run_gpu_kv_precision",
        "run_npu_load",
    }
)
PROMPT_FORMAT_MULTI_TURN_AGENT = "bfcl_tools_style_multi_turn_agent"
PROMPT_FORMAT_MULTI_TURN_XLAM = "bfcl_prompting_python_ast_xlam"
_MODEL_SPEC_OVERRIDE_ACTIVE = False
_EXPECTED_IR_DIR: Path | None = None
_RESOLVED_SPEC: dict[str, Any] | None = None

# Probe mix (A2): 10 single-turn (simple+parallel) + 10 multi-turn first-turn.
SINGLE_TURN_FILES = (
    ("simple_python", "BFCL_v4_simple_python.json", 5),
    ("parallel", "BFCL_v4_parallel.json", 5),
)
# Dispatch E KV-precision quality: 20 AST entries from simple_python + parallel.
AST_PRECISION_FILES = (
    ("simple_python", "BFCL_v4_simple_python.json", 10),
    ("parallel", "BFCL_v4_parallel.json", 10),
)
MULTI_TURN_FILES = (("multi_turn_base", "BFCL_v4_multi_turn_base.json", 10),)
# A3: multi_turn_base prefix (file holds exactly 200); full agent loop + multi_turn_checker.
MULTI_TURN_PROBE_FILES = (("multi_turn_base", "BFCL_v4_multi_turn_base.json", 200),)
MAXIMUM_STEP_LIMIT = 20
KV_PRECISION_LEVELS = ("f16", "u8", "u4")
NPU_DEFAULT_MAX_PROMPT_LEN = 2048
NPU_DEFAULT_PREFILL_CHUNK = 1024
RESIDENCY_MODES = ("RESIDENT", "NON_RESIDENT")
SLO_TTFT_S = 10.0
SLO_DECODE_TOK_S = 6.0
_NS_PER_S = 1_000_000_000
# Paired session-residency cells use this seed for entry selection / order (no sampling).
SESSION_RESIDENCY_SEED = 20260810
# Positive-control floors: turn-2 NON_RESIDENT TTFT must be cold-scale, not delta-scale.
# Contaminated (prefix-cache hit) observations: gpu_only ~0.13 s; arm A ~2.55 s.
# Cold references: gpu_only ~2.6 s; arm A ~50 s at ~3600 tokens.
NON_RESIDENT_COLD_TTFT_FLOOR_S = {
    "gpu_only": 1.0,
    "A": 20.0,
}
# Turn-2 TTFT must be at least this fraction of length-scaled turn-1 cold TTFT.
NON_RESIDENT_COLD_TTFT_RATIO_MIN = 0.40
# Fail-loud if turn-2 is clearly warm relative to turn-1 (prefix reuse).
NON_RESIDENT_WARM_TTFT_RATIO_MAX = 0.35

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
# Official QwenFCHandler uses newline-bounded tool_call bodies; keep both.
TOOL_CALL_RE_QWEN = re.compile(r"<tool_call>\n(.*?)\n</tool_call>", re.DOTALL)
# Empirical enable_thinking=False check (AM-024): generation must not open a think block.
THINK_OPEN_RE = re.compile(r"<think>", re.IGNORECASE)
IR_SHA256_EXPECTED = "c1821f29332faa48871c7f16426a21443fbc701fa4e4c8a581a8c51ab7bf2cb2"
PINNED_MULTI_TURN_ENTRIES = "multi_turn_probe_entries.json"
CLOUD_DEFAULT_MODEL = "claude-sonnet-5"
CLOUD_USD_PER_MTOK_IN = 3.0
CLOUD_USD_PER_MTOK_OUT = 15.0
MULTI_TURN_CLASS_TO_FILE = {
    "GorillaFileSystem": "gorilla_file_system.json",
    "MathAPI": "math_api.json",
    "MessageAPI": "message_api.json",
    "TwitterAPI": "posting_api.json",
    "TicketAPI": "ticket_api.json",
    "TradingBot": "trading_bot.json",
    "TravelAPI": "travel_booking.json",
    "VehicleControlAPI": "vehicle_control.json",
    "WebSearchAPI": "web_search.json",
    "MemoryAPI_kv": "memory_kv.json",
    "MemoryAPI_vector": "memory_vector.json",
    "MemoryAPI_rec_sum": "memory_rec_sum.json",
}


def _load_delta_n_cfg() -> dict[str, Any]:
    cfg = yaml.safe_load(DELTA_N_CFG.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise TypeError("configs/delta_n.yaml: expected mapping")
    return cfg


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


def apply_model_spec(override: str | Path | None) -> dict[str, Any]:
    """Bind MODEL_SPEC / MODEL_DIR from --model-spec or delta_n.yaml. Same errors as smoke."""
    global MODEL_SPEC, MODEL_DIR, _MODEL_SPEC_OVERRIDE_ACTIVE, _EXPECTED_IR_DIR, _RESOLVED_SPEC
    cfg = _load_delta_n_cfg()
    path = _model_spec_path(cfg, override)
    model_dir = _resolve_model(cfg, override)[1]
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    MODEL_SPEC = path
    MODEL_DIR = model_dir
    _EXPECTED_IR_DIR = model_dir.resolve()
    _RESOLVED_SPEC = spec if isinstance(spec, dict) else {}
    _MODEL_SPEC_OVERRIDE_ACTIVE = override is not None
    return _RESOLVED_SPEC


def model_identity_block() -> dict[str, Any]:
    """Fields that must appear on every model-loading report (run_id-at-point-of-use)."""
    spec = _RESOLVED_SPEC
    if spec is None:
        spec = yaml.safe_load(MODEL_SPEC.read_text(encoding="utf-8"))
        if not isinstance(spec, dict):
            spec = {}
    block: dict[str, Any] = {
        "name": spec.get("name"),
        "ir_sha256": spec.get("ir_sha256"),
        "ir_dir": str(MODEL_DIR),
        "revision": spec.get("revision"),
        "publisher_quantization": spec.get("publisher_quantization"),
        "spec": str(MODEL_SPEC),
    }
    if spec.get("note") is not None:
        block["note"] = spec["note"]
    return block


def _model_report_fields(**extra: Any) -> dict[str, Any]:
    spec_text = MODEL_SPEC.read_text(encoding="utf-8") if MODEL_SPEC.is_file() else ""
    identity = model_identity_block()
    pin = identity.get("ir_sha256")
    base = {
        **identity,
        "ir_sha256_pin": pin,
        "ir_sha256_pin_present_in_spec": bool(pin) and str(pin) in spec_text,
    }
    base.update(extra)
    return base


def _assert_load_path(path: str | Path, *, what: str) -> str:
    """When --model-spec is set, refuse any load path other than that spec's ir_dir."""
    resolved = Path(path).resolve()
    if not _MODEL_SPEC_OVERRIDE_ACTIVE:
        return str(path)
    if _EXPECTED_IR_DIR is None:
        raise RuntimeError("FATAL: --model-spec set but ir_dir was not resolved")
    if resolved != _EXPECTED_IR_DIR:
        raise RuntimeError(
            f"FATAL: {what} would load {resolved}; --model-spec ir_dir is "
            f"{_EXPECTED_IR_DIR}. Refusing to run a quality comparison against the wrong IR."
        )
    return str(resolved)


def _assert_pipeline_matches_ir(pipe: Any) -> None:
    if not _MODEL_SPEC_OVERRIDE_ACTIVE:
        return
    if _EXPECTED_IR_DIR is None:
        raise RuntimeError("FATAL: --model-spec set but ir_dir was not resolved")
    expected = _EXPECTED_IR_DIR
    for attr in ("models_path", "model_path", "model_dir", "_model_path", "_models_path"):
        val = getattr(pipe, attr, None)
        if val is None:
            continue
        try:
            got = Path(str(val)).resolve()
        except OSError:
            continue
        if got == expected or expected in got.parents or got.parent == expected:
            return
        raise RuntimeError(
            f"FATAL: LLMPipeline.{attr}={got} is not --model-spec ir_dir {expected}. "
            "Refusing to run a quality comparison against the wrong IR."
        )
    # Constructor-path assertion already ran; some GenAI builds expose no path attr.


def _hf_tokenizer() -> Any:
    from transformers import AutoTokenizer

    path = _assert_load_path(MODEL_DIR, what="AutoTokenizer")
    return AutoTokenizer.from_pretrained(path)


def _genai_tokenizer(ov_genai: Any) -> Any:
    path = _assert_load_path(MODEL_DIR, what="Tokenizer")
    return ov_genai.Tokenizer(path)


def _make_llm_pipeline(ov_genai: Any, device: str, *args: Any, **kwargs: Any) -> Any:
    path = _assert_load_path(MODEL_DIR, what="LLMPipeline")
    pipe = ov_genai.LLMPipeline(path, device, *args, **kwargs)
    _assert_pipeline_matches_ir(pipe)
    return pipe


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _bfcl_function_to_openai_tool(fn: dict[str, Any]) -> dict[str, Any]:
    """Map BFCL function schema → OpenAI tool dict for Qwen chat_template tools=."""
    params = dict(fn.get("parameters") or {})
    if params.get("type") == "dict":
        params = {**params, "type": "object"}
    return {
        "type": "function",
        "function": {
            "name": fn["name"],
            "description": fn.get("description", ""),
            "parameters": params,
        },
    }


def _iter_multi_turn_func_docs(raw_entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Load raw BFCL function docs for an entry's involved_classes."""
    doc_dir = BFCL_DATA / "multi_turn_func_doc"
    excluded = set(raw_entry.get("excluded_function") or [])
    rows: list[dict[str, Any]] = []
    for cls in raw_entry.get("involved_classes") or []:
        fname = MULTI_TURN_CLASS_TO_FILE.get(str(cls))
        if not fname:
            continue
        path = doc_dir / fname
        if not path.is_file():
            continue
        for fn in _read_jsonl(path):
            if fn.get("name") in excluded:
                continue
            rows.append(
                {
                    "involved_class": str(cls),
                    "source_file": fname,
                    "bfcl_function": fn,
                }
            )
    return rows


def _load_multi_turn_tools(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve multi-turn class docs into OpenAI tool schemas."""
    return [
        _bfcl_function_to_openai_tool(row["bfcl_function"])
        for row in _iter_multi_turn_func_docs(entry)
    ]


def _concretize(node: Any) -> Any:
    if isinstance(node, dict):
        return {
            k: (_materialize_param(v) if isinstance(v, list) else _concretize(v))
            for k, v in node.items()
        }
    if isinstance(node, list):
        return [_concretize(x) for x in node]
    return node


def _materialize_param(alts: Any) -> Any:
    if not isinstance(alts, list) or not alts:
        return ""
    return _concretize(alts[0])


def materialize_gold(reference: list) -> list[dict[str, Any]]:
    concrete: list[dict[str, Any]] = []
    for call in reference:
        name, params = next(iter(call.items()))
        materialized: dict[str, Any] = {}
        for key, value in params.items():
            chosen = _materialize_param(value)
            if chosen == "":
                continue
            materialized[key] = chosen
        concrete.append({name: materialized})
    return concrete


def extract_tool_calls_ast(text: str) -> list[dict[str, Any]] | None:
    """Parse Qwen <tool_call> JSON into BFCL AST list, or None if unparseable."""
    matches = TOOL_CALL_RE.findall(text)
    if not matches:
        # Fallback: raw JSON list/object
        stripped = text.strip()
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if isinstance(obj, dict) and "name" in obj:
            params = obj.get("arguments", obj.get("parameters", {}))
            return [{str(obj["name"]): params}]
        if isinstance(obj, list):
            out = []
            for item in obj:
                if isinstance(item, dict) and "name" in item:
                    params = item.get("arguments", item.get("parameters", {}))
                    out.append({str(item["name"]): params})
                elif isinstance(item, dict) and len(item) == 1:
                    out.append(item)
            return out or None
        return None
    out: list[dict[str, Any]] = []
    for raw in matches:
        try:
            call = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(call, dict) or "name" not in call:
            return None
        params = call.get("arguments", call.get("parameters", {}))
        if not isinstance(params, dict):
            return None
        out.append({str(call["name"]): params})
    return out


def _select_from_spec(spec: tuple[tuple[str, str, int], ...], *, kind: str) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for category, fname, n in spec:
        questions = _read_jsonl(BFCL_DATA / fname)
        answers = {str(a["id"]): a["ground_truth"] for a in _read_jsonl(BFCL_ANSWERS / fname)}
        for entry in questions[:n]:
            eid = str(entry["id"])
            row: dict[str, Any] = {
                "id": eid,
                "category": category,
                "kind": kind,
                "question": entry["question"],
                "function": entry.get("function") or [],
                "reference": answers[eid],
                "raw_entry": entry,
            }
            if kind == "multi_turn":
                row["function"] = []
                row["tools_from_docs"] = True
            selected.append(row)
    return selected


def select_entries() -> list[dict[str, Any]]:
    """A2 mix: 5 simple_python + 5 parallel + 10 multi_turn_base."""
    return _select_from_spec(SINGLE_TURN_FILES, kind="single_turn") + _select_from_spec(
        MULTI_TURN_FILES, kind="multi_turn"
    )


def select_ast_precision_entries() -> list[dict[str, Any]]:
    """Dispatch E: 10 simple_python + 10 parallel (AST-scored only)."""
    return _select_from_spec(AST_PRECISION_FILES, kind="single_turn")


def select_multi_turn_entries() -> list[dict[str, Any]]:
    """A3: multi_turn_base file prefix (MULTI_TURN_PROBE_FILES cap); no difficulty/API filter."""
    return _select_from_spec(MULTI_TURN_PROBE_FILES, kind="multi_turn")


def _ov_type_for_kv_precision(name: str) -> Any:
    from openvino import Type

    mapping = {
        "f16": Type.f16,
        "u8": Type.u8,
        "u4": Type.u4,
    }
    if name not in mapping:
        raise ValueError(
            f"unsupported KV precision {name!r}; expected one of {KV_PRECISION_LEVELS}"
        )
    return mapping[name]


def _type_to_precision_name(value: Any) -> str:
    if hasattr(value, "to_string") and callable(value.to_string):
        raw = str(value.to_string()).strip().lower()
    else:
        raw = str(value).strip().lower()
        # str(Type) is often "<Type: 'uint8_t'>"; peel the quoted token.
        if "'" in raw:
            raw = raw.split("'")[1].strip().lower()
    aliases = {
        "float16": "f16",
        "f16": "f16",
        "uint8_t": "u8",
        "u8": "u8",
        "uint8": "u8",
        "uint4_t": "u4",
        "u4": "u4",
        "uint4": "u4",
        "dynamic": "dynamic",
    }
    return aliases.get(raw, raw)


def read_kv_cache_precision(device: str = "GPU", *, core: Any | None = None) -> dict[str, Any]:
    """Read OpenVINO plugin property KV_CACHE_PRECISION (exact property name).

    Pass the same ``ov.Core`` instance used for ``set_property`` / pipeline load —
    a fresh Core does not observe another Core's sticky device properties.
    """
    import openvino as ov

    core = core if core is not None else ov.Core()
    try:
        value = core.get_property(device, "KV_CACHE_PRECISION")
        return {
            "property": "KV_CACHE_PRECISION",
            "device": device,
            "raw": str(value),
            "to_string": (str(value.to_string()) if hasattr(value, "to_string") else None),
            "normalized": _type_to_precision_name(value),
            "ok": True,
            "error": None,
        }
    except Exception as exc:
        return {
            "property": "KV_CACHE_PRECISION",
            "device": device,
            "raw": None,
            "to_string": None,
            "normalized": None,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def render_timing_style(tokenizer: Any, user_text: str) -> str:
    """Exact sealed-cell prompt style: user-only chat template, no tools."""
    return str(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
    )


def render_bfcl_tools_style(
    tokenizer: Any, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> str:
    """BFCL-capable prompt: native Qwen tools block + enable_thinking=False."""
    return str(
        tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
    )


def render_bfcl_prompting_xlam_style(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    """BFCL non-FC / prompting render: no tools= block; system prompt carries Python-AST format.

    Same enable_thinking=False empty-think control as the tools-style arm; only the
    response protocol (Python-AST list vs <tool_call> JSON) differs.
    """
    return str(
        tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
    )


def _bfcl_multi_turn_function_docs(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Raw BFCL function docs (not OpenAI tools) for prompting-mode system prompt."""
    return [row["bfcl_function"] for row in _iter_multi_turn_func_docs(entry["raw_entry"])]


def messages_for_xlam_prompting_turn0(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """First-turn messages with official BFCL prompting system prompt injected."""
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(BFCL_UNPACKED))
    from apu_characterization.cap01.bfcl_shims import install_bfcl_runtime_shims

    install_bfcl_runtime_shims()
    from bfcl_eval.model_handler.utils import system_prompt_pre_processing_chat_model

    turn0 = [dict(m) for m in entry["question"][0] if isinstance(m, dict)]
    func_docs = _bfcl_multi_turn_function_docs(entry)
    return system_prompt_pre_processing_chat_model(turn0, func_docs, str(entry["id"]))


def decode_execute_xlam(text: str) -> list[str]:
    """BFCL official prompting-mode decode: Python-AST list → execute strings."""
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(BFCL_UNPACKED))
    from apu_characterization.cap01.bfcl_shims import install_bfcl_runtime_shims

    install_bfcl_runtime_shims()
    from bfcl_eval.model_handler.utils import default_decode_execute_prompting

    return default_decode_execute_prompting(text)


def encode_execute_calls_as_xlam_text(calls: list[str]) -> str:
    """Wrap execute-format strings as a single Python-AST list generation."""
    return "[" + ", ".join(calls) + "]"


def _kwargs_normalize_execute_call(call: str, functions: list[dict[str, Any]]) -> str:
    """Rewrite positional args to kwargs using function schema property order.

    Official ``resolve_ast_call`` ignores positionals; gold has rare positionals.
    Kwargs-normalized XLAM text is what the prompt asks the model to emit.
    """
    import ast as _ast

    try:
        node = _ast.parse(call, mode="eval").body
    except SyntaxError:
        return call
    if not isinstance(node, _ast.Call) or not node.args:
        return call
    func_name = _ast.unparse(node.func) if hasattr(_ast, "unparse") else None
    if func_name is None:
        if isinstance(node.func, _ast.Name):
            func_name = node.func.id
        else:
            return call
    schema = next((f for f in functions if f.get("name") == func_name), None)
    if schema is None:
        return call
    props = list((schema.get("parameters") or {}).get("properties") or {})
    if len(node.args) > len(props):
        return call
    kwargs = {k.arg: k.value for k in node.keywords if k.arg}
    for i, arg in enumerate(node.args):
        kwargs[props[i]] = arg
    parts = []
    for name, val in kwargs.items():
        parts.append(f"{name}={_ast.unparse(val)}")
    return f"{func_name}({', '.join(parts)})"


def _chat_message_for_genai(msg: dict[str, Any]) -> dict[str, Any]:
    """Normalize a BFCL/OpenAI-style message for OpenVINO GenAI ChatHistory."""
    out: dict[str, Any] = {
        "role": str(msg.get("role") or ""),
        "content": str(msg.get("content") or ""),
    }
    if msg.get("name") is not None:
        out["name"] = str(msg["name"])
    return out


def build_bfcl_chat_history(
    ov_genai: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> Any:
    """Raw-message ChatHistory with tools + enable_thinking=False (single template apply)."""
    history = ov_genai.ChatHistory()
    history.set_tools(list(tools))
    history.set_extra_context({"enable_thinking": False})
    for msg in messages:
        history.append(_chat_message_for_genai(msg))
    return history


def render_genai_chat_history(genai_tokenizer: Any, history: Any) -> str:
    """Render ChatHistory with the GenAI tokenizer (picks up tools + extra_context)."""
    return str(genai_tokenizer.apply_chat_template(history, True))


def assert_no_think_in_generation(text: str, *, where: str) -> None:
    """Fail loudly if a generation opens ``<think>`` (config alone is not evidence)."""
    if THINK_OPEN_RE.search(text or ""):
        head = (text or "")[:240].replace("\n", "\\n")
        raise RuntimeError(
            f"FATAL: <think> appeared in generation at {where} "
            f"(enable_thinking=False not in force). head={head!r}"
        )


def assert_first_turn_token_equivalence(
    *,
    hf_tokenizer: Any,
    genai_tokenizer: Any,
    ov_genai: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    entry_id: str,
) -> dict[str, Any]:
    """First turn must be identical by construction across residency arms.

    RESIDENT path: GenAI ChatHistory → apply_chat_template (tools + thinking off).
    NON_RESIDENT path: same ChatHistory render, then cold string generate.
    Also require byte-identity with HF ``render_bfcl_tools_style`` (known-good BFCL path).
    """
    history = build_bfcl_chat_history(ov_genai, messages, tools)
    resident_render = render_genai_chat_history(genai_tokenizer, history)
    non_resident_render = resident_render
    hf_render = render_bfcl_tools_style(hf_tokenizer, messages, tools)

    r_ids = list(hf_tokenizer(resident_render)["input_ids"])
    n_ids = list(hf_tokenizer(non_resident_render)["input_ids"])
    hf_ids = list(hf_tokenizer(hf_render)["input_ids"])

    def _first_mismatch(a: list[int], b: list[int]) -> int | None:
        for i, (x, y) in enumerate(zip(a, b, strict=False)):
            if x != y:
                return i
        return None if len(a) == len(b) else min(len(a), len(b))

    if r_ids != n_ids:
        raise RuntimeError(
            f"FATAL: first-turn RESIDENT vs NON_RESIDENT token sequences diverge "
            f"for {entry_id}: len_R={len(r_ids)} len_N={len(n_ids)} "
            f"first_mismatch={_first_mismatch(r_ids, n_ids)}"
        )
    if r_ids != hf_ids:
        raise RuntimeError(
            f"FATAL: GenAI ChatHistory render != HF tools-style render for {entry_id}: "
            f"len_genai={len(r_ids)} len_hf={len(hf_ids)} "
            f"first_mismatch={_first_mismatch(r_ids, hf_ids)}"
        )
    if not resident_render.endswith("</think>\n\n"):
        raise RuntimeError(
            f"FATAL: first-turn render for {entry_id} does not end with "
            f"thinking-off assistant prefix; tail={resident_render[-80]!r}"
        )
    return {
        "entry_id": entry_id,
        "identical": True,
        "n_tokens": len(r_ids),
        "resident_sha256": hashlib.sha256(resident_render.encode("utf-8")).hexdigest(),
        "non_resident_sha256": hashlib.sha256(non_resident_render.encode("utf-8")).hexdigest(),
        "hf_sha256": hashlib.sha256(hf_render.encode("utf-8")).hexdigest(),
    }


def user_text_from_question(question: list) -> str:
    """First-turn user content (BFCL question is list[list[message]])."""
    turn0 = question[0]
    parts = [
        str(m.get("content", "")) for m in turn0 if isinstance(m, dict) and m.get("role") == "user"
    ]
    return "\n".join(p for p in parts if p).strip()


def messages_for_entry(entry: dict[str, Any]) -> list[dict[str, Any]]:
    turn0 = entry["question"][0]
    return [dict(m) for m in turn0 if isinstance(m, dict)]


def tools_for_entry(entry: dict[str, Any]) -> list[dict[str, Any]]:
    if entry["kind"] == "multi_turn":
        return _load_multi_turn_tools(entry["raw_entry"])
    return [_bfcl_function_to_openai_tool(fn) for fn in entry["function"]]


def inventory() -> dict[str, Any]:
    dist = next(BFCL_UNPACKED.glob("bfcl_eval-*.dist-info"), None)
    version = None
    if dist and (dist / "METADATA").is_file():
        for line in (dist / "METADATA").read_text(encoding="utf-8").splitlines():
            if line.startswith("Version:"):
                version = line.split(":", 1)[1].strip()
                break
    cats = []
    for p in sorted(BFCL_DATA.glob("BFCL_v4_*.json")):
        n_q = sum(1 for line in p.open(encoding="utf-8") if line.strip())
        ap = BFCL_ANSWERS / p.name
        n_a = sum(1 for line in ap.open(encoding="utf-8") if line.strip()) if ap.is_file() else None
        cats.append(
            {
                "file": p.name,
                "category": p.name.removeprefix("BFCL_v4_").removesuffix(".json"),
                "questions": n_q,
                "answers": n_a,
                "ground_truth_path": str(ap) if ap.is_file() else None,
            }
        )
    return {
        "package": "bfcl_eval",
        "version": version,
        "source": (
            "Apache-2.0 bfcl_eval wheel unpacked under CAP-01 live_sources; "
            "upstream https://github.com/ShishirPatil/gorilla/tree/main/"
            "berkeley-function-call-leaderboard"
        ),
        "unpacked_path": str(BFCL_UNPACKED),
        "data_path": str(BFCL_DATA),
        "ground_truth_dir": str(BFCL_ANSWERS),
        "categories": cats,
    }


def score_ast(
    *,
    functions: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    reference: list,
    test_category: str,
) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(BFCL_UNPACKED))
    # Shims MUST land before any bfcl_eval import that pulls java/js parsers.
    from apu_characterization.cap01.bfcl_shims import install_bfcl_runtime_shims

    install_bfcl_runtime_shims()
    from bfcl_eval.constants.enums import Language
    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker

    return ast_checker(
        functions,
        candidate,
        reference,
        Language.PYTHON,
        test_category,
        "cap01",
    )


def extract_tool_calls_qwen(text: str) -> list[dict[str, Any]]:
    """Extract Qwen <tool_call> JSON bodies (official QwenFCHandler pattern + loose)."""
    matches = TOOL_CALL_RE_QWEN.findall(text)
    if not matches:
        matches = TOOL_CALL_RE.findall(text)
    out: list[dict[str, Any]] = []
    for raw in matches:
        try:
            call = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(call, dict) and "name" in call:
            out.append(call)
    return out


def decode_execute_qwen(text: str) -> list[str]:
    """Map model text → BFCL execute strings via CAP-01 shims + convert_to_function_call."""
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(BFCL_UNPACKED))
    from apu_characterization.cap01.bfcl_shims import install_bfcl_runtime_shims

    install_bfcl_runtime_shims()
    from bfcl_eval.model_handler.utils import convert_to_function_call

    tool_calls = extract_tool_calls_qwen(text)
    if not tool_calls:
        raise ValueError("no_tool_calls")
    decoded_ast: list[dict[str, Any]] = []
    for item in tool_calls:
        args = item.get("arguments", item.get("parameters", {}))
        if not isinstance(args, dict):
            raise TypeError("tool_call_arguments_not_object")
        decoded_ast.append({str(item["name"]): args})
    return convert_to_function_call(decoded_ast)


def score_multi_turn(
    *,
    test_entry: dict[str, Any],
    ground_truth: list[list[str]],
    model_result_decoded: list[list[list[str]]] | None = None,
    test_category: str = "multi_turn_base",
    model_name: str | None = None,
) -> dict[str, Any]:
    """Official multi_turn_checker via CAP-01 wrapper.

    Unique ``model_name`` per call: the checker caches instances in
    ``globals()`` and reuse corrupts state.
    """
    sys.path.insert(0, str(ROOT))
    from apu_characterization.cap01.bfcl_cap01_multi_turn_checker import check

    return check(
        test_entry=test_entry,
        ground_truth=ground_truth,
        model_result_decoded=model_result_decoded,
        test_category=test_category,
        model_name=model_name,
    )


def run_multi_turn_gold_selftest(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Feed ground-truth execute strings into multi_turn_checker; expect all valid."""
    details: list[dict[str, Any]] = []
    for entry in entries:
        if entry["kind"] != "multi_turn":
            continue
        try:
            verdict = score_multi_turn(
                test_entry=entry["raw_entry"],
                ground_truth=entry["reference"],
                model_result_decoded=None,
                test_category=entry["category"],
            )
            ok = bool(verdict.get("valid"))
            err = (
                None
                if ok
                else {
                    k: verdict.get(k)
                    for k in ("error_message", "error_type", "details")
                    if k in verdict
                }
            )
        except Exception as exc:
            ok = False
            err = f"{type(exc).__name__}: {exc}"
        details.append(
            {
                "id": entry["id"],
                "category": entry["category"],
                "gold_multi_turn_valid": ok,
                "error": err,
            }
        )
    return {
        "n": len(details),
        "n_valid": sum(1 for d in details if d["gold_multi_turn_valid"]),
        "details": details,
        "checker": "bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker",
        "wrapper": "apu_characterization.cap01.bfcl_cap01_multi_turn_checker",
    }


def run_xlam_gold_decode_selftest(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Encode gold as XLAM Python-AST text → official decode → multi_turn_checker.

    One gold call in the 20-entry pin uses a positional arg; official
    ``resolve_ast_call`` drops positionals, so that call is kwargs-normalized
    from the function schema before encoding (matches what the prompt asks).
    """
    details: list[dict[str, Any]] = []
    n_positional_normalized = 0
    for entry in entries:
        if entry["kind"] != "multi_turn":
            continue
        functions = _bfcl_multi_turn_function_docs(entry)
        decoded_steps: list[list[list[str]]] = []
        parse_error: str | None = None
        for turn in entry["reference"]:
            if not turn:
                decoded_steps.append([])
                continue
            normalized: list[str] = []
            for call in turn:
                before = call
                after = _kwargs_normalize_execute_call(call, functions)
                if after != before:
                    n_positional_normalized += 1
                normalized.append(after)
            text = encode_execute_calls_as_xlam_text(normalized)
            try:
                decoded = decode_execute_xlam(text)
                decoded_steps.append([decoded])
            except Exception as exc:
                parse_error = f"{type(exc).__name__}: {exc}"
                break
        if parse_error is not None:
            details.append(
                {
                    "id": entry["id"],
                    "category": entry["category"],
                    "gold_xlam_decode_valid": False,
                    "error": parse_error,
                }
            )
            continue
        try:
            verdict = score_multi_turn(
                test_entry=entry["raw_entry"],
                ground_truth=entry["reference"],
                model_result_decoded=decoded_steps,
                test_category=entry["category"],
                model_name=f"xlam_gold_{entry['id']}".replace("-", "_").replace(".", "_"),
            )
            ok = bool(verdict.get("valid"))
            err = (
                None
                if ok
                else {
                    k: verdict.get(k)
                    for k in ("error_message", "error_type", "details")
                    if k in verdict
                }
            )
        except Exception as exc:
            ok = False
            err = f"{type(exc).__name__}: {exc}"
        details.append(
            {
                "id": entry["id"],
                "category": entry["category"],
                "gold_xlam_decode_valid": ok,
                "error": err,
            }
        )
    return {
        "n": len(details),
        "n_valid": sum(1 for d in details if d["gold_xlam_decode_valid"]),
        "n_positional_normalized": n_positional_normalized,
        "details": details,
        "decoder": "bfcl_eval.model_handler.utils.default_decode_execute_prompting",
        "checker": "bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker",
        "note": (
            "Gold execute strings wrapped as [call, ...] then decoded with the "
            "official prompting-mode path. Positional gold args kwargs-normalized "
            "because official resolve_ast_call ignores positionals."
        ),
    }


def run_acquire_tokenize(out_dir: Path) -> dict[str, Any]:
    inv = inventory()
    entries = select_entries()
    tokenizer = _hf_tokenizer()
    rows: list[dict[str, Any]] = []
    gold_checks: list[dict[str, Any]] = []

    for entry in entries:
        user_text = user_text_from_question(entry["question"])
        tools = tools_for_entry(entry)
        timing_prompt = render_timing_style(tokenizer, user_text)
        bfcl_prompt = render_bfcl_tools_style(tokenizer, messages_for_entry(entry), tools)
        n_timing = len(tokenizer(timing_prompt)["input_ids"])
        n_bfcl = len(tokenizer(bfcl_prompt)["input_ids"])
        row = {
            "id": entry["id"],
            "category": entry["category"],
            "kind": entry["kind"],
            "n_tools": len(tools),
            "timing_style_tokens": n_timing,
            "bfcl_tools_style_tokens": n_bfcl,
            "token_delta_bfcl_minus_timing": n_bfcl - n_timing,
            "user_chars": len(user_text),
        }
        rows.append(row)

        if entry["kind"] == "single_turn":
            gold = materialize_gold(entry["reference"])
            try:
                verdict = score_ast(
                    functions=entry["function"],
                    candidate=gold,
                    reference=entry["reference"],
                    test_category=entry["category"],
                )
                gold_ok = bool(verdict.get("valid"))
                gold_err = verdict.get("error")
            except Exception as exc:
                gold_ok = False
                gold_err = f"{type(exc).__name__}: {exc}"
            gold_checks.append(
                {
                    "id": entry["id"],
                    "category": entry["category"],
                    "gold_ast_valid": gold_ok,
                    "error": gold_err,
                }
            )

    mt_gold = run_multi_turn_gold_selftest(entries)

    ir_bin = MODEL_DIR / "openvino_model.bin"
    # Directory-level ir_sha256 is the FetchedModelSpec pin; also record bin hash.
    bin_sha = _sha256_file(ir_bin) if ir_bin.is_file() else None

    report = {
        "probe": "bfcl_feasibility",
        "mode": "acquire_tokenize",
        "bfcl": inv,
        "model": _model_report_fields(
            openvino_model_bin_sha256=bin_sha,
            enable_thinking=False,
            sealed_arm={
                "id": "gpu_only",
                "load_sequence": ["GPU"],
                "generate_device": "GPU",
                "apply_chat_template_at_generate": False,
            },
        ),
        "prompt_format_divergence": {
            "timing_arm": (
                "apply_chat_template([{role:user, content}], "
                "add_generation_prompt=True, enable_thinking=False); NO tools=; "
                "then GenerationConfig.apply_chat_template=False"
            ),
            "bfcl_labelling": (
                "apply_chat_template(messages, tools=[...], "
                "add_generation_prompt=True, enable_thinking=False); "
                "then GenerationConfig.apply_chat_template=False"
            ),
            "diverges": True,
            "how": (
                "Qwen3 chat_template injects a system '# Tools' block with "
                "<tools> schemas and <tool_call> instructions whenever tools= "
                "is non-empty. Timing cells never pass tools, so their prompts "
                "lack that system prefix. Formats are NOT close enough to treat "
                "as the same measurement condition without a re-measure under "
                "the labelling prompt format."
            ),
        },
        "entries": rows,
        "token_profile_summary": _summarize_tokens(rows),
        "ast_checker_gold_selftest": {
            "n": len(gold_checks),
            "n_valid": sum(1 for g in gold_checks if g["gold_ast_valid"]),
            "details": gold_checks,
            "note": ("Single-turn only. Multi-turn uses multi_turn_checker_gold_selftest."),
        },
        "multi_turn_checker_gold_selftest": mt_gold,
        "gpu_probe": {
            "status": "not_run",
            "reason": "acquire_tokenize mode only",
        },
        "opus_cost_estimate": None,
    }
    report["opus_cost_estimate"] = estimate_opus_cost(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "inventory.json").write_text(
        json.dumps(inv, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "probe_entries.json").write_text(
        json.dumps(entries, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (out_dir / "tokenize_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _summarize_tokens(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)

    def stats(vals: list[int]) -> dict[str, float | int]:
        return {
            "n": len(vals),
            "min": min(vals),
            "max": max(vals),
            "mean": sum(vals) / len(vals),
            "sum": sum(vals),
        }

    out: dict[str, Any] = {"overall": {}, "by_category": {}}
    out["overall"]["timing_style"] = stats([r["timing_style_tokens"] for r in rows])
    out["overall"]["bfcl_tools_style"] = stats([r["bfcl_tools_style_tokens"] for r in rows])
    out["overall"]["delta"] = stats([r["token_delta_bfcl_minus_timing"] for r in rows])
    for cat, rs in by_cat.items():
        out["by_category"][cat] = {
            "timing_style": stats([r["timing_style_tokens"] for r in rs]),
            "bfcl_tools_style": stats([r["bfcl_tools_style_tokens"] for r in rs]),
            "delta": stats([r["token_delta_bfcl_minus_timing"] for r in rs]),
        }
    return out


def estimate_opus_cost(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Cost of an Opus pass over a full labelling slice at $5/$25 per 1M (no cache).

    Slice definition (documented, not invented as a seal claim):
      all CAP-01-eligible single-turn BFCL subsets with answers
      (simple_*/parallel*/multiple*/live_*) = 2501 entries from inventory.
    Uses mean bfcl_tools_style prompt tokens from this probe as the prompt
    length prior; assumes 256 completion tokens/entry for a labelling pass.
    """
    single = [r for r in rows if r["kind"] == "single_turn"]
    multi = [r for r in rows if r["kind"] == "multi_turn"]
    mean_single = sum(r["bfcl_tools_style_tokens"] for r in single) / len(single) if single else 0.0
    mean_multi = sum(r["bfcl_tools_style_tokens"] for r in multi) / len(multi) if multi else 0.0
    # Full obtainable single-turn + parallel + multiple + live with answers:
    n_full = 0
    n_multi = 0
    for p in BFCL_DATA.glob("BFCL_v4_*.json"):
        cat = p.name.removeprefix("BFCL_v4_").removesuffix(".json")
        ap = BFCL_ANSWERS / p.name
        if not ap.is_file():
            continue
        n = sum(1 for line in ap.open(encoding="utf-8") if line.strip())
        if cat.startswith("multi_turn"):
            n_multi += n
        else:
            n_full += n
    completion_tok = 256
    prompt_m = (mean_single * n_full) / 1_000_000
    completion_m = (completion_tok * n_full) / 1_000_000
    # $5 / 1M input, $25 / 1M output
    cost = 5.0 * prompt_m + 25.0 * completion_m
    # Alternate: include multi_turn with multi-turn mean prompt length.
    prompt_m_all = (mean_single * n_full + mean_multi * n_multi) / 1_000_000
    completion_m_all = (completion_tok * (n_full + n_multi)) / 1_000_000
    cost_all = 5.0 * prompt_m_all + 25.0 * completion_m_all
    return {
        "pricing": {"input_usd_per_1m": 5.0, "output_usd_per_1m": 25.0, "cache": False},
        "slice_definition": (
            "primary: all BFCL v4 files with possible_answer except multi_turn_*; "
            "alternate adds multi_turn_* at multi-turn mean prompt length"
        ),
        "n_entries_single_turn_ast": n_full,
        "n_entries_multi_turn": n_multi,
        "assumed_mean_prompt_tokens_single_turn": mean_single,
        "assumed_mean_prompt_tokens_multi_turn": mean_multi,
        "assumed_completion_tokens_per_entry": completion_tok,
        "primary_single_turn_ast": {
            "prompt_millions": prompt_m,
            "completion_millions": completion_m,
            "estimated_usd": round(cost, 4),
        },
        "alternate_including_multi_turn": {
            "prompt_millions": prompt_m_all,
            "completion_millions": completion_m_all,
            "estimated_usd": round(cost_all, 4),
        },
        "probe_n20_usd": round(
            5.0 * (sum(r["bfcl_tools_style_tokens"] for r in rows) / 1e6)
            + 25.0 * (completion_tok * len(rows) / 1e6),
            4,
        ),
    }


def run_gpu(out_dir: Path, *, max_new_tokens: int = 512) -> dict[str, Any]:
    """gpu_only generate for the 20-entry probe. Caller must ensure host cleanliness."""
    import openvino_genai as ov_genai

    tokenize_report = run_acquire_tokenize(out_dir)
    entries = json.loads((out_dir / "probe_entries.json").read_text(encoding="utf-8"))
    tokenizer = _hf_tokenizer()

    t_load0 = time.perf_counter()
    pipe = _make_llm_pipeline(ov_genai, "GPU")
    load_s = time.perf_counter() - t_load0

    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = max_new_tokens
    cfg.do_sample = False
    cfg.apply_chat_template = False

    results: list[dict[str, Any]] = []
    for entry in entries:
        tools = tools_for_entry(entry)
        prompt = render_bfcl_tools_style(tokenizer, messages_for_entry(entry), tools)
        # Multi-turn: first turn only for this probe (full multi_turn_checker needs an agent loop).
        t0 = time.perf_counter()
        gen = pipe.generate([prompt], cfg)
        wall_s = time.perf_counter() - t0
        texts = getattr(gen, "texts", None)
        text = str(texts[0]) if texts else str(gen)
        metrics = getattr(gen, "perf_metrics", None)
        prompt_tokens = None
        completion_tokens = None
        if metrics is not None:
            try:
                prompt_tokens = int(metrics.get_num_input_tokens())
                completion_tokens = int(metrics.get_num_generated_tokens())
            except Exception:  # optional
                pass
        if prompt_tokens is None:
            prompt_tokens = len(tokenizer(prompt)["input_ids"])
        if completion_tokens is None:
            completion_tokens = len(tokenizer(text)["input_ids"])

        parsed = extract_tool_calls_ast(text)
        score: dict[str, Any]
        if entry["kind"] == "single_turn":
            if parsed is None:
                score = {
                    "valid": False,
                    "error": ["unparseable_tool_calls"],
                    "error_type": "probe:parse",
                }
            else:
                try:
                    score = score_ast(
                        functions=entry["function"],
                        candidate=parsed,
                        reference=entry["reference"],
                        test_category=entry["category"],
                    )
                except Exception as exc:
                    score = {
                        "valid": False,
                        "error": [f"{type(exc).__name__}: {exc}"],
                        "error_type": "probe:ast_exception",
                    }
        else:
            # First-turn weak check: name+arg string presence vs GT call strings.
            # Official scoring is multi_turn_checker; AST does not apply.
            gt0 = entry["reference"][0] if entry["reference"] else []
            score = {
                "valid": None,
                "ast_applicable": False,
                "first_turn_ground_truth": gt0,
                "parsed_ast": parsed,
                "note": (
                    "multi_turn requires multi_turn_checker + stateful execution; "
                    "not scored as AST accuracy in this probe"
                ),
            }

        results.append(
            {
                "id": entry["id"],
                "category": entry["category"],
                "kind": entry["kind"],
                "wall_s": wall_s,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "text_head": text[:500],
                "parsed_ast": parsed,
                "score": score,
            }
        )

    # Per-category accuracy (single-turn AST only)
    by_cat: dict[str, dict[str, int]] = {}
    for r in results:
        if r["kind"] != "single_turn":
            continue
        cat = r["category"]
        by_cat.setdefault(cat, {"n": 0, "correct": 0})
        by_cat[cat]["n"] += 1
        if r["score"].get("valid") is True:
            by_cat[cat]["correct"] += 1

    gpu_report = {
        **tokenize_report,
        "mode": "run_gpu",
        "gpu_probe": {
            "status": "complete",
            "isolation_mode": "OPERATOR_ASSERTED_CLEAN",
            "load_sequence": ["GPU"],
            "generate_device": "GPU",
            "enable_thinking": False,
            "apply_chat_template_at_generate": False,
            "prompt_format": "bfcl_tools_style",
            "model_load_s": load_s,
            "max_new_tokens": max_new_tokens,
            "per_entry": results,
            "accuracy_ast_single_turn": {
                cat: {
                    "correct": v["correct"],
                    "n": v["n"],
                    "accuracy": v["correct"] / v["n"] if v["n"] else None,
                }
                for cat, v in by_cat.items()
            },
            "multi_turn": {
                "n": sum(1 for r in results if r["kind"] == "multi_turn"),
                "ast_scored": False,
                "reason": "official multi_turn_checker only; first-turn generations retained",
            },
        },
    }
    (out_dir / "gpu_probe_report.json").write_text(
        json.dumps(gpu_report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return gpu_report


def _stats_int(vals: list[int]) -> dict[str, float | int]:
    if not vals:
        return {"n": 0, "min": None, "max": None, "mean": None, "sum": 0}
    return {
        "n": len(vals),
        "min": min(vals),
        "max": max(vals),
        "mean": sum(vals) / len(vals),
        "sum": sum(vals),
    }


def _stats_float(vals: list[float]) -> dict[str, float | int | None]:
    if not vals:
        return {"n": 0, "min": None, "max": None, "mean": None, "sum": 0.0}
    return {
        "n": len(vals),
        "min": min(vals),
        "max": max(vals),
        "mean": sum(vals) / len(vals),
        "sum": sum(vals),
    }


def _stats_float_list(vals: list[float | None]) -> dict[str, float | int | None]:
    clean = [float(v) for v in vals if v is not None]
    return _stats_float(clean)


def format_residency_delta_messages(msgs: list[dict[str, Any]]) -> str:
    """Deprecated delta formatter (DISPATCH J double-template path). Kept for import compat."""
    parts: list[str] = []
    for m in msgs:
        role = str(m.get("role") or "")
        content = str(m.get("content") or "")
        if role == "tool":
            parts.append(f"<tool_response>\n{content}\n</tool_response>")
        else:
            parts.append(content)
    return "\n".join(parts)


def _parse_execute_sig(s: str) -> tuple[str, tuple[tuple[str, str], ...]] | None:
    s = s.strip()
    try:
        node = py_ast.parse(s, mode="eval").body
    except SyntaxError:
        m = re.match(r"([A-Za-z_]\w*)\s*\(", s)
        return (m.group(1), (("__raw__", s),)) if m else None
    if not isinstance(node, py_ast.Call) or not isinstance(node.func, py_ast.Name):
        return None
    args: dict[str, Any] = {}
    for i, a in enumerate(node.args):
        try:
            args[f"#{i}"] = py_ast.literal_eval(a)
        except Exception:
            args[f"#{i}"] = py_ast.dump(a)
    for kw in node.keywords:
        key = kw.arg if kw.arg is not None else "**"
        try:
            args[key] = py_ast.literal_eval(kw.value)
        except Exception:
            args[key] = py_ast.dump(kw.value)
    items = tuple(sorted((k, repr(v)) for k, v in args.items()))
    return node.func.id, items


def structural_turn_correct(model_steps: list[list[str]], gold_calls: list[str]) -> dict[str, Any]:
    """F2-style per-turn accuracy: unordered multiset of execute-string signatures."""
    model_calls: list[str] = []
    for step in model_steps:
        model_calls.extend(step)
    m_sigs = [_parse_execute_sig(c) for c in model_calls]
    g_sigs = [_parse_execute_sig(c) for c in gold_calls]
    if any(s is None for s in m_sigs + g_sigs):

        def norm(x: str) -> str:
            return re.sub(r"\s+", "", x)

        exact = Counter(norm(c) for c in model_calls) == Counter(norm(c) for c in gold_calls)
        return {
            "correct": exact,
            "model_n": len(model_calls),
            "gold_n": len(gold_calls),
            "compare_mode": "normalized_string",
        }
    exact = Counter(m_sigs) == Counter(g_sigs)
    return {
        "correct": exact,
        "model_n": len(model_calls),
        "gold_n": len(gold_calls),
        "compare_mode": "parsed_sig_multiset",
    }


def _arm_device_config(arm_id: str) -> dict[str, Any]:
    import yaml

    cfg = yaml.safe_load(DELTA_N_CFG.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise TypeError("configs/delta_n.yaml: expected mapping")
    arms = cfg.get("arms")
    if not isinstance(arms, list):
        raise TypeError("configs/delta_n.yaml: missing arms list")
    arm = next((a for a in arms if a.get("id") == arm_id), None)
    if arm is None:
        known = [str(a.get("id")) for a in arms if isinstance(a, dict)]
        raise ValueError(f"no arm {arm_id!r} in delta_n.yaml (known: {known})")
    load_sequence = arm.get("load_sequence")
    generate_device = arm.get("generate_device")
    if not load_sequence or generate_device is None:
        raise ValueError(f"arm {arm_id}: incomplete load_sequence/generate_device")
    ov = cfg.get("openvino") or {}
    cpu_properties = dict(ov.get("cpu_properties") or {})
    arm_properties = dict(arm.get("properties") or {})
    resolved: list[dict[str, Any]] = []
    for device in load_sequence:
        props: dict[str, Any] = {}
        if str(device) == "CPU":
            props.update(cpu_properties)
        props.update(arm_properties)
        resolved.append({"device": str(device), "properties": props})
    return {
        "arm_id": arm_id,
        "label": arm.get("label"),
        "load_sequence": resolved,
        "generate_device": str(generate_device),
    }


def load_arm_pipeline(
    arm_id: str,
    *,
    enable_prefix_caching: bool | None = None,
) -> tuple[Any, dict[str, Any], float]:
    """Compile the generate-device pipeline for a delta_n arm. Returns (pipe, meta, load_s).

    ``enable_prefix_caching``:
      None  — omit SchedulerConfig; LLMPipeline's ContinuousBatching backend keeps its
              client-scenario default (prefix caching ON for CPU/GPU). Use for RESIDENT.
      False — pass SchedulerConfig(enable_prefix_caching=False). Required for NON_RESIDENT
              cold prefill (DISPATCH L): without this, string generate() reuses KV for the
              shared tools/conversation prefix on the same pipeline.
      True  — explicitly enable prefix caching via SchedulerConfig.
    """
    import openvino as ov
    import openvino_genai as ov_genai

    from seam.ov_kv_precision import (
        enforce_kv_cache_precision,
        materialize_pipeline_properties,
        requested_kv_cache_precision,
    )

    device_config = _arm_device_config(arm_id)
    pipes: dict[str, Any] = {}
    loads: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    scheduler_config_meta: dict[str, Any] | None = None
    pipeline_config: dict[str, Any] = {}
    if enable_prefix_caching is not None:
        sc = ov_genai.SchedulerConfig()
        sc.enable_prefix_caching = bool(enable_prefix_caching)
        pipeline_config["scheduler_config"] = sc
        scheduler_config_meta = {
            "enable_prefix_caching": bool(sc.enable_prefix_caching),
            "source": "explicit_SchedulerConfig",
        }
    for entry in device_config["load_sequence"]:
        device = str(entry["device"])
        properties_raw = dict(entry.get("properties") or {})
        properties = materialize_pipeline_properties(properties_raw)
        requested_kv = requested_kv_cache_precision(properties_raw)
        core = ov.Core()
        if requested_kv is not None:
            core.set_property(device, {"KV_CACHE_PRECISION": properties["KV_CACHE_PRECISION"]})
        if pipeline_config:
            pipes[device] = _make_llm_pipeline(ov_genai, device, pipeline_config, **properties)
        else:
            pipes[device] = _make_llm_pipeline(ov_genai, device, **properties)
        kv_check = enforce_kv_cache_precision(device=device, requested=requested_kv, core=core)
        loads.append(
            {
                "device": device,
                "properties": properties_raw,
                "kv_cache_precision": kv_check,
                "scheduler_config": scheduler_config_meta,
            }
        )
        if not kv_check["match"]:
            raise RuntimeError(
                f"KV_PRECISION_MISMATCH arm={arm_id} device={device}: "
                f"{kv_check.get('failure_mode')}"
            )
    load_s = time.perf_counter() - t0
    gen_dev = str(device_config["generate_device"])
    if gen_dev not in pipes:
        raise RuntimeError(f"generate_device {gen_dev} missing from pipes {sorted(pipes)}")
    meta = {
        "device_config": device_config,
        "loads": loads,
        "scheduler_config": scheduler_config_meta
        or {
            "enable_prefix_caching": None,
            "source": "LLMPipeline_default_CB_prefix_caching_on",
        },
        "openvino": ov.__version__,
        "openvino_genai": getattr(ov_genai, "__version__", "unknown"),
    }
    return pipes[gen_dev], meta, load_s


def _timed_generate(
    pipe: Any,
    ov_genai: Any,
    prompt: Any,
    cfg: Any,
) -> dict[str, Any]:
    """Generate with dual-source TTFT + decode_tok_s (same instrument as smoke_delta_prefill).

    ``prompt`` may be a pre-rendered string (NON_RESIDENT / legacy) or a GenAI
    ``ChatHistory`` (RESIDENT — template applied once inside generate).
    """
    from seam.backends.local_openvino import (
        _extract_metrics,
        _make_ttft_streamer,
        resolve_ttft_ns,
    )

    streamer = _make_ttft_streamer(ov_genai)
    t0 = time.perf_counter_ns()
    streamer.t0_ns = t0
    exc: BaseException | None = None
    result: Any = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            if isinstance(prompt, str):
                result = pipe.generate([prompt], cfg, streamer)
            else:
                result = pipe.generate(prompt, cfg, streamer)
    except Exception as e:
        exc = e
    wall_ns = time.perf_counter_ns() - t0
    wall_s = wall_ns / _NS_PER_S
    out: dict[str, Any] = {
        "ok": exc is None,
        "error": f"{type(exc).__name__}: {exc}" if exc is not None else None,
        "wall_s": wall_s,
        "ttft_s": None,
        "decode_tok_s": None,
        "ttft_ns": None,
        "ttft_source": None,
        "prompt_tokens_reported": None,
        "generated_tokens": None,
        "text": None,
    }
    if exc is not None:
        return out
    texts = getattr(result, "texts", None)
    text = str(texts[0]) if texts else str(result)
    out["text"] = text
    metrics = getattr(result, "perf_metrics", None)
    metrics_ttft_ns, prompt_tokens, tokens_from_metrics = _extract_metrics(metrics)
    ttft_ns, ttft_source = resolve_ttft_ns(metrics_ttft_ns, streamer.ttft_ns)
    out["ttft_ns"] = ttft_ns
    out["ttft_source"] = ttft_source
    out["prompt_tokens_reported"] = prompt_tokens if prompt_tokens else None
    generated = int(tokens_from_metrics) if tokens_from_metrics else None
    out["generated_tokens"] = generated
    if ttft_ns is not None and ttft_ns > 0:
        ttft_s = ttft_ns / _NS_PER_S
        out["ttft_s"] = ttft_s
        if generated is not None and generated >= 2 and wall_s > ttft_s:
            out["decode_tok_s"] = (generated - 1) / (wall_s - ttft_s)
    return out


def _generation_cfg(
    ov_genai: Any,
    *,
    max_new_tokens: int,
    apply_chat_template: bool,
) -> Any:
    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = int(max_new_tokens)
    cfg.do_sample = False
    cfg.apply_chat_template = bool(apply_chat_template)
    return cfg


def run_multi_turn_agent_entry(
    *,
    pipe: Any,
    tokenizer: Any,
    cfg: Any,
    entry: dict[str, Any],
    residency_mode: str | None = None,
    ov_genai: Any | None = None,
) -> dict[str, Any]:
    """Full BFCL multi-turn agent loop (user turns x ≤MAXIMUM_STEP_LIMIT steps).

    residency_mode:
      None           — legacy: full cold prompt every generate (no chat mode)
      RESIDENT       — ChatHistory held across the session; ``generate(ChatHistory)``
                       applies the template once (tools + enable_thinking=False) and
                       retains KV via history-prefix continuation
      NON_RESIDENT   — same ChatHistory render each step, cold string generate with
                       ``apply_chat_template=False``, on a pipeline loaded with
                       ``SchedulerConfig(enable_prefix_caching=False)`` so ContinuousBatching
                       cannot reuse KV across turns
    """
    import openvino_genai as _ov_genai_mod

    ov_genai = ov_genai if ov_genai is not None else _ov_genai_mod
    mode = residency_mode.upper() if residency_mode else None
    if mode is not None and mode not in RESIDENCY_MODES:
        raise ValueError(f"residency_mode must be one of {RESIDENCY_MODES} or None")

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(BFCL_UNPACKED))
    from apu_characterization.cap01.bfcl_shims import install_bfcl_runtime_shims

    install_bfcl_runtime_shims()
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
        execute_multi_turn_func_call,
        is_empty_execute_response,
    )

    raw = entry["raw_entry"]
    tools = tools_for_entry(entry)
    initial_config = raw.get("initial_config") or {}
    involved_classes = raw.get("involved_classes") or []
    test_entry_id = str(raw["id"])
    test_category = entry["category"]
    model_name = f"probe_gpu_{test_entry_id}".replace("-", "_").replace(".", "_")
    if mode:
        model_name = f"probe_{mode.lower()}_{test_entry_id}".replace("-", "_").replace(".", "_")

    # Warm instances (same as BFCL inference_multi_turn_*), is_evaL_run=False.
    execute_multi_turn_func_call(
        [],
        initial_config,
        involved_classes,
        model_name,
        test_entry_id,
        long_context=("long_context" in test_category or "composite" in test_category),
        is_evaL_run=False,
    )

    messages: list[dict[str, Any]] = []
    all_model_response: list[list[str]] = []
    all_decoded: list[list[list[str]]] = []
    turn_metrics: list[dict[str, Any]] = []
    prompt_token_samples: list[int] = []
    completion_token_samples: list[int] = []
    context_growth: list[dict[str, Any]] = []
    force_quit = False
    session_generated_once = False
    first_turn_equiv: dict[str, Any] | None = None
    prev_turn0_prompt_tokens: int | None = None
    t_entry0 = time.perf_counter()
    wall_s = 0.0
    max_new = int(getattr(cfg, "max_new_tokens", 512) or 512)
    genai_tokenizer = pipe.get_tokenizer() if mode is not None else None
    # RESIDENT: one ChatHistory for the whole entry (tools + thinking off).
    resident_history: Any | None = (
        build_bfcl_chat_history(ov_genai, [], tools) if mode == "RESIDENT" else None
    )

    def _finish_chat_safe() -> None:
        with contextlib.suppress(Exception), warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            pipe.finish_chat()

    try:
        for turn_idx, turn_msgs in enumerate(entry["question"]):
            t_turn0 = time.perf_counter()
            t_tool_exec = 0.0
            t_template_build = 0.0
            t_tokenize = 0.0
            t_generate = 0.0
            new_turn_msgs = [dict(m) for m in turn_msgs if isinstance(m, dict)]
            messages.extend(new_turn_msgs)
            if resident_history is not None:
                for m in new_turn_msgs:
                    resident_history.append(_chat_message_for_genai(m))

            turn_responses: list[str] = []
            turn_decoded_steps: list[list[str]] = []
            turn_step_metrics: list[dict[str, Any]] = []
            step = 0
            turn_ttft_s: float | None = None
            turn_decode_tok_s: float | None = None
            turn_generated: int | None = None
            turn_prompt_tokens: int | None = None
            turn_delta_tokens: int | None = None
            turn_slo_ok: bool | None = None

            while True:
                _t = time.perf_counter()
                prompt_full = render_bfcl_tools_style(tokenizer, messages, tools)
                t_template_build += time.perf_counter() - _t
                _t = time.perf_counter()
                prompt_n = len(tokenizer(prompt_full)["input_ids"])
                t_tokenize += time.perf_counter() - _t
                prompt_token_samples.append(prompt_n)

                if mode == "RESIDENT":
                    assert resident_history is not None and genai_tokenizer is not None
                    if not session_generated_once:
                        first_turn_equiv = assert_first_turn_token_equivalence(
                            hf_tokenizer=tokenizer,
                            genai_tokenizer=genai_tokenizer,
                            ov_genai=ov_genai,
                            messages=messages,
                            tools=tools,
                            entry_id=str(entry["id"]),
                        )
                    gen_input: Any = resident_history
                    input_kind = "chat_history_resident"
                    _t = time.perf_counter()
                    gen_input_tokens = len(
                        tokenizer(render_genai_chat_history(genai_tokenizer, resident_history))[
                            "input_ids"
                        ]
                    )
                    t_tokenize += time.perf_counter() - _t
                    cfg_use = _generation_cfg(
                        ov_genai,
                        max_new_tokens=max_new,
                        apply_chat_template=True,
                    )
                elif mode == "NON_RESIDENT":
                    assert genai_tokenizer is not None
                    _t = time.perf_counter()
                    hist = build_bfcl_chat_history(ov_genai, messages, tools)
                    rendered = render_genai_chat_history(genai_tokenizer, hist)
                    t_template_build += time.perf_counter() - _t
                    if not session_generated_once:
                        first_turn_equiv = assert_first_turn_token_equivalence(
                            hf_tokenizer=tokenizer,
                            genai_tokenizer=genai_tokenizer,
                            ov_genai=ov_genai,
                            messages=messages,
                            tools=tools,
                            entry_id=str(entry["id"]),
                        )
                    gen_input = rendered
                    input_kind = "full_prompt"
                    _t = time.perf_counter()
                    gen_input_tokens = len(tokenizer(rendered)["input_ids"])
                    t_tokenize += time.perf_counter() - _t
                    cfg_use = _generation_cfg(
                        ov_genai,
                        max_new_tokens=max_new,
                        apply_chat_template=False,
                    )
                else:
                    gen_input = prompt_full
                    input_kind = "full_prompt"
                    gen_input_tokens = prompt_n
                    cfg_use = cfg

                context_growth.append(
                    {
                        "turn": turn_idx,
                        "step": step,
                        "prompt_tokens": prompt_n,
                        "delta_tokens_vs_prev_turn": (
                            None
                            if prev_turn0_prompt_tokens is None or step != 0
                            else prompt_n - prev_turn0_prompt_tokens
                        ),
                        "generate_input_tokens": gen_input_tokens,
                        "input_kind": input_kind,
                        "n_messages": len(messages),
                        "residency_mode": mode,
                    }
                )

                prompt_tokens_reported: int | None = None
                if mode is None:
                    t0 = time.perf_counter()
                    gen = pipe.generate([prompt_full], cfg)
                    wall_s = time.perf_counter() - t0
                    t_generate += wall_s
                    texts = getattr(gen, "texts", None)
                    text = str(texts[0]) if texts else str(gen)
                    metrics = getattr(gen, "perf_metrics", None)
                    completion_n = None
                    ttft_s = None
                    decode_tok_s = None
                    if metrics is not None:
                        with contextlib.suppress(Exception):
                            completion_n = int(metrics.get_num_generated_tokens())
                        try:
                            ttft_ms = float(metrics.get_ttft().mean)
                            if ttft_ms > 0:
                                ttft_s = ttft_ms / 1000.0
                        except Exception:
                            pass
                        try:
                            prompt_tokens_reported = int(metrics.get_num_input_tokens())
                        except Exception:
                            prompt_tokens_reported = None
                    if completion_n is None:
                        completion_n = len(tokenizer(text)["input_ids"])
                    if ttft_s is not None and completion_n >= 2 and wall_s > ttft_s:
                        decode_tok_s = (completion_n - 1) / (wall_s - ttft_s)
                    gen_ok = True
                    gen_err = None
                else:
                    timed = _timed_generate(pipe, ov_genai, gen_input, cfg_use)
                    wall_s = float(timed["wall_s"])
                    t_generate += wall_s
                    text = timed.get("text") or ""
                    completion_n = timed.get("generated_tokens")
                    if completion_n is None:
                        completion_n = len(tokenizer(text)["input_ids"])
                    ttft_s = timed.get("ttft_s")
                    decode_tok_s = timed.get("decode_tok_s")
                    gen_ok = bool(timed["ok"])
                    gen_err = timed.get("error")
                    reported = timed.get("prompt_tokens_reported")
                    prompt_tokens_reported = int(reported) if reported is not None else None

                completion_token_samples.append(int(completion_n))
                session_generated_once = True
                turn_responses.append(text)

                step_rec = {
                    "turn": turn_idx,
                    "step": step,
                    "prompt_tokens": prompt_n,
                    "generate_input_tokens": gen_input_tokens,
                    "prompt_tokens_reported": prompt_tokens_reported,
                    "input_kind": input_kind,
                    "ttft_s": ttft_s,
                    "decode_tok_s": decode_tok_s,
                    "generated_tokens": int(completion_n),
                    "wall_s": wall_s,
                    "ok": gen_ok,
                    "error": gen_err,
                }
                turn_step_metrics.append(step_rec)

                if step == 0:
                    turn_ttft_s = ttft_s
                    turn_decode_tok_s = decode_tok_s
                    turn_generated = int(completion_n)
                    turn_prompt_tokens = prompt_n
                    turn_delta_tokens = (
                        None
                        if prev_turn0_prompt_tokens is None
                        else prompt_n - prev_turn0_prompt_tokens
                    )
                    turn_slo_ok = (
                        ttft_s is not None
                        and decode_tok_s is not None
                        and ttft_s <= SLO_TTFT_S
                        and decode_tok_s >= SLO_DECODE_TOK_S
                    )

                if gen_ok:
                    assert_no_think_in_generation(
                        text,
                        where=f"{entry['id']}/turn{turn_idx}/step{step}/{mode}",
                    )

                messages.append({"role": "assistant", "content": text})
                if resident_history is not None:
                    resident_history.append({"role": "assistant", "content": text})

                if not gen_ok:
                    break

                try:
                    decoded = decode_execute_qwen(text)
                    if is_empty_execute_response(decoded):
                        break
                except Exception:
                    break

                turn_decoded_steps.append(decoded)
                _t = time.perf_counter()
                execution_results, _instances = execute_multi_turn_func_call(
                    decoded,
                    initial_config,
                    involved_classes,
                    model_name,
                    test_entry_id,
                    long_context=("long_context" in test_category or "composite" in test_category),
                    is_evaL_run=False,
                )
                t_tool_exec += time.perf_counter() - _t
                for exec_str, exec_result in zip(decoded, execution_results, strict=False):
                    tool_msg = {
                        "role": "tool",
                        "name": exec_str,
                        "content": str(exec_result),
                    }
                    messages.append(tool_msg)
                    if resident_history is not None:
                        resident_history.append(_chat_message_for_genai(tool_msg))
                step += 1
                if step > MAXIMUM_STEP_LIMIT:
                    force_quit = True
                    break

            if turn_prompt_tokens is not None:
                prev_turn0_prompt_tokens = turn_prompt_tokens

            all_model_response.append(turn_responses)
            all_decoded.append(turn_decoded_steps)

            per_turn_acc = None
            if turn_idx < len(entry["reference"]):
                per_turn_acc = structural_turn_correct(
                    turn_decoded_steps, entry["reference"][turn_idx]
                )

            turn_wall_s = time.perf_counter() - t_turn0
            phases = finalize_phase_timers(
                turn_wall_s=turn_wall_s,
                t_tool_exec=t_tool_exec,
                t_template_build=t_template_build,
                t_tokenize=t_tokenize,
                t_generate=t_generate,
            )
            turn_metrics.append(
                {
                    "turn": turn_idx,
                    "n_generations": len(turn_responses),
                    "n_decoded_steps": len(turn_decoded_steps),
                    "last_wall_s": wall_s,
                    "ttft_s": turn_ttft_s,
                    "prompt_tokens": turn_prompt_tokens,
                    "delta_tokens_vs_prev_turn": turn_delta_tokens,
                    "decode_tok_s": turn_decode_tok_s,
                    "generated_tokens": turn_generated,
                    "slo_ttft_s": SLO_TTFT_S,
                    "slo_decode_tok_s": SLO_DECODE_TOK_S,
                    "slo_ok": turn_slo_ok,
                    "per_turn_accuracy": per_turn_acc,
                    "steps": turn_step_metrics,
                    # Additive phase timers (future sealed runs). Not consumed by replay yet.
                    **phases,
                }
            )
            if force_quit:
                break
    finally:
        if mode == "RESIDENT":
            _finish_chat_safe()

    wall_entry_s = time.perf_counter() - t_entry0

    score: dict[str, Any]
    if force_quit or len(all_model_response) != len(entry["reference"]):
        score = {
            "valid": False,
            "error_type": "multi_turn:force_terminated",
            "error_message": (
                f"turns_model={len(all_model_response)} "
                f"turns_gt={len(entry['reference'])} force_quit={force_quit}"
            ),
        }
    else:
        try:
            score = score_multi_turn(
                test_entry=raw,
                ground_truth=entry["reference"],
                model_result_decoded=all_decoded,
                test_category=test_category,
            )
        except Exception as exc:
            score = {
                "valid": False,
                "error_type": "probe:multi_turn_exception",
                "error_message": f"{type(exc).__name__}: {exc}",
            }

    prompt_by_turn = [g["prompt_tokens"] for g in context_growth if g["step"] == 0]
    deltas_by_turn = [g.get("delta_tokens_vs_prev_turn") for g in context_growth if g["step"] == 0]
    slo_flags = [t.get("slo_ok") for t in turn_metrics if t.get("slo_ok") is not None]
    per_turn_correct = [
        bool(t["per_turn_accuracy"]["correct"])
        for t in turn_metrics
        if t.get("per_turn_accuracy") is not None
    ]
    return {
        "id": entry["id"],
        "category": entry["category"],
        "kind": "multi_turn",
        "residency_mode": mode,
        "wall_s": wall_entry_s,
        "force_quit": force_quit,
        "n_user_turns": len(entry["question"]),
        "n_completed_turns": len(all_model_response),
        "prompt_tokens_all_steps": prompt_token_samples,
        "completion_tokens_all_steps": completion_token_samples,
        "prompt_tokens_sum": sum(prompt_token_samples),
        "completion_tokens_sum": sum(completion_token_samples),
        "prompt_tokens_first_step_per_turn": prompt_by_turn,
        "delta_tokens_vs_prev_turn": deltas_by_turn,
        "context_growth": context_growth,
        "context_growth_delta_tokens": (
            (context_growth[-1]["prompt_tokens"] - context_growth[0]["prompt_tokens"])
            if len(context_growth) >= 2
            else 0
        ),
        "turn_metrics": turn_metrics,
        "first_turn_token_equivalence": first_turn_equiv,
        "session_summary": {
            "total_latency_s": wall_entry_s,
            "turn_count": len(turn_metrics),
            "fraction_turns_slo_ok": (
                sum(1 for x in slo_flags if x) / len(slo_flags) if slo_flags else None
            ),
            "n_turns_slo_ok": sum(1 for x in slo_flags if x),
            "n_turns_slo_scored": len(slo_flags),
            "per_turn_accuracy_correct": sum(1 for x in per_turn_correct if x),
            "per_turn_accuracy_n": len(per_turn_correct),
            "per_turn_accuracy": (
                sum(1 for x in per_turn_correct if x) / len(per_turn_correct)
                if per_turn_correct
                else None
            ),
            "measured_context_growth_tokens": (
                (prompt_by_turn[-1] - prompt_by_turn[0]) if len(prompt_by_turn) >= 2 else 0
            ),
            "n_generations_with_think": sum(
                1 for turn in all_model_response for t in turn if THINK_OPEN_RE.search(t or "")
            ),
        },
        # Seal complete generations (DISPATCH K); heads retained for skim.
        "model_result_raw": all_model_response,
        "model_result_raw_heads": [[t[:300] for t in turn] for turn in all_model_response],
        "model_result_decoded": all_decoded,
        "score": {
            "valid": score.get("valid"),
            "error_type": score.get("error_type"),
            "error_message": score.get("error_message") or score.get("error"),
        },
    }


def run_gpu_multi_turn(out_dir: Path, *, max_new_tokens: int = 512) -> dict[str, Any]:
    """gpu_only full multi-turn probe for 20 multi_turn_base entries."""
    import openvino_genai as ov_genai

    out_dir.mkdir(parents=True, exist_ok=True)
    entries = select_multi_turn_entries()
    (out_dir / "multi_turn_probe_entries.json").write_text(
        json.dumps(entries, indent=2, default=str) + "\n", encoding="utf-8"
    )

    gold = run_multi_turn_gold_selftest(entries)
    (out_dir / "multi_turn_gold_selftest.json").write_text(
        json.dumps(gold, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    tokenizer = _hf_tokenizer()
    # First-turn token profile (no agent loop) for distribution context.
    token_rows: list[dict[str, Any]] = []
    for entry in entries:
        tools = tools_for_entry(entry)
        prompt = render_bfcl_tools_style(tokenizer, messages_for_entry(entry), tools)
        token_rows.append(
            {
                "id": entry["id"],
                "category": entry["category"],
                "kind": "multi_turn",
                "timing_style_tokens": len(
                    tokenizer(
                        render_timing_style(tokenizer, user_text_from_question(entry["question"]))
                    )["input_ids"]
                ),
                "bfcl_tools_style_tokens": len(tokenizer(prompt)["input_ids"]),
                "token_delta_bfcl_minus_timing": 0,
                "n_tools": len(tools),
                "n_user_turns": len(entry["question"]),
            }
        )
    for r in token_rows:
        r["token_delta_bfcl_minus_timing"] = r["bfcl_tools_style_tokens"] - r["timing_style_tokens"]

    t_load0 = time.perf_counter()
    pipe = _make_llm_pipeline(ov_genai, "GPU")
    load_s = time.perf_counter() - t_load0

    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = max_new_tokens
    cfg.do_sample = False
    cfg.apply_chat_template = False

    results: list[dict[str, Any]] = []
    for entry in entries:
        print(f"[multi_turn] {entry['id']} …", flush=True)
        results.append(
            run_multi_turn_agent_entry(pipe=pipe, tokenizer=tokenizer, cfg=cfg, entry=entry)
        )
        # Incremental artifact so a kill still leaves partial evidence.
        (out_dir / "multi_turn_gpu_probe_partial.json").write_text(
            json.dumps(results, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    correct = sum(1 for r in results if r["score"].get("valid") is True)
    all_prompt = [t for r in results for t in r["prompt_tokens_all_steps"]]
    all_completion = [t for r in results for t in r["completion_tokens_all_steps"]]
    walls = [float(r["wall_s"]) for r in results]
    growth = [int(r["context_growth_delta_tokens"]) for r in results]
    first_prompts = [t for r in results for t in r["prompt_tokens_first_step_per_turn"]]

    by_cat: dict[str, dict[str, int]] = {}
    for r in results:
        cat = r["category"]
        by_cat.setdefault(cat, {"n": 0, "correct": 0})
        by_cat[cat]["n"] += 1
        if r["score"].get("valid") is True:
            by_cat[cat]["correct"] += 1

    report = {
        "probe": "bfcl_feasibility_multi_turn",
        "mode": "run_gpu_multi_turn",
        "model": _model_report_fields(
            enable_thinking=False,
            sealed_arm={
                "id": "gpu_only",
                "load_sequence": ["GPU"],
                "generate_device": "GPU",
                "apply_chat_template_at_generate": False,
            },
        ),
        "multi_turn_checker_gold_selftest": gold,
        "first_turn_token_profile": {
            "entries": token_rows,
            "summary": _summarize_tokens(token_rows),
        },
        "gpu_probe": {
            "status": "complete",
            "isolation_mode": "OPERATOR_ASSERTED_CLEAN",
            "load_sequence": ["GPU"],
            "generate_device": "GPU",
            "enable_thinking": False,
            "apply_chat_template_at_generate": False,
            "prompt_format": "bfcl_tools_style_multi_turn_agent",
            "model_load_s": load_s,
            "max_new_tokens": max_new_tokens,
            "maximum_step_limit": MAXIMUM_STEP_LIMIT,
            "per_entry": results,
            "accuracy_multi_turn": {
                cat: {
                    "correct": v["correct"],
                    "n": v["n"],
                    "accuracy": v["correct"] / v["n"] if v["n"] else None,
                }
                for cat, v in by_cat.items()
            },
            "accuracy_overall": {
                "correct": correct,
                "n": len(results),
                "accuracy": correct / len(results) if results else None,
            },
            "wall_clock_s": _stats_float(walls),
            "prompt_tokens_per_generation": _stats_int(all_prompt),
            "completion_tokens_per_generation": _stats_int(all_completion),
            "prompt_tokens_first_step_per_turn": _stats_int(first_prompts),
            "context_growth_delta_tokens_per_entry": _stats_int(growth),
            "a2_single_turn_ast_reference": {
                "source": "derived/bfcl_feasibility/gpu_probe_report.json",
                "simple_python": {"correct": 3, "n": 5},
                "parallel": {"correct": 5, "n": 5},
                "note": (
                    "Cited from sealed A2 probe artifact for contrast only; "
                    "not re-measured in this multi-turn run."
                ),
            },
        },
    }
    (out_dir / "multi_turn_gpu_probe_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return report


def run_multi_turn_xlam_agent_entry(
    *,
    pipe: Any,
    tokenizer: Any,
    cfg: Any,
    entry: dict[str, Any],
) -> dict[str, Any]:
    """Multi-turn agent loop identical to run_multi_turn_agent_entry (gpu_only path),
    except prompt/decode use BFCL prompting-mode Python-AST (XLAM-format arm).
    """
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(BFCL_UNPACKED))
    from apu_characterization.cap01.bfcl_shims import install_bfcl_runtime_shims

    install_bfcl_runtime_shims()
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
        execute_multi_turn_func_call,
        is_empty_execute_response,
    )

    raw = entry["raw_entry"]
    func_docs = _bfcl_multi_turn_function_docs(entry)
    initial_config = raw.get("initial_config") or {}
    involved_classes = raw.get("involved_classes") or []
    test_entry_id = str(raw["id"])
    test_category = entry["category"]
    model_name = f"probe_gpu_xlam_{test_entry_id}".replace("-", "_").replace(".", "_")

    execute_multi_turn_func_call(
        [],
        initial_config,
        involved_classes,
        model_name,
        test_entry_id,
        long_context=("long_context" in test_category or "composite" in test_category),
        is_evaL_run=False,
    )

    messages: list[dict[str, Any]] = []
    all_model_response: list[list[str]] = []
    all_decoded: list[list[list[str]]] = []
    calls_emitted_per_generation: list[int] = []
    turn_metrics: list[dict[str, Any]] = []
    prompt_token_samples: list[int] = []
    completion_token_samples: list[int] = []
    context_growth: list[dict[str, Any]] = []
    force_quit = False
    prev_turn0_prompt_tokens: int | None = None
    t_entry0 = time.perf_counter()
    wall_s = 0.0

    for turn_idx, turn_msgs in enumerate(entry["question"]):
        new_turn_msgs = [dict(m) for m in turn_msgs if isinstance(m, dict)]
        if turn_idx == 0:
            from bfcl_eval.model_handler.utils import system_prompt_pre_processing_chat_model

            new_turn_msgs = system_prompt_pre_processing_chat_model(
                new_turn_msgs, func_docs, test_entry_id
            )
        messages.extend(new_turn_msgs)

        turn_responses: list[str] = []
        turn_decoded_steps: list[list[str]] = []
        turn_step_metrics: list[dict[str, Any]] = []
        turn_calls_per_gen: list[int] = []
        step = 0
        turn_ttft_s: float | None = None
        turn_decode_tok_s: float | None = None
        turn_generated: int | None = None
        turn_prompt_tokens: int | None = None
        turn_delta_tokens: int | None = None
        turn_slo_ok: bool | None = None

        while True:
            prompt_full = render_bfcl_prompting_xlam_style(tokenizer, messages)
            prompt_n = len(tokenizer(prompt_full)["input_ids"])
            prompt_token_samples.append(prompt_n)
            context_growth.append(
                {
                    "turn": turn_idx,
                    "step": step,
                    "prompt_tokens": prompt_n,
                    "delta_tokens_vs_prev_turn": (
                        None
                        if prev_turn0_prompt_tokens is None or step != 0
                        else prompt_n - prev_turn0_prompt_tokens
                    ),
                    "generate_input_tokens": prompt_n,
                    "input_kind": "full_prompt",
                    "n_messages": len(messages),
                    "residency_mode": None,
                }
            )

            t0 = time.perf_counter()
            gen = pipe.generate([prompt_full], cfg)
            wall_s = time.perf_counter() - t0
            texts = getattr(gen, "texts", None)
            text = str(texts[0]) if texts else str(gen)
            metrics = getattr(gen, "perf_metrics", None)
            completion_n = None
            ttft_s = None
            decode_tok_s = None
            prompt_tokens_reported: int | None = None
            if metrics is not None:
                with contextlib.suppress(Exception):
                    completion_n = int(metrics.get_num_generated_tokens())
                try:
                    ttft_ms = float(metrics.get_ttft().mean)
                    if ttft_ms > 0:
                        ttft_s = ttft_ms / 1000.0
                except Exception:
                    pass
                try:
                    prompt_tokens_reported = int(metrics.get_num_input_tokens())
                except Exception:
                    prompt_tokens_reported = None
            if completion_n is None:
                completion_n = len(tokenizer(text)["input_ids"])
            if ttft_s is not None and completion_n >= 2 and wall_s > ttft_s:
                decode_tok_s = (completion_n - 1) / (wall_s - ttft_s)

            completion_token_samples.append(int(completion_n))
            turn_responses.append(text)
            assert_no_think_in_generation(
                text,
                where=f"{entry['id']}/turn{turn_idx}/step{step}/xlam",
            )

            step_rec = {
                "turn": turn_idx,
                "step": step,
                "prompt_tokens": prompt_n,
                "generate_input_tokens": prompt_n,
                "prompt_tokens_reported": prompt_tokens_reported,
                "input_kind": "full_prompt",
                "ttft_s": ttft_s,
                "decode_tok_s": decode_tok_s,
                "generated_tokens": int(completion_n),
                "wall_s": wall_s,
                "ok": True,
                "error": None,
                "calls_emitted": None,
            }
            turn_step_metrics.append(step_rec)

            if step == 0:
                turn_ttft_s = ttft_s
                turn_decode_tok_s = decode_tok_s
                turn_generated = int(completion_n)
                turn_prompt_tokens = prompt_n
                turn_delta_tokens = (
                    None
                    if prev_turn0_prompt_tokens is None
                    else prompt_n - prev_turn0_prompt_tokens
                )
                turn_slo_ok = (
                    ttft_s is not None
                    and decode_tok_s is not None
                    and ttft_s <= SLO_TTFT_S
                    and decode_tok_s >= SLO_DECODE_TOK_S
                )

            messages.append({"role": "assistant", "content": text})

            try:
                decoded = decode_execute_xlam(text)
                n_calls = len(decoded)
                if is_empty_execute_response(decoded):
                    step_rec["calls_emitted"] = 0
                    calls_emitted_per_generation.append(0)
                    turn_calls_per_gen.append(0)
                    break
            except Exception:
                step_rec["calls_emitted"] = 0
                calls_emitted_per_generation.append(0)
                turn_calls_per_gen.append(0)
                break

            step_rec["calls_emitted"] = n_calls
            calls_emitted_per_generation.append(n_calls)
            turn_calls_per_gen.append(n_calls)
            turn_decoded_steps.append(decoded)
            execution_results, _instances = execute_multi_turn_func_call(
                decoded,
                initial_config,
                involved_classes,
                model_name,
                test_entry_id,
                long_context=("long_context" in test_category or "composite" in test_category),
                is_evaL_run=False,
            )
            for exec_str, exec_result in zip(decoded, execution_results, strict=False):
                messages.append(
                    {
                        "role": "tool",
                        "name": exec_str,
                        "content": str(exec_result),
                    }
                )
            step += 1
            if step > MAXIMUM_STEP_LIMIT:
                force_quit = True
                break

        if turn_prompt_tokens is not None:
            prev_turn0_prompt_tokens = turn_prompt_tokens

        all_model_response.append(turn_responses)
        all_decoded.append(turn_decoded_steps)

        per_turn_acc = None
        if turn_idx < len(entry["reference"]):
            per_turn_acc = structural_turn_correct(turn_decoded_steps, entry["reference"][turn_idx])

        turn_metrics.append(
            {
                "turn": turn_idx,
                "n_generations": len(turn_responses),
                "n_decoded_steps": len(turn_decoded_steps),
                "calls_emitted_per_generation": turn_calls_per_gen,
                "last_wall_s": wall_s,
                "ttft_s": turn_ttft_s,
                "prompt_tokens": turn_prompt_tokens,
                "delta_tokens_vs_prev_turn": turn_delta_tokens,
                "decode_tok_s": turn_decode_tok_s,
                "generated_tokens": turn_generated,
                "slo_ttft_s": SLO_TTFT_S,
                "slo_decode_tok_s": SLO_DECODE_TOK_S,
                "slo_ok": turn_slo_ok,
                "per_turn_accuracy": per_turn_acc,
                "steps": turn_step_metrics,
            }
        )
        if force_quit:
            break

    wall_entry_s = time.perf_counter() - t_entry0

    score: dict[str, Any]
    if force_quit or len(all_model_response) != len(entry["reference"]):
        score = {
            "valid": False,
            "error_type": "multi_turn:force_terminated",
            "error_message": (
                f"turns_model={len(all_model_response)} "
                f"turns_gt={len(entry['reference'])} force_quit={force_quit}"
            ),
        }
    else:
        try:
            score = score_multi_turn(
                test_entry=raw,
                ground_truth=entry["reference"],
                model_result_decoded=all_decoded,
                test_category=test_category,
                model_name=model_name + "_score",
            )
        except Exception as exc:
            score = {
                "valid": False,
                "error_type": "probe:multi_turn_exception",
                "error_message": f"{type(exc).__name__}: {exc}",
            }

    prompt_by_turn = [g["prompt_tokens"] for g in context_growth if g["step"] == 0]
    deltas_by_turn = [g.get("delta_tokens_vs_prev_turn") for g in context_growth if g["step"] == 0]
    slo_flags = [t.get("slo_ok") for t in turn_metrics if t.get("slo_ok") is not None]
    per_turn_correct = [
        bool(t["per_turn_accuracy"]["correct"])
        for t in turn_metrics
        if t.get("per_turn_accuracy") is not None
    ]
    return {
        "id": entry["id"],
        "category": entry["category"],
        "kind": "multi_turn",
        "residency_mode": None,
        "prompt_format": PROMPT_FORMAT_MULTI_TURN_XLAM,
        "wall_s": wall_entry_s,
        "force_quit": force_quit,
        "n_user_turns": len(entry["question"]),
        "n_completed_turns": len(all_model_response),
        "prompt_tokens_all_steps": prompt_token_samples,
        "completion_tokens_all_steps": completion_token_samples,
        "prompt_tokens_sum": sum(prompt_token_samples),
        "completion_tokens_sum": sum(completion_token_samples),
        "prompt_tokens_first_step_per_turn": prompt_by_turn,
        "delta_tokens_vs_prev_turn": deltas_by_turn,
        "context_growth": context_growth,
        "context_growth_delta_tokens": (
            (context_growth[-1]["prompt_tokens"] - context_growth[0]["prompt_tokens"])
            if len(context_growth) >= 2
            else 0
        ),
        "calls_emitted_per_generation": calls_emitted_per_generation,
        "calls_emitted_max": max(calls_emitted_per_generation)
        if calls_emitted_per_generation
        else 0,
        "n_generations_with_multi_call": sum(1 for n in calls_emitted_per_generation if n > 1),
        "turn_metrics": turn_metrics,
        "session_summary": {
            "total_latency_s": wall_entry_s,
            "turn_count": len(turn_metrics),
            "fraction_turns_slo_ok": (
                sum(1 for x in slo_flags if x) / len(slo_flags) if slo_flags else None
            ),
            "n_turns_slo_ok": sum(1 for x in slo_flags if x),
            "n_turns_slo_scored": len(slo_flags),
            "per_turn_accuracy_correct": sum(1 for x in per_turn_correct if x),
            "per_turn_accuracy_n": len(per_turn_correct),
            "per_turn_accuracy": (
                sum(1 for x in per_turn_correct if x) / len(per_turn_correct)
                if per_turn_correct
                else None
            ),
            "measured_context_growth_tokens": (
                (prompt_by_turn[-1] - prompt_by_turn[0]) if len(prompt_by_turn) >= 2 else 0
            ),
            "n_generations_with_think": sum(
                1 for turn in all_model_response for t in turn if THINK_OPEN_RE.search(t or "")
            ),
            "calls_emitted_max": (
                max(calls_emitted_per_generation) if calls_emitted_per_generation else 0
            ),
            "n_generations_with_multi_call": sum(1 for n in calls_emitted_per_generation if n > 1),
        },
        "model_result_raw": all_model_response,
        "model_result_raw_heads": [[t[:300] for t in turn] for turn in all_model_response],
        "model_result_decoded": all_decoded,
        "score": {
            "valid": score.get("valid"),
            "error_type": score.get("error_type"),
            "error_message": score.get("error_message") or score.get("error"),
        },
    }


def run_gpu_multi_turn_xlam(out_dir: Path, *, max_new_tokens: int = 512) -> dict[str, Any]:
    """gpu_only multi-turn arm: identical loop/checker, BFCL prompting Python-AST protocol."""
    import openvino_genai as ov_genai

    out_dir.mkdir(parents=True, exist_ok=True)
    entries = select_multi_turn_entries()
    (out_dir / "multi_turn_probe_entries.json").write_text(
        json.dumps(entries, indent=2, default=str) + "\n", encoding="utf-8"
    )

    gold = run_multi_turn_gold_selftest(entries)
    (out_dir / "multi_turn_gold_selftest.json").write_text(
        json.dumps(gold, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    gold_xlam = run_xlam_gold_decode_selftest(entries)
    (out_dir / "multi_turn_xlam_gold_decode_selftest.json").write_text(
        json.dumps(gold_xlam, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    tokenizer = _hf_tokenizer()
    entry0 = entries[0]
    tools_style_prompt = render_bfcl_tools_style(
        tokenizer, messages_for_entry(entry0), tools_for_entry(entry0)
    )
    xlam_prompt = render_bfcl_prompting_xlam_style(
        tokenizer, messages_for_xlam_prompting_turn0(entry0)
    )
    prompt_audit = {
        "entry_id": entry0["id"],
        "tools_style_prompt_format": PROMPT_FORMAT_MULTI_TURN_AGENT,
        "xlam_prompt_format": PROMPT_FORMAT_MULTI_TURN_XLAM,
        "tools_style_first_turn_prompt_verbatim": tools_style_prompt,
        "xlam_first_turn_prompt_verbatim": xlam_prompt,
        "prompt_construction": {
            "method": "official_bfcl_eval",
            "system_prompt": (
                "bfcl_eval.model_handler.utils.system_prompt_pre_processing_chat_model"
            ),
            "decode": "bfcl_eval.model_handler.utils.default_decode_execute_prompting",
            "render": (
                "HF AutoTokenizer.apply_chat_template without tools=, "
                "enable_thinking=False (same think control as tools-style arm)"
            ),
            "note": (
                "BFCL non-FC / prompting Python-AST list format "
                "[func(arg=val), ...]. Not Salesforce xLAM JSON-array handler."
            ),
        },
    }
    (out_dir / "xlam_vs_tools_first_turn_prompt_audit.json").write_text(
        json.dumps(prompt_audit, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    token_rows: list[dict[str, Any]] = []
    for entry in entries:
        prompt = render_bfcl_prompting_xlam_style(
            tokenizer, messages_for_xlam_prompting_turn0(entry)
        )
        tools_prompt = render_bfcl_tools_style(
            tokenizer, messages_for_entry(entry), tools_for_entry(entry)
        )
        token_rows.append(
            {
                "id": entry["id"],
                "category": entry["category"],
                "kind": "multi_turn",
                "xlam_prompting_tokens": len(tokenizer(prompt)["input_ids"]),
                "bfcl_tools_style_tokens": len(tokenizer(tools_prompt)["input_ids"]),
                "token_delta_xlam_minus_tools": (
                    len(tokenizer(prompt)["input_ids"]) - len(tokenizer(tools_prompt)["input_ids"])
                ),
                "n_tools": len(tools_for_entry(entry)),
                "n_user_turns": len(entry["question"]),
            }
        )

    t_load0 = time.perf_counter()
    pipe = _make_llm_pipeline(ov_genai, "GPU")
    load_s = time.perf_counter() - t_load0

    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = max_new_tokens
    cfg.do_sample = False
    cfg.apply_chat_template = False

    results: list[dict[str, Any]] = []
    for entry in entries:
        print(f"[multi_turn_xlam] {entry['id']} …", flush=True)
        results.append(
            run_multi_turn_xlam_agent_entry(pipe=pipe, tokenizer=tokenizer, cfg=cfg, entry=entry)
        )
        (out_dir / "multi_turn_xlam_gpu_probe_partial.json").write_text(
            json.dumps(results, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    correct = sum(1 for r in results if r["score"].get("valid") is True)
    all_prompt = [t for r in results for t in r["prompt_tokens_all_steps"]]
    all_completion = [t for r in results for t in r["completion_tokens_all_steps"]]
    walls = [float(r["wall_s"]) for r in results]
    growth = [int(r["context_growth_delta_tokens"]) for r in results]
    first_prompts = [t for r in results for t in r["prompt_tokens_first_step_per_turn"]]
    all_calls = [n for r in results for n in r["calls_emitted_per_generation"]]

    by_cat: dict[str, dict[str, int]] = {}
    for r in results:
        cat = r["category"]
        by_cat.setdefault(cat, {"n": 0, "correct": 0})
        by_cat[cat]["n"] += 1
        if r["score"].get("valid") is True:
            by_cat[cat]["correct"] += 1

    report = {
        "probe": "bfcl_feasibility_multi_turn_xlam",
        "mode": "run_gpu_multi_turn_xlam",
        "model": _model_report_fields(
            enable_thinking=False,
            sealed_arm={
                "id": "gpu_only",
                "load_sequence": ["GPU"],
                "generate_device": "GPU",
                "apply_chat_template_at_generate": False,
            },
        ),
        "multi_turn_checker_gold_selftest": gold,
        "xlam_gold_decode_selftest": gold_xlam,
        "first_turn_prompt_audit": {
            "entry_id": prompt_audit["entry_id"],
            "prompt_construction": prompt_audit["prompt_construction"],
            "xlam_first_turn_prompt_verbatim": xlam_prompt,
            "tools_style_first_turn_prompt_verbatim": tools_style_prompt,
        },
        "first_turn_token_profile": {
            "entries": token_rows,
        },
        "gpu_probe": {
            "status": "complete",
            "isolation_mode": "OPERATOR_ASSERTED_CLEAN",
            "load_sequence": ["GPU"],
            "generate_device": "GPU",
            "enable_thinking": False,
            "apply_chat_template_at_generate": False,
            "prompt_format": PROMPT_FORMAT_MULTI_TURN_XLAM,
            "contrast_prompt_format": PROMPT_FORMAT_MULTI_TURN_AGENT,
            "model_load_s": load_s,
            "max_new_tokens": max_new_tokens,
            "maximum_step_limit": MAXIMUM_STEP_LIMIT,
            "per_entry": results,
            "accuracy_multi_turn": {
                cat: {
                    "correct": v["correct"],
                    "n": v["n"],
                    "accuracy": v["correct"] / v["n"] if v["n"] else None,
                }
                for cat, v in by_cat.items()
            },
            "accuracy_overall": {
                "correct": correct,
                "n": len(results),
                "accuracy": correct / len(results) if results else None,
            },
            "wall_clock_s": _stats_float(walls),
            "prompt_tokens_per_generation": _stats_int(all_prompt),
            "completion_tokens_per_generation": _stats_int(all_completion),
            "prompt_tokens_first_step_per_turn": _stats_int(first_prompts),
            "context_growth_delta_tokens_per_entry": _stats_int(growth),
            "calls_emitted_per_generation": _stats_int(all_calls),
            "n_generations_with_multi_call": sum(1 for n in all_calls if n > 1),
            "fraction_generations_multi_call": (
                sum(1 for n in all_calls if n > 1) / len(all_calls) if all_calls else None
            ),
        },
    }
    (out_dir / "multi_turn_xlam_gpu_probe_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return report


def _unique_checker_name(prefix: str) -> str:
    raw = f"{prefix}_{uuid.uuid4().hex[:12]}"
    return re.sub(r"[^A-Za-z0-9_]", "_", raw)


def _cloud_usd(prompt_tokens: int, completion_tokens: int) -> float:
    return (
        prompt_tokens * CLOUD_USD_PER_MTOK_IN + completion_tokens * CLOUD_USD_PER_MTOK_OUT
    ) / 1_000_000.0


def load_pinned_multi_turn_entries(out_dir: Path) -> tuple[list[dict[str, Any]], Path]:
    """LOAD the sealed 20-entry pin. Do not reselect."""
    path = out_dir / PINNED_MULTI_TURN_ENTRIES
    if not path.is_file():
        raise FileNotFoundError(
            f"Pinned multi-turn entries missing: {path}. "
            "Do not reselect; the cloud arm must use the same 20 entries."
        )
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or len(entries) != 20:
        n = len(entries) if isinstance(entries, list) else type(entries).__name__
        raise ValueError(f"Expected 20 pinned entries in {path}, got {n}")
    return entries, path


def gold_decoded_for_entry(entry: dict[str, Any]) -> list[list[list[str]]]:
    sys.path.insert(0, str(ROOT))
    from apu_characterization.cap01.bfcl_cap01_multi_turn_checker import (
        gold_decoded_from_ground_truth,
    )

    return gold_decoded_from_ground_truth(entry["reference"])


def pad_decoded_to_gold(
    decoded: list[list[list[str]]] | None, n_gold_turns: int
) -> list[list[list[str]]]:
    md = list(decoded or [])
    while len(md) < n_gold_turns:
        md.append([])
    return md[:n_gold_turns]


def official_per_turn_from_decoded(
    *,
    entry: dict[str, Any],
    model_result_decoded: list[list[list[str]]] | None,
    name_prefix: str,
) -> dict[str, Any]:
    """Official checker per-turn via prefixes. Unique model_name per call."""
    gold = entry["reference"]
    n = len(gold)
    md = pad_decoded_to_gold(model_result_decoded, n)
    raw = entry["raw_entry"]
    test_category = entry.get("category") or "multi_turn_base"
    turn_valid: list[bool] = []
    first_fail: int | None = None
    first_fail_error: dict[str, Any] | None = None
    prefix_errors: list[dict[str, Any]] = []
    for t in range(n):
        try:
            score = score_multi_turn(
                test_entry=raw,
                ground_truth=gold[: t + 1],
                model_result_decoded=md[: t + 1],
                test_category=test_category,
                model_name=_unique_checker_name(f"{name_prefix}_t{t}"),
            )
        except Exception as exc:
            score = {
                "valid": False,
                "error_type": f"probe:per_turn_exception:{type(exc).__name__}",
                "error_message": f"{type(exc).__name__}: {exc}",
            }
        ok = bool(score.get("valid"))
        turn_valid.append(ok)
        rec = {
            "turn": t,
            "valid": ok,
            "error_type": score.get("error_type"),
            "error_message": score.get("error_message") or score.get("error"),
        }
        prefix_errors.append(rec)
        if not ok and first_fail is None:
            first_fail = t
            first_fail_error = {
                "error_type": rec["error_type"],
                "error_message": rec["error_message"],
            }
    return {
        "source": "official_multi_turn_checker_prefixes",
        "turn_valid": turn_valid,
        "turns_correct": sum(1 for v in turn_valid if v),
        "turns_total": n,
        "first_failure_turn": first_fail,
        "first_failure_error": first_fail_error,
        "prefix_scores": prefix_errors,
    }


def score_decoded_trajectory(
    *,
    entry: dict[str, Any],
    model_result_decoded: list[list[list[str]]] | None,
    name_prefix: str,
    force_quit: bool = False,
    entry_error: str | None = None,
) -> dict[str, Any]:
    """Full-trajectory official score + per-turn prefixes. Unique names per call."""
    gold = entry["reference"]
    n = len(gold)
    md = pad_decoded_to_gold(model_result_decoded, n)
    raw = entry["raw_entry"]
    test_category = entry.get("category") or "multi_turn_base"
    n_model_turns = len(model_result_decoded or [])
    if entry_error:
        score: dict[str, Any] = {
            "valid": False,
            "error_type": "probe:entry_error",
            "error_message": entry_error,
        }
    elif force_quit or n_model_turns != n:
        score = {
            "valid": False,
            "error_type": "multi_turn:force_terminated",
            "error_message": (f"turns_model={n_model_turns} turns_gt={n} force_quit={force_quit}"),
        }
    else:
        try:
            score = score_multi_turn(
                test_entry=raw,
                ground_truth=gold,
                model_result_decoded=md,
                test_category=test_category,
                model_name=_unique_checker_name(f"{name_prefix}_full"),
            )
        except Exception as exc:
            score = {
                "valid": False,
                "error_type": "probe:multi_turn_exception",
                "error_message": f"{type(exc).__name__}: {exc}",
            }
    per_turn = official_per_turn_from_decoded(
        entry=entry,
        model_result_decoded=md,
        name_prefix=name_prefix,
    )
    slim = {
        "valid": score.get("valid"),
        "error_type": score.get("error_type"),
        "error_message": score.get("error_message") or score.get("error"),
    }
    return {"score": slim, "per_turn": per_turn}


def summarize_per_turn_arm(rows: list[dict[str, Any]]) -> dict[str, Any]:
    turns_correct = 0
    turns_total = 0
    hist: Counter[str] = Counter()
    per_entry: list[dict[str, Any]] = []
    recoverable = True
    notes: list[str] = []
    for row in rows:
        pt = row.get("per_turn")
        eid = row.get("id")
        if not isinstance(pt, dict):
            recoverable = False
            notes.append(f"{eid}: per_turn missing")
            per_entry.append(
                {
                    "id": eid,
                    "turns_correct": None,
                    "turns_total": None,
                    "first_failure_turn": None,
                    "trajectory_valid": row.get("score", {}).get("valid")
                    if isinstance(row.get("score"), dict)
                    else None,
                    "recoverable": False,
                }
            )
            continue
        tc = int(pt["turns_correct"])
        tt = int(pt["turns_total"])
        turns_correct += tc
        turns_total += tt
        fail = pt.get("first_failure_turn")
        key = "none" if fail is None else str(fail)
        hist[key] += 1
        per_entry.append(
            {
                "id": eid,
                "turns_correct": tc,
                "turns_total": tt,
                "first_failure_turn": fail,
                "turn_valid": pt.get("turn_valid"),
                "trajectory_valid": row.get("score", {}).get("valid")
                if isinstance(row.get("score"), dict)
                else None,
                "recoverable": True,
            }
        )
    return {
        "turns_correct": turns_correct,
        "turns_total": turns_total,
        "first_failure_turn_distribution": dict(sorted(hist.items(), key=lambda kv: kv[0])),
        "per_entry": per_entry,
        "recoverable": recoverable,
        "notes": notes,
    }


def extract_local_arm_from_gpu_report(
    *,
    gpu_report: dict[str, Any],
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Re-score sealed local decoded trajectories. No GPU generate."""
    by_id = {e["id"]: e for e in entries}
    per_entry = (gpu_report.get("gpu_probe") or {}).get("per_entry")
    stored = (gpu_report.get("gpu_probe") or {}).get("accuracy_overall") or {}
    if not isinstance(per_entry, list) or not per_entry:
        return {
            "recoverable": False,
            "reason": (
                "multi_turn_gpu_probe_report.json has no gpu_probe.per_entry; "
                "cannot recover official per-turn from stored trajectories. "
                "Local arm not re-run."
            ),
            "trajectory_stored": stored,
            "turns_correct": None,
            "turns_total": None,
            "first_failure_turn_distribution": None,
            "per_entry": [],
        }
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for pe in per_entry:
        eid = pe.get("id")
        entry = by_id.get(eid)
        if entry is None:
            missing.append(str(eid))
            continue
        decoded = pe.get("model_result_decoded")
        if decoded is None:
            missing.append(str(eid))
            rows.append(
                {
                    "id": eid,
                    "score": pe.get("score") or {"valid": None},
                    "per_turn": None,
                    "stored_trajectory_valid": (pe.get("score") or {}).get("valid"),
                    "decoded_present": False,
                }
            )
            continue
        scored = score_decoded_trajectory(
            entry=entry,
            model_result_decoded=decoded,
            name_prefix=f"local_rescored_{eid}",
            force_quit=bool(pe.get("force_quit")),
        )
        rows.append(
            {
                "id": eid,
                "score": scored["score"],
                "per_turn": scored["per_turn"],
                "stored_trajectory_valid": (pe.get("score") or {}).get("valid"),
                "rescored_trajectory_valid": scored["score"].get("valid"),
                "decoded_present": True,
            }
        )
    summary = summarize_per_turn_arm(rows)
    n_valid = sum(
        1 for r in rows if isinstance(r.get("score"), dict) and r["score"].get("valid") is True
    )
    if missing:
        summary["recoverable"] = False
        summary["notes"] = [
            *(summary.get("notes") or []),
            f"missing_or_undecoded: {missing}",
        ]
    summary.update(
        {
            "source_report": "derived/bfcl_feasibility/multi_turn_gpu_probe_report.json",
            "trajectory_stored": {
                "correct": stored.get("correct"),
                "n": stored.get("n"),
            },
            "trajectory_rescored": {
                "correct": n_valid,
                "n": len(rows),
            },
            "gpu_report_entry_ids": [pe.get("id") for pe in per_entry],
        }
    )
    return summary


def _bfcl_functions_to_anthropic_tools(
    functions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(BFCL_UNPACKED))
    from apu_characterization.cap01.bfcl_shims import install_bfcl_runtime_shims

    install_bfcl_runtime_shims()
    from bfcl_eval.constants.enums import ModelStyle
    from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
    from bfcl_eval.model_handler.utils import convert_to_tool

    return convert_to_tool(functions, GORILLA_TO_OPENAPI, ModelStyle.ANTHROPIC)


def anthropic_tools_for_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Native Anthropic tools from BFCL function docs. Not Qwen <tools> text."""
    docs = _iter_multi_turn_func_docs(entry["raw_entry"])
    raw_fns = [row["bfcl_function"] for row in docs]
    tools = _bfcl_functions_to_anthropic_tools(raw_fns)
    mapping = []
    for row, tool in zip(docs, tools, strict=True):
        fn = row["bfcl_function"]
        mapping.append(
            {
                "bfcl_name": fn.get("name"),
                "involved_class": row["involved_class"],
                "source_file": row["source_file"],
                "anthropic_name": tool.get("name"),
                "bfcl_parameters_type": (fn.get("parameters") or {}).get("type"),
                "anthropic_input_schema_type": (tool.get("input_schema") or {}).get("type"),
            }
        )
    return {
        "tools": tools,
        "mapping": mapping,
        "mapping_note": (
            "Each BFCL function doc (name, description, parameters, optional response) "
            "maps to one Anthropic tool {name, description, input_schema} via "
            "bfcl_eval.model_handler.utils.convert_to_tool(..., ModelStyle.ANTHROPIC). "
            "parameters.type dict→object; response schema is appended to description. "
            "Tools are sent as the API tools= argument (native tool_use / tool_result "
            "blocks). The Qwen chat-template <tools> text is not placed in the prompt."
        ),
        "n_tools": len(tools),
        "involved_classes": list(entry["raw_entry"].get("involved_classes") or []),
    }


def decode_execute_anthropic(tool_use_blocks: list[dict[str, Any]]) -> list[str]:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(BFCL_UNPACKED))
    from apu_characterization.cap01.bfcl_shims import install_bfcl_runtime_shims

    install_bfcl_runtime_shims()
    from bfcl_eval.model_handler.utils import convert_to_function_call

    if not tool_use_blocks:
        raise ValueError("no_tool_calls")
    decoded_ast: list[dict[str, Any]] = []
    for item in tool_use_blocks:
        args = item.get("input", item.get("arguments", {}))
        if not isinstance(args, dict):
            raise TypeError("tool_call_arguments_not_object")
        decoded_ast.append({str(item["name"]): args})
    return convert_to_function_call(decoded_ast)


def _anthropic_content_to_dicts(content: Any) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for block in content or []:
        kind = getattr(block, "type", None)
        if kind == "text":
            blocks.append({"type": "text", "text": str(getattr(block, "text", "") or "")})
        elif kind == "tool_use":
            raw_in = getattr(block, "input", {})
            blocks.append(
                {
                    "type": "tool_use",
                    "id": str(getattr(block, "id", "") or ""),
                    "name": str(getattr(block, "name", "") or ""),
                    "input": raw_in if isinstance(raw_in, dict) else {},
                }
            )
        else:
            dumped = getattr(block, "model_dump", None)
            if callable(dumped):
                blocks.append(dumped())
    return blocks


def _anthropic_create_kwargs(
    *,
    model: str,
    max_tokens: int,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
        "tools": tools,
    }
    # claude-sonnet-5 rejects temperature (400 deprecated). Omit for that id;
    # send 0 for any other --model override.
    if model != CLOUD_DEFAULT_MODEL:
        kwargs["temperature"] = 0
    return kwargs


def run_cloud_multi_turn_agent_entry(
    *,
    client: Any,
    model: str,
    max_tokens: int,
    entry: dict[str, Any],
    running_usd: float,
) -> dict[str, Any]:
    """Same agent loop as run_gpu_multi_turn; model call is Anthropic native tool-use."""
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(BFCL_UNPACKED))
    from apu_characterization.cap01.bfcl_shims import install_bfcl_runtime_shims

    install_bfcl_runtime_shims()
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
        execute_multi_turn_func_call,
        is_empty_execute_response,
    )

    raw = entry["raw_entry"]
    tool_pack = anthropic_tools_for_entry(entry)
    tools = tool_pack["tools"]
    initial_config = raw.get("initial_config") or {}
    involved_classes = raw.get("involved_classes") or []
    test_entry_id = str(raw["id"])
    test_category = entry["category"]
    model_name = f"probe_cloud_{test_entry_id}".replace("-", "_").replace(".", "_")

    execute_multi_turn_func_call(
        [],
        initial_config,
        involved_classes,
        model_name,
        test_entry_id,
        long_context=("long_context" in test_category or "composite" in test_category),
        is_evaL_run=False,
    )

    messages: list[dict[str, Any]] = []
    all_model_response: list[list[str]] = []
    all_decoded: list[list[list[str]]] = []
    call_records: list[dict[str, Any]] = []
    force_quit = False
    abort_error: str | None = None
    usd_entry = 0.0
    prompt_tokens_sum = 0
    completion_tokens_sum = 0

    def _finish_score() -> dict[str, Any]:
        scored = score_decoded_trajectory(
            entry=entry,
            model_result_decoded=all_decoded,
            name_prefix=f"cloud_check_{test_entry_id}",
            force_quit=force_quit,
            entry_error=abort_error,
        )
        return {
            "id": entry["id"],
            "category": entry["category"],
            "kind": "multi_turn",
            "force_quit": force_quit,
            "n_user_turns": len(entry["question"]),
            "n_completed_turns": len(all_model_response),
            "entry_error": abort_error,
            "anthropic_tools": tools,
            "tool_mapping": tool_pack["mapping"],
            "tool_mapping_note": tool_pack["mapping_note"],
            "n_tools": tool_pack["n_tools"],
            "involved_classes": tool_pack["involved_classes"],
            "calls": call_records,
            "prompt_tokens_sum": prompt_tokens_sum,
            "completion_tokens_sum": completion_tokens_sum,
            "usd": usd_entry,
            "model_result_decoded": all_decoded,
            "model_result_raw_heads": [[t[:300] for t in turn] for turn in all_model_response],
            "score": scored["score"],
            "per_turn": scored["per_turn"],
        }

    for turn_idx, turn_msgs in enumerate(entry["question"]):
        for m in turn_msgs:
            if not isinstance(m, dict):
                continue
            content = m.get("content")
            text = content if isinstance(content, str) else str(content)
            messages.append(
                {
                    "role": m.get("role") or "user",
                    "content": [{"type": "text", "text": text}],
                }
            )

        turn_responses: list[str] = []
        turn_decoded_steps: list[list[str]] = []
        step = 0
        while True:
            create_kwargs = _anthropic_create_kwargs(
                model=model,
                max_tokens=max_tokens,
                messages=messages,
                tools=tools,
            )
            t0 = time.perf_counter()
            try:
                response = client.messages.create(**create_kwargs)
                latency_s = time.perf_counter() - t0
            except Exception as exc:
                latency_s = time.perf_counter() - t0
                abort_error = f"api_error: {type(exc).__name__}: {exc}"
                call_records.append(
                    {
                        "turn": turn_idx,
                        "step": step,
                        "ok": False,
                        "error": abort_error,
                        "prompt_tokens": None,
                        "completion_tokens": None,
                        "latency_s": latency_s,
                        "latency_s_note": (
                            "network-inclusive; not a timing measurement; "
                            "non-comparable to local wall clocks; do not cite as latency"
                        ),
                        "usd": 0.0,
                        "running_usd": running_usd + usd_entry,
                    }
                )
                all_model_response.append(turn_responses)
                all_decoded.append(turn_decoded_steps)
                return _finish_score()

            usage = getattr(response, "usage", None)
            prompt_n = int(getattr(usage, "input_tokens", 0) or 0) if usage is not None else 0
            completion_n = int(getattr(usage, "output_tokens", 0) or 0) if usage is not None else 0
            call_usd = _cloud_usd(prompt_n, completion_n)
            usd_entry += call_usd
            prompt_tokens_sum += prompt_n
            completion_tokens_sum += completion_n
            running_after = running_usd + usd_entry
            content_blocks = _anthropic_content_to_dicts(getattr(response, "content", []))
            tool_uses = [b for b in content_blocks if b.get("type") == "tool_use"]
            text_bits = [b.get("text") or "" for b in content_blocks if b.get("type") == "text"]
            text = "\n".join(t for t in text_bits if t)
            raw_head = (
                text
                if text
                else json.dumps(
                    [{"name": b.get("name"), "input": b.get("input")} for b in tool_uses]
                )
            )
            turn_responses.append(raw_head)
            call_records.append(
                {
                    "turn": turn_idx,
                    "step": step,
                    "ok": True,
                    "error": None,
                    "stop_reason": getattr(response, "stop_reason", None),
                    "prompt_tokens": prompt_n,
                    "completion_tokens": completion_n,
                    "latency_s": latency_s,
                    "latency_s_note": (
                        "network-inclusive; not a timing measurement; "
                        "non-comparable to local wall clocks; do not cite as latency"
                    ),
                    "usd": call_usd,
                    "running_usd": running_after,
                    "n_tool_use_blocks": len(tool_uses),
                }
            )
            messages.append({"role": "assistant", "content": content_blocks})

            if not tool_uses:
                break
            try:
                decoded = decode_execute_anthropic(tool_uses)
                if is_empty_execute_response(decoded):
                    abort_error = "malformed_tool_call: empty_execute_from_tool_use"
                    break
            except Exception as exc:
                abort_error = f"malformed_tool_call: {type(exc).__name__}: {exc}"
                break

            turn_decoded_steps.append(decoded)
            execution_results, _instances = execute_multi_turn_func_call(
                decoded,
                initial_config,
                involved_classes,
                model_name,
                test_entry_id,
                long_context=("long_context" in test_category or "composite" in test_category),
                is_evaL_run=False,
            )
            tool_result_content = []
            for block, exec_result in zip(tool_uses, execution_results, strict=True):
                tool_result_content.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.get("id") or "",
                        "content": str(exec_result),
                    }
                )
            messages.append({"role": "user", "content": tool_result_content})
            step += 1
            if step > MAXIMUM_STEP_LIMIT:
                force_quit = True
                break

        all_model_response.append(turn_responses)
        all_decoded.append(turn_decoded_steps)
        if abort_error or force_quit:
            break

    return _finish_score()


def write_multi_turn_cloud_markdown(report: dict[str, Any], path: Path) -> None:
    """Numbers only. No wall clocks. No interpretation."""
    local = report.get("local_arm") or {}
    cloud = report.get("cloud_arm") or {}
    local_traj = local.get("trajectory_stored") or local.get("trajectory_rescored") or {}
    cloud_traj = cloud.get("trajectory") or {}
    spend = report.get("spend") or {}
    entry_ids = report.get("entry_ids") or []
    local_dist = local.get("first_failure_turn_distribution") or {}
    cloud_dist = cloud.get("first_failure_turn_distribution") or {}
    local_pe = {r["id"]: r for r in (local.get("per_entry") or []) if "id" in r}
    cloud_pe = {r["id"]: r for r in (cloud.get("per_entry") or []) if "id" in r}

    def _frac(correct: Any, n: Any) -> str:
        if correct is None or n is None:
            return "unrecovered/unrecovered"
        return f"{correct}/{n}"

    lines = [
        "# BFCL multi-turn local vs cloud",
        "",
        f"entries: {len(entry_ids)}",
        f"entry_ids: {json.dumps(entry_ids)}",
        f"entry_ids_match_pinned_file: {report.get('entry_ids_match_pinned_file')}",
        f"entry_ids_match_gpu_report: {report.get('entry_ids_match_gpu_report')}",
        f"checker: {report.get('checker')}",
        f"wrapper: {report.get('wrapper')}",
        "",
        "## Trajectory (official multi_turn_checker)",
        f"local {_frac(local_traj.get('correct'), local_traj.get('n'))}",
        f"cloud {_frac(cloud_traj.get('correct'), cloud_traj.get('n'))}",
        "",
        "## Per-turn (official checker prefixes)",
        f"local {_frac(local.get('turns_correct'), local.get('turns_total'))}",
        f"cloud {_frac(cloud.get('turns_correct'), cloud.get('turns_total'))}",
        f"local_recoverable: {local.get('recoverable')}",
        f"cloud_recoverable: {cloud.get('recoverable')}",
        "",
        "## First-failure turn distribution",
        f"local: {json.dumps(local_dist)}",
        f"cloud: {json.dumps(cloud_dist)}",
        "",
        "id\tlocal_first_failure_turn\tcloud_first_failure_turn",
    ]
    for eid in entry_ids:
        lf = (local_pe.get(eid) or {}).get("first_failure_turn")
        cf = (cloud_pe.get(eid) or {}).get("first_failure_turn")
        lines.append(f"{eid}\t{lf}\t{cf}")
    lines.extend(
        [
            "",
            "## Spend",
            f"prompt_tokens_sum: {spend.get('prompt_tokens_sum')}",
            f"completion_tokens_sum: {spend.get('completion_tokens_sum')}",
            f"usd: {spend.get('usd')}",
            (
                "rates_usd_per_1M: "
                f"in={spend.get('usd_per_mtok_in')} "
                f"out={spend.get('usd_per_mtok_out')}"
            ),
            "",
        ]
    )
    if local.get("recoverable") is False:
        lines.extend(
            [
                "## Local per-turn recovery",
                str(local.get("reason") or local.get("notes")),
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_cloud_multi_turn_scoring_selftest(out_dir: Path) -> dict[str, Any]:
    """Offline: gold decoded through the cloud report/scoring path. No API."""
    entries, pin_path = load_pinned_multi_turn_entries(out_dir)
    entry_ids = [e["id"] for e in entries]
    gold_rows: list[dict[str, Any]] = []
    for entry in entries:
        decoded = gold_decoded_for_entry(entry)
        scored = score_decoded_trajectory(
            entry=entry,
            model_result_decoded=decoded,
            name_prefix=f"cloud_gold_{entry['id']}",
        )
        gold_rows.append(
            {
                "id": entry["id"],
                "score": scored["score"],
                "per_turn": scored["per_turn"],
            }
        )
    gold_summary = summarize_per_turn_arm(gold_rows)
    n_valid = sum(1 for r in gold_rows if r["score"].get("valid") is True)
    gpu_path = out_dir / "multi_turn_gpu_probe_report.json"
    if gpu_path.is_file():
        gpu_report = json.loads(gpu_path.read_text(encoding="utf-8"))
        local_arm = extract_local_arm_from_gpu_report(gpu_report=gpu_report, entries=entries)
        gpu_ids = local_arm.get("gpu_report_entry_ids") or []
        ids_match_gpu = entry_ids == gpu_ids
    else:
        local_arm = {
            "recoverable": False,
            "reason": f"{gpu_path} missing; local per-turn not recovered.",
        }
        ids_match_gpu = False
    report = {
        "probe": "bfcl_feasibility_multi_turn_cloud_scoring_selftest",
        "mode": "run_cloud_multi_turn",
        "selftest": True,
        "api_called": False,
        "pinned_entries_path": str(pin_path),
        "entry_ids": entry_ids,
        "entry_ids_match_pinned_file": True,
        "entry_ids_match_gpu_report": ids_match_gpu,
        "checker": "bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker",
        "wrapper": "apu_characterization.cap01.bfcl_cap01_multi_turn_checker",
        "gold": {
            "n_valid": n_valid,
            "n": len(gold_rows),
            "trajectory": {"correct": n_valid, "n": len(gold_rows)},
            **gold_summary,
            "details": gold_rows,
        },
        "local_arm": local_arm,
    }
    out_path = out_dir / "cloud_multi_turn_scoring_selftest.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    report["artifact"] = str(out_path)
    return report


def run_cloud_multi_turn(
    out_dir: Path,
    *,
    model: str = CLOUD_DEFAULT_MODEL,
    max_tokens: int = 512,
) -> dict[str, Any]:
    """Paid Anthropic arm on the pinned 20 multi_turn_base entries."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Refusing to start the paid "
            "cloud multi-turn run. Export ANTHROPIC_API_KEY in the environment "
            "(never pass it on the command line)."
        )
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError(
            "The anthropic package is not installed in this environment. "
            "Install it into .venv-seam before the paid run."
        ) from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    entries, pin_path = load_pinned_multi_turn_entries(out_dir)
    entry_ids = [e["id"] for e in entries]
    gpu_path = out_dir / "multi_turn_gpu_probe_report.json"
    if gpu_path.is_file():
        gpu_report = json.loads(gpu_path.read_text(encoding="utf-8"))
        local_arm = extract_local_arm_from_gpu_report(gpu_report=gpu_report, entries=entries)
        gpu_ids = local_arm.get("gpu_report_entry_ids") or []
        ids_match_gpu = entry_ids == gpu_ids
    else:
        local_arm = {
            "recoverable": False,
            "reason": (
                f"{gpu_path} missing; cannot recover official per-turn from "
                "stored local trajectories. Local arm not re-run."
            ),
            "trajectory_stored": None,
            "turns_correct": None,
            "turns_total": None,
            "first_failure_turn_distribution": None,
            "per_entry": [],
        }
        ids_match_gpu = False

    # Client picks up ANTHROPIC_API_KEY from the environment. Do not log or store it.
    client = anthropic.Anthropic()
    temperature_sent = model != CLOUD_DEFAULT_MODEL
    results: list[dict[str, Any]] = []
    running_usd = 0.0
    example_tools: dict[str, Any] | None = None
    for entry in entries:
        print(f"[cloud_multi_turn] {entry['id']} …", flush=True)
        if example_tools is None:
            example_tools = anthropic_tools_for_entry(entry)
        row = run_cloud_multi_turn_agent_entry(
            client=client,
            model=model,
            max_tokens=max_tokens,
            entry=entry,
            running_usd=running_usd,
        )
        results.append(row)
        running_usd = sum(float(r.get("usd") or 0.0) for r in results)
        (out_dir / "cloud_multi_turn_partial.json").write_text(
            json.dumps(results, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    n_valid = sum(1 for r in results if r.get("score", {}).get("valid") is True)
    prompt_sum = sum(int(r.get("prompt_tokens_sum") or 0) for r in results)
    completion_sum = sum(int(r.get("completion_tokens_sum") or 0) for r in results)
    usd_total = sum(float(r.get("usd") or 0.0) for r in results)
    cloud_summary = summarize_per_turn_arm(results)
    report = {
        "probe": "bfcl_feasibility_multi_turn_cloud",
        "mode": "run_cloud_multi_turn",
        "selftest": False,
        "pinned_entries_path": str(pin_path),
        "entry_ids": entry_ids,
        "entry_ids_match_pinned_file": True,
        "entry_ids_match_gpu_report": ids_match_gpu,
        "checker": "bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker",
        "wrapper": "apu_characterization.cap01.bfcl_cap01_multi_turn_checker",
        "model": {
            "id": model,
            "temperature": 0,
            "temperature_sent": temperature_sent,
            "temperature_omit_reason": (
                None
                if temperature_sent
                else (
                    "claude-sonnet-5 rejects temperature (400 deprecated); "
                    "omitted rather than sent. Recorded 2026-08-02."
                )
            ),
            "max_tokens": max_tokens,
            "max_tokens_matches_run_gpu_multi_turn_default": max_tokens == 512,
        },
        "tool_format": {
            "api": "anthropic_native_tool_use",
            "qwen_chat_template_tools_text_in_prompt": False,
            "mapping_note": (example_tools or {}).get("mapping_note"),
            "example_entry_id": entries[0]["id"] if entries else None,
            "example_anthropic_tools": (example_tools or {}).get("tools"),
            "example_mapping": (example_tools or {}).get("mapping"),
        },
        "latency_s_note": (
            "Per-call latency_s in per_entry_full.calls is network-inclusive and "
            "not a timing measurement. Do not cite it as latency. This run is "
            "not latency-measured; machine cleanliness gates do not apply."
        ),
        "local_arm": local_arm,
        "cloud_arm": {
            "trajectory": {"correct": n_valid, "n": len(results)},
            "turns_correct": cloud_summary["turns_correct"],
            "turns_total": cloud_summary["turns_total"],
            "first_failure_turn_distribution": cloud_summary["first_failure_turn_distribution"],
            "recoverable": cloud_summary["recoverable"],
            "notes": cloud_summary["notes"],
            "per_entry": cloud_summary["per_entry"],
            "per_entry_full": results,
        },
        "spend": {
            "prompt_tokens_sum": prompt_sum,
            "completion_tokens_sum": completion_sum,
            "usd": usd_total,
            "usd_per_mtok_in": CLOUD_USD_PER_MTOK_IN,
            "usd_per_mtok_out": CLOUD_USD_PER_MTOK_OUT,
            "n_api_errors": sum(1 for r in results if r.get("entry_error")),
        },
    }
    json_path = out_dir / "cloud_multi_turn_report.json"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    md_path = out_dir / "MULTI_TURN_CLOUD_REPORT.md"
    write_multi_turn_cloud_markdown(report, md_path)
    report["artifact"] = str(json_path)
    report["markdown"] = str(md_path)
    return report


def _host_uptime_s() -> float | None:
    """Seconds since boot (GetTickCount64). Additive per-entry field for onset curves."""
    try:
        return round(float(ctypes.windll.kernel32.GetTickCount64()) / 1000.0, 3)
    except Exception:
        return None


def _host_available_mb() -> dict[str, Any]:
    """Available physical memory in MB at call time.

    Uses GlobalMemoryStatusEx.ullAvailPhys (fast, per-entry). Gate scripts use
    Get-Counter '\\Memory\\Available MBytes' (includes standby); method is recorded
    so the two are not silently equated.
    """
    try:

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            raise OSError("GlobalMemoryStatusEx failed")
        return {
            "available_mb": round(float(stat.ullAvailPhys) / (1024.0 * 1024.0), 1),
            "available_method": "GlobalMemoryStatusEx.ullAvailPhys",
        }
    except Exception as exc:
        return {
            "available_mb": None,
            "available_method": f"unreadable:{type(exc).__name__}",
        }


def run_session_residency(
    out_dir: Path,
    *,
    arm_id: str,
    residency_mode: str,
    n_entries: int = 20,
    max_new_tokens: int = 512,
    seed: int = SESSION_RESIDENCY_SEED,
) -> dict[str, Any]:
    """Session-level RESIDENT / NON_RESIDENT A/B cell on real BFCL multi_turn_base."""
    import openvino_genai as ov_genai

    mode = residency_mode.upper()
    if mode not in RESIDENCY_MODES:
        raise ValueError(f"residency_mode must be one of {RESIDENCY_MODES}, got {residency_mode!r}")
    if n_entries < 1:
        raise ValueError("n_entries must be >= 1")

    out_dir.mkdir(parents=True, exist_ok=True)
    all_entries = select_multi_turn_entries()
    # Paired cells: identical file-order prefix; seed recorded (no shuffle).
    # select_multi_turn_entries applies no difficulty or API filter — plain questions[:n].
    entries = all_entries[:n_entries]
    reduced_reason = None

    entries_path = out_dir / f"session_residency_entries_{arm_id}_{mode}.json"
    entries_path.write_text(json.dumps(entries, indent=2, default=str) + "\n", encoding="utf-8")
    gold = run_multi_turn_gold_selftest(entries)
    (out_dir / f"session_residency_gold_{arm_id}_{mode}.json").write_text(
        json.dumps(gold, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    # Scorer selftest is a hard gate: gold path must be 20/20 before any generation.
    n_gold = int(gold.get("n") or 0)
    n_valid = int(gold.get("n_valid") or 0)
    if n_gold != len(entries) or n_valid != n_gold or n_gold < 1:
        raise SystemExit(
            f"REFUSED -- multi_turn_checker gold selftest {n_valid}/{n_gold} "
            f"(need {len(entries)}/{len(entries)}) before arm {arm_id}/{mode} runs"
        )

    tokenizer = _hf_tokenizer()
    # DISPATCH L: NON_RESIDENT must disable ContinuousBatching prefix caching.
    # RESIDENT keeps LLMPipeline default (prefix caching ON) for history continuation.
    pipe, load_meta, load_s = load_arm_pipeline(
        arm_id,
        enable_prefix_caching=False if mode == "NON_RESIDENT" else None,
    )
    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = max_new_tokens
    cfg.do_sample = False
    cfg.apply_chat_template = False

    results: list[dict[str, Any]] = []
    partial_name = f"session_residency_{arm_id}_{mode}_partial.json"
    for entry in entries:
        # Per-entry host state (X-2): long arms span hours; onset is within-arm.
        uptime_s = _host_uptime_s()
        avail = _host_available_mb()
        print(
            f"[session_residency {arm_id}/{mode}] {entry['id']} "
            f"uptime_s={uptime_s} available_mb={avail.get('available_mb')} …",
            flush=True,
        )
        row = run_multi_turn_agent_entry(
            pipe=pipe,
            tokenizer=tokenizer,
            cfg=cfg,
            entry=entry,
            residency_mode=mode,
            ov_genai=ov_genai,
        )
        row["uptime_s"] = uptime_s
        row["available_mb"] = avail.get("available_mb")
        row["available_method"] = avail.get("available_method")
        results.append(row)
        (out_dir / partial_name).write_text(
            json.dumps(results, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    turn_ttfts: list[float | None] = []
    turn_decodes: list[float | None] = []
    turn_slo: list[bool] = []
    session_latencies: list[float] = []
    per_turn_acc_correct = 0
    per_turn_acc_n = 0
    all_deltas: list[int] = []
    growths: list[int] = []
    for r in results:
        session_latencies.append(float(r["wall_s"]))
        growths.append(int(r.get("context_growth_delta_tokens") or 0))
        for t in r.get("turn_metrics") or []:
            turn_ttfts.append(t.get("ttft_s"))
            turn_decodes.append(t.get("decode_tok_s"))
            if t.get("slo_ok") is not None:
                turn_slo.append(bool(t["slo_ok"]))
            acc = t.get("per_turn_accuracy")
            if acc is not None:
                per_turn_acc_n += 1
                if acc.get("correct"):
                    per_turn_acc_correct += 1
            d = t.get("delta_tokens_vs_prev_turn")
            if d is not None:
                all_deltas.append(int(d))

    traj_correct = sum(1 for r in results if r["score"].get("valid") is True)
    report = {
        "probe": "bfcl_session_residency",
        "mode": "run_session_residency",
        "arm_id": arm_id,
        "residency_mode": mode,
        "seed": seed,
        "n_entries_requested": n_entries,
        "n_entries_run": len(entries),
        "n_entries_reduced_reason": reduced_reason,
        "slo": {"ttft_s_max": SLO_TTFT_S, "decode_tok_s_min": SLO_DECODE_TOK_S},
        "model": _model_report_fields(
            enable_thinking=False,
            arm=load_meta.get("device_config"),
            loads=load_meta.get("loads"),
            apply_chat_template_at_generate=False,
        ),
        "multi_turn_checker_gold_selftest": gold,
        "gpu_probe": {
            "status": "complete",
            "isolation_mode": "OPERATOR_ASSERTED_CLEAN",
            "model_load_s": load_s,
            "max_new_tokens": max_new_tokens,
            "maximum_step_limit": MAXIMUM_STEP_LIMIT,
            "prompt_format": "bfcl_chat_history_tools_thinking_off",
            "fix_approach": (
                "ChatHistory with set_tools + set_extra_context(enable_thinking=False); "
                "never feed a pre-rendered multi-role string into start_chat "
                "(DISPATCH J double-template root cause)."
            ),
            "residency_semantics": {
                "RESIDENT": (
                    "one ChatHistory held across the entry; generate(ChatHistory) applies "
                    "the template once per call (tools + enable_thinking=False) and retains "
                    "KV via history-prefix continuation"
                ),
                "NON_RESIDENT": (
                    "same ChatHistory render each step; cold generate([rendered]) with "
                    "apply_chat_template=False; pipeline loaded with "
                    "SchedulerConfig(enable_prefix_caching=False) so CB cannot reuse KV "
                    "(DISPATCH L). Rejected alternative: fresh LLMPipeline per turn — "
                    "unnecessary once prefix caching is disabled, and would conflate load "
                    "time with TTFT."
                ),
            }[mode],
            "cold_fix": {
                "chosen": "a_disable_prefix_caching",
                "enable_prefix_caching": (False if mode == "NON_RESIDENT" else None),
                "scheduler_config": load_meta.get("scheduler_config"),
                "rejected": "b_fresh_pipeline_per_turn",
                "rejection_reason": (
                    "openvino_genai 2026.2.1 LLMPipeline ContinuousBatching backend "
                    "exposes SchedulerConfig.enable_prefix_caching; LLMPipeline client "
                    "default is ON (issue #2415). Explicit False is sufficient to force "
                    "full cold prefill without reloading weights each turn."
                ),
            },
            "per_entry": results,
            "per_entry_host_onset": {
                "fields": ["uptime_s", "available_mb", "available_method"],
                "uptime_method": "GetTickCount64_s",
                "available_method_note": (
                    "Per-entry available_mb uses GlobalMemoryStatusEx.ullAvailPhys; "
                    "launch gates use Get-Counter Available MBytes (includes standby). "
                    "Methods differ; do not equate without conversion."
                ),
                "purpose": (
                    "Within-arm onset: regress entry TTFT / session latency against "
                    "uptime_s across a long cell (cpu-p NON_RESIDENT ~9630 s). "
                    "If TTFT trends up with uptime_s inside one arm, that curve "
                    "replaces the chosen 2 h gate with a measured knee."
                ),
            },
            "accuracy_trajectory": {
                "correct": traj_correct,
                "n": len(results),
                "accuracy": traj_correct / len(results) if results else None,
            },
            "accuracy_per_turn_f2": {
                "correct": per_turn_acc_correct,
                "n": per_turn_acc_n,
                "accuracy": (per_turn_acc_correct / per_turn_acc_n if per_turn_acc_n else None),
            },
            "session_total_latency_s": _stats_float(session_latencies),
            "turn_ttft_s": _stats_float_list(turn_ttfts),
            "turn_decode_tok_s": _stats_float_list(turn_decodes),
            "fraction_turns_slo_ok": (
                sum(1 for x in turn_slo if x) / len(turn_slo) if turn_slo else None
            ),
            "n_turns_slo_ok": sum(1 for x in turn_slo if x),
            "n_turns_slo_scored": len(turn_slo),
            "context_growth_delta_tokens_per_entry": _stats_int(growths),
            "per_turn_delta_tokens": _stats_int(all_deltas),
            "stack": {
                "openvino": load_meta.get("openvino"),
                "openvino_genai": load_meta.get("openvino_genai"),
            },
        },
    }
    out_name = f"session_residency_{arm_id}_{mode}_report.json"
    (out_dir / out_name).write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    report["artifact"] = str(out_dir / out_name)
    return report


def run_session_residency_cold_control(
    out_dir: Path,
    *,
    arm_id: str = "gpu_only",
    max_new_tokens: int = 512,
    seed: int = SESSION_RESIDENCY_SEED,
) -> dict[str, Any]:
    """Positive control: NON_RESIDENT turn-2 must be full cold prefill, not delta.

    One multi_turn_base entry. Asserts:
      - turn-2 generate_input_tokens == full conversation length (>> delta)
      - metrics prompt_tokens_reported ≈ full length when present (not delta-sized)
      - turn-2 TTFT within cold tolerance of length-scaled turn-1 (and absolute floor)
    Fails loudly if turn-2 costs delta-sized time (prefix-cache contamination).
    """
    import openvino_genai as ov_genai

    out_dir.mkdir(parents=True, exist_ok=True)
    entry = select_multi_turn_entries()[0]
    if len(entry.get("question") or []) < 2:
        raise RuntimeError(
            f"cold control requires >=2 user turns; entry {entry['id']} has "
            f"{len(entry.get('question') or [])}"
        )

    tokenizer = _hf_tokenizer()
    pipe, load_meta, load_s = load_arm_pipeline(arm_id, enable_prefix_caching=False)
    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = max_new_tokens
    cfg.do_sample = False
    cfg.apply_chat_template = False

    print(
        f"[cold_control {arm_id}/NON_RESIDENT] {entry['id']} "
        f"(prefix_caching=False, load_s={load_s:.2f}) …",
        flush=True,
    )
    result = run_multi_turn_agent_entry(
        pipe=pipe,
        tokenizer=tokenizer,
        cfg=cfg,
        entry=entry,
        residency_mode="NON_RESIDENT",
        ov_genai=ov_genai,
    )

    turns = result.get("turn_metrics") or []
    failures: list[str] = []
    if len(turns) < 2:
        failures.append(
            f"need turn-2 metrics; got n_completed_turns={len(turns)} (entry={entry['id']})"
        )
        turn1 = turns[0] if turns else {}
        turn2 = {}
    else:
        turn1 = turns[0]
        turn2 = turns[1]

    n1 = turn1.get("prompt_tokens")
    n2 = turn2.get("prompt_tokens")
    delta = turn2.get("delta_tokens_vs_prev_turn")
    ttft1 = turn1.get("ttft_s")
    ttft2 = turn2.get("ttft_s")
    # Prefer step-0 generate_input_tokens / reported from the first generation of turn 2.
    step0_t2 = None
    for s in turn2.get("steps") or []:
        if s.get("step") == 0:
            step0_t2 = s
            break
    gen_in_t2 = step0_t2.get("generate_input_tokens") if step0_t2 else turn2.get("prompt_tokens")
    reported_t2 = step0_t2.get("prompt_tokens_reported") if step0_t2 else None

    cold_floor = float(NON_RESIDENT_COLD_TTFT_FLOOR_S.get(arm_id, 1.0))
    checks: dict[str, Any] = {
        "entry_id": entry["id"],
        "arm_id": arm_id,
        "model_load_s": load_s,
        "turn1_prompt_tokens": n1,
        "turn2_prompt_tokens": n2,
        "turn2_delta_tokens": delta,
        "turn2_generate_input_tokens": gen_in_t2,
        "turn2_prompt_tokens_reported": reported_t2,
        "turn1_ttft_s": ttft1,
        "turn2_ttft_s": ttft2,
        "cold_ttft_floor_s": cold_floor,
        "cold_ttft_ratio_min": NON_RESIDENT_COLD_TTFT_RATIO_MIN,
        "warm_ttft_ratio_max": NON_RESIDENT_WARM_TTFT_RATIO_MAX,
    }

    if n2 is None or gen_in_t2 is None:
        failures.append("turn-2 prompt/generate_input token counts missing")
    else:
        if int(gen_in_t2) != int(n2):
            failures.append(f"turn-2 generate_input_tokens={gen_in_t2} != full prompt_tokens={n2}")
        if delta is not None:
            if int(gen_in_t2) <= int(delta) * 2:
                failures.append(
                    f"turn-2 generate_input_tokens={gen_in_t2} looks delta-sized "
                    f"(delta={delta}); expected full conversation length"
                )
            if reported_t2 is not None and int(reported_t2) <= int(delta) * 2:
                failures.append(
                    f"turn-2 prompt_tokens_reported={reported_t2} is delta-sized "
                    f"(delta={delta}); expected full prefill token count"
                )
        if reported_t2 is not None and n2:
            # Allow small tokenizer/path drift; forbid delta-shaped reports.
            rel = abs(int(reported_t2) - int(n2)) / max(int(n2), 1)
            checks["turn2_reported_vs_full_rel_err"] = rel
            if rel > 0.10 and (delta is None or int(reported_t2) <= int(delta) * 2):
                failures.append(
                    f"turn-2 prompt_tokens_reported={reported_t2} diverges from full "
                    f"n2={n2} (rel_err={rel:.3f}) and is not full-length"
                )

    if ttft1 is None or ttft2 is None or not n1 or not n2:
        failures.append(
            f"missing TTFT/token inputs for cold check: ttft1={ttft1} ttft2={ttft2} n1={n1} n2={n2}"
        )
    else:
        scaled_floor = float(ttft1) * (float(n2) / float(n1)) * NON_RESIDENT_COLD_TTFT_RATIO_MIN
        checks["turn2_ttft_scaled_floor_s"] = scaled_floor
        checks["turn2_over_turn1_raw"] = float(ttft2) / float(ttft1)
        # Absolute floor catches arm-specific warm contamination.
        if float(ttft2) < cold_floor:
            failures.append(
                f"FATAL: turn-2 TTFT={ttft2:.4f}s < cold floor {cold_floor:.2f}s "
                f"for arm={arm_id} (delta-sized / prefix-cache hit)"
            )
        # Relative to length-scaled turn-1 cold prefill.
        if float(ttft2) < scaled_floor:
            failures.append(
                f"FATAL: turn-2 TTFT={ttft2:.4f}s < {NON_RESIDENT_COLD_TTFT_RATIO_MIN:.2f}x "
                f"length-scaled turn-1 cold ({scaled_floor:.4f}s); "
                f"ttft1={ttft1:.4f}s n1={n1} n2={n2}"
            )
        # Explicit warm-ratio trap (contaminated gpu_only was ~0.13/2.6 ≈ 0.05).
        if float(ttft2) < float(ttft1) * NON_RESIDENT_WARM_TTFT_RATIO_MAX:
            failures.append(
                f"FATAL: turn-2 TTFT={ttft2:.4f}s is < "
                f"{NON_RESIDENT_WARM_TTFT_RATIO_MAX:.2f}x turn-1={ttft1:.4f}s "
                f"(prefix reuse / delta-sized cost)"
            )

    passed = len(failures) == 0
    report = {
        "probe": "bfcl_session_residency_cold_control",
        "mode": "session_residency_cold_control",
        "seed": seed,
        "passed": passed,
        "failures": failures,
        "checks": checks,
        "cold_fix": {
            "chosen": "a_disable_prefix_caching",
            "rejected": "b_fresh_pipeline_per_turn",
            "rejection_reason": (
                "SchedulerConfig.enable_prefix_caching=False is available on "
                "LLMPipeline via config={'scheduler_config': ...} (openvino_genai "
                "2026.2.1; CB backend defaults prefix caching ON — issue #2415). "
                "Fresh pipeline per turn rejected unless this control fails."
            ),
            "scheduler_config": load_meta.get("scheduler_config"),
            "model_load_s_separate_from_ttft": load_s,
        },
        "stack": {
            "openvino": load_meta.get("openvino"),
            "openvino_genai": load_meta.get("openvino_genai"),
        },
        "model": _model_report_fields(
            arm=load_meta.get("device_config"),
            loads=load_meta.get("loads"),
        ),
        "entry_result": result,
    }
    path = out_dir / f"session_residency_cold_control_{arm_id}.json"
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    report["artifact"] = str(path)
    if passed:
        print(
            f"[cold_control] PASS arm={arm_id} turn2_ttft={ttft2}s "
            f"n2={n2} delta={delta} reported={reported_t2}",
            flush=True,
        )
    else:
        print("[cold_control] FAIL — NON_RESIDENT is not cold:", flush=True)
        for f in failures:
            print(f"  - {f}", flush=True)
    return report


def run_session_residency_render_smoke(out_dir: Path) -> dict[str, Any]:
    """Offline first-turn equivalence smoke (no model load / no generate).

    Builds RESIDENT and NON_RESIDENT intended renders for the paired 20-entry
    prefix and fails loudly on any token mismatch or thinking-on tail.
    """
    import openvino_genai as ov_genai

    out_dir.mkdir(parents=True, exist_ok=True)
    entries = select_multi_turn_entries()[:20]
    hf_tokenizer = _hf_tokenizer()
    genai_tokenizer = _genai_tokenizer(ov_genai)
    rows: list[dict[str, Any]] = []
    for entry in entries:
        tools = tools_for_entry(entry)
        messages = messages_for_entry(entry)
        row = assert_first_turn_token_equivalence(
            hf_tokenizer=hf_tokenizer,
            genai_tokenizer=genai_tokenizer,
            ov_genai=ov_genai,
            messages=messages,
            tools=tools,
            entry_id=str(entry["id"]),
        )
        rows.append(row)
    report = {
        "probe": "bfcl_session_residency_render_smoke",
        "mode": "session_residency_render_smoke",
        "n_entries": len(rows),
        "all_identical": all(r["identical"] for r in rows),
        "fix_approach": (
            "ChatHistory + set_tools + enable_thinking=False; NON_RESIDENT uses the "
            "same render as a cold string (no start_chat double-template)."
        ),
        "per_entry": rows,
        "model": _model_report_fields(),
        "stack": {
            "openvino_genai": getattr(ov_genai, "__version__", "unknown"),
        },
    }
    path = out_dir / "session_residency_render_smoke.json"
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    report["artifact"] = str(path)
    return report


def compare_session_residency_cells(
    resident_report: dict[str, Any],
    non_resident_report: dict[str, Any],
) -> dict[str, Any]:
    """Paired comparisons for one arm (RESIDENT vs NON_RESIDENT reports)."""
    arm = resident_report.get("arm_id")
    r_entries = {e["id"]: e for e in resident_report.get("gpu_probe", {}).get("per_entry", [])}
    n_entries = {e["id"]: e for e in non_resident_report.get("gpu_probe", {}).get("per_entry", [])}
    common = sorted(set(r_entries) & set(n_entries))
    ttft_r: list[float] = []
    ttft_n: list[float] = []
    lat_r: list[float] = []
    lat_n: list[float] = []
    acc_diff_turns = 0
    acc_compared_turns = 0
    raw_diff_gens = 0
    raw_compared_gens = 0
    think_r = 0
    think_n = 0
    gen_tok_r: list[int] = []
    gen_tok_n: list[int] = []
    hit_cap_r = 0
    hit_cap_n = 0
    for eid in common:
        er, en = r_entries[eid], n_entries[eid]
        lat_r.append(float(er["wall_s"]))
        lat_n.append(float(en["wall_s"]))
        raw_r = er.get("model_result_raw") or []
        raw_n = en.get("model_result_raw") or []
        for turn_r, turn_n in zip(raw_r, raw_n, strict=False):
            for tr_txt, tn_txt in zip(turn_r, turn_n, strict=False):
                raw_compared_gens += 1
                if tr_txt != tn_txt:
                    raw_diff_gens += 1
                if THINK_OPEN_RE.search(tr_txt or ""):
                    think_r += 1
                if THINK_OPEN_RE.search(tn_txt or ""):
                    think_n += 1
        for tr, tn in zip(er.get("turn_metrics") or [], en.get("turn_metrics") or [], strict=False):
            if tr.get("ttft_s") is not None:
                ttft_r.append(float(tr["ttft_s"]))
            if tn.get("ttft_s") is not None:
                ttft_n.append(float(tn["ttft_s"]))
            if tr.get("generated_tokens") is not None:
                gt = int(tr["generated_tokens"])
                gen_tok_r.append(gt)
                if gt >= 512:
                    hit_cap_r += 1
            if tn.get("generated_tokens") is not None:
                gt = int(tn["generated_tokens"])
                gen_tok_n.append(gt)
                if gt >= 512:
                    hit_cap_n += 1
            ar = (tr.get("per_turn_accuracy") or {}).get("correct")
            an = (tn.get("per_turn_accuracy") or {}).get("correct")
            if ar is not None and an is not None:
                acc_compared_turns += 1
                if bool(ar) != bool(an):
                    acc_diff_turns += 1

    def _median(vals: list[float]) -> float | None:
        if not vals:
            return None
        s = sorted(vals)
        m = len(s) // 2
        return s[m] if len(s) % 2 else 0.5 * (s[m - 1] + s[m])

    sum_lat_r = sum(lat_r)
    sum_lat_n = sum(lat_n)
    return {
        "arm_id": arm,
        "n_common_entries": len(common),
        "entry_ids": common,
        "ttft_s": {
            "RESIDENT": _stats_float(ttft_r),
            "NON_RESIDENT": _stats_float(ttft_n),
            "median_RESIDENT": _median(ttft_r),
            "median_NON_RESIDENT": _median(ttft_n),
            "median_ratio_R_over_N": (
                (_median(ttft_r) / _median(ttft_n))
                if _median(ttft_r) is not None and _median(ttft_n) not in (None, 0)
                else None
            ),
        },
        "fraction_turns_slo_ok": {
            "RESIDENT": resident_report.get("gpu_probe", {}).get("fraction_turns_slo_ok"),
            "NON_RESIDENT": non_resident_report.get("gpu_probe", {}).get("fraction_turns_slo_ok"),
        },
        "session_total_latency": {
            "RESIDENT_sum_s": sum_lat_r,
            "NON_RESIDENT_sum_s": sum_lat_n,
            "ratio_R_over_N": (sum_lat_r / sum_lat_n) if sum_lat_n else None,
            "RESIDENT_stats": _stats_float(lat_r),
            "NON_RESIDENT_stats": _stats_float(lat_n),
        },
        "per_turn_accuracy": {
            "RESIDENT": resident_report.get("gpu_probe", {}).get("accuracy_per_turn_f2"),
            "NON_RESIDENT": non_resident_report.get("gpu_probe", {}).get("accuracy_per_turn_f2"),
            "n_turns_compared_paired": acc_compared_turns,
            "n_turns_differing": acc_diff_turns,
            "note": (
                "Open question: with double-templating removed, greedy RESIDENT and "
                "NON_RESIDENT outputs should match; any difference is stack-side "
                "non-equivalence."
            ),
        },
        "output_identity": {
            "n_generations_compared_paired": raw_compared_gens,
            "n_generations_text_differing": raw_diff_gens,
            "identical": raw_compared_gens > 0 and raw_diff_gens == 0,
            "n_generations_with_think_RESIDENT": think_r,
            "n_generations_with_think_NON_RESIDENT": think_n,
        },
        "over_generation": {
            "mean_generated_tokens_RESIDENT": (
                sum(gen_tok_r) / len(gen_tok_r) if gen_tok_r else None
            ),
            "mean_generated_tokens_NON_RESIDENT": (
                sum(gen_tok_n) / len(gen_tok_n) if gen_tok_n else None
            ),
            "n_turns_hit_max_new_tokens_RESIDENT": hit_cap_r,
            "n_turns_hit_max_new_tokens_NON_RESIDENT": hit_cap_n,
            "n_turns_scored_RESIDENT": len(gen_tok_r),
            "n_turns_scored_NON_RESIDENT": len(gen_tok_n),
        },
        "context_growth": {
            "RESIDENT": resident_report.get("gpu_probe", {}).get(
                "context_growth_delta_tokens_per_entry"
            ),
            "NON_RESIDENT": non_resident_report.get("gpu_probe", {}).get(
                "context_growth_delta_tokens_per_entry"
            ),
            "per_turn_delta_tokens_RESIDENT": resident_report.get("gpu_probe", {}).get(
                "per_turn_delta_tokens"
            ),
            "per_turn_delta_tokens_NON_RESIDENT": non_resident_report.get("gpu_probe", {}).get(
                "per_turn_delta_tokens"
            ),
        },
    }


def run_attention_window_inventory(out_dir: Path) -> dict[str, Any]:
    """Offline: does GenAI 2026.2.1 expose sliding-window / attention-sink KV eviction?"""
    import openvino_genai as ov_genai

    sc = ov_genai.SchedulerConfig()
    sa = ov_genai.SparseAttentionConfig()
    report = {
        "probe": "attention_window_support",
        "mode": "attention_window_inventory",
        "stack": {
            "openvino_genai": getattr(ov_genai, "__version__", "unknown"),
            "has_PipelineConfig": hasattr(ov_genai, "PipelineConfig"),
            "has_CacheEvictionConfig": hasattr(ov_genai, "CacheEvictionConfig"),
            "has_SchedulerConfig": hasattr(ov_genai, "SchedulerConfig"),
            "has_SparseAttentionConfig": hasattr(ov_genai, "SparseAttentionConfig"),
            "has_ContinuousBatchingPipeline": hasattr(ov_genai, "ContinuousBatchingPipeline"),
        },
        "verdict": {
            "axis_exists": True,
            "mechanism": (
                "SchedulerConfig.use_cache_eviction + SchedulerConfig.cache_eviction_config "
                "(openvino_genai.CacheEvictionConfig)"
            ),
            "attention_sink_field": "CacheEvictionConfig.start_size",
            "sliding_recent_field": "CacheEvictionConfig.recent_size",
            "max_cache_field": "CacheEvictionConfig.max_cache_size",
            "enable_flag": "SchedulerConfig.use_cache_eviction",
            "pipeline_path": (
                "ContinuousBatchingPipeline, or LLMPipeline with "
                '{"scheduler_config": SchedulerConfig(...)}'
            ),
            "pipeline_config_class": None,
            "plain_plugin_property_named_attention_sink": False,
            "plain_plugin_property_named_sliding_window": False,
            "note": (
                "No PipelineConfig class in this package. No ATTENTION_SINK / SLIDING_WINDOW "
                "plugin property on Core SUPPORTED_PROPERTIES. Eviction is the GenAI "
                "CacheEvictionConfig path (start tokens retained = sink; recent tokens = "
                "window). SparseAttentionConfig.TRISHAPE also exposes "
                "num_retained_start_tokens_in_cache / num_retained_recent_tokens_in_cache "
                "for sparse prefill, which is related but not KV eviction."
            ),
        },
        "scheduler_config_defaults": {
            "use_cache_eviction": bool(sc.use_cache_eviction),
            "use_sparse_attention": bool(sc.use_sparse_attention),
            "cache_eviction_config": str(sc.cache_eviction_config),
            "sparse_attention_config": {
                "mode": str(sa.mode),
                "num_retained_start_tokens_in_cache": int(sa.num_retained_start_tokens_in_cache),
                "num_retained_recent_tokens_in_cache": int(sa.num_retained_recent_tokens_in_cache),
            },
        },
        "aggregation_modes": [x for x in dir(ov_genai.AggregationMode) if x.isupper()],
        "sparse_attention_modes": [x for x in dir(ov_genai.SparseAttentionMode) if x.isupper()],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "attention_window_inventory.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["artifact"] = str(path)
    return report


def run_kv_precision_property_smoke(out_dir: Path) -> dict[str, Any]:
    """Offline: confirm GPU accepts KV_CACHE_PRECISION={f16,u8,u4} via Core set/get.

    Pipeline-load readback (effect under LLMPipeline) still requires a clean gpu_only run.
    """
    import openvino as ov
    from openvino import Type

    core = ov.Core()
    device = "GPU"
    before = read_kv_cache_precision(device, core=core)
    rows: list[dict[str, Any]] = []
    for name in KV_PRECISION_LEVELS:
        try:
            core.set_property(device, {"KV_CACHE_PRECISION": _ov_type_for_kv_precision(name)})
            after = read_kv_cache_precision(device, core=core)
            rows.append(
                {
                    "requested": name,
                    "set_ok": True,
                    "readback": after,
                    "took_effect": after.get("normalized") == name,
                    "error": None,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "requested": name,
                    "set_ok": False,
                    "readback": None,
                    "took_effect": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    # Restore dynamic default so we do not leave a sticky Core-level override.
    try:
        core.set_property(device, {"KV_CACHE_PRECISION": Type.dynamic})
        restored = read_kv_cache_precision(device, core=core)
    except Exception as exc:
        restored = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    report = {
        "probe": "kv_precision_property_smoke",
        "mode": "kv_precision_property_smoke",
        "property": "KV_CACHE_PRECISION",
        "device": device,
        "openvino": ov.__version__,
        "before": before,
        "levels": rows,
        "restored": restored,
        "pipeline_load_confirmation": {
            "status": "not_run",
            "reason": (
                "Core set/get on one ov.Core confirms the property is accepted and "
                "sticky. Confirming LLMPipeline compilation honors it requires "
                "run_gpu_kv_precision on a clean host (Available>=7000, no Tier-1)."
            ),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "kv_precision_property_smoke.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["artifact"] = str(path)
    return report


def _score_single_turn_entry(
    *,
    pipe: Any,
    tokenizer: Any,
    cfg: Any,
    entry: dict[str, Any],
) -> dict[str, Any]:
    tools = tools_for_entry(entry)
    prompt = render_bfcl_tools_style(tokenizer, messages_for_entry(entry), tools)
    t0 = time.perf_counter()
    gen = pipe.generate([prompt], cfg)
    wall_s = time.perf_counter() - t0
    texts = getattr(gen, "texts", None)
    text = str(texts[0]) if texts else str(gen)
    metrics = getattr(gen, "perf_metrics", None)
    prompt_tokens = None
    completion_tokens = None
    if metrics is not None:
        try:
            prompt_tokens = int(metrics.get_num_input_tokens())
            completion_tokens = int(metrics.get_num_generated_tokens())
        except Exception:  # optional
            pass
    if prompt_tokens is None:
        prompt_tokens = len(tokenizer(prompt)["input_ids"])
    if completion_tokens is None:
        completion_tokens = len(tokenizer(text)["input_ids"])
    parsed = extract_tool_calls_ast(text)
    if parsed is None:
        score: dict[str, Any] = {
            "valid": False,
            "error": ["unparseable_tool_calls"],
            "error_type": "probe:parse",
        }
    else:
        try:
            score = score_ast(
                functions=entry["function"],
                candidate=parsed,
                reference=entry["reference"],
                test_category=entry["category"],
            )
        except Exception as exc:
            score = {
                "valid": False,
                "error": [f"{type(exc).__name__}: {exc}"],
                "error_type": "probe:ast_exception",
            }
    return {
        "id": entry["id"],
        "category": entry["category"],
        "kind": entry["kind"],
        "wall_s": wall_s,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "text_head": text[:500],
        "parsed_ast": parsed,
        "score": score,
    }


def run_gpu_kv_precision(
    out_dir: Path,
    *,
    max_new_tokens: int = 512,
    precisions: tuple[str, ...] = KV_PRECISION_LEVELS,
) -> dict[str, Any]:
    """gpu_only AST accuracy at KV_CACHE_PRECISION in {f16,u8,u4}.

    Exact property: OpenVINO ``KV_CACHE_PRECISION``, passed as LLMPipeline device config
    and confirmed via Core get_property readback after each load.
    """
    import gc

    import openvino as ov
    import openvino_genai as ov_genai
    from openvino import Type

    out_dir.mkdir(parents=True, exist_ok=True)
    entries = select_ast_precision_entries()
    (out_dir / "kv_precision_probe_entries.json").write_text(
        json.dumps(entries, indent=2, default=str) + "\n", encoding="utf-8"
    )
    tokenizer = _hf_tokenizer()
    ir_bin = MODEL_DIR / "openvino_model.bin"
    bin_sha = _sha256_file(ir_bin) if ir_bin.is_file() else None

    per_precision: dict[str, Any] = {}
    for prec in precisions:
        print(f"[kv_precision] loading GPU with KV_CACHE_PRECISION={prec} …", flush=True)
        # One Core for set + readback; also pass the same property into LLMPipeline
        # so compilation cannot silently ignore a Core-only sticky default.
        core = ov.Core()
        ov_type = _ov_type_for_kv_precision(prec)
        before = read_kv_cache_precision("GPU", core=core)
        core.set_property("GPU", {"KV_CACHE_PRECISION": ov_type})
        after_set = read_kv_cache_precision("GPU", core=core)
        t_load0 = time.perf_counter()
        pipe = _make_llm_pipeline(
            ov_genai,
            "GPU",
            {"KV_CACHE_PRECISION": ov_type},
        )
        load_s = time.perf_counter() - t_load0
        after_load = read_kv_cache_precision("GPU", core=core)
        took_effect = after_set.get("normalized") == prec and after_load.get("normalized") == prec

        cfg = ov_genai.GenerationConfig()
        cfg.max_new_tokens = max_new_tokens
        cfg.do_sample = False
        cfg.apply_chat_template = False

        results: list[dict[str, Any]] = []
        for entry in entries:
            print(f"[kv_precision:{prec}] {entry['id']} …", flush=True)
            results.append(
                _score_single_turn_entry(pipe=pipe, tokenizer=tokenizer, cfg=cfg, entry=entry)
            )
            (out_dir / f"kv_precision_{prec}_partial.json").write_text(
                json.dumps(results, indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )

        by_cat: dict[str, dict[str, int]] = {}
        for r in results:
            cat = r["category"]
            by_cat.setdefault(cat, {"n": 0, "correct": 0})
            by_cat[cat]["n"] += 1
            if r["score"].get("valid") is True:
                by_cat[cat]["correct"] += 1
        correct = sum(1 for r in results if r["score"].get("valid") is True)
        per_precision[prec] = {
            "requested": prec,
            "property": "KV_CACHE_PRECISION",
            "property_value_passed": str(ov_type),
            "readback_before_load": before,
            "readback_after_set": after_set,
            "readback_after_load": after_load,
            "took_effect": took_effect,
            "model_load_s": load_s,
            "per_entry": results,
            "accuracy_ast": {
                cat: {
                    "correct": v["correct"],
                    "n": v["n"],
                    "accuracy": v["correct"] / v["n"] if v["n"] else None,
                }
                for cat, v in by_cat.items()
            },
            "accuracy_overall": {
                "correct": correct,
                "n": len(results),
                "accuracy": correct / len(results) if results else None,
            },
        }
        del pipe
        gc.collect()
        with contextlib.suppress(Exception):
            core.set_property("GPU", {"KV_CACHE_PRECISION": Type.dynamic})

    report = {
        "probe": "bfcl_kv_precision_quality",
        "mode": "run_gpu_kv_precision",
        "status": "complete",
        "isolation_mode": "OPERATOR_ASSERTED_CLEAN",
        "model": _model_report_fields(
            openvino_model_bin_sha256=bin_sha,
            enable_thinking=False,
            sealed_arm={
                "id": "gpu_only",
                "load_sequence": ["GPU"],
                "generate_device": "GPU",
                "apply_chat_template_at_generate": False,
            },
        ),
        "kv_cache_precision_property": "KV_CACHE_PRECISION",
        "precisions_requested": list(precisions),
        "n_entries": len(entries),
        "entry_spec": list(AST_PRECISION_FILES),
        "prompt_format": "bfcl_tools_style",
        "max_new_tokens": max_new_tokens,
        "per_precision": per_precision,
        "accuracy_summary": {p: per_precision[p]["accuracy_overall"] for p in precisions},
        "effect_summary": {
            p: {
                "took_effect": per_precision[p]["took_effect"],
                "readback_after_load": per_precision[p]["readback_after_load"],
            }
            for p in precisions
        },
    }
    path = out_dir / "kv_precision_gpu_probe_report.json"
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return report


def run_npu_load(
    out_dir: Path,
    *,
    max_prompt_lens: tuple[int, ...] = (512, 1024, 2048, 4096),
    prefill_chunk: int = NPU_DEFAULT_PREFILL_CHUNK,
) -> dict[str, Any]:
    """Try load_sequence [NPU] for Qwen3-4B-int4-ov; report compile errors + context ceiling."""
    import gc

    import openvino as ov
    import openvino_genai as ov_genai

    out_dir.mkdir(parents=True, exist_ok=True)
    core = ov.Core()
    devices = list(core.available_devices)
    attempts: list[dict[str, Any]] = []
    max_ok: int | None = None

    for mpl in max_prompt_lens:
        # GenAI accepts MAX_PROMPT_LEN; NPU plugin also documents NPUW_LLM_MAX_PROMPT_LEN.
        props = {
            "MAX_PROMPT_LEN": int(mpl),
            "NPUW_LLM_MAX_PROMPT_LEN": int(mpl),
            "NPUW_LLM_PREFILL_CHUNK_SIZE": int(prefill_chunk),
        }
        print(f"[npu_load] LLMPipeline(..., 'NPU', {props}) …", flush=True)
        t0 = time.perf_counter()
        try:
            pipe = _make_llm_pipeline(ov_genai, "NPU", props)
            load_s = time.perf_counter() - t0
            attempts.append(
                {
                    "max_prompt_len": mpl,
                    "properties": props,
                    "load_ok": True,
                    "load_s": load_s,
                    "error": None,
                }
            )
            max_ok = mpl
            del pipe
            gc.collect()
        except Exception as exc:
            load_s = time.perf_counter() - t0
            attempts.append(
                {
                    "max_prompt_len": mpl,
                    "properties": props,
                    "load_ok": False,
                    "load_s": load_s,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            # Stop climbing once a length fails after a success, or keep going if all fail.
            if max_ok is not None:
                break

    report = {
        "probe": "npu_feasibility",
        "mode": "run_npu_load",
        "status": "complete",
        "isolation_mode": "OPERATOR_ASSERTED_CLEAN",
        "stack": {
            "openvino": ov.__version__,
            "openvino_genai": getattr(ov_genai, "__version__", "unknown"),
            "available_devices": devices,
        },
        "model": _model_report_fields(
            load_sequence=["NPU"],
            generate_device="NPU",
        ),
        "static_shape_properties": {
            "MAX_PROMPT_LEN": "GenAI config option (error strings reference this name)",
            "NPUW_LLM_MAX_PROMPT_LEN": "NPUW / intel_npu plugin property",
            "NPUW_LLM_PREFILL_CHUNK_SIZE": "NPUW chunked-prefill (openvino#34617)",
            "platform_defaults": {
                "MAX_PROMPT_LEN": NPU_DEFAULT_MAX_PROMPT_LEN,
                "NPUW_LLM_PREFILL_CHUNK_SIZE": NPU_DEFAULT_PREFILL_CHUNK,
            },
        },
        "attempts": attempts,
        "max_prompt_len_loaded_ok": max_ok,
        "load_success_any": any(a["load_ok"] for a in attempts),
    }
    path = out_dir / "npu_load_probe_report.json"
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "acquire_tokenize",
            "run_gpu",
            "multi_turn_gold_selftest",
            "run_gpu_multi_turn",
            "run_gpu_multi_turn_xlam",
            "run_cloud_multi_turn",
            "run_session_residency",
            "compare_session_residency",
            "session_residency_cold_control",
            "session_residency_render_smoke",
            "attention_window_inventory",
            "kv_precision_property_smoke",
            "run_gpu_kv_precision",
            "run_npu_load",
        ),
        default="acquire_tokenize",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--model",
        default=CLOUD_DEFAULT_MODEL,
        help="Anthropic model id for run_cloud_multi_turn (default claude-sonnet-5)",
    )
    parser.add_argument(
        "--model-spec",
        default=None,
        help=(
            "FetchedModelSpec YAML. Default: openvino.model_spec in configs/delta_n.yaml "
            "(Qwen3-4B-int4-ov). Pass configs/models/Qwen3-8B-int4-ov.yaml to select 8B. "
            "Does not change the yaml default. Applies to every local-IR mode."
        ),
    )
    parser.add_argument(
        "--print-model-identity",
        action="store_true",
        help=(
            "resolve --model-spec (or the delta_n.yaml default) and print the report "
            "identity block; no GPU load"
        ),
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help=(
            "run_cloud_multi_turn only: score gold decoded through the new "
            "report/scoring path offline (no API)"
        ),
    )
    parser.add_argument(
        "--arm",
        default="gpu_only",
        help="delta_n.yaml arm id for run_session_residency (default gpu_only)",
    )
    parser.add_argument(
        "--residency-mode",
        choices=RESIDENCY_MODES,
        default="RESIDENT",
        help="RESIDENT or NON_RESIDENT for run_session_residency",
    )
    parser.add_argument(
        "--n-entries",
        type=int,
        default=20,
        help="multi_turn_base entries to run (paired prefix; default 20 for all cells)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SESSION_RESIDENCY_SEED,
        help="recorded pairing seed (entry order is the fixed 20-prefix; no shuffle)",
    )
    parser.add_argument(
        "--resident-report",
        type=Path,
        default=None,
        help="RESIDENT report JSON for compare_session_residency",
    )
    parser.add_argument(
        "--non-resident-report",
        type=Path,
        default=None,
        help="NON_RESIDENT report JSON for compare_session_residency",
    )
    args = parser.parse_args()
    if args.print_model_identity:
        try:
            apply_model_spec(args.model_spec)
        except (ValueError, TypeError, OSError) as exc:
            print(f"FATAL: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(model_identity_block(), indent=2, default=str))
        return 0
    if not BFCL_UNPACKED.is_dir():
        print("FATAL: bfcl_eval unpacked wheel missing at", BFCL_UNPACKED, file=sys.stderr)
        return 2
    if args.model_spec is not None or args.mode in LOCAL_MODEL_MODES:
        try:
            apply_model_spec(args.model_spec)
        except (ValueError, TypeError, OSError) as exc:
            print(f"FATAL: {exc}", file=sys.stderr)
            return 2
    if args.mode == "acquire_tokenize":
        report = run_acquire_tokenize(args.out)
        print(json.dumps({"ok": True, "mode": report["mode"], "out": str(args.out)}, indent=2))
        print("token_profile_summary:", json.dumps(report["token_profile_summary"], indent=2))
        print("gold_selftest:", json.dumps(report["ast_checker_gold_selftest"], indent=2))
        print(
            "multi_turn_gold:",
            json.dumps(report["multi_turn_checker_gold_selftest"], indent=2),
        )
        print("opus_cost:", json.dumps(report["opus_cost_estimate"], indent=2))
        return 0
    if args.mode == "attention_window_inventory":
        report = run_attention_window_inventory(args.out)
        print(
            json.dumps(
                {
                    "ok": True,
                    "mode": report["mode"],
                    "verdict": report["verdict"],
                },
                indent=2,
            )
        )
        return 0
    if args.mode == "kv_precision_property_smoke":
        report = run_kv_precision_property_smoke(args.out)
        print(
            json.dumps(
                {
                    "ok": True,
                    "mode": report["mode"],
                    "property": report["property"],
                    "levels": [
                        {
                            "requested": r["requested"],
                            "took_effect": r["took_effect"],
                            "set_ok": r["set_ok"],
                        }
                        for r in report["levels"]
                    ],
                    "out": report.get("artifact"),
                },
                indent=2,
            )
        )
        return 0
    if args.mode == "run_gpu_kv_precision":
        report = run_gpu_kv_precision(args.out, max_new_tokens=args.max_new_tokens)
        print(json.dumps({"ok": True, "mode": report["mode"], "out": str(args.out)}, indent=2))
        print("accuracy_summary:", json.dumps(report["accuracy_summary"], indent=2))
        print("effect_summary:", json.dumps(report["effect_summary"], indent=2, default=str))
        return 0
    if args.mode == "run_npu_load":
        report = run_npu_load(args.out)
        print(json.dumps({"ok": True, "mode": report["mode"], "out": str(args.out)}, indent=2))
        print(
            "load_success_any:",
            report["load_success_any"],
            "max_prompt_len_loaded_ok:",
            report["max_prompt_len_loaded_ok"],
        )
        for a in report["attempts"]:
            if a["load_ok"]:
                print(f"  OK mpl={a['max_prompt_len']} load_s={a['load_s']:.1f}")
            else:
                print(f"  FAIL mpl={a['max_prompt_len']}: {a['error']}")
        return 0 if report["load_success_any"] else 4
    if args.mode == "multi_turn_gold_selftest":
        out_dir = args.out
        out_dir.mkdir(parents=True, exist_ok=True)
        entries = select_multi_turn_entries()
        gold = run_multi_turn_gold_selftest(entries)
        path = out_dir / "multi_turn_gold_selftest.json"
        path.write_text(
            json.dumps(gold, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "ok": True,
                    "mode": "multi_turn_gold_selftest",
                    "n_valid": gold["n_valid"],
                    "n": gold["n"],
                    "out": str(path),
                },
                indent=2,
            )
        )
        return 0 if gold["n_valid"] == gold["n"] and gold["n"] > 0 else 3
    if args.mode == "run_gpu_multi_turn":
        report = run_gpu_multi_turn(args.out, max_new_tokens=args.max_new_tokens)
        print(json.dumps({"ok": True, "mode": report["mode"], "out": str(args.out)}, indent=2))
        print(
            "multi_turn_gold:",
            json.dumps(report["multi_turn_checker_gold_selftest"], indent=2),
        )
        print(
            "accuracy:",
            json.dumps(report["gpu_probe"]["accuracy_multi_turn"], indent=2),
        )
        print(
            "wall_clock_s:",
            json.dumps(report["gpu_probe"]["wall_clock_s"], indent=2),
        )
        return 0
    if args.mode == "run_gpu_multi_turn_xlam":
        out_dir = args.out
        if out_dir.resolve() == DEFAULT_OUT.resolve():
            out_dir = DEFAULT_OUT_XLAM
        # Refuse collision with sealed tools-style report trees.
        blocked = {
            (ROOT / "derived" / "bfcl_feasibility").resolve(),
            (ROOT / "derived" / "bfcl_feasibility_8b").resolve(),
        }
        if out_dir.resolve() in blocked:
            raise SystemExit(
                f"FATAL: --out {out_dir} collides with a sealed tools-style report dir. "
                f"Use a distinct path (default for this mode: {DEFAULT_OUT_XLAM})."
            )
        report = run_gpu_multi_turn_xlam(out_dir, max_new_tokens=args.max_new_tokens)
        print(
            json.dumps(
                {
                    "ok": True,
                    "mode": report["mode"],
                    "out": str(out_dir),
                    "prompt_format": report["gpu_probe"]["prompt_format"],
                },
                indent=2,
            )
        )
        print(
            "xlam_gold_decode:",
            json.dumps(report["xlam_gold_decode_selftest"], indent=2),
        )
        print(
            "accuracy:",
            json.dumps(report["gpu_probe"]["accuracy_multi_turn"], indent=2),
        )
        return 0
    if args.mode == "run_cloud_multi_turn":
        if args.selftest:
            report = run_cloud_multi_turn_scoring_selftest(args.out)
            gold = report.get("gold") or {}
            n_valid = int(gold.get("n_valid") or 0)
            n = int(gold.get("n") or 0)
            print(
                json.dumps(
                    {
                        "ok": n_valid == n and n > 0,
                        "mode": report["mode"],
                        "selftest": True,
                        "api_called": False,
                        "n_valid": n_valid,
                        "n": n,
                        "entry_ids": report.get("entry_ids"),
                        "entry_ids_match_pinned_file": report.get("entry_ids_match_pinned_file"),
                        "gold_turns_correct": gold.get("turns_correct"),
                        "gold_turns_total": gold.get("turns_total"),
                        "local_arm_trajectory_stored": (
                            (report.get("local_arm") or {}).get("trajectory_stored")
                        ),
                        "local_arm_turns_correct": (
                            (report.get("local_arm") or {}).get("turns_correct")
                        ),
                        "local_arm_turns_total": (
                            (report.get("local_arm") or {}).get("turns_total")
                        ),
                        "local_arm_first_failure_turn_distribution": (
                            (report.get("local_arm") or {}).get("first_failure_turn_distribution")
                        ),
                        "artifact": report.get("artifact"),
                    },
                    indent=2,
                    default=str,
                )
            )
            return 0 if n_valid == n and n > 0 else 3
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print(
                "FATAL: ANTHROPIC_API_KEY is not set. Refusing to start the paid "
                "cloud multi-turn run. Export ANTHROPIC_API_KEY in the environment "
                "(never pass it on the command line).",
                file=sys.stderr,
            )
            return 2
        report = run_cloud_multi_turn(
            args.out,
            model=args.model,
            max_tokens=args.max_new_tokens,
        )
        print(
            json.dumps(
                {
                    "ok": True,
                    "mode": report["mode"],
                    "entry_ids": report.get("entry_ids"),
                    "cloud_trajectory": (report.get("cloud_arm") or {}).get("trajectory"),
                    "spend_usd": (report.get("spend") or {}).get("usd"),
                    "artifact": report.get("artifact"),
                    "markdown": report.get("markdown"),
                },
                indent=2,
                default=str,
            )
        )
        return 0
    if args.mode == "session_residency_render_smoke":
        report = run_session_residency_render_smoke(args.out)
        print(
            json.dumps(
                {
                    "ok": bool(report.get("all_identical")),
                    "mode": report["mode"],
                    "n_entries": report["n_entries"],
                    "all_identical": report["all_identical"],
                    "artifact": report.get("artifact"),
                },
                indent=2,
            )
        )
        return 0 if report.get("all_identical") else 3
    if args.mode == "session_residency_cold_control":
        report = run_session_residency_cold_control(
            args.out,
            arm_id=args.arm,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
        )
        print(
            json.dumps(
                {
                    "ok": bool(report.get("passed")),
                    "mode": report["mode"],
                    "passed": report["passed"],
                    "failures": report.get("failures"),
                    "checks": report.get("checks"),
                    "cold_fix": report.get("cold_fix"),
                    "artifact": report.get("artifact"),
                },
                indent=2,
                default=str,
            )
        )
        return 0 if report.get("passed") else 3
    if args.mode == "run_session_residency":
        report = run_session_residency(
            args.out,
            arm_id=args.arm,
            residency_mode=args.residency_mode,
            n_entries=args.n_entries,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
        )
        print(
            json.dumps(
                {
                    "ok": True,
                    "mode": report["mode"],
                    "arm_id": report["arm_id"],
                    "residency_mode": report["residency_mode"],
                    "n_entries_run": report["n_entries_run"],
                    "n_entries_reduced_reason": report["n_entries_reduced_reason"],
                    "fraction_turns_slo_ok": report["gpu_probe"]["fraction_turns_slo_ok"],
                    "accuracy_per_turn_f2": report["gpu_probe"]["accuracy_per_turn_f2"],
                    "session_total_latency_s": report["gpu_probe"]["session_total_latency_s"],
                    "artifact": report.get("artifact"),
                },
                indent=2,
                default=str,
            )
        )
        return 0
    if args.mode == "compare_session_residency":
        if not args.resident_report or not args.non_resident_report:
            print(
                "FATAL: compare_session_residency requires --resident-report and "
                "--non-resident-report",
                file=sys.stderr,
            )
            return 2
        resident = json.loads(args.resident_report.read_text(encoding="utf-8"))
        non_resident = json.loads(args.non_resident_report.read_text(encoding="utf-8"))
        cmp = compare_session_residency_cells(resident, non_resident)
        out_path = args.out / f"session_residency_compare_{cmp.get('arm_id')}.json"
        args.out.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(cmp, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"ok": True, "artifact": str(out_path), **cmp}, indent=2, default=str))
        return 0
    report = run_gpu(args.out, max_new_tokens=args.max_new_tokens)
    print(json.dumps({"ok": True, "mode": report["mode"], "out": str(args.out)}, indent=2))
    print(
        "accuracy:",
        json.dumps(report["gpu_probe"]["accuracy_ast_single_turn"], indent=2),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
