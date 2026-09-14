"""OpenVINO ``KV_CACHE_PRECISION`` materialization and readback enforcement.

YAML / JSON specs keep string names (``f16`` / ``u8`` / ``u4``). Pipeline load requires
``openvino.Type``. Effect is confirmed only by reading the property back from the **same**
``ov.Core`` used for ``set_property`` / load - a fresh Core does not observe another Core's
sticky device property (see ``tools/bfcl_feasibility_probe.py``).

A requested/readback mismatch is a failed cell, not a measurement.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "KV_ELEMENTS_PER_TOKEN",
    "KV_PRECISION_NAMES",
    "enforce_kv_cache_precision",
    "kv_bytes_per_token",
    "materialize_pipeline_properties",
    "normalize_kv_precision_name",
    "ov_type_for_kv_precision",
    "read_kv_cache_precision",
    "requested_kv_cache_precision",
]

KV_PRECISION_NAMES: tuple[str, ...] = ("f16", "u8", "u4")

# Qwen3-4B: 2 * n_layers * n_kv_heads * head_dim (elements, not bytes).
KV_ELEMENTS_PER_TOKEN = 2 * 36 * 8 * 128  # 73728 elements/token
_KV_BYTES_PER_ELEMENT: dict[str, float] = {"f16": 2.0, "u8": 1.0, "u4": 0.5}

_ALIASES: dict[str, str] = {
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


def kv_bytes_per_token(precision: Any) -> float | None:
    """Live-KV bytes/token for a pinned precision name, else None.

    Returns None when precision is unpinned (``dynamic``, unset, or unrecognised)
    so callers record the gap rather than assume an unmeasured value.
    Pass the **readback** normalized name (same source as manifest
    ``kv_cache_precision_readback``), not the requested property alone.
    """
    if precision is None:
        return None
    name = normalize_kv_precision_name(precision)
    if not name:
        return None
    bytes_per = _KV_BYTES_PER_ELEMENT.get(name)
    if bytes_per is None:
        return None
    return float(KV_ELEMENTS_PER_TOKEN) * bytes_per


def normalize_kv_precision_name(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "to_string") and callable(value.to_string):
        raw = str(value.to_string()).strip().lower()
    else:
        raw = str(value).strip().lower()
        if "'" in raw:
            raw = raw.split("'")[1].strip().lower()
    return _ALIASES.get(raw, raw)


def ov_type_for_kv_precision(name: str) -> Any:
    from openvino import Type

    key = normalize_kv_precision_name(name)
    mapping = {
        "f16": Type.f16,
        "u8": Type.u8,
        "u4": Type.u4,
    }
    if key not in mapping:
        raise ValueError(
            f"unsupported KV_CACHE_PRECISION {name!r}; expected one of {KV_PRECISION_NAMES}"
        )
    return mapping[key]


def requested_kv_cache_precision(properties: dict[str, Any] | None) -> str | None:
    if not properties:
        return None
    if "KV_CACHE_PRECISION" not in properties:
        return None
    name = normalize_kv_precision_name(properties["KV_CACHE_PRECISION"])
    return name or None


def materialize_pipeline_properties(properties: dict[str, Any] | None) -> dict[str, Any]:
    """Copy properties, converting string ``KV_CACHE_PRECISION`` to ``openvino.Type``."""
    out = dict(properties or {})
    if "KV_CACHE_PRECISION" not in out:
        return out
    raw = out["KV_CACHE_PRECISION"]
    # Already an OV Type (has to_string and is not a plain str).
    if not isinstance(raw, str) and hasattr(raw, "to_string"):
        return out
    out["KV_CACHE_PRECISION"] = ov_type_for_kv_precision(str(raw))
    return out


def read_kv_cache_precision(device: str, *, core: Any | None = None) -> dict[str, Any]:
    import openvino as ov

    core = core if core is not None else ov.Core()
    try:
        value = core.get_property(device, "KV_CACHE_PRECISION")
        return {
            "property": "KV_CACHE_PRECISION",
            "device": device,
            "raw": str(value),
            "to_string": (str(value.to_string()) if hasattr(value, "to_string") else None),
            "normalized": normalize_kv_precision_name(value),
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


def enforce_kv_cache_precision(
    *,
    device: str,
    requested: str | None,
    core: Any,
) -> dict[str, Any]:
    """Read back ``KV_CACHE_PRECISION`` and compare to the requested name when set.

    When ``requested`` is None (arm has no KV override), readback is still recorded and
    ``match`` is True - there is nothing to enforce.
    """
    readback = read_kv_cache_precision(device, core=core)
    if requested is None:
        return {
            "requested": None,
            "readback": readback,
            "match": True,
            "enforced": False,
            "failure_mode": None,
        }
    req = normalize_kv_precision_name(requested)
    got = readback.get("normalized")
    if not readback.get("ok"):
        return {
            "requested": req,
            "readback": readback,
            "match": False,
            "enforced": True,
            "failure_mode": "kv_cache_precision_readback_unavailable",
        }
    if got != req:
        return {
            "requested": req,
            "readback": readback,
            "match": False,
            "enforced": True,
            "failure_mode": f"kv_cache_precision_mismatch:requested={req}:got={got}",
        }
    return {
        "requested": req,
        "readback": readback,
        "match": True,
        "enforced": True,
        "failure_mode": None,
    }
