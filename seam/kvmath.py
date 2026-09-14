"""KV-cache geometry read from the model's own config, not assumed.

E-FILTER's primary endpoint is **peak resident KV bytes**, so the bytes-per-token constant is
load-bearing. It is derived analytically:

.. code-block:: text

    kv_bytes_per_token = 2 * n_layers * n_kv_heads * head_dim * dtype_bytes
                         ^-- K and V

Three hazards this module exists to close.

**``head_dim`` is read explicitly, never derived.** Qwen3 sets ``head_dim=128`` with
``hidden_size=2560`` and ``num_attention_heads=32``, so ``hidden_size / n_heads`` gives 80 and a
KV constant 1.6x too small. The same trap is recorded in ``censor/kv_math.py``.

**The KV dtype is a runtime property, not a model property.** The weights are INT4; the cache is
not. OpenVINO's CPU plugin chooses the cache precision, so :func:`probe_kv_cache_precision` reads
it back from the device rather than assuming it, and the assumed value is recorded as an explicit
fallback when the readback is unavailable.

**Analytic is not observed.** This module produces an analytic figure. Log observed process RSS
next to it so the two are comparable rather than conflated - the analytic value excludes weights,
activations, allocator slack, and any runtime-internal copies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from seam.errors import ConfigError
from seam.hashing import sha256_file
from seam.jsonlog import log_event

__all__ = [
    "BYTES_PER_ELEMENT",
    "KV_FORMULA",
    "KvGeometry",
    "load_kv_geometry",
    "probe_kv_cache_precision",
]

#: Bytes per KV element by precision name, keyed on the names OpenVINO and HF configs use.
BYTES_PER_ELEMENT: Final[dict[str, int]] = {
    "f32": 4,
    "float32": 4,
    "f16": 2,
    "float16": 2,
    "bf16": 2,
    "bfloat16": 2,
    "u8": 1,
    "uint8": 1,
    "i8": 1,
    "int8": 1,
}

#: Recorded in the manifest next to the number, so the number is never a bare constant.
KV_FORMULA: Final = "2 * n_layers * n_kv_heads * head_dim * dtype_bytes"


@dataclass(frozen=True, slots=True)
class KvGeometry:
    """KV-cache geometry for one model, with the provenance of every term."""

    model_name: str
    n_layers: int
    n_kv_heads: int
    head_dim: int
    n_attention_heads: int
    kv_dtype: str
    kv_dtype_bytes: int
    #: How ``kv_dtype`` was established: ``device_readback`` or ``assumed:<reason>``.
    kv_dtype_source: str
    config_path: str
    config_sha256: str

    @property
    def bytes_per_token(self) -> int:
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * self.kv_dtype_bytes

    def bytes_for(self, context_tokens: int) -> int:
        return self.bytes_per_token * int(context_tokens)

    def to_record(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "n_layers": self.n_layers,
            "n_kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "n_attention_heads": self.n_attention_heads,
            "kv_dtype": self.kv_dtype,
            "kv_dtype_bytes": self.kv_dtype_bytes,
            "kv_dtype_source": self.kv_dtype_source,
            "kv_bytes_per_token": self.bytes_per_token,
            "formula": KV_FORMULA,
            "config_path": self.config_path,
            "config_sha256": self.config_sha256,
            "excludes": ["weights", "activations", "allocator_slack", "runtime_internal_copies"],
        }


def probe_kv_cache_precision(device: str = "CPU") -> tuple[str | None, str]:
    """Read the device's KV cache precision back from OpenVINO.

    Returns:
        ``(precision_name, source)``. ``precision_name`` is None when the property is unavailable,
        in which case ``source`` records why - a missing readback is reported, never replaced with a
        plausible-looking default here.
    """
    try:
        import openvino as ov

        value = ov.Core().get_property(device, "KV_CACHE_PRECISION")
    except Exception as exc:  # reported as a source string; never silently swallowed
        log_event(
            "kvmath.kv_cache_precision_unavailable",
            severity="warning",
            message=f"could not read KV_CACHE_PRECISION from {device}: {type(exc).__name__}",
            device=device,
            error=str(exc),
        )
        return None, f"unavailable:{type(exc).__name__}"

    name = str(getattr(value, "to_string", lambda: value)()).strip().lower()
    log_event(
        "kvmath.kv_cache_precision_read",
        message=f"{device} KV_CACHE_PRECISION={name}",
        device=device,
        precision=name,
    )
    return name, "device_readback"


def load_kv_geometry(
    model_dir: Path,
    *,
    device: str = "CPU",
    assumed_kv_dtype: str = "f16",
) -> KvGeometry:
    """Load KV geometry from ``model_dir/config.json`` and the device's cache precision.

    Args:
        model_dir: Directory holding the OpenVINO IR and the HF ``config.json``.
        device: OpenVINO device whose KV cache precision is read back.
        assumed_kv_dtype: Fallback precision used **and recorded as assumed** when the readback is
            unavailable.

    Raises:
        ConfigError: If the config is missing, or omits a term the formula needs. A guessed
            ``head_dim`` would silently scale the study's primary endpoint.
    """
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise ConfigError(
            f"no config.json in {model_dir}: KV geometry cannot be derived and must not be guessed"
        )
    config: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))

    missing = [
        key
        for key in ("num_hidden_layers", "num_key_value_heads", "head_dim", "num_attention_heads")
        if config.get(key) is None
    ]
    if missing:
        raise ConfigError(
            f"{config_path} omits {missing}. head_dim in particular is read explicitly and never "
            f"derived from hidden_size / num_attention_heads: for Qwen3 that division gives 80 "
            f"instead of 128 and understates KV bytes by 1.6x."
        )

    probed, source = probe_kv_cache_precision(device)
    dtype = probed if probed in BYTES_PER_ELEMENT else assumed_kv_dtype
    if probed is not None and probed not in BYTES_PER_ELEMENT:
        source = f"assumed:{assumed_kv_dtype}_after_unrecognised_readback_{probed}"
    elif probed is None:
        source = f"assumed:{assumed_kv_dtype}_{source}"
    if dtype not in BYTES_PER_ELEMENT:
        raise ConfigError(f"unknown KV dtype {dtype!r}; known: {sorted(BYTES_PER_ELEMENT)}")

    geometry = KvGeometry(
        model_name=str(config.get("model_type") or model_dir.name),
        n_layers=int(config["num_hidden_layers"]),
        n_kv_heads=int(config["num_key_value_heads"]),
        head_dim=int(config["head_dim"]),
        n_attention_heads=int(config["num_attention_heads"]),
        kv_dtype=dtype,
        kv_dtype_bytes=BYTES_PER_ELEMENT[dtype],
        kv_dtype_source=source,
        config_path=config_path.as_posix(),
        config_sha256=sha256_file(config_path),
    )
    log_event(
        "kvmath.geometry_loaded",
        message=(
            f"{geometry.model_name}: {geometry.bytes_per_token} KV bytes/token "
            f"({geometry.n_layers}L x {geometry.n_kv_heads}KV x {geometry.head_dim}D x 2 x "
            f"{geometry.kv_dtype_bytes}B, dtype {geometry.kv_dtype} via {geometry.kv_dtype_source})"
        ),
        **geometry.to_record(),
    )
    return geometry
