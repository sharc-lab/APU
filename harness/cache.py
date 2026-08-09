"""Content-addressed record-replay cache for raw Ollama calls.

Keyed on SHA-256 of (model, prompt, params). Stores raw output + telemetry so
every repeated (probe, depth, rep) cell is free and deterministic.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_CACHE_DIR = Path(__file__).parent.parent / ".cache" / "calls"


def _key(model: str, prompt: str, params: dict[str, Any]) -> str:
    payload = json.dumps(
        {"model": model, "prompt": prompt, "params": params},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def get(model: str, prompt: str, params: dict[str, Any]) -> dict[str, Any] | None:
    path = _CACHE_DIR / (_key(model, prompt, params) + ".json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def put(model: str, prompt: str, params: dict[str, Any], record: dict[str, Any]) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / (_key(model, prompt, params) + ".json")
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
