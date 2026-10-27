"""Device profile (schema apu-device-profile/1): what machine, runtime, model and windows a run or audit is for.

{
  "schema": "apu-device-profile/1",
  "profile_id": "short-stable-id",
  "hardware": {"hostname", "cpu", "gpu", "memory_gb", "os", "driver"},       null = not recorded (reported as a gap)
  "runtime":  {"name": "ollama", "version", "server_cmd": [...], "server_env": {KEY: VALUE}},
  "model":    {"name": "qwen3:14b", "quant", "digest"},
  "windows":  [{"tier": "ollama_default" | "ollama_ctx_<N>", "window": N}, ...],   the windows to test
  "cloud_model": "claude-haiku-5-5",                                              priced by src/cloud/prices.py
  "sources":  {field: "where the value came from"}                                optional, free text
}
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from apu_bench import PROFILE_SCHEMA

HARDWARE_TAGS = ("hostname", "cpu", "gpu", "memory_gb", "os", "driver")
_TIER = re.compile(r"^ollama_(default|ctx_\d+(_[a-z0-9]+)?)$")


class ProfileError(ValueError):
    pass


def validate(p: dict) -> list[str]:
    """Raises ProfileError on a structural problem; returns the list of missing (null) tags, which a report shows."""
    if p.get("schema") != PROFILE_SCHEMA:
        raise ProfileError(f"schema must be {PROFILE_SCHEMA!r}, got {p.get('schema')!r}")
    for k in ("profile_id", "hardware", "runtime", "model", "windows"):
        if k not in p:
            raise ProfileError(f"missing {k}")
    if not p["runtime"].get("name"):
        raise ProfileError("runtime.name is required")
    if not p["model"].get("name"):
        raise ProfileError("model.name is required")
    if not p["windows"]:
        raise ProfileError("windows is empty")
    seen = set()
    for w in p["windows"]:
        t = w.get("tier")
        if not t or not _TIER.match(t):
            raise ProfileError(f"bad tier {t!r} (ollama_default or ollama_ctx_<N>[_suffix])")
        if t in seen:
            raise ProfileError(f"duplicate tier {t}")
        seen.add(t)
        if not isinstance(w.get("window"), int) or w["window"] <= 0:
            raise ProfileError(f"{t}: window must be a positive int")
        if t.startswith("ollama_ctx_") and int(t.split("_")[2]) != w["window"]:
            raise ProfileError(f"{t}: window {w['window']} does not match the tier")
    missing = [f"hardware.{k}" for k in HARDWARE_TAGS if p["hardware"].get(k) in (None, "")]
    missing += [f"runtime.{k}" for k in ("version",) if not p["runtime"].get(k)]
    missing += [f"model.{k}" for k in ("quant", "digest") if not p["model"].get(k)]
    return missing


def load(path) -> tuple[dict, str, list[str]]:
    """(profile, sha256 of the file bytes, missing tags)."""
    raw = Path(path).read_bytes()
    p = json.loads(raw.decode("utf-8"))
    return p, hashlib.sha256(raw).hexdigest(), validate(p)


def window_of(p: dict, tier: str) -> int:
    return next(w["window"] for w in p["windows"] if w["tier"] == tier)
