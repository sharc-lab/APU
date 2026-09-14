"""SEAM project-state MCP server. Read-only, stdlib-only protocol.

Why this exists
---------------
SEAM is worked by several agents in parallel. Each otherwise re-derives project
state by reading a dozen files, and they can disagree — which already happened
once, when a stale governing-document hash was cited as authoritative.

This server is a single read-only source of truth. It never writes, never
mutates, and never touches credentials.

Design principle: report DECLARED state and INFERRED state separately, and flag
disagreement rather than reconciling it. Declared state is what
``configs/project_state.yaml`` says. Inferred state is what artifacts on disk
actually show. Divergence means either the declared file is stale or an artifact
is not what it should be, and which one matters. That is a human call.

Dependencies
------------
The MCP protocol is implemented directly over stdio JSON-RPC using only the
standard library, deliberately. Adding the ``mcp`` SDK would put an unused-in-
production pin in seam/requirements.txt, and per that file's own policy "an
unused pin is a claim that something was tested". It also removes a class of
venv/interpreter mismatch failures.

PyYAML is used if importable (it is already a declared SEAM dependency) and
degrades gracefully if not.

STDOUT CARRIES PROTOCOL ONLY. All diagnostics go to stderr.

Run:
    python tools/seam_mcp/project_state.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from seam.errors import RawStoreError
from seam.manifest import load_run_manifest

RAW = REPO_ROOT / "raw"
PIN_FILE = REPO_ROOT / "GOVERNING_DOCS.sha256"
STATE_FILE = REPO_ROOT / "configs" / "project_state.yaml"
PLATFORM_CFG = REPO_ROOT / "configs" / "platforms" / "aipc-c1.yaml"
REPO_CFG = REPO_ROOT / "configs" / "repo.yaml"

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "seam-project-state", "version": "1.0.0"}


# ------------------------------------------------------------ tool registry

TOOLS: dict[str, dict[str, Any]] = {}


def tool(description: str, schema: dict[str, Any] | None = None) -> Callable[..., Any]:
    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        TOOLS[fn.__name__] = {
            "name": fn.__name__,
            "description": description,
            "inputSchema": schema or {"type": "object", "properties": {}},
            "fn": fn,
        }
        return fn

    return decorate


# ------------------------------------------------------------------ helpers


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None or not path.is_file():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_corrected_manifest(run_id: str) -> dict[str, Any] | None:
    """Load a run manifest with provenance corrections applied (AM-036).

    Sealed ``raw/`` bytes are write-once; ``derived/manifest_corrections/`` supplies the
    authoritative operator view via :func:`seam.manifest.load_run_manifest`.
    """
    try:
        return load_run_manifest(run_id, repo_root=REPO_ROOT)
    except RawStoreError:
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class Run:
    run_id: str
    manifest: dict[str, Any]
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return str(self.manifest.get("workload", {}).get("kind", "unknown"))

    @property
    def target(self) -> str | None:
        return self.manifest.get("target")

    def brief(self) -> dict[str, Any]:
        power = self.manifest.get("power_state", {}) or {}
        return {
            "run_id": self.run_id,
            "kind": self.kind,
            "target": self.target,
            "verdict": self.summary.get("verdict"),
            "timestamp_utc": self.manifest.get("timestamp_utc"),
            "git_sha": self.manifest.get("git_sha"),
            "git_dirty": self.manifest.get("git_dirty"),
            "elevated": self.manifest.get("elevated"),
            "on_battery": power.get("on_battery"),
            "charging": power.get("charging"),
            "refusal_reasons": self.summary.get("refusal_reasons"),
            "self_check": (self.manifest.get("integrity", {}) or {}).get("self_check"),
        }


def _all_runs() -> list[Run]:
    runs: list[Run] = []
    if not RAW.is_dir():
        return runs
    for entry in sorted(RAW.iterdir()):
        if not entry.is_dir() or entry.name.startswith("_"):
            continue
        manifest = _load_corrected_manifest(entry.name)
        if manifest is None:
            continue
        runs.append(Run(entry.name, manifest, _load_json(entry / "summary.json") or {}))
    return runs


# -------------------------------------------------------------------- tools


@tool(
    "Current SEAM project state: declared (configs/project_state.yaml) vs inferred "
    "(artifacts on disk), with disagreements listed. Disagreement is a finding for a "
    "human, never silently reconciled. Call this first in any new session."
)
def seam_status() -> dict[str, Any]:
    declared = _load_yaml(STATE_FILE)
    platform = _load_yaml(PLATFORM_CFG)
    topology = (platform.get("topology") or {}) if platform else {}
    runs = _all_runs()

    inferred: dict[str, Any] = {
        "topology_verified": bool(topology.get("verified")),
        "topology_citing_run": (topology.get("measured") or {}).get("run_id"),
        "p_cpus": topology.get("p_cpus"),
        "lpe_cpus": topology.get("lpe_cpus"),
        "sealed_runs": len(runs),
        "run_kinds": sorted({r.kind for r in runs}),
        "telemetry_module_present": (REPO_ROOT / "seam" / "telemetry").is_dir(),
        "backends_module_present": (REPO_ROOT / "seam" / "backends").is_dir(),
        "energy_calibration_present": (REPO_ROOT / "derived" / "energy_calibration.json").is_file(),
        "noise_floor_present": (REPO_ROOT / "derived" / "noise_floor.json").is_file(),
    }
    inferred["m1_accepted"] = bool(
        inferred["topology_verified"] and inferred["topology_citing_run"]
    )
    inferred["m2_started"] = inferred["telemetry_module_present"]

    disagreements: list[str] = []
    for key, value in (declared.get("inferred_expectations") or {}).items():
        if key in inferred and inferred[key] != value:
            disagreements.append(f"{key}: declared={value!r} inferred={inferred[key]!r}")
    if not declared:
        disagreements.append(
            "configs/project_state.yaml absent or unreadable — inferred state only."
        )

    return {
        "declared": declared,
        "inferred": inferred,
        "disagreements": disagreements,
        "note": (
            "Disagreement is a finding for a human. Do not reconcile by editing "
            "either source without an amendment entry."
        ),
    }


@tool(
    "List sealed runs under raw/, optionally filtered by workload kind and/or target.",
    {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "description": "e.g. topology_verify, battery_counter_char, microbench",
            },
            "target": {
                "type": "string",
                "description": "cpu-p | cpu-lpe | igpu | npu",
            },
        },
    },
)
def seam_runs(kind: str | None = None, target: str | None = None) -> dict[str, Any]:
    runs = _all_runs()
    if kind:
        runs = [r for r in runs if r.kind == kind]
    if target:
        runs = [r for r in runs if r.target == target]
    return {
        "count": len(runs),
        "filter": {"kind": kind, "target": target},
        "runs": [r.brief() for r in runs],
    }


@tool(
    "Full manifest and summary for one run_id.",
    {
        "type": "object",
        "properties": {"run_id": {"type": "string"}},
        "required": ["run_id"],
    },
)
def seam_run(run_id: str) -> dict[str, Any]:
    entry = RAW / run_id
    if not entry.is_dir():
        return {"error": f"no sealed run {run_id!r} under raw/"}
    manifest = _load_corrected_manifest(run_id)
    if manifest is None:
        return {"error": f"no manifest for run {run_id!r} under raw/"}
    return {
        "run_id": run_id,
        "manifest": manifest,
        "summary": _load_json(entry / "summary.json"),
        "files": sorted(p.name for p in entry.iterdir() if p.is_file()),
    }


@tool(
    "Governing-document SHA-256 pin verdicts (AM-009): match, drift, or missing. "
    "Reports drift but deliberately does not judge whether it is known supersession "
    "or unexplained — that requires reading AMENDMENTS.md."
)
def seam_pins() -> dict[str, Any]:
    if not PIN_FILE.is_file():
        return {
            "status": "no_pin_file",
            "detail": f"{PIN_FILE.name} absent — pins are not being verified (AM-009).",
        }

    results: list[dict[str, Any]] = []
    for line in PIN_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        expected, rel = parts[0], parts[1].strip()
        target = REPO_ROOT / rel
        if not target.is_file():
            results.append({"path": rel, "status": "missing"})
            continue
        actual = _sha256(target)
        results.append(
            {
                "path": rel,
                "status": "match" if actual == expected else "drift",
                "expected": expected,
                "actual": actual,
                "bytes": target.stat().st_size,
            }
        )

    drifted = [r for r in results if r["status"] != "match"]
    return {
        "status": "ok" if not drifted else "attention",
        "files": results,
        "note": (
            "Drift may be KNOWN SUPERSESSION (a logged amendment moved the pin) or "
            "UNEXPLAINED (stop and investigate). Check AMENDMENTS.md. Never edit the "
            "pin list to silence a mismatch you do not understand."
        )
        if drifted
        else None,
    }


@tool("AM ledger parsed from AMENDMENTS.md: identifier, title, status, and which are open.")
def seam_amendments() -> dict[str, Any]:
    path = REPO_ROOT / "AMENDMENTS.md"
    if not path.is_file():
        return {"error": "AMENDMENTS.md not found"}

    entries: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    heading = re.compile(r"^#{2,3}\s+(AM-\d+)\s*[—-]\s*(.+?)\s*$")
    status_re = re.compile(r"\b(RESOLVED|OPEN|N/?A|SUPERSEDED|DEFERRED|WITHDRAWN)\b", re.IGNORECASE)

    for line in path.read_text(encoding="utf-8").splitlines():
        match = heading.match(line)
        if match:
            current = {"id": match.group(1), "title": match.group(2), "status": None}
            entries.append(current)
            continue
        if current and current["status"] is None:
            found = status_re.search(line)
            if found:
                current["status"] = found.group(1).upper()

    return {
        "count": len(entries),
        "amendments": entries,
        "open": [e["id"] for e in entries if e["status"] in (None, "OPEN")],
    }


@tool("raw/ payload size against the AM-014 ceiling, with per-run breakdown.")
def seam_raw_usage() -> dict[str, Any]:
    cfg = _load_yaml(REPO_CFG).get("raw_retention", {}) or {}
    ceiling_mb = cfg.get("ceiling_mb", 100)

    total = 0
    per_run: dict[str, int] = {}
    if RAW.is_dir():
        for entry in RAW.iterdir():
            if entry.name.startswith("_") or not entry.is_dir():
                continue
            size = sum(p.stat().st_size for p in entry.rglob("*") if p.is_file())
            per_run[entry.name] = size
            total += size

    ceiling_bytes = int(float(ceiling_mb) * 1_000_000)
    return {
        "total_bytes": total,
        "total_mb": round(total / 1_000_000, 3),
        "ceiling_mb": ceiling_mb,
        "headroom_mb": round((ceiling_bytes - total) / 1_000_000, 3),
        "over_ceiling": total > ceiling_bytes,
        "excluded_globs": sorted(cfg.get("exclude_globs", [])),
        "per_run_bytes": per_run,
    }


@tool(
    "Platform A configuration, plus the explicit list of peak/derived figures that "
    "must never be cited as measurements."
)
def seam_platform() -> dict[str, Any]:
    return {
        "config": _load_yaml(PLATFORM_CFG),
        "do_not_cite_as_measured": [
            "50 TOPS — NPU peak INT8, not achieved throughput",
            "~120 GB/s — derived from LPDDR5X-7467 on a 128-bit bus, not measured",
            "180 TOPS — Platform B CPU+GPU+NPU aggregate, not an NPU figure",
            "1.38x cluster_separation_ratio — interpreter-bound clustering "
            "discriminant, NOT a P vs LP-E performance ratio",
        ],
    }


# ------------------------------------------------------------ JSON-RPC loop


def _result(request_id: Any, payload: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": payload}


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def handle(message: dict[str, Any]) -> dict[str, Any] | None:
    method = message.get("method")
    request_id = message.get("id")
    params = message.get("params") or {}

    # Notifications carry no id and expect no response.
    if request_id is None:
        return None

    if method == "initialize":
        return _result(
            request_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": SERVER_INFO,
            },
        )

    if method == "ping":
        return _result(request_id, {})

    if method == "tools/list":
        return _result(
            request_id,
            {
                "tools": [
                    {
                        "name": spec["name"],
                        "description": spec["description"],
                        "inputSchema": spec["inputSchema"],
                    }
                    for spec in TOOLS.values()
                ]
            },
        )

    if method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments") or {}
        spec = TOOLS.get(name)
        if spec is None:
            return _error(request_id, -32602, f"unknown tool: {name}")
        try:
            payload = spec["fn"](**arguments)
            text = json.dumps(payload, indent=2, default=str)
            return _result(request_id, {"content": [{"type": "text", "text": text}]})
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            return _result(
                request_id,
                {
                    "content": [{"type": "text", "text": f"tool error: {exc}"}],
                    "isError": True,
                },
            )

    return _error(request_id, -32601, f"method not found: {method}")


def main() -> int:
    sys.stderr.write(f"seam-project-state: serving from {REPO_ROOT}\n")
    sys.stderr.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue

        response = handle(message)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
