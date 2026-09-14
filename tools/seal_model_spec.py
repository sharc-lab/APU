"""Shared sealer model-spec resolution: CLI vs what the session recorded."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _norm(path: Path, root: Path) -> Path:
    p = path if path.is_absolute() else (root / path)
    return p.resolve()


def _unwrap_ps_list(obj: Any) -> list[Any]:
    """PowerShell ConvertTo-Json often wraps List[object] as {value, Count}."""
    if isinstance(obj, dict) and isinstance(obj.get("value"), list):
        return list(obj["value"])
    if isinstance(obj, list):
        return obj
    return []


def _ir_sha256_from_spec(spec_path: Path) -> str:
    for line in spec_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("ir_sha256:"):
            return s.split(":", 1)[1].strip().strip('"').strip("'").lower()
    raise SystemExit(f"FATAL: ir_sha256 missing from model spec {spec_path}")


def recorded_model_specs_from_plan(plan: dict[str, Any] | None, root: Path) -> list[Path]:
    """Return unique absolute model-spec paths recorded on the session plan."""
    if not isinstance(plan, dict):
        return []
    found: list[Path] = []
    for key in ("model_specs", "model_spec"):
        raw = plan.get(key)
        if isinstance(raw, str) and raw.strip():
            found.append(_norm(Path(raw), root))
        elif isinstance(raw, list):
            for item in raw:
                if isinstance(item, str) and item.strip():
                    found.append(_norm(Path(item), root))
    canary = plan.get("canary") or {}
    if isinstance(canary, dict):
        cms = canary.get("model_spec") or plan.get("canary_model_spec")
        if isinstance(cms, str) and cms.strip():
            found.append(_norm(Path(cms), root))
    cells = _unwrap_ps_list(plan.get("cells"))
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        ms = cell.get("model_spec")
        if isinstance(ms, str) and ms.strip():
            found.append(_norm(Path(ms), root))
    # unique, stable order
    out: list[Path] = []
    seen: set[Path] = set()
    for p in found:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _parse_cli_specs(
    cli_spec: str | Path | None,
    cli_specs: list[str] | None,
    root: Path,
) -> list[Path] | None:
    """CLI may be a single path, a comma-separated list, or repeated --model-spec."""
    parts: list[str] = []
    if cli_specs:
        for item in cli_specs:
            parts.extend(p.strip() for p in str(item).split(",") if p.strip())
    elif cli_spec is not None and str(cli_spec).strip():
        parts.extend(p.strip() for p in str(cli_spec).split(",") if p.strip())
    if not parts:
        return None
    return [_norm(Path(p), root) for p in parts]


def resolve_sealer_model_spec_set(
    *,
    root: Path,
    plan: dict[str, Any] | None,
    cells: list[dict[str, Any]] | None = None,
    cli_spec: str | Path | None = None,
    cli_specs: list[str] | None = None,
) -> list[dict[str, str]]:
    """Resolve the declared model-spec SET for a seal.

    Declared set = CLI paths (one, comma-separated, or repeated) when given,
    otherwise every distinct path recorded on the plan.

    Fails only when:
      - nothing declares a model spec
      - a cell's model_spec is outside the declared set
      - a declared path is missing on disk / has no ir_sha256
      - a cell's recorded ir_sha256 disagrees with the pin YAML

    Returns a list of {path, ir_sha256} for every distinct declared path that
    appears on a cell (and every declared path, so a canary-only pin still
    lands in the seal when declared).
    """
    recorded = recorded_model_specs_from_plan(plan, root)
    declared = _parse_cli_specs(cli_spec, cli_specs, root)
    if declared is None:
        declared = list(recorded)
    if not declared:
        raise SystemExit(
            "FATAL: session plan/manifest does not record a model spec path and "
            "no --model-spec set was supplied. Cannot seal a weight-IR identity "
            "that was never written."
        )

    declared_set = set(declared)
    for p in declared:
        if not p.is_file():
            raise SystemExit(f"FATAL: model spec not found: {p}")

    cell_list = cells if cells is not None else _unwrap_ps_list((plan or {}).get("cells"))
    for cell in cell_list:
        if not isinstance(cell, dict):
            continue
        ms = cell.get("model_spec")
        if not isinstance(ms, str) or not ms.strip():
            raise SystemExit(
                "FATAL: cell missing model_spec; cannot seal a multi-weight "
                f"session without per-cell identity (cell_index={cell.get('cell_index')})"
            )
        path = _norm(Path(ms), root)
        if path not in declared_set:
            raise SystemExit(
                "FATAL: cell model_spec is outside the declared set.\n"
                f"  cell_index={cell.get('cell_index')} arm={cell.get('arm')} "
                f"nc={cell.get('n_cached')} mode={cell.get('mode')} "
                f"r={cell.get('repeat')}\n"
                f"  cell model_spec: {path}\n"
                f"  declared set:    {[str(p) for p in declared]}"
            )
        pin_sha = _ir_sha256_from_spec(path)
        cell_sha = str(cell.get("ir_sha256") or "").strip().lower()
        if cell_sha and cell_sha != pin_sha:
            raise SystemExit(
                "FATAL: cell ir_sha256 disagrees with pin YAML.\n"
                f"  cell_index={cell.get('cell_index')} model_spec={path}\n"
                f"  cell ir_sha256: {cell_sha}\n"
                f"  pin  ir_sha256: {pin_sha}"
            )

    # Record every distinct declared (path, ir_sha256). Prefer stable order of
    # `declared`, then any recorded extras already filtered by the cell check.
    out: list[dict[str, str]] = []
    seen: set[Path] = set()
    for p in declared:
        if p in seen:
            continue
        seen.add(p)
        out.append(
            {
                "path": str(p),
                "path_repo": str(p.relative_to(root)).replace("\\", "/")
                if p.is_relative_to(root)
                else str(p),
                "ir_sha256": _ir_sha256_from_spec(p),
            }
        )
    return out


def resolve_sealer_model_spec(
    *,
    root: Path,
    default_spec: Path,
    cli_spec: str | Path | None,
    plan: dict[str, Any] | None,
) -> Path:
    """Resolve a SINGLE pin YAML for sealers that still emit one promote model block.

    Default is unchanged (``default_spec``) so existing single-model seals stay
    byte-identical when the session recorded that same path.

    Multi-weight sessions must use ``resolve_sealer_model_spec_set`` (W-2).
    """
    recorded = recorded_model_specs_from_plan(plan, root)
    if not recorded:
        raise SystemExit(
            "FATAL: session plan/manifest does not record a model spec path. "
            "Cannot seal a weight-IR identity that was never written. "
            "Re-run the matrix with tools/run_delta_prefill_matrix.ps1 that emits "
            "model_specs / per-cell model_spec + ir_sha256, then seal."
        )
    if len(recorded) > 1:
        raise SystemExit(
            "FATAL: session recorded multiple model specs "
            f"{[str(p) for p in recorded]}; this sealer entrypoint still takes a "
            "single model block. Use resolve_sealer_model_spec_set / --model-spec "
            "as a SET on seal_delta_prefill_session (W-2), or split seals per IR."
        )
    chosen = _norm(Path(cli_spec) if cli_spec is not None else default_spec, root)
    if recorded[0] != chosen:
        raise SystemExit(
            "FATAL: sealer model spec disagrees with the session record.\n"
            f"  sealer (--model-spec or default): {chosen}\n"
            f"  session recorded:                {recorded[0]}\n"
            "Refusing to write a false model block into the seal."
        )
    if not chosen.is_file():
        raise SystemExit(f"FATAL: model spec not found: {chosen}")
    return chosen
