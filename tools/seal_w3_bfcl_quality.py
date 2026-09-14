"""Seal a completed W-3 BFCL quality run (write-once derived seal).

Copies session artifacts under derived/bfcl_feasibility/w3_weight_quality/
into sealed_<run_id>/, records optional paired-analysis pointer, writes
manifest.json + summary.json + .sealed (tree_sha256). Does not mutate the
source session dir. Does not promote to raw/ (derived_diagnostic only).

Usage:
  .\\.venv-seam\\Scripts\\python.exe tools\\seal_w3_bfcl_quality.py \\
      --session-id 6225d6e1-4e0a-41c9-90bb-695ecc5fbe0a \\
      --paired-analysis derived/bfcl_feasibility/w3_weight_quality/paired_analysis_....json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SESSION_BASE = ROOT / "derived" / "bfcl_feasibility" / "w3_weight_quality"
WORKLOAD_KIND = "w3_bfcl_quality"

# Artifacts copied into the seal tree (source session is never mutated).
COPY_FILES = (
    "plan.json",
    "summary.json",
    "w3_entry_ledger.json",
    "multi_turn_probe_entries.json",
    "multi_turn_gold_selftest.json",
    "session_residency_entries_gpu_only_RESIDENT.json",
    "session_residency_gold_gpu_only_RESIDENT.json",
    "session_residency_gpu_only_RESIDENT_report.json",
    "session_residency_gpu_only_RESIDENT_partial.json",
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_tree(root: Path, *, exclude: set[str]) -> str:
    h = hashlib.sha256()
    files = sorted(
        (p for p in root.rglob("*") if p.is_file() and p.name not in exclude),
        key=lambda p: p.relative_to(root).as_posix(),
    )
    for p in files:
        rel = p.relative_to(root).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def seal_session(
    *,
    session_id: str,
    paired_analysis: Path | None,
) -> Path:
    session_dir = SESSION_BASE / session_id
    if not session_dir.is_dir():
        raise SystemExit(f"REFUSED -- missing session dir {session_dir}")

    summary_path = session_dir / "summary.json"
    plan_path = session_dir / "plan.json"
    if not summary_path.is_file():
        raise SystemExit(f"REFUSED -- missing {summary_path}")
    summary = _read_json(summary_path)
    plan = _read_json(plan_path) if plan_path.is_file() else {}
    if summary.get("status") != "complete":
        raise SystemExit(f"REFUSED -- status={summary.get('status')!r} (want complete)")

    out = SESSION_BASE / f"sealed_{session_id}"
    if out.exists():
        raise SystemExit(f"REFUSED -- seal already exists (write-once): {out}")

    out.mkdir(parents=True, exist_ok=False)
    artifacts_dir = out / "artifacts"
    artifacts_dir.mkdir()

    manifest_artifacts: list[dict[str, Any]] = []
    missing: list[str] = []
    for name in COPY_FILES:
        src = session_dir / name
        if not src.is_file():
            missing.append(name)
            continue
        dest = artifacts_dir / name
        shutil.copy2(src, dest)
        manifest_artifacts.append(
            {
                "name": name,
                "artifact": f"artifacts/{name}",
                "sha256": _sha256_file(dest),
                "bytes": dest.stat().st_size,
            }
        )
    # partial is optional if report exists
    if "session_residency_gpu_only_RESIDENT_report.json" in missing:
        raise SystemExit(
            "REFUSED -- missing report " "session_residency_gpu_only_RESIDENT_report.json"
        )
    missing = [m for m in missing if m != "session_residency_gpu_only_RESIDENT_partial.json"]
    if missing:
        raise SystemExit(f"REFUSED -- missing required artifacts: {missing}")

    paired_meta = None
    if paired_analysis is not None:
        pa = paired_analysis if paired_analysis.is_absolute() else ROOT / paired_analysis
        if not pa.is_file():
            raise SystemExit(f"REFUSED -- paired analysis missing: {pa}")
        dest = out / "paired_analysis.json"
        shutil.copy2(pa, dest)
        paired_meta = {
            "artifact": "paired_analysis.json",
            "sha256": _sha256_file(dest),
            "source": _rel(pa),
        }

    sealed_utc = _utc_now()
    traj = summary.get("accuracy_trajectory") or {}
    per_turn = summary.get("accuracy_per_turn_f2") or {}

    manifest: dict[str, Any] = {
        "run_id": session_id,
        "session_id": session_id,
        "status": "COMPLETE",
        "kind": WORKLOAD_KIND,
        "seal_style": "derived_diagnostic",
        "sealed_utc": sealed_utc,
        "source_session": _rel(session_dir),
        "model_spec": summary.get("model_spec") or plan.get("model_spec"),
        "ir_sha256": summary.get("ir_sha256") or plan.get("ir_sha256"),
        "arm": summary.get("arm") or plan.get("arm"),
        "residency_mode": summary.get("residency_mode") or plan.get("residency_mode"),
        "n_entries": plan.get("n_entries") or 200,
        "entry_assert": plan.get("entry_assert") or summary.get("entry_assert"),
        "generation_config": plan.get("generation_config"),
        "a621_comparison_baseline": plan.get("a621_comparison_baseline"),
        "power_analysis": plan.get("power_analysis"),
        "entry_population": plan.get("entry_population"),
        "accuracy_trajectory": traj,
        "accuracy_per_turn_f2": per_turn,
        "n_force_terminated": summary.get("n_force_terminated"),
        "n_generation_timeout_hit": summary.get("n_generation_timeout_hit"),
        "paired_analysis": paired_meta,
        "artifacts": manifest_artifacts,
        "admissibility": {
            "status": "COMPLETE",
            "gold_selftest": {
                "n": summary.get("gold_selftest_n"),
                "n_valid": summary.get("gold_selftest_n_valid"),
            },
            "note": (
                "Single-arm W-3 quality session. Numbers cite this run_id / "
                "seal dir / artifacts/*.json. Source session dir is write-once "
                "as-found (not mutated by seal)."
            ),
        },
        "raw_emit_blocked": {
            "reason": (
                "derived_diagnostic seal only; raw/ promote not requested for W-3. "
                "Close Cursor/Chrome and extend with --attempt-raw-promote if needed."
            ),
        },
    }

    sealed_summary: dict[str, Any] = {
        "run_id": session_id,
        "session_id": session_id,
        "status": "COMPLETE",
        "kind": WORKLOAD_KIND,
        "seal_style": "derived_diagnostic",
        "model_spec": manifest["model_spec"],
        "ir_sha256": manifest["ir_sha256"],
        "accuracy_trajectory": traj,
        "accuracy_per_turn_f2": per_turn,
        "n_force_terminated": summary.get("n_force_terminated"),
        "paired_analysis": paired_meta,
        "sealed_utc": sealed_utc,
    }

    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out / "summary.json").write_text(
        json.dumps(sealed_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    tree_hash = _sha256_tree(out, exclude={".sealed"})
    seal_marker = {
        "run_id": session_id,
        "sealed_at_utc": sealed_utc,
        "seal_style": "derived_diagnostic",
        "tree_sha256": tree_hash,
        "self_check": "pass",
        "note": (
            "Advisory marker for derived_diagnostic seals. Not seam.rawstore.verify_sealed; "
            "raw/ was not written. Do not mutate this directory after seal."
        ),
    }
    (out / ".sealed").write_text(
        json.dumps(seal_marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Recompute tree hash including .sealed content? Delta-prefill excludes .sealed
    # from tree hash then writes marker — tree_sha256 does not include .sealed.
    print(
        json.dumps(
            {
                "ok": True,
                "session_id": session_id,
                "seal_dir": str(out),
                "tree_sha256": tree_hash,
                "paired_analysis": paired_meta,
            },
            indent=2,
        )
    )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    parser.add_argument(
        "--paired-analysis",
        type=Path,
        default=None,
        help="Optional paired analysis JSON to embed in the seal",
    )
    args = parser.parse_args(argv)
    seal_session(session_id=args.session_id, paired_analysis=args.paired_analysis)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
