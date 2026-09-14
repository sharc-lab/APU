"""Seal a completed C-2 TTFT-bound limit session (write-once derived seal).

Copies session artifacts under derived/c2_ttft/<session_id>/ into
sealed_<session_id>/, writes manifest.json + summary.json + .sealed
(tree_sha256). Does not mutate the source session dir. Does not promote
to raw/ (derived_diagnostic only).

Usage:
  .\\.venv-seam\\Scripts\\python.exe tools\\seal_c2_ttft.py \\
      --session-id 62395fdb-1899-415f-b708-6adc81a24dda
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

SESSION_BASE = ROOT / "derived" / "c2_ttft"
WORKLOAD_KIND = "c2_ttft_bound_limit"
IR_PIN = "c1821f29332faa48871c7f16426a21443fbc701fa4e4c8a581a8c51ab7bf2cb2"
BIN_PIN = "074214fa29ab1b479536fc7184ef3036e7b355a5835e8590b61b70a8de1df2f9"

COPY_FILES = (
    "plan.json",
    "summary.json",
    "arm_results.json",
    "probes.ndjson",
    "watchdog_kills.jsonl",
)

ARM_EXPECT = {
    "gpu_only_f16": "f16",
    "gpu_only_u8": "u8",
    "gpu_only_u4": "u4",
}


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


def _integrity(session_dir: Path, summary: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    probes_path = session_dir / "probes.ndjson"
    if not probes_path.is_file():
        raise SystemExit(f"REFUSED -- missing {probes_path}")
    probes = [
        json.loads(line)
        for line in probes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    work_results = sorted((session_dir / "work").glob("*.result.json"))
    if not work_results:
        raise SystemExit("REFUSED -- no work/*.result.json")

    arms = summary.get("arm_results") or []
    if len(arms) != 3:
        raise SystemExit(f"REFUSED -- expected 3 arms, got {len(arms)}")
    for arm in arms:
        if arm.get("status") != "complete":
            raise SystemExit(f"REFUSED -- arm {arm.get('arm_id')} status={arm.get('status')!r}")
        if arm.get("ttft_limit_n") != 10000 and arm.get("ceiling_n_cached") != 10000:
            raise SystemExit(
                f"REFUSED -- arm {arm.get('arm_id')} unexpected limit "
                f"ttft={arm.get('ttft_limit_n')} ceiling={arm.get('ceiling_n_cached')}"
            )

    null_prefill = [p for p in probes if p.get("prefill_s") is None]
    if null_prefill:
        raise SystemExit(f"REFUSED -- {len(null_prefill)} probes with null prefill_s")

    ir = plan.get("ir_sha256") or summary.get("ir_sha256")
    if ir != IR_PIN:
        raise SystemExit(f"REFUSED -- plan ir_sha256 {ir!r} != pin {IR_PIN}")

    model_dirs: set[str] = set()
    kv_ok = 0
    peak_rss_n = 0
    for rpath in work_results:
        d = _read_json(rpath)
        spec = d.get("spec") or {}
        if spec.get("model_dir"):
            model_dirs.add(str(spec["model_dir"]))
        loads = d.get("loads") or []
        if not loads:
            raise SystemExit(f"REFUSED -- no loads in {rpath.name}")
        kv = loads[0].get("kv_cache_precision") or {}
        arm_id = str(spec.get("arm_id") or rpath.name.split(".")[0])
        expect = ARM_EXPECT[arm_id]
        readback = (kv.get("readback") or {}).get("normalized")
        if kv.get("match") is not True or kv.get("requested") != expect or readback != expect:
            raise SystemExit(f"REFUSED -- KV mismatch in {rpath.name}: {kv}")
        kv_ok += 1
        gen = d.get("generation") or {}
        if gen.get("peak_rss_bytes") is None:
            raise SystemExit(f"REFUSED -- missing peak_rss_bytes in {rpath.name}")
        if gen.get("prefill_s") is None and d.get("completed"):
            raise SystemExit(f"REFUSED -- completed with null prefill_s: {rpath.name}")
        peak_rss_n += 1

    if len(model_dirs) != 1:
        raise SystemExit(f"REFUSED -- model_dir not unique: {model_dirs}")
    model_dir = Path(next(iter(model_dirs)))
    bin_path = model_dir / "openvino_model.bin"
    if not bin_path.is_file():
        raise SystemExit(f"REFUSED -- missing IR bin {bin_path}")
    disk_bin = _sha256_file(bin_path)
    if disk_bin != BIN_PIN:
        raise SystemExit(f"REFUSED -- openvino_model.bin sha256 {disk_bin} != {BIN_PIN}")

    wd = session_dir / "watchdog_kills.jsonl"
    if not wd.is_file():
        raise SystemExit("REFUSED -- missing watchdog_kills.jsonl")
    wd_lines = [ln for ln in wd.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not wd_lines:
        raise SystemExit("REFUSED -- empty watchdog_kills.jsonl")

    primary = summary.get("primary_claim_eval") or plan.get("primary_claim_eval") or {}
    limits = summary.get("ttft_limits") or primary.get("limits") or {}
    for arm_id in ARM_EXPECT:
        if limits.get(arm_id) != 10000:
            raise SystemExit(f"REFUSED -- ttft_limits[{arm_id}]={limits.get(arm_id)}")

    # Criterion path does not emit canaries; record absence explicitly.
    canary_applicable = False

    return {
        "status": "integrity_pass",
        "arms_converged": True,
        "n_arms": 3,
        "ttft_limits": limits,
        "primary_prediction_held": bool(primary.get("primary_prediction_held")),
        "n_probes_ndjson": len(probes),
        "n_work_results": len(work_results),
        "prefill_s_non_null_every_probe": True,
        "kv_readback_match_every_cell": True,
        "n_kv_ok": kv_ok,
        "ir_sha256": ir,
        "ir_sha256_pin_ok": True,
        "openvino_model_bin_sha256": disk_bin,
        "openvino_model_bin_pin_ok": True,
        "model_dir": str(model_dir),
        "watchdog_log_present": True,
        "watchdog_log_lines": len(wd_lines),
        "orchestrator_ws_peak": {
            "field": "generation.peak_rss_bytes",
            "n_cells_with_peak": peak_rss_n,
            "note": (
                "C-2 runs via tools/run_c1_ceiling.py (Python worker), not the "
                "PowerShell delta_prefill orchestrator_ws_guard. Per-probe "
                "peak_rss_bytes is the recorded working-set peak."
            ),
        },
        "canary": {
            "criterion_path_runs_canary": canary_applicable,
            "note": (
                "Session 62395fdb predates INF-1 canary wiring on ttft_slo; "
                "see derived/c2_ttft/NOTE_62395fdb_unguarded.md. Later ttft_slo "
                "runs must emit canaries and abort on FAIL_CANARY_DRIFT."
            ),
        },
        "unguarded_session_note": {
            "path": "derived/c2_ttft/NOTE_62395fdb_unguarded.md",
            "mitigating_observation": (
                "three sequentially-run arms all landed on exactly 10,000; "
                "incidental evidence the instrument held, not a canary."
            ),
        },
    }


def seal_session(*, session_id: str, allow_unguarded: bool = False) -> Path:
    session_dir = SESSION_BASE / session_id
    if not session_dir.is_dir():
        raise SystemExit(f"REFUSED -- missing session dir {session_dir}")

    summary_path = session_dir / "summary.json"
    plan_path = session_dir / "plan.json"
    if not summary_path.is_file() or not plan_path.is_file():
        raise SystemExit("REFUSED -- missing plan.json or summary.json")
    summary = _read_json(summary_path)
    plan = _read_json(plan_path)
    if summary.get("status") != "complete":
        raise SystemExit(f"REFUSED -- status={summary.get('status')!r} (want complete)")

    # INF-1b: refuse unarmed canary seal unless UNGUARDED / AllowUnguarded.
    from tools.ttft_slo_canary import (
        CanaryUnarmedSealRefuse,
        assert_seal_requires_armed_or_unguarded,
    )

    canary = summary.get("canary") or plan.get("canary") or {}
    gate = canary.get("canary_gate") or {}
    armed = bool(gate.get("armed"))
    unguarded_flag = bool(summary.get("UNGUARDED") or plan.get("UNGUARDED"))
    try:
        seal_guard = assert_seal_requires_armed_or_unguarded(
            armed=armed,
            allow_unguarded=bool(allow_unguarded) or unguarded_flag,
            unguarded_already=unguarded_flag,
        )
    except CanaryUnarmedSealRefuse as exc:
        raise SystemExit(f"REFUSED -- {exc.detail}") from exc
    if seal_guard.get("UNGUARDED"):
        summary["UNGUARDED"] = True
        plan["UNGUARDED"] = True

    integrity = _integrity(session_dir, summary, plan)

    out = SESSION_BASE / f"sealed_{session_id}"
    if out.exists():
        raise SystemExit(f"REFUSED -- seal already exists (write-once): {out}")

    out.mkdir(parents=True, exist_ok=False)
    artifacts_dir = out / "artifacts"
    artifacts_dir.mkdir()
    work_dest = artifacts_dir / "work"
    work_dest.mkdir()

    manifest_artifacts: list[dict[str, Any]] = []
    for name in COPY_FILES:
        src = session_dir / name
        if not src.is_file():
            raise SystemExit(f"REFUSED -- missing required artifact: {name}")
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

    work_src = session_dir / "work"
    n_work_copied = 0
    for src in sorted(work_src.glob("*")):
        if not src.is_file():
            continue
        dest = work_dest / src.name
        shutil.copy2(src, dest)
        n_work_copied += 1
    manifest_artifacts.append(
        {
            "name": "work/",
            "artifact": "artifacts/work/",
            "n_files": n_work_copied,
            "note": "Per-probe spec/result JSON; hashed via seal tree_sha256.",
        }
    )

    sealed_utc = _utc_now()
    primary = summary.get("primary_claim_eval") or plan.get("primary_claim_eval") or {}
    manifest: dict[str, Any] = {
        "run_id": session_id,
        "session_id": session_id,
        "status": "COMPLETE",
        "kind": WORKLOAD_KIND,
        "seal_style": "derived_diagnostic",
        "sealed_utc": sealed_utc,
        "UNGUARDED": bool(summary.get("UNGUARDED")),
        "source_session": _rel(session_dir),
        "model_spec": summary.get("model_spec") or plan.get("model_spec"),
        "ir_sha256": integrity["ir_sha256"],
        "criterion": summary.get("criterion") or plan.get("criterion"),
        "slo_s": plan.get("slo_s") or 10.0,
        "ttft_limits": integrity["ttft_limits"],
        "primary_claim_eval": primary,
        "integrity": integrity,
        "artifacts": manifest_artifacts,
        "raw_emit_blocked": {
            "reason": ("derived_diagnostic seal only; raw/ promote not requested for C-2.")
        },
    }
    sealed_summary: dict[str, Any] = {
        "run_id": session_id,
        "session_id": session_id,
        "status": "COMPLETE",
        "kind": WORKLOAD_KIND,
        "seal_style": "derived_diagnostic",
        "ir_sha256": integrity["ir_sha256"],
        "ttft_limits": integrity["ttft_limits"],
        "primary_prediction_held": integrity["primary_prediction_held"],
        "integrity_status": integrity["status"],
        "sealed_utc": sealed_utc,
        "UNGUARDED": bool(summary.get("UNGUARDED")),
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
        "integrity_status": integrity["status"],
        "note": (
            "Advisory marker for derived_diagnostic seals. Not "
            "seam.rawstore.verify_sealed; raw/ was not written. Do not mutate "
            "this directory after seal."
        ),
    }
    (out / ".sealed").write_text(
        json.dumps(seal_marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "ok": True,
                "session_id": session_id,
                "seal_dir": str(out),
                "seal_id": f"sealed_{session_id}",
                "tree_sha256": tree_hash,
                "integrity_status": integrity["status"],
                "ttft_limits": integrity["ttft_limits"],
            },
            indent=2,
        )
    )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    parser.add_argument(
        "--allow-unguarded",
        action="store_true",
        help="Permit seal when canary armed==false; writes UNGUARDED into seal.",
    )
    args = parser.parse_args(argv)
    seal_session(session_id=args.session_id, allow_unguarded=bool(args.allow_unguarded))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
