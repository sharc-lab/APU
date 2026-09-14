"""H-1 launch canary: same fixed cell as C-2 / matrix (INF-1b).

Fixed cell (docs/CANARY_PROTOCOL.md / tools/ttft_slo_canary.py):
  arm=gpu_only_f16  n_cached=4000  delta=400  mode=RESIDENT
  C=3  rel_drift_floor=0.05  onset_s=657

Runs the opening canary before H-1 spawn. Trip => FAIL_CANARY_DRIFT (exit 2).
Also runs the dual-bound N / unarmed-seal preflight (no silent unguarded).

Usage:
  .venv-seam\\Scripts\\python.exe tools/run_h1_canary_opening.py --out DIR
  .venv-seam\\Scripts\\python.exe tools/run_h1_canary_opening.py --logic-only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.ttft_slo_canary import (
    CALIBRATION_C,
    CANARY_ARM,
    CANARY_DELTA,
    CANARY_MODE,
    CANARY_N_CACHED,
    ONSET_S,
    REL_DRIFT_FLOOR,
    CanaryBudgetRefuse,
    CanaryDriftAbort,
    CanaryUnarmedSealRefuse,
    TtftSloCanaryGuard,
    assert_canary_budget_fits,
    assert_seal_requires_armed_or_unguarded,
    derive_canary_every_n,
    new_canary_gate,
    update_canary_drift_bookkeeping,
)


def _logic_preflight(*, mean_wall_s: float = 9.93) -> dict:
    """Dual-bound N + arming schedule + unarmed-seal refuse (no GPU)."""
    out: dict = {"fixed_cell": {
        "arm": CANARY_ARM,
        "n_cached": CANARY_N_CACHED,
        "delta": CANARY_DELTA,
        "mode": CANARY_MODE,
    }, "calibration_c": CALIBRATION_C, "rel_drift_floor": REL_DRIFT_FLOOR, "onset_s": ONSET_S}

    cases = []
    for planned in (39, 300):
        assert_canary_budget_fits(planned, calibration_c=CALIBRATION_C)
        d = derive_canary_every_n(
            [mean_wall_s] * max(1, planned - 1),
            planned_probe_count=planned,
            calibration_c=CALIBRATION_C,
            onset_s=ONSET_S,
        )
        n = int(d["n"])
        gate = new_canary_gate(calibration_c=CALIBRATION_C, rel_drift_floor=REL_DRIFT_FLOOR)
        prior: list[dict] = []
        idx = 0

        def fire() -> None:
            nonlocal idx
            rec = {
                "classification": "OK",
                "turn1_prefill_s": 2.66 + 0.001 * idx,
                "turn2_prefill_s": 1.06,
                "canary_index": idx,
            }
            bk = update_canary_drift_bookkeeping(
                rec=rec, gate=gate, prior_ok_canaries=prior, calibration_c=CALIBRATION_C
            )
            prior.append(bk["rec"])
            idx += 1

        fire()
        since = 0
        for _ in range(planned):
            since += 1
            if since >= n:
                fire()
                since = 0
        if not gate.get("armed"):
            raise SystemExit(
                f"REFUSED -- planned={planned} N={n} did not arm "
                f"(n_onset={d.get('n_onset')} n_budget={d.get('n_budget')} "
                f"bound={d.get('binding_bound')})"
            )
        cases.append({
            "planned_probe_count": planned,
            "n": n,
            "n_onset": d.get("n_onset"),
            "n_budget": d.get("n_budget"),
            "binding_bound": d.get("binding_bound"),
            "armed": True,
            "n_canaries": idx,
        })

    try:
        assert_seal_requires_armed_or_unguarded(armed=False, allow_unguarded=False)
        raise SystemExit("REFUSED -- unarmed seal must refuse without AllowUnguarded")
    except CanaryUnarmedSealRefuse:
        pass

    out["cases"] = cases
    out["unarmed_seal_refused"] = True
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=None, help="Work dir for opening canary artifacts")
    p.add_argument(
        "--model-spec",
        type=Path,
        default=ROOT / "configs" / "models" / "Qwen3-4B-int4-ov.yaml",
    )
    p.add_argument(
        "--planned-probe-count",
        type=int,
        default=300,
        help="Budget for guard init (H-1 default 300; must fit C+1 canaries)",
    )
    p.add_argument(
        "--logic-only",
        action="store_true",
        help="Dual-bound + seal refuse only (no GPU opening cell)",
    )
    p.add_argument("--allow-unguarded", action="store_true")
    args = p.parse_args(argv)

    print(
        f"[h1_canary] fixed_cell arm={CANARY_ARM} nc={CANARY_N_CACHED} "
        f"d={CANARY_DELTA} mode={CANARY_MODE} C={CALIBRATION_C} "
        f"floor={REL_DRIFT_FLOOR} onset_s={ONSET_S}",
        flush=True,
    )

    logic = _logic_preflight()
    for c in logic["cases"]:
        print(
            f"[h1_canary] derive planned={c['planned_probe_count']} "
            f"n_onset={c['n_onset']} n_budget={c['n_budget']} "
            f"N={c['n']} bound={c['binding_bound']} armed={c['armed']}",
            flush=True,
        )
    print("[h1_canary] unarmed seal refused (default)", flush=True)

    if args.logic_only:
        print(json.dumps({"ok": True, "logic": logic}, indent=2, sort_keys=True))
        return 0

    out = args.out or (ROOT / "derived" / "h1_hybrid" / "_canary_opening")
    out.mkdir(parents=True, exist_ok=True)
    model_spec = args.model_spec if args.model_spec.is_absolute() else ROOT / args.model_spec
    try:
        guard = TtftSloCanaryGuard(
            root=ROOT,
            model_spec=model_spec,
            work_dir=out / "canaries",
            planned_probe_count=int(args.planned_probe_count),
            allow_unguarded=bool(args.allow_unguarded),
        )
    except CanaryBudgetRefuse as exc:
        print(f"REFUSED -- {exc.detail}", flush=True)
        return 2

    print(
        f"[h1_canary] opening cell (planned={args.planned_probe_count} "
        f"every_n={guard.n_every} bound={(guard.n_derivation or {}).get('binding_bound')})",
        flush=True,
    )
    try:
        guard.opening()
    except CanaryDriftAbort as exc:
        print(f"REFUSED -- FAIL_CANARY_DRIFT: {exc.detail}", flush=True)
        (out / "opening_abort.json").write_text(
            json.dumps({"detail": exc.detail, "canary": exc.canary_record}, indent=2) + "\n",
            encoding="utf-8",
        )
        return 2

    frag = guard.plan_fragment()
    (out / "opening_plan.json").write_text(
        json.dumps({"logic": logic, "canary": frag, "canaries": guard.canaries}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "n_canaries": len(guard.canaries),
                "armed": bool(guard.gate.get("armed")),
                "fixed_cell": frag["fixed_cell"],
                "out": str(out),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
