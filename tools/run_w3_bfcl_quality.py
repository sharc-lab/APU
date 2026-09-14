"""W-3 BFCL quality worker — int8 (axis 4) against the a621ff7d entry pin.

Detached payload for tools/launch_w3.ps1. No measurement when imported.

Steps:
  1. Select the fixed 20 multi_turn_base prefix; write multi_turn_probe_entries.json.
  2. ASSERT entry ids == a621ff7d gpu_only RESIDENT pin; refuse on mismatch.
  3. Gold selftest via multi_turn_checker; refuse unless 20/20 BEFORE generation.
  4. Run session_residency RESIDENT / gpu_only on --model-spec (int8).
  5. Write per-entry ledger: model_spec, ir_sha256, per-turn pass/fail,
     failure bucket, wall times, force_quit / timeout visibility.

The probe loads one FetchedModelSpec per process. int4 cannot share this
session; do not work around with a second process under the same run_id.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

A621_SESSION = (
    ROOT
    / "derived"
    / "bfcl_feasibility"
    / "session_residency"
    / "a621ff7d-2919-463d-aaf6-673f9e6bafbc"
)
A621_ENTRIES = A621_SESSION / "session_residency_entries_gpu_only_RESIDENT.json"
DEFAULT_MODEL_SPEC = ROOT / "configs" / "models" / "Qwen3-4B-int8-ov.yaml"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _load_ids(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, list):
        raise SystemExit(f"REFUSED -- expected entry list in {path}")
    ids: list[str] = []
    for row in raw:
        if not isinstance(row, dict) or "id" not in row:
            raise SystemExit(f"REFUSED -- entry missing id in {path}")
        ids.append(str(row["id"]))
    return ids


def _ir_sha256(spec_path: Path) -> str:
    for line in spec_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("ir_sha256:"):
            return s.split(":", 1)[1].strip().strip('"').strip("'").lower()
    raise SystemExit(f"REFUSED -- ir_sha256 missing from {spec_path}")


def _entry_ledger_row(
    *,
    entry_result: dict[str, Any],
    model_spec: Path,
    ir_sha256: str,
) -> dict[str, Any]:
    score = entry_result.get("score") or {}
    turns: list[dict[str, Any]] = []
    for tm in entry_result.get("turn_metrics") or []:
        acc = tm.get("per_turn_accuracy") or {}
        calls_per_gen = tm.get("calls_emitted_per_generation")
        if calls_per_gen is None:
            # Additive field for future runs; sealed ledgers lack it.
            emitted = None
        else:
            emitted = any(int(n or 0) > 0 for n in calls_per_gen)
        turns.append(
            {
                "turn": tm.get("turn"),
                "pass": bool(acc.get("correct")) if acc else None,
                "per_turn_accuracy": acc,
                "wall_s_last_generation": tm.get("last_wall_s"),
                "n_generations": tm.get("n_generations"),
                "ttft_s": tm.get("ttft_s"),
                "decode_tok_s": tm.get("decode_tok_s"),
                "slo_ok": tm.get("slo_ok"),
                # Additive: whether any generation in this turn emitted a
                # parseable tool call. Sealed runs untouched (field absent there).
                "emitted_parseable_tool_call": emitted,
                # Additive phase timers (future runs). Sealed ledgers omit these.
                "turn_wall_s": tm.get("turn_wall_s"),
                "t_tool_exec": tm.get("t_tool_exec"),
                "t_template_build": tm.get("t_template_build"),
                "t_tokenize": tm.get("t_tokenize"),
                "t_generate": tm.get("t_generate"),
                "t_other": tm.get("t_other"),
            }
        )
    force_quit = bool(entry_result.get("force_quit"))
    error_type = score.get("error_type")
    # Instrument has no wall-clock generation timeout; force_quit is step-limit.
    return {
        "id": entry_result.get("id"),
        "model_spec": str(model_spec),
        "ir_sha256": ir_sha256,
        "trajectory_pass": score.get("valid") is True,
        "failure_bucket": error_type,
        "force_quit": force_quit,
        "force_terminated": error_type == "multi_turn:force_terminated" or force_quit,
        # Visible so slowness cannot masquerade as a timeout the instrument never had.
        "generation_timeout_hit": False,
        "generation_timeout_note": (
            "bfcl_feasibility_probe session_residency has no wall-clock generation "
            "timeout; termination taxonomy is multi_turn:force_terminated via "
            "MAXIMUM_STEP_LIMIT (20). generation_timeout_hit is always false."
        ),
        "wall_s": entry_result.get("wall_s"),
        "n_completed_turns": entry_result.get("n_completed_turns"),
        "n_user_turns": entry_result.get("n_user_turns"),
        "per_turn": turns,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--model-spec",
        type=Path,
        default=DEFAULT_MODEL_SPEC,
        help="FetchedModelSpec YAML (W-3 primary: Qwen3-4B-int8-ov.yaml)",
    )
    parser.add_argument("--arm", default="gpu_only")
    parser.add_argument(
        "--residency-mode", default="RESIDENT", choices=("RESIDENT", "NON_RESIDENT")
    )
    parser.add_argument("--n-entries", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--a621-entries",
        type=Path,
        default=A621_ENTRIES,
        help="Reference entry list whose ids must match exactly",
    )
    args = parser.parse_args(argv)

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    model_spec = args.model_spec if args.model_spec.is_absolute() else ROOT / args.model_spec
    if not model_spec.is_file():
        print(f"REFUSED -- model spec not found: {model_spec}", flush=True)
        return 2
    if not args.a621_entries.is_file():
        print(f"REFUSED -- a621 reference entries missing: {args.a621_entries}", flush=True)
        return 2

    # Import probe after path setup; bind model spec before any IR load.
    import tools.bfcl_feasibility_probe as probe

    probe.apply_model_spec(model_spec)
    ir_sha = _ir_sha256(model_spec)

    # Generation path: tools/bfcl_feasibility_probe.py run_session_residency /
    # generate_turn — identical for W-3 and a621ff7d (do_sample=False = greedy).
    # Library defaults for temperature/top_p/top_k/rng_seed exist on
    # openvino_genai.GenerationConfig but do not reach a sampler when
    # do_sample=False. seed=20260810 is entry-order pairing only (no shuffle);
    # it is not a sampler seed.
    generation_config = {
        "source": "tools/bfcl_feasibility_probe.py generate_turn GenerationConfig",
        "mode": "greedy",
        "do_sample": False,
        "max_new_tokens": int(args.max_new_tokens),
        "apply_chat_template": False,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 18446744073709551615,
        "rng_seed": 0,
        "sampler_active": False,
        "temperature_top_p_top_k_rng_note": (
            "Values above are openvino_genai.GenerationConfig library defaults "
            "after setting do_sample=False / max_new_tokens / apply_chat_template. "
            "With do_sample=False they do not reach the sampler; decoding is greedy "
            "argmax. A one-entry difference at n=20 is NOT sampling noise."
        ),
        "pairing_seed": int(args.seed),
        "pairing_seed_note": (
            "seed reaches select_multi_turn_entries / entry-order pairing only "
            "(no shuffle when n equals the fixed prefix). Not a sampler seed."
        ),
        "identical_to_a621ff7d": True,
        "identical_note": (
            "a621ff7d used the same generate_turn GenerationConfig construction "
            "(do_sample=False, max_new_tokens=512, apply_chat_template=False) via "
            "run_session_residency; no alternate sampler knobs in that path."
        ),
    }
    a621_report_rel = (
        "derived/bfcl_feasibility/session_residency/"
        "a621ff7d-2919-463d-aaf6-673f9e6bafbc/"
        "session_residency_gpu_only_RESIDENT_report.json"
    )
    a621_report_path = ROOT / Path(*a621_report_rel.split("/"))
    a621_rep = json.loads(a621_report_path.read_text(encoding="utf-8-sig"))
    a621_gp = a621_rep["gpu_probe"]
    a621_traj = a621_gp["accuracy_trajectory"]
    a621_f2 = a621_gp["accuracy_per_turn_f2"]
    a621_comparison_baseline = {
        "session_id": "a621ff7d-2919-463d-aaf6-673f9e6bafbc",
        "arm_id": a621_rep.get("arm_id"),
        "residency_mode": a621_rep.get("residency_mode"),
        "model_spec": (a621_rep.get("model") or {}).get("spec"),
        "artifact": a621_report_rel,
        "artifact_absolute": str(a621_report_path),
        "trajectory": {
            "correct": int(a621_traj["correct"]),
            "n": int(a621_traj["n"]),
            "accuracy": float(a621_traj["accuracy"]),
            "field": "gpu_probe.accuracy_trajectory",
        },
        "per_turn": {
            "correct": int(a621_f2["correct"]),
            "total": int(a621_f2["n"]),
            "accuracy": float(a621_f2["accuracy"]),
            "field": "gpu_probe.accuracy_per_turn_f2",
        },
        "note": (
            "Explicit W-3 comparison baseline copied from a621ff7d "
            "session_residency_gpu_only_RESIDENT_report.json so the seal is "
            "auditable without hunting the other run. Entry ids and generation "
            "config are held fixed; model_spec is taken from that report "
            "(weight axis may differ from W-3 int8)."
        ),
    }
    # Power / MDE pre-registration (before generation). Turn count for the
    # n=200 design is projected from the 20-entry pilot turn density; restate
    # at sealed achieved turn count when the arm finishes.
    n_run = int(args.n_entries)
    n_pilot = 20
    turns_int4_pilot, turns_int8_pilot = 68, 70
    n_turns_proj = int(round(((turns_int4_pilot + turns_int8_pilot) / 2) / n_pilot * n_run))
    # Two-proportion MDE: delta = (z_{1-a/2}+z_{1-b}) * sqrt(2 p (1-p) / n)
    # alpha=0.05 two-sided, power=0.80; planning p from pilot pooled rate.
    _z = 1.959963984540054 + 0.841621233572914
    _p_pool = (30 + 33) / (turns_int4_pilot + turns_int8_pilot)

    def _mde(n: float, p: float = _p_pool) -> float:
        return _z * (2.0 * p * (1.0 - p) / n) ** 0.5

    mde_unadj = _mde(float(n_turns_proj))
    mde_de2 = _mde(float(n_turns_proj) / 2.0)
    mde_pilot = _mde((turns_int4_pilot + turns_int8_pilot) / 2.0)
    power_analysis = {
        "endpoint": "per_turn_f2 two-proportion (int4 vs int8)",
        "alpha": 0.05,
        "power": 0.80,
        "two_sided": True,
        "design_effect": 2.0,
        "design_effect_note": ("DE=2 for intra-entry clustering of turns; neff = n_turns / DE."),
        "planning_p_pooled": _p_pool,
        "n_turns_planning": {
            "source": "scaled from 20-entry pilot turn density; restate at sealed",
            "n_entries_pilot": n_pilot,
            "n_entries_run": n_run,
            "int4_pilot_turns": turns_int4_pilot,
            "int8_pilot_turns": turns_int8_pilot,
            "projected_n_turns_per_arm": n_turns_proj,
        },
        "mde_absolute_accuracy": {
            "unadjusted": mde_unadj,
            "unadjusted_pp": round(100.0 * mde_unadj, 1),
            "design_effect_2": mde_de2,
            "design_effect_2_pp": round(100.0 * mde_de2, 1),
            "formula": "delta=(z_a/2+z_b)*sqrt(2*p*(1-p)/n); DE=2 uses n/2",
        },
        "pilot_20_entry_superseded": {
            "int4_per_turn": "30/68",
            "int8_per_turn": "33/70",
            "z": 0.36,
            "p": 0.72,
            "mde_pp_approx": round(100.0 * mde_pilot, 1),
            "note": (
                "Underpowered at MDE ~24 percentage points; superseded by the "
                "n=200 design, not contradicted."
            ),
        },
        "predicted_direction": None,
        "predicted_direction_note": (
            "No predicted direction for the quality effect. Same reason as "
            "amendment 2026-08-28c: register a directional prediction only where "
            "a derivation exists (ir_bytes+KV bands for size/bandwidth). BFCL "
            "per-turn quality has no analogous derivation from weight bit-width."
        ),
    }
    entry_population = {
        "selection": "select_multi_turn_entries / _select_from_spec questions[:n]",
        "difficulty_filter": False,
        "api_filter": False,
        "note": (
            "select_multi_turn_entries applies no difficulty or API filter — "
            "the original 20 were simply the file's first 20 — so the entry "
            "population was never biased relative to the benchmark. That "
            "eliminates biased subsampling as an explanation for the gap to "
            "published numbers."
        ),
    }
    plan: dict[str, Any] = {
        "kind": "w3_bfcl_quality",
        "session_id": args.session_id,
        "started_utc": _utc_now(),
        "arm": args.arm,
        "residency_mode": args.residency_mode,
        "n_entries": args.n_entries,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "model_spec": str(model_spec),
        "ir_sha256": ir_sha,
        "generation_config": generation_config,
        "a621_reference_session": "a621ff7d-2919-463d-aaf6-673f9e6bafbc",
        "a621_entries_path": str(args.a621_entries),
        "a621_comparison_baseline": a621_comparison_baseline,
        "power_analysis": power_analysis,
        "entry_population": entry_population,
        "dual_weight_in_session": False,
        "dual_weight_note": (
            "tools/bfcl_feasibility_probe.py binds one --model-spec per process "
            "(apply_model_spec + single LLMPipeline). int4 cannot run in the SAME "
            "session as int8. Reported rather than worked around with a second "
            "process under this run_id (W-2 within-session pairing does not apply)."
        ),
        "status": "running",
    }
    (out_dir / "plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # --- entry pin (ordered prefix of a621ff7d; not set-membership) ---
    entries = probe.select_multi_turn_entries()[: args.n_entries]
    pin_path = out_dir / probe.PINNED_MULTI_TURN_ENTRIES
    pin_path.write_text(json.dumps(entries, indent=2, default=str) + "\n", encoding="utf-8")
    got_ids = [str(e["id"]) for e in entries]
    ref_ids = _load_ids(args.a621_entries)
    plan["entry_assert"] = {
        "mode": "prefix",
        "n_reference": len(ref_ids),
        "n_run": len(got_ids),
    }
    if got_ids[: len(ref_ids)] != ref_ids:
        plan["status"] = "refused_entry_id_mismatch"
        plan["entry_ids_got"] = got_ids
        plan["entry_ids_a621"] = ref_ids
        (out_dir / "plan.json").write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            "REFUSED -- multi_turn_probe_entries.json prefix != a621ff7d "
            f"(mode=prefix n_reference={len(ref_ids)} n_run={len(got_ids)})",
            flush=True,
        )
        print(f"  got n={len(got_ids)} ref n={len(ref_ids)}", flush=True)
        for i, (a, b) in enumerate(zip(got_ids, ref_ids, strict=False)):
            if a != b:
                print(f"  first mismatch at [{i}]: got={a} a621={b}", flush=True)
                break
        if len(got_ids) < len(ref_ids):
            print(
                f"  length shortfall got={len(got_ids)} a621={len(ref_ids)} "
                "(prefix assert requires n_run >= n_reference)",
                flush=True,
            )
        elif len(got_ids) != len(ref_ids):
            print(
                f"  length differs got={len(got_ids)} a621={len(ref_ids)} "
                "(prefix mode allows n_run > n_reference when ordered prefix matches)",
                flush=True,
            )
        return 3
    (out_dir / "plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"entry_ids ASSERT ok: prefix mode — first {len(ref_ids)} of {len(got_ids)} "
        f"match a621ff7d (not a full-set match)",
        flush=True,
    )

    # --- gold selftest BEFORE generation (also enforced inside run_session_residency) ---
    gold = probe.run_multi_turn_gold_selftest(entries)
    gold_path = out_dir / "multi_turn_gold_selftest.json"
    gold_path.write_text(
        json.dumps(gold, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    n_gold = int(gold.get("n") or 0)
    n_valid = int(gold.get("n_valid") or 0)
    plan["gold_selftest"] = {"n": n_gold, "n_valid": n_valid, "path": str(gold_path)}
    if n_gold != len(entries) or n_valid != n_gold or n_gold < 1:
        plan["status"] = "refused_gold_selftest"
        plan["ended_utc"] = _utc_now()
        (out_dir / "plan.json").write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            f"REFUSED -- gold selftest {n_valid}/{n_gold} (need {len(entries)}/{len(entries)})",
            flush=True,
        )
        return 3
    print(f"gold selftest PASS: {n_valid}/{n_gold}", flush=True)

    # --- arm ---
    print(
        f"running session_residency arm={args.arm} mode={args.residency_mode} "
        f"model_spec={model_spec}",
        flush=True,
    )
    report = probe.run_session_residency(
        out_dir,
        arm_id=args.arm,
        residency_mode=args.residency_mode,
        n_entries=args.n_entries,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    per_entry_raw = (report.get("gpu_probe") or {}).get("per_entry") or []
    ledger = [
        _entry_ledger_row(entry_result=row, model_spec=model_spec, ir_sha256=ir_sha)
        for row in per_entry_raw
    ]
    traj = (report.get("gpu_probe") or {}).get("accuracy_trajectory") or {}
    per_turn = (report.get("gpu_probe") or {}).get("accuracy_per_turn_f2") or {}
    n_force = sum(1 for row in ledger if row["force_terminated"])
    n_timeout = sum(1 for row in ledger if row["generation_timeout_hit"])

    summary = {
        "kind": "w3_bfcl_quality",
        "session_id": args.session_id,
        "model_spec": str(model_spec),
        "ir_sha256": ir_sha,
        "arm": args.arm,
        "residency_mode": args.residency_mode,
        "entry_ids_match_a621": True,
        "entry_assert": plan.get("entry_assert"),
        "gold_selftest_n_valid": n_valid,
        "gold_selftest_n": n_gold,
        "accuracy_trajectory": traj,
        "accuracy_per_turn_f2": per_turn,
        "n_force_terminated": n_force,
        "n_generation_timeout_hit": n_timeout,
        "dual_weight_in_session": False,
        "per_entry": ledger,
        "report_artifact": report.get("artifact"),
        "ended_utc": _utc_now(),
        "status": "complete",
    }
    (out_dir / "w3_entry_ledger.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    plan["status"] = "complete"
    plan["ended_utc"] = summary["ended_utc"]
    plan["accuracy_trajectory"] = traj
    plan["n_force_terminated"] = n_force
    plan["n_generation_timeout_hit"] = n_timeout
    plan["ledger"] = str(out_dir / "w3_entry_ledger.json")
    (out_dir / "plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "ok": True,
                "session_id": args.session_id,
                "trajectory": traj,
                "per_turn": per_turn,
                "n_force_terminated": n_force,
                "n_generation_timeout_hit": n_timeout,
                "out": str(out_dir),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
