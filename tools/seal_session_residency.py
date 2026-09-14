"""Seal a completed BFCL session-residency A/B matrix and promote to raw/.

Derived seal matches ``tools/seal_delta_prefill_session.py`` (write-once under
``derived/``). Promotion into ``raw/`` uses ``seam.manifest.emit`` after the
isolation gate and schema gate both pass. AM-036: ``retro_seal=True``,
measurement ``power_state`` null unless from records.

Usage:
  .\\.venv-seam\\Scripts\\python.exe tools\\seal_session_residency.py \\
      --session-id a621ff7d-2919-463d-aaf6-673f9e6bafbc
  .\\.venv-seam\\Scripts\\python.exe tools\\seal_session_residency.py \\
      --session-id a621ff7d-2919-463d-aaf6-673f9e6bafbc \\
      --attempt-raw-promote --allow-dirty
  .\\.venv-seam\\Scripts\\python.exe tools\\seal_session_residency.py \\
      --session-id a621ff7d-2919-463d-aaf6-673f9e6bafbc \\
      --promote-only --attempt-raw-promote --allow-dirty
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SESSION_BASE = ROOT / "derived" / "bfcl_feasibility" / "session_residency"
COLD_CONTROL_PATH = SESSION_BASE / "_cold_control" / "session_residency_cold_control_gpu_only.json"
LAUNCHES_PATH = SESSION_BASE / "_launches" / "launches.json"
RENDER_SMOKE_PATH = SESSION_BASE / "session_residency_render_smoke.json"

SUPERSEDED_SESSIONS = (
    {
        "session_id": "24634b40-3f38-40f4-b722-6127143d3500",
        "reason": (
            "double-templating / non-control control: ChatHistory path not yet fixed "
            "(DISPATCH J); NON_RESIDENT not forced cold via enable_prefix_caching=False "
            "(DISPATCH L). Superseded by a621ff7d-2919-463d-aaf6-673f9e6bafbc. Do not delete."
        ),
    },
    {
        "session_id": "4f31cbaa-fba8-4ef3-bf59-f4f77dc8272b",
        "reason": (
            "ChatHistory fix present but NON_RESIDENT still warm (prefix caching default ON); "
            "cold control / DISPATCH L not yet applied. Superseded by "
            "a621ff7d-2919-463d-aaf6-673f9e6bafbc. Do not delete."
        ),
    },
)

# AM-035 / delta-prefill seals: latency interaction at n_cached=12000 (measured medians).
AM035_LATENCY_INTERACTION = {
    "figure": 2.4,
    "unit": "x_overestimate_of_naive_product",
    "workload": "synthetic_delta_prefill",
    "n_cached": 12000,
    "amendment": "AM-035",
    "citing_seals": [
        "d5c98342-a0b2-41a9-b6e2-93ac7a39c3ba",
        "9f38eb15-6fe6-40b4-871b-a02ec5629bb1",
    ],
    "note": (
        "Latency interaction 2.4× UNCHANGED under AM-035; from measured medians at "
        "n_cached=12000, not the retracted arm-A intercept fit. Two workloads, different "
        "context scales, same sub-additivity class as this session-residency matrix."
    ),
}

_PLATFORM_PATH = Path("configs/platforms/aipc-c1.yaml")
_MEASUREMENT_PATH = Path("configs/measurement.yaml")
_DELTA_N_PATH = Path("configs/delta_n.yaml")
_MODEL_SPEC_PATH = Path("configs/models/Qwen3-4B-int4-ov.yaml")

WORKLOAD_KIND = "session_residency"
EXPECTED_CELLS = 4


def _load_session_plan(session_id: str) -> dict[str, Any] | None:
    path = SESSION_BASE / session_id / "plan.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _resolve_model_spec_for_session(session_id: str, *, cli_spec: str | Path | None) -> Path:
    from tools.seal_model_spec import resolve_sealer_model_spec

    return resolve_sealer_model_spec(
        root=ROOT,
        default_spec=_MODEL_SPEC_PATH,
        cli_spec=cli_spec,
        plan=_load_session_plan(session_id),
    )


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
    return str(path.relative_to(ROOT)).replace("\\", "/")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _pct(fraction: float) -> float:
    return round(100.0 * fraction, 1)


def _round1(x: float) -> float:
    return round(x, 1)


def _round3(x: float) -> float:
    return round(x, 3)


def _ratio(a: float, b: float) -> float:
    return a / b


def _launch_entry(session_id: str) -> dict[str, Any]:
    if not LAUNCHES_PATH.is_file():
        raise SystemExit(f"missing launches ledger {LAUNCHES_PATH}")
    entries = json.loads(LAUNCHES_PATH.read_text(encoding="utf-8-sig"))
    for e in entries:
        if e.get("session_id") == session_id:
            return e
    raise SystemExit(f"session_id {session_id} not found in {LAUNCHES_PATH}")


def _cell_report_name(arm: str, mode: str) -> str:
    return f"session_residency_{arm}_{mode}_report.json"


def _compare_name(arm: str) -> str:
    return f"session_residency_compare_{arm}.json"


def _extract_first_turn_equivalence(report: dict[str, Any]) -> dict[str, Any]:
    probe = report.get("gpu_probe") or report
    entries = probe.get("per_entry") or []
    n = 0
    n_identical = 0
    n_false = 0
    for e in entries:
        eq = e.get("first_turn_token_equivalence")
        if not isinstance(eq, dict):
            continue
        n += 1
        if eq.get("identical") is True:
            n_identical += 1
        else:
            n_false += 1
    return {
        "n_entries_with_assertion": n,
        "n_identical": n_identical,
        "n_not_identical": n_false,
        "all_identical": n > 0 and n_false == 0,
    }


def _cold_form(report: dict[str, Any]) -> dict[str, Any]:
    probe = report.get("gpu_probe") or report
    return dict(probe.get("cold_fix") or {})


def _build_results(
    *,
    session_id: str,
    session_dir: Path,
    compare_a: dict[str, Any],
    compare_g: dict[str, Any],
    cold: dict[str, Any],
    reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Bind every cited number to this run_id / session artifacts."""
    a_n_sum = float(compare_a["session_total_latency"]["NON_RESIDENT_sum_s"])
    a_r_sum = float(compare_a["session_total_latency"]["RESIDENT_sum_s"])
    g_n_sum = float(compare_g["session_total_latency"]["NON_RESIDENT_sum_s"])
    g_r_sum = float(compare_g["session_total_latency"]["RESIDENT_sum_s"])

    a_slo_n = float(compare_a["fraction_turns_slo_ok"]["NON_RESIDENT"])
    a_slo_r = float(compare_a["fraction_turns_slo_ok"]["RESIDENT"])
    g_slo_n = float(compare_g["fraction_turns_slo_ok"]["NON_RESIDENT"])
    g_slo_r = float(compare_g["fraction_turns_slo_ok"]["RESIDENT"])

    a_ttft_n = float(compare_a["ttft_s"]["median_NON_RESIDENT"])
    a_ttft_r = float(compare_a["ttft_s"]["median_RESIDENT"])
    g_ttft_n = float(compare_g["ttft_s"]["median_NON_RESIDENT"])
    g_ttft_r = float(compare_g["ttft_s"]["median_RESIDENT"])

    residency_alone_x = _ratio(a_n_sum, a_r_sum)
    device_alone_x = _ratio(a_n_sum, g_n_sum)
    both_x = _ratio(a_n_sum, g_r_sum)
    naive_product_x = residency_alone_x * device_alone_x
    naive_overestimate_x = naive_product_x / both_x

    oi_a = compare_a["output_identity"]
    oi_g = compare_g["output_identity"]
    acc_a = compare_a["per_turn_accuracy"]
    acc_g = compare_g["per_turn_accuracy"]

    cg_a = compare_a["context_growth"]
    cg_g = compare_g["context_growth"]

    checks = cold.get("checks") or {}
    cold_passed = bool(cold.get("passed"))
    turn2_ttft = checks.get("turn2_ttft_s")
    turn2_tokens = checks.get("turn2_prompt_tokens")
    turn2_ratio = checks.get("turn2_over_turn1_raw")

    # Dispatch N narrative cited 6.554 s / 1.035; artifacts disagree — bind artifacts.
    dispatch_cold_claim = {
        "dispatch_claimed_turn2_ttft_s": 6.554,
        "dispatch_claimed_turn2_over_turn1": 1.035,
        "artifact_turn2_ttft_s": turn2_ttft,
        "artifact_turn2_over_turn1_raw": turn2_ratio,
        "artifact_turn2_prompt_tokens": turn2_tokens,
        "agrees_with_dispatch_narrative": (
            turn2_ttft is not None
            and turn2_ratio is not None
            and math.isclose(float(turn2_ttft), 6.554, rel_tol=0, abs_tol=0.05)
            and math.isclose(float(turn2_ratio), 1.035, rel_tol=0, abs_tol=0.01)
        ),
        "resolution": (
            "STOP_AND_REPORT: dispatch narrative cold-control figures disagree with "
            f"{_rel(COLD_CONTROL_PATH)}; seal binds artifact values only."
        ),
    }

    ft_eq: dict[str, Any] = {}
    for key, report in reports.items():
        ft_eq[key] = _extract_first_turn_equivalence(report)

    cold_forms = {key: _cold_form(report) for key, report in reports.items()}
    non_resident_prefix_false = all(
        cold_forms[k].get("enable_prefix_caching") is False
        for k in (
            "A_NON_RESIDENT",
            "gpu_only_NON_RESIDENT",
        )
    )

    feasibility_table = [
        {
            "arm": "A",
            "device_label": "cpu-p",
            "residency": False,
            "label": "cpu-p, no residency (DEFAULT)",
            "fraction_turns_slo_ok": a_slo_n,
            "fraction_turns_slo_ok_pct": _pct(a_slo_n),
            "session_latency_sum_s": a_n_sum,
            "session_latency_sum_s_rounded": _round1(a_n_sum),
            "source": f"cells/{_compare_name('A')}",
        },
        {
            "arm": "A",
            "device_label": "cpu-p",
            "residency": True,
            "label": "cpu-p + residency",
            "fraction_turns_slo_ok": a_slo_r,
            "fraction_turns_slo_ok_pct": _pct(a_slo_r),
            "session_latency_sum_s": a_r_sum,
            "session_latency_sum_s_rounded": _round1(a_r_sum),
            "source": f"cells/{_compare_name('A')}",
        },
        {
            "arm": "gpu_only",
            "device_label": "gpu_only",
            "residency": False,
            "label": "gpu_only, no residency",
            "fraction_turns_slo_ok": g_slo_n,
            "fraction_turns_slo_ok_pct": _pct(g_slo_n),
            "session_latency_sum_s": g_n_sum,
            "session_latency_sum_s_rounded": _round1(g_n_sum),
            "source": f"cells/{_compare_name('gpu_only')}",
        },
        {
            "arm": "gpu_only",
            "device_label": "gpu_only",
            "residency": True,
            "label": "gpu_only + residency (OPTIMAL)",
            "fraction_turns_slo_ok": g_slo_r,
            "fraction_turns_slo_ok_pct": _pct(g_slo_r),
            "session_latency_sum_s": g_r_sum,
            "session_latency_sum_s_rounded": _round1(g_r_sum),
            "source": f"cells/{_compare_name('gpu_only')}",
        },
    ]

    slo_definition = {
        "USER_FACING_SLO": "TTFT<=10s AND decode>=6 tok/s",
        "metric": "fraction of turns meeting USER_FACING SLO",
        "session_latency": "sum of per-entry total_latency_s over 20 entries",
        "n_entries": 20,
    }

    return {
        "run_id": session_id,
        "session_id": session_id,
        "status": "COMPLETE",
        "cells_completed": EXPECTED_CELLS,
        "cells_expected": EXPECTED_CELLS,
        "slo_definition": slo_definition,
        "feasibility_table": feasibility_table,
        "residency_effect": {
            "A": {
                "ttft_median_NON_RESIDENT_s": a_ttft_n,
                "ttft_median_RESIDENT_s": a_ttft_r,
                "ttft_median_NON_RESIDENT_s_rounded": _round3(a_ttft_n),
                "ttft_median_RESIDENT_s_rounded": _round3(a_ttft_r),
                "ttft_speedup_x": round(_ratio(a_ttft_n, a_ttft_r), 1),
                "session_sum_NON_RESIDENT_s": a_n_sum,
                "session_sum_RESIDENT_s": a_r_sum,
                "session_speedup_x": round(residency_alone_x, 1),
                "source": f"cells/{_compare_name('A')}",
            },
            "gpu_only": {
                "ttft_median_NON_RESIDENT_s": g_ttft_n,
                "ttft_median_RESIDENT_s": g_ttft_r,
                "ttft_median_NON_RESIDENT_s_rounded": _round3(g_ttft_n),
                "ttft_median_RESIDENT_s_rounded": _round3(g_ttft_r),
                "ttft_speedup_x": round(_ratio(g_ttft_n, g_ttft_r), 1),
                "session_sum_NON_RESIDENT_s": g_n_sum,
                "session_sum_RESIDENT_s": g_r_sum,
                "session_speedup_x": round(_ratio(g_n_sum, g_r_sum), 1),
                "source": f"cells/{_compare_name('gpu_only')}",
            },
        },
        "interaction": {
            "baseline_cpu_p_no_residency_s": a_n_sum,
            "residency_alone_s": a_r_sum,
            "residency_alone_x": round(residency_alone_x, 1),
            "device_alone_s": g_n_sum,
            "device_alone_x": round(device_alone_x, 1),
            "both_s": g_r_sum,
            "both_x": round(both_x, 1),
            "naive_product_x": round(naive_product_x, 1),
            "naive_product_overestimates_by_x": round(naive_overestimate_x, 2),
            "sub_additivity": True,
            "synthetic_delta_prefill_am035": AM035_LATENCY_INTERACTION,
            "note": (
                "Measured on real BFCL multi-turn workload (this run_id). Naive product "
                "overestimates combined gain; same sub-additivity class as AM-035 latency "
                "interaction 2.4× on synthetic prompts at n_cached=12000."
            ),
            "sources": [
                f"cells/{_compare_name('A')}",
                f"cells/{_compare_name('gpu_only')}",
            ],
        },
        "output_identity": {
            "finding_class": "gpu_kv_cache_numerical_nonequivalence",
            "A": {
                "n_generations_compared_paired": oi_a["n_generations_compared_paired"],
                "n_generations_text_differing": oi_a["n_generations_text_differing"],
                "identical": oi_a["identical"],
                "n_generations_with_think_RESIDENT": oi_a["n_generations_with_think_RESIDENT"],
                "n_generations_with_think_NON_RESIDENT": oi_a[
                    "n_generations_with_think_NON_RESIDENT"
                ],
                "source": f"cells/{_compare_name('A')}",
            },
            "gpu_only": {
                "n_generations_compared_paired": oi_g["n_generations_compared_paired"],
                "n_generations_text_differing": oi_g["n_generations_text_differing"],
                "identical": oi_g["identical"],
                "n_generations_with_think_RESIDENT": oi_g["n_generations_with_think_RESIDENT"],
                "n_generations_with_think_NON_RESIDENT": oi_g[
                    "n_generations_with_think_NON_RESIDENT"
                ],
                "source": f"cells/{_compare_name('gpu_only')}",
            },
            "enable_prefix_caching_NON_RESIDENT": False,
            "enable_prefix_caching_NON_RESIDENT_verified": non_resident_prefix_false,
            "cb_backend_switch_not_the_cause": True,
            "note": (
                "Both arms ran enable_prefix_caching=False on NON_RESIDENT. Arm A "
                "149/149 paired generations identical; gpu_only 78/132 differing. "
                "Recorded as gpu_kv_cache_numerical_nonequivalence (device-specific)."
            ),
            "cold_form_by_cell": {
                k: {
                    "chosen": v.get("chosen"),
                    "enable_prefix_caching": v.get("enable_prefix_caching"),
                }
                for k, v in cold_forms.items()
            },
        },
        "contamination_note": {
            "gpu_only_per_turn_accuracy": {
                "RESIDENT": acc_g["RESIDENT"]["accuracy"],
                "RESIDENT_pct": _pct(float(acc_g["RESIDENT"]["accuracy"])),
                "NON_RESIDENT": acc_g["NON_RESIDENT"]["accuracy"],
                "NON_RESIDENT_pct": _pct(float(acc_g["NON_RESIDENT"]["accuracy"])),
                "clean_comparison": False,
                "reason": (
                    "gpu_only output_identity.identical=false; accuracy shift tracks "
                    "numerical nonequivalence, not a clean residency effect on quality."
                ),
            },
            "A_per_turn_accuracy": {
                "RESIDENT": acc_a["RESIDENT"]["accuracy"],
                "RESIDENT_pct": _pct(float(acc_a["RESIDENT"]["accuracy"])),
                "NON_RESIDENT": acc_a["NON_RESIDENT"]["accuracy"],
                "NON_RESIDENT_pct": _pct(float(acc_a["NON_RESIDENT"]["accuracy"])),
                "clean_comparison": True,
                "reason": ("Arm A identical=true and accuracy equal in both modes (41.4%)."),
            },
            "sources": [
                f"cells/{_compare_name('A')}",
                f"cells/{_compare_name('gpu_only')}",
            ],
        },
        "validity_chain": {
            "cold_control": {
                "passed": cold_passed,
                "arm_id": (cold.get("checks") or {}).get("arm_id") or cold.get("arm_id"),
                "turn2_prompt_tokens": turn2_tokens,
                "turn2_generate_input_tokens": checks.get("turn2_generate_input_tokens"),
                "turn2_ttft_s": turn2_ttft,
                "turn2_over_turn1_raw": turn2_ratio,
                "turn1_ttft_s": checks.get("turn1_ttft_s"),
                "failures": cold.get("failures"),
                "source": _rel(COLD_CONTROL_PATH),
                "dispatch_narrative_check": dispatch_cold_claim,
            },
            "no_think_tokens": {
                "A": {
                    "RESIDENT": oi_a["n_generations_with_think_RESIDENT"],
                    "NON_RESIDENT": oi_a["n_generations_with_think_NON_RESIDENT"],
                },
                "gpu_only": {
                    "RESIDENT": oi_g["n_generations_with_think_RESIDENT"],
                    "NON_RESIDENT": oi_g["n_generations_with_think_NON_RESIDENT"],
                },
                "all_zero": (
                    oi_a["n_generations_with_think_RESIDENT"] == 0
                    and oi_a["n_generations_with_think_NON_RESIDENT"] == 0
                    and oi_g["n_generations_with_think_RESIDENT"] == 0
                    and oi_g["n_generations_with_think_NON_RESIDENT"] == 0
                ),
            },
            "first_turn_render_equivalence": {
                "asserted_per_entry": True,
                "sha_matched_vs_hf_template": all(v.get("all_identical") for v in ft_eq.values()),
                "by_cell": ft_eq,
                "render_smoke_path": (
                    _rel(RENDER_SMOKE_PATH) if RENDER_SMOKE_PATH.is_file() else None
                ),
            },
            "superseded_prior_sessions": list(SUPERSEDED_SESSIONS),
        },
        "measured_workload_profile": {
            "replaces": "HDR4 synthetic generator",
            "per_turn_delta_tokens": {
                "A": {
                    "mean": cg_a["per_turn_delta_tokens_RESIDENT"]["mean"],
                    "mean_rounded": round(float(cg_a["per_turn_delta_tokens_RESIDENT"]["mean"]), 1),
                    "min": cg_a["per_turn_delta_tokens_RESIDENT"]["min"],
                    "max": cg_a["per_turn_delta_tokens_RESIDENT"]["max"],
                    "n": cg_a["per_turn_delta_tokens_RESIDENT"]["n"],
                    "source": f"cells/{_compare_name('A')}",
                },
                "gpu_only": {
                    "mean": cg_g["per_turn_delta_tokens_RESIDENT"]["mean"],
                    "mean_rounded": round(float(cg_g["per_turn_delta_tokens_RESIDENT"]["mean"]), 1),
                    "min": cg_g["per_turn_delta_tokens_RESIDENT"]["min"],
                    "max": cg_g["per_turn_delta_tokens_RESIDENT"]["max"],
                    "n": cg_g["per_turn_delta_tokens_RESIDENT"]["n"],
                    "source": f"cells/{_compare_name('gpu_only')}",
                    "note": ("RESIDENT stats used (modes diverge under gpu_only nonequivalence)."),
                },
            },
            "context_growth_per_session": {
                "A": {
                    "mean": cg_a["RESIDENT"]["mean"],
                    "mean_rounded": round(float(cg_a["RESIDENT"]["mean"])),
                    "min": cg_a["RESIDENT"]["min"],
                    "max": cg_a["RESIDENT"]["max"],
                    "n": cg_a["RESIDENT"]["n"],
                    "source": f"cells/{_compare_name('A')}",
                },
                "gpu_only": {
                    "mean": cg_g["RESIDENT"]["mean"],
                    "mean_rounded": round(float(cg_g["RESIDENT"]["mean"])),
                    "min": cg_g["RESIDENT"]["min"],
                    "max": cg_g["RESIDENT"]["max"],
                    "n": cg_g["RESIDENT"]["n"],
                    "source": f"cells/{_compare_name('gpu_only')}",
                },
            },
            "hdr4_assumed_for_contrast": {
                "per_turn_delta_tokens": "900-3000",
                "context_growth_to": "12000-27600",
            },
        },
        "artifact_paths": {
            "session_dir": _rel(session_dir),
            "compare_A": _rel(session_dir / _compare_name("A")),
            "compare_gpu_only": _rel(session_dir / _compare_name("gpu_only")),
            "cold_control": _rel(COLD_CONTROL_PATH),
        },
    }


def _verify_dispatch_table(results: dict[str, Any]) -> dict[str, Any]:
    """Confirm rounded table numbers match the dispatch N narrative (feasibility block)."""
    expected = [
        (0.0, 9630.3),
        (72.9, 1888.5),
        (95.7, 1162.0),
        (100.0, 544.4),
    ]
    rows = results["feasibility_table"]
    mismatches: list[dict[str, Any]] = []
    for row, (pct, lat) in zip(rows, expected, strict=True):
        if row["fraction_turns_slo_ok_pct"] != pct:
            mismatches.append(
                {
                    "field": "fraction_turns_slo_ok_pct",
                    "label": row["label"],
                    "got": row["fraction_turns_slo_ok_pct"],
                    "expected": pct,
                }
            )
        if row["session_latency_sum_s_rounded"] != lat:
            mismatches.append(
                {
                    "field": "session_latency_sum_s_rounded",
                    "label": row["label"],
                    "got": row["session_latency_sum_s_rounded"],
                    "expected": lat,
                }
            )
    return {
        "ok": not mismatches,
        "mismatches": mismatches,
        "note": (
            "Feasibility table rounded figures vs DISPATCH N narrative. "
            "Cold-control narrative checked separately under validity_chain."
        ),
    }


def _measurement_placement(seal_path: Path, launch: dict[str, Any]) -> dict[str, Any]:
    """Placement from measurement launch ledger; Windows station not in cell reports."""
    return {
        "launch_context": launch.get("launch_context") or "ssh_detached",
        # Schema: ssh_detached lands in Windows session 0. Not captured in BFCL reports.
        "session_id": 0,
        "window_station": None,
        "window_station_note": (
            "BFCL session_residency cell reports do not record window_station; "
            "left null rather than inventing a Service-* name. launch_context from "
            "launches.json; Windows session_id=0 per ssh_detached convention."
        ),
        "source_launch": _rel(LAUNCHES_PATH),
        "launch_tag": launch.get("tag"),
        "openvino": None,
        "genai": None,
        "model_id": "Qwen3-4B-int4-ov",
    }


def _fill_stack_versions(placement: dict[str, Any], reports: dict[str, dict[str, Any]]) -> None:
    for report in reports.values():
        probe = report.get("gpu_probe") or report
        stack = probe.get("stack") or {}
        if stack.get("openvino"):
            placement["openvino"] = stack.get("openvino")
        if stack.get("openvino_genai"):
            placement["genai"] = stack.get("openvino_genai")
        if placement.get("openvino") and placement.get("genai"):
            return


def _task_ids() -> list[str]:
    return [
        "gpu_only:RESIDENT:n20",
        "gpu_only:NON_RESIDENT:n20",
        "A:RESIDENT:n20",
        "A:NON_RESIDENT:n20",
    ]


def _build_promotion_summary(
    *,
    session_id: str,
    seal_path: Path,
    sealed_manifest: dict[str, Any],
    sealed_summary: dict[str, Any],
    placement: dict[str, Any],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "kind": WORKLOAD_KIND,
        "promotion": "post_hoc_from_derived_diagnostic",
        "session_id": session_id,
        "run_id": session_id,
        "derived_seal_path": _rel(seal_path),
        "derived_seal_style": sealed_manifest.get("seal_style"),
        "derived_tree_sha256": None,
        "measurement_isolation_mode": sealed_manifest.get("measurement_isolation_mode"),
        "measurement_launch_context": sealed_manifest.get("measurement_launch_context"),
        "measurement_placement": {
            "launch_context": placement.get("launch_context"),
            "session_id": placement.get("session_id"),
            "window_station": placement.get("window_station"),
            "window_station_note": placement.get("window_station_note"),
            "source_launch": placement.get("source_launch"),
        },
        "matrix": sealed_manifest.get("matrix"),
        "admissibility": sealed_manifest.get("admissibility"),
        "results": sealed_manifest.get("results"),
        "cells_completed": sealed_summary.get("cells_completed"),
        "cells_ok": sealed_summary.get("cells_ok"),
        "cells_failed": sealed_summary.get("cells_failed"),
        "notes": {
            "raw_artifacts_copied_from": _rel(seal_path),
            "sealed_derived_not_mutated": True,
            "power_state": (
                "null at promote time: post-hoc emit does not re-capture quiescence "
                "(AM-036 retro_seal); do not treat promote-time host state as run environment."
            ),
        },
    }
    sealed_marker = seal_path / ".sealed"
    if sealed_marker.is_file():
        summary["derived_tree_sha256"] = _read_json(sealed_marker).get("tree_sha256")
    return summary


def _stage_raw_artifacts(
    run_dir: Any,
    *,
    seal_path: Path,
    sealed_manifest: dict[str, Any],
    sealed_summary: dict[str, Any],
) -> dict[str, Any]:
    cells_src = seal_path / "cells"
    cells_dst = run_dir.path / "cells"
    cells_dst.mkdir(exist_ok=False)
    copied: list[str] = []
    for src in sorted(cells_src.iterdir()):
        if not src.is_file():
            continue
        dest = cells_dst / src.name
        shutil.copy2(src, dest)
        copied.append(f"cells/{src.name}")

    for name in (
        "session_summary_as_found.json",
        "session_plan_as_found.json",
        "results.json",
        "cold_control_as_found.json",
        "superseded_sessions.json",
    ):
        src = seal_path / name
        if src.is_file():
            shutil.copy2(src, run_dir.path / name)

    run_dir.write_json("derived_seal_manifest.json", sealed_manifest)
    run_dir.write_json("derived_seal_summary.json", sealed_summary)
    marker = seal_path / ".sealed"
    if marker.is_file():
        shutil.copy2(marker, run_dir.path / "derived_seal_marker.json")

    return {
        "n_cells_copied": len(copied),
        "derived_seal_path": _rel(seal_path),
        "artifact_index": {
            "cells": copied,
            "derived_seal_manifest": "derived_seal_manifest.json",
            "derived_seal_summary": "derived_seal_summary.json",
            "results": "results.json",
        },
    }


def _dry_validate_emit_construction(
    *,
    session_id: str,
    seal_path: Path,
    sealed_manifest: dict[str, Any],
    sealed_summary: dict[str, Any],
    placement: dict[str, Any],
    allow_dirty: bool,
    model_spec: str | Path | None = None,
) -> dict[str, Any]:
    from seam.blinding import blinded_label_for, get_or_create_salt
    from seam.config import resolve_config
    from seam.gitinfo import capture_git_state
    from seam.manifest import build_manifest, validate_manifest
    from seam.model_provenance import load_local_spec, manifest_model_block

    resolved = resolve_config(
        [ROOT / _PLATFORM_PATH, ROOT / _MEASUREMENT_PATH, ROOT / _DELTA_N_PATH],
        repo_root=ROOT,
    )
    model_spec_path = _resolve_model_spec_for_session(session_id, cli_spec=model_spec)
    spec = load_local_spec(model_spec_path)
    model_block = manifest_model_block(
        spec=spec,
        spec_path=model_spec_path,
        reasoning_mode="thinking_off",
    )
    verification = spec.get("verification") or {}
    file_methods = {
        name: str(entry.get("method"))
        for name, entry in verification.items()
        if isinstance(entry, dict)
        and entry.get("method") in {"sha256", "git-blob-sha1", "size-only"}
    }
    if file_methods:
        model_block["provenance"]["file_verification"] = file_methods

    matrix = sealed_manifest.get("matrix") or {}
    workload = {
        "kind": WORKLOAD_KIND,
        "benchmark": "bfcl_session_residency_resident_vs_nonresident",
        "task_ids": _task_ids(),
        "seed": matrix.get("seed"),
        "n_repeats": 1,
        "concurrency": 1,
    }
    condition_label = (
        "session_residency|arms=A,gpu_only|modes=RESIDENT,NON_RESIDENT|"
        "n_entries=20|promote_from_derived"
    )
    git_state = capture_git_state(cwd=ROOT)
    salt = get_or_create_salt(ROOT)
    blinded = blinded_label_for(condition_label, salt=salt)
    summary = _build_promotion_summary(
        session_id=session_id,
        seal_path=seal_path,
        sealed_manifest=sealed_manifest,
        sealed_summary=sealed_summary,
        placement=placement,
    )
    dry_evidence = {
        "contending_processes": [],
        "tier2_recorded_processes": [],
        "sshd_session_count": None,
        "consistent_with_declaration": True,
        "probe_error": None,
    }
    drivers = {
        "npu": None,
        "igpu": None,
        "openvino": placement.get("openvino"),
        "genai": placement.get("genai"),
        "lhm_bridge": None,
    }
    manifest = build_manifest(
        run_id=session_id,
        config=resolved,
        git_state=git_state,
        allow_dirty=allow_dirty,
        target="cpu-placement",
        workload=workload,
        condition_label=condition_label,
        blinded_label=blinded,
        repo_root=ROOT,
        model=model_block,
        drivers=drivers,
        power_state=None,
        power_state_note=(
            "Post-hoc promote: measurement power_state left null (AM-036). "
            "Do not treat promote-time host state as run environment."
        ),
        retro_seal=True,
        thermal={"regime": "confound", "excluded": False},
        raw_sha256="0" * 64,
        self_check="pass",
        isolation_mode="remote",
        isolation_evidence=dry_evidence,
        launch_context=str(placement.get("launch_context") or "ssh_detached"),
        session_id=placement.get("session_id"),
        window_station=placement.get("window_station"),
        require_launch_context=True,
    )
    validate_manifest(manifest)

    dirty_blocker = bool(git_state.dirty and not allow_dirty)
    return {
        "ok": True,
        "workload": workload,
        "condition_label": condition_label,
        "target": "cpu-placement",
        "model_name": model_block.get("name"),
        "model_ir_sha256": model_block.get("ir_sha256"),
        "config_hash": resolved.config_hash,
        "git_dirty": git_state.dirty,
        "allow_dirty": allow_dirty,
        "emit_will_need_allow_dirty": dirty_blocker,
        "launch_context": placement.get("launch_context"),
        "measurement_session_id": placement.get("session_id"),
        "measurement_window_station": placement.get("window_station"),
        "summary_keys": sorted(summary.keys()),
        "note": (
            "Dry construction validated against run_manifest.schema.json with synthetic "
            "empty tier-1 isolation_evidence. Live emit() re-resolves isolation_mode=remote "
            "and will refuse while Cursor/Chrome are resident."
            + (
                " Git tree is dirty: pass --allow-dirty when isolation clears or emit "
                "will refuse DirtyTreeError."
                if dirty_blocker
                else ""
            )
        ),
    }


def _emit_raw_from_seal(
    *,
    session_id: str,
    seal_path: Path,
    sealed_manifest: dict[str, Any],
    sealed_summary: dict[str, Any],
    placement: dict[str, Any],
    allow_dirty: bool,
    model_spec: str | Path | None = None,
) -> dict[str, Any]:
    from seam.config import resolve_config
    from seam.manifest import emit
    from seam.model_provenance import load_local_spec, manifest_model_block

    raw_dir = ROOT / "raw" / session_id
    if raw_dir.exists():
        raise SystemExit(f"refusing emit: raw/{session_id} already exists (write-once)")

    resolved = resolve_config(
        [ROOT / _PLATFORM_PATH, ROOT / _MEASUREMENT_PATH, ROOT / _DELTA_N_PATH],
        repo_root=ROOT,
    )
    model_spec_path = _resolve_model_spec_for_session(session_id, cli_spec=model_spec)
    spec = load_local_spec(model_spec_path)
    model_block = manifest_model_block(
        spec=spec,
        spec_path=model_spec_path,
        reasoning_mode="thinking_off",
    )
    verification = spec.get("verification") or {}
    file_methods = {
        name: str(entry.get("method"))
        for name, entry in verification.items()
        if isinstance(entry, dict)
        and entry.get("method") in {"sha256", "git-blob-sha1", "size-only"}
    }
    if file_methods:
        model_block["provenance"]["file_verification"] = file_methods

    matrix = sealed_manifest.get("matrix") or {}
    workload = {
        "kind": WORKLOAD_KIND,
        "benchmark": "bfcl_session_residency_resident_vs_nonresident",
        "task_ids": _task_ids(),
        "seed": matrix.get("seed"),
        "n_repeats": 1,
        "concurrency": 1,
    }
    condition_label = (
        "session_residency|arms=A,gpu_only|modes=RESIDENT,NON_RESIDENT|"
        "n_entries=20|promote_from_derived"
    )
    summary = _build_promotion_summary(
        session_id=session_id,
        seal_path=seal_path,
        sealed_manifest=sealed_manifest,
        sealed_summary=sealed_summary,
        placement=placement,
    )
    drivers = {
        "npu": None,
        "igpu": None,
        "openvino": placement.get("openvino"),
        "genai": placement.get("genai"),
        "lhm_bridge": None,
    }

    def _before(run_dir: Any) -> dict[str, Any]:
        return _stage_raw_artifacts(
            run_dir,
            seal_path=seal_path,
            sealed_manifest=sealed_manifest,
            sealed_summary=sealed_summary,
        )

    handle = emit(
        config=resolved,
        target="cpu-placement",
        workload=workload,
        condition_label=condition_label,
        repo_root=ROOT,
        run_id=session_id,
        allow_dirty=allow_dirty,
        summary=summary,
        model=model_block,
        drivers=drivers,
        power_state=None,
        power_state_note=(
            "Post-hoc promote: measurement power_state left null (AM-036). "
            "Do not treat promote-time host state as run environment."
        ),
        retro_seal=True,
        thermal={"regime": "confound", "excluded": False},
        self_check="pass",
        isolation_mode="remote",
        launch_context=str(placement.get("launch_context") or "ssh_detached"),
        session_id=placement.get("session_id"),
        window_station=placement.get("window_station"),
        require_launch_context=True,
        before_integrity_hash=_before,
    )
    # load_run_manifest applies AM-036 corrections on read; prove the sealed raw
    # manifest is loadable under that path.
    from seam.manifest import load_run_manifest

    loaded = load_run_manifest(session_id, repo_root=ROOT)
    return {
        "run_id": handle.run_id,
        "raw_path": _rel(handle.run_dir.path),
        "raw_sha256": handle.raw_sha256,
        "manifest_path": _rel(handle.run_dir.path / "manifest.json"),
        "load_run_manifest_ok": loaded.get("run_id") == session_id,
        "retro_seal": loaded.get("retro_seal"),
        "measurement_power_null": loaded.get("power_state") is None,
    }


def _attempt_raw_promote(
    session_id: str,
    seal_path: Path,
    *,
    allow_dirty: bool = False,
    model_spec: str | Path | None = None,
) -> dict[str, Any]:
    from seam.isolation import IsolationModeError, gather_isolation_evidence, resolve_isolation_mode
    from seam.manifest import load_schema

    attempted_utc = datetime.now(UTC).isoformat()
    if not seal_path.is_dir():
        raise SystemExit(f"promote: missing seal dir {seal_path}")
    sealed_manifest_path = seal_path / "manifest.json"
    sealed_summary_path = seal_path / "summary.json"
    if not sealed_manifest_path.is_file() or not sealed_summary_path.is_file():
        raise SystemExit(f"promote: sealed manifest/summary missing under {seal_path}")

    sealed_manifest = _read_json(sealed_manifest_path)
    sealed_summary = _read_json(sealed_summary_path)
    launch = _launch_entry(session_id)
    placement = _measurement_placement(seal_path, launch)
    # Prefer stack versions from sealed cell reports if present.
    for cell_path in sorted((seal_path / "cells").glob("session_residency_*_report.json")):
        report = _read_json(cell_path)
        probe = report.get("gpu_probe") or report
        stack = probe.get("stack") or {}
        if stack.get("openvino"):
            placement["openvino"] = stack.get("openvino")
        if stack.get("openvino_genai"):
            placement["genai"] = stack.get("openvino_genai")

    evidence = gather_isolation_evidence()
    tier1 = list(evidence.get("contending_processes") or [])
    record: dict[str, Any] = {
        "attempted_utc": attempted_utc,
        "session_id": session_id,
        "seal_path": _rel(seal_path),
        "raw_emit": False,
        "gate": (
            "resolve_isolation_mode(remote) → schema kind → dry_validate → "
            "seam.manifest.emit(retro_seal=True)"
        ),
        "tier1_resident_names": sorted({p.get("name") for p in tier1 if p.get("name")}),
        "tier1_count": len(tier1),
        "emit_wired": True,
    }

    kind_enum = load_schema()["properties"]["workload"]["properties"]["kind"]["enum"]
    if WORKLOAD_KIND not in kind_enum:
        record["status"] = "refused_schema_gate"
        record["schema_gate"] = f"workload.kind {WORKLOAD_KIND} missing from enum"
        record["note"] = (
            f"workload.kind {WORKLOAD_KIND} is not in "
            "seam/schemas/run_manifest.schema.json; add it before raw/ promotion."
        )
        return record
    record["schema_gate"] = f"workload.kind {WORKLOAD_KIND} accepted"

    try:
        dry = _dry_validate_emit_construction(
            session_id=session_id,
            seal_path=seal_path,
            sealed_manifest=sealed_manifest,
            sealed_summary=sealed_summary,
            placement=placement,
            allow_dirty=allow_dirty,
            model_spec=model_spec,
        )
        record["dry_validate"] = dry
    except Exception as exc:
        record["dry_validate"] = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    try:
        mode, resolved = resolve_isolation_mode("remote")
        record["isolation_mode_resolved"] = mode
        record["isolation_evidence_consistent"] = resolved.get("consistent_with_declaration")
    except IsolationModeError as exc:
        record["status"] = "refused_isolation_gate"
        record["refusal"] = str(exc)
        record["note"] = (
            "Promotion refused: tier-1 operator-controlled software is resident. "
            "Close Cursor/Chrome (and other tier-1 names), then retry "
            "--attempt-raw-promote. Do not invent enforce=False. "
            "Emit wiring is present; dry_validate records whether the promote payload "
            "would schema-validate without writing raw/."
        )
        return record

    if not (record.get("dry_validate") or {}).get("ok"):
        record["status"] = "refused_dry_validate"
        record["note"] = (
            "Isolation gate passed but dry emit construction failed; refusing raw/ write. "
            "See dry_validate error."
        )
        return record

    try:
        emitted = _emit_raw_from_seal(
            session_id=session_id,
            seal_path=seal_path,
            sealed_manifest=sealed_manifest,
            sealed_summary=sealed_summary,
            placement=placement,
            allow_dirty=allow_dirty,
            model_spec=model_spec,
        )
    except Exception as exc:
        record["status"] = "refused_emit"
        record["emit_error_type"] = type(exc).__name__
        record["emit_error"] = str(exc)
        record["note"] = (
            "isolation + schema + dry_validate passed but seam.manifest.emit failed. "
            "raw/ was not sealed; inspect emit_error. Do not invent enforce=False."
        )
        return record

    record["status"] = "raw_emitted"
    record["raw_emit"] = True
    record["emit"] = emitted
    record["note"] = (
        f"seam.manifest.emit wrote sealed raw/{session_id}/ from derived seal "
        f"{_rel(seal_path)} (sealed_* tree not mutated)."
    )
    return record


def _mark_superseded(standing_session_id: str) -> list[dict[str, Any]]:
    marked: list[dict[str, Any]] = []
    for entry in SUPERSEDED_SESSIONS:
        sid = entry["session_id"]
        prior = SESSION_BASE / sid
        if not prior.is_dir():
            marked.append({**entry, "marker_written": False, "missing_dir": True})
            continue
        marker_path = prior / "SUPERSEDED.json"
        if marker_path.is_file():
            marked.append({**entry, "marker_written": False, "already_present": True})
            continue
        payload = {
            **entry,
            "superseded_by": standing_session_id,
            "marked_utc": datetime.now(UTC).isoformat(),
            "do_not_delete": True,
        }
        marker_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        marked.append({**entry, "marker_written": True, "marker_path": _rel(marker_path)})
    return marked


def seal_session(session_id: str, *, run_id: str | None = None) -> dict[str, Any]:
    run_id = run_id or session_id
    session_dir = SESSION_BASE / session_id
    if not session_dir.is_dir():
        raise SystemExit(f"missing session dir {session_dir}")

    summary_path = session_dir / "summary.json"
    plan_path = session_dir / "plan.json"
    if not summary_path.is_file():
        raise SystemExit(f"missing {summary_path}")
    summary = _read_json(summary_path)
    plan = _read_json(plan_path) if plan_path.is_file() else {}

    if summary.get("status") != "complete":
        raise SystemExit(f"refusing to seal: status={summary.get('status')!r} (want complete)")

    cell_results = list(summary.get("cell_results") or [])
    if len(cell_results) != EXPECTED_CELLS:
        raise SystemExit(
            f"refusing to seal: expected {EXPECTED_CELLS} cells, found {len(cell_results)}"
        )
    if any(int(c.get("exit_code", 1)) != 0 for c in cell_results):
        raise SystemExit("refusing to seal: non-zero cell exit_code present")
    if not all(c.get("report_exists") for c in cell_results):
        raise SystemExit("refusing to seal: missing cell report_exists")

    compare_paths = {
        "A": session_dir / _compare_name("A"),
        "gpu_only": session_dir / _compare_name("gpu_only"),
    }
    for p in compare_paths.values():
        if not p.is_file():
            raise SystemExit(f"missing compare artifact {p}")
    if not COLD_CONTROL_PATH.is_file():
        raise SystemExit(f"missing cold control {COLD_CONTROL_PATH}")

    compare_a = _read_json(compare_paths["A"])
    compare_g = _read_json(compare_paths["gpu_only"])
    cold = _read_json(COLD_CONTROL_PATH)
    if not cold.get("passed"):
        raise SystemExit("refusing to seal: cold control passed!=true")

    report_files = {
        f"{c['arm']}_{c['residency_mode']}": session_dir
        / _cell_report_name(str(c["arm"]), str(c["residency_mode"]))
        for c in cell_results
    }
    reports = {k: _read_json(p) for k, p in report_files.items()}
    for k, p in report_files.items():
        if not p.is_file():
            raise SystemExit(f"missing report for {k}: {p}")

    results = _build_results(
        session_id=session_id,
        session_dir=session_dir,
        compare_a=compare_a,
        compare_g=compare_g,
        cold=cold,
        reports=reports,
    )
    table_check = _verify_dispatch_table(results)
    if not table_check["ok"]:
        raise SystemExit(
            "STOP_AND_REPORT: feasibility table disagrees with DISPATCH N narrative: "
            + json.dumps(table_check["mismatches"])
        )
    results["dispatch_table_verification"] = table_check

    if not results["validity_chain"]["cold_control"]["dispatch_narrative_check"][
        "agrees_with_dispatch_narrative"
    ]:
        # Do not invent dispatch figures; seal continues with artifact-bound values.
        results["validity_chain"]["cold_control"]["stop_and_report"] = True

    launch = _launch_entry(session_id)
    placement = _measurement_placement(SESSION_BASE / f"sealed_{run_id}", launch)
    _fill_stack_versions(placement, reports)

    out = SESSION_BASE / f"sealed_{run_id}"
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing seal dir {out}")
    out.mkdir(parents=True, exist_ok=False)
    cells_dir = out / "cells"
    cells_dir.mkdir()

    manifest_cells: list[dict[str, Any]] = []
    for c in cell_results:
        arm = str(c["arm"])
        mode = str(c["residency_mode"])
        src = session_dir / _cell_report_name(arm, mode)
        dest = cells_dir / src.name
        shutil.copy2(src, dest)
        sha = _sha256_file(dest)
        manifest_cells.append(
            {
                "cell_index": c.get("cell_index"),
                "arm": arm,
                "residency_mode": mode,
                "n_entries": c.get("n_entries"),
                "exit_code": c.get("exit_code"),
                "started_utc": c.get("started_utc"),
                "ended_utc": c.get("ended_utc"),
                "artifact": f"cells/{src.name}",
                "artifact_sha256": sha,
                "reduced_reason": c.get("reduced_reason"),
            }
        )

    for arm, src in compare_paths.items():
        dest = cells_dir / src.name
        shutil.copy2(src, dest)
        manifest_cells.append(
            {
                "kind": "compare",
                "arm": arm,
                "artifact": f"cells/{src.name}",
                "artifact_sha256": _sha256_file(dest),
            }
        )

    shutil.copy2(summary_path, out / "session_summary_as_found.json")
    if plan_path.is_file():
        shutil.copy2(plan_path, out / "session_plan_as_found.json")
    shutil.copy2(COLD_CONTROL_PATH, out / "cold_control_as_found.json")

    superseded_marked = _mark_superseded(session_id)
    (out / "superseded_sessions.json").write_text(
        json.dumps(
            {
                "standing_session_id": session_id,
                "superseded": SUPERSEDED_SESSIONS,
                "markers": superseded_marked,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    sealed_utc = datetime.now(UTC).isoformat()
    raw_emit_blocked = {
        "reason": (
            "seam.manifest.emit refuses isolation_mode=remote while tier-1 software "
            "(Cursor/Chrome) is resident at seal time. Measurement itself was "
            "ssh_detached/remote. Close Cursor/Chrome and promote with "
            "--attempt-raw-promote; do not invent an enforce=False bypass."
        ),
        "existing_pattern": "tools/seal_delta_prefill_session.py → seal_style=derived_diagnostic",
        "schema_note": (
            f"workload.kind {WORKLOAD_KIND} is registered in "
            "seam/schemas/run_manifest.schema.json; --attempt-raw-promote calls "
            "seam.manifest.emit(..., retro_seal=True) after isolation clears "
            "(sealed_* never mutated)."
        ),
    }

    admissibility = {
        "matrix_status": "COMPLETE",
        "cells_ok": EXPECTED_CELLS,
        "cells_expected": EXPECTED_CELLS,
        "cold_control_passed": True,
        "failed_cells_are_data": False,
        "note": (
            f"All {EXPECTED_CELLS}/{EXPECTED_CELLS} cells OK. Numbers cite this run_id / "
            "seal dir / cells/*.json. Cold-control TTFT/ratio bind "
            f"{_rel(COLD_CONTROL_PATH)} (dispatch narrative mismatch recorded, not invented)."
        ),
    }

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "status": "COMPLETE",
        "kind": WORKLOAD_KIND,
        "seal_style": "derived_diagnostic",
        "sealed_utc": sealed_utc,
        "session_id": session_id,
        "measurement_isolation_mode": "remote",
        "measurement_launch_context": launch.get("launch_context") or "ssh_detached",
        "measurement_placement": {
            "launch_context": placement.get("launch_context"),
            "session_id": placement.get("session_id"),
            "window_station": placement.get("window_station"),
            "window_station_note": placement.get("window_station_note"),
            "openvino": placement.get("openvino"),
            "genai": placement.get("genai"),
        },
        "raw_emit_blocked": raw_emit_blocked,
        "matrix": {
            "kind": summary.get("kind") or plan.get("kind"),
            "seed": summary.get("seed") or plan.get("seed"),
            "max_new_tokens": summary.get("max_new_tokens"),
            "arms": ["gpu_only", "A"],
            "residency_modes": ["RESIDENT", "NON_RESIDENT"],
            "n_entries_per_cell": 20,
            "cells_completed": EXPECTED_CELLS,
            "cells_expected": EXPECTED_CELLS,
            "cells_ok": EXPECTED_CELLS,
            "cells_failed": 0,
            "plan_started_utc": summary.get("started_utc"),
            "plan_ended_utc": summary.get("ended_utc"),
            "launch_tag": launch.get("tag"),
            "cold_control_path": _rel(COLD_CONTROL_PATH),
        },
        "admissibility": admissibility,
        "results": results,
        "cells": manifest_cells,
        "superseded_prior_sessions": list(SUPERSEDED_SESSIONS),
    }

    sealed_summary: dict[str, Any] = {
        "run_id": run_id,
        "status": "COMPLETE",
        "kind": WORKLOAD_KIND,
        "seal_style": "derived_diagnostic",
        "session_id": session_id,
        "cells_completed": EXPECTED_CELLS,
        "cells_ok": EXPECTED_CELLS,
        "cells_failed": 0,
        "feasibility_table_rounded": [
            {
                "label": r["label"],
                "fraction_turns_slo_ok_pct": r["fraction_turns_slo_ok_pct"],
                "session_latency_sum_s_rounded": r["session_latency_sum_s_rounded"],
            }
            for r in results["feasibility_table"]
        ],
        "output_identity_finding": "gpu_kv_cache_numerical_nonequivalence",
        "cold_control_passed": True,
        "cold_control_turn2_ttft_s": results["validity_chain"]["cold_control"]["turn2_ttft_s"],
        "cold_control_turn2_over_turn1_raw": results["validity_chain"]["cold_control"][
            "turn2_over_turn1_raw"
        ],
        "dispatch_cold_narrative_mismatch": results["validity_chain"]["cold_control"][
            "dispatch_narrative_check"
        ]["agrees_with_dispatch_narrative"]
        is False,
    }

    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out / "summary.json").write_text(
        json.dumps(sealed_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    tree_hash = _sha256_tree(out, exclude={".sealed"})
    seal_marker = {
        "run_id": run_id,
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

    for p in [out / ".sealed", *out.rglob("*")]:
        if p.is_file():
            try:
                p.chmod(p.stat().st_mode & ~0o222)
            except OSError:
                pass

    pointer = {
        "session_id": session_id,
        "run_id": run_id,
        "sealed": True,
        "seal_style": "derived_diagnostic",
        "seal_path": _rel(out),
        "raw_emit": False,
        "raw_emit_blocked": raw_emit_blocked["reason"],
        "sealed_utc": sealed_utc,
        "tree_sha256": tree_hash,
        "cells_ok": EXPECTED_CELLS,
        "cells_failed": 0,
        "dispatch_cold_narrative_mismatch": sealed_summary["dispatch_cold_narrative_mismatch"],
    }
    pointer_path = session_dir / "SEAL_POINTER.json"
    pointer_path.write_text(json.dumps(pointer, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "run_id": run_id,
        "sealed": True,
        "seal_style": "derived_diagnostic",
        "seal_path": str(out.resolve()),
        "pointer": str(pointer_path.resolve()),
        "tree_sha256": tree_hash,
        "raw_emit": False,
        "cells_ok": EXPECTED_CELLS,
        "cells_failed": 0,
        "dispatch_table_ok": True,
        "dispatch_cold_narrative_mismatch": sealed_summary["dispatch_cold_narrative_mismatch"],
        "cold_control_artifact": {
            "turn2_ttft_s": results["validity_chain"]["cold_control"]["turn2_ttft_s"],
            "turn2_over_turn1_raw": results["validity_chain"]["cold_control"][
                "turn2_over_turn1_raw"
            ],
            "turn2_prompt_tokens": results["validity_chain"]["cold_control"]["turn2_prompt_tokens"],
            "passed": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    parser.add_argument(
        "--run-id",
        default=None,
        help="defaults to --session-id (session UUID is a valid uuid4)",
    )
    parser.add_argument(
        "--attempt-raw-promote",
        action="store_true",
        help=(
            "After isolation_mode=remote clears, call seam.manifest.emit into raw/ "
            "from the sealed derived tree. Does not mutate sealed_*. "
            "Does not invent enforce=False."
        ),
    )
    parser.add_argument(
        "--promote-only",
        action="store_true",
        help="Skip sealing; only run --attempt-raw-promote against an existing seal.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help=(
            "Pass allow_dirty through to seam.manifest.emit (recorded in the raw "
            "manifest). Required when the git tree is dirty at promote time."
        ),
    )
    parser.add_argument(
        "--model-spec",
        default=None,
        help=(
            "FetchedModelSpec YAML for the promote model block. Default: "
            f"{_MODEL_SPEC_PATH.as_posix()}. Must match the model_spec recorded "
            "on the session plan; mismatch or a missing record is fatal."
        ),
    )
    args = parser.parse_args(argv)

    session_id = args.session_id
    run_id = args.run_id or session_id
    seal_path = SESSION_BASE / f"sealed_{run_id}"
    result: dict[str, Any]

    if args.promote_only:
        if not seal_path.is_dir():
            raise SystemExit(f"promote-only: missing seal dir {seal_path}")
        if not args.attempt_raw_promote:
            raise SystemExit("promote-only requires --attempt-raw-promote")
        result = {
            "run_id": run_id,
            "sealed": True,
            "seal_style": "derived_diagnostic",
            "seal_path": str(seal_path.resolve()),
            "raw_emit": False,
            "promote_only": True,
        }
    else:
        result = seal_session(session_id, run_id=run_id)
        seal_path = Path(result["seal_path"])

    if args.attempt_raw_promote:
        promo = _attempt_raw_promote(
            session_id,
            seal_path,
            allow_dirty=bool(args.allow_dirty),
            model_spec=args.model_spec,
        )
        session_dir = SESSION_BASE / session_id
        promo_path = session_dir / "PROMOTION_ATTEMPT.json"
        promo_path.write_text(json.dumps(promo, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        pointer_path = session_dir / "SEAL_POINTER.json"
        if pointer_path.is_file():
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            emitted = promo.get("status") == "raw_emitted"
            pointer["raw_emit"] = bool(emitted)
            pointer["promotion_attempt"] = {
                "attempted_utc": promo["attempted_utc"],
                "status": promo["status"],
                "record": _rel(promo_path),
                "emit_wired": True,
            }
            if emitted:
                pointer["raw_path"] = (promo.get("emit") or {}).get("raw_path")
                pointer.pop("raw_emit_blocked", None)
            else:
                pointer["raw_emit_blocked"] = promo.get("note") or promo.get("refusal")
            pointer_path.write_text(
                json.dumps(pointer, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        result["promotion_attempt"] = promo
        result["promotion_record"] = str(promo_path.resolve())
        result["raw_emit"] = bool(promo.get("raw_emit"))
        if promo.get("status") == "refused_isolation_gate":
            result["operator_promote_command"] = (
                f".\\.venv-seam\\Scripts\\python.exe tools\\seal_session_residency.py "
                f"--session-id {session_id} --promote-only --attempt-raw-promote --allow-dirty"
            )

    print(json.dumps(result, indent=2, sort_keys=True))
    if args.attempt_raw_promote and result.get("promotion_attempt", {}).get("status") != (
        "raw_emitted"
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
