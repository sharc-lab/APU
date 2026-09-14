"""ceiling_a - max context under isolation for a selected arm pair (default A + A_prime).

Phase 3 of ΔN. Arms are chosen via ``--arms`` (resolved from configs/delta_n.yaml);
default is ``A,A_prime``. A ``B,B_prime`` selection measures the iGPU-resident pair the
same way. ``stop_if_ceiling_at_position_limit`` fires when *any* selected arm PASSes the
top rung (40000); the triggering ``arm_id`` is recorded on the seal.

Interleaving (configs/delta_n.yaml): at each rung every *ascending* arm is measured one
repeat per round, in a freshly randomized order derived from ``randomization_seed``,
before the ladder advances. The primary/replicate spread on the *ceiling* (not throughput)
is what sets ΔN's detectability threshold under ``materiality.detectable_multiple``.

This driver matches the revised config. ``seam.tools.delta_n`` is the sequential
pre-phase design and must not be used for this measurement.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from seam.backends.local_openvino import runtime_info
from seam.config import ResolvedConfig, resolve_config
from seam.errors import SeamError
from seam.isolation import harness_termination_record, resolve_isolation_mode
from seam.launch_context import resolve_launch_context
from seam.manifest import emit
from seam.model_provenance import load_local_spec, manifest_model_block
from seam.powerstate import capture_power_state, manifest_power_state
from seam.rawstore import open_run_dir
from seam.stdio_utf8 import configure_utf8_stdio
from seam.tools._winpower import assert_system_required
from seam.tools.acceptance_instrumentation import host_memory_snapshot
from seam.tools.delta_n import (
    _DELTA_N_PATH,
    _MEASUREMENT_PATH,
    _PLATFORM_PATH,
    _log,
    _utc,
    measured_repeat,
    memory_now,
    observe_processes,
    prompt_for,
)
from seam.tools.prompt_a_lifecycle import assert_acyclic, open_in_progress_run

# Sealed attrib kv_geometry_resolution (run_id 1fd81d2a-3bae-40b1-b2a6-b9c5a50586b1):
# cpu-p-specific: live KV on cpu-p is u8 → 2 * 36 * 8 * 128 * 1 = 73728 bytes/token.
# Precision-aware live-KV sizing for GPU / pinned arms: seam.ov_kv_precision.kv_bytes_per_token.
KV_BYTES_PER_TOKEN = 73728
KV_BYTES_CITING_RUN = "1fd81d2a-3bae-40b1-b2a6-b9c5a50586b1"

# Model position limit cited in configs/delta_n.yaml materiality / ceiling_a_gate narrative.
POSITION_LIMIT_TOKENS = 40960

PREREQUISITE_ACCEPTANCE_RUN_ID = "cb8e8d7c-74dd-4e72-ae69-8db7239a7b0f"

DEFAULT_ARMS_CSV = "A,A_prime"
DEFAULT_ARMS = ("A", "A_prime")
ARM_BENCHMARK = "ceiling_a_context_ceiling_v1"
VERDICT_BENCHMARK = "ceiling_a_verdict_v1"

MB = 1024.0 * 1024.0


def parse_arms(arms_csv: str) -> tuple[str, ...]:
    """Parse comma-separated arm ids; preserve order; reject empties and duplicates."""
    parts = [p.strip() for p in str(arms_csv).split(",") if p.strip()]
    if not parts:
        raise SeamError("arms list is empty")
    seen: set[str] = set()
    out: list[str] = []
    for arm_id in parts:
        if arm_id in seen:
            raise SeamError(f"duplicate arm_id in --arms: {arm_id}")
        seen.add(arm_id)
        out.append(arm_id)
    return tuple(out)


# ------------------------------------------------------------------------------------------
# Failure classification (verbatim, distinct)
# ------------------------------------------------------------------------------------------


def classify_failure(result: dict[str, Any]) -> dict[str, Any]:
    """Classify a non-pass child result. Inadmissible retries never reach here as fails."""
    outcome = str(result.get("outcome") or "")
    failure_mode = result.get("failure_mode")
    phase = str(result.get("phase_reached") or "")
    child = result.get("child") or {}
    exc = child.get("exception") or {}
    exc_type = str(exc.get("type") or "")
    exc_msg = str(exc.get("message") or "")
    combined = f"{failure_mode or ''} {exc_type} {exc_msg}".lower()

    if outcome == "pass":
        return {
            "class": "pass",
            "detail": None,
            "phase": phase,
            "failure_mode_raw": failure_mode,
        }

    if result.get("timed_out") or (failure_mode and str(failure_mode).startswith("timeout:")):
        return {
            "class": "timeout",
            "detail": failure_mode,
            "phase": phase,
            "failure_mode_raw": failure_mode,
            "timeout_s_configured": True,
        }

    oom_markers = (
        "out of memory",
        "oom",
        "std::bad_alloc",
        "bad_alloc",
        "cannot allocate",
        "not enough memory",
        "memoryerror",
        "failed to allocate",
        "allocation failure",
        "c0000005",  # AV often follows a refused allocation on Windows
        "0xc0000005",
    )
    if any(m in combined for m in oom_markers):
        cls = "oom_allocation_failure"
    elif outcome == "load_failure" or (
        not result.get("reached_resident") and "generat" not in phase and phase != "done"
    ):
        cls = "exception_at_load"
    elif "generat" in phase or phase == "done" or result.get("reached_resident"):
        cls = "exception_at_generate"
    else:
        cls = "exception_at_load"

    return {
        "class": cls,
        "detail": failure_mode,
        "phase": phase,
        "failure_mode_raw": failure_mode,
        "exception_type": exc_type or None,
        "exception_message": (exc_msg[:800] if exc_msg else None),
    }


def _termination_reason(
    *,
    failure_classification: dict[str, Any],
    n_tokens: int,
    ceiling_at_position_limit: bool = False,
) -> str:
    """Coarse per-rung/cell termination: pass | memory | timeout | position_limit | other."""
    if ceiling_at_position_limit or n_tokens >= POSITION_LIMIT_TOKENS:
        return "position_limit"
    cls = str(failure_classification.get("class") or "")
    if cls == "pass":
        return "pass"
    if cls == "timeout":
        return "timeout"
    if cls == "oom_allocation_failure":
        return "memory"
    return "other"


def _repeat_memory_instrumentation(
    *,
    n_tokens: int,
    result: dict[str, Any],
    host_at_repeat_start: dict[str, Any],
) -> dict[str, Any]:
    """Per-repeat fields required by the ceiling_a dispatch (not per-arm aggregates)."""
    child = result.get("child") or {}
    generation = child.get("generation") or {}
    wslock = child.get("working_set_lock") or {}

    peak_ws = generation.get("peak_rss_bytes")
    peak_commit = generation.get("peak_commit_bytes")
    free_start_mb = generation.get("free_physical_mb_start")
    if free_start_mb is None:
        free_start_mb = generation.get("free_memory_mb_start")
    free_peak_mb = generation.get("free_physical_mb_at_peak")

    free_start_bytes = (
        int(float(free_start_mb) * MB)
        if free_start_mb is not None
        else host_at_repeat_start.get("free_physical_bytes")
    )
    free_peak_bytes = int(float(free_peak_mb) * MB) if free_peak_mb is not None else None

    granted = wslock.get("granted")
    win32_error = wslock.get("last_error")
    if win32_error is None:
        win32_error = wslock.get("set_last_error") or wslock.get("create_last_error")

    prefill_s = generation.get("prefill_s")
    if prefill_s is None and generation.get("ttft_ns") is not None:
        prefill_s = float(generation["ttft_ns"]) / 1e9
    decode_tok_s = generation.get("decode_tok_s")
    if decode_tok_s is None:
        decode_tok_s = generation.get("r_decode_tok_s")

    return {
        "free_physical_bytes_at_start": free_start_bytes,
        "free_physical_bytes_at_peak": free_peak_bytes,
        "free_physical_at_peak": free_peak_bytes,
        "peak_working_set_bytes": int(peak_ws) if peak_ws is not None else None,
        "peak_ws_bytes": int(peak_ws) if peak_ws is not None else None,
        "peak_commit_bytes": int(peak_commit) if peak_commit is not None else None,
        "prefill_s": prefill_s,
        "decode_tok_s": decode_tok_s,
        "kv_bytes_expected": int(n_tokens) * KV_BYTES_PER_TOKEN,
        "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
        "kv_bytes_citing_run_id": KV_BYTES_CITING_RUN,
        "working_set_lock": {
            "requested": wslock.get("requested", wslock.get("attempted")),
            "granted": granted,
            "denied": (False if granted is None else (not bool(granted))),
            "win32_error": win32_error,
            "last_error_meaning": wslock.get("last_error_meaning"),
            "mode": wslock.get("mode"),
            "minimum_bytes": wslock.get("minimum_bytes"),
            "maximum_bytes": wslock.get("maximum_bytes"),
            "raw": wslock,
        },
        "host_at_repeat_start": host_at_repeat_start,
    }


# ------------------------------------------------------------------------------------------
# Checkpoint / resume
# ------------------------------------------------------------------------------------------


def _session_dir(root: Path, session_id: str) -> Path:
    return root / "derived" / "ceiling_a" / session_id


def _checkpoint_path(session_dir: Path) -> Path:
    return session_dir / "checkpoint.json"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def load_checkpoint(session_dir: Path) -> dict[str, Any] | None:
    path = _checkpoint_path(session_dir)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_checkpoint(session_dir: Path, state: dict[str, Any]) -> None:
    state["checkpoint_utc"] = _utc()
    _atomic_write_json(_checkpoint_path(session_dir), state)
    # Heartbeat for operators polling over SSH without reading the full checkpoint.
    hb = {
        "session_id": state.get("session_id"),
        "phase": state.get("phase"),
        "stage": state.get("current_stage"),
        "n_tokens": state.get("current_n_tokens"),
        "round": state.get("current_round"),
        "arms_selected": list(state.get("arms_selected") or []),
        "arm_state": {
            arm_id: {
                "highest_pass": arm.get("highest_pass"),
                "lowest_nonpass": arm.get("lowest_nonpass"),
                "ascending": arm.get("ascending"),
            }
            for arm_id, arm in (state.get("arm_state") or {}).items()
        },
        "n_cells_completed": len(state.get("completed_cells") or []),
        "early_exit_fired": state.get("early_exit_fired"),
        "heartbeat_utc": _utc(),
    }
    _atomic_write_json(session_dir / "heartbeat.json", hb)


def _cell_key(stage: str, n_tokens: int, round_i: int, arm_id: str) -> str:
    return f"{stage}|n{n_tokens}|r{round_i}|{arm_id}"


def _arm_order_for(
    *,
    seed: int,
    stage: str,
    n_tokens: int,
    round_i: int,
    active: list[str],
) -> list[str]:
    """Deterministic shuffle so resume re-derives the same order for a completed round."""
    material = f"{int(seed)}|{stage}|{int(n_tokens)}|{int(round_i)}|{','.join(active)}"
    rng = random.Random(material)
    order = list(active)
    rng.shuffle(order)
    return order


# ------------------------------------------------------------------------------------------
# Rung / ladder
# ------------------------------------------------------------------------------------------


def _child_spec_for_arm(
    *,
    cfg: dict[str, Any],
    arm: dict[str, Any],
    model_dir: str,
    prompt: dict[str, Any],
    p_cpus: list[int],
) -> dict[str, Any]:
    cpu_properties = dict(cfg["openvino"]["cpu_properties"])
    arm_properties = dict(arm.get("properties") or {})
    affinity = list(p_cpus) if cfg["openvino"]["affinity"] == "p_cpus" else None
    load_sequence: list[dict[str, Any]] = []
    for device in arm["load_sequence"]:
        props: dict[str, Any] = {}
        if device == "CPU":
            props.update(cpu_properties)
        props.update(arm_properties)
        load_sequence.append({"device": device, "properties": props})
    wslock = {
        "mode": str(cfg["working_set_lock"]["mode"]),
        "minimum_bytes": int(cfg["working_set_lock"]["minimum_bytes"]),
        "maximum_bytes": int(cfg["working_set_lock"]["maximum_bytes"]),
    }
    return {
        "model_dir": model_dir,
        "load_sequence": load_sequence,
        "generate_device": arm["generate_device"],
        "arm_id": arm.get("id"),
        "arm_properties": arm_properties,
        "prompt_path": prompt["path"],
        "max_new_tokens": int(cfg["ladder"]["max_new_tokens"]),
        "affinity_cpus": affinity,
        "rss_interval_s": 0.05,
        "wslock": wslock,
    }


def _rung_verdict(outcomes: list[str], *, repeats: int, pass_requires_all: bool) -> str:
    n_pass = sum(1 for o in outcomes if o == "pass")
    if pass_requires_all:
        if n_pass == repeats:
            return "PASS"
        if n_pass == 0:
            return "FAIL"
        return "MIXED"
    if n_pass == repeats:
        return "PASS"
    if n_pass == 0:
        return "FAIL"
    return "MIXED"


def _summarize_rung(
    *,
    arm_id: str,
    n_tokens: int,
    stage: str,
    records: list[dict[str, Any]],
    realized_orders: list[list[str]],
    repeats: int,
    pass_requires_all: bool,
) -> dict[str, Any]:
    outcomes = [r["result"]["outcome"] for r in records]
    verdict = _rung_verdict(outcomes, repeats=repeats, pass_requires_all=pass_requires_all)
    failure_classes = [
        r["failure_classification"]["class"] for r in records if r["result"]["outcome"] != "pass"
    ]
    return {
        "arm_id": arm_id,
        "n_tokens": n_tokens,
        "stage": stage,
        "repeats": repeats,
        "n_pass": sum(1 for o in outcomes if o == "pass"),
        "verdict": verdict,
        "pass_requires_all_repeats": pass_requires_all,
        "failure_modes": sorted(
            {r["result"]["failure_mode"] for r in records if r["result"]["failure_mode"]}
        ),
        "failure_classes": sorted(set(failure_classes)),
        "repeat_spread": f"{sum(1 for o in outcomes if o == 'pass')}/{repeats} passed",
        "realized_arm_orders": realized_orders,
        "per_repeat_memory": [r.get("memory_instrumentation") for r in records],
        "generation_wall_s": [
            (r["result"]["child"].get("generation") or {}).get("wall_s") for r in records
        ],
        "paging_gate_verdicts": [
            (r.get("envelope") or {}).get("paging_gate_verdict") for r in records
        ],
    }


def run_interleaved_ladder(
    *,
    root: Path,
    cfg: dict[str, Any],
    p_cpus: list[int],
    session_dir: Path,
    state: dict[str, Any],
    arms_by_id: dict[str, dict[str, Any]],
    model_dir: str,
    tokenizer: Any,
    prompt_cache: dict[int, dict[str, Any]],
    run_dirs: dict[str, Any],
    work_dir: Path,
    active_arms: tuple[str, ...],
) -> dict[str, Any]:
    """Ascending interleaved ladder + interleaved bisection for the selected arms."""
    ladder = cfg["ladder"]
    unit = str(ladder["filler_unit"])
    repeats = int(ladder["repeats"])
    pass_requires_all = bool(ladder.get("pass_requires_all_repeats", True))
    resolution = int(ladder["bisect_resolution_tokens"])
    round_to = int(ladder["bisect_round_to"])
    seed = int(cfg["randomization_seed"])
    rungs = [int(v) for v in ladder["rungs"]]
    top_rung = max(rungs)
    arms_selected = tuple(active_arms)

    completed_cells: set[str] = set(state.get("completed_cells") or [])
    arm_state = state["arm_state"]
    rung_records: dict[str, list[dict[str, Any]]] = state.setdefault("rung_records", {})
    realized_order_log: list[dict[str, Any]] = state.setdefault("realized_order_log", [])

    def persist() -> None:
        state["completed_cells"] = sorted(completed_cells)
        save_checkpoint(session_dir, state)

    def measure_cell(
        *,
        stage: str,
        n_tokens: int,
        round_i: int,
        arm_id: str,
        arm_order: list[str],
        prompt: dict[str, Any],
    ) -> dict[str, Any] | None:
        key = _cell_key(stage, n_tokens, round_i, arm_id)
        if key in completed_cells:
            # Reload the sealed-on-disk record for rung summary reconstruction.
            return (state.get("cell_payloads") or {}).get(key)

        host_at_start = host_memory_snapshot()
        child_spec = _child_spec_for_arm(
            cfg=cfg,
            arm=arms_by_id[arm_id],
            model_dir=model_dir,
            prompt=prompt,
            p_cpus=p_cpus,
        )
        tag = f"{arm_id}.n{n_tokens}.{stage}.r{round_i}"
        record = measured_repeat(
            root=root,
            cfg=cfg,
            p_cpus=p_cpus,
            work_dir=work_dir,
            child_spec=child_spec,
            label=f"ceiling-a/{arm_id}/{n_tokens}/{round_i}",
            tag=tag,
        )
        record["arm_id"] = arm_id
        record["n_tokens"] = n_tokens
        record["repeat_index"] = round_i
        record["stage"] = stage
        record["prompt"] = prompt
        record["realized_arm_order"] = list(arm_order)
        record["controls"] = observe_processes(cfg["controls"]["observe_processes"])
        record["failure_classification"] = classify_failure(record["result"])
        record["memory_instrumentation"] = _repeat_memory_instrumentation(
            n_tokens=n_tokens,
            result=record["result"],
            host_at_repeat_start=host_at_start,
        )
        record["termination_reason"] = _termination_reason(
            failure_classification=record["failure_classification"],
            n_tokens=n_tokens,
        )
        run_dirs[arm_id].append_ndjson("repeats.ndjson", record)

        mem = record["memory_instrumentation"]
        state.setdefault("cell_payloads", {})[key] = {
            "key": key,
            "arm_id": arm_id,
            "n_tokens": n_tokens,
            "repeat_index": round_i,
            "stage": stage,
            "realized_arm_order": list(arm_order),
            "prefill_s": mem.get("prefill_s"),
            "decode_tok_s": mem.get("decode_tok_s"),
            "peak_ws_bytes": mem.get("peak_ws_bytes"),
            "free_physical_at_peak": mem.get("free_physical_at_peak"),
            "kv_bytes_expected": mem.get("kv_bytes_expected"),
            "termination_reason": record["termination_reason"],
            "result": {
                "outcome": record["result"]["outcome"],
                "failure_mode": record["result"]["failure_mode"],
                "phase_reached": record["result"]["phase_reached"],
                "timed_out": record["result"].get("timed_out"),
                "reached_resident": record["result"].get("reached_resident"),
                "child": {
                    "generation": (record["result"].get("child") or {}).get("generation"),
                    "working_set_lock": (record["result"].get("child") or {}).get(
                        "working_set_lock"
                    ),
                    "standing_reservation": (record["result"].get("child") or {}).get(
                        "standing_reservation"
                    ),
                    "exception": (record["result"].get("child") or {}).get("exception"),
                    "completed": (record["result"].get("child") or {}).get("completed"),
                },
            },
            "failure_classification": record["failure_classification"],
            "memory_instrumentation": record["memory_instrumentation"],
            "envelope": record.get("envelope"),
            "started_utc": record.get("started_utc"),
            "ended_utc": record.get("ended_utc"),
            "window_utc": record.get("window_utc"),
            "attempt_used": record.get("attempt_used"),
            "attempts": record.get("attempts"),
            "admissible": record.get("admissible"),
        }
        completed_cells.add(key)
        state["current_stage"] = stage
        state["current_n_tokens"] = n_tokens
        state["current_round"] = round_i
        state["last_cell_key"] = key
        persist()
        _log(
            "ceiling_a.cell",
            arm=arm_id,
            n=n_tokens,
            round=round_i,
            stage=stage,
            outcome=record["result"]["outcome"],
            failure_class=record["failure_classification"]["class"],
            termination_reason=record["termination_reason"],
            arm_order=arm_order,
            peak_ws=record["memory_instrumentation"].get("peak_working_set_bytes"),
            free_peak=record["memory_instrumentation"].get("free_physical_bytes_at_peak"),
            kv_expected=record["memory_instrumentation"].get("kv_bytes_expected"),
            prefill_s=record["memory_instrumentation"].get("prefill_s"),
            decode_tok_s=record["memory_instrumentation"].get("decode_tok_s"),
            wslock_granted=record["memory_instrumentation"]["working_set_lock"].get("granted"),
        )
        return state["cell_payloads"][key]

    def run_rung_interleaved(stage: str, n_tokens: int, active: list[str]) -> dict[str, dict]:
        prompt = prompt_for(
            root=root, tokenizer=tokenizer, n_tokens=n_tokens, unit=unit, cache=prompt_cache
        )
        per_arm_cells: dict[str, list[dict[str, Any]]] = {a: [] for a in active}
        orders_this_rung: list[list[str]] = []

        for round_i in range(repeats):
            order = _arm_order_for(
                seed=seed, stage=stage, n_tokens=n_tokens, round_i=round_i, active=active
            )
            orders_this_rung.append(order)
            realized_order_log.append(
                {
                    "stage": stage,
                    "n_tokens": n_tokens,
                    "round": round_i,
                    "arm_order": list(order),
                    "utc": _utc(),
                }
            )
            persist()
            _log(
                "ceiling_a.round_order",
                stage=stage,
                n=n_tokens,
                round=round_i,
                arm_order=order,
            )
            for arm_id in order:
                cell = measure_cell(
                    stage=stage,
                    n_tokens=n_tokens,
                    round_i=round_i,
                    arm_id=arm_id,
                    arm_order=order,
                    prompt=prompt,
                )
                assert cell is not None
                per_arm_cells[arm_id].append(cell)

        summaries: dict[str, dict[str, Any]] = {}
        for arm_id in active:
            # Rebuild measured_repeat-shaped records for summarize
            records_for_summary = []
            for cell in per_arm_cells[arm_id]:
                records_for_summary.append(
                    {
                        "result": cell["result"],
                        "failure_classification": cell["failure_classification"],
                        "memory_instrumentation": cell["memory_instrumentation"],
                        "envelope": cell.get("envelope") or {},
                    }
                )
            summary = _summarize_rung(
                arm_id=arm_id,
                n_tokens=n_tokens,
                stage=stage,
                records=records_for_summary,
                realized_orders=orders_this_rung,
                repeats=repeats,
                pass_requires_all=pass_requires_all,
            )
            summaries[arm_id] = summary
            rung_key = f"{stage}|n{n_tokens}|{arm_id}"
            rung_records[rung_key] = summary
            run_dirs[arm_id].append_ndjson("rungs.ndjson", summary)
            _log(
                "ceiling_a.rung",
                arm=arm_id,
                n=n_tokens,
                verdict=summary["verdict"],
                n_pass=summary["n_pass"],
                stage=stage,
                failure_classes=summary["failure_classes"],
            )

        state["last_completed_rung"] = {"stage": stage, "n_tokens": n_tokens, "arms": list(active)}
        persist()
        return summaries

    # ----- ascending ladder -----
    state["phase"] = "ladder"
    state["current_stage"] = "ladder"
    persist()

    for n_tokens in rungs:
        active = [a for a in arms_selected if arm_state[a]["ascending"]]
        if not active:
            break
        summaries = run_rung_interleaved("ladder", n_tokens, active)
        for arm_id, summary in summaries.items():
            if summary["verdict"] == "PASS":
                arm_state[arm_id]["highest_pass"] = n_tokens
            else:
                arm_state[arm_id]["lowest_nonpass"] = n_tokens
                arm_state[arm_id]["ascending"] = False
                arm_state[arm_id]["boundary_verdict"] = summary["verdict"]
                arm_state[arm_id]["boundary_repeat_spread"] = summary["repeat_spread"]
                arm_state[arm_id]["boundary_failure_classes"] = summary["failure_classes"]
                arm_state[arm_id]["boundary_failure_modes"] = summary["failure_modes"]
        persist()

        # Early exit (ceiling_a_gate): fires when ANY selected arm PASSes the top rung.
        # Triggering arm_id is the first listed among those that PASS (so -Arms gpu_only
        # early-exits when gpu_only PASSes 40000; -Arms A,A_prime still triggers on A).
        if n_tokens == top_rung and bool(
            cfg["ceiling_a_gate"]["stop_if_ceiling_at_position_limit"]
        ):
            trigger_arm = next(
                (a for a in arms_selected if (summaries.get(a) or {}).get("verdict") == "PASS"),
                None,
            )
            if trigger_arm is not None:
                for arm_id, summary in summaries.items():
                    if summary["verdict"] == "PASS":
                        arm_state[arm_id]["highest_pass"] = top_rung
                        arm_state[arm_id]["lowest_nonpass"] = POSITION_LIMIT_TOKENS
                        arm_state[arm_id]["ascending"] = False
                        arm_state[arm_id]["ceiling_at_position_limit"] = True
                        arm_state[arm_id]["early_exit_bracket"] = [
                            top_rung,
                            POSITION_LIMIT_TOKENS,
                        ]
                state["early_exit_fired"] = True
                state["early_exit_gate_arm_id"] = trigger_arm
                state["early_exit_reason"] = (
                    f"arm {trigger_arm} PASSed top rung {top_rung}; "
                    f"ceiling({trigger_arm}) lies in [{top_rung}, {POSITION_LIMIT_TOKENS}]; "
                    "position limit binds before memory; ΔN is not the measurement to take. "
                    f"Gate trigger arm_id={trigger_arm}; arms_selected="
                    f"{list(arms_selected)}."
                )
                _log(
                    "ceiling_a.early_exit",
                    top_rung=top_rung,
                    position_limit=POSITION_LIMIT_TOKENS,
                    early_exit_gate_arm_id=trigger_arm,
                    trigger_arm_verdict="PASS",
                    arms_selected=list(arms_selected),
                    reason=state["early_exit_reason"],
                )
                persist()
                break

    # ----- bisection (skipped when early exit placed the bracket at the position limit) -----
    state["phase"] = "bisect"
    state["current_stage"] = "bisect"
    persist()

    if not state.get("early_exit_fired"):
        # Bisect each arm that has a finite bracket, interleaved at every midpoint.
        while True:
            candidates: list[tuple[str, int, int, int]] = []
            for arm_id in arms_selected:
                low = int(arm_state[arm_id]["highest_pass"])
                high = arm_state[arm_id]["lowest_nonpass"]
                if high is None:
                    continue
                high = int(high)
                if high - low <= resolution:
                    continue
                midpoint = ((low + high) // 2 // round_to) * round_to
                if midpoint <= low or midpoint >= high:
                    continue
                candidates.append((arm_id, low, high, midpoint))
            if not candidates:
                break

            # Group arms that share the same midpoint this step so one interleaved rung covers them.
            by_mid: dict[int, list[str]] = {}
            for arm_id, _low, _high, mid in candidates:
                by_mid.setdefault(mid, []).append(arm_id)

            for midpoint, arm_ids in sorted(by_mid.items()):
                summaries = run_rung_interleaved("bisect", midpoint, arm_ids)
                for arm_id, summary in summaries.items():
                    if summary["verdict"] == "PASS":
                        arm_state[arm_id]["highest_pass"] = midpoint
                    else:
                        arm_state[arm_id]["lowest_nonpass"] = midpoint
                        arm_state[arm_id]["boundary_verdict"] = summary["verdict"]
                        arm_state[arm_id]["boundary_repeat_spread"] = summary["repeat_spread"]
                        arm_state[arm_id]["boundary_failure_classes"] = summary["failure_classes"]
                        arm_state[arm_id]["boundary_failure_modes"] = summary["failure_modes"]
                    state.setdefault("bisect_steps", []).append(
                        {
                            "arm_id": arm_id,
                            "n_tokens": midpoint,
                            "verdict": summary["verdict"],
                            "utc": _utc(),
                        }
                    )
                persist()

    # Arms that never failed: ceiling is a lower bound at highest_pass.
    for arm_id in arms_selected:
        if arm_state[arm_id]["lowest_nonpass"] is None:
            arm_state[arm_id]["ceiling_is_lower_bound"] = True
        else:
            arm_state[arm_id]["ceiling_is_lower_bound"] = bool(
                arm_state[arm_id].get("ceiling_at_position_limit")
            )

    state["phase"] = "ladder_complete"
    persist()
    return state


# ------------------------------------------------------------------------------------------
# Seal
# ------------------------------------------------------------------------------------------


def _arm_summary_from_state(
    *,
    arm_id: str,
    label: str,
    arm_cfg: dict[str, Any],
    state: dict[str, Any],
    prompt_cache: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    st = state["arm_state"][arm_id]
    highest = int(st["highest_pass"])
    lowest = st["lowest_nonpass"]
    rungs = [
        rec for key, rec in (state.get("rung_records") or {}).items() if rec.get("arm_id") == arm_id
    ]
    rungs.sort(key=lambda r: (0 if r["stage"] == "ladder" else 1, r["n_tokens"]))
    return {
        "arm_id": arm_id,
        "label": label,
        "load_sequence": arm_cfg["load_sequence"],
        "generate_device": arm_cfg["generate_device"],
        "ceiling_tokens": highest,
        "ceiling_bracket": [highest, lowest],
        "ceiling_bracket_width": None if lowest is None else int(lowest) - highest,
        "ceiling_is_lower_bound": bool(st.get("ceiling_is_lower_bound")),
        "ceiling_at_position_limit": bool(st.get("ceiling_at_position_limit")),
        "boundary_verdict": st.get("boundary_verdict"),
        "boundary_repeat_spread": st.get("boundary_repeat_spread"),
        "boundary_failure_classes": st.get("boundary_failure_classes") or [],
        "boundary_failure_modes": st.get("boundary_failure_modes") or [],
        "rungs": rungs,
        "bisect_steps": [
            step for step in (state.get("bisect_steps") or []) if step.get("arm_id") == arm_id
        ],
        "prompts_used": {
            str(n): prompt_cache[n]
            for n in sorted(prompt_cache)
            if any(r.get("n_tokens") == n for r in rungs)
        },
    }


def build_ceiling_verdict(
    arm_summaries: dict[str, dict[str, Any]],
    cfg: dict[str, Any],
    state: dict[str, Any],
    *,
    all_arm_ids: list[str],
) -> dict[str, Any]:
    arms_selected = tuple(state.get("arms_selected") or list(arm_summaries))
    ceiling_tokens = {
        arm_id: int(arm_summaries[arm_id]["ceiling_tokens"]) for arm_id in arms_selected
    }
    ceiling_bracket = {arm_id: arm_summaries[arm_id]["ceiling_bracket"] for arm_id in arms_selected}
    if len(arms_selected) >= 2:
        pair_spread = abs(ceiling_tokens[arms_selected[0]] - ceiling_tokens[arms_selected[1]])
        spread_def = (
            f"|ceiling({arms_selected[0]}) - ceiling({arms_selected[1]})|; "
            "the noise floor for later ΔN under materiality.detectable_multiple"
        )
    else:
        pair_spread = 0
        spread_def = "single-arm selection; no primary/replicate spread"
    primary = arms_selected[0]
    early = bool(state.get("early_exit_fired"))
    return {
        "phase": "ceiling_a",
        "arms_selected": list(arms_selected),
        "arms_measured": list(arms_selected),
        "arms_not_measured": [a for a in all_arm_ids if a not in arms_selected],
        "ceiling_tokens": ceiling_tokens,
        "ceiling_bracket": ceiling_bracket,
        "aa_spread_tokens": pair_spread,
        "aa_spread_definition": spread_def,
        "detectable_multiple": float(cfg["materiality"]["detectable_multiple"]),
        "delta_n_detectability_threshold_tokens": (
            float(cfg["materiality"]["detectable_multiple"]) * pair_spread
        ),
        "early_exit_fired": early,
        "early_exit_reason": state.get("early_exit_reason"),
        "early_exit_gate_arm_id": state.get("early_exit_gate_arm_id"),
        "early_exit_gate_note": (
            "stop_if_ceiling_at_position_limit fires when any selected arm PASSes "
            "the top rung; early_exit_gate_arm_id records which arm triggered"
        ),
        "position_limit_tokens": POSITION_LIMIT_TOKENS,
        "stop_if_ceiling_at_position_limit": bool(
            cfg["ceiling_a_gate"]["stop_if_ceiling_at_position_limit"]
        ),
        "bisection_path": list(state.get("bisect_steps") or []),
        "realized_order_log": list(state.get("realized_order_log") or []),
        "verdict": (
            "early_exit_position_limit"
            if early
            else (
                "ceiling_bracketed"
                if arm_summaries[primary]["ceiling_bracket"][1] is not None
                else "ceiling_lower_bound_only"
            )
        ),
        "next_phase": (
            "stop - do not run arm B / three-arm ΔN"
            if early
            else "stop - ceiling_a complete; three-arm ΔN is a separate dispatch"
        ),
        "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
        "kv_bytes_citing_run_id": KV_BYTES_CITING_RUN,
    }


def _seal_arm(
    *,
    root: Path,
    resolved: ResolvedConfig,
    cfg: dict[str, Any],
    spec: dict[str, Any],
    model_spec_path: Path,
    arm_id: str,
    arm_summary: dict[str, Any],
    run_dir: Any,
    run_id: str,
    state: dict[str, Any],
    allow_dirty: bool,
    launch_context: str,
    placement: dict[str, Any],
    power_request: dict[str, Any],
    isolation_discipline: dict[str, Any],
) -> str:
    arms_selected = list(state.get("arms_selected") or DEFAULT_ARMS)
    power = capture_power_state()
    summary = {
        "experiment_id": "ceiling_a",
        "benchmark": ARM_BENCHMARK,
        "phase": "ceiling_a",
        "run_id": run_id,
        "session_id_ceiling_a": state["session_id"],
        "prerequisite_acceptance_run_id": PREREQUISITE_ACCEPTANCE_RUN_ID,
        "arms_selected": arms_selected,
        "arm": arm_summary,
        "ceiling_tokens": arm_summary["ceiling_tokens"],
        "ceiling_bracket": arm_summary["ceiling_bracket"],
        "early_exit_fired": bool(state.get("early_exit_fired")),
        "early_exit_gate_arm_id": state.get("early_exit_gate_arm_id"),
        "resumes": list(state.get("resumes") or []),
        "isolation_discipline": isolation_discipline,
        "power_request_orchestrator": power_request,
        "design": {
            "ladder": cfg["ladder"],
            "working_set_lock": cfg["working_set_lock"],
            "generation": cfg["generation"],
            "admissibility": cfg["admissibility"],
            "paging": cfg["paging"],
            "randomization_seed": cfg["randomization_seed"],
            "openvino": cfg["openvino"],
        },
        "instrumentation_note": (
            "Per-repeat free_physical / peak_working_set / peak_commit / kv_bytes_expected / "
            "working_set_lock are on repeats.ndjson (memory_instrumentation). Peak working set "
            "reproduces across acceptance pairs; free_physical_at_peak does not - a ceiling is "
            "defined by the quantity that does not reproduce, recorded per repeat."
        ),
        "scope": {
            "arms_selected": arms_selected,
            "arms_in_phase": arms_selected,
            "arm_B_excluded": "B" not in arms_selected,
            "three_arm_delta_n": "not in this phase",
            "early_exit_gate": (
                "any selected arm PASS at top rung (stop_if_ceiling_at_position_limit); "
                "trigger recorded in early_exit_gate_arm_id"
            ),
        },
        "power_start": asdict(power),
    }
    assert_acyclic(summary, label=f"ceiling_a arm {arm_id} run_id={run_id}")
    run_dir.write_json("ceiling_a_arm_summary.json", summary)
    handle = emit(
        config=resolved,
        target="cpu-p",
        workload={
            "kind": "microbench",
            "benchmark": ARM_BENCHMARK,
            "task_ids": [],
            "seed": int(cfg["randomization_seed"]),
            "n_repeats": int(cfg["ladder"]["repeats"]),
            "concurrency": 1,
        },
        condition_label=(
            f"ceiling_a|{arm_id}|cpu-p|seed={cfg['randomization_seed']}|"
            f"early_exit={bool(state.get('early_exit_fired'))}"
        ),
        repo_root=root,
        run_id=run_id,
        existing_run_dir=run_dir,
        allow_dirty=allow_dirty,
        summary=summary,
        model=manifest_model_block(
            spec=spec, spec_path=model_spec_path, reasoning_mode="thinking_off"
        ),
        drivers=asdict(runtime_info()),
        power_state=manifest_power_state(power, background_quiesced=False),
        thermal={"regime": "confound", "excluded": False},
        self_check="pass",
        launch_context=launch_context,
        session_id=placement.get("session_id"),
        window_station=placement.get("window_station"),
        require_launch_context=True,
    )
    return handle.run_id


def _seal_verdict(
    *,
    root: Path,
    resolved: ResolvedConfig,
    cfg: dict[str, Any],
    arm_run_ids: dict[str, str],
    arm_summaries: dict[str, dict[str, Any]],
    verdict: dict[str, Any],
    run_id: str,
    state: dict[str, Any],
    allow_dirty: bool,
    launch_context: str,
    placement: dict[str, Any],
    power_request: dict[str, Any],
    isolation_discipline: dict[str, Any],
) -> str:
    arms_selected = list(state.get("arms_selected") or list(arm_run_ids))
    power = capture_power_state()
    summary = {
        "experiment_id": "ceiling_a",
        "benchmark": VERDICT_BENCHMARK,
        "phase": "ceiling_a",
        "run_id": run_id,
        "session_id_ceiling_a": state["session_id"],
        "prerequisite_acceptance_run_id": PREREQUISITE_ACCEPTANCE_RUN_ID,
        "arms_selected": arms_selected,
        "purpose": (
            "Derived ceiling_a verdict: ceilings for the selected arms, primary/replicate "
            "spread on the ceiling, bracket, bisection path, and whether the arm-A-specific "
            "position-limit early exit fired."
        ),
        "source_arm_run_ids": dict(arm_run_ids),
        "arms": arm_summaries,
        "verdict": verdict,
        "resumes": list(state.get("resumes") or []),
        "isolation_discipline": isolation_discipline,
        "power_request_orchestrator": power_request,
        "power_start": asdict(power),
    }
    assert_acyclic(summary, label=f"ceiling_a verdict run_id={run_id}")
    task_ids = [arm_run_ids[a] for a in arms_selected]
    handle = emit(
        config=resolved,
        target="cpu-p",
        workload={
            "kind": "aa",
            "benchmark": VERDICT_BENCHMARK,
            "task_ids": task_ids,
            "seed": int(cfg["randomization_seed"]),
            "n_repeats": len(arms_selected),
            "concurrency": 1,
        },
        condition_label=(
            "ceiling_a_verdict|"
            + "|".join(f"{a}={arm_run_ids[a]}" for a in arms_selected)
            + f"|early_exit={verdict['early_exit_fired']}"
        ),
        repo_root=root,
        run_id=run_id,
        allow_dirty=allow_dirty,
        summary=summary,
        drivers=asdict(runtime_info()),
        power_state=manifest_power_state(power, background_quiesced=False),
        thermal={"regime": "confound", "excluded": False},
        self_check="pass",
        launch_context=launch_context,
        session_id=placement.get("session_id"),
        window_station=placement.get("window_station"),
        require_launch_context=True,
    )
    return handle.run_id


# ------------------------------------------------------------------------------------------
# Orchestrate
# ------------------------------------------------------------------------------------------


def _wait_pre_run_settle(configured_s: float) -> dict[str, Any]:
    started_utc = _utc()
    t0 = time.monotonic()
    _log("ceiling_a.pre_run_settle_start", configured_s=configured_s, started_utc=started_utc)
    time.sleep(configured_s)
    actual_wait_s = time.monotonic() - t0
    ended_utc = _utc()
    record = {
        "configured_s": configured_s,
        "actual_wait_s": actual_wait_s,
        "started_utc": started_utc,
        "ended_utc": ended_utc,
        "applied_on_this_path": True,
        "distinct_from": "recovery.settle_s",
    }
    _log(
        "ceiling_a.pre_run_settle_end",
        configured_s=configured_s,
        actual_wait_s=actual_wait_s,
        ended_utc=ended_utc,
    )
    return record


def _new_arm_state() -> dict[str, Any]:
    return {
        "highest_pass": 0,
        "lowest_nonpass": None,
        "ascending": True,
        "ceiling_at_position_limit": False,
        "ceiling_is_lower_bound": False,
        "boundary_verdict": None,
        "boundary_repeat_spread": None,
        "boundary_failure_classes": [],
        "boundary_failure_modes": [],
    }


def orchestrate(
    *,
    allow_dirty: bool,
    resume_session_id: str | None = None,
    smoke: bool = False,
    skip_pre_run_settle: bool = False,
    arms: tuple[str, ...] | None = None,
    model_spec: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    resolved: ResolvedConfig = resolve_config(
        [root / _PLATFORM_PATH, root / _MEASUREMENT_PATH, root / _DELTA_N_PATH],
        repo_root=root,
    )
    cfg = resolved.data

    if not cfg["topology"]["verified"]:
        raise SeamError("topology is not verified; a cpu-p ceiling would be uninterpretable")
    p_cpus = [int(c) for c in cfg["topology"]["p_cpus"]]

    isolation_block = cfg.get("isolation") or {}
    if "pre_run_settle_s" not in isolation_block:
        raise SeamError("configs/delta_n.yaml must declare isolation.pre_run_settle_s")
    pre_run_settle_s = float(isolation_block["pre_run_settle_s"])

    if smoke:
        cfg = dict(cfg)
        cfg["ladder"] = {
            **cfg["ladder"],
            "rungs": [256],
            "repeats": 1,
            "bisect_resolution_tokens": 4096,
        }
        cfg["recovery"] = {**cfg["recovery"], "settle_s": 2, "max_wait_s": 30}
        cfg["isolation"] = {**cfg["isolation"], "pre_run_settle_s": 0}
        pre_run_settle_s = 0.0
        skip_pre_run_settle = True
        # Smoke is harness validation only - never a sealed remote measurement.
        launch_context, placement = resolve_launch_context(required=False)
        if launch_context is None:
            launch_context = "local_console"
            placement = {}
        isolation_mode = "local"
    else:
        launch_context, placement = resolve_launch_context(required=True)
        isolation_mode, _evidence = resolve_isolation_mode()
        if launch_context != "ssh_detached":
            raise SeamError(
                f"ceiling_a requires launch_context=ssh_detached, got {launch_context!r}. "
                "Do not run the hours-long ladder under Cursor and label it remote."
            )
        if isolation_mode != "remote":
            raise SeamError(f"ceiling_a requires isolation_mode=remote, got {isolation_mode!r}")

    model_spec_path = (
        Path(model_spec) if model_spec is not None else Path(cfg["openvino"]["model_spec"])
    )
    if not model_spec_path.is_absolute():
        model_spec_path = root / model_spec_path
    spec = load_local_spec(model_spec_path)
    model_dir = str(spec["ir_dir"])
    if not Path(model_dir).is_dir():
        raise SeamError(f"model IR directory missing: {model_dir}")

    arms_by_id = {arm["id"]: arm for arm in cfg["arms"]}
    all_arm_ids = [arm["id"] for arm in cfg["arms"]]
    requested_arms = tuple(arms) if arms is not None else DEFAULT_ARMS

    termination = harness_termination_record()

    # Hold PowerRequestSystemRequired for the WHOLE ladder (hours). Refuse if assertion fails.
    orch_power = assert_system_required(
        reason=(
            "SEAM ceiling_a orchestrator; PowerRequestSystemRequired across "
            "pre_run_settle, interleaved ladder, and bisection"
        ),
        role="ceiling_a_orchestrator",
    )
    _log(
        "ceiling_a.orchestrate_power_request",
        succeeded=orch_power.record.get("succeeded"),
        mechanism=orch_power.record.get("mechanism"),
        error=orch_power.record.get("error"),
    )
    if not orch_power.record.get("succeeded"):
        orch_power.release()
        raise SeamError(
            "PowerRequestSystemRequired failed to assert; refusing to start ceiling_a. "
            f"record={orch_power.record}"
        )

    try:
        if resume_session_id:
            session_id = resume_session_id
            session_dir = _session_dir(root, session_id)
            state = load_checkpoint(session_dir)
            if state is None:
                raise SeamError(f"no checkpoint at {session_dir}")
            if state.get("phase") == "sealed":
                raise SeamError(f"session {session_id} is already sealed; not resuming")
            ckpt_arms = tuple(state.get("arms_selected") or DEFAULT_ARMS)
            if arms is not None and requested_arms != ckpt_arms:
                raise SeamError(
                    f"--arms {list(requested_arms)!r} does not match checkpoint "
                    f"arms_selected {list(ckpt_arms)!r}"
                )
            active_arms = ckpt_arms
            for arm_id in active_arms:
                if arm_id not in arms_by_id:
                    raise SeamError(f"configs/delta_n.yaml missing arm {arm_id}")
            state["arms_selected"] = list(active_arms)
            state.setdefault("resumes", []).append(
                {
                    "resumed_utc": _utc(),
                    "from_phase": state.get("phase"),
                    "last_cell_key": state.get("last_cell_key"),
                    "last_completed_rung": state.get("last_completed_rung"),
                    "arms_selected": list(active_arms),
                }
            )
            save_checkpoint(session_dir, state)
            _log(
                "ceiling_a.resume",
                session_id=session_id,
                from_phase=state.get("phase"),
                n_cells=len(state.get("completed_cells") or []),
                arms_selected=list(active_arms),
            )
            pre_run_settle = {
                "configured_s": pre_run_settle_s,
                "actual_wait_s": 0.0,
                "applied_on_this_path": False,
                "skipped_on_resume": True,
                "note": "pre_run_settle already applied on the original launch; not re-waited",
            }
            # Re-open existing in-progress run dirs (create_run_dir refuses reuse).
            run_dirs = {}
            for arm_id in active_arms:
                run_id = state["arm_run_ids"][arm_id]
                run_dir = open_run_dir(run_id, repo_root=root)
                if run_dir.is_sealed():
                    raise SeamError(
                        f"arm {arm_id} run {run_id} is already sealed; cannot resume into it"
                    )
                run_dir.write_json(
                    "resume_marker.json",
                    {
                        "experiment_id": "ceiling_a",
                        "arm_id": arm_id,
                        "session_id": session_id,
                        "arms_selected": list(active_arms),
                        "resumed_utc": _utc(),
                        "n_resumes": len(state.get("resumes") or []),
                    },
                )
                run_dirs[arm_id] = run_dir
            work_dir = session_dir / "work"
        else:
            active_arms = requested_arms
            for arm_id in active_arms:
                if arm_id not in arms_by_id:
                    raise SeamError(f"configs/delta_n.yaml missing arm {arm_id}")
            session_id = str(uuid.uuid4())
            session_dir = _session_dir(root, session_id)
            session_dir.mkdir(parents=True, exist_ok=True)
            work_dir = session_dir / "work"
            work_dir.mkdir(parents=True, exist_ok=True)

            arm_run_ids = {arm_id: str(uuid.uuid4()) for arm_id in active_arms}
            verdict_run_id = str(uuid.uuid4())
            state = {
                "session_id": session_id,
                "phase": "starting",
                "experiment_id": "ceiling_a",
                "prerequisite_acceptance_run_id": PREREQUISITE_ACCEPTANCE_RUN_ID,
                "arms_selected": list(active_arms),
                "arm_run_ids": arm_run_ids,
                "verdict_run_id": verdict_run_id,
                "randomization_seed": int(cfg["randomization_seed"]),
                "smoke": smoke,
                "arm_state": {arm_id: _new_arm_state() for arm_id in active_arms},
                "completed_cells": [],
                "cell_payloads": {},
                "rung_records": {},
                "realized_order_log": [],
                "bisect_steps": [],
                "resumes": [],
                "early_exit_fired": False,
                "started_utc": _utc(),
            }
            save_checkpoint(session_dir, state)

            if skip_pre_run_settle or pre_run_settle_s <= 0:
                pre_run_settle = {
                    "configured_s": pre_run_settle_s,
                    "actual_wait_s": 0.0,
                    "applied_on_this_path": False,
                    "skipped": True,
                }
            else:
                pre_run_settle = _wait_pre_run_settle(pre_run_settle_s)

            run_dirs = {}
            for arm_id in active_arms:
                run_dirs[arm_id] = open_in_progress_run(
                    root=root,
                    run_id=arm_run_ids[arm_id],
                    marker={
                        "experiment_id": "ceiling_a",
                        "arm_id": arm_id,
                        "session_id": session_id,
                        "arms_selected": list(active_arms),
                        "started_utc": _utc(),
                        "prerequisite_acceptance_run_id": PREREQUISITE_ACCEPTANCE_RUN_ID,
                    },
                )

        isolation_discipline = {
            "pre_run_settle": pre_run_settle,
            "interactive_process_termination": termination,
            "power_request": dict(orch_power.record),
            "isolation_mode": isolation_mode,
            "launch_context": launch_context,
        }
        state["isolation_discipline"] = isolation_discipline
        state["arms_selected"] = list(active_arms)
        save_checkpoint(session_dir, state)

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        prompt_cache: dict[int, dict[str, Any]] = {}

        _log(
            "ceiling_a.start",
            session_id=session_id,
            arms_selected=list(active_arms),
            arm_run_ids=state["arm_run_ids"],
            verdict_run_id=state["verdict_run_id"],
            smoke=smoke,
            prerequisite=PREREQUISITE_ACCEPTANCE_RUN_ID,
            **memory_now(),
        )

        run_interleaved_ladder(
            root=root,
            cfg=cfg,
            p_cpus=p_cpus,
            session_dir=session_dir,
            state=state,
            arms_by_id=arms_by_id,
            model_dir=model_dir,
            tokenizer=tokenizer,
            prompt_cache=prompt_cache,
            run_dirs=run_dirs,
            work_dir=work_dir,
            active_arms=active_arms,
        )

        arm_summaries = {
            arm_id: _arm_summary_from_state(
                arm_id=arm_id,
                label=arms_by_id[arm_id]["label"],
                arm_cfg=arms_by_id[arm_id],
                state=state,
                prompt_cache=prompt_cache,
            )
            for arm_id in active_arms
        }
        verdict = build_ceiling_verdict(arm_summaries, cfg, state, all_arm_ids=all_arm_ids)

        if smoke:
            state["phase"] = "smoke_complete"
            state["smoke_arm_summaries"] = arm_summaries
            state["smoke_verdict"] = verdict
            save_checkpoint(session_dir, state)
            _log("ceiling_a.smoke_complete", session_id=session_id, note="not sealed")
            return {
                "session_id": session_id,
                "sealed": False,
                "smoke": True,
                "arms_selected": list(active_arms),
                "arm_run_ids": state["arm_run_ids"],
                "verdict_run_id": state["verdict_run_id"],
                "verdict": verdict,
                "heartbeat_path": str(session_dir / "heartbeat.json"),
                "checkpoint_path": str(_checkpoint_path(session_dir)),
            }

        state["phase"] = "sealing"
        save_checkpoint(session_dir, state)

        sealed_arm_ids: dict[str, str] = {}
        for arm_id in active_arms:
            sealed_arm_ids[arm_id] = _seal_arm(
                root=root,
                resolved=resolved,
                cfg=cfg,
                spec=spec,
                model_spec_path=model_spec_path,
                arm_id=arm_id,
                arm_summary=arm_summaries[arm_id],
                run_dir=run_dirs[arm_id],
                run_id=state["arm_run_ids"][arm_id],
                state=state,
                allow_dirty=allow_dirty,
                launch_context=launch_context,
                placement=placement,
                power_request=dict(orch_power.record),
                isolation_discipline=isolation_discipline,
            )
            _log("ceiling_a.arm_sealed", arm=arm_id, run_id=sealed_arm_ids[arm_id])

        verdict_run_id = _seal_verdict(
            root=root,
            resolved=resolved,
            cfg=cfg,
            arm_run_ids=sealed_arm_ids,
            arm_summaries=arm_summaries,
            verdict=verdict,
            run_id=state["verdict_run_id"],
            state=state,
            allow_dirty=allow_dirty,
            launch_context=launch_context,
            placement=placement,
            power_request=dict(orch_power.record),
            isolation_discipline=isolation_discipline,
        )

        result = {
            "session_id": session_id,
            "sealed": True,
            "smoke": False,
            "arms_selected": list(active_arms),
            "arm_run_ids": sealed_arm_ids,
            "verdict_run_id": verdict_run_id,
            "ceiling_tokens": verdict["ceiling_tokens"],
            "aa_spread_tokens": verdict["aa_spread_tokens"],
            "early_exit_fired": verdict["early_exit_fired"],
            "verdict": verdict["verdict"],
            "next_phase": verdict["next_phase"],
            "heartbeat_path": str(session_dir / "heartbeat.json"),
            "checkpoint_path": str(_checkpoint_path(session_dir)),
            "prerequisite_acceptance_run_id": PREREQUISITE_ACCEPTANCE_RUN_ID,
        }
        state["phase"] = "sealed"
        state["result"] = result
        save_checkpoint(session_dir, state)
        (session_dir / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        _log("ceiling_a.done", **{k: v for k, v in result.items() if k != "verdict"})
        return result
    finally:
        orch_power.release()


def main(argv: list[str] | None = None) -> int:
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument(
        "--orchestrate",
        action="store_true",
        help="remote/ssh_detached ceiling_a: settle, interleaved ladder for --arms, seal",
    )
    parser.add_argument(
        "--resume",
        metavar="SESSION_ID",
        help="resume from derived/ceiling_a/<session_id>/checkpoint.json",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="harness validation: one tiny rung, one repeat, no seal; may run under local",
    )
    parser.add_argument(
        "--arms",
        default=DEFAULT_ARMS_CSV,
        help=(f"comma-separated arm ids from configs/delta_n.yaml (default: {DEFAULT_ARMS_CSV})"),
    )
    parser.add_argument(
        "--model-spec",
        type=Path,
        default=None,
        help=(
            "FetchedModelSpec YAML. Default: openvino.model_spec in configs/delta_n.yaml "
            "(Qwen3-4B-int4-ov). Pass configs/models/Qwen3-8B-int4-ov.yaml to select 8B."
        ),
    )
    parser.add_argument(
        "--result-json",
        type=Path,
        help="write the compact result object here when finished",
    )
    args = parser.parse_args(argv)

    if not args.orchestrate and not args.resume and not args.smoke:
        raise SystemExit(
            "ceiling_a: pass --orchestrate (detached remote), --resume SESSION_ID, or --smoke"
        )

    arms_selected = parse_arms(args.arms)
    result = orchestrate(
        allow_dirty=args.allow_dirty or args.smoke,
        resume_session_id=args.resume,
        smoke=args.smoke,
        arms=arms_selected,
        model_spec=args.model_spec,
    )
    if args.result_json:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
